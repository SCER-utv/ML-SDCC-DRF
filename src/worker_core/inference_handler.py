import os
import time
import gc
import joblib
import numpy as np
import pandas as pd
from src.model.model_factory import ModelFactory


# Handles model inference execution including real-time single predictions and memory-efficient bulk evaluations
class InferenceHandler:

    # Initializes the inference handler with AWS manager and configured chunk size
    def __init__(self, aws_manager, config):
        self.aws = aws_manager
        self.chunk_size = config.get("inference_chunksize", 500000)

    # Processes the inference task by downloading the model and routing to either real-time or bulk chunked prediction
    def process(self, task_data):
        job_id = task_data['job_id']
        task_id = task_data['task_id']

        try:
            bucket, model_key = self.aws.parse_s3_uri(task_data['model_s3_uri'])
        except Exception as e:
            raise ValueError(f"Invalid model URI provided: {task_data.get('model_s3_uri')} - {e}")

        test_uri = task_data.get('test_dataset_uri', 'N/A (Real-time inference)')
        print(f" [INFER] Source dataset: {test_uri}")

        # Download the assigned partial model from S3 into temporary storage
        local_model_path = f"/tmp/model_{job_id}_{task_id}.joblib"
        try:
            self.aws.s3_client.download_file(bucket, model_key, local_model_path)
            rf = joblib.load(local_model_path)
        finally:
            if os.path.exists(local_model_path):
                os.remove(local_model_path)

        # Retrieve generic ML metadata directly from the clean payload
        task_type = task_data.get('task_type', 'classification')
        target_col = task_data.get('target_column', 'Label')

        # Instantiate the correct Model Handler dynamically based on the ML task
        ml_handler = ModelFactory.get_model(task_type=task_type, target_column=target_col)

        # CASE A: Real-time inference for a single data tuple
        if 'tuple_data' in task_data:
            print(f" [INFER] Single tuple real-time prediction in progress...")

            data_array = np.array(task_data['tuple_data']).reshape(1, -1)

            # Extract predictions from each tree in the loaded chunk
            all_pred = [float(tree.predict(data_array)[0]) for tree in rf.estimators_]
            return {"tipo": "singolo", "valore": all_pred}

        # CASE B: Bulk inference processing the full test dataset in memory-efficient chunks
        print(f" [INFER] Bulk inference started (Chunksize: {self.chunk_size})")
        start_time = time.time()
        all_predictions = []

        # Process predictions chunk by chunk and aggressively free RAM
        for chunk in pd.read_csv(task_data['test_dataset_uri'], chunksize=self.chunk_size, low_memory=False):
            # Delegate entirely to the robust ML Handler (handles NaN, inf, column drops, and specific vote aggregation)
            chunk_results = ml_handler.process_and_predict(rf, chunk)

            all_predictions.append(chunk_results)
            del chunk, chunk_results
            gc.collect()

        if not all_predictions:
            raise ValueError(f"No predictions generated. Check if the test file is empty: {test_uri}")

        numpy_results = np.concatenate(all_predictions)
        print(f" [INFER] Generated {len(numpy_results)} predictions in {time.time() - start_time:.2f}s")

        # Compress, save locally as a numpy array, and upload back to S3
        local_npy_path = f"/tmp/results_{job_id}_{task_id}.npy"
        s3_key = f"results/{job_id}/{task_id}.npy"

        try:
            np.save(local_npy_path, numpy_results)
            self.aws.s3_client.upload_file(local_npy_path, bucket, s3_key)
        finally:
            if os.path.exists(local_npy_path):
                os.remove(local_npy_path)

        return {"tipo": "bulk", "valore": f"s3://{bucket}/{s3_key}"}