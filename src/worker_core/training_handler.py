import os
import time
import joblib
import pandas as pd
from src.model.model_factory import ModelFactory


# Handles the execution of partial training tasks on worker nodes
class TrainingHandler:

    # Initializes the handler with the AWS manager instance
    def __init__(self, aws_manager, config):
        self.aws = aws_manager

    # Processes the assigned training chunk and uploads the resulting partial model to S3
    def process(self, task_data):
        job_id = task_data['job_id']
        task_id = task_data['task_id']
        dataset_uri = task_data['dataset_s3_path']

        print(f" [TRAIN] Starting {task_id}. Downloading {task_data['num_rows']} rows...")
        print(f" [TRAIN] Dataset source: {dataset_uri}")

        # 1. Perform zero-waste RAM partial reading of the dataset from S3
        skip = task_data.get('skip_rows', 0)
        df = pd.read_csv(dataset_uri, skiprows=range(1, skip + 1) if skip > 0 else None, nrows=task_data['num_rows'])

        # Apply a global fallback for missing values to prevent training crashes
        df = df.fillna(0)

        start_time = time.time()

        # 2. Execute training logic dynamically via ModelFactory
        target_col = task_data.get("custom_target_col", "Label")
        task_type = task_data.get("task_type", "classification")

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in the dataset chunk!")

        # The ModelFactory now acts as our single source of truth for handling different ML tasks
        ml_handler = ModelFactory.get_model(task_type=task_type, target_column=target_col)

        """
        # ==========================================================
        # TEST 1.1 (HARD CRASH WORKER)
        # ==========================================================
        print("\n" + "!"*50)
        print(f" [TEST 1.1] PREPARING TO TRAIN TASK {task_data['task_id']}")
        print(" [TEST 1.1] You have 15 seconds to KILL THIS WORKER!")
        print(" [TEST 1.1] SQS Heartbeat is running. If killed, it drops.")
        print("!"*50 + "\n")
        time.sleep(15)
        # ==========================================================
         """

        """
        # ==========================================================
        # TEST 1.2 (SOFT CRASH WORKER)
        # ==========================================================
        if task_data['task_id'] == "task_1":
            print("\n" + "!"*50)
            print(" [TEST 1.2] SIMULATING OUT-OF-MEMORY / PYTHON EXCEPTION")
            print("!"*50 + "\n")
            raise MemoryError("Simulated Soft Crash: RAM Exhausted!")
        # ==========================================================
        """

        rf = ml_handler.process_and_train(df, task_data)

        print(f" [Job: {job_id} | Task: {task_id}] Training completed in {time.time() - start_time:.2f}s")

        # 3. Serialize the model locally, upload to S3, and clean up safely
        local_path = f"/tmp/{task_id}_{job_id}.joblib"
        bucket, _ = self.aws.parse_s3_uri(dataset_uri)

        # Flat S3 Structure: models/<job_id>/task_X.joblib
        s3_key = f"models/{job_id}/{task_id}.joblib"

        try:
            joblib.dump(rf, local_path)
            print(f" [TRAIN] Uploading model chunk to S3 (s3://{bucket}/{s3_key})...")
            self.aws.s3_client.upload_file(local_path, bucket, s3_key)
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)

        return f"s3://{bucket}/{s3_key}"