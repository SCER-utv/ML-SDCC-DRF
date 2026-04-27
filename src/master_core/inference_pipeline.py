import time
import json
import math


# Orchestrates both bulk (test set evaluation) and real-time inference workflows
class InferencePipeline:

    def __init__(self, aws_manager, evaluation_manager):
        self.aws = aws_manager
        self.evaluator = evaluation_manager

    # Executes inference over an entire dataset across distributed workers
    def run_bulk(self, job_data, job_id):
        target_model = job_data['target_model']
        test_url = job_data.get('test_url', '')
        task_type = job_data.get('task_type', 'classification')
        target_col = job_data.get('target_column', 'Label')
        total_start_time = job_data.get('client_start_time', time.time())

        if not test_url:
            raise ValueError("Test URL is missing. Cannot perform bulk inference.")

        try:
            bucket, test_key = self.aws.parse_s3_uri(test_url)
        except Exception:
            raise ValueError(f"Invalid S3 URL format for test target: {test_url}")


        if not self.aws.check_s3_file_exists(bucket, test_key):
            raise ValueError(f"CRITICAL: The test file {test_url} DOES NOT exist on S3.")


        # 1. Check model chunks and provision instances
        model_s3_uris = self.aws.count_model_parts(self.aws.bucket, target_model)
        num_workers = len(model_s3_uris)

        print(f" [BULK-INFER] Target model '{target_model}' chunked into {num_workers} parts. Scaling workers...")
        self.aws.scale_worker_infrastructure(num_workers)

        # 2. Fault tolerance state recovery
        db_start, historical_train_time, s3_inference_results, start_infer, original_train_url = self._recover_bulk_state(job_id, target_model)

        if db_start:
            total_start_time = db_start

        """
        # ==========================================================
        # TEST 2.3 (MASTER CRASH PRE-FANOUT INFERENCE)
        # ==========================================================
        print("\n" + "!"*50)
        print(" [TEST 2.3] ABOUT TO DISPATCH BULK INFERENCE TASKS.")
        print(" [TEST 2.3] You have 15 seconds to kill the Master!")
        print(" [TEST 2.3] DynamoDB state is recovered, SQS is empty.")
        print("!"*50 + "\n")
        time.sleep(15)
        # ==========================================================
        """

        # 3. Fan-out task generation
        self._dispatch_bulk_tasks(job_id, test_url, model_s3_uris, s3_inference_results, task_type, target_col)
        current_inference_duration = time.time() - start_infer
        self.aws.update_job_state(job_id, set(), s3_inference_results, total_start_time, True, training_time = historical_train_time, inference_time = current_inference_duration, train_url = original_train_url)

        """
        # ==========================================================
        # TEST 4.1b (MASTER NETWORK PARTITION - BULK INFERENCE)
        # ==========================================================
        print("\n" + "!"*50)
        print(" [TEST 4.1b] NETWORK PARTITION: THE MASTER IS ISOLATED!")
        print(" [TEST 4.1b] The Master is going offline for 45 seconds (Sleep).")
        print(" [TEST 4.1b] Workers will continue evaluating chunks and filling SQS.")
        print("!"*50 + "\n")
        time.sleep(45)
        # ==========================================================
        """

        # 4. Wait for workers to process chunks
        self._wait_for_bulk_workers(job_id, num_workers, s3_inference_results, total_start_time,
                                                     historical_train_time, start_infer, original_train_url)

        """
        # ==========================================================
        # TEST 2.6 (MASTER CRASH BEFORE FINAL AGGREGATION)
        # ==========================================================
        print("\n" + "!"*50)
        print(" [TEST 2.6] ALL ACKS RECEIVED. WORKERS ARE IDLE.")
        print(" [TEST 2.6] You have 15 seconds to kill the Master!")
        print(" [TEST 2.6] Upon restart, it should JUMP DIRECTLY to Evaluation!")
        print("!"*50 + "\n")
        time.sleep(15)
        # ==========================================================
        """

        # 5. Delegate final evaluation and metric calculation
        num_trees, weights, strat = self._calculate_inference_weights(target_model, num_workers)

        # We pass the flattened data to the evaluator
        final_metrics = self.evaluator.aggregate_and_evaluate(
            job_data, job_id, s3_inference_results, num_workers,
            num_trees, weights, historical_train_time, start_infer, strat
        )

        if final_metrics:
            self.aws.update_job_state(
                job_id,
                set(),
                s3_inference_results,
                total_start_time,
                True,
                training_time=historical_train_time,
                inference_time=final_metrics['infer_time_sec'],
                train_url=original_train_url
            )

        # 6. Notify the client
        self._send_client_response(job_id, job_data.get('mode', 'bulk_infer'), time.time() - total_start_time, metrics=final_metrics)

    # Processes a single data tuple for low-latency prediction
    def run_realtime(self, job_data, job_id):
        target_model = job_data['target_model']
        tuple_data = job_data['tuple_data']
        task_type = job_data.get('task_type')
        total_start_time = job_data.get('client_start_time', time.time())

        # 1. Check chunks and provision cluster
        model_s3_uris = self.aws.count_model_parts(self.aws.bucket, target_model)
        num_workers = len(model_s3_uris)

        print(f" [REAL-TIME] Target model '{target_model}' chunked into {num_workers} parts. Scaling workers...")
        start_provisioning = time.time()
        self.aws.scale_worker_infrastructure(num_workers)
        provisioning_time = time.time() - start_provisioning

        # 2. Dispatch the tuple to all workers
        inference_pure_start = time.time()
        self._dispatch_realtime_tasks(job_id, model_s3_uris, tuple_data, task_type)

        """
        # ==========================================================
        # TEST 4.1c (MASTER NETWORK PARTITION - REAL-TIME INFERENCE)
        # ==========================================================
        print("\n" + "!"*50)
        print(" [TEST 4.1c] NETWORK PARTITION: THE MASTER IS ISOLATED!")
        print(" [TEST 4.1c] The Master goes offline for 15 seconds (Sleep).")
        print(" [TEST 4.1c] SQS acts as a buffer for the fast Real-Time replies.")
        print("!"*50 + "\n")
        time.sleep(15) 
        # ==========================================================
        """

        """
        # ==========================================================
        # TEST 3.3 (MASTER CRASH IN INFERENCE REAL-TIME)
        # ==========================================================
        print("\n" + "!"*50)
        print(" [TEST 3.3] TASKS SENT TO WORKERS. WAITING FOR RESPONSES...")
        print(" [TEST 3.3] You have 15 seconds to restart the Master!")
        print("!"*50 + "\n")
        time.sleep(15)
        # ==========================================================
        """

        # 3. Rapidly gather votes from memory
        total_received_votes, pure_inference_time = self._wait_for_realtime_workers(num_workers, inference_pure_start)

        """
        # ==========================================================
        # TEST 3.4 (MASTER CRASH DURING REAL-TIME AGGREGATION)
        # ==========================================================
        print("\n" + "!"*50)
        print(" [TEST 3.4] ALL VOTES GATHERED IN RAM. READY TO AGGREGATE.")
        print(" [TEST 3.4] You have 15 seconds to kill the Master!")
        print(" [TEST 3.4] RAM will be wiped. Master MUST restart from scratch.")
        print("!"*50 + "\n")
        time.sleep(15)
        # ==========================================================
        """

        # 4. Aggregate votes via majority or mean
        final_prediction, task_str = self._aggregate_realtime_results(task_type, total_received_votes)

        # 5. Display metrics and notify client
        total_run_time = time.time() - total_start_time
        print(f"\n{'=' * 60}\n REAL-TIME PREDICTION ({task_str}): {final_prediction:.2f}\n{'-' * 60}")
        print(f" AWS Provisioning Time (Cold Start):   {provisioning_time:.2f}s")
        print(f" Pure Inference Time (SQS + CPU):      {pure_inference_time:.2f}s")
        print(f" TOTAL Global System Latency:          {total_run_time:.2f}s\n{'=' * 60}\n")

        self._send_client_response(job_id, "infer", total_run_time, prediction=float(final_prediction),
                                   task_str=task_str)

    # --- Support Methods ---

    # Retrieves previous training duration and current inference state from DynamoDB
    def _recover_bulk_state(self, job_id, target_model):
        _, _, _, _, historical_train_time, _, original_train_url = self.aws.get_job_state(target_model)
        _, s3_inference_results, db_total_start, _, _, db_infer_time, _ = self.aws.get_job_state(job_id)
        start_infer = time.time() - db_infer_time
        return db_total_start, historical_train_time, s3_inference_results, start_infer, original_train_url

    # Queues inference payloads for each worker
    def _dispatch_bulk_tasks(self, job_id, test_url, model_s3_uris, s3_inference_results, task_type, target_col):
        infer_queue = self.aws.sqs_queues["infer_task"]

        for i, uri in enumerate(model_s3_uris):
            task_id = f"task_{i + 1}"
            if task_id not in s3_inference_results:
                infer_task = {
                    "job_id": job_id,
                    "task_id": task_id,
                    "test_dataset_uri": test_url,
                    "model_s3_uri": uri,
                    "task_type": task_type,
                    "target_column": target_col,
                }
                self.aws.sqs_client.send_message(QueueUrl=infer_queue, MessageBody=json.dumps(infer_task))
                print(f" [INFER DISPATCH] Task {task_id} sent to inference queue.")
                
                """
                # ==========================================================
                # TEST 2.4 (MASTER CRASH MID-FANOUT INFERENCE)
                # ==========================================================
                if i == len(model_s3_uris) // 2:
                    print("\n" + "!"*50)
                    print(f" [TEST 2.4] MID-FANOUT: {i+1} / {len(model_s3_uris)} TASKS SENT.")
                    print(" [TEST 2.4] Kill the Master NOW to test S3 .npy IDEMPOTENCY!")
                    print("!"*50 + "\n")
                    time.sleep(15)
                # ==========================================================
                """

    # Polls the SQS response queue until all workers complete their chunk evaluation
    def _wait_for_bulk_workers(self, job_id, num_workers, s3_inference_results, total_start_time, historical_train_time, start_infer, original_train_url):
        infer_resp_queue = self.aws.sqs_queues["infer_response"]
        print("\n [EVENT LOOP] Master listening actively for Worker inference responses...\n")

        """
        # ==========================================================
        # TEST 2.5a (MASTER CRASH POST-FANOUT INFERENCE)
        # ==========================================================
        print("\n" + "!"*50)
        print(" [TEST 2.5a] ALL INFERENCE TASKS SENT. WAITING FOR ACKS.")
        print(" [TEST 2.5a] You have 15 seconds to kill the Master!")
        print("!"*50 + "\n")
        time.sleep(15)
        # ==========================================================
        """
    
        while len(s3_inference_results) < num_workers:
            res_infer = self.aws.sqs_client.receive_message(QueueUrl=infer_resp_queue, MaxNumberOfMessages=10,
                                                            WaitTimeSeconds=2)
            if 'Messages' in res_infer:
                for msg in res_infer['Messages']:
                    body = json.loads(msg['Body'])

                    msg_job_id = body.get('job_id')
                    if msg_job_id != job_id:
                        print(f" [CLEANUP] Removed zombie message from job: {msg_job_id}")
                        self.aws.sqs_client.delete_message(
                            QueueUrl=infer_resp_queue,
                            ReceiptHandle=msg['ReceiptHandle']
                        )
                        continue


                    task_id = body['task_id']

                    # Safe URI extraction for backward compatibility
                    s3_votes_uri = body['s3_voti_uri']['valore'] if isinstance(body['s3_voti_uri'], dict) else body[
                        's3_voti_uri']

                    if task_id not in s3_inference_results:
                        s3_inference_results[task_id] = s3_votes_uri
                        print(
                            f" [ACK] Worker completed Bulk Inference for {task_id}! ({len(s3_inference_results)}/{num_workers})")

                        self.aws.update_job_state(
                            job_id, set(), s3_inference_results, total_start_time,
                            True, training_time=historical_train_time,inference_time=time.time() - start_infer, train_url=original_train_url
                        )

                        """
                        # ==========================================================
                        # TEST 2.5b (MASTER CRASH MID-ACK RECEPTION INFERENCE)
                        # ==========================================================
                        if len(s3_inference_results) == num_workers // 2:
                            print("\n" + "!"*50)
                            print(f" [TEST 2.5b] RECEIVED HALF ACKs ({len(s3_inference_results)}/{num_workers}).")
                            print(" [TEST 2.5b] State saved to DynamoDB. Kill the Master NOW!")
                            print(" [TEST 2.5b] Upon restart, it should ONLY wait for the remaining ones.")
                            print("!"*50 + "\n")
                            time.sleep(15)
                        # ==========================================================
                        """

                    self.aws.sqs_client.delete_message(QueueUrl=infer_resp_queue, ReceiptHandle=msg['ReceiptHandle'])
                    if len(s3_inference_results) == num_workers:
                        break

        return time.time() - start_infer

    # Parses the target model ID to extract the exact number of trees for weight calculation
    def _calculate_inference_weights(self, target_model, num_workers):
        parts = target_model.split('_')
        trees_part = next((p for p in parts if 'trees' in p), None)

        if trees_part:
            try:
                num_trees = int(trees_part.replace('trees', ''))
            except ValueError:
                print(f" [WARNING] Invalid tree format in '{trees_part}'. Using fallback.")
                num_trees = num_workers * 25
        else:
            print(" [WARNING] 'trees' word not found in model ID. Using fallback.")
            num_trees = num_workers * 25

        strat = "heterogeneous" if "heterogeneous" in target_model else "homogeneous"

        # Distributes remaining trees (modulo) across the first workers
        weights = [
            math.floor(num_trees / num_workers) + (1 if i < (num_trees % num_workers) else 0)
            for i in range(num_workers)
        ]

        print(f" [RESOLVER] Model: {strat.upper()} | Total trees: {num_trees} | Weights: {weights}")

        return num_trees, weights, strat

    # Broadcasts a single tuple payload to all workers
    def _dispatch_realtime_tasks(self, job_id, model_s3_uris, tuple_data, task_type):
        infer_task_queue = self.aws.sqs_queues["infer_task"]
        for i, uri in enumerate(model_s3_uris):
            task = {
                "job_id": job_id,
                "task_id": f"task_infer_rt_{i + 1}",
                "model_s3_uri": uri,
                "tuple_data": tuple_data,
                "task_type": task_type
            }
            self.aws.sqs_client.send_message(QueueUrl=infer_task_queue, MessageBody=json.dumps(task))

    # Rapidly collects in-memory votes from workers for real-time predictions
    def _wait_for_realtime_workers(self, num_workers, start_time):
        total_received_votes = []
        read_messages = 0
        infer_resp_queue = self.aws.sqs_queues["infer_response"]

        while read_messages < num_workers:
            res = self.aws.sqs_client.receive_message(QueueUrl=infer_resp_queue, WaitTimeSeconds=2)
            if 'Messages' in res:
                for msg in res['Messages']:
                    body = json.loads(msg['Body'])
                    res_data = body['s3_voti_uri']

                    if isinstance(res_data, dict) and res_data.get("tipo") == "singolo":
                        worker_predictions = res_data['valore']
                        total_received_votes.extend(worker_predictions)
                        read_messages += 1
                        print(f"   -> Gathered {len(worker_predictions)} votes from worker.")

                    self.aws.sqs_client.delete_message(QueueUrl=infer_resp_queue, ReceiptHandle=msg['ReceiptHandle'])

        return total_received_votes, time.time() - start_time

    # Computes the final real-time prediction using voting or averaging
    def _aggregate_realtime_results(self, task_type, total_received_votes):
        if task_type == 'classification':
            final_prediction = max(set(total_received_votes), key=total_received_votes.count)
            votes_0 = total_received_votes.count(0)
            votes_1 = total_received_votes.count(1)
            task_str = "Classification (Majority Vote)"
            print(f" [POLL] Class 0: {votes_0} votes | Class 1: {votes_1} votes")
        else:
            final_prediction = sum(total_received_votes) / len(total_received_votes)
            task_str = "Regression (Mean)"

        return final_prediction, task_str

    # Returns completion status and predictions to the client application via SQS
    def _send_client_response(self, job_id, mode, total_time, prediction=None, task_str=None, metrics=None):
        client_response_queue = self.aws.sqs_queues["client_response"]
        if client_response_queue:
            payload = {
                "job_id": job_id,
                "mode": mode,
                "total_time_sec": round(total_time, 2),
                "status": "SUCCESS"
            }
            if prediction is not None:
                payload["prediction"] = prediction
                payload["task_type"] = task_str

            if metrics:
                payload["metrics"] = metrics

            try:
                self.aws.sqs_client.send_message(QueueUrl=client_response_queue, MessageBody=json.dumps(payload))
                if prediction is not None:
                    print(f" [SUCCESS] Real-Time Prediction sent back to Client via SQS.")
            except Exception as e:
                print(f" [ERROR] Failed to send response to client: {e}")