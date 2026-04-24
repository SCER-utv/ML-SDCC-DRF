import time
import json
import math
from src.utils.config import load_config


# Orchestrates the distributed training workflow
class TrainingPipeline:

    def __init__(self, aws_manager):
        self.aws = aws_manager
        self.config = load_config()

    # Executes the training process across distributed workers
    def run(self, job_data, job_id):
        num_workers = job_data['num_workers']
        train_url = job_data.get('train_url', '')

        # 1. Fault tolerance state recovery
        completed_train_tasks, s3_inference_results, start_train, tasks_dispatched, training_time, final_train_url = self._recover_or_initialize_state(
            job_id, train_url)

        # 2. Infrastructure provisioning
        self.aws.scale_worker_infrastructure(num_workers)
        time.sleep(10)

        # 3. Dataset validation and row counting
        calculated_train_rows = None
        if not tasks_dispatched:
            calculated_train_rows = self._ensure_dataset_ready(train_url)
        
        """
        # ==========================================================
        # TEST 1.3 (MASTER CRASH PRE-FANOUT)
        # ==========================================================
        print("\n" + "!"*50)
        print(" [TEST 1.3] DATA READY, ABOUT TO SEND TASKS TO SQS.")
        print(" [TEST 1.3] You have 15 seconds to kill the Master!")
        print(" [TEST 1.3] DynamoDB 'tasks_dispatched' is still FALSE.")
        print("!"*50 + "\n")
        time.sleep(15)
        # ==========================================================
        """

        # 4. Fan-out task generation
        if not tasks_dispatched:
            self._generate_tasks(job_data, job_id, calculated_train_rows)
            tasks_dispatched = True
            self.aws.update_job_state(job_id, completed_train_tasks, s3_inference_results, start_train,
                                      tasks_dispatched, training_time, 0.0, final_train_url)
        else:
            print(" [RECOVERY] SQS Fan-Out skipped to prevent duplicates.")

        """
        # ==========================================================
        # TEST 4.1 (MASTER NETWORK PARTITION SIMULATION)
        # ==========================================================
        print("\n" + "!"*50)
        print(" [TEST 4.1] NETWORK PARTITION: THE MASTER IS ISOLATED!")
        print(" [TEST 4.1] The Master is going offline for 45 seconds (Sleep).")
        print(" [TEST 4.1] Workers will continue to finish tasks and fill the SQS queues.")
        print("!"*50 + "\n")
        time.sleep(45)
        # Upon "waking up", the Master will find the ACKs ready and empty the queue in bulk.
        # ==========================================================
        """

        # 5. Wait for worker results via SQS polling
        self._wait_for_workers(job_id, num_workers, completed_train_tasks, start_train, tasks_dispatched, final_train_url)

        # 6. Closure and client notification
        total_run_time = time.time() - start_train
        print(f" [TIMERS] Distributed Training completed in {total_run_time:.2f}s")

        """
        # ==========================================================
        # TEST 1.6 (MASTER CRASH BEFORE FINAL ACK TO CLIENT)
        # ==========================================================
        print("\n" + "!"*50)
        print(" [TEST 1.6] TRAINING 100% COMPLETE. STATE SAVED.")
        print(" [TEST 1.6] You have 15 seconds to kill the Master!")
        print(" [TEST 1.6] Master hasn't responded to the Client yet.")
        print("!"*50 + "\n")
        time.sleep(15)
        # ==========================================================
        """

        if job_data.get('mode') == 'train':
            self._send_client_response(job_id, "train", total_run_time)

    # Recovers previous training state from DynamoDB or initializes a new one
    def _recover_or_initialize_state(self, job_id, train_url):
        completed_train_tasks, s3_results, db_start, tasks_dispatched, train_time, _, db_train_url= self.aws.get_job_state(job_id)

        if db_start is None:
            start_train = time.time()
            self.aws.update_job_state(job_id, completed_train_tasks, s3_results, start_train, False, train_time, 0.0, train_url)
        else:
            start_train = db_start
            print(f" [RECOVERY] Restored. Current state: {len(completed_train_tasks)} Train tasks complete. (Train URL: {train_url})")

        return completed_train_tasks, s3_results, start_train, tasks_dispatched, train_time, train_url

    # Validates the existence of the training dataset on S3 and counts its total rows
    def _ensure_dataset_ready(self, train_url):
        if not train_url:
            raise ValueError("Training URL is missing from the job configuration.")

        try:
            bucket, target_train_key = self.aws.parse_s3_uri(train_url)
        except Exception:
            raise ValueError(f"Invalid S3 URL format for training target: {train_url}")

        # Native fault tolerance: does S3 file exist?
        if self.aws.check_s3_file_exists(bucket, target_train_key):
            print(f" [PIPELINE] Dataset ready on {train_url}.")
            return self.aws.get_total_rows_s3_select(bucket, target_train_key)
        else:
            raise ValueError(f"Impossible to start job: the training file {train_url} DOES NOT exist on S3.")

    # Fetches optimized hyperparameter strategies from a user-provided S3 JSON file
    def _fetch_custom_strategies(self, strategies_url, strategy_type, num_trees, num_workers, task_type):
        default_criterion = "gini" if task_type == 'classification' else 'squared_error'
        fallback = [{"max_depth": None, "max_features": "sqrt", "criterion": default_criterion}] * num_workers

        if not strategies_url:
            return fallback

        try:
            # Parse the provided S3 URL
            bucket, key = self.aws.parse_s3_uri(strategies_url)

            # Download the config file from S3
            response = self.aws.s3_client.get_object(Bucket=bucket, Key=key)
            strategies_data = json.loads(response['Body'].read().decode('utf-8'))

            def get_closest_key(data_dict, target_val):
                valid_keys = [int(k) for k in data_dict.keys() if k.isdigit()]
                if not valid_keys:
                    return None
                if target_val in valid_keys:
                    return str(target_val)
                # Find the closest key
                closest = min(valid_keys, key=lambda k: abs(k - target_val))
                return str(closest)

            if strategy_type == "homogeneous":
                matched_key = get_closest_key(strategies_data, num_trees)
                if matched_key:
                    conf = strategies_data[matched_key]
                    if matched_key != str(num_trees):
                        print(
                            f" [INFO] Exact homogeneous config for {num_trees} trees not found. Falling back to closest match: {matched_key} trees.")
                    else:
                        print(f" [INFO] Loaded Homogeneous strategy from {strategies_url} for {num_trees} trees.")
                    return [conf] * num_workers

            elif strategy_type == "heterogeneous":
                matched_key = get_closest_key(strategies_data, num_workers)
                if matched_key:
                    conf_list = strategies_data[matched_key]
                    if matched_key != str(num_workers):
                        print(
                            f" [INFO] Exact heterogeneous config for {num_workers} workers not found. Falling back to closest match: {matched_key} workers.")
                    else:
                        print(f" [INFO] Loaded Heterogeneous strategy from {strategies_url} for {num_workers} workers.")
                    return conf_list

        except Exception as e:
            print(f" [WARNING] Could not load strategies from {strategies_url} ({e}). Using standard fallback.")

        return fallback

    # Generates and queues individual training tasks for each worker via SQS
    def _generate_tasks(self, job_data, job_id, total_rows):
        num_workers = job_data['num_workers']
        num_trees_total = job_data['num_trees']
        train_s3_uri = job_data['train_url']
        task_type = job_data.get('task_type')
        target_col = job_data.get('target_column', 'Label')

        target_strategies = job_data.get('custom_hyperparams')
        strategy_type = job_data.get('strategy', 'homogeneous')
        strategies_url = job_data.get('strategies_url')

        # Fallback to user-provided S3 JSON configuration (or standard ML if it fails/is missing)
        if not target_strategies:
            target_strategies = self._fetch_custom_strategies(strategies_url, strategy_type, num_trees_total,
                                                              num_workers, task_type)

        rows_per_worker = total_rows // num_workers
        remainder_rows = total_rows % num_workers
        trees_per_worker = math.floor(num_trees_total / num_workers)
        trees_remainder = num_trees_total % num_workers
        current_skip = 0

        print(f" [INFO] Distributing {num_trees_total} trees across {num_workers} training tasks...")
        for i in range(num_workers):
            trees = trees_per_worker + (1 if i < trees_remainder else 0)
            n_rows = rows_per_worker + (remainder_rows if i == num_workers - 1 else 0)
            #circular buffer for lack of perfect configuration
            conf = target_strategies[i % len(target_strategies)]

            raw_depth = conf.get('max_depth')
            max_depth = None if raw_depth in ["None", None] else (
                int(raw_depth) if str(raw_depth).lstrip('-').isdigit() else None)

            raw_features = conf.get('max_features', 'sqrt')
            if raw_features not in ["sqrt", "log2", "None", None]:
                try:
                    val_float = float(raw_features)
                    raw_features = int(val_float) if val_float.is_integer() else val_float
                except:
                    raw_features = "sqrt"

            raw_samples = float(conf.get('max_samples', 1.0))

            task_payload = {
                "job_id": job_id,
                "task_id": f"task_{i + 1}",
                "seed": i * 1000,
                "dataset_s3_path": train_s3_uri,
                "trees": trees,
                "skip_rows": current_skip,
                "num_rows": n_rows,
                "max_depth": max_depth,
                "max_features": raw_features,
                "criterion": conf.get('criterion'),
                "min_samples_split": conf.get('min_samples_split', 2),
                "min_samples_leaf": conf.get('min_samples_leaf', 1),
                "max_samples": raw_samples,
                "class_weight": conf.get('class_weight', None),
                "n_jobs": conf.get('n_jobs', -1),
                "custom_target_col": target_col,
                "task_type": task_type
            }

            current_skip += n_rows
            self.aws.sqs_client.send_message(QueueUrl=self.aws.sqs_queues["train_task"],
                                             MessageBody=json.dumps(task_payload))
            print(f" Enqueued {task_payload['task_id']} ({trees} trees).")

            """
            # ==========================================================
            # TEST 1.4 (MASTER CRASH MID-FANOUT)
            # ==========================================================
            if i == num_workers // 2:
                print("\n" + "!"*50)
                print(f" [TEST 1.4] HALFWAY THROUGH DISPATCH ({i+1}/{num_workers} tasks sent).")
                print(" [TEST 1.4] Kill the Master NOW to create duplicates!")
                print("!"*50 + "\n")
                time.sleep(15)
            # ==========================================================
            """

    # Polls the SQS response queue until all workers complete their training tasks
    def _wait_for_workers(self, job_id, num_workers, completed_train_tasks, start_train, tasks_dispatched, train_url):
        print("\n [EVENT LOOP] Master listening actively for Worker responses...\n")
        train_resp_queue = self.aws.sqs_queues["train_response"]

        """
        # ==========================================================
        # TEST 1.5 (MASTER CRASH POST-FANOUT)
        # ==========================================================
        print("\n" + "!"*50)
        print(" [TEST 1.5] ALL TASKS SENT. TASKS_DISPATCHED = TRUE.")
        print(" [TEST 1.5] You have 15 seconds to kill the Master!")
        print(" [TEST 1.5] Workers are training. Master is just waiting.")
        print("!"*50 + "\n")
        time.sleep(15)
        # ==========================================================
        """

        while len(completed_train_tasks) < num_workers:
            res_train = self.aws.sqs_client.receive_message(QueueUrl=train_resp_queue, MaxNumberOfMessages=10,
                                                            WaitTimeSeconds=2)
            if 'Messages' in res_train:
                for msg in res_train['Messages']:
                    train_resp = json.loads(msg['Body'])
                    task_id = train_resp['task_id']

                    if task_id not in completed_train_tasks:
                        completed_train_tasks.add(task_id)
                        print(
                            f" [ACK] Worker completed training for {task_id}! ({len(completed_train_tasks)}/{num_workers})")
                        self.aws.update_job_state(job_id, completed_train_tasks, {}, start_train, tasks_dispatched, 0.0,
                                                  0.0, train_url)

                    self.aws.sqs_client.delete_message(QueueUrl=train_resp_queue, ReceiptHandle=msg['ReceiptHandle'])

        training_time = time.time() - start_train
        self.aws.update_job_state(job_id, completed_train_tasks, {}, start_train, tasks_dispatched, training_time, 0.0, train_url)
        print("\n [PIPELINE] All Workers completed their Training tasks!")

    # Notifies the client that the training pipeline has finished
    def _send_client_response(self, job_id, mode, total_time):
        client_response_queue = self.aws.sqs_queues["client_response"]
        if client_response_queue:
            payload = {
                "job_id": job_id,
                "target_model": job_id,
                "mode": mode,
                "total_time_sec": round(total_time, 2),
                "status": "SUCCESS"
            }
            self.aws.sqs_client.send_message(QueueUrl=client_response_queue, MessageBody=json.dumps(payload))