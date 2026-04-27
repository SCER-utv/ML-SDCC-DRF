import os
import json
import time
import joblib
import boto3
import botocore


# Handles all communications between the local client and the AWS infrastructure
class ClientAWSManager:

    def __init__(self, config):
        # 1. Retrieving base params from environment variables
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.bucket = os.getenv("S3_BUCKET_NAME")

        if not self.bucket:
            raise ValueError(" [CRITICAL] S3_BUCKET_NAME environment variable is not set!")

        # 2. Initializing AWS clients
        self.s3_client = boto3.client('s3', region_name=self.region)
        self.sqs_client = boto3.client('sqs', region_name=self.region)
        self.ssm_client = boto3.client('ssm', region_name=self.region)

        config_key = os.getenv("S3_CONFIG_KEY", "config/ssm_paths.json")
        ssm_paths = self._load_remote_config(config_key)

        # 3. Initializing queues by retrieving name from SSM and URL
        print(" [INIT] SQS URL dynamic runtime resolution ...")
        self.client_queue_url = self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_client")))
        self.client_resp_queue = self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_client_resp")))

    # Used to load the SSM path file from S3
    def _load_remote_config(self, key):
        try:
            print(f" [INIT] Download infrastructural configuration from s3://{self.bucket}/{key}...")
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            config_data = json.loads(response['Body'].read().decode('utf-8'))
            return config_data
        except Exception as e:
            print(f" [CRITICAL ERROR] Impossible to load configuration {key} from S3: {e}")
            raise e

    # Retrieves a parameter from AWS SSM Parameter Store
    def _get_ssm_parameter(self, param_name):
        try:
            response = self.ssm_client.get_parameter(Name=param_name, WithDecryption=False)
            return response['Parameter']['Value']
        except Exception as e:
            print(f" [SSM ERROR] Impossible to extract parameter {param_name}: {e}")
            raise e

    # Resolves the exact queue URL given its name
    def _resolve_sqs_url(self, queue_name):
        try:
            response = self.sqs_client.get_queue_url(QueueName=queue_name)
            return response['QueueUrl']
        except Exception as e:
            print(f" [SQS ERROR] Impossible to resolute URL for the queue '{queue_name}': {e}")
            raise e

    # Scans S3 to find all trained models using the flat architecture
    def list_available_models(self):
        prefix = "models/"
        resp = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix, Delimiter='/')
        models = []

        if 'CommonPrefixes' in resp:
            for obj in resp['CommonPrefixes']:
                # The folder name itself is the job_id (e.g., job_100trees_4workers_1776195424)
                folder_name = obj['Prefix'].replace(prefix, '').strip('/')
                models.append(folder_name)

        return models

    # Reads the header of a CSV on S3 to retrieve column names.
    # If a target_column is provided, it removes it from the list (useful for ML feature extraction).
    def get_csv_headers_from_s3(self, s3_bucket, s3_key, target_column_to_remove=None):
        try:
            response = self.s3_client.get_object(Bucket=s3_bucket, Key=s3_key)

            # Extracts first row
            first_line = next(response['Body'].iter_lines()).decode('utf-8')
            response['Body'].close()

            all_columns = [col.strip() for col in first_line.split(',')]

            # It removes target col if needed
            if target_column_to_remove and target_column_to_remove in all_columns:
                all_columns.remove(target_column_to_remove)

            return all_columns

        except Exception as e:
            print(f" [WARNING] Unable to read header from S3: {e}")
            return []

    # Extracts bucket and key from an S3 URL safely
    @staticmethod
    def parse_s3_uri(s3_uri):
        if s3_uri is None or s3_uri == "":
            return "", ""
        parts = s3_uri.replace("s3://", "").split("/", 1)
        if len(parts) < 2:
            return parts[0], ""
        return parts[0], parts[1]

    # Rapid check of a file existence on S3
    def check_s3_file_exists(self, bucket, key):
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                return False
            raise e

    # Downloads worker .joblib files and merges them into a single local Scikit-Learn model
    def download_and_merge_model(self, target_model):
        print(f"\n" + "-" * 40)
        print(f" [DOWNLOAD] Fetching model chunks for {target_model}...")

        # Searching chunks directly in models/<target_model>/
        prefix = f"models/{target_model}/"
        resp = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)

        if 'Contents' not in resp:
            print(" [ERROR] No chunks found for this model on S3.")
            return

        os.makedirs("tmp_downloads", exist_ok=True)
        downloaded_files = []

        for obj in resp['Contents']:
            if obj['Key'].endswith('.joblib'):
                file_name = obj['Key'].split('/')[-1]
                local_path = os.path.join("tmp_downloads", file_name)
                self.s3_client.download_file(self.bucket, obj['Key'], local_path)
                downloaded_files.append(local_path)
                print(f"   -> Downloaded {file_name}")

        if not downloaded_files:
            print(" [ERROR] No .joblib files found.")
            return

        print("\n [MERGE] Aggregating distributed trees into a single Scikit-Learn Model...")
        base_model = None

        for file in downloaded_files:
            rf_chunk = joblib.load(file)
            if base_model is None:
                base_model = rf_chunk
            else:
                base_model.estimators_.extend(rf_chunk.estimators_)
                base_model.n_estimators += len(rf_chunk.estimators_)
            os.remove(file)

        try:
            os.rmdir("tmp_downloads")
        except:
            pass

        output_filename = f"{target_model}_aggregated.pkl"
        joblib.dump(base_model, output_filename)

        print("\n" + "=" * 60)
        print(f" SUCCESS! Model aggregated and saved locally.")
        print(f" File Name  : {output_filename}")
        print(f" Total Trees: {base_model.n_estimators}")
        print("=" * 60 + "\n")

    # Dispatches the job to the Master Node and patiently waits for the response
    def dispatch_and_wait(self, payload):
        print("\n" + "=" * 60)
        print(" Dispatching request to Master Node...")

        try:
            self.sqs_client.send_message(
                QueueUrl=self.client_queue_url,
                MessageBody=json.dumps(payload),
                MessageGroupId="ML_Jobs",
                MessageDeduplicationId=payload['job_id']
            )
            print(f" [SUCCESS] Message enqueued successfully.")
            print(f" [INFO] Generated Job ID: {payload['job_id']}")

            print(f"\n [WAIT] Waiting for response from the cluster...")
            print(" (If machines are cold-starting or training, this may take some time)")

            start_wait = time.time()
            result_found = False

            # 15 minutes timeout
            while time.time() - start_wait < 900:
                res = self.sqs_client.receive_message(QueueUrl=self.client_resp_queue, MaxNumberOfMessages=1,
                                                      WaitTimeSeconds=20)

                if 'Messages' in res:
                    for msg in res['Messages']:
                        body = json.loads(msg['Body'])
                        receipt = msg['ReceiptHandle']

                        # Check if it's the response for our exact job
                        if body.get("job_id") == payload['job_id']:
                            print("\n" + "=" * 60)

                            if body.get("status") == "FAILED":
                                print(" CRITICAL ERROR FROM MASTER NODE")
                                print("=" * 60)
                                print(f" REASON: {body.get('message', 'Unknown Error')}")
                                print("=" * 60 + "\n")

                                self.sqs_client.delete_message(QueueUrl=self.client_resp_queue, ReceiptHandle=receipt)
                                result_found = True
                                break

                            # Handle SUCCESS responses based on the mode
                            if payload['mode'] == 'train':
                                print(" DISTRIBUTED TRAINING COMPLETED!")
                                print("=" * 60)
                                print(f" YOUR MODEL ID IS: >>> {body.get('job_id')} <<<")
                            elif payload['mode'] == 'train_and_infer':
                                print(" END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
                                print("=" * 60)
                                print(f" YOUR MODEL ID IS: >>> {body.get('job_id')} <<<")
                                print(" Metrics saved on S3.")
                            elif payload['mode'] == 'infer':
                                print(" REAL-TIME PREDICTION RECEIVED!")
                                print("=" * 60)
                                print(f" Task Type       : {body.get('task_type')}")
                                print(f" PREDICTION      : >>> {body.get('prediction')} <<<")
                            elif payload['mode'] == 'bulk_infer':
                                print(" BULK INFERENCE COMPLETED!")
                                print("=" * 60)
                                print(f" Metrics saved to S3. Check the results file.")

                            if "metrics" in body:
                                print("\n --- EVALUATION RESULTS ---")
                                for metric_name, value in body["metrics"].items():
                                    display_name = metric_name.replace('_', ' ').title()
                                    if isinstance(value, float):
                                        print(f"  {display_name:<20}: {value:.4f}")
                                    else:
                                        print(f"  {display_name:<20}: {value}")
                                print(" --------------------------")
                                print(" Metrics successfully logged to S3.")

                            print(f"\n Cluster Latency : {body.get('total_time_sec')} seconds")
                            print("=" * 60 + "\n")

                            self.sqs_client.delete_message(QueueUrl=self.client_resp_queue, ReceiptHandle=receipt)
                            result_found = True
                            break
                        else:
                            # Not for us, put the message back in the queue immediately
                            try:
                                self.sqs_client.change_message_visibility(
                                    QueueUrl=self.client_resp_queue,
                                    ReceiptHandle=receipt,
                                    VisibilityTimeout=0
                                )
                            except Exception:
                                pass
                if result_found:
                    break

            if not result_found:
                print("\n [TIMEOUT] The cluster took too long to respond. Check Master logs.")

        except Exception as e:
            print(f"\n [CRITICAL ERROR] Failed to dispatch SQS message: {e}")