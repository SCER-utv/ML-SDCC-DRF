import json
import os
import io
import time
import random
import boto3
import botocore
import pandas as pd
from botocore.exceptions import ClientError


# This class handles all AWS interactions from the Master node
class AWSManager:

    def __init__(self, config):
        # 1. Retrieving base parameters from environment variables
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.bucket = os.getenv("S3_BUCKET_NAME")

        if not self.bucket:
            raise ValueError(" [CRITICAL] S3_BUCKET_NAME environment variable is not set!")

        # 2. Initializing AWS clients
        self.s3_client = boto3.client('s3', region_name=self.region)
        self.asg_client = boto3.client('autoscaling', region_name=self.region)
        self.ec2_client = boto3.client('ec2', region_name=self.region)
        self.sqs_client = boto3.client('sqs', region_name=self.region)
        self.ssm_client = boto3.client('ssm', region_name=self.region)

        # Get S3 key to retrieve configuration from SSM
        config_key = os.getenv("S3_CONFIG_KEY", "config/ssm_paths.json")
        ssm_paths = self._load_remote_config(config_key)

        # 3. Retrieve SSM parameters
        print(" [INIT] Retrieving infrastructure from AWS SSM...")
        self.asg_name = self._get_ssm_parameter(ssm_paths.get("asg_worker"))
        self.dynamodb_table = self._get_ssm_parameter(ssm_paths.get("dynamodb_table"))

        self.dynamodb = boto3.resource('dynamodb', region_name=self.region)

        # 4. Initializing queues by retrieving name from SSM and URL
        print(" [INIT] SQS URL dynamic runtime resolution...")
        self.sqs_queues = {
            "client": self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_client"))),
            "client_response": self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_client_resp"))),
            "train_task": self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_train"))),
            "train_response": self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_train_resp"))),
            "infer_task": self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_infer"))),
            "infer_response": self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_infer_resp")))
        }

        self.config = config

    # Used to load the SSM path mapping file from S3
    def _load_remote_config(self, key):
        try:
            print(f" [INIT] Download infrastructural configuration from s3://{self.bucket}/{key}...")
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            config_data = json.loads(response['Body'].read().decode('utf-8'))
            return config_data
        except Exception as e:
            print(f" [CRITICAL ERROR] Impossible to load configuration {key} from S3: {e}")
            raise e

    # Extracts the value (resource name) from AWS SSM Parameter Store
    def _get_ssm_parameter(self, param_name):
        try:
            response = self.ssm_client.get_parameter(Name=param_name, WithDecryption=False)
            return response['Parameter']['Value']
        except Exception as e:
            print(f" [SSM ERROR] Impossible to extract parameter {param_name}: {e}")
            raise e

    # Gets a complete URL of an SQS queue
    def _resolve_sqs_url(self, queue_name):
        try:
            response = self.sqs_client.get_queue_url(QueueName=queue_name)
            return response['QueueUrl']
        except Exception as e:
            print(f" [SQS ERROR] Impossible to resolute URL for the queue '{queue_name}': {e}")
            raise e

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

    # Executes a SQL query on S3 to count rows of a CSV file efficiently
    def get_total_rows_s3_select(self, bucket, key):
        print(f" [S3-SELECT] Executing 'SELECT count(*)' on s3://{bucket}/{key}...")
        try:
            resp = self.s3_client.select_object_content(
                Bucket=bucket, Key=key,
                ExpressionType='SQL', Expression='SELECT count(*) FROM S3Object',
                InputSerialization={'CSV': {'FileHeaderInfo': 'USE', 'AllowQuotedRecordDelimiter': False}},
                OutputSerialization={'CSV': {}}
            )
            for event in resp['Payload']:
                if 'Records' in event:
                    total_rows = int(event['Records']['Payload'].decode('utf-8').strip())
                    print(f" [S3 Select] Found {total_rows} rows!")
                    return total_rows
            return 0
        except Exception as e:
            print(f" [S3-SELECT ERROR] Failed query: {e}")
            raise e

    # Retrieves all distributed model chunks (.joblib) saved on S3 for a specific job ID
    def count_model_parts(self, bucket, target_model):
        # We now look directly in the flat models folder using the Job ID as prefix
        prefix = f"models/{target_model}/"
        resp = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        chunks = [f"s3://{bucket}/{obj['Key']}" for obj in resp.get('Contents', []) if obj['Key'].endswith('.joblib')]

        if not chunks:
            print(f" [WARNING S3] No .joblib file found on s3://{bucket}/{prefix}")

        return chunks

    @staticmethod
    def extract_dataset_name(s3_url):
        if not s3_url or not isinstance(s3_url, str):
            return "unknown_dataset"
        filename = s3_url.split('/')[-1]
        return filename.replace('.csv', '')

    # Saves the final evaluation metrics as a standalone CSV file on S3
    def save_metrics(self, report_data, dataset_name):

        # dynamic file path generation
        metrics_key = f"metrics/{dataset_name}_metrics.csv"
        print(f" [AWS] Saving final metrics to s3://{self.bucket}/{metrics_key}...")

        new_row_df = pd.DataFrame([report_data])

        # Convert to a Pandas DataFrame representing the new single row
        new_row_df = pd.DataFrame([report_data])

        try:
            # Try to download the existing file to append the new row
            response = self.s3_client.get_object(Bucket=self.bucket, Key=metrics_key)
            existing_df = pd.read_csv(io.BytesIO(response['Body'].read()))
            updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            print(" [AWS] Existing metrics file found. Appending new results.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                # File does not exist, this is the first execution for this dataset
                updated_df = new_row_df
                print(" [AWS] New dataset detected. Creating new metrics file.")
            else:
                print(f" [CRITICAL ERROR] Error accessing S3 object: {e}")
                return

        try:
            # Save the updated DataFrame to CSV in memory buffer
            csv_buffer = io.StringIO()
            updated_df.to_csv(csv_buffer, index=False)

            # Upload the buffer directly to S3
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=metrics_key,
                Body=csv_buffer.getvalue()
            )
            print(" [AWS] Metrics successfully saved on S3.")
        except Exception as e:
            print(f" [CRITICAL ERROR] Could not save metrics to S3: {e}")

    # Deletes temporary .npy prediction files generated by workers during inference
    def cleanup_s3_inference_files(self, s3_inference_results):
        print(" [CLEANUP] Deleting temporary .npy from S3...")
        deleted_count = 0
        for task_id, s3_uri in s3_inference_results.items():
            try:
                bucket, key = self.parse_s3_uri(s3_uri)
                self.s3_client.delete_object(Bucket=bucket, Key=key)
                deleted_count += 1
            except Exception as e:
                print(f" [CLEANUP ERROR] Delete error of {s3_uri}: {e}")
        print(f" [CLEANUP] Removed {deleted_count} temporary files successfully.")

    # Dynamically scales EC2 instances using the Auto Scaling Group
    def scale_worker_infrastructure(self, num_workers):
        print(f" [ASG] Setting desired capacity to {num_workers} workers...")
        self.asg_client.update_auto_scaling_group(
            AutoScalingGroupName=self.asg_name, MinSize=0, DesiredCapacity=num_workers, MaxSize=10
        )

        if num_workers == 0:
            return

        print(f" [ASG] Waiting for instances to start for tagging...")
        found_instances = []

        # 120 seconds timeout loop
        for _ in range(24):
            time.sleep(5)
            response = self.ec2_client.describe_instances(
                Filters=[
                    {'Name': 'tag:aws:autoscaling:groupName', 'Values': [self.asg_name]},
                    {'Name': 'instance-state-name', 'Values': ['pending', 'running']}
                ]
            )

            for reservation in response.get('Reservations', []):
                for inst in reservation.get('Instances', []):
                    found_instances.append(inst['InstanceId'])

            if len(found_instances) >= num_workers:
                break

        if len(found_instances) > 0:
            if len(found_instances) < num_workers:
                print(f" [ASG WARN] Requested {num_workers} workers, but AWS provided {len(found_instances)}. Proceeding degraded.")
            else:
                print(f" [ASG] Found {len(found_instances)} instances. Applying name tags...")

            for i, instance_id in enumerate(found_instances):
                try:
                    self.ec2_client.create_tags(
                        Resources=[instance_id],
                        Tags=[{'Key': 'Name', 'Value': f"DRF-worker-{i + 1}"}]
                    )
                except Exception:
                    pass
            print(" [ASG] Name tags applied successfully.")
        else:
            print(" [ASG CRITICAL] No instances provided by ASG within timeout!")

    # Retrieves the progress state of a specific job from DynamoDB
    def get_job_state(self, job_id):
        table = self.dynamodb.Table(self.dynamodb_table)
        try:
            response = table.get_item(Key={'job_id': job_id})
            if 'Item' in response:
                start_time = float(response['Item'].get('start_time'))
                tasks_dispatched = response['Item'].get('tasks_dispatched', False)
                training_time = float(response['Item'].get('tempo_training', 0.0))
                inference_time = float(response['Item'].get('tempo_inferenza', 0.0))
                train_url = response['Item'].get('train_url', 'unknown_url')

                return (set(response['Item'].get('completed_train', [])),
                        response['Item'].get('completed_infer', {}),
                        start_time, tasks_dispatched, training_time, inference_time, train_url)
        except Exception:
            pass
        return set(), {}, None, False, 0.0, 0.0, "unknown_url"

    # Updates the progress state of a job in DynamoDB
    def update_job_state(self, job_id, completed_train_set, completed_infer_dict, start_time, tasks_dispatched,
                         training_time=0.0, inference_time=0.0, train_url="unknown_url"):
        table = self.dynamodb.Table(self.dynamodb_table)
        table.put_item(Item={
            'job_id': job_id,
            'completed_train': list(completed_train_set),
            'completed_infer': completed_infer_dict,
            'start_time': str(start_time),
            'tasks_dispatched': tasks_dispatched,
            'tempo_training': str(training_time),
            'tempo_inferenza': str(inference_time),
            'train_url': train_url
        })

    # Deletes a processed message from an SQS queue
    def delete_message(self, queue_url, receipt_handle):
        self.sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)