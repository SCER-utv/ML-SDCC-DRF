import json
import threading

from src.master_core.evaluation_manager import EvaluationManager
from src.master_core.inference_pipeline import InferencePipeline
from src.master_core.training_pipeline import TrainingPipeline
from src.utils.config import load_config

from src.aws.aws_manager import AWSManager
from src.utils.job_paths import JobPaths


# extends sqs message visibility in the background to prevent timeout during long jobs
def extend_client_sqs_visibility(aws, queue_url, receipt_handle, stop_event):
    while not stop_event.is_set():
        stop_event.wait(20)
        if not stop_event.is_set():
            try:
                aws.sqs_client.change_message_visibility(QueueUrl=queue_url, ReceiptHandle=receipt_handle,
                                                         VisibilityTimeout=60)
            except Exception:
                pass


# NUOVA VERSIONE MINIMALISTA: prende gli URL diretti e crea le chiavi S3 piatte
def resolve_paths(job_data):
    job_id = job_data['job_id']

    # Estraiamo direttamente gli URL nudi e crudi forniti dal client
    train_url = job_data.get('train_url', "")
    test_url = job_data.get('test_url', "")

    # Struttura piatta per i risultati S3
    metrics_key = f"metrics/{job_id}_results.csv"

    # Manteniamo la classe JobPaths per compatibilità con le pipeline,
    # ma disattiviamo lo split (raw_source_to_split=None)
    dataset_paths = JobPaths(
        train_url=train_url,
        test_url=test_url,
        metrics_key=metrics_key,
        raw_source_to_split=None
    )

    job_data['dataset_paths'] = dataset_paths
    return job_data


# initializes components and starts the main event loop to orchestrate client jobs
def main():
    print(" [MASTER] Components initialization...")
    config = load_config()

    aws = AWSManager(config)
    evaluator = EvaluationManager(aws)
    trainer = TrainingPipeline(aws)
    inferencer = InferencePipeline(aws, evaluator)

    CLIENT_QUEUE_URL = aws.sqs_queues["client"]
    print(" [MASTER] System ready. Listening for pure computation Jobs from client...")

    while True:
        response = aws.sqs_client.receive_message(QueueUrl=CLIENT_QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=20)

        if 'Messages' in response:
            client_msg = response['Messages'][0]
            receipt_handle = client_msg['ReceiptHandle']
            raw_job_data = json.loads(client_msg['Body'])

            raw_job_data.setdefault('job_id', client_msg['MessageId'])
            mode = raw_job_data.get('mode', 'train')
            job_id = raw_job_data['job_id']

            print(f"\n{'=' * 50}\n STARTING PIPELINE: {job_id} (Mode: {mode})\n{'=' * 50}")

            # Chiamata pulita, senza bisogno di config o aws
            job_data = resolve_paths(raw_job_data)

            stop_event = threading.Event()
            heartbeat_thread = threading.Thread(target=extend_client_sqs_visibility,
                                                args=(aws, CLIENT_QUEUE_URL, receipt_handle, stop_event))
            heartbeat_thread.start()

            try:
                # dynamically route the job to the appropriate pipeline
                if mode == 'train':
                    trainer.run(job_data, job_id)
                elif mode == 'bulk_infer':
                    inferencer.run_bulk(job_data, job_id)
                elif mode == 'infer':
                    inferencer.run_realtime(job_data, job_id)
                elif mode == 'train_and_infer':
                    trainer.run(job_data, job_id)
                    job_data['target_model'] = job_id
                    inferencer.run_bulk(job_data, job_id)
                else:
                    print(f" [WARNING] Unknown mode requested: {mode}")

            except Exception as e:
                error_msg = str(e)
                print(f" [CRITICAL ERROR] Pipeline execution failed: {error_msg}")

                # NATIVE FAULT TOLERANCE: Invia l'errore al client tramite SQS
                try:
                    error_payload = {
                        "job_id": job_id,
                        "status": "FAILED",
                        "message": error_msg
                    }

                    resp_queue_url = aws.sqs_queues.get("client_response")
                    if resp_queue_url:
                        aws.sqs_client.send_message(
                            QueueUrl=resp_queue_url,
                            MessageBody=json.dumps(error_payload)
                        )
                        print(" [MASTER] Error successfully communicated to Client via SQS.")
                    else:
                        print(" [MASTER ERROR] Could not find response queue URL to notify client.")
                except Exception as sqs_err:
                    print(f" [MASTER ERROR] Failed to send error message to SQS: {sqs_err}")
            finally:
                # ensure heartbeat stops and the processed message is deleted from the queue
                stop_event.set()
                heartbeat_thread.join()
                aws.delete_message(CLIENT_QUEUE_URL, receipt_handle)
                print(f" JOB {job_id} PROCESSED AND REMOVED FROM QUEUE.\n")


if __name__ == "__main__":
    main()