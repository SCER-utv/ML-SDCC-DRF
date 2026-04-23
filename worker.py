import json
import time
import threading

from src.aws.worker_aws_manager import WorkerAWSManager
from src.utils.config import load_config
from src.worker_core.inference_handler import InferenceHandler
from src.worker_core.training_handler import TrainingHandler


# Executes a task (training or inference), manages the heartbeat, and responds to the master
def handle_task(aws, msg, in_queue, out_queue, handler_process_func, task_type):
    body = json.loads(msg['Body'])
    receipt = msg['ReceiptHandle']
    stop_heartbeat = threading.Event()

    # Start a background thread to keep the SQS message alive during long computations
    aws.start_heartbeat(in_queue, receipt, stop_heartbeat)

    try:
        # Dynamically invokes the process method of the appropriate handler class
        result = handler_process_func(body)

        # Build the response payload based on the task type
        response = {
            "job_id": body['job_id'],
            "task_id": body['task_id'],
        }

        if task_type == 'train':
            response["s3_model_uri"] = result
        else:
            response["s3_voti_uri"] = result

        # Send completion ACK back to the Master node
        aws.sqs_client.send_message(QueueUrl=out_queue, MessageBody=json.dumps(response))
        aws.delete_message(in_queue, receipt)
        print(f" [{task_type.upper()}] {body['task_id']} completed successfully!\n")

    except Exception as e:
        print(f" \n [FAULT TOLERANCE] Critical error during {task_type}: {e}")
        # Release the message immediately so another worker can pick it up (NACK)
        aws.release_message(in_queue, receipt)
        time.sleep(5)

    finally:
        # Ensure the heartbeat thread shuts down cleanly
        stop_heartbeat.set()


# Initializes the worker node, instantiates handlers, and starts the priority polling loop
def main():
    print(" [WORKER] Node initialized, waiting for tasks...")
    config = load_config()

    aws = WorkerAWSManager(config)
    trainer = TrainingHandler(aws, config)
    inferencer = InferenceHandler(aws, config)

    q_train_in = aws.sqs_queues["train_task"]
    q_train_out = aws.sqs_queues["train_response"]
    q_infer_in = aws.sqs_queues["infer_task"]
    q_infer_out = aws.sqs_queues["infer_response"]

    while True:
        try:
            # Priority 1: Training Tasks
            msg = aws.poll_queue(q_train_in, wait_time=5)
            if msg:
                handle_task(aws, msg, q_train_in, q_train_out, trainer.process, "train")
                continue

            # Priority 2: Inference Tasks
            msg = aws.poll_queue(q_infer_in, wait_time=5)
            if msg:
                handle_task(aws, msg, q_infer_in, q_infer_out, inferencer.process, "infer")
                continue

            # Brief pause if queues are empty to avoid CPU thrashing
            time.sleep(2)

        except Exception as e:
            print(f" [SYSTEM ERROR] Main polling loop encountered an error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n [WORKER] Terminated by user. Shutting down gracefully.")