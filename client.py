import sys
import time

from src.utils.auxiliary import extract_dataset_name
from src.utils.config import load_config
from src.aws.client_aws_manager import ClientAWSManager
from src.client.cli_prompts import CLI


# Main entry point for the interactive command-line interface
def main():
    try:
        config = load_config()
    except Exception as e:
        print(f" [CRITICAL] Error loading config.json: {e}")
        sys.exit(1)

    cli = CLI(config)
    aws = ClientAWSManager(config)

    cli.show_welcome()

    # Gather user intentions and operation flow through interactive prompts
    mode = cli.prompt_operation_mode()

    # Prompt user directly for the necessary URLs based on the selected mode
    dataset_info = cli.prompt_dataset_selection(mode, aws)

    cluster_config = {}
    if mode in ['train', 'train_and_infer']:
        cluster_config = cli.prompt_cluster_config(dataset_info, aws)

    target_model = None
    if mode in ['infer', 'bulk_infer', 'download']:
        target_model = cli.prompt_model_selection(aws)

    if mode == 'download':
        aws.download_and_merge_model(target_model)
        sys.exit(0)

    # Handle real-time inference inputs and feature extraction
    tuple_data = None
    if mode == 'infer':
        # Safely extract the train_url to fetch CSV headers for user guidance
        train_url = dataset_info.get('train_url', "")
        s3_key = train_url.replace(f"s3://{aws.bucket}/", "") if train_url else ""

        tuple_data = cli.prompt_realtime_input(aws, train_url, dataset_info)

    dataset_url = dataset_info.get('train_url') or dataset_info.get('test_url') or ""
    ds_name = extract_dataset_name(dataset_url)

    # Generate a clean, flat Job ID
    timestamp = int(time.time())
    if mode in ['train', 'train_and_infer']:
        w = cluster_config.get('workers', 0)
        t = cluster_config.get('trees', 0)
        strat = cluster_config.get('strategy')
        job_id = f"job_{ds_name}_{strat}_{t}trees_{w}workers_{timestamp}"
    else:
        job_id = f"req_{ds_name}_{timestamp}"

    # Construct the minimal payload representing the client contract for the master node
    payload = {
        "mode": mode,
        "job_id": job_id,
        "train_url": dataset_info.get('train_url', ""),
        "test_url": dataset_info.get('test_url', ""),
        "target_column": dataset_info.get('target_col', ""),
        "task_type": dataset_info.get('task_type', "classification"),

        "num_workers": cluster_config.get('workers', 0),
        "num_trees": cluster_config.get('trees', 0),
        "strategy": cluster_config.get('strategy', 'homogeneous'),
        "custom_hyperparams": cluster_config.get('custom_hyperparams'),
        "strategies_url": cluster_config.get('strategies_url', ""),

        "target_model": target_model,
        "tuple_data": tuple_data,
        "client_start_time": time.time()
    }

    print(f" [SYSTEM] Dispatching job {job_id} to the cluster...")

    # Dispatch the payload to the queue and wait for cluster response
    aws.dispatch_and_wait(payload)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n [SYSTEM] Client terminated by user.")
        sys.exit(0)