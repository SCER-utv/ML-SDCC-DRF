import sys
import datetime


class CLI:

    # Initializes the CLI, config is kept for potential future extensions like sqs queues setup
    def __init__(self, config):
        self.config = config

    # Clears the terminal screen for a cleaner user interface
    def clear_screen(self):
        print("\n" * 2)

    # Displays the main welcome banner
    def show_welcome(self):
        self.clear_screen()
        print("=" * 60)
        print("  DISTRIBUTED RANDOM FOREST - CLI CLIENT ")
        print("=" * 60)

    # Prompts the user to select the primary operation mode
    def prompt_operation_mode(self):
        print("\n Select Operation Mode:")
        print("  1) Distributed Training (Training Only)")
        print("  2) Bulk Inference (Test Set Evaluation)")
        print("  3) End-to-End Pipeline (Train + Evaluate)")
        print("  4) Real-Time Inference (Single Prediction)")
        print("  5) Download Aggregated Model")

        while True:
            mode_choice = input("\n Enter 1, 2, 3, 4 or 5: ").strip()
            if mode_choice == '1': return 'train'
            if mode_choice == '2': return 'bulk_infer'
            if mode_choice == '3': return 'train_and_infer'
            if mode_choice == '4': return 'infer'
            if mode_choice == '5': return 'download'
            print(" [ERROR] Invalid choice. Please try again.")

    # Prompts the user to provide the exact S3 URLs and target column based on the operation mode
    def prompt_dataset_selection(self, mode):
        # If downloading, we don't need dataset info
        if mode == 'download':
            return {}

        print("\n" + "-" * 40)
        print(" Dataset Configuration")

        dataset_info = {
            "train_url": "", "test_url": "",
            "target_col": "", "task_type": "classification"
        }

        # Helper function to validate S3 URI input
        def get_s3_input(prompt_text):
            while True:
                url = input(prompt_text).strip()
                if url.startswith("s3://") and url.endswith(".csv"):
                    return url
                print(" [ERROR] Invalid format. Must start with 's3://' and end with '.csv'.")

        # Ask only for what is strictly necessary based on the mode
        if mode in ['train', 'train_and_infer']:
            dataset_info["train_url"] = get_s3_input("\n Enter the S3 URL of the TRAINING Dataset: ")

        if mode in ['bulk_infer', 'train_and_infer']:
            dataset_info["test_url"] = get_s3_input("\n Enter the S3 URL of the TEST Dataset: ")

        if mode == 'infer':
            dataset_info["train_url"] = get_s3_input(
                "\n Enter the S3 URL of the TRAINING Dataset (used to extract feature names): ")

        dataset_info["target_col"] = input(
            "\n Enter the EXACT name of the Target Column to predict (e.g., Label): ").strip()

        print("\n Specify the ML Task Type:")
        print("  1) Binary Classification\n  2) Regression")
        while True:
            task_choice = input(" Enter 1 or 2: ").strip()
            if task_choice in ['1', '2']:
                dataset_info["task_type"] = "classification" if task_choice == '1' else "regression"
                break
            print(" [ERROR] Invalid choice. Please enter 1 or 2.")

        return dataset_info

    # Gathers cluster settings and machine learning hyperparameters
    def prompt_cluster_config(self, dataset_info):
        print("\n" + "-" * 40)
        print(" Cluster & Hyperparameter Configuration")

        config_data = {}
        while True:
            try:
                config_data['workers'] = int(input("\n Enter number of Workers (1-8): "))

                if config_data['workers'] <= 0 or config_data['workers'] > 8:
                    print(" [ERROR] Values must be between 1-8.")
                else:
                    break
            except ValueError:
                print(" [ERROR] Invalid input. Please enter integers only.")

        while True:
            try:
                config_data['trees'] = int(input(" Enter TOTAL number of Trees (>= number of workers): "))

                if config_data['trees'] <= 0:
                    print(" [ERROR] Values must be greater than zero.")
                elif config_data['trees'] < config_data['workers']:
                    print(" [ERROR] Values must be greater or equal to number of workers.")
                else:
                    break
            except ValueError:
                print(" [ERROR] Invalid input. Please enter integers only.")

        print("\n Select Training Strategy:")
        print("  1) Homogeneous  [Same parameters for all workers]")
        print("  2) Heterogeneous [Different parameters per worker, variance boosting]")
        while True:
            strat_choice = input(" Enter 1 or 2: ").strip()
            if strat_choice in ['1', '2']:
                config_data['strategy'] = "homogeneous" if strat_choice == '1' else "heterogeneous"
                break
            print(" [ERROR] Invalid choice.")

        if config_data['strategy'] == "heterogeneous":
            print("\n [INFO] Heterogeneous strategy requires different parameters for each worker.")
            print(" Select Hyperparameter Source:")
            print("  1) Manual Configuration")
            print("  2) Load from S3 JSON file")
            while True:
                hyper_source = input(" Enter 1 or 2: ").strip()
                if hyper_source == '1':
                    hyper_source = '2'  # Map to manual logic
                    break
                elif hyper_source == '2':
                    hyper_source = '3'  # Map to S3 logic
                    break
                print(" [ERROR] Invalid choice.")
        else:
            print("\n Select Hyperparameter Source:")
            print("  1) Default Generic Parameters (Fast execution using Standard Scikit-Learn)")
            print("  2) Manual Configuration")
            print("  3) Load strategies from S3 JSON file")
            while True:
                hyper_source = input(" Enter 1, 2, or 3: ").strip()
                if hyper_source in ['1', '2', '3']:
                    break
                print(" [ERROR] Invalid choice.")

        config_data['custom_hyperparams'] = None
        config_data['strategies_url'] = None

        if hyper_source == '3':
            while True:
                url = input("\n Enter the S3 URL for the strategies JSON: ").strip()
                if url.startswith("s3://") and url.endswith(".json"):
                    config_data['strategies_url'] = url
                    break
                print(" [ERROR] Invalid format. Must start with 's3://' and end with '.json'.")
        elif hyper_source == '2':
            print("\n [MANUAL HYPERPARAMETERS CONFIGURATION]")
            config_data['custom_hyperparams'] = []
            iterations = 1 if config_data['strategy'] == "homogeneous" else config_data['workers']

            for w in range(iterations):
                if config_data['strategy'] == "heterogeneous":
                    print(f"\n --- Configuring Worker {w + 1}/{config_data['workers']} ---")
                else:
                    print("\n --- Configuring Global Parameters ---")

                raw_depth = input(" Max Depth (int, or blank for None): ").strip()
                max_depth = int(raw_depth) if raw_depth.isdigit() else None

                raw_split = input(" Min Samples Split (int or float, default: 2): ").strip()
                try:
                    min_samples_split = float(raw_split) if '.' in raw_split else int(raw_split)
                except ValueError:
                    min_samples_split = 2

                raw_leaf = input(" Min Samples Leaf (int or float, default: 1): ").strip()
                try:
                    min_samples_leaf = float(raw_leaf) if '.' in raw_leaf else int(raw_leaf)
                except ValueError:
                    min_samples_leaf = 1

                raw_features = input(
                    " Max Features ['sqrt', 'log2', float < 1.0, or blank for None] (Default: sqrt): ").strip()
                if not raw_features or raw_features == "sqrt":
                    max_features = "sqrt"
                elif raw_features == "log2":
                    max_features = "log2"
                elif raw_features.lower() == "none":
                    max_features = None
                else:
                    try:
                        max_features = float(raw_features)
                    except ValueError:
                        max_features = "sqrt"

                raw_samples = input(" Max Samples per Tree [0.1 - 1.0, or blank for None] (Default: 1.0): ").strip()
                if raw_samples.lower() == "none":
                    max_samples = None
                else:
                    try:
                        max_samples = float(raw_samples)
                    except ValueError:
                        max_samples = 1.0

                criterion = input(" Criterion [gini, entropy, squared_error] (Leave blank for default): ").strip()
                if not criterion:
                    criterion = "gini" if "classification" in dataset_info.get('task_type',
                                                                               '').lower() else "squared_error"

                class_weight = None
                if "classification" in dataset_info.get('task_type', '').lower():
                    raw_cw = input(" Class Weight [balanced, balanced_subsample] (Leave blank for None): ").strip()
                    if raw_cw in ["balanced", "balanced_subsample"]:
                        class_weight = raw_cw

                worker_params = {
                    "max_depth": max_depth, "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf, "max_features": max_features,
                    "max_samples": max_samples, "criterion": criterion,
                    "class_weight": class_weight, "n_jobs": -1
                }
                config_data['custom_hyperparams'].append(worker_params)

            if config_data['strategy'] == "homogeneous":
                config_data['custom_hyperparams'] = config_data['custom_hyperparams'] * config_data['workers']

        return config_data

    # Allows the user to select an existing distributed model from S3 or paste an ID
    def prompt_model_selection(self, aws_manager):
        print("\n" + "-" * 40)
        print(" Select Target Model ID:")
        print("  1) Paste a specific Model ID")
        print("  2) Scan S3 for all available models")

        while True:
            sel_method = input("\n Enter 1 or 2: ").strip()
            if sel_method in ['1', '2']: break
            print(" [ERROR] Invalid choice.")

        if sel_method == '1':
            valid_models = aws_manager.list_available_models()
            while True:
                target_model = input("\n Paste the exact Model ID (e.g., job_...): ").strip()
                if target_model not in valid_models:
                    print(f" [ERROR] Model '{target_model}' not found in S3. Please check for typos.")
                else:
                    return target_model
        else:
            print("\n [SEARCH] Scanning S3 for saved models...")
            # We assume list_available_models now returns all model names dynamically
            models = aws_manager.list_available_models()

            if not models:
                print("\n [ERROR] No trained models found in the bucket. Run a training job first!")
                sys.exit(0)

            print("\n=== AVAILABLE MODELS ===")
            for i, m in enumerate(models):
                try:
                    parts = m.split('_')
                    trees_count = next((p.replace('trees', '') for p in parts if 'trees' in p), "?")
                    workers_count = next((p.replace('workers', '') for p in parts if 'workers' in p), "?")

                    date_formatted, time_formatted = "????/??/??", "??:??:??"

                    # Parses Unix Timestamp from the job ID
                    if len(parts) >= 1 and parts[-1].isdigit():
                        try:
                            timestamp = int(parts[-1])
                            dt = datetime.datetime.fromtimestamp(timestamp)
                            date_formatted = dt.strftime("%d/%m/%Y")
                            time_formatted = dt.strftime("%H:%M:%S")
                        except Exception:
                            pass

                    print(
                        f"  [{i}]  Trees: {trees_count:<4} | Workers: {workers_count:<2} | Date: {date_formatted} {time_formatted}  (ID: {m})")
                except Exception:
                    print(f"  [{i}] {m}")

            while True:
                try:
                    model_choice = int(input(f"\n Select Model ID [0-{len(models) - 1}]: "))
                    if 0 <= model_choice < len(models):
                        return models[model_choice]
                    print(" [ERROR] Invalid ID selected.")
                except ValueError:
                    print(" [ERROR] Please enter a valid number.")

    # Prompts for manual data entry of features to perform real-time inference
    def prompt_realtime_input(self, aws_manager, dataset_s3_key, dataset_info):
        feature_names = None
        if dataset_s3_key:
            feature_names = aws_manager.get_feature_names_from_s3(dataset_s3_key,
                                                                  target_column=dataset_info.get('target_col', ''))

        required_features = len(feature_names) if feature_names else 0

        print("\n" + "-" * 40)
        print(" Real-Time Prediction Input")
        if required_features > 0:
            print(f" WARNING: Target model expects EXACTLY {required_features} features!")
            print(f"\n Expected layout: \n {', '.join(feature_names)}")
        else:
            print(
                " WARNING: Could not automatically extract feature names from the dataset. Please enter values blindly.")

        while True:
            prompt_text = f" Enter {required_features} comma-separated values: " if required_features > 0 else " Enter the comma-separated values: "
            raw_tuple = input(prompt_text).strip()
            try:
                tuple_data = [float(x.strip()) for x in raw_tuple.split(',')]
                if required_features == 0 or len(tuple_data) == required_features:
                    return tuple_data
                print(f" [ERROR] Expected {required_features} values, got {len(tuple_data)}.")
            except ValueError:
                print(" [ERROR] Formatting error. Use numbers only separated by comma (e.g., 10.5, 3).")