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

    # Helper function to validate S3 URI input
    def _get_s3_input(self, prompt_text, aws, expected_extension=".csv"):
        while True:
            # Allow the user to abort if they realize they don't have the file
            url = input(prompt_text).strip()
            if url.lower() == 'exit':
                print(" [SYSTEM] Operation aborted by user. Exiting...")
                sys.exit(0)

            # 1. Format Validation
            if not (url.startswith("s3://") and url.endswith(expected_extension)):
                print(f" [ERROR] Invalid format. Must start with 's3://' and end with '{expected_extension}'.")
                continue

            # 2. Existence Validation
            print(" [VALIDATION] Verifying file existence on S3...")
            try:
                bucket, key = aws.parse_s3_uri(url)
                if aws.check_s3_file_exists(bucket, key):
                    print(" [SUCCESS] File found and validated!")
                    return url
                else:
                    print(f" [ERROR] File DOES NOT exist at '{url}'.")
                    print(" Please check for typos or type 'abort' to exit.")
            except Exception as e:
                print(f" [ERROR] Failed to validate S3 URL: {e}")

    # Prompts the user to provide the exact S3 URLs and target column based on the operation mode
    def prompt_dataset_selection(self, mode, aws):
        # If downloading, we don't need dataset info
        if mode == 'download':
            return {}

        print("\n" + "-" * 40)
        print(" Dataset Configuration")

        dataset_info = {
            "train_url": "", "test_url": "",
            "target_col": "", "task_type": "classification"
        }

        url_to_check = None
        # Ask only for what is strictly necessary based on the mode
        if mode in ['train', 'train_and_infer']:
            dataset_info["train_url"] = self._get_s3_input("\n Enter the S3 URL of the TRAINING Dataset (or type 'exit'): ", aws)
            url_to_check = dataset_info["train_url"]

        if mode in ['bulk_infer', 'train_and_infer']:
            dataset_info["test_url"] = self._get_s3_input("\n Enter the S3 URL of the TEST Dataset (or type 'exit'): ", aws)
            url_to_check = dataset_info["test_url"]

        if mode == 'infer':
            dataset_info["train_url"] = self._get_s3_input("\n Enter the S3 URL of the TRAINING Dataset (used to extract feature names)(or type 'exit'): ", aws)
            url_to_check = dataset_info["train_url"]

        available_columns = []
        if url_to_check:
            bucket, key = aws.parse_s3_uri(url_to_check)
            print(" [VALIDATION] Fetching column headers from S3...")
            try:
                available_columns = aws.get_csv_headers_from_s3(bucket, key)
            except Exception as e:
                print(f" [WARNING] Could not fetch headers for validation: {e}")


        while True:
            target_input = input("\n Enter the EXACT name of the Target Column to predict (Semantic validation not executed, make sure to insert the right Target column): ").strip()

            if available_columns:
                if target_input in available_columns:
                    dataset_info["target_col"] = target_input
                    break
                else:
                    print(f" [ERROR] Column '{target_input}' NOT FOUND in the dataset.")
                    print(f" Available columns are: {', '.join(available_columns[:5])}... (and others)")
            else:
                # Fallback if columns retrieve does not work
                dataset_info["target_col"] = target_input
                break



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
    def prompt_cluster_config(self, dataset_info, aws):
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
            config_data['strategies_url'] = self._get_s3_input(
                "\n Enter the S3 URL for the strategies JSON (or type 'exit' to exit): ",
                aws,
                expected_extension=".json"
            )

        elif hyper_source == '2':
            print("\n [MANUAL HYPERPARAMETERS CONFIGURATION]")
            config_data['custom_hyperparams'] = []
            iterations = 1 if config_data['strategy'] == "homogeneous" else config_data['workers']

            is_classification = "classification" in dataset_info.get('task_type', '').lower()

            for w in range(iterations):
                if config_data['strategy'] == "heterogeneous":
                    print(f"\n --- Configuring Worker {w + 1}/{config_data['workers']} ---")
                else:
                    print("\n --- Configuring Global Parameters ---")

                # 1. Max Depth
                while True:
                    raw_depth = input(" Max Depth [int >= 1, or blank for None]: ").strip()
                    if not raw_depth or raw_depth.lower() == "none":
                        max_depth = None
                        break
                    if raw_depth.isdigit() and int(raw_depth) >= 1:
                        max_depth = int(raw_depth)
                        break
                    print(" [ERROR] Max Depth must be a positive integer (>= 1) or left blank.")

                # 2. Min Samples Split
                while True:
                    raw_split = input(" Min Samples Split [int >= 2, or float (0.0-1.0], default: 2]: ").strip()
                    if not raw_split:
                        min_samples_split = 2
                        break
                    try:
                        val = float(raw_split)
                        if '.' in raw_split and 0.0 < val <= 1.0:  # It's a percentage (float)
                            min_samples_split = val
                            break
                        elif val.is_integer() and int(val) >= 2:  # It's an integer number of samples
                            min_samples_split = int(val)
                            break
                        else:
                            print(" [ERROR] Must be an integer >= 2, or a float between 0.0 and 1.0.")
                    except ValueError:
                        print(" [ERROR] Invalid number format.")

                # 3. Min Samples Leaf
                while True:
                    raw_leaf = input(" Min Samples Leaf [int >= 1, or float (0.0-1.0), default: 1]: ").strip()
                    if not raw_leaf:
                        min_samples_leaf = 1
                        break
                    try:
                        val = float(raw_leaf)
                        if '.' in raw_leaf and 0.0 < val < 1.0:  # Percentage (float)
                            min_samples_leaf = val
                            break
                        elif val.is_integer() and int(val) >= 1:  # Integer
                            min_samples_leaf = int(val)
                            break
                        else:
                            print(" [ERROR] Must be an integer >= 1, or a float between 0.0 and 1.0.")
                    except ValueError:
                        print(" [ERROR] Invalid number format.")

                # 4. Max Features
                while True:
                    raw_features = input(
                        " Max Features ['sqrt', 'log2', float <= 1.0, blank for None. Default: sqrt]: ").strip().lower()
                    if not raw_features or raw_features == "sqrt":
                        max_features = "sqrt"
                        break
                    if raw_features == "log2":
                        max_features = "log2"
                        break
                    if raw_features == "none":
                        max_features = None
                        break
                    try:
                        val = float(raw_features)
                        if 0.0 < val <= 1.0:
                            max_features = val
                            break
                        else:
                            print(" [ERROR] Float must be strictly > 0.0 and <= 1.0")
                    except ValueError:
                        print(" [ERROR] Invalid input. Use 'sqrt', 'log2', 'None', or a valid float.")

                # 5. Max Samples (for bootstrap)
                while True:
                    raw_samples = input(
                        " Max Samples per Tree [float (0.0-1.0], blank for None. Default: None]: ").strip().lower()
                    if not raw_samples or raw_samples == "none":
                        max_samples = None
                        break
                    try:
                        val = float(raw_samples)
                        if 0.0 < val <= 1.0:
                            max_samples = val
                            break
                        else:
                            print(" [ERROR] Float must be strictly > 0.0 and <= 1.0")
                    except ValueError:
                        print(" [ERROR] Invalid number format.")

                # 6. Criterion (depends on the ML task type)
                valid_criteria = ['gini', 'entropy', 'log_loss'] if is_classification else ['squared_error',
                                                                                            'absolute_error',
                                                                                            'friedman_mse',
                                                                                            'poisson']
                default_criterion = 'gini' if is_classification else 'squared_error'

                while True:
                    raw_criterion = input(
                        f" Criterion {valid_criteria} [Default: {default_criterion}]: ").strip().lower()
                    if not raw_criterion:
                        criterion = default_criterion
                        break
                    if raw_criterion in valid_criteria:
                        criterion = raw_criterion
                        break
                    print(f" [ERROR] Invalid criterion. You must choose from: {valid_criteria}")

                # 7. Class Weight (Classification only)
                class_weight = None
                if is_classification:
                    while True:
                        raw_cw = input(
                            " Class Weight ['balanced', 'balanced_subsample', blank for None]: ").strip().lower()
                        if not raw_cw or raw_cw == "none":
                            class_weight = None
                            break
                        if raw_cw in ["balanced", "balanced_subsample"]:
                            class_weight = raw_cw
                            break
                        print(" [ERROR] Invalid choice. Select 'balanced', 'balanced_subsample', or leave blank.")

                # Assemble the dictionary for the current worker
                worker_params = {
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                    "max_samples": max_samples,
                    "criterion": criterion,
                    "class_weight": class_weight,
                    "n_jobs": -1
                }
                config_data['custom_hyperparams'].append(worker_params)

            # If the strategy is homogeneous, duplicate the configuration for all workers
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
    def prompt_realtime_input(self, aws_manager, dataset_url, dataset_info):
        feature_names = None
        if dataset_url:
            bucket, key = aws_manager.parse_s3_uri(dataset_url)
            feature_names = aws_manager.get_csv_headers_from_s3(bucket, key,
                                                                  target_column_to_remove=dataset_info.get('target_col', ''))

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