import io
import sys
import time
import os
import gc
import json

import boto3
import botocore
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, accuracy_score, mean_squared_error, r2_score, mean_absolute_error,
    f1_score, precision_score, recall_score, average_precision_score, mean_absolute_percentage_error
)

# ---------------------------------------------------------
# 1. SETUP AND CREDENTIALS (ENVIRONMENT VARIABLES)
# ---------------------------------------------------------
# Retrieve AWS Region from environment variable (defaults to eu-central-1 if missing)
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")

# Retrieve Target Bucket from environment variable
TARGET_BUCKET = os.getenv("TARGET_BUCKET")

# Safety check: ensure the target bucket is provided before proceeding
if not TARGET_BUCKET:
    print("\n [CRITICAL] TARGET_BUCKET environment variable is not set!")
    print(" Please export it in your terminal before running the script.")
    print(" Example: export TARGET_BUCKET=\"your-s3-bucket\"\n")
    sys.exit(1)

s3_client = boto3.client('s3', region_name=AWS_REGION)

# Define target datasets and tree grid
TARGET_DATASETS = ["taxi", "synthetic"]
TREES_GRID = [25, 50, 75, 100]

# HARDCODED DATASET METADATA
# Modify S3 paths if your files are located in different folders
DATASETS_METADATA = {
    "taxi": {
        "task_type": "regression",
        "target_col": "Label",
        "train_path": "data/taxi/taxi_1M_train.csv",
        "test_path": "data/taxi/taxi_1M_test.csv"
    },
    "synthetic": {
        "task_type": "classification",
        "target_col": "Label",
        "train_path": "data/synthetic/synthetic_1M_train.csv",
        "test_path": "data/synthetic/synthetic_1M_test.csv"
    }
}


# ---------------------------------------------------------
# 2. S3 UTILITY FUNCTIONS
# ---------------------------------------------------------
def load_json_config_from_s3(bucket, dataset_name):
    """
    Downloads the JSON file containing the optimal configurations for the specific dataset.
    """
    s3_key = f"configs/{dataset_name}_homogeneous_conf.json"

    print(f" [CONFIG] Fetching hyperparams from s3://{bucket}/{s3_key}...")
    try:
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        config_dict = json.loads(response['Body'].read().decode('utf-8'))
        return config_dict
    except Exception as e:
        print(f" [CRITICAL] Failed to load JSON config for {dataset_name}: {e}")
        return None


def load_dataset_from_s3(bucket, key):
    """
    Downloads the dataset from S3 directly into memory (RAM) as a Pandas DataFrame.
    """
    print(f" [DOWNLOAD] Fetching s3://{bucket}/{key} into RAM...")
    start_dl = time.time()
    response = s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(response['Body'].read()))
    print(f" [DOWNLOAD] Completed in {time.time() - start_dl:.2f}s. Loaded {len(df)} rows.")
    return df


def save_unified_baseline_metrics(dataset, n_trees, train_time, inf_time, metrics_dict):
    """
    Saves all metrics in a single global CSV file.
    Metrics not relevant to the current task will remain empty to keep the format clean.
    """
    s3_key = "results/unified_baseline_results.csv"

    # Prepare an "empty" row with all possible columns
    row_data = {
        'Dataset': dataset,
        'Trees': n_trees,
        'Train_Time': round(train_time, 2),
        'Infer_Time': round(inf_time, 2),
        'ROC-AUC': None, 'PR-AUC': None, 'F1-Score': None, 'Precision': None, 'Recall': None, 'Accuracy': None,
        'RMSE': None, 'MAE': None, 'MAPE': None, 'R2 Score': None
    }

    # Update "None" values with the metrics calculated in this run
    row_data.update(metrics_dict)
    new_row_df = pd.DataFrame([row_data])

    try:
        obj = s3_client.get_object(Bucket=TARGET_BUCKET, Key=s3_key)
        df_existing = pd.read_csv(io.BytesIO(obj['Body'].read()), sep=';', decimal=',')
        df_final = pd.concat([df_existing, new_row_df], ignore_index=True)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            # If the file does not exist, create it from scratch
            df_final = new_row_df
        else:
            print(f" [METRICS ERROR] Unexpected S3 error: {e}")
            return

    csv_buffer = io.StringIO()
    # na_rep='' prevents writing "NaN" as text in the excel/csv file
    df_final.to_csv(csv_buffer, index=False, sep=';', decimal=',', na_rep='')

    s3_client.put_object(Bucket=TARGET_BUCKET, Key=s3_key, Body=csv_buffer.getvalue())
    print(f" [METRICS] Baseline results securely appended to: s3://{TARGET_BUCKET}/{s3_key}")


# ---------------------------------------------------------
# 3. MAIN BENCHMARK LOOP
# ---------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print(" PURE NATIVE BASELINE BENCHMARK (SCIKIT-LEARN)")
    print("=" * 60)
    print(f" Target Datasets: {TARGET_DATASETS}")
    print(f" Target Bucket  : {TARGET_BUCKET}")
    print(f" Trees Grid     : {TREES_GRID}")
    print("=" * 60 + "\n")

    for dataset in TARGET_DATASETS:
        print("\n" + "*" * 50)
        print(f" STARTING DATASET ELABORATION: {dataset.upper()}")
        print("*" * 50)

        # A. Retrieve hyperparameters
        best_params_dict = load_json_config_from_s3(TARGET_BUCKET, dataset)
        if not best_params_dict:
            print(f" [CRITICAL] Skipping {dataset} due to missing config file.")
            continue

        dataset_meta = DATASETS_METADATA.get(dataset)
        if not dataset_meta:
            print(f" [CRITICAL] Missing metadata for {dataset}. Skipping.")
            continue

        task_type = dataset_meta['task_type']
        target_col = dataset_meta['target_col']
        train_s3_key = dataset_meta['train_path']
        test_s3_key = dataset_meta['test_path']

        # B. Load Dataset and split X, y
        try:
            print(" [RAM ALLOCATION] Loading Train/Test sets and splitting features...")
            df_train = load_dataset_from_s3(TARGET_BUCKET, train_s3_key)
            X_train = df_train.drop(columns=[target_col])
            y_train = df_train[target_col]

            df_test = load_dataset_from_s3(TARGET_BUCKET, test_s3_key)
            X_test = df_test.drop(columns=[target_col])
            y_true = df_test[target_col].values

            # Delete old dataframes to free up precious RAM
            del df_train
            del df_test
            gc.collect()

        except Exception as e:
            print(f" [CRITICAL] Memory or S3 Error during dataset loading: {e}")
            continue

        # C. Training and Inference
        for trees in TREES_GRID:
            print(f"\n --- STARTING BENCHMARK FOR {trees} TREES ---")

            try:
                params = best_params_dict[str(trees)].copy()
            except KeyError:
                print(f" [WARNING] No parameter for {trees} trees in JSON. Skipping.")
                continue

            # Clean up extra parameters (if present in json) to avoid sklearn errors
            if 'trees' in params:
                del params['trees']

            params["n_estimators"] = trees
            params["random_state"] = 42
            params["n_jobs"] = -1

            # Initialize NATIVE model
            if task_type == 'classification':
                model = RandomForestClassifier(**params)
            else:
                model = RandomForestRegressor(**params)

            print(f" [BASELINE TRAIN] Training native model with params: {params}")
            train_start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - train_start
            print(f" [BASELINE TRAIN] Completed in {train_time:.2f}s")

            # NATIVE INFERENCE (C-Optimized)
            print(f" [BASELINE INFER] Executing prediction on {len(X_test)} rows in a single pass...")
            infer_start = time.time()

            if task_type == 'classification':
                # Scikit-learn instantly calculates probabilities and hard predictions
                y_prob = model.predict_proba(X_test)[:, 1]
                final_prediction = model.predict(X_test)

                infer_time = time.time() - infer_start
                print(f" [BASELINE INFER] Completed in {infer_time:.2f}s")

                auc = roc_auc_score(y_true, y_prob)
                acc = accuracy_score(y_true, final_prediction)
                f1 = f1_score(y_true, final_prediction, zero_division=0)
                prec = precision_score(y_true, final_prediction, zero_division=0)
                rec = recall_score(y_true, final_prediction, zero_division=0)
                pr_auc = average_precision_score(y_true, y_prob)

                print(f" [EVALUATION] ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | Acc: {acc:.4f}")

                metrics_dict = {
                    'ROC-AUC': round(auc, 4), 'PR-AUC': round(pr_auc, 4),
                    'F1-Score': round(f1, 4), 'Precision': round(prec, 4),
                    'Recall': round(rec, 4), 'Accuracy': round(acc, 4)
                }

            else:
                predictions = model.predict(X_test)

                infer_time = time.time() - infer_start
                print(f" [BASELINE INFER] Completed in {infer_time:.2f}s")

                mse = mean_squared_error(y_true, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, predictions)
                mae = mean_absolute_error(y_true, predictions)
                mape = mean_absolute_percentage_error(y_true, predictions)

                print(f" [EVALUATION] RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2 Score: {r2:.4f}")

                metrics_dict = {
                    'RMSE': round(rmse, 4), 'MAE': round(mae, 4),
                    'MAPE': round(mape, 4), 'R2 Score': round(r2, 4)
                }

            # D. Save to unified CSV
            save_unified_baseline_metrics(dataset, trees, train_time, infer_time, metrics_dict)

        # Total RAM cleanup at the end of the dataset
        print(f"\n [CLEANUP] Releasing memory for {dataset.upper()}...")
        del X_train, y_train, X_test, y_true
        gc.collect()

    print("\n [SUCCESS] All unified baselines completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n [SYSTEM] Baseline terminated by user.")
        sys.exit(0)
