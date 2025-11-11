# ---------------------------------------------------------------------------
# MLflow Run Cleanup Utility (Portable Version)
# ---------------------------------------------------------------------------
# This script deletes all but the N most recent run folders in your MLflow
# experiment directory to save disk space.
# It automatically sets MLflow tracking to your current project folder.
# Adjust 'keep_last_n' as needed.
# ---------------------------------------------------------------------------

import os
import shutil
import mlflow
from pathlib import Path

# 🔧 Step 1: Dynamically set MLflow tracking URI to your current project directory
mlflow_tracking_path = f"file:///{os.getcwd().replace('\\', '/')}/mlruns"
mlflow.set_tracking_uri(mlflow_tracking_path)

print(f"✅ MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

# 🔧 Step 2: Define experiment directory and target experiment name
mlruns_dir = os.path.join(os.getcwd(), 'mlruns')
experiment_name = "baseline_models"

# 🧭 Step 3: Find experiment_id for the given experiment name
def find_experiment_id(mlruns_dir, experiment_name):
    if not os.path.exists(mlruns_dir):
        print(f"⚠️ No 'mlruns' directory found at: {mlruns_dir}")
        return None

    for d in os.listdir(mlruns_dir):
        if d.isdigit():
            meta_path = os.path.join(mlruns_dir, d, 'meta.yaml')
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if f"name: {experiment_name}" in content:
                        return d
    return None

# 🔍 Step 4: Locate and clean up old runs
experiment_id = find_experiment_id(mlruns_dir, experiment_name)

if experiment_id is None:
    print(f"⚠️ Experiment '{experiment_name}' not found in {mlruns_dir}.")
else:
    exp_path = os.path.join(mlruns_dir, experiment_id)
    run_dirs = [
        d for d in os.listdir(exp_path)
        if os.path.isdir(os.path.join(exp_path, d)) and len(d) == 32
    ]

    if not run_dirs:
        print(f"ℹ️ No run folders found in {exp_path}.")
    else:
        # Sort by modification time (most recent last)
        run_dirs = sorted(run_dirs, key=lambda d: os.path.getmtime(os.path.join(exp_path, d)))

        keep_last_n = 3  # 👈 Change this number to keep more or fewer runs
        to_delete = run_dirs[:-keep_last_n] if len(run_dirs) > keep_last_n else []

        for d in to_delete:
            run_path = os.path.join(exp_path, d)
            print(f"🗑️ Deleting old run folder: {run_path}")
            shutil.rmtree(run_path, ignore_errors=True)

        if not to_delete:
            print(f"✅ Nothing to delete. Found {len(run_dirs)} run folders, keeping all.")
        else:
            print(f"✅ Deleted {len(to_delete)} old run folders. Kept {keep_last_n} most recent runs.")

print("✨ Cleanup complete! Your MLflow environment is now portable and clean.")
