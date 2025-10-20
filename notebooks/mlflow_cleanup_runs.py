# MLflow run cleanup utility
# This cell deletes all but the N most recent run folders in your experiment directory to save disk space.
# Adjust 'keep_last_n' as needed.

import os
import shutil
from pathlib import Path

# Set your experiment directory (update if needed)
mlruns_dir = os.path.join(os.getcwd(), 'mlruns')
experiment_id = None
experiment_name = "baseline_models"

# Find experiment_id for the given experiment_name
def find_experiment_id(mlruns_dir, experiment_name):
    for d in os.listdir(mlruns_dir):
        if d.isdigit():
            meta_path = os.path.join(mlruns_dir, d, 'meta.yaml')
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    if f"name: {experiment_name}" in f.read():
                        return d
    return None

experiment_id = find_experiment_id(mlruns_dir, experiment_name)
if experiment_id is None:
    print(f"Experiment '{experiment_name}' not found in {mlruns_dir}.")
else:
    exp_path = os.path.join(mlruns_dir, experiment_id)
    run_dirs = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d)) and len(d) == 32]
    # Sort by last modified time (most recent last)
    run_dirs = sorted(run_dirs, key=lambda d: os.path.getmtime(os.path.join(exp_path, d)))
    keep_last_n = 3  # Change this to keep more or fewer runs
    to_delete = run_dirs[:-keep_last_n] if len(run_dirs) > keep_last_n else []
    for d in to_delete:
        run_path = os.path.join(exp_path, d)
        print(f"Deleting run folder: {run_path}")
        shutil.rmtree(run_path)
    if not to_delete:
        print(f"Nothing to delete. {len(run_dirs)} run folders found, keeping all.")
    else:
        print(f"Deleted {len(to_delete)} old run folders. Kept {keep_last_n} most recent runs.")
