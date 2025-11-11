import os
import json
import joblib


def test_baseline_model_files_exist():
    """Ensure the saved baseline model and metadata exist in artifacts directory."""
    model_dir = os.path.join("artifacts", "models", "baseline")
    pipeline_path = os.path.join(model_dir, "baseline_pipeline.joblib")
    meta_path = os.path.join(model_dir, "metadata.json")

    assert os.path.isdir(model_dir), f"Model directory not found: {model_dir}"
    assert os.path.isfile(pipeline_path), f"Saved pipeline not found: {pipeline_path}"
    assert os.path.isfile(meta_path), f"Metadata file not found: {meta_path}"


def test_load_pipeline_and_predict():
    """Load the pipeline and run a tiny inference to verify it works end-to-end."""
    model_dir = os.path.join("artifacts", "models", "baseline")
    pipeline_path = os.path.join(model_dir, "baseline_pipeline.joblib")
    meta_path = os.path.join(model_dir, "metadata.json")

    # Load pipeline
    pipeline = joblib.load(pipeline_path)
    assert hasattr(pipeline, "predict"), "Loaded object does not expose a predict() method"

    # Load metadata and check format
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Basic checks on metadata
    if "classes" in meta:
        assert isinstance(meta["classes"], list), "metadata['classes'] should be a list"

    # Prepare a small sample input. Most baseline pipelines expect raw text inputs (list-like).
    sample_input = ["Unable to access my account, please help."]

    # Run prediction and verify output shape
    preds = pipeline.predict(sample_input)
    # preds can be numpy array or list-like
    assert len(preds) == len(sample_input), "Prediction length does not match input length"
