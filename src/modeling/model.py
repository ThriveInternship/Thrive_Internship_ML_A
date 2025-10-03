
from transformers import AutoModelForSequenceClassification

def load_model(model_name_or_path: str, num_labels: int = 4):
    return AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
