from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import uvicorn

# --- CONFIG ---
id2label = {
    0: 'account',
    1: 'billing',
    2: 'other',
    3: 'technical'
}

model_save_path = "/content/drive/MyDrive/distilbert_ticket_classifier_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- APP INITIALIZATION ---
app = FastAPI(title="Customer Support Ticket Classifier API")

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL LOADING ---
from pathlib import Path

# Candidate locations to look for the trained model (order matters)
repo_root = Path(__file__).resolve().parents[1]
candidates = [
    Path(model_save_path),
    # common notebook/Colab layout used in this repo
    repo_root / 'distilbert_ticket_classifier_model',
    repo_root / 'artifacts' / 'models' / 'distilbert_ticket_classifier_model',
    repo_root / 'artifacts' / 'models' / 'distilbert' / 'final',
    repo_root / 'artifacts' / 'models' / 'distilbert' / 'distilbert_ticket_classifier_model',
]

loaded_model = None
loaded_tokenizer = None

last_err = None
for cand in candidates:
    if cand and cand.exists() and cand.is_dir():
        try:
            print(f"Attempting to load model from: {cand}")
            loaded_tokenizer = DistilBertTokenizerFast.from_pretrained(str(cand), local_files_only=True)
            loaded_model = DistilBertForSequenceClassification.from_pretrained(str(cand), local_files_only=True)
            loaded_model.to(device)
            loaded_model.eval()
            print(f"✓ Loaded tokenizer and model from: {cand}")
            break
        except Exception as e:
            last_err = e
            print(f"❌ Failed loading from {cand}: {e}")
            # try next candidate

if loaded_model is None:
    # Provide helpful diagnostics
    cand_list = '\n'.join([str(p) for p in candidates])
    msg = (
        "Could not find or load a local trained model. Tried the following paths:\n"
        f"{cand_list}\n\n"
        "Make sure the trained model folder (containing tokenizer.json, tokenizer_config.json, config.json and pytorch_model.bin/model.safetensors)\n"
        "is present in one of the above locations, or update `model_save_path` in this file to point to your model directory.\n"
    )
    if last_err is not None:
        msg += f"Last loading error: {last_err}\n"
    print(msg)
    raise FileNotFoundError(msg)

# --- DATA MODEL ---
class TextInput(BaseModel):
    text: str

# --- ROUTES ---
@app.get("/")
def root():
    return {"message": "Ticket Classifier API is running ✅"}

@app.post("/predict")
async def classify_ticket(input_text: TextInput):
    """
    Classifies a single ticket text and returns the predicted label, confidence, 
    and probabilities for all categories.
    """
    # Tokenize input
    enc = loaded_tokenizer(
        input_text.text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # Predict
    with torch.no_grad():
        outputs = loaded_model(**enc)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

        predicted_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_id].item()

        # Probabilities for all categories
        confidences_dict = {id2label[i]: float(probs[0][i]) for i in range(len(id2label))}

    # Map ID to label
    predicted_label = id2label.get(predicted_id, f"Unknown ({predicted_id})")

    return {
        "predicted_category": predicted_label,
        "confidence": round(confidence * 100, 2),
        "confidences": confidences_dict
    }


# --- MAIN ENTRY ---
if __name__ == "__main__":
    uvicorn.run("app1:app", host="0.0.0.0", port=8000, reload=False)