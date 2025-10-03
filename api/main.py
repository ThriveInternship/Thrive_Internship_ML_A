
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI(title="Ticket Classifier API")
MODEL_DIR = "models/distilbert-ticket-classifier"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

class Ticket(BaseModel):
    text: str

@app.post("/predict")
def predict(ticket: Ticket):
    inputs = tok([ticket.text], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = mdl(**inputs).logits
    idx = int(torch.argmax(logits, dim=-1)[0])
    label = mdl.config.id2label.get(idx, str(idx))
    return {"label": label}
