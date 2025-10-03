
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_pipeline(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tok, mdl

def predict(texts, tok, mdl):
    inputs = tok(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = mdl(**inputs).logits
    ids = torch.argmax(logits, dim=-1).tolist()
    id2label = {int(k):v for k,v in mdl.config.id2label.items()} if mdl.config.id2label else {0:"Billing",1:"Technical",2:"Account",3:"Other"}
    return [id2label[i] for i in ids]

if __name__ == "__main__":
    tok, mdl = load_pipeline("models/distilbert-ticket-classifier")
    print(predict(["Payment failed at checkout"], tok, mdl))
