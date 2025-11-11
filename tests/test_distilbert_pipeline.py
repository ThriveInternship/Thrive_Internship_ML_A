from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1️⃣ Load tokenizer + model (pretrained DistilBERT fine-tuned on text classification)
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"  # sentiment classifier
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# 2️⃣ Sample customer-support tickets
texts = [
    "I can’t log into my account. Please help!",          # Account issue
    "I was double charged on my last bill.",              # Billing issue
    "The app keeps crashing whenever I open it.",         # Technical issue
    "Thank you for your help, everything is fixed now."   # Other
]

# 3️⃣ Tokenize
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 4️⃣ Run inference (no gradients)
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)
    labels = torch.argmax(predictions, dim=1)

# 5️⃣ Map model labels to human-readable names (the SST-2 model has 0=NEG, 1=POS)
label_map = {0: "Negative", 1: "Positive"}

# 6️⃣ Print results
for text, label in zip(texts, labels):
    print(f"\nText: {text}\nPredicted sentiment: {label_map[label.item()]}")
