
# 🧠 Customer Support Ticket Classifier

End-to-end MLOps project to classify customer support tickets into **Billing**, **Technical**, **Account**, or **Other**.
This project covers data exploration, feature engineering, model training (DistilBERT), API deployment with FastAPI, containerization (Docker), CI/CD automation, and monitoring.

---

## 🚀 Quickstart

```bash
# Create and activate virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python -m pip install --upgrade pip
```

---

## 📊 Week 1 — Dataset Exploration

1. Place the dataset at:

   ```
   data/customer_support_tickets_clean_500.csv
   ```
2. Open your assigned notebook inside the `notebooks/` folder.
3. Follow the task assigned by your **group leader** (e.g., EDA, feature engineering, model training).
4. Commit and push your notebook to your group branch (see instructions below).

---

## 🧩 Project Structure

```
src/                # training and inference code
api/                # FastAPI app
tests/              # unit tests
monitoring/         # Prometheus / Grafana setup
.github/workflows/  # CI/CD workflows
data/               # datasets (gitignored)
notebooks/          # Jupyter notebooks (group work area)
reports/            # team findings and weekly summaries
```

---

## 👥 Groups Workflow

Each intern group has its own **branch** (e.g., `group-a`, `group-b`, `group-c`, `group-d`).
All work happens inside your group’s branch — not `main`.

### 🪜 Steps to Start

1. **Clone** the repository:

   ```bash
   git clone https://github.com/<your-org>/customer-support-ticket-classifier.git
   cd customer-support-ticket-classifier
   ```
2. **Checkout** your group branch:

   ```bash
   git checkout group-a
   ```
3. **Create your feature branch** (optional for your own notebook):

   ```bash
   git checkout -b group-a-01-eda
   ```
4. **Work on your assigned notebook** in the `notebooks/` folder (e.g., `01-eda-group-a.ipynb`).
5. **Push changes** and open a pull request into your group branch.

---

## 📘 Notebook Naming Convention

Follow this naming pattern for all notebooks:

```
<two-digit-order>-<task>-group-<letter>.ipynb
```

✅ **Examples**

* `01-eda-group-a.ipynb`
* `02-feature-eng-group-a.ipynb`
* `03-model-training-group-a.ipynb`

Each notebook is owned by one intern to avoid merge conflicts.
If you need to make changes to another person’s notebook, create a copy (e.g., `01-eda-group-a-v2.ipynb`).

---

## 🧹 Important Notes

* Do not push directly to `main`.
* Always push work to your **group branch** or feature branch derived from it.
* Clean outputs before committing (restart kernel → run all cells).
* Each week, **group leaders** will review and merge all notebooks into the group branch.

---

*Last updated: 2025-10-10 17:52 UTC*

---

Would you like me to turn this updated README into a properly formatted **Markdown file** (`README.md`) you can directly upload to GitHub?
