
# Customer Support Ticket Classifier

End-to-end MLOps project: classify support tickets into **Billing, Technical, Account, Other**.
Includes training (DistilBERT), API (FastAPI), Docker deploy, CI/CD, and monitoring.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pip install --upgrade pip
```

### Week 1 — Dataset Exploration
- Put the dataset at `data/customer_support_tickets_clean_500.csv`
- Open `explore_dataset.ipynb` and run all cells

### Project Structure
```
src/            # training/inference code
api/            # FastAPI app
tests/          # unit tests
monitoring/     # Prometheus/Grafana
.github/workflows/  # CI/CD
data/           # datasets (gitignored by default)
reports/        # team findings
```

### Groups workflow
- Each group works on its **own branch** (e.g., `group-1`, `group-2`, ...).
- Feature branches merge into the group branch via PRs.
- Weekly, group branches may be merged to `main` for milestone comparisons.

_Last updated: 2025-10-03 15:28 UTC_
