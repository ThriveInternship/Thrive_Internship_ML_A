
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict():
    resp = client.post("/predict", json={"text":"Payment failed"})
    assert resp.status_code == 200
    assert "label" in resp.json()
