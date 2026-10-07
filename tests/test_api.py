from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "ResearchRAG API is running."


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_ask():
    response = client.post(
        "/ask",
        json={
            "question": "What is multi-head attention?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "citations" in data
    assert "sources" in data

    assert isinstance(data["answer"], str)
    assert isinstance(data["citations"], list)
    assert isinstance(data["sources"], list)


def test_empty_question():
    response = client.post(
        "/ask",
        json={
            "question": ""
        }
    )

    assert response.status_code == 422