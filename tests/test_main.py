from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@patch("src.app.main.client")
def test_models_endpoint(mock_client):
    mock_models = {"models": [{"name": "llama2"}, {"name": "mistral"}]}
    mock_client.list.return_value = mock_models

    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == mock_models
    mock_client.list.assert_called_once()


@patch("src.app.main.client")
def test_pull_endpoint(mock_client):
    mock_response = {"status": "success"}
    mock_client.pull.return_value = mock_response

    response = client.get("/pull?model=llama2")
    assert response.status_code == 200
    assert response.json() == mock_response
    mock_client.pull.assert_called_once_with(model="llama2")


@patch("src.app.main.client")
def test_ask_endpoint_success(mock_client):
    mock_response = {
        "response": "The sky appears blue due to Rayleigh scattering."
    }
    mock_client.generate.return_value = mock_response

    response = client.get(
        "/ask?query=Why%20is%20the%20sky%20blue?&model=llama2"
    )
    assert response.status_code == 200
    assert response.json() == {
        "answer": "The sky appears blue due to Rayleigh scattering."
    }
    mock_client.generate.assert_called_once_with(
        model="llama2", prompt="Why is the sky blue?"
    )


@patch("src.app.main.client")
def test_ask_endpoint_download_failure(mock_client):
    # First call to generate raises ResponseError
    mock_client.generate.side_effect = Exception("ResponseError")
    # Download fails
    mock_client.pull.side_effect = Exception("Failed to download")

    with patch("src.app.main.ResponseError", Exception):
        with pytest.raises(Exception):
            client.get(
                "/ask?query=Why%20is%20the%20sky%20blue?&model=unknown_model"
            )

    mock_client.pull.assert_called_once_with(model="unknown_model")
