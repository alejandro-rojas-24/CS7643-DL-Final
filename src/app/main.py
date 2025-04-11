import logging
import os

from fastapi import FastAPI, Query
from .models import AskResponse, HealthResponse

import ollama
from ollama._types import ResponseError

logger = logging.getLogger(__name__)

app = FastAPI(
    title="A-Team LLM API",
    description="API for querying large language models via Ollama",
    version="0.1.0",
)

ollama_url = os.getenv("OLLAMA_API_URL", "http://ollama:11434")
client = ollama.Client(host=ollama_url)


@app.get("/health")
def health() -> HealthResponse:
    """
    Health check endpoint to verify the service is running correctly.

    Returns:
        HealthResponse: Object containing the health status of the service
    """
    logger.info("Health check endpoint called")
    return HealthResponse(status="healthy")


@app.get("/models")
def models():
    """
    List all available language models in the Ollama service.

    Returns:
        dict: Dictionary containing information about available models
    """
    response = client.list()
    logger.info(f"Models endpoint called, found models: {response}")
    # convert to ModelsResponse
    return response


@app.get("/pull")
def pull(
    model: str = Query(
        default="llama2",
        description="The model to download to ollama.",
    ),
):
    """
    Download a language model to the Ollama service.

    Args:
        model (str): Name of the model to download (default: "llama2")

    Returns:
        dict: Response from the Ollama pull operation
    """
    return client.pull(model=model)


@app.get("/ask", response_model=AskResponse)
def ask(
    query: str = Query(
        default="Why is the sky blue?",
        description="The query to ask the model",
    ),
    model: str = Query(
        default="llama2", description="The model to use for the query."
    ),
) -> AskResponse:
    """
    Query a language model with a specific prompt.

    This endpoint automatically downloads the model if it's not already available.

    Args:
        query (str): The prompt to send to the language model
        model (str): The specific model to use for generation (default: "llama2")

    Returns:
        AskResponse: Object containing the model's answer to the query

    Raises:
        Exception: If the model cannot be downloaded or another error occurs
    """
    logger.info(f"Ask endpoint called with model: {model} - Query: {query}")

    try:
        logger.info(f"Generating response using model: {model}")
        response = client.generate(model=model, prompt=query)
    except ResponseError:
        logger.warning(f"Model {model} not found, attempting to download...")
        try:
            client.pull(model=model)
            logger.info(f"Successfully downloaded model: {model}")
            # Retry generation after downloading
            response = client.generate(model=model, prompt=query)
        except Exception as download_error:
            logger.exception(
                f"Failed to download model {model}: {str(download_error)}"
            )
            raise

    logger.info("Successfully generated response")
    return AskResponse(answer=response.get("response", ""))
