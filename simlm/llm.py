import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Literal

import ollama
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, GenerateContentResponse
from ollama import GenerateResponse
from openai import APIError, OpenAI, RateLimitError
from openai.types.responses.response import Response
from pydantic import BaseModel
import functools
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

logger = logging.getLogger(__name__)

load_dotenv()


def log_llm_response(func):
    """Decorator to log LLM requests and responses."""

    @functools.wraps(func)
    def wrapper(self, prompt: str, temperature: float = 0.0, *args, **kwargs):
        client_name = self.__class__.__name__
        logger.debug(
            f"{client_name} request: model={self.model_name}, temperature={temperature}, prompt={prompt}"
        )

        response = func(self, prompt, temperature, *args, **kwargs)

        logger.info(
            f"{client_name} model:{self.model_name}, response: {response}"
        )

        return response

    return wrapper


class LLMResponse(BaseModel):
    """Pydantic model representing a parsed response from an LLM."""

    height: float
    horizontal_velocity: float
    reasoning: str | None
    critique: str | None

    def validate(self) -> None:
        if self.reasoning is None and self.critique is None:
            logger.warning(
                "Response does not contain either reasoning or critique"
            )
        if self.reasoning is not None and self.critique is not None:
            logger.warning("Response contains both reasoning and critique")


class LLMClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(
        self,
        model_name: str,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        """
        Initialize the LLM client.

        Args:
            model_name: Name of the model to use
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay in seconds between retries
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # factory method to create an appropriate LLM client based on the
    # model identifier
    @classmethod
    def from_model_service(
        cls,
        model_service: Literal["ollama", "google", "openai"],
        model_name: str,
        **kwargs,
    ) -> "LLMClient":
        """
        Create an appropriate LLM client based on the model identifier.

        Args:
            model_identifier: Name of the model (e.g., "gpt-3.5-turbo", "ollama/llama3")
            **kwargs: Additional arguments to pass to the client constructor

        Returns:
            An initialized LLM client or None if the model provider is not available
        """
        if model_service == "ollama":
            return OllamaClient(model_name=model_name, **kwargs)
        elif model_service == "google":
            return GoogleClient(model_name=model_name, **kwargs)
        elif model_service == "openai":
            return OpenAIClient(model_name=model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported model service: {model_service}")

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.0,
    ) -> str | None:
        """Generate a response from the LLM."""
        pass

    def generate_and_parse(
        self,
        prompt: str,
        temperature: float = 0.0,
    ) -> LLMResponse | None:
        """
        Generate a response and parse the JSON output.

        Args:
            prompt: The input prompt for the LLM
            temperature: Sampling temperature

        Returns:
            Parsed LLMResponse object or None if parsing fails
        """
        response_text = self.generate_response(prompt, temperature)
        if not response_text:
            return None
        return _parse_json_output(response_text)


class OpenAIClient(LLMClient):
    """Client for OpenAI's LLM API."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the OpenAI client.

        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key (will use environment variable if not provided)
        """
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=self.api_key)
        logger.debug(
            f"OpenAI client initialized with model: {self.model_name}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        retry=retry_if_exception_type((RateLimitError, APIError)),
        reraise=True,
    )
    def _call_openai_api(
        self,
        prompt: str,
        temperature: float,
    ) -> Response:
        """Make API call to OpenAI with retry logic."""
        return self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=temperature,
        )

    @log_llm_response
    def generate_response(
        self, prompt: str, temperature: float = 0.0
    ) -> str | None:
        """
        Generate a response using the OpenAI API.

        Args:
            prompt: The input prompt for the LLM
            temperature: Sampling temperature

        Returns:
            The LLM's response text, or None if an error occurred
        """

        try:
            response = self._call_openai_api(prompt, temperature)
            return response.output_text
        except Exception as e:
            logger.error(f"An error occurred with OpenAI: {e}")
            return None


class GoogleClient(LLMClient):
    """Client for Google's Gemini API."""

    def __init__(
        self, model_name: str, api_key: str | None = None, **kwargs
    ) -> None:
        """
        Initialize the Google Gemini client.

        Args:
            model_name: Name of the Gemini model to use
            api_key: Google API key (will use environment variable if not provided)
        """
        model_name = model_name.removeprefix("google/").removeprefix("models/")
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.client = genai.Client(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def _call_gemini_api(
        self, model: str, prompt: str, temperature: float
    ) -> GenerateContentResponse:
        """Make API call to Gemini with retry logic."""
        return self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=temperature,
            ),
        )

    @log_llm_response
    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.0,
    ) -> str | None:
        """
        Generate a response using the Google Gemini API.

        Args:
            prompt: The input prompt for the LLM
            temperature: Sampling temperature

        Returns:
            The LLM's response text, or None if an error occurred
        """
        try:
            response = self._call_gemini_api(
                self.model_name,
                prompt,
                temperature,
            )
            return response.text
        except Exception as e:
            logger.error(f"An error occurred with Google Gemini: {e}")
            return None


class OllamaClient(LLMClient):
    """Client for local Ollama API."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        timeout: int = 10,
        **kwargs,
    ) -> None:
        """
        Initialize the Ollama client.

        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for the Ollama API
        """
        model_name = model_name.removeprefix("ollama/")
        super().__init__(model_name, **kwargs)
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )
        self.client = ollama.Client(host=self.base_url, timeout=timeout)

    def is_model_available(self) -> bool:
        """Check if the specified model is available."""
        try:
            models_list = self.client.list()
            for model_tuple in models_list:
                for model_data in model_tuple[1]:
                    if model_data.model == self.model_name:
                        return True
            return False
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False

    def download_model(self) -> bool:
        """Download the specified model if not available."""
        try:
            if not self.is_model_available():
                logger.info(f"Downloading model {self.model_name}...")
                # Remove timeout for download which can take a few minutes
                downloader_client = ollama.Client(host=self.base_url)
                downloader_client.pull(self.model_name)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        reraise=True,
    )
    def _call_ollama_api(
        self,
        model: str,
        prompt: str,
        temperature: float,
    ) -> GenerateResponse:
        """Make API call to Ollama with retry logic."""
        return self.client.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature},
            format="json",
        )

    @log_llm_response
    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.0,
    ) -> str | None:
        """
        Generate a response using the Ollama API.

        Args:
            prompt: The input prompt for the LLM
            temperature: Sampling temperature

        Returns:
            The LLM's response text, or None if an error occurred
        """
        if not self.is_model_available():
            self.download_model()
        try:
            response = self._call_ollama_api(
                self.model_name,
                prompt,
                temperature,
            )
            return response.get("response")
        except Exception as e:
            logger.error(f"An error occurred with Ollama: {e}")
            return None


class LLMClientFactory:
    """Factory class to create LLM clients based on model identifier."""

    @staticmethod
    def create_client(model_identifier: str, **kwargs) -> LLMClient | None:
        """
        Create an appropriate LLM client based on the model identifier.

        Args:
            model_identifier: Name of the model (e.g., "gpt-3.5-turbo", "ollama/llama3")
            **kwargs: Additional arguments to pass to the client constructor

        Returns:
            An initialized LLM client or None if the model provider is not available
        """
        if model_identifier.startswith("ollama/"):
            return OllamaClient(model_name=model_identifier, **kwargs)
        elif model_identifier.startswith("google/"):
            return GoogleClient(model_name=model_identifier, **kwargs)
        else:
            return OpenAIClient(model_name=model_identifier, **kwargs)


def _parse_json_output(response_text: str) -> LLMResponse | None:
    """
    Parse the JSON output from the LLM response.

    Args:
        response_text: Raw text response from the LLM

    Returns:
        Parsed LLMResponse object or None if parsing fails
    """
    if not response_text:
        logger.error("Cannot parse empty response")
        return None

    # Extract JSON object from response text
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not match:
        logger.error(f"No JSON object found in response: {response_text}")
        return None

    json_text = match.group(0).strip()

    try:
        data = json.loads(json_text)

        if not isinstance(data, dict):
            logger.error(f"Parsed JSON is not a dictionary: {data}")
            return None

        # Create response object using Pydantic for validation
        try:
            response = LLMResponse(
                height=float(data["height"]),
                horizontal_velocity=float(data["horizontal_velocity"]),
                reasoning=data.get("reasoning"),
                critique=data.get("critique"),
            )
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Failed to create LLMResponse object: {e}")
            return None

        try:
            response.validate()
        except ValueError as e:
            logger.error(f"Validation failed: {e}")
            return None

        return response

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.debug(f"Raw response was: {response_text}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during JSON parsing: {e}")
        return None
