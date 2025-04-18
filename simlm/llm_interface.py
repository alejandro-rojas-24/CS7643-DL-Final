import json
import re
import time
import os

import ollama
from google import genai
from google.genai.types import GenerateContentConfig
from openai import APIError, OpenAI, RateLimitError

from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Warning: Failed to initialize OpenAI client: {e}")
    openai_client = None

try:
    google_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Warning: Failed to initialize Google Gemini client: {e}")
    google_client = None

try:
    ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
except Exception as e:
    print(f"Warning: Failed to initialize Ollama client: {e}")
    ollama_client = None


def call_openai_api(
    model_name,
    prompt,
    max_retries=3,
    delay=5,
    temperature=0.7,
):
    """Calls the OpenAI Chat Completions API."""
    if not openai_client:
        print("Error: OpenAI client not initialized.")
        return None

    messages = [
        {"role": "system", "content": "Respond only with valid JSON object."},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            return content
        except RateLimitError:
            print(
                f"OpenAI Rate limit exceeded. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
        except APIError as e:
            print(
                f"OpenAI API Error: {e}. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
        except Exception as e:
            print(f"An unexpected error occurred with OpenAI: {e}")
            return None  # Non-retriable error
    print("Error: Max retries reached for OpenAI API call.")
    return None


def call_ollama_api(
    model_name,
    prompt,
    max_retries=3,
    delay=5,
    temperature=0.7,
):
    """Calls a local Ollama API using the ollama Python library."""
    if ollama_client is None:
        print("Error: Ollama server not available.")
        return None

    for attempt in range(max_retries):
        try:
            # Use the ollama client to generate a response
            response = ollama_client.generate(
                model=model_name,
                prompt=prompt,
                options={"temperature": temperature},
                format="json",  # Request JSON output
            )

            # Extract the response content
            content = response.get("response")
            return content
        except ollama.ResponseError as e:
            print(
                f"Ollama request failed: {e}. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
        except Exception as e:
            print(f"An unexpected error occurred with Ollama: {e}")
            return None  # Non-retriable error
    print("Error: Max retries reached for Ollama API call.")
    return None


def call_google_api(
    model_name,
    prompt,
    max_retries=3,
    delay=5,
    temperature=0.7,
):
    """Calls the Google Gemini API."""
    if google_client is None:
        print("Error: Google is not configured.")
        return None
    for attempt in range(max_retries):
        try:
            response = google_client.models.generate_content(
                model=model_name.removeprefix("google/").removeprefix(
                    "models/"
                ),
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=temperature,
                ),
            )
            print(response.text)
            return response.text
        except Exception as e:
            print(f"An unexpected error occurred with Ollama: {e}")
            return None
        time.sleep(delay)
    return None


def get_llm_response(model_identifier, prompt, temperature=0.7):
    """
    Gets a response from the specified LLM.

    Args:
        model_identifier (str): Name of the model (e.g., "gpt-3.5-turbo", "ollama/llama3").
        prompt (str): The input prompt for the LLM.
        temperature (float): Sampling temperature.

    Returns:
        str or None: The LLM's response text, or None if an error occurred.
    """
    if model_identifier.startswith("ollama/"):
        ollama_model_name = model_identifier.split("/", 1)[1]
        return call_ollama_api(
            ollama_model_name, prompt, temperature=temperature
        )
    if model_identifier.startswith("google/"):
        ollama_model_name = model_identifier.split("/", 1)[1]
        return call_google_api(
            ollama_model_name, prompt, temperature=temperature
        )
    elif openai_client:
        return call_openai_api(
            model_identifier, prompt, temperature=temperature
        )
    else:
        print(
            f"Error: Model provider for '{model_identifier}' not configured or available."
        )
        return None


def parse_llm_json_output(response_text):
    """
    Parses the expected JSON output from the LLM.

    Args:
        response_text (str): The raw text response from the LLM.

    Returns:
        dict or None: A dictionary containing 'height', 'horizontal_velocity',
                      and optionally 'reasoning' or 'critique', or None if parsing fails.
    """
    if not response_text:
        return None

    # pull out relevant JSON object from the response text
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not match:
        print(f"Error: No JSON object found in response: {response_text}")
        return None
    response_text = match.group(0).strip()

    try:
        data = json.loads(response_text)

        # Basic validation
        if not isinstance(data, dict):
            print(f"Error: Parsed JSON is not a dictionary: {data}")
            return None

        # Check for required keys - be flexible as reasoning/critique might be missing
        required_numeric_keys = ["height", "horizontal_velocity"]
        parsed_data = {}

        for key in required_numeric_keys:
            if key not in data:
                print(
                    f"Error: Missing required key '{key}' in LLM JSON output: {data}"
                )
                return None
            try:
                # Convert to float, handle potential non-numeric values
                parsed_data[key] = float(data[key])
            except (ValueError, TypeError):
                print(
                    f"Error: Could not convert '{key}' value '{data[key]}' to float."
                )
                return None

        # Add optional keys if present
        if "reasoning" in data and isinstance(data["reasoning"], str):
            parsed_data["reasoning"] = data["reasoning"]
        if "critique" in data and isinstance(data["critique"], str):
            parsed_data["critique"] = data["critique"]

        # Parameter sanity checks
        if not (
            0 < parsed_data["height"] < 1000
        ):  # e.g., height between 0 and 1km
            print(
                f"Warning: Parsed height {parsed_data['height']} seems unrealistic."
            )
        if not (
            0 < parsed_data["horizontal_velocity"] < 1000
        ):  # e.g., velocity < 1km/s
            print(
                f"Warning: Parsed velocity {parsed_data['horizontal_velocity']} seems unrealistic."
            )

        return parsed_data

    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse LLM response as JSON: {e}")
        print(f"Raw response was: {response_text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during JSON parsing: {e}")
        return None
