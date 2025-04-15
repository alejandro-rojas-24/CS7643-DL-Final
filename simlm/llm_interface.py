import json
import requests
from openai import OpenAI, APIError, RateLimitError
import time
from config import OPENAI_API_KEY, OLLAMA_BASE_URL

# OpenAI Client
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Warning: Failed to initialize OpenAI client: {e}")
    openai_client = None

# Ollama Configuration
ollama_available = False
try:
    # Quick check if Ollama server is running
    response = requests.get(OLLAMA_BASE_URL)
    if response.status_code == 200:
        ollama_available = True
        print("Ollama server found.")
    else:
        print(f"Warning: Ollama server not responding at {OLLAMA_BASE_URL} (Status: {response.status_code})")
except requests.exceptions.ConnectionError:
    print(f"Warning: Ollama server not reachable at {OLLAMA_BASE_URL}. Connection refused.")
except Exception as e:
     print(f"Warning: Unknown error checking Ollama status: {e}")


def call_openai_api(model_name, prompt, max_retries=3, delay=5, temperature=0.7):
    """Calls the OpenAI Chat Completions API."""
    if not openai_client:
        print("Error: OpenAI client not initialized.")
        return None

    messages = [
        {"role": "system", "content": "Respond only with valid JSON object."},
        {"role": "user", "content": prompt}
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
            print(f"OpenAI Rate limit exceeded. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
            time.sleep(delay)
        except APIError as e:
            print(f"OpenAI API Error: {e}. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
            time.sleep(delay)
        except Exception as e:
            print(f"An unexpected error occurred with OpenAI: {e}")
            return None # Non-retriable error
    print("Error: Max retries reached for OpenAI API call.")
    return None

def call_ollama_api(model_name, prompt, max_retries=3, delay=5, temperature=0.7):
    """Calls a local Ollama API."""
    if not ollama_available:
         print("Error: Ollama server not available.")
         return None

    api_url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "format": "json", # Request JSON output
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    headers = {'Content-Type': 'application/json'}

    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, json=payload, headers=headers)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            # Ollama wraps the JSON in a 'response' field within its own JSON structure
            response_data = response.json()
            content = response_data.get("response")
            # print(f"Debug Ollama Raw Response: {content}") # Debugging
            return content
        except requests.exceptions.RequestException as e:
            print(f"Ollama request failed: {e}. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
            time.sleep(delay)
        except json.JSONDecodeError:
             print(f"Ollama response was not valid JSON: {response.text}")
             return None # Cannot retry if response format is wrong
        except Exception as e:
            print(f"An unexpected error occurred with Ollama: {e}")
            return None # Non-retriable error
    print("Error: Max retries reached for Ollama API call.")
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
        return call_ollama_api(ollama_model_name, prompt, temperature=temperature)
    elif openai_client: 
        return call_openai_api(model_identifier, prompt, temperature=temperature)
    else:
        print(f"Error: Model provider for '{model_identifier}' not configured or available.")
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
                 print(f"Error: Missing required key '{key}' in LLM JSON output: {data}")
                 return None
             try:
                 # Convert to float, handle potential non-numeric values
                 parsed_data[key] = float(data[key])
             except (ValueError, TypeError):
                 print(f"Error: Could not convert '{key}' value '{data[key]}' to float.")
                 return None

        # Add optional keys if present
        if "reasoning" in data and isinstance(data["reasoning"], str):
            parsed_data["reasoning"] = data["reasoning"]
        if "critique" in data and isinstance(data["critique"], str):
             parsed_data["critique"] = data["critique"]

        # Parameter sanity checks 
        if not (0 < parsed_data["height"] < 1000): # e.g., height between 0 and 1km
            print(f"Warning: Parsed height {parsed_data['height']} seems unrealistic.")
        if not (0 < parsed_data["horizontal_velocity"] < 1000): # e.g., velocity < 1km/s
            print(f"Warning: Parsed velocity {parsed_data['horizontal_velocity']} seems unrealistic.")


        return parsed_data

    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse LLM response as JSON: {e}")
        print(f"Raw response was: {response_text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during JSON parsing: {e}")
        return None