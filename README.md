# CS7643-DL-Final

Repository to host our code and output for our final project

Based off of the paper [SimLM: Can Language Models Infer Parameters of Physical Systems?](https://arxiv.org/abs/2312.14215)

## Project Overview

This project explores the capabilities of language models in understanding and inferring parameters of physical systems. We implement physics simulations using Pymunk to generate training data and test the ability of language models to predict physical parameters from system descriptions.

## Project Structure

Need to update

## Core Functionality

* **Physics Simulation:** Accurate 2D projectile simulation using Pymunk, including bounce detection and configurable ground geometry (flat, sinusoidal, interpolated).
* **LLM Interaction:** Support for multiple LLM backends (OpenAI, Ollama) with JSON output parsing and basic error handling.
* **Prompting Strategies:** Implementation of SimLM (iterative reasoning, simulation feedback, self-critique) and baseline CoT.
* **Templating:** Use of Jinja2 for flexible and maintainable prompt construction.
* **Experiment Runner:** `main.py` allows running predefined experiments (A, B, C from the paper) across different models and strategies.

## Dependencies

All required Python packages are listed in `requirements.txt`.

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/alejandro-rojas-24/CS7643-DL-Final.git
    cd simlm_project
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure LLMs:**
    * **OpenAI:** Set your API key in `config.py` (variable `OPENAI_API_KEY`) or as an environment variable `OPENAI_API_KEY`.
    * **Ollama:**
        * Ensure you have Ollama installed and running ([ollama.com](https://ollama.com/)).
        * Pull the desired models (e.g., `ollama pull llama3`, `ollama pull mistral`).
        * Verify the `OLLAMA_BASE_URL` in `config.py` matches your Ollama server address (default is `http://localhost:11434`).

## Usage

### Environment Setup

We recommend using a virtual environment to manage dependencies.
Specifically conda is recommended to manage the dependencies for this project if
using python 3.10 or higher.

An environment file is provided in the `environment.yaml` file. To create and activate the
environment, run the following commands:

```bash
conda env create -f environment.yaml
conda activate cs7643-dl-final
```

To deactivate the environment, run:

```bash
conda deactivate
```


### Running Experiments

The main script to run experiments is `main.py`.

1. **Configure the desired model:** Edit the `model` variable found within the `config.yml`. Examples:

    ```yaml
    model: "gpt-3.5-turbo" # "ollama/llama2" "google/gemini-2.0-flash-lite"
    ```

2. **Run the experiments:**

    ```bash
    python main.py --config config.yml
    ```

    This will execute the defined experiments (currently A (no parameters), B (amplitude and frequency), and C (difficulty)) using the selected model for both the CoT baseline and the SimLM strategy. Output, including parameters, simulation results, and errors, will be printed to the console.

3. **Customize:** Modify `main.py` or `config.py` to change experiment parameters, target distances, number of bounces, models used, or implement loading/saving of results and examples.

    ```yaml
    experiment: "flat" # "sine" "interpolated"
    elasticity: 0.9
    mass: 1.0
    radius: 0.05
    difficulty: 0.5
    ```

### Ollama Docker Usage

First, ensure [Docker Desktop is installed](https://docs.docker.com/desktop/)
and running.

Build the docker image from the docker directory:

```bash
cd docker
docker build -t cs7643-dl-final .
```

Spin up the containers:

```bash
docker compose up -d
```

Open <http://localhost:11434/api/tags> in your browser to view downloaded models
and confirm ollama is running.

Running

```bash
python main.py --config.yml
```

will automatically connect to the locally running server.

To stop the containers:

```bash
docker compose down
```

## Available Models

For a full set of available Ollama models, visit the
[Ollama Model Library](https://ollama.com/library).

For a full set of available Google models, visit the
[Gemini API Docs](https://ai.google.dev/gemini-api/docs/models).
Note that while most models are available for free, some may require a paid
subscription or have usage limits.

For a full set of available OpenAI models, visit the [OpenAI Pricing Page](https://platform.openai.com/docs/pricing).

## Recommended Models

The collection of recommended models below is based on the
state-of-the-art larg language models as of April 2025.
Due to hardware limitations, Ollama models (run locally) generally stay under
20GB.

The models listed below include many dimensions to explore, including

- Size (number of parameters)
- Model (e.g. gemma, mistral, phi, llama, deepseek, gemini, gpt)
- Small, generalized, and thinking models of the same flavor (e.g. flash-lite vs
  flash vs pro)
- Open sourced vs proprietary models (e.g. Ollama models vs Google vs OpenAI)

Note that the OpenAI models do include a small fee.
GPT-3.5-turbo is included because it is the closest model to treat as a baseline to the
SimLM paper.
GPT-4.1-nano is included as the cheapest state of the art model available
from OpenAI.


### Recommended Ollama Models 

| Model Name | Parameters | Size | Description |
|------------|------------|------|-------------|
| `gemma3:1b` | 1 billion | 815MB | From Google |
| `gemma3:4b` | 4 billion | 3.3GB | From Google |
| `gemma3:12b` | 12 billion | 8.1GB | From Google |
| `gemma3:27b` | 27 billion | 17GB | From Google |
| `mistral-small3.1` | 24 billion | 15GB | Mistral's small model |
| `mistral` | 7 billion | 4.1GB | The 7B model released by Mistral AI, updated to version 0.3.  |
| `phi4` | 14 billion | 8.1GB | Phi model from Microsoft |
| `phi4-mini` | 3.8 billion | 2.5GB | Phi-4-mini brings significant enhancements in multilingual support, reasoning, and mathematics, and now, the long-awaited function calling feature is finally supported.  |
| `llama3.2:1b` | 1 billion | 1.2GB | Llama 3.2 model from Meta |
| `llama3.2:3b` | 3 billion | 2.8GB | Llama 3.2 model from Meta |
| `deepseek-r1:1.5b` | 1.5 billion | 1.2GB | DeepSeek's first-generation of reasoning models with comparable performance to OpenAI-o1. |
| `deepseek-r1:7b` | 7 billion | 4.7GB | DeepSeek's first-generation of reasoning models with comparable performance to OpenAI-o1. |
| `deepskeek-r1:14b` | 14 billion | 9.0GB | DeepSeek's first-generation of reasoning models with comparable performance to OpenAI-o1. |
| `deepseek-r1:32b` | 32 billion | 20GB | DeepSeek's first-generation of reasoning models with comparable performance to OpenAI-o1. |

### Recommended Google Models

| Model Name | Cost | Description |
|------------|------|-------------|
| `gemini-2.5-flash-preview-04-17` | Free | Flash model from Google |
| `gemini-2.5-pro-exp-03-25` | Free | Gemini 2.5 Pro model, experimental version with enhanced capabilities |
| `gemini-2.0-flash` | Free | Flash model from Google |
| `gemini-2.0-flash-lite` | Free | Flash Lite model from Google |
| `gemini-1.5-flash` | Free | Flash model from Google |
| `gemini-1.5-flash-8b` | Free | Gemini 1.5 Flash-8B is a small model designed for lower intelligence tasks.  |
| `gemini-1.5-flash-pro` | Free | Flash model from Google |

### Recommended OpenAI Models

| Model Name | Input Cost | Output Cost | Description |
|------------|------------------|------|-------------|
| `gpt-3.5-turbo` | $0.50 per million tokens | $1.50 per million  | Fast and cost-effective model for general tasks |
| `gpt-4.1-nano` | $0.025 per million tokens | $0.40 per million tokens | Smallest GPT-4 model |



---

## Configuration

Project settings and experiment parameters are managed through a configuration file (e.g., `config.yaml`). This allows for easy modification of settings without changing the core Python code.

Modify the values in `config.yaml` (or your chosen configuration file) to control various aspects of the simulation and experiments:

* **Experiment Setup:** Defines the specific scenario being run.
  * `experiment`: Selects the ground type ("flat", "sine", "interpolated").
  * `target_distance`, `target_bounce_number`, `target_tolerance`: Defines the objective for the projectile.
  * `max_iterations`: Maximum refinement steps for the SimLM strategy.
  * `few_shot_examples_path`: Path to optional few-shot examples file.

* **LLM Selection:** Configures the language model to be used.
  * `model`: Specifies the model identifier (e.g., "gpt-3.5-turbo", "ollama/llama3").
  * `temperature`: Controls the randomness of the LLM's output.

* **Simulation Parameters:** Core physics engine settings.
  * `fps`: Simulation frames per second (influences accuracy).
  * `gravity_y`: Acceleration due to gravity.
  * `max_duration`: Maximum simulation time if the target bounces aren't reached.
  * `elasticity`, `mass`, `radius`: Physical properties of the projectile.

* **Ground Generation:** Parameters defining the terrain.
  * `x_min`, `x_max`, `step`: Defines the range and resolution for creating ground segments.
  * `friction`: Friction coefficient of the ground surface.
  * `amplitude`, `frequency`: Parameters for sinusoidal ground types (e.g., 'sine' experiment).
  * `difficulty`: Interpolation factor for the 'interpolated' ground experiment (0=easy, 1=hard).

