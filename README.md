# CS7643-DL-Final

Repository to host our code and output for our final project

Based off of the paper [SimLM: Can Language Models Infer Parameters of Physical Systems?](https://arxiv.org/abs/2312.14215)

## Project Overview

This project explores the capabilities of language models in understanding and inferring parameters of physical systems. We implement physics simulations using Pymunk to generate training data and test the ability of language models to predict physical parameters from system descriptions.

## Core Functionality

* **Physics Simulation:** Accurate 2D projectile simulation using Pymunk, including bounce detection and configurable ground geometry (flat, sinusoidal, interpolated).
* **LLM Interaction:** Support for multiple LLM backends (OpenAI, Ollama) with JSON output parsing and basic error handling.
* **Prompting Strategies:** Implementation of SimLM (iterative reasoning, simulation feedback, self-critique) and baseline CoT.
* **Templating:** Use of Jinja2 for flexible and maintainable prompt construction.
* **Experiment Runner:** `main.py` allows running predefined experiments (A, B, C from the paper) across different models and strategies.

## Dependencies

All required Python packages are listed in `requirements.txt`.

## Setup

**Clone the repository:**

```bash
git clone https://github.com/alejandro-rojas-24/CS7643-DL-Final.git
cd CS7643-DL-Final
```

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
python main.py --config config.yml
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
| `deepseek-r1:14b` | 14 billion | 9.0GB | DeepSeek's first-generation of reasoning models with comparable performance to OpenAI-o1. |
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

Project settings and experiment parameters are managed through the `config.yml` configuration file. This hierarchical YAML format allows for easy modification of settings without changing the core Python code.

Modify the values in `config.yml` to control various aspects of the simulation and experiments:

*   **`experiment`**: Defines the overall experiment setup.
    *   `type`: Selects the prompting strategy to run.
        *   *Options:* `"baseline_cot"`, `"simlm"`
    *   `visualize`: Boolean flag (`true`/`false`) to enable plotting the final trajectory after a successful run.
    *   `save`: Boolean flag (`true`/`false`) to enable saving detailed run results to a JSON Lines file.
    *   `few_shot_examples_path`: Path to an optional JSON Lines file containing examples for few-shot prompting.
    *   `target_distance`: The target x-coordinate (in meters) for the specified bounce.
    *   `target_bounce_number`: Which bounce number should land near the target distance (e.g., `3` for the third bounce).
    *   `tolerance`: The acceptable error margin (in meters) around the `target_distance` for a run to be considered successful.
    *   `max_iterations`: Maximum number of refinement steps (LLM reasoning -> Simulation -> LLM critique cycles) for the `simlm` strategy.

*   **`simulation`**: Core physics engine settings.
    *   `fps`: Simulation frames per second (higher values increase accuracy but slow down simulation).
    *   `max_duration`: Maximum simulation time (in seconds) allowed per attempt before timing out (e.g., if the target bounce isn't reached).
    *   `gravity`: Defines gravitational acceleration.
        *   `y`: Acceleration in the y-direction (typically negative, e.g., `-9.81` m/s²).

*   **`projectile`**: Physical properties of the simulated projectile.
    *   `elasticity`: Coefficient of restitution (bounciness) when colliding with the ground (0=inelastic, 1=perfectly elastic).
    *   `mass`: Mass of the projectile (in kg).
    *   `radius`: Radius of the projectile (in meters).

*   **`ground`**: Parameters defining the terrain the projectile bounces on.
    *   `type`: Selects the type of ground geometry.
        *   *Options:* `"flat"`, `"sine"`, `"interpolated"`
    *   `friction`: Friction coefficient of the ground surface.
    *   `x_min`, `x_max`: The range of x-coordinates (in meters) over which the ground geometry is generated.
    *   `step`: The resolution (in meters) used to create the piecewise linear segments representing the ground.
    *   `amplitude`, `frequency`: Parameters used when `ground.type` is `"sine"`. Defines the amplitude and frequency of the `y = amplitude * sin(frequency * x)` function.
    *   `difficulty`: A factor between 0 and 1 used only when `ground.type` is `"interpolated"`. It linearly interpolates between the `easy` and `hard` ground functions (`(1-difficulty)*easy + difficulty*hard`).
    *   `easy`: Parameters defining the "easy" surface for interpolation.
        *   `amplitude`, `frequency`: Parameters for the easy sinusoid.
    *   `hard`: Parameters defining the "hard" surface for interpolation (sum of sinusoids).
        *   `amplitudes`: A list of amplitudes for each component sinusoid.
        *   `frequencies`: A list of frequencies for each component sinusoid.

*   **`llm`**: Configuration for the Language Model interaction.
    *   `service`: Specifies which LLM provider API to use.
        *   *Options:* `"google"`, `"ollama"`, `"openai"`
    *   `model_name`: The specific identifier for the model within the selected service (e.g., `"gemini-2.0-flash-lite"`, `"llama3.2:3b"`, `"gpt-4.1-nano"`). Ensure this model is available/supported by the chosen service.
    *   `temperature`: Controls the randomness of the LLM's output (0=deterministic, higher values=more random). Typically ranges from 0.0 to 1.0.
## Analysis Pipeline

Once experiments are run using `main.py` or `batch.ipynb`, the results (typically saved to a JSON Lines file like `results/experiment_results.jsonl` if enabled) need to be processed to evaluate the performance of different models and prompting strategies. The typical analysis pipeline involves the following steps:

1.  **Data Loading:** Read the generated JSON Lines file(s) containing the detailed results from each experimental run (e.g., using Python libraries like `pandas` or the standard `json` library). Each line represents one complete run (either CoT or SimLM for a specific setup).
2.  **Data Cleaning & Preparation:**
    *   Filter out any failed runs (e.g., where the LLM failed to produce valid JSON or the simulation timed out without reaching the target bounce).
    *   Parse relevant fields like the final error, predicted parameters (h, v), number of iterations (for SimLM), ground type, model used, and strategy.
3.  **Aggregation & Grouping:** Group the cleaned data based on the factors being compared:
    *   Experiment Type (A: Flat, B: Sine, C: Interpolated)
    *   Ground Difficulty (for Experiment C)
    *   LLM Model Identifier
    *   Prompting Strategy (CoT, SimLM)
    *   Number of Few-Shot Examples (if implemented and varied)
4.  **Metric Calculation:** For each group, calculate key performance metrics:
    *   **Mean Absolute Error (MAE):** The average absolute difference between the achieved distance of the target bounce (e.g., 3rd bounce) and the target distance (e.g., 50m). This is the primary metric used in the SimLM paper.
    *   **Success Rate:** The percentage of runs within each group where the final error was less than or equal to the defined tolerance (e.g., 1.0m).
    *   **Relative Error (for comparing strategies):** For specific comparisons (like SimLM vs. CoT on Experiment C), calculate the ratio of MAE (e.g., `MAE_SimLM / MAE_CoT`). Values less than 1 indicate SimLM performed better.
    *   **Average Iterations (for SimLM):** Calculate the mean number of iterations SimLM took to converge or reach the maximum limit.
5.  **Statistical Significance (Optional):** Use statistical tests (e.g., two-sample t-tests, as mentioned in the paper) to determine if observed differences in MAE or success rates between groups (e.g., SimLM 1-shot vs CoT 1-shot on Experiment B) are statistically significant.
6.  **Visualization:** Generate plots to visually represent the findings, similar to those in the SimLM paper:
    *   Bar charts comparing MAE across models, strategies, and few-shot counts (e.g., Figures 1 & 4 in the paper).
    *   Line or bar plots showing how MAE or relative error changes with increasing ground difficulty in Experiment C (e.g., Figure 3).

This pipeline allows for systematic comparison and evaluation of how well different language models, augmented with simulation capabilities (SimLM) or not (CoT), can infer parameters for physical systems under varying conditions. Tools like `pandas`, `numpy`, `scipy.stats`, `matplotlib`, and `seaborn` are commonly used for these tasks.