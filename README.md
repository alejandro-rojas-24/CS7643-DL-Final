# CS7643-DL-Final
Repository to host our code and output for our final project

Based off of the paper [SimLM: Can Language Models Infer Parameters of Physical Systems?](https://arxiv.org/abs/2312.14215)

## Project Overview
This project explores the capabilities of language models in understanding and inferring parameters of physical systems. We implement physics simulations using Pymunk to generate training data and test the ability of language models to predict physical parameters from system descriptions.


## Project Structure
<<<<<<< HEAD
- `physics_sim/`: Contains physics simulation code
  - `simple_projectile_example.py`: Example implementation of a projectile motion simulation

## Usage
The project currently includes a simple projectile motion simulation that demonstrates:
- Physics-based simulation using Pymunk
- Ground geometry generation
- Projectile trajectory calculation
- Visualization of results

To run the example simulation:
```bash
python physics_sim/simple_projectile_example.py
```
=======
Need to update

## Core Functionality

*   **Physics Simulation:** Accurate 2D projectile simulation using Pymunk, including bounce detection and configurable ground geometry (flat, sinusoidal, interpolated).
*   **LLM Interaction:** Support for multiple LLM backends (OpenAI, Ollama) with JSON output parsing and basic error handling.
*   **Prompting Strategies:** Implementation of SimLM (iterative reasoning, simulation feedback, self-critique) and baseline CoT.
*   **Templating:** Use of Jinja2 for flexible and maintainable prompt construction.
*   **Experiment Runner:** `main.py` allows running predefined experiments (A, B, C from the paper) across different models and strategies.


## Dependencies

All required Python packages are listed in `requirements.txt`.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/alejandro-rojas-24/CS7643-DL-Final.git
    cd simlm_project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure LLMs:**
    *   **OpenAI:** Set your API key in `config.py` (variable `OPENAI_API_KEY`) or as an environment variable `OPENAI_API_KEY`.
    *   **Ollama:**
        *   Ensure you have Ollama installed and running ([ollama.com](https://ollama.com/)).
        *   Pull the desired models (e.g., `ollama pull llama3`, `ollama pull mistral`).
        *   Verify the `OLLAMA_BASE_URL` in `config.py` matches your Ollama server address (default is `http://localhost:11434`).

## Usage

The main script to run experiments is `main.py`.

1.  **Configure the desired model:** Edit the `model` variable found within the `config.yml`. Examples:
    ```yaml
    model: "gpt-3.5-turbo" # "ollama/llama2" "google/gemini-2.0-flash-lite"
    ```

2.  **Run the experiments:**
    ```bash
    python main.py --config config.yml
    ```
    This will execute the defined experiments (currently A (no parameters), B (amplitude and frequency), and C (difficulty)) using the selected model for both the CoT baseline and the SimLM strategy. Output, including parameters, simulation results, and errors, will be printed to the console.

3.  **Customize:** Modify `main.py` or `config.py` to change experiment parameters, target distances, number of bounces, models used, or implement loading/saving of results and examples.
    ```yaml
    experiment: "flat" # "sine" "interpolated"
    elasticity: 0.9
    mass: 1.0
    radius: 0.05
    difficulty: 0.5
    ```

## Ollama Docker Usage

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






Okay, here's the updated "Configuration" section for your README.md, reflecting the structured configuration approach shown in your YAML snippet. It assumes you'll be loading these settings from a file (e.g., `config.yaml`) into your Python code, replacing the direct use of constants from `config.py`.

---

## Configuration

Project settings and experiment parameters are managed through a configuration file (e.g., `config.yaml`). This allows for easy modification of settings without changing the core Python code.

Modify the values in `config.yaml` (or your chosen configuration file) to control various aspects of the simulation and experiments:

*   **Experiment Setup:** Defines the specific scenario being run.
    *   `experiment`: Selects the ground type ("flat", "sine", "interpolated").
    *   `target_distance`, `target_bounce_number`, `target_tolerance`: Defines the objective for the projectile.
    *   `max_iterations`: Maximum refinement steps for the SimLM strategy.
    *   `few_shot_examples_path`: Path to optional few-shot examples file.

*   **LLM Selection:** Configures the language model to be used.
    *   `model`: Specifies the model identifier (e.g., "gpt-3.5-turbo", "ollama/llama3").
    *   `temperature`: Controls the randomness of the LLM's output.

*   **Simulation Parameters:** Core physics engine settings.
    *   `fps`: Simulation frames per second (influences accuracy).
    *   `gravity_y`: Acceleration due to gravity.
    *   `max_duration`: Maximum simulation time if the target bounces aren't reached.
    *   `elasticity`, `mass`, `radius`: Physical properties of the projectile.

*   **Ground Generation:** Parameters defining the terrain.
    *   `x_min`, `x_max`, `step`: Defines the range and resolution for creating ground segments.
    *   `friction`: Friction coefficient of the ground surface.
    *   `amplitude`, `frequency`: Parameters for sinusoidal ground types (e.g., 'sine' experiment).
    *   `difficulty`: Interpolation factor for the 'interpolated' ground experiment (0=easy, 1=hard).

---
>>>>>>> 3e1d2746f38f0b43c54bb6c21f137767304256f8
