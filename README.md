# CS7643-DL-Final
Repository to host our code and output for our final project

Based off of the paper [SimLM: Can Language Models Infer Parameters of Physical Systems?](https://arxiv.org/abs/2312.14215)

## Project Overview
This project explores the capabilities of language models in understanding and inferring parameters of physical systems. We implement physics simulations using Pymunk to generate training data and test the ability of language models to predict physical parameters from system descriptions.


## Project Structure
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
python main.py
```

will automatically connect to the locally running server.


To stop the containers:

```bash
docker compose down
```

## Analysis

- Models
- Experiments
- Batching


## Core Functionality

*   **Physics Simulation:** Accurate 2D projectile simulation using Pymunk, including bounce detection and configurable ground geometry (flat, sinusoidal, interpolated).
*   **LLM Interaction:** Support for multiple LLM backends (OpenAI, Ollama) with JSON output parsing and basic error handling.
*   **Prompting Strategies:** Implementation of SimLM (iterative reasoning, simulation feedback, self-critique) and baseline CoT.
*   **Templating:** Use of Jinja2 for flexible and maintainable prompt construction.
*   **Experiment Runner:** `main.py` allows running predefined experiments (A, B, C from the paper) across different models and strategies.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
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

1.  **Configure the desired model:** Edit the `model` variable near the top of the `if __name__ == "__main__":` block in `main.py`. Examples:
    ```python
    # model = "gpt-3.5-turbo"
    # model = "gpt-4-turbo-preview"
    model = "ollama/llama3"
    # model = "ollama/mistral"
    ```

2.  **Run the experiments:**
    ```bash
    python main.py
    ```
    This will execute the defined experiments (currently A, B, and C with one difficulty level) using the selected model for both the CoT baseline and the SimLM strategy. Output, including parameters, simulation results, and errors, will be printed to the console.

3.  **Customize:** Modify `main.py` or `config.py` to change experiment parameters, target distances, number of bounces, models used, or implement loading/saving of results and examples.

## Configuration

Key simulation and experiment parameters can be adjusted in `config.py`:
*   `SIM_FPS`, `SIM_GRAVITY_Y`, `SIM_PROJECTILE_ELASTICITY`: Physics simulation settings.
*   `TARGET_DISTANCE_M`, `TARGET_BOUNCE_NUMBER`, `TARGET_TOLERANCE_M`: Task definition.
*   `SIMLM_MAX_ITERATIONS`: Maximum refinement steps for the SimLM strategy.
*   Ground function definitions and parameters (`GROUND_X_MIN`, `GROUND_X_MAX`, etc.).
*   LLM API endpoints and keys.

## Dependencies

All required Python packages are listed in `requirements.txt`.