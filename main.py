import yaml
import json
import os

from argparse import ArgumentParser
from importlib import import_module

from simlm.ground import FlatGround, SineGround, InterpolatedGround
from simlm.projectiles import ProjectileSimulator
from simlm.runner import SimLMRunner


def save_results(results, filename="results/experiment_results.jsonl"):
    """Appends results to a JSON Lines file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, "a") as f:
            f.write(json.dumps(results) + "\n")
    except Exception as e:
        print(f"Error saving results: {e}")


def load_examples(filepath="examples/few_shot_data.yaml"):
    """Placeholder function to load few-shot examples."""
    # Example structure expected by templates:
    # [ {'query': str, 'reasoning': str, 'answer_json': str}, ... ] for CoT
    # [ {'query': str, 'history': list_of_steps, 'final_answer_json': str, 'target': float}, ... ] for SimLM
    print(f"Note: Example loading from {filepath} is not implemented.")
    return None


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", "-c", help="Configuration file for simlm.", default="config.yml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load Few-Shot Examples
    # few_shot_examples_cot = load_examples(config.get("few_shot_examples_path", "examples/few_shot_data.yml"))
    # few_shot_examples_simlm = load_examples(config.get("few_shot_examples_path", "examples/few_shot_data.yml"))
    few_shot_examples_cot = None  # Using 0-shot for simplicity now
    few_shot_examples_simlm = None

    # Run Experiments
    experiment = config.get("experiment", "flat")
    runner = SimLMRunner(config)

    # Experiment A: Flat Ground
    if experiment == "flat":    
        print("\nExperiment A: Flat Ground")
        ground = FlatGround()

    # Experiment B: Uneven Sinusoid Ground
    if experiment == "sine":
        print("\nExperiment B: Sine Ground")
        ground = SineGround(
            config.get("amplitude", 0.25), 
            config.get("frequency", 0.5)
        )

    # Experiment C: Varying Difficulty 
    if experiment == "interpolated":
        print("\nExperiment C: Varying Difficulty")
        difficulty = config.get("difficulty", 0.5)
        interpol_ground = InterpolatedGround(difficulty)

    results_cot = runner.run_baseline_cot(ground, few_shot_examples_cot)
    save_results(results_cot)
    results_simlm = runner.run_simlm(ground, few_shot_examples_simlm)
    save_results(results_simlm)