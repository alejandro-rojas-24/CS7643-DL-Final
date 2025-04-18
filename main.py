import yaml
import json
import os

from argparse import ArgumentParser

from simlm.ground import FlatGround, HardGround, SineGround
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

    runner = SimLMRunner(config)

    #     model_identifier=llm.get("model", "gpt-3.5-turbo"),
    #     temperature=llm.get("temperature", 0.5),
    # )

    # Load Few-Shot Examples
    # few_shot_examples_cot = load_examples(config.get("few_shot_examples_path", "examples/few_shot_data.yml"))
    # few_shot_examples_simlm = load_examples(config.get("few_shot_examples_path", "examples/few_shot_data.yml"))
    few_shot_examples_cot = None  # Using 0-shot for simplicity now
    few_shot_examples_simlm = None

    # Run Experiments
    # Experiment A: Flat Ground
    print("\nExperiment A: Flat Ground")
    flat_ground = FlatGround()
    result_a_cot = runner.run_baseline_cot(flat_ground, few_shot_examples_cot)
    save_results(result_a_cot)
    result_a_simlm = runner.run_simlm(flat_ground, few_shot_examples_simlm)
    save_results(result_a_simlm)

    # Experiment B: Uneven Sinusoid Ground
    print("\nExperiment B: Sine Ground")
    sine_ground = SineGround(0.25, 0.5)
    result_b_cot = runner.run_baseline_cot(sine_ground, few_shot_examples_cot)
    save_results(result_b_cot)
    result_b_simlm = runner.run_simlm(sine_ground, few_shot_examples_simlm)
    save_results(result_b_simlm)

    # Experiment C: Hard ground
    print("\nExperiment C: Hard Ground")
    hard_ground = HardGround()
    result_c_cot = runner.run_baseline_cot(hard_ground, few_shot_examples_cot)
    save_results(result_c_cot)
    result_c_simlm = runner.run_simlm(hard_ground, few_shot_examples_simlm)
    save_results(result_c_simlm)
