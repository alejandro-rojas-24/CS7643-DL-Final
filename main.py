import os
import json
import config 
from simlm.runner import SimLMRunner
from simlm.projectiles import ProjectileSimulator
from simlm.grounds import flat_ground, sine_ground, interpolated_ground
import numpy as np 

def save_results(results, filename="results/experiment_results.jsonl"):
    """Appends results to a JSON Lines file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, 'a') as f:
            f.write(json.dumps(results) + '\n')
    except Exception as e:
        print(f"Error saving results: {e}")

def load_examples(filepath=config.FEW_SHOT_EXAMPLES_PATH):
    """Placeholder function to load few-shot examples."""
    # Example structure expected by templates:
    # [ {'query': str, 'reasoning': str, 'answer_json': str}, ... ] for CoT
    # [ {'query': str, 'history': list_of_steps, 'final_answer_json': str, 'target': float}, ... ] for SimLM
    print(f"Note: Example loading from {filepath} is not implemented.")
    return None 

if __name__ == "__main__":
    # TODO - Need to integrate with Mac Attack's containers (ollama)
    # Select Model 
    model = config.MODEL
    runner = SimLMRunner(model_identifier=model, temperature=config.TEMPERATURE)

    # Load Few-Shot Examples
    # few_shot_examples_cot = load_examples("examples/cot_examples.yaml")
    # few_shot_examples_simlm = load_examples("examples/simlm_examples.yaml")
    few_shot_examples_cot = None # Using 0-shot for simplicity now
    few_shot_examples_simlm = None

    # Run Experiments
    # Experiment A: Flat Ground
    print("\nExperiment A: Flat Ground")
    result_a_cot = runner.run_baseline_cot(flat_ground, few_shot_examples_cot)
    save_results(result_a_cot)
    result_a_simlm = runner.run_simlm(flat_ground, few_shot_examples_simlm)
    save_results(result_a_simlm)

    # Experiment B: Uneven Sinusoid Ground
    print("\nExperiment B: Sine Ground")
    result_b_cot = runner.run_baseline_cot(sine_ground, few_shot_examples_cot)
    save_results(result_b_cot)
    result_b_simlm = runner.run_simlm(sine_ground, few_shot_examples_simlm)
    save_results(result_b_simlm)