import json
import logging
import os
from argparse import ArgumentParser

from simlm.config import Config
from simlm.projectiles import ProjectileSimulator
from simlm.runner import SimLMRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def save_results(results, filename="results/experiment_results.jsonl"):
    """Appends results to a JSON Lines file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, "a") as f:
            f.write(json.dumps(results) + "\n")
    except Exception as e:
        logger.exception(f"Error saving results: {e}")


def load_examples(filepath="examples/few_shot_data.yaml"):
    """Placeholder function to load few-shot examples."""
    # Example structure expected by templates:
    # [ {'query': str, 'reasoning': str, 'answer_json': str}, ... ] for CoT
    # [ {'query': str, 'history': list_of_steps, 'final_answer_json': str, 'target': float}, ... ] for SimLM
    
    raise NotImplementedError


def run_experiment(config: Config) -> tuple[SimLMRunner, dict]:
    """Run the experiment based on the provided configuration."""
    runner = SimLMRunner(config)
    experiment_type = config.experiment.type
    if experiment_type == "baseline_cot":
        return runner, runner.run_baseline_cot()
    elif experiment_type == "simlm":
        return runner, runner.run_simlm()
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


def plot_trajectory(runner, results) -> None:
    """Plot the trajectory of the projectile."""
    final_h = results["final_h"]
    final_v = results["final_v"]
    plot_sim = ProjectileSimulator(runner.fps)

    plot_sim.add_ground(
        runner.x_min,
        runner.x_max,
        runner.step,
        runner.ground,
        runner.friction,
    )
    plot_sim.add_projectile(
        final_h,
        final_v,
        runner.mass,
        runner.radius,
        runner.elasticity,
    )

    plot_duration = runner.max_duration
    plot_sim.simulate(
        target_bounces=len(results["bounce_locations"]),
        duration=plot_duration,
    )

    title = f"SimLM Final Trajectory (h={final_h:.2f}, v={final_v:.2f})"
    plot_sim.plot_trajectory(title=title)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        help="Configuration file for simlm.",
        default="config.yml",
    )
    args = parser.parse_args()

    config = Config.from_file(args.config)

    runner, results = run_experiment(config)
    if config.experiment.save_results:
        save_results(results)

    if config.experiment.visualize:
        plot_trajectory(runner, results)

    print("\nExperiment Runs Complete.")
