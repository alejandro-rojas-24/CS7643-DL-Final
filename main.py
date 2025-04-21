import yaml
import json
import os

from argparse import ArgumentParser

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
    parser.add_argument(
        "--config", "-c", help="Configuration file for simlm.", default="config.yml"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
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
        ground = SineGround(config.get("amplitude", 0.25), config.get("frequency", 0.5))

    # Experiment C: Varying Difficulty
    if experiment == "interpolated":
        print("\nExperiment C: Varying Difficulty")
        difficulty = config.get("difficulty", 0.5)
        ground = InterpolatedGround(difficulty)

    results_cot = runner.run_baseline_cot(ground, few_shot_examples_cot)
    save_results(results_cot)
    results_simlm = runner.run_simlm(ground, few_shot_examples_simlm)
    save_results(results_simlm)

    # Plot final trajectory for the last SimLM run
    if (
        config.get("visualize", False)
        and results_simlm
        and "final_h" in results_simlm
        and "final_v" in results_simlm # Added check for final_v for safety
    ):
        print("\nPlotting final trajectory for successful SimLM run (Experiment C)...")

        # 1. Create a simulator instance specifically for plotting
        final_h = results_simlm["final_h"]
        final_v = results_simlm["final_v"]
        plot_sim = ProjectileSimulator(runner.fps) # Use the same fps as the runner

        # 2. Set up the ground and projectile with the final parameters
        plot_sim.add_ground(
            runner.x_min,
            runner.x_max,
            runner.step,
            ground, # Assuming 'ground' is the correct y_func function
            runner.friction,
        )
        plot_sim.add_projectile(
            final_h, final_v, runner.mass, runner.radius, runner.elasticity
        )

        # 3. Run the simulation to generate internal trajectory and bounces
        #    - Run for a desired duration (e.g., original duration or half).
        #    - Set a high target_bounces so duration is the limiting factor.
        #    - We don't strictly need the return values here as they are stored internally.
        plot_duration = runner.max_duration # Or runner.max_duration / 2 if you prefer shorter plot
        # Use a large number for target_bounces if duration should be the main limiter
        plot_sim.simulate(target_bounces=100, duration=plot_duration)

        # 4. Call plot_trajectory, which now uses the internal data
        #    - No need to pass trajectory points.
        #    - No need to manually set bounce_locations.
        plot_sim.plot_trajectory(title=f"SimLM Final Trajectory (h={final_h:.2f}, v={final_v:.2f})") # Added params to title

    print("\nExperiment Runs Complete.")
    # # Plot final trajectory for the last SimLM run
    # if (
    #     results_simlm
    #     and results_simlm.get("success") 
    #     and "final_h" in results_simlm
    # ):
    #     print("\nPlotting final trajectory for successful SimLM run (Experiment C)...")
    #     # Need to re-run sim to get full trajectory points
    #     final_h = results_simlm["final_h"]
    #     final_v = results_simlm["final_v"]
    #     plot_sim = ProjectileSimulator(runner.fps)
    #     plot_sim.add_ground(
    #         runner.x_min,
    #         runner.x_max,
    #         runner.step,
    #         ground,
    #         runner.friction,
    #     )
    #     plot_sim.add_projectile(
    #         final_h, final_v, runner.mass, runner.radius, runner.elasticity
    #     )
    #     # Simulate for a bit longer to see the full path
    #     trajectory = plot_sim.get_trajectory(duration=runner.max_duration / 2)
    #     plot_sim.bounce_locations = results_simlm.get(
    #         "bounce_locations", []
    #     )  # Add bounces back for plot
    #     plot_sim.plot_trajectory(trajectory, title=f"SimLM Final Trajectory")

    # print("\nExperiment Runs Complete.")
