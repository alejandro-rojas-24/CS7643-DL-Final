import logging
import os
import time

from jinja2 import Environment, FileSystemLoader, select_autoescape

from simlm.config import Config
from simlm.ground import (
    EasyGround,
    FlatGround,
    Ground,
    HardGround,
    InterpolatedGround,
    SineGround,
)
from simlm.llm import LLMClient
from simlm.projectiles import ProjectileSimulator

# Jinja Setup
template_dir = os.path.join(os.path.dirname(__file__), "prompt_templates")
jinja_env = Environment(
    loader=FileSystemLoader(template_dir),
    autoescape=select_autoescape(["html", "xml", "j2"]),
    trim_blocks=True,
    lstrip_blocks=True,
)

logger = logging.getLogger(__name__)


def ordinal(n: int) -> str:
    """Add ordinal number suffix"""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    return str(n) + suffix


jinja_env.filters["ordinal"] = ordinal


def calculate_error(bounce_locations, target_bounce_num, target_dist):
    """Calculates the error for the target bounce."""
    if bounce_locations is None or len(bounce_locations) < target_bounce_num:
        return None, None
    actual_dist = bounce_locations[target_bounce_num - 1]
    error = abs(actual_dist - target_dist)
    return actual_dist, error


class SimLMRunner:
    def __init__(self, config: Config) -> None:
        self.config = config
        # LLM
        self.model_service = config.llm.service
        self.model_name = config.llm.model_name
        self.temperature = config.llm.temperature

        # Simulation
        self.fps = config.simulation.fps
        self.gravity_y = config.simulation.gravity.y
        self.max_duration = config.simulation.max_duration
        self.elasticity = config.projectile.elasticity
        self.mass = config.projectile.mass
        self.radius = config.projectile.radius
        self.simulator = ProjectileSimulator(
            fps=self.fps,
            gravity_y=self.gravity_y,
        )
        # Ground
        self.x_min = config.ground.x_min
        self.x_max = config.ground.x_max
        self.step = config.ground.step
        self.friction = config.ground.friction

        self.ground_type = config.ground.type
        if self.ground_type == "flat":
            self.ground = FlatGround()

        # Experiment B: Uneven Sinusoid Ground
        elif self.ground_type == "sine":
            self.ground = SineGround(
                config.ground.amplitude,
                config.ground.frequency,
            )

        # Experiment C: Varying Difficulty
        elif self.ground_type == "interpolated":
            difficulty = config.ground.difficulty
            easy_ground = EasyGround(
                config.ground.easy.amplitude,
                config.ground.easy.frequency,
            )
            hard_ground = HardGround(
                config.ground.hard.amplitudes,
                config.ground.hard.frequencies,
            )
            self.ground = InterpolatedGround(
                difficulty=difficulty,
                easy_ground=easy_ground,
                hard_ground=hard_ground,
            )

        # Experiment
        self.distance = config.experiment.target_distance
        self.bounce_number = config.experiment.target_bounce_number
        self.tolerance = config.experiment.tolerance
        self.max_iterations = config.experiment.max_iterations

    def _run_simulation(self, h, v, ground: Ground):
        """Sets up and runs a single simulation instance."""
        # Reset the simulator space for a clean run
        self.simulator._setup_space()
        self.simulator.add_ground(
            self.x_min,
            self.x_max,
            self.step,
            ground,
            self.friction,
        )
        self.simulator.add_projectile(
            h,
            v,
            self.mass,
            self.radius,
            self.elasticity,
        )
        bounce_locs = self.simulator.simulate(
            target_bounces=self.bounce_number,
            duration=self.max_duration,
        )
        return bounce_locs

    def run_baseline_cot(self, few_shot_examples=None):
        """Runs the baseline Chain-of-Thought method."""
        print(f"\nRunning Baseline CoT for {self.ground}")
        start_time = time.time()

        template = jinja_env.get_template("cot_prompt.j2")
        prompt = template.render(
            gravity_y=self.gravity_y,
            elasticity=self.elasticity,
            target_bounce_number=self.bounce_number,
            target_distance=self.distance,
            target_tolerance=self.tolerance,
            examples=few_shot_examples,
            ground_description=self.ground.description,
        )

        client = LLMClient.from_model_service(
            self.model_service,
            self.model_name,
        )

        data = client.generate_and_parse(
            prompt=prompt,
            temperature=self.temperature,
        )

        if not data:
            logger.error("Failed to get valid parameters from LLM.")
            return {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": False,
                "config": self.config.model_dump(),
                "error": "LLM Parsing Failed",
                "model": self.model_name,
            }

        h = data.height
        v = data.horizontal_velocity
        reasoning = data.reasoning
        print(f"LLM Proposed: h={h:.2f}m, v={v:.2f}m/s")
        print(f"LLM Reasoning: {reasoning}")

        bounce_locs = self._run_simulation(h, v, self.ground)
        actual_dist, error = calculate_error(
            bounce_locs, self.bounce_number, self.distance
        )

        end_time = time.time()
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "success": error is not None and error <= self.tolerance,
            "config": self.config.model_dump(),
            "predicted_h": h,
            "predicted_v": v,
            "final_h": h,
            "final_v": v,
            "reasoning": reasoning,
            "bounce_locations": bounce_locs,
            "actual_distance_bounce_3": actual_dist,
            "error": error,
            "within_tolerance": error is not None and error <= self.tolerance,
            "time_taken": end_time - start_time,
        }
        print(
            f"Result: Bounces={bounce_locs}, Target Bounce Dist={actual_dist:.2f}m, Error={error:.2f}m"
            if error is not None
            else "Result: Target bounce not reached."
        )
        print(f"Finished CoT in {result['time_taken']:.2f}s")
        return result

    def run_simlm(self, few_shot_examples=None):
        """Runs the SimLM iterative method."""
        print(f"\nRunning SimLM for {self.ground}")
        start_time = time.time()
        history = []  # Stores dicts for each step: {'type': 'reasoning'/'simulation'/'critique', ...}
        current_h, current_v = None, None

        for iteration in range(self.max_iterations):
            print(f"\nSimLM Iteration {iteration + 1}/{self.max_iterations}")

            # Prepare Prompt (Reasoning or Critique)
            if iteration == 0:
                template = jinja_env.get_template("reasoning_prompt.j2")
                prompt = template.render(
                    gravity_y=self.gravity_y,
                    elasticity=self.elasticity,
                    target_bounce_number=self.bounce_number,
                    target_distance=self.distance,
                    target_tolerance=self.tolerance,
                    examples=few_shot_examples,
                    ground_description=self.ground.description,
                )
                step_type = "reasoning"
            else:
                template = jinja_env.get_template("critique_prompt.j2")
                prompt = template.render(
                    gravity_y=self.gravity_y,
                    elasticity=self.elasticity,
                    target_bounce_number=self.bounce_number,
                    target_distance=self.distance,
                    target_tolerance=self.tolerance,
                    ground_description=self.ground.description,
                    history=history,  # Pass the whole history
                )
                step_type = "critique"

            # 2. Call LLM
            client = LLMClient.from_model_service(
                self.model_service,
                self.model_name,
            )
            data = client.generate_and_parse(
                prompt=prompt,
                temperature=self.temperature,
            )

            if not data:
                logger.error(
                    f"Error: Failed to get valid {step_type} and parameters from LLM on iteration {iteration + 1}."
                )
                final_error = (
                    history[-1]["error"]
                    if history and "error" in history[-1]
                    else None
                )
                success_flag = (
                    final_error is not None and final_error <= self.tolerance
                )
                return {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "success": success_flag,
                    "config": self.config.model_dump(),
                    "error": f"LLM Parsing Failed Iter {iteration + 1}",
                    "history": history,
                }

            current_h = data.height
            current_v = data.horizontal_velocity
            text_content = data.reasoning or data.critique

            history.append(
                {
                    "type": step_type,
                    "text": text_content,
                    "h": current_h,
                    "v": current_v,
                    "iteration": iteration + 1,
                }
            )

            # Run Simulation
            bounce_locs = self._run_simulation(
                current_h, current_v, self.ground
            )
            actual_dist, error = calculate_error(
                bounce_locs,
                self.bounce_number,
                self.distance,
            )

            print(
                f"Simulation Result: Bounces={bounce_locs}, Target Bounce Dist={actual_dist:.2f}m, Error={error:.2f}m"
                if error is not None
                else "Simulation Result: Target bounce not reached."
            )

            history.append(
                {
                    "type": "simulation",
                    "bounce_locs": bounce_locs,
                    "actual_dist": actual_dist,
                    "error": error,
                    "iteration": iteration + 1,
                }
            )

            # Check Success Condition (as per paper: LLM prompted to stop if requirements met)
            if error is not None and error <= self.tolerance:
                print(
                    f"Success! Target achieved within tolerance at iteration {iteration + 1}."
                )
                break  # Exit loop on success

        # End of loop or break
        end_time = time.time()
        final_error = (
            history[-1]["error"]
            if history and "error" in history[-1]
            else None
        )
        final_dist = (
            history[-1]["actual_dist"]
            if history and "actual_dist" in history[-1]
            else None
        )
        final_bounces = (
            history[-1]["bounce_locs"]
            if history and "bounce_locs" in history[-1]
            else []
        )

        success_flag = (
            final_error is not None and final_error <= self.tolerance
        )

        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "success": success_flag,
            "config": self.config.model_dump(),
            # Estimate based on steps stored
            "iterations_run": len(history) // 3
            + (1 if len(history) % 3 > 0 else 0),
            "final_h": current_h,
            "final_v": current_v,
            "bounce_locations": final_bounces,
            "actual_distance_bounce_3": final_dist,
            "error": final_error,
            "within_tolerance": success_flag,
            "history": history,
            "time_taken": end_time - start_time,
        }
        print(
            f"Finished SimLM in {result['time_taken']:.2f}s. Final Error: {final_error:.2f}m"
            if final_error is not None
            else "Finished SimLM. Target bounce not reached."
        )
        return result
