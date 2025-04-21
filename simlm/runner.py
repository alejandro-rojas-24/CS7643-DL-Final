import os
import time

from jinja2 import Environment, FileSystemLoader, select_autoescape

from simlm.ground import Ground
from simlm.llm_interface import get_llm_response, parse_llm_json_output
from simlm.projectiles import ProjectileSimulator

# Jinja Setup
template_dir = os.path.join(os.path.dirname(__file__), "prompt_templates")
jinja_env = Environment(
    loader=FileSystemLoader(template_dir),
    autoescape=select_autoescape(["html", "xml", "j2"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


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
    def __init__(self, config: dict) -> None:

        # LLM
        self.model_identifier = config.get("model", "gpt-3.5-turbo")
        self.temperature = config.get("temperature", 0.5)

        # Simulation
        self.fps = config.get("fps", 1000)
        self.gravity_y = config.get("gravity_y", -9.81)
        self.max_duration = config.get("max_duration", 20)
        self.elasticity = config.get("elasticity", 0.9)
        self.mass = config.get("mass", 1.0)
        self.radius = config.get("radius", 0.05)
        self.simulator = ProjectileSimulator(fps=self.fps, gravity_y=self.gravity_y)
        # Ground
        self.x_min = config.get("x_min", -10.0)
        self.x_max = config.get("x_max", 100.0)
        self.step = config.get("step", 0.1)
        self.friction = config.get("friction", 0.8)

        # Experiment
        self.distance = config.get("target_distance", 50.0)
        self.bounce_number = config.get("target_bounce_number", 3)
        self.tolerance = config.get("target_tolerance", 1.0)
        self.max_iterations = config.get("max_iterations", 5)

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
        self.simulator.add_projectile(h, v, self.mass, self.radius, self.elasticity)
        bounce_locs = self.simulator.simulate(
            target_bounces=self.bounce_number,
            duration=self.max_duration,
        )
        return bounce_locs

    def run_baseline_cot(self, ground: Ground, few_shot_examples=None):
        """Runs the baseline Chain-of-Thought method."""
        print(f"\nRunning Baseline CoT for {ground}")
        start_time = time.time()

        template = jinja_env.get_template("cot_prompt.j2")
        prompt = template.render(
            gravity_y=self.gravity_y,
            elasticity=self.elasticity,
            target_bounce_number=self.bounce_number,
            target_distance=self.distance,
            target_tolerance=self.tolerance,
            examples=few_shot_examples,
            ground_description=str(ground),
        )

        raw_response = get_llm_response(self.model_identifier, prompt, self.temperature)
        parsed_data = parse_llm_json_output(raw_response)

        if (
            not parsed_data
            or "height" not in parsed_data
            or "horizontal_velocity" not in parsed_data
        ):
            print("Error: Failed to get valid parameters from LLM.")
            return {"success": False, "error": "LLM Parsing Failed"}

        h = parsed_data["height"]
        v = parsed_data["horizontal_velocity"]
        reasoning = parsed_data.get("reasoning", "N/A")
        print(f"LLM Proposed: h={h:.2f}m, v={v:.2f}m/s")
        print(f"LLM Reasoning: {reasoning}")

        bounce_locs = self._run_simulation(h, v, ground)
        actual_dist, error = calculate_error(
            bounce_locs, self.bounce_number, self.distance
        )

        end_time = time.time()
        result = {
            "success": error is not None and error <= self.tolerance,
            "strategy": "CoT",
            "model": self.model_identifier,
            "ground": str(ground),
            "predicted_h": h,
            "predicted_v": v,
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

    def run_simlm(self, ground: Ground, few_shot_examples=None):
        """Runs the SimLM iterative method."""
        print(f"\nRunning SimLM for {ground}")
        start_time = time.time()
        history = (
            []
        )  # Stores dicts for each step: {'type': 'reasoning'/'simulation'/'critique', ...}
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
                    ground_description=str(ground),
                )
                expected_keys = ["reasoning", "height", "horizontal_velocity"]
                step_type = "reasoning"
            else:
                template = jinja_env.get_template("critique_prompt.j2")
                prompt = template.render(
                    gravity_y=self.gravity_y,
                    elasticity=self.elasticity,
                    target_bounce_number=self.bounce_number,
                    target_distance=self.distance,
                    target_tolerance=self.tolerance,
                    ground_description=str(ground),
                    history=history,  # Pass the whole history
                )
                expected_keys = ["critique", "height", "horizontal_velocity"]
                step_type = "critique"

            # Call LLM
            raw_response = get_llm_response(
                self.model_identifier, prompt, self.temperature
            )
            parsed_data = parse_llm_json_output(raw_response)

            if not parsed_data or not all(key in parsed_data for key in expected_keys):
                print(
                    f"Error: Failed to get valid {step_type} and parameters from LLM on iteration {iteration + 1}."
                )
                final_error = (
                    history[-1]["error"] if history and "error" in history[-1] else None
                )
                success_flag = final_error is not None and final_error <= self.tolerance
                return {
                    "success": success_flag,
                    "error": f"LLM Parsing Failed Iter {iteration + 1}",
                    "history": history,
                }

            current_h = parsed_data["height"]
            current_v = parsed_data["horizontal_velocity"]
            text_content = parsed_data.get("reasoning") or parsed_data.get(
                "critique", "N/A"
            )
            print(
                f"LLM {step_type.capitalize()}: h={current_h:.2f}m, v={current_v:.2f}m/s"
            )
            print(f"LLM Text: {text_content}")

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
            bounce_locs = self._run_simulation(current_h, current_v, ground)
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
            history[-1]["error"] if history and "error" in history[-1] else None
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

        success_flag = final_error is not None and final_error <= self.tolerance

        result = {
            "success": success_flag,
            "strategy": "SimLM",
            "model": self.model_identifier,
            "ground": str(ground),
            # Estimate based on steps stored
            "iterations_run": len(history) // 3 + (1 if len(history) % 3 > 0 else 0),
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
