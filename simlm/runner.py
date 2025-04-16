# simlm_runner.py
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
import numpy as np
import time
import random

from simlm.projectiles import ProjectileSimulator
from simlm.llm_interface import get_llm_response, parse_llm_json_output
import config 

# Jinja Setup
template_dir = os.path.join(os.path.dirname(__file__), 'prompt_templates')
jinja_env = Environment(
    loader=FileSystemLoader(template_dir),
    autoescape=select_autoescape(['html', 'xml', 'j2']),
    trim_blocks=True,
    lstrip_blocks=True
)
# Add ordinal number filter
def ordinal(n):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix
jinja_env.filters['ordinal'] = ordinal


def calculate_error(bounce_locations, target_bounce_num, target_dist):
    """Calculates the error for the target bounce."""
    if bounce_locations is None or len(bounce_locations) < target_bounce_num:
        return None, None 
    actual_dist = bounce_locations[target_bounce_num - 1]
    error = abs(actual_dist - target_dist)
    return actual_dist, error

def get_ground_description(func):
    """Generates a simple description of the ground function."""
    if func.__name__ == "flat_ground":
        return "y = 0 (Flat)"
    elif func.__name__ == "sine_ground":
        return "y = sin(x) (Simple Sinusoid)"
    else:
        try:
            import inspect
            src = inspect.getsource(func)
            # Extract return statement or key part
            desc = src.split('return')[-1].strip()
            if len(desc) > 100: desc = func.__name__
            return f"y = f(x) where f is defined as: {desc}"
        except:
            return func.__name__ 

class SimLMRunner:
    def __init__(self, model_identifier, temperature=0.7):
        self.model_identifier = model_identifier
        self.temperature = temperature
        self.simulator = ProjectileSimulator() 

    def _run_simulation(self, h, v, ground_func):
        """Sets up and runs a single simulation instance."""
        # Reset the simulator space for a clean run
        self.simulator._setup_space()
        self.simulator.add_ground(config.GROUND_X_MIN, config.GROUND_X_MAX,
                                  config.GROUND_STEP, ground_func, config.SIM_GROUND_FRICTION)
        self.simulator.add_projectile(h, v, config.SIM_PROJECTILE_MASS, config.SIM_PROJECTILE_RADIUS, config.SIM_PROJECTILE_ELASTICITY)
        bounce_locs = self.simulator.simulate(
            target_bounces=config.TARGET_BOUNCE_NUMBER,
            duration=config.SIM_MAX_DURATION
        )
        return bounce_locs

    def run_baseline_cot(self, ground_func, few_shot_examples=None):
        """Runs the baseline Chain-of-Thought method."""
        print(f"\nRunning Baseline CoT for {get_ground_description(ground_func)}")
        start_time = time.time()

        template = jinja_env.get_template("cot_prompt.j2")
        prompt = template.render(
            gravity_y=config.SIM_GRAVITY_Y,
            elasticity=config.SIM_PROJECTILE_ELASTICITY,
            target_bounce_number=config.TARGET_BOUNCE_NUMBER,
            target_distance=config.TARGET_DISTANCE,
            target_tolerance=config.TARGET_TOLERANCE,
            examples=few_shot_examples,
            ground_description=get_ground_description(ground_func)
        )

        raw_response = get_llm_response(self.model_identifier, prompt, self.temperature)
        parsed_data = parse_llm_json_output(raw_response)

        if not parsed_data or "height" not in parsed_data or "horizontal_velocity" not in parsed_data:
            print("Error: Failed to get valid parameters from LLM.")
            return {"success": False, "error": "LLM Parsing Failed"}

        h = parsed_data["height"]
        v = parsed_data["horizontal_velocity"]
        reasoning = parsed_data.get("reasoning", "N/A")
        print(f"LLM Proposed: h={h:.2f}m, v={v:.2f}m/s")
        print(f"LLM Reasoning: {reasoning}")

        bounce_locs = self._run_simulation(h, v, ground_func)
        actual_dist, error = calculate_error(bounce_locs, config.TARGET_BOUNCE_NUMBER, config.TARGET_DISTANCE)

        end_time = time.time()
        result = {
            "success": error is not None and error <= config.TARGET_TOLERANCE,
            "strategy": "CoT",
            "model": self.model_identifier,
            "ground": get_ground_description(ground_func),
            "predicted_h": h,
            "predicted_v": v,
            "reasoning": reasoning,
            "bounce_locations": bounce_locs,
            "actual_distance_bounce_3": actual_dist,
            "error": error,
            "within_tolerance": error is not None and error <= config.TARGET_TOLERANCE,
            "time_taken": end_time - start_time,
        }
        print(f"Result: Bounces={bounce_locs}, Target Bounce Dist={actual_dist:.2f}m, Error={error:.2f}m" if error is not None else "Result: Target bounce not reached.")
        print(f"Finished CoT in {result['time_taken']:.2f}s")
        return result


    def run_simlm(self, ground_func, few_shot_examples=None):
        """Runs the SimLM iterative method."""
        print(f"\nRunning SimLM for {get_ground_description(ground_func)}")
        start_time = time.time()
        history = [] # Stores dicts for each step: {'type': 'reasoning'/'simulation'/'critique', ...}
        current_h, current_v = None, None

        for iteration in range(config.SIMLM_MAX_ITERATIONS):
            print(f"\nSimLM Iteration {iteration + 1}/{config.SIMLM_MAX_ITERATIONS}")

            # 1. Prepare Prompt (Reasoning or Critique)
            if iteration == 0:
                template = jinja_env.get_template("reasoning_prompt.j2")
                prompt = template.render(
                    gravity_y=config.SIM_GRAVITY_Y,
                    elasticity=config.SIM_PROJECTILE_ELASTICITY,
                    target_bounce_number=config.TARGET_BOUNCE_NUMBER,
                    target_distance=config.TARGET_DISTANCE,
                    target_tolerance=config.TARGET_TOLERANCE,
                    examples=few_shot_examples,
                    ground_description=get_ground_description(ground_func)
                )
                expected_keys = ["reasoning", "height", "horizontal_velocity"]
                step_type = "reasoning"
            else:
                template = jinja_env.get_template("critique_prompt.j2")
                prompt = template.render(
                    gravity_y=config.SIM_GRAVITY_Y,
                    elasticity=config.SIM_PROJECTILE_ELASTICITY,
                    target_bounce_number=config.TARGET_BOUNCE_NUMBER,
                    target_distance=config.TARGET_DISTANCE,
                    target_tolerance=config.TARGET_TOLERANCE,
                    ground_description=get_ground_description(ground_func),
                    history=history # Pass the whole history
                )
                expected_keys = ["critique", "height", "horizontal_velocity"]
                step_type = "critique"

            # 2. Call LLM
            raw_response = get_llm_response(self.model_identifier, prompt, self.temperature)
            parsed_data = parse_llm_json_output(raw_response)

            if not parsed_data or not all(key in parsed_data for key in expected_keys):
                print(f"Error: Failed to get valid {step_type} and parameters from LLM on iteration {iteration+1}.")
                final_error = history[-1]['error'] if history and 'error' in history[-1] else None
                success_flag = final_error is not None and final_error <= config.TARGET_TOLERANCE
                return {"success": success_flag, "error": f"LLM Parsing Failed Iter {iteration+1}", "history": history}

            current_h = parsed_data["height"]
            current_v = parsed_data["horizontal_velocity"]
            text_content = parsed_data.get("reasoning") or parsed_data.get("critique", "N/A")
            print(f"LLM {step_type.capitalize()}: h={current_h:.2f}m, v={current_v:.2f}m/s")
            print(f"LLM Text: {text_content}")

            history.append({
                "type": step_type,
                "text": text_content,
                "h": current_h,
                "v": current_v,
                "iteration": iteration + 1
            })

            # 3. Run Simulation
            bounce_locs = self._run_simulation(current_h, current_v, ground_func)
            actual_dist, error = calculate_error(bounce_locs, config.TARGET_BOUNCE_NUMBER, config.TARGET_DISTANCE)

            print(f"Simulation Result: Bounces={bounce_locs}, Target Bounce Dist={actual_dist:.2f}m, Error={error:.2f}m" if error is not None else "Simulation Result: Target bounce not reached.")

            history.append({
                "type": "simulation",
                "bounce_locs": bounce_locs,
                "actual_dist": actual_dist,
                "error": error,
                "iteration": iteration + 1
            })

            # 4. Check Success Condition (as per paper: LLM prompted to stop if requirements met)
            # We implement this check externally here.
            if error is not None and error <= config.TARGET_TOLERANCE:
                print(f"Success! Target achieved within tolerance at iteration {iteration + 1}.")
                break # Exit loop on success

        # End of loop or break
        end_time = time.time()
        final_error = history[-1]['error'] if history and 'error' in history[-1] else None
        final_dist = history[-1]['actual_dist'] if history and 'actual_dist' in history[-1] else None
        final_bounces = history[-1]['bounce_locs'] if history and 'bounce_locs' in history[-1] else []

        success_flag = final_error is not None and final_error <= config.TARGET_TOLERANCE

        result = {
            "success": success_flag,
            "strategy": "SimLM",
            "model": self.model_identifier,
            "ground": get_ground_description(ground_func),
            "iterations_run": len(history) // 3 + (1 if len(history)%3 > 0 else 0) , # Estimate based on steps stored
            "final_h": current_h,
            "final_v": current_v,
            "bounce_locations": final_bounces,
            "actual_distance_bounce_3": final_dist,
            "error": final_error,
            "within_tolerance": success_flag,
            "history": history,
            "time_taken": end_time - start_time,
        }
        print(f"Finished SimLM in {result['time_taken']:.2f}s. Final Error: {final_error:.2f}m" if final_error is not None else "Finished SimLM. Target bounce not reached.")
        return result