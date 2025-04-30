import json
import random
from pathlib import Path

def get_nested_value(data_dict, key_list, default=None):
    val = data_dict
    try:
        for key in key_list:
            val = val[key]
        return val
    except (KeyError, TypeError, IndexError):
        return default


def load_examples(filepath, strategy_type, num_examples=3, success_only=True, filter_dict=None):
    """
    Loads and formats few-shot examples from a JSON Lines results file.

    Args:
        filepath (str or Path): Path to the JSON Lines results file.
        strategy_type (str): The type of examples to format ('cot' or 'simlm').
        num_examples (int): The maximum number of examples to return.
        success_only (bool): If True, only load examples marked as successful.
        filter_dict (dict, optional): A dictionary to filter examples based on
                                     key-value pairs (supports nested keys via dot notation,
                                     e.g., {'config.ground.type': 'flat'}). Defaults to None.

    Returns:
        list: A list of formatted example dictionaries, ready for Jinja2 templates,
              or an empty list if no suitable examples are found.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        print(f"Warning: Example file not found at {filepath}")
        return []

    potential_examples = []
    print(f"\nLoading examples from: {filepath} for strategy: {strategy_type}")
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                try:
                    run_data = json.loads(line)

                    # Filter by Success
                    if success_only and not run_data.get('success', False):
                        continue

                    # Filter by specified criteria in filter_dict
                    match = True
                    if filter_dict:
                        for key_path, expected_value in filter_dict.items():
                            # Handle nested keys specified with dots
                            keys = key_path.split('.')
                            actual_value = get_nested_value(run_data, keys)
                            if actual_value != expected_value:
                                match = False
                                break
                    if not match:
                        continue

                    # Format based on strategy type
                    formatted_example = None
                    # Generate a generic query based on config 
                    target_bounce = get_nested_value(run_data, ['config', 'experiment', 'target_bounce_number'], 3)
                    target_dist = get_nested_value(run_data, ['config', 'experiment', 'target_distance'], 50.0)
                    ground_type = get_nested_value(run_data, ['config', 'ground', 'type'], 'unknown').title()
                    ground_difficulty = get_nested_value(run_data, ['config', 'ground', 'difficulty'])
                    ground_desc = f"{ground_type} ground"

                    query = (f"Find h and v for the {target_bounce}{'th' if 11<=target_bounce<=13 else {1:'st', 2:'nd', 3:'rd'}.get(target_bounce%10, 'th')} "
                             f"bounce to land near {target_dist}m on {ground_desc}.")


                    # Extract final H and V
                    final_h = run_data.get('final_h')
                    final_v = run_data.get('final_v')

                    if final_h is None or final_v is None:
                         print(f"Warning: Skipping example on line {i+1} due to missing final_h or final_v.")
                         continue 

                    # Formatting for CoT
                    if strategy_type == 'baseline_cot':
                        reasoning = run_data.get('reasoning', 'No reasoning provided.')
                        answer_dict = {"height": float(final_h), "horizontal_velocity": float(final_v)}
                        answer_json_str = json.dumps(answer_dict, indent=2)

                        formatted_example = {
                            "query": query,
                            "reasoning": reasoning,
                            "answer_json": answer_json_str
                        }

                    # Formatting for SimLM
                    elif strategy_type == 'simlm':
                        history = run_data.get('history')
                        # Need to ensure history format matches template expectation if different
                        if not history or not isinstance(history, list):
                             print(f"Warning: Skipping SimLM example on line {i+1} due to missing or invalid 'history'.")
                             continue # Skip if history is missing/invalid for SimLM

                        # Ensure h and v are formatted correctly for JSON
                        final_answer_dict = {"height": float(final_h), "horizontal_velocity": float(final_v)}
                        final_answer_json_str = json.dumps(final_answer_dict, indent=2)

                        formatted_example = {
                            "query": query,
                            "history": history, # Assumes history is already structured list of dicts
                            "final_answer_json": final_answer_json_str,
                            "target": target_dist # Pass target distance for context
                        }

                    if formatted_example:
                        potential_examples.append(formatted_example)

                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line {i+1} in example file: {e}")
                except Exception as e:
                    print(f"Warning: Error processing line {i+1} in example file: {e}")


    except FileNotFoundError:
        print(f"Error: Example file not found at {filepath}")
        return []
    except Exception as e:
         print(f"Error reading example file {filepath}: {e}")
         return []

    # Select random subset if more examples found than needed
    if len(potential_examples) > num_examples:
        selected_examples = random.sample(potential_examples, num_examples)
        print(f"Selected {num_examples} random examples from {len(potential_examples)} candidates.")
    else:
        selected_examples = potential_examples
        print(f"Loaded {len(selected_examples)} examples.")

    return selected_examples