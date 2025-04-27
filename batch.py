from main import run_experiment, save_results
from simlm.config import (
    Config,
    ExperimentConfig,
    LLMConfig,
    GroundConfig,
)
from itertools import product


def main() -> None:
    """Run the batch experiments."""
    experiment_types = ["baseline_cot", "simlm"]
    models = [
        ("openai", "gpt-3.5-turbo"),
        ("openai", "gpt-4.1-nano"),
        ("google", "gemini-2.0-flash"),
        ("google", "gemini-2.5-flash-preview-04-17"),
        ("google", "gemini-1.5-flash"),
        ("ollama", "gemma3:1b"),
        ("ollama", "gemma3:4b"),
        # ("ollama", "gemma3:12b"),
        # ("ollama", "gemma3:27b"),
        # ("ollama", "llama3.2:1b"),
        # ("ollama", "llama3.2:3b"),
        # ("ollama", "llama3.1:8b"),
    ]
    flat_grounds = [{"type": "flat"}]
    sine_grounds = [{"type": "sine", "amplitude": 0.5, "frequency": 1.0}]
    interpolated_grounds = [
        # {
        #     "type": "interpolated",
        #     "difficulty": difficulty,
        #     "easy": {"amplitude": 0.15, "frequency": 0.25},
        #     "hard": {
        #         "amplitudes": [0.6, 0.15, 0.05],
        #         "frequencies": [0.9, 2.25, 4.5],
        #     },
        # }
        # for difficulty in np.arange(0.0, 1.1, 0.1)
    ]
    grounds = flat_grounds + sine_grounds + interpolated_grounds

    configs = []

    for experiment_type, (model_service, model_name), ground in product(
        experiment_types,
        models,
        grounds,
    ):
        config = Config(
            experiment=ExperimentConfig(
                type=experiment_type,
                visualize=False,
                save_results=True,
            ),
            llm=LLMConfig(
                service=model_service,
                model_name=model_name,
                temperature=0.0,
            ),
            ground=GroundConfig(**ground),
        )
        configs.append(config)
        print(config)

    for config in configs:
        print(config)
        runner, results = run_experiment(config)
        save_results(results)


if __name__ == "__main__":
    main()
