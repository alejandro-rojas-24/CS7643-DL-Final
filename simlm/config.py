from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Configuration for experiment parameters."""

    type: Literal["baseline_cot", "simlm"] = "baseline_cot"
    visualize: bool = True
    save_results: bool = True
    few_shot_examples_path: str | Path = "results/experiment_results-1260.jsonl"
    target_distance: float = 50.0
    target_bounce_number: int = 3
    tolerance: int = 1
    max_iterations: int = 5
    few_shot: int = 0

class LLMConfig(BaseModel):
    """Configuration for the language model."""

    service: Literal["openai", "ollama", "google"] = "google"
    model_name: str = "gemini-2.0-flash-lite"
    temperature: float = 0.0


class Coordinate(BaseModel):
    """Represents a 3D coordinate."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class SimulationConfig(BaseModel):
    """Configuration for physics simulation."""

    fps: int = 1000
    max_duration: int = 20
    gravity: Coordinate = Field(
        default_factory=lambda: Coordinate(x=0.0, y=-9.81, z=0.0)
    )


class ProjectileConfig(BaseModel):
    """Configuration for projectile properties."""

    elasticity: float = 0.9
    mass: float = 1.0
    radius: float = 0.05


class EasyGroundConfig(BaseModel):
    """Configuration for easy ground properties."""

    amplitude: float = 1
    frequency: float = 0.5


class HardGroundConfig(BaseModel):
    """Configuration for hard ground properties."""

    amplitudes: list[float] | float = [0.6, 0.15, 0.05]
    frequencies: list[float] | float = [0.9, 2.25, 4.5]


class GroundConfig(BaseModel):
    """Configuration for ground properties."""

    type: Literal["flat", "sine", "interpolated"] = "flat"
    friction: float = 0.8
    x_min: int = -200
    x_max: int = 400
    step: float = 0.1
    amplitude: float = 1
    frequency: float = 0.5
    difficulty: float = 0.9
    easy: EasyGroundConfig = Field(default_factory=EasyGroundConfig)
    hard: HardGroundConfig = Field(default_factory=HardGroundConfig)


class Config(BaseModel):
    """Main configuration class containing all config components."""

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    projectile: ProjectileConfig = Field(default_factory=ProjectileConfig)
    ground: GroundConfig = Field(default_factory=GroundConfig)

    @classmethod
    def from_file(cls, file_path: str) -> "Config":
        """
        Load configuration from a YAML file.

        Args:
          file_path: Path to the YAML configuration file

        Returns:
          Config object with all configuration parameters

        Raises:
          FileNotFoundError: If the configuration file does not exist
          ValueError: If the configuration file has invalid format
        """
        try:
            with open(file_path, "r") as file:
                config_data = yaml.safe_load(file)

            if not config_data:
                return Config()

            return cls.model_validate(config_data)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: {file_path}"
            )
        except yaml.YAMLError:
            raise ValueError(
                f"Invalid YAML format in configuration file: {file_path}"
            )
        except ValueError as e:
            raise ValueError(f"Invalid configuration structure: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {str(e)}")

    def to_file(self, file_path: str) -> None:
        """
        Save the configuration to a YAML file.

        Args:
          file_path: Path to save the YAML configuration file

        Raises:
          IOError: If there is an error writing to the file
        """
        try:
            with open(file_path, "w") as file:
                yaml.dump(self.model_dump(), file)
        except IOError as e:
            raise IOError(f"Failed to save configuration: {str(e)}")
