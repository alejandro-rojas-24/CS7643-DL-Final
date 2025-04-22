from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray


class Ground(ABC):
    """Base class for all ground types."""

    @abstractmethod
    def height(
        self, x: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        """
        Return the height of the ground at position x.

        Args:
            x: Position or array of positions at which to calculate height

        Returns:
            Height(s) of the ground at given position(s)
        """
        pass

    @property
    def description(self) -> str:
        """Return a description of the ground type."""
        return "A generic ground type with no specific properties."

    def plot(
        self,
        x_min: float = -10,
        x_max: float = 10,
        num_points: int = 1000,
        title: str | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Visualize the ground profile.

        Args:
            x_min: Minimum x value to plot
            x_max: Maximum x value to plot
            num_points: Number of points to use for plotting
            title: Custom title for the plot (defaults to class name)

        Returns:
            Figure and axes objects for the plot
        """
        x = np.linspace(x_min, x_max, num_points)
        y = self.height(x)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y)
        ax.set_xlabel("x (meters)")
        ax.set_ylabel("Height (meters)")
        ax.set_title(title or self.__class__.__name__)
        ax.grid(True)

        return fig, ax

    def __call__(
        self, x: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        """
        Allow the ground to be called like a function.

        Args:
            x: Position or array of positions

        Returns:
            Height(s) at given position(s)
        """
        return self.height(x)

    def __repr__(self) -> str:
        """Return string representation of the ground object."""
        return self.__class__.__name__

    def __str__(self) -> str:
        """Return string representation of the ground object."""
        return self.__repr__()


class FlatGround(Ground):
    """
    Flat ground with zero height everywhere.

    Experiment A - Ground with constant zero height.
    """

    def height(
        self,
        x: float | NDArray[np.float64],
    ) -> float | NDArray[np.float64]:
        """
        Return height of zero for any x position.

        Args:
            x: Position or array of positions

        Returns:
            Zero height(s) at all position(s)
        """
        return 0.0 * x

    def __repr__(self) -> str:
        """Return string representation of the ground object."""
        return f"{self.__class__.__name__}()"


class SineGround(Ground):
    """
    Ground with sinusoidal profile.

    Experiment B: y = a*sin(f*x), a=1, f=1 (assuming x in meters)
    """

    @property
    def description(self) -> str:
        """Return a description of the ground type."""
        return f"A sinusoidal ground with amplitude {self.amplitude} and frequency {self.frequency}."

    def __init__(self, amplitude: float = 1.0, frequency: float = 1.0) -> None:
        """
        Initialize sinusoidal ground.

        Args:
            amplitude: Height amplitude of the sine wave
            frequency: Spatial frequency of the sine wave
        """
        self.amplitude = amplitude
        self.frequency = frequency

    def height(
        self, x: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        """
        Return the height of sinusoidal ground at position x.

        Args:
            x: Position or array of positions

        Returns:
            Height(s) at given position(s)
        """
        return self.amplitude * np.sin(self.frequency * x)

    def __repr__(self) -> str:
        """Return string representation with parameter values."""
        return f"{self.__class__.__name__}(amplitude={self.amplitude}, frequency={self.frequency})"


class EasyGround(Ground):
    """
    Simple sinusoidal ground with low amplitude and frequency.

    Experiment C - Easy surface.
    """

    @property
    def description(self) -> str:
        """Return a description of the ground type."""
        return f"A sinusoidal ground with amplitude {self.amplitude} and frequency {self.frequency}."

    def __init__(
        self,
        amplitude: float = 0.15,
        frequency: float = 0.25,
    ) -> None:
        """
        Initialize easy ground.

        Args:
            amplitude: Height amplitude of the sine wave (default: 0.15)
            frequency: Spatial frequency of the sine wave (default: 0.25)
        """
        self.amplitude = amplitude
        self.frequency = frequency

    def height(
        self, x: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        """
        Return the height of easy ground at position x.

        Args:
            x: Position or array of positions

        Returns:
            Height(s) at given position(s)
        """
        return self.amplitude * np.sin(self.frequency * x)

    def __repr__(self) -> str:
        """Return string representation with parameter values."""
        return f"{self.__class__.__name__}(amplitude={self.amplitude}, frequency={self.frequency})"


class HardGround(Ground):
    """
    Complex ground composed of multiple sine waves.

    Experiment C - Hard surface.
    """

    @property
    def description(self) -> str:
        """Return a description of the ground type."""
        return f"A complex ground with {len(self.amplitudes)} sine components."

    def __init__(
        self,
        amplitudes: list[float] | NDArray[np.float64] = np.array(
            [0.6, 0.15, 0.05]
        ),
        frequencies: list[float] | NDArray[np.float64] = np.array(
            [0.9, 2.25, 4.5]
        ),
    ) -> None:
        """
        Initialize hard ground with multiple sine components.

        Args:
            amplitudes: List of amplitudes for each sine component
            frequencies: List of frequencies for each sine component

        Raises:
            ValueError: If amplitudes and frequencies have different shapes
        """
        # Ensure amplitudes and frequencies are numpy arrays and have
        # the same shape
        if isinstance(amplitudes, list):
            amplitudes = np.array(amplitudes, dtype=np.float64)
        if isinstance(frequencies, list):
            frequencies = np.array(frequencies, dtype=np.float64)
        if amplitudes.shape != frequencies.shape:
            raise ValueError(
                "Amplitudes and frequencies must have the same shape"
            )
        self.amplitudes = amplitudes
        self.frequencies = frequencies

    def height(
        self, x: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        """
        Return the height of hard ground at position x.

        Args:
            x: Position or array of positions

        Returns:
            Height(s) at given position(s) as sum of sine components
        """
        return np.sum(
            self.amplitudes[:, np.newaxis]
            * np.sin(self.frequencies[:, np.newaxis] * x),
            axis=0,
        )

    def __repr__(self) -> str:
        """Return string representation with parameter values."""
        return f"{self.__class__.__name__}(amplitudes={self.amplitudes}, frequencies={self.frequencies})"


class InterpolatedGround(Ground):
    """
    Ground that interpolates between easy and hard profiles.

    Experiment C - Interpolated surface with configurable difficulty.
    """

    @property
    def description(self) -> str:
        """Return a description of the ground type."""
        return (
            "An interpolated ground with difficulty with a mixture of "
            f"{1 - self.difficulty} easy and {self.difficulty} hard surfaces.\n"
            f"Easy ground: {self.easy_ground.description}.\n"
            f"Hard ground: {self.hard_ground.description}."
        )

    def __init__(
        self,
        difficulty: float = 0.5,
        easy_ground: EasyGround | None = None,
        hard_ground: HardGround | None = None,
    ) -> None:
        """
        Initialize interpolated ground.

        Args:
            difficulty: Interpolation factor between easy (0) and hard (1)
            easy_ground: Easy ground instance to use (creates default if None)
            hard_ground: Hard ground instance to use (creates default if None)

        Raises:
            ValueError: If difficulty is not between 0 and 1
        """
        if not (0 <= difficulty <= 1):
            raise ValueError("Difficulty must be between 0 and 1")
        self.difficulty = difficulty
        self.easy_ground = easy_ground if easy_ground else EasyGround()
        self.hard_ground = hard_ground if hard_ground else HardGround()

    def height(
        self, x: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        """
        Return the height of interpolated ground at position x.

        Args:
            x: Position or array of positions

        Returns:
            Height(s) at given position(s) interpolated between easy and hard ground
        """
        return (1 - self.difficulty) * self.easy_ground.height(
            x
        ) + self.difficulty * self.hard_ground.height(x)

    def __repr__(self) -> str:
        """Return string representation with parameter values."""
        return f"{self.__class__.__name__}(difficulty={self.difficulty}, easy_ground={self.easy_ground}, hard_ground={self.hard_ground})"
