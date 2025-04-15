import numpy as np

# Ground Functions 
def flat_ground(x):
    """Experiment A"""
    return 0.0 * x

def sine_ground(x):
    """Experiment B: y = a*sin(f*x), a=1, f=1 (assuming x in meters)"""
    return 1.0 * np.sin(1.0 * x)

def easy_ground(x):
    """Experiment C - Easy surface"""
    return 0.15 * np.sin(0.25 * x)

def hard_ground(x):
    """Experiment C - Hard surface"""
    return 0.6 * np.sin(0.9 * x) + 0.15 * np.sin(2.25 * x) + 0.05 * np.sin(4.5 * x)

def interpolated_ground(difficulty):
    """Experiment C - Interpolated surface"""
    if not (0 <= difficulty <= 1):
        raise ValueError("Difficulty must be between 0 and 1")
    def ground_func(x):
        return (1 - difficulty) * easy_ground(x) + difficulty * hard_ground(x)
    return ground_func