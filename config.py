import os

from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEMPERATURE = 0.5
MODEL = "ollama/llama2"  # "google/gemini-2.0-flash-lite"  # "gpt-3.5-turbo"

# Simulation Configuration
SIM_FPS = 1000
SIM_GRAVITY_Y = -9.81  # m/s^2
SIM_GROUND_FRICTION = 0.8  # Unsure on specifics
SIM_PROJECTILE_ELASTICITY = 0.9
SIM_PROJECTILE_MASS = 1.0
SIM_PROJECTILE_RADIUS = 0.05  # meters
SIM_MAX_DURATION = 20  # seconds

# Experiment Configuration
TARGET_DISTANCE = 50.0
TARGET_BOUNCE_NUMBER = 3
TARGET_TOLERANCE = 1.0
SIMLM_MAX_ITERATIONS = 5

# Ground Generation Parameters
GROUND_X_MIN = -10.0  # meters
GROUND_X_MAX = 100.0  # meters
GROUND_STEP = 0.1  # meters

# Example/Data Loading
FEW_SHOT_EXAMPLES_PATH = "examples/few_shot_data.yaml"
