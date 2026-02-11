import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# API Configuration
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "mimo-v2-flash")

# =============================================================================
# Simulation Configuration
# =============================================================================

TICK_DURATION_MINUTES = 10

# Initial simulation time (Monday 8:00 AM)
SIMULATION_START_TIME = datetime(2024, 1, 1, 8, 0)

# Default agent status values
DEFAULT_AGENT_STATUS = {"fatigue": "low", "stress": "low"}

# =============================================================================
# File Paths
# =============================================================================

DEFAULT_SCENARIO_PATH = "data/scenario_office_escape.json"
