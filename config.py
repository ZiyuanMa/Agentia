import os
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

# =============================================================================
# File Paths
# =============================================================================

DEFAULT_WORLD_PATH = "data/world.json"
DEFAULT_AGENTS_PATH = "data/agents.json"
