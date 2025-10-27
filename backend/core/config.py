"""Configuration for backend services.

Centralizes environment variables for external services and server URLs.
"""
from typing import Optional
import os
 
 
try:
    from dotenv import load_dotenv  # type: ignore
    _ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _ENV_PATH = os.path.join(_ROOT_DIR, ".env")
    if os.path.exists(_ENV_PATH):
        load_dotenv(_ENV_PATH)
    else:
        load_dotenv()
except Exception:
    pass

class Settings:
    """Simple settings container using environment variables.

    - HUGGINGFACE_API_TOKEN: optional token for Hugging Face Inference
    - OLLAMA_BASE_URL: optional base URL for local Ollama server
    - BACKEND_URL: used by frontend to resolve backend (fallback only)
    """

    HUGGINGFACE_API_TOKEN: Optional[str] = os.environ.get("HUGGINGFACE_API_TOKEN")
    OLLAMA_BASE_URL: Optional[str] = os.environ.get("OLLAMA_BASE_URL")
    BACKEND_URL: str = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

    # --- Suggestion thresholds (configurable; defaults mirror previous heuristics) ---
    SPD_THRESHOLD: float = float(os.environ.get("SPD_THRESHOLD", "0.1"))
    DIR_MIN_RATIO: float = float(os.environ.get("DIR_MIN_RATIO", "0.8"))
    PED_THRESHOLD: float = float(os.environ.get("PED_THRESHOLD", "0.05"))
    GROUP_RATIO_MIN: float = float(os.environ.get("GROUP_RATIO_MIN", "0.5"))
    TARGET_RATIO_MIN: float = float(os.environ.get("TARGET_RATIO_MIN", "0.6"))

    # Toggle to include general process tips in suggestions
    ALWAYS_ON_TIPS: bool = os.environ.get("ALWAYS_ON_TIPS", "true").lower() in ("1", "true", "yes", "y")

    # --- Metric toggles and integration flags ---
    USE_FAIRLEARN: bool = os.environ.get("USE_FAIRLEARN", "false").lower() in ("1", "true", "yes", "y")
    INCLUDE_EQUAL_OPPORTUNITY: bool = os.environ.get("INCLUDE_EQUAL_OPPORTUNITY", "true").lower() in ("1", "true", "yes", "y")
    INCLUDE_EQUALIZED_ODDS: bool = os.environ.get("INCLUDE_EQUALIZED_ODDS", "true").lower() in ("1", "true", "yes", "y")
    INCLUDE_DEMOGRAPHIC_PARITY: bool = os.environ.get("INCLUDE_DEMOGRAPHIC_PARITY", "true").lower() in ("1", "true", "yes", "y")

    WEIGHT_SPD: float = float(os.environ.get("WEIGHT_SPD", "0.3"))
    WEIGHT_DIR: float = float(os.environ.get("WEIGHT_DIR", "0.25"))
    WEIGHT_PED: float = float(os.environ.get("WEIGHT_PED", "0.2"))
    WEIGHT_EO: float = float(os.environ.get("WEIGHT_EO", "0.15"))
    WEIGHT_EQODDS: float = float(os.environ.get("WEIGHT_EQODDS", "0.1"))
    WEIGHT_DP: float = float(os.environ.get("WEIGHT_DP", "0.1"))

settings = Settings()