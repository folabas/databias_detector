"""Configuration for backend services.

Centralizes environment variables for external services and server URLs.
"""
from typing import Optional
import os

# Attempt to load variables from a project-level .env file if present
try:
    from dotenv import load_dotenv  # type: ignore
    _ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _ENV_PATH = os.path.join(_ROOT_DIR, ".env")
    if os.path.exists(_ENV_PATH):
        load_dotenv(_ENV_PATH)
    else:
        # Fall back to default search (current working dir)
        load_dotenv()
except Exception:
    # If python-dotenv isn't installed or any error occurs, just skip
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

settings = Settings()