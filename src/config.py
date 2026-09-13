"""
Runway Sentinel AI — Centralised Configuration & Fail-Fast Startup.

All other src/ modules import from here. If GROQ_API_KEY is missing
and RUNWAYGUARD_MOCK is not set, this module raises immediately at
import time so misconfigured deployments are caught instantly.

Dual key-resolution strategy:
  1. Read os.environ ("GROQ_API_KEY")  — local .env / Dockerfile ENV
  2. Fall back to st.secrets            — Streamlit Community Cloud
"""
import os
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency guard
    load_dotenv = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("runwaysentinel.config")


def _load_local_env_file() -> None:
    """Load .env from the active working directory, preserving existing env vars."""
    env_file = Path.cwd() / ".env"
    if not env_file.exists():
        return

    if load_dotenv is not None:
        load_dotenv(env_file, override=False)
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env_file()

# ── Demo / CI override ─────────────────────────────────────────────────────────
# Set RUNWAYGUARD_MOCK=true to run the full UI with synthetic data.
# Useful for Streamlit Community Cloud demo mode and offline CI runs.
USE_MOCK: bool = os.getenv("RUNWAYGUARD_MOCK", "false").lower() in ("true", "1", "yes")

# ── Model selection (overridable via env) ──────────────────────────────────────
VISION_MODEL: str = os.getenv(
    "VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "llama3-70b-8192")
REPORT_MODEL: str = os.getenv("REPORT_MODEL", "llama3-70b-8192")


def _get_api_key() -> str:
    """
    Resolve GROQ_API_KEY from environment variables or Streamlit secrets.

    Resolution order:
      1. os.environ["GROQ_API_KEY"]   — always wins (local .env, Dockerfile)
      2. st.secrets["GROQ_API_KEY"]   — Streamlit Community Cloud dashboard
    """
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        try:
            import streamlit as st  # noqa: PLC0415  (lazy import intentional)

            key = (st.secrets.get("GROQ_API_KEY") or "").strip()
        except Exception:
            # Streamlit not installed, or called outside a Streamlit process
            pass
    return key


# ── API Key — Fail-safe resolution ────────────────────────────────────────────
NO_API_KEY_SET: bool = False

if USE_MOCK:
    GROQ_API_KEY: str = "mock-key-not-used"
    logger.warning(
        "Runway Sentinel AI is running in MOCK mode (RUNWAYGUARD_MOCK=true). "
        "No real Groq API calls will be made."
    )
else:
    GROQ_API_KEY = _get_api_key()
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is required when RUNWAYGUARD_MOCK is not enabled. "
            "Set GROQ_API_KEY in the environment or Streamlit Cloud secrets."
        )

    logger.info(
        "Config loaded | Vision: %s | Agent: %s | Report: %s",
        VISION_MODEL,
        AGENT_MODEL,
        REPORT_MODEL,
    )

