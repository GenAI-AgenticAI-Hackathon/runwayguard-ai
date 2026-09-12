"""
RunwayGuard AI — Centralised Configuration & Fail-Fast Startup.

All other src/ modules import from here. If GROQ_API_KEY is missing
and RUNWAYGUARD_MOCK is not enabled, this module raises immediately at
import time so misconfigured deployments fail loudly and fast.
"""
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("runwayguard.config")

# ── Demo / Offline Mock Mode ──────────────────────────────────────────────────
# Set RUNWAYGUARD_MOCK=true to run the system with synthetic scenarios.
USE_MOCK: bool = os.getenv("RUNWAYGUARD_MOCK", "false").lower() in ("true", "1", "yes")

# ── Model Configuration ───────────────────────────────────────────────────────
VISION_MODEL: str = os.getenv(
    "VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "llama3-70b-8192")
REPORT_MODEL: str = os.getenv("REPORT_MODEL", "llama3-70b-8192")

# ── API Key Validation (Fail-Fast) ───────────────────────────────────────────
if USE_MOCK:
    GROQ_API_KEY: str = "mock-key-not-used"
    logger.warning("RunwayGuard AI running in MOCK mode (RUNWAYGUARD_MOCK=true). No live Groq API calls will be made.")
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
    if not GROQ_API_KEY:
        raise ValueError(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  CRITICAL: GROQ_API_KEY environment variable is not set!     ║\n"
            "║                                                              ║\n"
            "║  RunwayGuard AI requires a valid Groq API key to function.    ║\n"
            "║                                                              ║\n"
            "║  Resolution:                                                 ║\n"
            "║    1. Copy .env.example to .env and set your GROQ_API_KEY.   ║\n"
            "║    2. Or export GROQ_API_KEY='your-key-here'.                ║\n"
            "║    3. Or export RUNWAYGUARD_MOCK=true for offline demo mode. ║\n"
            "╚══════════════════════════════════════════════════════════════╝"
        )
    logger.info("RunwayGuard AI configuration loaded. Vision Model: %s | Agent Model: %s", VISION_MODEL, AGENT_MODEL)
