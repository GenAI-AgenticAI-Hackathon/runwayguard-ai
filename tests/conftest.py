"""
pytest Configuration & Test Environment Bootstrap.

Sets dummy environment variables before any tests import src.config,
ensuring fail-fast import assertions pass cleanly during automated testing.
"""
import os
import pytest

# Inject test credentials before any test imports src.config
os.environ["GROQ_API_KEY"] = "gsk_test_mock_key_for_offline_ci"
os.environ["RUNWAYGUARD_MOCK"] = "false"
os.environ["VISION_MODEL"] = "meta-llama/llama-4-scout-17b-16e-instruct"
os.environ["AGENT_MODEL"] = "llama3-70b-8192"
os.environ["REPORT_MODEL"] = "llama3-70b-8192"
