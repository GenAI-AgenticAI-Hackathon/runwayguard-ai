"""
Runway Sentinel AI — Comprehensive Pipeline Unit Tests.

All external Groq API calls are mocked using unittest.mock to ensure:
  - 100% offline test execution.
  - Zero API token consumption during CI/CD workflows.
  - Deterministic and reproducible outcomes.
"""
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_clear_detection():
    return {
        "fod_present": False,
        "object_class": "none",
        "risk_raw": "LOW",
        "location_estimate": "none",
        "confidence": 0.99,
        "source_file": "clear_runway.jpg",
    }


@pytest.fixture
def sample_critical_detection():
    return {
        "fod_present": True,
        "object_class": "metal debris",
        "risk_raw": "HIGH",
        "location_estimate": "touchdown zone center",
        "confidence": 0.95,
        "source_file": "hazard_frame_042.jpg",
    }


@pytest.fixture
def sample_medium_detection():
    return {
        "fod_present": True,
        "object_class": "cone",
        "risk_raw": "MEDIUM",
        "location_estimate": "runway edge left",
        "confidence": 0.85,
        "source_file": "edge_cone.jpg",
    }


# ── Test Suite: Risk Assessment (Pure Deterministic Python) ───────────────────

class TestRiskAssessment:
    def test_clear_detection_yields_zero_risk(self, sample_clear_detection):
        from src.risk import compute_risk_score
        result = compute_risk_score(sample_clear_detection)
        assert result["numeric_score"] == 0.0
        assert result["risk_category"] == "CLEAR"
        assert result["breakdown"]["fod_present"] is False

    def test_critical_detection_yields_high_score(self, sample_critical_detection):
        from src.risk import compute_risk_score
        result = compute_risk_score(sample_critical_detection)
        assert result["numeric_score"] >= 7.5
        assert result["risk_category"] == "CRITICAL"
        assert "formula" in result["breakdown"]

    def test_medium_detection_categorization(self, sample_medium_detection):
        from src.risk import compute_risk_score
        result = compute_risk_score(sample_medium_detection)
        assert 2.0 <= result["numeric_score"] < 7.5
        assert result["risk_category"] in ("MEDIUM", "HIGH")

    def test_score_never_exceeds_ten(self):
        from src.risk import compute_risk_score
        extreme_detection = {
            "fod_present": True,
            "object_class": "metal debris",
            "risk_raw": "HIGH",
            "location_estimate": "touchdown zone",
            "confidence": 1.0,
        }
        result = compute_risk_score(extreme_detection)
        assert result["numeric_score"] <= 10.0


# ── Test Suite: Detection (Mocked Groq Vision) ────────────────────────────────

class TestDetection:
    def _create_mock_chat_completion(self, content_str: str) -> MagicMock:
        mock_msg = MagicMock()
        mock_msg.content = content_str
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    @patch("src.detection.Groq")
    def test_detect_fod_valid_payload(self, mock_groq_class, tmp_path):
        from src.detection import detect_fod
        import src.detection as det_mod

        # Reset cached singleton
        det_mod._client = None

        dummy_img = tmp_path / "runway.jpg"
        dummy_img.write_bytes(b"dummy-image-binary")

        raw_json = json.dumps({
            "fod_present": True,
            "object_class": "metal debris",
            "risk_raw": "HIGH",
            "location_estimate": "touchdown zone center",
            "confidence": 0.92,
        })
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._create_mock_chat_completion(raw_json)
        mock_groq_class.return_value = mock_client

        output = detect_fod(str(dummy_img))
        assert output["fod_present"] is True
        assert output["object_class"] == "metal debris"
        assert output["risk_raw"] == "HIGH"
        assert output["confidence"] == 0.92
        assert output["source_file"] == "runway.jpg"

    @patch("src.detection.Groq")
    def test_detect_fod_strips_markdown_fences(self, mock_groq_class, tmp_path):
        from src.detection import detect_fod
        import src.detection as det_mod

        det_mod._client = None
        dummy_img = tmp_path / "fence.jpg"
        dummy_img.write_bytes(b"dummy-bytes")

        fenced_payload = "```json\n{\"fod_present\": false, \"object_class\": \"none\", \"risk_raw\": \"LOW\", \"location_estimate\": \"none\", \"confidence\": 0.98}\n```"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._create_mock_chat_completion(fenced_payload)
        mock_groq_class.return_value = mock_client

        output = detect_fod(str(dummy_img))
        assert output["fod_present"] is False
        assert output["object_class"] == "none"


# ── Test Suite: Agentic Layer (Mocked Groq Tool Calling) ──────────────────────

class TestAgent:
    def _create_mock_tool_completion(self, tool_name: str, tool_args: dict) -> MagicMock:
        tool_call = MagicMock()
        tool_call.function.name = tool_name
        tool_call.function.arguments = json.dumps(tool_args)
        mock_msg = MagicMock()
        mock_msg.tool_calls = [tool_call]
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    @patch("src.agent.Groq")
    def test_agent_executes_halt_traffic_tool(self, mock_groq_class, sample_critical_detection):
        from src.agent import run_agent
        from src.risk import compute_risk_score
        import src.agent as agent_mod

        agent_mod._client = None
        risk = compute_risk_score(sample_critical_detection)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._create_mock_tool_completion(
            "halt_traffic", {"reason": "Metal debris on active touchdown zone"}
        )
        mock_groq_class.return_value = mock_client

        audit = run_agent(sample_critical_detection, risk)
        assert len(audit["actions_executed"]) == 1
        action = audit["actions_executed"][0]
        assert action["tool"] == "halt_traffic"
        assert action["result"]["success"] is True
        assert "timestamp" in audit

    @patch("src.agent.Groq")
    def test_agent_fallback_mechanism(self, mock_groq_class, sample_critical_detection):
        from src.agent import run_agent
        from src.risk import compute_risk_score
        import src.agent as agent_mod

        agent_mod._client = None
        risk = compute_risk_score(sample_critical_detection)

        # Mock LLM returning empty text instead of tool call
        mock_msg = MagicMock()
        mock_msg.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_groq_class.return_value = mock_client

        audit = run_agent(sample_critical_detection, risk)
        assert len(audit["actions_executed"]) == 1
        action = audit["actions_executed"][0]
        assert action["via"] == "deterministic_fallback"
        assert action["tool"] == "halt_traffic"


# ── Test Suite: Reporting (Mocked Grounded Generation) ────────────────────────

class TestReporting:
    @patch("src.reporting.Groq")
    def test_generate_report_sections(self, mock_groq_class, sample_critical_detection):
        from src.reporting import generate_report
        import src.reporting as report_mod

        report_mod._client = None
        mock_report_body = (
            "## 1. DETECTION SUMMARY\nMetal debris detected in touchdown zone.\n\n"
            "## 2. RISK ASSESSMENT\nDeterministic score: 8.74/10 (CRITICAL).\n\n"
            "## 3. ACTIONS TAKEN\nTool halt_traffic() invoked immediately."
        )
        mock_msg = MagicMock()
        mock_msg.content = mock_report_body
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_groq_class.return_value = mock_client

        mock_risk = {"numeric_score": 8.74, "risk_category": "CRITICAL", "breakdown": {}}
        mock_audit = {
            "timestamp": "2026-09-12T09:00:00Z",
            "actions_executed": [{
                "tool": "halt_traffic",
                "args": {"reason": "Metal debris"},
                "result": {"success": True, "detail": "Traffic halted"},
            }],
        }

        report = generate_report(sample_critical_detection, mock_risk, mock_audit)
        assert "DETECTION SUMMARY" in report
        assert "RISK ASSESSMENT" in report
        assert "ACTIONS TAKEN" in report
        assert "RUNWAYGUARD AI" in report or "RUNWAY" in report


# ── Test Suite: Fail-Fast Config Validation ───────────────────────────────────

class TestConfigFailFast:
    def test_config_raises_when_api_key_missing_and_not_mock(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("RUNWAYGUARD_MOCK", "false")

        # Evict src.config from sys.modules to force module re-execution
        for mod in list(sys.modules.keys()):
            if "src.config" in mod:
                del sys.modules[mod]

        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            import src.config  # noqa: F401

    def test_config_succeeds_in_mock_mode_without_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("RUNWAYGUARD_MOCK", "true")

        for mod in list(sys.modules.keys()):
            if "src.config" in mod:
                del sys.modules[mod]

        import src.config as cfg
        assert cfg.USE_MOCK is True
        assert cfg.GROQ_API_KEY == "mock-key-not-used"


# ── Test Suite: Streamlit Command Center Architecture ─────────────────────────

class TestStreamlitAppStructure:
    def test_mock_scenarios_schema(self):
        from app import MOCK_SCENARIOS
        assert len(MOCK_SCENARIOS) >= 4
        for s in MOCK_SCENARIOS:
            assert "fod_present" in s
            assert "object_class" in s
            assert "risk_category" in s
            assert "numeric_score" in s
            assert "action" in s
            assert "confidence" in s

    def test_app_colors_and_helpers(self):
        from app import risk_color, action_color, action_panel_class
        assert risk_color("CRITICAL") == "#ef4444"
        assert risk_color("CLEAR") == "#10b981"
        assert action_color("halt_traffic") == "#ef4444"
        assert action_panel_class("halt_traffic") == "action-panel-red"
