"""
RunwayGuard AI — Reporting Module.

Generates a formal, hallucination-free aviation incident report
AFTER detection, risk assessment, and agent action have executed.
Strictly grounded in verified system outputs.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from groq import Groq

from src.config import GROQ_API_KEY, REPORT_MODEL

logger = logging.getLogger("runwayguard.reporting")

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def format_action_list(actions_executed: List[Dict[str, Any]]) -> str:
    """Format executed agent actions into grounded text for report generation."""
    if not actions_executed:
        return "  (No agent actions recorded)"
    lines = []
    for idx, act in enumerate(actions_executed, 1):
        lines.append(
            f"  {idx}. Tool Invoked: {act['tool']}()\n"
            f"     Arguments: {act.get('args', {})}\n"
            f"     Status: {'SUCCESS' if act['result'].get('success') else 'FAILED'}\n"
            f"     Execution Detail: {act['result'].get('detail', 'N/A')}"
        )
    return "\n".join(lines)


def generate_report(detection: Dict[str, Any], risk_score: Dict[str, Any], audit_entry: Dict[str, Any]) -> str:
    """
    Synthesize an aviation incident report from verified detection and action data.

    Args:
        detection: Output from detect_fod()
        risk_score: Output from compute_risk_score()
        audit_entry: Output from run_agent()

    Returns:
        Formatted incident report text string.
    """
    actions_summary = format_action_list(audit_entry.get("actions_executed", []))

    grounding_data = (
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║     RUNWAYGUARD AI — GROUNDED OPERATIONAL INCIDENT DATA      ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n\n"
        f"SOURCE IDENTIFIER : {detection.get('source_file', 'n/a')}\n"
        f"TIMESTAMP (UTC)   : {audit_entry.get('timestamp', 'unknown')}\n\n"
        "─── 1. COMPUTER VISION SENSOR DATA ────────────────────────────\n"
        f"  FOD Present         : {detection.get('fod_present')}\n"
        f"  Identified Object   : {detection.get('object_class')}\n"
        f"  Raw Vision Risk     : {detection.get('risk_raw')}\n"
        f"  Estimated Location  : {detection.get('location_estimate')}\n"
        f"  Detection Confidence: {detection.get('confidence', 0.0):.0%}\n\n"
        "─── 2. DETERMINISTIC RISK EVALUATION ──────────────────────────\n"
        f"  Calculated Score    : {risk_score.get('numeric_score')} / 10.0\n"
        f"  Risk Classification : {risk_score.get('risk_category')}\n"
        f"  Mathematical Model  : {risk_score.get('breakdown', {}).get('formula', 'N/A')}\n"
        f"  Raw Vision Component: {risk_score.get('breakdown', {}).get('raw_component', 'N/A')}\n"
        f"  Object Weight       : {risk_score.get('breakdown', {}).get('object_weight', 'N/A')}\n"
        f"  Location Weight     : {risk_score.get('breakdown', {}).get('location_weight', 'N/A')}\n"
        f"  Confidence Factor   : {risk_score.get('breakdown', {}).get('confidence', 'N/A')}\n\n"
        "─── 3. AGENTIC ACTIONS EXECUTED ───────────────────────────────\n"
        f"{actions_summary}\n"
        "════════════════════════════════════════════════════════════════\n"
    )

    prompt = (
        f"{grounding_data}\n"
        "Using ONLY the verified operational data above, generate a formal aviation safety "
        "incident report formatted into three numbered sections:\n\n"
        "## 1. DETECTION SUMMARY\n"
        "Detail the target object, location, and confidence level from the sensor data.\n\n"
        "## 2. RISK ASSESSMENT\n"
        "Detail the computed numeric risk score, categorical rating, and deterministic factors.\n\n"
        "## 3. ACTIONS TAKEN\n"
        "State every executed tool function, arguments, and real-world operational outcome.\n\n"
        "CRITICAL CONSTRAINTS:\n"
        "- Do NOT invent, assume, or extrapolate details not present in the data block.\n"
        "- Do NOT use speculative terms (e.g., 'might', 'likely', 'could be').\n"
        "- Maintain formal, standard aviation safety technical language."
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=REPORT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the RunwayGuard Aviation Safety Incident Reporting Specialist. "
                    "Your role is to synthesize raw detection telemetry, deterministic risk evaluations, "
                    "and executed agent tool-calls into formal, ICAO/FAA-compliant incident reports.\n\n"
                    "CONSTRAINTS:\n"
                    "1. ZERO HALLUCINATION: You must only include data explicitly provided in the input logs. "
                    "If a detail (like weather or flight numbers) is missing, explicitly state \"Data Unavailable\". Do not invent scenarios.\n"
                    "2. MATHEMATICAL RISK GROUNDING: You must calculate and verify the final risk score using the exact mathematical formula provided in your system context. "
                    "The formula is: Numeric Score = Raw Vision Component * Object Weight * Location Weight * Confidence * (10/3). "
                    "You must show the step-by-step mathematical execution using the explicitly provided variables in your report.\n"
                    "3. TONE: Objective, clinical, and strictly professional.\n\n"
                    "REQUIRED REPORT STRUCTURE:\n"
                    "- Incident ID & Timestamp\n"
                    "- FOD Classification & Sensor Confidence\n"
                    "- Location Telemetry\n"
                    "- Mathematical Risk Assessment (Show formula application)\n"
                    "- Automated Actions Executed\n"
                    "- Summary Statement"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=650,
    )

    body = response.choices[0].message.content.strip()
    header = (
        f"{'=' * 64}\n"
        f"  RUNWAYGUARD AI — AVIATION INCIDENT REPORT\n"
        f"  Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f"  Source    : {detection.get('source_file', 'n/a')}\n"
        f"{'=' * 64}\n\n"
    )
    return header + body
