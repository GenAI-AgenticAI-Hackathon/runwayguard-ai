"""
RunwayGuard AI — Agentic Layer.

Implements native Groq tool/function calling:
  1. Accepts the deterministic risk score and detection payload.
  2. Forces the LLM to choose and invoke concrete Python functions via tool_choice="required".
  3. Executes the invoked tools and captures a structured, timestamped audit log.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

from groq import Groq

from src.config import AGENT_MODEL, GROQ_API_KEY

logger = logging.getLogger("runwayguard.agent")

_client: Groq | None = None

# Session audit log
audit_log: List[Dict[str, Any]] = []


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


# ── Real Executable Tool Functions ────────────────────────────────────────────

def alert_ground_crew(reason: str, location: str = "unknown") -> Dict[str, Any]:
    """Notify airport ground crew of a detected hazard."""
    msg = f"[ACTION EXECUTED] ALERT → Ground crew notified. Location: {location}. Reason: {reason}"
    logger.info(msg)
    return {"action": "alert_ground_crew", "success": True, "detail": msg}


def halt_traffic(reason: str) -> Dict[str, Any]:
    """Immediately suspend all active runway operations."""
    msg = f"[ACTION EXECUTED] HALT → ALL RUNWAY OPERATIONS SUSPENDED. Reason: {reason}"
    logger.warning(msg)
    return {"action": "halt_traffic", "success": True, "detail": msg}


def dispatch_sweep_team(location: str, object_description: str) -> Dict[str, Any]:
    """Dispatch runway sweeping vehicle to remove foreign object debris."""
    msg = f"[ACTION EXECUTED] DISPATCH → Sweep team en route to '{location}' for: {object_description}"
    logger.info(msg)
    return {"action": "dispatch_sweep_team", "success": True, "detail": msg}


def log_only(summary: str) -> Dict[str, Any]:
    """Log the incident into audit records without physical ground intervention."""
    msg = f"[ACTION EXECUTED] LOG → Incident recorded. Summary: {summary}"
    logger.info(msg)
    return {"action": "log_only", "success": True, "detail": msg}


# ── Tool Registry & Schemas ───────────────────────────────────────────────────

TOOL_FUNCTIONS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "alert_ground_crew": alert_ground_crew,
    "halt_traffic": halt_traffic,
    "dispatch_sweep_team": dispatch_sweep_team,
    "log_only": log_only,
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "alert_ground_crew",
            "description": "Notify ground crew of FOD. Use for MEDIUM risk (score 2.0-5.0).",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Description of hazard"},
                    "location": {"type": "string", "description": "Runway zone location"},
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "halt_traffic",
            "description": "Suspend all runway operations. Use for HIGH or CRITICAL risk (score >= 5.0).",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Reason for suspension"},
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispatch_sweep_team",
            "description": "Send physical sweep team to remove FOD. Use for LOW risk or in tandem with halt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Runway zone location"},
                    "object_description": {"type": "string", "description": "Description of the debris"},
                },
                "required": ["location", "object_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_only",
            "description": "Log scan without physical intervention. Use when risk category is CLEAR.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Summary of observations"},
                },
                "required": ["summary"],
            },
        },
    },
]


# ── Deterministic Fallback ────────────────────────────────────────────────────

def _deterministic_fallback(detection: Dict[str, Any], risk_score: Dict[str, Any]) -> Dict[str, Any]:
    """Guaranteed safety fallback if the LLM fails to return a valid tool call."""
    logger.warning("LLM did not return a tool call; executing deterministic fallback.")
    cat = risk_score.get("risk_category", "CLEAR")
    if cat in ("CRITICAL", "HIGH"):
        result = halt_traffic(reason=f"High-risk FOD detected: {detection.get('object_class', 'unknown')}")
    elif cat == "MEDIUM":
        result = alert_ground_crew(
            reason=f"FOD detected: {detection.get('object_class', 'unknown')}",
            location=detection.get("location_estimate", "unknown"),
        )
    elif cat == "LOW":
        result = dispatch_sweep_team(
            location=detection.get("location_estimate", "unknown"),
            object_description=detection.get("object_class", "unknown"),
        )
    else:
        result = log_only(summary="No FOD detected. Runway confirmed clear.")
    return {"tool": result["action"], "args": {}, "result": result, "via": "deterministic_fallback"}


# ── Decision Engine ───────────────────────────────────────────────────────────

def run_agent(detection: Dict[str, Any], risk_score: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pass detection and deterministic risk score to Groq native function calling,
    execute the chosen Python function(s), and record an audit log entry.

    Returns:
        Audit entry dictionary with executed actions.
    """
    user_prompt = (
        "FOD DETECTION REPORT — AUTONOMOUS SAFETY ACTION REQUIRED\n\n"
        f"  FOD Present:          {detection.get('fod_present')}\n"
        f"  Object Class:         {detection.get('object_class')}\n"
        f"  Vision Risk Level:    {detection.get('risk_raw')}\n"
        f"  Location on Runway:   {detection.get('location_estimate')}\n"
        f"  Vision Confidence:    {detection.get('confidence', 0.0):.0%}\n"
        f"  Deterministic Score:  {risk_score.get('numeric_score')} / 10.0\n"
        f"  Risk Category:        {risk_score.get('risk_category')}\n\n"
        "Select and invoke the appropriate tool function(s) immediately based on the above data."
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are RunwayGuard AI's Autonomous ATC Hazard Response Agent. "
                    "Your sole objective is to route verified detection states to actionable safety protocols via strict tool invocation.\n\n"
                    "CRITICAL CONSTRAINTS:\n"
                    "1. NO PLAIN TEXT: You MUST NOT output conversational text, explanations, or reasoning.\n"
                    "2. TOOL CALLS ONLY: You MUST ONLY respond by invoking the provided tool functions.\n"
                    "3. MULTI-ACTION EXECUTION: If a risk level requires multiple actions, you must call ALL required tools simultaneously in a single parallel batch.\n\n"
                    "RISK-TO-ACTION MATRIX (Execute Exactly):\n"
                    "- CLEAR    -> call: log_only\n"
                    "- LOW      -> call: dispatch_sweep_team\n"
                    "- MEDIUM   -> call: alert_ground_crew\n"
                    "- HIGH     -> call: halt_traffic AND dispatch_sweep_team\n"
                    "- CRITICAL -> call: halt_traffic AND alert_ground_crew AND dispatch_sweep_team"
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        tools=TOOL_SCHEMAS,
        tool_choice="required",
        max_tokens=512,
    )

    actions_executed: List[Dict[str, Any]] = []
    message = response.choices[0].message

    if message.tool_calls:
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            if tool_name in TOOL_FUNCTIONS:
                exec_result = TOOL_FUNCTIONS[tool_name](**tool_args)
            else:
                logger.error("Unknown tool requested: %s", tool_name)
                exec_result = {"action": tool_name, "success": False, "detail": "Unknown tool function."}

            actions_executed.append({
                "tool": tool_name,
                "args": tool_args,
                "result": exec_result,
                "via": "llm_tool_call",
            })
    else:
        actions_executed.append(_deterministic_fallback(detection, risk_score))

    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detection_input": {
            "fod_present": detection.get("fod_present"),
            "object_class": detection.get("object_class"),
            "location": detection.get("location_estimate"),
            "confidence": detection.get("confidence"),
            "vision_risk_raw": detection.get("risk_raw"),
            "source_file": detection.get("source_file", "n/a"),
        },
        "risk_score": risk_score,
        "actions_executed": actions_executed,
    }
    audit_log.append(audit_entry)
    logger.info("Agent executed %d action(s): %s", len(actions_executed), [a["tool"] for a in actions_executed])
    return audit_entry
