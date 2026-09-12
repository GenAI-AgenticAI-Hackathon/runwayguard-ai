"""
RunwayGuard AI — Risk Assessment Module.

PURELY DETERMINISTIC. Zero LLM involvement. Zero randomness.
All scoring is an explicit, inspectable Python mathematical formula.
Every weight table and threshold can be audited, tested, and fine-tuned.
"""
import logging
from typing import Any, Dict

logger = logging.getLogger("runwayguard.risk")

# ── Inspectable Weight Tables ─────────────────────────────────────────────────

# Threat severity by object class (0.0 = harmless, 1.0 = catastrophic)
OBJECT_CLASS_WEIGHTS: Dict[str, float] = {
    "none": 0.00,
    "paper": 0.20,
    "plastic bag": 0.30,
    "bird": 0.50,
    "cone": 0.55,
    "luggage": 0.65,
    "tire fragment": 0.80,
    "tool": 0.85,
    "metal debris": 1.00,
    "unknown": 0.60,
}

# Base numeric score for raw vision model risk assessment
RAW_RISK_BASE: Dict[str, float] = {
    "LOW": 1.0,
    "MEDIUM": 2.0,
    "HIGH": 3.0,
}

# Criticality based on location on runway (active centerline & touchdown zone highest)
LOCATION_CENTRALITY: Dict[str, float] = {
    "touchdown zone": 1.00,
    "center": 0.90,
    "threshold": 0.85,
    "runway": 0.75,
    "runway edge": 0.40,
    "taxiway": 0.35,
    "apron": 0.20,
    "none": 0.00,
}


def get_object_weight(object_class: str) -> float:
    """Retrieve object weight from table, fallback to default unknown weight."""
    obj = object_class.lower()
    for key, weight in OBJECT_CLASS_WEIGHTS.items():
        if key in obj:
            return weight
    return OBJECT_CLASS_WEIGHTS["unknown"]


def get_location_weight(location_estimate: str) -> float:
    """Retrieve location weight based on proximity to active runway zones."""
    loc = location_estimate.lower()
    for key, weight in LOCATION_CENTRALITY.items():
        if key in loc:
            return weight
    return 0.50


def score_to_category(score: float, fod_present: bool) -> str:
    """Map numeric score (0.0 - 10.0) to aviation risk category."""
    if not fod_present or score == 0.0:
        return "CLEAR"
    if score < 2.0:
        return "LOW"
    if score < 5.0:
        return "MEDIUM"
    if score < 7.5:
        return "HIGH"
    return "CRITICAL"


def compute_risk_score(detection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a deterministic, auditable risk score from a detection payload.

    Formula:
        numeric_score = raw_component * obj_weight * loc_weight * confidence * (10 / 3)
        maximum score = 3.0 * 1.0 * 1.0 * 1.0 * (10 / 3) = 10.0

    Args:
        detection: Validated output dictionary from detect_fod()

    Returns:
        Dictionary containing numeric_score, risk_category, and inspectable breakdown.
    """
    fod_present = detection.get("fod_present", False)

    if not fod_present:
        return {
            "numeric_score": 0.0,
            "risk_category": "CLEAR",
            "breakdown": {
                "fod_present": False,
                "note": "No FOD detected — score is 0.0 by definition.",
            },
        }

    raw_component = RAW_RISK_BASE.get(detection.get("risk_raw", "LOW"), 1.0)
    obj_weight = get_object_weight(detection.get("object_class", "unknown"))
    loc_weight = get_location_weight(detection.get("location_estimate", "unknown"))
    confidence = float(detection.get("confidence", 0.5))

    numeric_score = round(
        min(raw_component * obj_weight * loc_weight * confidence * (10.0 / 3.0), 10.0), 2
    )
    risk_category = score_to_category(numeric_score, fod_present)

    result = {
        "numeric_score": numeric_score,
        "risk_category": risk_category,
        "breakdown": {
            "fod_present": fod_present,
            "raw_component": raw_component,
            "object_weight": obj_weight,
            "location_weight": loc_weight,
            "confidence": confidence,
            "formula": "raw_component * obj_weight * loc_weight * confidence * (10/3)",
        },
    }
    logger.info("Risk calculation: Score=%.2f | Category=%s", numeric_score, risk_category)
    return result
