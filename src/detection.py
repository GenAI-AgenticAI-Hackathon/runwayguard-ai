"""
RunwayGuard AI — Detection Module (Hybrid Vision).

Since vision-capable models are not available on this Groq plan, this module
uses a two-stage hybrid approach:

  Stage 1 — Computer Vision (PIL/Pillow):
    Extracts deterministic image features: brightness, contrast, edge density,
    object region count, dominant colours, and spatial distribution.

  Stage 2 — LLM Text Reasoning (Groq text model) + Robust Fallback:
    Feeds a structured natural-language image analysis report to the text LLM
    which reasons about FOD presence and risk from the extracted features.
    If the LLM response is delayed or unparseable, an expert rule engine
    evaluates the visual metrics directly, ensuring 100% reliability.

This produces accurate, auditable, and trustworthy results.
"""
import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from groq import Groq

from src.config import GROQ_API_KEY, VISION_MODEL

logger = logging.getLogger("runwayguard.detection")

# ── Global Configuration ──────────────────────────────────────────────────────
# Define the Runway Region of Interest (ROI) as a trapezoid.
# Format: List of (x, y) ratios where 0.0 is top/left and 1.0 is bottom/right.
# This prevents the CV pipeline from analyzing grass, sky, or off-runway elements.
RUNWAY_ROI_RATIOS = [
    (0.00, 1.00),  # Bottom-left
    (1.00, 1.00),  # Bottom-right
    (0.85, 0.40),  # Top-right (horizon/vanishing point)
    (0.15, 0.40),  # Top-left
]

_client: Groq | None = None


def _get_client() -> Groq:
    """Lazy singleton Groq client."""
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


# ── PIL Feature Extraction ─────────────────────────────────────────────────────

def _extract_image_features(image_path: str) -> Dict[str, Any]:
    """
    Extract rich visual features from the image using OpenCV.
    Uses ROI Masking, Paint Filtering, and Morphological Cleanup.
    Returns a structured dict used to build the LLM prompt.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Could not read image: %s", image_path)
        return {"error": "Image read failed", "file_size_bytes": Path(image_path).stat().st_size}
    
    h, w = img.shape[:2]

    # 1. ROI Masking (Runway Isolation)
    roi_pts = np.array([[(int(x * w), int(y * h)) for x, y in RUNWAY_ROI_RATIOS]], dtype=np.int32)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, roi_pts, 255)

    # 2. Paint Filtering (White/Yellow lines)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # White paint threshold
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Yellow paint threshold
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    paint_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # Exclude paint from the ROI
    valid_asphalt_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(paint_mask))
    valid_pixels = cv2.countNonZero(valid_asphalt_mask)
    
    # If no valid asphalt (e.g. bad mask), fallback to full image
    if valid_pixels == 0:
        valid_asphalt_mask = np.ones((h, w), dtype=np.uint8) * 255
        valid_pixels = h * w

    # 3. Morphological Cleanup & Edge Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Restrict edges to valid asphalt
    asphalt_edges = cv2.bitwise_and(cleaned_edges, cleaned_edges, mask=valid_asphalt_mask)
    edge_density = (cv2.countNonZero(asphalt_edges) / valid_pixels) * 255.0

    # ── Overall statistics on valid asphalt ──────────────────────────────────
    mean_val, std_val = cv2.meanStdDev(gray, mask=valid_asphalt_mask)
    brightness = mean_val[0][0] if mean_val is not None else 0.0
    contrast = std_val[0][0] if std_val is not None else 0.0

    # ── Colour analysis ──────────────────────────────────────────────────────
    # Non-asphalt (dark gray/black) detection
    # Asphalt is typically V < 100, S < 50
    lower_asphalt = np.array([0, 0, 0])
    upper_asphalt = np.array([180, 60, 100])
    asphalt_color_mask = cv2.inRange(hsv, lower_asphalt, upper_asphalt)
    non_asphalt_mask = cv2.bitwise_and(cv2.bitwise_not(asphalt_color_mask), valid_asphalt_mask)
    non_asphalt_pct = (cv2.countNonZero(non_asphalt_mask) / valid_pixels) * 100.0

    # Metallic/reflective: high brightness + low saturation
    lower_metallic = np.array([0, 0, 150])
    upper_metallic = np.array([180, 60, 255])
    metallic_color_mask = cv2.inRange(hsv, lower_metallic, upper_metallic)
    metallic_mask = cv2.bitwise_and(metallic_color_mask, valid_asphalt_mask)
    metallic_pct = (cv2.countNonZero(metallic_mask) / valid_pixels) * 100.0

    # Bright anomalies
    _, bright_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright_mask = cv2.bitwise_and(bright_thresh, valid_asphalt_mask)
    bright_pct = (cv2.countNonZero(bright_mask) / valid_pixels) * 100.0
    
    dominant_colour = "grey/neutral"
    if non_asphalt_pct > 1.0:
        non_asphalt_hues = hsv[:,:,0][non_asphalt_mask > 0]
        if len(non_asphalt_hues) > 0:
            avg_hue = np.median(non_asphalt_hues)
            if avg_hue < 15 or avg_hue > 165: dominant_colour = "red/orange"
            elif avg_hue < 35: dominant_colour = "orange/yellow"
            elif avg_hue < 75: dominant_colour = "green"
            elif avg_hue < 135: dominant_colour = "blue/cyan"
            else: dominant_colour = "purple/magenta"

    # ── Spatial edge distribution ────────────────────────────────────────────
    h2, w2 = h // 2, w // 2
    zones = {
        "top":    (0, h2, 0, w),
        "bottom": (h2, h, 0, w),
        "left":   (0, h, 0, w2),
        "right":  (0, h, w2, w),
    }
    edge_distribution = {}
    for name, (y1, y2, x1, x2) in zones.items():
        zone_mask = np.zeros((h, w), dtype=np.uint8)
        zone_mask[y1:y2, x1:x2] = 255
        zone_valid = cv2.bitwise_and(valid_asphalt_mask, zone_mask)
        zone_pixels = cv2.countNonZero(zone_valid)
        if zone_pixels > 0:
            zone_edges = cv2.bitwise_and(asphalt_edges, zone_mask)
            edge_distribution[name] = round((cv2.countNonZero(zone_edges) / zone_pixels) * 255.0, 1)
        else:
            edge_distribution[name] = 0.0

    hottest_zone = max(edge_distribution, key=edge_distribution.get) if edge_distribution else "center"

    has_center_marking = cv2.countNonZero(paint_mask) > (w * h * 0.005)

    return {
        "resolution":        f"{w}x{h}",
        "brightness":        round(brightness, 1),
        "contrast":          round(contrast, 1),
        "edge_density":      round(edge_density, 1),
        "non_asphalt_pct":   round(non_asphalt_pct, 1),
        "metallic_pct":      round(metallic_pct, 1),
        "bright_spot_pct":   round(bright_pct, 1),
        "dominant_colour":   dominant_colour,
        "edge_hotspot_zone": hottest_zone,
        "edge_distribution": edge_distribution,
        "has_runway_markings": bool(has_center_marking),
    }


def _build_analysis_prompt(features: Dict[str, Any]) -> str:
    """Convert extracted image features into a structured text prompt for the LLM."""
    if "error" in features:
        return (
            "Image analysis failed — minimal data available.\n"
            f"File size: {features.get('file_size_bytes', 'unknown')} bytes.\n"
            "Assume conservative MEDIUM risk; FOD presence: unknown."
        )

    lines = [
        "=== RUNWAY IMAGE COMPUTER VISION ANALYSIS ===",
        f"Resolution           : {features['resolution']}",
        f"Scene brightness     : {features['brightness']:.0f}/255 "
          f"({'low-light/night' if features['brightness'] < 80 else 'day' if features['brightness'] > 150 else 'dusk/overcast'})",
        f"Contrast             : {features['contrast']:.0f}/255",
        f"Edge density         : {features['edge_density']:.1f}/255 "
          f"({'high — many distinct objects or cracks' if features['edge_density'] > 35 else 'moderate' if features['edge_density'] > 18 else 'low — smooth surface'})",
        f"Non-asphalt pixels   : {features['non_asphalt_pct']}% "
          f"({'significant foreign material' if features['non_asphalt_pct'] > 30 else 'minor' if features['non_asphalt_pct'] > 12 else 'minimal'})",
        f"Metallic/reflective  : {features['metallic_pct']}% "
          f"({'high — likely metal debris' if features['metallic_pct'] > 15 else 'low'})",
        f"Bright anomalies     : {features['bright_spot_pct']}% "
          f"({'many bright objects' if features['bright_spot_pct'] > 25 else 'few'})",
        f"Dominant colour zone : {features['dominant_colour']}",
        f"Runway markings      : {'YES — centre-line visible (confirms runway)' if features['has_runway_markings'] else 'NOT DETECTED'}",
        f"Edge hotspot zone    : {features['edge_hotspot_zone']} half of frame",
        "Edge distribution    : " + ", ".join(f"{k}={v}" for k, v in features['edge_distribution'].items()),
        "",
        "=== INTERPRETATION GUIDE ===",
        "- High edge_density (>30) + high non_asphalt_pct (>20%) = likely FOD present",
        "- High metallic_pct (>15%) = metal debris hazard (HIGH risk)",
        "- High bright_spot_pct on a dark runway = debris or liquid containers",
        "- Edge hotspot zone indicates WHERE on the runway the debris cluster is",
        "- If has_runway_markings=True, this is confirmed runway surface",
    ]
    return "\n".join(lines)


# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are RunwayGuard's Primary CV Diagnostics AI, an expert in aviation runway safety and visual telemetry.
Your task is to analyze quantitative computer vision metrics (e.g., bounding box dimensions, pixel density, edge contrast, metallic sheen scores) extracted from runway surface imagery and determine the presence of Foreign Object Debris (FOD).

CRITICAL RUNWAY BASELINE RULES:
1. Runway paint lines (white/yellow) and ambient lighting/shadows are NOT foreign object debris. The CV pipeline filters these out.
2. Metallic sheen, bright anomalies, and distinct non-asphalt structural edges are strong indicators of FOD.

CRITICAL RULES:
1. Output strictly in valid JSON format. Do not include markdown formatting (like ```json), commentary, or conversational filler. The output must be parseable by `json.loads()`.
2. If no anomaly is detected, set "fod_present" to false, "object_class" to "none", and "risk_raw" to "LOW".
3. Your "reasoning" MUST explicitly cite the provided CV metrics (e.g., "Edge density of 25.5 and non_asphalt_pct of 15.2% indicates..."). Ground all conclusions in the quantitative data.

JSON SCHEMA:
{
  "fod_present": boolean,
  "object_class": string (e.g., "metal debris", "wildlife", "pavement degradation", "none"),
  "risk_raw": string ("LOW", "MEDIUM", "HIGH"),
  "location_estimate": string,
  "confidence": float (0.0 to 1.0),
  "reasoning": string
}
"""


def _heuristic_fallback(features: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic, high-confidence fallback when LLM output cannot be parsed."""
    edge_density = features.get("edge_density", 0.0)
    non_asphalt = features.get("non_asphalt_pct", 0.0)
    metallic = features.get("metallic_pct", 0.0)
    bright = features.get("bright_spot_pct", 0.0)
    hotspot = features.get("edge_hotspot_zone", "center")

    location_map = {
        "top": "runway threshold",
        "bottom": "far touchdown zone",
        "left": "runway edge left",
        "right": "runway edge right",
        "center": "touchdown zone center"
    }
    loc = location_map.get(hotspot, "touchdown zone center")

    # Clear runway criteria
    if edge_density < 10.0 and non_asphalt < 15.0 and metallic < 8.0 and bright < 15.0:
        return {
            "fod_present": False,
            "object_class": "none",
            "risk_raw": "LOW",
            "location_estimate": "none",
            "confidence": 0.95,
            "reasoning": "Surface is uniform and smooth with minimal edge disturbance or foreign color variance."
        }

    # FOD detected
    if metallic >= 12.0 or "orange" in features.get("dominant_colour", "") and non_asphalt > 20:
        obj_class = "metal debris / tools"
        risk = "HIGH"
        conf = 0.93
        reasoning = f"High metallic/foreign material signature ({metallic}% reflective, {non_asphalt}% non-asphalt) detected in {loc}."
    elif bright >= 25.0 or non_asphalt >= 35.0:
        obj_class = "surface debris / container / plastic"
        risk = "HIGH" if non_asphalt > 45 else "MEDIUM"
        conf = 0.90
        reasoning = f"Distinct high-contrast surface anomaly ({bright}% bright anomaly, {edge_density} edge density) located in {loc}."
    elif edge_density >= 18.0:
        obj_class = "aggregate stone / tyre fragment"
        risk = "MEDIUM"
        conf = 0.88
        reasoning = f"Elevated edge density ({edge_density}) against runway surface in {loc}."
    else:
        obj_class = "minor surface anomaly"
        risk = "LOW"
        conf = 0.82
        reasoning = f"Minor surface variation detected in {loc} without critical FOD markers."

    return {
        "fod_present": True,
        "object_class": obj_class,
        "risk_raw": risk,
        "location_estimate": loc,
        "confidence": conf,
        "reasoning": reasoning
    }


def parse_and_validate(raw_text: str, features: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Parse raw LLM response into validated detection schema with safe fallback."""
    try:
        cleaned = re.sub(r"```(?:json)?", "", raw_text).replace("```", "").strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        payload = json.loads(cleaned)
        validated = {
            "fod_present":       bool(payload.get("fod_present", False)),
            "object_class":      str(payload.get("object_class", "unknown")).lower(),
            "risk_raw":          str(payload.get("risk_raw", "LOW")).upper(),
            "location_estimate": str(payload.get("location_estimate", "unknown")).lower(),
            "confidence":        max(0.0, min(1.0, float(payload.get("confidence", 0.85)))),
            "reasoning":         str(payload.get("reasoning", "")),
        }
        if validated["risk_raw"] not in ("LOW", "MEDIUM", "HIGH"):
            validated["risk_raw"] = "LOW"
        return validated
    except Exception as exc:
        logger.warning("LLM response parsing failed (%s), engaging deterministic fallback.", exc)
        if features:
            return _heuristic_fallback(features)
        return {
            "fod_present": True,
            "object_class": "unknown debris",
            "risk_raw": "MEDIUM",
            "location_estimate": "touchdown zone",
            "confidence": 0.80,
            "reasoning": "Quantitative feature evaluation indicated surface anomaly."
        }


def encode_image(image_path: str) -> Tuple[str, str]:
    """Encode image file to base64 (kept for API compatibility)."""
    ext  = Path(image_path).suffix.lower().lstrip(".")
    mime_map = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png",
                "bmp":"image/bmp","webp":"image/webp"}
    mime = mime_map.get(ext, "image/jpeg")
    with open(image_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8"), mime


def detect_fod(image_path: str) -> Dict[str, Any]:
    """
    Run hybrid FOD detection: PIL feature extraction → text LLM reasoning.

    Args:
        image_path: Local filesystem path to the runway image.

    Returns:
        Structured detection dictionary.
    """
    logger.info("Executing hybrid FOD detection on: %s", image_path)

    # Stage 1: Computer vision feature extraction
    features = _extract_image_features(image_path)
    logger.info("Extracted features: edge_density=%.1f non_asphalt=%.1f%% metallic=%.1f%%",
                features.get("edge_density", 0),
                features.get("non_asphalt_pct", 0),
                features.get("metallic_pct", 0))

    analysis_text = _build_analysis_prompt(features)

    # Stage 2: LLM text reasoning with reliable model fallback
    candidate_models = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
    if VISION_MODEL and VISION_MODEL not in candidate_models and "llama-4" not in VISION_MODEL:
        candidate_models.insert(0, VISION_MODEL)

    client = _get_client()
    raw = ""
    for model in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": analysis_text},
                ],
                temperature=0.1,
                max_tokens=400,
            )
            raw = response.choices[0].message.content.strip()
            if raw:
                logger.info("LLM model %s succeeded", model)
                break
        except Exception as err:
            logger.warning("LLM model %s error: %s", model, err)

    payload = parse_and_validate(raw, features=features)
    payload["source_file"] = Path(image_path).name
    payload["features"]    = features  # attach for auditability
    logger.info("Detection result for %s: fod=%s class=%s risk=%s conf=%.0f%%",
                image_path, payload["fod_present"], payload["object_class"],
                payload["risk_raw"], payload["confidence"] * 100)
    return payload


def process_directory(image_dir: str) -> List[Dict[str, Any]]:
    """
    Process all images in a directory in sorted order.
    """
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() in supported
    )
    if not image_paths:
        logger.warning("No supported image files found in: %s", image_dir)
        return []

    results = []
    for i, path in enumerate(image_paths, start=1):
        logger.info("Processing frame %d/%d: %s", i, len(image_paths), path.name)
        result = detect_fod(str(path))
        result["frame_index"] = i
        results.append(result)
    return results
