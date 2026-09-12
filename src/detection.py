"""
RunwayGuard AI — Detection Module.

Pure Computer Vision functions: base64 image encoding and Groq vision-model FOD detection.
No Gradio imports. No UI logic. Fully unit-testable in isolation.
"""
import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from groq import Groq

from src.config import GROQ_API_KEY, VISION_MODEL

logger = logging.getLogger("runwayguard.detection")

_client: Groq | None = None


def _get_client() -> Groq:
    """Lazy singleton so Groq client is initialized on demand."""
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


SYSTEM_PROMPT = """\
You are a runway safety computer vision system.
Analyze the runway image and return ONLY a valid JSON object with exactly these keys:

{
  "fod_present": <boolean>,
  "object_class": <string: e.g. "metal debris", "bird", "cone", "tool", "tire fragment", "none">,
  "risk_raw": <string: exactly "LOW", "MEDIUM", or "HIGH">,
  "location_estimate": <string: e.g. "touchdown zone center", "runway edge left", "threshold right", "taxiway", "none">,
  "confidence": <float: 0.0 to 1.0>
}

Rules:
- If no FOD is present: fod_present=false, object_class="none", risk_raw="LOW", location_estimate="none".
- Return NOTHING except the JSON object. No markdown fences, no explanatory text.
"""


def encode_image(image_path: str) -> Tuple[str, str]:
    """
    Encode an image file to base64 string and determine MIME type.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (base64_encoded_string, mime_type).
    """
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "bmp": "image/bmp",
        "webp": "image/webp",
    }
    mime = mime_map.get(ext, "image/jpeg")
    with open(image_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8"), mime


def parse_and_validate(raw_text: str) -> Dict[str, Any]:
    """
    Parse raw response string into validated JSON detection schema.
    Strips markdown code fences if present.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw_text).replace("```", "").strip()
    payload = json.loads(cleaned)

    validated = {
        "fod_present": bool(payload.get("fod_present", False)),
        "object_class": str(payload.get("object_class", "unknown")).lower(),
        "risk_raw": str(payload.get("risk_raw", "LOW")).upper(),
        "location_estimate": str(payload.get("location_estimate", "unknown")).lower(),
        "confidence": max(0.0, min(1.0, float(payload.get("confidence", 0.5)))),
    }
    if validated["risk_raw"] not in ("LOW", "MEDIUM", "HIGH"):
        validated["risk_raw"] = "LOW"
    return validated


def detect_fod(image_path: str) -> Dict[str, Any]:
    """
    Run Groq vision model on a single image and return validated detection payload.

    Args:
        image_path: Local filesystem path to the runway image.

    Returns:
        Structured detection dictionary.
    """
    logger.info("Executing FOD detection on: %s", image_path)
    b64, mime = encode_image(image_path)

    client = _get_client()
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": "Analyze this runway image for FOD. Return JSON only."},
                ],
            },
        ],
        temperature=0.1,
        max_tokens=256,
    )

    raw = response.choices[0].message.content.strip()
    payload = parse_and_validate(raw)
    payload["source_file"] = Path(image_path).name
    logger.info("Detection result for %s: %s", image_path, payload)
    return payload


def process_directory(image_dir: str) -> List[Dict[str, Any]]:
    """
    Process all images in a directory in sorted order (simulating sequential frame feed).
    Not true real-time video object tracking (see LIMITATIONS.md).
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
