from io import BytesIO
from PIL import Image
import numpy as np
from typing import Dict

from ..utils.language import translate_if_available


def _simple_color_heuristic(image: Image.Image) -> str:
    # Very naive placeholder: classify based on dominant green ratio
    image_rgb = image.convert("RGB")
    arr = np.array(image_rgb)
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)
    green_ratio = float(np.mean(g / (r + b + 1e-6)))
    if green_ratio < 0.9:
        return "possible_leaf_blight"
    return "healthy"


def _advice_for_label(label: str) -> Dict[str, str]:
    # Placeholder rules. Later wire this to a CSV knowledge base.
    if label == "healthy":
        return {
            "title": "Healthy Leaf",
            "advice": "No disease detected. Continue regular irrigation and nutrient schedule.",
            "prevention": "Monitor weekly. Remove weeds. Use balanced NPK as per soil test.",
        }
    if label == "possible_leaf_blight":
        return {
            "title": "Possible Leaf Blight",
            "advice": "Early signs suggest blight. Isolate affected plants and improve airflow.",
            "prevention": "Avoid overhead watering. Consider copper-based fungicide as per label.",
        }
    return {
        "title": "Unknown",
        "advice": "Could not determine disease. Try clearer photo and ensure good lighting.",
        "prevention": "Consult local extension officer for confirmation.",
    }


def detect_disease_from_image(image_bytes: bytes, language: str) -> Dict[str, str]:
    image = Image.open(BytesIO(image_bytes))
    label = _simple_color_heuristic(image)
    info = _advice_for_label(label)
    info_translated = {k: translate_if_available(v, target_language=language) for k, v in info.items()}
    return {
        "label": label,
        "info": info_translated,
    }


