from typing import Optional

from ..utils.language import translate_if_available


def answer_farmer_question(query: str, language: Optional[str] = "en") -> str:
    # Placeholder rules for common intents; replace with real NLP later
    normalized = query.strip().lower()
    if any(k in normalized for k in ["hello", "hi", "namaste", "vanakkam", "namaskara", "namaskar"]):
        base = "Hello! How can I help with your crops today?"
    elif "fertilizer" in normalized or "manure" in normalized:
        base = "Use balanced fertilizer based on soil test; avoid overuse to prevent burn."
    elif "water" in normalized or "irrigation" in normalized:
        base = "Irrigate in early morning or evening to reduce evaporation."
    elif "pest" in normalized or "insect" in normalized:
        base = "Identify pest species; use targeted IPM methods and avoid broad-spectrum sprays."
    else:
        base = (
            "I can help with disease identification from images and basic crop guidance. "
            "Please upload a clear leaf photo or ask a specific question."
        )
    return translate_if_available(base, target_language=language or "en")


