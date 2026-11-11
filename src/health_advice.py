"""Risk labels and concise activity guidance adapted from Hong Kong EPD."""
from __future__ import annotations


EPD_AQHI_URL = "https://www.aqhi.gov.hk/en/what-is-aqhi/about-aqhi.html"
EPD_ADVICE_URL = "https://www.aqhi.gov.hk/en/health-advice/sub-health-advice.html"


def risk_band(value: float) -> str:
    """Map a continuous model output to the official integer-band boundaries."""
    if value <= 3:
        return "Low"
    if value <= 6:
        return "Moderate"
    if value <= 7:
        return "High"
    return "Very High"


def risk_colour(label: str) -> str:
    return {
        "Low": "#2eb67d",
        "Moderate": "#f3b61f",
        "High": "#f27d3d",
        "Very High": "#df3f5b",
    }[label]


def activity_advice(label: str) -> dict[str, str]:
    advice = {
        "Low": {
            "general": "No special response is normally required.",
            "sensitive": "No special response is normally required.",
        },
        "Moderate": {
            "general": "Normal outdoor activity is generally appropriate.",
            "sensitive": "If symptoms occur, consider reducing outdoor physical exertion.",
        },
        "High": {
            "general": "No general response is normally required, but monitor changes.",
            "sensitive": "Reduce outdoor physical exertion and time outdoors, especially near heavy traffic.",
        },
        "Very High": {
            "general": "Reduce outdoor physical exertion and time outdoors, especially near heavy traffic.",
            "sensitive": "Reduce outdoor exertion and outdoor exposure to a minimum.",
        },
    }
    return advice[label]

