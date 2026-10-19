"""Formatting and styling helpers for PredictIQ dashboard."""

from __future__ import annotations

from typing import Any, Dict


def format_currency(val: Any) -> str:
    """Format numeric value as USD currency."""
    if val is None:
        return "$0.00"
    try:
        num = float(val)
        return f"${num:,.2f}"
    except (ValueError, TypeError):
        return str(val)


def format_percentage(val: Any, decimals: int = 1) -> str:
    """Format numeric float (e.g. 0.265 or 26.5) as percentage."""
    if val is None:
        return "0.0%"
    try:
        num = float(val)
        # If value is in 0..1 range, scale to 0..100
        if 0.0 <= num <= 1.0 and num > 0:
            num = num * 100.0
        return f"{num:.{decimals}f}%"
    except (ValueError, TypeError):
        return str(val)


def format_probability(val: Any) -> str:
    """Format probability estimate as 4-decimal float or clean percentage."""
    if val is None:
        return "0.0000"
    try:
        num = float(val)
        return f"{num:.4f}"
    except (ValueError, TypeError):
        return str(val)


def format_number(val: Any) -> str:
    """Format integer with thousands separator."""
    if val is None:
        return "0"
    try:
        num = int(val)
        return f"{num:,}"
    except (ValueError, TypeError):
        return str(val)


def get_risk_tier_style(tier: str) -> Dict[str, str]:
    """Return color, background, and icon badge styling for risk tiers."""
    tier_lower = str(tier).lower().strip()
    if "high" in tier_lower:
        return {
            "label": "High Risk",
            "color": "#dc2626",
            "bg": "rgba(220, 38, 38, 0.12)",
            "border": "#f87171",
            "icon": "🔴",
            "badge_class": "badge-high-risk",
        }
    if "medium" in tier_lower:
        return {
            "label": "Medium Risk",
            "color": "#d97706",
            "bg": "rgba(217, 119, 6, 0.12)",
            "border": "#fbbf24",
            "icon": "🟡",
            "badge_class": "badge-medium-risk",
        }
    return {
        "label": "Low Risk",
        "color": "#16a34a",
        "bg": "rgba(22, 163, 74, 0.12)",
        "border": "#4ade80",
        "icon": "🟢",
        "badge_class": "badge-low-risk",
    }


def calculate_classification_margin(prob: Any, threshold: Any) -> float:
    """Calculate raw classification margin as (probability - threshold).

    Defaults safely if either input is invalid or missing.
    """
    try:
        p = float(prob) if prob is not None else 0.0
    except (ValueError, TypeError):
        p = 0.0
    try:
        t = float(threshold) if threshold is not None else 0.34
    except (ValueError, TypeError):
        t = 0.34
    return p - t


def format_margin_percentage_points(margin: Any, decimals: int = 2) -> str:
    """Format classification margin in percentage points (multiplied by 100).

    Example:
        0.3625  -> "+36.25 pp"
        -0.1250 -> "-12.50 pp"
        0.0     -> "+0.00 pp"
    """
    if margin is None:
        return f"+{0.0:.{decimals}f} pp"
    try:
        val = float(margin) * 100.0
        sign = "+" if val >= 0 else ""
        return f"{sign}{val:.{decimals}f} pp"
    except (ValueError, TypeError):
        return str(margin)

