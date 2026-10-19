"""Service status indicator component."""

from __future__ import annotations

import streamlit as st


def render_service_status_badge(status_code: str, label: str, detail: str) -> None:
    """Render an unobtrusive status badge reflecting API health and readiness."""
    if status_code == "healthy":
        bg_color = "rgba(16, 185, 129, 0.12)"
        border_color = "rgba(16, 185, 129, 0.4)"
        text_color = "#10b981"
        icon = "●"
    elif status_code == "unready":
        bg_color = "rgba(245, 158, 11, 0.12)"
        border_color = "rgba(245, 158, 11, 0.4)"
        text_color = "#f59e0b"
        icon = "▲"
    else:
        bg_color = "rgba(239, 68, 68, 0.12)"
        border_color = "rgba(239, 68, 68, 0.4)"
        text_color = "#ef4444"
        icon = "✕"

    badge_html = f"""
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: {bg_color};
        border: 1px solid {border_color};
        color: {text_color};
        border-radius: 9999px;
        padding: 0.3rem 0.85rem;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 0.75rem;
    " title="{detail}">
        <span>{icon}</span>
        <span>{label}</span>
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)
