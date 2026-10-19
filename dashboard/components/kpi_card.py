"""Reusable KPI metric card component for Streamlit."""

from __future__ import annotations

from typing import Optional

import streamlit as st


def render_kpi_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    delta: Optional[str] = None,
    color: Optional[str] = None,
    icon: Optional[str] = None,
) -> None:
    """Render a polished, responsive KPI card with modern typography."""
    val_color = color or "inherit"
    icon_html = f"<span style='font-size: 1.25rem; margin-right: 0.4rem;'>{icon}</span>" if icon else ""
    delta_html = (
        f"<span style='font-size: 0.8rem; font-weight: 600; color: #10b981; margin-left: 0.5rem;'>{delta}</span>"
        if delta
        else ""
    )
    subtitle_html = (
        f"<div style='font-size: 0.78rem; opacity: 0.75; margin-top: 0.25rem;'>{subtitle}</div>"
        if subtitle
        else ""
    )

    card_html = f"""
    <div style="
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        transition: transform 0.15s ease;
    ">
        <div style="font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; opacity: 0.8; margin-bottom: 0.35rem;">
            {icon_html}{title}
        </div>
        <div style="font-size: 1.85rem; font-weight: 700; color: {val_color}; line-height: 1.15;">
            {value}{delta_html}
        </div>
        {subtitle_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
