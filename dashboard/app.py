"""PredictIQ — Customer Churn Intelligence Dashboard (Streamlit).

Main entrypoint providing navigation between Customer Analytics,
Model Intelligence, and Real-Time Churn Prediction.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components.status_badge import render_service_status_badge
from dashboard.pages import churn_prediction, customer_analytics, model_intelligence
from dashboard.services.api_client import PredictIQClient

# Configure page layout and metadata
st.set_page_config(
    page_title="PredictIQ — Predictive Churn Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom responsive CSS
CUSTOM_CSS = """
<style>
    /* Metric and card typography polish */
    .stMetric label {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        opacity: 0.8 !important;
    }
    .stMetric div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
    }
    /* Clean sidebar container */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.15);
    }
    /* Tab formatting */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    /* Hide default footer decoration */
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main() -> None:
    """Application main controller."""
    api_client = PredictIQClient()

    # Sidebar Header & Branding
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.25rem;">
            <span style="font-size: 1.75rem;">⚡</span>
            <div>
                <span style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em;">PredictIQ</span>
            </div>
        </div>
        <div style="font-size: 0.8rem; opacity: 0.75; margin-bottom: 1rem;">
            Predictive Churn Intelligence & ML Platform
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API Health & Model Readiness Indicator
    status_code, badge_label, detail_msg = api_client.get_service_status()
    render_service_status_badge(status_code, badge_label, detail_msg)

    st.sidebar.markdown("---")

    # Main Page Navigation
    NAV_PAGES = {
        "📊 Customer Analytics": "analytics",
        "🧠 Model Intelligence": "intelligence",
        "🔮 Churn Prediction": "prediction",
    }

    selected_nav = st.sidebar.radio(
        "Navigation",
        options=list(NAV_PAGES.keys()),
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # Sidebar System Specs
    st.sidebar.markdown(
        """
        <div style="font-size: 0.75rem; opacity: 0.7; line-height: 1.6;">
            <div><strong>Champion Model:</strong> Random Forest</div>
            <div><strong>Calibration:</strong> Platt Sigmoid</div>
            <div><strong>Decision Threshold:</strong> p* = 0.3400</div>
            <div><strong>Platform Version:</strong> 0.5.0</div>
            <div><strong>Backend API:</strong> FastAPI + Uvicorn</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Page Routing
    page_key = NAV_PAGES[selected_nav]

    if page_key == "analytics":
        customer_analytics.render_page()
    elif page_key == "intelligence":
        model_intelligence.render_page()
    elif page_key == "prediction":
        churn_prediction.render_page(api_client)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; font-size: 0.75rem; opacity: 0.6; padding: 1rem 0;">
            PredictIQ Platform • Production ML Serving & Decision Intelligence • End-to-End Reference System
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
