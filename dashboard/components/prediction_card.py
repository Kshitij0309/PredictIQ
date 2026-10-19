"""Prediction result visualization component using native Streamlit widgets.

Replaces fragile raw HTML markup with native Streamlit components (metrics, columns,
progress bars, status banners, and captions) ensuring seamless light/dark theme support
and defensive handling of missing or malformed API fields.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from dashboard.utils.formatting import (
    calculate_classification_margin,
    format_margin_percentage_points,
    format_percentage,
    format_probability,
    get_risk_tier_style,
)


def extract_prediction_details(prediction: Dict[str, Any] | None) -> Dict[str, Any]:
    """Extract, validate, and compute prediction attributes with defensive fallbacks.

    Preserves backend values when provided; applies safe bounds and defaults
    when fields are missing or malformed.
    """
    if not isinstance(prediction, dict):
        prediction = {}

    # 1. Calibrated Churn Probability [0.0, 1.0]
    raw_prob = prediction.get("churn_probability")
    try:
        prob = float(raw_prob) if raw_prob is not None else 0.0
    except (ValueError, TypeError):
        prob = 0.0
    prob = max(0.0, min(1.0, prob))

    # 2. Decision Threshold [0.0, 1.0]
    raw_threshold = prediction.get("decision_threshold")
    try:
        threshold = float(raw_threshold) if raw_threshold is not None else 0.34
    except (ValueError, TypeError):
        threshold = 0.34
    threshold = max(0.0, min(1.0, threshold))

    # 3. Classification Margin = probability - threshold
    margin = calculate_classification_margin(prob, threshold)
    margin_pp = margin * 100.0

    # 4. Predicted Churn (preserve backend string if present)
    raw_churn = prediction.get("predicted_churn")
    if raw_churn is None:
        predicted_churn = "Yes" if prob >= threshold else "No"
    else:
        predicted_churn = str(raw_churn).strip()

    # 5. Risk Tier (preserve backend string if present)
    raw_tier = prediction.get("risk_tier")
    if raw_tier is None or not str(raw_tier).strip():
        if prob >= 0.60:
            risk_tier = "High Risk"
        elif prob >= threshold:
            risk_tier = "Medium Risk"
        else:
            risk_tier = "Low Risk"
    else:
        risk_tier = str(raw_tier).strip()

    is_churn = predicted_churn.lower() in ("yes", "true", "1") or (
        raw_churn is None and prob >= threshold
    )

    return {
        "predicted_churn": predicted_churn,
        "is_churn": is_churn,
        "probability": prob,
        "threshold": threshold,
        "margin": margin,
        "margin_pp": margin_pp,
        "risk_tier": risk_tier,
    }


def render_prediction_card(prediction: Dict[str, Any] | None) -> None:
    """Render comprehensive prediction result using native Streamlit components.

    Replaces fragile raw HTML markup with native Streamlit widgets:
    - st.error / st.warning / st.success for the verdict banner
    - st.columns and st.metric for probabilities, threshold, and margin
    - st.progress with threshold indicators for the visual probability scale
    - st.info / st.warning for retention recommendations
    - st.caption for methodological caveats and non-causal guidance
    """
    details = extract_prediction_details(prediction)
    prob = details["probability"]
    threshold = details["threshold"]
    margin = details["margin"]
    predicted_churn = details["predicted_churn"]
    risk_tier = details["risk_tier"]
    is_churn = details["is_churn"]

    tier_lower = risk_tier.lower()
    style = get_risk_tier_style(risk_tier)

    # 1. Prediction Verdict & Risk Tier Banner (Native Streamlit Status Alert)
    with st.container():
        if is_churn:
            if "high" in tier_lower:
                st.error(
                    f"### 🚨 Churn Predicted: **{predicted_churn} (At Risk)**\n\n"
                    f"**Risk Classification:** {style['icon']} **{risk_tier}** — "
                    "Customer exhibits high likelihood of cancellation under current service profile."
                )
            else:
                st.warning(
                    f"### ⚠️ Churn Predicted: **{predicted_churn} (Elevated Risk)**\n\n"
                    f"**Risk Classification:** {style['icon']} **{risk_tier}** — "
                    "Customer churn probability exceeds the operational decision cutoff."
                )
        else:
            st.success(
                f"### ✅ Churn Predicted: **{predicted_churn} (Stable Customer)**\n\n"
                f"**Risk Classification:** {style['icon']} **{risk_tier}** — "
                "Customer churn probability is below the operational decision cutoff."
            )

        st.markdown("")

        # 2. Key Metrics Row (Native st.metric widgets)
        m_col1, m_col2, m_col3 = st.columns(3)

        with m_col1:
            st.metric(
                label="Calibrated Probability",
                value=format_percentage(prob, decimals=2),
                delta=f"{format_probability(prob)} score",
                help="Sigmoid-calibrated probability of customer churning in the next billing cycle.",
            )

        with m_col2:
            st.metric(
                label="Decision Threshold (p*)",
                value=format_percentage(threshold, decimals=2),
                delta=f"{format_probability(threshold)} cutoff",
                delta_color="off",
                help="Optimal cost-sensitive threshold derived from 5-fold cross-validation.",
            )

        with m_col3:
            margin_pp_str = format_margin_percentage_points(margin, decimals=2)
            st.metric(
                label="Classification Margin",
                value=margin_pp_str,
                delta=f"{margin:+.4f} (probability − threshold)",
                delta_color="inverse" if margin >= 0 else "normal",
                help="Signed difference (probability − threshold) in percentage points (multiplied by 100) and raw decimal.",
            )

        st.markdown("")

        # 3. Probability Bar with Clearly Identified Threshold (Native st.progress)
        st.markdown("##### 📈 Churn Probability Scale")
        progress_text = (
            f"Calibrated Probability: {format_percentage(prob, decimals=2)} "
            f"| Cutoff Threshold: {format_percentage(threshold, decimals=2)}"
        )
        st.progress(prob, text=progress_text)

        # Scale markers directly beneath progress bar
        s_col1, s_col2, s_col3 = st.columns([1, 1, 1])
        with s_col1:
            st.caption("🟢 **0.0%** (Safe / Low Risk)")
        with s_col2:
            st.caption(
                f"▲ **Cutoff: {format_percentage(threshold, decimals=2)}** ({format_probability(threshold)})"
            )
        with s_col3:
            st.caption(
                "🔴 **100.0%** (High Likelihood)",
                help="Probabilistic model estimate, not deterministic certainty.",
            )

        st.markdown("")

        # 4. Short Retention Action Recommendation
        st.markdown("##### 🎯 Recommended Retention Strategy")
        if "high" in tier_lower:
            st.warning(
                "🚨 **Immediate Priority Intervention:** High churn risk detected. "
                "Recommend outreach with long-term contract discounts, bundled tech support review, "
                "or account billing review before the next cycle."
            )
        elif "medium" in tier_lower:
            st.info(
                "⚡ **Proactive Engagement:** Probability exceeds decision cutoff. "
                "Recommend sending satisfaction surveys, highlighting online security add-ons, "
                "or monitoring billing payment method changes."
            )
        else:
            st.info(
                "💎 **Standard Relationship Maintenance:** Low churn likelihood. "
                "Continue standard engagement, promote automated billing incentives, "
                "and maintain normal service communications."
            )

        # 5. Methodological & Non-Causal Guidance
        st.caption(
            "ℹ️ **Methodological Guidance:** Model estimates reflect observed historical statistical associations, "
            "not deterministic certainty or verified causal drivers. Retention actions should be validated through "
            "controlled experiments (A/B testing) rather than assumed causal levers."
        )
