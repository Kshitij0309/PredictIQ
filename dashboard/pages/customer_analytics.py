"""Customer Analytics & Churn Demographics page for PredictIQ."""

from __future__ import annotations

import altair as alt
import streamlit as st

from dashboard.components.kpi_card import render_kpi_card
from dashboard.services.data_service import (
    compute_breakdown,
    compute_overview_metrics,
    filter_dataset,
    load_customer_data,
)
from dashboard.utils.formatting import format_currency, format_number, format_percentage


def render_page() -> None:
    """Render Customer Analytics page."""
    st.markdown("## 📊 Customer Analytics & Churn Demographics")
    st.markdown(
        "Explore real customer population dynamics, contract distributions, and churn behavior "
        "across the IBM Telco Customer Churn dataset."
    )

    # 1. Load real dataset
    df, meta = load_customer_data()

    if not meta.get("available") or df.empty:
        st.error(
            "⚠️ **Dataset Unavailable**: Could not locate the customer dataset.\n\n"
            f"**Error Details:** {meta.get('error', 'File not found on disk')}\n\n"
            "**Resolution Instructions:**\n"
            "- In Docker: Verify that `./data:/app/data:ro` volume is mounted in `docker-compose.yml` and `PREDICTIQ_DATA_DIR=/app/data` is set.\n"
            "- On Host: Ensure `data/interim/Telco-Customer-Churn-cleaned.csv` or `data/raw/Telco-Customer-Churn.csv` exists.\n"
            "- Alternatively, configure the `PREDICTIQ_DATA_DIR` environment variable to point to the directory containing the dataset."
        )
        return

    # Dataset provenance badge
    ds_type = meta.get("dataset_type", "canonical_cleaned")
    src_path = meta.get("source_path", "")
    if ds_type == "raw_fallback":
        st.warning(
            f"⚠️ **Using Raw Dataset Fallback:** `{src_path}`. "
            "Canonical cleaned dataset (`data/interim/Telco-Customer-Churn-cleaned.csv`) is recommended for full analytics."
        )
    else:
        st.caption(f"📁 **Dataset Source:** `{src_path}` ({meta.get('total_records', 0):,} records loaded)")

    # Total dataset stats
    total_metrics = compute_overview_metrics(df)

    # 2. Sidebar Filters
    st.sidebar.markdown("### 🔍 Customer Filters")

    contract_opts = sorted(list(df["Contract"].dropna().unique())) if "Contract" in df.columns else []
    selected_contracts = st.sidebar.multiselect(
        "Contract Term",
        options=contract_opts,
        default=contract_opts,
        help="Filter customers by contract duration",
    )

    internet_opts = sorted(list(df["InternetService"].dropna().unique())) if "InternetService" in df.columns else []
    selected_internet = st.sidebar.multiselect(
        "Internet Service",
        options=internet_opts,
        default=internet_opts,
        help="Filter customers by internet connection type",
    )

    payment_opts = sorted(list(df["PaymentMethod"].dropna().unique())) if "PaymentMethod" in df.columns else []
    selected_payments = st.sidebar.multiselect(
        "Payment Method",
        options=payment_opts,
        default=payment_opts,
        help="Filter customers by billing payment method",
    )

    senior_choice = st.sidebar.selectbox(
        "Senior Citizen",
        options=["All", "Yes", "No"],
        index=0,
    )

    partner_choice = st.sidebar.selectbox(
        "Partner",
        options=["All", "Yes", "No"],
        index=0,
    )

    dependents_choice = st.sidebar.selectbox(
        "Dependents",
        options=["All", "Yes", "No"],
        index=0,
    )

    min_tenure = int(df["tenure"].min()) if "tenure" in df.columns else 0
    max_tenure = int(df["tenure"].max()) if "tenure" in df.columns else 72
    selected_tenure = st.sidebar.slider(
        "Tenure Range (Months)",
        min_value=min_tenure,
        max_value=max_tenure,
        value=(min_tenure, max_tenure),
    )

    min_charges = float(df["MonthlyCharges"].min()) if "MonthlyCharges" in df.columns else 18.0
    max_charges = float(df["MonthlyCharges"].max()) if "MonthlyCharges" in df.columns else 120.0
    selected_charges = st.sidebar.slider(
        "Monthly Charges ($ USD)",
        min_value=round(min_charges, 1),
        max_value=round(max_charges, 1),
        value=(round(min_charges, 1), round(max_charges, 1)),
    )

    if st.sidebar.button("🔄 Clear All Filters", use_container_width=True):
        st.rerun()

    # Apply filters
    filtered_df = filter_dataset(
        df,
        contracts=selected_contracts,
        internet_services=selected_internet,
        payment_methods=selected_payments,
        senior_citizen=senior_choice,
        partner=partner_choice,
        dependents=dependents_choice,
        tenure_range=selected_tenure,
        monthly_charges_range=selected_charges,
    )

    filtered_metrics = compute_overview_metrics(filtered_df)
    is_filtered = len(filtered_df) < len(df)

    # 3. Filter Banner Callout
    if is_filtered:
        pct_of_total = len(filtered_df) / len(df) * 100
        st.info(
            f"🎯 **Filtered Subset:** Showing **{format_number(len(filtered_df))}** of "
            f"**{format_number(len(df))}** customers (**{pct_of_total:.1f}%** of total cohort)."
        )

    # 4. KPI Cards Row
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        render_kpi_card(
            title="Total Customers",
            value=format_number(filtered_metrics["total_customers"]),
            subtitle=f"Full cohort: {format_number(total_metrics['total_customers'])}",
            icon="👥",
        )

    with kpi_col2:
        render_kpi_card(
            title="Churned Customers",
            value=format_number(filtered_metrics["churn_count"]),
            subtitle=f"Churn rate: {format_percentage(filtered_metrics['churn_pct'])}",
            color="#dc2626",
            icon="📉",
        )

    with kpi_col3:
        render_kpi_card(
            title="Retained Customers",
            value=format_number(filtered_metrics["retained_count"]),
            subtitle=f"Retention rate: {format_percentage(filtered_metrics['retained_pct'])}",
            color="#16a34a",
            icon="🛡️",
        )

    with kpi_col4:
        render_kpi_card(
            title="Monthly Revenue",
            value=format_currency(filtered_metrics["total_monthly_revenue"]),
            subtitle=f"Avg: {format_currency(filtered_metrics['avg_monthly_charges'])} / mo",
            color="#2563eb",
            icon="💵",
        )

    st.markdown("---")

    # 5. Core Analytical Breakdowns
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Churn Rate by Contract Type")
        contract_data = compute_breakdown(filtered_df, "Contract")
        if not contract_data.empty:
            chart = (
                alt.Chart(contract_data)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Contract:N", title="Contract Type", sort="-y"),
                    y=alt.Y("churn_rate_pct:Q", title="Churn Rate (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("Contract:N", legend=None, scale=alt.Scale(scheme="tableau10")),
                    tooltip=[
                        alt.Tooltip("Contract:N", title="Contract"),
                        alt.Tooltip("churn_rate_pct:Q", title="Churn Rate (%)"),
                        alt.Tooltip("churn_count:Q", title="Churned Count"),
                        alt.Tooltip("total_customers:Q", title="Total Customers"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "💡 **Insight:** Month-to-month contract customers exhibit substantially higher churn rates "
                "compared to 1-year and 2-year contracted accounts."
            )
        else:
            st.warning("No contract data available for the current filter selection.")

    with chart_col2:
        st.markdown("#### Churn Patterns across Tenure Cohorts")
        tenure_data = compute_breakdown(filtered_df, "tenure_cohort")
        if not tenure_data.empty:
            chart = (
                alt.Chart(tenure_data)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("tenure_cohort:N", title="Tenure Group", sort=["0-12 mos", "13-24 mos", "25-48 mos", "49-72 mos"]),
                    y=alt.Y("churn_rate_pct:Q", title="Churn Rate (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("tenure_cohort:N", legend=None, scale=alt.Scale(scheme="reds")),
                    tooltip=[
                        alt.Tooltip("tenure_cohort:N", title="Cohort"),
                        alt.Tooltip("churn_rate_pct:Q", title="Churn Rate (%)"),
                        alt.Tooltip("churn_count:Q", title="Churned Count"),
                        alt.Tooltip("total_customers:Q", title="Total"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "💡 **Insight:** New customers (0–12 months tenure) experience the steepest attrition risk; "
                "churn likelihood drops significantly as customer relationship matures."
            )
        else:
            st.warning("No tenure cohort data available for the current selection.")

    # 6. Service & Payment Method Breakdowns
    sub_col1, sub_col2 = st.columns(2)

    with sub_col1:
        st.markdown("#### Churn by Internet Service Provider")
        internet_data = compute_breakdown(filtered_df, "InternetService")
        if not internet_data.empty:
            chart = (
                alt.Chart(internet_data)
                .mark_bar()
                .encode(
                    x=alt.X("churn_rate_pct:Q", title="Churn Rate (%)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("InternetService:N", title="Internet Service", sort="-x"),
                    color=alt.Color("InternetService:N", legend=None),
                    tooltip=[
                        alt.Tooltip("InternetService:N", title="Service"),
                        alt.Tooltip("churn_rate_pct:Q", title="Churn Rate (%)"),
                        alt.Tooltip("total_customers:Q", title="Customers"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "💡 **Insight:** Fiber optic subscribers churn at twice the rate of DSL subscribers, "
                "often driven by price sensitivity and high monthly charges."
            )

    with sub_col2:
        st.markdown("#### Churn by Billing Payment Method")
        payment_data = compute_breakdown(filtered_df, "PaymentMethod")
        if not payment_data.empty:
            chart = (
                alt.Chart(payment_data)
                .mark_bar()
                .encode(
                    x=alt.X("churn_rate_pct:Q", title="Churn Rate (%)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("PaymentMethod:N", title="Payment Method", sort="-x"),
                    color=alt.Color("PaymentMethod:N", legend=None),
                    tooltip=[
                        alt.Tooltip("PaymentMethod:N", title="Method"),
                        alt.Tooltip("churn_rate_pct:Q", title="Churn Rate (%)"),
                        alt.Tooltip("total_customers:Q", title="Customers"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "💡 **Insight:** Electronic check users exhibit the highest churn frequency; "
                "automated bank transfer and credit card payments demonstrate higher loyalty."
            )

    st.markdown("---")

    # 7. Customer Data Explorer (Privacy-Safe)
    with st.expander("🔎 View Customer Records (Privacy-Safe Sample)", expanded=False):
        st.caption("Displays feature records without exposing sensitive internal identifiers.")
        display_cols = [
            c for c in filtered_df.columns
            if c not in ("customerID", "is_churn")
        ]
        st.dataframe(
            filtered_df[display_cols].head(100),
            use_container_width=True,
            height=300,
        )

    # 8. Methodological Disclaimer
    st.info(
        "ℹ️ **Methodological Note:** Observed statistical associations in customer demographic and "
        "billing attributes reflect historical predictive correlations. They do not constitute verified "
        "causal mechanisms. Retention strategies should test interventions through randomized experiments."
    )


if __name__ == "__main__":
    render_page()

