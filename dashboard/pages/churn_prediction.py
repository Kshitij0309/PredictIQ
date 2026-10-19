"""Real-Time Churn Scoring and Batch Prediction page for PredictIQ."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard.components.prediction_card import render_prediction_card
from dashboard.services.api_client import PredictIQAPIError, PredictIQClient
from dashboard.utils.csv_validator import generate_sample_batch_csv, validate_batch_dataframe
from dashboard.utils.formatting import format_number


def render_page(api_client: PredictIQClient) -> None:
    """Render Churn Prediction page."""
    st.markdown("## 🔮 Real-Time Churn Scoring & Batch Evaluation")
    st.markdown(
        "Generate calibrated churn probability estimates, optimal risk tier assignments, and "
        "retention recommendations using the live PredictIQ FastAPI service."
    )

    tab_single, tab_batch = st.tabs([
        "👤 Single Customer Prediction",
        "📁 CSV Batch Evaluation (Up to 500 Records)",
    ])

    # =========================================================================
    # TAB 1: Single Customer Prediction
    # =========================================================================
    with tab_single:
        st.markdown("### Single Customer Feature Inputs")
        st.caption(
            "Enter customer demographic, subscription, and account details. All inference is processed "
            "remotely by the FastAPI service."
        )

        with st.form("single_prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("##### 👤 Demographics & Identity")
                gender = st.selectbox("Gender", options=["Female", "Male"], index=0)
                senior_citizen = st.selectbox("Senior Citizen", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
                partner = st.selectbox("Partner", options=["Yes", "No"], index=1)
                dependents = st.selectbox("Dependents", options=["No", "Yes"], index=0)
                tenure = st.number_input(
                    "Tenure (Months with Company)",
                    min_value=0,
                    max_value=120,
                    value=1,
                    step=1,
                    help="Number of months customer has subscribed",
                )

            with col2:
                st.markdown("##### 📦 Services & Add-ons")
                phone_service = st.selectbox("Phone Service", options=["No", "Yes"], index=0)
                multiple_lines = st.selectbox(
                    "Multiple Lines",
                    options=["No phone service", "No", "Yes"],
                    index=0,
                )
                internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"], index=0)
                online_security = st.selectbox("Online Security", options=["No", "Yes", "No internet service"], index=0)
                online_backup = st.selectbox("Online Backup", options=["Yes", "No", "No internet service"], index=0)
                device_protection = st.selectbox("Device Protection", options=["No", "Yes", "No internet service"], index=0)
                tech_support = st.selectbox("Tech Support", options=["No", "Yes", "No internet service"], index=0)
                streaming_tv = st.selectbox("Streaming TV", options=["No", "Yes", "No internet service"], index=0)
                streaming_movies = st.selectbox("Streaming Movies", options=["No", "Yes", "No internet service"], index=0)

            with col3:
                st.markdown("##### 💳 Contract & Billing")
                contract = st.selectbox(
                    "Contract Duration",
                    options=["Month-to-month", "One year", "Two year"],
                    index=0,
                )
                paperless_billing = st.selectbox("Paperless Billing", options=["Yes", "No"], index=0)
                payment_method = st.selectbox(
                    "Payment Method",
                    options=[
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    index=0,
                )
                monthly_charges = st.number_input(
                    "Monthly Charges ($ USD)",
                    min_value=0.0,
                    max_value=500.0,
                    value=29.85,
                    step=1.0,
                    format="%.2f",
                )
                total_charges = st.number_input(
                    "Total Charges to Date ($ USD)",
                    min_value=0.0,
                    max_value=20000.0,
                    value=29.85,
                    step=5.0,
                    format="%.2f",
                    help="Historical cumulative billing amount",
                )

            submitted = st.form_submit_button("⚡ Run Churn Prediction", use_container_width=True)

        if submitted:
            payload: Dict[str, Any] = {
                "gender": gender,
                "SeniorCitizen": int(senior_citizen),
                "Partner": partner,
                "Dependents": dependents,
                "tenure": int(tenure),
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": float(monthly_charges),
                "TotalCharges": float(total_charges),
            }

            with st.spinner("Connecting to PredictIQ API for calibrated inference..."):
                try:
                    result = api_client.predict(payload)
                    st.success("✅ Prediction completed successfully.")
                    render_prediction_card(result)
                except PredictIQAPIError as err:
                    if err.status_code == 503:
                        st.error(
                            "🛑 **Model Pipeline Not Ready (HTTP 503):** The PredictIQ backend service is running, "
                            "but the machine learning model artifact is not loaded in memory. "
                            f"Detail: {err.detail}"
                        )
                    elif err.status_code == 422:
                        st.error(
                            "⚠️ **Input Validation Failed (HTTP 422):** The request payload failed schema validation. "
                            f"Detail: {err.detail}"
                        )
                    else:
                        st.error(f"❌ **API Request Failed:** {err.message}")
                except Exception as err:
                    st.error(f"❌ **Unexpected Error:** {err}")

    # =========================================================================
    # TAB 2: CSV Batch Evaluation
    # =========================================================================
    with tab_batch:
        st.markdown("### Batch Evaluation via CSV Upload")
        st.caption(
            "Upload a CSV file containing customer records (up to 500 rows). "
            "The file will be validated against the API schema before batch scoring."
        )

        b_col1, b_col2 = st.columns([3, 1])
        with b_col1:
            uploaded_file = st.file_uploader(
                "Choose customer CSV file",
                type=["csv"],
                help="Requires all 19 customer feature columns. Optional 'customerID' column will be preserved.",
            )
        with b_col2:
            st.markdown("##### Need a Template?")
            sample_csv = generate_sample_batch_csv()
            st.download_button(
                label="📥 Download Sample CSV",
                data=sample_csv,
                file_name="predictiq_batch_sample.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
            except Exception as read_err:
                st.error(f"❌ Could not parse uploaded CSV file: {read_err}")
                return

            st.markdown(f"**Loaded Preview:** `{uploaded_file.name}` ({len(raw_df):,} rows)")
            st.dataframe(raw_df.head(5), use_container_width=True)

            # Validate DataFrame
            is_valid, records, customer_ids, error_msg = validate_batch_dataframe(raw_df)

            if not is_valid:
                st.error(f"🛑 **Validation Error:** {error_msg}")
            else:
                st.success(f"✅ Schema validation passed for **{len(records):,} records**.")

                if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                    with st.spinner(f"Evaluating {len(records)} customer records via PredictIQ API..."):
                        try:
                            batch_response = api_client.predict_batch(records)
                            predictions = batch_response.get("predictions", [])
                            batch_threshold = batch_response.get("batch_threshold", 0.34)

                            st.success(f"🎉 Evaluated {len(predictions):,} records in batch!")

                            # Reassemble DataFrame with row association preserved
                            results_list = []
                            for idx, pred in enumerate(predictions):
                                row_data = raw_df.iloc[idx].to_dict()
                                row_data["Predicted_Churn"] = pred["predicted_churn"]
                                row_data["Churn_Probability"] = pred["churn_probability"]
                                row_data["Decision_Threshold"] = pred["decision_threshold"]
                                row_data["Risk_Tier"] = pred["risk_tier"]
                                results_list.append(row_data)

                            results_df = pd.DataFrame(results_list)

                            # Summary KPI Row
                            total_scored = len(results_df)
                            churn_flags = results_df["Predicted_Churn"].str.lower().isin(["yes", "1", "true"]).sum()
                            ret_flags = total_scored - churn_flags
                            churn_rate = (churn_flags / total_scored * 100) if total_scored > 0 else 0

                            k1, k2, k3, k4 = st.columns(4)
                            with k1:
                                st.metric("Total Scored", format_number(total_scored))
                            with k2:
                                st.metric("Predicted Churn", format_number(churn_flags), f"{churn_rate:.1f}% rate")
                            with k3:
                                st.metric("Predicted Retained", format_number(ret_flags))
                            with k4:
                                st.metric("Threshold Applied", f"{batch_threshold:.2f}")

                            # Breakdown by Risk Tier
                            st.markdown("#### Risk Tier Distribution")
                            tier_counts = results_df["Risk_Tier"].value_counts().reset_index()
                            tier_counts.columns = ["Risk_Tier", "Count"]
                            st.dataframe(tier_counts, use_container_width=True)

                            # Results Table
                            st.markdown("#### Detailed Prediction Results")
                            # Highlight columns of interest
                            display_cols = (
                                (["customerID"] if "customerID" in results_df.columns else [])
                                + ["Predicted_Churn", "Churn_Probability", "Risk_Tier", "Contract", "tenure", "MonthlyCharges"]
                            )
                            st.dataframe(results_df[display_cols].head(50), use_container_width=True)

                            # Downloadable CSV
                            csv_buffer = io.StringIO()
                            results_df.to_csv(csv_buffer, index=False)
                            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                            st.download_button(
                                label="💾 Download Complete Predictions CSV",
                                data=csv_buffer.getvalue(),
                                file_name=f"predictiq_predictions_{timestamp_str}.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

                        except PredictIQAPIError as err:
                            st.error(f"🛑 **Batch Prediction Failed:** {err.message}")
                        except Exception as err:
                            st.error(f"❌ **Unexpected Error during batch prediction:** {err}")


if __name__ == "__main__":
    render_page(PredictIQClient())

