"""Unit tests for prediction card rendering and formatting utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dashboard.components.prediction_card import (
    extract_prediction_details,
    render_prediction_card,
)
from dashboard.utils.formatting import (
    calculate_classification_margin,
    format_margin_percentage_points,
    format_percentage,
    format_probability,
)


class TestFormattingUtils:
    """Test suite for probability and margin formatting functions."""

    def test_format_percentage_standard_decimal(self) -> None:
        assert format_percentage(0.7025, decimals=2) == "70.25%"
        assert format_percentage(0.34, decimals=2) == "34.00%"
        assert format_percentage(0.0, decimals=2) == "0.00%"
        assert format_percentage(1.0, decimals=2) == "100.00%"

    def test_format_percentage_scaled_input(self) -> None:
        assert format_percentage(70.25, decimals=2) == "70.25%"
        assert format_percentage(34.0, decimals=1) == "34.0%"

    def test_format_percentage_edge_cases(self) -> None:
        assert format_percentage(None) == "0.0%"
        assert format_percentage("invalid") == "invalid"

    def test_format_probability(self) -> None:
        assert format_probability(0.7025) == "0.7025"
        assert format_probability(0.34) == "0.3400"
        assert format_probability(None) == "0.0000"
        assert format_probability("bad") == "bad"

    def test_calculate_classification_margin(self) -> None:
        margin = calculate_classification_margin(0.7025, 0.34)
        assert pytest.approx(margin, rel=1e-5) == 0.3625

        neg_margin = calculate_classification_margin(0.15, 0.34)
        assert pytest.approx(neg_margin, rel=1e-5) == -0.19

        zero_margin = calculate_classification_margin(0.34, 0.34)
        assert pytest.approx(zero_margin, rel=1e-5) == 0.0

    def test_calculate_classification_margin_edge_cases(self) -> None:
        # Fallbacks when missing/malformed
        assert pytest.approx(calculate_classification_margin(None, 0.34), rel=1e-5) == -0.34
        assert pytest.approx(calculate_classification_margin("bad", "invalid"), rel=1e-5) == -0.34

    def test_format_margin_percentage_points(self) -> None:
        assert format_margin_percentage_points(0.3625, decimals=2) == "+36.25 pp"
        assert format_margin_percentage_points(-0.19, decimals=2) == "-19.00 pp"
        assert format_margin_percentage_points(0.0, decimals=2) == "+0.00 pp"
        assert format_margin_percentage_points(None) == "+0.00 pp"
        assert format_margin_percentage_points("bad") == "bad"


class TestExtractPredictionDetails:
    """Test suite for extract_prediction_details defensive parsing."""

    def test_extract_standard_high_risk_response(self) -> None:
        payload = {
            "predicted_churn": "Yes",
            "churn_probability": 0.7025,
            "decision_threshold": 0.34,
            "risk_tier": "High Risk",
        }
        details = extract_prediction_details(payload)
        assert details["predicted_churn"] == "Yes"
        assert details["is_churn"] is True
        assert pytest.approx(details["probability"], rel=1e-5) == 0.7025
        assert pytest.approx(details["threshold"], rel=1e-5) == 0.34
        assert pytest.approx(details["margin"], rel=1e-5) == 0.3625
        assert pytest.approx(details["margin_pp"], rel=1e-5) == 36.25
        assert details["risk_tier"] == "High Risk"

    def test_extract_standard_low_risk_response(self) -> None:
        payload = {
            "predicted_churn": "No",
            "churn_probability": 0.1250,
            "decision_threshold": 0.34,
            "risk_tier": "Low Risk",
        }
        details = extract_prediction_details(payload)
        assert details["predicted_churn"] == "No"
        assert details["is_churn"] is False
        assert pytest.approx(details["probability"], rel=1e-5) == 0.1250
        assert pytest.approx(details["margin"], rel=1e-5) == -0.2150
        assert details["risk_tier"] == "Low Risk"

    def test_extract_empty_dictionary(self) -> None:
        details = extract_prediction_details({})
        assert details["predicted_churn"] == "No"
        assert details["is_churn"] is False
        assert details["probability"] == 0.0
        assert details["threshold"] == 0.34
        assert pytest.approx(details["margin"], rel=1e-5) == -0.34
        assert details["risk_tier"] == "Low Risk"

    def test_extract_none_input(self) -> None:
        details = extract_prediction_details(None)
        assert details["predicted_churn"] == "No"
        assert details["probability"] == 0.0
        assert details["threshold"] == 0.34

    def test_extract_clamped_probability_bounds(self) -> None:
        # Out-of-bounds probability clamped to [0.0, 1.0]
        details_high = extract_prediction_details({"churn_probability": 1.5})
        assert details_high["probability"] == 1.0

        details_low = extract_prediction_details({"churn_probability": -0.5})
        assert details_low["probability"] == 0.0

    def test_extract_malformed_types(self) -> None:
        details = extract_prediction_details({
            "churn_probability": "not_a_number",
            "decision_threshold": None,
            "predicted_churn": 1,
            "risk_tier": None,
        })
        assert details["probability"] == 0.0
        assert details["threshold"] == 0.34
        assert details["predicted_churn"] == "1"
        assert details["is_churn"] is True
        assert details["risk_tier"] == "Low Risk"


class TestRenderPredictionCardComponent:
    """Test suite verifying Streamlit native component rendering."""

    @patch("dashboard.components.prediction_card.st")
    def test_render_high_risk_prediction(self, mock_st: MagicMock) -> None:
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.container.return_value.__enter__ = MagicMock()
        mock_st.container.return_value.__exit__ = MagicMock()

        payload = {
            "predicted_churn": "Yes",
            "churn_probability": 0.7025,
            "decision_threshold": 0.34,
            "risk_tier": "High Risk",
        }
        render_prediction_card(payload)

        # Confirm native components were called
        assert mock_st.error.called
        assert mock_st.progress.called
        assert mock_st.warning.called

    @patch("dashboard.components.prediction_card.st")
    def test_render_low_risk_prediction(self, mock_st: MagicMock) -> None:
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.container.return_value.__enter__ = MagicMock()
        mock_st.container.return_value.__exit__ = MagicMock()

        payload = {
            "predicted_churn": "No",
            "churn_probability": 0.15,
            "decision_threshold": 0.34,
            "risk_tier": "Low Risk",
        }
        render_prediction_card(payload)

        # Low risk uses success banner and info recommendation
        assert mock_st.success.called
        assert mock_st.progress.called
        assert mock_st.info.called

    @patch("dashboard.components.prediction_card.st")
    def test_render_missing_and_none_payloads_do_not_crash(self, mock_st: MagicMock) -> None:
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.container.return_value.__enter__ = MagicMock()
        mock_st.container.return_value.__exit__ = MagicMock()

        # Neither empty dict nor None should raise exceptions
        render_prediction_card({})
        render_prediction_card(None)
        assert mock_st.progress.called

