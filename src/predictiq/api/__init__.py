"""PredictIQ REST API package."""

from predictiq.api.app import create_app
from predictiq.api.main import app

__all__ = ["create_app", "app"]
