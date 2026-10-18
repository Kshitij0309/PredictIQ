"""Default application entrypoint for the PredictIQ FastAPI REST service."""

from __future__ import annotations

import os

import uvicorn

from predictiq.api.app import create_app

# Default application instance for Uvicorn ASGI servers
app = create_app()


def run() -> None:
    """Run the API server with environment-driven host and port settings."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    reload = os.getenv("ENVIRONMENT", "development").lower() == "development"

    uvicorn.run(
        "predictiq.api.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
    )


if __name__ == "__main__":
    run()
