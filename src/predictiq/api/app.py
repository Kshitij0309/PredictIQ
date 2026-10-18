"""FastAPI application factory, lifespan management, and middleware for PredictIQ."""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from predictiq.api.routes import router
from predictiq.api.schemas import ErrorResponse
from predictiq.config import Config, get_config
from predictiq.inference.predictor import ChurnPredictor, InferenceValidationError

logger = logging.getLogger("predictiq.api")


def create_app(
    predictor: Optional[ChurnPredictor] = None,
    config: Optional[Config] = None,
) -> FastAPI:
    """Create and configure a production-ready FastAPI application instance.

    Args:
        predictor: Optional pre-loaded ChurnPredictor instance (useful for testing).
        config: Optional pre-loaded Config instance.

    Returns:
        FastAPI: Configured application instance.
    """
    app_config = config or get_config()
    is_production = app_config.environment == "production"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: Record boot timestamp
        app.state.start_time = time.time()

        if predictor is not None:
            logger.info("Using pre-injected ChurnPredictor instance.")
            app.state.predictor = predictor
            app.state.model_loaded = True
            app.state.metadata = predictor.metadata
            app.state.load_error = None
        else:
            model_path_env = os.getenv("MODEL_PATH")
            try:
                logger.info(f"Loading champion model pipeline (MODEL_PATH='{model_path_env}')...")
                loaded_predictor = ChurnPredictor.from_saved_model(model_path=model_path_env)
                app.state.predictor = loaded_predictor
                app.state.model_loaded = True
                app.state.metadata = loaded_predictor.metadata
                app.state.load_error = None
                logger.info(
                    f"Model pipeline successfully loaded: '{loaded_predictor.metadata.get('model_name')}' "
                    f"with frozen threshold p*={loaded_predictor.decision_threshold}"
                )
            except Exception as err:
                logger.error(f"Startup model loading failed: {err}")
                app.state.predictor = None
                app.state.model_loaded = False
                app.state.metadata = {}
                app.state.load_error = str(err)

                strict_startup = os.getenv("PREDICTIQ_STRICT_STARTUP", "false").lower() in ("true", "1")
                if strict_startup:
                    raise RuntimeError(
                        f"PredictIQ API startup aborted: required model artifact could not be loaded: {err}"
                    ) from err

        yield

        # Shutdown
        logger.info("PredictIQ API service shutdown complete.")

    # Configure documentation availability: hidden in production unless explicitly overridden
    enable_docs = os.getenv("ENABLE_DOCS", "false").lower() in ("true", "1") or not is_production
    docs_url = "/docs" if enable_docs else None
    redoc_url = "/redoc" if enable_docs else None
    openapi_url = "/openapi.json" if enable_docs else None

    app = FastAPI(
        title="PredictIQ Inference API",
        description="Production predictive intelligence REST service for customer churn probability scoring.",
        version="0.1.0",
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )

    # Attach config
    app.state.config = app_config

    # Middleware: Request ID tracking and structured latency logging
    @app.middleware("http")
    async def structured_logging_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        start_t = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_t) * 1000.0
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

        # Safe logging: logs HTTP metadata only, strictly never customer record attributes
        logger.info(
            f"request_id={req_id} method={request.method} path={request.url.path} "
            f"status={response.status_code} latency_ms={duration_ms:.2f}"
        )
        return response

    # Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError):
        error_details = []
        for err in exc.errors():
            loc = " -> ".join(str(item) for item in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            error_details.append(f"{loc}: {msg}")

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="Request Validation Error",
                detail="; ".join(error_details) if error_details else "Invalid request payload format.",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
        )

    @app.exception_handler(InferenceValidationError)
    async def inference_validation_handler(request: Request, exc: InferenceValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="Inference Validation Error",
                detail=str(exc),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=f"HTTP {exc.status_code}",
                detail=str(exc.detail),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error processing {request.method} {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal Server Error",
                detail="An unexpected error occurred while processing the request.",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
        )

    # Register API routes
    app.include_router(router)

    return app
