"""Configuration management for PredictIQ.

Provides typed configuration models, validation, and environment variable
loading using Python standard library facilities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

ALLOWED_ENVIRONMENTS: Set[str] = {
    "development",
    "testing",
    "staging",
    "production",
}

ALLOWED_LOG_LEVELS: Set[str] = {
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
}


class ConfigurationError(ValueError):
    """Raised when application configuration is invalid or missing required values."""
    pass


def _resolve_project_root() -> Path:
    """Resolve repository root directory from file location or environment."""
    custom_root = os.getenv("PREDICTIQ_ROOT_DIR") or os.getenv("PROJECT_ROOT")
    if custom_root:
        return Path(custom_root).resolve()
    # src/predictiq/config.py -> 3 levels up to repo root
    return Path(__file__).resolve().parent.parent.parent


def parse_port(value: str | int, name: str) -> int:
    """Parse and validate network port value."""
    if isinstance(value, float):
        raise ConfigurationError(
            f"Configuration '{name}' must be an integer, got float '{value}'"
        )
    try:
        if isinstance(value, str) and "." in value:
            raise ValueError(f"Port string contains decimal: {value}")
        port_num = int(value)
    except (TypeError, ValueError) as err:
        raise ConfigurationError(
            f"Configuration '{name}' must be an integer, got '{value}'"
        ) from err

    if not (1 <= port_num <= 65535):
        raise ConfigurationError(
            f"Configuration '{name}' must be between 1 and 65535, got {port_num}"
        )
    return port_num


def load_env_file(env_path: Optional[Path | str] = None) -> None:
    """Parse a simple .env file into os.environ without third-party dependencies.

    Existing environment variables are preserved (not overwritten).
    """
    if env_path is None:
        env_path = _resolve_project_root() / ".env"
    else:
        env_path = Path(env_path)

    if not env_path.is_file():
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Remove surrounding single or double quotes if present
                    if (value.startswith('"') and value.endswith('"')) or (
                        value.startswith("'") and value.endswith("'")
                    ):
                        value = value[1:-1]
                    os.environ.setdefault(key, value)
    except OSError as err:
        raise ConfigurationError(f"Failed to read .env file at '{env_path}': {err}") from err


@dataclass(frozen=True)
class PathConfig:
    """Directory paths for the PredictIQ platform."""
    root_dir: Path = field(default_factory=_resolve_project_root)
    data_dir: Path = field(default=Path())
    raw_data_dir: Path = field(default=Path())
    interim_data_dir: Path = field(default=Path())
    processed_data_dir: Path = field(default=Path())
    external_data_dir: Path = field(default=Path())
    artifacts_dir: Path = field(default=Path())

    @classmethod
    def from_env(cls, root: Optional[Path] = None) -> PathConfig:
        root_dir = root or _resolve_project_root()

        custom_data_dir = os.getenv("DATA_DIR")
        data_dir = Path(custom_data_dir).resolve() if custom_data_dir else (root_dir / "data")

        custom_artifacts_dir = os.getenv("ARTIFACTS_DIR")
        artifacts_dir = (
            Path(custom_artifacts_dir).resolve()
            if custom_artifacts_dir
            else (root_dir / "artifacts")
        )

        return cls(
            root_dir=root_dir,
            data_dir=data_dir,
            raw_data_dir=data_dir / "raw",
            interim_data_dir=data_dir / "interim",
            processed_data_dir=data_dir / "processed",
            external_data_dir=data_dir / "external",
            artifacts_dir=artifacts_dir,
        )

    def validate(self) -> None:
        """Validate that project root is an existing directory."""
        if not self.root_dir.is_dir():
            raise ConfigurationError(
                f"Configured project root directory does not exist: {self.root_dir}"
            )


@dataclass(frozen=True)
class DatabaseConfig:
    """PostgreSQL database connectivity parameters."""
    host: str = "localhost"
    port: int = 5432
    user: str = "predictiq"
    password: str = "predictiq_dev_password"
    database: str = "predictiq_mlflow"

    @classmethod
    def from_env(cls) -> DatabaseConfig:
        host = os.getenv("POSTGRES_HOST", "localhost").strip()
        port_raw = os.getenv("POSTGRES_PORT", "5432")
        port = parse_port(port_raw, "POSTGRES_PORT")
        user = os.getenv("POSTGRES_USER", "predictiq").strip()
        password = os.getenv("POSTGRES_PASSWORD", "predictiq_dev_password")
        database = os.getenv("POSTGRES_DB", "predictiq_mlflow").strip()

        config = cls(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate database parameters."""
        if not self.host:
            raise ConfigurationError("Database host must not be empty")
        if not self.user:
            raise ConfigurationError("Database user must not be empty")
        if not self.database:
            raise ConfigurationError("Database name must not be empty")
        if not (1 <= self.port <= 65535):
            raise ConfigurationError(f"Database port must be between 1 and 65535, got {self.port}")

    @property
    def connection_url(self) -> str:
        """Standard PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def sqlalchemy_url(self) -> str:
        """SQLAlchemy driver URL for PostgreSQL."""
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass(frozen=True)
class MLflowConfig:
    """MLflow tracking server configuration."""
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "predictiq-default"
    port: int = 5000
    artifact_destination: str = "/mlflow/artifacts"

    @classmethod
    def from_env(cls) -> MLflowConfig:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000").strip()
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "predictiq-default").strip()
        port_raw = os.getenv("MLFLOW_PORT", "5000")
        port = parse_port(port_raw, "MLFLOW_PORT")
        artifact_dest = os.getenv("MLFLOW_ARTIFACT_DESTINATION", "/mlflow/artifacts").strip()

        config = cls(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            port=port,
            artifact_destination=artifact_dest,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate MLflow configuration values."""
        if not self.tracking_uri:
            raise ConfigurationError("MLflow tracking_uri must not be empty")
        if not self.experiment_name:
            raise ConfigurationError("MLflow experiment_name must not be empty")
        if not (1 <= self.port <= 65535):
            raise ConfigurationError(f"MLflow port must be between 1 and 65535, got {self.port}")


@dataclass(frozen=True)
class Config:
    """Master application configuration container for PredictIQ."""
    environment: str = "development"
    log_level: str = "INFO"
    paths: PathConfig = field(default_factory=PathConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    @classmethod
    def from_env(
        cls,
        root: Optional[Path] = None,
        load_env: bool = True,
        env_file: Optional[Path | str] = None,
    ) -> Config:
        """Instantiate configuration from environment variables."""
        if load_env:
            load_env_file(env_file)

        env_mode = os.getenv("ENVIRONMENT", "development").strip().lower()
        if env_mode not in ALLOWED_ENVIRONMENTS:
            raise ConfigurationError(
                f"Invalid environment '{env_mode}'. Allowed: {sorted(ALLOWED_ENVIRONMENTS)}"
            )

        log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
        if log_level not in ALLOWED_LOG_LEVELS:
            raise ConfigurationError(
                f"Invalid log_level '{log_level}'. Allowed: {sorted(ALLOWED_LOG_LEVELS)}"
            )

        paths = PathConfig.from_env(root=root)
        paths.validate()

        database = DatabaseConfig.from_env()
        mlflow = MLflowConfig.from_env()

        return cls(
            environment=env_mode,
            log_level=log_level,
            paths=paths,
            database=database,
            mlflow=mlflow,
        )

    def validate(self) -> None:
        """Validate all nested configuration objects."""
        if self.environment not in ALLOWED_ENVIRONMENTS:
            raise ConfigurationError(f"Invalid environment '{self.environment}'")
        if self.log_level not in ALLOWED_LOG_LEVELS:
            raise ConfigurationError(f"Invalid log_level '{self.log_level}'")
        self.paths.validate()
        self.database.validate()
        self.mlflow.validate()


_CACHED_CONFIG: Optional[Config] = None


def load_config(
    force_reload: bool = False,
    root: Optional[Path] = None,
    env_file: Optional[Path | str] = None,
) -> Config:
    """Load, validate, and optionally cache the application configuration."""
    global _CACHED_CONFIG
    if _CACHED_CONFIG is None or force_reload:
        _CACHED_CONFIG = Config.from_env(root=root, env_file=env_file)
    return _CACHED_CONFIG


def get_config() -> Config:
    """Convenience getter for cached application configuration."""
    return load_config()
