"""Deterministic foundation tests for PredictIQ configuration and project integrity."""

import os
from pathlib import Path

import pytest

import predictiq
from predictiq.config import (
    Config,
    ConfigurationError,
    DatabaseConfig,
    MLflowConfig,
    PathConfig,
    get_config,
    load_config,
    load_env_file,
    parse_port,
)


class TestProjectIntegrity:
    """Validate core package installation, versioning, and interface contract."""

    def test_package_metadata(self):
        """Package must expose standard version string."""
        assert hasattr(predictiq, "__version__")
        assert isinstance(predictiq.__version__, str)
        assert len(predictiq.__version__.split(".")) >= 3

    def test_package_exports(self):
        """Top-level package must export essential config classes and factories."""
        expected_exports = {
            "Config",
            "ConfigurationError",
            "DatabaseConfig",
            "MLflowConfig",
            "PathConfig",
            "get_config",
            "load_config",
        }
        for export_name in expected_exports:
            assert hasattr(predictiq, export_name)
            assert export_name in predictiq.__all__


class TestPortParsing:
    """Validate network port validation rules."""

    @pytest.mark.parametrize("valid_port", [1, 80, 5000, 5432, 65535, "8080", "5000"])
    def test_parse_valid_ports(self, valid_port):
        parsed = parse_port(valid_port, "TEST_PORT")
        assert isinstance(parsed, int)
        assert 1 <= parsed <= 65535

    @pytest.mark.parametrize("invalid_port", [0, -1, 65536, 70000, "abc", "", None, 3.14])
    def test_parse_invalid_ports(self, invalid_port):
        with pytest.raises(ConfigurationError):
            parse_port(invalid_port, "TEST_PORT")


class TestPathConfiguration:
    """Validate path resolution and directory tree mapping."""

    def test_default_paths(self):
        paths = PathConfig.from_env()
        assert paths.root_dir.is_dir()
        assert paths.data_dir == paths.root_dir / "data"
        assert paths.raw_data_dir == paths.root_dir / "data" / "raw"
        assert paths.interim_data_dir == paths.root_dir / "data" / "interim"
        assert paths.processed_data_dir == paths.root_dir / "data" / "processed"
        assert paths.external_data_dir == paths.root_dir / "data" / "external"
        assert paths.artifacts_dir == paths.root_dir / "artifacts"

    def test_custom_paths_override(self, monkeypatch, tmp_path):
        custom_data = tmp_path / "custom_data"
        custom_artifacts = tmp_path / "custom_artifacts"
        monkeypatch.setenv("DATA_DIR", str(custom_data))
        monkeypatch.setenv("ARTIFACTS_DIR", str(custom_artifacts))

        paths = PathConfig.from_env(root=tmp_path)
        assert paths.data_dir == custom_data.resolve()
        assert paths.artifacts_dir == custom_artifacts.resolve()
        assert paths.raw_data_dir == custom_data.resolve() / "raw"


class TestDatabaseConfiguration:
    """Validate database config defaults, validation, and URL builders."""

    def test_defaults(self, monkeypatch):
        # Clear any environment variables that might alter defaults
        for env_var in [
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_DB",
        ]:
            monkeypatch.delenv(env_var, raising=False)

        db = DatabaseConfig.from_env()
        assert db.host == "localhost"
        assert db.port == 5432
        assert db.user == "predictiq"
        assert db.password == "predictiq_dev_password"
        assert db.database == "predictiq_mlflow"
        assert "postgresql://predictiq:predictiq_dev_password@localhost:5432/predictiq_mlflow" == db.connection_url
        assert "postgresql+psycopg2://predictiq:predictiq_dev_password@localhost:5432/predictiq_mlflow" == db.sqlalchemy_url

    def test_overrides(self, monkeypatch):
        monkeypatch.setenv("POSTGRES_HOST", "db.internal")
        monkeypatch.setenv("POSTGRES_PORT", "5433")
        monkeypatch.setenv("POSTGRES_USER", "custom_user")
        monkeypatch.setenv("POSTGRES_PASSWORD", "custom_pass")
        monkeypatch.setenv("POSTGRES_DB", "custom_db")

        db = DatabaseConfig.from_env()
        assert db.host == "db.internal"
        assert db.port == 5433
        assert db.user == "custom_user"
        assert db.password == "custom_pass"
        assert db.database == "custom_db"

    @pytest.mark.parametrize(
        "kwargs, error_match",
        [
            ({"host": ""}, "[Hh]ost must not be empty"),
            ({"user": ""}, "[Uu]ser must not be empty"),
            ({"database": ""}, "[Dd]atabase name must not be empty"),
            ({"port": 70000}, "[Pp]ort must be between 1 and 65535"),
        ],
    )
    def test_invalid_database_params(self, kwargs, error_match):
        base = {
            "host": "localhost",
            "port": 5432,
            "user": "predictiq",
            "password": "pw",
            "database": "db",
        }
        base.update(kwargs)
        cfg = DatabaseConfig(**base)
        with pytest.raises(ConfigurationError, match=error_match):
            cfg.validate()


class TestMLflowConfiguration:
    """Validate MLflow configuration defaults and validation."""

    def test_defaults(self, monkeypatch):
        for env_var in [
            "MLFLOW_TRACKING_URI",
            "MLFLOW_EXPERIMENT_NAME",
            "MLFLOW_PORT",
            "MLFLOW_ARTIFACT_DESTINATION",
        ]:
            monkeypatch.delenv(env_var, raising=False)

        mlflow_cfg = MLflowConfig.from_env()
        assert mlflow_cfg.tracking_uri == "http://localhost:5000"
        assert mlflow_cfg.experiment_name == "predictiq-default"
        assert mlflow_cfg.port == 5000
        assert mlflow_cfg.artifact_destination == "/mlflow/artifacts"

    def test_overrides(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.internal:5001")
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "churn-experiment")
        monkeypatch.setenv("MLFLOW_PORT", "5001")
        monkeypatch.setenv("MLFLOW_ARTIFACT_DESTINATION", "s3://bucket/mlflow")

        mlflow_cfg = MLflowConfig.from_env()
        assert mlflow_cfg.tracking_uri == "http://mlflow.internal:5001"
        assert mlflow_cfg.experiment_name == "churn-experiment"
        assert mlflow_cfg.port == 5001
        assert mlflow_cfg.artifact_destination == "s3://bucket/mlflow"

    def test_invalid_mlflow_port(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_PORT", "99999")
        with pytest.raises(ConfigurationError):
            MLflowConfig.from_env()


class TestMasterConfiguration:
    """Validate top-level Config factory, environment modes, and caching."""

    def test_default_config(self, monkeypatch):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        monkeypatch.delenv("LOG_LEVEL", raising=False)

        cfg = Config.from_env(load_env=False)
        assert cfg.environment == "development"
        assert cfg.log_level == "INFO"
        assert isinstance(cfg.paths, PathConfig)
        assert isinstance(cfg.database, DatabaseConfig)
        assert isinstance(cfg.mlflow, MLflowConfig)

    @pytest.mark.parametrize("env_mode", ["development", "testing", "staging", "production"])
    def test_allowed_environments(self, monkeypatch, env_mode):
        monkeypatch.setenv("ENVIRONMENT", env_mode)
        cfg = Config.from_env(load_env=False)
        assert cfg.environment == env_mode

    def test_invalid_environment_raises(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "invalid_env_mode")
        with pytest.raises(ConfigurationError, match="Invalid environment"):
            Config.from_env(load_env=False)

    @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_allowed_log_levels(self, monkeypatch, log_level):
        monkeypatch.setenv("LOG_LEVEL", log_level)
        cfg = Config.from_env(load_env=False)
        assert cfg.log_level == log_level

    def test_invalid_log_level_raises(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        with pytest.raises(ConfigurationError, match="Invalid log_level"):
            Config.from_env(load_env=False)

    def test_load_and_get_config_caching(self):
        cfg1 = load_config(force_reload=True)
        cfg2 = get_config()
        assert cfg1 is cfg2

        cfg3 = load_config(force_reload=True)
        assert cfg3 is not None


class TestDotEnvLoader:
    """Validate custom stdlib .env parser."""

    def test_load_env_file(self, tmp_path, monkeypatch):
        env_file = tmp_path / "test.env"
        env_file.write_text(
            "# Comment line\n"
            "TEST_VAR_A=hello\n"
            "TEST_VAR_B=\"quoted value\"\n"
            "TEST_VAR_C='single quoted'\n"
            "EXISTING_VAR=new_value\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("EXISTING_VAR", "original_value")

        load_env_file(env_file)
        assert os.getenv("TEST_VAR_A") == "hello"
        assert os.getenv("TEST_VAR_B") == "quoted value"
        assert os.getenv("TEST_VAR_C") == "single quoted"
        # Preserves pre-existing environment variable
        assert os.getenv("EXISTING_VAR") == "original_value"
