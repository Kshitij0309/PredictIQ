# PredictIQ — Windows PowerShell Development Setup Guide

This guide provides exact step-by-step instructions for establishing, running, and verifying the local PredictIQ development environment on Windows using PowerShell.

---

## 1. Prerequisites

Before beginning, ensure the following software is installed on your Windows workstation:

| Prerequisite | Minimum Version | Verification Command | Expected Output |
|---|---|---|---|
| **Python** | 3.10+ (Verified on 3.13) | `python --version` | `Python 3.13.x` |
| **Git** | 2.40+ | `git --version` | `git version 2.x.windows.x` |
| **Docker Desktop** | 24.0+ / Compose v2.20+ | `docker compose version` | `Docker Compose version v2.x` or `v5.x` |

---

## 2. Directory & Repository Setup

Open **Windows PowerShell** (or Windows Terminal) and navigate to the project directory:

```powershell
# Navigate to the PredictIQ repository root
Set-Location -Path "D:\Projects\PredictIQ"

# Verify directory contents
Get-ChildItem -Name
```

**Expected Output:**
```
.env.example
.gitignore
artifacts
CURRENT_STATE.md
data
docker-compose.yml
docs
NEXT_TASK.md
notebooks
pyproject.toml
README.md
src
tests
```

---

## 3. Python Virtual Environment Configuration

Create and activate an isolated Python virtual environment inside the repository:

```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment (Process-scoped bypass if execution policy restricts scripts)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& ".\.venv\Scripts\Activate.ps1"

# 3. Verify Python executable path
(Get-Command python).Source
```

**Expected Output:**
```
D:\Projects\PredictIQ\.venv\Scripts\python.exe
```

---

## 4. Install Dependencies

Install the foundation package and development dependencies in editable mode:

```powershell
# Upgrade pip to latest version
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip

# Install PredictIQ in editable mode with development dependencies
& ".\.venv\Scripts\pip.exe" install -e ".[dev]"
```

**Expected Output:**
```
Successfully installed predictiq-0.1.0 pytest-9.x ...
```

---

## 5. Local Environment Variables (.env)

Create your local `.env` configuration file from the template:

```powershell
# Copy template to .env
Copy-Item -Path ".env.example" -Destination ".env"

# Verify file creation
Test-Path -Path ".env"
```

**Expected Output:**
```
True
```

*Note: Default settings in `.env` are configured for local development on `localhost`.*

---

## 6. Run Foundation Tests

Execute the automated pytest suite to verify project integrity and configuration validation:

```powershell
# Execute test suite with verbose output
& ".\.venv\Scripts\pytest.exe" tests -v
```

**Expected Output:**
```
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.1.1, pluggy-1.7.0
rootdir: D:\Projects\PredictIQ
configfile: pyproject.toml
testpaths: tests
collected 42 items

tests/test_config.py::TestProjectIntegrity::test_package_metadata PASSED [  2%]
tests/test_config.py::TestProjectIntegrity::test_package_exports PASSED  [  4%]
tests/test_config.py::TestPortParsing::test_parse_valid_ports PASSED     [ 21%]
tests/test_config.py::TestPortParsing::test_parse_invalid_ports PASSED   [ 40%]
tests/test_config.py::TestPathConfiguration::test_default_paths PASSED   [ 42%]
tests/test_config.py::TestDatabaseConfiguration::test_defaults PASSED    [ 47%]
tests/test_config.py::TestMLflowConfiguration::test_defaults PASSED      [ 66%]
tests/test_config.py::TestMasterConfiguration::test_default_config PASSED[ 83%]
tests/test_config.py::TestDotEnvLoader::test_load_env_file PASSED        [100%]

============================= 42 passed in 0.08s ==============================
```

---

## 7. Docker Infrastructure Operations

Docker Compose manages PostgreSQL and MLflow services for experiment tracking and metadata persistence.

### Validate Compose Configuration
```powershell
docker compose config
```

### Start Services (When Docker Desktop is Running)
```powershell
# Start services in detached mode
docker compose up -d

# Check running container status and health
docker compose ps
```

**Expected Output:**
```
NAME                 IMAGE                     COMMAND                  SERVICE    STATUS
predictiq-postgres   postgres:16.8-alpine      "docker-entrypoint.s…"   postgres   Up (healthy)
predictiq-mlflow     ghcr.io/mlflow/mlflow:v2.19.0 "mlflow server --hos…" mlflow     Up (healthy)
```

### View Service Logs
```powershell
# Follow logs for all services
docker compose logs -f

# Follow logs for MLflow specifically
docker compose logs -f mlflow
```

### Access Local Endpoints
- **MLflow Tracking UI:** [http://localhost:5000](http://localhost:5000)
- **PostgreSQL Database:** `127.0.0.1:5432` (User: `predictiq`, Database: `predictiq_mlflow`)

### Stop Services
```powershell
# Stop containers while preserving persistent volumes
docker compose down

# Stop containers and remove volumes (DESTRUCTIVE - wipes tracking metadata)
# docker compose down -v
```

---

## 8. Troubleshooting & Recovery Guidance

### Issue 1: PowerShell Script Execution Policy Blocked
**Symptom:**
```
File D:\Projects\PredictIQ\.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.
```
**Resolution:**
Run this in your current PowerShell session to enable execution for only the current process:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Issue 2: Docker Daemon Unavailable (`npipe` Error)
**Symptom:**
```
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```
**Resolution:**
1. Open the Start Menu, search for **Docker Desktop**, and launch it.
2. Wait until the whale icon in the system tray shows Docker is running and healthy.
3. Test connectivity with:
   ```powershell
   docker info
   ```

### Issue 3: Port Collision (Port 5432 or 5000 already in use)
**Symptom:**
`Bind for 127.0.0.1:5432 failed: port is already allocated`
**Resolution:**
Check what process is bound to the conflicting port:
```powershell
Get-NetTCPConnection -LocalPort 5432 -ErrorAction SilentlyContinue | Select-Object OwningProcess
```
Either terminate the competing process or override the mapped port in `.env`:
```ini
POSTGRES_PORT=5433
MLFLOW_PORT=5001
```
Then restart containers with `docker compose up -d`.
