# PredictIQ — Exploratory Analysis & Notebook Guidelines

This directory houses Jupyter notebooks for interactive analysis, exploratory data analysis (EDA), prototype modeling, and visualization.

## Organization & Naming Conventions
Notebooks should follow a sequential naming convention denoting the pipeline stage and objective:
- `01_eda_<topic>.ipynb`: Initial data exploration, distribution analysis, and anomaly scanning.
- `02_features_<topic>.ipynb`: Feature hypothesis testing and transformation prototyping.
- `03_modeling_<topic>.ipynb`: Model architecture experimentation and metric comparison.
- `04_evaluation_<topic>.ipynb`: Error analysis, residual examination, and fairness checks.

## Guidelines for Reproducibility
1. **Dynamic Environment Configuration**:
   Always load project settings through the centralized configuration module:
   ```python
   from predictiq.config import get_config
   config = get_config()
   raw_data_path = config.paths.raw_data_dir
   ```
2. **Never Hardcode Absolute Paths**:
   Rely on `config.paths` to resolve data and artifact directories relative to the repository root.
3. **Execution Order**:
   Before committing, restart the kernel and run all cells (`Kernel -> Restart & Run All`) to ensure deterministic top-to-bottom execution.
4. **Data Hygiene & Security**:
   - Never store API keys, passwords, database credentials, or secret tokens in notebook cells.
   - Do not commit large datasets into git. Heavy generated outputs should be written to `data/interim/` or `data/processed/` (which are ignored by `.gitignore`).
5. **Promotion to Library Code**:
   Once functions or transformations mature and stabilize, refactor them into modular Python packages under `src/predictiq/` rather than keeping business logic duplicated across notebooks.
