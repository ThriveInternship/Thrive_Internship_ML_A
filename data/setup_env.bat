@echo off
echo ==================================================
echo     🚀 SPTECH - ML & MLOps Environment Setup
echo ==================================================
echo.

REM Create a clean virtual environment
if exist .venv (
    echo [INFO] Existing virtual environment detected. Skipping creation.
) else (
    echo [STEP 1] Creating virtual environment...
    python -m venv .venv
)

echo [STEP 2] Activating virtual environment...
call .venv\Scripts\activate

echo [STEP 3] Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel

echo [STEP 4] Installing core Python packages...
pip install --upgrade pandas numpy matplotlib seaborn scikit-learn ipykernel

echo [STEP 5] Installing MLOps and NLP dependencies...
pip install --upgrade torch==2.4.0 transformers==4.30.2 mlflow==2.5.0 fastapi==0.95.2 uvicorn==0.22.0 prometheus-client pytest python-dotenv

echo [STEP 6] Validating installed packages...
pip check

echo [STEP 7] Ensuring Jupyter Notebook support...
python -m ipykernel install --user --name=.venv --display-name "ML & MLOps Env"

echo.
echo ✅ Environment setup complete!
echo --------------------------------------------------
echo To activate the environment manually, run:
echo     call .venv\Scripts\activate
echo Then launch VS Code and select the "ML & MLOps Env" kernel.
echo --------------------------------------------------
pause
