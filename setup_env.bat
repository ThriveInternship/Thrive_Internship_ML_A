@echo off
echo ==========================================
echo 🚀 Setting up Python Virtual Environment
echo ==========================================

REM Create virtual environment
python -m venv .venv

REM Activate virtual environment
call .venv\Scripts\activate

echo.
echo ==========================================
echo 🔄 Upgrading pip, setuptools, and wheel
echo ==========================================
python -m pip install --upgrade pip setuptools wheel

echo.
echo ==========================================
echo ⚙️ Installing Build Essentials
echo ==========================================
REM Optional: ensures numpy can build if no binary is available
pip install meson ninja cmake

echo.
echo ==========================================
echo 📦 Installing Project Dependencies
echo ==========================================
REM Install from requirements.txt if available
if exist requirements.txt (
    pip install --upgrade -r requirements.txt
) else (
    echo ⚠️ requirements.txt not found! Installing core packages manually...
    pip install --only-binary=:all: numpy
    pip install pandas matplotlib seaborn scikit-learn fastapi uvicorn mlflow prometheus-client pytest python-dotenv
)

echo.
echo ==========================================
echo ✅ Installation Complete
echo ==========================================
echo.
python -m pip show pandas >nul 2>&1
if %errorlevel%==0 (
    echo 🧠 Environment ready! You can now open VS Code and select the .venv kernel.
) else (
    echo ❌ Something went wrong. Try rerunning this script as Administrator.
)

pause
