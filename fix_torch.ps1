# ===========================================
# Fix PyTorch DLL Initialization Error (CPU)
# ===========================================

Write-Host "🔧 Cleaning old environment..." -ForegroundColor Cyan

# Remove old virtual environment (if exists)
if (Test-Path ".\venv") {
    Remove-Item -Recurse -Force ".\venv"
    Write-Host "Old virtual environment removed." -ForegroundColor Yellow
}

# Create new virtual environment
Write-Host "🆕 Creating new virtual environment..."
python -m venv venv

# Activate it
Write-Host "⚙️ Activating environment..."
& ".\venv\Scripts\activate.ps1"

# Upgrade pip
Write-Host "⬆️ Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
Write-Host "📦 Installing required libraries..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib mlflow

# Verify torch import
Write-Host "🧠 Verifying PyTorch installation..."
$torchCheck = python -c "import torch; print('PyTorch OK ✅', torch.__version__)"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ PyTorch successfully installed and verified!" -ForegroundColor Green
} else {
    Write-Host "`n❌ PyTorch import failed. Check above errors." -ForegroundColor Red
}

Write-Host "`nAll done! You can now activate your environment with:`n    venv\Scripts\activate"
