#!/bin/bash
set -e

echo "=========================================="
echo "Setting up Python environment for Llaminar"
echo "=========================================="

# Define paths
if [[ "$GITHUB_ACTIONS" == "true" ]]; then
  WORKSPACE_DIR="."
else
  WORKSPACE_DIR="/workspaces/llaminar"
fi

VENV_DIR="$WORKSPACE_DIR/.venv"
REQUIREMENTS_FILE="$WORKSPACE_DIR/requirements.txt"
FETCH_MODELS_SCRIPT="$WORKSPACE_DIR/scripts/fetch_test_models.sh"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created at $VENV_DIR"
else
    echo "[1/4] Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
echo "[2/4] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "    Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies if requirements.txt exists
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "[3/4] Installing Python dependencies from requirements.txt..."
    pip install -r "$REQUIREMENTS_FILE"
    echo "✓ Dependencies installed successfully"
else
    echo "[3/4] No requirements.txt found, skipping dependency installation"
fi

# Install python/reference package in editable mode for snapshot generation
PYTHON_REFERENCE_DIR="$WORKSPACE_DIR/python"
if [ -d "$PYTHON_REFERENCE_DIR" ]; then
    echo "    Installing python/reference package in editable mode..."
    # Add the python directory to PYTHONPATH for snapshot scripts
    export PYTHONPATH="$PYTHON_REFERENCE_DIR:$PYTHONPATH"
    echo "    ✓ Added $PYTHON_REFERENCE_DIR to PYTHONPATH"
    echo "    Note: python/reference modules now importable for snapshot generation"
fi

# Fetch test models if script exists
if [ -f "$FETCH_MODELS_SCRIPT" ]; then
    echo "[4/4] Fetching test models..."
    bash "$FETCH_MODELS_SCRIPT"
    echo "✓ Test models fetched successfully"
else
    echo "[4/4] Model fetch script not found at $FETCH_MODELS_SCRIPT, skipping"
fi

# Add venv activation to bashrc if not already there
BASHRC="$HOME/.bashrc"
VENV_ACTIVATION="source $VENV_DIR/bin/activate"

if ! grep -q "$VENV_ACTIVATION" "$BASHRC" 2>/dev/null; then
    echo "" >> "$BASHRC"
    echo "# Auto-activate Llaminar Python virtual environment" >> "$BASHRC"
    echo "$VENV_ACTIVATION" >> "$BASHRC"
    echo "✓ Added venv auto-activation to ~/.bashrc"
fi

# Add PYTHONPATH to bashrc for persistent snapshot script support
PYTHONPATH_EXPORT="export PYTHONPATH=\"$PYTHON_REFERENCE_DIR:\$PYTHONPATH\""
if ! grep -q "PYTHONPATH.*$PYTHON_REFERENCE_DIR" "$BASHRC" 2>/dev/null; then
    echo "" >> "$BASHRC"
    echo "# Add python/reference to PYTHONPATH for snapshot generation" >> "$BASHRC"
    echo "$PYTHONPATH_EXPORT" >> "$BASHRC"
    echo "✓ Added PYTHONPATH configuration to ~/.bashrc"
fi

echo ""
echo "=========================================="
echo "Python environment setup complete!"
echo "=========================================="
echo "Virtual environment: $VENV_DIR"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo ""
echo "To manually activate the environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo "=========================================="
