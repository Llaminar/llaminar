# Python Environment Setup for Llaminar Devcontainer

## Overview

This document describes the automated Python environment setup for the Llaminar development container.

## Changes Made

### 1. Dockerfile Updates (`/.devcontainer/Dockerfile`)

**Added:**
- `python3.12-venv` package to support virtual environment creation

**Removed:**
- System-wide Python package installation (`pip install --break-system-packages`)
- This was causing conflicts with the externally-managed-environment error

### 2. Setup Script (`.devcontainer/setup-python-env.sh`)

Created an automated setup script that:

1. **Creates Python Virtual Environment**
   - Location: `/workspaces/llaminar/.venv`
   - Uses Python 3.12
   
2. **Installs Dependencies**
   - Reads from `/workspaces/llaminar/requirements.txt`
   - Installs PyTorch (CPU version)
   - Installs HuggingFace Transformers and related packages
   
3. **Fetches Test Models**
   - Runs `/workspaces/llaminar/scripts/fetch_test_models.sh`
   - Downloads Qwen2.5 models in multiple quantization formats
   
4. **Auto-activation**
   - Adds venv activation to `~/.bashrc`
   - Virtual environment automatically activates on terminal open

### 3. Devcontainer Configuration (`.devcontainer/devcontainer.json`)

**Updated `postCreateCommand`:**
```json
"postCreateCommand": "sudo chown -R vscode:vscode /workspace && git config --global --add safe.directory /workspaces/llaminar && bash /workspaces/llaminar/.devcontainer/setup-python-env.sh && cmake --version && gcc --version"
```

**Added Python-specific VS Code settings:**
```json
"python.defaultInterpreterPath": "/workspaces/llaminar/.venv/bin/python",
"python.terminal.activateEnvironment": true,
"python.terminal.activateEnvInCurrentTerminal": true,
"python.analysis.extraPaths": ["/workspaces/llaminar/python"],
"python.autoComplete.extraPaths": ["/workspaces/llaminar/python"]
```

**Added container environment variables:**
```json
"containerEnv": {
    "VIRTUAL_ENV": "/workspaces/llaminar/.venv",
    "PATH": "/workspaces/llaminar/.venv/bin:${containerEnv:PATH}"
}
```

These settings ensure:
- VS Code Python extension uses the venv interpreter by default
- Python code in `python/` directory is discoverable
- Virtual environment is activated automatically in all contexts
- PATH includes venv binaries first

This runs the Python setup script automatically when the container is created.

## Usage

### Automatic Setup (On Container Creation)

When you rebuild the devcontainer, the Python environment is automatically set up:

1. Virtual environment is created at `/workspaces/llaminar/.venv`
2. All Python dependencies are installed from `requirements.txt`
3. Test models are downloaded to `/workspaces/llaminar/models/`
4. Virtual environment is auto-activated in all new terminals

### Manual Setup (If Needed)

If you need to re-run the setup manually:

```bash
bash /workspaces/llaminar/.devcontainer/setup-python-env.sh
```

### Verifying the Setup

Check that PyTorch is installed:

```bash
source /workspaces/llaminar/.venv/bin/activate
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
```

Check that models are downloaded:

```bash
ls -lh /workspaces/llaminar/models/
```

### Running Parity Tests

With the venv active, run the parity tests:

```bash
# From workspace root
ctest --test-dir build --output-on-failure -R "ParityFramework"
```

## Python Dependencies Installed

From `requirements.txt`:

- **torch>=2.6.0** - PyTorch (CPU version)
- **transformers>=4.45.0** - HuggingFace Transformers
- **safetensors>=0.4.0** - Safe tensor serialization
- **accelerate>=0.34.0** - HuggingFace training acceleration
- **bitsandbytes>=0.43.0** - Quantization support
- **numpy>=1.24.0** - Numerical computing
- **pytest>=8.0.0** - Testing framework
- **sentencepiece>=0.2.0** - Tokenization
- **tokenizers>=0.19.0** - Fast tokenization
- **huggingface-hub>=0.24.0** - Model hub access

## Test Models Downloaded

Six quantization formats of Qwen2.5-0.5B-Instruct (~3GB total):

1. `qwen2.5-0.5b-instruct-q2_k.gguf` (396MB)
2. `qwen2.5-0.5b-instruct-q4_0.gguf` (409MB)
3. `qwen2.5-0.5b-instruct-q4_k_m.gguf` (469MB)
4. `qwen2.5-0.5b-instruct-q5_0.gguf` (468MB)
5. `qwen2.5-0.5b-instruct-q6_k.gguf` (621MB)
6. `qwen2.5-0.5b-instruct-q8_0.gguf` (645MB)

## Troubleshooting

### Virtual Environment Not Activating

If the venv doesn't auto-activate, manually activate it:

```bash
source /workspaces/llaminar/.venv/bin/activate
```

### Dependencies Not Installed

Re-run the setup script:

```bash
rm -rf /workspaces/llaminar/.venv
bash /workspaces/llaminar/.devcontainer/setup-python-env.sh
```

### Models Not Downloaded

Manually fetch models:

```bash
bash /workspaces/llaminar/scripts/fetch_test_models.sh
```

### Import Errors

Ensure you're in the virtual environment:

```bash
which python
# Should output: /workspaces/llaminar/.venv/bin/python
```

## Benefits

1. **No System Package Conflicts** - Uses isolated venv instead of system Python
2. **Reproducible Environment** - All dependencies specified in requirements.txt
3. **Auto-configured** - Works out of the box on container creation
4. **Test-Ready** - Models pre-downloaded for parity testing
5. **Clean Workflow** - Venv auto-activates in every terminal

## Related Files

- `.devcontainer/Dockerfile` - Container image definition
- `.devcontainer/devcontainer.json` - VS Code devcontainer configuration  
- `.devcontainer/setup-python-env.sh` - Python environment setup script
- `requirements.txt` - Python package dependencies
- `scripts/fetch_test_models.sh` - Model download script
- `CMakeLists.txt` - Updated to set correct working directory for tests
