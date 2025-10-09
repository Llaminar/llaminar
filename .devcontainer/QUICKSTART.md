# Quick Start: Python Environment for Parity Testing

## First Time Setup (Automatic on Container Creation)

When you open the devcontainer for the first time or rebuild it:

1. ✅ Python venv is created at `.venv/`
2. ✅ PyTorch & dependencies installed
3. ✅ Test models downloaded (~3GB)
4. ✅ Venv auto-activates in all terminals

**No manual action needed!**

## Verify Setup

```bash
# Check Python environment
python --version
which python  # Should show /workspaces/llaminar/.venv/bin/python

# Check packages
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"

# Check models
ls -lh models/*.gguf
```

## Running Parity Tests

```bash
# Run OpenBLAS vs PyTorch parity test
ctest --test-dir build --output-on-failure -R "ParityFramework.OpenBLASPrefillVsPyTorch" -V

# Run all parity tests
ctest --test-dir build --output-on-failure -R "ParityFramework"
```

## Common Tasks

### Re-run Setup Script

```bash
bash .devcontainer/setup-python-env.sh
```

### Clean Re-install

```bash
rm -rf .venv
bash .devcontainer/setup-python-env.sh
```

### Add New Python Package

```bash
# Add to requirements.txt first, then:
source .venv/bin/activate
pip install -r requirements.txt
```

### Check Venv Status

```bash
# See if venv is active (should show .venv in prompt)
echo $VIRTUAL_ENV

# Manually activate if needed
source .venv/bin/activate
```

## Testing Workflow

1. **Build C++ code:**
   ```bash
   cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
   cmake --build build --parallel
   ```

2. **Run smoke tests:**
   ```bash
   ctest --test-dir build -R "Smoke"
   ```

3. **Run parity tests (needs Python):**
   ```bash
   ctest --test-dir build -R "ParityFramework"
   ```

4. **Run specific test:**
   ```bash
   ctest --test-dir build -R "OpenBLASPrefillVsPyTorch" -V
   ```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Activate venv: `source .venv/bin/activate` |
| `No test model found` | Run: `bash scripts/fetch_test_models.sh` |
| Venv not auto-activating | Check `.bashrc` has venv activation line |
| Import errors | Re-install: `pip install -r requirements.txt` |

## File Locations

- **Virtual environment:** `/workspaces/llaminar/.venv/`
- **Python packages:** `/workspaces/llaminar/.venv/lib/python3.12/site-packages/`
- **Test models:** `/workspaces/llaminar/models/`
- **Setup script:** `/workspaces/llaminar/.devcontainer/setup-python-env.sh`
- **Requirements:** `/workspaces/llaminar/requirements.txt`
