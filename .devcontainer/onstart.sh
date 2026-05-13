#!/bin/bash
# Devcontainer onstart script
# This runs every time the container starts (not just on creation)

set -e

# Log to file for debugging container startup issues
LOGFILE="/tmp/onstart.log"
exec > >(tee -a "$LOGFILE") 2>&1
echo ""
echo "========================================"
echo "[onstart] $(date '+%Y-%m-%d %H:%M:%S') - Starting onstart.sh"
echo "[onstart] Running as user: $(whoami) ($(id))"
echo "========================================"

echo "[onstart] Configuring system limits and permissions..."

# Set unlimited locked memory for GPU P2P transfers and DMA operations
# This is required for ROCm/HIP/HSA runtime to communicate with AMD GPUs.
# Without this, hipGetDeviceCount returns hipErrorNoDevice (error 100)
# and rocminfo fails with HSA_STATUS_ERROR_OUT_OF_RESOURCES.

# Create persistent limits.conf for new login sessions
LIMITS_FILE="/etc/security/limits.d/99-gpu-memlock.conf"
if [ ! -f "$LIMITS_FILE" ]; then
    echo "# GPU P2P transfer requires unlimited locked memory" | sudo tee "$LIMITS_FILE"
    echo "* soft memlock unlimited" | sudo tee -a "$LIMITS_FILE"
    echo "* hard memlock unlimited" | sudo tee -a "$LIMITS_FILE"
    echo "root soft memlock unlimited" | sudo tee -a "$LIMITS_FILE"
    echo "root hard memlock unlimited" | sudo tee -a "$LIMITS_FILE"
    echo "[onstart] Created $LIMITS_FILE for unlimited memlock"
else
    echo "[onstart] $LIMITS_FILE already exists"
fi

# CRITICAL: Apply unlimited memlock to init process (PID 1) so all child processes inherit it
# This is necessary because limits.conf only applies at PAM login, but devcontainer
# processes don't go through PAM. Setting it on PID 1 propagates to all new processes.
if sudo prlimit --memlock=unlimited:unlimited --pid 1 2>/dev/null; then
    echo "[onstart] Set memlock unlimited on init (PID 1) - all new processes will inherit"
else
    echo "[onstart] Warning: Could not set memlock on PID 1"
fi

# Also set it for the current shell session
if sudo prlimit --memlock=unlimited:unlimited --pid $$ 2>/dev/null; then
    echo "[onstart] Set memlock unlimited for current process ($$)"
fi

# Verify the setting
CURRENT_MEMLOCK=$(ulimit -l 2>/dev/null || echo "unknown")
echo "[onstart] Current memlock limit: $CURRENT_MEMLOCK"

# Fix render device permissions (host may assign different GIDs)
# The host's render group GID may not match the container's render group,
# so we use chmod 666 to give everyone access. We also try chgrp as backup.
if [ -d /dev/dri ]; then
    # Log current state for debugging
    echo "[onstart] Current /dev/dri permissions:"
    ls -la /dev/dri/ 2>/dev/null || true
    
    # Wait briefly for devices to be fully available (race condition mitigation)
    sleep 1
    
    # Fix render device permissions - try multiple approaches
    for dev in /dev/dri/render*; do
        if [ -e "$dev" ]; then
            sudo chmod 666 "$dev" 2>/dev/null || true
            # Also try to change group to render (may fail if render group doesn't exist)
            sudo chgrp render "$dev" 2>/dev/null || true
        fi
    done
    
    # Verify fix was applied
    echo "[onstart] Fixed /dev/dri permissions:"
    ls -la /dev/dri/ 2>/dev/null || true
fi

# Ensure kfd device is accessible
if [ -e /dev/kfd ]; then
    sudo chmod 666 /dev/kfd 2>/dev/null || true
    sudo chgrp video /dev/kfd 2>/dev/null || true
    echo "[onstart] Fixed /dev/kfd permissions"
fi

# Report GPU status
echo "[onstart] Checking GPU availability..."

# Check CUDA GPUs
if command -v nvidia-smi &>/dev/null; then
    CUDA_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    echo "[onstart] Found $CUDA_COUNT CUDA GPU(s)"
fi

# Check ROCm GPUs
if command -v rocm-smi &>/dev/null; then
    ROCM_COUNT=$(rocm-smi --showid 2>/dev/null | grep -c "GPU" || echo 0)
    echo "[onstart] Found $ROCM_COUNT ROCm GPU(s)"
fi

# Create models symlink when the workspace does not already have a models path.
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspaces/llaminar}"
if [ ! -e "$WORKSPACE_DIR/models" ]; then
    ln -s /opt/llaminar-models "$WORKSPACE_DIR/models"
    echo "[onstart] Created symlink for models directory"
elif [ -L "$WORKSPACE_DIR/models" ]; then
    echo "[onstart] Symlink for models directory already exists"
else
    echo "[onstart] Workspace models directory already exists"
fi

echo "[onstart] Done."
