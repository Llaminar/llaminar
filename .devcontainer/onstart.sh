#!/bin/bash
# Devcontainer onstart script
# This runs every time the container starts (not just on creation)

set -e

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
if [ -d /dev/dri ]; then
    sudo chmod 666 /dev/dri/render* 2>/dev/null || true
    echo "[onstart] Fixed /dev/dri/render* permissions"
fi

# Ensure kfd device is accessible
if [ -e /dev/kfd ]; then
    sudo chmod 666 /dev/kfd 2>/dev/null || true
    echo "[onstart] Fixed /dev/kfd permissions"
fi

# Enable PCIe BAR access for AMD GPUs (required for cross-vendor P2P: CUDA <-> ROCm)
# This allows DirectP2P to map AMD GPU BARs for direct PCIe transfers
echo "[onstart] Setting up PCIe BAR access for AMD GPUs..."
AMD_BAR_COUNT=0
for device in /sys/bus/pci/devices/*; do
    if [ -f "$device/vendor" ] && [ -f "$device/class" ]; then
        vendor=$(cat "$device/vendor" 2>/dev/null)
        class=$(cat "$device/class" 2>/dev/null)
        # AMD vendor (0x1002) and display class (0x03xxxx)
        if [ "$vendor" = "0x1002" ] && [[ "$class" == 0x03* ]]; then
            if [ -f "$device/resource0" ]; then
                size=$(stat -c%s "$device/resource0" 2>/dev/null || echo 0)
                if [ "$size" -gt 0 ]; then
                    sudo chmod 666 "$device/resource0" 2>/dev/null || true
                    AMD_BAR_COUNT=$((AMD_BAR_COUNT + 1))
                fi
            fi
        fi
    fi
done
if [ "$AMD_BAR_COUNT" -gt 0 ]; then
    echo "[onstart] Enabled PCIe BAR access for $AMD_BAR_COUNT AMD GPU(s)"
else
    echo "[onstart] No AMD GPU BARs found (PCIe BAR P2P not available)"
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

echo "[onstart] Done."
