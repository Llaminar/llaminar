#!/bin/bash
#
# Setup PCIe BAR access for AMD GPUs
# This enables cross-vendor GPU P2P via PCIe BAR between NVIDIA and AMD GPUs
#
# Usage: sudo ./scripts/setup-pcie-bar-access.sh [--temp|--permanent]
#   --temp      : One-time permission change (lost on reboot)
#   --permanent : Install udev rule (persistent)
#
# Requirements: root/sudo access

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

UDEV_RULE_PATH="/etc/udev/rules.d/99-amd-gpu-bar.rules"

# Check for root
if [ "$(id -u)" != "0" ]; then
    echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
    exit 1
fi

# Find AMD GPU BAR files
find_amd_bars() {
    local bars=()
    for device in /sys/bus/pci/devices/*; do
        if [ -f "$device/vendor" ] && [ -f "$device/class" ]; then
            vendor=$(cat "$device/vendor")
            class=$(cat "$device/class")
            # AMD vendor (0x1002) and display class (0x03xxxx)
            if [ "$vendor" = "0x1002" ] && [[ "$class" == 0x03* ]]; then
                if [ -f "$device/resource0" ]; then
                    local size=$(stat -c%s "$device/resource0" 2>/dev/null || echo 0)
                    if [ "$size" -gt 0 ]; then
                        bars+=("$device/resource0")
                    fi
                fi
            fi
        fi
    done
    echo "${bars[@]}"
}

# Temporary permission change
setup_temp() {
    echo -e "${YELLOW}Setting temporary permissions on AMD GPU BAR files...${NC}"
    local bars=($(find_amd_bars))
    
    if [ ${#bars[@]} -eq 0 ]; then
        echo -e "${RED}No AMD GPU BAR files found${NC}"
        exit 1
    fi
    
    for bar in "${bars[@]}"; do
        echo "  chmod 666 $bar"
        chmod 666 "$bar"
    done
    
    echo -e "${GREEN}Done! AMD GPU BARs are now accessible.${NC}"
    echo -e "${YELLOW}Note: This is temporary and will be lost on reboot.${NC}"
    echo "      Use --permanent to install a udev rule for persistence."
}

# Permanent udev rule installation
setup_permanent() {
    echo -e "${YELLOW}Installing udev rule for AMD GPU BAR access...${NC}"
    
    cat > "$UDEV_RULE_PATH" << 'EOF'
# AMD GPU PCIe BAR access for cross-vendor P2P (CUDA <-> ROCm)
# Allows non-root users to access AMD GPU BARs for direct PCIe transfers
SUBSYSTEM=="pci", ATTR{vendor}=="0x1002", ATTR{class}=="0x030000", \
    RUN+="/bin/chmod 0666 /sys/bus/pci/devices/%k/resource0"
SUBSYSTEM=="pci", ATTR{vendor}=="0x1002", ATTR{class}=="0x030200", \
    RUN+="/bin/chmod 0666 /sys/bus/pci/devices/%k/resource0"
EOF

    echo "  Created $UDEV_RULE_PATH"
    
    # Reload udev rules
    echo "  Reloading udev rules..."
    udevadm control --reload-rules
    
    # Trigger for existing devices
    echo "  Triggering udev for existing AMD GPUs..."
    udevadm trigger --subsystem-match=pci --attr-match=vendor=0x1002
    
    # Also apply immediately to existing files
    setup_temp
    
    echo -e "${GREEN}Done! udev rule installed and permissions applied.${NC}"
    echo "      This will persist across reboots."
}

# Remove udev rule
remove_permanent() {
    echo -e "${YELLOW}Removing udev rule...${NC}"
    
    if [ -f "$UDEV_RULE_PATH" ]; then
        rm -f "$UDEV_RULE_PATH"
        udevadm control --reload-rules
        echo -e "${GREEN}Removed $UDEV_RULE_PATH${NC}"
    else
        echo -e "${YELLOW}No udev rule found at $UDEV_RULE_PATH${NC}"
    fi
}

# Show status
show_status() {
    echo "=== AMD GPU PCIe BAR Access Status ==="
    echo ""
    
    if [ -f "$UDEV_RULE_PATH" ]; then
        echo -e "udev rule: ${GREEN}Installed${NC} ($UDEV_RULE_PATH)"
    else
        echo -e "udev rule: ${YELLOW}Not installed${NC}"
    fi
    echo ""
    
    echo "AMD GPU BAR files:"
    local bars=($(find_amd_bars))
    
    if [ ${#bars[@]} -eq 0 ]; then
        echo "  (none found)"
    else
        for bar in "${bars[@]}"; do
            local perms=$(stat -c%a "$bar" 2>/dev/null || echo "???")
            local size=$(stat -c%s "$bar" 2>/dev/null || echo 0)
            local size_gb=$((size / 1024 / 1024 / 1024))
            
            if [ "$perms" = "666" ] || [ "$perms" = "777" ]; then
                echo -e "  $bar: ${GREEN}accessible${NC} (${size_gb}GB, mode $perms)"
            else
                echo -e "  $bar: ${RED}restricted${NC} (${size_gb}GB, mode $perms)"
            fi
        done
    fi
}

# Main
case "${1:-}" in
    --temp|-t)
        setup_temp
        ;;
    --permanent|-p)
        setup_permanent
        ;;
    --remove|-r)
        remove_permanent
        ;;
    --status|-s|"")
        show_status
        ;;
    --help|-h)
        echo "Usage: $0 [--temp|--permanent|--remove|--status]"
        echo ""
        echo "Options:"
        echo "  --temp, -t      One-time permission change (lost on reboot)"
        echo "  --permanent, -p Install udev rule (persistent across reboots)"
        echo "  --remove, -r    Remove udev rule"
        echo "  --status, -s    Show current status (default)"
        echo "  --help, -h      Show this help"
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
