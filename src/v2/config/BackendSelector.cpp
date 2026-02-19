/**
 * @file BackendSelector.cpp
 * @brief Implementation of automatic backend selection
 */

#include "BackendSelector.h"
#include "../backends/ComputeBackend.h"
#include "../utils/Logger.h"
#include <algorithm>

namespace llaminar2 {

// =========================================================================
// NUMA Lookup Helper
// =========================================================================

/**
 * @brief Look up the NUMA node for a DeviceId via DeviceManager inventory.
 * @return NUMA node (>=0), or -1 if device not found or DeviceManager not initialized.
 */
static int numaNodeForDevice(const DeviceId& dev) {
    const auto& dm = DeviceManager::instance();
    for (const auto& cd : dm.devices()) {
        bool type_match = false;
        switch (dev.type) {
            case DeviceType::CUDA: type_match = (cd.type == ComputeBackendType::GPU_CUDA); break;
            case DeviceType::ROCm: type_match = (cd.type == ComputeBackendType::GPU_ROCM); break;
            case DeviceType::CPU:  type_match = (cd.type == ComputeBackendType::CPU);      break;
        }
        if (type_match && cd.device_id == dev.ordinal) {
            return cd.numa_node;
        }
    }
    return -1;  // Unknown
}

/**
 * @brief Check if all GPU devices in the list are on the same NUMA node.
 * @return true if same NUMA (or only one GPU, or NUMA info unavailable)
 */
static bool areDevicesSameNuma(const std::vector<DeviceId>& devices) {
    int first_numa = -1;
    for (const auto& dev : devices) {
        if (!dev.is_gpu()) continue;
        int numa = numaNodeForDevice(dev);
        if (numa < 0) continue;  // Unknown NUMA — don't block
        if (first_numa < 0) {
            first_numa = numa;
        } else if (numa != first_numa) {
            return false;
        }
    }
    return true;
}

// =========================================================================
// PP Transfer Backend Selection
// =========================================================================

CollectiveBackendType BackendSelector::selectForTransfer(DeviceId src, DeviceId dst) {
    // For cross-vendor GPU pairs, check NUMA before delegating to type-only overload
    bool src_gpu = src.is_gpu();
    bool dst_gpu = dst.is_gpu();
    if (src_gpu && dst_gpu && src.type != dst.type) {
        int src_numa = numaNodeForDevice(src);
        int dst_numa = numaNodeForDevice(dst);
        if (src_numa >= 0 && dst_numa >= 0 && src_numa != dst_numa) {
            LOG_WARN("Cross-NUMA heterogeneous GPU PP transfer (" << src.to_string()
                     << " NUMA " << src_numa << " → " << dst.to_string()
                     << " NUMA " << dst_numa << "): HOST backend selected "
                     "instead of PCIeBAR — this will be slow! "
                     "For best performance, use GPUs on the same NUMA node "
                     "so PCIeBAR peer-to-peer transfers can be used.");
            return CollectiveBackendType::HOST;
        }
    }
    return selectForTransfer(src.type, dst.type);
}

CollectiveBackendType BackendSelector::selectForTransfer(DeviceType src_type, DeviceType dst_type) {
    // Same device type
    if (src_type == dst_type) {
        switch (src_type) {
            case DeviceType::CUDA:
                return CollectiveBackendType::NCCL;
            case DeviceType::ROCm:
                return CollectiveBackendType::RCCL;
            case DeviceType::CPU:
            default:
                return CollectiveBackendType::HOST;
        }
    }

    // Cross-vendor GPU transfer
    bool src_gpu = (src_type == DeviceType::CUDA || src_type == DeviceType::ROCm);
    bool dst_gpu = (dst_type == DeviceType::CUDA || dst_type == DeviceType::ROCm);

    if (src_gpu && dst_gpu) {
        // CUDA ↔ ROCm: PCIeBAR (caller with DeviceId checks NUMA before reaching here)
        return CollectiveBackendType::PCIE_BAR;
    }

    // GPU ↔ CPU: use HOST staging
    return CollectiveBackendType::HOST;
}

// =========================================================================
// TP Domain Backend Selection
// =========================================================================

CollectiveBackendType BackendSelector::selectForTPDomain(const std::vector<DeviceId>& devices) {
    if (devices.empty()) {
        return CollectiveBackendType::HOST;
    }

    if (devices.size() == 1) {
        // Single device: no collective needed, but HOST as placeholder
        return CollectiveBackendType::HOST;
    }

    // Check device composition
    bool has_cuda = false;
    bool has_rocm = false;
    bool has_cpu = false;

    for (const auto& dev : devices) {
        switch (dev.type) {
            case DeviceType::CUDA:
                has_cuda = true;
                break;
            case DeviceType::ROCm:
                has_rocm = true;
                break;
            case DeviceType::CPU:
            default:
                has_cpu = true;
                break;
        }
    }

    // All CUDA → NCCL
    if (has_cuda && !has_rocm && !has_cpu) {
        return CollectiveBackendType::NCCL;
    }

    // All ROCm → RCCL
    if (has_rocm && !has_cuda && !has_cpu) {
        return CollectiveBackendType::RCCL;
    }

    // Mixed CUDA + ROCm (no CPU)
    if (has_cuda && has_rocm && !has_cpu) {
        if (devices.size() == 2 && areDevicesSameNuma(devices)) {
            // 2-device cross-vendor on same NUMA: PCIeBAR
            return CollectiveBackendType::PCIE_BAR;
        } else if (devices.size() == 2) {
            // 2-device cross-vendor but cross-NUMA: HOST staging
            LOG_WARN("Cross-NUMA heterogeneous GPU TP (" << devices[0].to_string()
                     << ", " << devices[1].to_string() << "): HOST backend selected "
                     "instead of PCIeBAR — this will be slow! "
                     "For best performance, use GPUs on the same NUMA node "
                     "so PCIeBAR peer-to-peer transfers can be used.");
            return CollectiveBackendType::HOST;
        } else {
            // 3+ device cross-vendor: orchestrated heterogeneous
            return CollectiveBackendType::HETEROGENEOUS;
        }
    }

    // All CPU → MPI
    if (has_cpu && !has_cuda && !has_rocm) {
        return CollectiveBackendType::MPI;
    }

    // Mixed GPU + CPU → HOST (stage through CPU)
    return CollectiveBackendType::HOST;
}

// =========================================================================
// Backend Availability
// =========================================================================

bool BackendSelector::isBackendCompiled(CollectiveBackendType backend) {
    switch (backend) {
        case CollectiveBackendType::NCCL:
#ifdef HAVE_CUDA
            return true;
#else
            return false;
#endif

        case CollectiveBackendType::RCCL:
#ifdef HAVE_ROCM
            return true;
#else
            return false;
#endif

        case CollectiveBackendType::PCIE_BAR:
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            return true;
#elif defined(HAVE_CUDA) || defined(HAVE_ROCM)
            // PCIeBAR also works for same-vendor P2P in some cases
            return true;
#else
            return false;
#endif

        case CollectiveBackendType::HETEROGENEOUS:
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            return true;
#else
            return false;
#endif

        case CollectiveBackendType::UPI:
        case CollectiveBackendType::MPI:
        case CollectiveBackendType::HOST:
        case CollectiveBackendType::AUTO:
            return true;  // Always available
    }
    return false;
}

bool BackendSelector::isBackendUsable(CollectiveBackendType backend,
                                       const std::vector<DeviceId>& devices) {
    if (!isBackendCompiled(backend)) {
        return false;
    }

    if (devices.empty()) {
        return backend == CollectiveBackendType::HOST;
    }

    switch (backend) {
        case CollectiveBackendType::NCCL:
            // NCCL requires all devices to be CUDA
            return std::all_of(devices.begin(), devices.end(),
                [](const DeviceId& d) { return d.type == DeviceType::CUDA; });

        case CollectiveBackendType::RCCL:
            // RCCL requires all devices to be ROCm
            return std::all_of(devices.begin(), devices.end(),
                [](const DeviceId& d) { return d.type == DeviceType::ROCm; });

        case CollectiveBackendType::PCIE_BAR:
            // PCIeBAR requires exactly 2 GPU devices on the same NUMA node
            if (devices.size() != 2) return false;
            if (!std::all_of(devices.begin(), devices.end(),
                [](const DeviceId& d) { return d.is_gpu(); })) return false;
            return areDevicesSameNuma(devices);

        case CollectiveBackendType::HETEROGENEOUS:
            // Heterogeneous requires mixed GPU vendors
            return isCrossVendor(devices) && devices.size() >= 2;

        case CollectiveBackendType::UPI:
            // UPI requires all CPU devices
            return std::all_of(devices.begin(), devices.end(),
                [](const DeviceId& d) { return d.type == DeviceType::CPU; });

        case CollectiveBackendType::MPI:
        case CollectiveBackendType::HOST:
            return true;  // Always usable

        case CollectiveBackendType::AUTO:
            return false;  // AUTO is not a concrete backend
    }
    return false;
}

// =========================================================================
// Utility
// =========================================================================

CollectiveBackendType BackendSelector::resolve(CollectiveBackendType backend,
                                                const std::vector<DeviceId>& devices) {
    if (backend != CollectiveBackendType::AUTO) {
        return backend;
    }
    return selectForTPDomain(devices);
}

bool BackendSelector::isHomogeneous(const std::vector<DeviceId>& devices) {
    if (devices.size() <= 1) {
        return true;
    }
    DeviceType first_type = devices[0].type;
    return std::all_of(devices.begin(), devices.end(),
        [first_type](const DeviceId& d) { return d.type == first_type; });
}

bool BackendSelector::hasGPU(const std::vector<DeviceId>& devices) {
    return std::any_of(devices.begin(), devices.end(),
        [](const DeviceId& d) { return d.is_gpu(); });
}

bool BackendSelector::isCrossVendor(const std::vector<DeviceId>& devices) {
    bool has_cuda = std::any_of(devices.begin(), devices.end(),
        [](const DeviceId& d) { return d.type == DeviceType::CUDA; });
    bool has_rocm = std::any_of(devices.begin(), devices.end(),
        [](const DeviceId& d) { return d.type == DeviceType::ROCm; });
    return has_cuda && has_rocm;
}

} // namespace llaminar2
