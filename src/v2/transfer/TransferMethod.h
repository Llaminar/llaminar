#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "backends/DeviceId.h"
#include "tensors/CoherenceState.h"

namespace llaminar2
{

    // ============================================================================
    // TransferMethod — HOW data moves between devices
    // ============================================================================
    //
    // Determined by TransferEngine::planTransfer() based on source device,
    // destination device, and memory residency type. Each method maps to a
    // specific sequence of IBackend calls.
    //

    enum class TransferMethod : uint8_t
    {
        NOOP,                          // Same device — no transfer needed
        HOST_TO_DEVICE,                // Standard H2D via IBackend::hostToDevice()
        DEVICE_TO_HOST,                // Standard D2H via IBackend::deviceToHost()
        DEVICE_TO_DEVICE_SAME_BACKEND, // P2P within same vendor (CUDA↔CUDA or ROCm↔ROCm)
        HOST_STAGED,                   // Generic cross-vendor: D2H → memcpy → H2D
        MAPPED_NOOP,                   // Zero-copy mapped memory — no transfer needed
    };

    inline constexpr std::string_view to_string(TransferMethod method)
    {
        switch (method)
        {
        case TransferMethod::NOOP:
            return "NOOP";
        case TransferMethod::HOST_TO_DEVICE:
            return "HOST_TO_DEVICE";
        case TransferMethod::DEVICE_TO_HOST:
            return "DEVICE_TO_HOST";
        case TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND:
            return "DEVICE_TO_DEVICE_SAME_BACKEND";
        case TransferMethod::HOST_STAGED:
            return "HOST_STAGED";
        case TransferMethod::MAPPED_NOOP:
            return "MAPPED_NOOP";
        }
        return "UNKNOWN";
    }

    // ============================================================================
    // MemoryDescriptor — snapshot of where a tensor's data physically lives
    // ============================================================================
    //
    // Created via MemoryDescriptor::fromTensor() to read all pointer state from
    // a TensorBase. This decouples transfer logic from TensorBase internals.
    //

    struct MemoryDescriptor
    {
        // Primary locations
        void *host_ptr = nullptr;   // Host data pointer
        void *device_ptr = nullptr; // Primary GPU buffer pointer
        DeviceId device;            // Current GPU device (or CPU)
        size_t size_bytes = 0;      // Data size in bytes
        MemoryResidency residency = MemoryResidency::STANDARD;

        // Mapped-specific (only valid if residency == MAPPED)
        void *mapped_host_ptr = nullptr;   // Host-visible pointer for mapped memory
        void *mapped_device_ptr = nullptr; // Device-visible pointer for mapped memory

        /// Describe the descriptor for debugging
        std::string describe() const
        {
            std::string s = "MemoryDescriptor{";
            s += "device=" + device.to_string();
            s += ", size=" + std::to_string(size_bytes);
            s += ", residency=" + std::string(to_string(residency));
            s += ", host=" + std::to_string(host_ptr != nullptr);
            s += ", device_ptr=" + std::to_string(device_ptr != nullptr);
            if (residency == MemoryResidency::MAPPED)
            {
                s += ", mapped_host=" + std::to_string(mapped_host_ptr != nullptr);
                s += ", mapped_device=" + std::to_string(mapped_device_ptr != nullptr);
            }
            s += "}";
            return s;
        }
    };

    // ============================================================================
    // TransferRequest — what to transfer and how
    // ============================================================================

    struct TransferRequest
    {
        MemoryDescriptor source;    // Source tensor memory state
        DeviceId target_device;     // Where to move data
        TransferMethod method;      // How to transfer (from planTransfer)
        void *target_ptr = nullptr; // Pre-allocated destination buffer (nullptr = allocate)
    };

    // ============================================================================
    // TransferResult — outcome of a transfer operation
    // ============================================================================

    struct TransferResult
    {
        bool success = false;
        TransferMethod method_used = TransferMethod::NOOP;
        uint64_t elapsed_ns = 0;
        std::string error; // Non-empty on failure

        static TransferResult ok(TransferMethod method, uint64_t ns = 0)
        {
            return {true, method, ns, {}};
        }

        static TransferResult fail(TransferMethod method, std::string msg)
        {
            return {false, method, 0, std::move(msg)};
        }
    };

} // namespace llaminar2
