#pragma once

#include <string>

#include "backends/DeviceId.h"
#include "tensors/CoherenceState.h"
#include "transfer/TransferMethod.h"

// Forward declarations
namespace llaminar2
{
    class IBackend;
    class TensorBase;
} // namespace llaminar2

namespace llaminar2
{

    /// TransferEngine — Unified, testable data movement between host and devices.
    ///
    /// This is a stateless utility class. All state lives in the TensorBase (via
    /// MemoryDescriptor snapshots) and in the backends. Thread-safe by design.
    ///
    /// Usage for coherence:
    ///   TransferEngine engine;
    ///   engine.upload(tensor, DeviceId::cuda(0));    // ensureOnDevice replacement
    ///   engine.download(tensor);                      // ensureOnHost replacement
    ///
    /// Usage for PP activation transfer:
    ///   engine.transferActivation(tensor, next_stage_device);
    ///
    /// Canonical decision tree (planTransfer):
    ///   src == dst                       → NOOP
    ///   residency == MAPPED              → MAPPED_NOOP
    ///   CPU → GPU                        → HOST_TO_DEVICE
    ///   GPU → CPU                        → DEVICE_TO_HOST
    ///   GPU → GPU same vendor            → DEVICE_TO_DEVICE_SAME_BACKEND
    ///   GPU → GPU cross-vendor           → HOST_STAGED
    ///
    class TransferEngine
    {
    public:
        TransferEngine() = default;

        /// Construct a TransferEngine with an injected backend resolver
        /// (for testing with MockBackend).
        using BackendResolver = IBackend *(*)(DeviceId);
        explicit TransferEngine(BackendResolver resolver) : resolve_(resolver) {}

        // ========================================================================
        // Plan: pure logic — determines HOW to transfer. No side effects.
        // ========================================================================

        /// Determine the transfer method for moving data between devices.
        /// This is a pure function: no state, no side effects, fully testable.
        static TransferMethod planTransfer(DeviceId src, DeviceId dst, MemoryResidency residency);

        /// Human-readable description of a transfer plan (for --explain-placement).
        static std::string describeTransferPlan(DeviceId src, DeviceId dst, MemoryResidency residency);

        // ========================================================================
        // Execute: actually moves data.
        // ========================================================================

        /// Execute a planned transfer using the given request.
        /// Dispatches to the correct IBackend based on TransferMethod.
        TransferResult execute(const TransferRequest &request);

        // ========================================================================
        // High-level API: operates on TensorBase directly.
        // ========================================================================

        /// Upload tensor data from host to target device.
        /// Updates TensorCoherenceState appropriately.
        TransferResult upload(TensorBase *tensor, DeviceId target_device);

        /// Download tensor data from device to host.
        /// Updates TensorCoherenceState appropriately.
        TransferResult download(TensorBase *tensor);

        /// Transfer activation data for pipeline parallelism.
        /// Handles all memory residency variants (standard, mapped).
        TransferResult transferActivation(TensorBase *tensor, DeviceId target_device);

        /// Full upload lifecycle: replaces TensorBase::ensureOnDevice() logic.
        /// Handles mapped, event wait, device migration, allocation, H2D.
        /// Caller must hold coherence_mutex_ and check graph-capture/CPU bailouts.
        TransferResult uploadFull(TensorBase *tensor, DeviceId target_device, void *stream = nullptr);

        /// Full download lifecycle: replaces TensorBase::ensureOnHost() logic.
        /// Handles mapped sync, event wait, standard D2H.
        /// Caller must hold coherence_mutex_.
        TransferResult downloadFull(TensorBase *tensor, void *stream = nullptr);

        // ========================================================================
        // Singleton access (using default backend resolver)
        // ========================================================================

        /// Get the default TransferEngine instance.
        static TransferEngine &instance();

    private:
        /// Resolve a backend for the given device.
        /// Uses injected resolver if set, otherwise the global BackendManager.
        IBackend *resolveBackend(DeviceId device) const;

        /// Execute HOST_TO_DEVICE transfer
        TransferResult executeHostToDevice(const TransferRequest &req);

        /// Execute DEVICE_TO_HOST transfer
        TransferResult executeDeviceToHost(const TransferRequest &req);

        /// Execute DEVICE_TO_DEVICE_SAME_BACKEND transfer
        TransferResult executeDeviceToDeviceSameBackend(const TransferRequest &req);

        /// Execute HOST_STAGED transfer (generic cross-vendor bounce)
        TransferResult executeHostStaged(const TransferRequest &req);

        /// Log transfer if tracing is enabled
        void traceTransfer(const TransferRequest &req, const TransferResult &result) const;

        /// Wait for a GPU completion event.
        static bool waitForEventWithProxy(IBackend *backend, void *event, int device_id,
                                          const DeviceId &gpu_device);

        BackendResolver resolve_ = nullptr;
    };

    /// Create a MemoryDescriptor from a TensorBase.
    /// This reads all pointer state from the tensor without modifying it.
    MemoryDescriptor makeMemoryDescriptor(const TensorBase *tensor);

} // namespace llaminar2
