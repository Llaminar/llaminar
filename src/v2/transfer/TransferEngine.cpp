#include "transfer/TransferEngine.h"

#include <chrono>
#include <cstring>
#include <sstream>

#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "tensors/TensorClasses.h"
#include "utils/DebugEnv.h"
#include "utils/KernelProfiler.h"
#include "utils/Logger.h"
#include "utils/StackTrace.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace llaminar2
{

    // ============================================================================
    // Singleton
    // ============================================================================

    TransferEngine &TransferEngine::instance()
    {
        static TransferEngine engine;
        return engine;
    }

    // ============================================================================
    // planTransfer — pure logic, no side effects
    // ============================================================================

    TransferMethod TransferEngine::planTransfer(DeviceId src, DeviceId dst, MemoryResidency residency)
    {
        // Host-resident tensors never move to device
        if (residency == MemoryResidency::HOST_RESIDENT)
            return TransferMethod::NOOP;

        // Mapped memory is always in-place
        if (residency == MemoryResidency::MAPPED)
            return TransferMethod::MAPPED_NOOP;

        // Same device — nothing to do
        if (src == dst)
            return TransferMethod::NOOP;

        // CPU ↔ GPU
        if (src.is_cpu() && dst.is_gpu())
            return TransferMethod::HOST_TO_DEVICE;

        if (src.is_gpu() && dst.is_cpu())
            return TransferMethod::DEVICE_TO_HOST;

        // GPU ↔ GPU
        if (src.is_gpu() && dst.is_gpu())
        {
            // Same vendor (CUDA↔CUDA or ROCm↔ROCm) — direct P2P
            if (src.type == dst.type)
                return TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND;

            // Cross-vendor (CUDA↔ROCm) — host-staged transfer
            return TransferMethod::HOST_STAGED;
        }

        // CPU → CPU is a no-op (same host)
        if (src.is_cpu() && dst.is_cpu())
            return TransferMethod::NOOP;

        LOG_WARN("[TransferEngine] Unhandled device combination: "
                 << src.toString() << " → " << dst.toString());
        return TransferMethod::NOOP;
    }

    std::string TransferEngine::describeTransferPlan(DeviceId src, DeviceId dst, MemoryResidency residency)
    {
        auto method = planTransfer(src, dst, residency);
        std::ostringstream ss;
        ss << src.toString() << " → " << dst.toString()
           << " [" << to_string(residency) << "] → " << to_string(method);
        return ss.str();
    }

    // ============================================================================
    // execute — dispatch on TransferMethod
    // ============================================================================

    TransferResult TransferEngine::execute(const TransferRequest &request)
    {
        auto start = std::chrono::steady_clock::now();

        TransferResult result;
        switch (request.method)
        {
        case TransferMethod::NOOP:
        case TransferMethod::MAPPED_NOOP:
            result = TransferResult::ok(request.method);
            break;

        case TransferMethod::HOST_TO_DEVICE:
            result = executeHostToDevice(request);
            break;

        case TransferMethod::DEVICE_TO_HOST:
            result = executeDeviceToHost(request);
            break;

        case TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND:
            result = executeDeviceToDeviceSameBackend(request);
            break;

        case TransferMethod::HOST_STAGED:
            result = executeHostStaged(request);
            break;
        }

        auto end = std::chrono::steady_clock::now();
        result.elapsed_ns =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        result.method_used = request.method;

        traceTransfer(request, result);
        return result;
    }

    // ============================================================================
    // High-level TensorBase API
    // ============================================================================

    TransferResult TransferEngine::upload(TensorBase *tensor, DeviceId target_device)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");

        // Determine current residency and source device
        MemoryResidency residency = tensor->memoryResidency();
        DeviceId src = DeviceId::cpu();
        if (tensor->gpu_device_.has_value() && tensor->deviceValid())
            src = tensor->gpu_device_.value();

        // Plan the transfer
        TransferMethod method = planTransfer(src, target_device, residency);

        // If already on the target device with valid data, no-op
        if (method == TransferMethod::NOOP || method == TransferMethod::MAPPED_NOOP)
            return TransferResult::ok(method);

        // For H2D: we need host data to be valid
        if (method == TransferMethod::HOST_TO_DEVICE && !tensor->hostValid())
            return TransferResult::fail(method, "host data not valid for upload");

        // Build the request
        TransferRequest req;
        req.source = makeMemoryDescriptor(tensor);
        req.target_device = target_device;
        req.method = method;

        // Ensure we have a device buffer
        void *dst_ptr = tensor->getOrAllocateDeviceBuffer(target_device);
        if (!dst_ptr)
            return TransferResult::fail(method, "failed to allocate device buffer on " + target_device.toString());
        req.target_ptr = dst_ptr;

        auto result = execute(req);

        // Update tensor state on success
        if (result.success)
        {
            tensor->applyCoherenceOp_(CoherenceOp::UPLOAD); // Both host and device now valid

            // GPU_ONLY policy: free host data now that device has it
            if (tensor->memoryResidency() == MemoryResidency::GPU_ONLY &&
                !tensor->is_raw_data_released())
            {
                tensor->release_host_weight_data();
            }
        }

        return result;
    }

    TransferResult TransferEngine::download(TensorBase *tensor)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");

        // If host is already valid, no-op
        if (tensor->hostValid())
            return TransferResult::ok(TransferMethod::NOOP);

        // If device is not valid either, error
        if (!tensor->deviceValid() || !tensor->gpu_device_.has_value())
            return TransferResult::fail(TransferMethod::DEVICE_TO_HOST,
                                        "neither host nor device data is valid");

        MemoryResidency residency = tensor->memoryResidency();
        DeviceId src = tensor->gpu_device_.value();
        TransferMethod method = planTransfer(src, DeviceId::cpu(), residency);

        if (method == TransferMethod::MAPPED_NOOP)
            return TransferResult::ok(method);

        TransferRequest req;
        req.source = makeMemoryDescriptor(tensor);
        req.target_device = DeviceId::cpu();
        req.method = method;
        req.target_ptr = nullptr; // Host pointer is already in source descriptor

        auto result = execute(req);

        if (result.success)
        {
            tensor->applyCoherenceOp_(CoherenceOp::DOWNLOAD); // Both host and device now valid
        }

        return result;
    }

    TransferResult TransferEngine::transferActivation(TensorBase *tensor, DeviceId target_device)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");

        // Determine where the data currently lives
        DeviceId src = DeviceId::cpu();

        // If device has authoritative data, source is GPU
        if (tensor->gpu_device_.has_value() && tensor->deviceValid())
            src = tensor->gpu_device_.value();
        else if (!tensor->hostValid())
            return TransferResult::fail(TransferMethod::NOOP,
                                        "tensor has no valid data on any device");

        // Determine residency
        MemoryResidency residency = tensor->memoryResidency();

        TransferMethod method = planTransfer(src, target_device, residency);

        if (method == TransferMethod::NOOP || method == TransferMethod::MAPPED_NOOP)
            return TransferResult::ok(method);

        TransferRequest req;
        req.source = makeMemoryDescriptor(tensor);
        req.target_device = target_device;
        req.method = method;

        // For activation transfer, ensure destination buffer exists
        if (target_device.is_gpu())
        {
            void *dst_ptr = tensor->getOrAllocateDeviceBuffer(target_device);
            if (!dst_ptr)
                return TransferResult::fail(method,
                                            "failed to allocate device buffer for activation transfer to " +
                                                target_device.toString());
            req.target_ptr = dst_ptr;
        }

        auto result = execute(req);

        if (result.success)
        {
            if (target_device.is_gpu())
            {
                tensor->applyCoherenceOp_(CoherenceOp::MARK_DEVICE_DIRTY);
                tensor->authoritative_device_ = target_device;
            }
            else
            {
                tensor->applyCoherenceOp_(CoherenceOp::DOWNLOAD);
            }
        }

        return result;
    }

    // ============================================================================
    // Private: execute*() methods
    // ============================================================================

    TransferResult TransferEngine::executeHostToDevice(const TransferRequest &req)
    {
        if (!req.source.host_ptr)
            return TransferResult::fail(req.method, "source host_ptr is null");
        if (!req.target_ptr)
            return TransferResult::fail(req.method, "target device ptr is null");

        IBackend *backend = resolveBackend(req.target_device);
        if (!backend)
            return TransferResult::fail(req.method,
                                        "no backend for device " + req.target_device.toString());

        bool ok = backend->hostToDevice(req.target_ptr, req.source.host_ptr,
                                        req.source.size_bytes, req.target_device.ordinal);
        if (!ok)
            return TransferResult::fail(req.method, "hostToDevice failed");

        return TransferResult::ok(req.method);
    }

    TransferResult TransferEngine::executeDeviceToHost(const TransferRequest &req)
    {
        if (!req.source.device_ptr)
            return TransferResult::fail(req.method, "source device_ptr is null");
        if (!req.source.host_ptr)
            return TransferResult::fail(req.method, "destination host_ptr is null (no host buffer)");

        IBackend *backend = resolveBackend(req.source.device);
        if (!backend)
            return TransferResult::fail(req.method,
                                        "no backend for source device " + req.source.device.toString());

        bool ok = backend->deviceToHost(req.source.host_ptr, req.source.device_ptr,
                                        req.source.size_bytes, req.source.device.ordinal);
        if (!ok)
            return TransferResult::fail(req.method, "deviceToHost failed");

        return TransferResult::ok(req.method);
    }

    TransferResult TransferEngine::executeDeviceToDeviceSameBackend(const TransferRequest &req)
    {
        if (!req.source.device_ptr)
            return TransferResult::fail(req.method, "source device_ptr is null");
        if (!req.target_ptr)
            return TransferResult::fail(req.method, "target device ptr is null");

        // Same-vendor GPU P2P — use D2H + H2D through the backend.
        // IBackend doesn't have a direct D2D method, so we bounce through host.
        IBackend *src_backend = resolveBackend(req.source.device);
        IBackend *dst_backend = resolveBackend(req.target_device);

        if (!src_backend)
            return TransferResult::fail(req.method,
                                        "no backend for source " + req.source.device.toString());
        if (!dst_backend)
            return TransferResult::fail(req.method,
                                        "no backend for target " + req.target_device.toString());

        // Host bounce: D2H → memcpy is implicit if same host buffer
        if (!req.source.host_ptr)
            return TransferResult::fail(req.method, "host buffer needed for same-backend D2D bounce");

        // Step 1: D2H from source
        bool d2h = src_backend->deviceToHost(
            req.source.host_ptr, req.source.device_ptr,
            req.source.size_bytes, req.source.device.ordinal);
        if (!d2h)
            return TransferResult::fail(req.method, "D2H step of same-backend D2D failed");

        // Step 2: H2D to target
        bool h2d = dst_backend->hostToDevice(
            req.target_ptr, req.source.host_ptr,
            req.source.size_bytes, req.target_device.ordinal);
        if (!h2d)
            return TransferResult::fail(req.method, "H2D step of same-backend D2D failed");

        return TransferResult::ok(req.method);
    }

    TransferResult TransferEngine::executeHostStaged(const TransferRequest &req)
    {
        // HOST_STAGED: D2H from source GPU → memcpy → H2D to target GPU
        // Used for cross-vendor transfers.

        if (!req.source.device_ptr)
            return TransferResult::fail(req.method, "source device_ptr is null for host staged");
        if (!req.source.host_ptr)
            return TransferResult::fail(req.method, "host buffer needed for host staged bounce");
        if (!req.target_ptr)
            return TransferResult::fail(req.method, "target device ptr is null for host staged");

        IBackend *src_backend = resolveBackend(req.source.device);
        if (!src_backend)
            return TransferResult::fail(req.method,
                                        "no backend for source " + req.source.device.toString());

        // Step 1: D2H
        bool d2h = src_backend->deviceToHost(
            req.source.host_ptr, req.source.device_ptr,
            req.source.size_bytes, req.source.device.ordinal);
        if (!d2h)
            return TransferResult::fail(req.method, "D2H step of host staged failed");

        // Step 2: H2D to target
        IBackend *dst_backend = resolveBackend(req.target_device);
        if (!dst_backend)
            return TransferResult::fail(req.method,
                                        "no backend for target " + req.target_device.toString());

        bool h2d = dst_backend->hostToDevice(
            req.target_ptr, req.source.host_ptr,
            req.source.size_bytes, req.target_device.ordinal);
        if (!h2d)
            return TransferResult::fail(req.method, "H2D step of host staged failed");

        return TransferResult::ok(req.method);
    }

    // ============================================================================
    // Backend resolution
    // ============================================================================

    bool TransferEngine::waitForEventWithProxy(IBackend *backend, void *event, int device_id,
                                               const DeviceId &gpu_device)
    {
        return backend->waitForEvent(event, device_id);
    }

    // ============================================================================
    // uploadFull — full ensureOnDevice lifecycle (called with coherence_mutex_ held)
    // ============================================================================

    TransferResult TransferEngine::uploadFull(TensorBase *tensor, DeviceId target_device, void *stream)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");

        // ===== HOST-RESIDENT FAST PATH =====
        // Tensors marked HOST_RESIDENT are consumed on host only (e.g., embedding
        // tables that get repacked into a device workspace by the kernel).
        // Skip device allocation and upload entirely.
        if (tensor->memoryResidency() == MemoryResidency::HOST_RESIDENT)
            return TransferResult::ok(TransferMethod::NOOP);

        const bool trace = debugEnv().rocm.trace_coherence;
        auto overall_start = std::chrono::high_resolution_clock::now();

        // ===== ZERO-COPY MAPPED MEMORY FAST PATH =====
        if (tensor->is_mapped_ && tensor->mapped_device_ptr_ != nullptr)
        {
            if (!tensor->gpu_device_.has_value() || *tensor->gpu_device_ != target_device)
            {
                tensor->gpu_device_ = target_device;
            }
            if (tensor->gpu_data_ptr_ != tensor->mapped_device_ptr_)
            {
                tensor->gpu_data_ptr_ = tensor->mapped_device_ptr_;
            }
            tensor->setCoherenceState_(TensorCoherenceState::MAPPED);

            if (trace)
            {
                LOG_INFO("[TransferEngine::uploadFull] ZERO-COPY: Tensor is mapped, no memcpy needed");
            }
            return TransferResult::ok(TransferMethod::MAPPED_NOOP);
        }

        // ===== ALREADY ON TARGET DEVICE (with event wait) =====
        if (tensor->gpu_data_ptr_ && tensor->gpu_device_.has_value() &&
            *tensor->gpu_device_ == target_device && ::llaminar2::isDeviceValid(tensor->coherence_state_))
        {
            if (tensor->device_completion_event_)
            {
                IBackend *backend = tensor->resolveBackend(target_device);
                if (backend)
                {
                    const int backend_device_id = target_device.gpu_ordinal();
                    if (!waitForEventWithProxy(backend, tensor->device_completion_event_, backend_device_id, target_device))
                    {
                        LOG_WARN("[TransferEngine::uploadFull] Event wait failed for tensor "
                                 << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                                 << " on " << target_device.toString()
                                 << ", falling back to backend synchronize");
                        if (!backend->synchronize(backend_device_id))
                        {
                            return TransferResult::fail(TransferMethod::NOOP,
                                                        "backend synchronize failed for " + target_device.toString());
                        }
                    }
                }
            }
            return TransferResult::ok(TransferMethod::NOOP);
        }

        // Get backend for target device
        IBackend *target_backend = tensor->resolveBackend(target_device);
        if (!target_backend)
        {
            return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                        "No backend available for device " + target_device.toString());
        }

        int backend_device_id = target_device.gpu_ordinal();

        // ===== DEVICE MIGRATION (secondary buffers) =====
        LOG_DEBUG("[TransferEngine::uploadFull] tensor=" << static_cast<void *>(tensor)
                                                         << " target_device=" << target_device.toString()
                                                         << " gpu_data_ptr_=" << tensor->gpu_data_ptr_
                                                         << " gpu_device_=" << (tensor->gpu_device_.has_value() ? tensor->gpu_device_->toString() : "none")
                                                         << " device_completion_event_=" << tensor->device_completion_event_);
        if (tensor->gpu_data_ptr_ && tensor->gpu_device_.has_value() && *tensor->gpu_device_ != target_device)
        {
            DeviceId old_device = *tensor->gpu_device_;
            LOG_DEBUG("[TransferEngine::uploadFull] Device migration: " << tensor->gpu_device_->toString()
                                                                        << " -> " << target_device.toString());

            const int target_key = TensorBase::packDeviceId(target_device);
            auto sec_it = tensor->secondary_device_buffers_.find(target_key);
            if (sec_it != tensor->secondary_device_buffers_.end() && sec_it->second != nullptr)
            {
                // Preserve current primary in secondary map
                int old_key = TensorBase::packDeviceId(*tensor->gpu_device_);
                if (tensor->secondary_device_buffers_.find(old_key) == tensor->secondary_device_buffers_.end())
                {
                    tensor->secondary_device_buffers_[old_key] = tensor->gpu_data_ptr_;
                }

                // Promote target secondary buffer to primary
                void *promoted_ptr = sec_it->second;
                tensor->secondary_device_buffers_.erase(sec_it);

                tensor->gpu_data_ptr_ = promoted_ptr;
                tensor->gpu_device_ = target_device;
                tensor->setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE);

                if (tensor->device_completion_event_)
                {
                    IBackend *old_backend = tensor->resolveBackend(old_device);
                    int old_backend_device_id = old_device.gpu_ordinal();
                    if (old_backend)
                    {
                        old_backend->destroyEvent(tensor->device_completion_event_, old_backend_device_id);
                    }
                    tensor->device_completion_event_ = nullptr;
                    tensor->event_device_.reset();
                }

                LOG_DEBUG("[TransferEngine::uploadFull] Promoted secondary buffer to primary for "
                          << target_device.toString() << " ptr=" << promoted_ptr);
            }
            else
            {
                // No existing target secondary buffer. Park current primary.
                int old_key = TensorBase::packDeviceId(*tensor->gpu_device_);
                if (tensor->secondary_device_buffers_.find(old_key) == tensor->secondary_device_buffers_.end())
                {
                    tensor->secondary_device_buffers_[old_key] = tensor->gpu_data_ptr_;
                }

                if (tensor->device_completion_event_)
                {
                    IBackend *old_backend = tensor->resolveBackend(*tensor->gpu_device_);
                    int old_backend_device_id = tensor->gpu_device_->gpu_ordinal();
                    if (old_backend)
                    {
                        LOG_DEBUG("[TransferEngine::uploadFull] Destroying old completion event on device "
                                  << tensor->gpu_device_->toString() << " before migrating to " << target_device.toString());
                        old_backend->destroyEvent(tensor->device_completion_event_, old_backend_device_id);
                    }
                    tensor->device_completion_event_ = nullptr;
                    tensor->event_device_.reset();
                }

                tensor->gpu_data_ptr_ = nullptr;
                tensor->gpu_device_.reset();
                tensor->applyCoherenceOp_(CoherenceOp::RELEASE_DEVICE);

                LOG_DEBUG("[TransferEngine::uploadFull] Parked previous primary buffer for "
                          << old_device.toString() << " and allocating fresh buffer for "
                          << target_device.toString());
            }
        }

        // ===== KERNEL-MANAGED DATA SKIP =====
        // If the tensor has GEMM weights registered in KernelFactory's prepared
        // GEMM registry, the GPU pipeline has already uploaded the data into pooled
        // VRAM.  The kernel owns the device copy — skip raw upload.
        if (!tensor->gpu_data_ptr_ && !::llaminar2::isDeviceValid(tensor->coherence_state_) &&
            target_device.is_gpu() && tensor->hasPreparedDeviceState())
        {
            LOG_DEBUG("[TransferEngine::uploadFull] Skipping raw upload for tensor "
                      << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                      << " — kernel manages its own device representation (prepared GEMM registry)");
            return TransferResult::ok(TransferMethod::NOOP);
        }

        // ===== ALLOCATE ON TARGET DEVICE =====
        size_t bytes = tensor->byte_size();
        if (!tensor->gpu_data_ptr_)
        {
            auto alloc_start = std::chrono::high_resolution_clock::now();
            tensor->gpu_data_ptr_ = target_backend->allocate(bytes, backend_device_id);
            auto alloc_end = std::chrono::high_resolution_clock::now();
            auto alloc_us = std::chrono::duration_cast<std::chrono::microseconds>(alloc_end - alloc_start).count();

            if (trace)
            {
                LOG_INFO("[TransferEngine::uploadFull] backend->allocate(" << bytes << " bytes) took " << alloc_us << " us");
            }

            LOG_DEBUG("[GPU_ALLOC] tensor=" << static_cast<void *>(tensor)
                                            << " name=" << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                                            << " gpu_ptr=" << tensor->gpu_data_ptr_ << " bytes=" << bytes
                                            << " device=" << target_device.toString()
                                            << " ordinal=" << backend_device_id);

            if (!tensor->gpu_data_ptr_)
            {
                return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                            "Failed to allocate " + std::to_string(bytes) + " bytes on " + target_device.toString());
            }
            tensor->gpu_device_ = target_device;
            tensor->setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE);

            // Pin host memory for fast DMA transfers
            auto pin_start = std::chrono::high_resolution_clock::now();
            tensor->ensureHostPinned();
            auto pin_end = std::chrono::high_resolution_clock::now();
            auto pin_us = std::chrono::duration_cast<std::chrono::microseconds>(pin_end - pin_start).count();

            if (trace && pin_us > 100)
            {
                LOG_INFO("[TransferEngine::uploadFull] ensureHostPinned() took " << pin_us << " us");
            }
        }

        // ===== H2D UPLOAD =====
        if (!::llaminar2::isDeviceValid(tensor->coherence_state_))
        {
            if (!::llaminar2::isHostValid(tensor->coherence_state_))
            {
                return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                            "COHERENCE ERROR: Both host and device are invalid");
            }

            const void *src = tensor->raw_host_data_ptr();
            if (!src)
            {
                target_backend->free(tensor->gpu_data_ptr_, backend_device_id);
                tensor->gpu_data_ptr_ = nullptr;
                tensor->gpu_device_.reset();
                return TransferResult::fail(TransferMethod::HOST_TO_DEVICE, "Host data pointer is null");
            }

            // Transfer tracing
            const auto &trace_cfg = debugEnv().transfer_tracing;
            if (trace_cfg.enabled && !trace_cfg.only_d2h && bytes >= trace_cfg.min_bytes)
            {
                std::ostringstream msg;
                msg << "[TRANSFER TRACE] H2D transfer: " << bytes << " bytes"
                    << ", tensor=" << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                    << ", shape=[" << tensor->rows() << "x" << tensor->cols() << "]"
                    << ", device=" << target_device.toString();

                if (trace_cfg.include_stacktrace)
                {
                    msg << "\n"
                        << captureStackTrace(2, 16);
                }

                if (trace_cfg.throw_on_transfer)
                {
                    throw TransferViolationException(msg.str(), captureStackTrace(2, 32));
                }
                else
                {
                    LOG_WARN(msg.str());
                }
            }

            auto h2d_start = std::chrono::high_resolution_clock::now();
            bool h2d_ok = target_backend->hostToDevice(tensor->gpu_data_ptr_, src, bytes, backend_device_id, stream);
            auto h2d_end = std::chrono::high_resolution_clock::now();
            auto h2d_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(h2d_end - h2d_start).count();
            auto h2d_us = h2d_ns / 1000;
            double bandwidth_gbps = (bytes / 1e9) / (h2d_us / 1e6);

            TransferProfiler::recordH2D(bytes, static_cast<uint64_t>(h2d_ns));

            if (trace_cfg.enabled)
            {
                trace_cfg.recordH2D(bytes);
            }

            if (trace)
            {
                LOG_INFO("[TransferEngine::uploadFull] hostToDevice(" << bytes << " bytes) took "
                                                                      << h2d_us << " us (" << bandwidth_gbps << " GB/s)");
            }

            if (!h2d_ok)
            {
                target_backend->free(tensor->gpu_data_ptr_, backend_device_id);
                tensor->gpu_data_ptr_ = nullptr;
                tensor->gpu_device_.reset();
                return TransferResult::fail(TransferMethod::HOST_TO_DEVICE, "hostToDevice failed");
            }

            tensor->applyCoherenceOp_(CoherenceOp::UPLOAD);

            // GPU_ONLY policy: free host data now that device has it
            if (tensor->memoryResidency() == MemoryResidency::GPU_ONLY &&
                !tensor->is_raw_data_released())
            {
                tensor->release_host_weight_data();
            }

            LOG_DEBUG("[TransferEngine::uploadFull] Uploaded " << bytes
                                                               << " bytes to device " << target_device.toString()
                                                               << " (backend device ID: " << backend_device_id << ")");
        }

        auto overall_end = std::chrono::high_resolution_clock::now();
        auto overall_us = std::chrono::duration_cast<std::chrono::microseconds>(overall_end - overall_start).count();
        if (trace && overall_us > 1000)
        {
            LOG_INFO("[TransferEngine::uploadFull] TOTAL took " << overall_us << " us for " << bytes << " bytes");
        }

        return TransferResult::ok(TransferMethod::HOST_TO_DEVICE);
    }

    // ============================================================================
    // downloadFull — full ensureOnHost lifecycle (called with coherence_mutex_ held)
    // ============================================================================

    TransferResult TransferEngine::downloadFull(TensorBase *tensor, void *stream)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");

        // ===== ZERO-COPY MAPPED MEMORY PATH =====
        if (tensor->is_mapped_)
        {
            if (tensor->mapped_needs_sync_ && tensor->gpu_device_.has_value())
            {
                IBackend *backend = tensor->resolveBackend(*tensor->gpu_device_);
                if (backend)
                {
                    int backend_device_id = tensor->gpu_device_->gpu_ordinal();

                    auto t0 = std::chrono::high_resolution_clock::now();

                    if (tensor->device_completion_event_)
                    {
                        LOG_TRACE("[TransferEngine::downloadFull] ZERO-COPY: Waiting on completion event");
                        if (!waitForEventWithProxy(backend, tensor->device_completion_event_, backend_device_id, *tensor->gpu_device_))
                        {
                            LOG_WARN("[TransferEngine::downloadFull] Event wait failed for mapped tensor");
                        }
                    }
                    else
                    {
                        LOG_WARN("[TransferEngine::downloadFull] ZERO-COPY: No completion event, using stream sync");
                        backend->synchronize(backend_device_id);
                    }

                    auto t1 = std::chrono::high_resolution_clock::now();
                    auto elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (elapsed_ms > 1.0)
                    {
                        LOG_WARN("[TransferEngine::downloadFull] MAPPED SYNC took " << elapsed_ms << " ms"
                                                                                    << " (event=" << (tensor->device_completion_event_ ? "yes" : "NO") << ")");
                    }
                }
                tensor->mapped_needs_sync_ = false;
            }

            tensor->setCoherenceState_(TensorCoherenceState::MAPPED);
            tensor->authoritative_device_ = std::nullopt;
            return TransferResult::ok(TransferMethod::MAPPED_NOOP);
        }

        // ===== HOST ALREADY VALID =====
        if (::llaminar2::isHostValid(tensor->coherence_state_))
        {
            LOG_TRACE("[TransferEngine::downloadFull] Host already valid, skipping sync");
            return TransferResult::ok(TransferMethod::NOOP);
        }

        // Device must be valid if host is invalid
        if (!::llaminar2::isDeviceValid(tensor->coherence_state_))
        {
            return TransferResult::fail(TransferMethod::DEVICE_TO_HOST,
                                        "COHERENCE ERROR: Both host and device are invalid");
        }

        // ===== STANDARD GPU D2H =====
        if (tensor->gpu_data_ptr_ && tensor->gpu_device_.has_value())
        {
            IBackend *backend = tensor->resolveBackend(*tensor->gpu_device_);
            if (!backend)
            {
                return TransferResult::fail(TransferMethod::DEVICE_TO_HOST,
                                            "No backend available for device " + tensor->gpu_device_->toString());
            }

            int backend_device_id = tensor->gpu_device_->gpu_ordinal();

            size_t bytes = tensor->byte_size();
            void *dst = tensor->raw_host_data_ptr();
            if (!dst)
            {
                return TransferResult::fail(TransferMethod::DEVICE_TO_HOST, "Host data pointer is null");
            }

            // Fine-grained sync: wait for this tensor's completion event
            if (tensor->device_completion_event_)
            {
                LOG_TRACE("[TransferEngine::downloadFull] Using event-based sync (waiting for specific kernel)");
                if (!waitForEventWithProxy(backend, tensor->device_completion_event_, backend_device_id, *tensor->gpu_device_))
                {
                    LOG_WARN("[TransferEngine::downloadFull] Event wait failed, falling back to full sync");
                    backend->synchronize(backend_device_id);
                }
            }
            else
            {
                LOG_TRACE("[TransferEngine::downloadFull] No completion event, using full device sync");
                backend->synchronize(backend_device_id);
            }

            // Transfer tracing for D2H debugging
            const auto &trace_cfg = debugEnv().transfer_tracing;
            if (trace_cfg.enabled && bytes >= trace_cfg.min_bytes)
            {
                std::ostringstream msg;
                msg << "[TRANSFER TRACE] D2H transfer: " << bytes << " bytes"
                    << ", tensor=" << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                    << ", shape=[" << tensor->rows() << "x" << tensor->cols() << "]"
                    << ", device=" << tensor->gpu_device_->toString();

                if (trace_cfg.include_stacktrace)
                {
                    msg << "\n"
                        << captureStackTrace(2, 16);
                }

                if (trace_cfg.throw_on_transfer)
                {
                    throw TransferViolationException(msg.str(), captureStackTrace(2, 32));
                }
                else
                {
                    LOG_WARN(msg.str());
                }
            }

            LOG_DEBUG("[TransferEngine::downloadFull] ATTEMPTING D2H: "
                      << "tensor=" << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                      << " gpu_data_ptr=" << static_cast<const void *>(tensor->gpu_data_ptr_)
                      << " dst=" << static_cast<const void *>(dst)
                      << " bytes=" << bytes
                      << " device=" << tensor->gpu_device_->toString()
                      << " backend_device_id=" << backend_device_id);

            auto d2h_start = std::chrono::high_resolution_clock::now();
            bool d2h_ok = backend->deviceToHost(dst, tensor->gpu_data_ptr_, bytes, backend_device_id, stream);
            auto d2h_end = std::chrono::high_resolution_clock::now();
            auto d2h_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(d2h_end - d2h_start).count();

            if (!d2h_ok)
            {
                return TransferResult::fail(TransferMethod::DEVICE_TO_HOST, "deviceToHost failed");
            }

            // Optional transfer trace diagnostic. This only samples the first
            // few floats, so keep it out of normal logs and validation output.
            if (trace_cfg.enabled)
            {
                const float *fp = static_cast<const float *>(dst);
                size_t check_count = std::min(bytes / sizeof(float), static_cast<size_t>(8));
                bool all_zero = true;
                for (size_t i = 0; i < check_count; ++i)
                {
                    if (fp[i] != 0.0f)
                    {
                        all_zero = false;
                        break;
                    }
                }
                if (all_zero && check_count > 0 && bytes >= 1024)
                {
                    LOG_DEBUG("[TransferEngine::downloadFull] D2H leading sample is zero: tensor="
                              << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                              << " bytes=" << bytes
                              << " device=" << tensor->gpu_device_->toString()
                              << " gpu_ptr=" << static_cast<const void *>(tensor->gpu_data_ptr_)
                              << " dst=" << static_cast<const void *>(dst)
                              << " d2h_ns=" << d2h_ns
                              << " had_event=" << (tensor->device_completion_event_ ? "yes" : "no"));
                }
            }

            TransferProfiler::recordD2H(bytes, static_cast<uint64_t>(d2h_ns));

            if (trace_cfg.enabled)
            {
                trace_cfg.recordD2H(bytes);
            }

            tensor->applyCoherenceOp_(CoherenceOp::DOWNLOAD);
            tensor->authoritative_device_ = std::nullopt;

            LOG_DEBUG("[TransferEngine::downloadFull] Downloaded " << bytes
                                                                   << " bytes from device " << tensor->gpu_device_->toString()
                                                                   << " (backend device ID: " << backend_device_id << ")");
        }

        return TransferResult::ok(TransferMethod::DEVICE_TO_HOST);
    }

    IBackend *TransferEngine::resolveBackend(DeviceId device) const
    {
        if (resolve_)
            return resolve_(device);
        return getBackendFor(device);
    }

    // ============================================================================
    // Transfer tracing
    // ============================================================================

    void TransferEngine::traceTransfer(const TransferRequest &req, const TransferResult &result) const
    {
        const auto &tracing = debugEnv().transfer_tracing;
        if (!tracing.enabled)
            return;

        auto method_str = to_string(req.method);
        auto src_str = req.source.device.toString();
        auto dst_str = req.target_device.toString();

        if (result.success)
        {
            LOG_DEBUG("[TransferEngine] " << method_str
                                          << " " << src_str << " → " << dst_str
                                          << " (" << req.source.size_bytes << " bytes"
                                          << ", " << result.elapsed_ns / 1000 << " μs)");
        }
        else
        {
            LOG_WARN("[TransferEngine] FAILED " << method_str
                                                << " " << src_str << " → " << dst_str
                                                << ": " << result.error);
        }
    }

    // ============================================================================
    // MemoryDescriptor factory
    // ============================================================================

    MemoryDescriptor makeMemoryDescriptor(const TensorBase *tensor)
    {
        MemoryDescriptor desc;

        if (!tensor)
            return desc;

        // Host pointer
        desc.host_ptr = const_cast<void *>(tensor->raw_data());
        desc.size_bytes = tensor->byte_size();

        // GPU pointer and device
        if (tensor->gpu_device_.has_value())
        {
            desc.device = tensor->gpu_device_.value();
            desc.device_ptr = tensor->gpu_data_ptr_;
        }
        else
        {
            desc.device = DeviceId::cpu();
        }

        // Memory residency from canonical source
        desc.residency = tensor->memoryResidency();
        if (desc.residency == MemoryResidency::MAPPED)
        {
            desc.mapped_host_ptr = tensor->mapped_host_ptr_;
            desc.mapped_device_ptr = tensor->mapped_device_ptr_;
        }

        return desc;
    }

} // namespace llaminar2
