#include "transfer/TransferEngine.h"

#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "collective/BackendRouter.h"
#include "collective/ICollectiveBackend.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
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
    namespace
    {
        /**
         * @brief Resolve the coherence implementation hidden behind ITensor.
         *
         * Production tensors are TensorBase implementations. Keeping this
         * checked conversion inside TransferEngine preserves ITensor as the
         * orchestration-facing abstraction and turns an unsupported tensor
         * implementation into an immediate contract failure.
         */
        TensorBase *requireTensorBase(ITensor *tensor, const char *operation)
        {
            if (!tensor)
            {
                throw std::invalid_argument(
                    std::string(operation) + " requires a tensor");
            }

            auto *base = dynamic_cast<TensorBase *>(tensor);
            if (!base)
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " requires a coherence-aware TensorBase implementation");
            }
            return base;
        }

        /**
         * @brief Resolve the one tensor that owns physical transfer state.
         *
         * Tensor wrappers deliberately delegate pointer and coherence queries
         * to an inner tensor.  TransferEngine must therefore lock and mutate
         * that same inner object, never the wrapper's dormant TensorBase
         * fields.  Requiring a canonical owner at this boundary gives every
         * transfer API the same structural rule.
         */
        TensorBase *requireTransferStorageOwner(
            ITensor *tensor,
            const char *operation)
        {
            TensorBase *base = requireTensorBase(tensor, operation);
            TensorBase *owner = base->transferStorageOwner();
            if (!owner)
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " resolved a null transfer-storage owner");
            }
            if (owner->transferStorageOwner() != owner)
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " resolved a non-canonical transfer-storage owner");
            }
            return owner;
        }

        /**
         * @brief Resolve the persistent stream owned by a GPU transfer context.
         *
         * Backend APIs deliberately reject null streams. Tensor-aware transfer
         * operations choose their stream here, before touching a backend, so
         * setup, staging, and publication all name one stable owner.
         */
        void *requireTransferStream(
            DeviceId device,
            const char *operation)
        {
            if (!device.is_gpu())
            {
                throw std::invalid_argument(
                    std::string(operation) +
                    " requires a GPU transfer endpoint");
            }

            void *const stream =
                GPUDeviceContextPool::instance()
                    .getContext(device)
                    .defaultStream();
            if (!stream)
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " could not resolve a non-null transfer stream for " +
                    device.toString());
            }
            return stream;
        }

        /**
         * @brief Describe the tensor ownership contract at a failed transfer boundary.
         *
         * A stage can prepare both arena activations and model weights through
         * the same TransferEngine API.  Include the state that controls
         * `uploadFull()` so a hard residency failure identifies the offending
         * tensor and distinguishes an invalid arena publication from a
         * host-resident or prepared-weight classification error.
         */
        std::string tensorTransferState(const TensorBase &tensor)
        {
            std::ostringstream description;
            description
                << " tensor='"
                << (tensor.debugName().empty()
                        ? std::string("(unnamed)")
                        : tensor.debugName())
                << "' coherence=" << to_string(tensor.coherenceState())
                << " residency=" << to_string(tensor.memoryResidency())
                << " prepared_device_state="
                << (tensor.hasPreparedDeviceState() ? "true" : "false")
                << " current_device="
                << (tensor.current_device().has_value()
                        ? tensor.current_device()->toString()
                        : std::string("none"))
                << " gpu_ptr=" << tensor.gpu_data_ptr();
            return description.str();
        }
    } // namespace

    // ============================================================================
    // Singleton
    // ============================================================================

    TransferEngine &TransferEngine::instance()
    {
        static TransferEngine engine;
        return engine;
    }

    void TransferEngine::prepareDeviceInput(
        ITensor *tensor,
        DeviceId target_device,
        void *stream)
    {
        if (!target_device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::prepareDeviceInput requires a GPU target");
        }
        if (!stream)
        {
            throw std::invalid_argument(
                "TransferEngine::prepareDeviceInput requires the exact "
                "non-null consumer stream");
        }

        TensorBase *base = requireTransferStorageOwner(
            tensor,
            "TransferEngine::prepareDeviceInput");
        if (!base->ensureOnDevice(target_device, stream))
        {
            throw std::runtime_error(
                "TransferEngine::prepareDeviceInput failed to place tensor on " +
                target_device.toString());
        }
        if (!base->gpu_data_ptr() ||
            !base->is_on_device(target_device))
        {
            throw std::runtime_error(
                "TransferEngine::prepareDeviceInput completed without storage on " +
                target_device.toString() +
                tensorTransferState(*base));
        }
    }

    void TransferEngine::requireDeviceInput(
        ITensor *tensor,
        DeviceId target_device,
        void *consumer_stream)
    {
        if (!target_device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::requireDeviceInput requires a GPU target");
        }
        if (!consumer_stream)
        {
            throw std::invalid_argument(
                "TransferEngine::requireDeviceInput requires the exact "
                "non-null consumer stream");
        }

        TensorBase *base = requireTransferStorageOwner(
            tensor,
            "TransferEngine::requireDeviceInput");
        std::lock_guard<std::mutex> lock(base->coherence_mutex_);

        if (!base->gpu_data_ptr_ ||
            !base->gpu_device_.has_value() ||
            *base->gpu_device_ != target_device ||
            !::llaminar2::isDeviceValid(base->coherence_state_))
        {
            throw std::runtime_error(
                "TransferEngine::requireDeviceInput requires valid pre-existing "
                "storage on " +
                target_device.toString() + tensorTransferState(*base));
        }

        if (!base->device_completion_event_)
            return;

        /*
         * A capture body may consume only dependencies established before
         * beginCapture(). Importing an event recorded by uncaptured work here is
         * rejected by CUDA/HIP and, more importantly, hides an incomplete graph
         * boundary. The exact event/stream identity is established by the
         * executor's pre-capture input pass.
         */
        if (isGraphCaptureActive())
        {
            if (base->last_joined_completion_event_ ==
                    base->device_completion_event_ &&
                base->last_joined_consumer_stream_ == consumer_stream)
            {
                return;
            }

            throw std::runtime_error(
                "TransferEngine::requireDeviceInput found an external producer "
                "event that was not joined to the exact consumer stream before "
                "GPU graph capture began for " +
                target_device.toString() + tensorTransferState(*base));
        }

        if (!base->event_device_.has_value() ||
            *base->event_device_ != target_device)
        {
            throw std::runtime_error(
                "TransferEngine::requireDeviceInput found completion-event "
                "ownership that does not match " +
                target_device.toString() + tensorTransferState(*base));
        }

        IBackend *backend = base->resolveBackend(target_device);
        if (!backend)
        {
            throw std::runtime_error(
                "TransferEngine::requireDeviceInput could not resolve backend "
                "for " +
                target_device.toString() + tensorTransferState(*base));
        }

        const int backend_device_id = target_device.gpu_ordinal();
        if (!backend->streamWaitEvent(
                consumer_stream,
                base->device_completion_event_,
                backend_device_id))
        {
            throw std::runtime_error(
                "TransferEngine::requireDeviceInput failed to join producer "
                "event on consumer stream for " +
                target_device.toString() + tensorTransferState(*base));
        }

        /*
         * Record only after the backend accepts the wait. This is a proof about
         * one event generation on one stream, not a general coherence flag.
         * The next device publication clears it before re-recording the event.
         */
        base->last_joined_completion_event_ = base->device_completion_event_;
        base->last_joined_consumer_stream_ = consumer_stream;
    }

    void TransferEngine::prepareHostInput(ITensor *tensor)
    {
        TensorBase *base = requireTransferStorageOwner(
            tensor,
            "TransferEngine::prepareHostInput");
        const TransferResult result = instance().download(base);
        if (!result.success)
        {
            throw std::runtime_error(
                "TransferEngine::prepareHostInput failed to materialize tensor: " +
                result.error);
        }
        if (!base->hostValid())
        {
            throw std::runtime_error(
                "TransferEngine::prepareHostInput completed without valid host storage");
        }
    }

    void TransferEngine::allocateDeviceStorage(
        ITensor *tensor,
        DeviceId target_device)
    {
        if (!target_device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::allocateDeviceStorage requires a GPU target");
        }

        TensorBase *base = requireTransferStorageOwner(
            tensor,
            "TransferEngine::allocateDeviceStorage");
        if (!base->allocateOnDevice(target_device))
        {
            throw std::runtime_error(
                "TransferEngine::allocateDeviceStorage failed to allocate tensor on " +
                target_device.toString());
        }
        if (!base->gpu_data_ptr() ||
            !base->current_device().has_value() ||
            *base->current_device() != target_device)
        {
            throw std::runtime_error(
                "TransferEngine::allocateDeviceStorage completed without storage on " +
                target_device.toString());
        }
    }

    void TransferEngine::prepareDeviceOutput(
        ITensor *tensor,
        DeviceId target_device,
        void *stream)
    {
        if (!stream)
        {
            throw std::invalid_argument(
                "TransferEngine::prepareDeviceOutput requires the exact "
                "non-null producer stream");
        }
        allocateDeviceStorage(tensor, target_device);
    }

    void TransferEngine::publishDeviceWrite(
        TensorBase *tensor,
        DeviceId device,
        void *producer_stream)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishDeviceWrite");
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::publishDeviceWrite requires a GPU device");
        }
        if (!producer_stream)
        {
            throw std::invalid_argument(
                "TransferEngine::publishDeviceWrite requires the exact "
                "non-null producer stream");
        }
        tensor->publishDeviceWriteStateWithEvent(device, producer_stream);
    }

    void TransferEngine::publishDeviceWrite(
        ITensor *tensor,
        DeviceId device,
        void *producer_stream)
    {
        publishDeviceWrite(
            requireTensorBase(tensor, "TransferEngine::publishDeviceWrite"),
            device,
            producer_stream);
    }

    void TransferEngine::publishCompletedDeviceWrite(
        TensorBase *tensor,
        DeviceId device)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishCompletedDeviceWrite");
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::publishCompletedDeviceWrite requires a GPU device");
        }
        tensor->publishCompletedDeviceWriteState(device);
    }

    void TransferEngine::publishCompletedDeviceWrite(
        ITensor *tensor,
        DeviceId device)
    {
        publishCompletedDeviceWrite(
            requireTensorBase(
                tensor,
                "TransferEngine::publishCompletedDeviceWrite"),
            device);
    }

    void TransferEngine::publishCurrentDeviceWrite(
        TensorBase *tensor,
        void *producer_stream)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishCurrentDeviceWrite");
        const auto device = tensor->current_device();
        if (!device.has_value() || !device->is_gpu())
        {
            throw std::runtime_error(
                "TransferEngine::publishCurrentDeviceWrite requires an "
                "unambiguous current GPU device");
        }
        publishDeviceWrite(tensor, *device, producer_stream);
    }

    void TransferEngine::publishCurrentDeviceWrite(
        ITensor *tensor,
        void *producer_stream)
    {
        publishCurrentDeviceWrite(
            requireTensorBase(
                tensor,
                "TransferEngine::publishCurrentDeviceWrite"),
            producer_stream);
    }

    void TransferEngine::publishGraphOwnedDeviceWrite(
        TensorBase *tensor,
        DeviceId device)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishGraphOwnedDeviceWrite");
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "TransferEngine::publishGraphOwnedDeviceWrite requires a GPU device");
        }
        tensor->publishGraphOwnedDeviceWriteState(device);
    }

    void TransferEngine::publishGraphOwnedDeviceWrite(
        ITensor *tensor,
        DeviceId device)
    {
        publishGraphOwnedDeviceWrite(
            requireTensorBase(
                tensor,
                "TransferEngine::publishGraphOwnedDeviceWrite"),
            device);
    }

    void TransferEngine::publishGraphOwnedCurrentDeviceWrite(
        TensorBase *tensor)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishGraphOwnedCurrentDeviceWrite");
        const auto device = tensor->current_device();
        if (!device.has_value() || !device->is_gpu())
        {
            throw std::runtime_error(
                "TransferEngine::publishGraphOwnedCurrentDeviceWrite requires "
                "an unambiguous current GPU device");
        }
        publishGraphOwnedDeviceWrite(tensor, *device);
    }

    void TransferEngine::publishGraphOwnedCurrentDeviceWrite(ITensor *tensor)
    {
        publishGraphOwnedCurrentDeviceWrite(
            requireTensorBase(
                tensor,
                "TransferEngine::publishGraphOwnedCurrentDeviceWrite"));
    }

    void TransferEngine::publishHostWrite(TensorBase *tensor)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishHostWrite");
        tensor->publishHostWriteState();
    }

    void TransferEngine::publishHostWrite(ITensor *tensor)
    {
        publishHostWrite(
            requireTensorBase(tensor, "TransferEngine::publishHostWrite"));
    }

    void TransferEngine::publishSynchronized(TensorBase *tensor)
    {
        tensor = requireTransferStorageOwner(
            tensor,
            "TransferEngine::publishSynchronized");
        tensor->publishSynchronizedState();
    }

    void TransferEngine::publishSynchronized(ITensor *tensor)
    {
        publishSynchronized(
            requireTensorBase(tensor, "TransferEngine::publishSynchronized"));
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
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "tensor has no canonical transfer-storage owner");
        }

        /*
         * uploadFull() is the single implementation of allocation, primary
         * device promotion, H2D transfer, event joining, and coherence
         * publication.  Keeping a second implementation here previously let a
         * successful copy land in secondary storage without updating
         * gpu_data_ptr_ or gpu_device_.  Public callers now enter the same
         * lifecycle as TensorBase::ensureOnDevice().
         */
        std::lock_guard<std::mutex> lock(tensor->coherence_mutex_);
        return uploadFull(tensor, target_device, nullptr);
    }

    TransferResult TransferEngine::download(TensorBase *tensor)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "tensor has no canonical transfer-storage owner");
        }

        /*
         * downloadFull() owns completion-event validation, D2H ordering, and
         * host publication.  Delegating to it prevents wrapper and concrete
         * tensors from acquiring subtly different host-read semantics.
         */
        std::lock_guard<std::mutex> lock(tensor->coherence_mutex_);
        return downloadFull(tensor, nullptr);
    }

    TransferResult TransferEngine::transferActivation(
        TensorBase *tensor,
        DeviceId target_device,
        size_t bytes_override)
    {
        if (!tensor)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor");
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "tensor has no canonical transfer-storage owner");
        }

        /*
         * Select the actual authoritative source. The authoritative copy may
         * live in a secondary device buffer after a previous pipeline handoff,
         * so gpu_device_ alone is not a sufficient source-of-truth.
         */
        DeviceId src = DeviceId::cpu();
        const auto authoritative_device = tensor->getAuthoritativeDevice();
        if (authoritative_device.has_value() &&
            authoritative_device->is_gpu() &&
            tensor->deviceValid())
        {
            src = *authoritative_device;
        }
        else if (!tensor->hostValid())
        {
            return TransferResult::fail(TransferMethod::NOOP,
                                        "tensor has no valid data on any device");
        }

        const MemoryResidency residency = tensor->memoryResidency();
        TransferMethod method = planTransfer(src, target_device, residency);

        if (method == TransferMethod::NOOP || method == TransferMethod::MAPPED_NOOP)
        {
            /*
             * A valid copy can be stored in the secondary map. Promote it so
             * gpu_data_ptr() and current_device() describe the same storage
             * without pretending that a transfer occurred.
             */
            if (target_device.is_gpu() &&
                (!tensor->gpu_device_.has_value() ||
                 *tensor->gpu_device_ != target_device))
            {
                void *target_ptr =
                    tensor->getOrAllocateDeviceBuffer(target_device);
                if (!target_ptr)
                {
                    return TransferResult::fail(
                        method,
                        "authoritative device has no activation buffer on " +
                            target_device.toString());
                }
                if (tensor->gpu_device_.has_value() && tensor->gpu_data_ptr_)
                {
                    tensor->secondary_device_buffers_
                        [TensorBase::packDeviceId(*tensor->gpu_device_)] =
                        tensor->gpu_data_ptr_;
                }
                tensor->gpu_device_ = target_device;
                tensor->gpu_data_ptr_ = target_ptr;
                tensor->secondary_device_buffers_.erase(
                    TensorBase::packDeviceId(target_device));
            }
            return TransferResult::ok(method);
        }

        TransferRequest req;
        req.source = makeMemoryDescriptor(tensor);
        if (bytes_override > req.source.size_bytes)
        {
            return TransferResult::fail(
                method,
                "activation byte override exceeds tensor allocation");
        }
        if (bytes_override != 0)
            req.source.size_bytes = bytes_override;
        req.source.device = src;
        if (src.is_gpu())
        {
            req.source.device_ptr =
                tensor->getOrAllocateDeviceBuffer(src);
            if (!req.source.device_ptr)
            {
                return TransferResult::fail(
                    method,
                    "authoritative source has no activation buffer on " +
                        src.toString());
            }
        }
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
                void *target_ptr = req.target_ptr;
                if (!tensor->gpu_device_.has_value() ||
                    *tensor->gpu_device_ != target_device)
                {
                    if (tensor->gpu_device_.has_value() &&
                        tensor->gpu_data_ptr_)
                    {
                        tensor->secondary_device_buffers_
                            [TensorBase::packDeviceId(*tensor->gpu_device_)] =
                            tensor->gpu_data_ptr_;
                    }
                    tensor->gpu_device_ = target_device;
                    tensor->gpu_data_ptr_ = target_ptr;
                    tensor->secondary_device_buffers_.erase(
                        TensorBase::packDeviceId(target_device));
                }
                publishDeviceWrite(
                    tensor,
                    target_device,
                    requireTransferStream(
                        target_device,
                        "TransferEngine::transferActivation publication"));
            }
            else
            {
                tensor->applyCoherenceOp_(CoherenceOp::DOWNLOAD);
            }
        }

        return result;
    }

    TransferResult TransferEngine::copyActivation(TensorBase *src, TensorBase *dst,
                                                  DeviceId dst_device, size_t bytes)
    {
        if (!src || !dst)
            return TransferResult::fail(TransferMethod::NOOP, "null tensor in copyActivation");
        src = src->transferStorageOwner();
        dst = dst->transferStorageOwner();
        if (!src || !dst ||
            src->transferStorageOwner() != src ||
            dst->transferStorageOwner() != dst)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "activation has no canonical transfer-storage owner");
        }
        if (bytes == 0)
            return TransferResult::ok(TransferMethod::NOOP);

        // ----------------------------------------------------------------------
        // Host or mapped destination: a plain host-side copy is correct and
        // cheapest. Mapped memory shares host/device storage, so writing the
        // host side is immediately visible to the device — no transfer needed.
        // ----------------------------------------------------------------------
        if (dst_device.is_cpu() || dst->is_mapped_ || src->is_mapped_)
        {
            const void *host_src = src->data();   // ensures src is host-valid
            void *host_dst = dst->mutable_data(); // host (or mapped) storage
            if (!host_src || !host_dst)
                return TransferResult::fail(TransferMethod::DEVICE_TO_HOST,
                                            "null host pointer in copyActivation host path");
            std::memcpy(host_dst, host_src, bytes);
            if (dst_device.is_gpu())
            {
                /*
                 * Mapped host memory is immediately visible to the GPU, but the
                 * coherence publication still needs a concrete completion
                 * token. Record it on the destination backend's owned default
                 * stream so later GPU consumers never inherit eventless device
                 * authority from this host-visible transfer boundary.
                 */
                publishCompletedDeviceWrite(dst, dst_device);
            }
            return TransferResult::ok(dst->is_mapped_ ? TransferMethod::MAPPED_NOOP
                                                      : TransferMethod::DEVICE_TO_HOST);
        }

        // ----------------------------------------------------------------------
        // GPU destination: ensure a device buffer exists. We deliberately do NOT
        // upload host data here (the buffer is about to be overwritten by the
        // copy below).
        // ----------------------------------------------------------------------
        void *dst_ptr = dst->getOrAllocateDeviceBuffer(dst_device);
        if (!dst_ptr)
            return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                        "failed to allocate dst device buffer on " + dst_device.toString());

        // Locate the authoritative source data. NOTE: we use getAuthoritativeDevice()
        // rather than gpu_device_: after a cross-vendor transferActivation(), the
        // tensor's primary gpu_device_/gpu_data_ptr_ still point at the producing
        // GPU (e.g. CUDA), while the authoritative copy lives in a secondary buffer
        // on the consumer GPU (e.g. ROCm).
        const auto auth = src->getAuthoritativeDevice();
        const bool src_on_gpu = auth.has_value() && auth->is_gpu();
        const DeviceId src_device = src_on_gpu ? *auth : DeviceId::cpu();

        TransferResult result;

        // Decide whether a direct device-to-device path exists for this pair.
        // Same physical GPU is always direct (intra-VRAM memcpy). Different GPUs
        // of the SAME vendor go through the collective backend (NCCL/RCCL/peer
        // DMA). Only cross-vendor pairs (CUDA↔ROCm) have no direct path and must
        // bounce through the host.
        const bool same_physical_gpu = src_on_gpu && src_device == dst_device;
        const bool same_vendor_diff_gpu =
            src_on_gpu && !same_physical_gpu && src_device.type == dst_device.type;

        if (same_physical_gpu)
        {
            // Same physical GPU: pure device-to-device copy — no host bounce.
            void *src_ptr = src->getOrAllocateDeviceBuffer(src_device);
            IBackend *backend = resolveBackend(dst_device);
            if (!backend)
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "no backend for " + dst_device.toString());
            if (!src_ptr)
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "source has no device buffer on " + dst_device.toString());
            void *const destination_stream =
                requireTransferStream(
                    dst_device,
                    "TransferEngine::copyActivation same-device copy");
            if (!backend->deviceToDevice(
                    dst_ptr,
                    src_ptr,
                    bytes,
                    dst_device.gpu_ordinal(),
                    destination_stream))
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "deviceToDevice failed on " + dst_device.toString());
            result = TransferResult::ok(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
        }
        else if (same_vendor_diff_gpu)
        {
            // Same-vendor, different GPU (e.g. cuda:0 → cuda:1): use the
            // collective backend's peer copy (NCCL/RCCL or peer DMA). No host
            // bounce — the data moves directly across the PCIe/NVLink fabric.
            void *src_ptr = src->getOrAllocateDeviceBuffer(src_device);
            auto *router = GlobalBackendRouter::get();
            ICollectiveBackend *backend =
                router ? router->getBackendForCopy(src_device, dst_device) : nullptr;
            if (!src_ptr)
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "source has no device buffer on " + src_device.toString());
            if (!backend || !backend->supportsCopy(src_device, dst_device))
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "no collective backend supports peer copy " +
                                                src_device.toString() + " -> " + dst_device.toString());
            if (!backend->copy(dst_ptr, dst_device, src_ptr, src_device, bytes))
                return TransferResult::fail(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND,
                                            "peer copy failed " + src_device.toString() +
                                                " -> " + dst_device.toString());
            result = TransferResult::ok(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
        }
        else
        {
            // Cross-vendor GPU pair (CUDA↔ROCm) or host-resident source: there
            // is no direct device path across vendors, so stage through the
            // source tensor's own host buffer and upload into the dst device
            // buffer. This is what makes heterogeneous CUDA↔ROCm PP work.
            const void *host_src = nullptr;
            if (src_on_gpu)
            {
                void *src_host = src->raw_host_data_ptr();
                void *src_dev = src->getOrAllocateDeviceBuffer(src_device);
                IBackend *src_backend = resolveBackend(src_device);
                if (!src_host || !src_dev || !src_backend)
                    return TransferResult::fail(TransferMethod::HOST_STAGED,
                                                "missing src host/device buffer or backend for staged copy");
                void *const source_stream =
                    requireTransferStream(
                        src_device,
                        "TransferEngine::copyActivation staged D2H");
                if (!src_backend->deviceToHost(
                        src_host,
                        src_dev,
                        bytes,
                        src_device.gpu_ordinal(),
                        source_stream))
                    return TransferResult::fail(TransferMethod::HOST_STAGED,
                                                "D2H step failed in copyActivation");
                host_src = src_host;
            }
            else
            {
                host_src = src->data(); // host-resident source
            }

            IBackend *dst_backend = resolveBackend(dst_device);
            if (!dst_backend || !host_src)
                return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                            "missing dst backend or host src in copyActivation");
            void *const destination_stream =
                requireTransferStream(
                    dst_device,
                    "TransferEngine::copyActivation staged H2D");
            if (!dst_backend->hostToDevice(
                    dst_ptr,
                    host_src,
                    bytes,
                    dst_device.gpu_ordinal(),
                    destination_stream))
                return TransferResult::fail(TransferMethod::HOST_TO_DEVICE,
                                            "H2D step failed in copyActivation");
            result = TransferResult::ok(src_on_gpu ? TransferMethod::HOST_STAGED
                                                   : TransferMethod::HOST_TO_DEVICE);
        }

        // ----------------------------------------------------------------------
        // Promote dst_ptr to the primary device buffer if it currently lives in
        // the secondary map (so later gpu_data_ptr() returns the buffer we just
        // wrote), then mark the destination authoritative on dst_device.
        // ----------------------------------------------------------------------
        if (!dst->gpu_device_.has_value() || *dst->gpu_device_ != dst_device)
        {
            // Preserve any existing primary buffer as a secondary so it is not leaked.
            if (dst->gpu_device_.has_value() && dst->gpu_data_ptr_)
                dst->secondary_device_buffers_[TensorBase::packDeviceId(*dst->gpu_device_)] =
                    dst->gpu_data_ptr_;
            dst->gpu_device_ = dst_device;
            dst->gpu_data_ptr_ = dst_ptr;
            dst->secondary_device_buffers_.erase(TensorBase::packDeviceId(dst_device));
        }
        /*
         * copyActivation() currently completes its selected transport before
         * returning. Publish a fresh backend event after that boundary rather
         * than erasing completion metadata with a plain coherence transition.
         * The event gives every later device consumer one uniform dependency
         * contract regardless of whether the bytes arrived via intra-device,
         * peer, or deliberately heterogeneous host-staged transport.
         */
        publishDeviceWrite(
            dst,
            dst_device,
            requireTransferStream(
                dst_device,
                "TransferEngine::copyActivation publication"));

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

        bool ok = backend->hostToDevice(
            req.target_ptr,
            req.source.host_ptr,
            req.source.size_bytes,
            req.target_device.ordinal,
            requireTransferStream(
                req.target_device,
                "TransferEngine::executeHostToDevice"));
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

        bool ok = backend->deviceToHost(
            req.source.host_ptr,
            req.source.device_ptr,
            req.source.size_bytes,
            req.source.device.ordinal,
            requireTransferStream(
                req.source.device,
                "TransferEngine::executeDeviceToHost"));
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

        /*
         * Same-vendor movement is a collective-backend responsibility. NCCL
         * and RCCL select their best available P2P transport; silently bouncing
         * through host memory here would hide a broken device path and violate
         * the device-owned pipeline contract.
         */
        auto *router = GlobalBackendRouter::get();
        ICollectiveBackend *backend =
            router
                ? router->getBackendForCopy(
                      req.source.device,
                      req.target_device)
                : nullptr;
        if (!backend ||
            !backend->supportsCopy(
                req.source.device,
                req.target_device))
        {
            return TransferResult::fail(
                req.method,
                "no direct collective backend supports " +
                    req.source.device.toString() + " -> " +
                    req.target_device.toString());
        }
        if (!backend->copy(
                req.target_ptr,
                req.target_device,
                req.source.device_ptr,
                req.source.device,
                req.source.size_bytes))
        {
            return TransferResult::fail(
                req.method,
                "collective backend copy failed " +
                    req.source.device.toString() + " -> " +
                    req.target_device.toString());
        }

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
            req.source.size_bytes, req.source.device.ordinal,
            requireTransferStream(
                req.source.device,
                "TransferEngine::executeHostStaged source"));
        if (!d2h)
            return TransferResult::fail(req.method, "D2H step of host staged failed");

        // Step 2: H2D to target
        IBackend *dst_backend = resolveBackend(req.target_device);
        if (!dst_backend)
            return TransferResult::fail(req.method,
                                        "no backend for target " + req.target_device.toString());

        bool h2d = dst_backend->hostToDevice(
            req.target_ptr, req.source.host_ptr,
            req.source.size_bytes, req.target_device.ordinal,
            requireTransferStream(
                req.target_device,
                "TransferEngine::executeHostStaged destination"));
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
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "tensor has no canonical transfer-storage owner");
        }

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
                LOG_TRACE("[TransferEngine::uploadFull] ZERO-COPY: Tensor is mapped, no memcpy needed");
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
                    if (stream)
                    {
                        // Non-blocking: make the consuming stream wait for the event.
                        // Same-stream waits are no-ops in hardware (ordering is implicit).
                        // Cross-stream waits correctly serialize without blocking the CPU.
                        if (!backend->streamWaitEvent(stream, tensor->device_completion_event_, backend_device_id))
                        {
                            return TransferResult::fail(
                                TransferMethod::NOOP,
                                "Stream event wait failed for tensor '" +
                                    (tensor->debug_name_.empty()
                                         ? std::string("(unnamed)")
                                         : tensor->debug_name_) +
                                    "' on " + target_device.toString() +
                                    "; refusing a host-blocking fallback");
                        }
                    }
                    else
                    {
                        // No stream provided (host-side access like tensor->data()).
                        // Must block CPU until GPU work completes.
                        if (!waitForEventWithProxy(backend, tensor->device_completion_event_, backend_device_id, target_device))
                        {
                            LOG_ERROR("[TransferEngine::uploadFull] Event wait failed for tensor '"
                                      << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                                      << "' on device " << target_device.toString()
                                      << " — this indicates a corrupted or invalid completion event "
                                      << "(e.g., event recorded during graph capture)");
                            return TransferResult::fail(TransferMethod::NOOP,
                                                        "Event wait failed: completion event is invalid");
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
        LOG_TRACE("[TransferEngine::uploadFull] tensor=" << static_cast<void *>(tensor)
                                                         << " target_device=" << target_device.toString()
                                                         << " gpu_data_ptr_=" << tensor->gpu_data_ptr_
                                                         << " gpu_device_=" << (tensor->gpu_device_.has_value() ? tensor->gpu_device_->toString() : "none")
                                                         << " device_completion_event_=" << tensor->device_completion_event_);
        if (tensor->gpu_data_ptr_ && tensor->gpu_device_.has_value() && *tensor->gpu_device_ != target_device)
        {
            DeviceId old_device = *tensor->gpu_device_;
            LOG_TRACE("[TransferEngine::uploadFull] Device migration: " << tensor->gpu_device_->toString()
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

                LOG_TRACE("[TransferEngine::uploadFull] Promoted secondary buffer to primary for "
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
                        LOG_TRACE("[TransferEngine::uploadFull] Destroying old completion event on device "
                                  << tensor->gpu_device_->toString() << " before migrating to " << target_device.toString());
                        old_backend->destroyEvent(tensor->device_completion_event_, old_backend_device_id);
                    }
                    tensor->device_completion_event_ = nullptr;
                    tensor->event_device_.reset();
                }

                tensor->gpu_data_ptr_ = nullptr;
                tensor->gpu_device_.reset();
                tensor->applyCoherenceOp_(CoherenceOp::RELEASE_DEVICE);

                LOG_TRACE("[TransferEngine::uploadFull] Parked previous primary buffer for "
                          << old_device.toString() << " and allocating fresh buffer for "
                          << target_device.toString());
            }
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
                LOG_TRACE("[TransferEngine::uploadFull] backend->allocate(" << bytes << " bytes) took " << alloc_us << " us");
            }

            LOG_TRACE("[GPU_ALLOC] tensor=" << static_cast<void *>(tensor)
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
                LOG_TRACE("[TransferEngine::uploadFull] ensureHostPinned() took " << pin_us << " us");
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

            void *const upload_stream =
                stream
                    ? stream
                    : requireTransferStream(
                          target_device,
                          "TransferEngine::uploadFull");
            auto h2d_start = std::chrono::high_resolution_clock::now();
            bool h2d_ok = target_backend->hostToDevice(
                tensor->gpu_data_ptr_,
                src,
                bytes,
                backend_device_id,
                upload_stream);
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
                LOG_TRACE("[TransferEngine::uploadFull] hostToDevice(" << bytes << " bytes) took "
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

            LOG_TRACE("[TransferEngine::uploadFull] Uploaded " << bytes
                                                               << " bytes to device " << target_device.toString()
                                                               << " (backend device ID: " << backend_device_id << ")");
        }

        auto overall_end = std::chrono::high_resolution_clock::now();
        auto overall_us = std::chrono::duration_cast<std::chrono::microseconds>(overall_end - overall_start).count();
        if (trace && overall_us > 1000)
        {
            LOG_TRACE("[TransferEngine::uploadFull] TOTAL took " << overall_us << " us for " << bytes << " bytes");
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
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
        {
            return TransferResult::fail(
                TransferMethod::NOOP,
                "tensor has no canonical transfer-storage owner");
        }

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
                            return TransferResult::fail(
                                TransferMethod::MAPPED_NOOP,
                                "Mapped tensor completion event wait failed");
                        }
                    }
                    else
                    {
                        return TransferResult::fail(
                            TransferMethod::MAPPED_NOOP,
                            "Mapped GPU tensor has no completion event");
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

            /*
             * Explicit-stream host publication is already an ordering contract:
             * the caller passes the stream that produced the device bytes, and
             * IBackend::deviceToHost() enqueues the D2H copy on that stream and
             * synchronizes it before returning.  Waiting a tensor completion
             * event first is redundant, and on HIP it can be illegal when the
             * event came from graph-capture bookkeeping for a replay stream.
             *
             * Keep the hard-error event path for ordinary host publication with
             * no producer stream; those callers do not have a stream dependency
             * that can safely replace the completion event.
             */
            if (stream)
            {
                LOG_TRACE("[TransferEngine::downloadFull] Explicit producer stream supplied; D2H copy will synchronize that stream");
            }
            else if (tensor->device_completion_event_)
            {
                LOG_TRACE("[TransferEngine::downloadFull] Using event-based sync (waiting for specific kernel)");
                if (!waitForEventWithProxy(backend, tensor->device_completion_event_, backend_device_id, *tensor->gpu_device_))
                {
                    LOG_ERROR("[TransferEngine::downloadFull] Event wait failed for tensor '"
                              << tensor->debug_name_ << "' on device " << tensor->gpu_device_->toString()
                              << " — this indicates a corrupted or invalid completion event "
                              << "(e.g., event recorded during graph capture)");
                    return TransferResult::fail(TransferMethod::DEVICE_TO_HOST,
                                                "Event wait failed: completion event is invalid");
                }
            }
            else
            {
                return TransferResult::fail(
                    TransferMethod::DEVICE_TO_HOST,
                    "GPU tensor has no completion event or explicit producer stream");
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

            LOG_TRACE("[TransferEngine::downloadFull] ATTEMPTING D2H: "
                      << "tensor=" << (tensor->debug_name_.empty() ? "(unnamed)" : tensor->debug_name_)
                      << " gpu_data_ptr=" << static_cast<const void *>(tensor->gpu_data_ptr_)
                      << " dst=" << static_cast<const void *>(dst)
                      << " bytes=" << bytes
                      << " device=" << tensor->gpu_device_->toString()
                      << " backend_device_id=" << backend_device_id);

            void *const download_stream =
                stream
                    ? stream
                    : requireTransferStream(
                          *tensor->gpu_device_,
                          "TransferEngine::downloadFull");
            auto d2h_start = std::chrono::high_resolution_clock::now();
            bool d2h_ok = backend->deviceToHost(
                dst,
                tensor->gpu_data_ptr_,
                bytes,
                backend_device_id,
                download_stream);
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
                    LOG_TRACE("[TransferEngine::downloadFull] D2H leading sample is zero: tensor="
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

            LOG_TRACE("[TransferEngine::downloadFull] Downloaded " << bytes
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
            LOG_TRACE("[TransferEngine] " << method_str
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
        tensor = tensor->transferStorageOwner();
        if (!tensor || tensor->transferStorageOwner() != tensor)
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
