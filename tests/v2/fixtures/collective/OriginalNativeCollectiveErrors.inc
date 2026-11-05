/**
 * @file OriginalNativeCollectiveErrors.inc
 * @brief Literal pre-fix coordinator byte-enqueue bodies for deferred-diagnostic negative controls.
 * These original bodies deliberately retain shared mutable errors. CPU runtime
 * doubles and ordered writes expose crossed readback without executing a GPU
 * or invoking a C++ data race. No production entrypoint includes this fixture.
 */
namespace original_nccl {
    bool NCCLCoordinator::nativeBroadcastBytesOnStream(const void* send, void* receive,
        const NativeCollectiveBytes& bytes, int root, int participant, void* stream)
    {
#ifdef HAVE_NCCL
        if (!initialized_.load() || !send || !receive ||
            reinterpret_cast<std::uintptr_t>(stream) <= 2 ||
            !bytes.participantValid(num_devices_, participant, root))
        {
            last_error_ = "NCCL byte Broadcast changed its original participant/root/buffer/stream";
            return false;
        }
        const auto selected = cudaSetDevice(device_ordinals_[participant]);
        if (selected != cudaSuccess)
        {
            last_error_ = std::string("NCCL byte Broadcast device selection failed: ") + cudaGetErrorString(selected);
            return false;
        }
        const auto producer_error = cudaGetLastError();
        if (producer_error != cudaSuccess)
        {
            last_error_ = std::string("NCCL byte Broadcast original producer failed: ") + cudaGetErrorString(producer_error);
            return false;
        }
        const auto result = nccl::nativeBroadcastBytes(send, receive, bytes, root,
            static_cast<nccl::ncclComm_t>(comms_[participant]), stream);
        if (result != nccl::ncclSuccess)
        {
            last_error_ = std::string("NCCL byte Broadcast enqueue failed: ") +
                nccl::ncclGetErrorString(result);
            return false;
        }
        const auto enqueue_error = cudaGetLastError();
        if (enqueue_error != cudaSuccess)
        {
            last_error_ = std::string("NCCL byte Broadcast runtime enqueue failed: ") + cudaGetErrorString(enqueue_error);
            return false;
        }
        return true;
#else
        (void)send; (void)receive; (void)bytes; (void)root; (void)participant; (void)stream;
        last_error_ = "NCCL not available";
        return false;
#endif
    }
}
namespace original_rccl {
    bool RCCLCoordinator::nativeBroadcastBytesOnStream(const void* send, void* receive,
        const NativeCollectiveBytes& bytes, int root, int participant, void* stream)
    {
#ifdef HAVE_RCCL
        if (!initialized_.load() || !send || !receive ||
            reinterpret_cast<std::uintptr_t>(stream) <= 2 ||
            !bytes.participantValid(num_devices_, participant, root))
        {
            last_error_ = "RCCL byte Broadcast changed its original participant/root/buffer/stream";
            return false;
        }
        const auto selected = selectHipDevice(device_ordinals_[participant]);
        if (selected != hipSuccess)
        {
            last_error_ = std::string("RCCL byte Broadcast device selection failed: ") + hipGetErrorString(selected);
            return false;
        }
        const auto producer_error = hipGetLastError();
        if (producer_error != hipSuccess)
        {
            last_error_ = std::string("RCCL byte Broadcast original producer failed: ") + hipGetErrorString(producer_error);
            return false;
        }
        const auto result = rccl::nativeBroadcastBytes(send, receive, bytes, root,
            static_cast<rccl::ncclComm_t>(comms_[participant]), stream);
        if (result != rccl::ncclSuccess)
        {
            last_error_ = std::string("RCCL byte Broadcast enqueue failed: ") +
                rccl::ncclGetErrorString(result);
            return false;
        }
        const auto enqueue_error = hipGetLastError();
        if (enqueue_error != hipSuccess)
        {
            last_error_ = std::string("RCCL byte Broadcast runtime enqueue failed: ") + hipGetErrorString(enqueue_error);
            return false;
        }
        collective_performed_.store(true);
        return true;
#else
        (void)send; (void)receive; (void)bytes; (void)root; (void)participant; (void)stream;
        last_error_ = "RCCL not available";
        return false;
#endif
    }
}
