/** @file OriginalNativeEventDestructionBodies.inc
 * @brief Frozen actual pre-fix cleanup bodies for fake-SDK original-red controls. */
    void NvidiaDeviceContext::destroyEvent(void *event)
    {
        if (event == nullptr)
        {
            return;
        }

        cudaError_t set_err = cudaSetDevice(device_ordinal_);
        if (set_err != cudaSuccess)
        {
            LOG_ERROR("[NvidiaDeviceContext] cudaSetDevice(" << device_ordinal_
                                                             << ") failed in destroyEvent: " << cudaGetErrorString(set_err));
            return;
        }

        cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
        CUDA_CHECK_VOID(cudaEventDestroy(cuda_event));
    }

    void AMDDeviceContext::destroyEvent(void *event)
    {
        if (event == nullptr)
        {
            return;
        }

        hipEvent_t hip_event = static_cast<hipEvent_t>(event);
        if (!setAMDDeviceForResource(device_ordinal_, "destroyEvent"))
        {
            return;
        }
        HIP_CHECK_VOID(hipEventDestroy(hip_event));
    }

    void CUDABackend::destroyEvent(void *event, int device_id)
    {
        if (!event || device_id >= device_count_ || device_id < 0)
        {
            return;
        }

        CUDA_WARN_IF_FAIL(cudaSetDevice(device_id)); // cleanup path
        cudaEvent_t cuda_event = reinterpret_cast<cudaEvent_t>(event);
        CUDA_WARN_IF_FAIL(cudaEventDestroy(cuda_event));
    }

    void ROCmBackend::destroyEvent(void *event, int device_id)
    {
        if (!event || device_id >= device_count_ || device_id < 0)
        {
            return;
        }

        HipDeviceSaveRestore device_guard;
        HIP_WARN_IF_FAIL(hipSetDevice(device_id));
        hipEvent_t hip_event = reinterpret_cast<hipEvent_t>(event);
        HIP_WARN_IF_FAIL(hipEventDestroy(hip_event));
    }
