/**
 * @file OriginalDeviceBindingBodies.inc
 * @brief Frozen original native coordinator selection bodies, replayed through recorded native statuses.
 *
 * Only native type/function names are supplied by local shims. The actual
 * function-local TLS bodies below are retained verbatim from the frozen source.
 */
/** @brief Actual original CUDA allreduce selection interval. */
bool originalCUDAAllreduce(RecordedNative &native, int ordinal)
{
    using cudaError_t = int;
    constexpr int cudaSuccess = 0;
    auto cudaSetDevice = [&](int selected) { return native.select(selected); };
    auto cudaGetErrorString = [](int) { return "recorded native error"; };
    std::string last_error_;
        static thread_local int tl_last_cuda_device = -1;
        if (tl_last_cuda_device != ordinal)
        {
            cudaError_t err = cudaSetDevice(ordinal);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice failed: ") + cudaGetErrorString(err);
                return false;
            }
            tl_last_cuda_device = ordinal;
        }
    return true;
}
/** @brief Actual original CUDA allgather selection interval. */
bool originalCUDAAllgather(RecordedNative &native, int ordinal)
{
    using cudaError_t = int;
    constexpr int cudaSuccess = 0;
    auto cudaSetDevice = [&](int selected) { return native.select(selected); };
    auto cudaGetErrorString = [](int) { return "recorded native error"; };
    std::string last_error_;
        static thread_local int tl_last_cuda_device_for_allgather = -1;
        if (tl_last_cuda_device_for_allgather != ordinal)
        {
            cudaError_t err = cudaSetDevice(ordinal);
            if (err != cudaSuccess)
            {
                last_error_ = std::string("cudaSetDevice failed: ") + cudaGetErrorString(err);
                return false;
            }
            tl_last_cuda_device_for_allgather = ordinal;
        }
    return true;
}
/** @brief Actual original HIP allreduce selection interval. */
bool originalHIPAllreduce(RecordedNative &native, int ordinal)
{
    using hipError_t = int;
    constexpr int hipSuccess = 0;
    auto trackedHipSetDevice = [&](int selected) { return native.select(selected); };
    auto hipGetErrorString = [](int) { return "recorded native error"; };
    std::string last_error_;
        static thread_local int tl_last_hip_device = -1;
        if (tl_last_hip_device != ordinal)
        {
            hipError_t err = trackedHipSetDevice(ordinal);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
                return false;
            }
            tl_last_hip_device = ordinal;
        }
    return true;
}
/** @brief Actual original HIP allgather selection interval. */
bool originalHIPAllgather(RecordedNative &native, int ordinal)
{
    using hipError_t = int;
    constexpr int hipSuccess = 0;
    auto trackedHipSetDevice = [&](int selected) { return native.select(selected); };
    auto hipGetErrorString = [](int) { return "recorded native error"; };
    std::string last_error_;
        static thread_local int tl_last_hip_device_for_allgather = -1;
        if (tl_last_hip_device_for_allgather != ordinal)
        {
            hipError_t err = trackedHipSetDevice(ordinal);
            if (err != hipSuccess)
            {
                last_error_ = std::string("hipSetDevice failed: ") + hipGetErrorString(err);
                return false;
            }
            tl_last_hip_device_for_allgather = ordinal;
        }
    return true;
}
