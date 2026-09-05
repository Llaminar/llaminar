/**
 * @file ROCmEmbeddingKernelT.cpp
 * @brief ROCm embedding kernel host-side implementation
 */

#include "ROCmEmbeddingKernelT.h"
#include "../../../tensors/Tensors.h"
#include "utils/Logger.h"
#include "utils/ROCmKernelProfiler.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../common/EmbeddingWorkspaceContract.h"
#include "../../common/PreparedEmbeddingWeights.h"
#include "../ROCmKernelBase.h"
#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../backends/rocm/ROCmBackend.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>
#include <climits>

// Forward declarations for HIP kernels (defined in ROCmEmbeddingKernels.hip)
extern "C"
{
    hipError_t hipOps_embedding_fp32(
        const float *embed_data,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        int vocab_size,
        int vocab_offset,
        hipStream_t stream);

    hipError_t hipOps_embedding_fp16_source_fp32(
        const uint16_t *embed_data,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        int vocab_size,
        int vocab_offset,
        hipStream_t stream);

    hipError_t hipOps_embedding_bf16_source_fp32(
        const uint16_t *embed_data,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        int vocab_size,
        int vocab_offset,
        hipStream_t stream);

    hipError_t hipOps_embedding_bf16(
        const float *embed_data,
        const int *token_ids,
        uint16_t *output,
        int num_tokens,
        int d_model,
        hipStream_t stream);

    hipError_t hipOps_embedding_fp16(
        const float *embed_data,
        const int *token_ids,
        uint16_t *output,
        int num_tokens,
        int d_model,
        hipStream_t stream);

    hipError_t hipOps_embedding_q8_1(
        const float *embed_data,
        const int *token_ids,
        void *output,
        int num_tokens,
        int d_model,
        hipStream_t stream);

    hipError_t hipOps_embedding_q8(
        const void *embed_q8,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        int blocks_per_row,
        int vocab_size,
        int vocab_offset,
        int debug_probe,
        hipStream_t stream);
}

namespace llaminar2
{

    ROCmEmbeddingKernelT::~ROCmEmbeddingKernelT()
    {
        if (h_token_ids_)
        {
            (void)hipHostFree(h_token_ids_);
            h_token_ids_ = nullptr;
        }
    }

    void ROCmEmbeddingKernelT::bindGPUStream(ExplicitGPUStream stream)
    {
        gpu_stream_ = stream.get();

        int current_device = -1;
        if (hipGetDevice(&current_device) == hipSuccess && current_device >= 0)
        {
            std::lock_guard<std::mutex> lock(stream_mutex_);
            stream_by_device_[current_device] = stream.get();
        }
    }

    void ROCmEmbeddingKernelT::clearGPUStreamBinding()
    {
        gpu_stream_ = nullptr;

        int current_device = -1;
        if (hipGetDevice(&current_device) == hipSuccess && current_device >= 0)
        {
            std::lock_guard<std::mutex> lock(stream_mutex_);
            stream_by_device_.erase(current_device);
        }
    }

    void *ROCmEmbeddingKernelT::getStream() const
    {
        int current_device = -1;
        if (hipGetDevice(&current_device) == hipSuccess && current_device >= 0)
        {
            std::lock_guard<std::mutex> lock(stream_mutex_);
            auto it = stream_by_device_.find(current_device);
            if (it != stream_by_device_.end())
            {
                return it->second;
            }
        }

        return requireExplicitGPUStreamBinding(gpu_stream_, "ROCmEmbeddingKernelT");
    }

    void ROCmEmbeddingKernelT::setDynamicTokenIds(const int *token_ids, int num_tokens)
    {
        dynamic_params_active_ = false;
        dynamic_token_count_ = 0;
        device_token_ids_active_ = false;
        device_token_ids_ = nullptr;
        device_token_count_ = 0;

        if (!token_ids || num_tokens <= 0)
        {
            return;
        }

        if (num_tokens > max_token_ids_)
        {
            if (h_token_ids_)
            {
                (void)hipHostFree(h_token_ids_);
                h_token_ids_ = nullptr;
            }

            hipError_t alloc_err = hipHostMalloc(reinterpret_cast<void **>(&h_token_ids_),
                                                 static_cast<size_t>(num_tokens) * sizeof(int),
                                                 hipHostMallocDefault);
            if (alloc_err != hipSuccess)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] Failed to allocate pinned token buffer: "
                          << hipGetErrorString(alloc_err));
                return;
            }
            max_token_ids_ = num_tokens;
        }

        std::memcpy(h_token_ids_, token_ids, static_cast<size_t>(num_tokens) * sizeof(int));

        int dev = (device_idx_ >= 0) ? device_idx_ : 0;
        hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(dev));
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to set device " << dev << ": " << hipGetErrorString(set_err));
            return;
        }

        DeviceWorkspaceManager *workspace = nullptr;
        {
            std::lock_guard<std::mutex> lock(workspace_mutex_);
            auto it = workspace_by_device_.find(dev);
            if (it != workspace_by_device_.end())
            {
                workspace = it->second;
            }
            else
            {
                workspace = workspace_;
            }
        }

        if (!workspace || !workspace->isAllocated())
        {
            return;
        }

        int *d_token_ids = static_cast<int *>(workspace->getBuffer(EmbeddingWorkspaceBuffers::TOKEN_IDS));
        if (!d_token_ids)
        {
            return;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t copy_err = hipMemcpyAsync(d_token_ids, h_token_ids_,
                                             static_cast<size_t>(num_tokens) * sizeof(int),
                                             hipMemcpyHostToDevice,
                                             stream);
        if (copy_err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to preload token_ids to GPU: "
                      << hipGetErrorString(copy_err));
            return;
        }

        dynamic_token_count_ = num_tokens;
        dynamic_params_active_ = true;
        preload_stream_ = getStream();
    }

    void ROCmEmbeddingKernelT::setDynamicDeviceTokenIds(
        const void *token_ids_device,
        int num_tokens)
    {
        dynamic_params_active_ = false;
        dynamic_token_count_ = 0;
        preload_stream_ = nullptr;
        device_token_ids_ = static_cast<const int *>(token_ids_device);
        device_token_count_ = num_tokens;
        device_token_ids_active_ = token_ids_device && num_tokens > 0;
    }

    void ROCmEmbeddingKernelT::resetDynamicState()
    {
        dynamic_params_active_ = false;
        dynamic_token_count_ = 0;
        device_token_ids_active_ = false;
        device_token_ids_ = nullptr;
        device_token_count_ = 0;
        preload_stream_ = nullptr;
        {
            std::lock_guard<std::mutex> lock(stream_mutex_);
            stream_by_device_.clear();
        }
        // h_token_ids_ buffer is preserved — it's reusable for the next session
    }

    namespace
    {
        bool validatePointerForDevice(const void *ptr,
                                      int expected_device,
                                      const char *ptr_name,
                                      bool fail_on_query_error)
        {
            if (!ptr)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] " << ptr_name << " is null");
                return false;
            }

            hipPointerAttribute_t attr{};
            hipError_t attr_err = hipPointerGetAttributes(&attr, ptr);
            if (attr_err != hipSuccess)
            {
                if (fail_on_query_error)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] Failed to query pointer attributes for " << ptr_name
                                                                                               << " ptr=" << ptr << " err=" << hipGetErrorString(attr_err)
                                                                                               << " expected_device=" << expected_device);
                    ROCmBackend::dumpRecentPointerEvents(32);
                    return false;
                }
                return true;
            }

            if (attr.device != expected_device)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] " << ptr_name << " buffer on wrong device: ptr=" << ptr
                                                    << " attr.device=" << attr.device << " expected=" << expected_device);
                ROCmBackend::dumpRecentPointerEvents(32);
                return false;
            }

            ROCmPointerOwnerInfo owner_info{};
            if (ROCmBackend::queryPointerOwner(ptr, owner_info) && owner_info.active && owner_info.device_id != expected_device)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] " << ptr_name << " owner mismatch: ptr=" << ptr
                                                    << " owner.device=" << owner_info.device_id << " expected=" << expected_device
                                                    << " owner.base=" << owner_info.base_ptr << " owner.bytes=" << owner_info.size_bytes
                                                    << " owner.seq=" << owner_info.sequence);
                ROCmBackend::dumpRecentPointerEvents(32);
                return false;
            }

            return true;
        }

        bool validateTokenIdsHost(const int *token_ids,
                                  int num_tokens,
                                  int vocab_size,
                                  bool fail_on_invalid)
        {
            if (!token_ids || num_tokens <= 0)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] Invalid token buffer on host: token_ids=" << token_ids
                                                                                            << " num_tokens=" << num_tokens);
                return false;
            }

            int min_id = token_ids[0];
            int max_id = token_ids[0];
            int first_invalid_pos = -1;
            int first_invalid_id = -1;

            for (int i = 0; i < num_tokens; ++i)
            {
                int value = token_ids[i];
                min_id = std::min(min_id, value);
                max_id = std::max(max_id, value);
                const bool has_upper_bound = vocab_size > 0;
                if ((value < 0 || (has_upper_bound && value >= vocab_size)) && first_invalid_pos < 0)
                {
                    first_invalid_pos = i;
                    first_invalid_id = value;
                }
            }

            LOG_DEBUG("[ROCmEmbeddingKernelT] Host token stats: num_tokens=" << num_tokens
                                                                             << " vocab_size=" << (vocab_size > 0 ? std::to_string(vocab_size) : std::string("unchecked"))
                                                                             << " min_id=" << min_id
                                                                             << " max_id=" << max_id
                                                                             << " first_id=" << token_ids[0]
                                                                             << " last_id=" << token_ids[num_tokens - 1]);

            if (first_invalid_pos >= 0)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] Host token out of range at pos=" << first_invalid_pos
                                                                                   << " id=" << first_invalid_id
                                                                                   << " vocab_size=" << vocab_size);
                return !fail_on_invalid;
            }

            return true;
        }

        /**
         * @brief Publish proof that one compact verifier embedding launch ran.
         *
         * M=1 calls are the serial oracle and intentionally do not emit this
         * counter. The device-token tag prevents a host upload from accidentally
         * satisfying the device-owned MTP regression gate.
         */
        void recordROCmGroupedEmbeddingCall(
            const TensorBase *embed_table,
            int num_tokens,
            int d_model,
            int device,
            bool uses_device_token_ids,
            const char *weight_route)
        {
            if (num_tokens < 2)
                return;

            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_embedding_grouped_verifier_rows_calls",
                1.0,
                "verifier",
                DeviceId::rocm(device).to_string(),
                {{"weight_format", tensorTypeName(embed_table->native_type())},
                 {"verifier_rows", std::to_string(num_tokens)},
                 {"d_model", std::to_string(d_model)},
                 {"weight_route", weight_route},
                 {"token_source", uses_device_token_ids ? "device" : "host_workspace"},
                 {"invocation_policy", "single_grouped_launch"}});
        }
    }

    bool ROCmEmbeddingKernelT::apply(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        float *output,
        const IMPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        int dev = (device_idx >= 0) ? device_idx : device_idx_;

        hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(dev));
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to set device " << dev << ": " << hipGetErrorString(set_err));
            return false;
        }
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        if (!stream)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] apply_tensor requires an explicit non-null HIP stream");
            return false;
        }

        const int launch_vocab_size = explicit_vocab_range_ && local_vocab_size_ > 0
                                          ? local_vocab_size_
                                          : INT_MAX;
        const int launch_vocab_offset = explicit_vocab_range_ ? vocab_offset_ : 0;
        hipError_t err = hipOps_embedding_fp32(embed_data, token_ids, output, num_tokens, d_model,
                                               launch_vocab_size, launch_vocab_offset, static_cast<hipStream_t>(getStream()));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] FP32 kernel failed: " << hipGetErrorString(err));
            return false;
        }

        return true;
    }

    bool ROCmEmbeddingKernelT::apply_bf16(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        uint16_t *output,
        const IMPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        int dev = (device_idx >= 0) ? device_idx : device_idx_;

        hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(dev));
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to set device " << dev << ": " << hipGetErrorString(set_err));
            return false;
        }

        hipError_t err = hipOps_embedding_bf16(embed_data, token_ids, output, num_tokens, d_model, static_cast<hipStream_t>(getStream()));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] BF16 kernel failed: " << hipGetErrorString(err));
            return false;
        }

        return true;
    }

    bool ROCmEmbeddingKernelT::apply_fp16(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        uint16_t *output,
        const IMPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        int dev = (device_idx >= 0) ? device_idx : device_idx_;

        hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(dev));
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to set device " << dev << ": " << hipGetErrorString(set_err));
            return false;
        }

        hipError_t err = hipOps_embedding_fp16(embed_data, token_ids, output, num_tokens, d_model, static_cast<hipStream_t>(getStream()));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] FP16 kernel failed: " << hipGetErrorString(err));
            return false;
        }

        return true;
    }

    bool ROCmEmbeddingKernelT::apply_q8_1(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        void *output,
        const IMPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        int dev = (device_idx >= 0) ? device_idx : device_idx_;

        hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(dev));
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to set device " << dev << ": " << hipGetErrorString(set_err));
            return false;
        }

        if (d_model % 32 != 0)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Q8_1 requires d_model to be multiple of 32, got " << d_model);
            return false;
        }

        hipError_t err = hipOps_embedding_q8_1(embed_data, token_ids, output, num_tokens, d_model, static_cast<hipStream_t>(getStream()));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Q8_1 kernel failed: " << hipGetErrorString(err));
            return false;
        }

        return true;
    }

    bool ROCmEmbeddingKernelT::apply_tensor(
        const TensorBase *embed_table,
        const int *token_ids,
        int num_tokens,
        int d_model,
        TensorBase *output,
        const IMPIContext *mpi_ctx,
        int device_idx)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::EMBEDDING_LOOKUP, static_cast<hipStream_t>(getStream()));
        (void)mpi_ctx;

        const bool serialize_embedding_stage = debugEnv().validation.serialize_embedding_stage;
        static std::mutex global_serialize_embedding_mutex;
        std::unique_lock<std::mutex> serialize_lock(global_serialize_embedding_mutex, std::defer_lock);
        if (serialize_embedding_stage)
        {
            serialize_lock.lock();
        }

        if (!embed_table || !output)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] apply_tensor: null tensor pointer");
            return false;
        }

        // Output must be FP32
        if (output->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Output must be FP32 tensor, got " << static_cast<int>(output->native_type()));
            return false;
        }

        auto *output_fp32 = dynamic_cast<FP32Tensor *>(output);
        if (!output_fp32)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Output tensor cast to FP32 failed");
            return false;
        }

        // Set target ROCm device
        int dev = (device_idx >= 0) ? device_idx : device_idx_;
        hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(dev));
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to set device " << dev << ": " << hipGetErrorString(set_err));
            return false;
        }
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        if (!stream)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] apply_tensor requires an explicit non-null HIP stream");
            return false;
        }

        DeviceWorkspaceManager *workspace = nullptr;
        {
            std::lock_guard<std::mutex> lock(workspace_mutex_);
            auto it = workspace_by_device_.find(dev);
            if (it != workspace_by_device_.end())
            {
                workspace = it->second;
            }
            else
            {
                workspace = workspace_;
            }
        }

        // =====================================================================
        // Step 1: Get token_ids buffer from workspace and copy data
        // =====================================================================
        if (!validateROCmWorkspaceBinding(workspace, dev, "ROCmEmbeddingKernelT"))
        {
            return false;
        }

        int *workspace_token_ids = static_cast<int *>(workspace->getBuffer(EmbeddingWorkspaceBuffers::TOKEN_IDS));
        if (!workspace_token_ids)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Workspace buffer '" << EmbeddingWorkspaceBuffers::TOKEN_IDS << "' not found");
            return false;
        }

        const bool use_device_token_ids =
            device_token_ids_active_ &&
            device_token_count_ == num_tokens &&
            device_token_ids_ != nullptr;
        int *d_token_ids = use_device_token_ids
                               ? const_cast<int *>(device_token_ids_)
                               : workspace_token_ids;

        const bool validate_gpu_ptrs = debugEnv().validation.validate_gpu_ptrs;
        const int host_token_vocab_bound =
            allow_out_of_range_token_ids_
                ? 0
                : static_cast<int>(embed_table->rows());
        if (!use_device_token_ids &&
            validate_gpu_ptrs &&
            !validateTokenIdsHost(token_ids, num_tokens, host_token_vocab_bound, /*fail_on_invalid=*/true))
        {
            return false;
        }

        if (validate_gpu_ptrs &&
            !validatePointerForDevice(d_token_ids, dev, "TOKEN_IDS", /*fail_on_query_error=*/true))
        {
            return false;
        }

        size_t token_bytes = static_cast<size_t>(num_tokens) * sizeof(int);
        // Use async copy on the device stream so this operation is compatible
        // with HIP/CUDA stream capture (GPU graph recording). Synchronous
        // hipMemcpy uses the legacy stream which would create a dependency on
        // a capturing stream, causing capture to fail.
        hipError_t err = hipSuccess;
        // Verify preloaded data matches current request to prevent stale tokens
        // after clear_cache(). The kernel is cached in KernelFactory and
        // dynamic_params_active_ persists across graph rebuilds.
        // Also verify stream match: setDynamicTokenIds() may have run on a
        // different stream than the current gpu_stream_ if the graph capture
        // controller reassigned stage streams after updateDynamicParams().
        const bool token_ids_preloaded =
            dynamic_params_active_ &&
            dynamic_token_count_ == num_tokens &&
            preload_stream_ == getStream() &&
            token_ids &&
            h_token_ids_ &&
            std::memcmp(h_token_ids_, token_ids, static_cast<size_t>(num_tokens) * sizeof(int)) == 0;
        if (!use_device_token_ids && !token_ids_preloaded)
        {
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[ROCmEmbeddingKernelT::apply_tensor] token_ids not preloaded during graph capture — "
                          "updateDynamicParams must be called before capture/replay");
                return false;
            }
            dynamic_params_active_ = false;
            dynamic_token_count_ = 0;
            if (!token_ids)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] Host token IDs are null and no device token source is active");
                return false;
            }
            err = hipMemcpyAsync(workspace_token_ids, token_ids, token_bytes, hipMemcpyHostToDevice, stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] Failed to copy token_ids to GPU: " << hipGetErrorString(err));
                return false;
            }
        }

        // =====================================================================
        // Step 2: Get GPU pointer for output
        // =====================================================================
        float *d_output = static_cast<float *>(output_fp32->gpu_data_ptr());
        if (!d_output)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Output GPU pointer is null");
            return false;
        }
        if (validate_gpu_ptrs &&
            !validatePointerForDevice(d_output, dev, "OUTPUT", /*fail_on_query_error=*/true))
        {
            return false;
        }

        const bool capture_active = isGraphCaptureActive();
        const bool sync_embedding_stage =
            debugEnv().validation.sync_each_stage ||
            debugEnv().validation.sync_after_embedding_stage;
        const bool allow_sync_embedding_stage = sync_embedding_stage && !capture_active;

        // =====================================================================
        // Step 3: Route by embedding table format
        // =====================================================================

        // --- Fast path: FP32 tensor already on GPU ---
        auto *embed_fp32 = dynamic_cast<const FP32Tensor *>(embed_table);
        if (embed_fp32 && embed_fp32->isOnGPU())
        {
            float *d_embed = const_cast<float *>(static_cast<const float *>(embed_fp32->gpu_data_ptr()));
            if (validate_gpu_ptrs &&
                !validatePointerForDevice(d_embed, dev, "EMBED_FP32", /*fail_on_query_error=*/true))
            {
                return false;
            }
            LOG_DEBUG("[ROCmEmbeddingKernelT] FP32 fast path: d_embed=" << static_cast<void *>(d_embed)
                                                                        << " num_tokens=" << num_tokens << " d_model=" << d_model);
            const int launch_vocab_size = explicit_vocab_range_ && local_vocab_size_ > 0
                                              ? local_vocab_size_
                                              : static_cast<int>(embed_fp32->rows());
            const int launch_vocab_offset = explicit_vocab_range_ ? vocab_offset_ : 0;
            err = hipOps_embedding_fp32(d_embed, d_token_ids, d_output, num_tokens, d_model,
                                        launch_vocab_size, launch_vocab_offset, stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] FP32 kernel failed: " << hipGetErrorString(err));
                return false;
            }

            if (allow_sync_embedding_stage)
            {
                hipError_t sync_err = hipStreamSynchronize(stream);
                if (sync_err != hipSuccess)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] FP32 embedding stream sync failed: "
                              << hipGetErrorString(sync_err));
                    ROCmBackend::dumpRecentPointerEvents(64);
                    return false;
                }
            }
            recordROCmGroupedEmbeddingCall(
                embed_table,
                num_tokens,
                d_model,
                dev,
                use_device_token_ids,
                "resident_fp32");
            return true;
        }

        // --- Native FP16/BF16 path: preserve loaded table precision on device ---
        const TensorType table_type = embed_table->native_type();
        if ((table_type == TensorType::FP16 || table_type == TensorType::BF16) &&
            embed_table->isOnGPU())
        {
            const auto *d_embed = static_cast<const uint16_t *>(embed_table->gpu_data_ptr());
            if (validate_gpu_ptrs &&
                !validatePointerForDevice(d_embed, dev, "EMBED_FLOAT16", /*fail_on_query_error=*/true))
            {
                return false;
            }

            const int launch_vocab_size = explicit_vocab_range_ && local_vocab_size_ > 0
                                              ? local_vocab_size_
                                              : static_cast<int>(embed_table->rows());
            const int launch_vocab_offset = explicit_vocab_range_ ? vocab_offset_ : 0;
            err = table_type == TensorType::FP16
                      ? hipOps_embedding_fp16_source_fp32(
                            d_embed, d_token_ids, d_output, num_tokens, d_model,
                            launch_vocab_size, launch_vocab_offset, stream)
                      : hipOps_embedding_bf16_source_fp32(
                            d_embed, d_token_ids, d_output, num_tokens, d_model,
                            launch_vocab_size, launch_vocab_offset, stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] Native " << tensorTypeName(table_type)
                                                           << " embedding launch failed: "
                                                           << hipGetErrorString(err));
                return false;
            }
            if (allow_sync_embedding_stage)
            {
                const hipError_t sync_err = hipStreamSynchronize(stream);
                if (sync_err != hipSuccess)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] Native floating embedding stream sync failed: "
                              << hipGetErrorString(sync_err));
                    return false;
                }
            }
            recordROCmGroupedEmbeddingCall(
                embed_table,
                num_tokens,
                d_model,
                dev,
                use_device_token_ids,
                table_type == TensorType::FP16 ? "resident_fp16" : "resident_bf16");
            return true;
        }

        // --- Quantized path: consume model-owned prepared EmbedQ8 weights ---
        if (dynamic_cast<const IINT8Unpackable *>(embed_table))
        {
            const DeviceId dev_id = DeviceId::rocm(dev);
            const PreparedEmbeddingHandle *prepared = prepared_embedding_handle_;
            const bool prepared_matches =
                prepared &&
                prepared->tensor == embed_table &&
                prepared->device_id == dev_id &&
                prepared->weights &&
                prepared->weights->device_id == dev_id &&
                prepared->weights->device_data &&
                prepared->weights->d_model == d_model &&
                prepared->weights->blocks_per_row > 0 &&
                prepared->weights->vocab_size > 0;
            if (!prepared_matches)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] Quantized GPU embedding requires matching prepared device weights: "
                          << "tensor=" << static_cast<const void *>(embed_table)
                          << " format=" << tensorTypeName(embed_table->native_type())
                          << " device=" << dev_id.to_string()
                          << " d_model=" << d_model);
                return false;
            }

            void *d_embed_q8 = prepared->weights->device_data;
            const size_t blocks_per_row = prepared->weights->blocks_per_row;
            const int vocab_offset = static_cast<int>(prepared->weights->vocab_offset);
            const int local_vocab_size = static_cast<int>(prepared->weights->vocab_size);
            // Validation readbacks require D2H plus stream synchronization, both
            // illegal inside HIP graph capture. The launch itself still consumes
            // the device token IDs and remains graph-capturable.
            if (use_device_token_ids &&
                debugEnv().validation.validate_buffers &&
                num_tokens > 0 &&
                !capture_active)
            {
                std::vector<int> sampled_tokens(static_cast<size_t>(num_tokens), -1);
                hipError_t token_copy_err = hipMemcpyAsync(
                    sampled_tokens.data(),
                    d_token_ids,
                    static_cast<size_t>(num_tokens) * sizeof(int),
                    hipMemcpyDeviceToHost,
                    stream);
                if (token_copy_err != hipSuccess)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] Failed to read device token IDs for validation: "
                              << hipGetErrorString(token_copy_err));
                    return false;
                }
                token_copy_err = hipStreamSynchronize(stream);
                if (token_copy_err != hipSuccess)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] Failed to synchronize device token validation: "
                              << hipGetErrorString(token_copy_err));
                    return false;
                }
                const bool zero_token_rows_allowed =
                    allow_out_of_range_token_ids_ ||
                    (mpi_ctx && mpi_ctx->world_size() > 1);
                for (int i = 0; i < num_tokens; ++i)
                {
                    const int token_id = sampled_tokens[static_cast<size_t>(i)];
                    const bool in_local_range =
                        token_id >= vocab_offset &&
                        token_id < vocab_offset + local_vocab_size;
                    if (!in_local_range && !zero_token_rows_allowed)
                    {
                        LOG_ERROR("[ROCmEmbeddingKernelT] Device-token embedding would zero single-device token="
                                  << token_id << " local_vocab_size=" << local_vocab_size
                                  << " vocab_offset=" << vocab_offset
                                  << " num_tokens=" << num_tokens);
                        return false;
                    }
                    LOG_DEBUG("[ROCmEmbeddingKernelT] Device-token embedding validation token="
                              << token_id << " local_vocab_size=" << local_vocab_size
                              << " vocab_offset=" << vocab_offset
                              << " in_local_range=" << (in_local_range ? 1 : 0));
                }
            }
            err = hipOps_embedding_q8(d_embed_q8, d_token_ids, d_output,
                                      num_tokens, d_model,
                                      static_cast<int>(blocks_per_row),
                                      local_vocab_size,
                                      vocab_offset,
                                      (validate_gpu_ptrs && dev == 0 && !capture_active) ? 1 : 0,
                                      stream);
            if (validate_gpu_ptrs)
            {
                hipError_t launch_err = hipPeekAtLastError();
                if (launch_err != hipSuccess)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] EmbedQ8 post-launch error: " << hipGetErrorString(launch_err)
                                                                                   << " dev=" << dev
                                                                                   << " stream=" << static_cast<void *>(stream)
                                                                                   << " d_embed_q8=" << d_embed_q8
                                                                                   << " d_token_ids=" << static_cast<void *>(d_token_ids)
                                                                                   << " d_output=" << static_cast<void *>(d_output));
                    ROCmBackend::dumpRecentPointerEvents(64);
                    return false;
                }
            }
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] EmbedQ8 kernel failed: " << hipGetErrorString(err));
                return false;
            }

            if (allow_sync_embedding_stage)
            {
                hipError_t sync_err = hipStreamSynchronize(stream);
                if (sync_err != hipSuccess)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] EmbedQ8 stream sync failed: "
                              << hipGetErrorString(sync_err));
                    ROCmBackend::dumpRecentPointerEvents(64);
                    return false;
                }
            }
            recordROCmGroupedEmbeddingCall(
                embed_table,
                num_tokens,
                d_model,
                dev,
                use_device_token_ids,
                "prepared_device_embed_q8");
            return true;
        }

        LOG_ERROR("[ROCmEmbeddingKernelT] Embedding table type "
                  << tensorTypeName(embed_table->native_type())
                  << " is not a resident floating table and has no prepared quantized representation");
        return false;
    }

    // =============================================================================
    // IWorkspaceConsumer Interface Implementation
    // =============================================================================

    WorkspaceRequirements ROCmEmbeddingKernelT::getWorkspaceRequirements(
        int m, int n, int k) const
    {
        (void)n; // Unused for embedding
        (void)k; // Persistent embedding weights are not graph workspace
        return embedding_workspace::requirements({.graph_rows = m});
    }

    void ROCmEmbeddingKernelT::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        if (!workspace)
        {
            std::lock_guard<std::mutex> lock(workspace_mutex_);
            workspace_ = nullptr;
            workspace_by_device_.clear();
            return;
        }

        int dev_key = device_idx_;
        const DeviceId ws_device = workspace->device();
        dev_key = ws_device.toKernelDeviceIndex();

        {
            std::lock_guard<std::mutex> lock(workspace_mutex_);
            workspace_ = workspace;
            workspace_by_device_[dev_key] = workspace;
        }
    }

    bool ROCmEmbeddingKernelT::hasWorkspace() const
    {
        int current_device = -1;
        DeviceWorkspaceManager *workspace = nullptr;

        if (hipGetDevice(&current_device) == hipSuccess && current_device >= 0)
        {
            std::lock_guard<std::mutex> lock(workspace_mutex_);
            auto it = workspace_by_device_.find(current_device);
            if (it != workspace_by_device_.end())
            {
                workspace = it->second;
            }
        }

        if (!workspace)
        {
            workspace = workspace_;
        }

        return workspace != nullptr && workspace->isAllocated();
    }

    DeviceWorkspaceManager *ROCmEmbeddingKernelT::getWorkspace() const
    {
        int current_device = -1;
        if (hipGetDevice(&current_device) == hipSuccess && current_device >= 0)
        {
            std::lock_guard<std::mutex> lock(workspace_mutex_);
            auto it = workspace_by_device_.find(current_device);
            if (it != workspace_by_device_.end())
            {
                return it->second;
            }
        }

        return workspace_;
    }

} // namespace llaminar2
