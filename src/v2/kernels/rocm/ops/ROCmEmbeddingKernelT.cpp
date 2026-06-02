/**
 * @file ROCmEmbeddingKernelT.cpp
 * @brief ROCm embedding kernel host-side implementation
 */

#include "ROCmEmbeddingKernelT.h"
#include "utils/Logger.h"
#include "utils/ROCmKernelProfiler.h"
#include "utils/DebugEnv.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../common/EmbedQ8Repack.h"
#include "../../common/PreparedEmbeddingWeights.h"
#include "../../KernelFactory.h"
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

        std::lock_guard<std::mutex> lock(canary_mutex_);
        for (auto &entry : canary_by_device_)
        {
            auto &buf = entry.second;
            if (!buf.base)
            {
                continue;
            }

            try
            {
                (void)HipDeviceGuard::setDevice(entry.first);
                (void)hipFree(buf.base);
            }
            catch (...)
            {
                // Best-effort cleanup in destructor.
            }

            buf = DebugCanaryBuffer{};
        }
    }

    void ROCmEmbeddingKernelT::setGPUStream(void *stream)
    {
        gpu_stream_ = stream;

        int current_device = -1;
        if (hipGetDevice(&current_device) == hipSuccess && current_device >= 0)
        {
            std::lock_guard<std::mutex> lock(stream_mutex_);
            stream_by_device_[current_device] = stream;
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

        return gpu_stream_ ? gpu_stream_ : (device_ctx_ ? device_ctx_->defaultStream() : nullptr);
    }

    void ROCmEmbeddingKernelT::setDynamicTokenIds(const int *token_ids, int num_tokens)
    {
        dynamic_params_active_ = false;
        dynamic_token_count_ = 0;

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

    void ROCmEmbeddingKernelT::resetDynamicState()
    {
        dynamic_params_active_ = false;
        dynamic_token_count_ = 0;
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
                if ((value < 0 || value >= vocab_size) && first_invalid_pos < 0)
                {
                    first_invalid_pos = i;
                    first_invalid_id = value;
                }
            }

            LOG_DEBUG("[ROCmEmbeddingKernelT] Host token stats: num_tokens=" << num_tokens
                                                                             << " vocab_size=" << vocab_size
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

        hipError_t err = hipOps_embedding_fp32(embed_data, token_ids, output, num_tokens, d_model, INT_MAX, 0, static_cast<hipStream_t>(getStream()));
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

        int *d_token_ids = static_cast<int *>(workspace->getBuffer(EmbeddingWorkspaceBuffers::TOKEN_IDS));
        if (!d_token_ids)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Workspace buffer '" << EmbeddingWorkspaceBuffers::TOKEN_IDS << "' not found");
            return false;
        }

        const DeviceId workspace_device = workspace->device();

        const bool validate_gpu_ptrs = debugEnv().validation.validate_gpu_ptrs;
        if (validate_gpu_ptrs && !validateTokenIdsHost(token_ids, num_tokens, static_cast<int>(embed_table->rows()), /*fail_on_invalid=*/true))
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
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipSuccess;
        // Verify preloaded data matches current request to prevent stale tokens
        // after clear_cache(). The kernel is cached in KernelFactory and
        // dynamic_params_active_ persists across graph rebuilds.
        // Also verify stream match: setDynamicTokenIds() may have run on a
        // different stream than the current gpu_stream_ if the graph capture
        // controller reassigned stage streams after updateDynamicParams().
        const bool token_ids_preloaded = dynamic_params_active_ && dynamic_token_count_ == num_tokens && preload_stream_ == getStream() && h_token_ids_ && std::memcmp(h_token_ids_, token_ids, static_cast<size_t>(num_tokens) * sizeof(int)) == 0;
        if (!token_ids_preloaded)
        {
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[ROCmEmbeddingKernelT::apply_tensor] token_ids not preloaded during graph capture — "
                          "updateDynamicParams must be called before capture/replay");
                return false;
            }
            dynamic_params_active_ = false;
            dynamic_token_count_ = 0;
            err = hipMemcpyAsync(d_token_ids, token_ids, token_bytes, hipMemcpyHostToDevice, stream);
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
            err = hipOps_embedding_fp32(d_embed, d_token_ids, d_output, num_tokens, d_model,
                                        static_cast<int>(embed_fp32->rows()), 0, stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] FP32 kernel failed: " << hipGetErrorString(err));
                return false;
            }

            if (sync_embedding_stage)
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
            return true;
        }

        // --- Quantized path: repack to EmbedQ8 via IINT8Unpackable ---
        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(embed_table);
        if (unpackable)
        {
            constexpr size_t kCanaryGuardBytes = 64 * 1024;
            constexpr unsigned char kCanaryPrePattern = 0xA5;
            constexpr unsigned char kCanaryPostPattern = 0x5A;

            // --- Preferred path: use model-owned PreparedWeightStore handle ---
            using namespace llaminar::v2::kernels;
            const DeviceId dev_id = DeviceId::rocm(dev);
            const PreparedEmbeddingHandle *prepared = nullptr;
            if (prepared_embedding_handle_ && prepared_embedding_handle_->device_id == dev_id)
                prepared = prepared_embedding_handle_;

            void *d_embed_q8 = nullptr;
            size_t blocks_per_row = 0;
            int vocab_offset = 0;
            int local_vocab_size = static_cast<int>(embed_table->rows());

            if (prepared && prepared->weights && prepared->weights->device_data)
            {
                // Fast path: GPU-resident prepared data from weight loading
                d_embed_q8 = prepared->weights->device_data;
                blocks_per_row = prepared->weights->blocks_per_row;
                vocab_offset = static_cast<int>(prepared->weights->vocab_offset);
                local_vocab_size = static_cast<int>(prepared->weights->vocab_size);
            }
            else
            {
                // Fallback: workspace-based lazy repack (for tests, CPU-only, etc.)
                d_embed_q8 = workspace ? workspace->getBuffer(EmbeddingWorkspaceBuffers::EMBED_TABLE) : nullptr;
                if (!d_embed_q8)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] No prepared embedding weights and no workspace EMBED_TABLE buffer");
                    return false;
                }

                // Validate device ownership of workspace pointers (critical for no-P2P systems).
                if (validate_gpu_ptrs)
                {
                    if (!validatePointerForDevice(d_embed_q8, dev, "EMBED_TABLE", /*fail_on_query_error=*/true))
                    {
                        return false;
                    }
                }

                bool needs_upload = false;
                {
                    std::lock_guard<std::mutex> lock(embed_cache_mutex_);
                    auto it = cached_embed_table_by_device_.find(dev);
                    const TensorBase *cached_for_device =
                        (it != cached_embed_table_by_device_.end()) ? it->second : cached_embed_table_;
                    needs_upload = (cached_for_device != embed_table);
                }
                if (needs_upload)
                {
                    auto repacked = repackEmbeddingToQ8(embed_table, d_model);

                    const size_t embed_buf_size = workspace->getBufferSize(EmbeddingWorkspaceBuffers::EMBED_TABLE);
                    if (embed_buf_size < repacked.byte_size)
                    {
                        LOG_ERROR("[ROCmEmbeddingKernelT] EMBED_TABLE workspace too small: have=" << embed_buf_size
                                                                                                  << " need=" << repacked.byte_size
                                                                                                  << " vocab=" << repacked.vocab_size
                                                                                                  << " d_model=" << d_model
                                                                                                  << " blocks_per_row=" << repacked.blocks_per_row
                                                                                                  << " workspace=" << static_cast<void *>(workspace)
                                                                                                  << " device=" << workspace_device.to_string());
                        return false;
                    }

                    err = hipMemcpyAsync(d_embed_q8, repacked.data.data(), repacked.byte_size,
                                         hipMemcpyHostToDevice,
                                         static_cast<hipStream_t>(getStream()));
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmEmbeddingKernelT] Failed to upload EmbedQ8 data: " << hipGetErrorString(err));
                        return false;
                    }

                    blocks_per_row = repacked.blocks_per_row;

                    {
                        std::lock_guard<std::mutex> lock(embed_cache_mutex_);
                        cached_embed_table_ = embed_table;
                        cached_embed_table_by_device_[dev] = embed_table;
                    }
                    LOG_DEBUG("[ROCmEmbeddingKernelT] Uploaded EmbedQ8 embedding (workspace fallback): "
                              << tensorTypeName(embed_table->native_type()) << " "
                              << repacked.vocab_size << "x" << d_model
                              << " → " << (repacked.byte_size / (1024 * 1024)) << " MB"
                              << " (" << repacked.blocks_per_row << " blocks/row)");
                }
                else
                {
                    blocks_per_row = (static_cast<size_t>(d_model) + 31) / 32;
                }
            }
            const size_t output_bytes = static_cast<size_t>(num_tokens) * static_cast<size_t>(d_model) * sizeof(float);
            const bool use_dev0_canary = validate_gpu_ptrs && (dev == 0) && !capture_active;
            float *kernel_output = d_output;
            void *canary_base = nullptr;

            if (use_dev0_canary)
            {
                std::lock_guard<std::mutex> lock(canary_mutex_);
                auto &canary = canary_by_device_[dev];
                const size_t required_total = output_bytes + (2 * kCanaryGuardBytes);

                if (!canary.base || canary.total_bytes < required_total || canary.payload_bytes < output_bytes)
                {
                    if (canary.base)
                    {
                        (void)hipFree(canary.base);
                        canary = DebugCanaryBuffer{};
                    }

                    void *new_base = nullptr;
                    err = hipMalloc(&new_base, required_total);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmEmbeddingKernelT] Failed to allocate EmbedQ8 canary buffer: "
                                  << hipGetErrorString(err) << " bytes=" << required_total << " dev=" << dev);
                        return false;
                    }

                    canary.base = new_base;
                    canary.guard_bytes = kCanaryGuardBytes;
                    canary.payload_bytes = output_bytes;
                    canary.total_bytes = required_total;
                    canary.payload = reinterpret_cast<float *>(static_cast<unsigned char *>(new_base) + kCanaryGuardBytes);
                }

                canary_base = canary.base;
                kernel_output = canary.payload;

                err = hipMemsetAsync(canary_base, kCanaryPrePattern, kCanaryGuardBytes, stream);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] Failed to init pre-guard: " << hipGetErrorString(err));
                    return false;
                }
                err = hipMemsetAsync(static_cast<unsigned char *>(canary_base) + kCanaryGuardBytes + output_bytes,
                                     kCanaryPostPattern,
                                     kCanaryGuardBytes,
                                     stream);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] Failed to init post-guard: " << hipGetErrorString(err));
                    return false;
                }
            }

            err = hipOps_embedding_q8(d_embed_q8, d_token_ids, kernel_output,
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

            if (use_dev0_canary)
            {
                hipError_t sync_err = hipStreamSynchronize(stream);
                if (sync_err != hipSuccess)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] EmbedQ8 canary sync failed: "
                              << hipGetErrorString(sync_err));
                    ROCmBackend::dumpRecentPointerEvents(64);
                    return false;
                }

                std::vector<unsigned char> pre(kCanaryGuardBytes);
                std::vector<unsigned char> post(kCanaryGuardBytes);

                err = hipMemcpyAsync(pre.data(), canary_base, kCanaryGuardBytes, hipMemcpyDeviceToHost,
                                     static_cast<hipStream_t>(getStream()));
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] Failed to read pre-guard: " << hipGetErrorString(err));
                    return false;
                }
                err = hipMemcpyAsync(post.data(), static_cast<unsigned char *>(canary_base) + kCanaryGuardBytes + output_bytes,
                                     kCanaryGuardBytes, hipMemcpyDeviceToHost,
                                     static_cast<hipStream_t>(getStream()));
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] Failed to read post-guard: " << hipGetErrorString(err));
                    return false;
                }
                // Synchronize to ensure canary data is available on host
                (void)hipStreamSynchronize(static_cast<hipStream_t>(getStream()));

                auto find_mismatch = [](const std::vector<unsigned char> &buf, unsigned char expected) -> size_t
                {
                    for (size_t i = 0; i < buf.size(); ++i)
                    {
                        if (buf[i] != expected)
                        {
                            return i;
                        }
                    }
                    return static_cast<size_t>(-1);
                };

                const size_t pre_bad = find_mismatch(pre, kCanaryPrePattern);
                const size_t post_bad = find_mismatch(post, kCanaryPostPattern);
                if (pre_bad != static_cast<size_t>(-1) || post_bad != static_cast<size_t>(-1))
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] EmbedQ8 canary corruption detected"
                              << " dev=" << dev
                              << " pre_bad=" << ((pre_bad == static_cast<size_t>(-1)) ? -1 : static_cast<int>(pre_bad))
                              << " post_bad=" << ((post_bad == static_cast<size_t>(-1)) ? -1 : static_cast<int>(post_bad))
                              << " d_output=" << static_cast<void *>(d_output)
                              << " kernel_output=" << static_cast<void *>(kernel_output));
                    ROCmBackend::dumpRecentPointerEvents(64);
                    return false;
                }

                err = hipMemcpyAsync(d_output, kernel_output, output_bytes, hipMemcpyDeviceToDevice, stream);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] Failed to copy canary payload to output: "
                              << hipGetErrorString(err));
                    return false;
                }
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
            else if (use_dev0_canary)
            {
                hipError_t sync_err = hipStreamSynchronize(stream);
                if (sync_err != hipSuccess)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] EmbedQ8 post-copy sync failed: "
                              << hipGetErrorString(sync_err));
                    ROCmBackend::dumpRecentPointerEvents(64);
                    return false;
                }
            }
            return true;
        }

        // No FP32 fallback — embedding table must be either FP32-on-GPU or IINT8Unpackable
        LOG_ERROR("[ROCmEmbeddingKernelT] Embedding table type "
                  << tensorTypeName(embed_table->native_type())
                  << " is not FP32-on-GPU and does not implement IINT8Unpackable");
        return false;
    }

    // =============================================================================
    // IWorkspaceConsumer Interface Implementation
    // =============================================================================

    WorkspaceRequirements ROCmEmbeddingKernelT::getWorkspaceRequirements(
        int m, int n, int k) const
    {
        (void)n; // Unused for embedding

        WorkspaceRequirements reqs;

        // Buffer 1: Token IDs [max_seq_len × sizeof(int)]
        // m is the maximum sequence length
        size_t token_ids_bytes = static_cast<size_t>(m) * sizeof(int);
        reqs.buffers.push_back({
            EmbeddingWorkspaceBuffers::TOKEN_IDS,
            token_ids_bytes,
            256, // Alignment for HIP
            true // Required
        });

        // Buffer 2: Embedding table temp [vocab_size × blocks_per_row × sizeof(EmbedQ8Block)]
        // Only needed when PreparedEmbeddingWeights are NOT available (test/fallback path).
        // When weights are prepared during loading, the prepared data lives in its own
        // GPU allocation and this workspace buffer is unused.
        if (!prepared_embedding_handle_)
        {
            constexpr size_t DEFAULT_VOCAB_SIZE = 151936;
            size_t vocab_size = (n > 0) ? static_cast<size_t>(n) : DEFAULT_VOCAB_SIZE;
            size_t d_model = (k > 0) ? static_cast<size_t>(k) : 896;
            size_t blocks_per_row = (d_model + 31) / 32;
            size_t embed_table_bytes = vocab_size * blocks_per_row * sizeof(EmbedQ8Block);
            reqs.buffers.push_back({EmbeddingWorkspaceBuffers::EMBED_TABLE,
                                    embed_table_bytes,
                                    256,
                                    true});
        }

        return reqs;
    }

    void ROCmEmbeddingKernelT::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        if (!workspace)
        {
            {
                std::lock_guard<std::mutex> lock(workspace_mutex_);
                workspace_ = nullptr;
                workspace_by_device_.clear();
            }
            {
                std::lock_guard<std::mutex> lock(embed_cache_mutex_);
                cached_embed_table_ = nullptr;
                cached_embed_table_by_device_.clear();
            }
            return;
        }

        int dev_key = device_idx_;
        const DeviceId ws_device = workspace->device();
        dev_key = ws_device.toKernelDeviceIndex();

        bool workspace_changed = false;
        {
            std::lock_guard<std::mutex> lock(workspace_mutex_);
            auto it = workspace_by_device_.find(dev_key);
            workspace_changed = (it == workspace_by_device_.end()) || (it->second != workspace);
            workspace_ = workspace;
            workspace_by_device_[dev_key] = workspace;
        }

        // Only invalidate embed cache when the workspace actually changes.
        // Re-binding the same workspace (e.g. on graph rebuild with cached buffers)
        // should not force a ~300ms embedding repack + upload.
        if (workspace_changed)
        {
            std::lock_guard<std::mutex> lock(embed_cache_mutex_);
            cached_embed_table_ = nullptr;
            cached_embed_table_by_device_[dev_key] = nullptr;
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
