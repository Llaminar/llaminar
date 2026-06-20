#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// Forward-declare opaque handle for row-major weight transpose
typedef struct CUDARowMajorWeights_ CUDARowMajorWeights;

namespace llaminar2
{
    class TensorBase;
    enum class TensorType;

    namespace cuda
    {
        enum class CUDAPackedWeightFamily
        {
            NativeVNNI,
        };

        struct CUDAPackedWeights
        {
            struct DeviceUpload
            {
                uint8_t *d_native_vnni = nullptr;
                uint16_t *d_native_scales = nullptr;
                uint16_t *d_native_mins = nullptr;
                uint32_t *d_native_emins = nullptr;
            };

            CUDAPackedWeightFamily preferred_family = CUDAPackedWeightFamily::NativeVNNI;
            CUDAPackedWeightFamily active_family = CUDAPackedWeightFamily::NativeVNNI;

            std::vector<uint8_t> native_vnni;
            std::vector<uint16_t> native_scales;
            std::vector<uint16_t> native_mins;
            std::vector<uint32_t> native_emins;
            uint8_t native_codebook_id = 0;
            uint32_t native_blocks_per_row = 0;

            int K = 0;
            int N = 0;

            mutable std::mutex upload_mutex;
            std::unordered_map<int, DeviceUpload> device_uploads;

            uint8_t *d_native_vnni = nullptr;
            uint16_t *d_native_scales = nullptr;
            uint16_t *d_native_mins = nullptr;
            uint32_t *d_native_emins = nullptr;
            int cuda_device_id = -1;
            bool uploaded = false;

            TensorBase *source_tensor_ = nullptr;

            // Row-major transpose for ROWPAR GEMV (lazily created, freed in destructor)
            CUDARowMajorWeights *rowmajor_ = nullptr;

            CUDAPackedWeights() = default;
            CUDAPackedWeights(const CUDAPackedWeights &) = delete;
            CUDAPackedWeights &operator=(const CUDAPackedWeights &) = delete;

            CUDAPackedWeights(CUDAPackedWeights &&other) noexcept
            {
                *this = std::move(other);
            }

            CUDAPackedWeights &operator=(CUDAPackedWeights &&other) noexcept
            {
                if (this != &other)
                {
                    std::scoped_lock guard(upload_mutex, other.upload_mutex);
                    preferred_family = other.preferred_family;
                    active_family = other.active_family;
                    native_vnni = std::move(other.native_vnni);
                    native_scales = std::move(other.native_scales);
                    native_mins = std::move(other.native_mins);
                    native_emins = std::move(other.native_emins);
                    native_codebook_id = other.native_codebook_id;
                    native_blocks_per_row = other.native_blocks_per_row;
                    K = other.K;
                    N = other.N;
                    device_uploads = std::move(other.device_uploads);
                    d_native_vnni = other.d_native_vnni;
                    d_native_scales = other.d_native_scales;
                    d_native_mins = other.d_native_mins;
                    d_native_emins = other.d_native_emins;
                    cuda_device_id = other.cuda_device_id;
                    uploaded = other.uploaded;
                    source_tensor_ = other.source_tensor_;
                    rowmajor_ = other.rowmajor_;

                    other.d_native_vnni = nullptr;
                    other.d_native_scales = nullptr;
                    other.d_native_mins = nullptr;
                    other.d_native_emins = nullptr;
                    other.cuda_device_id = -1;
                    other.uploaded = false;
                    other.K = 0;
                    other.N = 0;
                    other.native_codebook_id = 0;
                    other.native_blocks_per_row = 0;
                    other.source_tensor_ = nullptr;
                    other.rowmajor_ = nullptr;
                }
                return *this;
            }

            ~CUDAPackedWeights();
        };

        CUDAPackedWeightFamily classifyCUDAPackedWeightFamily(TensorType type);
        const char *cudaPackedWeightFamilyName(CUDAPackedWeightFamily family);
        bool packWeightsToCUDA(const TensorBase *tensor, CUDAPackedWeights &out);

        /**
         * @brief Batch-packed MoE expert weights for CUDA (single GPU allocation for all experts).
         *
         * Instead of packing 256 experts into 256 separate CUDAPackedWeights (each with its
         * own host allocation + H2D upload), this packs all experts into one contiguous slab
         * per array and uploads once. Per-expert GEMM kernels reference offset device pointers
         * into the shared allocation.
         *
         * This reduces:
         * - Host allocations: 256 → 1 per weight per layer
         * - H2D uploads: 256 → 1 per weight per layer
         * - KernelFactory cache entries: 256 → 0 (bypasses KernelFactory)
         */
        struct MoEBatchPackedWeightsCUDA
        {
            // Host buffers: all experts concatenated.
            // Expert e's data starts at offset e * <per_expert_size>.
            std::vector<uint8_t> all_vnni;
            std::vector<uint16_t> all_scales;
            std::vector<uint16_t> all_mins;   // Empty if symmetric format
            std::vector<uint32_t> all_emins;  // Empty if no emins

            int num_experts = 0;
            int rows_per_expert = 0;  // N per expert
            int K = 0;
            int blocks_per_row = 0;
            uint8_t codebook_id = 0;

            // Per-expert sizes for offset calculation
            size_t vnni_bytes_per_expert = 0;
            size_t scales_per_expert = 0;   // element count
            size_t mins_per_expert = 0;     // element count (0 if symmetric)
            size_t emins_per_expert = 0;    // element count (0 if no emins)

            // Device upload (one big allocation per array)
            struct DeviceUpload
            {
                uint8_t *d_vnni = nullptr;
                uint16_t *d_scales = nullptr;
                uint16_t *d_mins = nullptr;
                uint32_t *d_emins = nullptr;
            };

            std::mutex upload_mutex;
            std::unordered_map<int, DeviceUpload> device_uploads;

            MoEBatchPackedWeightsCUDA() = default;
            MoEBatchPackedWeightsCUDA(const MoEBatchPackedWeightsCUDA &) = delete;
            MoEBatchPackedWeightsCUDA &operator=(const MoEBatchPackedWeightsCUDA &) = delete;
            ~MoEBatchPackedWeightsCUDA();

            /// Upload all experts to a CUDA device (one allocation per array).
            bool uploadToDevice(int cuda_device_id);

            /// Get per-expert device pointers (offset into shared allocation).
            DeviceUpload getExpertDevicePointers(int cuda_device_id, int expert_id) const;

            /// Free host buffers after upload to reclaim memory.
            void freeHostBuffers();
        };

        /**
         * @brief Pack all MoE experts from 2D expert views into a single batch.
         *
         * @param expert_views  Per-expert 2D tensor views (from extractExpertViews)
         * @param num_experts   Number of experts
         * @param rows_per_expert  Rows (N) per expert
         * @return Shared pointer to batch-packed weights, or nullptr on failure.
         */
        std::shared_ptr<MoEBatchPackedWeightsCUDA> packMoEExpertsCUDA(
            const std::vector<std::shared_ptr<TensorBase>> &expert_views,
            int num_experts, int rows_per_expert);

    } // namespace cuda
} // namespace llaminar2