#pragma once

#include <cstdint>
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
            Int8Expanded,
            NativeVNNI,
        };

        struct CUDAPackedWeights
        {
            struct DeviceUpload
            {
                int8_t *d_int8_data = nullptr;
                float *d_scales = nullptr;
                int8_t *d_int8_data_tc_blocked = nullptr;
                uint8_t *d_native_vnni = nullptr;
                uint16_t *d_native_scales = nullptr;
                uint16_t *d_native_mins = nullptr;
                uint32_t *d_native_emins = nullptr;
            };

            CUDAPackedWeightFamily preferred_family = CUDAPackedWeightFamily::Int8Expanded;
            CUDAPackedWeightFamily active_family = CUDAPackedWeightFamily::Int8Expanded;

            // Int8Expanded fallback buffers remain populated even when NativeVNNI
            // is the active execution family.
            std::vector<int8_t> int8_data;
            std::vector<float> scales;

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

            int8_t *d_int8_data = nullptr;
            float *d_scales = nullptr;
            int8_t *d_int8_data_tc_blocked = nullptr;
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
                    int8_data = std::move(other.int8_data);
                    scales = std::move(other.scales);
                    native_vnni = std::move(other.native_vnni);
                    native_scales = std::move(other.native_scales);
                    native_mins = std::move(other.native_mins);
                    native_emins = std::move(other.native_emins);
                    native_codebook_id = other.native_codebook_id;
                    native_blocks_per_row = other.native_blocks_per_row;
                    K = other.K;
                    N = other.N;
                    device_uploads = std::move(other.device_uploads);
                    d_int8_data = other.d_int8_data;
                    d_scales = other.d_scales;
                    d_int8_data_tc_blocked = other.d_int8_data_tc_blocked;
                    d_native_vnni = other.d_native_vnni;
                    d_native_scales = other.d_native_scales;
                    d_native_mins = other.d_native_mins;
                    d_native_emins = other.d_native_emins;
                    cuda_device_id = other.cuda_device_id;
                    uploaded = other.uploaded;
                    source_tensor_ = other.source_tensor_;
                    rowmajor_ = other.rowmajor_;

                    other.d_int8_data = nullptr;
                    other.d_scales = nullptr;
                    other.d_int8_data_tc_blocked = nullptr;
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

    } // namespace cuda
} // namespace llaminar2