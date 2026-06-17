/**
 * @file VerifierDecodeEquivalentGemmRows.h
 * @brief Helpers for running tiny MTP verifier batches as serial M=1 GEMVs.
 */

#pragma once

#include "../../../backends/BackendManager.h"
#include "../../../backends/IBackend.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <memory>

namespace llaminar2
{
    namespace verifier_gemm_rows
    {
        /**
         * @brief Allocate a tiny FP32 scratch tensor used to replay verifier rows.
         *
         * MTP grouped verifier graphs have prefill-shaped M=2..4 tensors, but
         * their acceptance contract is serial decode equivalence.  The scratch
         * tensors let stages feed one row at a time through the exact M=1 GEMV
         * interface while keeping the public stage output tensor unchanged.
         */
        inline std::shared_ptr<FP32Tensor> makeScratchFP32(
            size_t rows,
            size_t cols,
            DeviceId device,
            void *stream)
        {
            auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
            if (device.is_gpu())
                tensor->allocateOnDevice(device, stream);
            return tensor;
        }

        /**
         * @brief Ensure scratch exists before graph capture consumes it.
         *
         * GPU graph capture cannot contain allocations.  If a caller reaches a
         * capture with an unallocated scratch tensor, fail loudly so the owning
         * stage can be moved to a declared workspace rather than racing a lazy
         * allocation into a captured graph.
         */
        inline bool ensureScratchFP32(
            std::shared_ptr<FP32Tensor> &tensor,
            size_t rows,
            size_t cols,
            DeviceId device,
            void *stream,
            const char *stage_name,
            const char *scratch_name)
        {
            const std::vector<size_t> expected_shape{rows, cols};
            if (!tensor || tensor->shape() != expected_shape)
            {
                if (device.is_gpu() && isGraphCaptureActive())
                {
                    LOG_ERROR("[" << stage_name << "] Cannot allocate verifier scratch tensor '"
                                  << scratch_name << "' during graph capture");
                    return false;
                }
                tensor = makeScratchFP32(rows, cols, device, stream);
            }

            if (device.is_gpu() && !tensor->gpu_data_ptr())
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[" << stage_name << "] Verifier scratch tensor '"
                                  << scratch_name
                                  << "' was not device-resident before graph capture");
                    return false;
                }
                if (!tensor->allocateOnDevice(device, stream))
                {
                    LOG_ERROR("[" << stage_name << "] Failed to allocate verifier scratch tensor '"
                                  << scratch_name << "' on " << device.to_string());
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Mark a GPU tensor's device contents authoritative on a stream.
         */
        inline void markDeviceOutputWritten(TensorBase *tensor, DeviceId device, void *stream)
        {
            if (tensor && device.is_gpu())
                tensor->transitionToWithEvent(TensorCoherenceState::DEVICE_AUTHORITATIVE, device, stream);
        }

        /**
         * @brief Copy one FP32 row between device-resident tensors on an explicit stream.
         */
        inline bool copyFP32DeviceRow(
            TensorBase *dst,
            int dst_row,
            int dst_cols,
            const TensorBase *src,
            int src_row,
            int src_cols,
            int copy_cols,
            DeviceId device,
            void *stream,
            const char *stage_name,
            const char *label)
        {
            if (!stream)
            {
                LOG_ERROR("[" << stage_name << "] " << label
                              << " requires an explicit GPU stream");
                return false;
            }

            IBackend *backend = getBackendFor(device);
            if (!backend)
            {
                LOG_ERROR("[" << stage_name << "] No backend for " << device.to_string()
                              << " while copying " << label);
                return false;
            }

            auto *dst_ptr = static_cast<float *>(dst ? dst->gpu_data_ptr() : nullptr);
            const auto *src_ptr = static_cast<const float *>(src ? src->gpu_data_ptr() : nullptr);
            if (!dst_ptr || !src_ptr)
            {
                LOG_ERROR("[" << stage_name << "] Null device pointer while copying "
                              << label << " dst=" << static_cast<void *>(dst_ptr)
                              << " src=" << static_cast<const void *>(src_ptr));
                return false;
            }

            const size_t dst_offset =
                static_cast<size_t>(dst_row) * static_cast<size_t>(dst_cols);
            const size_t src_offset =
                static_cast<size_t>(src_row) * static_cast<size_t>(src_cols);
            const size_t bytes = static_cast<size_t>(copy_cols) * sizeof(float);
            const bool ok = backend->deviceCopyAsync(
                dst_ptr + dst_offset,
                src_ptr + src_offset,
                bytes,
                device.gpu_ordinal(),
                stream);
            if (!ok)
            {
                LOG_ERROR("[" << stage_name << "] Device row copy failed for "
                              << label << " bytes=" << bytes);
                return false;
            }
            return true;
        }

        /**
         * @brief Copy one host row between FP32-backed tensors.
         */
        inline bool copyHostRow(
            const TensorBase *src,
            int src_row,
            int src_cols,
            TensorBase *dst,
            int dst_row,
            int dst_cols,
            int copy_cols,
            const char *stage_name,
            const char *label)
        {
            const float *src_data = src ? src->data() : nullptr;
            float *dst_data = dst ? dst->mutable_data() : nullptr;
            if (!src_data || !dst_data)
            {
                LOG_ERROR("[" << stage_name << "] Null host pointer while copying "
                              << label);
                return false;
            }

            std::copy(src_data + static_cast<size_t>(src_row) * src_cols,
                      src_data + static_cast<size_t>(src_row) * src_cols + copy_cols,
                      dst_data + static_cast<size_t>(dst_row) * dst_cols);
            return true;
        }
    } // namespace verifier_gemm_rows
} // namespace llaminar2
