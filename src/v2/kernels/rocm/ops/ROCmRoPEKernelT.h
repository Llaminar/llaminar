/**
 * @file ROCmRoPEKernelT.h
 * @brief ROCm RoPE kernel with typed precision support
 *
 * HIP/ROCm implementation of Rotary Positional Embeddings (RoPE) following
 * the same template pattern as CUDARoPEKernelT. Specialized for FP32,
 * BF16, and FP16 precisions.
 *
 * @author Llaminar Team
 * @date 2025-01-17
 */

#pragma once

#include "../../../execution/RuntimeConfig.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/BlockStructures.h"
#include <cstdint>

namespace llaminar2
{
    namespace rocm
    {

        // =========================================================================
        // Primary Template (Unsupported Precisions)
        // =========================================================================

        template <ActivationPrecision Precision>
        class ROCmRoPEKernelT
        {
            static_assert(Precision == ActivationPrecision::FP32 ||
                              Precision == ActivationPrecision::BF16 ||
                              Precision == ActivationPrecision::FP16,
                          "ROCmRoPEKernelT only supports FP32, BF16, and FP16 precisions");
        };

        // =========================================================================
        // FP32 Specialization
        // =========================================================================

        template <>
        class ROCmRoPEKernelT<ActivationPrecision::FP32> : public ITensorRoPE
        {
        public:
            using StorageType = float;

            explicit ROCmRoPEKernelT(int device_idx = 0, float rope_theta = 10000.0f)
                : device_idx_(device_idx), rope_theta_(rope_theta) {}
            ~ROCmRoPEKernelT() override = default;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            // ===== ITensorRoPE interface =====
            bool apply(
                float *data, float *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, bool interleaved,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            bool apply_bf16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            bool apply_fp16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            bool apply_q8_1(
                void *data, void *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            // ===== Typed API =====
            bool apply_typed(
                float *Q,
                float *K,
                const int *position_ids,
                int seq_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                float rope_theta,
                int device_idx = -1);

            static constexpr ActivationPrecision precision() { return ActivationPrecision::FP32; }
            static const char *precision_name() { return "FP32"; }

        private:
            int device_idx_;
            float rope_theta_;
        };

        // =========================================================================
        // BF16 Specialization
        // =========================================================================

        template <>
        class ROCmRoPEKernelT<ActivationPrecision::BF16> : public ITensorRoPE
        {
        public:
            using StorageType = uint16_t;

            explicit ROCmRoPEKernelT(int device_idx = 0, float rope_theta = 10000.0f)
                : device_idx_(device_idx), rope_theta_(rope_theta) {}
            ~ROCmRoPEKernelT() override = default;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            // ===== ITensorRoPE interface =====
            bool apply(
                float *data, float *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, bool interleaved,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)interleaved;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_bf16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override;

            bool apply_fp16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            bool apply_q8_1(
                void *data, void *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            // ===== Typed API =====
            bool apply_typed(
                uint16_t *Q,
                uint16_t *K,
                const int *position_ids,
                int seq_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                float rope_theta,
                int device_idx = -1);

            static constexpr ActivationPrecision precision() { return ActivationPrecision::BF16; }
            static const char *precision_name() { return "BF16"; }

        private:
            int device_idx_;
            float rope_theta_;
        };

        // =========================================================================
        // FP16 Specialization
        // =========================================================================

        template <>
        class ROCmRoPEKernelT<ActivationPrecision::FP16> : public ITensorRoPE
        {
        public:
            using StorageType = uint16_t;

            explicit ROCmRoPEKernelT(int device_idx = 0, float rope_theta = 10000.0f)
                : device_idx_(device_idx), rope_theta_(rope_theta) {}
            ~ROCmRoPEKernelT() override = default;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            // ===== ITensorRoPE interface =====
            bool apply(
                float *data, float *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, bool interleaved,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)interleaved;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_bf16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            bool apply_fp16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override;

            bool apply_q8_1(
                void *data, void *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            // ===== Typed API =====
            bool apply_typed(
                uint16_t *Q,
                uint16_t *K,
                const int *position_ids,
                int seq_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                float rope_theta,
                int device_idx = -1);

            static constexpr ActivationPrecision precision() { return ActivationPrecision::FP16; }
            static const char *precision_name() { return "FP16"; }

        private:
            int device_idx_;
            float rope_theta_;
        };

    } // namespace rocm
} // namespace llaminar2
