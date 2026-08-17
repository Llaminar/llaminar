/**
 * @file CPUResidualAddKernelT.h
 * @brief CPU implementation of residual add kernel
 * @author David Sanftenberg
 *
 * Template-specialized implementations of ITensorResidualAdd for CPU.
 */

#pragma once

#include "../../../execution/config/RuntimeConfig.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../CPUKernelBase.h"

#include <algorithm>
#include <cstddef>
#include <string>

namespace llaminar2
{

    namespace residual_add_detail
    {
        /**
         * @brief Largest flat span that is cheaper on the caller's SIMD lane.
         *
         * Entering even a bounded OpenMP team costs more than adding the
         * 8--64 KiB activation spans common to M=1..4 verifier calls.  The
         * arithmetic is element-independent, so a caller-thread SIMD loop is
         * both byte-equivalent and the economical grouped implementation for
         * these geometries.  An existing outer team is never collapsed: its
         * already-live workers continue through the ordinary workshare path.
         */
        inline constexpr size_t kCallerThreadElementLimit = 16u * 1024u;

        /**
         * @brief Return whether a standalone invocation should remain serial.
         *
         * A kernel entered by an existing OpenMP team must be reached by every
         * worker, because its `omp for` is a collective worksharing construct.
         * Consequently only a standalone caller may select the SIMD branch.
         */
        inline bool useCallerThreadSIMD(size_t num_elements)
        {
            return !omp_in_parallel() &&
                   num_elements <= kCallerThreadElementLimit;
        }

        /**
         * @brief Return the exact standalone team size for one flat span.
         *
         * A non-serial call always uses the configured process-wide team.
         * Sub-maximal teams are deliberately forbidden in the inference path:
         * libgomp may retire the omitted workers, forcing the next NativeVNNI
         * projection to recreate them at millisecond-scale cost.
         */
        inline int standaloneTeamSize(size_t num_elements)
        {
            return useCallerThreadSIMD(num_elements)
                       ? 1
                       : omp_get_max_threads();
        }

        /**
         * @brief Describe the launch policy selected for grouped telemetry.
         */
        inline const char *invocationPolicy(size_t num_elements)
        {
            if (omp_in_parallel())
                return "existing_team_flat_workshare";
            if (num_elements <= kCallerThreadElementLimit)
                return "caller_thread_simd";
            return "full_flat_workshare";
        }

        /**
         * @brief Derive the active verifier-row count from a flat element span.
         *
         * ResidualAdd's interface is intentionally flat because row boundaries
         * do not affect elementwise arithmetic. Production activation tensors
         * still expose their logical row width, allowing telemetry to recover
         * M from the active element count even when the backing tensor reserves
         * capacity for more rows.
         *
         * @param input Production input tensor carrying the logical row width.
         * @param num_elements Number of active values passed to the kernel.
         * @return M when the span is row-aligned, otherwise zero.
         */
        inline int activeRows(
            const TensorBase *input,
            size_t num_elements)
        {
            if (!input || input->cols() == 0 || num_elements % input->cols() != 0)
                return 0;
            return static_cast<int>(num_elements / input->cols());
        }

        /**
         * @brief Publish one economical CPU grouped residual-add invocation.
         *
         * A grouped residual add is one contiguous OpenMP workshare over the
         * complete M-row span. No row replay is needed or allowed. M=1 serial
         * witnesses are omitted so focused integration tests can require one
         * and only one grouped production record.
         */
        inline void recordCPUGroupedCall(
            const TensorBase *input,
            size_t num_elements)
        {
            const int rows = activeRows(input, num_elements);
            if (rows < 2)
                return;

            // Do not materialize strings or a tag map in ordinary inference.
            // PerfStats filters are collection gates, so the producer must
            // reject a disabled domain before constructing its payload.
            if (!PerfStatsCollector::isDomainEnabled("kernel"))
                return;

            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_residual_add_grouped_verifier_rows_calls",
                1.0,
                "verifier",
                "cpu",
                {{"tensor_format", tensorTypeName(input->native_type())},
                 {"verifier_rows", std::to_string(rows)},
                 {"cols", std::to_string(input->cols())},
                 {"active_elements", std::to_string(num_elements)},
                 {"invocation_policy", invocationPolicy(num_elements)},
                 {"standalone_team_threads",
                  std::to_string(standaloneTeamSize(num_elements))}});
        }
    } // namespace residual_add_detail

    // ==========================================================================
    // FP32 Specialization
    // ==========================================================================

    template <ActivationPrecision Precision>
    class CPUResidualAddKernelT;

    template <>
    class CPUResidualAddKernelT<ActivationPrecision::FP32> : public ITensorResidualAdd, public CPUKernelBase
    {
    public:
        CPUResidualAddKernelT() = default;
        ~CPUResidualAddKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1;
        }

        bool apply(
            const float *input, const float *residual, float *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)mpi_ctx;
            (void)device_idx;

            if (residual_add_detail::useCallerThreadSIMD(num_elements))
            {
#pragma omp simd
                for (size_t i = 0; i < num_elements; ++i)
                    output[i] = input[i] + residual[i];
                return true;
            }

            auto do_work = [&]()
            {
#pragma omp for schedule(static)
                for (size_t i = 0; i < num_elements; ++i)
                {
                    output[i] = input[i] + residual[i];
                }
            };
            OMP_WORKSHARE_REGION(do_work);

            return true;
        }

        bool apply_tensor(
            const TensorBase *input,
            const TensorBase *residual,
            TensorBase *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            KERNEL_PROFILE_SCOPE(KernelType::RESIDUAL_ADD);
            if (!input || !residual || !output)
                return false;
            if (input->native_type() != TensorType::FP32 ||
                residual->native_type() != TensorType::FP32 ||
                output->native_type() != TensorType::FP32)
                return false;

            const bool ok = apply(
                input->data(),
                residual->data(),
                output->mutable_data(),
                num_elements,
                mpi_ctx,
                device_idx);
            if (ok)
                residual_add_detail::recordCPUGroupedCall(input, num_elements);
            return ok;
        }
    };

    // ==========================================================================
    // BF16 Specialization
    // ==========================================================================

    template <>
    class CPUResidualAddKernelT<ActivationPrecision::BF16> : public ITensorResidualAdd, public CPUKernelBase
    {
    public:
        CPUResidualAddKernelT() = default;
        ~CPUResidualAddKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1;
        }

        bool apply(
            const float *input, const float *residual, float *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)input;
            (void)residual;
            (void)output;
            (void)num_elements;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // BF16 kernel doesn't handle FP32
        }

        bool apply_bf16(
            const uint16_t *input, const uint16_t *residual, uint16_t *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)mpi_ctx;
            (void)device_idx;

            if (residual_add_detail::useCallerThreadSIMD(num_elements))
            {
#pragma omp simd
                for (size_t i = 0; i < num_elements; ++i)
                {
                    const float in_f = simd::bf16_to_fp32(input[i]);
                    const float res_f = simd::bf16_to_fp32(residual[i]);
                    output[i] = simd::fp32_to_bf16(in_f + res_f);
                }
                return true;
            }

            auto do_work = [&]()
            {
#pragma omp for schedule(static)
                for (size_t i = 0; i < num_elements; ++i)
                {
                    float in_f = simd::bf16_to_fp32(input[i]);
                    float res_f = simd::bf16_to_fp32(residual[i]);
                    output[i] = simd::fp32_to_bf16(in_f + res_f);
                }
            };
            OMP_WORKSHARE_REGION(do_work);

            return true;
        }

        bool apply_tensor(
            const TensorBase *input,
            const TensorBase *residual,
            TensorBase *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            KERNEL_PROFILE_SCOPE(KernelType::RESIDUAL_ADD);
            if (!input || !residual || !output)
                return false;
            if (input->native_type() != TensorType::BF16 ||
                residual->native_type() != TensorType::BF16 ||
                output->native_type() != TensorType::BF16)
                return false;

            const bool ok = apply_bf16(
                static_cast<const uint16_t *>(input->raw_data()),
                static_cast<const uint16_t *>(residual->raw_data()),
                static_cast<uint16_t *>(output->raw_mutable_data()),
                num_elements,
                mpi_ctx,
                device_idx);
            if (ok)
                residual_add_detail::recordCPUGroupedCall(input, num_elements);
            return ok;
        }
    };

    // ==========================================================================
    // FP16 Specialization
    // ==========================================================================

    template <>
    class CPUResidualAddKernelT<ActivationPrecision::FP16> : public ITensorResidualAdd, public CPUKernelBase
    {
    public:
        CPUResidualAddKernelT() = default;
        ~CPUResidualAddKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1;
        }

        bool apply(
            const float *input, const float *residual, float *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)input;
            (void)residual;
            (void)output;
            (void)num_elements;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // FP16 kernel doesn't handle FP32
        }

        bool apply_fp16(
            const uint16_t *input, const uint16_t *residual, uint16_t *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)mpi_ctx;
            (void)device_idx;

            if (residual_add_detail::useCallerThreadSIMD(num_elements))
            {
#pragma omp simd
                for (size_t i = 0; i < num_elements; ++i)
                {
                    const float in_f = simd::fp16_to_fp32(input[i]);
                    const float res_f = simd::fp16_to_fp32(residual[i]);
                    output[i] = simd::fp32_to_fp16(in_f + res_f);
                }
                return true;
            }

            auto do_work = [&]()
            {
#pragma omp for schedule(static)
                for (size_t i = 0; i < num_elements; ++i)
                {
                    float in_f = simd::fp16_to_fp32(input[i]);
                    float res_f = simd::fp16_to_fp32(residual[i]);
                    output[i] = simd::fp32_to_fp16(in_f + res_f);
                }
            };
            OMP_WORKSHARE_REGION(do_work);

            return true;
        }

        bool apply_tensor(
            const TensorBase *input,
            const TensorBase *residual,
            TensorBase *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            KERNEL_PROFILE_SCOPE(KernelType::RESIDUAL_ADD);
            if (!input || !residual || !output)
                return false;
            if (input->native_type() != TensorType::FP16 ||
                residual->native_type() != TensorType::FP16 ||
                output->native_type() != TensorType::FP16)
                return false;

            const bool ok = apply_fp16(
                static_cast<const uint16_t *>(input->raw_data()),
                static_cast<const uint16_t *>(residual->raw_data()),
                static_cast<uint16_t *>(output->raw_mutable_data()),
                num_elements,
                mpi_ctx,
                device_idx);
            if (ok)
                residual_add_detail::recordCPUGroupedCall(input, num_elements);
            return ok;
        }
    };

} // namespace llaminar2
