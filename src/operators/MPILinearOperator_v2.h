/**
 * @file MPILinearOperator_v2.h
 * @brief Simplified MPI-aware linear projection with FP32/BF16 activation support
 *
 * @section Design Principles
 * 1. Support both FP32 and BF16 intermediate activations
 * 2. Always accumulate in FP32 internally (numerical stability)
 * 3. Return output in same format as input (FP32→FP32, BF16→BF16)
 * 4. Dispatch to optimal GEMM backend based on activation type
 * 5. Remove all legacy cruft (Q8_0 streaming, slab cache, etc.)
 *
 * @section Contract
 * Inputs:
 *  - inputs[0]: Activations [seq_len, in_dim] (FP32 or BF16)
 *  - inputs[1]: Weights [out_dim, in_dim] (any quantized format or FP32)
 *  - inputs[2] (optional): Bias [out_dim] (FP32)
 * Outputs:
 *  - outputs[0]: Output [seq_len, out_dim] (matches input format)
 *
 * @section Distribution Strategy
 * - Weights: Row-wise partitioned across ranks ([local_out_dim, in_dim])
 * - Activations: Replicated on all ranks
 * - Output: Gathered from all ranks via MPI_Allgatherv
 *
 * @author David Sanftenberg
 * @date October 2025
 */

#pragma once

#include "../MpiKernelBase.h"
#include <memory>
#include <vector>

namespace llaminar
{

    class MPILinearOperator_v2 : public MPIOperatorBase
    {
    public:
        explicit MPILinearOperator_v2(MPI_Comm comm = MPI_COMM_WORLD);
        ~MPILinearOperator_v2() = default;

        // OperatorBase interface
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override;

    private:
        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;

        std::string getOperatorType() const override { return "MPILinearOperator_v2"; }
        size_t getExpectedInputCount() const override { return 2; } // input + weight (bias optional)
        size_t getExpectedOutputCount() const override { return 1; }

        /**
         * @brief Execute FP32 activation path
         */
        bool executeFP32(const std::shared_ptr<TensorBase> &input,
                         const std::shared_ptr<TensorBase> &weight,
                         const std::shared_ptr<TensorBase> &bias,
                         std::shared_ptr<TensorBase> &output);

        /**
         * @brief Execute BF16 activation path (accumulate in FP32, convert back to BF16)
         */
        bool executeBF16(const std::shared_ptr<TensorBase> &input,
                         const std::shared_ptr<TensorBase> &weight,
                         const std::shared_ptr<TensorBase> &bias,
                         std::shared_ptr<TensorBase> &output);

        /**
         * @brief Get or cache weight (quantized: global ref, FP32: distributed copy)
         */
        std::shared_ptr<TensorBase> getOrCacheWeight(const std::shared_ptr<TensorBase> &global_weight,
                                                     size_t output_size,
                                                     int local_output_size,
                                                     size_t input_size);

        /**
         * @brief Distribute FP32 weight rows across ranks (only for non-quantized weights)
         */
        void distributeWeightFP32(const std::shared_ptr<TensorBase> &global_weight,
                                  std::shared_ptr<TensorBase> &local_weight,
                                  size_t output_size);

        /**
         * @brief Distribute bias across ranks (matches weight distribution)
         */
        void distributeBias(const std::shared_ptr<TensorBase> &global_bias,
                            std::shared_ptr<TensorBase> &local_bias,
                            size_t output_size);

        /**
         * @brief Gather local outputs from all ranks
         */
        void gatherOutput(const float *local_output, float *global_output,
                          size_t seq_len, size_t local_output_size, size_t global_output_size);

        /**
         * @brief Add bias to output tensor (in-place)
         */
        void addBias(float *output, const float *bias,
                     size_t seq_len, size_t output_size);

        // Weight/bias distribution cache (avoids repeated memcpy)
        struct CacheKey
        {
            const void *ptr;
            size_t size;
            bool operator<(const CacheKey &other) const
            {
                if (ptr != other.ptr)
                    return ptr < other.ptr;
                return size < other.size;
            }
        };

        std::map<CacheKey, std::shared_ptr<TensorBase>> weight_cache_;
        std::map<CacheKey, std::shared_ptr<TensorBase>> bias_cache_;
    };

} // namespace llaminar
