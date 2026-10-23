/**
 * @file FixedFloatingProjectionGroup.h
 * @brief Typed immutable address snapshots for captured custom floating GEMMs.
 *
 * Native launches copy at most eight exact matrix addresses by value. Arenas
 * and prepared weights retain physical ownership; no pointer-table publication
 * or device count readback is needed to recover these immutable addresses.
 * Row geometry/live authority and native16 encoding remain launch-owned policy.
 */
#pragma once
#include "kernels/common/FloatingPointGemmWorkspaceABI.h"
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_FLOAT_GROUP_HD __host__ __device__
#else
#define LLAMINAR_FLOAT_GROUP_HD
#endif

namespace llaminar2
{
/**
 * @brief Nonempty, bounded captured bindings with an explicit weight storage type.
 * @tparam Weight float for FP32 weights or uint16_t for native FP16/BF16 words.
 * @details Construction validates all active entries before submission. Copies
 * retain addresses, never host container lifetimes or mutable activation values.
 * Consumers keep their original common M/N/K and source-encoding policy.
 */
template<class Weight> class FixedFloatingProjectionGroup final
{
    static_assert(std::is_same_v<Weight,float> || std::is_same_v<Weight,std::uint16_t>);
public:
    /**
     * @brief Bind the first complete projection without reading device storage.
     * @param input Persistent FP32 activation matrix.
     * @param weight Persistent matrix in the declared native storage type.
     * @param output Persistent FP32 result matrix.
     * @throws std::invalid_argument If any active address is null.
     */
    [[nodiscard]] static FixedFloatingProjectionGroup single(
        const float *input,const Weight *weight,float *output)
    {
        requireAddresses(input,weight,output);
        return FixedFloatingProjectionGroup(input,weight,output);
    }
    /**
     * @brief Append one exact binding within the canonical capture cardinality.
     * @param input Persistent FP32 activation matrix.
     * @param weight Persistent native weight matrix.
     * @param output Persistent FP32 result matrix.
     * @throws std::invalid_argument If an address is null.
     * @throws std::length_error If all eight slots are already live.
     */
    void append(const float *input,const Weight *weight,float *output)
    {
        requireAddresses(input,weight,output);
        if (count_==int(floating_gemm_abi::kMaxBatchedProjections))
            throw std::length_error("Floating projection group exceeds its capture ABI");
        inputs_[count_]=input;weights_[count_]=weight;outputs_[count_]=output;++count_;
    }
    /** @brief Return the immutable number of live matrix bindings. @return Count in1..8. */
    LLAMINAR_FLOAT_GROUP_HD int size() const noexcept { return count_; }
    /** @brief Borrow FP32 A. @param index Admitted grid-z index. @return Original address. */
    LLAMINAR_FLOAT_GROUP_HD const float *input(int index) const noexcept { return inputs_[index]; }
    /** @brief Borrow typed B. @param index Admitted grid-z index. @return Original address. */
    LLAMINAR_FLOAT_GROUP_HD const Weight *weight(int index) const noexcept { return weights_[index]; }
    /** @brief Borrow FP32 C. @param index Admitted grid-z index. @return Original address. */
    LLAMINAR_FLOAT_GROUP_HD float *output(int index) const noexcept { return outputs_[index]; }
private:
    /**
     * @brief Reject incomplete active metadata before a producer can enqueue.
     * @param input Original FP32 activation address.
     * @param weight Original typed source address.
     * @param output Original FP32 result address.
     * @throws std::invalid_argument If any active address is null.
     */
    static void requireAddresses(const float *input,const Weight *weight,float *output)
    {
        if (!input || !weight || !output)
            throw std::invalid_argument("Floating projection group requires complete immutable addresses");
    }
    /**
     * @brief Only checked construction can create a live capture descriptor.
     * @param input Validated FP32 activation address.
     * @param weight Validated native source address.
     * @param output Validated FP32 result address.
     */
    FixedFloatingProjectionGroup(const float *input,const Weight *weight,float *output) noexcept
        : inputs_{input},weights_{weight},outputs_{output},count_(1) {}
    const float *inputs_[floating_gemm_abi::kMaxBatchedProjections]; ///< Original FP32 A bindings.
    const Weight *weights_[floating_gemm_abi::kMaxBatchedProjections]; ///< Exact prepared B bindings.
    float *outputs_[floating_gemm_abi::kMaxBatchedProjections]; ///< Original FP32 C bindings.
    int count_; ///< Active entries, never allocation capacity or device live rows.
};
/** @brief Captured FP32 A/B projections with FP32 output. */
using FixedFP32ProjectionGroup=FixedFloatingProjectionGroup<float>;
/** @brief Captured FP32 A/native16 B projections with FP32 output. */
using FixedFP32x16ProjectionGroup=FixedFloatingProjectionGroup<std::uint16_t>;
static_assert(std::is_trivially_copyable_v<FixedFP32ProjectionGroup>);
static_assert(std::is_trivially_copyable_v<FixedFP32x16ProjectionGroup>);
static_assert(!std::is_default_constructible_v<FixedFP32ProjectionGroup>);
static_assert(!std::is_default_constructible_v<FixedFP32x16ProjectionGroup>);
}
#undef LLAMINAR_FLOAT_GROUP_HD
