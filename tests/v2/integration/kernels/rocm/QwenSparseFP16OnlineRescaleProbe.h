/**
 * @file QwenSparseFP16OnlineRescaleProbe.h
 * @brief Private device-evaluator witnesses for the online-rescale experiment.
 *
 * These immutable input/output records are owned only by the focused native
 * fixture. They expose neither a production execution mode nor persistent
 * inference state. Device-loaded zero arguments keep the exp evaluator dynamic.
 */
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
namespace llaminar2::qwen4exp::online_rescale_trial
{
/** @brief One authored original online state and independent runtime exp operand. */
struct Input
{
    float maximum; ///< Original running maximum.
    float tile_maximum; ///< Original tile maximum.
    float denominator; ///< Original running denominator, including signed zero/NaN.
    float zero; ///< Independently loaded +0/-0; compiler cannot assume its value.
    float total; ///< Original FP32 numerator before update.
    float weighted; ///< Original widened half-PV tile numerator.
    float tile_denominator; ///< Original tile denominator before scaling.
};
/** @brief One side's scales and updates, written by exactly one thread. */
struct SideResult
{
    float previous; ///< Original/candidate previous scale.
    float tile; ///< Original/candidate tile scale.
    float total; ///< Common compiled numerator multiply/FMA result.
    float denominator; ///< Common compiled denominator multiply/FMA result.
};
/** @brief Two disjoint typed side records plus original evaluator metadata. */
struct Output
{
    float next_maximum; ///< Unchanged original fmax result, owned by side0.
    float zero_exponential; ///< Compiled runtime-zero expf, owned by side0.
    SideResult sides[2]; ///< Original0/candidate1, dynamically indexed after common arithmetic.
};
/**
 * @brief Launch a bounded test-only evaluator witness on its exact stream.
 * @param input Persistent disjoint device inputs,1..256 records.
 * @param output Persistent device observations with the same record capacity.
 * @param count Immutable admitted record count; never a downloaded live count.
 * @param stream Exact nondefault fixture stream.
 * @throws std::invalid_argument For invalid pointers/extents/alignment or stream.
 * @details Enqueues one diagnostic kernel; no allocation, copies or host queries.
 */
void evaluate(const Input *input,Output *output,int count,hipStream_t stream);
/** @return Borrowed compiled witness function for cold native-resource authentication. */
const void *evaluatorFunction() noexcept;
}
