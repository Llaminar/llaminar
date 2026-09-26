/**
 * @file MTPDepthPolicyTypes.h
 * @brief Dependency-free policy keys shared by admission and resident MTP control.
 *
 * These keys identify offline-trained economics, not live execution state.
 * Keeping them independent of RuntimeConfig lets CPU and GPU controllers use
 * the same learned rules without pulling host orchestration into a GPU kernel.
 */
#pragma once

namespace llaminar2
{
    /** @brief Verification law; greedy is the argmax sampling specialization. */
    enum class MTPVerifyMode { Greedy, SpeculativeSampling };

    /** @brief Coarse continuation backend used by the offline depth trainer. */
    enum class MTPDepthPolicyBackend { Any, CPU, CUDA, ROCm };

    /** @brief Dense and routed models have distinct grouped-verifier economics. */
    enum class MTPDepthPolicyModelClass { Any, Dense, MoE };
}
