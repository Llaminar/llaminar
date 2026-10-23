/**
 * @file QwenSparseAttentionArithmetic.h
 * @brief Setup-owned selected-attention arithmetic, independent of M and context.
 *
 * Graph construction freezes this policy before capture. The FP16 tile program
 * is an explicit ROCm experiment with FP32 online totals; it never substitutes
 * another program after failed admission. The existing fixed FP32 program stays
 * the default on every backend. No native headers or live state are needed here.
 */
#pragma once
#include "backends/DeviceType.h"
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace llaminar2::qwen4exp
{
/** @brief Closed arithmetic programs; neither M nor live replay chooses a program. */
enum class QwenSparseAttentionArithmetic : std::uint8_t {
    FixedFP32=0, ///< Original FP32 partition arithmetic and ordered merge.
    TileFP16WithFP32Totals=1 ///< Experimental FP16 tile products with FP32 online totals.
};
/** @brief Preserve the existing source arithmetic unless setup explicitly selects an experiment. */
inline constexpr auto kDefaultQwenSparseAttentionArithmetic=QwenSparseAttentionArithmetic::FixedFP32;
/**
 * @brief Serialize the unique setup and diagnostic spelling.
 * @param policy Immutable arithmetic copied into the native launch arguments.
 * @return fp32 or fp16_tile, without an alias or automatic selection.
 * @throws std::invalid_argument For an unknown enum value.
 */
inline constexpr std::string_view qwenSparseAttentionArithmeticName(QwenSparseAttentionArithmetic policy)
{
    switch(policy) {
    case QwenSparseAttentionArithmetic::FixedFP32:return "fp32";
    case QwenSparseAttentionArithmetic::TileFP16WithFP32Totals:return "fp16_tile";
    }
    throw std::invalid_argument("QSA attention arithmetic policy is unknown");
}
/**
 * @brief Parse only the explicit setup experiment vocabulary.
 * @param value LLAMINAR_QFN_SPARSE_ATTENTION value, read once by DebugEnv setup.
 * @return Exactly the requested arithmetic policy.
 * @throws std::invalid_argument For an empty, unknown or aliased value.
 */
inline constexpr QwenSparseAttentionArithmetic parseQwenSparseAttentionArithmetic(std::string_view value)
{
    if(value=="fp32")return QwenSparseAttentionArithmetic::FixedFP32;
    if(value=="fp16_tile")return QwenSparseAttentionArithmetic::TileFP16WithFP32Totals;
    throw std::invalid_argument("LLAMINAR_QFN_SPARSE_ATTENTION requires fp32 or fp16_tile");
}
/**
 * @brief Reject an arithmetic program unsupported by the actual attention participant.
 * @param policy Immutable explicitly selected arithmetic program.
 * @param backend Actual selected-attention backend; expert placement is independent.
 * @throws std::invalid_argument For an unknown backend/policy or non-ROCm FP16 experiment.
 * @details This static check performs no backend initialization or live-state read.
 */
inline void validateQwenSparseAttentionArithmetic(QwenSparseAttentionArithmetic policy,DeviceType backend)
{
    (void)qwenSparseAttentionArithmeticName(policy);
    if(backend!=DeviceType::CPU && backend!=DeviceType::CUDA && backend!=DeviceType::ROCm)
        throw std::invalid_argument("QSA attention requires an explicit CPU, CUDA or ROCm participant");
    if(policy==QwenSparseAttentionArithmetic::TileFP16WithFP32Totals && backend!=DeviceType::ROCm)
        throw std::invalid_argument("QSA fp16_tile is an explicit ROCm attention experiment; CPU/CUDA are unsupported");
}
}
