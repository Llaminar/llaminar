/**
 * @file QwenSparseAttentionArithmetic.h
 * @brief Setup-owned selected-attention arithmetic, independent of M and context.
 *
 * Graph construction freezes this policy before capture. The FP16 tile program
 * is an explicit ROCm experiment with FP32 online totals; it never substitutes
 * another program after failed admission. The existing fixed FP32 program stays
 * the default on every backend. The separate immutable setup schedule resolves
 * to8/33 before capture; native arguments never carry an unresolved phase.
 * No native headers, environment reads or live state are needed here.
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
    TileFP16WithFP32Totals=1, ///< Experimental8-part FP16 tile products with FP32 online totals.
    TileFP16WithFP32Totals33=2 ///< Resolved33-part FP16 program; phase selection stays outside native arguments.
};
/** @brief Preserve the existing source arithmetic unless setup explicitly selects an experiment. */
inline constexpr auto kDefaultQwenSparseAttentionArithmetic=QwenSparseAttentionArithmetic::FixedFP32;
/**
 * @brief Serialize the unique setup and diagnostic spelling.
 * @param policy Immutable arithmetic copied into the native launch arguments.
 * @return fp32, fp16_tile or resolved fp16_tile_33; the last is not a setup-parser spelling.
 * @throws std::invalid_argument For an unknown enum value.
 */
inline constexpr std::string_view qwenSparseAttentionArithmeticName(QwenSparseAttentionArithmetic policy)
{
    switch(policy) {
    case QwenSparseAttentionArithmetic::FixedFP32:return "fp32";
    case QwenSparseAttentionArithmetic::TileFP16WithFP32Totals:return "fp16_tile";
    case QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33:return "fp16_tile_33";
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
    if(policy!=QwenSparseAttentionArithmetic::FixedFP32 && backend!=DeviceType::ROCm)
        throw std::invalid_argument("QSA fp16_tile is an explicit ROCm attention experiment; CPU/CUDA are unsupported");
}

/** @brief Closed setup schedule, resolved to a concrete program before native submission. */
enum class QwenSparseAttentionPhaseSchedule : std::uint8_t {
    Uniform=0, ///< Preserve the existing explicitly selected arithmetic on both phases.
    Prefill8Decode33=1 ///< Explicit ROCm/native-FP16 diagnostic; no shape-derived phase.
};
/**
 * @brief Serialize the sole immutable schedule spelling.
 * @param schedule Typed setup scalar shared by planning and graph construction.
 * @return uniform or prefill8_decode33; never a native arithmetic name.
 * @throws std::invalid_argument For an unknown schedule.
 */
inline constexpr std::string_view qwenSparseAttentionPhaseScheduleName(QwenSparseAttentionPhaseSchedule schedule)
{
    switch(schedule) {
    case QwenSparseAttentionPhaseSchedule::Uniform:return "uniform";
    case QwenSparseAttentionPhaseSchedule::Prefill8Decode33:return "prefill8_decode33";
    }
    throw std::invalid_argument("QSA attention phase schedule is unknown");
}
/**
 * @brief Parse an explicit setup-only schedule without reading the environment.
 * @param value Exact text supplied by the immutable startup snapshot adapter.
 * @return Closed typed schedule; unset handling belongs to that adapter.
 * @throws std::invalid_argument For unknown, empty or aliased text.
 */
inline constexpr QwenSparseAttentionPhaseSchedule parseQwenSparseAttentionPhaseSchedule(std::string_view value)
{
    if(value=="uniform")return QwenSparseAttentionPhaseSchedule::Uniform;
    if(value=="prefill8_decode33")return QwenSparseAttentionPhaseSchedule::Prefill8Decode33;
    throw std::invalid_argument("QSA attention schedule requires uniform or prefill8_decode33");
}
/**
 * @brief Check setup schedule/backend without loading or querying a backend.
 * @param schedule Already parsed immutable startup scalar.
 * @param backend Actual participant backend, independent of expert placement.
 * @throws std::invalid_argument For unknown schedule/backend or foreign explicit33 diagnostic.
 */
inline void validateQwenSparseAttentionPhaseSchedule(QwenSparseAttentionPhaseSchedule schedule,DeviceType backend)
{
    (void)qwenSparseAttentionPhaseScheduleName(schedule);
    if(backend!=DeviceType::CPU && backend!=DeviceType::CUDA && backend!=DeviceType::ROCm)
        throw std::invalid_argument("QSA attention schedule requires CPU, CUDA or ROCm");
    if(schedule==QwenSparseAttentionPhaseSchedule::Prefill8Decode33 && backend!=DeviceType::ROCm)
        throw std::invalid_argument("QSA prefill8_decode33 is an explicit ROCm attention diagnostic");
}
/**
 * @brief Return the sealed native record count for one resolved program.
 * @param policy Concrete arithmetic; never an unresolved setup schedule or phase.
 * @return 8 for existing programs and33 for the explicit33 program.
 * @throws std::invalid_argument For an unknown arithmetic value.
 */
inline constexpr int qwenSparseAttentionPartialCount(QwenSparseAttentionArithmetic policy)
{
    switch(policy) {
    case QwenSparseAttentionArithmetic::FixedFP32:
    case QwenSparseAttentionArithmetic::TileFP16WithFP32Totals:return 8;
    case QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33:return 33;
    }
    throw std::invalid_argument("QSA attention partial count requires a resolved arithmetic program");
}
/**
 * @brief Resolve the existing canonical ordinary-token/retained-verifier Decode bound.
 * @param retainedMtpCandidateRows Existing admitted flattened retained target maximum;0 when absent.
 * @return max(1,R), without inferring live rows or phase from shape.
 * @throws std::invalid_argument For negative retained capacity.
 */
inline constexpr int qwenSparseAttentionDecodeRowCapacity(int retainedMtpCandidateRows)
{
    if(retainedMtpCandidateRows<0)throw std::invalid_argument("QSA retained Decode capacity is negative");
    return retainedMtpCandidateRows>0 ? retainedMtpCandidateRows : 1;
}
}
