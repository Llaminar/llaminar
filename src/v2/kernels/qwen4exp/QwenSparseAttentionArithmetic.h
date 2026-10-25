/**
 * @file QwenSparseAttentionArithmetic.h
 * @brief Setup-owned selected-attention arithmetic, independent of M and context.
 *
 * Graph construction freezes this policy before capture. The FP16 tile program
 * uses FP32 online totals on ROCm; it never substitutes
 * another program after failed admission. Defaults resolve once per admitted participant; explicit
 * FP32 remains the original control program. The separate immutable setup schedule resolves
 * to8/33 before capture; native arguments never carry an unresolved phase.
 * No native headers, environment reads or live state are needed here.
 */
#pragma once
#include "backends/DeviceId.h"
#include "tensors/TensorType.h"
#include <optional>
#include <vector>
#include <memory>
#include <algorithm>
#include <utility>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace llaminar2::qwen4exp
{
/** @brief Closed arithmetic programs; neither M nor live replay chooses a program. */
enum class QwenSparseAttentionArithmetic : std::uint8_t
{
    FixedFP32 = 0,              ///< Original FP32 partition arithmetic and ordered merge.
    TileFP16WithFP32Totals = 1, ///< 8-part FP16 tile products with FP32 online totals.
    TileFP16WithFP32Totals33 =
        2 ///< Resolved33-part FP16 program; phase selection stays outside native arguments.
};
/** @brief Original native/control initializer; installed serving defaults resolve at participant
 * admission. */
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
/** @brief Optional setup intent; absence is distinct from explicit FP32/Uniform. */
struct QwenSparseAttentionSetupRequest
{
    std::optional<QwenSparseAttentionArithmetic> arithmetic;
    std::optional<QwenSparseAttentionPhaseSchedule> schedule;
    bool operator==(const QwenSparseAttentionSetupRequest &) const = default;
};
/** @brief Concrete program sealed before planning; never an unresolved default. */
struct QwenSparseAttentionResolvedPolicy
{
    QwenSparseAttentionArithmetic arithmetic = QwenSparseAttentionArithmetic::FixedFP32;
    QwenSparseAttentionPhaseSchedule schedule = QwenSparseAttentionPhaseSchedule::Uniform;
    bool operator==(const QwenSparseAttentionResolvedPolicy &) const = default;
};
/**
 * @brief Validate a concrete copied policy without performing default resolution.
 * @param policy Already sealed setup pair, never a native per-phase result.
 * @param backend Actual attention participant.
 * @throws std::invalid_argument For unknown values, unsupported backends or conflicting pairs.
 */
inline void validateQwenSparseAttentionResolvedPolicy(QwenSparseAttentionResolvedPolicy policy,
                                                      DeviceType backend)
{
    validateQwenSparseAttentionArithmetic(policy.arithmetic, backend);
    validateQwenSparseAttentionPhaseSchedule(policy.schedule, backend);
    if (policy.arithmetic == QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33 ||
        (policy.schedule == QwenSparseAttentionPhaseSchedule::Prefill8Decode33 &&
         policy.arithmetic != QwenSparseAttentionArithmetic::TileFP16WithFP32Totals))
        throw std::invalid_argument("QSA admitted policy is unresolved or conflicting");
}
/** @brief Declared attention participant geometry, not expert placement or live rows. */
struct QwenSparseAttentionSetupGeometry
{
    DeviceType backend;
    TensorType key_storage;
    TensorType value_storage;
    int request_capacity;
    int head_dimension;
    int rotary_dimension;
};
/**
 * @brief Resolve one participant exactly once before its BOM is authored.
 * @param request Frozen optional startup choices, never a live environment view.
 * @param geometry Authenticated model and native storage geometry for this participant.
 * @return Concrete arithmetic and phase schedule. Explicit unsupported requests throw.
 * @throws std::invalid_argument For malformed geometry, unknown policies or conflicts.
 * @details Installed defaults select the existing ROCm FP16 split program only
 * in its declared single-request geometry. Other declared regimes retain their
 * original FP32 Uniform program. This is not recovery from failed admission.
 */
inline QwenSparseAttentionResolvedPolicy
resolveQwenSparseAttentionSetup(const QwenSparseAttentionSetupRequest &request,
                                const QwenSparseAttentionSetupGeometry &geometry)
{
    if (geometry.request_capacity <= 0 || geometry.head_dimension <= 0 ||
        geometry.rotary_dimension <= 0 || geometry.rotary_dimension > geometry.head_dimension)
        throw std::invalid_argument("QSA setup requires positive canonical participant geometry");
    if (request.arithmetic)
    {
        (void)qwenSparseAttentionArithmeticName(*request.arithmetic);
        if (*request.arithmetic == QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33)
            throw std::invalid_argument(
                "QSA setup cannot request a resolved Decode33 native program");
    }
    if (request.schedule)
        (void)qwenSparseAttentionPhaseScheduleName(*request.schedule);
    const bool native_half = geometry.backend == DeviceType::ROCm &&
                             geometry.key_storage == TensorType::FP16 &&
                             geometry.value_storage == TensorType::FP16 &&
                             geometry.head_dimension == 256 && geometry.rotary_dimension == 64;
    const bool split_geometry = native_half && geometry.request_capacity == 1;
    const auto arithmetic = request.arithmetic.value_or(
        split_geometry || request.schedule == QwenSparseAttentionPhaseSchedule::Prefill8Decode33
            ? QwenSparseAttentionArithmetic::TileFP16WithFP32Totals
            : QwenSparseAttentionArithmetic::FixedFP32);
    const auto schedule = request.schedule.value_or(
        arithmetic == QwenSparseAttentionArithmetic::TileFP16WithFP32Totals && split_geometry
            ? QwenSparseAttentionPhaseSchedule::Prefill8Decode33
            : QwenSparseAttentionPhaseSchedule::Uniform);
    validateQwenSparseAttentionArithmetic(arithmetic, geometry.backend);
    validateQwenSparseAttentionPhaseSchedule(schedule, geometry.backend);
    if (arithmetic != QwenSparseAttentionArithmetic::FixedFP32 && !native_half)
        throw std::invalid_argument(
            "QSA FP16 setup requires ROCm native FP16 KV and supported model geometry");
    if (schedule == QwenSparseAttentionPhaseSchedule::Prefill8Decode33 &&
        (!split_geometry || arithmetic != QwenSparseAttentionArithmetic::TileFP16WithFP32Totals))
        throw std::invalid_argument("QSA split setup requires FP16 and one declared request");
    return {arithmetic, schedule};
}
/** @brief Immutable projection of one admitted attention participant, without accounting. */
struct QwenSparseAttentionParticipantProgram
{
    const DeviceId device;
    const QwenSparseAttentionResolvedPolicy policy;
    const int first_layer;
    const int last_layer;
    /**
     * @brief Construct one exact immutable participant copied from admission.
     * @param owner Physical execution participant.
     * @param program Already resolved setup policy.
     * @param first Original first global layer, inclusive.
     * @param last Original last global layer, inclusive.
     * @throws std::invalid_argument For invalid identity or policy.
     */
    QwenSparseAttentionParticipantProgram(DeviceId owner,
                                          QwenSparseAttentionResolvedPolicy program,
                                          int first,
                                          int last)
        : device(owner), policy(program), first_layer(first), last_layer(last)
    {
        if (!device.is_valid() || first < 0 || last < first)
            throw std::invalid_argument("QSA admitted participant has invalid source identity");
        validateQwenSparseAttentionResolvedPolicy(policy, device.type);
    }
};
/**
 * @brief Publish checked policy copies directly from admitted device inputs.
 * @param participants Only actual attention graph owners, never expert-only endpoints.
 * @return Shared immutable setup metadata; it owns no allocation or byte ledger.
 * @throws std::invalid_argument For missing, duplicate, invalid or conflicting participants.
 */
inline std::shared_ptr<const std::vector<QwenSparseAttentionParticipantProgram>>
sealQwenSparseAttentionParticipants(std::vector<QwenSparseAttentionParticipantProgram> participants)
{
    if (participants.empty())
        throw std::invalid_argument("QSA admission has no attention participants");
    for (auto it = participants.begin(); it != participants.end(); ++it)
    {
        if (!it->device.is_valid())
            throw std::invalid_argument("QSA admission has an invalid device");
        validateQwenSparseAttentionResolvedPolicy(it->policy, it->device.type);
        if (std::any_of(participants.begin(),
                        it,
                        [&](const auto &old)
                        {
                            return old.device == it->device && old.first_layer <= it->last_layer &&
                                   it->first_layer <= old.last_layer;
                        }))
            throw std::invalid_argument("QSA admission duplicates an attention participant");
    }
    return std::make_shared<const std::vector<QwenSparseAttentionParticipantProgram>>(
        std::move(participants));
}
/**
 * @brief Borrow the actual admitted program; factories never resolve defaults.
 * @param participants Immutable projection retained by the admission owner.
 * @param device Exact participant that owns the attention graph.
 * @param first_layer First original global layer, inclusive.
 * @param last_layer Last original global layer, inclusive.
 * @return The previously resolved value, without environment or model state reads.
 * @throws std::invalid_argument For a missing or duplicate actual participant.
 */
inline QwenSparseAttentionResolvedPolicy admittedQwenSparseAttentionPolicy(
    const std::shared_ptr<const std::vector<QwenSparseAttentionParticipantProgram>> &participants,
    DeviceId device,
    int first_layer,
    int last_layer)
{
    if (!participants)
        throw std::invalid_argument("QSA factory has no admitted policy projection");
    const auto found = std::find_if(participants->begin(),
                                    participants->end(),
                                    [&](const auto &entry)
                                    {
                                        return entry.device == device &&
                                               entry.first_layer == first_layer &&
                                               entry.last_layer == last_layer;
                                    });
    if (found == participants->end())
        throw std::invalid_argument("QSA factory participant is absent from admission");
    if (std::count_if(participants->begin(),
                      participants->end(),
                      [&](const auto &entry)
                      {
                          return entry.device == device && entry.first_layer == first_layer &&
                                 entry.last_layer == last_layer;
                      }) != 1)
        throw std::invalid_argument("QSA factory participant is duplicated in admission");
    validateQwenSparseAttentionResolvedPolicy(found->policy, device.type);
    return found->policy;
}

} // namespace llaminar2::qwen4exp
