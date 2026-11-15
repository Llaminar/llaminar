/**
 * @file ROCmNativeQuantisedVerifierLaunch.h
 * @brief Shared grouped operand ABI and native M16 verifier submission contract.
 *
 * Row descriptors borrow original device publications. Physical strides and
 * canonical serial-M1 partition counts survive every dot/WMMA selection.
 * Candidate scopes affect only subsequent construction on one host worker;
 * they own no inference state and are independent of prefill candidate scopes.
 */
#pragma once
#include "kernels/common/ProjectionRowWork.h"
#include "kernels/common/CapturedVerifierExtent.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include <cstdint>
#include <optional>

inline constexpr int NATIVE_QUANTISED_SMALL_M_MAX_BATCH = 8;

/**
 * @brief Per-launch projection inventory with one borrowed row owner.
 *
 * Each physical scratch tile copies this descriptor, rebases only its outputs,
 * and slices rows in the original controller coordinate system. No device
 * pointer is rebound during replay and no projection maintains its own count.
 */
struct NativeQuantisedSmallMBatchedArgs
{
    int num_projections = 0;
    int total_grid_n = 0;
    int N[NATIVE_QUANTISED_SMALL_M_MAX_BATCH] = {};
    int grid_offsets[NATIVE_QUANTISED_SMALL_M_MAX_BATCH + 1] = {};
    const uint8_t* payload[NATIVE_QUANTISED_SMALL_M_MAX_BATCH] = {};
    const uint16_t* scales[NATIVE_QUANTISED_SMALL_M_MAX_BATCH] = {};
    const uint16_t* mins[NATIVE_QUANTISED_SMALL_M_MAX_BATCH] = {};
    const uint32_t* emins[NATIVE_QUANTISED_SMALL_M_MAX_BATCH] = {};
    const float* bias[NATIVE_QUANTISED_SMALL_M_MAX_BATCH] = {};
    float* output[NATIVE_QUANTISED_SMALL_M_MAX_BATCH] = {};
    float* partial[NATIVE_QUANTISED_SMALL_M_MAX_BATCH] = {};
    uint8_t codebook[NATIVE_QUANTISED_SMALL_M_MAX_BATCH] = {};
    // Preserve the aligned projection planes at the front of the launch ABI.
    // Row admission dies before decoder dispatch, unlike those live operands.
    llaminar2::DeviceRowRange rows; ///< Original prefix/grid geometry.
    llaminar2::ProjectionRowWork work; ///< Independent original-row arithmetic admission.
};

namespace llaminar2::rocm {
/** @brief Original complete public operations covered by the verifier economy panel. */
enum class NativeQuantisedVerifierOperationKind { Projection, SwiGLUDown };
/** @brief Immutable original operation; a singleton policy subgroup cannot mint one. */
struct NativeQuantisedVerifierOperation {
    NativeQuantisedVerifierOperationKind kind; ///< Public operation before any grouping.
    int device, m, n, k; ///< Original participant and full physical shape.
    const uint8_t *weights; ///< Original prepared payload, not an incidental same-shape leaf.
    NativeQuantisedSourceIdentity source; ///< Complete source provenance, including superblock identity.
};
/** @brief Scope a complete public operation; an empty binding hides any outer admission. */
class ScopedNativeQuantisedVerifierOperation final {
public:
    /** @param operation Validated unsliced, unbiased public operation, or unmeasured work. */
    explicit ScopedNativeQuantisedVerifierOperation(std::optional<NativeQuantisedVerifierOperation> operation) noexcept;
    /** @brief Restore the exact enclosing worker scope. */
    ~ScopedNativeQuantisedVerifierOperation();
    ScopedNativeQuantisedVerifierOperation(const ScopedNativeQuantisedVerifierOperation&) = delete;
    ScopedNativeQuantisedVerifierOperation& operator=(const ScopedNativeQuantisedVerifierOperation&) = delete;
    /** @return Original operation on this construction worker, if measured. */
    [[nodiscard]] static std::optional<NativeQuantisedVerifierOperation> current() noexcept;
private:
    std::optional<NativeQuantisedVerifierOperation> operation_; ///< Immutable construction binding only.
    const ScopedNativeQuantisedVerifierOperation *previous_; ///< Exact enclosing scope.
    static thread_local const ScopedNativeQuantisedVerifierOperation *active_; ///< No replay-time host policy.
};
/** @brief Generated serial policy qualification, excluding diagnostic tuning overrides. */
enum class NativeQuantisedVerifierPolicyMatch { Unmeasured, CanonicalSerial };
/** @brief Actual producer arguments before physical scratch tiling or row slicing. */
struct NativeQuantisedVerifierWork {
    DeviceRowRange rows; ///< Original count binding and complete physical geometry.
    const uint8_t *weights; ///< First physical payload; only original single projections qualify.
    int n, k, projections; ///< Actual producer dimensions and bundle population.
    uint8_t physical_codebook; ///< Actual decoder identity, distinct from source provenance.
    NativeQuantisedVerifierPolicyMatch policy; ///< Exact canonical M1 partition and launch policy.
};
/** @brief First-class grouped implementations; Automatic retains qualified selection. */
enum class NativeQuantisedVerifierCandidate { Automatic, Dot4, WmmaM16N64 };
/** @brief Construction-only selection bound to one authenticated device. */
struct NativeQuantisedVerifierSelection {
    NativeQuantisedVerifierCandidate candidate; ///< Exact implementation identity.
    int device; ///< Admitted native device, never a live scheduling decision.
};
/** @brief Return this worker's immutable construction override, if bound. */
[[nodiscard]] std::optional<NativeQuantisedVerifierSelection> nativeQuantisedVerifierSelection() noexcept;
/**
 * @brief Bind one candidate before graph construction and restore the prior scope.
 * @details Native WMMA requires actual gfx1100; unsupported forcing throws.
 */
class ScopedNativeQuantisedVerifierCandidate final {
public:
    /** @param candidate Reviewed candidate. @param device Actual participant ordinal. */
    ScopedNativeQuantisedVerifierCandidate(NativeQuantisedVerifierCandidate candidate, int device);
    /** @brief Restore this worker's enclosing construction selection. */
    ~ScopedNativeQuantisedVerifierCandidate();
    ScopedNativeQuantisedVerifierCandidate(const ScopedNativeQuantisedVerifierCandidate&)=delete;
    ScopedNativeQuantisedVerifierCandidate& operator=(const ScopedNativeQuantisedVerifierCandidate&)=delete;
private:
    std::optional<NativeQuantisedVerifierSelection> previous_; ///< Nested scope restoration only.
    const ScopedNativeQuantisedVerifierCandidate *previous_scope_; ///< Exact LIFO owner, never device state.
};
/**
 * @brief Resolve the selected native producer without reading a device count.
 * @param device Actual participant ordinal. @param decode_equivalent Serial-M1 arithmetic mode.
 * @param work Actual original producer binding, before row slicing.
 * @return Explicit diagnostic candidate or the measured exact-hosted producer.
 * @throws std::invalid_argument For a mismatched device or arithmetic phase.
 */
[[nodiscard]] NativeQuantisedVerifierCandidate nativeQuantisedVerifierProducer(
    int device, bool decode_equivalent, const NativeQuantisedVerifierWork &work);
/** @brief Direct standalone and fused output differ in their required final +0/bias. */
enum class VerifierWmmaPublication { Standalone, Batched, OrderedPartial };
/** @brief Complete borrowed producer arguments; the caller retains reducer ownership. */
struct NativeQuantisedVerifierWmmaArguments {
    const int8_t *activations; ///< Physical row-major quantized input.
    const float *activation_scales; ///< Physical per-K32 activation scales.
    const int32_t *activation_sums; ///< Original optional whole-block sum plane; fused callers require it.
    NativeQuantisedSmallMBatchedArgs projections; ///< Original bundle and row authority.
    int device; ///< Exact participant owning the authenticated nondefault stream.
    int k; ///< Physical contraction width, multiple of 32.
    int partitions; ///< Canonical serial-M1 split count, including empty tail planes.
    int row_workers; ///< Fixed bounded worker grid; cursors visit original live intervals.
    bool has_request_work; ///< Compile-time ABI choice authenticated before capture.
    VerifierWmmaPublication publication; ///< Exact destination and bias arithmetic.
};
/**
 * @brief Submit one native packet preserving the original row and serial arithmetic contracts.
 * @param args Complete persistent operands, physical strides and serial policy.
 * @param stream Exact nondefault producer stream.
 * @throws std::invalid_argument For incomplete operands or geometry.
 */
void submitNativeQuantisedVerifierWmma(const NativeQuantisedVerifierWmmaArguments &args, void *stream);
/**
 * @brief Publish this TU's immutable codebook grids during existing setup.
 * @param device Actual native ordinal. @param iq3s IQ3_S table. @param iq3xxs IQ3_XXS table.
 * @param iq2s IQ2_S table. @param iq2xs IQ2_XS table. @param iq2xxs IQ2_XXS table.
 * @param iq1s Shared IQ1 table. @return True after all tables are published.
 */
bool initializeNativeQuantisedVerifierWmmaGrids(int device, const void *iq3s,
    const void *iq3xxs, const void *iq2s, const void *iq2xs, const void *iq2xxs, const void *iq1s);
}
