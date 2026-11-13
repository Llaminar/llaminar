/**
 * @file Qwen4ExpPredictorPolicy.h
 * @brief Source-authenticated predictor attention and normalization metadata.
 *
 * This allocation-free policy travels with planning metadata. Dense exported
 * predictors never acquire QSA buffers or borrow the target's selected IDs.
 */
#pragma once
#include "../../execution/mtp/MTPShiftedCacheOrigin.h"
#include <cstdint>
namespace llaminar2
{
/** @brief Exact attention-state identity; dense attention is never inferred from absence in a sparse list. */
enum class Qwen4ExpLayerKind : uint8_t { GDN, QSA, Dense };
/** @brief Exact exported predictor attention, with no automatic substitution. */
enum class Qwen4ExpPredictorAttention : uint8_t { Dense, Sparse };
/** @brief GGUF predictor hidden normalization axis; gain storage remains flat. */
enum class Qwen4ExpPredictorHiddenNorm : uint8_t { PerBranch };
/** @brief Physical artifact convention authenticated during metadata admission. */
enum class Qwen4ExpPredictorSource : uint8_t { Inline, SharedGGUF };
/** @brief Exact source policy retained independently of the immutable target directory. */
struct Qwen4ExpPredictorMetadata
{
    uint32_t source_layer;
    Qwen4ExpPredictorAttention attention;
    Qwen4ExpPredictorHiddenNorm hidden_norm;
    Qwen4ExpPredictorSource source;
    /**
     * @brief Authenticate this policy's source convention and main/predictor boundary.
     * @param main_layers Exact target decoder layer count.
     * @return True only for the supported published inline or shared export conventions.
     */
    bool validForMainLayers(uint32_t main_layers) const noexcept
    {
        return source_layer == main_layers &&
            (attention == Qwen4ExpPredictorAttention::Dense || attention == Qwen4ExpPredictorAttention::Sparse) &&
            hidden_norm == Qwen4ExpPredictorHiddenNorm::PerBranch &&
            (source == Qwen4ExpPredictorSource::Inline || source == Qwen4ExpPredictorSource::SharedGGUF) &&
            (attention != Qwen4ExpPredictorAttention::Dense || source == Qwen4ExpPredictorSource::SharedGGUF);
    }
    /** @return Published QFN's visible zero-hidden first-token seed; this is not a selectable mode. */
    constexpr MTPShiftedCacheOrigin cacheOrigin() const noexcept
    { return MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero; }
    bool operator==(const Qwen4ExpPredictorMetadata &) const = default;
};
} // namespace llaminar2
