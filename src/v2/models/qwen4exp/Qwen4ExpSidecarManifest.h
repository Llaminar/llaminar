/**
 * @file Qwen4ExpSidecarManifest.h
 * @brief Exact native source/role contract for an explicitly selected shared MTP GGUF.
 *
 * Main and predictor metadata remain separate immutable authorities. A shared
 * predictor borrows the existing target embedding/head identities; it never
 * synthesizes a merged GGUF, changes compression metadata, or copies globals.
 */
#pragma once
#include "Qwen4ExpConfig.h"
#include "Qwen4ExpPredictorPolicy.h"
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
/** @brief Original artifact that physically owns a source tensor. */
enum class Qwen4ExpTensorSource : uint8_t { Main, Predictor };
/** @brief Native immutable coordinates; borrowed target globals retain their original names. */
struct Qwen4ExpSidecarTensor
{
    Qwen4ExpTensorSource source;
    GGUFTensorInfo tensor;
    bool consumed; ///< False only for exported indexer operands in a dense predictor.
};

/** @brief Validated metadata-only manifest for one external shared predictor. */
class Qwen4ExpSidecarManifest final
{
public:
    /**
     * @brief Authenticate both exact directories without opening tensor payloads.
     * @param main Original complete target split directory, with no inline predictor.
     * @param predictor Original separate shared MTP directory.
     * @throws std::invalid_argument For mismatched metadata, roles, shapes, native formats or tokenizer.
     */
    Qwen4ExpSidecarManifest(const GGUFModel &main, const GGUFModel &predictor);
    /** @return Original global predictor block index, independent of cache-local ordinals. */
    uint32_t sourceLayer() const noexcept { return source_layer_; }
    /** @return Attention selection encoded by the source compression entry. */
    Qwen4ExpPredictorAttention attention() const noexcept { return attention_; }
    /** @return Explicit GGUF hidden norm grouping used before the predictor FC. */
    Qwen4ExpPredictorHiddenNorm hiddenNorm() const noexcept { return Qwen4ExpPredictorHiddenNorm::PerBranch; }
    /** @return The same authenticated source policy in allocation-free planning form. */
    Qwen4ExpPredictorMetadata predictorMetadata() const noexcept
    { return {source_layer_, attention_, hiddenNorm(), Qwen4ExpPredictorSource::SharedGGUF}; }
    /** @return Sparse selector geometry only when the artifact actually declares it. */
    const std::optional<Qwen4ExpSparseAttentionGeometry> &sparseSelector() const noexcept { return sparse_; }
    /** @return Distinct compute/trunk/head widths authenticated against the target. */
    const Qwen4ExpActivationGeometry &activations() const noexcept { return activations_; }
    /** @return All native source descriptors, including dormant exported indexer operands. */
    const std::vector<Qwen4ExpSidecarTensor> &sources() const noexcept { return sources_; }
    /** @return Exact source/coordinates for one named role; absence is fatal. */
    const Qwen4ExpSidecarTensor &tensor(const std::string &name) const;
    /** @return Only graph-consumed names; dense predictors never request indexer preparation. */
    std::vector<std::string> consumedNames() const;
private:
    uint32_t source_layer_ = 0;
    Qwen4ExpPredictorAttention attention_ = Qwen4ExpPredictorAttention::Dense;
    Qwen4ExpActivationGeometry activations_{};
    std::optional<Qwen4ExpSparseAttentionGeometry> sparse_;
    std::vector<Qwen4ExpSidecarTensor> sources_;
};
} // namespace llaminar2
