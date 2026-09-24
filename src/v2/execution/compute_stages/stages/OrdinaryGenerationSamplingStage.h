/**
 * @file OrdinaryGenerationSamplingStage.h
 * @brief Captured, non-speculative sampling and response/frontier publication.
 *
 * The graph's terminal owner binds fresh full-vocabulary logits and existing
 * arena scratch. Position, stop policy, response and continuation stay on device;
 * no host token, mutable host counter, speculative verifier or sidecar is needed.
 * Masks precede this stage. Optional history penalties consume the same count
 * rows that its fused response publication advances exactly once per token.
 */
#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../kernels/common/SamplingMath.h"

#include <variant>

namespace llaminar2
{
struct MTPGreedyPenaltyPolicy;
class IBackend;

/**
 * @brief One captured sampling boundary for ordinary prefill or decode logits.
 *
 * The parent owns admission and suppresses this stage after completion. Scratch
 * sample outputs are deliberately distinct from the live frontier: an admitted
 * sampler cannot overwrite continuation before publication validates the commit.
 * All bindings are borrowed; BufferArena/PMA remains the sole allocation owner.
 */
class OrdinaryGenerationSamplingStage final : public IComputeStage
{
public:
    /** @brief Deterministic lowest-token argmax, with no random-draw state. */
    struct Greedy
    {
        /** @return All greedy policies have the same arithmetic identity. */
        bool operator==(const Greedy &) const = default;
    };

    /** @brief Immutable sampling law; each draw is keyed by live logical position. */
    struct Stochastic
    {
        int top_k = 0; ///< Positive compact support, at most the supported sampler capacity.
        float top_p = 1.0F; ///< Nucleus mass in (0,1].
        float temperature = 1.0F; ///< Strictly positive, finite logit temperature.
        const uint64_t *seeds = nullptr; ///< One admitted device seed per request; bytes may change on reset.
        /** @return Exact launch policy, including every request's random stream. */
        bool operator==(const Stochastic &) const = default;
    };

    /** @brief Persistent sampler scratch reused sequentially across request rows. */
    struct Workspace
    {
        float *values = nullptr; ///< Compact probabilities, or greedy maximum value.
        int32_t *indices = nullptr; ///< Compact token IDs for stochastic sampling.
        int capacity = 0; ///< Entries available in both compact rows.
        float *partial_values = nullptr; ///< Backend top-k/argmax reduction workspace.
        int32_t *partial_indices = nullptr; ///< Matching reduction indices.
        int partial_capacity = 0; ///< Entries available in each reduction row.
        /** @return Every persistent scratch address and capacity embedded in capture. */
        bool operator==(const Workspace &) const = default;
    };

    /** @brief Complete immutable stage binding; request bytes are never capture identity. */
    struct Params
    {
        STAGE_PARAMS_COMMON_FIELDS;
        IBackend *backend = nullptr; ///< Backend of the explicit owning GPU.
        float *logits = nullptr; ///< Fresh full-vocabulary rows; stochastic penalties consume them in place.
        int vocab_size = 0;
        int logits_row_stride = 0;
        std::variant<Greedy, Stochastic> policy = Greedy{};
        Workspace workspace;
        sampling_math::OrdinaryGenerationPublication publication;
        const MTPGreedyPenaltyPolicy *penalties = nullptr; ///< Optional shared request policy; null explicitly admits no penalty transform.
        /** @return Complete pointer, geometry and sampling-law identity. */
        bool operator==(const Params &) const = default;
    };
    static_assert(StageParamsRequired<Params>);

    /** @brief Freeze valid bindings; malformed policies fail before any kernel can launch.
     *  @throws std::invalid_argument for incomplete or inconsistent captured bindings. */
    explicit OrdinaryGenerationSamplingStage(Params params);
    /** @brief Enqueue sampler and atomic logical publication on the exact graph stream. */
    bool execute(IDeviceContext *ctx) override;
    /** @return Dedicated generation-boundary stage identity. */
    ComputeStageType type() const override { return ComputeStageType::ORDINARY_GENERATION_SAMPLING; }
    /** @return Stable diagnostic name independent of request data. */
    std::string name() const override { return "ordinary_generation_sampling"; }
    /** @return CUDA and HIP share the same controller and seeded arithmetic. */
    bool supportsBackend(ComputeBackendType backend) const override;
    /** @return All work is enqueue-only and uses pre-bound persistent memory. */
    bool isGraphCapturable() const override { return true; }
    /** @return This participant-local stage contains no implicit rank coordination. */
    bool isCollectiveStage() const override { return false; }
    /** @return No automatic transfers: producers already publish exact resident bindings. */
    CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
    /** @return Semantic arena inputs/outputs consumed by graph validation. */
    StageBufferContract bufferContract() const override;
    /** @return Whether an existing capture embeds exactly these immutable bindings. */
    bool hasSameCaptureIdentity(const Params &other) const noexcept { return params_ == other; }

protected:
    /** @return Geometry/policy diagnostics only; never download live device control. */
    StageDumpInfo buildDumpInfoImpl() const override;

private:
    /** @brief Reject incomplete geometry, non-finite policies and mutable-output aliasing. */
    void validate() const;
    const Params params_; ///< Stable until every graph embedding this stage is retired.
};
} // namespace llaminar2
