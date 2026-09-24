/**
 * @file CUDACanonicalKpartFold.h
 * @brief Typed launch identity for an ascending CTA-local K-partition fold.
 *
 * A fold changes physical publication, never the serial arithmetic partition
 * tree. The caller supplies an already resolved canonical partition count.
 * Construction proves the reduction and native thread-capacity constraints;
 * the immutable plan can then be retained as part of captured launch identity.
 * This candidate does not select itself or change the generated decode policy.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace llaminar2
{
/** @brief Compiled column geometries, independent of arithmetic partition count. */
enum class CUDACanonicalKpartColumns : uint8_t
{
    Sixteen = 16,
    ThirtyTwo = 32,
};

/** @brief Immutable, device-free proof of one legal CTA-local fold geometry. */
class CUDACanonicalKpartFoldPlan final
{
public:
    /**
     * @brief Admit a compiled geometry without rounding or replacing partitions.
     * @param columns Physical output-column ownership per CTA.
     * @param partitions Exact canonical K-partition count from the serial policy.
     * @param reduction_elements Logical K, a positive multiple of 32 elements.
     * @return A valid immutable plan, or no value for an unsupported geometry.
     */
    [[nodiscard]] static constexpr std::optional<CUDACanonicalKpartFoldPlan> create(
        CUDACanonicalKpartColumns columns, int partitions, int reduction_elements) noexcept
    {
        const int width = static_cast<int>(columns);
        if ((columns != CUDACanonicalKpartColumns::Sixteen &&
             columns != CUDACanonicalKpartColumns::ThirtyTwo) ||
            reduction_elements <= 0 || reduction_elements % 32 != 0 ||
            partitions <= 0 || partitions > reduction_elements / 32 ||
            partitions > kMaximumThreads / width)
            return std::nullopt;
        return CUDACanonicalKpartFoldPlan(columns, partitions, reduction_elements);
    }

    /** @return Compiled physical column geometry. */
    [[nodiscard]] constexpr CUDACanonicalKpartColumns columns() const noexcept { return columns_; }
    /** @return Exact, unchanged arithmetic partition count. */
    [[nodiscard]] constexpr int partitions() const noexcept { return partitions_; }
    /** @return K authenticated when the plan was constructed. */
    [[nodiscard]] constexpr int reductionElements() const noexcept { return reduction_elements_; }
    /** @return Native threads, one per column and partition pair. */
    [[nodiscard]] constexpr int threads() const noexcept
    {
        return static_cast<int>(columns_) * partitions_;
    }
    /** @return CTA-local partial bytes; no persistent global workspace is needed. */
    [[nodiscard]] constexpr size_t sharedBytes() const noexcept
    {
        return static_cast<size_t>(threads()) * sizeof(float);
    }
    /** @brief Every geometry/arithmetic field participates in capture identity. */
    constexpr bool operator==(const CUDACanonicalKpartFoldPlan &) const noexcept = default;

private:
    static constexpr int kMaximumThreads = 1024; ///< Native CUDA CTA capacity, not an economy cutoff.
    /** @brief Construct only after create() has rejected invalid physical/arithmetic states. */
    constexpr CUDACanonicalKpartFoldPlan(CUDACanonicalKpartColumns columns, int partitions, int k) noexcept
        : columns_(columns), partitions_(partitions), reduction_elements_(k) {}
    CUDACanonicalKpartColumns columns_;
    int partitions_;
    int reduction_elements_;
};

/** @brief Compiler and occupancy evidence for one exact candidate specialization. */
struct CUDACanonicalKpartFoldResources
{
    int registers_per_thread = 0; ///< Native compiler register allocation.
    size_t local_bytes = 0; ///< Nonzero local storage disqualifies a timing candidate.
    size_t static_shared_bytes = 0; ///< Static CTA storage, in addition to plan.sharedBytes().
    int maximum_threads = 0; ///< Compiled specialization's launch bound.
    int active_blocks_per_sm = 0; ///< Occupancy for the exact supplied plan.
};
}

/**
 * @brief Query the compiled candidate before timing, without launching GPU work.
 * @param codebook Physical execution codebook.
 * @param plan Exact candidate identity whose occupancy is being measured.
 * @param resources Receives complete evidence only on success.
 * @return False for an unsupported specialization or runtime query failure.
 * @note Call on the owning current device before recording a graph.
 */
extern "C" bool cudaNativeVNNIGemvTuned_fusedKpar_resources(
    uint8_t codebook, const llaminar2::CUDACanonicalKpartFoldPlan &plan,
    llaminar2::CUDACanonicalKpartFoldResources &resources);

/**
 * @brief Launch one exact serial-row fold candidate on a caller-owned stream.
 *
 * This explicit candidate entrypoint is used by correctness/economy tooling;
 * ordinary runtime selection remains owned by the generated dispatch policy.
 * Input/output pointers refer to persistent caller-owned device buffers.
 * No allocation, transfer, host callback, or stream synchronization occurs.
 *
 * @param activations Quantized activation row [K], aligned for int4 reads.
 * @param payload Existing GPU packed weight payload.
 * @param scales Primary prepared weight metadata [K/32,N].
 * @param secondary Required secondary metadata for asymmetric/dual-scale formats.
 * @param extended_minima Required extended minima for dual-scale asymmetric formats.
 * @param output Final FP32 row [N].
 * @param activation_scales Existing FP32 block scales [K/32].
 * @param n Positive output width.
 * @param alpha Scale applied to each partial before folding.
 * @param beta Optional old-output coefficient after the fold.
 * @param existing Required prior row when beta is nonzero.
 * @param bias Optional output-column bias [N].
 * @param codebook Physical execution codebook, including normalized promotion forms.
 * @param device Exact CUDA ordinal owning the buffers and stream.
 * @param stream Exact non-default CUDA stream; null/implicit streams are rejected.
 * @param plan Immutable physical geometry and canonical reduction identity.
 * @return Whether the exact launch was admitted and submitted successfully.
 */
extern "C" bool cudaNativeVNNIGemvTuned_fusedKpar_fp32(
    const int8_t *activations, const uint8_t *payload, const uint16_t *scales,
    const uint16_t *secondary, const uint32_t *extended_minima, float *output,
    const float *activation_scales, int n, float alpha, float beta,
    const float *existing, const float *bias, uint8_t codebook, int device,
    void *stream, const llaminar2::CUDACanonicalKpartFoldPlan &plan);
