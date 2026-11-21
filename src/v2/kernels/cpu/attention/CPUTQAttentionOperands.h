/**
 * @file CPUTQAttentionOperands.h
 * @brief Checked borrowed operands for native CPU compressed-centroid attention.
 *
 * A view borrows one cache-owned physical head without extending its lifetime,
 * allocating, changing metadata or maintaining a capacity ledger. Runtime head
 * dimensions are restricted to the existing TQ tensor ABI. Production wrappers
 * bind original query/accumulator spans once before ISA dispatch; native leaves
 * cannot receive an unchecked mutable cache alias. PhysicalMemoryAuthority and
 * the enclosing cache publication lease remain the storage/lifetime authorities.
 */
#pragma once
#include "kernels/common/DeviceFP32NumericalContract.h"
#include "kernels/cpu/turboquant/TurboQuantCodebook.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>

namespace llaminar2::cpu::tq_attention
{
/** @brief Existing physical encodings; keys and values do not change their bytes. */
enum class Format { TQ4, TQ8 };
/** @brief Original norm for score estimators versus fitted value reconstruction radius. */
enum class NormRole { Source, Reconstruction };

/** @brief Compare bounded borrowed byte envelopes without constructing overflowing end pointers.
 * @param a First root; @param an Its complete physical bytes.
 * @param b Second root; @param bn Its complete physical bytes.
 * @return Whether the nonempty envelopes intersect; no storage is reserved here.
 */
[[nodiscard]] inline bool overlaps(const void *a, std::size_t an, const void *b, std::size_t bn) noexcept
{
    if (!an || !bn) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}

/** @brief Immutable, validated geometry and selected norm for one original physical cache head.
 * @tparam Encoding Existing TQ4 or TQ8 block layout.
 * @tparam Role Exact metadata word used by this arithmetic program.
 */
template <Format Encoding, NormRole Role> class HeadView final
{
    static_assert(Encoding == Format::TQ4 || Encoding == Format::TQ8);
    static_assert(Role == NormRole::Source || Role == NormRole::Reconstruction);
public:
    /** @brief Bind one complete original head, or an explicit empty noncollective operation.
     * @param block Cache-owned physical root, required only for a nonempty head.
     * @param dimensions Existing tensor head width: 64, 128 or 256; zero is empty.
     * @return Borrowed read view; only the selected immutable norm is retained.
     * @throws std::invalid_argument For unsupported geometry or missing live bytes.
     * @throws std::domain_error For a nonfinite/negative selected published norm.
     */
    [[nodiscard]] static HeadView bind(const std::uint8_t *block, int dimensions)
    {
        switch (dimensions)
        {
        case 0: return HeadView(nullptr, 0, 0);
        case 64: case 128: case 256: break;
        default: throw std::invalid_argument("CPU TQ attention requires a complete 64/128/256-coordinate head");
        }
        if (!block) throw std::invalid_argument("CPU TQ attention requires a live physical head");
        float norm;
        std::memcpy(&norm, block + (Role == NormRole::Source ? 0 : sizeof(float)), sizeof(norm));
        if (!std::isfinite(norm) || norm < 0.0f)
            throw std::domain_error("CPU TQ attention requires a finite nonnegative selected norm");
        return HeadView(block, dimensions, norm);
    }
    /** @return Original immutable physical block root, null only when empty. */
    [[nodiscard]] const std::uint8_t *data() const noexcept { return block_; }
    /** @return Validated complete logical head width, or zero. */
    [[nodiscard]] int dimensions() const noexcept { return dimensions_; }
    /** @return Original selected source/reconstruction norm, never a substituted radius. */
    [[nodiscard]] float norm() const noexcept { return norm_; }
    /** @return Complete physical block bytes, including both original norm words. */
    [[nodiscard]] std::size_t bytes() const noexcept
    { return dimensions_ ? 2 * sizeof(float) + std::size_t(dimensions_) / (Encoding == Format::TQ4 ? 2 : 1) : 0; }
    /** @return Whether the original source program contributes nonzero work. */
    [[nodiscard]] bool active() const noexcept { return dimensions_ && norm_ >= 1e-30f; }
private:
    /** @brief Retain checked synchronous borrows, not another storage owner.
     * @param block Validated root; @param dimensions Validated ABI width; @param norm Selected original metadata.
     */
    HeadView(const std::uint8_t *block, int dimensions, float norm) noexcept
        : block_(block), dimensions_(dimensions), norm_(norm) {}
    const std::uint8_t *block_;
    int dimensions_;
    float norm_;
};

/** @brief Checked original query and source-norm head for one score program.
 * @tparam Encoding Physical centroid layout; no decoded head is materialized.
 */
template <Format Encoding> class DotOperand final
{
public:
    /** @brief Bind original disjoint query/cache spans before selecting a native arithmetic leaf.
     * @param query Original FP32 coordinates; @param block Immutable complete physical head.
     * @param dimensions Existing ABI width or explicit zero work.
     * @return Checked borrowed score operands.
     * @throws std::invalid_argument For missing, aliased or malformed original storage.
     * @throws std::domain_error For invalid selected cache metadata.
     */
    [[nodiscard]] static DotOperand bind(const float *query, const std::uint8_t *block, int dimensions)
    {
        const auto head = HeadView<Encoding, NormRole::Source>::bind(block, dimensions);
        if (dimensions && (!query || overlaps(query, std::size_t(dimensions) * sizeof(float), block, head.bytes())))
            throw std::invalid_argument("CPU TQ query overlaps or lacks its original cache/query storage");
        return DotOperand(query, head);
    }
    /** @return Original FP32 query root; never a compacted or rotated copy. */
    [[nodiscard]] const float *query() const noexcept { return query_; }
    /** @return Original immutable complete cache head and selected source norm. */
    [[nodiscard]] const auto &head() const noexcept { return head_; }
    /** @return Whether the existing source-norm threshold admits arithmetic. */
    [[nodiscard]] bool active() const noexcept { return head_.active(); }
private:
    /** @brief Seal validated original borrows.
     * @param query Checked FP32 query; @param head Checked immutable physical source head.
     */
    DotOperand(const float *query, HeadView<Encoding, NormRole::Source> head) noexcept : query_(query), head_(head) {}
    const float *query_;
    HeadView<Encoding, NormRole::Source> head_;
};

/** @brief Checked numerator update, retaining exactly one separately rounded weight/radius product.
 * @tparam Encoding Original value centroid encoding.
 */
template <Format Encoding> class AccumOperand final
{
public:
    /** @brief Exclude cache/codebook mutation before any numerator store.
     * @param output Original complete FP32 numerator head, updated in place.
     * @param block Immutable physical value head; @param weight Original finite softmax weight.
     * @param dimensions Existing tensor ABI width, or explicit zero work.
     * @return Checked borrowed numerator update with the original reconstruction radius.
     * @throws std::invalid_argument For missing/aliased or unsupported storage.
     * @throws std::domain_error For nonfinite weight, selected norm or their live product.
     */
    [[nodiscard]] static AccumOperand bind(float *output, const std::uint8_t *block, float weight, int dimensions)
    {
        const auto head = HeadView<Encoding, NormRole::Reconstruction>::bind(block, dimensions);
        if (!std::isfinite(weight)) throw std::domain_error("CPU TQ attention requires a finite value weight");
        const auto count = std::size_t(head.dimensions()) * sizeof(float);
        const float *book = Encoding == Format::TQ4 ? TQ4_CENTROIDS.data() : TQ8_CENTROIDS.data();
        const auto book_bytes = Encoding == Format::TQ4 ? sizeof(TQ4_CENTROIDS) : sizeof(TQ8_CENTROIDS);
        if (dimensions && (!output || overlaps(output, count, block, head.bytes()) || overlaps(output, count, book, book_bytes)))
            throw std::invalid_argument("CPU TQ numerator aliases or lacks original cache/codebook storage");
        const bool active = head.active() && std::abs(weight) >= 1e-30f;
        const float combined = active ? device_fp32_contract::multiply(weight, head.norm()) : 0.0f;
        if (!std::isfinite(combined)) throw std::domain_error("CPU TQ weighted reconstruction radius is nonfinite");
        return AccumOperand(output, head, weight, combined, active);
    }
    /** @return Original complete writable numerator head. */
    [[nodiscard]] float *output() const noexcept { return output_; }
    /** @return Original immutable physical head and reconstruction norm. */
    [[nodiscard]] const auto &head() const noexcept { return head_; }
    /** @return Original finite weight for equivalent legacy ISA adapters. */
    [[nodiscard]] float weight() const noexcept { return weight_; }
    /** @return Exactly one rounded product of original weight and reconstruction radius. */
    [[nodiscard]] float combinedWeight() const noexcept { return combined_; }
    /** @return Whether the original threshold program admits this numerator update. */
    [[nodiscard]] bool active() const noexcept { return active_; }
private:
    /** @brief Seal one validated update without allocating or publishing cache state.
     * @param output Original numerator; @param head Original head; @param weight Original softmax weight.
     * @param combined Rounded product; @param active Original threshold decision.
     */
    AccumOperand(float *output, HeadView<Encoding, NormRole::Reconstruction> head, float weight, float combined, bool active) noexcept
        : output_(output), head_(head), weight_(weight), combined_(combined), active_(active) {}
    float *output_;
    HeadView<Encoding, NormRole::Reconstruction> head_;
    float weight_, combined_;
    bool active_;
};
} // namespace llaminar2::cpu::tq_attention
