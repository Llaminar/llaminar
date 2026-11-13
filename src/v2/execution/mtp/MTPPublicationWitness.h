/**
 * @file MTPPublicationWitness.h
 * @brief Typed conservation witness from the actual committed publication counts.
 *
 * Output emission and main-state publication advance different frontiers when
 * one condition row crosses a transaction boundary. This transient value is
 * constructed from the owning publisher's counts, never from draft acceptance,
 * rollback or a ready-token guess. It adds no persistent controller state.
 */
#pragma once
#include <cstdint>
#include <climits>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MTP_PUBLICATION_HD __host__ __device__ __forceinline__
#else
#define LLAMINAR_MTP_PUBLICATION_HD inline
#endif
namespace llaminar2
{
/** @brief Validated actual output/state counts; the default placeholder is invalid. */
class MTPPublicationWitness
{
public:
    /** @brief Create an invalid output placeholder, never implicit publication evidence. */
    LLAMINAR_MTP_PUBLICATION_HD MTPPublicationWitness() noexcept = default;
    /** @brief Validate exact source counts before narrowing either one-row debt.
     * @param output_count Actual compact/host output rows including its condition.
     * @param leading_count Actual already-emitted leading rows, exactly zero or one.
     * @param published_count Actual main-state input rows successfully published.
     * @param stopped Actual source stopped-on-output result, including rejected stops.
     * @param out Written only after all count and one-row bounds hold.
     * @return False without modifying output for absent or incompatible evidence.
     * No depth/capture capacity is inferred here; the caller owns those bounds.
     */
    LLAMINAR_MTP_PUBLICATION_HD static bool fromCounts(int output_count, int leading_count,
        int published_count, bool stopped, MTPPublicationWitness *out) noexcept
    {
        if (!out || output_count <= 0 || published_count < 0 || published_count > output_count ||
            leading_count < 0 || leading_count > 1 || leading_count >= output_count ||
            output_count - published_count > 1) return false;
        MTPPublicationWitness value;
        value.output_count_ = output_count; value.leading_count_ = leading_count;
        value.published_count_ = published_count; value.stopped_ = stopped;
        *out = value; return true;
    }
    /** @brief Reconstruct counts only from explicitly recorded lossless publication bits.
     * @param emitted Actual recorded output delta, never inferred from acceptance.
     * @param incoming Exact recorded leading-row debt.
     * @param outgoing Exact recorded raw output-minus-publication debt, even on stop.
     * @param stopped Exact recorded model-stop observation.
     * @param out Validated result; unchanged on any overflow or invalid geometry.
     * @return Whether the supplied complete witness can represent nonnegative publication.
     */
    LLAMINAR_MTP_PUBLICATION_HD static bool fromEmitted(std::uint32_t emitted, bool incoming,
        bool outgoing, bool stopped, MTPPublicationWitness *out) noexcept
    {
        if (emitted == 0 || emitted > static_cast<std::uint32_t>(INT_MAX - static_cast<int>(incoming))) return false;
        const int output = static_cast<int>(emitted) + static_cast<int>(incoming);
        return fromCounts(output, static_cast<int>(incoming), output - static_cast<int>(outgoing), stopped, out);
    }
    /** @return Whether this value was constructed from a valid complete source witness. */
    LLAMINAR_MTP_PUBLICATION_HD bool valid() const noexcept
    {
        return output_count_ > 0 && published_count_ >= 0 && published_count_ <= output_count_ &&
            leading_count_ >= 0 && leading_count_ <= 1 && leading_count_ < output_count_ &&
            output_count_ - published_count_ <= 1;
    }
    /** @return Actual newly emitted rows O-L; callers must require valid(). */
    LLAMINAR_MTP_PUBLICATION_HD std::uint32_t newlyEmittedTokens() const noexcept
    { return static_cast<std::uint32_t>(output_count_ - leading_count_); }
    /** @return Actual published main-state input rows, not computed verifier rows. */
    LLAMINAR_MTP_PUBLICATION_HD std::uint32_t publishedStateTokens() const noexcept
    { return static_cast<std::uint32_t>(published_count_); }
    /** @return Exact already-emitted incoming condition debt L. */
    LLAMINAR_MTP_PUBLICATION_HD bool incomingDebt() const noexcept { return leading_count_ != 0; }
    /** @return Raw O-P debt; stopping does not erase this actual count difference. */
    LLAMINAR_MTP_PUBLICATION_HD bool outgoingDebt() const noexcept { return output_count_ != published_count_; }
    /** @return Actual stopped-on-output independently of rejection and budget exhaustion. */
    LLAMINAR_MTP_PUBLICATION_HD bool modelStopped() const noexcept { return stopped_; }
private:
    int output_count_ = 0, leading_count_ = 0, published_count_ = 0;
    bool stopped_ = false;
};
} // namespace llaminar2
#undef LLAMINAR_MTP_PUBLICATION_HD
