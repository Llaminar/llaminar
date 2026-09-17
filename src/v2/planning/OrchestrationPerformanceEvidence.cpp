/**
 * @file OrchestrationPerformanceEvidence.cpp
 * @brief Checked, resource-order-aware arithmetic for automatic planning evidence.
 *
 * Validation is deliberately fatal: replacing a missing rate with zero or a
 * guessed backend constant changes which topology wins. All computations use
 * seconds and base work units. No device, model, allocation, or MPI call occurs
 * here, so mathematical regressions remain fast and device-free.
 */
#include "planning/OrchestrationPerformanceEvidence.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @brief Reject invalid caller data before it reaches max/sum comparisons. */
        void requireNonnegative(double value, const char *name)
        {
            if (!std::isfinite(value) || value < 0.0)
                throw std::invalid_argument(std::string(name) + " must be finite and nonnegative");
        }
        /** @brief Reject arithmetic overflow instead of allowing a bogus candidate ranking. */
        double finiteResult(double value)
        {
            if (!std::isfinite(value))
                throw std::overflow_error("Planning latency is not representable in seconds");
            return value;
        }
    }

    PlanningServiceObservation::PlanningServiceObservation(PlanningWorkUnit unit,
        double completed_work, double elapsed_seconds, std::string provenance)
        : unit_(unit), completed_work_(completed_work), elapsed_seconds_(elapsed_seconds),
          provenance_(std::move(provenance))
    {
        if (unit != PlanningWorkUnit::Bytes && unit != PlanningWorkUnit::ArithmeticOperations)
            throw std::invalid_argument("Planning observation has an invalid work unit");
        requireNonnegative(completed_work, "Completed planning work");
        requireNonnegative(elapsed_seconds, "Planning observation seconds");
        if (completed_work == 0.0 || elapsed_seconds == 0.0 ||
            provenance_.find_first_not_of(" \t\n\r\f\v") == std::string::npos)
            throw std::invalid_argument("Planning observation requires positive completed work/time and provenance");
        const double rate = unitsPerSecond();
        if (!std::isfinite(rate) || rate == 0.0)
            throw std::invalid_argument("Planning observation has an unrepresentable service rate");
    }

    double PlanningServiceObservation::secondsFor(PlanningWorkUnit unit, double work) const
    {
        if (unit != unit_)
            throw std::invalid_argument("Planning observation work-unit mismatch");
        requireNonnegative(work, "Predicted planning work");
        const double result = finiteResult(work / unitsPerSecond());
        if (work > 0.0 && result == 0.0)
            throw std::overflow_error("Positive planning work underflowed to zero latency");
        return result;
    }

    double planningArithmeticSeconds(const PlanningServiceObservation &decode, int prefill_rows,
        const PlanningServiceObservation &prefill, double rows, double operations)
    {
        if (decode.unit() != PlanningWorkUnit::ArithmeticOperations ||
            prefill.unit() != PlanningWorkUnit::ArithmeticOperations || prefill_rows <= 1 ||
            !std::isfinite(rows) || rows < 1)
            throw std::invalid_argument("Arithmetic service requires positive complete-row operation observations");
        requireNonnegative(operations, "Predicted arithmetic work");
        // Interpolate invocation time, not time per row. The latter introduces
        // a quadratic M term and can manufacture a falling invocation latency.
        // Outside the sampled interval retain the endpoint rate; do not invent
        // a superlinear speedup from a larger unseen matrix.
        const double fraction = std::clamp((rows - 1) / (prefill_rows - 1), 0.0, 1.0);
        const double reciprocal_rate = std::lerp(1 / decode.unitsPerSecond(),
            prefill_rows / prefill.unitsPerSecond(), fraction) / std::min(rows, double(prefill_rows));
        const double seconds = finiteResult(operations * reciprocal_rate);
        if (operations > 0 && seconds == 0)
            throw std::overflow_error("Positive arithmetic work underflowed to zero latency");
        return seconds;
    }

    PlanningKernelService::PlanningKernelService(PlanningServiceObservation arithmetic,
        PlanningServiceObservation memory, double invocation_seconds)
        : arithmetic_(std::move(arithmetic)), memory_(std::move(memory)),
          invocation_seconds_(invocation_seconds)
    {
        if (arithmetic_.unit() != PlanningWorkUnit::ArithmeticOperations ||
            memory_.unit() != PlanningWorkUnit::Bytes)
            throw std::invalid_argument("Kernel planning requires distinct arithmetic and memory evidence");
        requireNonnegative(invocation_seconds_, "Kernel invocation overhead");
    }

    double PlanningKernelService::seconds(double operations, double traffic_bytes) const
    {
        return finiteResult(invocation_seconds_ + std::max(
            arithmetic_.secondsFor(PlanningWorkUnit::ArithmeticOperations, operations),
            memory_.secondsFor(PlanningWorkUnit::Bytes, traffic_bytes)));
    }

    PlanningTransferLeg::PlanningTransferLeg(PlanningServiceObservation bandwidth,
        double message_seconds)
        : bandwidth_(std::move(bandwidth)), message_seconds_(message_seconds)
    {
        if (bandwidth_.unit() != PlanningWorkUnit::Bytes)
            throw std::invalid_argument("Transfer planning requires completed byte-service evidence");
        requireNonnegative(message_seconds_, "Link message overhead");
    }

    double PlanningTransferLeg::seconds(std::size_t payload_bytes) const
    {
        return finiteResult(message_seconds_ + bandwidth_.secondsFor(
            PlanningWorkUnit::Bytes, static_cast<double>(payload_bytes)));
    }

    PlanningPayloadServiceCurve::PlanningPayloadServiceCurve(std::size_t small_bytes,
        double small_seconds, std::size_t large_bytes, double large_seconds)
        : small_bytes_(small_bytes), large_bytes_(large_bytes),
          small_seconds_(small_seconds), large_seconds_(large_seconds)
    {
        requireNonnegative(small_seconds, "Small communication observation");
        requireNonnegative(large_seconds, "Large communication observation");
        if (!small_bytes || large_bytes < small_bytes || small_seconds == 0 || large_seconds == 0)
            throw std::invalid_argument("Payload service needs ordered positive sizes and completed times");
        // Sampling noise must not create a negative incremental byte price.
        // This changes only the prediction envelope, never the source receipt.
        large_seconds_ = std::max(small_seconds, large_seconds);
        if (small_bytes == large_bytes) small_seconds_ = large_seconds_;
    }

    double PlanningPayloadServiceCurve::seconds(std::size_t payload_bytes) const
    {
        if (!payload_bytes) throw std::invalid_argument("Payload service cannot price an absent message");
        if (payload_bytes <= small_bytes_) return small_seconds_;
        if (payload_bytes <= large_bytes_)
            return std::lerp(small_seconds_, large_seconds_,
                static_cast<double>(payload_bytes - small_bytes_) / (large_bytes_ - small_bytes_));
        // Use effective bulk throughput outside the sampled interval. Retaining
        // a zero fitted slope here would silently make arbitrarily large
        // payloads cost no more than a one-byte startup packet.
        return finiteResult(large_seconds_ * (static_cast<double>(payload_bytes) / large_bytes_));
    }

    PlanningChunkPipeline::PlanningChunkPipeline(std::size_t chunk_bytes)
        : chunk_bytes_(chunk_bytes)
    {
        if (chunk_bytes == 0)
            throw std::invalid_argument("Planning chunk pipeline requires a positive chunk size");
    }

    double planningTransferSeconds(std::span<const PlanningTransferLeg> legs,
        std::size_t payload_bytes, const PlanningTransferSchedule &schedule)
    {
        if (legs.empty()) throw std::invalid_argument("Planning transfer requires a nonempty directed path");
        if (std::holds_alternative<PlanningSerialTransfer>(schedule) || payload_bytes == 0)
        {
            // The same payload traverses every dependent leg. In particular,
            // two equal-rate host-staged legs take twice one leg's payload time,
            // not payload/min(rate1, rate2). Zero-byte control still traverses both.
            double total = 0.0;
            for (const auto &leg : legs) total = finiteResult(total + leg.seconds(payload_bytes));
            return total;
        }

        const auto chunk_bytes = std::get<PlanningChunkPipeline>(schedule).chunkBytes();
        const auto full_chunks = payload_bytes / chunk_bytes;
        const auto tail_bytes = payload_bytes % chunk_bytes;
        double prefix_service = 0.0, prefix_bottleneck = 0.0, tail_completion = 0.0;
        for (const auto &leg : legs)
        {
            // For j identical full chunks, completion at stage i is the sum
            // of service times through i plus (j-1) times their maximum. This
            // closed form avoids a loop over millions of chunks during search.
            double last_full_completion = 0.0;
            if (full_chunks != 0)
            {
                const double service = leg.seconds(chunk_bytes);
                prefix_service = finiteResult(prefix_service + service);
                prefix_bottleneck = std::max(prefix_bottleneck, service);
                last_full_completion = finiteResult(prefix_service +
                    static_cast<double>(full_chunks - 1) * prefix_bottleneck);
            }
            if (tail_bytes != 0)
            {
                // A short tail waits for both its previous leg and this leg's
                // last full chunk. Simply adding a full-path tail duration to
                // the makespan would double-count overlap at the drain edge.
                tail_completion = finiteResult(std::max(tail_completion,
                    last_full_completion) + leg.seconds(tail_bytes));
            }
        }
        return tail_bytes != 0 ? tail_completion : finiteResult(prefix_service +
            static_cast<double>(full_chunks - 1) * prefix_bottleneck);
    }

    double planningSerialSeconds(std::span<const double> costs)
    {
        double total = 0.0;
        for (const double cost : costs)
        {
            requireNonnegative(cost, "Dependent planning cost");
            total = finiteResult(total + cost);
        }
        return total;
    }

    double planningIndependentSeconds(std::span<const double> costs)
    {
        double longest = 0.0;
        for (const double cost : costs)
        {
            requireNonnegative(cost, "Independent planning cost");
            longest = std::max(longest, cost);
        }
        return longest;
    }
}
