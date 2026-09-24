/**
 * @file OrchestrationPerformanceEvidence.h
 * @brief Validated service observations and protocol-aware latency arithmetic.
 *
 * These setup-only values contain neither inventory nor allocation accounting.
 * A measurement owner supplies completed observations from the exact production
 * device/kernel/link being priced. Missing or failed measurements cannot be
 * represented as zero bandwidth, and bytes cannot be confused with operations.
 * The resulting latency is a prediction, never a measured model benchmark.
 *
 * This arithmetic does not discover, sample, or select a transport. In particular,
 * a host-staged transfer is serial unless its actual protocol explicitly permits
 * chunk pipelining. Independent-path composition is valid only for lanes whose
 * production schedule and resource ownership permit concurrent execution.
 */
#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <variant>

namespace llaminar2
{
    /** @brief Units of completed work; arithmetic operations count one FMA as two. */
    enum class PlanningWorkUnit { Bytes, ArithmeticOperations };

    /** @brief An indivisible completed work/time observation with its provenance. */
    class PlanningServiceObservation final
    {
    public:
        /**
         * @brief Seal strictly positive finite work and elapsed time.
         * @param unit Meaning of the work numerator, always in base SI units.
         * @param completed_work Work actually completed within the timed interval.
         * @param elapsed_seconds Positive elapsed seconds, not milliseconds.
         * @param provenance Exact sample identity/method; whitespace-only is invalid.
         * @throws std::invalid_argument for absent, nonfinite or unrepresentable rates.
         *
         * A transfer failure must propagate before construction. Callers retain
         * device, format, geometry and worker identity with this observation;
         * this value grants no permission to reuse it for a different kernel.
         */
        PlanningServiceObservation(PlanningWorkUnit unit, double completed_work,
            double elapsed_seconds, std::string provenance);
        /** @return Work-unit identity checked before applying this measurement. */
        PlanningWorkUnit unit() const noexcept { return unit_; }
        /** @return Observed base units per second, not a vendor peak specification. */
        double unitsPerSecond() const noexcept { return completed_work_ / elapsed_seconds_; }
        /** @return Original completed work, retained rather than only a rounded derived rate. */
        double completedWork() const noexcept { return completed_work_; }
        /** @return Original timed interval for a reproducible evidence receipt. */
        double elapsedSeconds() const noexcept { return elapsed_seconds_; }
        /** @return Evidence explaining the completed measurement. */
        const std::string &provenance() const noexcept { return provenance_; }
        /**
         * @brief Predict time for nonnegative work of the same unit.
         * @throws std::invalid_argument for unit mismatch or invalid work.
         * @throws std::overflow_error for a nonrepresentable positive prediction.
         */
        double secondsFor(PlanningWorkUnit unit, double work) const;
    private:
        PlanningWorkUnit unit_;
        double completed_work_;
        double elapsed_seconds_;
        std::string provenance_;
    };

    /**
     * @brief Estimate arithmetic from one authenticated family's M=1 and bounded prefill samples.
     * @param decode Single-row effective arithmetic service, not byte bandwidth.
     * @param prefill_rows Positive measured M greater than one.
     * @param prefill The same operation family's service at prefill_rows.
     * @param rows Actual mean nonempty invocation rows, at least one and possibly fractional.
     * @param operations Total arithmetic work being priced, not source storage bytes.
     * @return Seconds from linear invocation-time interpolation and endpoint-rate extrapolation.
     *
     * Family, endpoint and precision authentication remain the catalog's job.
     * This shared arithmetic cannot establish that GEMM measures attention, or
     * that quantized GOPS are FP32 FLOPS. Memory bounds are composed separately.
     */
    double planningArithmeticSeconds(const PlanningServiceObservation &decode, int prefill_rows,
        const PlanningServiceObservation &prefill, double rows, double operations);

    /** @brief Compute and memory service evidence for one applicable kernel family. */
    class PlanningKernelService final
    {
    public:
        /**
         * @brief Require separate operation and byte observations plus fixed overhead.
         * @param arithmetic Completed arithmetic observation for this kernel family.
         * @param memory Completed memory-service observation for this working set.
         * @param invocation_seconds Measured fixed overhead per invocation, nonnegative.
         * @throws std::invalid_argument for wrong units or invalid fixed overhead.
         */
        PlanningKernelService(PlanningServiceObservation arithmetic,
            PlanningServiceObservation memory, double invocation_seconds);
        /**
         * @brief Price one invocation using overhead plus the limiting service demand.
         * @param operations Arithmetic operations actually issued, including padding.
         * @param traffic_bytes Bytes serviced, not the allocated tensor capacity.
         * @return Fixed overhead plus max(arithmetic seconds, memory seconds).
         *
         * Independent invocations are priced separately. Summing all weights
         * before applying the roofline would hide serial memory-bound and
         * compute-bound kernels behind an overlap that does not exist.
         */
        double seconds(double operations, double traffic_bytes) const;
        /** @return Exact compute observation for diagnostics and evidence receipts. */
        const PlanningServiceObservation &arithmetic() const noexcept { return arithmetic_; }
        /** @return Exact memory observation for diagnostics and evidence receipts. */
        const PlanningServiceObservation &memory() const noexcept { return memory_; }
    private:
        PlanningServiceObservation arithmetic_;
        PlanningServiceObservation memory_;
        double invocation_seconds_;
    };

    /** @brief One directed link leg, including the protocol's per-message startup cost. */
    class PlanningTransferLeg final
    {
    public:
        /** @brief Require byte-rate evidence and a nonnegative measured message latency. */
        PlanningTransferLeg(PlanningServiceObservation bandwidth, double message_seconds);
        /** @return Startup plus payload service; even a zero-byte control message has latency. */
        double seconds(std::size_t payload_bytes) const;
        /** @return Completed directed-link evidence, without inventing reverse symmetry. */
        const PlanningServiceObservation &bandwidth() const noexcept { return bandwidth_; }
    private:
        PlanningServiceObservation bandwidth_;
        double message_seconds_;
    };

    /**
     * @brief Bounded payload/time interpolation for one unchanged communication protocol.
     *
     * This is an estimate, not a new measured bandwidth. The small-payload
     * observation remains the latency floor; a noisy faster large observation
     * is raised to that floor. Within the observed range, interpolate complete
     * invocation times. Beyond it, extend at the large observation's effective
     * byte rate, so a flat/noisy pair never implies free unlimited traffic.
     * Physical endpoints, collective degree and wire precision belong to the
     * caller's authenticated service. This curve cannot change any of them.
     */
    class PlanningPayloadServiceCurve final
    {
    public:
        /**
         * @brief Seal two positive payload sizes and completed invocation times.
         * @param small_bytes Positive smaller payload, or the same size as large_bytes.
         * @param small_seconds Complete smaller invocation, in seconds.
         * @param large_bytes Greater/equal payload measured on the same protocol.
         * @param large_seconds Complete larger invocation, in seconds.
         * @throws std::invalid_argument for unordered sizes or invalid observations.
         *
         * Equal sizes retain the slower observation and define one point, not
         * a fabricated slope. Original observations remain in the owning receipt.
         */
        PlanningPayloadServiceCurve(std::size_t small_bytes, double small_seconds,
            std::size_t large_bytes, double large_seconds);
        /**
         * @return Positive monotone predicted invocation time for a positive payload.
         * @throws std::invalid_argument for a zero payload; absent operations are not messages.
         * @throws std::overflow_error if extrapolation cannot be represented.
         */
        double seconds(std::size_t payload_bytes) const;
    private:
        std::size_t small_bytes_, large_bytes_;
        double small_seconds_, large_seconds_;
    };

    /** @brief Each complete payload leg finishes before the next begins. */
    struct PlanningSerialTransfer final {};
    /**
     * @brief Protocol with independent link legs and retained chunk staging slots.
     *
     * This is a declared execution fact, not permission to add pipeline buffers.
     * Shared-link or single-buffer protocols must use PlanningSerialTransfer.
     */
    class PlanningChunkPipeline final
    {
    public:
        /** @brief Reject a zero chunk size before evaluating any protocol. */
        explicit PlanningChunkPipeline(std::size_t chunk_bytes);
        /** @return Payload bytes per full chunk, excluding the final partial chunk. */
        std::size_t chunkBytes() const noexcept { return chunk_bytes_; }
    private:
        std::size_t chunk_bytes_;
    };
    /** @brief Exactly one explicit transfer schedule; no implicit faster-path substitution. */
    using PlanningTransferSchedule = std::variant<PlanningSerialTransfer, PlanningChunkPipeline>;

    /**
     * @brief Price a directed path using its declared serial or chunk-pipelined schedule.
     * @param legs Ordered nonempty path, with independent observations per direction/leg.
     * @param payload_bytes Actual payload, not the source or destination allocation size.
     * @param schedule Existing production protocol; the estimator cannot choose it.
     * @return Complete path latency including fill/drain, per-chunk overhead and short tail.
     * @throws std::invalid_argument for an empty path.
     * @throws std::overflow_error for an unrepresentable duration.
     *
     * Complexity is O(legs), independent of chunk count or payload length.
     */
    double planningTransferSeconds(std::span<const PlanningTransferLeg> legs,
        std::size_t payload_bytes, const PlanningTransferSchedule &schedule);

    /** @brief Sum dependent costs; an empty, explicitly absent phase costs zero. */
    double planningSerialSeconds(std::span<const double> costs);
    /**
     * @brief Critical-path maximum for proven independent lanes, not shared resources.
     * @throws std::invalid_argument for negative or nonfinite input costs.
     */
    double planningIndependentSeconds(std::span<const double> costs);
}
