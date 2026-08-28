/**
 * @file PerfStatsCollector.h
 * @brief Unified structured performance counter and timer collection.
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2
{
    struct PerfStatRecord
    {
        enum class Kind
        {
            Counter,
            Timer,
            OrderedSequence
        };

        Kind kind = Kind::Counter;
        std::string domain;
        std::string name;
        std::string phase;
        std::string device;
        std::map<std::string, std::string> tags;

        uint64_t count = 0;
        double value = 0.0;
        uint64_t total_ns = 0;
        uint64_t min_ns = 0;
        uint64_t max_ns = 0;
        /** Number of canonical 64-bit words folded into ordered evidence. */
        uint64_t sequence_word_count = 0;
        /** First half of the ordered, topology-bounded sequence fingerprint. */
        uint64_t sequence_digest_lo = 0;
        /** Independent second half of the ordered sequence fingerprint. */
        uint64_t sequence_digest_hi = 0;
    };

    class PerfStatsCollector
    {
    public:
        using Tags = std::map<std::string, std::string>;
        using Clock = std::chrono::steady_clock;

        /**
         * @brief Exact record family retained across a measurement reset.
         *
         * Domain-only retention is appropriate for wholly setup-owned domains
         * such as graph kernel inventory. Mixed domains need a narrower key:
         * graph capture evidence and measured replay counters both live in
         * `forward_graph`, and retaining that complete domain would contaminate
         * the benchmark window with warmup runtime activity.
         */
        struct RecordFamily
        {
            std::string domain; ///< Exact PerfStats domain.
            std::string name;   ///< Exact counter/timer name within the domain.
        };

        /**
         * @brief Return whether any structured performance evidence is requested.
         *
         * This broad gate controls inexpensive producers only. Expensive
         * instrumentation families, such as per-stage CPU clocks or GPU
         * events, must additionally consult their family-specific gate.
         */
        static bool isEnabled();

        /**
         * @brief Return whether one structured evidence domain is requested.
         *
         * `LLAMINAR_PERF_STATS_FILTER` is a collection contract, not merely an
         * export formatter. A non-empty filter enables only matching domains
         * (including a qualified `domain.name` prefix), allowing expensive
         * producers to remain entirely dormant during focused measurements.
         * An enabled export with no filter retains all domains.
         *
         * @param domain Exact PerfStats domain owned by the caller.
         */
        static bool isDomainEnabled(std::string_view domain);

        /**
         * @brief Return whether per-stage CPU wall-clock timing is requested.
         *
         * A generic PerfStats export does not enable this hot-path
         * instrumentation. Callers must explicitly request a `stage_cpu` or
         * `stage_cpu_detail` filter, or set
         * `LLAMINAR_PERF_STATS_CPU_STAGE_TIMING=1`.
         */
        static bool cpuStageTimingEnabled();

        /**
         * @brief Return whether per-stage GPU event timing is requested.
         */
        static bool gpuStageEventTimingEnabled();
        static void reset();
        static void resetPreservingDomains(
            const std::vector<std::string> &domains_to_preserve);

        /**
         * @brief Reset measured evidence while retaining named setup records.
         *
         * A record survives when either its complete domain appears in
         * `domains_to_preserve` or its exact `(domain, name)` pair appears in
         * `record_families_to_preserve`. Tags, phase, device, count, and timing
         * aggregates remain intact for surviving records.
         *
         * @param domains_to_preserve Complete domains owned by immutable setup.
         * @param record_families_to_preserve Exact families from mixed domains.
         */
        static void resetPreserving(
            const std::vector<std::string> &domains_to_preserve,
            const std::vector<RecordFamily> &record_families_to_preserve);

        static void addCounter(
            std::string domain,
            std::string name,
            double value = 1.0,
            std::string phase = {},
            std::string device = {},
            Tags tags = {});

        static void recordTimingNs(
            std::string domain,
            std::string name,
            uint64_t duration_ns,
            std::string phase = {},
            std::string device = {},
            Tags tags = {});

        /**
         * @brief Fold one ordered lifecycle step into a bounded evidence row.
         *
         * PerfStats keys describe stable aggregation dimensions. Request,
         * token, command, and generation identifiers must therefore never be
         * placed in tags: doing so turns the collector into an unbounded event
         * log and adds progressively more map work to inference. This method
         * retains the exact order-sensitive protocol witness in two 64-bit
         * fingerprints while the key remains bounded by topology and phase.
         *
         * Equal step counts, word counts, and both independent digests provide
         * bounded order-sensitive evidence that two independently executing
         * roles observed the same canonical sequence. Runtime protocol
         * validation remains the authority that rejects a mismatched ticket.
         * The word encoding is explicitly little-endian and is stable across
         * CPU architectures and device backends.
         *
         * @param domain Stable PerfStats domain.
         * @param name Stable evidence-family name.
         * @param words Canonical words for exactly one ordered step; non-empty.
         * @param phase Stable execution phase.
         * @param device Stable device or participant description.
         * @param tags Stable, bounded aggregation dimensions only.
         */
        static void recordOrderedSequenceStep(
            std::string domain,
            std::string name,
            std::initializer_list<uint64_t> words,
            std::string phase = {},
            std::string device = {},
            Tags tags = {});

        static std::vector<PerfStatRecord> snapshot(
            const std::vector<std::string> &filters = {});

        static std::string jsonString(
            const std::vector<std::string> &filters = {});

        static std::string csvString(
            const std::vector<std::string> &filters = {});

        static std::string summaryString(
            const std::vector<std::string> &filters = {},
            size_t max_records = 120);

        static bool writeJson(
            const std::string &path,
            const std::vector<std::string> &filters = {});

        static bool writeCsv(
            const std::string &path,
            const std::vector<std::string> &filters = {});

        static void printSummary(
            const std::vector<std::string> &filters = {},
            size_t max_records = 120);

        /**
         * @brief Export configured reports without allowing MPI ranks to race.
         *
         * Plain JSON/CSV paths are written by rank zero. A path containing the
         * literal `{rank}` token opts into one report per participant and is
         * expanded with that process's MPI rank before writing.
         *
         * @return true when every requested export completed successfully.
         */
        static bool flushFromEnv();

        class ScopedTimer
        {
        public:
            ScopedTimer(
                std::string domain,
                std::string name,
                std::string phase = {},
                std::string device = {},
                Tags tags = {});

            ~ScopedTimer();

            ScopedTimer(const ScopedTimer &) = delete;
            ScopedTimer &operator=(const ScopedTimer &) = delete;
            ScopedTimer(ScopedTimer &&) = delete;
            ScopedTimer &operator=(ScopedTimer &&) = delete;

        private:
            bool enabled_ = false;
            std::string domain_;
            std::string name_;
            std::string phase_;
            std::string device_;
            Tags tags_;
            Clock::time_point start_{};
        };
    };

} // namespace llaminar2
