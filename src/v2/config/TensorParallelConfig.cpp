/**
 * @file TensorParallelConfig.cpp
 * @brief Exact proportional query, KV, FFN and vocabulary ownership.
 *
 * Replicated KV retains the full original bank and global GQA mapping; local
 * query intervals stay integral without an artificial KV-count alignment.
 * CPU participants share a host address space; canonical local ranks retain
 * their distinct shards and ambiguous address-only lookups fail explicitly.
 * Model-specific GDN key-head boundary validation remains a separate guard.
 * Opt-in query coverage authenticates exact original intervals and their union;
 * logical totals count each original head once, never each physical replica.
 */

#include "TensorParallelConfig.h"
#include "collective/ILocalTPContext.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace llaminar2
{

    // =========================================================================
    // DeviceShardingAssignment
    // =========================================================================

    std::string DeviceShardingAssignment::toString() const
    {
        std::ostringstream oss;
        oss << "Rank " << local_rank << " (" << device.to_string() << "): "
            << "heads[" << head_start << ":" << headEnd() << "]=" << head_count
            << ", kv_heads[" << kv_head_start << ":" << kvHeadEnd() << "]=" << kv_head_count
            << ", d_ff[" << d_ff_start << ":" << dFFEnd() << "]=" << d_ff_count
            << ", vocab[" << vocab_start << ":" << vocabEnd() << "]=" << vocab_count
            << " (" << (work_fraction * 100.0f) << "%)";
        return oss.str();
    }

    // =========================================================================
    // TensorParallelConfig - Constructor
    // =========================================================================

    TensorParallelConfig::TensorParallelConfig(std::vector<DeviceShardingAssignment> assignments)
        : assignments_(std::move(assignments))
    {
        computeTotals();

        // Detect if proportional (check if all work fractions are equal)
        if (assignments_.size() > 1)
        {
            float first_fraction = assignments_[0].work_fraction;
            float tolerance = 0.01f; // 1% tolerance
            is_proportional_ = std::any_of(
                assignments_.begin() + 1, assignments_.end(),
                [first_fraction, tolerance](const DeviceShardingAssignment &a)
                {
                    return std::abs(a.work_fraction - first_fraction) > tolerance;
                });
        }
    }

    TensorParallelConfig::TensorParallelConfig(
        std::vector<DeviceShardingAssignment> assignments, KVHeadCoverageDeclaration source,
        std::optional<QueryHeadPartitionDeclaration> query_partition)
        : assignments_(std::move(assignments)), kv_coverage_(source), query_partition_(std::move(query_partition))
    {
        // Validate every endpoint and sum before legacy int arithmetic can run.
        const auto query_error = queryPartitionError();
        if (!query_error.empty()) throw std::invalid_argument(query_error);
        const auto coverage_error = kvCoverageError();
        if (!coverage_error.empty()) throw std::invalid_argument(coverage_error);
        computeTotals();
        const auto error = validationError();
        if (!error.empty()) throw std::invalid_argument(error);
        if (assignments_.size() > 1)
            is_proportional_ = std::any_of(assignments_.begin() + 1, assignments_.end(),
                [&](const auto &a) { return std::abs(a.work_fraction - assignments_[0].work_fraction) > 0.01f; });
    }

    TensorParallelConfig::TensorParallelConfig(
        std::vector<DeviceShardingAssignment> assignments, QueryHeadPartitionDeclaration query_partition)
        : assignments_(std::move(assignments)), query_partition_(std::move(query_partition))
    {
        const auto query_error = queryPartitionError();
        if (!query_error.empty()) throw std::invalid_argument(query_error);
        computeTotals();
        const auto error = validationError();
        if (!error.empty()) throw std::invalid_argument(error);
        if (assignments_.size() > 1)
            is_proportional_ = std::any_of(assignments_.begin() + 1, assignments_.end(),
                [&](const auto &a) { return std::abs(a.work_fraction - assignments_[0].work_fraction) > 0.01f; });
    }

    // =========================================================================
    // TensorParallelConfig - Accessors
    // =========================================================================

    const DeviceShardingAssignment &TensorParallelConfig::forDevice(DeviceId device) const
    {
        auto it = std::find_if(
            assignments_.begin(), assignments_.end(),
            [device](const DeviceShardingAssignment &a)
            { return a.device == device; });

        if (it == assignments_.end())
        {
            throw std::out_of_range(
                "Device not found in TensorParallelConfig: " + device.to_string());
        }
        // A CPU DeviceId names host storage, not a NUMA participant. Selecting
        // the first match would silently give later ranks the wrong shard.
        if (std::any_of(it + 1, assignments_.end(),
                        [device](const auto &a) { return a.device == device; }))
            throw std::invalid_argument(
                "Ambiguous device in TensorParallelConfig; use forRank: " + device.to_string());
        return *it;
    }

    const DeviceShardingAssignment &TensorParallelConfig::forRank(int rank) const
    {
        if (rank < 0 || rank >= static_cast<int>(assignments_.size()))
        {
            throw std::out_of_range(
                "Rank " + std::to_string(rank) + " out of bounds [0, " +
                std::to_string(assignments_.size()) + ")");
        }
        return assignments_[rank];
    }

    // =========================================================================
    // TensorParallelConfig - Validation
    // =========================================================================

    std::string TensorParallelConfig::queryPartitionError() const
    {
        if (!query_partition_) return {};
        if (assignments_.empty() || assignments_.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
            return "Query consumer partition has no representable participants";
        const int total = query_partition_->originalQueryHeads();
        const int grain = query_partition_->granularity();
        if (kv_coverage_ && kv_coverage_->query_heads != total)
            return "Query and KV declarations name different original head spaces";
        std::int64_t query_sum = 0, ffn_sum = 0, vocab_sum = 0, kv_sum = 0;
        bool kv_full_replicas = true;
        for (size_t rank = 0; rank < assignments_.size(); ++rank)
        {
            const auto &a = assignments_[rank];
            if (a.local_rank != static_cast<int>(rank) || a.head_start < 0 || a.head_count <= 0 ||
                a.head_start >= total || a.head_count > total - a.head_start ||
                a.head_start % grain || a.head_count % grain)
                return "Query assignment cuts an original consumer head or has stale participant identity";
            for (const auto &[first, count] : {std::pair{a.d_ff_start, a.d_ff_count},
                    std::pair{a.vocab_start, a.vocab_count}, std::pair{a.kv_head_start, a.kv_head_count}})
                if (first < 0 || count <= 0 || count > std::numeric_limits<int>::max() - first)
                    return "Aligned query assignment has empty or unrepresentable physical intervals";
            query_sum += a.head_count; ffn_sum += a.d_ff_count; vocab_sum += a.vocab_count; kv_sum += a.kv_head_count;
            kv_full_replicas &= a.kv_head_start == 0 && a.kv_head_count == assignments_.front().kv_head_count;
            if (query_sum > total || ffn_sum > std::numeric_limits<int>::max() || vocab_sum > std::numeric_limits<int>::max())
                return "Aligned query assignment totals exceed original or representable bounds";
        }
        if (query_sum != total) return "Aligned query assignment misses original query heads";
        if (!kv_coverage_)
        {
            const std::int64_t logical_kv = kv_full_replicas ? assignments_.front().kv_head_count : kv_sum;
            if (logical_kv > std::numeric_limits<int>::max())
                return "Aligned query assignment logical KV total overflows";
            for (const auto &a : assignments_)
            {
                KVHeadInterval needed;
                try { needed = resolveQueryKVHeadCoverage({total, static_cast<int>(logical_kv)}, a.head_start, a.head_count); }
                catch (const std::invalid_argument &error) { return error.what(); }
                // Keeping legacy KV means keeping its exact full-bank/disjoint
                // intervals. A new query grain cannot quietly borrow missing
                // heads or reinterpret those old intervals as subset replicas.
                if (a.kv_head_start > needed.first || a.kvHeadEnd() < needed.end())
                    return "Legacy KV interval cannot serve the aligned original query consumer";
            }
        }
        return {};
    }

    std::string TensorParallelConfig::kvCoverageError() const
    {
        if (!kv_coverage_) return {};
        const auto source = *kv_coverage_;
        try { validateKVHeadCoverageDeclaration(source); }
        catch (const std::invalid_argument &error) { return error.what(); }
        if (assignments_.empty() || assignments_.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
            return "KV query coverage has no representable participant membership";
        std::vector<KVHeadInterval> intervals;
        std::int64_t query_sum = 0, ffn_sum = 0, vocab_sum = 0;
        for (size_t rank = 0; rank < assignments_.size(); ++rank)
        {
            const auto &a = assignments_[rank];
            if (a.local_rank != static_cast<int>(rank))
                return "KV-covered assignment local rank differs from canonical participant order";
            KVHeadInterval expected;
            try { expected = resolveQueryKVHeadCoverage(source, a.head_start, a.head_count); }
            catch (const std::invalid_argument &error) { return error.what(); }
            if (a.kv_head_start != expected.first || a.kv_head_count != expected.count)
                return "KV interval does not equal its original query consumer coverage";
            for (const auto &[first, count] : {std::pair{a.d_ff_start, a.d_ff_count},
                                              std::pair{a.vocab_start, a.vocab_count}})
                if (first < 0 || count <= 0 || count > std::numeric_limits<int>::max() - first)
                    return "KV-covered assignment has invalid FFN/vocabulary interval";
            query_sum += a.head_count; ffn_sum += a.d_ff_count; vocab_sum += a.vocab_count;
            if (query_sum > source.query_heads || ffn_sum > std::numeric_limits<int>::max() ||
                vocab_sum > std::numeric_limits<int>::max())
                return "KV-covered assignment totals exceed original or representable bounds";
            intervals.push_back(expected);
        }
        if (query_sum != source.query_heads) return "KV query assignment does not cover original global Q heads";
        std::sort(intervals.begin(), intervals.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });
        int end = 0;
        for (const auto &interval : intervals)
        {
            if (interval.first > end) return "KV query coverage has an unowned original head";
            end = std::max(end, interval.end());
        }
        if (end != source.kv_heads) return "KV query coverage does not cover the complete original KV bank";
        return {};
    }

    void TensorParallelConfig::computeTotals()
    {
        total_heads_ = 0;
        total_kv_heads_ = 0;
        total_d_ff_ = 0;
        total_vocab_ = 0;

        for (const auto &a : assignments_)
        {
            total_heads_ += a.head_count;
            if (!kv_coverage_ && !query_partition_) total_kv_heads_ += a.kv_head_count;
            total_d_ff_ += a.d_ff_count;
            total_vocab_ += a.vocab_count;
        }

        if (kv_coverage_)
        {
            total_kv_heads_ = kv_coverage_->kv_heads;
            return;
        }

        if (query_partition_)
        {
            std::int64_t count = 0;
            for (const auto &a : assignments_) count += a.kv_head_count;
            // A representable disjoint total was checked before this call.
            // Full replicas can exceed int in physical sum; the original
            // reduction below supplies their one logical original count.
            total_kv_heads_ = count <= std::numeric_limits<int>::max() ? static_cast<int>(count) : 0;
        }

        // GQA replication fix: when KV heads are replicated across all devices
        // (all have kv_head_start=0 with same kv_head_count), the sum overcounts.
        // Use one device's count as the actual total.
        if (assignments_.size() > 1 && assignments_[0].kv_head_count > 0)
        {
            bool all_replicated = true;
            for (size_t i = 1; i < assignments_.size(); ++i)
            {
                if (assignments_[i].kv_head_start != 0 ||
                    assignments_[i].kv_head_count != assignments_[0].kv_head_count)
                {
                    all_replicated = false;
                    break;
                }
            }
            if (all_replicated && assignments_[0].kv_head_start == 0)
            {
                total_kv_heads_ = assignments_[0].kv_head_count;
            }
        }
    }

    bool TensorParallelConfig::validate() const
    {
        return validationError().empty();
    }

    std::string TensorParallelConfig::validationError() const
    {
        if (const auto error = queryPartitionError(); !error.empty()) return error;
        if (const auto error = kvCoverageError(); !error.empty()) return error;
        if (assignments_.empty())
        {
            return "No device assignments";
        }

        // Check for valid devices
        for (size_t i = 0; i < assignments_.size(); ++i)
        {
            if (!assignments_[i].device.is_valid())
            {
                return "Assignment " + std::to_string(i) + " has invalid device";
            }
            if (assignments_[i].head_count <= 0 ||
                assignments_[i].kv_head_count <= 0 ||
                assignments_[i].d_ff_count <= 0 ||
                assignments_[i].vocab_count <= 0)
            {
                return "Assignment " + std::to_string(i) +
                       " has an empty query, KV, FFN, or vocabulary shard";
            }
        }

        // CPU DeviceId is one shared host address space. Its shard identity is
        // the canonical participant rank, while GPU devices remain exclusive.
        // NUMA-domain membership is authenticated by the execution-domain owner.
        for (size_t i = 0; i < assignments_.size(); ++i)
            if (assignments_[i].device.is_cpu() && assignments_[i].local_rank != static_cast<int>(i))
                return "CPU assignment local rank differs from canonical participant order";

        // Check for duplicate GPU devices
        for (size_t i = 0; i < assignments_.size(); ++i)
        {
            for (size_t j = i + 1; j < assignments_.size(); ++j)
            {
                if (!assignments_[i].device.is_cpu() && assignments_[i].device == assignments_[j].device)
                {
                    return "Duplicate device: " + assignments_[i].device.to_string();
                }
            }
        }

        // Check for overlapping head ranges
        // Detect GQA KV replication: all devices have same kv_head_start=0 and same count
        bool kv_replicated = assignments_.size() > 1;
        if (kv_replicated)
        {
            for (size_t i = 1; i < assignments_.size(); ++i)
            {
                if (assignments_[i].kv_head_start != assignments_[0].kv_head_start ||
                    assignments_[i].kv_head_count != assignments_[0].kv_head_count)
                {
                    kv_replicated = false;
                    break;
                }
            }
        }

        for (size_t i = 0; i < assignments_.size(); ++i)
        {
            for (size_t j = i + 1; j < assignments_.size(); ++j)
            {
                const auto &a = assignments_[i];
                const auto &b = assignments_[j];

                // Check Q head overlap
                if (a.head_start < b.headEnd() && b.head_start < a.headEnd())
                {
                    return "Overlapping Q head ranges between rank " +
                           std::to_string(i) + " and " + std::to_string(j);
                }

                // Check KV head overlap (skip when GQA-replicated — all devices have same KV range)
                if (!kv_coverage_ && !kv_replicated && a.kv_head_start < b.kvHeadEnd() && b.kv_head_start < a.kvHeadEnd())
                {
                    return "Overlapping KV head ranges between rank " +
                           std::to_string(i) + " and " + std::to_string(j);
                }

                // Check d_ff overlap
                if (a.d_ff_start < b.dFFEnd() && b.d_ff_start < a.dFFEnd())
                {
                    return "Overlapping d_ff ranges between rank " +
                           std::to_string(i) + " and " + std::to_string(j);
                }

                // Check vocab overlap
                if (a.vocab_start < b.vocabEnd() && b.vocab_start < a.vocabEnd())
                {
                    return "Overlapping vocab ranges between rank " +
                           std::to_string(i) + " and " + std::to_string(j);
                }
            }
        }

        // Check contiguity (ranges should be contiguous with no gaps)
        std::vector<const DeviceShardingAssignment *> sorted;
        for (const auto &a : assignments_)
        {
            sorted.push_back(&a);
        }

        // Sort by head_start to check contiguity
        std::sort(sorted.begin(), sorted.end(),
                  [](const DeviceShardingAssignment *a, const DeviceShardingAssignment *b)
                  {
                      return a->head_start < b->head_start;
                  });

        // First assignment should start at 0
        if (sorted[0]->head_start != 0)
        {
            return "Head ranges do not start at 0 (starts at " +
                   std::to_string(sorted[0]->head_start) + ")";
        }

        // Each subsequent assignment should start where previous ended
        for (size_t i = 1; i < sorted.size(); ++i)
        {
            if (sorted[i]->head_start != sorted[i - 1]->headEnd())
            {
                return "Gap in head ranges between " +
                       std::to_string(sorted[i - 1]->headEnd()) + " and " +
                       std::to_string(sorted[i]->head_start);
            }
        }

        return ""; // No error
    }

    // =========================================================================
    // TensorParallelConfig - Factory Methods
    // =========================================================================

    TensorParallelConfig TensorParallelConfig::equalSplit(
        int world_size,
        int n_heads,
        int n_kv_heads,
        int d_ff,
        int vocab_size,
        std::optional<std::vector<DeviceId>> devices,
        KVHeadCoveragePolicy kv_coverage,
        std::optional<QueryHeadPartitionDeclaration> query_partition)
    {
        if (world_size <= 0)
        {
            throw std::invalid_argument("world_size must be positive");
        }
        if (n_heads <= 0 || n_kv_heads <= 0 || d_ff <= 0 || vocab_size <= 0)
        {
            throw std::invalid_argument("All dimension parameters must be positive");
        }
        if (n_heads < world_size)
        {
            throw std::invalid_argument("Cannot split " + std::to_string(n_heads) +
                                        " heads across " + std::to_string(world_size) + " devices");
        }

        // Create uniform fractions
        std::vector<float> fractions(world_size, 1.0f / world_size);

        // Create default devices if not provided
        std::vector<DeviceId> device_list;
        if (devices.has_value())
        {
            device_list = devices.value();
            if (static_cast<int>(device_list.size()) != world_size)
            {
                throw std::invalid_argument("devices.size() must equal world_size");
            }
        }
        else
        {
            for (int i = 0; i < world_size; ++i)
            {
                device_list.push_back(DeviceId::cuda(i));
            }
        }

        // Use proportionalSplit with equal fractions
        auto config = proportionalSplit(device_list, fractions, n_heads, n_kv_heads, d_ff, vocab_size, kv_coverage, std::move(query_partition));

        // Mark as non-proportional (equal split)
        config.is_proportional_ = false;

        return config;
    }

    TensorParallelConfig TensorParallelConfig::proportionalSplit(
        const std::vector<DeviceId> &devices,
        const std::vector<float> &work_fractions,
        int n_heads,
        int n_kv_heads,
        int d_ff,
        int vocab_size,
        KVHeadCoveragePolicy kv_coverage,
        std::optional<QueryHeadPartitionDeclaration> query_partition)
    {
        if (devices.size() != work_fractions.size())
        {
            throw std::invalid_argument("devices.size() != work_fractions.size()");
        }
        if (devices.empty())
        {
            throw std::invalid_argument("devices must not be empty");
        }
        if (n_heads <= 0 || n_kv_heads <= 0 || d_ff <= 0 || vocab_size <= 0)
        {
            throw std::invalid_argument("All dimension parameters must be positive");
        }

        if (kv_coverage != KVHeadCoveragePolicy::Legacy && kv_coverage != KVHeadCoveragePolicy::QueryInterval)
            throw std::invalid_argument("Unknown KV head coverage policy");
        if (kv_coverage == KVHeadCoveragePolicy::QueryInterval)
            validateKVHeadCoverageDeclaration({n_heads, n_kv_heads});

        if (query_partition && (query_partition->originalQueryHeads() != n_heads ||
            static_cast<size_t>(n_heads / query_partition->granularity()) < devices.size()))
            throw std::invalid_argument("Query consumer source differs or has too few complete groups for participants");

        // Validate work fractions
        for (float f : work_fractions)
        {
            if (f < 0.0f || ((kv_coverage == KVHeadCoveragePolicy::QueryInterval || query_partition.has_value()) && !std::isfinite(f)))
            {
                throw std::invalid_argument("Work fractions must be non-negative");
            }
        }

        // Normalize work fractions to sum to 1.0
        float sum = std::accumulate(work_fractions.begin(), work_fractions.end(), 0.0f);
        if (sum <= 0.0f || ((kv_coverage == KVHeadCoveragePolicy::QueryInterval || query_partition.has_value()) && !std::isfinite(sum)))
        {
            throw std::invalid_argument("Sum of work fractions must be positive");
        }

        std::vector<float> normalized(work_fractions.size());
        for (size_t i = 0; i < work_fractions.size(); ++i)
        {
            normalized[i] = work_fractions[i] / sum;
        }

        // Distribute heads, KV heads, d_ff, and vocab proportionally

        // GQA-aware KV head distribution: when n_kv_heads < tp_degree,
        // replicate KV heads on all devices instead of column-parallel sharding.
        // This is the standard Megatron-LM approach for GQA models where
        // KV head count is less than the tensor parallelism degree.
        const bool kv_replicated = n_kv_heads < static_cast<int>(devices.size());
        std::vector<int> kv_head_counts;
        if (kv_replicated)
        {
            kv_head_counts.assign(devices.size(), n_kv_heads);
            // Candidate enumeration invokes this splitter for every optional
            // width.  Replication is expected geometry, not an operator event;
            // retain the useful diagnosis without flooding normal planning
            // output once per candidate.
            LOG_DEBUG("[TensorParallelConfig] GQA replication: n_kv_heads=" << n_kv_heads
                                                                            << " < tp_degree=" << devices.size()
                                                                            << ", replicating K/V on all devices");
        }
        else
        {
            kv_head_counts = distributeProportionally(n_kv_heads, normalized, 1);
        }

        // Replicated KV retains the original global GQA ratio and query-head
        // offset. Local Q widths therefore need no KV-count alignment: imposing
        // it turns QFN TP8's exact three-head intervals into nonintegral GDN
        // partitions. GDN admission still independently rejects cut key heads.
        std::vector<int> head_counts = distributeProportionally(n_heads, normalized,
            query_partition ? query_partition->granularity() : 1);
        std::vector<int> d_ff_counts = distributeProportionally(d_ff, normalized, 32); // 32-aligned
        std::vector<int> vocab_counts = distributeProportionally(vocab_size, normalized, 1);

        // Build assignments
        std::vector<DeviceShardingAssignment> assignments;
        int head_offset = 0;
        int kv_offset = 0;
        int d_ff_offset = 0;
        int vocab_offset = 0;

        for (size_t i = 0; i < devices.size(); ++i)
        {
            DeviceShardingAssignment a;
            a.device = devices[i];
            a.local_rank = static_cast<int>(i);
            a.work_fraction = normalized[i];

            a.head_start = head_offset;
            a.head_count = head_counts[i];
            head_offset += head_counts[i];

            // When KV is replicated, all devices start at 0 and have all heads
            a.kv_head_start = kv_replicated ? 0 : kv_offset;
            a.kv_head_count = kv_head_counts[i];
            if (kv_coverage == KVHeadCoveragePolicy::QueryInterval)
            {
                const auto interval = resolveQueryKVHeadCoverage({n_heads, n_kv_heads}, a.head_start, a.head_count);
                a.kv_head_start = interval.first;
                a.kv_head_count = interval.count;
            }
            if (!kv_replicated)
                kv_offset += kv_head_counts[i];

            a.d_ff_start = d_ff_offset;
            a.d_ff_count = d_ff_counts[i];
            d_ff_offset += d_ff_counts[i];

            a.vocab_start = vocab_offset;
            a.vocab_count = vocab_counts[i];
            vocab_offset += vocab_counts[i];

            assignments.push_back(a);
        }

        TensorParallelConfig config = kv_coverage == KVHeadCoveragePolicy::QueryInterval
            ? TensorParallelConfig(std::move(assignments), KVHeadCoverageDeclaration{n_heads, n_kv_heads}, std::move(query_partition))
            : query_partition ? TensorParallelConfig(std::move(assignments), std::move(*query_partition))
                              : TensorParallelConfig(std::move(assignments));
        config.is_proportional_ = true;
        return config;
    }

    TensorParallelConfig TensorParallelConfig::singleDevice(
        DeviceId device,
        int n_heads,
        int n_kv_heads,
        int d_ff,
        int vocab_size)
    {
        DeviceShardingAssignment a;
        a.device = device;
        a.local_rank = 0;
        a.work_fraction = 1.0f;
        a.head_start = 0;
        a.head_count = n_heads;
        a.kv_head_start = 0;
        a.kv_head_count = n_kv_heads;
        a.d_ff_start = 0;
        a.d_ff_count = d_ff;
        a.vocab_start = 0;
        a.vocab_count = vocab_size;

        return TensorParallelConfig({a});
    }

    TensorParallelConfig TensorParallelConfig::fromLocalTPContext(
        const ILocalTPContext &local_tp_ctx,
        int n_heads,
        int n_kv_heads,
        int d_ff,
        int vocab_size,
        KVHeadCoveragePolicy kv_coverage,
        std::optional<QueryHeadPartitionDeclaration> query_partition)
    {
        const auto &devices = local_tp_ctx.devices();
        const auto &weights = local_tp_ctx.weights();
        int degree = local_tp_ctx.degree();

        if (degree <= 0)
        {
            throw std::invalid_argument("LOCAL TP degree must be positive");
        }
        if (static_cast<int>(devices.size()) != degree)
        {
            throw std::invalid_argument(
                "Device count (" + std::to_string(devices.size()) +
                ") does not match degree (" + std::to_string(degree) + ")");
        }
        if (n_heads <= 0 || n_kv_heads <= 0 || d_ff <= 0 || vocab_size <= 0)
        {
            throw std::invalid_argument("All dimension parameters must be positive");
        }

        // Convert GlobalDeviceAddress to DeviceId
        std::vector<DeviceId> device_ids;
        device_ids.reserve(degree);
        for (const auto &addr : devices)
        {
            device_ids.push_back(addr.toLocalDeviceId());
        }

        // Use provided weights or create equal weights if empty
        std::vector<float> work_fractions;
        if (weights.empty())
        {
            // Equal split across all devices
            work_fractions.assign(degree, 1.0f / static_cast<float>(degree));
        }
        else
        {
            if (static_cast<int>(weights.size()) != degree)
            {
                throw std::invalid_argument(
                    "Weights count (" + std::to_string(weights.size()) +
                    ") does not match degree (" + std::to_string(degree) + ")");
            }
            work_fractions = weights;
        }

        // Delegate to proportionalSplit which handles the distribution logic
        return proportionalSplit(device_ids, work_fractions, n_heads, n_kv_heads, d_ff, vocab_size, kv_coverage, std::move(query_partition));
    }

    // =========================================================================
    // TensorParallelConfig - Private Helpers
    // =========================================================================

    int TensorParallelConfig::roundToAlignment(int value, int alignment)
    {
        if (alignment <= 1)
            return value;
        return (value / alignment) * alignment;
    }

    std::vector<int> TensorParallelConfig::distributeProportionally(
        int total,
        const std::vector<float> &fractions,
        int alignment)
    {
        size_t n = fractions.size();
        std::vector<int> result(n, 0);

        if (n == 0 || total <= 0)
        {
            return result;
        }

        // First pass: assign floor of proportional amounts (with alignment)
        int assigned = 0;
        for (size_t i = 0; i < n; ++i)
        {
            int ideal = static_cast<int>(std::floor(fractions[i] * total));
            if (alignment > 1)
            {
                ideal = roundToAlignment(ideal, alignment);
            }
            result[i] = ideal;
            assigned += ideal;
        }

        // Second pass: distribute remainder to devices with largest fractional parts
        // But respect alignment - only add alignment-sized chunks
        int remainder = total - assigned;

        if (alignment > 1)
        {
            // With alignment, we need to be more careful
            // Distribute full alignment chunks, prioritizing devices by fraction
            while (remainder >= alignment)
            {
                // Find device with smallest current assignment relative to its fraction
                size_t best_idx = 0;
                float best_deficit = -1.0f;
                for (size_t i = 0; i < n; ++i)
                {
                    float expected = fractions[i] * total;
                    float deficit = expected - result[i];
                    if (deficit > best_deficit)
                    {
                        best_deficit = deficit;
                        best_idx = i;
                    }
                }
                result[best_idx] += alignment;
                remainder -= alignment;
            }

            // Handle any sub-alignment remainder by giving to first device
            // This ensures we don't lose elements
            if (remainder > 0)
            {
                result[0] += remainder;
            }
        }
        else
        {
            // No alignment constraint - distribute one at a time
            // Sort indices by fractional part (descending)
            std::vector<std::pair<float, size_t>> fractional_parts;
            for (size_t i = 0; i < n; ++i)
            {
                float ideal = fractions[i] * total;
                float frac = ideal - std::floor(ideal);
                fractional_parts.push_back({frac, i});
            }
            std::sort(fractional_parts.begin(), fractional_parts.end(),
                      [](const auto &a, const auto &b)
                      { return a.first > b.first; });

            // Assign remainder to devices with largest fractional parts
            for (int r = 0; r < remainder && r < static_cast<int>(n); ++r)
            {
                result[fractional_parts[r].second]++;
            }
        }

        return result;
    }

    std::string TensorParallelConfig::toString() const
    {
        std::ostringstream oss;
        oss << "TensorParallelConfig(world_size=" << worldSize()
            << ", proportional=" << (is_proportional_ ? "true" : "false")
            << ", heads=" << total_heads_
            << ", kv_heads=" << total_kv_heads_
            << ", d_ff=" << total_d_ff_
            << ", vocab=" << total_vocab_ << "):\n";
        for (const auto &a : assignments_)
        {
            oss << "  " << a.toString() << "\n";
        }
        return oss.str();
    }

} // namespace llaminar2
