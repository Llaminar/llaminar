/**
 * @file GpuKVCacheGroupedVerifierHarness.h
 * @brief Shared all-format proof harness for grouped GPU KV publication.
 *
 * MTP verifies several future rows concurrently.  Publishing those K/V rows
 * is correct only when the native cache payload and ring metadata are exactly
 * what M independent serial decode appends would have produced.  This harness
 * drives the public production cache interface through the graph-captured,
 * device-resident publication route.  There is deliberately no static-head
 * variant: a host-selected ring head would test an architecture that
 * production GPU inference no longer permits.
 *
 * The backend test translation units supply only graph runtime mechanics.  All
 * format, layout, depth, topology, byte comparison, and perf-counter policy is
 * centralized here so CUDA and ROCm cannot silently drift to different test
 * matrices.
 */

#pragma once

#include <gtest/gtest.h>

#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/IKVCache.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "utils/TestTensorFactory.h"
#include "utils/VerifierRowTestInventory.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test::gpu_kv_verifier
{
    /** @brief One cache precision and accepted source-tensor combination. */
    struct FormatCase
    {
        ActivationPrecision cache_precision;
        TensorType source_k_type;
        TensorType source_v_type;
        const char *cache_label;
        const char *source_label;
    };

    /**
     * @brief Complete source-conversion surface advertised by GPU ring caches.
     *
     * FP32 and BF16 caches accept their native source.  FP16 and Q8_1 caches
     * additionally expose production conversion kernels for every activation
     * tensor format.  The asymmetric TurboQuant cache accepts either FP32
     * projection rows for fused quantize-to-ring publication or already
     * prepared TQ8 K plus TQ4 V rows for direct device-to-device publication.
     */
    inline constexpr std::array<FormatCase, 12> kFormatCases = {{
        {ActivationPrecision::FP32, TensorType::FP32, TensorType::FP32, "FP32", "FP32"},
        {ActivationPrecision::BF16, TensorType::BF16, TensorType::BF16, "BF16", "BF16"},
        {ActivationPrecision::FP16, TensorType::FP32, TensorType::FP32, "FP16", "FP32"},
        {ActivationPrecision::FP16, TensorType::FP16, TensorType::FP16, "FP16", "FP16"},
        {ActivationPrecision::FP16, TensorType::BF16, TensorType::BF16, "FP16", "BF16"},
        {ActivationPrecision::FP16, TensorType::Q8_1, TensorType::Q8_1, "FP16", "Q8_1"},
        {ActivationPrecision::Q8_1, TensorType::FP32, TensorType::FP32, "Q8_1", "FP32"},
        {ActivationPrecision::Q8_1, TensorType::FP16, TensorType::FP16, "Q8_1", "FP16"},
        {ActivationPrecision::Q8_1, TensorType::BF16, TensorType::BF16, "Q8_1", "BF16"},
        {ActivationPrecision::Q8_1, TensorType::Q8_1, TensorType::Q8_1, "Q8_1", "Q8_1"},
        {ActivationPrecision::TQ8, TensorType::FP32, TensorType::FP32, "TQ8/TQ4", "FP32"},
        {ActivationPrecision::TQ8, TensorType::TQ8, TensorType::TQ4, "TQ8/TQ4", "TQ8/TQ4"},
    }};

    /** @brief One native cache family and a lossless append source for read tests. */
    struct CacheReadFormatCase
    {
        ActivationPrecision cache_precision;
        TensorType append_type;
        const char *label;
    };

    /**
     * @brief Every GPU cache family consumed by production attention.
     *
     * TQ append uses FP32 projections because compressed K and V have different
     * native types. The other families use their own storage format so the read
     * proof is independent of append-conversion coverage above.
     */
    inline constexpr std::array<CacheReadFormatCase, 5> kCacheReadFormats = {{
        {ActivationPrecision::FP32, TensorType::FP32, "FP32"},
        {ActivationPrecision::FP16, TensorType::FP16, "FP16"},
        {ActivationPrecision::BF16, TensorType::BF16, "BF16"},
        {ActivationPrecision::Q8_1, TensorType::Q8_1, "Q8_1"},
        {ActivationPrecision::TQ8, TensorType::FP32, "TQ8/TQ4"},
    }};

    /** @brief Replicated and LocalTP-sharded factory configurations. */
    struct CacheTopology
    {
        const char *label;
        int total_heads;
        int local_heads;
        int head_start;

        int effectiveLocalHeads() const
        {
            return local_heads > 0 ? local_heads : total_heads;
        }
    };

    inline constexpr std::array<CacheTopology, 2> kTopologies = {{
        {"replicated", 2, 0, 0},
        {"local_tp_shard", 4, 2, 2},
    }};

    /** @brief Enable structured route counters for the lifetime of one test. */
    class ScopedPerfStats
    {
    public:
        ScopedPerfStats()
        {
            if (const char *old = std::getenv("LLAMINAR_PERF_STATS_SUMMARY"))
            {
                had_old_value_ = true;
                old_value_ = old;
            }
            setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ~ScopedPerfStats()
        {
            if (had_old_value_)
                setenv("LLAMINAR_PERF_STATS_SUMMARY", old_value_.c_str(), 1);
            else
                unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ScopedPerfStats(const ScopedPerfStats &) = delete;
        ScopedPerfStats &operator=(const ScopedPerfStats &) = delete;

    private:
        bool had_old_value_ = false;
        std::string old_value_;
    };

    /** @brief Own a production cache and any declared graph workspace. */
    struct BoundCache
    {
        // Destruction is reverse declaration order: release the cache before
        // the workspace whose pointers it borrowed.
        std::unique_ptr<DeviceWorkspaceManager> workspace;
        std::unique_ptr<IKVCache> cache;
    };

    /** @brief Allocate a deterministic native activation tensor. */
    inline std::unique_ptr<ITensor> makeTensor(
        TensorType type,
        const std::vector<size_t> &shape,
        uint32_t seed,
        int head_dim)
    {
        switch (type)
        {
        case TensorType::FP32:
            return TestTensorFactory::createFP32Random(shape, -0.75f, 0.75f, seed);
        case TensorType::FP16:
            return TestTensorFactory::createFP16Random(shape, -0.75f, 0.75f, seed);
        case TensorType::BF16:
            return TestTensorFactory::createBF16Random(shape, -0.75f, 0.75f, seed);
        case TensorType::Q8_1:
            return TestTensorFactory::createQ8_1Random(shape, -0.75f, 0.75f, seed);
        case TensorType::TQ8:
        case TensorType::TQ4:
        {
            std::unique_ptr<ITensor> tensor;
            if (type == TensorType::TQ8)
                tensor = std::make_unique<TQ8Tensor>(shape, head_dim);
            else
                tensor = std::make_unique<TQ4Tensor>(shape, head_dim);

            // Prepared TQ publication is a native-byte transport contract; it
            // must preserve every payload byte without dequantizing it.  Fill
            // all block metadata and codes with deterministic non-zero bytes
            // so an offset, stride, or partial-copy bug is immediately visible.
            auto *bytes = static_cast<uint8_t *>(tensor->raw_mutable_data());
            uint32_t state = seed == 0 ? 0x9E3779B9u : seed;
            for (size_t index = 0; index < tensor->size_bytes(); ++index)
            {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                bytes[index] = static_cast<uint8_t>(state & 0xFFu);
            }
            return tensor;
        }
        default:
            throw std::invalid_argument("Unsupported GPU KV verifier source tensor format");
        }
    }

    /**
     * @brief Extract one verifier position into a serial position-major tensor.
     *
     * Native bytes are copied instead of dequantizing and requantizing.  Thus
     * the serial oracle sees exactly the same source values and quantization
     * metadata as the grouped call, including Q8_1 scales and integer sums.
     */
    inline std::unique_ptr<ITensor> extractPositionRow(
        const ITensor &source,
        TensorType source_type,
        int verifier_row,
        int verifier_rows,
        int local_heads,
        int head_dim,
        bool source_head_major,
        uint32_t seed)
    {
        const int kv_dim = local_heads * head_dim;
        auto row = makeTensor(
            source_type, {1, static_cast<size_t>(kv_dim)}, seed, head_dim);
        const auto *src = static_cast<const uint8_t *>(source.raw_data());
        auto *dst = static_cast<uint8_t *>(row->raw_mutable_data());
        if (!src || !dst)
            throw std::runtime_error("GPU KV verifier tensor did not expose native host storage");

        if (source.shape().empty() || source.shape()[0] == 0)
            throw std::runtime_error("GPU KV verifier source has no physical rows");
        const size_t destination_row_bytes = row->size_bytes();
        const size_t source_physical_row_bytes =
            source.size_bytes() / source.shape()[0];
        if (!source_head_major)
        {
            if (source_physical_row_bytes != destination_row_bytes)
                throw std::runtime_error("Position-major verifier source row has an unexpected native stride");
            std::memcpy(dst,
                        src + static_cast<size_t>(verifier_row) * source_physical_row_bytes,
                        destination_row_bytes);
            return row;
        }

        const size_t head_bytes = source_physical_row_bytes;
        if (destination_row_bytes != static_cast<size_t>(local_heads) * head_bytes)
            throw std::runtime_error("Verifier source head is not aligned to native storage blocks");
        for (int head = 0; head < local_heads; ++head)
        {
            const size_t source_row = static_cast<size_t>(head) * verifier_rows + verifier_row;
            std::memcpy(dst + static_cast<size_t>(head) * head_bytes,
                        src + source_row * head_bytes,
                        head_bytes);
        }
        return row;
    }

    /** @brief Upload a concrete TensorBase on the test's mandatory stream. */
    inline void ensureOnDevice(ITensor *tensor, DeviceId device, void *stream)
    {
        auto *base = dynamic_cast<TensorBase *>(tensor);
        ASSERT_NE(base, nullptr);
        ASSERT_TRUE(base->ensureOnDevice(device, stream));
        ASSERT_NE(base->gpu_data_ptr(), nullptr);
    }

    /** @brief Instantiate the same cache through the production KernelFactory. */
    inline BoundCache makeBoundCache(
        DeviceId device,
        ActivationPrecision precision,
        const CacheTopology &topology,
        int max_seq_len,
        int head_dim,
        const TurboQuantContext *tq_context,
        int batch_size = 1)
    {
        using Factory = llaminar::v2::kernels::KernelFactory;
        using Config = llaminar::v2::kernels::KVCacheConfig;

        Config config{
            .precision = precision,
            .device = device,
            .num_layers = 1,
            .batch_size = batch_size,
            .max_seq_len = max_seq_len,
            .n_kv_heads = topology.total_heads,
            .head_dim = head_dim,
            .local_n_kv_heads = topology.local_heads,
            .kv_head_start = topology.head_start,
            .turboquant_ctx = tq_context,
        };

        BoundCache bound;
        bound.cache = Factory::createKVCache(config);
        if (!bound.cache)
            throw std::runtime_error("KernelFactory returned a null GPU KV cache");

        if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(bound.cache.get()))
        {
            const auto requirements = consumer->getWorkspaceRequirements(
                max_seq_len, batch_size, 0);
            const size_t budget = requirements.total_bytes_with_alignment() + 4096;
            bound.workspace = std::make_unique<DeviceWorkspaceManager>(device, budget);
            if (!bound.workspace->allocate(requirements))
                throw std::runtime_error("Failed to allocate declared GPU KV cache workspace");
            consumer->bindWorkspace(bound.workspace.get());
            if (!consumer->hasWorkspace())
                throw std::runtime_error("GPU KV cache rejected its declared workspace");
        }
        return bound;
    }

    /** @brief Stable route key shared by expected and observed counter matrices. */
    inline std::string routeKey(
        const std::string &cache_format,
        const std::string &source_format,
        int verifier_rows,
        int head_dim,
        const std::string &source_layout,
        const std::string &execution_mode,
        const std::string &topology)
    {
        std::ostringstream out;
        out << cache_format << '|' << source_format << "|M" << verifier_rows
            << "|D" << head_dim
            << '|' << source_layout << '|' << execution_mode << '|' << topology;
        return out.str();
    }

    /** @brief Read one required tag or return an unmistakable missing marker. */
    inline std::string tag(const PerfStatRecord &record, const char *name)
    {
        const auto it = record.tags.find(name);
        return it == record.tags.end() ? std::string("<missing:") + name + '>' : it->second;
    }

    /**
     * @brief Run the complete production grouped KV publication proof.
     *
     * `grouped_append` owns backend graph API calls and must execute the native
     * grouped publication inside a captured graph. `observe_state` is an
     * integration-test-only observation boundary: after the graph stream is
     * fenced, it copies the canonical device head/count into the supplied
     * value. It must never consult or update a cache-owned host mirror.
     */
    template <typename GroupedAppender, typename DeviceStateObserver>
    void runAllFormatGroupedVerifierSweep(
        DeviceId device,
        const char *backend_label,
        const char *counter_name,
        void *stream,
        GroupedAppender &&grouped_append,
        DeviceStateObserver &&observe_state)
    {
        constexpr int max_seq_len = kGroupedVerifierRuntimeRows.back();
        constexpr int prefix_rows = max_seq_len - 2;
        constexpr std::array<int, 2> head_dims = {64, 128};
        constexpr std::array<bool, 2> source_layouts = {false, true};
        ScopedPerfStats perfstats;
        ASSERT_TRUE(PerfStatsCollector::isEnabled());
        std::set<std::string> expected_routes;
        uint32_t seed = 1001;

        for (const auto &format : kFormatCases)
        {
            for (const int head_dim : head_dims)
            {
                TurboQuantContext tq_context(head_dim, 0xC0FFEEu);
                for (const auto &topology : kTopologies)
                {
                    const int local_heads = topology.effectiveLocalHeads();
                    const int kv_dim = local_heads * head_dim;
                    for (const bool source_head_major : source_layouts)
                    {
                        for (const int verifier_rows : kGroupedVerifierRuntimeRows)
                        {
                            const char *layout_label = source_head_major
                                                           ? "head_major"
                                                           : "position_major";
                            constexpr const char *execution_mode = "graph_captured";
                            SCOPED_TRACE(std::string(backend_label) +
                                         " cache=" + format.cache_label +
                                         " source=" + format.source_label +
                                         " topology=" + topology.label +
                                         " layout=" + layout_label +
                                         " M=" + std::to_string(verifier_rows) +
                                         " D=" + std::to_string(head_dim) +
                                         " execution=" + execution_mode);

                            auto serial = makeBoundCache(
                                device, format.cache_precision, topology,
                                max_seq_len, head_dim, &tq_context);
                            auto grouped = makeBoundCache(
                                device, format.cache_precision, topology,
                                max_seq_len, head_dim, &tq_context);

                            // The prefix leaves exactly two free positions.
                            // M=2 fills the ring, while every larger certified
                            // depth wraps without assigning two source rows to
                            // the same destination slot in one grouped launch.
                            const TensorType history_type =
                                format.cache_precision == ActivationPrecision::TQ8
                                    ? TensorType::FP32
                                    : (format.cache_precision == ActivationPrecision::FP32
                                           ? TensorType::FP32
                                           : format.cache_precision == ActivationPrecision::BF16
                                                 ? TensorType::BF16
                                                 : format.cache_precision == ActivationPrecision::FP16
                                                       ? TensorType::FP16
                                                       : TensorType::Q8_1);
                            auto history_k = makeTensor(
                                history_type,
                                {prefix_rows, static_cast<size_t>(kv_dim)}, seed++, head_dim);
                            auto history_v = makeTensor(
                                history_type,
                                {prefix_rows, static_cast<size_t>(kv_dim)}, seed++, head_dim);
                            ensureOnDevice(history_k.get(), device, stream);
                            ensureOnDevice(history_v.get(), device, stream);
                            ASSERT_TRUE(serial.cache->appendWithStream(
                                0, 0, history_k.get(), history_v.get(), prefix_rows, stream));
                            ASSERT_TRUE(grouped.cache->appendWithStream(
                                0, 0, history_k.get(), history_v.get(), prefix_rows, stream));

                            const size_t source_rows = source_head_major
                                                           ? static_cast<size_t>(local_heads * verifier_rows)
                                                           : static_cast<size_t>(verifier_rows);
                            const size_t source_cols = source_head_major
                                                           ? static_cast<size_t>(head_dim)
                                                           : static_cast<size_t>(kv_dim);
                            auto verifier_k = makeTensor(
                                format.source_k_type, {source_rows, source_cols}, seed++, head_dim);
                            auto verifier_v = makeTensor(
                                format.source_v_type, {source_rows, source_cols}, seed++, head_dim);
                            ensureOnDevice(verifier_k.get(), device, stream);
                            ensureOnDevice(verifier_v.get(), device, stream);

                            std::vector<std::unique_ptr<ITensor>> serial_k_rows;
                            std::vector<std::unique_ptr<ITensor>> serial_v_rows;
                            serial_k_rows.reserve(verifier_rows);
                            serial_v_rows.reserve(verifier_rows);
                            for (int row_index = 0; row_index < verifier_rows; ++row_index)
                            {
                                auto k_row = extractPositionRow(
                                    *verifier_k, format.source_k_type, row_index,
                                    verifier_rows, local_heads, head_dim,
                                    source_head_major, seed++);
                                auto v_row = extractPositionRow(
                                    *verifier_v, format.source_v_type, row_index,
                                    verifier_rows, local_heads, head_dim,
                                    source_head_major, seed++);
                                ensureOnDevice(k_row.get(), device, stream);
                                ensureOnDevice(v_row.get(), device, stream);
                                ASSERT_TRUE(serial.cache->appendWithStream(
                                    0, 0, k_row.get(), v_row.get(), 1, stream));
                                serial_k_rows.push_back(std::move(k_row));
                                serial_v_rows.push_back(std::move(v_row));
                            }

                            ASSERT_TRUE(grouped_append(
                                *grouped.cache, verifier_k.get(), verifier_v.get(),
                                verifier_rows, stream));

                            IKVCache::KVCacheSequenceState serial_state;
                            IKVCache::KVCacheSequenceState grouped_state;
                            ASSERT_TRUE(observe_state(
                                *serial.cache, max_seq_len, stream, &serial_state));
                            ASSERT_TRUE(observe_state(
                                *grouped.cache, max_seq_len, stream, &grouped_state));
                            EXPECT_EQ(grouped_state.cached_tokens, serial_state.cached_tokens);
                            EXPECT_EQ(grouped_state.implementation_head,
                                      serial_state.implementation_head);
                            EXPECT_EQ(grouped_state.wrapped, serial_state.wrapped);

                            const auto layout = serial.cache->logicalBlockLayout(
                                0, serial_state.cached_tokens);
                            ASSERT_GT(layout.k_bytes, 0u);
                            ASSERT_GT(layout.v_bytes, 0u);
                            EXPECT_EQ(layout.local_kv_heads, local_heads);
                            EXPECT_EQ(layout.kv_head_start, topology.head_start);
                            std::vector<uint8_t> serial_k(layout.k_bytes, 0);
                            std::vector<uint8_t> serial_v(layout.v_bytes, 0);
                            std::vector<uint8_t> grouped_k(layout.k_bytes, 0);
                            std::vector<uint8_t> grouped_v(layout.v_bytes, 0);
                            const IKVCache::KVCacheLogicalBlockDescriptor descriptor{
                                .layer = 0,
                                .seq_idx = 0,
                                .logical_token_start = 0,
                                .token_count = serial_state.cached_tokens,
                                .stream = stream,
                            };
                            ASSERT_TRUE(serial.cache->exportLogicalBlock(
                                descriptor, serial_k.data(), serial_v.data()));
                            ASSERT_TRUE(grouped.cache->exportLogicalBlock(
                                descriptor, grouped_k.data(), grouped_v.data()));
                            EXPECT_EQ(grouped_k, serial_k)
                                << "Grouped K cache payload is not serial-decode byte exact";
                            EXPECT_EQ(grouped_v, serial_v)
                                << "Grouped V cache payload is not serial-decode byte exact";

                            // Prefix-cache restore consumes the same native
                            // payload contract as accepted-state checkpointing.
                            // Round-trip every matrix cell through a fresh
                            // production cache so newly supported formats (in
                            // particular asymmetric TQ and LocalTP shards) do
                            // not pass publication while remaining unrestorable.
                            auto restored = makeBoundCache(
                                device, format.cache_precision, topology,
                                max_seq_len, head_dim, &tq_context);
                            ASSERT_TRUE(restored.cache->importLogicalBlock(
                                descriptor, serial_k.data(), serial_v.data()));
                            IKVCache::KVCacheSequenceState restored_state;
                            ASSERT_TRUE(observe_state(
                                *restored.cache, max_seq_len, stream, &restored_state));
                            EXPECT_EQ(restored_state.cached_tokens,
                                      serial_state.cached_tokens);
                            std::vector<uint8_t> restored_k(layout.k_bytes, 0);
                            std::vector<uint8_t> restored_v(layout.v_bytes, 0);
                            ASSERT_TRUE(restored.cache->exportLogicalBlock(
                                descriptor, restored_k.data(), restored_v.data()));
                            EXPECT_EQ(restored_k, serial_k)
                                << "Prefix restore changed native K payload bytes";
                            EXPECT_EQ(restored_v, serial_v)
                                << "Prefix restore changed native V payload bytes";

                            expected_routes.insert(routeKey(
                                format.cache_label, format.source_label,
                                verifier_rows, head_dim,
                                layout_label, execution_mode, topology.label));
                        }
                    }
                }
            }
        }

        std::set<std::string> observed_routes;
        uint64_t total_calls = 0;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {std::string("kernel.") + counter_name}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter)
                continue;
            EXPECT_EQ(tag(record, "source_k_layout"), tag(record, "source_v_layout"));
            EXPECT_EQ(tag(record, "commit_policy"), "single_grouped_metadata_commit");
            const std::string source_k_format = tag(record, "source_k_format");
            const std::string source_v_format = tag(record, "source_v_format");
            const std::string source_format = source_k_format == source_v_format
                                                  ? source_k_format
                                                  : source_k_format + '/' + source_v_format;
            observed_routes.insert(routeKey(
                tag(record, "cache_format"), source_format,
                std::stoi(tag(record, "verifier_rows")),
                std::stoi(tag(record, "head_dim")),
                tag(record, "source_k_layout"), tag(record, "execution_mode"),
                tag(record, "topology")));
            total_calls += record.count;
        }

        EXPECT_EQ(observed_routes, expected_routes)
            << PerfStatsCollector::summaryString(
                   {std::string("kernel.") + counter_name}, 300);
        EXPECT_EQ(total_calls, expected_routes.size())
            << "Every matrix cell must enter the grouped implementation exactly once\n"
            << PerfStatsCollector::summaryString(
                   {std::string("kernel.") + counter_name}, 300);
    }

    /**
     * @brief Prove the production converted cache read is batch invariant.
     *
     * Each matrix cell creates two request-local rings. Request zero wraps and
     * request one remains short, then the captured grouped path materializes
     * both in one launch sequence. Its live FP16 words must equal independent
     * scalar reads byte-for-byte and every padded word must be zero. Full and
     * partial RoPE dimensions catch both request-position leakage and incorrect
     * conversion/RoPE ordering.
     */
    template <typename CapturedReader, typename DeviceByteCopier,
              typename StreamSynchronizer>
    void runAllFormatConvertedDeviceReadSweep(
        DeviceId device,
        const char *backend_label,
        void *stream,
        CapturedReader &&captured_read,
        DeviceByteCopier &&copy_device_bytes,
        StreamSynchronizer &&synchronize_stream)
    {
        constexpr int batch_size = 2;
        constexpr int max_seq_len = 6;
        constexpr std::array<int, batch_size> expected_counts = {6, 4};
        constexpr std::array<int, 2> head_dims = {64, 128};
        uint32_t seed = 0xA11CEu;

        for (const auto &format : kCacheReadFormats)
        {
            for (const int head_dim : head_dims)
            {
                TurboQuantContext tq_context(head_dim, 0xC0FFEEu);
                for (const auto &topology : kTopologies)
                {
                    const int local_heads = topology.effectiveLocalHeads();
                    const int kv_dim = local_heads * head_dim;
                    for (const int rope_dim : {head_dim, head_dim / 2})
                    {
                        SCOPED_TRACE(std::string(backend_label) +
                                     " converted cache=" + format.label +
                                     " topology=" + topology.label +
                                     " D=" + std::to_string(head_dim) +
                                     " rope_dim=" + std::to_string(rope_dim));
                        auto cache = makeBoundCache(
                            device, format.cache_precision, topology,
                            max_seq_len, head_dim, &tq_context, batch_size);
                        std::vector<std::unique_ptr<ITensor>> append_lifetime;

                        auto append_chunk = [&](int request, int rows)
                        {
                            auto k = makeTensor(
                                format.append_type,
                                {static_cast<size_t>(rows),
                                 static_cast<size_t>(kv_dim)},
                                seed++, head_dim);
                            auto v = makeTensor(
                                format.append_type,
                                {static_cast<size_t>(rows),
                                 static_cast<size_t>(kv_dim)},
                                seed++, head_dim);
                            ensureOnDevice(k.get(), device, stream);
                            ensureOnDevice(v.get(), device, stream);
                            ASSERT_TRUE(cache.cache->appendWithStream(
                                0, request, k.get(), v.get(), rows, stream));
                            append_lifetime.push_back(std::move(k));
                            append_lifetime.push_back(std::move(v));
                        };

                        append_chunk(/*request=*/0, /*rows=*/5);
                        append_chunk(/*request=*/0, /*rows=*/4);
                        append_chunk(/*request=*/1, /*rows=*/4);
                        ASSERT_TRUE(synchronize_stream(stream));

                        IKVCache::KVReadParams read{
                            .rope_theta = 1000000.0f,
                            .position_start = 0,
                            .n_kv_heads = local_heads,
                            .head_dim = head_dim,
                            .rope_dim = rope_dim,
                            .requested_token_count = 0,
                            .turboquant_ctx = &tq_context,
                            .gpu_stream = stream,
                        };
                        std::array<std::vector<uint16_t>, batch_size> expected_k;
                        std::array<std::vector<uint16_t>, batch_size> expected_v;
                        for (int request = 0; request < batch_size; ++request)
                        {
                            ITensor *serial_k = nullptr;
                            ITensor *serial_v = nullptr;
                            int serial_count = 0;
                            ASSERT_TRUE(cache.cache->get_kv_converted(
                                0, request, ActivationPrecision::FP16,
                                &serial_k, &serial_v, &serial_count, &read));
                            ASSERT_EQ(serial_count, expected_counts[request]);
                            ASSERT_NE(serial_k, nullptr);
                            ASSERT_NE(serial_v, nullptr);
                            ASSERT_TRUE(synchronize_stream(stream));
                            const size_t live_words =
                                static_cast<size_t>(serial_count) * kv_dim;
                            expected_k[request].resize(live_words);
                            expected_v[request].resize(live_words);
                            ASSERT_TRUE(copy_device_bytes(
                                expected_k[request].data(),
                                serial_k->gpu_data_ptr(),
                                live_words * sizeof(uint16_t), stream));
                            ASSERT_TRUE(copy_device_bytes(
                                expected_v[request].data(),
                                serial_v->gpu_data_ptr(),
                                live_words * sizeof(uint16_t), stream));
                        }

                        ITensor *grouped_k = nullptr;
                        ITensor *grouped_v = nullptr;
                        ASSERT_TRUE(captured_read(
                            *cache.cache, batch_size, max_seq_len, read,
                            &grouped_k, &grouped_v, stream));
                        ASSERT_NE(grouped_k, nullptr);
                        ASSERT_NE(grouped_v, nullptr);
                        const size_t request_words =
                            static_cast<size_t>(max_seq_len) * kv_dim;
                        const size_t grouped_words = batch_size * request_words;
                        std::vector<uint16_t> actual_k(grouped_words);
                        std::vector<uint16_t> actual_v(grouped_words);
                        ASSERT_TRUE(copy_device_bytes(
                            actual_k.data(), grouped_k->gpu_data_ptr(),
                            grouped_words * sizeof(uint16_t), stream));
                        ASSERT_TRUE(copy_device_bytes(
                            actual_v.data(), grouped_v->gpu_data_ptr(),
                            grouped_words * sizeof(uint16_t), stream));

                        for (int request = 0; request < batch_size; ++request)
                        {
                            const size_t output_offset =
                                static_cast<size_t>(request) * request_words;
                            const size_t live_words =
                                static_cast<size_t>(expected_counts[request]) * kv_dim;
                            const auto expect_live_words_equal =
                                [&](const char *tensor_name,
                                    const std::vector<uint16_t> &actual,
                                    const std::vector<uint16_t> &expected)
                            {
                                for (size_t word = 0; word < live_words; ++word)
                                {
                                    const uint16_t actual_word =
                                        actual[output_offset + word];
                                    const uint16_t expected_word = expected[word];
                                    if (actual_word == expected_word)
                                        continue;

                                    const size_t token = word / static_cast<size_t>(kv_dim);
                                    const size_t element = word % static_cast<size_t>(kv_dim);
                                    const size_t head = element / static_cast<size_t>(head_dim);
                                    const size_t column = element % static_cast<size_t>(head_dim);
                                    ADD_FAILURE()
                                        << "Grouped converted " << tensor_name
                                        << " is not serial byte exact"
                                        << " request=" << request
                                        << " token=" << token
                                        << " head=" << head
                                        << " column=" << column
                                        << " word=" << word
                                        << " actual_bits=" << actual_word
                                        << " expected_bits=" << expected_word;
                                    break;
                                }
                            };
                            expect_live_words_equal("K", actual_k, expected_k[request]);
                            expect_live_words_equal("V", actual_v, expected_v[request]);

                            const auto expect_padding_is_positive_zero =
                                [&](const char *tensor_name,
                                    const std::vector<uint16_t> &actual)
                            {
                                for (size_t word = live_words; word < request_words; ++word)
                                {
                                    const uint16_t actual_word =
                                        actual[output_offset + word];
                                    if (actual_word == uint16_t{0})
                                        continue;

                                    const size_t token = word / static_cast<size_t>(kv_dim);
                                    const size_t element = word % static_cast<size_t>(kv_dim);
                                    const size_t head = element / static_cast<size_t>(head_dim);
                                    const size_t column = element % static_cast<size_t>(head_dim);
                                    ADD_FAILURE()
                                        << "Grouped converted " << tensor_name
                                        << " padding is not +0"
                                        << " request=" << request
                                        << " token=" << token
                                        << " head=" << head
                                        << " column=" << column
                                        << " actual_bits=" << actual_word;
                                    break;
                                }
                            };
                            expect_padding_is_positive_zero("K", actual_k);
                            expect_padding_is_positive_zero("V", actual_v);
                        }
                    }
                }
            }
        }
    }

} // namespace llaminar2::test::gpu_kv_verifier
