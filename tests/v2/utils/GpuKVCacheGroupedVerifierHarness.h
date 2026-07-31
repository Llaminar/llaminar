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
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
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

    /** @brief Return whether a precision names either physical TurboQuant policy. */
    constexpr bool isTurboQuantCachePrecision(ActivationPrecision precision)
    {
        return precision == ActivationPrecision::TQ4 ||
               precision == ActivationPrecision::TQ8;
    }

    /**
     * @brief Complete source-conversion surface advertised by GPU ring caches.
     *
     * FP32 and BF16 caches accept their native source. FP16 and Q8_1 caches
     * additionally expose production conversion kernels for every activation
     * tensor format. Each TurboQuant storage policy accepts either FP32
     * projection rows for fused quantize-to-ring publication or its exact
     * prepared native pair for direct device-to-device publication:
     * `TQ4` selects TQ8-K/TQ4-V, while `TQ8` selects TQ8-K/TQ8-V.
     */
    inline constexpr std::array<FormatCase, 14> kFormatCases = {{
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
        {ActivationPrecision::TQ4, TensorType::FP32, TensorType::FP32, "TQ8-K/TQ4-V", "FP32"},
        {ActivationPrecision::TQ4, TensorType::TQ8, TensorType::TQ4, "TQ8-K/TQ4-V", "TQ8/TQ4"},
        {ActivationPrecision::TQ8, TensorType::FP32, TensorType::FP32, "TQ8-K/TQ8-V", "FP32"},
        {ActivationPrecision::TQ8, TensorType::TQ8, TensorType::TQ8, "TQ8-K/TQ8-V", "TQ8"},
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
    inline constexpr std::array<CacheReadFormatCase, 6> kCacheReadFormats = {{
        {ActivationPrecision::FP32, TensorType::FP32, "FP32"},
        {ActivationPrecision::FP16, TensorType::FP16, "FP16"},
        {ActivationPrecision::BF16, TensorType::BF16, "BF16"},
        {ActivationPrecision::Q8_1, TensorType::Q8_1, "Q8_1"},
        {ActivationPrecision::TQ4, TensorType::FP32, "TQ8-K/TQ4-V"},
        {ActivationPrecision::TQ8, TensorType::FP32, "TQ8-K/TQ8-V"},
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
        int batch_size = 1,
        int n_layers = 1)
    {
        using Factory = llaminar::v2::kernels::KernelFactory;
        using Config = llaminar::v2::kernels::KVCacheConfig;

        Config config{
            .precision = precision,
            .device = device,
            .num_layers = n_layers,
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

    /**
     * @brief One adversarial grouped-verifier transaction.
     *
     * The payload vectors are prepared before any graph replay begins.  The
     * stress loop therefore performs no host allocation while the simulated
     * inference lifetime is active.  `accepted_rows` is the serial-visible
     * state prefix selected by the verifier; rows after that prefix are
     * physically written by the grouped graph but must remain unreachable.
     */
    struct AdversarialKVLifecycleWave
    {
        int verifier_rows = 0;
        int accepted_rows = 0;
        bool restore_prefix_before = false;
        bool restore_from_ram = false;
        bool append_main_decode_after = false;
        std::vector<float> grouped_k;
        std::vector<float> grouped_v;
        std::vector<float> main_k;
        std::vector<float> main_v;
    };

    /**
     * @brief Pure serial model for one canonical ring head/count pair.
     *
     * Payload byte equality is proven against a second production cache.  This
     * tiny model independently proves that both caches expose the state serial
     * decode requires, so a shared metadata bug cannot make the comparison pass
     * by coincidence.
     */
    struct AdversarialKVRingState
    {
        int head = 0;
        int count = 0;

        void resetToPrefix(int prefix_rows, int capacity)
        {
            head = prefix_rows % capacity;
            count = std::min(prefix_rows, capacity);
        }

        void advance(int rows, int capacity)
        {
            head = (head + rows) % capacity;
            count = std::min(capacity, count + rows);
        }
    };

    /**
     * @brief Construct four hostile acceptance rotations over the complete M range.
     *
     * Every runtime verifier width is exercised with first-row acceptance,
     * middle rejection, last-row rejection, and all-accepted publication.
     * Each rotation begins by restoring the same prefix snapshot.  Device-tier
     * and RAM-tier restores alternate, proving that graph reuse does not retain
     * stale pre-restore metadata.  Periodic one-row main-graph appends cover the
     * handoff between speculative and ordinary decode.
     */
    inline std::vector<AdversarialKVLifecycleWave>
    makeAdversarialKVLifecycleWaves(int kv_dim)
    {
        constexpr int kMaxVerifierRows =
            kGroupedVerifierRuntimeRows.back();
        constexpr int kAcceptanceRotations = 4;
        std::vector<AdversarialKVLifecycleWave> waves;
        waves.reserve(
            kAcceptanceRotations * kGroupedVerifierRuntimeRows.size());

        for (int rotation = 0;
             rotation < kAcceptanceRotations;
             ++rotation)
        {
            for (size_t m_index = 0;
                 m_index < kGroupedVerifierRuntimeRows.size();
                 ++m_index)
            {
                const int verifier_rows =
                    kGroupedVerifierRuntimeRows[m_index];
                int accepted_rows = 1;
                switch (rotation)
                {
                case 0:
                    accepted_rows = 1;
                    break;
                case 1:
                    accepted_rows =
                        std::max(1, verifier_rows / 2);
                    break;
                case 2:
                    accepted_rows =
                        std::max(1, verifier_rows - 1);
                    break;
                default:
                    accepted_rows = verifier_rows;
                    break;
                }

                AdversarialKVLifecycleWave wave{
                    .verifier_rows = verifier_rows,
                    .accepted_rows = accepted_rows,
                    .restore_prefix_before = m_index == 0,
                    .restore_from_ram = (rotation % 2) != 0,
                    .append_main_decode_after =
                        ((rotation *
                              static_cast<int>(
                                  kGroupedVerifierRuntimeRows.size()) +
                          static_cast<int>(m_index)) %
                         5) == 4,
                    .grouped_k =
                        std::vector<float>(
                            static_cast<size_t>(kMaxVerifierRows) *
                            static_cast<size_t>(kv_dim)),
                    .grouped_v =
                        std::vector<float>(
                            static_cast<size_t>(kMaxVerifierRows) *
                            static_cast<size_t>(kv_dim)),
                    .main_k =
                        std::vector<float>(
                            static_cast<size_t>(kv_dim)),
                    .main_v =
                        std::vector<float>(
                            static_cast<size_t>(kv_dim)),
                };

                const uint32_t wave_number =
                    static_cast<uint32_t>(waves.size());
                for (int row = 0;
                     row < kMaxVerifierRows;
                     ++row)
                {
                    for (int column = 0;
                         column < kv_dim;
                         ++column)
                    {
                        const size_t index =
                            static_cast<size_t>(row) * kv_dim +
                            static_cast<size_t>(column);
                        const int k_code =
                            static_cast<int>(
                                (wave_number * 37u +
                                 static_cast<uint32_t>(row * 17) +
                                 static_cast<uint32_t>(column * 3)) %
                                251u) -
                            125;
                        const int v_code =
                            static_cast<int>(
                                (wave_number * 53u +
                                 static_cast<uint32_t>(row * 11) +
                                 static_cast<uint32_t>(column * 5)) %
                                241u) -
                            120;
                        wave.grouped_k[index] =
                            static_cast<float>(k_code) / 128.0f;
                        wave.grouped_v[index] =
                            static_cast<float>(v_code) / 128.0f;
                    }
                }
                for (int column = 0;
                     column < kv_dim;
                     ++column)
                {
                    wave.main_k[static_cast<size_t>(column)] =
                        static_cast<float>(
                            static_cast<int>(
                                (wave_number * 29u +
                                 static_cast<uint32_t>(column * 7)) %
                                193u) -
                            96) /
                        128.0f;
                    wave.main_v[static_cast<size_t>(column)] =
                        static_cast<float>(
                            static_cast<int>(
                                (wave_number * 31u +
                                 static_cast<uint32_t>(column * 13)) %
                                197u) -
                            98) /
                        128.0f;
                }
                waves.push_back(std::move(wave));
            }
        }
        return waves;
    }

    /**
     * @brief Stress graph-reused MTP/main advancement and prefix restoration.
     *
     * This is the KV-cache analogue of the adversarial MoE transfer state
     * machine suite.  It keeps one executable per runtime verifier width alive
     * for the complete test and drives all executables against the same
     * multi-layer cache.  The graph body is exactly:
     *
     * 1. Capture the immutable pre-verifier device state.
     * 2. Publish all grouped verifier K/V rows in every layer.
     * 3. Commit only the accepted serial-visible prefix from device metadata.
     *
     * Three explicit streams model production ownership.  A producer stream
     * uploads persistent input mailboxes and performs prefix/main-graph
     * mutations.  A graph stream waits on the producer event and launches the
     * selected captured graph.  An observer stream waits on graph completion
     * and copies head/count rows only into a persistent device observation
     * matrix.  The host sees that matrix once, after every wave has completed.
     *
     * `Runtime` is a thin backend adapter supplied by the CUDA and ROCm test
     * translation units.  Its Graph, Event, and DeviceBuffer types are RAII
     * owners, which makes exceptional test exits release all backend objects.
     */
    template <typename Runtime>
    void runAdversarialKVLifecycleStress(
        DeviceId device,
        const char *backend_label,
        ActivationPrecision cache_precision,
        const char *cache_label,
        void *producer_stream,
        void *graph_stream,
        void *observer_stream,
        Runtime &runtime)
    {
        constexpr int kLayerCount = 5;
        constexpr int kBatchSize = 1;
        constexpr int kCapacity = 23;
        constexpr int kPrefixRows = 17;
        constexpr int kKVHeads = 2;
        /*
         * TurboQuant's physical block codecs are defined for the production
         * attention widths 64, 128, and 256. Use the smallest valid width so
         * this shared lifecycle remains inexpensive while exercising the real
         * TQ kernels rather than a synthetic test-only geometry.
         */
        constexpr int kHeadDim = 64;
        constexpr int kKVDim = kKVHeads * kHeadDim;
        constexpr int kMaxVerifierRows =
            kGroupedVerifierRuntimeRows.back();
        constexpr CacheTopology kTopology{
            "replicated",
            kKVHeads,
            0,
            0,
        };

        ASSERT_NE(producer_stream, nullptr);
        ASSERT_NE(graph_stream, nullptr);
        ASSERT_NE(observer_stream, nullptr);

        TurboQuantContext tq_context(kHeadDim, 0xA17E5EEDu);
        auto grouped = makeBoundCache(
            device,
            cache_precision,
            kTopology,
            kCapacity,
            kHeadDim,
            &tq_context,
            kBatchSize,
            kLayerCount);
        auto serial = makeBoundCache(
            device,
            cache_precision,
            kTopology,
            kCapacity,
            kHeadDim,
            &tq_context,
            kBatchSize,
            kLayerCount);
        ASSERT_TRUE(
            grouped.cache
                ->supportsDeviceResidentSequenceStatePublication());
        ASSERT_EQ(
            grouped.cache->deviceSequenceStateCheckpointBytes(),
            static_cast<size_t>(2 * kLayerCount) *
                sizeof(int32_t));

        auto prefix_k = makeTensor(
            TensorType::FP32,
            {kPrefixRows, static_cast<size_t>(kKVDim)},
            0x13579BDFu,
            kHeadDim);
        auto prefix_v = makeTensor(
            TensorType::FP32,
            {kPrefixRows, static_cast<size_t>(kKVDim)},
            0x2468ACE0u,
            kHeadDim);
        auto grouped_k = makeTensor(
            TensorType::FP32,
            {kMaxVerifierRows, static_cast<size_t>(kKVDim)},
            0x10203040u,
            kHeadDim);
        auto grouped_v = makeTensor(
            TensorType::FP32,
            {kMaxVerifierRows, static_cast<size_t>(kKVDim)},
            0x50607080u,
            kHeadDim);
        auto serial_row_k = makeTensor(
            TensorType::FP32,
            {1, static_cast<size_t>(kKVDim)},
            0x11112222u,
            kHeadDim);
        auto serial_row_v = makeTensor(
            TensorType::FP32,
            {1, static_cast<size_t>(kKVDim)},
            0x33334444u,
            kHeadDim);
        ensureOnDevice(
            prefix_k.get(), device, producer_stream);
        ensureOnDevice(
            prefix_v.get(), device, producer_stream);
        ensureOnDevice(
            grouped_k.get(), device, producer_stream);
        ensureOnDevice(
            grouped_v.get(), device, producer_stream);
        ensureOnDevice(
            serial_row_k.get(), device, producer_stream);
        ensureOnDevice(
            serial_row_v.get(), device, producer_stream);

        for (int layer = 0;
             layer < kLayerCount;
             ++layer)
        {
            ASSERT_TRUE(grouped.cache->appendWithStream(
                layer,
                0,
                prefix_k.get(),
                prefix_v.get(),
                kPrefixRows,
                producer_stream));
            ASSERT_TRUE(serial.cache->appendWithStream(
                layer,
                0,
                prefix_k.get(),
                prefix_v.get(),
                kPrefixRows,
                producer_stream));
        }
        ASSERT_TRUE(runtime.synchronizeStream(producer_stream));

        const auto prefix_layout =
            grouped.cache->logicalBlockLayout(
                0,
                kPrefixRows);
        ASSERT_GT(prefix_layout.k_bytes, 0u);
        ASSERT_GT(prefix_layout.v_bytes, 0u);

        struct PrefixLayerSnapshot
        {
            typename Runtime::DeviceBuffer device_k;
            typename Runtime::DeviceBuffer device_v;
            std::vector<uint8_t> host_k;
            std::vector<uint8_t> host_v;
        };
        std::vector<PrefixLayerSnapshot> prefix_snapshots;
        prefix_snapshots.reserve(kLayerCount);
        for (int layer = 0;
             layer < kLayerCount;
             ++layer)
        {
            PrefixLayerSnapshot snapshot{
                .device_k =
                    runtime.allocateDeviceBuffer(
                        prefix_layout.k_bytes),
                .device_v =
                    runtime.allocateDeviceBuffer(
                        prefix_layout.v_bytes),
                .host_k =
                    std::vector<uint8_t>(
                        prefix_layout.k_bytes),
                .host_v =
                    std::vector<uint8_t>(
                        prefix_layout.v_bytes),
            };
            ASSERT_TRUE(snapshot.device_k.valid());
            ASSERT_TRUE(snapshot.device_v.valid());
            const IKVCache::KVCacheLogicalBlockDescriptor
                device_descriptor{
                    .layer = layer,
                    .seq_idx = 0,
                    .logical_token_start = 0,
                    .token_count = kPrefixRows,
                    .stream = producer_stream,
                    .payload_domain =
                        IKVCache::
                            KVCacheLogicalBlockPayloadDomain::
                                Device,
                };
            ASSERT_TRUE(grouped.cache->exportLogicalBlock(
                device_descriptor,
                snapshot.device_k.data(),
                snapshot.device_v.data()));
            ASSERT_TRUE(
                runtime.copyDeviceToHostAsync(
                    snapshot.host_k.data(),
                    snapshot.device_k.data(),
                    prefix_layout.k_bytes,
                    producer_stream));
            ASSERT_TRUE(
                runtime.copyDeviceToHostAsync(
                    snapshot.host_v.data(),
                    snapshot.device_v.data(),
                    prefix_layout.v_bytes,
                    producer_stream));
            prefix_snapshots.push_back(
                std::move(snapshot));
        }
        ASSERT_TRUE(runtime.synchronizeStream(producer_stream));

        auto checkpoint =
            runtime.allocateDeviceBuffer(
                grouped.cache
                    ->deviceSequenceStateCheckpointBytes());
        auto target_cached_tokens =
            runtime.allocateDeviceBuffer(sizeof(int32_t));
        auto accepted_state_rows =
            runtime.allocateDeviceBuffer(sizeof(int32_t));
        auto publication_ok =
            runtime.allocateDeviceBuffer(sizeof(int32_t));
        ASSERT_TRUE(checkpoint.valid());
        ASSERT_TRUE(target_cached_tokens.valid());
        ASSERT_TRUE(accepted_state_rows.valid());
        ASSERT_TRUE(publication_ok.valid());

        std::vector<typename Runtime::Graph> graphs;
        graphs.reserve(
            kGroupedVerifierRuntimeRows.size());
        for (const int verifier_rows :
             kGroupedVerifierRuntimeRows)
        {
            auto graph = runtime.captureGraph(
                graph_stream,
                [&]() -> bool
                {
                    std::string checkpoint_error;
                    if (!grouped.cache
                             ->captureDeviceSequenceStateCheckpoint(
                                 0,
                                 checkpoint.data(),
                                 checkpoint.size(),
                                 graph_stream,
                                 &checkpoint_error))
                    {
                        return false;
                    }
                    for (int layer = 0;
                         layer < kLayerCount;
                         ++layer)
                    {
                        if (!grouped.cache
                                 ->appendVerifierRowsDecodeEquivalent(
                                     layer,
                                     0,
                                     grouped_k.get(),
                                     grouped_v.get(),
                                     verifier_rows,
                                     graph_stream))
                        {
                            return false;
                        }
                    }

                    IKVCache::
                        DeviceSequenceStatePublicationRequest
                            request{
                                .request_count = 1,
                                .first_seq_idx = 0,
                                .target_cached_tokens_device =
                                    static_cast<const int32_t *>(
                                        target_cached_tokens
                                            .data()),
                                .accepted_state_counts_device =
                                    static_cast<const int32_t *>(
                                        accepted_state_rows
                                            .data()),
                                .publication_ok_flags_device =
                                    static_cast<const int32_t *>(
                                        publication_ok.data()),
                                .basis =
                                    IKVCache::
                                        DeviceSequenceStatePublicationBasis::
                                            CapturedBase,
                                .base_sequence_state_checkpoint_device =
                                    checkpoint.data(),
                                .base_sequence_state_checkpoint_bytes =
                                    checkpoint.size(),
                                .stream = graph_stream,
                            };
                    std::string publication_error;
                    return grouped.cache
                        ->publishSequenceStateFromDeviceMetadata(
                            request,
                            &publication_error);
                });
            ASSERT_TRUE(graph.valid())
                << backend_label
                << " failed to capture adversarial KV graph M="
                << verifier_rows;
            graphs.push_back(std::move(graph));
        }

        const auto waves =
            makeAdversarialKVLifecycleWaves(kKVDim);
        const size_t observation_cells =
            waves.size() *
            static_cast<size_t>(kLayerCount);
        auto observed_heads =
            runtime.allocateDeviceBuffer(
                observation_cells * sizeof(int32_t));
        auto observed_counts =
            runtime.allocateDeviceBuffer(
                observation_cells * sizeof(int32_t));
        ASSERT_TRUE(observed_heads.valid());
        ASSERT_TRUE(observed_counts.valid());

        std::vector<typename Runtime::Event> producer_ready;
        std::vector<typename Runtime::Event> graph_done;
        std::vector<typename Runtime::Event> observation_done;
        producer_ready.reserve(waves.size());
        graph_done.reserve(waves.size());
        observation_done.reserve(waves.size());
        for (size_t wave_index = 0;
             wave_index < waves.size();
             ++wave_index)
        {
            producer_ready.push_back(
                runtime.createEvent());
            graph_done.push_back(
                runtime.createEvent());
            observation_done.push_back(
                runtime.createEvent());
            ASSERT_TRUE(
                producer_ready.back().valid());
            ASSERT_TRUE(graph_done.back().valid());
            ASSERT_TRUE(
                observation_done.back().valid());
        }

        std::vector<AdversarialKVRingState>
            expected_after_graph;
        expected_after_graph.reserve(waves.size());
        AdversarialKVRingState oracle;
        oracle.resetToPrefix(
            kPrefixRows,
            kCapacity);
        const int32_t ok_value = 1;
        const size_t grouped_bytes =
            static_cast<size_t>(kMaxVerifierRows) *
            static_cast<size_t>(kKVDim) *
            sizeof(float);
        const size_t row_bytes =
            static_cast<size_t>(kKVDim) *
            sizeof(float);

        for (size_t wave_index = 0;
             wave_index < waves.size();
             ++wave_index)
        {
            const auto &wave = waves[wave_index];
            SCOPED_TRACE(
                std::string(backend_label) +
                " cache=" +
                cache_label +
                " lifecycle wave=" +
                std::to_string(wave_index) +
                " M=" +
                std::to_string(wave.verifier_rows) +
                " accepted=" +
                std::to_string(wave.accepted_rows));

            if (wave.restore_prefix_before)
            {
                const IKVCache::StateResetContext
                    reset_context{
                        .boundary =
                            IKVCache::
                                StateResetBoundary::
                                    PrefixReplacement,
                        .execution_stream =
                            producer_stream,
                        .reason =
                            "adversarial-prefix-reuse",
                    };
                ASSERT_TRUE(
                    grouped.cache->resetRequestState(
                        reset_context));
                ASSERT_TRUE(
                    serial.cache->resetRequestState(
                        reset_context));
                for (int layer = 0;
                     layer < kLayerCount;
                     ++layer)
                {
                    const auto &snapshot =
                        prefix_snapshots[
                            static_cast<size_t>(
                                layer)];
                    const auto domain =
                        wave.restore_from_ram
                            ? IKVCache::
                                  KVCacheLogicalBlockPayloadDomain::
                                      Host
                            : IKVCache::
                                  KVCacheLogicalBlockPayloadDomain::
                                      Device;
                    const IKVCache::
                        KVCacheLogicalBlockDescriptor
                            descriptor{
                                .layer = layer,
                                .seq_idx = 0,
                                .logical_token_start = 0,
                                .token_count =
                                    kPrefixRows,
                                .stream =
                                    producer_stream,
                                .payload_domain =
                                    domain,
                            };
                    void *source_k =
                        wave.restore_from_ram
                            ? const_cast<uint8_t *>(
                                  snapshot.host_k
                                      .data())
                            : snapshot.device_k.data();
                    void *source_v =
                        wave.restore_from_ram
                            ? const_cast<uint8_t *>(
                                  snapshot.host_v
                                      .data())
                            : snapshot.device_v.data();
                    ASSERT_TRUE(
                        grouped.cache
                            ->importLogicalBlock(
                                descriptor,
                                source_k,
                                source_v));
                    ASSERT_TRUE(
                        serial.cache
                            ->importLogicalBlock(
                                descriptor,
                                source_k,
                                source_v));
                }
                oracle.resetToPrefix(
                    kPrefixRows,
                    kCapacity);
            }

            ASSERT_TRUE(
                runtime.copyHostToDeviceAsync(
                    grouped_k->gpu_data_ptr(),
                    wave.grouped_k.data(),
                    grouped_bytes,
                    producer_stream));
            ASSERT_TRUE(
                runtime.copyHostToDeviceAsync(
                    grouped_v->gpu_data_ptr(),
                    wave.grouped_v.data(),
                    grouped_bytes,
                    producer_stream));
            const int32_t target_count =
                std::min(
                    kCapacity,
                    oracle.count +
                        wave.accepted_rows);
            const int32_t accepted_rows =
                wave.accepted_rows;
            ASSERT_TRUE(
                runtime.copyHostToDeviceAsync(
                    target_cached_tokens.data(),
                    &target_count,
                    sizeof(target_count),
                    producer_stream));
            ASSERT_TRUE(
                runtime.copyHostToDeviceAsync(
                    accepted_state_rows.data(),
                    &accepted_rows,
                    sizeof(accepted_rows),
                    producer_stream));
            ASSERT_TRUE(
                runtime.copyHostToDeviceAsync(
                    publication_ok.data(),
                    &ok_value,
                    sizeof(ok_value),
                    producer_stream));
            ASSERT_TRUE(runtime.recordEvent(
                producer_ready[wave_index],
                producer_stream));

            ASSERT_TRUE(runtime.waitEvent(
                graph_stream,
                producer_ready[wave_index]));
            const auto graph_it =
                std::find(
                    kGroupedVerifierRuntimeRows.begin(),
                    kGroupedVerifierRuntimeRows.end(),
                    wave.verifier_rows);
            ASSERT_NE(
                graph_it,
                kGroupedVerifierRuntimeRows.end());
            const size_t graph_index =
                static_cast<size_t>(
                    std::distance(
                        kGroupedVerifierRuntimeRows.begin(),
                        graph_it));
            ASSERT_TRUE(runtime.launchGraph(
                graphs[graph_index],
                graph_stream));
            ASSERT_TRUE(runtime.recordEvent(
                graph_done[wave_index],
                graph_stream));

            oracle.advance(
                wave.accepted_rows,
                kCapacity);
            expected_after_graph.push_back(oracle);

            ASSERT_TRUE(runtime.waitEvent(
                observer_stream,
                graph_done[wave_index]));
            for (int layer = 0;
                 layer < kLayerCount;
                 ++layer)
            {
                const size_t cell =
                    wave_index *
                        static_cast<size_t>(
                            kLayerCount) +
                    static_cast<size_t>(layer);
                auto *head_destination =
                    static_cast<uint8_t *>(
                        observed_heads.data()) +
                    cell * sizeof(int32_t);
                auto *count_destination =
                    static_cast<uint8_t *>(
                        observed_counts.data()) +
                    cell * sizeof(int32_t);
                ASSERT_TRUE(
                    runtime
                        .copyDeviceToDeviceAsync(
                            head_destination,
                            grouped.cache
                                ->deviceRingHeadPtr(
                                    layer,
                                    0),
                            sizeof(int32_t),
                            observer_stream));
                ASSERT_TRUE(
                    runtime
                        .copyDeviceToDeviceAsync(
                            count_destination,
                            grouped.cache
                                ->deviceCachedTokenCountPtr(
                                    layer,
                                    0),
                            sizeof(int32_t),
                            observer_stream));
            }
            ASSERT_TRUE(runtime.recordEvent(
                observation_done[wave_index],
                observer_stream));

            ASSERT_TRUE(runtime.waitEvent(
                producer_stream,
                observation_done[wave_index]));
            for (int row = 0;
                 row < wave.accepted_rows;
                 ++row)
            {
                const auto *row_k =
                    wave.grouped_k.data() +
                    static_cast<size_t>(row) *
                        kKVDim;
                const auto *row_v =
                    wave.grouped_v.data() +
                    static_cast<size_t>(row) *
                        kKVDim;
                ASSERT_TRUE(
                    runtime.copyHostToDeviceAsync(
                        serial_row_k
                            ->gpu_data_ptr(),
                        row_k,
                        row_bytes,
                        producer_stream));
                ASSERT_TRUE(
                    runtime.copyHostToDeviceAsync(
                        serial_row_v
                            ->gpu_data_ptr(),
                        row_v,
                        row_bytes,
                        producer_stream));
                for (int layer = 0;
                     layer < kLayerCount;
                     ++layer)
                {
                    ASSERT_TRUE(
                        serial.cache
                            ->appendWithStream(
                                layer,
                                0,
                                serial_row_k.get(),
                                serial_row_v.get(),
                                1,
                                producer_stream));
                }
            }

            if (wave.append_main_decode_after)
            {
                ASSERT_TRUE(
                    runtime.copyHostToDeviceAsync(
                        serial_row_k
                            ->gpu_data_ptr(),
                        wave.main_k.data(),
                        row_bytes,
                        producer_stream));
                ASSERT_TRUE(
                    runtime.copyHostToDeviceAsync(
                        serial_row_v
                            ->gpu_data_ptr(),
                        wave.main_v.data(),
                        row_bytes,
                        producer_stream));
                for (int layer = 0;
                     layer < kLayerCount;
                     ++layer)
                {
                    ASSERT_TRUE(
                        grouped.cache
                            ->appendWithStream(
                                layer,
                                0,
                                serial_row_k.get(),
                                serial_row_v.get(),
                                1,
                                producer_stream));
                    ASSERT_TRUE(
                        serial.cache
                            ->appendWithStream(
                                layer,
                                0,
                                serial_row_k.get(),
                                serial_row_v.get(),
                                1,
                                producer_stream));
                }
                oracle.advance(1, kCapacity);
            }
        }

        auto final_ready = runtime.createEvent();
        ASSERT_TRUE(final_ready.valid());
        ASSERT_TRUE(runtime.recordEvent(
            final_ready,
            producer_stream));
        ASSERT_TRUE(runtime.waitEvent(
            observer_stream,
            final_ready));

        std::vector<int32_t> host_heads(
            observation_cells,
            -1);
        std::vector<int32_t> host_counts(
            observation_cells,
            -1);
        ASSERT_TRUE(
            runtime.copyDeviceToHostAsync(
                host_heads.data(),
                observed_heads.data(),
                observation_cells *
                    sizeof(int32_t),
                observer_stream));
        ASSERT_TRUE(
            runtime.copyDeviceToHostAsync(
                host_counts.data(),
                observed_counts.data(),
                observation_cells *
                    sizeof(int32_t),
                observer_stream));

        std::array<int32_t, kLayerCount>
            final_grouped_heads{};
        std::array<int32_t, kLayerCount>
            final_grouped_counts{};
        std::array<int32_t, kLayerCount>
            final_serial_heads{};
        std::array<int32_t, kLayerCount>
            final_serial_counts{};
        for (int layer = 0;
             layer < kLayerCount;
             ++layer)
        {
            ASSERT_TRUE(
                runtime.copyDeviceToHostAsync(
                    &final_grouped_heads[
                        static_cast<size_t>(
                            layer)],
                    grouped.cache
                        ->deviceRingHeadPtr(
                            layer,
                            0),
                    sizeof(int32_t),
                    observer_stream));
            ASSERT_TRUE(
                runtime.copyDeviceToHostAsync(
                    &final_grouped_counts[
                        static_cast<size_t>(
                            layer)],
                    grouped.cache
                        ->deviceCachedTokenCountPtr(
                            layer,
                            0),
                    sizeof(int32_t),
                    observer_stream));
            ASSERT_TRUE(
                runtime.copyDeviceToHostAsync(
                    &final_serial_heads[
                        static_cast<size_t>(
                            layer)],
                    serial.cache
                        ->deviceRingHeadPtr(
                            layer,
                            0),
                    sizeof(int32_t),
                    observer_stream));
            ASSERT_TRUE(
                runtime.copyDeviceToHostAsync(
                    &final_serial_counts[
                        static_cast<size_t>(
                            layer)],
                    serial.cache
                        ->deviceCachedTokenCountPtr(
                            layer,
                            0),
                    sizeof(int32_t),
                    observer_stream));
        }
        ASSERT_TRUE(
            runtime.synchronizeStream(
                observer_stream));

        for (size_t wave_index = 0;
             wave_index < waves.size();
             ++wave_index)
        {
            for (int layer = 0;
                 layer < kLayerCount;
                 ++layer)
            {
                const size_t cell =
                    wave_index *
                        static_cast<size_t>(
                            kLayerCount) +
                    static_cast<size_t>(layer);
                EXPECT_EQ(
                    host_heads[cell],
                    expected_after_graph[
                        wave_index]
                        .head)
                    << backend_label
                    << " cache=" << cache_label
                    << " wave="
                    << wave_index
                    << " layer="
                    << layer;
                EXPECT_EQ(
                    host_counts[cell],
                    expected_after_graph[
                        wave_index]
                        .count)
                    << backend_label
                    << " cache=" << cache_label
                    << " wave="
                    << wave_index
                    << " layer="
                    << layer;
            }
        }

        for (int layer = 0;
             layer < kLayerCount;
             ++layer)
        {
            const size_t index =
                static_cast<size_t>(layer);
            EXPECT_EQ(
                final_grouped_heads[index],
                oracle.head);
            EXPECT_EQ(
                final_grouped_counts[index],
                oracle.count);
            EXPECT_EQ(
                final_serial_heads[index],
                oracle.head);
            EXPECT_EQ(
                final_serial_counts[index],
                oracle.count);

            const auto final_layout =
                grouped.cache
                    ->logicalBlockLayout(
                        layer,
                        oracle.count);
            std::vector<uint8_t> grouped_payload_k(
                final_layout.k_bytes);
            std::vector<uint8_t> grouped_payload_v(
                final_layout.v_bytes);
            std::vector<uint8_t> serial_payload_k(
                final_layout.k_bytes);
            std::vector<uint8_t> serial_payload_v(
                final_layout.v_bytes);
            const IKVCache::
                KVCacheLogicalBlockDescriptor
                    descriptor{
                        .layer = layer,
                        .seq_idx = 0,
                        .logical_token_start = 0,
                        .token_count =
                            oracle.count,
                        .stream =
                            observer_stream,
                        .payload_domain =
                            IKVCache::
                                KVCacheLogicalBlockPayloadDomain::
                                    Host,
                    };
            ASSERT_TRUE(
                grouped.cache->exportLogicalBlock(
                    descriptor,
                    grouped_payload_k.data(),
                    grouped_payload_v.data()));
            ASSERT_TRUE(
                serial.cache->exportLogicalBlock(
                    descriptor,
                    serial_payload_k.data(),
                    serial_payload_v.data()));
            EXPECT_EQ(
                grouped_payload_k,
                serial_payload_k)
                << backend_label
                << " cache=" << cache_label
                << " adversarial K payload mismatch layer="
                << layer;
            EXPECT_EQ(
                grouped_payload_v,
                serial_payload_v)
                << backend_label
                << " cache=" << cache_label
                << " adversarial V payload mismatch layer="
                << layer;
        }
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
                                isTurboQuantCachePrecision(format.cache_precision)
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
     * @brief Prove logical prefix blocks remain device-owned across harvest and restore.
     *
     * The matrix covers every native GPU cache family, both supported head
     * dimensions, and replicated plus LocalTP-sharded storage. Each source ring
     * is deliberately wrapped before harvest. The production route then gathers
     * its logical rows into device buffers, scatters those buffers into a fresh
     * cache, and gathers the restored rows again without an intervening host
     * observation.
     *
     * A host-domain export is retained solely as the canonical assertion oracle.
     * Device bytes cross to the host only after the complete D2D pipeline has
     * been enqueued. This separation catches accidental D2H/H2D substitution,
     * stale host ring metadata, TQ K/V stride confusion, and floating-point
     * signed-zero differences without making host publication part of the
     * production route under test.
     *
     * @tparam DeviceAllocator Callable `(size_t) -> void *` for test buffers.
     * @tparam DeviceReleaser Callable `(void *)` for test-buffer destruction.
     * @tparam AsyncDeviceByteCopier Callable performing test-boundary D2H copy.
     * @tparam StreamSynchronizer Callable fencing the explicit backend stream.
     * @param device Backend device used by the production cache factory.
     * @param backend_label Human-readable backend name for assertion traces.
     * @param standard_counter Device logical-block counter for ordinary caches.
     * @param tq_counter Device logical-block counter for asymmetric TQ caches.
     * @param stream Mandatory non-default CUDA/HIP stream.
     * @param allocate_device Allocate one backend-native device byte range.
     * @param release_device Release a range allocated by @p allocate_device.
     * @param copy_device_bytes_async Enqueue one D2H assertion-boundary copy.
     * @param synchronize_stream Fence @p stream after all observation copies.
     */
    template <typename DeviceAllocator, typename DeviceReleaser,
              typename AsyncDeviceByteCopier, typename StreamSynchronizer>
    void runAllFormatDeviceLogicalBlockSweep(
        DeviceId device,
        const char *backend_label,
        const char *standard_export_counter,
        const char *standard_import_counter,
        const char *tq_export_counter,
        const char *tq_import_counter,
        void *stream,
        DeviceAllocator &&allocate_device,
        DeviceReleaser &&release_device,
        AsyncDeviceByteCopier &&copy_device_bytes_async,
        StreamSynchronizer &&synchronize_stream)
    {
        constexpr int max_seq_len = 6;
        constexpr std::array<int, 2> head_dims = {64, 128};
        ScopedPerfStats perfstats;
        ASSERT_TRUE(PerfStatsCollector::isEnabled());
        ASSERT_NE(stream, nullptr);
        uint32_t seed = 0xD3A1CEu;
        uint64_t expected_standard_cells = 0;
        uint64_t expected_tq_cells = 0;

        for (const auto &format : kCacheReadFormats)
        {
            for (const int head_dim : head_dims)
            {
                TurboQuantContext tq_context(head_dim, 0xC0FFEEu);
                for (const auto &topology : kTopologies)
                {
                    SCOPED_TRACE(std::string(backend_label) +
                                 " device logical block cache=" + format.label +
                                 " topology=" + topology.label +
                                 " D=" + std::to_string(head_dim));
                    const int local_heads = topology.effectiveLocalHeads();
                    const int kv_dim = local_heads * head_dim;
                    auto source = makeBoundCache(
                        device, format.cache_precision, topology,
                        max_seq_len, head_dim, &tq_context);
                    std::vector<std::unique_ptr<ITensor>> append_lifetime;

                    const auto append_chunk = [&](int rows)
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

                        // The logical codec canonicalizes floating -0 to +0.
                        // Seed one negative zero in ordinary floating caches so
                        // the device gather must implement the same byte rule as
                        // the host diagnostic oracle. TQ stores quantized bytes,
                        // so its FP32 append source is intentionally left alone.
                        if (!isTurboQuantCachePrecision(format.cache_precision))
                        {
                            if (format.append_type == TensorType::FP32)
                            {
                                static_cast<uint32_t *>(k->raw_mutable_data())[0] =
                                    0x80000000u;
                                static_cast<uint32_t *>(v->raw_mutable_data())[0] =
                                    0x80000000u;
                            }
                            else if (format.append_type == TensorType::FP16 ||
                                     format.append_type == TensorType::BF16)
                            {
                                static_cast<uint16_t *>(k->raw_mutable_data())[0] =
                                    0x8000u;
                                static_cast<uint16_t *>(v->raw_mutable_data())[0] =
                                    0x8000u;
                            }
                        }

                        ensureOnDevice(k.get(), device, stream);
                        ensureOnDevice(v.get(), device, stream);
                        ASSERT_TRUE(source.cache->appendWithStream(
                            0, 0, k.get(), v.get(), rows, stream));
                        append_lifetime.push_back(std::move(k));
                        append_lifetime.push_back(std::move(v));
                    };

                    // Five rows followed by four rows leaves the six newest
                    // logical rows split across both physical ends of the ring.
                    append_chunk(5);
                    append_chunk(4);
                    ASSERT_TRUE(synchronize_stream(stream));

                    const auto layout =
                        source.cache->logicalBlockLayout(0, max_seq_len);
                    ASSERT_GT(layout.k_bytes, 0u);
                    ASSERT_GT(layout.v_bytes, 0u);
                    ASSERT_TRUE(layout.device_resident);
                    EXPECT_EQ(layout.local_kv_heads, local_heads);
                    EXPECT_EQ(layout.kv_head_start, topology.head_start);

                    std::vector<uint8_t> canonical_k(layout.k_bytes, 0);
                    std::vector<uint8_t> canonical_v(layout.v_bytes, 0);
                    const IKVCache::KVCacheLogicalBlockDescriptor host_descriptor{
                        .layer = 0,
                        .seq_idx = 0,
                        .logical_token_start = 0,
                        .token_count = max_seq_len,
                        .stream = stream,
                        .payload_domain =
                            IKVCache::KVCacheLogicalBlockPayloadDomain::Host,
                    };
                    ASSERT_TRUE(source.cache->exportLogicalBlock(
                        host_descriptor, canonical_k.data(), canonical_v.data()));

                    void *source_k = allocate_device(layout.k_bytes);
                    void *source_v = allocate_device(layout.v_bytes);
                    void *restored_k = allocate_device(layout.k_bytes);
                    void *restored_v = allocate_device(layout.v_bytes);
                    ASSERT_NE(source_k, nullptr);
                    ASSERT_NE(source_v, nullptr);
                    ASSERT_NE(restored_k, nullptr);
                    ASSERT_NE(restored_v, nullptr);

                    const IKVCache::KVCacheLogicalBlockDescriptor device_descriptor{
                        .layer = 0,
                        .seq_idx = 0,
                        .logical_token_start = 0,
                        .token_count = max_seq_len,
                        .stream = stream,
                        .payload_domain =
                            IKVCache::KVCacheLogicalBlockPayloadDomain::Device,
                    };
                    IKVCache::KVCacheLogicalBlockDescriptor null_stream_descriptor =
                        device_descriptor;
                    null_stream_descriptor.stream = nullptr;
                    EXPECT_THROW(
                        source.cache->exportLogicalBlock(
                            null_stream_descriptor, source_k, source_v),
                        std::invalid_argument)
                        << "GPU device export must fail hard before it can enter "
                           "a default stream";

                    ASSERT_TRUE(source.cache->exportLogicalBlock(
                        device_descriptor, source_k, source_v));
                    auto restored = makeBoundCache(
                        device, format.cache_precision, topology,
                        max_seq_len, head_dim, &tq_context);
                    EXPECT_THROW(
                        restored.cache->importLogicalBlock(
                            null_stream_descriptor, source_k, source_v),
                        std::invalid_argument)
                        << "GPU device import must fail hard before it can enter "
                           "a default stream";
                    ASSERT_TRUE(restored.cache->importLogicalBlock(
                        device_descriptor, source_k, source_v));
                    ASSERT_TRUE(restored.cache->exportLogicalBlock(
                        device_descriptor, restored_k, restored_v));

                    std::vector<uint8_t> observed_source_k(layout.k_bytes, 0);
                    std::vector<uint8_t> observed_source_v(layout.v_bytes, 0);
                    std::vector<uint8_t> observed_restored_k(layout.k_bytes, 0);
                    std::vector<uint8_t> observed_restored_v(layout.v_bytes, 0);
                    ASSERT_TRUE(copy_device_bytes_async(
                        observed_source_k.data(), source_k,
                        layout.k_bytes, stream));
                    ASSERT_TRUE(copy_device_bytes_async(
                        observed_source_v.data(), source_v,
                        layout.v_bytes, stream));
                    ASSERT_TRUE(copy_device_bytes_async(
                        observed_restored_k.data(), restored_k,
                        layout.k_bytes, stream));
                    ASSERT_TRUE(copy_device_bytes_async(
                        observed_restored_v.data(), restored_v,
                        layout.v_bytes, stream));
                    ASSERT_TRUE(synchronize_stream(stream));

                    EXPECT_EQ(observed_source_k, canonical_k)
                        << "Device harvest changed canonical K bytes";
                    EXPECT_EQ(observed_source_v, canonical_v)
                        << "Device harvest changed canonical V bytes";
                    EXPECT_EQ(observed_restored_k, canonical_k)
                        << "D2D restore changed canonical K bytes";
                    EXPECT_EQ(observed_restored_v, canonical_v)
                        << "D2D restore changed canonical V bytes";

                    release_device(source_k);
                    release_device(source_v);
                    release_device(restored_k);
                    release_device(restored_v);

                    if (isTurboQuantCachePrecision(format.cache_precision))
                        ++expected_tq_cells;
                    else
                        ++expected_standard_cells;
                }
            }
        }

        const auto sum_counter = [](const char *counter_name)
        {
            uint64_t calls = 0;
            for (const auto &record : PerfStatsCollector::snapshot(
                     {std::string("prefix_cache.") + counter_name}))
            {
                if (record.kind == PerfStatRecord::Kind::Counter)
                    calls += record.count;
            }
            return calls;
        };

        // Each cell exports the source and restored cache once, but imports
        // only into the restored cache. Rejected null-stream probes must not
        // increment any production-route counter.
        EXPECT_EQ(sum_counter(standard_export_counter),
                  expected_standard_cells * 2);
        EXPECT_EQ(sum_counter(standard_import_counter),
                  expected_standard_cells);
        EXPECT_EQ(sum_counter(tq_export_counter), expected_tq_cells * 2);
        EXPECT_EQ(sum_counter(tq_import_counter), expected_tq_cells);
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
