/**
 * @file QwenSparseDeviceTest.h
 * @brief Shared captured CUDA/HIP QSA conformance and exact-candidate harness.
 *
 * Backend translation units provide QSA_RT, QSA_STREAM, QSA_GRAPH, QSA_EXEC,
 * QSA_EVENT, and the three production launch bindings. All allocation, copies,
 * and result synchronization are explicit fixture boundaries outside capture.
 * The core attention observations prove the loops decoded selected rows only.
 */
#pragma once
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "kernels/qwen4exp/QwenSparseMath.h"
#include "kernels/qwen4exp/QwenSparseReplay.h"
#include "kernels/qwen4exp/Qwen4ExpStateIO.h"
#include "QwenSparseNativeCodecFixture.h"
#include "QwenSparseRotaryReference.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace llaminar2::qwen4exp::QSA_SUITE
{
/** @brief Native API failure is fatal with its exact runtime diagnostic. */
template<class Status> void check(Status status)
{
    if (status != QSA_RT(Success)) throw std::runtime_error(QSA_RT(GetErrorString)(status));
}
/** @brief Fixture-only device allocation; no buffer is created during capture. */
template<class T> struct Buffer
{
    T *data = nullptr;
    std::size_t size;
    /** @brief Allocate the admitted persistent native extent. */
    explicit Buffer(std::size_t count) : size(count) { check(QSA_RT(Malloc)(reinterpret_cast<void **>(&data), count * sizeof(T))); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    /** @brief Retire storage after the fixture collected its terminal result. */
    ~Buffer() { (void)QSA_RT(Free)(data); }
    /** @brief Upload fixture state before graph recording/replay. */
    void upload(const T *source, QSA_STREAM stream) { check(QSA_RT(MemcpyAsync)(data, source, size * sizeof(T), QSA_RT(MemcpyHostToDevice), stream)); }
    /** @brief Collect terminal fixture output after captured execution. */
    std::vector<T> download(QSA_STREAM stream)
    {
        std::vector<T> result(size);
        check(QSA_RT(MemcpyAsync)(result.data(), data, size * sizeof(T), QSA_RT(MemcpyDeviceToHost), stream));
        check(QSA_RT(StreamSynchronize)(stream));
        return result;
    }
};
/** @brief Deterministic fixture values independent of standard random libraries. */
std::vector<float> values(std::size_t size, std::uint32_t seed)
{
    std::vector<float> result(size);
    for (auto &value : result)
    {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        value = float(seed >> 8) * (1.0f / 8388608.0f) - 1.0f;
    }
    return result;
}
/**
 * @brief Retire partially constructed fixture uploads before any backing unwinds.
 * @details The normal fixture destructor closes the stream first and clears the
 * original handle; on constructor failure this last member owns its exact join.
 */
struct FixtureCopyRetirement final
{
    QSA_STREAM *stream;
    ~FixtureCopyRetirement()
    {
        if (*stream && QSA_RT(StreamSynchronize)(*stream) != QSA_RT(Success)) std::terminate();
    }
};
/** @brief Persistent captured fixture containing the real production QSA pipeline. */
struct Fixture
{
    int rows, capacity, positions, blocks;
    QSA_STREAM stream = nullptr;
    QSA_GRAPH graph = nullptr;
    QSA_EXEC executable = nullptr;
    QwenSparseReplay replay;
    std::vector<float> host_query, host_raw, host_partial, host_pools, norm, cosine, sine, host_core_query, host_gate;
    std::vector<float> host_key, host_value;
    std::vector<std::uint16_t> key16, value16;
    Buffer<QwenSparseReplay> state;
    Buffer<float> query, raw, partial, pools, query_norm, key_norm, cos, sin, prepared, scores, core_query, gate, output;
    Buffer<std::int32_t> selected, count, accepted;
    Buffer<QwenSparseAttentionObservation> observations;
    Buffer<QwenSparseAttentionPartial> partials;
    Buffer<unsigned char> keys, vals;
    QwenSparseIndexer indexer;
    QwenSparseAttention attention;
    FixtureCopyRetirement copyRetirement{&stream};
    /** @brief Bind one production-shaped FP32/FP16/BF16 candidate before capture. */
    Fixture(int row_count, int prefix, int token_capacity, QwenSparseScalar representation)
        : rows(row_count), capacity(token_capacity), positions(capacity + 17), blocks((capacity + 3) / 4),
          replay{prefix, rows, 17}, host_query(values(std::size_t(rows) * 512, 3)),
          host_raw(values(std::size_t(rows) * 128, 5)), host_partial(values(512, 7)),
          host_pools(values(std::size_t(blocks) * 128, 11)), norm(128, 1.0f),
          cosine(std::size_t(positions) * 32), sine(cosine.size()),
          host_core_query(values(std::size_t(rows) * 6144, 13)), host_gate(values(host_core_query.size(), 17)),
          host_key(values(std::size_t(capacity) * 512, 19)), host_value(values(host_key.size(), 23)),
          key16(host_key.size()), value16(host_value.size()), state(1), query(host_query.size()), raw(host_raw.size()),
          partial(512), pools(host_pools.size()), query_norm(128), key_norm(128), cos(cosine.size()), sin(sine.size()),
          prepared(query.size), scores(std::size_t(rows) * blocks), core_query(host_core_query.size()), gate(host_gate.size()),
          output(host_core_query.size()), selected(std::size_t(rows) * QwenSparseSelected), count(rows), accepted(1),
          observations(std::size_t(rows) * 24), partials(std::size_t(rows) * 24 * QwenSparsePartitions),
          keys(host_key.size() * (representation == QwenSparseScalar::FP32 ? 4 : 2)), vals(keys.size)
    {
        check(QSA_RT(StreamCreateWithFlags)(&stream, QSA_RT(StreamNonBlocking)));
        for (int position = 0; position < positions; ++position)
            for (int pair = 0; pair < 32; ++pair)
            {
                const float angle = float(position) * std::pow(10000.0f, -float(pair) / 32.0f);
                cosine[std::size_t(position) * 32 + pair] = std::cos(angle);
                sine[std::size_t(position) * 32 + pair] = std::sin(angle);
            }
        for (std::size_t i = 0; i < host_key.size(); ++i)
        {
            key16[i] = representation == QwenSparseScalar::FP16 ? detail::encodeHalf(host_key[i]) : std::uint16_t(detail::floatBits(detail::roundRepresentation(host_key[i], QwenSparseScalar::BF16)) >> 16);
            value16[i] = representation == QwenSparseScalar::FP16 ? detail::encodeHalf(host_value[i]) : std::uint16_t(detail::floatBits(detail::roundRepresentation(host_value[i], QwenSparseScalar::BF16)) >> 16);
        }
        state.upload(&replay, stream); query.upload(host_query.data(), stream); raw.upload(host_raw.data(), stream);
        partial.upload(host_partial.data(), stream); pools.upload(host_pools.data(), stream);
        query_norm.upload(norm.data(), stream); key_norm.upload(norm.data(), stream);
        cos.upload(cosine.data(), stream); sin.upload(sine.data(), stream);
        core_query.upload(host_core_query.data(), stream); gate.upload(host_gate.data(), stream);
        keys.upload(reinterpret_cast<const unsigned char *>(representation == QwenSparseScalar::FP32 ? static_cast<const void *>(host_key.data()) : key16.data()), stream);
        vals.upload(reinterpret_cast<const unsigned char *>(representation == QwenSparseScalar::FP32 ? static_cast<const void *>(host_value.data()) : value16.data()), stream);
        indexer = {.replay = state.data, .query = query.data, .raw_keys = raw.data, .partial_keys = partial.data,
            .pooled_keys = pools.data, .query_norm = query_norm.data, .key_norm = key_norm.data,
            .cosine = cos.data, .sine = sin.data, .prepared_query = prepared.data, .scores = scores.data,
            .selected = selected.data, .selected_count = count.data, .row_capacity = rows, .token_capacity = capacity,
            .block_capacity = blocks, .position_capacity = positions, .representation = representation};
        const std::size_t scalar = representation == QwenSparseScalar::FP32 ? 4 : 2;
        attention = {.replay = state.data, .query = core_query.data,
            .kv = {.keys = {.data = keys.data, .token_stride_bytes = scalar * 512, .head_stride_bytes = scalar * 256,
                           .codec = representation == QwenSparseScalar::FP32 ? TensorType::FP32 : representation == QwenSparseScalar::FP16 ? TensorType::FP16 : TensorType::BF16},
                   .values = {.data = vals.data, .token_stride_bytes = scalar * 512, .head_stride_bytes = scalar * 256,
                           .codec = representation == QwenSparseScalar::FP32 ? TensorType::FP32 : representation == QwenSparseScalar::FP16 ? TensorType::FP16 : TensorType::BF16}},
            .selected = selected.data, .selected_count = count.data, .gate = gate.data, .output = output.data,
            .gpu_partials = partials.data, .observations = observations.data, .row_capacity = rows, .token_capacity = capacity};
        check(QSA_RT(StreamSynchronize)(stream));
    }
    /** @brief Retire graph objects after all terminal collection has completed. */
    ~Fixture()
    {
        if (stream && QSA_RT(StreamSynchronize)(stream) != QSA_RT(Success)) std::terminate();
        if (executable) (void)QSA_RT(GraphExecDestroy)(executable);
        if (graph) (void)QSA_RT(GraphDestroy)(graph);
        if (stream) (void)QSA_RT(StreamDestroy)(stream);
        stream = nullptr;
    }
    /** @brief Record one complete six-kernel production sparse transaction. */
    void capture()
    {
        check(QSA_RT(StreamBeginCapture)(stream, QSA_RT(StreamCaptureModeThreadLocal)));
        try
        {
            QSA_INDEXER(indexer, stream);
            QSA_ATTENTION(attention, stream);
        }
        catch (...)
        {
            QSA_GRAPH abandoned = nullptr;
            (void)QSA_RT(StreamEndCapture)(stream, &abandoned);
            if (abandoned) (void)QSA_RT(GraphDestroy)(abandoned);
            throw;
        }
        check(QSA_RT(StreamEndCapture)(stream, &graph));
        std::size_t nodes = 0;
        check(QSA_RT(GraphGetNodes)(graph, nullptr, &nodes));
        EXPECT_EQ(nodes, detail::isRotated(attention.kv.keys) ? 7u : 6u);
        check(QSA_RT(GraphInstantiate)(&executable, graph, nullptr, nullptr, 0));
    }
    /** @brief Replay the retained production graph without allocation or host state reads. */
    void launch() { check(QSA_RT(GraphLaunch)(executable, stream)); }
};

/** @test One retained graph republishes exact preappend rows and copies only live tail words. */
TEST(QSA_SUITE, CapturedPreAppendReplayAndCheckpointExtents)
{
    QSA_STREAM stream=nullptr;QSA_GRAPH graph=nullptr;QSA_EXEC executable=nullptr;
    check(QSA_RT(StreamCreateWithFlags)(&stream,QSA_RT(StreamNonBlocking)));
    constexpr int capacity=17,rows=5;
    Buffer<int> cursor(1),prefix(1),live(1),positions(rows),header(1);
    Buffer<QwenSparseReplay> replay(1);
    Buffer<attention::AttentionDeviceParams> metadata(rows);
    Buffer<std::uint32_t> source(3*128),destination(3*128);
    std::vector<std::uint32_t> bits(source.size),poison(source.size,0xdeadbeefu);
    for(std::size_t i=0;i<bits.size();++i)bits[i]=0x7fc00000u+static_cast<unsigned>(i);
    source.upload(bits.data(),stream);
    const QwenSparsePrepareReplay prepare{{cursor.data,prefix.data,capacity,QwenSparseRingCursor::NextWriteRow},
        live.data,positions.data,replay.data,metadata.data,rows,0};
    const QwenStateTransfer copy{source.data,destination.data,128*4,3,
        QwenStateRowSource::CanonicalModuloFour,prefix.data,0,header.data};
    check(QSA_RT(StreamSynchronize)(stream));
    check(QSA_RT(StreamBeginCapture)(stream,QSA_RT(StreamCaptureModeThreadLocal)));
    QSA_PREPARE(prepare,stream);QSA_TRANSFER(copy,stream);
    check(QSA_RT(StreamEndCapture)(stream,&graph));
    check(QSA_RT(GraphInstantiate)(&executable,graph,nullptr,nullptr,0));
    for(int count:{7,4,6,5,7})for(int active:{5,1,0,3}) {
        const int origin=12,next=(origin+count)%capacity;
        int absolute[rows];for(int row=0;row<rows;++row)absolute[row]=100+count+row;
        cursor.upload(&next,stream);prefix.upload(&count,stream);live.upload(&active,stream);positions.upload(absolute,stream);
        destination.upload(poison.data(),stream);
        check(QSA_RT(GraphLaunch)(executable,stream));
        const auto result=replay.download(stream);const auto params=metadata.download(stream);
        const auto actual=destination.download(stream);const auto valid=header.download(stream);
        ASSERT_EQ(result[0].prefix_tokens,count);ASSERT_EQ(result[0].live_rows,active);ASSERT_EQ(result[0].position_origin,100);
        ASSERT_EQ(valid[0],count%4);
        for(int row=0;row<rows;++row) {
            EXPECT_EQ(params[row].kv_len,row<active ? count+row+1 : 0);
            EXPECT_EQ(params[row].ring_row_origin,row<active ? origin : 0);
            EXPECT_EQ(params[row].position_offset,row<active ? absolute[row] : 0);
        }
        for(std::size_t word=0;word<actual.size();++word)
            EXPECT_EQ(actual[word],word<std::size_t(count%4)*128 ? bits[word] : poison[word]);
    }
    check(QSA_RT(GraphExecDestroy)(executable));check(QSA_RT(GraphDestroy)(graph));check(QSA_RT(StreamDestroy)(stream));
}

/** @test Captured analytic query preparation uses independent FP32 rotary phases at long positions. */
TEST(QSA_SUITE, CapturedAnalyticRotaryMatchesIndependentPyTorchPhases)
{
    for(float theta:{10000.0f,10000000.0f})for(int position:{0,17,4096,16384,65536}) {
        Fixture f(1,0,8,QwenSparseScalar::FP32);
        f.replay.position_origin=position;f.state.upload(&f.replay,f.stream);
        std::fill(f.host_query.begin(),f.host_query.end(),0.0f);
        for(int head=0;head<4;++head)std::fill_n(f.host_query.data()+head*128,32,1.0f);
        f.query.upload(f.host_query.data(),f.stream);
        f.indexer.rotary_source=QwenSparseRotarySource::AnalyticTheta;f.indexer.rope_theta=theta;
        f.indexer.inverse_frequencies=makeQwenSparsePowerFrequencies(theta,64);
        f.indexer.cosine=f.indexer.sine=nullptr;
        check(QSA_RT(StreamSynchronize)(f.stream));f.capture();f.launch();
        const auto actual=f.prepared.download(f.stream);
        const float inverse=1.0f/std::sqrt(0.25f+1e-6f);
        for(const auto &phase:test::RotaryReferences)if(phase.theta==theta && phase.position==position)
            for(int head=0;head<4;++head) {
                EXPECT_NEAR(actual[head*128+phase.pair],inverse*phase.cosine,3e-6f);
                EXPECT_NEAR(actual[head*128+phase.pair+32],inverse*phase.sine,3e-6f);
            }
    }
}

TEST(QSA_SUITE, CapturedCausalSelectionNativeReadsAndReset)
{
    for (auto representation : {QwenSparseScalar::FP32, QwenSparseScalar::FP16, QwenSparseScalar::BF16})
    {
        Fixture f(15, 2047, 4096, representation);
        std::vector<float> prepared(f.prepared.size), scores(f.scores.size);
        std::vector<std::int32_t> selected(f.selected.size), count(f.rows);
        auto reference = f.indexer;
        reference.replay = &f.replay; reference.query = f.host_query.data(); reference.raw_keys = f.host_raw.data();
        reference.partial_keys = f.host_partial.data(); reference.pooled_keys = f.host_pools.data();
        reference.query_norm = reference.key_norm = f.norm.data(); reference.cosine = f.cosine.data(); reference.sine = f.sine.data();
        reference.prepared_query = prepared.data(); reference.scores = scores.data(); reference.selected = selected.data(); reference.selected_count = count.data();
        cpuQwenSparseIndexer(reference);
        check(QSA_RT(MemsetAsync)(f.prepared.data, 0xff, f.prepared.size * sizeof(float), f.stream));
        check(QSA_RT(MemsetAsync)(f.scores.data, 0xff, f.scores.size * sizeof(float), f.stream));
        check(QSA_RT(MemsetAsync)(f.partials.data, 0xff, f.partials.size * sizeof(QwenSparseAttentionPartial), f.stream));
        check(QSA_RT(MemsetAsync)(f.selected.data, 0xff, f.selected.size * sizeof(std::int32_t), f.stream));
        check(QSA_RT(MemsetAsync)(f.count.data, 0xff, f.count.size * sizeof(std::int32_t), f.stream));
        f.capture(); f.launch();
        auto actual_selected = f.selected.download(f.stream);
        auto actual_count = f.count.download(f.stream);
        auto actual_prepared = f.prepared.download(f.stream);
        for (int row = 0; row < f.rows; ++row)
        {
            ASSERT_EQ(actual_count[row], count[row]);
            EXPECT_EQ(std::memcmp(actual_selected.data() + row * QwenSparseSelected, selected.data() + row * QwenSparseSelected,
                                  count[row] * sizeof(std::int32_t)), 0);
        }
        for (std::size_t i = 0; i < prepared.size(); ++i)
            EXPECT_NEAR(actual_prepared[i], prepared[i], representation == QwenSparseScalar::FP32 ? 2e-6f : .032f);
        auto actual_output = f.output.download(f.stream);
        std::vector<float> expected_output(actual_output.size());
        auto attn = f.attention;
        attn.replay = &f.replay; attn.query = f.host_core_query.data(); attn.gate = f.host_gate.data();
        attn.kv.keys.data = representation == QwenSparseScalar::FP32 ? static_cast<const void *>(f.host_key.data()) : f.key16.data();
        attn.kv.values.data = representation == QwenSparseScalar::FP32 ? static_cast<const void *>(f.host_value.data()) : f.value16.data();
        attn.selected = actual_selected.data(); attn.selected_count = actual_count.data(); attn.output = expected_output.data(); attn.observations = nullptr;
        cpuQwenSparseAttention(attn);
        for (std::size_t i = 0; i < expected_output.size(); ++i) EXPECT_NEAR(actual_output[i], expected_output[i], 3e-6f);
        auto observations = f.observations.download(f.stream);
        for (int row = 0; row < f.rows; ++row)
            for (int head = 0; head < 24; ++head)
            {
                const auto &observation = observations[row * 24 + head];
                EXPECT_EQ(observation.selected_tokens, actual_count[row]);
                EXPECT_EQ(observation.key_elements, std::uint64_t(actual_count[row]) * 256);
                EXPECT_EQ(observation.value_elements, std::uint64_t(actual_count[row]) * 256);
                EXPECT_LT(observation.key_elements, std::uint64_t(f.capacity) * 256);
            }
        std::vector<float> zero_query(f.host_query.size(), 0.0f);
        f.query.upload(zero_query.data(), f.stream); f.launch();
        actual_selected = f.selected.download(f.stream); actual_count = f.count.download(f.stream);
        for (int row = 0; row < f.rows; ++row)
        {
            const int visible = 2048 + row;
            ASSERT_EQ(actual_count[row], 2048 + visible % 4);
            for (int token = 0; token < 2048; ++token) EXPECT_EQ(actual_selected[row * QwenSparseSelected + token], token);
            for (int tail = 0; tail < visible % 4; ++tail)
                EXPECT_EQ(actual_selected[row * QwenSparseSelected + 2048 + tail], visible / 4 * 4 + tail);
        }
        // Reuse the identical graph after a large-to-small and then empty reset.
        check(QSA_RT(MemsetAsync)(f.partials.data, 0xff, f.partials.size * sizeof(QwenSparseAttentionPartial), f.stream));
        f.replay = {0, 1, 17}; f.state.upload(&f.replay, f.stream); f.launch();
        actual_count = f.count.download(f.stream);
        EXPECT_EQ(actual_count[0], 1);
        for (int row = 1; row < f.rows; ++row) EXPECT_EQ(actual_count[row], 0);
        f.replay.live_rows = 0; f.state.upload(&f.replay, f.stream); f.launch();
        actual_count = f.count.download(f.stream);
        observations = f.observations.download(f.stream);
        EXPECT_TRUE(std::all_of(actual_count.begin(), actual_count.end(), [](int n) { return n == 0; }));
        for (const auto &observation : observations) EXPECT_EQ(observation.key_elements + observation.value_elements, 0u);
    }
}

TEST(QSA_SUITE, CapturedGroupedMatchesSerialBytesAndAcceptanceRollback)
{
    Fixture f(15, 2050, 4096, QwenSparseScalar::FP16);
    f.capture(); f.launch();
    const auto grouped_selected = f.selected.download(f.stream);
    const auto grouped_count = f.count.download(f.stream);
    const auto grouped_output = f.output.download(f.stream);
    const auto grouped_prepared = f.prepared.download(f.stream);
    auto indexer = f.indexer;
    auto attention = f.attention;
    indexer.row_capacity = attention.row_capacity = 1;
    const std::int32_t one = 1;
    f.accepted.upload(&one, f.stream);
    for (int row = 0; row < 15; ++row)
    {
        f.replay = {2050 + row, 1, 17}; f.state.upload(&f.replay, f.stream);
        indexer.query = f.query.data + row * 512; indexer.raw_keys = f.raw.data + row * 128;
        attention.query = f.core_query.data + row * 6144; attention.gate = f.gate.data + row * 6144;
        QSA_GRAPH graph = nullptr; QSA_EXEC executable = nullptr;
        check(QSA_RT(StreamBeginCapture)(f.stream, QSA_RT(StreamCaptureModeThreadLocal)));
        QSA_INDEXER(indexer, f.stream); QSA_ATTENTION(attention, f.stream);
        QSA_COMMIT((QwenSparsePartialCommit{.replay = f.state.data, .prior_partial = f.partial.data,
            .raw_keys = indexer.raw_keys, .acceptance = AcceptedRequestPrefixes::counted(DeviceRequestRowRanges::deviceCounted(1,1,&f.state.data->live_rows),f.accepted.data), .partial_keys = f.partial.data}), f.stream);
        check(QSA_RT(StreamEndCapture)(f.stream, &graph));
        check(QSA_RT(GraphInstantiate)(&executable, graph, nullptr, nullptr, 0));
        check(QSA_RT(GraphLaunch)(executable, f.stream));
        const auto actual_count = f.count.download(f.stream);
        const auto actual_selected = f.selected.download(f.stream);
        const auto actual_prepared = f.prepared.download(f.stream);
        const auto actual_output = f.output.download(f.stream);
        ASSERT_EQ(actual_count[0], grouped_count[row]);
        EXPECT_EQ(std::memcmp(actual_selected.data(), grouped_selected.data() + row * QwenSparseSelected, actual_count[0] * sizeof(std::int32_t)), 0);
        EXPECT_EQ(std::memcmp(actual_prepared.data(), grouped_prepared.data() + row * 512, 512 * sizeof(float)), 0);
        EXPECT_EQ(std::memcmp(actual_output.data(), grouped_output.data() + row * 6144, 6144 * sizeof(float)), 0);
        check(QSA_RT(GraphExecDestroy)(executable)); check(QSA_RT(GraphDestroy)(graph));
    }
    // Restore the transaction base and accept three rows, crossing the pool
    // boundary. Future completed pools remain unreachable until re-prepared.
    f.replay = {2050, 15, 17}; f.state.upload(&f.replay, f.stream); f.partial.upload(f.host_partial.data(), f.stream);
    const std::int32_t three = 3; f.accepted.upload(&three, f.stream);
    const auto rollbackAcceptance = AcceptedRequestPrefixes::counted(
        DeviceRequestRowRanges::deviceCounted(1, f.rows, &f.state.data->live_rows), f.accepted.data);
    // The grouped transaction still owns all15 candidates; only this decision
    // accepts three. Narrowing capacity to the serial oracle's one-row lane
    // would reject the native live count before the rollback kernel can run.
    ASSERT_EQ(rollbackAcceptance.resolveFor(0, f.replay.live_rows, three, 0), three);
    QSA_COMMIT((QwenSparsePartialCommit{.replay = f.state.data, .prior_partial = f.partial.data,
        .raw_keys = f.raw.data, .acceptance = rollbackAcceptance, .partial_keys = f.partial.data}), f.stream);
    const auto partial = f.partial.download(f.stream);
    EXPECT_EQ(std::memcmp(partial.data(), f.host_raw.data() + 2 * 128, 128 * sizeof(float)), 0);
}

TEST(QSA_SUITE, EveryNativeCodecAndQ16BlockUsesCanonicalDeviceRingAndBasis)
{
    Fixture f(2, 5, 8, QwenSparseScalar::FP16);
    std::vector<attention::AttentionDeviceParams> mapping(2);
    for (int row = 0; row < 2; ++row)
        mapping[row] = {.kv_len = 6 + row, .kv_stride = 8, .position_offset = 22 + row,
                        .ring_row_origin = 6, .ring_row_capacity = 8};
    Buffer<attention::AttentionDeviceParams> device_mapping(2);
    device_mapping.upload(mapping.data(), f.stream);
    const auto codecs = {TensorType::FP32, TensorType::FP16, TensorType::BF16, TensorType::Q8_1,
                         TensorType::Q16_1, TensorType::AQ8, TensorType::TQ4, TensorType::TQ8};
    for (auto key_codec : codecs)
    for (auto value_codec : codecs)
    {
        if (value_codec == TensorType::AQ8) continue;
        for (int block_size : {32, 64, 128})
        {
            if (block_size != 32 && key_codec != TensorType::Q16_1 && value_codec != TensorType::Q16_1) continue;
            test::NativeCodecFixture key_fixture(key_codec, 8, block_size), value_fixture(value_codec, 8, block_size);
            Buffer<unsigned char> keys(key_fixture.storage.size()), values(value_fixture.storage.size());
            Buffer<float> key_rotation(2 * 65536), value_rotation(2 * 65536), key_centroids(key_fixture.centroids.size()),
                          value_centroids(value_fixture.centroids.size()), anchor(512);
            keys.upload(key_fixture.storage.data(), f.stream); values.upload(value_fixture.storage.data(), f.stream);
            key_rotation.upload(key_fixture.rotation.data(), f.stream); value_rotation.upload(value_fixture.rotation.data(), f.stream);
            key_centroids.upload(key_fixture.centroids.data(), f.stream); value_centroids.upload(value_fixture.centroids.data(), f.stream);
            anchor.upload(key_fixture.anchor.data(), f.stream);
            f.attention.kv = {.keys = key_fixture.bank(), .values = value_fixture.bank(),
                .addressing = QwenSparseKVAddressing::RowParameters, .row_parameters = device_mapping.data};
            f.attention.kv.keys.data = keys.data; f.attention.kv.values.data = values.data;
            f.attention.kv.keys.rotations[0] = key_rotation.data; f.attention.kv.keys.rotations[1] = key_rotation.data + 65536;
            f.attention.kv.values.rotations[0] = value_rotation.data; f.attention.kv.values.rotations[1] = value_rotation.data + 65536;
            f.attention.kv.keys.centroids = key_centroids.data; f.attention.kv.values.centroids = value_centroids.data;
            f.attention.kv.keys.anchor = anchor.data;
            f.replay = {5, 2, 17}; f.state.upload(&f.replay, f.stream);
            check(QSA_RT(StreamSynchronize)(f.stream));
            f.capture(); f.launch();
            const auto selected = f.selected.download(f.stream);
            const auto count = f.count.download(f.stream);
            const auto actual = f.output.download(f.stream);
            const auto observations = f.observations.download(f.stream);
            std::vector<float> expected(actual.size());
            auto reference = f.attention;
            reference.replay = &f.replay; reference.query = f.host_core_query.data(); reference.gate = f.host_gate.data();
            reference.kv.keys = key_fixture.reference(true); reference.kv.values = value_fixture.reference();
            reference.kv.row_parameters = mapping.data(); reference.selected = selected.data(); reference.selected_count = count.data();
            reference.output = expected.data(); reference.observations = nullptr;
            cpuQwenSparseAttention(reference);
            for (std::size_t i = 0; i < actual.size(); ++i) ASSERT_NEAR(actual[i], expected[i], 1e-5f);
            for (int row = 0; row < 2; ++row)
                for (int head = 0; head < 24; ++head)
                {
                    EXPECT_EQ(observations[row * 24 + head].key_elements, std::uint64_t(count[row]) * 256);
                    EXPECT_EQ(observations[row * 24 + head].value_elements, std::uint64_t(count[row]) * 256);
                }
            for (int row = 0; row < 2; ++row)
            {
                f.replay = {5 + row, 1, 17}; f.state.upload(&f.replay, f.stream);
                auto serial = f.attention;
                serial.query = f.core_query.data + row * 6144; serial.gate = f.gate.data + row * 6144;
                serial.selected = f.selected.data + row * QwenSparseSelected; serial.selected_count = f.count.data + row;
                serial.kv.row_parameters = device_mapping.data + row; serial.row_capacity = 1;
                QSA_GRAPH graph = nullptr; QSA_EXEC executable = nullptr;
                check(QSA_RT(StreamBeginCapture)(f.stream, QSA_RT(StreamCaptureModeThreadLocal)));
                QSA_ATTENTION(serial, f.stream);
                check(QSA_RT(StreamEndCapture)(f.stream, &graph));
                check(QSA_RT(GraphInstantiate)(&executable, graph, nullptr, nullptr, 0));
                check(QSA_RT(GraphLaunch)(executable, f.stream));
                const auto serial_output = f.output.download(f.stream);
                EXPECT_EQ(std::memcmp(serial_output.data(), actual.data() + row * 6144, 6144 * sizeof(float)), 0);
                check(QSA_RT(GraphExecDestroy)(executable)); check(QSA_RT(GraphDestroy)(graph));
            }
            f.replay = {5, 2, 17}; f.state.upload(&f.replay, f.stream);
            for (auto &params : mapping) params.ring_row_origin = 1;
            device_mapping.upload(mapping.data(), f.stream);
            f.launch();
            const auto remapped = f.output.download(f.stream);
            reference.replay = &f.replay;
            cpuQwenSparseAttention(reference);
            for (std::size_t i = 0; i < remapped.size(); ++i) ASSERT_NEAR(remapped[i], expected[i], 1e-5f);
            for (auto &params : mapping) params.ring_row_origin = 6;
            device_mapping.upload(mapping.data(), f.stream);
            check(QSA_RT(GraphExecDestroy)(f.executable)); f.executable = nullptr;
            check(QSA_RT(GraphDestroy)(f.graph)); f.graph = nullptr;
        }
    }
}

TEST(QSA_SUITE, FloatingPreRotaryTransformsOnlySelectedKeys)
{
    Fixture f(3, 3, 16, QwenSparseScalar::FP16);
    f.attention.kv.key_policy = {.encoding = attention::AttentionKeyCacheEncoding::PreRotaryDeviceTransform,
                                 .rope_theta = 10000.0f, .partial_rotary_factor = 0.25f};
    f.attention.kv.key_cosine = f.cos.data; f.attention.kv.key_sine = f.sin.data;
    f.attention.kv.key_rotary_dim = 64; f.attention.kv.key_position_capacity = f.positions;
    f.capture(); f.launch();
    const auto selected = f.selected.download(f.stream);
    const auto count = f.count.download(f.stream);
    const auto actual = f.output.download(f.stream);
    const auto observations = f.observations.download(f.stream);
    std::vector<float> rotated(f.host_key.size());
    for (int token = 0; token < 16; ++token)
        for (int head = 0; head < 2; ++head)
            for (int d = 0; d < 256; ++d)
            {
                const std::size_t offset = (std::size_t(token) * 2 + head) * 256;
                float value = detail::decodeHalf(f.key16[offset + d]);
                if (d < 64)
                {
                    const int partner = d < 32 ? d + 32 : d - 32;
                    const float c = detail::roundRepresentation(f.cosine[std::size_t(17 + token) * 32 + d % 32], QwenSparseScalar::FP16);
                    const float s = detail::roundRepresentation(f.sine[std::size_t(17 + token) * 32 + d % 32], QwenSparseScalar::FP16);
                    const float other = detail::decodeHalf(f.key16[offset + partner]) * (d < 32 ? -1.0f : 1.0f);
                    const float first = detail::roundRepresentation(value * c, QwenSparseScalar::FP16);
                    const float second = detail::roundRepresentation(other * s, QwenSparseScalar::FP16);
                    value = detail::roundRepresentation(first + second, QwenSparseScalar::FP16);
                }
                rotated[offset + d] = value;
            }
    std::vector<float> expected(actual.size());
    auto reference = f.attention;
    reference.replay = &f.replay; reference.query = f.host_core_query.data(); reference.gate = f.host_gate.data();
    reference.kv.keys = {.data = rotated.data(), .token_stride_bytes = 512 * sizeof(float), .head_stride_bytes = 256 * sizeof(float)};
    reference.kv.values.data = f.value16.data();
    reference.kv.key_policy = {};
    reference.selected = selected.data(); reference.selected_count = count.data(); reference.output = expected.data(); reference.observations = nullptr;
    cpuQwenSparseAttention(reference);
    for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 3e-6f);
    for (int row = 0; row < 3; ++row)
        for (int head = 0; head < 24; ++head)
        {
            EXPECT_EQ(observations[row * 24 + head].key_elements, std::uint64_t(count[row]) * (256 + 64));
            EXPECT_EQ(observations[row * 24 + head].value_elements, std::uint64_t(count[row]) * 256);
        }
}

/** @test Selected pre-RoPE keys use the actual ordinary-RoPE invariant producer at long positions. */
TEST(QSA_SUITE, NativeCoreRotaryMatchesCanonicalEngineFrequencyPublication)
{
    Fixture f(1,7,8,QwenSparseScalar::FP32);
    f.replay.position_origin=65536;f.state.upload(&f.replay,f.stream);
    f.indexer.rotary_source=QwenSparseRotarySource::AnalyticTheta;f.indexer.rope_theta=10000000.0f;
    f.indexer.inverse_frequencies=makeQwenSparsePowerFrequencies(f.indexer.rope_theta,64);
    f.indexer.cosine=f.indexer.sine=nullptr;
    f.attention.kv.key_policy={.encoding=attention::AttentionKeyCacheEncoding::PreRotaryDeviceTransform,
        .rope_theta=10000000.0f,.partial_rotary_factor=0.25f};
    f.attention.kv.key_rotary_source=QwenSparseRotarySource::CanonicalEngineTheta;
    f.attention.kv.key_negative_log_theta=-std::log(10000000.0f);
    Buffer<float> frequency(32);int ordinal=0;check(QSA_RT(GetDevice)(&ordinal));
    ASSERT_TRUE(QSA_POPULATE_ROPE(frequency.data,64,10000000.0f,ordinal,f.stream));
    Buffer<float> canonicalKeys(f.host_key.size());canonicalKeys.upload(f.host_key.data(),f.stream);
    for(int token=0;token<f.capacity;++token)
        ASSERT_TRUE(QSA_APPLY_ROPE(canonicalKeys.data+std::size_t(token)*512,nullptr,frequency.data,
            65536+token,2,0,256,64,ordinal,f.stream));
    const auto rotated=canonicalKeys.download(f.stream);
    f.capture();f.launch();
    const auto selected=f.selected.download(f.stream),count=f.count.download(f.stream);
    const auto actual=f.output.download(f.stream);
    std::vector<float> expected(actual.size());auto reference=f.attention;
    reference.replay=&f.replay;reference.query=f.host_core_query.data();reference.gate=f.host_gate.data();
    reference.kv.keys.data=rotated.data();reference.kv.values.data=f.host_value.data();reference.kv.key_policy={};
    reference.selected=selected.data();reference.selected_count=count.data();reference.output=expected.data();reference.observations=nullptr;
    cpuQwenSparseAttention(reference);
    for(std::size_t i=0;i<actual.size();++i)ASSERT_NEAR(actual[i],expected[i],3e-6f);
}

TEST(QSA_SUITE, ParticipantLocalTwelveQueryOneKVHeadMatchesFullHeadBytes)
{
    Fixture f(2, 3, 16, QwenSparseScalar::FP16);
    f.capture(); f.launch();
    const auto full = f.output.download(f.stream);
    check(QSA_RT(GraphExecDestroy)(f.executable)); f.executable = nullptr;
    check(QSA_RT(GraphDestroy)(f.graph)); f.graph = nullptr;
    std::vector<float> query(2 * 3072), gate(query.size());
    for (int row = 0; row < 2; ++row)
    {
        std::copy_n(f.host_core_query.data() + row * 6144 + 3072, 3072, query.data() + row * 3072);
        std::copy_n(f.host_gate.data() + row * 6144 + 3072, 3072, gate.data() + row * 3072);
    }
    Buffer<float> local_query(query.size()), local_gate(gate.size());
    local_query.upload(query.data(), f.stream); local_gate.upload(gate.data(), f.stream);
    f.attention.query = local_query.data; f.attention.gate = local_gate.data;
    f.attention.query_heads = 12; f.attention.kv_heads = 1;
    f.attention.query_head_start = 12; f.attention.kv_head_start = 1;
    f.attention.kv.keys.data = f.keys.data + f.attention.kv.keys.head_stride_bytes;
    f.attention.kv.values.data = f.vals.data + f.attention.kv.values.head_stride_bytes;
    check(QSA_RT(StreamSynchronize)(f.stream));
    f.capture(); f.launch();
    const auto local = f.output.download(f.stream);
    for (int row = 0; row < 2; ++row)
        EXPECT_EQ(std::memcmp(local.data() + row * 3072, full.data() + row * 6144 + 3072, 3072 * sizeof(float)), 0);
}

/** @test All eight TP8 Q3 intervals retain the full original two-KV cache and exact bytes. */
TEST(QSA_SUITE, ParticipantLocalThreeQueriesReplicatedTwoKVHeadsMatchFullHeadBytes)
{
    for(const auto representation:{QwenSparseScalar::FP32,QwenSparseScalar::FP16,QwenSparseScalar::BF16}) {
        Fixture f(2,3,16,representation);
        // This guard precedes full capture too, so an infrastructure exception
        // cannot free a cache or stream while the full oracle is outstanding.
        struct RetireGraph {
            Fixture &fixture;
            /** @brief Retire captured work before local query/gate operands leave scope. */
            ~RetireGraph() noexcept {
                if(QSA_RT(StreamSynchronize)(fixture.stream)!=QSA_RT(Success))std::terminate();
                if(fixture.executable) {
                    if(QSA_RT(GraphExecDestroy)(fixture.executable)!=QSA_RT(Success))std::terminate();
                    fixture.executable=nullptr;
                }
                if(fixture.graph) {
                    if(QSA_RT(GraphDestroy)(fixture.graph)!=QSA_RT(Success))std::terminate();
                    fixture.graph=nullptr;
                }
            }
        } fullRetirement{f};
        f.capture();f.launch();const auto full=f.output.download(f.stream);
        check(QSA_RT(GraphExecDestroy)(f.executable));f.executable=nullptr;
        check(QSA_RT(GraphDestroy)(f.graph));f.graph=nullptr;
        for(int participant=0;participant<8;++participant) {
            constexpr int width=3*256;
            const int start=3*participant;
            std::vector<float> query(f.rows*width),gate(query.size()),poison(f.output.size,-12345.0f);
            for(int row=0;row<f.rows;++row) {
                std::copy_n(f.host_core_query.data()+row*6144+start*256,width,query.data()+row*width);
                std::copy_n(f.host_gate.data()+row*6144+start*256,width,gate.data()+row*width);
            }
            Buffer<float> localQuery(query.size()),localGate(gate.size());
            RetireGraph localRetirement{f}; // last: captured local operands retire first
            localQuery.upload(query.data(),f.stream);localGate.upload(gate.data(),f.stream);
            // Each replay starts at the same original indexer prestate. The
            // indexer mutates these banks; a previous participant is no oracle.
            f.partial.upload(f.host_partial.data(),f.stream);
            f.pools.upload(f.host_pools.data(),f.stream);
            f.state.upload(&f.replay,f.stream);f.output.upload(poison.data(),f.stream);
            f.attention.query=localQuery.data;f.attention.gate=localGate.data;
            f.attention.query_heads=3;f.attention.kv_heads=2;
            f.attention.query_head_start=start;f.attention.kv_head_start=0;
            // Cache pointers/strides remain the complete native source bank.
            auto invalid=f.attention;invalid.query_head_start=22;
            EXPECT_THROW(validateQwenSparseAttention(invalid),std::invalid_argument);
            invalid=f.attention;invalid.query_head_start=-1;
            EXPECT_THROW(validateQwenSparseAttention(invalid),std::invalid_argument);
            invalid=f.attention;invalid.query_head_start=21;invalid.kv_heads=1;
            EXPECT_THROW(validateQwenSparseAttention(invalid),std::invalid_argument);
            check(QSA_RT(StreamSynchronize)(f.stream));
            f.capture();f.launch();const auto local=f.output.download(f.stream);
            for(int row=0;row<f.rows;++row)
                EXPECT_EQ(std::memcmp(local.data()+row*width,full.data()+row*6144+start*256,width*sizeof(float)),0)
                    <<"participant="<<participant<<" row="<<row;
            for(std::size_t i=f.rows*width;i<local.size();++i)EXPECT_EQ(local[i],poison[i]);
        }
    }
}

TEST(QSA_SUITE, ExactCandidateProfilerLaunch)
{
    const char *rows_env = std::getenv("LLAMINAR_QSA_PROFILE_M");
    const char *context_env = std::getenv("LLAMINAR_QSA_PROFILE_CONTEXT");
    if (!rows_env || !context_env) GTEST_SKIP() << "Set exact candidate M and context for isolated profiling";
    const int rows = std::stoi(rows_env), context = std::stoi(context_env);
    ASSERT_GT(rows, 0); ASSERT_GE(context, rows);
    Fixture f(rows, context - rows, context, QwenSparseScalar::FP16);
    std::unique_ptr<test::NativeCodecFixture> tq_keys, tq_values;
    std::unique_ptr<Buffer<unsigned char>> tq_key_bytes, tq_value_bytes;
    std::unique_ptr<Buffer<float>> tq_key_rotation, tq_value_rotation, tq_key_centroids, tq_value_centroids;
    if(std::getenv("LLAMINAR_QSA_PROFILE_PRODUCTION")) {
        f.indexer.representation=QwenSparseScalar::FP32;
        f.indexer.rotary_source=QwenSparseRotarySource::AnalyticTheta;
        f.indexer.rope_theta=10000000.0f;
        f.indexer.inverse_frequencies=makeQwenSparsePowerFrequencies(f.indexer.rope_theta,64);
        f.indexer.cosine=f.indexer.sine=nullptr;
    }
    if (std::getenv("LLAMINAR_QSA_PROFILE_TQ"))
    {
        tq_keys = std::make_unique<test::NativeCodecFixture>(TensorType::TQ8, context, 32, false);
        tq_values = std::make_unique<test::NativeCodecFixture>(TensorType::TQ4, context, 32, false);
        tq_key_bytes = std::make_unique<Buffer<unsigned char>>(tq_keys->storage.size());
        tq_value_bytes = std::make_unique<Buffer<unsigned char>>(tq_values->storage.size());
        tq_key_rotation = std::make_unique<Buffer<float>>(2 * 65536); tq_value_rotation = std::make_unique<Buffer<float>>(2 * 65536);
        tq_key_centroids = std::make_unique<Buffer<float>>(256); tq_value_centroids = std::make_unique<Buffer<float>>(16);
        tq_key_bytes->upload(tq_keys->storage.data(), f.stream); tq_value_bytes->upload(tq_values->storage.data(), f.stream);
        tq_key_rotation->upload(tq_keys->rotation.data(), f.stream); tq_value_rotation->upload(tq_values->rotation.data(), f.stream);
        tq_key_centroids->upload(tq_keys->centroids.data(), f.stream); tq_value_centroids->upload(tq_values->centroids.data(), f.stream);
        f.attention.kv.keys = tq_keys->bank(); f.attention.kv.values = tq_values->bank();
        f.attention.kv.keys.data = tq_key_bytes->data; f.attention.kv.values.data = tq_value_bytes->data;
        f.attention.kv.keys.rotations[0] = tq_key_rotation->data; f.attention.kv.keys.rotations[1] = tq_key_rotation->data + 65536;
        f.attention.kv.values.rotations[0] = tq_value_rotation->data; f.attention.kv.values.rotations[1] = tq_value_rotation->data + 65536;
        f.attention.kv.keys.centroids = tq_key_centroids->data; f.attention.kv.values.centroids = tq_value_centroids->data;
        check(QSA_RT(StreamSynchronize)(f.stream));
    }
    f.capture();
    if (std::getenv("LLAMINAR_QSA_TIMING"))
    {
        QSA_EVENT begin = nullptr, end = nullptr;
        check(QSA_RT(EventCreate)(&begin)); check(QSA_RT(EventCreate)(&end));
        for (int repeat = 0; repeat < 3; ++repeat) f.launch();
        check(QSA_RT(EventRecord)(begin, f.stream));
        for (int repeat = 0; repeat < 10; ++repeat) f.launch();
        check(QSA_RT(EventRecord)(end, f.stream)); check(QSA_RT(EventSynchronize)(end));
        float milliseconds = 0;
        check(QSA_RT(EventElapsedTime)(&milliseconds, begin, end));
        std::cout << "QSA_CANDIDATE,M=" << rows << ",context=" << context
                  << ",kv=" << (tq_keys ? "TQ8_TQ4" : "FP16")
                  << ",indexer=" << (std::getenv("LLAMINAR_QSA_PROFILE_PRODUCTION") ? "FP32_ANALYTIC" : "FP16_TABLE") << ",milliseconds=" << milliseconds / 10 << '\n';
        check(QSA_RT(EventDestroy)(begin)); check(QSA_RT(EventDestroy)(end));
    }
    else f.launch();
    auto counts = f.count.download(f.stream);
    auto observations = f.observations.download(f.stream);
    for (int row = 0; row < rows; ++row)
    {
        const int visible = context - rows + row + 1;
        ASSERT_EQ(counts[row], std::min(visible / 4, 512) * 4 + visible % 4);
        for (int head = 0; head < 24; ++head)
        {
            EXPECT_EQ(observations[row * 24 + head].key_elements, std::uint64_t(counts[row]) * 256);
            EXPECT_EQ(observations[row * 24 + head].value_elements, std::uint64_t(counts[row]) * 256);
        }
    }
}
/** @test Isolate acceptance-only publication for one exact profiler candidate. */
TEST(QSA_SUITE, ExactCommitProfilerLaunch)
{
    const char *rows_env=std::getenv("LLAMINAR_QSA_PROFILE_M");
    const char *context_env=std::getenv("LLAMINAR_QSA_PROFILE_CONTEXT");
    if(!rows_env || !context_env)GTEST_SKIP()<<"Set exact candidate M and context for isolated commit profiling";
    const int rows=std::stoi(rows_env),context=std::stoi(context_env);
    ASSERT_GT(rows,0);ASSERT_GE(context,rows);
    QSA_STREAM stream=nullptr;
    check(QSA_RT(StreamCreateWithFlags)(&stream,QSA_RT(StreamNonBlocking)));
    Buffer<QwenSparseReplay> state(1);
    Buffer<float> prior(512),raw(std::size_t(rows)*128),output(512);
    Buffer<std::int32_t> accepted(1);
    const QwenSparseReplay replay{context-rows,rows,0};
    const auto priorValues=values(512,71),rawValues=values(std::size_t(rows)*128,73);
    state.upload(&replay,stream);prior.upload(priorValues.data(),stream);raw.upload(rawValues.data(),stream);accepted.upload(&rows,stream);
    check(QSA_RT(StreamSynchronize)(stream));
    QSA_GRAPH graph=nullptr;QSA_EXEC executable=nullptr;
    check(QSA_RT(StreamBeginCapture)(stream,QSA_RT(StreamCaptureModeGlobal)));
    QSA_COMMIT((QwenSparsePartialCommit{state.data,prior.data,raw.data,AcceptedRequestPrefixes::counted(DeviceRequestRowRanges::deviceCounted(1,rows,&state.data->live_rows),accepted.data),output.data}),stream);
    check(QSA_RT(StreamEndCapture)(stream,&graph));
    check(QSA_RT(GraphInstantiate)(&executable,graph,nullptr,nullptr,0));
    check(QSA_RT(GraphLaunch)(executable,stream));
    const auto result=output.download(stream);
    const int tail=context%4;
    for(int row=0;row<tail;++row) {
        const int source=rows-tail+row;
        const float *expected=source<0 ? priorValues.data()+((context-rows)%4+source)*128 : rawValues.data()+source*128;
        EXPECT_EQ(std::memcmp(result.data()+row*128,expected,128*sizeof(float)),0);
    }
    check(QSA_RT(GraphExecDestroy)(executable));check(QSA_RT(GraphDestroy)(graph));check(QSA_RT(StreamDestroy)(stream));
}
} // namespace llaminar2::qwen4exp::QSA_SUITE

namespace llaminar2::qwen4exp::QSA_SUITE
{
/** @brief Last-declared join before the borrowed full-pipeline fixture can unwind. */
struct BoundedPipelineRetirement final
{
    QSA_STREAM stream;
    ~BoundedPipelineRetirement()
    {
        if (QSA_RT(StreamSynchronize)(stream) != QSA_RT(Success)) std::terminate();
    }
};
/** @test Actual FP32 producer, all cast policies, exact selection and native attention bytes. */
TEST(QSA_SUITE, BoundedCandidateFullIndexerAndAttentionBytes)
{
    for (auto policy : {QwenSparsePoolSelection::Bounded256_512,
                       QwenSparsePoolSelection::Bounded512_512,
                       QwenSparsePoolSelection::Bounded1024_1024})
        for (auto representation : {QwenSparseScalar::FP32, QwenSparseScalar::FP16, QwenSparseScalar::BF16})
        {
            Fixture installed(9, 4093, 4102, representation);
            BoundedPipelineRetirement installedRetirement{installed.stream};
            Fixture candidate(9, 4093, 4102, representation);
            BoundedPipelineRetirement candidateRetirement{candidate.stream};
            candidate.indexer.pool_selection = policy;
            installed.capture(); candidate.capture();
            for (int live : {9, 1, 0, 5})
            {
                installed.replay.live_rows = live; candidate.replay.live_rows = live;
                installed.state.upload(&installed.replay, installed.stream);
                candidate.state.upload(&candidate.replay, candidate.stream);
                // Every inactive score/index/output word is a dirty suffix. No
                // candidate may normalize it or read it to manufacture a count.
                std::vector<float> scorePoison(installed.scores.size, -123.0f);
                std::vector<float> outputPoison(installed.output.size, -77.0f);
                std::vector<std::int32_t> indexPoison(installed.selected.size, -99);
                BoundedPipelineRetirement poisonInstalled{installed.stream};
                BoundedPipelineRetirement poisonCandidate{candidate.stream};
                installed.scores.upload(scorePoison.data(), installed.stream);
                candidate.scores.upload(scorePoison.data(), candidate.stream);
                installed.output.upload(outputPoison.data(), installed.stream);
                candidate.output.upload(outputPoison.data(), candidate.stream);
                installed.selected.upload(indexPoison.data(), installed.stream);
                candidate.selected.upload(indexPoison.data(), candidate.stream);
                installed.launch(); candidate.launch();
                const auto aScores = installed.scores.download(installed.stream);
                const auto bScores = candidate.scores.download(candidate.stream);
                const auto aOutput = installed.output.download(installed.stream);
                const auto bOutput = candidate.output.download(candidate.stream);
                EXPECT_EQ(std::memcmp(aScores.data(), bScores.data(), aScores.size() * sizeof(float)), 0);
                EXPECT_EQ(installed.count.download(installed.stream), candidate.count.download(candidate.stream));
                EXPECT_EQ(installed.selected.download(installed.stream), candidate.selected.download(candidate.stream));
                EXPECT_EQ(std::memcmp(aOutput.data(), bOutput.data(), aOutput.size() * sizeof(float)), 0);
                ASSERT_FALSE(::testing::Test::HasFailure());
            }
        }
}
} // namespace llaminar2::qwen4exp::QSA_SUITE
