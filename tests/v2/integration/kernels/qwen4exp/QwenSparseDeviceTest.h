/**
 * @file QwenSparseDeviceTest.h
 * @brief Shared captured CUDA/HIP QSA conformance and exact-candidate harness.
 *
 * Backend translation units provide QSA_RT, QSA_STREAM, QSA_GRAPH, QSA_EXEC,
 * QSA_EVENT, and the three production launch bindings. All allocation, copies,
 * and result synchronization are explicit fixture boundaries outside capture.
 * The core attention observations prove the loops decoded selected rows only.
 * Compact consumer fixtures retain original head bytes/codec metadata and prove
 * logical degrees1..8 on one device, without implying eight physical devices.
 */
#pragma once

// Host-compiled adapters own the runtime; compiler-driver macros do not.
#if defined(QSA_CUDA_RUNTIME) == defined(QSA_ROCM_RUNTIME)
#error "QSA conformance requires exactly one explicit runtime adapter"
#endif
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "kernels/qwen4exp/QwenSparseMath.h"
#include "kernels/qwen4exp/QwenSparseReplay.h"
#include "kernels/qwen4exp/QwenSparseQueryRowWork.h"
#include "kernels/qwen4exp/Qwen4ExpStateIO.h"
#include "QwenSparseNativeCodecFixture.h"
#include "QwenSparseRotaryReference.h"
#include <gtest/gtest.h>
#include <cxxabi.h>
#include <algorithm>
#include <array>
#include <memory>
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
        const std::size_t indexerNodes = indexer.query_demand == QwenSparseQueryDemand::ProvenNoRanking ? 2u :
            rows <= llaminar2::attention::kMaxGroupedVerifierAttentionRows ? 3u : 4u;
        EXPECT_EQ(nodes, indexerNodes + (detail::isRotated(attention.kv.keys) ? 3u : 2u));
        check(QSA_RT(GraphInstantiate)(&executable, graph, nullptr, nullptr, 0));
    }
    /** @brief Replay the retained production graph without allocation or host state reads. */
    void launch() { check(QSA_RT(GraphLaunch)(executable, stream)); }
};

/**
 * @brief Prove optional observations leave every FP32 attention intermediate unchanged.
 * @details Compare the production captured ON result with a captured OFF launch
 * over the identical selected rows and immutable native codec buffers. Counter
 * fields are intentionally excluded: OFF leaves nonempty diagnostic storage
 * untouched, while empty partitions always retire their metadata.
 * @param f Fixture whose observed production graph has completed its last replay.
 */
void expectObservationNeutralAttention(Fixture &f)
{
    const auto expected_output = f.output.download(f.stream);
    const auto expected_partials = f.partials.download(f.stream);
    auto unobserved = f.attention;
    unobserved.observations = nullptr;
    QSA_GRAPH graph = nullptr; QSA_EXEC executable = nullptr;
    check(QSA_RT(StreamBeginCapture)(f.stream, QSA_RT(StreamCaptureModeThreadLocal)));
    QSA_ATTENTION(unobserved, f.stream);
    check(QSA_RT(StreamEndCapture)(f.stream, &graph));
    check(QSA_RT(GraphInstantiate)(&executable, graph, nullptr, nullptr, 0));
    check(QSA_RT(GraphLaunch)(executable, f.stream));
    const auto actual_output = f.output.download(f.stream);
    const auto actual_partials = f.partials.download(f.stream);
    EXPECT_EQ(std::memcmp(actual_output.data(), expected_output.data(), actual_output.size() * sizeof(float)), 0);
    for (std::size_t i = 0; i < actual_partials.size(); ++i)
    {
        EXPECT_EQ(std::memcmp(&actual_partials[i].maximum, &expected_partials[i].maximum, sizeof(float)), 0);
        EXPECT_EQ(std::memcmp(&actual_partials[i].denominator, &expected_partials[i].denominator, sizeof(float)), 0);
        EXPECT_EQ(std::memcmp(actual_partials[i].weighted, expected_partials[i].weighted, 256 * sizeof(float)), 0);
    }
    check(QSA_RT(GraphExecDestroy)(executable)); check(QSA_RT(GraphDestroy)(graph));
}

/**
 * @test Retained observation modes preserve exact partial/output bytes across
 * partition boundaries, empty rows and large-to-small request reset.
 * @details Existing all-codec and pre-RoPE cases independently cover decoded
 * coordinate totals and the CPU attention oracle; this case stresses lifecycle
 * transitions with deliberately dirty inactive payload and metadata.
 */
TEST(QSA_SUITE, CapturedObservationNeutralPartialBytesAndRetainedReset)
{
    for (int rows = 1; rows <= 16; ++rows)
    for (bool transform : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "M=" << rows << " transform=" << transform);
        Fixture on(rows, 2050, 4096, QwenSparseScalar::FP16);
        Fixture off(rows, 2050, 4096, QwenSparseScalar::FP16);
        if (transform)
            for (auto *f : {&on, &off})
            {
                f->attention.kv.key_policy = {.encoding = attention::AttentionKeyCacheEncoding::PreRotaryDeviceTransform,
                    .rope_theta = 10000.0f, .partial_rotary_factor = 0.25f};
                f->attention.kv.key_cosine = f->cos.data; f->attention.kv.key_sine = f->sin.data;
                f->attention.kv.key_rotary_dim = 64; f->attention.kv.key_position_capacity = f->positions;
            }
        off.attention.observations = nullptr;
        on.capture(); off.capture();
        // Keep both exact graph identities while changing only published replay
        // data. Zero live rows must leave all arithmetic payload untouched.
        for (auto replay : {QwenSparseReplay{2050, rows, 17}, QwenSparseReplay{256, rows, 17},
                            QwenSparseReplay{255, rows, 17}, QwenSparseReplay{257, 1, 17},
                            QwenSparseReplay{0, 0, 17}, QwenSparseReplay{0, 1, 17}})
        {
            SCOPED_TRACE(::testing::Message() << "prefix=" << replay.prefix_tokens << " live=" << replay.live_rows);
            std::vector<QwenSparseAttentionPartial> dirty(on.partials.size);
            for (auto &p : dirty)
            {
                p.maximum = 123.0f; p.denominator = 456.0f;
                std::fill(std::begin(p.weighted), std::end(p.weighted), -789.0f);
                p.key_elements = p.value_elements = 0xdeadbeefu;
            }
            for (auto *f : {&on, &off})
            {
                f->replay = replay; f->state.upload(&f->replay, f->stream);
                f->partials.upload(dirty.data(), f->stream);
                f->output.upload(f->host_core_query.data(), f->stream);
                f->launch();
            }
            const auto a = on.partials.download(on.stream), b = off.partials.download(off.stream);
            const auto output = on.output.download(on.stream), unobserved = off.output.download(off.stream);
            EXPECT_EQ(std::memcmp(output.data(), unobserved.data(), output.size() * sizeof(float)), 0);
            const auto counts = on.count.download(on.stream);
            const auto observations = on.observations.download(on.stream);
            for (int row = 0; row < rows; ++row)
            for (int head = 0; head < 24; ++head)
            {
                std::uint64_t key_coordinates = 0, value_coordinates = 0;
                for (int part = 0; part < QwenSparsePartitions; ++part)
                {
                    const auto i = (row * 24 + head) * QwenSparsePartitions + part;
                    EXPECT_EQ(std::memcmp(&a[i].maximum, &b[i].maximum, sizeof(float)), 0);
                    EXPECT_EQ(std::memcmp(&a[i].denominator, &b[i].denominator, sizeof(float)), 0);
                    EXPECT_EQ(std::memcmp(a[i].weighted, b[i].weighted, sizeof(a[i].weighted)), 0);
                    if (row >= replay.live_rows)
                    {
                        EXPECT_EQ(std::memcmp(&a[i], &dirty[i], sizeof(a[i])), 0);
                        continue;
                    }
                    // Independent coordinate enumeration models logical decoder
                    // calls, including pre-RoPE partners, rather than the kernel
                    // count formula or any physical codec traffic estimate.
                    std::uint32_t keys = 0, values = 0;
                    for (int token = part * QwenSparsePartitionTokens;
                         token < counts[row] && token < (part + 1) * QwenSparsePartitionTokens; ++token)
                        for (int d = 0; d < 256; ++d)
                        {
                            ++keys; ++values;
                            if (transform && d < 64) ++keys;
                        }
                    EXPECT_EQ(a[i].key_elements, keys); EXPECT_EQ(a[i].value_elements, values);
                    key_coordinates += keys; value_coordinates += values;
                    if (!values)
                    {
                        EXPECT_EQ(a[i].maximum, -INFINITY); EXPECT_EQ(a[i].denominator, 0.0f);
                        EXPECT_EQ(std::memcmp(a[i].weighted, dirty[i].weighted, sizeof(a[i].weighted)), 0);
                    }
                }
                EXPECT_EQ(observations[row * 24 + head].key_elements, key_coordinates);
                EXPECT_EQ(observations[row * 24 + head].value_elements, value_coordinates);
                EXPECT_EQ(observations[row * 24 + head].selected_tokens, row < replay.live_rows ? counts[row] : 0);
            }
        }
    }
}

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
    const auto grouped_partials = f.partials.download(f.stream);
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
        const auto serial_partials = f.partials.download(f.stream);
        for (int head = 0; head < 24; ++head)
        for (int part = 0; part < QwenSparsePartitions; ++part)
        {
            const auto &expected = grouped_partials[(row * 24 + head) * QwenSparsePartitions + part];
            const auto &actual = serial_partials[head * QwenSparsePartitions + part];
            EXPECT_EQ(std::memcmp(&actual.maximum, &expected.maximum, sizeof(float)), 0);
            EXPECT_EQ(std::memcmp(&actual.denominator, &expected.denominator, sizeof(float)), 0);
            if (expected.denominator > 0.0f)
                EXPECT_EQ(std::memcmp(actual.weighted, expected.weighted, sizeof(actual.weighted)), 0);
        }
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
            expectObservationNeutralAttention(f);
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
    expectObservationNeutralAttention(f);
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

/**
 * @brief Test-only exact native-bank upload with retirement before every backing unwind.
 *
 * Logical bytes and metadata come only from the original serialized fixture or
 * its canonical compact slice. No codec conversion, full-bank expansion or
 * physical-device count is inferred from these single-device fixtures.
 */
struct ConsumerNativeUpload final
{
    QSA_STREAM stream;
    QwenSparseNativeBank descriptor;
    Buffer<unsigned char> bytes;
    std::unique_ptr<Buffer<float>> rotation,centroids,anchor;
    FixtureCopyRetirement retirement{&stream};
    /**
     * @brief Upload the exact original bank, retaining source bytes through completion.
     * @param source Original independently packed native fixture.
     * @param execution_stream Exact fixture stream; no upload occurs in capture.
     */
    ConsumerNativeUpload(const test::NativeCodecFixture &source,QSA_STREAM execution_stream)
        :ConsumerNativeUpload(source.storage,source.rotation,source.centroids,source.anchor,source.bank(),execution_stream){}
    /**
     * @brief Upload a consumer's exact compact native bytes and local codec metadata.
     * @param source Canonically assigned compact slice, retained through retirement.
     * @param execution_stream Exact same-device fixture stream.
     */
    ConsumerNativeUpload(const test::ConsumerLocalCodecFixture &source,QSA_STREAM execution_stream)
        :ConsumerNativeUpload(source.storage,source.rotation,source.source.centroids,source.anchor,source.bank(),execution_stream){}
    /** @return Native device view with exact source/compact token and head strides. */
    QwenSparseNativeBank bank() const {return descriptor;}
private:
    /**
     * @brief Materialize only the exact persistent bank/table extents outside capture.
     * @param storage Original serialized native bytes, never requantized.
     * @param rotations Matching original-head rotations for rotated codecs only.
     * @param codes Original immutable TQ codebook, unchanged between full/local banks.
     * @param anchors Corresponding original AQ8 request anchors, if applicable.
     * @param native Complete host descriptor supplying codec/block/stride identity.
     * @param execution_stream Original fixture stream retained by its owner.
     */
    ConsumerNativeUpload(const std::vector<unsigned char> &storage,const std::vector<float> &rotations,
        const std::vector<float> &codes,const std::vector<float> &anchors,QwenSparseNativeBank native,QSA_STREAM execution_stream)
        :stream(execution_stream),descriptor(native),bytes(storage.size())
    {
        if(detail::isRotated(native))
        {rotation=std::make_unique<Buffer<float>>(rotations.size());centroids=std::make_unique<Buffer<float>>(codes.size());}
        if(native.codec==TensorType::AQ8)anchor=std::make_unique<Buffer<float>>(anchors.size());
        bytes.upload(storage.data(),stream);descriptor.data=bytes.data;
        descriptor.rotations[0]=nullptr;descriptor.rotations[1]=nullptr;descriptor.centroids=nullptr;descriptor.anchor=nullptr;
        if(rotation)
        {
            rotation->upload(rotations.data(),stream);centroids->upload(codes.data(),stream);
            descriptor.rotations[0]=rotation->data;
            if(rotations.size()>65536)descriptor.rotations[1]=rotation->data+65536;
            descriptor.centroids=centroids->data;
        }
        if(anchor){anchor->upload(anchors.data(),stream);descriptor.anchor=anchor->data;}
    }
};
/** @brief Retire every native node before any local operand/metadata buffer unwinds. */
struct ConsumerSparseGraphRetirement final
{
    Fixture &fixture;
    /** @brief Join the original stream, then destroy its exact current retained graph. */
    ~ConsumerSparseGraphRetirement()
    {
        if(QSA_RT(StreamSynchronize)(fixture.stream)!=QSA_RT(Success))std::terminate();
        if(fixture.executable)
        {if(QSA_RT(GraphExecDestroy)(fixture.executable)!=QSA_RT(Success))std::terminate();fixture.executable=nullptr;}
        if(fixture.graph)
        {if(QSA_RT(GraphDestroy)(fixture.graph)!=QSA_RT(Success))std::terminate();fixture.graph=nullptr;}
    }
};
/**
 * @brief Capture the original production selected-attention operation without changing its arithmetic.
 * @param f Same-device fixture whose original replay/selection remain device-owned inputs.
 * @note The indexer is not part of this consumer-local KV mapping certificate.
 */
void captureConsumerSparseAttention(Fixture &f)
{
    check(QSA_RT(StreamBeginCapture)(f.stream,QSA_RT(StreamCaptureModeThreadLocal)));
    try{QSA_ATTENTION(f.attention,f.stream);}
    catch(...)
    {
        QSA_GRAPH abandoned=nullptr;(void)QSA_RT(StreamEndCapture)(f.stream,&abandoned);
        if(abandoned)(void)QSA_RT(GraphDestroy)(abandoned);throw;
    }
    check(QSA_RT(StreamEndCapture)(f.stream,&f.graph));
    std::size_t count=0;check(QSA_RT(GraphGetNodes)(f.graph,nullptr,&count));
    EXPECT_EQ(count,detail::isRotated(f.attention.kv.keys)?3u:2u);
    check(QSA_RT(GraphInstantiate)(&f.executable,f.graph,nullptr,nullptr,0));
}

/** @test Every logical degree1..8 consumes only compact native heads with exact full-head replay bytes. */
TEST(QSA_SUITE, CompactConsumerKVEveryDegreeMatchesFullOriginalBytes)
{
    constexpr int rows=3,tokens=8,dimension=256,full_width=24*dimension;
    const auto grain=QueryHeadPartitionDeclaration::fromOriginalConsumers(24,{{16,48}});
    for(const auto codec:{TensorType::FP32,TensorType::FP16,TensorType::BF16,TensorType::Q8_1,
            TensorType::Q16_1,TensorType::AQ8,TensorType::TQ4,TensorType::TQ8})
    for(int block:{32,64,128})
    {
        if(block!=32 && codec!=TensorType::Q16_1)continue;
        SCOPED_TRACE(::testing::Message()<<"codec="<<int(codec)<<" block="<<block);
        Fixture f(rows,5,tokens,QwenSparseScalar::FP32);
        test::NativeCodecFixture source_keys(codec,tokens,block,false),source_values(codec==TensorType::AQ8?TensorType::FP32:codec,tokens,block,false);
        ConsumerNativeUpload full_keys(source_keys,f.stream),full_values(source_values,f.stream);
        std::vector<std::int32_t> selected(rows*QwenSparseSelected,-99),counts{3,4,5};
        for(int row=0;row<rows;++row)for(int i=0;i<counts[row];++i)selected[row*QwenSparseSelected+i]=i;
        ConsumerSparseGraphRetirement full_retirement{f};
        f.selected.upload(selected.data(),f.stream);f.count.upload(counts.data(),f.stream);
        f.attention.kv={.keys=full_keys.bank(),.values=full_values.bank(),.addressing=QwenSparseKVAddressing::FixedLogicalView,
            .logical_view={.logical_row_origin=6,.physical_row_capacity=tokens}};
        check(QSA_RT(StreamSynchronize)(f.stream));captureConsumerSparseAttention(f);f.launch();
        const auto full=f.output.download(f.stream);
        check(QSA_RT(GraphExecDestroy)(f.executable));f.executable=nullptr;
        check(QSA_RT(GraphDestroy)(f.graph));f.graph=nullptr;
        for(int degree=1;degree<=8;++degree)
        {
            const auto config=TensorParallelConfig::equalSplit(degree,24,2,10240,248320,
                std::nullopt,KVHeadCoveragePolicy::QueryInterval,grain);
            for(int rank=0;rank<degree;++rank)
            {
                const auto &a=config.forRank(rank);const int width=a.head_count*dimension;
                SCOPED_TRACE(::testing::Message()<<"degree="<<degree<<" rank="<<rank);
                test::ConsumerLocalCodecFixture compact_keys(source_keys,config,rank),compact_values(source_values,config,rank);
                ConsumerNativeUpload keys(compact_keys,f.stream),values(compact_values,f.stream);
                ASSERT_EQ(keys.bytes.size,std::size_t(tokens*a.kv_head_count)*source_keys.head_bytes);
                ASSERT_EQ(values.bytes.size,std::size_t(tokens*a.kv_head_count)*source_values.head_bytes);
                ASSERT_EQ(keys.bank().token_stride_bytes,std::size_t(a.kv_head_count)*source_keys.head_bytes);
                ASSERT_EQ(values.bank().token_stride_bytes,std::size_t(a.kv_head_count)*source_values.head_bytes);
                std::vector<float> query(rows*width),gate(query.size()),poison(query.size()+16,-923.25f),expected=poison;
                for(int row=0;row<rows;++row)
                {
                    std::copy_n(f.host_core_query.data()+row*full_width+a.head_start*dimension,width,query.data()+row*width);
                    std::copy_n(f.host_gate.data()+row*full_width+a.head_start*dimension,width,gate.data()+row*width);
                }
                Buffer<float> local_query(query.size()),local_gate(gate.size()),output(poison.size());
                Buffer<QwenSparseAttentionPartial> partials(std::size_t(rows*a.head_count)*QwenSparsePartitions);
                Buffer<QwenSparseAttentionObservation> observations(rows*a.head_count);
                std::vector<QwenSparseAttentionObservation> initial(observations.size);
                std::memset(initial.data(),0xa5,initial.size()*sizeof(initial[0]));
                ConsumerSparseGraphRetirement retirement{f};
                local_query.upload(query.data(),f.stream);local_gate.upload(gate.data(),f.stream);output.upload(poison.data(),f.stream);
                observations.upload(initial.data(),f.stream);
                f.attention.query=local_query.data;f.attention.gate=local_gate.data;f.attention.output=output.data;
                f.attention.query_heads=a.head_count;f.attention.query_head_start=a.head_start;
                f.attention.kv_heads=a.kv_head_count;f.attention.kv_head_start=a.kv_head_start;
                f.attention.kv.keys=keys.bank();f.attention.kv.values=values.bank();
                f.attention.gpu_partials=partials.data;f.attention.observations=observations.data;
                check(QSA_RT(StreamSynchronize)(f.stream));captureConsumerSparseAttention(f);
                for(int history=0;history<20;++history)
                {
                    const int live=std::array{1,3,1,0,2}[std::size_t(history)%5];f.replay.live_rows=live;
                    for(int row=0;row<rows;++row)counts[row]=row<live?row+3:-987;
                    f.state.upload(&f.replay,f.stream);f.count.upload(counts.data(),f.stream);
                    for(int row=0;row<live;++row)std::copy_n(full.data()+row*full_width+a.head_start*dimension,width,expected.data()+row*width);
                    f.launch();const auto actual=output.download(f.stream);const auto traffic=observations.download(f.stream);
                    EXPECT_EQ(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),0)<<"history="<<history;
                    // The native reducer publishes zero reads for inactive
                    // diagnostic rows, while preserving their output payload.
                    // Keep that established GPU observation contract; struct
                    // padding is not a published counter field.
                    for(int row=live;row<rows;++row)for(int head=0;head<a.head_count;++head)
                    {
                        const auto &value=traffic[row*a.head_count+head];
                        EXPECT_EQ(value.key_elements,0u);
                        EXPECT_EQ(value.value_elements,0u);
                        EXPECT_EQ(value.selected_tokens,0u);
                    }
                    for(int row=0;row<live;++row)for(int head=0;head<a.head_count;++head)
                    {
                        const auto &value=traffic[row*a.head_count+head];
                        EXPECT_EQ(value.key_elements,std::uint64_t(counts[row])*dimension);
                        EXPECT_EQ(value.value_elements,std::uint64_t(counts[row])*dimension);
                        EXPECT_EQ(value.selected_tokens,counts[row]);
                    }
                }
            }
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
        for(int context:{4102,65536})
        {
            if(context==65536 && representation!=QwenSparseScalar::FP32)continue;
            const int rows=context==65536?1:9,prefix=context-rows;
            Fixture installed(rows,prefix,context,representation);
            BoundedPipelineRetirement installedRetirement{installed.stream};
            Fixture candidate(rows,prefix,context,representation);
            BoundedPipelineRetirement candidateRetirement{candidate.stream};
            candidate.indexer.pool_selection = policy;
            if(context==65536)for(auto* fixture:{&installed,&candidate}){
                fixture->indexer.rotary_source=QwenSparseRotarySource::AnalyticTheta;fixture->indexer.rope_theta=10000000.f;
                fixture->indexer.inverse_frequencies=makeQwenSparsePowerFrequencies(10000000.f,64);
                fixture->indexer.cosine=fixture->indexer.sine=nullptr;
            }
            installed.capture(); candidate.capture();
            for (int live : {rows,1,0,std::min(rows,5),rows})
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

namespace llaminar2::qwen4exp::QSA_SUITE
{
/**
 * @test Retained per-row query preparation preserves native indexer/attention bytes.
 *
 * Both policies record the real pipeline once per representation and M.
 * Query demand is chosen independently per original row by the pool equation;
 * skipped prefixes and inactive suffixes retain their exact changing poison. Native
 * replay changes only original publications and fixture input bytes. Poisoned
 * skipped Q and untouched prepared-Q prefixes/inactive suffixes prove the gate cannot read stale
 * query capacity, while the AllLive graph supplies the exact same-backend oracle.
 * The existing CPU native-cache consumer independently checks active attention
 * values using the captured original selected IDs and original query/KV bytes.
 */
TEST(QSA_SUITE, CapturedRankingOnlyQueryPreparationPreservesNativeResults)
{
    constexpr int tokens=2080,headWidth=256;
    for(const auto selection:{QwenSparsePoolSelection::Installed,QwenSparsePoolSelection::Bounded1024_1024})
    for(const auto representation:{QwenSparseScalar::FP32,QwenSparseScalar::FP16,QwenSparseScalar::BF16})
    for(const int rows:{1,3,16})
    {
        SCOPED_TRACE(::testing::Message()<<"representation="<<int(representation)<<" M="<<rows);
        Fixture ordinary(rows,2052-rows,tokens,representation);
        BoundedPipelineRetirement ordinaryRetirement{ordinary.stream};
        Fixture gated(rows,2052-rows,tokens,representation);
        BoundedPipelineRetirement gatedRetirement{gated.stream};
        ordinary.indexer.query_demand=QwenSparseQueryDemand::AllLive;
        gated.indexer.query_demand=QwenSparseQueryDemand::RankingOnly;
        // One original global query head exercises the unchanged selected-KV
        // consumer without expanding this focused gate case into a head sweep.
        // The fixture's complete two-head native KV encoding remains intact.
        ordinary.attention.query_heads=gated.attention.query_heads=1;
        Fixture serial(1,2050,tokens,representation);
        BoundedPipelineRetirement serialRetirement{serial.stream};
        serial.attention.query_heads=1;
        ordinary.indexer.pool_selection=gated.indexer.pool_selection=serial.indexer.pool_selection=selection;
        ordinary.capture();gated.capture();
        // Inspect runtime-owned captured arguments without reading any device
        // publication. The two demand specializations must be distinct native
        // functions and carry their exact original replay/row-work identity.
        const auto queryFunction=[&](Fixture &f) {
            std::size_t count=0;check(QSA_RT(GraphGetNodes)(f.graph,nullptr,&count));
            std::vector<QSA_RT(GraphNode_t)> nodes(count);
            check(QSA_RT(GraphGetNodes)(f.graph,nodes.data(),&count));
            void *function=nullptr;
            for(auto node:nodes) {
                QSA_RT(GraphNodeType) type;check(QSA_RT(GraphNodeGetType)(node,&type));
                if(type!=QSA_RT(GraphNodeTypeKernel))throw std::runtime_error("Query fixture contains a non-kernel node");
                QSA_RT(KernelNodeParams) native{};check(QSA_RT(GraphKernelNodeGetParams)(node,&native));
                // Grouped rows retain one fused pool/query launch; prefill uses
                // the independent original query program. Both carry the exact
                // original queryWork tuple inspected below.
                const unsigned queryGrid = rows <= attention::kMaxGroupedVerifierAttentionRows
                    ? unsigned((rows+6)/4+rows*4) : unsigned(rows*4);
                if(native.gridDim.x!=queryGrid||native.gridDim.y!=1||native.gridDim.z!=1||native.blockDim.x!=128)continue;
                if(function||native.blockDim.y!=1||native.blockDim.z!=1||native.sharedMemBytes||!native.kernelParams)
                    throw std::runtime_error("Query preparation has ambiguous geometry/arguments");
                const auto &params=*static_cast<const QwenSparseIndexer *>(native.kernelParams[0]);
                const auto &work=*static_cast<const ProjectionRowWork *>(native.kernelParams[1]);
                if(params.replay!=f.indexer.replay||params.row_capacity!=rows||params.token_capacity!=tokens||
                   params.query_demand!=f.indexer.query_demand||params.query!=f.indexer.query||
                   params.prepared_query!=f.indexer.prepared_query||
                   work.captureIdentity()!=qwenSparseIndexerQueryWork(f.indexer).repeatEachRow(4).captureIdentity())
                    throw std::runtime_error("Query preparation lost its original replay/work binding");
                function=native.func;
            }
            if(!function)throw std::runtime_error("Query preparation native function is missing");
            return function;
        };
        EXPECT_NE(queryFunction(ordinary),queryFunction(gated));
        // Capture the original single-row indexer plus accepted-tail commit
        // once. This serial replay is a diagnostic oracle only, and does not
        // add unrelated core-attention arithmetic to the row reference.
        check(QSA_RT(StreamBeginCapture)(serial.stream,QSA_RT(StreamCaptureModeThreadLocal)));
        try {
            QSA_INDEXER(serial.indexer,serial.stream);
            QSA_COMMIT((QwenSparsePartialCommit{.replay=serial.state.data,.prior_partial=serial.partial.data,
                .raw_keys=serial.raw.data,.acceptance=AcceptedRequestPrefixes::fixed(
                    DeviceRequestRowRanges::deviceCounted(1,1,&serial.state.data->live_rows),1),
                .partial_keys=serial.partial.data}),serial.stream);
        } catch(...) {
            QSA_GRAPH abandoned=nullptr;
            (void)QSA_RT(StreamEndCapture)(serial.stream,&abandoned);
            if(abandoned)(void)QSA_RT(GraphDestroy)(abandoned);
            throw;
        }
        check(QSA_RT(StreamEndCapture)(serial.stream,&serial.graph));
        std::size_t serialNodes=0;
        check(QSA_RT(GraphGetNodes)(serial.graph,nullptr,&serialNodes));
        EXPECT_EQ(serialNodes,4u); // Grouped M1 indexer3 plus accepted-tail commit1.
        check(QSA_RT(GraphInstantiate)(&serial.executable,serial.graph,nullptr,nullptr,0));
        const auto ordinaryExecutable=ordinary.executable,gatedExecutable=gated.executable;
        const int partial=std::min(rows,2);
        int iteration=0;
        for(const auto replay:std::array{QwenSparseReplay{2060,rows,17},
                QwenSparseReplay{2051-rows,rows,17},QwenSparseReplay{2060-rows,rows,17},
                QwenSparseReplay{3,std::min(rows,3),17},QwenSparseReplay{2060,0,17},
                QwenSparseReplay{2050,rows,17},QwenSparseReplay{2051,1,17},QwenSparseReplay{0,0,17},
                QwenSparseReplay{2051-partial,partial,17},QwenSparseReplay{2052-partial,partial,17},
                QwenSparseReplay{2060,rows,17}})
        {
            SCOPED_TRACE(::testing::Message()<<"prefix="<<replay.prefix_tokens<<" live="<<replay.live_rows);
            const auto needed=[&](int row){return row<replay.live_rows&&(replay.prefix_tokens+row+1)/4>512;};
            const size_t liveWords=size_t(replay.live_rows)*512;
            const std::uint32_t preparedBits=0x7fc12000u+static_cast<unsigned>(iteration++);
            float preparedSentinel;std::memcpy(&preparedSentinel,&preparedBits,sizeof(preparedSentinel));
            const float outputSentinel=-8123.25f;
            std::vector<float> ordinaryQuery(ordinary.query.size,NAN),gatedQuery(gated.query.size,NAN);
            std::copy_n(ordinary.host_query.begin(),liveWords,ordinaryQuery.begin());
            for(int row=0;row<replay.live_rows;++row)if(needed(row))
                std::copy_n(gated.host_query.begin()+std::size_t(row)*512,512,gatedQuery.begin()+std::size_t(row)*512);
            const auto priorPools=gated.pools.download(gated.stream);
            std::vector<float> preparedPoison(ordinary.prepared.size,preparedSentinel);
            std::vector<float> scorePoison(ordinary.scores.size,-1357.f),outputPoison(ordinary.output.size,outputSentinel);
            std::vector<std::int32_t> selectedPoison(ordinary.selected.size,-77),countPoison(ordinary.count.size,-77);
            // Last-declared joins keep every host upload buffer alive even when
            // an assertion or backend failure exits this replay early.
            BoundedPipelineRetirement ordinaryUploads{ordinary.stream},gatedUploads{gated.stream},serialUploads{serial.stream};
            ordinary.replay=gated.replay=replay;
            ordinary.query.upload(ordinaryQuery.data(),ordinary.stream);
            gated.query.upload(gatedQuery.data(),gated.stream);
            for(auto *f:{&ordinary,&gated}) {
                f->state.upload(&f->replay,f->stream);
                f->prepared.upload(preparedPoison.data(),f->stream);
                f->scores.upload(scorePoison.data(),f->stream);
                f->selected.upload(selectedPoison.data(),f->stream);
                f->count.upload(countPoison.data(),f->stream);
                f->output.upload(outputPoison.data(),f->stream);
                f->launch();
            }
            const auto ordinaryPrepared=ordinary.prepared.download(ordinary.stream);
            const auto gatedPrepared=gated.prepared.download(gated.stream);
            EXPECT_TRUE(std::all_of(ordinaryPrepared.begin(),ordinaryPrepared.begin()+liveWords,
                [](float value){return std::isfinite(value);}));
            for(int row=0;row<rows;++row) {
                const auto offset=std::size_t(row)*512;
                const auto *expected=needed(row)?ordinaryPrepared.data()+offset:preparedPoison.data()+offset;
                EXPECT_EQ(std::memcmp(gatedPrepared.data()+offset,expected,512*sizeof(float)),0)<<"original row="<<row;
            }
            const auto ordinaryPools=ordinary.pools.download(ordinary.stream),gatedPools=gated.pools.download(gated.stream);
            const auto ordinaryScores=ordinary.scores.download(ordinary.stream),gatedScores=gated.scores.download(gated.stream);
            const auto ordinarySelected=ordinary.selected.download(ordinary.stream),gatedSelected=gated.selected.download(gated.stream);
            const auto ordinaryCount=ordinary.count.download(ordinary.stream),gatedCount=gated.count.download(gated.stream);
            EXPECT_EQ(std::memcmp(ordinaryPools.data(),gatedPools.data(),ordinaryPools.size()*sizeof(float)),0);
            for(int block=0;block<gated.blocks;++block)
                if(block<replay.prefix_tokens/4||block>=(replay.prefix_tokens+replay.live_rows)/4)
                    EXPECT_EQ(std::memcmp(gatedPools.data()+std::size_t(block)*128,
                        priorPools.data()+std::size_t(block)*128,128*sizeof(float)),0)<<"untouched pool="<<block;
            const auto originalRaw=gated.raw.download(gated.stream),originalPartial=gated.partial.download(gated.stream);
            const auto originalQueries=gated.query.download(gated.stream);
            EXPECT_EQ(std::memcmp(originalRaw.data(),gated.host_raw.data(),originalRaw.size()*sizeof(float)),0);
            EXPECT_EQ(std::memcmp(originalPartial.data(),gated.host_partial.data(),originalPartial.size()*sizeof(float)),0);
            EXPECT_EQ(std::memcmp(originalQueries.data(),gatedQuery.data(),gatedQuery.size()*sizeof(float)),0);
            EXPECT_EQ(std::memcmp(ordinaryScores.data(),gatedScores.data(),ordinaryScores.size()*sizeof(float)),0);
            EXPECT_EQ(ordinarySelected,gatedSelected);EXPECT_EQ(ordinaryCount,gatedCount);
            for(int row=0;row<rows;++row) {
                const int visible=replay.prefix_tokens+row+1;
                EXPECT_EQ(gatedCount[row],row<replay.live_rows ? std::min(visible/4,512)*4+visible%4 : 0);
            }
            const auto ordinaryOutput=ordinary.output.download(ordinary.stream),gatedOutput=gated.output.download(gated.stream);
            EXPECT_EQ(std::memcmp(ordinaryOutput.data(),gatedOutput.data(),ordinaryOutput.size()*sizeof(float)),0);
            EXPECT_TRUE(std::all_of(gatedOutput.begin()+size_t(replay.live_rows)*headWidth,gatedOutput.end(),
                [&](float value){return value==outputSentinel;}));
            auto oracle=ordinary.attention;
            std::vector<float> expected(ordinary.output.size,outputSentinel);
            oracle.replay=&ordinary.replay;oracle.query=ordinary.host_core_query.data();oracle.gate=ordinary.host_gate.data();
            oracle.kv.keys.data=representation==QwenSparseScalar::FP32 ?
                static_cast<const void *>(ordinary.host_key.data()) : ordinary.key16.data();
            oracle.kv.values.data=representation==QwenSparseScalar::FP32 ?
                static_cast<const void *>(ordinary.host_value.data()) : ordinary.value16.data();
            oracle.selected=ordinarySelected.data();oracle.selected_count=ordinaryCount.data();oracle.output=expected.data();
            oracle.gpu_partials=nullptr;oracle.observations=nullptr;
            ASSERT_NO_THROW(cpuQwenSparseAttention(oracle));
            for(size_t word=0;word<size_t(replay.live_rows)*headWidth;++word)
                ASSERT_NEAR(gatedOutput[word],expected[word],3e-6f)<<"word="<<word;
            const auto originalState=gated.state.download(gated.stream);
            ASSERT_EQ(originalState.size(),1u);
            EXPECT_EQ(std::memcmp(originalState.data(),&replay,sizeof(replay)),0);
            // Diagnostic row replay uses the original AllLive native producer,
            // not a copied arithmetic oracle or a production serial path.
            serial.pools.upload(priorPools.data(),serial.stream);
            serial.partial.upload(ordinary.host_partial.data(),serial.stream);
            for(int row=0;row<replay.live_rows;++row) {
                serial.replay={replay.prefix_tokens+row,1,replay.position_origin};
                serial.state.upload(&serial.replay,serial.stream);
                serial.query.upload(ordinary.host_query.data()+std::size_t(row)*512,serial.stream);
                serial.selected.upload(selectedPoison.data(),serial.stream);
                serial.count.upload(countPoison.data(),serial.stream);
                serial.raw.upload(ordinary.host_raw.data()+std::size_t(row)*128,serial.stream);
                serial.launch();
                const auto onePrepared=serial.prepared.download(serial.stream);
                const auto oneSelected=serial.selected.download(serial.stream);
                const auto oneCount=serial.count.download(serial.stream);
                if(needed(row))EXPECT_EQ(std::memcmp(gatedPrepared.data()+std::size_t(row)*512,onePrepared.data(),512*sizeof(float)),0);
                EXPECT_EQ(gatedCount[row],oneCount[0]);
                EXPECT_EQ(std::memcmp(gatedSelected.data()+std::size_t(row)*QwenSparseSelected,
                    oneSelected.data(),QwenSparseSelected*sizeof(std::int32_t)),0);
            }
            const auto serialPools=serial.pools.download(serial.stream);
            EXPECT_EQ(std::memcmp(gatedPools.data(),serialPools.data(),gatedPools.size()*sizeof(float)),0);
            EXPECT_EQ(ordinary.executable,ordinaryExecutable);EXPECT_EQ(gated.executable,gatedExecutable);
            ASSERT_FALSE(::testing::Test::HasFailure());
        }
    }
}
/** @brief Ordinary completion edges retain the selected SDK's exact graph ABI. */
inline auto noRankingParents(QSA_RT(GraphNode_t) node, QSA_RT(GraphNode_t) *parents, std::size_t *count)
{
#if defined(QSA_CUDA_RUNTIME) && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    return QSA_RT(GraphNodeGetDependencies)(node,parents,nullptr,count);
#else
    return QSA_RT(GraphNodeGetDependencies)(node,parents,count);
#endif
}

/** @brief Prove the two first native nodes are exactly pool preparation then selection. */
void inspectNoRankingIndexer(Fixture &f)
{
    std::size_t count=0;
    check(QSA_RT(GraphGetNodes)(f.graph,nullptr,&count));
    std::vector<QSA_RT(GraphNode_t)> nodes(count);
    check(QSA_RT(GraphGetNodes)(f.graph,nodes.data(),&count));
    ASSERT_EQ(count,4u); // pool + select + unchanged post-RoPE attention producer/reducer
    QSA_RT(GraphNode_t) current=nullptr;
    for (const auto node : nodes) {
        std::size_t parents=0;check(noRankingParents(node,nullptr,&parents));
        if (!parents) { ASSERT_EQ(current,nullptr);current=node; }
    }
    ASSERT_NE(current,nullptr);
    for (int role=0;role<2;++role) {
        QSA_RT(GraphNodeType) type;check(QSA_RT(GraphNodeGetType)(current,&type));
        ASSERT_EQ(type,QSA_RT(GraphNodeTypeKernel));
        QSA_RT(KernelNodeParams) native{};check(QSA_RT(GraphKernelNodeGetParams)(current,&native));
        ASSERT_NE(native.func,nullptr);ASSERT_NE(native.kernelParams,nullptr);
        ASSERT_EQ(native.extra,nullptr);ASSERT_EQ(native.sharedMemBytes,0u);
        EXPECT_EQ(native.gridDim.x,role==0 ? unsigned((f.rows+6)/4) : unsigned(f.rows));
        EXPECT_EQ(native.gridDim.y,1u);EXPECT_EQ(native.gridDim.z,1u);
        EXPECT_EQ(native.blockDim.x,role==0 ? 128u : 256u);
        EXPECT_EQ(native.blockDim.y,1u);EXPECT_EQ(native.blockDim.z,1u);
        QwenSparseIndexer p;
        std::memcpy(&p,native.kernelParams[0],sizeof(p));
        EXPECT_EQ(p.replay,f.indexer.replay);EXPECT_EQ(p.raw_keys,f.indexer.raw_keys);
        EXPECT_EQ(p.partial_keys,f.indexer.partial_keys);EXPECT_EQ(p.pooled_keys,f.indexer.pooled_keys);
        EXPECT_EQ(p.key_norm,f.indexer.key_norm);EXPECT_EQ(p.cosine,f.indexer.cosine);EXPECT_EQ(p.sine,f.indexer.sine);
        EXPECT_EQ(p.selected,f.indexer.selected);EXPECT_EQ(p.selected_count,f.indexer.selected_count);
        EXPECT_EQ(p.row_capacity,f.rows);EXPECT_EQ(p.token_capacity,f.capacity);EXPECT_EQ(p.block_capacity,f.blocks);
        EXPECT_EQ(p.position_capacity,f.positions);EXPECT_EQ(p.rotary_dim,f.indexer.rotary_dim);
        EXPECT_EQ(p.norm_epsilon,f.indexer.norm_epsilon);EXPECT_EQ(p.representation,f.indexer.representation);
        EXPECT_EQ(p.rotary_source,f.indexer.rotary_source);EXPECT_EQ(p.rope_theta,f.indexer.rope_theta);
        EXPECT_EQ(p.inverse_frequencies,f.indexer.inverse_frequencies);EXPECT_EQ(p.transaction_prior,nullptr);
        EXPECT_EQ(p.pool_selection,f.indexer.pool_selection);EXPECT_EQ(p.selection_observations,nullptr);
        EXPECT_FALSE(p.work_counters.bound());
        EXPECT_EQ(p.query_demand,QwenSparseQueryDemand::ProvenNoRanking);
        EXPECT_EQ(p.query,nullptr);EXPECT_EQ(p.query_norm,nullptr);
        EXPECT_EQ(p.prepared_query,nullptr);EXPECT_EQ(p.scores,nullptr);
        const char *raw=nullptr;
#if defined(QSA_ROCM_RUNTIME)
        raw=hipKernelNameRefByPtr(native.func,f.stream);
        const std::string space="llaminar2::qwen4exp::rocm_sparse_detail::";
#else
        check(cudaFuncGetName(&raw,native.func));
        const std::string space="llaminar2::qwen4exp::cuda_sparse_detail::";
#endif
        ASSERT_NE(raw,nullptr);
        int status=-1;char *decoded=abi::__cxa_demangle(raw,nullptr,nullptr,&status);
        const std::string name=decoded ? decoded : "";std::free(decoded);
        ASSERT_EQ(status,0);
        const auto policy=f.indexer.pool_selection;
        const std::string expected=role==0 ? space+"preparePools(llaminar2::qwen4exp::QwenSparseIndexer)" :
            policy==QwenSparsePoolSelection::Installed ? space+"selectPoolsWide(llaminar2::qwen4exp::QwenSparseIndexer)" :
            "void "+space+"selectPoolsBounded<"+
                std::string(policy==QwenSparsePoolSelection::Bounded256_512 ? "256, 512" :
                            policy==QwenSparsePoolSelection::Bounded512_512 ? "512, 512" : "1024, 1024")+
                ">(llaminar2::qwen4exp::QwenSparseIndexer)";
        EXPECT_EQ(name,expected);
        QSA_RT(FuncAttributes) resources{};check(QSA_RT(FuncGetAttributes)(&resources,native.func));
        // CUDA's inherited accurate AnalyticTheta sinf/cosf slow reduction
        // owns 28 explicit local bytes, rounded to a 32-byte linked frame.
        // This is not a register spill; selection and HIP retain zero frames.
#if defined(QSA_CUDA_RUNTIME)
        EXPECT_EQ(resources.localSizeBytes,role==0 ? 32u : 0u);
#else
        EXPECT_EQ(resources.localSizeBytes,0u);
#endif
        if (!role) EXPECT_EQ(resources.sharedSizeBytes,512u);
        // Locate the unique ordinary child; inspecting a name elsewhere in the
        // graph cannot substitute for the actual ordered producer-consumer edge.
        QSA_RT(GraphNode_t) next=nullptr;
        for (const auto node : nodes) {
            std::size_t n=0;check(noRankingParents(node,nullptr,&n));
            if (n==1) {
                QSA_RT(GraphNode_t) parent=nullptr;check(noRankingParents(node,&parent,&n));
                if (parent==current) { ASSERT_EQ(next,nullptr);next=node; }
            }
        }
        ASSERT_NE(next,nullptr);current=next;
    }
}

/** @test Omitted query/score launches preserve complete attention and state on retained dirty graphs. */
TEST(QSA_SUITE, CapturedProvenNoRankingOmitsQueryWorkAndPreservesRequiredBytes)
{
    constexpr int tokens=2080;
    for (int rows=1;rows<=16;++rows)
    for (const auto representation : {QwenSparseScalar::FP32,QwenSparseScalar::FP16,QwenSparseScalar::BF16})
    {
        // Every row capacity is covered; other selector geometries are sampled
        // at the endpoints while the common replay guard is exercised everywhere.
        const std::vector<QwenSparsePoolSelection> policies=rows==1 || rows==16 ?
            std::vector<QwenSparsePoolSelection>{QwenSparsePoolSelection::Installed,QwenSparsePoolSelection::Bounded256_512,
                QwenSparsePoolSelection::Bounded512_512,QwenSparsePoolSelection::Bounded1024_1024} :
            std::vector<QwenSparsePoolSelection>{QwenSparsePoolSelection::Installed};
        for (const auto policy : policies) {
            SCOPED_TRACE(::testing::Message()<<"M="<<rows<<" repr="<<int(representation)<<" selector="<<int(policy));
            Fixture full(rows,3,tokens,representation),omitted(rows,3,tokens,representation);
            full.attention.query_heads=omitted.attention.query_heads=1;
            full.indexer.pool_selection=omitted.indexer.pool_selection=policy;
            omitted.indexer.query_demand=QwenSparseQueryDemand::ProvenNoRanking;
            omitted.indexer.query=nullptr;omitted.indexer.query_norm=nullptr;
            omitted.indexer.prepared_query=nullptr;omitted.indexer.scores=nullptr;
            std::vector<float> queryPoison(omitted.query.size,-333.5f),preparedPoison(omitted.prepared.size,-777.25f),
                scorePoison(omitted.scores.size,-999.5f),outputPoison(omitted.output.size,-3125.5f);
            std::vector<int> selectedPoison(omitted.selected.size,-91),countPoison(rows,-91);
            BoundedPipelineRetirement fullRetirement{full.stream},omittedRetirement{omitted.stream};
            full.capture();omitted.capture();inspectNoRankingIndexer(omitted);
            const auto fullExec=full.executable,omittedExec=omitted.executable;
            for (const auto replay : {QwenSparseReplay{3,rows,17},QwenSparseReplay{2051-rows,rows,17},
                    QwenSparseReplay{2051,0,17},QwenSparseReplay{2,std::min(rows,2),17},QwenSparseReplay{3,rows,17}}) {
                full.replay=omitted.replay=replay;
                for (auto *f : {&full,&omitted}) {
                    f->state.upload(&f->replay,f->stream);
                    f->selected.upload(selectedPoison.data(),f->stream);f->count.upload(countPoison.data(),f->stream);
                    f->output.upload(outputPoison.data(),f->stream);
                }
                omitted.query.upload(queryPoison.data(),omitted.stream);
                omitted.prepared.upload(preparedPoison.data(),omitted.stream);
                omitted.scores.upload(scorePoison.data(),omitted.stream);
                full.launch();omitted.launch();
                EXPECT_EQ(full.selected.download(full.stream),omitted.selected.download(omitted.stream));
                EXPECT_EQ(full.count.download(full.stream),omitted.count.download(omitted.stream));
                const auto fullPools=full.pools.download(full.stream),omittedPools=omitted.pools.download(omitted.stream);
                EXPECT_EQ(std::memcmp(fullPools.data(),omittedPools.data(),fullPools.size()*sizeof(float)),0);
                const auto fullOutput=full.output.download(full.stream),omittedOutput=omitted.output.download(omitted.stream);
                EXPECT_EQ(std::memcmp(fullOutput.data(),omittedOutput.data(),fullOutput.size()*sizeof(float)),0);
                EXPECT_EQ(omitted.query.download(omitted.stream),queryPoison);
                EXPECT_EQ(omitted.prepared.download(omitted.stream),preparedPoison);
                EXPECT_EQ(omitted.scores.download(omitted.stream),scorePoison);
                EXPECT_EQ(omitted.raw.download(omitted.stream),omitted.host_raw);
                EXPECT_EQ(omitted.partial.download(omitted.stream),omitted.host_partial);
                EXPECT_EQ(full.executable,fullExec);EXPECT_EQ(omitted.executable,omittedExec);
                ASSERT_FALSE(::testing::Test::HasFailure());
            }
        }
    }
}
} // namespace llaminar2::qwen4exp::QSA_SUITE
