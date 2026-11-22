/**
 * @file Test__CUDARetainedHCFFN.cpp
 * @brief Replay recorded SM86 QFN local HC/FFN inputs through the actual retained program.
 *
 * The trace stores deterministic recipes for original native operands and the
 * staged serial CUDA output digests. It includes zero local routes on an active
 * row, partially local routes, and compact-to-expanded placement changes with
 * an intentionally stale unpinned bank. Capture precedes every replay.
 */
#include "backends/GPUDeviceProcessLifetime.h"
#include "backends/cuda/CUDAGraphCapture.h"
#include "backends/cuda/CUDAHardwareMetadata.h"
#include "execution/moe/MoERuntimeTable.h"
#include "kernels/cuda/moe/CUDARetainedHCFFN.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "utils/FNV1a.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <unistd.h>
#include <vector>

namespace
{
using namespace llaminar2;
using namespace llaminar2::cuda::moe;
namespace formats = native_quantised_formats;

constexpr int hidden = 2560, width = 320, trunk = 4 * hidden, top_k = 10, experts = 512;
// The trace contains the original serial Down partition boundaries from SM86.
constexpr int hc_partitions = 46;

/** @brief Record-fragment identity and authoring placement for one original input. */
struct TraceFrame
{
    uint32_t seed;
    int local_routes; ///< 0 absent, 1 alternating ownership, 2 all local.
    uint64_t expected;
};

/** @brief Immutable checkpoint source, serial output trace and optional moved case. */
struct TracePair
{
    NativeQuantisedFormatInfo gate, down;
    std::array<TraceFrame, 3> rows;
    uint64_t moved;
};

constexpr std::array traces{
    TracePair{formats::Q4_K, formats::Q5_1, {{{112,1,0x7e53f715e0a94fc3}, {113,2,0x7985324e94a8a58d},
        {114,0,0x77d7e4b12fdfbfa9}}}, 0x913ba845977e16ee},
    TracePair{formats::Q4_K, formats::Q8_0, {{{112,1,0x579c7d3286509cfe}, {113,2,0xa3536a870c550a76},
        {114,0,0x77d7e4b12fdfbfa9}}}, 0},
    TracePair{formats::Q5_K, formats::Q8_0, {{{112,1,0x362bf7d8f0f2612f}, {113,2,0x80abf505f45754e4},
        {114,0,0x77d7e4b12fdfbfa9}}}, 0xafb24f858b6d3e19},
};

/** @brief Throw with the exact CUDA failure so a test cannot continue on stale bytes. */
void requireCUDA(cudaError_t result)
{
    if (result != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(result));
}

/** @brief Per-replay native stream, kept alive until the graph and all buffers retire. */
struct Stream
{
    cudaStream_t value{};
    Stream() { requireCUDA(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { (void)cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

/** @brief Own the native DOT inventory including the conditional child graph. */
struct NativeGraphDump
{
    char path[sizeof("/tmp/llaminar-retained-hc-XXXXXX")] = "/tmp/llaminar-retained-hc-XXXXXX";
    NativeGraphDump()
    {
        const int fd = mkstemp(path);
        if (fd < 0)
            throw std::runtime_error("Cannot create the native conditional inventory artifact");
        close(fd);
    }
    ~NativeGraphDump() { std::remove(path); }
    NativeGraphDump(const NativeGraphDump&) = delete;
    NativeGraphDump& operator=(const NativeGraphDump&) = delete;
};

/** @brief Typed fixture storage. Every transfer occurs outside captured execution. */
template<class T>
class Array
{
public:
    explicit Array(std::size_t count) : count_(count) { requireCUDA(cudaMalloc(&data_, count * sizeof(T))); }
    ~Array() { (void)cudaFree(data_); }
    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    T* data() const { return data_; }

    /** @brief Materialize the exact trace bytes before the next replay. */
    template<class Recipe>
    void set(cudaStream_t stream, Recipe recipe)
    {
        std::vector<T> values(count_);
        for (std::size_t i = 0; i < count_; ++i)
            values[i] = recipe(i);
        requireCUDA(cudaMemcpyAsync(data_, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice, stream));
        requireCUDA(cudaStreamSynchronize(stream));
    }

    /** @brief Read this publication after native replay and extend the recorded digest. */
    uint64_t digest(cudaStream_t stream, uint64_t seed = kFNV1a64OffsetBasis) const
    {
        std::vector<T> values(count_);
        requireCUDA(cudaMemcpyAsync(values.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
        requireCUDA(cudaStreamSynchronize(stream));
        return fnv1a64(values.data(), values.size() * sizeof(T), seed);
    }

private:
    T* data_ = nullptr;
    const std::size_t count_;
};

/** @brief Fixed bit recipe used by the serial capture at each recorded seed. */
uint32_t bits(std::size_t index, uint32_t seed)
{
    uint32_t value = static_cast<uint32_t>(index) * 0x9e3779b9u + seed;
    value ^= value >> 16u;
    value *= 0x85ebca6bu;
    return value ^ (value >> 13u);
}

/** @brief Decode one exact binary32 input from the capture recipe. */
float signedValue(std::size_t index, uint32_t seed)
{
    return (static_cast<int>(bits(index, seed) % 2047u) - 1023) * 0.0001220703125f;
}

/** @brief Original packed native planes, including the recorded lossless Q5 migration representation. */
struct Weight
{
    Array<uint8_t> bytes;
    Array<uint16_t> scales, mins;
    DeviceNativeQuantisedMatrixDesc view;

    /** @brief Reconstruct one serial checkpoint operand and its original format identity. */
    Weight(cudaStream_t stream, int n, int k, NativeQuantisedFormatInfo source, uint32_t seed,
        bool moved, bool hc = false)
        : bytes(std::size_t(n) * k / 32 * (moved ? 32u : source.payload_bytes)),
          scales(std::size_t(n) * k / 32), mins(source.is_asymmetric ? std::size_t(n) * k / 32 : 1)
    {
        bytes.set(stream, [&](std::size_t i) {
            if (hc)
                return static_cast<uint8_t>(seed == 73 ? int(bits(i, seed) % 91) - 45 : int(bits(i, seed) % 83) - 41);
            if (!moved)
                return static_cast<uint8_t>(bits(i, seed));
            const auto offset = i / 32 * 20, lane = i % 32;
            const auto low = uint8_t(bits(offset + lane % 16, seed));
            const auto high = uint8_t(bits(offset + 16 + lane / 8, seed));
            return uint8_t(((low >> (lane / 16 * 4)) & 15) | (((high >> (lane % 8)) & 1) << 4));
        });
        const auto scale_seed = hc ? (seed == 73 ? 17u : 39u) : seed + 1;
        scales.set(stream, [&](std::size_t i) { return uint16_t(0x1e00 + (bits(i, scale_seed) & 0x1ff)); });
        mins.set(stream, [&](std::size_t i) { return uint16_t(0x1e00 + (bits(i, seed + 2) & 0x1ff)); });
        view = {.payload = bytes.data(), .scales = scales.data(),
            .mins = source.is_asymmetric ? mins.data() : nullptr, .n = n, .k = k,
            .blocks_per_row = unsigned(k / 32),
            .codebook_id = moved ? kNativeQuantisedExpandedInt8MinCodebook : source.codebook_id,
            .allocation_payload_bytes_per_block = uint8_t(moved ? 32 : source.payload_bytes),
            .allocation_has_mins = uint8_t(source.is_asymmetric),
            .source_codebook_id = source.codebook_id, .source_is_superblock = uint8_t(source.is_superblock),
            .source_identity_present = 1};
    }
};

/** @brief Run exactly one original retained program with a complete, counted positive branch. */
void replay(const TracePair& trace, int ordinal, unsigned resident_blocks)
{
    const Stream stream;
    const auto native = stream.value;
    std::vector<std::unique_ptr<Weight>> weights;
    const auto weight = [&](int n, int k, NativeQuantisedFormatInfo format,
        uint32_t seed, bool moved = false, bool hc = false) {
        weights.push_back(std::make_unique<Weight>(native, n, k, format, seed, moved, hc));
        return weights.back()->view;
    };
    const auto hc_down = weight(width, trunk, formats::Q8_0, 73, false, true);
    const auto hc_up = weight(trunk, width, formats::Q8_0, 153, false, true);
    const RetainedM1Expert shared{weight(width, hidden, formats::Q8_0, 61),
        weight(width, hidden, formats::Q8_0, 62), weight(hidden, width, formats::Q8_0, 63)};
    std::array<RetainedM1Expert, top_k * 2> compact{}, moved{};
    for (int expert = 0; expert < int(compact.size()); ++expert)
    {
        const uint32_t seed = 200 + 3 * expert;
        compact[expert] = {weight(width, hidden, trace.gate, seed), weight(width, hidden, trace.gate, seed + 1),
            weight(hidden, width, trace.down, seed + 2)};
        moved[expert] = compact[expert];
        if (trace.gate.codebook_id == formats::Q5_K.codebook_id)
        {
            moved[expert].gate = weight(width, hidden, trace.gate, seed, true);
            moved[expert].up = weight(width, hidden, trace.gate, seed + 1, true);
        }
        if (trace.down.codebook_id == formats::Q5_1.codebook_id)
            moved[expert].down = weight(hidden, width, trace.down, seed + 2, true);
    }

    Array<int> live{1}, order{top_k + 2}, ledger_ids{top_k}, ledger_participants{top_k};
    Array<float> residual{trunk}, block{hidden}, write_gate{4}, gain{trunk}, written{trunk},
        normalized{trunk}, norm_scales{trunk / 32}, partials{width * hc_partitions}, raw{width},
        activated{width}, primary_scales{width / 32}, secondary_scales{width / 32}, up{trunk}, mixed{hidden},
        logits{experts}, route_indices{top_k}, route_weights{top_k}, ledger_weights{top_k},
        routed{top_k * hidden}, shared_out{hidden}, hidden_scales{hidden / 32},
        routed_scales{top_k * width / 32}, shared_scales{width / 32};
    Array<int8_t> norm_codes{trunk}, primary{width}, secondary{width}, hidden_codes{hidden},
        routed_codes{top_k * width}, shared_codes{width};
    Array<uint16_t> router{experts * hidden};
    Array<DeviceMoELayerRuntime> runtime{1};
    Array<DeviceMoEOverlayEpochTicket> ticket{1};
    Array<RetainedM1Expert> selected{top_k};
    const auto rows = DeviceRowRange::deviceCounted(1, live.data());
    const RetainedM1HCFFN args{
        .hc = {
            .write = {{1, live.data()}, hidden, residual.data(), block.data(), write_gate.data(), written.data()},
            .norm = {{1, live.data()}, hidden, 1.e-6f, written.data(), gain.data(), normalized.data()},
            .down = hc_down, .down_partitions = {hc_partitions, (trunk / 32 + hc_partitions - 1) / hc_partitions},
            .down_fold = HCDownFold::OrderedPartitions,
            .norm_codes = norm_codes.data(), .norm_scales = norm_scales.data(), .down_partials = partials.data(),
            .raw_down = raw.data(),
            .down_publication = {rows, width, activated.data(), primary.data(), secondary.data(),
                primary_scales.data(), secondary_scales.data()},
            .up = {hc_up, rows, Q8ResidualQuantizerPolicy::CUDAFloatScaleNativeReciprocal, raw.data(), up.data(),
                primary.data(), secondary.data(), primary_scales.data(), secondary_scales.data()}, .mixed = mixed.data(),
        },
        .router = {runtime.data(), router.data(), logits.data(), route_indices.data(), route_weights.data(),
            nullptr, experts, top_k, RoutedExpertRowExecutionPolicy::ParticipantAssigned, true},
        .shared = shared, .selected = selected.data(), .route_order = order.data(), .routed_width = width,
        .routed_workspace = {routed_codes.data(), routed_scales.data()},
        .shared_workspace = {shared_codes.data(), shared_scales.data()},
        .hidden_codes = hidden_codes.data(), .hidden_scales = hidden_scales.data(),
        .routed_contributions = routed.data(), .shared_contribution = shared_out.data(),
        .route_publication = RetainedM1RoutePublication::Raw,
        .shared_publication = RetainedM1SharedPublication::PartialBeforeCollective,
    };
    constexpr uint64_t generation = 1;
    const auto program = RetainedM1Program::prepare(DeviceId::cuda(ordinal), args, trace.gate.codebook_id,
        trace.down.codebook_id, FullNativeHCFFNOutputDemand::all(), false, generation);
    ASSERT_EQ(program.admission().resources().private_bytes, 0u);

    llaminar2::CUDAGraphCapture capture(native, ordinal);
    ASSERT_TRUE(capture.beginCapture());
    program.launch(native, generation);
    ASSERT_TRUE(capture.endCapture());
    ASSERT_TRUE(capture.instantiate());
    std::vector<GPUGraphKernelNodeInfo> kernels;
    std::string error;
    ASSERT_TRUE(capture.inspectKernelNodes(kernels, &error)) << error;
    ASSERT_EQ(kernels.size(), 1u);
    EXPECT_EQ(kernels.front().grid_x, 1u);
    EXPECT_EQ(kernels.front().block_x, 32u);
    EXPECT_EQ(capture.nodeCount(), 2u);
    // CUDA's public kernel-node queries stop at the conditional node. Its
    // driver DOT inventory includes the body and its actual submitted extent.
    const NativeGraphDump dump;
    requireCUDA(cudaGraphDebugDotPrint(capture.graph(), dump.path, cudaGraphDebugDotFlagsKernelNodeParams));
    std::ifstream file(dump.path);
    const std::string graph{std::istreambuf_iterator<char>{file}, {}};
    const auto resident = graph.find("retainedQFNM1HCFFNKernel");
    ASSERT_NE(resident, std::string::npos);
    EXPECT_NE(graph.find(std::to_string(resident_blocks) + ",256,0", resident), std::string::npos);

    const auto publish = [&](TraceFrame frame, uint32_t epoch, bool physical_moved) {
        residual.set(native, [&](std::size_t i) { return signedValue(i, frame.seed); });
        block.set(native, [&](std::size_t i) { return signedValue(i, frame.seed + 1); });
        write_gate.set(native, [&](std::size_t i) { return signedValue(i, frame.seed + 2); });
        gain.set(native, [&](std::size_t i) { return 0.95f + signedValue(i, frame.seed + 3); });
        router.set(native, [&](std::size_t i) {
            return __half_as_ushort(__float2half(frame.seed == 112 ? 0.f : signedValue(i, frame.seed + 661)));
        });
        live.set(native, [](std::size_t) { return 1; });
        auto state = std::make_unique<DeviceMoELayerRuntime>();
        state->active_epoch = epoch;
        state->expert_count = experts;
        state->top_k = top_k;
        state->route_expert_ids = ledger_ids.data();
        state->route_participant_ids = ledger_participants.data();
        state->route_weights = ledger_weights.data();
        state->prefill_token_capacity = 1;
        state->prefill_route_capacity = top_k;
        state->participant_id = 0;
        state->participant_count = 2;
        const unsigned bank_id = trace.moved ? epoch % 2 : 0;
        if (trace.moved)
        {
            state->active_bank = bank_id ^ 1;
            state->overlay_epoch_ticket = ticket.data();
            ticket.set(native, [&](std::size_t) {
                return DeviceMoEOverlayEpochTicket{.epoch = epoch, .selector = (uint64_t(epoch) << 1) | bank_id};
            });
        }
        auto& bank = state->banks[bank_id];
        bank.epoch = epoch;
        bank.expert_count = experts;
        for (int expert = 0; expert < experts; ++expert)
        {
            const bool local = frame.local_routes == 2 || (frame.local_routes == 1 && expert % 2 == 0);
            auto& entry = bank.experts[expert];
            entry.owner_participant = local ? 0 : 1;
            entry.logical_expert_id = expert;
            entry.local_slot = local ? expert : -1;
            bank.local_compute_mask[expert] = local;
            bank.replica_role[expert] = uint8_t(local ? DeviceMoEReplicaRole::Primary : DeviceMoEReplicaRole::None);
            bank.resident_participant_mask[expert] = local ? 1u : 0u;
            bank.overlay_route_participant[expert] = entry.owner_participant;
            if (!local) continue;
            const auto& original = (physical_moved && expert % 3 != 0 ? moved : compact)[expert % compact.size()];
            entry.gate = original.gate;
            entry.up = original.up;
            entry.down = original.down;
            entry.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid | DeviceMoEExpertFlags::Resident |
                DeviceMoEExpertFlags::PreferredOwner | DeviceMoEExpertFlags::LocalCompute);
        }
        if (!trace.moved)
            state->banks[1] = bank;
        runtime.set(native, [&](std::size_t) { return *state; });
        ASSERT_TRUE(capture.launch());
        uint64_t actual = kFNV1a64OffsetBasis;
        for (const auto* publication : {&written, &normalized, &raw, &activated, &up, &mixed,
                &logits, &route_indices, &route_weights, &routed, &shared_out})
            actual = publication->digest(native, actual);
        EXPECT_EQ(actual, frame.expected) << "seed=" << frame.seed << " epoch=" << epoch;
    };

    for (std::size_t i = 0; i < trace.rows.size(); ++i)
        publish(trace.rows[i], i + 2, false);
    if (trace.moved)
    {
        const TraceFrame movement{250, 1, trace.moved};
        publish(movement, 20, false);
        publish(movement, 21, true);
    }

    // A zero accepted extent clears stale route metadata, but owes no math or
    // stores to any activation/contribution. The following replay starts clean.
    live.set(native, [](std::size_t) { return 0; });
    const auto previous_routed = routed.digest(native);
    const auto previous_shared = shared_out.digest(native);
    const auto previous_hidden = mixed.digest(native);
    ASSERT_TRUE(capture.launch());
    EXPECT_EQ(routed.digest(native), previous_routed);
    EXPECT_EQ(shared_out.digest(native), previous_shared);
    EXPECT_EQ(mixed.digest(native), previous_hidden);
    const std::array<float, top_k> invalid_routes{-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f};
    EXPECT_EQ(route_indices.digest(native), fnv1a64(invalid_routes.data(), sizeof(invalid_routes)));
    publish(trace.rows.front(), 2, false);
}
}

TEST(CUDARetainedHCFFN, RecordedCountedPublications)
{
    llaminar2::GPUDeviceProcessLifetime lifetime;
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    ASSERT_GT(count, 0);
    llaminar2::initializeCUDAHardwareMetadata();
    int ordinal = 0;
    for (; ordinal < count; ++ordinal)
    {
        const auto& device = llaminar2::cudaHardwareMetadata().at(ordinal);
        if (device.sm_major == 8 && device.sm_minor == 6)
            break;
    }
    ASSERT_LT(ordinal, count) << "preflight requires a visible original SM86 device";
    ASSERT_EQ(cudaSetDevice(ordinal), cudaSuccess);
    const auto& device = llaminar2::cudaHardwareMetadata().at(ordinal);
    const auto blocks = std::min(2u * static_cast<unsigned>(device.sm_count), hidden / retained_hc_read_warp_channels);
    for (const auto& trace : traces)
    {
        SCOPED_TRACE(unsigned(trace.gate.codebook_id));
        SCOPED_TRACE(unsigned(trace.down.codebook_id));
        ASSERT_NO_THROW(replay(trace, ordinal, blocks));
    }
}
