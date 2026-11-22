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
#include "kernels/cuda/moe/CUDARetainedGroupedFFN.h"
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

/** @return The visible original SM86 ordinal selected by the native preflight contract. */
int selectVisibleSM86()
{
    int count = 0;
    requireCUDA(cudaGetDeviceCount(&count));
    initializeCUDAHardwareMetadata();
    for (int ordinal = 0; ordinal < count; ++ordinal)
    {
        const auto& device = cudaHardwareMetadata().at(ordinal);
        if (device.sm_major == 8 && device.sm_minor == 6)
        {
            requireCUDA(cudaSetDevice(ordinal));
            return ordinal;
        }
    }
    throw std::runtime_error("Preflight requires a visible original SM86 device");
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

/** @brief Verify exact native edges from the router's acquire through the two packet readers. */
void assertEarlyPacketDependencies(cudaGraph_t graph, const void* hidden, const void* route, const void* retired)
{
    std::size_t count = 0;
    requireCUDA(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    requireCUDA(cudaGraphGetNodes(graph, nodes.data(), &count));
    cudaGraphNode_t resident = nullptr, hidden_packet = nullptr, route_packet = nullptr, after_join = nullptr;
    for (const auto node : nodes)
    {
        cudaGraphNodeType type{};
        requireCUDA(cudaGraphNodeGetType(node, &type));
        if (type == cudaGraphNodeTypeKernel)
        {
            cudaKernelNodeParams args{};
            requireCUDA(cudaGraphKernelNodeGetParams(node, &args));
            if (args.blockDim.x == 256)
                resident = node;
        }
        if (type != cudaGraphNodeTypeMemcpy)
            continue;
        cudaMemcpy3DParms args{};
        requireCUDA(cudaGraphMemcpyNodeGetParams(node, &args));
        if (args.dstPtr.ptr == hidden) hidden_packet = node;
        if (args.dstPtr.ptr == route) route_packet = node;
        if (args.dstPtr.ptr == retired) after_join = node;
    }
    ASSERT_TRUE(resident && hidden_packet && route_packet && after_join);
    const auto dependencies = [](cudaGraphNode_t node) {
        std::size_t size = 0;
        requireCUDA(cudaGraphNodeGetDependencies(node, nullptr, nullptr, &size));
        std::vector<cudaGraphNode_t> dependencies(size);
        requireCUDA(cudaGraphNodeGetDependencies(node, dependencies.data(), nullptr, &size));
        return dependencies;
    };
    const auto first = dependencies(hidden_packet);
    ASSERT_EQ(first.size(), 1u);
    cudaGraphNodeType acquire_type{};
    requireCUDA(cudaGraphNodeGetType(first.front(), &acquire_type));
    EXPECT_EQ(acquire_type, cudaGraphNodeTypeKernel);
    EXPECT_NE(first.front(), resident);
    EXPECT_EQ(dependencies(first.front()), dependencies(resident));
    EXPECT_EQ(dependencies(route_packet), std::vector<cudaGraphNode_t>{hidden_packet});
    const auto joined = dependencies(after_join);
    ASSERT_EQ(joined.size(), 2u);
    EXPECT_NE(std::find(joined.begin(), joined.end(), resident), joined.end());
    EXPECT_NE(std::find(joined.begin(), joined.end(), route_packet), joined.end());
}

/** @brief Select the original invocation contract under test. */
enum class RetainedReplayRole { Counted, ActiveEarlyPacket };

/** @brief Run the retained positive and empty branches through the original serial trace. */
void replay(const TracePair& trace, int ordinal, unsigned resident_blocks, RetainedReplayRole role)
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
    Array<std::uint64_t> signal{1};
    Array<float> packet_hidden{hidden}, packet_route{top_k}, joined_routed{top_k * hidden};
    const bool early = role == RetainedReplayRole::ActiveEarlyPacket;
    const auto* active = early ? nullptr : live.data();
    const auto rows = early ? DeviceRowRange::fullyActive(1) : DeviceRowRange::deviceCounted(1, live.data());
    const RetainedM1HCFFN args{
        .hc = {
            .write = {{1, active}, hidden, residual.data(), block.data(), write_gate.data(), written.data()},
            .norm = {{1, active}, hidden, 1.e-6f, written.data(), gain.data(), normalized.data()},
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
        .early_packet_ready = early ? signal.data() : nullptr,
    };
    constexpr uint64_t generation = 1;
    if (!early)
    {
        auto incomplete = args;
        incomplete.early_packet_ready = signal.data();
        EXPECT_THROW(RetainedM1Program::prepare(DeviceId::cuda(ordinal), incomplete, trace.gate.codebook_id,
            trace.down.codebook_id, FullNativeHCFFNOutputDemand::all(), false, generation), std::invalid_argument);
    }
    const auto program = RetainedM1Program::prepare(DeviceId::cuda(ordinal), args, trace.gate.codebook_id,
        trace.down.codebook_id, FullNativeHCFFNOutputDemand::all(), false, generation);
    ASSERT_EQ(program.admission().resources().private_bytes, 0u);

    llaminar2::CUDAGraphCapture capture(native, ordinal);
    CUDAEarlyPacketPublication packet;
    ASSERT_TRUE(capture.beginCapture());
    if (early)
        packet.recordProducer(native, signal.data());
    program.launch(native, generation);
    if (early)
    {
        ASSERT_TRUE(packet.recordConsumer(native, [&] {
            return cudaMemcpyAsync(packet_hidden.data(), mixed.data(), hidden * sizeof(float),
                cudaMemcpyDeviceToDevice, native) == cudaSuccess;
        }));
        ASSERT_TRUE(packet.recordConsumer(native, [&] {
            return cudaMemcpyAsync(packet_route.data(), route_indices.data(), top_k * sizeof(float),
                cudaMemcpyDeviceToDevice, native) == cudaSuccess;
        }));
        requireCUDA(cudaMemcpyAsync(joined_routed.data(), routed.data(), top_k * hidden * sizeof(float),
            cudaMemcpyDeviceToDevice, native));
    }
    ASSERT_TRUE(capture.endCapture());
    ASSERT_TRUE(capture.instantiate());
    std::vector<GPUGraphKernelNodeInfo> kernels;
    std::string error;
    ASSERT_TRUE(capture.inspectKernelNodes(kernels, &error)) << error;
    if (early)
    {
        ASSERT_EQ(kernels.size(), 2u);
        assertEarlyPacketDependencies(capture.graph(), packet_hidden.data(), packet_route.data(), joined_routed.data());
        resident_blocks = std::min(resident_blocks, program.admission().maximumGrid() - 1u);
    }
    else
    {
        ASSERT_EQ(kernels.size(), 1u);
        EXPECT_EQ(kernels.front().grid_x, 1u);
        EXPECT_EQ(kernels.front().block_x, 32u);
        EXPECT_EQ(capture.nodeCount(), 2u);
    }
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
        if (early)
        {
            EXPECT_EQ(packet_hidden.digest(native), mixed.digest(native));
            EXPECT_EQ(packet_route.digest(native), route_indices.digest(native));
            EXPECT_EQ(joined_routed.digest(native), routed.digest(native));
        }
    };

    for (std::size_t i = 0; i < trace.rows.size(); ++i)
        publish(trace.rows[i], i + 2, false);
    if (trace.moved)
    {
        const TraceFrame movement{250, 1, trace.moved};
        publish(movement, 20, false);
        publish(movement, 21, true);
    }

    if (early)
        return;

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
    const int ordinal = selectVisibleSM86();
    const auto& device = cudaHardwareMetadata().at(ordinal);
    const auto blocks = std::min(2u * static_cast<unsigned>(device.sm_count), hidden / retained_hc_read_warp_channels);
    for (const auto& trace : traces)
    {
        SCOPED_TRACE(unsigned(trace.gate.codebook_id));
        SCOPED_TRACE(unsigned(trace.down.codebook_id));
        ASSERT_NO_THROW(replay(trace, ordinal, blocks, RetainedReplayRole::Counted));
    }
}

/** @test Capture the complete fully-active main model row with original sibling packet readers. */
TEST(CUDARetainedHCFFN, RecordedEarlyPacketPublications)
{
    const int ordinal = selectVisibleSM86();
    const auto& device = cudaHardwareMetadata().at(ordinal);
    const auto blocks = std::min(2u * static_cast<unsigned>(device.sm_count), hidden / retained_hc_read_warp_channels);
    for (const auto& trace : traces)
        ASSERT_NO_THROW(replay(trace, ordinal, blocks, RetainedReplayRole::ActiveEarlyPacket));
}

#if CUDART_VERSION >= 12080
#include "CUDARetainedGroupedFFNReplay.inc"

const void* cudaDeviceSizedProducerWitness();
const void* cudaDeviceSizedConsumerWitness();
bool cudaDeviceSizedPlannerWitness(const int* count, cudaGraphDeviceNode_t producer,
    cudaGraphDeviceNode_t consumer, int* result, cudaStream_t stream);

/** @test Actual native live grids and publication are preserved through an in-place timeline. */
TEST(CUDARetainedHCFFN, DeviceSizedWorkContract)
{
    const int ordinal = selectVisibleSM86();

    constexpr std::size_t capacity = 8;
    const Stream stream;
    Array<int> count_owner{1}, producer{capacity}, output{capacity}, plan_status{1};
    CUDAGraphCapture parent(stream.value, ordinal);
    auto fragment = parent.createOrderedTimelineFragment();
    ASSERT_TRUE(fragment);
    ASSERT_TRUE(fragment->beginCapture());
    auto* producer_address = producer.data();
    auto* output_address = output.data();
    void* produce_args[]{&producer_address};
    void* consume_args[]{&producer_address, &output_address};
    const std::array launches{
        cudaKernelNodeParams{.func = const_cast<void*>(cudaDeviceSizedProducerWitness()),
            .gridDim = dim3(1), .blockDim = dim3(1), .kernelParams = produce_args},
        cudaKernelNodeParams{.func = const_cast<void*>(cudaDeviceSizedConsumerWitness()),
            .gridDim = dim3(1), .blockDim = dim3(1), .kernelParams = consume_args},
    };
    {
        ScopedCUDAFinalTimelineRecording scope(*fragment);
        ASSERT_NO_THROW(appendCUDADeviceSizedKernels(stream.value, launches, [&](auto handles) {
            return cudaDeviceSizedPlannerWitness(count_owner.data(), handles[0], handles[1],
                plan_status.data(), stream.value);
        }));
    }
    ASSERT_TRUE(fragment->endCapture());
    const std::array steps{GPUOrderedTimelineStep{.name = "live_grid", .capture = fragment.get()}};
    ASSERT_TRUE(parent.buildOrderedTimelineTransaction(steps));
    ASSERT_TRUE(parent.instantiate());

    for (const int live : {8, 3, 0, 5, 1, 0, 8})
    {
        SCOPED_TRACE(live);
        count_owner.set(stream.value, [live](std::size_t) { return live; });
        output.set(stream.value, [](std::size_t) { return 0; });
        ASSERT_TRUE(parent.launch());
        constexpr int success = cudaSuccess;
        EXPECT_EQ(plan_status.digest(stream.value), fnv1a64(&success, sizeof(success)));
        std::array<int, capacity> expected{};
        std::fill_n(expected.begin(), live, live * live);
        EXPECT_EQ(output.digest(stream.value), fnv1a64(expected.data(), sizeof(expected)));
    }
}
#endif

/** @brief Own one CUDA process lifetime across all selected native fixtures. */
int main(int argc, char** argv)
{
    GPUDeviceProcessLifetime lifetime;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
