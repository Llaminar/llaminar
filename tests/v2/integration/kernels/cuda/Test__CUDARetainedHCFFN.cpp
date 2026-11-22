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
#include "kernels/cuda/moe/CUDAMoEOverlayActivationPacketKernels.h"
#include "kernels/cuda/moe/CUDARetainedHCFFN.h"
#include "kernels/cuda/moe/CUDARetainedGroupedFFN.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "utils/FNV1a.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <thread>
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
enum class RetainedReplayRole { Counted, ActiveEarlyPacket, ActiveShardEarlyPacket };

/** @brief Separate mapped packet and timeline storage for each original target. */
struct DispatchLane
{
    std::array<int, 1> rows{};
    std::array<int, 2> offsets{};
    std::array<int, top_k> expert_ids{}, original_slots{}, compact_slots{};
    std::array<float, top_k> weights{};
    std::array<float, hidden> payload{};
    MoEOverlayActivationEpochControl epoch{};
};

/** @brief Own the independent mapped CPU packet lanes for the launch contract. */
class MappedDispatchLanes
{
public:
    MappedDispatchLanes(DeviceId owner, unsigned lanes)
        : region(TransferEngine::instance().allocateMappedHostRegion(lanes * sizeof(DispatchLane), std::array{owner})),
          host(static_cast<DispatchLane*>(region->mutableHostData())),
          device(static_cast<DispatchLane*>(region->deviceAlias(owner)))
    {
        std::uninitialized_value_construct_n(host, lanes);
    }
    MappedDispatchLanes(const MappedDispatchLanes&) = delete;
    MappedDispatchLanes& operator=(const MappedDispatchLanes&) = delete;

    const std::shared_ptr<MappedHostTransferRegion> region;
    DispatchLane* const host;
    DispatchLane* const device;
};

/** @brief Decode the pointer-free original F640/F320 fragment record carried by the captured runtime. */
DeviceMoEIntermediateShardBinding originalFragmentTrace()
{
    struct BindingRecord
    {
        MoEOverlayJointRouteTarget target;
        uint32_t hidden_columns, source_columns, first_column, local_columns;
    };
    static_assert(sizeof(BindingRecord) == sizeof(DeviceMoEIntermediateShardBinding));
    const BindingRecord record{MoEOverlayJointRouteTarget::fromOriginalCohort(experts, 0, 0, 0),
        hidden, 2 * width, 0, width};
    // Only the replay decoder reconstructs a device ABI snapshot. Serving
    // admission still derives the binding from its authenticated source family.
    return std::bit_cast<DeviceMoEIntermediateShardBinding>(record);
}

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

    Array<int> live{1}, order{top_k + 2}, route_plan_ready{1}, ledger_ids{top_k}, ledger_participants{top_k};
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
    const bool early = role != RetainedReplayRole::Counted;
    const bool shard = role == RetainedReplayRole::ActiveShardEarlyPacket;
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
            nullptr, experts, top_k, shard ? RoutedExpertRowExecutionPolicy::IntermediateShardLocal
                : RoutedExpertRowExecutionPolicy::ParticipantAssigned, true},
        .shared = shared, .selected = selected.data(), .route_order = order.data(),
        .route_plan_ready = route_plan_ready.data(), .routed_width = width,
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
    if (early)
        packet.prepare(native, signal.data());
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
        if (shard)
        {
            // The retained kernel must initialize its own publication before
            // any worker can consume the next route plan.
            route_plan_ready.set(native, [](std::size_t) { return 7; });
        }
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
        if (shard)
            state->intermediate_shard = originalFragmentTrace();
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
            entry.owner_participant = shard ? -1 : local ? 0 : 1;
            entry.logical_expert_id = expert;
            entry.local_slot = local ? expert : -1;
            bank.local_compute_mask[expert] = local;
            bank.replica_role[expert] = uint8_t(local && !shard ? DeviceMoEReplicaRole::Primary : DeviceMoEReplicaRole::None);
            bank.resident_participant_mask[expert] = local && !shard ? 1u : 0u;
            bank.overlay_route_participant[expert] = shard ? MoEOverlayRouteTarget::cohort(local ? 0 : 1).word()
                                                          : entry.owner_participant;
            if (!local) continue;
            const auto& original = (physical_moved && expert % 3 != 0 ? moved : compact)[expert % compact.size()];
            entry.gate = original.gate;
            entry.up = original.up;
            entry.down = original.down;
            entry.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid | DeviceMoEExpertFlags::Resident |
                DeviceMoEExpertFlags::LocalCompute);
            if (shard)
                entry.projection_set = DeviceMoEProjectionSet::IntermediateShard;
            else
                entry.flags |= toMoEExpertFlags(DeviceMoEExpertFlags::PreferredOwner);
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
        if (shard)
        {
            constexpr int published = 1;
            EXPECT_EQ(route_plan_ready.digest(native), fnv1a64(&published, sizeof(published)));
        }
        if (early)
        {
            constexpr std::uint64_t empty = 0;
            EXPECT_EQ(signal.digest(native), fnv1a64(&empty, sizeof(empty)));
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

/** @test Replay serial publications with independent original local fragment ownership. */
TEST(CUDARetainedHCFFN, RecordedFragmentRouting)
{
    const int ordinal = selectVisibleSM86();
    const auto& device = cudaHardwareMetadata().at(ordinal);
    const auto blocks = std::min(2u * static_cast<unsigned>(device.sm_count), hidden / retained_hc_read_warp_channels);
    for (const auto& trace : traces)
        ASSERT_NO_THROW(replay(trace, ordinal, blocks, RetainedReplayRole::ActiveShardEarlyPacket));
}

/** @test Native mapped DMA copies live original packets and preserves the empty destinations and lease. */
TEST(CUDARetainedHCFFN, MappedDmaDispatchContract)
{
    const int ordinal = selectVisibleSM86();
    const auto device = DeviceId::cuda(ordinal);
    const Stream stream;
    const auto native = stream.value;
    using namespace cuda_moe_packet;
    constexpr unsigned lane_capacity = 3;
    constexpr unsigned stage = 1;
    auto& transfers = TransferEngine::instance();
    MappedDispatchLanes lanes(device, lane_capacity), original(device, lane_capacity);
    Array<MoEOverlayActivationDeviceEpochGrant> grant{lane_capacity}, original_grant{lane_capacity};
    Array<MoEOverlayActivationSingleRowDispatchPackLaunch> original_batch{lane_capacity};
    const auto input = transfers.allocateDeviceTransferBuffer(hidden * sizeof(float), device);
    Array<float> ids{top_k}, weights{top_k};
    Array<int> count_input{1}, row_count{1}, placement{experts};
    Array<unsigned> epoch{1};
    Array<DeviceMoEOverlayEpochTicket> ticket{1};
    Array<DeviceMoEOverlayEpochStatus> status{1};
    std::array<float, hidden> seeded_input{};
    for (std::size_t i = 0; i < seeded_input.size(); ++i)
        seeded_input[i] = signedValue(i, 112);
    requireCUDA(cudaMemcpyAsync(input->mutableDeviceData(), seeded_input.data(), sizeof(seeded_input),
        cudaMemcpyHostToDevice, native));
    ids.set(native, [](auto i) { return static_cast<float>(i); });
    weights.set(native, [](auto i) { return static_cast<float>(i + 1) / 64.f; });
    epoch.set(native, [](auto) { return 2u; });
    ticket.set(native, [](auto) { return DeviceMoEOverlayEpochTicket{.epoch = 2, .selector = 4}; });
    status.set(native, [](auto) { return DeviceMoEOverlayEpochStatus{}; });
    const MoEOverlayRoutePlacementBankDeviceView bank{placement.data(), epoch.data()};
    const auto build = [&](MappedDispatchLanes& storage, auto& grants, unsigned lane) {
        auto& device = storage.device[lane];
        return MoEOverlayActivationSingleRowDispatchPackLaunch{
            .packet = {.hidden_rows_fp32 = static_cast<const float*>(input->deviceData()), .route_expert_ids_fp32 = ids.data(),
                .route_weights = weights.data(), .placement = {.banks = {bank, bank},
                    .ticket = ticket.data(), .status = status.data(), .expert_count = experts},
                .active_row_count_device = row_count.data(),
                .packet = {device.rows.data(), device.offsets.data(), device.expert_ids.data(),
                    device.weights.data(), device.original_slots.data(), device.compact_slots.data(),
                    device.payload.data(), 1, top_k, hidden, top_k},
                .control = &device.epoch, .grant = grants.data() + lane,
                .target_participant_id = static_cast<int>(lane), .physical_rows = 1,
                .stage_ordinal = stage, .model_layer_index = 1},
            .publication = {&device.epoch.buffers[stage].dispatch_signal.value, 1},
        };
    };
    const auto original_single = build(original, original_grant, 0);
    original_batch.set(native, [&](auto lane) { return build(original, original_grant, lane); });
    const MoEOverlayActivationSingleRowDispatchBatchLaunch original_many{original_batch.data(), lane_capacity};
    std::vector<ResidentMappedDispatch> prepared;
    for (unsigned lane = 0; lane < lane_capacity; ++lane)
    {
        prepared.push_back({build(lanes, grant, lane), transfers.prepareDeviceToMappedHostCopy(*input, *lanes.region,
            lane * sizeof(DispatchLane) + offsetof(DispatchLane, payload), sizeof(seeded_input))});
    }
    ASSERT_TRUE(prepareResidentDispatchPacket(ordinal));

    for (const unsigned lane_count : {1u, lane_capacity})
    {
        const auto publishCount = [&] {
            requireCUDA(cudaMemcpyAsync(row_count.data(), count_input.data(), sizeof(int), cudaMemcpyDeviceToDevice, native));
        };
        CUDAGraphCapture candidate(native, ordinal), control(native, ordinal);
        ASSERT_TRUE(candidate.beginCapture());
        publishCount();
        ASSERT_TRUE(recordResidentDispatchPackets(std::span(prepared).first(lane_count), ordinal, native));
        ASSERT_TRUE(candidate.endCapture());
        ASSERT_TRUE(candidate.instantiate());
        std::vector<GPUGraphKernelNodeInfo> inventory;
        std::string error;
        ASSERT_TRUE(candidate.inspectKernelNodes(inventory, &error)) << error;
        ASSERT_EQ(inventory.size(), 2u * lane_count);
        EXPECT_EQ(std::count_if(inventory.begin(), inventory.end(), [](const auto& node) {
            return node.grid_x == 1 && node.block_x == 32;
        }), lane_count);
        EXPECT_EQ(std::count_if(inventory.begin(), inventory.end(), [](const auto& node) {
            return node.grid_x == 1 && node.block_x == 1;
        }), lane_count);

        const NativeGraphDump dump;
        requireCUDA(cudaGraphDebugDotPrint(candidate.graph(), dump.path,
            cudaGraphDebugDotFlagsKernelNodeParams | cudaGraphDebugDotFlagsMemcpyNodeParams));
        std::ifstream dot(dump.path);
        const std::string graph{std::istreambuf_iterator<char>{dot}, {}};
        EXPECT_NE(graph.find("prepareResidentDmaMetadata"), std::string::npos);
        EXPECT_NE(graph.find("publishResidentDmaPacket"), std::string::npos);

        ASSERT_TRUE(control.beginCapture());
        publishCount();
        ASSERT_TRUE(lane_count == 1u ? cudaMoEOverlayActivationPackSingleRowDispatch(&original_single, ordinal, native)
                               : cudaMoEOverlayActivationPackSingleRowDispatchBatch(&original_many, ordinal, native));
        ASSERT_TRUE(control.endCapture());
        ASSERT_TRUE(control.instantiate());

        for (const unsigned round : {0u, 1u, 2u, 3u})
        {
            const bool active = round != 2u;
            count_input.set(native, [=](auto) { return active ? 1 : 0; });
            placement.set(native, [=](auto slot) { return int(slot % 2u == 0 ? round % lane_count : lane_count); });
            const auto initial = [=](auto lane) {
                return MoEOverlayActivationDeviceEpochGrant{
                    .digest = {16 + lane, 32 + round}, .generation = 3 + round, .placement_epoch = 2,
                    .stage_count = 2, .physical_rows = 1, .last_published_stage = 0,
                    .endpoint = static_cast<unsigned>(MoEOverlayActivationEndpoint::Continuation),
                    .state = static_cast<unsigned>(MoEOverlayActivationEndpointState::Active),
                    .code = static_cast<unsigned>(MoEOverlayActivationStatusCode::Success),
                    .graph_role = MoEOverlayInferenceGraphRole::MainDecode};
            };
            grant.set(native, initial);
            original_grant.set(native, initial);
            for (unsigned lane = 0; lane < lane_count; ++lane)
            {
                lanes.host[lane] = original.host[lane] = {};
                lanes.host[lane].payload.fill(-1.f);
                original.host[lane].payload.fill(-1.f);
            }
            ASSERT_TRUE(control.launch());
            requireCUDA(cudaStreamSynchronize(native));
            ASSERT_TRUE(candidate.launch());
            for (unsigned lane = 0; lane < lane_count; ++lane)
            {
                auto& actual = lanes.host[lane];
                const auto& expected = original.host[lane];
                auto& publication = actual.epoch.buffers[stage];
                // The original CPU endpoint acquires the lease before the
                // graph as a whole retires. Observe that same publication edge.
                std::atomic_ref<std::uint64_t> lease(publication.dispatch_signal.value);
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
                std::uint64_t published = 0;
                do
                {
                    published = lease.load(std::memory_order_acquire);
                    if (published != 0)
                        break;
                    std::this_thread::yield();
                }
                while (std::chrono::steady_clock::now() < deadline);
                EXPECT_EQ(published, 1u);
                if (published != 1)
                    break;
                const auto& descriptor = publication.dispatch_descriptor;
                const unsigned live_entries = active && round % lane_count == lane ? top_k / 2u : 0u;
                EXPECT_EQ(descriptor.timeline, 1u);
                EXPECT_EQ(descriptor.live_entries, live_entries);
                EXPECT_EQ(descriptor.live_rows, live_entries != 0u);
                EXPECT_EQ(descriptor.payload_bytes, moeOverlayDispatchPayloadBytes(descriptor.live_rows, live_entries, hidden));
                EXPECT_EQ(actual.rows, expected.rows);
                EXPECT_EQ(actual.offsets, expected.offsets);
                EXPECT_EQ(actual.expert_ids, expected.expert_ids);
                EXPECT_EQ(actual.original_slots, expected.original_slots);
                EXPECT_EQ(actual.compact_slots, expected.compact_slots);
                EXPECT_EQ(std::memcmp(actual.weights.data(), expected.weights.data(), sizeof(actual.weights)), 0);
                EXPECT_EQ(std::memcmp(actual.payload.data(), expected.payload.data(), sizeof(actual.payload)), 0);
            }
            requireCUDA(cudaStreamSynchronize(native));
        }
    }
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
