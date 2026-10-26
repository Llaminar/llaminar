/**
 * @file CUDADenseCompactSnapshotsRegression.h
 * @brief Captured exact-dense correction retention and accepted-prefix regression.
 *
 * Reuses the padded-GDN fixture's device, graph, transfer and retirement owners.
 * Original planar dense kernels supply independent native snapshots and M1
 * outputs. The compact candidate must preserve every byte, including revised
 * acceptance after the original input bank has changed. No model policy is
 * selected by these tests; retention formats and terminal contracts are transaction-local. Explicit
 * Full/Threshold tests preserve Dense arithmetic while independently checking
 * per-request device-live representation, protocol and exact publication spans.
 * Explicit post-convolution events additionally capture original native
 * deinterleave on the legacy arm; host-expanded arrays remain oracle-only.
 * Disabled native events include retention, accepted publication and following
 * decode over evolving banks, with final timed bytes checked before reset.
 * Explicit phase diagnostics interleave owned timestamp nodes and emit separately
 * named instrumented records; they never replace canonical complete timing.
 */
#pragma once
#ifdef HAVE_CUDA
#include "kernels/gdn/GDNDenseCompactSnapshots.h"
#include "kernels/gdn/GDNPreparedRetainedInvocation.h"
#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "transfer/TransferEngine.h"
#include "tensors/Tensors.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "kernels/gdn/GDNCUDADenseCapturePlan.h"
#include "backends/ComputeBackend.h"
#include "backends/DeviceUUID.h"
#include <algorithm>
#include <string>
#include <bit>
#include <limits>
#include <set>

namespace
{
namespace cuda_dense_compact_test
{
using namespace llaminar2;
using namespace llaminar2::gdn;
using Layout = GDNRecurrenceStateLayout;

/** @brief Fail immediately before publishing a successful byte/timing receipt.
 * @param condition Required invariant.
 * @param message Precise failed operation. */
inline void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

/** @brief Strict byte oracle, retaining signed zero and nonfinite payloads.
 * @param actual Observed complete physical vector.
 * @param expected Independently authored/native-produced vector.
 * @param label Context for the first differing word. */
inline void equal(const std::vector<float> &actual, const std::vector<float> &expected,
                  const char *label)
{
    require(actual.size() == expected.size(), "Dense compact observation extent differs");
    if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0) return;
    for (std::size_t i = 0; i < actual.size(); ++i)
        if (std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(expected[i]))
        {
            std::ostringstream out;
            out << label << " word=" << i << " actual=0x" << std::hex
                << std::bit_cast<std::uint32_t>(actual[i]) << " expected=0x"
                << std::bit_cast<std::uint32_t>(expected[i]);
            throw std::runtime_error(out.str());
        }
}

/** @brief Independent test expectation from authored cold policy and request count.
 * @param program Captured test policy, not device execution state.
 * @param live Ordered authored request rows.
 * @return Whether this request must retain every Dense row. */
inline bool expectsFullRetention(GDNDenseRetentionProgram program, int live)
{
    return program.mode == GDNDenseRetentionMode::FullSnapshots ||
        (program.mode == GDNDenseRetentionMode::LiveRowThreshold &&
            unsigned(live) < program.compact_min_live_rows);
}

/** @brief Preserve all Exact schemas while naming new explicit retained programs.
 * @param program Authored immutable program.
 * @param live Request count, or -1 for a mixed-request summary without one mode claim.
 * @return Empty for legacy Exact; otherwise closed identity/mode marker fields. */
inline std::string retentionMarkerFields(GDNDenseRetentionProgram program, int live = -1)
{
    if (program.mode == GDNDenseRetentionMode::ExactCorrections) return "";
    std::string fields = std::string(" retention_program=") +
        (program.mode == GDNDenseRetentionMode::FullSnapshots ? "full_snapshots" : "live_row_threshold") +
        " compact_min_live_rows=" + std::to_string(program.compact_min_live_rows) + " retention_encoding=DFS1";
    if (live >= 0) fields += std::string(" actual_retention=") +
        (expectsFullRetention(program, live) ? "full_snapshots" : "exact_corrections");
    return fields;
}

/** @brief Preserve mandatory-terminal schemas and explicitly label retained-only observations.
 * @param kind Cold captured terminal program, never inferred from an absent pointer.
 * @return Empty for Dense defaults; explicit extra fields only for retained-only filters. */
inline std::string terminalMarkerFields(GDNDenseCompactTerminalKind kind)
{
    return kind == GDNDenseCompactTerminalKind::DenseDestination ? "" :
        " compact_terminal_policy=retained_prefixes_only compact_terminal_destination_present=0";
}

/** @brief Label only new explicit KeyMajor events, preserving old ValueMajor output.
 * @param axes Actual captured persistent/checkpoint state axes.
 * @param terminal_kind Actual candidate output contract.
 * @return KeyMajor identity and explicit required-terminal identity when present. */
inline std::string eventStateLayoutFields(Layout axes, GDNDenseCompactTerminalKind terminal_kind)
{
    if (axes == Layout::ValueMajorKeyContiguous) return "";
    require(axes == Layout::KeyMajorValueContiguous, "Dense event state axes invalid");
    return std::string(" state_layout=key_major_value_contiguous") +
        (terminal_kind == GDNDenseCompactTerminalKind::DenseDestination ?
            " compact_terminal_policy=dense_destination compact_terminal_destination_present=1" : "");
}

/** @brief Observe real hardware for a model-free cold capture without installing a fake inventory.
 * @param geometry Exact immutable request/head/row/width capture geometry.
 * @param axes Exact persistent/checkpoint matrix orientation.
 * @param terminal Explicit required-Dense or retained-prefixes-only output contract.
 * @param stream Existing fixture-owned explicit stream.
 * @return Same immutable plan type used by the canonical inventory entrypoint. */
inline CUDADenseRetentionCapturePlan fixtureCapturePlan(const GDNDenseCompactGeometry &geometry,
    Layout axes, GDNDenseCompactTerminalKind terminal, cudaStream_t stream)
{
    int device = -1;
    checkCuda(cudaStreamGetDevice(stream, &device), "Dense fixture native stream device");
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, device), "Dense fixture native hardware observation");
    llaminar2::ComputeDevice observed{};
    observed.type = llaminar2::ComputeBackendType::GPU_CUDA;
    observed.device_id = device;
    observed.compute_units = properties.multiProcessorCount;
    observed.compute_capability = properties.major * 10 + properties.minor;
    observed.uuid = llaminar2::formatDeviceUUID(properties.uuid.bytes);
    return prepareObservedCUDADenseRetentionCapturePlan(geometry, axes, terminal, observed, stream);
}

/** @brief Exact native transaction and original dense controls with retained graphs.
 * @details Captures are constructed only after every buffer exists. An always
 * armed exceptional retirement owner outlives the ordinary observation owner,
 * protecting subsequent submissions even after an earlier finish(). */
class Cell final
{
  public:
    static constexpr int Guard = 16;
    static constexpr float Poison = -917.25f;
    static constexpr std::uint64_t Generation = 0x100000019ULL;
    int heads, keys, dk, dv, requests, rows, offset, state_stride, merged_stride;
    bool normalize, terminal_alias;
    const GDNDenseCompactTerminalKind terminal_kind; ///< Immutable captured output contract.
    const GDNDenseAcceptedSelectionKind selection_kind; ///< Captured metadata meaning.
    const int selection_stride; ///< Existing source-view stride, not a conversion buffer.
    Layout axes;
    GDNDenseCompactGeometry geometry;
    std::vector<float> merged_host, q_host, k_host, v_host, alpha_host, beta_host,
        a_host, bias_host, initial_host;
    std::vector<int> live_host, accepted_host;
    CudaStreamHandle stream;
    CudaFloatBuffer merged, query, key, value, alpha, beta, a, bias, initial,
        terminal, compact, output, oracle_state, oracle_output, snapshots,
        oracle_terminal, accepted_state, following, oracle_following;
    CudaIntBuffer live, accepted;
    const CUDADenseRetentionCapturePlan capture_plan; ///< Cold exact-stream scheduling authority, never keyed by live L.
    std::unique_ptr<CudaCapturedGraph> capture, dense, publication, publication_alias;
    CudaMergedReplayRetirement observation, exceptional_retirement;

    /** @brief Author identical raw inputs and capture candidate/control programs.
     * @param h Local value heads, including small-grid and ordinary model shapes.
     * @param kh Encoded Q/K heads; source mapping may repeat unevenly.
     * @param key_width Original CUDA64/128 width.
     * @param value_width Exact width including incomplete column tiles.
     * @param r Independent request partitions.
     * @param m Physical verifier rows per request.
     * @param layout Immutable Dense/checkpoint axes.
     * @param norm Original Q/K normalization choice.
     * @param alias Required terminal may exactly equal original Dense input.
     * @param edge Include zero/subnormal/infinity/NaN arithmetic witnesses.
     * @param program Explicit immutable retention; default preserves every Exact control.
     * @param terminal_output Explicit terminal program; existing calls remain DenseDestination.
     * @param source_head_offset Frozen modulo head origin; old fixtures retain -5, production-shaped events use0.
     * @param selected_kind Existing metadata meaning, Counts by default.
     * @param selected_stride Positive metadata element stride. */
    Cell(int h, int kh, int key_width, int value_width, int r, int m, Layout layout,
         bool norm = true, bool alias = false, bool edge = false, GDNDenseRetentionProgram program = {},
         GDNDenseCompactTerminalKind terminal_output = GDNDenseCompactTerminalKind::DenseDestination,
         int source_head_offset = -5,
         GDNDenseAcceptedSelectionKind selected_kind = GDNDenseAcceptedSelectionKind::AcceptedCounts,
         int selected_stride = 1)
        : heads(h), keys(kh), dk(key_width), dv(value_width), requests(r), rows(m),
          offset(source_head_offset), state_stride(h * key_width * value_width),
          merged_stride(2 * kh * key_width + h * value_width), normalize(norm),
          terminal_alias(alias), terminal_kind(terminal_output), selection_kind(selected_kind), selection_stride(selected_stride), axes(layout), geometry(h, key_width, value_width, r, m, program),
          merged_host(std::size_t(r * m + r) * merged_stride + Guard),
          q_host(std::size_t(r * m + r) * h * key_width + Guard),
          k_host(q_host.size()), v_host(std::size_t(r * m + r) * h * value_width + Guard),
          alpha_host(std::size_t(r * m + r) * h + Guard), beta_host(alpha_host.size()),
          a_host(h + Guard, -0.17f), bias_host(h + Guard, 0.03f),
          initial_host(geometry.denseBankFloats() + Guard, Poison),
          live_host(r, m), accepted_host(std::size_t(r) * selected_stride, 0), merged(merged_host.size()),
          query(q_host.size()), key(k_host.size()), value(v_host.size()),
          alpha(alpha_host.size()), beta(beta_host.size()), a(a_host.size()), bias(bias_host.size()),
          initial(initial_host.size()), terminal(terminal_kind == GDNDenseCompactTerminalKind::DenseDestination ? initial_host.size() : Guard, Poison),
          compact(geometry.bankFloats() + Guard, Poison),
          output(geometry.outputFloats() + Guard, Poison), oracle_state(initial_host.size()),
          oracle_output(geometry.outputFloats() + Guard, Poison),
          snapshots(std::size_t(r * m) * state_stride + Guard, Poison),
          oracle_terminal(initial_host.size(), Poison), accepted_state(initial_host.size(), Poison),
          following(std::size_t(r) * h * value_width + Guard, Poison),
          oracle_following(following.count, Poison), live(live_host), accepted(accepted_host),
          capture_plan(fixtureCapturePlan(geometry, axes, terminal_kind, stream.stream)),
          observation(stream.stream), exceptional_retirement(stream.stream)
    {
        require(terminal_kind == GDNDenseCompactTerminalKind::DenseDestination ||
            terminal_kind == GDNDenseCompactTerminalKind::RetainedPrefixesOnly, "Unknown Dense terminal contract");
        require(terminal_kind == GDNDenseCompactTerminalKind::DenseDestination || !terminal_alias,
            "Retained-only capture cannot request a terminal alias");
        for (std::size_t i = 0; i < merged_host.size(); ++i)
            merged_host[i] = float(int((i * 13 + i / 37) % 97) - 48) * 0.0031f;
        for (std::size_t i = 0; i < alpha_host.size(); ++i)
        {
            alpha_host[i] = float(int(i % 11) - 5) * 0.09f;
            beta_host[i] = float(int(i % 13) - 6) * 0.11f;
        }
        for (int request = 0; request < r; ++request)
            for (int head = 0; head < h; ++head)
                for (int k = 0; k < dk; ++k)
                    for (int v = 0; v < dv; ++v)
                    {
                        const auto at = (std::size_t(request) * h + head) * dk * dv +
                            (axes == Layout::KeyMajorValueContiguous ? std::size_t(k) * dv + v : std::size_t(v) * dk + k);
                        initial_host[at] = float((k * 7 + v * 3 + head + request) % 53 - 26) * 0.00027f;
                    }
        if (edge)
        {
            // Source/state changes enter both native arms; expected values are
            // never computed by host libm or an allclose nonfinite exception.
            initial_host[0] = -0.0f;
            initial_host[1] = std::numeric_limits<float>::denorm_min();
            initial_host[2] = -std::numeric_limits<float>::denorm_min();
            initial_host[3] = std::numeric_limits<float>::infinity();
            initial_host[4] = -std::numeric_limits<float>::infinity();
            initial_host[5] = std::bit_cast<float>(std::uint32_t{0x7fc01234});
            merged_host[0] = -0.0f;
            merged_host[1] = std::numeric_limits<float>::denorm_min();
        }
        for (int row = 0; row < r * m + r; ++row)
            for (int head = 0; head < h; ++head)
            {
                int source = (head + offset) % kh;
                if (source < 0) source += kh;
                const auto from = std::size_t(row) * merged_stride;
                const auto qk_to = (std::size_t(row) * h + head) * dk;
                std::copy_n(merged_host.data() + from + source * dk, dk, q_host.data() + qk_to);
                std::copy_n(merged_host.data() + from + kh * dk + source * dk, dk, k_host.data() + qk_to);
                std::copy_n(merged_host.data() + from + 2 * kh * dk + head * dv, dv,
                    v_host.data() + (std::size_t(row) * h + head) * dv);
            }
        merged.copyFrom(merged_host); query.copyFrom(q_host); key.copyFrom(k_host); value.copyFrom(v_host);
        alpha.copyFrom(alpha_host); beta.copyFrom(beta_host); a.copyFrom(a_host); bias.copyFrom(bias_host);
        initial.copyFrom(initial_host); oracle_state.copyFrom(initial_host);
        capture = std::make_unique<CudaCapturedGraph>(stream.stream, [&] {
            return launchCUDADenseCompactRecurrence(input(), stream.stream, capture_plan);
        });
        dense = std::make_unique<CudaCapturedGraph>(stream.stream, [&] {
            return launchCUDAPlanarRecurrence(denseInput(), stream.stream);
        });
        publication = std::make_unique<CudaCapturedGraph>(stream.stream, [&] {
            return launchCUDADenseCompactAcceptedPrefix({transaction(), acceptedSelection(), accepted_state.ptr}, stream.stream);
        });
        publication_alias = std::make_unique<CudaCapturedGraph>(stream.stream, [&] {
            return launchCUDADenseCompactAcceptedPrefix({transaction(), acceptedSelection(), initial.ptr}, stream.stream);
        });
        inspect(*capture, true);
        inspect(*publication, false);
    }

    /** @return Single checked compact identity; no alternate sizing calculation. */
    GDNDenseCompactTransaction transaction() const
    {
        return {geometry, initial.ptr, compact.ptr, geometry.bankFloats(), Generation,
            axes, GDNDenseCompactArithmetic::CUDAEightPartFusedUpdateV1, geometry.encoding()};
    }

    /** @return Exact captured view of the existing metadata allocation, without conversion. */
    GDNDenseAcceptedSelection acceptedSelection() const
    {
        if (selection_kind == GDNDenseAcceptedSelectionKind::AcceptedCounts)
            return GDNDenseAcceptedSelection::acceptedCounts(geometry, accepted.ptr, accepted_host.size(), selection_stride);
        if (selection_kind == GDNDenseAcceptedSelectionKind::OwnRequestFlattenedRows)
            return GDNDenseAcceptedSelection::ownRequestFlattenedRows(geometry, accepted.ptr, accepted_host.size(), selection_stride);
        require(selection_kind == GDNDenseAcceptedSelectionKind::RequestTerminalLengths,
            "Dense compact selection kind invalid");
        return GDNDenseAcceptedSelection::requestTerminalLengths(geometry, accepted.ptr, accepted_host.size(), selection_stride);
    }

    /** @return Exact raw merged sources with legacy state/snapshot fields absent. */
    GDNDenseCompactCaptureInput input() const
    {
        return {{DeviceRequestRowRanges::deviceCounted(requests, rows, live.ptr),
            keys, heads, dk, dv, offset, normalize, merged.ptr, alpha.ptr, beta.ptr,
            a.ptr, bias.ptr, output.ptr, nullptr, nullptr, nullptr, 0, 0, axes},
            transaction(), terminal_kind == GDNDenseCompactTerminalKind::DenseDestination ?
                GDNDenseCompactTerminalOutput::denseDestination(terminal_alias ? initial.ptr : terminal.ptr) :
                GDNDenseCompactTerminalOutput::retainedPrefixesOnly()};
    }

    /** @return Original planar body forced by real full-row snapshot ownership. */
    PlanarRecurrenceInput denseInput() const
    {
        return {DeviceRequestRowRanges::deviceCounted(requests, rows, live.ptr),
            heads, dk, dv, normalize, query.ptr, key.ptr, value.ptr, alpha.ptr, beta.ptr,
            a.ptr, bias.ptr, oracle_output.ptr, oracle_state.ptr, oracle_terminal.ptr,
            snapshots.ptr, state_stride, requests * rows, axes};
    }

    /** @brief Authenticate native family before interpreting its single typed argument.
     * @param graph Actual retained executable source graph.
     * @param capturing Select capture versus accepted reconstruction ABI. */
    void inspect(const CudaCapturedGraph &graph, bool capturing, float *accepted_destination = nullptr) const
    {
        std::size_t count = 0;
        checkCuda(cudaGraphGetNodes(graph.graph, nullptr, &count), "dense compact node count");
        require(count == 1, "Dense compact entrypoint must be one kernel");
        cudaGraphNode_t node{};
        checkCuda(cudaGraphGetNodes(graph.graph, &node, &count), "dense compact node");
        cudaGraphNodeType kind{};
        checkCuda(cudaGraphNodeGetType(node, &kind), "dense compact node kind");
        require(kind == cudaGraphNodeTypeKernel, "Dense compact graph contains nonkernel work");
        cudaKernelNodeParams p{};
        checkCuda(cudaGraphKernelNodeGetParams(node, &p), "dense compact arguments");
        const char *name = nullptr;
        checkCuda(cudaFuncGetName(&name, p.func), "dense compact function name");
        require(name && std::string(name).find(capturing ? capture_plan.captureFunctionFamily() :
            "cuda_gdn_dense_compact_accepted_prefix_kernel") != std::string::npos,
            "Dense compact graph selected foreign native function");
        require(p.kernelParams && p.kernelParams[0], "Dense compact argument absent");
        require(p.gridDim.x == unsigned(requests * heads) && p.gridDim.z == 1 &&
            p.gridDim.y == unsigned(capturing ? (dv + 31) / 32 :
                geometry.matrixTileCount(4096)) &&
            p.blockDim.x == 256 && p.blockDim.y == 1 && p.blockDim.z == 1 &&
            p.sharedMemBytes == unsigned(capturing ? (2 * dk + 512) * sizeof(float) : 0),
            "Dense compact launch geometry differs");
        if (capturing)
        {
            auto actual = input();
            std::memcpy(&actual, p.kernelParams[0], sizeof(actual));
            require(actual.transaction.storage == compact.ptr && actual.transaction.initial_state == initial.ptr &&
                actual.transaction.generation == Generation && actual.transaction.layout == axes &&
                actual.transaction.encoding == geometry.encoding() &&
                actual.transaction.geometry.retentionProgram().mode == geometry.retentionProgram().mode &&
                actual.transaction.geometry.retentionProgram().compact_min_live_rows == geometry.retentionProgram().compact_min_live_rows &&
                actual.transaction.geometry.bankFloats() == geometry.bankFloats() &&
                actual.sources.rows.lengthOwner() == live.ptr && actual.sources.merged == merged.ptr &&
                actual.sources.output == output.ptr && actual.terminal.kind() == terminal_kind && actual.terminal.destination() == input().terminal.destination(),
                "Dense compact captured ownership differs");
        }
        else
        {
            GDNDenseCompactAcceptedPrefixInput actual{transaction(), acceptedSelection(), accepted_state.ptr};
            std::memcpy(&actual, p.kernelParams[0], sizeof(actual));
            require(actual.transaction.storage == compact.ptr && actual.transaction.generation == Generation &&
                actual.transaction.encoding == geometry.encoding() &&
                actual.transaction.geometry.retentionProgram().mode == geometry.retentionProgram().mode &&
                actual.transaction.geometry.retentionProgram().compact_min_live_rows == geometry.retentionProgram().compact_min_live_rows &&
                actual.selection.sameBinding(acceptedSelection()) &&
                actual.destination_state == (accepted_destination ? accepted_destination : accepted_state.ptr),
                "Dense compact accepted ownership differs");
        }
        cudaFuncAttributes resources{};
        checkCuda(cudaFuncGetAttributes(&resources, p.func), "dense compact resources");
        require(resources.localSizeBytes == 0, "Dense compact native private memory");
    }

    /** @brief Publish mutable request counts on the exact stream; vectors live through retirement. */
    void publishCounts()
    {
        checkCuda(cudaMemcpyAsync(live.ptr, live_host.data(), requests * sizeof(int),
            cudaMemcpyHostToDevice, stream.stream), "dense compact live counts");
        checkCuda(cudaMemcpyAsync(accepted.ptr, accepted_host.data(), accepted_host.size() * sizeof(int),
            cudaMemcpyHostToDevice, stream.stream), "dense compact accepted counts");
    }

    /** @brief Reset only data; graph topology and every embedded pointer stay retained. */
    void reset()
    {
        observation.finish();
        initial.copyFrom(initial_host); oracle_state.copyFrom(initial_host);
        terminal.fill(Poison); oracle_terminal.fill(Poison); snapshots.fill(Poison);
        output.fill(Poison); oracle_output.fill(Poison); accepted_state.fill(Poison);
        following.fill(Poison); oracle_following.fill(Poison);
    }

    /** @return Original dense state after each request's selected accepted count.
     * @param shots Native full-row snapshots in original request/row coordinates.
     * @param counts Independent accepted prefix for every request. */
    std::vector<float> selectedState(const std::vector<float> &shots, const std::vector<int> &counts) const
    {
        auto expected = initial_host;
        for (int request = 0; request < requests; ++request)
            if (counts[request])
                std::copy_n(shots.data() + std::size_t(request * rows + counts[request] - 1) * state_stride,
                    state_stride, expected.data() + std::size_t(request) * state_stride);
        return expected;
    }

    /** @brief Author protocol independently of the production representation resolver.
     * @param live Authored request-local count, already within this fixture's capacity.
     * @param generation Actual immutable prepared owner identity; legacy tests retain their fixed token.
     * @return Ten raw words; legacy Exact writes/checks only the first seven. */
    std::array<std::uint32_t, 10> expectedRetentionHeader(int live, std::uint64_t generation = Generation) const
    {
        const auto program = geometry.retentionProgram();
        const bool split = program.mode != GDNDenseRetentionMode::ExactCorrections;
        return {std::uint32_t(split ? GDNDenseCompactEncoding::SplitRetentionV1 : GDNDenseCompactEncoding::ExactCorrectionsV1),
            unsigned(live), unsigned(live), std::uint32_t(generation), std::uint32_t(generation >> 32),
            std::uint32_t(axes), std::uint32_t(GDNDenseCompactArithmetic::CUDAEightPartFusedUpdateV1),
            std::uint32_t(expectsFullRetention(program, live) ? GDNDenseRetentionMode::FullSnapshots : GDNDenseRetentionMode::ExactCorrections),
            std::uint32_t(program.mode), program.compact_min_live_rows};
    }

    /** @brief Check the actual last timed retained header and every post-row matrix.
     * @param shots Independent last-transaction native full snapshots.
     * @details cp0/revised-zero and untouched spans are proved by dirty functional
     * replay; this checks timed policy and matrix publication before any reset. */
    void checkTimedRetention(const std::vector<float> &shots) const
    {
        const auto observed = compact.toHost();
        const auto program = geometry.retentionProgram();
        const bool split = program.mode != GDNDenseRetentionMode::ExactCorrections;
        for (int request = 0; request < requests; ++request)
            for (int head = 0; head < heads; ++head)
            {
                const auto base = (std::size_t(request) * heads + head) * geometry.headStrideFloats();
                const auto words = expectedRetentionHeader(live_host[request]);
                require(std::memcmp(observed.data() + base, words.data(), (split ? 10 : 7) * sizeof(std::uint32_t)) == 0,
                    "Dense timed retention policy/header differs");
                const bool full = expectsFullRetention(program, live_host[request]);
                const int last = full ? live_host[request] : live_host[request] / 8;
                for (int slot = 1; slot <= last; ++slot)
                {
                    const int completed = full ? slot : slot * 8;
                    const auto *source = shots.data() + std::size_t(request * rows + completed - 1) * state_stride +
                        std::size_t(head) * dk * dv;
                    require(std::memcmp(observed.data() + base + geometry.checkpointOffsetFloats() +
                        std::size_t(slot) * geometry.checkpointStrideFloats(), source,
                        geometry.matrixFloats() * sizeof(float)) == 0, "Dense timed retained matrix differs");
                }
            }
    }

    /** @brief Prove capture output, immutable checkpoints and all unpublished padding.
     * @param before Actual dirty compact bank before this retained replay.
     * @param shots Independently computed original dense snapshots.
     * @param retained_observation Optional actual external arena bytes; no payload normalization.
     * @param terminal_observation Optional actual external required-terminal bytes.
     * @param generation Frozen workspace owner id used by the prepared invocation.
     * @param output_observation Optional actual stage tensor output with its exact logical extent. */
    void checkCapture(const std::vector<float> &before, const std::vector<float> &shots,
        const std::vector<float> *retained_observation = nullptr,
        const std::vector<float> *terminal_observation = nullptr,
        std::uint64_t generation = Generation,
        const std::vector<float> *output_observation = nullptr)
    {
        auto expected_output = oracle_output.toHost();
        if (output_observation) {
            require(output_observation->size() == geometry.outputFloats(), "External stage output has a foreign logical extent");
            expected_output.resize(geometry.outputFloats());
        }
        equal(output_observation ? *output_observation : output.toHost(), expected_output, "dense compact forward output");
        if (terminal_kind == GDNDenseCompactTerminalKind::DenseDestination)
        {
            auto expected_terminal = selectedState(shots, live_host);
            if (terminal_observation && terminal_observation->size() == geometry.denseBankFloats())
                expected_terminal.resize(geometry.denseBankFloats());
            equal(terminal_observation ? *terminal_observation : (terminal_alias ? initial.toHost() : terminal.toHost()),
                expected_terminal, "dense compact actual terminal");
        }
        else
            equal(terminal.toHost(), std::vector<float>(terminal.count, Poison),
                "retained-only unbound terminal guard changed");
        const auto observed = retained_observation ? *retained_observation : compact.toHost();
        auto expected = before;
        for (int request = 0; request < requests; ++request)
            for (int head = 0; head < heads; ++head)
            {
                const auto base = (std::size_t(request) * heads + head) * geometry.headStrideFloats();
                const auto program = geometry.retentionProgram();
                const bool full = expectsFullRetention(program, live_host[request]);
                const bool split = program.mode != GDNDenseRetentionMode::ExactCorrections;
                const auto words = expectedRetentionHeader(live_host[request], generation);
                const std::size_t header_words = split ? 10 : 7;
                for (std::size_t i = 0; i < header_words; ++i) expected[base + i] = std::bit_cast<float>(words[i]);
                const int checkpoints = 1 + (full ? live_host[request] : live_host[request] / 8);
                std::vector<GDNDenseCompactByteSpan> spans{{0, header_words * sizeof(std::uint32_t)}};
                for (int slot = 0; slot < checkpoints; ++slot)
                {
                    const int completed = full ? slot : slot * 8;
                    const auto *source = completed ? shots.data() + std::size_t(request * rows + completed - 1) * state_stride +
                        std::size_t(head) * dk * dv : initial_host.data() + (std::size_t(request) * heads + head) * dk * dv;
                    const auto at = geometry.checkpointOffsetFloats() + std::size_t(slot) * geometry.checkpointStrideFloats();
                    std::copy_n(source, geometry.matrixFloats(), expected.data() + base + at);
                    spans.push_back({at * sizeof(float), geometry.matrixFloats() * sizeof(float)});
                }
                // Full has NO live correction records. Exclusive arena overlap
                // is accounted by the real matrix spans above; no extra clear.
                if (!full) for (int row = 0; row < live_host[request]; ++row)
                {
                    const auto at = geometry.recordsOffsetFloats() + std::size_t(row) * geometry.recordStrideFloats();
                    std::copy_n(observed.data() + base + at, geometry.recordFloats(), expected.data() + base + at);
                    spans.push_back({at * sizeof(float), geometry.recordFloats() * sizeof(float)});
                }
                for (std::size_t index = 0; index < spans.size(); ++index)
                {
                    GDNDenseCompactByteSpan actual{};
                    require(geometry.publishedSpanFor(unsigned(live_host[request]), index, actual) &&
                        actual.offset_bytes == spans[index].offset_bytes && actual.live_bytes == spans[index].live_bytes,
                        "Dense retention live span differs from independent oracle");
                }
                GDNDenseCompactByteSpan past{1, 1};
                require(!geometry.publishedSpanFor(unsigned(live_host[request]), spans.size(), past) &&
                    !past.offset_bytes && !past.live_bytes, "Dense retention exposes inactive/padded payload");
            }
        equal(observed, expected, "dense compact checkpoints and untouched capacity");
        equal(merged.toHost(), merged_host, "dense compact source immutable");
        if (!terminal_alias) equal(initial.toHost(), initial_host, "dense compact input immutable");
        if (terminal_kind == GDNDenseCompactTerminalKind::RetainedPrefixesOnly)
        {
            equal(query.toHost(), q_host, "retained-only query source immutable");
            equal(key.toHost(), k_host, "retained-only key source immutable");
            equal(value.toHost(), v_host, "retained-only value source immutable");
            equal(alpha.toHost(), alpha_host, "retained-only alpha source immutable");
            equal(beta.toHost(), beta_host, "retained-only beta source immutable");
            equal(a.toHost(), a_host, "retained-only gate source immutable");
            equal(bias.toHost(), bias_host, "retained-only bias source immutable");
        }
    }

    /** @brief Advance actual published and independent accepted banks by one native dense row.
     * @param expected Exact accepted state before the following native decode. */
    void checkFollowing(const std::vector<float> &expected)
    {
        oracle_state.copyFrom(expected);
        for (int request = 0; request < requests; ++request)
        {
            const int row = requests * rows + request;
            const auto launch = [&](float *state, float *out) {
                return launchCUDAPlanarRecurrence({DeviceRequestRowRanges::fullyActive(1), heads, dk, dv, normalize,
                    query.ptr + std::size_t(row) * heads * dk, key.ptr + std::size_t(row) * heads * dk,
                    value.ptr + std::size_t(row) * heads * dv, alpha.ptr + std::size_t(row) * heads,
                    beta.ptr + std::size_t(row) * heads, a.ptr, bias.ptr, out + std::size_t(request) * heads * dv,
                    state + std::size_t(request) * state_stride, state + std::size_t(request) * state_stride,
                    nullptr, 0, 0, axes}, stream.stream);
            };
            require(launch(accepted_state.ptr, following.ptr), "Dense compact next row rejected");
            require(launch(oracle_state.ptr, oracle_following.ptr), "Dense oracle next row rejected");
        }
        observation.finish();
        equal(accepted_state.toHost(), oracle_state.toHost(), "dense compact next-row state");
        equal(following.toHost(), oracle_following.toHost(), "dense compact next-row output");
    }

    /** @brief Compare every original dense grouped output/snapshot to real M1 calls.
     * @param shots Actual original grouped dense snapshots. */
    void checkSerial(const std::vector<float> &shots)
    {
        oracle_state.copyFrom(initial_host);
        std::vector<float> expected_output(geometry.outputFloats() + Guard, Poison);
        std::fill_n(expected_output.begin(), geometry.outputFloats(), 0.0f);
        oracle_output.copyFrom(expected_output);
        for (int request = 0; request < requests; ++request)
            for (int local = 0; local < live_host[request]; ++local)
            {
                const auto row = std::size_t(request * rows + local);
                auto *state = oracle_state.ptr + std::size_t(request) * state_stride;
                require(launchCUDAPlanarRecurrence({DeviceRequestRowRanges::fullyActive(1), heads, dk, dv, normalize,
                    query.ptr + row * heads * dk, key.ptr + row * heads * dk, value.ptr + row * heads * dv,
                    alpha.ptr + row * heads, beta.ptr + row * heads, a.ptr, bias.ptr,
                    oracle_output.ptr + row * heads * dv, state, state, nullptr, 0, 0, axes}, stream.stream),
                    "Dense serial oracle launch rejected");
                observation.finish();
                const auto actual = oracle_state.toHost();
                require(std::memcmp(actual.data() + std::size_t(request) * state_stride,
                    shots.data() + row * state_stride, state_stride * sizeof(float)) == 0,
                    "Original grouped dense snapshot differs from serial row");
            }
        equal(output.toHost(), oracle_output.toHost(), "dense compact versus serial output");
    }

    /** @brief Exercise retained dirty counts and every revised accepted prefix.
     * @param live_schedule Optional authored per-request sequence; empty preserves legacy four replays. */
    void run(const std::vector<std::vector<int>> &live_schedule = {})
    {
        const int replays = live_schedule.empty() ? 4 : int(live_schedule.size());
        for (int replay = 0; replay < replays; ++replay)
        {
            reset();
            if (live_schedule.empty())
                for (int request = 0; request < requests; ++request)
                    live_host[request] = replay == 0 || replay == 3 ? rows :
                        replay == 1 ? std::max(0, rows - request - 1) : 0;
            else
            {
                require(live_schedule[replay].size() == std::size_t(requests), "Dense retained request schedule extent differs");
                live_host = live_schedule[replay];
                for (int live : live_host) require(live >= 0 && live <= rows, "Dense retained live count outside capacity");
            }
            std::fill(accepted_host.begin(), accepted_host.end(), 0);
            publishCounts(); observation.finish();
            const auto before = compact.toHost();
            capture->launch(stream.stream); dense->launch(stream.stream); observation.finish();
            const auto shots = snapshots.toHost();
            checkCapture(before, shots);
            if (replay == 0 || !live_schedule.empty()) checkSerial(shots);
            const auto retained = compact.toHost();
            // Descending revisions prove cp0/checkpoints do not borrow the
            // initial bank after a preceding accepted publication changes it.
            for (int selected = rows; selected >= 0; --selected)
            {
                for (int request = 0; request < requests; ++request)
                    accepted_host[request] = std::min(selected, live_host[request]);
                publishCounts();
                publication->launch(stream.stream);
                publication_alias->launch(stream.stream);
                observation.finish();
                const auto expected = selectedState(shots, accepted_host);
                equal(accepted_state.toHost(), expected, "dense compact accepted prefix");
                equal(initial.toHost(), expected, "dense compact revised in-place acceptance");
                checkFollowing(expected);
            }
            equal(compact.toHost(), retained, "dense compact retained source changed by publication");
        }
        std::cout << "CUDA_GDN_DENSE_COMPACT_BYTES heads=" << heads << " dk=" << dk << " dv=" << dv
            << " requests=" << requests << " rows=" << rows << " axes=" << int(axes)
            << " normalize=" << normalize << " terminal_alias=" << terminal_alias
            << retentionMarkerFields(geometry.retentionProgram()) << terminalMarkerFields(terminal_kind)
            << " dirty_replays=" << replays << " all_acceptances=1 serial_rows=1 next_row=1 passed=1\n";
    }
};
} // namespace cuda_dense_compact_test
} // namespace

TEST_F(Test__CUDAGDNPaddedRealLength, DenseCompactExactAcceptedPrefixes)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "dense compact selected device");
    using namespace cuda_dense_compact_test;
    for (int dk : {64, 128})
        for (auto axes : {Layout::KeyMajorValueContiguous, Layout::ValueMajorKeyContiguous})
            for (int rows = 1; rows <= 16; ++rows)
            {
                SCOPED_TRACE(::testing::Message() << "DK=" << dk << " M=" << rows << " axes=" << int(axes));
                Cell(3, 2, dk, rows % 2 ? 17 : 65, 1, rows, axes, rows % 3 != 0, rows % 4 == 0).run();
            }
}

TEST_F(Test__CUDAGDNPaddedRealLength, DenseCompactRaggedAndNonfinite)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "dense compact selected device");
    using namespace cuda_dense_compact_test;
    for (int dk : {64, 128})
        for (auto axes : {Layout::KeyMajorValueContiguous, Layout::ValueMajorKeyContiguous})
        {
            Cell(3, 4, dk, 129, 3, 31, axes, true, false).run();
            Cell(3, 4, dk, 33, 3, 9, axes, false, true, true).run();
            Cell(3, 2, dk, 1, 1, 3, axes, false).run();
            Cell(3, 2, dk, 512, 1, 8, axes, true).run();
        }
}

TEST_F(Test__CUDAGDNPaddedRealLength, DenseCompactAdmission)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "dense compact selected device");
    using namespace cuda_dense_compact_test;
    Cell cell(3, 2, 128, 65, 3, 9, Layout::KeyMajorValueContiguous);
    const auto original = cell.input();
    EXPECT_FALSE(launchCUDADenseCompactRecurrence(original, nullptr, cell.capture_plan));
    for (int fault = 0; fault < 6; ++fault)
    {
        auto bad = original;
        if (fault == 0) bad.transaction.storage_floats = bad.transaction.geometry.bankFloats() - 1;
        if (fault == 1) bad.transaction.generation = 0;
        if (fault == 2) bad.sources.snapshots = cell.snapshots.ptr;
        if (fault == 3) bad.sources.output = cell.initial.ptr;
        if (fault == 4) bad.transaction.layout = static_cast<Layout>(255);
        if (fault == 5) bad.terminal = GDNDenseCompactTerminalOutput::denseDestination(cell.initial.ptr + 1);
        EXPECT_FALSE(launchCUDADenseCompactRecurrence(bad, cell.stream.stream, cell.capture_plan));
    }
    auto selected = GDNDenseCompactAcceptedPrefixInput{cell.transaction(), cell.acceptedSelection(), cell.accepted_state.ptr};
    EXPECT_FALSE(launchCUDADenseCompactAcceptedPrefix(selected, nullptr));
    selected.destination_state = cell.initial.ptr + 1;
    EXPECT_FALSE(launchCUDADenseCompactAcceptedPrefix(selected, cell.stream.stream));
    cell.observation.finish();
    equal(cell.initial.toHost(), cell.initial_host, "Rejected dense compact input untouched");
    equal(cell.compact.toHost(), std::vector<float>(cell.compact.count, Cell::Poison), "Rejected dense compact output untouched");
}

/** @test Packed and scalar accepted endpoints retain identical words, guards and revised prefixes.
 * @details Aligned widths exercise four-word stores on both axes; odd widths and
 * every float-aligned offset exercise legal endpoints without assuming float4
 * alignment. All expected values come from original native dense snapshots.
 * Dirty request holes and zero/full revisions reuse the same captured graphs.
 */
TEST_F(Test__CUDAGDNPaddedRealLength, DenseCompactPublicationEndpointAlignment)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "dense endpoint alignment selected device");
    using namespace cuda_dense_compact_test;
    const std::array<GDNDenseRetentionProgram, 3> programs{{
        {GDNDenseRetentionMode::ExactCorrections, 0},
        {GDNDenseRetentionMode::FullSnapshots, 0},
        {GDNDenseRetentionMode::LiveRowThreshold, 4}}};
    for (int dk : {64, 128})
        for (int dv : {4, 128, 129})
            for (auto axes : {Layout::KeyMajorValueContiguous, Layout::ValueMajorKeyContiguous})
                for (const auto program : programs)
                {
                    SCOPED_TRACE(::testing::Message() << "DK=" << dk << " DV=" << dv
                        << " axes=" << int(axes) << " program=" << unsigned(program.mode));
                    Cell cell(3, 2, dk, dv, 3, 16, axes, true, false, true, program,
                        GDNDenseCompactTerminalKind::RetainedPrefixesOnly);
                    CudaFloatBuffer endpoint(cell.geometry.denseBankFloats() + 2 * Cell::Guard,
                        Cell::Poison);
                    std::array<std::unique_ptr<CudaCapturedGraph>, 4> publications;
                    for (std::size_t offset = 0; offset < publications.size(); ++offset)
                        publications[offset] = std::make_unique<CudaCapturedGraph>(cell.stream.stream, [&] {
                            return launchCUDADenseCompactAcceptedPrefix({cell.transaction(),
                                cell.acceptedSelection(), endpoint.ptr + Cell::Guard + offset}, cell.stream.stream);
                        });
                    // This observer retires queued work before graphs/endpoints
                    // unwind, including an assertion or failed byte comparison.
                    CudaMergedReplayRetirement retirement(cell.stream.stream);
                    for (const auto &counts : std::array<std::array<int, 3>, 3>{{
                            {16, 16, 16}, {16, 3, 0}, {16, 16, 16}}})
                    {
                        cell.reset();
                        cell.live_host.assign(counts.begin(), counts.end());
                        std::fill(cell.accepted_host.begin(), cell.accepted_host.end(), 0);
                        cell.publishCounts();
                        cell.capture->launch(cell.stream.stream);
                        cell.dense->launch(cell.stream.stream);
                        retirement.finish();
                        const auto shots = cell.snapshots.toHost();
                        const auto retained = cell.compact.toHost();
                        for (int selected : {16, 7, 8, 1, 0, 15, 16})
                        {
                            for (int request = 0; request < cell.requests; ++request)
                                cell.accepted_host[request] = std::min(selected, cell.live_host[request]);
                            cell.publishCounts();
                            retirement.finish();
                            const auto expected_state = cell.selectedState(shots, cell.accepted_host);
                            for (std::size_t offset = 0; offset < publications.size(); ++offset)
                            {
                                endpoint.fill(Cell::Poison);
                                publications[offset]->launch(cell.stream.stream);
                                retirement.finish();
                                std::vector<float> expected(endpoint.count, Cell::Poison);
                                std::copy_n(expected_state.begin(), cell.geometry.denseBankFloats(),
                                    expected.begin() + Cell::Guard + offset);
                                equal(endpoint.toHost(), expected,
                                    "Dense accepted endpoint alignment/state/guard differs");
                            }
                        }
                        equal(cell.compact.toHost(), retained,
                            "Dense accepted endpoint changed immutable retained source");
                    }
                }
}

namespace
{
namespace cuda_dense_compact_test
{
/** @brief Explicit timing boundary; existing pre-expanded results keep their scope. */
enum class CompleteEventInputBoundary { PreExpandedPlanarControl, PostConvolutionMerged };

/** @brief Existing native deinterleave over cold-owned, initially poisoned planar scratch.
 * @details Host-expanded Cell arrays are an independent oracle only. This scratch
 * is never populated with their data. Every measured legacy transaction launches
 * the actual production native copy using the same immutable device row owner.
 */
class EventPlanarPreparation final
{
  private:
    Cell &cell_; ///< Existing original raw operands and count authority.
    int ordinal_; ///< Actual selected participant; no implicit device choice.

    /** @brief Reuse the production checked physical layout extent before allocation.
     * @param cell Existing admitted immutable geometry.
     * @return Complete scratch plus the existing trailing test guard.
     * @throws std::runtime_error When the native geometry is not representable. */
    static std::size_t elements(const Cell &cell)
    {
        const int per_request = gdnDeinterleaveElementsPerRequest(cell.input().sources.rows,
            cell.keys, cell.heads, cell.dk, cell.dv);
        require(per_request > 0, "Post-convolution preparation geometry invalid");
        return std::size_t(per_request) * cell.requests + Cell::Guard;
    }

  public:
    CudaFloatBuffer storage; ///< One fixed native output allocation, never host-expanded input.

    /** @brief Allocate fixed scratch before capture/measurement.
     * @param cell Existing original input/geometry/count owners.
     * @param ordinal Actual selected native device. */
    EventPlanarPreparation(Cell &cell, int ordinal)
        : cell_(cell), ordinal_(ordinal), storage(elements(cell), Cell::Poison) {}
    /** @return Physical Q plane base matching the production workspace layout. */
    float *query() const { return storage.ptr; }
    /** @return Physical K plane base; live counts never move its origin. */
    float *key() const { return query() + std::size_t(cell_.requests) * cell_.rows * cell_.heads * cell_.dk; }
    /** @return Physical V plane base; inactive rows keep their admitted stride. */
    float *value() const { return key() + std::size_t(cell_.requests) * cell_.rows * cell_.heads * cell_.dk; }
    /** @brief Poison scratch outside all event intervals, never repair it with host operands. */
    void poison() { storage.fill(Cell::Poison); }
    /** @brief Enqueue the actual original device-only deinterleave.
     * @return Native admission/enqueue result on the existing exact stream. */
    bool enqueue() const
    {
        return cudaGDN_deinterleave_qkv(cell_.merged.ptr, query(), key(), value(),
            cell_.input().sources.rows, cell_.keys, cell_.heads, cell_.dk, cell_.dv,
            cell_.offset, ordinal_, cell_.stream.stream);
    }
    /** @return Existing original planar recurrence tuple consuming ONLY native scratch. */
    PlanarRecurrenceInput input() const
    {
        auto result = cell_.denseInput();
        result.query = query(); result.key = key(); result.value = value();
        return result;
    }
    /** @brief Check every actual live copied word, inactive hole and guard after completion. */
    void checkPrepared() const
    {
        std::vector<float> expected(storage.count, Cell::Poison);
        const auto q_plane = std::size_t(cell_.requests) * cell_.rows * cell_.heads * cell_.dk;
        for (int request = 0; request < cell_.requests; ++request)
            for (int row = 0; row < cell_.live_host[request]; ++row)
            {
                const auto flat = std::size_t(request * cell_.rows + row);
                std::copy_n(cell_.q_host.data() + flat * cell_.heads * cell_.dk,
                    cell_.heads * cell_.dk, expected.data() + flat * cell_.heads * cell_.dk);
                std::copy_n(cell_.k_host.data() + flat * cell_.heads * cell_.dk,
                    cell_.heads * cell_.dk, expected.data() + q_plane + flat * cell_.heads * cell_.dk);
                std::copy_n(cell_.v_host.data() + flat * cell_.heads * cell_.dv,
                    cell_.heads * cell_.dv, expected.data() + 2 * q_plane + flat * cell_.heads * cell_.dv);
            }
        equal(storage.toHost(), expected, "Actual native deinterleave/live holes/guard differs");
    }
    /** @brief Prove the direct-merged candidate never used the private legacy scratch. */
    void checkUntouched() const
    { equal(storage.toHost(), std::vector<float>(storage.count, Cell::Poison), "Direct merged candidate touched preparation scratch"); }
};

/** @brief Enqueue the actual following native M1, consuming the arm's current live bank.
 * @param cell Existing raw operands and independent complete bank owners.
 * @param state Actual evolving arm bank, never replaced by oracle data.
 * @param output Actual following-row destination with its existing guard.
 * @param boundary Existing planar scope or explicit post-convolution raw merged M1. */
inline bool eventFollowing(Cell &cell, float *state, float *output,
    CompleteEventInputBoundary boundary = CompleteEventInputBoundary::PreExpandedPlanarControl)
{
    const auto row = std::size_t(cell.rows);
    if (boundary == CompleteEventInputBoundary::PostConvolutionMerged)
        return launchCUDAMergedRecurrence({DeviceRequestRowRanges::fullyActive(1),
            cell.keys, cell.heads, cell.dk, cell.dv, cell.offset, cell.normalize,
            cell.merged.ptr + row * cell.merged_stride,
            cell.alpha.ptr + row * cell.heads, cell.beta.ptr + row * cell.heads,
            cell.a.ptr, cell.bias.ptr, output, state, state, nullptr, 0, 0, cell.axes}, cell.stream.stream);
    return launchCUDAPlanarRecurrence({DeviceRequestRowRanges::fullyActive(1),
        cell.heads, cell.dk, cell.dv, cell.normalize,
        cell.query.ptr + row * cell.heads * cell.dk,
        cell.key.ptr + row * cell.heads * cell.dk,
        cell.value.ptr + row * cell.heads * cell.dv,
        cell.alpha.ptr + row * cell.heads, cell.beta.ptr + row * cell.heads,
        cell.a.ptr, cell.bias.ptr, output, state, state, nullptr, 0, 0, cell.axes}, cell.stream.stream);
}

/** @brief Select the accepted native snapshot with the original production copy kernel.
 * @details accepted0 leaves the unchanged input bank untouched. Full-row snapshot
 * capture never writes that bank, so this exactly matches compact cp0 restoration.
 * @param cell Existing counted acceptance and native full-snapshot storage.
 * @param state Arm's live bank at this transaction's actual initial state.
 * @param ordinal Authored device; never selects an implicit/default stream. */
inline bool eventAcceptedCopy(Cell &cell, float *state, int ordinal)
{
    return cudaGDN_gpu_copy_capture_terminal_rows_from_device_lengths(
        state, cell.snapshots.ptr, cell.accepted.ptr, 1, cell.rows, cell.rows,
        cell.state_stride, ordinal, cell.stream.stream);
}

/** @return One actual kernel's authenticated captured launch for schedule comparison.
 * @param graph Existing one-operation capture with typed inputs/owned lifetime. */
inline cudaKernelNodeParams eventLeaf(const CudaCapturedGraph &graph)
{
    std::size_t count = 0;
    checkCuda(cudaGraphGetNodes(graph.graph, nullptr, &count), "dense events leaf count");
    require(count == 1, "Dense events leaf is not one native kernel");
    cudaGraphNode_t node{};
    checkCuda(cudaGraphGetNodes(graph.graph, &node, &count), "dense events leaf");
    cudaGraphNodeType kind{};
    checkCuda(cudaGraphNodeGetType(node, &kind), "dense events leaf kind");
    require(kind == cudaGraphNodeTypeKernel, "Dense events leaf contains nonkernel work");
    cudaKernelNodeParams result{};
    checkCuda(cudaGraphKernelNodeGetParams(node, &result), "dense events leaf args");
    return result;
}

/** @brief Closed test-only choice; canonical graphs never contain phase events. */
enum class CompleteEventInstrumentation { Canonical, PhaseEventNodes };

/** @brief Cold-owned timestamp nodes for the same64 evolving transactions.
 * @details Event objects precede their borrowing captures in local lifetime.
 * The existing terminal retirement guard joins replay before graphs and events
 * are destroyed. No allocation, host callback or extra kernel enters replay.
 */
class CompletePhaseEvents final
{
  public:
    static constexpr int Repeats = 64;
    static constexpr int Boundaries = 3 * Repeats + 1;

    /** @brief Allocate both arms' timing-enabled event handles before capture.
     * @throws std::runtime_error When a CUDA event cannot be created. */
    CompletePhaseEvents()
    {
        try
        {
            for (auto &arm : events_)
                for (auto &event : arm)
                    checkCuda(cudaEventCreateWithFlags(&event, cudaEventDefault),
                        "Dense phase event create");
        }
        catch (...)
        {
            destroy();
            throw;
        }
    }
    /** @brief Destroy observations after borrowing captures and joined replays retire. */
    ~CompletePhaseEvents() { destroy(); }
    CompletePhaseEvents(const CompletePhaseEvents &) = delete;
    CompletePhaseEvents &operator=(const CompletePhaseEvents &) = delete;

    /** @brief Return one exact handle for native captured-node authentication.
     * @param arm Legacy0 or compact1, owning separate timestamp objects.
     * @param boundary Initial0 or the boundary following each original kernel.
     * @return Existing non-null event owned by this observer. */
    cudaEvent_t event(int arm, int boundary) const
    {
        require(arm >= 0 && arm < 2 && boundary >= 0 && boundary < Boundaries,
            "Dense phase event coordinate invalid");
        return events_[arm][boundary];
    }

    /** @brief Add a real timing-event node during cold stream capture only.
     * @param arm Exact graph whose events are being recorded.
     * @param boundary Original ordered kernel boundary0..192.
     * @param stream Exact non-default stream borrowed by that captured graph.
     * @details External recording explicitly creates an event record node;
     * ordinary capture dependency events cannot stand in for timestamps. */
    void record(int arm, int boundary, cudaStream_t stream) const
    {
        require(reinterpret_cast<std::uintptr_t>(stream) > 2,
            "Dense phase event requires an exact stream");
        checkCuda(cudaEventRecordWithFlags(event(arm, boundary), stream, cudaEventRecordExternal),
            "Dense captured phase event record");
    }

    /** @brief Read actual phase means after the existing terminal observer joined.
     * @param arm Completed graph whose retained timestamps are being inspected.
     * @return Microseconds per transaction for capture, publication, following row.
     * @details Timestamp nodes perturb scheduling. These observations are
     * instrumented diagnostics, never substitutes for canonical total timings.
     * No synchronization is added: not-ready or invalid observations fail. */
    std::array<double, 3> means(int arm) const
    {
        std::array<double, 3> result{};
        for (int transaction = 0; transaction < Repeats; ++transaction)
            for (int phase = 0; phase < 3; ++phase)
            {
                const int begin = 3 * transaction + phase;
                float milliseconds = 0.0f;
                checkCuda(cudaEventElapsedTime(&milliseconds, event(arm, begin), event(arm, begin + 1)),
                    "Dense captured phase elapsed time");
                require(std::isfinite(milliseconds) && milliseconds >= 0.0f,
                    "Dense phase elapsed time invalid");
                result[phase] += 1000.0 * static_cast<double>(milliseconds) / Repeats;
            }
        return result;
    }

  private:
    /** @brief Retire exactly the already-created handles, including partial construction. */
    void destroy() noexcept
    {
        for (auto &arm : events_)
            for (auto &event : arm)
                if (event)
                {
                    (void)cudaEventDestroy(event);
                    event = nullptr;
                }
    }
    std::array<std::array<cudaEvent_t, Boundaries>, 2> events_{}; ///< One authority per timestamp.
};

/** @brief Report native static residency capacity for the actual diagnostic leaves.
 * @param cell Existing owned shape/live metadata and exact-stream captures.
 * @param leaves Actual capture, publication and following-row kernel parameters.
 * @param arm Legacy0 or candidate1, matching the instrumented event bank.
 * @details This is a cold runtime resource query, not a profiler or achieved
 * occupancy measurement. It executes no GPU work and enters no event interval. */
inline void reportPhaseStaticCapacity(const Cell &cell,
    const std::array<cudaKernelNodeParams, 3> &leaves, int arm)
{
    int device = -1;
    checkCuda(cudaGetDevice(&device), "Dense phase capacity device");
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, device), "Dense phase capacity properties");
    const std::array<const char *, 3> roles{"capture", "publication", "following"};
    for (std::size_t phase = 0; phase < leaves.size(); ++phase)
    {
        const auto &leaf = leaves[phase];
        const auto threads = std::size_t(leaf.blockDim.x) * leaf.blockDim.y * leaf.blockDim.z;
        require(threads > 0 && threads <= std::size_t(std::numeric_limits<int>::max()),
            "Dense phase capacity block geometry invalid");
        int active_blocks = 0;
        checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
            leaf.func, int(threads), leaf.sharedMemBytes), "Dense phase static block capacity");
        cudaFuncAttributes attributes{};
        checkCuda(cudaFuncGetAttributes(&attributes, leaf.func), "Dense phase actual function resources");
        const char *name = nullptr;
        checkCuda(cudaFuncGetName(&name, leaf.func), "Dense phase capacity function name");
        require(name && active_blocks > 0 && properties.multiProcessorCount > 0,
            "Dense phase static capacity unavailable");
        std::cout << "CUDA_GDN_DENSE_COMPACT_INSTRUMENTED_CAPACITY heads=" << cell.heads
            << " M=" << cell.rows << " live_rows=" << cell.live_host[0]
            << " accepted=" << cell.accepted_host[0] << " arm=" << arm << " phase=" << roles[phase]
            << " device=" << device << " multiprocessors=" << properties.multiProcessorCount
            << " grid_x=" << leaf.gridDim.x << " grid_y=" << leaf.gridDim.y << " grid_z=" << leaf.gridDim.z
            << " threads=" << threads << " dynamic_shared_bytes=" << leaf.sharedMemBytes
            << " static_shared_bytes=" << attributes.sharedSizeBytes << " local_bytes=" << attributes.localSizeBytes
            << " registers_per_thread=" << attributes.numRegs << " capacity_blocks_per_sm=" << active_blocks
            << " instrumented=1 static_capacity_only=1 achieved_occupancy_claim=0 function=" << name << '\n';
    }
}

/** @brief Authenticate the actual preparation leaf's complete production tuple.
 * @param cell Existing original source and count geometry.
 * @param preparation Fixed scratch whose pointer layout the native stage consumes.
 * @param graph Actual one-node production deinterleave capture. */
inline void inspectEventPreparation(const Cell &cell, const EventPlanarPreparation &preparation,
    const CudaCapturedGraph &graph)
{
    const auto p = eventLeaf(graph);
    const char *name = nullptr;
    checkCuda(cudaFuncGetName(&name, p.func), "Post-convolution prep native name");
    require(name && std::string(name).find("cuda_gdn_deinterleave_qkv_kernel") != std::string::npos,
        "Post-convolution preparation selected foreign function");
    const auto rows = cell.input().sources.rows;
    const int elements = gdnDeinterleaveElementsPerRequest(rows, cell.keys, cell.heads, cell.dk, cell.dv);
    require(p.gridDim.x == unsigned(1 + (elements - 1) / 256) && p.gridDim.y == unsigned(cell.requests) &&
        p.gridDim.z == 1 && p.blockDim.x == 256 && p.blockDim.y == 1 && p.blockDim.z == 1 && p.sharedMemBytes == 0,
        "Post-convolution preparation geometry differs");
    const void *pointers[]{cell.merged.ptr, preparation.query(), preparation.key(), preparation.value()};
    require(p.kernelParams != nullptr, "Post-convolution preparation arguments absent");
    for (int arg = 0; arg < 4; ++arg)
    {
        const void *actual = nullptr;
        require(p.kernelParams[arg] != nullptr, "Post-convolution preparation pointer absent");
        std::memcpy(&actual, p.kernelParams[arg], sizeof(actual));
        require(actual == pointers[arg], "Post-convolution preparation pointer differs");
    }
    auto actual_rows = rows;
    require(p.kernelParams[4] != nullptr, "Post-convolution preparation row owner absent");
    std::memcpy(&actual_rows, p.kernelParams[4], sizeof(actual_rows));
    require(actual_rows.requests() == rows.requests() && actual_rows.rowsPerRequest() == rows.rowsPerRequest() &&
        actual_rows.lengthOwner() == rows.lengthOwner(), "Post-convolution preparation row authority differs");
    const int scalars[]{cell.keys, cell.heads, cell.dk, cell.dv, cell.offset};
    for (int arg = 5; arg < 10; ++arg)
    {
        int actual = 0;
        require(p.kernelParams[arg] != nullptr, "Post-convolution preparation scalar absent");
        std::memcpy(&actual, p.kernelParams[arg], sizeof(actual));
        require(actual == scalars[arg - 5], "Post-convolution preparation head mapping differs");
    }
}

/** @brief Authenticate exact ordered repeated native functions and launch geometries.
 * @param cell Existing correctly typed capture seed and bank owners.
 * @param graph Actual event graph, not a separately constructed timing surrogate.
 * @param leaves Ordered transaction capture, acceptance and next-row native launches.
 * @param repeats Independent authored transaction count.
 * @param preparation Optional actual native deinterleave leaf, preceding legacy capture only.
 * @param phases Optional preallocated diagnostic timestamp owner; null keeps the canonical graph.
 * @param arm Exact timestamp bank for this graph, ignored without instrumentation.
 * @details Native arguments retain the same physical banks on every iteration;
 * pointer/coordinate values are compared against the corresponding owned leaf.
 */
inline void inspectEventSchedule(const Cell &cell, const CudaCapturedGraph &graph,
    const std::array<cudaKernelNodeParams, 3> &leaves, int repeats,
    const CompletePhaseEvents *phases = nullptr, int arm = 0,
    const cudaKernelNodeParams *preparation = nullptr)
{
    std::size_t count = 0, roots = 0;
    checkCuda(cudaGraphGetNodes(graph.graph, nullptr, &count), "dense events schedule count");
    checkCuda(cudaGraphGetRootNodes(graph.graph, nullptr, &roots), "dense events roots");
    require(!phases || !preparation, "Instrumented phases and post-convolution preparation are separate scopes");
    const int roles = preparation ? 4 : 3;
    const int expected_nodes = phases ? 6 * repeats + 1 : roles * repeats;
    require(!phases || repeats == CompletePhaseEvents::Repeats, "Dense phase repeat count differs");
    require(count == std::size_t(expected_nodes) && roots == 1,
        "Dense events complete transaction count/root differs");
    cudaGraphNode_t node{};
    checkCuda(cudaGraphGetRootNodes(graph.graph, &node, &roots), "dense events first node");
    for (int index = 0; index < expected_nodes; ++index)
    {
        cudaGraphNodeType kind{};
        checkCuda(cudaGraphNodeGetType(node, &kind), "dense events node type");
        if (phases && index % 2 == 0)
        {
            require(kind == cudaGraphNodeTypeEventRecord, "Dense phase boundary is not an event record node");
            cudaEvent_t actual_event = nullptr;
            checkCuda(cudaGraphEventRecordNodeGetEvent(node, &actual_event), "Dense phase node handle");
            require(actual_event == phases->event(arm, index / 2), "Dense phase event owner/order differs");
        }
        else
        {
        require(kind == cudaGraphNodeTypeKernel, "Dense events graph contains nonkernel work");
        cudaKernelNodeParams actual{};
        checkCuda(cudaGraphKernelNodeGetParams(node, &actual), "dense events node params");
        const int role = (phases ? index / 2 : index) % roles;
        const auto &expected = preparation && role == 0 ? *preparation : leaves[role - (preparation ? 1 : 0)];
        require(actual.func == expected.func && actual.gridDim.x == expected.gridDim.x &&
            actual.gridDim.y == expected.gridDim.y && actual.gridDim.z == expected.gridDim.z &&
            actual.blockDim.x == expected.blockDim.x && actual.blockDim.y == expected.blockDim.y &&
            actual.blockDim.z == expected.blockDim.z && actual.sharedMemBytes == expected.sharedMemBytes &&
            actual.kernelParams && expected.kernelParams,
            "Dense events ordered native function/geometry differs");
        const char *name = nullptr;
        checkCuda(cudaFuncGetName(&name, actual.func), "dense events function name");
        require(name != nullptr, "Dense events native function identity absent");
        const std::string family(name);
        if (family.find("cuda_gdn_deinterleave_qkv_kernel") != std::string::npos)
        {
            require(preparation && role == 0, "Unexpected preparation in retained transaction");
            for (int arg = 0; arg < 4; ++arg)
            {
                const void *a = nullptr, *b = nullptr;
                require(actual.kernelParams[arg] && expected.kernelParams[arg], "Native preparation pointer absent");
                std::memcpy(&a, actual.kernelParams[arg], sizeof(a));
                std::memcpy(&b, expected.kernelParams[arg], sizeof(b));
                require(a == b, "Native preparation scratch/source changed");
            }
            auto a = cell.input().sources.rows, b = a;
            require(actual.kernelParams[4] && expected.kernelParams[4], "Native preparation count owner absent");
            std::memcpy(&a, actual.kernelParams[4], sizeof(a));
            std::memcpy(&b, expected.kernelParams[4], sizeof(b));
            require(a.requests() == b.requests() && a.rowsPerRequest() == b.rowsPerRequest() &&
                a.lengthOwner() == b.lengthOwner(), "Native preparation row authority changed");
            for (int arg = 5; arg < 10; ++arg)
            {
                int a = 0, b = 0;
                require(actual.kernelParams[arg] && expected.kernelParams[arg], "Native preparation mapping absent");
                std::memcpy(&a, actual.kernelParams[arg], sizeof(a));
                std::memcpy(&b, expected.kernelParams[arg], sizeof(b));
                require(a == b, "Native preparation head mapping changed");
            }
        }
        else if (family.find("dense_compact_capture_kernel") != std::string::npos ||
            family.find("retained_prefixes_capture_kernel") != std::string::npos ||
            family.find("dense_compact_packed_capture_kernel") != std::string::npos ||
            family.find("retained_prefixes_packed_capture_kernel") != std::string::npos)
        {
            require(actual.kernelParams[0] && expected.kernelParams[0], "Dense events compact argument absent");
            auto a = cell.input();
            auto b = a;
            std::memcpy(&a, actual.kernelParams[0], sizeof(a));
            std::memcpy(&b, expected.kernelParams[0], sizeof(b));
            require(a.transaction.initial_state == b.transaction.initial_state &&
                a.transaction.storage == b.transaction.storage &&
                a.terminal.destination() == b.terminal.destination() && a.terminal.kind() == b.terminal.kind() &&
                a.sources.output == b.sources.output && a.sources.merged == b.sources.merged &&
                a.sources.rows.lengthOwner() == b.sources.rows.lengthOwner() &&
                a.transaction.generation == b.transaction.generation &&
                a.transaction.layout == b.transaction.layout &&
                a.transaction.encoding == b.transaction.encoding &&
                a.transaction.geometry.retentionProgram().mode == b.transaction.geometry.retentionProgram().mode &&
                a.transaction.geometry.retentionProgram().compact_min_live_rows == b.transaction.geometry.retentionProgram().compact_min_live_rows &&
                a.transaction.geometry.rowsPerRequest() == b.transaction.geometry.rowsPerRequest(),
                "Dense events compact capture banks differ");
        }
        else if (family.find("dense_compact_accepted_prefix_kernel") != std::string::npos)
        {
            require(actual.kernelParams[0] && expected.kernelParams[0], "Dense events publication argument absent");
            GDNDenseCompactAcceptedPrefixInput a{cell.transaction(), cell.acceptedSelection(), cell.initial.ptr};
            auto b = a;
            std::memcpy(&a, actual.kernelParams[0], sizeof(a));
            std::memcpy(&b, expected.kernelParams[0], sizeof(b));
            require(a.transaction.storage == b.transaction.storage && a.destination_state == b.destination_state &&
                a.selection.sameBinding(b.selection) && a.transaction.generation == b.transaction.generation &&
                a.transaction.layout == b.transaction.layout &&
                a.transaction.encoding == b.transaction.encoding &&
                a.transaction.geometry.retentionProgram().mode == b.transaction.geometry.retentionProgram().mode &&
                a.transaction.geometry.retentionProgram().compact_min_live_rows == b.transaction.geometry.retentionProgram().compact_min_live_rows,
                "Dense events compact publication banks differ");
        }
        else
        {
            // Authenticate known legacy families BEFORE reading their pointer tuple.
            const bool copy = family.find("cuda_gdn_copy_capture_rows_from_device_indices_kernel") != std::string::npos;
            const bool merged = family.find("cuda_gdn_merged_recurrent_step_kernel") != std::string::npos ||
                family.find("cuda_gdn_value_major_merged_recurrent_step_kernel") != std::string::npos;
            const bool dense = merged || family.find("cuda_gdn_recurrent_step_kernel") != std::string::npos ||
                family.find("cuda_gdn_value_major_recurrent_step_kernel") != std::string::npos;
            require(copy || dense, "Dense events selected an unknown legacy ABI");
            for (int arg = 0; arg < (copy ? 3 : 10); ++arg)
            {
                require(actual.kernelParams[arg] && expected.kernelParams[arg], "Dense events legacy argument missing");
                const void *a = nullptr, *b = nullptr;
                std::memcpy(&a, actual.kernelParams[arg], sizeof(a));
                std::memcpy(&b, expected.kernelParams[arg], sizeof(b));
                require(a == b, "Dense events legacy native endpoint differs");
            }
            const auto same_argument = [&](int argument, std::size_t bytes) {
                require(actual.kernelParams[argument] && expected.kernelParams[argument],
                    "Dense events legacy scalar argument absent");
                require(std::memcmp(actual.kernelParams[argument], expected.kernelParams[argument], bytes) == 0,
                    "Dense events legacy scalar policy/coordinate differs");
            };
            if (copy)
            {
                for (int argument = 3; argument <= 7; ++argument) same_argument(argument, sizeof(int));
            }
            else
            {
                for (int argument = 10; argument <= 13; ++argument) same_argument(argument, sizeof(int));
                same_argument(14, sizeof(bool));
                same_argument(15, sizeof(const int *));
                same_argument(16, sizeof(int));
                same_argument(17, sizeof(float *));
                same_argument(18, sizeof(int));
                same_argument(19, sizeof(int));
                if (merged) { same_argument(20, sizeof(int)); same_argument(21, sizeof(int)); }
            }
        }
        }
        std::size_t successors = 0;
        checkCuda(cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, &successors), "dense events successor count");
        require(successors == unsigned(index + 1 < expected_nodes), "Dense events graph is not an exact linear chain");
        if (successors)
        {
            cudaGraphNode_t next{};
            checkCuda(cudaGraphNodeGetDependentNodes(node, &next, nullptr, &successors), "dense events next node");
            node = next;
        }
    }
}

/** @brief Build an untimed independent serial-row native oracle of the evolving transaction.
 * @param oracle Separate owners; none can repair or replace either timed arm's banks.
 * @param repeats Exact same number of accepted blocks and following rows.
 * @param ordinal Authored device for the existing native accepted-row copy.
 * @param live_rows Active prefix or -1 for the original fully-live schedule.
 * @details D2D copies retain each native M1 post-row state for the old accepted
 * copy. These diagnostic copies/serial launches are never in either timed arm.
 */
inline void runEventOracle(Cell &oracle, int repeats, int ordinal, int live_rows = -1)
{
    const int live = live_rows < 0 ? oracle.rows : live_rows;
    require(live >= 0 && live <= oracle.rows, "Dense event oracle live extent invalid");
    for (int repeat = 0; repeat < repeats; ++repeat)
    {
        if (live < oracle.rows)
            checkCuda(cudaMemsetAsync(oracle.output.ptr, 0,
                oracle.geometry.outputFloats() * sizeof(float), oracle.stream.stream),
                "Dense event oracle inactive output zero");
        checkCuda(cudaMemcpyAsync(oracle.oracle_state.ptr, oracle.initial.ptr,
            oracle.geometry.denseBankBytes(), cudaMemcpyDeviceToDevice, oracle.stream.stream), "dense events oracle initial");
        for (int row = 0; row < live; ++row)
        {
            const auto index = std::size_t(row);
            require(launchCUDAPlanarRecurrence({DeviceRequestRowRanges::fullyActive(1),
                oracle.heads, oracle.dk, oracle.dv, oracle.normalize,
                oracle.query.ptr + index * oracle.heads * oracle.dk,
                oracle.key.ptr + index * oracle.heads * oracle.dk,
                oracle.value.ptr + index * oracle.heads * oracle.dv,
                oracle.alpha.ptr + index * oracle.heads, oracle.beta.ptr + index * oracle.heads,
                oracle.a.ptr, oracle.bias.ptr, oracle.output.ptr + index * oracle.heads * oracle.dv,
                oracle.oracle_state.ptr, oracle.oracle_state.ptr, nullptr, 0, 0, oracle.axes}, oracle.stream.stream),
                "Dense events original serial row rejected");
            checkCuda(cudaMemcpyAsync(oracle.snapshots.ptr + index * oracle.state_stride, oracle.oracle_state.ptr,
                oracle.geometry.denseBankBytes(), cudaMemcpyDeviceToDevice, oracle.stream.stream), "dense events oracle snapshot");
        }
        require(eventAcceptedCopy(oracle, oracle.initial.ptr, ordinal), "Dense events oracle accepted copy rejected");
        require(eventFollowing(oracle, oracle.initial.ptr, oracle.following.ptr), "Dense events oracle following rejected");
    }
    oracle.observation.finish();
}

/** @brief Closed test-only acceptance schedules; neither is a production policy. */
enum class EventAcceptanceSpectrum
{
    Representative, ///< Original zero/middle/full24-cell screen, unchanged marker schema.
    EveryPrefixM8M16, ///< Explicit A0..M52-cell screen including worst suffix7.
    PhysicalM16LiveCounts, ///< Physical16 with independent live prefix; no duplicate live16 timings.
    ProductionH32M4M16EveryPrefix ///< Separate full-live production-shape spectrum; original schedules unchanged.
};

/** @brief Poison only inactive verifier operands, leaving the actual following row finite.
 * @param cell Existing exact physical-M16 owners and host source staging.
 * @param live Active prefix; physical row16 belongs to the next-row oracle.
 * @details NaNs make accidental inactive-row arithmetic observable without changing
 * any production liveness/count rule. Source bytes are checked after every arm.
 */
inline void poisonInactiveEventRows(Cell &cell, int live)
{
    require(cell.requests == 1 && live >= 0 && live <= cell.rows,
        "Inactive event poison requires one valid request");
    const float poison = std::bit_cast<float>(std::uint32_t{0x7fc12345});
    for (int row = live; row < cell.rows; ++row)
    {
        std::fill_n(cell.merged_host.begin() + std::size_t(row) * cell.merged_stride,
            cell.merged_stride, poison);
        std::fill_n(cell.q_host.begin() + std::size_t(row) * cell.heads * cell.dk,
            cell.heads * cell.dk, poison);
        std::fill_n(cell.k_host.begin() + std::size_t(row) * cell.heads * cell.dk,
            cell.heads * cell.dk, poison);
        std::fill_n(cell.v_host.begin() + std::size_t(row) * cell.heads * cell.dv,
            cell.heads * cell.dv, poison);
        std::fill_n(cell.alpha_host.begin() + std::size_t(row) * cell.heads, cell.heads, poison);
        std::fill_n(cell.beta_host.begin() + std::size_t(row) * cell.heads, cell.heads, poison);
    }
    cell.merged.copyFrom(cell.merged_host); cell.query.copyFrom(cell.q_host);
    cell.key.copyFrom(cell.k_host); cell.value.copyFrom(cell.v_host);
    cell.alpha.copyFrom(cell.alpha_host); cell.beta.copyFrom(cell.beta_host);
}

/** @brief Preserve existing schemas while identifying only the new live-prefix schedule.
 * @param spectrum Closed test-only schedule, not a production dispatch mode.
 * @param live Actual authored live prefix for this retained device-count owner.
 * @return Exact extra marker fields; original representative schema stays empty.
 */
inline std::string eventSpectrumFields(EventAcceptanceSpectrum spectrum, int live)
{
    if (spectrum == EventAcceptanceSpectrum::EveryPrefixM8M16)
        return " acceptance_spectrum=every_prefix_m8_m16";
    if (spectrum == EventAcceptanceSpectrum::PhysicalM16LiveCounts)
        return " acceptance_spectrum=physical_m16_live_counts live_rows=" + std::to_string(live) +
            " executed_rows_per_block=" + std::to_string(live + 1) + " inactive_source=nan_poison";
    if (spectrum == EventAcceptanceSpectrum::ProductionH32M4M16EveryPrefix)
        return " acceptance_spectrum=production_h32_m4_m16_every_prefix live_rows=" + std::to_string(live) +
            " executed_rows_per_block=" + std::to_string(live + 1) + " inactive_source=none";
    return "";
}

/** @brief Time one complete evolving-state cell with actual final bytes checked before reset.
 * @param ordinal Actual fixture-owned CUDA participant.
 * @param heads Authored local value heads; existing schedules12/24, production-shaped schedule32.
 * @param rows Physical verifier capacity, independent of accepted rows.
 * @param accepted Device-authored test count A0..M, immutable within this captured graph.
 * @param selection Original role ordinal, or A itself in the explicit every-prefix spectrum.
 * @param spectrum Exact test-only marker identity, not a serving selector.
 * @param live_rows Active prefix, or -1 to preserve original fully-live schedule.
 * @param program Explicit candidate retention; default preserves all existing filters/schemas.
 * @param terminal_kind Explicit candidate terminal contract; legacy filters retain DenseDestination.
 * @param axes Explicit captured state axes; old filters preserve their ValueMajor program.
 * @param instrumentation Closed diagnostic event-node choice; canonical remains unchanged.
 * @param boundary Explicit source boundary; old calls retain pre-expanded planar controls.
 * @param source_key_heads Explicit positive source KH for a production-shaped test, or0 to preserve heads/3.
 * @details Both arms include64 repeated verifier/publication/following transactions.
 * Existing stream, byte, graph, terminal and original serial-oracle proofs remain identical.
 * Planar full-snapshot versus merged exact-compact source scope remains explicit.
 */
inline void runCompleteEventCell(int ordinal, int heads, int rows, int accepted, int selection,
    EventAcceptanceSpectrum spectrum, int live_rows = -1, GDNDenseRetentionProgram program = {},
    GDNDenseCompactTerminalKind terminal_kind = GDNDenseCompactTerminalKind::DenseDestination,
    Layout axes = Layout::ValueMajorKeyContiguous,
    CompleteEventInstrumentation instrumentation = CompleteEventInstrumentation::Canonical,
    CompleteEventInputBoundary boundary = CompleteEventInputBoundary::PreExpandedPlanarControl,
    int source_key_heads = 0)
{
    constexpr int Repeats = 64, Warmups = 5, Samples = 12;
    require(boundary == CompleteEventInputBoundary::PreExpandedPlanarControl ||
        boundary == CompleteEventInputBoundary::PostConvolutionMerged, "Complete event input boundary invalid");
    const bool postconv = boundary == CompleteEventInputBoundary::PostConvolutionMerged;
    const int source_offset = postconv ? 0 : -5;
    const int keys = source_key_heads == 0 ? heads / 3 : source_key_heads;
    require(keys > 0, "Complete event source KH must be positive");
    Cell cell(heads, keys, 128, 128, 1, rows, axes, true, false, false, program, terminal_kind, source_offset);
    Cell oracle(heads, keys, 128, 128, 1, rows, axes, true, false, false, {}, GDNDenseCompactTerminalKind::DenseDestination, source_offset);
    const int live = live_rows < 0 ? rows : live_rows;
    require(live >= 0 && live <= rows && accepted >= 0 && accepted <= live,
        "Dense event live/accepted extent invalid");
    if (spectrum == EventAcceptanceSpectrum::PhysicalM16LiveCounts)
    {
        poisonInactiveEventRows(cell, live);
        poisonInactiveEventRows(oracle, live);
    }
    cell.live_host[0] = live; cell.accepted_host[0] = accepted;
    oracle.live_host[0] = live; oracle.accepted_host[0] = accepted;
    cell.publishCounts(); oracle.publishCounts();
    cell.observation.finish(); oracle.observation.finish();
    runEventOracle(oracle, Repeats, ordinal, live);
    const auto expected_state = oracle.initial.toHost();
    const auto expected_group_output = oracle.output.toHost();
    const auto expected_following = oracle.following.toHost();
    const auto expected_full_terminal = oracle.oracle_state.toHost();
    const auto expected_snapshots = oracle.snapshots.toHost();
    std::unique_ptr<EventPlanarPreparation> preparation;
    if (postconv) preparation = std::make_unique<EventPlanarPreparation>(cell, ordinal);
    const bool production_h32 = spectrum == EventAcceptanceSpectrum::ProductionH32M4M16EveryPrefix;
    if (production_h32)
        require(heads == 32 && (rows == 4 || rows == 16) && live == rows && source_key_heads > 0 &&
            postconv && axes == Layout::KeyMajorValueContiguous &&
            program.mode == GDNDenseRetentionMode::ExactCorrections &&
            terminal_kind == GDNDenseCompactTerminalKind::RetainedPrefixesOnly &&
            instrumentation == CompleteEventInstrumentation::Canonical,
            "Production H32 events require the exact full-live KeyMajor post-convolution contract");
    const std::string production_fields = production_h32 ?
        " requests=1 source_key_heads=" + std::to_string(keys) : "";
    const std::string operand_fields = (postconv ?
        " source_layout=merged_raw_vs_merged_raw dense_input=merged_raw compact_input=merged_raw input_layouts_differ=0 legacy_preparation=native_deinterleave candidate_preparation=direct_merged common_convolution_excluded=1 continuation=raw_merged_m1"
        " legacy_deinterleave_nodes=64 legacy_kernel_nodes=256 candidate_kernel_nodes=192 source_head_offset=0 key_repeat=modulo" :
        " source_layout=planar_full_vs_merged_compact dense_input=expanded_planar compact_input=merged_raw input_layouts_differ=1") + production_fields;
    std::unique_ptr<CompletePhaseEvents> phase_events;
    require(instrumentation == CompleteEventInstrumentation::Canonical ||
        instrumentation == CompleteEventInstrumentation::PhaseEventNodes, "Dense event instrumentation invalid");
    if (instrumentation == CompleteEventInstrumentation::PhaseEventNodes)
    {
        require(!postconv && axes == Layout::KeyMajorValueContiguous &&
            terminal_kind == GDNDenseCompactTerminalKind::RetainedPrefixesOnly,
            "Phase diagnostic requires explicit KeyMajor retained-only contract");
        phase_events = std::make_unique<CompletePhaseEvents>();
    }
    const char *marker_prefix = postconv ? "CUDA_GDN_DENSE_POST_CONV_" :
        phase_events ? "CUDA_GDN_DENSE_COMPACT_INSTRUMENTED_" : "CUDA_GDN_DENSE_COMPACT_";
    const std::string instrument_fields = phase_events ?
        " instrumented=1 canonical_timing=0 event_record_nodes=193 kernel_nodes=192" : "";
    CudaCapturedGraph dense_copy(cell.stream.stream, [&] {
        return eventAcceptedCopy(cell, cell.oracle_state.ptr, ordinal);
    });
    CudaCapturedGraph compact_next(cell.stream.stream, [&] {
        return eventFollowing(cell, cell.initial.ptr, cell.following.ptr, boundary);
    });
    CudaCapturedGraph dense_next(cell.stream.stream, [&] {
        return eventFollowing(cell, cell.oracle_state.ptr, cell.oracle_following.ptr, boundary);
    });
    std::unique_ptr<CudaCapturedGraph> preparation_leaf, prepared_dense_leaf;
    cudaKernelNodeParams preparation_params{};
    if (preparation)
    {
        preparation_leaf = std::make_unique<CudaCapturedGraph>(cell.stream.stream, [&] { return preparation->enqueue(); });
        prepared_dense_leaf = std::make_unique<CudaCapturedGraph>(cell.stream.stream, [&] {
            return launchCUDAPlanarRecurrence(preparation->input(), cell.stream.stream);
        });
        inspectEventPreparation(cell, *preparation, *preparation_leaf);
        preparation_params = eventLeaf(*preparation_leaf);
    }
    CudaCapturedGraph candidate(cell.stream.stream, [&] {
        if (phase_events) phase_events->record(1, 0, cell.stream.stream);
        for (int i = 0; i < Repeats; ++i)
        {
            if (!launchCUDADenseCompactRecurrence(cell.input(), cell.stream.stream, cell.capture_plan)) return false;
            if (phase_events) phase_events->record(1, 3 * i + 1, cell.stream.stream);
            if (!launchCUDADenseCompactAcceptedPrefix({cell.transaction(), cell.acceptedSelection(), cell.initial.ptr}, cell.stream.stream)) return false;
            if (phase_events) phase_events->record(1, 3 * i + 2, cell.stream.stream);
            if (!eventFollowing(cell, cell.initial.ptr, cell.following.ptr, boundary)) return false;
            if (phase_events) phase_events->record(1, 3 * i + 3, cell.stream.stream);
        }
        return true;
    });
    CudaCapturedGraph control(cell.stream.stream, [&] {
        if (phase_events) phase_events->record(0, 0, cell.stream.stream);
        for (int i = 0; i < Repeats; ++i)
        {
            if (preparation && !preparation->enqueue()) return false;
            if (!launchCUDAPlanarRecurrence(preparation ? preparation->input() : cell.denseInput(), cell.stream.stream)) return false;
            if (phase_events) phase_events->record(0, 3 * i + 1, cell.stream.stream);
            if (!eventAcceptedCopy(cell, cell.oracle_state.ptr, ordinal)) return false;
            if (phase_events) phase_events->record(0, 3 * i + 2, cell.stream.stream);
            if (!eventFollowing(cell, cell.oracle_state.ptr, cell.oracle_following.ptr, boundary)) return false;
            if (phase_events) phase_events->record(0, 3 * i + 3, cell.stream.stream);
        }
        return true;
    });
    inspectEventSchedule(cell, candidate, {eventLeaf(*cell.capture), eventLeaf(*cell.publication_alias), eventLeaf(compact_next)}, Repeats, phase_events.get(), 1);
    inspectEventSchedule(cell, control, {eventLeaf(prepared_dense_leaf ? *prepared_dense_leaf : *cell.dense), eventLeaf(dense_copy), eventLeaf(dense_next)}, Repeats, phase_events.get(), 0, preparation ? &preparation_params : nullptr);
    if (phase_events)
    {
        reportPhaseStaticCapacity(cell, {eventLeaf(*cell.capture), eventLeaf(*cell.publication_alias), eventLeaf(compact_next)}, 1);
        reportPhaseStaticCapacity(cell, {eventLeaf(*cell.dense), eventLeaf(dense_copy), eventLeaf(dense_next)}, 0);
    }
    CudaMergedBoundaryEvents events;
    CudaMergedReplayRetirement timing_retirement(cell.stream.stream);
    for (int sample = -Warmups; sample < Samples; ++sample)
    {
        std::array<double, 2> us{};
        std::array<std::array<double, 3>, 2> phase_us{};
        for (int order = 0; order < 2; ++order)
        {
            const int arm = (sample & 1) ? 1 - order : order;
            cell.reset();
            if (preparation) preparation->poison();
            if (sample < 0) (arm ? candidate : control).launch(cell.stream.stream);
            else us[arm] = events.measure(arm ? candidate : control, cell.stream.stream, Repeats);
            cell.observation.finish();
            equal(arm ? cell.initial.toHost() : cell.oracle_state.toHost(), expected_state,
                "Dense events actual final timed evolving state");
            equal(arm ? cell.output.toHost() : cell.oracle_output.toHost(), expected_group_output,
                "Dense events actual final timed verifier output");
            equal(arm ? cell.following.toHost() : cell.oracle_following.toHost(), expected_following,
                "Dense events actual final timed following output");
            if (arm)
            {
                if (terminal_kind == GDNDenseCompactTerminalKind::DenseDestination)
                    equal(cell.terminal.toHost(), expected_full_terminal, "Dense events required full terminal");
                else
                    equal(cell.terminal.toHost(), std::vector<float>(cell.terminal.count, Cell::Poison),
                        "Dense events retained-only unbound terminal guard changed");
                if (program.mode != GDNDenseRetentionMode::ExactCorrections ||
                    terminal_kind == GDNDenseCompactTerminalKind::RetainedPrefixesOnly) cell.checkTimedRetention(expected_snapshots);
            }
            else equal(cell.snapshots.toHost(), expected_snapshots, "Dense events actual full snapshots");
            if (preparation) { if (arm) preparation->checkUntouched(); else preparation->checkPrepared(); }
            equal(cell.merged.toHost(), cell.merged_host, "Dense events immutable merged source");
            equal(cell.query.toHost(), cell.q_host, "Dense events immutable planar query");
            equal(cell.key.toHost(), cell.k_host, "Dense events immutable planar key");
            equal(cell.value.toHost(), cell.v_host, "Dense events immutable planar value");
            equal(cell.alpha.toHost(), cell.alpha_host, "Dense events immutable alpha");
            equal(cell.beta.toHost(), cell.beta_host, "Dense events immutable beta");
            equal(cell.a.toHost(), cell.a_host, "Dense events immutable gate scale");
            equal(cell.bias.toHost(), cell.bias_host, "Dense events immutable time bias");
            if (sample >= 0 && phase_events) phase_us[arm] = phase_events->means(arm);
            if (sample >= 0)
                std::cout << marker_prefix << "TIMED_BYTES heads=" << heads << " M=" << rows
                    << " accepted=" << accepted << " selection=" << selection << eventSpectrumFields(spectrum, live) << retentionMarkerFields(program, live) << terminalMarkerFields(terminal_kind) << eventStateLayoutFields(axes, terminal_kind) << instrument_fields << " sample=" << sample
                    << " arm=" << arm << " repeats=" << Repeats
                    << operand_fields
                    << " actual_final_timed=1 next_row_in_timed_graph=1 serial_oracle=1 passed=1\n";
        }
        if (sample >= 0)
            std::cout << std::setprecision(9) << marker_prefix << "PAIR heads=" << heads
                << " DK=128 DV=128 M=" << rows << " accepted=" << accepted << " selection=" << selection << eventSpectrumFields(spectrum, live) << retentionMarkerFields(program, live) << terminalMarkerFields(terminal_kind) << eventStateLayoutFields(axes, terminal_kind) << instrument_fields
                << " sample=" << sample << " repeats=" << Repeats << " warmups=" << Warmups
                << operand_fields
                << " dense_full_snapshots_copy_next_us=" << us[0] << " compact_capture_publish_next_us=" << us[1]
                << " dense_us_per_executed_row=" << us[0] / (live + 1)
                << " compact_us_per_executed_row=" << us[1] / (live + 1) << '\n';
        if (sample >= 0 && phase_events)
            std::cout << std::setprecision(9) << marker_prefix << "PHASE_PAIR heads=" << heads
                << " DK=128 DV=128 M=" << rows << " live_rows=" << live << " accepted=" << accepted
                << " sample=" << sample << " repeats=" << Repeats << " warmups=" << Warmups
                << " state_layout=key_major_value_contiguous compact_terminal_policy=retained_prefixes_only"
                << " compact_terminal_destination_present=0 instrumented=1 canonical_timing=0"
                << " event_record_nodes=193 kernel_nodes=192 phase_mean_denominator=64"
                << retentionMarkerFields(program, live)
                << " legacy_capture_us=" << phase_us[0][0] << " legacy_publication_us=" << phase_us[0][1]
                << " legacy_following_us=" << phase_us[0][2]
                << " candidate_capture_us=" << phase_us[1][0] << " candidate_publication_us=" << phase_us[1][1]
                << " candidate_following_us=" << phase_us[1][2]
                << " legacy_instrumented_total_us=" << us[0] << " candidate_instrumented_total_us=" << us[1]
                << " actual_final_timed=1 next_row_in_timed_graph=1 serial_oracle=1 passed=1\n";
    }
    std::cout << marker_prefix << "EVENT_CELL heads=" << heads << " M=" << rows
        << " accepted=" << accepted << " selection=" << selection << eventSpectrumFields(spectrum, live) << retentionMarkerFields(program, live) << terminalMarkerFields(terminal_kind) << eventStateLayoutFields(axes, terminal_kind) << instrument_fields
        << " DK=128 DV=128 samples=" << Samples << " repeats=" << Repeats
        << " roles=2" << operand_fields
        << " strict_partial_available=" << (live > 1)
        << " compact_required_terminal=" << int(terminal_kind == GDNDenseCompactTerminalKind::DenseDestination)
        << " dense_terminal_in_snapshots=1 passed=1\n";
}

/** @brief Preserve the original24-cell schedule and output schema exactly.
 * @param ordinal Actual fixture-owned CUDA participant. */
inline void runCompleteEvents(int ordinal)
{
    for (int heads : {12, 24})
        for (int rows : {1, 3, 8, 16})
            for (int selection : {0, 1, 2})
            {
                const int accepted = selection == 0 ? 0 : selection == 1 ? std::max(1, rows / 2) : rows;
                runCompleteEventCell(ordinal, heads, rows, accepted, selection, EventAcceptanceSpectrum::Representative);
            }
}

/** @brief Cover every accepted suffix, including seven updates before each checkpoint.
 * @param ordinal Actual fixture-owned CUDA participant.
 * @details H12/H24 x M8/M16 x A0..M yields52 independent labelled cells. Each
 * cell uses the same five warmups,12 alternating pairs and64 evolving repeats.
 */
inline void runEveryAcceptedPrefixEvents(int ordinal)
{
    for (int heads : {12, 24})
        for (int rows : {8, 16})
            for (int accepted = 0; accepted <= rows; ++accepted)
                runCompleteEventCell(ordinal, heads, rows, accepted, accepted,
                    EventAcceptanceSpectrum::EveryPrefixM8M16);
}

/** @brief Physical-M16 reduced-live complete operations; live16 uses retained all-prefix evidence.
 * @param ordinal Actual fixture-owned CUDA participant.
 * @details 48cells: H12/24 times unique zero/middle/full acceptance for live
 * 0/1/2/3/4/7/8/9/15. Counts remain device-owned in both timed graphs.
 */
inline void runPhysicalM16LiveEvents(int ordinal)
{
    for (int heads : {12, 24})
        for (int live : {0, 1, 2, 3, 4, 7, 8, 9, 15})
        {
            std::vector<int> accepted{0, live / 2, live};
            std::sort(accepted.begin(), accepted.end());
            accepted.erase(std::unique(accepted.begin(), accepted.end()), accepted.end());
            for (int count : accepted)
                runCompleteEventCell(ordinal, heads, 16, count, count,
                    EventAcceptanceSpectrum::PhysicalM16LiveCounts, live);
        }
}

} // namespace cuda_dense_compact_test
} // namespace

TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseCompactCompleteOperationEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "dense compact event selected device");
    cuda_dense_compact_test::runCompleteEvents(cuda_ordinal_);
}

TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseCompactEveryAcceptedPrefixEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "dense compact acceptance spectrum selected device");
    cuda_dense_compact_test::runEveryAcceptedPrefixEvents(cuda_ordinal_);
}

/** @test One retained physical16 graph traverses full/ragged/empty/full counts without recapture. */
TEST_F(Test__CUDAGDNPaddedRealLength, DenseCompactPhysicalM16LiveReplay)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "Dense physical16 live replay device");
    using namespace cuda_dense_compact_test;
    for (int heads : {12, 24})
    {
        Cell cell(heads, heads / 3, 128, 128, 1, 16, Layout::ValueMajorKeyContiguous);
        const auto merged = cell.merged_host, query = cell.q_host, key = cell.k_host,
            value = cell.v_host, alpha = cell.alpha_host, beta = cell.beta_host;
        for (int live : {16, 0, 1, 2, 3, 4, 7, 8, 9, 15, 0, 16})
        {
            cell.reset();
            cell.merged_host = merged; cell.q_host = query; cell.k_host = key;
            cell.v_host = value; cell.alpha_host = alpha; cell.beta_host = beta;
            poisonInactiveEventRows(cell, live);
            cell.live_host[0] = live; cell.accepted_host[0] = 0;
            cell.publishCounts(); cell.observation.finish();
            const auto before = cell.compact.toHost();
            cell.capture->launch(cell.stream.stream); cell.dense->launch(cell.stream.stream);
            cell.observation.finish();
            const auto shots = cell.snapshots.toHost();
            cell.checkCapture(before, shots);
            cell.checkSerial(shots);
            const auto retained = cell.compact.toHost();
            for (int accepted : {live, 0, live / 2, live})
            {
                cell.accepted_host[0] = accepted; cell.publishCounts();
                cell.publication->launch(cell.stream.stream);
                cell.publication_alias->launch(cell.stream.stream);
                cell.observation.finish();
                const auto expected = cell.selectedState(shots, cell.accepted_host);
                equal(cell.accepted_state.toHost(), expected, "Physical16 accepted prefix");
                equal(cell.initial.toHost(), expected, "Physical16 revised acceptance");
                cell.checkFollowing(expected);
            }
            equal(cell.compact.toHost(), retained, "Physical16 retained compact changed");
        }
    }
}

/** @test Complete physical16 live-prefix cost includes real accepted publication and following M1. */
TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseCompactPhysicalM16LiveEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "Dense physical16 live events device");
    cuda_dense_compact_test::runPhysicalM16LiveEvents(cuda_ordinal_);
}

/** @test Full/threshold policies retain both native math and immutable rollback under mixed counts. */
TEST_F(Test__CUDAGDNPaddedRealLength, DenseSplitRetentionMixedRequestReplay)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "Dense split retained replay device");
    using namespace cuda_dense_compact_test;
    const std::array<GDNDenseRetentionProgram, 5> programs{{
        {GDNDenseRetentionMode::FullSnapshots, 0}, {GDNDenseRetentionMode::LiveRowThreshold, 2},
        {GDNDenseRetentionMode::LiveRowThreshold, 3}, {GDNDenseRetentionMode::LiveRowThreshold, 4},
        {GDNDenseRetentionMode::LiveRowThreshold, 8}}};
    const std::vector<std::vector<int>> schedule{{16, 16, 16}, {0, 1, 2}, {2, 3, 4},
        {8, 0, 3}, {4, 2, 8}, {3, 4, 0}, {7, 8, 9}, {0, 0, 0}, {16, 16, 16}};
    for (const auto program : programs)
        for (int dk : {64, 128})
            for (const auto axes : {Layout::KeyMajorValueContiguous, Layout::ValueMajorKeyContiguous})
            {
                SCOPED_TRACE(::testing::Message() << "DK=" << dk << " axes=" << int(axes)
                    << " program=" << unsigned(program.mode) << " threshold=" << program.compact_min_live_rows);
                Cell cell(3, 2, dk, dk == 64 ? 17 : 65, 3, 16, axes, true,
                    program.mode == GDNDenseRetentionMode::FullSnapshots,
                    program.mode == GDNDenseRetentionMode::FullSnapshots || program.compact_min_live_rows == 4, program);
                auto bad = cell.input();
                bad.transaction.encoding = GDNDenseCompactEncoding::ExactCorrectionsV1;
                EXPECT_FALSE(launchCUDADenseCompactRecurrence(bad, cell.stream.stream, cell.capture_plan));
                auto selected = GDNDenseCompactAcceptedPrefixInput{cell.transaction(), cell.acceptedSelection(), cell.accepted_state.ptr};
                selected.transaction.encoding = GDNDenseCompactEncoding::ExactCorrectionsV1;
                EXPECT_FALSE(launchCUDADenseCompactAcceptedPrefix(selected, cell.stream.stream));
                cell.run(schedule);
            }
}

namespace
{
namespace cuda_dense_compact_test
{
/** @brief Reuse the exact existing live48 schedule with one explicit candidate retention policy.
 * @param ordinal Actual fixture-selected CUDA participant.
 * @param program Full or threshold4 candidate; this is not a serving/default selector. */
inline void runSplitRetentionLiveEvents(int ordinal, GDNDenseRetentionProgram program)
{
    for (int heads : {12, 24})
        for (int live : {0, 1, 2, 3, 4, 7, 8, 9, 15})
            for (int accepted : std::set<int>{0, live / 2, live})
                runCompleteEventCell(ordinal, heads, 16, accepted, accepted,
                    EventAcceptanceSpectrum::PhysicalM16LiveCounts, live, program);
}
}
}

/** @test Full retention versus unchanged legacy complete chain on exact live48 work. */
TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseFullRetentionPhysicalM16LiveEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "Dense full retention event device");
    cuda_dense_compact_test::runSplitRetentionLiveEvents(cuda_ordinal_, {llaminar2::gdn::GDNDenseRetentionMode::FullSnapshots, 0});
}

/** @test Device-live threshold4 versus unchanged legacy complete chain on exact live48 work. */
TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseThresholdRetentionPhysicalM16LiveEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "Dense threshold retention event device");
    cuda_dense_compact_test::runSplitRetentionLiveEvents(cuda_ordinal_, {llaminar2::gdn::GDNDenseRetentionMode::LiveRowThreshold, 4});
}

/** @test More than one lane-stride metadata iteration preserves all accepted bytes and request holes. */
TEST_F(Test__CUDAGDNPaddedRealLength, DensePublisherMetadataMultipleLaneIterations)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "Dense publisher validation selected device");
    using namespace cuda_dense_compact_test;
    const std::array<GDNDenseRetentionProgram, 3> programs{{
        {GDNDenseRetentionMode::ExactCorrections, 0},
        {GDNDenseRetentionMode::FullSnapshots, 0},
        {GDNDenseRetentionMode::LiveRowThreshold, 2}}};
    const std::vector<std::vector<int>> schedule{
        {3, 3, 3, 3, 3, 3, 3, 3, 3},
        {0, 1, 2, 3, 0, 1, 2, 3, 0},
        {3, 0, 1, 2, 3, 0, 1, 2, 3},
        {0, 0, 0, 0, 0, 0, 0, 0, 0},
        {3, 3, 3, 3, 3, 3, 3, 3, 3}};
    for (const auto program : programs)
        for (int dk : {64, 128})
            for (auto axes : {Layout::KeyMajorValueContiguous, Layout::ValueMajorKeyContiguous})
            {
                SCOPED_TRACE(::testing::Message() << "DK=" << dk << " axes=" << int(axes)
                    << " program=" << unsigned(program.mode));
                // 33*9=297 owners: lanes0..40 must validate a second owner,
                // including a live final request after an earlier request hole.
                Cell cell(33, 11, dk, 1, 9, 3, axes, true,
                    program.mode == GDNDenseRetentionMode::FullSnapshots, false, program);
                ASSERT_GT(std::size_t(cell.requests) * cell.heads, 256u);
                cell.run(schedule);
            }
}

/** @test Explicit retained-only capture preserves all states/revisions without a terminal owner. */
TEST_F(Test__CUDAGDNPaddedRealLength, DenseRetainedPrefixesOnlyReplay)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "retained-only selected device");
    using namespace cuda_dense_compact_test;
    const std::array<GDNDenseRetentionProgram, 3> programs{{
        {GDNDenseRetentionMode::ExactCorrections, 0}, {GDNDenseRetentionMode::FullSnapshots, 0},
        {GDNDenseRetentionMode::LiveRowThreshold, 4}}};
    const std::vector<std::vector<int>> schedule{{16, 16, 16}, {0, 1, 2}, {2, 3, 4},
        {8, 0, 3}, {4, 2, 8}, {3, 4, 0}, {7, 8, 9}, {0, 0, 0}, {16, 16, 16}};
    for (const auto program : programs)
        for (int dk : {64, 128})
            for (const auto axes : {Layout::KeyMajorValueContiguous, Layout::ValueMajorKeyContiguous})
            {
                SCOPED_TRACE(::testing::Message() << "DK=" << dk << " axes=" << int(axes)
                    << " program=" << unsigned(program.mode));
                Cell cell(3, 2, dk, dk == 64 ? 17 : 65, 3, 16, axes, true, false, true, program,
                    GDNDenseCompactTerminalKind::RetainedPrefixesOnly);
                ASSERT_EQ(cell.input().terminal.kind(), GDNDenseCompactTerminalKind::RetainedPrefixesOnly);
                ASSERT_EQ(cell.input().terminal.destination(), nullptr);
                ASSERT_EQ(cell.terminal.count, std::size_t(Cell::Guard));
                cell.run(schedule);
            }
}

namespace
{
namespace cuda_dense_compact_test
{
/** @brief Same live48 complete-operation work with explicit retained-only output.
 * @param ordinal Actual fixture-owned participant.
 * @param program Explicit candidate retention; no serving or arithmetic selection. */
inline void runRetainedPrefixesOnlyLiveEvents(int ordinal, GDNDenseRetentionProgram program)
{
    for (int heads : {12, 24})
        for (int live : {0, 1, 2, 3, 4, 7, 8, 9, 15})
            for (int accepted : std::set<int>{0, live / 2, live})
                runCompleteEventCell(ordinal, heads, 16, accepted, accepted,
                    EventAcceptanceSpectrum::PhysicalM16LiveCounts, live, program,
                    GDNDenseCompactTerminalKind::RetainedPrefixesOnly);
}
}
}

/** @test Exact retained-only output versus legacy capture/publication/next M1. */
TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseExactRetainedPrefixesOnlyPhysicalM16LiveEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "retained-only selected device");
    cuda_dense_compact_test::runRetainedPrefixesOnlyLiveEvents(cuda_ordinal_,
        {llaminar2::gdn::GDNDenseRetentionMode::ExactCorrections, 0});
}

/** @test Full retained-only output versus legacy capture/publication/next M1. */
TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseFullRetainedPrefixesOnlyPhysicalM16LiveEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "retained-only selected device");
    cuda_dense_compact_test::runRetainedPrefixesOnlyLiveEvents(cuda_ordinal_,
        {llaminar2::gdn::GDNDenseRetentionMode::FullSnapshots, 0});
}

/** @test Threshold retained-only output versus legacy capture/publication/next M1. */
TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseThresholdRetainedPrefixesOnlyPhysicalM16LiveEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "retained-only selected device");
    cuda_dense_compact_test::runRetainedPrefixesOnlyLiveEvents(cuda_ordinal_,
        {llaminar2::gdn::GDNDenseRetentionMode::LiveRowThreshold, 4});
}

namespace
{
namespace cuda_dense_compact_test
{
/** @brief Measure the unchanged live48 schedule on CUDA's actual default axes.
 * @param ordinal Existing fixture-owned CUDA participant.
 * @param program ExactCorrections or Threshold4 retained representation.
 * @param terminal_kind Explicit candidate endpoint requirement. */
inline void runKeyMajorLiveEvents(int ordinal, GDNDenseRetentionProgram program,
    GDNDenseCompactTerminalKind terminal_kind)
{
    for (int heads : {12, 24})
        for (int live : {0, 1, 2, 3, 4, 7, 8, 9, 15})
            for (int accepted : std::set<int>{0, live / 2, live})
                runCompleteEventCell(ordinal, heads, 16, accepted, accepted,
                    EventAcceptanceSpectrum::PhysicalM16LiveCounts, live, program, terminal_kind,
                    Layout::KeyMajorValueContiguous);
}
}
}

/** @test KeyMajor Exact RequiredTerminal complete capture/publication/next-row events. */
TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseExactKeyMajorRequiredTerminalPhysicalM16LiveEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "KeyMajor Dense events selected device");
    cuda_dense_compact_test::runKeyMajorLiveEvents(cuda_ordinal_, {llaminar2::gdn::GDNDenseRetentionMode::ExactCorrections, 0},
        llaminar2::gdn::GDNDenseCompactTerminalKind::DenseDestination);
}

/** @test KeyMajor Exact RetainedPrefixesOnly complete capture/publication/next-row events. */
TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseExactKeyMajorRetainedPrefixesOnlyPhysicalM16LiveEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "KeyMajor Dense events selected device");
    cuda_dense_compact_test::runKeyMajorLiveEvents(cuda_ordinal_, {llaminar2::gdn::GDNDenseRetentionMode::ExactCorrections, 0},
        llaminar2::gdn::GDNDenseCompactTerminalKind::RetainedPrefixesOnly);
}

/** @test KeyMajor Threshold RequiredTerminal complete capture/publication/next-row events. */
TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseThresholdKeyMajorRequiredTerminalPhysicalM16LiveEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "KeyMajor Dense events selected device");
    cuda_dense_compact_test::runKeyMajorLiveEvents(cuda_ordinal_, {llaminar2::gdn::GDNDenseRetentionMode::LiveRowThreshold, 4},
        llaminar2::gdn::GDNDenseCompactTerminalKind::DenseDestination);
}

/** @test KeyMajor Threshold RetainedPrefixesOnly complete capture/publication/next-row events. */
TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseThresholdKeyMajorRetainedPrefixesOnlyPhysicalM16LiveEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "KeyMajor Dense events selected device");
    cuda_dense_compact_test::runKeyMajorLiveEvents(cuda_ordinal_, {llaminar2::gdn::GDNDenseRetentionMode::LiveRowThreshold, 4},
        llaminar2::gdn::GDNDenseCompactTerminalKind::RetainedPrefixesOnly);
}

namespace
{
namespace cuda_dense_compact_test
{
/** @brief Attribute six actual evolving KM transactions without replacing canonical timing.
 * @param ordinal Existing fixture-owned CUDA participant.
 * @param program ExactCorrections or Threshold4 candidate retention. */
inline void runKeyMajorPhaseEvents(int ordinal, GDNDenseRetentionProgram program)
{
    for (int heads : {12, 24})
        for (int live : {2, 8, 15})
            runCompleteEventCell(ordinal, heads, 16, live, live,
                EventAcceptanceSpectrum::PhysicalM16LiveCounts, live, program,
                GDNDenseCompactTerminalKind::RetainedPrefixesOnly, Layout::KeyMajorValueContiguous,
                CompleteEventInstrumentation::PhaseEventNodes);
}
}
}

/** @test Instrumented Exact KM capture/publication/following attribution; not canonical timing. */
TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseExactKeyMajorRetainedPrefixesOnlyPhaseEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "KeyMajor phase diagnostic selected device");
    cuda_dense_compact_test::runKeyMajorPhaseEvents(cuda_ordinal_, {llaminar2::gdn::GDNDenseRetentionMode::ExactCorrections, 0});
}

/** @test Instrumented Threshold KM capture/publication/following attribution; not canonical timing. */
TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseThresholdKeyMajorRetainedPrefixesOnlyPhaseEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "KeyMajor phase diagnostic selected device");
    cuda_dense_compact_test::runKeyMajorPhaseEvents(cuda_ordinal_, {llaminar2::gdn::GDNDenseRetentionMode::LiveRowThreshold, 4});
}

namespace
{
namespace cuda_dense_compact_test
{
/** @brief Time native verifier preparation/publication plus raw M1 continuation.
 * @param ordinal Actual fixture device.
 * @param program ExactCorrections or Threshold4, never a serving selector.
 * @details Common convolution is excluded from both arms. Both use the same
 * original CUDA raw-merged M1 continuation on the selected persistent axes.
 */
inline void runPostConvolutionLiveEvents(int ordinal, GDNDenseRetentionProgram program)
{
    for (int heads : {12, 24})
        for (int live : {0, 1, 2, 3, 4, 7, 8, 9, 15})
            for (int accepted : std::set<int>{0, live / 2, live})
                runCompleteEventCell(ordinal, heads, 16, accepted, accepted,
                    EventAcceptanceSpectrum::PhysicalM16LiveCounts, live, program,
                    GDNDenseCompactTerminalKind::RetainedPrefixesOnly, Layout::KeyMajorValueContiguous, CompleteEventInstrumentation::Canonical,
                    CompleteEventInputBoundary::PostConvolutionMerged);
}
}
}

/** @test Explicit complete post-convolution Exact verifier boundary; not serving throughput. */
TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseExactPostConvolutionRawPhysicalM16LiveEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "Post-convolution event device");
    cuda_dense_compact_test::runPostConvolutionLiveEvents(cuda_ordinal_, {llaminar2::gdn::GDNDenseRetentionMode::ExactCorrections, 0});
}

/** @test Explicit complete post-convolution Threshold verifier boundary; not serving throughput. */
TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseThresholdPostConvolutionRawPhysicalM16LiveEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "Post-convolution event device");
    cuda_dense_compact_test::runPostConvolutionLiveEvents(cuda_ordinal_, {llaminar2::gdn::GDNDenseRetentionMode::LiveRowThreshold, 4});
}

/** @test Explicit complete post-convolution Threshold8 checkpoint-boundary verifier experiment; not serving throughput. */
TEST_F(Test__CUDAGDNPaddedRealLength, DISABLED_DenseThreshold8PostConvolutionRawPhysicalM16LiveEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "Post-convolution event device");
    cuda_dense_compact_test::runPostConvolutionLiveEvents(cuda_ordinal_, {llaminar2::gdn::GDNDenseRetentionMode::LiveRowThreshold, 8});
}

/** @test Native-only planar preparation preserves live holes and revised accepted continuations.
 * @details Ratio3/offset0 matches production grouped head packing. Host-expanded
 * arrays are byte oracles only: the original full capture reads native scratch.
 */
TEST_F(Test__CUDAGDNPaddedRealLength, DensePostConvolutionNativePreparationReplay)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "Post-convolution preparation test device");
    using namespace cuda_dense_compact_test;
    for (int dk : {64, 128})
        for (auto axes : {Layout::KeyMajorValueContiguous, Layout::ValueMajorKeyContiguous})
        {
            SCOPED_TRACE(::testing::Message() << "DK=" << dk << " axes=" << int(axes));
            Cell cell(6, 2, dk, dk == 64 ? 17 : 65, 3, 16, axes, true, false, true, {},
                GDNDenseCompactTerminalKind::RetainedPrefixesOnly, 0);
            EventPlanarPreparation preparation(cell, cuda_ordinal_);
            CudaCapturedGraph prepare(cell.stream.stream, [&] { return preparation.enqueue(); });
            CudaCapturedGraph dense(cell.stream.stream, [&] {
                return launchCUDAPlanarRecurrence(preparation.input(), cell.stream.stream);
            });
            inspectEventPreparation(cell, preparation, prepare);
            CudaMergedReplayRetirement retirement(cell.stream.stream);
            for (const auto &counts : std::array<std::array<int, 3>, 5>{{
                    {16, 16, 16}, {2, 0, 8}, {0, 3, 1}, {0, 0, 0}, {16, 16, 16}}})
            {
                cell.reset(); preparation.poison();
                cell.live_host.assign(counts.begin(), counts.end());
                std::fill(cell.accepted_host.begin(), cell.accepted_host.end(), 0);
                cell.publishCounts(); retirement.finish();
                const auto before = cell.compact.toHost();
                cell.capture->launch(cell.stream.stream);
                prepare.launch(cell.stream.stream);
                dense.launch(cell.stream.stream);
                retirement.finish();
                const auto shots = cell.snapshots.toHost();
                preparation.checkPrepared();
                cell.checkCapture(before, shots);
                cell.checkSerial(shots);
                const auto retained = cell.compact.toHost();
                for (int selected : {16, 7, 8, 1, 0, 15, 16})
                {
                    for (int request = 0; request < cell.requests; ++request)
                        cell.accepted_host[request] = std::min(selected, cell.live_host[request]);
                    cell.publishCounts();
                    cell.publication->launch(cell.stream.stream);
                    cell.publication_alias->launch(cell.stream.stream);
                    retirement.finish();
                    const auto expected = cell.selectedState(shots, cell.accepted_host);
                    equal(cell.accepted_state.toHost(), expected, "Native prepared accepted bank differs");
                    equal(cell.initial.toHost(), expected, "Native prepared revised original bank differs");
                    cell.checkFollowing(expected);
                    preparation.checkPrepared();
                }
                equal(cell.compact.toHost(), retained, "Native prepared publication changed retained source");
            }
        }
}

/** @test Existing strided metadata drives mixed-request skips/restores without conversion buffers. */
TEST_F(Test__CUDAGDNPaddedRealLength, DenseCompactTypedDeviceSelectionReplay)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "Typed retained selection device");
    using namespace cuda_dense_compact_test;
    using namespace llaminar2::gdn;
    const std::array<GDNDenseRetentionProgram, 3> programs{{
        {}, {GDNDenseRetentionMode::FullSnapshots, 0},
        {GDNDenseRetentionMode::LiveRowThreshold, 4}}};
    const std::array<std::array<int, 3>, 4> live_schedule{{
        {16, 16, 16}, {0, 3, 9}, {0, 0, 0}, {16, 16, 16}}};
    for (auto kind : {GDNDenseAcceptedSelectionKind::AcceptedCounts,
        GDNDenseAcceptedSelectionKind::OwnRequestFlattenedRows,
        GDNDenseAcceptedSelectionKind::RequestTerminalLengths})
        for (auto program : programs)
            for (auto axes : {Layout::KeyMajorValueContiguous, Layout::ValueMajorKeyContiguous})
            {
                std::vector<int> metadata(9), observed_metadata(9);
                Cell cell(2, 1, 128, 65, 3, 16, axes, true, false, false,
                    program, GDNDenseCompactTerminalKind::RetainedPrefixesOnly, -5, kind, 3);
                const auto binding = cell.acceptedSelection();
                cell.inspect(*cell.capture, true);
                cell.inspect(*cell.publication, false);
                cell.inspect(*cell.publication_alias, false, cell.initial.ptr);
                ASSERT_FALSE(::testing::Test::HasFailure());
                for (const auto &live : live_schedule)
                {
                    cell.reset();
                    cell.live_host.assign(live.begin(), live.end());
                    cell.publishCounts();
                    cell.observation.finish();
                    const auto before = cell.compact.toHost();
                    cell.capture->launch(cell.stream.stream);
                    cell.dense->launch(cell.stream.stream);
                    cell.observation.finish();
                    const auto shots = cell.snapshots.toHost();
                    cell.checkCapture(before, shots);
                    cell.checkSerial(shots);
                    const auto retained = cell.compact.toHost();
                    cell.accepted_state.copyFrom(cell.initial_host);
                    cell.initial.copyFrom(cell.initial_host);
                    auto previous = cell.initial_host;
                    auto previous_alias = cell.initial_host;
                    // Full -> mixed holes -> zero/no-op -> full uses the SAME native graphs.
                    for (int step = 0; step < 4; ++step)
                    {
                        std::vector<int> accepted(3, 0);
                        std::vector<bool> skip(3, false);
                        std::fill(cell.accepted_host.begin(), cell.accepted_host.end(),
                            std::numeric_limits<int>::max());
                        for (int request = 0; request < 3; ++request)
                        {
                            accepted[request] = step == 0 || step == 3 ? live[request] :
                                (step == 1 ? live[request] / 2 : 0);
                            skip[request] = kind != GDNDenseAcceptedSelectionKind::AcceptedCounts &&
                                (live[request] == 0 || step == 2 || (step == 1 && request == 1));
                            int observed = accepted[request];
                            if (kind == GDNDenseAcceptedSelectionKind::OwnRequestFlattenedRows)
                            {
                                if (!accepted[request]) skip[request] = true;
                                observed = skip[request] ? -1 : request * cell.rows + accepted[request] - 1;
                            }
                            else if (kind == GDNDenseAcceptedSelectionKind::RequestTerminalLengths && skip[request])
                                observed = 0;
                            cell.accepted_host[request * cell.selection_stride] = observed;
                        }
                        cell.publishCounts();
                        cell.observation.finish();
                        checkCuda(cudaMemcpyAsync(metadata.data(), cell.accepted.ptr,
                            metadata.size() * sizeof(int), cudaMemcpyDeviceToHost, cell.stream.stream),
                            "Typed selection source snapshot");
                        cell.observation.finish();
                        cell.publication->launch(cell.stream.stream);
                        cell.publication_alias->launch(cell.stream.stream);
                        cell.observation.finish();
                        auto expected = cell.selectedState(shots, accepted);
                        auto expected_alias = expected;
                        for (int request = 0; request < 3; ++request)
                            if (skip[request])
                            {
                                const auto offset = std::size_t(request) * cell.state_stride;
                                std::copy_n(previous.data() + offset, cell.state_stride, expected.data() + offset);
                                std::copy_n(previous_alias.data() + offset, cell.state_stride, expected_alias.data() + offset);
                            }
                        equal(cell.accepted_state.toHost(), expected, "Typed retained restore/no-op");
                        equal(cell.initial.toHost(), expected_alias, "Typed retained alias restore/no-op");
                        checkCuda(cudaMemcpyAsync(observed_metadata.data(), cell.accepted.ptr,
                            observed_metadata.size() * sizeof(int), cudaMemcpyDeviceToHost, cell.stream.stream),
                            "Typed selection source recheck");
                        cell.observation.finish();
                        require(observed_metadata == metadata, "Typed selection metadata changed");
                        equal(cell.compact.toHost(), retained, "Typed retained source immutable");
                        require(cell.acceptedSelection().sameBinding(binding), "Typed retained binding changed");
                        cell.checkFollowing(expected);
                        previous = cell.accepted_state.toHost();
                        previous_alias = expected_alias;
                    }
                }
    }
}

/** @test Cold scheduling authenticates actual device/stream and both graph-function schedules. */
TEST_F(Test__CUDAGDNPaddedRealLength, DenseCaptureGridSchedulingColdBinding)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "Dense grid schedule device");
    using namespace cuda_dense_compact_test;
    using namespace llaminar2::gdn;
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, cuda_ordinal_), "Dense grid schedule properties");
    llaminar2::ComputeDevice observed{};
    observed.type = llaminar2::ComputeBackendType::GPU_CUDA;
    observed.device_id = cuda_ordinal_;
    observed.compute_units = properties.multiProcessorCount;
    observed.compute_capability = properties.major * 10 + properties.minor;
    observed.uuid = llaminar2::formatDeviceUUID(properties.uuid.bytes);
    for (auto terminal : {GDNDenseCompactTerminalKind::DenseDestination,
        GDNDenseCompactTerminalKind::RetainedPrefixesOnly})
        for (bool packed : {false, true})
        {
            const int heads = packed ? observed.compute_units / 4 + 1 : 1;
            const int dv = packed ? 128 : 1;
            Cell cell(heads, 1, 128, dv, 1, 3, Layout::KeyMajorValueContiguous,
                true, false, false, {GDNDenseRetentionMode::LiveRowThreshold, 2}, terminal);
            const auto identity = cell.capture_plan;
            EXPECT_EQ(identity.schedule(), packed ? CUDADenseRetentionSchedule::CheckpointWordTwoCTA :
                CUDADenseRetentionSchedule::PartitionOneCTA);
            EXPECT_EQ(identity.device(), cuda_ordinal_);
            EXPECT_EQ(identity.multiprocessors(), observed.compute_units);
            EXPECT_EQ(identity.deviceUUID(), observed.uuid);
            EXPECT_TRUE(identity.matches(cell.input(), cell.stream.stream));
            cell.inspect(*cell.capture, true);
            ASSERT_FALSE(::testing::Test::HasFailure());
            CudaStreamHandle other;
            EXPECT_FALSE(identity.matches(cell.input(), other.stream));
            EXPECT_FALSE(launchCUDADenseCompactRecurrence(cell.input(), other.stream, identity));
            for (void *bad : {static_cast<void *>(nullptr), reinterpret_cast<void *>(std::uintptr_t{1}),
                reinterpret_cast<void *>(std::uintptr_t{2})})
                EXPECT_THROW((void)prepareObservedCUDADenseRetentionCapturePlan(cell.geometry,
                    cell.axes, terminal, observed, bad), std::runtime_error);
            auto wrong = observed;
            ++wrong.device_id;
            EXPECT_THROW((void)prepareObservedCUDADenseRetentionCapturePlan(cell.geometry,
                cell.axes, terminal, wrong, cell.stream.stream), std::runtime_error);
            wrong = observed; ++wrong.compute_units;
            EXPECT_THROW((void)prepareObservedCUDADenseRetentionCapturePlan(cell.geometry,
                cell.axes, terminal, wrong, cell.stream.stream), std::runtime_error);
            wrong = observed; wrong.uuid[0] = wrong.uuid[0] == '0' ? '1' : '0';
            EXPECT_THROW((void)prepareObservedCUDADenseRetentionCapturePlan(cell.geometry,
                cell.axes, terminal, wrong, cell.stream.stream), std::runtime_error);
            auto changed = cell.input();
            changed.transaction.layout = Layout::ValueMajorKeyContiguous;
            EXPECT_FALSE(identity.matches(changed, cell.stream.stream));
            EXPECT_FALSE(launchCUDADenseCompactRecurrence(changed, cell.stream.stream, identity));
            // Query rejection is tested inside recording; the normal planned
            // launch still captures successfully without any property API call.
            CudaCapturedGraph cold_rejection(cell.stream.stream, [&] {
                EXPECT_THROW((void)prepareObservedCUDADenseRetentionCapturePlan(cell.geometry,
                    cell.axes, terminal, observed, cell.stream.stream), std::runtime_error);
                return launchCUDADenseCompactRecurrence(cell.input(), cell.stream.stream, identity);
            });
            cell.inspect(cold_rejection, true);
            ASSERT_FALSE(::testing::Test::HasFailure());
            cell.run({{3}, {0}, {3}});
            EXPECT_TRUE(cell.capture_plan.sameBinding(identity));
        }
}


namespace
{
namespace cuda_dense_compact_test
{
/** @brief Prove actual prepared kernel ownership without replacing its cache banks.
 * @param ordinal Actual selected backend device, observed through normal fixture startup.
 * @details All allocations and hardware discovery are cold. Two native streams
 * are ordered by a real producer event; terminal joins are test observations only.
 * The existing workspace manager owns both retained and optional terminal views.
 */
inline void preparedInvocationReplay(int ordinal)
{
    const auto device = DeviceId::cuda(ordinal);
    for (int requests : {1, 3})
    for (bool select_secondary : {false, true})
    for (int dk : {64, 128})
    for (auto axes : {Layout::KeyMajorValueContiguous, Layout::ValueMajorKeyContiguous})
    for (auto terminal_kind : {GDNDenseCompactTerminalKind::DenseDestination,
        GDNDenseCompactTerminalKind::RetainedPrefixesOnly})
    {
        Cell cell(2, 1, dk, dk == 64 ? 17 : 65, requests, 9, axes,
            true, false, false, {}, terminal_kind);
        CudaStreamHandle publication_stream;
        const int state = cell.state_stride;
        const int primary_size = select_secondary ? state / 2 : state;
        const int secondary_size = select_secondary ? state : state * 2;
        const int maximum = std::max(primary_size, secondary_size);
        CudaFloatBuffer primary(primary_size + Cell::Guard, Cell::Poison);
        CudaFloatBuffer secondary(secondary_size + Cell::Guard, Cell::Poison);
        CudaFloatBuffer packed(std::size_t(requests) * maximum + Cell::Guard, Cell::Poison);
        CUDAGatedDeltaNet kernel(ordinal);
        require(kernel.bindDeviceState({primary.ptr, primary_size, secondary.ptr, secondary_size,
            packed.ptr, std::size_t(requests) * maximum, requests, axes}), "Prepared kernel state binding failed");
        kernel.setGPUStream(cell.stream.stream);
        const auto retained_count = cell.compact.count;
        const auto terminal_count = cell.initial_host.size();
        WorkspaceRequirements requirements;
        requirements.buffers.emplace_back("prepared_retained", retained_count * sizeof(float), 64);
        if (terminal_kind == GDNDenseCompactTerminalKind::DenseDestination)
            requirements.buffers.emplace_back("prepared_terminal", terminal_count * sizeof(float), 64);
        DeviceWorkspaceManager workspace(device, requirements.total_bytes() + 4096);
        require(workspace.allocate(requirements), "Prepared retained workspace allocation failed");
        auto *retained = static_cast<float *>(workspace.getBuffer("prepared_retained"));
        float *terminal = terminal_kind == GDNDenseCompactTerminalKind::DenseDestination ?
            static_cast<float *>(workspace.getBuffer("prepared_terminal")) : nullptr;
        GDNRetainedInvocationBinding binding{cell.geometry, terminal_kind,
            {retained, workspace.getBufferSize("prepared_retained") / sizeof(float)},
            terminal ? std::span<float>(terminal, workspace.getBufferSize("prepared_terminal") / sizeof(float)) :
                std::span<float>{}, workspace.id(), cell.stream.stream};
        auto program = kernel.prepareRetainedInvocation(cell.input().sources, binding);
        auto same = kernel.prepareRetainedInvocation(cell.input().sources, binding);
        require(program && same && program->sameCaptureIdentity(*same), "Prepared immutable identity mismatch");
        const auto frozen = program->captureInput();
        require(frozen.transaction.initial_state == (requests == 1 ?
            (select_secondary ? secondary.ptr : primary.ptr) : packed.ptr), "Prepared state source authority differs");
        require(program->publicationDestination() == frozen.transaction.initial_state,
            "Prepared publication must target the authoritative active bank");
        require(frozen.transaction.generation == workspace.id() &&
            program->binding().workspace_owner_identity == workspace.id() &&
            program->binding().capture_stream == cell.stream.stream,
            "Prepared workspace/stream authority differs");
        const auto expected_plan = prepareCUDADenseRetentionCapturePlan(cell.geometry, axes, terminal_kind, ordinal, cell.stream.stream);
        const std::string expected_family = expected_plan.captureFunctionFamily();
        require(program->captureProgramIdentity() == 256ULL + (static_cast<std::uint64_t>(terminal_kind) << 4) + static_cast<std::uint64_t>(expected_plan.schedule()),
            "Prepared compiled program identity differs");
        std::vector<float> retained_poison(retained_count, Cell::Poison);
        std::vector<float> terminal_poison(terminal_count, Cell::Poison);
        std::vector<float> primary_source(primary.count, Cell::Poison), secondary_source(secondary.count, Cell::Poison);
        std::vector<float> packed_source(packed.count, Cell::Poison);
        std::unique_ptr<CudaCapturedGraph> capture, publication;
        struct Retirement final
        {
            cudaStream_t capture_stream, publish_stream;
            cudaEvent_t ready = nullptr;
            /** @brief Own both streams and the producer event until borrowed bytes retire. */
            explicit Retirement(cudaStream_t a, cudaStream_t b) : capture_stream(a), publish_stream(b)
            { checkCuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "Prepared event create"); }
            /** @brief Terminal join precedes graph, workspace and staging destruction. */
            ~Retirement()
            {
                if (cudaStreamSynchronize(capture_stream) != cudaSuccess ||
                    cudaStreamSynchronize(publish_stream) != cudaSuccess) std::terminate();
                if (ready) (void)cudaEventDestroy(ready);
            }
        } retirement(cell.stream.stream, publication_stream.stream);
        const auto read = [&](const float *pointer, std::size_t count, cudaStream_t stream) {
            std::vector<float> result(count);
            Retirement join(stream, stream);
            checkCuda(cudaMemcpyAsync(result.data(), pointer, count * sizeof(float),
                cudaMemcpyDeviceToHost, stream), "Prepared actual bank read");
            checkCuda(cudaStreamSynchronize(stream), "Prepared observation join");
            return result;
        };
        capture = std::make_unique<CudaCapturedGraph>(cell.stream.stream, [&] { return program->capture(); });
        publication = std::make_unique<CudaCapturedGraph>(publication_stream.stream, [&] {
            return program->publish(cell.acceptedSelection(), publication_stream.stream);
        });
        // Authenticate the actual native function before decoding its typed tuple.
        std::size_t nodes = 0;
        checkCuda(cudaGraphGetNodes(capture->graph, nullptr, &nodes), "Prepared capture nodes");
        require(nodes == 1, "Prepared capture must contain exactly its native recurrence");
        cudaGraphNode_t node{};
        checkCuda(cudaGraphGetNodes(capture->graph, &node, &nodes), "Prepared capture node");
        cudaGraphNodeType node_kind{};
        checkCuda(cudaGraphNodeGetType(node, &node_kind), "Prepared capture node kind");
        require(node_kind == cudaGraphNodeTypeKernel, "Prepared capture node is not native recurrence");
        cudaKernelNodeParams parameters{};
        checkCuda(cudaGraphKernelNodeGetParams(node, &parameters), "Prepared capture arguments");
        const char *family = nullptr;
        checkCuda(cudaFuncGetName(&family, parameters.func), "Prepared function name");
        require(family && std::string(family).find(expected_family) != std::string::npos,
            "Prepared capture used a foreign native family");
        require(parameters.kernelParams && parameters.kernelParams[0], "Prepared typed capture tuple absent");
        require(parameters.gridDim.x == unsigned(requests * cell.heads) &&
            parameters.gridDim.y == unsigned((cell.dv + 31) / 32) && parameters.gridDim.z == 1 &&
            parameters.blockDim.x == 256 && parameters.blockDim.y == 1 && parameters.blockDim.z == 1 &&
            parameters.sharedMemBytes == std::size_t(2 * dk + 512) * sizeof(float), "Prepared capture launch geometry differs");
        auto recorded = frozen;
        std::memcpy(&recorded, parameters.kernelParams[0], sizeof(recorded));
        require(recorded.transaction.initial_state == frozen.transaction.initial_state &&
            recorded.transaction.storage == retained && recorded.transaction.generation == workspace.id() &&
            recorded.sources.merged == cell.merged.ptr && recorded.sources.output == cell.output.ptr &&
            recorded.sources.rows.lengthOwner() == cell.live.ptr && recorded.terminal.kind() == terminal_kind &&
            recorded.terminal.destination() == terminal &&
            recorded.transaction.geometry.requests() == requests && recorded.transaction.geometry.heads() == cell.heads &&
            recorded.transaction.geometry.keyWidth() == cell.dk && recorded.transaction.geometry.valueWidth() == cell.dv &&
            recorded.transaction.geometry.rowsPerRequest() == cell.rows &&
            recorded.transaction.encoding == cell.geometry.encoding() && recorded.transaction.layout == axes,
            "Prepared captured tuple differs");
        std::size_t publication_nodes = 0;
        checkCuda(cudaGraphGetNodes(publication->graph, nullptr, &publication_nodes), "Prepared publication nodes");
        std::vector<cudaGraphNode_t> publication_node_list(publication_nodes);
        checkCuda(cudaGraphGetNodes(publication->graph, publication_node_list.data(), &publication_nodes), "Prepared publication node list");
        unsigned accepted_kernels = 0;
        for (const auto publication_node : publication_node_list)
        {
            cudaGraphNodeType kind{};
            checkCuda(cudaGraphNodeGetType(publication_node, &kind), "Prepared publication node kind");
            if (kind != cudaGraphNodeTypeKernel)
            {
                require(kind == cudaGraphNodeTypeMemcpy, "Prepared publication has an unexpected nonkernel operation");
                continue;
            }
            cudaKernelNodeParams arguments{};
            checkCuda(cudaGraphKernelNodeGetParams(publication_node, &arguments), "Prepared publication arguments");
            const char *publication_family = nullptr;
            checkCuda(cudaFuncGetName(&publication_family, arguments.func), "Prepared publication function");
            require(publication_family && std::string(publication_family).find(
                "cuda_gdn_dense_compact_accepted_prefix_kernel") != std::string::npos,
                "Prepared publication selected an unexpected kernel");
            require(arguments.kernelParams && arguments.kernelParams[0], "Prepared publication tuple missing");
            GDNDenseCompactAcceptedPrefixInput actual{frozen.transaction, cell.acceptedSelection(), program->publicationDestination()};
            std::memcpy(&actual, arguments.kernelParams[0], sizeof(actual));
            require(actual.transaction.storage == retained && actual.transaction.initial_state == frozen.transaction.initial_state &&
                actual.transaction.generation == workspace.id() && actual.selection.sameBinding(cell.acceptedSelection()) &&
                actual.destination_state == program->publicationDestination() &&
                arguments.gridDim.x == unsigned(requests * cell.heads) &&
                arguments.gridDim.y == unsigned(cell.geometry.matrixTileCount(4096)) &&
                arguments.blockDim.x == 256 && arguments.sharedMemBytes == 0,
                "Prepared publication tuple/geometry differs");
            ++accepted_kernels;
        }
        require(accepted_kernels == 1, "Prepared publication must contain one accepted-prefix kernel");
        // A serial ordinary sibling may change legacy workspaces, not this handle.
        kernel.bindVerifierStateCaptureWorkspace(cell.snapshots.ptr, requests * cell.rows, state);
        kernel.bindSpeculativeStateWorkspace(cell.oracle_state.ptr, state);
        kernel.bindVerifierStateCaptureWorkspace(nullptr, 0, 0);
        kernel.bindSpeculativeStateWorkspace(nullptr, 0);
        require(program->sameCaptureIdentity(*same) &&
            program->captureInput().transaction.storage == retained,
            "Ordinary sibling changed prepared retained bindings");
        for (int replay = 0; replay < 4; ++replay)
        {
            cell.reset();
            for (int r = 0; r < requests; ++r)
                cell.live_host[r] = replay == 2 ? 0 : replay == 1 ? (r == 0 ? 3 : r == 1 ? 0 : 8) : 9;
            std::fill(cell.accepted_host.begin(), cell.accepted_host.end(), 0);
            cell.publishCounts();
            cell.observation.finish();
            std::fill(primary_source.begin(), primary_source.end(), Cell::Poison);
            std::fill(secondary_source.begin(), secondary_source.end(), Cell::Poison);
            std::fill(packed_source.begin(), packed_source.end(), Cell::Poison);
            if (requests == 1)
            {
                auto &scalar = select_secondary ? secondary_source : primary_source;
                std::copy_n(cell.initial_host.begin(), state, scalar.begin());
            }
            else std::copy_n(cell.initial_host.begin(), std::size_t(requests) * state, packed_source.begin());
            primary.copyFrom(primary_source); secondary.copyFrom(secondary_source); packed.copyFrom(packed_source);
            checkCuda(cudaMemcpyAsync(retained, retained_poison.data(), retained_poison.size() * sizeof(float),
                cudaMemcpyHostToDevice, cell.stream.stream), "Prepared retained poison");
            if (terminal) checkCuda(cudaMemcpyAsync(terminal, terminal_poison.data(), terminal_poison.size() * sizeof(float),
                cudaMemcpyHostToDevice, cell.stream.stream), "Prepared terminal poison");
            capture->launch(cell.stream.stream);
            cell.dense->launch(cell.stream.stream);
            checkCuda(cudaEventRecord(retirement.ready, cell.stream.stream), "Prepared producer completion");
            checkCuda(cudaStreamWaitEvent(publication_stream.stream, retirement.ready, 0), "Prepared consumer event wait");
            checkCuda(cudaStreamSynchronize(cell.stream.stream), "Prepared captured observation");
            const auto shots = cell.snapshots.toHost();
            const auto retained_bytes = read(retained, retained_count, cell.stream.stream);
            const auto terminal_bytes = terminal ? read(terminal, terminal_count, cell.stream.stream) : terminal_poison;
            cell.checkCapture(retained_poison, shots, &retained_bytes,
                terminal ? &terminal_bytes : nullptr, workspace.id());
            equal(primary.toHost(), primary_source, "Prepared capture mutated primary scalar");
            equal(secondary.toHost(), secondary_source, "Prepared capture mutated secondary scalar");
            equal(packed.toHost(), packed_source, "Prepared capture mutated authoritative packed state");
            for (int selected : {9, 0, 4, 1, 8, 9})
            {
                for (int r = 0; r < requests; ++r) cell.accepted_host[r] = std::min(selected, cell.live_host[r]);
                cell.publishCounts();
                checkCuda(cudaEventRecord(retirement.ready, cell.stream.stream), "Prepared metadata completion");
                checkCuda(cudaStreamWaitEvent(publication_stream.stream, retirement.ready, 0), "Prepared metadata event wait");
                publication->launch(publication_stream.stream);
                checkCuda(cudaStreamSynchronize(publication_stream.stream), "Prepared actual publication join");
                const auto expected = cell.selectedState(shots, cell.accepted_host);
                const auto actual = read(program->publicationDestination(), std::size_t(requests) * state, publication_stream.stream);
                equal(actual, std::vector<float>(expected.begin(), expected.begin() + std::size_t(requests) * state),
                    "Prepared actual accepted bank");
                const auto scalar = read(select_secondary ? secondary.ptr : primary.ptr, state, publication_stream.stream);
                equal(scalar, std::vector<float>(expected.begin(), expected.begin() + state), "Prepared request-zero scalar coherence");
                const auto request_zero = read(packed.ptr, state, publication_stream.stream);
                equal(request_zero, scalar, "Prepared request-zero packed coherence");
                auto expected_primary = primary_source;
                auto expected_secondary = secondary_source;
                auto expected_packed = packed_source;
                auto &selected_scalar = select_secondary ? expected_secondary : expected_primary;
                std::copy_n(expected.begin(), state, selected_scalar.begin());
                std::copy_n(expected.begin(), std::size_t(requests) * state, expected_packed.begin());
                equal(primary.toHost(), expected_primary, "Prepared primary ownership/guard");
                equal(secondary.toHost(), expected_secondary, "Prepared secondary ownership/guard");
                equal(packed.toHost(), expected_packed, "Prepared packed stride/unused ownership/guard");
                equal(read(retained, retained_count, publication_stream.stream), retained_bytes,
                    "Prepared publication changed immutable retained prefixes");
                cell.oracle_state.copyFrom(expected);
                for (int r = 0; r < requests; ++r)
                {
                    const auto row = std::size_t(requests * cell.rows + r);
                    const auto next = [&](float *bank, float *out) {
                        return launchCUDAPlanarRecurrence({DeviceRequestRowRanges::fullyActive(1), cell.heads, cell.dk, cell.dv,
                            cell.normalize, cell.query.ptr + row * cell.heads * cell.dk, cell.key.ptr + row * cell.heads * cell.dk,
                            cell.value.ptr + row * cell.heads * cell.dv, cell.alpha.ptr + row * cell.heads,
                            cell.beta.ptr + row * cell.heads, cell.a.ptr, cell.bias.ptr, out + std::size_t(r) * cell.heads * cell.dv,
                            bank + std::size_t(r) * state, bank + std::size_t(r) * state, nullptr, 0, 0, axes}, publication_stream.stream);
                    };
                    require(next(program->publicationDestination(), cell.following.ptr) &&
                        next(cell.oracle_state.ptr, cell.oracle_following.ptr), "Prepared actual following native row rejected");
                }
                checkCuda(cudaStreamSynchronize(publication_stream.stream), "Prepared following observation");
                equal(read(program->publicationDestination(), std::size_t(requests) * state, publication_stream.stream),
                    read(cell.oracle_state.ptr, std::size_t(requests) * state, publication_stream.stream), "Prepared following state bytes");
                equal(cell.following.toHost(), cell.oracle_following.toHost(), "Prepared following output bytes");
            }
        }
    }
}
} // namespace cuda_dense_compact_test
} // namespace

/** @test Real prepared adapters preserve scalar/packed ownership and immutable bindings across captured replay. */
TEST_F(Test__CUDAGDNPaddedRealLength, DensePreparedRetainedInvocationOwnershipReplay)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "Prepared adapter selected native device");
    cuda_dense_compact_test::preparedInvocationReplay(cuda_ordinal_);
}


namespace
{
namespace cuda_dense_compact_test
{
/** @brief Capture the actual recurrence stage over canonical Tensor storage leases.
 * @param device Existing native participant selected by normal test startup.
 * @details Source/output tensors use TransferEngine physical storage, whereas
 * the cache-state fixture remains owned by the actual backend kernel binding.
 * Stage/workspace/reset operations reuse the same two captured executables.
 */
inline void retainedStageReplay(DeviceId device)
{
    auto context = IDeviceContext::create(device, 1);
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        auto *backend = getBackendFor(device);
        require(backend != nullptr, "Retained stage backend missing");
        void *const stream = ExplicitGPUStream(worker.defaultStream()).get();
        for (bool production_geometry : {false, true})
        for (int requests : {1, 3})
        for (auto terminal_kind : {GDNDenseCompactTerminalKind::DenseDestination,
            GDNDenseCompactTerminalKind::RetainedPrefixesOnly})
        {
            constexpr int heads = 2, keys = 1, dk = 128, rows = 9;
            // Keep the original tail geometry and additionally exercise the
            // served square matrix in this backend's actual cache orientation.
            const int dv = production_geometry ? 128 : 17;
            constexpr auto axes = Layout::KeyMajorValueContiguous;
            SCOPED_TRACE(::testing::Message() << "requests=" << requests << " DV=" << dv
                << " axes=" << int(axes) << " terminal=" << unsigned(terminal_kind));
            Cell cell(heads, keys, dk, dv, requests, rows, axes,
                true, false, false, {}, terminal_kind, 0);
            CudaStreamHandle publication_stream;
            CUDAGatedDeltaNet kernel(device.gpu_ordinal());
            kernel.setGPUStream(stream);
            FP32Tensor merged({requests * rows, cell.merged_stride}), alpha({requests * rows, heads}),
                beta({requests * rows, heads}), a({heads}), bias({heads}), output({requests * rows, heads * dv}), live_state({requests, heads * dk * dv});
            INT32Tensor lengths({requests}), choices({requests * 2});
            std::memcpy(live_state.raw_mutable_data(), cell.initial_host.data(), live_state.size_bytes());
            const std::array<std::pair<FP32Tensor *, const std::vector<float> *>, 5> sources{{
                {&merged, &cell.merged_host}, {&alpha, &cell.alpha_host}, {&beta, &cell.beta_host},
                {&a, &cell.a_host}, {&bias, &cell.bias_host}}};
            for (const auto &[tensor, authored] : sources)
                std::memcpy(tensor->raw_mutable_data(), authored->data(), tensor->size_bytes());
            std::fill_n(static_cast<float *>(output.raw_mutable_data()), output.numel(), Cell::Poison);
            std::fill_n(static_cast<int *>(lengths.raw_mutable_data()), lengths.numel(), rows);
            std::fill_n(static_cast<int *>(choices.raw_mutable_data()), choices.numel(), -1);
            for (TensorBase *tensor : std::array<TensorBase *, 9>{&merged, &alpha, &beta, &a, &bias, &output, &lengths, &choices, &live_state})
            {
                TransferEngine::prepareDeviceInput(tensor, device, stream);
                TransferEngine::requireDeviceInput(tensor, device, stream);
            }
            auto *const live_pointer = static_cast<float *>(live_state.gpu_data_ptr());
            require(kernel.bindDeviceState({live_pointer, cell.state_stride, nullptr, 0,
                live_pointer, live_state.numel(), requests, cell.axes}), "Retained stage kernel cache owner missing");
            GDNRecurrenceStage::Params params;
            params.device_id = device;
            params.gpu_retained_workspace = gdn_workspace::GPURetainedRecurrenceWorkspace(cell.geometry, terminal_kind);
            params.Q = params.K = params.V = &merged;
            params.alpha = &alpha; params.beta = &beta; params.A_log = &a; params.dt_bias = &bias;
            params.output = &output;
            params.seq_len = requests * rows; params.request_count = requests; params.request_seq_len = rows;
            params.request_seq_lens_device = static_cast<const int *>(lengths.gpu_data_ptr());
            params.n_heads = heads; params.n_k_heads = keys; params.d_k = dk; params.d_v = dv;
            params.global_v_head_offset = 0; params.use_qk_l2norm = true;
            params.layer_idx = 0; params.workspace_namespace = "grouped_mtp_verifier";
            params.speculative_state_slot_rows = requests * rows; params.kernel = &kernel;
            params.qkv_buffer_id = BufferId::GDN_RECURRENCE_IN;
            params.alpha_buffer_id = BufferId::GDN_ALPHA; params.beta_buffer_id = BufferId::GDN_BETA;
            params.output_buffer_id = BufferId::ATTN_OUTPUT;
            const auto footprint = params.gpu_retained_workspace->footprint();
            DeviceWorkspaceManager workspace(device, footprint.totalBytes() + 4096);
            GDNRecurrenceStage stage(params);
            const auto requirements = stage.getWorkspaceRequirements(requests * rows);
            require(requirements.buffers.size() == (terminal_kind == GDNDenseCompactTerminalKind::DenseDestination ? 2u : 1u) &&
                requirements.buffers.front().size_bytes == footprint.slot_bytes &&
                requirements.buffers.front().content_lifetime == WorkspaceContentLifetime::SerialGraphFamily,
                "Retained stage workspace deviates from its admitted contract");
            require(workspace.allocate(requirements), "Retained stage workspace allocation failed");
            stage.bindWorkspace(&workspace);
            stage.setGPUStream(stream);
            require(stage.graphLaunchPreparationPolicy() == GraphLaunchPreparationPolicy::CaptureOnly &&
                stage.supportsGraphCaptureAfterLaunchPreparation(), "Retained stage cold capture policy missing");
            require(stage.hasVerifierStateCapture() && !stage.isGraphCapturable(),
                "Retained stage declaration must not advertise an unprepared captured invocation");
            require(stage.prepareGraphLaunch(context.get(), stream) && stage.hasVerifierStateCapture() && stage.isGraphCapturable(),
                "Actual retained stage preparation failed");
            const auto identity = stage.graphCaptureVariantSignature();
            auto *retained = static_cast<float *>(workspace.getBuffer(requirements.buffers.front().name));
            float *terminal = terminal_kind == GDNDenseCompactTerminalKind::DenseDestination ?
                static_cast<float *>(workspace.getBuffer(requirements.buffers[1].name)) : nullptr;
            const auto retained_floats = workspace.getBufferSize(requirements.buffers.front().name) / sizeof(float);
            std::vector<float> poison(retained_floats, Cell::Poison), terminal_poison(cell.geometry.denseBankFloats(), Cell::Poison);
            std::vector<int> decisions(choices.numel(), -1);
            const auto contract = stage.bufferContract();
            require(contract.inputs.size() == 3 && contract.outputs.size() == 1 && contract.weight_tensors.size() == 2 &&
                contract.inputs[0].id == BufferId::GDN_RECURRENCE_IN && contract.inputs[1].id == BufferId::GDN_ALPHA &&
                contract.inputs[2].id == BufferId::GDN_BETA && contract.outputs[0].id == BufferId::ATTN_OUTPUT &&
                contract.weight_tensors[0] == &a && contract.weight_tensors[1] == &bias,
                "Actual retained stage contract changed its exact Tensor inventory");
            GraphCaptureDependencyLedger capture_ledger(device, stream, {
                GraphCaptureDependencyLedger::StagePlan{.stage_identity = &stage, .stage_name = "actual_retained_gdn_stage",
                    .external_inputs = {merged.transferStorageOwner(), alpha.transferStorageOwner(), beta.transferStorageOwner(),
                        a.transferStorageOwner(), bias.transferStorageOwner()},
                    .outputs = {output.transferStorageOwner()}}}, "actual retained GDN stage Tensor contract");
            std::unique_ptr<IGPUGraphCapture> capture, publication;
            struct Retirement final
            {
                IWorkerGPUContext &worker;
                IBackend *backend;
                DeviceId device;
                void *stream, *publication_stream, *ready;
                /** @brief Keep both streams, real ordering event and all later graph borrows alive. */
                Retirement(IWorkerGPUContext &w, IBackend *b, DeviceId d, void *s, void *p)
                    : worker(w), backend(b), device(d), stream(s), publication_stream(p), ready(w.createEvent()) {}
                /** @brief Test-only final join before graph/tensor/workspace/native owner release. */
                ~Retirement()
                {
                    if (!backend->synchronizeStream(stream, device.gpu_ordinal()) ||
                        !backend->synchronizeStream(publication_stream, device.gpu_ordinal())) std::terminate();
                    worker.destroyEvent(ready);
                }
            } retirement(worker, backend, device, stream, publication_stream.stream);
            const auto read = [&](const float *pointer, std::size_t count, void *exact_stream) {
                std::vector<float> result(count);
                Retirement join(worker, backend, device, exact_stream, exact_stream);
                require(backend->deviceToHost(result.data(), pointer, count * sizeof(float), device.gpu_ordinal(), exact_stream),
                    "Retained stage actual observation copy failed");
                require(backend->synchronizeStream(exact_stream, device.gpu_ordinal()), "Retained stage actual observation join failed");
                return result;
            };
            capture = worker.createGraphCapture(stream);
            {
                ScopedBackendGraphCapture recording(worker, *capture, "actual retained GDN stage", &capture_ledger);
                require(recording.begin(), "Retained stage capture begin failed");
                ScopedGraphCaptureStage active(&stage);
                require(stage.execute(context.get()), "Actual retained stage capture failed");
                TransferEngine::publishDeviceWrite(&output, device, stream);
                active.complete(); recording.finish();
            }
            publication = worker.createGraphCapture(publication_stream.stream);
            {
                ScopedBackendGraphCapture recording(worker, *publication, "actual retained GDN stage publication");
                require(recording.begin(), "Retained stage publication capture begin failed");
                ScopedGraphCaptureStage active(&stage);
                const std::span<const int> view(static_cast<const int *>(choices.gpu_data_ptr()), choices.numel());
                require(requests == 1 ? stage.restoreVerifierStateCaptureRowFromDeviceIndex(view, publication_stream.stream) :
                    stage.restoreVerifierStateCaptureRowsFromDeviceIndices(view, requests, 2, publication_stream.stream),
                    "Actual retained stage typed publication failed");
                active.complete(); recording.finish();
            }
            require(capture->instantiate() && publication->instantiate(), "Retained stage executables failed");

            const auto expected_plan = prepareCUDADenseRetentionCapturePlan(cell.geometry, cell.axes, terminal_kind,
                device.gpu_ordinal(), stream);
            const std::string expected_family = expected_plan.captureFunctionFamily();
            std::vector<GPUGraphKernelNodeInfo> capture_kernels, publication_kernels;
            std::string inspection_error;
            require(capture->inspectKernelNodes(capture_kernels, &inspection_error) && capture_kernels.size() == 1 &&
                capture_kernels.front().name_resolved && capture_kernels.front().name.find(expected_family) != std::string::npos &&
                capture_kernels.front().grid_x == unsigned(requests * heads) &&
                capture_kernels.front().grid_y == unsigned((dv + 31) / 32), "Actual stage capture native identity differs");
            require(publication->inspectKernelNodes(publication_kernels, &inspection_error) && publication_kernels.size() == 1 &&
                publication_kernels.front().name_resolved && publication_kernels.front().name.find("cuda_gdn_dense_compact_accepted_prefix_kernel") != std::string::npos &&
                publication_kernels.front().grid_x == unsigned(requests * heads) &&
                publication_kernels.front().grid_y == unsigned(cell.geometry.matrixTileCount(4096)),
                "Actual stage publication native identity differs");
            const auto capture_nodes = capture->nodeCount(), publication_nodes = publication->nodeCount();
            require(capture_nodes > 0 && publication_nodes > 0, "Retained stage omitted its native work");
            for (int replay = 0; replay < 4; ++replay)
            {
                stage.resetSessionStatePreservingCapturedReplay();
                require(stage.prepareGraphLaunch(context.get(), stream) && stage.hasVerifierStateCapture() &&
                    stage.graphCaptureVariantSignature() == identity && stage.getWorkspace() == &workspace &&
                    workspace.getBuffer(requirements.buffers.front().name) == retained,
                    "Request reset changed retained topology/physical leases");
                cell.reset();
                // The real output deliberately keeps prior replay bytes in
                // inactive rows. Seed only the oracle's logical output from
                // that actual pretransaction image; its own guards stay poison.
                // The unchanged native oracle then overwrites exactly its live rows.
                const auto output_before = read(static_cast<const float *>(output.gpu_data_ptr()), output.numel(), stream);
                std::vector<float> oracle_output_before(cell.oracle_output.count, Cell::Poison);
                std::copy(output_before.begin(), output_before.end(), oracle_output_before.begin());
                cell.oracle_output.copyFrom(oracle_output_before);
                require(backend->hostToDeviceOnStream(live_pointer, cell.initial_host.data(), live_state.size_bytes(),
                    device.gpu_ordinal(), stream), "Retained stage cache-owned request reset failed");
                for (int r = 0; r < requests; ++r)
                    cell.live_host[r] = replay == 2 ? 0 : replay == 1 ? (r == 0 ? 3 : r == 1 ? 0 : 8) : rows;
                cell.publishCounts(); cell.observation.finish();
                require(backend->hostToDeviceOnStream(lengths.gpu_data_ptr(), cell.live_host.data(), lengths.size_bytes(),
                    device.gpu_ordinal(), stream), "Retained stage authored live-count publication failed");
                require(backend->hostToDeviceOnStream(retained, poison.data(), poison.size() * sizeof(float),
                    device.gpu_ordinal(), stream), "Retained stage dirty arena publication failed");
                if (terminal) require(backend->hostToDeviceOnStream(terminal, terminal_poison.data(), terminal_poison.size() * sizeof(float),
                    device.gpu_ordinal(), stream), "Retained stage terminal sentinel failed");
                require(capture->launch(), "Retained stage replay failed");
                cell.dense->launch(cell.stream.stream); cell.observation.finish();
                require(worker.recordEventChecked(retirement.ready, stream) &&
                    worker.waitEventChecked(retirement.ready, publication_stream.stream), "Retained stage producer event edge failed");
                const auto shots = cell.snapshots.toHost();
                const auto retained_bytes = read(retained, retained_floats, stream);
                const auto terminal_bytes = terminal ? read(terminal, cell.geometry.denseBankFloats(), stream) : terminal_poison;
                const auto actual_output = read(static_cast<const float *>(output.gpu_data_ptr()), output.numel(), stream);
                cell.checkCapture(poison, shots, &retained_bytes, terminal ? &terminal_bytes : nullptr, workspace.id(), &actual_output);
                equal(read(live_pointer, live_state.numel(), stream),
                    std::vector<float>(cell.initial_host.begin(), cell.initial_host.begin() + live_state.numel()),
                    "Actual retained capture changed canonical cache state");
                auto previous = cell.initial_host;
                for (int selected : {9, 0, 4, 1, 8, 9})
                {
                    auto accepted = cell.live_host;
                    for (int r = 0; r < requests; ++r)
                    {
                        accepted[r] = std::min(selected, cell.live_host[r]);
                        decisions[2 * r] = accepted[r] ? r * rows + accepted[r] - 1 : -1;
                        decisions[2 * r + 1] = 0x12345678;
                    }
                    require(backend->hostToDeviceOnStream(choices.gpu_data_ptr(), decisions.data(), choices.size_bytes(),
                        device.gpu_ordinal(), stream), "Retained stage owned selection publication failed");
                    require(worker.recordEventChecked(retirement.ready, stream) &&
                        worker.waitEventChecked(retirement.ready, publication_stream.stream), "Retained stage selection event edge failed");
                    require(publication->launch(), "Retained stage accepted replay failed");
                    require(backend->synchronizeStream(publication_stream.stream, device.gpu_ordinal()), "Retained stage accepted observation join failed");
                    auto expected = cell.selectedState(shots, accepted);
                    for (int r = 0; r < requests; ++r)
                        if (!accepted[r]) std::copy_n(previous.begin() + std::size_t(r) * cell.state_stride,
                            cell.state_stride, expected.begin() + std::size_t(r) * cell.state_stride);
                    equal(read(live_pointer, live_state.numel(), publication_stream.stream),
                        std::vector<float>(expected.begin(), expected.begin() + live_state.numel()), "Actual stage flattened-row restore/NoOp");
                    equal(read(retained, retained_floats, publication_stream.stream), retained_bytes,
                        "Actual stage accepted publication changed immutable prior");
                    stage.clearVerifierStateCaptureBindingAfterPublication();
                    require(stage.hasVerifierStateCapture(), "Actual stage retired revised-prefix ownership");
                    cell.oracle_state.copyFrom(expected);
                    for (int r = 0; r < requests; ++r)
                    {
                        const auto row = std::size_t(requests * rows + r);
                        const auto next = [&](float *state, float *out) {
                            return launchCUDAPlanarRecurrence({DeviceRequestRowRanges::fullyActive(1), heads, dk, dv, true,
                                cell.query.ptr + row * heads * dk, cell.key.ptr + row * heads * dk, cell.value.ptr + row * heads * dv,
                                cell.alpha.ptr + row * heads, cell.beta.ptr + row * heads, cell.a.ptr, cell.bias.ptr,
                                out + std::size_t(r) * heads * dv, state + std::size_t(r) * cell.state_stride,
                                state + std::size_t(r) * cell.state_stride, nullptr, 0, 0, cell.axes}, publication_stream.stream);
                        };
                        require(next(live_pointer, cell.following.ptr) && next(cell.oracle_state.ptr, cell.oracle_following.ptr),
                            "Actual stage following native M1 failed");
                    }
                    require(backend->synchronizeStream(publication_stream.stream, device.gpu_ordinal()), "Actual stage following observation join failed");
                    equal(read(live_pointer, live_state.numel(), publication_stream.stream),
                        read(cell.oracle_state.ptr, live_state.numel(), publication_stream.stream), "Actual stage following state bytes");
                    equal(cell.following.toHost(), cell.oracle_following.toHost(), "Actual stage following output bytes");
                    previous = read(live_pointer, live_state.numel(), publication_stream.stream);
                }
                require(capture->nodeCount() == capture_nodes && publication->nodeCount() == publication_nodes,
                    "Actual stage request reset recaptured topology");
                for (const auto &[tensor, authored] : sources)
                    equal(read(static_cast<const float *>(tensor->gpu_data_ptr()), tensor->numel(), stream),
                        std::vector<float>(authored->begin(), authored->begin() + tensor->numel()), "Actual leased source bytes changed");
            }
        }
    }).get();
}
} // namespace cuda_dense_compact_test
} // namespace

/** @test Actual stage leases, capture-only preparation, publication and reset preserve native recurrence bytes. */
TEST_F(Test__CUDAGDNPaddedRealLength, DenseRetainedStageTensorLeaseReplay)
{
    SKIP_IF_NO_CUDA();
    cuda_dense_compact_test::retainedStageReplay(DeviceId::cuda(cuda_ordinal_));
}

namespace
{
namespace cuda_dense_compact_test
{
/** @brief Prove typed NoOp preserves both native destinations, including a deliberately stale
 * mirror.
 * @param prepared Shared helper branch; this CUDA test passes false and tests direct native
 * publication only. Actual CUDA prepared ownership is covered by its separate existing adapter
 * regression.
 * @details Original native dense snapshots are the byte oracle. Borrowed storage,
 * graphs, staging and banks all outlive the final retirement guard. */
inline void conditionalMirrorReplay(bool prepared)
{
    for (int requests : {1, 3})
        for (int dk : {64, 128})
            for (auto axes : {Layout::KeyMajorValueContiguous, Layout::ValueMajorKeyContiguous})
                for (auto kind : {GDNDenseAcceptedSelectionKind::AcceptedCounts,
                                  GDNDenseAcceptedSelectionKind::OwnRequestFlattenedRows,
                                  GDNDenseAcceptedSelectionKind::RequestTerminalLengths})
                {
                    Cell cell(2, 1, dk, dk == 64 ? 17 : 64, requests, 9, axes, true, false, false,
                              {}, GDNDenseCompactTerminalKind::RetainedPrefixesOnly, -5, kind, 3);
                    const auto state = std::size_t(cell.state_stride);
                    CudaFloatBuffer mirror_bank(state + Cell::Guard + 1, Cell::Poison);
                    CudaFloatBuffer primary(state + Cell::Guard, Cell::Poison);
                    CudaFloatBuffer secondary(2 * state + Cell::Guard, Cell::Poison);
                    CudaFloatBuffer packed(2 * state * requests + Cell::Guard, Cell::Poison);
                    CUDAGatedDeltaNet kernel(0);
                    WorkspaceRequirements requirements;
                    requirements.buffers.emplace_back("mirror_retained", cell.geometry.bankBytes(),
                                                      64);
                    DeviceWorkspaceManager workspace(DeviceId::cuda(0),
                                                     requirements.total_bytes() + 4096);
                    require(workspace.allocate(requirements),
                            "Mirror retained workspace allocation failed");
                    std::unique_ptr<IGDNPreparedRetainedInvocation> invocation;
                    if (prepared)
                    {
                        require(kernel.bindDeviceState({primary.ptr, int(state), secondary.ptr,
                                                        int(2 * state), packed.ptr,
                                                        2 * state * requests, requests, axes}),
                                "Mirror canonical binding failed");
                        GDNRetainedInvocationBinding binding{
                            cell.geometry,
                            GDNDenseCompactTerminalKind::RetainedPrefixesOnly,
                            {static_cast<float *>(workspace.getBuffer("mirror_retained")),
                             workspace.getBufferSize("mirror_retained") / sizeof(float)},
                            {},
                            workspace.id(),
                            cell.stream.stream};
                        invocation =
                            kernel.prepareRetainedInvocation(cell.input().sources, binding);
                    }
                    float *destination =
                        prepared ? invocation->publicationDestination() : cell.accepted_state.ptr;
                    float *mirror =
                        prepared ? (requests == 1 ? packed.ptr : primary.ptr) : mirror_bank.ptr + 1;
                    std::vector<float> expected(std::size_t(requests) * state, Cell::Poison);
                    std::vector<float> mirror_expected(state, Cell::Poison);
                    std::vector<float> poison(state, Cell::Poison);
                    std::unique_ptr<CudaCapturedGraph> capture, publication;
                    struct Retirement final
                    {
                        cudaStream_t stream;
                        /** @brief Retire graphs before every borrowed host or device owner is
                         * destroyed. */
                        ~Retirement()
                        {
                            if (cudaStreamSynchronize(stream) != cudaSuccess)
                                std::terminate();
                        }
                    } retirement{cell.stream.stream};
                    const auto read = [&](const float *pointer, std::size_t count)
                    {
                        std::vector<float> result(count);
                        Retirement join{cell.stream.stream};
                        checkCuda(cudaMemcpyAsync(result.data(), pointer, count * sizeof(float),
                                                  cudaMemcpyDeviceToHost, cell.stream.stream),
                                  "Mirror actual bank read");
                        checkCuda(cudaStreamSynchronize(cell.stream.stream),
                                  "Mirror terminal observation");
                        return result;
                    };
                    const auto write = [&](float *pointer, const std::vector<float> &values)
                    {
                        Retirement join{cell.stream.stream};
                        checkCuda(cudaMemcpyAsync(pointer, values.data(),
                                                  values.size() * sizeof(float),
                                                  cudaMemcpyHostToDevice, cell.stream.stream),
                                  "Mirror data publication");
                        checkCuda(cudaStreamSynchronize(cell.stream.stream),
                                  "Mirror host staging retirement");
                    };
                    EXPECT_THROW((void)GDNDenseCompactRequestZeroMirror::borrow(cell.geometry,
                                                                                mirror, state - 1),
                                 std::invalid_argument);
                    auto alias = GDNDenseCompactAcceptedPrefixInput{
                        cell.transaction(), cell.acceptedSelection(), cell.accepted_state.ptr,
                        GDNDenseCompactRequestZeroMirror::borrow(cell.geometry,
                                                                 cell.accepted_state.ptr, state)};
                    EXPECT_FALSE(validGDNDenseCompactAcceptedPrefixStorage(alias));
                    alias.request_zero_mirror = GDNDenseCompactRequestZeroMirror::borrow(
                        cell.geometry, cell.initial.ptr, state);
                    EXPECT_FALSE(validGDNDenseCompactAcceptedPrefixStorage(alias));
                    alias.request_zero_mirror = GDNDenseCompactRequestZeroMirror::borrow(
                        cell.geometry, cell.compact.ptr, state);
                    EXPECT_FALSE(validGDNDenseCompactAcceptedPrefixStorage(alias));
                    capture = std::make_unique<CudaCapturedGraph>(
                        cell.stream.stream,
                        [&]
                        {
                            return prepared ? invocation->capture()
                                            : launchCUDADenseCompactRecurrence(cell.input(),
                                                                               cell.stream.stream,
                                                                               cell.capture_plan);
                        });
                    publication = std::make_unique<CudaCapturedGraph>(
                        cell.stream.stream,
                        [&]
                        {
                            if (prepared)
                                return invocation->publish(cell.acceptedSelection(),
                                                           cell.stream.stream);
                            return launchCUDADenseCompactAcceptedPrefix(
                                {cell.transaction(), cell.acceptedSelection(), destination,
                                 GDNDenseCompactRequestZeroMirror::borrow(cell.geometry, mirror,
                                                                          state)},
                                cell.stream.stream);
                        });
                    // Authenticate the native family before decoding the complete optional view.
                    std::size_t count = 0;
                    checkCuda(cudaGraphGetNodes(publication->graph, nullptr, &count),
                              "Mirror publication node count");
                    require(count == 1,
                            "Conditional mirror must be part of the one publisher kernel");
                    cudaGraphNode_t node{};
                    checkCuda(cudaGraphGetNodes(publication->graph, &node, &count),
                              "Mirror publication node");
                    cudaKernelNodeParams params{};
                    checkCuda(cudaGraphKernelNodeGetParams(node, &params),
                              "Mirror publication arguments");
                    const char *name = nullptr;
                    checkCuda(cudaFuncGetName(&name, params.func), "Mirror publisher name");
                    require(name &&
                                std::string(name).find(
                                    "cuda_gdn_dense_compact_accepted_prefix_kernel") !=
                                    std::string::npos &&
                                params.kernelParams && params.kernelParams[0],
                            "Mirror publisher family/tuple missing");
                    auto actual = GDNDenseCompactAcceptedPrefixInput{
                        cell.transaction(), cell.acceptedSelection(), destination};
                    std::memcpy(&actual, params.kernelParams[0], sizeof(actual));
                    require(actual.destination_state == destination &&
                                actual.request_zero_mirror.destination() == mirror &&
                                actual.request_zero_mirror.availableFloats() == state &&
                                actual.selection.sameBinding(cell.acceptedSelection()),
                            "Conditional mirror capture identity differs");
                    for (int replay = 0; replay < 4; ++replay)
                    {
                        cell.reset();
                        for (int r = 0; r < requests; ++r)
                            cell.live_host[r] = replay == 2   ? 0
                                                : replay == 1 ? (r == 0   ? 3
                                                                 : r == 1 ? 0
                                                                          : 8)
                                                              : 9;
                        cell.publishCounts();
                        if (prepared)
                        {
                            write(destination,
                                  std::vector<float>(cell.initial_host.begin(),
                                                     cell.initial_host.begin() +
                                                         std::size_t(requests) * state));
                            expected.assign(cell.initial_host.begin(),
                                            cell.initial_host.begin() +
                                                std::size_t(requests) * state);
                        }
                        else
                        {
                            expected.assign(std::size_t(requests) * state, Cell::Poison);
                            write(destination, expected);
                        }
                        write(mirror, poison);
                        mirror_expected = poison;
                        capture->launch(cell.stream.stream);
                        cell.dense->launch(cell.stream.stream);
                        checkCuda(cudaStreamSynchronize(cell.stream.stream),
                                  "Mirror terminal observation");
                        const auto shots = cell.snapshots.toHost();
                        for (int revision = 0; revision < 4; ++revision)
                        {
                            const bool counts =
                                kind == GDNDenseAcceptedSelectionKind::AcceptedCounts;
                            // Poison a stale nonauthoritative mirror before a request-zero NoOp.
                            // Counts0 is not NoOp and must replace this poison with cp0.
                            if (revision == 1 || revision == 2)
                            {
                                write(mirror, poison);
                                mirror_expected = poison;
                            }
                            for (int r = 0; r < requests; ++r)
                            {
                                const bool skip =
                                    !counts && (cell.live_host[r] == 0 || revision == 2 ||
                                                (revision == 1 && (r % 2 == 0)));
                                const int accepted = revision == 2   ? 0
                                                     : revision == 1 ? cell.live_host[r] / 2
                                                                     : cell.live_host[r];
                                cell.accepted_host[std::size_t(r) * 3] =
                                    counts ? accepted
                                    : skip ? (kind == GDNDenseAcceptedSelectionKind::
                                                          OwnRequestFlattenedRows
                                                  ? -1
                                                  : 0)
                                    : kind == GDNDenseAcceptedSelectionKind::OwnRequestFlattenedRows
                                        ? r * cell.rows + accepted - 1
                                        : accepted;
                                if (!skip)
                                {
                                    const float *source =
                                        accepted
                                            ? shots.data() +
                                                  std::size_t(r * cell.rows + accepted - 1) * state
                                            : cell.initial_host.data() + std::size_t(r) * state;
                                    std::copy_n(source, state,
                                                expected.data() + std::size_t(r) * state);
                                    if (r == 0)
                                        std::copy_n(source, state, mirror_expected.begin());
                                }
                            }
                            cell.publishCounts();
                            publication->launch(cell.stream.stream);
                            checkCuda(cudaStreamSynchronize(cell.stream.stream),
                                      "Mirror terminal observation");
                            equal(read(destination, expected.size()), expected,
                                  "Conditional mirror actual accepted bank");
                            equal(read(mirror, state), mirror_expected,
                                  "Conditional mirror typed NoOp/Restore semantics");
                        }
                    }
                }
}
} // namespace cuda_dense_compact_test
} // namespace

/** @test Optional request-zero mirror is part of the validated publisher and preserves semantic
 * NoOp. */
TEST_F(Test__CUDAGDNPaddedRealLength, DenseCompactConditionalRequestZeroMirrorReplay)
{
    SKIP_IF_NO_CUDA();
    cuda_dense_compact_test::conditionalMirrorReplay(false);
}


namespace
{
namespace cuda_dense_compact_test
{
/** @brief Measure actual H32 verifier shape with every revised accepted-prefix boundary.
 * @param ordinal Actual fixture-selected CUDA device; canonical cold inventory selects scheduling.
 * @details Physical M4 and M16 correspond to fixed depths3 and15 plus the main-condition row.
 * Both arms start from the same raw post-convolution source; legacy includes native deinterleave.
 * Original native serial recurrence, actual timed endpoints and following M1 proofs are unchanged.
 * These22 cells are separate from all existing live48/representative/spectrum registrations.
 */
inline void runProductionH32PostConvolutionEvents(int ordinal)
{
    constexpr int SourceKeyHeads = 16;
    for (int rows : {4, 16})
        for (int accepted = 0; accepted <= rows; ++accepted)
            runCompleteEventCell(ordinal, 32, rows, accepted, accepted,
                                 EventAcceptanceSpectrum::ProductionH32M4M16EveryPrefix, rows,
                                 {GDNDenseRetentionMode::ExactCorrections, 0},
                                 GDNDenseCompactTerminalKind::RetainedPrefixesOnly,
                                 Layout::KeyMajorValueContiguous,
                                 CompleteEventInstrumentation::Canonical,
                                 CompleteEventInputBoundary::PostConvolutionMerged, SourceKeyHeads);
}
} // namespace cuda_dense_compact_test
} // namespace

/** @test Actual H32/M4+M16 complete post-convolution all-prefix economics; not whole-model
 * throughput. */
TEST_F(Test__CUDAGDNPaddedRealLength,
       DISABLED_DenseExactPostConvolutionProductionH32EveryPrefixEvents)
{
    SKIP_IF_NO_CUDA();
    checkCuda(cudaSetDevice(cuda_ordinal_), "Production H32 post-convolution selected device");
    cuda_dense_compact_test::runProductionH32PostConvolutionEvents(cuda_ordinal_);
}

#endif
