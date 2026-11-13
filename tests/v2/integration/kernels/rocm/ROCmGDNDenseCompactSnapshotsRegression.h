/**
 * @file ROCmGDNDenseCompactSnapshotsRegression.h
 * @brief Captured exact-dense correction retention and accepted-prefix regression.
 *
 * Reuses the padded-GDN fixture's HIP device, graph, transfer and retirement owners.
 * Original planar dense kernels supply independent native snapshots and M1
 * outputs. The compact candidate must preserve every byte, including revised
 * acceptance after the original input bank has changed. No model policy is
 * selected by these tests; the retained correction format is transaction-local.
 * The original optimized raw M1 route remains the following-row timed control;
 * convolution/prepared-QK production timing remains a separate existing suite.
 * Explicit post-convolution events additionally capture original native
 * deinterleave on the legacy arm; host-expanded arrays remain oracle-only.
 * Disabled native events include retention, accepted publication and following
 * decode over evolving banks, with final timed bytes checked before reset.
 */
#pragma once
#ifdef HAVE_ROCM
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
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
namespace rocm_dense_compact_test
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

/** @brief Exact native transaction and original dense controls with retained graphs.
 * @details Captures are constructed only after every buffer exists. The existing
 * always-armed retirement owner is last-declared and joins before graph/buffer
 * destruction, including exceptions after an earlier diagnostic finish(). */
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
    HipStreamHandle stream;
    HipFloatBuffer merged, query, key, value, alpha, beta, a, bias, initial,
        terminal, compact, output, oracle_state, oracle_output, snapshots,
        oracle_terminal, accepted_state, following, oracle_following;
    HipIntBuffer live, accepted;
    std::unique_ptr<HipCapturedGraph> capture, dense, publication, publication_alias;
    gdn_merged_extended::Cell::Retirement retirement;

    /** @brief Author identical raw inputs and capture candidate/control programs.
     * @param h Local value heads, including small-grid and ordinary model shapes.
     * @param kh Encoded Q/K heads; source mapping may repeat unevenly.
     * @param key_width Original ROCm64/128 width.
     * @param value_width Exact width including incomplete column tiles.
     * @param r Independent request partitions.
     * @param m Physical verifier rows per request.
     * @param layout Immutable Dense/checkpoint axes.
     * @param norm Original Q/K normalization choice.
     * @param alias Required terminal may exactly equal original Dense input.
     * @param edge Include zero/subnormal/infinity/NaN arithmetic witnesses.
     * @param program Immutable retained policy; absent preserves fixed DFC1 controls.
     * @param terminal_output Explicit terminal program; existing calls remain DenseDestination.
     * @param source_head_offset Frozen modulo head origin; old fixtures retain -5, production-shaped events use0.
     * @param selected_kind Existing metadata meaning, Counts by default.
     * @param selected_stride Positive metadata element stride. */
    Cell(int h, int kh, int key_width, int value_width, int r, int m, Layout layout,
         bool norm = true, bool alias = false, bool edge = false,
         GDNDenseRetentionProgram program = {},
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
          retirement{stream.stream}
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
        capture = std::make_unique<HipCapturedGraph>(stream.stream, [&] {
            return launchROCmDenseCompactRecurrence(input(), stream.stream);
        });
        dense = std::make_unique<HipCapturedGraph>(stream.stream, [&] {
            return launchROCmPlanarRecurrence(denseInput(), stream.stream);
        });
        publication = std::make_unique<HipCapturedGraph>(stream.stream, [&] {
            return launchROCmDenseCompactAcceptedPrefix({transaction(), acceptedSelection(), accepted_state.ptr}, stream.stream);
        });
        publication_alias = std::make_unique<HipCapturedGraph>(stream.stream, [&] {
            return launchROCmDenseCompactAcceptedPrefix({transaction(), acceptedSelection(), initial.ptr}, stream.stream);
        });
        inspect(*capture, true);
        inspect(*publication, false);
    }

    /** @brief Observe terminal test completion before downloads/reset; never inside capture/events. */
    void finish() const { checkHip(hipStreamSynchronize(stream.stream), "Dense compact terminal fixture join"); }

    /** @return Single checked compact identity; no alternate sizing calculation. */
    GDNDenseCompactTransaction transaction() const
    {
        return {geometry, initial.ptr, compact.ptr, geometry.bankFloats(), Generation,
            axes, GDNDenseCompactArithmetic::ROCmFourPartFusedUpdateV1, geometry.encoding()};
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
    void inspect(const HipCapturedGraph &graph, bool capturing, float *accepted_destination = nullptr) const
    {
        std::size_t count = 0;
        checkHip(hipGraphGetNodes(graph.graph, nullptr, &count), "dense compact node count");
        require(count == 1, "Dense compact entrypoint must be one kernel");
        hipGraphNode_t node{};
        checkHip(hipGraphGetNodes(graph.graph, &node, &count), "dense compact node");
        hipGraphNodeType kind{};
        checkHip(hipGraphNodeGetType(node, &kind), "dense compact node kind");
        require(kind == hipGraphNodeTypeKernel, "Dense compact graph contains nonkernel work");
        hipKernelNodeParams p{};
        checkHip(hipGraphKernelNodeGetParams(node, &p), "dense compact arguments");
        const char *name = nullptr;
        name = hipKernelNameRefByPtr(p.func, stream.stream);
        require(name && std::string(name).find(capturing ? (terminal_kind == GDNDenseCompactTerminalKind::DenseDestination ?
                "rocm_gdn_dense_compact_capture_kernel" : "rocm_gdn_retained_prefixes_capture_kernel") :
            "rocm_gdn_dense_compact_accepted_prefix_kernel") != std::string::npos,
            "Dense compact graph selected foreign native function");
        require(p.kernelParams && p.kernelParams[0], "Dense compact argument absent");
        require(p.gridDim.x == unsigned(requests * heads) && p.gridDim.z == 1 &&
            p.gridDim.y == unsigned(capturing ? (dv + 31) / 32 : geometry.matrixTileCount(4096)) &&
            p.blockDim.x == unsigned(capturing ? 128 : 256) && p.blockDim.y == 1 && p.blockDim.z == 1 &&
            p.sharedMemBytes == unsigned(capturing ? (dk == 64 ? 1552 : 7184) : 0),
            "Dense compact launch geometry differs");
        if (capturing)
        {
            auto actual = input();
            std::memcpy(&actual, p.kernelParams[0], sizeof(actual));
            require(actual.transaction.storage == compact.ptr && actual.transaction.initial_state == initial.ptr &&
                actual.transaction.generation == Generation && actual.transaction.layout == axes &&
                actual.transaction.arithmetic == GDNDenseCompactArithmetic::ROCmFourPartFusedUpdateV1 &&
                actual.transaction.storage_floats == geometry.bankFloats() &&
                actual.transaction.geometry.heads() == heads && actual.transaction.geometry.keyWidth() == dk &&
                actual.transaction.geometry.valueWidth() == dv && actual.transaction.geometry.requests() == requests &&
                actual.transaction.geometry.rowsPerRequest() == rows &&
                actual.transaction.geometry.bankFloats() == geometry.bankFloats() &&
                actual.transaction.encoding == geometry.encoding() &&
                actual.transaction.geometry.retentionProgram().mode == geometry.retentionProgram().mode &&
                actual.transaction.geometry.retentionProgram().compact_min_live_rows ==
                    geometry.retentionProgram().compact_min_live_rows &&
                actual.sources.rows.lengthOwner() == live.ptr && actual.sources.rows.requests() == requests &&
                actual.sources.rows.rowsPerRequest() == rows && actual.sources.key_heads == keys &&
                actual.sources.value_heads == heads && actual.sources.key_width == dk && actual.sources.value_width == dv &&
                actual.sources.global_value_offset == offset && actual.sources.use_qk_l2norm == normalize &&
                actual.sources.recurrence_layout == axes && actual.sources.merged == merged.ptr &&
                actual.sources.alpha == alpha.ptr && actual.sources.beta_raw == beta.ptr &&
                actual.sources.A_log == a.ptr && actual.sources.dt_bias == bias.ptr &&
                !actual.sources.initial_state && !actual.sources.updated_state && !actual.sources.snapshots &&
                !actual.sources.snapshot_stride_floats && !actual.sources.max_snapshot_rows &&
                actual.sources.output == output.ptr && actual.terminal.kind() == terminal_kind && actual.terminal.destination() == input().terminal.destination(),
                "Dense compact captured ownership differs");
        }
        else
        {
            GDNDenseCompactAcceptedPrefixInput actual{transaction(), acceptedSelection(), accepted_state.ptr};
            std::memcpy(&actual, p.kernelParams[0], sizeof(actual));
            require(actual.transaction.storage == compact.ptr && actual.transaction.generation == Generation &&
                actual.transaction.arithmetic == GDNDenseCompactArithmetic::ROCmFourPartFusedUpdateV1 &&
                actual.transaction.layout == axes && actual.transaction.storage_floats == geometry.bankFloats() &&
                actual.transaction.encoding == geometry.encoding() &&
                actual.transaction.geometry.retentionProgram().mode == geometry.retentionProgram().mode &&
                actual.transaction.geometry.retentionProgram().compact_min_live_rows ==
                    geometry.retentionProgram().compact_min_live_rows &&
                actual.selection.sameBinding(acceptedSelection()) &&
                actual.destination_state == (accepted_destination ? accepted_destination : accepted_state.ptr),
                "Dense compact accepted ownership differs");
        }
        hipFuncAttributes resources{};
        checkHip(hipFuncGetAttributes(&resources, p.func), "dense compact resources");
        require(resources.localSizeBytes == 0, "Dense compact native private memory");
    }

    /** @brief Publish mutable request counts on the exact stream; vectors live through retirement. */
    void publishCounts()
    {
        checkHip(hipMemcpyAsync(live.ptr, live_host.data(), requests * sizeof(int),
            hipMemcpyHostToDevice, stream.stream), "dense compact live counts");
        checkHip(hipMemcpyAsync(accepted.ptr, accepted_host.data(), accepted_host.size() * sizeof(int),
            hipMemcpyHostToDevice, stream.stream), "dense compact accepted counts");
    }

    /** @brief Reset only data; graph topology and every embedded pointer stay retained. */
    void reset()
    {
        finish();
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
            std::uint32_t(axes), std::uint32_t(GDNDenseCompactArithmetic::ROCmFourPartFusedUpdateV1),
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
                return launchROCmPlanarRecurrence({DeviceRequestRowRanges::fullyActive(1), heads, dk, dv, normalize,
                    query.ptr + std::size_t(row) * heads * dk, key.ptr + std::size_t(row) * heads * dk,
                    value.ptr + std::size_t(row) * heads * dv, alpha.ptr + std::size_t(row) * heads,
                    beta.ptr + std::size_t(row) * heads, a.ptr, bias.ptr, out + std::size_t(request) * heads * dv,
                    state + std::size_t(request) * state_stride, state + std::size_t(request) * state_stride,
                    nullptr, 0, 0, axes}, stream.stream);
            };
            require(launch(accepted_state.ptr, following.ptr), "Dense compact next row rejected");
            require(launch(oracle_state.ptr, oracle_following.ptr), "Dense oracle next row rejected");
        }
        finish();
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
                require(launchROCmPlanarRecurrence({DeviceRequestRowRanges::fullyActive(1), heads, dk, dv, normalize,
                    query.ptr + row * heads * dk, key.ptr + row * heads * dk, value.ptr + row * heads * dv,
                    alpha.ptr + row * heads, beta.ptr + row * heads, a.ptr, bias.ptr,
                    oracle_output.ptr + row * heads * dv, state, state, nullptr, 0, 0, axes}, stream.stream),
                    "Dense serial oracle launch rejected");
                finish();
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
            publishCounts(); finish();
            const auto before = compact.toHost();
            capture->launch(stream.stream); dense->launch(stream.stream); finish();
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
                finish();
                const auto expected = selectedState(shots, accepted_host);
                equal(accepted_state.toHost(), expected, "dense compact accepted prefix");
                equal(initial.toHost(), expected, "dense compact revised in-place acceptance");
                checkFollowing(expected);
            }
            equal(compact.toHost(), retained, "dense compact retained source changed by publication");
        }
        std::cout << "ROCM_GDN_DENSE_COMPACT_BYTES heads=" << heads << " dk=" << dk << " dv=" << dv
            << " requests=" << requests << " rows=" << rows << " axes=" << int(axes)
            << " normalize=" << normalize << " terminal_alias=" << terminal_alias
            << retentionMarkerFields(geometry.retentionProgram()) << terminalMarkerFields(terminal_kind)
            << " dirty_replays=" << replays << " all_acceptances=1 serial_rows=1 next_row=1 passed=1\n";
    }
};
} // namespace rocm_dense_compact_test
} // namespace

TEST(Test__ROCmGDNPaddedRealLength, DenseCompactExactAcceptedPrefixes)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "dense compact selected device");
    using namespace rocm_dense_compact_test;
    for (int dk : {64, 128})
        for (auto axes : {Layout::KeyMajorValueContiguous, Layout::ValueMajorKeyContiguous})
            for (int rows = 1; rows <= 16; ++rows)
            {
                SCOPED_TRACE(::testing::Message() << "DK=" << dk << " M=" << rows << " axes=" << int(axes));
                Cell(3, 2, dk, rows % 2 ? 17 : 65, 1, rows, axes, rows % 3 != 0, rows % 4 == 0).run();
            }
}

TEST(Test__ROCmGDNPaddedRealLength, DenseCompactRaggedAndNonfinite)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "dense compact selected device");
    using namespace rocm_dense_compact_test;
    for (int dk : {64, 128})
        for (auto axes : {Layout::KeyMajorValueContiguous, Layout::ValueMajorKeyContiguous})
        {
            Cell(3, 4, dk, 129, 3, 31, axes, true, false).run();
            Cell(3, 4, dk, 33, 3, 9, axes, false, true, true).run();
            Cell(3, 2, dk, 1, 1, 3, axes, false).run();
            Cell(3, 2, dk, 512, 1, 8, axes, true).run();
        }
}

TEST(Test__ROCmGDNPaddedRealLength, DenseCompactAdmission)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "dense compact selected device");
    using namespace rocm_dense_compact_test;
    Cell cell(3, 2, 128, 65, 3, 9, Layout::KeyMajorValueContiguous);
    const auto original = cell.input();
    EXPECT_FALSE(launchROCmDenseCompactRecurrence(original, nullptr));
    EXPECT_FALSE(launchROCmDenseCompactRecurrence(original, reinterpret_cast<void *>(std::uintptr_t(1))));
    EXPECT_FALSE(launchROCmDenseCompactRecurrence(original, reinterpret_cast<void *>(std::uintptr_t(2))));
    for (int fault = 0; fault < 8; ++fault)
    {
        auto bad = original;
        if (fault == 0) bad.transaction.storage_floats = bad.transaction.geometry.bankFloats() - 1;
        if (fault == 1) bad.transaction.generation = 0;
        if (fault == 2) bad.sources.snapshots = cell.snapshots.ptr;
        if (fault == 3) bad.sources.output = cell.initial.ptr;
        if (fault == 4) bad.transaction.layout = static_cast<Layout>(255);
        if (fault == 5) bad.terminal = GDNDenseCompactTerminalOutput::denseDestination(cell.initial.ptr + 1);
        if (fault == 6) bad.transaction.arithmetic = GDNDenseCompactArithmetic::CUDAEightPartFusedUpdateV1;
        if (fault == 7) ++bad.transaction.storage;
        EXPECT_FALSE(launchROCmDenseCompactRecurrence(bad, cell.stream.stream));
    }
    auto selected = GDNDenseCompactAcceptedPrefixInput{cell.transaction(), cell.acceptedSelection(), cell.accepted_state.ptr};
    EXPECT_FALSE(launchROCmDenseCompactAcceptedPrefix(selected, nullptr));
    EXPECT_FALSE(launchROCmDenseCompactAcceptedPrefix(selected, reinterpret_cast<void *>(std::uintptr_t(1))));
    EXPECT_FALSE(launchROCmDenseCompactAcceptedPrefix(selected, reinterpret_cast<void *>(std::uintptr_t(2))));
    selected.transaction.arithmetic = GDNDenseCompactArithmetic::CUDAEightPartFusedUpdateV1;
    EXPECT_FALSE(launchROCmDenseCompactAcceptedPrefix(selected, cell.stream.stream));
    selected.transaction = cell.transaction();
    selected.destination_state = cell.initial.ptr + 1;
    EXPECT_FALSE(launchROCmDenseCompactAcceptedPrefix(selected, cell.stream.stream));
    cell.finish();
    equal(cell.initial.toHost(), cell.initial_host, "Rejected dense compact input untouched");
    equal(cell.compact.toHost(), std::vector<float>(cell.compact.count, Cell::Poison), "Rejected dense compact output untouched");
}

namespace
{
namespace rocm_dense_compact_test
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
    HipFloatBuffer storage; ///< One fixed native output allocation, never host-expanded input.

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
        return rocmGDN_deinterleave_qkv(cell_.merged.ptr, query(), key(), value(),
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

/** @brief Enqueue the original optimized raw-merged M1, consuming the actual live bank.
 * @param cell Existing raw operands and independent complete bank owners.
 * @param state Actual evolving arm bank, never replaced by oracle data.
 * @param output Actual following-row destination with its existing guard.
 * @details ValueMajor uses the existing scalar/vector ordinary kernel. This is
 * post-convolution raw recurrence, not an unpriced prepared-QK experiment.
 * @return Original native raw M1 admission/enqueue status. */
inline bool eventFollowing(Cell &cell, float *state, float *output)
{
    const auto row = std::size_t(cell.rows);
    return launchROCmMergedRecurrence({DeviceRequestRowRanges::fullyActive(1),
        cell.keys, cell.heads, cell.dk, cell.dv, cell.offset, cell.normalize,
        cell.merged.ptr + row * cell.merged_stride,
        cell.alpha.ptr + row * cell.heads, cell.beta.ptr + row * cell.heads,
        cell.a.ptr, cell.bias.ptr, output, state, state, nullptr, 0, 0, cell.axes}, cell.stream.stream);
}

/** @brief Select the accepted native snapshot with the original production copy kernel.
 * @details accepted0 leaves the unchanged input bank untouched. Full-row snapshot
 * capture never writes that bank, so this exactly matches compact cp0 restoration.
 * @param cell Existing counted acceptance and native full-snapshot storage.
 * @param state Arm's live bank at this transaction's actual initial state.
 * @param ordinal Authored device; never selects an implicit/default stream.
 * @return Original native counted-copy admission/enqueue status. */
inline bool eventAcceptedCopy(Cell &cell, float *state, int ordinal)
{
    return rocmGDN_gpu_publish_capture_terminal_rows_from_device_lengths(
        state, state, cell.snapshots.ptr, cell.accepted.ptr, 1, cell.rows, cell.rows,
        cell.state_stride, ordinal, cell.stream.stream);
}

/** @return One actual kernel's authenticated captured launch for schedule comparison.
 * @param graph Existing one-operation capture with typed inputs/owned lifetime. */
inline hipKernelNodeParams eventLeaf(const HipCapturedGraph &graph)
{
    std::size_t count = 0;
    checkHip(hipGraphGetNodes(graph.graph, nullptr, &count), "dense events leaf count");
    require(count == 1, "Dense events leaf is not one native kernel");
    hipGraphNode_t node{};
    checkHip(hipGraphGetNodes(graph.graph, &node, &count), "dense events leaf");
    hipGraphNodeType kind{};
    checkHip(hipGraphNodeGetType(node, &kind), "dense events leaf kind");
    require(kind == hipGraphNodeTypeKernel, "Dense events leaf contains nonkernel work");
    hipKernelNodeParams result{};
    checkHip(hipGraphKernelNodeGetParams(node, &result), "dense events leaf args");
    return result;
}

/** @brief Authenticate the actual preparation leaf's complete production tuple.
 * @param cell Existing original source and count geometry.
 * @param preparation Fixed scratch whose pointer layout the native stage consumes.
 * @param graph Actual one-node production deinterleave capture. */
inline void inspectEventPreparation(const Cell &cell, const EventPlanarPreparation &preparation,
    const HipCapturedGraph &graph)
{
    const auto p = eventLeaf(graph);
    const char *name = nullptr;
    name = hipKernelNameRefByPtr(p.func, cell.stream.stream);
    require(name && std::string(name).find("rocm_gdn_deinterleave_qkv_kernel") != std::string::npos,
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
 * @details Native arguments retain the same physical banks on every iteration;
 * pointer/coordinate values are compared against the corresponding owned leaf.
 */
inline void inspectEventSchedule(const Cell &cell, const HipCapturedGraph &graph,
    const std::array<hipKernelNodeParams, 3> &leaves, int repeats,
    const hipKernelNodeParams *preparation = nullptr)
{
    std::size_t count = 0, roots = 0;
    checkHip(hipGraphGetNodes(graph.graph, nullptr, &count), "dense events schedule count");
    checkHip(hipGraphGetRootNodes(graph.graph, nullptr, &roots), "dense events roots");
    const int roles = preparation ? 4 : 3;
    require(count == std::size_t(roles * repeats) && roots == 1,
        "Dense events complete transaction count/root differs");
    hipGraphNode_t node{};
    checkHip(hipGraphGetRootNodes(graph.graph, &node, &roots), "dense events first node");
    for (int index = 0; index < roles * repeats; ++index)
    {
        hipGraphNodeType kind{};
        checkHip(hipGraphNodeGetType(node, &kind), "dense events node type");
        require(kind == hipGraphNodeTypeKernel, "Dense events graph contains nonkernel work");
        hipKernelNodeParams actual{};
        checkHip(hipGraphKernelNodeGetParams(node, &actual), "dense events node params");
        const int role = index % roles;
        const auto &expected = preparation && role == 0 ? *preparation : leaves[role - (preparation ? 1 : 0)];
        require(actual.func == expected.func && actual.gridDim.x == expected.gridDim.x &&
            actual.gridDim.y == expected.gridDim.y && actual.gridDim.z == expected.gridDim.z &&
            actual.blockDim.x == expected.blockDim.x && actual.blockDim.y == expected.blockDim.y &&
            actual.blockDim.z == expected.blockDim.z && actual.sharedMemBytes == expected.sharedMemBytes &&
            actual.kernelParams && expected.kernelParams,
            "Dense events ordered native function/geometry differs");
        const char *name = nullptr;
        name = hipKernelNameRefByPtr(actual.func, cell.stream.stream);
        require(name != nullptr, "Dense events native function identity absent");
        const std::string family(name);
        if (family.find("rocm_gdn_deinterleave_qkv_kernel") != std::string::npos)
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
            family.find("retained_prefixes_capture_kernel") != std::string::npos)
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
                a.transaction.encoding == b.transaction.encoding &&
                a.transaction.geometry.retentionProgram().mode == b.transaction.geometry.retentionProgram().mode &&
                a.transaction.geometry.retentionProgram().compact_min_live_rows ==
                    b.transaction.geometry.retentionProgram().compact_min_live_rows &&
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
                a.transaction.encoding == b.transaction.encoding &&
                a.transaction.geometry.retentionProgram().mode == b.transaction.geometry.retentionProgram().mode &&
                a.transaction.geometry.retentionProgram().compact_min_live_rows ==
                    b.transaction.geometry.retentionProgram().compact_min_live_rows,
                "Dense events compact publication banks differ");
        }
        else
        {
            // Authenticate known legacy families BEFORE reading their pointer tuple.
            const bool copy = family.find("rocm_gdn_publish_capture_rows_from_device_indices_kernel") != std::string::npos;
            const bool dense = family.find("rocm_gdn_recurrent_step_kernel") != std::string::npos ||
                family.find("rocm_gdn_value_major_recurrent_step_kernel") != std::string::npos ||
                family.find("rocm_gdn_value_major_scalar_kernel") != std::string::npos;
            require(copy || dense, "Dense events selected an unknown legacy ABI");
            for (int arg = 0; arg < (copy ? 4 : 10); ++arg)
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
                for (int argument = 4; argument <= 8; ++argument) same_argument(argument, sizeof(int));
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
                if (family.find("rocm_gdn_value_major_") != std::string::npos)
                {
                    same_argument(20, sizeof(int));
                    same_argument(21, sizeof(int));
                }
            }
        }
        std::size_t successors = 0;
        checkHip(hipGraphNodeGetDependentNodes(node, nullptr, &successors), "dense events successor count");
        require(successors == unsigned(index + 1 < roles * repeats), "Dense events graph is not an exact linear chain");
        if (successors)
        {
            hipGraphNode_t next{};
            checkHip(hipGraphNodeGetDependentNodes(node, &next, &successors), "dense events next node");
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
            checkHip(hipMemsetAsync(oracle.output.ptr, 0,
                oracle.geometry.outputFloats() * sizeof(float), oracle.stream.stream),
                "Dense event oracle inactive output zero");
        checkHip(hipMemcpyAsync(oracle.oracle_state.ptr, oracle.initial.ptr,
            oracle.geometry.denseBankBytes(), hipMemcpyDeviceToDevice, oracle.stream.stream), "dense events oracle initial");
        for (int row = 0; row < live; ++row)
        {
            const auto index = std::size_t(row);
            require(launchROCmPlanarRecurrence({DeviceRequestRowRanges::fullyActive(1),
                oracle.heads, oracle.dk, oracle.dv, oracle.normalize,
                oracle.query.ptr + index * oracle.heads * oracle.dk,
                oracle.key.ptr + index * oracle.heads * oracle.dk,
                oracle.value.ptr + index * oracle.heads * oracle.dv,
                oracle.alpha.ptr + index * oracle.heads, oracle.beta.ptr + index * oracle.heads,
                oracle.a.ptr, oracle.bias.ptr, oracle.output.ptr + index * oracle.heads * oracle.dv,
                oracle.oracle_state.ptr, oracle.oracle_state.ptr, nullptr, 0, 0, oracle.axes}, oracle.stream.stream),
                "Dense events original serial row rejected");
            checkHip(hipMemcpyAsync(oracle.snapshots.ptr + index * oracle.state_stride, oracle.oracle_state.ptr,
                oracle.geometry.denseBankBytes(), hipMemcpyDeviceToDevice, oracle.stream.stream), "dense events oracle snapshot");
        }
        require(eventAcceptedCopy(oracle, oracle.initial.ptr, ordinal), "Dense events oracle accepted copy rejected");
        require(eventFollowing(oracle, oracle.initial.ptr, oracle.following.ptr), "Dense events oracle following rejected");
    }
    oracle.finish();
}

/** @brief Closed test-only acceptance schedules; neither is a production policy. */
enum class EventAcceptanceSpectrum
{
    Representative, ///< Original zero/middle/full24-cell screen, unchanged marker schema.
    EveryPrefixM8M16, ///< Explicit A0..M52-cell screen including worst suffix7.
    PhysicalM16LiveCounts ///< Physical16 and independent device-counted live prefix.
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
    return "";
}

/** @brief Time one complete evolving-state cell with actual final bytes checked before reset.
 * @param ordinal Actual fixture-owned ROCm participant.
 * @param heads Authored local value heads12/24.
 * @param rows Physical verifier capacity, independent of accepted rows.
 * @param accepted Device-authored test count A0..M, immutable within this captured graph.
 * @param selection Original role ordinal, or A itself in the explicit every-prefix spectrum.
 * @param spectrum Exact test-only marker identity, not a serving selector.
 * @param live_rows Active prefix, or -1 to preserve the original fully-live schedule.
 * @param program Explicit retained candidate; absent preserves existing fixed controls.
 * @param terminal_kind Explicit candidate terminal contract; legacy filters retain DenseDestination.
 * @param boundary Explicit source boundary; old calls retain pre-expanded planar controls.
 * @details Both arms include64 repeated verifier/publication/following transactions.
 * Existing stream, byte, graph, terminal and original serial-oracle proofs remain identical.
 * Planar full-snapshot versus merged exact-compact source scope remains explicit.
 */
inline void runCompleteEventCell(int ordinal, int heads, int rows, int accepted, int selection,
    EventAcceptanceSpectrum spectrum, int live_rows = -1, GDNDenseRetentionProgram program = {},
    GDNDenseCompactTerminalKind terminal_kind = GDNDenseCompactTerminalKind::DenseDestination,
    CompleteEventInputBoundary boundary = CompleteEventInputBoundary::PreExpandedPlanarControl)
{
    constexpr int Repeats = 64, Warmups = 5, Samples = 12;
    require(boundary == CompleteEventInputBoundary::PreExpandedPlanarControl ||
        boundary == CompleteEventInputBoundary::PostConvolutionMerged, "Complete event input boundary invalid");
    const bool postconv = boundary == CompleteEventInputBoundary::PostConvolutionMerged;
    const int source_offset = postconv ? 0 : -5;
    Cell cell(heads, heads / 3, 128, 128, 1, rows, Layout::ValueMajorKeyContiguous, true, false, false, program, terminal_kind, source_offset);
    Cell oracle(heads, heads / 3, 128, 128, 1, rows, Layout::ValueMajorKeyContiguous, true, false, false, {}, GDNDenseCompactTerminalKind::DenseDestination, source_offset);
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
    cell.finish(); oracle.finish();
    runEventOracle(oracle, Repeats, ordinal, live);
    const auto expected_state = oracle.initial.toHost();
    const auto expected_group_output = oracle.output.toHost();
    const auto expected_following = oracle.following.toHost();
    const auto expected_full_terminal = oracle.oracle_state.toHost();
    const auto expected_snapshots = oracle.snapshots.toHost();
    std::unique_ptr<EventPlanarPreparation> preparation;
    if (postconv) preparation = std::make_unique<EventPlanarPreparation>(cell, ordinal);
    const std::string operand_fields = postconv ?
        " state_layout=value_major_key_contiguous source_layout=merged_raw_vs_merged_raw dense_input=merged_raw compact_input=merged_raw input_layouts_differ=0 legacy_preparation=native_deinterleave candidate_preparation=direct_merged common_convolution_excluded=1 continuation=raw_merged_m1"
        " legacy_deinterleave_nodes=64 legacy_kernel_nodes=256 candidate_kernel_nodes=192 source_head_offset=0 key_repeat=modulo" :
        " source_layout=planar_full_vs_merged_compact dense_input=expanded_planar compact_input=merged_raw input_layouts_differ=1";
    const char *marker_prefix = postconv ? "ROCM_GDN_DENSE_POST_CONV_" : "ROCM_GDN_DENSE_COMPACT_";
    HipCapturedGraph dense_copy(cell.stream.stream, [&] {
        return eventAcceptedCopy(cell, cell.oracle_state.ptr, ordinal);
    });
    HipCapturedGraph compact_next(cell.stream.stream, [&] {
        return eventFollowing(cell, cell.initial.ptr, cell.following.ptr);
    });
    HipCapturedGraph dense_next(cell.stream.stream, [&] {
        return eventFollowing(cell, cell.oracle_state.ptr, cell.oracle_following.ptr);
    });
    std::unique_ptr<HipCapturedGraph> preparation_leaf, prepared_dense_leaf;
    hipKernelNodeParams preparation_params{};
    if (preparation)
    {
        preparation_leaf = std::make_unique<HipCapturedGraph>(cell.stream.stream, [&] { return preparation->enqueue(); });
        prepared_dense_leaf = std::make_unique<HipCapturedGraph>(cell.stream.stream, [&] {
            return launchROCmPlanarRecurrence(preparation->input(), cell.stream.stream);
        });
        inspectEventPreparation(cell, *preparation, *preparation_leaf);
        preparation_params = eventLeaf(*preparation_leaf);
    }
    HipCapturedGraph candidate(cell.stream.stream, [&] {
        for (int i = 0; i < Repeats; ++i)
            if (!launchROCmDenseCompactRecurrence(cell.input(), cell.stream.stream) ||
                !launchROCmDenseCompactAcceptedPrefix({cell.transaction(), cell.acceptedSelection(), cell.initial.ptr}, cell.stream.stream) ||
                !eventFollowing(cell, cell.initial.ptr, cell.following.ptr)) return false;
        return true;
    });
    HipCapturedGraph control(cell.stream.stream, [&] {
        for (int i = 0; i < Repeats; ++i)
            if ((preparation && !preparation->enqueue()) ||
                !launchROCmPlanarRecurrence(preparation ? preparation->input() : cell.denseInput(), cell.stream.stream) ||
                !eventAcceptedCopy(cell, cell.oracle_state.ptr, ordinal) ||
                !eventFollowing(cell, cell.oracle_state.ptr, cell.oracle_following.ptr)) return false;
        return true;
    });
    inspectEventSchedule(cell, candidate, {eventLeaf(*cell.capture), eventLeaf(*cell.publication_alias), eventLeaf(compact_next)}, Repeats);
    inspectEventSchedule(cell, control, {eventLeaf(prepared_dense_leaf ? *prepared_dense_leaf : *cell.dense), eventLeaf(dense_copy), eventLeaf(dense_next)}, Repeats, preparation ? &preparation_params : nullptr);
    gdn_paired_perf::Events events;
    gdn_merged_extended::Cell::Retirement timing_retirement{cell.stream.stream};
    for (int sample = -Warmups; sample < Samples; ++sample)
    {
        std::array<double, 2> us{};
        for (int order = 0; order < 2; ++order)
        {
            const int arm = (sample & 1) ? 1 - order : order;
            cell.reset();
            if (preparation) preparation->poison();
            if (sample < 0) (arm ? candidate : control).launch(cell.stream.stream);
            else us[arm] = events.measure(arm ? candidate : control, cell.stream.stream, Repeats);
            cell.finish();
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
            if (sample >= 0)
                std::cout << marker_prefix << "TIMED_BYTES heads=" << heads << " M=" << rows
                    << " accepted=" << accepted << " selection=" << selection << eventSpectrumFields(spectrum, live) << retentionMarkerFields(program, live) << terminalMarkerFields(terminal_kind) << " sample=" << sample
                    << " arm=" << arm << " repeats=" << Repeats
                    << operand_fields
                    << " actual_final_timed=1 next_row_in_timed_graph=1 serial_oracle=1 passed=1\n";
        }
        if (sample >= 0)
            std::cout << std::setprecision(9) << marker_prefix << "PAIR heads=" << heads
                << " DK=128 DV=128 M=" << rows << " accepted=" << accepted << " selection=" << selection << eventSpectrumFields(spectrum, live) << retentionMarkerFields(program, live) << terminalMarkerFields(terminal_kind)
                << " sample=" << sample << " repeats=" << Repeats << " warmups=" << Warmups
                << operand_fields
                << " dense_full_snapshots_copy_next_us=" << us[0] << " compact_capture_publish_next_us=" << us[1]
                << " dense_us_per_executed_row=" << us[0] / (live + 1)
                << " compact_us_per_executed_row=" << us[1] / (live + 1) << '\n';
    }
    std::cout << marker_prefix << "EVENT_CELL heads=" << heads << " M=" << rows
        << " accepted=" << accepted << " selection=" << selection << eventSpectrumFields(spectrum, live) << retentionMarkerFields(program, live) << terminalMarkerFields(terminal_kind)
        << " DK=128 DV=128 samples=" << Samples << " repeats=" << Repeats
        << " roles=2" << operand_fields
        << " strict_partial_available=" << (live > 1)
        << " compact_required_terminal=" << int(terminal_kind == GDNDenseCompactTerminalKind::DenseDestination)
        << " dense_terminal_in_snapshots=1 passed=1\n";
}

/** @brief Measure representative M1/M3 complete-operation cells.
 * @param ordinal Actual fixture-owned ROCm participant. */
inline void runCompleteEvents(int ordinal)
{
    for (int heads : {12, 24})
        for (int rows : {1, 3})
            for (int accepted : (rows == 1 ? std::vector<int>{0, 1} : std::vector<int>{0, 1, 3}))
                runCompleteEventCell(ordinal, heads, rows, accepted, accepted, EventAcceptanceSpectrum::Representative);
}

/** @brief Cover every accepted suffix, including seven updates before each checkpoint.
 * @param ordinal Actual fixture-owned ROCm participant.
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
 * @param ordinal Actual fixture-owned ROCm participant.
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

} // namespace rocm_dense_compact_test
} // namespace

TEST(Test__ROCmGDNPaddedRealLength, DISABLED_DenseCompactCompleteOperationEvents)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "dense compact event selected device");
    rocm_dense_compact_test::runCompleteEvents(0);
}

TEST(Test__ROCmGDNPaddedRealLength, DISABLED_DenseCompactEveryAcceptedPrefixEvents)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "dense compact acceptance spectrum selected device");
    rocm_dense_compact_test::runEveryAcceptedPrefixEvents(0);
}
/** @test One retained physical16 graph traverses full/ragged/empty/full counts without recapture. */
TEST(Test__ROCmGDNPaddedRealLength, DenseCompactPhysicalM16LiveReplay)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "Dense physical16 live replay device");
    using namespace rocm_dense_compact_test;
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
            cell.publishCounts(); cell.finish();
            const auto before = cell.compact.toHost();
            cell.capture->launch(cell.stream.stream); cell.dense->launch(cell.stream.stream);
            cell.finish();
            const auto shots = cell.snapshots.toHost();
            cell.checkCapture(before, shots);
            cell.checkSerial(shots);
            const auto retained = cell.compact.toHost();
            for (int accepted : {live, 0, live / 2, live})
            {
                cell.accepted_host[0] = accepted; cell.publishCounts();
                cell.publication->launch(cell.stream.stream);
                cell.publication_alias->launch(cell.stream.stream);
                cell.finish();
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
TEST(Test__ROCmGDNPaddedRealLength, DISABLED_DenseCompactPhysicalM16LiveEvents)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "Dense physical16 live events device");
    rocm_dense_compact_test::runPhysicalM16LiveEvents(0);
}
/** @test Full/threshold policies retain both native math and immutable rollback under mixed counts. */
TEST(Test__ROCmGDNPaddedRealLength, DenseSplitRetentionMixedRequestReplay)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "Dense split retained replay device");
    using namespace rocm_dense_compact_test;
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
                EXPECT_FALSE(launchROCmDenseCompactRecurrence(bad, cell.stream.stream));
                auto selected = GDNDenseCompactAcceptedPrefixInput{cell.transaction(), cell.acceptedSelection(), cell.accepted_state.ptr};
                selected.transaction.encoding = GDNDenseCompactEncoding::ExactCorrectionsV1;
                EXPECT_FALSE(launchROCmDenseCompactAcceptedPrefix(selected, cell.stream.stream));
                cell.run(schedule);
            }
}

namespace
{
namespace rocm_dense_compact_test
{
/** @brief Reuse the exact existing live48 schedule with one explicit candidate retention policy.
 * @param ordinal Actual fixture-selected ROCm participant.
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
TEST(Test__ROCmGDNPaddedRealLength, DISABLED_DenseFullRetentionPhysicalM16LiveEvents)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "Dense full retention event device");
    rocm_dense_compact_test::runSplitRetentionLiveEvents(0, {llaminar2::gdn::GDNDenseRetentionMode::FullSnapshots, 0});
}

/** @test Device-live threshold4 versus unchanged legacy complete chain on exact live48 work. */
TEST(Test__ROCmGDNPaddedRealLength, DISABLED_DenseThresholdRetentionPhysicalM16LiveEvents)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "Dense threshold retention event device");
    rocm_dense_compact_test::runSplitRetentionLiveEvents(0, {llaminar2::gdn::GDNDenseRetentionMode::LiveRowThreshold, 4});
}

/** @test More than one lane-stride metadata iteration preserves all accepted bytes and request holes. */
TEST(Test__ROCmGDNPaddedRealLength, DensePublisherMetadataMultipleLaneIterations)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "Dense publisher validation selected device");
    using namespace rocm_dense_compact_test;
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
TEST(Test__ROCmGDNPaddedRealLength, DenseRetainedPrefixesOnlyReplay)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "retained-only selected device");
    using namespace rocm_dense_compact_test;
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
namespace rocm_dense_compact_test
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
TEST(Test__ROCmGDNPaddedRealLength, DISABLED_DenseExactRetainedPrefixesOnlyPhysicalM16LiveEvents)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "retained-only selected device");
    rocm_dense_compact_test::runRetainedPrefixesOnlyLiveEvents(0,
        {llaminar2::gdn::GDNDenseRetentionMode::ExactCorrections, 0});
}

/** @test Full retained-only output versus legacy capture/publication/next M1. */
TEST(Test__ROCmGDNPaddedRealLength, DISABLED_DenseFullRetainedPrefixesOnlyPhysicalM16LiveEvents)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "retained-only selected device");
    rocm_dense_compact_test::runRetainedPrefixesOnlyLiveEvents(0,
        {llaminar2::gdn::GDNDenseRetentionMode::FullSnapshots, 0});
}

/** @test Threshold retained-only output versus legacy capture/publication/next M1. */
TEST(Test__ROCmGDNPaddedRealLength, DISABLED_DenseThresholdRetainedPrefixesOnlyPhysicalM16LiveEvents)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "retained-only selected device");
    rocm_dense_compact_test::runRetainedPrefixesOnlyLiveEvents(0,
        {llaminar2::gdn::GDNDenseRetentionMode::LiveRowThreshold, 4});
}

namespace
{
namespace rocm_dense_compact_test
{
/** @brief Time native verifier preparation/publication plus raw M1 continuation.
 * @param ordinal Actual fixture device.
 * @param program ExactCorrections or Threshold4, never a serving selector.
 * @details Common convolution is excluded from both arms. ROCm continuation is
 * the original optimized RAW merged M1, not its prepared producer/consumer pair.
 */
inline void runPostConvolutionLiveEvents(int ordinal, GDNDenseRetentionProgram program)
{
    for (int heads : {12, 24})
        for (int live : {0, 1, 2, 3, 4, 7, 8, 9, 15})
            for (int accepted : std::set<int>{0, live / 2, live})
                runCompleteEventCell(ordinal, heads, 16, accepted, accepted,
                    EventAcceptanceSpectrum::PhysicalM16LiveCounts, live, program,
                    GDNDenseCompactTerminalKind::RetainedPrefixesOnly,
                    CompleteEventInputBoundary::PostConvolutionMerged);
}
}
}

/** @test Explicit complete post-convolution Exact verifier boundary; not serving throughput. */
TEST(Test__ROCmGDNPaddedRealLength, DISABLED_DenseExactPostConvolutionRawPhysicalM16LiveEvents)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "Post-convolution event device");
    rocm_dense_compact_test::runPostConvolutionLiveEvents(0, {llaminar2::gdn::GDNDenseRetentionMode::ExactCorrections, 0});
}

/** @test Explicit complete post-convolution Threshold verifier boundary; not serving throughput. */
TEST(Test__ROCmGDNPaddedRealLength, DISABLED_DenseThresholdPostConvolutionRawPhysicalM16LiveEvents)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "Post-convolution event device");
    rocm_dense_compact_test::runPostConvolutionLiveEvents(0, {llaminar2::gdn::GDNDenseRetentionMode::LiveRowThreshold, 4});
}

/** @test Native-only planar preparation preserves live holes and revised accepted continuations.
 * @details Ratio3/offset0 matches production grouped head packing. Host-expanded
 * arrays are byte oracles only: the original full capture reads native scratch.
 */
TEST(Test__ROCmGDNPaddedRealLength, DensePostConvolutionNativePreparationReplay)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "Post-convolution preparation test device");
    using namespace rocm_dense_compact_test;
    for (int dk : {64, 128})
        for (auto axes : {Layout::KeyMajorValueContiguous, Layout::ValueMajorKeyContiguous})
        {
            SCOPED_TRACE(::testing::Message() << "DK=" << dk << " axes=" << int(axes));
            Cell cell(6, 2, dk, dk == 64 ? 17 : 65, 3, 16, axes, true, false, true, {},
                GDNDenseCompactTerminalKind::RetainedPrefixesOnly, 0);
            EventPlanarPreparation preparation(cell, 0);
            HipCapturedGraph prepare(cell.stream.stream, [&] { return preparation.enqueue(); });
            HipCapturedGraph dense(cell.stream.stream, [&] {
                return launchROCmPlanarRecurrence(preparation.input(), cell.stream.stream);
            });
            inspectEventPreparation(cell, preparation, prepare);
            gdn_merged_extended::Cell::Retirement retirement{cell.stream.stream};
            for (const auto &counts : std::array<std::array<int, 3>, 5>{{
                    {16, 16, 16}, {2, 0, 8}, {0, 3, 1}, {0, 0, 0}, {16, 16, 16}}})
            {
                cell.reset(); preparation.poison();
                cell.live_host.assign(counts.begin(), counts.end());
                std::fill(cell.accepted_host.begin(), cell.accepted_host.end(), 0);
                cell.publishCounts(); cell.finish();
                const auto before = cell.compact.toHost();
                cell.capture->launch(cell.stream.stream);
                prepare.launch(cell.stream.stream);
                dense.launch(cell.stream.stream);
                cell.finish();
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
                    cell.finish();
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
TEST(Test__ROCmGDNPaddedRealLength, DenseCompactTypedDeviceSelectionReplay)
{
    ASSERT_TRUE(hasROCm());
    checkHip(hipSetDevice(0), "Typed retained selection device");
    using namespace rocm_dense_compact_test;
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
                    cell.finish();
                    const auto before = cell.compact.toHost();
                    cell.capture->launch(cell.stream.stream);
                    cell.dense->launch(cell.stream.stream);
                    cell.finish();
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
                        cell.finish();
                        checkHip(hipMemcpyAsync(metadata.data(), cell.accepted.ptr,
                            metadata.size() * sizeof(int), hipMemcpyDeviceToHost, cell.stream.stream),
                            "Typed selection source snapshot");
                        cell.finish();
                        cell.publication->launch(cell.stream.stream);
                        cell.publication_alias->launch(cell.stream.stream);
                        cell.finish();
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
                        checkHip(hipMemcpyAsync(observed_metadata.data(), cell.accepted.ptr,
                            observed_metadata.size() * sizeof(int), hipMemcpyDeviceToHost, cell.stream.stream),
                            "Typed selection source recheck");
                        cell.finish();
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


namespace
{
namespace rocm_dense_compact_test
{
/** @brief Prove actual prepared kernel ownership without replacing its cache banks.
 * @param ordinal Actual selected backend device, observed through normal fixture startup.
 * @details All allocations and hardware discovery are cold. Two native streams
 * are ordered by a real producer event; terminal joins are test observations only.
 * The existing workspace manager owns both retained and optional terminal views.
 */
inline void preparedInvocationReplay(int ordinal)
{
    const auto device = DeviceId::rocm(ordinal);
    for (int requests : {1, 3})
    for (bool select_secondary : {false, true})
    for (int dk : {64, 128})
    for (auto axes : {Layout::KeyMajorValueContiguous, Layout::ValueMajorKeyContiguous})
    for (auto terminal_kind : {GDNDenseCompactTerminalKind::DenseDestination,
        GDNDenseCompactTerminalKind::RetainedPrefixesOnly})
    {
        Cell cell(2, 1, dk, dk == 64 ? 17 : 65, requests, 9, axes,
            true, false, false, {}, terminal_kind);
        HipStreamHandle publication_stream;
        const int state = cell.state_stride;
        const int primary_size = select_secondary ? state / 2 : state;
        const int secondary_size = select_secondary ? state : state * 2;
        const int maximum = std::max(primary_size, secondary_size);
        HipFloatBuffer primary(primary_size + Cell::Guard, Cell::Poison);
        HipFloatBuffer secondary(secondary_size + Cell::Guard, Cell::Poison);
        HipFloatBuffer packed(std::size_t(requests) * maximum + Cell::Guard, Cell::Poison);
        ROCmGatedDeltaNet kernel(ordinal);
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
        const std::string expected_family = terminal_kind == GDNDenseCompactTerminalKind::DenseDestination ?
            "rocm_gdn_dense_compact_capture_kernel" : "rocm_gdn_retained_prefixes_capture_kernel";
        require(program->captureProgramIdentity() == 512ULL + (static_cast<std::uint64_t>(terminal_kind) << 4),
            "Prepared compiled program identity differs");
        std::vector<float> retained_poison(retained_count, Cell::Poison);
        std::vector<float> terminal_poison(terminal_count, Cell::Poison);
        std::vector<float> primary_source(primary.count, Cell::Poison), secondary_source(secondary.count, Cell::Poison);
        std::vector<float> packed_source(packed.count, Cell::Poison);
        std::unique_ptr<HipCapturedGraph> capture, publication;
        struct Retirement final
        {
            hipStream_t capture_stream, publish_stream;
            hipEvent_t ready = nullptr;
            /** @brief Own both streams and the producer event until borrowed bytes retire. */
            explicit Retirement(hipStream_t a, hipStream_t b) : capture_stream(a), publish_stream(b)
            { checkHip(hipEventCreateWithFlags(&ready, hipEventDisableTiming), "Prepared event create"); }
            /** @brief Terminal join precedes graph, workspace and staging destruction. */
            ~Retirement()
            {
                if (hipStreamSynchronize(capture_stream) != hipSuccess ||
                    hipStreamSynchronize(publish_stream) != hipSuccess) std::terminate();
                if (ready) (void)hipEventDestroy(ready);
            }
        } retirement(cell.stream.stream, publication_stream.stream);
        const auto read = [&](const float *pointer, std::size_t count, hipStream_t stream) {
            std::vector<float> result(count);
            Retirement join(stream, stream);
            checkHip(hipMemcpyAsync(result.data(), pointer, count * sizeof(float),
                hipMemcpyDeviceToHost, stream), "Prepared actual bank read");
            checkHip(hipStreamSynchronize(stream), "Prepared observation join");
            return result;
        };
        capture = std::make_unique<HipCapturedGraph>(cell.stream.stream, [&] { return program->capture(); });
        publication = std::make_unique<HipCapturedGraph>(publication_stream.stream, [&] {
            return program->publish(cell.acceptedSelection(), publication_stream.stream);
        });
        // Authenticate the actual native function before decoding its typed tuple.
        std::size_t nodes = 0;
        checkHip(hipGraphGetNodes(capture->graph, nullptr, &nodes), "Prepared capture nodes");
        require(nodes == 1, "Prepared capture must contain exactly its native recurrence");
        hipGraphNode_t node{};
        checkHip(hipGraphGetNodes(capture->graph, &node, &nodes), "Prepared capture node");
        hipGraphNodeType node_kind{};
        checkHip(hipGraphNodeGetType(node, &node_kind), "Prepared capture node kind");
        require(node_kind == hipGraphNodeTypeKernel, "Prepared capture node is not native recurrence");
        hipKernelNodeParams parameters{};
        checkHip(hipGraphKernelNodeGetParams(node, &parameters), "Prepared capture arguments");
        const char *family = nullptr;
        family = hipKernelNameRefByPtr(parameters.func, cell.stream.stream);
        require(family && std::string(family).find(expected_family) != std::string::npos,
            "Prepared capture used a foreign native family");
        require(parameters.kernelParams && parameters.kernelParams[0], "Prepared typed capture tuple absent");
        require(parameters.gridDim.x == unsigned(requests * cell.heads) &&
            parameters.gridDim.y == unsigned((cell.dv + 31) / 32) && parameters.gridDim.z == 1 &&
            parameters.blockDim.x == 128 && parameters.blockDim.y == 1 && parameters.blockDim.z == 1 &&
            parameters.sharedMemBytes == std::size_t(2 * dk + 4 + 256 + (dk == 128 ? 10 * 128 : 0)) * sizeof(float), "Prepared capture launch geometry differs");
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
        checkHip(hipGraphGetNodes(publication->graph, nullptr, &publication_nodes), "Prepared publication nodes");
        std::vector<hipGraphNode_t> publication_node_list(publication_nodes);
        checkHip(hipGraphGetNodes(publication->graph, publication_node_list.data(), &publication_nodes), "Prepared publication node list");
        unsigned accepted_kernels = 0;
        for (const auto publication_node : publication_node_list)
        {
            hipGraphNodeType kind{};
            checkHip(hipGraphNodeGetType(publication_node, &kind), "Prepared publication node kind");
            if (kind != hipGraphNodeTypeKernel)
            {
                require(kind == hipGraphNodeTypeMemcpy, "Prepared publication has an unexpected nonkernel operation");
                continue;
            }
            hipKernelNodeParams arguments{};
            checkHip(hipGraphKernelNodeGetParams(publication_node, &arguments), "Prepared publication arguments");
            const char *publication_family = nullptr;
            publication_family = hipKernelNameRefByPtr(arguments.func, publication_stream.stream);
            require(publication_family && std::string(publication_family).find(
                "rocm_gdn_dense_compact_accepted_prefix_kernel") != std::string::npos,
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
            cell.finish();
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
            checkHip(hipMemcpyAsync(retained, retained_poison.data(), retained_poison.size() * sizeof(float),
                hipMemcpyHostToDevice, cell.stream.stream), "Prepared retained poison");
            if (terminal) checkHip(hipMemcpyAsync(terminal, terminal_poison.data(), terminal_poison.size() * sizeof(float),
                hipMemcpyHostToDevice, cell.stream.stream), "Prepared terminal poison");
            capture->launch(cell.stream.stream);
            cell.dense->launch(cell.stream.stream);
            checkHip(hipEventRecord(retirement.ready, cell.stream.stream), "Prepared producer completion");
            checkHip(hipStreamWaitEvent(publication_stream.stream, retirement.ready, 0), "Prepared consumer event wait");
            checkHip(hipStreamSynchronize(cell.stream.stream), "Prepared captured observation");
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
                checkHip(hipEventRecord(retirement.ready, cell.stream.stream), "Prepared metadata completion");
                checkHip(hipStreamWaitEvent(publication_stream.stream, retirement.ready, 0), "Prepared metadata event wait");
                publication->launch(publication_stream.stream);
                checkHip(hipStreamSynchronize(publication_stream.stream), "Prepared actual publication join");
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
                        return launchROCmPlanarRecurrence({DeviceRequestRowRanges::fullyActive(1), cell.heads, cell.dk, cell.dv,
                            cell.normalize, cell.query.ptr + row * cell.heads * cell.dk, cell.key.ptr + row * cell.heads * cell.dk,
                            cell.value.ptr + row * cell.heads * cell.dv, cell.alpha.ptr + row * cell.heads,
                            cell.beta.ptr + row * cell.heads, cell.a.ptr, cell.bias.ptr, out + std::size_t(r) * cell.heads * cell.dv,
                            bank + std::size_t(r) * state, bank + std::size_t(r) * state, nullptr, 0, 0, axes}, publication_stream.stream);
                    };
                    require(next(program->publicationDestination(), cell.following.ptr) &&
                        next(cell.oracle_state.ptr, cell.oracle_following.ptr), "Prepared actual following native row rejected");
                }
                checkHip(hipStreamSynchronize(publication_stream.stream), "Prepared following observation");
                equal(read(program->publicationDestination(), std::size_t(requests) * state, publication_stream.stream),
                    read(cell.oracle_state.ptr, std::size_t(requests) * state, publication_stream.stream), "Prepared following state bytes");
                equal(cell.following.toHost(), cell.oracle_following.toHost(), "Prepared following output bytes");
            }
        }
    }
}
} // namespace rocm_dense_compact_test
} // namespace

/** @test Real prepared adapters preserve scalar/packed ownership and immutable bindings across captured replay. */
TEST(Test__ROCmGDNPaddedRealLength, DensePreparedRetainedInvocationOwnershipReplay)
{
    if (!hasROCm()) GTEST_SKIP() << "No ROCm device available";
    checkHip(hipSetDevice(0), "Prepared adapter selected native device");
    rocm_dense_compact_test::preparedInvocationReplay(0);
}


namespace
{
namespace rocm_dense_compact_test
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
            const auto axes = production_geometry ? Layout::ValueMajorKeyContiguous : Layout::KeyMajorValueContiguous;
            SCOPED_TRACE(::testing::Message() << "requests=" << requests << " DV=" << dv
                << " axes=" << int(axes) << " terminal=" << unsigned(terminal_kind));
            Cell cell(heads, keys, dk, dv, requests, rows, axes,
                true, false, false, {}, terminal_kind, 0);
            HipStreamHandle publication_stream;
            ROCmGatedDeltaNet kernel(device.gpu_ordinal());
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

            const std::string expected_family = terminal_kind == GDNDenseCompactTerminalKind::DenseDestination ?
                "rocm_gdn_dense_compact_capture_kernel" : "rocm_gdn_retained_prefixes_capture_kernel";
            std::vector<GPUGraphKernelNodeInfo> capture_kernels, publication_kernels;
            std::string inspection_error;
            require(capture->inspectKernelNodes(capture_kernels, &inspection_error) && capture_kernels.size() == 1 &&
                capture_kernels.front().name_resolved && capture_kernels.front().name.find(expected_family) != std::string::npos &&
                capture_kernels.front().grid_x == unsigned(requests * heads) &&
                capture_kernels.front().grid_y == unsigned((dv + 31) / 32), "Actual stage capture native identity differs");
            require(publication->inspectKernelNodes(publication_kernels, &inspection_error) && publication_kernels.size() == 1 &&
                publication_kernels.front().name_resolved && publication_kernels.front().name.find("rocm_gdn_dense_compact_accepted_prefix_kernel") != std::string::npos &&
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
                cell.publishCounts(); cell.finish();
                require(backend->hostToDeviceOnStream(lengths.gpu_data_ptr(), cell.live_host.data(), lengths.size_bytes(),
                    device.gpu_ordinal(), stream), "Retained stage authored live-count publication failed");
                require(backend->hostToDeviceOnStream(retained, poison.data(), poison.size() * sizeof(float),
                    device.gpu_ordinal(), stream), "Retained stage dirty arena publication failed");
                if (terminal) require(backend->hostToDeviceOnStream(terminal, terminal_poison.data(), terminal_poison.size() * sizeof(float),
                    device.gpu_ordinal(), stream), "Retained stage terminal sentinel failed");
                require(capture->launch(), "Retained stage replay failed");
                cell.dense->launch(cell.stream.stream); cell.finish();
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
                            return launchROCmPlanarRecurrence({DeviceRequestRowRanges::fullyActive(1), heads, dk, dv, true,
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
} // namespace rocm_dense_compact_test
} // namespace

/** @test Actual stage leases, capture-only preparation, publication and reset preserve native recurrence bytes. */
TEST(Test__ROCmGDNPaddedRealLength, DenseRetainedStageTensorLeaseReplay)
{
    if (!hasROCm()) GTEST_SKIP() << "ROCm unavailable";
    rocm_dense_compact_test::retainedStageReplay(DeviceId::rocm(0));
}

namespace
{
namespace rocm_dense_compact_test
{
/** @brief Prove typed NoOp preserves both native destinations, including a deliberately stale
 * mirror.
 * @param prepared Use the real ROCm prepared adapter instead of direct native publication.
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
                    HipFloatBuffer mirror_bank(state + Cell::Guard + 1, Cell::Poison);
                    HipFloatBuffer primary(state + Cell::Guard, Cell::Poison);
                    HipFloatBuffer secondary(2 * state + Cell::Guard, Cell::Poison);
                    HipFloatBuffer packed(2 * state * requests + Cell::Guard, Cell::Poison);
                    ROCmGatedDeltaNet kernel(0);
                    WorkspaceRequirements requirements;
                    requirements.buffers.emplace_back("mirror_retained", cell.geometry.bankBytes(),
                                                      64);
                    DeviceWorkspaceManager workspace(DeviceId::rocm(0),
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
                    std::unique_ptr<HipCapturedGraph> capture, publication;
                    struct Retirement final
                    {
                        hipStream_t stream;
                        /** @brief Retire graphs before every borrowed host or device owner is
                         * destroyed. */
                        ~Retirement()
                        {
                            if (hipStreamSynchronize(stream) != hipSuccess)
                                std::terminate();
                        }
                    } retirement{cell.stream.stream};
                    const auto read = [&](const float *pointer, std::size_t count)
                    {
                        std::vector<float> result(count);
                        Retirement join{cell.stream.stream};
                        checkHip(hipMemcpyAsync(result.data(), pointer, count * sizeof(float),
                                                hipMemcpyDeviceToHost, cell.stream.stream),
                                 "Mirror actual bank read");
                        checkHip(hipStreamSynchronize(cell.stream.stream),
                                 "Mirror terminal observation");
                        return result;
                    };
                    const auto write = [&](float *pointer, const std::vector<float> &values)
                    {
                        Retirement join{cell.stream.stream};
                        checkHip(hipMemcpyAsync(pointer, values.data(),
                                                values.size() * sizeof(float),
                                                hipMemcpyHostToDevice, cell.stream.stream),
                                 "Mirror data publication");
                        checkHip(hipStreamSynchronize(cell.stream.stream),
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
                    capture = std::make_unique<HipCapturedGraph>(
                        cell.stream.stream,
                        [&]
                        {
                            return prepared ? invocation->capture()
                                            : launchROCmDenseCompactRecurrence(cell.input(),
                                                                               cell.stream.stream);
                        });
                    publication = std::make_unique<HipCapturedGraph>(
                        cell.stream.stream,
                        [&]
                        {
                            if (prepared)
                                return invocation->publish(cell.acceptedSelection(),
                                                           cell.stream.stream);
                            return launchROCmDenseCompactAcceptedPrefix(
                                {cell.transaction(), cell.acceptedSelection(), destination,
                                 GDNDenseCompactRequestZeroMirror::borrow(cell.geometry, mirror,
                                                                          state)},
                                cell.stream.stream);
                        });
                    // Authenticate the native family before decoding the complete optional view.
                    std::size_t count = 0;
                    checkHip(hipGraphGetNodes(publication->graph, nullptr, &count),
                             "Mirror publication node count");
                    require(count == 1,
                            "Conditional mirror must be part of the one publisher kernel");
                    hipGraphNode_t node{};
                    checkHip(hipGraphGetNodes(publication->graph, &node, &count),
                             "Mirror publication node");
                    hipKernelNodeParams params{};
                    checkHip(hipGraphKernelNodeGetParams(node, &params),
                             "Mirror publication arguments");
                    const char *name = nullptr;
                    name = hipKernelNameRefByPtr(params.func, cell.stream.stream);
                    require(name &&
                                std::string(name).find(
                                    "rocm_gdn_dense_compact_accepted_prefix_kernel") !=
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
                        checkHip(hipStreamSynchronize(cell.stream.stream),
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
                            checkHip(hipStreamSynchronize(cell.stream.stream),
                                     "Mirror terminal observation");
                            equal(read(destination, expected.size()), expected,
                                  "Conditional mirror actual accepted bank");
                            equal(read(mirror, state), mirror_expected,
                                  "Conditional mirror typed NoOp/Restore semantics");
                        }
                    }
                }
}
} // namespace rocm_dense_compact_test
} // namespace

/** @test Optional request-zero mirror is part of the validated publisher and preserves semantic
 * NoOp. */
TEST(Test__ROCmGDNPaddedRealLength, DenseCompactConditionalRequestZeroMirrorReplay)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    rocm_dense_compact_test::conditionalMirrorReplay(false);
}
/** @test Real ROCm adapter preserves a poisoned nonauthoritative bank for typed NoOp. */
TEST(Test__ROCmGDNPaddedRealLength, DensePreparedRetainedInvocationNoOpMirrors)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm device available";
    rocm_dense_compact_test::conditionalMirrorReplay(true);
}

#endif
