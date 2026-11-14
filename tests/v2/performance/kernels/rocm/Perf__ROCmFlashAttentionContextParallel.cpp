/**
 * @file Perf__ROCmFlashAttentionContextParallel.cpp
 * @brief Captured ROCm FA2 query/context policy and launch-geometry tournament.
 *
 * This benchmark measures the two byte-equivalent ROCm FlashAttention prefill
 * schedules through the same native wrappers used by production. Fixed-query
 * scaling records the actual default score owner and native graph resources
 * as retained history grows and shrinks without recapture. The FP16 fixtures
 * share CUDA's deterministic input bytes and verify every automatic output
 * outside timing against canonical bytes after poisoned retained replay.
 * A native gfx1100 paired experiment retains the same fixed query bucket and
 * complete device-count producer graph while comparing ordered Dot2 and
 * four-plane WMMA QK through growing and reset K/V spans. Its context oracle
 * independently exercises the guarded producer and ordered summary reducer.
 * Q/K pairs use retained singleton timeline events inside complete transaction
 * batches; host graph submission gaps do not enter those intervals.
 * Query-sequence
 * execution gives one workgroup a complete K/V scan. K/V-context execution
 * gives a bounded persistent grid canonical 256-key partitions and then merges
 * those summaries in ascending key order on device.
 *
 * The benchmark owns three separate responsibilities:
 *
 * 1. sweep every useful persistent-slot and reducer-wavefront candidate over
 *    representative 64-, 128-, and 256-element Qwen head geometries;
 * 2. certify the installed generic policy over every distinct participant-local
 *    geometry induced by released Qwen models, TP=1/2/4/8, and all native K/V
 *    formats; and
 * 3. stress the same captured transaction at 256K resident context.
 *
 * All allocations, initialization, graph construction, and parameter uploads
 * occur outside measured replay. Timed samples bracket batches of complete HIP
 * graph transactions with stream events and synchronize only the terminal
 * event. There are no per-replay allocations, transfers, host callbacks, or
 * blocking synchronizations in the measured path.
 */

#include <gtest/gtest.h>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "backends/rocm/HIPGraphCapture.h"
#endif

#include "kernels/attention/AttentionDeviceParams.h"
#include "kernels/rocm/attention/ROCmFlashAttentionLaunchPolicy.h"
#include "kernels/rocm/attention/ROCmFlashAttentionKernelT.h"
#include "kernels/rocm/attention/ROCmFlashAttentionQK.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/GpuTensorView.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef HAVE_ROCM
extern "C"
{
    /** @brief Launch the direct MI50 FA2 schedule over FP32 K/V. */
    int hipFlashAttn_prefill_fa2(
        const float *Q, const float *K, const float *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask, void *stream, int head_start, int gqa_n_rep);

    /** @brief Launch the direct MI50 FA2 schedule over FP16 K/V. */
    int hipFlashAttn_prefill_fa2_fp16(
        const float *Q, const void *K, const void *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask, void *stream, int head_start, int gqa_n_rep);

    /** @brief Launch the direct MI50 FA2 schedule over BF16 K/V. */
    int hipFlashAttn_prefill_fa2_bf16(
        const float *Q, const void *K, const void *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask, void *stream, int head_start, int gqa_n_rep);

    /** @brief Launch the direct MI50 FA2 schedule over Q8_1 K/V. */
    int hipFlashAttn_prefill_fa2_q8_1(
        const float *Q, const void *K, const void *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask, void *stream, int head_start, int gqa_n_rep);

#define DECLARE_ROCM_FA2_CONTEXT_WRAPPER(name)                              \
    int name(                                                              \
        const float *Q, const void *K, const void *V, float *O,             \
        float *O_partial, float *m_partial, float *l_partial,               \
        int batch_size, int seq_len, int kv_capacity,                       \
        int n_heads, int n_kv_heads, int head_dim,                          \
        bool causal, int window_size, int position_offset,                  \
        const llaminar2::attention::AttentionDeviceParams *device_params,   \
        const float *mask, int context_partition_size,                      \
        int max_context_partitions, int context_partition_slots,            \
        int context_phase_block_slots,                                      \
        int device_direct_partition_limit,                                  \
        int reducer_dimension_wavefronts, int reducer_block_slots,          \
        void *stream,                                                       \
        int head_start, int gqa_n_rep)

    /** @brief Launch the two-node context transaction over FP32 K/V. */
    DECLARE_ROCM_FA2_CONTEXT_WRAPPER(
        hipFlashAttn_prefill_fa2_context_parallel);
    /** @brief Launch the two-node context transaction over FP16 K/V. */
    DECLARE_ROCM_FA2_CONTEXT_WRAPPER(
        hipFlashAttn_prefill_fa2_fp16_context_parallel);
    /** @brief Launch the two-node context transaction over BF16 K/V. */
    DECLARE_ROCM_FA2_CONTEXT_WRAPPER(
        hipFlashAttn_prefill_fa2_bf16_context_parallel);
    /** @brief Launch the two-node context transaction over Q8_1 K/V. */
    DECLARE_ROCM_FA2_CONTEXT_WRAPPER(
        hipFlashAttn_prefill_fa2_q8_1_context_parallel);

#undef DECLARE_ROCM_FA2_CONTEXT_WRAPPER
}
#endif

namespace
{
    using llaminar2::attention::AttentionDeviceParams;
    namespace rocm_policy = llaminar2::rocm::fa2_policy;

    /** Native K/V representations accepted by the production ROCm FA2 path. */
    enum class NativeKVFormat : std::uint8_t
    {
        FP32,
        FP16,
        BF16,
        Q8_1,
    };

    /** @return Stable diagnostic name for one native K/V representation. */
    [[nodiscard]] constexpr const char *nativeKVFormatName(
        NativeKVFormat format) noexcept
    {
        switch (format)
        {
        case NativeKVFormat::FP32:
            return "FP32";
        case NativeKVFormat::FP16:
            return "FP16";
        case NativeKVFormat::BF16:
            return "BF16";
        case NativeKVFormat::Q8_1:
            return "Q8_1";
        }
        return "invalid";
    }

    inline constexpr std::array<NativeKVFormat, 4> kNativeKVFormats{{
        NativeKVFormat::FP32,
        NativeKVFormat::FP16,
        NativeKVFormat::BF16,
        NativeKVFormat::Q8_1,
    }};

    /** Full-model attention geometry used only to derive physical participants. */
    struct ModelAttentionGeometry
    {
        const char *release_label = nullptr;
        int n_heads = 0;
        int n_kv_heads = 0;
        int head_dim = 0;
    };

    /**
     * Released Qwen attention geometries, including explicit MTP-sidecar aliases.
     *
     * Main and MTP graphs with the same physical geometry intentionally collapse
     * to one participant record below. Keeping both aliases here documents that
     * the policy evidence applies to both graph roles without teaching dispatch
     * model names or graph-stage names.
     */
    inline constexpr std::array<ModelAttentionGeometry, 12>
        kQwenAttentionGeometries{{
            {"Qwen2.5-0.5B", 14, 2, 64},
            {"Qwen2.5-1.5B", 12, 2, 128},
            {"Qwen2.5-3B", 16, 2, 128},
            {"Qwen2.5-7B", 28, 4, 128},
            {"Qwen2.5-14B/32B", 40, 8, 128},
            {"Qwen3.5-0.8B/2B", 8, 2, 256},
            {"Qwen3.5-4B/9B", 16, 4, 256},
            {"Qwen3.5/3.6-27B", 24, 4, 256},
            {"Qwen3.5/3.6-35B-A3B", 16, 2, 256},
            {"Qwen3.5-122B-A10B/397B-A17B", 32, 2, 256},
            {"Qwen3.6-27B-MTP", 24, 4, 256},
            {"Qwen3.6-35B-A3B-MTP", 16, 2, 256},
        }};

    /** Exact participant-local geometry passed to one ROCm FA2 launcher. */
    struct ParticipantAttentionGeometry
    {
        std::string sources;
        int n_heads = 0;
        int n_kv_heads = 0;
        int head_dim = 0;
        int head_start = 0;
        int replicated_gqa_n_rep = 0;

        /** @return Geometry-only key used to remove redundant model aliases. */
        [[nodiscard]] auto key() const noexcept
        {
            return std::tuple{
                n_heads,
                n_kv_heads,
                head_dim,
                head_start,
                replicated_gqa_n_rep};
        }
    };

    /**
     * @brief Resolve the rank-zero participant geometry for an equal TP split.
     *
     * Query heads are sharded whenever the split is integral. K/V heads are
     * sharded when divisible by TP and otherwise remain replicated, matching the
     * graph's compact-GQA placement policy. Rank zero is sufficient for economy;
     * other head starts change addresses but not launch geometry or instruction
     * count, and are covered by the byte-exact integration sweep.
     */
    [[nodiscard]] ParticipantAttentionGeometry resolveParticipantGeometry(
        const ModelAttentionGeometry &model,
        int tp_degree)
    {
        if (tp_degree <= 0 || model.n_heads % tp_degree != 0)
            return {};

        ParticipantAttentionGeometry participant{
            .sources = std::string(model.release_label) + "@TP" +
                       std::to_string(tp_degree),
            .n_heads = model.n_heads / tp_degree,
            .n_kv_heads = model.n_kv_heads,
            .head_dim = model.head_dim,
            .head_start = 0,
            .replicated_gqa_n_rep = 0,
        };
        if (model.n_kv_heads % tp_degree == 0)
        {
            participant.n_kv_heads = model.n_kv_heads / tp_degree;
        }
        else
        {
            participant.replicated_gqa_n_rep =
                model.n_heads / model.n_kv_heads;
        }
        return participant;
    }

    /**
     * @brief Build every distinct participant geometry induced by Qwen TP splits.
     *
     * Dispatch consumes geometry rather than model identity. Duplicates are
     * therefore combined while their source aliases remain visible in CSV output.
     */
    [[nodiscard]] std::vector<ParticipantAttentionGeometry>
    buildDistinctParticipantGeometries()
    {
        constexpr std::array<int, 4> tp_degrees{{1, 2, 4, 8}};
        std::vector<ParticipantAttentionGeometry> participants;
        for (const ModelAttentionGeometry &model : kQwenAttentionGeometries)
        {
            for (const int tp_degree : tp_degrees)
            {
                ParticipantAttentionGeometry candidate =
                    resolveParticipantGeometry(model, tp_degree);
                if (candidate.n_heads <= 0)
                    continue;

                const auto existing = std::find_if(
                    participants.begin(),
                    participants.end(),
                    [&](const ParticipantAttentionGeometry &participant)
                    {
                        return participant.key() == candidate.key();
                    });
                if (existing == participants.end())
                {
                    participants.push_back(std::move(candidate));
                }
                else
                {
                    existing->sources += "|" + candidate.sources;
                }
            }
        }
        return participants;
    }

    /** @return Exact storage bytes for one capacity-sized native K/V tensor. */
    [[nodiscard]] std::size_t nativeKVBytes(
        NativeKVFormat format,
        int kv_capacity,
        int n_kv_heads,
        int head_dim)
    {
        if (kv_capacity <= 0 || n_kv_heads <= 0 || head_dim <= 0)
            return 0;
        const std::size_t rows =
            static_cast<std::size_t>(kv_capacity) *
            static_cast<std::size_t>(n_kv_heads);
        switch (format)
        {
        case NativeKVFormat::FP32:
            return rows * static_cast<std::size_t>(head_dim) * sizeof(float);
        case NativeKVFormat::FP16:
        case NativeKVFormat::BF16:
            return rows * static_cast<std::size_t>(head_dim) *
                   sizeof(std::uint16_t);
        case NativeKVFormat::Q8_1:
        {
            constexpr std::size_t block_elements = 32;
            constexpr std::size_t block_bytes = 36;
            const std::size_t blocks_per_row =
                (static_cast<std::size_t>(head_dim) + block_elements - 1) /
                block_elements;
            return rows * blocks_per_row * block_bytes;
        }
        }
        return 0;
    }

#ifdef HAVE_ROCM
    /** RAII owner for one captured HIP graph and executable instance. */
    struct CapturedHIPGraph
    {
        hipGraph_t graph = nullptr;
        hipGraphExec_t exec = nullptr;
        std::size_t node_count = 0;

        /** Destroy executable state before its source graph. */
        void reset() noexcept
        {
            if (exec)
                (void)hipGraphExecDestroy(exec);
            if (graph)
                (void)hipGraphDestroy(graph);
            graph = nullptr;
            exec = nullptr;
            node_count = 0;
        }

        ~CapturedHIPGraph() { reset(); }

        CapturedHIPGraph() = default;
        CapturedHIPGraph(const CapturedHIPGraph &) = delete;
        CapturedHIPGraph &operator=(const CapturedHIPGraph &) = delete;
    };

    /** Result of timing one captured physical transaction at one live K/V span. */
    struct TransactionTiming
    {
        double median_us = 0.0;
        int replays_per_sample = 0;
        std::size_t graph_nodes = 0;
        std::array<double, 3> samples_us{}; ///< Unsorted native-event samples, retained for scaling evidence.
    };

    /**
     * @brief Persistent arena and graph owner for one geometry/format/capacity.
     *
     * Maximum-M buffers are allocated once. Each immutable M bucket is then
     * captured through the production direct launcher and any requested context
     * candidate. Live K/V length is changed only by publishing the existing
     * device parameter record; neither graph is recaptured as context grows.
     */
    class CapturedROCmFA2Benchmark final
    {
    public:
        CapturedROCmFA2Benchmark(
            ParticipantAttentionGeometry geometry,
            NativeKVFormat format,
            int maximum_query_rows,
            int kv_capacity,
            int compute_unit_count,
            std::size_t lds_capacity_bytes,
            int device = 0)
            : geometry_(std::move(geometry)),
              format_(format),
              maximum_query_rows_(maximum_query_rows),
              kv_capacity_(kv_capacity),
              compute_unit_count_(compute_unit_count),
              lds_capacity_bytes_(lds_capacity_bytes),
              device_(device)
        {
            initialize();
        }

        /** @brief Drain submitted work before retiring graph bindings on every test exit. */
        ~CapturedROCmFA2Benchmark()
        {
            (void)hipSetDevice(device_);
            // A failed terminal-event record may leave an earlier graph live.
            // This test-only retirement boundary precedes all borrowed storage.
            if (stream_) (void)hipStreamSynchronize(stream_);
            for (auto &owner : default_timing_) owner.reset();
            for (auto &fragment : default_timing_fragments_) fragment.reset();
            default_graph_.reset();
            default_kernel_.reset();
            default_workspace_.reset();
            if (d_default_live_length_) (void)hipFree(d_default_live_length_);
            context_graph_.reset();
            direct_graph_.reset();
            if (start_)
                (void)hipEventDestroy(start_);
            if (stop_)
                (void)hipEventDestroy(stop_);
            if (d_q_)
                (void)hipFree(d_q_);
            if (d_k_)
                (void)hipFree(d_k_);
            if (d_v_)
                (void)hipFree(d_v_);
            if (d_direct_output_)
                (void)hipFree(d_direct_output_);
            if (d_context_output_)
                (void)hipFree(d_context_output_);
            if (d_partial_output_)
                (void)hipFree(d_partial_output_);
            if (d_partial_m_)
                (void)hipFree(d_partial_m_);
            if (d_partial_l_)
                (void)hipFree(d_partial_l_);
            if (d_params_)
                (void)hipFree(d_params_);
            if (stream_)
                (void)hipStreamDestroy(stream_);
        }

        CapturedROCmFA2Benchmark(const CapturedROCmFA2Benchmark &) = delete;
        CapturedROCmFA2Benchmark &operator=(
            const CapturedROCmFA2Benchmark &) = delete;

        /** @return True when every persistent resource was created. */
        [[nodiscard]] bool ready() const noexcept { return ready_; }

        /** @return First setup, capture, or replay failure. */
        [[nodiscard]] const std::string &error() const noexcept { return error_; }

        /** @return Geometry represented by this persistent arena. */
        [[nodiscard]] const ParticipantAttentionGeometry &geometry() const noexcept
        {
            return geometry_;
        }

        /**
         * @brief Capture the direct graph for one immutable M bucket.
         * @param query_rows Query rows embedded in the graph launch geometry.
         */
        bool prepareQueryRows(int query_rows)
        {
            if (!ready_ || query_rows <= 0 ||
                query_rows > maximum_query_rows_)
            {
                setError("invalid query-row bucket");
                return false;
            }
            current_query_rows_ = query_rows;
            context_graph_.reset();
            return captureDirectGraph();
        }

        /**
         * @brief Capture one exact context candidate for the current M bucket.
         * @param explicit_slots Zero for generic policy or an exact slot count.
         * @param explicit_wavefronts Zero for generic policy or exact reducer width.
         * @param explicit_reducer_blocks Zero for generic policy or an exact
         *        persistent reducer-grid width.
         * @param explicit_phase_blocks Zero for generic policy or an exact
         *        persistent phase-grid width.
         */
        bool captureContextCandidate(
            int explicit_slots,
            int explicit_wavefronts,
            int explicit_reducer_blocks = 0,
            int explicit_phase_blocks = 0)
        {
            if (!ready_ || current_query_rows_ <= 0)
            {
                setError("context capture requires a prepared query bucket");
                return false;
            }

            context_plan_ = rocm_policy::selectROCmFA2PrefillParallelPlan(
                makePolicyGeometry(
                    llaminar2::attention::AttentionPrefillParallelAxis::
                        KeyValueContext),
                explicit_slots,
                explicit_wavefronts,
                explicit_reducer_blocks,
                explicit_phase_blocks);
            if (!context_plan_.valid ||
                !context_plan_.usesContextParallelism())
            {
                setError("invalid context candidate geometry");
                return false;
            }
            if (context_plan_.partial_output_bytes > partial_output_bytes_ ||
                context_plan_.partial_m_bytes > partial_m_bytes_ ||
                context_plan_.partial_l_bytes > partial_l_bytes_)
            {
                setError("context candidate exceeds persistent workspace");
                return false;
            }
            return captureGraph(context_graph_, /*context=*/true);
        }

        /** @return Generic geometry-selected plan for the current M bucket. */
        [[nodiscard]] rocm_policy::ROCmFA2PrefillParallelPlan policyPlan() const
        {
            return rocm_policy::selectROCmFA2PrefillParallelPlan(
                makePolicyGeometry(
                    llaminar2::attention::AttentionPrefillParallelAxis::
                        GeometrySelected));
        }

        /** @return Exact forced-context plan currently embedded in the graph. */
        [[nodiscard]] const rocm_policy::ROCmFA2PrefillParallelPlan &
        contextPlan() const noexcept
        {
            return context_plan_;
        }


        /**
         * @brief Capture the production automatic bridge with device-owned KV count.
         * @return False when exact native geometry or workspace is rejected.
         *
         * Native tournament controls remain separate. The focused query plan
         * owns parameter workspace and a retained device-count producer before
         * compute_tensor's selected attention node on every replay.
         */
        bool captureDefaultGraph()
        {
            const auto plan = policyPlan();
            if (!ready_ || !plan.valid || plan.usesContextParallelism() ||
                format_ != NativeKVFormat::FP16)
            {
                setError("default fixed-query capture requires an admitted FP16 query plan");
                return false;
            }
            using namespace llaminar2;
            const DeviceId device = DeviceId::rocm(device_);
            WorkspaceRequirements requirements;
            requirements.buffers.push_back({rocm::AttentionWorkspaceBuffers::DEVICE_PARAMS,
                sizeof(attention::AttentionDeviceParams), 256, true});
            default_workspace_ = std::make_unique<DeviceWorkspaceManager>(
                device, requirements.total_bytes_with_alignment());
            if (!default_workspace_->allocate(requirements))
            {
                setError("default attention parameter workspace allocation failed");
                return false;
            }
            default_kernel_ = std::make_unique<
                rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32>>(device_);
            default_kernel_->setGPUStream(stream_);
            default_kernel_->bindWorkspace(default_workspace_.get());
            if (!allocate(reinterpret_cast<void **>(&d_default_live_length_),
                          sizeof(int), "default device KV count") ||
                !publishLiveLength(current_query_rows_) ||
                !check(hipStreamSynchronize(stream_), "default capture preparation"))
                return false;
            bool submitted = false;
            hipError_t end_status = hipSuccess;
            {
                GraphCaptureGuard guard;
                if (!check(hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal),
                           "default begin capture")) return false;
                submitted = enqueueDefaultTransaction();
                end_status = hipStreamEndCapture(stream_, &default_graph_.graph);
            }
            if (!submitted || !check(end_status, "default end capture") ||
                !check(hipGraphGetNodes(default_graph_.graph, nullptr, &default_graph_.node_count),
                       "default graph inventory") || default_graph_.node_count != 2 ||
                !check(hipGraphInstantiate(&default_graph_.exec, default_graph_.graph,
                                          nullptr, nullptr, 0), "default graph instantiate"))
            {
                setError("default attention must capture one device producer and one query body");
                return false;
            }
            return true;
        }

        /** @brief Prepare immutable one/sixteen-operation timing graphs.
         * @return Whether actual production kernels and internal event owners match.
         * Both batch sizes are captured once, before any live-prefix sample.
         */
        bool prepareDefaultTiming()
        {
            using namespace llaminar2;
            if (!default_graph_.exec || default_timing_[0] || default_timing_[1])
            {
                setError("default timing requires one fresh prepared production graph");
                return false;
            }
            hipKernelNodeParams attention{};
            if (!defaultKernelInventory(attention)) return false;
            const char *symbol = defaultKernelName(attention);
            if (!symbol) { setError("production attention symbol is unavailable"); return false; }
            const std::string attention_name(symbol);
            for (size_t index = 0; index < default_timing_.size(); ++index)
            {
                const int replays = index == 0 ? 1 : 16;
                auto &owner = default_timing_[index];
                auto &fragment = default_timing_fragments_[index];
                owner = std::make_unique<HIPGraphCapture>(stream_, device_);
                fragment = owner->createOrderedTimelineFragment();
                if (!fragment) { setError("default timing fragment creation failed"); return false; }
                bool submitted = true;
                {
                    GraphCaptureGuard guard;
                    if (!fragment->beginCapture()) { setError("default timing capture begin failed"); return false; }
                    for (int replay = 0; replay < replays && submitted; ++replay)
                        submitted = enqueueDefaultTransaction();
                    if (!fragment->endCapture()) submitted = false;
                }
                const std::array<GPUOrderedTimelineStep,1> steps{{{
                    .name = "complete_attention_batch", .kind = GPUOrderedTimelineStepKind::CapturedFragment,
                    .capture = fragment.get()}}};
                if (!submitted || fragment->nodeCount() != size_t(2 * replays) ||
                    !owner->buildOrderedTimelineTransaction(steps, GPUOrderedTimelineInstrumentation::PerStepEvents) ||
                    owner->nodeCount() != size_t(2 * replays + 2) ||
                    owner->consumeOrderedTimelineTiming().state != GPUOrderedTimelineTimingState::AwaitingLaunch)
                { setError("default timing must retain each producer/body and exactly two internal events"); return false; }
                std::vector<GPUGraphKernelNodeInfo> kernels;
                if (!owner->inspectKernelNodes(kernels) || kernels.size() != size_t(2 * replays))
                { setError("default timing production inventory is incomplete"); return false; }
                int attention_nodes = 0;
                for (const auto &kernel : kernels)
                {
                    if (kernel.local_memory_bytes_per_thread != 0)
                    { setError("default timing kernel uses scratch"); return false; }
                    if (kernel.name.find("flash_attention_2_mi50_kernel") == std::string::npos) continue;
                    ++attention_nodes;
                    if (kernel.name != attention_name || kernel.block_x != attention.blockDim.x ||
                        kernel.grid_x != attention.gridDim.x || kernel.grid_y != attention.gridDim.y ||
                        kernel.dynamic_shared_memory_bytes != attention.sharedMemBytes)
                    { setError("timing attention differs from the production graph"); return false; }
                }
                if (attention_nodes != replays || !owner->instantiate())
                { setError("default timing graph instantiation/inventory failed"); return false; }
            }
            return true;
        }

        /** @brief Poison both consumers before a timing-owner replay.
         * @return Whether poison was ordered on the exact stream outside timing.
         */
        bool poisonDefaultTimingState()
        {
            const size_t bytes = size_t(current_query_rows_) * geometry_.n_heads * geometry_.head_dim * sizeof(float);
            return default_workspace_ &&
                check(hipMemsetAsync(d_direct_output_, 0xff, bytes, stream_), "poison timed output") &&
                check(hipMemsetAsync(default_workspace_->getBuffer(llaminar2::rocm::AttentionWorkspaceBuffers::DEVICE_PARAMS),
                    0xff, sizeof(AttentionDeviceParams), stream_), "poison timed parameters");
        }

        /** @brief Observe the timing graph's result without replaying a repairing oracle.
         * @param live_kv_len Expected device-published key extent.
         * @param output Receives bytes from the last timed transaction.
         * @return Whether publication and terminal observation are complete.
         */
        bool observeDefaultTimingBytes(int live_kv_len, std::vector<float> &output)
        {
            output.resize(size_t(current_query_rows_) * geometry_.n_heads * geometry_.head_dim);
            AttentionDeviceParams published{};
            /** @brief Retire partial observations before either borrowed host owner. */
            struct ObservationRetirement
            {
                hipStream_t stream;
                bool joined = false;
                /** @brief Error returns also drain earlier successful asynchronous copies. */
                ~ObservationRetirement()
                {
                    if (!joined && hipStreamSynchronize(stream) != hipSuccess) std::terminate();
                }
            } retirement{stream_};
            if (!check(hipMemcpyAsync(output.data(), d_direct_output_, output.size()*sizeof(float),
                    hipMemcpyDeviceToHost, stream_), "observe timed output") ||
                !check(hipMemcpyAsync(&published, default_workspace_->getBuffer(llaminar2::rocm::AttentionWorkspaceBuffers::DEVICE_PARAMS),
                    sizeof(published), hipMemcpyDeviceToHost, stream_), "observe timed publication") ||
                !check(hipStreamSynchronize(stream_), "timed observation terminal completion")) return false;
            retirement.joined = true;
            if (published.kv_len != live_kv_len || published.kv_stride != kv_capacity_ ||
                published.position_offset != live_kv_len - current_query_rows_ ||
                published.mask_stride != live_kv_len || published.ring_row_origin != 0 || published.ring_row_capacity != 0)
            { setError("timing graph did not overwrite the complete parameter publication"); return false; }
            return true;
        }

        /**
         * @brief Inspect the actual automatic attention node outside timed replay.
         * @param native Receives its launch function, grid and LDS allocation.
         * @return False unless precisely one retained kernel owns attention LDS.
         */
        bool defaultKernelInventory(hipKernelNodeParams &native)
        {
            std::array<hipGraphNode_t, 2> nodes{};
            std::size_t count = default_graph_.node_count;
            if (!default_graph_.graph || count != nodes.size() ||
                !check(hipGraphGetNodes(default_graph_.graph, nodes.data(), &count),
                       "default inventory nodes")) return false;
            int attention_nodes = 0;
            for (const auto node : nodes)
            {
                hipKernelNodeParams parameters{};
                if (!check(hipGraphKernelNodeGetParams(node, &parameters),
                           "default inventory kernel")) return false;
                if (parameters.sharedMemBytes)
                {
                    native = parameters;
                    ++attention_nodes;
                }
            }
            if (attention_nodes != 1)
            {
                setError("default inventory must contain one attention LDS owner");
                return false;
            }
            return true;
        }

        /** @return Native symbol attached to the observed node on its exact stream. */
        [[nodiscard]] const char *defaultKernelName(const hipKernelNodeParams &native) const
        {
            return hipKernelNameRefByPtr(native.func, stream_);
        }

        /** Time the retained automatic producer/attention transaction. */
        [[nodiscard]] TransactionTiming measureDefault(int live_kv_len)
        {
            return measureGraph(default_graph_, live_kv_len);
        }

        /**
         * @brief Measure one complete retained producer/attention batch for paired sampling.
         * @param live_kv_len Device input published before the start event.
         * @return Per-operation microseconds, or zero after a native failure.
         * No warmup or oracle enters this interval; callers alternate paired engines.
         */
        [[nodiscard]] double measureDefaultBatch(int live_kv_len)
        {
            using namespace llaminar2;
            auto &owner = default_timing_[live_kv_len < 2048 ? 1 : 0];
            if (!ready_ || !owner || !publishLiveLength(live_kv_len) || !owner->launch() ||
                !check(hipEventRecord(stop_, stream_), "internal timing terminal event") ||
                !check(hipEventSynchronize(stop_), "internal timing completion")) return 0.0;
            const auto observed = owner->consumeOrderedTimelineTiming();
            if (observed.state != GPUOrderedTimelineTimingState::Complete || observed.samples.size() != 1 ||
                observed.samples.front().name != "complete_attention_batch" ||
                observed.samples.front().kind != GPUOrderedTimelineStepKind::CapturedFragment ||
                !std::isfinite(observed.samples.front().elapsed_ms) || observed.samples.front().elapsed_ms <= 0 ||
                owner->consumeOrderedTimelineTiming().state != GPUOrderedTimelineTimingState::AwaitingLaunch)
            { setError("internal timing interval is missing, stale or has the wrong owner"); return 0.0; }
            return observed.samples.front().elapsed_ms * 1000.0 / defaultBatchSize(live_kv_len);
        }

        /** @return Complete default graph transactions in one native-event sample.
         * Short-prefix samples retain sixteen operations so submillisecond graph
         * intervals do not make paired comparisons depend on clock transitions.
         * Long prefixes retain the bounded workload budget used by other panels.
         */
        [[nodiscard]] int defaultBatchSize(int live_kv_len) const noexcept
        {
            return live_kv_len < 2048 ? 16 : 1;
        }

        /**
         * @brief Retain automatic bytes after dirty canonical and repeated-replay checks.
         * @param live_kv_len Visible K/V rows for both retained schedules.
         * @param output Receives the complete fixed-query output outside timing.
         * @return Whether the context oracle, producer publication and copy succeeded.
         */
        bool validatedDefaultBytes(int live_kv_len, std::vector<float> &output)
        {
            if (!validateDefaultOutput(live_kv_len)) return false;
            output.resize(static_cast<size_t>(current_query_rows_) * geometry_.n_heads * geometry_.head_dim);
            return check(hipMemcpyAsync(output.data(), d_direct_output_, output.size() * sizeof(float),
                                       hipMemcpyDeviceToHost, stream_), "paired output observation") &&
                check(hipStreamSynchronize(stream_), "paired observation terminal completion");
        }

        /**
         * @brief Inspect every retained context node, including the ordered reducer.
         * @param nodes Receives the native kernel parameters outside replay.
         * @return False unless the complete three-kernel context transaction exists.
         */
        bool contextKernelInventory(std::vector<hipKernelNodeParams> &nodes)
        {
            if (!context_graph_.graph || context_graph_.node_count != 3) return false;
            std::array<hipGraphNode_t, 3> native_nodes{};
            size_t count = native_nodes.size();
            if (!check(hipGraphGetNodes(context_graph_.graph, native_nodes.data(), &count),
                       "paired context inventory")) return false;
            nodes.resize(count);
            for (size_t index = 0; index < count; ++index)
                if (!check(hipGraphKernelNodeGetParams(native_nodes[index], &nodes[index]),
                           "paired context kernel inventory")) return false;
            return count == native_nodes.size();
        }

        /** @brief Inspect the retained one-node direct graph before an exact profile. */
        bool directKernelInventory(hipKernelNodeParams &native)
        {
            if (!direct_graph_.graph || direct_graph_.node_count != 1) return false;
            hipGraphNode_t node = nullptr;
            size_t count = 1;
            return check(hipGraphGetNodes(direct_graph_.graph, &node, &count), "direct inventory") &&
                count == 1 && check(hipGraphKernelNodeGetParams(node, &native), "direct kernel inventory");
        }

        /**
         * @brief Submit one complete automatic graph for isolated profiling.
         * @param live_kv_len Published device input before the measured submission.
         * @return False when capture, submission or terminal observation fails.
         */
        bool replayDefaultOnceForProfiler(int live_kv_len)
        {
            return profileGraph(default_graph_, live_kv_len, "ROCmFA2DefaultExactCandidate");
        }

        /**
         * @brief Prove complete automatic writes and canonical bytes outside timing.
         * @param live_kv_len Visible keys including the current physical query rows.
         * @return False on incomplete publication, nonfinite output or changed bytes.
         *
         * The independent forced-context transaction uses the fixture-owned
         * parameter record, not the automatic producer's workspace record. Its
         * one-partition direct limit forces the canonical summary/reducer at
         * every M512 point. Both outputs and canonical scratch start as NaNs.
         * Each automatic replay poisons its own parameter record as well, so
         * retained producer publication and full-output overwrite are observed.
         * The profiler entry never calls this untimed oracle.
         */
        bool validateDefaultOutput(int live_kv_len)
        {
            if (!ready_ || !default_graph_.exec || !context_graph_.exec ||
                context_plan_.device_direct_partition_limit != 1 ||
                !publishLiveLength(live_kv_len))
            {
                setError("default output validation requires independent retained canonical context");
                return false;
            }
            const size_t elements = static_cast<size_t>(current_query_rows_) *
                geometry_.n_heads * geometry_.head_dim;
            const size_t bytes = elements * sizeof(float);
            std::vector<float> expected(elements), observed(elements), previous;
            if (!check(hipMemsetAsync(d_context_output_, 0xff, bytes, stream_), "poison canonical output") ||
                !check(hipMemsetAsync(d_partial_output_, 0xff, partial_output_bytes_, stream_), "poison canonical numerator") ||
                !check(hipMemsetAsync(d_partial_m_, 0xff, partial_m_bytes_, stream_), "poison canonical maxima") ||
                !check(hipMemsetAsync(d_partial_l_, 0xff, partial_l_bytes_, stream_), "poison canonical denominator") ||
                !check(hipGraphLaunch(context_graph_.exec, stream_), "canonical validation replay") ||
                !check(hipMemcpyAsync(expected.data(), d_context_output_, bytes,
                                      hipMemcpyDeviceToHost, stream_), "read canonical validation output") ||
                !check(hipStreamSynchronize(stream_), "canonical validation terminal completion"))
                return false;
            void *parameters = default_workspace_->getBuffer(
                llaminar2::rocm::AttentionWorkspaceBuffers::DEVICE_PARAMS);
            for (int replay = 0; replay < 2; ++replay)
            {
                AttentionDeviceParams published{};
                if (!check(hipMemsetAsync(d_direct_output_, 0xff, bytes, stream_), "poison automatic output") ||
                    !check(hipMemsetAsync(parameters, 0xff, sizeof(published), stream_), "poison automatic parameters") ||
                    !check(hipGraphLaunch(default_graph_.exec, stream_), "automatic validation replay") ||
                    !check(hipMemcpyAsync(observed.data(), d_direct_output_, bytes,
                                          hipMemcpyDeviceToHost, stream_), "read automatic validation output") ||
                    !check(hipMemcpyAsync(&published, parameters, sizeof(published),
                                          hipMemcpyDeviceToHost, stream_), "read automatic publication") ||
                    !check(hipStreamSynchronize(stream_), "automatic validation terminal completion"))
                    return false;
                if (published.kv_len != live_kv_len || published.kv_stride != kv_capacity_ ||
                    published.position_offset != live_kv_len - current_query_rows_ ||
                    published.mask_stride != live_kv_len ||
                    published.ring_row_origin != 0 || published.ring_row_capacity != 0)
                {
                    setError("automatic retained count producer published the wrong visible geometry");
                    return false;
                }
                for (size_t index = 0; index < elements; ++index)
                    if (!std::isfinite(expected[index]) || !std::isfinite(observed[index]))
                    {
                        setError("canonical or automatic validation output contains poison/nonfinite data");
                        return false;
                    }
                if (std::memcmp(expected.data(), observed.data(), bytes) != 0 ||
                    (!previous.empty() && std::memcmp(previous.data(), observed.data(), bytes) != 0))
                {
                    setError("automatic FP32 arithmetic changed canonical or retained replay bytes");
                    return false;
                }
                previous = observed;
            }
            return true;
        }

        /** Time the already captured direct transaction at one live K/V length. */
        [[nodiscard]] TransactionTiming measureDirect(int live_kv_len)
        {
            return measureGraph(direct_graph_, live_kv_len);
        }

        /** Time the already captured context transaction at one live K/V length. */
        [[nodiscard]] TransactionTiming measureContext(int live_kv_len)
        {
            return measureGraph(context_graph_, live_kv_len);
        }

        /**
         * @brief Replay one exact transaction for an external profiler.
         *
         * Parameter publication is completed before the graph launch. The only
         * blocking operation is the terminal event after the transaction, so a
         * rocprof kernel filter observes one exact phase/reducer candidate rather
         * than a benchmark loop containing unrelated launch geometries.
         */
        bool replayOnceForProfiler(bool context, int live_kv_len)
        {
            CapturedHIPGraph &captured = context ? context_graph_ : direct_graph_;
            return profileGraph(captured, live_kv_len,
                context ? "ROCmFA2ContextExactCandidate" : "ROCmFA2DirectExactCandidate");
        }

    private:
        /**
         * @brief Collect one exact retained graph after untimed publication/warmup.
         * @param captured Complete production graph selected before profiling.
         * @param live_kv_len Device-owned live input published outside the region.
         * @param label Exact transaction named in the ROCTX trace.
         * @return Successful replay, terminal completion and profiler cleanup.
         */
        bool profileGraph(CapturedHIPGraph &captured, int live_kv_len, const char *label)
        {
            if (!captured.exec || !publishLiveLength(live_kv_len)) return false;
            for (int warmup = 0; warmup < 2; ++warmup)
                if (!check(hipGraphLaunch(captured.exec, stream_), "profiler warmup replay")) return false;
            if (!check(hipEventRecord(stop_, stream_), "profiler warmup terminal event") ||
                !check(hipEventSynchronize(stop_), "profiler warmup completion")) return false;
            if (roctxRangePushA(label) < 0)
            {
                setError("exact profiler range publication failed");
                return false;
            }
            /** Retire profiler state even when a native error unwinds the observation. */
            struct ProfileRegion
            {
                bool resumed = false, open = true;
                /** @return Successful pause/pop while retiring the scope exactly once. */
                bool finish()
                {
                    if (!open) return true;
                    open = false;
                    const int paused = resumed ? roctxProfilerPause(0) : 0;
                    const int popped = roctxRangePop();
                    return paused == 0 && popped >= 0;
                }
                /** Unconditionally restore paused profiling before leaving the helper. */
                ~ProfileRegion() { (void)finish(); }
            } region;
            if (roctxProfilerResume(0) != 0)
            {
                setError("exact profiler resume failed");
                return false;
            }
            region.resumed = true;
            const bool launched = check(hipGraphLaunch(captured.exec, stream_), "exact profiler replay");
            const bool recorded = check(hipEventRecord(stop_, stream_), "exact profiler terminal event");
            const bool joined = recorded && check(hipEventSynchronize(stop_), "exact profiler completion");
            const bool retired = region.finish();
            if (!retired) setError("exact profiler pause/range cleanup failed");
            return launched && joined && retired;
        }

        /** Record the first failure so callers receive the original cause. */
        void setError(const std::string &message)
        {
            if (error_.empty())
                error_ = message;
            ready_ = false;
        }

        /** Convert one HIP status into a stable benchmark failure. */
        bool check(hipError_t status, const char *operation)
        {
            if (status == hipSuccess)
                return true;
            std::ostringstream message;
            message << operation << ": " << hipGetErrorString(status);
            setError(message.str());
            return false;
        }

        /** Allocate one persistent device buffer and retain an actionable label. */
        bool allocate(void **pointer, std::size_t bytes, const char *label)
        {
            if (!pointer || bytes == 0)
            {
                setError(std::string("invalid allocation for ") + label);
                return false;
            }
            const hipError_t status = hipMalloc(pointer, bytes);
            if (status == hipSuccess)
                return true;
            std::ostringstream message;
            message << "hipMalloc(" << label << ", " << bytes
                    << " bytes): " << hipGetErrorString(status);
            setError(message.str());
            return false;
        }

        /** Build one immutable capture-time policy request. */
        [[nodiscard]] rocm_policy::ROCmFA2PrefillParallelGeometry
        makePolicyGeometry(
            llaminar2::attention::AttentionPrefillParallelAxis axis) const
        {
            return {
                .batch_size = 1,
                .query_rows = current_query_rows_,
                .local_query_heads = geometry_.n_heads,
                .head_dim = geometry_.head_dim,
                .kv_capacity = kv_capacity_,
                .compute_unit_count = compute_unit_count_,
                .lds_capacity_bytes = lds_capacity_bytes_,
                .requested_axis = axis,
                .kv_format = format_ == NativeKVFormat::FP16 ? llaminar2::TensorType::FP16
                           : format_ == NativeKVFormat::BF16 ? llaminar2::TensorType::BF16
                           : format_ == NativeKVFormat::Q8_1 ? llaminar2::TensorType::Q8_1
                                                          : llaminar2::TensorType::FP32,
                .visibility = rocm_policy::ROCmFA2VisibilityPolicy::CausalFullContext,
            };
        }

        /**
         * @brief Fill Q/K/V with cheap, deterministic, non-uniform bit patterns.
         *
         * A zero-filled performance arena makes every value row identical and
         * lets broken attention arithmetic accidentally pass a byte comparison.
         * The performance harness does not need expensive random generation,
         * but it does need different K/V regions to affect online-softmax state.
         * Thirty-two stream-ordered memsets provide that signal once during
         * setup without adding a benchmark-only CUDA kernel to production.
         *
         * @param q_elements Number of FP32 query elements in the arena.
         * @param kv_elements Number of FP16 elements in each K/V arena.
         * @return True when every asynchronous initialization was submitted.
         */
        bool initializeDeterministicTensorPatterns(
            size_t q_elements,
            size_t kv_elements)
        {
            if (hipMemsetAsync(
                    d_q_, 0x3c,
                    q_elements * sizeof(float), stream_) != hipSuccess)
            {
                return false;
            }

            constexpr size_t kPatternRegions = 32;
            const size_t region_elements =
                (kv_elements + kPatternRegions - 1) / kPatternRegions;
            for (size_t region = 0; region < kPatternRegions; ++region)
            {
                const size_t begin = region * region_elements;
                if (begin >= kv_elements)
                    break;
                const size_t count =
                    std::min(region_elements, kv_elements - begin);
                const int k_pattern = 0x28 + static_cast<int>(region % 12);
                const int v_pattern = 0x30 + static_cast<int>((region * 5) % 12);
                if (hipMemsetAsync(
                        static_cast<unsigned char *>(d_k_) +
                            begin * sizeof(uint16_t),
                        k_pattern,
                        count * sizeof(uint16_t),
                        stream_) != hipSuccess ||
                    hipMemsetAsync(
                        static_cast<unsigned char *>(d_v_) +
                            begin * sizeof(uint16_t),
                        v_pattern,
                        count * sizeof(uint16_t),
                        stream_) != hipSuccess)
                {
                    return false;
                }
            }
            return true;
        }

        /** Create stream/events and all maximum-capacity device buffers. */
        void initialize()
        {
            if (device_ < 0 || geometry_.n_heads <= 0 || geometry_.n_kv_heads <= 0 ||
                geometry_.head_dim <= 0 || maximum_query_rows_ <= 0 ||
                kv_capacity_ <= 0 || compute_unit_count_ <= 0 ||
                lds_capacity_bytes_ == 0)
            {
                setError("invalid persistent benchmark geometry");
                return;
            }
            if (!check(hipSetDevice(device_), "benchmark device binding") || !check(
                    hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
                    "hipStreamCreateWithFlags") ||
                !check(hipEventCreate(&start_), "hipEventCreate(start)") ||
                !check(hipEventCreate(&stop_), "hipEventCreate(stop)"))
            {
                return;
            }

            current_query_rows_ = maximum_query_rows_;
            const rocm_policy::ROCmFA2PrefillParallelPlan maximum_plan =
                rocm_policy::selectROCmFA2PrefillParallelPlan(
                    makePolicyGeometry(
                        llaminar2::attention::AttentionPrefillParallelAxis::
                            KeyValueContext));
            if (!maximum_plan.valid || !maximum_plan.usesContextParallelism())
            {
                setError("maximum context workspace plan is invalid");
                return;
            }

            const std::size_t output_elements =
                static_cast<std::size_t>(maximum_query_rows_) *
                static_cast<std::size_t>(geometry_.n_heads) *
                static_cast<std::size_t>(geometry_.head_dim);
            const std::size_t q_bytes = output_elements * sizeof(float);
            const std::size_t kv_bytes = nativeKVBytes(
                format_,
                kv_capacity_,
                geometry_.n_kv_heads,
                geometry_.head_dim);
            partial_output_bytes_ = maximum_plan.partial_output_bytes;
            partial_m_bytes_ = maximum_plan.partial_m_bytes;
            partial_l_bytes_ = maximum_plan.partial_l_bytes;

            if (!allocate(&d_q_, q_bytes, "Q") ||
                !allocate(&d_k_, kv_bytes, "K") ||
                !allocate(&d_v_, kv_bytes, "V") ||
                !allocate(&d_direct_output_, q_bytes, "direct output") ||
                !allocate(&d_context_output_, q_bytes, "context output") ||
                !allocate(
                    &d_partial_output_,
                    partial_output_bytes_,
                    "context output summaries") ||
                !allocate(&d_partial_m_, partial_m_bytes_, "context m summaries") ||
                !allocate(&d_partial_l_, partial_l_bytes_, "context l summaries") ||
                !allocate(
                    reinterpret_cast<void **>(&d_params_),
                    sizeof(AttentionDeviceParams),
                    "attention device parameters"))
            {
                return;
            }

            // The automatic FP16 scaling fixture uses exactly CUDA's bytes.
            // Other native-format tournaments retain their original zero recipe.
            const bool inputs_initialized = format_ == NativeKVFormat::FP16
                ? initializeDeterministicTensorPatterns(output_elements, kv_bytes / sizeof(uint16_t))
                : check(hipMemsetAsync(d_q_, 0, q_bytes, stream_), "memset Q") &&
                  check(hipMemsetAsync(d_k_, 0, kv_bytes, stream_), "memset K") &&
                  check(hipMemsetAsync(d_v_, 0, kv_bytes, stream_), "memset V");
            if (!inputs_initialized)
            {
                setError("Deterministic attention input initialization failed");
                return;
            }
            if (
                !check(
                    hipMemsetAsync(d_direct_output_, 0, q_bytes, stream_),
                    "memset direct output") ||
                !check(
                    hipMemsetAsync(d_context_output_, 0, q_bytes, stream_),
                    "memset context output") ||
                !publishLiveLength(maximum_query_rows_) ||
                !check(hipStreamSynchronize(stream_), "initialization completion"))
            {
                return;
            }
            ready_ = true;
        }

        /** Publish device-owned live context for the next graph replay. */
        bool publishLiveLength(int live_kv_len)
        {
            if (live_kv_len < current_query_rows_ || live_kv_len > kv_capacity_)
            {
                setError("live K/V length is outside captured capacity");
                return false;
            }
            if (d_default_live_length_ &&
                !check(hipMemcpyAsync(d_default_live_length_, &live_kv_len, sizeof(live_kv_len),
                                      hipMemcpyHostToDevice, stream_), "publish default KV count"))
                return false;
            const AttentionDeviceParams params{
                .kv_len = live_kv_len,
                .kv_stride = kv_capacity_,
                .position_offset = live_kv_len - current_query_rows_,
                .mask_stride = kv_capacity_,
            };
            return check(
                hipMemcpyAsync(
                    d_params_,
                    &params,
                    sizeof(params),
                    hipMemcpyHostToDevice,
                    stream_),
                "publish attention parameters");
        }

        /** Dispatch one direct production wrapper for the selected native format. */
        [[nodiscard]] int launchDirect()
        {
            const auto invoke = [&](auto launcher) -> int
            {
                return launcher(
                    static_cast<const float *>(d_q_),
                    d_k_,
                    d_v_,
                    static_cast<float *>(d_direct_output_),
                    1,
                    current_query_rows_,
                    kv_capacity_,
                    geometry_.n_heads,
                    geometry_.n_kv_heads,
                    geometry_.head_dim,
                    true,
                    -1,
                    0,
                    d_params_,
                    nullptr,
                    static_cast<void *>(stream_),
                    geometry_.head_start,
                    geometry_.replicated_gqa_n_rep);
            };
            switch (format_)
            {
            case NativeKVFormat::FP32:
                return hipFlashAttn_prefill_fa2(
                    static_cast<const float *>(d_q_),
                    static_cast<const float *>(d_k_),
                    static_cast<const float *>(d_v_),
                    static_cast<float *>(d_direct_output_),
                    1,
                    current_query_rows_,
                    kv_capacity_,
                    geometry_.n_heads,
                    geometry_.n_kv_heads,
                    geometry_.head_dim,
                    true,
                    -1,
                    0,
                    d_params_,
                    nullptr,
                    static_cast<void *>(stream_),
                    geometry_.head_start,
                    geometry_.replicated_gqa_n_rep);
            case NativeKVFormat::FP16:
                return invoke(hipFlashAttn_prefill_fa2_fp16);
            case NativeKVFormat::BF16:
                return invoke(hipFlashAttn_prefill_fa2_bf16);
            case NativeKVFormat::Q8_1:
                return invoke(hipFlashAttn_prefill_fa2_q8_1);
            }
            return -1;
        }

        /** Dispatch one forced context transaction for the selected native format. */
        [[nodiscard]] int launchContext()
        {
            const auto invoke = [&](auto launcher) -> int
            {
                return launcher(
                    static_cast<const float *>(d_q_),
                    d_k_,
                    d_v_,
                    static_cast<float *>(d_context_output_),
                    static_cast<float *>(d_partial_output_),
                    static_cast<float *>(d_partial_m_),
                    static_cast<float *>(d_partial_l_),
                    1,
                    current_query_rows_,
                    kv_capacity_,
                    geometry_.n_heads,
                    geometry_.n_kv_heads,
                    geometry_.head_dim,
                    true,
                    -1,
                    0,
                    d_params_,
                    nullptr,
                    rocm_policy::kROCmFA2CanonicalContextPartitionKeys,
                    context_plan_.max_context_partitions,
                    context_plan_.context_partition_slots,
                    context_plan_.context_phase_block_slots,
                    context_plan_.device_direct_partition_limit,
                    context_plan_.reducer_dimension_wavefronts,
                    context_plan_.reducer_block_slots,
                    static_cast<void *>(stream_),
                    geometry_.head_start,
                    geometry_.replicated_gqa_n_rep);
            };
            switch (format_)
            {
            case NativeKVFormat::FP32:
                return invoke(hipFlashAttn_prefill_fa2_context_parallel);
            case NativeKVFormat::FP16:
                return invoke(hipFlashAttn_prefill_fa2_fp16_context_parallel);
            case NativeKVFormat::BF16:
                return invoke(hipFlashAttn_prefill_fa2_bf16_context_parallel);
            case NativeKVFormat::Q8_1:
                return invoke(hipFlashAttn_prefill_fa2_q8_1_context_parallel);
            }
            return -1;
        }

        /** Capture and instantiate one direct or context graph transaction. */
        bool captureGraph(CapturedHIPGraph &captured, bool context)
        {
            captured.reset();
            if (!check(hipStreamSynchronize(stream_), "pre-capture completion") ||
                !check(
                    hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal),
                    "hipStreamBeginCapture"))
            {
                return false;
            }

            const int launch_status = context ? launchContext() : launchDirect();
            hipGraph_t graph = nullptr;
            const hipError_t end_status = hipStreamEndCapture(stream_, &graph);
            if (launch_status != 0)
            {
                if (graph)
                    (void)hipGraphDestroy(graph);
                setError(context ? "context launcher rejected capture"
                                 : "direct launcher rejected capture");
                return false;
            }
            if (!check(end_status, "hipStreamEndCapture") || !graph)
                return false;

            captured.graph = graph;
            if (!check(
                    hipGraphGetNodes(
                        captured.graph,
                        nullptr,
                        &captured.node_count),
                    "hipGraphGetNodes") ||
                !check(
                    hipGraphInstantiate(
                        &captured.exec,
                        captured.graph,
                        nullptr,
                        nullptr,
                        0),
                    "hipGraphInstantiate"))
            {
                captured.reset();
                return false;
            }
            const std::size_t expected_nodes = context ? 3U : 1U;
            if (captured.node_count != expected_nodes)
            {
                std::ostringstream message;
                message << (context ? "context" : "direct")
                        << " graph captured " << captured.node_count
                        << " nodes; expected " << expected_nodes;
                setError(message.str());
                captured.reset();
                return false;
            }
            return true;
        }

        /** Capture the one-node direct graph for the current query bucket. */
        bool captureDirectGraph()
        {
            return captureGraph(direct_graph_, /*context=*/false);
        }

        /** Select a bounded replay batch from the actual work cardinality. */
        [[nodiscard]] int replayBatchSize(int live_kv_len) const noexcept
        {
            const long double work =
                static_cast<long double>(current_query_rows_) *
                static_cast<long double>(live_kv_len) *
                static_cast<long double>(geometry_.n_heads) *
                static_cast<long double>(geometry_.head_dim);
            if (work >= 1.0e9L)
                return 1;
            if (work >= 2.0e8L)
                return 2;
            if (work >= 5.0e7L)
                return 4;
            return 8;
        }

        /** Time batches of complete graph replays and return their median. */
        [[nodiscard]] TransactionTiming measureGraph(
            CapturedHIPGraph &captured,
            int live_kv_len)
        {
            TransactionTiming result{
                .median_us = 0.0,
                .replays_per_sample = 0,
                .graph_nodes = captured.node_count,
            };
            if (!ready_ || !captured.exec || !publishLiveLength(live_kv_len))
                return result;

            constexpr int warmup_replays = 2;
            for (int replay = 0; replay < warmup_replays; ++replay)
            {
                if (!check(
                        hipGraphLaunch(captured.exec, stream_),
                        "warmup hipGraphLaunch"))
                {
                    return result;
                }
            }
            if (!check(hipEventRecord(stop_, stream_), "warmup stop event") ||
                !check(hipEventSynchronize(stop_), "warmup completion"))
            {
                return result;
            }

            constexpr int sample_count = 3;
            const int replay_count = replayBatchSize(live_kv_len);
            std::array<double, sample_count> samples{};
            for (int sample = 0; sample < sample_count; ++sample)
            {
                samples[sample] = measureGraphBatch(captured, replay_count);
                if (!(samples[sample] > 0.0)) return result;
            }
            result.samples_us = samples;
            std::sort(samples.begin(), samples.end());
            result.median_us = samples[sample_count / 2];
            result.replays_per_sample = replay_count;
            return result;
        }

        /**
         * @brief Time one batch using the existing exact-stream terminal-event protocol.
         * @param captured Prepared complete graph transaction.
         * @param replays Positive number of transactions inside the event interval.
         * @return Native-event microseconds per transaction, or zero on failure.
         */
        [[nodiscard]] double measureGraphBatch(CapturedHIPGraph &captured, int replays)
        {
            if (!captured.exec || replays <= 0 ||
                !check(hipEventRecord(start_, stream_), "timing start event")) return 0.0;
            for (int replay = 0; replay < replays; ++replay)
                if (!check(hipGraphLaunch(captured.exec, stream_), "timed hipGraphLaunch")) return 0.0;
            if (!check(hipEventRecord(stop_, stream_), "timing stop event") ||
                !check(hipEventSynchronize(stop_), "timing completion")) return 0.0;
            float milliseconds = 0.0f;
            if (!check(hipEventElapsedTime(&milliseconds, start_, stop_), "hipEventElapsedTime")) return 0.0;
            return static_cast<double>(milliseconds) * 1000.0 / replays;
        }

        /** @brief Submit the complete production parameter producer and query body.
         * @return Whether both exact-stream operations were accepted.
         * This shared source prevents timing from omitting production work.
         */
        bool enqueueDefaultTransaction()
        {
            using namespace llaminar2;
            const DeviceId device = DeviceId::rocm(device_);
            GpuTensorView query(d_q_, current_query_rows_, geometry_.n_heads * geometry_.head_dim,
                                TensorType::FP32, device);
            GpuTensorView keys(d_k_, kv_capacity_, geometry_.n_kv_heads * geometry_.head_dim,
                               TensorType::FP16, device);
            GpuTensorView values(d_v_, kv_capacity_, geometry_.n_kv_heads * geometry_.head_dim,
                                 TensorType::FP16, device);
            GpuTensorView output(d_direct_output_, current_query_rows_,
                                 geometry_.n_heads * geometry_.head_dim, TensorType::FP32, device);
            const attention::AttentionExecutionPolicy automatic{
                .prefill_parallel_axis = attention::AttentionPrefillParallelAxis::GeometrySelected};
            const attention::AttentionPrefillCaptureGeometry capture{
                .batch_size = 1, .query_rows = current_query_rows_,
                .local_query_heads = geometry_.n_heads, .head_dim = geometry_.head_dim,
                .kv_capacity = kv_capacity_, .execution_policy = automatic,
                .native_kv_format = TensorType::FP16,
                .visibility = attention::AttentionVisibilityPolicy::CausalFullContext};
            return default_kernel_->prepareDynamicAttnParamsFromDeviceSequenceState(
                    d_default_live_length_, current_query_rows_, 1, stream_, kv_capacity_, nullptr, capture) &&
                    default_kernel_->compute_tensor(&query, &keys, &values, &output,
                        1, current_query_rows_, kv_capacity_, geometry_.n_heads,
                        geometry_.n_kv_heads, geometry_.head_dim, true, -1, nullptr, nullptr,
                        nullptr, 0, geometry_.head_start, geometry_.n_heads, geometry_.n_kv_heads,
                        geometry_.replicated_gqa_n_rep, automatic);
        }

        std::array<std::unique_ptr<llaminar2::HIPGraphCapture>,2> default_timing_; ///< Internal event parents, retired first.
        std::array<std::unique_ptr<llaminar2::IGPUGraphCapture>,2> default_timing_fragments_; ///< Complete production batches.
        CapturedHIPGraph default_graph_; ///< Complete production producer/attention transaction.
        std::unique_ptr<llaminar2::DeviceWorkspaceManager> default_workspace_; ///< Exact parameter owner.
        std::unique_ptr<llaminar2::rocm::ROCmFlashAttentionKernelT<
            llaminar2::ActivationPrecision::FP32>> default_kernel_; ///< Automatic production dispatch.
        int *d_default_live_length_ = nullptr; ///< Device-owned live KV input.
        ParticipantAttentionGeometry geometry_;
        NativeKVFormat format_ = NativeKVFormat::FP16;
        int maximum_query_rows_ = 0;
        int kv_capacity_ = 0;
        int compute_unit_count_ = 0;
        std::size_t lds_capacity_bytes_ = 0;
        int device_ = 0; ///< Actual visible native owner, retained by all banks and graphs.
        int current_query_rows_ = 0;
        rocm_policy::ROCmFA2PrefillParallelPlan context_plan_{};
        CapturedHIPGraph direct_graph_{};
        CapturedHIPGraph context_graph_{};
        hipStream_t stream_ = nullptr;
        hipEvent_t start_ = nullptr;
        hipEvent_t stop_ = nullptr;
        void *d_q_ = nullptr;
        void *d_k_ = nullptr;
        void *d_v_ = nullptr;
        void *d_direct_output_ = nullptr;
        void *d_context_output_ = nullptr;
        void *d_partial_output_ = nullptr;
        void *d_partial_m_ = nullptr;
        void *d_partial_l_ = nullptr;
        AttentionDeviceParams *d_params_ = nullptr;
        std::size_t partial_output_bytes_ = 0;
        std::size_t partial_m_bytes_ = 0;
        std::size_t partial_l_bytes_ = 0;
        bool ready_ = false;
        std::string error_;
    };

    /** Add one positive candidate while preserving sorted uniqueness. */
    void addCandidate(std::vector<int> &candidates, int value)
    {
        if (value <= 0 ||
            std::find(candidates.begin(), candidates.end(), value) !=
                candidates.end())
        {
            return;
        }
        candidates.push_back(value);
        std::sort(candidates.begin(), candidates.end());
    }

    /**
     * @brief Build a logarithmic slot tournament plus the current policy point.
     *
     * Persistent slot counts do not alter arithmetic. Powers of two expose each
     * occupancy regime without measuring every integer through a 1024-partition
     * graph, while the exact generic-policy and full-partition values prevent the
     * sparse lattice from missing either installed or unconstrained execution.
     */
    [[nodiscard]] std::vector<int> contextSlotCandidates(
        int max_partitions,
        int policy_slots)
    {
        std::vector<int> candidates;
        for (int slots = 1; slots > 0 && slots <= max_partitions; slots *= 2)
        {
            addCandidate(candidates, slots);
            if (slots > max_partitions / 2)
                break;
        }
        addCandidate(candidates, policy_slots);
        addCandidate(candidates, max_partitions);
        return candidates;
    }

    /**
     * @brief Build the persistent reducer-grid lattice used by the tournament.
     *
     * Powers of two expose launch-overhead knees. Half-, one-, and two-CU
     * envelopes test physical residency directly, while the policy and complete
     * logical grids ensure the installed point and historical schedule are both
     * represented. Values above the logical task count cannot do useful work
     * and are intentionally excluded.
     */
    [[nodiscard]] std::vector<int> reducerBlockCandidates(
        int logical_blocks,
        int compute_unit_count,
        int policy_blocks)
    {
        std::vector<int> candidates;
        for (int blocks = 1;
             blocks > 0 && blocks <= logical_blocks;
             blocks *= 2)
        {
            addCandidate(candidates, blocks);
            if (blocks > logical_blocks / 2)
                break;
        }
        addCandidate(candidates, (compute_unit_count + 1) / 2);
        addCandidate(candidates, compute_unit_count);
        addCandidate(candidates, 2 * compute_unit_count);
        addCandidate(candidates, policy_blocks);
        addCandidate(candidates, logical_blocks);
        candidates.erase(
            std::remove_if(
                candidates.begin(),
                candidates.end(),
                [logical_blocks](int value)
                {
                    return value > logical_blocks;
                }),
            candidates.end());
        return candidates;
    }

    /**
     * @brief Build the non-dominated physical phase-grid tournament lattice.
     *
     * The selected FA2 tiles permit only one resident workgroup per MI50 CU.
     * Candidate widths through eight queued CU waves are still measured to
     * expose scheduler/striding effects. A capacity-sized physical grid is
     * excluded because it cannot increase residency and recreates the proven
     * short-context empty-workgroup pathology.
     */
    [[nodiscard]] std::vector<int> phaseBlockCandidates(
        int logical_blocks,
        int compute_unit_count,
        int policy_blocks)
    {
        std::vector<int> candidates;
        for (const int blocks : {1, 2, 4, 8, 16})
            addCandidate(candidates, blocks);
        addCandidate(candidates, (compute_unit_count + 1) / 2);
        addCandidate(candidates, compute_unit_count);
        addCandidate(candidates, 2 * compute_unit_count);
        addCandidate(candidates, 4 * compute_unit_count);
        addCandidate(candidates, 8 * compute_unit_count);
        addCandidate(candidates, policy_blocks);
        candidates.erase(
            std::remove_if(
                candidates.begin(),
                candidates.end(),
                [logical_blocks](int value)
                {
                    return value > logical_blocks;
                }),
            candidates.end());
        return candidates;
    }

    /** @return Positive integer environment value or zero when absent/invalid. */
    [[nodiscard]] int positiveEnvironmentValue(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return 0;
        const long parsed = std::strtol(raw, nullptr, 10);
        return parsed > 0 && parsed <= std::numeric_limits<int>::max()
                   ? static_cast<int>(parsed)
                   : 0;
    }

    /** @return Environment string or an empty string when unset. */
    [[nodiscard]] std::string environmentString(const char *name)
    {
        const char *raw = std::getenv(name);
        return raw ? std::string(raw) : std::string{};
    }

    /** Deterministic ownership of disjoint geometry/format benchmark bundles. */
    struct DeterministicSweepShard
    {
        int index = 0; ///< Zero-based worker index.
        int count = 1; ///< Total cooperating workers.

        /** @return True when the shard declaration is internally consistent. */
        [[nodiscard]] bool valid() const noexcept
        {
            return count > 0 && index >= 0 && index < count;
        }

        /** @return True when this worker owns the given global bundle ordinal. */
        [[nodiscard]] bool owns(std::size_t ordinal) const noexcept
        {
            return valid() &&
                   ordinal % static_cast<std::size_t>(count) ==
                       static_cast<std::size_t>(index);
        }
    };

    /** Parse an integer environment value without making zero mean "unset". */
    [[nodiscard]] int environmentInteger(const char *name, int fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return fallback;
        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (!end || *end != '\0' ||
            parsed < std::numeric_limits<int>::min() ||
            parsed > std::numeric_limits<int>::max())
        {
            return fallback;
        }
        return static_cast<int>(parsed);
    }

    /**
     * @brief Read deterministic process sharding for multi-GPU sweep launches.
     *
     * A bundle is one persistent geometry/format arena. All M, live-context,
     * slot, and reducer candidates for that arena remain on one GPU, preserving
     * timing locality while guaranteeing that cooperating processes neither
     * duplicate nor omit work.
     */
    [[nodiscard]] DeterministicSweepShard sweepShardFromEnvironment()
    {
        return {
            .index = environmentInteger(
                "LLAMINAR_ROCM_FA2_SHARD_INDEX",
                0),
            .count = environmentInteger(
                "LLAMINAR_ROCM_FA2_SHARD_COUNT",
                1),
        };
    }
#endif

    /** Device fixture shared by the ROCm context-policy performance gates. */
    class ROCmFlashAttentionContextParallelPerf : public ::testing::Test
    {
    protected:
        /** @brief Admit immutable launch limits before direct benchmark HIP setup. */
        void SetUp() override
        {
#ifdef HAVE_ROCM
            llaminar2::initializeROCmHardwareMetadata();
            int device_count = 0;
            if (hipGetDeviceCount(&device_count) != hipSuccess ||
                device_count <= 0)
            {
                return;
            }
            if (hipSetDevice(0) != hipSuccess ||
                hipGetDeviceProperties(&properties_, 0) != hipSuccess)
            {
                return;
            }
            has_device_ = true;
#endif
        }

#ifdef HAVE_ROCM
        hipDeviceProp_t properties_{};
#endif
        bool has_device_ = false;
    };

    /**
     * @brief Tournament every useful persistent slot/reducer geometry on MI50.
     *
     * The four representative participant shapes cover each compiled head
     * dimension, a sharded compact-GQA topology, and the underfilled replicated
     * GQA topology that motivates context parallelism. Every native format is
     * measured at two M buckets and both medium and long resident contexts.
     */
    TEST_F(
        ROCmFlashAttentionContextParallelPerf,
        ContextSlotReducerTournamentAllNativeFormats)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "ROCm support is disabled";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device is available";

        const std::array<ParticipantAttentionGeometry, 4> representatives{{
            {"Qwen2.5-0.5B@TP2", 7, 1, 64, 0, 0},
            {"Qwen2.5-7B@TP4", 7, 1, 128, 0, 0},
            {"Qwen3.6-35B-A3B@TP8", 2, 2, 256, 0, 8},
            {"Qwen3.5-397B-A17B@TP8", 4, 2, 256, 0, 16},
        }};
        constexpr std::array<int, 2> query_rows{{17, 64}};
        constexpr std::array<int, 3> live_kv_lengths{{256, 8192, 131072}};
        constexpr int kv_capacity = 131072;
        const DeterministicSweepShard shard = sweepShardFromEnvironment();
        ASSERT_TRUE(shard.valid());

        std::cout
            << "\nROCm FA2 context slot/reducer tournament\n"
            << "geometry,format,M,live_kv,local_heads,local_kv_heads,head_dim,"
               "slots,phase_blocks,reducer_wavefronts,reducer_blocks,"
               "context_us,direct_us,speedup\n";

        int measured_candidates = 0;
        std::size_t bundle_ordinal = 0;
        for (const ParticipantAttentionGeometry &geometry : representatives)
        {
            for (const NativeKVFormat format : kNativeKVFormats)
            {
                const std::size_t current_bundle = bundle_ordinal++;
                if (!shard.owns(current_bundle))
                    continue;
                CapturedROCmFA2Benchmark benchmark(
                    geometry,
                    format,
                    query_rows.back(),
                    kv_capacity,
                    properties_.multiProcessorCount,
                    std::max<std::size_t>(
                        properties_.sharedMemPerBlock,
                        rocm_policy::kROCmFA2LDSCapacityBytes));
                ASSERT_TRUE(benchmark.ready())
                    << geometry.sources << ' ' << nativeKVFormatName(format)
                    << ": " << benchmark.error();

                for (const int m : query_rows)
                {
                    ASSERT_TRUE(benchmark.prepareQueryRows(m))
                        << benchmark.error();
                    const rocm_policy::ROCmFA2PrefillParallelPlan policy_plan =
                        benchmark.policyPlan();
                    ASSERT_TRUE(policy_plan.valid);

                    const std::vector<int> slots = contextSlotCandidates(
                        (kv_capacity +
                         rocm_policy::kROCmFA2CanonicalContextPartitionKeys - 1) /
                            rocm_policy::kROCmFA2CanonicalContextPartitionKeys,
                        rocm_policy::selectROCmFA2ContextPartitionSlots(
                            {
                                .batch_size = 1,
                                .query_rows = m,
                                .local_query_heads = geometry.n_heads,
                                .head_dim = geometry.head_dim,
                                .kv_capacity = kv_capacity,
                                .compute_unit_count =
                                    properties_.multiProcessorCount,
                                .lds_capacity_bytes =
                                    rocm_policy::kROCmFA2LDSCapacityBytes,
                                .requested_axis =
                                    llaminar2::attention::
                                        AttentionPrefillParallelAxis::
                                            KeyValueContext,
                            },
                            policy_plan.tile,
                            (kv_capacity +
                             rocm_policy::
                                 kROCmFA2CanonicalContextPartitionKeys -
                             1) /
                                rocm_policy::
                                    kROCmFA2CanonicalContextPartitionKeys));
                    const int maximum_wavefronts =
                        rocm_policy::maximumROCmFA2ReducerWavefronts(
                            geometry.head_dim);

                    std::array<TransactionTiming, live_kv_lengths.size()>
                        direct_timings{};
                    for (std::size_t live_index = 0;
                         live_index < live_kv_lengths.size();
                         ++live_index)
                    {
                        direct_timings[live_index] = benchmark.measureDirect(
                            live_kv_lengths[live_index]);
                        ASSERT_GT(direct_timings[live_index].median_us, 0.0)
                            << benchmark.error();
                    }

                    for (const int slot_count : slots)
                    {
                        for (int wavefronts = 1;
                             wavefronts <= maximum_wavefronts;
                             wavefronts *= 2)
                        {
                            ASSERT_TRUE(benchmark.captureContextCandidate(
                                slot_count,
                                wavefronts))
                                << benchmark.error();
                            for (std::size_t live_index = 0;
                                 live_index < live_kv_lengths.size();
                                 ++live_index)
                            {
                                const TransactionTiming context =
                                    benchmark.measureContext(
                                        live_kv_lengths[live_index]);
                                ASSERT_GT(context.median_us, 0.0)
                                    << benchmark.error();
                                const double speedup =
                                    direct_timings[live_index].median_us /
                                    context.median_us;
                                std::cout
                                    << geometry.sources << ','
                                    << nativeKVFormatName(format) << ','
                                    << m << ','
                                    << live_kv_lengths[live_index] << ','
                                    << geometry.n_heads << ','
                                    << geometry.n_kv_heads << ','
                                    << geometry.head_dim << ','
                                    << slot_count << ','
                                    << benchmark.contextPlan().context_phase_block_slots
                                    << ','
                                    << wavefronts << ','
                                    << benchmark.contextPlan().reducer_block_slots
                                    << ','
                                    << std::fixed << std::setprecision(3)
                                    << context.median_us << ','
                                    << direct_timings[live_index].median_us
                                    << ',' << speedup << '\n';
                                ++measured_candidates;
                            }
                        }
                    }
                }
            }
        }
        EXPECT_GT(measured_candidates, 0);
#endif
    }

    /**
     * @brief Isolate persistent reducer residency across all native formats.
     *
     * The first case is the short-context Qwen27 TP4 domain that exposed the
     * historical 192-workgroup no-op reducer. The two controls retain enough
     * long-context work to reveal under-residency. Context phase slots and
     * dimension wavefronts remain fixed at the generic policy values, so the
     * only changing variable is the physical number of reducer workgroups.
     */
    TEST_F(
        ROCmFlashAttentionContextParallelPerf,
        PersistentReducerBlockTournamentAllNativeFormats)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "ROCm support is disabled";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device is available";

        struct ReducerTournamentCase
        {
            ParticipantAttentionGeometry geometry;
            int query_rows = 0;
        };
        const std::array<ReducerTournamentCase, 3> cases{{
            {
                {"Qwen3.5/3.6-27B@TP4", 6, 1, 256, 0, 6},
                128,
            },
            {
                {"Qwen3.6-35B-A3B@TP8", 2, 2, 256, 0, 8},
                64,
            },
            {
                {"Qwen3.5-397B-A17B@TP8", 4, 2, 256, 0, 16},
                64,
            },
        }};
        constexpr std::array<int, 3> live_kv_lengths{{256, 8192, 131072}};
        constexpr int kv_capacity = 131072;
        const DeterministicSweepShard shard = sweepShardFromEnvironment();
        ASSERT_TRUE(shard.valid());

        std::cout
            << "\nROCm FA2 persistent reducer-block tournament\n"
            << "geometry,format,M,live_kv,slots,phase_blocks,"
               "reducer_wavefronts,reducer_blocks,logical_blocks,context_us,"
               "direct_us,speedup\n";

        int measured_candidates = 0;
        std::size_t bundle_ordinal = 0;
        for (const ReducerTournamentCase &test_case : cases)
        {
            for (const NativeKVFormat format : kNativeKVFormats)
            {
                const std::size_t current_bundle = bundle_ordinal++;
                if (!shard.owns(current_bundle))
                    continue;

                CapturedROCmFA2Benchmark benchmark(
                    test_case.geometry,
                    format,
                    test_case.query_rows,
                    kv_capacity,
                    properties_.multiProcessorCount,
                    std::max<std::size_t>(
                        properties_.sharedMemPerBlock,
                        rocm_policy::kROCmFA2LDSCapacityBytes));
                ASSERT_TRUE(benchmark.ready())
                    << test_case.geometry.sources << ' '
                    << nativeKVFormatName(format) << ": "
                    << benchmark.error();
                ASSERT_TRUE(
                    benchmark.prepareQueryRows(test_case.query_rows))
                    << benchmark.error();
                ASSERT_TRUE(benchmark.captureContextCandidate(0, 0, 0))
                    << benchmark.error();
                const auto generic_context = benchmark.contextPlan();
                ASSERT_TRUE(generic_context.usesContextParallelism());

                const int logical_blocks = static_cast<int>(
                    rocm_policy::rocmFA2ReducerLogicalBlocks(
                        {
                            .batch_size = 1,
                            .query_rows = test_case.query_rows,
                            .local_query_heads =
                                test_case.geometry.n_heads,
                            .head_dim = test_case.geometry.head_dim,
                            .kv_capacity = kv_capacity,
                            .compute_unit_count =
                                properties_.multiProcessorCount,
                            .lds_capacity_bytes =
                                rocm_policy::kROCmFA2LDSCapacityBytes,
                        },
                        generic_context.reducer_dimension_wavefronts));
                ASSERT_GT(logical_blocks, 0);
                const std::vector<int> block_candidates =
                    reducerBlockCandidates(
                        logical_blocks,
                        properties_.multiProcessorCount,
                        generic_context.reducer_block_slots);
                ASSERT_FALSE(block_candidates.empty());

                std::array<TransactionTiming, live_kv_lengths.size()>
                    direct_timings{};
                for (std::size_t live_index = 0;
                     live_index < live_kv_lengths.size();
                     ++live_index)
                {
                    direct_timings[live_index] = benchmark.measureDirect(
                        live_kv_lengths[live_index]);
                    ASSERT_GT(direct_timings[live_index].median_us, 0.0)
                        << benchmark.error();
                }

                for (const int reducer_blocks : block_candidates)
                {
                    ASSERT_TRUE(benchmark.captureContextCandidate(
                        generic_context.context_partition_slots,
                        generic_context.reducer_dimension_wavefronts,
                        reducer_blocks))
                        << benchmark.error();
                    for (std::size_t live_index = 0;
                         live_index < live_kv_lengths.size();
                         ++live_index)
                    {
                        const TransactionTiming context =
                            benchmark.measureContext(
                                live_kv_lengths[live_index]);
                        ASSERT_GT(context.median_us, 0.0)
                            << benchmark.error();
                        const double speedup =
                            direct_timings[live_index].median_us /
                            context.median_us;
                        std::cout
                            << test_case.geometry.sources << ','
                            << nativeKVFormatName(format) << ','
                            << test_case.query_rows << ','
                            << live_kv_lengths[live_index] << ','
                            << generic_context.context_partition_slots << ','
                            << generic_context.context_phase_block_slots << ','
                            << generic_context.reducer_dimension_wavefronts
                            << ',' << reducer_blocks << ',' << logical_blocks
                            << ',' << std::fixed << std::setprecision(3)
                            << context.median_us << ','
                            << direct_timings[live_index].median_us << ','
                            << speedup << '\n';
                        ++measured_candidates;
                    }
                }
            }
        }
        EXPECT_GT(measured_candidates, 0);
#endif
    }

    /**
     * @brief Tune physical phase residency independently of logical ownership.
     *
     * These geometries cover the Qwen27 short-context regression and two
     * underfilled long-context controls. Logical ownership remains one slot per
     * capacity partition and replay activates only live device-owned tasks;
     * changing this candidate therefore measures physical residency alone.
     */
    TEST_F(
        ROCmFlashAttentionContextParallelPerf,
        PersistentPhaseBlockTournamentAllNativeFormats)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "ROCm support is disabled";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device is available";

        struct PhaseTournamentCase
        {
            ParticipantAttentionGeometry geometry;
            int query_rows = 0;
        };
        const std::array<PhaseTournamentCase, 3> cases{{
            {
                {"Qwen3.5/3.6-27B@TP4", 6, 1, 256, 0, 6},
                128,
            },
            {
                {"Qwen3.6-35B-A3B@TP8", 2, 2, 256, 0, 8},
                64,
            },
            {
                {"Qwen3.5-397B-A17B@TP8", 4, 2, 256, 0, 16},
                64,
            },
        }};
        constexpr std::array<int, 3> live_kv_lengths{{256, 8192, 131072}};
        constexpr int kv_capacity = 131072;
        const DeterministicSweepShard shard = sweepShardFromEnvironment();
        ASSERT_TRUE(shard.valid());

        std::cout
            << "\nROCm FA2 persistent phase-block tournament\n"
            << "geometry,format,M,live_kv,logical_slots,phase_blocks,"
               "logical_capacity_blocks,reducer_wavefronts,reducer_blocks,"
               "context_us,direct_us,speedup\n";

        int measured_candidates = 0;
        std::size_t bundle_ordinal = 0;
        for (const PhaseTournamentCase &test_case : cases)
        {
            for (const NativeKVFormat format : kNativeKVFormats)
            {
                const std::size_t current_bundle = bundle_ordinal++;
                if (!shard.owns(current_bundle))
                    continue;

                CapturedROCmFA2Benchmark benchmark(
                    test_case.geometry,
                    format,
                    test_case.query_rows,
                    kv_capacity,
                    properties_.multiProcessorCount,
                    std::max<std::size_t>(
                        properties_.sharedMemPerBlock,
                        rocm_policy::kROCmFA2LDSCapacityBytes));
                ASSERT_TRUE(benchmark.ready())
                    << test_case.geometry.sources << ' '
                    << nativeKVFormatName(format) << ": "
                    << benchmark.error();
                ASSERT_TRUE(
                    benchmark.prepareQueryRows(test_case.query_rows))
                    << benchmark.error();
                ASSERT_TRUE(benchmark.captureContextCandidate(0, 0, 0, 0))
                    << benchmark.error();
                const auto generic_context = benchmark.contextPlan();
                ASSERT_TRUE(generic_context.usesContextParallelism());

                const auto policy_geometry =
                    rocm_policy::ROCmFA2PrefillParallelGeometry{
                        .batch_size = 1,
                        .query_rows = test_case.query_rows,
                        .local_query_heads = test_case.geometry.n_heads,
                        .head_dim = test_case.geometry.head_dim,
                        .kv_capacity = kv_capacity,
                        .compute_unit_count =
                            properties_.multiProcessorCount,
                        .lds_capacity_bytes =
                            rocm_policy::kROCmFA2LDSCapacityBytes,
                    };
                const int logical_capacity_blocks = static_cast<int>(
                    rocm_policy::rocmFA2ContextPhaseLogicalBlocks(
                        policy_geometry,
                        generic_context.tile,
                        generic_context.context_partition_slots));
                ASSERT_GT(logical_capacity_blocks, 0);
                const std::vector<int> block_candidates =
                    phaseBlockCandidates(
                        logical_capacity_blocks,
                        properties_.multiProcessorCount,
                        generic_context.context_phase_block_slots);
                ASSERT_FALSE(block_candidates.empty());

                std::array<TransactionTiming, live_kv_lengths.size()>
                    direct_timings{};
                for (std::size_t live_index = 0;
                     live_index < live_kv_lengths.size();
                     ++live_index)
                {
                    direct_timings[live_index] = benchmark.measureDirect(
                        live_kv_lengths[live_index]);
                    ASSERT_GT(direct_timings[live_index].median_us, 0.0)
                        << benchmark.error();
                }

                for (const int phase_blocks : block_candidates)
                {
                    ASSERT_TRUE(benchmark.captureContextCandidate(
                        generic_context.context_partition_slots,
                        generic_context.reducer_dimension_wavefronts,
                        generic_context.reducer_block_slots,
                        phase_blocks))
                        << benchmark.error();
                    for (std::size_t live_index = 0;
                         live_index < live_kv_lengths.size();
                         ++live_index)
                    {
                        const TransactionTiming context =
                            benchmark.measureContext(
                                live_kv_lengths[live_index]);
                        ASSERT_GT(context.median_us, 0.0)
                            << benchmark.error();
                        const double speedup =
                            direct_timings[live_index].median_us /
                            context.median_us;
                        std::cout
                            << test_case.geometry.sources << ','
                            << nativeKVFormatName(format) << ','
                            << test_case.query_rows << ','
                            << live_kv_lengths[live_index] << ','
                            << generic_context.context_partition_slots << ','
                            << phase_blocks << ','
                            << logical_capacity_blocks << ','
                            << generic_context.reducer_dimension_wavefronts
                            << ',' << generic_context.reducer_block_slots
                            << ',' << std::fixed << std::setprecision(3)
                            << context.median_us << ','
                            << direct_timings[live_index].median_us << ','
                            << speedup << '\n';
                        ++measured_candidates;
                    }
                }
            }
        }
        EXPECT_GT(measured_candidates, 0);
#endif
    }

    /**
     * @brief Isolate phase-slot residency for the Qwen27 TP4 policy miss.
     *
     * Each phase workgroup already strides logical context partitions and keeps
     * its Q tile resident in LDS. Sweeping every logarithmic slot envelope plus
     * the generic point determines whether a smaller fixed graph can remove the
     * one-partition launch penalty without surrendering 8K/128K throughput.
     * The reducer geometry remains fixed, so this test changes one axis only.
     */
    TEST_F(
        ROCmFlashAttentionContextParallelPerf,
        PersistentContextSlotTournamentQwen27TP4AllNativeFormats)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "ROCm support is disabled";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device is available";

        const ParticipantAttentionGeometry geometry{
            "Qwen3.5/3.6-27B@TP4",
            6,
            1,
            256,
            0,
            6,
        };
        constexpr int query_rows = 128;
        constexpr int kv_capacity = 131072;
        constexpr std::array<int, 3> live_kv_lengths{{256, 8192, 131072}};
        const DeterministicSweepShard shard = sweepShardFromEnvironment();
        ASSERT_TRUE(shard.valid());

        std::cout
            << "\nROCm FA2 Qwen27 TP4 persistent phase-slot tournament\n"
            << "geometry,format,M,live_kv,slots,phase_blocks,"
               "reducer_wavefronts,reducer_blocks,context_us,direct_us,"
               "speedup\n";

        int measured_candidates = 0;
        std::size_t bundle_ordinal = 0;
        for (const NativeKVFormat format : kNativeKVFormats)
        {
            const std::size_t current_bundle = bundle_ordinal++;
            if (!shard.owns(current_bundle))
                continue;

            CapturedROCmFA2Benchmark benchmark(
                geometry,
                format,
                query_rows,
                kv_capacity,
                properties_.multiProcessorCount,
                std::max<std::size_t>(
                    properties_.sharedMemPerBlock,
                    rocm_policy::kROCmFA2LDSCapacityBytes));
            ASSERT_TRUE(benchmark.ready())
                << nativeKVFormatName(format) << ": " << benchmark.error();
            ASSERT_TRUE(benchmark.prepareQueryRows(query_rows))
                << benchmark.error();
            ASSERT_TRUE(benchmark.captureContextCandidate(0, 0, 0))
                << benchmark.error();
            const auto generic_context = benchmark.contextPlan();
            ASSERT_TRUE(generic_context.usesContextParallelism());

            const std::vector<int> slot_candidates = contextSlotCandidates(
                generic_context.max_context_partitions,
                generic_context.context_partition_slots);
            std::array<TransactionTiming, live_kv_lengths.size()>
                direct_timings{};
            for (std::size_t live_index = 0;
                 live_index < live_kv_lengths.size();
                 ++live_index)
            {
                direct_timings[live_index] = benchmark.measureDirect(
                    live_kv_lengths[live_index]);
                ASSERT_GT(direct_timings[live_index].median_us, 0.0)
                    << benchmark.error();
            }

            for (const int slots : slot_candidates)
            {
                ASSERT_TRUE(benchmark.captureContextCandidate(
                    slots,
                    generic_context.reducer_dimension_wavefronts,
                    generic_context.reducer_block_slots))
                    << benchmark.error();
                for (std::size_t live_index = 0;
                     live_index < live_kv_lengths.size();
                     ++live_index)
                {
                    const TransactionTiming context =
                        benchmark.measureContext(
                            live_kv_lengths[live_index]);
                    ASSERT_GT(context.median_us, 0.0) << benchmark.error();
                    const double speedup =
                        direct_timings[live_index].median_us /
                        context.median_us;
                    std::cout
                        << geometry.sources << ','
                        << nativeKVFormatName(format) << ',' << query_rows
                        << ',' << live_kv_lengths[live_index] << ','
                        << slots << ','
                        << benchmark.contextPlan().context_phase_block_slots
                        << ','
                        << generic_context.reducer_dimension_wavefronts << ','
                        << generic_context.reducer_block_slots << ','
                        << std::fixed << std::setprecision(3)
                        << context.median_us << ','
                        << direct_timings[live_index].median_us << ','
                        << speedup << '\n';
                    ++measured_candidates;
                }
            }
        }
        EXPECT_GT(measured_candidates, 0);
#endif
    }

    /**
     * @brief Measure captured FA2 at one, two and four participant head shapes.
     *
     * Reuse the catalog's TP placement, including replicated GQA beyond the
     * physical K/V head count. The 64-row query appends to 448 resident rows;
     * main chunks start at position zero. This diagnostic retains raw event
     * samples and uses installed geometry policy, never a forced candidate.
     */
    TEST_F(ROCmFlashAttentionContextParallelPerf, CapturedPrefillShardScaling)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "ROCm support is disabled";
#else
        if (!has_device_) GTEST_SKIP() << "No ROCm device is available";
        const auto model = std::find_if(kQwenAttentionGeometries.begin(), kQwenAttentionGeometries.end(),
            [](const ModelAttentionGeometry &value) {
                return value.n_heads == 16 && value.n_kv_heads == 2 && value.head_dim == 256;
            });
        ASSERT_NE(model, kQwenAttentionGeometries.end());
        std::cout << "case,degree,M,live_kv,heads,kv_heads,sample_index,latency_us\n";
        for (const int degree : {1, 2, 4})
        {
            const auto geometry = resolveParticipantGeometry(*model, degree);
            CapturedROCmFA2Benchmark benchmark(geometry, NativeKVFormat::FP16,
                512, 4096, properties_.multiProcessorCount,
                std::max<std::size_t>(properties_.sharedMemPerBlock, rocm_policy::kROCmFA2LDSCapacityBytes));
            ASSERT_TRUE(benchmark.ready()) << benchmark.error();
            for (const int rows : {64, 448, 512})
            {
                ASSERT_TRUE(benchmark.prepareQueryRows(rows)) << benchmark.error();
                const auto policy = benchmark.policyPlan();
                ASSERT_TRUE(policy.valid);
                if (policy.usesContextParallelism())
                    ASSERT_TRUE(benchmark.captureContextCandidate(0, 0)) << benchmark.error();
                const int live_kv = rows == 64 ? 512 : rows;
                for (int repeat = 0; repeat < 3; ++repeat)
                {
                    const auto timing = policy.usesContextParallelism()
                        ? benchmark.measureContext(live_kv) : benchmark.measureDirect(live_kv);
                    ASSERT_GT(timing.median_us, 0.0) << benchmark.error();
                    for (size_t index = 0; index < timing.samples_us.size(); ++index)
                        std::cout << "fa2_captured_shard," << degree << ',' << rows << ',' << live_kv
                                  << ',' << geometry.n_heads << ',' << geometry.n_kv_heads
                                  << ',' << repeat * 3 + index << ',' << std::setprecision(12)
                                  << timing.samples_us[index] << '\n';
                }
            }
        }
#endif
    }

    /**
     * @brief Record fixed-M512 default attention scaling from empty through 64K history.
     *
     * Dense and MoE use their measured participant mappings and FP16 KV in a
     * persistent 131072-row arena. The capture-time selector receives actual
     * native format and visibility. Native LDS/register/occupancy observations
     * authenticate the selected graph before timing; each CSV row retains the
     * three unprofiled event samples. History is existing KV, so total visible
     * capacity is history plus the fixed 512-row query. The explicit 8192/8193
     * total-KV points expose the short/long arithmetic boundary, and the final
     * empty-history replay proves reuse after the largest span.
     */
    TEST_F(ROCmFlashAttentionContextParallelPerf, DefaultFixedQueryPrefixScaling)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "ROCm support is disabled";
#else
        if (!has_device_) GTEST_SKIP() << "No ROCm device is available";
        constexpr int rows = 512;
        constexpr int capacity = 131072;
        const std::array<ParticipantAttentionGeometry, 2> geometries{{
            {"dense", 12, 2, 256, 0, 0}, {"moe", 8, 1, 256, 0, 0}}};
        // 7680/7681 history means exactly 8192/8193 total visible KV rows.
        constexpr std::array<int, 12> histories{{
            0, 512, 1024, 2048, 4096, 7680, 7681, 8192, 16384, 32768, 65536, 0}};
        std::cout << "case,format,M,history,live_kv,capacity,heads,kv_heads,policy_mode,"
                     "score_storage,tile_lds_bytes,native_lds_bytes,native_registers,"
                     "native_private_bytes,resident_blocks_per_cu,grid_x,grid_y,grid_z,"
                     "graph_nodes,replays_per_sample,sample_index,latency_us,kernel_symbol\n";
        for (const auto &geometry : geometries)
        {
            CapturedROCmFA2Benchmark benchmark(geometry, NativeKVFormat::FP16,
                rows, capacity, properties_.multiProcessorCount,
                std::max<std::size_t>(properties_.sharedMemPerBlock,
                                     rocm_policy::kROCmFA2LDSCapacityBytes));
            ASSERT_TRUE(benchmark.ready()) << benchmark.error();
            ASSERT_TRUE(benchmark.prepareQueryRows(rows)) << benchmark.error();
            const auto plan = benchmark.policyPlan();
            ASSERT_TRUE(plan.valid);
            // These exact mappings fill the measured retained phase grid.
            // A context result here is a production-dispatch regression.
            ASSERT_FALSE(plan.usesContextParallelism());
            ASSERT_EQ(plan.tile.score_storage,
                      rocm_policy::ROCmFA2ScoreStorage::RegisterOwnedXorTile);
            ASSERT_TRUE(benchmark.captureDefaultGraph()) << benchmark.error();
            ASSERT_TRUE(benchmark.captureContextCandidate(0, 0)) << benchmark.error();
            ASSERT_EQ(benchmark.contextPlan().device_direct_partition_limit, 1);
            hipKernelNodeParams native{};
            ASSERT_TRUE(benchmark.defaultKernelInventory(native)) << benchmark.error();
            const char *kernel_name = benchmark.defaultKernelName(native);
            ASSERT_NE(kernel_name, nullptr);
            ASSERT_EQ(native.sharedMemBytes, plan.tile.lds_bytes);
            ASSERT_EQ(native.sharedMemBytes, 16384U);
            hipFuncAttributes attributes{};
            ASSERT_EQ(hipFuncGetAttributes(&attributes, native.func), hipSuccess);
            ASSERT_EQ(attributes.localSizeBytes, 0U);
            ASSERT_LE(attributes.numRegs, 64);
            int resident = 0;
            ASSERT_EQ(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &resident, native.func, 256, native.sharedMemBytes), hipSuccess);
            ASSERT_GE(resident, 4);
            for (const int history : histories)
            {
                const auto timing = benchmark.measureDefault(history + rows);
                ASSERT_GT(timing.median_us, 0.0) << benchmark.error();
                ASSERT_EQ(timing.graph_nodes, 2U);
                ASSERT_TRUE(benchmark.validateDefaultOutput(history + rows)) << benchmark.error();
                for (std::size_t sample = 0; sample < timing.samples_us.size(); ++sample)
                    std::cout << geometry.sources << ",FP16," << rows << ',' << history
                              << ',' << history + rows << ',' << capacity << ','
                              << geometry.n_heads << ',' << geometry.n_kv_heads << ','
                              << rocm_policy::rocmFA2PrefillPhysicalModeName(plan.mode) << ','
                              << rocm_policy::rocmFA2ScoreStorageName(plan.tile.score_storage) << ','
                              << plan.tile.lds_bytes << ',' << native.sharedMemBytes << ','
                              << attributes.numRegs << ',' << attributes.localSizeBytes << ','
                              << resident << ',' << native.gridDim.x << ',' << native.gridDim.y
                              << ',' << native.gridDim.z << ',' << timing.graph_nodes << ','
                              << timing.replays_per_sample << ',' << sample << ','
                              << std::setprecision(12) << timing.samples_us[sample] << ',' << kernel_name << '\n';
            }
        }
#endif
    }

    /** @brief Separate numerical timing-owner replay from sampled economics. */
    enum class QKMeasurement { Scaling, TimingReplay };

    /**
     * @brief Pair exact Dot2/WMMA production graphs at fixed M512 through 64K KV.
     *
     * Each engine retains its own identical deterministic operands, device-count
     * producer and automatic query graph. Context graphs are untimed independent
     * correctness oracles, including their guarded direct leg and ordered reducer.
     * Alternating 31-sample event pairs report measured economics without assuming
     * an improvement threshold. Returning to 512 keys proves dirty reset reuse.
     * @param has_device_ Whether the fixture admitted a native runtime.
     * @param candidate Concrete matrix or adaptive query arithmetic to qualify.
     * @param mode Functional event-owner replay or complete fixed-query timing.
     */
    void runNativeGfx1100QKPrefixScaling(bool has_device_,
        llaminar2::rocm::FlashAttentionQKEngine candidate,
        QKMeasurement mode = QKMeasurement::Scaling)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "ROCm support is disabled";
#else
        if (!has_device_) GTEST_SKIP() << "No ROCm device is available";
        const auto &inventory = llaminar2::rocmHardwareMetadata();
        int ordinal = -1;
        for (size_t index = 0; index < inventory.size(); ++index)
        {
            const auto &isa = inventory.at(static_cast<int>(index)).isa_target;
            if (isa == "gfx1100" || isa.starts_with("gfx1100:"))
            {
                ordinal = static_cast<int>(index);
                break;
            }
        }
        if (ordinal < 0) GTEST_SKIP() << "Native gfx1100 is not visible";
        const auto &hardware = inventory.at(ordinal);
        ASSERT_EQ(hipSetDevice(ordinal), hipSuccess);
        constexpr int rows = 512, capacity = 65536, sample_count = 31;
        constexpr std::array<int, 10> native_lengths{{1024, 2048, 4096, 8192, 8193,
                                                   16384, 32768, 65536, 512, 1024}};
        constexpr std::array<int, 13> adaptive_lengths{{512,1023,1024,1025,2048,4096,
            8192,8193,16384,32768,65536,512,1024}};
        constexpr std::array<int,22> replay_lengths{{512,1023,1024,1025,512,
            512,1023,1024,1025,512,512,1023,1024,1025,512,512,1023,1024,1025,512,65536,512}};
        const std::span<const int> live_lengths = mode == QKMeasurement::TimingReplay
            ? std::span<const int>(replay_lengths) :
            candidate == llaminar2::rocm::FlashAttentionQKEngine::ContextAdaptive
                ? std::span<const int>(adaptive_lengths) : std::span<const int>(native_lengths);
        const std::array<ParticipantAttentionGeometry, 2> geometries{{
            {"dense", 12, 2, 256, 0, 0}, {"moe", 8, 1, 256, 0, 0}}};
        using Engine = llaminar2::rocm::FlashAttentionQKEngine;
        const std::array<Engine, 2> engines{{Engine::OrderedDot2, candidate}};
        const std::array<const char *, 2> engine_names{{"ordered_dot2",
            candidate == Engine::ContextAdaptive ? "context_adaptive" : "wmma_four_plane"}};
        std::cout << "FA2_QK_DEVICE," << ordinal << ',' << hardware.isa_target << ','
                  << hardware.pci_address << ',' << hardware.compute_units << '\n';
        std::cout << "record,geometry,engine,M,live_kv,capacity,sample,replays_per_sample,"
                     "latency_us,dot2_over_wmma,latency_over_1024,kv_over_1024,"
                     "latency_per_key_over_1024\n";
        for (const auto &geometry : geometries)
        {
            std::array<std::unique_ptr<CapturedROCmFA2Benchmark>, 2> benchmarks;
            std::array<const void *, 2> attention_functions{};
            for (size_t path = 0; path < engines.size(); ++path)
            {
                llaminar2::rocm::ScopedFlashAttentionQKEngine selected(engines[path]);
                benchmarks[path] = std::make_unique<CapturedROCmFA2Benchmark>(geometry, NativeKVFormat::FP16,
                    rows, capacity, hardware.compute_units, hardware.lds_per_block, ordinal);
                auto &benchmark = *benchmarks[path];
                ASSERT_TRUE(benchmark.ready()) << benchmark.error();
                ASSERT_TRUE(benchmark.prepareQueryRows(rows)) << benchmark.error();
                ASSERT_TRUE(benchmark.policyPlan().valid);
                ASSERT_FALSE(benchmark.policyPlan().usesContextParallelism());
                ASSERT_TRUE(benchmark.captureDefaultGraph()) << benchmark.error();
                ASSERT_TRUE(benchmark.prepareDefaultTiming()) << benchmark.error();
                {
                    // The adaptive candidate owns only the measured direct graph.
                    // Its independent context oracle retains explicit Dot2.
                    llaminar2::rocm::ScopedFlashAttentionQKEngine oracle(
                        candidate == Engine::ContextAdaptive ? Engine::OrderedDot2 : engines[path]);
                    ASSERT_TRUE(benchmark.captureContextCandidate(0, 0)) << benchmark.error();
                }
                ASSERT_EQ(benchmark.contextPlan().device_direct_partition_limit, 1);
                hipKernelNodeParams native{};
                ASSERT_TRUE(benchmark.defaultKernelInventory(native)) << benchmark.error();
                attention_functions[path] = native.func;
                const auto inspect = [&](const std::vector<hipKernelNodeParams> &nodes, const char *scope, Engine expected_engine)
                {
                    int attention_nodes = 0, reducer_nodes = 0;
                    for (size_t index = 0; index < nodes.size(); ++index)
                    {
                        const auto &node = nodes[index];
                        const char *name = benchmark.defaultKernelName(node);
                        ASSERT_NE(name, nullptr);
                        const std::string symbol(name);
                        const bool attention = symbol.find("flash_attention_2_mi50_kernel") != std::string::npos;
                        if (attention)
                        {
                            ++attention_nodes;
                            // The final template bool names the compiled QK engine;
                            // REGISTER_SCORE_TILE is a separate preceding bool.
                            const bool matrix = symbol.find("Lb1EE") != std::string::npos ||
                                symbol.find(", true>(") != std::string::npos;
                            const bool ordered = symbol.find("Lb0EE") != std::string::npos ||
                                symbol.find(", false>(") != std::string::npos;
                            const bool matrix_expected = expected_engine != Engine::OrderedDot2;
                            ASSERT_TRUE(matrix_expected ? matrix : ordered) << symbol;
                            const bool adaptive = symbol.find("FlashAttentionQKBranchPolicyE1ELb1EE") != std::string::npos ||
                                symbol.find("FlashAttentionQKBranchPolicy)1, true>(") != std::string::npos;
                            ASSERT_EQ(adaptive, expected_engine == Engine::ContextAdaptive) << symbol;
                        }
                        if (symbol.find("flash_attention_2_context_reduce_mi50_kernel") != std::string::npos)
                            ++reducer_nodes;
                        hipFuncAttributes attributes{};
                        ASSERT_EQ(hipFuncGetAttributes(&attributes, node.func), hipSuccess);
                        ASSERT_EQ(attributes.localSizeBytes, 0U) << symbol;
                        const unsigned threads = node.blockDim.x * node.blockDim.y * node.blockDim.z;
                        ASSERT_GT(threads, 0U);
                        ASSERT_LE(threads, static_cast<unsigned>(attributes.maxThreadsPerBlock));
                        int resident = 0;
                        ASSERT_EQ(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                            &resident, node.func, threads, node.sharedMemBytes), hipSuccess);
                        ASSERT_GT(resident, 0) << symbol;
                        std::cout << "FA2_QK_RESOURCE," << geometry.sources << ',' << engine_names[path]
                                  << ',' << scope << ',' << index << ',' << attributes.numRegs << ','
                                  << node.sharedMemBytes << ',' << attributes.localSizeBytes << ',' << resident
                                  << ',' << threads << ',' << node.gridDim.x << ',' << node.gridDim.y << ','
                                  << node.gridDim.z << ',' << symbol << '\n';
                    }
                    ASSERT_EQ(attention_nodes, nodes.size() == 1 ? 1 : 2);
                    ASSERT_EQ(reducer_nodes, nodes.size() == 1 ? 0 : 1);
                };
                ASSERT_NO_FATAL_FAILURE(inspect({native}, "default_attention", engines[path]));
                std::vector<hipKernelNodeParams> context_nodes;
                ASSERT_TRUE(benchmark.contextKernelInventory(context_nodes)) << benchmark.error();
                ASSERT_NO_FATAL_FAILURE(inspect(context_nodes, "untimed_context_oracle", candidate == Engine::ContextAdaptive ? Engine::OrderedDot2 : engines[path]));
            }
            ASSERT_NE(attention_functions[0], attention_functions[1]);
            std::array<double, 2> reference_1024{};
            std::array<std::vector<std::pair<double,double>>,2> scaling_points;
            int replay_epoch = 0;
            for (const int live : live_lengths)
            {
                SCOPED_TRACE(geometry.sources + " live_kv=" + std::to_string(live));
                std::array<std::vector<float>, 2> bytes;
                for (size_t path = 0; path < engines.size(); ++path)
                    ASSERT_TRUE(benchmarks[path]->validatedDefaultBytes(live, bytes[path]))
                        << engine_names[path] << ':' << benchmarks[path]->error();
                ASSERT_EQ(bytes[0].size(), bytes[1].size());
                ASSERT_EQ(std::memcmp(bytes[0].data(), bytes[1].data(), bytes[0].size() * sizeof(float)), 0);
                if (mode == QKMeasurement::TimingReplay)
                {
                    for (int order = 0; order < 2; ++order)
                    {
                        const int path = (replay_epoch + order) % 2;
                        auto &benchmark = *benchmarks[path];
                        ASSERT_TRUE(benchmark.poisonDefaultTimingState()) << benchmark.error();
                        ASSERT_GT(benchmark.measureDefaultBatch(live),0.0) << benchmark.error();
                        std::vector<float> observed;
                        ASSERT_TRUE(benchmark.observeDefaultTimingBytes(live,observed)) << benchmark.error();
                        ASSERT_EQ(observed.size(),bytes[path].size());
                        ASSERT_EQ(std::memcmp(observed.data(),bytes[path].data(),observed.size()*sizeof(float)),0);
                    }
                    ++replay_epoch;
                    continue;
                }
                for (auto &benchmark : benchmarks)
                    ASSERT_TRUE(benchmark->poisonDefaultTimingState()) << benchmark->error();
                // Warm both already captured graphs outside the paired sample set.
                for (int warmup = 0; warmup < 2; ++warmup)
                    for (auto &benchmark : benchmarks)
                        ASSERT_GT(benchmark->measureDefaultBatch(live), 0.0) << benchmark->error();
                std::array<std::array<double, sample_count>, 2> samples{};
                for (int sample = 0; sample < sample_count; ++sample)
                    for (int order = 0; order < 2; ++order)
                    {
                        const int path = (sample + order) % 2;
                        ASSERT_TRUE(benchmarks[path]->poisonDefaultTimingState()) << benchmarks[path]->error();
                        samples[path][sample] = benchmarks[path]->measureDefaultBatch(live);
                        ASSERT_TRUE(std::isfinite(samples[path][sample]) && samples[path][sample] > 0.0)
                            << benchmarks[path]->error();
                        // Poison and observation stay outside the internal events.
                        // Each measured batch must publish its own complete bytes;
                        // warmups or another sample cannot repair a missing writer.
                        std::vector<float> observed;
                        ASSERT_TRUE(benchmarks[path]->observeDefaultTimingBytes(live,observed))
                            << benchmarks[path]->error();
                        ASSERT_EQ(observed.size(),bytes[path].size());
                        ASSERT_EQ(std::memcmp(observed.data(),bytes[path].data(),observed.size()*sizeof(float)),0);
                    }
                std::array<double, 2> medians{};
                for (size_t path = 0; path < engines.size(); ++path)
                {
                    auto sorted = samples[path];
                    std::sort(sorted.begin(), sorted.end());
                    medians[path] = sorted[sample_count / 2];
                    if (live == 1024 && reference_1024[path] == 0.0) reference_1024[path] = medians[path];
                    if (live >= 1024 && (live & (live - 1)) == 0 &&
                        (scaling_points[path].empty() || live > scaling_points[path].back().first))
                        scaling_points[path].emplace_back(live,medians[path]);
                }
                const double speedup = medians[0] / medians[1];
                for (size_t path = 0; path < engines.size(); ++path)
                {
                    const double latency_scaling = reference_1024[path] > 0.0
                        ? medians[path] / reference_1024[path] : 0.0;
                    const double key_scaling = static_cast<double>(live) / 1024.0;
                    for (int sample = 0; sample < sample_count; ++sample)
                        std::cout << "FA2_QK_SAMPLE," << geometry.sources << ',' << engine_names[path]
                                  << ',' << rows << ',' << live << ',' << capacity << ',' << sample << ','
                                  << benchmarks[path]->defaultBatchSize(live) << ',' << std::setprecision(12)
                                  << samples[path][sample] << ',' << speedup << ',' << latency_scaling << ','
                                  << key_scaling << ',' << latency_scaling / key_scaling << '\n';
                    std::cout << "FA2_QK_MEDIAN," << geometry.sources << ',' << engine_names[path]
                              << ',' << rows << ',' << live << ',' << capacity << ",-1,"
                              << benchmarks[path]->defaultBatchSize(live) << ',' << std::setprecision(12)
                              << medians[path] << ',' << speedup << ',' << latency_scaling << ','
                              << key_scaling << ',' << latency_scaling / key_scaling << '\n';
                }
                // Observe the timed owner before any oracle can repair its bytes.
                for (size_t path = 0; path < engines.size(); ++path)
                {
                    std::vector<float> replay;
                    ASSERT_TRUE(benchmarks[path]->observeDefaultTimingBytes(live, replay)) << benchmarks[path]->error();
                    ASSERT_EQ(std::memcmp(bytes[path].data(), replay.data(), replay.size() * sizeof(float)), 0);
                    ASSERT_TRUE(benchmarks[path]->validatedDefaultBytes(live, replay)) << benchmarks[path]->error();
                    ASSERT_EQ(std::memcmp(bytes[path].data(), replay.data(), replay.size() * sizeof(float)), 0);
                }
            }
            if (mode == QKMeasurement::TimingReplay) continue;
            for (size_t path = 0; path < engines.size(); ++path)
            {
                const auto &points = scaling_points[path];
                ASSERT_EQ(points.size(),7u);
                double mean_k = 0.0, mean_us = 0.0;
                for (const auto &[k,us] : points) { mean_k += k; mean_us += us; }
                mean_k /= points.size(); mean_us /= points.size();
                double covariance = 0.0, variance_k = 0.0, variance_us = 0.0;
                for (const auto &[k,us] : points)
                {
                    covariance += (k-mean_k)*(us-mean_us);
                    variance_k += (k-mean_k)*(k-mean_k);
                    variance_us += (us-mean_us)*(us-mean_us);
                }
                ASSERT_GT(variance_k,0.0); ASSERT_GT(variance_us,0.0);
                const double slope = covariance/variance_k;
                const double intercept = mean_us-slope*mean_k;
                double residual = 0.0;
                for (const auto &[k,us] : points)
                    residual += (us-intercept-slope*k)*(us-intercept-slope*k);
                const double r_squared = 1.0-residual/variance_us;
                const double last_doubling = points.back().second/points[points.size()-2].second;
                std::cout << "FA2_QK_LINEARITY," << geometry.sources << ',' << engine_names[path]
                          << ',' << slope << ',' << intercept << ',' << r_squared << ',' << last_doubling << '\n';
                // Fixed-M causal work is affine in KV, not proportional through
                // the origin. The long-end bound admits 10% timing variation
                // above ideal doubling while rejecting quadratic growth; the
                // full-curve fit also guards an earlier capacity-dependent bend.
                EXPECT_GT(slope,0.0);
                EXPECT_GE(r_squared,0.995) << geometry.sources << ':' << engine_names[path];
                EXPECT_LE(last_doubling,2.2) << geometry.sources << ':' << engine_names[path];
            }
        }
#endif
    }

    /** @test Replay actual internal timing parents through threshold, poison and reset. */
    TEST_F(ROCmFlashAttentionContextParallelPerf, CapturedBatchEventBoundaryReplay)
    {
        runNativeGfx1100QKPrefixScaling(has_device_,
            llaminar2::rocm::FlashAttentionQKEngine::ContextAdaptive, QKMeasurement::TimingReplay);
    }

    /** @test Preserve the original unconditional matrix candidate scaling owner. */
    TEST_F(ROCmFlashAttentionContextParallelPerf, NativeGfx1100QKWmmaFixedQueryPrefixScaling)
    {
        runNativeGfx1100QKPrefixScaling(has_device_, llaminar2::rocm::FlashAttentionQKEngine::WmmaFourPlane);
    }

    /** @test Measure one retained adaptive query kernel across threshold/reset and 64K. */
    TEST_F(ROCmFlashAttentionContextParallelPerf, NativeGfx1100QKAdaptiveFixedQueryPrefixScaling)
    {
        runNativeGfx1100QKPrefixScaling(has_device_, llaminar2::rocm::FlashAttentionQKEngine::ContextAdaptive);
    }

    /** @brief Certify generic captured attention policy across the complete Qwen catalog. */
    TEST_F(
        ROCmFlashAttentionContextParallelPerf,
        QwenCatalogGeometryPolicyAllNativeFormats)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "ROCm support is disabled";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device is available";

        constexpr std::array<int, 5> query_rows{{16, 17, 32, 64, 128}};
        constexpr std::array<int, 3> live_kv_lengths{{256, 8192, 131072}};
        constexpr int kv_capacity = 131072;
        constexpr double maximum_policy_regret_percent = 5.0;
        const std::vector<ParticipantAttentionGeometry> participants =
            buildDistinctParticipantGeometries();
        ASSERT_FALSE(participants.empty());
        const DeterministicSweepShard shard = sweepShardFromEnvironment();
        ASSERT_TRUE(shard.valid());

        std::cout
            << "\nROCm FA2 all-Qwen geometry policy gate\n"
            << "sources,format,M,live_kv,capacity,local_heads,local_kv_heads,"
               "head_dim,gqa_rep,policy_mode,slots,phase_blocks,"
               "device_direct_partition_limit,reducer_wavefronts,"
               "reducer_blocks,winner_mode,policy_us,winner_us,regret_pct,"
               "direct_us,context_us\n";

        int measured_domains = 0;
        int query_transaction_plans = 0;
        int immediate_partition_plans = 0;
        int bounded_device_direct_plans = 0;
        std::size_t bundle_ordinal = 0;
        for (const ParticipantAttentionGeometry &geometry : participants)
        {
            for (const NativeKVFormat format : kNativeKVFormats)
            {
                const std::size_t current_bundle = bundle_ordinal++;
                if (!shard.owns(current_bundle))
                    continue;
                CapturedROCmFA2Benchmark benchmark(
                    geometry,
                    format,
                    query_rows.back(),
                    kv_capacity,
                    properties_.multiProcessorCount,
                    std::max<std::size_t>(
                        properties_.sharedMemPerBlock,
                        rocm_policy::kROCmFA2LDSCapacityBytes));
                ASSERT_TRUE(benchmark.ready())
                    << geometry.sources << ' ' << nativeKVFormatName(format)
                    << ": " << benchmark.error();

                for (const int m : query_rows)
                {
                    ASSERT_TRUE(benchmark.prepareQueryRows(m))
                        << benchmark.error();
                    const rocm_policy::ROCmFA2PrefillParallelPlan policy_plan =
                        benchmark.policyPlan();
                    ASSERT_TRUE(policy_plan.valid);
                    ASSERT_TRUE(benchmark.captureContextCandidate(0, 0))
                        << benchmark.error();
                    const rocm_policy::ROCmFA2PrefillParallelPlan context_plan =
                        benchmark.contextPlan();

                    if (!policy_plan.usesContextParallelism())
                    {
                        ++query_transaction_plans;
                    }
                    else if (policy_plan.device_direct_partition_limit > 1)
                    {
                        ++bounded_device_direct_plans;
                    }
                    else
                    {
                        ++immediate_partition_plans;
                    }

                    for (const int live_kv_len : live_kv_lengths)
                    {
                        if (live_kv_len < m)
                            continue;
                        const TransactionTiming direct =
                            benchmark.measureDirect(live_kv_len);
                        const TransactionTiming context =
                            benchmark.measureContext(live_kv_len);
                        ASSERT_GT(direct.median_us, 0.0) << benchmark.error();
                        ASSERT_GT(context.median_us, 0.0) << benchmark.error();
                        ASSERT_EQ(direct.graph_nodes, 1U);
                        ASSERT_EQ(context.graph_nodes, 3U);

                        const bool context_won =
                            context.median_us < direct.median_us;
                        const double winner_us = std::min(
                            direct.median_us,
                            context.median_us);
                        const double policy_us =
                            policy_plan.usesContextParallelism()
                                ? context.median_us
                                : direct.median_us;
                        const double regret_percent =
                            100.0 * (policy_us / winner_us - 1.0);

                        std::cout
                            << geometry.sources << ','
                            << nativeKVFormatName(format) << ','
                            << m << ',' << live_kv_len << ',' << kv_capacity
                            << ',' << geometry.n_heads << ','
                            << geometry.n_kv_heads << ',' << geometry.head_dim
                            << ',' << geometry.replicated_gqa_n_rep << ','
                            << rocm_policy::rocmFA2PrefillPhysicalModeName(
                                   policy_plan.mode)
                            << ',' << context_plan.context_partition_slots << ','
                            << context_plan.context_phase_block_slots << ','
                            << context_plan.device_direct_partition_limit << ','
                            << context_plan.reducer_dimension_wavefronts << ','
                            << context_plan.reducer_block_slots << ','
                            << (context_won ? "key_value_context"
                                            : "query_sequence")
                            << ',' << std::fixed << std::setprecision(3)
                            << policy_us << ',' << winner_us << ','
                            << regret_percent << ',' << direct.median_us << ','
                            << context.median_us << '\n';

                        EXPECT_LE(
                            regret_percent,
                            maximum_policy_regret_percent)
                            << geometry.sources << ' '
                            << nativeKVFormatName(format) << " M=" << m
                            << " live_kv=" << live_kv_len << " policy="
                            << rocm_policy::rocmFA2PrefillPhysicalModeName(
                                   policy_plan.mode);
                        ++measured_domains;
                    }
                }
            }
        }
        EXPECT_GT(measured_domains, 0);
        if (shard.count == 1)
        {
            EXPECT_GT(bounded_device_direct_plans, 0);
            EXPECT_GT(immediate_partition_plans, 0);
            EXPECT_EQ(
                query_transaction_plans + bounded_device_direct_plans +
                    immediate_partition_plans,
                static_cast<int>(participants.size() * kNativeKVFormats.size() *
                                 query_rows.size()));
        }
#endif
    }

    /**
     * @brief Prove policy economy remains total at a 256K captured K/V capacity.
     *
     * One underfilled TP participant from each compiled head dimension is enough
     * to expose the long-context occupancy tradeoff. All native formats, M=17/64,
     * and both half/full live capacities are measured through unchanged graphs.
     */
    TEST_F(
        ROCmFlashAttentionContextParallelPerf,
        LongContext256KAllNativeFormats)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "ROCm support is disabled";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device is available";

        const std::array<ParticipantAttentionGeometry, 3> geometries{{
            {"Qwen2.5-0.5B@TP2", 7, 1, 64, 0, 0},
            {"Qwen2.5-7B@TP4", 7, 1, 128, 0, 0},
            {"Qwen3.6-35B-A3B@TP8", 2, 2, 256, 0, 8},
        }};
        constexpr std::array<int, 2> query_rows{{17, 64}};
        constexpr std::array<int, 2> live_kv_lengths{{131072, 262144}};
        constexpr int kv_capacity = 262144;
        constexpr double maximum_policy_regret_percent = 5.0;
        const DeterministicSweepShard shard = sweepShardFromEnvironment();
        ASSERT_TRUE(shard.valid());

        std::size_t bundle_ordinal = 0;
        for (const ParticipantAttentionGeometry &geometry : geometries)
        {
            for (const NativeKVFormat format : kNativeKVFormats)
            {
                const std::size_t current_bundle = bundle_ordinal++;
                if (!shard.owns(current_bundle))
                    continue;
                CapturedROCmFA2Benchmark benchmark(
                    geometry,
                    format,
                    query_rows.back(),
                    kv_capacity,
                    properties_.multiProcessorCount,
                    std::max<std::size_t>(
                        properties_.sharedMemPerBlock,
                        rocm_policy::kROCmFA2LDSCapacityBytes));
                ASSERT_TRUE(benchmark.ready())
                    << geometry.sources << ' ' << nativeKVFormatName(format)
                    << ": " << benchmark.error();
                for (const int m : query_rows)
                {
                    ASSERT_TRUE(benchmark.prepareQueryRows(m))
                        << benchmark.error();
                    ASSERT_TRUE(benchmark.captureContextCandidate(0, 0))
                        << benchmark.error();
                    const auto policy_plan = benchmark.policyPlan();
                    ASSERT_TRUE(policy_plan.valid);
                    for (const int live_kv_len : live_kv_lengths)
                    {
                        const TransactionTiming direct =
                            benchmark.measureDirect(live_kv_len);
                        const TransactionTiming context =
                            benchmark.measureContext(live_kv_len);
                        ASSERT_GT(direct.median_us, 0.0) << benchmark.error();
                        ASSERT_GT(context.median_us, 0.0) << benchmark.error();
                        const double winner_us = std::min(
                            direct.median_us,
                            context.median_us);
                        const double policy_us =
                            policy_plan.usesContextParallelism()
                                ? context.median_us
                                : direct.median_us;
                        const double regret_percent =
                            100.0 * (policy_us / winner_us - 1.0);
                        EXPECT_LE(
                            regret_percent,
                            maximum_policy_regret_percent)
                            << geometry.sources << ' '
                            << nativeKVFormatName(format) << " M=" << m
                            << " live_kv=" << live_kv_len;
                    }
                }
            }
        }
#endif
    }

    /**
     * @brief Launch one exact graph transaction for rocprof/ISA correlation.
     *
     * This test is skipped unless `LLAMINAR_ROCM_FA2_PROFILE_M` is positive.
     * Optional selectors are `..._KV`, `..._CAPACITY`, `..._SLOTS`,
     * `..._PHASE_BLOCKS`, `..._REDUCER_WAVEFRONTS`, `..._REDUCER_BLOCKS`,
     * `..._FORMAT`, `..._MODE`, and `..._GEOMETRY`. Mode `default` resolves
     * the installed geometry policy; `direct` and `context` isolate a schedule.
     * Prefix those modes with `wmma_` or `ordered_` to record one explicit QK
     * engine on the actual visible gfx1100. No new runtime environment key is
     * required. For the fixed-M512 pair, dense uses `qwen27_tp2` and MoE uses
     * `qwen35_tp2`; the retained capacity can stay65536 as live KV grows.
     * Named geometries are
     * `qwen35_tp1`, `qwen35_tp2`, `qwen35_tp4`, `qwen35_tp8` (the default),
     * `qwen7_tp1`, and `qwen27_tp4`. The Qwen35 degrees reproduce the exact
     * shard-scaling probe geometries, including replicated GQA after TP2;
     * the other models isolate packed-direct and context-parallel regimes.
     */
    TEST_F(
        ROCmFlashAttentionContextParallelPerf,
        ExactCandidateProfilerLaunch)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "ROCm support is disabled";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device is available";
        const int m = positiveEnvironmentValue("LLAMINAR_ROCM_FA2_PROFILE_M");
        if (m == 0)
            GTEST_SKIP() << "Set LLAMINAR_ROCM_FA2_PROFILE_M to profile a point";
        ASSERT_EQ(::testing::UnitTest::GetInstance()->test_to_run_count(), 1)
            << "Exact attention profiling requires one selected GTest";
        const int live_kv = std::max(
            m,
            positiveEnvironmentValue("LLAMINAR_ROCM_FA2_PROFILE_KV"));
        const int capacity = std::max(
            live_kv,
            positiveEnvironmentValue("LLAMINAR_ROCM_FA2_PROFILE_CAPACITY"));
        const int slots =
            positiveEnvironmentValue("LLAMINAR_ROCM_FA2_PROFILE_SLOTS");
        const int phase_blocks = positiveEnvironmentValue(
            "LLAMINAR_ROCM_FA2_PROFILE_PHASE_BLOCKS");
        const int reducer_wavefronts = positiveEnvironmentValue(
            "LLAMINAR_ROCM_FA2_PROFILE_REDUCER_WAVEFRONTS");
        const int reducer_blocks = positiveEnvironmentValue(
            "LLAMINAR_ROCM_FA2_PROFILE_REDUCER_BLOCKS");
        const std::string requested_format =
            environmentString("LLAMINAR_ROCM_FA2_PROFILE_FORMAT");
        std::string requested_mode =
            environmentString("LLAMINAR_ROCM_FA2_PROFILE_MODE");
        const std::string requested_geometry =
            environmentString("LLAMINAR_ROCM_FA2_PROFILE_GEOMETRY");
        using QKEngine = llaminar2::rocm::FlashAttentionQKEngine;
        QKEngine qk_engine = QKEngine::Automatic;
        if (requested_mode.starts_with("wmma_"))
        {
            qk_engine = QKEngine::WmmaFourPlane;
            requested_mode.erase(0, 5);
        }
        else if (requested_mode.starts_with("adaptive_"))
        {
            qk_engine = QKEngine::ContextAdaptive;
            requested_mode.erase(0, 9);
        }
        else if (requested_mode.starts_with("ordered_"))
        {
            qk_engine = QKEngine::OrderedDot2;
            requested_mode.erase(0, 8);
        }
        int ordinal = 0, compute_units = properties_.multiProcessorCount;
        size_t lds_capacity = std::max<size_t>(properties_.sharedMemPerBlock,
                                             rocm_policy::kROCmFA2LDSCapacityBytes);
        if (qk_engine != QKEngine::Automatic)
        {
            const auto &inventory = llaminar2::rocmHardwareMetadata();
            ordinal = -1;
            for (size_t index = 0; index < inventory.size(); ++index)
            {
                const auto &candidate = inventory.at(static_cast<int>(index));
                if (candidate.isa_target == "gfx1100" || candidate.isa_target.starts_with("gfx1100:"))
                {
                    ordinal = candidate.ordinal;
                    compute_units = candidate.compute_units;
                    lds_capacity = candidate.lds_per_block;
                    break;
                }
            }
            if (ordinal < 0) GTEST_SKIP() << "Native gfx1100 is not visible";
        }

        NativeKVFormat format = NativeKVFormat::FP16;
        if (requested_format == "FP32")
            format = NativeKVFormat::FP32;
        else if (requested_format == "BF16")
            format = NativeKVFormat::BF16;
        else if (requested_format == "Q8_1")
            format = NativeKVFormat::Q8_1;
        else
            ASSERT_TRUE(requested_format.empty() || requested_format == "FP16")
                << "Unknown LLAMINAR_ROCM_FA2_PROFILE_FORMAT="
                << requested_format;

        ASSERT_TRUE(
            requested_mode.empty() || requested_mode == "direct" ||
            requested_mode == "context" || requested_mode == "default")
            << "Unknown LLAMINAR_ROCM_FA2_PROFILE_MODE=" << requested_mode;
        ParticipantAttentionGeometry geometry{};
        if (requested_geometry.empty() ||
            requested_geometry == "qwen35_tp1" ||
            requested_geometry == "qwen35_tp2" ||
            requested_geometry == "qwen35_tp4" ||
            requested_geometry == "qwen35_tp8")
        {
            const int degree = requested_geometry == "qwen35_tp1" ? 1
                               : requested_geometry == "qwen35_tp2" ? 2
                               : requested_geometry == "qwen35_tp4" ? 4
                                                                      : 8;
            // Share the scaling fixture's sharding calculation. In particular,
            // replicated GQA retains the full model's ratio, not the TP degree.
            geometry = resolveParticipantGeometry(
                {"Qwen3.6-35B-A3B", 16, 2, 256}, degree);
        }
        else if (requested_geometry == "qwen7_tp1")
        {
            geometry = {
                "Qwen2.5-7B@TP1", 28, 4, 128, 0, 0};
        }
        else if (requested_geometry == "qwen27_tp2" || requested_geometry == "qwen27_tp4")
        {
            const int degree = requested_geometry == "qwen27_tp2" ? 2 : 4;
            geometry = {degree == 2 ? "Qwen3.5/3.6-27B@TP2" : "Qwen3.5/3.6-27B@TP4",
                        24 / degree, 4 / degree, 256, 0, 0};
        }
        else
        {
            FAIL() << "Unknown LLAMINAR_ROCM_FA2_PROFILE_GEOMETRY="
                   << requested_geometry;
        }
        llaminar2::rocm::ScopedFlashAttentionQKEngine selected_qk(qk_engine);
        CapturedROCmFA2Benchmark benchmark(
            geometry,
            format,
            m,
            capacity,
            compute_units,
            lds_capacity,
            ordinal);
        ASSERT_TRUE(benchmark.ready()) << benchmark.error();
        ASSERT_TRUE(benchmark.prepareQueryRows(m)) << benchmark.error();
        const auto authenticate_profile = [&](const std::vector<hipKernelNodeParams> &nodes)
        {
            int matrix_nodes = 0;
            for (const auto &node : nodes)
            {
                const char *name = benchmark.defaultKernelName(node);
                ASSERT_NE(name, nullptr);
                const std::string symbol(name);
                if (symbol.find("flash_attention_2_mi50_kernel") != std::string::npos)
                {
                    const bool wmma = symbol.find("Lb1EE") != std::string::npos ||
                        symbol.find(", true>(") != std::string::npos;
                    if (qk_engine != QKEngine::Automatic)
                        ASSERT_EQ(wmma, qk_engine == QKEngine::WmmaFourPlane || qk_engine == QKEngine::ContextAdaptive) << symbol;
                    const bool adaptive = symbol.find("FlashAttentionQKBranchPolicyE1ELb1EE") != std::string::npos ||
                        symbol.find("FlashAttentionQKBranchPolicy)1, true>(") != std::string::npos;
                    ASSERT_EQ(adaptive, qk_engine == QKEngine::ContextAdaptive) << symbol;
                    matrix_nodes += wmma;
                }
                hipFuncAttributes attributes{};
                ASSERT_EQ(hipFuncGetAttributes(&attributes, node.func), hipSuccess);
                ASSERT_EQ(attributes.localSizeBytes, 0U) << symbol;
                std::cout << "FA2_QK_PROFILE_RESOURCE," << ordinal << ',' << requested_mode << ','
                          << m << ',' << live_kv << ',' << attributes.numRegs << ',' << node.sharedMemBytes
                          << ',' << attributes.localSizeBytes << ',' << node.blockDim.x << ','
                          << node.gridDim.x << ',' << node.gridDim.y << ',' << node.gridDim.z << ',' << symbol << '\n';
            }
            if (qk_engine == QKEngine::WmmaFourPlane || qk_engine == QKEngine::ContextAdaptive) ASSERT_GT(matrix_nodes, 0);
        };
        if (requested_mode == "default")
        {
            ASSERT_TRUE(benchmark.captureDefaultGraph()) << benchmark.error();
            hipKernelNodeParams node{};
            ASSERT_TRUE(benchmark.defaultKernelInventory(node)) << benchmark.error();
            ASSERT_NO_FATAL_FAILURE(authenticate_profile({node}));
            ASSERT_TRUE(benchmark.replayDefaultOnceForProfiler(live_kv)) << benchmark.error();
            return;
        }
        const bool context = requested_mode != "direct";
        if (context)
        {
            ASSERT_TRUE(benchmark.captureContextCandidate(
                slots,
                reducer_wavefronts,
                reducer_blocks,
                phase_blocks))
                << benchmark.error();
        }
        std::vector<hipKernelNodeParams> nodes;
        if (context)
            ASSERT_TRUE(benchmark.contextKernelInventory(nodes)) << benchmark.error();
        else
        {
            hipKernelNodeParams node{};
            ASSERT_TRUE(benchmark.directKernelInventory(node)) << benchmark.error();
            nodes.push_back(node);
        }
        ASSERT_NO_FATAL_FAILURE(authenticate_profile(nodes));
        ASSERT_TRUE(benchmark.replayOnceForProfiler(context, live_kv))
            << benchmark.error();
#endif
    }
} // namespace
