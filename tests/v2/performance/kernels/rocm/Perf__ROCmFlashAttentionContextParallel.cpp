/**
 * @file Perf__ROCmFlashAttentionContextParallel.cpp
 * @brief Captured ROCm FA2 query/context policy and launch-geometry tournament.
 *
 * This benchmark measures the two byte-equivalent ROCm FlashAttention prefill
 * schedules through the same native wrappers used by production. Query-sequence
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
#endif

#include "kernels/attention/AttentionDeviceParams.h"
#include "kernels/rocm/attention/ROCmFlashAttentionLaunchPolicy.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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
            std::size_t lds_capacity_bytes)
            : geometry_(std::move(geometry)),
              format_(format),
              maximum_query_rows_(maximum_query_rows),
              kv_capacity_(kv_capacity),
              compute_unit_count_(compute_unit_count),
              lds_capacity_bytes_(lds_capacity_bytes)
        {
            initialize();
        }

        ~CapturedROCmFA2Benchmark()
        {
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
            if (!publishLiveLength(live_kv_len) || !captured.exec)
                return false;
            if (!check(hipGraphLaunch(captured.exec, stream_), "hipGraphLaunch") ||
                !check(hipEventRecord(stop_, stream_), "hipEventRecord(stop)") ||
                !check(hipEventSynchronize(stop_), "hipEventSynchronize(stop)"))
            {
                return false;
            }
            return true;
        }

    private:
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
            };
        }

        /** Create stream/events and all maximum-capacity device buffers. */
        void initialize()
        {
            if (geometry_.n_heads <= 0 || geometry_.n_kv_heads <= 0 ||
                geometry_.head_dim <= 0 || maximum_query_rows_ <= 0 ||
                kv_capacity_ <= 0 || compute_unit_count_ <= 0 ||
                lds_capacity_bytes_ == 0)
            {
                setError("invalid persistent benchmark geometry");
                return;
            }
            if (!check(
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

            if (!check(hipMemsetAsync(d_q_, 0, q_bytes, stream_), "memset Q") ||
                !check(hipMemsetAsync(d_k_, 0, kv_bytes, stream_), "memset K") ||
                !check(hipMemsetAsync(d_v_, 0, kv_bytes, stream_), "memset V") ||
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
                if (!check(hipEventRecord(start_, stream_), "timing start event"))
                    return result;
                for (int replay = 0; replay < replay_count; ++replay)
                {
                    if (!check(
                            hipGraphLaunch(captured.exec, stream_),
                            "timed hipGraphLaunch"))
                    {
                        return result;
                    }
                }
                if (!check(hipEventRecord(stop_, stream_), "timing stop event") ||
                    !check(hipEventSynchronize(stop_), "timing completion"))
                {
                    return result;
                }
                float elapsed_ms = 0.0F;
                if (!check(
                        hipEventElapsedTime(&elapsed_ms, start_, stop_),
                        "hipEventElapsedTime"))
                {
                    return result;
                }
                samples[sample] =
                    static_cast<double>(elapsed_ms) * 1000.0 /
                    static_cast<double>(replay_count);
            }
            std::sort(samples.begin(), samples.end());
            result.median_us = samples[sample_count / 2];
            result.replays_per_sample = replay_count;
            return result;
        }

        ParticipantAttentionGeometry geometry_;
        NativeKVFormat format_ = NativeKVFormat::FP16;
        int maximum_query_rows_ = 0;
        int kv_capacity_ = 0;
        int compute_unit_count_ = 0;
        std::size_t lds_capacity_bytes_ = 0;
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
        void SetUp() override
        {
#ifdef HAVE_ROCM
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
     * @brief Gate generic mode selection across all Qwen/TP/native-format domains.
     *
     * Exact model aliases are deduplicated by the geometry visible to dispatch.
     * M=16 exercises the grouped-verifier range and larger buckets exercise
     * prefill geometry. Fixed 128K capture capacity is replayed at short,
     * medium, and full live context without recapture.
     */
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
     * `..._FORMAT`, `..._MODE`, and `..._GEOMETRY`. Named geometries are
     * `qwen35_tp8` (the default), `qwen7_tp1`, and `qwen27_tp4`; they isolate
     * the replicated-GQA, packed device-direct, and context-parallel regimes.
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
        const std::string requested_mode =
            environmentString("LLAMINAR_ROCM_FA2_PROFILE_MODE");
        const std::string requested_geometry =
            environmentString("LLAMINAR_ROCM_FA2_PROFILE_GEOMETRY");

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

        const bool context = requested_mode != "direct";
        ASSERT_TRUE(
            requested_mode.empty() || requested_mode == "direct" ||
            requested_mode == "context")
            << "Unknown LLAMINAR_ROCM_FA2_PROFILE_MODE=" << requested_mode;
        ParticipantAttentionGeometry geometry{};
        if (requested_geometry.empty() ||
            requested_geometry == "qwen35_tp8")
        {
            geometry = {
                "Qwen3.6-35B-A3B@TP8", 2, 2, 256, 0, 8};
        }
        else if (requested_geometry == "qwen7_tp1")
        {
            geometry = {
                "Qwen2.5-7B@TP1", 28, 4, 128, 0, 0};
        }
        else if (requested_geometry == "qwen27_tp4")
        {
            geometry = {
                "Qwen3.5/3.6-27B@TP4", 6, 1, 256, 0, 0};
        }
        else
        {
            FAIL() << "Unknown LLAMINAR_ROCM_FA2_PROFILE_GEOMETRY="
                   << requested_geometry;
        }
        CapturedROCmFA2Benchmark benchmark(
            geometry,
            format,
            m,
            capacity,
            properties_.multiProcessorCount,
            std::max<std::size_t>(
                properties_.sharedMemPerBlock,
                rocm_policy::kROCmFA2LDSCapacityBytes));
        ASSERT_TRUE(benchmark.ready()) << benchmark.error();
        ASSERT_TRUE(benchmark.prepareQueryRows(m)) << benchmark.error();
        if (context)
        {
            ASSERT_TRUE(benchmark.captureContextCandidate(
                slots,
                reducer_wavefronts,
                reducer_blocks,
                phase_blocks))
                << benchmark.error();
        }
        ASSERT_TRUE(benchmark.replayOnceForProfiler(context, live_kv))
            << benchmark.error();
#endif
    }
} // namespace
