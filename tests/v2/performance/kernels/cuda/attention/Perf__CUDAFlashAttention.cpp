/**
 * @file Perf__CUDAFlashAttention.cpp
 * @brief Performance benchmarks for CUDA Flash Attention kernels
 *
 * Measures throughput (TFLOPS, tokens/sec) for:
 *   - FA2 pipelined prefill (Ampere SM 8.0+, warp specialization, WMMA)
 *   - Flash Decoding (split-K parallelism)
 *   - The production Qwen3.6-35B-A3B FP16-KV attention geometry replayed
 *     through a one-node CUDA graph across every ordinary prefill bucket and
 *     resident contexts through 128K tokens.
 *
 * **Tested Configurations**:
 * - Physical query buckets: 64 through 4096 rows
 * - Resident KV lengths: matching prompt spans through 131072 tokens
 * - Head dimensions: 64, 128, and Qwen3.6's 256
 * - GQA ratios: 1:1 (MHA), 4:1 (GQA), 8:1 (GQA Qwen)
 *
 * **Performance Metrics**:
 * - TFLOPS: FLOPs / time (attention complexity: 4 * seq^2 * head_dim * n_heads)
 * - Tokens/sec: seq_len / time_per_attention
 * - Memory bandwidth: data moved / time
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA

// Include project headers BEFORE CUDATestUtils.h
#include "tensors/Tensors.h"
#include "execution/config/RuntimeConfig.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "utils/MPIContext.h"
#include "backends/cuda/CUDABackend.h"
#include "backends/cuda/CUDAGraphCapture.h"
#include "kernels/cuda/attention/CUDAFlashAttentionLaunchPolicy.h"
#include "kernels/cuda/attention/CUDAFlashAttentionKernelT.h"
#include "utils/DebugEnv.h"
#include <cuda_runtime.h>

// Include test utils
#include "../../../../utils/CUDATestUtils.h"

#include <array>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <limits>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <memory>
#include <sstream>
#include <string>

using namespace llaminar2;
using namespace llaminar2::cuda;
using namespace llaminar2::test::cuda;

extern "C"
{
    /**
     * @brief Launch the production CUDA FA2 prefill specialization over FP16 KV.
     *
     * The benchmark calls the same C ABI used by
     * CUDAFlashAttentionKernelT's tensor path.  All allocation, initialization,
     * and graph construction occur before the timed replay interval.
     */
    int cudaFlashAttn_prefill_fa2_fp16kv(
        const float *Q,
        const void *K_fp16,
        const void *V_fp16,
        float *O,
        int batch_size,
        int seq_len,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        bool causal,
        int window_size,
        int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep);

    /**
     * @brief Launch the fixed-partition arithmetic reference as ordered nodes.
     */
    int cudaFlashAttn_prefill_fa2_fp16kv_partitioned_sequence(
        const float *Q,
        const void *K_fp16,
        const void *V_fp16,
        float *O,
        float *O_partial,
        float *m_partial,
        float *l_partial,
        int batch_size,
        int seq_len,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        bool causal,
        int window_size,
        int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        int context_partition_size,
        int max_context_partitions,
        int context_partition_slots,
        int device_direct_partition_limit,
        int reducer_dimension_warps,
        void *capture_conditional,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep);

    /**
     * @brief Launch the same fixed-partition math as one context-parallel grid.
     */
    int cudaFlashAttn_prefill_fa2_fp16kv_context_parallel(
        const float *Q,
        const void *K_fp16,
        const void *V_fp16,
        float *O,
        float *O_partial,
        float *m_partial,
        float *l_partial,
        int batch_size,
        int seq_len,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        bool causal,
        int window_size,
        int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        int context_partition_size,
        int max_context_partitions,
        int context_partition_slots,
        int device_direct_partition_limit,
        int reducer_dimension_warps,
        void *capture_conditional,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep);

    /** @brief Publish explicit geometry and an optional native branch condition. */
    int cudaFlashAttn_prepare_device_params_from_geometry(
        void *device_params,
        int kv_len,
        int kv_stride,
        int position_offset,
        int query_rows,
        unsigned long long prefill_branch_condition,
        int direct_kv_limit,
        void *stream);
}

namespace
{

    // ============================================================================
    // Performance Test Configuration
    // ============================================================================

    constexpr int WARMUP_ITERATIONS = 10;
    constexpr int BENCHMARK_ITERATIONS = 100;

    constexpr int QWEN36_MAX_PREFILL_ROWS = 4096;
    constexpr int QWEN36_MAX_CONTEXT = 131072;

    /** Maximum relative latency regret accepted for a physical FA2 policy. */
    constexpr double kFA2PolicyRelativeRegret = 0.05;

    /**
     * Fixed launch budget for CUDA's replay-local native conditional decision.
     *
     * A false IF-only body still costs roughly one graph-scheduler decision. On
     * very small attention grids that fixed cost can exceed five percent even
     * though the useful root kernel has the winning physical geometry. The
     * adaptive gate therefore allows the larger of five percent or this strict
     * absolute dispatch budget. It does not excuse context arithmetic, kernel
     * launch, or reducer regressions; all of those remain on the measured graph
     * critical path and must fit the same end-to-end ceiling.
     */
    constexpr double kFA2NativeConditionalDispatchBudgetUs = 3.5;

    /**
     * @brief Return the end-to-end latency ceiling for a native adaptive graph.
     * @param winner_us Fastest measured fixed physical transaction in microseconds.
     * @return Maximum acceptable complete adaptive-graph latency in microseconds.
     */
    [[nodiscard]] constexpr double fa2AdaptiveEconomyCeilingUs(
        double winner_us)
    {
        return winner_us + std::max(
                               winner_us * kFA2PolicyRelativeRegret,
                               kFA2NativeConditionalDispatchBudgetUs);
    }

    /**
     * @brief Physical attention geometry shared by one or more model releases.
     *
     * Kernel dispatch intentionally consumes this geometry rather than a model
     * name.  The release label exists only to make benchmark output actionable;
     * it never participates in the production launch decision.
     */
    struct AttentionGeometry
    {
        const char *release_label;
        int n_heads;
        int n_kv_heads;
        int head_dim;
        int head_start = 0;
        int replicated_gqa_n_rep = 0;

        /**
         * @brief Return zero for local KV indexing or the global replicated ratio.
         */
        [[nodiscard]] int launchGqaRep() const
        {
            return replicated_gqa_n_rep;
        }
    };

    inline constexpr AttentionGeometry kQwen36MoE35BAttention{
        .release_label = "Qwen3.6-35B-A3B",
        .n_heads = 16,
        .n_kv_heads = 2,
        .head_dim = 256,
    };

    /**
     * @brief Distinct released Qwen attention geometries used by policy gates.
     *
     * Releases sharing one physical head geometry deliberately share one row:
     * model names never participate in CUDA launch dispatch.
     */
    inline constexpr std::array<AttentionGeometry, 10>
        kQwenAttentionGeometries{{
            {"Qwen2.5-0.5B", 14, 2, 64},
            {"Qwen2.5-1.5B", 12, 2, 128},
            {"Qwen2.5-3B", 16, 2, 128},
            {"Qwen2.5-7B", 28, 4, 128},
            {"Qwen2.5-14B", 40, 8, 128},
            {"Qwen3.5-0.8B/2B", 8, 2, 256},
            {"Qwen3.5-4B/9B", 16, 4, 256},
            {"Qwen3.5/3.6-27B", 24, 4, 256},
            {"Qwen3.5/3.6-35B-A3B", 16, 2, 256},
            {"Qwen3.5-122B-A10B/397B-A17B", 32, 2, 256},
        }};

    /** Participant-local ownership of the model's K/V head tensor. */
    enum class KVHeadPlacement
    {
        Sharded,   ///< This participant addresses only its local K/V heads.
        Replicated ///< This participant addresses all model K/V heads.
    };

    /**
     * @brief Resolved participant attention geometry for one equal TP split.
     */
    struct TensorParallelAttentionGeometry
    {
        AttentionGeometry local{}; ///< Exact geometry passed to the launcher.
        KVHeadPlacement kv_placement = KVHeadPlacement::Sharded;
        bool valid = false; ///< False when Q heads cannot split evenly.
    };

    /**
     * @brief Resolve local Q/KV geometry using production's sharding policy.
     *
     * Query heads are split equally. K/V heads are sharded when divisible by
     * the TP degree; compact GQA tensors are otherwise replicated and mapped by
     * global query-head position. Keeping this calculation in one typed helper
     * prevents policy tournaments from silently benchmarking a head mapping
     * that the graph builder could never produce.
     *
     * @param model Full-model attention geometry.
     * @param tp_degree Number of equal tensor-parallel participants.
     * @return Concrete local geometry, or an invalid result for uneven Q heads.
     */
    [[nodiscard]] constexpr TensorParallelAttentionGeometry
    resolveTensorParallelAttentionGeometry(
        const AttentionGeometry &model,
        int tp_degree)
    {
        if (tp_degree <= 0 || model.n_heads <= 0 ||
            model.n_kv_heads <= 0 || model.n_heads % tp_degree != 0)
        {
            return {};
        }

        AttentionGeometry local = model;
        local.n_heads = model.n_heads / tp_degree;
        if (model.n_kv_heads % tp_degree == 0)
        {
            local.n_kv_heads = model.n_kv_heads / tp_degree;
            local.replicated_gqa_n_rep = 0;
            return {
                .local = local,
                .kv_placement = KVHeadPlacement::Sharded,
                .valid = true,
            };
        }

        local.n_kv_heads = model.n_kv_heads;
        local.replicated_gqa_n_rep =
            model.n_heads / model.n_kv_heads;
        return {
            .local = local,
            .kv_placement = KVHeadPlacement::Replicated,
            .valid = true,
        };
    }

    /** @brief Stable diagnostic name for a typed K/V placement decision. */
    [[nodiscard]] constexpr const char *kvHeadPlacementName(
        KVHeadPlacement placement)
    {
        return placement == KVHeadPlacement::Sharded
                   ? "sharded"
                   : "replicated";
    }

    /**
     * @brief Test configuration
     */
    struct BenchConfig
    {
        int seq_len;
        int kv_len; // For decode: historical context length
        int n_heads;
        int n_kv_heads;
        int head_dim;
        const char *description;

        int gqaRatio() const { return n_heads / n_kv_heads; }

        // Attention FLOPs: Q @ K^T = seq * kv_len * head_dim, Softmax = seq * kv_len,
        // Att @ V = seq * kv_len * head_dim. Total per head: ~4 * seq * kv_len * head_dim
        double computeTFLOPs() const
        {
            double flops_per_head = 4.0 * seq_len * kv_len * head_dim;
            return (flops_per_head * n_heads) / 1e12;
        }

        // Memory: Q, K, V reads + O write (simplified)
        double computeMemoryGB() const
        {
            size_t q_bytes = seq_len * n_heads * head_dim * sizeof(float);
            size_t k_bytes = kv_len * n_kv_heads * head_dim * sizeof(float);
            size_t v_bytes = kv_len * n_kv_heads * head_dim * sizeof(float);
            size_t o_bytes = seq_len * n_heads * head_dim * sizeof(float);
            return (q_bytes + k_bytes + v_bytes + o_bytes) / (1024.0 * 1024.0 * 1024.0);
        }
    };

    /**
     * @brief One production-shaped prefill attention measurement point.
     *
     * `query_rows` is the immutable captured graph bucket. `kv_len` is the
     * resident cache span visible to that chunk. Long prompts therefore grow
     * `kv_len` through 128K while retaining a query bucket no larger than the
     * production 4096-row ceiling.
     */
    struct ProductionPrefillPoint
    {
        int query_rows;
        int kv_len;
        const char *label;
    };

    /**
     * @brief Timing and launch geometry observed for one captured replay.
     */
    struct ProductionPrefillMeasurement
    {
        double latency_us = 0.0;
        double query_tokens_per_second = 0.0;
        double effective_tflops = 0.0;
        int grid_blocks = 0;
        int block_threads = 0;
        int tile_kv = 0;
        double waves_per_gpu = 0.0;
    };

    /**
     * @brief End-to-end captured timings for the context-partition hypothesis.
     */
    struct ContextPartitionMeasurement
    {
        double direct_latency_us = 0.0;    ///< Current one-node production FA2.
        double sequence_latency_us = 0.0;  ///< Ordered phase-one graph nodes plus merge.
        double context_latency_us = 0.0;   ///< Parallel phase-one grid plus merge.
        int live_context_partitions = 0;   ///< Partitions containing live K/V rows.
        int captured_context_partitions = 0; ///< Capacity-sized graph envelope.
        int context_partition_slots = 0; ///< Persistent phase-one slots per head/tile.
        int device_direct_partition_limit = 0; ///< Device-selected direct prefix.
        int reducer_dimension_warps = 0;   ///< Independent output stripes per row.
        std::size_t direct_graph_nodes = 0;
        std::size_t sequence_graph_nodes = 0;
        std::size_t context_graph_nodes = 0;
    };

    /**
     * @brief Parse a positive profiler selector, returning zero when absent.
     * @param name Environment variable name.
     */
    int positiveEnvironmentValue(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return 0;
        const int value = std::atoi(raw);
        return value > 0 ? value : 0;
    }

    /**
     * @brief Restore the process query-partition policy after a tournament.
     *
     * Production reads the typed setting once while constructing each graph.
     * The performance tournament changes that same capture-time input between
     * graph constructions so every real compiled specialization is measured
     * through the production launcher.  No test-only kernel entry point or
     * runtime branch is introduced.
     */
    class ScopedFA2QueryWarpGroupPolicy final
    {
    public:
        ScopedFA2QueryWarpGroupPolicy()
            : original_(mutableDebugEnv().attention.cuda_fa2_q_warp_groups)
        {
        }

        ~ScopedFA2QueryWarpGroupPolicy()
        {
            mutableDebugEnv().attention.cuda_fa2_q_warp_groups = original_;
        }

        ScopedFA2QueryWarpGroupPolicy(
            const ScopedFA2QueryWarpGroupPolicy &) = delete;
        ScopedFA2QueryWarpGroupPolicy &operator=(
            const ScopedFA2QueryWarpGroupPolicy &) = delete;

        /**
         * @brief Select policy mode or one exact compiled query grouping.
         * @param query_warp_groups Zero for generic policy, otherwise candidate.
         */
        void select(int query_warp_groups)
        {
            mutableDebugEnv().attention.cuda_fa2_q_warp_groups =
                query_warp_groups;
        }

    private:
        int original_ = 0;
    };

    /**
     * @brief Restore the process K/V tile policy after a physical tournament.
     *
     * The override feeds the same capture-time production selector used by an
     * ordinary graph. It never changes a live graph and therefore cannot add a
     * runtime branch, recapture request, or host decision to inference.
     */
    class ScopedFA2KVTilePolicy final
    {
    public:
        ScopedFA2KVTilePolicy()
            : original_(mutableDebugEnv().attention.cuda_fa2_tile_kv)
        {
        }

        ~ScopedFA2KVTilePolicy()
        {
            mutableDebugEnv().attention.cuda_fa2_tile_kv = original_;
        }

        ScopedFA2KVTilePolicy(const ScopedFA2KVTilePolicy &) = delete;
        ScopedFA2KVTilePolicy &operator=(
            const ScopedFA2KVTilePolicy &) = delete;

        /**
         * @brief Select generic policy or one exact compiled physical tile.
         * @param tile_kv Zero for policy, otherwise 16, 32, or 64.
         */
        void select(int tile_kv)
        {
            mutableDebugEnv().attention.cuda_fa2_tile_kv = tile_kv;
        }

    private:
        int original_ = 0;
    };

    /**
     * @brief Own one persistent Qwen3.6 FP16-KV attention arena and replay graph.
     *
     * The maximum Q/O and K/V allocations are made once. Each measurement
     * records one production FA2 launch into a CUDA graph, warms that graph,
     * then times batches of graph replays with one terminal event wait. There
     * are no allocations, transfers, host callbacks, or synchronization calls
     * between individual timed replays.
     */
    class CapturedProductionPrefill final
    {
    public:
        /**
         * @brief Build one persistent arena for a physical attention geometry.
         * @param geometry Query/KV head geometry used by every captured launch.
         * @param maximum_query_rows Largest query bucket admitted by this arena.
         * @param maximum_context Largest resident KV span admitted by this arena.
         */
        CapturedProductionPrefill(
            AttentionGeometry geometry,
            int maximum_query_rows,
            int maximum_context)
            : geometry_(geometry),
              maximum_query_rows_(maximum_query_rows),
              maximum_context_(maximum_context)
        {
            initialize();
        }

        ~CapturedProductionPrefill()
        {
            destroyGraph();
            if (start_)
                (void)cudaEventDestroy(start_);
            if (stop_)
                (void)cudaEventDestroy(stop_);
            if (d_q_)
                (void)cudaFree(d_q_);
            if (d_k_)
                (void)cudaFree(d_k_);
            if (d_v_)
                (void)cudaFree(d_v_);
            if (d_output_)
                (void)cudaFree(d_output_);
            if (stream_)
                (void)cudaStreamDestroy(stream_);
        }

        CapturedProductionPrefill(const CapturedProductionPrefill &) = delete;
        CapturedProductionPrefill &operator=(const CapturedProductionPrefill &) = delete;

        /** @brief Return whether persistent setup completed successfully. */
        [[nodiscard]] bool ready() const { return ready_; }

        /** @brief Return the first setup or capture error. */
        [[nodiscard]] const std::string &error() const { return error_; }

        /**
         * @brief Capture and time one immutable query/context geometry.
         * @param point Physical query bucket and resident KV length.
         * @param profiler_mode True when NCU should observe one replay only.
         */
        ProductionPrefillMeasurement measure(
            const ProductionPrefillPoint &point,
            bool profiler_mode)
        {
            ProductionPrefillMeasurement measurement;
            if (!ready_ || !capture(point))
                return measurement;

            float single_replay_ms = 0.0f;
            if (cudaEventRecord(start_, stream_) != cudaSuccess ||
                cudaGraphLaunch(graph_exec_, stream_) != cudaSuccess ||
                cudaEventRecord(stop_, stream_) != cudaSuccess ||
                cudaEventSynchronize(stop_) != cudaSuccess ||
                cudaEventElapsedTime(&single_replay_ms, start_, stop_) != cudaSuccess)
            {
                fail("CUDA calibration replay failed");
                return measurement;
            }

            constexpr double kTargetTimedBatchMs = 40.0;
            const int timed_replays = profiler_mode
                                          ? 1
                                          : std::clamp(
                                                static_cast<int>(
                                                    kTargetTimedBatchMs /
                                                    std::max(0.001f, single_replay_ms)),
                                                1,
                                                100);
            const int sample_count = profiler_mode ? 1 : 5;
            std::vector<double> latency_samples;
            latency_samples.reserve(sample_count);

            for (int sample = 0; sample < sample_count; ++sample)
            {
                if (cudaEventRecord(start_, stream_) != cudaSuccess)
                {
                    fail("CUDA timed start event failed");
                    return measurement;
                }
                for (int replay = 0; replay < timed_replays; ++replay)
                {
                    if (cudaGraphLaunch(graph_exec_, stream_) != cudaSuccess)
                    {
                        fail("CUDA timed graph replay failed");
                        return measurement;
                    }
                }
                if (cudaEventRecord(stop_, stream_) != cudaSuccess ||
                    cudaEventSynchronize(stop_) != cudaSuccess)
                {
                    fail("CUDA terminal timing event failed");
                    return measurement;
                }

                float elapsed_ms = 0.0f;
                if (cudaEventElapsedTime(&elapsed_ms, start_, stop_) != cudaSuccess)
                {
                    fail("CUDA elapsed-time query failed");
                    return measurement;
                }
                latency_samples.push_back(
                    static_cast<double>(elapsed_ms) * 1000.0 / timed_replays);
            }

            std::sort(latency_samples.begin(), latency_samples.end());
            measurement.latency_us =
                latency_samples[latency_samples.size() / 2];
            measurement.query_tokens_per_second =
                static_cast<double>(point.query_rows) * 1.0e6 /
                measurement.latency_us;

            const int prefix_rows = point.kv_len - point.query_rows;
            const double attended_pairs =
                static_cast<double>(point.query_rows) * prefix_rows +
                static_cast<double>(point.query_rows) *
                    (point.query_rows + 1) / 2.0;
            const double flops =
                4.0 * attended_pairs * geometry_.n_heads * geometry_.head_dim;
            measurement.effective_tflops =
                flops / measurement.latency_us / 1.0e6;
            measurement.grid_blocks = captured_grid_blocks_;
            measurement.block_threads = captured_block_threads_;
            measurement.tile_kv = captured_tile_kv_;
            measurement.waves_per_gpu =
                static_cast<double>(captured_grid_blocks_) /
                std::max(1, sm_count_);
            return measurement;
        }

        /**
         * @brief Surface one final device result after every timed replay.
         *
         * The single-value D2H is deliberately outside all timing and is the
         * only result materialization in this benchmark.
         */
        bool finalResultIsFinite()
        {
            float result = 0.0f;
            return ready_ &&
                   cudaMemcpyAsync(
                       &result,
                       d_output_,
                       sizeof(result),
                       cudaMemcpyDeviceToHost,
                       stream_) == cudaSuccess &&
                   cudaStreamSynchronize(stream_) == cudaSuccess &&
                   std::isfinite(result);
        }

    private:
        /** @brief Preserve the first failure and make subsequent work inert. */
        void fail(const std::string &message)
        {
            if (error_.empty())
                error_ = message;
            ready_ = false;
        }

        /** @brief Destroy only the geometry-specific captured graph. */
        void destroyGraph()
        {
            if (graph_exec_)
            {
                (void)cudaGraphExecDestroy(graph_exec_);
                graph_exec_ = nullptr;
            }
            if (graph_)
            {
                (void)cudaGraphDestroy(graph_);
                graph_ = nullptr;
            }
            captured_grid_blocks_ = 0;
            captured_block_threads_ = 0;
            captured_tile_kv_ = 0;
        }

        /** @brief Allocate and initialize the persistent maximum-capacity arena. */
        void initialize()
        {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
            {
                fail("No CUDA device");
                return;
            }

            cudaDeviceProp properties{};
            if (cudaSetDevice(0) != cudaSuccess ||
                cudaGetDeviceProperties(&properties, 0) != cudaSuccess ||
                cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess ||
                cudaEventCreate(&start_) != cudaSuccess ||
                cudaEventCreate(&stop_) != cudaSuccess)
            {
                fail("CUDA device, stream, or event setup failed");
                return;
            }
            sm_count_ = properties.multiProcessorCount;

            const size_t q_elements =
                static_cast<size_t>(maximum_query_rows_) *
                geometry_.n_heads * geometry_.head_dim;
            const size_t kv_elements =
                static_cast<size_t>(maximum_context_) *
                geometry_.n_kv_heads * geometry_.head_dim;
            if (cudaMalloc(
                    reinterpret_cast<void **>(&d_q_),
                    q_elements * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&d_k_, kv_elements * sizeof(uint16_t)) != cudaSuccess ||
                cudaMalloc(&d_v_, kv_elements * sizeof(uint16_t)) != cudaSuccess ||
                cudaMalloc(
                    reinterpret_cast<void **>(&d_output_),
                    q_elements * sizeof(float)) != cudaSuccess)
            {
                fail("CUDA persistent attention arena allocation failed");
                return;
            }

            if (cudaMemsetAsync(
                    d_q_, 0, q_elements * sizeof(float), stream_) != cudaSuccess ||
                cudaMemsetAsync(
                    d_k_, 0, kv_elements * sizeof(uint16_t), stream_) != cudaSuccess ||
                cudaMemsetAsync(
                    d_v_, 0, kv_elements * sizeof(uint16_t), stream_) != cudaSuccess ||
                cudaMemsetAsync(
                    d_output_, 0, q_elements * sizeof(float), stream_) != cudaSuccess ||
                cudaStreamSynchronize(stream_) != cudaSuccess)
            {
                fail("CUDA persistent attention arena initialization failed");
                return;
            }
            ready_ = true;
        }

        /** @brief Launch one production FP16-KV FA2 operation on the exact stream. */
        int launch(const ProductionPrefillPoint &point)
        {
            return cudaFlashAttn_prefill_fa2_fp16kv(
                d_q_,
                d_k_,
                d_v_,
                d_output_,
                /*batch_size=*/1,
                point.query_rows,
                point.kv_len,
                geometry_.n_heads,
                geometry_.n_kv_heads,
                geometry_.head_dim,
                /*causal=*/true,
                /*window_size=*/-1,
                /*position_offset=*/point.kv_len - point.query_rows,
                /*device_params=*/nullptr,
                /*mask=*/nullptr,
                static_cast<void *>(stream_),
                /*device_idx=*/0,
                /*head_start=*/geometry_.head_start,
                /*gqa_n_rep=*/geometry_.launchGqaRep());
        }

        /**
         * @brief Record one immutable launch and recover its real grid geometry.
         */
        bool capture(const ProductionPrefillPoint &point)
        {
            if (point.query_rows <= 0 ||
                point.query_rows > maximum_query_rows_ ||
                point.kv_len < point.query_rows ||
                point.kv_len > maximum_context_)
            {
                fail("Invalid production prefill benchmark geometry");
                return false;
            }

            destroyGraph();

            // Prime function attributes and the compiled specialization before
            // capture. This launch is setup and never contributes timing.
            if (launch(point) != 0 ||
                cudaStreamSynchronize(stream_) != cudaSuccess)
            {
                fail("CUDA FA2 setup launch failed");
                return false;
            }

            bool launch_recorded = false;
            cudaError_t end_status = cudaSuccess;
            {
                GraphCaptureGuard capture_guard;
                if (cudaStreamBeginCapture(
                        stream_, cudaStreamCaptureModeGlobal) != cudaSuccess)
                {
                    fail("CUDA FA2 graph begin-capture failed");
                    return false;
                }
                launch_recorded = launch(point) == 0;
                end_status = cudaStreamEndCapture(stream_, &graph_);
            }
            if (!launch_recorded || end_status != cudaSuccess || !graph_ ||
                cudaGraphInstantiate(
                    &graph_exec_, graph_, nullptr, nullptr, 0) != cudaSuccess)
            {
                fail("CUDA FA2 graph capture or instantiation failed");
                return false;
            }

            size_t node_count = 0;
            if (cudaGraphGetNodes(graph_, nullptr, &node_count) != cudaSuccess ||
                node_count != 1)
            {
                fail("CUDA FA2 benchmark graph must contain exactly one kernel node");
                return false;
            }
            std::array<cudaGraphNode_t, 1> nodes{};
            if (cudaGraphGetNodes(graph_, nodes.data(), &node_count) != cudaSuccess)
            {
                fail("CUDA FA2 graph node query failed");
                return false;
            }
            cudaGraphNodeType node_type{};
            cudaKernelNodeParams params{};
            if (cudaGraphNodeGetType(nodes.front(), &node_type) != cudaSuccess ||
                node_type != cudaGraphNodeTypeKernel ||
                cudaGraphKernelNodeGetParams(nodes.front(), &params) != cudaSuccess)
            {
                fail("CUDA FA2 graph did not expose one kernel launch");
                return false;
            }
            captured_grid_blocks_ =
                static_cast<int>(params.gridDim.x * params.gridDim.y * params.gridDim.z);
            captured_block_threads_ =
                static_cast<int>(params.blockDim.x * params.blockDim.y * params.blockDim.z);
            const int pv_warps_per_query_group =
                geometry_.head_dim > 128
                    ? mutableDebugEnv().attention.cuda_fa2_hd256_pv_warps
                    : 1;
            const int consumer_warps =
                captured_block_threads_ / 32 - 2;
            if (pv_warps_per_query_group <= 0 ||
                consumer_warps <= 0 ||
                consumer_warps % pv_warps_per_query_group != 0)
            {
                fail("CUDA FA2 graph exposed an invalid consumer-warp geometry");
                return false;
            }
            const int query_warp_groups =
                consumer_warps / pv_warps_per_query_group;
            for (const int tile_kv : {16, 32, 64})
            {
                if (params.sharedMemBytes ==
                    llaminar2::cuda::fa2_policy::fa2DynamicSharedMemoryBytes(
                        geometry_.head_dim,
                        query_warp_groups,
                        tile_kv))
                {
                    captured_tile_kv_ = tile_kv;
                    break;
                }
            }
            if (captured_tile_kv_ == 0)
            {
                fail("CUDA FA2 graph shared memory does not identify a compiled K/V tile");
                return false;
            }
            return true;
        }

        cudaStream_t stream_ = nullptr;
        cudaEvent_t start_ = nullptr;
        cudaEvent_t stop_ = nullptr;
        cudaGraph_t graph_ = nullptr;
        cudaGraphExec_t graph_exec_ = nullptr;
        float *d_q_ = nullptr;
        void *d_k_ = nullptr;
        void *d_v_ = nullptr;
        float *d_output_ = nullptr;
        int sm_count_ = 0;
        int captured_grid_blocks_ = 0;
        int captured_block_threads_ = 0;
        int captured_tile_kv_ = 0;
        AttentionGeometry geometry_{};
        int maximum_query_rows_ = 0;
        int maximum_context_ = 0;
        bool ready_ = false;
        std::string error_;
    };

    /**
     * @brief Own and compare three captured FA2 context scheduling transactions.
     *
     * The fixture allocates maximum-capacity Q/K/V, output, device-parameter,
     * and summary buffers once. For each immutable query bucket it captures:
     *
     * 1. the current one-node production FA2 launch;
     * 2. optionally, one diagnostic phase-one node per capacity partition plus
     *    a reducer;
     * 3. one production context-parallel phase-one grid plus the same reducer.
     *
     * Both production graphs use the full cache-capacity envelope. Live K/V
     * length and position are published through `AttentionDeviceParams` before
     * capture, exactly as they are for device-owned inference replay. A bounded
     * persistent phase grid strides over only the live canonical partitions;
     * capacity-tail partitions neither launch dedicated blocks nor publish
     * neutral summary traffic.
     *
     * Timed graph replays perform no allocation, transfer, host callback, or
     * synchronization.  Events bracket a batch of complete graph transactions,
     * and only the terminal timing event is synchronized.  Output materialization
     * is a separate post-measurement diagnostic.
     */
    class CapturedContextPartitionPrefill final
    {
    public:
        /**
         * @brief Allocate one persistent hypothesis arena.
         * @param geometry Participant-local attention geometry.
         * @param maximum_query_rows Largest measured query bucket.
         * @param maximum_context Largest resident K/V span.
         * @param context_partition_size Fixed logical K/V partition width.
         * @param include_sequence_reference Whether to capture and time the
         *        many-node arithmetic diagnostic in addition to production.
         */
        CapturedContextPartitionPrefill(
            AttentionGeometry geometry,
            int maximum_query_rows,
            int maximum_context,
            int context_partition_size,
            bool include_sequence_reference)
            : geometry_(geometry),
              maximum_query_rows_(maximum_query_rows),
              maximum_context_(maximum_context),
              context_partition_size_(context_partition_size),
              include_sequence_reference_(include_sequence_reference),
              maximum_context_partitions_(
                  context_partition_size > 0
                      ? (maximum_context + context_partition_size - 1) /
                            context_partition_size
                      : 0)
        {
            initialize();
        }

        ~CapturedContextPartitionPrefill()
        {
            destroyGraphs();
            if (start_)
                (void)cudaEventDestroy(start_);
            if (stop_)
                (void)cudaEventDestroy(stop_);
            if (d_q_)
                (void)cudaFree(d_q_);
            if (d_k_)
                (void)cudaFree(d_k_);
            if (d_v_)
                (void)cudaFree(d_v_);
            if (d_direct_output_)
                (void)cudaFree(d_direct_output_);
            if (d_sequence_output_)
                (void)cudaFree(d_sequence_output_);
            if (d_context_output_)
                (void)cudaFree(d_context_output_);
            if (d_partial_output_)
                (void)cudaFree(d_partial_output_);
            if (d_partial_m_)
                (void)cudaFree(d_partial_m_);
            if (d_partial_l_)
                (void)cudaFree(d_partial_l_);
            if (d_device_params_)
                (void)cudaFree(d_device_params_);
            if (stream_)
                (void)cudaStreamDestroy(stream_);
        }

        CapturedContextPartitionPrefill(
            const CapturedContextPartitionPrefill &) = delete;
        CapturedContextPartitionPrefill &operator=(
            const CapturedContextPartitionPrefill &) = delete;

        /** @brief Return whether persistent setup and latest capture succeeded. */
        [[nodiscard]] bool ready() const { return ready_; }

        /** @brief Return the first setup, capture, or replay failure. */
        [[nodiscard]] const std::string &error() const { return error_; }

        /**
         * @brief Capture and time complete production transactions.
         * @param point Query bucket and resident K/V span.
         * @param profiler_mode True to execute one replay/sample for NCU.
         * @param reducer_dimension_warps Zero for policy or an exact candidate.
         * @param device_direct_partition_limit Zero forces context publication;
         *        positive values exercise the production device-adaptive graph.
         * @return Median transaction latencies and captured node counts.
         */
        ContextPartitionMeasurement measure(
            const ProductionPrefillPoint &point,
            bool profiler_mode,
            int reducer_dimension_warps,
            int context_partition_slots = 0,
            int device_direct_partition_limit = 0)
        {
            ContextPartitionMeasurement measurement;
            if (!ready_ ||
                !capture(
                    point,
                    reducer_dimension_warps,
                    context_partition_slots,
                    device_direct_partition_limit))
                return measurement;

            measurement.direct_latency_us =
                measureGraph(direct_graph_, profiler_mode);
            if (include_sequence_reference_)
            {
                measurement.sequence_latency_us =
                    measureGraph(sequence_graph_, profiler_mode);
            }
            measurement.context_latency_us =
                measureGraph(context_graph_, profiler_mode);
            measurement.live_context_partitions =
                captured_live_context_partitions_;
            measurement.captured_context_partitions =
                captured_context_partitions_;
            measurement.context_partition_slots =
                captured_context_partition_slots_;
            measurement.device_direct_partition_limit =
                captured_device_direct_partition_limit_;
            measurement.reducer_dimension_warps =
                captured_reducer_dimension_warps_;
            measurement.direct_graph_nodes = direct_graph_.node_count;
            measurement.sequence_graph_nodes = sequence_graph_.node_count;
            measurement.context_graph_nodes = context_graph_.node_count;
            return measurement;
        }

        /**
         * @brief Compare captured direct and context schedules byte-for-byte.
         *
         * The optional ordered-node diagnostic joins the comparison when it was
         * requested at construction. All graphs replay once on the same exact
         * stream. Their outputs are copied only after terminal writes complete;
         * this diagnostic is outside all timing and graph capture.
         */
        bool canonicalOutputsAreByteEqual()
        {
            if (!ready_ || !direct_graph_.exec || !context_graph_.exec ||
                (include_sequence_reference_ && !sequence_graph_.exec))
                return false;

            if (cudaGraphLaunch(direct_graph_.exec, stream_) != cudaSuccess ||
                (include_sequence_reference_ &&
                 cudaGraphLaunch(sequence_graph_.exec, stream_) != cudaSuccess) ||
                cudaGraphLaunch(context_graph_.exec, stream_) != cudaSuccess)
            {
                fail("CUDA canonical comparison replay failed");
                return false;
            }

            const size_t output_elements =
                static_cast<size_t>(captured_query_rows_) *
                geometry_.n_heads * geometry_.head_dim;
            std::vector<float> direct_output(output_elements);
            std::vector<float> sequence_output(
                include_sequence_reference_ ? output_elements : 0);
            std::vector<float> context_output(output_elements);
            if (cudaMemcpyAsync(
                    direct_output.data(),
                    d_direct_output_,
                    output_elements * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream_) != cudaSuccess ||
                (include_sequence_reference_ &&
                 cudaMemcpyAsync(
                     sequence_output.data(),
                     d_sequence_output_,
                     output_elements * sizeof(float),
                     cudaMemcpyDeviceToHost,
                     stream_) != cudaSuccess) ||
                cudaMemcpyAsync(
                    context_output.data(),
                    d_context_output_,
                    output_elements * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream_) != cudaSuccess ||
                cudaStreamSynchronize(stream_) != cudaSuccess)
            {
                fail("CUDA canonical comparison materialization failed");
                return false;
            }

            const size_t output_bytes = output_elements * sizeof(float);
            const bool sequence_equal =
                !include_sequence_reference_ ||
                std::memcmp(
                    direct_output.data(),
                    sequence_output.data(),
                    output_bytes) == 0;
            return sequence_equal &&
                   std::memcmp(
                       direct_output.data(),
                       context_output.data(),
                       output_bytes) == 0;
        }

    private:
        /** @brief One owned CUDA graph and its executable transaction. */
        struct CapturedGraph
        {
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            std::size_t node_count = 0;
        };

        /** @brief Preserve the first failure and make later work inert. */
        void fail(const std::string &message)
        {
            if (error_.empty())
                error_ = message;
            ready_ = false;
        }

        /** @brief Destroy one captured graph in executable-before-source order. */
        static void destroyGraph(CapturedGraph &captured)
        {
            if (captured.exec)
            {
                (void)cudaGraphExecDestroy(captured.exec);
                captured.exec = nullptr;
            }
            if (captured.graph)
            {
                (void)cudaGraphDestroy(captured.graph);
                captured.graph = nullptr;
            }
            captured.node_count = 0;
        }

        /** @brief Destroy every geometry-specific graph transaction. */
        void destroyGraphs()
        {
            destroyGraph(direct_graph_);
            destroyGraph(sequence_graph_);
            destroyGraph(context_graph_);
            captured_query_rows_ = 0;
            captured_live_context_partitions_ = 0;
            captured_context_partitions_ = 0;
            captured_context_partition_slots_ = 0;
            captured_device_direct_partition_limit_ = 0;
            captured_reducer_dimension_warps_ = 0;
        }

        /** @brief Allocate and initialize all persistent maximum-capacity buffers. */
        void initialize()
        {
            if (maximum_query_rows_ <= 0 || maximum_context_ <= 0 ||
                context_partition_size_ !=
                    llaminar2::cuda::fa2_policy::
                        kFA2CanonicalContextPartitionKeys ||
                maximum_context_partitions_ <= 0)
            {
                fail("Invalid context-partition benchmark capacity");
                return;
            }

            int device_count = 0;
            cudaDeviceProp properties{};
            if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
                device_count <= 0 || cudaSetDevice(0) != cudaSuccess ||
                cudaGetDeviceProperties(&properties, 0) != cudaSuccess ||
                cudaStreamCreateWithFlags(
                    &stream_, cudaStreamNonBlocking) != cudaSuccess ||
                cudaEventCreate(&start_) != cudaSuccess ||
                cudaEventCreate(&stop_) != cudaSuccess)
            {
                fail("CUDA context-partition device setup failed");
                return;
            }
            sm_count_ = properties.multiProcessorCount;

            const size_t q_elements =
                static_cast<size_t>(maximum_query_rows_) *
                geometry_.n_heads * geometry_.head_dim;
            const size_t kv_elements =
                static_cast<size_t>(maximum_context_) *
                geometry_.n_kv_heads * geometry_.head_dim;
            const size_t summary_scalars =
                static_cast<size_t>(maximum_query_rows_) *
                geometry_.n_heads * maximum_context_partitions_;
            const size_t summary_elements =
                summary_scalars * geometry_.head_dim;

            if (cudaMalloc(
                    reinterpret_cast<void **>(&d_q_),
                    q_elements * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&d_k_, kv_elements * sizeof(uint16_t)) != cudaSuccess ||
                cudaMalloc(&d_v_, kv_elements * sizeof(uint16_t)) != cudaSuccess ||
                cudaMalloc(
                    reinterpret_cast<void **>(&d_direct_output_),
                    q_elements * sizeof(float)) != cudaSuccess ||
                cudaMalloc(
                    reinterpret_cast<void **>(&d_sequence_output_),
                    q_elements * sizeof(float)) != cudaSuccess ||
                cudaMalloc(
                    reinterpret_cast<void **>(&d_context_output_),
                    q_elements * sizeof(float)) != cudaSuccess ||
                cudaMalloc(
                    reinterpret_cast<void **>(&d_partial_output_),
                    summary_elements * sizeof(float)) != cudaSuccess ||
                cudaMalloc(
                    reinterpret_cast<void **>(&d_partial_m_),
                    summary_scalars * sizeof(float)) != cudaSuccess ||
                cudaMalloc(
                    reinterpret_cast<void **>(&d_partial_l_),
                    summary_scalars * sizeof(float)) != cudaSuccess ||
                cudaMalloc(
                    reinterpret_cast<void **>(&d_device_params_),
                    sizeof(attention::AttentionDeviceParams)) != cudaSuccess)
            {
                fail("CUDA context-partition persistent arena allocation failed");
                return;
            }

            if (!initializeDeterministicTensorPatterns(q_elements, kv_elements) ||
                cudaMemsetAsync(
                    d_direct_output_, 0,
                    q_elements * sizeof(float), stream_) != cudaSuccess ||
                cudaMemsetAsync(
                    d_sequence_output_, 0,
                    q_elements * sizeof(float), stream_) != cudaSuccess ||
                cudaMemsetAsync(
                    d_context_output_, 0,
                    q_elements * sizeof(float), stream_) != cudaSuccess ||
                cudaMemsetAsync(
                    d_partial_output_, 0,
                    summary_elements * sizeof(float), stream_) != cudaSuccess ||
                cudaMemsetAsync(
                    d_partial_m_, 0,
                    summary_scalars * sizeof(float), stream_) != cudaSuccess ||
                cudaMemsetAsync(
                    d_partial_l_, 0,
                    summary_scalars * sizeof(float), stream_) != cudaSuccess ||
                cudaMemsetAsync(
                    d_device_params_, 0,
                    sizeof(attention::AttentionDeviceParams), stream_) != cudaSuccess ||
                cudaStreamSynchronize(stream_) != cudaSuccess)
            {
                fail("CUDA context-partition persistent arena initialization failed");
                return;
            }
            ready_ = true;
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
            if (cudaMemsetAsync(
                    d_q_, 0x3c,
                    q_elements * sizeof(float), stream_) != cudaSuccess)
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
                if (cudaMemsetAsync(
                        static_cast<unsigned char *>(d_k_) +
                            begin * sizeof(uint16_t),
                        k_pattern,
                        count * sizeof(uint16_t),
                        stream_) != cudaSuccess ||
                    cudaMemsetAsync(
                        static_cast<unsigned char *>(d_v_) +
                            begin * sizeof(uint16_t),
                        v_pattern,
                        count * sizeof(uint16_t),
                        stream_) != cudaSuccess)
                {
                    return false;
                }
            }
            return true;
        }

        /** @brief Submit the current one-node production FA2 launch. */
        int launchDirect(const ProductionPrefillPoint &point)
        {
            return cudaFlashAttn_prefill_fa2_fp16kv(
                d_q_,
                d_k_,
                d_v_,
                d_direct_output_,
                /*batch_size=*/1,
                point.query_rows,
                maximum_context_,
                geometry_.n_heads,
                geometry_.n_kv_heads,
                geometry_.head_dim,
                /*causal=*/true,
                /*window_size=*/-1,
                /*position_offset=*/0,
                d_device_params_,
                /*mask=*/nullptr,
                static_cast<void *>(stream_),
                /*device_idx=*/0,
                geometry_.head_start,
                geometry_.launchGqaRep());
        }

        /**
         * @brief Publish one graph-owned attention geometry and optional branch.
         *
         * The direct and adaptive timing graphs both include this producer. In
         * the adaptive graph it additionally sets the native conditional handle,
         * matching the production cache-count writer without adding a benchmark-
         * only predicate kernel.
         */
        int publishDeviceParams(
            const ProductionPrefillPoint &point,
            cudaGraphConditionalHandle condition = 0,
            int direct_partition_limit = 0)
        {
            const int direct_kv_limit =
                direct_partition_limit * context_partition_size_;
            return cudaFlashAttn_prepare_device_params_from_geometry(
                d_device_params_,
                point.kv_len,
                maximum_context_,
                point.kv_len - point.query_rows,
                /*query_rows=*/1,
                static_cast<unsigned long long>(condition),
                direct_kv_limit,
                static_cast<void *>(stream_));
        }

        /**
         * @brief Submit one canonical fixed-partition transaction.
         * @param point Immutable query/KV geometry.
         * @param context_parallel True for one wide phase-one grid.
         */
        int launchCanonical(
            const ProductionPrefillPoint &point,
            bool context_parallel,
            int reducer_dimension_warps = 0,
            int device_direct_partition_limit = 0,
            CUDAActiveCaptureConditional *capture_conditional = nullptr)
        {
            auto launcher = context_parallel
                                ? cudaFlashAttn_prefill_fa2_fp16kv_context_parallel
                                : cudaFlashAttn_prefill_fa2_fp16kv_partitioned_sequence;
            return launcher(
                d_q_,
                d_k_,
                d_v_,
                context_parallel ? d_context_output_ : d_sequence_output_,
                d_partial_output_,
                d_partial_m_,
                d_partial_l_,
                /*batch_size=*/1,
                point.query_rows,
                maximum_context_,
                geometry_.n_heads,
                geometry_.n_kv_heads,
                geometry_.head_dim,
                /*causal=*/true,
                /*window_size=*/-1,
                /*position_offset=*/0,
                d_device_params_,
                /*mask=*/nullptr,
                context_partition_size_,
                maximum_context_partitions_,
                captured_context_partition_slots_,
                device_direct_partition_limit,
                reducer_dimension_warps,
                capture_conditional,
                static_cast<void *>(stream_),
                /*device_idx=*/0,
                geometry_.head_start,
                geometry_.launchGqaRep());
        }

        /**
         * @brief Record one immutable graph and authenticate all nodes as kernels.
         * @tparam Launch Callable that submits the transaction on `stream_`.
         */
        template <typename Launch>
        bool captureGraph(CapturedGraph &captured, Launch &&launch)
        {
            bool launch_recorded = false;
            cudaError_t end_status = cudaSuccess;
            {
                GraphCaptureGuard capture_guard;
                if (cudaStreamBeginCapture(
                        stream_, cudaStreamCaptureModeGlobal) != cudaSuccess)
                {
                    fail("CUDA context transaction graph begin-capture failed");
                    return false;
                }
                launch_recorded = launch() == 0;
                end_status = cudaStreamEndCapture(stream_, &captured.graph);
            }
            if (!launch_recorded || end_status != cudaSuccess ||
                !captured.graph ||
                cudaGraphInstantiate(
                    &captured.exec,
                    captured.graph,
                    nullptr,
                    nullptr,
                    0) != cudaSuccess)
            {
                fail("CUDA context transaction capture or instantiation failed");
                return false;
            }

            if (cudaGraphGetNodes(
                    captured.graph,
                    nullptr,
                    &captured.node_count) != cudaSuccess ||
                captured.node_count == 0)
            {
                fail("CUDA context transaction graph has no nodes");
                return false;
            }
            std::vector<cudaGraphNode_t> nodes(captured.node_count);
            if (cudaGraphGetNodes(
                    captured.graph,
                    nodes.data(),
                    &captured.node_count) != cudaSuccess)
            {
                fail("CUDA context transaction node query failed");
                return false;
            }
            std::size_t root_kernel_count = 0;
            std::size_t root_conditional_count = 0;
            for (cudaGraphNode_t node : nodes)
            {
                cudaGraphNodeType type{};
                if (cudaGraphNodeGetType(node, &type) != cudaSuccess)
                {
                    fail("CUDA context transaction node classification failed");
                    return false;
                }
                if (type == cudaGraphNodeTypeKernel)
                {
                    ++root_kernel_count;
                    continue;
                }
                if (type == cudaGraphNodeTypeConditional)
                {
                    ++root_conditional_count;
                    continue;
                }
                fail("CUDA context transaction contains a forbidden root node");
                return false;
            }
            if (root_kernel_count == 0 || root_conditional_count > 1)
            {
                fail("CUDA context transaction must expose a predicate/compute kernel and at most one native conditional");
                return false;
            }
            return true;
        }

        /**
         * @brief Prime specializations, then capture direct/sequence/context graphs.
         * @param point Immutable query and resident-context geometry.
         * @param reducer_dimension_warps Zero for policy or an exact candidate.
         * @param context_partition_slots Zero for policy or an exact candidate.
         * @param device_direct_partition_limit Device-owned direct prefix for
         *        the adaptive context graph; zero forces context publication.
         */
        bool capture(
            const ProductionPrefillPoint &point,
            int reducer_dimension_warps,
            int context_partition_slots,
            int device_direct_partition_limit)
        {
            if (point.query_rows <= 0 ||
                point.query_rows > maximum_query_rows_ ||
                point.kv_len < point.query_rows ||
                point.kv_len > maximum_context_)
            {
                fail("Invalid context-partition measurement geometry");
                return false;
            }

            destroyGraphs();
            captured_query_rows_ = point.query_rows;
            captured_live_context_partitions_ =
                (point.kv_len + context_partition_size_ - 1) /
                context_partition_size_;
            captured_context_partitions_ =
                maximum_context_partitions_;
            captured_context_partition_slots_ =
                llaminar2::cuda::fa2_policy::
                    selectFA2ContextPartitionSlots(
                        {
                            .batch_size = 1,
                            .query_rows = point.query_rows,
                            .local_query_heads = geometry_.n_heads,
                            .head_dim = geometry_.head_dim,
                            .sm_count = sm_count_,
                        },
                        maximum_context_partitions_,
                        context_partition_slots);
            captured_reducer_dimension_warps_ =
                llaminar2::cuda::fa2_policy::
                    selectFA2ReducerDimensionWarps(
                        {
                            .batch_size = 1,
                            .query_rows = point.query_rows,
                            .local_query_heads = geometry_.n_heads,
                            .head_dim = geometry_.head_dim,
                            .sm_count = sm_count_,
                        },
                        reducer_dimension_warps);
            captured_device_direct_partition_limit_ =
                device_direct_partition_limit;
            if (captured_context_partition_slots_ <= 0 ||
                captured_reducer_dimension_warps_ <= 0 ||
                captured_device_direct_partition_limit_ < 0 ||
                captured_device_direct_partition_limit_ >
                    captured_context_partitions_)
            {
                fail("CUDA context transaction reducer policy is invalid");
                return false;
            }

            const attention::AttentionDeviceParams device_params{
                .kv_len = point.kv_len,
                .kv_stride = maximum_context_,
                .position_offset = point.kv_len - point.query_rows,
                .mask_stride = maximum_context_,
            };
            if (cudaMemcpyAsync(
                    d_device_params_,
                    &device_params,
                    sizeof(device_params),
                    cudaMemcpyHostToDevice,
                    stream_) != cudaSuccess)
            {
                fail("CUDA context transaction device parameter publication failed");
                return false;
            }

            // Function attributes and every template specialization are primed
            // before capture.  One setup synchronization covers all three paths.
            if (launchDirect(point) != 0 ||
                (include_sequence_reference_ &&
                 launchCanonical(
                     point,
                     /*context_parallel=*/false,
                     reducer_dimension_warps,
                     /*device_direct_partition_limit=*/0) != 0) ||
                launchCanonical(
                    point,
                    /*context_parallel=*/true,
                    reducer_dimension_warps,
                    captured_device_direct_partition_limit_) != 0 ||
                cudaStreamSynchronize(stream_) != cudaSuccess)
            {
                fail("CUDA context transaction setup launch failed");
                return false;
            }

            if (!captureGraph(
                    direct_graph_,
                    [&]()
                    {
                        return publishDeviceParams(point) == 0
                                   ? launchDirect(point)
                                   : -1;
                    }) ||
                (include_sequence_reference_ &&
                 !captureGraph(
                     sequence_graph_,
                     [&]()
                     {
                         return publishDeviceParams(point) == 0
                                    ? launchCanonical(
                                          point,
                                          /*context_parallel=*/false,
                                          reducer_dimension_warps,
                                          /*device_direct_partition_limit=*/0)
                                    : -1;
                     })) ||
                !captureGraph(
                    context_graph_,
                    [&]()
                    {
                        if (captured_device_direct_partition_limit_ > 0)
                        {
                            CUDAActiveCaptureConditional transaction(
                                stream_,
                                CUDAActiveCaptureConditionalKind::IfOnly);
                            if (!transaction.ready() ||
                                !transaction.publishPredicate(
                                    [&](cudaGraphConditionalHandle condition)
                                    {
                                        return publishDeviceParams(
                                                   point,
                                                   condition,
                                                   captured_device_direct_partition_limit_) == 0;
                                    }))
                            {
                                fail(
                                    "CUDA context transaction could not fuse branch publication into device params");
                                return -1;
                            }
                            return launchCanonical(
                                point,
                                /*context_parallel=*/true,
                                reducer_dimension_warps,
                                captured_device_direct_partition_limit_,
                                &transaction);
                        }

                        if (publishDeviceParams(point) != 0)
                            return -1;
                        return launchCanonical(
                            point,
                            /*context_parallel=*/true,
                            reducer_dimension_warps,
                            captured_device_direct_partition_limit_);
                    }))
            {
                return false;
            }

            if (direct_graph_.node_count != 2 ||
                (include_sequence_reference_ &&
                 sequence_graph_.node_count !=
                     static_cast<size_t>(captured_context_partitions_ + 2)) ||
                context_graph_.node_count != 3u)
            {
                fail("CUDA context transaction graph topology is not exact");
                return false;
            }
            return true;
        }

        /**
         * @brief Return median microseconds for complete captured transactions.
         */
        double measureGraph(
            const CapturedGraph &captured,
            bool profiler_mode)
        {
            float calibration_ms = 0.0f;
            if (cudaEventRecord(start_, stream_) != cudaSuccess ||
                cudaGraphLaunch(captured.exec, stream_) != cudaSuccess ||
                cudaEventRecord(stop_, stream_) != cudaSuccess ||
                cudaEventSynchronize(stop_) != cudaSuccess ||
                cudaEventElapsedTime(
                    &calibration_ms, start_, stop_) != cudaSuccess)
            {
                fail("CUDA context transaction calibration failed");
                return 0.0;
            }

            constexpr double kTargetTimedBatchMs = 40.0;
            const int timed_replays = profiler_mode
                                          ? 1
                                          : std::clamp(
                                                static_cast<int>(
                                                    kTargetTimedBatchMs /
                                                    std::max(
                                                        0.001f,
                                                        calibration_ms)),
                                                1,
                                                100);
            const int samples = profiler_mode ? 1 : 5;
            std::vector<double> latency_samples;
            latency_samples.reserve(samples);

            for (int sample = 0; sample < samples; ++sample)
            {
                if (cudaEventRecord(start_, stream_) != cudaSuccess)
                {
                    fail("CUDA context transaction timing start failed");
                    return 0.0;
                }
                for (int replay = 0; replay < timed_replays; ++replay)
                {
                    if (cudaGraphLaunch(captured.exec, stream_) != cudaSuccess)
                    {
                        fail("CUDA context transaction replay failed");
                        return 0.0;
                    }
                }
                if (cudaEventRecord(stop_, stream_) != cudaSuccess ||
                    cudaEventSynchronize(stop_) != cudaSuccess)
                {
                    fail("CUDA context transaction terminal event failed");
                    return 0.0;
                }
                float elapsed_ms = 0.0f;
                if (cudaEventElapsedTime(
                        &elapsed_ms, start_, stop_) != cudaSuccess)
                {
                    fail("CUDA context transaction elapsed-time query failed");
                    return 0.0;
                }
                latency_samples.push_back(
                    static_cast<double>(elapsed_ms) * 1000.0 / timed_replays);
            }
            std::sort(latency_samples.begin(), latency_samples.end());
            return latency_samples[latency_samples.size() / 2];
        }

        cudaStream_t stream_ = nullptr;
        cudaEvent_t start_ = nullptr;
        cudaEvent_t stop_ = nullptr;
        CapturedGraph direct_graph_{};
        CapturedGraph sequence_graph_{};
        CapturedGraph context_graph_{};
        float *d_q_ = nullptr;
        void *d_k_ = nullptr;
        void *d_v_ = nullptr;
        float *d_direct_output_ = nullptr;
        float *d_sequence_output_ = nullptr;
        float *d_context_output_ = nullptr;
        float *d_partial_output_ = nullptr;
        float *d_partial_m_ = nullptr;
        float *d_partial_l_ = nullptr;
        attention::AttentionDeviceParams *d_device_params_ = nullptr;
        AttentionGeometry geometry_{};
        int maximum_query_rows_ = 0;
        int maximum_context_ = 0;
        int context_partition_size_ = 0;
        bool include_sequence_reference_ = false;
        int sm_count_ = 0;
        int maximum_context_partitions_ = 0;
        int captured_query_rows_ = 0;
        int captured_live_context_partitions_ = 0;
        int captured_context_partitions_ = 0;
        int captured_context_partition_slots_ = 0;
        int captured_device_direct_partition_limit_ = 0;
        int captured_reducer_dimension_warps_ = 0;
        bool ready_ = false;
        std::string error_;
    };

    // ============================================================================
    // CUDA Flash Attention Performance Test Fixture
    // ============================================================================

    class CUDAFlashAttentionPerf : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Check CUDA availability
            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount(&device_count);
            if (err != cudaSuccess || device_count == 0)
            {
                GTEST_SKIP() << "No CUDA devices available";
            }

            // Initialize device manager
            DeviceManager::instance().initialize(-1);
            if (!DeviceManager::instance().has_gpu())
            {
                GTEST_SKIP() << "No CUDA GPU available";
            }

            // Get device properties
            cudaGetDeviceProperties(&device_props_, 0);
            compute_capability_ = device_props_.major * 10 + device_props_.minor;

            std::cout << "Device: " << device_props_.name
                      << " (SM " << device_props_.major << "." << device_props_.minor << ")"
                      << std::endl;

            // Every measured launch uses one explicit non-default stream.  The
            // event synchronization below is benchmark result collection; no
            // synchronization is captured into the production hot path.
            ASSERT_EQ(
                cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                cudaSuccess);
            ASSERT_EQ(cudaEventCreate(&start_event_), cudaSuccess);
            ASSERT_EQ(cudaEventCreate(&stop_event_), cudaSuccess);

            // Initialize MPI context (single rank for local test)
            mpi_ctx_ = std::make_unique<MPIContext>(0, 1, MPI_COMM_SELF);

            // Initialize kernel
            kernel_ = std::make_unique<CUDAFlashAttentionKernelT<ActivationPrecision::FP32>>(0);
            kernel_->bindGPUStream(
                ExplicitGPUStream(static_cast<void *>(stream_)));
        }

        void TearDown() override
        {
            if (kernel_)
                kernel_->clearGPUStreamBinding();
            if (start_event_)
                cudaEventDestroy(start_event_);
            if (stop_event_)
                cudaEventDestroy(stop_event_);
            kernel_.reset();
            mpi_ctx_.reset();
            if (stream_)
                cudaStreamDestroy(stream_);
        }

        /**
         * @brief Generate random FP32 data
         */
        std::vector<float> randomFP32(size_t count, float scale = 0.1f)
        {
            std::vector<float> data(count);
            std::mt19937 rng(42);
            std::normal_distribution<float> dist(0.0f, scale);
            for (auto &v : data)
                v = dist(rng);
            return data;
        }

        /**
         * @brief Run prefill benchmark for a single configuration
         *
         * NOTE: Currently only head_dim=64 is supported by the CUDA Flash Attention kernels.
         * head_dim=128 support requires kernel modifications to handle larger shared memory
         * requirements and different WMMA fragment shapes.
         */
        void runPrefillBenchmark(const BenchConfig &cfg)
        {
            const int seq_len = cfg.seq_len;
            const int n_heads = cfg.n_heads;
            const int n_kv_heads = cfg.n_kv_heads;
            const int head_dim = cfg.head_dim;

            const size_t q_size = seq_len * n_heads * head_dim;
            const size_t kv_size = seq_len * n_kv_heads * head_dim;
            const size_t out_size = seq_len * n_heads * head_dim;

            // Generate host data
            auto Q_data = randomFP32(q_size);
            auto K_data = randomFP32(kv_size);
            auto V_data = randomFP32(kv_size);

            // Allocate device memory
            float *d_Q, *d_K, *d_V, *d_output;
            cudaMalloc(&d_Q, q_size * sizeof(float));
            cudaMalloc(&d_K, kv_size * sizeof(float));
            cudaMalloc(&d_V, kv_size * sizeof(float));
            cudaMalloc(&d_output, out_size * sizeof(float));

            // Copy data to device
            cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);

            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; ++i)
            {
                kernel_->compute(
                    d_Q, d_K, d_V, d_output,
                    seq_len, n_heads, n_kv_heads, head_dim,
                    true,                               // causal
                    -1,                                 // window_size
                    nullptr, nullptr, nullptr, nullptr, // workspace buffers
                    false,                              // use_bf16
                    mpi_ctx_.get(),
                    0 // device_idx
                );
            }
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

            // Benchmark
            std::vector<float> times_ms;
            times_ms.reserve(BENCHMARK_ITERATIONS);

            for (int i = 0; i < BENCHMARK_ITERATIONS; ++i)
            {
                cudaEventRecord(start_event_, stream_);
                kernel_->compute(
                    d_Q, d_K, d_V, d_output,
                    seq_len, n_heads, n_kv_heads, head_dim,
                    true, -1, nullptr, nullptr, nullptr, nullptr, false,
                    mpi_ctx_.get(), 0);
                cudaEventRecord(stop_event_, stream_);
                cudaEventSynchronize(stop_event_);

                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start_event_, stop_event_);
                times_ms.push_back(ms);
            }

            // Cleanup
            cudaFree(d_Q);
            cudaFree(d_K);
            cudaFree(d_V);
            cudaFree(d_output);

            // Calculate statistics
            std::sort(times_ms.begin(), times_ms.end());
            float median_ms = times_ms[times_ms.size() / 2];
            float min_ms = times_ms.front();
            float max_ms = times_ms.back();

            // Calculate metrics
            double tflops_achieved = cfg.computeTFLOPs() / (median_ms / 1000.0);
            double memory_gb_s = cfg.computeMemoryGB() / (median_ms / 1000.0);
            double tokens_per_sec = seq_len / (median_ms / 1000.0);

            // Print results
            printPrefillResult(cfg, median_ms, min_ms, max_ms, tflops_achieved, memory_gb_s, tokens_per_sec);
        }

        /**
         * @brief Run decode benchmark for a single configuration
         *
         * NOTE: Currently only head_dim=64 is supported by the CUDA Flash Attention kernels.
         */
        void runDecodeBenchmark(const BenchConfig &cfg)
        {
            const int seq_len = 1; // Decode: single token
            const int kv_len = cfg.kv_len;
            const int n_heads = cfg.n_heads;
            const int n_kv_heads = cfg.n_kv_heads;
            const int head_dim = cfg.head_dim;

            const size_t q_size = n_heads * head_dim;
            const size_t kv_size = kv_len * n_kv_heads * head_dim;
            const size_t out_size = n_heads * head_dim;
            auto requirements = kernel_->getWorkspaceRequirements(1, n_heads, head_dim);
            workspace_ = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::cuda(0), requirements.total_bytes_with_alignment() + 4096);
            ASSERT_TRUE(workspace_->allocate(requirements));
            kernel_->bindWorkspace(workspace_.get());

            // Generate host data
            auto Q_data = randomFP32(q_size);
            auto K_data = randomFP32(kv_size);
            auto V_data = randomFP32(kv_size);

            // Allocate device memory
            float *d_Q, *d_K, *d_V, *d_output;
            cudaMalloc(&d_Q, q_size * sizeof(float));
            cudaMalloc(&d_K, kv_size * sizeof(float));
            cudaMalloc(&d_V, kv_size * sizeof(float));
            cudaMalloc(&d_output, out_size * sizeof(float));

            // Copy data to device
            cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);

            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; ++i)
            {
                kernel_->compute_decode(
                    d_Q, d_K, d_V, d_output,
                    seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                    true, // causal
                    0     // position_offset
                );
            }
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

            // Benchmark
            std::vector<float> times_ms;
            times_ms.reserve(BENCHMARK_ITERATIONS);

            for (int i = 0; i < BENCHMARK_ITERATIONS; ++i)
            {
                cudaEventRecord(start_event_, stream_);
                kernel_->compute_decode(
                    d_Q, d_K, d_V, d_output,
                    seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                    true, 0 // causal, position_offset
                );
                cudaEventRecord(stop_event_, stream_);
                cudaEventSynchronize(stop_event_);

                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start_event_, stop_event_);
                times_ms.push_back(ms);
            }

            // Cleanup
            cudaFree(d_Q);
            cudaFree(d_K);
            cudaFree(d_V);
            cudaFree(d_output);

            // Calculate statistics
            std::sort(times_ms.begin(), times_ms.end());
            float median_ms = times_ms[times_ms.size() / 2];
            float min_ms = times_ms.front();
            float max_ms = times_ms.back();

            // Calculate latency-focused metrics
            double latency_us = median_ms * 1000.0; // Convert to microseconds
            double tokens_per_sec = 1.0 / (median_ms / 1000.0);

            // Print results
            printDecodeResult(cfg, latency_us, min_ms * 1000.0, max_ms * 1000.0, tokens_per_sec);
        }

        void printTableHeader(const std::string &test_name)
        {
            std::cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════════════╗" << std::endl;
            std::cout << "║ " << std::setw(92) << std::left << test_name << "║" << std::endl;
            std::cout << "║ Device: " << std::setw(84) << std::left << device_props_.name << "║" << std::endl;
            std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════════════╣" << std::endl;
        }

        void printPrefillHeader()
        {
            std::cout << "│ Config                    │ Median(ms) │ Min(ms) │ Max(ms) │ TFLOPS │ GB/s   │ Tok/s    │" << std::endl;
            std::cout << "├───────────────────────────┼────────────┼─────────┼─────────┼────────┼────────┼──────────┤" << std::endl;
        }

        void printPrefillResult(const BenchConfig &cfg,
                                float median_ms, float min_ms, float max_ms,
                                double tflops, double gb_s, double tok_s)
        {
            std::stringstream config_str;
            config_str << "seq=" << std::setw(4) << cfg.seq_len
                       << " h=" << cfg.n_heads << "/" << cfg.n_kv_heads
                       << " d=" << cfg.head_dim;

            std::cout << "│ " << std::setw(25) << std::left << config_str.str()
                      << " │ " << std::setw(10) << std::right << std::fixed << std::setprecision(3) << median_ms
                      << " │ " << std::setw(7) << std::fixed << std::setprecision(3) << min_ms
                      << " │ " << std::setw(7) << std::fixed << std::setprecision(3) << max_ms
                      << " │ " << std::setw(6) << std::fixed << std::setprecision(2) << tflops
                      << " │ " << std::setw(6) << std::fixed << std::setprecision(0) << gb_s
                      << " │ " << std::setw(8) << std::fixed << std::setprecision(0) << tok_s
                      << " │" << std::endl;
        }

        void printDecodeHeader()
        {
            std::cout << "│ Config (KV cache len)     │ Median(μs) │ Min(μs) │ Max(μs) │ Tok/s      │" << std::endl;
            std::cout << "├───────────────────────────┼────────────┼─────────┼─────────┼────────────┤" << std::endl;
        }

        void printDecodeResult(const BenchConfig &cfg,
                               double median_us, double min_us, double max_us, double tok_s)
        {
            std::stringstream config_str;
            config_str << "kv=" << std::setw(5) << cfg.kv_len
                       << " h=" << cfg.n_heads << "/" << cfg.n_kv_heads
                       << " d=" << cfg.head_dim;

            std::cout << "│ " << std::setw(25) << std::left << config_str.str()
                      << " │ " << std::setw(10) << std::right << std::fixed << std::setprecision(1) << median_us
                      << " │ " << std::setw(7) << std::fixed << std::setprecision(1) << min_us
                      << " │ " << std::setw(7) << std::fixed << std::setprecision(1) << max_us
                      << " │ " << std::setw(10) << std::fixed << std::setprecision(0) << tok_s
                      << " │" << std::endl;
        }

        void printTableFooter()
        {
            std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════════╝" << std::endl;
        }

    protected:
        cudaDeviceProp device_props_;
        int compute_capability_;
        cudaStream_t stream_ = nullptr;
        cudaEvent_t start_event_ = nullptr;
        cudaEvent_t stop_event_ = nullptr;
        std::unique_ptr<IMPIContext> mpi_ctx_;
        std::unique_ptr<CUDAFlashAttentionKernelT<ActivationPrecision::FP32>> kernel_;
        std::unique_ptr<DeviceWorkspaceManager> workspace_;
    };

    // ============================================================================
    // Prefill Benchmarks
    // ============================================================================

    /**
     * @brief Sweep the real Qwen3.6-35B-A3B FP16-KV captured prefill regime.
     *
     * Diagonal points expose scaling as the physical query bucket grows.  The
     * long-context points hold that query bucket fixed while increasing the
     * resident prefix through 128K, which directly reveals whether the current
     * Q-tiled kernel leaves too few SMs active and warrants a sequence/context-
     * parallel mode.  Production never executes a monolithic 128K query matrix;
     * it replays at most 4096 query rows against the growing cache, so this
     * matrix follows the deployed execution contract rather than inventing a
     * quadratic surrogate.
     *
     * Set both `LLAMINAR_ATTN_PROFILE_M` and
     * `LLAMINAR_ATTN_PROFILE_KV_LEN` to select one listed point for NCU. In
     * profiler mode the graph is replayed once after calibration.
     */
    TEST_F(CUDAFlashAttentionPerf, Prefill_Qwen36MoE_CapturedFP16KVScaling)
    {
        const std::array<ProductionPrefillPoint, 25> points{{
            {64, 64, "bucket"},
            {128, 128, "bucket"},
            {256, 256, "bucket"},
            {384, 384, "bucket"},
            {512, 512, "dashboard bucket"},
            {1024, 1024, "bucket"},
            {2048, 2048, "bucket"},
            {4096, 4096, "maximum query bucket"},
            {64, 8192, "8K resident context"},
            {128, 8192, "8K resident context"},
            {512, 8192, "8K resident context"},
            {4096, 8192, "8K resident context"},
            {64, 32768, "32K resident context"},
            {128, 32768, "32K resident context"},
            {512, 32768, "32K resident context"},
            {4096, 32768, "32K resident context"},
            {16, 131072, "128K narrow-Q boundary"},
            {32, 131072, "128K narrow-Q boundary"},
            {48, 131072, "128K narrow-Q boundary"},
            {64, 131072, "128K resident context"},
            {80, 131072, "128K narrow-Q boundary"},
            {96, 131072, "128K narrow-Q boundary"},
            {128, 131072, "128K resident context"},
            {512, 131072, "128K resident context"},
            {4096, 131072, "128K resident context"},
        }};

        const int requested_m =
            positiveEnvironmentValue("LLAMINAR_ATTN_PROFILE_M");
        const int requested_kv =
            positiveEnvironmentValue("LLAMINAR_ATTN_PROFILE_KV_LEN");
        ASSERT_EQ(requested_m == 0, requested_kv == 0)
            << "Profiler selection requires both LLAMINAR_ATTN_PROFILE_M and "
               "LLAMINAR_ATTN_PROFILE_KV_LEN";
        const bool profiler_mode = requested_m > 0;

        CapturedProductionPrefill benchmark(
            kQwen36MoE35BAttention,
            QWEN36_MAX_PREFILL_ROWS,
            QWEN36_MAX_CONTEXT);
        if (!benchmark.ready())
        {
            if (benchmark.error() == "No CUDA device")
                GTEST_SKIP() << benchmark.error();
            FAIL() << benchmark.error();
        }

        std::cout
            << "\nCaptured CUDA FA2 Qwen3.6-35B-A3B "
               "(16q/2kv, d=256, FP16 KV)\n"
            << "M,kv_len,latency_us,query_tok_s,effective_tflops,"
               "grid_blocks,block_threads,waves_per_gpu,label\n";

        bool measured_requested_point = false;
        for (const ProductionPrefillPoint &point : points)
        {
            if (profiler_mode &&
                (point.query_rows != requested_m || point.kv_len != requested_kv))
            {
                continue;
            }

            const ProductionPrefillMeasurement measurement =
                benchmark.measure(point, profiler_mode);
            ASSERT_GT(measurement.latency_us, 0.0)
                << "M=" << point.query_rows << " kv_len=" << point.kv_len
                << " error=" << benchmark.error();
            ASSERT_TRUE(std::isfinite(measurement.effective_tflops));
            ASSERT_GT(measurement.grid_blocks, 0);
            ASSERT_GT(measurement.block_threads, 0);

            std::cout << point.query_rows << ","
                      << point.kv_len << ","
                      << std::fixed << std::setprecision(3)
                      << measurement.latency_us << ","
                      << std::setprecision(1)
                      << measurement.query_tokens_per_second << ","
                      << std::setprecision(3)
                      << measurement.effective_tflops << ","
                      << measurement.grid_blocks << ","
                      << measurement.block_threads << ","
                      << std::setprecision(2)
                      << measurement.waves_per_gpu << ","
                      << point.label << "\n";
            measured_requested_point = true;
        }

        ASSERT_TRUE(measured_requested_point)
            << "Requested CUDA attention profiler point is not in the sweep";
        EXPECT_TRUE(benchmark.finalResultIsFinite());
    }

    /**
     * @brief Tournament query partitions across the released Qwen geometries.
     *
     * The matrix includes every distinct attention geometry from the compact
     * Qwen2.5 dense family, the Qwen3.5/3.6 dense family through 27B, and the
     * 35B/122B/397B MoE releases.  Releases that share physical geometry share
     * one row because production dispatch deliberately has no model-name key.
     *
     * Each legal compiled query grouping is replayed at M=64 and M=128 against
     * an 8K resident cache for single-device execution and every valid equal
     * tensor split at TP=2, TP=4, and TP=8.  KV heads are locally sharded when
     * divisible by the TP degree and explicitly replicated otherwise, matching
     * the two production attention-state policies.  The generic policy is then
     * replayed independently and must remain within five percent of the
     * observed winner.  This makes the geometry formula accountable to real
     * hardware while retaining total dispatch for unseen head counts, TP
     * shards, M values, and device sizes.
     */
    TEST_F(CUDAFlashAttentionPerf, Prefill_QwenCatalog_QueryPartitionTournament)
    {
        const std::array<ProductionPrefillPoint, 2> points{{
            {64, 8192, "M64"},
            {128, 8192, "M128"},
        }};
        constexpr std::array tp_degrees{1, 2, 4, 8};

        ScopedFA2QueryWarpGroupPolicy policy_scope;
        std::cout
            << "\nCaptured CUDA FA2 Qwen query-partition tournament "
               "(FP16 KV, 8K context)\n"
            << "release,tp,M,local_heads,local_kv_heads,kv_policy,head_dim,"
               "policy_groups,winner_groups,policy_us,winner_us,"
               "policy_regret_pct,candidate_us\n";

        for (const AttentionGeometry &geometry : kQwenAttentionGeometries)
        {
            for (const int tp_degree : tp_degrees)
            {
                // Uneven Q-head assignments are a separate placement policy;
                // this tournament exercises the equal tensor splits requested
                // by the production LocalTP/NodeTP graph contracts.
                if (geometry.n_heads % tp_degree != 0)
                    continue;

                AttentionGeometry local_geometry = geometry;
                local_geometry.n_heads = geometry.n_heads / tp_degree;
                const bool shard_kv_heads =
                    geometry.n_kv_heads % tp_degree == 0;
                if (shard_kv_heads)
                {
                    local_geometry.n_kv_heads =
                        geometry.n_kv_heads / tp_degree;
                    local_geometry.replicated_gqa_n_rep = 0;
                }
                else
                {
                    local_geometry.n_kv_heads = geometry.n_kv_heads;
                    local_geometry.replicated_gqa_n_rep =
                        geometry.n_heads / geometry.n_kv_heads;
                }

                CapturedProductionPrefill benchmark(
                    local_geometry,
                    /*maximum_query_rows=*/128,
                    /*maximum_context=*/8192);
                ASSERT_TRUE(benchmark.ready())
                    << geometry.release_label << " TP=" << tp_degree
                    << ": " << benchmark.error();

                const int maximum_groups =
                    llaminar2::cuda::fa2_policy::maximumFA2QueryWarpGroups(
                        geometry.head_dim);
                ASSERT_GT(maximum_groups, 0) << geometry.release_label;

                for (const ProductionPrefillPoint &point : points)
                {
                    std::array<double, 7> candidate_us{};
                    double winner_us = std::numeric_limits<double>::infinity();
                    int winner_groups = 0;

                    for (int groups = 1; groups <= maximum_groups; ++groups)
                    {
                        policy_scope.select(groups);
                        const ProductionPrefillMeasurement measurement =
                            benchmark.measure(point, /*profiler_mode=*/false);
                        ASSERT_GT(measurement.latency_us, 0.0)
                            << geometry.release_label << " TP=" << tp_degree
                            << " M=" << point.query_rows
                            << " groups=" << groups
                            << " error=" << benchmark.error();
                        candidate_us[groups] = measurement.latency_us;
                        if (measurement.latency_us < winner_us)
                        {
                            winner_us = measurement.latency_us;
                            winner_groups = groups;
                        }
                    }

                    const llaminar2::cuda::fa2_policy::FA2QueryPartitionGeometry
                        policy_geometry{
                            .batch_size = 1,
                            .query_rows = point.query_rows,
                            .local_query_heads = local_geometry.n_heads,
                            .head_dim = geometry.head_dim,
                            .sm_count = device_props_.multiProcessorCount,
                        };
                    const int policy_groups =
                        llaminar2::cuda::fa2_policy::selectFA2QueryWarpGroups(
                            policy_geometry);
                    ASSERT_GT(policy_groups, 0) << geometry.release_label;

                    policy_scope.select(/*query_warp_groups=*/0);
                    const ProductionPrefillMeasurement policy_measurement =
                        benchmark.measure(point, /*profiler_mode=*/false);
                    ASSERT_GT(policy_measurement.latency_us, 0.0)
                        << geometry.release_label << " TP=" << tp_degree
                        << " M=" << point.query_rows
                        << " error=" << benchmark.error();
                    const double regret_percent =
                        100.0 * (policy_measurement.latency_us / winner_us - 1.0);

                    std::ostringstream candidates;
                    for (int groups = 1; groups <= maximum_groups; ++groups)
                    {
                        if (groups > 1)
                            candidates << ';';
                        candidates << groups << ':' << std::fixed
                                   << std::setprecision(3) << candidate_us[groups];
                    }

                    std::cout << geometry.release_label << ','
                              << tp_degree << ','
                              << point.query_rows << ','
                              << local_geometry.n_heads << ','
                              << local_geometry.n_kv_heads << ','
                              << (shard_kv_heads ? "sharded" : "replicated") << ','
                              << geometry.head_dim << ','
                              << policy_groups << ','
                              << winner_groups << ','
                              << std::fixed << std::setprecision(3)
                              << policy_measurement.latency_us << ','
                              << winner_us << ','
                              << regret_percent << ','
                              << candidates.str() << '\n';

                    EXPECT_LE(regret_percent, 5.0)
                        << geometry.release_label << " TP=" << tp_degree
                        << " M=" << point.query_rows
                        << " policy_groups=" << policy_groups
                        << " winner_groups=" << winner_groups;
                }

                EXPECT_TRUE(benchmark.finalResultIsFinite())
                    << geometry.release_label << " TP=" << tp_degree
                    << ": " << benchmark.error();
            }
        }
    }

    /**
     * @brief Tournament every physical K/V tile over the Qwen/TP work surface.
     *
     * TILE_KV changes shared-memory residency, pipeline/barrier count, and the
     * amount of K/V work amortized per iteration. A shared-memory-only selector
     * cannot infer that tradeoff: a two-block tile has no occupancy advantage
     * when the complete query grid contains fewer blocks than the GPU has SMs.
     * This gate therefore measures all fitting 16/32/64-row specializations at
     * short diagonal buckets and an 8K resident context for every released Qwen
     * head geometry and every valid equal TP split through eight participants.
     * The 128K points determine whether the largest physical tile has a late
     * context crossover before it can be called dominated.
     *
     * The production policy is captured and replayed independently after the
     * candidates. It must identify one measured specialization and stay within
     * five percent of the observed winner. Exact output equivalence across all
     * physical tiles is owned by the CUDA FlashAttention parity integration
     * suite; this test owns economy and dispatch coverage.
     */
    TEST_F(CUDAFlashAttentionPerf, Prefill_QwenCatalog_PhysicalKVTileTournament)
    {
        const std::array<ProductionPrefillPoint, 8> points{{
            {64, 64, "short-M64"},
            {128, 128, "short-M128"},
            {512, 512, "short-M512"},
            {64, 8192, "context-M64"},
            {128, 8192, "context-M128"},
            {512, 8192, "context-M512"},
            {64, 131072, "long-context-M64"},
            {512, 131072, "long-context-M512"},
        }};
        constexpr std::array tp_degrees{1, 2, 4, 8};
        constexpr std::array physical_tiles{16, 32, 64};
        const std::size_t maximum_dynamic_shared_memory =
            std::max<std::size_t>(
                device_props_.sharedMemPerBlock,
                device_props_.sharedMemPerBlockOptin);

        ScopedFA2KVTilePolicy tile_scope;
        std::cout
            << "\nCaptured CUDA FA2 Qwen physical-KV-tile tournament "
               "(FP16 KV)\n"
            << "release,tp,M,kv_len,local_heads,local_kv_heads,kv_policy,"
               "head_dim,query_groups,policy_tile,winner_tile,policy_us,"
               "winner_us,policy_regret_pct,candidate_us\n";

        for (const AttentionGeometry &geometry : kQwenAttentionGeometries)
        {
            for (const int tp_degree : tp_degrees)
            {
                if (geometry.n_heads % tp_degree != 0)
                    continue;

                AttentionGeometry local_geometry = geometry;
                local_geometry.n_heads = geometry.n_heads / tp_degree;
                const bool shard_kv_heads =
                    geometry.n_kv_heads % tp_degree == 0;
                if (shard_kv_heads)
                {
                    local_geometry.n_kv_heads =
                        geometry.n_kv_heads / tp_degree;
                    local_geometry.replicated_gqa_n_rep = 0;
                }
                else
                {
                    local_geometry.n_kv_heads = geometry.n_kv_heads;
                    local_geometry.replicated_gqa_n_rep =
                        geometry.n_heads / geometry.n_kv_heads;
                }

                CapturedProductionPrefill benchmark(
                    local_geometry,
                    /*maximum_query_rows=*/512,
                    /*maximum_context=*/131072);
                ASSERT_TRUE(benchmark.ready())
                    << geometry.release_label << " TP=" << tp_degree
                    << ": " << benchmark.error();

                for (const ProductionPrefillPoint &point : points)
                {
                    const llaminar2::cuda::fa2_policy::FA2QueryPartitionGeometry
                        query_geometry{
                            .batch_size = 1,
                            .query_rows = point.query_rows,
                            .local_query_heads = local_geometry.n_heads,
                            .head_dim = geometry.head_dim,
                            .sm_count = device_props_.multiProcessorCount,
                        };
                    const int query_groups =
                        llaminar2::cuda::fa2_policy::selectFA2QueryWarpGroups(
                            query_geometry);
                    ASSERT_GT(query_groups, 0) << geometry.release_label;

                    std::array<double, physical_tiles.size()> candidate_us{};
                    double winner_us = std::numeric_limits<double>::infinity();
                    int winner_tile = 0;
                    for (std::size_t index = 0;
                         index < physical_tiles.size();
                         ++index)
                    {
                        const int tile_kv = physical_tiles[index];
                        const std::size_t shared_memory =
                            llaminar2::cuda::fa2_policy::
                                fa2DynamicSharedMemoryBytes(
                                    geometry.head_dim,
                                    query_groups,
                                    tile_kv);
                        if (shared_memory == 0 ||
                            shared_memory > maximum_dynamic_shared_memory)
                        {
                            continue;
                        }

                        tile_scope.select(tile_kv);
                        const ProductionPrefillMeasurement measurement =
                            benchmark.measure(point, /*profiler_mode=*/false);
                        ASSERT_GT(measurement.latency_us, 0.0)
                            << geometry.release_label << " TP=" << tp_degree
                            << " M=" << point.query_rows
                            << " KV=" << point.kv_len
                            << " tile_kv=" << tile_kv
                            << " error=" << benchmark.error();
                        ASSERT_EQ(measurement.tile_kv, tile_kv)
                            << geometry.release_label << " TP=" << tp_degree;
                        candidate_us[index] = measurement.latency_us;
                        if (measurement.latency_us < winner_us)
                        {
                            winner_us = measurement.latency_us;
                            winner_tile = tile_kv;
                        }
                    }
                    ASSERT_GT(winner_tile, 0) << geometry.release_label;

                    tile_scope.select(/*generic policy=*/0);
                    const ProductionPrefillMeasurement policy_measurement =
                        benchmark.measure(point, /*profiler_mode=*/false);
                    ASSERT_GT(policy_measurement.latency_us, 0.0)
                        << geometry.release_label << " TP=" << tp_degree
                        << " M=" << point.query_rows
                        << " KV=" << point.kv_len
                        << " error=" << benchmark.error();
                    ASSERT_NE(
                        std::find(
                            physical_tiles.begin(),
                            physical_tiles.end(),
                            policy_measurement.tile_kv),
                        physical_tiles.end());
                    const double regret_percent =
                        100.0 *
                        (policy_measurement.latency_us / winner_us - 1.0);

                    std::ostringstream candidates;
                    for (std::size_t index = 0;
                         index < physical_tiles.size();
                         ++index)
                    {
                        if (index > 0)
                            candidates << ';';
                        candidates << physical_tiles[index] << ':';
                        if (candidate_us[index] > 0.0)
                        {
                            candidates << std::fixed << std::setprecision(3)
                                       << candidate_us[index];
                        }
                        else
                        {
                            candidates << "unsupported";
                        }
                    }

                    std::cout << geometry.release_label << ','
                              << tp_degree << ','
                              << point.query_rows << ','
                              << point.kv_len << ','
                              << local_geometry.n_heads << ','
                              << local_geometry.n_kv_heads << ','
                              << (shard_kv_heads ? "sharded" : "replicated")
                              << ',' << geometry.head_dim << ','
                              << query_groups << ','
                              << policy_measurement.tile_kv << ','
                              << winner_tile << ','
                              << std::fixed << std::setprecision(3)
                              << policy_measurement.latency_us << ','
                              << winner_us << ','
                              << regret_percent << ','
                              << candidates.str() << '\n';

                    EXPECT_LE(regret_percent, 5.0)
                        << geometry.release_label << " TP=" << tp_degree
                        << " M=" << point.query_rows
                        << " KV=" << point.kv_len
                        << " policy_tile=" << policy_measurement.tile_kv
                        << " winner_tile=" << winner_tile;
                }

                EXPECT_TRUE(benchmark.finalResultIsFinite())
                    << geometry.release_label << " TP=" << tp_degree
                    << ": " << benchmark.error();
            }
        }
    }

    /**
     * @brief Measure genuine K/V context parallelism for an underfilled TP shard.
     *
     * Qwen3.6-35B-A3B TP8 exposes two local query heads and replicated K/V.  At
     * small M, ordinary query tiling leaves most GA102 SMs idle while every CTA
     * scans a long resident cache.  The context transaction splits that scan
     * into fixed canonical partitions, writes device-resident summaries, and
     * merges them in one deterministic device kernel.
     *
     * The ordered-node reference and context-grid candidate execute the same
     * phase-one specialization and reducer.  Byte equality is mandatory at
     * every point; timings establish whether the added context residency repays
     * the summary traffic and reducer launch.  No economy threshold is installed
     * until the tournament identifies a stable crossover.
     */
    TEST_F(
        CUDAFlashAttentionPerf,
        Prefill_Qwen36TP8_ContextParallelTransactionHypothesis)
    {
        constexpr AttentionGeometry kQwen36TP8Geometry{
            .release_label = "Qwen3.6-35B-A3B TP8",
            .n_heads = 2,
            .n_kv_heads = 2,
            .head_dim = 256,
            .head_start = 0,
            .replicated_gqa_n_rep = 8,
        };
        constexpr int kContextPartitionSize =
            llaminar2::cuda::fa2_policy::
                kFA2CanonicalContextPartitionKeys;
        constexpr std::array<ProductionPrefillPoint, 10> points{{
            {1, 131072, "M1-KV128K"},
            {2, 131072, "M2-KV128K"},
            {4, 131072, "M4-KV128K"},
            {8, 131072, "M8-KV128K"},
            {16, 8192, "M16-KV8K"},
            {16, 32768, "M16-KV32K"},
            {16, 131072, "M16-KV128K"},
            {64, 8192, "M64-KV8K"},
            {64, 32768, "M64-KV32K"},
            {64, 131072, "M64-KV128K"},
        }};

        const int requested_m =
            positiveEnvironmentValue("LLAMINAR_FA2_CONTEXT_PROFILE_M");
        const int requested_kv =
            positiveEnvironmentValue("LLAMINAR_FA2_CONTEXT_PROFILE_KV");
        const int requested_reducer_warps =
            positiveEnvironmentValue(
                "LLAMINAR_FA2_CONTEXT_PROFILE_REDUCER_WARPS");
        const bool profiler_mode =
            requested_m > 0 || requested_kv > 0 ||
            requested_reducer_warps > 0;
        ASSERT_EQ(requested_m > 0, requested_kv > 0)
            << "Set both LLAMINAR_FA2_CONTEXT_PROFILE_M and "
               "LLAMINAR_FA2_CONTEXT_PROFILE_KV";
        ASSERT_TRUE(
            requested_reducer_warps == 0 ||
            requested_reducer_warps == 1 ||
            requested_reducer_warps == 2 ||
            requested_reducer_warps == 4 ||
            requested_reducer_warps == 8)
            << "LLAMINAR_FA2_CONTEXT_PROFILE_REDUCER_WARPS must be "
               "1, 2, 4, or 8";
        ASSERT_FALSE(requested_reducer_warps > 0 && requested_m == 0)
            << "Reducer profiling requires an exact M and KV point";
        CapturedContextPartitionPrefill benchmark(
            kQwen36TP8Geometry,
            /*maximum_query_rows=*/64,
            /*maximum_context=*/131072,
            kContextPartitionSize,
            /*include_sequence_reference=*/!profiler_mode);
        ASSERT_TRUE(benchmark.ready()) << benchmark.error();

        std::cout
            << "\nCaptured CUDA FA2 fixed-context transaction hypothesis "
               "(Qwen3.6-35B-A3B TP8, FP16 KV)\n"
            << "M,kv_len,partition_size,live_partitions,captured_partitions,"
               "context_partition_slots,reducer_dimension_warps,"
               "direct_us,sequence_us,context_us,"
               "direct_over_context,sequence_over_context,direct_nodes,"
               "sequence_nodes,context_nodes,label\n";

        bool measured_requested_point = false;
        for (const ProductionPrefillPoint &point : points)
        {
            if (profiler_mode &&
                (point.query_rows != requested_m ||
                 point.kv_len != requested_kv))
            {
                continue;
            }

            const int maximum_reducer_warps =
                llaminar2::cuda::fa2_policy::
                    maximumFA2ReducerDimensionWarps(
                        kQwen36TP8Geometry.head_dim);
            for (int reducer_warps = 1;
                 reducer_warps <= maximum_reducer_warps;
                 reducer_warps *= 2)
            {
                if (profiler_mode && requested_reducer_warps > 0 &&
                    reducer_warps != requested_reducer_warps)
                {
                    continue;
                }

                const ContextPartitionMeasurement measurement =
                    benchmark.measure(
                        point,
                        profiler_mode,
                        reducer_warps);
                ASSERT_GT(measurement.direct_latency_us, 0.0)
                    << point.label << ": " << benchmark.error();
                if (!profiler_mode)
                {
                    ASSERT_GT(measurement.sequence_latency_us, 0.0)
                        << point.label << ": " << benchmark.error();
                }
                ASSERT_GT(measurement.context_latency_us, 0.0)
                    << point.label << ": " << benchmark.error();
                ASSERT_EQ(
                    measurement.live_context_partitions,
                    (point.kv_len + kContextPartitionSize - 1) /
                        kContextPartitionSize);
                ASSERT_EQ(
                    measurement.captured_context_partitions,
                    (QWEN36_MAX_CONTEXT + kContextPartitionSize - 1) /
                        kContextPartitionSize);
                ASSERT_EQ(
                    measurement.reducer_dimension_warps,
                    reducer_warps);
                ASSERT_TRUE(benchmark.canonicalOutputsAreByteEqual())
                    << point.label << ": " << benchmark.error();

                std::cout << point.query_rows << ','
                          << point.kv_len << ','
                          << kContextPartitionSize << ','
                          << measurement.live_context_partitions << ','
                          << measurement.captured_context_partitions << ','
                          << measurement.context_partition_slots << ','
                          << measurement.reducer_dimension_warps << ','
                          << std::fixed << std::setprecision(3)
                          << measurement.direct_latency_us << ','
                          << measurement.sequence_latency_us << ','
                          << measurement.context_latency_us << ','
                          << measurement.direct_latency_us /
                                 measurement.context_latency_us
                          << ','
                          << measurement.sequence_latency_us /
                                 measurement.context_latency_us
                          << ','
                          << measurement.direct_graph_nodes << ','
                          << measurement.sequence_graph_nodes << ','
                          << measurement.context_graph_nodes << ','
                          << point.label << '\n';
                measured_requested_point = true;
            }
        }

        ASSERT_TRUE(measured_requested_point)
            << "Requested context-parallel profiler point is not in the matrix";
    }

    /**
     * @brief Tune persistent context slots where query and context modes cross.
     *
     * The complete Qwen policy tournament showed that every query-sequence
     * winner had only one or two persistent context slots.  This focused gate
     * distinguishes an under-filled phase-one grid from an inherent reducer
     * crossover before production grows a device-side mode switch.  It uses
     * representative HD128 and HD256 geometries, including the largest released
     * GQA ratio, at both a short live prefix and the complete 128K envelope.
     *
     * Each candidate is a complete two-node captured transaction.  The timing
     * loop contains no allocation, transfer, synchronization, or host callback;
     * post-measurement materialization proves that changing physical slot count
     * leaves the canonical-partition arithmetic byte identical to direct FA2.
     */
    TEST_F(
        CUDAFlashAttentionPerf,
        Prefill_ContextParallelPersistentSlotTournament)
    {
        constexpr int kContextPartitionSize =
            llaminar2::cuda::fa2_policy::
                kFA2CanonicalContextPartitionKeys;
        constexpr int kCapturedKVCapacity = 131072;
        constexpr std::array<int, 6> live_kv_lengths{{
            256, 512, 1024, 4096, 8192, 131072}};
        constexpr std::array<int, 8> slot_candidates{{1, 2, 4, 8, 12, 16, 24, 32}};
        constexpr std::array<AttentionGeometry, 3> geometries{{
            {"HD128-H28-GQA7", 28, 4, 128},
            {"HD256-H16-GQA8", 16, 2, 256},
            {"HD256-H32-GQA16", 32, 2, 256},
        }};

        std::cout
            << "\nCaptured CUDA FA2 persistent context-slot tournament "
               "(FP16 KV, fixed 128K capacity)\n"
            << "geometry,M,live_kv,slots,reducer_dimension_warps,"
               "direct_us,context_us,direct_over_context,byte_equal\n";

        for (const AttentionGeometry &geometry : geometries)
        {
            CapturedContextPartitionPrefill benchmark(
                geometry,
                /*maximum_query_rows=*/128,
                /*maximum_context=*/kCapturedKVCapacity,
                kContextPartitionSize,
                /*include_sequence_reference=*/false);
            ASSERT_TRUE(benchmark.ready())
                << geometry.release_label << ": " << benchmark.error();

            for (const int live_kv_len : live_kv_lengths)
            {
                const ProductionPrefillPoint point{
                    .query_rows = 128,
                    .kv_len = live_kv_len,
                    .label = geometry.release_label,
                };
                for (const int slots : slot_candidates)
                {
                    const ContextPartitionMeasurement measurement =
                        benchmark.measure(
                            point,
                            /*profiler_mode=*/false,
                            /*production reducer policy=*/0,
                            slots);
                    ASSERT_GT(measurement.direct_latency_us, 0.0)
                        << geometry.release_label << " KV=" << live_kv_len
                        << " slots=" << slots << ": " << benchmark.error();
                    ASSERT_GT(measurement.context_latency_us, 0.0)
                        << geometry.release_label << " KV=" << live_kv_len
                        << " slots=" << slots << ": " << benchmark.error();
                    ASSERT_EQ(measurement.context_partition_slots, slots);
                    const bool byte_equal =
                        benchmark.canonicalOutputsAreByteEqual();
                    ASSERT_TRUE(byte_equal)
                        << geometry.release_label << " KV=" << live_kv_len
                        << " slots=" << slots << ": " << benchmark.error();

                    std::cout
                        << geometry.release_label << ",128," << live_kv_len
                        << ',' << slots << ','
                        << measurement.reducer_dimension_warps << ','
                        << std::fixed << std::setprecision(3)
                        << measurement.direct_latency_us << ','
                        << measurement.context_latency_us << ','
                        << measurement.direct_latency_us /
                               measurement.context_latency_us
                        << ',' << (byte_equal ? "true" : "false") << '\n';
                }
            }
        }
    }

    /**
     * @brief Compare device-direct windows without changing captured capacity.
     *
     * Every positive limit captures the same three-node root topology: a fused
     * predicate producer, a guarded lean query root, and an IF-only context
     * transaction. Changing the limit moves only device-owned output ownership;
     * it never changes graph structure. This tournament isolates the resulting
     * fixed dispatch cost from context arithmetic at the first four canonical
     * prefixes for representative HD128 and HD256 crossover geometries.
     */
    TEST_F(
        CUDAFlashAttentionPerf,
        Prefill_DeviceAdaptiveDirectPartitionLimitTournament)
    {
        struct AdaptiveGeometry
        {
            AttentionGeometry geometry;
            int context_partition_slots;
        };

        constexpr int kContextPartitionSize =
            llaminar2::cuda::fa2_policy::
                kFA2CanonicalContextPartitionKeys;
        constexpr int kCapturedKVCapacity = 131072;
        constexpr std::array<int, 4> live_kv_lengths{{256, 512, 1024, 2048}};
        constexpr std::array<int, 4> direct_limits{{1, 2, 4, 8}};
        constexpr std::array<AdaptiveGeometry, 2> geometries{{
            {{"HD128-H28-GQA7", 28, 4, 128}, 24},
            {{"HD256-H16-GQA8", 16, 2, 256}, 21},
        }};

        std::cout
            << "\nCaptured CUDA FA2 device-direct partition-limit tournament "
               "(M=128, FP16 KV, fixed 128K capacity)\n"
            << "geometry,live_kv,direct_limit,graph_nodes,direct_us,"
               "adaptive_us,regret_pct,byte_equal\n";

        for (const AdaptiveGeometry &candidate : geometries)
        {
            CapturedContextPartitionPrefill benchmark(
                candidate.geometry,
                /*maximum_query_rows=*/128,
                /*maximum_context=*/kCapturedKVCapacity,
                kContextPartitionSize,
                /*include_sequence_reference=*/false);
            ASSERT_TRUE(benchmark.ready())
                << candidate.geometry.release_label << ": "
                << benchmark.error();

            for (const int live_kv_len : live_kv_lengths)
            {
                const ProductionPrefillPoint point{
                    .query_rows = 128,
                    .kv_len = live_kv_len,
                    .label = candidate.geometry.release_label,
                };
                for (const int direct_limit : direct_limits)
                {
                    const ContextPartitionMeasurement measurement =
                        benchmark.measure(
                            point,
                            /*profiler_mode=*/false,
                            /*production reducer policy=*/0,
                            candidate.context_partition_slots,
                            direct_limit);
                    ASSERT_GT(measurement.direct_latency_us, 0.0)
                        << benchmark.error();
                    ASSERT_GT(measurement.context_latency_us, 0.0)
                        << benchmark.error();
                    ASSERT_EQ(measurement.context_graph_nodes, 3u);
                    const bool byte_equal =
                        benchmark.canonicalOutputsAreByteEqual();
                    ASSERT_TRUE(byte_equal)
                        << candidate.geometry.release_label
                        << " KV=" << live_kv_len
                        << " limit=" << direct_limit << ": "
                        << benchmark.error();

                    const double winner_us = std::min(
                        measurement.direct_latency_us,
                        measurement.context_latency_us);
                    const double regret_percent =
                        100.0 *
                        (measurement.context_latency_us / winner_us - 1.0);
                    std::cout
                        << candidate.geometry.release_label << ','
                        << live_kv_len << ',' << direct_limit << ','
                        << measurement.context_graph_nodes << ','
                        << std::fixed << std::setprecision(3)
                        << measurement.direct_latency_us << ','
                        << measurement.context_latency_us << ','
                        << regret_percent << ','
                        << (byte_equal ? "true" : "false") << '\n';
                }
            }
        }
    }

    /**
     * @brief Certify adaptive transaction economy across the live-prefix range.
     *
     * Slot tuning at 8K and 128K identifies the throughput plateau, but one
     * fixed captured graph must also replay efficiently immediately after cache
     * reset. These candidates are the geometry-derived results of a 32-SM-wave
     * budget. The production direct-prefix selector is passed to the graph, so
     * each replay measures the actual device branch rather than a host-selected
     * surrogate. The sweep starts at one canonical partition and doubles
     * through the first production-scale prefix before jumping to 128K.
     */
    TEST_F(
        CUDAFlashAttentionPerf,
        Prefill_DeviceAdaptiveTunedSlotsLivePrefixCrossover)
    {
        struct TunedSlotGeometry
        {
            AttentionGeometry geometry;
            int context_partition_slots;
        };

        constexpr int kContextPartitionSize =
            llaminar2::cuda::fa2_policy::
                kFA2CanonicalContextPartitionKeys;
        constexpr int kCapturedKVCapacity = 131072;
        constexpr std::array<int, 7> live_kv_lengths{{
            256, 512, 1024, 2048, 4096, 8192, 131072}};
        constexpr std::array<TunedSlotGeometry, 5> geometries{{
            {{"HD64-H14-GQA7", 14, 2, 64}, 47},
            {{"HD64-H7-GQA7", 7, 1, 64}, 47},
            {{"HD128-H28-GQA7", 28, 4, 128}, 24},
            {{"HD256-H16-GQA8", 16, 2, 256}, 21},
            {{"HD256-H32-GQA16", 32, 2, 256}, 11},
        }};

        std::cout
            << "\nCaptured CUDA FA2 device-adaptive live-prefix crossover "
               "(M=128, FP16 KV, fixed 128K capacity)\n"
            << "geometry,live_kv,live_partitions,slots,direct_limit,direct_us,"
               "adaptive_us,direct_over_adaptive,regret_pct,byte_equal\n";

        for (const TunedSlotGeometry &candidate : geometries)
        {
            CapturedContextPartitionPrefill benchmark(
                candidate.geometry,
                /*maximum_query_rows=*/128,
                /*maximum_context=*/kCapturedKVCapacity,
                kContextPartitionSize,
                /*include_sequence_reference=*/false);
            ASSERT_TRUE(benchmark.ready())
                << candidate.geometry.release_label << ": "
                << benchmark.error();

            const auto adaptive_plan =
                llaminar2::cuda::fa2_policy::selectFA2PrefillParallelPlan({
                    .batch_size = 1,
                    .query_rows = 128,
                    .local_query_heads = candidate.geometry.n_heads,
                    .head_dim = candidate.geometry.head_dim,
                    .kv_capacity = kCapturedKVCapacity,
                    .sm_count = device_props_.multiProcessorCount,
                    .requested_axis =
                        attention::AttentionPrefillParallelAxis::
                            GeometrySelected,
                });
            ASSERT_TRUE(adaptive_plan.valid);
            ASSERT_TRUE(adaptive_plan.usesDeviceAdaptiveParallelism());
            ASSERT_EQ(
                adaptive_plan.context_partition_slots,
                candidate.context_partition_slots);

            for (const int live_kv_len : live_kv_lengths)
            {
                const ProductionPrefillPoint point{
                    .query_rows = 128,
                    .kv_len = live_kv_len,
                    .label = candidate.geometry.release_label,
                };
                const ContextPartitionMeasurement measurement =
                    benchmark.measure(
                        point,
                        /*profiler_mode=*/false,
                        /*production reducer policy=*/0,
                        candidate.context_partition_slots,
                        adaptive_plan.device_direct_partition_limit);
                ASSERT_GT(measurement.direct_latency_us, 0.0)
                    << candidate.geometry.release_label
                    << " KV=" << live_kv_len << ": " << benchmark.error();
                ASSERT_GT(measurement.context_latency_us, 0.0)
                    << candidate.geometry.release_label
                    << " KV=" << live_kv_len << ": " << benchmark.error();
                const bool byte_equal =
                    benchmark.canonicalOutputsAreByteEqual();
                ASSERT_TRUE(byte_equal)
                    << candidate.geometry.release_label
                    << " KV=" << live_kv_len << ": " << benchmark.error();

                const double winner_us = std::min(
                    measurement.direct_latency_us,
                    measurement.context_latency_us);
                const double regret_percent =
                    100.0 * (measurement.context_latency_us / winner_us - 1.0);
                EXPECT_LE(
                    measurement.context_latency_us,
                    fa2AdaptiveEconomyCeilingUs(winner_us))
                    << candidate.geometry.release_label
                    << " KV=" << live_kv_len
                    << " regret_percent=" << regret_percent;

                std::cout
                    << candidate.geometry.release_label << ',' << live_kv_len
                    << ',' << measurement.live_context_partitions << ','
                    << measurement.context_partition_slots << ','
                    << measurement.device_direct_partition_limit << ','
                    << std::fixed << std::setprecision(3)
                    << measurement.direct_latency_us << ','
                    << measurement.context_latency_us << ','
                    << measurement.direct_latency_us /
                           measurement.context_latency_us
                    << ',' << regret_percent
                    << ',' << (byte_equal ? "true" : "false") << '\n';
            }
        }
    }

    /**
     * @brief Certify query/context policy economy over all released Qwen heads.
     *
     * This is the production-envelope tournament. Every context graph captures
     * the complete 128K cache stride while `AttentionDeviceParams` supplies
     * live prefixes from one canonical partition through the complete envelope.
     * The physical graph therefore remains unchanged across prefix restore and
     * subsequent cache advancement.
     *
     * Every distinct released Qwen head geometry is measured at each valid
     * equal TP degree through eight participants. M=17 is the first prefill row
     * count not owned by grouped decode; M=32/64/128 cross the compiled query
     * tile boundaries. Explicit query and context schedules establish the
     * hardware winner, while the generic geometry policy must remain within
     * five percent. Randomized byte-totality remains in the CUDA integration
     * sweep; the non-uniform persistent patterns here catch topology-sensitive
     * arithmetic drift without contaminating timed replay.
     */
    TEST_F(
        CUDAFlashAttentionPerf,
        Prefill_QwenCatalog_ContextParallelPolicyTournament)
    {
        constexpr int kContextPartitionSize =
            llaminar2::cuda::fa2_policy::
                kFA2CanonicalContextPartitionKeys;
        constexpr int kCapturedKVCapacity = 131072;
        constexpr std::array<int, 4> query_rows{{17, 32, 64, 128}};
        constexpr std::array<int, 5> live_kv_lengths{{
            256, 512, 1024, 8192, 131072}};
        constexpr std::array<int, 4> tp_degrees{{1, 2, 4, 8}};

        int query_policy_selections = 0;
        int context_policy_selections = 0;
        int measured_domains = 0;

        std::cout
            << "\nCaptured CUDA FA2 Qwen query/context policy tournament "
               "(FP16 KV, fixed 128K capacity)\n"
            << "release,tp,M,live_kv,capacity,local_heads,local_kv_heads,"
               "kv_policy,head_dim,live_partitions,captured_partitions,"
               "context_partition_slots,device_direct_partition_limit,"
               "reducer_dimension_warps,policy_mode,"
               "winner_mode,policy_us,"
               "winner_us,policy_regret_pct,direct_us,context_us\n";

        for (const AttentionGeometry &model : kQwenAttentionGeometries)
        {
            for (const int tp_degree : tp_degrees)
            {
                const TensorParallelAttentionGeometry participant =
                    resolveTensorParallelAttentionGeometry(model, tp_degree);
                if (!participant.valid)
                    continue;

                CapturedContextPartitionPrefill benchmark(
                    participant.local,
                    /*maximum_query_rows=*/query_rows.back(),
                    /*maximum_context=*/kCapturedKVCapacity,
                    kContextPartitionSize,
                    /*include_sequence_reference=*/false);
                ASSERT_TRUE(benchmark.ready())
                    << model.release_label << " TP=" << tp_degree
                    << ": " << benchmark.error();

                for (const int m : query_rows)
                {
                    const llaminar2::cuda::fa2_policy::
                        FA2PrefillParallelPlan policy_plan =
                            llaminar2::cuda::fa2_policy::
                                selectFA2PrefillParallelPlan({
                                    .batch_size = 1,
                                    .query_rows = m,
                                    .local_query_heads =
                                        participant.local.n_heads,
                                    .head_dim = participant.local.head_dim,
                                    .kv_capacity = kCapturedKVCapacity,
                                    .sm_count =
                                        device_props_.multiProcessorCount,
                                    .requested_axis =
                                        attention::
                                            AttentionPrefillParallelAxis::
                                                GeometrySelected,
                                });
                    ASSERT_TRUE(policy_plan.valid)
                        << model.release_label << " TP=" << tp_degree
                        << " M=" << m;

                    if (policy_plan.usesContextParallelism())
                        ++context_policy_selections;
                    else
                        ++query_policy_selections;

                    for (const int live_kv_len : live_kv_lengths)
                    {
                        const std::string point_label =
                            "M" + std::to_string(m) + "-KV" +
                            std::to_string(live_kv_len);
                        const ProductionPrefillPoint point{
                            .query_rows = m,
                            .kv_len = live_kv_len,
                            .label = point_label.c_str(),
                        };
                        const ContextPartitionMeasurement measurement =
                            benchmark.measure(
                                point,
                                /*profiler_mode=*/false,
                                /*production reducer policy=*/0,
                                /*production slot policy=*/0,
                                policy_plan.device_direct_partition_limit);

                        ASSERT_GT(measurement.direct_latency_us, 0.0)
                            << model.release_label << " TP=" << tp_degree
                            << " " << point_label << ": "
                            << benchmark.error();
                        ASSERT_GT(measurement.context_latency_us, 0.0)
                            << model.release_label << " TP=" << tp_degree
                            << " " << point_label << ": "
                            << benchmark.error();
                        ASSERT_EQ(measurement.sequence_latency_us, 0.0);
                        ASSERT_EQ(measurement.sequence_graph_nodes, 0u);
                        ASSERT_EQ(measurement.direct_graph_nodes, 2u);
                        ASSERT_EQ(measurement.context_graph_nodes, 3u);
                        ASSERT_EQ(
                            measurement.live_context_partitions,
                            (live_kv_len + kContextPartitionSize - 1) /
                                kContextPartitionSize);
                        ASSERT_EQ(
                            measurement.captured_context_partitions,
                            policy_plan.max_context_partitions == 0
                                ? (kCapturedKVCapacity +
                                   kContextPartitionSize - 1) /
                                      kContextPartitionSize
                                : policy_plan.max_context_partitions);
                        ASSERT_EQ(
                            measurement.context_partition_slots,
                            llaminar2::cuda::fa2_policy::
                                selectFA2ContextPartitionSlots(
                                    {
                                        .batch_size = 1,
                                        .query_rows = m,
                                        .local_query_heads =
                                            participant.local.n_heads,
                                        .head_dim = participant.local.head_dim,
                                        .sm_count =
                                            device_props_.multiProcessorCount,
                                    },
                                    measurement.captured_context_partitions));
                        ASSERT_EQ(
                            measurement.device_direct_partition_limit,
                            policy_plan.device_direct_partition_limit);
                        ASSERT_EQ(
                            measurement.reducer_dimension_warps,
                            llaminar2::cuda::fa2_policy::
                                selectFA2ReducerDimensionWarps({
                                    .batch_size = 1,
                                    .query_rows = m,
                                    .local_query_heads =
                                        participant.local.n_heads,
                                    .head_dim = participant.local.head_dim,
                                    .sm_count =
                                        device_props_.multiProcessorCount,
                                }));
                        ASSERT_TRUE(benchmark.canonicalOutputsAreByteEqual())
                            << model.release_label << " TP=" << tp_degree
                            << " " << point_label << ": "
                            << benchmark.error();

                        const bool context_won =
                            measurement.context_latency_us <
                            measurement.direct_latency_us;
                        const double winner_us = std::min(
                            measurement.direct_latency_us,
                            measurement.context_latency_us);
                        const double policy_us =
                            policy_plan.usesContextParallelism()
                                ? measurement.context_latency_us
                                : measurement.direct_latency_us;
                        const double regret_percent =
                            100.0 * (policy_us / winner_us - 1.0);

                        std::cout
                            << model.release_label << ','
                            << tp_degree << ','
                            << m << ','
                            << live_kv_len << ','
                            << kCapturedKVCapacity << ','
                            << participant.local.n_heads << ','
                            << participant.local.n_kv_heads << ','
                            << kvHeadPlacementName(
                                   participant.kv_placement)
                            << ',' << participant.local.head_dim << ','
                            << measurement.live_context_partitions << ','
                            << measurement.captured_context_partitions << ','
                            << measurement.context_partition_slots << ','
                            << measurement.device_direct_partition_limit << ','
                            << measurement.reducer_dimension_warps << ','
                            << llaminar2::cuda::fa2_policy::
                                   fa2PrefillPhysicalModeName(
                                       policy_plan.mode)
                            << ','
                            << (context_won
                                    ? "key_value_context"
                                    : "query_sequence")
                            << ',' << std::fixed << std::setprecision(3)
                            << policy_us << ','
                            << winner_us << ','
                            << regret_percent << ','
                            << measurement.direct_latency_us << ','
                            << measurement.context_latency_us << '\n';

                        const double economy_ceiling_us =
                            policy_plan.usesDeviceAdaptiveParallelism()
                                ? fa2AdaptiveEconomyCeilingUs(winner_us)
                                : winner_us *
                                      (1.0 + kFA2PolicyRelativeRegret);
                        EXPECT_LE(policy_us, economy_ceiling_us)
                            << model.release_label << " TP=" << tp_degree
                            << " M=" << m << " live_kv=" << live_kv_len
                            << " regret_percent=" << regret_percent
                            << " policy="
                            << llaminar2::cuda::fa2_policy::
                                   fa2PrefillPhysicalModeName(
                                       policy_plan.mode)
                            << " winner="
                            << (context_won
                                    ? "key_value_context"
                                    : "query_sequence");
                        ++measured_domains;
                    }
                }
            }
        }

        EXPECT_GT(measured_domains, 0);
        EXPECT_GT(query_policy_selections, 0)
            << "Generic policy must exercise query-sequence execution";
        EXPECT_GT(context_policy_selections, 0)
            << "Generic policy must exercise K/V-context execution";
    }

    TEST_F(CUDAFlashAttentionPerf, Prefill_SmallSeqLen_Qwen05B)
    {
        printTableHeader("Flash Attention Prefill - Small Sequences (Qwen 0.5B config)");
        printPrefillHeader();

        // Qwen 0.5B: 14 heads, 2 KV heads, head_dim=64
        std::vector<BenchConfig> configs = {
            {128, 128, 14, 2, 64, "Short prompt"},
            {256, 256, 14, 2, 64, "Medium prompt"},
            {512, 512, 14, 2, 64, "Long prompt"},
        };

        for (const auto &cfg : configs)
        {
            runPrefillBenchmark(cfg);
        }

        printTableFooter();
    }

    TEST_F(CUDAFlashAttentionPerf, Prefill_MediumSeqLen_Qwen7B)
    {
        printTableHeader("Flash Attention Prefill - Medium Sequences (Qwen 7B config)");
        printPrefillHeader();

        // Qwen 7B: 32 heads, 8 KV heads (GQA 4:1), head_dim=128
        std::vector<BenchConfig> configs = {
            {512, 512, 32, 8, 128, "Medium prompt"},
            {1024, 1024, 32, 8, 128, "Long prompt"},
            {2048, 2048, 32, 8, 128, "Very long prompt"},
        };

        for (const auto &cfg : configs)
        {
            runPrefillBenchmark(cfg);
        }

        printTableFooter();
    }

    TEST_F(CUDAFlashAttentionPerf, Prefill_LongSeqLen_Qwen72B)
    {
        printTableHeader("Flash Attention Prefill - Long Sequences (Qwen 72B config)");
        printPrefillHeader();

        // Qwen 72B: 64 heads, 8 KV heads (GQA 8:1), head_dim=128
        std::vector<BenchConfig> configs = {
            {2048, 2048, 64, 8, 128, "Long context"},
            {4096, 4096, 64, 8, 128, "Very long context"},
            {8192, 8192, 64, 8, 128, "Maximum context"},
        };

        for (const auto &cfg : configs)
        {
            runPrefillBenchmark(cfg);
        }

        printTableFooter();
    }

    TEST_F(CUDAFlashAttentionPerf, Prefill_HeadDimComparison)
    {
        printTableHeader("Flash Attention Prefill - Head Dimension Comparison");
        printPrefillHeader();

        // Compare head_dim=64 vs head_dim=128
        std::vector<BenchConfig> configs = {
            {1024, 1024, 32, 8, 64, "head_dim=64"},
            {1024, 1024, 32, 8, 128, "head_dim=128"},
            {2048, 2048, 32, 8, 64, "head_dim=64 long"},
            {2048, 2048, 32, 8, 128, "head_dim=128 long"},
        };

        for (const auto &cfg : configs)
        {
            runPrefillBenchmark(cfg);
        }

        printTableFooter();
    }

    // ============================================================================
    // Decode Benchmarks (Latency-focused)
    // ============================================================================

    TEST_F(CUDAFlashAttentionPerf, Decode_VaryingKVCacheLen)
    {
        printTableHeader("Flash Decoding - Varying KV Cache Length (Qwen 0.5B)");
        printDecodeHeader();

        // Qwen 0.5B: 14 heads, 2 KV heads, head_dim=64
        std::vector<BenchConfig> configs = {
            {1, 64, 14, 2, 64, "Early decode"},
            {1, 256, 14, 2, 64, "Short context"},
            {1, 1024, 14, 2, 64, "Medium context"},
            {1, 4096, 14, 2, 64, "Long context"},
            {1, 8192, 14, 2, 64, "Maximum context"},
        };

        for (const auto &cfg : configs)
        {
            runDecodeBenchmark(cfg);
        }

        printTableFooter();
    }

    TEST_F(CUDAFlashAttentionPerf, Decode_GQAComparison)
    {
        printTableHeader("Flash Decoding - GQA Ratio Comparison (kv_len=2048)");
        printDecodeHeader();

        // Compare different GQA ratios
        std::vector<BenchConfig> configs = {
            {1, 2048, 32, 32, 128, "MHA 1:1"},
            {1, 2048, 32, 8, 128, "GQA 4:1"},
            {1, 2048, 64, 8, 128, "GQA 8:1"},
            {1, 2048, 32, 4, 128, "GQA 8:1 alt"},
        };

        for (const auto &cfg : configs)
        {
            runDecodeBenchmark(cfg);
        }

        printTableFooter();
    }

    TEST_F(CUDAFlashAttentionPerf, Decode_LargeModel_Qwen72B)
    {
        printTableHeader("Flash Decoding - Large Model (Qwen 72B)");
        printDecodeHeader();

        // Qwen 72B: 64 heads, 8 KV heads, head_dim=128
        std::vector<BenchConfig> configs = {
            {1, 512, 64, 8, 128, "Short generation"},
            {1, 2048, 64, 8, 128, "Medium generation"},
            {1, 8192, 64, 8, 128, "Long generation"},
            {1, 16384, 64, 8, 128, "Very long generation"},
        };

        for (const auto &cfg : configs)
        {
            runDecodeBenchmark(cfg);
        }

        printTableFooter();
    }

    // ============================================================================
    // Roofline Analysis
    // ============================================================================

    TEST_F(CUDAFlashAttentionPerf, RooflineAnalysis)
    {
        printTableHeader("Flash Attention Roofline Analysis");

        // Get SM count and theoretical peak
        int sm_count = device_props_.multiProcessorCount;
        // FP32 throughput per SM (typically 128 FP32 ops per clock per SM on Ampere)
        // Clock rate is in MHz, so we need to convert
        double base_clock_ghz = 1.5;              // Typical sustained clock for modern GPUs
        double fp32_ops_per_sm_per_cycle = 128.0; // Ampere has 128 FP32 cores per SM
        double theoretical_tflops = sm_count * fp32_ops_per_sm_per_cycle * base_clock_ghz / 1000.0;

        // Memory bandwidth (convert from MHz to GB/s)
        // memoryBusWidth is in bits, memoryClockRate is in kHz (need to check units)
        double memory_bw_gb_s = 936.0; // RTX 3090: ~936 GB/s

        std::cout << "│ Theoretical Performance (estimated for Ampere):                                             │" << std::endl;
        std::cout << "│   SM count: " << std::setw(80) << std::left
                  << std::to_string(sm_count) << "│" << std::endl;
        std::cout << "│   FP32 Compute: ~" << std::setw(75) << std::left
                  << (std::to_string(static_cast<int>(theoretical_tflops)) + " TFLOPS (rough estimate)") << "│" << std::endl;
        std::cout << "│   Memory BW: ~" << std::setw(77) << std::left
                  << (std::to_string(static_cast<int>(memory_bw_gb_s)) + " GB/s (RTX 3090 spec)") << "│" << std::endl;
        std::cout << "├──────────────────────────────────────────────────────────────────────────────────────────────┤" << std::endl;

        printPrefillHeader();

        // Representative config for roofline
        BenchConfig cfg = {2048, 2048, 32, 8, 128, "Roofline test"};
        runPrefillBenchmark(cfg);

        std::cout << "├──────────────────────────────────────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│ Arithmetic Intensity: " << std::fixed << std::setprecision(2)
                  << std::setw(69) << std::left
                  << (std::to_string((cfg.computeTFLOPs() * 1e12) / (cfg.computeMemoryGB() * 1e9)) + " FLOPS/byte")
                  << "│" << std::endl;

        printTableFooter();
    }

} // anonymous namespace

#else // !HAVE_CUDA

TEST(CUDAFlashAttentionPerf, NoCUDA)
{
    GTEST_SKIP() << "CUDA not available - skipping Flash Attention performance tests";
}

#endif // HAVE_CUDA
