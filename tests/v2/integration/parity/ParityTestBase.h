/**
 * @file ParityTestBase.h
 * @brief Base class and utilities for PyTorch parity tests
 *
 * Provides standardized infrastructure for comparing Llaminar inference
 * against PyTorch ground truth. All parity tests should inherit from
 * ParityTestBase to ensure consistent:
 *
 * - Metric calculations (cosine similarity, KL divergence, Top-K overlap)
 * - Table visualization of layer-by-layer results
 * - Pass/fail assertions with configurable thresholds
 * - Snapshot loading and regeneration
 *
 * MPI Support:
 *   For tensor-parallel tests, set mpi_ctx_ in SetUp() and override
 *   getDeviceForRank() instead of getDevice(). The base class handles:
 *   - Printing only on rank 0
 *   - Snapshot regeneration only on rank 0 (with barrier)
 *   - MPI barriers at key synchronization points
 *
 * Usage (single-rank):
 *   class Test__MyCUDAParity : public ParityTestBase {
 *   protected:
 *       void SetUp() override {
 *           config_.cosine_threshold = 0.99f;
 *           config_.early_layers_count = 6;
 *           ParityTestBase::SetUp();  // Regenerates snapshots
 *       }
 *
 *       DeviceId getDevice() override { return gpu_device_; }
 *       std::string getBackendName() override { return "CUDA"; }
 *   };
 *
 * Usage (tensor-parallel MPI):
 *   class Test__MyTPParity : public ParityTestBase {
 *   protected:
 *       void SetUp() override {
 *           mpi_ctx_ = std::make_shared<MPIContext>();
 *           config_.cosine_threshold = 0.94f;  // Relaxed for TP
 *           ParityTestBase::SetUp();
 *       }
 *       DeviceId getDeviceForRank() override {
 *           return (mpi_ctx_->rank() == 0) ? DeviceId::cuda(0) : DeviceId::rocm(0);
 *       }
 *       std::string getBackendName() override { return "TensorParallel"; }
 *   };
 *
 * @author David Sanftenberg
 * @date 2026-01-11
 */

#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <iomanip>
#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <cctype>

// libfort for formatted table output
#include "fort.hpp"

#include "loaders/ModelContext.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "execution/debug/TPSnapshot.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"

// Pipeline parallelism support
#include "config/PipelineConfig.h"
#include "collective/BackendRouter.h"
#include "collective/BackendRouter.h"

// Modern orchestration runner support (for incremental migration)
#include "utils/TestOrchestrationHelper.h"
#include "execution/runner/IOrchestrationRunner.h"
#include "kernels/KernelFactory.h"
#include "backends/BackendManager.h"
#include "backends/GlobalDeviceAddress.h"
#ifdef HAVE_CUDA
#include "kernels/cuda/ops/CUDAEmbeddingKernelT.h"
#include <cuda_runtime.h>
// Deterministic mode: disables FP32 atomicAdd in split-K/stream-K GEMM
extern "C" void cudaNativeVNNIPrefill_setDeterministicMode(bool enabled);
#endif
#ifdef HAVE_ROCM
#include "kernels/rocm/ops/ROCmEmbeddingKernelT.h"
#endif
#include "utils/Logger.h"
#include "utils/MPIContext.h"
#include "backends/DeviceId.h"
#include "backends/ComputeBackend.h"

// NumPy .npy file loading
#include <cnpy.h>

// MPI for tensor-parallel tests
#include <mpi.h>

// For snapshot cache mutex
#include <mutex>

// For CSV results directory creation
#include <filesystem>
#include <cstdlib>

namespace llaminar2::test::parity
{

    // =============================================================================
    // Configuration
    // =============================================================================

    /**
     * @brief Configuration for parity test thresholds
     *
     * Different backends (CPU, CUDA, ROCm) may need different thresholds
     * due to varying quantization schemes and numerical precision.
     */
    struct ParityConfig
    {
        // Model and test setup — MUST be set by subclass (no defaults)
        std::string model_path;
        std::string snapshot_dir;
        std::string prompt;
        std::vector<int> token_ids;
        int decode_steps = 5;

        // Layer-by-layer thresholds
        float cosine_threshold = 0.99f;  ///< Minimum avg cosine similarity for layer pass
        bool use_avg_cosine = true;      ///< Use avg (true) or min (false) cosine for pass criteria
        int early_layers_count = 6;      ///< Number of early layers to enforce threshold on
        int min_early_layers_passed = 6; ///< Minimum early layers that must pass

        // LM_HEAD thresholds
        float kl_threshold = 0.15f;      ///< Maximum KL divergence for logits
        float min_top1_accuracy = 60.0f; ///< Minimum Top-1 accuracy percentage
        float min_top5_accuracy = 80.0f; ///< Minimum Top-5 accuracy percentage
        int pytorch_top1_in_topk = 3;    ///< PyTorch's top-1 must be in llaminar's top-K (0=disabled)

        // Decode thresholds (for incremental decode tests)
        float decode_cosine_threshold = 0.99f;
        float min_decode_pass_rate = 0.8f; ///< Minimum fraction of decode steps that must pass

        /// Stages to exclude from per-layer parity comparison.
        /// Used for GLOBAL scope TP where column-parallel stages (Q/K/V projections)
        /// produce partial outputs that can't be directly compared to full PyTorch outputs.
        std::vector<std::string> excluded_stages;
    };

    // =============================================================================
    // Result Structures
    // =============================================================================

    /**
     * @brief Distribution statistics for a single tensor
     */
    struct TensorDistributionStats
    {
        float min = 0.0f;
        float max = 0.0f;
        float mean = 0.0f;
        float stddev = 0.0f;
        float kurtosis = 0.0f;
        float skewness = 0.0f; ///< Asymmetry; symmetric quant schemes assume ~0
        float p95 = 0.0f;
        float p99 = 0.0f;
        float outlier_fraction = 0.0f; ///< Fraction of |x| > 6σ (SmoothQuant indicator)
        float dynamic_range = 0.0f;    ///< log2(max|x| / median|x|) — bits needed
        float sparsity = 0.0f;         ///< Fraction of |x| < 1e-6
        float zero_fraction = 0.0f;    ///< Fraction of x == 0 exactly
        size_t nan_count = 0;          ///< Number of NaN values
        size_t inf_count = 0;          ///< Number of ±Inf values
        size_t element_count = 0;
    };

    /**
     * @brief Result of comparing a single tensor/stage
     */
    struct StageComparisonResult
    {
        std::string stage_name;
        bool passed = false;
        float cosine_similarity = 0.0f;
        float cosine_drop = 0.0f; ///< Drop from previous stage (positive = error introduced)
        float rel_l2_norm = 0.0f;
        float max_abs_diff = 0.0f;
        float kl_divergence = 0.0f;
        float snr_db = 0.0f;        ///< Signal-to-noise ratio in dB: 10·log10(‖signal‖²/‖error‖²)
        float rmse = 0.0f;          ///< Root mean squared error
        float error_entropy = 0.0f; ///< Shannon entropy of error histogram (bits)
        size_t total_elements = 0;
        TensorDistributionStats llaminar_stats;
        TensorDistributionStats pytorch_stats;
    };

    /**
     * @brief Aggregated statistics for a single layer
     */
    struct LayerStats
    {
        int layer_idx = 0;
        float avg_cosine_sim = 0.0f;
        float min_cosine_sim = 1.0f;
        std::string worst_stage;
        float max_cosine_drop = 0.0f; ///< Largest single-stage cosine drop (error introduced)
        std::string max_drop_stage;   ///< Stage that introduced the most error
        int stages_compared = 0;
        bool passed = false;
        float max_kurtosis = 0.0f;                        ///< Highest excess kurtosis across stages
        std::string max_kurtosis_stage;                   ///< Stage with highest kurtosis
        std::vector<StageComparisonResult> stage_results; ///< Per-stage detailed results
    };

    /**
     * @brief Summary of parity test results
     */
    struct ParityTestSummary
    {
        // Embedding
        float embedding_cosine = 0.0f;
        bool embedding_passed = false;

        // Per-layer stats
        std::vector<LayerStats> layer_stats;

        // LM_HEAD
        float lm_head_cosine = 0.0f;
        float lm_head_kl = 0.0f;
        float lm_head_top1 = 0.0f;
        float lm_head_top5 = 0.0f;
        bool lm_head_pytorch_top1_in_top3 = false; ///< PyTorch's top-1 is in llaminar's top-3
        bool lm_head_passed = false;

        // Overall
        int early_layers_passed = 0;
        int total_layers_passed = 0;
        bool overall_passed = false;
    };

    /**
     * @brief Statistics for a single decode step
     */
    struct DecodeStepStats
    {
        int step_idx = 0;
        float cosine_similarity = 0.0f;
        float kl_divergence = 0.0f;
        float top1_overlap = 0.0f;
        float top5_overlap = 0.0f;
        int llaminar_token = -1;
        int pytorch_token = -1;
        bool token_match = false;
        bool top5_match = false; ///< True if PyTorch top-1 appears in Llaminar top-5
        bool top3_match = false; ///< True if PyTorch top-1 appears in Llaminar top-3
        bool passed = false;
        std::vector<LayerStats> layer_stats; ///< Per-layer cosine similarity for this decode step
    };

    /**
     * @brief Summary of incremental decode parity results
     */
    struct DecodeParitySummary
    {
        std::vector<DecodeStepStats> step_stats;
        int steps_passed = 0;
        int steps_total = 0;
        int top1_matches = 0;
        int top3_matches = 0;
        int top5_matches = 0;
        float avg_cosine = 0.0f;
        float avg_kl = 0.0f;
        float top1_accuracy = 0.0f;
        float top3_accuracy = 0.0f;
        float top5_accuracy = 0.0f;
        bool overall_passed = false;
    };

    // =============================================================================
    // TP-Aware Result Structures
    // =============================================================================

    /**
     * @brief Per-device comparison result for tensor-parallel parity
     */
    struct TPDeviceComparisonResult
    {
        std::string device_id;          ///< Device identifier (e.g., "rank0_cuda0")
        int device_index = 0;           ///< Index in TP group
        float cosine_similarity = 0.0f; ///< Cosine vs corresponding PyTorch slice
        size_t slice_start = 0;         ///< Start column in PyTorch reference
        size_t slice_size = 0;          ///< Number of elements compared
        bool passed = false;
    };

    /**
     * @brief TP-aware comparison result for a single stage
     */
    struct TPStageComparisonResult
    {
        std::string stage_name;
        SnapshotShardingMode sharding_mode = SnapshotShardingMode::UNKNOWN;

        // Per-device comparisons (for column-parallel stages)
        std::vector<TPDeviceComparisonResult> device_results;

        // Combined result (concatenated partial outputs vs full PyTorch)
        float combined_cosine = 0.0f;
        size_t combined_elements = 0;
        bool combined_passed = false;

        // Overall for this stage
        bool passed = false;
    };

    /**
     * @brief TP-aware layer statistics
     */
    struct TPLayerStats
    {
        int layer_idx = 0;
        int tp_degree = 1;

        // Per-stage results
        std::vector<TPStageComparisonResult> stage_results;

        // Aggregated metrics
        float avg_combined_cosine = 0.0f;
        float min_combined_cosine = 1.0f;
        std::string worst_stage;
        int stages_compared = 0;
        bool passed = false;
    };

    /**
     * @brief TP-aware parity test summary
     */
    struct TPParityTestSummary
    {
        int tp_degree = 1;
        std::vector<std::string> device_names; ///< Device IDs in TP group

        // Embedding
        TPStageComparisonResult embedding_result;

        // Per-layer stats
        std::vector<TPLayerStats> layer_stats;

        // LM_HEAD (always gathered, so single combined result)
        float lm_head_cosine = 0.0f;
        float lm_head_kl = 0.0f;
        float lm_head_top1 = 0.0f;
        float lm_head_top5 = 0.0f;
        bool lm_head_pytorch_top1_in_top3 = false;
        bool lm_head_passed = false;

        // Overall
        int early_layers_passed = 0;
        int total_layers_passed = 0;
        bool overall_passed = false;
    };

    // =============================================================================
    // Device and Parallelism Configuration (Model-Agnostic)
    // =============================================================================

    /**
     * @brief Device type for parity tests
     * Note: Named ParityDeviceType to avoid collision with llaminar2::DeviceType
     */
    enum class ParityDeviceType
    {
        CPU,
        CUDA,
        ROCm
    };

    /**
     * @brief Parallelism strategy for multi-device tests
     */
    enum class Parallelism
    {
        None,        ///< Single device, no parallelism
        LocalTP,     ///< Local Tensor Parallelism (multi-device, single process)
        LocalPP,     ///< Local Pipeline Parallelism (multi-device, single process)
        NodeLocalTP, ///< Node-Local Tensor Parallelism (multi-rank MPI, same node)
        NodeLocalPP, ///< Node-Local Pipeline Parallelism (multi-rank MPI, same node)
        GlobalTP,    ///< Global Tensor Parallelism (multi-rank MPI, cross-node)
    };

    /**
     * @brief Collective backend for parallel tests
     */
    enum class Collective
    {
        None,          ///< No collective needed (single device)
        HOST,          ///< Host-based collective (staging through CPU)
        NCCL,          ///< NVIDIA NCCL (CUDA-CUDA)
        RCCL,          ///< AMD RCCL (ROCm-ROCm)
        HETEROGENEOUS, ///< Cross-vendor heterogeneous (CUDA-ROCm)
        MPI,           ///< MPI backend (for global TP, cross-rank)
    };

    // =============================================================================
    // Device Configuration Utilities
    // =============================================================================

    inline DeviceId toDeviceId(ParityDeviceType type, int index = 0)
    {
        switch (type)
        {
        case ParityDeviceType::CPU:
            return DeviceId::cpu();
        case ParityDeviceType::CUDA:
            return DeviceId::cuda(index);
        case ParityDeviceType::ROCm:
            return DeviceId::rocm(index);
        }
        return DeviceId::cpu();
    }

    inline GlobalDeviceAddress toGlobalAddress(ParityDeviceType type, int index = 0)
    {
        switch (type)
        {
        case ParityDeviceType::CPU:
            return GlobalDeviceAddress::cpu();
        case ParityDeviceType::CUDA:
            return GlobalDeviceAddress::cuda(index);
        case ParityDeviceType::ROCm:
            return GlobalDeviceAddress::rocm(index);
        }
        return GlobalDeviceAddress::cpu();
    }

    inline CollectiveBackendType toCollectiveBackend(Collective c)
    {
        switch (c)
        {
        case Collective::None:
            return CollectiveBackendType::HOST;
        case Collective::HOST:
            return CollectiveBackendType::HOST;
        case Collective::NCCL:
            return CollectiveBackendType::NCCL;
        case Collective::RCCL:
            return CollectiveBackendType::RCCL;
        case Collective::HETEROGENEOUS:
            return CollectiveBackendType::HETEROGENEOUS;
        case Collective::MPI:
            return CollectiveBackendType::MPI;
        }
        return CollectiveBackendType::HOST;
    }

    inline std::string deviceTypeName(ParityDeviceType type)
    {
        switch (type)
        {
        case ParityDeviceType::CPU:
            return "CPU";
        case ParityDeviceType::CUDA:
            return "CUDA";
        case ParityDeviceType::ROCm:
            return "ROCm";
        }
        return "Unknown";
    }

    inline std::string parallelismName(Parallelism p)
    {
        switch (p)
        {
        case Parallelism::None:
            return "None";
        case Parallelism::LocalTP:
            return "LocalTP";
        case Parallelism::LocalPP:
            return "LocalPP";
        case Parallelism::NodeLocalTP:
            return "NodeLocalTP";
        case Parallelism::NodeLocalPP:
            return "NodeLocalPP";
        case Parallelism::GlobalTP:
            return "GlobalTP";
        }
        return "Unknown";
    }

    inline std::string collectiveName(Collective c)
    {
        switch (c)
        {
        case Collective::None:
            return "None";
        case Collective::HOST:
            return "Host";
        case Collective::NCCL:
            return "NCCL";
        case Collective::RCCL:
            return "RCCL";
        case Collective::HETEROGENEOUS:
            return "HETEROGENEOUS";
        case Collective::MPI:
            return "MPI";
        }
        return "Unknown";
    }

    // =============================================================================
    // Hardware Detection Utilities
    // =============================================================================

    inline bool isMpiInitialized()
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        return initialized != 0;
    }

    inline int getCudaDeviceCount()
    {
#ifdef HAVE_CUDA
        if (auto *backend = getCUDABackend())
            return backend->deviceCount();
#endif
        return 0;
    }

    inline int getRocmDeviceCount()
    {
#ifdef HAVE_ROCM
        if (auto *backend = getROCmBackend())
            return backend->deviceCount();
#endif
        return 0;
    }

    // =============================================================================
    // Backend Thresholds for Parity Tests
    // =============================================================================

    /**
     * @brief Backend-specific parity thresholds
     *
     * Different backends may have different numerical precision characteristics.
     * These thresholds define what constitutes "passing" parity for each backend.
     */
    struct BackendThresholds
    {
        float cosine_threshold = 0.999f;               ///< Min avg cosine for layer pass
        float decode_cosine_threshold = 0.99f;         ///< Threshold for decode parity
        int early_layers_count = 4;                    ///< Number of early layers to check
        int min_early_layers_passed = 4;               ///< Min early layers that must pass
        float kl_threshold = 0.05f;                    ///< Max KL divergence for logits
        std::vector<std::string> excluded_stages = {}; ///< Stages to exclude from parity comparison
        float min_top1_accuracy = 80.0f;               ///< Min Top-1 accuracy %
        float min_top5_accuracy = 80.0f;               ///< Min Top-5 accuracy %
        float min_decode_pass_rate = 0.8f;             ///< Min fraction of decode steps passing
        int pytorch_top1_in_topk = 3;                  ///< PyTorch's top-1 must be in llaminar's top-K (0=disabled)
    };

    // =============================================================================
    // Test Configuration (Model-Agnostic)
    // =============================================================================

    /**
     * @brief Complete declarative test configuration
     *
     * The devices vector is the single source of truth for device configuration.
     * Use the helper methods device_count() and primary_device() to derive values.
     */
    struct TestConfig
    {
        std::string name;                      ///< Human-readable test name
        std::vector<ParityDeviceType> devices; ///< Device list (heterogeneous supported)
        Parallelism parallelism;               ///< Parallelism strategy

        /// Collective backend for TP modes (LocalTP, GlobalTP, NodeLocalTP).
        /// PP modes should leave this as None — PP transfers are auto-selected
        /// by LocalPPContext based on device vendor types. For hybrid PP+TP,
        /// use tp_collective instead.
        Collective collective = Collective::None;

        BackendThresholds thresholds; ///< Parity thresholds
        std::string skip_reason;      ///< If non-empty, test will skip with this message
        int mpi_ranks = 1;            ///< Required MPI ranks for GlobalTP tests

        /// Model path override. If non-empty, overrides ParityConfig::model_path.
        /// Use this to test different quantization formats (e.g., Q4_0 vs Q8_0)
        /// which exercise different GEMM code paths (native-VNNI vs INT8-VNNI).
        std::string model_path;

        /// Snapshot directory override. If non-empty, overrides ParityConfig::snapshot_dir.
        /// Must be unique per model to avoid snapshot collisions between quant formats.
        std::string snapshot_dir;

        /// PP stage device sizes for hybrid PP+TP configurations
        /// Example: {2, 1} means stage 0 has 2 devices (TP domain), stage 1 has 1 device
        /// Empty means one device per stage (pure PP) or all devices in one domain (pure TP)
        std::vector<int> pp_stage_sizes;

        /// TP backend for stages that are TP domains (only used when pp_stage_sizes has entries > 1)
        Collective tp_collective = Collective::None;

        /// Runner precision controls (defaults preserve current parity behavior)
        ActivationPrecision activation_precision = ActivationPrecision::FP32;
        KVCachePrecision kv_cache_precision = KVCachePrecision::AUTO;

        /// Decode steps override. 0 means use ParityConfig default (5).
        int decode_steps = 0;

        // Derived accessors
        size_t device_count() const { return devices.size(); }
        ParityDeviceType primary_device() const { return devices.empty() ? ParityDeviceType::CPU : devices[0]; }
        bool is_local_tp() const { return parallelism == Parallelism::LocalTP; }
        bool is_local_pp() const { return parallelism == Parallelism::LocalPP; }
        bool is_node_local_tp() const { return parallelism == Parallelism::NodeLocalTP; }
        bool is_node_local_pp() const { return parallelism == Parallelism::NodeLocalPP; }
        bool is_global_tp() const { return parallelism == Parallelism::GlobalTP; }
        /// Returns true for any cross-rank TP (NodeLocal or Global)
        bool is_cross_rank_tp() const { return is_node_local_tp() || is_global_tp(); }
        /// Returns true for any cross-rank PP
        bool is_cross_rank_pp() const { return is_node_local_pp(); }
        /// Returns true for any cross-rank parallelism (TP or PP)
        bool is_cross_rank() const { return is_cross_rank_tp() || is_cross_rank_pp(); }
        bool is_single_device() const { return parallelism == Parallelism::None && devices.size() == 1; }
        bool should_skip() const { return !skip_reason.empty(); }

        /// Check if this is a hybrid PP+TP configuration (PP with TP domains inside stages)
        bool is_hybrid_pp_tp() const
        {
            if (!is_local_pp() || pp_stage_sizes.empty())
                return false;
            // Hybrid if any stage has more than 1 device
            for (int size : pp_stage_sizes)
            {
                if (size > 1)
                    return true;
            }
            return false;
        }

        /// Get number of PP stages (uses pp_stage_sizes if set, otherwise device_count)
        size_t num_pp_stages() const
        {
            if (!pp_stage_sizes.empty())
                return pp_stage_sizes.size();
            return device_count();
        }
    };

    /**
     * @brief Check if a TestConfig can run on current hardware
     * @return std::nullopt if available, or a skip reason string
     */
    inline std::optional<std::string> checkHardwareAvailability(const TestConfig &cfg)
    {
        // Check explicit skip reason first
        if (cfg.should_skip())
            return cfg.skip_reason;

        // Check MPI initialization for LocalTP/LocalPP/NodeLocalTP/NodeLocalPP/GlobalTP tests
        if (cfg.is_local_tp() || cfg.is_local_pp() || cfg.is_cross_rank())
        {
            if (!isMpiInitialized())
                return "Test requires MPI (run with mpirun)";
        }

        // Check MPI world size for cross-rank tests
        if (cfg.is_cross_rank())
        {
            int world_size = 1;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            if (world_size < cfg.mpi_ranks)
            {
                return "Cross-rank test requires " + std::to_string(cfg.mpi_ranks) +
                       " MPI ranks (got " + std::to_string(world_size) + ")";
            }
        }

        int cuda_count = getCudaDeviceCount();
        int rocm_count = getRocmDeviceCount();

        int required_cuda = 0, required_rocm = 0;
        for (auto dt : cfg.devices)
        {
            if (dt == ParityDeviceType::CUDA)
                required_cuda++;
            if (dt == ParityDeviceType::ROCm)
                required_rocm++;
        }

        if (required_cuda > cuda_count)
            return "Need " + std::to_string(required_cuda) + " CUDA devices, found " + std::to_string(cuda_count);
        if (required_rocm > rocm_count)
            return "Need " + std::to_string(required_rocm) + " ROCm devices, found " + std::to_string(rocm_count);

        return std::nullopt; // Hardware is available
    }

    // =============================================================================
    // Metric Computation Functions
    // =============================================================================

    /**
     * @brief Compute cosine similarity between two vectors
     *
     * Cosine similarity measures directional alignment, ignoring magnitude.
     * Preferred for embedding comparisons because quantization noise affects
     * magnitude but preserves direction.
     *
     * @return Value in [-1, 1], where 1 = identical direction
     */
    /**
     * @brief Compute excess kurtosis of a float array.
     *
     * Excess kurtosis = E[(x-μ)^4] / σ^4 - 3.
     * Normal distribution has excess kurtosis = 0.
     * High kurtosis (>10) indicates heavy-tailed outliers that destroy
     * int8 quantization accuracy.
     */
    inline float computeKurtosis(const float *data, size_t size)
    {
        if (size < 4)
            return 0.0f;

        double sum = 0.0, sum2 = 0.0;
        for (size_t i = 0; i < size; ++i)
        {
            double v = static_cast<double>(data[i]);
            sum += v;
            sum2 += v * v;
        }
        double mean = sum / static_cast<double>(size);
        double var = sum2 / static_cast<double>(size) - mean * mean;
        if (var < 1e-30)
            return 0.0f;

        double sum4 = 0.0;
        for (size_t i = 0; i < size; ++i)
        {
            double d = static_cast<double>(data[i]) - mean;
            double d2 = d * d;
            sum4 += d2 * d2;
        }
        double m4 = sum4 / static_cast<double>(size);
        return static_cast<float>(m4 / (var * var) - 3.0);
    }

    /**
     * @brief Compute distribution statistics for a float tensor
     *
     * Computes min, max, mean, stddev, kurtosis, and percentiles (p95, p99).
     * Uses OpenMP for large tensors.
     */
    inline TensorDistributionStats computeDistributionStats(const float *data, size_t size)
    {
        TensorDistributionStats stats;
        stats.element_count = size;
        if (size == 0)
            return stats;

        // Pass 1: min, max, sum, sum2, nan/inf/zero/sparsity counts
        double sum = 0.0, sum2 = 0.0;
        float vmin = std::numeric_limits<float>::max();
        float vmax = std::numeric_limits<float>::lowest();
        size_t nan_count = 0, inf_count = 0, zero_count = 0, sparse_count = 0;
        constexpr float sparsity_eps = 1e-6f;

#pragma omp parallel for reduction(+ : sum, sum2, nan_count, inf_count, zero_count, sparse_count) \
    reduction(min : vmin) reduction(max : vmax) schedule(static) if (size > 8192)
        for (size_t i = 0; i < size; ++i)
        {
            float v = data[i];
            if (std::isnan(v))
            {
                nan_count++;
                continue;
            }
            if (std::isinf(v))
            {
                inf_count++;
                continue;
            }
            if (v < vmin)
                vmin = v;
            if (v > vmax)
                vmax = v;
            if (v == 0.0f)
                zero_count++;
            if (std::abs(v) < sparsity_eps)
                sparse_count++;
            double d = static_cast<double>(v);
            sum += d;
            sum2 += d * d;
        }
        stats.nan_count = nan_count;
        stats.inf_count = inf_count;
        stats.zero_fraction = static_cast<float>(zero_count) / static_cast<float>(size);
        stats.sparsity = static_cast<float>(sparse_count) / static_cast<float>(size);

        size_t finite_count = size - nan_count - inf_count;
        if (finite_count == 0)
            return stats;

        stats.min = vmin;
        stats.max = vmax;
        double mean = sum / static_cast<double>(finite_count);
        stats.mean = static_cast<float>(mean);
        double var = sum2 / static_cast<double>(finite_count) - mean * mean;
        double sd = std::sqrt(std::max(var, 0.0));
        stats.stddev = static_cast<float>(sd);

        // Pass 2: kurtosis, skewness, outliers
        if (var > 1e-30)
        {
            double sum3 = 0.0, sum4 = 0.0;
            size_t outlier_count = 0;
            double outlier_threshold = 6.0 * sd;
#pragma omp parallel for reduction(+ : sum3, sum4, outlier_count) schedule(static) if (size > 8192)
            for (size_t i = 0; i < size; ++i)
            {
                float v = data[i];
                if (std::isnan(v) || std::isinf(v))
                    continue;
                double d = static_cast<double>(v) - mean;
                double d2 = d * d;
                sum3 += d2 * d;
                sum4 += d2 * d2;
                if (std::abs(d) > outlier_threshold)
                    outlier_count++;
            }
            double n = static_cast<double>(finite_count);
            stats.kurtosis = static_cast<float>(sum4 / (n * var * var) - 3.0);
            stats.skewness = static_cast<float>(sum3 / (n * sd * sd * sd));
            stats.outlier_fraction = static_cast<float>(outlier_count) / static_cast<float>(finite_count);
        }

        // Percentiles via approximate approach: sample if very large
        if (finite_count >= 4)
        {
            // For large tensors, subsample to cap percentile cost
            constexpr size_t MAX_PERCENTILE_SAMPLES = 100000;
            std::vector<float> abs_vals;
            if (finite_count <= MAX_PERCENTILE_SAMPLES)
            {
                abs_vals.reserve(finite_count);
                for (size_t i = 0; i < size; ++i)
                {
                    float v = data[i];
                    if (!std::isnan(v) && !std::isinf(v))
                        abs_vals.push_back(std::abs(v));
                }
            }
            else
            {
                // Deterministic stride-based sampling
                abs_vals.reserve(MAX_PERCENTILE_SAMPLES);
                size_t stride = size / MAX_PERCENTILE_SAMPLES;
                for (size_t i = 0; i < size && abs_vals.size() < MAX_PERCENTILE_SAMPLES; i += stride)
                {
                    float v = data[i];
                    if (!std::isnan(v) && !std::isinf(v))
                        abs_vals.push_back(std::abs(v));
                }
            }

            size_t n = abs_vals.size();
            if (n >= 4)
            {
                size_t p50_idx = n / 2;
                size_t p95_idx = static_cast<size_t>(0.95 * (n - 1));
                size_t p99_idx = static_cast<size_t>(0.99 * (n - 1));

                std::nth_element(abs_vals.begin(), abs_vals.begin() + p99_idx, abs_vals.end());
                stats.p99 = abs_vals[p99_idx];

                std::nth_element(abs_vals.begin(), abs_vals.begin() + p95_idx, abs_vals.begin() + p99_idx);
                stats.p95 = abs_vals[p95_idx];

                std::nth_element(abs_vals.begin(), abs_vals.begin() + p50_idx, abs_vals.begin() + p95_idx);
                float median_abs = abs_vals[p50_idx];

                float max_abs = std::max(std::abs(vmin), std::abs(vmax));
                if (median_abs > 1e-10f && max_abs > 1e-10f)
                    stats.dynamic_range = std::log2(max_abs / median_abs);
            }
        }

        return stats;
    }

    inline float computeCosineSimilarity(const float *a, const float *b, size_t size)
    {
        double dot_product = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;

#pragma omp parallel for reduction(+ : dot_product, norm_a, norm_b) schedule(static) if (size > 8192)
        for (size_t i = 0; i < size; ++i)
        {
            dot_product += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }

        double denominator = std::sqrt(norm_a) * std::sqrt(norm_b);
        if (denominator < 1e-10)
        {
            return 0.0f;
        }

        return static_cast<float>(dot_product / denominator);
    }

    /**
     * @brief Compute KL divergence between probability distributions
     *
     * KL(P || Q) measures how P diverges from Q.
     * First applies softmax to convert logits to probabilities.
     *
     * @param actual_logits Llaminar logits (unnormalized)
     * @param expected_logits PyTorch logits (unnormalized)
     * @param size Total elements (seq_len * vocab_size)
     * @param vocab_size Vocabulary size for per-position softmax
     * @return Average KL divergence per position (in nats)
     */
    inline float computeKLDivergence(
        const float *actual_logits,
        const float *expected_logits,
        size_t size,
        size_t vocab_size)
    {
        size_t seq_len = size / vocab_size;
        double total_kl = 0.0;

#pragma omp parallel for reduction(+ : total_kl) schedule(static) if (seq_len > 1)
        for (size_t pos = 0; pos < seq_len; ++pos)
        {
            const float *actual_row = actual_logits + pos * vocab_size;
            const float *expected_row = expected_logits + pos * vocab_size;

            // Find max for numerical stability (log-sum-exp trick)
            float max_actual = actual_row[0];
            float max_expected = expected_row[0];
            for (size_t i = 1; i < vocab_size; ++i)
            {
                max_actual = std::max(max_actual, actual_row[i]);
                max_expected = std::max(max_expected, expected_row[i]);
            }

            // Compute softmax denominators
            double sum_exp_actual = 0.0;
            double sum_exp_expected = 0.0;
            for (size_t i = 0; i < vocab_size; ++i)
            {
                sum_exp_actual += std::exp(static_cast<double>(actual_row[i] - max_actual));
                sum_exp_expected += std::exp(static_cast<double>(expected_row[i] - max_expected));
            }
            double log_sum_actual = max_actual + std::log(sum_exp_actual);
            double log_sum_expected = max_expected + std::log(sum_exp_expected);

            // KL divergence: KL(expected || actual)
            double pos_kl = 0.0;
            for (size_t i = 0; i < vocab_size; ++i)
            {
                double log_p = expected_row[i] - log_sum_expected;
                double log_q = actual_row[i] - log_sum_actual;
                double p = std::exp(log_p);

                if (p > 1e-10)
                {
                    pos_kl += p * (log_p - log_q);
                }
            }
            total_kl += pos_kl;
        }

        return static_cast<float>(total_kl / seq_len);
    }

    /**
     * @brief Compute Top-K overlap between two sets of logits
     *
     * Checks if the top K tokens predicted by both models overlap.
     * This is a "smoke test" for decision quality.
     *
     * @return Overlap percentage in [0, 1]
     */
    inline float computeTopKOverlap(
        const float *actual_logits,
        const float *expected_logits,
        size_t size,
        size_t vocab_size,
        int k)
    {
        size_t seq_len = size / vocab_size;
        double total_overlap = 0.0;

#pragma omp parallel for reduction(+ : total_overlap) schedule(static) if (seq_len > 1)
        for (size_t pos = 0; pos < seq_len; ++pos)
        {
            const float *actual_row = actual_logits + pos * vocab_size;
            const float *expected_row = expected_logits + pos * vocab_size;

            // Use a min-heap of size K instead of allocating vocab_size pairs
            auto get_top_k = [&](const float *logits)
            {
                // Min-heap: smallest of top-K on top, so we can eject it when we find larger
                std::vector<std::pair<float, int>> heap;
                heap.reserve(k + 1);
                for (size_t i = 0; i < vocab_size; ++i)
                {
                    if (static_cast<int>(heap.size()) < k)
                    {
                        heap.push_back({logits[i], static_cast<int>(i)});
                        if (static_cast<int>(heap.size()) == k)
                            std::make_heap(heap.begin(), heap.end(),
                                           [](const auto &a, const auto &b)
                                           { return a.first > b.first; });
                    }
                    else if (logits[i] > heap.front().first)
                    {
                        std::pop_heap(heap.begin(), heap.end(),
                                      [](const auto &a, const auto &b)
                                      { return a.first > b.first; });
                        heap.back() = {logits[i], static_cast<int>(i)};
                        std::push_heap(heap.begin(), heap.end(),
                                       [](const auto &a, const auto &b)
                                       { return a.first > b.first; });
                    }
                }
                std::vector<int> indices(heap.size());
                for (size_t i = 0; i < heap.size(); ++i)
                    indices[i] = heap[i].second;
                std::sort(indices.begin(), indices.end());
                return indices;
            };

            auto actual_topk = get_top_k(actual_row);
            auto expected_topk = get_top_k(expected_row);

            std::vector<int> intersection;
            std::set_intersection(
                actual_topk.begin(), actual_topk.end(),
                expected_topk.begin(), expected_topk.end(),
                std::back_inserter(intersection));

            total_overlap += static_cast<double>(intersection.size()) / k;
        }

        return static_cast<float>(total_overlap / seq_len);
    }

    /**
     * @brief Check if PyTorch's top-1 token appears in llaminar's top-K
     *
     * For a single logit row: finds the argmax token in expected_logits (PyTorch)
     * and checks if that token is among the top-K tokens in actual_logits (llaminar).
     *
     * For multiple rows (seq_len > 1): returns the fraction of positions where
     * the check passes.
     *
     * This is a clearer gate than Top-K overlap: it answers "does llaminar consider
     * the correct answer to be a plausible choice?"
     *
     * @param actual_logits Llaminar logits
     * @param expected_logits PyTorch reference logits
     * @param size Total float count (seq_len * vocab_size)
     * @param vocab_size Vocabulary size
     * @param k Top-K to search in llaminar (e.g., 3)
     * @return Fraction of positions where PyTorch's top-1 is in llaminar's top-K [0, 1]
     */
    inline float pytorchTop1InLlaminarTopK(
        const float *actual_logits,
        const float *expected_logits,
        size_t size,
        size_t vocab_size,
        int k)
    {
        size_t seq_len = size / vocab_size;
        if (seq_len == 0 || vocab_size == 0)
            return 0.0f;

        int hits = 0;
#pragma omp parallel for reduction(+ : hits) schedule(static) if (seq_len > 1)
        for (size_t pos = 0; pos < seq_len; ++pos)
        {
            const float *actual_row = actual_logits + pos * vocab_size;
            const float *expected_row = expected_logits + pos * vocab_size;

            // Find PyTorch's argmax
            int pytorch_argmax = 0;
            float pytorch_max = expected_row[0];
            for (size_t i = 1; i < vocab_size; ++i)
            {
                if (expected_row[i] > pytorch_max)
                {
                    pytorch_max = expected_row[i];
                    pytorch_argmax = static_cast<int>(i);
                }
            }

            // Find llaminar's top-K tokens using a min-heap of size K
            std::vector<std::pair<float, int>> heap;
            heap.reserve(k + 1);
            for (size_t i = 0; i < vocab_size; ++i)
            {
                if (static_cast<int>(heap.size()) < k)
                {
                    heap.push_back({actual_row[i], static_cast<int>(i)});
                    if (static_cast<int>(heap.size()) == k)
                        std::make_heap(heap.begin(), heap.end(),
                                       [](const auto &a, const auto &b)
                                       { return a.first > b.first; });
                }
                else if (actual_row[i] > heap.front().first)
                {
                    std::pop_heap(heap.begin(), heap.end(),
                                  [](const auto &a, const auto &b)
                                  { return a.first > b.first; });
                    heap.back() = {actual_row[i], static_cast<int>(i)};
                    std::push_heap(heap.begin(), heap.end(),
                                   [](const auto &a, const auto &b)
                                   { return a.first > b.first; });
                }
            }

            // Check if PyTorch's argmax is in llaminar's top-K
            for (size_t i = 0; i < heap.size(); ++i)
            {
                if (heap[i].second == pytorch_argmax)
                {
                    hits++;
                    break;
                }
            }
        }

        return static_cast<float>(hits) / seq_len;
    }

    // =============================================================================
    // Table Rendering
    // =============================================================================

    /**
     * @brief Render a formatted parity results table to stdout using libfort
     *
     * Produces a consistent Unicode box-drawing table showing:
     * - Per-layer cosine similarity (avg and min)
     * - Worst stage per layer
     * - Pass/fail status with checkmarks
     * - LM_HEAD KL divergence and Top-K accuracy
     *
     * Uses libfort for clean, automatic column sizing and Unicode borders.
     */
    inline void renderParityTable(
        const ParityTestSummary &summary,
        const ParityConfig &config,
        const std::string &backend_name)
    {
        // Helper: format float to string with precision
        auto fmt_f6 = [](float v) -> std::string
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6) << v;
            return ss.str();
        };

        auto fmt_f4 = [](float v) -> std::string
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4) << v;
            return ss.str();
        };

        auto fmt_f1 = [](float v) -> std::string
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1) << v;
            return ss.str();
        };

        // Helper: status icon
        auto status_str = [](bool passed) -> std::string
        {
            return passed ? "✓" : "✗";
        };

        std::cout << "\n";

        // =========================================================================
        // Title table
        // =========================================================================
        {
            fort::utf8_table title_table;
            title_table.set_border_style(FT_DOUBLE2_STYLE);

            std::ostringstream title_ss;
            title_ss << backend_name << " vs PyTorch LAYER-BY-LAYER PARITY";

            std::ostringstream subtitle_ss;
            subtitle_ss << "Threshold: " << (config.use_avg_cosine ? "avg" : "min")
                        << " cosine similarity >= " << std::fixed << std::setprecision(3)
                        << config.cosine_threshold;

            title_table << title_ss.str() << fort::endr;
            title_table << subtitle_ss.str() << fort::endr;

            title_table[0][0].set_cell_text_align(fort::text_align::center);
            title_table[1][0].set_cell_text_align(fort::text_align::center);
            title_table.row(0).set_cell_row_type(fort::row_type::header);

            std::cout << title_table.to_string();
        }

        // =========================================================================
        // Main parity table
        // =========================================================================
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        // Header row
        table << fort::header
              << "Layer" << "Avg Cosine" << "Min Cosine" << "Worst Stage"
              << "Max Drop" << "Drop Stage" << "Kurtosis" << "OK"
              << fort::endr;

        // Set column alignments
        table.column(0).set_cell_text_align(fort::text_align::center);
        table.column(1).set_cell_text_align(fort::text_align::right);
        table.column(2).set_cell_text_align(fort::text_align::right);
        table.column(3).set_cell_text_align(fort::text_align::left);
        table.column(4).set_cell_text_align(fort::text_align::right);
        table.column(5).set_cell_text_align(fort::text_align::left);
        table.column(6).set_cell_text_align(fort::text_align::right);
        table.column(7).set_cell_text_align(fort::text_align::center);

        // Embedding row
        table << "EMBEDDING"
              << fmt_f6(summary.embedding_cosine)
              << fmt_f6(summary.embedding_cosine)
              << "-"
              << "-"
              << "-"
              << "-"
              << status_str(summary.embedding_passed)
              << fort::endr;

        // Per-layer rows
        for (const auto &stats : summary.layer_stats)
        {
            std::ostringstream layer_ss;
            layer_ss << "Layer " << stats.layer_idx;

            std::string kurtosis_str;
            if (stats.max_kurtosis > 0.0f)
            {
                std::ostringstream ks;
                ks << std::fixed << std::setprecision(0) << stats.max_kurtosis;
                kurtosis_str = ks.str();
            }
            else
            {
                kurtosis_str = "-";
            }

            std::string drop_str = (stats.max_cosine_drop > 0.001f)
                                       ? fmt_f6(stats.max_cosine_drop)
                                       : "-";
            std::string drop_stage = stats.max_drop_stage.empty() ? "-" : stats.max_drop_stage;

            table << layer_ss.str()
                  << fmt_f6(stats.avg_cosine_sim)
                  << fmt_f6(stats.min_cosine_sim)
                  << stats.worst_stage
                  << drop_str
                  << drop_stage
                  << kurtosis_str
                  << status_str(stats.passed)
                  << fort::endr;
        }

        // Separator before LM_HEAD
        table << fort::separator;

        // LM_HEAD row with extra info
        std::ostringstream lm_info;
        lm_info << "KL=" << fmt_f4(summary.lm_head_kl)
                << " Top1=" << fmt_f1(summary.lm_head_top1 * 100.0f) << "%";

        table << "LM_HEAD"
              << fmt_f6(summary.lm_head_cosine)
              << fmt_f6(summary.lm_head_cosine)
              << lm_info.str()
              << "-"
              << "-"
              << "-"
              << status_str(summary.lm_head_passed)
              << fort::endr;

        std::cout << table.to_string();

        // =========================================================================
        // Summary footer
        // =========================================================================
        std::cout << "\nLM_HEAD Top-5: " << std::fixed << std::setprecision(1)
                  << (summary.lm_head_top5 * 100.0f) << "%\n";
        std::cout << "Early layers passed: " << summary.early_layers_passed
                  << "/" << config.early_layers_count << "\n";
        std::cout << "LM_HEAD KL divergence: " << std::fixed << std::setprecision(4)
                  << summary.lm_head_kl << " (threshold: " << config.kl_threshold << ")\n";
    }

    /**
     * @brief Render a TP-aware parity results table to stdout using libfort
     *
     * For tensor-parallel tests, shows:
     * - Per-device cosine similarity against PyTorch slices
     * - Combined (concatenated) result vs full PyTorch
     * - Sharding mode for each stage
     *
     * Supports an arbitrary number of TP devices with dynamic column sizing.
     * Uses libfort for clean, Unicode box-drawing table formatting.
     */
    inline void renderTPParityTable(
        const TPParityTestSummary &summary,
        const ParityConfig &config,
        const std::string &backend_name)
    {
        const size_t num_devices = summary.device_names.size();

        // Helper: sharding mode to string
        auto mode_str = [](SnapshotShardingMode mode) -> std::string
        {
            switch (mode)
            {
            case SnapshotShardingMode::REPLICATED:
                return "R";
            case SnapshotShardingMode::COLUMN_PARALLEL:
                return "C";
            case SnapshotShardingMode::ROW_PARALLEL:
                return "W";
            case SnapshotShardingMode::GATHERED:
                return "G";
            default:
                return "?";
            }
        };

        // Helper: status icon
        auto status_str = [](bool passed) -> std::string
        {
            return passed ? "✓" : "✗";
        };

        // Helper: format float to string
        auto fmt_f6 = [](float v) -> std::string
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6) << v;
            return ss.str();
        };

        // =========================================================================
        // Title table (separate table for clean title block)
        // =========================================================================
        std::cout << "\n";
        {
            fort::utf8_table title_table;
            title_table.set_border_style(FT_DOUBLE2_STYLE);

            std::ostringstream title_ss;
            title_ss << backend_name << " vs PyTorch TP-AWARE PARITY";

            std::ostringstream subtitle_ss;
            subtitle_ss << summary.tp_degree << "-way LOCAL TP, Threshold: cosine >= "
                        << std::fixed << std::setprecision(3) << config.cosine_threshold;

            title_table << title_ss.str() << fort::endr;
            title_table << subtitle_ss.str() << fort::endr;

            title_table[0][0].set_cell_text_align(fort::text_align::center);
            title_table[1][0].set_cell_text_align(fort::text_align::center);
            title_table.row(0).set_cell_row_type(fort::row_type::header);

            std::cout << title_table.to_string();
        }

        // =========================================================================
        // Main parity table
        // =========================================================================
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        // Build header row: Stage | Dev0 | Dev1 | ... | Combined | Mode | Status
        table << fort::header << "Stage";
        for (size_t i = 0; i < num_devices; ++i)
        {
            table << summary.device_names[i];
        }
        table << "Combined" << "Mode" << "OK" << fort::endr;

        // Set center alignment for all columns
        for (size_t col = 0; col < num_devices + 4; ++col)
        {
            table.column(col).set_cell_text_align(fort::text_align::center);
        }

        // Embedding row
        {
            const auto &emb = summary.embedding_result;
            table << "EMBEDDING";
            for (size_t i = 0; i < num_devices; ++i)
            {
                if (i < emb.device_results.size())
                {
                    table << fmt_f6(emb.device_results[i].cosine_similarity);
                }
                else
                {
                    table << "-";
                }
            }
            table << fmt_f6(emb.combined_cosine)
                  << mode_str(emb.sharding_mode)
                  << status_str(emb.passed)
                  << fort::endr;
        }

        // Per-layer rows
        for (const auto &layer : summary.layer_stats)
        {
            // Layer header row
            std::ostringstream layer_ss;
            layer_ss << "Layer " << layer.layer_idx;

            table << fort::separator;
            table << layer_ss.str();
            for (size_t i = 0; i < num_devices; ++i)
            {
                table << "";
            }
            table << fmt_f6(layer.avg_combined_cosine)
                  << ""
                  << status_str(layer.passed)
                  << fort::endr;

            // Per-stage details (limit to 6 for readability)
            int stage_count = 0;
            for (const auto &stage : layer.stage_results)
            {
                if (stage_count++ >= 6)
                    break;

                // Truncate stage name if needed
                std::string stage_name = stage.stage_name;
                if (stage_name.size() > 14)
                {
                    stage_name = stage_name.substr(0, 12) + "..";
                }

                table << stage_name;
                for (size_t i = 0; i < num_devices; ++i)
                {
                    if (i < stage.device_results.size())
                    {
                        table << fmt_f6(stage.device_results[i].cosine_similarity);
                    }
                    else
                    {
                        table << "-";
                    }
                }
                table << fmt_f6(stage.combined_cosine)
                      << mode_str(stage.sharding_mode)
                      << status_str(stage.passed)
                      << fort::endr;
            }
        }

        // LM_HEAD row
        table << fort::separator;
        table << "LM_HEAD";
        for (size_t i = 0; i < num_devices; ++i)
        {
            table << "[gathered]";
        }
        table << fmt_f6(summary.lm_head_cosine)
              << "G"
              << status_str(summary.lm_head_passed)
              << fort::endr;

        std::cout << table.to_string();

        // =========================================================================
        // Summary footer
        // =========================================================================
        std::cout << "\nSharding modes: R=Replicated, C=Column-parallel, W=roW-parallel, G=Gathered\n";
        std::cout << std::fixed << std::setprecision(4)
                  << "LM_HEAD: KL=" << summary.lm_head_kl
                  << std::setprecision(1)
                  << " Top-1=" << (summary.lm_head_top1 * 100.0f) << "%"
                  << " Top-5=" << (summary.lm_head_top5 * 100.0f) << "%\n";
        std::cout << "Early layers passed: " << summary.early_layers_passed
                  << "/" << config.early_layers_count << "\n";
    }

    // =============================================================================
    // Base Test Class
    // =============================================================================

    /**
     * @brief Base class for PyTorch parity tests
     *
     * Provides common infrastructure for comparing Llaminar backends against
     * PyTorch ground truth. Subclasses must implement:
     * - getDevice() OR getDeviceForRank() - Return the DeviceId to use for inference
     * - getBackendName() - Return a display name (e.g., "CUDA", "CPU", "ROCm")
     *
     * For MPI/tensor-parallel tests:
     * - Set mpi_ctx_ in SetUp() before calling ParityTestBase::SetUp()
     * - Override getDeviceForRank() instead of getDevice()
     *
     * Optional overrides:
     * - setupDeviceSpecific() - Device-specific initialization (e.g., CUDA checks)
     */
    class ParityTestBase : public ::testing::Test
    {
    private:
        // Static set of snapshot generations completed this run.
        // This prevents regenerating snapshots for every test case in a suite.
        // Key: snapshot_dir|model_path
        static inline std::set<std::string> s_generated_snapshots_;
        static inline std::mutex s_snapshot_mutex_;

        static std::string sanitizeSnapshotToken(const std::string &input)
        {
            std::string token;
            token.reserve(input.size());

            bool last_was_underscore = false;
            for (unsigned char ch : input)
            {
                if (std::isalnum(ch))
                {
                    token.push_back(static_cast<char>(std::tolower(ch)));
                    last_was_underscore = false;
                }
                else if (!last_was_underscore)
                {
                    token.push_back('_');
                    last_was_underscore = true;
                }
            }

            while (!token.empty() && token.front() == '_')
                token.erase(token.begin());
            while (!token.empty() && token.back() == '_')
                token.pop_back();

            return token.empty() ? "model" : token;
        }

        static std::string inferSnapshotDirFromModelPath(const std::string &model_path)
        {
            std::string filename = model_path;
            const size_t slash = filename.find_last_of("/\\");
            if (slash != std::string::npos)
                filename = filename.substr(slash + 1);

            const size_t dot = filename.rfind('.');
            if (dot != std::string::npos)
                filename = filename.substr(0, dot);

            return "pytorch_" + sanitizeSnapshotToken(filename) + "_snapshots";
        }

        void resolveSnapshotDirIfNeeded()
        {
            if (config_.snapshot_dir.empty())
            {
                config_.snapshot_dir = inferSnapshotDirFromModelPath(config_.model_path);
            }
        }

        std::string snapshotCacheKey() const
        {
            return config_.snapshot_dir + "|" + config_.model_path;
        }

    protected:
        ParityConfig config_;
        std::shared_ptr<ModelContext> model_ctx_;
        std::unique_ptr<IInferenceRunner> runner_;
        std::unordered_map<std::string, std::vector<float>> pytorch_snapshots_;

        // MPI context for tensor-parallel tests (optional, null for single-rank)
        std::shared_ptr<IMPIContext> mpi_ctx_;

        // Modern orchestration runner (for incremental migration from IInferenceRunner)
        // Tests can use either runner_ OR orch_runner_, but not both simultaneously.
        // Use setupOrchestrationRunner() to initialize orch_runner_, or
        // setupPipeline() to initialize runner_ (legacy path).
        std::unique_ptr<IOrchestrationRunner> orch_runner_;

        // Optional TestConfig for declarative test configuration (LocalPP, LocalTP, etc.)
        // Subclasses override cfg() to return their test configuration.
        // Default returns a single-device CPU config for backward compatibility.
        std::optional<TestConfig> test_config_;

        // =============================================================================
        // Test Configuration
        // =============================================================================

        /**
         * @brief Get the test configuration (override in subclasses)
         * @return Reference to TestConfig for this test
         *
         * Subclasses with declarative config should override this to return
         * their TestConfig. Default returns a single-device CPU configuration.
         */
        virtual const TestConfig &cfg() const
        {
            static const TestConfig default_config = {
                .name = "SingleDevice",
                .devices = {ParityDeviceType::CPU},
                .parallelism = Parallelism::None,
                .collective = Collective::None,
                .thresholds = {},
                .skip_reason = "",
                .mpi_ranks = 1};
            if (test_config_.has_value())
                return test_config_.value();
            return default_config;
        }

        // =============================================================================
        // MPI Helper Methods
        // =============================================================================

        /**
         * @brief Check if this is rank 0 (or single-rank mode)
         * @return true if rank 0 or no MPI context
         */
        bool isRank0() const
        {
            return !mpi_ctx_ || mpi_ctx_->rank() == 0;
        }

        /**
         * @brief Get MPI rank (0 if no MPI context)
         */
        int mpiRank() const
        {
            return mpi_ctx_ ? mpi_ctx_->rank() : 0;
        }

        /**
         * @brief Get MPI world size (1 if no MPI context)
         */
        int mpiWorldSize() const
        {
            return mpi_ctx_ ? mpi_ctx_->world_size() : 1;
        }

        /**
         * @brief Execute MPI barrier if MPI context exists
         */
        void mpiBarrier()
        {
            if (mpi_ctx_)
            {
                mpi_ctx_->barrier();
            }
        }

        // =============================================================================
        // Device Selection (override one of these)
        // =============================================================================

        /**
         * @brief Get the device to use for inference (single-rank tests)
         * @return DeviceId (e.g., DeviceId::cpu(), DeviceId::cuda(0))
         *
         * Override this for single-rank tests. For MPI tests, override
         * getDeviceForRank() instead.
         */
        virtual DeviceId getDevice()
        {
            // Default implementation calls getDeviceForRank() for MPI compatibility
            return getDeviceForRank();
        }

        /**
         * @brief Get the device for this MPI rank (tensor-parallel tests)
         * @return DeviceId for the current rank
         *
         * Override this for MPI tests to return different devices per rank.
         * Default implementation returns CPU.
         */
        virtual DeviceId getDeviceForRank()
        {
            return DeviceId::cpu();
        }

        /**
         * @brief Get the backend name for display
         * @return Name string (e.g., "CUDA", "CPU", "ROCm")
         */
        virtual std::string getBackendName() = 0;

        // =============================================================================
        // Weight Distribution (override for tensor parallelism)
        // =============================================================================

        /**
         * @brief Get the weight distribution strategy
         * @return WeightDistributionStrategy (REPLICATED for single-rank, SHARDED for TP)
         *
         * Override this for tensor-parallel tests to enable weight sharding.
         */
        virtual WeightDistributionStrategy getWeightStrategy()
        {
            return WeightDistributionStrategy::REPLICATED;
        }

        /**
         * @brief Configure model after loading (optional hook)
         * @param model_ctx The loaded model context
         *
         * Override this for tensor-parallel tests to configure weight sharding schema.
         * Called after ModelContext::create() but before createInferenceRunner().
         */
        virtual void configureModel(std::shared_ptr<ModelContext> model_ctx)
        {
            // Default: no additional configuration
        }

        /**
         * @brief Device-specific setup (optional)
         *
         * Override to add device availability checks, GPU initialization, etc.
         * Call GTEST_SKIP() if device is not available.
         */
        virtual void setupDeviceSpecific() {}

        void SetUp() override
        {
            // Enable deterministic CUDA GEMM for reproducible parity results.
            // Stream-K is disabled, and BK64 split-K is capped to 1 because
            // that path's FP32 accumulation-order drift compounds through the
            // attention stack. BK256 remains enabled for FFN-down shapes where
            // the current parity diagnostics show split_k>1 is bitwise-stable.
            setenv("LLAMINAR_DETERMINISTIC", "1", 1);
#ifdef HAVE_CUDA
            cudaNativeVNNIPrefill_setDeterministicMode(true);
#endif

            // Start log file capture for this test run (rank 0 only)
            if (isRank0())
            {
                auto dir = ensureResultsDir();
                auto log_path = dir / "test_log.txt";
                if (Logger::getInstance().setLogFile(log_path.string()))
                {
                    LOG_INFO("[Parity] Log file: " << log_path.string());
                }
            }

            // CRITICAL: Clear kernel caches at test start for clean state.
            // TearDown() also clears, but this guards against incomplete teardown
            // from a prior test (crash, skip, or assertion failure) leaving stale
            // GEMM engines, embedding caches, or prepared-weight handles.
            llaminar::v2::kernels::KernelFactory::clearCache();
#ifdef HAVE_CUDA
            llaminar2::CUDAEmbeddingKernelT::clearGlobalEmbeddingCache();
#endif

            // Device-specific setup first (may skip)
            setupDeviceSpecific();

            // Skip cleanly if the model GGUF isn't staged on this runner.
            // Missing models otherwise produce confusing snapshot-regeneration
            // failures inside Python rather than a clear "model not found".
            if (!config_.model_path.empty() && !std::filesystem::exists(config_.model_path))
            {
                GTEST_SKIP() << "Model file not found: " << config_.model_path
                             << " (populate MODELS_DIR on the runner)";
            }

            resolveSnapshotDirIfNeeded();

            // Regenerate snapshots only on rank 0 to avoid race conditions
            // and redundant work. All ranks wait at barrier before proceeding.
            // OPTIMIZATION: Skip regeneration if snapshots already exist on disk
            // (metadata.txt is the marker file written by all Python generators).
            // This is critical for MPI_PROCS>1 tests where popen()/fork() inside
            // an MPI-managed process can crash the HNP event loop.
            if (isRank0())
            {
                bool need_regen = false;
                {
                    std::lock_guard<std::mutex> lock(s_snapshot_mutex_);
                    need_regen = (s_generated_snapshots_.find(snapshotCacheKey()) == s_generated_snapshots_.end());
                }

                if (need_regen)
                {
                    // Check disk first — snapshots may exist from a prior test run
                    auto metadata_path = std::filesystem::path(config_.snapshot_dir) / "metadata.txt";
                    if (std::filesystem::exists(metadata_path))
                    {
                        LOG_INFO("[" << getBackendName() << " Parity] Found existing snapshots on disk: " << config_.snapshot_dir);
                        need_regen = false;
                    }
                }

                if (need_regen)
                {
                    if (!regeneratePyTorchSnapshots())
                    {
                        FAIL() << "PyTorch snapshot generation failed";
                    }
                    // Mark as generated
                    std::lock_guard<std::mutex> lock(s_snapshot_mutex_);
                    s_generated_snapshots_.insert(snapshotCacheKey());
                }
                else
                {
                    // Mark in cache so subsequent parameterized cases skip the disk check too
                    std::lock_guard<std::mutex> lock(s_snapshot_mutex_);
                    s_generated_snapshots_.insert(snapshotCacheKey());
                    LOG_DEBUG("[" << getBackendName() << " Parity] Reusing cached snapshots from: " << config_.snapshot_dir);
                }
            }
            mpiBarrier(); // All ranks wait for snapshots to be ready
        }

        void TearDown() override
        {
            // Barrier before teardown to ensure all ranks are done
            mpiBarrier();

            // CRITICAL: Clear kernel cache BEFORE destroying model context!
            // KernelFactory::clearCache() accesses tensor->rocm_cache_ and tensor->cuda_cache_
            // to free device memory. If we destroy the tensors first (via model_ctx_.reset()),
            // clearCache() would be accessing freed memory (use-after-free).
            llaminar::v2::kernels::KernelFactory::clearCache();

            // CRITICAL: Clear embedding caches to prevent test pollution!
            // The embedding kernels cache workspace-to-tensor mappings statically.
            // Without clearing, subsequent tests may use stale cached pointers.
#ifdef HAVE_CUDA
            llaminar2::CUDAEmbeddingKernelT::clearGlobalEmbeddingCache();
#endif

            model_ctx_.reset();
            runner_.reset();
            orch_runner_.reset(); // Clean up modern orchestration runner
            pytorch_snapshots_.clear();

            // CRITICAL: Synchronize and clear error state on all GPU devices!
            // After heterogeneous tests (CUDA+ROCm), the HIP runtime can be left
            // in a bad state that causes subsequent ROCm-only tests to fail with
            // "invalid argument" on kernel launch. Synchronizing each backend
            // cleans up any lingering issues.
#ifdef HAVE_CUDA
            if (auto *cuda_backend = llaminar2::getCUDABackend())
            {
                cuda_backend->synchronize(0);
                // Clear CUDA sticky error state to prevent async kernel errors
                // from one test case propagating to the next test's first CUDA API call.
                cudaGetLastError();
            }
#endif
#ifdef HAVE_ROCM
            if (auto *rocm_backend = llaminar2::getROCmBackend())
            {
                // Synchronize ALL ROCm devices, not just device 0.
                // TP configs use multiple ROCm GPUs; leaving device 1+
                // unsynchronized causes ROCm runtime corruption (null pointer
                // in memobj map) when the next test config allocates memory.
                for (int d = 0; d < rocm_backend->deviceCount(); ++d)
                {
                    rocm_backend->synchronize(d);
                }
            }
#endif

            // Close log file at end of test
            Logger::getInstance().closeLogFile();
        }

        /**
         * @brief Regenerate PyTorch snapshots from the GGUF model
         *
         * Override in derived classes to use architecture-specific generators
         * (e.g., Qwen3.5 requires a dedicated generator for GDN layers).
         */
        virtual bool regeneratePyTorchSnapshots()
        {
            LOG_INFO("[" << getBackendName() << " Parity] Regenerating PyTorch snapshots from GGUF: " << config_.model_path);

            std::ostringstream cmd;
            // Source devcontainer venv if present, else fall back to system
            // python3 (which is what the CI builder image uses, where Python
            // deps were installed via `pip install --break-system-packages`).
            //
            // IMPORTANT: CTest sets OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 for the
            // Llaminar test process (intentional: mpirun -np 1 handles affinity itself).
            // Those env vars are inherited by this python3 subprocess, which would pin
            // PyTorch's CPU forward pass to a single thread — catastrophic for large
            // models. We unset them and let PyTorch/OpenMP/MKL use all available cores.
            cmd << "bash -c 'unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS OMP_PROC_BIND OMP_PLACES KMP_AFFINITY; "
                << "[ -f /workspaces/llaminar/.venv/bin/activate ] && source /workspaces/llaminar/.venv/bin/activate; python3"
                << " python/reference/generate_qwen_pipeline_snapshots.py"
                << " --model " << config_.model_path
                << " --prompt \"" << config_.prompt << "\""
                << " --output " << config_.snapshot_dir
                << " --decode-steps " << config_.decode_steps
                << "' 2>&1";

            FILE *pipe = popen(cmd.str().c_str(), "r");
            if (!pipe)
            {
                LOG_ERROR("[Parity] Failed to execute snapshot generator");
                return false;
            }

            char buffer[256];
            std::string output;
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
            {
                output += buffer;
            }

            int exit_code = pclose(pipe);
            if (exit_code != 0)
            {
                LOG_ERROR("[Parity] Snapshot generation failed:\n"
                          << output);
                return false;
            }

            LOG_INFO("[Parity] Snapshots regenerated successfully");
            return true;
        }

        /**
         * @brief Load PyTorch snapshot from .npy file
         */
        std::vector<float> loadPyTorchSnapshot(const std::string &name)
        {
            if (pytorch_snapshots_.find(name) != pytorch_snapshots_.end())
            {
                return pytorch_snapshots_[name];
            }

            std::string npy_path = config_.snapshot_dir + "/" + name + ".npy";

            try
            {
                cnpy::NpyArray arr = cnpy::npy_load(npy_path);

                std::vector<float> data;
                if (arr.word_size == sizeof(float))
                {
                    float *data_ptr = arr.data<float>();
                    data.assign(data_ptr, data_ptr + arr.num_vals);
                }
                else if (arr.word_size == sizeof(double))
                {
                    double *data_ptr = arr.data<double>();
                    data.resize(arr.num_vals);
                    for (size_t i = 0; i < arr.num_vals; ++i)
                    {
                        data[i] = static_cast<float>(data_ptr[i]);
                    }
                }
                else
                {
                    LOG_ERROR("[Parity] Unsupported data type in snapshot '" << name << "'");
                    return {};
                }

                pytorch_snapshots_[name] = data;
                return data;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[Parity] Failed to load snapshot '" << name << "': " << e.what());
                return {};
            }
        }

        // =================================================================
        // GDN V-head permutation for parity comparison
        // =================================================================
        //
        // Llaminar stores GDN V/Z heads in "ratio-grouped" order:
        //   [ratio_0 of all groups, ratio_1 of all groups, ...]
        // PyTorch stores them in "interleaved" order:
        //   [group_0 all ratios, group_1 all ratios, ...]
        //
        // With n_k_heads=16, n_v_heads=32: heads_per_group = 2.
        //   Llaminar: [even_heads_0..15, odd_heads_16..31]
        //   PyTorch:  [h0, h1, h2, h3, ..., h31] interleaved
        //
        // The inverse permutation maps LL ordering → PT ordering:
        //   inv_perm[pt_head] = ll_head
        //   where ll_head = (pt_head % heads_per_group) * n_k_heads
        //                  + (pt_head / heads_per_group)
        //
        // This applies to the V portion of QKV_PROJECTION, and to
        // GDN_Z_PROJECTION, GDN_DELTA_RULE_OUTPUT, GDN_NORM_GATE_OUTPUT.

        /**
         * @brief Check if the model has GDN layers with non-trivial head permutation
         * @return {n_k_heads, n_v_heads, d_state} or {0,0,0} if no permutation needed
         */
        struct GDNHeadConfig
        {
            int n_k_heads = 0;
            int n_v_heads = 0;
            int d_state = 0;

            bool needsPermutation() const
            {
                return n_k_heads > 0 && n_v_heads > n_k_heads && d_state > 0;
            }

            int headsPerGroup() const
            {
                return n_k_heads > 0 ? n_v_heads / n_k_heads : 1;
            }
        };

        GDNHeadConfig getGDNHeadConfig() const
        {
            if (!model_ctx_)
                return {};

            const auto &arch = model_ctx_->architecture();
            const auto &meta = model_ctx_->model().metadata;

            auto getMetaInt = [&](const std::string &suffix) -> int
            {
                auto it = meta.find(arch + "." + suffix);
                if (it == meta.end())
                    return 0;
                const auto &val = it->second;
                if (val.type == GGUFValueType::UINT32)
                    return static_cast<int>(val.asUInt32());
                if (val.type == GGUFValueType::UINT64)
                    return static_cast<int>(val.asUInt64());
                return 0;
            };

            GDNHeadConfig cfg;
            cfg.n_k_heads = getMetaInt("ssm.group_count");
            cfg.n_v_heads = getMetaInt("ssm.time_step_rank");
            cfg.d_state = getMetaInt("ssm.state_size");
            return cfg;
        }

        /**
         * @brief Read MoE configuration from GGUF metadata
         */
        struct MoEConfig
        {
            int num_experts = 0;
            int top_k = 0;
        };

        MoEConfig getMoEConfig() const
        {
            if (!model_ctx_)
                return {};

            const auto &arch = model_ctx_->architecture();
            const auto &meta = model_ctx_->model().metadata;

            auto getMetaInt = [&](const std::string &suffix) -> int
            {
                auto it = meta.find(arch + "." + suffix);
                if (it == meta.end())
                    return 0;
                const auto &val = it->second;
                if (val.type == GGUFValueType::UINT32)
                    return static_cast<int>(val.asUInt32());
                if (val.type == GGUFValueType::UINT64)
                    return static_cast<int>(val.asUInt64());
                return 0;
            };

            MoEConfig cfg;
            cfg.num_experts = getMetaInt("expert_count");
            cfg.top_k = getMetaInt("expert_used_count");
            return cfg;
        }

        /**
         * @brief Apply GDN V-head permutation to Llaminar data for comparison.
         *
         * For stages that contain V-head-ordered data (Z projection, delta rule,
         * norm gate), permutes from Llaminar's ratio-grouped order to PyTorch's
         * interleaved order. For QKV_PROJECTION, only the V portion is permuted.
         *
         * @return Permuted copy if permutation was applied, empty vector otherwise.
         *         When non-empty, use permuted.data() instead of llaminar_data.
         */
        std::vector<float> applyGDNHeadPermutation(
            const float *llaminar_data,
            size_t size,
            const std::string &stage,
            const GDNHeadConfig &gdn) const
        {
            if (!gdn.needsPermutation())
                return {};

            const int n_k = gdn.n_k_heads;
            const int n_v = gdn.n_v_heads;
            const int d = gdn.d_state;
            const int hpg = gdn.headsPerGroup(); // heads per group (n_v / n_k)

            // Build inverse permutation: inv[pt_head] = ll_head
            std::vector<int> inv_perm(static_cast<size_t>(n_v));
            for (int pt_h = 0; pt_h < n_v; ++pt_h)
            {
                int ratio = pt_h % hpg;
                int group = pt_h / hpg;
                inv_perm[static_cast<size_t>(pt_h)] = ratio * n_k + group;
            }

            auto permuteHeads = [&](const float *src, size_t total_elements) -> std::vector<float>
            {
                const size_t head_dim = static_cast<size_t>(d);
                const size_t n_heads = static_cast<size_t>(n_v);
                const size_t tokens = total_elements / (n_heads * head_dim);
                if (tokens * n_heads * head_dim != total_elements)
                    return {}; // Size doesn't match expected layout

                std::vector<float> out(total_elements);
                for (size_t t = 0; t < tokens; ++t)
                {
                    for (size_t pt_h = 0; pt_h < n_heads; ++pt_h)
                    {
                        size_t ll_h = static_cast<size_t>(inv_perm[pt_h]);
                        std::memcpy(
                            &out[(t * n_heads + pt_h) * head_dim],
                            &src[(t * n_heads + ll_h) * head_dim],
                            head_dim * sizeof(float));
                    }
                }
                return out;
            };

            if (stage == "GDN_Z_PROJECTION" ||
                stage == "GDN_DELTA_RULE_OUTPUT" ||
                stage == "GDN_NORM_GATE_OUTPUT")
            {
                return permuteHeads(llaminar_data, size);
            }

            if (stage == "QKV_PROJECTION")
            {
                // QKV layout: [seq, Q(n_k*d) | K(n_k*d) | V(n_v*d)]
                const size_t q_dim = static_cast<size_t>(n_k * d);
                const size_t k_dim = static_cast<size_t>(n_k * d);
                const size_t v_dim = static_cast<size_t>(n_v * d);
                const size_t qkv_dim = q_dim + k_dim + v_dim;
                const size_t tokens = size / qkv_dim;
                if (tokens * qkv_dim != size)
                    return {}; // Size doesn't match

                // Permute only the V portion
                std::vector<float> out(llaminar_data, llaminar_data + size); // copy all
                for (size_t t = 0; t < tokens; ++t)
                {
                    const float *v_src = llaminar_data + t * qkv_dim + q_dim + k_dim;
                    float *v_dst = out.data() + t * qkv_dim + q_dim + k_dim;
                    for (size_t pt_h = 0; pt_h < static_cast<size_t>(n_v); ++pt_h)
                    {
                        size_t ll_h = static_cast<size_t>(inv_perm[pt_h]);
                        std::memcpy(
                            &v_dst[pt_h * static_cast<size_t>(d)],
                            &v_src[ll_h * static_cast<size_t>(d)],
                            static_cast<size_t>(d) * sizeof(float));
                    }
                }
                return out;
            }

            return {}; // Not a GDN stage
        }

        /**
         * @brief Compare tensors and compute metrics
         */
        StageComparisonResult compareTensors(
            const float *actual,
            const std::vector<float> &expected,
            size_t size,
            const std::string &stage_name = "")
        {
            StageComparisonResult result;
            result.stage_name = stage_name;
            result.total_elements = size;

            if (expected.empty() || expected.size() != size)
            {
                return result;
            }

            // Fused pass: compute cosine/L2/max_abs_diff and error histogram in one traversal
            double sum_sq_diff = 0.0;
            double sum_sq_expected = 0.0;
            double dot_product = 0.0;
            double norm_actual_sq = 0.0;
            double norm_expected_sq = 0.0;
            float max_abs_diff = 0.0f;

#pragma omp parallel for reduction(+ : sum_sq_diff, sum_sq_expected, dot_product, norm_actual_sq, norm_expected_sq) \
    reduction(max : max_abs_diff) schedule(static) if (size > 8192)
            for (size_t i = 0; i < size; ++i)
            {
                double a = static_cast<double>(actual[i]);
                double e = static_cast<double>(expected[i]);
                double diff = a - e;
                sum_sq_diff += diff * diff;
                sum_sq_expected += e * e;
                dot_product += a * e;
                norm_actual_sq += a * a;
                norm_expected_sq += e * e;

                float abs_diff = std::abs(actual[i] - expected[i]);
                if (abs_diff > max_abs_diff)
                    max_abs_diff = abs_diff;
            }
            result.max_abs_diff = max_abs_diff;

            // Relative L2
            if (sum_sq_expected > 1e-10)
            {
                result.rel_l2_norm = static_cast<float>(std::sqrt(sum_sq_diff / sum_sq_expected));
            }

            // Cosine similarity
            double norm_product = std::sqrt(norm_actual_sq) * std::sqrt(norm_expected_sq);
            if (norm_product > 1e-10)
            {
                result.cosine_similarity = static_cast<float>(dot_product / norm_product);
            }

            // RMSE
            result.rmse = static_cast<float>(std::sqrt(sum_sq_diff / static_cast<double>(size)));

            // SNR in dB: 10·log10(‖signal‖² / ‖error‖²)
            if (sum_sq_diff > 1e-30)
            {
                result.snr_db = static_cast<float>(10.0 * std::log10(sum_sq_expected / sum_sq_diff));
            }
            else if (sum_sq_expected > 1e-30)
            {
                result.snr_db = 100.0f; // Perfect match, cap at 100 dB
            }

            // Error histogram entropy — separate pass needed because max_abs_diff
            // must be known first for binning
            {
                constexpr int N_BINS = 64;
                std::array<size_t, N_BINS> bins = {};
                float max_err = result.max_abs_diff;
                if (max_err > 1e-10f)
                {
                    float inv_max = static_cast<float>(N_BINS - 1) / max_err;
                    // Can't trivially OMP-reduce an array, but this loop is
                    // cheap compared to distribution stats, so keep it serial
                    for (size_t i = 0; i < size; ++i)
                    {
                        float abs_err = std::abs(actual[i] - expected[i]);
                        int bin = std::min(static_cast<int>(abs_err * inv_max), N_BINS - 1);
                        bins[bin]++;
                    }
                    double entropy = 0.0;
                    double n = static_cast<double>(size);
                    for (int b = 0; b < N_BINS; ++b)
                    {
                        if (bins[b] == 0)
                            continue;
                        double p = static_cast<double>(bins[b]) / n;
                        entropy -= p * std::log2(p);
                    }
                    result.error_entropy = static_cast<float>(entropy);
                }
            }

            // Distribution stats for both tensors
            result.llaminar_stats = computeDistributionStats(actual, size);
            result.pytorch_stats = computeDistributionStats(expected.data(), size);

            result.passed = (result.cosine_similarity >= config_.cosine_threshold);
            return result;
        }

        /**
         * @brief Compare MoE routing indices (expert selection).
         *
         * For each token, computes the set overlap between selected expert IDs
         * (Jaccard-like: |intersection| / top_k). Stores mean overlap in
         * cosine_similarity for consistent table rendering.
         *
         * @param actual    Llaminar routing indices [seq_len * top_k] (int cast to float)
         * @param expected  PyTorch routing indices [seq_len * top_k] (int cast to float)
         * @param size      Total elements (seq_len * top_k)
         * @param top_k     Number of experts selected per token
         */
        StageComparisonResult compareRoutingIndices(
            const float *actual,
            const std::vector<float> &expected,
            size_t size,
            int top_k,
            const std::string &stage_name = "MOE_ROUTING_INDICES")
        {
            StageComparisonResult result;
            result.stage_name = stage_name;
            result.total_elements = size;

            if (expected.empty() || expected.size() != size || top_k <= 0)
                return result;

            const size_t seq_len = size / static_cast<size_t>(top_k);
            double total_overlap = 0.0;
            double total_rank_corr = 0.0;
            int position_0_match = 0; // How often the top-1 expert matches

            for (size_t t = 0; t < seq_len; ++t)
            {
                const float *ll_row = actual + t * top_k;
                const float *pt_row = expected.data() + t * top_k;

                // Build sets for intersection
                std::set<int> ll_set, pt_set;
                for (int k = 0; k < top_k; ++k)
                {
                    ll_set.insert(static_cast<int>(ll_row[k]));
                    pt_set.insert(static_cast<int>(pt_row[k]));
                }

                // Set intersection size
                int overlap = 0;
                for (int id : ll_set)
                    if (pt_set.count(id))
                        overlap++;

                total_overlap += static_cast<double>(overlap) / top_k;

                // Top-1 match (most-weighted expert)
                if (static_cast<int>(ll_row[0]) == static_cast<int>(pt_row[0]))
                    position_0_match++;
            }

            result.cosine_similarity = static_cast<float>(total_overlap / seq_len);
            // Repurpose snr_db for top-1 match rate (both are diagnostic "quality" metrics)
            result.snr_db = static_cast<float>(position_0_match) / static_cast<float>(seq_len);
            result.max_abs_diff = 1.0f - result.cosine_similarity; // "distance" from perfect
            result.passed = (result.cosine_similarity >= config_.cosine_threshold);
            return result;
        }

        /**
         * @brief Compare MoE routing weights (expert contributions).
         *
         * For each token, creates a sparse [num_experts]-dim vector where
         * vec[expert_id] = routing_weight, then computes cosine similarity
         * between the two sparse vectors. This naturally handles different
         * expert orderings and partial set overlaps.
         *
         * @param actual_weights   Llaminar routing weights [seq_len * top_k]
         * @param expected_weights PyTorch routing weights [seq_len * top_k]
         * @param actual_indices   Llaminar routing indices [seq_len * top_k]
         * @param expected_indices PyTorch routing indices [seq_len * top_k]
         * @param size             Elements in weight arrays (seq_len * top_k)
         * @param top_k            Experts per token
         * @param num_experts      Total expert count (for sparse vector dim)
         */
        StageComparisonResult compareRoutingWeights(
            const float *actual_weights,
            const std::vector<float> &expected_weights,
            const float *actual_indices,
            const std::vector<float> &expected_indices,
            size_t size,
            int top_k,
            int num_experts,
            const std::string &stage_name = "MOE_ROUTING_WEIGHTS")
        {
            StageComparisonResult result;
            result.stage_name = stage_name;
            result.total_elements = size;

            if (expected_weights.empty() || expected_weights.size() != size || top_k <= 0)
                return result;

            const size_t seq_len = size / static_cast<size_t>(top_k);
            double total_cosine = 0.0;
            double total_l1 = 0.0;
            float max_weight_diff = 0.0f;

            for (size_t t = 0; t < seq_len; ++t)
            {
                // Build sparse weight vectors indexed by expert ID
                std::vector<float> ll_sparse(num_experts, 0.0f);
                std::vector<float> pt_sparse(num_experts, 0.0f);

                for (int k = 0; k < top_k; ++k)
                {
                    int ll_id = static_cast<int>(actual_indices[t * top_k + k]);
                    int pt_id = static_cast<int>(expected_indices[t * top_k + k]);
                    if (ll_id >= 0 && ll_id < num_experts)
                        ll_sparse[ll_id] = actual_weights[t * top_k + k];
                    if (pt_id >= 0 && pt_id < num_experts)
                        pt_sparse[pt_id] = expected_weights[t * top_k + k];
                }

                // Cosine similarity of sparse weight vectors
                double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
                double l1 = 0.0;
                for (int e = 0; e < num_experts; ++e)
                {
                    dot += ll_sparse[e] * pt_sparse[e];
                    norm_a += ll_sparse[e] * ll_sparse[e];
                    norm_b += pt_sparse[e] * pt_sparse[e];
                    float diff = std::abs(ll_sparse[e] - pt_sparse[e]);
                    l1 += diff;
                    if (diff > max_weight_diff)
                        max_weight_diff = diff;
                }

                double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
                total_cosine += (denom > 1e-30) ? (dot / denom) : 0.0;
                total_l1 += l1;
            }

            result.cosine_similarity = static_cast<float>(total_cosine / seq_len);
            result.rmse = static_cast<float>(total_l1 / seq_len); // Repurpose for mean L1
            result.max_abs_diff = max_weight_diff;
            result.passed = (result.cosine_similarity >= config_.cosine_threshold);
            return result;
        }

        /**
         * @brief Check if a stage name is a MoE routing stage needing special comparison
         */
        static bool isRoutingStage(const std::string &stage)
        {
            return stage == "MOE_ROUTING_INDICES" || stage == "MOE_ROUTING_WEIGHTS";
        }

        /**
         * @brief Compare TP snapshot against PyTorch reference with sharding awareness
         *
         * For column-parallel stages:
         * 1. Compare each device's partial output against corresponding PyTorch slice
         * 2. Concatenate all device outputs and compare against full PyTorch
         *
         * For row-parallel/replicated stages:
         * 1. Verify all devices have consistent data
         * 2. Compare combined output against full PyTorch
         *
         * @param tp_snapshot The TPSnapshot from RankOrchestrator::getTPSnapshot()
         * @param pytorch_data Full PyTorch reference data
         * @param pytorch_rows Number of rows in PyTorch data (seq_len)
         * @param pytorch_cols Number of columns in PyTorch data (feature_dim)
         * @return TPStageComparisonResult with per-device and combined metrics
         */
        TPStageComparisonResult compareTPSnapshot(
            TPSnapshot &tp_snapshot,
            const std::vector<float> &pytorch_data,
            size_t pytorch_rows,
            size_t pytorch_cols)
        {
            TPStageComparisonResult result;

            // Debug: print first few values for layer0 attention to diagnose
            bool debug_print = (tp_snapshot.key.find("layer0_ATTENTION_CONTEXT") != std::string::npos);
            result.stage_name = tp_snapshot.key;
            result.sharding_mode = tp_snapshot.mode;

            if (pytorch_data.empty())
            {
                LOG_WARN("[TP Parity] No PyTorch data for stage " << tp_snapshot.key);
                return result;
            }

            const int tp_degree = static_cast<int>(tp_snapshot.device_data.size());
            if (tp_degree == 0)
            {
                LOG_WARN("[TP Parity] No device data for stage " << tp_snapshot.key);
                return result;
            }

            LOG_DEBUG("[TP Parity] Comparing stage " << tp_snapshot.key
                                                     << " mode=" << shardingModeToString(tp_snapshot.mode)
                                                     << " tp_degree=" << tp_degree
                                                     << " pytorch_size=" << pytorch_data.size()
                                                     << " (" << pytorch_rows << "x" << pytorch_cols << ")");

            // Per-device comparison for column-parallel stages
            if (tp_snapshot.mode == SnapshotShardingMode::COLUMN_PARALLEL)
            {
                for (int dev_idx = 0; dev_idx < tp_degree; ++dev_idx)
                {
                    const auto &dev_data = tp_snapshot.device_data[dev_idx];

                    // Compute which slice of PyTorch this device should match
                    size_t slice_start = computeSliceStartCol(dev_idx, tp_degree, pytorch_cols);
                    size_t slice_cols = computeSliceColCount(dev_idx, tp_degree, pytorch_cols);

                    // Extract PyTorch slice
                    std::vector<float> pytorch_slice = extractColumnSlice(
                        pytorch_data.data(), pytorch_rows, pytorch_cols, slice_start, slice_cols);

                    // Debug: print actual values for diagnosis - VERBOSE for ATTENTION_CONTEXT and Q_ROPE stages
                    bool debug_verbose = (tp_snapshot.key.find("ATTENTION_CONTEXT") != std::string::npos) ||
                                         (tp_snapshot.key.find("Q_ROPE") != std::string::npos && tp_snapshot.key.find("layer0") != std::string::npos);
                    if (debug_verbose)
                    {
                        LOG_INFO("[TP Debug] " << tp_snapshot.key << " device " << dev_idx
                                               << " pytorch_rows=" << pytorch_rows
                                               << " pytorch_cols=" << pytorch_cols
                                               << " slice_start=" << slice_start
                                               << " slice_cols=" << slice_cols
                                               << " pytorch_slice.size=" << pytorch_slice.size()
                                               << " dev_data.size=" << dev_data.data.size()
                                               << " dev_data.cols=" << dev_data.cols);

                        // Print first 8 values from each
                        std::stringstream ss_pt, ss_ll;
                        for (size_t i = 0; i < std::min(size_t(8), pytorch_slice.size()); ++i)
                            ss_pt << std::setprecision(6) << pytorch_slice[i] << ", ";
                        for (size_t i = 0; i < std::min(size_t(8), dev_data.data.size()); ++i)
                            ss_ll << std::setprecision(6) << dev_data.data[i] << ", ";
                        LOG_INFO("[TP Debug]   Row0 PyTorch: " << ss_pt.str());
                        LOG_INFO("[TP Debug]   Row0 Llaminar: " << ss_ll.str());

                        // Row 1 values (at offset slice_cols for Llaminar, slice_cols for pytorch_slice)
                        size_t row_stride_ll = dev_data.cols > 0 ? dev_data.cols : slice_cols;
                        size_t row_stride_pt = slice_cols;
                        if (pytorch_slice.size() > row_stride_pt + 8 && dev_data.data.size() > row_stride_ll + 8)
                        {
                            std::stringstream ss3, ss4;
                            for (size_t i = 0; i < 8; ++i)
                                ss3 << std::setprecision(6) << pytorch_slice[row_stride_pt + i] << ", ";
                            for (size_t i = 0; i < 8; ++i)
                                ss4 << std::setprecision(6) << dev_data.data[row_stride_ll + i] << ", ";
                            LOG_INFO("[TP Debug]   Row1 PyTorch (stride=" << row_stride_pt << "): " << ss3.str());
                            LOG_INFO("[TP Debug]   Row1 Llaminar (stride=" << row_stride_ll << "): " << ss4.str());
                        }
                    }

                    // Compare device data against slice
                    size_t compare_size = std::min(dev_data.data.size(), pytorch_slice.size());
                    float cosine = computeCosineSimilarity(
                        dev_data.data.data(), pytorch_slice.data(), compare_size);

                    // Per-row cosine diagnostic for ATTENTION_CONTEXT
                    if (debug_verbose && slice_cols > 0)
                    {
                        size_t num_rows = compare_size / slice_cols;
                        float max_abs_diff = 0.0f;
                        size_t max_diff_idx = 0;
                        std::stringstream row_cosines;
                        for (size_t r = 0; r < num_rows && r < 9; ++r)
                        {
                            float rc = computeCosineSimilarity(
                                dev_data.data.data() + r * slice_cols,
                                pytorch_slice.data() + r * slice_cols,
                                slice_cols);
                            row_cosines << "row" << r << "=" << std::fixed << std::setprecision(6) << rc << " ";
                            // Find max absolute difference in this row
                            for (size_t c = 0; c < slice_cols; ++c)
                            {
                                float diff = std::abs(dev_data.data[r * slice_cols + c] - pytorch_slice[r * slice_cols + c]);
                                if (diff > max_abs_diff)
                                {
                                    max_abs_diff = diff;
                                    max_diff_idx = r * slice_cols + c;
                                }
                            }
                        }
                        LOG_INFO("[TP Diag] " << tp_snapshot.key << " dev" << dev_idx
                                              << " per-row cosine: " << row_cosines.str());
                        LOG_INFO("[TP Diag] " << tp_snapshot.key << " dev" << dev_idx
                                              << " max_abs_diff=" << max_abs_diff
                                              << " at idx=" << max_diff_idx
                                              << " (row=" << max_diff_idx / slice_cols
                                              << " col=" << max_diff_idx % slice_cols << ")"
                                              << " llaminar=" << dev_data.data[max_diff_idx]
                                              << " pytorch=" << pytorch_slice[max_diff_idx]);
                    }

                    TPDeviceComparisonResult dev_result;
                    dev_result.device_id = dev_data.device_id.toString();
                    dev_result.device_index = dev_idx;
                    dev_result.cosine_similarity = cosine;
                    dev_result.slice_start = slice_start;
                    dev_result.slice_size = compare_size;
                    dev_result.passed = (cosine >= config_.cosine_threshold);

                    LOG_DEBUG("[TP Parity] Device " << dev_idx << " (" << dev_result.device_id << ")"
                                                    << " slice=[" << slice_start << "," << slice_start + slice_cols << ")"
                                                    << " cosine=" << cosine
                                                    << " device_size=" << dev_data.data.size()
                                                    << " slice_size=" << pytorch_slice.size());

                    result.device_results.push_back(std::move(dev_result));
                }
            }
            else
            {
                // For replicated/row-parallel, each device should match full PyTorch
                for (int dev_idx = 0; dev_idx < tp_degree; ++dev_idx)
                {
                    const auto &dev_data = tp_snapshot.device_data[dev_idx];

                    size_t compare_size = std::min(dev_data.data.size(), pytorch_data.size());
                    float cosine = computeCosineSimilarity(
                        dev_data.data.data(), pytorch_data.data(), compare_size);

                    TPDeviceComparisonResult dev_result;
                    dev_result.device_id = dev_data.device_id.toString();
                    dev_result.device_index = dev_idx;
                    dev_result.cosine_similarity = cosine;
                    dev_result.slice_start = 0;
                    dev_result.slice_size = compare_size;
                    dev_result.passed = (cosine >= config_.cosine_threshold);

                    result.device_results.push_back(std::move(dev_result));
                }
            }

            // Compute combined result
            size_t combined_size = 0;
            const float *combined_ptr = tp_snapshot.getCombinedData(combined_size);

            if (combined_ptr && combined_size > 0)
            {
                size_t compare_size = std::min(combined_size, pytorch_data.size());
                result.combined_cosine = computeCosineSimilarity(
                    combined_ptr, pytorch_data.data(), compare_size);
                result.combined_elements = compare_size;
                result.combined_passed = (result.combined_cosine >= config_.cosine_threshold);

                LOG_DEBUG("[TP Parity] Combined: size=" << combined_size
                                                        << " pytorch_size=" << pytorch_data.size()
                                                        << " cosine=" << result.combined_cosine);
            }
            else
            {
                LOG_WARN("[TP Parity] Failed to compute combined data for " << tp_snapshot.key);
            }

            // Overall pass: combined must pass
            result.passed = result.combined_passed;

            return result;
        }

        /**
         * @brief Setup the inference pipeline
         *
         * MPI-aware: Uses getDevice() which calls getDeviceForRank() for per-rank device selection.
         * Passes mpi_ctx_ to createInferenceRunner (nullptr for single-rank tests).
         * Uses getWeightStrategy() for weight distribution (REPLICATED or SHARDED).
         * Calls configureModel() hook for tensor-parallel schema configuration.
         */
        bool setupPipeline()
        {
            DeviceManager::instance().initialize(-1);

            // Load model with MPI context and weight strategy
            model_ctx_ = ModelContext::create(
                config_.model_path,
                mpi_ctx_,           // nullptr for single-rank
                nullptr,            // placement_map
                nullptr,            // factory
                getWeightStrategy() // REPLICATED or SHARDED
            );
            if (!model_ctx_)
            {
                LOG_ERROR("[Parity] Failed to load model");
                return false;
            }

            // Allow subclasses to configure model (e.g., weight sharding schema)
            configureModel(model_ctx_);

            InferenceRunnerConfig inf_config;
            inf_config.max_seq_len = 4096;
            inf_config.batch_size = 1;
            inf_config.force_graph = true;
            inf_config.activation_precision = cfg().activation_precision;
            inf_config.kv_cache_precision = cfg().kv_cache_precision;

            // Enable mapped memory for GPU devices to avoid slow D2H syncs during snapshot capture
            // This works for both CUDA and ROCm - mapped memory enables zero-copy host access
            DeviceId device = getDevice();
            if (device.is_gpu())
            {
                inf_config.use_mapped_memory = true;
                if (isRank0())
                {
                    LOG_INFO("[" << getBackendName() << " Parity] Enabling mapped memory for GPU snapshot capture");
                }
            }

            // Pass mpi_ctx_ to enable tensor parallelism (nullptr for single-rank tests)
            runner_ = createInferenceRunner(model_ctx_, mpi_ctx_, device, inf_config);
            if (!runner_)
            {
                LOG_ERROR("[Parity] Failed to create inference runner");
                return false;
            }

            runner_->enableSnapshotCapture();
            if (isRank0())
            {
                LOG_INFO("[" << getBackendName() << " Parity] Inference runner created"
                             << (mpi_ctx_ ? " (MPI world_size=" + std::to_string(mpiWorldSize()) + ")" : ""));
            }
            return true;
        }

        /**
         * @brief Setup pipeline for LocalPP (Pipeline Parallelism) tests
         *
         * Creates a pipeline parallel configuration where layers are split across
         * multiple devices. Uses RankOrchestrator with PP mode which:
         * - Creates per-stage DeviceGraphOrchestrator instances
         * - Handles sequential forward execution through stages
         * - Manages activation transfer via LocalPPContext
         *
         * This is model-agnostic: any model that reports blockCount() can be used.
         * Uses cfg() to get device list, collective backend, etc.
         *
         * @return true if setup succeeded, false on error
         */
        bool setupLocalPPPipeline()
        {
            DeviceManager::instance().initialize(-1);

            // Initialize GlobalBackendRouter for activation transfers between PP stages
            GlobalBackendRouter::initForTests();

            // Load model with REPLICATED strategy (each stage needs full model access
            // for weight lookup - layer partitioning happens at runtime)
            model_ctx_ = ModelContext::create(
                config_.model_path,
                mpi_ctx_,
                nullptr,
                nullptr,
                WeightDistributionStrategy::REPLICATED);

            if (!model_ctx_)
            {
                LOG_ERROR("[Parity] Failed to load model");
                return false;
            }

            // Get model parameters (model-agnostic)
            int n_layers = model_ctx_->blockCount();

            // Build GlobalDeviceAddress list from TestConfig
            std::vector<GlobalDeviceAddress> all_device_addresses;
            int cuda_idx = 0, rocm_idx = 0;
            for (auto dt : cfg().devices)
            {
                switch (dt)
                {
                case ParityDeviceType::CPU:
                    all_device_addresses.push_back(GlobalDeviceAddress::cpu());
                    break;
                case ParityDeviceType::CUDA:
                    all_device_addresses.push_back(GlobalDeviceAddress::cuda(cuda_idx++));
                    break;
                case ParityDeviceType::ROCm:
                    all_device_addresses.push_back(GlobalDeviceAddress::rocm(rocm_idx++));
                    break;
                }
            }

            // Determine number of PP stages:
            // - If pp_stage_sizes is set, use its length (supports hybrid PP+TP)
            // - Otherwise, assume 1 device per stage (pure PP)
            int num_stages = static_cast<int>(cfg().num_pp_stages());
            if (num_stages < 1)
                num_stages = 1;

            // Calculate layer boundaries (equal split)
            int layers_per_stage = n_layers / num_stages;

            if (cfg().is_hybrid_pp_tp())
            {
                LOG_INFO("[Parity] Hybrid PP+TP configuration: " << num_stages << " PP stages, "
                                                                 << n_layers << " layers");
                for (size_t i = 0; i < cfg().pp_stage_sizes.size(); ++i)
                {
                    LOG_INFO("[Parity]   Stage " << i << " has " << cfg().pp_stage_sizes[i]
                                                 << " device(s)" << (cfg().pp_stage_sizes[i] > 1 ? " (TP domain)" : ""));
                }
            }
            else
            {
                LOG_INFO("[Parity] LocalPP configuration: " << num_stages << " stages, "
                                                            << n_layers << " layers");
            }

            // Create RankOrchestrator::Config with PP mode (or TP_PP for hybrid)
            RankOrchestrator::Config mdo_config;
            mdo_config.mode = cfg().is_hybrid_pp_tp()
                                  ? RankOrchestrator::ParallelismMode::TP_PP
                                  : RankOrchestrator::ParallelismMode::PP;
            mdo_config.max_seq_len = 4096;
            mdo_config.batch_size = 1;
            mdo_config.activation_precision = ActivationPrecision::FP32;

            // Match single-device parity setup: GPU snapshot capture uses mapped
            // memory to avoid stage-by-stage D2H races on reused activation buffers.
            for (auto dt : cfg().devices)
            {
                if (dt == ParityDeviceType::CUDA || dt == ParityDeviceType::ROCm)
                {
                    mdo_config.use_mapped_memory = true;
                    break;
                }
            }

            // Create PP stage configurations
            // Track which device index we're at in the flat device list
            size_t device_offset = 0;

            for (int s = 0; s < num_stages; ++s)
            {
                RankOrchestrator::PPStageConfig stage_config;
                stage_config.first_layer = s * layers_per_stage;
                stage_config.last_layer = (s == num_stages - 1) ? n_layers : (s + 1) * layers_per_stage;
                stage_config.has_embedding = (s == 0);
                stage_config.has_lm_head = (s == num_stages - 1);

                // Determine how many devices this stage gets
                int stage_device_count = 1;
                if (!cfg().pp_stage_sizes.empty() && s < static_cast<int>(cfg().pp_stage_sizes.size()))
                {
                    stage_device_count = cfg().pp_stage_sizes[s];
                }

                // Add devices for this stage
                for (int d = 0; d < stage_device_count && device_offset < all_device_addresses.size(); ++d)
                {
                    stage_config.stage_devices.push_back(all_device_addresses[device_offset++]);
                }

                // If this is a TP domain (multiple devices), set TP backend
                if (stage_config.isTPDomain())
                {
                    // Use tp_collective for intra-stage TP backend
                    stage_config.tp_backend = toCollectiveBackend(cfg().tp_collective);

                    // Equal TP weights for this domain
                    for (int d = 0; d < stage_device_count; ++d)
                    {
                        stage_config.tp_weights.push_back(1.0f / static_cast<float>(stage_device_count));
                    }
                }

                // Log stage configuration
                std::ostringstream devices_str;
                for (size_t i = 0; i < stage_config.stage_devices.size(); ++i)
                {
                    if (i > 0)
                        devices_str << ", ";
                    devices_str << stage_config.stage_devices[i].toString();
                }
                LOG_INFO("[Parity]   Stage " << s << ": layers ["
                                             << stage_config.first_layer << ", " << stage_config.last_layer << ") on "
                                             << (stage_config.isTPDomain() ? "TP(" : "")
                                             << devices_str.str()
                                             << (stage_config.isTPDomain() ? ")" : ""));

                mdo_config.pp_stages.push_back(stage_config);
            }

            // Validate PP config
            if (!mdo_config.validate())
            {
                LOG_ERROR("[Parity] Invalid RankOrchestrator PP config");
                return false;
            }

            // Create RankOrchestrator with PP mode
            auto multi_orch = std::make_unique<RankOrchestrator>(model_ctx_, mdo_config);

            if (!multi_orch)
            {
                LOG_ERROR("[Parity] Failed to create RankOrchestrator for PP");
                return false;
            }

            // Enable snapshot capture for parity testing
            multi_orch->enableSnapshotCapture();

            LOG_INFO("[Parity] RankOrchestrator " << (cfg().is_hybrid_pp_tp() ? "TP_PP" : "PP")
                                                  << " created with " << num_stages << " stages");

            // Transfer ownership to base class runner_
            runner_ = std::move(multi_orch);

            return true;
        }

        // =========================================================================
        // Modern Orchestration Runner Support (for incremental migration)
        // =========================================================================

        /**
         * @brief Setup an OrchestrationRunner for parity testing
         *
         * This is the modern alternative to setupPipeline(). Creates an
         * OrchestrationRunner using the TestOrchestrationHelper utilities.
         *
         * Unlike setupPipeline(), this method:
         * - Uses the OrchestrationConfig/OrchestrationRunner pattern
         * - Has a simpler, more declarative API
         * - Supports all orchestration features (PP, LOCAL TP, etc.)
         *
         * @param orch_config Pre-configured OrchestrationConfig
         * @return true on success, false on failure
         *
         * @note Call this OR setupPipeline(), not both. The active runner is
         *       whichever was last set up successfully.
         */
        bool setupOrchestrationRunner(const OrchestrationConfig &orch_config)
        {
            // Clear any existing runner
            orch_runner_.reset();

            // Create runner via TestOrchestrationHelper
            orch_runner_ = test::TestOrchestrationHelper::create(orch_config);
            if (!orch_runner_)
            {
                LOG_ERROR("[Parity] Failed to create OrchestrationRunner");
                return false;
            }

            // Initialize the runner
            if (!orch_runner_->initialize())
            {
                LOG_ERROR("[Parity] Failed to initialize OrchestrationRunner: " << orch_runner_->lastError());
                orch_runner_.reset();
                return false;
            }

            // Enable snapshot capture for parity testing
            orch_runner_->enableSnapshotCapture();

            if (isRank0())
            {
                LOG_INFO("[" << getBackendName() << " Parity] OrchestrationRunner created and initialized");
            }
            return true;
        }

        /**
         * @brief Setup a simple single-device OrchestrationRunner
         *
         * Convenience wrapper that creates a simple OrchestrationConfig
         * and calls setupOrchestrationRunner().
         *
         * @return true on success, false on failure
         */
        bool setupSimpleOrchestrationRunner()
        {
            OrchestrationConfig config;
            config.model_path = config_.model_path;
            config.device_mode = DeviceAssignmentMode::EXPLICIT;
            config.device_for_this_rank = GlobalDeviceAddress::fromLocalDeviceId(getDevice());
            config.max_seq_len = 4096;
            // No PP, no TP - simple single-device execution
            return setupOrchestrationRunner(config);
        }

        /**
         * @brief Run forward pass with whichever runner is active
         *
         * This helper allows tests to work with either the legacy IInferenceRunner
         * or the modern IOrchestrationRunner without changing test logic.
         *
         * @param tokens Input token IDs
         * @return true on success, false on failure
         */
        bool runForward(const std::vector<int32_t> &tokens)
        {
            if (orch_runner_)
            {
                // Modern path: use prefill
                if (!orch_runner_->prefill(tokens))
                {
                    LOG_ERROR("[Parity] OrchestrationRunner prefill failed: " << orch_runner_->lastError());
                    return false;
                }
                // For single-step forward (prefill only), no decode needed
                return true;
            }
            else if (runner_)
            {
                // Legacy path: use forward()
                return runner_->forward(tokens.data(), tokens.size());
            }
            LOG_ERROR("[Parity] No runner available - call setupPipeline() or setupOrchestrationRunner() first");
            return false;
        }

        /**
         * @brief Get logits from whichever runner is active
         *
         * @return Pointer to logits, or nullptr if unavailable
         */
        const float *getActiveLogits() const
        {
            if (orch_runner_)
            {
                return orch_runner_->lastLogits();
            }
            else if (runner_)
            {
                return runner_->logits();
            }
            return nullptr;
        }

        /**
         * @brief Get vocabulary size from whichever runner is active
         *
         * @return Vocabulary size, or 0 if unavailable
         */
        int getActiveVocabSize() const
        {
            if (orch_runner_)
            {
                return orch_runner_->vocabSize();
            }
            else if (runner_)
            {
                return static_cast<int>(runner_->vocab_size());
            }
            return 0;
        }

        /**
         * @brief Get snapshot from whichever runner is active
         *
         * @param key Snapshot identifier
         * @param out_size Output parameter for snapshot size
         * @return Pointer to snapshot data, or nullptr if not found
         */
        const float *activeSnapshot(const std::string &key, size_t &out_size) const
        {
            if (orch_runner_)
            {
                return orch_runner_->getSnapshot(key, out_size);
            }
            else if (runner_)
            {
                return runner_->getSnapshot(key, out_size);
            }
            out_size = 0;
            return nullptr;
        }

        /**
         * @brief Get snapshot keys from whichever runner is active
         *
         * @return Vector of snapshot keys, empty if no runner
         */
        std::vector<std::string> activeSnapshotKeys() const
        {
            if (orch_runner_)
            {
                return orch_runner_->getSnapshotKeys();
            }
            else if (runner_)
            {
                return runner_->getSnapshotKeys();
            }
            return {};
        }

        /**
         * @brief Clear KV cache on whichever runner is active
         */
        void activeClearCache()
        {
            if (orch_runner_)
            {
                orch_runner_->clearCache();
            }
            else if (runner_)
            {
                runner_->clear_cache();
            }
        }

        /**
         * @brief Clear snapshots on whichever runner is active
         */
        void activeClearSnapshots()
        {
            if (orch_runner_)
            {
                orch_runner_->clearSnapshots();
            }
            else if (runner_)
            {
                runner_->clearSnapshots();
            }
        }

        /**
         * @brief Check if an orchestration runner is active
         *
         * @return true if orch_runner_ is set, false otherwise
         */
        bool hasOrchestrationRunner() const
        {
            return orch_runner_ != nullptr;
        }

        /**
         * @brief Check if a legacy runner is active
         *
         * @return true if runner_ is set, false otherwise
         */
        bool hasLegacyRunner() const
        {
            return runner_ != nullptr;
        }

        /**
         * @brief Read decode tokens from metadata file
         */
        std::vector<int> readDecodeTokensFromMetadata()
        {
            std::string metadata_path = config_.snapshot_dir + "/metadata.txt";
            std::ifstream file(metadata_path);
            if (!file.is_open())
            {
                LOG_WARN("[Parity] Could not open metadata file: " << metadata_path);
                return {};
            }

            std::vector<int> decode_tokens;
            std::string line;
            while (std::getline(file, line))
            {
                // Look for "decode_tokens: X,Y,Z" format
                if (line.find("decode_tokens:") == 0)
                {
                    size_t colon_pos = line.find(':');
                    if (colon_pos != std::string::npos)
                    {
                        std::string tokens_str = line.substr(colon_pos + 1);
                        // Trim leading whitespace
                        size_t start = tokens_str.find_first_not_of(" \t");
                        if (start != std::string::npos)
                        {
                            tokens_str = tokens_str.substr(start);
                        }
                        // Parse comma-separated token IDs
                        std::stringstream ss(tokens_str);
                        std::string token_str;
                        while (std::getline(ss, token_str, ','))
                        {
                            // Trim whitespace from token
                            size_t tok_start = token_str.find_first_not_of(" \t");
                            size_t tok_end = token_str.find_last_not_of(" \t");
                            if (tok_start != std::string::npos && tok_end != std::string::npos)
                            {
                                token_str = token_str.substr(tok_start, tok_end - tok_start + 1);
                            }
                            if (!token_str.empty())
                            {
                                try
                                {
                                    decode_tokens.push_back(std::stoi(token_str));
                                }
                                catch (const std::exception &e)
                                {
                                    LOG_WARN("[Parity] Failed to parse token: " << token_str);
                                }
                            }
                        }
                    }
                    break; // Found decode_tokens line, done
                }
            }
            return decode_tokens;
        }

        /**
         * @brief Read prefill token IDs from metadata file
         *
         * Models with different tokenizers (e.g., Qwen3.5 vocab_size=248320 vs
         * Qwen2/3 vocab_size=151936) produce different token IDs for the same prompt.
         * This reads the actual token_ids used by the PyTorch reference generator
         * so the C++ test uses matching input.
         */
        std::vector<int> readPrefillTokensFromMetadata()
        {
            std::string metadata_path = config_.snapshot_dir + "/metadata.txt";
            std::ifstream file(metadata_path);
            if (!file.is_open())
            {
                LOG_WARN("[Parity] Could not open metadata file: " << metadata_path);
                return {};
            }

            std::vector<int> tokens;
            std::string line;
            while (std::getline(file, line))
            {
                if (line.find("token_ids:") == 0)
                {
                    size_t colon_pos = line.find(':');
                    if (colon_pos != std::string::npos)
                    {
                        std::string tokens_str = line.substr(colon_pos + 1);
                        size_t start = tokens_str.find_first_not_of(" \t");
                        if (start != std::string::npos)
                            tokens_str = tokens_str.substr(start);

                        std::stringstream ss(tokens_str);
                        std::string token_str;
                        while (std::getline(ss, token_str, ','))
                        {
                            size_t tok_start = token_str.find_first_not_of(" \t");
                            size_t tok_end = token_str.find_last_not_of(" \t");
                            if (tok_start != std::string::npos && tok_end != std::string::npos)
                                token_str = token_str.substr(tok_start, tok_end - tok_start + 1);
                            if (!token_str.empty())
                            {
                                try
                                {
                                    tokens.push_back(std::stoi(token_str));
                                }
                                catch (const std::exception &)
                                {
                                    LOG_WARN("[Parity] Failed to parse prefill token: " << token_str);
                                }
                            }
                        }
                    }
                    break;
                }
            }
            return tokens;
        }

        /**
         * @brief Run single-device prefill parity test and return summary
         *
         * This is the main test driver for SINGLE-DEVICE tests - compares
         * layer-by-layer against PyTorch. Calls setupPipeline() to create
         * a single-device runner, then delegates to runPrefillParity().
         *
         * For multi-device LOCAL TP tests, use runTPPrefillParity() instead.
         * For LocalPP tests, call setupPipeline() first, then runPrefillParity().
         */
        ParityTestSummary runSingleDevicePrefillParity()
        {
            // Setup single-device pipeline then delegate
            EXPECT_TRUE(setupPipeline()) << "Pipeline setup failed";
            return runPrefillParity();
        }

        /**
         * @brief Run prefill parity test using existing runner_
         *
         * Compares layer-by-layer Llaminar snapshots against PyTorch reference.
         * Requires runner_ to be already set up (via setupPipeline() or
         * setupLocalPPPipeline() etc).
         *
         * Use this for LocalPP tests where you want to set up the PP pipeline
         * first, then run parity comparison without overwriting the runner.
         *
         * @return ParityTestSummary with layer-by-layer metrics
         */
        ParityTestSummary runPrefillParity()
        {
            ParityTestSummary summary;

            // Require runner_ to be already set up
            if (!runner_)
            {
                LOG_ERROR("[Parity] runPrefillParity() called but runner_ is null - "
                          "ensure setupPipeline() or setupLocalPPPipeline() was called first");
                return summary;
            }

            // Run prefill
            bool success = runner_->forward(config_.token_ids.data(), config_.token_ids.size());
            EXPECT_TRUE(success) << "Prefill forward pass failed";
            if (!success)
                return summary;

            int n_layers = static_cast<int>(model_ctx_->model().block_count);

            // Stages to compare per layer
            std::vector<std::string> per_layer_stages = {
                "ATTENTION_NORM",
                // GDN sub-stages (skipped for FA layers where they don't exist)
                "QKV_PROJECTION", "GDN_Z_PROJECTION", "GDN_DELTA_RULE_OUTPUT", "GDN_NORM_GATE_OUTPUT",
                // Standard attention sub-stages (skipped for GDN layers)
                "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION",
                "Q_NORM", "K_NORM", // Qwen3 per-head QK RMSNorm (skipped if not available)
                "Q_ROPE", "K_ROPE",
                "ATTENTION_CONTEXT", "ATTENTION_OUTPUT", "ATTENTION_RESIDUAL",
                "FFN_NORM",
                // Dense FFN sub-stages (skipped for MoE layers)
                "FFN_GATE", "FFN_UP", "FFN_SWIGLU", "FFN_DOWN",
                // MoE sub-stages (skipped for dense FFN layers)
                "MOE_ROUTER_OUTPUT", "MOE_ROUTING_INDICES", "MOE_ROUTING_WEIGHTS",
                "MOE_EXPERT_OUTPUT", "MOE_SHARED_EXPERT_OUTPUT", "MOE_SHARED_GATE_OUTPUT", "MOE_COMBINED_OUTPUT",
                "FFN_RESIDUAL"};
            auto snapshot_keys = runner_->getSnapshotKeys();
            std::set<std::string> available_snapshots(snapshot_keys.begin(), snapshot_keys.end());

            // Compare embedding
            auto pytorch_embedding = loadPyTorchSnapshot("EMBEDDING");
            if (available_snapshots.count("EMBEDDING"))
            {
                size_t llaminar_size;
                const float *llaminar_data = runner_->getSnapshot("EMBEDDING", llaminar_size);

                // Debug sizes and first values (rank 0 only)
                if (isRank0())
                {
                    LOG_INFO("[Parity Debug] EMBEDDING - llaminar_size=" << llaminar_size
                                                                         << " pytorch_size=" << pytorch_embedding.size());
                    if (llaminar_data && llaminar_size >= 8)
                    {
                        LOG_INFO("[Parity Debug] Llaminar first 8: "
                                 << llaminar_data[0] << "," << llaminar_data[1] << ","
                                 << llaminar_data[2] << "," << llaminar_data[3] << ","
                                 << llaminar_data[4] << "," << llaminar_data[5] << ","
                                 << llaminar_data[6] << "," << llaminar_data[7]);
                    }
                    if (!pytorch_embedding.empty() && pytorch_embedding.size() >= 8)
                    {
                        LOG_INFO("[Parity Debug] PyTorch first 8: "
                                 << pytorch_embedding[0] << "," << pytorch_embedding[1] << ","
                                 << pytorch_embedding[2] << "," << pytorch_embedding[3] << ","
                                 << pytorch_embedding[4] << "," << pytorch_embedding[5] << ","
                                 << pytorch_embedding[6] << "," << pytorch_embedding[7]);
                    }
                }

                if (llaminar_data && !pytorch_embedding.empty())
                {
                    summary.embedding_cosine = computeCosineSimilarity(
                        llaminar_data, pytorch_embedding.data(),
                        std::min(llaminar_size, pytorch_embedding.size()));
                }
            }
            summary.embedding_passed = (summary.embedding_cosine >= config_.cosine_threshold);

            // Pre-compute GDN head config for V-head permutation
            const auto gdn_cfg = getGDNHeadConfig();
            const auto moe_cfg = getMoEConfig();
            if (gdn_cfg.needsPermutation())
            {
                LOG_INFO("[Parity] GDN V-head permutation active: n_k=" << gdn_cfg.n_k_heads
                                                                        << " n_v=" << gdn_cfg.n_v_heads
                                                                        << " d=" << gdn_cfg.d_state
                                                                        << " heads_per_group=" << gdn_cfg.headsPerGroup());
            }

            // Compare each layer
            for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx)
            {
                LayerStats stats;
                stats.layer_idx = layer_idx;
                float sum_cosine = 0.0f;

                for (const auto &stage : per_layer_stages)
                {
                    // Skip excluded stages (e.g., Q/K/V_PROJECTION for GLOBAL scope TP)
                    if (!config_.excluded_stages.empty())
                    {
                        bool is_excluded = std::find(
                                               config_.excluded_stages.begin(),
                                               config_.excluded_stages.end(),
                                               stage) != config_.excluded_stages.end();
                        if (is_excluded)
                            continue;
                    }

                    std::string llaminar_key = "layer" + std::to_string(layer_idx) + "_" + stage;
                    std::string pytorch_key = llaminar_key;

                    if (!available_snapshots.count(llaminar_key))
                        continue;

                    auto pytorch_data = loadPyTorchSnapshot(pytorch_key);
                    if (pytorch_data.empty())
                        continue;

                    size_t llaminar_size;
                    const float *llaminar_data = runner_->getSnapshot(llaminar_key, llaminar_size);
                    if (!llaminar_data)
                        continue;

                    // Apply GDN V-head permutation if needed (Llaminar ratio-grouped → PyTorch interleaved)
                    auto permuted = applyGDNHeadPermutation(llaminar_data, llaminar_size, stage, gdn_cfg);
                    const float *compare_data = permuted.empty() ? llaminar_data : permuted.data();

                    StageComparisonResult result;
                    if (stage == "MOE_ROUTING_INDICES")
                    {
                        result = compareRoutingIndices(compare_data, pytorch_data, llaminar_size,
                                                       moe_cfg.top_k, stage);
                    }
                    else if (stage == "MOE_ROUTING_WEIGHTS")
                    {
                        std::string idx_key = "layer" + std::to_string(layer_idx) + "_MOE_ROUTING_INDICES";
                        size_t ll_idx_size;
                        const float *ll_idx = runner_->getSnapshot(idx_key, ll_idx_size);
                        auto pt_idx = loadPyTorchSnapshot(idx_key);
                        if (ll_idx && !pt_idx.empty())
                            result = compareRoutingWeights(compare_data, pytorch_data, ll_idx, pt_idx,
                                                           llaminar_size, moe_cfg.top_k, moe_cfg.num_experts, stage);
                        else
                            result = compareTensors(compare_data, pytorch_data, llaminar_size, stage);
                    }
                    else
                    {
                        result = compareTensors(compare_data, pytorch_data, llaminar_size, stage);
                    }
                    stats.stages_compared++;
                    sum_cosine += result.cosine_similarity;
                    stats.stage_results.push_back(result);

                    // Per-stage cosine logging for diagnostics
                    LOG_INFO("[Parity] Layer " << layer_idx << " " << stage
                                               << " cosine=" << std::fixed << std::setprecision(6) << result.cosine_similarity
                                               << " size=" << llaminar_size);

                    // Per-row cosine diagnostic for ATTENTION_CONTEXT (layer 0 only)
                    if (stage == "ATTENTION_CONTEXT" && layer_idx == 0)
                    {
                        size_t compare_size = std::min(llaminar_size, pytorch_data.size());
                        size_t head_dim_val = model_ctx_->model().key_length > 0 ? model_ctx_->model().key_length : (model_ctx_->model().embedding_length / model_ctx_->model().head_count);
                        size_t n_cols = model_ctx_->model().head_count * head_dim_val;
                        size_t n_rows = compare_size / n_cols;
                        std::stringstream row_cosines;
                        float max_abs_diff = 0.0f;
                        size_t max_diff_idx = 0;
                        for (size_t r = 0; r < n_rows && r < 9; ++r)
                        {
                            float rc = computeCosineSimilarity(
                                llaminar_data + r * n_cols,
                                pytorch_data.data() + r * n_cols,
                                n_cols);
                            row_cosines << "row" << r << "=" << std::fixed << std::setprecision(6) << rc << " ";
                            for (size_t c = 0; c < n_cols; ++c)
                            {
                                float diff = std::abs(llaminar_data[r * n_cols + c] - pytorch_data[r * n_cols + c]);
                                if (diff > max_abs_diff)
                                {
                                    max_abs_diff = diff;
                                    max_diff_idx = r * n_cols + c;
                                }
                            }
                        }
                        LOG_INFO("[SingleGPU Diag] " << stage << " per-row cosine: " << row_cosines.str());
                        LOG_INFO("[SingleGPU Diag] " << stage
                                                     << " max_abs_diff=" << max_abs_diff
                                                     << " at idx=" << max_diff_idx
                                                     << " (row=" << max_diff_idx / n_cols
                                                     << " col=" << max_diff_idx % n_cols << ")"
                                                     << " llaminar=" << llaminar_data[max_diff_idx]
                                                     << " pytorch=" << pytorch_data[max_diff_idx]);
                    }

                    // Track kurtosis for activation outlier monitoring
                    float kurt = result.llaminar_stats.kurtosis;
                    if (kurt > stats.max_kurtosis)
                    {
                        stats.max_kurtosis = kurt;
                        stats.max_kurtosis_stage = stage;
                    }

                    if (result.cosine_similarity < stats.min_cosine_sim)
                    {
                        stats.min_cosine_sim = result.cosine_similarity;
                        stats.worst_stage = stage;
                    }
                }

                // Compute per-stage cosine drops (error introduced by each stage)
                if (stats.stage_results.size() >= 2)
                {
                    for (size_t s = 1; s < stats.stage_results.size(); ++s)
                    {
                        float drop = stats.stage_results[s - 1].cosine_similarity -
                                     stats.stage_results[s].cosine_similarity;
                        stats.stage_results[s].cosine_drop = drop;

                        if (drop > stats.max_cosine_drop)
                        {
                            stats.max_cosine_drop = drop;
                            stats.max_drop_stage = stats.stage_results[s].stage_name;
                        }
                    }
                }

                if (stats.stages_compared > 0)
                {
                    stats.avg_cosine_sim = sum_cosine / stats.stages_compared;
                }

                // Pass criteria based on config
                float check_value = config_.use_avg_cosine ? stats.avg_cosine_sim : stats.min_cosine_sim;
                stats.passed = (check_value >= config_.cosine_threshold);

                summary.layer_stats.push_back(stats);
            }

            // Compare LM_HEAD
            auto pytorch_lm_head = loadPyTorchSnapshot("LM_HEAD");
            if (available_snapshots.count("LM_HEAD") && !pytorch_lm_head.empty())
            {
                size_t llaminar_size;
                const float *llaminar_data = runner_->getSnapshot("LM_HEAD", llaminar_size);
                if (llaminar_data)
                {
                    size_t vocab_size = model_ctx_->model().vocab_size;
                    size_t llaminar_seq_len = llaminar_size / vocab_size;
                    size_t pytorch_seq_len = pytorch_lm_head.size() / vocab_size;

                    // Llaminar may only output the last row (last-token-only optimization).
                    // PyTorch always outputs all rows. Compare the last row from each.
                    size_t llaminar_last_offset = (llaminar_seq_len > 0) ? (llaminar_seq_len - 1) * vocab_size : 0;
                    size_t pytorch_last_offset = (pytorch_seq_len > 0) ? (pytorch_seq_len - 1) * vocab_size : 0;

                    // Cosine similarity on last-row logits only
                    auto result = compareTensors(
                        llaminar_data + llaminar_last_offset,
                        std::vector<float>(pytorch_lm_head.begin() + pytorch_last_offset,
                                           pytorch_lm_head.begin() + pytorch_last_offset + vocab_size),
                        vocab_size, "LM_HEAD");
                    summary.lm_head_cosine = result.cosine_similarity;

                    if (llaminar_seq_len > 0 && pytorch_seq_len > 0)
                    {
                        summary.lm_head_kl = computeKLDivergence(
                            llaminar_data + llaminar_last_offset,
                            pytorch_lm_head.data() + pytorch_last_offset,
                            vocab_size, vocab_size);

                        summary.lm_head_top1 = computeTopKOverlap(
                            llaminar_data + llaminar_last_offset,
                            pytorch_lm_head.data() + pytorch_last_offset,
                            vocab_size, vocab_size, 1);

                        summary.lm_head_top5 = computeTopKOverlap(
                            llaminar_data + llaminar_last_offset,
                            pytorch_lm_head.data() + pytorch_last_offset,
                            vocab_size, vocab_size, 5);

                        // Check if PyTorch's top-1 token is in llaminar's top-K
                        if (config_.pytorch_top1_in_topk > 0)
                        {
                            float topk_recall = pytorchTop1InLlaminarTopK(
                                llaminar_data + llaminar_last_offset,
                                pytorch_lm_head.data() + pytorch_last_offset,
                                vocab_size, vocab_size, config_.pytorch_top1_in_topk);
                            summary.lm_head_pytorch_top1_in_top3 = (topk_recall >= 1.0f - 1e-6f);
                        }
                        else
                        {
                            summary.lm_head_pytorch_top1_in_top3 = true; // gate disabled
                        }
                    }
                }
            }
            summary.lm_head_passed = (summary.lm_head_kl < config_.kl_threshold) &&
                                     ((summary.lm_head_top1 * 100.0f) >= config_.min_top1_accuracy) &&
                                     ((summary.lm_head_top5 * 100.0f) >= config_.min_top5_accuracy) &&
                                     (config_.pytorch_top1_in_topk <= 0 || summary.lm_head_pytorch_top1_in_top3);

            // Count early layers passed
            summary.early_layers_passed = summary.embedding_passed ? 1 : 0;
            for (int i = 0; i < std::min(config_.early_layers_count, static_cast<int>(summary.layer_stats.size())); ++i)
            {
                if (summary.layer_stats[i].passed)
                    summary.early_layers_passed++;
            }

            // Count total layers passed
            summary.total_layers_passed = summary.embedding_passed ? 1 : 0;
            for (const auto &stats : summary.layer_stats)
            {
                if (stats.passed)
                    summary.total_layers_passed++;
            }

            // Overall pass
            summary.overall_passed = (summary.early_layers_passed >= config_.min_early_layers_passed) &&
                                     summary.lm_head_passed;

            return summary;
        }

        /**
         * @brief Run TP-aware prefill parity test
         *
         * For tensor-parallel tests, this compares:
         * 1. Per-device partial outputs against corresponding PyTorch slices
         * 2. Combined (concatenated) outputs against full PyTorch reference
         *
         * Requires runner_ to be a RankOrchestrator (will cast and check).
         *
         * @return TPParityTestSummary with per-device and combined metrics
         */
        TPParityTestSummary runTPPrefillParity()
        {
            TPParityTestSummary summary;

            // Only setup pipeline if not already configured
            // (Test may have already called setupLocalTPPipeline() or similar)
            if (!runner_)
            {
                EXPECT_TRUE(setupPipeline()) << "Pipeline setup failed";
                if (!runner_)
                    return summary;
            }

            // Try to cast to RankOrchestrator for TP snapshot access
            auto *multi_device = dynamic_cast<RankOrchestrator *>(runner_.get());
            if (!multi_device)
            {
                LOG_ERROR("[TP Parity] runner_ is not a RankOrchestrator - "
                          "ensure test calls setupLocalTPPipeline() or similar before runTPPrefillParity()");
                return summary;
            }

            summary.tp_degree = multi_device->device_count();
            for (int i = 0; i < summary.tp_degree; ++i)
            {
                auto *dev_runner = multi_device->deviceRunner(i);
                if (dev_runner)
                {
                    // Use index-based name for simplicity
                    // RankOrchestrator stores device info in config
                    summary.device_names.push_back("TP_rank_" + std::to_string(i));
                }
            }

            LOG_INFO("[TP Parity] Running with " << summary.tp_degree << " devices");

            // Run prefill
            LOG_INFO("[TP Parity] Calling forward() with " << config_.token_ids.size() << " tokens...");
            bool success = runner_->forward(config_.token_ids.data(), config_.token_ids.size());
            LOG_INFO("[TP Parity] forward() returned: " << (success ? "SUCCESS" : "FAILURE"));
            EXPECT_TRUE(success) << "Prefill forward pass failed";
            if (!success)
                return summary;

            int n_layers = static_cast<int>(model_ctx_->model().block_count);
            size_t seq_len = config_.token_ids.size();
            size_t d_model = model_ctx_->model().embedding_length;
            size_t n_heads = model_ctx_->headCount();
            size_t d_ff = model_ctx_->feedForwardLength();
            size_t vocab_size = model_ctx_->model().vocab_size;

            // Stages to compare per layer with their expected dimensions
            struct StageInfo
            {
                std::string name;
                size_t cols; // Expected columns (may be sharded)
            };
            size_t n_kv_heads = model_ctx_->headCountKV();
            size_t head_dim = d_model / n_heads;
            size_t kv_dim = n_kv_heads * head_dim;
            std::vector<StageInfo> per_layer_stages = {
                {"ATTENTION_NORM", d_model},
                {"Q_PROJECTION", d_model},      // DIAGNOSTIC: check QKV GEMM output
                {"K_PROJECTION", kv_dim},       // DIAGNOSTIC: check K projection
                {"V_PROJECTION", kv_dim},       // DIAGNOSTIC: check V projection
                {"Q_ROPE", d_model},            // DIAGNOSTIC: check RoPE output
                {"K_ROPE", kv_dim},             // DIAGNOSTIC: check K RoPE output
                {"ATTENTION_CONTEXT", d_model}, // num_heads * head_dim = d_model
                {"ATTENTION_OUTPUT", d_model},
                {"FFN_NORM", d_model},
                {"FFN_SWIGLU", d_ff},
                {"FFN_DOWN", d_model},
                {"FFN_RESIDUAL", d_model}};

            // Get snapshot keys
            auto snapshot_keys = runner_->getSnapshotKeys();
            std::set<std::string> available_snapshots(snapshot_keys.begin(), snapshot_keys.end());
            LOG_INFO("[TP Parity] Got " << snapshot_keys.size() << " snapshot keys after forward()");
            if (snapshot_keys.size() < 50)
            {
                LOG_WARN("[TP Parity] WARNING: Expected 200+ snapshot keys, got only " << snapshot_keys.size());
                for (const auto &key : snapshot_keys)
                {
                    LOG_INFO("[TP Parity]   Available: " << key);
                }
            }

            // Compare embedding
            auto pytorch_embedding = loadPyTorchSnapshot("EMBEDDING");
            if (available_snapshots.count("EMBEDDING") && !pytorch_embedding.empty())
            {
                TPSnapshot tp_snap = multi_device->getTPSnapshot("EMBEDDING");
                summary.embedding_result = compareTPSnapshot(
                    tp_snap, pytorch_embedding, seq_len, d_model);
            }

            // Compare each layer
            for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx)
            {
                TPLayerStats layer_stats;
                layer_stats.layer_idx = layer_idx;
                layer_stats.tp_degree = summary.tp_degree;

                float sum_combined_cosine = 0.0f;

                for (const auto &stage_info : per_layer_stages)
                {
                    std::string llaminar_key = "layer" + std::to_string(layer_idx) + "_" + stage_info.name;
                    std::string pytorch_key = llaminar_key;

                    if (!available_snapshots.count(llaminar_key))
                        continue;

                    auto pytorch_data = loadPyTorchSnapshot(pytorch_key);
                    if (pytorch_data.empty())
                        continue;

                    TPSnapshot tp_snap = multi_device->getTPSnapshot(llaminar_key);
                    auto stage_result = compareTPSnapshot(
                        tp_snap, pytorch_data, seq_len, stage_info.cols);

                    layer_stats.stage_results.push_back(stage_result);
                    layer_stats.stages_compared++;
                    sum_combined_cosine += stage_result.combined_cosine;

                    if (stage_result.combined_cosine < layer_stats.min_combined_cosine)
                    {
                        layer_stats.min_combined_cosine = stage_result.combined_cosine;
                        layer_stats.worst_stage = stage_info.name;
                    }
                }

                if (layer_stats.stages_compared > 0)
                {
                    layer_stats.avg_combined_cosine = sum_combined_cosine / layer_stats.stages_compared;
                }

                // ====================================================================
                // CPU-side RoPE crosscheck for layer 0 to diagnose TP Q_ROPE divergence
                // ====================================================================
                if (layer_idx == 0)
                {
                    auto pytorch_qp = loadPyTorchSnapshot("layer0_Q_PROJECTION");
                    auto pytorch_qr = loadPyTorchSnapshot("layer0_Q_ROPE");
                    TPSnapshot tp_qp = multi_device->getTPSnapshot("layer0_Q_PROJECTION");
                    TPSnapshot tp_qr = multi_device->getTPSnapshot("layer0_Q_ROPE");

                    if (!tp_qp.device_data.empty() && !tp_qr.device_data.empty() &&
                        !pytorch_qp.empty() && !pytorch_qr.empty())
                    {
                        const auto &qp_dev0 = tp_qp.device_data[0].data;
                        const auto &qr_dev0 = tp_qr.device_data[0].data;
                        size_t local_cols = tp_qp.device_data[0].cols;

                        if (local_cols > 0 && qp_dev0.size() >= seq_len * local_cols &&
                            qr_dev0.size() >= seq_len * local_cols)
                        {
                            float theta_base = model_ctx_->model().rope_theta;
                            size_t hd = head_dim;
                            size_t half_hd = hd / 2;

                            LOG_INFO("[RoPE CrossCheck] theta_base=" << theta_base
                                                                     << " head_dim=" << hd << " local_cols=" << local_cols
                                                                     << " pytorch_qp_cols=" << d_model);

                            // Check multiple RoPE pairs across different rows and heads
                            for (size_t row : {0ul, 4ul, 8ul})
                            {
                                if (row >= seq_len)
                                    break;
                                for (size_t pair_i : {0ul, 4ul, 15ul, 31ul})
                                {
                                    if (pair_i >= half_hd)
                                        continue;
                                    // Check head 0 and head 1
                                    for (size_t h : {0ul, 1ul})
                                    {
                                        size_t col_i0 = h * hd + pair_i;           // first half
                                        size_t col_i1 = h * hd + pair_i + half_hd; // second half
                                        if (col_i1 >= local_cols)
                                            continue;

                                        size_t idx_i0 = row * local_cols + col_i0;
                                        size_t idx_i1 = row * local_cols + col_i1;

                                        // Llaminar Q_PROJ values
                                        float x0_ll = qp_dev0[idx_i0];
                                        float x1_ll = qp_dev0[idx_i1];

                                        // PyTorch Q_PROJ values (full 896 cols)
                                        float x0_pt = pytorch_qp[row * d_model + col_i0];
                                        float x1_pt = pytorch_qp[row * d_model + col_i1];

                                        // Compute expected RoPE
                                        float inv_freq_i = std::exp(-std::log(theta_base) * 2.0f * static_cast<float>(pair_i) / static_cast<float>(hd));
                                        float angle = static_cast<float>(row) * inv_freq_i;
                                        float cos_a = std::cos(angle);
                                        float sin_a = std::sin(angle);

                                        float cpu_i0 = x0_ll * cos_a - x1_ll * sin_a;
                                        float cpu_i1 = x0_ll * sin_a + x1_ll * cos_a;

                                        // Actual outputs
                                        float ll_i0 = qr_dev0[idx_i0];
                                        float ll_i1 = qr_dev0[idx_i1];
                                        float pt_i0 = pytorch_qr[row * d_model + col_i0];
                                        float pt_i1 = pytorch_qr[row * d_model + col_i1];

                                        float diff_i0 = std::abs(ll_i0 - pt_i0);
                                        float diff_i1 = std::abs(ll_i1 - pt_i1);

                                        if (diff_i0 > 0.01f || diff_i1 > 0.01f)
                                        {
                                            LOG_INFO("[RoPE CrossCheck] MISMATCH row=" << row
                                                                                       << " head=" << h << " pair=" << pair_i
                                                                                       << " | Q_PROJ: ll_x0=" << x0_ll << " pt_x0=" << x0_pt
                                                                                       << " ll_x1=" << x1_ll << " pt_x1=" << x1_pt
                                                                                       << " | inv_freq=" << inv_freq_i << " angle=" << angle
                                                                                       << " cos=" << cos_a << " sin=" << sin_a
                                                                                       << " | CPU_expected: i0=" << cpu_i0 << " i1=" << cpu_i1
                                                                                       << " | GPU_actual: i0=" << ll_i0 << " i1=" << ll_i1
                                                                                       << " | PyTorch: i0=" << pt_i0 << " i1=" << pt_i1
                                                                                       << " | diff_i0=" << diff_i0 << " diff_i1=" << diff_i1);
                                        }
                                    }
                                }
                            }
                            // Summary: compute full CPU RoPE on row 8 and measure cosine
                            if (seq_len > 8)
                            {
                                std::vector<float> cpu_rope_row8(local_cols);
                                const float *qp_row8 = qp_dev0.data() + 8 * local_cols;
                                size_t local_heads = local_cols / hd;
                                for (size_t h = 0; h < local_heads; ++h)
                                {
                                    for (size_t i = 0; i < half_hd; ++i)
                                    {
                                        float inv_f = std::exp(-std::log(theta_base) * 2.0f * static_cast<float>(i) / static_cast<float>(hd));
                                        float ang = 8.0f * inv_f;
                                        float c = std::cos(ang);
                                        float s = std::sin(ang);
                                        float x0 = qp_row8[h * hd + i];
                                        float x1 = qp_row8[h * hd + i + half_hd];
                                        cpu_rope_row8[h * hd + i] = x0 * c - x1 * s;
                                        cpu_rope_row8[h * hd + i + half_hd] = x0 * s + x1 * c;
                                    }
                                }
                                float cos_cpu_vs_ll = computeCosineSimilarity(
                                    cpu_rope_row8.data(), qr_dev0.data() + 8 * local_cols, local_cols);
                                float cos_cpu_vs_pt = computeCosineSimilarity(
                                    cpu_rope_row8.data(),
                                    extractColumnSlice(pytorch_qr.data(), seq_len, d_model, 0, local_cols).data() + 8 * local_cols,
                                    local_cols);
                                LOG_INFO("[RoPE CrossCheck] Row8 cosine: CPU_vs_llaminar=" << std::fixed << std::setprecision(6) << cos_cpu_vs_ll
                                                                                           << " CPU_vs_pytorch=" << cos_cpu_vs_pt);
                            }
                        }
                    }
                }

                // Pass criteria based on combined cosine
                layer_stats.passed = (layer_stats.avg_combined_cosine >= config_.cosine_threshold);

                summary.layer_stats.push_back(layer_stats);
            }

            // Compare LM_HEAD (always gathered)
            auto pytorch_lm_head = loadPyTorchSnapshot("LM_HEAD");
            if (available_snapshots.count("LM_HEAD") && !pytorch_lm_head.empty())
            {
                TPSnapshot tp_snap = multi_device->getTPSnapshot("LM_HEAD");

                size_t combined_size = 0;
                const float *combined_ptr = tp_snap.getCombinedData(combined_size);

                if (combined_ptr && combined_size > 0)
                {
                    // Llaminar may only output the last row (last-token-only optimization).
                    // PyTorch always outputs all rows. Compare the last row from each.
                    size_t llaminar_seq_len = combined_size / vocab_size;
                    size_t pytorch_seq_len = pytorch_lm_head.size() / vocab_size;
                    size_t llaminar_last_offset = (llaminar_seq_len > 0) ? (llaminar_seq_len - 1) * vocab_size : 0;
                    size_t pytorch_last_offset = (pytorch_seq_len > 0) ? (pytorch_seq_len - 1) * vocab_size : 0;

                    // Cosine similarity on last-row logits only
                    summary.lm_head_cosine = computeCosineSimilarity(
                        combined_ptr + llaminar_last_offset,
                        pytorch_lm_head.data() + pytorch_last_offset, vocab_size);

                    if (llaminar_seq_len > 0 && pytorch_seq_len > 0)
                    {
                        summary.lm_head_kl = computeKLDivergence(
                            combined_ptr + llaminar_last_offset,
                            pytorch_lm_head.data() + pytorch_last_offset,
                            vocab_size, vocab_size);

                        summary.lm_head_top1 = computeTopKOverlap(
                            combined_ptr + llaminar_last_offset,
                            pytorch_lm_head.data() + pytorch_last_offset,
                            vocab_size, vocab_size, 1);

                        summary.lm_head_top5 = computeTopKOverlap(
                            combined_ptr + llaminar_last_offset,
                            pytorch_lm_head.data() + pytorch_last_offset,
                            vocab_size, vocab_size, 5);

                        // Check if PyTorch's top-1 token is in llaminar's top-K
                        if (config_.pytorch_top1_in_topk > 0)
                        {
                            float topk_recall = pytorchTop1InLlaminarTopK(
                                combined_ptr + llaminar_last_offset,
                                pytorch_lm_head.data() + pytorch_last_offset,
                                vocab_size, vocab_size, config_.pytorch_top1_in_topk);
                            summary.lm_head_pytorch_top1_in_top3 = (topk_recall >= 1.0f - 1e-6f);
                        }
                        else
                        {
                            summary.lm_head_pytorch_top1_in_top3 = true; // gate disabled
                        }
                    }
                }
            }
            summary.lm_head_passed = (summary.lm_head_kl < config_.kl_threshold) &&
                                     ((summary.lm_head_top1 * 100.0f) >= config_.min_top1_accuracy) &&
                                     ((summary.lm_head_top5 * 100.0f) >= config_.min_top5_accuracy) &&
                                     (config_.pytorch_top1_in_topk <= 0 || summary.lm_head_pytorch_top1_in_top3);

            // Count early layers passed
            summary.early_layers_passed = summary.embedding_result.passed ? 1 : 0;
            for (int i = 0; i < std::min(config_.early_layers_count, static_cast<int>(summary.layer_stats.size())); ++i)
            {
                if (summary.layer_stats[i].passed)
                    summary.early_layers_passed++;
            }

            // Count total layers passed
            summary.total_layers_passed = summary.embedding_result.passed ? 1 : 0;
            for (const auto &stats : summary.layer_stats)
            {
                if (stats.passed)
                    summary.total_layers_passed++;
            }

            // Overall pass
            summary.overall_passed = (summary.early_layers_passed >= config_.min_early_layers_passed) &&
                                     summary.lm_head_passed;

            return summary;
        }

        /**
         * @brief Run TP-aware incremental decode parity test
         *
         * Tests autoregressive generation by comparing LM_HEAD outputs at each
         * decode step against PyTorch reference. For TP tests, the multi-device
         * orchestrator handles logit gathering internally.
         *
         * Requires:
         * - runner_ to be pre-configured (typically by setupLocalTPPipeline())
         * - PyTorch snapshots with decode_step{N}_LM_HEAD.npy files
         * - metadata.txt with decode_tokens line
         *
         * This is for MULTI-DEVICE LOCAL TP tests.
         * For single-device tests, use runSingleDeviceDecodeParity() instead.
         *
         * @return DecodeParitySummary with per-step and aggregate results
         */
        DecodeParitySummary runTPDecodeParity()
        {
            DecodeParitySummary summary;

            // TP tests require pre-configured runner
            if (!runner_)
            {
                LOG_ERROR("[TP Decode Parity] runner_ is null - "
                          "ensure test calls setupLocalTPPipeline() or similar before runTPDecodeParity()");
                return summary;
            }

            // Verify we have a multi-device orchestrator
            auto *multi_device = dynamic_cast<RankOrchestrator *>(runner_.get());
            if (!multi_device)
            {
                LOG_ERROR("[TP Decode Parity] runner_ is not a RankOrchestrator - "
                          "ensure test calls setupLocalTPPipeline() or similar before runTPDecodeParity()");
                return summary;
            }

            // Check if decode snapshots exist
            auto decode_step0 = loadPyTorchSnapshot("decode_step0_LM_HEAD");
            if (decode_step0.empty())
            {
                LOG_WARN("Decode snapshots not found - skipping decode parity test");
                return summary;
            }

            // Read expected tokens from metadata
            std::vector<int> pytorch_decode_tokens = readDecodeTokensFromMetadata();
            if (pytorch_decode_tokens.empty())
            {
                LOG_WARN("No decode_tokens in metadata - skipping decode parity test");
                return summary;
            }

            // Run prefill first (required to initialize KV cache)
            bool success = runner_->forward(config_.token_ids.data(), config_.token_ids.size());
            EXPECT_TRUE(success) << "Prefill failed";
            if (!success)
                return summary;

            size_t vocab_size = model_ctx_->model().vocab_size;

            // Process each decode step
            size_t num_decode_steps = std::min(pytorch_decode_tokens.size(),
                                               static_cast<size_t>(config_.decode_steps));

            float sum_cosine = 0.0f;
            float sum_kl = 0.0f;

            for (size_t step = 0; step < num_decode_steps; ++step)
            {
                std::string step_prefix = "decode_step" + std::to_string(step);

                // Load PyTorch reference for this step
                auto pytorch_lm_head = loadPyTorchSnapshot(step_prefix + "_LM_HEAD");
                if (pytorch_lm_head.empty())
                {
                    break; // No more decode snapshots
                }

                summary.steps_total++;

                // Get token for this decode step
                int current_token = pytorch_decode_tokens[step];

                // Clear snapshots from previous step
                runner_->clearSnapshots();

                // Run single-token decode
                std::vector<int> decode_token = {current_token};
                success = runner_->forward(decode_token.data(), 1);
                EXPECT_TRUE(success) << "Decode step " << step << " failed";
                if (!success)
                    continue;

                // Get Llaminar's logits (RankOrchestrator gathers from all devices)
                const float *llaminar_logits = runner_->logits();
                if (!llaminar_logits)
                {
                    LOG_WARN("No logits for decode step " << step);
                    continue;
                }

                // Python decode snapshots may contain the full sequence
                // (shape [1, seq_len, vocab]) when generated via full-sequence
                // forward. Extract the LAST position's logits for comparison.
                const float *pytorch_logits = pytorch_lm_head.data();
                size_t pytorch_logits_count = pytorch_lm_head.size();
                if (pytorch_logits_count > vocab_size)
                {
                    size_t pytorch_seq_len = pytorch_logits_count / vocab_size;
                    size_t last_offset = (pytorch_seq_len - 1) * vocab_size;
                    pytorch_logits = pytorch_lm_head.data() + last_offset;
                    pytorch_logits_count = vocab_size;
                }

                // Compare with PyTorch
                DecodeStepStats step_stats;
                step_stats.step_idx = static_cast<int>(step);

                step_stats.cosine_similarity = computeCosineSimilarity(
                    llaminar_logits, pytorch_logits,
                    std::min(vocab_size, pytorch_logits_count));

                step_stats.kl_divergence = computeKLDivergence(
                    llaminar_logits, pytorch_logits,
                    vocab_size, vocab_size);

                step_stats.top1_overlap = computeTopKOverlap(
                    llaminar_logits, pytorch_logits,
                    vocab_size, vocab_size, 1);

                step_stats.top5_overlap = computeTopKOverlap(
                    llaminar_logits, pytorch_logits,
                    vocab_size, vocab_size, 5);

                // Find argmax tokens
                step_stats.llaminar_token = 0;
                step_stats.pytorch_token = 0;
                float max_l = llaminar_logits[0];
                float max_p = pytorch_logits[0];

                for (size_t i = 1; i < vocab_size; ++i)
                {
                    if (llaminar_logits[i] > max_l)
                    {
                        max_l = llaminar_logits[i];
                        step_stats.llaminar_token = static_cast<int>(i);
                    }
                    if (pytorch_logits[i] > max_p)
                    {
                        max_p = pytorch_logits[i];
                        step_stats.pytorch_token = static_cast<int>(i);
                    }
                }

                step_stats.token_match = (step_stats.llaminar_token == step_stats.pytorch_token);
                if (step_stats.token_match)
                    summary.top1_matches++;

                // Check if PyTorch's top-1 token appears in Llaminar's top-5
                step_stats.top5_match = (step_stats.top5_overlap >= 0.2f - 1e-6f); // At least 1/5 overlap
                if (step_stats.top5_match)
                    summary.top5_matches++;

                // Check if PyTorch's top-1 token appears in Llaminar's top-K
                if (config_.pytorch_top1_in_topk > 0)
                {
                    float topk_recall = pytorchTop1InLlaminarTopK(
                        llaminar_logits, pytorch_logits,
                        vocab_size, vocab_size, config_.pytorch_top1_in_topk);
                    step_stats.top3_match = (topk_recall >= 1.0f - 1e-6f);
                }
                else
                {
                    step_stats.top3_match = true; // gate disabled
                }
                if (step_stats.top3_match)
                    summary.top3_matches++;

                // Pass criteria: either cosine >= threshold OR KL < threshold
                step_stats.passed = (step_stats.cosine_similarity >= config_.decode_cosine_threshold) ||
                                    (step_stats.kl_divergence < config_.kl_threshold);

                if (step_stats.passed)
                {
                    summary.steps_passed++;
                }

                sum_cosine += step_stats.cosine_similarity;
                sum_kl += step_stats.kl_divergence;

                summary.step_stats.push_back(step_stats);
            }

            // Compute averages and top1/top5 accuracy
            if (summary.steps_total > 0)
            {
                summary.avg_cosine = sum_cosine / summary.steps_total;
                summary.avg_kl = sum_kl / summary.steps_total;
                summary.top1_accuracy = 100.0f * summary.top1_matches / summary.steps_total;
                summary.top3_accuracy = 100.0f * summary.top3_matches / summary.steps_total;
                summary.top5_accuracy = 100.0f * summary.top5_matches / summary.steps_total;
            }

            // Overall pass criteria
            int min_steps_required = static_cast<int>(summary.steps_total * config_.min_decode_pass_rate);
            bool topk_gate = (config_.pytorch_top1_in_topk <= 0) ||
                             (summary.top3_matches == summary.steps_total);
            summary.overall_passed = (summary.steps_passed >= min_steps_required) &&
                                     (summary.top5_accuracy >= config_.min_top5_accuracy) &&
                                     (summary.avg_cosine >= config_.decode_cosine_threshold) &&
                                     topk_gate;

            return summary;
        }

        /**
         * @brief Assert TP parity criteria and render table
         *
         * Call this after runTPPrefillParity() to apply assertions.
         */
        void assertTPParity(const TPParityTestSummary &summary)
        {
            // Render the TP-aware table (rank 0 only)
            if (isRank0())
            {
                renderTPParityTable(summary, config_, getBackendName());
            }

            // Assertions
            EXPECT_GE(summary.early_layers_passed, config_.min_early_layers_passed)
                << "At least " << config_.min_early_layers_passed << " of the first "
                << config_.early_layers_count << " layers should pass TP parity";

            EXPECT_LT(summary.lm_head_kl, config_.kl_threshold)
                << "LM_HEAD KL divergence too high: " << summary.lm_head_kl;

            EXPECT_GE(summary.lm_head_top1 * 100.0f, config_.min_top1_accuracy)
                << "LM_HEAD Top-1 accuracy too low: " << (summary.lm_head_top1 * 100.0f)
                << "% (required: " << config_.min_top1_accuracy << "%)";

            EXPECT_GE(summary.lm_head_top5 * 100.0f, config_.min_top5_accuracy)
                << "LM_HEAD Top-5 accuracy too low: " << (summary.lm_head_top5 * 100.0f)
                << "% (required: " << config_.min_top5_accuracy << "%)";

            if (config_.pytorch_top1_in_topk > 0)
            {
                EXPECT_TRUE(summary.lm_head_pytorch_top1_in_top3)
                    << "PyTorch's top-1 token must appear in llaminar's top-" << config_.pytorch_top1_in_topk << " for LM_HEAD";
            }
        }

        /**
         * @brief Assert standard parity criteria
         *
         * Call this after runSingleDevicePrefillParity() to apply standard assertions.
         * MPI-aware: Only renders table on rank 0.
         */
        void assertParity(const ParityTestSummary &summary)
        {
            // Render the table first (rank 0 only)
            if (isRank0())
            {
                renderParityTable(summary, config_, getBackendName());
            }

            // Export CSV results
            exportPrefillCSV(summary);

            // For cross-rank PP, each rank only validates assertions relevant to its stage:
            // - Head rank: has early layers and embedding, validates early_layers_passed
            // - Tail rank: has LM_HEAD logits, validates logit metrics
            // Detection: if LM_HEAD summary is all-zero and we have a multi-rank MPI context,
            // this rank likely lacks the LM head. Similarly for early layers.
            const bool has_lm_head_data = (summary.lm_head_cosine > 0.0f || summary.lm_head_top1 > 0.0f);
            const bool has_early_layer_data = (summary.early_layers_passed > 0 || summary.embedding_passed);

            // Early layer assertions (skip if this rank has no early layer data, e.g. PP tail)
            if (has_early_layer_data)
            {
                EXPECT_GE(summary.early_layers_passed, config_.min_early_layers_passed)
                    << "At least " << config_.min_early_layers_passed << " of the first "
                    << config_.early_layers_count << " layers should pass parity (cosine >= "
                    << config_.cosine_threshold << ")";
            }

            // LM_HEAD assertions (skip if this rank has no LM_HEAD data, e.g. PP head)
            if (has_lm_head_data)
            {
                EXPECT_LT(summary.lm_head_kl, config_.kl_threshold)
                    << "LM_HEAD KL divergence too high: " << summary.lm_head_kl
                    << " (threshold: " << config_.kl_threshold << ")";

                EXPECT_GE(summary.lm_head_top1 * 100.0f, config_.min_top1_accuracy)
                    << "LM_HEAD Top-1 accuracy too low: " << (summary.lm_head_top1 * 100.0f)
                    << "% (required: " << config_.min_top1_accuracy << "%)";

                EXPECT_GE(summary.lm_head_top5 * 100.0f, config_.min_top5_accuracy)
                    << "LM_HEAD Top-5 accuracy too low: " << (summary.lm_head_top5 * 100.0f)
                    << "% (required: " << config_.min_top5_accuracy << "%)";

                if (config_.pytorch_top1_in_topk > 0)
                {
                    EXPECT_TRUE(summary.lm_head_pytorch_top1_in_top3)
                        << "PyTorch's top-1 token must appear in llaminar's top-" << config_.pytorch_top1_in_topk << " for LM_HEAD";
                }
            }
        }

        // =========================================================================
        // Incremental Decode Parity Testing
        // =========================================================================

        /**
         * @brief Run single-device incremental decode parity test
         *
         * Tests autoregressive generation by comparing LM_HEAD outputs at each
         * decode step against PyTorch reference. Requires:
         * - PyTorch snapshots with decode_step{N}_LM_HEAD.npy files
         * - metadata.txt with decode_tokens line
         *
         * This is for SINGLE-DEVICE tests. Calls setupPipeline() then delegates
         * to runDecodeParity().
         * For multi-device LOCAL TP tests, use runTPDecodeParity() instead.
         * For LocalPP tests, call setupPipeline() first, then runDecodeParity().
         *
         * @return DecodeParitySummary with per-step and aggregate results
         */
        DecodeParitySummary runSingleDeviceDecodeParity()
        {
            // Setup single-device pipeline then delegate
            EXPECT_TRUE(setupPipeline()) << "Pipeline setup failed";
            return runDecodeParity();
        }

        /**
         * @brief Run decode parity test using existing runner_
         *
         * Compares incremental decode logits against PyTorch reference snapshots.
         * Requires runner_ to be already set up (via setupPipeline() or
         * setupLocalPPPipeline() etc).
         *
         * Use this for LocalPP tests where you want to set up the PP pipeline
         * first, then run parity comparison without overwriting the runner.
         *
         * @return DecodeParitySummary with per-step and aggregate results
         */
        DecodeParitySummary runDecodeParity()
        {
            DecodeParitySummary summary;

            // Stages to compare per layer during decode (same as prefill)
            const std::vector<std::string> decode_per_layer_stages = {
                "ATTENTION_NORM",
                "QKV_PROJECTION", "GDN_Z_PROJECTION", "GDN_DELTA_RULE_OUTPUT", "GDN_NORM_GATE_OUTPUT",
                "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION",
                "Q_NORM", "K_NORM",
                "Q_ROPE", "K_ROPE",
                "ATTENTION_CONTEXT", "ATTENTION_OUTPUT", "ATTENTION_RESIDUAL",
                "FFN_NORM",
                "FFN_GATE", "FFN_UP", "FFN_SWIGLU", "FFN_DOWN",
                "MOE_ROUTER_OUTPUT", "MOE_ROUTING_INDICES", "MOE_ROUTING_WEIGHTS",
                "MOE_EXPERT_OUTPUT", "MOE_SHARED_EXPERT_OUTPUT", "MOE_SHARED_GATE_OUTPUT", "MOE_COMBINED_OUTPUT",
                "FFN_RESIDUAL"};

            // Check if decode snapshots exist
            auto decode_step0 = loadPyTorchSnapshot("decode_step0_LM_HEAD");
            if (decode_step0.empty())
            {
                LOG_WARN("Decode snapshots not found - skipping decode parity test");
                return summary;
            }

            // Read expected tokens from metadata
            std::vector<int> pytorch_decode_tokens = readDecodeTokensFromMetadata();
            if (pytorch_decode_tokens.empty())
            {
                LOG_WARN("No decode_tokens in metadata - skipping decode parity test");
                return summary;
            }

            // Require runner_ to be already set up
            if (!runner_)
            {
                LOG_ERROR("[Parity] runDecodeParity() called but runner_ is null - "
                          "ensure setupPipeline() or setupLocalPPPipeline() was called first");
                return summary;
            }

            // Run prefill first (required to initialize KV cache)
            bool success = runner_->forward(config_.token_ids.data(), config_.token_ids.size());
            EXPECT_TRUE(success) << "Prefill failed";
            if (!success)
                return summary;

            size_t vocab_size = model_ctx_->model().vocab_size;

            // Process each decode step
            size_t num_decode_steps = std::min(pytorch_decode_tokens.size(),
                                               static_cast<size_t>(config_.decode_steps));

            float sum_cosine = 0.0f;
            float sum_kl = 0.0f;

            for (size_t step = 0; step < num_decode_steps; ++step)
            {
                std::string step_prefix = "decode_step" + std::to_string(step);

                // Load PyTorch reference for this step
                auto pytorch_lm_head = loadPyTorchSnapshot(step_prefix + "_LM_HEAD");
                if (pytorch_lm_head.empty())
                {
                    break; // No more decode snapshots
                }

                summary.steps_total++;

                // Get token for this decode step
                int current_token = pytorch_decode_tokens[step];

                // Clear snapshots from previous step
                runner_->clearSnapshots();

                // Run single-token decode
                std::vector<int> decode_token = {current_token};
                success = runner_->forward(decode_token.data(), 1);
                EXPECT_TRUE(success) << "Decode step " << step << " failed";
                if (!success)
                    continue;

                // Get Llaminar's LM_HEAD output
                size_t decode_logits_size;
                const float *llaminar_logits = runner_->getSnapshot("LM_HEAD", decode_logits_size);
                if (!llaminar_logits)
                {
                    LOG_WARN("No LM_HEAD snapshot for decode step " << step);
                    continue;
                }

                // Python decode snapshots may contain the full sequence
                // (shape [1, seq_len, vocab]) when generated via full-sequence
                // forward. Extract the LAST position's logits for comparison.
                const float *pytorch_logits = pytorch_lm_head.data();
                size_t pytorch_logits_count = pytorch_lm_head.size();
                if (pytorch_logits_count > vocab_size)
                {
                    size_t pytorch_seq_len = pytorch_logits_count / vocab_size;
                    size_t last_offset = (pytorch_seq_len - 1) * vocab_size;
                    pytorch_logits = pytorch_lm_head.data() + last_offset;
                    pytorch_logits_count = vocab_size;
                }

                // Compare with PyTorch
                DecodeStepStats step_stats;
                step_stats.step_idx = static_cast<int>(step);

                // ---------------------------------------------------------------
                // Per-layer cosine similarity comparison for this decode step
                // ---------------------------------------------------------------
                {
                    int n_layers = static_cast<int>(model_ctx_->model().block_count);
                    auto snapshot_keys = runner_->getSnapshotKeys();
                    std::set<std::string> available_snapshots(snapshot_keys.begin(), snapshot_keys.end());
                    const auto gdn_cfg_decode = getGDNHeadConfig();
                    const auto moe_cfg_decode = getMoEConfig();

                    for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx)
                    {
                        LayerStats stats;
                        stats.layer_idx = layer_idx;
                        float sum_cosine = 0.0f;

                        for (const auto &stage : decode_per_layer_stages)
                        {
                            // Skip excluded stages
                            if (!config_.excluded_stages.empty())
                            {
                                bool is_excluded = std::find(
                                                       config_.excluded_stages.begin(),
                                                       config_.excluded_stages.end(),
                                                       stage) != config_.excluded_stages.end();
                                if (is_excluded)
                                    continue;
                            }

                            // Llaminar snapshot key: layer{N}_{STAGE}
                            std::string llaminar_key = "layer" + std::to_string(layer_idx) + "_" + stage;
                            if (!available_snapshots.count(llaminar_key))
                                continue;

                            // PyTorch snapshot key: decode_step{N}_layer{L}_{STAGE}
                            std::string pytorch_key = step_prefix + "_layer" + std::to_string(layer_idx) + "_" + stage;
                            auto pytorch_data = loadPyTorchSnapshot(pytorch_key);
                            if (pytorch_data.empty())
                                continue;

                            size_t llaminar_size;
                            const float *llaminar_data = runner_->getSnapshot(llaminar_key, llaminar_size);
                            if (!llaminar_data)
                                continue;

                            // PyTorch decode snapshots contain full-sequence output
                            // (all positions including prompt), while Llaminar only
                            // captures the last position during incremental decode.
                            // Extract last position from PyTorch data to match sizes.
                            if (pytorch_data.size() > llaminar_size && llaminar_size > 0 &&
                                pytorch_data.size() % llaminar_size == 0)
                            {
                                size_t last_offset = pytorch_data.size() - llaminar_size;
                                pytorch_data = std::vector<float>(
                                    pytorch_data.begin() + static_cast<ptrdiff_t>(last_offset),
                                    pytorch_data.end());
                            }
                            else if (pytorch_data.size() != llaminar_size)
                            {
                                continue; // Size mismatch, skip
                            }

                            // Apply GDN V-head permutation if needed
                            auto permuted_decode = applyGDNHeadPermutation(
                                llaminar_data, llaminar_size, stage, gdn_cfg_decode);
                            const float *decode_compare = permuted_decode.empty() ? llaminar_data : permuted_decode.data();

                            StageComparisonResult result;
                            if (stage == "MOE_ROUTING_INDICES")
                            {
                                result = compareRoutingIndices(decode_compare, pytorch_data, llaminar_size,
                                                               moe_cfg_decode.top_k, stage);
                            }
                            else if (stage == "MOE_ROUTING_WEIGHTS")
                            {
                                std::string idx_key = "decode_step" + std::to_string(step) + "_layer" + std::to_string(layer_idx) + "_MOE_ROUTING_INDICES";
                                size_t ll_idx_size;
                                const float *ll_idx = runner_->getSnapshot(idx_key, ll_idx_size);
                                auto pt_idx = loadPyTorchSnapshot(idx_key);
                                if (ll_idx && !pt_idx.empty())
                                    result = compareRoutingWeights(decode_compare, pytorch_data, ll_idx, pt_idx,
                                                                   llaminar_size, moe_cfg_decode.top_k,
                                                                   moe_cfg_decode.num_experts, stage);
                                else
                                    result = compareTensors(decode_compare, pytorch_data, llaminar_size, stage);
                            }
                            else
                            {
                                result = compareTensors(decode_compare, pytorch_data, llaminar_size, stage);
                            }
                            stats.stages_compared++;
                            sum_cosine += result.cosine_similarity;
                            stats.stage_results.push_back(result);

                            if (result.cosine_similarity < stats.min_cosine_sim)
                            {
                                stats.min_cosine_sim = result.cosine_similarity;
                                stats.worst_stage = stage;
                            }
                        }

                        // Compute per-stage cosine drops (error introduced by each stage)
                        if (stats.stage_results.size() >= 2)
                        {
                            for (size_t s = 1; s < stats.stage_results.size(); ++s)
                            {
                                float drop = stats.stage_results[s - 1].cosine_similarity -
                                             stats.stage_results[s].cosine_similarity;
                                stats.stage_results[s].cosine_drop = drop;

                                if (drop > stats.max_cosine_drop)
                                {
                                    stats.max_cosine_drop = drop;
                                    stats.max_drop_stage = stats.stage_results[s].stage_name;
                                }
                            }
                        }

                        if (stats.stages_compared > 0)
                        {
                            stats.avg_cosine_sim = sum_cosine / stats.stages_compared;
                        }
                        stats.passed = (stats.avg_cosine_sim >= config_.decode_cosine_threshold);

                        step_stats.layer_stats.push_back(stats);
                    }
                }

                step_stats.cosine_similarity = computeCosineSimilarity(
                    llaminar_logits, pytorch_logits,
                    std::min(decode_logits_size, pytorch_logits_count));

                step_stats.kl_divergence = computeKLDivergence(
                    llaminar_logits, pytorch_logits,
                    decode_logits_size, vocab_size);

                step_stats.top1_overlap = computeTopKOverlap(
                    llaminar_logits, pytorch_logits,
                    decode_logits_size, vocab_size, 1);

                step_stats.top5_overlap = computeTopKOverlap(
                    llaminar_logits, pytorch_logits,
                    decode_logits_size, vocab_size, 5);

                // Find argmax tokens
                step_stats.llaminar_token = 0;
                step_stats.pytorch_token = 0;
                float max_l = llaminar_logits[0];
                float max_p = pytorch_logits[0];

                for (size_t i = 1; i < vocab_size; ++i)
                {
                    if (llaminar_logits[i] > max_l)
                    {
                        max_l = llaminar_logits[i];
                        step_stats.llaminar_token = static_cast<int>(i);
                    }
                    if (pytorch_logits[i] > max_p)
                    {
                        max_p = pytorch_logits[i];
                        step_stats.pytorch_token = static_cast<int>(i);
                    }
                }

                step_stats.token_match = (step_stats.llaminar_token == step_stats.pytorch_token);
                if (step_stats.token_match)
                {
                    summary.top1_matches++;
                }

                // Check if PyTorch's top-1 token appears in Llaminar's top-5
                step_stats.top5_match = (step_stats.top5_overlap >= 0.2f - 1e-6f);
                if (step_stats.top5_match)
                {
                    summary.top5_matches++;
                }

                // Check if PyTorch's top-1 token appears in Llaminar's top-K
                if (config_.pytorch_top1_in_topk > 0)
                {
                    float topk_recall = pytorchTop1InLlaminarTopK(
                        llaminar_logits, pytorch_logits,
                        vocab_size, vocab_size, config_.pytorch_top1_in_topk);
                    step_stats.top3_match = (topk_recall >= 1.0f - 1e-6f);
                }
                else
                {
                    step_stats.top3_match = true; // gate disabled
                }
                if (step_stats.top3_match)
                {
                    summary.top3_matches++;
                }

                // Pass criteria: either cosine >= threshold OR KL < threshold
                step_stats.passed = (step_stats.cosine_similarity >= config_.decode_cosine_threshold) ||
                                    (step_stats.kl_divergence < config_.kl_threshold);

                if (step_stats.passed)
                {
                    summary.steps_passed++;
                }

                sum_cosine += step_stats.cosine_similarity;
                sum_kl += step_stats.kl_divergence;

                summary.step_stats.push_back(step_stats);
            }

            // Compute aggregate statistics
            if (summary.steps_total > 0)
            {
                summary.avg_cosine = sum_cosine / summary.steps_total;
                summary.avg_kl = sum_kl / summary.steps_total;
                summary.top1_accuracy = 100.0f * summary.top1_matches / summary.steps_total;
                summary.top3_accuracy = 100.0f * summary.top3_matches / summary.steps_total;
                summary.top5_accuracy = 100.0f * summary.top5_matches / summary.steps_total;
            }

            // Overall pass criteria
            int min_steps_required = static_cast<int>(summary.steps_total * config_.min_decode_pass_rate);
            bool topk_gate = (config_.pytorch_top1_in_topk <= 0) ||
                             (summary.top3_matches == summary.steps_total);
            summary.overall_passed = (summary.steps_passed >= min_steps_required) &&
                                     (summary.top5_accuracy >= config_.min_top5_accuracy) &&
                                     (summary.avg_cosine >= config_.decode_cosine_threshold) &&
                                     topk_gate;

            return summary;
        }

        /**
         * @brief Render decode parity results as Unicode table using libfort
         */
        void renderDecodeParityTable(const DecodeParitySummary &summary, const std::string &backend_name)
        {
            // Helper: format float to string with precision
            auto fmt_f6 = [](float v) -> std::string
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(6) << v;
                return ss.str();
            };

            auto fmt_f4 = [](float v) -> std::string
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(4) << v;
                return ss.str();
            };

            auto fmt_f1 = [](float v) -> std::string
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(1) << v;
                return ss.str();
            };

            auto fmt_f3 = [](float v) -> std::string
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(3) << v;
                return ss.str();
            };

            // Helper: status icon
            auto status_str = [](bool passed) -> std::string
            {
                return passed ? "✓" : "✗";
            };

            std::cout << "\n";

            // =========================================================================
            // Title table
            // =========================================================================
            {
                fort::utf8_table title_table;
                title_table.set_border_style(FT_DOUBLE2_STYLE);

                std::ostringstream title_ss;
                title_ss << backend_name << " INCREMENTAL DECODE PARITY";

                std::ostringstream subtitle_ss;
                subtitle_ss << "Threshold: cosine >= " << fmt_f3(config_.decode_cosine_threshold)
                            << " OR KL < " << fmt_f3(config_.kl_threshold);

                title_table << title_ss.str() << fort::endr;
                title_table << subtitle_ss.str() << fort::endr;

                title_table[0][0].set_cell_text_align(fort::text_align::center);
                title_table[1][0].set_cell_text_align(fort::text_align::center);
                title_table.row(0).set_cell_row_type(fort::row_type::header);

                std::cout << title_table.to_string();
            }

            // =========================================================================
            // Main decode parity table
            // =========================================================================
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Header row
            table << fort::header
                  << "Step" << "Cosine" << "KL" << "Llaminar" << "PyTorch" << ("InTop" + std::to_string(config_.pytorch_top1_in_topk > 0 ? config_.pytorch_top1_in_topk : 3)) << "OK"
                  << fort::endr;

            // Set column alignments
            table.column(0).set_cell_text_align(fort::text_align::right);
            table.column(1).set_cell_text_align(fort::text_align::right);
            table.column(2).set_cell_text_align(fort::text_align::right);
            table.column(3).set_cell_text_align(fort::text_align::right);
            table.column(4).set_cell_text_align(fort::text_align::right);
            table.column(5).set_cell_text_align(fort::text_align::center);
            table.column(6).set_cell_text_align(fort::text_align::center);

            // Per-step rows
            for (const auto &step : summary.step_stats)
            {
                std::string match_marker = step.token_match ? " ✓" : " ✗";

                std::ostringstream llaminar_ss;
                llaminar_ss << step.llaminar_token << match_marker;

                table << step.step_idx
                      << fmt_f6(step.cosine_similarity)
                      << fmt_f6(step.kl_divergence)
                      << llaminar_ss.str()
                      << step.pytorch_token
                      << status_str(step.top3_match)
                      << status_str(step.passed)
                      << fort::endr;
            }

            std::cout << table.to_string();

            // =========================================================================
            // Summary footer table
            // =========================================================================
            {
                fort::utf8_table summary_table;
                summary_table.set_border_style(FT_DOUBLE2_STYLE);

                std::ostringstream summary_ss;
                int k = config_.pytorch_top1_in_topk > 0 ? config_.pytorch_top1_in_topk : 3;
                summary_ss << "SUMMARY:  Steps=" << summary.steps_passed << "/" << summary.steps_total
                           << "  AvgCosine=" << fmt_f4(summary.avg_cosine)
                           << "  Top1=" << fmt_f1(summary.top1_accuracy) << "%"
                           << "  RefInTop" << k << "=" << summary.top3_matches << "/" << summary.steps_total
                           << "  Top5=" << fmt_f1(summary.top5_accuracy) << "%"
                           << "  " << (summary.overall_passed ? "✓ PASSED" : "✗ FAILED");

                summary_table << summary_ss.str() << fort::endr;
                summary_table[0][0].set_cell_text_align(fort::text_align::center);

                std::cout << summary_table.to_string();
            }

            std::cout << std::endl;
        }

        /**
         * @brief Render per-layer cosine similarity for a specific decode step
         *
         * Shows the layer-by-layer breakdown to identify where divergence starts.
         * Typically rendered for step 0 (first decode step) since that's most diagnostic.
         */
        void renderDecodeLayerParityTable(
            const DecodeStepStats &step,
            const std::string &backend_name)
        {
            if (step.layer_stats.empty())
                return;

            auto fmt_f6 = [](float v) -> std::string
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(6) << v;
                return ss.str();
            };

            auto status_str = [](bool passed) -> std::string
            {
                return passed ? "✓" : "✗";
            };

            std::cout << "\n";

            // Title
            {
                fort::utf8_table title_table;
                title_table.set_border_style(FT_DOUBLE2_STYLE);

                std::ostringstream title_ss;
                title_ss << backend_name << " DECODE STEP " << step.step_idx
                         << " LAYER-BY-LAYER PARITY";

                title_table << title_ss.str() << fort::endr;
                title_table[0][0].set_cell_text_align(fort::text_align::center);
                title_table.row(0).set_cell_row_type(fort::row_type::header);

                std::cout << title_table.to_string();
            }

            // Layer table
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            table << fort::header
                  << "Layer" << "Avg Cosine" << "Min Cosine" << "Worst Stage"
                  << "Max Drop" << "Drop Stage" << "Stages"
                  << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::center);
            table.column(1).set_cell_text_align(fort::text_align::right);
            table.column(2).set_cell_text_align(fort::text_align::right);
            table.column(3).set_cell_text_align(fort::text_align::left);
            table.column(4).set_cell_text_align(fort::text_align::right);
            table.column(5).set_cell_text_align(fort::text_align::left);
            table.column(6).set_cell_text_align(fort::text_align::right);

            for (const auto &stats : step.layer_stats)
            {
                if (stats.stages_compared == 0)
                    continue;

                std::ostringstream layer_ss;
                layer_ss << "Layer " << stats.layer_idx;

                std::string drop_str = (stats.max_cosine_drop > 0.001f)
                                           ? fmt_f6(stats.max_cosine_drop)
                                           : "-";
                std::string drop_stage = stats.max_drop_stage.empty() ? "-" : stats.max_drop_stage;

                table << layer_ss.str()
                      << fmt_f6(stats.avg_cosine_sim)
                      << fmt_f6(stats.min_cosine_sim)
                      << stats.worst_stage
                      << drop_str
                      << drop_stage
                      << stats.stages_compared
                      << fort::endr;
            }

            std::cout << table.to_string();
        }

        /**
         * @brief Assert decode parity criteria
         *
         * Call this after runSingleDeviceDecodeParity() or runTPDecodeParity()
         * to apply standard assertions. MPI-aware: Only renders table on rank 0.
         */
        void assertDecodeParity(const DecodeParitySummary &summary)
        {
            // Skip if no decode steps were tested
            if (summary.steps_total == 0)
            {
                GTEST_SKIP() << "No decode snapshots found - skipping decode parity assertions";
            }

            // For cross-rank PP, only the tail rank has valid logits for decode comparison.
            // Detect by checking if any decode step produced valid metrics.
            const bool has_logit_data = (summary.steps_passed > 0 || summary.avg_cosine > 0.0f);

            // Render table first (rank 0 only)
            if (isRank0())
            {
                renderDecodeParityTable(summary, getBackendName());

                // Render layer-by-layer breakdown for the first decode step
                // (most diagnostic for identifying where divergence originates)
                if (!summary.step_stats.empty() &&
                    !summary.step_stats[0].layer_stats.empty())
                {
                    renderDecodeLayerParityTable(summary.step_stats[0], getBackendName());
                }
            }

            // Export CSV results
            exportDecodeCSV(summary);

            // Assertions — skip on ranks that lack logit data (PP non-tail ranks)
            if (!has_logit_data)
                return;

            int min_steps_required = static_cast<int>(summary.steps_total * config_.min_decode_pass_rate);
            EXPECT_GE(summary.steps_passed, min_steps_required)
                << "Not enough decode steps passed: " << summary.steps_passed << "/" << summary.steps_total
                << " (required: " << min_steps_required << ")";

            EXPECT_GE(summary.top5_accuracy, config_.min_top5_accuracy)
                << "Top-5 accuracy too low: " << summary.top5_accuracy << "%"
                << " (required: " << config_.min_top5_accuracy << "%)";

            EXPECT_GE(summary.avg_cosine, config_.decode_cosine_threshold)
                << "Average cosine too low: " << summary.avg_cosine
                << " (required: " << config_.decode_cosine_threshold << ")";

            if (config_.pytorch_top1_in_topk > 0)
            {
                EXPECT_EQ(summary.top3_matches, summary.steps_total)
                    << "PyTorch's top-1 token must appear in llaminar's top-" << config_.pytorch_top1_in_topk << " for every decode step. "
                    << "Failed: " << summary.top3_matches << "/" << summary.steps_total;
            }
        }

        // =========================================================================
        // CSV Results Export
        // =========================================================================

        /**
         * @brief Get abbreviated git hash of current HEAD
         */
        static std::string getGitHash()
        {
            std::string hash = "unknown";
            FILE *pipe = popen("git rev-parse --short HEAD 2>/dev/null", "r");
            if (pipe)
            {
                char buf[64];
                if (fgets(buf, sizeof(buf), pipe))
                {
                    hash = buf;
                    // Trim trailing newline
                    while (!hash.empty() && (hash.back() == '\n' || hash.back() == '\r'))
                        hash.pop_back();
                }
                pclose(pipe);
            }
            return hash;
        }

        /**
         * @brief Get current GTest test name as "TestSuite/TestName"
         */
        static std::string getTestName()
        {
            const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
            if (!info)
                return "unknown";
            std::string suite = info->test_suite_name() ? info->test_suite_name() : "";
            std::string name = info->name() ? info->name() : "";
            // For parameterized tests, name includes the parameter
            return suite + "/" + name;
        }

        /**
         * @brief Derive the parity results directory from __FILE__ path
         *
         * ParityTestBase.h lives at tests/v2/integration/parity/ParityTestBase.h
         * Results go to tests/v2/integration/parity/results/<git_hash>/<test_name>/
         */
        static std::filesystem::path getResultsDir()
        {
            // Use __FILE__ to find the parity directory
            std::filesystem::path this_file(__FILE__);
            std::filesystem::path parity_dir = this_file.parent_path(); // .../tests/v2/integration/parity

            std::string git_hash = getGitHash();
            std::string test_name = getTestName();

            // Sanitize test_name for filesystem (replace / with _)
            std::string safe_name;
            for (char c : test_name)
            {
                if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' || c == '|')
                    safe_name += '_';
                else
                    safe_name += c;
            }

            return parity_dir / "results" / git_hash / safe_name;
        }

        /**
         * @brief Ensure results directory exists
         */
        static std::filesystem::path ensureResultsDir()
        {
            auto dir = getResultsDir();
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec)
            {
                LOG_WARN("Failed to create results directory: " << dir << " (" << ec.message() << ")");
            }
            return dir;
        }

        /**
         * @brief Export prefill parity results to CSV
         *
         * Writes two CSV files:
         *   - prefill_layers.csv: Per-layer metrics
         *   - prefill_summary.csv: LM_HEAD and overall metrics
         */
        void exportPrefillCSV(const ParityTestSummary &summary)
        {
            if (!isRank0())
                return;

            auto dir = ensureResultsDir();
            std::string backend = getBackendName();

            // --- prefill_layers.csv ---
            {
                auto path = dir / "prefill_layers.csv";
                std::ofstream f(path);
                if (!f.is_open())
                {
                    LOG_WARN("Cannot write " << path);
                    return;
                }

                f << "backend,layer,avg_cosine,min_cosine,worst_stage,max_cosine_drop,max_drop_stage,stages_compared,max_kurtosis,max_kurtosis_stage,passed\n";

                // Embedding row
                f << backend << ",EMBEDDING,"
                  << summary.embedding_cosine << ",,"
                  << ",,,,,,"
                  << (summary.embedding_passed ? "true" : "false") << "\n";

                for (const auto &ls : summary.layer_stats)
                {
                    f << backend << ","
                      << ls.layer_idx << ","
                      << ls.avg_cosine_sim << ","
                      << ls.min_cosine_sim << ","
                      << ls.worst_stage << ","
                      << ls.max_cosine_drop << ","
                      << ls.max_drop_stage << ","
                      << ls.stages_compared << ","
                      << ls.max_kurtosis << ","
                      << ls.max_kurtosis_stage << ","
                      << (ls.passed ? "true" : "false") << "\n";
                }

                LOG_INFO("[CSV] Wrote " << path);
            }

            // --- prefill_summary.csv ---
            {
                auto path = dir / "prefill_summary.csv";
                std::ofstream f(path);
                if (!f.is_open())
                {
                    LOG_WARN("Cannot write " << path);
                    return;
                }

                f << "backend,lm_head_cosine,lm_head_kl,lm_head_top1,lm_head_top5,"
                  << "lm_head_pytorch_top1_in_topk,early_layers_passed,total_layers_passed,overall_passed\n";

                f << backend << ","
                  << summary.lm_head_cosine << ","
                  << summary.lm_head_kl << ","
                  << summary.lm_head_top1 << ","
                  << summary.lm_head_top5 << ","
                  << (summary.lm_head_pytorch_top1_in_top3 ? "true" : "false") << ","
                  << summary.early_layers_passed << ","
                  << summary.total_layers_passed << ","
                  << (summary.overall_passed ? "true" : "false") << "\n";

                LOG_INFO("[CSV] Wrote " << path);
            }

            // --- prefill_stages.csv (detailed per-stage distribution stats) ---
            exportPrefillStagesCSV(summary);
        }

        /**
         * @brief Export decode parity results to CSV
         *
         * Writes two CSV files:
         *   - decode_steps.csv: Per-step LM_HEAD metrics
         *   - decode_layers.csv: Per-step per-layer metrics (if layer stats captured)
         */
        void exportDecodeCSV(const DecodeParitySummary &summary)
        {
            if (!isRank0())
                return;

            auto dir = ensureResultsDir();
            std::string backend = getBackendName();

            // --- decode_steps.csv ---
            {
                auto path = dir / "decode_steps.csv";
                std::ofstream f(path);
                if (!f.is_open())
                {
                    LOG_WARN("Cannot write " << path);
                    return;
                }

                f << "backend,step,cosine,kl_divergence,top1_overlap,top5_overlap,"
                  << "llaminar_token,pytorch_token,token_match,top3_match,top5_match,passed\n";

                for (const auto &ss : summary.step_stats)
                {
                    f << backend << ","
                      << ss.step_idx << ","
                      << ss.cosine_similarity << ","
                      << ss.kl_divergence << ","
                      << ss.top1_overlap << ","
                      << ss.top5_overlap << ","
                      << ss.llaminar_token << ","
                      << ss.pytorch_token << ","
                      << (ss.token_match ? "true" : "false") << ","
                      << (ss.top3_match ? "true" : "false") << ","
                      << (ss.top5_match ? "true" : "false") << ","
                      << (ss.passed ? "true" : "false") << "\n";
                }

                LOG_INFO("[CSV] Wrote " << path);
            }

            // --- decode_layers.csv ---
            {
                bool has_layer_data = false;
                for (const auto &ss : summary.step_stats)
                {
                    if (!ss.layer_stats.empty())
                    {
                        has_layer_data = true;
                        break;
                    }
                }

                if (!has_layer_data)
                    return;

                auto path = dir / "decode_layers.csv";
                std::ofstream f(path);
                if (!f.is_open())
                {
                    LOG_WARN("Cannot write " << path);
                    return;
                }

                f << "backend,step,layer,avg_cosine,min_cosine,worst_stage,max_cosine_drop,max_drop_stage,stages_compared,passed\n";

                for (const auto &ss : summary.step_stats)
                {
                    for (const auto &ls : ss.layer_stats)
                    {
                        if (ls.stages_compared == 0)
                            continue;

                        f << backend << ","
                          << ss.step_idx << ","
                          << ls.layer_idx << ","
                          << ls.avg_cosine_sim << ","
                          << ls.min_cosine_sim << ","
                          << ls.worst_stage << ","
                          << ls.max_cosine_drop << ","
                          << ls.max_drop_stage << ","
                          << ls.stages_compared << ","
                          << (ls.passed ? "true" : "false") << "\n";
                    }
                }

                LOG_INFO("[CSV] Wrote " << path);
            }

            // --- decode_stages.csv (detailed per-stage distribution stats) ---
            exportDecodeStagesCSV(summary);
        }

        /**
         * @brief Write the CSV header columns for TensorDistributionStats
         *
         * Used by both prefill and decode stage CSV writers.
         * @param prefix "llaminar_" or "pytorch_"
         */
        static void writeDistributionStatsHeader(std::ofstream &f, const std::string &prefix)
        {
            f << prefix << "min," << prefix << "max,"
              << prefix << "mean," << prefix << "stddev,"
              << prefix << "kurtosis," << prefix << "skewness,"
              << prefix << "p95," << prefix << "p99,"
              << prefix << "outlier_frac," << prefix << "dynamic_range,"
              << prefix << "sparsity," << prefix << "zero_frac,"
              << prefix << "nan_count," << prefix << "inf_count,"
              << prefix << "elements";
        }

        /**
         * @brief Write the CSV data columns for TensorDistributionStats
         */
        static void writeDistributionStatsData(std::ofstream &f, const TensorDistributionStats &s)
        {
            f << s.min << "," << s.max << ","
              << s.mean << "," << s.stddev << ","
              << s.kurtosis << "," << s.skewness << ","
              << s.p95 << "," << s.p99 << ","
              << s.outlier_fraction << "," << s.dynamic_range << ","
              << s.sparsity << "," << s.zero_fraction << ","
              << s.nan_count << "," << s.inf_count << ","
              << s.element_count;
        }

        /**
         * @brief Export per-stage detailed distribution stats for prefill
         *
         * Called from exportPrefillCSV. Writes prefill_stages.csv with
         * per-tensor (per-stage) numerical analysis of both llaminar and pytorch tensors.
         */
        void exportPrefillStagesCSV(const ParityTestSummary &summary)
        {
            if (!isRank0())
                return;

            auto dir = ensureResultsDir();
            std::string backend = getBackendName();

            auto path = dir / "prefill_stages.csv";
            std::ofstream f(path);
            if (!f.is_open())
            {
                LOG_WARN("Cannot write " << path);
                return;
            }

            f << "backend,layer,stage,cosine,cosine_drop,rel_l2,max_abs_diff,snr_db,rmse,error_entropy,";
            writeDistributionStatsHeader(f, "llaminar_");
            f << ",";
            writeDistributionStatsHeader(f, "pytorch_");
            f << "\n";

            for (const auto &ls : summary.layer_stats)
            {
                for (const auto &sr : ls.stage_results)
                {
                    f << backend << ","
                      << ls.layer_idx << ","
                      << sr.stage_name << ","
                      << sr.cosine_similarity << ","
                      << sr.cosine_drop << ","
                      << sr.rel_l2_norm << ","
                      << sr.max_abs_diff << ","
                      << sr.snr_db << ","
                      << sr.rmse << ","
                      << sr.error_entropy << ",";
                    writeDistributionStatsData(f, sr.llaminar_stats);
                    f << ",";
                    writeDistributionStatsData(f, sr.pytorch_stats);
                    f << "\n";
                }
            }

            LOG_INFO("[CSV] Wrote " << path);
        }

        /**
         * @brief Export per-stage detailed distribution stats for decode
         *
         * Called from exportDecodeCSV. Writes decode_stages.csv with
         * per-tensor (per-stage) numerical analysis of both llaminar and pytorch tensors.
         */
        void exportDecodeStagesCSV(const DecodeParitySummary &summary)
        {
            if (!isRank0())
                return;

            bool has_stage_data = false;
            for (const auto &ss : summary.step_stats)
            {
                for (const auto &ls : ss.layer_stats)
                {
                    if (!ls.stage_results.empty())
                    {
                        has_stage_data = true;
                        break;
                    }
                }
                if (has_stage_data)
                    break;
            }
            if (!has_stage_data)
                return;

            auto dir = ensureResultsDir();
            std::string backend = getBackendName();

            auto path = dir / "decode_stages.csv";
            std::ofstream f(path);
            if (!f.is_open())
            {
                LOG_WARN("Cannot write " << path);
                return;
            }

            f << "backend,step,layer,stage,cosine,cosine_drop,rel_l2,max_abs_diff,snr_db,rmse,error_entropy,";
            writeDistributionStatsHeader(f, "llaminar_");
            f << ",";
            writeDistributionStatsHeader(f, "pytorch_");
            f << "\n";

            for (const auto &ss : summary.step_stats)
            {
                for (const auto &ls : ss.layer_stats)
                {
                    for (const auto &sr : ls.stage_results)
                    {
                        f << backend << ","
                          << ss.step_idx << ","
                          << ls.layer_idx << ","
                          << sr.stage_name << ","
                          << sr.cosine_similarity << ","
                          << sr.cosine_drop << ","
                          << sr.rel_l2_norm << ","
                          << sr.max_abs_diff << ","
                          << sr.snr_db << ","
                          << sr.rmse << ","
                          << sr.error_entropy << ",";
                        writeDistributionStatsData(f, sr.llaminar_stats);
                        f << ",";
                        writeDistributionStatsData(f, sr.pytorch_stats);
                        f << "\n";
                    }
                }
            }

            LOG_INFO("[CSV] Wrote " << path);
        }
    };

} // namespace llaminar2::test::parity
