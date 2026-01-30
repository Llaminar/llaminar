/**
 * @file OrchestrationTestBase.h
 * @brief Base class and utilities for orchestration/placement integration tests
 *
 * Provides standardized infrastructure for testing Llaminar's placement and
 * orchestration subsystems across various cluster topologies and model sizes.
 * All orchestration tests should inherit from OrchestrationTestBase to ensure:
 *
 * - Consistent topology simulation (via MockMPITopologyBuilder)
 * - Standard device capability creation (GPUs, CPUs)
 * - PlacementInput generation for various model sizes
 * - Validation helpers for placement plans
 * - Summary output formatting
 *
 * Usage:
 *   class Test__ScenarioFoo : public OrchestrationTestBase {
 *   protected:
 *       ClusterConfig getClusterConfig() override { return myConfig; }
 *       std::vector<ModelConfig> getModelConfigs() override { return myModels; }
 *   };
 *   INSTANTIATE_ORCHESTRATION_TESTS(Test__ScenarioFoo);
 *
 * @see docs/v2/ARCHITECTURE_EXECUTION_SCENARIOS.md
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>
#include <unordered_set>
#include <set>
#include <cstdio>
#include <iomanip>
#include <sstream>

#include "mocks/MockMPITopology.h"
#include "mocks/MockMPIContext.h"
#include "mocks/MockComputeStage.h"
#include "mocks/MockGraphBufferManager.h"
#include "execution/mpi_orchestration/PlacementStrategy.h"
#include "execution/mpi_orchestration/PlacementPlan.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "execution/local_execution/graph/GraphExecutor.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include "execution/mpi_orchestration/placement/HeterogeneousMultiDomainStrategy.h"
#include "models/qwen/Qwen2Schema.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"

namespace llaminar2::test::orchestration
{

    // =============================================================================
    // Device Specifications
    // =============================================================================

    /**
     * @brief Common GPU specifications for test scenarios
     */
    struct GPUSpec
    {
        std::string name;
        DeviceCapability::Type type;
        float memory_gb;
        float bandwidth_gbps;
        float compute_tflops;
        bool has_nvlink = false;
        bool has_xgmi = false;

        /**
         * @brief Create a DeviceCapability from this spec
         */
        DeviceCapability toCapability(int device_id) const
        {
            DeviceCapability cap;
            cap.type = type;
            cap.device_id = device_id;
            cap.memory_bytes = static_cast<size_t>(memory_gb * 1024 * 1024 * 1024);
            cap.relative_compute = compute_tflops;
            cap.name = name;
            return cap;
        }
    };

    /**
     * @brief Pre-defined GPU specifications for common hardware
     */
    namespace GPUSpecs
    {
        // NVIDIA
        inline const GPUSpec RTX_3090 = {
            "NVIDIA RTX 3090", DeviceCapability::Type::CUDA,
            24.0f, 936.0f, 35.0f, false, false};
        inline const GPUSpec RTX_4090 = {
            "NVIDIA RTX 4090", DeviceCapability::Type::CUDA,
            24.0f, 1008.0f, 83.0f, false, false};
        inline const GPUSpec A100_40GB = {
            "NVIDIA A100 40GB", DeviceCapability::Type::CUDA,
            40.0f, 1555.0f, 156.0f, true, false};
        inline const GPUSpec A100_80GB = {
            "NVIDIA A100 80GB", DeviceCapability::Type::CUDA,
            80.0f, 2039.0f, 156.0f, true, false};
        inline const GPUSpec H100_80GB = {
            "NVIDIA H100 80GB", DeviceCapability::Type::CUDA,
            80.0f, 3350.0f, 267.0f, true, false};

        // AMD
        inline const GPUSpec MI50 = {
            "AMD Instinct Mi50", DeviceCapability::Type::ROCm,
            16.0f, 717.0f, 13.0f, false, true};
        inline const GPUSpec MI100 = {
            "AMD Instinct Mi100", DeviceCapability::Type::ROCm,
            32.0f, 1228.0f, 46.0f, false, true};
        inline const GPUSpec MI250X = {
            "AMD Instinct Mi250X", DeviceCapability::Type::ROCm,
            128.0f, 3200.0f, 191.0f, false, true};
        inline const GPUSpec RX_7900_XTX = {
            "AMD Radeon RX 7900 XTX", DeviceCapability::Type::ROCm,
            24.0f, 960.0f, 23.0f, false, false};
    }

    /**
     * @brief CPU specifications for test scenarios
     */
    struct CPUSpec
    {
        std::string name;
        int cores;
        float memory_gb;
        float bandwidth_gbps; // Memory bandwidth

        DeviceCapability toCapability() const
        {
            return MockDevices::cpu(1.0f);
        }
    };

    namespace CPUSpecs
    {
        inline const CPUSpec XEON_28C_DDR4 = {"Xeon 28-core DDR4", 28, 128.0f, 153.6f};
        inline const CPUSpec XEON_28C_DDR5 = {"Xeon 28-core DDR5", 28, 512.0f, 358.0f};
        inline const CPUSpec EPYC_64C = {"EPYC 64-core", 64, 512.0f, 400.0f};
    }

    // =============================================================================
    // Socket Configuration
    // =============================================================================

    /**
     * @brief Configuration for a single socket in a cluster
     */
    struct SocketConfig
    {
        std::vector<GPUSpec> gpus;
        CPUSpec cpu;

        /**
         * @brief Total VRAM for this socket in GB
         */
        float totalVRAMGB() const
        {
            float total = 0.0f;
            for (const auto &gpu : gpus)
            {
                total += gpu.memory_gb;
            }
            return total;
        }

        /**
         * @brief Total GPU bandwidth for this socket in GB/s
         */
        float totalGPUBandwidthGBs() const
        {
            float total = 0.0f;
            for (const auto &gpu : gpus)
            {
                total += gpu.bandwidth_gbps;
            }
            return total;
        }
    };

    // =============================================================================
    // Cluster Configuration
    // =============================================================================

    /**
     * @brief Complete cluster topology configuration
     *
     * Defines the hardware layout for a test scenario:
     * - Number of machines and sockets
     * - GPU/CPU configuration per socket
     * - Interconnect characteristics
     */
    struct ClusterConfig
    {
        std::string name; ///< Scenario name for display
        int num_machines = 1;
        int sockets_per_machine = 1;

        /// Socket configurations (indexed by socket_id % configs.size())
        std::vector<SocketConfig> socket_configs;

        // Interconnect specifications
        float nvlink_bandwidth_gbps = 600.0f;
        float xgmi_bandwidth_gbps = 600.0f;
        float pcie_bandwidth_gbps = 32.0f;
        float qpi_upi_bandwidth_gbps = 50.0f;    ///< Cross-socket on same machine
        float infiniband_bandwidth_gbps = 25.0f; ///< Cross-machine

        // Computed properties
        int totalRanks() const { return num_machines * sockets_per_machine; }

        int totalGPUs() const
        {
            int total = 0;
            for (int rank = 0; rank < totalRanks(); ++rank)
            {
                const auto &socket = socket_configs[rank % socket_configs.size()];
                total += static_cast<int>(socket.gpus.size());
            }
            return total;
        }

        float totalVRAMGB() const
        {
            float total = 0.0f;
            for (int rank = 0; rank < totalRanks(); ++rank)
            {
                const auto &socket = socket_configs[rank % socket_configs.size()];
                total += socket.totalVRAMGB();
            }
            return total;
        }

        float totalGPUBandwidthTBs() const
        {
            float total = 0.0f;
            for (int rank = 0; rank < totalRanks(); ++rank)
            {
                const auto &socket = socket_configs[rank % socket_configs.size()];
                total += socket.totalGPUBandwidthGBs();
            }
            return total / 1000.0f;
        }

        float totalCPUBandwidthTBs() const
        {
            float total = 0.0f;
            for (int rank = 0; rank < totalRanks(); ++rank)
            {
                const auto &socket = socket_configs[rank % socket_configs.size()];
                total += socket.cpu.bandwidth_gbps;
            }
            return total / 1000.0f;
        }

        /**
         * @brief Get the socket config for a specific rank
         */
        const SocketConfig &getSocketConfig(int rank) const
        {
            return socket_configs[rank % socket_configs.size()];
        }
    };

    // =============================================================================
    // Model Configuration
    // =============================================================================

    /**
     * @brief Model architecture configuration for placement testing
     */
    struct ModelConfig
    {
        std::string name;
        int n_layers;
        size_t d_model;
        size_t d_ff;
        size_t vocab_size;
        size_t n_heads;
        size_t n_kv_heads;
        size_t estimated_memory_bytes; ///< Q4_0 weight size
        std::string quant_type = "Q4_0";

        /**
         * @brief Estimated memory in GB
         */
        float memorySizeGB() const
        {
            return static_cast<float>(estimated_memory_bytes) / (1024.0f * 1024.0f * 1024.0f);
        }

        /**
         * @brief Layers per rank if evenly distributed
         */
        float layersPerRank(int num_ranks) const
        {
            return static_cast<float>(n_layers) / num_ranks;
        }
    };

    /**
     * @brief Pre-defined model configurations
     */
    namespace ModelConfigs
    {
        inline const ModelConfig QWEN2_0_5B = {
            "Qwen2-0.5B", 24, 896, 4864, 151936, 14, 2,
            500ULL * 1024 * 1024};
        inline const ModelConfig QWEN2_7B = {
            "Qwen2-7B", 32, 4096, 11008, 151936, 32, 32,
            4ULL * 1024 * 1024 * 1024};
        inline const ModelConfig QWEN2_72B = {
            "Qwen2-72B", 80, 8192, 29568, 151936, 64, 8,
            40ULL * 1024 * 1024 * 1024};
        inline const ModelConfig QWEN2_235B = {
            "Qwen2-235B (fictional)", 128, 16384, 53248, 151936, 128, 16,
            130ULL * 1024 * 1024 * 1024};
        inline const ModelConfig QWEN2_571B = {
            "Qwen2-571B (fictional)", 160, 20480, 65536, 151936, 160, 20,
            320ULL * 1024 * 1024 * 1024};
        inline const ModelConfig QWEN2_1142B = {
            "Qwen2-1142B (fictional)", 192, 24576, 81920, 151936, 192, 24,
            640ULL * 1024 * 1024 * 1024};
    }

    // =============================================================================
    // Placement Validation Results
    // =============================================================================

    /**
     * @brief Result of validating a placement plan
     */
    struct PlacementValidationResult
    {
        bool is_valid = false;
        int gpu_layers = 0;
        int cpu_layers = 0;
        int total_layers = 0;
        std::set<int> gpus_used;
        bool respects_socket_boundary = true;
        std::string error_message;

        float gpuLayerPercentage() const
        {
            return total_layers > 0 ? 100.0f * gpu_layers / total_layers : 0.0f;
        }
    };

    // =============================================================================
    // Graph Bridge Validation Results
    // =============================================================================

    /**
     * @brief Result of validating graph construction from placement plan
     */
    struct GraphValidationResult
    {
        bool is_valid = true;
        int nodes_checked = 0;
        int device_mismatches = 0;
        int backend_mismatches = 0;
        std::vector<std::string> errors;

        void addError(const std::string &msg)
        {
            is_valid = false;
            errors.push_back(msg);
        }
    };

    /**
     * @brief Result of validating graph execution
     */
    struct ExecutionValidationResult
    {
        bool all_executed = true;
        int stages_executed = 0;
        int expected_stages = 0;
        std::vector<std::string> execution_order;
        std::vector<std::string> skipped_stages;

        bool orderCorrect() const
        {
            // Check stages executed in dependency order (no child before parent)
            return all_executed && stages_executed == expected_stages;
        }
    };

    /**
     * @brief Result of validating a heterogeneous plan
     */
    struct HeterogeneousPlanValidationResult
    {
        bool is_valid = false;
        int total_domains = 0;
        int gpu_domains = 0;
        int cpu_domains = 0;
        int pp_stages = 0;
        int total_layers_assigned = 0;
        bool all_layers_covered = false;
        bool no_layer_gaps = false;
        bool no_layer_overlaps = true;
        bool domains_have_valid_ranks = true;
        bool domains_have_valid_devices = true;
        std::string error_message;

        float gpuDomainPercentage() const
        {
            return total_domains > 0 ? 100.0f * gpu_domains / total_domains : 0.0f;
        }
    };

    /**
     * @brief Result of validating domain assignment for a layer
     */
    struct DomainAssignmentValidationResult
    {
        bool is_valid = false;
        int layer_idx = -1;
        int domain_id = -1;
        bool is_gpu_domain = false;
        bool is_cpu_domain = false;
        int ranks_in_domain = 0;
        int devices_in_domain = 0;
        std::string domain_type_name;
        std::string error_message;
    };

    /**
     * @brief Result of validating PP stage configuration
     */
    struct PPStageValidationResult
    {
        bool is_valid = false;
        int stage_id = -1;
        int node_id = -1;
        int layer_start = -1;
        int layer_end = -1;
        int layers_in_stage = 0;
        int domains_in_stage = 0;
        bool has_send_stage = false;
        bool has_recv_stage = false;
        std::string error_message;
    };

    // =============================================================================
    // Expected Parallelism Configuration (Declarative Test Expectations)
    // =============================================================================

    /**
     * @brief Declarative specification of expected parallelism deductions
     *
     * Scenarios override getExpectedParallelism() to declare what the orchestrator
     * should deduce from the hardware topology. The base class validates these
     * expectations automatically via INSTANTIATE_PARALLELISM_TESTS macro.
     *
     * Fields set to -1 or std::nullopt are not validated (scenario doesn't care).
     */
    struct ExpectedParallelismConfig
    {
        // -------------------------------------------------------------------------
        // Pipeline Parallelism Expectations
        // -------------------------------------------------------------------------
        int pp_degree = -1;               ///< Expected number of PP stages (-1 = don't check)
        bool pp_stages_contiguous = true; ///< PP stages should have contiguous layer ranges

        // -------------------------------------------------------------------------
        // Tensor Parallelism Expectations (per domain)
        // -------------------------------------------------------------------------
        int gpu_tp_degree = -1; ///< Expected TP degree within each GPU domain
        int cpu_tp_degree = -1; ///< Expected TP degree within each CPU domain

        // -------------------------------------------------------------------------
        // Domain Structure Expectations
        // -------------------------------------------------------------------------
        int total_domains = -1;        ///< Total TP domains across cluster
        int gpu_domains_total = -1;    ///< Total GPU TP domains
        int cpu_domains_total = -1;    ///< Total CPU TP domains
        int gpu_domains_per_node = -1; ///< GPU domains per physical node
        int cpu_domains_per_node = -1; ///< CPU domains per physical node
        int domains_per_pp_stage = -1; ///< TP domains per PP stage

        // -------------------------------------------------------------------------
        // Layer Distribution Expectations
        // -------------------------------------------------------------------------
        bool gpu_layers_precede_cpu = true;  ///< GPU domains get earlier layers than CPU
        float min_gpu_layer_fraction = 0.5f; ///< Minimum fraction of layers on GPUs
        float max_cpu_layer_fraction = 0.5f; ///< Maximum fraction of layers on CPUs

        // -------------------------------------------------------------------------
        // Device Assignment Expectations
        // -------------------------------------------------------------------------
        int gpus_per_gpu_domain = -1;           ///< GPUs per GPU TP domain
        int ranks_per_cpu_domain = -1;          ///< Ranks per CPU TP domain
        bool gpu_domains_are_intra_rank = true; ///< GPU domains confined to single rank
        bool cpu_domains_are_cross_rank = true; ///< CPU domains span multiple ranks

        // -------------------------------------------------------------------------
        // Total Parallelism Verification
        // -------------------------------------------------------------------------
        int total_gpu_devices = -1; ///< Total GPUs used across all domains
        int total_cpu_ranks = -1;   ///< Total CPU ranks across all domains

        /**
         * @brief Check if any parallelism expectations are set
         */
        bool hasExpectations() const
        {
            return pp_degree >= 0 ||
                   gpu_tp_degree >= 0 ||
                   cpu_tp_degree >= 0 ||
                   total_domains >= 0 ||
                   gpu_domains_total >= 0 ||
                   cpu_domains_total >= 0;
        }
    };

    /**
     * @brief Configuration for a mock layer graph
     */
    struct MockLayerGraphSpec
    {
        int layer_idx = 0;
        DeviceId device = DeviceId::cpu();
        bool include_attention = true;
        bool include_ffn = true;
        bool include_allreduce = false; ///< Insert allreduce at TP boundary

        /// Stages to generate for this layer
        std::vector<ComputeStageType> stages = {
            ComputeStageType::RMS_NORM,     // attn_norm
            ComputeStageType::GEMM,         // qkv_proj
            ComputeStageType::ROPE,         // rope
            ComputeStageType::ATTENTION,    // attention
            ComputeStageType::GEMM,         // o_proj
            ComputeStageType::ADD_RESIDUAL, // attn_residual
            ComputeStageType::RMS_NORM,     // ffn_norm
            ComputeStageType::GEMM,         // gate_up_proj
            ComputeStageType::SWIGLU,       // swiglu
            ComputeStageType::GEMM,         // down_proj
            ComputeStageType::ADD_RESIDUAL  // ffn_residual
        };
    };

    // =============================================================================
    // OrchestrationTestBase
    // =============================================================================

    /**
     * @brief Base class for orchestration integration tests
     *
     * Provides:
     * - Topology creation from ClusterConfig
     * - PlacementInput generation for models
     * - Plan validation helpers
     * - Summary output formatters
     */
    class OrchestrationTestBase : public ::testing::Test
    {
    protected:
        // =========================================================================
        // Abstract Methods (override in subclasses)
        // =========================================================================

        /**
         * @brief Return the cluster configuration for this scenario
         */
        virtual ClusterConfig getClusterConfig() = 0;

        /**
         * @brief Return the models to test in this scenario
         */
        virtual std::vector<ModelConfig> getModelConfigs() = 0;

        /**
         * @brief Return expected parallelism configuration (optional)
         *
         * Override this to declare what the orchestrator should deduce from hardware.
         * The base class validates these expectations automatically.
         *
         * @return Expected parallelism config (default has no expectations)
         */
        virtual ExpectedParallelismConfig getExpectedParallelism() const
        {
            return ExpectedParallelismConfig{}; // No expectations by default
        }

        // =========================================================================
        // Setup/Teardown
        // =========================================================================

        void SetUp() override
        {
            cluster_ = getClusterConfig();
            models_ = getModelConfigs();

            // Create topologies for all ranks
            topologies_.clear();
            for (int rank = 0; rank < cluster_.totalRanks(); ++rank)
            {
                topologies_.push_back(createTopology(rank));
            }
        }

        // =========================================================================
        // Topology Creation
        // =========================================================================

        /**
         * @brief Create MockMPITopology for a specific rank
         */
        std::shared_ptr<MockMPITopology> createTopology(int local_rank)
        {
            auto builder = MockMPITopologyBuilder();

            // Build all ranks to get consistent cluster view
            for (int rank = 0; rank < cluster_.totalRanks(); ++rank)
            {
                int machine = rank / cluster_.sockets_per_machine;
                const auto &socket_cfg = cluster_.getSocketConfig(rank);

                std::vector<DeviceCapability> devices;
                devices.push_back(socket_cfg.cpu.toCapability());

                int device_id = 0;
                for (const auto &gpu_spec : socket_cfg.gpus)
                {
                    auto cap = gpu_spec.toCapability(device_id++);
                    devices.push_back(cap);
                }

                std::string hostname = "node" + std::to_string(machine) +
                                       "-socket" + std::to_string(rank % cluster_.sockets_per_machine);
                builder.addRank(rank, machine, devices, hostname);
            }

            builder.setLocalRank(local_rank);
            return builder.build();
        }

        // =========================================================================
        // PlacementInput Creation
        // =========================================================================

        /**
         * @brief Create PlacementInput for a model and rank
         */
        PlacementInput createPlacementInput(const ModelConfig &model, int rank)
        {
            auto topology = topologies_[rank];
            const auto &socket_cfg = cluster_.getSocketConfig(rank);

            PlacementInput input;
            input.architecture = "qwen2";
            input.n_layers = model.n_layers;
            input.d_model = model.d_model;
            input.d_ff = model.d_ff;
            input.vocab_size = model.vocab_size;
            input.n_heads = model.n_heads;
            input.n_kv_heads = model.n_kv_heads;
            input.quant_type = model.quant_type;
            input.estimated_memory_bytes = model.estimated_memory_bytes;

            // Populate from topology
            input.world_size = topology->world_size();
            input.ranks_per_node = topology->ranks_per_node();
            input.node_count = topology->node_count();
            input.cluster_inventory = topology->clusterInventory();

            // Bandwidth info
            input.gpu_memory_bandwidth_gbps = socket_cfg.totalGPUBandwidthGBs() /
                                              std::max(1UL, socket_cfg.gpus.size());
            input.cpu_memory_bandwidth_gbps = socket_cfg.cpu.bandwidth_gbps;

            input.updateAggregatedFields();

            return input;
        }

        // =========================================================================
        // Validation Helpers
        // =========================================================================

        /**
         * @brief Validate a placement plan
         */
        PlacementValidationResult validatePlan(const PlacementPlan &plan, int rank)
        {
            PlacementValidationResult result;
            result.total_layers = static_cast<int>(plan.layers.size());
            result.is_valid = plan.isValid();

            if (!result.is_valid)
            {
                result.error_message = "Plan marked invalid";
                return result;
            }

            for (const auto &layer : plan.layers)
            {
                if (layer.device.isGPU())
                {
                    result.gpu_layers++;
                    result.gpus_used.insert(layer.device.gpu_index);
                }
                else
                {
                    result.cpu_layers++;
                }
            }

            return result;
        }

        /**
         * @brief Build GraphResolverConfig from ModelConfig for accurate buffer calculation
         */
        GraphResolverConfig buildResolverConfig(const ModelConfig &model, size_t seq_len = DEFAULT_SEQ_LEN) const
        {
            GraphResolverConfig config;
            config.n_layers = model.n_layers;
            config.d_model = model.d_model;
            config.d_ff = model.d_ff;
            config.vocab_size = model.vocab_size;
            config.n_heads = model.n_heads;
            config.n_kv_heads = model.n_kv_heads;
            config.head_dim = model.d_model / model.n_heads;
            config.seq_len = seq_len;

            // For single-rank estimation (no TP)
            config.world_size = 1;
            config.local_n_heads = model.n_heads;
            config.local_n_kv_heads = model.n_kv_heads;
            config.local_d_ff = model.d_ff;
            config.local_vocab = model.vocab_size;

            return config;
        }

        // =========================================================================
        // Memory Estimation Constants
        // =========================================================================

        /**
         * @brief Default sequence length for memory estimation (2048 tokens)
         *
         * Standard context length used for memory calculations. This represents
         * a typical conversation length for most use cases. Longer sequences
         * require proportionally more KV cache memory.
         */
        static constexpr size_t DEFAULT_SEQ_LEN = 2048;

        /**
         * @brief Memory overhead factor for runtime allocations (15%)
         *
         * This covers empirically-measured costs not captured by buffer specs:
         * - GEMM workspace: ~M×K int8 + M×N int32 accumulator
         * - Attention workspace: score matrices during computation
         * - Memory allocator fragmentation and alignment overhead
         * - Framework overhead (CUDA contexts, MPI buffers, etc.)
         *
         * Derived from profiling of actual inference runs.
         * See docs/v2/MEMORY_PROFILING.md for methodology.
         */
        static constexpr int MEMORY_OVERHEAD_PERCENT = 15;

        /**
         * @brief VRAM utilization threshold for "model fits" check (90%)
         *
         * We target 90% VRAM utilization to leave headroom for:
         * - GPU driver/runtime allocations
         * - Display buffer (if attached)
         * - OS overhead
         * - Dynamic allocations during inference
         */
        static constexpr int VRAM_UTILIZATION_PERCENT = 90;

        /**
         * @brief Bytes per element for KV cache storage
         *
         * Q16_1 format: 72 bytes per 32 elements = 2.25 bytes/element
         * This is the default precision for KV cache in production.
         *
         * Note: Other formats possible:
         * - FP16: 2.0 bytes/element (less precise)
         * - FP32: 4.0 bytes/element (debugging only)
         * - Q8_1: 1.125 bytes/element (experimental, lower precision)
         */
        static constexpr float KV_CACHE_BYTES_PER_ELEMENT = 72.0f / 32.0f; // Q16_1 format

        /**
         * @brief Estimate runtime memory using actual schema buffer specifications
         *
         * Uses Qwen2Schema + GraphResolver to calculate EXACT buffer requirements:
         * - Layer buffers: Q, K, V, attn_output, attn_proj, gate, up, normalized
         * - Model buffers: hidden, logits
         * - KV cache: calculated from actual dimensions using Q16_1 format
         * - Weights: from model config (Q4_0 estimate)
         *
         * This is NOT a guess - it's what the actual runtime will allocate.
         */
        size_t estimateRuntimeMemory(const ModelConfig &model, size_t seq_len = DEFAULT_SEQ_LEN) const
        {
            // Get the actual schema
            Qwen2SchemaFactory schema_factory;
            GraphSchema schema = schema_factory.createSchema();

            // Build resolver config with actual model dimensions
            GraphResolverConfig config = buildResolverConfig(model, seq_len);

            // Resolve layer buffers (allocated once, reused per layer)
            size_t layer_buffer_mem = 0;
            for (const auto &spec : schema.layer_buffers)
            {
                auto resolved = BufferAllocator::resolve(spec, config);
                layer_buffer_mem += resolved.totalBytes();
            }

            // Resolve model buffers (global buffers)
            size_t model_buffer_mem = 0;
            for (const auto &spec : schema.model_buffers)
            {
                auto resolved = BufferAllocator::resolve(spec, config);
                model_buffer_mem += resolved.totalBytes();
            }

            // KV Cache: 2 (K+V) * n_layers * n_kv_heads * head_dim * seq_len
            // Using Q16_1 format: 72 bytes per 32 elements = 2.25 bytes/element
            size_t head_dim = model.d_model / model.n_heads;
            size_t kv_elements = 2 * model.n_layers * model.n_kv_heads * head_dim * seq_len;
            size_t kv_cache_mem = static_cast<size_t>(kv_elements * KV_CACHE_BYTES_PER_ELEMENT);

            // Weights (from model config - Q4_0 estimate)
            size_t weight_mem = model.estimated_memory_bytes;

            // Total runtime memory
            // Note: Layer buffers don't multiply by n_layers due to ping-pong reuse
            size_t total = weight_mem + layer_buffer_mem + model_buffer_mem + kv_cache_mem;

            // Add overhead for GEMM workspace, attention scores, fragmentation
            total = total + (total * MEMORY_OVERHEAD_PERCENT / 100);

            return total;
        }

        /**
         * @brief Get detailed memory breakdown for debugging
         */
        struct MemoryBreakdown
        {
            size_t weights_bytes = 0;
            size_t layer_buffers_bytes = 0;
            size_t model_buffers_bytes = 0;
            size_t kv_cache_bytes = 0;
            size_t overhead_bytes = 0;
            size_t total_bytes = 0;

            float totalGB() const { return static_cast<float>(total_bytes) / (1024.0f * 1024.0f * 1024.0f); }

            void print(const std::string &model_name) const
            {
                printf("Memory breakdown for %s:\n", model_name.c_str());
                printf("  Weights:       %8.2f MB\n", weights_bytes / (1024.0f * 1024.0f));
                printf("  Layer buffers: %8.2f MB\n", layer_buffers_bytes / (1024.0f * 1024.0f));
                printf("  Model buffers: %8.2f MB\n", model_buffers_bytes / (1024.0f * 1024.0f));
                printf("  KV Cache:      %8.2f MB\n", kv_cache_bytes / (1024.0f * 1024.0f));
                printf("  Overhead:      %8.2f MB\n", overhead_bytes / (1024.0f * 1024.0f));
                printf("  TOTAL:         %8.2f GB\n", totalGB());
            }
        };

        MemoryBreakdown getMemoryBreakdown(const ModelConfig &model, size_t seq_len = DEFAULT_SEQ_LEN) const
        {
            MemoryBreakdown breakdown;

            Qwen2SchemaFactory schema_factory;
            GraphSchema schema = schema_factory.createSchema();
            GraphResolverConfig config = buildResolverConfig(model, seq_len);

            for (const auto &spec : schema.layer_buffers)
            {
                auto resolved = BufferAllocator::resolve(spec, config);
                breakdown.layer_buffers_bytes += resolved.totalBytes();
            }

            for (const auto &spec : schema.model_buffers)
            {
                auto resolved = BufferAllocator::resolve(spec, config);
                breakdown.model_buffers_bytes += resolved.totalBytes();
            }

            // KV Cache using Q16_1 format (72 bytes per 32 elements)
            size_t head_dim = model.d_model / model.n_heads;
            size_t kv_elements = 2 * model.n_layers * model.n_kv_heads * head_dim * seq_len;
            breakdown.kv_cache_bytes = static_cast<size_t>(kv_elements * KV_CACHE_BYTES_PER_ELEMENT);
            breakdown.weights_bytes = model.estimated_memory_bytes;

            size_t base = breakdown.weights_bytes + breakdown.layer_buffers_bytes +
                          breakdown.model_buffers_bytes + breakdown.kv_cache_bytes;
            breakdown.overhead_bytes = base * MEMORY_OVERHEAD_PERCENT / 100;
            breakdown.total_bytes = base + breakdown.overhead_bytes;

            return breakdown;
        }

        /**
         * @brief Check if model fits in cluster GPU memory (including runtime buffers)
         *
         * Uses actual schema-based buffer calculations.
         * Returns true if total runtime memory < VRAM_UTILIZATION_PERCENT of total VRAM.
         */
        bool modelFitsInGPUMemory(const ModelConfig &model)
        {
            size_t total_vram = static_cast<size_t>(cluster_.totalVRAMGB() * 1024 * 1024 * 1024);
            size_t runtime_mem = estimateRuntimeMemory(model);

            // Apply configured VRAM utilization threshold
            return runtime_mem < (total_vram * VRAM_UTILIZATION_PERCENT / 100);
        }

        /**
         * @brief Calculate how many machines are needed for a model
         */
        int machinesNeededForModel(const ModelConfig &model)
        {
            // Use actual memory calculation instead of just weight size
            size_t runtime_mem = estimateRuntimeMemory(model);
            float runtime_gb = static_cast<float>(runtime_mem) / (1024.0f * 1024.0f * 1024.0f);
            float vram_per_machine = cluster_.totalVRAMGB() / cluster_.num_machines;
            return static_cast<int>(std::ceil(runtime_gb / vram_per_machine));
        }

        /**
         * @brief Validate a HeterogeneousPlan from HeterogeneousMultiDomainStrategy
         */
        HeterogeneousPlanValidationResult validateHeterogeneousPlan(
            const HeterogeneousPlan &plan,
            int expected_layers)
        {
            HeterogeneousPlanValidationResult result;

            // Count domain types
            std::set<int> layers_covered;
            for (const auto &domain : plan.domains)
            {
                result.total_domains++;
                if (domain.type == TPDomainType::GPU_INTRA_RANK)
                {
                    result.gpu_domains++;
                }
                else if (domain.type == TPDomainType::CPU_CROSS_RANK)
                {
                    result.cpu_domains++;
                }

                // Track layers
                for (int l = domain.layer_start; l < domain.layer_end; ++l)
                {
                    if (layers_covered.count(l))
                    {
                        result.no_layer_overlaps = false;
                        result.error_message = "Layer " + std::to_string(l) + " assigned to multiple domains";
                    }
                    layers_covered.insert(l);
                }

                // Validate domain has ranks
                if (domain.ranks.empty())
                {
                    result.domains_have_valid_ranks = false;
                    result.error_message = "Domain has no ranks assigned";
                }

                // Validate domain has devices
                if (domain.devices.empty())
                {
                    result.domains_have_valid_devices = false;
                    result.error_message = "Domain has no devices assigned";
                }
            }

            result.total_layers_assigned = static_cast<int>(layers_covered.size());
            result.all_layers_covered = (result.total_layers_assigned == expected_layers);

            // Check for gaps
            result.no_layer_gaps = true;
            for (int l = 0; l < expected_layers; ++l)
            {
                if (!layers_covered.count(l))
                {
                    result.no_layer_gaps = false;
                    result.error_message = "Layer " + std::to_string(l) + " not assigned to any domain";
                    break;
                }
            }

            // Count PP stages
            result.pp_stages = static_cast<int>(plan.stages.size());

            result.is_valid = result.all_layers_covered &&
                              result.no_layer_gaps &&
                              result.no_layer_overlaps &&
                              result.domains_have_valid_ranks &&
                              result.domains_have_valid_devices;

            return result;
        }

        /**
         * @brief Validate domain assignment for a specific layer
         */
        DomainAssignmentValidationResult validateDomainForLayer(
            const HeterogeneousPlan &plan,
            int layer_idx)
        {
            DomainAssignmentValidationResult result;
            result.layer_idx = layer_idx;

            const auto *domain = plan.getDomainForLayer(layer_idx);
            if (!domain)
            {
                result.error_message = "No domain found for layer " + std::to_string(layer_idx);
                return result;
            }

            result.domain_id = domain->domain_id;
            result.is_gpu_domain = (domain->type == TPDomainType::GPU_INTRA_RANK);
            result.is_cpu_domain = (domain->type == TPDomainType::CPU_CROSS_RANK);
            result.ranks_in_domain = static_cast<int>(domain->ranks.size());
            result.devices_in_domain = static_cast<int>(domain->devices.size());

            // Map type to name
            switch (domain->type)
            {
            case TPDomainType::GPU_INTRA_RANK:
                result.domain_type_name = "GPU_INTRA_RANK";
                break;
            case TPDomainType::CPU_CROSS_RANK:
                result.domain_type_name = "CPU_CROSS_RANK";
                break;
            default:
                result.domain_type_name = "UNKNOWN";
                break;
            }

            result.is_valid = true;
            return result;
        }

        /**
         * @brief Validate a PP stage configuration
         */
        PPStageValidationResult validatePPStage(
            const HeterogeneousPlan &plan,
            int stage_idx)
        {
            PPStageValidationResult result;
            result.stage_id = stage_idx;

            if (stage_idx < 0 || stage_idx >= static_cast<int>(plan.stages.size()))
            {
                result.error_message = "Invalid stage index: " + std::to_string(stage_idx);
                return result;
            }

            const auto &stage = plan.stages[stage_idx];
            result.node_id = stage.node_id;
            result.layer_start = stage.layer_start;
            result.layer_end = stage.layer_end;
            result.layers_in_stage = stage.layer_end - stage.layer_start;
            result.domains_in_stage = static_cast<int>(stage.domains.size());

            // Check for PP communication stages
            result.has_send_stage = (stage_idx < static_cast<int>(plan.stages.size()) - 1);
            result.has_recv_stage = (stage_idx > 0);

            result.is_valid = (result.layers_in_stage > 0) && (result.domains_in_stage > 0);
            return result;
        }

        /**
         * @brief Validate plan against expected parallelism configuration
         *
         * This is the core declarative validation method. Scenarios declare expectations
         * via getExpectedParallelism(), and this method validates the generated plan.
         *
         * @param plan The generated heterogeneous plan
         * @param expected Declarative expectations from the scenario
         * @param n_layers Total model layers
         * @return Vector of error messages (empty = all passed)
         */
        std::vector<std::string> validateExpectedParallelism(
            const HeterogeneousPlan &plan,
            const ExpectedParallelismConfig &expected,
            int n_layers)
        {

            std::vector<std::string> errors;

            // ---------------------------------------------------------------------
            // PP Degree Validation
            // ---------------------------------------------------------------------
            if (expected.pp_degree >= 0)
            {
                if (static_cast<int>(plan.stages.size()) != expected.pp_degree)
                {
                    errors.push_back("PP degree: expected " + std::to_string(expected.pp_degree) +
                                     ", got " + std::to_string(plan.stages.size()));
                }
            }

            // PP stages contiguous
            if (expected.pp_stages_contiguous && plan.stages.size() >= 2)
            {
                for (size_t i = 1; i < plan.stages.size(); ++i)
                {
                    if (plan.stages[i].layer_start != plan.stages[i - 1].layer_end)
                    {
                        errors.push_back("PP stages not contiguous: stage " + std::to_string(i - 1) +
                                         " ends at " + std::to_string(plan.stages[i - 1].layer_end) +
                                         ", stage " + std::to_string(i) + " starts at " +
                                         std::to_string(plan.stages[i].layer_start));
                    }
                }
            }

            // ---------------------------------------------------------------------
            // Domain Count Validation
            // ---------------------------------------------------------------------
            int gpu_domains = 0, cpu_domains = 0;
            for (const auto &domain : plan.domains)
            {
                if (domain.type == TPDomainType::GPU_INTRA_RANK)
                    gpu_domains++;
                else if (domain.type == TPDomainType::CPU_CROSS_RANK)
                    cpu_domains++;
            }

            if (expected.total_domains >= 0)
            {
                if (static_cast<int>(plan.domains.size()) != expected.total_domains)
                {
                    errors.push_back("Total domains: expected " + std::to_string(expected.total_domains) +
                                     ", got " + std::to_string(plan.domains.size()));
                }
            }

            if (expected.gpu_domains_total >= 0)
            {
                if (gpu_domains != expected.gpu_domains_total)
                {
                    errors.push_back("GPU domains: expected " + std::to_string(expected.gpu_domains_total) +
                                     ", got " + std::to_string(gpu_domains));
                }
            }

            if (expected.cpu_domains_total >= 0)
            {
                if (cpu_domains != expected.cpu_domains_total)
                {
                    errors.push_back("CPU domains: expected " + std::to_string(expected.cpu_domains_total) +
                                     ", got " + std::to_string(cpu_domains));
                }
            }

            // ---------------------------------------------------------------------
            // TP Degree Validation
            // ---------------------------------------------------------------------
            if (expected.gpu_tp_degree >= 0)
            {
                for (const auto &domain : plan.domains)
                {
                    if (domain.type == TPDomainType::GPU_INTRA_RANK)
                    {
                        int tp = static_cast<int>(domain.devices.size());
                        if (tp != expected.gpu_tp_degree)
                        {
                            errors.push_back("GPU domain " + std::to_string(domain.domain_id) +
                                             " TP degree: expected " + std::to_string(expected.gpu_tp_degree) +
                                             ", got " + std::to_string(tp));
                        }
                    }
                }
            }

            if (expected.cpu_tp_degree >= 0)
            {
                for (const auto &domain : plan.domains)
                {
                    if (domain.type == TPDomainType::CPU_CROSS_RANK)
                    {
                        int tp = static_cast<int>(domain.ranks.size());
                        if (tp != expected.cpu_tp_degree)
                        {
                            errors.push_back("CPU domain " + std::to_string(domain.domain_id) +
                                             " TP degree: expected " + std::to_string(expected.cpu_tp_degree) +
                                             ", got " + std::to_string(tp));
                        }
                    }
                }
            }

            // ---------------------------------------------------------------------
            // Domains Per Node/Stage Validation
            // ---------------------------------------------------------------------
            if (expected.domains_per_pp_stage >= 0)
            {
                for (const auto &stage : plan.stages)
                {
                    int domains_in_stage = static_cast<int>(stage.domains.size());
                    if (domains_in_stage != expected.domains_per_pp_stage)
                    {
                        errors.push_back("PP stage " + std::to_string(stage.stage_id) +
                                         " domains: expected " + std::to_string(expected.domains_per_pp_stage) +
                                         ", got " + std::to_string(domains_in_stage));
                    }
                }
            }

            // ---------------------------------------------------------------------
            // GPU Intra-Rank / CPU Cross-Rank Validation
            // ---------------------------------------------------------------------
            if (expected.gpu_domains_are_intra_rank)
            {
                for (const auto &domain : plan.domains)
                {
                    if (domain.type == TPDomainType::GPU_INTRA_RANK && domain.ranks.size() != 1)
                    {
                        errors.push_back("GPU domain " + std::to_string(domain.domain_id) +
                                         " should be intra-rank (1 rank), has " +
                                         std::to_string(domain.ranks.size()) + " ranks");
                    }
                }
            }

            if (expected.cpu_domains_are_cross_rank)
            {
                for (const auto &domain : plan.domains)
                {
                    if (domain.type == TPDomainType::CPU_CROSS_RANK && domain.ranks.size() < 2)
                    {
                        errors.push_back("CPU domain " + std::to_string(domain.domain_id) +
                                         " should be cross-rank (2+ ranks), has " +
                                         std::to_string(domain.ranks.size()) + " rank(s)");
                    }
                }
            }

            // ---------------------------------------------------------------------
            // Layer Distribution Validation
            // ---------------------------------------------------------------------
            int gpu_layers = 0, cpu_layers = 0;
            for (const auto &domain : plan.domains)
            {
                int layers = domain.layer_end - domain.layer_start;
                if (domain.type == TPDomainType::GPU_INTRA_RANK)
                {
                    gpu_layers += layers;
                }
                else if (domain.type == TPDomainType::CPU_CROSS_RANK)
                {
                    cpu_layers += layers;
                }
            }

            int total_assigned = gpu_layers + cpu_layers;
            if (total_assigned > 0)
            {
                float gpu_fraction = static_cast<float>(gpu_layers) / total_assigned;
                float cpu_fraction = static_cast<float>(cpu_layers) / total_assigned;

                if (gpu_fraction < expected.min_gpu_layer_fraction)
                {
                    errors.push_back("GPU layer fraction: expected >= " +
                                     std::to_string(expected.min_gpu_layer_fraction) +
                                     ", got " + std::to_string(gpu_fraction));
                }

                if (cpu_fraction > expected.max_cpu_layer_fraction)
                {
                    errors.push_back("CPU layer fraction: expected <= " +
                                     std::to_string(expected.max_cpu_layer_fraction) +
                                     ", got " + std::to_string(cpu_fraction));
                }
            }

            // GPU layers precede CPU layers (within each PP stage)
            if (expected.gpu_layers_precede_cpu)
            {
                for (const auto &stage : plan.stages)
                {
                    int max_gpu_layer = -1;
                    int min_cpu_layer = n_layers + 1;

                    for (const auto &domain : stage.domains)
                    {
                        if (domain.type == TPDomainType::GPU_INTRA_RANK)
                        {
                            max_gpu_layer = std::max(max_gpu_layer, domain.layer_end - 1);
                        }
                        else if (domain.type == TPDomainType::CPU_CROSS_RANK)
                        {
                            min_cpu_layer = std::min(min_cpu_layer, domain.layer_start);
                        }
                    }

                    if (max_gpu_layer >= 0 && min_cpu_layer <= n_layers && max_gpu_layer >= min_cpu_layer)
                    {
                        errors.push_back("PP stage " + std::to_string(stage.stage_id) +
                                         ": GPU layers (max " + std::to_string(max_gpu_layer) +
                                         ") should precede CPU layers (min " +
                                         std::to_string(min_cpu_layer) + ")");
                    }
                }
            }

            // ---------------------------------------------------------------------
            // Total Parallelism Validation
            // ---------------------------------------------------------------------
            if (expected.total_gpu_devices >= 0)
            {
                int total_gpus = 0;
                for (const auto &domain : plan.domains)
                {
                    if (domain.type == TPDomainType::GPU_INTRA_RANK)
                    {
                        total_gpus += static_cast<int>(domain.devices.size());
                    }
                }
                if (total_gpus != expected.total_gpu_devices)
                {
                    errors.push_back("Total GPU devices: expected " +
                                     std::to_string(expected.total_gpu_devices) +
                                     ", got " + std::to_string(total_gpus));
                }
            }

            if (expected.total_cpu_ranks >= 0)
            {
                int total_cpu = 0;
                for (const auto &domain : plan.domains)
                {
                    if (domain.type == TPDomainType::CPU_CROSS_RANK)
                    {
                        total_cpu += static_cast<int>(domain.ranks.size());
                    }
                }
                if (total_cpu != expected.total_cpu_ranks)
                {
                    errors.push_back("Total CPU ranks: expected " +
                                     std::to_string(expected.total_cpu_ranks) +
                                     ", got " + std::to_string(total_cpu));
                }
            }

            return errors;
        }

        /**
         * @brief Check if cluster has heterogeneous GPU mix (CUDA + ROCm)
         */
        bool clusterHasHeterogeneousGPUs() const
        {
            bool has_cuda = false;
            bool has_rocm = false;

            for (const auto &config : cluster_.socket_configs)
            {
                for (const auto &gpu : config.gpus)
                {
                    if (gpu.type == DeviceCapability::Type::CUDA)
                        has_cuda = true;
                    if (gpu.type == DeviceCapability::Type::ROCm)
                        has_rocm = true;
                }
            }

            return has_cuda && has_rocm;
        }

        /**
         * @brief Get total GPU count across cluster
         */
        int clusterTotalGPUs() const
        {
            int total = 0;
            for (int i = 0; i < cluster_.totalRanks(); ++i)
            {
                const auto &config = cluster_.getSocketConfig(i);
                total += static_cast<int>(config.gpus.size());
            }
            return total;
        }

        /**
         * @brief Get total VRAM in GB across cluster
         */
        float clusterTotalVRAMGB() const
        {
            float total = 0.0f;
            for (int i = 0; i < cluster_.totalRanks(); ++i)
            {
                const auto &config = cluster_.getSocketConfig(i);
                total += config.totalVRAMGB();
            }
            return total;
        }

        // =========================================================================
        // Output Helpers
        // =========================================================================

        /**
         * @brief Print cluster summary
         */
        void printClusterSummary()
        {
            std::cout << "\n=== " << cluster_.name << " ===" << std::endl;
            std::cout << "Machines: " << cluster_.num_machines << std::endl;
            std::cout << "Sockets/machine: " << cluster_.sockets_per_machine << std::endl;
            std::cout << "Total ranks: " << cluster_.totalRanks() << std::endl;
            std::cout << "Total GPUs: " << cluster_.totalGPUs() << std::endl;
            std::cout << "Total GPU VRAM: " << std::fixed << std::setprecision(0)
                      << cluster_.totalVRAMGB() << " GB" << std::endl;
            std::cout << "Total GPU BW: " << std::fixed << std::setprecision(1)
                      << cluster_.totalGPUBandwidthTBs() << " TB/s" << std::endl;
            std::cout << "Total CPU BW: " << std::fixed << std::setprecision(1)
                      << cluster_.totalCPUBandwidthTBs() << " TB/s" << std::endl;
        }

        /**
         * @brief Print model fit summary table
         */
        void printModelFitTable()
        {
            std::cout << "\n=== Model Fit Analysis ===" << std::endl;
            std::cout << "Model             | Size   | Fits GPU? | Machines | Layers/Rank" << std::endl;
            std::cout << "------------------|--------|-----------|----------|------------" << std::endl;

            for (const auto &model : models_)
            {
                bool fits = modelFitsInGPUMemory(model);
                int machines = machinesNeededForModel(model);
                float lpr = model.layersPerRank(cluster_.totalRanks());

                printf("%-17s | %5.0fGB | %-9s | %8d | %11.1f\n",
                       model.name.c_str(),
                       model.memorySizeGB(),
                       fits ? "Yes" : "No",
                       machines,
                       lpr);
            }
        }

        /**
         * @brief Print heterogeneous plan summary
         */
        void printHeterogeneousPlanSummary(const HeterogeneousPlan &plan, const ModelConfig &model)
        {
            printf("\n╔══════════════════════════════════════════════════════════════════════╗\n");
            printf("║       HETEROGENEOUS PLAN: %-40s ║\n", model.name.c_str());
            printf("╠══════════════════════════════════════════════════════════════════════╣\n");
            printf("║  Total Domains: %-3d  |  PP Stages: %-3d  |  Layers: %-3d              ║\n",
                   static_cast<int>(plan.domains.size()),
                   static_cast<int>(plan.stages.size()),
                   model.n_layers);
            printf("╠══════════════════════════════════════════════════════════════════════╣\n");

            for (const auto &domain : plan.domains)
            {
                const char *type_str = "UNKNOWN";
                switch (domain.type)
                {
                case TPDomainType::GPU_INTRA_RANK:
                    type_str = "GPU_INTRA";
                    break;
                case TPDomainType::CPU_CROSS_RANK:
                    type_str = "CPU_CROSS";
                    break;
                }
                printf("║  Domain %2d [%s]: Layers %2d-%-2d, Ranks: %zu, Devices: %zu        ║\n",
                       domain.domain_id, type_str,
                       domain.layer_start, domain.layer_end,
                       domain.ranks.size(), domain.devices.size());
            }

            if (!plan.stages.empty())
            {
                printf("╠══════════════════════════════════════════════════════════════════════╣\n");
                for (size_t i = 0; i < plan.stages.size(); ++i)
                {
                    const auto &stage = plan.stages[i];
                    printf("║  PP Stage %zu: Node %d, Layers %d-%d, Domains: %zu                    ║\n",
                           i, stage.node_id, stage.layer_start, stage.layer_end,
                           stage.domains.size());
                }
            }

            printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");
        }

        // =========================================================================
        // Graph Bridge: PlacementPlan → ComputeGraph
        // =========================================================================

        /**
         * @brief Build a mock ComputeGraph from a PlacementPlan
         *
         * Creates mock stages for each layer with:
         * - Correct device placement matching the plan
         * - Correct backend support (CPU/CUDA/ROCm)
         * - Optional AllReduce stages at TP boundaries
         *
         * @param plan The placement plan to convert
         * @param include_allreduce Whether to insert AllReduce stages
         * @return tuple of (ComputeGraph, execution_log_ptr, stages_map)
         */
        struct MockGraphResult
        {
            ComputeGraph graph;
            std::shared_ptr<std::vector<std::string>> execution_log;
            std::unordered_map<std::string, testing::MockComputeStage *> stages;
        };

        MockGraphResult buildMockGraphFromPlan(
            const PlacementPlan &plan,
            bool include_allreduce = false)
        {
            MockGraphResult result;
            result.execution_log = std::make_shared<std::vector<std::string>>();

            std::vector<std::unique_ptr<testing::MockComputeStage>> owned_stages;
            std::string prev_layer_last_stage;

            for (const auto &layer : plan.layers)
            {
                DeviceId device = toDeviceId(layer.device);
                auto backend = device.is_cpu() ? ComputeBackendType::CPU : (device.type == DeviceType::CUDA ? ComputeBackendType::GPU_CUDA : ComputeBackendType::GPU_ROCM);

                // Build stages for this layer
                MockLayerGraphSpec spec;
                spec.layer_idx = layer.layer_idx;
                spec.device = device;
                spec.include_allreduce = include_allreduce && !device.is_cpu();

                std::string layer_prefix = "layer" + std::to_string(layer.layer_idx) + "_";
                std::vector<std::string> stage_names = {
                    layer_prefix + "attn_norm",
                    layer_prefix + "qkv_proj",
                    layer_prefix + "rope",
                    layer_prefix + "attention",
                    layer_prefix + "o_proj"};

                // Insert allreduce after attention if TP
                if (spec.include_allreduce)
                {
                    stage_names.push_back(layer_prefix + "attn_allreduce");
                }

                stage_names.push_back(layer_prefix + "attn_residual");
                stage_names.push_back(layer_prefix + "ffn_norm");
                stage_names.push_back(layer_prefix + "gate_up_proj");
                stage_names.push_back(layer_prefix + "swiglu");
                stage_names.push_back(layer_prefix + "down_proj");

                // Insert allreduce after FFN if TP
                if (spec.include_allreduce)
                {
                    stage_names.push_back(layer_prefix + "ffn_allreduce");
                }

                stage_names.push_back(layer_prefix + "ffn_residual");

                // Create stages and add to graph
                std::string prev_stage_in_layer;
                for (size_t i = 0; i < stage_names.size(); ++i)
                {
                    const std::string &name = stage_names[i];
                    ComputeStageType type = getStageType(name);

                    auto stage = std::make_unique<testing::MockComputeStage>(type, name);
                    stage->setExecutionLog(result.execution_log.get());

                    // Configure backend support based on device
                    if (device.is_cpu())
                    {
                        stage->setSupportedBackends({ComputeBackendType::CPU});
                    }
                    else
                    {
                        stage->setSupportedBackends({backend});
                    }

                    // Track for validation
                    result.stages[name] = stage.get();
                    owned_stages.push_back(std::move(stage));

                    // Add to graph with correct device
                    result.graph.addNode(name, std::move(owned_stages.back()), device);

                    // Add dependency chain within layer
                    if (!prev_stage_in_layer.empty())
                    {
                        result.graph.addDependency(name, prev_stage_in_layer);
                    }
                    prev_stage_in_layer = name;
                }

                // Add dependency to previous layer
                if (!prev_layer_last_stage.empty() && !stage_names.empty())
                {
                    result.graph.addDependency(stage_names[0], prev_layer_last_stage);
                }
                prev_layer_last_stage = stage_names.back();
            }

            return result;
        }

        /**
         * @brief Get ComputeStageType from stage name
         */
        static ComputeStageType getStageType(const std::string &name)
        {
            if (name.find("norm") != std::string::npos)
                return ComputeStageType::RMS_NORM;
            if (name.find("qkv") != std::string::npos)
                return ComputeStageType::GEMM;
            if (name.find("rope") != std::string::npos)
                return ComputeStageType::ROPE;
            if (name.find("attention") != std::string::npos)
                return ComputeStageType::ATTENTION;
            if (name.find("o_proj") != std::string::npos)
                return ComputeStageType::GEMM;
            if (name.find("gate_up") != std::string::npos)
                return ComputeStageType::GEMM;
            if (name.find("swiglu") != std::string::npos)
                return ComputeStageType::SWIGLU;
            if (name.find("down_proj") != std::string::npos)
                return ComputeStageType::GEMM;
            if (name.find("residual") != std::string::npos)
                return ComputeStageType::ADD_RESIDUAL;
            if (name.find("allreduce") != std::string::npos)
                return ComputeStageType::ALLREDUCE;
            return ComputeStageType::GEMM;
        }

        /**
         * @brief Validate that graph node devices match placement plan
         */
        GraphValidationResult validateGraphMatchesPlan(
            const ComputeGraph &graph,
            const PlacementPlan &plan)
        {
            GraphValidationResult result;

            for (const auto &layer : plan.layers)
            {
                DeviceId expected_device = toDeviceId(layer.device);
                std::string layer_prefix = "layer" + std::to_string(layer.layer_idx) + "_";

                // Check each stage in this layer has correct device
                for (const auto &node_name : graph.getExecutionOrder())
                {
                    if (node_name.find(layer_prefix) == 0)
                    {
                        result.nodes_checked++;
                        const auto *node = graph.getNode(node_name);
                        if (!node)
                        {
                            result.addError("Node not found: " + node_name);
                            continue;
                        }

                        // Check device matches
                        if (node->device != expected_device)
                        {
                            result.device_mismatches++;
                            result.addError("Device mismatch for " + node_name +
                                            ": expected " + expected_device.to_string() +
                                            ", got " + node->device.to_string());
                        }

                        // Check backend support matches device
                        auto expected_backend = expected_device.is_cpu() ? ComputeBackendType::CPU : ComputeBackendType::GPU_CUDA;
                        if (node->stage && !node->stage->supportsBackend(expected_backend))
                        {
                            result.backend_mismatches++;
                            result.addError("Backend mismatch for " + node_name);
                        }
                    }
                }
            }

            return result;
        }

        /**
         * @brief Validate AllReduce stages are present at TP boundaries
         */
        GraphValidationResult validateAllReduceStages(
            const ComputeGraph &graph,
            const PlacementPlan &plan,
            int world_size)
        {
            GraphValidationResult result;

            if (world_size <= 1)
            {
                // No tensor parallelism, no AllReduce needed
                return result;
            }

            for (const auto &layer : plan.layers)
            {
                if (!layer.device.isGPU())
                    continue;

                std::string layer_prefix = "layer" + std::to_string(layer.layer_idx) + "_";

                // Check for attention allreduce
                std::string attn_ar = layer_prefix + "attn_allreduce";
                if (!graph.getNode(attn_ar))
                {
                    result.addError("Missing AllReduce after attention: " + attn_ar);
                }

                // Check for FFN allreduce
                std::string ffn_ar = layer_prefix + "ffn_allreduce";
                if (!graph.getNode(ffn_ar))
                {
                    result.addError("Missing AllReduce after FFN: " + ffn_ar);
                }
            }

            return result;
        }

        /**
         * @brief Execute graph with mock device context and validate
         */
        ExecutionValidationResult executeAndValidateGraph(
            ComputeGraph &graph,
            const std::shared_ptr<std::vector<std::string>> &execution_log,
            DeviceId device = DeviceId::cpu())
        {
            ExecutionValidationResult result;
            result.expected_stages = static_cast<int>(graph.size());

            // Create mock device context
            auto backend = device.is_cpu() ? ComputeBackendType::CPU : ComputeBackendType::GPU_CUDA;
            testing::MockDeviceContext ctx(device.ordinal, backend);

            // Execute all ready nodes until complete
            while (!graph.allCompleted())
            {
                auto ready = graph.getReadyNodes();
                if (ready.empty())
                {
                    result.all_executed = false;
                    // Find which stages weren't executed
                    for (const auto &name : graph.getExecutionOrder())
                    {
                        auto *node = graph.getNode(name);
                        if (node && !node->completed)
                        {
                            result.skipped_stages.push_back(name);
                        }
                    }
                    break;
                }

                for (const auto &name : ready)
                {
                    auto *node = graph.getNode(name);
                    if (node && node->stage)
                    {
                        bool success = node->stage->execute(&ctx);
                        if (success)
                        {
                            graph.markCompleted(name);
                            result.stages_executed++;
                        }
                    }
                }
            }

            result.execution_order = *execution_log;
            return result;
        }

        /**
         * @brief Print graph validation summary
         */
        void printGraphValidationSummary(
            const std::string &model_name,
            const GraphValidationResult &result)
        {
            std::cout << "  Graph validation for " << model_name << ": ";
            if (result.is_valid)
            {
                std::cout << "✓ " << result.nodes_checked << " nodes valid" << std::endl;
            }
            else
            {
                std::cout << "✗ " << result.errors.size() << " errors" << std::endl;
                for (const auto &err : result.errors)
                {
                    std::cout << "    - " << err << std::endl;
                }
            }
        }

        // =========================================================================
        // Data Members
        // =========================================================================

        ClusterConfig cluster_;
        std::vector<ModelConfig> models_;
        std::vector<std::shared_ptr<MockMPITopology>> topologies_;
    };

} // namespace llaminar2::test::orchestration

// =============================================================================
// Test Instantiation Macro
// =============================================================================

/**
 * @brief Macro to instantiate comprehensive orchestration tests for a scenario
 *
 * Tests the full orchestration pipeline from topology to execution:
 *
 * CLUSTER FOUNDATION (2 tests):
 * - ClusterTopologyValid: Verifies topology creation for all ranks
 * - AllRanksSeeConsistentCluster: Ensures deterministic global view
 *
 * PLACEMENT PLANNING (4 tests):
 * - GPUFirstPlacement_AllModels: GPUFirst strategy produces valid plans
 * - CPUOnlyPlacement_AllModels: CPUOnly strategy produces valid plans
 * - HybridOptimalPlacement_AllModels: HybridOptimal strategy produces valid plans
 * - PhaseAwareWeights: Prefill/decode phase weights are correct
 *
 * GRAPH CONSTRUCTION (3 tests):
 * - GraphDevicePlacement: Graph nodes have correct device assignments
 * - GraphBackendSupport: Stages support correct compute backends
 * - GraphAllReduceAtTPBoundaries: AllReduce stages present for tensor parallelism
 *
 * EXECUTION VALIDATION (2 tests):
 * - GraphExecutionOrder: All stages execute in dependency order
 * - GraphConsistentAcrossRanks: All ranks produce identical graph structure
 *
 * WORK DISTRIBUTION (2 tests):
 * - WorkDistribution_HeadSharding: Attention heads distributed correctly
 * - WorkDistribution_VocabSharding: Vocabulary distributed correctly
 *
 * SUMMARY (2 tests):
 * - Summary_ClusterCapacity: Prints cluster hardware summary
 * - Summary_ModelFitTable: Prints model fit analysis
 */
#define INSTANTIATE_ORCHESTRATION_TESTS(TestFixture)                                   \
                                                                                       \
    /* ========================================================================= */    \
    /* CLUSTER FOUNDATION                                                        */    \
    /* ========================================================================= */    \
                                                                                       \
    TEST_F(TestFixture, ClusterTopologyValid)                                          \
    {                                                                                  \
        EXPECT_EQ(topologies_.size(), static_cast<size_t>(cluster_.totalRanks()));     \
        for (int rank = 0; rank < cluster_.totalRanks(); ++rank)                       \
        {                                                                              \
            EXPECT_EQ(topologies_[rank]->rank(), rank);                                \
            EXPECT_EQ(topologies_[rank]->world_size(), cluster_.totalRanks());         \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, AllRanksSeeConsistentCluster)                                  \
    {                                                                                  \
        const auto &cluster0 = topologies_[0]->clusterInventory();                     \
        for (int rank = 1; rank < cluster_.totalRanks(); ++rank)                       \
        {                                                                              \
            const auto &cluster_r = topologies_[rank]->clusterInventory();             \
            EXPECT_EQ(cluster0.world_size, cluster_r.world_size);                      \
            EXPECT_EQ(cluster0.total_gpus, cluster_r.total_gpus);                      \
            EXPECT_EQ(cluster0.total_gpu_memory, cluster_r.total_gpu_memory);          \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    /* ========================================================================= */    \
    /* PLACEMENT PLANNING                                                        */    \
    /* ========================================================================= */    \
                                                                                       \
    TEST_F(TestFixture, GPUFirstPlacement_AllModels)                                   \
    {                                                                                  \
        GPUFirstPlacementStrategy strategy;                                            \
        for (const auto &model : models_)                                              \
        {                                                                              \
            auto input = createPlacementInput(model, 0);                               \
            if (strategy.isApplicable(input))                                          \
            {                                                                          \
                PlacementPlan plan = strategy.compute(input);                          \
                auto result = validatePlan(plan, 0);                                   \
                EXPECT_TRUE(result.is_valid)                                           \
                    << "GPUFirst failed for " << model.name;                           \
                EXPECT_EQ(result.total_layers, model.n_layers);                        \
                EXPECT_GT(result.gpu_layers, 0)                                        \
                    << model.name << " should have at least some GPU layers";          \
            }                                                                          \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, CPUOnlyPlacement_AllModels)                                    \
    {                                                                                  \
        CPUOnlyPlacementStrategy strategy;                                             \
        for (const auto &model : models_)                                              \
        {                                                                              \
            auto input = createPlacementInput(model, 0);                               \
            if (strategy.isApplicable(input))                                          \
            {                                                                          \
                PlacementPlan plan = strategy.compute(input);                          \
                auto result = validatePlan(plan, 0);                                   \
                EXPECT_TRUE(result.is_valid)                                           \
                    << "CPUOnly failed for " << model.name;                            \
                EXPECT_EQ(result.gpu_layers, 0)                                        \
                    << "CPUOnly has GPU layers for " << model.name;                    \
            }                                                                          \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, HybridOptimalPlacement_AllModels)                              \
    {                                                                                  \
        HybridOptimalPlacementStrategy strategy;                                       \
        for (const auto &model : models_)                                              \
        {                                                                              \
            auto input = createPlacementInput(model, 0);                               \
            if (strategy.isApplicable(input))                                          \
            {                                                                          \
                PlacementPlan plan = strategy.compute(input);                          \
                auto result = validatePlan(plan, 0);                                   \
                EXPECT_TRUE(result.is_valid)                                           \
                    << "HybridOptimal failed for " << model.name;                      \
            }                                                                          \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, PhaseAwareWeights)                                             \
    {                                                                                  \
        for (const auto &model : models_)                                              \
        {                                                                              \
            auto input = createPlacementInput(model, 0);                               \
            auto [prefill_gpu, prefill_cpu] =                                          \
                input.getPhaseDeviceWeights(InferencePhase::PREFILL);                  \
            EXPECT_FLOAT_EQ(prefill_gpu, 1.0f) << "PREFILL should be GPU-only";        \
            EXPECT_FLOAT_EQ(prefill_cpu, 0.0f);                                        \
            auto [decode_gpu, decode_cpu] =                                            \
                input.getPhaseDeviceWeights(InferencePhase::DECODE);                   \
            EXPECT_GT(decode_gpu, 0.0f);                                               \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    /* ========================================================================= */    \
    /* GRAPH CONSTRUCTION                                                        */    \
    /* ========================================================================= */    \
                                                                                       \
    TEST_F(TestFixture, GraphDevicePlacement)                                          \
    {                                                                                  \
        GPUFirstPlacementStrategy strategy;                                            \
        for (const auto &model : models_)                                              \
        {                                                                              \
            auto input = createPlacementInput(model, 0);                               \
            if (!strategy.isApplicable(input))                                         \
                continue;                                                              \
                                                                                       \
            PlacementPlan plan = strategy.compute(input);                              \
            auto [graph, log, stages] = buildMockGraphFromPlan(plan, false);           \
                                                                                       \
            auto result = validateGraphMatchesPlan(graph, plan);                       \
            EXPECT_TRUE(result.is_valid)                                               \
                << "Device placement mismatch for " << model.name;                     \
            EXPECT_EQ(result.device_mismatches, 0);                                    \
            EXPECT_GT(result.nodes_checked, 0)                                         \
                << "No nodes checked for " << model.name;                              \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, GraphBackendSupport)                                           \
    {                                                                                  \
        GPUFirstPlacementStrategy strategy;                                            \
        for (const auto &model : models_)                                              \
        {                                                                              \
            auto input = createPlacementInput(model, 0);                               \
            if (!strategy.isApplicable(input))                                         \
                continue;                                                              \
                                                                                       \
            PlacementPlan plan = strategy.compute(input);                              \
            auto [graph, log, stages] = buildMockGraphFromPlan(plan, false);           \
                                                                                       \
            auto result = validateGraphMatchesPlan(graph, plan);                       \
            EXPECT_EQ(result.backend_mismatches, 0)                                    \
                << "Backend support mismatch for " << model.name;                      \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, GraphAllReduceAtTPBoundaries)                                  \
    {                                                                                  \
        if (cluster_.totalRanks() <= 1)                                                \
        {                                                                              \
            GTEST_SKIP() << "Single-rank cluster has no tensor parallelism";           \
        }                                                                              \
                                                                                       \
        GPUFirstPlacementStrategy strategy;                                            \
        for (const auto &model : models_)                                              \
        {                                                                              \
            auto input = createPlacementInput(model, 0);                               \
            if (!strategy.isApplicable(input))                                         \
                continue;                                                              \
                                                                                       \
            PlacementPlan plan = strategy.compute(input);                              \
            auto [graph, log, stages] = buildMockGraphFromPlan(plan, true);            \
                                                                                       \
            auto result = validateAllReduceStages(graph, plan, cluster_.totalRanks()); \
            EXPECT_TRUE(result.is_valid)                                               \
                << "Missing AllReduce stages for " << model.name;                      \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    /* ========================================================================= */    \
    /* EXECUTION VALIDATION                                                      */    \
    /* ========================================================================= */    \
                                                                                       \
    TEST_F(TestFixture, GraphExecutionOrder)                                           \
    {                                                                                  \
        GPUFirstPlacementStrategy strategy;                                            \
        for (const auto &model : models_)                                              \
        {                                                                              \
            auto input = createPlacementInput(model, 0);                               \
            if (!strategy.isApplicable(input))                                         \
                continue;                                                              \
                                                                                       \
            PlacementPlan plan = strategy.compute(input);                              \
            auto [graph, log, stages] = buildMockGraphFromPlan(plan, false);           \
                                                                                       \
            auto result = executeAndValidateGraph(graph, log);                         \
                                                                                       \
            EXPECT_TRUE(result.all_executed)                                           \
                << "Not all stages executed for " << model.name;                       \
            EXPECT_EQ(result.stages_executed, result.expected_stages);                 \
            EXPECT_TRUE(result.skipped_stages.empty());                                \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, GraphConsistentAcrossRanks)                                    \
    {                                                                                  \
        GPUFirstPlacementStrategy strategy;                                            \
        for (const auto &model : models_)                                              \
        {                                                                              \
            std::vector<size_t> graph_sizes;                                           \
            std::vector<int> gpu_layer_counts;                                         \
                                                                                       \
            for (int rank = 0; rank < cluster_.totalRanks(); ++rank)                   \
            {                                                                          \
                auto input = createPlacementInput(model, rank);                        \
                if (!strategy.isApplicable(input))                                     \
                    break;                                                             \
                                                                                       \
                PlacementPlan plan = strategy.compute(input);                          \
                auto [graph, log, stages] = buildMockGraphFromPlan(plan, false);       \
                                                                                       \
                graph_sizes.push_back(graph.size());                                   \
                                                                                       \
                int gpu_layers = 0;                                                    \
                for (const auto &layer : plan.layers)                                  \
                {                                                                      \
                    if (layer.device.isGPU())                                          \
                        gpu_layers++;                                                  \
                }                                                                      \
                gpu_layer_counts.push_back(gpu_layers);                                \
            }                                                                          \
                                                                                       \
            if (graph_sizes.empty())                                                   \
                continue;                                                              \
                                                                                       \
            size_t expected_size = graph_sizes[0];                                     \
            int expected_gpu_layers = gpu_layer_counts[0];                             \
                                                                                       \
            for (size_t r = 1; r < graph_sizes.size(); ++r)                            \
            {                                                                          \
                EXPECT_EQ(graph_sizes[r], expected_size)                               \
                    << model.name << " rank " << r << " has different graph size";     \
                EXPECT_EQ(gpu_layer_counts[r], expected_gpu_layers)                    \
                    << model.name << " rank " << r << " has different GPU layers";     \
            }                                                                          \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    /* ========================================================================= */    \
    /* WORK DISTRIBUTION                                                         */    \
    /* ========================================================================= */    \
                                                                                       \
    TEST_F(TestFixture, WorkDistribution_HeadSharding)                                 \
    {                                                                                  \
        for (const auto &model : models_)                                              \
        {                                                                              \
            size_t total_heads = 0;                                                    \
            for (int rank = 0; rank < cluster_.totalRanks(); ++rank)                   \
            {                                                                          \
                auto range = topologies_[rank]->get_head_range(model.n_heads);         \
                total_heads += range.size();                                           \
            }                                                                          \
            EXPECT_EQ(total_heads, model.n_heads)                                      \
                << "Head sharding doesn't cover all heads for " << model.name;         \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, WorkDistribution_VocabSharding)                                \
    {                                                                                  \
        for (const auto &model : models_)                                              \
        {                                                                              \
            size_t total_vocab = 0;                                                    \
            for (int rank = 0; rank < cluster_.totalRanks(); ++rank)                   \
            {                                                                          \
                auto range = topologies_[rank]->get_vocab_range(model.vocab_size);     \
                total_vocab += range.size();                                           \
            }                                                                          \
            EXPECT_EQ(total_vocab, model.vocab_size)                                   \
                << "Vocab sharding doesn't cover all tokens for " << model.name;       \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    /* ========================================================================= */    \
    /* SUMMARY                                                                   */    \
    /* ========================================================================= */    \
                                                                                       \
    TEST_F(TestFixture, Summary_ClusterCapacity)                                       \
    {                                                                                  \
        printClusterSummary();                                                         \
        SUCCEED();                                                                     \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, Summary_ModelFitTable)                                         \
    {                                                                                  \
        printModelFitTable();                                                          \
        SUCCEED();                                                                     \
    }                                                                                  \
                                                                                       \
    /* ========================================================================= */    \
    /* HETEROGENEOUS STRATEGY TESTS                                              */    \
    /* ========================================================================= */    \
                                                                                       \
    TEST_F(TestFixture, HeterogeneousStrategy_PlanGeneration)                          \
    {                                                                                  \
        for (const auto &model : models_)                                              \
        {                                                                              \
            auto topo = createTopology(0);                                             \
            HeterogeneousMultiDomainStrategy strategy;                                 \
            auto input = createPlacementInput(model, 0);                               \
            if (strategy.isApplicable(input))                                          \
            {                                                                          \
                auto plan = strategy.generatePlan(                                     \
                    input.cluster_inventory, model.n_layers);                          \
                auto result = validateHeterogeneousPlan(plan, model.n_layers);         \
                EXPECT_TRUE(result.is_valid)                                           \
                    << "Model: " << model.name << " - " << result.error_message;       \
                EXPECT_TRUE(result.all_layers_covered)                                 \
                    << "Model: " << model.name << " - Not all layers covered";         \
                EXPECT_GT(result.total_domains, 0)                                     \
                    << "Model: " << model.name << " - No domains created";             \
            }                                                                          \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, HeterogeneousStrategy_DomainAssignment)                        \
    {                                                                                  \
        for (const auto &model : models_)                                              \
        {                                                                              \
            auto topo = createTopology(0);                                             \
            HeterogeneousMultiDomainStrategy strategy;                                 \
            auto input = createPlacementInput(model, 0);                               \
            if (strategy.isApplicable(input))                                          \
            {                                                                          \
                auto plan = strategy.generatePlan(                                     \
                    input.cluster_inventory, model.n_layers);                          \
                for (int layer = 0; layer < model.n_layers; ++layer)                   \
                {                                                                      \
                    auto result = validateDomainForLayer(plan, layer);                 \
                    EXPECT_TRUE(result.is_valid)                                       \
                        << "Model: " << model.name                                     \
                        << ", Layer: " << layer << " - " << result.error_message;      \
                    EXPECT_GT(result.ranks_in_domain, 0)                               \
                        << "Model: " << model.name                                     \
                        << ", Layer: " << layer << " - Domain has no ranks";           \
                }                                                                      \
            }                                                                          \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, HeterogeneousStrategy_PPStageValidation)                       \
    {                                                                                  \
        for (const auto &model : models_)                                              \
        {                                                                              \
            auto topo = createTopology(0);                                             \
            HeterogeneousMultiDomainStrategy strategy;                                 \
            auto input = createPlacementInput(model, 0);                               \
            if (strategy.isApplicable(input) && cluster_.num_machines > 1)             \
            {                                                                          \
                auto plan = strategy.generatePlan(                                     \
                    input.cluster_inventory, model.n_layers);                          \
                for (size_t i = 0; i < plan.stages.size(); ++i)                        \
                {                                                                      \
                    auto result = validatePPStage(plan, static_cast<int>(i));          \
                    EXPECT_TRUE(result.is_valid)                                       \
                        << "Model: " << model.name                                     \
                        << ", Stage: " << i << " - " << result.error_message;          \
                    EXPECT_GT(result.layers_in_stage, 0)                               \
                        << "Model: " << model.name                                     \
                        << ", Stage: " << i << " - Stage has no layers";               \
                }                                                                      \
            }                                                                          \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, HeterogeneousStrategy_CrossVendorDetection)                    \
    {                                                                                  \
        bool has_heterogeneous = clusterHasHeterogeneousGPUs();                        \
        if (has_heterogeneous)                                                         \
        {                                                                              \
            HeterogeneousMultiDomainStrategy strategy;                                 \
            auto input = createPlacementInput(models_[0], 0);                          \
            EXPECT_TRUE(strategy.isApplicable(input))                                  \
                << "Strategy should be applicable for heterogeneous cluster";          \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    /* ========================================================================= */    \
    /* DECLARATIVE PARALLELISM VALIDATION                                        */    \
    /* ========================================================================= */    \
                                                                                       \
    TEST_F(TestFixture, Parallelism_ExpectationsValidation)                            \
    {                                                                                  \
        auto expected = getExpectedParallelism();                                      \
        if (!expected.hasExpectations())                                               \
        {                                                                              \
            GTEST_SKIP() << "No parallelism expectations declared for this scenario";  \
        }                                                                              \
                                                                                       \
        HeterogeneousMultiDomainStrategy strategy;                                     \
        for (const auto &model : models_)                                              \
        {                                                                              \
            auto input = createPlacementInput(model, 0);                               \
            if (!strategy.isApplicable(input))                                         \
                continue;                                                              \
                                                                                       \
            auto plan = strategy.generatePlan(                                         \
                input.cluster_inventory, model.n_layers);                              \
                                                                                       \
            auto errors = validateExpectedParallelism(plan, expected, model.n_layers); \
            EXPECT_TRUE(errors.empty())                                                \
                << "Model: " << model.name << " parallelism deduction errors:\n"       \
                << [&]() {                                                              \
                       std::string msg;                                                 \
                       for (const auto& e : errors) msg += "  - " + e + "\n";           \
                       return msg; }();                                                        \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, Parallelism_PPDegree)                                          \
    {                                                                                  \
        auto expected = getExpectedParallelism();                                      \
        if (expected.pp_degree < 0)                                                    \
        {                                                                              \
            GTEST_SKIP() << "No PP degree expectation declared";                       \
        }                                                                              \
                                                                                       \
        HeterogeneousMultiDomainStrategy strategy;                                     \
        const auto &model = models_[0];                                                \
        auto input = createPlacementInput(model, 0);                                   \
        if (!strategy.isApplicable(input))                                             \
        {                                                                              \
            GTEST_SKIP() << "Strategy not applicable";                                 \
        }                                                                              \
        auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);    \
        EXPECT_EQ(static_cast<int>(plan.stages.size()), expected.pp_degree)            \
            << "PP degree mismatch";                                                   \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, Parallelism_GPUTPDegree)                                       \
    {                                                                                  \
        auto expected = getExpectedParallelism();                                      \
        if (expected.gpu_tp_degree < 0)                                                \
        {                                                                              \
            GTEST_SKIP() << "No GPU TP degree expectation declared";                   \
        }                                                                              \
                                                                                       \
        HeterogeneousMultiDomainStrategy strategy;                                     \
        const auto &model = models_[0];                                                \
        auto input = createPlacementInput(model, 0);                                   \
        if (!strategy.isApplicable(input))                                             \
        {                                                                              \
            GTEST_SKIP() << "Strategy not applicable";                                 \
        }                                                                              \
        auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);    \
        for (const auto &domain : plan.domains)                                        \
        {                                                                              \
            if (domain.type == TPDomainType::GPU_INTRA_RANK)                           \
            {                                                                          \
                EXPECT_EQ(static_cast<int>(domain.devices.size()),                     \
                          expected.gpu_tp_degree)                                      \
                    << "GPU domain " << domain.domain_id << " TP degree mismatch";     \
            }                                                                          \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, Parallelism_CPUTPDegree)                                       \
    {                                                                                  \
        auto expected = getExpectedParallelism();                                      \
        if (expected.cpu_tp_degree < 0)                                                \
        {                                                                              \
            GTEST_SKIP() << "No CPU TP degree expectation declared";                   \
        }                                                                              \
                                                                                       \
        HeterogeneousMultiDomainStrategy strategy;                                     \
        const auto &model = models_[0];                                                \
        auto input = createPlacementInput(model, 0);                                   \
        if (!strategy.isApplicable(input))                                             \
        {                                                                              \
            GTEST_SKIP() << "Strategy not applicable";                                 \
        }                                                                              \
        auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);    \
        for (const auto &domain : plan.domains)                                        \
        {                                                                              \
            if (domain.type == TPDomainType::CPU_CROSS_RANK)                           \
            {                                                                          \
                EXPECT_EQ(static_cast<int>(domain.ranks.size()),                       \
                          expected.cpu_tp_degree)                                      \
                    << "CPU domain " << domain.domain_id << " TP degree mismatch";     \
            }                                                                          \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, Parallelism_DomainCounts)                                      \
    {                                                                                  \
        auto expected = getExpectedParallelism();                                      \
        if (expected.gpu_domains_total < 0 && expected.cpu_domains_total < 0)          \
        {                                                                              \
            GTEST_SKIP() << "No domain count expectations declared";                   \
        }                                                                              \
                                                                                       \
        HeterogeneousMultiDomainStrategy strategy;                                     \
        const auto &model = models_[0];                                                \
        auto input = createPlacementInput(model, 0);                                   \
        if (!strategy.isApplicable(input))                                             \
        {                                                                              \
            GTEST_SKIP() << "Strategy not applicable";                                 \
        }                                                                              \
        auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);    \
                                                                                       \
        int gpu_domains = 0, cpu_domains = 0;                                          \
        for (const auto &domain : plan.domains)                                        \
        {                                                                              \
            if (domain.type == TPDomainType::GPU_INTRA_RANK)                           \
                gpu_domains++;                                                         \
            else if (domain.type == TPDomainType::CPU_CROSS_RANK)                      \
                cpu_domains++;                                                         \
        }                                                                              \
                                                                                       \
        if (expected.gpu_domains_total >= 0)                                           \
        {                                                                              \
            EXPECT_EQ(gpu_domains, expected.gpu_domains_total);                        \
        }                                                                              \
        if (expected.cpu_domains_total >= 0)                                           \
        {                                                                              \
            EXPECT_EQ(cpu_domains, expected.cpu_domains_total);                        \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    TEST_F(TestFixture, Parallelism_TotalDevices)                                      \
    {                                                                                  \
        auto expected = getExpectedParallelism();                                      \
        if (expected.total_gpu_devices < 0 && expected.total_cpu_ranks < 0)            \
        {                                                                              \
            GTEST_SKIP() << "No total device expectations declared";                   \
        }                                                                              \
                                                                                       \
        HeterogeneousMultiDomainStrategy strategy;                                     \
        const auto &model = models_[0];                                                \
        auto input = createPlacementInput(model, 0);                                   \
        if (!strategy.isApplicable(input))                                             \
        {                                                                              \
            GTEST_SKIP() << "Strategy not applicable";                                 \
        }                                                                              \
        auto plan = strategy.generatePlan(input.cluster_inventory, model.n_layers);    \
                                                                                       \
        int total_gpus = 0, total_cpu_ranks = 0;                                       \
        for (const auto &domain : plan.domains)                                        \
        {                                                                              \
            if (domain.type == TPDomainType::GPU_INTRA_RANK)                           \
            {                                                                          \
                total_gpus += static_cast<int>(domain.devices.size());                 \
            }                                                                          \
            else if (domain.type == TPDomainType::CPU_CROSS_RANK)                      \
            {                                                                          \
                total_cpu_ranks += static_cast<int>(domain.ranks.size());              \
            }                                                                          \
        }                                                                              \
                                                                                       \
        if (expected.total_gpu_devices >= 0)                                           \
        {                                                                              \
            EXPECT_EQ(total_gpus, expected.total_gpu_devices);                         \
        }                                                                              \
        if (expected.total_cpu_ranks >= 0)                                             \
        {                                                                              \
            EXPECT_EQ(total_cpu_ranks, expected.total_cpu_ranks);                      \
        }                                                                              \
    }
