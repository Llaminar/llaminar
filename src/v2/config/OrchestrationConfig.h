/**
 * @file OrchestrationConfig.h
 * @brief Configuration structures for heterogeneous orchestration
 *
 * Defines configuration for:
 * - Device assignment modes (auto, local, explicit)
 * - Tensor parallelism scope and configuration
 * - Pipeline parallelism configuration
 * - Named domains for complex multi-GPU scenarios
 * - Collective backend selection
 *
 * Usage patterns:
 *
 * Simple TP across 2 GPUs:
 *   --tp 2 --device cuda:0 --device cuda:1
 *
 * Explicit domain with NCCL:
 *   --define-domain "gpu_tp=cuda:0,cuda:1;backend=nccl"
 *
 * Hybrid PP+TP:
 *   --pp 2 --define-domain "stage0=cuda:0,cuda:1" --define-domain "stage1=rocm:0,rocm:1"
 *   --pp-stage "0=stage0:0-13" --pp-stage "1=stage1:14-27"
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "backends/GlobalDeviceAddress.h"
#include "CollectiveBackendType.h"
#include "execution/config/RuntimeConfig.h" // For FusedAttentionBackend
#include "execution/parallelism_tree/ParallelismTree.h"
#include <string>
#include <vector>
#include <optional>
#include <utility>
#include <memory>

namespace llaminar2
{

    // =========================================================================
    // Enums
    // =========================================================================

    /**
     * @brief Scope of tensor parallelism
     */
    enum class TPScope
    {
        AUTO,       ///< Automatically determine based on topology
        LOCAL,      ///< TP within single MPI rank (intra-rank: NVLink, PCIeBAR, NCCL)
        NODE_LOCAL, ///< TP across MPI ranks on same physical node (UPI, shmem, cross-process NCCL)
        GLOBAL,     ///< TP across all nodes (MPI over InfiniBand/Ethernet)
        HYBRID      ///< Hierarchical: local TP + global PP
    };

    /**
     * @brief How devices are assigned to MPI ranks
     */
    enum class DeviceAssignmentMode
    {
        AUTO,        ///< Automatic based on topology detection
        LOCAL_GPU,   ///< Each rank gets its local GPU (rank N -> GPU N)
        ROUND_ROBIN, ///< Round-robin across available devices
        EXPLICIT     ///< Explicit mapping via --device-map
    };

    /**
     * @brief How layers are split across PP stages
     */
    enum class PPSplitMode
    {
        EQUAL,    ///< Equal number of layers per stage
        WEIGHTED, ///< Proportional to device compute capacity
        MANUAL    ///< Manual layer assignment via --pp-stage
    };

    /**
     * @brief MPI bootstrap tuning profile
     */
    enum class MPIProfile
    {
        AUTO,  ///< Conservative defaults, auto-upgrade to tuned for explicit CPU intent
        TUNED, ///< Force tuned CPU-centric bootstrap defaults
    };

    // CollectiveBackendType is defined in CollectiveBackendType.h

    // =========================================================================
    // Conversion functions
    // =========================================================================

    const char *tpScopeToString(TPScope scope);
    const char *deviceAssignmentModeToString(DeviceAssignmentMode mode);
    const char *ppSplitModeToString(PPSplitMode mode);
    const char *mpiProfileToString(MPIProfile profile);
    // collectiveBackendTypeToString and parseCollectiveBackendType declared in CollectiveBackendType.h

    std::optional<TPScope> parseTpScope(const std::string &str);
    std::optional<DeviceAssignmentMode> parseDeviceAssignmentMode(const std::string &str);
    std::optional<PPSplitMode> parsePpSplitMode(const std::string &str);
    std::optional<MPIProfile> parseMPIProfile(const std::string &str);

    // =========================================================================
    // DomainDefinition
    // =========================================================================

    /**
     * @brief Named domain definition for complex TP configurations
     *
     * Format: "name=device1,device2,...[;weights=w1,w2,...][;backend=type]"
     *
     * Examples:
     *   "gpu_tp=cuda:0,cuda:1" -> Equal split across 2 CUDA GPUs
     *   "mixed=cuda:0,rocm:0;weights=0.73,0.27" -> Proportional split
     *   "fast=cuda:0,cuda:1;backend=nccl" -> Force NCCL backend
     */
    struct DomainDefinition
    {
        std::string name;                         ///< Domain name (e.g., "gpu_tp", "stage0")
        std::vector<GlobalDeviceAddress> devices; ///< Devices in this domain
        std::vector<float> weights;               ///< Optional: proportional weights (must sum to 1.0)
        CollectiveBackendType backend = CollectiveBackendType::AUTO;

        /**
         * @brief Parse domain definition from string
         * @param spec Format: "name=device1,device2[;weights=w1,w2][;backend=type]"
         * @return Parsed DomainDefinition
         * @throws std::invalid_argument if parsing fails
         */
        static DomainDefinition parse(const std::string &spec);

        /**
         * @brief Try to parse domain definition from string
         * @param spec Format: "name=device1,device2[;weights=w1,w2][;backend=type]"
         * @return Parsed DomainDefinition or nullopt if invalid
         */
        static std::optional<DomainDefinition> tryParse(const std::string &spec);

        /**
         * @brief Check if weights are specified
         */
        bool hasWeights() const { return !weights.empty(); }

        /**
         * @brief Validate the domain definition
         * @return Empty vector if valid, otherwise list of errors
         */
        std::vector<std::string> validate() const;

        /**
         * @brief String representation for logging
         */
        std::string toString() const;
    };

    // =========================================================================
    // PPStageDefinition
    // =========================================================================

    /**
     * @brief Pipeline parallel stage definition
     *
     * Maps a PP stage to a domain and layer range.
     *
     * Format: "stage_id=domain_name:first_layer-last_layer"
     *
     * Examples:
     *   "0=gpu_tp:0-13" -> Stage 0 uses domain "gpu_tp" for layers 0-13
     *   "1=cpu_tp:14-27" -> Stage 1 uses domain "cpu_tp" for layers 14-27
     */
    struct PPStageDefinition
    {
        int stage_id;            ///< PP stage index (0, 1, 2, ...)
        std::string domain_name; ///< Name of domain handling this stage
        int first_layer;         ///< First layer index (inclusive)
        int last_layer;          ///< Last layer index (inclusive)

        /**
         * @brief Parse PP stage definition from string
         * @param spec Format: "stage_id=domain_name:first_layer-last_layer"
         * @return Parsed PPStageDefinition
         * @throws std::invalid_argument if parsing fails
         */
        static PPStageDefinition parse(const std::string &spec);

        /**
         * @brief Try to parse PP stage definition from string
         * @param spec Format: "stage_id=domain_name:first_layer-last_layer"
         * @return Parsed PPStageDefinition or nullopt if invalid
         */
        static std::optional<PPStageDefinition> tryParse(const std::string &spec);

        /**
         * @brief Get number of layers in this stage
         */
        int layerCount() const { return last_layer - first_layer + 1; }

        /**
         * @brief Validate the stage definition
         * @return Empty vector if valid, otherwise list of errors
         */
        std::vector<std::string> validate() const;

        /**
         * @brief String representation for logging
         */
        std::string toString() const;
    };

    // =========================================================================
    // OrchestrationConfig
    // =========================================================================

    /**
     * @brief Complete orchestration configuration
     *
     * Contains all configuration for device assignment, tensor parallelism,
     * pipeline parallelism, and collective backends.
     */
    struct OrchestrationConfig
    {
        // =========================================================================
        // Introspection Flags
        // =========================================================================

        bool dry_run = false;           ///< Show configuration without executing
        bool explain_placement = false; ///< Explain device placement decisions
        bool show_topology = false;     ///< Show detected topology and exit
        bool show_numa = false;         ///< Show NUMA configuration and exit
        bool validate_only = false;     ///< Validate configuration without running

        // =========================================================================
        // Device Assignment
        // =========================================================================

        DeviceAssignmentMode device_mode = DeviceAssignmentMode::AUTO;

        /// Device for this specific rank (--device cuda:0)
        std::optional<GlobalDeviceAddress> device_for_this_rank;

        /// True when --device explicitly includes NUMA (e.g., 1:rocm:0, host:1:rocm:0)
        bool device_for_this_rank_numa_explicit = false;

        /// True when --device cpu shorthand is used (CPU TP across local NUMA nodes)
        bool cpu_global_tp_all_local = false;

        /// Explicit device map: rank -> device (--device-map "0=cuda:0,1=cuda:1")
        std::vector<std::pair<int, GlobalDeviceAddress>> device_map;

        /// For device_map entries: rank -> whether NUMA was explicitly specified
        std::vector<std::pair<int, bool>> device_map_numa_explicit;

        // =========================================================================
        // Named Domains (Complex Scenarios)
        // =========================================================================

        /// Domain definitions (--define-domain "name=devices...")
        std::vector<DomainDefinition> domain_definitions;

        /// PP stage definitions (--pp-stage "stage=domain:layers")
        std::vector<PPStageDefinition> pp_stage_definitions;

        // =========================================================================
        // Simple TP Options
        // =========================================================================

        int tp_degree = 1; ///< TP parallelism degree (--tp)
        TPScope tp_scope = TPScope::AUTO;

        /// Explicit TP devices (--tp-devices "cuda:0,cuda:1")
        std::vector<GlobalDeviceAddress> tp_devices;

        /// TP weights for proportional distribution (--tp-weights "0.73,0.27")
        std::vector<float> tp_weights;

        /// Local TP degree for hybrid parallelism (--tp-local)
        int tp_local_degree = 1;

        /// Global TP degree for hybrid parallelism (--tp-global)
        int tp_global_degree = 1;

        // =========================================================================
        // Simple PP Options
        // =========================================================================

        int pp_degree = 1; ///< PP parallelism degree (--pp)
        PPSplitMode pp_split = PPSplitMode::EQUAL;

        // =========================================================================
        // Layer Placement
        // =========================================================================

        int cpu_layers = 0;            ///< Number of layers on CPU (--cpu-layers)
        bool cpu_layers_first = false; ///< Put CPU layers first (--cpu-layers-first)

        // =========================================================================
        // Backend
        // =========================================================================

        CollectiveBackendType default_backend = CollectiveBackendType::AUTO;

        // =========================================================================
        // Config File
        // =========================================================================

        std::string config_file_path; ///< Path to YAML config file (--config)

        // =========================================================================
        // Recursive Topology Tree (Global PP Phase 8)
        // =========================================================================

        /// Inline topology specification (--topology "PP(...)")
        std::string topology_string;

        /// Path to topology YAML file (--topology-file)
        std::string topology_file_path;

        /// Parsed topology tree (populated by parser if --topology or --topology-file specified)
        std::optional<ParallelismTree> topology_tree;

        // =========================================================================
        // Model Configuration
        // =========================================================================

        std::string model_path; ///< Path to GGUF model file
        int max_seq_len = 4096; ///< Maximum sequence length
        bool use_mmap = true;   ///< Use memory-mapped file loading

        // =========================================================================
        // Inference Configuration
        // =========================================================================

        std::string prompt; ///< Input prompt
        int n_predict = -1; ///< Tokens to generate (-1 = until EOS)
        int batch_size = 1; ///< Batch size
        int n_threads = -1; ///< Thread count (-1 = auto)
        int seed = -1;      ///< RNG seed (-1 = random)

        // =========================================================================
        // Sampling Configuration
        // =========================================================================

        float temperature = 0.8f;   ///< Sampling temperature
        int top_k = 40;             ///< Top-K sampling
        float top_p = 0.9f;         ///< Top-P (nucleus) sampling
        bool deterministic = false; ///< Force deterministic mode

        // =========================================================================
        // Chat Configuration
        // =========================================================================

        bool chat_mode = false;             ///< Interactive chat mode
        bool single_shot_chat = false;      ///< Single prompt with chat template
        std::string system_prompt;          ///< System message
        std::string chat_template_override; ///< Template override

        // =========================================================================
        // Benchmark Configuration
        // =========================================================================

        bool benchmark_mode = false; ///< Run benchmark

        // =========================================================================
        // Server Configuration
        // =========================================================================

        bool serve_mode = false;              ///< Run HTTP server (--serve)
        int serve_port = 8080;                ///< Server port (--port)
        std::string serve_host = "127.0.0.1"; ///< Server bind address (--host)

        // =========================================================================
        // Fused Attention Configuration
        // =========================================================================

        bool use_fused_attention = false; ///< Use fused attention+Wo kernel
        FusedAttentionBackend fused_attention_backend = FusedAttentionBackend::JIT;

        // =========================================================================
        // MPI Bootstrap Configuration
        // =========================================================================

        int mpi_procs = 0;                         ///< MPI process count (0 = auto)
        std::string hostfile;                      ///< MPI hostfile path
        bool mpi_dry_run = false;                  ///< Print MPI config and exit
        bool mpi_verbose = false;                  ///< Verbose MPI output
        bool mpi_no_bootstrap = false;             ///< Disable auto-bootstrap
        bool mpi_oversubscribe = false;            ///< Allow oversubscription
        MPIProfile mpi_profile = MPIProfile::AUTO; ///< MPI bootstrap profile (--mpi-profile auto|tuned)

        // =========================================================================
        // Verbosity and Debug
        // =========================================================================

        int verbose_level = 0;     ///< 0=INFO, 1=DEBUG, 2=TRACE
        bool list_devices = false; ///< List devices and exit
        bool show_help = false;    ///< Show help and exit

        // =========================================================================
        // Memory Constraints
        // =========================================================================

        std::optional<size_t> max_gpu_memory_mb; ///< Maximum GPU memory in MB
        std::optional<size_t> max_cpu_memory_mb; ///< Maximum CPU memory in MB

        // =========================================================================
        // MoE Configuration
        // =========================================================================

        bool moe_shared_experts_gpu = true; ///< Place shared experts on GPU
        bool moe_sparse_experts_cpu = true; ///< Place sparse experts on CPU

        // =========================================================================
        // Precision
        // =========================================================================

        std::string activation_precision = "fp32"; ///< "fp32", "bf16", "fp16", "q8_1"
        std::string kv_cache_precision = "auto";   ///< "auto" (q16_1 on CPU, fp16 on GPU), "fp32", "fp16", "q8_1", "q16_1"

        // =========================================================================
        // Weight Sharding
        // =========================================================================

        bool shard_weights = false;           ///< Enable weight sharding
        bool disable_weight_sharding = false; ///< Disable weight sharding (--no-shard-weights)

        // =========================================================================
        // Heterogeneous Mode
        // =========================================================================

        bool heterogeneous_mode = false;   ///< Enable heterogeneous mode
        float cpu_compute_fraction = 0.2f; ///< CPU compute fraction (0.0 to 1.0)
        bool disable_gpu_tp = false;       ///< Disable GPU tensor parallelism
        bool disable_cpu_tp = false;       ///< Disable CPU tensor parallelism
        int min_layers_per_domain = 2;     ///< Minimum layers per domain

        // =========================================================================
        // Methods
        // =========================================================================

        /**
         * @brief Check if configuration uses named domains
         *
         * Named domains are used when:
         * - domain_definitions is non-empty, OR
         * - pp_stage_definitions is non-empty
         *
         * @return true if named domains are used
         */
        bool usesNamedDomains() const;

        /**
         * @brief Validate the configuration
         * @return Empty vector if valid, otherwise list of error messages
         */
        std::vector<std::string> validate() const;

        /**
         * @brief String representation for logging
         */
        std::string toString() const;

        /**
         * @brief Create a default configuration
         */
        static OrchestrationConfig defaults();
    };

} // namespace llaminar2
