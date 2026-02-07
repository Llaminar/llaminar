/**
 * @file OrchestrationRunner.cpp
 * @brief Implementation of OrchestrationRunner
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "OrchestrationRunner.h"
#include "../../config/OrchestrationConfigParser.h"
#include "../../config/TPPPValidator.h"
#include "../mpi_orchestration/ExecutionPlanBuilder.h"
#include "../factory/InferenceRunnerFactory.h"
#include "../local_execution/orchestrators/MultiDeviceOrchestrator.h"
#include "../parallelism_tree/ParallelismTree.h"
#include "../../collective/LocalTPContext.h"
#include "../../collective/ILocalPPContext.h"
#include "../../loaders/ModelContext.h"
#include "../../loaders/ModelContextConfig.h"
#include "../../loaders/ModelLoader.h"
#include "../../backends/ComputeBackend.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

    // =========================================================================
    // Construction
    // =========================================================================

    OrchestrationRunner::OrchestrationRunner(
        OrchestrationConfig config,
        std::unique_ptr<IExecutionPlanBuilder> plan_builder)
        : config_(std::move(config)), plan_builder_(std::move(plan_builder)), sampler_(0)
    {
        if (!plan_builder_)
        {
            plan_builder_ = createExecutionPlanBuilder();
        }
    }

    OrchestrationRunner::OrchestrationRunner(
        OrchestrationConfig config,
        RankExecutionPlan plan)
        : config_(std::move(config)), plan_(std::move(plan)), plan_built_(true), sampler_(0)
    {
    }

    OrchestrationRunner::~OrchestrationRunner()
    {
        shutdown();
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool OrchestrationRunner::initialize()
    {
        if (initialized_)
        {
            return true;
        }

        try
        {
            // Step 1: Initialize MPI if needed
            if (!initializeMPI())
            {
                return false;
            }

            // Step 2: Build execution plan (if not pre-built)
            if (!buildExecutionPlan())
            {
                return false;
            }

            // Step 3: Setup LOCAL TP context
            if (!setupLocalTPContext())
            {
                return false;
            }

            // Step 3.5: Setup LOCAL PP context
            if (!setupLocalPPContext())
            {
                return false;
            }

            // Step 4: Load model weights
            if (!loadWeights())
            {
                return false;
            }

            // Step 5: Validate TP/PP configuration against model architecture
            if (!validateTPPPConfiguration())
            {
                return false;
            }

            // Step 6: Build compute graph
            if (!buildComputeGraph())
            {
                return false;
            }

            initialized_ = true;
            LOG_INFO("OrchestrationRunner initialized successfully");
            return true;
        }
        catch (const std::exception &e)
        {
            return setError(std::string("Initialization failed: ") + e.what());
        }
    }

    void OrchestrationRunner::shutdown()
    {
        if (!initialized_)
        {
            return;
        }

        // Release resources in reverse order
        runner_.reset();
        local_pp_ctx_.reset();
        local_tp_ctx_.reset();
        model_ctx_.reset();

        initialized_ = false;
        LOG_DEBUG("OrchestrationRunner shut down");
    }

    // =========================================================================
    // Inference
    // =========================================================================

    bool OrchestrationRunner::prefill(const std::vector<int32_t> &prompt_tokens)
    {
        if (!initialized_)
        {
            setError("Runner not initialized");
            return false;
        }

        if (prompt_tokens.empty())
        {
            setError("Empty prompt tokens");
            return false;
        }

        // For PP: head stage receives activations from external source
        // (handled by MPI in distributed setting)
        if (!isPipelineHead())
        {
            // Non-head stages wait for activations from previous stage
            receiveActivationsFromPrevStage();
        }

        // Run forward pass
        try
        {
            if (!runner_->forward(prompt_tokens.data(),
                                  static_cast<int>(prompt_tokens.size())))
            {
                return setError("Forward pass failed during prefill");
            }
        }
        catch (const std::exception &e)
        {
            return setError(std::string("Prefill failed: ") + e.what());
        }

        // For PP: non-tail stages send activations to next stage
        if (!isPipelineTail())
        {
            sendActivationsToNextStage();
        }

        // Store last token for next decode step
        last_token_ = prompt_tokens.back();

        return true;
    }

    GenerationResult OrchestrationRunner::decodeStep()
    {
        GenerationResult result;

        if (!initialized_)
        {
            result.error = "Runner not initialized";
            return result;
        }

        // For PP: non-head stages receive from previous
        if (!isPipelineHead())
        {
            receiveActivationsFromPrevStage();
        }

        // Run single-token forward with last token
        if (!runner_->forward(&last_token_, 1))
        {
            result.error = "Forward pass failed during decode";
            return result;
        }

        const float *logits = runner_->logits();
        if (!logits)
        {
            result.error = "No logits available";
            return result;
        }

        // For PP: send to next stage if not tail
        if (!isPipelineTail())
        {
            sendActivationsToNextStage();
            // Non-tail stages don't sample
            return result;
        }

        // Tail stage: sample token
        int vocab = vocabSize();
        std::vector<float> logits_vec(logits, logits + vocab);

        SamplingParams params;
        params.temperature = 0.0f; // Greedy by default
        int token = sampler_.sample(logits_vec, params);

        result.tokens.push_back(token);
        last_token_ = token; // Store for next decode step

        // Check stop tokens
        for (int32_t stop : stop_tokens_)
        {
            if (token == stop)
            {
                result.is_complete = true;
                break;
            }
        }

        return result;
    }

    GenerationResult OrchestrationRunner::generate(
        const std::vector<int32_t> &prompt_tokens,
        int max_new_tokens,
        const SamplingParams &sampling)
    {
        GenerationResult result;

        if (!initialized_)
        {
            result.error = "Runner not initialized";
            return result;
        }

        // Prefill
        if (!prefill(prompt_tokens))
        {
            result.error = last_error_;
            return result;
        }

        // Decode loop
        sampler_ = Sampler(sampling.seed);

        for (int i = 0; i < max_new_tokens; ++i)
        {
            // Use decodeStep() which uses last_token_ internally
            GenerationResult step = decodeStep();

            if (!step.error.empty())
            {
                result.error = step.error;
                return result;
            }

            // Collect tokens from step (on tail stage)
            for (int32_t token : step.tokens)
            {
                result.tokens.push_back(token);
            }

            if (step.is_complete)
            {
                result.is_complete = true;
                break;
            }
        }

        return result;
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    const RankExecutionPlan &OrchestrationRunner::executionPlan() const
    {
        return plan_;
    }

    const OrchestrationConfig &OrchestrationRunner::config() const
    {
        return config_;
    }

    // =========================================================================
    // Status
    // =========================================================================

    bool OrchestrationRunner::isInitialized() const
    {
        return initialized_;
    }

    const std::string &OrchestrationRunner::lastError() const
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        return last_error_;
    }

    int OrchestrationRunner::vocabSize() const
    {
        if (!runner_)
        {
            return 0;
        }
        return runner_->vocab_size();
    }

    int OrchestrationRunner::currentPosition() const
    {
        if (!runner_)
        {
            return 0;
        }
        return runner_->get_position();
    }

    void OrchestrationRunner::clearCache()
    {
        if (runner_)
        {
            runner_->clear_cache();
        }
    }

    // =========================================================================
    // Advanced
    // =========================================================================

    const float *OrchestrationRunner::lastLogits() const
    {
        if (!runner_)
        {
            return nullptr;
        }
        return runner_->logits();
    }

    void OrchestrationRunner::setStopTokens(const std::vector<int32_t> &stop_tokens)
    {
        stop_tokens_ = stop_tokens;
    }

    // =========================================================================
    // Initialization Helpers
    // =========================================================================

    bool OrchestrationRunner::initializeMPI()
    {
        // Check if multi-rank MPI is needed
        bool needs_multi_rank_mpi = config_.pp_degree > 1 ||
                                    config_.tp_scope == TPScope::GLOBAL ||
                                    config_.tp_scope == TPScope::HYBRID;

        if (!needs_multi_rank_mpi)
        {
            // For single-rank execution, create a local-only MPI context
            // This is needed for TensorFactory creation in ModelContext
            LOG_DEBUG("Creating single-rank MPI context for local execution");
            mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
            return true;
        }

        // Create or reuse MPI context using factory
        mpi_ctx_ = MPIContextFactory::global();
        if (!mpi_ctx_)
        {
            return setError("Failed to get MPI context");
        }

        LOG_INFO("MPI initialized: rank " << mpi_ctx_->rank()
                                          << " of " << mpi_ctx_->world_size());

        return true;
    }

    bool OrchestrationRunner::buildExecutionPlan()
    {
        if (plan_built_)
        {
            LOG_DEBUG("Using pre-built execution plan");
            return true;
        }

        if (!plan_builder_)
        {
            return setError("No plan builder available");
        }

        // Gather cluster inventory
        ClusterInventory inventory = gatherClusterInventory();

        // Get model config (need to load model metadata first)
        // Read actual model metadata from the GGUF file for accurate plan building
        ModelConfig model_config;
        if (!config_.model_path.empty())
        {
            ModelLoader metadata_loader;
            if (metadata_loader.loadModel(config_.model_path))
            {
                model_config.n_layers = static_cast<int>(metadata_loader.blockCount());
                model_config.n_heads = static_cast<int>(metadata_loader.headCount());
                model_config.n_kv_heads = static_cast<int>(metadata_loader.headCountKV());
                model_config.hidden_size = static_cast<int>(metadata_loader.embeddingLength());
                LOG_DEBUG("Model metadata for plan building: n_layers=" << model_config.n_layers
                                                                        << " n_heads=" << model_config.n_heads
                                                                        << " n_kv_heads=" << model_config.n_kv_heads
                                                                        << " hidden_size=" << model_config.hidden_size);
            }
            else
            {
                LOG_WARN("Failed to read model metadata from " << config_.model_path
                                                               << ", using defaults for plan building");
                model_config.n_layers = 24;
                model_config.n_heads = 32;
                model_config.n_kv_heads = 8;
                model_config.hidden_size = 4096;
            }
        }
        else
        {
            // No model path (testing) - use defaults
            model_config.n_layers = 24;
            model_config.n_heads = 32;
            model_config.n_kv_heads = 8;
            model_config.hidden_size = 4096;
        }

        // Validate config
        auto errors = plan_builder_->validateConfig(config_, model_config, inventory);
        if (!errors.empty())
        {
            std::string error_msg = "Config validation failed:";
            for (const auto &e : errors)
            {
                error_msg += "\n  - " + e;
            }
            return setError(error_msg);
        }

        // Build plan for this rank
        int my_rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
        plan_ = plan_builder_->buildPlanForRank(config_, model_config, inventory, my_rank);

        // Validate the built plan
        auto plan_errors = plan_.validate();
        if (!plan_errors.empty())
        {
            std::string error_msg = "Plan validation failed:";
            for (const auto &e : plan_errors)
            {
                error_msg += "\n  - " + e;
            }
            return setError(error_msg);
        }

        plan_built_ = true;
        return true;
    }

    ClusterInventory OrchestrationRunner::gatherClusterInventory()
    {
        ClusterInventory inventory;

        // Initialize DeviceManager to enumerate all available devices
        DeviceManager::instance().initialize(-1); // -1 = no NUMA filtering, see all devices
        const auto &devices = DeviceManager::instance().devices();

        // Helper to convert ComputeBackendType to DeviceType
        auto toDeviceType = [](ComputeBackendType backend) -> DeviceType
        {
            switch (backend)
            {
            case ComputeBackendType::GPU_CUDA:
                return DeviceType::CUDA;
            case ComputeBackendType::GPU_ROCM:
                return DeviceType::ROCm;
            case ComputeBackendType::GPU_VULKAN:
                return DeviceType::Vulkan;
            case ComputeBackendType::GPU_METAL:
                return DeviceType::Metal;
            case ComputeBackendType::CPU:
            default:
                return DeviceType::CPU;
            }
        };

        // For single-rank execution, create a simple inventory
        if (!mpi_ctx_ || mpi_ctx_->world_size() == 1)
        {
            RankInventory rank_inv;
            rank_inv.rank = 0;
            rank_inv.hostname = "localhost";
            rank_inv.numa_nodes = 1;
            rank_inv.node_id = 0;
            rank_inv.local_rank = 0;

            // Add CPU by default
            rank_inv.cpu.type = DeviceType::CPU;
            rank_inv.cpu.local_device_id = 0;
            rank_inv.cpu_cores = 1;

            // Enumerate actual GPUs from DeviceManager
            int gpu_idx = 0;
            for (const auto &dev : devices)
            {
                if (dev.type != ComputeBackendType::CPU)
                {
                    DeviceInfo gpu;
                    gpu.type = toDeviceType(dev.type);
                    gpu.local_device_id = gpu_idx++;
                    gpu.memory_bytes = dev.total_memory_bytes;
                    gpu.free_memory_bytes = dev.free_memory_bytes;
                    gpu.name = dev.name;
                    gpu.numa_node = dev.numa_node;
                    gpu.compute_capability_major = dev.compute_capability / 10;
                    gpu.compute_capability_minor = dev.compute_capability % 10;
                    rank_inv.gpus.push_back(gpu);

                    LOG_DEBUG("[gatherClusterInventory] Found GPU: " << dev.name
                                                                     << " (" << dev.total_memory_bytes / (1024 * 1024 * 1024) << " GB)");
                }
            }

            // If explicit tp_devices are configured, use those instead (override)
            if (!config_.tp_devices.empty())
            {
                LOG_DEBUG("[gatherClusterInventory] Using explicitly configured TP devices (count="
                          << config_.tp_devices.size() << ")");
                rank_inv.gpus.clear();
                for (size_t i = 0; i < config_.tp_devices.size(); ++i)
                {
                    const auto &addr = config_.tp_devices[i];
                    DeviceInfo gpu;
                    gpu.type = addr.device_type;
                    gpu.local_device_id = static_cast<int>(i);
                    gpu.memory_bytes = 0; // Unknown without actual enumeration
                    rank_inv.gpus.push_back(gpu);
                }
            }

            inventory.ranks.push_back(rank_inv);
            inventory.world_size = 1;
            inventory.node_count = 1;
            inventory.total_gpus = static_cast<int>(rank_inv.gpus.size());

            LOG_INFO("[gatherClusterInventory] Discovered " << inventory.total_gpus << " GPU(s)");
            return inventory;
        }

        // For multi-rank: would gather from all ranks via MPI_Allgather
        // For now, assume homogeneous setup based on this rank's devices
        int world_size = mpi_ctx_->world_size();
        int gpus_per_rank = 0;

        // Count GPUs from DeviceManager
        for (const auto &dev : devices)
        {
            if (dev.type != ComputeBackendType::CPU)
            {
                gpus_per_rank++;
            }
        }

        for (int r = 0; r < world_size; ++r)
        {
            RankInventory rank_inv;
            rank_inv.rank = r;
            rank_inv.hostname = "localhost"; // Would be gathered from each rank
            rank_inv.numa_nodes = 2;         // Assume 2 NUMA nodes per machine
            rank_inv.node_id = r / 2;        // Assume 2 ranks per node
            rank_inv.local_rank = r % 2;

            // Add CPU
            rank_inv.cpu.type = DeviceType::CPU;
            rank_inv.cpu.local_device_id = 0;
            rank_inv.cpu_cores = 1;

            // Add GPUs from DeviceManager enumeration
            int gpu_idx = 0;
            for (const auto &dev : devices)
            {
                if (dev.type != ComputeBackendType::CPU)
                {
                    DeviceInfo gpu;
                    gpu.type = toDeviceType(dev.type);
                    gpu.local_device_id = gpu_idx++;
                    gpu.memory_bytes = dev.total_memory_bytes;
                    gpu.free_memory_bytes = dev.free_memory_bytes;
                    gpu.name = dev.name;
                    gpu.numa_node = dev.numa_node;
                    gpu.compute_capability_major = dev.compute_capability / 10;
                    gpu.compute_capability_minor = dev.compute_capability % 10;
                    rank_inv.gpus.push_back(gpu);
                }
            }

            // Override with explicit tp_devices if configured
            if (!config_.tp_devices.empty())
            {
                rank_inv.gpus.clear();
                for (size_t i = 0; i < config_.tp_devices.size(); ++i)
                {
                    const auto &addr = config_.tp_devices[i];
                    DeviceInfo gpu;
                    gpu.type = addr.device_type;
                    gpu.local_device_id = static_cast<int>(i);
                    gpu.memory_bytes = 0;
                    rank_inv.gpus.push_back(gpu);
                }
                gpus_per_rank = static_cast<int>(config_.tp_devices.size());
            }

            inventory.ranks.push_back(rank_inv);
        }

        // Set cluster-level fields
        inventory.world_size = world_size;
        inventory.node_count = (world_size + 1) / 2; // Assume 2 ranks per node
        inventory.total_gpus = world_size * gpus_per_rank;

        LOG_INFO("[gatherClusterInventory] Discovered " << inventory.total_gpus
                                                        << " GPU(s) across " << world_size << " ranks");
        return inventory;
    }

    bool OrchestrationRunner::setupLocalTPContext()
    {
        // Check if LOCAL TP is configured
        if (plan_.local_tp_devices.empty())
        {
            LOG_DEBUG("No LOCAL TP devices configured");
            return true;
        }

        // Create LOCAL TP context using factory function
        local_tp_ctx_ = createLocalTPContext(
            plan_.local_tp_devices,
            plan_.local_tp_weights,
            plan_.local_tp_backend);
        if (!local_tp_ctx_)
        {
            return setError("Failed to create LOCAL TP context");
        }

        LOG_INFO("LOCAL TP context created with " << plan_.local_tp_devices.size() << " devices");
        return true;
    }

    bool OrchestrationRunner::setupLocalPPContext()
    {
        // Check if LOCAL PP is configured
        if (plan_.local_pp_devices.size() <= 1)
        {
            LOG_DEBUG("No LOCAL PP devices configured (or only single device)");
            return true;
        }

        // Build LocalPPConfig from execution plan
        LocalPPConfig pp_config;
        pp_config.stage_devices = plan_.local_pp_devices;
        pp_config.layer_boundaries = plan_.local_pp_layer_boundaries;

        // Validate configuration
        if (!pp_config.isValid())
        {
            return setError("Invalid LOCAL PP configuration");
        }

        // Create LOCAL PP context using factory function
        local_pp_ctx_ = createLocalPPContext(pp_config);
        if (!local_pp_ctx_)
        {
            return setError("Failed to create LOCAL PP context");
        }

        LOG_INFO("LOCAL PP context created with " << pp_config.numStages()
                                                  << " stages on " << plan_.local_pp_devices.size() << " devices");
        return true;
    }

    bool OrchestrationRunner::loadWeights()
    {
        // Get model path from config
        std::string model_path = config_.model_path;

        // Skip weight loading if no model path (for testing)
        if (model_path.empty())
        {
            LOG_DEBUG("No model path specified, skipping weight loading");
            return true;
        }

        // Create ModelContextConfig from the execution plan
        // This automatically configures:
        // - Layer range (first_layer, last_layer) for PP
        // - Global weight flags (has_embedding, has_lm_head) for PP
        // - Shard info (shard_index, total_shards, work_fraction) for TP
        // - Appropriate strategy (REPLICATED, SHARDED, or layer-partitioned)
        ModelContextConfig weight_config = ModelContextConfig::fromExecutionPlan(plan_);
        weight_config.mpi_ctx = mpi_ctx_;
        weight_config.weight_precision = WeightPrecision::NATIVE;

        // Validate config
        auto errors = weight_config.validate();
        if (!errors.empty())
        {
            std::ostringstream oss;
            oss << "Invalid ModelContextConfig from execution plan:\n";
            for (const auto &err : errors)
            {
                oss << "  - " << err << "\n";
            }
            return setError(oss.str());
        }

        LOG_DEBUG("Weight loading config: " << weight_config.toString());

        // Create ModelContext using the unified config-based factory method
        // Use NATIVE weight precision to preserve quantization (Q4_0, Q8_0, etc.)
        // for efficient GPU kernels rather than dequantizing to FP32
        model_ctx_ = ModelContext::create(model_path, weight_config);

        if (!model_ctx_)
        {
            return setError("Failed to create ModelContext for: " + model_path);
        }

        // Create tokenizer from model context
        tokenizer_ = createTokenizer(model_ctx_);
        if (!tokenizer_)
        {
            LOG_WARN("Failed to create tokenizer from model context");
        }

        LOG_INFO("Model context created from: " << model_path
                                                << " (layers " << weight_config.first_layer << "-" << weight_config.last_layer
                                                << ", embedding=" << weight_config.has_embedding
                                                << ", lm_head=" << weight_config.has_lm_head << ")");
        return true;
    }

    bool OrchestrationRunner::validateTPPPConfiguration()
    {
        // Skip validation if no model loaded (testing mode)
        if (!model_ctx_)
        {
            LOG_DEBUG("No model context, skipping TP/PP validation");
            return true;
        }

        // Run validation
        auto result = TPPPValidator::validate(config_, *model_ctx_);

        // Log warnings (but don't fail)
        for (const auto &warning : result.warnings)
        {
            LOG_WARN("[TP/PP Config] " << warning);
        }

        // Check for errors
        if (!result.valid)
        {
            std::ostringstream oss;
            oss << "TP/PP configuration is incompatible with model architecture:\n";
            for (const auto &error : result.errors)
            {
                oss << "  - " << error << "\n";
            }
            return setError(oss.str());
        }

        LOG_INFO("TP/PP configuration validated against model architecture");
        return true;
    }

    bool OrchestrationRunner::buildComputeGraph()
    {
        // Check if LOCAL TP is configured (multiple devices within this rank)
        if (hasLocalTP())
        {
            return buildMultiDeviceComputeGraph();
        }

        // Single-device path
        return buildSingleDeviceComputeGraph();
    }

    bool OrchestrationRunner::hasLocalTP() const
    {
        return plan_.local_tp_devices.size() > 1;
    }

    MultiDeviceOrchestrator::Config OrchestrationRunner::buildMultiDeviceConfig() const
    {
        MultiDeviceOrchestrator::Config config;

        // Copy devices from execution plan
        config.devices = plan_.local_tp_devices;

        // Copy weights (or default to equal)
        if (!plan_.local_tp_weights.empty())
        {
            config.weights = plan_.local_tp_weights;
        }

        // Copy backend selection
        config.backend = plan_.local_tp_backend;

        // Copy runtime settings from orchestration config
        config.max_seq_len = config_.max_seq_len;
        config.batch_size = 1;
        config.activation_precision = ActivationPrecision::FP32;

        return config;
    }

    std::shared_ptr<ITokenizer> OrchestrationRunner::tokenizer() const
    {
        return tokenizer_;
    }

    bool OrchestrationRunner::buildMultiDeviceComputeGraph()
    {
        // Validate that all requested devices actually exist in hardware
        const auto &dm = DeviceManager::instance();
        for (size_t i = 0; i < plan_.local_tp_devices.size(); ++i)
        {
            auto local_device = plan_.local_tp_devices[i].toLocalDeviceId();
            if (!dm.deviceExists(local_device))
            {
                return setError("TP device " + std::to_string(i) + " (" +
                                local_device.toString() +
                                ") is not available. Available devices: " +
                                dm.availableDevicesString());
            }
        }

        LOG_INFO("[OrchestrationRunner] Execution strategy: MULTI-DEVICE (LOCAL TP)");
        LOG_INFO("[OrchestrationRunner]   TP degree: " << plan_.local_tp_devices.size());

        // Log each device
        for (size_t i = 0; i < plan_.local_tp_devices.size(); ++i)
        {
            const auto &dev = plan_.local_tp_devices[i];
            std::string weight_str = "";
            if (i < plan_.local_tp_weights.size())
            {
                weight_str = " (weight=" + std::to_string(plan_.local_tp_weights[i]) + ")";
            }
            LOG_INFO("[OrchestrationRunner]   Device " << i << ": " << dev.toString() << weight_str);
        }

        // Build multi-device config from execution plan
        auto mdo_config = buildMultiDeviceConfig();

        // Validate config
        if (!mdo_config.validate())
        {
            return setError("Invalid multi-device configuration");
        }

        // Create MultiDeviceOrchestrator via factory
        // Note: local_tp_ctx_ was already created in setupLocalTPContext()
        auto multi_orchestrator = createMultiDeviceOrchestrator(
            model_ctx_,
            std::move(local_tp_ctx_),
            mdo_config);

        if (!multi_orchestrator)
        {
            return setError("Failed to create MultiDeviceOrchestrator");
        }

        // Store as IInferenceRunner (MultiDeviceOrchestrator extends it)
        runner_ = std::move(multi_orchestrator);

        LOG_INFO("Multi-device compute graph built successfully");
        return true;
    }

    bool OrchestrationRunner::buildSingleDeviceComputeGraph()
    {
        // Determine target device from execution plan
        DeviceId device = DeviceId::cpu();
        std::string device_source = "default (CPU)";

        if (!plan_.local_tp_devices.empty())
        {
            device = plan_.primary_device.toLocalDeviceId();
            device_source = "plan.local_tp_devices[0]";
        }
        else if (!plan_.primary_device.hostname.empty())
        {
            device = plan_.primary_device.toLocalDeviceId();
            device_source = "plan.primary_device";
        }

        // Validate that the requested device actually exists in hardware
        const auto &dm = DeviceManager::instance();
        if (!dm.deviceExists(device))
        {
            return setError("Requested device " + device.toString() +
                            " is not available. Available devices: " +
                            dm.availableDevicesString());
        }

        // Log execution strategy decision
        LOG_INFO("[OrchestrationRunner] Execution strategy: SINGLE-DEVICE");
        LOG_INFO("[OrchestrationRunner]   Target device: " << device.toString());
        LOG_INFO("[OrchestrationRunner]   Device source: " << device_source);
        if (device.is_cpu())
        {
            LOG_INFO("[OrchestrationRunner]   Backend: CPU (OpenBLAS/AVX-512)");
        }
        else if (device.is_cuda())
        {
            LOG_INFO("[OrchestrationRunner]   Backend: CUDA (GPU " << device.ordinal << ")");
        }
        else if (device.is_rocm())
        {
            LOG_INFO("[OrchestrationRunner]   Backend: ROCm (GPU " << device.ordinal << ")");
        }

        // Configure inference runner
        InferenceRunnerConfig runner_config;
        runner_config.max_seq_len = config_.max_seq_len;
        runner_config.batch_size = 1;

        // Create runner via factory (returns IInferenceRunner)
        if (model_ctx_)
        {
            runner_ = createInferenceRunner(
                model_ctx_,
                mpi_ctx_,
                device,
                runner_config);
        }

        if (!runner_ && model_ctx_)
        {
            return setError("Failed to create inference runner");
        }

        LOG_INFO("[OrchestrationRunner] Compute graph built successfully");
        return true;
    }

    // =========================================================================
    // PP Communication Helpers
    // =========================================================================

    void OrchestrationRunner::sendActivationsToNextStage()
    {
        if (!plan_.next_rank.has_value())
        {
            return;
        }

        // TODO: Implement MPI_Send for activations
        // This requires serializing the hidden states and sending to next_rank
        LOG_DEBUG("PP: Would send activations to rank " << *plan_.next_rank);
    }

    void OrchestrationRunner::receiveActivationsFromPrevStage()
    {
        if (!plan_.prev_rank.has_value())
        {
            return;
        }

        // TODO: Implement MPI_Recv for activations
        // This requires receiving hidden states from prev_rank
        LOG_DEBUG("PP: Would receive activations from rank " << *plan_.prev_rank);
    }

    bool OrchestrationRunner::isPipelineHead() const
    {
        return !plan_.prev_rank.has_value();
    }

    bool OrchestrationRunner::isPipelineTail() const
    {
        return !plan_.next_rank.has_value();
    }

    // =========================================================================
    // Error Handling
    // =========================================================================

    bool OrchestrationRunner::setError(const std::string &error)
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        last_error_ = error;
        LOG_ERROR(error);
        return false;
    }

    // =========================================================================
    // Snapshot API
    // =========================================================================

    void OrchestrationRunner::enableSnapshotCapture(const std::string &output_dir)
    {
        if (runner_)
        {
            runner_->enableSnapshotCapture(output_dir);
        }
    }

    void OrchestrationRunner::disableSnapshotCapture()
    {
        if (runner_)
        {
            runner_->disableSnapshotCapture();
        }
    }

    void OrchestrationRunner::clearSnapshots()
    {
        if (runner_)
        {
            runner_->clearSnapshots();
        }
    }

    const float *OrchestrationRunner::getSnapshot(const std::string &key, size_t &out_size) const
    {
        if (runner_)
        {
            return runner_->getSnapshot(key, out_size);
        }
        out_size = 0;
        return nullptr;
    }

    std::vector<std::string> OrchestrationRunner::getSnapshotKeys() const
    {
        if (runner_)
        {
            return runner_->getSnapshotKeys();
        }
        return {};
    }

} // namespace llaminar2
