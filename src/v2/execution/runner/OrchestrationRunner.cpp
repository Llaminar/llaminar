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
#include "../parallelism_tree/TreeToRunnerCompiler.h"
#include "../../collective/LocalTPContext.h"
#include "../../collective/ILocalPPContext.h"
#include "../../collective/BackendRouter.h"
#include "../../loaders/ModelContext.h"
#include "../../loaders/ModelContextConfig.h"
#include "../../loaders/ModelLoader.h"
#include "../local_execution/graph/SchemaFactoryRegistry.h"
#include "../../backends/ComputeBackend.h"
#include "../../tensors/TensorFactory.h"
#include "../../utils/Logger.h"
#include "../../utils/MPITopology.h"
#include "../../utils/NodeDetection.h"
#include "../../utils/NUMATopology.h"
#include "../../utils/WeightLoadingProfiler.h"

#include <algorithm>
#include <cctype>

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

    OrchestrationRunner::OrchestrationRunner(
        OrchestrationConfig config,
        RankExecutionPlan plan,
        std::unique_ptr<IInferenceRunner> runner)
        : config_(std::move(config)), plan_(std::move(plan)), plan_built_(true),
          runner_(std::move(runner)), initialized_(true), sampler_(0)
    {
    }

    OrchestrationRunner::OrchestrationRunner(
        OrchestrationConfig config,
        RankExecutionPlan plan,
        std::unique_ptr<IInferenceRunner> runner,
        std::shared_ptr<IMPIContext> mpi_ctx)
        : config_(std::move(config)), plan_(std::move(plan)), plan_built_(true),
          mpi_ctx_(std::move(mpi_ctx)), runner_(std::move(runner)),
          initialized_(true), sampler_(0)
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

        auto syncInitStep = [&](bool local_ok, const char *step_name) -> bool
        {
            if (!mpi_ctx_ || mpi_ctx_->world_size() <= 1)
            {
                return local_ok;
            }

            int ok = local_ok ? 1 : 0;
            int global_ok = 0;
            MPI_Allreduce(&ok, &global_ok, 1, MPI_INT, MPI_MIN, mpi_ctx_->communicator());
            if (global_ok == 0)
            {
                if (local_ok)
                {
                    setError(std::string("Initialization failed on another rank at step: ") + step_name);
                }
                return false;
            }
            return true;
        };

        try
        {
            // Step 1: Initialize MPI if needed
            if (!initializeMPI())
            {
                return false;
            }
            if (!syncInitStep(true, "initializeMPI"))
            {
                return false;
            }

            // Step 2: Build execution plan (if not pre-built)
            if (!buildExecutionPlan())
            {
                syncInitStep(false, "buildExecutionPlan");
                return false;
            }
            if (!syncInitStep(true, "buildExecutionPlan"))
            {
                return false;
            }

            // Step 3: Setup LOCAL TP context
            if (!setupLocalTPContext())
            {
                syncInitStep(false, "setupLocalTPContext");
                return false;
            }
            if (!syncInitStep(true, "setupLocalTPContext"))
            {
                return false;
            }

            // Step 3.5: Setup LOCAL PP context
            if (!setupLocalPPContext())
            {
                syncInitStep(false, "setupLocalPPContext");
                return false;
            }
            if (!syncInitStep(true, "setupLocalPPContext"))
            {
                return false;
            }

            // Step 4: Load model weights
            if (!loadWeights())
            {
                syncInitStep(false, "loadWeights");
                return false;
            }
            if (!syncInitStep(true, "loadWeights"))
            {
                return false;
            }

            // Step 5: Validate TP/PP configuration against model architecture
            if (!validateTPPPConfiguration())
            {
                syncInitStep(false, "validateTPPPConfiguration");
                return false;
            }
            if (!syncInitStep(true, "validateTPPPConfiguration"))
            {
                return false;
            }

            // Step 5b: Validate context length against model max
            if (!validateContextLength())
            {
                syncInitStep(false, "validateContextLength");
                return false;
            }
            if (!syncInitStep(true, "validateContextLength"))
            {
                return false;
            }

            // Step 6: Build compute graph
            if (!buildComputeGraph())
            {
                syncInitStep(false, "buildComputeGraph");
                return false;
            }
            if (!syncInitStep(true, "buildComputeGraph"))
            {
                return false;
            }

            initialized_ = true;

            // Cache model-recommended sampling params (for API consumers)
            if (model_ctx_)
            {
                const std::string arch = model_ctx_->architecture();
                if (SchemaFactoryRegistry::isSupported(arch))
                {
                    auto factory = SchemaFactoryRegistry::getFactory(arch);
                    if (factory)
                    {
                        recommended_sampling_params_ = factory->getRecommendedSamplingParams();
                        if (recommended_sampling_params_.has_penalties() || recommended_sampling_params_.temperature != 1.0f)
                        {
                            LOG_INFO("[OrchestrationRunner] Model-recommended sampling: "
                                     << "temp=" << recommended_sampling_params_.temperature
                                     << " top_p=" << recommended_sampling_params_.top_p
                                     << " top_k=" << recommended_sampling_params_.top_k
                                     << " presence_penalty=" << recommended_sampling_params_.presence_penalty
                                     << " frequency_penalty=" << recommended_sampling_params_.frequency_penalty);
                        }
                    }
                }
            }

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

        // Broadcast to worker ranks so they prefill with the same tokens
        if (mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->rank() == 0 && mpi_ctx_->world_size() > 1)
        {
            broadcastCommand(MPICommand::PREFILL);
            int32_t n_tokens = static_cast<int32_t>(prompt_tokens.size());
            mpi_ctx_->broadcast_int32(&n_tokens, 1, 0);
            // const_cast is safe: rank 0 is the sender, buffer is not modified
            mpi_ctx_->broadcast_int32(const_cast<int32_t *>(prompt_tokens.data()),
                                      static_cast<size_t>(n_tokens), 0);
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

        // Signal that prefill logits are ready for sampling.
        // The first decodeStep() will sample from these logits directly
        // instead of re-feeding the last prompt token (which would cause
        // the model to see it twice at consecutive positions, corrupting
        // GDN recurrence state and KV cache entries).
        prefill_logits_ready_ = true;

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

        // Broadcast to worker ranks so they run decode in lockstep
        if (mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->rank() == 0 && mpi_ctx_->world_size() > 1)
            broadcastCommand(MPICommand::DECODE_STEP);

        if (prefill_logits_ready_)
        {
            // First decode step after prefill: sample from the already-computed
            // prefill logits instead of re-feeding the last prompt token.
            // This avoids processing the last token twice (which corrupts GDN
            // recurrence state and creates duplicate KV cache entries).
            prefill_logits_ready_ = false;
        }
        else
        {
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

            // For PP: send to next stage if not tail
            if (!isPipelineTail())
            {
                sendActivationsToNextStage();
                // Non-tail stages don't sample
                return result;
            }
        }

        // Tail stage: try GPU-side sampling first, fall back to CPU
        // Note: GPU sampling cannot apply presence/frequency penalties
        // (penalty history is only on CPU), so force CPU path when penalties are active.
        int token = -1;

        if (active_sampling_params_.has_penalties())
        {
            // Penalties require CPU-side sampling with token history
            token = -1; // Force CPU fallback below
        }
        else if (active_sampling_params_.is_greedy())
        {
            // Try GPU-side greedy (argmax)
            token = runner_->sampleGreedyOnDevice();
        }
        else
        {
            // Try GPU-side top-k/top-p
            token = runner_->sampleOnDevice(active_sampling_params_);
            if (token >= 0)
            {
                LOG_TRACE("[decodeStep] GPU top-k/top-p sampled token=" << token);
            }
        }

        if (token < 0)
        {
            // Fallback: CPU-side sampling (requires logits D2H)
            LOG_TRACE("[decodeStep] GPU sampling returned -1, falling back to CPU");
            const float *logits = runner_->logits();
            if (!logits)
            {
                result.error = "No logits available";
                return result;
            }
            int vocab = vocabSize();
            token = sampler_.sample(logits, static_cast<size_t>(vocab), active_sampling_params_);
        }

        // Record token for presence/frequency penalty tracking
        sampler_.record_token(token);

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
        // After prefill, the first decodeStep() samples from the prefill logits
        // directly (via prefill_logits_ready_ flag) instead of re-feeding the
        // last prompt token. GPU-side sampling (sampleGreedyOnDevice) works on
        // device logits without D2H, so no gather is needed for GPU models.
        // For CPU models, logits are already on host.
        if (!prefill(prompt_tokens))
        {
            result.error = last_error_;
            return result;
        }

        // Store sampling params for decodeStep() and configure GPU-side decode
        active_sampling_params_ = sampling;
        sampler_ = Sampler(sampling.seed);

        // Enable GPU-side logits skip for decode (GPU sampling avoids full D2H)
        runner_->setSkipLogitsGatherDecode(true);

        for (int i = 0; i < max_new_tokens; ++i)
        {
            // Use decodeStep() which uses last_token_ internally
            GenerationResult step = decodeStep();

            if (!step.error.empty())
            {
                result.error = step.error;
                break;
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

        // Restore normal logits gathering after generation
        runner_->setSkipLogitsGatherDecode(false);

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
        // Broadcast to worker ranks so they clear their KV caches in lockstep
        if (mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->rank() == 0 && mpi_ctx_->world_size() > 1)
            broadcastCommand(MPICommand::CLEAR_CACHE);

        if (runner_)
        {
            runner_->clear_cache();
        }
        prefill_logits_ready_ = false;
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
            std::shared_ptr<IMPIContext> metadata_mpi_ctx = mpi_ctx_;
            if (!metadata_mpi_ctx)
            {
                metadata_mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
            }
            TensorFactory metadata_factory(*metadata_mpi_ctx);
            ModelLoader metadata_loader(&metadata_factory);
            metadata_loader.setUseMmap(false); // Only reading header metadata, skip mmap
            bool metadata_ok = false;
            try
            {
                metadata_ok = metadata_loader.loadModel(config_.model_path);
            }
            catch (const std::exception &e)
            {
                LOG_WARN("Failed to read model metadata from " << config_.model_path
                                                               << " (" << e.what() << "), using defaults for plan building");
                metadata_ok = false;
            }
            if (metadata_ok)
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

        // Ensure DeviceManager is initialized with NUMA-aware filtering by default.
        // This avoids accidentally broadening visibility to cross-socket devices in local execution.
        auto &dm = DeviceManager::instance();
        if (dm.devices().empty())
        {
            auto numa_info = NUMATopology::detectLocalNUMANode();
            int target_numa_node = 0;
            if (numa_info.detection_succeeded && numa_info.local_numa_node >= 0)
            {
                target_numa_node = numa_info.local_numa_node;
            }
            dm.initialize(target_numa_node, false); // Tables already printed pre-MPI
        }
        const auto &devices = dm.devices();

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
            for (const auto &dev : devices)
            {
                if (dev.type != ComputeBackendType::CPU)
                {
                    DeviceInfo gpu;
                    gpu.type = toDeviceType(dev.type);
                    gpu.local_device_id = dev.device_id;
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

        // Multi-rank execution: build local RankInventory and exchange via MPI_Allgatherv.
        const int world_size = mpi_ctx_->world_size();
        const int rank = mpi_ctx_->rank();

        MPI_Comm comm = mpi_ctx_->communicator();

        RankInventory local_rank_inv;
        local_rank_inv.rank = rank;
        local_rank_inv.node_id = -1;
        local_rank_inv.local_rank = 0;
        local_rank_inv.numa_nodes = 1;

        // Hostname
        char hostname_buf[MPI_MAX_PROCESSOR_NAME] = {0};
        int hostname_len = 0;
        if (MPI_Get_processor_name(hostname_buf, &hostname_len) == MPI_SUCCESS && hostname_len > 0)
        {
            local_rank_inv.hostname.assign(hostname_buf, static_cast<size_t>(hostname_len));
        }
        else
        {
            local_rank_inv.hostname = "unknown";
        }

        // Detect local rank within physical node (shared-memory communicator)
        MPI_Comm local_comm = MPI_COMM_NULL;
        if (MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm) == MPI_SUCCESS)
        {
            int local_rank = 0;
            int local_world = 1;
            MPI_Comm_rank(local_comm, &local_rank);
            MPI_Comm_size(local_comm, &local_world);
            local_rank_inv.local_rank = local_rank;
            local_rank_inv.numa_nodes = local_world;
            MPI_Comm_free(&local_comm);
        }

        // Populate CPU info
        local_rank_inv.cpu.type = DeviceType::CPU;
        local_rank_inv.cpu.local_device_id = 0;
        local_rank_inv.cpu_cores = 1;

        // Populate GPU info from DeviceManager
        for (const auto &dev : devices)
        {
            if (dev.type == ComputeBackendType::CPU)
            {
                continue;
            }

            DeviceInfo gpu;
            gpu.type = toDeviceType(dev.type);
            gpu.local_device_id = dev.device_id;
            gpu.memory_bytes = dev.total_memory_bytes;
            gpu.free_memory_bytes = dev.free_memory_bytes;
            gpu.name = dev.name;
            gpu.numa_node = dev.numa_node;
            gpu.compute_capability_major = dev.compute_capability / 10;
            gpu.compute_capability_minor = dev.compute_capability % 10;
            local_rank_inv.gpus.push_back(gpu);
        }

        // Override with explicit tp_devices if configured
        if (!config_.tp_devices.empty())
        {
            local_rank_inv.gpus.clear();
            for (size_t i = 0; i < config_.tp_devices.size(); ++i)
            {
                const auto &addr = config_.tp_devices[i];
                DeviceInfo gpu;
                gpu.type = addr.device_type;
                gpu.local_device_id = static_cast<int>(i);
                gpu.memory_bytes = 0;
                local_rank_inv.gpus.push_back(gpu);
            }
        }

        // Serialize local inventory
        std::vector<uint8_t> local_data = MPITopology::serializeRankInventory(local_rank_inv);
        const int local_size = static_cast<int>(local_data.size());

        // Gather serialized sizes from all ranks
        std::vector<int> all_sizes(world_size, 0);
        MPI_Allgather(
            &local_size, 1, MPI_INT,
            all_sizes.data(), 1, MPI_INT,
            comm);

        // Compute displacements for allgatherv
        std::vector<int> displacements(world_size, 0);
        int total_size = 0;
        for (int r = 0; r < world_size; ++r)
        {
            displacements[r] = total_size;
            total_size += all_sizes[r];
        }

        std::vector<uint8_t> all_data(static_cast<size_t>(total_size));
        MPI_Allgatherv(
            local_data.data(), local_size, MPI_BYTE,
            all_data.data(), all_sizes.data(), displacements.data(), MPI_BYTE,
            comm);

        // Deserialize inventories from all ranks
        inventory.world_size = world_size;
        inventory.ranks.resize(world_size);
        for (int r = 0; r < world_size; ++r)
        {
            const uint8_t *ptr = all_data.data() + displacements[r];
            const size_t size = static_cast<size_t>(all_sizes[r]);
            try
            {
                inventory.ranks[r] = MPITopology::deserializeRankInventory(ptr, size);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[gatherClusterInventory] Failed to deserialize rank " << r << ": " << e.what());
                inventory.ranks[r].rank = r;
                inventory.ranks[r].hostname = "error";
                inventory.ranks[r].node_id = -1;
            }
        }

        // Build deterministic node_id mapping from hostname (single source of truth).
        // If a hostfile is configured, use it to determine node ordering.
        std::vector<std::string> hostnames;
        hostnames.reserve(static_cast<size_t>(world_size));
        for (const auto &rank_inv : inventory.ranks)
        {
            hostnames.push_back(rank_inv.hostname);
        }

        NodeDetectionResult detection;
        if (!config_.hostfile.empty())
        {
            detection = NodeDetection::fromHostnames(hostnames, config_.hostfile);
        }
        else
        {
            detection = NodeDetection::fromHostnames(hostnames);
        }
        for (int r = 0; r < world_size; ++r)
        {
            inventory.ranks[r].node_id = detection.node_ids[r];
        }

        inventory.node_count = detection.node_count;
        inventory.buildNodeAggregations();

        LOG_INFO("[gatherClusterInventory] Discovered " << inventory.total_gpus
                                                        << " GPU(s) across " << world_size
                                                        << " ranks on " << inventory.node_count << " node(s)");
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

        // When LOCAL PP with TP domains is active, TP is per-stage (each PP stage
        // creates its own TP context inside the nested MDO). Skip global TP context.
        if (plan_.usesLocalPP())
        {
            LOG_DEBUG("LOCAL PP active — TP will be handled per-stage by nested MDOs");
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
        weight_config.use_mmap = config_.use_mmap;

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
        {
            ScopedWeightLoadTimer timer(WeightLoadPhase::GGUF_PARSE);
            model_ctx_ = ModelContext::create(model_path, weight_config);
        }

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

    bool OrchestrationRunner::validateContextLength()
    {
        if (!model_ctx_)
            return true;

        const int model_max = model_ctx_->contextLength();
        if (model_max <= 0)
        {
            LOG_DEBUG("Model does not report max context length, skipping validation");
            return true;
        }

        if (config_.max_seq_len > model_max)
        {
            LOG_ERROR("Requested context length " << config_.max_seq_len
                                                  << " exceeds model maximum of " << model_max
                                                  << ". Use -c " << model_max << " or smaller.");
            return setError("Context length " + std::to_string(config_.max_seq_len) +
                            " exceeds model maximum of " + std::to_string(model_max));
        }

        LOG_INFO("Context length: " << config_.max_seq_len
                                    << " / " << model_max << " (model max)");
        return true;
    }

    bool OrchestrationRunner::buildComputeGraph()
    {
        ScopedWeightLoadTimer timer(WeightLoadPhase::GRAPH_BUILD);

        // Check if LOCAL PP is configured (takes priority over TP, because
        // TP-in-PP composition creates per-stage TP contexts inside the MDO)
        if (plan_.usesLocalPP())
        {
            return buildLocalPPComputeGraph();
        }

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

    std::shared_ptr<ITokenizer> OrchestrationRunner::tokenizer() const
    {
        return tokenizer_;
    }

    const std::string &OrchestrationRunner::architecture() const
    {
        static const std::string kEmpty;
        return model_ctx_ ? model_ctx_->architecture() : kEmpty;
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

        // Build config from execution plan via canonical factory
        auto mdo_config = MultiDeviceOrchestrator::Config::fromPlan(plan_);

        LOG_INFO("[OrchestrationRunner] Multi-device precision config: activation="
                 << activationPrecisionToString(mdo_config.activation_precision)
                 << ", kv_cache=" << kvCachePrecisionToString(mdo_config.kv_cache_precision));

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

    bool OrchestrationRunner::buildLocalPPComputeGraph()
    {
        const auto &pp_devices = plan_.local_pp_devices;
        const auto &boundaries = plan_.local_pp_layer_boundaries;

        if (pp_devices.size() < 2 || boundaries.size() < pp_devices.size() + 1)
        {
            return setError("Invalid LOCAL PP plan: need >=2 devices and matching layer boundaries");
        }

        LOG_INFO("[OrchestrationRunner] Execution strategy: LOCAL PIPELINE PARALLEL");
        LOG_INFO("[OrchestrationRunner]   PP stages: " << pp_devices.size());

        // Validate all devices exist
        const auto &dm = DeviceManager::instance();
        for (size_t i = 0; i < pp_devices.size(); ++i)
        {
            auto local_device = pp_devices[i].toLocalDeviceId();
            if (!dm.deviceExists(local_device))
            {
                return setError("PP device " + std::to_string(i) + " (" +
                                local_device.toString() +
                                ") is not available. Available devices: " +
                                dm.availableDevicesString());
            }
        }

        // Build config from execution plan via canonical factory
        auto mdo_config = MultiDeviceOrchestrator::Config::fromPlan(plan_);

        // Log PP stage details
        for (size_t i = 0; i < mdo_config.pp_stages.size(); ++i)
        {
            const auto &stage = mdo_config.pp_stages[i];
            LOG_INFO("[OrchestrationRunner]   Stage " << i << ": "
                                                      << pp_devices[i].toString()
                                                      << " layers [" << stage.first_layer << ", "
                                                      << stage.last_layer << ") "
                                                      << (stage.has_embedding ? "[+embed] " : "")
                                                      << (stage.has_lm_head ? "[+lm_head] " : ""));
        }

        if (!mdo_config.validate())
        {
            return setError("Invalid LOCAL PP configuration");
        }

        // Ensure GlobalBackendRouter is initialized for inter-stage transfers.
        // LOCAL PP uses TensorBase::transferTo() which routes through the backend router.
        GlobalBackendRouter::initForTests();

        std::unique_ptr<MultiDeviceOrchestrator> orch;
        orch = std::make_unique<MultiDeviceOrchestrator>(model_ctx_, mdo_config);
        runner_ = std::move(orch);

        LOG_INFO("Local PP compute graph built successfully");
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
        const bool strict_numa = plan_.primary_device_numa_explicit;

        const bool device_available = strict_numa
                                          ? dm.deviceExists(plan_.primary_device, true)
                                          : dm.deviceExists(device);

        if (!device_available)
        {
            if (strict_numa)
            {
                return setError("Requested device " + plan_.primary_device.toString() +
                                " is not available on the specified NUMA node. Available devices: " +
                                dm.availableDevicesString());
            }

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
            LOG_INFO("[OrchestrationRunner]   Backend: CPU (OneDNN/AVX-512)");
        }
        else if (device.is_cuda())
        {
            LOG_INFO("[OrchestrationRunner]   Backend: CUDA (GPU " << device.ordinal << ")");
        }
        else if (device.is_rocm())
        {
            LOG_INFO("[OrchestrationRunner]   Backend: ROCm (GPU " << device.ordinal << ")");
        }

        // Build config from execution plan via canonical factory
        auto runner_config = InferenceRunnerConfig::fromPlan(plan_);
        runner_config.hostfile = config_.hostfile;

        LOG_INFO("[OrchestrationRunner] Single-device precision config: activation="
                 << activationPrecisionToString(runner_config.activation_precision)
                 << ", kv_cache=" << kvCachePrecisionToString(runner_config.kv_cache_precision));

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

    // =========================================================================
    // Profiling
    // =========================================================================

    const GraphExecutorStats *OrchestrationRunner::executorStats() const
    {
        if (runner_)
        {
            return runner_->executorStats();
        }
        return nullptr;
    }

    void OrchestrationRunner::resetExecutorStats()
    {
        if (runner_)
        {
            runner_->resetExecutorStats();
        }
    }

    int OrchestrationRunner::sampleGreedyOnDevice()
    {
        if (runner_)
        {
            return runner_->sampleGreedyOnDevice();
        }
        return -1;
    }

    int OrchestrationRunner::sampleOnDevice(const SamplingParams &params)
    {
        if (runner_)
        {
            return runner_->sampleOnDevice(params);
        }
        return -1;
    }

    void OrchestrationRunner::setSkipLogitsGatherDecode(bool skip)
    {
        // Broadcast to worker ranks
        if (mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->rank() == 0 && mpi_ctx_->world_size() > 1)
        {
            broadcastCommand(MPICommand::SKIP_LOGITS_DECODE);
            int32_t val = skip ? 1 : 0;
            mpi_ctx_->broadcast_int32(&val, 1, 0);
        }

        if (runner_)
        {
            runner_->setSkipLogitsGatherDecode(skip);
        }
    }

    void OrchestrationRunner::setSkipLogitsGatherPrefill(bool skip)
    {
        if (runner_)
        {
            runner_->setSkipLogitsGatherPrefill(skip);
        }
    }

    void OrchestrationRunner::setSuppressTimeline(bool suppress)
    {
        if (runner_)
        {
            runner_->setSuppressTimeline(suppress);
        }
    }

    void OrchestrationRunner::setAccumulatePrefill(bool accumulate)
    {
        if (runner_)
        {
            runner_->setAccumulatePrefill(accumulate);
        }
    }

    void OrchestrationRunner::flushStageTimeline()
    {
        if (runner_)
        {
            runner_->flushStageTimeline();
        }
    }

    void OrchestrationRunner::setSamplingParams(const SamplingParams &params)
    {
        // Broadcast to worker ranks
        if (mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->rank() == 0 && mpi_ctx_->world_size() > 1)
        {
            broadcastCommand(MPICommand::SET_SAMPLING);
            float params_buf[4] = {
                params.temperature,
                params.top_p,
                static_cast<float>(params.top_k),
                static_cast<float>(params.seed)};
            mpi_ctx_->broadcast(params_buf, 4, 0);
        }

        active_sampling_params_ = params;
        // Reset token history for new conversation/request so penalties start fresh
        sampler_.reset_history();
    }

    SamplingParams OrchestrationRunner::getRecommendedSamplingParams() const
    {
        return recommended_sampling_params_;
    }

    // =========================================================================
    // MPI Worker Loop (non-root ranks in server mode)
    // =========================================================================

    void OrchestrationRunner::broadcastCommand(MPICommand cmd)
    {
        if (!mpi_coordinated_mode_ || !mpi_ctx_ || mpi_ctx_->world_size() <= 1)
            return;

        int32_t tag = static_cast<int32_t>(cmd);
        mpi_ctx_->broadcast_int32(&tag, 1, 0);
    }

    void OrchestrationRunner::shutdownMPIWorkers()
    {
        if (!mpi_ctx_ || mpi_ctx_->world_size() <= 1)
            return;

        LOG_INFO("[MPI] Rank 0 sending SHUTDOWN to worker ranks");
        broadcastCommand(MPICommand::SHUTDOWN);
    }

    void OrchestrationRunner::runMPIWorkerLoop()
    {
        if (!mpi_ctx_ || mpi_ctx_->rank() == 0)
        {
            LOG_WARN("[MPIWorkerLoop] Should only be called on non-root ranks");
            return;
        }

        LOG_INFO("[MPIWorkerLoop] Rank " << mpi_ctx_->rank()
                                         << " entering worker loop");

        while (true)
        {
            // Wait for command from rank 0
            int32_t tag = 0;
            mpi_ctx_->broadcast_int32(&tag, 1, 0);
            auto cmd = static_cast<MPICommand>(tag);

            switch (cmd)
            {
            case MPICommand::CLEAR_CACHE:
            {
                clearCache();
                break;
            }

            case MPICommand::SET_SAMPLING:
            {
                // Receive sampling params
                float params_buf[4]; // temperature, top_p, top_k, seed
                mpi_ctx_->broadcast(params_buf, 4, 0);
                SamplingParams sp;
                sp.temperature = params_buf[0];
                sp.top_p = params_buf[1];
                sp.top_k = static_cast<int>(params_buf[2]);
                sp.seed = static_cast<uint64_t>(params_buf[3]);
                setSamplingParams(sp);
                break;
            }

            case MPICommand::PREFILL:
            {
                // Receive token count then tokens
                int32_t n_tokens = 0;
                mpi_ctx_->broadcast_int32(&n_tokens, 1, 0);

                std::vector<int32_t> tokens(n_tokens);
                mpi_ctx_->broadcast_int32(tokens.data(), static_cast<size_t>(n_tokens), 0);

                prefill(tokens);
                break;
            }

            case MPICommand::DECODE_STEP:
            {
                decodeStep();
                break;
            }

            case MPICommand::SKIP_LOGITS_DECODE:
            {
                int32_t skip = 0;
                mpi_ctx_->broadcast_int32(&skip, 1, 0);
                runner_->setSkipLogitsGatherDecode(skip != 0);
                break;
            }

            case MPICommand::SHUTDOWN:
            {
                LOG_INFO("[MPIWorkerLoop] Rank " << mpi_ctx_->rank()
                                                 << " received SHUTDOWN");
                return;
            }

            default:
                LOG_WARN("[MPIWorkerLoop] Unknown command: " << tag);
                break;
            }
        }
    }

} // namespace llaminar2
