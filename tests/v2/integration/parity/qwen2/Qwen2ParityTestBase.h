/**
 * @file Qwen2ParityTestBase.h
 * @brief Base class and macros for Qwen2 PyTorch parity tests
 *
 * Provides model-specific infrastructure for Qwen2 parity testing.
 * Backend-specific tests (CPU, CUDA, ROCm) inherit from this and
 * only need to provide configuration - the test cases are generated
 * automatically via INSTANTIATE_QWEN2_PARITY_TESTS macro.
 *
 * Usage:
 *   class Test__Qwen2_CPU_vs_PyTorch : public Qwen2ParityTestBase {
 *   protected:
 *       BackendThresholds getBackendThresholds() override {
 *           return {.cosine_threshold=0.999f, .early_layers_count=4, ...};
 *       }
 *       DeviceId getDevice() override { return DeviceId::cpu(); }
 *       std::string getBackendName() override { return "CPU"; }
 *   };
 *   INSTANTIATE_QWEN2_PARITY_TESTS(Test__Qwen2_CPU_vs_PyTorch);
 *
 * @author David Sanftenberg
 * @date 2026-01-11
 */

#pragma once

#include "../ParityTestBase.h"
#include "models/qwen/Qwen2Schema.h"
#include "models/qwen/Qwen2Graph.h"
#include "execution/local_execution/orchestrators/MultiDeviceOrchestrator.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/runner/OrchestrationRunner.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "execution/local_execution/graph/GraphExecutor.h"
#include "execution/local_execution/graph/GraphBufferManager.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "collective/ILocalTPContext.h"
#include "collective/LocalTPContext.h"
#include "collective/ILocalPPContext.h"
#include "collective/IGlobalTPContext.h"
#include "collective/GlobalTPContext.h"
#include "collective/PPStage.h"
#include "config/PipelineConfig.h"
#include "tensors/TensorFactory.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/CPUKVCache.h"
#include "utils/Sampler.h"
#include "backends/GlobalDeviceAddress.h"
#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "collective/BackendRouter.h"
#include "../../../mocks/MockLocalPPContext.h"

namespace llaminar2::test::parity::qwen2
{

    // =============================================================================
    // Import common types from base parity namespace
    // =============================================================================
    using llaminar2::test::parity::BackendThresholds;
    using llaminar2::test::parity::checkHardwareAvailability;
    using llaminar2::test::parity::Collective;
    using llaminar2::test::parity::collectiveName;
    using llaminar2::test::parity::deviceTypeName;
    using llaminar2::test::parity::getCudaDeviceCount;
    using llaminar2::test::parity::getRocmDeviceCount;
    using llaminar2::test::parity::isMpiInitialized;
    using llaminar2::test::parity::Parallelism;
    using llaminar2::test::parity::parallelismName;
    using llaminar2::test::parity::ParityDeviceType;
    using llaminar2::test::parity::TestConfig;
    using llaminar2::test::parity::toCollectiveBackend;
    using llaminar2::test::parity::toDeviceId;
    using llaminar2::test::parity::toGlobalAddress;

    // =============================================================================
    // Qwen2-Specific Hardware Detection (supplements base utilities)
    // =============================================================================

    inline bool isNcclAvailable()
    {
#ifdef HAVE_NCCL
        return true;
#else
        return false;
#endif
    }

    inline bool isRcclAvailable()
    {
#ifdef HAVE_RCCL
        return true;
#else
        return false;
#endif
    }

    inline bool isPcieBarAvailable()
    {
        return getCudaDeviceCount() > 0 && getRocmDeviceCount() > 0;
    }

    // =============================================================================
    // Qwen2-Specific Hardware Availability Check (extends base check)
    // =============================================================================

    /**
     * @brief Extended hardware availability check for Qwen2 tests
     *
     * Extends the base checkHardwareAvailability() with collective backend checks
     * (NCCL, RCCL, PCIeBAR availability).
     */
    inline std::optional<std::string> checkQwen2HardwareAvailability(const TestConfig &cfg)
    {
        // First run base checks (device counts, MPI, etc.)
        if (auto reason = checkHardwareAvailability(cfg))
            return reason;

        // Additional checks for collective backends
        switch (cfg.collective)
        {
        case Collective::NCCL:
            if (!isNcclAvailable())
                return "NCCL not available";
            break;
        case Collective::RCCL:
            if (!isRcclAvailable())
                return "RCCL not available";
            break;
        case Collective::PCIeBAR:
            if (!isPcieBarAvailable())
                return "PCIeBAR requires both CUDA and ROCm devices";
            break;
        case Collective::MPI:
        case Collective::HOST:
        case Collective::None:
            // These are always available
            break;
        }

        return std::nullopt;
    }

    // =============================================================================
    // Base Test Class
    // =============================================================================

    /**
     * @brief Base class for Qwen2-specific parity tests
     *
     * Inherits from ParityTestBase and adds Qwen2-specific configuration.
     * Subclasses only need to implement:
     * - getBackendThresholds() - Return backend-specific thresholds
     * - getDevice() - Return DeviceId for inference
     * - getBackendName() - Return display name
     * - setupDeviceSpecific() (optional) - Device initialization
     */
    class Qwen2ParityTestBase : public ParityTestBase
    {
    protected:
        /**
         * @brief Get backend-specific threshold configuration
         * @return BackendThresholds struct with cosine/KL thresholds
         */
        virtual BackendThresholds getBackendThresholds() = 0;

        void SetUp() override
        {
            // Apply backend-specific thresholds
            auto thresholds = getBackendThresholds();
            config_.cosine_threshold = thresholds.cosine_threshold;
            config_.decode_cosine_threshold = thresholds.decode_cosine_threshold;
            config_.use_avg_cosine = true;
            config_.early_layers_count = thresholds.early_layers_count;
            config_.min_early_layers_passed = thresholds.min_early_layers_passed;
            config_.kl_threshold = thresholds.kl_threshold;
            config_.excluded_stages = thresholds.excluded_stages;

            ParityTestBase::SetUp();
        }
    };

    /**
     * @brief Config-driven base class for declarative parity tests
     *
     * This base class handles all the imperative setup for both single-device
     * and LocalTP configurations based on a TestConfig. Derived classes just
     * provide the configuration via getTestConfig().
     */
    template <typename Derived>
    class ConfigDrivenParityTest : public Qwen2ParityTestBase
    {
    protected:
        std::unique_ptr<MultiDeviceOrchestrator> multi_orch_;

        /**
         * @brief Get the test configuration (implement in derived class)
         */
        const TestConfig &cfg() const
        {
            return static_cast<const Derived *>(this)->getTestConfig();
        }

        // ==========================================================================
        // Qwen2ParityTestBase overrides - all derived from cfg()
        // ==========================================================================

        BackendThresholds getBackendThresholds() override
        {
            return cfg().thresholds;
        }

        std::string getBackendName() override
        {
            return cfg().name;
        }

        DeviceId getDevice() override
        {
            return toDeviceId(cfg().primary_device(), 0);
        }

        DeviceId getDeviceForRank() override
        {
            return toDeviceId(cfg().primary_device(), 0);
        }

        WeightDistributionStrategy getWeightStrategy() override
        {
            // LocalTP shards weights across devices
            // LocalPP uses LAYER_PARTITIONED - each stage loads only its assigned layers
            // GlobalTP shards weights across MPI ranks
            if (cfg().is_local_tp())
                return WeightDistributionStrategy::SHARDED;
            else if (cfg().is_local_pp())
                return WeightDistributionStrategy::LAYER_PARTITIONED;
            else if (cfg().is_global_tp())
                return WeightDistributionStrategy::SHARDED;
            else
                return WeightDistributionStrategy::REPLICATED;
        }

        void configureModel(std::shared_ptr<ModelContext> model_ctx) override
        {
            if (cfg().is_local_tp() || cfg().is_global_tp())
            {
                Qwen2SchemaFactory schema_factory;
                model_ctx->weightManager()->setWeightShardingConfig(
                    schema_factory.getWeightShardingConfig());
            }
        }

        // ==========================================================================
        // SetUp / TearDown
        // ==========================================================================

        void SetUp() override
        {
            // Check hardware availability (includes MPI check for LocalTP + NCCL/RCCL/PCIeBAR)
            if (auto skip_reason = checkQwen2HardwareAvailability(cfg()))
            {
                GTEST_SKIP() << *skip_reason;
            }

            // MPI setup for LOCAL TP/PP (MPI_Initialized already checked above)
            if (cfg().is_local_tp() || cfg().is_local_pp())
            {
                int rank = 0, world_size = 1;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &world_size);

                if (world_size != 1)
                {
                    GTEST_SKIP() << "LOCAL TP/PP test must run with -np 1 (got " << world_size << ")";
                }

                mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);
            }
            // MPI setup for GLOBAL TP (requires multiple ranks)
            else if (cfg().is_global_tp())
            {
                int rank = 0, world_size = 1;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &world_size);

                if (world_size < cfg().mpi_ranks)
                {
                    GTEST_SKIP() << "GLOBAL TP test requires " << cfg().mpi_ranks
                                 << " MPI ranks (got " << world_size << ")";
                }

                mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);
            }

            // Print test header
            LOG_INFO("╔══════════════════════════════════════════════════════════════════╗");
            LOG_INFO("║  PARITY TEST: " << cfg().name);
            LOG_INFO("╠══════════════════════════════════════════════════════════════════╣");
            LOG_INFO("║  Devices: " << cfg().device_count() << "x " << deviceTypeName(cfg().primary_device()));
            LOG_INFO("║  Parallelism: " << parallelismName(cfg().parallelism));
            LOG_INFO("║  Collective: " << collectiveName(cfg().collective));
            LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");

            Qwen2ParityTestBase::SetUp();
        }

        void TearDown() override
        {
            multi_orch_.reset();
            global_tp_ctx_.reset();
            pp_orchestrator_.reset();
            Qwen2ParityTestBase::TearDown();
        }

        // ==========================================================================
        // Pipeline Setup
        // ==========================================================================

        bool setupPipeline()
        {
            if (cfg().is_local_tp())
                return setupLocalTPPipeline();
            else if (cfg().is_local_pp())
                return setupLocalPPPipeline();
            else if (cfg().is_global_tp())
                return setupGlobalTPPipeline();
            else
                return ParityTestBase::setupPipeline();
        }

        bool setupLocalTPPipeline()
        {
            DeviceManager::instance().initialize(-1);

            model_ctx_ = ModelContext::create(
                config_.model_path,
                mpi_ctx_,
                nullptr,
                nullptr,
                WeightDistributionStrategy::SHARDED);

            if (!model_ctx_)
            {
                LOG_ERROR("[Parity] Failed to load model");
                return false;
            }

            configureModel(model_ctx_);

            // Build device list from config
            std::vector<GlobalDeviceAddress> devices;
            std::vector<float> weights;

            int cuda_idx = 0, rocm_idx = 0;
            for (auto dt : cfg().devices)
            {
                switch (dt)
                {
                case ParityDeviceType::CPU:
                    devices.push_back(GlobalDeviceAddress::cpu());
                    break;
                case ParityDeviceType::CUDA:
                    devices.push_back(GlobalDeviceAddress::cuda(cuda_idx++));
                    break;
                case ParityDeviceType::ROCm:
                    devices.push_back(GlobalDeviceAddress::rocm(rocm_idx++));
                    break;
                }
                weights.push_back(1.0f / static_cast<float>(cfg().device_count()));
            }

            auto tp_ctx = createLocalTPContext(
                devices, weights, toCollectiveBackend(cfg().collective));

            if (!tp_ctx)
            {
                LOG_ERROR("[Parity] Failed to create LocalTPContext");
                return false;
            }

            LOG_INFO("[Parity] LocalTPContext: degree=" << tp_ctx->degree()
                                                        << ", backend=" << static_cast<int>(tp_ctx->backend()));

            MultiDeviceOrchestrator::Config orch_config;
            orch_config.devices = devices;
            orch_config.weights = weights;
            orch_config.backend = toCollectiveBackend(cfg().collective);
            orch_config.max_seq_len = 4096;
            orch_config.batch_size = 1;

            multi_orch_ = std::make_unique<MultiDeviceOrchestrator>(
                model_ctx_, std::move(tp_ctx), orch_config);

            if (!multi_orch_)
            {
                LOG_ERROR("[Parity] Failed to create MultiDeviceOrchestrator");
                return false;
            }

            multi_orch_->enableSnapshotCapture();

            LOG_INFO("[Parity] MultiDeviceOrchestrator created with "
                     << multi_orch_->device_count() << " devices");

            runner_.reset(multi_orch_.release());
            multi_orch_ = nullptr;

            return true;
        }

        /**
         * @brief Setup pipeline for LocalPP tests using MultiDeviceOrchestrator PP mode
         *
         * Creates a pipeline parallel configuration where layers are split across
         * multiple devices. Uses MultiDeviceOrchestrator with PP mode which:
         * - Creates per-stage DeviceGraphOrchestrator instances
         * - Handles sequential forward execution through stages
         * - Manages activation transfer via LocalPPContext
         *
         * @return true if setup succeeded, false on error
         */
        bool setupLocalPPPipeline()
        {
            // Delegate to model-agnostic base class implementation
            return ParityTestBase::setupLocalPPPipeline();
        }

        /**
         * @brief Setup pipeline for GlobalTP tests using MPI
         *
         * Creates a Global TP configuration where weights are sharded across
         * multiple MPI ranks. Each rank participates in the TP domain and
         * contributes to collective operations via MPI.
         *
         * Key features:
         * - Uses GlobalTPContext for cross-rank collective operations
         * - MPI_COMM_WORLD is used as the domain communicator
         * - Each rank operates on its local CPU device
         *
         * @return true if setup succeeded, false on error
         */
        bool setupGlobalTPPipeline()
        {
            DeviceManager::instance().initialize(-1);

            // For GlobalTP, weights are sharded across MPI ranks
            model_ctx_ = ModelContext::create(
                config_.model_path,
                mpi_ctx_,
                nullptr,
                nullptr,
                WeightDistributionStrategy::SHARDED);

            if (!model_ctx_)
            {
                LOG_ERROR("[Parity] Failed to load model");
                return false;
            }

            configureModel(model_ctx_);

            // Get MPI info
            int rank = mpi_ctx_->rank();
            int world_size = mpi_ctx_->world_size();

            LOG_INFO("[Parity] GlobalTP setup: rank " << rank << "/" << world_size);

            // Build world_ranks vector (all ranks participate)
            std::vector<int> world_ranks;
            for (int r = 0; r < world_size; ++r)
            {
                world_ranks.push_back(r);
            }

            // Create GlobalTPContext using MPI_COMM_WORLD
            // Domain ID 0, all ranks participate
            global_tp_ctx_ = GlobalTPContext::createForTest(
                MPI_COMM_WORLD,
                0, // domain_id
                world_ranks);

            if (!global_tp_ctx_)
            {
                LOG_ERROR("[Parity] Failed to create GlobalTPContext");
                return false;
            }

            LOG_INFO("[Parity] GlobalTPContext created: degree=" << global_tp_ctx_->degree()
                                                                 << ", myIndex=" << global_tp_ctx_->myIndex());

            // For GlobalTP parity test, we need to use the graph execution path
            // similar to LocalTP, but with GlobalTPContext for collectives
            // TODO: Implement GlobalTP graph runner when ready
            // For now, GlobalTP parity tests will use a simplified approach
            // where we verify the GlobalTPContext operations work correctly

            // Placeholder: Use single-device execution on each rank for now
            // The parity comparison happens per-rank
            return ParityTestBase::setupPipeline();
        }

    protected:
        // PP-specific storage (production DeviceGraphOrchestrator for unified PP)
        std::unique_ptr<DeviceGraphOrchestrator> pp_orchestrator_;

        // GlobalTP-specific storage
        std::unique_ptr<GlobalTPContext> global_tp_ctx_;
    };

} // namespace llaminar2::test::parity::qwen2
