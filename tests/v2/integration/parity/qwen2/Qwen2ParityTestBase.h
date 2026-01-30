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
#include "execution/local_execution/orchestrators/MultiDeviceOrchestrator.h"
#include "collective/ILocalTPContext.h"
#include "collective/LocalTPContext.h"
#include "backends/GlobalDeviceAddress.h"
#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"

namespace llaminar2::test::parity::qwen2
{

    // =============================================================================
    // Device Configuration Types
    // =============================================================================

    /**
     * @brief Device type for parity tests
     * Note: Named ParityParityDeviceType to avoid collision with llaminar2::DeviceType
     */
    enum class ParityDeviceType
    {
        CPU,
        CUDA,
        ROCm
    };

    /**
     * @brief Parallelism strategy
     */
    enum class Parallelism
    {
        None,    ///< Single device, no parallelism
        LocalTP, ///< Local Tensor Parallelism (multi-device, single process)
        // Future: GlobalTP, PipelineParallel, Hybrid
    };

    /**
     * @brief Collective backend for tensor parallelism
     */
    enum class Collective
    {
        None,    ///< No collective needed (single device)
        NCCL,    ///< NVIDIA NCCL (CUDA-CUDA)
        RCCL,    ///< AMD RCCL (ROCm-ROCm)
        PCIeBAR, ///< Cross-vendor via PCIe BAR (CUDA-ROCm)
        // Future: MPI (for global TP)
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
        case Collective::NCCL:
            return CollectiveBackendType::NCCL;
        case Collective::RCCL:
            return CollectiveBackendType::RCCL;
        case Collective::PCIeBAR:
            return CollectiveBackendType::PCIE_BAR;
        case Collective::None:
            return CollectiveBackendType::MPI;
        }
        return CollectiveBackendType::MPI;
    }

    inline std::string deviceTypeName(ParityDeviceType t)
    {
        switch (t)
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

    inline std::string collectiveName(Collective c)
    {
        switch (c)
        {
        case Collective::None:
            return "None";
        case Collective::NCCL:
            return "NCCL";
        case Collective::RCCL:
            return "RCCL";
        case Collective::PCIeBAR:
            return "PCIeBAR";
        }
        return "Unknown";
    }

    // =============================================================================
    // Hardware Detection
    // =============================================================================

    inline int getCudaDeviceCount()
    {
        auto &dm = DeviceManager::instance();
        if (dm.devices().empty())
            dm.initialize(-1);

        int count = 0;
        for (const auto &dev : dm.devices())
            if (dev.type == ComputeBackendType::GPU_CUDA)
                count++;
        return count;
    }

    inline int getRocmDeviceCount()
    {
        auto &dm = DeviceManager::instance();
        if (dm.devices().empty())
            dm.initialize(-1);

        int count = 0;
        for (const auto &dev : dm.devices())
            if (dev.type == ComputeBackendType::GPU_ROCM)
                count++;
        return count;
    }

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

    /**
     * @brief Check if MPI is initialized
     */
    inline bool isMpiInitialized()
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        return initialized != 0;
    }

    // =============================================================================
    // Configuration Structures
    // =============================================================================

    /**
     * @brief Backend-specific threshold configuration
     *
     * Different backends have different quantization characteristics:
     * - CPU (Q8_1): Per-block quantization, tighter thresholds
     * - CUDA (INT8): Per-row symmetric, relaxed thresholds
     * - ROCm: Similar to CUDA
     */
    struct BackendThresholds
    {
        float cosine_threshold = 0.99f;        ///< Minimum cosine similarity for layer pass
        float decode_cosine_threshold = 0.99f; ///< Minimum avg cosine for decode steps
        int early_layers_count = 6;            ///< Number of early layers to check strictly
        int min_early_layers_passed = 6;       ///< Minimum early layers that must pass
        float kl_threshold = 0.15f;            ///< Maximum KL divergence for LM_HEAD

        /// Stages to exclude from per-layer parity comparison.
        std::vector<std::string> excluded_stages;
    };

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
        Collective collective;                 ///< Collective backend
        BackendThresholds thresholds;          ///< Parity thresholds
        std::string skip_reason;               ///< If non-empty, test will skip with this message

        // Derived accessors
        size_t device_count() const { return devices.size(); }
        ParityDeviceType primary_device() const { return devices.empty() ? ParityDeviceType::CPU : devices[0]; }
        bool is_local_tp() const { return parallelism == Parallelism::LocalTP; }
        bool is_single_device() const { return parallelism == Parallelism::None && devices.size() == 1; }
        bool should_skip() const { return !skip_reason.empty(); }
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

        // Check MPI initialization for LocalTP tests
        if (cfg.is_local_tp())
        {
            if (!isMpiInitialized())
                return "LocalTP requires MPI (run with mpirun -np 1)";
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
        case Collective::None:
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
            return cfg().is_local_tp() ? WeightDistributionStrategy::SHARDED
                                       : WeightDistributionStrategy::REPLICATED;
        }

        void configureModel(std::shared_ptr<ModelContext> model_ctx) override
        {
            if (cfg().is_local_tp())
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
            // Check hardware availability (includes MPI check for LocalTP)
            if (auto skip_reason = checkHardwareAvailability(cfg()))
            {
                GTEST_SKIP() << *skip_reason;
            }

            // MPI setup for LOCAL TP (MPI_Initialized already checked above)
            if (cfg().is_local_tp())
            {
                int rank = 0, world_size = 1;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &world_size);

                if (world_size != 1)
                {
                    GTEST_SKIP() << "LOCAL TP test must run with -np 1 (got " << world_size << ")";
                }

                mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);
            }

            // Print test header
            LOG_INFO("╔══════════════════════════════════════════════════════════════════╗");
            LOG_INFO("║  PARITY TEST: " << cfg().name);
            LOG_INFO("╠══════════════════════════════════════════════════════════════════╣");
            LOG_INFO("║  Devices: " << cfg().device_count() << "x " << deviceTypeName(cfg().primary_device()));
            LOG_INFO("║  Parallelism: " << (cfg().is_local_tp() ? "LOCAL TP" : "None"));
            LOG_INFO("║  Collective: " << collectiveName(cfg().collective));
            LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");

            Qwen2ParityTestBase::SetUp();
        }

        void TearDown() override
        {
            multi_orch_.reset();
            Qwen2ParityTestBase::TearDown();
        }

        // ==========================================================================
        // Pipeline Setup
        // ==========================================================================

        bool setupPipeline()
        {
            if (cfg().is_local_tp())
                return setupLocalTPPipeline();
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
    };

} // namespace llaminar2::test::parity::qwen2
