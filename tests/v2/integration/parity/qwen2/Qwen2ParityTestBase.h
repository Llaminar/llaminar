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
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/runner/OrchestrationRunner.h"
// Tree-based pipeline compilation (dogfooding ParallelismTree + TreeToRunnerCompiler)
#include "execution/parallelism_tree/ParallelismTree.h"
#include "execution/parallelism_tree/TreeToRunnerCompiler.h"
#include "execution/factory/FactoryPPStageConfig.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/device/DeviceContext.h"
// GlobalOrchestrator (cross-rank PP + global TP)
#include "execution/global/GlobalOrchestrator.h"
#include "execution/global_pp/GlobalPPTopology.h"
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
     * Extends the base checkHardwareAvailability() with collective backend checks.
     * For TP modes, checks cfg.collective. For hybrid PP+TP, checks cfg.tp_collective.
     * For pure PP, no collective check needed (transfers are auto-selected).
     */
    inline std::optional<std::string> checkQwen2HardwareAvailability(const TestConfig &cfg)
    {
        // First run base checks (device counts, MPI, etc.)
        if (auto reason = checkHardwareAvailability(cfg))
            return reason;

        // Determine which collective to check:
        // - TP modes use cfg.collective
        // - Hybrid PP+TP uses cfg.tp_collective (intra-stage TP backend)
        // - Pure PP has no collective to check
        Collective effective_collective = cfg.collective;
        if (cfg.is_local_pp() && cfg.is_hybrid_pp_tp())
            effective_collective = cfg.tp_collective;

        switch (effective_collective)
        {
        case Collective::NCCL:
            if (!isNcclAvailable())
                return "NCCL not available";
            break;
        case Collective::RCCL:
            if (!isRcclAvailable())
                return "RCCL not available";
            break;
        case Collective::HETEROGENEOUS:
            if (!isPcieBarAvailable())
                return "HETEROGENEOUS requires both CUDA and ROCm devices";
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
    // Tree-Based Pipeline Construction (dogfooding ParallelismTree)
    // =============================================================================

    /**
     * @brief Build a ParallelismTree from a TestConfig
     *
     * Converts the declarative TestConfig (devices, parallelism, collective, pp_stage_sizes)
     * into a ParallelismTree suitable for compilation via TreeToRunnerCompiler.
     *
     * Mapping:
     *   - Parallelism::None      → Single DEVICE leaf
     *   - Parallelism::LocalTP   → TP root with DEVICE children (one per device)
     *   - Parallelism::LocalPP   → PP root with DEVICE/TP children per stage
     *   - Parallelism::GlobalTP  → TP root with DEVICE children across MPI ranks
     *
     * @param cfg Test configuration
     * @param n_layers Total transformer layers (from model_ctx->blockCount())
     * @return ParallelismTree with layers assigned
     */
    inline ParallelismTree buildTreeFromTestConfig(const TestConfig &cfg, int n_layers)
    {
        // Build GlobalDeviceAddress list from TestConfig devices
        auto buildDeviceAddresses = [](const std::vector<ParityDeviceType> &devices)
        {
            std::vector<GlobalDeviceAddress> addrs;
            int cuda_idx = 0, rocm_idx = 0;
            for (auto dt : devices)
            {
                switch (dt)
                {
                case ParityDeviceType::CPU:
                    addrs.push_back(GlobalDeviceAddress::cpu());
                    break;
                case ParityDeviceType::CUDA:
                    addrs.push_back(GlobalDeviceAddress::cuda(cuda_idx++));
                    break;
                case ParityDeviceType::ROCm:
                    addrs.push_back(GlobalDeviceAddress::rocm(rocm_idx++));
                    break;
                }
            }
            return addrs;
        };

        auto all_devices = buildDeviceAddresses(cfg.devices);
        const int owning_rank = 0; // Parity tests are single-rank (rank 0)

        ParallelismTree tree;
        tree.world_size = 1;

        switch (cfg.parallelism)
        {
        case Parallelism::None:
        {
            // Single DEVICE leaf
            tree.root = Device(all_devices[0], owning_rank);
            break;
        }

        case Parallelism::LocalTP:
        {
            // TP root with DEVICE children
            tree.root = TP("local_tp", all_devices, owning_rank,
                           toCollectiveBackend(cfg.collective));
            break;
        }

        case Parallelism::LocalPP:
        {
            if (cfg.is_hybrid_pp_tp())
            {
                // Hybrid PP+TP: PP root with mixed TP/DEVICE children
                std::vector<ParallelismNode> pp_children;
                size_t device_offset = 0;

                for (size_t s = 0; s < cfg.pp_stage_sizes.size(); ++s)
                {
                    int stage_device_count = cfg.pp_stage_sizes[s];

                    if (stage_device_count > 1)
                    {
                        // This stage is a TP domain
                        std::vector<GlobalDeviceAddress> stage_devices;
                        for (int d = 0; d < stage_device_count && device_offset < all_devices.size(); ++d)
                        {
                            stage_devices.push_back(all_devices[device_offset++]);
                        }

                        // Use tp_collective for intra-stage TP backend
                        CollectiveBackendType tp_backend = toCollectiveBackend(cfg.tp_collective);

                        pp_children.push_back(
                            TP("stage" + std::to_string(s) + "_tp",
                               stage_devices, owning_rank, tp_backend));
                    }
                    else
                    {
                        // Single device stage
                        if (device_offset < all_devices.size())
                        {
                            pp_children.push_back(
                                Device(all_devices[device_offset++], owning_rank));
                        }
                    }
                }

                tree.root = PP("local_pp", std::move(pp_children));
            }
            else
            {
                // Pure PP: one device per stage
                std::vector<ParallelismNode> pp_children;
                for (size_t i = 0; i < all_devices.size(); ++i)
                {
                    pp_children.push_back(Device(all_devices[i], owning_rank));
                }
                tree.root = PP("local_pp", std::move(pp_children));
            }
            break;
        }

        case Parallelism::GlobalTP:
        {
            // GlobalTP: TP root with DEVICE children across MPI ranks
            // Each rank gets one device
            std::vector<ParallelismNode> tp_children;
            for (int r = 0; r < cfg.mpi_ranks; ++r)
            {
                GlobalDeviceAddress addr = (r < static_cast<int>(all_devices.size()))
                                               ? all_devices[r]
                                               : GlobalDeviceAddress::cpu();
                tp_children.push_back(Device(addr, r));
            }
            tree.root = TP("global_tp", std::move(tp_children),
                           toCollectiveBackend(cfg.collective));
            tree.world_size = cfg.mpi_ranks;
            break;
        }
        }

        // Assign layers
        tree.assignLayers(n_layers);

        // Validate
        auto errors = tree.validate();
        if (!errors.empty())
        {
            std::string msg = "Tree validation failed:\n";
            for (const auto &e : errors)
                msg += "  - " + e + "\n";
            LOG_ERROR("[Parity] " << msg);
        }

        LOG_INFO("[Parity] Built parallelism tree:\n"
                 << tree.toString());

        return tree;
    }

    /**
     * @brief Create real runner factories for TreeToRunnerCompiler
     *
     * These factories bridge the tree compiler to the existing InferenceRunnerFactory
     * and RankOrchestrator infrastructure. They are the "real implementation"
     * that the compiler calls when it encounters DEVICE/TP/PP nodes.
     */
    namespace tree_factories
    {

        /**
         * @brief Factory that creates a DeviceGraphOrchestrator from a DEVICE node
         *
         * Uses the existing createInferenceRunner() factory which handles all the
         * GraphConfig setup, weight loading, and graph construction.
         */
        inline TreeToRunnerCompiler::DeviceRunnerFactory makeDeviceFactory(
            std::shared_ptr<IMPIContext> mpi_ctx,
            const InferenceRunnerConfig &base_config)
        {
            return [mpi_ctx, base_config](
                       const ParallelismNode &node,
                       const std::shared_ptr<IModelContext> &model_ctx) -> std::unique_ptr<IInferenceRunner>
            {
                // Convert GlobalDeviceAddress → DeviceId for existing factory
                DeviceId device = node.device.toLocalDeviceId();

                // Build runner config with PP stage info from tree node
                InferenceRunnerConfig config = base_config;

                // Tree uses inclusive last_layer, FactoryPPStageConfig uses exclusive
                FactoryPPStageConfig pp_cfg;
                pp_cfg.first_layer = node.first_layer;
                pp_cfg.last_layer = node.last_layer + 1;
                pp_cfg.has_embedding = node.has_embedding;
                pp_cfg.has_lm_head = node.has_lm_head;

                auto concrete_ctx = std::dynamic_pointer_cast<ModelContext>(model_ctx);
                if (!concrete_ctx)
                {
                    LOG_ERROR("[TreeFactory] model_ctx is not a concrete ModelContext");
                    return nullptr;
                }

                return createPPStageRunner(concrete_ctx, device, pp_cfg, config);
            };
        }

        /**
         * @brief Factory that creates a RankOrchestrator(TP) from a TP node
         *
         * The child_runners are already compiled DeviceGraphOrchestrators.
         * We wrap them in a RankOrchestrator for TP coordination.
         */
        inline TreeToRunnerCompiler::TPRunnerFactory makeTPFactory()
        {
            return [](const ParallelismNode &node,
                      std::vector<std::unique_ptr<IInferenceRunner>> child_runners,
                      const std::shared_ptr<IModelContext> &model_ctx) -> std::unique_ptr<IInferenceRunner>
            {
                // Build device list and weights from tree node children
                std::vector<GlobalDeviceAddress> devices;
                std::vector<float> weights;
                for (const auto &child : node.children)
                {
                    auto leaves = child.leafDevices();
                    for (const auto *leaf : leaves)
                    {
                        devices.push_back(leaf->device);
                    }
                }

                // Equal weights if not specified
                if (node.tp_weights.empty())
                {
                    float w = 1.0f / static_cast<float>(devices.size());
                    weights.assign(devices.size(), w);
                }
                else
                {
                    weights = node.tp_weights;
                }

                // Create LocalTPContext
                auto tp_ctx = createLocalTPContext(devices, weights, node.backend);
                if (!tp_ctx)
                {
                    LOG_ERROR("[TreeFactory] Failed to create LocalTPContext");
                    return nullptr;
                }

                // Build MDO config for TP mode
                RankOrchestrator::Config mdo_config;
                mdo_config.mode = RankOrchestrator::ParallelismMode::TP;
                mdo_config.devices = devices;
                mdo_config.weights = weights;
                mdo_config.backend = node.backend;
                mdo_config.max_seq_len = 4096;
                mdo_config.batch_size = 1;

                // If this TP node handles a subset of layers (inside PP), configure nested PP stage
                int total_layers = model_ctx->blockCount();
                if (node.first_layer > 0 || node.last_layer < total_layers - 1)
                {
                    // Tree uses inclusive last_layer, FactoryPPStageConfig uses exclusive
                    FactoryPPStageConfig pp_cfg;
                    pp_cfg.first_layer = node.first_layer;
                    pp_cfg.last_layer = node.last_layer + 1;
                    pp_cfg.has_embedding = node.has_embedding;
                    pp_cfg.has_lm_head = node.has_lm_head;
                    mdo_config.nested_pp_stage_config = pp_cfg;
                }

                auto orch = std::make_unique<RankOrchestrator>(
                    model_ctx, mdo_config, std::move(tp_ctx));

                return orch;
            };
        }

        // =============================================================================
        // LocalPPTestRunner — test-only PP wrapper using pre-compiled child runners
        // =============================================================================

        /**
         * @brief Test-only IInferenceRunner wrapper for local Pipeline Parallelism
         *
         * Wraps pre-compiled child runners (from TreeToRunnerCompiler) with
         * PP forward sequencing: runs stages sequentially, transferring the
         * hidden state between them via a LocalPPContext.
         *
         * This avoids creating a RankOrchestrator(TP_PP) from scratch,
         * which fails for hybrid PP+TP because the MDO's internal
         * initializePPDeviceRunners() creates PP-stage-filtered model contexts
         * that lack global weights (output_norm, output/lm_head) needed by
         * createTestableInferenceRunner() in the nested TP MDO.
         *
         * Instead, the tree compiler creates child runners correctly:
         *   - TP stage: MDO(TP) from FULL model context with nested_pp_stage_config
         *   - Device stage: DGO from createPPStageRunner with partial weights
         *
         * This wrapper sequences those pre-built runners with PP semantics.
         */
        class LocalPPTestRunner : public IInferenceRunner
        {
        public:
            LocalPPTestRunner(
                std::vector<std::unique_ptr<IInferenceRunner>> stage_runners,
                std::unique_ptr<ILocalPPContext> pp_ctx)
                : stage_runners_(std::move(stage_runners)),
                  pp_ctx_(std::move(pp_ctx))
            {
                if (stage_runners_.empty())
                {
                    throw std::invalid_argument("LocalPPTestRunner: no stage runners provided");
                }
                LOG_INFO("[LocalPPTestRunner] Created with " << stage_runners_.size()
                                                             << " PP stages");
            }

            // ================================================================
            // Core Inference API
            // ================================================================

            bool forward(const int *tokens, int seq_len) override
            {
                // Stage 0: run with tokens (has embedding)
                if (!stage_runners_[0]->forward(tokens, seq_len))
                {
                    LOG_ERROR("[LocalPPTestRunner] Stage 0 forward failed");
                    return false;
                }

                // Subsequent stages: transfer hidden state, then run
                for (size_t i = 1; i < stage_runners_.size(); ++i)
                {
                    TensorBase *hidden = stage_runners_[i - 1]->getHiddenState();
                    if (!hidden)
                    {
                        LOG_ERROR("[LocalPPTestRunner] Stage " << (i - 1)
                                                               << " has no hidden state to transfer");
                        return false;
                    }

                    // Transfer hidden state between devices
                    if (pp_ctx_)
                    {
                        if (!pp_ctx_->transfer(hidden, static_cast<int>(i - 1),
                                               static_cast<int>(i)))
                        {
                            LOG_ERROR("[LocalPPTestRunner] Transfer from stage "
                                      << (i - 1) << " to stage " << i << " failed");
                            return false;
                        }
                    }

                    stage_runners_[i]->setHiddenState(hidden);

                    // Forward with nullptr tokens — stage uses hidden state input
                    if (!stage_runners_[i]->forward(nullptr, seq_len))
                    {
                        LOG_ERROR("[LocalPPTestRunner] Stage " << i << " forward failed");
                        return false;
                    }

                    stage_runners_[i]->clearHiddenStateInput();
                }

                current_position_ += seq_len;
                return true;
            }

            const float *logits() const override
            {
                return stage_runners_.back()->logits();
            }

            int vocab_size() const override
            {
                return stage_runners_.back()->vocab_size();
            }

            void clear_cache() override
            {
                for (auto &runner : stage_runners_)
                    runner->clear_cache();
                current_position_ = 0;
            }

            int get_position() const override
            {
                return current_position_;
            }

            ExecutionPath executionPath() const override
            {
                return ExecutionPath::GRAPH;
            }

            const char *architecture() const override
            {
                return stage_runners_.front()->architecture();
            }

            // ================================================================
            // Snapshot API — aggregate from all stages
            // ================================================================

            void enableSnapshotCapture(const std::string &output_dir = "") override
            {
                for (auto &runner : stage_runners_)
                    runner->enableSnapshotCapture(output_dir);
            }

            void disableSnapshotCapture() override
            {
                for (auto &runner : stage_runners_)
                    runner->disableSnapshotCapture();
            }

            void clearSnapshots() override
            {
                for (auto &runner : stage_runners_)
                    runner->clearSnapshots();
            }

            const float *getSnapshot(const std::string &key, size_t &out_size) const override
            {
                // Search all stages for the snapshot key
                for (const auto &runner : stage_runners_)
                {
                    const float *data = runner->getSnapshot(key, out_size);
                    if (data)
                        return data;
                }
                out_size = 0;
                return nullptr;
            }

            std::vector<std::string> getSnapshotKeys() const override
            {
                std::vector<std::string> all_keys;
                for (const auto &runner : stage_runners_)
                {
                    auto keys = runner->getSnapshotKeys();
                    all_keys.insert(all_keys.end(), keys.begin(), keys.end());
                }
                return all_keys;
            }

            // ================================================================
            // Hidden State API — delegate to first/last stage
            // ================================================================

            TensorBase *getHiddenState() override
            {
                return stage_runners_.back()->getHiddenState();
            }

            const TensorBase *getHiddenState() const override
            {
                return stage_runners_.back()->getHiddenState();
            }

            void setHiddenState(TensorBase *hidden_state) override
            {
                stage_runners_.front()->setHiddenState(hidden_state);
            }

            bool hasHiddenStateInput() const override
            {
                return stage_runners_.front()->hasHiddenStateInput();
            }

            void clearHiddenStateInput() override
            {
                for (auto &runner : stage_runners_)
                    runner->clearHiddenStateInput();
            }

        private:
            std::vector<std::unique_ptr<IInferenceRunner>> stage_runners_;
            std::unique_ptr<ILocalPPContext> pp_ctx_;
            int current_position_ = 0;
        };

        /**
         * @brief Factory that creates a PP runner from a PP node
         *
         * For pure PP (all stages are single devices): creates a
         * RankOrchestrator(PP) from scratch (works correctly because
         * MDO(PP) uses createPPStageRunner for each stage, handling partial
         * weights properly).
         *
         * For hybrid PP+TP: uses LocalPPTestRunner to wrap the pre-compiled
         * child runners. This avoids creating MDO(TP_PP) from scratch, which
         * would fail because the MDO's internal initializePPDeviceRunners()
         * creates stage-filtered model contexts that lack global weights
         * needed by the nested TP MDO's createTestableInferenceRunner().
         */
        inline TreeToRunnerCompiler::LocalPPRunnerFactory makeLocalPPFactory()
        {
            return [](const ParallelismNode &node,
                      std::vector<std::unique_ptr<IInferenceRunner>> child_runners,
                      const std::shared_ptr<IModelContext> &model_ctx) -> std::unique_ptr<IInferenceRunner>
            {
                // Check if any stage is a TP domain
                bool has_tp_stages = false;
                for (const auto &child : node.children)
                {
                    if (child.type == ParallelismNodeType::TENSOR_PARALLEL)
                    {
                        has_tp_stages = true;
                        break;
                    }
                }

                // =========================================================
                // Both pure PP and hybrid PP+TP use MDO's production path.
                // MDO::detectMode() auto-selects PP vs TP_PP based on
                // whether any stage has multiple devices (isTPDomain()).
                // =========================================================
                (void)has_tp_stages; // Used only for logging below
                (void)child_runners; // MDO creates its own runners internally

                RankOrchestrator::Config mdo_config;
                mdo_config.max_seq_len = 4096;
                mdo_config.batch_size = 1;

                for (const auto &child : node.children)
                {
                    RankOrchestrator::PPStageConfig stage_cfg;
                    stage_cfg.first_layer = child.first_layer;
                    stage_cfg.last_layer = child.last_layer + 1;
                    stage_cfg.has_embedding = child.has_embedding;
                    stage_cfg.has_lm_head = child.has_lm_head;

                    auto leaves = child.leafDevices();
                    for (const auto *leaf : leaves)
                    {
                        stage_cfg.stage_devices.push_back(leaf->device);
                    }

                    // Propagate TP config from tree node
                    if (child.type == ParallelismNodeType::TENSOR_PARALLEL)
                    {
                        stage_cfg.tp_weights = child.tp_weights;
                        stage_cfg.tp_backend = child.backend;
                    }

                    mdo_config.pp_stages.push_back(std::move(stage_cfg));
                }

                // AUTO mode: MDO detects PP vs TP_PP based on stage device counts
                mdo_config.mode = RankOrchestrator::ParallelismMode::AUTO;

                if (!mdo_config.validate())
                {
                    LOG_ERROR("[TreeFactory] Invalid PP config from tree");
                    return nullptr;
                }

                auto orch = std::make_unique<RankOrchestrator>(model_ctx, mdo_config);
                return orch;
            };
        }

    } // namespace tree_factories

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

        /**
         * @brief Apply model path/snapshot dir overrides from TestConfig
         *
         * Override in subclasses that have access to TestConfig (e.g., ConfigDrivenParityTest).
         * Default implementation does nothing (preserves ParityConfig defaults).
         */
        virtual void applyModelOverrides() {}

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
            config_.min_top1_accuracy = thresholds.min_top1_accuracy;
            config_.min_top5_accuracy = thresholds.min_top5_accuracy;
            config_.min_decode_pass_rate = thresholds.min_decode_pass_rate;
            config_.pytorch_top1_in_topk = thresholds.pytorch_top1_in_topk;

            // Apply model path/snapshot dir overrides from TestConfig if available
            applyModelOverrides();

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
        std::unique_ptr<RankOrchestrator> multi_orch_;

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

        void applyModelOverrides() override
        {
            // Qwen2 defaults (pushed down from ParityTestBase)
            if (config_.model_path.empty())
                config_.model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
            if (config_.prompt.empty())
                config_.prompt = "The quick brown fox jumps over the lazy dog";
            if (config_.token_ids.empty())
                config_.token_ids = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};

            // Per-test overrides (from TestConfig)
            if (!cfg().model_path.empty())
                config_.model_path = cfg().model_path;
            if (!cfg().snapshot_dir.empty())
                config_.snapshot_dir = cfg().snapshot_dir;
            if (cfg().decode_steps > 0)
                config_.decode_steps = cfg().decode_steps;
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
            // LocalPP: LAYER_PARTITIONED is semantically correct (PP = layer split),
            // but MDO creates per-stage ModelContexts internally via createForPPStage(),
            // so top-level strategy is actually irrelevant for PP.
            // GlobalTP shards weights across MPI ranks
            if (cfg().is_local_tp() || cfg().is_cross_rank_tp())
                return WeightDistributionStrategy::SHARDED;
            else if (cfg().is_local_pp() || cfg().is_cross_rank_pp())
                return WeightDistributionStrategy::LAYER_PARTITIONED;
            else
                return WeightDistributionStrategy::REPLICATED;
        }

        void configureModel(std::shared_ptr<ModelContext> model_ctx) override
        {
            if (cfg().is_local_tp() || cfg().is_cross_rank_tp())
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
            // Check hardware availability (includes MPI check for LocalTP + NCCL/RCCL/HOST)
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
            // MPI setup for cross-rank (NodeLocal or Global, requires multiple ranks)
            else if (cfg().is_cross_rank())
            {
                int rank = 0, world_size = 1;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &world_size);

                if (world_size < cfg().mpi_ranks)
                {
                    GTEST_SKIP() << "Cross-rank test requires " << cfg().mpi_ranks
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
            if (cfg().is_local_pp())
            {
                if (cfg().is_hybrid_pp_tp())
                    LOG_INFO("║  TP Collective: " << collectiveName(cfg().tp_collective));
                // Pure PP: no collective to display (transfers auto-selected)
            }
            else
            {
                LOG_INFO("║  Collective: " << collectiveName(cfg().collective));
            }
            if (!cfg().model_path.empty())
                LOG_INFO("║  Model: " << cfg().model_path);
            LOG_INFO("╚══════════════════════════════════════════════════════════════════╝");

            Qwen2ParityTestBase::SetUp();
        }

        void TearDown() override
        {
            multi_orch_.reset();
            global_orchestrator_ptr_ = nullptr;
            global_tp_ctx_.reset();
            pp_orchestrator_.reset();
            Qwen2ParityTestBase::TearDown();
        }

        // ==========================================================================
        // Pipeline Setup — Tree-based (dogfooding ParallelismTree + Compiler)
        // ==========================================================================

        /**
         * @brief Setup pipeline by building a ParallelismTree and compiling it
         *
         * This is the tree-based alternative to the old imperative setup methods.
         * It dogfoods the ParallelismTree builder + TreeToRunnerCompiler infrastructure
         * that the production OrchestrationRunner uses, giving us confidence that
         * the tree→runner compilation produces correct inference results.
         *
         * Flow:
         *   1. Load model (ModelContext::create)
         *   2. Build ParallelismTree from TestConfig via buildTreeFromTestConfig()
         *   3. Create real runner factories (DEVICE, TP, PP)
         *   4. Compile tree → IInferenceRunner via TreeToRunnerCompiler::compile()
         *   5. Enable snapshot capture
         */
        bool setupPipeline()
        {
            // Cross-rank TP or PP uses GlobalOrchestrator
            if (cfg().is_cross_rank())
                return setupGlobalOrchestratorPipeline();

            // Single device also uses old path (no tree needed)
            if (cfg().is_single_device())
                return ParityTestBase::setupPipeline();

            // ============================================================
            // Tree-based path: LocalTP, LocalPP, HybridPP+TP
            // ============================================================
            return setupTreePipeline();
        }

        /**
         * @brief Tree-based pipeline setup for LocalTP, LocalPP, and HybridPP+TP
         *
         * Builds a ParallelismTree from TestConfig, creates real factories
         * that produce DeviceGraphOrchestrators and RankOrchestrators,
         * then compiles the tree into a nested IInferenceRunner hierarchy.
         */
        bool setupTreePipeline()
        {
            DeviceManager::instance().initialize(-1);

            // For PP, initialize GlobalBackendRouter for activation transfers
            if (cfg().is_local_pp())
            {
                GlobalBackendRouter::initForTests();
            }

            // Determine weight strategy
            WeightDistributionStrategy weight_strategy = getWeightStrategy();

            // Load model
            model_ctx_ = ModelContext::create(
                config_.model_path,
                mpi_ctx_,
                nullptr,
                nullptr,
                weight_strategy);

            if (!model_ctx_)
            {
                LOG_ERROR("[Parity/Tree] Failed to load model");
                return false;
            }

            // Configure model (weight sharding schema for TP)
            configureModel(model_ctx_);

            int n_layers = model_ctx_->blockCount();

            // Step 1: Build the parallelism tree from TestConfig
            auto tree = buildTreeFromTestConfig(cfg(), n_layers);

            // Step 2: Build the compile context with real factories
            InferenceRunnerConfig base_runner_config;
            base_runner_config.max_seq_len = 4096;
            base_runner_config.batch_size = 1;
            base_runner_config.force_graph = true;
            base_runner_config.use_mapped_memory = true; // For GPU snapshot capture

            TreeToRunnerCompiler::CompileContext compile_ctx;
            compile_ctx.model_ctx = model_ctx_;
            compile_ctx.my_rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            compile_ctx.world_size = mpi_ctx_ ? mpi_ctx_->world_size() : 1;
            compile_ctx.max_seq_len = 4096;
            compile_ctx.batch_size = 1;
            compile_ctx.hidden_dim = model_ctx_->concreteLoader().getModel().embedding_length;
            compile_ctx.vocab_size = model_ctx_->concreteLoader().getModel().vocab_size;

            // Step 3: Wire up real factories
            compile_ctx.device_runner_factory = tree_factories::makeDeviceFactory(
                mpi_ctx_, base_runner_config);
            compile_ctx.tp_runner_factory = tree_factories::makeTPFactory();
            compile_ctx.local_pp_runner_factory = tree_factories::makeLocalPPFactory();

            // Step 4: Compile tree → runner
            LOG_INFO("[Parity/Tree] Compiling parallelism tree for " << cfg().name);

            runner_ = TreeToRunnerCompiler::compile(tree, compile_ctx);
            if (!runner_)
            {
                LOG_ERROR("[Parity/Tree] TreeToRunnerCompiler::compile() returned nullptr");
                return false;
            }

            // Step 5: Enable snapshot capture
            runner_->enableSnapshotCapture();

            LOG_INFO("[Parity/Tree] Pipeline created via tree compilation for " << cfg().name);
            return true;
        }

        // ==========================================================================
        // Legacy imperative setup methods (kept for GlobalTP and fallback)
        // These are superseded by setupTreePipeline() for LocalTP and LocalPP.
        // ==========================================================================

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

            RankOrchestrator::Config orch_config;
            orch_config.devices = devices;
            orch_config.weights = weights;
            orch_config.backend = toCollectiveBackend(cfg().collective);
            orch_config.max_seq_len = 4096;
            orch_config.batch_size = 1;

            multi_orch_ = std::make_unique<RankOrchestrator>(
                model_ctx_, orch_config, std::move(tp_ctx));

            if (!multi_orch_)
            {
                LOG_ERROR("[Parity] Failed to create RankOrchestrator");
                return false;
            }

            multi_orch_->enableSnapshotCapture();

            LOG_INFO("[Parity] RankOrchestrator created with "
                     << multi_orch_->device_count() << " devices");

            runner_.reset(multi_orch_.release());
            multi_orch_ = nullptr;

            return true;
        }

        /**
         * @brief Setup pipeline for LocalPP tests using RankOrchestrator PP mode
         *
         * Creates a pipeline parallel configuration where layers are split across
         * multiple devices. Uses RankOrchestrator with PP mode which:
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
        bool setupGlobalOrchestratorPipeline()
        {
            DeviceManager::instance().initialize(-1);

            const int rank = mpi_ctx_->rank();
            const int world_size = mpi_ctx_->world_size();

            // Step 1: Load model
            model_ctx_ = ModelContext::create(
                config_.model_path,
                mpi_ctx_,
                nullptr,            // placement_map
                nullptr,            // factory
                getWeightStrategy());
            if (!model_ctx_)
            {
                LOG_ERROR("[Parity] Failed to load model");
                return false;
            }
            configureModel(model_ctx_);

            const int n_layers = model_ctx_->blockCount();
            const int vocab_size = model_ctx_->vocabSize();
            const int d_model = model_ctx_->embeddingLength();
            const std::string arch_name = model_ctx_->architecture();

            // Step 2: Build GlobalPPTopology
            GlobalPPTopology topology;

            if (cfg().is_cross_rank_tp())
            {
                // Pure global TP: single stage, all ranks, all layers
                GlobalPPStageSpec stage;
                stage.stage_id = 0;
                stage.first_layer = 0;
                stage.last_layer = n_layers - 1;
                stage.has_embedding = true;
                stage.has_lm_head = true;
                stage.is_global_tp = true;
                for (int r = 0; r < world_size; ++r)
                    stage.participating_ranks.push_back(r);

                topology = GlobalPPTopology::build({stage}, n_layers, world_size);
            }
            else if (cfg().is_cross_rank_pp())
            {
                // Pure global PP: one stage per rank, equal layer split
                int layers_per_rank = n_layers / world_size;
                int remainder = n_layers % world_size;
                std::vector<GlobalPPStageSpec> stages;

                int layer_offset = 0;
                for (int r = 0; r < world_size; ++r)
                {
                    int count = layers_per_rank + (r < remainder ? 1 : 0);
                    GlobalPPStageSpec stage;
                    stage.stage_id = r;
                    stage.first_layer = layer_offset;
                    stage.last_layer = layer_offset + count - 1;
                    stage.has_embedding = (r == 0);
                    stage.has_lm_head = (r == world_size - 1);
                    stage.is_global_tp = false;
                    stage.owning_rank = r;
                    stages.push_back(stage);
                    layer_offset += count;
                }

                topology = GlobalPPTopology::build(std::move(stages), n_layers, world_size);
            }
            else
            {
                LOG_ERROR("[Parity] setupGlobalOrchestratorPipeline() called for unsupported parallelism");
                return false;
            }

            // Validate topology
            auto errors = topology.validate();
            if (!errors.empty())
            {
                for (const auto &err : errors)
                    LOG_ERROR("[Parity] Topology error: " << err);
                return false;
            }

            // Step 3: Create per-rank runner
            InferenceRunnerConfig inf_config;
            inf_config.max_seq_len = 4096;
            inf_config.batch_size = 1;
            inf_config.force_graph = true;
            inf_config.activation_precision = cfg().activation_precision;
            inf_config.kv_cache_precision = cfg().kv_cache_precision;

            DeviceId device = getDevice();
            if (device.is_gpu())
                inf_config.use_mapped_memory = true;

            if (cfg().is_cross_rank_pp())
            {
                // PP: configure runner for this rank's layer range
                const auto *my_stage = topology.stageForLayer(
                    topology.stages[rank].first_layer);
                if (!my_stage)
                {
                    LOG_ERROR("[Parity] No stage found for rank " << rank);
                    return false;
                }

                FactoryPPStageConfig pp_cfg;
                pp_cfg.first_layer = my_stage->first_layer;
                pp_cfg.last_layer = my_stage->last_layer + 1; // exclusive
                pp_cfg.has_embedding = my_stage->has_embedding;
                pp_cfg.has_lm_head = my_stage->has_lm_head;

                // Use createPPStageRunner which builds a partial graph with only
                // this rank's layers (skips embedding/LM head as appropriate).
                // createInferenceRunner ignores pp_stage_config and builds a full graph.
                runner_ = createPPStageRunner(model_ctx_, device, pp_cfg, inf_config);
                if (!runner_)
                {
                    LOG_ERROR("[Parity] Failed to create PP stage runner for rank " << rank);
                    return false;
                }
                runner_->enableSnapshotCapture();
            }
            else
            {
                // Pure TP: create full-model runner
                runner_ = createInferenceRunner(model_ctx_, mpi_ctx_, device, inf_config);
                if (!runner_)
                {
                    LOG_ERROR("[Parity] Failed to create per-rank inference runner");
                    return false;
                }
                runner_->enableSnapshotCapture();
            }

            // Step 4: Build GlobalOrchestrator wrapping the per-rank runner
            GlobalOrchestrator::Config go_config;
            go_config.topology = std::move(topology);
            go_config.rank = rank;
            go_config.world_size = world_size;
            go_config.mpi_ctx = mpi_ctx_.get();
            go_config.rank_runner = std::move(runner_);  // Transfer ownership
            go_config.vocab_size = vocab_size;
            go_config.d_model = d_model;
            go_config.architecture_name = arch_name;

            auto go = std::make_unique<GlobalOrchestrator>(std::move(go_config));

            // Keep a non-owning pointer for tests that need GlobalOrchestrator-specific APIs
            global_orchestrator_ptr_ = go.get();

            // Re-assign runner_ so parity test infrastructure uses GlobalOrchestrator
            runner_ = std::move(go);

            LOG_INFO("[Parity] GlobalOrchestrator setup complete (rank " << rank
                     << "/" << world_size << ")");

            // Step 5: Also create GlobalTPContext for infrastructure tests
            // (allreduce, broadcast, barrier verification).
            if (cfg().is_cross_rank_tp())
            {
                std::vector<int> world_ranks;
                for (int r = 0; r < world_size; ++r)
                    world_ranks.push_back(r);

                global_tp_ctx_ = GlobalTPContext::createForTest(
                    MPI_COMM_WORLD,
                    0, // domain_id
                    world_ranks);

                if (!global_tp_ctx_)
                {
                    LOG_ERROR("[Parity] Failed to create GlobalTPContext");
                    return false;
                }

                LOG_INFO("[Parity] GlobalTP context: degree=" << global_tp_ctx_->degree()
                                                              << ", myIndex=" << global_tp_ctx_->myIndex());
            }

            return true;
        }

    protected:
        // PP-specific storage (production DeviceGraphOrchestrator for unified PP)
        std::unique_ptr<DeviceGraphOrchestrator> pp_orchestrator_;

        // Non-owning pointer to GlobalOrchestrator (owned by runner_)
        GlobalOrchestrator *global_orchestrator_ptr_ = nullptr;

        // GlobalTP-specific storage (for infrastructure tests)
        std::unique_ptr<GlobalTPContext> global_tp_ctx_;
    };

} // namespace llaminar2::test::parity::qwen2
