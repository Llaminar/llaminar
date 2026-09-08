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
#include "models/qwen/QwenStandardGraph.h"
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
                // Apply proportional layer split weights if specified
                if (!cfg.pp_weights.empty())
                {
                    tree.root.tp_weights = cfg.pp_weights;
                }
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

            uint64_t moeRuntimeMovementEpoch() const override
            {
                uint64_t epoch = 0;
                for (const auto &runner : stage_runners_)
                {
                    if (runner)
                    {
                        epoch = std::max(epoch, runner->moeRuntimeMovementEpoch());
                    }
                }
                return epoch;
            }

            // ================================================================
            // Snapshot API — aggregate from all stages
            // ================================================================

            void enableSnapshotCapture(const std::string &output_dir = "") override
            {
                for (auto &runner : stage_runners_)
                    runner->enableSnapshotCapture(output_dir);
            }

            void setSnapshotCaptureFilter(const std::vector<std::string> &keys) override
            {
                for (auto &runner : stage_runners_)
                    runner->setSnapshotCaptureFilter(keys);
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
         * @brief Whether multi-device routed execution requires ExpertOverlay.
         *
         * Dense fixtures leave this disabled. MoE fixtures override it so a
         * simple homogeneous LocalTP declaration enters through the same
         * OrchestrationRunner normalization used by the server instead of the
         * retired tree-compiler residency path.
         */
        virtual bool requiresUniversalMoEAuthority() const
        {
            return false;
        }

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
            config_.mtp_kl_threshold = thresholds.mtp_kl_threshold;
            config_.excluded_stages = thresholds.excluded_stages;
            config_.allreduce_stages = thresholds.allreduce_stages;
            config_.min_top1_accuracy = thresholds.min_top1_accuracy;
            config_.min_top5_accuracy = thresholds.min_top5_accuracy;
            config_.min_decode_pass_rate = thresholds.min_decode_pass_rate;
            config_.pytorch_top1_in_topk = thresholds.pytorch_top1_in_topk;

            // Apply model path/snapshot dir overrides from TestConfig if available
            applyModelOverrides();

            ParityTestBase::SetUp();
            if (this->HasFatalFailure() ||
                !std::filesystem::exists(config_.model_path))
            {
                return;
            }

            auto reference_tokens = readPrefillTokensFromMetadata();
            if (productionParityCampaignEnabled())
            {
                ASSERT_FALSE(reference_tokens.empty())
                    << "Production parity reference metadata has no token_ids";
            }
            if (!reference_tokens.empty())
                config_.token_ids = std::move(reference_tokens);
            configureExactProductionParityPrefillGraphBucket();
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
         * @brief Project the typed model's optimized recursive checkpoint cut.
         * @return Explicit model surface, or empty for reference-pack discovery.
         */
        std::vector<std::string>
        productionParityDeclaredMTPCheckpointSurface() const override
        {
            if constexpr (requires(const Derived &fixture)
                          {
                              fixture.modelParityCase()
                                  .model.mtp_checkpoint_surface;
                          })
            {
                return static_cast<const Derived *>(this)
                    ->modelParityCase()
                    .model.mtp_checkpoint_surface;
            }
            return {};
        }

        /**
         * @brief One bounded non-overlay model authority retained by a process campaign.
         *
         * The slot is template-local, so unrelated fixtures/topologies never
         * share mutable loader state.  It deliberately excludes KV precision
         * from its key: KV storage belongs to the runner, while model tensors
         * and their additive PreparedWeightStore remain invariant across those
         * runner policies.  A key change evicts the old context before loading
         * another model, bounding host and device weight residency. ExpertOverlay
         * does not use this slot: only an initialized production runner can emit
         * the typed placement certificate required to reuse routed weights.
         */
        struct CampaignModelContextSlot
        {
            std::string key;
            std::shared_ptr<ModelContext> context;
        };

        static inline std::mutex campaign_model_context_mutex_;
        static inline CampaignModelContextSlot campaign_model_context_slot_;

        bool productionParityProcessCampaignEnabled() const
        {
            return DebugEnv::isTruthyEnv(
                "LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN");
        }

        std::string productionParityModelContextKey(
            WeightDistributionStrategy strategy) const
        {
            /*
             * This key describes only the immutable model and prepared-weight
             * authority.  Durable-rebalance mode, LLEP planner scalars, and
             * routed-row assignment policies belong to the newly constructed
             * runner/graph for each cell; including them here forced a complete
             * GGUF reload even when physical expert ownership was unchanged.
             * Owner order, participants, tier capacity, device topology, and
             * raw-weight lifetime remain keyed because they can change which
             * prepared payloads the bounded context must own.
             */
            std::ostringstream key;
            key << config_.model_path
                << "|strategy=" << static_cast<int>(strategy)
                << "|parallelism=" << static_cast<int>(cfg().parallelism)
                << "|collective=" << static_cast<int>(cfg().collective)
                << "|tp_collective=" << static_cast<int>(cfg().tp_collective)
                << "|activation=" << static_cast<int>(cfg().activation_precision)
                << "|routed_compute="
                << static_cast<int>(cfg().routed_expert_compute_policy)
                << "|routed_owner_order="
                << static_cast<int>(cfg().routed_expert_owner_order)
                << "|moe_release_raw="
                << (cfg().moe_rebalance.release_raw_expert_weights ? 1 : 0)
                << "|mpi_ranks=" << cfg().mpi_ranks
                << "|devices=";
            for (const auto device : cfg().devices)
                key << static_cast<int>(device) << ',';
            key << "|pp_stage_sizes=";
            for (const int size : cfg().pp_stage_sizes)
                key << size << ',';
            key << "|pp_weights=";
            for (const float weight : cfg().pp_weights)
                key << std::setprecision(9) << weight << ',';
            if (cfg().moe_routed_expert_plan)
            {
                const auto &plan = *cfg().moe_routed_expert_plan;
                key << "|moe_plan_enabled=" << (plan.enabled ? 1 : 0)
                    << "|moe_plan_topology="
                    << static_cast<int>(plan.topology)
                    << "|moe_plan_owner_order="
                    << static_cast<int>(plan.owner_order);
                for (const auto &domain : plan.domains)
                {
                    key << "|moe_domain=" << domain.name << ':'
                        << static_cast<int>(domain.scope) << ':'
                        << static_cast<int>(domain.routed_compute_policy) << ':'
                        << static_cast<int>(domain.routed_phase_policy);
                    for (const auto &participant : domain.participants)
                        key << ':' << participant.toString();
                }
                for (const auto &tier : plan.routed_tiers)
                {
                    key << "|moe_tier=" << tier.name << ':' << tier.domain
                        << ':' << tier.priority << ':'
                        << tier.max_experts_per_layer << ':'
                        << tier.memory_budget_bytes << ':'
                        << (tier.fallback ? 1 : 0);
                }
            }
            return key.str();
        }

        /**
         * @brief Reuse immutable model/prepared weights across KV-policy cells.
         *
         * Exact runner, arena, graph, stream, and request state are rebuilt for
         * every cell.  Only the model-owned weight authority is retained, and a
         * changed model/topology key is evicted synchronously rather than kept
         * as an unbounded second resident model.
         */
        std::shared_ptr<ModelContext> acquireParityModelContext(
            WeightDistributionStrategy strategy) override
        {
            production_parity_model_context_reused_ = false;
            if (!productionParityProcessCampaignEnabled())
            {
                return Qwen2ParityTestBase::acquireParityModelContext(strategy);
            }

            const std::string key = productionParityModelContextKey(strategy);
            std::lock_guard<std::mutex> lock(campaign_model_context_mutex_);
            auto &slot = campaign_model_context_slot_;
            if (slot.context && slot.key == key)
            {
                production_parity_model_context_reused_ = true;
                LOG_INFO("[ProductionParity] Reusing model-owned prepared weights for "
                         << cfg().name);
                return slot.context;
            }

            if (slot.context)
            {
                // The preceding fixture teardown synchronized every device and
                // retired its runner.  Clear tensor-indexed process caches while
                // the old context is still alive, then release that sole slot.
                llaminar::v2::kernels::KernelFactory::clearCache();
                slot.context.reset();
                slot.key.clear();
            }

            auto context =
                Qwen2ParityTestBase::acquireParityModelContext(strategy);
            if (context)
            {
                slot.key = key;
                slot.context = context;
            }
            return context;
        }

        bool preserveParityPipelineCachesBetweenTests() const override
        {
            return productionParityProcessCampaignEnabled();
        }

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
            if (!cfg().prompt.empty())
                config_.prompt = cfg().prompt;
            if (!cfg().token_ids.empty())
                config_.token_ids = cfg().token_ids;
            if (cfg().decode_steps > 0)
                config_.decode_steps = cfg().decode_steps;
            config_.collective_evidence_source =
                cfg().collective_evidence_source.value_or(
                    cfg().is_local_tp()
                        ? ParityCollectiveEvidenceSource::
                              PostCollectiveSnapshot
                        : ParityCollectiveEvidenceSource::
                              CrossRankPartials);
            config_.uses_cross_rank_pipeline = cfg().is_cross_rank_pp();
            config_.moe_rebalance_exercise = cfg().moe_rebalance_exercise;
            config_.moe_movement_expectation =
                cfg().moe_movement_expectation;
            config_.mtp_expectation = cfg().mtp_expectation;
            config_.mtp_expected_draft_depth =
                cfg().mtp_expected_draft_depth;
            config_.mtp_expected_graph_capacity =
                cfg().mtp_expected_graph_capacity;
            config_.mtp_recursive_aggregate_cosine_floor =
                cfg().mtp_recursive_aggregate_cosine_floor;
            config_.graph_snapshot_policy = cfg().graph_snapshot_policy;
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
                if (productionParityCampaignEnabled())
                {
                    FAIL() << "Production parity prerequisite failed for "
                           << cfg().name << ": " << *skip_reason;
                }
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
                    if (productionParityCampaignEnabled())
                    {
                        FAIL() << "Production LOCAL TP/PP campaign requires -np 1 (got "
                               << world_size << ")";
                    }
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
                    if (productionParityCampaignEnabled())
                    {
                        FAIL() << "Production cross-rank campaign requires "
                               << cfg().mpi_ranks << " MPI ranks (got "
                               << world_size << ")";
                    }
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

        /**
         * @brief Require the first authenticated request to seed the cache.
         *
         * @param state Deep runtime snapshot captured after fresh prefill.
         * @param checkpoint_passed Whether the full Hugging Face prefill proof passed.
         * @param checkpoint_cosine Fresh prefill LM-head cosine for CSV evidence.
         */
        void assertFreshPrefixRestoreSeed(
            const PrefixRuntimeStateSnapshot &state,
            bool checkpoint_passed,
            float checkpoint_cosine)
        {
            assertProductionParityFreshPrefixSeed(
                state,
                checkpoint_passed,
                checkpoint_cosine);
        }

        /**
         * @brief Prove the decode prefill restored the complete fresh state.
         *
         * @param fresh_state Byte-hashed state after the initial prefill.
         * @param checkpoint_passed Whether restored-state decode parity passed.
         * @param checkpoint_cosine Mean restored-state decode cosine.
         */
        void assertFullPrefixRestore(
            const PrefixRuntimeStateSnapshot &fresh_state,
            bool checkpoint_passed,
            float checkpoint_cosine)
        {
            assertProductionParityCompletePrefixRestore(
                fresh_state,
                checkpoint_passed,
                checkpoint_cosine);
        }

        /**
         * @brief Prove a cached prompt plus one authenticated token is partial.
         *
         * The ordinary decode comparison has already certified every stage for
         * the same prompt-plus-token state against Hugging Face. This replay
         * checks that suffix prefill reaches the byte-identical KV/terminal
         * state and independently compares its live LM-head checkpoint with the
         * authenticated decode-step-zero tensor.
         */
        void assertPartialPrefixRestore()
        {
            assertProductionParityPartialPrefixRestore();
        }

        /**
         * @brief Run the complete real-weight parity contract in one runner session.
         *
         * The former PrefillParity, DecodeParity, and SnapshotInfrastructure
         * cases each rebuilt the same model and production runner.  This campaign
         * retains every numerical comparison and CSV, validates snapshot
         * publication after the actual prefill, resets request data without
         * rebuilding topology, then validates incremental decode.  LocalTP uses
         * its shard-aware comparison path; all other placements use the semantic
         * snapshot path, including cross-rank reductions.
         */
        void runProductionParityCampaign()
        {
            beginProductionParityEvidence();
            ASSERT_TRUE(setupProductionParityPipeline())
                << "Production parity pipeline setup failed";
            ASSERT_TRUE(hasOrchestrationRunner())
                << "Production parity must enter through the server-facing runner";

            const auto prefill = runPrefillParity();
            assertParity(prefill);

            const PrefixRuntimeStateSnapshot fresh_prefix_state =
                activePrefixStateProbe();
            assertFreshPrefixRestoreSeed(
                fresh_prefix_state,
                prefill.overall_passed,
                prefill.lm_head_cosine);

            if (config_.moe_rebalance_exercise.request_after_prefill)
            {
                ASSERT_TRUE(driveParityMoERebalanceMaintenance(
                    "prefill",
                    static_cast<std::uint64_t>(config_.token_ids.size())))
                    << "Production ExpertOverlay rejected post-prefill maintenance";
            }

            assertProductionParitySnapshotInfrastructure();

            // Reset only request-owned data.  The runner, prepared weights,
            // arenas, graph cache, and exact stream bindings remain authoritative.
            activeClearSnapshots();
            activeClearCache();

            DecodeParitySummary decode = runDecodeParity(
                ParityDecodePrefillMode::CompletePrefixRestore);

            if (decode.steps_total == 0)
            {
                ADD_FAILURE()
                    << "Production parity requires authenticated incremental-decode references";
            }
            else
            {
                assertDecodeParity(decode);
            }

            assertFullPrefixRestore(
                fresh_prefix_state,
                decode.overall_passed,
                decode.avg_cosine);
            assertPartialPrefixRestore();

            finishProductionParityEvidence();
        }

        // ==========================================================================
        // Pipeline Setup — Tree-based (dogfooding ParallelismTree + Compiler)
        // ==========================================================================

        /**
         * @brief Copy declarative precision and MoE policy into a runner config.
         *
         * Cross-rank parity previously stopped at activation/KV precision, so a
         * test could name random ownership, LLEP, or dynamic maintenance while
         * the production factory still consumed its defaults.  Keep this one
         * typed boundary shared by tree-compiled and MPI runners.
         */
        void applyDeclarativeRunnerConfig(InferenceRunnerConfig &runner_config)
        {
            runner_config.activation_precision = cfg().activation_precision;
            runner_config.kv_cache_precision = cfg().kv_cache_precision;
            runner_config.tp_allreduce_precision_override =
                cfg().tp_allreduce_precision_override;
            runner_config.routed_expert_compute_policy =
                cfg().routed_expert_compute_policy;
            runner_config.routed_expert_owner_order =
                cfg().routed_expert_owner_order;
            runner_config.moe_hot_expert_cache = cfg().moe_hot_expert_cache;
            runner_config.moe_routed_prefill = cfg().moe_routed_prefill;
            runner_config.moe_rebalance = cfg().moe_rebalance;
            runner_config.moe_routed_expert_plan = cfg().moe_routed_expert_plan;
            runner_config.moe_expert_overlay_mpi_ctx = mpi_ctx_;
        }

        static std::string orchestrationActivationPrecisionValue(
            ActivationPrecision precision)
        {
            switch (precision)
            {
            case ActivationPrecision::FP32:
                return "fp32";
            case ActivationPrecision::BF16:
                return "bf16";
            case ActivationPrecision::FP16:
                return "fp16";
            case ActivationPrecision::Q8_1:
                return "q8_1";
            case ActivationPrecision::Q16_1:
                return "q16_1";
            case ActivationPrecision::Hybrid:
                return "hybrid";
            case ActivationPrecision::HybridQ16:
                return "hybridq16";
            case ActivationPrecision::TQ4:
            case ActivationPrecision::TQ8:
                throw std::invalid_argument(
                    "TurboQuant is a KV/cache format, not a supported activation precision");
            }
            throw std::invalid_argument("Unknown activation precision");
        }

        static std::string orchestrationKVCachePrecisionValue(
            KVCachePrecision precision)
        {
            switch (precision)
            {
            case KVCachePrecision::AUTO:
                return "auto";
            case KVCachePrecision::FP32:
                return "fp32";
            case KVCachePrecision::FP16:
                return "fp16";
            case KVCachePrecision::Q8_1:
                return "q8_1";
            case KVCachePrecision::Q16_1:
                return "q16_1";
            case KVCachePrecision::TQ4:
                return "tq4";
            case KVCachePrecision::TQ:
                return "tq";
            }
            throw std::invalid_argument("Unknown KV-cache precision");
        }

        /**
         * @brief Project the canonical production-campaign policy into a runner.
         *
         * Prefix restore is mandatory for every production parity cell and is
         * therefore installed even while older suites are being migrated to
         * the typed definition adapter. A typed case, when present, remains
         * the sole authority for precision, MTP, and ExpertOverlay policy.
         *
         * @param config Mutable production configuration immediately before
         *        runner construction.
         */
        void applyCanonicalProductionRuntimePolicy(
            OrchestrationConfig &config) const
        {
            applyProductionParityPrefixRestorePolicy(config);

            if constexpr (requires(
                              const Derived &fixture,
                              OrchestrationConfig &runtime)
                          {
                              fixture.modelParityCase().applyRuntimePolicy(
                                  runtime);
                          })
            {
                static_cast<const Derived *>(this)
                    ->modelParityCase()
                    .applyRuntimePolicy(config);
            }
        }

        /** @brief Install deterministic greedy sampling on the active production runner. */
        void installProductionParitySampling()
        {
            ASSERT_NE(orch_runner_.get(), nullptr);
            SamplingParams greedy;
            greedy.temperature = 0.0f;
            greedy.top_k = 1;
            greedy.top_p = 1.0f;
            greedy.seed = 1;
            orch_runner_->setSamplingParams(greedy);
        }

        /**
         * @brief Enter production through one generated typed model-parity case.
         *
         * The case owns model, topology, precision, prefix, MTP, and routed
         * policy.  The fixture contributes only the resolved/staged GGUF path
         * and current MPI rank, then enables the same greedy request policy
         * used by every reference comparison.
         */
        bool setupTypedProductionOrchestrationPipeline()
        {
            if constexpr (requires(const Derived &fixture)
                          {
                              fixture.modelParityCase().makeOrchestrationConfig(
                                  std::string{}, 0);
                          })
            {
                const auto &test_case =
                    static_cast<const Derived *>(this)->modelParityCase();
                OrchestrationConfig config =
                    test_case.makeOrchestrationConfig(
                        config_.model_path, mpiRank());

                /*
                 * The typed definition owns the production prefix policy,
                 * while the campaign harness owns artifact isolation. Apply
                 * the latter after the typed projection so ordinary service
                 * archives can never turn another cell's mandatory fresh
                 * request into a restore hit. The rank suffix also prevents
                 * TP participants with different KV shards from sharing one
                 * durable archive.
                 */
                applyProductionParityPrefixRestorePolicy(config);
                config.deterministic = requiresUniversalMoEAuthority();
                if (!setupOrchestrationRunner(config))
                    return false;
                installProductionParitySampling();
                return true;
            }
            else
            {
                LOG_ERROR(
                    "[Parity] Typed production setup requested by a fixture without ModelParityCase");
                return false;
            }
        }

        /** @return Legacy participant addresses without reconstructing topology elsewhere. */
        std::vector<GlobalDeviceAddress> legacyParticipantAddresses(
            bool cross_rank) const
        {
            std::vector<GlobalDeviceAddress> addresses;
            addresses.reserve(cfg().devices.size());
            int cuda_ordinal = 0;
            int rocm_ordinal = 0;
            int cpu_ordinal = 0;
            for (const auto device : cfg().devices)
            {
                switch (device)
                {
                case ParityDeviceType::CPU:
                    addresses.push_back(
                        cross_rank
                            ? GlobalDeviceAddress::cpu(cpu_ordinal++)
                            : GlobalDeviceAddress::cpu());
                    break;
                case ParityDeviceType::CUDA:
                    addresses.push_back(
                        GlobalDeviceAddress::cuda(cuda_ordinal++));
                    break;
                case ParityDeviceType::ROCm:
                    addresses.push_back(
                        GlobalDeviceAddress::rocm(rocm_ordinal++));
                    break;
                }
            }
            return addresses;
        }

        /**
         * @brief Build legacy-declared local PP through the production runner.
         *
         * This compatibility bridge is intentionally confined to declarations
         * not yet migrated to `ModelParityCase`. It projects their explicit
         * stage grouping once into named domains; all execution, prefix lookup,
         * graph capture, transfer, and snapshots are still production-owned.
         */
        bool setupLegacyLocalPPProductionOrchestrationPipeline()
        {
            TensorFactory metadata_tensor_factory(*mpi_ctx_);
            ModelLoader metadata_loader(&metadata_tensor_factory);
            try
            {
                metadata_loader.loadModel(config_.model_path);
            }
            catch (const std::exception &error)
            {
                LOG_ERROR(
                    "[Parity] Failed to read model metadata for local PP: "
                    << error.what());
                return false;
            }

            const auto addresses = legacyParticipantAddresses(false);
            const std::vector<int> stage_sizes =
                cfg().pp_stage_sizes.empty()
                    ? std::vector<int>(addresses.size(), 1)
                    : cfg().pp_stage_sizes;
            if (stage_sizes.empty() ||
                std::accumulate(
                    stage_sizes.begin(), stage_sizes.end(), 0) !=
                    static_cast<int>(addresses.size()))
            {
                LOG_ERROR(
                    "[Parity] Local PP stage groups must cover every participant exactly once");
                return false;
            }
            const int layer_count = static_cast<int>(
                metadata_loader.getModel().block_count);
            if (layer_count < static_cast<int>(stage_sizes.size()))
            {
                LOG_ERROR(
                    "[Parity] Local PP has more stages than transformer layers");
                return false;
            }

            std::vector<int> stage_layer_counts(stage_sizes.size(), 0);
            if (cfg().pp_weights.empty())
            {
                const int base = layer_count /
                                 static_cast<int>(stage_sizes.size());
                const int extra = layer_count %
                                  static_cast<int>(stage_sizes.size());
                for (std::size_t stage = 0;
                     stage < stage_sizes.size(); ++stage)
                {
                    stage_layer_counts[stage] = base +
                        (static_cast<int>(stage) < extra ? 1 : 0);
                }
            }
            else
            {
                if (cfg().pp_weights.size() != stage_sizes.size())
                {
                    LOG_ERROR(
                        "[Parity] Local PP weights must match the stage count");
                    return false;
                }
                const float total_weight = std::accumulate(
                    cfg().pp_weights.begin(), cfg().pp_weights.end(), 0.0f);
                if (!(total_weight > 0.0f))
                {
                    LOG_ERROR("[Parity] Local PP weights must sum positive");
                    return false;
                }
                int assigned = 0;
                for (std::size_t stage = 0;
                     stage < stage_sizes.size(); ++stage)
                {
                    if (stage + 1u == stage_sizes.size())
                    {
                        stage_layer_counts[stage] = layer_count - assigned;
                        break;
                    }
                    int count = std::max(
                        1,
                        static_cast<int>(
                            cfg().pp_weights[stage] / total_weight *
                                layer_count +
                            0.5f));
                    count = std::min(
                        count,
                        layer_count - assigned -
                            static_cast<int>(stage_sizes.size() - stage - 1u));
                    stage_layer_counts[stage] = count;
                    assigned += count;
                }
            }

            OrchestrationConfig config = OrchestrationConfig::defaults();
            config.model_path = config_.model_path;
            config.max_seq_len = 4096;
            config.batch_size = 1;
            config.tp_degree = 1;
            config.pp_degree = static_cast<int>(stage_sizes.size());
            config.pp_split = PPSplitMode::MANUAL;
            config.device_mode = DeviceAssignmentMode::AUTO;
            config.deterministic = false;
            config.tp_allreduce_precision_override =
                cfg().tp_allreduce_precision_override;

            std::size_t participant_offset = 0u;
            int layer_offset = 0;
            for (std::size_t stage = 0; stage < stage_sizes.size(); ++stage)
            {
                DomainDefinition domain;
                domain.name = "parity_stage_" + std::to_string(stage);
                domain.scope = TPScope::RANK_LOCAL;
                domain.owner_rank = mpiRank();
                for (int member = 0; member < stage_sizes[stage]; ++member)
                {
                    domain.devices.push_back(
                        addresses.at(participant_offset++));
                }
                if (domain.devices.size() > 1u)
                {
                    domain.backend =
                        toCollectiveBackend(cfg().tp_collective);
                    domain.weights.assign(
                        domain.devices.size(),
                        1.0f / static_cast<float>(domain.devices.size()));
                }
                config.domain_definitions.push_back(std::move(domain));
                config.pp_stage_definitions.push_back(PPStageDefinition{
                    .stage_id = static_cast<int>(stage),
                    .domain_name =
                        "parity_stage_" + std::to_string(stage),
                    .first_layer = layer_offset,
                    .last_layer =
                        layer_offset + stage_layer_counts[stage] - 1,
                });
                layer_offset += stage_layer_counts[stage];
            }
            applyCanonicalProductionRuntimePolicy(config);
            if (!setupOrchestrationRunner(config))
                return false;
            installProductionParitySampling();
            return true;
        }

        /**
         * @brief Build legacy cross-rank TP/PP through the serving runner.
         *
         * Every rank receives one explicit primary participant. Cross-rank TP
         * selects node/global scope while PP retains the production equal-split
         * planner. This removes the test-owned GlobalOrchestrator lifecycle
         * from canonical campaigns without changing focused collective tests.
         */
        bool setupLegacyCrossRankProductionOrchestrationPipeline()
        {
            const auto addresses = legacyParticipantAddresses(true);
            if (addresses.size() !=
                static_cast<std::size_t>(cfg().mpi_ranks))
            {
                LOG_ERROR(
                    "[Parity] Cross-rank production topology requires exactly one participant per MPI rank");
                return false;
            }

            OrchestrationConfig config = OrchestrationConfig::defaults();
            config.model_path = config_.model_path;
            config.max_seq_len = 4096;
            config.batch_size = 1;
            config.device_mode = DeviceAssignmentMode::EXPLICIT;
            config.default_backend = toCollectiveBackend(cfg().collective);
            config.tp_allreduce_precision_override =
                cfg().tp_allreduce_precision_override;
            config.deterministic = false;
            for (int rank = 0; rank < cfg().mpi_ranks; ++rank)
            {
                const auto &address =
                    addresses.at(static_cast<std::size_t>(rank));
                config.device_map.emplace_back(rank, address);
                config.device_map_numa_explicit.emplace_back(
                    rank, address.hasValidNuma());
            }

            if (cfg().is_cross_rank_tp())
            {
                config.tp_degree = cfg().mpi_ranks;
                config.tp_scope = cfg().is_node_tp()
                                      ? TPScope::NODE_LOCAL
                                      : TPScope::GLOBAL;
                config.pp_degree = 1;
                config.shard_weights = true;
            }
            else if (cfg().is_cross_rank_pp())
            {
                config.tp_degree = 1;
                config.pp_degree = cfg().mpi_ranks;
                config.pp_split = PPSplitMode::EQUAL;
            }
            else
            {
                LOG_ERROR(
                    "[Parity] Cross-rank production setup received a rank-local topology");
                return false;
            }
            applyCanonicalProductionRuntimePolicy(config);
            if (!setupOrchestrationRunner(config))
                return false;
            installProductionParitySampling();
            return true;
        }

        /**
         * @brief Select the one production runner lifecycle for a campaign cell.
         *
         * Typed cases are authoritative. Legacy declarations enter explicit
         * compatibility projections only until their source records are
         * migrated; no production campaign falls back to a direct runner tree.
         */
        bool setupProductionParityPipeline()
        {
            if constexpr (requires(const Derived &fixture)
                          {
                              fixture.modelParityCase().makeOrchestrationConfig(
                                  std::string{}, 0);
                          })
            {
                return setupTypedProductionOrchestrationPipeline();
            }

            if (cfg().moe_routed_expert_plan &&
                cfg().moe_routed_expert_plan->usesExpertOverlayAuthority())
            {
                return setupExpertOverlayMoEOrchestrationPipeline();
            }
            if (requiresUniversalMoEAuthority() && cfg().is_local_tp())
                return setupImplicitLocalTPMoEOrchestrationPipeline();
            if (cfg().moe_rebalance.mode ==
                MoERebalanceRuntimeMode::Dynamic)
            {
                LOG_ERROR(
                    "[Parity] Dynamic MoE production parity requires an ExpertOverlay authority");
                return false;
            }
            if (cfg().is_cross_rank())
                return setupLegacyCrossRankProductionOrchestrationPipeline();
            if (cfg().is_single_device())
                return setupSingleDeviceOrchestrationPipeline();
            if (cfg().is_local_tp())
                return setupRankLocalTPOrchestrationPipeline();
            if (cfg().is_local_pp())
                return setupLegacyLocalPPProductionOrchestrationPipeline();
            LOG_ERROR("[Parity] Unsupported production parity topology");
            return false;
        }

        /**
         * @brief Build one single-device runner through the server-facing API.
         *
         * The legacy direct `IInferenceRunner::forward()` surface cannot own a
         * cross-request prefix lookup. Canonical parity therefore enters the
         * same `OrchestrationRunner::prefill()` lifecycle used by HTTP serving,
         * while retaining graph snapshots and exact precision policy.
         *
         * @return True after the production runner is initialized.
         */
        bool setupSingleDeviceOrchestrationPipeline()
        {
            OrchestrationConfig config = OrchestrationConfig::defaults();
            config.model_path = config_.model_path;
            config.max_seq_len = 4096;
            config.batch_size = 1;
            config.activation_precision =
                orchestrationActivationPrecisionValue(
                    cfg().activation_precision);
            config.kv_cache_precision =
                orchestrationKVCachePrecisionValue(
                    cfg().kv_cache_precision);
            config.tp_allreduce_precision_override =
                cfg().tp_allreduce_precision_override;
            config.device_mode = DeviceAssignmentMode::AUTO;
            config.device_for_this_rank =
                GlobalDeviceAddress::fromLocalDeviceId(getDevice());
            config.tp_degree = 1;
            config.pp_degree = 1;
            config.deterministic = false;
            applyCanonicalProductionRuntimePolicy(config);

            if (!setupOrchestrationRunner(config))
                return false;

            SamplingParams greedy;
            greedy.temperature = 0.0f;
            greedy.top_k = 1;
            greedy.top_p = 1.0f;
            greedy.seed = 1;
            orch_runner_->setSamplingParams(greedy);
            return true;
        }

        /**
         * @brief Build rank-local tensor parallelism through the serving API.
         *
         * The former tree-compiled test runner had no cross-request prefix
         * authority. OrchestrationRunner owns the production TP graph,
         * collective, prefix lifecycle, and semantic post-collective snapshot
         * surface used by serving.
         *
         * @return True after the production TP runner is initialized.
         */
        bool setupRankLocalTPOrchestrationPipeline()
        {
            OrchestrationConfig config = OrchestrationConfig::defaults();
            config.model_path = config_.model_path;
            config.max_seq_len = 4096;
            config.batch_size = 1;
            config.activation_precision =
                orchestrationActivationPrecisionValue(
                    cfg().activation_precision);
            config.kv_cache_precision =
                orchestrationKVCachePrecisionValue(
                    cfg().kv_cache_precision);
            config.tp_allreduce_precision_override =
                cfg().tp_allreduce_precision_override;
            config.device_mode = DeviceAssignmentMode::AUTO;
            config.tp_scope = TPScope::RANK_LOCAL;
            config.default_backend =
                toCollectiveBackend(cfg().collective);
            config.pp_degree = 1;
            config.deterministic = false;

            if constexpr (requires(const Derived &fixture)
                          { fixture.modelParityCase(); })
            {
                for (const auto &participant :
                     static_cast<const Derived *>(this)
                         ->modelParityCase()
                         .topology.participants)
                {
                    if (!participant.world_rank ||
                        *participant.world_rank == mpiRank())
                    {
                        config.tp_devices.push_back(participant.address);
                    }
                }
            }
            else
            {
                int cuda_ordinal = 0;
                int rocm_ordinal = 0;
                for (const auto device : cfg().devices)
                {
                    switch (device)
                    {
                    case ParityDeviceType::CPU:
                        config.tp_devices.push_back(
                            GlobalDeviceAddress::cpu());
                        break;
                    case ParityDeviceType::CUDA:
                        config.tp_devices.push_back(
                            GlobalDeviceAddress::cuda(cuda_ordinal++));
                        break;
                    case ParityDeviceType::ROCm:
                        config.tp_devices.push_back(
                            GlobalDeviceAddress::rocm(rocm_ordinal++));
                        break;
                    }
                }
            }
            if (config.tp_devices.size() < 2u)
            {
                LOG_ERROR(
                    "[Parity] Rank-local TP topology resolved fewer than two participants");
                return false;
            }
            config.tp_degree =
                static_cast<int>(config.tp_devices.size());
            applyCanonicalProductionRuntimePolicy(config);

            if (!setupOrchestrationRunner(config))
                return false;

            SamplingParams greedy;
            greedy.temperature = 0.0f;
            greedy.top_k = 1;
            greedy.top_p = 1.0f;
            greedy.seed = 1;
            orch_runner_->setSamplingParams(greedy);
            return true;
        }

        /**
         * @brief Build the production runner for one authoritative ExpertOverlay plan.
         *
         * Every enabled routed-expert placement plan selects ExpertOverlay,
         * including a homogeneous one-tier plan. Static ownership, current-
         * batch LLEP, and durable Dynamic maintenance must therefore enter the
         * same OrchestrationRunner lifecycle. The named continuation domain is
         * the only dense/routed placement authority; inventory binding resolves
         * its ranks and devices without a second device map or ordinary TP
         * configuration that could drift from it.
         *
         * @return true when the production runner initialized successfully.
         */
        bool setupExpertOverlayMoEOrchestrationPipeline()
        {
            if (!cfg().is_cross_rank_tp())
            {
                LOG_ERROR("[Parity] This ExpertOverlay parity pipeline requires "
                          "a cross-rank TP continuation domain");
                return false;
            }
            if (!cfg().moe_routed_expert_plan ||
                !cfg().moe_routed_expert_plan->usesExpertOverlayAuthority())
            {
                LOG_ERROR("[Parity] ExpertOverlay parity requires one enabled typed "
                          "routed-expert placement plan");
                return false;
            }

            const auto &plan = *cfg().moe_routed_expert_plan;
            const std::string continuation_domain =
                plan.continuation_domain.empty()
                    ? plan.effectiveBaseModelDomain()
                    : plan.continuation_domain;
            const auto domain_it = std::find_if(
                plan.domains.begin(),
                plan.domains.end(),
                [&](const RoutedExpertDomain &domain)
                {
                    return domain.name == continuation_domain;
                });
            if (domain_it == plan.domains.end() ||
                static_cast<int>(domain_it->participants.size()) !=
                    cfg().mpi_ranks)
            {
                LOG_ERROR("[Parity] ExpertOverlay continuation domain must declare "
                          "exactly one participant per MPI rank");
                return false;
            }

            OrchestrationConfig config = OrchestrationConfig::defaults();
            config.model_path = config_.model_path;
            config.max_seq_len = 4096;
            config.batch_size = 1;
            config.activation_precision =
                orchestrationActivationPrecisionValue(
                    cfg().activation_precision);
            config.kv_cache_precision =
                orchestrationKVCachePrecisionValue(
                    cfg().kv_cache_precision);
            config.tp_allreduce_precision_override =
                cfg().tp_allreduce_precision_override;
            // The named continuation domain carries its own typed dense-TP
            // policy. Keeping ordinary TP at degree one prevents a second
            // topology authority from being synthesized beside ExpertOverlay.
            config.tp_degree = 1;
            config.pp_degree = 1;
            config.default_backend =
                toCollectiveBackend(cfg().collective);
            config.device_mode = DeviceAssignmentMode::AUTO;
            config.deterministic = true;
            config.routed_expert_compute_policy =
                cfg().routed_expert_compute_policy;
            config.routed_expert_owner_order =
                cfg().routed_expert_owner_order;
            config.moe_hot_expert_cache = cfg().moe_hot_expert_cache;
            config.moe_routed_prefill = cfg().moe_routed_prefill;
            config.moe_rebalance = cfg().moe_rebalance;
            config.moe_routed_expert_plan = cfg().moe_routed_expert_plan;
            applyCanonicalProductionRuntimePolicy(config);

            /*
             * Let the production runner own model loading, plan resolution,
             * model-aware tier freezing, and prepared-weight certification as
             * one transition. A bare context from the generic campaign cache
             * cannot prove any of those facts. Genuine reuse must enter through
             * setupOrchestrationRunner(config, ModelContextReuseContract).
             */
            production_parity_model_context_reused_ = false;
            if (!setupOrchestrationRunner(config))
                return false;

            SamplingParams greedy;
            greedy.temperature = 0.0f;
            greedy.top_k = 1;
            greedy.top_p = 1.0f;
            greedy.seed = 1;
            orch_runner_->setSamplingParams(greedy);
            return true;
        }

        /**
         * @brief Build homogeneous rank-local MoE TP through production setup.
         *
         * The public configuration names only ordinary LocalTP intent. During
         * initialization OrchestrationRunner resolves the devices, synthesizes
         * the canonical one-domain/one-tier ExpertOverlay authority, freezes
         * capacity, and builds the same graph family used by the server. This
         * keeps the parity fixture unaware of implicit tier implementation
         * details while ensuring that no legacy MoE controller survives beside
         * the universal authority.
         *
         * @return true after the production runner and snapshot graph are ready.
         */
        bool setupImplicitLocalTPMoEOrchestrationPipeline()
        {
            if (!cfg().is_local_tp() || cfg().device_count() < 2)
            {
                LOG_ERROR("[Parity] Implicit LocalTP MoE authority requires at "
                          "least two rank-local participants");
                return false;
            }

            std::vector<GlobalDeviceAddress> devices;
            devices.reserve(cfg().devices.size());
            int cuda_index = 0;
            int rocm_index = 0;
            for (const auto device : cfg().devices)
            {
                switch (device)
                {
                case ParityDeviceType::CPU:
                    devices.push_back(GlobalDeviceAddress::cpu());
                    break;
                case ParityDeviceType::CUDA:
                    devices.push_back(
                        GlobalDeviceAddress::cuda(cuda_index++));
                    break;
                case ParityDeviceType::ROCm:
                    devices.push_back(
                        GlobalDeviceAddress::rocm(rocm_index++));
                    break;
                }
            }

            OrchestrationConfig config = OrchestrationConfig::defaults();
            config.model_path = config_.model_path;
            config.max_seq_len = 4096;
            config.batch_size = 1;
            config.activation_precision =
                orchestrationActivationPrecisionValue(
                    cfg().activation_precision);
            config.kv_cache_precision =
                orchestrationKVCachePrecisionValue(
                    cfg().kv_cache_precision);
            config.tp_allreduce_precision_override =
                cfg().tp_allreduce_precision_override;
            config.tp_degree = static_cast<int>(devices.size());
            config.tp_scope = TPScope::RANK_LOCAL;
            config.tp_devices = devices;
            config.tp_weights.assign(
                devices.size(),
                1.0f / static_cast<float>(devices.size()));
            config.pp_degree = 1;
            config.default_backend =
                toCollectiveBackend(cfg().collective);
            config.device_mode = DeviceAssignmentMode::AUTO;
            config.deterministic = true;
            config.routed_expert_compute_policy =
                cfg().routed_expert_compute_policy;
            config.routed_expert_owner_order =
                cfg().routed_expert_owner_order;
            config.moe_hot_expert_cache = cfg().moe_hot_expert_cache;
            config.moe_routed_prefill = cfg().moe_routed_prefill;
            config.moe_rebalance = cfg().moe_rebalance;
            applyCanonicalProductionRuntimePolicy(config);

            /*
             * The implicit authority is synthesized from live model metadata.
             * Production must therefore create the ModelContext only after it
             * has resolved and frozen that plan. Supplying a context prepared
             * before normalization would violate the exact reuse certificate
             * contract and is correctly rejected by OrchestrationRunner.
             */
            if (!setupOrchestrationRunner(config))
                return false;

            SamplingParams greedy;
            greedy.temperature = 0.0f;
            greedy.top_k = 1;
            greedy.top_p = 1.0f;
            greedy.seed = 1;
            orch_runner_->setSamplingParams(greedy);
            return true;
        }

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
            if (cfg().moe_routed_expert_plan &&
                cfg().moe_routed_expert_plan->usesExpertOverlayAuthority())
            {
                return setupExpertOverlayMoEOrchestrationPipeline();
            }

            if (requiresUniversalMoEAuthority() && cfg().is_local_tp())
                return setupImplicitLocalTPMoEOrchestrationPipeline();

            if (cfg().moe_rebalance.mode == MoERebalanceRuntimeMode::Dynamic)
            {
                LOG_ERROR("[Parity] Dynamic MoE requires an enabled ExpertOverlay "
                          "placement plan; the retired ordinary-TP controller is "
                          "not a production fallback");
                return false;
            }

            // Cross-rank TP or PP uses GlobalOrchestrator
            if (cfg().is_cross_rank())
                return setupGlobalOrchestratorPipeline();

            // A production campaign must exercise the server-facing prefix
            // lifecycle rather than the direct runner forward adapter.
            if (cfg().is_single_device())
                return setupSingleDeviceOrchestrationPipeline();
            if (cfg().is_local_tp())
                return setupRankLocalTPOrchestrationPipeline();

            // ============================================================
            // Tree-based path: LocalPP and HybridPP+TP
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
            model_ctx_ = acquireParityModelContext(weight_strategy);

            if (!model_ctx_)
            {
                LOG_ERROR("[Parity/Tree] Failed to load model");
                return false;
            }

            int n_layers = model_ctx_->blockCount();

            // Step 1: Build the parallelism tree from TestConfig
            auto tree = buildTreeFromTestConfig(cfg(), n_layers);

            // Step 2: Build the compile context with real factories
            InferenceRunnerConfig base_runner_config;
            base_runner_config.max_seq_len = 4096;
            base_runner_config.batch_size = 1;
            base_runner_config.force_graph = true;
            applyDeclarativeRunnerConfig(base_runner_config);

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

            model_ctx_ = acquireParityModelContext(
                WeightDistributionStrategy::SHARDED);

            if (!model_ctx_)
            {
                LOG_ERROR("[Parity] Failed to load model");
                return false;
            }

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
            model_ctx_ = acquireParityModelContext(getWeightStrategy());
            if (!model_ctx_)
            {
                LOG_ERROR("[Parity] Failed to load model");
                return false;
            }
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
            applyDeclarativeRunnerConfig(inf_config);

            DeviceId device = getDevice();
            if (device.is_gpu())

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
            go_config.rank_runner = std::move(runner_); // Transfer ownership
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
