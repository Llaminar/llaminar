/**
 * @file Qwen36MoEParityTestBase.h
 * @brief Shared Qwen3.6 MoE parity harness for prefix cache, MTP, and expert-overlay tests.
 *
 * This header owns the integration-test runners that compare Llaminar V2 MoE
 * execution against PyTorch metadata snapshots and against fresh Llaminar
 * baselines.  The helpers deliberately keep request-owned state boundaries
 * visible: KV cache rows, prefix-cache terminal state, MTP sidecar state, GDN
 * recurrence state, and MoE rebalance placement all have distinct lifetimes and
 * must be reset or restored by the first-class owner for that lifetime.
 */

#pragma once

#include "Qwen36DenseParityTestBase.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/HardwareInventory.h"
#include "collective/BackendRouter.h"
#include "collective/LocalTPContext.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"
#include "execution/moe/MoERoutedExpertPlacementPlanner.h"
#include "execution/mtp/MTPDecodeCatchup.h"
#include "execution/mtp/MTPStateTransaction.h"
#include "execution/mtp/MTPSpecStateContract.h"
#include "execution/mtp/MTPSpecTransactionDriver.h"
#include "kernels/KernelFactory.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <cnpy.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <vector>

#ifdef HAVE_CUDA
extern "C"
{
    void cudaNativeVNNIPrefill_setDeterministicMode(bool enabled);
    bool cudaNativeVNNIPrefill_getDeterministicMode();
}
#endif

namespace llaminar2::test::parity::qwen36
{
    enum class MoEPrefixParityTopology
    {
        SingleDevice,
        NodeLocalTP,
        ExpertOverlayCuda2TPHotOnly,
        ExpertOverlayRocm2TPHotOnly,
        ExpertOverlayRocm2TPHotCpu2LocalTPCold,
    };

    struct MoEPrefixRestoreParityCase
    {
        std::string name;
        MoEPrefixParityTopology topology = MoEPrefixParityTopology::SingleDevice;
        std::vector<GlobalDeviceAddress> devices;
        std::vector<std::string> model_envs;
        std::string default_model_path;
        std::vector<std::string> metadata_envs;
        std::string default_metadata_path;
        std::string prompt = "The quick brown fox jumps over the lazy dog";
        std::string kv_cache_precision = "auto";
        std::string tp_allreduce_precision_override;
        int decode_steps = 3;
        int max_seq_len = 96;
        int prefix_restore_prompt_token_limit = 0;
        int required_cuda_devices = 0;
        int required_rocm_devices = 0;
        int required_cpu_sockets = 0;
        int mpi_ranks = 1;
        std::shared_ptr<MoERoutedExpertPlacementPlan> moe_routed_expert_plan;
        std::optional<MoERebalanceRuntimeConfig> moe_rebalance;
        std::vector<std::pair<std::string, std::string>> env_overrides;
    };

    inline size_t gib(size_t value)
    {
        return value * 1024ull * 1024ull * 1024ull;
    }

    inline std::string qwen36MoEBenchmarkPrompt()
    {
        return "The following is a comprehensive analysis of machine learning systems "
               "and their applications in modern computing environments. "
               "We will explore the fundamental concepts, examine practical implementations, "
               "and discuss the future directions of this rapidly evolving field. "
               "Machine learning has transformed how we approach problem-solving across "
               "numerous domains, from natural language processing to computer vision, "
               "from autonomous vehicles to medical diagnosis. "
               "The key to understanding these systems lies in grasping the underlying "
               "mathematical foundations while also appreciating the engineering challenges "
               "involved in deploying them at scale. "
               "Let us begin our exploration with an overview of the main paradigms: "
               "supervised learning, unsupervised learning, and reinforcement learning. "
               "Each of these approaches has its own strengths and is suited to different "
               "types of problems. In supervised learning, we train models using labeled data, "
               "where the correct output is known for each input example. "
               "This approach is particularly effective for classification and regression tasks. "
               "Unsupervised learning, on the other hand, deals with finding patterns in data "
               "without explicit labels. Clustering, dimensionality reduction, and anomaly detection "
               "are common applications. Reinforcement learning takes a different approach, "
               "where agents learn optimal behaviors through interaction with an environment, "
               "receiving rewards or penalties based on their actions. "
               "Deep learning, a subset of machine learning, has revolutionized the field "
               "by enabling the training of neural networks with many layers. "
               "These deep neural networks can learn hierarchical representations of data, "
               "automatically extracting features at multiple levels of abstraction. "
               "Convolutional neural networks have become the standard for image processing, "
               "while recurrent neural networks and transformers excel at sequential data. "
               "The transformer architecture, introduced in 2017, has become particularly influential, "
               "forming the basis for large language models like GPT, BERT, and LLaMA. "
               "These models are trained on vast amounts of text data and can perform "
               "a wide range of natural language tasks with impressive accuracy. "
               "The training process involves optimizing millions or billions of parameters "
               "using gradient descent and backpropagation algorithms. "
               "Modern training infrastructure relies on specialized hardware like GPUs and TPUs, "
               "distributed computing frameworks, and sophisticated optimization techniques. "
               "Transfer learning has emerged as a powerful paradigm, allowing models "
               "pre-trained on large datasets to be fine-tuned for specific tasks "
               "with relatively little additional data. This approach has democratized "
               "access to state-of-the-art AI capabilities for researchers and practitioners "
               "who may not have the resources to train large models from scratch. "
               "As we look to the future, several exciting developments are on the horizon. "
               "Multimodal models that can process text, images, audio, and video together "
               "are becoming increasingly sophisticated. Federated learning enables "
               "training on distributed data while preserving privacy. "
               "Neural architecture search automates the design of optimal network structures. "
               "And new hardware accelerators promise to make AI more efficient and accessible. "
               "The ethical implications of these technologies cannot be overlooked. "
               "Issues of bias, fairness, transparency, and accountability must be addressed "
               "as AI systems become more prevalent in society. Responsible AI development "
               "requires collaboration between technologists, policymakers, and the public "
               "to ensure these powerful tools benefit humanity as a whole.";
    }

    inline std::chrono::steady_clock::time_point parityPhaseStart()
    {
        return std::chrono::steady_clock::now();
    }

    inline void logMoEParityPhase(
        const MoEPrefixRestoreParityCase &test_case,
        const char *phase,
        std::chrono::steady_clock::time_point start)
    {
        const auto elapsed = std::chrono::steady_clock::now() - start;
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
        std::cerr << "[qwen36-moe-parity] case=" << test_case.name
                  << " phase=" << phase
                  << " elapsed_ms=" << ms << '\n';
    }

    class ScopedMoEParityProductionMode
    {
    public:
        explicit ScopedMoEParityProductionMode(bool enabled)
            : enabled_(enabled)
        {
            if (!enabled_)
            {
                return;
            }

            if (const char *old_value = std::getenv("LLAMINAR_DETERMINISTIC"))
            {
                had_old_deterministic_env_ = true;
                old_deterministic_env_ = old_value;
            }

#ifdef HAVE_CUDA
            old_cuda_prefill_deterministic_ = cudaNativeVNNIPrefill_getDeterministicMode();
#endif

            /*
             * Default parity remains a production-path canary, so deterministic
             * dispatch is forced off even when the surrounding shell enables it.
             * LLAMINAR_PARITY_DIAGNOSTIC_DETERMINISTIC is an explicit debugging
             * override used to prove whether a tiny prefix-invariance delta is
             * caused by non-deterministic CUDA prefill kernels; it is never set
             * by the test cases themselves.
             */
            const char *diagnostic_deterministic =
                std::getenv("LLAMINAR_PARITY_DIAGNOSTIC_DETERMINISTIC");
            const bool force_diagnostic_determinism =
                diagnostic_deterministic &&
                diagnostic_deterministic[0] != '\0' &&
                std::strcmp(diagnostic_deterministic, "0") != 0 &&
                std::strcmp(diagnostic_deterministic, "false") != 0 &&
                std::strcmp(diagnostic_deterministic, "FALSE") != 0;

            setenv("LLAMINAR_DETERMINISTIC", force_diagnostic_determinism ? "1" : "0", 1);
            mutableDebugEnv().reload();
#ifdef HAVE_CUDA
            cudaNativeVNNIPrefill_setDeterministicMode(force_diagnostic_determinism);
#endif
            llaminar::v2::kernels::KernelFactory::clearCache();
        }

        ~ScopedMoEParityProductionMode()
        {
            if (!enabled_)
            {
                return;
            }

#ifdef HAVE_CUDA
            cudaNativeVNNIPrefill_setDeterministicMode(old_cuda_prefill_deterministic_);
#endif
            if (had_old_deterministic_env_)
            {
                setenv("LLAMINAR_DETERMINISTIC", old_deterministic_env_.c_str(), 1);
            }
            else
            {
                unsetenv("LLAMINAR_DETERMINISTIC");
            }
            mutableDebugEnv().reload();
            llaminar::v2::kernels::KernelFactory::clearCache();
        }

        ScopedMoEParityProductionMode(const ScopedMoEParityProductionMode &) = delete;
        ScopedMoEParityProductionMode &operator=(const ScopedMoEParityProductionMode &) = delete;

    private:
        bool enabled_ = false;
        bool had_old_deterministic_env_ = false;
        std::string old_deterministic_env_;
#ifdef HAVE_CUDA
        bool old_cuda_prefill_deterministic_ = false;
#endif
    };

    class ScopedMoEPrefixCaseEnvironment
    {
    public:
        explicit ScopedMoEPrefixCaseEnvironment(
            const std::vector<std::pair<std::string, std::string>> &values)
        {
            for (const auto &[name, value] : values)
            {
                Entry entry;
                entry.name = name;
                if (const char *old_value = std::getenv(name.c_str()))
                {
                    entry.had_old_value = true;
                    entry.old_value = old_value;
                }
                entries_.push_back(std::move(entry));
                setenv(name.c_str(), value.c_str(), 1);
            }

            if (!entries_.empty())
            {
                mutableDebugEnv().reload();
                llaminar::v2::kernels::KernelFactory::clearCache();
            }
        }

        ~ScopedMoEPrefixCaseEnvironment()
        {
            for (auto it = entries_.rbegin(); it != entries_.rend(); ++it)
            {
                if (it->had_old_value)
                {
                    setenv(it->name.c_str(), it->old_value.c_str(), 1);
                }
                else
                {
                    unsetenv(it->name.c_str());
                }
            }

            if (!entries_.empty())
            {
                mutableDebugEnv().reload();
                llaminar::v2::kernels::KernelFactory::clearCache();
            }
        }

        ScopedMoEPrefixCaseEnvironment(const ScopedMoEPrefixCaseEnvironment &) = delete;
        ScopedMoEPrefixCaseEnvironment &operator=(const ScopedMoEPrefixCaseEnvironment &) = delete;

    private:
        struct Entry
        {
            std::string name;
            bool had_old_value = false;
            std::string old_value;
        };

        std::vector<Entry> entries_;
    };

    class ScopedCudaMoEFusedVerifierPrefillRoutes
    {
    public:
        ScopedCudaMoEFusedVerifierPrefillRoutes()
            : old_gateup_kpart_decode_(mutableDebugEnv().gemm.cuda_moe_gateup_kpart_decode),
              old_down_kpart_decode_(mutableDebugEnv().gemm.cuda_moe_down_kpart_decode),
              old_prefill_fuse_swiglu_(mutableDebugEnv().gemm.cuda_moe_prefill_fuse_swiglu),
              old_prefill_tile_m_(mutableDebugEnv().gemm.cuda_moe_prefill_tile_m)
        {
            auto &gemm = mutableDebugEnv().gemm;
            const bool allow_split_k_decode = !gemm.deterministic;
            gemm.cuda_moe_gateup_kpart_decode = allow_split_k_decode;
            gemm.cuda_moe_down_kpart_decode = allow_split_k_decode;
            gemm.cuda_moe_prefill_fuse_swiglu = true;
            gemm.cuda_moe_prefill_tile_m = 0;
            llaminar::v2::kernels::KernelFactory::clearCache();
        }

        ~ScopedCudaMoEFusedVerifierPrefillRoutes()
        {
            auto &gemm = mutableDebugEnv().gemm;
            gemm.cuda_moe_gateup_kpart_decode = old_gateup_kpart_decode_;
            gemm.cuda_moe_down_kpart_decode = old_down_kpart_decode_;
            gemm.cuda_moe_prefill_fuse_swiglu = old_prefill_fuse_swiglu_;
            gemm.cuda_moe_prefill_tile_m = old_prefill_tile_m_;
            llaminar::v2::kernels::KernelFactory::clearCache();
        }

        ScopedCudaMoEFusedVerifierPrefillRoutes(const ScopedCudaMoEFusedVerifierPrefillRoutes &) = delete;
        ScopedCudaMoEFusedVerifierPrefillRoutes &operator=(const ScopedCudaMoEFusedVerifierPrefillRoutes &) = delete;

    private:
        bool old_gateup_kpart_decode_ = false;
        bool old_down_kpart_decode_ = false;
        bool old_prefill_fuse_swiglu_ = false;
        int old_prefill_tile_m_ = 0;
    };

    /**
     * @brief Owns the lifetime of MoE verifier snapshot diagnostics.
     *
     * The grouped-verifier diagnostic compares compact all-position verifier
     * rows against row-by-row serial decode snapshots.  It does not need the
     * long prompt prefill or the setup-token forwards.  Keeping capture disabled
     * until the verifier base is restored avoids allocating graph snapshot
     * staging buffers for hundreds of prefill rows, which can exhaust 24 GB
     * cards before the verifier path under test even starts.
     */
    class ScopedMoEVerifierSnapshotCapture
    {
    public:
        ScopedMoEVerifierSnapshotCapture() = default;

        /**
         * @brief Start capture on the supplied runner after expensive setup work.
         *
         * Snapshot capture is intentionally armed as late as possible.  The
         * first grouped verifier forward and the serial replay forwards still
         * capture every instrumented stage, but the large prefill graph remains
         * a normal production-style forward without diagnostic storage.
         */
        void enable(IInferenceRunner &runner,
                    const std::vector<std::string> &snapshot_keys = {})
        {
            if (active_)
            {
                return;
            }

            runner_ = &runner;
            if (!snapshot_keys.empty())
            {
                runner_->setSnapshotCaptureFilter(snapshot_keys);
            }
            runner_->enableSnapshotCapture();
            active_ = true;
        }

        /**
         * @brief Clear captured tensors while leaving capture armed.
         */
        void clear() const
        {
            if (active_ && runner_)
            {
                runner_->clearSnapshots();
            }
        }

        /**
         * @brief Disable capture before leaving the diagnostic helper.
         */
        void disable()
        {
            if (active_ && runner_)
            {
                runner_->disableSnapshotCapture();
            }
            runner_ = nullptr;
            active_ = false;
        }

        ~ScopedMoEVerifierSnapshotCapture()
        {
            disable();
        }

        ScopedMoEVerifierSnapshotCapture(const ScopedMoEVerifierSnapshotCapture &) = delete;
        ScopedMoEVerifierSnapshotCapture &operator=(const ScopedMoEVerifierSnapshotCapture &) = delete;

    private:
        IInferenceRunner *runner_ = nullptr;
        bool active_ = false;
    };

    inline bool shouldForceMoEParityProductionMode(
        const MoEPrefixRestoreParityCase &)
    {
        /*
         * Real-model parity is a production-path canary. Force deterministic
         * mode off even if the surrounding shell exports LLAMINAR_DETERMINISTIC.
         */
        return true;
    }

    inline bool applyHostMoERebalanceIfNeeded(IOrchestrationRunner &runner)
    {
        if (runner.usesDeviceSideMoERebalanceController())
            return true;
        return runner.maybeApplyMoERebalance();
    }

    inline RoutedExpertDomain localTPMoEDomain(
        const std::string &name,
        CollectiveBackendType backend,
        std::vector<GlobalDeviceAddress> participants)
    {
        RoutedExpertDomain domain;
        domain.name = name;
        domain.scope = ExecutionDomainScope::LOCAL;
        domain.backend = backend;
        domain.participants = std::move(participants);
        domain.owner_rank = 0;
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        return domain;
    }

    inline RoutedExpertTier routedTier(
        const std::string &name,
        const std::string &domain,
        int priority,
        int max_experts_per_layer,
        size_t memory_budget_bytes,
        bool fallback = false)
    {
        RoutedExpertTier tier;
        tier.name = name;
        tier.domain = domain;
        tier.priority = priority;
        tier.max_experts_per_layer = max_experts_per_layer;
        tier.memory_budget_bytes = memory_budget_bytes;
        tier.fallback = fallback;
        return tier;
    }

    /**
     * @brief Owns the LocalTP context needed by direct expert-overlay runner helpers.
     *
     * Most prefix/MTP overlay cells enter through OrchestrationRunner, which
     * builds a RankOrchestrator and hands every per-device runner a live
     * LocalTP context plus its participant index.  A few focused verifier-row
     * helpers intentionally build one runner directly so they can drive
     * all-position verifier publication APIs without the full parity harness.
     * Those helpers still need the same continuation-domain TP context whenever
     * the MoE overlay plan requests phase-split dense decode; otherwise the
     * runner advertises dense_tp_decode_replicated but never materializes the
     * replicated dense/MTP sidecar weights.
     */
    struct MoEContinuationLocalTPHarness
    {
        std::unique_ptr<LocalTPContext> context; ///< Owned TP context; must outlive the runner.
        int participant_index = 0;               ///< Index of the runner device in the continuation domain.
        std::string error;                       ///< Non-empty when the fixture describes an invalid TP domain.
    };

    /**
     * @brief Owns a focused verifier proof runner and its model context.
     *
     * The model context is retained beside the runner so both direct
     * DeviceGraphOrchestrator and RankOrchestrator paths keep their shared
     * weight manager alive for the whole proof.
     */
    struct MoEVerifierProofRunner
    {
        std::shared_ptr<ModelContext> model_ctx;    ///< Concrete GGUF-backed model context.
        std::unique_ptr<IInferenceRunner> runner;   ///< Direct runner or LocalTP RankOrchestrator.
        std::string error;                          ///< Non-empty if runner construction failed.
    };

    /**
     * @brief Format MoE overlay validation failures for assertion messages.
     *
     * @param validation Result returned by validateMoERoutedExpertPlacementPlan().
     * @return Human-readable bullet list of validation errors.
     */
    inline std::string formatMoEOverlayValidationErrors(
        const MoERoutedExpertPlacementValidationResult &validation)
    {
        std::ostringstream message;
        for (const auto &error : validation.errors)
            message << "\n - " << error;
        return message.str();
    }

    /**
     * @brief Extract the Qwen3.6 MoE dimensions needed by the overlay planner.
     *
     * The planner owns explicit routed-expert placement.  Direct focused tests
     * must feed it the same model-derived dimensions as the broader math parity
     * suite so RankOrchestrator receives a first-class planned overlay instead
     * of a declarative request with no layer placements.
     *
     * @param ctx Loaded Qwen3.6 MoE model context.
     * @return Planner metadata for routed/shared expert capacity decisions.
     */
    inline MoERoutedExpertModelMetadata qwen36MoEPlannerMetadataFromModel(
        const ModelContext &ctx)
    {
        const auto &loader = ctx.concreteLoader();
        const std::string arch = ctx.architecture();

        MoERoutedExpertModelMetadata metadata;
        metadata.num_layers = ctx.totalBlockCount();
        metadata.num_experts = loader.getInt(arch + ".expert_count", 0);
        metadata.d_model = ctx.embeddingLength();
        metadata.routed_intermediate_size =
            loader.getInt(arch + ".expert_feed_forward_length", 0);
        if (metadata.routed_intermediate_size == 0)
            metadata.routed_intermediate_size = ctx.feedForwardLength();
        metadata.has_shared_expert =
            loader.getInt(arch + ".expert_shared_count", 0) > 0;
        metadata.shared_intermediate_size =
            metadata.has_shared_expert
                ? metadata.routed_intermediate_size
                : 0;
        metadata.routed_quant_type = "IQ3";
        metadata.shared_quant_type = "IQ3";
        return metadata;
    }

    /**
     * @brief Plan a declarative MoE overlay fixture against the loaded model.
     *
     * @param requested Declarative fixture plan.
     * @param ctx Loaded model context that provides layer/expert dimensions.
     * @param error Optional error sink.
     * @return Planned overlay with explicit layer placements, or nullptr on error.
     */
    inline std::shared_ptr<MoERoutedExpertPlacementPlan> planMoEOverlayForVerifierProof(
        const std::shared_ptr<MoERoutedExpertPlacementPlan> &requested,
        const ModelContext &ctx,
        std::string *error)
    {
        if (!requested)
            return nullptr;

        try
        {
            const auto metadata = qwen36MoEPlannerMetadataFromModel(ctx);
            auto planned =
                MoERoutedExpertPlacementPlanner::plan(*requested, metadata).planned_plan;

            MoERoutedExpertPlacementValidationOptions options;
            options.layer_count = metadata.num_layers;
            options.routed_expert_count = metadata.num_experts;
            const auto validation =
                validateMoERoutedExpertPlacementPlan(planned, options);
            if (!validation.ok())
            {
                if (error)
                {
                    *error =
                        "invalid planned Qwen3.6 MoE verifier overlay:" +
                        formatMoEOverlayValidationErrors(validation);
                }
                return nullptr;
            }
            return std::make_shared<MoERoutedExpertPlacementPlan>(std::move(planned));
        }
        catch (const std::exception &e)
        {
            if (error)
                *error = e.what();
            return nullptr;
        }
    }

    /**
     * @brief Build the continuation-domain LocalTP context for a direct runner.
     *
     * @param test_case Expert-overlay fixture being driven directly.
     * @param execution_device Local device used to construct the runner.
     * @return Harness containing an initialized LocalTP context, or an inactive
     *         harness when the fixture is not a multi-participant dense TP case.
     */
    inline MoEContinuationLocalTPHarness createMoEContinuationLocalTPHarness(
        const MoEPrefixRestoreParityCase &test_case,
        const DeviceId &execution_device)
    {
        MoEContinuationLocalTPHarness harness;
        const auto &plan = test_case.moe_routed_expert_plan;
        if (!plan ||
            !plan->continuation_domain_spec.dense_tp_enabled ||
            plan->continuation_domain.empty())
        {
            return harness;
        }

        const auto domain_it = std::find_if(
            plan->domains.begin(),
            plan->domains.end(),
            [&](const RoutedExpertDomain &domain)
            {
                return domain.name == plan->continuation_domain;
            });
        if (domain_it == plan->domains.end())
        {
            harness.error =
                "expert-overlay fixture names continuation domain '" +
                plan->continuation_domain + "' but no matching domain exists";
            return harness;
        }
        if (domain_it->scope != ExecutionDomainScope::LOCAL ||
            domain_it->participants.size() <= 1)
        {
            return harness;
        }

        int participant_index = -1;
        for (size_t i = 0; i < domain_it->participants.size(); ++i)
        {
            if (domain_it->participants[i].toLocalDeviceId() == execution_device)
            {
                participant_index = static_cast<int>(i);
                break;
            }
        }
        if (participant_index < 0)
        {
            std::ostringstream oss;
            oss << "runner device " << execution_device.toString()
                << " is not a participant in continuation domain '"
                << domain_it->name << "'";
            harness.error = oss.str();
            return harness;
        }

        try
        {
            auto context = std::make_unique<LocalTPContext>(
                domain_it->participants,
                domain_it->weights,
                domain_it->backend);
            /*
             * Qwen3.5/GDN graph construction asks the TP context for myIndex()
             * while computing local GDN state offsets. RankOrchestrator sets
             * this before building per-device runners; direct helpers must do
             * the same so they exercise the real LocalTP geometry.
             */
            context->setCurrentDeviceIndex(participant_index);
            harness.participant_index = participant_index;
            harness.context = std::move(context);
        }
        catch (const std::exception &e)
        {
            harness.error =
                "failed to create continuation LocalTP context for domain '" +
                domain_it->name + "': " + e.what();
        }
        return harness;
    }

    /**
     * @brief Create the runner used by focused MoE verifier/publication proofs.
     *
     * True single-device cases use the direct graph runner.  Phase-split
     * expert-overlay cases must instead use RankOrchestrator so all LocalTP
     * participants enter graph stages and collectives together.  This mirrors
     * production execution while keeping the proof code focused on verifier-row
     * publication.
     *
     * @param test_case Expert-overlay or single-device fixture.
     * @param model_path GGUF model path selected by the fixture.
     * @param execution_device Primary device for single-device construction.
     * @param config Runner configuration shared with the proof.
     * @return Runner bundle with either @ref runner set or @ref error populated.
     */
    inline MoEVerifierProofRunner createMoEVerifierProofRunner(
        const MoEPrefixRestoreParityCase &test_case,
        const std::string &model_path,
        const DeviceId &execution_device,
        const InferenceRunnerConfig &config)
    {
        MoEVerifierProofRunner result;
        const bool use_rank_orchestrator =
            [&]()
        {
            const auto &plan = test_case.moe_routed_expert_plan;
            if (!plan ||
                !plan->continuation_domain_spec.dense_tp_enabled ||
                plan->continuation_domain.empty())
            {
                return false;
            }
            const auto domain_it = std::find_if(
                plan->domains.begin(),
                plan->domains.end(),
                [&](const RoutedExpertDomain &domain)
                {
                    return domain.name == plan->continuation_domain;
                });
            return domain_it != plan->domains.end() &&
                   domain_it->scope == ExecutionDomainScope::LOCAL &&
                   domain_it->participants.size() > 1;
        }();
        const WeightDistributionStrategy weight_strategy =
            use_rank_orchestrator
                ? WeightDistributionStrategy::SHARDED
                : WeightDistributionStrategy::REPLICATED;
        result.model_ctx =
            createQwen36ParityModelContext(
                model_path,
                execution_device,
                weight_strategy);
        if (!result.model_ctx)
        {
            result.error = "failed to create Qwen3.6 MoE verifier proof model context";
            return result;
        }

        InferenceRunnerConfig effective_config = config;
        MoEPrefixRestoreParityCase effective_case = test_case;
        if (test_case.moe_routed_expert_plan)
        {
            std::string plan_error;
            auto planned_plan =
                planMoEOverlayForVerifierProof(
                    test_case.moe_routed_expert_plan,
                    *result.model_ctx,
                    &plan_error);
            if (!planned_plan)
            {
                result.error =
                    "failed to plan MoE verifier overlay: " + plan_error;
                return result;
            }
            effective_case.moe_routed_expert_plan = planned_plan;
            effective_config.moe_routed_expert_plan = planned_plan;
        }

        if (!use_rank_orchestrator)
        {
            result.runner =
                createInferenceRunner(
                    result.model_ctx,
                    nullptr,
                    execution_device,
                    effective_config);
            if (!result.runner)
                result.error = "failed to create direct verifier proof runner";
            return result;
        }

        auto local_tp_harness =
            createMoEContinuationLocalTPHarness(
                effective_case,
                execution_device);
        if (!local_tp_harness.error.empty())
        {
            result.error = local_tp_harness.error;
            return result;
        }
        if (!local_tp_harness.context)
        {
            result.error =
                "planned LocalTP verifier overlay did not produce a continuation TP context";
            return result;
        }

        RankOrchestrator::Config rank_config;
        rank_config.mode = RankOrchestrator::ParallelismMode::TP;
        rank_config.devices = local_tp_harness.context->devices();
        rank_config.weights = local_tp_harness.context->weights();
        rank_config.backend = local_tp_harness.context->backend();
        rank_config.max_seq_len = static_cast<size_t>(config.max_seq_len);
        rank_config.batch_size = config.batch_size;
        rank_config.activation_precision = config.activation_precision;
        rank_config.kv_cache_scale_k = config.kv_cache_scale_k;
        rank_config.kv_cache_scale_v = config.kv_cache_scale_v;
        rank_config.kv_cache_precision = config.kv_cache_precision;
        rank_config.tp_allreduce_precision_override =
            config.tp_allreduce_precision_override;
        rank_config.prefix_cache = config.prefix_cache;
        rank_config.mtp = config.mtp;
        rank_config.routed_expert_compute_policy = config.routed_expert_compute_policy;
        rank_config.moe_hot_expert_cache = config.moe_hot_expert_cache;
        rank_config.moe_rebalance = config.moe_rebalance;
        rank_config.use_mapped_memory = config.use_mapped_memory;
        rank_config.prepared_weight_store = config.prepared_weight_store;
        rank_config.moe_routed_expert_plan =
            effective_config.moe_routed_expert_plan;
        rank_config.moe_expert_overlay_mpi_ctx =
            effective_config.moe_expert_overlay_mpi_ctx;

        result.runner =
            createRankOrchestrator(
                result.model_ctx,
                std::move(local_tp_harness.context),
                rank_config);
        if (!result.runner)
            result.error = "failed to create LocalTP verifier proof RankOrchestrator";
        return result;
    }

    inline std::shared_ptr<MoERoutedExpertPlacementPlan> qwen36MoEOverlayPlanRocm2TPHotCpu2LocalTPCold()
    {
        constexpr const char *kRocmHotDomain = "qwen36_moe_rocm_hot";
        constexpr const char *kCpuColdDomain = "qwen36_moe_cpu_cold";

        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan->continuation_domain = kRocmHotDomain;
        plan->shared_expert_domain = kRocmHotDomain;
        plan->domains = {
            localTPMoEDomain(
                kRocmHotDomain,
                CollectiveBackendType::RCCL,
                {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)}),
            localTPMoEDomain(
                kCpuColdDomain,
                CollectiveBackendType::UPI,
                {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)}),
        };
        plan->routed_tiers = {
            routedTier("hot", kRocmHotDomain, 0, 240, gib(4)),
            routedTier("cold", kCpuColdDomain, 1, 0, 0, true),
        };
        /*
         * MTP verifier rows are dense decode work even when base-layer routed
         * experts spill to a CPU cold tier.  Keep the continuation domain in
         * the same phase-split policy used by the hot-only MTP fixtures so the
         * ROCm hot devices own the replicated terminal dense/MTP sidecar while
         * the CPU domain remains responsible only for cold routed expert rows.
         */
        plan->continuation_domain_spec.setDensePolicy(
            DenseParallelPolicy::PrefillTensorParallelDecodeReplicated);
        return plan;
    }

    inline std::shared_ptr<MoERoutedExpertPlacementPlan> qwen36MoEOverlayPlanRocm2TPHotOnly()
    {
        constexpr const char *kRocmHotDomain = "qwen36_moe_rocm_hot";

        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan->continuation_domain = kRocmHotDomain;
        plan->shared_expert_domain = kRocmHotDomain;
        plan->domains = {
            localTPMoEDomain(
                kRocmHotDomain,
                CollectiveBackendType::RCCL,
                {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)}),
        };
        plan->routed_tiers = {
            routedTier("hot", kRocmHotDomain, 0, 256, gib(8)),
        };
        plan->continuation_domain_spec.setDensePolicy(
            DenseParallelPolicy::PrefillTensorParallelDecodeReplicated);
        return plan;
    }

    inline std::shared_ptr<MoERoutedExpertPlacementPlan> qwen36MoEOverlayPlanCuda2TPHotOnly()
    {
        constexpr const char *kCudaHotDomain = "qwen36_moe_cuda_hot";

        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan->continuation_domain = kCudaHotDomain;
        plan->shared_expert_domain = kCudaHotDomain;
        plan->domains = {
            localTPMoEDomain(
                kCudaHotDomain,
                CollectiveBackendType::NCCL,
                {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}),
        };
        plan->routed_tiers = {
            routedTier("hot", kCudaHotDomain, 0, 256, gib(8)),
        };
        plan->continuation_domain_spec.setDensePolicy(
            DenseParallelPolicy::PrefillTensorParallelDecodeReplicated);
        return plan;
    }

    inline bool regenerateQwen36MoEMetadata(
        const std::string &model_path,
        const std::filesystem::path &metadata_path,
        const std::string &prompt,
        int decode_steps,
        std::string *output)
    {
        std::filesystem::create_directories(metadata_path.parent_path());

        std::string script =
            "unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS "
            "OMP_PROC_BIND OMP_PLACES KMP_AFFINITY; "
            "[ -f /workspaces/llaminar/.venv/bin/activate ] && "
            "source /workspaces/llaminar/.venv/bin/activate; "
            "python3 python/reference/generate_qwen35_moe_pipeline_snapshots.py";
        script += " --model " + shellQuote(model_path);
        script += " --prompt " + shellQuote(prompt);
        script += " --decode-steps " + std::to_string(decode_steps);
        script += " --output " + shellQuote(metadata_path.parent_path().string());
        script += " --metadata-only";

        const std::string command = "bash -c " + shellQuote(script) + " 2>&1";
        FILE *pipe = popen(command.c_str(), "r");
        if (!pipe)
        {
            if (output)
            {
                *output = "failed to spawn python MoE metadata generator";
            }
            return false;
        }

        char buffer[512];
        std::string captured;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
        {
            captured += buffer;
        }

        const int exit_code = pclose(pipe);
        if (output)
        {
            *output = std::move(captured);
        }
        return exit_code == 0;
    }

    inline bool regenerateQwen36MoEDecodeSnapshots(
        const std::string &model_path,
        const std::filesystem::path &metadata_path,
        const std::string &prompt,
        int decode_steps,
        bool include_mtp_sidecar_snapshots,
        std::string *output)
    {
        std::filesystem::create_directories(metadata_path.parent_path());

        std::string script =
            "unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS "
            "OMP_PROC_BIND OMP_PLACES KMP_AFFINITY; "
            "[ -f /workspaces/llaminar/.venv/bin/activate ] && "
            "source /workspaces/llaminar/.venv/bin/activate; "
            "python3 python/reference/generate_qwen35_moe_pipeline_snapshots.py";
        script += " --model " + shellQuote(model_path);
        script += " --prompt " + shellQuote(prompt);
        script += " --decode-steps " + std::to_string(decode_steps);
        script += " --output " + shellQuote(metadata_path.parent_path().string());
        script += " --decode-snapshots-only";
        if (include_mtp_sidecar_snapshots)
            script += " --mtp-sidecar-snapshots";

        const std::string command = "bash -c " + shellQuote(script) + " 2>&1";
        FILE *pipe = popen(command.c_str(), "r");
        if (!pipe)
        {
            if (output)
            {
                *output = "failed to spawn python MoE decode snapshot generator";
            }
            return false;
        }

        char buffer[512];
        std::string captured;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
        {
            captured += buffer;
        }

        const int exit_code = pclose(pipe);
        if (output)
        {
            *output = std::move(captured);
        }
        return exit_code == 0;
    }

    inline bool qwen36MoEDecodeSnapshotsLookUsable(
        const std::filesystem::path &metadata_path,
        const std::string &expected_prompt,
        int decode_steps,
        bool require_mtp_sidecar_snapshots = false)
    {
        if (!metadataLooksUsable(metadata_path, expected_prompt, decode_steps))
        {
            return false;
        }

        const auto dir = metadata_path.parent_path();
        if (decode_steps <= 0)
        {
            return true;
        }

        if (!std::filesystem::exists(dir / "decode_step0_LM_HEAD.npy") ||
            !std::filesystem::exists(dir / "decode_step0_layer0_ATTENTION_NORM.npy"))
        {
            return false;
        }

        if (!require_mtp_sidecar_snapshots)
        {
            return true;
        }

        const std::vector<std::string> required_mtp_sidecar_files = {
            "decode_step0_MTP_TERMINAL_HIDDEN_ROW_SELECT.npy",
            "decode_step0_MTP0_EMBEDDING.npy",
            "decode_step0_MTP0_NORM_HIDDEN.npy",
            "decode_step0_MTP0_NORM_EMBEDDING.npy",
            "decode_step0_MTP0_CONCAT.npy",
            "decode_step0_MTP0_FC.npy",
            "decode_step0_MTP0_ATTENTION_NORM.npy",
            "decode_step0_MTP0_Q_PROJECTION.npy",
            "decode_step0_MTP0_Q_NORM.npy",
            "decode_step0_MTP0_K_PROJECTION.npy",
            "decode_step0_MTP0_K_NORM.npy",
            "decode_step0_MTP0_V_PROJECTION.npy",
            "decode_step0_MTP0_ATTENTION_CONTEXT.npy",
            "decode_step0_MTP0_ATTENTION_CONTEXT_GATED.npy",
            "decode_step0_MTP0_ATTENTION_OUTPUT.npy",
            "decode_step0_MTP0_FFN_NORM.npy",
            "decode_step0_MTP0_MOE_ROUTER_OUTPUT.npy",
            "decode_step0_MTP0_MOE_ROUTING_INDICES.npy",
            "decode_step0_MTP0_MOE_ROUTING_WEIGHTS.npy",
            "decode_step0_MTP0_MOE_EXPERT_OUTPUT.npy",
            "decode_step0_MTP0_MOE_SHARED_EXPERT_OUTPUT.npy",
            "decode_step0_MTP0_MOE_SHARED_GATE_OUTPUT.npy",
            "decode_step0_MTP0_MOE_COMBINED_OUTPUT.npy",
            "decode_step0_MTP0_FFN_RESIDUAL.npy",
            "decode_step0_MTP0_FINAL_NORM.npy",
            "decode_step0_MTP0_LM_HEAD.npy",
        };
        return std::all_of(
            required_mtp_sidecar_files.begin(),
            required_mtp_sidecar_files.end(),
            [&](const std::string &file)
            {
                return std::filesystem::exists(dir / file);
            });
    }

    inline void ensurePyTorchMoEDecodeSnapshots(
        const MoEPrefixRestoreParityCase &test_case,
        const std::string &model_path,
        const std::filesystem::path &metadata_path,
        bool require_mtp_sidecar_snapshots = false)
    {
        if (qwen36MoEDecodeSnapshotsLookUsable(
                metadata_path,
                test_case.prompt,
                test_case.decode_steps,
                require_mtp_sidecar_snapshots))
        {
            return;
        }

        std::string output;
        ASSERT_TRUE(regenerateQwen36MoEDecodeSnapshots(
            model_path,
            metadata_path,
            test_case.prompt,
            test_case.decode_steps,
            require_mtp_sidecar_snapshots,
            &output))
            << test_case.name << " failed to regenerate PyTorch MoE decode snapshots at "
            << metadata_path.parent_path() << "\n"
            << output;

        ASSERT_TRUE(qwen36MoEDecodeSnapshotsLookUsable(
            metadata_path,
            test_case.prompt,
            test_case.decode_steps,
            require_mtp_sidecar_snapshots))
            << test_case.name << " regenerated MoE decode snapshots are incomplete at "
            << metadata_path.parent_path() << "\n"
            << output;
    }

    inline void ensurePyTorchMoEMetadata(
        const MoEPrefixRestoreParityCase &test_case,
        const std::string &model_path,
        const std::filesystem::path &metadata_path)
    {
        if (metadataLooksUsable(metadata_path, test_case.prompt, test_case.decode_steps))
        {
            return;
        }

        std::string output;
        ASSERT_TRUE(regenerateQwen36MoEMetadata(
            model_path,
            metadata_path,
            test_case.prompt,
            test_case.decode_steps,
            &output))
            << test_case.name << " failed to regenerate PyTorch MoE metadata at "
            << metadata_path << "\n"
            << output;

        ASSERT_TRUE(metadataLooksUsable(metadata_path, test_case.prompt, test_case.decode_steps))
            << test_case.name << " regenerated MoE metadata is incomplete at "
            << metadata_path << "\n"
            << output;
    }

    inline std::optional<std::string> moePrefixParitySkipReason(
        const MoEPrefixRestoreParityCase &test_case)
    {
        const int world_size = mpiWorldSize();
        if (test_case.topology == MoEPrefixParityTopology::NodeLocalTP)
        {
            if (world_size != test_case.mpi_ranks)
            {
                std::ostringstream oss;
                oss << test_case.name << " requires exactly "
                    << test_case.mpi_ranks << " MPI rank(s), got "
                    << world_size;
                return oss.str();
            }
        }
        else if (world_size != 1)
        {
            return test_case.name + " is a local topology test and must run with one MPI rank";
        }

        if (test_case.required_cuda_devices > 0 || test_case.required_rocm_devices > 0)
        {
            auto &dm = DeviceManager::instance();
            dm.initialize(-1, false);
            if (dm.cuda_device_count() < test_case.required_cuda_devices)
            {
                std::ostringstream oss;
                oss << test_case.name << " requires "
                    << test_case.required_cuda_devices
                    << " CUDA device(s)";
                return oss.str();
            }
            if (dm.rocm_device_count() < test_case.required_rocm_devices)
            {
                std::ostringstream oss;
                oss << test_case.name << " requires "
                    << test_case.required_rocm_devices
                    << " ROCm device(s)";
                return oss.str();
            }
        }

        if (test_case.required_cpu_sockets > 0)
        {
            const auto hw = HardwareInventory::detect();
            if (hw.num_sockets() < test_case.required_cpu_sockets)
            {
                std::ostringstream oss;
                oss << test_case.name << " requires "
                    << test_case.required_cpu_sockets
                    << " CPU socket(s)";
                return oss.str();
            }
        }

        return std::nullopt;
    }

    inline OrchestrationConfig makeMoEPrefixRestoreConfig(
        const MoEPrefixRestoreParityCase &test_case,
        const std::string &model_path,
        bool enable_prefix_cache,
        int block_size,
        bool enable_mtp = false,
        int mtp_draft_tokens = 1,
        MTPDepthPolicyConfig mtp_depth_policy = {})
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path;
        config.max_seq_len = test_case.max_seq_len;
        config.batch_size = 1;
        config.activation_precision = "fp32";
        config.kv_cache_precision = test_case.kv_cache_precision;
        config.tp_allreduce_precision_override =
            test_case.tp_allreduce_precision_override;
        config.prefix_cache.enabled = enable_prefix_cache;
        config.prefix_cache.storage_mode = enable_prefix_cache
                                               ? PrefixCacheStorageMode::Ram
                                               : PrefixCacheStorageMode::Disabled;
        config.prefix_cache.block_size = block_size;
        config.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Auto;
        config.prefix_cache.ram_budget_bytes = 4ull * 1024ull * 1024ull * 1024ull;
        config.mtp.enabled = enable_mtp;
        config.mtp.draft_tokens = std::max(1, mtp_draft_tokens);
        config.mtp.depth_policy = mtp_depth_policy;
        config.moe_routed_expert_plan = test_case.moe_routed_expert_plan;
        if (test_case.moe_rebalance)
        {
            config.moe_rebalance = *test_case.moe_rebalance;
        }
        if (config.moe_rebalance.mode == MoERebalanceRuntimeMode::LLEP &&
            config.moe_rebalance.prefill_window_tokens <= 0)
        {
            /*
             * Prefix-restore parity compares cache-disabled baselines with
             * cache-enabled restores.  LLEP plans routes over the current
             * prefill window, so the test pins that request-local owner to the
             * prefix-cache block size rather than letting the uncached baseline
             * use one monolithic prompt transaction.
             */
            config.moe_rebalance.prefill_window_tokens = block_size;
        }

        switch (test_case.topology)
        {
        case MoEPrefixParityTopology::SingleDevice:
            config.tp_degree = 1;
            config.pp_degree = 1;
            config.device_for_this_rank = test_case.devices.empty()
                                              ? GlobalDeviceAddress::cpu()
                                              : test_case.devices.front();
            break;
        case MoEPrefixParityTopology::NodeLocalTP:
            config.tp_degree = test_case.mpi_ranks;
            config.tp_scope = TPScope::NODE_LOCAL;
            config.pp_degree = 1;
            config.default_backend = CollectiveBackendType::MPI;
            config.device_mode = DeviceAssignmentMode::EXPLICIT;
            config.device_map.clear();
            config.device_map_numa_explicit.clear();
            for (int rank = 0;
                 rank < test_case.mpi_ranks &&
                 rank < static_cast<int>(test_case.devices.size());
                 ++rank)
            {
                config.device_map.emplace_back(rank, test_case.devices[rank]);
                config.device_map_numa_explicit.emplace_back(
                    rank,
                    test_case.devices[rank].hasValidNuma());
            }
            break;
        case MoEPrefixParityTopology::ExpertOverlayCuda2TPHotOnly:
        case MoEPrefixParityTopology::ExpertOverlayRocm2TPHotOnly:
        case MoEPrefixParityTopology::ExpertOverlayRocm2TPHotCpu2LocalTPCold:
            config.tp_degree = 1;
            config.pp_degree = 1;
            break;
        }

        return config;
    }

    /**
     * @brief Verifies that an MTP parity case can fit its full prompt and decode budget.
     *
     * MTP restore parity intentionally feeds the complete PyTorch metadata prompt
     * into both the baseline runner and the prefix-cache runner.  Partial-hit
     * prefix restore tests may trim that prompt elsewhere, but MTP cache restore
     * is validating the full request boundary and must not rely on that trim.
     * Failing this check means the fixture's declared context window is too
     * small for the metadata it selected, and the fix is to resize the fixture or
     * choose smaller metadata rather than allowing a late device-capacity error.
     *
     * @param test_case Parity case whose context window and decode budget are declared.
     * @param prompt_token_count Number of prompt tokens loaded from PyTorch metadata.
     * @return GoogleTest assertion result with a detailed capacity mismatch message.
     */
    inline ::testing::AssertionResult moEMTPParityCaseFitsContextWindow(
        const MoEPrefixRestoreParityCase &test_case,
        size_t prompt_token_count)
    {
        const int prompt_tokens = static_cast<int>(prompt_token_count);
        const int required_tokens = prompt_tokens + test_case.decode_steps;
        if (required_tokens <= test_case.max_seq_len)
            return ::testing::AssertionSuccess();

        return ::testing::AssertionFailure()
               << test_case.name
               << " MTP parity requires prompt tokens (" << prompt_tokens
               << ") + decode steps (" << test_case.decode_steps
               << ") <= max_seq_len (" << test_case.max_seq_len
               << "). Increase the case max_seq_len or select smaller metadata "
                  "before constructing runners.";
    }

    inline MoEPrefixRestoreParityCase qwen36MoEPrefixParityCase(
        const std::string &name,
        MoEPrefixParityTopology topology)
    {
        MoEPrefixRestoreParityCase test_case{
            .name = name,
            .topology = topology,
            .model_envs = {
                "LLAMINAR_QWEN36_MOE_MODEL",
                "LLAMINAR_PARITY_MOE_MODEL",
            },
            .default_model_path = "/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
            .metadata_envs = {
                "LLAMINAR_QWEN36_MOE_PARITY_METADATA",
                "LLAMINAR_PARITY_MOE_METADATA",
            },
            .default_metadata_path = "pytorch_qwen36_moe_snapshots/metadata.txt",
            .prompt = "The quick brown fox jumps over the lazy dog",
            .kv_cache_precision = "auto",
            .decode_steps = 3,
            .max_seq_len = 96,
        };

        switch (topology)
        {
        case MoEPrefixParityTopology::SingleDevice:
            test_case.devices = {GlobalDeviceAddress::rocm(0)};
            test_case.required_rocm_devices = 1;
            break;
        case MoEPrefixParityTopology::NodeLocalTP:
            test_case.devices = {
                GlobalDeviceAddress::cpu(0),
                GlobalDeviceAddress::cpu(1),
            };
            test_case.required_cpu_sockets = 2;
            test_case.mpi_ranks = 2;
            break;
        case MoEPrefixParityTopology::ExpertOverlayCuda2TPHotOnly:
            test_case.devices = {
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::cuda(1),
            };
            test_case.required_cuda_devices = 2;
            test_case.tp_allreduce_precision_override = "schema";
            test_case.moe_routed_expert_plan =
                qwen36MoEOverlayPlanCuda2TPHotOnly();
            break;
        case MoEPrefixParityTopology::ExpertOverlayRocm2TPHotOnly:
            test_case.devices = {
                GlobalDeviceAddress::rocm(0),
                GlobalDeviceAddress::rocm(1),
            };
            test_case.required_rocm_devices = 2;
            test_case.tp_allreduce_precision_override = "schema";
            test_case.moe_routed_expert_plan =
                qwen36MoEOverlayPlanRocm2TPHotOnly();
            break;
        case MoEPrefixParityTopology::ExpertOverlayRocm2TPHotCpu2LocalTPCold:
            test_case.devices = {
                GlobalDeviceAddress::rocm(0),
                GlobalDeviceAddress::rocm(1),
                GlobalDeviceAddress::cpu(0),
                GlobalDeviceAddress::cpu(1),
            };
            test_case.required_rocm_devices = 2;
            test_case.required_cpu_sockets = 2;
            test_case.moe_routed_expert_plan =
                qwen36MoEOverlayPlanRocm2TPHotCpu2LocalTPCold();
            break;
        }

        return test_case;
    }

    inline void loadMoEReferenceInputs(
        const MoEPrefixRestoreParityCase &test_case,
        std::string *model_path,
        std::vector<int32_t> *prompt_tokens,
        std::vector<int32_t> *expected_tokens)
    {
        if (auto skip_reason = moePrefixParitySkipReason(test_case))
        {
            GTEST_SKIP() << *skip_reason;
        }

        *model_path = firstEnvOrDefault(
            test_case.model_envs,
            test_case.default_model_path);
        if (!std::filesystem::exists(*model_path))
        {
            GTEST_SKIP() << test_case.name << " model not found: " << *model_path;
        }

        const std::filesystem::path metadata_path = firstEnvOrDefault(
            test_case.metadata_envs,
            test_case.default_metadata_path);
        ensurePyTorchMoEMetadata(test_case, *model_path, metadata_path);

        *prompt_tokens = readTokenListFromMetadata(metadata_path, "token_ids");
        const auto pytorch_decode_tokens = readTokenListFromMetadata(metadata_path, "decode_tokens");
        ASSERT_FALSE(prompt_tokens->empty());
        ASSERT_GE(pytorch_decode_tokens.size(), static_cast<size_t>(test_case.decode_steps));

        expected_tokens->assign(
            pytorch_decode_tokens.begin(),
            pytorch_decode_tokens.begin() + test_case.decode_steps);
    }

    inline bool moeReferenceInputsStoppedCurrentTest()
    {
        return ::testing::Test::IsSkipped() ||
               ::testing::Test::HasFatalFailure();
    }

    inline bool moECaseExpectsAllPositionSpecPublication(
        const MoEPrefixRestoreParityCase &test_case)
    {
        /*
         * Direct all-position MoE publication is a capability of a fully-owned
         * verifier graph, not a property of the model alone.  SingleDevice GPU
         * MoE now uses the grouped decode-equivalent outcome publisher instead:
         * it keeps the target verifier mathematically decode-equivalent while
         * publishing accepted state from compact device metadata.
         */
        (void)test_case;
        return false;
    }

    inline bool moECaseExpectsGroupedOutcomeDevicePublication(
        const MoEPrefixRestoreParityCase &test_case)
    {
        if (test_case.devices.empty())
        {
            return false;
        }

        const bool all_gpu =
            std::all_of(
                test_case.devices.begin(),
                test_case.devices.end(),
                [](const GlobalDeviceAddress &device)
                {
                    return device.isGPU();
                });
        if (!all_gpu)
            return false;

        return test_case.topology == MoEPrefixParityTopology::SingleDevice ||
               test_case.topology ==
                   MoEPrefixParityTopology::ExpertOverlayCuda2TPHotOnly ||
               test_case.topology ==
                   MoEPrefixParityTopology::ExpertOverlayRocm2TPHotOnly;
    }

    inline bool moEPrefixCaseUsesGPU(const MoEPrefixRestoreParityCase &test_case)
    {
        return std::any_of(
            test_case.devices.begin(),
            test_case.devices.end(),
            [](const GlobalDeviceAddress &device)
            {
                return device.isGPU();
            });
    }

    inline bool moEPrefixCaseUsesCUDA(const MoEPrefixRestoreParityCase &test_case)
    {
        return std::any_of(
            test_case.devices.begin(),
            test_case.devices.end(),
            [](const GlobalDeviceAddress &device)
            {
                return device.isCUDA();
            });
    }

    inline bool moEPrefixCaseUsesROCm(const MoEPrefixRestoreParityCase &test_case)
    {
        return std::any_of(
            test_case.devices.begin(),
            test_case.devices.end(),
            [](const GlobalDeviceAddress &device)
            {
                return device.isROCm();
            });
    }

    inline bool hasMTPPerfCounter(
        const std::vector<PerfStatRecord> &records,
        const char *name)
    {
        return std::any_of(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.kind == PerfStatRecord::Kind::Counter &&
                       record.domain == "mtp" &&
                       record.name == name;
            });
    }

    inline bool hasPerfCounter(
        const std::vector<PerfStatRecord> &records,
        const char *domain,
        const char *name)
    {
        return std::any_of(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.kind == PerfStatRecord::Kind::Counter &&
                       record.domain == domain &&
                       record.name == name;
            });
    }

    inline double perfCounterSum(
        const std::vector<PerfStatRecord> &records,
        const std::string &domain,
        const std::string &name)
    {
        double total = 0.0;
        for (const auto &record : records)
        {
            if (record.kind == PerfStatRecord::Kind::Counter &&
                record.domain == domain &&
                record.name == name)
            {
                total += record.value;
            }
        }
        return total;
    }

    inline void expectPerfCounterPositive(
        const std::vector<PerfStatRecord> &records,
        const std::string &domain,
        const std::string &name,
        const std::string &context)
    {
        EXPECT_GT(perfCounterSum(records, domain, name), 0.0)
            << context << " should publish non-zero " << domain << "."
            << name << ".\n"
            << PerfStatsCollector::summaryString({domain});
    }

    inline void expectPerfCounterZero(
        const std::vector<PerfStatRecord> &records,
        const std::string &domain,
        const std::string &name,
        const std::string &context)
    {
        EXPECT_EQ(perfCounterSum(records, domain, name), 0.0)
            << context << " should not publish error counter " << domain
            << "." << name << ".\n"
            << PerfStatsCollector::summaryString({domain});
    }

    /**
     * @brief Assert that prefix-cache restore counters prove a full-block hit.
     *
     * The prefix-state probe verifies restored tensor ownership.  These
     * perfstats assertions make sure the integration matrix also exercised the
     * prefix-cache machinery itself instead of accidentally comparing two
     * uncached requests that happen to produce the same tokens.
     */
    inline void expectMoEPrefixCachePerfPath(
        const std::vector<PerfStatRecord> &records,
        const std::string &context)
    {
        expectPerfCounterPositive(
            records,
            "prefix_cache",
            "block_hits",
            context);
        expectPerfCounterPositive(
            records,
            "prefix_cache",
            "populate_restores",
            context);
    }

    /**
     * @brief Assert that MoE rebalance counters match the mode named by a fixture.
     *
     * Dynamic and LLEP use different movement planners.  Token parity can pass
     * even if a planner never moved anything, so phase-split tests require
     * non-zero mode-specific counters plus zero descriptor/copy health errors.
     */
    inline void expectMoERebalancePerfPath(
        const MoEPrefixRestoreParityCase &test_case,
        const std::vector<PerfStatRecord> &records,
        const std::string &context)
    {
        if (!test_case.moe_rebalance)
            return;

        const auto mode = test_case.moe_rebalance->mode;
        if (mode == MoERebalanceRuntimeMode::Dynamic)
        {
            expectPerfCounterPositive(
                records,
                "moe_rebalance",
                "device_rebalance_dynamic_ownership_swap_attempts",
                context);
            expectPerfCounterPositive(
                records,
                "moe_rebalance",
                "device_rebalance_dynamic_ownership_swap_accepts",
                context);
        }
        else if (mode == MoERebalanceRuntimeMode::LLEP)
        {
            expectPerfCounterPositive(
                records,
                "moe_rebalance",
                "device_rebalance_llep_weight_transfer_count",
                context);
            expectPerfCounterPositive(
                records,
                "moe_rebalance",
                "device_rebalance_llep_assignment_span_count",
                context);
            expectPerfCounterPositive(
                records,
                "moe_rebalance",
                "device_rebalance_planned_arrivals",
                context);
            expectPerfCounterPositive(
                records,
                "moe_rebalance",
                "device_rebalance_copy_copied_arrivals",
                context);
            expectPerfCounterPositive(
                records,
                "moe_rebalance",
                "device_rebalance_apply_applied_arrivals",
                context);
        }

        for (const char *error_counter : {
                 "device_rebalance_copy_missing_source_descriptors",
                 "device_rebalance_apply_missing_source_descriptors",
                 "device_rebalance_copy_missing_destination_slots",
                 "device_rebalance_apply_missing_destination_slots",
                 "device_rebalance_copy_descriptor_mismatches",
                 "device_rebalance_apply_descriptor_mismatches",
                 "device_rebalance_copy_invalid_plan_entries",
                 "device_rebalance_apply_invalid_plan_entries",
                 "device_rebalance_controller_last_error_code"})
        {
            expectPerfCounterZero(records, "moe_rebalance", error_counter, context);
        }
    }

    /**
     * @brief Assert that the backend small-M verifier kernels were actually hit.
     *
     * MTP verifier rows are economical only when the backend uses its small-M
     * dispatch path.  These checks are intentionally backend-specific because
     * CUDA and ROCm publish separate kernel counter names.
     */
    inline void expectMoEBackendKernelPerfPath(
        const MoEPrefixRestoreParityCase &test_case,
        const std::vector<PerfStatRecord> &records,
        const std::string &context)
    {
        if (moEPrefixCaseUsesCUDA(test_case))
        {
            expectPerfCounterPositive(
                records,
                "kernel",
                "cuda_native_vnni_small_m_calls",
                context);
        }
        if (moEPrefixCaseUsesROCm(test_case))
        {
            expectPerfCounterPositive(
                records,
                "kernel",
                "rocm_native_vnni_small_m_calls",
                context);
        }
    }

    /**
     * @brief Prove the topology-specific MTP-head ownership policy at runtime.
     *
     * CPU NodeLocalTP deliberately keeps the very large MTP head column-sharded
     * and graph-allgathers only the compact verifier-row logits.  GPU LocalTP
     * mirrors the head on every child, while single-device execution already
     * owns the full vocabulary.  This assertion prevents token parity from
     * passing if NodeLocalTP silently skips its required gather, and also guards
     * against reintroducing a tiny collective into the mirrored ownership modes.
     */
    inline void expectMoEMTPHeadOwnershipPerfPath(
        const MoEPrefixRestoreParityCase &test_case,
        const std::vector<PerfStatRecord> &records,
        const std::string &context)
    {
        constexpr const char *counter_name =
            "global_tp_sidecar_logit_allgathers";
        if (test_case.topology == MoEPrefixParityTopology::NodeLocalTP)
        {
            expectPerfCounterPositive(records, "mtp", counter_name, context);
            return;
        }

        expectPerfCounterZero(records, "mtp", counter_name, context);
    }

    inline bool hasMTPPerfRecordTag(
        const std::vector<PerfStatRecord> &records,
        const char *name,
        const char *tag_key,
        const char *tag_value)
    {
        return std::any_of(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                if (record.domain != "mtp" || record.name != name)
                    return false;
                const auto it = record.tags.find(tag_key);
                return it != record.tags.end() && it->second == tag_value;
            });
    }

    /**
     * @brief Assert that a greedy MoE MTP request used the expected verifier lane.
     *
     * MoE parity compares response tokens against a no-MTP Llaminar baseline,
     * but token equality alone does not prove the request used the intended
     * vLLM-style grouped transaction.  These perfstats assertions make the
     * ExpertOverlay matrix catch accidental regressions away from grouped
     * decode-equivalent verifier rows, while still keeping direct all-position
     * publication disabled.
     *
     * @param test_case MoE parity fixture under test.
     * @param records Perfstats snapshot captured immediately after the request.
     * @param context Human-readable request label for assertion failures.
     */
    inline void expectMoEGreedyMTPPublicationPath(
        const MoEPrefixRestoreParityCase &test_case,
        const std::vector<PerfStatRecord> &records,
        const std::string &context)
    {
        const bool used_retired_serial_replay =
            hasMTPPerfCounter(
                records,
                "decode_equivalent_sequential_verifier_runs");
        const bool used_grouped_host_publication =
            hasMTPPerfCounter(
                records,
                "grouped_outcome_host_publication_uses") &&
            hasMTPPerfCounter(
                records,
                "grouped_outcome_host_state_publications");
        const bool used_grouped_device_publication =
            hasMTPPerfCounter(
                records,
                "grouped_outcome_device_resident_publication_uses") &&
            (hasMTPPerfCounter(records, "spec_state_publications") ||
             hasMTPPerfCounter(records, "spec_state_batch_publications") ||
             hasMTPPerfCounter(
                 records,
                 "rank_grouped_decode_equivalent_spec_state_batch_publications"));
        const bool used_direct_all_position_publication =
            hasMTPPerfCounter(
                records,
                "all_position_state_publication_verifier_runs");
        const bool used_grouped_verifier =
            hasMTPPerfCounter(
                records,
                "grouped_decode_equivalent_greedy_verifier_runs");

        if (moECaseExpectsGroupedOutcomeDevicePublication(test_case))
        {
            EXPECT_TRUE(used_grouped_device_publication)
                << context << " should publish grouped MoE verifier rows "
                   "through the device-resident mailbox.\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_TRUE(used_grouped_verifier)
                << context << " should run grouped greedy MoE verifier rows.\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_FALSE(used_retired_serial_replay)
                << context << " must not use row-serial MoE verifier replay "
                   "when grouped device publication is available.\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_FALSE(used_direct_all_position_publication)
                << context << " must not promote direct all-position MoE "
                   "publication outside the proven lane.\n"
                << PerfStatsCollector::summaryString({"mtp"});
            return;
        }

        EXPECT_TRUE(used_grouped_verifier)
            << context << " should run grouped decode-equivalent MoE verifier rows.\n"
            << PerfStatsCollector::summaryString({"mtp"});
        EXPECT_TRUE(used_grouped_device_publication || used_grouped_host_publication)
            << context << " should publish grouped MoE verifier state through "
               "an explicit grouped publication path.\n"
            << PerfStatsCollector::summaryString({"mtp"});
        EXPECT_FALSE(used_retired_serial_replay)
            << context << " must not use the retired row-serial MoE verifier replay.\n"
            << PerfStatsCollector::summaryString({"mtp"});
        EXPECT_FALSE(used_direct_all_position_publication)
            << context << " must not use unproven MoE direct all-position "
               "publication.\n"
            << PerfStatsCollector::summaryString({"mtp"});
    }

    inline std::map<std::string, std::vector<float>> captureMoERunnerSnapshots(
        IInferenceRunner &runner);
    inline std::map<std::string, std::vector<float>> captureMoERunnerSnapshots(
        IOrchestrationRunner &runner);
    inline std::vector<std::string> qwen36MoEGroupedVerifierSnapshotKeys();
    inline std::vector<std::string> qwen36MoEPrefixInvarianceSnapshotKeys();
    inline std::string compareMoEPrefixLeadingSnapshotRows(
        const std::map<std::string, std::vector<float>> &full_prefill,
        int full_rows,
        const std::map<std::string, std::vector<float>> &prefix_prefill,
        int prefix_rows,
        const std::vector<std::string> &keys);
    inline std::string compareMoEPrefixSuffixSnapshotRows(
        const std::map<std::string, std::vector<float>> &split_suffix,
        const std::map<std::string, std::vector<float>> &restored_suffix,
        int suffix_local_rows,
        int cache_suffix_start_row,
        int cache_suffix_rows,
        const std::vector<std::string> &keys);
    inline std::string describeMoEPrefixLeadingSnapshotRows(
        const std::map<std::string, std::vector<float>> &full_prefill,
        int full_rows,
        const std::map<std::string, std::vector<float>> &prefix_prefill,
        int prefix_rows,
        const std::vector<std::string> &keys);

    inline void runMoEPrefixRestoreParity(
        const MoEPrefixRestoreParityCase &test_case,
        PrefixRestoreParityMode mode)
    {
        ScopedMoEParityProductionMode production_mode(
            shouldForceMoEParityProductionMode(test_case));
        ScopedMoEPrefixCaseEnvironment case_env(test_case.env_overrides);
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        auto phase_start = parityPhaseStart();
        loadMoEReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        if (moeReferenceInputsStoppedCurrentTest())
        {
            return;
        }
        logMoEParityPhase(test_case, "reference-inputs", phase_start);

        constexpr int kPartialPrefixBlockSize = 256;
        if (mode == PrefixRestoreParityMode::PartialHit &&
            test_case.prefix_restore_prompt_token_limit > 0 &&
            static_cast<int>(prompt_tokens.size()) > test_case.prefix_restore_prompt_token_limit)
        {
            ASSERT_GT(test_case.prefix_restore_prompt_token_limit, kPartialPrefixBlockSize)
                << "partial prefix restore prompt cap must leave a non-terminal cached block";
            prompt_tokens.resize(
                static_cast<size_t>(test_case.prefix_restore_prompt_token_limit));
        }
        if (mode == PrefixRestoreParityMode::PartialHit)
        {
            ASSERT_GT(prompt_tokens.size(), static_cast<size_t>(kPartialPrefixBlockSize))
                << "partial prefix restore parity needs a non-terminal cached block";
        }

        const int block_size = mode == PrefixRestoreParityMode::FullHit
                                   ? static_cast<int>(prompt_tokens.size())
                                   : kPartialPrefixBlockSize;
        std::unique_ptr<ScopedEnvironmentValues> partial_prefix_graph_env;
        if (mode == PrefixRestoreParityMode::PartialHit)
        {
            const char *disable_prefill_graphs_env =
                std::getenv("LLAMINAR_PARITY_DISABLE_PREFILL_GRAPHS");
            const bool disable_prefill_graphs =
                disable_prefill_graphs_env &&
                disable_prefill_graphs_env[0] != '\0' &&
                std::strcmp(disable_prefill_graphs_env, "0") != 0 &&
                std::strcmp(disable_prefill_graphs_env, "false") != 0 &&
                std::strcmp(disable_prefill_graphs_env, "FALSE") != 0;
            const int final_remainder =
                static_cast<int>(prompt_tokens.size()) % block_size;
            std::ostringstream buckets;
            if (final_remainder > 0 && final_remainder != block_size)
                buckets << final_remainder << ',';
            buckets << block_size;
            const std::string bucket_sizes = buckets.str();

            /*
             * Production partial-hit parity intentionally exercises bucketed
             * prefill graph capture because prefix restore and long-context
             * replay share that hot path.  The opt-in diagnostic below narrows
             * root-cause work to non-captured prefill without weakening the
             * default gate or silently bypassing graph replay in production.
             */
            if (disable_prefill_graphs)
            {
                partial_prefix_graph_env.reset(new ScopedEnvironmentValues({
                    {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "0"},
                    {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1000000000"},
                }));
            }
            else
            {
                partial_prefix_graph_env.reset(new ScopedEnvironmentValues({
                    {"LLAMINAR_GPU_GRAPHS", "1"},
                    {"LLAMINAR_PREFILL_GRAPH_BUCKETS", "1"},
                    {"LLAMINAR_PREFILL_GRAPH_MIN_SEQ", "1"},
                    {"LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", bucket_sizes.c_str()},
                }));
            }
        }

        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;
        auto decode_current_request =
            [&](IOrchestrationRunner &runner,
                int max_tokens,
                const char *label) -> GenerationResult
        {
            GenerationResult result;
            int remaining = max_tokens;
            runner.setSkipLogitsGatherDecode(true);
            while (remaining > 0)
            {
                runner.setDecodeStepTokenBudget(remaining);
                GenerationResult step = runner.decodeStep();
                runner.setDecodeStepTokenBudget(0);
                if (!step.error.empty())
                {
                    result.error = std::string(label) + ": " + step.error;
                    runner.setSkipLogitsGatherDecode(false);
                    return result;
                }
                if (step.tokens.empty())
                {
                    result.error = std::string(label) + ": decodeStep produced no tokens";
                    runner.setSkipLogitsGatherDecode(false);
                    return result;
                }
                if (static_cast<int>(step.tokens.size()) > remaining)
                {
                    result.error = std::string(label) + ": decodeStep exceeded token budget";
                    runner.setSkipLogitsGatherDecode(false);
                    return result;
                }
                result.tokens.insert(result.tokens.end(), step.tokens.begin(), step.tokens.end());
                remaining -= static_cast<int>(step.tokens.size());
            }
            runner.setSkipLogitsGatherDecode(false);
            return result;
        };

        GenerationResult baseline_result;
        PrefixRuntimeStateSnapshot baseline_snapshot;
        GenerationResult first;
        PrefixRuntimeStateSnapshot after_first;
        GenerationResult second;
        PrefixRuntimeStateSnapshot after_second;
        bool partial_hit_reused_diagnostic_runners = false;
        auto shutdown_and_release_runner =
            [](auto &runner)
        {
            if (runner)
            {
                runner->shutdown();
                runner.reset();
            }
            llaminar::v2::kernels::KernelFactory::clearCache();
        };

        std::vector<int32_t> first_prompt = prompt_tokens;
        if (mode == PrefixRestoreParityMode::PartialHit)
        {
            first_prompt.assign(
                prompt_tokens.begin(),
                prompt_tokens.begin() + block_size);
        }

        if (mode == PrefixRestoreParityMode::PartialHit)
        {
            int prefix_probe_segment_split_tokens = block_size;
            if (const char *split_env =
                    std::getenv("LLAMINAR_PARITY_PREFIX_PROBE_KV_SEGMENT_SPLIT"))
            {
                const int requested_split = std::atoi(split_env);
                if (requested_split > 0)
                    prefix_probe_segment_split_tokens = requested_split;
            }
            const std::string prefix_probe_segment_split =
                std::to_string(prefix_probe_segment_split_tokens);
            ScopedEnvironmentValues probe_hash_env({
                {"LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS", "1"},
                {"LLAMINAR_PREFIX_PROBE_HASH_KV_SEGMENTS", "1"},
                {"LLAMINAR_PREFIX_PROBE_KV_SEGMENT_SPLIT", prefix_probe_segment_split.c_str()},
                {"LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE", "1"},
            });

            PrefixRuntimeStateSnapshot full_prefill_probe;
            std::map<std::string, std::vector<float>> block_prefill_snapshots;
            const auto prefix_invariance_keys = qwen36MoEPrefixInvarianceSnapshotKeys();
            {
                auto prefill_block_baseline = factory->createFromOrchestrationConfig(
                    makeMoEPrefixRestoreConfig(test_case, model_path, false, block_size));
                ASSERT_NE(prefill_block_baseline, nullptr);
                phase_start = parityPhaseStart();
                ASSERT_TRUE(prefill_block_baseline->initialize())
                    << prefill_block_baseline->lastError();
                logMoEParityPhase(
                    test_case,
                    "prefix-prefill-block-baseline.initialize",
                    phase_start);
                ScopedEnvironmentValues kv_snapshot_env({
                    {"LLAMINAR_DEBUG_KV_APPEND_SOURCE_SNAPSHOT", "1"},
                    {"LLAMINAR_DEBUG_KV_CACHE_SNAPSHOT", "1"},
                });
                prefill_block_baseline->enableSnapshotCapture();
                prefill_block_baseline->setSnapshotCaptureFilter(prefix_invariance_keys);
                ASSERT_TRUE(prefill_block_baseline->prefill(first_prompt))
                    << prefill_block_baseline->lastError();
                block_prefill_snapshots =
                    captureMoERunnerSnapshots(*prefill_block_baseline);
                prefill_block_baseline->disableSnapshotCapture();
                shutdown_and_release_runner(prefill_block_baseline);
            }
            {
                auto prefill_baseline = factory->createFromOrchestrationConfig(
                    makeMoEPrefixRestoreConfig(test_case, model_path, false, block_size));
                ASSERT_NE(prefill_baseline, nullptr);
                phase_start = parityPhaseStart();
                ASSERT_TRUE(prefill_baseline->initialize())
                    << prefill_baseline->lastError();
                logMoEParityPhase(
                    test_case,
                    "prefix-prefill-state-baseline.initialize",
                    phase_start);
                prefill_baseline->setSamplingParams(greedy);
                ASSERT_TRUE(prefill_baseline->prefill(prompt_tokens))
                    << prefill_baseline->lastError();
                full_prefill_probe = prefill_baseline->prefixStateProbe();
                phase_start = parityPhaseStart();
                baseline_result = decode_current_request(
                    *prefill_baseline,
                    test_case.decode_steps,
                    "prefix-prefill-state-baseline.decode");
                logMoEParityPhase(
                    test_case,
                    "prefix-prefill-state-baseline.decode",
                    phase_start);
                baseline_snapshot = prefill_baseline->prefixStateProbe();
                shutdown_and_release_runner(prefill_baseline);
            }

            PrefixRuntimeStateSnapshot restored_prefill_probe;
            PrefixRuntimeStateSnapshot seed_prefill_probe;
            std::optional<PrefixRuntimeStateSnapshot> split_prefill_probe;
            std::optional<PrefixRuntimeStateSnapshot> split_seed_probe;
            std::map<std::string, std::vector<float>> seed_prefill_snapshots;
            std::map<std::string, std::vector<float>> restored_suffix_snapshots;
            std::map<std::string, std::vector<float>> split_suffix_snapshots;
            std::map<std::string, std::vector<float>> restored_first_suffix_snapshots;
            std::map<std::string, std::vector<float>> split_first_suffix_snapshots;
            const bool split_prefill_state_diag_enabled =
                std::getenv("LLAMINAR_PARITY_PREFIX_SPLIT_DIAG") != nullptr;
            const bool suffix_snapshot_diag_enabled =
                std::getenv("LLAMINAR_PARITY_PREFIX_SUFFIX_SNAPSHOT_DIAG") != nullptr;
            {
                auto prefill_cached = factory->createFromOrchestrationConfig(
                    makeMoEPrefixRestoreConfig(test_case, model_path, true, block_size));
                ASSERT_NE(prefill_cached, nullptr);
                phase_start = parityPhaseStart();
                ASSERT_TRUE(prefill_cached->initialize())
                    << prefill_cached->lastError();
                logMoEParityPhase(
                    test_case,
                    "prefix-prefill-state-cached.initialize",
                    phase_start);
                prefill_cached->setSamplingParams(greedy);
                PrefixRuntimeStateSnapshot after_prefill_seed;
                {
                    ScopedEnvironmentValues kv_snapshot_env({
                        {"LLAMINAR_DEBUG_KV_APPEND_SOURCE_SNAPSHOT", "1"},
                        {"LLAMINAR_DEBUG_KV_CACHE_SNAPSHOT", "1"},
                    });
                    prefill_cached->enableSnapshotCapture();
                    prefill_cached->setSnapshotCaptureFilter(prefix_invariance_keys);
                    ASSERT_TRUE(prefill_cached->prefill(first_prompt))
                        << prefill_cached->lastError();
                    seed_prefill_snapshots = captureMoERunnerSnapshots(*prefill_cached);
                    after_prefill_seed = prefill_cached->prefixStateProbe();
                    prefill_cached->disableSnapshotCapture();
                }
                seed_prefill_probe = after_prefill_seed;
                EXPECT_TRUE(after_prefill_seed.prefix_cache_ready);
                EXPECT_GE(after_prefill_seed.prefix_cache_inserts, 1u);
                prefill_cached->clearSnapshots();
                phase_start = parityPhaseStart();
                first = decode_current_request(
                    *prefill_cached,
                    test_case.decode_steps,
                    "prefix-prefill-state-cached.first-decode");
                logMoEParityPhase(
                    test_case,
                    "prefix-prefill-state-cached.first-decode",
                    phase_start);
                after_first = prefill_cached->prefixStateProbe();
                if (suffix_snapshot_diag_enabled)
                {
                    prefill_cached->enableSnapshotCapture();
                    prefill_cached->setSnapshotCaptureFilter(prefix_invariance_keys);
                    ASSERT_TRUE(prefill_cached->prefill(prompt_tokens))
                        << prefill_cached->lastError();
                    restored_suffix_snapshots =
                        captureMoERunnerSnapshots(*prefill_cached);
                    prefill_cached->disableSnapshotCapture();
                }
                else
                {
                    ASSERT_TRUE(prefill_cached->prefill(prompt_tokens))
                        << prefill_cached->lastError();
                }
                restored_prefill_probe = prefill_cached->prefixStateProbe();
                phase_start = parityPhaseStart();
                second = decode_current_request(
                    *prefill_cached,
                    test_case.decode_steps,
                    "prefix-prefill-state-cached.second-decode");
                logMoEParityPhase(
                    test_case,
                    "prefix-prefill-state-cached.second-decode",
                    phase_start);
                after_second = prefill_cached->prefixStateProbe();
                shutdown_and_release_runner(prefill_cached);
            }
            if (suffix_snapshot_diag_enabled)
            {
                const int first_suffix_rows = std::min(
                    block_size,
                    static_cast<int>(prompt_tokens.size()) -
                        static_cast<int>(first_prompt.size()));
                ASSERT_GT(first_suffix_rows, 0)
                    << "partial prefix suffix diagnostic requires suffix tokens";
                const auto first_suffix_end =
                    prompt_tokens.begin() +
                    static_cast<std::ptrdiff_t>(first_prompt.size() + first_suffix_rows);
                const std::vector<int32_t> first_suffix_request(
                    prompt_tokens.begin(),
                    first_suffix_end);
                const std::vector<int32_t> first_suffix_prompt(
                    prompt_tokens.begin() + static_cast<std::ptrdiff_t>(first_prompt.size()),
                    first_suffix_end);

                {
                    auto restored_first_suffix =
                        factory->createFromOrchestrationConfig(
                            makeMoEPrefixRestoreConfig(test_case, model_path, true, block_size));
                    ASSERT_NE(restored_first_suffix, nullptr);
                    phase_start = parityPhaseStart();
                    ASSERT_TRUE(restored_first_suffix->initialize())
                        << restored_first_suffix->lastError();
                    logMoEParityPhase(
                        test_case,
                        "prefix-prefill-restored-first-suffix.initialize",
                        phase_start);
                    restored_first_suffix->setSamplingParams(greedy);
                    ASSERT_TRUE(restored_first_suffix->prefill(first_prompt))
                        << restored_first_suffix->lastError();
                    const GenerationResult restored_first_decode =
                        decode_current_request(
                            *restored_first_suffix,
                            test_case.decode_steps,
                            "prefix-prefill-restored-first-suffix.first-decode");
                    ASSERT_TRUE(restored_first_decode.error.empty())
                        << restored_first_decode.error;
                    restored_first_suffix->clearSnapshots();
                    {
                        ScopedEnvironmentValues kv_snapshot_env({
                            {"LLAMINAR_DEBUG_KV_APPEND_SOURCE_SNAPSHOT", "1"},
                            {"LLAMINAR_DEBUG_KV_CACHE_SNAPSHOT", "1"},
                        });
                        restored_first_suffix->enableSnapshotCapture();
                        restored_first_suffix->setSnapshotCaptureFilter(prefix_invariance_keys);
                        ASSERT_TRUE(restored_first_suffix->prefill(first_suffix_request))
                            << restored_first_suffix->lastError();
                        restored_first_suffix_snapshots =
                            captureMoERunnerSnapshots(*restored_first_suffix);
                        restored_first_suffix->disableSnapshotCapture();
                    }
                    shutdown_and_release_runner(restored_first_suffix);
                }

                {
                    auto split_first_suffix =
                        factory->createFromOrchestrationConfig(
                            makeMoEPrefixRestoreConfig(test_case, model_path, false, block_size));
                    ASSERT_NE(split_first_suffix, nullptr);
                    phase_start = parityPhaseStart();
                    ASSERT_TRUE(split_first_suffix->initialize())
                        << split_first_suffix->lastError();
                    logMoEParityPhase(
                        test_case,
                        "prefix-prefill-split-first-suffix.initialize",
                        phase_start);
                    split_first_suffix->setSamplingParams(greedy);
                    ASSERT_TRUE(split_first_suffix->prefill(first_prompt))
                        << split_first_suffix->lastError();
                    split_first_suffix->clearSnapshots();
                    {
                        ScopedEnvironmentValues kv_snapshot_env({
                            {"LLAMINAR_DEBUG_KV_APPEND_SOURCE_SNAPSHOT", "1"},
                            {"LLAMINAR_DEBUG_KV_CACHE_SNAPSHOT", "1"},
                        });
                        split_first_suffix->enableSnapshotCapture();
                        split_first_suffix->setSnapshotCaptureFilter(prefix_invariance_keys);
                        ASSERT_TRUE(split_first_suffix->prefill(first_suffix_prompt))
                            << split_first_suffix->lastError();
                        split_first_suffix_snapshots =
                            captureMoERunnerSnapshots(*split_first_suffix);
                        split_first_suffix->disableSnapshotCapture();
                    }
                    shutdown_and_release_runner(split_first_suffix);
                }
            }
            if (split_prefill_state_diag_enabled || suffix_snapshot_diag_enabled)
            {
                auto split_prefill = factory->createFromOrchestrationConfig(
                    makeMoEPrefixRestoreConfig(test_case, model_path, false, block_size));
                ASSERT_NE(split_prefill, nullptr);
                phase_start = parityPhaseStart();
                ASSERT_TRUE(split_prefill->initialize())
                    << split_prefill->lastError();
                logMoEParityPhase(
                    test_case,
                    "prefix-prefill-split-baseline.initialize",
                    phase_start);
                split_prefill->setSamplingParams(greedy);
                ASSERT_TRUE(split_prefill->prefill(first_prompt))
                    << split_prefill->lastError();
                split_seed_probe = split_prefill->prefixStateProbe();
                std::vector<int32_t> suffix_prompt(
                    prompt_tokens.begin() + static_cast<std::ptrdiff_t>(first_prompt.size()),
                    prompt_tokens.end());
                ASSERT_FALSE(suffix_prompt.empty());
                if (suffix_snapshot_diag_enabled)
                {
                    split_prefill->enableSnapshotCapture();
                    split_prefill->setSnapshotCaptureFilter(prefix_invariance_keys);
                    ASSERT_TRUE(split_prefill->prefill(suffix_prompt))
                        << split_prefill->lastError();
                    split_suffix_snapshots =
                        captureMoERunnerSnapshots(*split_prefill);
                    split_prefill->disableSnapshotCapture();
                }
                else
                {
                    ASSERT_TRUE(split_prefill->prefill(suffix_prompt))
                        << split_prefill->lastError();
                }
                split_prefill_probe = split_prefill->prefixStateProbe();
                shutdown_and_release_runner(split_prefill);
            }
            partial_hit_reused_diagnostic_runners = true;

            EXPECT_FALSE(block_prefill_snapshots.empty())
                << "block prefill produced no prefix-invariance snapshots";
            EXPECT_FALSE(seed_prefill_snapshots.empty())
                << "seed prefill produced no prefix-invariance snapshots";
            auto missing_required_prefix_snapshots =
                [](const std::map<std::string, std::vector<float>> &snapshots,
                   const std::vector<std::string> &keys)
            {
                std::vector<std::string> missing;
                for (const auto &key : keys)
                {
                    if (snapshots.find(key) == snapshots.end())
                        missing.push_back(key);
                }
                return missing;
            };
            const auto block_missing =
                missing_required_prefix_snapshots(block_prefill_snapshots, prefix_invariance_keys);
            const auto seed_missing =
                missing_required_prefix_snapshots(seed_prefill_snapshots, prefix_invariance_keys);
            auto join_missing = [](const std::vector<std::string> &keys)
            {
                std::ostringstream oss;
                for (const auto &key : keys)
                    oss << "\n  " << key;
                return oss.str();
            };
            EXPECT_TRUE(block_missing.empty())
                << "block prefill missing prefix-invariance snapshots:"
                << join_missing(block_missing);
            EXPECT_TRUE(seed_missing.empty())
                << "seed prefill missing prefix-invariance snapshots:"
                << join_missing(seed_missing);
            const std::string prefix_snapshot_mismatch =
                compareMoEPrefixLeadingSnapshotRows(
                    block_prefill_snapshots,
                    block_size,
                    seed_prefill_snapshots,
                    static_cast<int>(first_prompt.size()),
                    prefix_invariance_keys);
            EXPECT_TRUE(prefix_snapshot_mismatch.empty())
                << prefix_snapshot_mismatch;

            MTPRuntimeSnapshotComparisonOptions compare_options;
            /*
             * Prefix-cache replay is a production-state restore contract.  A
             * GPU partial hit must reproduce the same pre-decode KV/GDN state
             * as an uncached full prefill; otherwise later greedy decode can
             * drift even when metadata and prefix boundary snapshots agree.
             */
            compare_options.compare_main_kv_payload_hashes = true;
            compare_options.compare_shifted_mtp_kv = false;
            compare_options.compare_gdn_hashes = true;
            const MTPStateValidationResult prefill_state_match =
                compareMTPRuntimeStateSnapshots(
                    full_prefill_probe,
                    restored_prefill_probe,
                    compare_options);
            std::string split_prefill_diagnostic =
                "split baseline disabled; set LLAMINAR_PARITY_PREFIX_SPLIT_DIAG=1";
            if (split_prefill_probe)
            {
                const MTPStateValidationResult full_vs_split =
                    compareMTPRuntimeStateSnapshots(
                        full_prefill_probe,
                        *split_prefill_probe,
                        compare_options);
                const MTPStateValidationResult split_vs_restored =
                    compareMTPRuntimeStateSnapshots(
                        *split_prefill_probe,
                        restored_prefill_probe,
                        compare_options);
                split_prefill_diagnostic =
                    std::string("full_vs_split=") +
                    (full_vs_split ? "match" : full_vs_split.reason) +
                    "; split_vs_restored=" +
                    (split_vs_restored ? "match" : split_vs_restored.reason);
                if (split_seed_probe)
                {
                    const MTPStateValidationResult seed_vs_split_seed =
                        compareMTPRuntimeStateSnapshots(
                            seed_prefill_probe,
                            *split_seed_probe,
                            compare_options);
                    split_prefill_diagnostic +=
                        "; seed_vs_split_seed=" +
                        std::string(seed_vs_split_seed
                                        ? "match"
                                        : seed_vs_split_seed.reason);
                }
            }
            std::string suffix_snapshot_diagnostic =
                "suffix snapshot diagnostic disabled; set LLAMINAR_PARITY_PREFIX_SUFFIX_SNAPSHOT_DIAG=1";
            if (suffix_snapshot_diag_enabled)
            {
                const int suffix_rows =
                    static_cast<int>(prompt_tokens.size()) -
                    static_cast<int>(first_prompt.size());
                const int first_suffix_rows =
                    std::min(block_size, suffix_rows);
                std::ostringstream suffix_report;
                if (restored_first_suffix_snapshots.empty() ||
                    split_first_suffix_snapshots.empty())
                {
                    suffix_report
                        << "first_suffix_snapshots=captured no comparable snapshots";
                }
                else
                {
                    const std::string first_suffix_mismatch =
                        compareMoEPrefixSuffixSnapshotRows(
                            split_first_suffix_snapshots,
                            restored_first_suffix_snapshots,
                            first_suffix_rows,
                            static_cast<int>(first_prompt.size()),
                            first_suffix_rows,
                            prefix_invariance_keys);
                    suffix_report
                        << "first_suffix_snapshots="
                        << (first_suffix_mismatch.empty()
                                ? std::string("match")
                                : first_suffix_mismatch);
                }
                if (restored_suffix_snapshots.empty() || split_suffix_snapshots.empty())
                {
                    suffix_report
                        << "; final_suffix_snapshots=captured no comparable snapshots";
                }
                else
                {
                    const std::string suffix_mismatch =
                        compareMoEPrefixSuffixSnapshotRows(
                            split_suffix_snapshots,
                            restored_suffix_snapshots,
                            suffix_rows,
                            static_cast<int>(first_prompt.size()),
                            suffix_rows,
                            prefix_invariance_keys);
                    suffix_report
                        << "; final_suffix_snapshots="
                        << (suffix_mismatch.empty()
                                ? std::string("match")
                                : suffix_mismatch);
                }
                suffix_snapshot_diagnostic = suffix_report.str();
            }
            auto summarize_prefill_probe =
                [](const PrefixRuntimeStateSnapshot &probe)
            {
                auto join_ints = [](const std::vector<int> &values)
                {
                    std::ostringstream joined;
                    for (size_t i = 0; i < values.size(); ++i)
                    {
                        if (i > 0)
                            joined << ",";
                        joined << values[i];
                    }
                    return joined.str();
                };
                std::ostringstream oss;
                oss << "pos=" << probe.current_position
                    << " moe_epoch=" << probe.moe_runtime_movement_epoch
                    << " live_epoch=" << probe.live_state_epoch
                    << " live_mutations=" << probe.live_state_mutations
                    << " last_live_mutation=" << probe.last_live_state_mutation_reason
                    << "/" << probe.last_live_state_mutation_operation
                    << " seq=[" << join_ints(probe.sequence_lengths) << "]"
                    << " kv_tokens=" << probe.totalCachedTokens()
                    << " gdn_layers=" << probe.gdn_layers.size();
                if (!probe.kv_caches.empty())
                {
                    const auto &cache = probe.kv_caches.front();
                    const size_t kv_limit =
                        std::min<size_t>(cache.layers.size(), 5);
                    for (size_t i = 0; i < kv_limit; ++i)
                    {
                        const auto &layer = cache.layers[i];
                        oss << " KV" << layer.global_layer
                            << "/S" << layer.seq_idx
                            << "/n=" << layer.cached_tokens
                            << "/k=" << layer.k_payload_hash
                            << "/v=" << layer.v_payload_hash;
                        if (layer.leading_segment_hash_available)
                        {
                            oss << "/lead" << layer.leading_segment_tokens
                                << "=" << layer.leading_k_payload_hash
                                << ":" << layer.leading_v_payload_hash;
                        }
                        if (layer.trailing_segment_hash_available)
                        {
                            oss << "/tail" << layer.trailing_segment_start
                                << "+" << layer.trailing_segment_tokens
                                << "=" << layer.trailing_k_payload_hash
                                << ":" << layer.trailing_v_payload_hash;
                        }
                    }
                }
                const size_t gdn_limit =
                    std::min<size_t>(probe.gdn_layers.size(), 4);
                for (size_t i = 0; i < gdn_limit; ++i)
                {
                    const auto &layer = probe.gdn_layers[i];
                    oss << " L" << layer.global_layer
                        << "/rh=" << layer.recurrence_hash
                        << "/ch=" << layer.conv_hash;
                    if (layer.device_state_hash_available)
                    {
                        oss << "/rdh=" << layer.recurrence_device_hash
                            << "/cdh=" << layer.conv_device_hash;
                    }
                    if (layer.local_device_state_hash_available)
                    {
                        oss << "/rlh=" << layer.recurrence_local_device_hash
                            << "/clh=" << layer.conv_local_device_hash;
                    }
                }
                return oss.str();
            };
            ASSERT_TRUE(prefill_state_match)
                << "partial prefix restore must reproduce full-prefill state before decode: "
                << prefill_state_match.reason
                << "\ncached-block boundary metrics:"
                << describeMoEPrefixLeadingSnapshotRows(
                       block_prefill_snapshots,
                       block_size,
                       seed_prefill_snapshots,
                       static_cast<int>(first_prompt.size()),
                       {"layer3_K_PROJECTION",
                        "layer3_V_PROJECTION",
                        "layer3_K_ROPE",
                        "layer3_KV_APPEND_SOURCE_K",
                        "layer3_KV_APPEND_SOURCE_V",
                        "layer3_KV_CACHE_K",
                        "layer3_KV_CACHE_V",
                        "layer31_K_PROJECTION",
                        "layer31_V_PROJECTION",
                        "layer31_K_ROPE",
                        "layer31_KV_APPEND_SOURCE_K",
                        "layer31_KV_APPEND_SOURCE_V",
                        "layer31_KV_CACHE_K",
                        "layer31_KV_CACHE_V"})
                << "\nfull: "
                << summarize_prefill_probe(full_prefill_probe)
                << "\nseed-block: "
                << summarize_prefill_probe(seed_prefill_probe)
                << (split_seed_probe
                        ? std::string("\nsplit-seed: ") +
                              summarize_prefill_probe(*split_seed_probe)
                        : std::string{})
                << "\nafter-first-decode: "
                << summarize_prefill_probe(after_first)
                << "\nrestored: "
                << summarize_prefill_probe(restored_prefill_probe)
                << "\nsplit-diagnostic: "
                << split_prefill_diagnostic
                << "\nsuffix-snapshot-diagnostic: "
                << suffix_snapshot_diagnostic;
        }

        if (!partial_hit_reused_diagnostic_runners)
        {
            auto baseline = factory->createFromOrchestrationConfig(
                makeMoEPrefixRestoreConfig(test_case, model_path, false, block_size));
            ASSERT_NE(baseline, nullptr);
            phase_start = parityPhaseStart();
            ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
            logMoEParityPhase(test_case, "prefix-baseline.initialize", phase_start);
            phase_start = parityPhaseStart();
            baseline_result = baseline->generate(prompt_tokens, test_case.decode_steps, greedy);
            logMoEParityPhase(test_case, "prefix-baseline.generate", phase_start);
            baseline_snapshot = baseline->prefixStateProbe();
            shutdown_and_release_runner(baseline);
        }

        ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
        ASSERT_EQ(baseline_result.tokens.size(), expected_tokens.size());
        EXPECT_EQ(baseline_snapshot.prefix_cache_hits, 0u);

        // The dedicated Qwen3.6 MoE math parity suite checks PyTorch logits and
        // layer snapshots. Prefix restore correctness is stricter in a different
        // direction: cache-enabled runs must reproduce the no-cache Llaminar
        // greedy stream exactly, including quantized top-1 swaps tolerated by
        // the math harness.
        const auto reference_tokens = baseline_result.tokens;
        if (!partial_hit_reused_diagnostic_runners)
        {
            auto cached = factory->createFromOrchestrationConfig(
                makeMoEPrefixRestoreConfig(test_case, model_path, true, block_size));
            ASSERT_NE(cached, nullptr);
            phase_start = parityPhaseStart();
            ASSERT_TRUE(cached->initialize()) << cached->lastError();
            logMoEParityPhase(test_case, "prefix-cached.initialize", phase_start);

            phase_start = parityPhaseStart();
            first = cached->generate(first_prompt, test_case.decode_steps, greedy);
            logMoEParityPhase(test_case, "prefix-cached.first-generate", phase_start);
            after_first = cached->prefixStateProbe();
            ASSERT_TRUE(first.error.empty()) << first.error;
            EXPECT_TRUE(after_first.prefix_cache_ready);
            EXPECT_GE(after_first.prefix_cache_inserts, 1u);
            if (mode == PrefixRestoreParityMode::FullHit)
            {
                ASSERT_EQ(first.tokens.size(), reference_tokens.size());
                EXPECT_EQ(first.tokens, reference_tokens);
            }

            phase_start = parityPhaseStart();
            second = cached->generate(prompt_tokens, test_case.decode_steps, greedy);
            logMoEParityPhase(test_case, "prefix-cached.second-generate", phase_start);
            after_second = cached->prefixStateProbe();
            shutdown_and_release_runner(cached);
        }
        else
        {
            ASSERT_TRUE(first.error.empty()) << first.error;
            EXPECT_TRUE(after_first.prefix_cache_ready);
            EXPECT_GE(after_first.prefix_cache_inserts, 1u);
        }

        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_EQ(second.tokens.size(), reference_tokens.size());
        auto summarize_prefix_probe_for_token_mismatch =
            [](const PrefixRuntimeStateSnapshot &probe)
        {
            std::ostringstream oss;
            oss << "ready=" << (probe.prefix_cache_ready ? "true" : "false")
                << " pos=" << probe.current_position
                << " moe_epoch=" << probe.moe_runtime_movement_epoch
                << " hits=" << probe.prefix_cache_hits
                << " inserts=" << probe.prefix_cache_inserts
                << " request.hit=" << (probe.prefix_request.hit ? "true" : "false")
                << " request.partial=" << (probe.prefix_request.partial_hit ? "true" : "false")
                << " request.matched=" << probe.prefix_request.matched_tokens
                << " request.terminal_logits="
                << (probe.prefix_request.terminal_logits_restored ? "true" : "false")
                << " kv_tokens=" << probe.totalCachedTokens()
                << " mtp_tokens=" << probe.totalMTPCachedTokens();
            return oss.str();
        };
        auto join_tokens_for_prefix_mismatch =
            [](const std::vector<int32_t> &tokens)
        {
            std::ostringstream oss;
            oss << '[';
            for (size_t i = 0; i < tokens.size(); ++i)
            {
                if (i != 0)
                    oss << ',';
                oss << tokens[i];
            }
            oss << ']';
            return oss.str();
        };
        EXPECT_EQ(second.tokens, reference_tokens)
            << "prefix restore decode stream diverged"
            << "\nexpected fixture tokens: "
            << join_tokens_for_prefix_mismatch(expected_tokens)
            << "\nbaseline tokens: "
            << join_tokens_for_prefix_mismatch(reference_tokens)
            << "\nfirst cached-run tokens: "
            << join_tokens_for_prefix_mismatch(first.tokens)
            << "\nsecond cached-run tokens: "
            << join_tokens_for_prefix_mismatch(second.tokens)
            << "\nbaseline probe: "
            << summarize_prefix_probe_for_token_mismatch(baseline_snapshot)
            << "\nafter first probe: "
            << summarize_prefix_probe_for_token_mismatch(after_first)
            << "\nafter second probe: "
            << summarize_prefix_probe_for_token_mismatch(after_second)
            << "\nmoe_rebalance_perfstats:\n"
            << PerfStatsCollector::summaryString({"moe_rebalance"});
        EXPECT_TRUE(after_second.prefix_cache_ready);
        EXPECT_GE(after_second.prefix_cache_hits, 1u);

        if (mode == PrefixRestoreParityMode::FullHit)
        {
            EXPECT_TRUE(after_second.prefix_request.hit);
            EXPECT_FALSE(after_second.prefix_request.partial_hit);
            EXPECT_EQ(after_second.prefix_request.matched_tokens,
                      static_cast<int>(prompt_tokens.size()));
            EXPECT_TRUE(after_second.prefix_request.terminal_logits_restored);
        }
        else
        {
            EXPECT_FALSE(after_second.prefix_request.hit);
            EXPECT_TRUE(after_second.prefix_request.partial_hit);
            EXPECT_EQ(after_second.prefix_request.matched_tokens, block_size);
            EXPECT_FALSE(after_second.prefix_request.terminal_logits_restored);
        }
    }

    /**
     * @brief Runs greedy MTP parity with optional prefix-cache reuse for a MoE fixture.
     *
     * The baseline runner has MTP disabled and establishes the exact greedy
     * token stream for the current quantized model and topology.  The MTP runner
     * then proves that verifier-side draft acceptance, prefix-cache terminal
     * restoration, and request-local MTP counters preserve that same stream over
     * a fresh request and an optional prefix-cache hit.
     *
     * @param test_case MoE parity fixture, including topology, metadata, and context window.
     * @param enable_prefix_cache Enables the second request that restores prefix and MTP state.
     * @param mtp_draft_tokens Number of speculative draft tokens requested per MTP step.
     */
    inline void runMoEMTPParity(
        const MoEPrefixRestoreParityCase &test_case,
        bool enable_prefix_cache,
        int mtp_draft_tokens = 1)
    {
        ScopedMoEParityProductionMode production_mode(
            shouldForceMoEParityProductionMode(test_case));
        ScopedMoEPrefixCaseEnvironment case_env(test_case.env_overrides);
        ScopedEnvironmentValues perf_stats_enabled({
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        });
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        auto phase_start = parityPhaseStart();
        loadMoEReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        if (moeReferenceInputsStoppedCurrentTest())
        {
            return;
        }
        logMoEParityPhase(test_case, "reference-inputs", phase_start);
        ASSERT_TRUE(moEMTPParityCaseFitsContextWindow(test_case, prompt_tokens.size()));

        const int block_size = enable_prefix_cache
                                   ? static_cast<int>(prompt_tokens.size())
                                   : 2;
        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto baseline = factory->createFromOrchestrationConfig(
            makeMoEPrefixRestoreConfig(test_case, model_path, false, block_size, false));
        ASSERT_NE(baseline, nullptr);
        phase_start = parityPhaseStart();
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        logMoEParityPhase(test_case, "mtp-baseline.initialize", phase_start);
        phase_start = parityPhaseStart();
        auto baseline_result = baseline->generate(prompt_tokens, test_case.decode_steps, greedy);
        logMoEParityPhase(test_case, "mtp-baseline.generate", phase_start);
        const auto baseline_snapshot = baseline->prefixStateProbe();
        baseline->shutdown();

        ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
        ASSERT_EQ(baseline_result.tokens.size(), expected_tokens.size());
        EXPECT_EQ(baseline_snapshot.prefix_cache_hits, 0u);
        EXPECT_EQ(baseline_snapshot.mtp_draft_steps, 0u);

        // MTP greedy verification must preserve the main-model Llaminar greedy
        // stream exactly. PyTorch layer/logit tolerances are enforced by the
        // Qwen3.6 MoE math parity tests.
        const auto reference_tokens = baseline_result.tokens;

        auto mtp = factory->createFromOrchestrationConfig(
            makeMoEPrefixRestoreConfig(
                test_case,
                model_path,
                enable_prefix_cache,
                block_size,
                true,
                mtp_draft_tokens));
        ASSERT_NE(mtp, nullptr);
        phase_start = parityPhaseStart();
        ASSERT_TRUE(mtp->initialize()) << mtp->lastError();
        logMoEParityPhase(test_case, "mtp.initialize", phase_start);

        PerfStatsCollector::reset();
        phase_start = parityPhaseStart();
        auto first = mtp->generate(prompt_tokens, test_case.decode_steps, greedy);
        logMoEParityPhase(test_case, "mtp.first-generate", phase_start);
        const auto after_first = mtp->prefixStateProbe();
        const auto first_records = PerfStatsCollector::snapshot(
            {"mtp", "prefix_cache", "moe_rebalance", "kernel"});
        ASSERT_TRUE(first.error.empty()) << first.error;
        ASSERT_EQ(first.tokens.size(), reference_tokens.size());
        EXPECT_EQ(first.tokens, reference_tokens);
        EXPECT_FALSE(after_first.mtp_bypassed) << after_first.mtp_bypass_reason;
        EXPECT_GE(after_first.mtp_draft_steps, 1u);
        EXPECT_GE(after_first.mtp_verifier_runs, 1u);
        expectMoEGreedyMTPPublicationPath(
            test_case,
            first_records,
            test_case.name + " first request");
        expectMoEBackendKernelPerfPath(
            test_case,
            first_records,
            test_case.name + " first request");

        if (!enable_prefix_cache)
        {
            mtp->shutdown();
            PerfStatsCollector::reset();
            return;
        }

        EXPECT_TRUE(after_first.prefix_cache_ready);
        EXPECT_GE(after_first.prefix_cache_inserts, 1u);
        EXPECT_GT(after_first.prefix_cache_mtp_state_bytes, 0u);

        PerfStatsCollector::reset();
        phase_start = parityPhaseStart();
        auto second = mtp->generate(prompt_tokens, test_case.decode_steps, greedy);
        logMoEParityPhase(test_case, "mtp.second-generate", phase_start);
        const auto after_second = mtp->prefixStateProbe();
        const auto second_records = PerfStatsCollector::snapshot(
            {"mtp", "prefix_cache", "moe_rebalance", "kernel"});
        mtp->shutdown();

        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_EQ(second.tokens.size(), reference_tokens.size());
        EXPECT_EQ(second.tokens, reference_tokens);
        EXPECT_TRUE(after_second.prefix_cache_ready);
        EXPECT_GE(after_second.prefix_cache_hits, 1u);
        EXPECT_TRUE(after_second.prefix_request.hit);
        EXPECT_EQ(after_second.prefix_request.matched_tokens,
                  static_cast<int>(prompt_tokens.size()));
        EXPECT_TRUE(after_second.prefix_request.terminal_logits_restored);
        EXPECT_TRUE(after_second.prefix_request.terminal_hidden_restored);
        EXPECT_TRUE(after_second.prefix_request.mtp_state_restored);
        EXPECT_FALSE(after_second.mtp_bypassed) << after_second.mtp_bypass_reason;
        // MTP counters are request-local: prove the restored-prefix request
        // still ran the verifier instead of expecting cumulative growth.
        const uint64_t expected_second_step_drafts = static_cast<uint64_t>(
            std::min(mtp_draft_tokens, std::max(0, test_case.decode_steps - 1)));
        EXPECT_GE(after_second.mtp_draft_steps, expected_second_step_drafts);
        if (expected_second_step_drafts > 0)
        {
            EXPECT_GE(after_second.mtp_verifier_runs, 1u);
            EXPECT_GE(after_second.mtp_verifier_token_count, expected_second_step_drafts + 1);
        }
        expectMoEGreedyMTPPublicationPath(
            test_case,
            second_records,
            test_case.name + " restored request");
        expectMoEPrefixCachePerfPath(
            second_records,
            test_case.name + " restored request");
        expectMoERebalancePerfPath(
            test_case,
            second_records,
            test_case.name + " restored request");
        expectMoEBackendKernelPerfPath(
            test_case,
            second_records,
            test_case.name + " restored request");
        PerfStatsCollector::reset();
    }

    /**
     * @brief Construct the dynamic-depth policy used by MoE stochastic MTP parity.
     *
     * MoE stochastic validation uses a deliberately short controller window so
     * the dynamic-depth tests prove the controller participates during the
     * integration run.  A fixed-depth-looking dynamic run is a coverage hole.
     */
    inline MTPDepthPolicyConfig qwen36MoEStochasticDynamicDepthPolicy(
        int max_depth = 3)
    {
        MTPDepthPolicyConfig depth_policy;
        depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        depth_policy.min_depth = 1;
        depth_policy.max_depth = std::max(1, max_depth);
        depth_policy.initial_depth = depth_policy.max_depth;
        depth_policy.window_size = 1;
        depth_policy.min_samples = 1;
        depth_policy.cooldown_steps = 0;
        return depth_policy;
    }

    /**
     * @brief Stochastic sampling parameters shared by MoE serial and MTP parity.
     *
     * Keeping the seeded sampling policy in one helper prevents the serial
     * reproducibility regression from drifting away from the MTP verifier cells.
     * These values intentionally exercise temperature, top-k, top-p, and sparse
     * penalty handling instead of collapsing into a greedy argmax check.
     */
    inline SamplingParams qwen36MoEStochasticVerifierSamplingParams()
    {
        SamplingParams stochastic;
        stochastic.temperature = 0.6f;
        stochastic.top_k = 20;
        stochastic.top_p = 0.95f;
        stochastic.presence_penalty = 0.25f;
        stochastic.seed = 123;
        return stochastic;
    }

    /**
     * @brief Prove same-seed serial stochastic decode is request-local.
     *
     * The stochastic MTP oracle is only meaningful when the corresponding
     * no-MTP production decode path is deterministic for the same prompt and
     * seed.  GPU LocalTP once sampled from a process-static RNG in
     * DeviceSampler, so a second no-MTP runner in the same test process could
     * consume a later RNG segment even with the same seed.  This regression
     * exercises the real production GPU decode path and requires the
     * logical-position sampler counter so the test cannot pass by silently
     * routing through a host logits fallback.
     */
    inline void runMoESerialStochasticSameSeedReplay(
        const MoEPrefixRestoreParityCase &test_case)
    {
        ScopedMoEParityProductionMode production_mode(
            shouldForceMoEParityProductionMode(test_case));
        ScopedMoEPrefixCaseEnvironment case_env(test_case.env_overrides);
        ScopedEnvironmentValues graph_env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
            {"LLAMINAR_ROCM_CONCURRENT_M2_ROWS", "0"},
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        });

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        auto phase_start = parityPhaseStart();
        loadMoEReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        if (moeReferenceInputsStoppedCurrentTest())
            return;
        logMoEParityPhase(test_case, "serial-stochastic-reference-inputs", phase_start);

        constexpr int block_size = 2;
        const int stochastic_decode_steps = std::max(2, test_case.decode_steps);
        const SamplingParams stochastic =
            qwen36MoEStochasticVerifierSamplingParams();
        auto factory = createOrchestrationRunnerFactory();
        auto run_once = [&](const char *label)
        {
            auto config =
                makeMoEPrefixRestoreConfig(test_case, model_path, false, block_size, false);
            auto runner = factory->createFromOrchestrationConfig(config);
            EXPECT_NE(runner, nullptr);
            if (!runner)
                return std::vector<int32_t>{};

            auto local_phase_start = parityPhaseStart();
            EXPECT_TRUE(runner->initialize()) << runner->lastError();
            const std::string init_phase = std::string(label) + ".initialize";
            logMoEParityPhase(test_case, init_phase.c_str(), local_phase_start);

            PerfStatsCollector::reset();
            local_phase_start = parityPhaseStart();
            auto result =
                runner->generate(prompt_tokens, stochastic_decode_steps, stochastic);
            const std::string generate_phase = std::string(label) + ".generate";
            logMoEParityPhase(test_case, generate_phase.c_str(), local_phase_start);
            const auto records = PerfStatsCollector::snapshot({"sampling", "mtp"});
            const auto snapshot = runner->prefixStateProbe();
            runner->shutdown();

            EXPECT_TRUE(result.error.empty()) << result.error;
            EXPECT_EQ(result.tokens.size(), static_cast<size_t>(stochastic_decode_steps));
            EXPECT_EQ(snapshot.mtp_draft_steps, 0u);
            EXPECT_EQ(snapshot.mtp_stochastic_accept_tests, 0u);
            EXPECT_GE(
                perfCounterSum(
                    records,
                    "sampling",
                    "main_stochastic_logical_position_samples"),
                static_cast<double>(stochastic_decode_steps))
                << test_case.name
                << " serial stochastic decode must use the logical-position "
                   "production sampler, not process-static RNG or host "
                   "logits fallback";
            return result.tokens;
        };

        const std::vector<int32_t> first = run_once("serial-stochastic.first");
        const std::vector<int32_t> second = run_once("serial-stochastic.second");
        ASSERT_FALSE(first.empty());
        ASSERT_FALSE(second.empty());
        EXPECT_EQ(second, first)
            << test_case.name
            << " serial stochastic decode must be reproducible for two "
               "independent same-seed requests in one process";
        PerfStatsCollector::reset();
    }

    inline void runMoEStochasticMTPVerifierParity(
        const MoEPrefixRestoreParityCase &test_case,
        int draft_depth = 1,
        bool require_stochastic_outcome_after_reuse = false,
        MTPDepthPolicyConfig mtp_depth_policy = {},
        bool enable_prefix_cache = false)
    {
        ScopedMoEParityProductionMode production_mode(
            shouldForceMoEParityProductionMode(test_case));
        ScopedMoEPrefixCaseEnvironment case_env(test_case.env_overrides);
        ASSERT_TRUE(test_case.topology == MoEPrefixParityTopology::SingleDevice ||
                    test_case.topology == MoEPrefixParityTopology::NodeLocalTP ||
                    test_case.topology ==
                        MoEPrefixParityTopology::ExpertOverlayCuda2TPHotOnly ||
                    test_case.topology ==
                        MoEPrefixParityTopology::ExpertOverlayRocm2TPHotOnly)
            << "MoE stochastic MTP verifier parity requires a full local owner, "
               "NodeLocalTP CPU domain, or homogeneous GPU expert-overlay TP domain";

        ScopedEnvironmentValues graph_env({
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_ROCM_CONCURRENT_DECODE", "0"},
            {"LLAMINAR_ROCM_CONCURRENT_M2_ROWS", "0"},
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        });

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        auto phase_start = parityPhaseStart();
        loadMoEReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        if (moeReferenceInputsStoppedCurrentTest())
        {
            return;
        }
        logMoEParityPhase(test_case, "stochastic-reference-inputs", phase_start);

        const int block_size = enable_prefix_cache
                                   ? static_cast<int>(prompt_tokens.size())
                                   : 2;
        const int requested_draft_depth = std::max(1, draft_depth);
        const bool dynamic_depth =
            mtp_depth_policy.mode == MTPDepthPolicyMode::Dynamic;
        /*
         * Reserve one output slot for the first sampled token plus the full
         * requested draft.  Without this lower bound, depth-3 and dynamic-depth
         * matrix cells can be budget-clamped to depth 2 while still reporting
         * successful stochastic generation and token parity.
         */
        const int stochastic_decode_steps = std::max(
            {2, test_case.decode_steps, requested_draft_depth + 1});
        auto factory = createOrchestrationRunnerFactory();

        SamplingParams stochastic = qwen36MoEStochasticVerifierSamplingParams();

        auto baseline_config =
            makeMoEPrefixRestoreConfig(test_case, model_path, false, block_size, false);
        auto baseline = factory->createFromOrchestrationConfig(baseline_config);
        ASSERT_NE(baseline, nullptr);
        phase_start = parityPhaseStart();
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        logMoEParityPhase(test_case, "stochastic-baseline.initialize", phase_start);
        phase_start = parityPhaseStart();
        auto baseline_result =
            baseline->generate(prompt_tokens, stochastic_decode_steps, stochastic);
        logMoEParityPhase(test_case, "stochastic-baseline.generate", phase_start);
        const auto baseline_snapshot = baseline->prefixStateProbe();
        baseline->shutdown();

        ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
        ASSERT_EQ(baseline_result.tokens.size(),
                  static_cast<size_t>(stochastic_decode_steps));
        EXPECT_EQ(baseline_snapshot.mtp_draft_steps, 0u);
        EXPECT_EQ(baseline_snapshot.mtp_stochastic_accept_tests, 0u);

        auto mtp_config =
            makeMoEPrefixRestoreConfig(
                test_case,
                model_path,
                enable_prefix_cache,
                block_size,
                true,
                requested_draft_depth,
                mtp_depth_policy);
        mtp_config.mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;

        auto mtp = factory->createFromOrchestrationConfig(mtp_config);
        ASSERT_NE(mtp, nullptr);
        phase_start = parityPhaseStart();
        ASSERT_TRUE(mtp->initialize()) << mtp->lastError();
        logMoEParityPhase(test_case, "stochastic-mtp.initialize", phase_start);

        PerfStatsCollector::reset();
        ASSERT_TRUE(PerfStatsCollector::isEnabled())
            << "MoE stochastic MTP verifier parity requires perf stats";
        phase_start = parityPhaseStart();
        auto mtp_result =
            mtp->generate(prompt_tokens, stochastic_decode_steps, stochastic);
        logMoEParityPhase(test_case, "stochastic-mtp.first-generate", phase_start);
        const auto after_first_mtp = mtp->prefixStateProbe();
        ASSERT_TRUE(mtp_result.error.empty()) << mtp_result.error;
        ASSERT_EQ(mtp_result.tokens.size(), static_cast<size_t>(stochastic_decode_steps));
        EXPECT_EQ(mtp_result.tokens, baseline_result.tokens)
            << "MoE stochastic MTP must match serial stochastic decode for "
               "the same seed on the first request";
        if (enable_prefix_cache)
        {
            EXPECT_TRUE(after_first_mtp.prefix_cache_ready);
            EXPECT_GE(after_first_mtp.prefix_cache_inserts, 1u);
            EXPECT_GT(after_first_mtp.prefix_cache_mtp_state_bytes, 0u);

            PerfStatsCollector::reset();
            phase_start = parityPhaseStart();
            auto restored_mtp_result =
                mtp->generate(prompt_tokens, stochastic_decode_steps, stochastic);
            logMoEParityPhase(
                test_case,
                "stochastic-mtp.prefix-restored-generate",
                phase_start);
            const auto after_restored_mtp = mtp->prefixStateProbe();
            const auto restored_records = PerfStatsCollector::snapshot(
                {"mtp", "prefix_cache", "moe_rebalance", "kernel"});

            ASSERT_TRUE(restored_mtp_result.error.empty())
                << restored_mtp_result.error;
            ASSERT_EQ(restored_mtp_result.tokens.size(), mtp_result.tokens.size());
            EXPECT_EQ(restored_mtp_result.tokens, baseline_result.tokens)
                << "MoE stochastic MTP after prefix restore must match serial "
                   "stochastic decode for the same seed";
            EXPECT_EQ(restored_mtp_result.tokens, mtp_result.tokens)
                << "MoE stochastic MTP with the same seed must be reproducible "
                   "after a prefix-cache restore";
            EXPECT_TRUE(after_restored_mtp.prefix_cache_ready);
            EXPECT_GE(after_restored_mtp.prefix_cache_hits, 1u);
            EXPECT_TRUE(after_restored_mtp.prefix_request.hit);
            EXPECT_EQ(after_restored_mtp.prefix_request.matched_tokens,
                      static_cast<int>(prompt_tokens.size()));
            EXPECT_TRUE(after_restored_mtp.prefix_request.terminal_logits_restored);
            EXPECT_TRUE(after_restored_mtp.prefix_request.terminal_hidden_restored);
            EXPECT_TRUE(after_restored_mtp.prefix_request.mtp_state_restored);
            EXPECT_FALSE(after_restored_mtp.mtp_bypassed)
                << after_restored_mtp.mtp_bypass_reason;
            EXPECT_GE(after_restored_mtp.mtp_stochastic_accept_tests, 1u);
            expectMoEPrefixCachePerfPath(
                restored_records,
                test_case.name + " stochastic restored-prefix request");
            if (test_case.topology == MoEPrefixParityTopology::NodeLocalTP &&
                !moEPrefixCaseUsesGPU(test_case))
            {
                expectPerfCounterPositive(
                    restored_records,
                    "prefix_cache",
                    "cpu_global_tp_terminal_logits_allgathers",
                    test_case.name +
                        " stochastic restored-prefix terminal logits");
            }
            expectMoERebalancePerfPath(
                test_case,
                restored_records,
                test_case.name + " stochastic restored-prefix request");
            expectMoEBackendKernelPerfPath(
                test_case,
                restored_records,
                test_case.name + " stochastic restored-prefix request");
            expectMoEMTPHeadOwnershipPerfPath(
                test_case,
                restored_records,
                test_case.name + " stochastic restored-prefix request");
            if (test_case.topology ==
                    MoEPrefixParityTopology::ExpertOverlayCuda2TPHotOnly ||
                test_case.topology ==
                    MoEPrefixParityTopology::ExpertOverlayRocm2TPHotOnly)
            {
                EXPECT_TRUE(hasMTPPerfRecordTag(
                    restored_records,
                    "gpu_decode_transaction_position_initializations",
                    "source",
                    "prefix_terminal_restore"))
                    << "GPU prefix restore must initialize MTP graph planning "
                       "from the scheduler-known prompt length, never from a "
                       "backend host position mirror.\n"
                    << PerfStatsCollector::summaryString({"mtp"});
                EXPECT_TRUE(hasMTPPerfRecordTag(
                    restored_records,
                    "rank_resident_logical_state_aggregate_invalidations",
                    "lifecycle",
                    "populate_prefix"))
                    << "LocalTP prefix replacement must explicitly retire the "
                       "rank aggregate that named the previous request's child "
                       "logical-state mailboxes.\n"
                    << PerfStatsCollector::summaryString({"mtp"});
                EXPECT_TRUE(hasMTPPerfCounter(
                    restored_records,
                    "gpu_decode_transaction_position_reads"))
                    << "GPU MTP after prefix restore must consume the "
                       "orchestration-owned transaction position.\n"
                    << PerfStatsCollector::summaryString({"mtp"});
                EXPECT_FALSE(hasMTPPerfCounter(
                    restored_records,
                    "cpu_decode_position_planning_reads"))
                    << "GPU MTP planning must never fall through to the CPU "
                       "runner-position path.\n"
                    << PerfStatsCollector::summaryString({"mtp"});
            }
        }

        mtp->clearCache();
        PerfStatsCollector::reset();
        phase_start = parityPhaseStart();
        auto reused_mtp_result =
            mtp->generate(prompt_tokens, stochastic_decode_steps, stochastic);
        logMoEParityPhase(test_case, "stochastic-mtp.reused-generate", phase_start);
        const auto after_reused_mtp = mtp->prefixStateProbe();
        const auto phase138_records =
            PerfStatsCollector::snapshot(
                {"mtp", "prefix_cache", "moe_rebalance", "kernel"});
        mtp->shutdown();

        ASSERT_TRUE(reused_mtp_result.error.empty()) << reused_mtp_result.error;
        ASSERT_EQ(reused_mtp_result.tokens.size(), mtp_result.tokens.size());
        EXPECT_EQ(reused_mtp_result.tokens, baseline_result.tokens)
            << "MoE stochastic MTP after clearCache() must match serial "
               "stochastic decode for the same seed";
        EXPECT_EQ(reused_mtp_result.tokens, mtp_result.tokens)
            << "MoE stochastic MTP with the same seed must be reproducible after clearCache()";
        expectMoEMTPHeadOwnershipPerfPath(
            test_case,
            phase138_records,
            test_case.name + " stochastic post-clearCache request");
        if (enable_prefix_cache &&
            test_case.topology == MoEPrefixParityTopology::NodeLocalTP &&
            !moEPrefixCaseUsesGPU(test_case))
        {
            expectPerfCounterPositive(
                phase138_records,
                "prefix_cache",
                "cpu_global_tp_terminal_logits_allgathers",
                test_case.name +
                    " stochastic post-clearCache terminal logits");
        }
        EXPECT_FALSE(after_reused_mtp.mtp_bypassed)
            << after_reused_mtp.mtp_bypass_reason;
        EXPECT_EQ(after_reused_mtp.mtp_request.verify_mode, "speculative-sampling");
        EXPECT_TRUE(after_reused_mtp.mtp_request.stochastic_verify);
        if (dynamic_depth)
        {
            EXPECT_TRUE(after_reused_mtp.mtp_request.adaptive_depth_enabled);
            EXPECT_EQ(after_reused_mtp.mtp_request.depth_policy_mode, "dynamic");
            EXPECT_GE(after_reused_mtp.mtp_depth_policy_windows, 1u);
            EXPECT_GE(after_reused_mtp.mtp_min_depth, mtp_depth_policy.min_depth);
            EXPECT_EQ(after_reused_mtp.mtp_max_depth, mtp_depth_policy.max_depth);
            EXPECT_GE(after_reused_mtp.mtp_current_depth, mtp_depth_policy.min_depth);
            EXPECT_LE(after_reused_mtp.mtp_current_depth, mtp_depth_policy.max_depth);
        }
        else
        {
            EXPECT_FALSE(after_reused_mtp.mtp_request.adaptive_depth_enabled);
            EXPECT_EQ(after_reused_mtp.mtp_max_depth, requested_draft_depth);
        }
        EXPECT_EQ(after_reused_mtp.mtp_transaction_validation_failures, 0u)
            << test_case.name
            << " MoE stochastic MTP hit MTP transaction validation failures";
        EXPECT_GE(after_reused_mtp.mtp_draft_steps, 1u);
        EXPECT_GE(after_reused_mtp.mtp_verifier_runs, 1u);
        EXPECT_GE(
            after_reused_mtp.mtp_verifier_token_count,
            static_cast<uint64_t>(requested_draft_depth + 1))
            << test_case.name
            << " stochastic depth coverage must include one complete verifier "
               "at the requested depth, not only a budget-clamped prefix";
        EXPECT_GE(after_reused_mtp.mtp_stochastic_accept_tests, 1u);
        EXPECT_EQ(after_reused_mtp.mtp_stochastic_accept_tests,
                  after_reused_mtp.mtp_stochastic_accepts +
                      after_reused_mtp.mtp_stochastic_residual_samples);
        EXPECT_GE(after_reused_mtp.mtp_stochastic_residual_samples +
                      after_reused_mtp.mtp_stochastic_terminal_samples,
                  1u);
        EXPECT_EQ(after_reused_mtp.mtp_request.stochastic_accept_tests,
                  after_reused_mtp.mtp_stochastic_accept_tests);
        EXPECT_EQ(after_reused_mtp.mtp_request.stochastic_accepts,
                  after_reused_mtp.mtp_stochastic_accepts);
        EXPECT_EQ(after_reused_mtp.mtp_request.stochastic_residual_samples,
                  after_reused_mtp.mtp_stochastic_residual_samples);
        EXPECT_EQ(after_reused_mtp.mtp_request.stochastic_terminal_samples,
                  after_reused_mtp.mtp_stochastic_terminal_samples);
        EXPECT_GE(after_reused_mtp.mtp_request.stochastic_acceptance_rate, 0.0);
        EXPECT_LE(after_reused_mtp.mtp_request.stochastic_acceptance_rate, 1.0);
        if (after_reused_mtp.mtp_stochastic_accept_tests > 0)
        {
            const double expected_rate =
                static_cast<double>(after_reused_mtp.mtp_stochastic_accepts) /
                static_cast<double>(after_reused_mtp.mtp_stochastic_accept_tests);
            EXPECT_NEAR(after_reused_mtp.mtp_request.stochastic_acceptance_rate,
                        expected_rate,
                        1e-12);
        }
        if (require_stochastic_outcome_after_reuse)
        {
            EXPECT_GT(after_reused_mtp.mtp_stochastic_accept_tests, 0u)
                << "Fixed-depth stochastic MTP must run at least one verifier "
                   "acceptance test after clearCache().\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_GT(after_reused_mtp.mtp_stochastic_residual_samples +
                          after_reused_mtp.mtp_stochastic_terminal_samples,
                      0u)
                << "Fixed-depth stochastic MTP must exercise stochastic "
                   "correction or terminal sampling after clearCache(); token "
                   "equality alone is not proof that the stochastic verifier "
                   "path ran.\n"
                << PerfStatsCollector::summaryString({"mtp"});
        }

        const bool used_grouped_stochastic_verifier =
            hasMTPPerfCounter(
                phase138_records,
                "grouped_decode_equivalent_stochastic_verifier_runs");
        const bool used_grouped_outcome_device_publication =
            hasMTPPerfCounter(
                phase138_records,
                "grouped_decode_equivalent_stochastic_verifier_runs") &&
            hasMTPPerfCounter(
                phase138_records,
                "grouped_outcome_device_resident_publication_uses") &&
            (hasMTPPerfCounter(phase138_records, "spec_state_publications") ||
             hasMTPPerfCounter(phase138_records, "spec_state_batch_publications") ||
             hasMTPPerfCounter(
                 phase138_records,
                 "rank_grouped_decode_equivalent_spec_state_batch_publications"));
        const bool used_all_position_publication =
            hasMTPPerfCounter(
                phase138_records,
                "all_position_state_publication_verifier_runs") &&
            hasMTPPerfCounter(phase138_records, "spec_state_publications");
        if (moECaseExpectsAllPositionSpecPublication(test_case))
        {
            EXPECT_TRUE(used_all_position_publication)
                << "GPU Qwen3.6 MoE stochastic MTP must exercise vLLM-style "
                   "all-position state publication\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_FALSE(used_grouped_stochastic_verifier)
                << "GPU Qwen3.6 MoE stochastic MTP must not also run grouped "
                   "decode-equivalent stochastic verification once direct "
                   "all-position publication is available\n"
                << PerfStatsCollector::summaryString({"mtp"});
        }
        else if (moECaseExpectsGroupedOutcomeDevicePublication(test_case))
        {
            EXPECT_TRUE(used_grouped_outcome_device_publication)
                << "GPU Qwen3.6 MoE stochastic MTP must exercise the grouped "
                   "decode-equivalent device-resident publication path\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_FALSE(used_all_position_publication)
                << "GPU Qwen3.6 MoE stochastic MTP must not silently switch "
                   "back to direct all-position publication; grouped outcome "
                   "is the proven MoE contract\n"
                << PerfStatsCollector::summaryString({"mtp"});
        }
        else
        {
            EXPECT_TRUE(used_grouped_stochastic_verifier)
                << "CPU Qwen3.6 MoE stochastic MTP must use grouped "
                   "decode-equivalent verification while direct all-position "
                   "publication is not advertised\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_FALSE(used_all_position_publication)
                << "CPU Qwen3.6 MoE stochastic MTP must not publish from an "
                   "unproven multi-row all-position verifier\n"
                << PerfStatsCollector::summaryString({"mtp"});
        }

        const bool homogeneous_gpu_local_tp =
            test_case.topology ==
                MoEPrefixParityTopology::ExpertOverlayCuda2TPHotOnly ||
            test_case.topology ==
                MoEPrefixParityTopology::ExpertOverlayRocm2TPHotOnly;
        if (homogeneous_gpu_local_tp && requested_draft_depth > 1)
        {
            EXPECT_TRUE(hasMTPPerfCounter(
                phase138_records,
                "device_resident_shifted_mtp_kv_commit_boundaries"))
                << "Depth>1 LocalTP MTP must commit at least one shifted-cache "
                   "suffix row from the canonical device boundary. This is a "
                   "mandatory production operation, independent of whether the "
                   "sampled draft is ultimately accepted.\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_TRUE(hasMTPPerfCounter(
                phase138_records,
                "device_resident_logical_state_mailboxes"))
                << "After a LocalTP shifted-cache suffix append advances the "
                   "live-state epoch, grouped accepted-state publication must "
                   "create a fresh arena-backed logical-state mailbox. The "
                   "retired shifted-KV mailbox was not the owner of this "
                   "request-lifetime state.\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_FALSE(hasMTPPerfCounter(
                phase138_records,
                "device_resident_logical_state_mailbox_retargets"))
                << "Grouped device-outcome commits are followed immediately "
                   "by accepted-state publication, which owns a fresh logical "
                   "mailbox for the new epoch. Retargeting the pre-publication "
                   "mailbox would add a redundant event and preserve a stale "
                   "lifecycle instead of exercising the economical production "
                   "path.\n"
                << PerfStatsCollector::summaryString({"mtp"});
            EXPECT_TRUE(hasMTPPerfCounter(
                phase138_records,
                "speculative_shifted_kv_token_position_fusions"))
                << "Depth>1 LocalTP MTP must prepare accepted suffix tokens and "
                   "absolute positions in the fused GPU primitive from compact "
                   "outcome metadata plus the resident pre-verifier base row. "
                   "Seeing publication counters without this counter would "
                   "permit a host-derived position path to return unnoticed.\n"
                << PerfStatsCollector::summaryString({"mtp"});
        }

        const bool used_retired_phase138_stochastic_candidate =
            hasMTPPerfCounter(
                phase138_records,
                "phase138_stochastic_spec_decode_runs");
        EXPECT_FALSE(used_retired_phase138_stochastic_candidate)
            << "Stateful Qwen3.6 MoE stochastic MTP must not use the retired "
               "accepted-count publication candidate\n"
            << PerfStatsCollector::summaryString({"mtp"});
    }

    inline void runMoEGreedyFreshRunnerDeterminism(
        const MoEPrefixRestoreParityCase &test_case)
    {
        ScopedMoEParityProductionMode production_mode(
            shouldForceMoEParityProductionMode(test_case));
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        auto phase_start = parityPhaseStart();
        loadMoEReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        if (moeReferenceInputsStoppedCurrentTest())
        {
            return;
        }
        logMoEParityPhase(test_case, "determinism.reference-inputs", phase_start);

        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        struct FreshRunnerTrace
        {
            GenerationResult result;
            std::vector<int> logits_argmax;
            std::vector<std::string> logits_topk;
        };

        auto local_argmax = [](const float *logits, int vocab_size) -> int
        {
            if (!logits || vocab_size <= 0)
                return -1;
            return static_cast<int>(std::max_element(logits, logits + vocab_size) - logits);
        };

        auto local_topk = [](const float *logits, int vocab_size, int k = 8) -> std::string
        {
            if (!logits || vocab_size <= 0 || k <= 0)
                return "<no logits>";
            std::vector<int> indices(static_cast<size_t>(vocab_size));
            std::iota(indices.begin(), indices.end(), 0);
            const int limit = std::min(k, vocab_size);
            std::partial_sort(
                indices.begin(),
                indices.begin() + limit,
                indices.end(),
                [logits](int lhs, int rhs)
                {
                    if (logits[lhs] == logits[rhs])
                        return lhs < rhs;
                    return logits[lhs] > logits[rhs];
                });

            std::ostringstream oss;
            for (int i = 0; i < limit; ++i)
            {
                if (i > 0)
                    oss << ", ";
                const int idx = indices[static_cast<size_t>(i)];
                oss << idx << ":" << logits[idx];
            }
            return oss.str();
        };

        auto trace_string = [](const FreshRunnerTrace &trace) -> std::string
        {
            std::ostringstream oss;
            oss << "tokens={";
            for (size_t i = 0; i < trace.result.tokens.size(); ++i)
            {
                if (i > 0)
                    oss << ", ";
                oss << trace.result.tokens[i];
            }
            oss << "}";
            for (size_t i = 0; i < trace.logits_topk.size(); ++i)
            {
                oss << "\n  step " << i
                    << " sampled="
                    << (i < trace.result.tokens.size() ? trace.result.tokens[i] : -1)
                    << " argmax="
                    << (i < trace.logits_argmax.size() ? trace.logits_argmax[i] : -1)
                    << " topk=[" << trace.logits_topk[i] << "]";
            }
            return oss.str();
        };

        auto run_once = [&](const char *phase) -> FreshRunnerTrace
        {
            FreshRunnerTrace trace;
            auto runner = factory->createFromOrchestrationConfig(
                makeMoEPrefixRestoreConfig(test_case, model_path, false, 2, false));
            EXPECT_NE(runner, nullptr);
            if (!runner)
            {
                trace.result.error = "failed to create runner";
                return trace;
            }
            auto start = parityPhaseStart();
            EXPECT_TRUE(runner->initialize()) << runner->lastError();
            logMoEParityPhase(test_case, phase, start);
            runner->setSamplingParams(greedy);
            runner->setSkipLogitsGatherDecode(false);
            if (!runner->prefill(prompt_tokens))
            {
                trace.result.error = runner->lastError();
                runner->shutdown();
                return trace;
            }
            const int vocab_size = runner->vocabSize();
            for (int step = 0; step < test_case.decode_steps; ++step)
            {
                auto step_result = runner->decodeStep();
                if (!step_result.error.empty())
                {
                    trace.result.error = step_result.error;
                    break;
                }
                if (!step_result.tokens.empty())
                {
                    trace.result.tokens.insert(
                        trace.result.tokens.end(),
                        step_result.tokens.begin(),
                        step_result.tokens.end());
                }
                const float *logits = runner->lastLogits();
                trace.logits_argmax.push_back(local_argmax(logits, vocab_size));
                trace.logits_topk.push_back(local_topk(logits, vocab_size));
            }
            runner->shutdown();
            return trace;
        };

        auto first = run_once("determinism.first.initialize");
        ASSERT_TRUE(first.result.error.empty()) << first.result.error;
        ASSERT_EQ(first.result.tokens.size(), expected_tokens.size())
            << trace_string(first);

        auto second = run_once("determinism.second.initialize");
        ASSERT_TRUE(second.result.error.empty()) << second.result.error;
        ASSERT_EQ(second.result.tokens.size(), first.result.tokens.size())
            << "first:\n"
            << trace_string(first)
            << "\nsecond:\n"
            << trace_string(second);
        EXPECT_EQ(second.result.tokens, first.result.tokens)
            << "first:\n"
            << trace_string(first)
            << "\nsecond:\n"
            << trace_string(second);
    }

    inline int argmaxToken(const float *logits, int vocab_size)
    {
        if (!logits || vocab_size <= 0)
            return -1;

        return static_cast<int>(std::max_element(logits, logits + vocab_size) - logits);
    }

    inline std::string topKSummary(const float *logits, int vocab_size, int k = 5)
    {
        if (!logits || vocab_size <= 0 || k <= 0)
            return "<no logits>";

        std::vector<int> indices(static_cast<size_t>(vocab_size));
        std::iota(indices.begin(), indices.end(), 0);
        const int limit = std::min(k, vocab_size);
        std::partial_sort(
            indices.begin(),
            indices.begin() + limit,
            indices.end(),
            [logits](int lhs, int rhs)
            {
                if (logits[lhs] == logits[rhs])
                    return lhs < rhs;
                return logits[lhs] > logits[rhs];
            });

        std::ostringstream oss;
        for (int i = 0; i < limit; ++i)
        {
            if (i > 0)
                oss << ", ";
            oss << indices[static_cast<size_t>(i)]
                << ":" << logits[indices[static_cast<size_t>(i)]];
        }
        return oss.str();
    }

    struct VerifierLogitMetrics
    {
        double cosine = 1.0;
        double rel_l2 = 0.0;
        double max_abs_diff = 0.0;
        size_t max_abs_index = 0;
        double symmetric_kl = 0.0;
    };

    inline VerifierLogitMetrics computeVerifierLogitMetrics(
        const float *actual_logits,
        const float *serial_logits,
        int vocab_size)
    {
        VerifierLogitMetrics metrics;
        if (!actual_logits || !serial_logits || vocab_size <= 0)
        {
            metrics.cosine = 0.0;
            metrics.rel_l2 = std::numeric_limits<double>::infinity();
            metrics.max_abs_diff = std::numeric_limits<double>::infinity();
            metrics.symmetric_kl = std::numeric_limits<double>::infinity();
            return metrics;
        }

        double dot = 0.0;
        double actual_norm = 0.0;
        double serial_norm = 0.0;
        double diff_norm = 0.0;
        float actual_max = actual_logits[0];
        float serial_max = serial_logits[0];
        for (int i = 0; i < vocab_size; ++i)
        {
            actual_max = std::max(actual_max, actual_logits[i]);
            serial_max = std::max(serial_max, serial_logits[i]);
            const double actual = static_cast<double>(actual_logits[i]);
            const double serial = static_cast<double>(serial_logits[i]);
            const double diff = actual - serial;
            dot += actual * serial;
            actual_norm += actual * actual;
            serial_norm += serial * serial;
            diff_norm += diff * diff;
            const double abs_diff = std::abs(diff);
            if (abs_diff > metrics.max_abs_diff)
            {
                metrics.max_abs_diff = abs_diff;
                metrics.max_abs_index = static_cast<size_t>(i);
            }
        }

        const double denom = std::sqrt(actual_norm * serial_norm);
        metrics.cosine = denom > 0.0 ? dot / denom : 1.0;
        metrics.rel_l2 = serial_norm > 0.0 ? std::sqrt(diff_norm / serial_norm)
                                           : std::sqrt(diff_norm);

        std::vector<double> actual_probs(static_cast<size_t>(vocab_size), 0.0);
        std::vector<double> serial_probs(static_cast<size_t>(vocab_size), 0.0);
        double actual_sum = 0.0;
        double serial_sum = 0.0;
        for (int i = 0; i < vocab_size; ++i)
        {
            const double actual =
                std::exp(static_cast<double>(actual_logits[i] - actual_max));
            const double serial =
                std::exp(static_cast<double>(serial_logits[i] - serial_max));
            actual_probs[static_cast<size_t>(i)] = actual;
            serial_probs[static_cast<size_t>(i)] = serial;
            actual_sum += actual;
            serial_sum += serial;
        }

        /*
         * KL is evaluated on the full softmax distribution.  Use a tiny floor
         * only to keep the diagnostic finite when a backend underflows a tail
         * probability differently; the floor is far below meaningful mass.
         */
        constexpr double probability_floor = 1.0e-300;
        double actual_to_serial = 0.0;
        double serial_to_actual = 0.0;
        for (int i = 0; i < vocab_size; ++i)
        {
            const double p = std::max(
                actual_probs[static_cast<size_t>(i)] / actual_sum,
                probability_floor);
            const double q = std::max(
                serial_probs[static_cast<size_t>(i)] / serial_sum,
                probability_floor);
            actual_to_serial += p * std::log(p / q);
            serial_to_actual += q * std::log(q / p);
        }
        metrics.symmetric_kl = 0.5 * (actual_to_serial + serial_to_actual);
        return metrics;
    }

    inline ::testing::AssertionResult verifierLogitsNumericallyEquivalent(
        const float *actual_logits,
        const float *serial_logits,
        int vocab_size,
        const std::string &label,
        double min_cosine = 0.99995,
        double max_rel_l2 = 0.005,
        double max_symmetric_kl = 1.0e-4)
    {
        const VerifierLogitMetrics metrics =
            computeVerifierLogitMetrics(actual_logits, serial_logits, vocab_size);
        if (metrics.cosine >= min_cosine &&
            metrics.rel_l2 <= max_rel_l2 &&
            metrics.symmetric_kl <= max_symmetric_kl)
        {
            return ::testing::AssertionSuccess();
        }

        return ::testing::AssertionFailure()
               << label
               << " logit distribution drift: cosine=" << metrics.cosine
               << " rel_l2=" << metrics.rel_l2
               << " symmetric_kl=" << metrics.symmetric_kl
               << " max_abs_diff=" << metrics.max_abs_diff
               << " max_abs_index=" << metrics.max_abs_index
               << " thresholds(cosine>=" << min_cosine
               << ", rel_l2<=" << max_rel_l2
               << ", symmetric_kl<=" << max_symmetric_kl << ")";
    }

    inline ::testing::AssertionResult verifierLogitsByteIdentical(
        const float *actual_logits,
        const float *serial_logits,
        int vocab_size,
        const std::string &label)
    {
        const size_t count = static_cast<size_t>(vocab_size);
        if (std::memcmp(actual_logits, serial_logits, count * sizeof(float)) == 0)
        {
            return ::testing::AssertionSuccess();
        }

        size_t first_mismatch = 0;
        while (first_mismatch < count &&
               std::memcmp(actual_logits + first_mismatch,
                           serial_logits + first_mismatch,
                           sizeof(float)) == 0)
        {
            ++first_mismatch;
        }

        uint32_t actual_bits = 0;
        uint32_t serial_bits = 0;
        if (first_mismatch < count)
        {
            std::memcpy(&actual_bits,
                        actual_logits + first_mismatch,
                        sizeof(actual_bits));
            std::memcpy(&serial_bits,
                        serial_logits + first_mismatch,
                        sizeof(serial_bits));
        }

        const VerifierLogitMetrics metrics =
            computeVerifierLogitMetrics(
                actual_logits,
                serial_logits,
                vocab_size);
        return ::testing::AssertionFailure()
               << label
               << " must be byte-identical to serial decode"
               << " first_mismatch=" << first_mismatch
               << " actual=" << (first_mismatch < count ? actual_logits[first_mismatch] : 0.0f)
               << " serial=" << (first_mismatch < count ? serial_logits[first_mismatch] : 0.0f)
               << " actual_bits=0x" << std::hex << actual_bits
               << " serial_bits=0x" << serial_bits << std::dec
               << " cosine=" << metrics.cosine
               << " rel_l2=" << metrics.rel_l2
               << " symmetric_kl=" << metrics.symmetric_kl
               << " max_abs_diff=" << metrics.max_abs_diff
               << " max_abs_index=" << metrics.max_abs_index;
    }

    /**
     * @brief Summarize live prefix state for decode-equivalence failures.
     *
     * The grouped-verifier regressions we care about are usually coherence bugs:
     * host-visible sequence position, KV ring heads, MTP KV heads, or GDN
     * recurrence/short-conv hashes have diverged between a live decode and a
     * restore-and-replay decode.  Keep the diagnostic compact enough for CTest
     * logs while preserving the fields that tell us which state owner drifted.
     */
    inline std::string summarizeMoEPrefixRuntimeProbe(
        const PrefixRuntimeStateSnapshot &probe)
    {
        auto join_ints = [](const std::vector<int> &values)
        {
            std::string out;
            for (size_t i = 0; i < values.size(); ++i)
            {
                if (i > 0)
                    out += ",";
                out += std::to_string(values[i]);
            }
            return out.empty() ? std::string("none") : out;
        };

        auto summarize_cache =
            [](const std::vector<PrefixKVCacheProbe> &caches)
        {
            std::string out;
            for (const auto &cache : caches)
            {
                if (!out.empty())
                    out += ";";
                out += cache.owner + ":";
                const size_t limit = std::min<size_t>(cache.layers.size(), 4);
                for (size_t i = 0; i < limit; ++i)
                {
                    if (i > 0)
                        out += ",";
                    const auto &layer = cache.layers[i];
                    out += "L" + std::to_string(layer.global_layer) +
                           "=" + std::to_string(layer.cached_tokens) +
                           "@" + std::to_string(layer.ring_head);
                }
                if (cache.layers.size() > limit)
                    out += ",...";
            }
            return out.empty() ? std::string("none") : out;
        };

        std::string gdn;
        const size_t gdn_limit = std::min<size_t>(probe.gdn_layers.size(), 4);
        for (size_t i = 0; i < gdn_limit; ++i)
        {
            if (i > 0)
                gdn += ",";
            const auto &layer = probe.gdn_layers[i];
            gdn += "L" + std::to_string(layer.global_layer) +
                   "/rh=" + std::to_string(layer.recurrence_hash) +
                   "/ch=" + std::to_string(layer.conv_hash);
            if (layer.device_state_hash_available)
            {
                gdn += "/rdh=" +
                       std::to_string(layer.recurrence_device_hash) +
                       "/cdh=" +
                       std::to_string(layer.conv_device_hash);
            }
        }
        if (probe.gdn_layers.size() > gdn_limit)
            gdn += ",...";
        if (gdn.empty())
            gdn = "none";

        return "pos=" + std::to_string(probe.current_position) +
               " live_epoch=" + std::to_string(probe.live_state_epoch) +
               " seq=[" + join_ints(probe.sequence_lengths) + "]" +
               " kv={" + summarize_cache(probe.kv_caches) + "}" +
               " mtp={" + summarize_cache(probe.mtp_kv_caches) + "}" +
               " gdn={" + gdn + "}";
    }

    inline std::string csvEscapeMoEDiagnostic(const std::string &value)
    {
        bool needs_quotes = false;
        for (char c : value)
        {
            if (c == ',' || c == '"' || c == '\n' || c == '\r')
            {
                needs_quotes = true;
                break;
            }
        }

        if (!needs_quotes)
        {
            return value;
        }

        std::string escaped;
        escaped.reserve(value.size() + 2);
        escaped.push_back('"');
        for (char c : value)
        {
            if (c == '"')
            {
                escaped.push_back('"');
            }
            escaped.push_back(c);
        }
        escaped.push_back('"');
        return escaped;
    }

    inline std::string joinTokensMoEDiagnostic(const std::vector<int32_t> &tokens)
    {
        std::ostringstream oss;
        for (size_t i = 0; i < tokens.size(); ++i)
        {
            if (i > 0)
            {
                oss << ' ';
            }
            oss << tokens[i];
        }
        return oss.str();
    }

    inline std::string joinStringsMoEDiagnostic(const std::vector<std::string> &values)
    {
        std::ostringstream oss;
        for (size_t i = 0; i < values.size(); ++i)
        {
            if (i > 0)
            {
                oss << ' ';
            }
            oss << values[i];
        }
        return oss.str();
    }

    inline std::string currentGitHashMoEDiagnostic()
    {
        std::string hash = "unknown";
        FILE *pipe = popen("git rev-parse --short HEAD 2>/dev/null", "r");
        if (!pipe)
        {
            return hash;
        }

        char buf[64];
        if (fgets(buf, sizeof(buf), pipe))
        {
            hash = buf;
            while (!hash.empty() && (hash.back() == '\n' || hash.back() == '\r'))
            {
                hash.pop_back();
            }
        }
        pclose(pipe);
        return hash;
    }

    inline std::filesystem::path moeDiagnosticResultsDir()
    {
        const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_name = "unknown";
        if (info)
        {
            test_name = std::string(info->test_suite_name() ? info->test_suite_name() : "suite") +
                        "_" +
                        std::string(info->name() ? info->name() : "test");
        }

        std::string safe_name;
        safe_name.reserve(test_name.size());
        for (char c : test_name)
        {
            switch (c)
            {
            case '/':
            case '\\':
            case ':':
            case '*':
            case '?':
            case '"':
            case '<':
            case '>':
            case '|':
                safe_name.push_back('_');
                break;
            default:
                safe_name.push_back(c);
                break;
            }
        }

        std::filesystem::path this_file(__FILE__);
        const auto parity_dir = this_file.parent_path().parent_path();
        const auto dir = parity_dir / "results" / currentGitHashMoEDiagnostic() / safe_name;
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        return dir;
    }

    struct MoESnapshotCompareRow
    {
        std::string comparison = "baseline_vs_mtp";
        int sync_idx = 0;
        int output_tokens = 0;
        std::string key;
        std::string reference_key;
        size_t elements = 0;
        double cosine = 0.0;
        double rel_l2 = 0.0;
        double symmetric_kl = 0.0;
        double max_abs_diff = 0.0;
        double left_l2 = 0.0;
        double right_l2 = 0.0;
        double left_mean = 0.0;
        double right_mean = 0.0;
        std::string left_label = "baseline";
        std::string right_label = "mtp";
        bool present_in_baseline = false;
        bool present_in_mtp = false;
        bool byte_identical = false;
        size_t first_mismatch_index = std::numeric_limits<size_t>::max();
        uint32_t first_left_bits = 0;
        uint32_t first_right_bits = 0;
    };

    /**
     * @brief Record the exact FP32 byte comparison for one diagnostic row.
     *
     * Numeric metrics remain useful for estimating how quickly a mismatch is
     * amplifying, but grouped verifier publication requires the production
     * M>1 row to be identical to the backend's production M=1 row.  Capturing
     * the first raw word here makes every caller report the same exact oracle
     * and gives a focused kernel regression the bit patterns it must reproduce.
     *
     * @param row Diagnostic record that will receive the exact result.
     * @param left First FP32 vector.
     * @param right Second FP32 vector.
     * @param size Number of FP32 elements in both vectors.
     */
    inline void recordMoESnapshotByteComparison(
        MoESnapshotCompareRow *row,
        const float *left,
        const float *right,
        size_t size)
    {
        if (!row || !left || !right || size == 0)
        {
            return;
        }

        row->byte_identical =
            std::memcmp(left, right, size * sizeof(float)) == 0;
        if (row->byte_identical)
        {
            return;
        }

        for (size_t i = 0; i < size; ++i)
        {
            uint32_t left_bits = 0;
            uint32_t right_bits = 0;
            std::memcpy(&left_bits, left + i, sizeof(left_bits));
            std::memcpy(&right_bits, right + i, sizeof(right_bits));
            if (left_bits == right_bits)
            {
                continue;
            }

            row->first_mismatch_index = i;
            row->first_left_bits = left_bits;
            row->first_right_bits = right_bits;
            return;
        }
    }

    /**
     * @brief Compare two stage snapshots as softmax distributions.
     *
     * Most MoE diagnostics compare hidden states rather than logits, so KL is
     * not a replacement for cosine or relative L2. It is still a useful
     * companion metric: applying a softmax exposes broad distribution-shape
     * drift that a top-token or max-absolute check can miss, while remaining
     * finite for arbitrary signed stage outputs.
     */
    inline double computeMoESnapshotSymmetricKL(
        const float *left,
        const float *right,
        size_t size)
    {
        if (!left || !right || size == 0)
            return std::numeric_limits<double>::infinity();

        float left_max = left[0];
        float right_max = right[0];
        for (size_t i = 1; i < size; ++i)
        {
            left_max = std::max(left_max, left[i]);
            right_max = std::max(right_max, right[i]);
        }

        std::vector<double> left_probs(size, 0.0);
        std::vector<double> right_probs(size, 0.0);
        double left_sum = 0.0;
        double right_sum = 0.0;
        for (size_t i = 0; i < size; ++i)
        {
            const double l =
                std::exp(static_cast<double>(left[i] - left_max));
            const double r =
                std::exp(static_cast<double>(right[i] - right_max));
            left_probs[i] = l;
            right_probs[i] = r;
            left_sum += l;
            right_sum += r;
        }

        constexpr double probability_floor = 1.0e-300;
        double left_to_right = 0.0;
        double right_to_left = 0.0;
        for (size_t i = 0; i < size; ++i)
        {
            const double p = std::max(left_probs[i] / left_sum, probability_floor);
            const double q = std::max(right_probs[i] / right_sum, probability_floor);
            left_to_right += p * std::log(p / q);
            right_to_left += q * std::log(q / p);
        }
        return 0.5 * (left_to_right + right_to_left);
    }

    struct MoEDiagnosticSnapshot
    {
        std::vector<int32_t> emitted_tokens;
        std::vector<int32_t> total_tokens;
        int current_position = 0;
        int logits_argmax = -1;
        std::string logits_top5;
        PrefixRuntimeStateSnapshot state;
        std::map<std::string, std::vector<float>> snapshots;
    };

    inline MoESnapshotCompareRow compareMoESnapshotKey(
        IOrchestrationRunner &baseline,
        IOrchestrationRunner &mtp,
        int sync_idx,
        int output_tokens,
        const std::string &key)
    {
        MoESnapshotCompareRow row;
        row.comparison = "baseline_vs_mtp";
        row.sync_idx = sync_idx;
        row.output_tokens = output_tokens;
        row.key = key;
        row.reference_key = key;
        row.left_label = "baseline";
        row.right_label = "mtp";

        size_t baseline_size = 0;
        size_t mtp_size = 0;
        const float *baseline_data = baseline.getSnapshot(key, baseline_size);
        const float *mtp_data = mtp.getSnapshot(key, mtp_size);
        row.present_in_baseline = baseline_data != nullptr && baseline_size > 0;
        row.present_in_mtp = mtp_data != nullptr && mtp_size > 0;
        if (!row.present_in_baseline || !row.present_in_mtp || baseline_size != mtp_size)
        {
            row.elements = std::max(baseline_size, mtp_size);
            return row;
        }

        row.elements = baseline_size;
        recordMoESnapshotByteComparison(
            &row,
            baseline_data,
            mtp_data,
            baseline_size);
        double dot = 0.0;
        double baseline_norm = 0.0;
        double mtp_norm = 0.0;
        double baseline_sum = 0.0;
        double mtp_sum = 0.0;
        double diff_norm = 0.0;
        for (size_t i = 0; i < baseline_size; ++i)
        {
            const double a = static_cast<double>(baseline_data[i]);
            const double b = static_cast<double>(mtp_data[i]);
            const double diff = a - b;
            dot += a * b;
            baseline_norm += a * a;
            mtp_norm += b * b;
            baseline_sum += a;
            mtp_sum += b;
            diff_norm += diff * diff;
            row.max_abs_diff = std::max(row.max_abs_diff, std::abs(diff));
        }

        row.left_l2 = std::sqrt(baseline_norm);
        row.right_l2 = std::sqrt(mtp_norm);
        row.left_mean = baseline_sum / static_cast<double>(baseline_size);
        row.right_mean = mtp_sum / static_cast<double>(baseline_size);
        const double denom = row.left_l2 * row.right_l2;
        row.cosine = denom > 0.0 ? dot / denom : 1.0;
        row.rel_l2 = baseline_norm > 0.0 ? std::sqrt(diff_norm / baseline_norm)
                                         : std::sqrt(diff_norm);
        row.symmetric_kl = computeMoESnapshotSymmetricKL(
            baseline_data,
            mtp_data,
            baseline_size);
        return row;
    }

    inline MoESnapshotCompareRow compareMoESnapshotKey(
        const MoEDiagnosticSnapshot &baseline,
        IOrchestrationRunner &mtp,
        int sync_idx,
        int output_tokens,
        const std::string &key)
    {
        MoESnapshotCompareRow row;
        row.comparison = "baseline_vs_mtp";
        row.sync_idx = sync_idx;
        row.output_tokens = output_tokens;
        row.key = key;
        row.reference_key = key;
        row.left_label = "baseline";
        row.right_label = "mtp";

        const auto baseline_it = baseline.snapshots.find(key);
        size_t mtp_size = 0;
        const float *mtp_data = mtp.getSnapshot(key, mtp_size);
        row.present_in_baseline = baseline_it != baseline.snapshots.end() &&
                                  !baseline_it->second.empty();
        row.present_in_mtp = mtp_data != nullptr && mtp_size > 0;
        if (!row.present_in_baseline || !row.present_in_mtp ||
            baseline_it->second.size() != mtp_size)
        {
            row.elements = std::max(
                baseline_it == baseline.snapshots.end() ? size_t{0} : baseline_it->second.size(),
                mtp_size);
            return row;
        }

        const auto &baseline_data = baseline_it->second;
        row.elements = baseline_data.size();
        recordMoESnapshotByteComparison(
            &row,
            baseline_data.data(),
            mtp_data,
            baseline_data.size());
        double dot = 0.0;
        double baseline_norm = 0.0;
        double mtp_norm = 0.0;
        double baseline_sum = 0.0;
        double mtp_sum = 0.0;
        double diff_norm = 0.0;
        for (size_t i = 0; i < baseline_data.size(); ++i)
        {
            const double a = static_cast<double>(baseline_data[i]);
            const double b = static_cast<double>(mtp_data[i]);
            const double diff = a - b;
            dot += a * b;
            baseline_norm += a * a;
            mtp_norm += b * b;
            baseline_sum += a;
            mtp_sum += b;
            diff_norm += diff * diff;
            row.max_abs_diff = std::max(row.max_abs_diff, std::abs(diff));
        }

        row.left_l2 = std::sqrt(baseline_norm);
        row.right_l2 = std::sqrt(mtp_norm);
        row.left_mean = baseline_sum / static_cast<double>(baseline_data.size());
        row.right_mean = mtp_sum / static_cast<double>(baseline_data.size());
        const double denom = row.left_l2 * row.right_l2;
        row.cosine = denom > 0.0 ? dot / denom : 1.0;
        row.rel_l2 = baseline_norm > 0.0 ? std::sqrt(diff_norm / baseline_norm)
                                         : std::sqrt(diff_norm);
        row.symmetric_kl = computeMoESnapshotSymmetricKL(
            baseline_data.data(),
            mtp_data,
            baseline_data.size());
        return row;
    }

    inline std::vector<float> loadMoEPyTorchSnapshot(
        const std::filesystem::path &snapshot_dir,
        const std::string &key)
    {
        const auto npy_path = snapshot_dir / (key + ".npy");
        try
        {
            cnpy::NpyArray arr = cnpy::npy_load(npy_path.string());
            std::vector<float> data;
            if (arr.word_size == sizeof(float))
            {
                const float *ptr = arr.data<float>();
                data.assign(ptr, ptr + arr.num_vals);
                return data;
            }
            if (arr.word_size == sizeof(double))
            {
                const double *ptr = arr.data<double>();
                data.resize(arr.num_vals);
                for (size_t i = 0; i < arr.num_vals; ++i)
                {
                    data[i] = static_cast<float>(ptr[i]);
                }
                return data;
            }
        }
        catch (const std::exception &)
        {
            return {};
        }
        return {};
    }

    inline std::vector<std::string> listMoEPyTorchSnapshotKeysForDecodeStep(
        const std::filesystem::path &snapshot_dir,
        int decode_step)
    {
        std::vector<std::string> keys;
        const std::string prefix = "decode_step" + std::to_string(decode_step) + "_";
        std::error_code ec;
        for (const auto &entry : std::filesystem::directory_iterator(snapshot_dir, ec))
        {
            if (ec || !entry.is_regular_file())
            {
                continue;
            }
            const auto path = entry.path();
            if (path.extension() != ".npy")
            {
                continue;
            }
            const std::string stem = path.stem().string();
            if (stem.rfind(prefix, 0) != 0)
            {
                continue;
            }
            keys.push_back(stem.substr(prefix.size()));
        }
        std::sort(keys.begin(), keys.end());
        return keys;
    }

    inline std::string pytorchReferenceKeyForMoEDiagnosticKey(const std::string &key)
    {
        static constexpr const char *kDecodeSidecarPrefix = "MTP_DECODE_SIDECAR_";
        const std::string prefix(kDecodeSidecarPrefix);
        if (key.rfind(prefix, 0) == 0)
        {
            return key.substr(prefix.size());
        }
        return key;
    }

    inline MoESnapshotCompareRow compareMoESnapshotVectors(
        const std::vector<float> &left,
        const float *right,
        size_t right_size,
        int sync_idx,
        int output_tokens,
        const std::string &key,
        const std::string &reference_key,
        const std::string &comparison,
        const std::string &left_label,
        const std::string &right_label)
    {
        MoESnapshotCompareRow row;
        row.comparison = comparison;
        row.sync_idx = sync_idx;
        row.output_tokens = output_tokens;
        row.key = key;
        row.reference_key = reference_key;
        row.left_label = left_label;
        row.right_label = right_label;
        row.present_in_baseline = !left.empty();
        row.present_in_mtp = right != nullptr && right_size > 0;

        const float *left_data = left.data();
        size_t left_size = left.size();
        std::vector<float> left_tail;
        if (row.present_in_baseline && row.present_in_mtp &&
            left_size > right_size && right_size > 0 &&
            left_size % right_size == 0)
        {
            left_tail.assign(
                left.end() - static_cast<ptrdiff_t>(right_size),
                left.end());
            left_data = left_tail.data();
            left_size = left_tail.size();
        }

        if (!row.present_in_baseline || !row.present_in_mtp || left_size != right_size)
        {
            row.elements = std::max(left_size, right_size);
            return row;
        }

        row.elements = left_size;
        recordMoESnapshotByteComparison(
            &row,
            left_data,
            right,
            left_size);
        double dot = 0.0;
        double left_norm = 0.0;
        double right_norm = 0.0;
        double left_sum = 0.0;
        double right_sum = 0.0;
        double diff_norm = 0.0;
        for (size_t i = 0; i < left_size; ++i)
        {
            const double a = static_cast<double>(left_data[i]);
            const double b = static_cast<double>(right[i]);
            const double diff = a - b;
            dot += a * b;
            left_norm += a * a;
            right_norm += b * b;
            left_sum += a;
            right_sum += b;
            diff_norm += diff * diff;
            row.max_abs_diff = std::max(row.max_abs_diff, std::abs(diff));
        }

        row.left_l2 = std::sqrt(left_norm);
        row.right_l2 = std::sqrt(right_norm);
        row.left_mean = left_sum / static_cast<double>(left_size);
        row.right_mean = right_sum / static_cast<double>(left_size);
        const double denom = row.left_l2 * row.right_l2;
        row.cosine = denom > 0.0 ? dot / denom : 1.0;
        row.rel_l2 = left_norm > 0.0 ? std::sqrt(diff_norm / left_norm)
                                     : std::sqrt(diff_norm);
        row.symmetric_kl = computeMoESnapshotSymmetricKL(
            left_data,
            right,
            left_size);
        return row;
    }

    inline MoEDiagnosticSnapshot captureMoEDiagnosticSnapshot(
        IOrchestrationRunner &runner,
        const std::vector<int32_t> &emitted_tokens,
        const std::vector<int32_t> &total_tokens)
    {
        MoEDiagnosticSnapshot snapshot;
        snapshot.emitted_tokens = emitted_tokens;
        snapshot.total_tokens = total_tokens;
        snapshot.state = runner.prefixStateProbe();
        snapshot.current_position = snapshot.state.current_position;
        const int vocab_size = runner.vocabSize();
        const float *logits = runner.lastLogits();
        snapshot.logits_argmax = argmaxToken(logits, vocab_size);
        snapshot.logits_top5 = topKSummary(logits, vocab_size, 5);

        for (const auto &key : runner.getSnapshotKeys())
        {
            size_t size = 0;
            const float *data = runner.getSnapshot(key, size);
            if (!data || size == 0)
            {
                continue;
            }
            snapshot.snapshots.emplace(key, std::vector<float>(data, data + size));
        }
        return snapshot;
    }

    inline void writeMoESnapshotCsvHeader(std::ofstream &csv)
    {
        csv << "comparison,sync_idx,output_tokens,key,reference_key,elements,"
               "cosine,rel_l2,symmetric_kl,max_abs_diff,left_l2,right_l2,left_mean,right_mean,left_label,right_label,"
               "present_left,present_right,byte_identical,first_mismatch_index,first_left_bits,first_right_bits\n";
    }

    inline void writeMoESnapshotCsvRow(std::ofstream &csv, const MoESnapshotCompareRow &row)
    {
        csv << csvEscapeMoEDiagnostic(row.comparison) << ','
            << row.sync_idx << ','
            << row.output_tokens << ','
            << csvEscapeMoEDiagnostic(row.key) << ','
            << csvEscapeMoEDiagnostic(row.reference_key) << ','
            << row.elements << ','
            << row.cosine << ','
            << row.rel_l2 << ','
            << row.symmetric_kl << ','
            << row.max_abs_diff << ','
            << row.left_l2 << ','
            << row.right_l2 << ','
            << row.left_mean << ','
            << row.right_mean << ','
            << csvEscapeMoEDiagnostic(row.left_label) << ','
            << csvEscapeMoEDiagnostic(row.right_label) << ','
            << (row.present_in_baseline ? "true" : "false") << ','
            << (row.present_in_mtp ? "true" : "false") << ','
            << (row.byte_identical ? "true" : "false") << ',';
        if (row.first_mismatch_index == std::numeric_limits<size_t>::max())
        {
            csv << "";
        }
        else
        {
            csv << row.first_mismatch_index;
        }
        csv << ','
            << "0x" << std::hex << std::setw(8) << std::setfill('0')
            << row.first_left_bits << ','
            << "0x" << std::setw(8) << row.first_right_bits
            << std::dec << std::setfill(' ') << '\n';
        csv.flush();
    }

    inline std::vector<std::string> unionSnapshotKeys(
        const std::vector<std::string> &lhs,
        const std::vector<std::string> &rhs)
    {
        std::set<std::string> keys(lhs.begin(), lhs.end());
        keys.insert(rhs.begin(), rhs.end());
        return {keys.begin(), keys.end()};
    }

    inline const MoESnapshotCompareRow *worstMoESnapshotRow(
        const std::vector<MoESnapshotCompareRow> &rows)
    {
        const MoESnapshotCompareRow *worst = nullptr;
        for (const auto &row : rows)
        {
            if (!row.present_in_baseline || !row.present_in_mtp)
            {
                return &row;
            }
            if (!worst || row.cosine < worst->cosine)
            {
                worst = &row;
            }
        }
        return worst;
    }

    inline bool isMTPSidecarSnapshotKey(const std::string &key)
    {
        return key.rfind("MTP", 0) == 0 || key.find("_MTP") != std::string::npos;
    }

    inline bool isComparableMoEDiagnosticRow(const MoESnapshotCompareRow &row)
    {
        return row.present_in_baseline && row.present_in_mtp && row.elements > 0;
    }

    inline const MoESnapshotCompareRow *firstMoEDiagnosticDivergence(
        const std::vector<MoESnapshotCompareRow> &rows,
        const std::string &comparison,
        double cosine_threshold,
        bool include_sidecar_keys)
    {
        for (const auto &row : rows)
        {
            if (row.comparison != comparison)
            {
                continue;
            }
            if (!include_sidecar_keys && isMTPSidecarSnapshotKey(row.key))
            {
                continue;
            }
            if (!isComparableMoEDiagnosticRow(row) || row.cosine < cosine_threshold)
            {
                return &row;
            }
        }
        return nullptr;
    }

    /**
     * @brief Return the first non-byte-identical operation in graph order.
     *
     * The caller is responsible for supplying rows in graph execution order.
     * This exact predicate is intentionally separate from the cosine-based
     * diagnostic above: cosine answers how large drift has become, while this
     * function identifies the operation that first violated grouped decode's
     * publication contract.
     */
    inline const MoESnapshotCompareRow *firstMoEDiagnosticByteDivergence(
        const std::vector<MoESnapshotCompareRow> &rows,
        const std::string &comparison,
        bool include_sidecar_keys)
    {
        for (const auto &row : rows)
        {
            if (row.comparison != comparison)
            {
                continue;
            }
            if (!include_sidecar_keys && isMTPSidecarSnapshotKey(row.key))
            {
                continue;
            }
            if (!isComparableMoEDiagnosticRow(row) || !row.byte_identical)
            {
                return &row;
            }
        }
        return nullptr;
    }

    inline const MoESnapshotCompareRow *worstComparableMoEDiagnosticRow(
        const std::vector<MoESnapshotCompareRow> &rows,
        const std::string &comparison,
        bool include_sidecar_keys)
    {
        const MoESnapshotCompareRow *worst = nullptr;
        for (const auto &row : rows)
        {
            if (row.comparison != comparison)
            {
                continue;
            }
            if (!include_sidecar_keys && isMTPSidecarSnapshotKey(row.key))
            {
                continue;
            }
            if (!isComparableMoEDiagnosticRow(row))
            {
                continue;
            }
            if (!worst || row.cosine < worst->cosine)
            {
                worst = &row;
            }
        }
        return worst;
    }

    inline const MoESnapshotCompareRow *firstMoESidecarReferenceGap(
        const std::vector<MoESnapshotCompareRow> &rows,
        const std::string &comparison)
    {
        for (const auto &row : rows)
        {
            if (row.comparison == comparison &&
                isMTPSidecarSnapshotKey(row.key) &&
                (!row.present_in_baseline || !row.present_in_mtp))
            {
                return &row;
            }
        }
        return nullptr;
    }

    inline std::string describeMoEDiagnosticRow(const MoESnapshotCompareRow &row)
    {
        std::ostringstream oss;
        oss << "comparison=" << row.comparison
            << " sync=" << row.sync_idx
            << " output_tokens=" << row.output_tokens
            << " key=" << row.key
            << " reference_key=" << row.reference_key
            << " cosine=" << row.cosine
            << " rel_l2=" << row.rel_l2
            << " symmetric_kl=" << row.symmetric_kl
            << " max_abs_diff=" << row.max_abs_diff
            << " present_left=" << (row.present_in_baseline ? "true" : "false")
            << " present_right=" << (row.present_in_mtp ? "true" : "false")
            << " byte_identical=" << (row.byte_identical ? "true" : "false");
        if (row.first_mismatch_index != std::numeric_limits<size_t>::max())
        {
            oss << " first_mismatch_index=" << row.first_mismatch_index
                << " left_bits=0x" << std::hex << std::setw(8)
                << std::setfill('0') << row.first_left_bits
                << " right_bits=0x" << std::setw(8)
                << row.first_right_bits << std::dec << std::setfill(' ');
        }
        return oss.str();
    }

    inline std::string describeMoEDiagnosticRows(const std::vector<MoESnapshotCompareRow> &rows)
    {
        const auto *worst = worstMoESnapshotRow(rows);
        if (!worst)
        {
            return "no comparable snapshot rows";
        }

        std::ostringstream oss;
        oss << "worst snapshot " << describeMoEDiagnosticRow(*worst);

        if (const auto *first_pt_mtp =
                firstMoEDiagnosticDivergence(rows, "pytorch_vs_mtp", 0.98, false))
        {
            oss << "\nfirst non-sidecar PyTorch-vs-MTP divergence "
                << describeMoEDiagnosticRow(*first_pt_mtp);
        }
        if (const auto *worst_pt_mtp =
                worstComparableMoEDiagnosticRow(rows, "pytorch_vs_mtp", false))
        {
            oss << "\nworst comparable non-sidecar PyTorch-vs-MTP row "
                << describeMoEDiagnosticRow(*worst_pt_mtp);
        }
        if (const auto *first_sidecar =
                firstMoESidecarReferenceGap(rows, "pytorch_vs_mtp"))
        {
            oss << "\nfirst sidecar PyTorch reference gap "
                << describeMoEDiagnosticRow(*first_sidecar);
        }
        return oss.str();
    }

    inline std::map<std::string, std::vector<float>> captureMoERunnerSnapshots(
        IInferenceRunner &runner)
    {
        std::map<std::string, std::vector<float>> snapshots;
        const auto *rank_runner = dynamic_cast<const RankOrchestrator *>(&runner);
        for (const auto &key : runner.getSnapshotKeys())
        {
            size_t size = 0;
            const float *data = nullptr;
            TPSnapshot tp_snapshot;
            if (rank_runner)
            {
                tp_snapshot = rank_runner->getTPSnapshot(key);
                data = tp_snapshot.getCombinedData(size);
            }
            else
            {
                data = runner.getSnapshot(key, size);
            }
            if (!data || size == 0)
            {
                continue;
            }
            snapshots.emplace(key, std::vector<float>(data, data + size));
        }
        return snapshots;
    }

    inline std::map<std::string, std::vector<float>> captureMoERunnerSnapshots(
        IOrchestrationRunner &runner)
    {
        std::map<std::string, std::vector<float>> snapshots;
        for (const auto &key : runner.getSnapshotKeys())
        {
            size_t size = 0;
            const float *data = runner.getSnapshot(key, size);
            if (!data || size == 0)
            {
                continue;
            }
            snapshots.emplace(key, std::vector<float>(data, data + size));
        }
        return snapshots;
    }

    /**
     * @brief Return stage snapshot keys that localize prefix-restore drift.
     *
     * Prefix restore failures frequently appear first as a KV payload mismatch
     * after suffix prefill.  The first full-attention layer proves that the
     * restored prefix boundary was imported correctly, while the last
     * full-attention layer catches late GDN/MoE/RoPE drift that can leave early
     * KV rows looking healthy.  Keeping both layers in the diagnostic filter
     * makes the failure report identify whether K projection, RoPE, append
     * source, or KV cache publication was the first divergent boundary.
     *
     * @return Ordered snapshot keys requested by prefix-invariance diagnostics.
     */
    inline std::vector<std::string> qwen36MoEPrefixInvarianceSnapshotKeys()
    {
        std::vector<std::string> keys = {"EMBEDDING"};
        for (int layer = 0; layer < 3; ++layer)
        {
            const std::string prefix = "layer" + std::to_string(layer);
            keys.push_back(prefix + "_ATTENTION_NORM");
            keys.push_back(prefix + "_QKV_PROJECTION");
            keys.push_back(prefix + "_GDN_Z_PROJECTION");
            keys.push_back(prefix + "_GDN_ALPHA");
            keys.push_back(prefix + "_GDN_BETA");
            keys.push_back(prefix + "_GDN_CONV1D_OUTPUT");
            keys.push_back(prefix + "_GDN_DELTA_RULE_OUTPUT");
            keys.push_back(prefix + "_GDN_NORM_GATE_OUTPUT");
            keys.push_back(prefix + "_ATTENTION_OUTPUT");
            keys.push_back(prefix + "_ATTENTION_OUTPUT_ALLREDUCED");
            keys.push_back(prefix + "_FFN_NORM");
            keys.push_back(prefix + "_MOE_ROUTING_INDICES");
            keys.push_back(prefix + "_MOE_ROUTING_WEIGHTS");
            keys.push_back(prefix + "_MOE_SHARED_EXPERT_OUTPUT");
            keys.push_back(prefix + "_MOE_EXPERT_OUTPUT");
            keys.push_back(prefix + "_MOE_SHARED_GATE_OUTPUT");
            keys.push_back(prefix + "_MOE_COMBINED_OUTPUT");
        }
        keys.push_back("layer3_ATTENTION_NORM");
        keys.push_back("layer3_Q_PROJECTION");
        keys.push_back("layer3_K_PROJECTION");
        keys.push_back("layer3_V_PROJECTION");
        keys.push_back("layer3_Q_ROPE");
        keys.push_back("layer3_K_ROPE");
        keys.push_back("layer3_KV_APPEND_SOURCE_K");
        keys.push_back("layer3_KV_APPEND_SOURCE_V");
        keys.push_back("layer3_KV_CACHE_K");
        keys.push_back("layer3_KV_CACHE_V");
        keys.push_back("layer31_ATTENTION_NORM");
        keys.push_back("layer31_Q_PROJECTION");
        keys.push_back("layer31_K_PROJECTION");
        keys.push_back("layer31_V_PROJECTION");
        keys.push_back("layer31_Q_ROPE");
        keys.push_back("layer31_K_ROPE");
        keys.push_back("layer31_KV_APPEND_SOURCE_K");
        keys.push_back("layer31_KV_APPEND_SOURCE_V");
        keys.push_back("layer31_KV_CACHE_K");
        keys.push_back("layer31_KV_CACHE_V");
        return keys;
    }

    /**
     * @brief Return a compact stage filter for grouped verifier decode proofs.
     *
     * The grouped verifier diagnostic compares an M=2..4 all-position verifier
     * pass against the same candidate rows replayed one token at a time.  The
     * most useful failure signal is the earliest layer-0 boundary where the
     * grouped row stops matching the serial row: attention input norm, Q/K/V,
     * RoPE, KV append payload, persistent KV cache payload, effective K/V after
     * cache conversion, attention context, Wo projection, or the first MoE
     * boundary.  Keeping the filter focused here avoids recording the whole
     * long-context prompt while still proving whether the bug lives in the
     * grouped attention/KV contract or in a downstream grouped verifier stage.
     *
     * @return Ordered snapshot keys requested by grouped verifier diagnostics.
     */
    inline std::vector<std::string> qwen36MoEGroupedVerifierSnapshotKeys()
    {
        std::vector<std::string> keys = {"EMBEDDING"};
        for (int layer = 0; layer < 2; ++layer)
        {
            const std::string prefix = "layer" + std::to_string(layer);
            keys.push_back(prefix + "_ATTENTION_NORM");
            keys.push_back(prefix + "_Q_PROJECTION");
            keys.push_back(prefix + "_K_PROJECTION");
            keys.push_back(prefix + "_V_PROJECTION");
            keys.push_back(prefix + "_Q_NORM");
            keys.push_back(prefix + "_K_NORM");
            keys.push_back(prefix + "_Q_ROPE");
            keys.push_back(prefix + "_K_ROPE");
            keys.push_back(prefix + "_FA_GATE");
            keys.push_back(prefix + "_QKV_PROJECTION");
            keys.push_back(prefix + "_GDN_Z_PROJECTION");
            keys.push_back(prefix + "_GDN_ALPHA");
            keys.push_back(prefix + "_GDN_BETA");
            keys.push_back(prefix + "_GDN_CONV1D_OUTPUT");
            keys.push_back(prefix + "_GDN_DELTA_RULE_OUTPUT");
            keys.push_back(prefix + "_GDN_NORM_GATE_OUTPUT");
            keys.push_back(prefix + "_GDN_OUTPUT");
            keys.push_back(prefix + "_KV_APPEND_SOURCE_K");
            keys.push_back(prefix + "_KV_APPEND_SOURCE_V");
            keys.push_back(prefix + "_KV_CACHE_K");
            keys.push_back(prefix + "_KV_CACHE_V");
            keys.push_back(prefix + "_ATTENTION_EFFECTIVE_K");
            keys.push_back(prefix + "_ATTENTION_EFFECTIVE_V");
            keys.push_back(prefix + "_ATTENTION_CONTEXT");
            keys.push_back(prefix + "_ATTENTION_CONTEXT_GATED");
            keys.push_back(prefix + "_ATTENTION_OUTPUT");
            keys.push_back(prefix + "_ATTENTION_OUTPUT_ALLREDUCED");
            keys.push_back(prefix + "_FFN_NORM_RESIDUAL_OUT");
            keys.push_back(prefix + "_FFN_NORM");
            keys.push_back(prefix + "_MOE_ROUTER_OUTPUT");
            keys.push_back(prefix + "_MOE_ROUTING_INDICES");
            keys.push_back(prefix + "_MOE_ROUTING_WEIGHTS");
            keys.push_back(prefix + "_MOE_EXPERT_OUTPUT");
            keys.push_back(prefix + "_MOE_SHARED_EXPERT_OUTPUT");
            keys.push_back(prefix + "_MOE_SHARED_GATE_OUTPUT");
            keys.push_back(prefix + "_MOE_COMBINED_OUTPUT");
            keys.push_back(prefix + "_MOE_SHARED_EXPERT_OUTPUT_ALLREDUCED");
            keys.push_back(prefix + "_MOE_COMBINED_OUTPUT_ALLREDUCED");
            keys.push_back(prefix + "_FFN_RESIDUAL");
        }
        /*
         * Qwen3.6 MoE has many layers, and LocalTP verifier bugs often emerge
         * after a numerically tiny per-layer drift accumulates.  Keep the deep
         * probe cheap by asking later layers only for replicated row-boundary
         * tensors instead of every local projection/state snapshot.
         */
        for (int layer = 2; layer < 40; ++layer)
        {
            const std::string prefix = "layer" + std::to_string(layer);
            const bool include_deep_effective_kv =
                debugEnv().attention.debug_effective_kv_snapshot &&
                debugEnv().attention.debugEffectiveKVSnapshotLayerSelected(layer);
            keys.push_back(prefix + "_ATTENTION_NORM_RESIDUAL_OUT");
            keys.push_back(prefix + "_ATTENTION_NORM");
            keys.push_back(prefix + "_QKV_PROJECTION");
            keys.push_back(prefix + "_GDN_Z_PROJECTION");
            keys.push_back(prefix + "_GDN_ALPHA");
            keys.push_back(prefix + "_GDN_BETA");
            keys.push_back(prefix + "_GDN_CONV1D_OUTPUT");
            keys.push_back(prefix + "_GDN_DELTA_RULE_OUTPUT");
            keys.push_back(prefix + "_GDN_NORM_GATE_OUTPUT");
            keys.push_back(prefix + "_GDN_OUTPUT");
            if (include_deep_effective_kv)
            {
                keys.push_back(prefix + "_ATTENTION_EFFECTIVE_K");
                keys.push_back(prefix + "_ATTENTION_EFFECTIVE_V");
                for (int row = 0; row < 16; ++row)
                {
                    keys.push_back(prefix + "_ATTENTION_EFFECTIVE_K_ROW" + std::to_string(row));
                    keys.push_back(prefix + "_ATTENTION_EFFECTIVE_V_ROW" + std::to_string(row));
                }
            }
            keys.push_back(prefix + "_ATTENTION_CONTEXT_GATED");
            keys.push_back(prefix + "_ATTENTION_OUTPUT");
            /*
             * Deep LocalTP drift often first becomes visible at the residual
             * norm that consumes the Wo/GDN projection.  Capture the gated
             * context before Wo, the projection output, and the replicated
             * allreduce boundary so a failed grouped verifier proof can
             * distinguish attention/gating drift from projection/collective
             * drift and fused residual-norm drift.
             */
            keys.push_back(prefix + "_ATTENTION_OUTPUT_ALLREDUCED");
            keys.push_back(prefix + "_FFN_NORM_RESIDUAL_OUT");
            keys.push_back(prefix + "_FFN_NORM");
            keys.push_back(prefix + "_MOE_ROUTER_OUTPUT");
            keys.push_back(prefix + "_MOE_ROUTING_INDICES");
            keys.push_back(prefix + "_MOE_ROUTING_WEIGHTS");
            keys.push_back(prefix + "_MOE_EXPERT_OUTPUT");
            keys.push_back(prefix + "_MOE_SHARED_EXPERT_OUTPUT");
            keys.push_back(prefix + "_MOE_SHARED_GATE_OUTPUT");
            keys.push_back(prefix + "_MOE_COMBINED_OUTPUT");
            keys.push_back(prefix + "_MOE_SHARED_EXPERT_OUTPUT_ALLREDUCED");
            keys.push_back(prefix + "_MOE_COMBINED_OUTPUT_ALLREDUCED");
            keys.push_back(prefix + "_FFN_RESIDUAL");
        }
        keys.push_back("FINAL_NORM");
        keys.push_back("LM_HEAD");
        keys.push_back("LM_HEAD_ROWS_SELECT");
        return keys;
    }

    inline std::string compareMoEPrefixLeadingSnapshotRows(
        const std::map<std::string, std::vector<float>> &full_prefill,
        int full_rows,
        const std::map<std::string, std::vector<float>> &prefix_prefill,
        int prefix_rows,
        const std::vector<std::string> &keys)
    {
        if (full_rows <= 0 || prefix_rows <= 0)
            return "invalid prefix-invariance row counts";

        std::ostringstream report;
        std::ostringstream trace;
        std::vector<std::string> missing_keys;
        bool found_divergence = false;
        bool compared_any = false;
        auto compare_equal_size_snapshot =
            [&](const std::string &key,
                const std::vector<float> &full,
                const std::vector<float> &prefix) -> bool
        {
            double dot = 0.0;
            double full_norm = 0.0;
            double prefix_norm = 0.0;
            double diff_norm = 0.0;
            double max_abs = 0.0;
            size_t max_idx = 0;
            for (size_t i = 0; i < full.size(); ++i)
            {
                const double a = static_cast<double>(full[i]);
                const double b = static_cast<double>(prefix[i]);
                const double diff = a - b;
                dot += a * b;
                full_norm += a * a;
                prefix_norm += b * b;
                diff_norm += diff * diff;
                const double abs_diff = std::abs(diff);
                if (abs_diff > max_abs)
                {
                    max_abs = abs_diff;
                    max_idx = i;
                }
            }

            const double full_l2 = std::sqrt(full_norm);
            const double prefix_l2 = std::sqrt(prefix_norm);
            const double denom = full_l2 * prefix_l2;
            const double cosine = denom > 0.0 ? dot / denom : 1.0;
            const double rel_l2 = full_norm > 0.0 ? std::sqrt(diff_norm / full_norm)
                                                  : std::sqrt(diff_norm);
            compared_any = true;
            trace << "\n  ok " << key
                  << " elements=" << full.size()
                  << " cosine=" << cosine
                  << " rel_l2=" << rel_l2
                  << " max_abs=" << max_abs;

            constexpr double kPrefixSnapshotMaxAbsTolerance = 1.0e-4;
            constexpr double kPrefixSnapshotRelL2Tolerance = 1.0e-5;
            if (max_abs <= kPrefixSnapshotMaxAbsTolerance ||
                rel_l2 <= kPrefixSnapshotRelL2Tolerance)
            {
                return false;
            }

            if (!found_divergence)
                report << "prefix-invariance stage rows diverged:";
            found_divergence = true;
            report << "\n  " << key
                   << " elements=" << full.size()
                   << " cosine=" << cosine
                   << " rel_l2=" << rel_l2
                   << " max_abs=" << max_abs
                   << " max_idx=" << max_idx
                   << " full=" << full[max_idx]
                   << " prefix=" << prefix[max_idx]
                   << "\ncompared prefix before divergence:" << trace.str();
            if (!missing_keys.empty())
            {
                report << "\nmissing/skipped prefix snapshots:";
                for (const auto &missing_key : missing_keys)
                    report << "\n  " << missing_key;
            }
            return true;
        };
        for (const auto &key : keys)
        {
            const auto full_it = full_prefill.find(key);
            const auto prefix_it = prefix_prefill.find(key);
            if (full_it == full_prefill.end() || prefix_it == prefix_prefill.end())
            {
                missing_keys.push_back(key);
                continue;
            }

            const auto &full = full_it->second;
            const auto &prefix = prefix_it->second;
            if (full.empty() || prefix.empty())
                continue;
            if (prefix.size() % static_cast<size_t>(prefix_rows) != 0)
            {
                if (full.size() == prefix.size())
                {
                    if (compare_equal_size_snapshot(key, full, prefix))
                        break;
                    continue;
                }
                if (!found_divergence)
                    report << "prefix-invariance stage rows diverged:";
                found_divergence = true;
                report << "\n  " << key
                       << " has non-row-aligned sizes full=" << full.size()
                       << " rows=" << full_rows
                       << " prefix=" << prefix.size()
                       << " rows=" << prefix_rows;
                continue;
            }

            const size_t prefix_cols = prefix.size() / static_cast<size_t>(prefix_rows);
            size_t effective_full_rows = static_cast<size_t>(full_rows);
            size_t full_cols = 0;
            if (full.size() % static_cast<size_t>(full_rows) == 0)
            {
                full_cols = full.size() / static_cast<size_t>(full_rows);
            }
            else if (prefix_cols > 0 && full.size() % prefix_cols == 0)
            {
                /*
                 * Graph-captured prefill can snapshot padded bucket rows
                 * rather than just real prompt rows. Infer that row count from
                 * the prefix tensor column count so the comparison still
                 * checks the real leading prefix rows.
                 */
                full_cols = prefix_cols;
                effective_full_rows = full.size() / prefix_cols;
            }
            else
            {
                if (!found_divergence)
                    report << "prefix-invariance stage rows diverged:";
                found_divergence = true;
                report << "\n  " << key
                       << " has non-row-aligned sizes full=" << full.size()
                       << " rows=" << full_rows
                       << " prefix=" << prefix.size()
                       << " rows=" << prefix_rows;
                continue;
            }
            if (full_cols != prefix_cols)
            {
                if (!found_divergence)
                    report << "prefix-invariance stage rows diverged:";
                found_divergence = true;
                report << "\n  " << key
                       << " column mismatch full_cols=" << full_cols
                       << " prefix_cols=" << prefix_cols;
                continue;
            }

            const size_t compare_rows = std::min(
                effective_full_rows,
                static_cast<size_t>(prefix_rows));
            const size_t compare_count = compare_rows * full_cols;
            double dot = 0.0;
            double full_norm = 0.0;
            double prefix_norm = 0.0;
            double diff_norm = 0.0;
            double max_abs = 0.0;
            size_t max_idx = 0;
            for (size_t i = 0; i < compare_count; ++i)
            {
                const double a = static_cast<double>(full[i]);
                const double b = static_cast<double>(prefix[i]);
                const double diff = a - b;
                dot += a * b;
                full_norm += a * a;
                prefix_norm += b * b;
                diff_norm += diff * diff;
                const double abs_diff = std::abs(diff);
                if (abs_diff > max_abs)
                {
                    max_abs = abs_diff;
                    max_idx = i;
                }
            }

            const double full_l2 = std::sqrt(full_norm);
            const double prefix_l2 = std::sqrt(prefix_norm);
            const double denom = full_l2 * prefix_l2;
            const double cosine = denom > 0.0 ? dot / denom : 1.0;
            const double rel_l2 = full_norm > 0.0 ? std::sqrt(diff_norm / full_norm)
                                                  : std::sqrt(diff_norm);
            compared_any = true;
            trace << "\n  ok " << key
                  << " rows=" << compare_rows
                  << " cols=" << full_cols
                  << " cosine=" << cosine
                  << " rel_l2=" << rel_l2
                  << " max_abs=" << max_abs;

            constexpr double kPrefixSnapshotMaxAbsTolerance = 1.0e-4;
            constexpr double kPrefixSnapshotRelL2Tolerance = 1.0e-5;
            if (max_abs > kPrefixSnapshotMaxAbsTolerance &&
                rel_l2 > kPrefixSnapshotRelL2Tolerance)
            {
                if (!found_divergence)
                    report << "prefix-invariance stage rows diverged:";
                found_divergence = true;
                report << "\n  " << key
                       << " rows=" << compare_rows
                       << " cols=" << full_cols
                       << " cosine=" << cosine
                       << " rel_l2=" << rel_l2
                       << " max_abs=" << max_abs
                       << " max_row=" << (full_cols == 0 ? 0 : max_idx / full_cols)
                       << " max_col=" << (full_cols == 0 ? 0 : max_idx % full_cols)
                       << " full=" << full[max_idx]
                       << " prefix=" << prefix[max_idx];
                report << "\ncompared prefix before divergence:" << trace.str();
                if (!missing_keys.empty())
                {
                    report << "\nmissing/skipped prefix snapshots:";
                    for (const auto &missing_key : missing_keys)
                        report << "\n  " << missing_key;
                }
                break;
            }
        }
        if (!compared_any && !found_divergence)
            return "no comparable prefix-invariance snapshots captured";
        return report.str();
    }

    /**
     * @brief Return whether a prefix diagnostic snapshot is an absolute KV-cache payload.
     *
     * Projection, RoPE, and append-source snapshots are suffix-local tensors:
     * row zero is the first token in the just-executed suffix request.  The
     * `*_KV_CACHE_K` and `*_KV_CACHE_V` snapshots instead describe the live
     * cache payload after the append, so row zero is the beginning of the whole
     * request history.  Suffix diagnostics must therefore compare absolute
     * cache rows for KV snapshots and local leading rows for all other tensors.
     */
    inline bool isMoEPrefixKVCachePayloadSnapshotKey(const std::string &key)
    {
        return key.find("_KV_CACHE_K") != std::string::npos ||
               key.find("_KV_CACHE_V") != std::string::npos;
    }

    /**
     * @brief Compare suffix diagnostic snapshots with explicit row lifetimes.
     *
     * Partial prefix-restore diagnostics capture two different tensor lifetime
     * shapes in the same snapshot map.  Stateless stage snapshots are scoped to
     * the suffix prefill and compare from local row zero.  KV-cache payload
     * snapshots are scoped to the full live cache and must compare the absolute
     * suffix interval `[cache_suffix_start_row, cache_suffix_start_row +
     * cache_suffix_rows)`.  Keeping this distinction first-class prevents the
     * diagnostic from accidentally rechecking only the restored prefix rows
     * when the bug lives in the first suffix append.
     *
     * @param split_suffix Snapshots from the uncached split-prefill baseline.
     * @param restored_suffix Snapshots from the cached partial-hit replay.
     * @param suffix_local_rows Number of rows owned by suffix-local snapshots.
     * @param cache_suffix_start_row Absolute KV-cache row where the suffix starts.
     * @param cache_suffix_rows Number of absolute KV-cache rows to compare.
     * @param keys Ordered diagnostic snapshot keys to compare.
     * @return Empty string when all comparable snapshots match; otherwise a
     *         detailed first-divergence report.
     */
    inline std::string compareMoEPrefixSuffixSnapshotRows(
        const std::map<std::string, std::vector<float>> &split_suffix,
        const std::map<std::string, std::vector<float>> &restored_suffix,
        int suffix_local_rows,
        int cache_suffix_start_row,
        int cache_suffix_rows,
        const std::vector<std::string> &keys)
    {
        if (suffix_local_rows <= 0 ||
            cache_suffix_start_row < 0 ||
            cache_suffix_rows <= 0)
        {
            return "invalid suffix snapshot row-range request";
        }

        struct SnapshotShape
        {
            bool ok = false;
            size_t rows = 0;
            size_t cols = 0;
            std::string error;
        };

        auto infer_shape =
            [](const std::string &label,
               const std::string &key,
               const std::vector<float> &values,
               int expected_rows,
               size_t column_hint) -> SnapshotShape
        {
            SnapshotShape shape;
            if (values.empty())
            {
                shape.error = label + " " + key + " is empty";
                return shape;
            }
            if (expected_rows <= 0)
            {
                shape.error = label + " " + key + " has invalid expected_rows=" +
                              std::to_string(expected_rows);
                return shape;
            }

            const size_t expected = static_cast<size_t>(expected_rows);
            if (values.size() % expected == 0)
            {
                shape.ok = true;
                shape.rows = expected;
                shape.cols = values.size() / expected;
                return shape;
            }
            if (column_hint > 0 && values.size() % column_hint == 0)
            {
                shape.ok = true;
                shape.rows = values.size() / column_hint;
                shape.cols = column_hint;
                return shape;
            }
            if (isMoEPrefixKVCachePayloadSnapshotKey(key))
            {
                /*
                 * Padded prefill graph capture snapshots the fixed bucket
                 * cache view.  A 640-token real request whose final chunk uses
                 * a 256-row bucket can therefore dump 768 absolute KV rows.
                 * Infer Qwen3.6 local KV payload widths before declaring the
                 * matrix incomparable so row-range diagnostics still inspect
                 * the real suffix interval.
                 */
                constexpr size_t kQwen36MoEKVSnapshotColumnHints[] = {
                    1024, 2048, 512, 256, 4096, 768, 384, 128, 64};
                const size_t minimum_rows = static_cast<size_t>(expected_rows);
                for (const size_t cols : kQwen36MoEKVSnapshotColumnHints)
                {
                    if (cols > 0 && values.size() % cols == 0)
                    {
                        const size_t rows = values.size() / cols;
                        if (rows >= minimum_rows)
                        {
                            shape.ok = true;
                            shape.rows = rows;
                            shape.cols = cols;
                            return shape;
                        }
                    }
                }
            }

            std::ostringstream oss;
            oss << label << " " << key
                << " has non-row-aligned size=" << values.size()
                << " expected_rows=" << expected_rows
                << " column_hint=" << column_hint;
            shape.error = oss.str();
            return shape;
        };

        std::ostringstream report;
        std::ostringstream trace;
        std::vector<std::string> missing_keys;
        bool found_divergence = false;
        bool compared_any = false;

        constexpr double kPrefixSnapshotMaxAbsTolerance = 1.0e-4;
        constexpr double kPrefixSnapshotRelL2Tolerance = 1.0e-5;

        for (const auto &key : keys)
        {
            const auto split_it = split_suffix.find(key);
            const auto restored_it = restored_suffix.find(key);
            if (split_it == split_suffix.end() ||
                restored_it == restored_suffix.end())
            {
                missing_keys.push_back(key);
                continue;
            }

            const bool kv_cache_snapshot =
                isMoEPrefixKVCachePayloadSnapshotKey(key);
            const int expected_rows =
                kv_cache_snapshot
                    ? cache_suffix_start_row + cache_suffix_rows
                    : suffix_local_rows;
            const int split_start =
                kv_cache_snapshot ? cache_suffix_start_row : 0;
            const int restored_start =
                kv_cache_snapshot ? cache_suffix_start_row : 0;
            const int rows_to_compare =
                kv_cache_snapshot ? cache_suffix_rows : suffix_local_rows;

            const auto &split_values = split_it->second;
            const auto &restored_values = restored_it->second;
            const SnapshotShape split_shape = infer_shape(
                "split", key, split_values, expected_rows, 0);
            if (!split_shape.ok)
            {
                missing_keys.push_back(split_shape.error);
                continue;
            }

            const SnapshotShape restored_shape = infer_shape(
                "restored", key, restored_values, expected_rows, split_shape.cols);
            if (!restored_shape.ok)
            {
                missing_keys.push_back(restored_shape.error);
                continue;
            }
            if (split_shape.cols != restored_shape.cols)
            {
                if (!found_divergence)
                    report << "suffix snapshot rows diverged:";
                found_divergence = true;
                report << "\n  " << key
                       << " column mismatch split_cols=" << split_shape.cols
                       << " restored_cols=" << restored_shape.cols;
                break;
            }
            if (static_cast<size_t>(split_start + rows_to_compare) > split_shape.rows ||
                static_cast<size_t>(restored_start + rows_to_compare) > restored_shape.rows)
            {
                if (!found_divergence)
                    report << "suffix snapshot rows diverged:";
                found_divergence = true;
                report << "\n  " << key
                       << " requested range exceeds snapshot rows"
                       << " split_start=" << split_start
                       << " restored_start=" << restored_start
                       << " rows=" << rows_to_compare
                       << " split_rows=" << split_shape.rows
                       << " restored_rows=" << restored_shape.rows
                       << " absolute_kv_cache=" << (kv_cache_snapshot ? 1 : 0);
                break;
            }

            double dot = 0.0;
            double split_norm = 0.0;
            double restored_norm = 0.0;
            double diff_norm = 0.0;
            double max_abs = 0.0;
            size_t max_row = 0;
            size_t max_col = 0;
            double max_split = 0.0;
            double max_restored = 0.0;
            for (int row = 0; row < rows_to_compare; ++row)
            {
                const size_t split_row =
                    static_cast<size_t>(split_start + row);
                const size_t restored_row =
                    static_cast<size_t>(restored_start + row);
                for (size_t col = 0; col < split_shape.cols; ++col)
                {
                    const double a = static_cast<double>(
                        split_values[split_row * split_shape.cols + col]);
                    const double b = static_cast<double>(
                        restored_values[restored_row * restored_shape.cols + col]);
                    const double diff = a - b;
                    dot += a * b;
                    split_norm += a * a;
                    restored_norm += b * b;
                    diff_norm += diff * diff;
                    const double abs_diff = std::abs(diff);
                    if (abs_diff > max_abs)
                    {
                        max_abs = abs_diff;
                        max_row = static_cast<size_t>(row);
                        max_col = col;
                        max_split = a;
                        max_restored = b;
                    }
                }
            }

            const double split_l2 = std::sqrt(split_norm);
            const double restored_l2 = std::sqrt(restored_norm);
            const double denom = split_l2 * restored_l2;
            const double cosine = denom > 0.0 ? dot / denom : 1.0;
            const double rel_l2 = split_norm > 0.0 ? std::sqrt(diff_norm / split_norm)
                                                   : std::sqrt(diff_norm);
            compared_any = true;
            trace << "\n  ok " << key
                  << " rows=" << rows_to_compare
                  << " cols=" << split_shape.cols
                  << " split_start=" << split_start
                  << " restored_start=" << restored_start
                  << " absolute_kv_cache=" << (kv_cache_snapshot ? 1 : 0)
                  << " cosine=" << cosine
                  << " rel_l2=" << rel_l2
                  << " max_abs=" << max_abs;

            if (max_abs > kPrefixSnapshotMaxAbsTolerance &&
                rel_l2 > kPrefixSnapshotRelL2Tolerance)
            {
                if (!found_divergence)
                    report << "suffix snapshot rows diverged:";
                found_divergence = true;
                report << "\n  " << key
                       << " rows=" << rows_to_compare
                       << " cols=" << split_shape.cols
                       << " split_start=" << split_start
                       << " restored_start=" << restored_start
                       << " absolute_kv_cache=" << (kv_cache_snapshot ? 1 : 0)
                       << " cosine=" << cosine
                       << " rel_l2=" << rel_l2
                       << " max_abs=" << max_abs
                       << " max_local_row=" << max_row
                       << " max_split_row=" << (static_cast<size_t>(split_start) + max_row)
                       << " max_restored_row=" << (static_cast<size_t>(restored_start) + max_row)
                       << " max_col=" << max_col
                       << " split=" << max_split
                       << " restored=" << max_restored
                       << "\ncompared suffix before divergence:" << trace.str();
                if (!missing_keys.empty())
                {
                    report << "\nmissing/skipped suffix snapshots:";
                    for (const auto &missing_key : missing_keys)
                        report << "\n  " << missing_key;
                }
                break;
            }
        }

        if (!compared_any && !found_divergence)
            return "no comparable suffix snapshots captured";
        return report.str();
    }

    inline std::string describeMoEPrefixLeadingSnapshotRows(
        const std::map<std::string, std::vector<float>> &full_prefill,
        int full_rows,
        const std::map<std::string, std::vector<float>> &prefix_prefill,
        int prefix_rows,
        const std::vector<std::string> &keys)
    {
        if (full_rows <= 0 || prefix_rows <= 0)
            return "invalid prefix-invariance row counts";

        std::ostringstream report;
        for (const auto &key : keys)
        {
            const auto full_it = full_prefill.find(key);
            const auto prefix_it = prefix_prefill.find(key);
            if (full_it == full_prefill.end() || prefix_it == prefix_prefill.end())
            {
                report << "\n  " << key << " missing";
                continue;
            }

            const auto &full = full_it->second;
            const auto &prefix = prefix_it->second;
            if (full.empty() || prefix.empty() ||
                prefix.size() % static_cast<size_t>(prefix_rows) != 0)
            {
                report << "\n  " << key
                       << " not row-aligned full_size=" << full.size()
                       << " full_rows=" << full_rows
                       << " prefix_size=" << prefix.size()
                       << " prefix_rows=" << prefix_rows;
                continue;
            }

            const size_t prefix_cols = prefix.size() / static_cast<size_t>(prefix_rows);
            size_t effective_full_rows = static_cast<size_t>(full_rows);
            size_t full_cols = 0;
            if (full.size() % static_cast<size_t>(full_rows) == 0)
            {
                full_cols = full.size() / static_cast<size_t>(full_rows);
            }
            else if (prefix_cols > 0 && full.size() % prefix_cols == 0)
            {
                full_cols = prefix_cols;
                effective_full_rows = full.size() / prefix_cols;
            }
            else
            {
                report << "\n  " << key
                       << " not row-aligned full_size=" << full.size()
                       << " full_rows=" << full_rows
                       << " prefix_size=" << prefix.size()
                       << " prefix_rows=" << prefix_rows;
                continue;
            }
            if (full_cols != prefix_cols)
            {
                report << "\n  " << key
                       << " column mismatch full_cols=" << full_cols
                       << " prefix_cols=" << prefix_cols;
                continue;
            }

            const size_t compare_rows = std::min(
                effective_full_rows,
                static_cast<size_t>(prefix_rows));
            const size_t compare_count = compare_rows * full_cols;
            double dot = 0.0;
            double full_norm = 0.0;
            double prefix_norm = 0.0;
            double diff_norm = 0.0;
            double max_abs = 0.0;
            size_t max_idx = 0;
            for (size_t i = 0; i < compare_count; ++i)
            {
                const double a = static_cast<double>(full[i]);
                const double b = static_cast<double>(prefix[i]);
                const double diff = a - b;
                dot += a * b;
                full_norm += a * a;
                prefix_norm += b * b;
                diff_norm += diff * diff;
                const double abs_diff = std::abs(diff);
                if (abs_diff > max_abs)
                {
                    max_abs = abs_diff;
                    max_idx = i;
                }
            }

            const double full_l2 = std::sqrt(full_norm);
            const double prefix_l2 = std::sqrt(prefix_norm);
            const double denom = full_l2 * prefix_l2;
            const double cosine = denom > 0.0 ? dot / denom : 1.0;
            const double rel_l2 = full_norm > 0.0 ? std::sqrt(diff_norm / full_norm)
                                                  : std::sqrt(diff_norm);
            report << "\n  " << key
                   << " rows=" << compare_rows
                   << " cols=" << full_cols
                   << " cosine=" << cosine
                   << " rel_l2=" << rel_l2
                   << " max_abs=" << max_abs
                   << " max_row=" << (full_cols == 0 ? 0 : max_idx / full_cols)
                   << " max_col=" << (full_cols == 0 ? 0 : max_idx % full_cols)
                   << " full=" << full[max_idx]
                   << " prefix=" << prefix[max_idx];
        }
        return report.str();
    }

    inline std::vector<float> selectMoEVerifierSnapshotRow(
        const std::vector<float> &all_position_snapshot,
        size_t serial_snapshot_size,
        int row)
    {
        if (serial_snapshot_size == 0 ||
            all_position_snapshot.size() <= serial_snapshot_size ||
            all_position_snapshot.size() % serial_snapshot_size != 0)
        {
            return all_position_snapshot;
        }

        const size_t rows = all_position_snapshot.size() / serial_snapshot_size;
        if (row < 0 || static_cast<size_t>(row) >= rows)
        {
            return all_position_snapshot;
        }

        const auto begin =
            all_position_snapshot.begin() +
            static_cast<std::ptrdiff_t>(static_cast<size_t>(row) * serial_snapshot_size);
        return std::vector<float>(
            begin,
            begin + static_cast<std::ptrdiff_t>(serial_snapshot_size));
    }

    /**
     * @brief Map a grouped-only verifier stage to its serial decode oracle.
     *
     * The grouped LM-head path first gathers the selected normalized hidden
     * rows into LM_HEAD_ROWS_SELECT.  Serial M=1 decode feeds FINAL_NORM
     * directly into the LM head and therefore has no row-select stage with the
     * same name.  Comparing the selected grouped row to serial FINAL_NORM proves
     * the gather itself without pretending the serial graph owns that stage.
     */
    inline std::string serialReferenceKeyForMoEGroupedVerifierSnapshot(
        const std::string &grouped_key)
    {
        if (grouped_key == "LM_HEAD_ROWS_SELECT")
        {
            return "FINAL_NORM";
        }
        return grouped_key;
    }

    inline void appendMoEAllPositionVerifierRows(
        const std::map<std::string, std::vector<float>> &all_position_snapshots,
        const std::map<std::string, std::vector<float>> &serial_snapshots,
        int row,
        int output_tokens,
        std::ofstream &csv,
        std::vector<MoESnapshotCompareRow> *rows)
    {
        std::vector<std::string> keys;
        keys.reserve(all_position_snapshots.size() + serial_snapshots.size());
        std::set<std::string> seen_keys;
        auto append_key_if_captured =
            [&](const std::string &key)
        {
            if (seen_keys.find(key) != seen_keys.end())
            {
                return;
            }
            if (all_position_snapshots.find(key) == all_position_snapshots.end() &&
                serial_snapshots.find(key) == serial_snapshots.end())
            {
                return;
            }
            seen_keys.insert(key);
            keys.push_back(key);
        };

        /*
         * std::map iteration is lexical, which puts FINAL_NORM before layer 0
         * and layer 10 before layer 2.  Preserve the declarative graph-stage
         * order so the first exact mismatch is the producer that introduced
         * drift rather than whichever downstream key sorts first.
         */
        for (const auto &key : qwen36MoEGroupedVerifierSnapshotKeys())
        {
            append_key_if_captured(key);
        }
        for (const auto &entry : all_position_snapshots)
        {
            append_key_if_captured(entry.first);
        }
        for (const auto &entry : serial_snapshots)
        {
            append_key_if_captured(entry.first);
        }

        const std::string comparison =
            "all_position_row" + std::to_string(row) +
            "_vs_serial_prefix" + std::to_string(output_tokens);
        for (const auto &key : keys)
        {
            const auto all_it = all_position_snapshots.find(key);
            const std::string reference_key =
                serialReferenceKeyForMoEGroupedVerifierSnapshot(key);
            const auto serial_it = serial_snapshots.find(reference_key);
            if (all_it == all_position_snapshots.end() ||
                serial_it == serial_snapshots.end())
            {
                MoESnapshotCompareRow missing;
                missing.comparison = comparison;
                missing.sync_idx = row;
                missing.output_tokens = output_tokens;
                missing.key = key;
                missing.reference_key = reference_key;
                missing.left_label = "all_position";
                missing.right_label = "serial";
                missing.present_in_baseline =
                    all_it != all_position_snapshots.end() &&
                    !all_it->second.empty();
                missing.present_in_mtp =
                    serial_it != serial_snapshots.end() &&
                    !serial_it->second.empty();
                missing.elements = std::max(
                    all_it == all_position_snapshots.end() ? size_t{0} : all_it->second.size(),
                    serial_it == serial_snapshots.end() ? size_t{0} : serial_it->second.size());
                rows->push_back(missing);
                writeMoESnapshotCsvRow(csv, missing);
                continue;
            }

            const auto selected = selectMoEVerifierSnapshotRow(
                all_it->second,
                serial_it->second.size(),
                row);
            auto compare = compareMoESnapshotVectors(
                selected,
                serial_it->second.data(),
                serial_it->second.size(),
                row,
                output_tokens,
                key,
                reference_key,
                comparison,
                "all_position",
                "serial");
            rows->push_back(compare);
            writeMoESnapshotCsvRow(csv, compare);
        }
    }

    inline void runMoEMainVerifierAllPositionRowsMatchSerialDecode(
        const MoEPrefixRestoreParityCase &test_case,
        bool use_row_indexed_logits = false,
        bool use_skip_gather = false,
        bool verify_compact_device_outcome = false,
        bool use_deferred_verifier_sync = false,
        bool verify_published_state_continuation = false,
        int verifier_row_count = 2,
        bool verify_device_resident_publication = false,
        bool expect_grouped_moe_verifier_prefill = false,
        bool force_first_speculative_rejection = false)
    {
        ScopedMoEParityProductionMode production_mode(
            shouldForceMoEParityProductionMode(test_case));
        std::unique_ptr<ScopedEnvironmentValues> perf_stats_enabled;
        if (expect_grouped_moe_verifier_prefill)
        {
            perf_stats_enabled.reset(new ScopedEnvironmentValues({
                {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
            }));
        }
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadMoEReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        if (moeReferenceInputsStoppedCurrentTest())
        {
            return;
        }
        ASSERT_GE(verifier_row_count, 1)
            << "verifier row proof must exercise at least one row";
        ASSERT_LE(verifier_row_count, 4)
            << "Phase 9.7 production proof currently targets M=1..4";
        ASSERT_GE(expected_tokens.size(), 2u)
            << "all-position verifier row regression needs two setup tokens; "
               "later verifier tokens are extended from a serial Llaminar "
               "oracle so the proof is not limited by PyTorch fixture length";

        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cpu()
                                    : test_case.devices.front().toLocalDeviceId();

        InferenceRunnerConfig config;
        config.max_seq_len = test_case.max_seq_len;
        config.batch_size = 1;
        config.force_graph = true;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = parseKVCachePrecision(test_case.kv_cache_precision);
        config.use_mapped_memory = false;
        config.mtp.enabled = true;
        config.mtp.draft_tokens = 1;
        config.moe_routed_expert_plan = test_case.moe_routed_expert_plan;

        auto proof_runner =
            createMoEVerifierProofRunner(test_case, model_path, device, config);
        ASSERT_TRUE(proof_runner.error.empty())
            << proof_runner.error;
        auto &runner = proof_runner.runner;
        ASSERT_NE(runner, nullptr);
        const int vocab = runner->vocab_size();
        ASSERT_GT(vocab, 0);
        const bool capture_operation_snapshots =
            !verify_published_state_continuation;

        auto sample_current = [&](const char *label) -> int32_t
        {
            int32_t sampled = runner->sampleGreedyOnDevice();
            if (sampled >= 0)
            {
                return sampled;
            }
            const float *logits = runner->logits();
            ADD_FAILURE() << "sampleGreedyOnDevice failed after " << label
                          << "; falling back to host-visible logits";
            return argmaxToken(logits, runner->vocab_size());
        };

        runner->setSuppressTimeline(true);
        runner->setSkipLogitsGatherDecode(use_skip_gather);
        runner->setSkipLogitsGatherPrefill(use_skip_gather);
        if (capture_operation_snapshots)
        {
            runner->enableSnapshotCapture();
        }
        else
        {
            /*
             * Publication proofs need live KV/GDN/short-conv state, compact
             * verifier outcomes, logits, and continuation tokens.  They do not
             * need per-stage tensor snapshots, and enabling those snapshots on
             * the 35B MoE GPU fixture can exhaust the 24 GB CUDA cards before
             * the MTP publication path is reached.
             */
            runner->disableSnapshotCapture();
        }

        ASSERT_TRUE(runner->forward(
            prompt_tokens.data(),
            static_cast<int>(prompt_tokens.size())))
            << "prefill forward failed";
        EXPECT_EQ(sample_current("prefill"), expected_tokens[0]);

        int32_t token_after_setup = -1;
        for (int i = 0; i < 2; ++i)
        {
            const int32_t token = expected_tokens[static_cast<size_t>(i)];
            ASSERT_TRUE(runner->forward(&token, 1))
                << "serial setup forward failed at token index " << i;
            /*
             * This regression proves that the all-position verifier rows are
             * decode-equivalent to serial Llaminar rows for a fixed token
             * sequence. Some benchmark-prompt MoE fixtures contain PyTorch vs
             * quantized-Llaminar near ties during setup, so do not make this
             * helper fail on the reference token here; dedicated PyTorch parity
             * tests own that contract. Sampling still flushes device-side logits
             * and catches sampleGreedyOnDevice failures.
             */
            token_after_setup = sample_current("serial setup");
        }
        ASSERT_GE(token_after_setup, 0)
            << "serial setup must produce the first verifier input token";

        if (verify_published_state_continuation)
        {
            /*
             * Publication tests intentionally include shifted-MTP cache state:
             * they mirror the production accepted-prefix boundary and then
             * prove continuation equivalence.  Pure verifier-row proofs do not
             * need sidecar maintenance, and keeping that state out of the base
             * snapshot makes the M=1..4 numeric checks isolate target-model
             * logits/KV/GDN equivalence rather than shifted-cache coherence.
             */
            const int production_sidecar_position = runner->get_position();
            ASSERT_TRUE(runner->commitMTPShiftedRowFromCurrentTerminalHidden(
                expected_tokens[0],
                /*already_appended_tokens=*/0,
                /*allow_speculative_discard=*/true,
                production_sidecar_position - 2))
                << "failed to advance shifted MTP cache for first setup token";
            ASSERT_TRUE(runner->commitMTPShiftedRowFromCurrentTerminalHidden(
                expected_tokens[1],
                /*already_appended_tokens=*/0,
                /*allow_speculative_discard=*/true,
                production_sidecar_position - 1))
                << "failed to advance shifted MTP cache for second setup token";
        }

        const PrefixStateSnapshot verifier_base = runner->captureLivePrefixState();
        ASSERT_TRUE(verifier_base.valid);
        std::optional<PrefixRuntimeStateSnapshot> verifier_base_probe;
        std::optional<PrefixRuntimeStateSnapshot> after_pre_verifier_shifted_commit_probe;
        if (verify_published_state_continuation)
        {
            verifier_base_probe = runner->prefixStateProbe();
        }

        std::vector<int32_t> verifier_tokens;
        verifier_tokens.reserve(static_cast<size_t>(verifier_row_count));
        int32_t next_verifier_token = token_after_setup;
        for (int i = 0; i < verifier_row_count; ++i)
        {
            verifier_tokens.push_back(next_verifier_token);
            ASSERT_TRUE(runner->forward(&next_verifier_token, 1))
                << "serial verifier-token extension failed at row " << i;
            next_verifier_token =
                sample_current("serial verifier-token extension");
            ASSERT_GE(next_verifier_token, 0)
                << "serial verifier-token extension must produce row "
                << (i + 1);
        }
        std::optional<int32_t> forced_rejection_expected_row0;
        if (force_first_speculative_rejection)
        {
            ASSERT_GE(verifier_tokens.size(), 2u)
                << "forced rejected-publication proof needs a first verifier "
                   "token plus one intentionally wrong draft token";
            forced_rejection_expected_row0 = verifier_tokens[1];
            int32_t rejected_draft = -1;
            for (int delta = 1; delta < vocab; ++delta)
            {
                const int32_t candidate =
                    static_cast<int32_t>(
                        (*forced_rejection_expected_row0 + delta) % vocab);
                if (candidate != *forced_rejection_expected_row0)
                {
                    rejected_draft = candidate;
                    break;
                }
            }
            ASSERT_GE(rejected_draft, 0)
                << "forced rejected-publication proof could not find a "
                   "vocabulary token different from the serial row-0 sample";
            verifier_tokens[1] = rejected_draft;
        }
        ASSERT_TRUE(runner->restoreLivePrefixState(verifier_base))
            << "verifier row proof must restore the captured base after "
               "serially extending verifier input tokens";

        runner->clearSnapshots();
        const int base_sidecar_position = runner->get_position();
        if (verify_published_state_continuation)
        {
            ASSERT_TRUE(runner->commitMTPShiftedRowFromCurrentTerminalHidden(
                verifier_tokens[0],
                /*already_appended_tokens=*/0,
                /*allow_speculative_discard=*/true,
                base_sidecar_position))
                << "shifted MTP row commit from terminal hidden must not perturb "
                   "main all-position verifier state";
        }
        if (verify_published_state_continuation)
        {
            after_pre_verifier_shifted_commit_probe = runner->prefixStateProbe();
        }
        if (!runner->supportsMTPSidecarPreservesMainState())
        {
            /*
             * This helper validates target-verifier row math.  MoE sidecars do
             * not yet advertise the stronger vLLM-style guarantee that shifted
             * sidecar maintenance leaves every live main-model surface untouched.
             * Production restores the verifier base before target verification
             * on those runners, so the diagnostic must exercise the same
             * contract instead of accidentally testing an unsupported shortcut.
             */
            ASSERT_TRUE(runner->restoreLivePrefixState(verifier_base))
                << "MoE all-position verifier diagnostics must restore the "
                   "verifier base after shifted-cache maintenance unless the "
                   "runner explicitly preserves main state";
        }
        if (use_row_indexed_logits)
        {
            MTPSpecDecodeMetadataShape shape;
            shape.max_requests = 1;
            shape.max_draft_tokens =
                static_cast<int>(verifier_tokens.size());
            MTPSpecDecodeVerifierDraftRequest verifier_request;
            verifier_request.request_id = 0;
            verifier_request.draft_tokens.assign(
                verifier_tokens.begin(),
                verifier_tokens.end());
            const MTPSpecDecodeVerifierInputPlan row_plan =
                buildMTPSpecDecodeVerifierInputPlan(
                    shape,
                    {verifier_request});
            ASSERT_TRUE(row_plan.ok) << row_plan.error;
            ASSERT_TRUE(runner->setMTPSpecVerifierInputPlan(row_plan));
            ASSERT_TRUE(runner->setComputeRowIndexedAllPositionLogits(
                true,
                row_plan.compact_logit_row_count));
        }
        if (use_deferred_verifier_sync)
        {
            runner->setMTPAllPositionVerifierSyncDeferralEnabled(true);
        }
        ASSERT_TRUE(runner->setComputeAllPositionLogits(true));
        if (verify_compact_device_outcome)
        {
            ASSERT_GE(verifier_tokens.size(), 2u)
                << "compact outcome regression needs at least one draft row "
                   "and one bonus/correction row";
            ASSERT_LE(verifier_tokens.size(), 4u)
                << "Phase 9.8 resident publication proof currently targets "
                   "verifier rows M=2..4";
            if (runner->supportsGreedyAllPositionBatchOutcomeOnDevice())
            {
                ASSERT_NE(nullptr,
                          runner->prepareMTPVerifierInputTokensOnDeviceFromHostRow(
                              verifier_tokens.data(),
                              static_cast<int>(verifier_tokens.size()),
                              static_cast<int>(verifier_tokens.size() - 1)))
                    << "compact greedy verifier regression requires a materialized "
                       "device verifier-token row";
            }
            else
            {
                FAIL()
                    << "compact verifier outcome proofs must exercise the "
                       "resident GPU reducer; grouped host-plan synthesis is "
                       "retired";
            }
        }
        if (expect_grouped_moe_verifier_prefill)
        {
            ASSERT_TRUE(device.is_gpu())
                << "grouped MoE verifier promotion is currently a SingleDevice "
                   "GPU contract; CPU/TP/overlay lanes stay on replay until "
                   "their own strict continuation gates exist.";
            PerfStatsCollector::reset();
        }
        ASSERT_TRUE(runner->forward(
            verifier_tokens.data(),
            static_cast<int>(verifier_tokens.size())))
            << "all-position verifier forward failed";
        if (expect_grouped_moe_verifier_prefill)
        {
            const char *routed_counter = device.is_rocm()
                                             ? "rocm_moe_grouped_prefill_active_expert_grid_calls"
                                             : "cuda_moe_grouped_prefill_swiglu_path_calls";
            const auto records = PerfStatsCollector::snapshot({"kernel", "mtp"});
            EXPECT_TRUE(hasPerfCounter(records, "kernel", routed_counter))
                << "SingleDevice GPU MoE verifier must exercise the routed "
                   "grouped prefill path.\n"
                << PerfStatsCollector::summaryString({"kernel", "mtp"});
            EXPECT_TRUE(hasMTPPerfCounter(
                records,
                "moe_shared_grouped_decode_equivalent_verifier_prefill_rows"))
                << "SingleDevice GPU MoE verifier must exercise the standalone "
                   "shared-expert grouped table-prefill verifier path alongside "
                   "the routed grouped prefill path.\n"
                << PerfStatsCollector::summaryString({"kernel", "mtp"});
            EXPECT_FALSE(hasMTPPerfCounter(
                records,
                "moe_combined_decode_equivalent_verifier_prefill_rows"))
                << "The combined routed+shared verifier owner is not accepted "
                   "for production; strict full-model continuation gates failed "
                   "after component-only microbenches looked healthy.\n"
                << PerfStatsCollector::summaryString({"kernel", "mtp"});

            auto tag_equals = [](const PerfStatRecord &record,
                                 const char *key,
                                 const char *value) -> bool
            {
                const auto it = record.tags.find(key);
                return it != record.tags.end() && it->second == value;
            };
            const bool has_retired_row_serial_replay =
                std::any_of(
                    records.begin(),
                    records.end(),
                    [&](const PerfStatRecord &record)
                    {
                        if (record.kind != PerfStatRecord::Kind::Counter ||
                            record.domain != "mtp")
                        {
                            return false;
                        }
                        if (record.name ==
                            "decode_equivalent_sequential_verifier_runs")
                        {
                            return true;
                        }
                        return record.name ==
                                   "moe_decode_equivalent_verifier_prefill_runs" &&
                               tag_equals(record, "route", "single_row_decode");
                    });
            EXPECT_FALSE(has_retired_row_serial_replay)
                << "GPU MoE verifier rows must not quietly fall back to "
                   "row-serial replay.  Grouped router/expert verifier counters "
                   "are allowed; explicit single-row decode routes are not.\n"
                << PerfStatsCollector::summaryString({"kernel", "mtp"});
        }
        std::vector<int32_t> all_position_rows(verifier_tokens.size(), -1);
        ASSERT_TRUE(runner->sampleGreedyFromAllPositionLogitsOnDeviceRows(
            0,
            static_cast<int>(all_position_rows.size()),
            all_position_rows.data()));
        std::optional<DeviceSpeculativeVerifyBatchOutcome> compact_outcome;
        std::optional<DeviceSpeculativeOutcomeHandle> resident_outcome;
        if (verify_compact_device_outcome)
        {
            if (verify_device_resident_publication)
            {
                ASSERT_TRUE(device.is_gpu())
                    << "device-resident publication is a GPU mailbox contract";
                DeviceSpeculativeOutcomeHandle handle;
                ASSERT_TRUE(runner->verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
                    verifier_tokens.data(),
                    static_cast<int>(verifier_tokens.size()),
                    /*stop_tokens=*/nullptr,
                    /*stop_token_count=*/0,
                    &handle))
                    << "resident compact outcome is required before publishing "
                       "accepted verifier state from device metadata";
                ASSERT_TRUE(handle.valid());
                DeviceSpeculativeVerifyBatchOutcome materialized;
                ASSERT_TRUE(runner->materializeDeviceSpeculativeOutcomesForHostResponse(
                    handle,
                    &materialized))
                    << "test-only host materialization should not affect the "
                       "resident handle consumed by publication";
                ASSERT_TRUE(materialized.ok);
                compact_outcome = materialized;
                resident_outcome = std::move(handle);
            }
            else
            {
                if (!compact_outcome.has_value())
                {
                    DeviceSpeculativeVerifyBatchOutcome outcome;
                    ASSERT_TRUE(runner->verifyGreedyAllPositionBatchOutcomeOnDevice(
                        verifier_tokens.data(),
                        static_cast<int>(verifier_tokens.size()),
                        /*stop_tokens=*/nullptr,
                        /*stop_token_count=*/0,
                        &outcome));
                    ASSERT_TRUE(outcome.ok);
                    compact_outcome = outcome;
                }
            }
        }
        if (use_deferred_verifier_sync)
        {
            runner->setMTPAllPositionVerifierSyncDeferralEnabled(false);
        }
        ASSERT_TRUE(runner->setComputeAllPositionLogits(false));
        if (use_row_indexed_logits)
        {
            ASSERT_TRUE(runner->setComputeRowIndexedAllPositionLogits(false, 0));
            runner->clearMTPSpecVerifierInputPlan();
        }
        if (compact_outcome.has_value())
        {
            if (force_first_speculative_rejection)
            {
                ASSERT_TRUE(forced_rejection_expected_row0.has_value());
                ASSERT_FALSE(compact_outcome->all_speculative_accepted)
                    << "forced rejected-publication proof must make the first "
                       "speculative draft disagree with the verifier row";
                ASSERT_FALSE(compact_outcome->sampled_terminal)
                    << "a rejected correction is host-visible output, not a "
                       "bonus-ready terminal state";
                ASSERT_FALSE(all_position_rows.empty());
                EXPECT_EQ(compact_outcome->accepted_speculative_prefix, 0);
                EXPECT_EQ(compact_outcome->target_verifier_state_commit_count, 1);
                EXPECT_EQ(compact_outcome->rejected_verified_token, all_position_rows.front());
                EXPECT_EQ(all_position_rows.front(), *forced_rejection_expected_row0)
                    << "the forced-rejection fixture should keep row 0 equal to "
                       "the serial verifier sample before replacing draft row 1";
                EXPECT_NE(verifier_tokens[1], all_position_rows.front())
                    << "draft row 1 must be deliberately wrong so publication "
                       "commits only the first verifier state row";
            }
            else
            {
                ASSERT_TRUE(compact_outcome->sampled_terminal)
                    << "accepted verifier rows should expose the bonus ready token"
                    << "\nverifier_tokens="
                    << joinTokensMoEDiagnostic(verifier_tokens)
                    << "\nall_position_rows="
                    << joinTokensMoEDiagnostic(all_position_rows)
                    << "\ncompact.accepted_prefix="
                    << compact_outcome->accepted_speculative_prefix
                    << "\ncompact.commit_count="
                    << compact_outcome->target_verifier_state_commit_count
                    << "\ncompact.output_count="
                    << compact_outcome->output_token_count
                    << "\ncompact.ready_token="
                    << compact_outcome->ready_token
                    << "\ncompact.rejected_token="
                    << compact_outcome->rejected_verified_token
                    << "\ncompact.all_accepted="
                    << (compact_outcome->all_speculative_accepted ? "true" : "false")
                    << "\ncompact.consumed_rows="
                    << compact_outcome->consumed_verifier_rows;
                EXPECT_EQ(
                    compact_outcome->accepted_speculative_prefix,
                    static_cast<int>(verifier_tokens.size()) - 1);
                EXPECT_EQ(
                    compact_outcome->target_verifier_state_commit_count,
                    static_cast<int>(verifier_tokens.size()));
                EXPECT_EQ(compact_outcome->ready_token, all_position_rows.back())
                    << "compact device outcome must sample the same terminal row as "
                       "direct row argmax";
            }
        }
        const float *all_position_logits = runner->getAllPositionLogits();
        ASSERT_NE(all_position_logits, nullptr);
        std::vector<float> all_position_logits_copy(
            all_position_logits,
            all_position_logits + static_cast<size_t>(all_position_rows.size()) *
                                      static_cast<size_t>(vocab));
        const auto all_position_snapshots =
            capture_operation_snapshots
                ? captureMoERunnerSnapshots(*runner)
                : std::map<std::string, std::vector<float>>{};

        if (verify_published_state_continuation)
        {
            ASSERT_GE(verifier_tokens.size(), 2u)
                << "publication continuation proof needs at least one draft "
                   "row plus the verifier terminal row";
            ASSERT_LE(verifier_tokens.size(), 4u)
                << "Phase 9.8 direct resident publication promotion is bounded "
                   "to the M=2..4 verifier rows proven by the focused suite";
            ScopedEnvironmentValues kv_payload_probe_env({
                {"LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS", "1"},
                {"LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE", "1"},
            });
            /*
             * The all-position verifier can produce correct logits while still
             * leaving the live KV/GDN state in a non-decode-equivalent shape.
             * Exercise the public vLLM-style publication boundary directly:
             * publish the accepted verifier rows, continue from the bonus row,
             * and compare that continuation against serially decoding those
             * same accepted rows from the captured verifier base.
             */
            ASSERT_TRUE(compact_outcome.has_value())
                << "published-state continuation proof must use the compact "
                   "verifier outcome that production consumes";
            MTPDecodeCatchupGreedyRequest publication_request_for_plan;
            publication_request_for_plan.draft_tokens = verifier_tokens;
            publication_request_for_plan.base_sidecar_position =
                base_sidecar_position;
            publication_request_for_plan.allow_speculative_discard = true;
            publication_request_for_plan.verifier_path =
                "phase98_device_resident_publication_proof";
            publication_request_for_plan.implementation_name =
                "device_resident";
            publication_request_for_plan.verifier_base_checkpoint =
                &verifier_base;

            MTPSpecDecodeMetadataShape publication_shape;
            publication_shape.max_requests = 1;
            publication_shape.max_draft_tokens =
                static_cast<int>(verifier_tokens.size());
            const MTPSpecTransactionBatchPlan transaction_plan =
                buildMTPSpecTransactionBatchPlanFromDeviceRejectionOutcomes(
                    publication_shape,
                    {0},
                    runner->vocab_size(),
                    {publication_request_for_plan},
                    {*compact_outcome},
                    {static_cast<int32_t>(verifier_base.cached_tokens)});
            ASSERT_TRUE(transaction_plan.ok)
                << transaction_plan.error;
            const MTPSpecStepPlanBatch &batch =
                transaction_plan.step_plans;
            ASSERT_EQ(batch.steps.size(), 1u);
            const MTPSpecStepPlan &step = batch.steps.front();
            ASSERT_EQ(
                step.accepted_count,
                compact_outcome->target_verifier_state_commit_count);
            ASSERT_EQ(
                step.target_cached_tokens,
                static_cast<int>(verifier_base.cached_tokens) +
                    step.accepted_count);
            const int committed_verifier_rows = step.accepted_count;
            ASSERT_GT(committed_verifier_rows, 0)
                << "publication proof requires at least one accepted verifier row";
            ASSERT_LE(
                committed_verifier_rows,
                static_cast<int>(verifier_tokens.size()))
                << "publication proof cannot commit more verifier rows than "
                   "the all-position forward produced";
            const int32_t publication_ready_token =
                all_position_rows[static_cast<size_t>(
                    committed_verifier_rows - 1)];

            ASSERT_TRUE(verify_device_resident_publication)
                << "published-state continuation proofs must use compact "
                   "device-resident publication";
            ASSERT_TRUE(resident_outcome.has_value())
                << "device-resident publication requires the compact outcome "
                   "handle produced on the verifier stream";
            DeviceSpeculativePublicationRequest publication_request;
            publication_request.outcome = *resident_outcome;
            publication_request.request_count = 1;
            publication_request.max_draft_tokens =
                static_cast<int>(verifier_tokens.size());
            publication_request.publish_mtp_shifted_kv = true;

            std::string publication_error;
            ASSERT_TRUE(runner->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
                publication_request,
                &publication_error))
                << publication_error;
            const DeviceResidentLogicalSequenceStateHandle logical_state =
                runner->deviceResidentLogicalSequenceState();
            ASSERT_TRUE(logical_state.valid())
                << "device-resident publication must leave a typed logical "
                   "state mailbox for subsequent planning";
            ASSERT_EQ(logical_state.request_count, 1);
            ASSERT_NE(
                logical_state.targetSequenceLengthDeviceForRequest(0),
                nullptr);
            ASSERT_NE(
                logical_state.acceptedStateCountDeviceForRequest(0),
                nullptr);
            ASSERT_NE(
                logical_state.nextConditionTokenDeviceForRequest(0),
                nullptr);
            ASSERT_NE(
                logical_state.publicationOkFlagDeviceForRequest(0),
                nullptr);
            const PrefixRuntimeStateSnapshot published_state_probe =
                runner->prefixStateProbe();
            const bool published_state_is_device_owned = logical_state.valid();

            auto summarize_runtime_state =
                [](const PrefixRuntimeStateSnapshot &probe) -> std::string
            {
                auto summarize_cache =
                    [](const std::vector<PrefixKVCacheProbe> &caches)
                    -> std::string
                {
                    std::string out;
                    for (const auto &cache : caches)
                    {
                        if (!out.empty())
                            out += ";";
                        out += cache.owner + ":";
                        const size_t limit =
                            std::min<size_t>(cache.layers.size(), 8);
                        for (size_t i = 0; i < limit; ++i)
                        {
                            if (i > 0)
                                out += ",";
                            const auto &layer = cache.layers[i];
                            out += "L" +
                                   std::to_string(layer.global_layer) +
                                   "/S" + std::to_string(layer.seq_idx) +
                                   "=" + std::to_string(layer.cached_tokens) +
                                   "@" + std::to_string(layer.ring_head);
                        }
                        if (cache.layers.size() > limit)
                            out += ",...";
                    }
                    return out.empty() ? std::string("none") : out;
                };

                auto join_ints = [](const std::vector<int> &values)
                {
                    std::string out;
                    for (size_t i = 0; i < values.size(); ++i)
                    {
                        if (i > 0)
                            out += ",";
                        out += std::to_string(values[i]);
                    }
                    return out.empty() ? std::string("none") : out;
                };

                std::string gdn;
                const size_t gdn_limit =
                    std::min<size_t>(probe.gdn_layers.size(), 8);
                for (size_t i = 0; i < gdn_limit; ++i)
                {
                    if (i > 0)
                        gdn += ",";
                    const auto &layer = probe.gdn_layers[i];
                    gdn += "L" + std::to_string(layer.global_layer) +
                           "/r=" + std::to_string(layer.recurrence_hash) +
                           "/c=" + std::to_string(layer.conv_hash);
                }
                if (probe.gdn_layers.size() > gdn_limit)
                    gdn += ",...";
                if (gdn.empty())
                    gdn = "none";

                return "pos=" + std::to_string(probe.current_position) +
                       " positions=[" + join_ints(probe.positions) + "]" +
                       " seq=[" + join_ints(probe.sequence_lengths) + "]" +
                       " kv={" + summarize_cache(probe.kv_caches) + "}" +
                       " mtp={" + summarize_cache(probe.mtp_kv_caches) + "}" +
                       " hidden=" + (probe.has_hidden ? "1" : "0") +
                       " logits=" + (probe.has_logits ? "1" : "0") +
                       " gdn={" + gdn + "}";
            };

            auto first_gdn_state_mismatch =
                [](const PrefixRuntimeStateSnapshot &lhs,
                   const PrefixRuntimeStateSnapshot &rhs) -> std::string
            {
                if (lhs.gdn_layers.size() != rhs.gdn_layers.size())
                {
                    return "layer-count " +
                           std::to_string(lhs.gdn_layers.size()) + " vs " +
                           std::to_string(rhs.gdn_layers.size());
                }
                for (size_t i = 0; i < lhs.gdn_layers.size(); ++i)
                {
                    const auto &a = lhs.gdn_layers[i];
                    const auto &b = rhs.gdn_layers[i];
                    if (a.global_layer != b.global_layer ||
                        a.recurrence_hash != b.recurrence_hash ||
                        a.conv_hash != b.conv_hash ||
                        a.recurrence_all_zero != b.recurrence_all_zero ||
                        a.conv_all_zero != b.conv_all_zero)
                    {
                        return "layer=" + std::to_string(a.global_layer) +
                               " rec=" +
                               std::to_string(a.recurrence_hash) + "/" +
                               std::to_string(b.recurrence_hash) +
                               " conv=" + std::to_string(a.conv_hash) +
                               "/" + std::to_string(b.conv_hash);
                    }
                }
                return "none";
            };

            auto generate_continuation =
                [&](const std::string &label,
                    int32_t input_token,
                    int count,
                    std::vector<int32_t> *out) -> bool
            {
                out->clear();
                int32_t next_input = input_token;
                for (int i = 0; i < count; ++i)
                {
                    if (!runner->forward(&next_input, 1))
                    {
                        ADD_FAILURE() << label
                                      << " continuation forward failed at step "
                                      << i;
                        return false;
                    }
                    const std::string sample_label =
                        label + " continuation step " + std::to_string(i);
                    const int32_t sampled =
                        sample_current(sample_label.c_str());
                    out->push_back(sampled);
                    next_input = sampled;
                }
                return true;
            };

            std::vector<int32_t> published_continuation;
            constexpr int publication_continuation_tokens = 8;
            ASSERT_TRUE(generate_continuation(
                "published all-position state",
                publication_ready_token,
                publication_continuation_tokens,
                &published_continuation));

            ASSERT_TRUE(runner->restoreLivePrefixState(verifier_base));
            if (capture_operation_snapshots)
            {
                runner->clearSnapshots();
            }
            PrefixRuntimeStateSnapshot serial_row0_state_probe;
            int32_t serial_ready = -1;
            for (int row = 0; row < committed_verifier_rows; ++row)
            {
                ASSERT_TRUE(runner->forward(&verifier_tokens[row], 1))
                    << "serial publication reference row " << row
                    << " forward failed";
                const int32_t sampled =
                    sample_current(
                        ("serial publication reference row " +
                         std::to_string(row))
                            .c_str());
                ASSERT_EQ(
                    sampled,
                    all_position_rows[static_cast<size_t>(row)])
                    << "publication regression requires each committed "
                       "verifier row to match serial decode before comparing "
                       "continuations";
                if (row == 0)
                {
                    serial_row0_state_probe = runner->prefixStateProbe();
                }
                serial_ready = sampled;
            }
            ASSERT_EQ(serial_ready, publication_ready_token)
                << "publication regression requires the same ready token on "
                   "published and serial paths";
            const PrefixRuntimeStateSnapshot serial_state_probe =
                runner->prefixStateProbe();
            MTPRuntimeSnapshotComparisonOptions publication_compare_options;
            publication_compare_options.compare_main_kv_payload_hashes = false;
            publication_compare_options.compare_shifted_mtp_kv = false;
            publication_compare_options.compare_gdn_hashes = false;
            /*
             * The CPU all-position verifier uses grouped projection rows before
             * the GDN stage, while the serial oracle uses one-row GEMV.  The
             * exact GDN/KV bytes are covered by lower-level parity tests; this
             * publication regression checks logical metadata and then proves the
             * accepted state by replaying a continuation from both paths.
             */
            const MTPStateValidationResult publication_state_match =
                compareMTPRuntimeStateSnapshots(
                    serial_state_probe,
                    published_state_probe,
                    publication_compare_options);
            MTPRuntimeSnapshotComparisonOptions strict_publication_compare_options;
            strict_publication_compare_options.compare_main_kv_payload_hashes = true;
            strict_publication_compare_options.compare_shifted_mtp_kv = true;
            strict_publication_compare_options.compare_gdn_hashes = true;
            const MTPStateValidationResult strict_publication_state_match =
                compareMTPRuntimeStateSnapshots(
                    serial_state_probe,
                    published_state_probe,
                    strict_publication_compare_options);
            MTPRuntimeSnapshotComparisonOptions main_only_publication_compare_options;
            main_only_publication_compare_options.compare_main_kv_payload_hashes = true;
            main_only_publication_compare_options.compare_shifted_mtp_kv = false;
            main_only_publication_compare_options.compare_gdn_hashes = true;
            const MTPStateValidationResult main_only_publication_state_match =
                compareMTPRuntimeStateSnapshots(
                    serial_state_probe,
                    published_state_probe,
                    main_only_publication_compare_options);

            EXPECT_TRUE(verifierGDNStateByteIdentical(
                published_state_probe,
                serial_state_probe,
                "MoE device-resident publication accepted GDN/short-conv state"))
                << "Accepted verifier-row publication must restore exactly the "
                   "same recurrent buffers that serial decode produced.  This "
                   "is checked before continuation so stale host mirrors cannot "
                   "hide device-owned state drift."
                << "\ncondition_prefix_tokens="
                << joinTokensMoEDiagnostic({expected_tokens[0], expected_tokens[1]})
                << "\nverifier_tokens="
                << joinTokensMoEDiagnostic(verifier_tokens)
                << "\nall_position_rows="
                << joinTokensMoEDiagnostic(all_position_rows)
                << "\ncommitted_verifier_rows="
                << committed_verifier_rows
                << "\nfirst_gdn_mismatch="
                << first_gdn_state_mismatch(
                       serial_state_probe,
                       published_state_probe)
                << "\npublished_state={"
                << summarizeMoEPrefixRuntimeProbe(published_state_probe)
                << "}\nserial_state={"
                << summarizeMoEPrefixRuntimeProbe(serial_state_probe)
                << "}\nmtp_perfstats="
                << PerfStatsCollector::summaryString({"mtp"});

            if (published_state_is_device_owned)
            {
                EXPECT_TRUE(publication_state_match)
                    << "MTP all-position publication must publish the same "
                       "runtime metadata as serial accepted-row decode before "
                       "continuation from its device-owned mailbox"
                    << "\nreason=" << publication_state_match.reason
                    << "\nstrict_reason="
                    << strict_publication_state_match.reason
                    << "\nmain_only_strict_reason="
                    << main_only_publication_state_match.reason
                    << "\nfirst_gdn_mismatch="
                    << first_gdn_state_mismatch(
                           serial_state_probe,
                           published_state_probe)
                    << "\nfirst_gdn_mismatch_vs_serial_row0="
                    << first_gdn_state_mismatch(
                           serial_row0_state_probe,
                           published_state_probe)
                    << "\nfirst_gdn_mismatch_base_vs_after_pre_verifier_shifted_commit="
                    << (verifier_base_probe.has_value() &&
                                after_pre_verifier_shifted_commit_probe.has_value()
                            ? first_gdn_state_mismatch(
                                  *verifier_base_probe,
                                  *after_pre_verifier_shifted_commit_probe)
                            : std::string("not-captured"))
                    << "\npublished_state={"
                    << summarize_runtime_state(published_state_probe) << "}"
                    << "\nverifier_base_state={"
                    << (verifier_base_probe.has_value()
                            ? summarize_runtime_state(*verifier_base_probe)
                            : std::string("not-captured"))
                    << "}"
                    << "\nafter_pre_verifier_shifted_commit_state={"
                    << (after_pre_verifier_shifted_commit_probe.has_value()
                            ? summarize_runtime_state(
                                  *after_pre_verifier_shifted_commit_probe)
                            : std::string("not-captured"))
                    << "}"
                    << "\nserial_row0_state={"
                    << summarize_runtime_state(serial_row0_state_probe) << "}"
                    << "\nserial_state={"
                    << summarize_runtime_state(serial_state_probe) << "}"
                    << "\nmtp_perfstats="
                    << PerfStatsCollector::summaryString({"mtp"});
            }

            std::vector<int32_t> serial_continuation;
            ASSERT_TRUE(generate_continuation(
                "serial accepted state",
                serial_ready,
                publication_continuation_tokens,
                &serial_continuation));

            EXPECT_EQ(published_continuation, serial_continuation)
                << "MTP all-position state publication must be continuation-equivalent"
                << "\ncondition_prefix_tokens="
                << joinTokensMoEDiagnostic({expected_tokens[0], expected_tokens[1]})
                << "\nverifier_tokens="
                << joinTokensMoEDiagnostic(verifier_tokens)
                << "\nall_position_rows="
                << joinTokensMoEDiagnostic(all_position_rows)
                << "\ncommitted_verifier_rows="
                << committed_verifier_rows
                << "\npublished_continuation="
                << joinTokensMoEDiagnostic(published_continuation)
                << "\nserial_continuation="
                << joinTokensMoEDiagnostic(serial_continuation)
                << "\nstrict_state_match="
                << (strict_publication_state_match
                        ? std::string("ok")
                        : strict_publication_state_match.reason)
                << "\nmain_only_strict_state_match="
                << (main_only_publication_state_match
                        ? std::string("ok")
                        : main_only_publication_state_match.reason)
                << "\nmtp_perfstats="
                << PerfStatsCollector::summaryString({"mtp"});

            ASSERT_TRUE(runner->restoreLivePrefixState(verifier_base));
        }

        ASSERT_TRUE(runner->restoreLivePrefixState(verifier_base));
        std::vector<int32_t> serial_rows(verifier_tokens.size(), -1);
        std::vector<std::map<std::string, std::vector<float>>>
            serial_snapshots_by_row;
        serial_snapshots_by_row.reserve(verifier_tokens.size());
        std::vector<std::string> serial_top5_by_row;
        serial_top5_by_row.reserve(verifier_tokens.size());
        std::vector<std::vector<float>> serial_logits_by_row;
        serial_logits_by_row.reserve(verifier_tokens.size());

        auto logits_have_materialized_signal =
            [](const float *logits, int count) -> bool
        {
            if (!logits || count <= 0)
                return false;
            for (int i = 0; i < count; ++i)
            {
                const float value = logits[i];
                if (std::isfinite(value) && std::fabs(value) > 0.0f)
                    return true;
            }
            return false;
        };

        if (use_skip_gather)
        {
            /*
             * The production path above intentionally keeps decode logits on
             * device and samples through the GPU row samplers.  The serial
             * oracle below is a diagnostic numeric comparison, so it must
             * explicitly re-enable host gathering before reading
             * runner->logits(); otherwise the host buffer can legally remain a
             * zero mirror while device sampling still succeeds.
             */
            runner->setSkipLogitsGatherDecode(false);
            runner->setSkipLogitsGatherPrefill(false);
        }

        /*
         * Phase 9.7 treats every verifier row as an independent publication
         * candidate.  Replaying from the same verifier base for row k ensures
         * the row is compared with exactly k+1 serial decode forwards, not with
         * state accidentally inherited from a previous diagnostic row.
         */
        for (size_t row = 0; row < verifier_tokens.size(); ++row)
        {
            ASSERT_TRUE(runner->restoreLivePrefixState(verifier_base));
            runner->clearSnapshots();
            for (size_t token_idx = 0; token_idx <= row; ++token_idx)
            {
                const int32_t token = verifier_tokens[token_idx];
                ASSERT_TRUE(runner->forward(&token, 1))
                    << "serial row " << row
                    << " verifier forward failed at token index "
                    << token_idx;
            }
            const std::string sample_label =
                "serial verifier row " + std::to_string(row);
            serial_rows[row] = sample_current(sample_label.c_str());
            const float *serial_logits = runner->logits();
            ASSERT_NE(serial_logits, nullptr)
                << "serial verifier row " << row
                << " must expose logits for numeric equivalence metrics";
            ASSERT_TRUE(logits_have_materialized_signal(serial_logits, vocab))
                << "serial verifier row " << row
                << " produced an all-zero/non-finite host logits oracle; "
                   "disable skip-gather before diagnostic serial replay";
            serial_logits_by_row.emplace_back(
                serial_logits,
                serial_logits + static_cast<size_t>(vocab));
            serial_top5_by_row.push_back(topKSummary(runner->logits(), vocab, 5));
            if (capture_operation_snapshots)
            {
                serial_snapshots_by_row.push_back(
                    captureMoERunnerSnapshots(*runner));
            }
        }

        std::filesystem::path csv_path;
        std::vector<MoESnapshotCompareRow> rows;
        if (capture_operation_snapshots)
        {
            const auto result_dir = moeDiagnosticResultsDir();
            csv_path =
                result_dir / "main_verifier_all_position_vs_serial.csv";
            std::ofstream csv(csv_path);
            ASSERT_TRUE(csv.is_open()) << "failed to open " << csv_path;
            writeMoESnapshotCsvHeader(csv);
            ASSERT_EQ(serial_snapshots_by_row.size(), verifier_tokens.size());
            for (size_t row = 0; row < verifier_tokens.size(); ++row)
            {
                appendMoEAllPositionVerifierRows(
                    all_position_snapshots,
                    serial_snapshots_by_row[row],
                    static_cast<int>(row),
                    static_cast<int>(row + 1),
                    csv,
                    &rows);
            }
        }

        auto first_bad_row = [&](const std::string &comparison)
            -> const MoESnapshotCompareRow *
        {
            if (!capture_operation_snapshots)
                return nullptr;
            return firstMoEDiagnosticDivergence(
                rows,
                comparison,
                0.9999,
                /*include_sidecar_keys=*/true);
        };
        const std::string snapshot_diagnostic_location =
            capture_operation_snapshots
                ? csv_path.string()
                : std::string(
                      "disabled for publication proof; byte-state and "
                      "continuation gates remain active");

        for (size_t row = 0; row < verifier_tokens.size(); ++row)
        {
            const std::string comparison =
                "all_position_row" + std::to_string(row) +
                "_vs_serial_prefix" + std::to_string(row + 1);
            const float *row_logits =
                all_position_logits_copy.data() +
                row * static_cast<size_t>(vocab);
            const std::string all_position_top5 =
                topKSummary(row_logits, vocab, 5);
            EXPECT_TRUE(verifierLogitsNumericallyEquivalent(
                row_logits,
                serial_logits_by_row[row].data(),
                vocab,
                "all-position row " + std::to_string(row) +
                    " vs serial prefix " + std::to_string(row + 1)))
                << "\ncondition_prefix_tokens="
                << joinTokensMoEDiagnostic({expected_tokens[0], expected_tokens[1]})
                << "\nverifier_tokens="
                << joinTokensMoEDiagnostic(verifier_tokens)
                << "\nrow all-position top5=[" << all_position_top5 << "]"
                << "\nrow serial top5=[" << serial_top5_by_row[row] << "]"
                << "\ndiagnostic CSV: " << snapshot_diagnostic_location;
            EXPECT_EQ(all_position_rows[row], serial_rows[row])
                << "all-position row " << row
                << " must match " << (row + 1)
                << " serial verifier decode step(s)"
                << "\ncondition_prefix_tokens="
                << joinTokensMoEDiagnostic({expected_tokens[0], expected_tokens[1]})
                << "\nverifier_tokens="
                << joinTokensMoEDiagnostic(verifier_tokens)
                << "\nall_position_rows="
                << joinTokensMoEDiagnostic(all_position_rows)
                << "\nserial_rows="
                << joinTokensMoEDiagnostic(serial_rows)
                << "\nrow all-position top5=[" << all_position_top5 << "]"
                << "\nrow serial top5=[" << serial_top5_by_row[row] << "]"
                << "\nfirst divergence="
                << (first_bad_row(comparison)
                        ? describeMoEDiagnosticRow(*first_bad_row(comparison))
                        : std::string("none"))
                << "\ndiagnostic CSV: " << snapshot_diagnostic_location;
        }

        runner->disableSnapshotCapture();
    }

    inline void runMoEMainVerifierDecodeEquivalentRowsMatchSerialDecode(
        const MoEPrefixRestoreParityCase &test_case,
        int verifier_row_count)
    {
        ScopedMoEParityProductionMode production_mode(
            shouldForceMoEParityProductionMode(test_case));
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadMoEReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        if (moeReferenceInputsStoppedCurrentTest())
        {
            return;
        }
        ASSERT_GE(verifier_row_count, 1)
            << "decode-equivalent verifier-row proof must exercise at least one row";
        ASSERT_LE(verifier_row_count, 4)
            << "Phase 9.7 production proof currently targets M=1..4";
        ASSERT_GE(expected_tokens.size(), 2u)
            << "decode-equivalent verifier row proof needs two setup tokens";

        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cpu()
                                    : test_case.devices.front().toLocalDeviceId();

        InferenceRunnerConfig config;
        config.max_seq_len = test_case.max_seq_len;
        config.batch_size = 1;
        config.force_graph = true;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = parseKVCachePrecision(test_case.kv_cache_precision);
        config.use_mapped_memory = false;
        config.mtp.enabled = true;
        config.mtp.draft_tokens = verifier_row_count;
        config.moe_routed_expert_plan = test_case.moe_routed_expert_plan;

        auto proof_runner =
            createMoEVerifierProofRunner(test_case, model_path, device, config);
        ASSERT_TRUE(proof_runner.error.empty())
            << proof_runner.error;
        auto &runner = proof_runner.runner;
        ASSERT_NE(runner, nullptr);
        ASSERT_GT(runner->vocab_size(), 0);
        auto sample_current = [&](const char *label) -> int32_t
        {
            int32_t sampled = runner->sampleGreedyOnDevice();
            if (sampled >= 0)
            {
                return sampled;
            }
            const float *logits = runner->logits();
            ADD_FAILURE() << "sampleGreedyOnDevice failed after " << label
                          << "; falling back to host-visible logits";
            return argmaxToken(logits, runner->vocab_size());
        };

        runner->setSuppressTimeline(true);
        runner->enableSnapshotCapture();

        ASSERT_TRUE(runner->forward(
            prompt_tokens.data(),
            static_cast<int>(prompt_tokens.size())))
            << "prefill forward failed";
        EXPECT_EQ(sample_current("prefill"), expected_tokens[0]);

        int32_t token_after_setup = -1;
        for (int i = 0; i < 2; ++i)
        {
            const int32_t token = expected_tokens[static_cast<size_t>(i)];
            ASSERT_TRUE(runner->forward(&token, 1))
                << "serial setup forward failed at token index " << i;
            token_after_setup = sample_current("serial setup");
        }
        ASSERT_GE(token_after_setup, 0)
            << "serial setup must produce the first verifier input token";

        /*
         * The shared production catch-up helper owns both main decode state and
         * shifted-MTP cache maintenance.  The two setup tokens above were
         * ordinary serial forwards in this isolated proof, so prime the shifted
         * cache to the same base position production would have reached before
         * starting a speculative transaction.
         */
        const int setup_sidecar_position = runner->get_position();
        ASSERT_TRUE(runner->commitMTPShiftedRowFromCurrentTerminalHidden(
            expected_tokens[0],
            /*already_appended_tokens=*/0,
            /*allow_speculative_discard=*/true,
            setup_sidecar_position - 2))
            << "failed to prime shifted MTP cache for first setup token";
        ASSERT_TRUE(runner->commitMTPShiftedRowFromCurrentTerminalHidden(
            expected_tokens[1],
            /*already_appended_tokens=*/0,
            /*allow_speculative_discard=*/true,
            setup_sidecar_position - 1))
            << "failed to prime shifted MTP cache for second setup token";

        const PrefixStateSnapshot verifier_base = runner->captureLivePrefixState();
        ASSERT_TRUE(verifier_base.valid);

        std::vector<int32_t> verifier_tokens;
        verifier_tokens.reserve(static_cast<size_t>(verifier_row_count));
        int32_t next_verifier_token = token_after_setup;
        for (int i = 0; i < verifier_row_count; ++i)
        {
            verifier_tokens.push_back(next_verifier_token);
            ASSERT_TRUE(runner->forward(&next_verifier_token, 1))
                << "serial verifier-token extension failed at row " << i;
            next_verifier_token =
                sample_current("serial verifier-token extension");
            ASSERT_GE(next_verifier_token, 0)
                << "serial verifier-token extension must produce row "
                << (i + 1);
        }
        const int32_t expected_ready_token = next_verifier_token;

        ASSERT_TRUE(runner->restoreLivePrefixState(verifier_base))
            << "decode-equivalent row proof must restore the verifier base "
               "before running the shared production catch-up helper";

        const int vocab = runner->vocab_size();
        const int base_sidecar_position = runner->get_position();
        std::vector<std::vector<float>> catchup_logits_by_row;
        std::vector<int32_t> catchup_samples_by_row;
        catchup_logits_by_row.reserve(static_cast<size_t>(verifier_row_count));
        catchup_samples_by_row.reserve(static_cast<size_t>(verifier_row_count));

        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens = verifier_tokens;
        request.base_sidecar_position = base_sidecar_position;
        request.allow_speculative_discard = true;
        request.verifier_path = "phase97_decode_equivalent_row_proof";
        request.implementation_name = "shared_stepwise";
        request.verifier_base_checkpoint = &verifier_base;

        /*
         * This callback is the production shared verifier's row boundary. It
         * samples where OrchestrationRunner samples, then snapshots the full
         * logit distribution so Phase 9.7 proves cosine, relative L2, and
         * symmetric KL rather than only top-token equality.
         */
        auto sample_after_forward = [&](int32_t) -> int32_t
        {
            const int32_t sampled =
                sample_current("decode-equivalent catch-up row");
            const float *logits = runner->logits();
            if (!logits)
            {
                ADD_FAILURE()
                    << "decode-equivalent catch-up row did not expose logits";
                return sampled;
            }
            catchup_samples_by_row.push_back(sampled);
            catchup_logits_by_row.emplace_back(
                logits,
                logits + static_cast<size_t>(vocab));
            return sampled;
        };

        MTPDecodeCatchupGreedyResult catchup =
            runSharedStepwiseMTPDecodeCatchupGreedy(
                *runner,
                request,
                sample_after_forward);
        ASSERT_TRUE(catchup.ok) << catchup.error;
        EXPECT_EQ(catchup.accepted_tokens, verifier_tokens)
            << "The proof fixture builds verifier draft tokens from the serial "
               "oracle, so the shared stepwise verifier should accept every "
               "row before producing the ready token.";
        EXPECT_TRUE(catchup.all_speculative_accepted);
        EXPECT_EQ(catchup.ready_token, expected_ready_token);
        ASSERT_EQ(catchup_logits_by_row.size(),
                  static_cast<size_t>(verifier_row_count));
        ASSERT_EQ(catchup_samples_by_row.size(),
                  static_cast<size_t>(verifier_row_count));

        auto generate_continuation =
            [&](const std::string &label,
                int32_t input_token,
                int count,
                std::vector<int32_t> *out) -> bool
        {
            out->clear();
            int32_t next_input = input_token;
            for (int i = 0; i < count; ++i)
            {
                if (!runner->forward(&next_input, 1))
                {
                    ADD_FAILURE() << label
                                  << " continuation forward failed at step "
                                  << i;
                    return false;
                }
                const std::string sample_label =
                    label + " continuation step " + std::to_string(i);
                const int32_t sampled =
                    sample_current(sample_label.c_str());
                out->push_back(sampled);
                next_input = sampled;
            }
            return true;
        };

        std::vector<int32_t> catchup_continuation;
        constexpr int continuation_tokens = 4;
        ASSERT_TRUE(generate_continuation(
            "decode-equivalent catch-up state",
            catchup.ready_token,
            continuation_tokens,
            &catchup_continuation));

        ASSERT_TRUE(runner->restoreLivePrefixState(verifier_base));
        for (int32_t token : verifier_tokens)
        {
            ASSERT_TRUE(runner->forward(&token, 1))
                << "serial continuation reference failed while replaying "
                   "verifier token "
                << token;
        }
        const int32_t serial_ready_token =
            sample_current("serial continuation reference");
        EXPECT_EQ(serial_ready_token, catchup.ready_token);
        std::vector<int32_t> serial_continuation;
        ASSERT_TRUE(generate_continuation(
            "serial verifier state",
            serial_ready_token,
            continuation_tokens,
            &serial_continuation));
        EXPECT_EQ(catchup_continuation, serial_continuation)
            << "decode-equivalent catch-up state must continue exactly like "
               "serial decode"
            << "\nverifier_tokens="
            << joinTokensMoEDiagnostic(verifier_tokens)
            << "\ncatchup_continuation="
            << joinTokensMoEDiagnostic(catchup_continuation)
            << "\nserial_continuation="
            << joinTokensMoEDiagnostic(serial_continuation);

        for (size_t row = 0; row < verifier_tokens.size(); ++row)
        {
            ASSERT_TRUE(runner->restoreLivePrefixState(verifier_base));
            for (size_t token_idx = 0; token_idx <= row; ++token_idx)
            {
                const int32_t token = verifier_tokens[token_idx];
                ASSERT_TRUE(runner->forward(&token, 1))
                    << "serial row " << row
                    << " verifier forward failed at token index "
                    << token_idx;
            }
            const std::string sample_label =
                "serial verifier row " + std::to_string(row);
            const int32_t serial_sample =
                sample_current(sample_label.c_str());
            const float *serial_logits = runner->logits();
            ASSERT_NE(serial_logits, nullptr)
                << "serial verifier row " << row
                << " must expose logits for numeric equivalence metrics";

            EXPECT_TRUE(verifierLogitsNumericallyEquivalent(
                catchup_logits_by_row[row].data(),
                serial_logits,
                vocab,
                "decode-equivalent catch-up row " + std::to_string(row) +
                    " vs serial prefix " + std::to_string(row + 1)))
                << "\ncondition_prefix_tokens="
                << joinTokensMoEDiagnostic({expected_tokens[0], expected_tokens[1]})
                << "\nverifier_tokens="
                << joinTokensMoEDiagnostic(verifier_tokens)
                << "\nrow catch-up top5=["
                << topKSummary(catchup_logits_by_row[row].data(), vocab, 5)
                << "]\nrow serial top5=["
                << topKSummary(serial_logits, vocab, 5)
                << "]";
            EXPECT_EQ(catchup_samples_by_row[row], serial_sample)
                << "decode-equivalent row " << row
                << " must sample the same token as serial replay"
                << "\nverifier_tokens="
                << joinTokensMoEDiagnostic(verifier_tokens);
        }

        runner->disableSnapshotCapture();
    }

    /**
     * @brief Prove MoE grouped verifier rows match serial decode rows.
     *
     * This is the MoE counterpart to the dense grouped verifier-row gate.  It
     * runs the verifier tokens as one compact all-position graph forward, reads
     * row-indexed logits for every semantic verifier row, and compares them
     * against serial decode prefixes with strict sampled-token and distribution
     * metrics.  The helper intentionally does not publish live state from the
     * grouped forward; direct accepted-state publication remains a separate
     * continuation-equivalence gate.
     */
    inline void runMoEMainVerifierGroupedRowsMatchSerialDecode(
        const MoEPrefixRestoreParityCase &test_case,
        int verifier_row_count)
    {
        ScopedMoEParityProductionMode production_mode(
            shouldForceMoEParityProductionMode(test_case));
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadMoEReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        if (moeReferenceInputsStoppedCurrentTest())
        {
            return;
        }
        ASSERT_GE(verifier_row_count, 1)
            << "grouped verifier-row proof must exercise at least one row";
        ASSERT_LE(verifier_row_count, 4)
            << "Phase 9.8 production proof currently targets M=1..4";
        ASSERT_GE(expected_tokens.size(), 2u)
            << "grouped verifier row proof needs two setup tokens";

        const DeviceId device = test_case.devices.empty()
                                    ? DeviceId::cpu()
                                    : test_case.devices.front().toLocalDeviceId();

        InferenceRunnerConfig config;
        config.max_seq_len = test_case.max_seq_len;
        config.batch_size = 1;
        config.force_graph = true;
        config.activation_precision = ActivationPrecision::FP32;
        config.kv_cache_precision = parseKVCachePrecision(test_case.kv_cache_precision);
        config.use_mapped_memory = false;
        config.mtp.enabled = true;
        config.mtp.draft_tokens = verifier_row_count;
        config.moe_routed_expert_plan = test_case.moe_routed_expert_plan;

        auto proof_runner =
            createMoEVerifierProofRunner(test_case, model_path, device, config);
        ASSERT_TRUE(proof_runner.error.empty())
            << proof_runner.error;
        auto &runner = proof_runner.runner;
        ASSERT_NE(runner, nullptr);
        ASSERT_GT(runner->vocab_size(), 0);
        const bool grouped_snapshot_diagnostic =
            std::getenv("LLAMINAR_MOE_GROUPED_VERIFIER_SNAPSHOT_DIAGNOSTIC") != nullptr;
        ScopedMoEVerifierSnapshotCapture grouped_snapshot_capture;

        auto sample_current = [&](const char *label) -> int32_t
        {
            int32_t sampled = runner->sampleGreedyOnDevice();
            if (sampled >= 0)
            {
                return sampled;
            }
            const float *logits = runner->logits();
            ADD_FAILURE() << "sampleGreedyOnDevice failed after " << label
                          << "; falling back to host-visible logits";
            return argmaxToken(logits, runner->vocab_size());
        };

        runner->setSuppressTimeline(true);

        ASSERT_TRUE(runner->forward(
            prompt_tokens.data(),
            static_cast<int>(prompt_tokens.size())))
            << "prefill forward failed";
        EXPECT_EQ(sample_current("prefill"), expected_tokens[0]);

        int32_t token_after_setup = -1;
        for (int i = 0; i < 2; ++i)
        {
            const int32_t token = expected_tokens[static_cast<size_t>(i)];
            ASSERT_TRUE(runner->forward(&token, 1))
                << "serial setup forward failed at token index " << i;
            token_after_setup = sample_current("serial setup");
        }
        ASSERT_GE(token_after_setup, 0)
            << "serial setup must produce the first grouped verifier input token";

        const PrefixStateSnapshot verifier_base = runner->captureLivePrefixState();
        ASSERT_TRUE(verifier_base.valid);

        std::vector<int32_t> verifier_tokens;
        verifier_tokens.reserve(static_cast<size_t>(verifier_row_count));
        int32_t next_verifier_token = token_after_setup;
        for (int i = 0; i < verifier_row_count; ++i)
        {
            verifier_tokens.push_back(next_verifier_token);
            ASSERT_TRUE(runner->forward(&next_verifier_token, 1))
                << "serial verifier-token extension failed at row " << i;
            next_verifier_token =
                sample_current("serial verifier-token extension");
            ASSERT_GE(next_verifier_token, 0)
                << "serial verifier-token extension must produce row "
                << (i + 1);
        }
        const int32_t expected_ready_token = next_verifier_token;
        const PrefixRuntimeStateSnapshot live_serial_extension_probe =
            runner->prefixStateProbe();

        ASSERT_TRUE(runner->restoreLivePrefixState(verifier_base))
            << "grouped verifier proof must restore the verifier base before "
               "running the all-position candidate";
        if (grouped_snapshot_diagnostic)
        {
            grouped_snapshot_capture.enable(
                *runner,
                qwen36MoEGroupedVerifierSnapshotKeys());
            grouped_snapshot_capture.clear();
        }

        MTPSpecDecodeMetadataShape shape;
        shape.max_requests = 1;
        shape.max_draft_tokens = static_cast<int>(verifier_tokens.size());
        MTPSpecDecodeVerifierDraftRequest verifier_request;
        verifier_request.request_id = 0;
        verifier_request.draft_tokens.assign(
            verifier_tokens.begin(),
            verifier_tokens.end());
        const MTPSpecDecodeVerifierInputPlan row_plan =
            buildMTPSpecDecodeVerifierInputPlan(shape, {verifier_request});
        ASSERT_TRUE(row_plan.ok) << row_plan.error;

        ASSERT_TRUE(runner->setMTPSpecVerifierInputPlan(row_plan));
        ASSERT_TRUE(runner->setComputeRowIndexedAllPositionLogits(
            true,
            row_plan.compact_logit_row_count));
        ASSERT_TRUE(runner->setComputeAllPositionLogits(true));
        if (grouped_snapshot_diagnostic)
        {
            grouped_snapshot_capture.clear();
        }
        ASSERT_TRUE(runner->forward(
            verifier_tokens.data(),
            static_cast<int>(verifier_tokens.size())))
            << "MoE grouped all-position verifier forward failed";

        std::vector<int32_t> grouped_rows(verifier_tokens.size(), -1);
        ASSERT_TRUE(runner->sampleGreedyFromAllPositionLogitsOnDeviceRows(
            0,
            static_cast<int>(grouped_rows.size()),
            grouped_rows.data()));

        const int vocab = runner->vocab_size();
        const float *grouped_logits = runner->getAllPositionLogits();
        ASSERT_NE(grouped_logits, nullptr);
        std::vector<float> grouped_logits_copy(
            grouped_logits,
            grouped_logits + static_cast<size_t>(grouped_rows.size()) *
                                 static_cast<size_t>(vocab));
        const auto grouped_snapshots =
            grouped_snapshot_diagnostic
                ? captureMoERunnerSnapshots(*runner)
                : std::map<std::string, std::vector<float>>{};
        /*
         * The hard oracle for this focused grouped-forward proof is the
         * restore-and-replay serial decode pass below.  The live extension probe
         * remains useful diagnostic context, but MTP sidecar state can mutate
         * while discovering verifier tokens, so do not let that earlier online
         * extension supersede the replayed serial row oracle.
         */

        ASSERT_TRUE(runner->setComputeAllPositionLogits(false));
        ASSERT_TRUE(runner->setComputeRowIndexedAllPositionLogits(false, 0));
        runner->clearMTPSpecVerifierInputPlan();

        std::vector<int32_t> serial_rows(verifier_tokens.size(), -1);
        std::vector<std::vector<float>> serial_logits_by_row;
        serial_logits_by_row.reserve(verifier_tokens.size());
        std::vector<std::map<std::string, std::vector<float>>>
            serial_snapshots_by_row;
        serial_snapshots_by_row.reserve(verifier_tokens.size());
        std::vector<std::string> serial_top5_by_row;
        serial_top5_by_row.reserve(verifier_tokens.size());

        for (size_t row = 0; row < verifier_tokens.size(); ++row)
        {
            ASSERT_TRUE(runner->restoreLivePrefixState(verifier_base));
            if (grouped_snapshot_diagnostic)
            {
                grouped_snapshot_capture.clear();
            }
            for (size_t token_idx = 0; token_idx <= row; ++token_idx)
            {
                const int32_t token = verifier_tokens[token_idx];
                ASSERT_TRUE(runner->forward(&token, 1))
                    << "MoE serial row " << row
                    << " verifier forward failed at token index "
                    << token_idx;
            }
            const std::string sample_label =
                "MoE serial grouped verifier row " + std::to_string(row);
            serial_rows[row] = sample_current(sample_label.c_str());
            const float *serial_logits = runner->logits();
            ASSERT_NE(serial_logits, nullptr)
                << "MoE serial verifier row " << row
                << " must expose logits for grouped numeric equivalence metrics";
            serial_logits_by_row.emplace_back(
                serial_logits,
                serial_logits + static_cast<size_t>(vocab));
            serial_top5_by_row.push_back(topKSummary(serial_logits, vocab, 5));
            if (grouped_snapshot_diagnostic)
            {
                serial_snapshots_by_row.push_back(
                    captureMoERunnerSnapshots(*runner));
            }
        }

        std::string grouped_snapshot_diagnostic_summary;
        std::vector<std::string> grouped_snapshot_byte_failures(
            verifier_tokens.size());
        if (grouped_snapshot_diagnostic)
        {
            const auto result_dir = moeDiagnosticResultsDir();
            const auto csv_path =
                result_dir / "grouped_verifier_rows_vs_serial.csv";
            std::ofstream csv(csv_path);
            ASSERT_TRUE(csv.is_open()) << "failed to open " << csv_path;
            writeMoESnapshotCsvHeader(csv);
            std::vector<MoESnapshotCompareRow> diagnostic_rows;
            for (size_t row = 0; row < verifier_tokens.size(); ++row)
            {
                appendMoEAllPositionVerifierRows(
                    grouped_snapshots,
                    serial_snapshots_by_row[row],
                    static_cast<int>(row),
                    static_cast<int>(row + 1),
                    csv,
                    &diagnostic_rows);
            }

            std::vector<const MoESnapshotCompareRow *> first_byte_divergences;
            first_byte_divergences.reserve(verifier_tokens.size());
            std::set<std::string> divergent_layer_prefixes;
            std::set<std::string> raw_key_set;
            for (size_t row = 0; row < verifier_tokens.size(); ++row)
            {
                const std::string comparison =
                    "all_position_row" + std::to_string(row) +
                    "_vs_serial_prefix" + std::to_string(row + 1);
                const auto *first_bad = firstMoEDiagnosticByteDivergence(
                    diagnostic_rows,
                    comparison,
                    /*include_sidecar_keys=*/true);
                if (!first_bad)
                {
                    continue;
                }

                first_byte_divergences.push_back(first_bad);
                grouped_snapshot_byte_failures[row] =
                    describeMoEDiagnosticRow(*first_bad);
                const size_t separator = first_bad->key.find('_');
                if (first_bad->key.rfind("layer", 0) == 0 &&
                    separator != std::string::npos)
                {
                    divergent_layer_prefixes.insert(
                        first_bad->key.substr(0, separator + 1));
                }
                else
                {
                    raw_key_set.insert(first_bad->key);
                }
            }

            /*
             * A first mismatch such as layer34_MOE_EXPERT_OUTPUT is only
             * reproducible when the dump also includes that layer's normalized
             * input, route IDs/weights, shared branch, and adjacent projection
             * outputs.  Include every captured tensor from each first-bad layer
             * for every row in the grouped batch.
             */
            for (const auto &key : qwen36MoEGroupedVerifierSnapshotKeys())
            {
                for (const auto &prefix : divergent_layer_prefixes)
                {
                    if (key.rfind(prefix, 0) == 0)
                    {
                        raw_key_set.insert(key);
                        break;
                    }
                }
            }

            const bool explicit_raw_snapshot_dump =
                std::getenv("LLAMINAR_MOE_GROUPED_VERIFIER_RAW_SNAPSHOT_DUMP") != nullptr;
            if (explicit_raw_snapshot_dump && raw_key_set.empty())
            {
                raw_key_set.insert("layer3_FFN_NORM");
                raw_key_set.insert("layer3_MOE_ROUTING_INDICES");
                raw_key_set.insert("layer3_MOE_ROUTING_WEIGHTS");
                raw_key_set.insert("layer3_MOE_EXPERT_OUTPUT");
                raw_key_set.insert("layer3_MOE_COMBINED_OUTPUT");
            }

            const bool raw_snapshot_dump =
                explicit_raw_snapshot_dump || !first_byte_divergences.empty();
            std::filesystem::path raw_snapshot_path;
            if (raw_snapshot_dump)
            {
                /*
                 * The comparison CSV deliberately stores compact metrics so it
                 * remains readable across the full 40-layer graph.  When a
                 * model-level drift needs to become a focused kernel
                 * regression, however, we also need the exact production inputs.
                 * A byte mismatch enables this dump automatically; the explicit
                 * environment flag remains useful for collecting the historical
                 * layer-3 fixture from a passing run.
                 */
                raw_snapshot_path = result_dir / "grouped_verifier_raw_snapshots.csv";
                std::ofstream raw_csv(raw_snapshot_path);
                ASSERT_TRUE(raw_csv.is_open()) << "failed to open " << raw_snapshot_path;
                raw_csv << "runner,row,key,index,value,hex_bits\n";

                auto write_raw_vector =
                    [&](const char *runner_label,
                        size_t row,
                        const std::string &key,
                        const std::vector<float> &values)
                {
                    for (size_t i = 0; i < values.size(); ++i)
                    {
                        uint32_t bits = 0;
                        std::memcpy(&bits, values.data() + i, sizeof(bits));
                        raw_csv << runner_label << ','
                                << row << ','
                                << key << ','
                                << i << ','
                                << std::setprecision(9) << values[i] << ','
                                << "0x" << std::hex << std::setw(8) << std::setfill('0') << bits
                                << std::dec << std::setfill(' ') << '\n';
                    }
                };

                for (size_t row = 0; row < verifier_tokens.size(); ++row)
                {
                    for (const auto &key : raw_key_set)
                    {
                        const auto grouped_it = grouped_snapshots.find(key);
                        const auto serial_it = serial_snapshots_by_row[row].find(key);
                        if (grouped_it != grouped_snapshots.end() &&
                            serial_it != serial_snapshots_by_row[row].end() &&
                            !grouped_it->second.empty() &&
                            !serial_it->second.empty())
                        {
                            write_raw_vector(
                                "grouped",
                                row,
                                key,
                                selectMoEVerifierSnapshotRow(
                                    grouped_it->second,
                                    serial_it->second.size(),
                                    static_cast<int>(row)));
                            write_raw_vector(
                                "serial",
                                row,
                                key,
                                serial_it->second);
                        }
                    }
                }
            }

            std::ostringstream diag;
            diag << "\ngrouped_snapshot_csv=" << csv_path;
            if (!raw_snapshot_path.empty())
                diag << "\ngrouped_raw_snapshot_csv=" << raw_snapshot_path;
            diag << "\nserial_decode_path_perfstats:\n"
                 << PerfStatsCollector::summaryString(
                        {"moe_runtime_decode",
                         "kernel.rocm_moe_grouped_decode_fused_calls",
                         "kernel.rocm_moe_grouped_down_decode_calls"});
            for (size_t row = 0; row < verifier_tokens.size(); ++row)
            {
                const std::string comparison =
                    "all_position_row" + std::to_string(row) +
                    "_vs_serial_prefix" + std::to_string(row + 1);
                if (const auto *first_bad = firstMoEDiagnosticByteDivergence(
                        diagnostic_rows,
                        comparison,
                        /*include_sidecar_keys=*/true))
                {
                    diag << "\nfirst_byte_divergent_stage_row" << row << ": "
                         << describeMoEDiagnosticRow(*first_bad);
                }
            }
            grouped_snapshot_diagnostic_summary = diag.str();
        }

        (void)expected_ready_token;
        (void)live_serial_extension_probe;

        for (size_t row = 0; row < verifier_tokens.size(); ++row)
        {
            if (grouped_snapshot_diagnostic)
            {
                EXPECT_TRUE(grouped_snapshot_byte_failures[row].empty())
                    << "MoE grouped verifier operation row " << row
                    << " must be byte-identical to production serial decode"
                    << "\nverifier_tokens="
                    << joinTokensMoEDiagnostic(verifier_tokens)
                    << "\nfirst_byte_divergence="
                    << grouped_snapshot_byte_failures[row]
                    << grouped_snapshot_diagnostic_summary;
            }

            const float *grouped_row_logits =
                grouped_logits_copy.data() + row * static_cast<size_t>(vocab);
            EXPECT_TRUE(verifierLogitsByteIdentical(
                grouped_row_logits,
                serial_logits_by_row[row].data(),
                vocab,
                "MoE grouped all-position row " + std::to_string(row) +
                    " vs serial prefix " + std::to_string(row + 1)))
                << "\ncondition_prefix_tokens="
                << joinTokensMoEDiagnostic({expected_tokens[0], expected_tokens[1]})
                << "\nverifier_tokens="
                << joinTokensMoEDiagnostic(verifier_tokens)
                << "\nrow grouped top5=["
                << topKSummary(grouped_row_logits, vocab, 5)
                << "]\nrow serial top5=["
                << serial_top5_by_row[row] << "]"
                << grouped_snapshot_diagnostic_summary;
            EXPECT_EQ(grouped_rows[row], serial_rows[row])
                << "MoE grouped row " << row
                << " must sample the same token as serial replay"
                << "\nverifier_tokens="
                << joinTokensMoEDiagnostic(verifier_tokens)
                << "\ngrouped_rows="
                << joinTokensMoEDiagnostic(grouped_rows)
                << "\nserial_rows="
                << joinTokensMoEDiagnostic(serial_rows)
                << grouped_snapshot_diagnostic_summary;
        }

        grouped_snapshot_capture.disable();
    }

    inline void runMoEMTPSidecarStageBreakdownDiagnostic(
        const MoEPrefixRestoreParityCase &test_case,
        int decode_token_budget)
    {
        ScopedMoEParityProductionMode production_mode(
            shouldForceMoEParityProductionMode(test_case));
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadMoEReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        if (moeReferenceInputsStoppedCurrentTest())
        {
            return;
        }
        ASSERT_GT(decode_token_budget, 0);

        const std::filesystem::path metadata_path = firstEnvOrDefault(
            test_case.metadata_envs,
            test_case.default_metadata_path);
        ensurePyTorchMoEDecodeSnapshots(
            test_case,
            model_path,
            metadata_path,
            /*require_mtp_sidecar_snapshots=*/true);
        if (moeReferenceInputsStoppedCurrentTest())
        {
            return;
        }
        const auto pytorch_snapshot_dir = metadata_path.parent_path();

        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        const auto result_dir = moeDiagnosticResultsDir();
        const auto token_csv_path = result_dir / "mtp_sidecar_token_trace.csv";
        const auto snapshot_csv_path = result_dir / "mtp_sidecar_snapshot_breakdown.csv";
        std::ofstream token_csv(token_csv_path);
        std::ofstream snapshot_csv(snapshot_csv_path);
        ASSERT_TRUE(token_csv.is_open()) << "failed to open " << token_csv_path;
        ASSERT_TRUE(snapshot_csv.is_open()) << "failed to open " << snapshot_csv_path;
        token_csv << "sync_idx,runner,emitted_tokens,total_tokens,current_position,"
                     "logits_argmax,top5,mtp_draft_steps,mtp_accepted_tokens,"
                     "mtp_rejected_tokens,mtp_rollbacks,snapshot_count\n";
        writeMoESnapshotCsvHeader(snapshot_csv);

        auto write_trace = [&](int sync_idx,
                               const char *runner_name,
                               const MoEDiagnosticSnapshot &snapshot)
        {
            token_csv << sync_idx << ','
                      << runner_name << ','
                      << csvEscapeMoEDiagnostic(joinTokensMoEDiagnostic(snapshot.emitted_tokens)) << ','
                      << csvEscapeMoEDiagnostic(joinTokensMoEDiagnostic(snapshot.total_tokens)) << ','
                      << snapshot.current_position << ','
                      << snapshot.logits_argmax << ','
                      << csvEscapeMoEDiagnostic(snapshot.logits_top5) << ','
                      << snapshot.state.mtp_draft_steps << ','
                      << snapshot.state.mtp_accepted_tokens << ','
                      << snapshot.state.mtp_rejected_tokens << ','
                      << snapshot.state.mtp_rollbacks << ','
                      << snapshot.snapshots.size() << '\n';
            token_csv.flush();
        };

        std::vector<int32_t> baseline_tokens;
        std::vector<MoEDiagnosticSnapshot> baseline_snapshots_by_output_count;
        baseline_snapshots_by_output_count.resize(static_cast<size_t>(decode_token_budget) + 1);

        {
            auto baseline = factory->createFromOrchestrationConfig(
                makeMoEPrefixRestoreConfig(test_case, model_path, false, 2, false));
            ASSERT_NE(baseline, nullptr);
            ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
            baseline->setSamplingParams(greedy);
            baseline->setSkipLogitsGatherDecode(false);
            baseline->enableSnapshotCapture();
            ASSERT_TRUE(baseline->prefill(prompt_tokens)) << baseline->lastError();

            for (int output_count = 1; output_count <= decode_token_budget; ++output_count)
            {
                baseline->clearSnapshots();
                baseline->setDecodeStepTokenBudget(1);
                GenerationResult baseline_step = baseline->decodeStep();
                baseline->setDecodeStepTokenBudget(0);
                ASSERT_TRUE(baseline_step.error.empty()) << baseline_step.error;
                ASSERT_FALSE(baseline_step.tokens.empty()) << "baseline decodeStep produced no tokens";
                ASSERT_EQ(baseline_step.tokens.size(), 1u);
                baseline_tokens.insert(
                    baseline_tokens.end(),
                    baseline_step.tokens.begin(),
                    baseline_step.tokens.end());
                auto snapshot = captureMoEDiagnosticSnapshot(
                    *baseline,
                    baseline_step.tokens,
                    baseline_tokens);
                write_trace(output_count - 1, "baseline", snapshot);
                baseline_snapshots_by_output_count[static_cast<size_t>(output_count)] =
                    std::move(snapshot);
            }
            baseline->disableSnapshotCapture();
            baseline->shutdown();
        }

        std::vector<int32_t> mtp_tokens;
        std::vector<MoESnapshotCompareRow> all_snapshot_rows;
        std::set<std::string> observed_mtp_sidecar_keys;

        auto mtp = factory->createFromOrchestrationConfig(
            makeMoEPrefixRestoreConfig(test_case, model_path, false, 2, true));
        ASSERT_NE(mtp, nullptr);
        ASSERT_TRUE(mtp->initialize()) << mtp->lastError();
        mtp->setSamplingParams(greedy);
        mtp->setSkipLogitsGatherDecode(false);
        mtp->enableSnapshotCapture();
        ASSERT_TRUE(mtp->prefill(prompt_tokens)) << mtp->lastError();

        int sync_idx = 0;
        while (static_cast<int>(mtp_tokens.size()) < decode_token_budget)
        {
            const int remaining = decode_token_budget - static_cast<int>(mtp_tokens.size());
            mtp->clearSnapshots();
            mtp->setDecodeStepTokenBudget(remaining);
            GenerationResult mtp_step = mtp->decodeStep();
            mtp->setDecodeStepTokenBudget(0);
            ASSERT_TRUE(mtp_step.error.empty()) << mtp_step.error;
            ASSERT_FALSE(mtp_step.tokens.empty()) << "MTP decodeStep produced no tokens";
            mtp_tokens.insert(mtp_tokens.end(), mtp_step.tokens.begin(), mtp_step.tokens.end());

            ASSERT_LE(mtp_tokens.size(), baseline_snapshots_by_output_count.size() - 1);
            auto mtp_snapshot = captureMoEDiagnosticSnapshot(*mtp, mtp_step.tokens, mtp_tokens);
            for (const auto &entry : mtp_snapshot.snapshots)
            {
                if (isMTPSidecarSnapshotKey(entry.first))
                {
                    observed_mtp_sidecar_keys.insert(entry.first);
                }
            }
            write_trace(sync_idx, "mtp", mtp_snapshot);

            const auto &baseline_snapshot =
                baseline_snapshots_by_output_count[mtp_tokens.size()];
            std::vector<std::string> baseline_keys;
            baseline_keys.reserve(baseline_snapshot.snapshots.size());
            for (const auto &entry : baseline_snapshot.snapshots)
            {
                baseline_keys.push_back(entry.first);
            }
            const int pytorch_decode_step =
                std::max(0, static_cast<int>(mtp_tokens.size()) - 2);
            auto keys = unionSnapshotKeys(baseline_keys, mtp->getSnapshotKeys());
            keys = unionSnapshotKeys(
                keys,
                listMoEPyTorchSnapshotKeysForDecodeStep(
                    pytorch_snapshot_dir,
                    pytorch_decode_step));
            for (const auto &key : keys)
            {
                auto row = compareMoESnapshotKey(
                    baseline_snapshot,
                    *mtp,
                    sync_idx,
                    static_cast<int>(mtp_tokens.size()),
                    key);
                all_snapshot_rows.push_back(row);
                writeMoESnapshotCsvRow(snapshot_csv, row);

                if (pytorch_decode_step < test_case.decode_steps)
                {
                    const std::string pytorch_reference_key =
                        pytorchReferenceKeyForMoEDiagnosticKey(key);
                    const std::string pytorch_key =
                        "decode_step" + std::to_string(pytorch_decode_step) + "_" +
                        pytorch_reference_key;
                    const auto pytorch_data = loadMoEPyTorchSnapshot(
                        pytorch_snapshot_dir,
                        pytorch_key);

                    const auto baseline_it = baseline_snapshot.snapshots.find(key);
                    const float *baseline_data = nullptr;
                    size_t baseline_size = 0;
                    if (baseline_it != baseline_snapshot.snapshots.end() &&
                        !baseline_it->second.empty())
                    {
                        baseline_data = baseline_it->second.data();
                        baseline_size = baseline_it->second.size();
                    }

                    auto baseline_pt_row = compareMoESnapshotVectors(
                        pytorch_data,
                        baseline_data,
                        baseline_size,
                        sync_idx,
                        static_cast<int>(mtp_tokens.size()),
                        key,
                        pytorch_key,
                        "pytorch_vs_baseline",
                        "pytorch",
                        "baseline");
                    all_snapshot_rows.push_back(baseline_pt_row);
                    writeMoESnapshotCsvRow(snapshot_csv, baseline_pt_row);

                    size_t mtp_size = 0;
                    const float *mtp_data = mtp->getSnapshot(key, mtp_size);
                    auto mtp_pt_row = compareMoESnapshotVectors(
                        pytorch_data,
                        mtp_data,
                        mtp_size,
                        sync_idx,
                        static_cast<int>(mtp_tokens.size()),
                        key,
                        pytorch_key,
                        "pytorch_vs_mtp",
                        "pytorch",
                        "mtp");
                    all_snapshot_rows.push_back(mtp_pt_row);
                    writeMoESnapshotCsvRow(snapshot_csv, mtp_pt_row);
                }
            }

            ++sync_idx;
        }

        const auto mtp_state = mtp->prefixStateProbe();
        mtp->disableSnapshotCapture();
        mtp->shutdown();

        auto find_mtp_sidecar_row = [&](const std::string &key) -> const MoESnapshotCompareRow *
        {
            for (const auto &row : all_snapshot_rows)
            {
                if (row.comparison == "pytorch_vs_mtp" &&
                    row.key == key &&
                    row.sync_idx == 0)
                {
                    return &row;
                }
            }
            return nullptr;
        };

        const MoESnapshotCompareRow *sidecar_ffn_residual_row =
            find_mtp_sidecar_row("MTP_DECODE_SIDECAR_MTP0_FFN_RESIDUAL");
        const MoESnapshotCompareRow *sidecar_lm_head_row =
            find_mtp_sidecar_row("MTP_DECODE_SIDECAR_MTP0_LM_HEAD");

        ASSERT_EQ(baseline_tokens.size(), mtp_tokens.size())
            << "diagnostic CSVs:\n"
            << token_csv_path << '\n'
            << snapshot_csv_path;
        EXPECT_EQ(mtp_tokens, baseline_tokens)
            << "baseline tokens: " << joinTokensMoEDiagnostic(baseline_tokens)
            << "\nmtp tokens: " << joinTokensMoEDiagnostic(mtp_tokens)
            << "\n" << describeMoEDiagnosticRows(all_snapshot_rows)
            << "\ndiagnostic CSVs:\n"
            << token_csv_path << '\n'
            << snapshot_csv_path
            << "\nbaseline current_position="
            << baseline_snapshots_by_output_count.back().current_position
            << " mtp current_position=" << mtp_state.current_position;
        EXPECT_FALSE(mtp_state.mtp_bypassed) << mtp_state.mtp_bypass_reason;
        EXPECT_GE(mtp_state.mtp_verifier_runs, 1u);
        ASSERT_NE(sidecar_ffn_residual_row, nullptr)
            << "MTP sidecar diagnostic did not compare decode_step0_MTP0_FFN_RESIDUAL"
            << "\ndiagnostic CSVs:\n"
            << token_csv_path << '\n'
            << snapshot_csv_path;
        EXPECT_TRUE(isComparableMoEDiagnosticRow(*sidecar_ffn_residual_row))
            << describeMoEDiagnosticRow(*sidecar_ffn_residual_row);
        EXPECT_GE(sidecar_ffn_residual_row->cosine, 0.98)
            << "MTP sidecar block-output PyTorch reference mismatch: "
            << describeMoEDiagnosticRow(*sidecar_ffn_residual_row)
            << "\ndiagnostic CSVs:\n"
            << token_csv_path << '\n'
            << snapshot_csv_path;
        ASSERT_NE(sidecar_lm_head_row, nullptr)
            << "MTP sidecar diagnostic did not compare decode_step0_MTP0_LM_HEAD"
            << "\ndiagnostic CSVs:\n"
            << token_csv_path << '\n'
            << snapshot_csv_path;
        EXPECT_TRUE(isComparableMoEDiagnosticRow(*sidecar_lm_head_row))
            << describeMoEDiagnosticRow(*sidecar_lm_head_row);
        EXPECT_GE(sidecar_lm_head_row->cosine, 0.98)
            << "MTP sidecar PyTorch reference mismatch: "
            << describeMoEDiagnosticRow(*sidecar_lm_head_row)
            << "\ndiagnostic CSVs:\n"
            << token_csv_path << '\n'
            << snapshot_csv_path;

        const std::vector<std::string> required_sidecar_keys = {
            "MTP_DECODE_SIDECAR_MTP0_EMBEDDING",
            "MTP_DECODE_SIDECAR_MTP0_NORM_HIDDEN",
            "MTP_DECODE_SIDECAR_MTP0_CONCAT",
            "MTP_DECODE_SIDECAR_MTP0_FC",
            "MTP_DECODE_SIDECAR_MTP0_ATTENTION_NORM",
            "MTP_DECODE_SIDECAR_MTP0_Q_PROJECTION",
            "MTP_DECODE_SIDECAR_MTP0_ATTENTION_CONTEXT",
            "MTP_DECODE_SIDECAR_MTP0_ATTENTION_OUTPUT",
            "MTP_DECODE_SIDECAR_MTP0_FFN_NORM",
            "MTP_DECODE_SIDECAR_MTP0_MOE_ROUTER_OUTPUT",
            "MTP_DECODE_SIDECAR_MTP0_MOE_ROUTING_INDICES",
            "MTP_DECODE_SIDECAR_MTP0_MOE_ROUTING_WEIGHTS",
            "MTP_DECODE_SIDECAR_MTP0_MOE_EXPERT_OUTPUT",
            "MTP_DECODE_SIDECAR_MTP0_MOE_SHARED_EXPERT_OUTPUT",
            "MTP_DECODE_SIDECAR_MTP0_MOE_SHARED_GATE_OUTPUT",
            "MTP_DECODE_SIDECAR_MTP0_MOE_COMBINED_OUTPUT",
            "MTP_DECODE_SIDECAR_MTP0_FFN_RESIDUAL",
            "MTP_DECODE_SIDECAR_MTP0_FINAL_NORM",
            "MTP_DECODE_SIDECAR_MTP0_LM_HEAD",
        };
        std::vector<std::string> missing_sidecar_keys;
        for (const auto &key : required_sidecar_keys)
        {
            if (observed_mtp_sidecar_keys.count(key) == 0)
            {
                missing_sidecar_keys.push_back(key);
            }
        }
        EXPECT_TRUE(missing_sidecar_keys.empty())
            << "MTP sidecar diagnostic snapshots are missing required keys: "
            << joinStringsMoEDiagnostic(missing_sidecar_keys)
            << "\nobserved MTP keys: "
            << joinStringsMoEDiagnostic(std::vector<std::string>(
                   observed_mtp_sidecar_keys.begin(),
                   observed_mtp_sidecar_keys.end()))
            << "\ndiagnostic CSVs:\n"
            << token_csv_path << '\n'
            << snapshot_csv_path;
    }

    inline void runMoEMTPBenchmarkStyleSkipGatherParity(
        const MoEPrefixRestoreParityCase &test_case,
        int decode_token_budget,
        int mtp_draft_tokens = 1,
        MTPDepthPolicyConfig mtp_depth_policy = {},
        bool allow_reference_prefix_only = false,
        bool force_production_mode = true)
    {
        ScopedMoEParityProductionMode production_mode(
            force_production_mode &&
            shouldForceMoEParityProductionMode(test_case));
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadMoEReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        if (moeReferenceInputsStoppedCurrentTest())
        {
            return;
        }
        ASSERT_GT(decode_token_budget, 0);

        ScopedCudaMoEFusedVerifierPrefillRoutes fused_verifier_prefill_routes;
        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto run_mtp_decode = [&]() -> std::vector<int32_t>
        {
            std::vector<int32_t> tokens;
            auto runner = factory->createFromOrchestrationConfig(
                makeMoEPrefixRestoreConfig(
                    test_case,
                    model_path,
                    false,
                    2,
                    true,
                    mtp_draft_tokens,
                    mtp_depth_policy));
            EXPECT_NE(runner, nullptr);
            if (!runner)
            {
                return tokens;
            }

            if (!runner->initialize())
            {
                ADD_FAILURE() << runner->lastError();
                return tokens;
            }
            runner->setSamplingParams(greedy);
            runner->setSkipLogitsGatherDecode(true);
            runner->setSkipLogitsGatherPrefill(true);

            auto prefill_once = [&]() -> bool
            {
                if (!runner->prefill(prompt_tokens))
                {
                    ADD_FAILURE() << runner->lastError();
                    return false;
                }
                return true;
            };

            auto decode_loop = [&](std::vector<int32_t> *out_tokens) -> bool
            {
                int produced = 0;
                while (produced < decode_token_budget)
                {
                    const int remaining = decode_token_budget - produced;
                    runner->setDecodeStepTokenBudget(remaining);
                    GenerationResult step = runner->decodeStep();
                    runner->setDecodeStepTokenBudget(0);
                    if (!step.error.empty())
                    {
                        ADD_FAILURE() << step.error;
                        return false;
                    }
                    if (step.tokens.empty())
                    {
                        ADD_FAILURE()
                            << "MTP benchmark-style decode produced no tokens";
                        return false;
                    }
                    if (step.tokens.size() > static_cast<size_t>(remaining))
                    {
                        ADD_FAILURE()
                            << "MTP benchmark-style decode exceeded remaining token budget: "
                            << step.tokens.size() << " > " << remaining;
                        return false;
                    }
                    if (out_tokens)
                    {
                        out_tokens->insert(
                            out_tokens->end(),
                            step.tokens.begin(),
                            step.tokens.end());
                    }
                    produced += static_cast<int>(step.tokens.size());
                    if (!applyHostMoERebalanceIfNeeded(*runner))
                    {
                        ADD_FAILURE() << runner->lastError();
                        return false;
                    }
                    if (step.is_complete)
                    {
                        return true;
                    }
                }
                return true;
            };

            runner->setSuppressTimeline(true);
            runner->clearCache();
            if (!prefill_once() || !decode_loop(nullptr))
            {
                runner->shutdown();
                return tokens;
            }

            runner->setSuppressTimeline(false);
            runner->clearCache();
            if (prefill_once())
            {
                (void)decode_loop(&tokens);
            }
            runner->setSkipLogitsGatherDecode(false);
            runner->setSkipLogitsGatherPrefill(false);
            runner->shutdown();
            return tokens;
        };

        auto run_no_mtp_decode = [&]() -> std::vector<int32_t>
        {
            std::vector<int32_t> tokens;
            auto runner = factory->createFromOrchestrationConfig(
                makeMoEPrefixRestoreConfig(test_case, model_path, false, 2, false));
            EXPECT_NE(runner, nullptr);
            if (!runner)
            {
                return tokens;
            }

            if (!runner->initialize())
            {
                ADD_FAILURE() << runner->lastError();
                return tokens;
            }
            runner->setSamplingParams(greedy);
            runner->setSkipLogitsGatherDecode(true);
            runner->setSkipLogitsGatherPrefill(true);

            auto prefill_once = [&]() -> bool
            {
                if (!runner->prefill(prompt_tokens))
                {
                    ADD_FAILURE() << runner->lastError();
                    return false;
                }
                return true;
            };

            auto decode_loop = [&](std::vector<int32_t> *out_tokens) -> bool
            {
                for (int produced = 0; produced < decode_token_budget; ++produced)
                {
                    GenerationResult step = runner->decodeStep();
                    if (!step.error.empty())
                    {
                        ADD_FAILURE() << step.error;
                        return false;
                    }
                    if (step.tokens.size() != 1u)
                    {
                        ADD_FAILURE()
                            << "No-MTP benchmark-style reference produced "
                            << step.tokens.size() << " tokens at step "
                            << produced;
                        return false;
                    }
                    if (out_tokens)
                    {
                        out_tokens->push_back(step.tokens.front());
                    }
                    if (!applyHostMoERebalanceIfNeeded(*runner))
                    {
                        ADD_FAILURE() << runner->lastError();
                        return false;
                    }
                    if (step.is_complete)
                    {
                        break;
                    }
                }
                return true;
            };

            runner->setSuppressTimeline(true);
            runner->clearCache();
            if (!prefill_once() || !decode_loop(nullptr))
            {
                runner->shutdown();
                return tokens;
            }

            runner->setSuppressTimeline(false);
            runner->clearCache();
            if (prefill_once())
            {
                (void)decode_loop(&tokens);
            }
            runner->setSkipLogitsGatherDecode(false);
            runner->setSkipLogitsGatherPrefill(false);
            runner->shutdown();
            return tokens;
        };

        std::vector<int32_t> reference_tokens;
        ASSERT_FALSE(expected_tokens.empty());
        if (expected_tokens.size() >= static_cast<size_t>(decode_token_budget))
        {
            reference_tokens.assign(
                expected_tokens.begin(),
                expected_tokens.begin() + decode_token_budget);
        }
        else if (allow_reference_prefix_only)
        {
            reference_tokens.assign(expected_tokens.begin(), expected_tokens.end());
        }
        else
        {
            reference_tokens = run_no_mtp_decode();
            ASSERT_EQ(reference_tokens.size(), static_cast<size_t>(decode_token_budget))
                << "No-MTP reference tokens: "
                << joinTokensMoEDiagnostic(reference_tokens);
        }
        ASSERT_FALSE(reference_tokens.empty());

        const auto mtp_tokens = run_mtp_decode();
        ASSERT_EQ(mtp_tokens.size(), static_cast<size_t>(decode_token_budget))
            << "reference prefix tokens: " << joinTokensMoEDiagnostic(reference_tokens)
            << "\nmtp tokens: " << joinTokensMoEDiagnostic(mtp_tokens);

        std::vector<int32_t> mtp_prefix(
            mtp_tokens.begin(),
            mtp_tokens.begin() + std::min<size_t>(
                mtp_tokens.size(),
                reference_tokens.size()));
        ASSERT_EQ(mtp_prefix.size(), reference_tokens.size())
            << "reference tokens: " << joinTokensMoEDiagnostic(reference_tokens)
            << "\nmtp tokens: " << joinTokensMoEDiagnostic(mtp_tokens);
        EXPECT_EQ(mtp_prefix, reference_tokens)
            << "benchmark-style skip-gather decode diverged"
            << "\nreference tokens: " << joinTokensMoEDiagnostic(reference_tokens)
              << "\nmtp tokens: " << joinTokensMoEDiagnostic(mtp_tokens);
    }

    inline void runMoEMTPPersistentSidecarMetadataParity(
        const MoEPrefixRestoreParityCase &test_case)
    {
        ASSERT_FALSE(test_case.devices.empty());
        ASSERT_TRUE(test_case.devices.front().toLocalDeviceId().is_gpu())
            << "Phase 9.6 persistent sidecar metadata is a GPU graph-capture "
               "contract; CPU validates grouped verifier rows separately.";

        ScopedEnvironmentValues perf_stats_enabled({
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        });
        PerfStatsCollector::reset();

        runMoEMTPBenchmarkStyleSkipGatherParity(
            test_case,
            /*decode_token_budget=*/8,
            /*mtp_draft_tokens=*/1,
            {},
            /*allow_reference_prefix_only=*/true);

        const auto records = PerfStatsCollector::snapshot({"mtp"});
        EXPECT_TRUE(
            hasMTPPerfCounter(
                records,
                "moe_mtp_sidecar_runtime_table_creations") ||
            hasMTPPerfCounter(
                records,
                "moe_mtp_sidecar_runtime_table_reuses"))
            << "GPU MoE MTP sidecars must use persistent depth-scoped "
               "runtime tables for router/expert metadata.\n"
            << PerfStatsCollector::summaryString({"mtp"});
        EXPECT_TRUE(hasMTPPerfRecordTag(
            records,
            "moe_mtp_sidecar_runtime_table_creations",
            "histogram_sync",
            "disabled") ||
            hasMTPPerfRecordTag(
                records,
                "moe_mtp_sidecar_runtime_table_reuses",
                "histogram_sync",
                "disabled"))
            << "MTP sidecar runtime tables must not register request-level "
               "decode histogram sync callbacks.\n"
            << PerfStatsCollector::summaryString({"mtp"});
        EXPECT_TRUE(hasMTPPerfCounter(
            records,
            "grouped_decode_equivalent_greedy_verifier_runs"))
            << "GPU MoE MTP should run grouped decode-equivalent verifier rows.\n"
            << PerfStatsCollector::summaryString({"mtp"});
        EXPECT_TRUE(hasMTPPerfCounter(
            records,
            "grouped_outcome_device_resident_publication_uses"))
            << "GPU MoE MTP should publish grouped verifier outcomes through "
               "the device-resident mailbox.\n"
            << PerfStatsCollector::summaryString({"mtp"});
        EXPECT_TRUE(hasMTPPerfCounter(
            records,
            "device_resident_logical_state_mailboxes"))
            << "GPU MoE MTP must publish an arena-backed logical-state mailbox "
               "after the accepted-state transaction advances the live epoch.\n"
            << PerfStatsCollector::summaryString({"mtp"});
        EXPECT_FALSE(hasMTPPerfCounter(
            records,
            "device_resident_logical_state_mailbox_retargets"))
            << "The grouped device-outcome path must publish a fresh mailbox "
               "after any shifted-row mutation instead of retargeting the "
               "pre-publication logical-state handle.\n"
            << PerfStatsCollector::summaryString({"mtp"});
        EXPECT_FALSE(hasMTPPerfCounter(
            records,
            "decode_equivalent_sequential_verifier_runs"))
            << "GPU MoE MTP must not use retired row-serial verifier replay.\n"
            << PerfStatsCollector::summaryString({"mtp"});
        EXPECT_FALSE(hasMTPPerfCounter(
            records,
            "all_position_state_publication_verifier_runs"))
            << "MoE direct all-position publication must remain disabled "
               "outside the proven single-owner lane.\n"
            << PerfStatsCollector::summaryString({"mtp"});

        PerfStatsCollector::reset();
    }

    inline void runMoEMainVerifierUsesGroupedDecodeEquivalentPublication(
        const MoEPrefixRestoreParityCase &test_case)
    {
        ASSERT_FALSE(moECaseExpectsAllPositionSpecPublication(test_case))
            << "This regression covers the grouped decode-equivalent MoE "
               "verifier contract when direct all-position publication is "
               "intentionally not advertised.";

        ScopedEnvironmentValues perf_stats_enabled({
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
            {"LLAMINAR_MTP_VERIFY_COMMIT_REPLAY_CHECK", "1"},
            {"LLAMINAR_MTP_VERIFY_COMMIT_REPLAY_DEPTH", "4"},
        });
        PerfStatsCollector::reset();

        runMoEMTPBenchmarkStyleSkipGatherParity(
            test_case,
            /*decode_token_budget=*/8,
            /*mtp_draft_tokens=*/1,
            {},
            /*allow_reference_prefix_only=*/true);

        const auto records = PerfStatsCollector::snapshot({"mtp"});
        EXPECT_TRUE(hasMTPPerfCounter(
            records,
            "grouped_decode_equivalent_greedy_verifier_runs"))
            << "MoE MTP must choose grouped decode-equivalent verifier rows "
               "rather than row-serial replay.\n"
            << PerfStatsCollector::summaryString({"mtp"});
        EXPECT_TRUE(
            hasMTPPerfCounter(records, "grouped_outcome_device_resident_publication_uses") ||
            hasMTPPerfCounter(records, "grouped_outcome_host_publication_uses"))
            << "MoE MTP must publish grouped verifier state through an "
               "explicit grouped publication path.\n"
            << PerfStatsCollector::summaryString({"mtp"});
        EXPECT_FALSE(hasMTPPerfCounter(
            records,
            "decode_equivalent_sequential_verifier_runs"))
            << "MoE MTP must not use retired row-serial verifier replay.\n"
            << PerfStatsCollector::summaryString({"mtp"});
        EXPECT_FALSE(hasMTPPerfCounter(
            records,
            "all_position_state_publication_verifier_runs"))
            << "MoE MTP unexpectedly used direct all-position publication "
               "after that capability was withdrawn for this lane.\n"
            << PerfStatsCollector::summaryString({"mtp"});

        PerfStatsCollector::reset();
    }

    inline void expectCudaMoEMTPVerifierFusedPrefillPath(int expected_seq_len = 2)
    {
        const auto records = PerfStatsCollector::snapshot({"kernel", "mtp"});
        auto tag_equals = [](const PerfStatRecord &record,
                             const char *key,
                             const char *value) -> bool
        {
            const auto it = record.tags.find(key);
            return it != record.tags.end() && it->second == value;
        };
        const int expected_routed_top_k = 8;
        const int expected_routed_experts = 256;
        const int expected_total_slots = expected_seq_len * expected_routed_top_k;
        const int expected_tile_m = expected_seq_len <= 4 ? 2 : 4;
        const std::string seq_len_tag = std::to_string(expected_seq_len);
        const std::string total_slots_tag = std::to_string(expected_total_slots);
        const std::string tile_m_tag = std::to_string(expected_tile_m);

        if (expected_seq_len == 1)
        {
            auto has_decode_equivalent_stage = [&](const char *stage) -> bool
            {
                return std::any_of(
                    records.begin(),
                    records.end(),
                    [&](const PerfStatRecord &record)
                    {
                        return record.domain == "mtp" &&
                               record.name == "moe_decode_equivalent_verifier_prefill_runs" &&
                               tag_equals(record, "stage", stage) &&
                               tag_equals(record, "seq_len", "1");
                    });
            };

            EXPECT_FALSE(has_decode_equivalent_stage("router") ||
                         has_decode_equivalent_stage("routed_expert") ||
                         has_decode_equivalent_stage("shared_expert"))
                << "CUDA Qwen3.6 MoE M=1 verifier must use ordinary production "
                   "decode, not decode-equivalent oracle paths.\n"
                << PerfStatsCollector::summaryString({"kernel", "mtp"});

            auto has_kernel_decode_counter = [&](const char *name,
                                                 const char *source,
                                                 const char *route,
                                                 const char *active_slots) -> bool
            {
                return std::any_of(
                    records.begin(),
                    records.end(),
                    [&](const PerfStatRecord &record)
                    {
                        return record.domain == "kernel" &&
                               record.name == name &&
                               tag_equals(record, "source", source) &&
                               tag_equals(record, "route", route) &&
                               tag_equals(record, "active_slots", active_slots);
                    });
            };

            EXPECT_TRUE(has_kernel_decode_counter(
                "cuda_moe_grouped_decode_fused_calls",
                "runtime",
                "fused_block_down",
                "8") ||
                        has_kernel_decode_counter(
                            "cuda_moe_grouped_decode_fused_calls",
                            "runtime_static_table",
                            "fused_block_down",
                            "8"))
                << "CUDA Qwen3.6 MoE M=1 verifier routed expert must consume "
                   "the DeviceMoELayerRuntime top-k row through the same fused "
                   "runtime decode kernel used by serial decode.\n"
                << PerfStatsCollector::summaryString({"kernel", "mtp"});
            EXPECT_TRUE(has_kernel_decode_counter(
                "cuda_moe_grouped_decode_gateup_calls",
                "table",
                "kpart",
                "1") ||
                        has_kernel_decode_counter(
                            "cuda_moe_grouped_decode_gateup_calls",
                            "table",
                            "serial",
                            "1"))
                << "CUDA Qwen3.6 MoE M=1 shared expert must keep the ordinary "
                   "single-row grouped table gate/up decode shortcut enabled.\n"
                << PerfStatsCollector::summaryString({"kernel", "mtp"});
            EXPECT_TRUE(has_kernel_decode_counter(
                "cuda_moe_grouped_decode_down_calls",
                "table",
                "kpart",
                "1") ||
                        has_kernel_decode_counter(
                            "cuda_moe_grouped_decode_down_calls",
                            "table",
                            "serial",
                            "1"))
                << "CUDA Qwen3.6 MoE M=1 shared expert must keep the ordinary "
                   "single-row grouped table down decode shortcut enabled.\n"
                << PerfStatsCollector::summaryString({"kernel", "mtp"});

            const auto combined = std::find_if(
                records.begin(),
                records.end(),
                [&](const PerfStatRecord &record)
                {
                    return record.domain == "mtp" &&
                           record.name == "moe_combined_decode_equivalent_verifier_prefill_rows";
                });
            ASSERT_EQ(combined, records.end())
                << "CUDA Qwen3.6 MoE M=1 verifier unexpectedly ran the unaccepted "
                << "combined routed+shared owner.\n"
                << PerfStatsCollector::summaryString({"kernel", "mtp"});
            return;
        }

        auto describe_swiglu_path_candidates = [&]() -> std::string
        {
            std::ostringstream out;
            for (const PerfStatRecord &record : records)
            {
                if (record.name != "cuda_moe_grouped_prefill_swiglu_path_calls")
                    continue;
                out << "\n  count=" << record.count
                    << " total=" << record.value
                    << " tags={";
                bool first = true;
                for (const auto &[key, value] : record.tags)
                {
                    if (!first)
                        out << ";";
                    first = false;
                    out << key << "=" << value;
                }
                out << "}";
            }
            return out.str();
        };

        const auto match = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name == "cuda_moe_grouped_prefill_swiglu_path_calls" &&
                       tag_equals(record, "swiglu_path", "fused") &&
                       tag_equals(record, "seq_len", seq_len_tag.c_str()) &&
                       tag_equals(record, "total_slots", total_slots_tag.c_str()) &&
                       tag_equals(record, "top_k", "8") &&
                       tag_equals(record, "num_experts", "256") &&
                       tag_equals(record, "tile_m", tile_m_tag.c_str()) &&
                       tag_equals(record, "tile_n", "64") &&
                       /*
                        * Active experts are the unique experts selected across
                        * the verifier rows.  Real router outputs can repeat an
                        * expert across rows, so the correct proof is that the
                        * routed grouped verifier kernel ran for the whole
                        * `seq_len * top_k` route set, not that every route
                        * happened to hit a distinct expert.
                        */
                       tag_equals(record, "gateup_route", "kpart_prefill") &&
                       tag_equals(record, "down_route", "ordered_kpart_prefill") &&
                       tag_equals(record, "down_accumulation", "row_ordered_kpart");
            });

        ASSERT_NE(match, records.end())
            << "CUDA Qwen3.6 MoE MTP verifier path did not exercise the fused "
            << "routed grouped prefill SwiGLU/down kernels with the "
            << "verifier-sized tile. This is the accepted production contract "
            << "for the routed branch while shared-expert work is owned by the "
            << "safe composite decode-equivalent verifier stage.\n"
            << "candidate cuda_moe_grouped_prefill_swiglu_path_calls:"
            << describe_swiglu_path_candidates() << "\n"
            << PerfStatsCollector::summaryString({"kernel", "mtp"});
        EXPECT_GT(match->count, 0u);

        const auto shared_grouped_table_prefill = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.domain == "mtp" &&
                       record.name == "moe_shared_grouped_decode_equivalent_verifier_prefill_rows" &&
                       tag_equals(record, "route", "grouped_table_prefill") &&
                       tag_equals(record, "stage", "shared_expert");
            });
        ASSERT_NE(shared_grouped_table_prefill, records.end())
            << "CUDA Qwen3.6 MoE MTP verifier did not run the standalone "
            << "shared-expert grouped table-prefill verifier path.\n"
            << PerfStatsCollector::summaryString({"kernel", "mtp"});

        const auto combined = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.domain == "mtp" &&
                       record.name == "moe_combined_decode_equivalent_verifier_prefill_rows";
            });
        ASSERT_EQ(combined, records.end())
            << "CUDA Qwen3.6 MoE MTP verifier unexpectedly ran the unaccepted "
            << "combined routed+shared owner.\n"
            << PerfStatsCollector::summaryString({"kernel", "mtp"});

    }

    inline void runMoEMTPDynamicDepthRequestStateResetBenchmarkStyle(
        const MoEPrefixRestoreParityCase &test_case,
        int decode_token_budget)
    {
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadMoEReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        if (moeReferenceInputsStoppedCurrentTest())
        {
            return;
        }
        ASSERT_GT(decode_token_budget, 0);
        (void)expected_tokens;

        ScopedCudaMoEFusedVerifierPrefillRoutes fused_verifier_prefill_routes;
        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        MTPDepthPolicyConfig dynamic_policy;
        dynamic_policy.mode = MTPDepthPolicyMode::Dynamic;
        dynamic_policy.min_depth = 1;
        dynamic_policy.max_depth = 1;
        dynamic_policy.initial_depth = 1;

        auto runner = factory->createFromOrchestrationConfig(
            makeMoEPrefixRestoreConfig(
                test_case,
                model_path,
                false,
                2,
                true,
                1,
                dynamic_policy));
        ASSERT_NE(runner, nullptr);
        if (!runner)
        {
            return;
        }

        ASSERT_TRUE(runner->initialize()) << runner->lastError();
        runner->setSamplingParams(greedy);
        runner->setSkipLogitsGatherDecode(true);
        runner->setSkipLogitsGatherPrefill(true);

        auto prefill_once = [&]() -> bool
        {
            if (!runner->prefill(prompt_tokens))
            {
                ADD_FAILURE() << runner->lastError();
                return false;
            }
            return true;
        };

        auto decode_loop = [&](std::vector<int32_t> *out_tokens) -> bool
        {
            int produced = 0;
            while (produced < decode_token_budget)
            {
                const int remaining = decode_token_budget - produced;
                runner->setDecodeStepTokenBudget(remaining);
                GenerationResult step = runner->decodeStep();
                runner->setDecodeStepTokenBudget(0);
                if (!step.error.empty())
                {
                    ADD_FAILURE() << step.error;
                    return false;
                }
                if (step.tokens.empty())
                {
                    ADD_FAILURE() << "decode produced no tokens";
                    return false;
                }
                if (step.tokens.size() > static_cast<size_t>(remaining))
                {
                    ADD_FAILURE() << "decode exceeded remaining token budget: "
                                  << step.tokens.size() << " > " << remaining;
                    return false;
                }
                if (out_tokens)
                {
                    out_tokens->insert(
                        out_tokens->end(),
                        step.tokens.begin(),
                        step.tokens.end());
                }
                produced += static_cast<int>(step.tokens.size());
                if (!applyHostMoERebalanceIfNeeded(*runner))
                {
                    ADD_FAILURE() << runner->lastError();
                    return false;
                }
                if (step.is_complete)
                {
                    return true;
                }
            }
            return true;
        };

        runner->setSuppressTimeline(true);
        runner->clearCache();
        ASSERT_TRUE(prefill_once());
        ASSERT_TRUE(decode_loop(nullptr));
        const auto warmup_state = runner->prefixStateProbe();
        ASSERT_GT(warmup_state.mtp_draft_steps, 0u);

        runner->clearCache();
        const auto cleared_state = runner->prefixStateProbe();
        EXPECT_EQ(cleared_state.mtp_draft_steps, 0u);
        EXPECT_EQ(cleared_state.mtp_accepted_tokens, 0u);
        EXPECT_EQ(cleared_state.mtp_rejected_tokens, 0u);
        EXPECT_EQ(cleared_state.mtp_rollbacks, 0u);
        EXPECT_EQ(cleared_state.mtp_depth_policy_windows, 0u);
        EXPECT_EQ(cleared_state.mtp_depth_policy_updates, 0u);
        EXPECT_EQ(cleared_state.mtp_current_depth, 1);
        EXPECT_EQ(cleared_state.mtp_min_depth, 1);
        EXPECT_EQ(cleared_state.mtp_max_depth, 1);

        std::vector<int32_t> measured_tokens;
        runner->setSuppressTimeline(false);
        ASSERT_TRUE(prefill_once());
        ASSERT_TRUE(decode_loop(&measured_tokens));
        EXPECT_EQ(measured_tokens.size(), static_cast<size_t>(decode_token_budget))
            << "measured tokens: " << joinTokensMoEDiagnostic(measured_tokens);
        const auto measured_state = runner->prefixStateProbe();
        EXPECT_GT(measured_state.mtp_draft_steps, 0u);
        EXPECT_LE(measured_state.mtp_draft_steps,
                  static_cast<uint64_t>(decode_token_budget));

        runner->setSkipLogitsGatherDecode(false);
        runner->setSkipLogitsGatherPrefill(false);
        runner->shutdown();
    }

    inline void expectCudaMoEMTPCorrectionReplayFusedPrefillPath()
    {
        const auto records = PerfStatsCollector::snapshot(
            {"kernel.cuda_moe_grouped_prefill_swiglu_path_calls",
             "kernel.cuda_moe_shared_expert_prefill_group_calls"});
        auto tag_equals = [](const PerfStatRecord &record,
                             const char *key,
                             const char *value) -> bool
        {
            const auto it = record.tags.find(key);
            return it != record.tags.end() && it->second == value;
        };

        const auto routed_replay = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name == "cuda_moe_grouped_prefill_swiglu_path_calls" &&
                       tag_equals(record, "swiglu_path", "fused") &&
                       tag_equals(record, "seq_len", "1") &&
                       tag_equals(record, "total_slots", "8") &&
                       tag_equals(record, "active_expert_slots", "8") &&
                       tag_equals(record, "tile_m", "2") &&
                       tag_equals(record, "tile_n", "64") &&
                       tag_equals(record, "gateup_route", "kpart_prefill") &&
                       tag_equals(record, "down_route", "ordered_kpart_prefill") &&
                       tag_equals(record, "down_accumulation", "row_ordered_kpart");
            });

        ASSERT_NE(routed_replay, records.end())
            << "CUDA Qwen3.6 MoE MTP verifier-row correction replay did not "
            << "exercise the fused routed-expert grouped prefill path for seq_len=1. "
            << "Rejected-token replay must keep the fused path correct instead "
            << "of falling back to the slower single-token decode route.\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.cuda_moe_grouped_prefill_swiglu_path_calls"});
        EXPECT_GT(routed_replay->count, 0u);

        const auto shared_replay = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name == "cuda_moe_grouped_prefill_swiglu_path_calls" &&
                       tag_equals(record, "swiglu_path", "fused") &&
                       tag_equals(record, "seq_len", "1") &&
                       tag_equals(record, "total_slots", "1") &&
                       tag_equals(record, "active_expert_slots", "1") &&
                       tag_equals(record, "num_experts", "1") &&
                       tag_equals(record, "tile_m", "2") &&
                       tag_equals(record, "tile_n", "64") &&
                       tag_equals(record, "gateup_route", "kpart_prefill") &&
                       tag_equals(record, "down_route", "ordered_kpart_prefill") &&
                       tag_equals(record, "down_accumulation", "row_ordered_kpart");
            });

        ASSERT_NE(shared_replay, records.end())
            << "CUDA Qwen3.6 MoE MTP verifier-row correction replay did not "
            << "exercise the fused shared-expert grouped prefill path for seq_len=1.\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.cuda_moe_grouped_prefill_swiglu_path_calls"});
        EXPECT_GT(shared_replay->count, 0u);

        const auto shared_group = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name == "cuda_moe_shared_expert_prefill_group_calls" &&
                       tag_equals(record, "seq_len", "1") &&
                       tag_equals(record, "active_expert_slots", "1") &&
                       tag_equals(record, "top_k", "1");
            });

        ASSERT_NE(shared_group, records.end())
            << "CUDA shared expert grouped correction-replay setup did not run.\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.cuda_moe_shared_expert_prefill_group_calls"});
        EXPECT_GT(shared_group->count, 0u);
    }

    inline void expectCudaMoEMTPVerifierGDNProjectionFusedPath()
    {
        const auto records = PerfStatsCollector::snapshot(
            {"kernel.gdn_projection_route",
             "kernel.cuda_fp32_batched_fused_projection_calls"});
        auto tag_equals = [](const PerfStatRecord &record,
                             const char *key,
                             const char *value) -> bool
        {
            const auto it = record.tags.find(key);
            return it != record.tags.end() && it->second == value;
        };

        const auto qkv_z = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name == "gdn_projection_route" &&
                       tag_equals(record, "route", "native_subgroup") &&
                       tag_equals(record, "m", "2") &&
                       tag_equals(record, "k", "2048") &&
                       tag_equals(record, "projections", "2") &&
                       tag_equals(record, "names", "qkv+z");
            });
        ASSERT_NE(qkv_z, records.end())
            << "CUDA Qwen3.6 MoE MTP verifier GDN qkv+z projections must stay "
            << "on the native fused subgroup route.\n"
            << PerfStatsCollector::summaryString({"kernel.gdn_projection_route"});

        const auto alpha_beta = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name == "gdn_projection_route" &&
                       tag_equals(record, "route", "same_kernel_mixed_codebook_subgroup") &&
                       tag_equals(record, "m", "2") &&
                       tag_equals(record, "k", "2048") &&
                       tag_equals(record, "projections", "2") &&
                       tag_equals(record, "names", "alpha+beta");
            });
        ASSERT_NE(alpha_beta, records.end())
            << "CUDA Qwen3.6 MoE MTP verifier GDN alpha+beta projections must use "
            << "the graph-capturable FP32 batched cuBLAS route instead of single "
            << "projection fallbacks.\n"
            << PerfStatsCollector::summaryString({"kernel.gdn_projection_route"});

        const auto fp32_batch = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name == "cuda_fp32_batched_fused_projection_calls" &&
                       tag_equals(record, "m", "2") &&
                       tag_equals(record, "k", "2048") &&
                       tag_equals(record, "n", "32") &&
                       tag_equals(record, "projections", "2") &&
                       tag_equals(record, "route", "cublas_batched_same_a");
            });
        ASSERT_NE(fp32_batch, records.end())
            << "CUDA Qwen3.6 MoE MTP verifier GDN alpha+beta projections did not "
            << "record the cuBLAS batched fused projection call.\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.cuda_fp32_batched_fused_projection_calls"});

        const auto fallback = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name == "gdn_projection_route" &&
                       tag_equals(record, "route", "fallback_single") &&
                       tag_equals(record, "m", "2") &&
                       tag_equals(record, "k", "2048");
            });
        ASSERT_EQ(fallback, records.end())
            << "CUDA Qwen3.6 MoE MTP verifier GDN projection still has a "
            << "single-projection fallback. Keep the fused routes correct.\n"
            << PerfStatsCollector::summaryString({"kernel.gdn_projection_route"});
    }

    inline void runMoENoMTPBenchmarkStyleSkipGatherArgmaxParity(
        const MoEPrefixRestoreParityCase &test_case,
        int decode_token_budget,
        int repetitions = 3)
    {
        ScopedMoEParityProductionMode production_mode(
            shouldForceMoEParityProductionMode(test_case));
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadMoEReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        if (moeReferenceInputsStoppedCurrentTest())
        {
            return;
        }
        ASSERT_GT(decode_token_budget, 0);
        ASSERT_GT(repetitions, 0);
        (void)expected_tokens;

        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto run_no_mtp_decode = [&](int repetition, bool check_gathered_argmax) -> std::vector<int32_t>
        {
            std::vector<int32_t> tokens;
            auto runner = factory->createFromOrchestrationConfig(
                makeMoEPrefixRestoreConfig(test_case, model_path, false, 2, false));
            EXPECT_NE(runner, nullptr);
            if (!runner)
            {
                return tokens;
            }

            if (!runner->initialize())
            {
                ADD_FAILURE() << runner->lastError();
                return tokens;
            }
            runner->setSamplingParams(greedy);
            runner->setSkipLogitsGatherDecode(true);
            runner->setSkipLogitsGatherPrefill(true);
            const int vocab_size = runner->vocabSize();
            if (vocab_size <= 0)
            {
                ADD_FAILURE() << "runner reported invalid vocab size " << vocab_size;
                runner->shutdown();
                return tokens;
            }

            auto prefill_once = [&]() -> bool
            {
                if (!runner->prefill(prompt_tokens))
                {
                    ADD_FAILURE() << runner->lastError();
                    return false;
                }
                return true;
            };

            auto decode_loop = [&](std::vector<int32_t> *out_tokens) -> bool
            {
                for (int produced = 0; produced < decode_token_budget; ++produced)
                {
                    GenerationResult step = runner->decodeStep();
                    if (!step.error.empty())
                    {
                        ADD_FAILURE() << step.error;
                        return false;
                    }
                    if (step.tokens.size() != 1u)
                    {
                        ADD_FAILURE()
                            << "No-MTP benchmark-style decode repetition "
                            << repetition
                            << " check_gathered_argmax=" << check_gathered_argmax
                            << " produced " << step.tokens.size()
                            << " tokens for a single decode step";
                        return false;
                    }
                    const int32_t token = step.tokens.front();
                    if (check_gathered_argmax)
                    {
                        const float *logits = runner->lastLogits();
                        if (!logits)
                        {
                            ADD_FAILURE()
                                << "No-MTP benchmark-style decode repetition "
                                << repetition
                                << " produced no gathered logits for step "
                                << produced;
                            return false;
                        }
                        const int top = argmaxToken(logits, vocab_size);
                        if (token != top)
                        {
                            ADD_FAILURE()
                                << "No-MTP benchmark-style GPU greedy sample "
                                << "does not match gathered logits argmax at repetition "
                                << repetition
                                << " step " << produced
                                << "\ntoken=" << token
                                << " gathered_argmax=" << top
                                << "\ntop-5: " << topKSummary(logits, vocab_size);
                            return false;
                        }
                    }
                    if (out_tokens)
                    {
                        out_tokens->push_back(token);
                    }
                    if (!applyHostMoERebalanceIfNeeded(*runner))
                    {
                        ADD_FAILURE() << runner->lastError();
                        return false;
                    }
                    if (step.is_complete)
                    {
                        break;
                    }
                }
                return true;
            };

            runner->setSuppressTimeline(true);
            runner->clearCache();
            if (!prefill_once() || !decode_loop(nullptr))
            {
                runner->shutdown();
                return tokens;
            }

            runner->setSuppressTimeline(false);
            runner->clearCache();
            if (prefill_once())
            {
                (void)decode_loop(&tokens);
            }
            runner->setSkipLogitsGatherDecode(false);
            runner->setSkipLogitsGatherPrefill(false);
            runner->shutdown();
            return tokens;
        };

        for (int repetition = 0; repetition < repetitions; ++repetition)
        {
            const auto tokens = run_no_mtp_decode(repetition, true);
            ASSERT_EQ(tokens.size(), static_cast<size_t>(decode_token_budget))
                << "repetition=" << repetition
                << "\nactual tokens: " << joinTokensMoEDiagnostic(tokens);
            if (!expected_tokens.empty())
            {
                const size_t prefix = std::min(tokens.size(), expected_tokens.size());
                ASSERT_GT(prefix, 0u);
                std::vector<int32_t> actual_prefix(
                    tokens.begin(),
                    tokens.begin() + static_cast<std::ptrdiff_t>(prefix));
                std::vector<int32_t> expected_prefix(
                    expected_tokens.begin(),
                    expected_tokens.begin() + static_cast<std::ptrdiff_t>(prefix));
                EXPECT_EQ(actual_prefix, expected_prefix)
                    << "No-MTP benchmark-style CUDA MoE decode must match the "
                    << "stable PyTorch-covered prefix. The benchmark prompt has "
                    << "known near-tie branches beyond this prefix, so longer "
                    << "exact-token equality is covered by MTP transaction replay "
                    << "rather than fresh-runner token identity."
                    << "\nexpected prefix: "
                    << joinTokensMoEDiagnostic(expected_prefix)
                    << "\nactual tokens: " << joinTokensMoEDiagnostic(tokens);
            }
        }
    }

    inline void runMoEMTPBudgetOneStepMatchesReference(
        const MoEPrefixRestoreParityCase &test_case,
        int decode_token_budget)
    {
        ScopedMoEParityProductionMode production_mode(
            shouldForceMoEParityProductionMode(test_case));
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadMoEReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        if (moeReferenceInputsStoppedCurrentTest())
        {
            return;
        }
        ASSERT_GT(decode_token_budget, 1);
        ASSERT_GE(expected_tokens.size(), static_cast<size_t>(decode_token_budget));

        ScopedCudaMoEFusedVerifierPrefillRoutes fused_verifier_prefill_routes;
        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto runner = factory->createFromOrchestrationConfig(
            makeMoEPrefixRestoreConfig(
                test_case,
                model_path,
                false,
                2,
                true,
                1));
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();
        runner->setSamplingParams(greedy);
        runner->setSkipLogitsGatherDecode(true);
        runner->setSkipLogitsGatherPrefill(true);
        ASSERT_TRUE(runner->prefill(prompt_tokens)) << runner->lastError();

        std::vector<int32_t> tokens;
        tokens.reserve(static_cast<size_t>(decode_token_budget));
        for (int produced = 0; produced < decode_token_budget; ++produced)
        {
            runner->setDecodeStepTokenBudget(1);
            GenerationResult step = runner->decodeStep();
            runner->setDecodeStepTokenBudget(0);
            ASSERT_TRUE(step.error.empty()) << step.error;
            ASSERT_EQ(step.tokens.size(), 1u)
                << "budget-limited MTP decode should emit exactly one token per step";
            tokens.push_back(step.tokens.front());
            ASSERT_TRUE(applyHostMoERebalanceIfNeeded(*runner)) << runner->lastError();
            if (step.is_complete)
                break;
        }

        runner->setSkipLogitsGatherDecode(false);
        runner->setSkipLogitsGatherPrefill(false);
        runner->shutdown();

        std::vector<int32_t> reference_tokens(
            expected_tokens.begin(),
            expected_tokens.begin() + decode_token_budget);
        ASSERT_EQ(tokens, reference_tokens)
            << "budget-limited MTP direct emits must advance main and shifted-cache state"
            << "\nreference tokens: " << joinTokensMoEDiagnostic(reference_tokens)
            << "\nmtp tokens: " << joinTokensMoEDiagnostic(tokens);
    }

    inline void runMoEIncrementalDecodeMatchesFullContext(
        const MoEPrefixRestoreParityCase &test_case)
    {
        ScopedMoEParityProductionMode production_mode(
            shouldForceMoEParityProductionMode(test_case));
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadMoEReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        if (moeReferenceInputsStoppedCurrentTest())
        {
            return;
        }
        ASSERT_GE(expected_tokens.size(), 2u);

        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto incremental = factory->createFromOrchestrationConfig(
            makeMoEPrefixRestoreConfig(test_case, model_path, false, 2, false));
        ASSERT_NE(incremental, nullptr);
        ASSERT_TRUE(incremental->initialize()) << incremental->lastError();
        incremental->setSamplingParams(greedy);
        incremental->setSkipLogitsGatherDecode(false);

        ASSERT_TRUE(incremental->prefill(prompt_tokens)) << incremental->lastError();
        const float *prefill_logits = incremental->lastLogits();
        ASSERT_NE(prefill_logits, nullptr);
        const int vocab_size = incremental->vocabSize();
        ASSERT_GT(vocab_size, 0);
        const int prefill_top = argmaxToken(prefill_logits, vocab_size);
        EXPECT_EQ(prefill_top, expected_tokens[0])
            << "prefill top-5: " << topKSummary(prefill_logits, vocab_size);

        GenerationResult first = incremental->decodeStep();
        ASSERT_TRUE(first.error.empty()) << first.error;
        ASSERT_EQ(first.tokens.size(), 1u);
        EXPECT_EQ(first.tokens[0], expected_tokens[0]);

        GenerationResult second = incremental->decodeStep();
        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_EQ(second.tokens.size(), 1u);
        const float *incremental_logits = incremental->lastLogits();
        ASSERT_NE(incremental_logits, nullptr);
        const int incremental_top = argmaxToken(incremental_logits, vocab_size);
        const std::string incremental_top5 = topKSummary(incremental_logits, vocab_size);
        incremental->shutdown();

        std::vector<int32_t> full_context_tokens = prompt_tokens;
        full_context_tokens.push_back(expected_tokens[0]);
        auto full_context = factory->createFromOrchestrationConfig(
            makeMoEPrefixRestoreConfig(test_case, model_path, false, 2, false));
        ASSERT_NE(full_context, nullptr);
        ASSERT_TRUE(full_context->initialize()) << full_context->lastError();
        ASSERT_TRUE(full_context->prefill(full_context_tokens)) << full_context->lastError();
        const float *full_logits = full_context->lastLogits();
        ASSERT_NE(full_logits, nullptr);
        const int full_top = argmaxToken(full_logits, full_context->vocabSize());
        const std::string full_top5 = topKSummary(full_logits, vocab_size);

        full_context->shutdown();

        EXPECT_EQ(full_top, expected_tokens[1])
            << "full-context top-5: " << full_top5;
        EXPECT_EQ(incremental_top, full_top)
            << "incremental top-5: " << incremental_top5
            << "\nfull-context top-5: " << full_top5;
        EXPECT_EQ(second.tokens[0], full_top)
            << "decodeStep sampled a token that is not the gathered-logits argmax; "
            << "incremental top-5: " << incremental_top5;
    }

} // namespace llaminar2::test::parity::qwen36
