#pragma once

#include "Qwen36DenseParityTestBase.h"
#include "backends/BackendManager.h"
#include "backends/HardwareInventory.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/moe/MoEExpertParallelPlan.h"
#include "kernels/KernelFactory.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <cnpy.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>

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
        int decode_steps = 3;
        int max_seq_len = 96;
        int required_cuda_devices = 0;
        int required_rocm_devices = 0;
        int required_cpu_sockets = 0;
        std::shared_ptr<MoEExpertParallelPlan> moe_expert_parallel_plan;
    };

    inline size_t gib(size_t value)
    {
        return value * 1024ull * 1024ull * 1024ull;
    }

    inline size_t mib(size_t value)
    {
        return value * 1024ull * 1024ull;
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

    inline std::string formatMiBForSkip(size_t bytes)
    {
        std::ostringstream oss;
        oss << (bytes / mib(1)) << " MiB";
        return oss.str();
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

    class ScopedMoEParityDeterministicMode
    {
    public:
        explicit ScopedMoEParityDeterministicMode(bool enabled)
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

            setenv("LLAMINAR_DETERMINISTIC", "1", 1);
            mutableDebugEnv().reload();
#ifdef HAVE_CUDA
            cudaNativeVNNIPrefill_setDeterministicMode(true);
#endif
            llaminar::v2::kernels::KernelFactory::clearCache();
        }

        ~ScopedMoEParityDeterministicMode()
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

        ScopedMoEParityDeterministicMode(const ScopedMoEParityDeterministicMode &) = delete;
        ScopedMoEParityDeterministicMode &operator=(const ScopedMoEParityDeterministicMode &) = delete;

    private:
        bool enabled_ = false;
        bool had_old_deterministic_env_ = false;
        std::string old_deterministic_env_;
#ifdef HAVE_CUDA
        bool old_cuda_prefill_deterministic_ = false;
#endif
    };

    class ScopedCudaMoEFusedVerifierPrefillRoutes
    {
    public:
        ScopedCudaMoEFusedVerifierPrefillRoutes()
            : old_gateup_kpart_decode_(mutableDebugEnv().gemm.cuda_moe_gateup_kpart_decode),
              old_down_kpart_decode_(mutableDebugEnv().gemm.cuda_moe_down_kpart_decode),
              old_prefill_fuse_swiglu_(mutableDebugEnv().gemm.cuda_moe_prefill_fuse_swiglu),
              old_prefill_tile_m_(mutableDebugEnv().gemm.cuda_moe_prefill_tile_m),
              old_grouped_prefill_(mutableDebugEnv().rocm.moe_grouped_prefill)
        {
            auto &gemm = mutableDebugEnv().gemm;
            const bool allow_split_k_decode = !gemm.deterministic;
            gemm.cuda_moe_gateup_kpart_decode = allow_split_k_decode;
            gemm.cuda_moe_down_kpart_decode = allow_split_k_decode;
            gemm.cuda_moe_prefill_fuse_swiglu = true;
            gemm.cuda_moe_prefill_tile_m = 0;
            mutableDebugEnv().rocm.moe_grouped_prefill = true;
            llaminar::v2::kernels::KernelFactory::clearCache();
        }

        ~ScopedCudaMoEFusedVerifierPrefillRoutes()
        {
            auto &gemm = mutableDebugEnv().gemm;
            gemm.cuda_moe_gateup_kpart_decode = old_gateup_kpart_decode_;
            gemm.cuda_moe_down_kpart_decode = old_down_kpart_decode_;
            gemm.cuda_moe_prefill_fuse_swiglu = old_prefill_fuse_swiglu_;
            gemm.cuda_moe_prefill_tile_m = old_prefill_tile_m_;
            mutableDebugEnv().rocm.moe_grouped_prefill = old_grouped_prefill_;
            llaminar::v2::kernels::KernelFactory::clearCache();
        }

        ScopedCudaMoEFusedVerifierPrefillRoutes(const ScopedCudaMoEFusedVerifierPrefillRoutes &) = delete;
        ScopedCudaMoEFusedVerifierPrefillRoutes &operator=(const ScopedCudaMoEFusedVerifierPrefillRoutes &) = delete;

    private:
        bool old_gateup_kpart_decode_ = false;
        bool old_down_kpart_decode_ = false;
        bool old_prefill_fuse_swiglu_ = false;
        int old_prefill_tile_m_ = 0;
        bool old_grouped_prefill_ = false;
    };

    inline bool shouldUseMoEParityDeterministicMode(
        const MoEPrefixRestoreParityCase &test_case)
    {
        return test_case.required_cuda_devices > 0;
    }

    inline ExpertComputeDomain localTPMoEDomain(
        const std::string &name,
        CollectiveBackendType backend,
        std::vector<GlobalDeviceAddress> participants)
    {
        ExpertComputeDomain domain;
        domain.name = name;
        domain.kind = ExpertDomainKind::LocalTP;
        domain.backend = backend;
        domain.participants = std::move(participants);
        domain.owner_rank = 0;
        domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
        return domain;
    }

    inline ExpertRoutedTier routedTier(
        const std::string &name,
        const std::string &domain,
        int priority,
        int max_experts_per_layer,
        size_t memory_budget_bytes,
        bool fallback = false)
    {
        ExpertRoutedTier tier;
        tier.name = name;
        tier.domain = domain;
        tier.priority = priority;
        tier.max_experts_per_layer = max_experts_per_layer;
        tier.memory_budget_bytes = memory_budget_bytes;
        tier.fallback = fallback;
        return tier;
    }

    inline std::shared_ptr<MoEExpertParallelPlan> qwen36MoEOverlayPlanRocm2TPHotCpu2LocalTPCold()
    {
        constexpr const char *kRocmHotDomain = "qwen36_moe_rocm_hot";
        constexpr const char *kCpuColdDomain = "qwen36_moe_cpu_cold";

        auto plan = std::make_shared<MoEExpertParallelPlan>();
        plan->enabled = true;
        plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan->residency_policy = ExpertResidencyPolicy::StaticById;
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
        return plan;
    }

    inline std::shared_ptr<MoEExpertParallelPlan> qwen36MoEOverlayPlanRocm2TPHotOnly()
    {
        constexpr const char *kRocmHotDomain = "qwen36_moe_rocm_hot";

        auto plan = std::make_shared<MoEExpertParallelPlan>();
        plan->enabled = true;
        plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan->residency_policy = ExpertResidencyPolicy::StaticById;
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
        if (world_size != 1)
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

        if (test_case.topology == MoEPrefixParityTopology::ExpertOverlayRocm2TPHotOnly)
        {
            IBackend *rocm = getROCmBackend();
            if (!rocm)
            {
                return test_case.name + " requires ROCm backend memory queries";
            }

            // This fixture is the real Qwen3.6 35B MoE IQ3_S hot-only overlay.
            // On 32 GiB ROCm cards the base graph setup leaves only ~8.7 GiB
            // free for the all-256-expert hot tier, just below the runner's
            // resident-weight preflight requirement plus safety margin. Keep
            // this as a test prerequisite; the runtime path still hard-fails
            // if a user asks for an infeasible plan directly.
            constexpr size_t kMinimumRocmHotOnlyTotalBytes = 40ull * 1024ull * 1024ull * 1024ull;
            constexpr size_t kQwen36MoEHotOnlyResidentBytes = 7386ull * 1024ull * 1024ull;
            for (const auto &device : test_case.devices)
            {
                if (!device.isROCm())
                {
                    continue;
                }

                const size_t total = rocm->deviceMemoryTotal(device.device_ordinal);
                const size_t free = rocm->deviceMemoryFree(device.device_ordinal);
                if (total == 0 || free == 0)
                {
                    return test_case.name + " cannot query ROCm VRAM for " +
                           device.toShortString();
                }

                if (total < kMinimumRocmHotOnlyTotalBytes)
                {
                    std::ostringstream oss;
                    oss << test_case.name << " requires ROCm participants with at least "
                        << formatMiBForSkip(kMinimumRocmHotOnlyTotalBytes)
                        << " total VRAM for the no-fallback hot-only resident expert plan"
                        << " (" << device.toShortString()
                        << " total=" << formatMiBForSkip(total) << ")";
                    return oss.str();
                }

                const size_t safety_margin = std::max(mib(512), total / size_t{20});
                const size_t required = kQwen36MoEHotOnlyResidentBytes + safety_margin;
                if (required > free)
                {
                    std::ostringstream oss;
                    oss << test_case.name << " requires "
                        << formatMiBForSkip(required)
                        << " free on " << device.toShortString()
                        << " for the no-fallback hot-only resident expert plan"
                        << " (free=" << formatMiBForSkip(free)
                        << ", safety_margin=" << formatMiBForSkip(safety_margin)
                        << ")";
                    return oss.str();
                }
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
        config.moe_expert_parallel_plan = test_case.moe_expert_parallel_plan;

        switch (test_case.topology)
        {
        case MoEPrefixParityTopology::SingleDevice:
            config.tp_degree = 1;
            config.pp_degree = 1;
            config.device_for_this_rank = test_case.devices.empty()
                                              ? GlobalDeviceAddress::cpu()
                                              : test_case.devices.front();
            break;
        case MoEPrefixParityTopology::ExpertOverlayRocm2TPHotOnly:
        case MoEPrefixParityTopology::ExpertOverlayRocm2TPHotCpu2LocalTPCold:
            config.tp_degree = 1;
            config.pp_degree = 1;
            break;
        }

        return config;
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
        case MoEPrefixParityTopology::ExpertOverlayRocm2TPHotOnly:
            test_case.devices = {
                GlobalDeviceAddress::rocm(0),
                GlobalDeviceAddress::rocm(1),
            };
            test_case.required_rocm_devices = 2;
            test_case.moe_expert_parallel_plan =
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
            test_case.moe_expert_parallel_plan =
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

    inline void runMoEPrefixRestoreParity(
        const MoEPrefixRestoreParityCase &test_case,
        PrefixRestoreParityMode mode)
    {
        ScopedMoEParityDeterministicMode deterministic_mode(
            shouldUseMoEParityDeterministicMode(test_case));
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

        const int block_size = mode == PrefixRestoreParityMode::FullHit
                                   ? static_cast<int>(prompt_tokens.size())
                                   : 4;
        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto baseline = factory->createFromOrchestrationConfig(
            makeMoEPrefixRestoreConfig(test_case, model_path, false, block_size));
        ASSERT_NE(baseline, nullptr);
        phase_start = parityPhaseStart();
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        logMoEParityPhase(test_case, "prefix-baseline.initialize", phase_start);
        phase_start = parityPhaseStart();
        auto baseline_result = baseline->generate(prompt_tokens, test_case.decode_steps, greedy);
        logMoEParityPhase(test_case, "prefix-baseline.generate", phase_start);
        const auto baseline_snapshot = baseline->prefixStateProbe();
        baseline->shutdown();

        ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
        ASSERT_EQ(baseline_result.tokens.size(), expected_tokens.size());
        EXPECT_EQ(baseline_snapshot.prefix_cache_hits, 0u);

        // The dedicated Qwen3.6 MoE math parity suite checks PyTorch logits and
        // layer snapshots. Prefix restore correctness is stricter in a different
        // direction: cache-enabled runs must reproduce the no-cache Llaminar
        // greedy stream exactly, including quantized top-1 swaps tolerated by
        // the math harness.
        const auto reference_tokens = baseline_result.tokens;

        auto cached = factory->createFromOrchestrationConfig(
            makeMoEPrefixRestoreConfig(test_case, model_path, true, block_size));
        ASSERT_NE(cached, nullptr);
        phase_start = parityPhaseStart();
        ASSERT_TRUE(cached->initialize()) << cached->lastError();
        logMoEParityPhase(test_case, "prefix-cached.initialize", phase_start);

        std::vector<int32_t> first_prompt = prompt_tokens;
        if (mode == PrefixRestoreParityMode::PartialHit)
        {
            ASSERT_GT(prompt_tokens.size(), 4u);
            first_prompt.assign(prompt_tokens.begin(), prompt_tokens.begin() + 4);
        }

        phase_start = parityPhaseStart();
        auto first = cached->generate(first_prompt, test_case.decode_steps, greedy);
        logMoEParityPhase(test_case, "prefix-cached.first-generate", phase_start);
        const auto after_first = cached->prefixStateProbe();
        ASSERT_TRUE(first.error.empty()) << first.error;
        EXPECT_TRUE(after_first.prefix_cache_ready);
        EXPECT_GE(after_first.prefix_cache_inserts, 1u);
        if (mode == PrefixRestoreParityMode::FullHit)
        {
            ASSERT_EQ(first.tokens.size(), reference_tokens.size());
            EXPECT_EQ(first.tokens, reference_tokens);
        }

        phase_start = parityPhaseStart();
        auto second = cached->generate(prompt_tokens, test_case.decode_steps, greedy);
        logMoEParityPhase(test_case, "prefix-cached.second-generate", phase_start);
        const auto after_second = cached->prefixStateProbe();
        cached->shutdown();

        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_EQ(second.tokens.size(), reference_tokens.size());
        EXPECT_EQ(second.tokens, reference_tokens);
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
            EXPECT_EQ(after_second.prefix_request.matched_tokens, 4);
            EXPECT_FALSE(after_second.prefix_request.terminal_logits_restored);
        }
    }

    inline void runMoEMTPParity(
        const MoEPrefixRestoreParityCase &test_case,
        bool enable_prefix_cache,
        int mtp_draft_tokens = 1)
    {
        ScopedMoEParityDeterministicMode deterministic_mode(
            shouldUseMoEParityDeterministicMode(test_case));
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

        phase_start = parityPhaseStart();
        auto first = mtp->generate(prompt_tokens, test_case.decode_steps, greedy);
        logMoEParityPhase(test_case, "mtp.first-generate", phase_start);
        const auto after_first = mtp->prefixStateProbe();
        ASSERT_TRUE(first.error.empty()) << first.error;
        ASSERT_EQ(first.tokens.size(), reference_tokens.size());
        EXPECT_EQ(first.tokens, reference_tokens);
        EXPECT_FALSE(after_first.mtp_bypassed) << after_first.mtp_bypass_reason;
        EXPECT_GE(after_first.mtp_draft_steps, 1u);
        EXPECT_GE(after_first.mtp_verifier_runs, 1u);

        if (!enable_prefix_cache)
        {
            mtp->shutdown();
            return;
        }

        EXPECT_TRUE(after_first.prefix_cache_ready);
        EXPECT_GE(after_first.prefix_cache_inserts, 1u);
        EXPECT_GT(after_first.prefix_cache_mtp_state_bytes, 0u);

        phase_start = parityPhaseStart();
        auto second = mtp->generate(prompt_tokens, test_case.decode_steps, greedy);
        logMoEParityPhase(test_case, "mtp.second-generate", phase_start);
        const auto after_second = mtp->prefixStateProbe();
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
    }

    inline void runMoEStochasticMTPVerifierParity(
        const MoEPrefixRestoreParityCase &test_case)
    {
        ScopedMoEParityDeterministicMode deterministic_mode(
            shouldUseMoEParityDeterministicMode(test_case));
        ASSERT_EQ(test_case.topology, MoEPrefixParityTopology::SingleDevice)
            << "MoE stochastic MTP verifier parity is currently single-device only";

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

        constexpr int block_size = 2;
        const int stochastic_decode_steps = std::max(2, test_case.decode_steps);
        auto factory = createOrchestrationRunnerFactory();

        SamplingParams stochastic;
        stochastic.temperature = 0.6f;
        stochastic.top_k = 20;
        stochastic.top_p = 0.95f;
        stochastic.presence_penalty = 0.25f;
        stochastic.seed = 123;

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
            makeMoEPrefixRestoreConfig(test_case, model_path, false, block_size, true, 1);
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
        ASSERT_TRUE(mtp_result.error.empty()) << mtp_result.error;
        ASSERT_EQ(mtp_result.tokens.size(), static_cast<size_t>(stochastic_decode_steps));

        mtp->clearCache();
        PerfStatsCollector::reset();
        phase_start = parityPhaseStart();
        auto reused_mtp_result =
            mtp->generate(prompt_tokens, stochastic_decode_steps, stochastic);
        logMoEParityPhase(test_case, "stochastic-mtp.reused-generate", phase_start);
        const auto after_reused_mtp = mtp->prefixStateProbe();
        const auto phase138_records = PerfStatsCollector::snapshot({"mtp"});
        mtp->shutdown();

        ASSERT_TRUE(reused_mtp_result.error.empty()) << reused_mtp_result.error;
        ASSERT_EQ(reused_mtp_result.tokens.size(), mtp_result.tokens.size());
        EXPECT_EQ(reused_mtp_result.tokens, mtp_result.tokens)
            << "MoE stochastic MTP with the same seed must be reproducible after clearCache()";
        EXPECT_FALSE(after_reused_mtp.mtp_bypassed)
            << after_reused_mtp.mtp_bypass_reason;
        EXPECT_EQ(after_reused_mtp.mtp_request.verify_mode, "speculative-sampling");
        EXPECT_TRUE(after_reused_mtp.mtp_request.stochastic_verify);
        EXPECT_EQ(after_reused_mtp.mtp_transaction_validation_failures, 0u)
            << test_case.name
            << " MoE stochastic MTP hit MTP transaction validation failures";
        EXPECT_GE(after_reused_mtp.mtp_draft_steps, 1u);
        EXPECT_GE(after_reused_mtp.mtp_verifier_runs, 1u);
        EXPECT_GE(after_reused_mtp.mtp_verifier_token_count, 2u);
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

        const bool used_decode_equivalent_stochastic_verifier =
            std::any_of(
                phase138_records.begin(),
                phase138_records.end(),
                [](const PerfStatRecord &record)
                {
                    return record.kind == PerfStatRecord::Kind::Counter &&
                           record.domain == "mtp" &&
                           record.name == "decode_equivalent_stochastic_verifier_runs";
                });
        EXPECT_TRUE(used_decode_equivalent_stochastic_verifier)
            << "Stateful Qwen3.6 MoE stochastic MTP parity must exercise the "
               "decode-equivalent verifier path until vLLM-style MoE state "
               "publication is accepted\n"
            << PerfStatsCollector::summaryString({"mtp"});

        const bool used_retired_phase138_stochastic_candidate =
            std::any_of(
                phase138_records.begin(),
                phase138_records.end(),
                [](const PerfStatRecord &record)
                {
                    return record.kind == PerfStatRecord::Kind::Counter &&
                           record.domain == "mtp" &&
                           record.name == "phase138_stochastic_spec_decode_runs";
                });
        EXPECT_FALSE(used_retired_phase138_stochastic_candidate)
            << "Stateful Qwen3.6 MoE stochastic MTP must not use the retired "
               "accepted-count publication candidate\n"
            << PerfStatsCollector::summaryString({"mtp"});
    }

    inline void runMoEGreedyFreshRunnerDeterminism(
        const MoEPrefixRestoreParityCase &test_case)
    {
        ScopedMoEParityDeterministicMode deterministic_mode(
            shouldUseMoEParityDeterministicMode(test_case));
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
        double max_abs_diff = 0.0;
        double left_l2 = 0.0;
        double right_l2 = 0.0;
        double left_mean = 0.0;
        double right_mean = 0.0;
        std::string left_label = "baseline";
        std::string right_label = "mtp";
        bool present_in_baseline = false;
        bool present_in_mtp = false;
    };

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
               "cosine,rel_l2,max_abs_diff,left_l2,right_l2,left_mean,right_mean,left_label,right_label,"
               "present_left,present_right\n";
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
            << row.max_abs_diff << ','
            << row.left_l2 << ','
            << row.right_l2 << ','
            << row.left_mean << ','
            << row.right_mean << ','
            << csvEscapeMoEDiagnostic(row.left_label) << ','
            << csvEscapeMoEDiagnostic(row.right_label) << ','
            << (row.present_in_baseline ? "true" : "false") << ','
            << (row.present_in_mtp ? "true" : "false") << '\n';
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
            << " max_abs_diff=" << row.max_abs_diff
            << " present_left=" << (row.present_in_baseline ? "true" : "false")
            << " present_right=" << (row.present_in_mtp ? "true" : "false");
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

    inline void runMoEMTPSidecarStageBreakdownDiagnostic(
        const MoEPrefixRestoreParityCase &test_case,
        int decode_token_budget)
    {
        ScopedMoEParityDeterministicMode deterministic_mode(
            shouldUseMoEParityDeterministicMode(test_case));
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
        bool allow_reference_prefix_only = false)
    {
        ScopedMoEParityDeterministicMode deterministic_mode(
            shouldUseMoEParityDeterministicMode(test_case));
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
                    if (!runner->maybeApplyMoERebalance())
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
                    if (!runner->maybeApplyMoERebalance())
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

    inline void expectCudaMoEMTPVerifierFusedPrefillPath()
    {
        const auto records = PerfStatsCollector::snapshot(
            {"kernel.cuda_moe_grouped_prefill_swiglu_path_calls"});
        auto tag_equals = [](const PerfStatRecord &record,
                             const char *key,
                             const char *value) -> bool
        {
            const auto it = record.tags.find(key);
            return it != record.tags.end() && it->second == value;
        };

        const auto match = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name == "cuda_moe_grouped_prefill_swiglu_path_calls" &&
                       tag_equals(record, "swiglu_path", "fused") &&
                       tag_equals(record, "tile_m", "2") &&
                       tag_equals(record, "tile_n", "64") &&
                       tag_equals(record, "active_expert_slots", "16") &&
                       tag_equals(record, "gateup_route", "kpart_swiglu") &&
                       tag_equals(record, "down_route", "kpart_prefill") &&
                       tag_equals(record, "down_accumulation", "token_direct");
            });

        ASSERT_NE(match, records.end())
            << "CUDA Qwen3.6 MoE MTP verifier path did not exercise the fused "
            << "split-K grouped prefill SwiGLU/down kernels with the verifier-sized tile. "
            << "This is a production-path regression: keep the fused path "
            << "correct instead of routing around it.\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.cuda_moe_grouped_prefill_swiglu_path_calls"});
        EXPECT_GT(match->count, 0u);
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
                if (!runner->maybeApplyMoERebalance())
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

    inline void expectCudaMoEMTPVerifierSharedExpertFusedPrefillPath()
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

        const auto shared_prefill = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name == "cuda_moe_grouped_prefill_swiglu_path_calls" &&
                       tag_equals(record, "swiglu_path", "fused") &&
                       tag_equals(record, "tile_m", "2") &&
                       tag_equals(record, "tile_n", "64") &&
                       tag_equals(record, "total_slots", "2") &&
                       tag_equals(record, "active_expert_slots", "1") &&
                       tag_equals(record, "num_experts", "1") &&
                       tag_equals(record, "gateup_route", "kpart_swiglu") &&
                       tag_equals(record, "down_route", "kpart_prefill") &&
                       tag_equals(record, "down_accumulation", "token_direct");
            });

        ASSERT_NE(shared_prefill, records.end())
            << "CUDA Qwen3.6 MoE MTP verifier shared expert did not exercise "
            << "the grouped split-K prefill route. Keep the fused shared-expert "
            << "path correct instead of silently falling back to dense GEMMs.\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.cuda_moe_grouped_prefill_swiglu_path_calls",
                    "kernel.cuda_moe_shared_expert_prefill_group_calls"});
        EXPECT_GT(shared_prefill->count, 0u);

        const auto shared_group = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name == "cuda_moe_shared_expert_prefill_group_calls" &&
                       tag_equals(record, "seq_len", "2") &&
                       tag_equals(record, "active_expert_slots", "1") &&
                       tag_equals(record, "top_k", "1");
            });

        ASSERT_NE(shared_group, records.end())
            << "CUDA shared expert grouped verifier setup did not run.\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.cuda_moe_shared_expert_prefill_group_calls"});
        EXPECT_GT(shared_group->count, 0u);
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
                       tag_equals(record, "gateup_route", "kpart_swiglu") &&
                       tag_equals(record, "down_route", "kpart_prefill") &&
                       tag_equals(record, "down_accumulation", "token_direct");
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
                       tag_equals(record, "gateup_route", "kpart_swiglu") &&
                       tag_equals(record, "down_route", "kpart_prefill") &&
                       tag_equals(record, "down_accumulation", "token_direct");
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
        ScopedMoEParityDeterministicMode deterministic_mode(
            shouldUseMoEParityDeterministicMode(test_case));
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
                    if (!runner->maybeApplyMoERebalance())
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
        ScopedMoEParityDeterministicMode deterministic_mode(
            shouldUseMoEParityDeterministicMode(test_case));
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
            ASSERT_TRUE(runner->maybeApplyMoERebalance()) << runner->lastError();
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
        ScopedMoEParityDeterministicMode deterministic_mode(
            shouldUseMoEParityDeterministicMode(test_case));
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
