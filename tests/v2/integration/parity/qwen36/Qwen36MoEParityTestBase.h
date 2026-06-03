#pragma once

#include "Qwen36DenseParityTestBase.h"
#include "backends/BackendManager.h"
#include "backends/HardwareInventory.h"
#include "execution/moe/MoEExpertParallelPlan.h"
#include "kernels/KernelFactory.h"
#include "utils/DebugEnv.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>

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

    inline void ensurePyTorchMoEMetadata(
        const MoEPrefixRestoreParityCase &test_case,
        const std::string &model_path,
        const std::filesystem::path &metadata_path)
    {
        if (metadataLooksUsable(metadata_path, test_case.decode_steps))
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

        ASSERT_TRUE(metadataLooksUsable(metadata_path, test_case.decode_steps))
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
        bool enable_mtp = false)
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
        config.mtp.draft_tokens = 1;
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
        bool enable_prefix_cache)
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
                true));
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
        EXPECT_GT(after_second.mtp_draft_steps, after_first.mtp_draft_steps);
        EXPECT_GT(after_second.mtp_verifier_runs, after_first.mtp_verifier_runs);
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
