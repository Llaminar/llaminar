/**
 * @file Test__Qwen36MoE_ExpertOverlay_PrefixMTP_Parity.cpp
 * @brief Prefix-cache and MTP parity coverage for Qwen3.6 MoE expert-overlay TP fixtures.
 *
 * The tests in this file exercise request-boundary state restoration across
 * CUDA and ROCm two-device expert-overlay plans.  They intentionally pair the
 * prefix cache with MTP verifier state, KV cache state, and MoE rebalance state
 * because those lifetimes are independent in production but must agree at a
 * request boundary.  The phase-split migration probes use the same long-decode
 * PyTorch metadata as the math parity suite, so their context window must be
 * large enough for the full metadata prompt even when an individual partial-hit
 * prefix-restore test later trims that prompt.
 */

#include "Qwen36MoEParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include "models/qwen35/Qwen35GraphConfigBuilder.h"
#include "utils/Logger.h"

#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string_view>
#include <stdexcept>
#include <sys/file.h>
#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    /**
     * @brief Context length used by phase-split probes backed by long-decode metadata.
     *
     * The long-decode PyTorch fixtures currently contain prompts in the low
     * thousands of tokens.  A 4096-token window matches the expert-overlay math
     * parity suite and keeps MTP prefix-restore tests honest: they run the full
     * metadata prompt instead of relying on the partial-hit prefix cap used by
     * the non-MTP restore tests.
     */
    constexpr int kPhaseSplitLongDecodeMaxSeqLen = 4096;

    /**
     * @brief Tokenize the full-tier server's structured-generation prompt exactly.
     *
     * The long-context server gate does not send a raw string to the model. It
     * sends a system/user conversation through the Qwen3.5 community template
     * override with thinking disabled. Reusing only the resulting row count in
     * an integration test misses the expert-routing pattern that drives Dynamic
     * ownership decisions. This helper therefore mirrors that production
     * request construction and returns the exact token IDs consumed by prefill.
     *
     * The expected 154-token geometry is asserted here as part of the
     * regression contract. If the tokenizer or bundled chat template changes,
     * the test must deliberately update both the server expectation and this
     * reproducer instead of silently exercising a different graph bucket.
     *
     * @param model_path Qwen3.6 GGUF used by the parity fixture.
     * @return Production chat-template token IDs for the structured request.
     * @throws std::runtime_error if tokenizer or template construction fails,
     *         or if the resulting request no longer contains 154 tokens.
     */
    std::vector<int32_t> structuredGenerationPromptTokens(
        const std::string &model_path)
    {
        const ModelContextConfig tokenizer_context_config{
            .strategy = WeightDistributionStrategy::REPLICATED,
            .use_mmap = true,
            .target_is_gpu = true,
        };
        auto tokenizer_context =
            ModelContext::create(model_path, tokenizer_context_config);
        if (!tokenizer_context)
        {
            throw std::runtime_error(
                "failed to parse Qwen3.6 GGUF metadata for structured prompt tokenization");
        }

        auto tokenizer = createTokenizer(tokenizer_context);
        if (!tokenizer)
        {
            throw std::runtime_error(
                "failed to construct Qwen3.6 tokenizer for structured prompt regression");
        }

        Qwen35GraphConfigBuilder config_builder;
        const auto template_override = config_builder.chatTemplateOverride();
        if (!template_override.has_value() || template_override->empty())
        {
            throw std::runtime_error(
                "Qwen3.5/3.6 structured prompt regression requires the production chat-template override");
        }
        tokenizer->setChatTemplate(
            ChatTemplate::create(*template_override, "", ""));

        constexpr int kRequestedNumberedLines = 220;
        std::ostringstream user_prompt;
        user_prompt
            << "Control marker for cache isolation: LCACHE-RESET-SENTINEL-593821.\n"
            << "Do not copy the control marker into the report.\n"
            << "Write " << kRequestedNumberedLines << " numbered lines.\n"
            << "Each line must use this exact format:\n"
            << "001 | one short distinct sentence about reliable inference\n"
            << "002 | one short distinct sentence about reliable inference\n"
            << "Keep each sentence concise and vary the wording.\n"
            << "Never restart numbering; count upward from 001.\n"
            << "After the final requested line, write END_OF_REPORT on its own line.\n"
            << "If the token limit interrupts the report, stop wherever the limit occurs.";

        const std::vector<ChatMessage> messages = {
            ChatMessage(
                "system",
                "<|think_off|>\nYou produce deterministic machine-checkable reports."),
            ChatMessage("user", user_prompt.str()),
        };
        const std::vector<int> encoded = tokenizer->encodeChat(
            messages,
            /*add_generation_prompt=*/true,
            /*tools_json=*/"",
            /*enable_thinking=*/false);
        if (encoded.size() != 154u)
        {
            throw std::runtime_error(
                "structured-generation prompt token count changed from 154 to " +
                std::to_string(encoded.size()));
        }

        return std::vector<int32_t>(encoded.begin(), encoded.end());
    }

    /**
     * @brief Serializes this process with same-backend expert-overlay parity cases.
     *
     * The fixture materializes multi-GPU runners and large pinned logits buffers.
     * CUDA and ROCm use disjoint devices and collective libraries, so each owns
     * a separate lock domain. Running the unfiltered suite retains a global lock
     * because it traverses both domains in one process.
     */
    class ScopedParityProcessLock
    {
    public:
        explicit ScopedParityProcessLock(const char *path)
        {
            fd_ = ::open(path, O_CREAT | O_RDWR, 0666);
            if (fd_ < 0)
            {
                throw std::runtime_error(
                    std::string("failed to open parity process lock ") +
                    path + ": " + std::strerror(errno));
            }
            if (::flock(fd_, LOCK_EX) != 0)
            {
                const std::string error =
                    std::string("failed to acquire parity process lock ") +
                    path + ": " + std::strerror(errno);
                ::close(fd_);
                fd_ = -1;
                throw std::runtime_error(error);
            }
        }

        ~ScopedParityProcessLock()
        {
            if (fd_ >= 0)
            {
                ::flock(fd_, LOCK_UN);
                ::close(fd_);
            }
        }

        ScopedParityProcessLock(const ScopedParityProcessLock &) = delete;
        ScopedParityProcessLock &operator=(const ScopedParityProcessLock &) = delete;

    private:
        int fd_ = -1;
    };

    /**
     * @brief Select the narrowest safe interprocess lock for this test command.
     *
     * Individual CUDA and ROCm filters may execute concurrently because their
     * GPU memory and collectives do not overlap. A filter mentioning both
     * backends, an unfiltered invocation, or an unfamiliar command retains the
     * conservative suite-wide lock.
     *
     * @param argc Number of command-line arguments.
     * @param argv Command-line arguments passed to GoogleTest.
     * @return Stable lock-file path for the selected backend domain.
     */
    const char *parityProcessLockPath(
        int argc,
        char *const argv[])
    {
        bool mentions_cuda = false;
        bool mentions_rocm = false;
        for (int index = 1; index < argc; ++index)
        {
            const std::string_view argument =
                argv[index] ? std::string_view(argv[index]) : std::string_view{};
            mentions_cuda =
                mentions_cuda ||
                argument.find("CUDA") != std::string_view::npos;
            mentions_rocm =
                mentions_rocm ||
                argument.find("ROCm") != std::string_view::npos;
        }

        if (mentions_cuda && !mentions_rocm)
            return "/tmp/llaminar_qwen36_moe_expert_overlay_cuda.lock";
        if (mentions_rocm && !mentions_cuda)
            return "/tmp/llaminar_qwen36_moe_expert_overlay_rocm.lock";
        return "/tmp/llaminar_qwen36_moe_expert_overlay_parity.lock";
    }

    /**
     * @brief Builds a rebalance policy that deliberately encourages expert movement.
     *
     * The values remove the usual production hysteresis so short parity prompts
     * can exercise phase-split expert placement changes.  The resulting config
     * is test-only and is paired with strict parity assertions rather than a
     * degraded fallback path.
     *
     * @param mode Dynamic or LLEP rebalance mode to exercise.
     * @return Runtime rebalance configuration for the migration probes.
     */
    MoERebalanceRuntimeConfig movementFriendlyRebalanceConfig(
        MoERebalanceRuntimeMode mode)
    {
        MoERebalanceRuntimeConfig config;
        config.mode = mode;
        config.window_size = 4;
        config.max_window_size = 4;
        config.window_growth_factor = 1.0f;
        config.dynamic_imbalance_threshold_per_mille = 0;
        config.dynamic_min_improvement_per_mille = 0;
        config.dynamic_max_swaps_per_layer = 20;
        config.dynamic_max_plan_entries_per_wave = 20;
        config.dynamic_min_window_activations = 0;
        config.device_min_load_spread_improvement = 0;
        config.device_min_load_spread_improvement_divisor = 0;
        config.device_min_wave_spread_improvement_per_payload_slot = 0;
        config.device_min_foreign_rows_per_transfer = 0;
        config.device_min_router_spread_improvement_per_payload_slot = 0;
        config.device_max_post_wave_load_spread_per_mille = 1000;
        config.device_maintenance_slack_tokens = 0;
        config.device_min_maintenance_period_tokens = 4;
        config.device_initial_maintenance_period_tokens = 4;
        if (mode == MoERebalanceRuntimeMode::LLEP)
        {
            /*
             * These cells are coverage tests, not production economics tests:
             * make the LLEP assignment capacity strict enough that both CUDA
             * and ROCm must publish transfer-backed migrations instead of
             * legally selecting static-owner assignment for already-balanced router loads.
             */
            config.device_llep_alpha_numerator = 1;
            config.device_llep_alpha_denominator = 2;
            config.device_llep_enable_balanced_skip = false;
        }
        config.release_raw_expert_weights = true;
        return config;
    }

    /**
     * @brief Promotes a short hot-only fixture to the canonical long MTP probe.
     *
     * The original hot-only canary intentionally uses a 96-token context and two
     * output rows because its third token is a quantized near-tie. That makes it
     * useful as a fast smoke test but leaves long absolute positions, repeated
     * grouped publication, and eight-token continuation untested unless runtime
     * expert movement is also enabled. This helper supplies the long geometry
     * independently of rebalance policy so static, Dynamic, and LLEP cells differ
     * by exactly one dimension.
     *
     * @param test_case Case object to update in place.
     */
    void configureLongContextMTPProbe(
        MoEPrefixRestoreParityCase &test_case)
    {
        test_case.name += " long-context";
        test_case.prompt = qwen36MoELongNeedleParityPrompt();
        test_case.reference_input_source =
            MoEReferenceInputSource::ModelTokenizer;
        test_case.metadata_envs.clear();
        test_case.default_metadata_path.clear();
        test_case.decode_steps = 8;
        test_case.max_seq_len = kPhaseSplitLongDecodeMaxSeqLen;
        test_case.prefix_restore_prompt_token_limit = 640;
        test_case.minimum_prompt_tokens = 640;
        test_case.env_overrides = {
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        };

        /*
         * Runtime rebalance defaults to Dynamic in production configuration.
         * A missing optional test override is therefore not static placement.
         * Declare Off explicitly so this fixture is a genuine long-position
         * control; configurePhaseSplitMigrationProbe() replaces this policy
         * with its movement-friendly Dynamic or LLEP configuration.
         */
        MoERebalanceRuntimeConfig static_placement;
        static_placement.mode = MoERebalanceRuntimeMode::Off;
        test_case.moe_rebalance = static_placement;
    }

    /**
     * @brief Converts a hot-only expert-overlay case into a phase-split migration probe.
     *
     * Prefix partial-hit tests cap the tokenized prompt locally, while MTP
     * restore tests use the complete deterministic ledger. Both obtain tokens
     * from Llaminar's production GGUF tokenizer because their correctness
     * oracle is a fresh serial Llaminar request, not an unused PyTorch decode.
     * The case must therefore declare a long-context window up front instead
     * of depending on the partial-prefix cap.
     *
     * @param test_case Case object to mutate.
     * @param mode Rebalance mode under test.
     */
    void configurePhaseSplitMigrationProbe(
        MoEPrefixRestoreParityCase &test_case,
        MoERebalanceRuntimeMode mode)
    {
        configureLongContextMTPProbe(test_case);
        test_case.name += mode == MoERebalanceRuntimeMode::LLEP
                              ? " LLEP phase-split migration"
                              : " Dynamic phase-split migration";
        /*
         * The shared long-context fixture emits eight requested tokens. At
         * grouped depth two that budget is sufficient to publish the first
         * token-four planning probe, but request reset may legitimately drain
         * it before a later transaction can publish and apply its payload.
         * Dedicated lifecycle tests below therefore use sixteen requested
         * tokens and explicitly require applied arrivals; these ordinary
         * parity cells remain the affordable single-request correctness gate.
         */
        test_case.moe_rebalance = movementFriendlyRebalanceConfig(mode);
        test_case.env_overrides.emplace_back(
            "LLAMINAR_MOE_DEVICE_REBALANCE_NO_WORK_BACKOFF_PERIODS",
            "0");
        test_case.env_overrides.emplace_back(
            "LLAMINAR_MOE_GPU_DIRECT_TRANSFER_WAVE_EXPERTS",
            "32");
        test_case.env_overrides.emplace_back(
            "LLAMINAR_MOE_DEVICE_REBALANCE_COMPACT_PAYLOAD_SLOTS",
            "32");
        if (mode == MoERebalanceRuntimeMode::LLEP)
        {
            test_case.env_overrides.emplace_back(
                "LLAMINAR_MOE_LLEP_PREFILL_TRANSFER_MODE",
                "full");
            test_case.env_overrides.emplace_back(
                "LLAMINAR_MOE_LLEP_PREFILL_MIN_ROUTED_ROWS",
                "0");
        }
    }

    /**
     * @brief Converts a short hot-only fixture into a real partial-prefix probe.
     *
     * The default hot-only metadata prompt is intentionally tiny so the MTP
     * canary tests stay affordable.  Partial-prefix restore, however, needs at
     * least one complete cached block plus a suffix.  This helper points the
     * hot-only partial-hit cells at the shared deterministic ledger and then
     * caps its production-tokenizer output at 640 rows, giving the cache a
     * non-terminal 256-token block and a meaningful suffix without turning the
     * hot-only canary itself into a full long-context decode test.
     *
     * @param test_case Case object to mutate.
     */
    void configureHotOnlyPartialPrefixProbe(
        MoEPrefixRestoreParityCase &test_case)
    {
        test_case.name += " long partial-prefix restore";
        test_case.prompt = qwen36MoELongNeedleParityPrompt();
        test_case.reference_input_source =
            MoEReferenceInputSource::ModelTokenizer;
        test_case.metadata_envs.clear();
        test_case.default_metadata_path.clear();
        test_case.max_seq_len = kPhaseSplitLongDecodeMaxSeqLen;
        test_case.prefix_restore_prompt_token_limit = 640;
        test_case.minimum_prompt_tokens = 640;
    }

    /**
     * @brief Converts a hot-only fixture into a stochastic MTP depth probe.
     *
     * The stochastic matrix needs enough decode steps to exercise fixed depths
     * 1/2/3 and the dynamic depth controller, but it should not inherit the
     * long ledger metadata used by rebalance migration cells.  This helper keeps
     * the GPU TP stochastic rows affordable while still using a prompt with
     * enough entropy to drive accept/reject sampling behavior.
     *
     * @param test_case Case object to mutate.
     * @param metadata_path Backend-specific PyTorch metadata for the benchmark prompt.
     */
    void configureHotOnlyStochasticBenchmarkProbe(
        MoEPrefixRestoreParityCase &test_case,
        const std::string &metadata_path)
    {
        test_case.name += " stochastic benchmark-prompt MTP";
        test_case.prompt = qwen36MoEBenchmarkPrompt();
        test_case.metadata_envs = {};
        test_case.default_metadata_path = metadata_path;
        test_case.decode_steps = 4;
        test_case.max_seq_len = 768;
    }

    /**
     * @brief Builds the mixed ROCm-hot/CPU-cold expert-overlay fixture.
     *
     * @return Qwen3.6 MoE prefix parity case with ROCm hot experts and CPU cold experts.
     */
    MoEPrefixRestoreParityCase expertOverlayCase()
    {
        return qwen36MoEPrefixParityCase(
            "Qwen3.6 MoE ExpertOverlay ROCm2TP hot + CPU2LocalTP cold parity",
            MoEPrefixParityTopology::ExpertOverlayRocm2TPHotCpu2LocalTPCold);
    }

    /**
     * @brief Builds the ROCm two-device hot-only expert-overlay fixture.
     *
     * @return Qwen3.6 MoE prefix parity case using ROCm local TP for hot experts.
     */
    MoEPrefixRestoreParityCase rocmOnlyExpertOverlayCase()
    {
        return qwen36MoEPrefixParityCase(
            "Qwen3.6 MoE ExpertOverlay ROCm2TP hot-only parity",
            MoEPrefixParityTopology::ExpertOverlayRocm2TPHotOnly);
    }

    /**
     * @brief Builds the ROCm hot-only stochastic benchmark fixture.
     *
     * @return ROCm two-device expert-overlay case with benchmark prompt metadata.
     */
    MoEPrefixRestoreParityCase rocmOnlyStochasticBenchmarkCase()
    {
        auto test_case = rocmOnlyExpertOverlayCase();
        configureHotOnlyStochasticBenchmarkProbe(
            test_case,
            "pytorch_qwen36_moe_expert_overlay_rocm2_mtp_diagnostic_snapshots/metadata.txt");
        return test_case;
    }

    /**
     * @brief Builds the CUDA two-device hot-only expert-overlay fixture.
     *
     * @return Qwen3.6 MoE prefix parity case using CUDA local TP for hot experts.
     */
    MoEPrefixRestoreParityCase cudaOnlyExpertOverlayCase()
    {
        auto test_case = qwen36MoEPrefixParityCase(
            "Qwen3.6 MoE ExpertOverlay CUDA2TP hot-only parity",
            MoEPrefixParityTopology::ExpertOverlayCuda2TPHotOnly);
        /*
         * Match the CUDA SingleDevice MoE guard: the default fixture's third
         * greedy token is a documented quantized near-tie (760 vs 71093).
         * Exact prefix/MTP restore equality is asserted on the stable prefix.
         */
        test_case.decode_steps = 2;
        return test_case;
    }

    /**
     * @brief Builds the CUDA hot-only stochastic benchmark fixture.
     *
     * @return CUDA two-device expert-overlay case with benchmark prompt metadata.
     */
    MoEPrefixRestoreParityCase cudaOnlyStochasticBenchmarkCase()
    {
        auto test_case = cudaOnlyExpertOverlayCase();
        configureHotOnlyStochasticBenchmarkProbe(
            test_case,
            "pytorch_qwen36_moe_expert_overlay_cuda2_mtp_diagnostic_snapshots/metadata.txt");
        return test_case;
    }

    /**
     * @brief Builds the CUDA hot-only partial-prefix fixture.
     *
     * @return CUDA hot-only expert-overlay case with a long enough prompt to
     * exercise partial prefix-cache restore.
     */
    MoEPrefixRestoreParityCase cudaOnlyPartialPrefixCase()
    {
        auto test_case = cudaOnlyExpertOverlayCase();
        configureHotOnlyPartialPrefixProbe(test_case);
        return test_case;
    }

    /**
     * @brief Builds the CUDA static-placement long-context MTP control.
     *
     * @return CUDA hot-only case with long positions and no runtime movement.
     */
    MoEPrefixRestoreParityCase cudaOnlyLongContextCase()
    {
        auto test_case = cudaOnlyExpertOverlayCase();
        configureLongContextMTPProbe(test_case);
        return test_case;
    }

    /**
     * @brief Builds the CUDA dynamic phase-split migration fixture.
     *
     * @return CUDA hot-only expert-overlay case with movement-friendly dynamic rebalance.
     */
    MoEPrefixRestoreParityCase cudaOnlyDynamicPhaseSplitCase()
    {
        auto test_case = cudaOnlyExpertOverlayCase();
        configurePhaseSplitMigrationProbe(
            test_case,
            MoERebalanceRuntimeMode::Dynamic);
        return test_case;
    }

    /**
     * @brief Builds the CUDA LLEP phase-split migration fixture.
     *
     * @return CUDA hot-only expert-overlay case with movement-friendly LLEP rebalance.
     */
    MoEPrefixRestoreParityCase cudaOnlyLLEPPhaseSplitCase()
    {
        auto test_case = cudaOnlyExpertOverlayCase();
        configurePhaseSplitMigrationProbe(
            test_case,
            MoERebalanceRuntimeMode::LLEP);
        return test_case;
    }

    /**
     * @brief Builds the ROCm dynamic phase-split migration fixture.
     *
     * @return ROCm hot-only expert-overlay case with movement-friendly dynamic rebalance.
     */
    MoEPrefixRestoreParityCase rocmOnlyDynamicPhaseSplitCase()
    {
        auto test_case = rocmOnlyExpertOverlayCase();
        configurePhaseSplitMigrationProbe(
            test_case,
            MoERebalanceRuntimeMode::Dynamic);
        return test_case;
    }

    /**
     * @brief Builds the ROCm hot-only partial-prefix fixture.
     *
     * @return ROCm hot-only expert-overlay case with a long enough prompt to
     * exercise partial prefix-cache restore.
     */
    MoEPrefixRestoreParityCase rocmOnlyPartialPrefixCase()
    {
        auto test_case = rocmOnlyExpertOverlayCase();
        configureHotOnlyPartialPrefixProbe(test_case);
        return test_case;
    }

    /**
     * @brief Builds the ROCm static-placement long-context MTP control.
     *
     * @return ROCm hot-only case with long positions and no runtime movement.
     */
    MoEPrefixRestoreParityCase rocmOnlyLongContextCase()
    {
        auto test_case = rocmOnlyExpertOverlayCase();
        configureLongContextMTPProbe(test_case);
        return test_case;
    }

    /**
     * @brief Builds the ROCm LLEP phase-split migration fixture.
     *
     * @return ROCm hot-only expert-overlay case with movement-friendly LLEP rebalance.
     */
    MoEPrefixRestoreParityCase rocmOnlyLLEPPhaseSplitCase()
    {
        auto test_case = rocmOnlyExpertOverlayCase();
        configurePhaseSplitMigrationProbe(
            test_case,
            MoERebalanceRuntimeMode::LLEP);
        return test_case;
    }

    /**
     * @brief Stress GPU rebalance request-boundary ordering without reloading model weights.
     *
     * @param test_case Backend-specific CUDA or ROCm Dynamic/LLEP phase-split fixture.
     *
     * The server regression that motivated this test appeared only after several
     * different prefills had alternated with grouped MTP decode. A single parity
     * request followed by one exact cache hit was therefore too narrow: it reused
     * one graph shape and gave the asynchronous token-four maintenance wave only
     * one subsequent consumer.
     *
     * This fixture keeps one production runner alive across the exact prefill
     * row-count history observed before the failing structured request:
     * 39, 68, 39, 62, 62, 431, 36, 1815, 1813, 1801, and 1305. Each history
     * request performs the canonical 64-token decode workload, which gives
     * asynchronous Dynamic/LLEP maintenance several opportunities to plan,
     * transfer, publish, and consume a new placement before request reset.
     *
     * The next request uses the exact 154-token structured-generation prompt
     * and 512-token completion budget from the server gate. The former
     * 96-token regression stopped before the original mirrored-terminal-hidden
     * divergence and therefore provided false confidence despite exercising
     * many grouped transactions. One final exact repeat makes prefix restore
     * mandatory after the long requests have already exceeded the canonical
     * 1 GiB RAM tier. No deterministic kernel mode, row replay, or host-side
     * rebalance fallback is enabled.
     *
     * Keeping the implementation backend-neutral is important. CUDA originally
     * exposed the lifecycle race, but ROCm owns the same asynchronous graph,
     * prefix-cache, and LLEP publication contract. Both backends therefore run
     * this exact request sequence and validate the same production counters.
     */
    void runRebalancedPrefillRequestBoundaryStress(
        MoEPrefixRestoreParityCase test_case)
    {
        ASSERT_TRUE(test_case.moe_rebalance.has_value())
            << "Rebalanced lifecycle stress requires an explicit runtime mode";

        /*
         * Keep the movement-friendly four-token histogram window and the early
         * token-four launch, but do not transport a fixed-capacity expert
         * payload every four decode tokens for the entire 1,232-token request
         * matrix. Each 64-token history request still launches one real
         * maintenance transaction, the 512-token structured request launches
         * eight times, and the final restore launches once. This preserves repeated
         * cross-request capture, transfer, apply, eviction, and restore coverage
         * while avoiding roughly two hundred redundant full-payload collectives.
         */
        constexpr int kStressMaintenancePeriodTokens = 64;
        test_case.moe_rebalance->device_min_maintenance_period_tokens =
            kStressMaintenancePeriodTokens;
        test_case.moe_rebalance->device_initial_maintenance_period_tokens = 4;

        /*
         * Match the canonical server cell before installing the scoped
         * environment. Prefill graph buckets are deliberately disabled for
         * this collective topology, while ordinary decode and grouped MTP
         * graphs remain enabled. The two-expert hot cache is also part of the
         * failing production configuration and materially affects ownership
         * pressure, so it belongs in the regression contract.
         */
        test_case.env_overrides.emplace_back(
            "LLAMINAR_PREFILL_GRAPH_BUCKETS",
            "0");
        test_case.env_overrides.emplace_back(
            "LLAMINAR_MOE_GPU_CACHE_EXPERTS_PER_LAYER",
            "2");
        ScopedMoEParityProductionMode production_mode(
            shouldForceMoEParityProductionMode(test_case));
        ScopedMoEPrefixCaseEnvironment case_env(test_case.env_overrides);
        ScopedEnvironmentValues perf_stats_enabled({
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
            /*
             * The canonical server gate records per-stage GPU events for every
             * request. Keep that instrumentation active here because
             * its event lifecycle overlaps the same graph-captured LLEP
             * maintenance streams that this regression is designed to stress.
             */
            {"LLAMINAR_PERF_STATS_GPU_STAGE_TIMING", "1"},
        });

        std::string model_path;
        std::vector<int32_t> reference_prompt;
        std::vector<int32_t> unused_reference_tokens;
        loadMoEReferenceInputs(
            test_case,
            &model_path,
            &reference_prompt,
            &unused_reference_tokens);
        if (moeReferenceInputsStoppedCurrentTest())
            return;

        /*
         * Preserve the server request order explicitly. The repeated 39- and
         * 62-row geometries represent different prompts in the server suite;
         * the loop below therefore gives every entry a unique first token so
         * equal geometry cannot accidentally turn into a cache hit.
         */
        constexpr std::array<size_t, 11> kProductionHistoryPromptLengths = {
            39u,
            68u,
            39u,
            62u,
            62u,
            431u,
            36u,
            1815u,
            1813u,
            1801u,
            1305u,
        };
        constexpr int kHistoryDecodeTokenBudget = 64;
        constexpr int kStructuredDecodeTokenBudget = 512;
        constexpr int kRestoreDecodeTokenBudget = 16;
        constexpr int kPrefixBlockSize = 64;
        const std::vector<int32_t> structured_prompt =
            structuredGenerationPromptTokens(model_path);
        ASSERT_GE(
            reference_prompt.size(),
            *std::max_element(
                kProductionHistoryPromptLengths.begin(),
                kProductionHistoryPromptLengths.end()))
            << "GPU rebalance lifecycle stress metadata is too short";

        auto factory = createOrchestrationRunnerFactory();
        OrchestrationConfig runner_config =
            makeMoEPrefixRestoreConfig(
                test_case,
                model_path,
                /*enable_prefix_cache=*/true,
                kPrefixBlockSize,
                /*enable_mtp=*/true,
                /*mtp_draft_tokens=*/2);
        /*
         * Match the canonical CUDA2/ROCm2 E2E cells rather than inheriting the
         * parity helper's intentionally generous 4 GiB RAM tier. One Qwen3.6
         * 64-token prefix block is roughly 102 MiB per participant. A 1 GiB
         * budget therefore forces the 443-token request and the surrounding
         * prompt identities through real LRU eviction, demotion, and later
         * terminal-state publication pressure before the exact structured
         * request begins. The former 4 GiB setting retained the whole matrix
         * and could not reproduce the long-context mirrored-state failure.
         */
        runner_config.prefix_cache.ram_budget_bytes =
            1024ull * 1024ull * 1024ull;
        auto runner =
            factory->createFromOrchestrationConfig(runner_config);
        ASSERT_NE(runner, nullptr);
        ASSERT_TRUE(runner->initialize()) << runner->lastError();
        /*
         * This is a transaction-lifetime regression, not a language-quality
         * assertion. Stop tokens would let an otherwise valid completion end
         * before the historical 512-token failure window and silently weaken
         * the workload whenever model weights or prompt formatting change.
         */
        runner->setStopTokens({});

        SamplingParams greedy;
        greedy.temperature = 0.0f;
        PerfStatsCollector::reset();

        for (size_t request_index = 0;
             request_index < kProductionHistoryPromptLengths.size();
             ++request_index)
        {
            const size_t prompt_length =
                kProductionHistoryPromptLengths[request_index];
            std::vector<int32_t> history_prompt(
                reference_prompt.begin(),
                reference_prompt.begin() +
                    static_cast<std::ptrdiff_t>(prompt_length));

            /*
             * A unique, ordinary vocabulary token prevents the equal-length
             * requests above from sharing their first cache block. All
             * remaining rows retain the real long-ledger routing pattern.
             */
            history_prompt.front() =
                static_cast<int32_t>(1000u + request_index);
            const GenerationResult result =
                runner->generate(
                    history_prompt,
                    kHistoryDecodeTokenBudget,
                    greedy);
            ASSERT_TRUE(result.error.empty())
                << test_case.name << " production-history request " << request_index
                << " failed for prompt_rows=" << history_prompt.size()
                << ": " << result.error;
            ASSERT_FALSE(result.tokens.empty())
                << test_case.name << " production-history request " << request_index
                << " emitted no tokens";
            ASSERT_EQ(
                result.tokens.size(),
                static_cast<size_t>(kHistoryDecodeTokenBudget))
                << test_case.name << " production-history request " << request_index
                << " ended before its transaction budget";

            const PrefixRuntimeStateSnapshot probe =
                runner->prefixStateProbe();
            EXPECT_FALSE(probe.mtp_bypassed)
                << "history_request=" << request_index
                << " reason=" << probe.mtp_bypass_reason;
            EXPECT_GE(probe.mtp_verifier_runs, 1u)
                << "history_request=" << request_index;
            EXPECT_FALSE(probe.prefix_request.hit)
                << "history_request=" << request_index
                << " prompt_rows=" << history_prompt.size();
        }

        const GenerationResult structured_result =
            runner->generate(
                structured_prompt,
                kStructuredDecodeTokenBudget,
                greedy);
        ASSERT_TRUE(structured_result.error.empty())
            << test_case.name << " structured request failed after production history: "
            << structured_result.error;
        ASSERT_FALSE(structured_result.tokens.empty())
            << test_case.name << " structured request emitted no tokens";
        ASSERT_EQ(
            structured_result.tokens.size(),
            static_cast<size_t>(kStructuredDecodeTokenBudget))
            << test_case.name
            << " structured request ended before the original server failure window";
        const PrefixRuntimeStateSnapshot structured_probe =
            runner->prefixStateProbe();
        EXPECT_FALSE(structured_probe.mtp_bypassed)
            << "structured request reason="
            << structured_probe.mtp_bypass_reason;
        EXPECT_GE(structured_probe.mtp_verifier_runs, 1u);
        EXPECT_FALSE(structured_probe.prefix_request.hit);

        const GenerationResult restore_result =
            runner->generate(
                structured_prompt,
                kRestoreDecodeTokenBudget,
                greedy);
        ASSERT_TRUE(restore_result.error.empty())
            << test_case.name << " structured restore failed: "
            << restore_result.error;
        ASSERT_FALSE(restore_result.tokens.empty())
            << test_case.name << " structured restore emitted no tokens";
        ASSERT_EQ(
            restore_result.tokens.size(),
            static_cast<size_t>(kRestoreDecodeTokenBudget))
            << test_case.name << " structured restore ended before its budget";
        const PrefixRuntimeStateSnapshot restore_probe =
            runner->prefixStateProbe();
        EXPECT_FALSE(restore_probe.mtp_bypassed)
            << "structured restore reason="
            << restore_probe.mtp_bypass_reason;
        EXPECT_GE(restore_probe.mtp_verifier_runs, 1u);
        ASSERT_TRUE(restore_probe.prefix_request.hit);
        ASSERT_EQ(
            restore_probe.prefix_request.matched_tokens,
            static_cast<int>(structured_prompt.size()));

        /*
         * Diagnostic export is deliberately outside the inference hot path. It
         * waits only after the entire production-shaped request sequence so
         * the assertions below observe completed graph-owned maintenance
         * records without inserting a host fence between the producer and
         * consumer requests under test.
         */
        runner->drainCompletedDecodeBoundaryMaintenanceDiagnostics();
        const auto records = PerfStatsCollector::snapshot(
            {"mtp", "prefix_cache", "moe_rebalance", "kernel"});
        runner->shutdown();

        expectMoEPrefixCachePerfPath(
            records,
            test_case.name + " lifecycle stress");
        expectMoEGreedyMTPPublicationPath(
            test_case,
            records,
            test_case.name + " lifecycle stress");
        expectMoEBackendKernelPerfPath(
            test_case,
            records,
            test_case.name + " lifecycle stress");
        expectPerfCounterPositive(
            records,
            "mtp",
            "live_prefix_checkpoint_logical_captures",
            test_case.name + " lifecycle stress");
        expectPerfCounterPositive(
            records,
            "mtp",
            "live_prefix_checkpoint_device_sequence_state_captures",
            test_case.name + " lifecycle stress");
        expectPerfCounterZero(
            records,
            "mtp",
            "live_prefix_checkpoint_payload_required",
            test_case.name + " lifecycle stress");
        EXPECT_TRUE(hasMTPPerfRecordTag(
            records,
            "live_prefix_checkpoint_terminal_hidden_captures",
            "implementation",
            "device_to_device"))
            << test_case.name
            << " lifecycle stress must keep terminal-hidden checkpoints in VRAM";
        EXPECT_TRUE(hasMTPPerfRecordTag(
            records,
            "live_prefix_checkpoint_device_sequence_state_captures",
            "cache",
            "shifted"))
            << test_case.name
            << " lifecycle stress must checkpoint every shifted cache's "
               "device-owned sequence metadata";
        EXPECT_TRUE(hasMTPPerfRecordTag(
            records,
            "live_prefix_checkpoint_device_sequence_state_captures",
            "implementation",
            "device_kernel"))
            << test_case.name
            << " lifecycle stress must keep shifted-cache rollback metadata "
               "entirely on device";
        /*
         * A single final diagnostics drain exports the currently completed
         * maintenance slot; it intentionally does not fence and replay every
         * prior request's copy/apply status. Prove that the selected rebalance
         * policy planned real transfer-backed assignments during this stress,
         * then use the common health gate for zero error counters and
         * producer/consumer event ordering. The ordinary two-request parity
         * cells drain each wave and remain the canonical copied/applied-arrival
         * counter proof.
         */
        if (test_case.moe_rebalance->mode == MoERebalanceRuntimeMode::LLEP)
        {
            expectPerfCounterPositive(
                records,
                "moe_rebalance",
                "device_rebalance_llep_weight_transfer_count",
                test_case.name + " lifecycle stress");
            expectPerfCounterPositive(
                records,
                "moe_rebalance",
                "device_rebalance_llep_assignment_span_count",
                test_case.name + " lifecycle stress");
        }
        else
        {
            expectDynamicRebalancePlacementPositive(
                records,
                test_case.name + " lifecycle stress");
        }
        expectPerfCounterPositive(
            records,
            "moe_rebalance",
            "device_rebalance_planned_arrivals",
            test_case.name + " lifecycle stress");
        expectPerfCounterPositive(
            records,
            "moe_rebalance",
            "device_rebalance_apply_applied_arrivals",
            test_case.name + " lifecycle stress");
        expectMoERebalancePerfPath(
            test_case,
            records,
            test_case.name + " lifecycle stress",
            /*require_fresh_movement=*/false);
    }
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, MTPGreedyMatchesBaselineTokens_CUDA2TPHotOnly)
{
    runMoEMTPParity(cudaOnlyExpertOverlayCase(), false);
}

/**
 * @brief Proves long-context CUDA grouped MTP independently of expert movement.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, MTPGreedyMatchesBaselineTokens_CUDA2TPLongContextHotOnly)
{
    runMoEMTPParity(cudaOnlyLongContextCase(), false);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefixCacheMTPRestore_CUDA2TPHotOnly)
{
    runMoEMTPParity(cudaOnlyExpertOverlayCase(), true);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, SerialStochasticSameSeedReplay_CUDA2TPHotOnly)
{
    runMoESerialStochasticSameSeedReplay(cudaOnlyStochasticBenchmarkCase());
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDepth1VerifierMatchesAfterClearCache_CUDA2TPHotOnly)
{
    runMoEStochasticMTPVerifierParity(cudaOnlyStochasticBenchmarkCase(), 1);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDepth2VerifierMatchesAfterClearCache_CUDA2TPHotOnly)
{
    runMoEStochasticMTPVerifierParity(cudaOnlyStochasticBenchmarkCase(), 2);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDepth3VerifierMatchesAfterClearCache_CUDA2TPHotOnly)
{
    runMoEStochasticMTPVerifierParity(
        cudaOnlyStochasticBenchmarkCase(),
        3,
        true);
}

/**
 * @brief Prove the maximum CUDA grouped verifier shape survives prefix restore.
 *
 * Dynamic depth may demote before a cache-hit request and thereby exercise
 * fewer than four target rows. Keep this fixed-depth cell so restored terminal
 * logits, hidden state, shifted MTP KV, resident outcome reduction, and direct
 * publication are all validated at the maximum supported verifier shape.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDepth3VerifierMatchesAfterPrefixRestore_CUDA2TPHotOnly)
{
    runMoEStochasticMTPVerifierParity(
        cudaOnlyStochasticBenchmarkCase(),
        3,
        true,
        MTPDepthPolicyConfig{},
        true);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDynamicDepthVerifierMatchesAfterClearCache_CUDA2TPHotOnly)
{
    runMoEStochasticMTPVerifierParity(
        cudaOnlyStochasticBenchmarkCase(),
        3,
        false,
        qwen36MoEStochasticDynamicDepthPolicy(3));
}

/**
 * @brief Isolates long-position CUDA stochastic MTP from prefix archival.
 *
 * The prefix-enabled long-context control below fails only after its explicit
 * clearCache() boundary. This companion retains the identical prompt, static
 * expert placement, dynamic-depth verifier, graph replay, and reused-runner
 * lifecycle while omitting prefix lookup/harvest/restore. A failure here names
 * generic request reset or captured replay; a pass names prefix-state lifetime.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDynamicDepthVerifierMatchesAfterClearCache_CUDA2TPLongContextHotOnly)
{
    runMoEStochasticMTPVerifierParity(
        cudaOnlyLongContextCase(),
        3,
        false,
        qwen36MoEStochasticDynamicDepthPolicy(3),
        false,
        20);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDynamicDepthVerifierMatchesAfterPrefixRestore_CUDA2TPHotOnly)
{
    runMoEStochasticMTPVerifierParity(
        cudaOnlyStochasticBenchmarkCase(),
        3,
        false,
        qwen36MoEStochasticDynamicDepthPolicy(3),
        true);
}

/**
 * @brief Proves CUDA stochastic grouped MTP at long absolute positions without
 *        runtime expert movement.
 *
 * This control uses the same long production-tokenized ledger and exact prefix
 * restore as the Dynamic/LLEP cells while retaining static expert placement.
 * It distinguishes a long-position stochastic verifier defect from a runtime
 * placement handoff defect.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDynamicDepthVerifierMatchesAfterPrefixRestore_CUDA2TPLongContextHotOnly)
{
    runMoEStochasticMTPVerifierParity(
        cudaOnlyLongContextCase(),
        3,
        false,
        qwen36MoEStochasticDynamicDepthPolicy(3),
        true);
}

/**
 * @brief Proves CUDA LocalTP grouped verifier rows match rowwise serial decode.
 *
 * Rejected-state publication assumes the verifier row logits are already
 * decode-equivalent.  This narrower gate exercises routing, grouped MoE
 * kernels, row-indexed all-position logits, and RankOrchestrator cross-shard
 * sampling before any accepted-state publication logic runs.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, GroupedVerifierRowsMatchSerial_CUDA2TPHotOnly)
{
    runMoEMainVerifierGroupedRowsMatchSerialDecode(
        cudaOnlyExpertOverlayCase(),
        /*verifier_row_count=*/2);
}

/**
 * @brief Proves CUDA resident publication handles a rejected correction row.
 *
 * The full prefix-cache MTP cell once exposed a mismatch where the grouped
 * verifier published one accepted state row, emitted a correction token, and then
 * continued from a state that was not bitwise-equivalent to serial decode. This
 * focused regression forces that shape directly: row 0 is accepted state, row 1
 * is an intentionally wrong draft, and compact device-resident publication must
 * match serial continuation after the correction token is consumed.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, DeviceResidentRejectPublicationMatchesSerial_CUDA2TPHotOnly)
{
    ScopedEnvironmentValues perf_stats_enabled({
        {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        {"LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS", "1"},
        {"LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE", "1"},
    });
    PerfStatsCollector::reset();
    runMoEMainVerifierAllPositionRowsMatchSerialDecode(
        cudaOnlyExpertOverlayCase(),
        /*use_row_indexed_logits=*/true,
        /*use_skip_gather=*/true,
        /*verify_compact_device_outcome=*/true,
        /*use_deferred_verifier_sync=*/true,
        /*verify_published_state_continuation=*/true,
        /*verifier_row_count=*/2,
        /*verify_device_resident_publication=*/true,
        /*expect_grouped_moe_verifier_prefill=*/false,
        /*force_first_speculative_rejection=*/true);
    PerfStatsCollector::reset();
}

/**
 * @brief Proves CUDA resident publication handles an all-accepted verifier row.
 *
 * Rejected-correction publication commits only the first target state and
 * therefore cannot prove the terminal-row path. This companion regression
 * accepts the speculative row, publishes every grouped verifier state from the
 * captured graph, and requires the resulting KV/GDN/short-conv state plus the
 * next eight decode tokens to remain byte-equivalent to serial decode.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, DeviceResidentAcceptedPublicationMatchesSerial_CUDA2TPHotOnly)
{
    ScopedEnvironmentValues perf_stats_enabled({
        {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        {"LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS", "1"},
        {"LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE", "1"},
    });
    PerfStatsCollector::reset();
    runMoEMainVerifierAllPositionRowsMatchSerialDecode(
        cudaOnlyExpertOverlayCase(),
        /*use_row_indexed_logits=*/true,
        /*use_skip_gather=*/true,
        /*verify_compact_device_outcome=*/true,
        /*use_deferred_verifier_sync=*/true,
        /*verify_published_state_continuation=*/true,
        /*verifier_row_count=*/2,
        /*verify_device_resident_publication=*/true,
        /*expect_grouped_moe_verifier_prefill=*/false,
        /*force_first_speculative_rejection=*/false);
    PerfStatsCollector::reset();
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefixRestorePartialHit_CUDA2TPHotOnly)
{
    runMoEPrefixRestoreParity(
        cudaOnlyPartialPrefixCase(),
        PrefixRestoreParityMode::PartialHit);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefixRestorePartialHit_CUDA2TPDynamicPhaseSplit)
{
    runMoEPrefixRestoreParity(
        cudaOnlyDynamicPhaseSplitCase(),
        PrefixRestoreParityMode::PartialHit);
}

/**
 * @brief Isolates long-context CUDA Dynamic MTP from prefix-cache activity.
 *
 * This control executes the same production-tokenized 2383-row prompt,
 * movement-friendly Dynamic maintenance policy, grouped verifier, and
 * device-resident publication path as the prefix-enabled cell below. The sole
 * difference is that prefill does not concurrently archive reusable state.
 * Pairing the two cells tells a core long-context verifier defect apart from a
 * prefix snapshot/import lifetime defect.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, MTPGreedyMatchesBaselineTokens_CUDA2TPDynamicPhaseSplit)
{
    runMoEMTPParity(cudaOnlyDynamicPhaseSplitCase(), false);
}

/**
 * @brief Ensures CUDA Dynamic expert movement preserves MTP prefix restore state.
 *
 * This is the Dynamic companion to the LLEP MTP prefix-cache cell below.  It
 * proves that grouped verifier publication, prefix-cache terminal state, and
 * runtime expert-placement maintenance remain coherent after a restored
 * request, not merely during a plain partial-prefix restore.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefixCacheMTPRestore_CUDA2TPDynamicPhaseSplit)
{
    runMoEMTPParity(cudaOnlyDynamicPhaseSplitCase(), true);
}

/**
 * @brief Proves stochastic grouped MTP remains serial-equivalent while CUDA
 *        Dynamic expert movement and exact prefix restore are active.
 *
 * This cell combines the device-owned speculative sampler, adaptive MTP depth,
 * runtime Dynamic placement, full NCCL graph capture, and restored recurrent
 * state in one production request sequence.  Separate stochastic and rebalance
 * tests cannot expose ordering defects between those independently asynchronous
 * paths.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDynamicDepthVerifierMatchesAfterPrefixRestore_CUDA2TPDynamicPhaseSplit)
{
    runMoEStochasticMTPVerifierParity(
        cudaOnlyDynamicPhaseSplitCase(),
        3,
        false,
        qwen36MoEStochasticDynamicDepthPolicy(3),
        true);
}

/**
 * @brief Reproduces asynchronous CUDA Dynamic publication races across requests.
 *
 * Dynamic and LLEP share the captured maintenance and prefix-cache ownership
 * protocol, but their device planners and placement mutations are distinct.
 * Keep a dedicated Dynamic cell so a late mirrored-terminal-hidden divergence
 * cannot hide behind the otherwise comprehensive LLEP lifecycle stress.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefillRequestBoundaryStress_CUDA2TPDynamicPhaseSplit)
{
    runRebalancedPrefillRequestBoundaryStress(cudaOnlyDynamicPhaseSplitCase());
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefixRestorePartialHit_CUDA2TPLLEPPhaseSplit)
{
    runMoEPrefixRestoreParity(
        cudaOnlyLLEPPhaseSplitCase(),
        PrefixRestoreParityMode::PartialHit);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefixCacheMTPRestore_CUDA2TPLLEPPhaseSplit)
{
    runMoEMTPParity(cudaOnlyLLEPPhaseSplitCase(), true);
}

/**
 * @brief Proves stochastic grouped MTP remains serial-equivalent while CUDA
 *        LLEP assignment and exact prefix restore are active.
 *
 * LLEP owns a distinct device planner and transfer-backed assignment path from
 * Dynamic rebalance.  This regression requires that path to coexist with the
 * adaptive stochastic verifier and its fully captured NCCL publication graph.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDynamicDepthVerifierMatchesAfterPrefixRestore_CUDA2TPLLEPPhaseSplit)
{
    runMoEStochasticMTPVerifierParity(
        cudaOnlyLLEPPhaseSplitCase(),
        3,
        false,
        qwen36MoEStochasticDynamicDepthPolicy(3),
        true);
}

/**
 * @brief Reproduces asynchronous CUDA LLEP publication races across requests.
 *
 * This is the focused lifecycle regression for the long-context server trap:
 * one initialized runner must survive the production prefill/decode history
 * and a subsequent exact prefix restore while graph-captured grouped MTP and
 * device maintenance remain live.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefillRequestBoundaryStress_CUDA2TPLLEPPhaseSplit)
{
    runRebalancedPrefillRequestBoundaryStress(cudaOnlyLLEPPhaseSplitCase());
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, MTPGreedyMatchesBaselineTokens_ROCm2TPHotOnly)
{
    runMoEMTPParity(rocmOnlyExpertOverlayCase(), false);
}

/**
 * @brief Proves long-context ROCm grouped MTP independently of expert movement.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, MTPGreedyMatchesBaselineTokens_ROCm2TPLongContextHotOnly)
{
    runMoEMTPParity(rocmOnlyLongContextCase(), false);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefixCacheMTPRestore_ROCm2TPHotOnly)
{
    runMoEMTPParity(rocmOnlyExpertOverlayCase(), true);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, SerialStochasticSameSeedReplay_ROCm2TPHotOnly)
{
    runMoESerialStochasticSameSeedReplay(rocmOnlyStochasticBenchmarkCase());
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDepth1VerifierMatchesAfterClearCache_ROCm2TPHotOnly)
{
    runMoEStochasticMTPVerifierParity(rocmOnlyStochasticBenchmarkCase(), 1);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDepth2VerifierMatchesAfterClearCache_ROCm2TPHotOnly)
{
    runMoEStochasticMTPVerifierParity(rocmOnlyStochasticBenchmarkCase(), 2);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDepth3VerifierMatchesAfterClearCache_ROCm2TPHotOnly)
{
    runMoEStochasticMTPVerifierParity(
        rocmOnlyStochasticBenchmarkCase(),
        3,
        true);
}

/**
 * @brief Prove the maximum ROCm grouped verifier shape survives prefix restore.
 *
 * This is deliberately symmetric with the CUDA cell above. It keeps the RCCL
 * collective, resident stochastic reducer, restored attention/GDN/short-conv
 * state, and accepted-state publication path covered at four verifier rows,
 * independent of dynamic-depth controller decisions.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDepth3VerifierMatchesAfterPrefixRestore_ROCm2TPHotOnly)
{
    runMoEStochasticMTPVerifierParity(
        rocmOnlyStochasticBenchmarkCase(),
        3,
        true,
        MTPDepthPolicyConfig{},
        true);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDynamicDepthVerifierMatchesAfterClearCache_ROCm2TPHotOnly)
{
    runMoEStochasticMTPVerifierParity(
        rocmOnlyStochasticBenchmarkCase(),
        3,
        false,
        qwen36MoEStochasticDynamicDepthPolicy(3));
}

/**
 * @brief Mirrors the long-position request-reset isolation control on ROCm.
 *
 * Keeping this matrix symmetric makes a shared lifecycle defect distinguishable
 * from CUDA graph replay or ROCm graph replay behavior without weakening the
 * production path or forcing deterministic diagnostic kernels.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDynamicDepthVerifierMatchesAfterClearCache_ROCm2TPLongContextHotOnly)
{
    runMoEStochasticMTPVerifierParity(
        rocmOnlyLongContextCase(),
        3,
        false,
        qwen36MoEStochasticDynamicDepthPolicy(3),
        false,
        20);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDynamicDepthVerifierMatchesAfterPrefixRestore_ROCm2TPHotOnly)
{
    runMoEStochasticMTPVerifierParity(
        rocmOnlyStochasticBenchmarkCase(),
        3,
        false,
        qwen36MoEStochasticDynamicDepthPolicy(3),
        true);
}

/**
 * @brief Mirrors the static-placement long-context stochastic control on ROCm.
 *
 * The fixture holds expert placement fixed while preserving the exact prompt,
 * MTP depth controller, RCCL grouped publication, and prefix restore used by
 * the movement cells. A failure here belongs to long-position stochastic MTP;
 * a pass isolates the defect to Dynamic/LLEP placement ordering.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDynamicDepthVerifierMatchesAfterPrefixRestore_ROCm2TPLongContextHotOnly)
{
    runMoEStochasticMTPVerifierParity(
        rocmOnlyLongContextCase(),
        3,
        false,
        qwen36MoEStochasticDynamicDepthPolicy(3),
        true);
}

/**
 * @brief Proves ROCm LocalTP grouped verifier rows match rowwise serial decode.
 *
 * This is the focused accuracy gate for the hot-only ROCm path before
 * rejected-token publication is exercised.  It keeps the regression surface
 * tight enough to identify row math, routing, and cross-shard sampling bugs
 * without running the full prefix-cache MTP harness.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, GroupedVerifierRowsMatchSerial_ROCm2TPHotOnly)
{
    runMoEMainVerifierGroupedRowsMatchSerialDecode(
        rocmOnlyExpertOverlayCase(),
        /*verifier_row_count=*/2);
}

/**
 * @brief Proves ROCm resident publication handles a rejected correction row.
 *
 * ROCm uses separate grouped MoE kernels and RCCL-backed LocalTP publication.
 * Keeping this rejected-correction proof next to the hot-only prefix-cache cell
 * gives the flaky long-context failure a deterministic regression: after
 * publishing exactly one accepted verifier state row, consuming the correction
 * token must produce the same continuation as serial decode from the verifier
 * base.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, DeviceResidentRejectPublicationMatchesSerial_ROCm2TPHotOnly)
{
    ScopedEnvironmentValues perf_stats_enabled({
        {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        {"LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS", "1"},
        {"LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE", "1"},
    });
    PerfStatsCollector::reset();
    runMoEMainVerifierAllPositionRowsMatchSerialDecode(
        rocmOnlyExpertOverlayCase(),
        /*use_row_indexed_logits=*/true,
        /*use_skip_gather=*/true,
        /*verify_compact_device_outcome=*/true,
        /*use_deferred_verifier_sync=*/true,
        /*verify_published_state_continuation=*/true,
        /*verifier_row_count=*/2,
        /*verify_device_resident_publication=*/true,
        /*expect_grouped_moe_verifier_prefill=*/false,
        /*force_first_speculative_rejection=*/true);
    PerfStatsCollector::reset();
}

/**
 * @brief Proves ROCm resident publication handles an all-accepted verifier row.
 *
 * This exercises the terminal accepted-row publication owner on both mirrored
 * RCCL participants. The published recurrent and cache state must be bytewise
 * equal to serial decode before an eight-token continuation is compared.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, DeviceResidentAcceptedPublicationMatchesSerial_ROCm2TPHotOnly)
{
    ScopedEnvironmentValues perf_stats_enabled({
        {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        {"LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS", "1"},
        {"LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE", "1"},
    });
    PerfStatsCollector::reset();
    runMoEMainVerifierAllPositionRowsMatchSerialDecode(
        rocmOnlyExpertOverlayCase(),
        /*use_row_indexed_logits=*/true,
        /*use_skip_gather=*/true,
        /*verify_compact_device_outcome=*/true,
        /*use_deferred_verifier_sync=*/true,
        /*verify_published_state_continuation=*/true,
        /*verifier_row_count=*/2,
        /*verify_device_resident_publication=*/true,
        /*expect_grouped_moe_verifier_prefill=*/false,
        /*force_first_speculative_rejection=*/false);
    PerfStatsCollector::reset();
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefixRestorePartialHit_ROCm2TPHotOnly)
{
    runMoEPrefixRestoreParity(
        rocmOnlyPartialPrefixCase(),
        PrefixRestoreParityMode::PartialHit);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefixRestorePartialHit_ROCm2TPDynamicPhaseSplit)
{
    runMoEPrefixRestoreParity(
        rocmOnlyDynamicPhaseSplitCase(),
        PrefixRestoreParityMode::PartialHit);
}

/**
 * @brief Mirrors the CUDA long-context Dynamic no-prefix isolation control.
 *
 * Symmetric coverage proves whether the failure belongs to shared verifier
 * state ownership or to one vendor's graph/KV implementation.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, MTPGreedyMatchesBaselineTokens_ROCm2TPDynamicPhaseSplit)
{
    runMoEMTPParity(rocmOnlyDynamicPhaseSplitCase(), false);
}

/**
 * @brief Ensures ROCm Dynamic expert movement preserves MTP prefix restore state.
 *
 * ROCm has separate grouped MoE kernels, RCCL synchronization, and graph-capture
 * behavior from CUDA.  Keeping this cell beside the CUDA Dynamic variant makes
 * the ExpertOverlay matrix catch backend-specific drift in the MTP/prefix-cache
 * state handoff.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefixCacheMTPRestore_ROCm2TPDynamicPhaseSplit)
{
    runMoEMTPParity(rocmOnlyDynamicPhaseSplitCase(), true);
}

/**
 * @brief Mirrors the stochastic Dynamic movement and prefix-restore proof on
 *        the ROCm/RCCL implementation.
 *
 * CUDA and ROCm use different grouped kernels, graph APIs, and collective
 * runtimes.  Keeping this cell symmetric prevents a device-owned mailbox or
 * adaptive-depth fix from becoming accidentally CUDA-specific.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDynamicDepthVerifierMatchesAfterPrefixRestore_ROCm2TPDynamicPhaseSplit)
{
    runMoEStochasticMTPVerifierParity(
        rocmOnlyDynamicPhaseSplitCase(),
        3,
        false,
        qwen36MoEStochasticDynamicDepthPolicy(3),
        true);
}

/**
 * @brief Mirrors the multi-request Dynamic lifecycle regression on ROCm.
 *
 * The full-tier server failure first appears after many successful MTP samples
 * and real Dynamic movement waves. This fixture retains one ROCm runner across
 * the production prompt-shape history and final prefix-restore transition,
 * making that ownership lifetime part of the focused integration gate.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefillRequestBoundaryStress_ROCm2TPDynamicPhaseSplit)
{
    runRebalancedPrefillRequestBoundaryStress(rocmOnlyDynamicPhaseSplitCase());
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefixRestorePartialHit_ROCm2TPLLEPPhaseSplit)
{
    runMoEPrefixRestoreParity(
        rocmOnlyLLEPPhaseSplitCase(),
        PrefixRestoreParityMode::PartialHit);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefixCacheMTPRestore_ROCm2TPLLEPPhaseSplit)
{
    runMoEMTPParity(rocmOnlyLLEPPhaseSplitCase(), true);
}

/**
 * @brief Mirrors the stochastic LLEP movement and prefix-restore proof on
 *        the ROCm/RCCL implementation.
 *
 * The test requires real LLEP transfer evidence, adaptive stochastic verifier
 * activity, exact same-seed token replay, and whole-graph RCCL execution in a
 * single long-context fixture.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, StochasticMTPDynamicDepthVerifierMatchesAfterPrefixRestore_ROCm2TPLLEPPhaseSplit)
{
    runMoEStochasticMTPVerifierParity(
        rocmOnlyLLEPPhaseSplitCase(),
        3,
        false,
        qwen36MoEStochasticDynamicDepthPolicy(3),
        true);
}

/**
 * @brief Mirrors the CUDA multi-request lifecycle regression on ROCm.
 *
 * ROCm owns the same graph-local routed-kernel state and asynchronous LLEP
 * maintenance contract. This companion prevents a future ownership or
 * request-boundary fix from becoming accidentally CUDA-only.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefillRequestBoundaryStress_ROCm2TPLLEPPhaseSplit)
{
    runRebalancedPrefillRequestBoundaryStress(rocmOnlyLLEPPhaseSplitCase());
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, MTPGreedyMatchesBaselineTokens_ROCm2TPHot_CPU2LocalTPCold)
{
    runMoEMTPParity(expertOverlayCase(), false);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefixCacheMTPRestore_ROCm2TPHot_CPU2LocalTPCold)
{
    runMoEMTPParity(expertOverlayCase(), true);
}

/**
 * @brief Guards every phase-split fixture against short or heavyweight inputs.
 *
 * This test is intentionally source-level and model-free.  The expensive
 * runtime cells validate device behavior; this one catches the fixture
 * regression where a missing long-context snapshot was regenerated from only
 * its 13-token header. It also proves CUDA/ROCm and Dynamic/LLEP use direct
 * production tokenization instead of launching a 35B PyTorch decode whose
 * outputs this state-lifetime suite does not consume.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PartialPrefixFixturesUseLongMetadata)
{
    const std::string expected_prompt = qwen36MoELongNeedleParityPrompt();
    const std::array<MoEPrefixRestoreParityCase, 6> test_cases = {
        cudaOnlyPartialPrefixCase(),
        rocmOnlyPartialPrefixCase(),
        cudaOnlyDynamicPhaseSplitCase(),
        cudaOnlyLLEPPhaseSplitCase(),
        rocmOnlyDynamicPhaseSplitCase(),
        rocmOnlyLLEPPhaseSplitCase(),
    };

    ASSERT_GT(expected_prompt.size(), 4096u)
        << "the shared long-ledger prompt unexpectedly lost its workload depth";
    EXPECT_NE(expected_prompt.find("LCJSON-ALPHA-314159"), std::string::npos);
    EXPECT_NE(expected_prompt.find("LCJSON-MIDDLE-271828"), std::string::npos);
    EXPECT_NE(expected_prompt.find("LCJSON-OMEGA-161803"), std::string::npos);

    for (const auto &test_case : test_cases)
    {
        EXPECT_EQ(test_case.prompt, expected_prompt);
        EXPECT_EQ(test_case.max_seq_len, kPhaseSplitLongDecodeMaxSeqLen);
        EXPECT_GT(test_case.prefix_restore_prompt_token_limit, 256);
        EXPECT_GE(test_case.minimum_prompt_tokens, 640u);
        EXPECT_EQ(
            test_case.reference_input_source,
            MoEReferenceInputSource::ModelTokenizer);
        EXPECT_TRUE(test_case.default_metadata_path.empty());
        EXPECT_TRUE(test_case.metadata_envs.empty());
    }
}

/**
 * @brief Guards the mixed hot/cold MTP fixture's dense decode ownership.
 *
 * The CPU cold tier is a routed-expert fallback, not the owner of verifier
 * logits.  This model-free check catches the regression where the mixed plan
 * forgot to request phase-split dense decode and therefore left the ROCm hot
 * MTP sidecar without terminal dense/MTP bindings.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, MixedHotColdMTPUsesPhaseSplitDenseDecode)
{
    const auto test_case = expertOverlayCase();
    ASSERT_TRUE(test_case.moe_routed_expert_plan);

    const auto &spec =
        test_case.moe_routed_expert_plan->continuation_domain_spec;
    EXPECT_EQ(spec.effectiveDensePolicy(),
              DenseParallelPolicy::PrefillTensorParallelDecodeReplicated);
    EXPECT_TRUE(spec.dense_tp_enabled);
    EXPECT_TRUE(spec.dense_decode_replicated);
    EXPECT_EQ(test_case.moe_routed_expert_plan->continuation_domain,
              "qwen36_moe_rocm_hot");
}

/**
 * @brief Keep disjoint CUDA and ROCm parity processes in separate lock domains.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, ProcessLockSeparatesGpuBackends)
{
    char program[] = "qwen36_moe_expert_overlay";
    char cuda_filter[] = "--gtest_filter=*CUDA2TPHotOnly";
    char rocm_filter[] = "--gtest_filter=*ROCm2TPHotOnly";
    char mixed_filter[] = "--gtest_filter=*CUDA2TPHotOnly:*ROCm2TPHotOnly";

    char *cuda_args[] = {program, cuda_filter};
    char *rocm_args[] = {program, rocm_filter};
    char *mixed_args[] = {program, mixed_filter};
    char *unfiltered_args[] = {program};

    EXPECT_STREQ(
        parityProcessLockPath(2, cuda_args),
        "/tmp/llaminar_qwen36_moe_expert_overlay_cuda.lock");
    EXPECT_STREQ(
        parityProcessLockPath(2, rocm_args),
        "/tmp/llaminar_qwen36_moe_expert_overlay_rocm.lock");
    EXPECT_STREQ(
        parityProcessLockPath(2, mixed_args),
        "/tmp/llaminar_qwen36_moe_expert_overlay_parity.lock");
    EXPECT_STREQ(
        parityProcessLockPath(1, unfiltered_args),
        "/tmp/llaminar_qwen36_moe_expert_overlay_parity.lock");
}

int main(int argc, char **argv)
{
    std::unique_ptr<ScopedParityProcessLock> parity_lock;
    try
    {
        parity_lock = std::make_unique<ScopedParityProcessLock>(
            parityProcessLockPath(argc, argv));
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to acquire Qwen36 MoE ExpertOverlay parity lock: "
                  << e.what() << std::endl;
        return 1;
    }

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    initializeLogging();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
