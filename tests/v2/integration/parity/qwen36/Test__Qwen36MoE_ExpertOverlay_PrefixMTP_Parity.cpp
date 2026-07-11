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
#include "utils/Logger.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
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
     * @brief Prompt header embedded in the long expert-overlay metadata files.
     *
     * metadataLooksUsable() intentionally verifies the prompt string before a
     * parity case consumes token_ids from a metadata file.  Keeping this header
     * beside the metadata-path helpers makes the partial-prefix fixtures
     * explicit: they run the long ledger request already generated for the
     * expert-overlay long-context matrix, then cap the request locally when a
     * partial cache hit is the behavior under test.
     */
    constexpr const char *kLongLedgerPromptHeader =
        "Task: read the ledger and return one minified JSON object.";

    /**
     * @brief Serializes this process with other large Qwen3.6 expert-overlay parity cases.
     *
     * The fixture materializes multi-GPU runners and large pinned logits buffers.
     * Taking an interprocess lock keeps unrelated CTest shards from trying to
     * consume the same GPU memory and collective resources concurrently.
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
     * @brief Converts a hot-only expert-overlay case into a phase-split migration probe.
     *
     * The metadata path points at long-decode PyTorch snapshots.  Prefix restore
     * partial-hit tests cap the prompt after loading metadata, while MTP restore
     * tests use the full metadata prompt; therefore the case must declare a
     * long-context window up front instead of depending on the prefix cap.
     *
     * @param test_case Case object to mutate.
     * @param mode Rebalance mode under test.
     * @param metadata_path Long-decode PyTorch metadata file to load.
     */
    void configurePhaseSplitMigrationProbe(
        MoEPrefixRestoreParityCase &test_case,
        MoERebalanceRuntimeMode mode,
        const std::string &metadata_path)
    {
        test_case.name += mode == MoERebalanceRuntimeMode::LLEP
                              ? " LLEP phase-split migration"
                              : " Dynamic phase-split migration";
        test_case.prompt = "Task: read the ledger and return one minified JSON object.";
        test_case.default_metadata_path = metadata_path;
        test_case.decode_steps = 3;
        test_case.max_seq_len = kPhaseSplitLongDecodeMaxSeqLen;
        test_case.prefix_restore_prompt_token_limit = 640;
        test_case.moe_rebalance = movementFriendlyRebalanceConfig(mode);
        test_case.env_overrides = {
            {"LLAMINAR_GPU_GRAPHS", "1"},
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
            {"LLAMINAR_MOE_DEVICE_REBALANCE_NO_WORK_BACKOFF_PERIODS", "0"},
            {"LLAMINAR_MOE_GPU_DIRECT_TRANSFER_WAVE_EXPERTS", "32"},
            {"LLAMINAR_MOE_DEVICE_REBALANCE_COMPACT_PAYLOAD_SLOTS", "32"},
        };
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
     * hot-only partial-hit cells at long ledger metadata and then caps the
     * prompt at 640 tokens, giving the cache a non-terminal 256-token block and
     * a meaningful suffix without turning the hot-only canary itself into a
     * long-context test.
     *
     * @param test_case Case object to mutate.
     * @param metadata_path Long-decode PyTorch metadata file to load.
     */
    void configureHotOnlyPartialPrefixProbe(
        MoEPrefixRestoreParityCase &test_case,
        const std::string &metadata_path)
    {
        test_case.name += " long partial-prefix restore";
        test_case.prompt = kLongLedgerPromptHeader;
        test_case.default_metadata_path = metadata_path;
        test_case.max_seq_len = kPhaseSplitLongDecodeMaxSeqLen;
        test_case.prefix_restore_prompt_token_limit = 640;
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
        configureHotOnlyPartialPrefixProbe(
            test_case,
            "pytorch_qwen36_moe_expert_overlay_cuda2_dynamic_phase_split_long_decode_snapshots/metadata.txt");
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
            MoERebalanceRuntimeMode::Dynamic,
            "pytorch_qwen36_moe_expert_overlay_cuda2_dynamic_phase_split_long_decode_snapshots/metadata.txt");
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
            MoERebalanceRuntimeMode::LLEP,
            "pytorch_qwen36_moe_expert_overlay_cuda2_llep_phase_split_long_decode_snapshots/metadata.txt");
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
            MoERebalanceRuntimeMode::Dynamic,
            "pytorch_qwen36_moe_expert_overlay_rocm2_dynamic_phase_split_long_decode_snapshots/metadata.txt");
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
        configureHotOnlyPartialPrefixProbe(
            test_case,
            "pytorch_qwen36_moe_expert_overlay_rocm2_dynamic_phase_split_long_decode_snapshots/metadata.txt");
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
            MoERebalanceRuntimeMode::LLEP,
            "pytorch_qwen36_moe_expert_overlay_rocm2_llep_phase_split_long_decode_snapshots/metadata.txt");
        return test_case;
    }
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, MTPGreedyMatchesBaselineTokens_CUDA2TPHotOnly)
{
    runMoEMTPParity(cudaOnlyExpertOverlayCase(), false);
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

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, MTPGreedyMatchesBaselineTokens_ROCm2TPHotOnly)
{
    runMoEMTPParity(rocmOnlyExpertOverlayCase(), false);
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

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, MTPGreedyMatchesBaselineTokens_ROCm2TPHot_CPU2LocalTPCold)
{
    runMoEMTPParity(expertOverlayCase(), false);
}

TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PrefixCacheMTPRestore_ROCm2TPHot_CPU2LocalTPCold)
{
    runMoEMTPParity(expertOverlayCase(), true);
}

/**
 * @brief Guards the partial-prefix fixtures against accidentally using tiny metadata.
 *
 * This test is intentionally source-level and model-free.  The expensive
 * partial-hit cells validate runtime behavior; this one catches the fixture
 * regression that made the matrix fail before any GPU work started.
 */
TEST(Qwen36MoEExpertOverlayPrefixMTPParity, PartialPrefixFixturesUseLongMetadata)
{
    const auto cuda_case = cudaOnlyPartialPrefixCase();
    const auto rocm_case = rocmOnlyPartialPrefixCase();

    for (const auto *test_case : {&cuda_case, &rocm_case})
    {
        EXPECT_EQ(test_case->prompt, kLongLedgerPromptHeader);
        EXPECT_EQ(test_case->max_seq_len, kPhaseSplitLongDecodeMaxSeqLen);
        EXPECT_GT(test_case->prefix_restore_prompt_token_limit, 256);
        EXPECT_NE(
            test_case->default_metadata_path.find("long_decode_snapshots"),
            std::string::npos);
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

int main(int argc, char **argv)
{
    std::unique_ptr<ScopedParityProcessLock> parity_lock;
    try
    {
        parity_lock = std::make_unique<ScopedParityProcessLock>(
            "/tmp/llaminar_qwen36_moe_expert_overlay_parity.lock");
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
