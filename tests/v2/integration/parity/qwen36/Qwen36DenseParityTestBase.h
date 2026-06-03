#pragma once

#include <gtest/gtest.h>
#include <mpi.h>

#include "backends/ComputeBackend.h"
#include "backends/GlobalDeviceAddress.h"
#include "config/OrchestrationConfig.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "utils/Sampler.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::test::parity::qwen36
{
    enum class DensePrefixParityTopology
    {
        SingleDevice,
        LocalTP,
        LocalPP,
        NodeLocalTP,
    };

    enum class PrefixRestoreParityMode
    {
        FullHit,
        PartialHit,
    };

    struct DensePrefixRestoreParityCase
    {
        std::string name;
        DensePrefixParityTopology topology = DensePrefixParityTopology::SingleDevice;
        std::vector<GlobalDeviceAddress> devices;
        std::vector<std::string> model_envs;
        std::string default_model_path;
        std::vector<std::string> metadata_envs;
        std::string default_metadata_path;
        std::string prompt = "The quick brown fox jumps over the lazy dog";
        std::string kv_cache_precision = "auto";
        int decode_steps = 3;
        int max_seq_len = 96;
        int main_layers = 64;
        int mpi_ranks = 1;
        int required_cuda_devices = 0;
        int required_rocm_devices = 0;
    };

    inline std::string shellQuote(const std::string &value)
    {
        std::string quoted = "'";
        for (char ch : value)
        {
            if (ch == '\'')
            {
                quoted += "'\\''";
            }
            else
            {
                quoted += ch;
            }
        }
        quoted += "'";
        return quoted;
    }

    inline std::string firstEnvOrDefault(
        const std::vector<std::string> &names,
        const std::string &fallback)
    {
        for (const auto &name : names)
        {
            const char *value = std::getenv(name.c_str());
            if (value && *value)
            {
                return value;
            }
        }
        return fallback;
    }

    inline int mpiWorldSize()
    {
        int world_size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        return world_size;
    }

    inline std::vector<int32_t> readTokenListFromMetadata(
        const std::filesystem::path &metadata_path,
        const std::string &key)
    {
        std::ifstream file(metadata_path);
        if (!file.is_open())
        {
            return {};
        }

        std::string line;
        const std::string prefix = key + ":";
        while (std::getline(file, line))
        {
            if (line.rfind(prefix, 0) != 0)
            {
                continue;
            }

            std::string tokens = line.substr(prefix.size());
            const size_t start = tokens.find_first_not_of(" \t");
            if (start != std::string::npos)
            {
                tokens = tokens.substr(start);
            }

            std::vector<int32_t> result;
            std::stringstream ss(tokens);
            std::string token;
            while (std::getline(ss, token, ','))
            {
                const size_t token_start = token.find_first_not_of(" \t");
                const size_t token_end = token.find_last_not_of(" \t");
                if (token_start == std::string::npos || token_end == std::string::npos)
                {
                    continue;
                }
                result.push_back(std::stoi(token.substr(token_start, token_end - token_start + 1)));
            }
            return result;
        }

        return {};
    }

    inline bool metadataLooksUsable(
        const std::filesystem::path &metadata_path,
        int required_decode_steps)
    {
        const auto token_ids = readTokenListFromMetadata(metadata_path, "token_ids");
        const auto decode_tokens = readTokenListFromMetadata(metadata_path, "decode_tokens");
        return !token_ids.empty() &&
               decode_tokens.size() >= static_cast<size_t>(required_decode_steps);
    }

    inline bool regenerateQwen36Metadata(
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
            "python3 python/reference/generate_qwen35_pipeline_snapshots.py";
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
                *output = "failed to spawn python metadata generator";
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

    inline void ensurePyTorchMetadata(
        const DensePrefixRestoreParityCase &test_case,
        const std::string &model_path,
        const std::filesystem::path &metadata_path)
    {
        if (metadataLooksUsable(metadata_path, test_case.decode_steps))
        {
            return;
        }

        std::string output;
        ASSERT_TRUE(regenerateQwen36Metadata(
            model_path,
            metadata_path,
            test_case.prompt,
            test_case.decode_steps,
            &output))
            << test_case.name << " failed to regenerate PyTorch metadata at "
            << metadata_path << "\n"
            << output;

        ASSERT_TRUE(metadataLooksUsable(metadata_path, test_case.decode_steps))
            << test_case.name << " regenerated metadata is incomplete at "
            << metadata_path << "\n"
            << output;
    }

    inline std::optional<std::string> densePrefixParitySkipReason(
        const DensePrefixRestoreParityCase &test_case)
    {
        const int world_size = mpiWorldSize();
        if (test_case.topology == DensePrefixParityTopology::NodeLocalTP)
        {
            if (world_size != test_case.mpi_ranks)
            {
                std::ostringstream oss;
                oss << test_case.name << " requires exactly "
                    << test_case.mpi_ranks << " MPI ranks (got "
                    << world_size << ")";
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

        return std::nullopt;
    }

    inline std::vector<PPStageDefinition> splitStages(
        int total_layers,
        const std::vector<GlobalDeviceAddress> &devices)
    {
        std::vector<PPStageDefinition> stages;
        const int stage_count = static_cast<int>(devices.size());
        if (stage_count <= 0 || total_layers <= 0)
        {
            return stages;
        }

        int first = 0;
        for (int stage = 0; stage < stage_count; ++stage)
        {
            const int next = ((stage + 1) * total_layers) / stage_count;
            const int last = std::max(first, next) - 1;
            stages.push_back(PPStageDefinition{
                stage,
                "stage" + std::to_string(stage),
                first,
                last,
            });
            first = last + 1;
        }
        return stages;
    }

    inline OrchestrationConfig makeDensePrefixRestoreConfig(
        const DensePrefixRestoreParityCase &test_case,
        const std::string &model_path,
        bool enable_prefix_cache,
        int block_size,
        bool enable_mtp = false,
        int mtp_draft_tokens = 1)
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
        config.prefix_cache.ram_budget_bytes = 1024ull * 1024ull * 1024ull;
        config.mtp.enabled = enable_mtp;
        config.mtp.draft_tokens = mtp_draft_tokens;

        switch (test_case.topology)
        {
        case DensePrefixParityTopology::SingleDevice:
            config.tp_degree = 1;
            config.pp_degree = 1;
            config.device_for_this_rank = test_case.devices.empty()
                                              ? GlobalDeviceAddress::cpu()
                                              : test_case.devices.front();
            break;

        case DensePrefixParityTopology::LocalTP:
            config.tp_degree = static_cast<int>(test_case.devices.size());
            config.tp_scope = TPScope::LOCAL;
            config.tp_devices = test_case.devices;
            config.pp_degree = 1;
            config.default_backend = CollectiveBackendType::RCCL;
            break;

        case DensePrefixParityTopology::LocalPP:
            config.tp_degree = 1;
            config.pp_degree = static_cast<int>(test_case.devices.size());
            config.pp_split = PPSplitMode::MANUAL;
            config.domain_definitions.clear();
            config.pp_stage_definitions = splitStages(test_case.main_layers, test_case.devices);
            for (size_t i = 0; i < test_case.devices.size(); ++i)
            {
                DomainDefinition domain;
                domain.name = "stage" + std::to_string(i);
                domain.devices = {test_case.devices[i]};
                domain.scope = TPScope::LOCAL;
                domain.owner_rank = 0;
                domain.backend = CollectiveBackendType::AUTO;
                config.domain_definitions.push_back(std::move(domain));
            }
            break;

        case DensePrefixParityTopology::NodeLocalTP:
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
        }

        return config;
    }

    inline DensePrefixRestoreParityCase qwen36DensePrefixParityCase(
        const std::string &name,
        DensePrefixParityTopology topology)
    {
        DensePrefixRestoreParityCase test_case{
            .name = name,
            .topology = topology,
            .model_envs = {
                "LLAMINAR_QWEN36_DENSE_MODEL",
                "LLAMINAR_PARITY_DENSE_MODEL",
            },
            .default_model_path = "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf",
            .metadata_envs = {
                "LLAMINAR_QWEN36_PARITY_METADATA",
                "LLAMINAR_PARITY_DENSE_METADATA",
            },
            .default_metadata_path = "pytorch_qwen36_dense_snapshots/metadata.txt",
            .prompt = "The quick brown fox jumps over the lazy dog",
            .kv_cache_precision = "auto",
            .decode_steps = 3,
            .max_seq_len = 96,
            .main_layers = 64,
        };

        switch (topology)
        {
        case DensePrefixParityTopology::SingleDevice:
            test_case.devices = {GlobalDeviceAddress::rocm(0)};
            test_case.required_rocm_devices = 1;
            break;
        case DensePrefixParityTopology::LocalTP:
            test_case.devices = {
                GlobalDeviceAddress::rocm(0),
                GlobalDeviceAddress::rocm(1),
            };
            test_case.required_rocm_devices = 2;
            break;
        case DensePrefixParityTopology::LocalPP:
            test_case.devices = {
                GlobalDeviceAddress::rocm(0),
                GlobalDeviceAddress::rocm(1),
            };
            test_case.required_rocm_devices = 2;
            break;
        case DensePrefixParityTopology::NodeLocalTP:
            test_case.devices = {
                GlobalDeviceAddress::cpu(0),
                GlobalDeviceAddress::cpu(1),
            };
            test_case.mpi_ranks = 2;
            break;
        }

        return test_case;
    }

    inline void loadReferenceInputs(
        const DensePrefixRestoreParityCase &test_case,
        std::string *model_path,
        std::vector<int32_t> *prompt_tokens,
        std::vector<int32_t> *expected_tokens)
    {
        if (auto skip_reason = densePrefixParitySkipReason(test_case))
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
        ensurePyTorchMetadata(test_case, *model_path, metadata_path);

        *prompt_tokens = readTokenListFromMetadata(metadata_path, "token_ids");
        const auto pytorch_decode_tokens = readTokenListFromMetadata(metadata_path, "decode_tokens");
        ASSERT_FALSE(prompt_tokens->empty());
        ASSERT_GE(pytorch_decode_tokens.size(), static_cast<size_t>(test_case.decode_steps));

        expected_tokens->assign(
            pytorch_decode_tokens.begin(),
            pytorch_decode_tokens.begin() + test_case.decode_steps);
    }

    inline void runDensePrefixRestoreParity(
        const DensePrefixRestoreParityCase &test_case,
        PrefixRestoreParityMode mode)
    {
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);

        const int block_size = mode == PrefixRestoreParityMode::FullHit
                                   ? static_cast<int>(prompt_tokens.size())
                                   : 4;
        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto baseline = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(test_case, model_path, false, block_size));
        ASSERT_NE(baseline, nullptr);
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        auto baseline_result = baseline->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto baseline_snapshot = baseline->prefixStateProbe();
        baseline->shutdown();

        ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
        ASSERT_EQ(baseline_result.tokens.size(), expected_tokens.size());
        EXPECT_EQ(baseline_result.tokens, expected_tokens);
        EXPECT_EQ(baseline_snapshot.prefix_cache_hits, 0u);

        auto cached = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(test_case, model_path, true, block_size));
        ASSERT_NE(cached, nullptr);
        ASSERT_TRUE(cached->initialize()) << cached->lastError();

        std::vector<int32_t> first_prompt = prompt_tokens;
        if (mode == PrefixRestoreParityMode::PartialHit)
        {
            ASSERT_GT(prompt_tokens.size(), 4u);
            first_prompt.assign(prompt_tokens.begin(), prompt_tokens.begin() + 4);
        }

        auto first = cached->generate(first_prompt, test_case.decode_steps, greedy);
        const auto after_first = cached->prefixStateProbe();
        ASSERT_TRUE(first.error.empty()) << first.error;
        EXPECT_TRUE(after_first.prefix_cache_ready);
        EXPECT_GE(after_first.prefix_cache_inserts, 1u);
        if (mode == PrefixRestoreParityMode::FullHit)
        {
            ASSERT_EQ(first.tokens.size(), expected_tokens.size());
            EXPECT_EQ(first.tokens, expected_tokens);
        }

        auto second = cached->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto after_second = cached->prefixStateProbe();
        cached->shutdown();

        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_EQ(second.tokens.size(), expected_tokens.size());
        EXPECT_EQ(second.tokens, expected_tokens);
        EXPECT_EQ(second.tokens, baseline_result.tokens);
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

    inline void runDenseSplitPrefillParity(
        const DensePrefixRestoreParityCase &test_case,
        int split_tokens)
    {
        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);
        ASSERT_GT(prompt_tokens.size(), static_cast<size_t>(split_tokens));

        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto baseline = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(test_case, model_path, false, split_tokens));
        ASSERT_NE(baseline, nullptr);
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        auto baseline_result = baseline->generate(prompt_tokens, test_case.decode_steps, greedy);
        baseline->shutdown();

        ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
        ASSERT_EQ(baseline_result.tokens, expected_tokens);

        auto split = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(test_case, model_path, false, split_tokens));
        ASSERT_NE(split, nullptr);
        ASSERT_TRUE(split->initialize()) << split->lastError();
        split->setSamplingParams(greedy);

        const std::vector<int32_t> first_prompt(
            prompt_tokens.begin(),
            prompt_tokens.begin() + split_tokens);
        const std::vector<int32_t> suffix(
            prompt_tokens.begin() + split_tokens,
            prompt_tokens.end());

        ASSERT_TRUE(split->prefill(first_prompt)) << split->lastError();
        ASSERT_TRUE(split->prefill(suffix)) << split->lastError();
        EXPECT_EQ(split->currentPosition(), static_cast<int>(prompt_tokens.size()));
        EXPECT_TRUE(split->prefixStateProbe().prefill_logits_ready);

        std::vector<int32_t> split_tokens_out;
        for (int i = 0; i < test_case.decode_steps; ++i)
        {
            GenerationResult step = split->decodeStep();
            ASSERT_TRUE(step.error.empty()) << step.error;
            ASSERT_EQ(step.tokens.size(), 1u);
            split_tokens_out.push_back(step.tokens.front());
        }
        split->shutdown();

        EXPECT_EQ(split_tokens_out, expected_tokens);
        EXPECT_EQ(split_tokens_out, baseline_result.tokens);
    }

    inline void runDenseMTPParity(
        const DensePrefixRestoreParityCase &test_case,
        bool enable_prefix_cache,
        int mtp_draft_tokens = 1)
    {
        ASSERT_GE(mtp_draft_tokens, 1);
        ASSERT_LE(mtp_draft_tokens, 3);

        std::string model_path;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> expected_tokens;
        loadReferenceInputs(test_case, &model_path, &prompt_tokens, &expected_tokens);

        const int block_size = enable_prefix_cache
                                   ? static_cast<int>(prompt_tokens.size())
                                   : 2;
        auto factory = createOrchestrationRunnerFactory();
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        auto baseline = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(test_case, model_path, false, block_size, false));
        ASSERT_NE(baseline, nullptr);
        ASSERT_TRUE(baseline->initialize()) << baseline->lastError();
        auto baseline_result = baseline->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto baseline_snapshot = baseline->prefixStateProbe();
        baseline->shutdown();

        ASSERT_TRUE(baseline_result.error.empty()) << baseline_result.error;
        ASSERT_EQ(baseline_result.tokens.size(), expected_tokens.size());
        EXPECT_EQ(baseline_result.tokens, expected_tokens);
        EXPECT_EQ(baseline_snapshot.prefix_cache_hits, 0u);
        EXPECT_EQ(baseline_snapshot.mtp_draft_steps, 0u);

        auto mtp = factory->createFromOrchestrationConfig(
            makeDensePrefixRestoreConfig(
                test_case,
                model_path,
                enable_prefix_cache,
                block_size,
                true,
                mtp_draft_tokens));
        ASSERT_NE(mtp, nullptr);
        ASSERT_TRUE(mtp->initialize()) << mtp->lastError();

        auto first = mtp->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto after_first = mtp->prefixStateProbe();
        ASSERT_TRUE(first.error.empty()) << first.error;
        ASSERT_EQ(first.tokens.size(), expected_tokens.size());
        EXPECT_EQ(first.tokens, expected_tokens);
        EXPECT_EQ(first.tokens, baseline_result.tokens);
        EXPECT_FALSE(after_first.mtp_bypassed) << after_first.mtp_bypass_reason;
        EXPECT_GE(after_first.mtp_draft_steps, static_cast<uint64_t>(mtp_draft_tokens));
        EXPECT_GE(after_first.mtp_verifier_runs, 1u);
        EXPECT_GE(after_first.mtp_verifier_token_count, static_cast<uint64_t>(mtp_draft_tokens + 1));

        if (!enable_prefix_cache)
        {
            mtp->shutdown();
            return;
        }

        EXPECT_TRUE(after_first.prefix_cache_ready);
        EXPECT_GE(after_first.prefix_cache_inserts, 1u);
        EXPECT_GT(after_first.prefix_cache_mtp_state_bytes, 0u);

        auto second = mtp->generate(prompt_tokens, test_case.decode_steps, greedy);
        const auto after_second = mtp->prefixStateProbe();
        mtp->shutdown();

        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_EQ(second.tokens.size(), expected_tokens.size());
        EXPECT_EQ(second.tokens, expected_tokens);
        EXPECT_EQ(second.tokens, baseline_result.tokens);
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

} // namespace llaminar2::test::parity::qwen36
