#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        namespace fs = std::filesystem;

        fs::path findRepoRoot()
        {
            std::vector<fs::path> starts;
            starts.push_back(fs::current_path());
            starts.push_back(fs::path(__FILE__));

            for (auto start : starts)
            {
                if (fs::is_regular_file(start))
                    start = start.parent_path();

                for (fs::path candidate = start; !candidate.empty(); candidate = candidate.parent_path())
                {
                    if (fs::exists(candidate / "src/v2/execution/moe/MoEOverlaySparseCollective.h") &&
                        fs::exists(candidate / "tests/v2/CMakeLists.txt"))
                    {
                        return candidate;
                    }

                    if (candidate == candidate.root_path())
                        break;
                }
            }

            return fs::current_path();
        }

        std::string readFile(const fs::path &path)
        {
            std::ifstream input(path);
            if (!input)
                return {};

            std::ostringstream buffer;
            buffer << input.rdbuf();
            return buffer.str();
        }

    } // namespace

    TEST(Test__MoEOverlayLegacyRuntimeQuarantined, LegacyRuntimeFilesAreDeleted)
    {
        const fs::path root = findRepoRoot();
        const std::vector<fs::path> deleted_paths = {
            "src/v2/execution/compute_stages/stages/MoEOverlayDomainRuntimeStage.h",
            "src/v2/execution/compute_stages/stages/MoEOverlayDomainRuntimeStage.cpp",
            "src/v2/execution/compute_stages/stages/MoEExpertOverlayLocalTPStage.h",
            "src/v2/execution/compute_stages/stages/MoEExpertOverlayLocalTPStage.cpp",
            "src/v2/execution/compute_stages/stages/MoEExpertOverlayCPUFallbackStage.h",
            "src/v2/execution/compute_stages/stages/MoEExpertOverlayCPUFallbackStage.cpp",
            "src/v2/execution/moe/IOverlayDomainRuntime.h",
            "src/v2/execution/moe/MoEOverlayDomainRuntime.h",
            "src/v2/execution/moe/MoEOverlayDomainRuntime.cpp",
            "src/v2/execution/moe/MoEExpertOverlayLocalTPExecutor.h",
            "src/v2/execution/moe/MoEExpertOverlayLocalTPExecutor.cpp",
            "src/v2/execution/moe/MoEOverlayCPUFallbackParticipantRunner.h",
            "src/v2/execution/moe/MoEOverlayCPUFallbackParticipantRunner.cpp",
            "src/v2/execution/moe/MoEOverlayDispatchCollective.h",
            "src/v2/execution/moe/MoEOverlayDispatchCollective.cpp",
            "src/v2/execution/moe/MoEOverlayMPIDispatchBackend.h",
            "src/v2/execution/moe/MoEOverlayMPIDispatchBackend.cpp",
            "src/v2/execution/moe/MoEExpertOverlayCPUFallback.h",
            "src/v2/execution/moe/MoEExpertOverlayCPUFallback.cpp",
        };

        for (const auto &relative_path : deleted_paths)
            EXPECT_FALSE(fs::exists(root / relative_path)) << relative_path;
    }

    TEST(Test__MoEOverlayLegacyRuntimeQuarantined, ProductionCodeDoesNotReferenceDeletedRuntime)
    {
        const fs::path root = findRepoRoot();
        const std::vector<fs::path> production_paths = {
            "src/v2/CMakeLists.txt",
            "src/v2/models/GraphTypes.h",
            "src/v2/models/qwen35moe/Qwen35MoEGraph.cpp",
            "src/v2/execution/factory/InferenceRunnerFactory.h",
            "src/v2/execution/factory/InferenceRunnerFactory.cpp",
            "src/v2/execution/runner/OrchestrationRunner.h",
            "src/v2/execution/runner/OrchestrationRunner.cpp",
            "src/v2/execution/local_execution/orchestrators/RankOrchestrator.h",
            "src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp",
        };
        const std::vector<std::string> deleted_tokens = {
            "LLAMINAR_MOE_LEGACY_OVERLAY_DOMAIN_RUNTIME",
            "legacyOverlayDomainRuntimeEnabled",
            "IOverlayDomainRuntime",
            "overlay_domain_runtime",
            "MoEOverlayDomainRuntimeStage",
            "MoEOverlayDomainRuntime",
            "MoEExpertOverlayLocalTPStage",
            "MoEExpertOverlayLocalTPExecutor",
            "MoEExpertOverlayCPUFallbackStage",
            "MoEExpertOverlayCPUFallback",
            "MoEOverlayCPUFallbackParticipantRunner",
            "MoEOverlayDispatchCollective",
            "MoEOverlayMPIDispatchBackend",
        };

        std::vector<std::string> failures;
        for (const auto &relative_path : production_paths)
        {
            const fs::path path = root / relative_path;
            ASSERT_TRUE(fs::exists(path)) << path;
            const std::string source = readFile(path);
            ASSERT_FALSE(source.empty()) << path;

            for (const auto &token : deleted_tokens)
            {
                if (source.find(token) != std::string::npos)
                    failures.push_back(relative_path.generic_string() + " contains deleted token " + token);
            }
        }

        EXPECT_TRUE(failures.empty()) << [&]
        {
            std::ostringstream out;
            for (const auto &failure : failures)
                out << failure << '\n';
            return out.str();
        }();
    }

} // namespace llaminar2::test
