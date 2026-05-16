#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        std::string readSourceFile(const std::string &path)
        {
            std::ifstream file(path);
            if (!file)
                return {};
            std::ostringstream contents;
            contents << file.rdbuf();
            return contents.str();
        }

        size_t countOccurrences(const std::string &haystack, const std::string &needle)
        {
            size_t count = 0;
            size_t pos = 0;
            while ((pos = haystack.find(needle, pos)) != std::string::npos)
            {
                ++count;
                pos += needle.size();
            }
            return count;
        }

        std::string requireSource(const std::string &path)
        {
            std::string source = readSourceFile(path);
            EXPECT_FALSE(source.empty()) << "failed to read " << path;
            return source;
        }

    } // namespace

    TEST(Test__MoEOverlayNoCPUFallbackSidecar, ProductionConstructionDoesNotReferenceCPUFallbackRunner)
    {
        const std::vector<std::string> production_sources = {
            "src/v2/execution/runner/OrchestrationRunner.cpp",
            "src/v2/execution/factory/InferenceRunnerFactory.cpp",
            "src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp",
        };

        for (const auto &path : production_sources)
        {
            const std::string source = requireSource(path);
            ASSERT_FALSE(source.empty());
            EXPECT_EQ(source.find("MoEOverlayCPUFallbackParticipantRunner"), std::string::npos) << path;
            EXPECT_EQ(source.find("createMoEOverlayCPUFallbackParticipantRunner"), std::string::npos) << path;
            EXPECT_EQ(source.find("runDispatchEndpointPump"), std::string::npos) << path;
        }
    }

    TEST(Test__MoEOverlayNoCPUFallbackSidecar, EndpointBackendConstructionIsRemoved)
    {
        const std::string source = requireSource("src/v2/execution/runner/OrchestrationRunner.cpp");
        ASSERT_FALSE(source.empty());

        EXPECT_EQ(source.find("ensureMoEOverlayDispatchBackend"), std::string::npos);
        EXPECT_EQ(source.find("beginMoEOverlayForwardDispatch"), std::string::npos);
        EXPECT_EQ(source.find("signalMoEOverlayForwardDone"), std::string::npos);
        EXPECT_EQ(source.find("signalMoEOverlayForwardCancel"), std::string::npos);
        EXPECT_EQ(source.find("std::make_shared<MoEOverlayMPIDispatchBackend>"), std::string::npos);
        EXPECT_EQ(source.find("legacyOverlayDomainRuntimeEnabled"), std::string::npos);
        EXPECT_EQ(source.find("LLAMINAR_MOE_LEGACY_OVERLAY_DOMAIN_RUNTIME"), std::string::npos);
        EXPECT_EQ(countOccurrences(source, "std::getenv("), 0u);
        EXPECT_EQ(source.find("LLAMINAR_MOE_CPU_FALLBACK"), std::string::npos);
        EXPECT_EQ(source.find("LLAMINAR_MOE_OVERLAY_ENDPOINT"), std::string::npos);
        EXPECT_EQ(source.find("LLAMINAR_MOE_GRAPH_NATIVE_ENDPOINT"), std::string::npos);
    }

} // namespace llaminar2::test