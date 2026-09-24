/**
 * @file Test__PipelineTopologyQueries.cpp
 * @brief Device-free contracts for declarative pipeline topology queries.
 *
 * These fixtures prove layer intervals, domain membership, transfer selection
 * and TP degree queries. They neither construct DGO nor certify a running PP
 * topology. Real rank/global orchestration is covered by integration tests.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <memory>

#include "config/PipelineConfig.h"
#include "backends/GlobalDeviceAddress.h"
#include "backends/DeviceId.h"

namespace llaminar2::test
{

    // =========================================================================
    // Test Fixture
    // =========================================================================

    /** @brief Small declaration fixtures; no runtime contexts are constructed. */
    class Test__PipelineTopologyQueries : public ::testing::Test
    {
    protected:
        /**
         * @brief Create a PipelineConfig with no internal TP (all degree-1 domains)
         */
        static PipelineConfig makeSimplePPConfig()
        {
            // PP(cuda:0, rocm:0, cpu) - no TP, should use flat LocalPPConfig
            PipelineConfig config;
            config.total_layers = 24;
            config.tp_domains = {
                {"cuda_stage", {DeviceId::cuda(0)}, CollectiveBackendType::AUTO},
                {"rocm_stage", {DeviceId::rocm(0)}, CollectiveBackendType::AUTO},
                {"cpu_stage", {DeviceId::cpu()}, CollectiveBackendType::AUTO}
            };
            config.pp_stages = {
                PPStageConfig::firstStage(0, "cuda_stage", 0, 8),
                PPStageConfig::middleStage(1, "rocm_stage", 8, 16),
                PPStageConfig::lastStage(2, "cpu_stage", 16, 24)
            };
            return config;
        }

        /**
         * @brief Create a PipelineConfig with internal TP (some degree > 1 domains)
         */
        static PipelineConfig makeTPInPPConfig()
        {
            // PP(TP(rocm:0, rocm:1), cuda:0, cpu) - has TP, should use HierarchicalPPConfig
            PipelineConfig config;
            config.total_layers = 24;
            config.tp_domains = {
                {"rocm_tp", {DeviceId::rocm(0), DeviceId::rocm(1)}, CollectiveBackendType::RCCL},
                {"cuda_single", {DeviceId::cuda(0)}, CollectiveBackendType::AUTO},
                {"cpu_single", {DeviceId::cpu()}, CollectiveBackendType::AUTO}
            };
            config.pp_stages = {
                PPStageConfig::firstStage(0, "rocm_tp", 0, 14),
                PPStageConfig::middleStage(1, "cuda_single", 14, 22),
                PPStageConfig::lastStage(2, "cpu_single", 22, 24)
            };
            return config;
        }

        /**
         * @brief Create a PipelineConfig with multiple TP domains
         */
        static PipelineConfig makeMultiTPConfig()
        {
            // PP(TP(rocm:0, rocm:1), TP(cuda:0, cuda:1), cpu)
            PipelineConfig config;
            config.total_layers = 28;
            config.tp_domains = {
                {"rocm_tp", {DeviceId::rocm(0), DeviceId::rocm(1)}, CollectiveBackendType::RCCL},
                {"cuda_tp", {DeviceId::cuda(0), DeviceId::cuda(1)}, CollectiveBackendType::NCCL},
                {"cpu_single", {DeviceId::cpu()}, CollectiveBackendType::AUTO}
            };
            config.pp_stages = {
                PPStageConfig::firstStage(0, "rocm_tp", 0, 12),
                PPStageConfig::middleStage(1, "cuda_tp", 12, 24),
                PPStageConfig::lastStage(2, "cpu_single", 24, 28)
            };
            return config;
        }
    };

    // =========================================================================
    // PipelineConfig Query Tests (hasTP, hasPP, etc.)
    // =========================================================================

    TEST_F(Test__PipelineTopologyQueries, SimpleConfig_HasPPButNoTP)
    {
        auto config = makeSimplePPConfig();

        EXPECT_TRUE(config.hasPP());     // 3 stages
        EXPECT_FALSE(config.hasTP());    // All domains have degree 1
        EXPECT_EQ(config.numStages(), 3);
        EXPECT_EQ(config.maxTPDegree(), 1);
    }

    TEST_F(Test__PipelineTopologyQueries, TPInPPConfig_HasBothPPAndTP)
    {
        auto config = makeTPInPPConfig();

        EXPECT_TRUE(config.hasPP());     // 3 stages
        EXPECT_TRUE(config.hasTP());     // First domain has degree 2
        EXPECT_EQ(config.numStages(), 3);
        EXPECT_EQ(config.maxTPDegree(), 2);
    }

    TEST_F(Test__PipelineTopologyQueries, MultiTPConfig_HasBothPPAndMultipleTP)
    {
        auto config = makeMultiTPConfig();

        EXPECT_TRUE(config.hasPP());
        EXPECT_TRUE(config.hasTP());
        EXPECT_EQ(config.numStages(), 3);
        EXPECT_EQ(config.maxTPDegree(), 2);  // Both TP domains have degree 2
    }

    // =========================================================================
    // Domain Lookup Tests
    // =========================================================================

    TEST_F(Test__PipelineTopologyQueries, TPInPPConfig_DomainDegrees)
    {
        auto config = makeTPInPPConfig();

        const auto* rocm_domain = config.getDomain("rocm_tp");
        ASSERT_NE(rocm_domain, nullptr);
        EXPECT_EQ(rocm_domain->degree(), 2);

        const auto* cuda_domain = config.getDomain("cuda_single");
        ASSERT_NE(cuda_domain, nullptr);
        EXPECT_EQ(cuda_domain->degree(), 1);

        const auto* cpu_domain = config.getDomain("cpu_single");
        ASSERT_NE(cpu_domain, nullptr);
        EXPECT_EQ(cpu_domain->degree(), 1);
    }

    TEST_F(Test__PipelineTopologyQueries, TPInPPConfig_GetDomainForStage)
    {
        auto config = makeTPInPPConfig();

        // Stage 0 → rocm_tp domain (degree 2)
        const auto* stage0_domain = config.getDomainForStage(0);
        ASSERT_NE(stage0_domain, nullptr);
        EXPECT_EQ(stage0_domain->name, "rocm_tp");
        EXPECT_EQ(stage0_domain->degree(), 2);

        // Stage 1 → cuda_single domain (degree 1)
        const auto* stage1_domain = config.getDomainForStage(1);
        ASSERT_NE(stage1_domain, nullptr);
        EXPECT_EQ(stage1_domain->name, "cuda_single");
        EXPECT_EQ(stage1_domain->degree(), 1);

        // Stage 2 → cpu_single domain (degree 1)
        const auto* stage2_domain = config.getDomainForStage(2);
        ASSERT_NE(stage2_domain, nullptr);
        EXPECT_EQ(stage2_domain->name, "cpu_single");
        EXPECT_EQ(stage2_domain->degree(), 1);
    }

    // =========================================================================
    // Layer Mapping Tests
    // =========================================================================

    TEST_F(Test__PipelineTopologyQueries, TPInPPConfig_LayerToStageMapping)
    {
        auto config = makeTPInPPConfig();

        // Layers 0-13 → Stage 0 (rocm_tp)
        EXPECT_EQ(config.getStageIdForLayer(0), 0);
        EXPECT_EQ(config.getStageIdForLayer(10), 0);
        EXPECT_EQ(config.getStageIdForLayer(13), 0);

        // Layers 14-21 → Stage 1 (cuda_single)
        EXPECT_EQ(config.getStageIdForLayer(14), 1);
        EXPECT_EQ(config.getStageIdForLayer(18), 1);
        EXPECT_EQ(config.getStageIdForLayer(21), 1);

        // Layers 22-23 → Stage 2 (cpu_single)
        EXPECT_EQ(config.getStageIdForLayer(22), 2);
        EXPECT_EQ(config.getStageIdForLayer(23), 2);
    }

    TEST_F(Test__PipelineTopologyQueries, TPInPPConfig_PPTransferNeeded)
    {
        auto config = makeTPInPPConfig();

        // Same stage → no transfer
        EXPECT_FALSE(config.needsPPTransfer(5, 10));   // Both in stage 0
        EXPECT_FALSE(config.needsPPTransfer(14, 18));  // Both in stage 1

        // Different stages → transfer needed
        EXPECT_TRUE(config.needsPPTransfer(13, 14));   // Stage 0 → 1
        EXPECT_TRUE(config.needsPPTransfer(21, 22));   // Stage 1 → 2
        EXPECT_TRUE(config.needsPPTransfer(0, 23));    // Stage 0 → 2
    }

    // =========================================================================
    // Validation Tests
    // =========================================================================

    TEST_F(Test__PipelineTopologyQueries, Config_Validation)
    {
        auto config = makeTPInPPConfig();

        std::string error_msg;
        EXPECT_TRUE(config.validate(&error_msg)) << "Validation failed: " << error_msg;
    }

    TEST_F(Test__PipelineTopologyQueries, InvalidConfig_MissingDomain)
    {
        PipelineConfig config;
        config.total_layers = 24;
        config.tp_domains = {
            {"existing_domain", {DeviceId::cuda(0)}, CollectiveBackendType::AUTO}
        };
        config.pp_stages = {
            PPStageConfig::firstStage(0, "nonexistent_domain", 0, 24)  // References missing domain
        };

        std::string error_msg;
        EXPECT_FALSE(config.validate(&error_msg));
        EXPECT_FALSE(error_msg.empty());
    }

    TEST_F(Test__PipelineTopologyQueries, InvalidConfig_LayerGap)
    {
        PipelineConfig config;
        config.total_layers = 24;
        config.tp_domains = {
            {"domain_a", {DeviceId::cuda(0)}, CollectiveBackendType::AUTO},
            {"domain_b", {DeviceId::rocm(0)}, CollectiveBackendType::AUTO}
        };
        // Gap between layers 10 and 15
        config.pp_stages = {
            PPStageConfig::firstStage(0, "domain_a", 0, 10),
            PPStageConfig::lastStage(1, "domain_b", 15, 24)  // Gap: layers 10-14 not covered
        };

        std::string error_msg;
        EXPECT_FALSE(config.validate(&error_msg));
    }

    // =========================================================================
    // Backend Selection Tests
    // =========================================================================

    TEST_F(Test__PipelineTopologyQueries, TPInPPConfig_TransferBackends)
    {
        auto config = makeTPInPPConfig();
        config.autoSelectBackends();  // Let it auto-select transfer backends

        // Check that transfer backends are set
        auto backend_0_1 = config.getTransferBackend(0, 1);
        auto backend_1_2 = config.getTransferBackend(1, 2);

        // Should not be AUTO after autoSelectBackends()
        // (Actual selection depends on BackendSelector logic)
        EXPECT_NE(backend_0_1, CollectiveBackendType::AUTO);
        EXPECT_NE(backend_1_2, CollectiveBackendType::AUTO);
    }

    // =========================================================================
    // Domain degree queries
    // =========================================================================

    /**
     * @brief Classify declaration shape independently of any execution strategy.
     */
    TEST_F(Test__PipelineTopologyQueries, DegreeQuery_SimpleConfigHasNoInternalTP)
    {
        auto config = makeSimplePPConfig();

        EXPECT_FALSE(config.hasTP());
    }

    TEST_F(Test__PipelineTopologyQueries, DegreeQuery_MixedConfigHasInternalTP)
    {
        auto config = makeTPInPPConfig();

        EXPECT_TRUE(config.hasTP());
    }

    /**
     * @brief Every declared stage resolves to the correct domain degree.
     */
    TEST_F(Test__PipelineTopologyQueries, StageBuilding_DetectsTPDomains)
    {
        auto config = makeTPInPPConfig();

        // These are declaration queries, not a simulation of context creation.
        for (int s = 0; s < config.numStages(); ++s)
        {
            const auto* domain = config.getDomainForStage(s);
            ASSERT_NE(domain, nullptr);

            bool is_tp_domain = domain->degree() > 1;

            if (s == 0)
            {
                EXPECT_TRUE(is_tp_domain) << "Stage 0 (rocm_tp) should be detected as TP domain";
            }
            else
            {
                EXPECT_FALSE(is_tp_domain) << "Stage " << s << " should be single device";
            }
        }
    }

    TEST_F(Test__PipelineTopologyQueries, StageBuilding_MultiTPBothDetected)
    {
        auto config = makeMultiTPConfig();

        int tp_domain_count = 0;
        int single_device_count = 0;

        for (int s = 0; s < config.numStages(); ++s)
        {
            const auto* domain = config.getDomainForStage(s);
            ASSERT_NE(domain, nullptr);

            if (domain->degree() > 1)
            {
                ++tp_domain_count;
            }
            else
            {
                ++single_device_count;
            }
        }

        EXPECT_EQ(tp_domain_count, 2) << "Should detect 2 TP domains (rocm_tp, cuda_tp)";
        EXPECT_EQ(single_device_count, 1) << "Should detect 1 single device (cpu_single)";
    }

} // namespace llaminar2::test
