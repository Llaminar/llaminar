/**
 * @file Test__TPContextFactory.cpp
 * @brief Unit tests for TPContextFactory
 *
 * Tests factory methods for creating LOCAL and GLOBAL tensor parallelism contexts.
 * Uses MPI_COMM_SELF for single-rank tests.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include "collective/TPContextFactory.h"
#include "collective/LocalTPContext.h"
#include "collective/GlobalTPContext.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"

using namespace llaminar2;

class Test__TPContextFactory : public ::testing::Test {
protected:
    void SetUp() override {
        int initialized;
        MPI_Initialized(&initialized);
        ASSERT_TRUE(initialized) << "MPI must be initialized for TPContextFactory tests";
    }
    
    // Helper to create a basic local TP plan
    RankExecutionPlan createLocalTPPlan(int device_count) {
        RankExecutionPlan plan;
        plan.rank = 0;
        plan.hostname = "test-host";
        plan.first_layer = 0;
        plan.last_layer = 27;
        
        for (int i = 0; i < device_count; ++i) {
            plan.local_tp_devices.push_back(GlobalDeviceAddress::cuda(i));
        }
        plan.local_tp_backend = CollectiveBackendType::AUTO;
        
        return plan;
    }
    
    // Helper to create a basic global TP plan
    RankExecutionPlan createGlobalTPPlan(int domain_id, int rank_in_domain, int domain_size) {
        RankExecutionPlan plan;
        plan.rank = rank_in_domain;
        plan.hostname = "test-host";
        plan.first_layer = 0;
        plan.last_layer = 27;
        
        plan.global_tp_domain_id = domain_id;
        plan.global_tp_rank_in_domain = rank_in_domain;
        plan.global_tp_domain_size = domain_size;
        
        return plan;
    }
    
    // Helper to create a domain participation
    TPDomainParticipation createLocalDomain(int device_count) {
        TPDomainParticipation domain;
        domain.domain_id = 0;
        domain.domain_name = "local_test";
        domain.my_index_in_domain = 0;
        domain.backend = CollectiveBackendType::AUTO;
        
        for (int i = 0; i < device_count; ++i) {
            domain.devices.push_back(GlobalDeviceAddress::cuda(i));
        }
        
        return domain;
    }
};

// =============================================================================
// CreateLocal Tests
// =============================================================================

TEST_F(Test__TPContextFactory, CreateLocal_ValidDevices) {
    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)
    };
    
    auto ctx = TPContextFactory::createLocal(devices);
    
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 2);
    EXPECT_TRUE(ctx->isLocal());
    EXPECT_FALSE(ctx->isGlobal());
}

TEST_F(Test__TPContextFactory, CreateLocal_WithWeights) {
    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)
    };
    std::vector<float> weights = {0.6f, 0.4f};
    
    auto ctx = TPContextFactory::createLocal(devices, weights);
    
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 2);
    
    // Verify weights through ILocalTPContext
    const auto& actual_weights = ctx->weights();
    ASSERT_EQ(actual_weights.size(), 2);
    EXPECT_FLOAT_EQ(actual_weights[0], 0.6f);
    EXPECT_FLOAT_EQ(actual_weights[1], 0.4f);
}

TEST_F(Test__TPContextFactory, CreateLocal_EmptyDevices) {
    std::vector<GlobalDeviceAddress> devices;
    
    auto ctx = TPContextFactory::createLocal(devices);
    
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(Test__TPContextFactory, CreateLocal_SingleDevice) {
    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cpu()
    };
    
    // Should work but with warning
    auto ctx = TPContextFactory::createLocal(devices);
    
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 1);
}

TEST_F(Test__TPContextFactory, CreateLocal_ExplicitBackend) {
    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)
    };
    
    auto ctx = TPContextFactory::createLocal(devices, {}, CollectiveBackendType::NCCL);
    
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->backend(), CollectiveBackendType::NCCL);
}

// =============================================================================
// CreateLocalFromPlan Tests
// =============================================================================

TEST_F(Test__TPContextFactory, CreateLocalFromPlan_ValidPlan) {
    auto plan = createLocalTPPlan(2);
    
    auto ctx = TPContextFactory::createLocalFromPlan(plan);
    
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 2);
    EXPECT_TRUE(ctx->isLocal());
}

TEST_F(Test__TPContextFactory, CreateLocalFromPlan_NoLocalTP) {
    RankExecutionPlan plan;
    plan.rank = 0;
    // No local_tp_devices set
    
    auto ctx = TPContextFactory::createLocalFromPlan(plan);
    
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(Test__TPContextFactory, CreateLocalFromPlan_SingleDevice) {
    auto plan = createLocalTPPlan(1);
    
    // Single device doesn't count as "uses local TP"
    EXPECT_FALSE(plan.usesLocalTP());
    
    auto ctx = TPContextFactory::createLocalFromPlan(plan);
    EXPECT_EQ(ctx, nullptr);
}

// =============================================================================
// CreateGlobal Tests
// =============================================================================

TEST_F(Test__TPContextFactory, CreateGlobal_ValidParams) {
    auto ctx = TPContextFactory::createGlobal(
        MPI_COMM_SELF,  // Single rank for testing
        /*domain_id=*/42,
        /*color=*/0,
        /*key=*/0
    );
    
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 1);  // MPI_COMM_SELF has size 1
    EXPECT_FALSE(ctx->isLocal());
    EXPECT_TRUE(ctx->isGlobal());
    EXPECT_EQ(ctx->domainId(), 42);
}

TEST_F(Test__TPContextFactory, CreateGlobal_NullComm) {
    auto ctx = TPContextFactory::createGlobal(
        MPI_COMM_NULL,
        /*domain_id=*/0,
        /*color=*/0,
        /*key=*/0
    );
    
    EXPECT_EQ(ctx, nullptr);
}

// =============================================================================
// CreateGlobalFromPlan Tests
// =============================================================================

TEST_F(Test__TPContextFactory, CreateGlobalFromPlan_ValidPlan) {
    auto plan = createGlobalTPPlan(/*domain_id=*/10, /*rank_in_domain=*/0, /*domain_size=*/2);
    
    auto ctx = TPContextFactory::createGlobalFromPlan(plan, MPI_COMM_SELF);
    
    ASSERT_NE(ctx, nullptr);
    EXPECT_FALSE(ctx->isLocal());
    EXPECT_TRUE(ctx->isGlobal());
    EXPECT_EQ(ctx->domainId(), 10);
}

TEST_F(Test__TPContextFactory, CreateGlobalFromPlan_NoGlobalTP) {
    RankExecutionPlan plan;
    plan.rank = 0;
    // No global_tp_domain_id set
    
    auto ctx = TPContextFactory::createGlobalFromPlan(plan, MPI_COMM_SELF);
    
    EXPECT_EQ(ctx, nullptr);
}

// =============================================================================
// Create (Main Factory) Tests
// =============================================================================

TEST_F(Test__TPContextFactory, Create_GlobalTPTakesPrecedence) {
    // Plan with both local and global TP
    RankExecutionPlan plan;
    plan.rank = 0;
    plan.first_layer = 0;
    plan.last_layer = 27;
    
    // Set up local TP
    plan.local_tp_devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)
    };
    
    // Set up global TP
    plan.global_tp_domain_id = 5;
    plan.global_tp_rank_in_domain = 0;
    plan.global_tp_domain_size = 2;
    
    // Global should take precedence
    auto ctx = TPContextFactory::create(plan, MPI_COMM_SELF);
    
    ASSERT_NE(ctx, nullptr);
    EXPECT_TRUE(ctx->isGlobal());  // Should be global, not local
    EXPECT_FALSE(ctx->isLocal());
}

TEST_F(Test__TPContextFactory, Create_LocalTPOnly) {
    auto plan = createLocalTPPlan(2);
    
    auto ctx = TPContextFactory::create(plan, MPI_COMM_SELF);
    
    ASSERT_NE(ctx, nullptr);
    EXPECT_TRUE(ctx->isLocal());
    EXPECT_FALSE(ctx->isGlobal());
}

TEST_F(Test__TPContextFactory, Create_GlobalTPOnly) {
    auto plan = createGlobalTPPlan(7, 0, 2);
    
    auto ctx = TPContextFactory::create(plan, MPI_COMM_SELF);
    
    ASSERT_NE(ctx, nullptr);
    EXPECT_FALSE(ctx->isLocal());
    EXPECT_TRUE(ctx->isGlobal());
}

TEST_F(Test__TPContextFactory, Create_NoTPConfigured) {
    RankExecutionPlan plan;
    plan.rank = 0;
    // No TP configured
    
    auto ctx = TPContextFactory::create(plan, MPI_COMM_SELF);
    
    EXPECT_EQ(ctx, nullptr);
}

// =============================================================================
// CreateFromDomain Tests
// =============================================================================

TEST_F(Test__TPContextFactory, CreateFromDomain_LocalDomain) {
    auto domain = createLocalDomain(2);
    
    auto ctx = TPContextFactory::createFromDomain(domain, MPI_COMM_SELF);
    
    ASSERT_NE(ctx, nullptr);
    EXPECT_TRUE(ctx->isLocal());
    EXPECT_EQ(ctx->degree(), 2);
}

TEST_F(Test__TPContextFactory, CreateFromDomain_EmptyDomain) {
    TPDomainParticipation domain;
    domain.domain_id = 0;
    domain.domain_name = "empty";
    // No devices
    
    auto ctx = TPContextFactory::createFromDomain(domain, MPI_COMM_SELF);
    
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(Test__TPContextFactory, CreateFromDomain_UPIBackendIsGlobal) {
    TPDomainParticipation domain;
    domain.domain_id = 1;
    domain.domain_name = "upi_domain";
    domain.devices = {GlobalDeviceAddress::cpu()};
    domain.my_index_in_domain = 0;
    domain.backend = CollectiveBackendType::UPI;  // UPI implies cross-rank
    
    auto ctx = TPContextFactory::createFromDomain(domain, MPI_COMM_SELF);
    
    ASSERT_NE(ctx, nullptr);
    EXPECT_TRUE(ctx->isGlobal());  // UPI backend indicates global
}

TEST_F(Test__TPContextFactory, CreateFromDomain_MPIBackendIsGlobal) {
    TPDomainParticipation domain;
    domain.domain_id = 2;
    domain.domain_name = "mpi_domain";
    domain.devices = {GlobalDeviceAddress::cpu()};
    domain.my_index_in_domain = 0;
    domain.backend = CollectiveBackendType::MPI;  // MPI implies cross-rank
    
    auto ctx = TPContextFactory::createFromDomain(domain, MPI_COMM_SELF);
    
    ASSERT_NE(ctx, nullptr);
    EXPECT_TRUE(ctx->isGlobal());
}

TEST_F(Test__TPContextFactory, CreateFromDomain_WithWeights) {
    TPDomainParticipation domain;
    domain.domain_id = 0;
    domain.domain_name = "weighted";
    domain.devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)
    };
    domain.weights = {0.7f, 0.3f};
    domain.my_index_in_domain = 0;
    domain.backend = CollectiveBackendType::NCCL;
    
    auto ctx = TPContextFactory::createFromDomain(domain, MPI_COMM_SELF);
    
    ASSERT_NE(ctx, nullptr);
    EXPECT_TRUE(ctx->isLocal());
    
    // Cast to check weights
    auto* local_ctx = dynamic_cast<ILocalTPContext*>(ctx.get());
    ASSERT_NE(local_ctx, nullptr);
    EXPECT_EQ(local_ctx->weights().size(), 2);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST_F(Test__TPContextFactory, CreateLocal_CPUDevices) {
    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cpu(),
        GlobalDeviceAddress::cpu()
    };
    
    // Multiple CPU devices (unusual but valid for testing)
    auto ctx = TPContextFactory::createLocal(devices);
    
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 2);
}

TEST_F(Test__TPContextFactory, CreateLocal_MixedDevices) {
    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0)
    };
    
    auto ctx = TPContextFactory::createLocal(devices);
    
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 2);
}

TEST_F(Test__TPContextFactory, CreateGlobal_DifferentDomainIds) {
    // Create two contexts with different domain IDs
    auto ctx1 = TPContextFactory::createGlobal(MPI_COMM_SELF, 1, 0, 0);
    auto ctx2 = TPContextFactory::createGlobal(MPI_COMM_SELF, 2, 0, 0);
    
    ASSERT_NE(ctx1, nullptr);
    ASSERT_NE(ctx2, nullptr);
    EXPECT_EQ(ctx1->domainId(), 1);
    EXPECT_EQ(ctx2->domainId(), 2);
}
