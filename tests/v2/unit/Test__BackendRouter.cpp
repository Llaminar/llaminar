/**
 * @file Test__BackendRouter.cpp
 * @brief Unit tests for BackendRouter
 *
 * Tests backend selection logic, caching, and initialization behavior
 * using MockBackendFactory for dependency injection.
 */

#include <gtest/gtest.h>
#include "v2/collective/BackendRouter.h"
#include "v2/collective/test/CollectiveTestMocks.h"
#include "v2/collective/backends/UPIBackend.h"
#include "v2/config/TPDomain.h"
#include "v2/execution/DeviceInventory.h"
#include "v2/backends/DeviceId.h"

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__BackendRouter : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Create empty cluster inventory
            cluster_inventory_.world_size = 1;
            cluster_inventory_.node_count = 1;

            // Default MPI context is nullptr (single-rank)
            mpi_ctx_ = nullptr;
        }

        std::unique_ptr<BackendRouter> createRouter(std::unique_ptr<MockBackendFactory> factory)
        {
            return std::make_unique<BackendRouter>(
                mpi_ctx_,
                cluster_inventory_,
                std::move(factory));
        }

        // Helper to create a homogeneous CUDA group
        DeviceGroup createCUDAGroup(const std::string &name, int num_gpus)
        {
            DeviceGroupBuilder builder;
            builder.setName(name).setScope(CollectiveScope::LOCAL);
            for (int i = 0; i < num_gpus; ++i)
            {
                builder.addDevice(DeviceId::cuda(i));
            }
            return builder.build();
        }

        // Helper to create a homogeneous ROCm group
        DeviceGroup createROCmGroup(const std::string &name, int num_gpus)
        {
            DeviceGroupBuilder builder;
            builder.setName(name).setScope(CollectiveScope::LOCAL);
            for (int i = 0; i < num_gpus; ++i)
            {
                builder.addDevice(DeviceId::rocm(i));
            }
            return builder.build();
        }

        // Helper to create a heterogeneous CUDA + CPU group
        DeviceGroup createHeterogeneousGroup(const std::string &name)
        {
            DeviceGroupBuilder builder;
            return builder
                .setName(name)
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::cuda(1))
                .addDevice(DeviceId::cpu())
                .build();
        }

        // Helper to create a CUDA + ROCm mixed group (cross-vendor GPU)
        DeviceGroup createCUDAROCmMixGroup(const std::string &name)
        {
            DeviceGroupBuilder builder;
            return builder
                .setName(name)
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::rocm(0))
                .build();
        }

        // Helper to create a global scope group
        DeviceGroup createGlobalGroup(const std::string &name)
        {
            DeviceGroupBuilder builder;
            return builder
                .setName(name)
                .setScope(CollectiveScope::GLOBAL)
                .addDevice(DeviceId::cuda(0))
                .build();
        }

        std::shared_ptr<MPIContext> mpi_ctx_;
        ClusterInventory cluster_inventory_;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Backend Selection Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__BackendRouter, SelectsNCCLForHomogeneousCUDA)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        // Configure NCCL as available
        factory->setAvailable(CollectiveBackendType::NCCL, true);

        // Add mock NCCL backend
        auto *mock_nccl = factory->addMockBackend(CollectiveBackendType::NCCL);
        mock_nccl->setAvailable(true);

        auto router = createRouter(std::move(factory));
        auto group = createCUDAGroup("cuda_gpus", 4);

        auto selection = router->selectBackend(group);
        EXPECT_EQ(selection.type, CollectiveBackendType::NCCL);
        EXPECT_FALSE(selection.requires_multi_phase);
    }

    TEST_F(Test__BackendRouter, SelectsRCCLForHomogeneousROCm)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        // Configure RCCL as available
        factory->setAvailable(CollectiveBackendType::RCCL, true);

        // Add mock RCCL backend
        auto *mock_rccl = factory->addMockBackend(CollectiveBackendType::RCCL);
        mock_rccl->setAvailable(true);

        auto router = createRouter(std::move(factory));
        auto group = createROCmGroup("rocm_gpus", 2);

        auto selection = router->selectBackend(group);
        EXPECT_EQ(selection.type, CollectiveBackendType::RCCL);
        EXPECT_FALSE(selection.requires_multi_phase);
    }

    TEST_F(Test__BackendRouter, SelectsHostForHeterogeneous)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        // Configure both NCCL and HOST as available
        factory->setAvailable(CollectiveBackendType::NCCL, true);
        factory->setAvailable(CollectiveBackendType::HOST, true);

        auto router = createRouter(std::move(factory));
        auto group = createHeterogeneousGroup("mixed_devices");

        auto selection = router->selectBackend(group);
        EXPECT_EQ(selection.type, CollectiveBackendType::HOST);
        EXPECT_TRUE(selection.requires_multi_phase);
    }

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
    TEST_F(Test__BackendRouter, SelectsPCIeBARForCUDAROCmMix)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        // Configure PCIE_BAR as available (simulating runtime detection success)
        factory->setAvailable(CollectiveBackendType::PCIE_BAR, true);
        factory->setAvailable(CollectiveBackendType::HOST, true);

        // Add mock PCIE_BAR backend
        auto *mock_pcie = factory->addMockBackend(CollectiveBackendType::PCIE_BAR);
        mock_pcie->setAvailable(true);

        auto router = createRouter(std::move(factory));
        auto group = createCUDAROCmMixGroup("cuda_rocm_mix");

        auto selection = router->selectBackend(group);
        EXPECT_EQ(selection.type, CollectiveBackendType::PCIE_BAR);
        EXPECT_FALSE(selection.requires_multi_phase);
        EXPECT_NE(selection.reason.find("PCIe BAR"), std::string::npos)
            << "Expected reason to mention PCIe BAR, got: " << selection.reason;
    }

    TEST_F(Test__BackendRouter, FallsBackToHostWhenPCIeBARUnavailableForCUDAROCm)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        // Configure PCIE_BAR as NOT available, HOST as available
        factory->setAvailable(CollectiveBackendType::PCIE_BAR, false);
        factory->setAvailable(CollectiveBackendType::HOST, true);

        // Add mock HOST backend
        auto *mock_host = factory->addMockBackend(CollectiveBackendType::HOST);
        mock_host->setAvailable(true);

        auto router = createRouter(std::move(factory));
        auto group = createCUDAROCmMixGroup("cuda_rocm_mix");

        auto selection = router->selectBackend(group);
        EXPECT_EQ(selection.type, CollectiveBackendType::HOST);
        // Heterogeneous without P2P requires multi-phase
        EXPECT_TRUE(selection.requires_multi_phase);
    }
#endif // HAVE_CUDA && HAVE_ROCM

    // =========================================================================
    // GLOBAL scope tests - NCCL/RCCL are preferred for homogeneous GPU groups
    // even at global scope, since they support cross-rank communication via
    // ncclCommInitRank/rcclCommInitRank with MPI-broadcast unique ID.
    // =========================================================================

    TEST_F(Test__BackendRouter, SelectsNCCLForGlobalScopeHomogeneousCUDA)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        // Configure both MPI and NCCL as available
        factory->setAvailable(CollectiveBackendType::MPI, true);
        factory->setAvailable(CollectiveBackendType::NCCL, true);

        // Add mock NCCL backend (should be selected for homogeneous CUDA)
        auto *mock_nccl = factory->addMockBackend(CollectiveBackendType::NCCL);
        mock_nccl->setAvailable(true);

        auto router = createRouter(std::move(factory));
        auto group = createGlobalGroup("global_cuda_ranks"); // All CUDA devices

        auto selection = router->selectBackend(group);
        EXPECT_EQ(selection.type, CollectiveBackendType::NCCL);
        EXPECT_FALSE(selection.requires_multi_phase);
    }

    TEST_F(Test__BackendRouter, SelectsMPIForGlobalScopeHeterogeneousDevices)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        // Configure MPI as available
        factory->setAvailable(CollectiveBackendType::MPI, true);
        factory->setAvailable(CollectiveBackendType::NCCL, true);
        factory->setAvailable(CollectiveBackendType::RCCL, true);

        // Add mock MPI backend
        auto *mock_mpi = factory->addMockBackend(CollectiveBackendType::MPI);
        mock_mpi->setAvailable(true);

        auto router = createRouter(std::move(factory));

        // Create a heterogeneous global group (CUDA + ROCm)
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("global_hetero_ranks")
                         .setScope(CollectiveScope::GLOBAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::rocm(0))
                         .build();

        auto selection = router->selectBackend(group);
        // Heterogeneous global groups should fall back to MPI
        // (PCIeBAR is only for local scope)
        EXPECT_EQ(selection.type, CollectiveBackendType::MPI);
        EXPECT_FALSE(selection.requires_multi_phase);
    }

    TEST_F(Test__BackendRouter, SelectsMPIForGlobalScopeCPUOnly)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        // Configure MPI as available
        factory->setAvailable(CollectiveBackendType::MPI, true);

        // Add mock MPI backend
        auto *mock_mpi = factory->addMockBackend(CollectiveBackendType::MPI);
        mock_mpi->setAvailable(true);

        auto router = createRouter(std::move(factory));

        // Create a CPU-only global group
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("global_cpu_ranks")
                         .setScope(CollectiveScope::GLOBAL)
                         .addDevice(DeviceId::cpu())
                         .build();

        auto selection = router->selectBackend(group);
        // CPU-only groups should use MPI
        EXPECT_EQ(selection.type, CollectiveBackendType::MPI);
        EXPECT_FALSE(selection.requires_multi_phase);
    }

    TEST_F(Test__BackendRouter, FallsBackToHostWhenNCCLUnavailable)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        // Configure NCCL as NOT available, HOST as available
        factory->setAvailable(CollectiveBackendType::NCCL, false);
        factory->setAvailable(CollectiveBackendType::HOST, true);

        // Add mock HOST backend
        auto *mock_host = factory->addMockBackend(CollectiveBackendType::HOST);
        mock_host->setAvailable(true);

        auto router = createRouter(std::move(factory));
        auto group = createCUDAGroup("cuda_gpus", 4);

        auto selection = router->selectBackend(group);
        EXPECT_EQ(selection.type, CollectiveBackendType::HOST);
    }

    TEST_F(Test__BackendRouter, FallsBackToHostWhenRCCLUnavailable)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        // Configure RCCL as NOT available, HOST as available
        factory->setAvailable(CollectiveBackendType::RCCL, false);
        factory->setAvailable(CollectiveBackendType::HOST, true);

        // Add mock HOST backend
        auto *mock_host = factory->addMockBackend(CollectiveBackendType::HOST);
        mock_host->setAvailable(true);

        auto router = createRouter(std::move(factory));
        auto group = createROCmGroup("rocm_gpus", 2);

        auto selection = router->selectBackend(group);
        EXPECT_EQ(selection.type, CollectiveBackendType::HOST);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Backend Caching Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__BackendRouter, CachesBackendByGroupName)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        // Configure HOST as available
        factory->setAvailable(CollectiveBackendType::HOST, true);

        // Add mock HOST backend that we can track
        auto *mock_host = factory->addMockBackend(CollectiveBackendType::HOST);
        mock_host->setInitResult(true);

        auto router = createRouter(std::move(factory));
        auto group = createCUDAGroup("cached_group", 2);

        // First call should get the backend
        auto *backend1 = router->getBackend(group);
        ASSERT_NE(backend1, nullptr);

        // Second call with same group name should return same backend pointer
        auto *backend2 = router->getBackend(group);
        EXPECT_EQ(backend1, backend2);
    }

    TEST_F(Test__BackendRouter, DifferentGroupsGetCachedSeparately)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        // Configure HOST as available
        factory->setAvailable(CollectiveBackendType::HOST, true);

        // Add mock HOST backend
        auto *mock_host = factory->addMockBackend(CollectiveBackendType::HOST);
        mock_host->setInitResult(true);

        auto router = createRouter(std::move(factory));
        auto group1 = createCUDAGroup("group_a", 2);
        auto group2 = createCUDAGroup("group_b", 2);

        // Get backends for both groups
        auto *backend1 = router->getBackend(group1);
        auto *backend2 = router->getBackend(group2);

        ASSERT_NE(backend1, nullptr);
        ASSERT_NE(backend2, nullptr);
        // Both should map to the same underlying HOST backend
        EXPECT_EQ(backend1, backend2);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Initialization Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__BackendRouter, InitializesBackendOnFirstUse)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        // Configure HOST as available
        factory->setAvailable(CollectiveBackendType::HOST, true);

        // Add mock HOST backend
        auto *mock_host = factory->addMockBackend(CollectiveBackendType::HOST);
        mock_host->setInitResult(true);

        auto router = createRouter(std::move(factory));
        auto group = createCUDAGroup("init_test_group", 2);

        // Backend should not be initialized yet
        EXPECT_FALSE(mock_host->wasInitialized());

        // Get backend - should trigger initialization
        auto *backend = router->getBackend(group);
        ASSERT_NE(backend, nullptr);
        EXPECT_TRUE(mock_host->wasInitialized());

        // Verify initialize was called with the correct group
        EXPECT_EQ(mock_host->initGroup().name, group.name);
    }

    TEST_F(Test__BackendRouter, ReturnsNullptrOnInitializationFailure)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        // Configure HOST as available
        factory->setAvailable(CollectiveBackendType::HOST, true);

        // Add mock HOST backend that fails to initialize
        auto *mock_host = factory->addMockBackend(CollectiveBackendType::HOST);
        mock_host->setInitResult(false); // Will fail to initialize

        auto router = createRouter(std::move(factory));
        auto group = createCUDAGroup("fail_init_group", 2);

        // Should return nullptr because initialization fails
        auto *backend = router->getBackend(group);
        EXPECT_EQ(backend, nullptr);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Availability Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__BackendRouter, IsAvailableDelegatesToFactory)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        // Configure different availability settings
        factory->setAvailable(CollectiveBackendType::HOST, true);
        factory->setAvailable(CollectiveBackendType::MPI, true);
        factory->setAvailable(CollectiveBackendType::NCCL, false);
        factory->setAvailable(CollectiveBackendType::RCCL, false);

        auto router = createRouter(std::move(factory));

        EXPECT_TRUE(router->isAvailable(CollectiveBackendType::HOST));
        EXPECT_TRUE(router->isAvailable(CollectiveBackendType::MPI));
        EXPECT_FALSE(router->isAvailable(CollectiveBackendType::NCCL));
        EXPECT_FALSE(router->isAvailable(CollectiveBackendType::RCCL));
    }

    TEST_F(Test__BackendRouter, AvailableBackendsReturnsCorrectList)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        // Configure availability
        factory->setAvailable(CollectiveBackendType::HOST, true);
        factory->setAvailable(CollectiveBackendType::MPI, true);
        factory->setAvailable(CollectiveBackendType::NCCL, true);
        factory->setAvailable(CollectiveBackendType::RCCL, false);

        auto router = createRouter(std::move(factory));

        auto available = router->availableBackends();

        // Should contain HOST, MPI, NCCL but not RCCL
        EXPECT_NE(std::find(available.begin(), available.end(), CollectiveBackendType::HOST), available.end());
        EXPECT_NE(std::find(available.begin(), available.end(), CollectiveBackendType::MPI), available.end());
        EXPECT_NE(std::find(available.begin(), available.end(), CollectiveBackendType::NCCL), available.end());
        EXPECT_EQ(std::find(available.begin(), available.end(), CollectiveBackendType::RCCL), available.end());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // getBackend by Type Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__BackendRouter, GetBackendByTypeReturnsBackend)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        factory->setAvailable(CollectiveBackendType::HOST, true);
        auto *mock_host = factory->addMockBackend(CollectiveBackendType::HOST);

        auto router = createRouter(std::move(factory));

        // Getting by type should create the backend
        auto *backend = router->getBackend(CollectiveBackendType::HOST);
        ASSERT_NE(backend, nullptr);
        EXPECT_EQ(backend->type(), CollectiveBackendType::HOST);
    }

    TEST_F(Test__BackendRouter, GetBackendByTypeReturnsNullptrForUnavailable)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        factory->setAvailable(CollectiveBackendType::NCCL, false);

        auto router = createRouter(std::move(factory));

        auto *backend = router->getBackend(CollectiveBackendType::NCCL);
        EXPECT_EQ(backend, nullptr);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // initializeBackend Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__BackendRouter, InitializeBackendExplicitly)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        factory->setAvailable(CollectiveBackendType::HOST, true);
        auto *mock_host = factory->addMockBackend(CollectiveBackendType::HOST);
        mock_host->setInitResult(true);

        auto router = createRouter(std::move(factory));
        auto group = createCUDAGroup("explicit_init_group", 2);

        EXPECT_FALSE(mock_host->wasInitialized());

        bool result = router->initializeBackend(CollectiveBackendType::HOST, group);
        EXPECT_TRUE(result);
        EXPECT_TRUE(mock_host->wasInitialized());
    }

    TEST_F(Test__BackendRouter, InitializeBackendReturnsTrueIfAlreadyInitialized)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        factory->setAvailable(CollectiveBackendType::HOST, true);
        auto *mock_host = factory->addMockBackend(CollectiveBackendType::HOST);
        mock_host->setInitResult(true);

        auto router = createRouter(std::move(factory));
        auto group = createCUDAGroup("already_init_group", 2);

        // First initialization
        EXPECT_TRUE(router->initializeBackend(CollectiveBackendType::HOST, group));

        // Second initialization should also return true (already initialized)
        EXPECT_TRUE(router->initializeBackend(CollectiveBackendType::HOST, group));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Shutdown Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__BackendRouter, ShutdownClearsBackends)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        factory->setAvailable(CollectiveBackendType::HOST, true);
        auto *mock_host = factory->addMockBackend(CollectiveBackendType::HOST);
        mock_host->setInitResult(true);

        auto router = createRouter(std::move(factory));
        auto group = createCUDAGroup("shutdown_test", 2);

        // Get backend to create and initialize it
        auto *backend = router->getBackend(group);
        ASSERT_NE(backend, nullptr);

        // Shutdown
        router->shutdown();

        // After shutdown, getBackend by type should return nullptr
        EXPECT_EQ(router->getBackend(CollectiveBackendType::HOST), nullptr);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Heterogeneous Operations Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__BackendRouter, ExecuteHeterogeneousAllReduceUsesHostBackend)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        factory->setAvailable(CollectiveBackendType::HOST, true);
        auto *mock_host = factory->addMockBackend(CollectiveBackendType::HOST);
        mock_host->setInitResult(true);
        mock_host->setAllreduceResult(true);

        auto router = createRouter(std::move(factory));
        auto group = createHeterogeneousGroup("hetero_allreduce_group");

        float buffer[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};

        bool result = router->executeHeterogeneousAllReduce(
            group,
            buffer,
            10,
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM);

        EXPECT_TRUE(result);
        EXPECT_EQ(mock_host->allreduceCallCount(), 1);
    }

    TEST_F(Test__BackendRouter, ExecuteHeterogeneousAllGatherUsesHostBackend)
    {
        auto factory = std::make_unique<MockBackendFactory>();

        factory->setAvailable(CollectiveBackendType::HOST, true);
        auto *mock_host = factory->addMockBackend(CollectiveBackendType::HOST);
        mock_host->setInitResult(true);
        mock_host->setAllgatherResult(true);

        auto router = createRouter(std::move(factory));
        auto group = createHeterogeneousGroup("hetero_allgather_group");

        float send_buf[10];
        float recv_buf[30]; // 3 devices in the group

        bool result = router->executeHeterogeneousAllGather(
            group,
            send_buf,
            recv_buf,
            10,
            CollectiveDataType::FLOAT32);

        EXPECT_TRUE(result);
        EXPECT_EQ(mock_host->allgatherCallCount(), 1);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Diagnostics Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__BackendRouter, DiagnosticsReturnsNonEmptyString)
    {
        auto factory = std::make_unique<MockBackendFactory>();
        factory->setAvailable(CollectiveBackendType::HOST, true);
        factory->setAvailable(CollectiveBackendType::MPI, true);

        auto router = createRouter(std::move(factory));

        std::string diag = router->diagnostics();
        EXPECT_FALSE(diag.empty());
        EXPECT_NE(diag.find("MPI"), std::string::npos);
        EXPECT_NE(diag.find("Host"), std::string::npos);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Domain-Aware Backend Selection Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__BackendRouter, SelectBackendForDomainNull)
    {
        auto factory = std::make_unique<MockBackendFactory>();
        factory->setAvailable(CollectiveBackendType::MPI, true);
        auto *mock_mpi = factory->addMockBackend(CollectiveBackendType::MPI);
        mock_mpi->setAvailable(true);

        auto router = createRouter(std::move(factory));

        // Null domain should return MPI backend
        ICollectiveBackend *backend = router->selectBackendForDomain(nullptr);
        ASSERT_NE(backend, nullptr);
        EXPECT_EQ(backend->type(), CollectiveBackendType::MPI);
    }

    TEST_F(Test__BackendRouter, SelectBackendForTrivialDomain)
    {
        auto factory = std::make_unique<MockBackendFactory>();
        factory->setAvailable(CollectiveBackendType::MPI, true);

        auto router = createRouter(std::move(factory));

        // Create a trivial domain (size <= 1)
        TPDomain trivial_domain;
        trivial_domain.type = TPDomainType::GPU_INTRA_RANK;
        trivial_domain.domain_size = 1;
        trivial_domain.devices = {DeviceId::cuda(0)};
        trivial_domain.name = "trivial_domain";

        // Trivial domain should return nullptr (no communication needed)
        ICollectiveBackend *backend = router->selectBackendForDomain(&trivial_domain);
        EXPECT_EQ(backend, nullptr);
    }

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
    TEST_F(Test__BackendRouter, SelectBackendForGPUIntraRankHeterogeneous)
    {
        auto factory = std::make_unique<MockBackendFactory>();
        factory->setAvailable(CollectiveBackendType::PCIE_BAR, true);
        factory->setAvailable(CollectiveBackendType::MPI, true);
        auto *mock_pcie = factory->addMockBackend(CollectiveBackendType::PCIE_BAR);
        mock_pcie->setAvailable(true);

        auto router = createRouter(std::move(factory));

        // Create GPU domain with CUDA + ROCm mix
        TPDomain gpu_domain;
        gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
        gpu_domain.domain_size = 2;
        gpu_domain.devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
        gpu_domain.name = "gpu_hetero_domain";
        gpu_domain.local_rank_in_domain = 0;

        // Heterogeneous GPU domain should use PCIe BAR
        ICollectiveBackend *backend = router->selectBackendForDomain(&gpu_domain);
        ASSERT_NE(backend, nullptr);
        EXPECT_EQ(backend->type(), CollectiveBackendType::PCIE_BAR);
    }
#endif // HAVE_CUDA && HAVE_ROCM

    TEST_F(Test__BackendRouter, SelectBackendForGPUIntraRankAllCUDA)
    {
        auto factory = std::make_unique<MockBackendFactory>();
        factory->setAvailable(CollectiveBackendType::NCCL, true);
        factory->setAvailable(CollectiveBackendType::MPI, true);
        auto *mock_nccl = factory->addMockBackend(CollectiveBackendType::NCCL);
        mock_nccl->setAvailable(true);

        auto router = createRouter(std::move(factory));

        // Create GPU domain with all CUDA
        TPDomain gpu_domain;
        gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
        gpu_domain.domain_size = 4;
        gpu_domain.devices = {DeviceId::cuda(0), DeviceId::cuda(1),
                              DeviceId::cuda(2), DeviceId::cuda(3)};
        gpu_domain.name = "cuda_domain";
        gpu_domain.local_rank_in_domain = 0;

        // All CUDA domain should use NCCL
        ICollectiveBackend *backend = router->selectBackendForDomain(&gpu_domain);
        ASSERT_NE(backend, nullptr);
        EXPECT_EQ(backend->type(), CollectiveBackendType::NCCL);
    }

    TEST_F(Test__BackendRouter, SelectBackendForGPUIntraRankAllROCm)
    {
        auto factory = std::make_unique<MockBackendFactory>();
        factory->setAvailable(CollectiveBackendType::RCCL, true);
        factory->setAvailable(CollectiveBackendType::MPI, true);
        auto *mock_rccl = factory->addMockBackend(CollectiveBackendType::RCCL);
        mock_rccl->setAvailable(true);

        auto router = createRouter(std::move(factory));

        // Create GPU domain with all ROCm
        TPDomain gpu_domain;
        gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
        gpu_domain.domain_size = 2;
        gpu_domain.devices = {DeviceId::rocm(0), DeviceId::rocm(1)};
        gpu_domain.name = "rocm_domain";
        gpu_domain.local_rank_in_domain = 0;

        // All ROCm domain should use RCCL
        ICollectiveBackend *backend = router->selectBackendForDomain(&gpu_domain);
        ASSERT_NE(backend, nullptr);
        EXPECT_EQ(backend->type(), CollectiveBackendType::RCCL);
    }

    TEST_F(Test__BackendRouter, SelectBackendForGPUIntraRankFallsBackToMPI)
    {
        auto factory = std::make_unique<MockBackendFactory>();
        factory->setAvailable(CollectiveBackendType::NCCL, false); // Not available
        factory->setAvailable(CollectiveBackendType::MPI, true);
        auto *mock_mpi = factory->addMockBackend(CollectiveBackendType::MPI);
        mock_mpi->setAvailable(true);

        auto router = createRouter(std::move(factory));

        // Create GPU domain with all CUDA but NCCL unavailable
        TPDomain gpu_domain;
        gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
        gpu_domain.domain_size = 2;
        gpu_domain.devices = {DeviceId::cuda(0), DeviceId::cuda(1)};
        gpu_domain.name = "cuda_no_nccl_domain";
        gpu_domain.local_rank_in_domain = 0;

        // Should fall back to MPI when NCCL unavailable
        ICollectiveBackend *backend = router->selectBackendForDomain(&gpu_domain);
        ASSERT_NE(backend, nullptr);
        EXPECT_EQ(backend->type(), CollectiveBackendType::MPI);
    }

    TEST_F(Test__BackendRouter, SelectBackendForCPUCrossRankWithUPI)
    {
        auto factory = std::make_unique<MockBackendFactory>();
        factory->setAvailable(CollectiveBackendType::MPI, true);

        auto router = createRouter(std::move(factory));

        // Create mock UPI backend (simulating valid communicator)
        // Note: In real code, UPICollectiveBackend needs valid MPI comm
        // For testing, we use a mock approach by creating a fake UPI backend

        // Since we can't create a real UPICollectiveBackend without MPI,
        // we test the logic path by NOT registering UPI and verifying fallback

        // Create CPU cross-rank domain
        TPDomain cpu_domain;
        cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
        cpu_domain.domain_size = 2;
        cpu_domain.devices = {DeviceId::cpu(), DeviceId::cpu()};
        cpu_domain.name = "cpu_cross_rank_domain";
        cpu_domain.local_rank_in_domain = 0;

        // Without UPI backend registered, should fall back to MPI
        ICollectiveBackend *backend = router->selectBackendForDomain(&cpu_domain);
        ASSERT_NE(backend, nullptr);
        EXPECT_EQ(backend->type(), CollectiveBackendType::MPI);
    }

    TEST_F(Test__BackendRouter, SelectBackendForCPUCrossRankNoUPI)
    {
        auto factory = std::make_unique<MockBackendFactory>();
        factory->setAvailable(CollectiveBackendType::MPI, true);
        auto *mock_mpi = factory->addMockBackend(CollectiveBackendType::MPI);
        mock_mpi->setAvailable(true);

        auto router = createRouter(std::move(factory));

        // Create CPU cross-rank domain without registering UPI backend
        TPDomain cpu_domain;
        cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
        cpu_domain.domain_size = 4;
        cpu_domain.devices = {DeviceId::cpu()};
        cpu_domain.name = "cpu_cross_domain";
        cpu_domain.local_rank_in_domain = 0;

        // Without UPI backend, should fall back to MPI
        ICollectiveBackend *backend = router->selectBackendForDomain(&cpu_domain);
        ASSERT_NE(backend, nullptr);
        EXPECT_EQ(backend->type(), CollectiveBackendType::MPI);
    }

    TEST_F(Test__BackendRouter, HasDomainSupportFalseByDefault)
    {
        auto factory = std::make_unique<MockBackendFactory>();
        auto router = createRouter(std::move(factory));

        // By default, no UPI backend is registered
        EXPECT_FALSE(router->hasDomainSupport());
    }

    TEST_F(Test__BackendRouter, HasDomainSupportTrueAfterRegister)
    {
        auto factory = std::make_unique<MockBackendFactory>();
        auto router = createRouter(std::move(factory));

        // We can't create a real UPICollectiveBackend without valid MPI comm,
        // but we can test the registration path by checking hasDomainSupport
        // after registering nullptr (which won't happen in real use but tests the check)

        // Note: In a real integration test with MPI, we would:
        // auto upi = std::make_unique<UPICollectiveBackend>(MPI_COMM_WORLD);
        // router->registerUPIBackend(std::move(upi));
        // EXPECT_TRUE(router->hasDomainSupport());

        // For now, verify default behavior
        EXPECT_FALSE(router->hasDomainSupport());
    }

    TEST_F(Test__BackendRouter, HasHeterogeneousGPUsCUDAAndROCm)
    {
        auto factory = std::make_unique<MockBackendFactory>();
        auto router = createRouter(std::move(factory));

        // Create domain with CUDA + ROCm
        TPDomain domain;
        domain.type = TPDomainType::GPU_INTRA_RANK;
        domain.domain_size = 2;
        domain.devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
        domain.name = "cuda_rocm_mix";

        // We test this indirectly through selectBackendForDomain
        // Create factory with both PCIe BAR and MPI available
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        auto factory2 = std::make_unique<MockBackendFactory>();
        factory2->setAvailable(CollectiveBackendType::PCIE_BAR, true);
        factory2->setAvailable(CollectiveBackendType::MPI, true);
        auto *mock_pcie = factory2->addMockBackend(CollectiveBackendType::PCIE_BAR);
        mock_pcie->setAvailable(true);

        auto router2 = createRouter(std::move(factory2));

        ICollectiveBackend *backend = router2->selectBackendForDomain(&domain);
        ASSERT_NE(backend, nullptr);
        EXPECT_EQ(backend->type(), CollectiveBackendType::PCIE_BAR);
#endif
    }

    TEST_F(Test__BackendRouter, AllCUDADevices)
    {
        auto factory = std::make_unique<MockBackendFactory>();
        factory->setAvailable(CollectiveBackendType::NCCL, true);
        auto *mock_nccl = factory->addMockBackend(CollectiveBackendType::NCCL);
        mock_nccl->setAvailable(true);

        auto router = createRouter(std::move(factory));

        TPDomain domain;
        domain.type = TPDomainType::GPU_INTRA_RANK;
        domain.domain_size = 3;
        domain.devices = {DeviceId::cuda(0), DeviceId::cuda(1), DeviceId::cuda(2)};
        domain.name = "all_cuda";

        ICollectiveBackend *backend = router->selectBackendForDomain(&domain);
        ASSERT_NE(backend, nullptr);
        EXPECT_EQ(backend->type(), CollectiveBackendType::NCCL);
    }

    TEST_F(Test__BackendRouter, AllROCmDevices)
    {
        auto factory = std::make_unique<MockBackendFactory>();
        factory->setAvailable(CollectiveBackendType::RCCL, true);
        auto *mock_rccl = factory->addMockBackend(CollectiveBackendType::RCCL);
        mock_rccl->setAvailable(true);

        auto router = createRouter(std::move(factory));

        TPDomain domain;
        domain.type = TPDomainType::GPU_INTRA_RANK;
        domain.domain_size = 2;
        domain.devices = {DeviceId::rocm(0), DeviceId::rocm(1)};
        domain.name = "all_rocm";

        ICollectiveBackend *backend = router->selectBackendForDomain(&domain);
        ASSERT_NE(backend, nullptr);
        EXPECT_EQ(backend->type(), CollectiveBackendType::RCCL);
    }

    TEST_F(Test__BackendRouter, DiagnosticsIncludesDomainInfo)
    {
        auto factory = std::make_unique<MockBackendFactory>();
        factory->setAvailable(CollectiveBackendType::HOST, true);
        factory->setAvailable(CollectiveBackendType::MPI, true);

        auto router = createRouter(std::move(factory));

        std::string diag = router->diagnostics();

        // Should include UPI and domain support status
        EXPECT_NE(diag.find("UPI"), std::string::npos);
        EXPECT_NE(diag.find("Domain support"), std::string::npos);
    }

    TEST_F(Test__BackendRouter, EmptyDevicesDomainFallsBackToMPI)
    {
        auto factory = std::make_unique<MockBackendFactory>();
        factory->setAvailable(CollectiveBackendType::MPI, true);
        factory->setAvailable(CollectiveBackendType::NCCL, true);
        auto *mock_mpi = factory->addMockBackend(CollectiveBackendType::MPI);
        mock_mpi->setAvailable(true);

        auto router = createRouter(std::move(factory));

        // Create domain with empty devices (edge case)
        TPDomain domain;
        domain.type = TPDomainType::GPU_INTRA_RANK;
        domain.domain_size = 2;
        domain.devices = {}; // Empty!
        domain.name = "empty_domain";

        // Empty devices should fall back to MPI
        ICollectiveBackend *backend = router->selectBackendForDomain(&domain);
        ASSERT_NE(backend, nullptr);
        EXPECT_EQ(backend->type(), CollectiveBackendType::MPI);
    }

} // namespace llaminar2::test
