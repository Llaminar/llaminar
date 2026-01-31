/**
 * @file Test__DeviceGroup.cpp
 * @brief Unit tests for DeviceGroup and DeviceGroupBuilder
 *
 * Tests the collective infrastructure's device group management:
 * - DeviceGroupBuilder for constructing groups
 * - DeviceGroup predicates (allCUDA, allROCm, hasGPU, etc.)
 * - Test helper functions from CollectiveTestMocks.h
 */

#include <gtest/gtest.h>
#include "v2/collective/DeviceGroup.h"
#include "v2/collective/test/CollectiveTestMocks.h"
#include "v2/backends/DeviceId.h"

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // DeviceGroupBuilder Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(Test__DeviceGroup, BuilderCreatesCUDAGroup)
    {
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("cuda_gpus")
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::cuda(1))
                         .addDevice(DeviceId::cuda(2))
                         .addDevice(DeviceId::cuda(3))
                         .setLocalRank(0)
                         .build();

        EXPECT_EQ(group.name, "cuda_gpus");
        EXPECT_EQ(group.size(), 4);
        EXPECT_EQ(group.cuda_count, 4);
        EXPECT_EQ(group.rocm_count, 0);
        EXPECT_EQ(group.cpu_count, 0);
        EXPECT_TRUE(group.is_homogeneous);
        EXPECT_EQ(group.primary_type, DeviceType::CUDA);
    }

    TEST(Test__DeviceGroup, BuilderCreatesROCmGroup)
    {
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("rocm_gpus")
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .setLocalRank(1)
                         .build();

        EXPECT_EQ(group.name, "rocm_gpus");
        EXPECT_EQ(group.size(), 2);
        EXPECT_EQ(group.cuda_count, 0);
        EXPECT_EQ(group.rocm_count, 2);
        EXPECT_EQ(group.cpu_count, 0);
        EXPECT_TRUE(group.is_homogeneous);
        EXPECT_EQ(group.primary_type, DeviceType::ROCm);
        EXPECT_EQ(group.local_rank, 1);
    }

    TEST(Test__DeviceGroup, BuilderCreatesHeterogeneousGroup)
    {
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("mixed_devices")
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::cuda(1))
                         .addDevice(DeviceId::cpu())
                         .setLocalRank(0)
                         .build();

        EXPECT_EQ(group.name, "mixed_devices");
        EXPECT_EQ(group.size(), 3);
        EXPECT_EQ(group.cuda_count, 2);
        EXPECT_EQ(group.rocm_count, 0);
        EXPECT_EQ(group.cpu_count, 1);
        EXPECT_FALSE(group.is_homogeneous);
        // Primary type should be CUDA (most common)
        EXPECT_EQ(group.primary_type, DeviceType::CUDA);
    }

    TEST(Test__DeviceGroup, BuilderComputesPrimaryTypeCorrectly)
    {
        // Case 1: More ROCm than CUDA
        {
            DeviceGroupBuilder builder;
            auto group = builder
                             .setName("rocm_majority")
                             .addDevice(DeviceId::cuda(0))
                             .addDevice(DeviceId::rocm(0))
                             .addDevice(DeviceId::rocm(1))
                             .addDevice(DeviceId::rocm(2))
                             .build();

            EXPECT_FALSE(group.is_homogeneous);
            EXPECT_EQ(group.primary_type, DeviceType::ROCm);
            EXPECT_EQ(group.cuda_count, 1);
            EXPECT_EQ(group.rocm_count, 3);
        }

        // Case 2: More CPU than GPU
        {
            DeviceGroupBuilder builder;
            auto group = builder
                             .setName("cpu_majority")
                             .addDevice(DeviceId::cpu())
                             .addDevice(DeviceId::cpu())
                             .addDevice(DeviceId::cpu())
                             .addDevice(DeviceId::cuda(0))
                             .build();

            EXPECT_FALSE(group.is_homogeneous);
            EXPECT_EQ(group.primary_type, DeviceType::CPU);
            EXPECT_EQ(group.cpu_count, 3);
            EXPECT_EQ(group.cuda_count, 1);
        }
    }

    TEST(Test__DeviceGroup, BuilderAddDevicesFromVector)
    {
        std::vector<DeviceId> devices = {
            DeviceId::cuda(0),
            DeviceId::cuda(1),
            DeviceId::cuda(2)};

        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("vector_group")
                         .addDevices(devices)
                         .setLocalRank(2)
                         .build();

        EXPECT_EQ(group.size(), 3);
        EXPECT_EQ(group.cuda_count, 3);
        EXPECT_TRUE(group.is_homogeneous);
        EXPECT_EQ(group.local_rank, 2);
    }

    TEST(Test__DeviceGroup, BuilderSetsScope)
    {
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("global_group")
                         .addDevice(DeviceId::cuda(0))
                         .setScope(CollectiveScope::GLOBAL)
                         .build();

        EXPECT_EQ(group.scope, CollectiveScope::GLOBAL);
        EXPECT_TRUE(group.isGlobal());
        EXPECT_FALSE(group.isLocal());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // DeviceGroup Predicate Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(Test__DeviceGroup, AllCUDAReturnsCorrectly)
    {
        // True case: All CUDA
        {
            auto group = createTestCUDAGroup(4);
            EXPECT_TRUE(group.allCUDA());
            EXPECT_FALSE(group.allROCm());
            EXPECT_FALSE(group.allCPU());
        }

        // False case: Mixed with CPU
        {
            auto group = createTestHeterogeneousGroup();
            EXPECT_FALSE(group.allCUDA());
        }
    }

    TEST(Test__DeviceGroup, AllROCmReturnsCorrectly)
    {
        // True case: All ROCm
        {
            DeviceGroupBuilder builder;
            auto group = builder
                             .setName("rocm_only")
                             .addDevice(DeviceId::rocm(0))
                             .addDevice(DeviceId::rocm(1))
                             .build();

            EXPECT_TRUE(group.allROCm());
            EXPECT_FALSE(group.allCUDA());
            EXPECT_FALSE(group.allCPU());
        }

        // False case: Mixed with CUDA
        {
            DeviceGroupBuilder builder;
            auto group = builder
                             .setName("mixed")
                             .addDevice(DeviceId::rocm(0))
                             .addDevice(DeviceId::cuda(0))
                             .build();

            EXPECT_FALSE(group.allROCm());
        }
    }

    TEST(Test__DeviceGroup, AllCPUReturnsCorrectly)
    {
        // True case: All CPU
        {
            DeviceGroupBuilder builder;
            auto group = builder
                             .setName("cpu_only")
                             .addDevice(DeviceId::cpu())
                             .addDevice(DeviceId::cpu())
                             .build();

            EXPECT_TRUE(group.allCPU());
            EXPECT_FALSE(group.allCUDA());
            EXPECT_FALSE(group.allROCm());
        }

        // False case: Has GPU
        {
            auto group = createTestHeterogeneousGroup();
            EXPECT_FALSE(group.allCPU());
        }
    }

    TEST(Test__DeviceGroup, HasGPUReturnsCorrectly)
    {
        // True case: Has CUDA
        {
            auto group = createTestCUDAGroup(2);
            EXPECT_TRUE(group.hasGPU());
        }

        // True case: Has ROCm
        {
            DeviceGroupBuilder builder;
            auto group = builder
                             .setName("rocm")
                             .addDevice(DeviceId::rocm(0))
                             .build();

            EXPECT_TRUE(group.hasGPU());
        }

        // True case: Mixed
        {
            auto group = createTestHeterogeneousGroup();
            EXPECT_TRUE(group.hasGPU());
        }

        // False case: CPU only
        {
            DeviceGroupBuilder builder;
            auto group = builder
                             .setName("cpu_only")
                             .addDevice(DeviceId::cpu())
                             .build();

            EXPECT_FALSE(group.hasGPU());
        }
    }

    TEST(Test__DeviceGroup, IsHeterogeneousReturnsCorrectly)
    {
        // True case: Mixed types
        {
            auto group = createTestHeterogeneousGroup();
            EXPECT_TRUE(group.isHeterogeneous());
            EXPECT_FALSE(group.is_homogeneous);
        }

        // False case: All same type
        {
            auto group = createTestCUDAGroup(4);
            EXPECT_FALSE(group.isHeterogeneous());
            EXPECT_TRUE(group.is_homogeneous);
        }
    }

    TEST(Test__DeviceGroup, LocalDeviceReturnsCorrectDevice)
    {
        // Test with local_rank = 0
        {
            auto group = createTestCUDAGroup(4, 0);
            auto local = group.localDevice();
            EXPECT_TRUE(local.is_cuda());
            EXPECT_EQ(local.ordinal, 0);
        }

        // Test with local_rank = 2
        {
            auto group = createTestCUDAGroup(4, 2);
            auto local = group.localDevice();
            EXPECT_TRUE(local.is_cuda());
            EXPECT_EQ(local.ordinal, 2);
        }

        // Test heterogeneous group with local_rank pointing to CPU
        {
            auto group = createTestHeterogeneousGroup();
            // createTestHeterogeneousGroup has [cuda(0), cuda(1), cpu()], local_rank=0
            // Change local_rank to 2 to get CPU
            DeviceGroupBuilder builder;
            auto hetero = builder
                              .setName("hetero")
                              .addDevice(DeviceId::cuda(0))
                              .addDevice(DeviceId::cuda(1))
                              .addDevice(DeviceId::cpu())
                              .setLocalRank(2)
                              .build();

            auto local = hetero.localDevice();
            EXPECT_TRUE(local.is_cpu());
        }
    }

    TEST(Test__DeviceGroup, LocalDeviceReturnsDefaultForInvalidRank)
    {
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("test")
                         .addDevice(DeviceId::cuda(0))
                         .setLocalRank(5) // Invalid: only 1 device
                         .build();

        // Should return CPU as fallback
        auto local = group.localDevice();
        EXPECT_TRUE(local.is_cpu());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Scope Predicate Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(Test__DeviceGroup, ScopePredicatesWork)
    {
        // Local scope
        {
            DeviceGroupBuilder builder;
            auto group = builder
                             .setName("local")
                             .addDevice(DeviceId::cuda(0))
                             .setScope(CollectiveScope::LOCAL)
                             .build();

            EXPECT_TRUE(group.isLocal());
            EXPECT_FALSE(group.isGlobal());
        }

        // Global scope
        {
            DeviceGroupBuilder builder;
            auto group = builder
                             .setName("global")
                             .addDevice(DeviceId::cuda(0))
                             .setScope(CollectiveScope::GLOBAL)
                             .build();

            EXPECT_FALSE(group.isLocal());
            EXPECT_TRUE(group.isGlobal());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Validation Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(Test__DeviceGroup, ValidationWorks)
    {
        // Valid group
        {
            auto group = createTestCUDAGroup(2, 0);
            EXPECT_TRUE(group.isValid());
        }

        // Invalid: empty devices
        {
            DeviceGroup empty;
            empty.local_rank = 0;
            EXPECT_FALSE(empty.isValid());
        }

        // Invalid: negative local_rank
        {
            DeviceGroupBuilder builder;
            auto group = builder
                             .setName("test")
                             .addDevice(DeviceId::cuda(0))
                             .setLocalRank(-1)
                             .build();

            EXPECT_FALSE(group.isValid());
        }

        // Invalid: local_rank >= size
        {
            DeviceGroupBuilder builder;
            auto group = builder
                             .setName("test")
                             .addDevice(DeviceId::cuda(0))
                             .setLocalRank(1) // Only 1 device, so rank 1 is invalid
                             .build();

            EXPECT_FALSE(group.isValid());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ToString Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(Test__DeviceGroup, ToStringFormatsCorrectly)
    {
        auto group = createTestCUDAGroup(2, 0);
        std::string str = group.toString();

        // Should contain the name and device info
        EXPECT_NE(str.find("test_cuda_group"), std::string::npos);
        EXPECT_NE(str.find("CUDA"), std::string::npos);
        // Local device should be marked with *
        EXPECT_NE(str.find("*"), std::string::npos);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Helper Function Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(Test__DeviceGroup, CreateTestCUDAGroupWorks)
    {
        auto group = createTestCUDAGroup(4, 2);

        EXPECT_EQ(group.name, "test_cuda_group");
        EXPECT_EQ(group.size(), 4);
        EXPECT_EQ(group.local_rank, 2);
        EXPECT_TRUE(group.allCUDA());
        EXPECT_TRUE(group.is_homogeneous);
        EXPECT_EQ(group.cuda_count, 4);
        EXPECT_TRUE(group.isValid());

        // Verify device ordinals
        for (int i = 0; i < 4; ++i)
        {
            EXPECT_TRUE(group.devices[i].is_cuda());
            EXPECT_EQ(group.devices[i].ordinal, i);
        }
    }

    TEST(Test__DeviceGroup, CreateTestHeterogeneousGroupWorks)
    {
        auto group = createTestHeterogeneousGroup();

        EXPECT_EQ(group.name, "test_hetero_group");
        EXPECT_EQ(group.size(), 3);
        EXPECT_EQ(group.local_rank, 0);
        EXPECT_FALSE(group.is_homogeneous);
        EXPECT_TRUE(group.isHeterogeneous());
        EXPECT_EQ(group.cuda_count, 2);
        EXPECT_EQ(group.cpu_count, 1);
        EXPECT_TRUE(group.hasGPU());
        EXPECT_FALSE(group.allCUDA());
        EXPECT_FALSE(group.allCPU());
        EXPECT_TRUE(group.isValid());

        // Verify device order: cuda(0), cuda(1), cpu()
        EXPECT_TRUE(group.devices[0].is_cuda());
        EXPECT_EQ(group.devices[0].ordinal, 0);
        EXPECT_TRUE(group.devices[1].is_cuda());
        EXPECT_EQ(group.devices[1].ordinal, 1);
        EXPECT_TRUE(group.devices[2].is_cpu());
    }

    TEST(Test__DeviceGroup, CreateTestDeviceGroupWorks)
    {
        std::vector<DeviceId> devices = {
            DeviceId::rocm(0),
            DeviceId::rocm(1),
            DeviceId::cpu()};

        auto group = createTestDeviceGroup("custom_group", devices, 1);

        EXPECT_EQ(group.name, "custom_group");
        EXPECT_EQ(group.size(), 3);
        EXPECT_EQ(group.local_rank, 1);
        EXPECT_EQ(group.rocm_count, 2);
        EXPECT_EQ(group.cpu_count, 1);
        EXPECT_FALSE(group.is_homogeneous);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Edge Cases
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(Test__DeviceGroup, SingleDeviceGroup)
    {
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("single")
                         .addDevice(DeviceId::cuda(5))
                         .setLocalRank(0)
                         .build();

        EXPECT_EQ(group.size(), 1);
        EXPECT_TRUE(group.is_homogeneous);
        EXPECT_TRUE(group.allCUDA());
        EXPECT_TRUE(group.isValid());

        auto local = group.localDevice();
        EXPECT_TRUE(local.is_cuda());
        EXPECT_EQ(local.ordinal, 5);
    }

    TEST(Test__DeviceGroup, EmptyGroupFromBuilder)
    {
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("empty")
                         .build();

        EXPECT_EQ(group.size(), 0);
        EXPECT_FALSE(group.isValid());
        // Empty group is technically homogeneous (all 0 of 0 devices are same type)
        EXPECT_TRUE(group.is_homogeneous);
    }

    TEST(Test__DeviceGroup, MixedCUDAAndROCm)
    {
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("mixed_gpu")
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::cuda(1))
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .build();

        EXPECT_FALSE(group.is_homogeneous);
        EXPECT_TRUE(group.isHeterogeneous());
        EXPECT_EQ(group.cuda_count, 2);
        EXPECT_EQ(group.rocm_count, 2);
        EXPECT_EQ(group.cpu_count, 0);
        EXPECT_TRUE(group.hasGPU());
        EXPECT_FALSE(group.allCUDA());
        EXPECT_FALSE(group.allROCm());
        // Primary type is CUDA when tied (CUDA checked first in finalizeMetadata)
        EXPECT_EQ(group.primary_type, DeviceType::CUDA);
    }

} // namespace llaminar2::test
