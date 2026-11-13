/**
 * @file Test__NativeStorageResourcePrograms.cpp
 * @brief Exact native permits remain resource-local across composed inventories.
 *
 * These device-free programs exercise every supported native allocator family
 * and one-to-eight GPU inventories through the canonical preparation/admission
 * callbacks. They certify accounting and descriptor identity, not native SDK
 * creation, graph capture or the hardware availability of those topologies.
 */
#include "fixtures/planning/NativeStoragePreparingReplayAccess.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <deque>
#include <map>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Inventories have one CPU physical allocator regardless of NUMA placement. */
        enum class DeviceInventory
        {
            CPU,
            CUDA,
            ROCm,
            AlternatingVendors
        };

        /** @brief Own one opaque test root field until the canonical callback retires it. */
        struct NativeRootClaim final
        {
            /** @brief Bind the context-free retirement callback for this root field. */
            explicit NativeRootClaim(const llaminarNativeStorageAuthorityV1 &authority)
                : release(authority.release)
            {
            }
            NativeRootClaim(const NativeRootClaim &) = delete;
            NativeRootClaim &operator=(const NativeRootClaim &) = delete;
            /** @brief Keep assertion failures from losing pending or materialized roots. */
            ~NativeRootClaim()
            {
                if (token && release(&token) != llaminarNativeStorageSuccessV1)
                    std::terminate();
            }
            void *token = nullptr;
            int32_t (*release)(void **);
        };

        /**
         * @brief Build one exact creator declaration for the chosen allocator.
         * @param anchor Unique source root cell in this program.
         * @param device Physical GPU identity, or CPU0 for a host root.
         * @param allocator Explicit native allocator kind, never selected after failure.
         * @param numa Host placement only; all host roots share CPU0 accounting.
         * @return Unbound immutable creator geometry for 64 physical bytes.
         */
        llaminarNativeRootRequestV1 declaration(std::uint64_t anchor, DeviceId device,
                                                std::uint32_t allocator, int numa = -1)
        {
            llaminarNativeRootRequestV1 result{};
            result.abi_version = 1;
            result.record_bytes = sizeof(result);
            result.shared_anchor_identity = anchor;
            result.requested_bytes = 61;
            result.physical_bytes = result.allocation_granularity = 64;
            result.resource_kind = device.is_cpu() ? llaminarNativeHostCPU0V1
                                   : device.is_cuda() ? llaminarNativeCUDADeviceV1
                                                      : llaminarNativeROCmDeviceV1;
            result.resource_ordinal = device.ordinal;
            result.host_numa = numa;
            result.owner_family = llaminarNativeP2pBackingV1;
            result.lifetime = llaminarNativeCommLifetimeV1;
            result.allocator_kind = allocator;
            result.phase = llaminarNativeRetainedSetupV1;
            return result;
        }

        /**
         * @brief Replay one complete preparation, descriptor-consumption and retirement program.
         * @param inventory Vendor composition; it never opens a real backend.
         * @param count Number of GPUs, with zero reserved for the CPU-only program.
         */
        void replayResourceProgram(DeviceInventory inventory, int count)
        {
            SCOPED_TRACE(::testing::Message() << "inventory=" << int(inventory)
                                             << " devices=" << count);
            std::vector<PhysicalMemoryResource> observed{{0, DeviceId::cpu(), 4096, 2048}};
            std::vector<std::pair<DeviceId, llaminarNativeRootRequestV1>> roots;
            std::uint64_t nextAnchor = 0x1000;
            const auto addRoot = [&](DeviceId device, std::uint32_t allocator, int numa = -1)
            {
                roots.emplace_back(device, declaration(nextAnchor++, device, allocator, numa));
            };
            // NUMA is placement metadata, including for the registered-file root.
            addRoot(DeviceId::cpu(), llaminarNativeCUDAHostVmmV1);
            addRoot(DeviceId::cpu(), llaminarNativeROCmUncachedHostVmmV1, 0);
            addRoot(DeviceId::cpu(), llaminarNativeRegisteredHostFileV1, 1);
            addRoot(DeviceId::cpu(), llaminarNativeHostAnonymousPagesV1);
            for (int index = 0; index < count; ++index)
            {
                const bool cuda = inventory == DeviceInventory::CUDA ||
                                  (inventory == DeviceInventory::AlternatingVendors && index % 2 == 0);
                const int ordinal = inventory == DeviceInventory::AlternatingVendors ? index / 2 : index;
                const auto device = cuda ? DeviceId::cuda(ordinal) : DeviceId::rocm(ordinal);
                observed.push_back({0, device, 1024, 256});
                addRoot(device, cuda ? llaminarNativeCUDADeviceVmmV1 : llaminarNativeROCmDeviceVmmV1);
                if (!cuda)
                    addRoot(device, llaminarNativeROCmUncachedDeviceVmmV1);
            }
            // Frozen resource indices must be canonical, independent of discovery order.
            std::reverse(observed.begin(), observed.end());
            auto preparing = NativeStoragePreparationReplayAccess::begin(observed);
            auto memory = preparing->memory();
            auto early = preparing->preparationAuthority();
            auto container = declaration(0x100, DeviceId::cpu(), llaminarNativeHostAnonymousPagesV1);
            container.owner_family = llaminarNativeCommMetadataV1;
            container.owner_identity = 0x101;
            container.setup_epoch = early.setup_epoch;
            NativeRootClaim containerClaim(early);
            ASSERT_EQ(early.claim(early.context, &container, &containerClaim.token),
                      llaminarNativeStorageSuccessV1);
            ASSERT_EQ(early.materialize(containerClaim.token), llaminarNativeStorageSuccessV1);
            container.owner_identity = container.setup_epoch = 0;
            auto snapshot = preparing->closeForPlanning();
            std::vector<NativeStorageCreatorDeclaration> creators{
                {NativeStorageCreatorUse::PreparationOwner, container}};
            for (const auto &[device, root] : roots)
                creators.push_back({NativeStorageCreatorUse::RetainedOwner, root});
            auto mandatory = NativeMandatoryStorageDeclarations::fromCreators(snapshot, creators);
            auto certificate = std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(mandatory.mandatoryBOM());
            auto final = preparing->admit(std::move(snapshot), std::move(mandatory), certificate);
            auto authority = final->authority();
            EXPECT_EQ(memory.get(), preparing->memory().get());
            EXPECT_EQ(early.setup_epoch, authority.setup_epoch);
            EXPECT_TRUE(memory->rankCommitted());
            EXPECT_FALSE(memory->rankComplete());
            std::map<DeviceId, std::size_t> expected{{DeviceId::cpu(), 64}};
            std::deque<NativeRootClaim> claims;
            for (auto [device, root] : roots)
            {
                root.owner_identity = root.shared_anchor_identity + 0x10000;
                root.setup_epoch = authority.setup_epoch;
                NativeRootClaim invalid(authority);
                auto changed = root;
                if (device.is_gpu())
                {
                    // A valid opposite-vendor allocator must not reuse this source slot,
                    // including when that vendor/ordinal is present in the same program.
                    changed.resource_kind = device.is_cuda() ? llaminarNativeROCmDeviceV1 : llaminarNativeCUDADeviceV1;
                    changed.allocator_kind = device.is_cuda() ? llaminarNativeROCmDeviceVmmV1 : llaminarNativeCUDADeviceVmmV1;
                }
                else
                {
                    // An unplanned CPU ordinal cannot manufacture another host pool.
                    changed.resource_ordinal = 1;
                }
                EXPECT_NE(authority.claim(authority.context, &changed, &invalid.token),
                          llaminarNativeStorageSuccessV1);
                EXPECT_EQ(invalid.token, nullptr);
                auto &claim = claims.emplace_back(authority);
                ASSERT_EQ(authority.claim(authority.context, &root, &claim.token),
                          llaminarNativeStorageSuccessV1);
                ASSERT_EQ(authority.materialize(claim.token), llaminarNativeStorageSuccessV1);
                expected[device] += 64;
                for (const auto &resource : observed)
                {
                    const auto owner = resource.device.is_cpu() ? PhysicalMemoryOwner::LocalCollective
                                                                : PhysicalMemoryOwner::NativeExecutionContext;
                    EXPECT_EQ(memory->claimedBytes(resource.device, owner,
                                                   PhysicalMemoryMaterializationKind::NewAllocation),
                              expected[resource.device]);
                }
            }
            EXPECT_TRUE(memory->rankComplete());
            final->sealForExecution();
            auto retirement = final->retirementGate();
            EXPECT_FALSE(retirement.retired());
            claims.clear();
            EXPECT_FALSE(retirement.retired());
            ASSERT_EQ(early.release(&containerClaim.token), llaminarNativeStorageSuccessV1);
            EXPECT_TRUE(retirement.retired());
            for (const auto &row : memory->rankAttestation())
                EXPECT_EQ(row.materialized_new_bytes, 0);
        }
    } // namespace

    TEST(NativeStorageResourcePrograms, CpuCudaRocmAndComposedInventoriesThroughEightDevices)
    {
        replayResourceProgram(DeviceInventory::CPU, 0);
        for (const auto inventory : {DeviceInventory::CUDA, DeviceInventory::ROCm,
                                     DeviceInventory::AlternatingVendors})
            for (const int count : {1, 2, 4, 8})
                if (inventory != DeviceInventory::AlternatingVendors || count >= 2)
                    replayResourceProgram(inventory, count);
    }
} // namespace llaminar2::test
