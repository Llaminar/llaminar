/** @file NativeStoragePreparingReplayAccess.h
 * @brief Privileged test-only source entry and adversarial snapshot duplication.
 */
#pragma once
#include "collective/NativeStoragePreparationPmaBinding.h"
#include <type_traits>

namespace llaminar2::test
{
    /** @brief Prove ordinary code cannot open the private preparation phase. */
    static_assert(!std::is_constructible_v<NativeStoragePreparationPmaBinding,
                                          std::span<const PhysicalMemoryResource>, int>);
    /** @brief An aggregate memory quota cannot construct final native callbacks. */
    static_assert(!std::is_constructible_v<NativeStoragePmaBinding,
                                          std::shared_ptr<PhysicalMemoryAuthority>>);

    /** @brief Detect a public exact-root entrypoint without granting test friendship. */
    template <class Epoch>
    concept ExposesNativeRootClaim = requires(Epoch &epoch,
                                             const PhysicalMemoryNativeRootDeclaration &root)
    {
        epoch.claimPreparedNative(root);
    };
    static_assert(!ExposesNativeRootClaim<PhysicalMemoryNativeSetupEpoch>);

    /** @brief Test-only access to setup and failure injection, never backing creation. */
    struct NativeStoragePreparationReplayAccess
    {
        /**
         * @brief Open the actual canonical Preparing state for a device-free replay.
         * @param observations Fixed allocator identities and capacities for this replay.
         * @param rank Physical owner of the source session.
         * @return The same restricted binding used by native preparation infrastructure.
         */
        static std::unique_ptr<NativeStoragePreparationPmaBinding> begin(
            std::span<const PhysicalMemoryResource> observations, int rank = 0)
        {
            return std::unique_ptr<NativeStoragePreparationPmaBinding>(
                new NativeStoragePreparationPmaBinding(observations, rank));
        }
        /**
         * @brief Inject failure at the real final precommit allocation boundary.
         * @param binding Preparation owner whose next admission is exercised.
         * @param fail Whether admission throws before publishing any changed state.
         */
        static void failBeforeCommit(NativeStoragePreparationPmaBinding &binding, bool fail)
        {
            binding.fail_allocation_before_commit_ = fail;
        }
        /**
         * @brief Construct an adversarial second handle to the same immutable snapshot.
         * @param snapshot Closed source snapshot retained by a competing admission.
         * @return A second identity-equivalent handle; successful admission remains one-shot.
         */
        static NativePreparationPlanningSnapshot duplicate(
            const NativePreparationPlanningSnapshot &snapshot)
        {
            return NativePreparationPlanningSnapshot(snapshot.snapshot_);
        }
    };
} // namespace llaminar2::test
