/**
 * @file NativeStoragePreparationPmaBinding.h
 * @brief Exact SDK preparation roots and one-shot final admission in the same PMA.
 *
 * Native source containers claim CPU0 backing before geometry production. Their
 * immutable creator declarations enter the original BOM before optional fill.
 * Final native callbacks inherit the same epoch, physical state and root leases.
 */
#pragma once
#include "NativeStoragePmaBinding.h"
#include <algorithm>
#include <atomic>
#include <new>
#include <span>
#include <vector>
namespace llaminar2
{
    namespace test
    {
        struct NativeStoragePreparationReplayAccess;
    }
    class NativeStorageSessionPreparation;
    /** @brief Source SDK creator role; aliases never contribute chargeable declarations. */
    enum class NativeStorageCreatorUse
    {
        PreparationOwner,
        RetainedOwner
    };
    /** @brief One actual SDK creator descriptor, not an estimated endpoint subtotal. */
    struct NativeStorageCreatorDeclaration
    {
        NativeStorageCreatorUse use;
        llaminarNativeRootRequestV1 geometry;
    };

    /** @brief Immutable authenticated source-native mandatory BOM created by SDK visitor
     * projection. */
    class NativeMandatoryStorageDeclarations final
    {
    public:
        NativeMandatoryStorageDeclarations(const NativeMandatoryStorageDeclarations &) = delete;
        NativeMandatoryStorageDeclarations &operator=(const NativeMandatoryStorageDeclarations &) =
            delete;
        NativeMandatoryStorageDeclarations(NativeMandatoryStorageDeclarations &&) noexcept =
            default;
        NativeMandatoryStorageDeclarations &operator=(
            NativeMandatoryStorageDeclarations &&) noexcept = default;
        /**
         * @brief Project exact creator records against actual preparation leases once.
         * @param snapshot Closed original-state preparation snapshot.
         * @param creators SDK-issued creator-only declarations; borrowed edges have no bytes here.
         * @return Mandatory native plan including preparation backing exactly once.
         * @throws Before allocation/publication for stale, duplicate or forged geometry/anchors.
         */
        static NativeMandatoryStorageDeclarations fromCreators(
            const NativePreparationPlanningSnapshot &snapshot,
            std::span<const NativeStorageCreatorDeclaration> creators)
        {
            if (!snapshot.valid())
                throw std::invalid_argument(
                    "Native SDK declarations require their closed preparation snapshot");
            NativeMandatoryStorageDeclarations result(snapshot.snapshot_);
            PhysicalMemoryPlanBuilder builder(snapshot.retainedPreparationBOM());
            std::vector<bool> matched(snapshot.declarations().size(), false);
            std::vector<std::uint64_t> anchors;
            anchors.reserve(creators.size());
            for (const auto &creator : creators)
            {
                auto request = creator.geometry;
                if (request.owner_identity || request.setup_epoch ||
                    !request.shared_anchor_identity ||
                    std::find(anchors.begin(), anchors.end(), request.shared_anchor_identity) !=
                        anchors.end())
                    throw std::invalid_argument("SDK creator must be unbound, unique and name its "
                                                "actual retained root cell");
                anchors.push_back(request.shared_anchor_identity);
                request.owner_identity = request.shared_anchor_identity;
                if (!NativeStoragePmaBinding::validGeometry(request) ||
                    request.owner_family < llaminarNativeCommMetadataV1 ||
                    request.owner_family > llaminarNativeDriverContextV1 ||
                    request.lifetime < llaminarNativeCommLifetimeV1 ||
                    request.lifetime > llaminarNativeProcessLifetimeV1 ||
                    (request.phase != llaminarNativeColdProbeV1 &&
                     request.phase != llaminarNativeRetainedSetupV1 &&
                     request.phase != llaminarNativeGraphRecordingHostMetadataV1))
                    throw std::invalid_argument(
                        "SDK creator physical declaration is malformed/incomplete");
                const auto declaration = convert(request);
                if (creator.use == NativeStorageCreatorUse::PreparationOwner)
                {
                    const auto originals = snapshot.declarations();
                    const auto found = std::find_if(
                        originals.begin(), originals.end(), [&](const auto &d)
                        { return d.shared_anchor_identity == declaration.shared_anchor_identity; });
                    if (found == originals.end() || !found->samePreparedGeometry(declaration))
                        throw std::logic_error("SDK preparation creator does not match its exact "
                                               "original lease/anchor declaration");
                    matched[std::size_t(found - originals.begin())] = true;
                }
                else if (creator.use == NativeStorageCreatorUse::RetainedOwner)
                {
                    if (std::any_of(snapshot.declarations().begin(), snapshot.declarations().end(),
                                    [&](const auto &d)
                                    {
                                        return d.shared_anchor_identity ==
                                               declaration.shared_anchor_identity;
                                    }))
                        throw std::logic_error("Retained SDK creator cannot charge existing "
                                               "preparation backing twice");
                    const auto observed = snapshot.initialObservations();
                    const auto found =
                        std::find_if(observed.begin(), observed.end(),
                                     [&](const auto &r)
                                     {
                                         return r.world_rank == snapshot.worldRank() &&
                                                r.device == declaration.device;
                                     });
                    if (found == observed.end())
                        throw std::out_of_range(
                            "SDK creator names an unobserved physical allocator");
                    builder.add(*found,
                                declaration.device.is_gpu()
                                    ? PhysicalMemoryOwner::NativeExecutionContext
                                    : PhysicalMemoryOwner::LocalCollective,
                                declaration.physical_bytes);
                    auto frozen = declaration;
                    frozen.owner_identity = 0;
                    result.future_.push_back(frozen);
                }
                else
                    throw std::invalid_argument("SDK creator has an invalid source lifetime role");
            }
            if (std::find(matched.begin(), matched.end(), false) != matched.end())
                throw std::logic_error(
                    "SDK creator visitor omitted an actual retained preparation owner");
            result.mandatory_ = builder.build();
            return result;
        }
        /** @return Exact source-native plan merged before optional expert/table fill. */
        [[nodiscard]] const PhysicalMemoryPlan &mandatoryBOM() const noexcept
        {
            return mandatory_;
        }

    private:
        friend class NativeStoragePreparationPmaBinding;
        /** @brief Retain only immutable source declarations and snapshot identity. */
        explicit NativeMandatoryStorageDeclarations(
            std::shared_ptr<const detail::PhysicalMemoryPreparationSnapshotState> value)
            : snapshot_(std::move(value))
        {
        }
        /** @brief Translate frozen common ABI geometry into the core's explicit source declaration.
         */
        static PhysicalMemoryNativeRootDeclaration convert(const llaminarNativeRootRequestV1 &r)
        {
            return NativeStoragePmaBinding::coreDeclaration(r);
        }
        std::shared_ptr<const detail::PhysicalMemoryPreparationSnapshotState> snapshot_;
        PhysicalMemoryPlan mandatory_;
        std::vector<PhysicalMemoryNativeRootDeclaration> future_;
    };

    /** @brief Stable native startup context; it is never a model allocation or second balance. */
    class NativeStoragePreparationPmaBinding final
    {
    public:
        NativeStoragePreparationPmaBinding(const NativeStoragePreparationPmaBinding &) = delete;
        NativeStoragePreparationPmaBinding &operator=(const NativeStoragePreparationPmaBinding &) =
            delete;
        NativeStoragePreparationPmaBinding(NativeStoragePreparationPmaBinding &&) = delete;
        NativeStoragePreparationPmaBinding &operator=(NativeStoragePreparationPmaBinding &&) =
            delete;
        /** @brief Abandon only unadmitted startup; successful transfer cannot close the final
         * epoch. */
        ~NativeStoragePreparationPmaBinding()
        {
            if (!transferred_)
                try
                {
                    epoch_.cancelPreparation();
                }
                catch (...)
                {
                    std::terminate();
                }
        }
        /** @return Same actual PMA state before/after final publication; ordinary methods are
         * phase-guarded. */
        [[nodiscard]] const std::shared_ptr<PhysicalMemoryAuthority> &memory() const noexcept
        {
            return memory_;
        }
        /** @return Explicit CPU0-only SDK preparation callbacks, permanently closed after final
         * admission. */
        [[nodiscard]] llaminarNativeStorageAuthorityV1 preparationAuthority() noexcept
        {
            return {1,
                    sizeof(llaminarNativeStorageAuthorityV1),
                    this,
                    claim,
                    NativeStoragePmaBinding::materialize,
                    NativeStoragePmaBinding::release,
                    epoch_.identity(),
                    0};
        }
        /** @return Actual session gate including preparation containers through final source/alias
         * retirement. */
        [[nodiscard]] NativeStorageRetirementGate retirementGate() const
        {
            return NativeStorageRetirementGate(epoch_);
        }
        /** @brief Freeze source preparation roots before mandatory geometry and optional placement.
         */
        [[nodiscard]] NativePreparationPlanningSnapshot closeForPlanning()
        {
            return NativePreparationPlanningSnapshot(epoch_.closePreparationForPlanning());
        }
        /**
         * @brief Commit final source admission without changing state, epoch or root ownership.
         * @param snapshot Original closed preparation snapshot; consumed only after successful
         * commit.
         * @param mandatory Exact SDK creator declarations merged before optional model/table fill.
         * @param certificate Complete final certificate using the original physical observations.
         * @return Prebuilt final callback owner, armed only by the noexcept commit section.
         * @throws Leaves counters, tokens, snapshot and epoch unchanged on every precommit failure.
         */
        std::shared_ptr<NativeStoragePmaBinding> admit(
            NativePreparationPlanningSnapshot &&snapshot,
            NativeMandatoryStorageDeclarations &&mandatory,
            std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> certificate)
        {
            if (transferred_ || !snapshot.valid() || mandatory.snapshot_ != snapshot.snapshot_ ||
                snapshot.epochIdentity() != epoch_.identity())
                throw std::logic_error(
                    "Final native admission requires the original unconsumed preparation identity");
            auto reservations = epoch_.prepareNativeAdmission(snapshot.snapshot_,
                                                              mandatory.mandatory_, certificate);
            std::vector<NativeStorageDeclaredReservation> declared;
            declared.reserve(reservations.size());
            // The final owner and its future source reservations are fully built
            // disarmed. No original epoch move or counter mutation has occurred.
            for (auto &reservation : reservations)
                declared.push_back(
                    {reservation.allocatorIdentity().device, std::move(reservation)});
            auto final = std::shared_ptr<NativeStoragePmaBinding>(
                new NativeStoragePmaBinding(epoch_, std::move(declared)));
            std::vector<PhysicalMemoryOwnerReservation> commitHandles;
            commitHandles.reserve(final->declared_.size());
            // Commit observes the exact pending states retained by final's native
            // owner. A temporary view copies handles privately without allocating
            // any new reservation/capacity or exposing them to ordinary callers.
            for (const auto &item : final->declared_)
                commitHandles.push_back(clonePending(item.reservation));
            auto permits = epoch_.prepareNativePermits(mandatory.future_, commitHandles);
            std::vector<PhysicalMemoryNativeRootDeclaration> retainedPreparation(
                snapshot.declarations().begin(), snapshot.declarations().end());
            if (fail_allocation_before_commit_)
                throw std::bad_alloc();
            struct Transfer
            {
                NativeStoragePreparationPmaBinding *source;
                NativeStoragePmaBinding *destination;
            } transfer{this, final.get()};
            epoch_.publishNativeAdmission(
                snapshot.snapshot_, mandatory.mandatory_, std::move(certificate), commitHandles,
                std::move(permits), std::move(retainedPreparation), &transfer,
                [](void *context) noexcept
                {
                    auto &value = *static_cast<Transfer *>(context);
                    value.destination->armed_ = true;
                    value.source->transferred_.store(true, std::memory_order_release);
                });
            snapshot.snapshot_.reset();
            mandatory.snapshot_.reset();
            return final;
        }
        /** @brief Cancel failed early setup without forgetting actual native physical roots. */
        void cancel()
        {
            if (transferred_)
                throw std::logic_error("Final native session owns admitted epoch");
            epoch_.cancelPreparation();
        }

    private:
        friend class NativeStorageSessionPreparation;
        friend struct test::NativeStoragePreparationReplayAccess;
        /**
         * @brief Create the one initially Preparing authority from typed native startup
         * infrastructure.
         * @param observations Original immutable physical observations, including local CPU0.
         * @param rank Physical MPI rank owning this native session's allocators.
         * @throws std::invalid_argument For invalid/duplicate observations or a missing local CPU0.
         */
        NativeStoragePreparationPmaBinding(std::span<const PhysicalMemoryResource> observations,
                                           int rank)
            : memory_(new PhysicalMemoryAuthority(
                  observations, rank, PhysicalMemoryAuthority::PreparingNativeStorageTag{})),
              epoch_(memory_->ledger_.beginPreparationEpoch(rank))
        {
        }
        /**
         * @brief Retain an unpublished reservation state for the final atomic commit view.
         * @param reservation Exact source owner already prebuilt for this same admission bundle.
         * @return Another private lifetime handle, with no new reserve or counter mutation.
         */
        static PhysicalMemoryOwnerReservation clonePending(
            const PhysicalMemoryOwnerReservation &reservation)
        {
            return PhysicalMemoryOwnerReservation(reservation.state_);
        }
        /**
         * @brief Preclaim an exact CPU0 SDK root in the original Preparing purpose.
         * @param context Stable session-owned preparation callback context.
         * @param request Actual source allocator declaration and original epoch identity.
         * @param output Actual native root's vacant token field; unchanged on rejection.
         * @return Fixed C ABI status; no exception crosses the native dependency boundary.
         */
        static int32_t claim(void *context, const llaminarNativeRootRequestV1 *request,
                             void **output) noexcept
        {
            if (!context || !request || !output || *output ||
                !NativeStoragePmaBinding::validGeometry(*request))
                return llaminarNativeStorageInvalidV1;
            auto &binding = *static_cast<NativeStoragePreparationPmaBinding *>(context);
            if (binding.transferred_ || request->setup_epoch != binding.epoch_.identity())
                return llaminarNativeStorageInvalidV1;
            if (request->owner_family < llaminarNativeCommMetadataV1 ||
                request->owner_family > llaminarNativeDriverContextV1 ||
                request->lifetime < llaminarNativeCommLifetimeV1 ||
                request->lifetime > llaminarNativeProcessLifetimeV1 ||
                (request->phase != llaminarNativeColdProbeV1 &&
                 request->phase != llaminarNativeRetainedSetupV1))
                return llaminarNativeStorageInvalidV1;
            try
            {
                const auto declaration = NativeMandatoryStorageDeclarations::convert(*request);
                if (declaration.device != DeviceId::cpu())
                    return llaminarNativeStorageUnsupportedV1;
                auto lease = binding.epoch_.claimPreparation(declaration);
                auto token =
                    std::make_unique<NativeStoragePmaBinding::Token>(std::move(lease), *request);
                *output = token.release();
                return llaminarNativeStorageSuccessV1;
            }
            catch (const PhysicalMemoryCapacityExhausted &)
            {
                return llaminarNativeStorageCapacityV1;
            }
            catch (const std::out_of_range &)
            {
                return llaminarNativeStorageUnsupportedV1;
            }
            catch (const std::logic_error &)
            {
                return llaminarNativeStorageInvalidV1;
            }
            catch (...)
            {
                return llaminarNativeStorageInternalV1;
            }
        }
        std::shared_ptr<PhysicalMemoryAuthority> memory_;
        PhysicalMemoryNativeSetupEpoch epoch_;
        std::atomic<bool> transferred_{false};
        bool fail_allocation_before_commit_ =
            false; ///< Test-friend injection at the real last allocation boundary.
    };
} // namespace llaminar2
