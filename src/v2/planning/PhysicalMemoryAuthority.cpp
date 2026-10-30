/**
 * @file PhysicalMemoryAuthority.cpp
 * @brief Topology-wide CPU/GPU memory authority implementation.
 *
 * Immutable BOM admission, explicit capacity ceilings and live materialization
 * share this authority. Capacity exhaustion is a typed candidate rejection;
 * malformed ownership or accounting overflow cannot masquerade as a smaller
 * admissible plan. Live claims and reservations retain one synchronized ledger.
 */

#include "PhysicalMemoryAuthority.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    std::size_t PhysicalMemoryAuthority::admissionCapacity(
        std::size_t observed_available_bytes,
        std::optional<std::size_t> explicit_limit_bytes) noexcept
    {
        return explicit_limit_bytes ? std::min(observed_available_bytes, *explicit_limit_bytes)
                                    : observed_available_bytes;
    }
    namespace
    {
        /** @brief Add topology totals without permitting a wrapped admission. */
        [[nodiscard]] std::size_t checkedAdd(
            std::size_t left,
            std::size_t right,
            const char *what)
        {
            if (right > std::numeric_limits<std::size_t>::max() - left)
            {
                throw std::overflow_error(
                    std::string("Physical memory ") + what +
                    " overflows size_t");
            }
            return left + right;
        }

        /**
         * @brief Mint only lifecycle identity, never an allocation or pointer ledger.
         * Unique identities prevent a prior source's request from matching a new
         * source epoch even when its native owner address is later reused.
         */
        [[nodiscard]] std::uint64_t nextNativeSetupIdentity()
        {
            static std::atomic<std::uint64_t> next{1};
            auto value = next.load(std::memory_order_relaxed);
            for (;;)
            {
                if (value == std::numeric_limits<std::uint64_t>::max())
                    throw std::overflow_error("Native setup epoch identity exhausted");
                if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed,
                                               std::memory_order_relaxed))
                    return value;
            }
        }

        /** @brief Test only the typed identity, never volatile capacity fields. */
        [[nodiscard]] bool matches(
            const PhysicalMemoryResource &resource,
            PhysicalMemoryAllocatorIdentity identity) noexcept
        {
            return resource.world_rank == identity.world_rank &&
                   resource.device == identity.device;
        }

        /** @brief Deterministic rank/device ordering for reports and wire data. */
        [[nodiscard]] bool resourceLess(
            const PhysicalMemoryBOM &left,
            const PhysicalMemoryBOM &right) noexcept
        {
            if (left.resource().world_rank != right.resource().world_rank)
            {
                return left.resource().world_rank <
                       right.resource().world_rank;
            }
            return left.resource().device < right.resource().device;
        }

        /** @brief Convert one valid owner to its bounded array index. */
        [[nodiscard]] std::size_t ownerIndex(PhysicalMemoryOwner owner)
        {
            const auto index = static_cast<std::size_t>(owner);
            if (index >= PhysicalMemoryBOM::ownerCount())
            {
                throw std::invalid_argument(
                    "Physical memory materialization received an invalid owner");
            }
            return index;
        }
    } // namespace

    namespace detail
    {
        /** @brief Mutable live counters protected by one setup-time mutex. */
        struct PhysicalMemoryMaterializationState
        {
            struct ResourceClaims
            {
                // Direct leases describe fixed backing allocations.
                std::array<std::size_t, PhysicalMemoryBOM::ownerCount()>
                    new_allocations{};
                std::array<std::size_t, PhysicalMemoryBOM::ownerCount()>
                    adopted_residency{};

                // A lazy pool commits its complete capacity once. Child leases
                // change only the materialized counter and never double-charge
                // that already committed capacity.
                std::array<std::size_t, PhysicalMemoryBOM::ownerCount()>
                    new_reservation_capacity{};
                std::array<std::size_t, PhysicalMemoryBOM::ownerCount()>
                    new_reservation_materialized{};
                // Preparing and exact SDK root preclaims are typed columns of
                // this same authority. They never create another capacity balance.
                std::array<std::size_t, PhysicalMemoryBOM::ownerCount()>
                    native_declared_preclaimed{};
                std::array<std::size_t, PhysicalMemoryBOM::ownerCount()> preparation_committed{},
                    preparation_materialized{};
            };

            explicit PhysicalMemoryMaterializationState(
                std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> admitted)
                : certificate(std::move(admitted)),
                  observations(
                      [&]
                      {
                          std::vector<PhysicalMemoryResource> result;
                          for (const auto &bom : certificate->plan().resources())
                              result.push_back(bom.resource());
                          return result;
                      }()),
                  claims(observations.size())
            {
            }

            /** @brief Preparing owns only immutable observations and the existing canonical
             * counters. */
            explicit PhysicalMemoryMaterializationState(
                std::span<const PhysicalMemoryResource> initial)
                : observations(
                      [&]
                      {
                          std::vector<PhysicalMemoryResource> result(initial.begin(),
                                                                     initial.end());
                          std::sort(result.begin(), result.end(),
                                    [](const auto &a, const auto &b)
                                    {
                                        return a.world_rank == b.world_rank
                                                   ? a.device < b.device
                                                   : a.world_rank < b.world_rank;
                                    });
                          return result;
                      }()),
                  phase(PhysicalMemoryAuthorityPhase::Preparing)
            {
                if (observations.empty())
                    throw std::invalid_argument(
                        "Native preparation needs original physical observations");

                for (std::size_t i = 0; i < observations.size(); ++i)
                    if (!observations[i].valid() ||
                        (observations[i].device.is_cpu() &&
                         observations[i].device != DeviceId::cpu()) ||
                        (i && observations[i].world_rank == observations[i - 1].world_rank &&
                         observations[i].device == observations[i - 1].device))
                        throw std::invalid_argument(
                            "Native preparation physical observation is invalid/duplicated");
                claims.resize(observations.size());
            }

            std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate>
                certificate;
            const std::vector<PhysicalMemoryResource> observations;
            std::vector<ResourceClaims> claims;
            PhysicalMemoryAuthorityPhase phase = PhysicalMemoryAuthorityPhase::Admitted;
            bool preparation_frozen = false;
            mutable std::mutex mutex;
        };

        /** @brief Epoch lifecycle facts; all mutation holds authority->mutex. */
        struct PhysicalMemoryNativeSetupEpochState
        {
            std::shared_ptr<PhysicalMemoryMaterializationState> authority;
            int world_rank = -1;
            std::uint64_t identity = 0;
            enum class Phase
            {
                Preparation,
                HostRecording,
                Sealed,
                Abandoned
            };
            Phase phase = Phase::Preparation;
            // Counts authenticate setup/publication lifetime, not physical bytes.
            std::size_t unpublished_roots = 0;
            std::size_t live_roots = 0;
            PhysicalMemoryPreparationRootState *preparation_roots = nullptr;
            std::vector<std::shared_ptr<PhysicalMemoryNativeRootPermitState>> native_permits;
            std::vector<PhysicalMemoryNativeRootDeclaration> admitted_preparation;
        };

        /** @brief Intrusive immutable declaration attached to its actual unique root lease. */
        struct PhysicalMemoryPreparationRootState
        {
            explicit PhysicalMemoryPreparationRootState(PhysicalMemoryNativeRootDeclaration value)
                : declaration(std::move(value))
            {
            }
            const PhysicalMemoryNativeRootDeclaration declaration;
            PhysicalMemoryPreparationRootState *previous = nullptr;
            PhysicalMemoryPreparationRootState *next = nullptr;
        };

        /** @brief Closed immutable planning value, never another authority or live counter ledger.
         */
        struct PhysicalMemoryPreparationSnapshotState
        {
            std::shared_ptr<PhysicalMemoryNativeSetupEpochState> epoch;
            PhysicalMemoryPlan retained;
            std::vector<PhysicalMemoryNativeRootDeclaration> declarations;
        };

        /**
         * @brief Shared lifetime state retained by a pool and all child leases.
         *
         * Every mutable field is protected by `authority->mutex`. The final
         * shared reference can disappear only after the public pool handle and
         * every child lease have gone away; its destructor then returns the
         * capacity commitment to the topology ledger.
         */
        struct PhysicalMemoryReservationState
        {
            PhysicalMemoryReservationState(
                std::shared_ptr<PhysicalMemoryMaterializationState> authority,
                std::size_t resource_index,
                PhysicalMemoryOwner owner,
                std::size_t capacity_bytes)
                : authority(std::move(authority)),
                  resource_index(resource_index),
                  owner(owner),
                  capacity_bytes(capacity_bytes)
            {
            }

            ~PhysicalMemoryReservationState() noexcept;

            std::shared_ptr<PhysicalMemoryMaterializationState> authority;
            std::size_t resource_index = 0u;
            PhysicalMemoryOwner owner = PhysicalMemoryOwner::Count;
            std::size_t capacity_bytes = 0u;
            std::size_t materialized_bytes = 0u;
            std::size_t native_preclaimed_bytes = 0u;
            bool registered = false;
        };

        /** @brief One immutable SDK source declaration and its canonical one-shot lifecycle. */
        struct PhysicalMemoryNativeRootPermitState
        {
            PhysicalMemoryNativeRootPermitState(
                PhysicalMemoryNativeRootDeclaration value,
                std::shared_ptr<PhysicalMemoryReservationState> owner)
                : declaration(std::move(value)), reservation(std::move(owner))
            {
            }
            const PhysicalMemoryNativeRootDeclaration declaration;
            const std::shared_ptr<PhysicalMemoryReservationState> reservation;
            bool consumed = false,
                 retired = false; ///< Existing authority mutex protects both lifecycle facts.
        };

        PhysicalMemoryReservationState::~PhysicalMemoryReservationState()
            noexcept
        {
            if (!registered)
                return;

            std::lock_guard lock(authority->mutex);
            const std::size_t index = static_cast<std::size_t>(owner);
            auto &claims = authority->claims[resource_index];

            // Last-reference destruction with a live child is impossible under
            // the RAII graph. Treat counter corruption as fatal instead of
            // silently publishing false capacity during teardown.
            if (materialized_bytes != 0u || native_preclaimed_bytes != 0u ||
                claims.new_reservation_capacity[index] < capacity_bytes)
            {
                std::terminate();
            }
            claims.new_reservation_capacity[index] -= capacity_bytes;
            registered = false;
        }
    } // namespace detail

    namespace
    {
        /** @brief Authenticate phase before optional certificate access; publication is immutable
         * afterward. */
        const std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> &requireAdmitted(
            const detail::PhysicalMemoryMaterializationState &state)
        {
            std::lock_guard lock(state.mutex);
            if (state.phase != PhysicalMemoryAuthorityPhase::Admitted || !state.certificate)
                throw std::logic_error(
                    "Ordinary physical storage requires the final Admitted authority phase");
            return state.certificate;
        }
    } // namespace

    std::string PhysicalMemoryAllocatorIdentity::toString() const
    {
        if (!valid())
            return "invalid-memory-allocator";
        return "rank=" +
               (world_rank < 0 ? std::string("unbound")
                               : std::to_string(world_rank)) +
               ";device=" + device.toString();
    }

    const PhysicalMemoryBOM *PhysicalMemoryPlan::find(
        PhysicalMemoryAllocatorIdentity identity) const noexcept
    {
        const auto found = std::find_if(
            resources_.begin(),
            resources_.end(),
            [&](const auto &bom) { return matches(bom.resource(), identity); });
        return found == resources_.end() ? nullptr : &*found;
    }

    bool PhysicalMemoryPlan::fits() const noexcept
    {
        return std::all_of(
            resources_.begin(),
            resources_.end(),
            [](const auto &bom) { return bom.fits(); });
    }

    std::size_t PhysicalMemoryPlan::totalBytes() const
    {
        std::size_t total = 0u;
        for (const auto &bom : resources_)
            total = checkedAdd(total, bom.totalBytes(), "plan total");
        return total;
    }

    std::size_t PhysicalMemoryPlan::incrementalBytes() const
    {
        std::size_t total = 0u;
        for (const auto &bom : resources_)
        {
            total = checkedAdd(
                total, bom.incrementalBytes(), "plan incremental total");
        }
        return total;
    }

    std::size_t PhysicalMemoryPlan::admissionAvailableBytes() const
    {
        std::size_t total = 0u;
        for (const auto &bom : resources_)
        {
            total = checkedAdd(
                total,
                bom.resource().admission_available_bytes,
                "plan available total");
        }
        return total;
    }

    std::optional<std::string>
    PhysicalMemoryPlan::requiredFootprintMismatch(
        const PhysicalMemoryPlan &required) const
    {
        for (const auto &required_bom : required.resources_)
        {
            const PhysicalMemoryAllocatorIdentity identity{
                .world_rank = required_bom.resource().world_rank,
                .device = required_bom.resource().device,
            };
            const auto *admitted_bom = find(identity);
            if (!admitted_bom)
            {
                return "Retained physical-memory authority has no admitted "
                       "resource for " +
                       identity.toString();
            }
            if (admitted_bom->resource().total_bytes !=
                required_bom.resource().total_bytes)
            {
                return "Physical capacity changed for " +
                       identity.toString() + ": admitted=" +
                       std::to_string(
                           admitted_bom->resource().total_bytes) +
                       " required=" +
                       std::to_string(
                           required_bom.resource().total_bytes);
            }

            for (std::size_t owner_index = 0u;
                 owner_index < PhysicalMemoryBOM::ownerCount();
                 ++owner_index)
            {
                const auto owner =
                    PhysicalMemoryBOM::ownerAt(owner_index);
                const std::size_t admitted_bytes =
                    admitted_bom->bytes(owner);
                const std::size_t required_bytes =
                    required_bom.bytes(owner);
                if (required_bytes > admitted_bytes)
                {
                    return "Required " + std::string(toString(owner)) +
                           " exceeds the retained physical-memory envelope "
                           "for " + identity.toString() + ": admitted=" +
                           std::to_string(admitted_bytes) + " required=" +
                           std::to_string(required_bytes);
                }
            }
        }
        return std::nullopt;
    }

    std::string PhysicalMemoryPlan::summary() const
    {
        std::ostringstream output;
        for (std::size_t index = 0; index < resources_.size(); ++index)
        {
            if (index != 0u)
                output << '\n';
            output << resources_[index].summary();
        }
        return output.str();
    }

    PhysicalMemoryPlanBuilder::PhysicalMemoryPlanBuilder(
        const PhysicalMemoryPlan &base)
        : plan_(base)
    {
    }

    PhysicalMemoryPlanBuilder &PhysicalMemoryPlanBuilder::add(
        const PhysicalMemoryBOM &bom)
    {
        if (!bom.resource().valid())
        {
            throw std::invalid_argument(
                "Physical memory aggregate requires a valid resource BOM");
        }
        const PhysicalMemoryAllocatorIdentity identity{
            .world_rank = bom.resource().world_rank,
            .device = bom.resource().device,
        };
        const auto found = std::find_if(
            plan_.resources_.begin(),
            plan_.resources_.end(),
            [&](const auto &existing)
            { return matches(existing.resource(), identity); });
        if (found == plan_.resources_.end())
        {
            plan_.resources_.push_back(bom);
            return *this;
        }
        if (!found->resource().sameAllocator(bom.resource()))
        {
            throw std::invalid_argument(
                "Physical memory contributors disagree on allocator observation for " +
                identity.toString());
        }

        PhysicalMemoryBOMBuilder combined(*found);
        for (std::size_t index = 0;
             index < PhysicalMemoryBOM::ownerCount();
             ++index)
        {
            const auto owner = PhysicalMemoryBOM::ownerAt(index);
            const auto &charge = bom.charge(owner);
            combined.add(
                owner,
                charge.planned_bytes,
                charge.already_resident_bytes);
        }
        *found = combined.build();
        return *this;
    }

    PhysicalMemoryPlanBuilder &PhysicalMemoryPlanBuilder::add(
        PhysicalMemoryResource resource,
        PhysicalMemoryOwner owner,
        std::size_t planned_bytes,
        std::size_t already_resident_bytes)
    {
        PhysicalMemoryBOMBuilder builder(std::move(resource));
        builder.add(owner, planned_bytes, already_resident_bytes);
        return add(builder.build());
    }

    PhysicalMemoryPlanBuilder &PhysicalMemoryPlanBuilder::addMutuallyExclusive(
        std::span<const PhysicalMemoryPlan> alternatives)
    {
        PhysicalMemoryPlanBuilder envelope;
        for (const auto &alternative : alternatives)
            for (const auto &bom : alternative.resources())
            {
                const auto *existing = envelope.plan_.find(
                    {bom.resource().world_rank, bom.resource().device});
                if (existing && !existing->resource().sameAllocator(bom.resource()))
                    throw std::invalid_argument("Mutually exclusive memory plans disagree on allocator observation");
                // Differences extend the envelope to this alternative's maximum.
                // Never put a second sum/capacity ledger in a sampling caller.
                PhysicalMemoryBOMBuilder extension(bom.resource());
                for (size_t index = 0; index < PhysicalMemoryBOM::ownerCount(); ++index)
                {
                    const auto owner = PhysicalMemoryBOM::ownerAt(index);
                    const auto &charge = bom.charge(owner);
                    if (charge.already_resident_bytes)
                        throw std::invalid_argument("Mutually exclusive memory plans cannot merge residency credits");
                    const size_t previous = existing ? existing->charge(owner).planned_bytes : 0;
                    if (charge.planned_bytes > previous)
                        extension.add(owner, charge.planned_bytes - previous);
                }
                envelope.add(extension.build());
            }
        auto combined = *this;
        for (const auto &bom : envelope.plan_.resources()) combined.add(bom);
        plan_ = std::move(combined.plan_);
        return *this;
    }

    PhysicalMemoryPlan PhysicalMemoryPlanBuilder::build() const
    {
        PhysicalMemoryPlan result = plan_;
        std::sort(
            result.resources_.begin(),
            result.resources_.end(),
            resourceLess);
        return result;
    }

    PhysicalMemoryPlanAdmissionCertificate::
        PhysicalMemoryPlanAdmissionCertificate(PhysicalMemoryPlan plan)
        : plan_(std::move(plan))
    {
        if (plan_.resources().empty())
        {
            throw std::invalid_argument(
                "Physical memory admission requires at least one resource");
        }
        if (!plan_.fits())
        {
            const auto failed = std::find_if(
                plan_.resources().begin(),
                plan_.resources().end(),
                [](const auto &bom) { return !bom.fits(); });
            throw PhysicalMemoryCapacityExhausted(
                "Physical memory topology admission failed: " +
                failed->summary());
        }
    }

    PhysicalMemoryAdmissionCertificate
    PhysicalMemoryPlanAdmissionCertificate::certificateFor(
        PhysicalMemoryAllocatorIdentity identity) const
    {
        const auto *bom = plan_.find(identity);
        if (!bom)
        {
            throw std::out_of_range(
                "Physical memory certificate has no resource " +
                identity.toString());
        }
        return PhysicalMemoryAdmissionCertificate(*bom);
    }

    PhysicalMemoryNativeSetupEpoch PhysicalMemoryMaterializationLedger::beginPreparationEpoch(
        int rank)
    {
        if (rank < 0 || std::none_of(
                            state_->observations.begin(), state_->observations.end(),
                            [&](const auto &r) { return r.world_rank == rank; }))
            throw std::invalid_argument("Preparation epoch requires its original physical rank");
        auto epoch = std::make_shared<detail::PhysicalMemoryNativeSetupEpochState>();
        epoch->authority = state_;
        epoch->world_rank = rank;
        std::lock_guard lock(state_->mutex);
        if (state_->phase != PhysicalMemoryAuthorityPhase::Preparing || state_->preparation_frozen)
            throw std::logic_error("Native preparation epoch requires open Preparing authority");
        epoch->identity = nextNativeSetupIdentity();
        return PhysicalMemoryNativeSetupEpoch(std::move(epoch));
    }

    PhysicalMemoryNativeSetupLease PhysicalMemoryNativeSetupEpoch::claimPreparation(
        const PhysicalMemoryNativeRootDeclaration &declaration) const
    {
        if (!epoch_ || declaration.device != DeviceId::cpu() || !declaration.owner_identity ||
            !declaration.shared_anchor_identity || !declaration.requested_bytes ||
            !declaration.physical_bytes || !declaration.allocation_granularity ||
            declaration.host_numa < -1 ||
            declaration.requested_bytes > std::numeric_limits<std::size_t>::max() -
                                              (declaration.allocation_granularity - 1) ||
            declaration.physical_bytes !=
                ((declaration.requested_bytes + declaration.allocation_granularity - 1) /
                 declaration.allocation_granularity) *
                    declaration.allocation_granularity)
            throw std::invalid_argument(
                "Preparing root requires exact source-issued CPU0 physical geometry");
        auto root = std::make_unique<detail::PhysicalMemoryPreparationRootState>(declaration);
        auto &state = *epoch_->authority;
        const auto found = std::find_if(
            state.observations.begin(), state.observations.end(), [&](const auto &r)
            { return r.world_rank == epoch_->world_rank && r.device == DeviceId::cpu(); });
        if (found == state.observations.end())
            throw std::out_of_range("Preparing source has no original CPU0 observation");
        const auto resource = std::size_t(found - state.observations.begin());
        std::lock_guard lock(state.mutex);
        if (state.phase != PhysicalMemoryAuthorityPhase::Preparing || state.preparation_frozen ||
            epoch_->phase != detail::PhysicalMemoryNativeSetupEpochState::Phase::Preparation)
            throw std::logic_error(
                "Preparation CPU0 growth is closed and cannot revive after source admission");
        for (auto *live = epoch_->preparation_roots; live; live = live->next)
            if (live->declaration.shared_anchor_identity == declaration.shared_anchor_identity)
                throw std::invalid_argument(
                    "Preparing creator root anchor is already owned by its actual source lease");
        auto &claims = state.claims[resource];
        std::size_t committed = 0;
        for (auto bytes : claims.preparation_committed)
            committed = checkedAdd(committed, bytes, "preparation commitments");
        if (committed > found->admission_available_bytes ||
            declaration.physical_bytes > found->admission_available_bytes - committed)
            throw PhysicalMemoryCapacityExhausted(
                "Native source preparation exceeds original CPU0 physical observation");
        if (epoch_->live_roots == std::numeric_limits<std::size_t>::max() ||
            epoch_->unpublished_roots == std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("Preparing physical owner lifecycle exhausted");
        PhysicalMemoryNativeSetupLease lease(epoch_, resource, PhysicalMemoryOwner::LocalCollective,
                                             declaration.physical_bytes);
        lease.preparation_root_ = std::move(root);
        auto *node = lease.preparation_root_.get();
        node->next = epoch_->preparation_roots;
        if (node->next)
            node->next->previous = node;
        epoch_->preparation_roots = node;
        claims.preparation_committed[ownerIndex(PhysicalMemoryOwner::LocalCollective)] +=
            declaration.physical_bytes;
        ++epoch_->live_roots;
        ++epoch_->unpublished_roots;
        return lease;
    }

    std::shared_ptr<const detail::PhysicalMemoryPreparationSnapshotState>
    PhysicalMemoryNativeSetupEpoch::closePreparationForPlanning() const
    {
        if (!epoch_)
            throw std::logic_error("Cannot close inactive source preparation");
        auto &state = *epoch_->authority;
        std::lock_guard lock(state.mutex);
        if (state.phase != PhysicalMemoryAuthorityPhase::Preparing || state.preparation_frozen ||
            epoch_->unpublished_roots)
            throw std::logic_error(
                "Preparing snapshot requires open, fully published physical roots");
        auto snapshot = std::make_shared<detail::PhysicalMemoryPreparationSnapshotState>();
        snapshot->epoch = epoch_;
        PhysicalMemoryPlanBuilder builder;
        for (std::size_t i = 0; i < state.observations.size(); ++i)
        {
            const auto &resource = state.observations[i];
            builder.add(resource, PhysicalMemoryOwner::LocalCollective, 0);
            for (std::size_t owner = 0; owner < PhysicalMemoryBOM::ownerCount(); ++owner)
                if (const auto bytes = state.claims[i].preparation_committed[owner])
                {
                    if (bytes != state.claims[i].preparation_materialized[owner])
                        throw std::logic_error("Preparing source root is unpublished");
                    builder.add(resource, PhysicalMemoryBOM::ownerAt(owner), bytes);
                }
        }
        for (auto *root = epoch_->preparation_roots; root; root = root->next)
            snapshot->declarations.push_back(root->declaration);
        std::sort(snapshot->declarations.begin(), snapshot->declarations.end(),
                  [](const auto &a, const auto &b)
                  { return a.shared_anchor_identity < b.shared_anchor_identity; });
        snapshot->retained = builder.build();
        // All possible allocation/validation failures above leave the original
        // source phase and every actual lease/token unchanged.
        state.preparation_frozen = true;
        return snapshot;
    }

    void PhysicalMemoryNativeSetupEpoch::cancelPreparation() const
    {
        if (!epoch_)
            return;
        auto &state = *epoch_->authority;
        std::lock_guard lock(state.mutex);
        if (state.phase == PhysicalMemoryAuthorityPhase::Admitted)
            throw std::logic_error("Admitted source cannot cancel/reopen its Preparing authority");
        state.phase = PhysicalMemoryAuthorityPhase::Cancelled;
        state.preparation_frozen = true;
        epoch_->phase = detail::PhysicalMemoryNativeSetupEpochState::Phase::Abandoned;
    }

    std::uint64_t NativePreparationPlanningSnapshot::epochIdentity() const
    {
        if (!snapshot_)
            throw std::logic_error("Native preparation snapshot already consumed");
        return snapshot_->epoch->identity;
    }
    int NativePreparationPlanningSnapshot::worldRank() const
    {
        if (!snapshot_)
            throw std::logic_error("Native preparation snapshot already consumed");
        return snapshot_->epoch->world_rank;
    }

    std::span<const PhysicalMemoryResource> NativePreparationPlanningSnapshot::initialObservations()
        const
    {
        if (!snapshot_)
            throw std::logic_error("Native preparation snapshot already consumed");
        return snapshot_->epoch->authority->observations;
    }

    const PhysicalMemoryPlan &NativePreparationPlanningSnapshot::retainedPreparationBOM() const
    {
        if (!snapshot_)
            throw std::logic_error("Native preparation snapshot already consumed");
        return snapshot_->retained;
    }

    std::span<const PhysicalMemoryNativeRootDeclaration> NativePreparationPlanningSnapshot::
        declarations() const
    {
        if (!snapshot_)
            throw std::logic_error("Native preparation snapshot already consumed");
        return snapshot_->declarations;
    }

    namespace
    {
        /**
         * @brief Revalidate every original observation, root anchor and mandatory source floor.
         * @param epoch Original source session retaining actual root leases.
         * @param snapshot Closed immutable planning value captured from that same epoch.
         * @param mandatory Exact SDK creator BOM priced before optional placement.
         * @param final Complete final source certificate with unchanged physical observations.
         * @pre The caller holds epoch->authority->mutex.
         * @throws Before mutation for stale roots, changed observations or incomplete source
         * charges.
         */
        void validateNativeAdmissionLocked(
            const std::shared_ptr<detail::PhysicalMemoryNativeSetupEpochState> &epoch,
            const std::shared_ptr<const detail::PhysicalMemoryPreparationSnapshotState> &snapshot,
            const PhysicalMemoryPlan &mandatory,
            const std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> &final)
        {
            if (!epoch || !snapshot || snapshot->epoch != epoch || !final)
                throw std::invalid_argument("Native admission requires its exact original "
                                            "epoch/snapshot and final certificate");
            const auto &state = *epoch->authority;
            if (state.phase != PhysicalMemoryAuthorityPhase::Preparing ||
                !state.preparation_frozen || state.certificate || epoch->unpublished_roots ||
                epoch->phase != detail::PhysicalMemoryNativeSetupEpochState::Phase::Preparation)
                throw std::logic_error(
                    "Native admission is not a fully published frozen Preparing source");
            const auto &resources = final->plan().resources();
            if (resources.size() != state.observations.size())
                throw std::invalid_argument(
                    "Final native admission changes/drops the original physical resource registry");
            for (std::size_t i = 0; i < resources.size(); ++i)
            {
                const auto &original = state.observations[i];
                const auto &candidate = resources[i].resource();
                if (candidate.world_rank != original.world_rank ||
                    candidate.device != original.device ||
                    candidate.total_bytes != original.total_bytes ||
                    candidate.admission_available_bytes != original.admission_available_bytes)
                    throw std::invalid_argument("Final native admission changed original "
                                                "observation or stable resource index");
                const auto *retained =
                    snapshot->retained.find({original.world_rank, original.device});
                if (!retained)
                    throw std::logic_error(
                        "Preparation snapshot lacks an original physical resource");
                for (std::size_t owner = 0; owner < PhysicalMemoryBOM::ownerCount(); ++owner)
                {
                    const auto kind = PhysicalMemoryBOM::ownerAt(owner);
                    const auto bytes = retained->charge(kind).incrementalBytes();
                    if (state.claims[i].preparation_committed[owner] != bytes ||
                        state.claims[i].preparation_materialized[owner] != bytes)
                        throw std::logic_error("A retained preparation root retired/changed after "
                                               "optional planning snapshot");
                    if (resources[i].charge(kind).incrementalBytes() < bytes)
                        throw std::logic_error("Final source plan omitted/relabelled preparation "
                                               "backing as residency");
                }
            }
            std::size_t roots = 0;
            for (auto *root = epoch->preparation_roots; root; root = root->next)
            {
                const auto &declaration = root->declaration;
                const auto found = std::lower_bound(
                    snapshot->declarations.begin(), snapshot->declarations.end(),
                    declaration.shared_anchor_identity, [](const auto &d, std::uint64_t anchor)
                    { return d.shared_anchor_identity < anchor; });
                if (found == snapshot->declarations.end() || *found != declaration)
                    throw std::logic_error(
                        "Native preparation root declaration/anchor changed after snapshot");
                ++roots;
            }
            if (roots != snapshot->declarations.size())
                throw std::logic_error("Native preparation lost an actual retained root owner");
            for (const auto &requirement : mandatory.resources())
            {
                const auto identity = PhysicalMemoryAllocatorIdentity{
                    requirement.resource().world_rank, requirement.resource().device};
                const auto *candidate = final->plan().find(identity);
                const auto *retained = snapshot->retained.find(identity);
                if (!candidate || !retained ||
                    requirement.resource().total_bytes != candidate->resource().total_bytes ||
                    requirement.resource().admission_available_bytes !=
                        candidate->resource().admission_available_bytes)
                    throw std::invalid_argument(
                        "Mandatory SDK descriptor names another original observation");
                for (std::size_t i = 0; i < PhysicalMemoryBOM::ownerCount(); ++i)
                {
                    const auto owner = PhysicalMemoryBOM::ownerAt(i);
                    const auto bytes = requirement.charge(owner).incrementalBytes();
                    if (requirement.charge(owner).already_resident_bytes ||
                        bytes < retained->charge(owner).incrementalBytes() ||
                        bytes > candidate->charge(owner).incrementalBytes())
                        throw std::logic_error("Final source plan omitted/duplicated/relabelled "
                                               "exact mandatory native storage");
                    if (bytes && owner != (identity.device.is_gpu()
                                               ? PhysicalMemoryOwner::NativeExecutionContext
                                               : PhysicalMemoryOwner::LocalCollective))
                        throw std::invalid_argument(
                            "Mandatory native descriptor charged an ordinary model owner");
                }
            }
        }
    } // namespace

    std::vector<PhysicalMemoryOwnerReservation> PhysicalMemoryNativeSetupEpoch::
        prepareNativeAdmission(
            const std::shared_ptr<const detail::PhysicalMemoryPreparationSnapshotState> &snapshot,
            const PhysicalMemoryPlan &mandatory,
            const std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> &final) const
    {
        if (!epoch_)
            throw std::logic_error("Inactive preparation cannot build admitted native owners");
        {
            std::lock_guard lock(epoch_->authority->mutex);
            validateNativeAdmissionLocked(epoch_, snapshot, mandatory, final);
        }
        std::vector<PhysicalMemoryOwnerReservation> reservations;
        const auto &observations = epoch_->authority->observations;
        for (std::size_t resource = 0; resource < observations.size(); ++resource)
        {
            const auto &observed = observations[resource];
            const PhysicalMemoryAllocatorIdentity identity{observed.world_rank, observed.device};
            const auto *required = mandatory.find(identity);
            const auto *retained = snapshot->retained.find(identity);
            if (!required)
                continue;
            for (std::size_t index = 0; index < PhysicalMemoryBOM::ownerCount(); ++index)
            {
                const auto owner = PhysicalMemoryBOM::ownerAt(index);
                const auto bytes = required->charge(owner).incrementalBytes() -
                                   retained->charge(owner).incrementalBytes();
                if (!bytes)
                    continue;
                auto state = std::make_shared<detail::PhysicalMemoryReservationState>(
                    epoch_->authority, resource, owner, bytes);
                // Pending reservation states own no committed capacity. They
                // are unobservable until final publication's noexcept section.
                reservations.push_back(PhysicalMemoryOwnerReservation(std::move(state)));
            }
        }
        return reservations;
    }

    void PhysicalMemoryNativeSetupEpoch::publishNativeAdmission(
        const std::shared_ptr<const detail::PhysicalMemoryPreparationSnapshotState> &snapshot,
        const PhysicalMemoryPlan &mandatory,
        std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> final,
        const std::vector<PhysicalMemoryOwnerReservation> &reservations,
        std::vector<std::shared_ptr<detail::PhysicalMemoryNativeRootPermitState>> permits,
        std::vector<PhysicalMemoryNativeRootDeclaration> retainedPreparations, void *context,
        void (*arm)(void *) noexcept) const
    {
        if (!epoch_ || !snapshot || !context || !arm)
            throw std::invalid_argument(
                "Native admission requires its prebuilt disarmed callback binding");
        auto &state = *epoch_->authority;
        std::lock_guard lock(state.mutex);
        validateNativeAdmissionLocked(epoch_, snapshot, mandatory, final);
        if (!epoch_->native_permits.empty())
            throw std::logic_error("Native descriptor program already published");
        if (retainedPreparations != snapshot->declarations)
            throw std::logic_error(
                "Prebuilt native commit bundle changed retained preparation declarations");
        for (const auto &permit : permits)
            if (!permit || permit->consumed || permit->retired || !permit->reservation ||
                permit->reservation->registered)
                throw std::logic_error(
                    "Final native descriptor permit is partial/already consumed");
        for (const auto &handle : reservations)
        {
            if (!handle.state_ || handle.state_->authority != epoch_->authority ||
                handle.state_->registered)
                throw std::logic_error("Native final reservation is foreign/already published");
            const auto &pending = *handle.state_;
            const auto index = ownerIndex(pending.owner);
            const auto &claims = state.claims[pending.resource_index];
            const auto limit = final->plan()
                                   .resources()[pending.resource_index]
                                   .charge(pending.owner)
                                   .incrementalBytes();
            const auto committed = checkedAdd(
                checkedAdd(claims.new_allocations[index], claims.new_reservation_capacity[index],
                           "native final commitments"),
                claims.preparation_committed[index], "native preparation commitments");
            if (committed > limit || pending.capacity_bytes > limit - committed)
                throw std::logic_error(
                    "Native mandatory reservation would double-spend original preparation backing");
        }
        // No allocation, callback into SDK, validation or throwing operation
        // follows this point. Same state, indices, epoch and actual tokens survive.
        static_assert(noexcept(state.certificate = std::move(final)));
        static_assert(noexcept(epoch_->native_permits = std::move(permits)));
        static_assert(noexcept(epoch_->admitted_preparation = std::move(retainedPreparations)));
        state.certificate = std::move(final);
        for (const auto &handle : reservations)
        {
            auto &pending = *handle.state_;
            state.claims[pending.resource_index]
                .new_reservation_capacity[std::size_t(pending.owner)] += pending.capacity_bytes;
            pending.registered = true;
        }
        epoch_->native_permits = std::move(permits);
        epoch_->admitted_preparation = std::move(retainedPreparations);
        state.phase = PhysicalMemoryAuthorityPhase::Admitted;
        arm(context);
    }

    std::vector<std::shared_ptr<detail::PhysicalMemoryNativeRootPermitState>>
    PhysicalMemoryNativeSetupEpoch::prepareNativePermits(
        std::span<const PhysicalMemoryNativeRootDeclaration> declarations,
        const std::vector<PhysicalMemoryOwnerReservation> &reservations) const
    {
        if (!epoch_)
            throw std::logic_error("Inactive source epoch cannot prepare native descriptors");
        std::vector<std::shared_ptr<detail::PhysicalMemoryNativeRootPermitState>> permits;
        permits.reserve(declarations.size());
        for (const auto &declaration : declarations)
        {
            if (declaration.owner_identity || !declaration.shared_anchor_identity ||
                !declaration.physical_bytes)
                throw std::invalid_argument(
                    "Prepared native root must retain immutable unbound source geometry");
            if (std::any_of(permits.begin(), permits.end(),
                            [&](const auto &p)
                            {
                                return p->declaration.shared_anchor_identity ==
                                       declaration.shared_anchor_identity;
                            }))
                throw std::invalid_argument("SDK source program duplicates a creator root anchor");
            const auto owner = declaration.device.is_gpu()
                                   ? PhysicalMemoryOwner::NativeExecutionContext
                                   : PhysicalMemoryOwner::LocalCollective;
            const auto found =
                std::find_if(reservations.begin(), reservations.end(),
                             [&](const auto &r)
                             {
                                 return r.allocatorIdentity() ==
                                            PhysicalMemoryAllocatorIdentity{epoch_->world_rank,
                                                                            declaration.device} &&
                                        r.owner() == owner;
                             });
            if (found == reservations.end() || !found->state_ || found->state_->registered)
                throw std::logic_error(
                    "SDK source descriptor lacks its exact unpublished original owner reserve");
            permits.push_back(std::make_shared<detail::PhysicalMemoryNativeRootPermitState>(
                declaration, found->state_));
        }
        for (const auto &reservation : reservations)
        {
            std::size_t extent = 0;
            for (const auto &permit : permits)
                if (permit->reservation == reservation.state_)
                    extent = checkedAdd(extent, permit->declaration.physical_bytes,
                                        "native descriptor creation extents");
            if (extent != reservation.capacityBytes())
                throw std::logic_error(
                    "Native source permit extents do not cover their exact owner reserve");
        }
        return permits;
    }

    PhysicalMemoryNativeSetupLease PhysicalMemoryNativeSetupEpoch::claimPreparedNative(
        const PhysicalMemoryNativeRootDeclaration &declaration) const
    {
        if (!epoch_ || !declaration.owner_identity || !declaration.shared_anchor_identity)
            throw std::invalid_argument(
                "Native Create requires its actual owner and prepared source anchor");
        auto &state = *epoch_->authority;
        std::lock_guard lock(state.mutex);
        if (state.phase != PhysicalMemoryAuthorityPhase::Admitted)
            throw std::logic_error(
                "Native source descriptor cannot materialize before final admission");
        const auto expected =
            declaration.source_phase == PhysicalMemoryNativeRootPhase::GraphRecordingHostMetadata
                ? detail::PhysicalMemoryNativeSetupEpochState::Phase::HostRecording
                : detail::PhysicalMemoryNativeSetupEpochState::Phase::Preparation;
        if (epoch_->phase != expected)
            throw std::logic_error(
                "Native descriptor growth is outside its exact setup/recording phase");
        const auto found = std::find_if(
            epoch_->native_permits.begin(), epoch_->native_permits.end(),
            [&](const auto &p)
            {
                return p->declaration.shared_anchor_identity == declaration.shared_anchor_identity;
            });
        if (found == epoch_->native_permits.end() ||
            !(*found)->declaration.samePreparedGeometry(declaration))
            throw std::logic_error("Native Create changed its frozen source descriptor/anchor");
        auto permit = *found;
        if (permit->consumed || permit->retired)
            throw std::logic_error(
                "Native source descriptor already consumed; cancelled Create cannot recycle it");
        auto reservation = permit->reservation;
        if (!reservation || !reservation->registered || reservation->authority != epoch_->authority)
            throw std::logic_error("Native source descriptor lost its canonical owner reserve");
        const auto bytes = declaration.physical_bytes;
        const auto owner = ownerIndex(reservation->owner);
        if (reservation->materialized_bytes > reservation->capacity_bytes ||
            reservation->native_preclaimed_bytes >
                reservation->capacity_bytes - reservation->materialized_bytes ||
            bytes > reservation->capacity_bytes - reservation->materialized_bytes -
                        reservation->native_preclaimed_bytes)
            throw std::logic_error(
                "Native source descriptor exceeds its exact original reservation");
        if (epoch_->live_roots == std::numeric_limits<std::size_t>::max() ||
            epoch_->unpublished_roots == std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("Native source root lifetime exhausted");
        PhysicalMemoryNativeSetupLease lease(epoch_, reservation->resource_index,
                                             reservation->owner, bytes);
        lease.declared_reservation_ = reservation;
        lease.permit_ = permit;
        permit->consumed = true;
        reservation->native_preclaimed_bytes += bytes;
        state.claims[reservation->resource_index].native_declared_preclaimed[owner] += bytes;
        ++epoch_->live_roots;
        ++epoch_->unpublished_roots;
        return lease;
    }

    std::uint64_t PhysicalMemoryNativeSetupEpoch::identity() const noexcept
    {
        return epoch_ ? epoch_->identity : 0;
    }

    void PhysicalMemoryNativeSetupEpoch::beginHostRecording() const
    {
        if (!epoch_)
            throw std::logic_error("Cannot record an inactive native setup epoch");
        std::lock_guard lock(epoch_->authority->mutex);
        if (epoch_->authority->phase != PhysicalMemoryAuthorityPhase::Admitted)
            throw std::logic_error("Host graph recording requires final source admission");
        if (epoch_->phase == detail::PhysicalMemoryNativeSetupEpochState::Phase::HostRecording)
            return;
        if (epoch_->phase == detail::PhysicalMemoryNativeSetupEpochState::Phase::Sealed ||
            epoch_->phase == detail::PhysicalMemoryNativeSetupEpochState::Phase::Abandoned ||
            epoch_->unpublished_roots)
            throw std::logic_error(
                "Native host recording requires complete physical setup and an unsealed epoch");
        epoch_->phase = detail::PhysicalMemoryNativeSetupEpochState::Phase::HostRecording;
    }

    void PhysicalMemoryNativeSetupEpoch::sealForExecution() const
    {
        if (!epoch_)
            throw std::logic_error("Cannot close an inactive native setup epoch");
        std::lock_guard lock(epoch_->authority->mutex);
        if (epoch_->authority->phase != PhysicalMemoryAuthorityPhase::Admitted)
            throw std::logic_error("Executable publication requires final source admission");
        if (epoch_->phase == detail::PhysicalMemoryNativeSetupEpochState::Phase::Abandoned)
            throw std::logic_error("Abandoned native setup cannot publish an executable");
        if (epoch_->unpublished_roots)
            throw std::logic_error(
                "Native capture cannot close setup with unpublished physical roots");
        std::size_t preparationRoots = 0;
        for (auto *root = epoch_->preparation_roots; root; root = root->next)
        {
            if (std::find(epoch_->admitted_preparation.begin(), epoch_->admitted_preparation.end(),
                          root->declaration) == epoch_->admitted_preparation.end())
                throw std::logic_error("Executable seal changed a retained preparation root");
            ++preparationRoots;
        }
        if (preparationRoots != epoch_->admitted_preparation.size())
            throw std::logic_error("Executable seal lost a mandatory retained preparation owner");
        for (const auto &permit : epoch_->native_permits)
            if (!permit->consumed || permit->retired)
                throw std::logic_error(
                    "Executable seal omitted/retired mandatory retained native root anchor=" +
                    std::to_string(permit->declaration.shared_anchor_identity));
        epoch_->phase = detail::PhysicalMemoryNativeSetupEpochState::Phase::Sealed;
    }

    void PhysicalMemoryNativeSetupEpoch::abandonSetup() const
    {
        if (!epoch_)
            return;
        std::lock_guard lock(epoch_->authority->mutex);
        if (epoch_->authority->phase != PhysicalMemoryAuthorityPhase::Admitted)
            throw std::logic_error("Only admitted native setup may close as abandoned");
        // Failure closure cannot publish an executable, release physical bytes,
        // or recycle a descriptor. Actual root leases still own retirement.
        // Destruction after a successful seal preserves its distinct terminal state.
        if (epoch_->phase != detail::PhysicalMemoryNativeSetupEpochState::Phase::Sealed)
            epoch_->phase = detail::PhysicalMemoryNativeSetupEpochState::Phase::Abandoned;
    }

    bool PhysicalMemoryNativeSetupEpoch::sealed() const
    {
        if (!epoch_)
            return false;
        std::lock_guard lock(epoch_->authority->mutex);
        return epoch_->phase == detail::PhysicalMemoryNativeSetupEpochState::Phase::Sealed ||
               epoch_->phase == detail::PhysicalMemoryNativeSetupEpochState::Phase::Abandoned;
    }

    bool PhysicalMemoryNativeSetupEpoch::retired() const
    {
        if (!epoch_)
            return false;
        std::lock_guard lock(epoch_->authority->mutex);
        return (epoch_->phase == detail::PhysicalMemoryNativeSetupEpochState::Phase::Sealed ||
                epoch_->phase == detail::PhysicalMemoryNativeSetupEpochState::Phase::Abandoned) &&
               !epoch_->live_roots && !epoch_->unpublished_roots;
    }

    void PhysicalMemoryNativeSetupEpoch::requireRetired() const
    {
        if (!retired())
            throw std::logic_error(
                "Native source re-admission requires final SDK physical-root/alias retirement");
    }

    PhysicalMemoryNativeSetupLease::PhysicalMemoryNativeSetupLease() = default;

    PhysicalMemoryNativeSetupLease::PhysicalMemoryNativeSetupLease(
        std::shared_ptr<detail::PhysicalMemoryNativeSetupEpochState> epoch,
        std::size_t resource_index, PhysicalMemoryOwner owner, std::size_t bytes)
        : epoch_(std::move(epoch)), resource_index_(resource_index), owner_(owner), bytes_(bytes)
    {
    }

    PhysicalMemoryNativeSetupLease::~PhysicalMemoryNativeSetupLease()
    {
        release();
    }

    PhysicalMemoryNativeSetupLease::PhysicalMemoryNativeSetupLease(
        PhysicalMemoryNativeSetupLease &&other) noexcept
        : epoch_(std::move(other.epoch_)),
          declared_reservation_(std::move(other.declared_reservation_)),
          preparation_root_(std::move(other.preparation_root_)), permit_(std::move(other.permit_)),
          resource_index_(other.resource_index_), owner_(other.owner_), bytes_(other.bytes_),
          materialized_(other.materialized_)
    {
        other.bytes_ = 0;
        other.owner_ = PhysicalMemoryOwner::Count;
        other.materialized_ = false;
    }

    PhysicalMemoryNativeSetupLease &PhysicalMemoryNativeSetupLease::operator=(
        PhysicalMemoryNativeSetupLease &&other) noexcept
    {
        if (this != &other)
        {
            release();
            epoch_ = std::move(other.epoch_);
            declared_reservation_ = std::move(other.declared_reservation_);
            preparation_root_ = std::move(other.preparation_root_);
            permit_ = std::move(other.permit_);
            resource_index_ = other.resource_index_;
            owner_ = other.owner_;
            bytes_ = other.bytes_;
            materialized_ = other.materialized_;
            other.bytes_ = 0;
            other.owner_ = PhysicalMemoryOwner::Count;
            other.materialized_ = false;
        }
        return *this;
    }

    void PhysicalMemoryNativeSetupLease::materialize()
    {
        if (!valid())
            throw std::logic_error("Cannot materialize an inactive native root claim");
        auto &state = *epoch_->authority;
        std::lock_guard lock(state.mutex);
        if (materialized_ ||
            epoch_->phase == detail::PhysicalMemoryNativeSetupEpochState::Phase::Sealed ||
            epoch_->phase == detail::PhysicalMemoryNativeSetupEpochState::Phase::Abandoned ||
            !epoch_->unpublished_roots)
            throw std::logic_error(
                "Native root physical publication is duplicate or its setup epoch closed");
        const auto owner = ownerIndex(owner_);
        auto &claims = state.claims[resource_index_];
        if (preparation_root_)
        {
            if (state.phase != PhysicalMemoryAuthorityPhase::Preparing ||
                state.preparation_frozen || claims.preparation_committed[owner] < bytes_ ||
                claims.preparation_materialized[owner] >
                    claims.preparation_committed[owner] - bytes_)
                throw std::logic_error(
                    "Preparation physical publication requires its live pending source claim");
            claims.preparation_materialized[owner] += bytes_;
        }
        else if (declared_reservation_)
        {
            auto &reserved = *declared_reservation_;
            if (reserved.native_preclaimed_bytes < bytes_ ||
                claims.native_declared_preclaimed[owner] < bytes_)
                throw std::logic_error(
                    "Declared native publication exceeds its pending reservation child");
            reserved.native_preclaimed_bytes -= bytes_;
            claims.native_declared_preclaimed[owner] -= bytes_;
            reserved.materialized_bytes += bytes_;
            claims.new_reservation_materialized[owner] += bytes_;
        }
        else
            throw std::logic_error(
                "Native root has no exact preparation/program reservation authority");
        --epoch_->unpublished_roots;
        materialized_ = true;
    }

    void PhysicalMemoryNativeSetupLease::release() noexcept
    {
        if (!valid())
            return;
        auto retained = epoch_;
        {
            auto &state = *retained->authority;
            std::lock_guard lock(state.mutex);
            const auto owner = std::size_t(owner_);
            auto &claims = state.claims[resource_index_];
            if (!retained->live_roots || (!materialized_ && !retained->unpublished_roots))
                std::terminate();
            if (preparation_root_)
            {
                if (claims.preparation_committed[owner] < bytes_ ||
                    (materialized_ && claims.preparation_materialized[owner] < bytes_))
                    std::terminate();
                claims.preparation_committed[owner] -= bytes_;
                if (materialized_)
                    claims.preparation_materialized[owner] -= bytes_;
                auto *node = preparation_root_.get();
                if (node->previous)
                    node->previous->next = node->next;
                else
                    retained->preparation_roots = node->next;
                if (node->next)
                    node->next->previous = node->previous;
            }
            else if (declared_reservation_)
            {
                auto &reserved = *declared_reservation_;
                auto &local =
                    materialized_ ? reserved.materialized_bytes : reserved.native_preclaimed_bytes;
                auto &aggregate = materialized_ ? claims.new_reservation_materialized[owner]
                                                : claims.native_declared_preclaimed[owner];
                if (local < bytes_ || aggregate < bytes_)
                    std::terminate();
                local -= bytes_;
                aggregate -= bytes_;
            }
            else
                std::terminate();
            if (!materialized_)
                --retained->unpublished_roots;
            --retained->live_roots;
            if (permit_)
            {
                if (!permit_->consumed || permit_->retired)
                    std::terminate();
                permit_->retired = true;
            }
        }
        permit_.reset();
        preparation_root_.reset();
        declared_reservation_.reset();
        epoch_.reset();
        bytes_ = 0;
        owner_ = PhysicalMemoryOwner::Count;
        materialized_ = false;
    }

    PhysicalMemoryAllocationLease::PhysicalMemoryAllocationLease(
        std::shared_ptr<detail::PhysicalMemoryMaterializationState> state,
        std::size_t resource_index,
        PhysicalMemoryOwner owner,
        PhysicalMemoryMaterializationKind kind,
        std::size_t bytes)
        : state_(std::move(state)),
          resource_index_(resource_index),
          owner_(owner),
          kind_(kind),
          bytes_(bytes)
    {
    }

    PhysicalMemoryAllocationLease::~PhysicalMemoryAllocationLease()
    {
        release();
    }

    PhysicalMemoryAllocationLease::PhysicalMemoryAllocationLease(
        PhysicalMemoryAllocationLease &&other) noexcept
        : state_(std::move(other.state_)),
          resource_index_(other.resource_index_),
          owner_(other.owner_),
          kind_(other.kind_),
          bytes_(other.bytes_)
    {
        other.owner_ = PhysicalMemoryOwner::Count;
        other.bytes_ = 0u;
    }

    PhysicalMemoryAllocationLease &PhysicalMemoryAllocationLease::operator=(
        PhysicalMemoryAllocationLease &&other) noexcept
    {
        if (this == &other)
            return *this;
        release();
        state_ = std::move(other.state_);
        resource_index_ = other.resource_index_;
        owner_ = other.owner_;
        kind_ = other.kind_;
        bytes_ = other.bytes_;
        other.owner_ = PhysicalMemoryOwner::Count;
        other.bytes_ = 0u;
        return *this;
    }

    bool PhysicalMemoryAllocationLease::valid() const noexcept
    {
        return state_ && bytes_ != 0u &&
               owner_ != PhysicalMemoryOwner::Count;
    }

    void PhysicalMemoryAllocationLease::release() noexcept
    {
        if (!valid())
            return;
        auto retained_state = state_;
        {
            std::lock_guard lock(retained_state->mutex);
            const std::size_t index = static_cast<std::size_t>(owner_);
            auto &claims = retained_state->claims[resource_index_];
            auto &counter =
                kind_ == PhysicalMemoryMaterializationKind::NewAllocation
                    ? claims.new_allocations[index]
                    : claims.adopted_residency[index];
            if (counter >= bytes_)
                counter -= bytes_;
            else
                counter = 0u;
        }
        state_.reset();
        owner_ = PhysicalMemoryOwner::Count;
        bytes_ = 0u;
    }

    PhysicalMemorySuballocationLease::PhysicalMemorySuballocationLease(
        std::shared_ptr<detail::PhysicalMemoryReservationState> state,
        std::size_t bytes)
        : state_(std::move(state)), bytes_(bytes)
    {
    }

    PhysicalMemorySuballocationLease::~PhysicalMemorySuballocationLease()
    {
        release();
    }

    PhysicalMemorySuballocationLease::PhysicalMemorySuballocationLease(
        PhysicalMemorySuballocationLease &&other) noexcept
        : state_(std::move(other.state_)), bytes_(other.bytes_)
    {
        other.bytes_ = 0u;
    }

    PhysicalMemorySuballocationLease &
    PhysicalMemorySuballocationLease::operator=(
        PhysicalMemorySuballocationLease &&other) noexcept
    {
        if (this == &other)
            return *this;
        release();
        state_ = std::move(other.state_);
        bytes_ = other.bytes_;
        other.bytes_ = 0u;
        return *this;
    }

    bool PhysicalMemorySuballocationLease::valid() const noexcept
    {
        return state_ && bytes_ != 0u;
    }

    void PhysicalMemorySuballocationLease::release() noexcept
    {
        if (!valid())
            return;

        auto retained_state = state_;
        auto &authority = *retained_state->authority;
        {
            std::lock_guard lock(authority.mutex);
            const std::size_t index =
                static_cast<std::size_t>(retained_state->owner);
            auto &aggregate =
                authority.claims[retained_state->resource_index]
                    .new_reservation_materialized[index];
            if (retained_state->materialized_bytes < bytes_ ||
                aggregate < bytes_)
            {
                std::terminate();
            }
            retained_state->materialized_bytes -= bytes_;
            aggregate -= bytes_;
        }
        state_.reset();
        bytes_ = 0u;
    }

    PhysicalMemoryOwnerReservation::PhysicalMemoryOwnerReservation(
        std::shared_ptr<detail::PhysicalMemoryReservationState> state)
        : state_(std::move(state))
    {
    }

    bool PhysicalMemoryOwnerReservation::valid() const noexcept
    {
        return state_ != nullptr;
    }

    std::size_t PhysicalMemoryOwnerReservation::capacityBytes() const noexcept
    {
        return state_ ? state_->capacity_bytes : 0u;
    }

    std::size_t PhysicalMemoryOwnerReservation::materializedBytes() const
    {
        if (!state_)
            return 0u;
        std::lock_guard lock(state_->authority->mutex);
        if (state_->authority->phase != PhysicalMemoryAuthorityPhase::Admitted ||
            !state_->registered)
            throw std::logic_error("Native pending reservation is not an admitted physical owner");
        return state_->materialized_bytes;
    }

    std::size_t PhysicalMemoryOwnerReservation::remainingBytes() const
    {
        if (!state_)
            return 0u;
        std::lock_guard lock(state_->authority->mutex);
        if (state_->authority->phase != PhysicalMemoryAuthorityPhase::Admitted ||
            !state_->registered)
            throw std::logic_error(
                "Native pending reservation has no ordinary allocation capacity");
        return state_->capacity_bytes - state_->materialized_bytes -
               state_->native_preclaimed_bytes;
    }

    PhysicalMemoryAllocatorIdentity PhysicalMemoryOwnerReservation::allocatorIdentity()
        const noexcept
    {
        if (!state_)
            return {};
        const auto &resource = state_->authority->observations[state_->resource_index];
        return {resource.world_rank, resource.device};
    }

    PhysicalMemoryOwner PhysicalMemoryOwnerReservation::owner() const noexcept
    {
        return state_ ? state_->owner : PhysicalMemoryOwner::Count;
    }

    PhysicalMemorySuballocationLease
    PhysicalMemoryOwnerReservation::claimAllocation(std::size_t bytes) const
    {
        if (!state_ || bytes == 0u)
        {
            throw std::invalid_argument(
                "Physical memory pool claim requires an open reservation and positive bytes");
        }

        std::lock_guard lock(state_->authority->mutex);
        if (state_->authority->phase != PhysicalMemoryAuthorityPhase::Admitted ||
            !state_->registered)
            throw std::logic_error(
                "Ordinary child allocation requires published Admitted owner capacity");
        if (state_->materialized_bytes > state_->capacity_bytes ||
            state_->native_preclaimed_bytes > state_->capacity_bytes - state_->materialized_bytes ||
            bytes > state_->capacity_bytes - state_->materialized_bytes -
                        state_->native_preclaimed_bytes)
        {
            throw std::logic_error(
                "Physical memory pool allocation exceeds reserved " +
                std::string(toString(state_->owner)) + " capacity");
        }

        const std::size_t index =
            static_cast<std::size_t>(state_->owner);
        state_->materialized_bytes += bytes;
        state_->authority->claims[state_->resource_index]
            .new_reservation_materialized[index] += bytes;
        return PhysicalMemorySuballocationLease(state_, bytes);
    }

    PhysicalMemoryMaterializationLedger::
        PhysicalMemoryMaterializationLedger(
            std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate>
                certificate)
        : state_(certificate
                     ? std::make_shared<
                           detail::PhysicalMemoryMaterializationState>(
                           std::move(certificate))
                     : throw std::invalid_argument(
                           "Physical memory materialization requires a shared admission authority"))
    {
    }

    const std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> &
    PhysicalMemoryMaterializationLedger::admission() const
    {
        return requireAdmitted(*state_);
    }

    PhysicalMemoryAllocationLease
    PhysicalMemoryMaterializationLedger::claimNewAllocation(
        PhysicalMemoryAllocatorIdentity identity,
        PhysicalMemoryOwner owner,
        std::size_t bytes)
    {
        return claim(
            identity,
            owner,
            PhysicalMemoryMaterializationKind::NewAllocation,
            bytes);
    }

    PhysicalMemoryAllocationLease
    PhysicalMemoryMaterializationLedger::adoptResidentAllocation(
        PhysicalMemoryAllocatorIdentity identity,
        PhysicalMemoryOwner owner,
        std::size_t bytes)
    {
        return claim(
            identity,
            owner,
            PhysicalMemoryMaterializationKind::AdoptedResident,
            bytes);
    }

    PhysicalMemoryOwnerReservation
    PhysicalMemoryMaterializationLedger::reserveNewAllocations(
        PhysicalMemoryAllocatorIdentity identity,
        PhysicalMemoryOwner owner,
        std::size_t bytes)
    {
        if (!identity.valid() || bytes == 0u)
        {
            throw std::invalid_argument(
                "Physical memory reservation requires a valid resource and positive bytes");
        }
        const std::size_t owner_index = ownerIndex(owner);
        const auto &resources = requireAdmitted(*state_)->plan().resources();
        const auto found = std::find_if(
            resources.begin(),
            resources.end(),
            [&](const auto &bom) { return matches(bom.resource(), identity); });
        if (found == resources.end())
        {
            throw std::out_of_range(
                "Physical memory reservation names an unadmitted resource " +
                identity.toString());
        }
        const std::size_t resource_index =
            static_cast<std::size_t>(found - resources.begin());
        const std::size_t limit = found->charge(owner).incrementalBytes();

        // Allocate the shared lifetime node before changing the ledger. This
        // keeps reservation publication strongly exception-safe.
        auto reservation =
            std::make_shared<detail::PhysicalMemoryReservationState>(
                state_, resource_index, owner, bytes);
        {
            std::lock_guard lock(state_->mutex);
            auto &claims = state_->claims[resource_index];
            const std::size_t direct =
                claims.new_allocations[owner_index] + claims.preparation_committed[owner_index];
            const std::size_t reserved =
                claims.new_reservation_capacity[owner_index];
            if (direct > limit || reserved > limit - direct ||
                bytes > limit - direct - reserved)
            {
                throw std::logic_error(
                    "Physical memory reservation exceeds admitted " +
                    std::string(toString(owner)) + " bytes for " +
                    identity.toString() +
                    " admitted_incremental=" + std::to_string(limit) +
                    " claimed=" + std::to_string(direct) +
                    " reserved=" + std::to_string(reserved) +
                    " requested=" + std::to_string(bytes));
            }
            claims.new_reservation_capacity[owner_index] += bytes;
            reservation->registered = true;
        }
        return PhysicalMemoryOwnerReservation(std::move(reservation));
    }

    PhysicalMemoryAllocationLease PhysicalMemoryMaterializationLedger::claim(
        PhysicalMemoryAllocatorIdentity identity,
        PhysicalMemoryOwner owner,
        PhysicalMemoryMaterializationKind kind,
        std::size_t bytes)
    {
        if (!identity.valid() || bytes == 0u)
        {
            throw std::invalid_argument(
                "Physical memory live claim requires a valid resource and positive bytes");
        }
        const std::size_t owner_index = ownerIndex(owner);
        const auto &resources = requireAdmitted(*state_)->plan().resources();
        const auto found = std::find_if(
            resources.begin(),
            resources.end(),
            [&](const auto &bom) { return matches(bom.resource(), identity); });
        if (found == resources.end())
        {
            throw std::out_of_range(
                "Physical memory live claim names an unadmitted resource " +
                identity.toString());
        }
        const std::size_t resource_index =
            static_cast<std::size_t>(found - resources.begin());
        const auto &charge = found->charge(owner);
        const std::size_t limit =
            kind == PhysicalMemoryMaterializationKind::NewAllocation
                ? charge.incrementalBytes()
                : charge.already_resident_bytes;

        std::lock_guard lock(state_->mutex);
        auto &claims = state_->claims[resource_index];
        auto &counter =
            kind == PhysicalMemoryMaterializationKind::NewAllocation
                ? claims.new_allocations[owner_index]
                : claims.adopted_residency[owner_index];
        const std::size_t reserved = kind == PhysicalMemoryMaterializationKind::NewAllocation
                                         ? claims.new_reservation_capacity[owner_index] +
                                               claims.preparation_committed[owner_index]
                                         : 0u;
        if (counter > limit || reserved > limit - counter ||
            bytes > limit - counter - reserved)
        {
            throw std::logic_error(
                "Physical memory live claim exceeds admitted " +
                std::string(toString(owner)) + " bytes for " +
                identity.toString() +
                " admitted=" + std::to_string(limit) +
                " claimed=" + std::to_string(counter) +
                " reserved=" + std::to_string(reserved) +
                " requested=" + std::to_string(bytes) +
                " kind=" +
                std::string(
                    kind == PhysicalMemoryMaterializationKind::NewAllocation
                        ? "new_allocation"
                        : "adopted_resident"));
        }
        counter += bytes;
        return PhysicalMemoryAllocationLease(
            state_, resource_index, owner, kind, bytes);
    }

    std::size_t PhysicalMemoryMaterializationLedger::claimedBytes(
        PhysicalMemoryAllocatorIdentity identity,
        PhysicalMemoryOwner owner,
        PhysicalMemoryMaterializationKind kind) const
    {
        const std::size_t owner_index = ownerIndex(owner);
        const auto &resources = requireAdmitted(*state_)->plan().resources();
        const auto found = std::find_if(
            resources.begin(),
            resources.end(),
            [&](const auto &bom) { return matches(bom.resource(), identity); });
        if (found == resources.end())
        {
            throw std::out_of_range(
                "Physical memory live query names an unadmitted resource " +
                identity.toString());
        }
        const std::size_t resource_index =
            static_cast<std::size_t>(found - resources.begin());
        std::lock_guard lock(state_->mutex);
        const auto &claims = state_->claims[resource_index];
        if (kind == PhysicalMemoryMaterializationKind::NewAllocation)
        {
            return claims.new_allocations[owner_index] +
                   claims.new_reservation_materialized[owner_index] +
                   claims.preparation_materialized[owner_index];
        }
        return claims.adopted_residency[owner_index];
    }

    std::size_t PhysicalMemoryMaterializationLedger::reservedBytes(
        PhysicalMemoryAllocatorIdentity identity,
        PhysicalMemoryOwner owner) const
    {
        const std::size_t owner_index = ownerIndex(owner);
        const auto &resources = requireAdmitted(*state_)->plan().resources();
        const auto found = std::find_if(
            resources.begin(),
            resources.end(),
            [&](const auto &bom) { return matches(bom.resource(), identity); });
        if (found == resources.end())
        {
            throw std::out_of_range(
                "Physical memory reservation query names an unadmitted resource " +
                identity.toString());
        }
        const std::size_t resource_index =
            static_cast<std::size_t>(found - resources.begin());
        std::lock_guard lock(state_->mutex);
        return state_->claims[resource_index]
            .new_reservation_capacity[owner_index];
    }

    std::size_t PhysicalMemoryMaterializationLedger::committedBytes(
        PhysicalMemoryAllocatorIdentity identity,
        PhysicalMemoryOwner owner,
        PhysicalMemoryMaterializationKind kind) const
    {
        const std::size_t owner_index = ownerIndex(owner);
        const auto &resources = requireAdmitted(*state_)->plan().resources();
        const auto found = std::find_if(
            resources.begin(),
            resources.end(),
            [&](const auto &bom) { return matches(bom.resource(), identity); });
        if (found == resources.end())
        {
            throw std::out_of_range(
                "Physical memory commitment query names an unadmitted resource " +
                identity.toString());
        }
        const std::size_t resource_index =
            static_cast<std::size_t>(found - resources.begin());
        std::lock_guard lock(state_->mutex);
        const auto &claims = state_->claims[resource_index];
        if (kind == PhysicalMemoryMaterializationKind::NewAllocation)
        {
            return claims.new_allocations[owner_index] +
                   claims.new_reservation_capacity[owner_index] +
                   claims.preparation_committed[owner_index];
        }
        return claims.adopted_residency[owner_index];
    }

    std::size_t PhysicalMemoryMaterializationLedger::
        remainingAdmittedNewAllocationBytes(
            PhysicalMemoryAllocatorIdentity identity,
            PhysicalMemoryOwner owner) const
    {
        const std::size_t owner_index = ownerIndex(owner);
        const auto &resources = requireAdmitted(*state_)->plan().resources();
        const auto found = std::find_if(
            resources.begin(),
            resources.end(),
            [&](const auto &bom) { return matches(bom.resource(), identity); });
        if (found == resources.end())
        {
            throw std::out_of_range(
                "Physical memory remaining-capacity query names an unadmitted resource " +
                identity.toString());
        }
        const std::size_t resource_index =
            static_cast<std::size_t>(found - resources.begin());
        const std::size_t limit = found->charge(owner).incrementalBytes();
        std::lock_guard lock(state_->mutex);
        const auto &claims = state_->claims[resource_index];
        const std::size_t direct =
            claims.new_allocations[owner_index] + claims.preparation_committed[owner_index];
        const std::size_t reserved =
            claims.new_reservation_capacity[owner_index];
        if (direct > limit || reserved > limit - direct)
            std::terminate();
        return limit - direct - reserved;
    }

    std::size_t PhysicalMemoryMaterializationLedger::unmaterializedNewBytes(
        PhysicalMemoryAllocatorIdentity identity) const
    {
        const auto &resources = requireAdmitted(*state_)->plan().resources();
        const auto found = std::find_if(resources.begin(), resources.end(),
            [&](const auto &bom) { return matches(bom.resource(), identity); });
        if (found == resources.end())
            throw std::out_of_range("Physical memory pending-demand query names an unadmitted resource " +
                identity.toString());
        const auto resource_index = static_cast<std::size_t>(found - resources.begin());
        std::lock_guard lock(state_->mutex);
        const auto &claims = state_->claims[resource_index];
        std::size_t pending = 0u;
        for (std::size_t index = 0; index < PhysicalMemoryBOM::ownerCount(); ++index)
        {
            const auto planned = found->charge(PhysicalMemoryBOM::ownerAt(index)).incrementalBytes();
            const auto live = checkedAdd(
                checkedAdd(claims.new_allocations[index],
                           claims.new_reservation_materialized[index], "materialized owner bytes"),
                claims.preparation_materialized[index], "source preparation materialization");
            if (live > planned)
                throw std::logic_error("Physical memory materialization exceeds its admitted owner line");
            pending = checkedAdd(pending, planned - live, "pending allocator demand");
        }
        return pending;
    }

    bool PhysicalMemoryMaterializationLedger::complete() const
    {
        const auto &resources = requireAdmitted(*state_)->plan().resources();
        std::lock_guard lock(state_->mutex);
        for (std::size_t resource = 0; resource < resources.size(); ++resource)
        {
            for (std::size_t owner_index = 0;
                 owner_index < PhysicalMemoryBOM::ownerCount();
                 ++owner_index)
            {
                const auto owner = PhysicalMemoryBOM::ownerAt(owner_index);
                const auto &charge = resources[resource].charge(owner);
                const auto &claims = state_->claims[resource];
                if (claims.new_allocations[owner_index] +
                            claims.new_reservation_materialized[owner_index] +
                            claims.preparation_materialized[owner_index] !=
                        charge.incrementalBytes() ||
                    claims.adopted_residency[owner_index] != charge.already_resident_bytes)
                {
                    return false;
                }
            }
        }
        return true;
    }

    void PhysicalMemoryMaterializationLedger::requireComplete() const
    {
        const auto &resources = requireAdmitted(*state_)->plan().resources();
        std::lock_guard lock(state_->mutex);
        for (std::size_t resource = 0; resource < resources.size(); ++resource)
        {
            for (std::size_t owner_index = 0;
                 owner_index < PhysicalMemoryBOM::ownerCount();
                 ++owner_index)
            {
                const auto owner = PhysicalMemoryBOM::ownerAt(owner_index);
                const auto &charge = resources[resource].charge(owner);
                const auto &claims = state_->claims[resource];
                const std::size_t materialized_new =
                    claims.new_allocations[owner_index] +
                    claims.new_reservation_materialized[owner_index] +
                    claims.preparation_materialized[owner_index];
                if (materialized_new ==
                        charge.incrementalBytes() &&
                    claims.adopted_residency[owner_index] ==
                        charge.already_resident_bytes)
                {
                    continue;
                }
                throw std::logic_error(
                    "Physical memory materialization is incomplete for " +
                    resources[resource].resource().id() + " owner=" + std::string(toString(owner)) +
                    " new=" + std::to_string(materialized_new) + "/" +
                    std::to_string(charge.incrementalBytes()) +
                    " reserved=" + std::to_string(claims.new_reservation_capacity[owner_index]) +
                    " retained=" + std::to_string(claims.adopted_residency[owner_index]) + "/" +
                    std::to_string(charge.already_resident_bytes) + "");
            }
        }
    }

    bool PhysicalMemoryMaterializationLedger::committed() const
    {
        const auto &resources = requireAdmitted(*state_)->plan().resources();
        std::lock_guard lock(state_->mutex);
        for (std::size_t resource = 0; resource < resources.size(); ++resource)
        {
            for (std::size_t owner_index = 0;
                 owner_index < PhysicalMemoryBOM::ownerCount();
                 ++owner_index)
            {
                const auto owner = PhysicalMemoryBOM::ownerAt(owner_index);
                const auto &charge = resources[resource].charge(owner);
                const auto &claims = state_->claims[resource];
                if (claims.new_allocations[owner_index] +
                            claims.new_reservation_capacity[owner_index] +
                            claims.preparation_committed[owner_index] !=
                        charge.incrementalBytes() ||
                    claims.adopted_residency[owner_index] != charge.already_resident_bytes)
                {
                    return false;
                }
            }
        }
        return true;
    }

    void PhysicalMemoryMaterializationLedger::requireCommitted() const
    {
        const auto &resources = requireAdmitted(*state_)->plan().resources();
        std::lock_guard lock(state_->mutex);
        for (std::size_t resource = 0; resource < resources.size(); ++resource)
        {
            for (std::size_t owner_index = 0;
                 owner_index < PhysicalMemoryBOM::ownerCount();
                 ++owner_index)
            {
                const auto owner = PhysicalMemoryBOM::ownerAt(owner_index);
                const auto &charge = resources[resource].charge(owner);
                const auto &claims = state_->claims[resource];
                const std::size_t committed_new = claims.new_allocations[owner_index] +
                                                  claims.new_reservation_capacity[owner_index] +
                                                  claims.preparation_committed[owner_index];
                if (committed_new == charge.incrementalBytes() &&
                    claims.adopted_residency[owner_index] ==
                        charge.already_resident_bytes)
                {
                    continue;
                }
                throw std::logic_error(
                    "Physical memory capacity commitment is incomplete for " +
                    resources[resource].resource().id() + " owner=" + std::string(toString(owner)) +
                    " new=" + std::to_string(committed_new) + "/" +
                    std::to_string(charge.incrementalBytes()) + " materialized=" +
                    std::to_string(claims.new_allocations[owner_index] +
                                   claims.new_reservation_materialized[owner_index] +
                                   claims.preparation_materialized[owner_index]) +
                    " retained=" + std::to_string(claims.adopted_residency[owner_index]) + "/" +
                    std::to_string(charge.already_resident_bytes));
            }
        }
    }

    std::vector<PhysicalMemoryOwnerAttestation>
    PhysicalMemoryMaterializationLedger::attestation() const
    {
        std::vector<PhysicalMemoryOwnerAttestation> result;
        result.reserve(state_->observations.size() * PhysicalMemoryBOM::ownerCount());
        std::lock_guard lock(state_->mutex);
        const bool admitted = state_->phase == PhysicalMemoryAuthorityPhase::Admitted;
        if (admitted && !state_->certificate)
            throw std::logic_error("Admitted source has no immutable certificate");
        for (std::size_t resource = 0; resource < state_->observations.size(); ++resource)
        {
            const auto &observed = state_->observations[resource];
            const auto &claims = state_->claims[resource];
            for (std::size_t owner = 0; owner < PhysicalMemoryBOM::ownerCount(); ++owner)
            {
                const auto kind = PhysicalMemoryBOM::ownerAt(owner);
                const auto *charge =
                    admitted ? &state_->certificate->plan().resources()[resource].charge(kind)
                             : nullptr;
                result.push_back({
                    .identity = {observed.world_rank, observed.device},
                    .phase = state_->phase,
                    .owner = kind,
                    .planned_new_bytes = charge ? charge->incrementalBytes() : 0u,
                    .planned_resident_bytes = charge ? charge->already_resident_bytes : 0u,
                    .materialized_new_bytes = checkedAdd(
                        checkedAdd(claims.new_allocations[owner],
                                   claims.new_reservation_materialized[owner],
                                   "source materialization"),
                        claims.preparation_materialized[owner], "preparation materialization"),
                    .committed_new_bytes = checkedAdd(
                        checkedAdd(claims.new_allocations[owner],
                                   claims.new_reservation_capacity[owner], "source commitments"),
                        claims.preparation_committed[owner], "preparation commitments"),
                    .adopted_resident_bytes = claims.adopted_residency[owner],
                    .preparation_committed_bytes = claims.preparation_committed[owner],
                    .preparation_materialized_bytes = claims.preparation_materialized[owner],
                    .native_declared_preclaimed_bytes = claims.native_declared_preclaimed[owner],
                });
            }
        }
        return result;
    }

    PhysicalMemoryAuthority::PhysicalMemoryAuthority(
        std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate>
            admission,
        int world_rank)
        : world_rank_(world_rank),
          ledger_(admission)
    {
        if (world_rank_ < 0)
        {
            throw std::invalid_argument(
                "Physical memory runtime authority requires a non-negative world rank");
        }
        const auto &resources = ledger_.admission()->plan().resources();
        const bool owns_resource = std::any_of(
            resources.begin(),
            resources.end(),
            [this](const auto &bom)
            { return bom.resource().world_rank == world_rank_; });
        if (!owns_resource)
        {
            throw std::invalid_argument(
                "Physical memory runtime authority rank owns no admitted allocator: rank=" +
                std::to_string(world_rank_));
        }
    }

    const std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> &PhysicalMemoryAuthority::
        admission() const
    {
        return ledger_.admission();
    }

    bool PhysicalMemoryAuthority::contains(DeviceId device) const
    {
        if (!device.is_valid())
            return false;
        return admission()->plan().find(
                   PhysicalMemoryAllocatorIdentity{
                       .world_rank = world_rank_,
                       .device = device,
                   }) != nullptr;
    }

    PhysicalMemoryAllocatorIdentity PhysicalMemoryAuthority::identity(
        DeviceId device) const
    {
        const PhysicalMemoryAllocatorIdentity result{
            .world_rank = world_rank_,
            .device = device,
        };
        if (!result.valid() || !admission()->plan().find(result))
        {
            throw std::out_of_range(
                "Physical memory runtime authority has no local resource " +
                result.toString());
        }
        return result;
    }

    std::size_t PhysicalMemoryAuthority::plannedBytes(
        DeviceId device,
        PhysicalMemoryOwner owner) const
    {
        const auto local_identity = identity(device);
        return admission()->plan().find(local_identity)->bytes(owner);
    }

    PhysicalMemoryAllocationLease
    PhysicalMemoryAuthority::claimNewAllocation(
        DeviceId device,
        PhysicalMemoryOwner owner,
        std::size_t bytes)
    {
        return ledger_.claimNewAllocation(identity(device), owner, bytes);
    }

    PhysicalMemoryMaterializationLedger::PhysicalMemoryMaterializationLedger(
        std::span<const PhysicalMemoryResource> observations)
        : state_(std::make_shared<detail::PhysicalMemoryMaterializationState>(observations))
    {
    }

    PhysicalMemoryAuthority::PhysicalMemoryAuthority(
        std::span<const PhysicalMemoryResource> observations, int rank, PreparingNativeStorageTag)
        : world_rank_(rank), ledger_(observations)
    {
        if (rank < 0 ||
            std::none_of(
                ledger_.state_->observations.begin(), ledger_.state_->observations.end(),
                [&](const auto &r) { return r.world_rank == rank && r.device == DeviceId::cpu(); }))
            throw std::invalid_argument(
                "Native Preparing requires its original local CPU0 physical allocator");
    }

    PhysicalMemoryAuthorityPhase PhysicalMemoryAuthority::phase() const
    {
        std::lock_guard lock(ledger_.state_->mutex);
        return ledger_.state_->phase;
    }

    PhysicalMemoryAllocationLease
    PhysicalMemoryAuthority::adoptResidentAllocation(
        DeviceId device,
        PhysicalMemoryOwner owner,
        std::size_t bytes)
    {
        return ledger_.adoptResidentAllocation(
            identity(device), owner, bytes);
    }

    PhysicalMemoryOwnerReservation
    PhysicalMemoryAuthority::reserveNewAllocations(
        DeviceId device,
        PhysicalMemoryOwner owner,
        std::size_t bytes)
    {
        return ledger_.reserveNewAllocations(
            identity(device), owner, bytes);
    }

    PhysicalMemoryAllocationLease PhysicalMemoryAuthority::claimMeasuredRuntimeContextStorage(
        PhysicalMemoryResource observed, std::size_t bytes) const
    {
        if (!observed.valid() || !observed.device.is_gpu() ||
            observed.world_rank != world_rank_ || !bytes)
            throw std::invalid_argument("Native execution-context storage requires positive same-rank GPU residency");
        const auto allocator = identity(observed.device);
        const auto *source = admission()->plan().find(allocator);
        if (!source || source->resource().total_bytes != observed.total_bytes)
            throw std::invalid_argument("Native execution-context observation changed its physical allocator");
        const auto pending = ledger_.unmaterializedNewBytes(allocator);
        if (pending > observed.admission_available_bytes)
            throw PhysicalMemoryCapacityExhausted("Native execution-context preparation would displace outstanding physical owners for " +
                observed.id() + " pending_bytes=" + std::to_string(pending));
        // Fresh driver capacity already excludes live backing. Protect only
        // the original plan's future allocations; lazy pool capacity must not
        // disappear just because its reservation was claimed before capture.
        observed.admission_available_bytes -= pending;
        PhysicalMemoryPlanBuilder builder;
        builder.add(observed, PhysicalMemoryOwner::NativeExecutionContext, bytes);
        PhysicalMemoryAuthority context(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), world_rank_);
        return context.claimNewAllocation(observed.device,
            PhysicalMemoryOwner::NativeExecutionContext, bytes);
    }

    std::size_t PhysicalMemoryAuthority::claimedBytes(
        DeviceId device,
        PhysicalMemoryOwner owner,
        PhysicalMemoryMaterializationKind kind) const
    {
        return ledger_.claimedBytes(identity(device), owner, kind);
    }

    std::size_t PhysicalMemoryAuthority::reservedBytes(
        DeviceId device,
        PhysicalMemoryOwner owner) const
    {
        return ledger_.reservedBytes(identity(device), owner);
    }

    std::size_t PhysicalMemoryAuthority::committedBytes(
        DeviceId device,
        PhysicalMemoryOwner owner,
        PhysicalMemoryMaterializationKind kind) const
    {
        return ledger_.committedBytes(
            identity(device), owner, kind);
    }

    std::size_t PhysicalMemoryAuthority::
        remainingAdmittedNewAllocationBytes(
            DeviceId device,
            PhysicalMemoryOwner owner) const
    {
        return ledger_.remainingAdmittedNewAllocationBytes(
            identity(device), owner);
    }

    bool PhysicalMemoryAuthority::rankComplete() const
    {
        const auto rows = rankAttestation();
        return std::all_of(rows.begin(), rows.end(),
                           [](const auto &row) { return row.complete(); });
    }

    void PhysicalMemoryAuthority::requireRankComplete() const
    {
        const auto rows = rankAttestation();
        for (const auto &row : rows)
            if (!row.complete())
                throw std::logic_error("Physical rank materialization incomplete for " +
                                       row.identity.toString() +
                                       " owner=" + std::string(toString(row.owner)) +
                                       " new=" + std::to_string(row.materialized_new_bytes) + "/" +
                                       std::to_string(row.planned_new_bytes) +
                                       " retained=" + std::to_string(row.adopted_resident_bytes) +
                                       "/" + std::to_string(row.planned_resident_bytes));
    }

    bool PhysicalMemoryAuthority::rankCommitted() const
    {
        const auto &resources = admission()->plan().resources();
        for (const auto &bom : resources)
        {
            if (bom.resource().world_rank != world_rank_)
                continue;
            for (std::size_t owner_index = 0;
                 owner_index < PhysicalMemoryBOM::ownerCount();
                 ++owner_index)
            {
                const auto owner = PhysicalMemoryBOM::ownerAt(owner_index);
                const auto &charge = bom.charge(owner);
                const DeviceId device = bom.resource().device;
                if (committedBytes(
                        device,
                        owner,
                        PhysicalMemoryMaterializationKind::NewAllocation) !=
                        charge.incrementalBytes() ||
                    committedBytes(
                        device,
                        owner,
                        PhysicalMemoryMaterializationKind::AdoptedResident) !=
                        charge.already_resident_bytes)
                {
                    return false;
                }
            }
        }
        return true;
    }

    void PhysicalMemoryAuthority::requireRankCommitted() const
    {
        const auto &resources = admission()->plan().resources();
        for (const auto &bom : resources)
        {
            if (bom.resource().world_rank != world_rank_)
                continue;
            for (std::size_t owner_index = 0;
                 owner_index < PhysicalMemoryBOM::ownerCount();
                 ++owner_index)
            {
                const auto owner = PhysicalMemoryBOM::ownerAt(owner_index);
                const auto &charge = bom.charge(owner);
                const DeviceId device = bom.resource().device;
                const auto newly_committed = committedBytes(
                    device,
                    owner,
                    PhysicalMemoryMaterializationKind::NewAllocation);
                const auto resident_committed = committedBytes(
                    device,
                    owner,
                    PhysicalMemoryMaterializationKind::AdoptedResident);
                if (newly_committed == charge.incrementalBytes() &&
                    resident_committed == charge.already_resident_bytes)
                {
                    continue;
                }
                throw std::logic_error(
                    "Rank-local physical memory capacity commitment is incomplete for " +
                    bom.resource().id() + " owner=" +
                    std::string(toString(owner)) + " new=" +
                    std::to_string(newly_committed) + "/" +
                    std::to_string(charge.incrementalBytes()) +
                    " materialized=" +
                    std::to_string(claimedBytes(
                        device,
                        owner,
                        PhysicalMemoryMaterializationKind::NewAllocation)) +
                    " retained=" +
                    std::to_string(resident_committed) + "/" +
                    std::to_string(charge.already_resident_bytes));
            }
        }
    }

    std::vector<PhysicalMemoryOwnerAttestation>
    PhysicalMemoryAuthority::rankAttestation() const
    {
        auto all = ledger_.attestation();
        std::erase_if(
            all,
            [this](const PhysicalMemoryOwnerAttestation &entry)
            { return entry.identity.world_rank != world_rank_; });
        return all;
    }

    std::string PhysicalMemoryAuthority::rankAttestationSummary(
        bool include_zero_lines) const
    {
        const auto rows = rankAttestation();
        std::ostringstream output;
        bool first = true;
        for (const auto &row : rows)
        {
            const bool zero = row.planned_new_bytes == 0u && row.planned_resident_bytes == 0u &&
                              row.materialized_new_bytes == 0u && row.committed_new_bytes == 0u &&
                              row.adopted_resident_bytes == 0u &&
                              row.native_declared_preclaimed_bytes == 0u;
            if (zero && !include_zero_lines)
                continue;
            if (!first)
                output << '\n';
            first = false;
            output << row.identity.toString() << " phase=" << toString(row.phase)
                   << " owner=" << toString(row.owner) << " planned_new=" << row.planned_new_bytes
                   << " committed_new=" << row.committed_new_bytes
                   << " materialized_new=" << row.materialized_new_bytes
                   << " planned_resident=" << row.planned_resident_bytes
                   << " adopted_resident=" << row.adopted_resident_bytes
                   << " preparation_committed=" << row.preparation_committed_bytes
                   << " preparation_materialized=" << row.preparation_materialized_bytes
                   << " native_declared_preclaimed=" << row.native_declared_preclaimed_bytes
                   << " committed=" << (row.committed() ? "true" : "false")
                   << " complete=" << (row.complete() ? "true" : "false");
        }
        if (first && !rows.empty() && rows.front().phase != PhysicalMemoryAuthorityPhase::Admitted)
            output << "rank=" << world_rank_ << " phase=" << toString(rows.front().phase)
                   << " source_plan_admitted=false";
        return output.str();
    }
} // namespace llaminar2
