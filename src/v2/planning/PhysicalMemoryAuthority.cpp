/**
 * @file PhysicalMemoryAuthority.cpp
 * @brief Topology-wide CPU/GPU memory authority implementation.
 */

#include "PhysicalMemoryAuthority.h"

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
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
            };

            explicit PhysicalMemoryMaterializationState(
                std::shared_ptr<
                    const PhysicalMemoryPlanAdmissionCertificate> admitted)
                : certificate(std::move(admitted)),
                  claims(certificate->plan().resources().size())
            {
            }

            std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate>
                certificate;
            std::vector<ResourceClaims> claims;
            mutable std::mutex mutex;
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
            bool registered = false;
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
            if (materialized_bytes != 0u ||
                claims.new_reservation_capacity[index] < capacity_bytes)
            {
                std::terminate();
            }
            claims.new_reservation_capacity[index] -= capacity_bytes;
            registered = false;
        }
    } // namespace detail

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
            throw std::invalid_argument(
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
        return state_->materialized_bytes;
    }

    std::size_t PhysicalMemoryOwnerReservation::remainingBytes() const
    {
        if (!state_)
            return 0u;
        std::lock_guard lock(state_->authority->mutex);
        return state_->capacity_bytes - state_->materialized_bytes;
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
        if (state_->materialized_bytes > state_->capacity_bytes ||
            bytes > state_->capacity_bytes - state_->materialized_bytes)
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
    PhysicalMemoryMaterializationLedger::admission() const noexcept
    {
        return state_->certificate;
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
        const auto &resources = state_->certificate->plan().resources();
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
                claims.new_allocations[owner_index];
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
        const auto &resources = state_->certificate->plan().resources();
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
        const std::size_t reserved =
            kind == PhysicalMemoryMaterializationKind::NewAllocation
                ? claims.new_reservation_capacity[owner_index]
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
        const auto &resources = state_->certificate->plan().resources();
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
                   claims.new_reservation_materialized[owner_index];
        }
        return claims.adopted_residency[owner_index];
    }

    std::size_t PhysicalMemoryMaterializationLedger::reservedBytes(
        PhysicalMemoryAllocatorIdentity identity,
        PhysicalMemoryOwner owner) const
    {
        const std::size_t owner_index = ownerIndex(owner);
        const auto &resources = state_->certificate->plan().resources();
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
        const auto &resources = state_->certificate->plan().resources();
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
                   claims.new_reservation_capacity[owner_index];
        }
        return claims.adopted_residency[owner_index];
    }

    std::size_t PhysicalMemoryMaterializationLedger::
        remainingAdmittedNewAllocationBytes(
            PhysicalMemoryAllocatorIdentity identity,
            PhysicalMemoryOwner owner) const
    {
        const std::size_t owner_index = ownerIndex(owner);
        const auto &resources = state_->certificate->plan().resources();
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
            claims.new_allocations[owner_index];
        const std::size_t reserved =
            claims.new_reservation_capacity[owner_index];
        if (direct > limit || reserved > limit - direct)
            std::terminate();
        return limit - direct - reserved;
    }

    bool PhysicalMemoryMaterializationLedger::complete() const
    {
        const auto &resources = state_->certificate->plan().resources();
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
                            claims.new_reservation_materialized[owner_index] !=
                        charge.incrementalBytes() ||
                    claims.adopted_residency[owner_index] !=
                        charge.already_resident_bytes)
                {
                    return false;
                }
            }
        }
        return true;
    }

    void PhysicalMemoryMaterializationLedger::requireComplete() const
    {
        const auto &resources = state_->certificate->plan().resources();
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
                    claims.new_reservation_materialized[owner_index];
                if (materialized_new ==
                        charge.incrementalBytes() &&
                    claims.adopted_residency[owner_index] ==
                        charge.already_resident_bytes)
                {
                    continue;
                }
                throw std::logic_error(
                    "Physical memory materialization is incomplete for " +
                    resources[resource].resource().id() + " owner=" +
                    std::string(toString(owner)) + " new=" +
                    std::to_string(materialized_new) +
                    "/" + std::to_string(charge.incrementalBytes()) +
                    " reserved=" +
                    std::to_string(
                        claims.new_reservation_capacity[owner_index]) +
                    " retained=" +
                    std::to_string(claims.adopted_residency[owner_index]) +
                    "/" +
                    std::to_string(charge.already_resident_bytes));
            }
        }
    }

    bool PhysicalMemoryMaterializationLedger::committed() const
    {
        const auto &resources = state_->certificate->plan().resources();
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
                            claims.new_reservation_capacity[owner_index] !=
                        charge.incrementalBytes() ||
                    claims.adopted_residency[owner_index] !=
                        charge.already_resident_bytes)
                {
                    return false;
                }
            }
        }
        return true;
    }

    void PhysicalMemoryMaterializationLedger::requireCommitted() const
    {
        const auto &resources = state_->certificate->plan().resources();
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
                const std::size_t committed_new =
                    claims.new_allocations[owner_index] +
                    claims.new_reservation_capacity[owner_index];
                if (committed_new == charge.incrementalBytes() &&
                    claims.adopted_residency[owner_index] ==
                        charge.already_resident_bytes)
                {
                    continue;
                }
                throw std::logic_error(
                    "Physical memory capacity commitment is incomplete for " +
                    resources[resource].resource().id() + " owner=" +
                    std::string(toString(owner)) + " new=" +
                    std::to_string(committed_new) + "/" +
                    std::to_string(charge.incrementalBytes()) +
                    " materialized=" +
                    std::to_string(
                        claims.new_allocations[owner_index] +
                        claims.new_reservation_materialized[owner_index]) +
                    " retained=" +
                    std::to_string(claims.adopted_residency[owner_index]) +
                    "/" +
                    std::to_string(charge.already_resident_bytes));
            }
        }
    }

    std::vector<PhysicalMemoryOwnerAttestation>
    PhysicalMemoryMaterializationLedger::attestation() const
    {
        const auto &resources = state_->certificate->plan().resources();
        std::vector<PhysicalMemoryOwnerAttestation> result;
        result.reserve(
            resources.size() * PhysicalMemoryBOM::ownerCount());

        /*
         * Copy every related counter while holding the one ledger mutex. This
         * is the only way a diagnostic can distinguish a lazy capacity
         * commitment from its currently materialized children without racing
         * a concurrent setup allocation or teardown.
         */
        std::lock_guard lock(state_->mutex);
        for (std::size_t resource_index = 0u;
             resource_index < resources.size();
             ++resource_index)
        {
            const auto &bom = resources[resource_index];
            const auto &claims = state_->claims[resource_index];
            const PhysicalMemoryAllocatorIdentity identity{
                .world_rank = bom.resource().world_rank,
                .device = bom.resource().device,
            };
            for (std::size_t owner_index = 0u;
                 owner_index < PhysicalMemoryBOM::ownerCount();
                 ++owner_index)
            {
                const auto owner = PhysicalMemoryBOM::ownerAt(owner_index);
                const auto &charge = bom.charge(owner);
                const std::size_t direct_new =
                    claims.new_allocations[owner_index];
                const std::size_t reserved_capacity =
                    claims.new_reservation_capacity[owner_index];
                const std::size_t reserved_materialized =
                    claims.new_reservation_materialized[owner_index];
                result.push_back({
                    .identity = identity,
                    .owner = owner,
                    .planned_new_bytes = charge.incrementalBytes(),
                    .planned_resident_bytes =
                        charge.already_resident_bytes,
                    .materialized_new_bytes = checkedAdd(
                        direct_new,
                        reserved_materialized,
                        "attested materialization"),
                    .committed_new_bytes = checkedAdd(
                        direct_new,
                        reserved_capacity,
                        "attested commitment"),
                    .adopted_resident_bytes =
                        claims.adopted_residency[owner_index],
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

    const std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> &
    PhysicalMemoryAuthority::admission() const noexcept
    {
        return ledger_.admission();
    }

    bool PhysicalMemoryAuthority::contains(DeviceId device) const noexcept
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
                if (claimedBytes(
                        device,
                        owner,
                        PhysicalMemoryMaterializationKind::NewAllocation) !=
                        charge.incrementalBytes() ||
                    claimedBytes(
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

    void PhysicalMemoryAuthority::requireRankComplete() const
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
                const auto newly_claimed = claimedBytes(
                    device,
                    owner,
                    PhysicalMemoryMaterializationKind::NewAllocation);
                const auto resident_claimed = claimedBytes(
                    device,
                    owner,
                    PhysicalMemoryMaterializationKind::AdoptedResident);
                if (newly_claimed == charge.incrementalBytes() &&
                    resident_claimed == charge.already_resident_bytes)
                {
                    continue;
                }
                throw std::logic_error(
                    "Rank-local physical memory materialization is incomplete for " +
                    bom.resource().id() + " owner=" +
                    std::string(toString(owner)) + " new=" +
                    std::to_string(newly_claimed) + "/" +
                    std::to_string(charge.incrementalBytes()) +
                    " retained=" + std::to_string(resident_claimed) + "/" +
                    std::to_string(charge.already_resident_bytes));
            }
        }
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
            const bool zero = row.planned_new_bytes == 0u &&
                              row.planned_resident_bytes == 0u &&
                              row.materialized_new_bytes == 0u &&
                              row.committed_new_bytes == 0u &&
                              row.adopted_resident_bytes == 0u;
            if (zero && !include_zero_lines)
                continue;
            if (!first)
                output << '\n';
            first = false;
            output << row.identity.toString()
                   << " owner=" << toString(row.owner)
                   << " planned_new=" << row.planned_new_bytes
                   << " committed_new=" << row.committed_new_bytes
                   << " materialized_new="
                   << row.materialized_new_bytes
                   << " planned_resident="
                   << row.planned_resident_bytes
                   << " adopted_resident="
                   << row.adopted_resident_bytes
                   << " committed="
                   << (row.committed() ? "true" : "false")
                   << " complete="
                   << (row.complete() ? "true" : "false");
        }
        return output.str();
    }
} // namespace llaminar2
