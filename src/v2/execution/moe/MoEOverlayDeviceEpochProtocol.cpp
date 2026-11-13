/**
 * @file MoEOverlayDeviceEpochProtocol.cpp
 * @brief Atomic CPU implementation of captured ExpertOverlay epoch ownership.
 */

#include "MoEOverlayDeviceEpochProtocol.h"

#include <atomic>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** @brief Form an atomic view over one ABI-owned 64-bit scalar. */
        std::atomic_ref<std::uint64_t> atomic64(std::uint64_t &value) noexcept
        {
            return std::atomic_ref<std::uint64_t>(value);
        }

        /** @brief Form an atomic view over one ABI-owned 32-bit scalar. */
        std::atomic_ref<std::uint32_t> atomic32(std::uint32_t &value) noexcept
        {
            return std::atomic_ref<std::uint32_t>(value);
        }

        /** @return Raw ABI value for a typed bank lifecycle. */
        constexpr std::uint32_t rawState(
            DeviceMoEOverlayEpochBankState state) noexcept
        {
            return static_cast<std::uint32_t>(state);
        }

        /** @return Whether a raw lifecycle value belongs to the public enum. */
        constexpr bool validState(std::uint32_t state) noexcept
        {
            return state <= rawState(
                                DeviceMoEOverlayEpochBankState::Retiring);
        }
    } // namespace

    void MoEOverlayDeviceEpochProtocol::initialize(
        DeviceMoEOverlayEpochControl &control,
        std::uint64_t initial_epoch,
        std::uint32_t initial_bank)
    {
        if (initial_epoch == 0u)
        {
            throw std::invalid_argument(
                "ExpertOverlay device epoch initialization requires a positive epoch");
        }
        if (initial_bank >= kDeviceMoEOverlayEpochBankCount)
        {
            throw std::invalid_argument(
                "ExpertOverlay device epoch initialization bank is outside the two-bank ABI");
        }

        /*
         * Initialization is model-setup work, but use the same atomic accesses as
         * runtime code so thread sanitizers see one coherent ownership discipline.
         */
        if (atomic64(control.published_selector).load(
                std::memory_order_seq_cst) != 0u ||
            atomic64(control.acquisitions_in_flight).load(
                std::memory_order_seq_cst) != 0u)
        {
            throw std::logic_error(
                "ExpertOverlay device epoch control is already initialized");
        }
        for (std::uint32_t bank = 0u;
             bank < kDeviceMoEOverlayEpochBankCount;
             ++bank)
        {
            if (atomic64(control.bank_epochs[bank]).load(
                    std::memory_order_seq_cst) != 0u ||
                atomic64(control.bank_readers[bank]).load(
                    std::memory_order_seq_cst) != 0u ||
                atomic32(control.bank_states[bank]).load(
                    std::memory_order_seq_cst) !=
                    rawState(DeviceMoEOverlayEpochBankState::Empty))
            {
                throw std::logic_error(
                    "ExpertOverlay device epoch control is not pristine");
            }
        }

        atomic64(control.bank_epochs[initial_bank]).store(
            initial_epoch, std::memory_order_seq_cst);
        atomic32(control.bank_states[initial_bank]).store(
            rawState(DeviceMoEOverlayEpochBankState::Published),
            std::memory_order_seq_cst);
        atomic64(control.published_selector).store(
            deviceMoEOverlayEpochSelector(
                /*generation=*/1u, initial_bank),
            std::memory_order_seq_cst);
    }

    MoEOverlayDeviceEpochProtocol::MoEOverlayDeviceEpochProtocol(
        DeviceMoEOverlayEpochControl &control) noexcept
        : control_(&control)
    {
    }

    bool MoEOverlayDeviceEpochProtocol::armFiniteParent(
        DeviceMoEOverlayEpochParent &parent) noexcept
    {
        using State = DeviceMoEOverlayEpochParentState;
        if (!control_ || parent.control != control_ || control_->finite_parent != &parent || parent.version != kDeviceMoEOverlayEpochParentVersion) return false;
        auto state = atomic32(parent.state);
        std::uint32_t expected = static_cast<std::uint32_t>(State::Idle);
        if (!state.compare_exchange_strong(expected, static_cast<std::uint32_t>(State::Arming)))
            return false;
        // Claim before checking original readers: an overlapping admission is
        // either already guarded or observes Arming and cannot enter this scope.
        const bool pristine = !parent.ticket.valid() && parent.ticket.epoch == 0u &&
            parent.ticket.selector == 0u && parent.ticket.parent_generation == 0u && parent.ticket.parent_sequence == 0u &&
            atomic64(parent.children).load() == 0u &&
            atomic64(parent.generation).load() != std::numeric_limits<std::uint64_t>::max() &&
            atomic64(parent.progress).load() != std::numeric_limits<std::uint64_t>::max() &&
            atomic64(parent.child_sequence).load() != std::numeric_limits<std::uint64_t>::max();
        if (!pristine) { state.store(static_cast<std::uint32_t>(State::Idle)); return false; }
        const auto prior = atomic64(control_->acquisitions_in_flight).fetch_add(1u);
        const auto selected=publishedBank();
        const auto selected_state=bankState(selected);
        // Arm selects no epoch. A publication may already have marked this
        // selected bank Retiring; the original guard still prevents reuse.
        bool idle = prior == 0u && publicationGeneration()!=0u &&
            bankEpoch(selected)!=0u &&
            (selected_state==DeviceMoEOverlayEpochBankState::Published ||
             selected_state==DeviceMoEOverlayEpochBankState::Retiring);
        for (std::uint32_t bank=0u;bank<kDeviceMoEOverlayEpochBankCount;++bank)
            idle = idle && atomic64(control_->bank_readers[bank]).load() == 0u;
        if (!idle) {
            atomic64(control_->acquisitions_in_flight).fetch_sub(1u);
            state.store(static_cast<std::uint32_t>(State::Idle)); return false;
        }
        atomic64(parent.generation).fetch_add(1u);
        atomic64(parent.progress).fetch_add(1u);
        state.store(static_cast<std::uint32_t>(State::Armed));
        return true;
    }

    std::optional<DeviceMoEOverlayEpochTicket>
    MoEOverlayDeviceEpochProtocol::tryAcquireFiniteChild(
        DeviceMoEOverlayEpochParent &parent, std::uint64_t admission_epoch,
        DeviceMoEOverlayEpochChildAdmission admission) noexcept
    {
        using State = DeviceMoEOverlayEpochParentState;
        if(admission!=DeviceMoEOverlayEpochChildAdmission::Published &&
           admission!=DeviceMoEOverlayEpochChildAdmission::PreparedPeer)return std::nullopt;
        const bool prepared_peer=admission==DeviceMoEOverlayEpochChildAdmission::PreparedPeer;
        if (!control_ || parent.control != control_ || control_->finite_parent != &parent || parent.version != kDeviceMoEOverlayEpochParentVersion) return std::nullopt;
        // This is the SAME canonical reuse guard as ordinary admission. It
        // also prevents End/rearm from crossing a partially admitted child.
        auto guard = atomic64(control_->acquisitions_in_flight);
        guard.fetch_add(1u);
        auto state = atomic32(parent.state);
        auto observed = state.load();
        if (observed == static_cast<std::uint32_t>(State::Armed)) {
            auto expected = observed;
            if (!state.compare_exchange_strong(expected, static_cast<std::uint32_t>(State::Activating))) {
                guard.fetch_sub(1u); return std::nullopt;
            }
            auto ticket = prepared_peer ? tryAcquirePreparedExact(admission_epoch) : tryAcquireAdmitted(admission_epoch);
            if (!ticket) {
                state.store(static_cast<std::uint32_t>(State::Armed));
                guard.fetch_sub(1u); return std::nullopt;
            }
            parent.ticket = *ticket;
            // Parent reader is installed before retiring its long-lived Armed
            // acquisition guard. The current child still owns its own guard.
            guard.fetch_sub(1u);
            state.store(static_cast<std::uint32_t>(State::Active));
            observed = static_cast<std::uint32_t>(State::Active);
        }
        if (observed != static_cast<std::uint32_t>(State::Active)) {
            guard.fetch_sub(1u); return std::nullopt;
        }
        const auto generation = atomic64(parent.generation).load();
        std::uint64_t empty=0u;
        if(!atomic64(parent.children).compare_exchange_strong(empty,1u)) {
            guard.fetch_sub(1u);return std::nullopt; // Finite comparisons admit serial child transactions.
        }
        if (generation == 0u || state.load() != static_cast<std::uint32_t>(State::Active) ||
            atomic64(parent.generation).load() != generation || !parent.ticket.valid() ||
            (admission_epoch != 0u && admission_epoch != parent.ticket.epoch)) {
            atomic64(parent.children).fetch_sub(1u); guard.fetch_sub(1u); return std::nullopt;
        }
        auto child = tryAcquire(parent.ticket.epoch,
            prepared_peer ? AdmissionPolicy::ExactPreparedPeer : AdmissionPolicy::PublishedFloor);
        if (!child || child->bank() != parent.ticket.bank()) {
            if (child && !release(*child)) {
                // Failed physical release is not a completed descendant: keep
                // its child obligation so End cannot certify retirement.
                guard.fetch_sub(1u);return std::nullopt;
            }
            atomic64(parent.children).fetch_sub(1u); guard.fetch_sub(1u); return std::nullopt;
        }
        const auto sequence=atomic64(parent.child_sequence).load();
        if(sequence==std::numeric_limits<std::uint64_t>::max()) {
            if(!release(*child)){guard.fetch_sub(1u);return std::nullopt;}
            atomic64(parent.children).fetch_sub(1u);guard.fetch_sub(1u);return std::nullopt;
        }
        child->parent_generation = generation;
        child->parent_sequence = atomic64(parent.child_sequence).fetch_add(1u)+1u;
        guard.fetch_sub(1u);
        return child;
    }

    bool MoEOverlayDeviceEpochProtocol::releaseFiniteChild(
        DeviceMoEOverlayEpochParent &parent, DeviceMoEOverlayEpochTicket &child) noexcept
    {
        using State = DeviceMoEOverlayEpochParentState;
        if (!control_ || parent.control != control_ || control_->finite_parent != &parent || parent.version != kDeviceMoEOverlayEpochParentVersion || !child.valid() || child.parent_generation == 0u ||
            child.parent_generation != atomic64(parent.generation).load() ||
            child.parent_sequence == 0u || child.parent_sequence != atomic64(parent.child_sequence).load() ||
            atomic64(parent.progress).load() == std::numeric_limits<std::uint64_t>::max() ||
            atomic32(parent.state).load() != static_cast<std::uint32_t>(State::Active) ||
            child.epoch != parent.ticket.epoch || child.bank() != parent.ticket.bank())
            return false;
        // Claim the existing child obligation before releasing the original
        // bank reader; End remains excluded by this canonical acquisition guard.
        auto guard=atomic64(control_->acquisitions_in_flight);guard.fetch_add(1u);
        auto children=atomic64(parent.children);std::uint64_t count=1u;
        if(!children.compare_exchange_strong(count,0u)) { guard.fetch_sub(1u);return false; }
        if (!release(child)) { children.fetch_add(1u); guard.fetch_sub(1u); return false; }
        child={};atomic64(parent.progress).fetch_add(1u);guard.fetch_sub(1u);return true;
    }

    bool MoEOverlayDeviceEpochProtocol::closeFiniteParent(
        DeviceMoEOverlayEpochParent &parent) noexcept
    {
        using State = DeviceMoEOverlayEpochParentState;
        if (!control_ || parent.control != control_ || control_->finite_parent != &parent || parent.version != kDeviceMoEOverlayEpochParentVersion ||
            atomic64(parent.progress).load()==std::numeric_limits<std::uint64_t>::max()) return false;
        auto state=atomic32(parent.state);auto prior=state.load();
        if (prior != static_cast<std::uint32_t>(State::Armed) &&
            prior != static_cast<std::uint32_t>(State::Active)) return false;
        if (!state.compare_exchange_strong(prior,static_cast<std::uint32_t>(State::Closing))) return false;
        const bool armed=prior==static_cast<std::uint32_t>(State::Armed);
        if (atomic64(parent.children).load()!=0u ||
            atomic64(control_->acquisitions_in_flight).load()!=(armed?1u:0u)) {
            state.store(prior);return false;
        }
        if (armed) {
            if (parent.ticket.epoch!=0u || parent.ticket.selector!=0u || parent.ticket.parent_generation!=0u || parent.ticket.parent_sequence!=0u) { state.store(prior);return false; }
            atomic64(control_->acquisitions_in_flight).fetch_sub(1u);
        } else if (parent.ticket.parent_generation!=0u || parent.ticket.parent_sequence!=0u || !release(parent.ticket)) { state.store(prior);return false; }
        parent.ticket={};atomic64(parent.progress).fetch_add(1u);state.store(static_cast<std::uint32_t>(State::Idle));return true;
    }

    std::optional<DeviceMoEOverlayEpochTicket>
    MoEOverlayDeviceEpochProtocol::tryAcquirePublished() noexcept
    {
        return tryAcquire(
            /*epoch=*/0u,
            AdmissionPolicy::PublishedFloor);
    }

    std::optional<DeviceMoEOverlayEpochTicket>
    MoEOverlayDeviceEpochProtocol::tryAcquireAdmitted(
        std::uint64_t admission_epoch) noexcept
    {
        return tryAcquire(
            admission_epoch,
            AdmissionPolicy::PublishedFloor);
    }

    std::optional<DeviceMoEOverlayEpochTicket>
    MoEOverlayDeviceEpochProtocol::tryAcquirePreparedExact(
        std::uint64_t exact_epoch) noexcept
    {
        if (exact_epoch == 0u)
            return std::nullopt;
        return tryAcquire(
            exact_epoch,
            AdmissionPolicy::ExactPreparedPeer);
    }

    std::optional<DeviceMoEOverlayEpochTicket>
    MoEOverlayDeviceEpochProtocol::tryAcquire(
        std::uint64_t requested_epoch,
        AdmissionPolicy policy) noexcept
    {
        if (!control_)
            return std::nullopt;

        /*
         * The guard begins before the selector read.  A retirement ordered
         * before this increment leaves us reading the new selector; a retirement
         * ordered after it observes either the guard or our installed reader.
         */
        atomic64(control_->acquisitions_in_flight).fetch_add(
            1u, std::memory_order_seq_cst);

        const std::uint64_t selector =
            atomic64(control_->published_selector).load(
                std::memory_order_seq_cst);
        const std::uint32_t published_bank =
            deviceMoEOverlayEpochSelectorBank(selector);
        const std::uint64_t generation =
            deviceMoEOverlayEpochSelectorGeneration(selector);
        if (generation == 0u ||
            published_bank >= kDeviceMoEOverlayEpochBankCount)
        {
            atomic64(control_->acquisitions_in_flight).fetch_sub(
                1u, std::memory_order_seq_cst);
            return std::nullopt;
        }

        const std::uint64_t required_epoch =
            requested_epoch == 0u
                ? bankEpoch(published_bank)
                : requested_epoch;
        const auto admitted_bank = bankForEpoch(required_epoch);
        if (!admitted_bank.has_value())
        {
            atomic64(control_->acquisitions_in_flight).fetch_sub(
                1u, std::memory_order_seq_cst);
            return std::nullopt;
        }
        const std::uint32_t bank = *admitted_bank;

        const auto state = bankState(bank);
        const std::uint64_t epoch = bankEpoch(bank);
        const bool exact_prepared =
            policy == AdmissionPolicy::ExactPreparedPeer &&
            state == DeviceMoEOverlayEpochBankState::Ready;
        if (epoch == 0u || epoch != required_epoch ||
            (!exact_prepared &&
             state != DeviceMoEOverlayEpochBankState::Published &&
             state != DeviceMoEOverlayEpochBankState::Retiring))
        {
            atomic64(control_->acquisitions_in_flight).fetch_sub(
                1u, std::memory_order_seq_cst);
            return std::nullopt;
        }

        atomic64(control_->bank_readers[bank]).fetch_add(
            1u, std::memory_order_seq_cst);
        atomic64(control_->acquisitions_in_flight).fetch_sub(
            1u, std::memory_order_seq_cst);
        const std::uint64_t ticket_generation =
            bank == published_bank
                ? generation
                : exact_prepared
                      ? (generation <
                                 std::numeric_limits<std::uint64_t>::max()
                             ? generation + 1u
                             : 0u)
                      : (generation > 1u ? generation - 1u : 0u);
        if (ticket_generation == 0u)
        {
            atomic64(control_->bank_readers[bank]).fetch_sub(
                1u, std::memory_order_seq_cst);
            return std::nullopt;
        }
        return DeviceMoEOverlayEpochTicket{
            .epoch = epoch,
            .selector = deviceMoEOverlayEpochSelector(
                ticket_generation, bank),
        };
    }

    bool MoEOverlayDeviceEpochProtocol::release(
        const DeviceMoEOverlayEpochTicket &ticket) noexcept
    {
        if (!control_ || !ticket.valid() ||
            ticket.bank() >= kDeviceMoEOverlayEpochBankCount ||
            bankEpoch(ticket.bank()) != ticket.epoch)
        {
            return false;
        }

        auto readers = atomic64(control_->bank_readers[ticket.bank()]);
        std::uint64_t observed = readers.load(std::memory_order_seq_cst);
        while (observed != 0u)
        {
            if (readers.compare_exchange_weak(
                    observed,
                    observed - 1u,
                    std::memory_order_seq_cst,
                    std::memory_order_seq_cst))
            {
                return true;
            }
        }
        return false;
    }

    std::optional<std::uint32_t>
    MoEOverlayDeviceEpochProtocol::reserveCandidate(
        std::uint64_t candidate_epoch)
    {
        const std::uint64_t selector =
            atomic64(control_->published_selector).load(
                std::memory_order_seq_cst);
        const std::uint64_t generation =
            deviceMoEOverlayEpochSelectorGeneration(selector);
        const std::uint32_t published =
            deviceMoEOverlayEpochSelectorBank(selector);
        if (generation == 0u || published >= kDeviceMoEOverlayEpochBankCount ||
            publishedEpoch() == 0u)
        {
            throw std::logic_error(
                "ExpertOverlay device epoch control has no valid publication");
        }
        if (candidate_epoch == 0u || candidate_epoch <= publishedEpoch())
        {
            throw std::invalid_argument(
                "ExpertOverlay device candidate epoch must increase monotonically");
        }

        const std::uint32_t candidate = 1u - published;
        if (bankReaderCount(candidate) != 0u || bankEpoch(candidate) != 0u)
            return std::nullopt;

        auto state = atomic32(control_->bank_states[candidate]);
        std::uint32_t expected =
            rawState(DeviceMoEOverlayEpochBankState::Empty);
        if (!state.compare_exchange_strong(
                expected,
                rawState(DeviceMoEOverlayEpochBankState::Candidate),
                std::memory_order_seq_cst,
                std::memory_order_seq_cst))
        {
            return std::nullopt;
        }

        /* Candidate state excludes inference; epoch publication completes its identity. */
        atomic64(control_->bank_epochs[candidate]).store(
            candidate_epoch, std::memory_order_seq_cst);
        return candidate;
    }

    void MoEOverlayDeviceEpochProtocol::markCandidateReady(
        std::uint64_t candidate_epoch)
    {
        const auto bank = bankForEpoch(candidate_epoch);
        if (!bank.has_value())
        {
            throw std::logic_error(
                "ExpertOverlay ready transition names no candidate epoch");
        }
        auto state = atomic32(control_->bank_states[*bank]);
        std::uint32_t expected =
            rawState(DeviceMoEOverlayEpochBankState::Candidate);
        if (!state.compare_exchange_strong(
                expected,
                rawState(DeviceMoEOverlayEpochBankState::Ready),
                std::memory_order_seq_cst,
                std::memory_order_seq_cst))
        {
            throw std::logic_error(
                "ExpertOverlay candidate may become ready exactly once");
        }
    }

    MoEOverlayDeviceEpochPublication
    MoEOverlayDeviceEpochProtocol::publishReadyCandidate(
        std::uint64_t candidate_epoch)
    {
        const auto candidate = bankForEpoch(candidate_epoch);
        if (!candidate.has_value() ||
            bankState(*candidate) != DeviceMoEOverlayEpochBankState::Ready)
        {
            throw std::logic_error(
                "ExpertOverlay publication requires the exact ready candidate");
        }

        const std::uint64_t old_selector =
            atomic64(control_->published_selector).load(
                std::memory_order_seq_cst);
        const std::uint32_t previous =
            deviceMoEOverlayEpochSelectorBank(old_selector);
        const std::uint64_t old_generation =
            deviceMoEOverlayEpochSelectorGeneration(old_selector);
        if (previous >= kDeviceMoEOverlayEpochBankCount ||
            previous == *candidate || old_generation == 0u ||
            old_generation ==
                (std::numeric_limits<std::uint64_t>::max() >> 1u) ||
            bankState(previous) !=
                DeviceMoEOverlayEpochBankState::Published)
        {
            throw std::logic_error(
                "ExpertOverlay device publication has invalid bank identity or generation");
        }

        const std::uint64_t previous_epoch = bankEpoch(previous);
        if (previous_epoch == 0u || candidate_epoch <= previous_epoch)
        {
            throw std::logic_error(
                "ExpertOverlay device publication is not epoch-monotonic");
        }

        /*
         * Make the candidate executable before the selector can name it.  The
         * single selector store is admission's linearization point.  Only after
         * that store may maintenance label the former publication Retiring.
         */
        atomic32(control_->bank_states[*candidate]).store(
            rawState(DeviceMoEOverlayEpochBankState::Published),
            std::memory_order_seq_cst);
        const std::uint64_t generation = old_generation + 1u;
        atomic64(control_->published_selector).store(
            deviceMoEOverlayEpochSelector(generation, *candidate),
            std::memory_order_seq_cst);
        atomic32(control_->bank_states[previous]).store(
            rawState(DeviceMoEOverlayEpochBankState::Retiring),
            std::memory_order_seq_cst);

        return {
            .previous_epoch = previous_epoch,
            .published_epoch = candidate_epoch,
            .previous_bank = previous,
            .published_bank = *candidate,
            .generation = generation,
        };
    }

    bool MoEOverlayDeviceEpochProtocol::abortCandidate(
        std::uint64_t candidate_epoch) noexcept
    {
        const auto bank = bankForEpoch(candidate_epoch);
        if (!bank.has_value() || bankReaderCount(*bank) != 0u)
            return false;

        auto state = atomic32(control_->bank_states[*bank]);
        std::uint32_t observed = state.load(std::memory_order_seq_cst);
        while (observed == rawState(DeviceMoEOverlayEpochBankState::Candidate) ||
               observed == rawState(DeviceMoEOverlayEpochBankState::Ready))
        {
            if (state.compare_exchange_weak(
                    observed,
                    rawState(DeviceMoEOverlayEpochBankState::Empty),
                    std::memory_order_seq_cst,
                    std::memory_order_seq_cst))
            {
                atomic64(control_->bank_epochs[*bank]).store(
                    0u, std::memory_order_seq_cst);
                return true;
            }
        }
        return false;
    }

    bool MoEOverlayDeviceEpochProtocol::tryRetire(
        std::uint64_t retiring_epoch)
    {
        const auto bank = bankForEpoch(retiring_epoch);
        if (!bank.has_value() ||
            bankState(*bank) != DeviceMoEOverlayEpochBankState::Retiring)
        {
            throw std::logic_error(
                "ExpertOverlay retirement requires the exact retiring epoch");
        }

        /*
         * Sequentially consistent guard observation is the CPU grace period.
         * A prior acquisition has either left this counter nonzero or already
         * installed its bank reader.  A later acquisition observes the newer
         * selector and cannot target this retiring bank.
         */
        if (acquisitionsInFlight() != 0u || bankReaderCount(*bank) != 0u)
            return false;

        auto state = atomic32(control_->bank_states[*bank]);
        std::uint32_t expected =
            rawState(DeviceMoEOverlayEpochBankState::Retiring);
        if (!state.compare_exchange_strong(
                expected,
                rawState(DeviceMoEOverlayEpochBankState::Empty),
                std::memory_order_seq_cst,
                std::memory_order_seq_cst))
        {
            return false;
        }
        atomic64(control_->bank_epochs[*bank]).store(
            0u, std::memory_order_seq_cst);
        return true;
    }

    std::uint64_t MoEOverlayDeviceEpochProtocol::publishedEpoch() const noexcept
    {
        const std::uint32_t bank = publishedBank();
        return bankEpoch(bank);
    }

    std::uint32_t MoEOverlayDeviceEpochProtocol::publishedBank() const noexcept
    {
        if (!control_)
            return kDeviceMoEOverlayEpochBankCount;
        return deviceMoEOverlayEpochSelectorBank(
            atomic64(control_->published_selector).load(
                std::memory_order_seq_cst));
    }

    std::uint64_t
    MoEOverlayDeviceEpochProtocol::publicationGeneration() const noexcept
    {
        if (!control_)
            return 0u;
        return deviceMoEOverlayEpochSelectorGeneration(
            atomic64(control_->published_selector).load(
                std::memory_order_seq_cst));
    }

    DeviceMoEOverlayEpochBankState
    MoEOverlayDeviceEpochProtocol::bankState(std::uint32_t bank) const noexcept
    {
        if (!control_ || bank >= kDeviceMoEOverlayEpochBankCount)
            return DeviceMoEOverlayEpochBankState::Empty;
        const std::uint32_t state =
            atomic32(control_->bank_states[bank]).load(
                std::memory_order_seq_cst);
        return validState(state)
                   ? static_cast<DeviceMoEOverlayEpochBankState>(state)
                   : DeviceMoEOverlayEpochBankState::Empty;
    }

    std::uint64_t MoEOverlayDeviceEpochProtocol::bankEpoch(
        std::uint32_t bank) const noexcept
    {
        if (!control_ || bank >= kDeviceMoEOverlayEpochBankCount)
            return 0u;
        return atomic64(control_->bank_epochs[bank]).load(
            std::memory_order_seq_cst);
    }

    std::uint64_t MoEOverlayDeviceEpochProtocol::bankReaderCount(
        std::uint32_t bank) const noexcept
    {
        if (!control_ || bank >= kDeviceMoEOverlayEpochBankCount)
            return 0u;
        return atomic64(control_->bank_readers[bank]).load(
            std::memory_order_seq_cst);
    }

    std::uint64_t
    MoEOverlayDeviceEpochProtocol::acquisitionsInFlight() const noexcept
    {
        if (!control_)
            return 0u;
        return atomic64(control_->acquisitions_in_flight).load(
            std::memory_order_seq_cst);
    }

    std::optional<std::uint32_t>
    MoEOverlayDeviceEpochProtocol::bankForEpoch(
        std::uint64_t epoch) const noexcept
    {
        if (!control_ || epoch == 0u)
            return std::nullopt;
        for (std::uint32_t bank = 0u;
             bank < kDeviceMoEOverlayEpochBankCount;
             ++bank)
        {
            if (bankEpoch(bank) == epoch)
                return bank;
        }
        return std::nullopt;
    }
} // namespace llaminar2
