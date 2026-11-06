/**
 * @file MoEOverlayRuntimeReaderDeadline.h
 * @brief Bounded host observation of original device retirement receipts.
 *
 * This owner stores only polling deadlines, not placement or lease authority.
 * Every expected peer retains the ordinary thirty-second no-progress budget.
 * An actual captured child terminal may extend only that peer's deadline;
 * repeated deferred receipts cannot keep a stalled finite parent alive.
 */
#pragma once

#include "MoEOverlayDeviceTransportProtocol.h"

#include <chrono>
#include <span>
#include <stdexcept>
#include <vector>

namespace llaminar2
{
    /** Deadline observer for one already authenticated physical transaction. */
    class MoEOverlayRuntimeReaderDeadline final
    {
    public:
        using Clock = std::chrono::steady_clock;
        /** Closed observation outcomes; Ready still requires the original protocol gate. */
        enum class Result { Pending, Ready, Invalid, Expired };

        /**
         * @brief Allocate polling state once before observing original peers.
         * @param count Exact topology cardinality, never inferred from routed work.
         * @param now Original monotonic start; explicit for deterministic tests.
         */
        MoEOverlayRuntimeReaderDeadline(std::size_t count, Clock::time_point now)
            : peers_(count, Peer{.deadline=now+std::chrono::seconds(30)})
        {
            if(!count)throw std::invalid_argument("Runtime reader deadline requires every original peer");
        }

        /**
         * @brief Consume authenticated receipts without extending a quiet peer.
         * @param observations Original transport's topology-ordered observations.
         * @param now Current monotonic clock, never device or policy state.
         * @return Pending, all Ready, a malformed regression, or an expired peer.
         */
        [[nodiscard]] Result observe(
            std::span<const MoEOverlayRuntimeReaderObservation> observations,
            Clock::time_point now) noexcept
        {
            using State=MoEOverlayRuntimeReaderObservation::State;
            if(observations.size()!=peers_.size())return Result::Invalid;
            bool ready=true;
            for(std::size_t i=0u;i<peers_.size();++i) {
                const auto &receipt=observations[i];auto &peer=peers_[i];
                if(peer.ready) {
                    if(receipt.state!=State::Ready)return Result::Invalid;
                    continue;
                }
                if(now>=peer.deadline)return Result::Expired;
                if(receipt.state==State::Ready) {peer.ready=true;continue;}
                ready=false;
                // Check expiration BEFORE accepting a late progress word. A
                // dead peer cannot retroactively repair the bounded contract.
                if(now>=peer.deadline)return Result::Expired;
                if(receipt.state==State::FiniteDeferred) {
                    if(!receipt.parent_generation || !receipt.parent_progress)return Result::Invalid;
                    if(peer.generation && (peer.generation!=receipt.parent_generation ||
                       receipt.parent_progress<peer.progress))return Result::Invalid;
                    if(!peer.generation) {
                        peer.generation=receipt.parent_generation;
                        peer.progress=receipt.parent_progress;
                    } else if(receipt.parent_progress>peer.progress) {
                        peer.progress=receipt.parent_progress;
                        peer.deadline=now+std::chrono::seconds(30);
                    }
                } else if(receipt.state!=State::Pending)return Result::Invalid;
            }
            return ready?Result::Ready:Result::Pending;
        }

    private:
        /** Polling-only state; no selected epoch, bank or device payload. */
        struct Peer
        {
            Clock::time_point deadline;
            std::uint64_t generation=0u;
            std::uint64_t progress=0u;
            bool ready=false;
        };
        std::vector<Peer> peers_;
    };
}
