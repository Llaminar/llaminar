/**
 * @file MoEOverlayPreparedExpertPayload.h
 * @brief Validated prepared-engine lifetimes carried by one placement epoch.
 *
 * A movable payload is either a complete expert or its owner-local gate/up
 * pair. Fixed down-output slices have model lifetime and never travel through
 * this value. Empty values represent non-resident slots; partially assembled
 * arrivals belong to the transfer builder, not to a publishable residency bank.
 * This class owns engine lifetimes only, never an owner map or memory ledger.
 */
#pragma once

#include "kernels/common/DeviceMoEFloatingMatrixDesc.h"

#include <array>
#include <memory>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    class ITensorGemm;

    /**
     * @brief Immutable projection family retained by the existing RCU lifecycle.
     *
     * Named construction authenticates the entire family before publication.
     * A gate/up payload cannot accidentally certify a full FFN: complete()
     * remains the stronger predicate used at complete-expert compute boundaries.
     */
    class MoEOverlayPreparedExpertPayload final
    {
    public:
        /** @brief Construct the only non-resident representation: no engines. */
        MoEOverlayPreparedExpertPayload() noexcept = default;

        /** @brief Retain the same complete immutable engine family. */
        MoEOverlayPreparedExpertPayload(const MoEOverlayPreparedExpertPayload &) = default;
        /** @brief Replace this lifetime with another complete family. */
        MoEOverlayPreparedExpertPayload &operator=(const MoEOverlayPreparedExpertPayload &) = default;
        /** @brief Transfer lifetimes and leave the source canonically empty. */
        MoEOverlayPreparedExpertPayload(MoEOverlayPreparedExpertPayload &&other) noexcept
            : engines_(std::move(other.engines_)), kind_(std::exchange(other.kind_, Kind::Empty)) {}
        /** @brief Transfer ownership without leaving a false-ready source. */
        MoEOverlayPreparedExpertPayload &operator=(MoEOverlayPreparedExpertPayload &&other) noexcept
        {
            if (this != &other)
            {
                engines_ = std::move(other.engines_);
                kind_ = std::exchange(other.kind_, Kind::Empty);
            }
            return *this;
        }

        /**
         * @brief Retain a complete expert; all three lifetimes are mandatory.
         * @throws std::invalid_argument If any projection is absent.
         */
        MoEOverlayPreparedExpertPayload(
            std::shared_ptr<ITensorGemm> gate,
            std::shared_ptr<ITensorGemm> up,
            std::shared_ptr<ITensorGemm> down)
            : MoEOverlayPreparedExpertPayload(
                  Kind::CompleteExpert, std::move(gate), std::move(up), std::move(down))
        {
        }

        /**
         * @brief Retain only the projections that follow a gate/up owner.
         * @throws std::invalid_argument If either movable projection is absent.
         */
        static MoEOverlayPreparedExpertPayload gateUp(
            std::shared_ptr<ITensorGemm> gate, std::shared_ptr<ITensorGemm> up)
        {
            return {Kind::GateUp, std::move(gate), std::move(up), {}};
        }

        /**
         * @brief Seal a transfer builder's exact declared family atomically.
         * @throws std::invalid_argument For partial, extra, or unknown projections.
         */
        static MoEOverlayPreparedExpertPayload fromProjections(
            DeviceMoEProjectionSet projections, std::array<std::shared_ptr<ITensorGemm>, 3> engines)
        {
            if (projections != DeviceMoEProjectionSet::CompleteExpert &&
                projections != DeviceMoEProjectionSet::GateUp)
                throw std::invalid_argument("Unknown prepared expert projection family");
            return {projections == DeviceMoEProjectionSet::CompleteExpert ? Kind::CompleteExpert : Kind::GateUp,
                    std::move(engines[0]), std::move(engines[1]), std::move(engines[2])};
        }

        /** @return Whether this value carries its complete declared payload. */
        [[nodiscard]] bool ready() const noexcept { return kind_ != Kind::Empty; }
        /** @return Whether every projection of a complete expert is present. */
        [[nodiscard]] bool complete() const noexcept { return kind_ == Kind::CompleteExpert; }
        /** @return Whether this is the canonical non-resident value. */
        [[nodiscard]] bool empty() const noexcept { return kind_ == Kind::Empty; }

        /** @return Whether the payload matches an immutable endpoint contract. */
        [[nodiscard]] bool readyFor(DeviceMoEProjectionSet projections) const noexcept
        {
            return (projections == DeviceMoEProjectionSet::CompleteExpert && complete()) ||
                   (projections == DeviceMoEProjectionSet::GateUp && kind_ == Kind::GateUp);
        }

        /**
         * @return The authenticated movable projection family.
         * @throws std::logic_error For an empty non-resident slot.
         */
        [[nodiscard]] DeviceMoEProjectionSet projections() const
        {
            if (empty())
                throw std::logic_error("Empty expert payload has no projection contract");
            return complete() ? DeviceMoEProjectionSet::CompleteExpert : DeviceMoEProjectionSet::GateUp;
        }

        /** @return Shared gate lifetime, or null for a non-resident slot. */
        [[nodiscard]] const std::shared_ptr<ITensorGemm> &gate() const noexcept { return engines_[0]; }
        /** @return Shared up lifetime, or null for a non-resident slot. */
        [[nodiscard]] const std::shared_ptr<ITensorGemm> &up() const noexcept { return engines_[1]; }
        /** @return Movable down lifetime; absent for gate/up-only payloads. */
        [[nodiscard]] const std::shared_ptr<ITensorGemm> &down() const noexcept { return engines_[2]; }

        /** @return Exact family and object identity, not merely matching shapes. */
        [[nodiscard]] bool sameIdentity(const MoEOverlayPreparedExpertPayload &other) const noexcept
        {
            return kind_ == other.kind_ && engines_ == other.engines_;
        }

    private:
        /** Only named factories can select a nonempty physical payload family. */
        enum class Kind { Empty, CompleteExpert, GateUp };

        /** @brief Validate once; readers never observe a partially built family. */
        MoEOverlayPreparedExpertPayload(
            Kind kind, std::shared_ptr<ITensorGemm> gate,
            std::shared_ptr<ITensorGemm> up, std::shared_ptr<ITensorGemm> down)
            : engines_{std::move(gate), std::move(up), std::move(down)}, kind_(kind)
        {
            if (!engines_[0] || !engines_[1] ||
                (kind == Kind::CompleteExpert ? !engines_[2] : static_cast<bool>(engines_[2])))
                throw std::invalid_argument("Prepared expert payload requires its exact declared projection family");
        }

        std::array<std::shared_ptr<ITensorGemm>, 3> engines_{};
        Kind kind_ = Kind::Empty;
    };
} // namespace llaminar2
