/**
 * @file W4A4WeightStorage.h
 * @brief Immutable, lifetime-owning native W4A4/H matrix storage.
 *
 * A CPU allocation is adopted with its existing PMA lease, never copied or
 * charged twice. File-backed matrices derive their pointer from a retained
 * MmapRegion and checked file coordinates, not an unrelated raw pointer.
 * Complete-row slices share that one physical owner. Prepared kernels retain
 * this handle while tensor-source handles retire, so pointer validity follows
 * ownership rather than a caller-maintained lifetime convention.
 */
#pragma once

#include "W4A4SourceView.h"
#include "AlignedVector.h"

namespace llaminar2
{
    class MmapRegion;

    namespace w4a4
    {
        /** @brief Copyable immutable lease on one admitted native matrix or complete-row subset. */
        class WeightStorage final
        {
        public:
            /** @brief Copies retain the same immutable owner; rvalue copies leave the source valid too. */
            WeightStorage(const WeightStorage &) = default;
            /** @brief Replace this handle while preserving both owners until assignment completes. */
            WeightStorage &operator=(const WeightStorage &) = default;

            /**
             * @brief Adopt exactly one already admitted CPU allocation without copying its bytes.
             * @param transform Source-declared online transform.
             * @param rows Complete output-row count.
             * @param columns Complete reduction width.
             * @param bytes Final immutable source bytes; the original allocation lease moves with them.
             * @param owner Exact admitted weight or ModelSourcePayload owner, never workspace/staging.
             * @return Sealed immutable storage, ready to be shared with prepared consumers.
             * @throws std::invalid_argument For malformed native data or a non-source/non-weight owner.
             * @throws std::logic_error For missing or mismatched physical admission.
             */
            [[nodiscard]] static WeightStorage adopt(Transform transform, std::size_t rows,
                std::size_t columns, AlignedVector<std::uint8_t> bytes, PhysicalMemoryOwner owner);

            /**
             * @brief Derive a native matrix directly from its retained mapping authority.
             * @param transform Source-declared online transform.
             * @param rows Complete output-row count.
             * @param columns Complete reduction width.
             * @param region Exact source mapping retained until the last consumer retires.
             * @param byte_offset Absolute byte coordinate inside that mapping.
             * @return Sealed storage borrowing only the exact matrix extent.
             * @throws std::invalid_argument For an absent/unaligned/out-of-bounds mapping or malformed source.
             *
             * The mapping's existing source admission remains authoritative;
             * this handle allocates no second tensor payload or resident ledger.
             */
            [[nodiscard]] static WeightStorage map(Transform transform, std::size_t rows,
                std::size_t columns, std::shared_ptr<const MmapRegion> region, std::size_t byte_offset);

            /**
             * @brief Share checked complete rows, preserving the original owner and exact policy.
             * @param first First row relative to this handle.
             * @param count Positive number of output rows.
             * @return Independently retained row handle, with no new payload allocation.
             * @throws std::out_of_range If the subset is not contained in this handle.
             */
            [[nodiscard]] WeightStorage rowSlice(std::size_t first, std::size_t count) const;

            /** @return Immutable borrowed numerical view; this owning handle must outlive the borrow. */
            [[nodiscard]] const SourceView &view() const & noexcept { return view_; }
            /** @brief A temporary owner cannot publish a longer-lived borrowed view. */
            const SourceView &view() const && = delete;
            /** @return Original CPU allocation endpoint; mapping storage uses the CPU source authority. */
            [[nodiscard]] DeviceId allocator() const noexcept;
            /** @return True only when the original physical owner is a retained file mapping. */
            [[nodiscard]] bool isMapped() const noexcept;

        private:
            struct Backing;
            /** @brief Bind a proven source subspan to its unchanged retained backing. */
            WeightStorage(std::shared_ptr<const Backing> backing, SourceView view) noexcept
                : backing_(std::move(backing)), view_(view) {}
            std::shared_ptr<const Backing> backing_; ///< Sole physical owner shared by immutable leases.
            SourceView view_; ///< Exact admitted subspan; never independently rebound.
        };
    }
} // namespace llaminar2
