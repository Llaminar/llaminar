/**
 * @file W4A4GPUTransformPolicy.h
 * @brief Immutable cold derivation of the native GPU Hadamard launch policy.
 *
 * A sealed complete-original-K source supplies the transform geometry. The
 * pinned nonzero quadratic residues fit two words because source admission
 * bounds Paley order by128. These words and the exact normalization are capture
 * metadata, not live GPU execution state, a lookup allocation or a host mirror.
 * This policy neither owns device addresses nor certifies their readiness;
 * the existing prepared-weight and arena authorities retain those duties.
 */
#pragma once

#include "W4A4DeviceViews.h"
#include "tensors/W4A4MatrixContract.h"
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <stdexcept>

namespace llaminar2::w4a4
{
    /** @brief Sealed host metadata for exactly one source-owned GPU H transform. */
    class GPUHadamardPolicy final
    {
    public:
        /** @brief Complete pointer-independent identity of the derived transform arithmetic. */
        struct CaptureIdentity
        {
            std::uint32_t arithmetic_version; ///< Pinned online-A4 arithmetic version.
            int original_columns;            ///< Complete original K, never a reduction shard.
            int rotation_length;             ///< Original source transform width.
            int paley, sylvester;             ///< Pinned ascending factorization.
            std::uint32_t normalization_bits; ///< Exact binary32 normalization embedded in kernels.
            std::array<std::uint64_t, 2> residues; ///< Nonzero quadratic residues, including unused zero bits.
            bool operator==(const CaptureIdentity &) const = default;
        };

        /**
         * @brief Derive immutable launch scalars from an already admitted H source.
         * @param source Complete native source contract, not an unvalidated geometry hint.
         * @return Sealed factorization, residue words and pinned binary32 normalization.
         * @throws std::invalid_argument For identity sources or incompatible host rounding.
         * @throws std::logic_error For a source factorization outside the shared device ABI.
         *
         * This is cold setup only. Volatile scalar boundaries retain the two
         * FP32 operations even if the caller's host compiler permits extended
         * intermediates; no per-invocation allocation or device work occurs.
         */
        [[nodiscard]] static GPUHadamardPolicy fromSource(const MatrixContract &source)
        {
            if (source.transform() != Transform::PaleySylvester || !source.hadamard())
                throw std::invalid_argument("GPU W4A4H policy requires an admitted Hadamard source");
            if (std::fegetround() != FE_TONEAREST)
                throw std::invalid_argument("GPU W4A4H policy requires round-to-nearest-even setup");
            const auto geometry = *source.hadamard();
            if (geometry.paley < 1 || geometry.paley > 128)
                throw std::logic_error("Admitted W4A4H factorization exceeds native residue-word ABI");
            std::array<std::uint64_t, 2> residues{};
            if (geometry.paley > 1)
            {
                const int prime = geometry.paley - 1;
                for (int value = 1; value < prime; ++value)
                {
                    const int residue = (value * value) % prime;
                    residues[residue / 64] |= std::uint64_t{1} << (residue % 64);
                }
            }
            volatile float root = std::sqrt(static_cast<float>(geometry.length));
            volatile float normalization = 1.0f / root;
            return GPUHadamardPolicy(CaptureIdentity{
                kArithmeticVersion, source.columns(), geometry.length, geometry.paley, geometry.sylvester,
                std::bit_cast<std::uint32_t>(static_cast<float>(normalization)), residues});
        }

        /** @return Exact immutable arithmetic identity; storage/event identities remain separate. */
        [[nodiscard]] const CaptureIdentity &captureIdentity() const noexcept { return identity_; }

        /**
         * @brief Apply only sealed arithmetic fields to an independently bound launch view.
         * @param view Borrowed physical operands whose original K must already match this source.
         * @throws std::invalid_argument When a caller presents a local-K or foreign matrix view.
         *
         * Pointer, stride, physical-row and lifetime checks belong to the cold
         * operand binding. This operation neither changes those addresses nor
         * turns raw pointers into an ownership or event-publication certificate.
         */
        void applyTo(device::HadamardView &view) const
        {
            if (view.columns != identity_.original_columns)
                throw std::invalid_argument("GPU W4A4H view must retain complete original K");
            view.rotation_length = identity_.rotation_length;
            view.paley = identity_.paley;
            view.sylvester = identity_.sylvester;
            view.normalization = std::bit_cast<float>(identity_.normalization_bits);
            view.residues_low = identity_.residues[0];
            view.residues_high = identity_.residues[1];
        }

    private:
        /** @brief Seal only metadata derived from the existing complete source authority. */
        explicit GPUHadamardPolicy(CaptureIdentity identity) noexcept : identity_(identity) {}
        const CaptureIdentity identity_; ///< Immutable policy; never a device-state shadow.
    };
} // namespace llaminar2::w4a4
