/**
 * @file GDNDeviceStateBinding.h
 * @brief Non-owning bindings for cache-owned GPU GDN recurrent state.
 *
 * GDN short-convolution and recurrence kernels mutate state that must survive
 * switches between prefill, decode, and MTP verifier graphs.  The hybrid KV
 * cache therefore owns the storage for the complete model-session lifetime,
 * while each kernel receives only this typed, non-owning view.
 *
 * Keeping ownership outside the kernels makes the device-memory lifecycle
 * explicit: all banks are planned before graph capture, their addresses remain
 * stable, and a kernel cannot silently allocate a replacement when a binding
 * is absent or undersized.
 */

#pragma once

#include <algorithm>
#include <cstddef>

namespace llaminar2
{
    /**
     * @brief Stable device storage assigned to one stateful GDN kernel.
     *
     * LocalTP may need two scalar live-state geometries: a participant-local
     * bank for sharded prefill and a mirrored full bank for decode/MTP.  The
     * request bank is sized for the larger geometry and all configured request
     * slots, so changing geometry never changes a captured pointer.
     */
    struct GDNDeviceStateBinding
    {
        float *primary_state = nullptr;       ///< Cache-owned primary live state; may alias request zero.
        int primary_state_floats = 0;         ///< FP32 elements in primary_state.
        float *secondary_state = nullptr;     ///< Optional second LocalTP geometry.
        int secondary_state_floats = 0;       ///< FP32 elements in secondary_state.
        float *request_state_bank = nullptr;  ///< Packed request-local live states and canonical request zero.
        size_t request_state_bank_floats = 0; ///< Complete request-bank capacity.
        int request_capacity = 0;             ///< Maximum independent requests.

        /**
         * @brief Return the largest scalar state geometry represented here.
         */
        int largestStateFloats() const noexcept
        {
            return std::max(primary_state_floats, secondary_state_floats);
        }

        /**
         * @brief Validate pointer, size, and request-bank invariants.
         *
         * A secondary pointer is required exactly when a distinct secondary
         * size is declared.  The request bank must cover every configured
         * request at the largest scalar geometry because kernels may switch
         * between local and full state without reallocating it.
         */
        bool valid() const noexcept
        {
            if (!primary_state || primary_state_floats <= 0 ||
                !request_state_bank || request_capacity <= 0)
            {
                return false;
            }

            const bool has_secondary = secondary_state_floats > 0;
            if (has_secondary != (secondary_state != nullptr))
                return false;
            if (has_secondary && secondary_state_floats == primary_state_floats)
                return false;

            /*
             * A single state geometry has one owner: packed request slot zero.
             * Distinct LocalTP geometries cannot alias the packed bank because
             * request rows use the active geometry's stride; either alias would
             * permit one geometry to overwrite bytes owned by the other.
             */
            if (!has_secondary && primary_state != request_state_bank)
                return false;
            if (has_secondary &&
                (primary_state == request_state_bank ||
                 secondary_state == request_state_bank))
            {
                return false;
            }

            const size_t required_request_floats =
                static_cast<size_t>(request_capacity) *
                static_cast<size_t>(largestStateFloats());
            return request_state_bank_floats >= required_request_floats;
        }
    };
} // namespace llaminar2
