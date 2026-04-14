/**
 * @file IBufferRegistration.h
 * @brief Buffer registration interface for collective backends
 *
 * Enables collective backends to track buffer locations for cross-device
 * communication. Backends that need to know where device buffers are
 * located can implement this interface.
 *
 * Usage Flow:
 * 1. Stage allocates device buffers during graph construction
 * 2. Stage registers buffers with the collective backend using a unique collective_id
 * 3. Backend stores buffer metadata (device, ptr, size)
 * 4. During execution, backend uses allreduceRegistered() with the collective_id
 * 5. Backend looks up buffer locations and performs the operation
 * 6. At shutdown, buffers are automatically unregistered
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../backends/DeviceId.h"
#include <cstddef>
#include <optional>
#include <string>

namespace llaminar2
{

    /**
     * @brief Metadata for a registered buffer
     *
     * Contains all information needed by collective backends to locate
     * and access a buffer on a specific device.
     */
    struct RegisteredBuffer
    {
        DeviceId device;   ///< Device where the buffer resides
        void *ptr;         ///< Device pointer to the buffer
        size_t size;       ///< Size of the buffer in bytes
        size_t bar_offset; ///< Offset within BAR region (legacy, unused)
        bool is_primary;   ///< True if this is the primary device in a collective

        /**
         * @brief Default constructor - creates an invalid/empty registration
         */
        RegisteredBuffer()
            : device(DeviceId::invalid()), ptr(nullptr), size(0), bar_offset(0), is_primary(false)
        {
        }

        /**
         * @brief Full constructor with all fields
         *
         * @param device Device where buffer resides
         * @param ptr Device pointer to buffer
         * @param size Buffer size in bytes
         * @param bar_offset Offset within BAR region (legacy, default 0)
         * @param is_primary True if primary device (default false)
         */
        RegisteredBuffer(DeviceId device, void *ptr, size_t size,
                         size_t bar_offset = 0, bool is_primary = false)
            : device(device), ptr(ptr), size(size), bar_offset(bar_offset), is_primary(is_primary)
        {
        }

        /**
         * @brief Check if this registration is valid
         * @return true if device is valid and ptr is not null
         */
        bool isValid() const
        {
            return device.is_valid() && ptr != nullptr;
        }
    };

    /**
     * @brief Interface for buffer registration in collective backends
     *
     * Collective backends that need to track buffer locations
     * implement this interface. Backends that don't need registration (like MPIBackend)
     * can use the default implementations that return success/empty.
     *
     * Thread Safety:
     * - Registration/unregistration should be done before collective operations
     * - Concurrent registration with the same collective_id is undefined
     * - Lookups during collective operations are safe
     */
    class IBufferRegistration
    {
    public:
        virtual ~IBufferRegistration() = default;

        /**
         * @brief Register a buffer for a collective operation
         *
         * Associates a device buffer with a collective operation identifier.
         * The collective_id should be unique per logical collective (e.g., "layer0_attn_allreduce").
         * Multiple devices can be registered under the same collective_id.
         *
         * @param collective_id Unique identifier for the collective operation
         * @param device Device where the buffer resides
         * @param buffer Device pointer to the buffer
         * @param size Size of the buffer in bytes
         * @return true on success, false on failure (e.g., duplicate registration)
         */
        virtual bool registerBuffer(const std::string &collective_id,
                                    DeviceId device,
                                    void *buffer,
                                    size_t size) = 0;

        /**
         * @brief Unregister a buffer for a collective operation
         *
         * Removes the buffer registration for the specified device.
         * Safe to call even if no buffer was registered.
         *
         * @param collective_id Identifier for the collective operation
         * @param device Device whose buffer should be unregistered
         */
        virtual void unregisterBuffer(const std::string &collective_id,
                                      DeviceId device) = 0;

        /**
         * @brief Get registration info for a buffer
         *
         * Looks up the registration for a specific device in a collective.
         *
         * @param collective_id Identifier for the collective operation
         * @param device Device to look up
         * @return RegisteredBuffer if found, std::nullopt if not registered
         */
        virtual std::optional<RegisteredBuffer> getBuffer(const std::string &collective_id,
                                                          DeviceId device) const = 0;

        /**
         * @brief Check if this backend requires buffer registration
         *
         * Backends that need to track buffer locations
         * return true. Backends that work with buffer pointers passed directly
         * to allreduce() (like MPIBackend) return false.
         *
         * When this returns true, callers should:
         * 1. Call registerBuffer() before any allreduceRegistered() calls
         * 2. Use allreduceRegistered() instead of allreduce()
         *
         * @return true if the backend requires buffer registration
         */
        virtual bool requiresBufferRegistration() const = 0;
    };

} // namespace llaminar2
