/**
 * @file CPUMappedPublicationRoot.h
 * @brief Immutable CPU mapped-publication backing declared before admission.
 *
 * One host allocator owns the physical pages; one exact GPU consumes their
 * stable native alias. TransferEngine owns the page rounding and materializes
 * this declaration without a caller-owned lease or a second live ledger.
 */
#pragma once
#include "planning/PhysicalMemoryBOM.h"
namespace llaminar2
{
/** @brief Exact ActivationArena root for an existing mapped CPU publication bank. */
class CPUMappedPublicationRoot final
{
public:
    /** @brief Declare mapped storage without native allocation or capacity admission.
     * @param payload_bytes Positive live storage capacity retained for publication.
     * @param host_allocator Exact CPU resource whose backing pages are admitted.
     * @param consumer Exact local CUDA or ROCm alias consumer.
     * @throws std::invalid_argument For incomplete or foreign endpoint identity.
     * @throws std::overflow_error For unrepresentable mapped backing geometry.
     */
    CPUMappedPublicationRoot(size_t payload_bytes, DeviceId host_allocator, DeviceId consumer);
    /** @return Requested storage capacity, independent of allocation padding. */
    size_t payloadBytes() const noexcept { return payload_bytes_; }
    /** @return Exact native page-rounded host allocation extent. */
    size_t physicalBytes() const noexcept { return physical_bytes_; }
    /** @return Sole physical CPU allocator. */
    DeviceId hostAllocator() const noexcept { return host_allocator_; }
    /** @return Exact GPU alias consumer. */
    DeviceId consumer() const noexcept { return consumer_; }
    /** @return Canonical owner for these existing publication pages. */
    PhysicalMemoryOwner owner() const noexcept { return PhysicalMemoryOwner::ActivationArena; }
    bool operator==(const CPUMappedPublicationRoot &) const = default;
private:
    size_t payload_bytes_;
    size_t physical_bytes_;
    DeviceId host_allocator_;
    DeviceId consumer_;
};
}
