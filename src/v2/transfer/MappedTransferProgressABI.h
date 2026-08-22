/**
 * @file MappedTransferProgressABI.h
 * @brief Fixed-width node-local command ABI for retained GPU transfer epochs.
 *
 * ExpertOverlay maintenance publishes bounded copy commands into host pages
 * that are mapped into one local GPU.  A retained graph snapshots those
 * commands, copies every active slot in parallel, and release-publishes one
 * completion record per slot.  The ABI contains process-local device virtual
 * addresses and is therefore intentionally invalid for inter-node transport.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace llaminar2
{
    /** Binary identity of a mapped transfer-progress command (`MTPR`). */
    inline constexpr std::uint32_t kMappedTransferProgressMagic =
        0x5250544du;

    /** Version shared by commands, device claims, and completions. */
    inline constexpr std::uint32_t kMappedTransferProgressVersion = 2u;

    /** @return Generation-bound low identity word for a command snapshot. */
    constexpr std::uint32_t mappedTransferProgressGenerationMagic(
        std::uint64_t generation) noexcept
    {
        return kMappedTransferProgressMagic ^
               static_cast<std::uint32_t>(generation);
    }

    /** @return Generation-bound high identity word for a command snapshot. */
    constexpr std::uint32_t mappedTransferProgressGenerationVersion(
        std::uint64_t generation) noexcept
    {
        return kMappedTransferProgressVersion ^
               static_cast<std::uint32_t>(generation >> 32u);
    }

    /** Device-authored terminal status for one command generation. */
    enum class MappedTransferProgressError : std::uint32_t
    {
        None = 0u,
        InvalidIdentity = 1u,
        InvalidAddress = 2u,
        InvalidByteCount = 3u,
    };

    /**
     * @brief Host-authored immutable copy request for one permanent lane slot.
     *
     * The owner writes every field except @ref generation, then release-stores
     * a positive generation. The GPU system-acquires that word before reading
     * the remaining fields. Generation-bound identity words and exact bitwise
     * complements make every address/count field self-validating, so even a
     * broken platform-coherence observation is rejected before dereferencing a
     * process-local address. A slot cannot be reused until its matching
     * completion has been acquired by the owner. Keeping this record on its own
     * cache line prevents device completion traffic from contending with the
     * next host publication.
     */
    struct alignas(64) MappedTransferProgressCommand
    {
        /** Magic XOR the low generation bits; validates the release word. */
        std::uint32_t generation_magic =
            mappedTransferProgressGenerationMagic(0u);
        /** Version XOR the high generation bits; validates the release word. */
        std::uint32_t generation_version =
            mappedTransferProgressGenerationVersion(0u);
        /** Published last; zero means that this slot has never been armed. */
        std::uint64_t generation = 0u;
        /** Process-local source address in this GPU's address space. */
        std::uint64_t source_address = 0u;
        /** Process-local destination address in this GPU's address space. */
        std::uint64_t destination_address = 0u;
        /** Positive bytes, bounded by the epoch's immutable maximum. */
        std::uint64_t bytes = 0u;
        /** Exact complement checked before the source can be dereferenced. */
        std::uint64_t source_complement = ~std::uint64_t{0u};
        /** Exact complement checked before the destination can be dereferenced. */
        std::uint64_t destination_complement = ~std::uint64_t{0u};
        /** Exact complement checked before the byte count can be consumed. */
        std::uint64_t bytes_complement = ~std::uint64_t{0u};
    };

    /**
     * @brief Device-resident snapshot consumed by one retained graph replay.
     *
     * A claim kernel copies one command into ordinary device memory before the
     * payload kernel runs.  The host may consequently inspect only the separate
     * completion record without racing a device reread of mutable command pages.
     */
    struct alignas(64) MappedTransferProgressClaim
    {
        std::uint32_t generation_magic = 0u;
        std::uint32_t generation_version = 0u;
        std::uint64_t generation = 0u;
        std::uint64_t source_address = 0u;
        std::uint64_t destination_address = 0u;
        std::uint64_t bytes = 0u;
        std::uint64_t source_complement = 0u;
        std::uint64_t destination_complement = 0u;
        std::uint64_t bytes_complement = 0u;
    };

    /**
     * @brief Device-authored system-visible result for one command generation.
     *
     * The payload kernel writes status and byte count, executes a system-scope
     * fence, and publishes @ref completed_generation last.  A host acquire load
     * of that generation therefore covers the complete copied byte range and
     * both diagnostic fields.
     */
    struct alignas(64) MappedTransferProgressCompletion
    {
        /** Published last; monotonically matches a completed command. */
        std::uint64_t completed_generation = 0u;
        /** Exact bytes copied, or zero when @ref error is non-zero. */
        std::uint64_t completed_bytes = 0u;
        std::uint32_t error = static_cast<std::uint32_t>(
            MappedTransferProgressError::None);
        std::uint32_t reserved0 = 0u;
        std::uint64_t reserved[5] = {};
    };

    static_assert(sizeof(MappedTransferProgressCommand) == 64u);
    static_assert(sizeof(MappedTransferProgressClaim) == 64u);
    static_assert(sizeof(MappedTransferProgressCompletion) == 64u);
    static_assert(alignof(MappedTransferProgressCommand) == 64u);
    static_assert(alignof(MappedTransferProgressClaim) == 64u);
    static_assert(alignof(MappedTransferProgressCompletion) == 64u);
    static_assert(std::is_trivially_copyable_v<
                  MappedTransferProgressCommand>);
    static_assert(std::is_trivially_copyable_v<
                  MappedTransferProgressClaim>);
    static_assert(std::is_trivially_copyable_v<
                  MappedTransferProgressCompletion>);
} // namespace llaminar2
