/**
 * @file DiskExtentQuota.h
 * @brief Typed finite quota and exact extent identity for disk tier storage.
 *
 * Design §11.3: disk extents have checked finite quota, exact namespace/file
 * identity, generation, content integrity and replacement/retirement ownership,
 * with old+new extents, in-flight writes, metadata and rounding accounted
 * concurrently. This authority is PMA-independent bookkeeping of ADMITTED
 * extent rights (the physical bytes remain owned by the storage authority);
 * a logical quota alone is not proof of physical availability, so callers must
 * still hold the real storage lease (plan P3/P5).
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace llaminar2
{
/** @brief One admitted disk extent with exact identity and integrity digest. */
struct DiskExtentKey
{
    std::uint64_t generation = 0;
    std::uint64_t namespace_id = 0;
    std::uint64_t digest = 0;
    std::size_t bytes = 0;
    bool operator==(const DiskExtentKey &) const = default;
};

/**
 * @brief Bounded quota ledger; overflow/subtraction are checked and typed.
 *
 * The journal capacity is an ADMITTED membership value, never an arbitrary
 * constant: the caller sizes it from its typed disk-extent membership (the
 * same authority that prices the storage), so an impossible extent set can
 * never hide behind a silently large record table.
 */
class DiskExtentQuota final
{
public:
    /**
     * @brief Open a quota ledger with the admitted byte bound and journal size.
     * @param quota_bytes Admitted byte capacity (0 = permanently off: admits nothing).
     * @param journal_capacity Admitted extent-record membership (must be positive).
     */
    DiskExtentQuota(std::size_t quota_bytes, std::size_t journal_capacity);
    /** @return Admitted extent bytes, metadata/rounding inclusive. */
    std::size_t reserved() const noexcept { return reserved_; }
    /** @return Admitted journal capacity (typed membership, not a constant). */
    std::size_t capacity() const noexcept { return entries_.size(); }
    /**
     * @brief Admit one extent if quota remains; never exposes partial state.
     * @return True on admission; false without mutating when quota or journal full.
     */
    bool admit(DiskExtentKey extent);
    /** @brief Non-mutating admission query (same rules as admit). The
     * demotion planner uses it as the real-overflow gate before any move. */
    bool wouldAdmit(DiskExtentKey extent) const;
    /** @brief Retire an exact admitted extent; unknown identity refuses. */
    bool retire(DiskExtentKey extent);
    /** @return Whether the exact extent identity is currently admitted. */
    bool contains(DiskExtentKey extent) const;
    /** @return Admitted extent count. */
    std::size_t count() const noexcept { return count_; }
    /** @return Total admitted bytes. */
    std::size_t bytesAdmitted() const noexcept { return reserved_; }
private:
    std::size_t quota_;
    std::size_t reserved_ = 0;
    std::size_t count_ = 0;
    std::vector<DiskExtentKey> entries_; ///< Admitted-membership-sized journal.
};
} // namespace llaminar2
