/**
 * @file PrefixRestoreSourceRetirement.h
 * @brief Exact-event source retirement at bounded prefix RAM admission boundaries.
 *
 * Cache metadata eviction does not release source aliases owned by an asynchronous
 * restore. Its producer alone can prove those reads complete. RAM admission asks
 * that authority to retire completed aliases before consulting the existing arena;
 * this interface owns neither storage, a capacity ledger, nor GPU execution state.
 */
#pragma once

namespace llaminar2
{
    /** @brief Borrowed restore authority; it must outlive its participant-local cache. */
    class IPrefixRestoreSourceRetirement
    {
    public:
        /** @brief Destroy the interface without changing any native event lifetime. */
        virtual ~IPrefixRestoreSourceRetirement() = default;

        /**
         * @brief Drop only source aliases whose exact native reads have completed.
         * @throws std::runtime_error or std::logic_error for an invalid completion authority.
         * This is nonblocking. Consumer ordering publications retain their own lifetime.
         */
        virtual void retireCompletedPrefixRestoreSources() const = 0;
    };
}
