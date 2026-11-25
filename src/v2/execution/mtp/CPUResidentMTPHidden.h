/**
 * @file CPUResidentMTPHidden.h
 * @brief Slot-local publication metadata over one PMA-owned CPU hidden-state root.
 *
 * The arena owns every float. This owner only tracks validity and ordering on
 * the runner thread; it is neither a scheduler nor a second memory ledger.
 * Main and draft publications occupy disjoint banks, independent of archival
 * and of compact sidecar transaction buffers. Retirement invalidates both banks.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace llaminar2
{
/** @brief CPU-only resident hidden publications; borrowed storage must outlive this object. */
class CPUResidentMTPHidden final
{
public:
    /** @brief Main state survives draft-chain overwrites. */
    enum class Bank : std::uint8_t { Main, Draft };
    /** @brief Immutable receipt; it never exposes bytes after a publication edge. */
    struct Lease
    {
        const CPUResidentMTPHidden *owner = nullptr;
        std::size_t index = 0;
        std::uint64_t generation = 0;
    };

    /**
     * @brief Bind exactly two banks of admitted resident slots, without owning bytes.
     * @param storage Exact flattened arena root, never an allocation capacity guess.
     * @param slots Admitted request capacity.
     * @param width Model residual width.
     * @throws std::invalid_argument For absent or inconsistent geometry.
     */
    CPUResidentMTPHidden(std::span<float> storage, std::size_t slots, std::size_t width)
        : storage_(storage), slots_(slots), width_(width)
    {
        if (!slots || !width || slots > std::numeric_limits<std::size_t>::max() / 2 / width ||
            storage.size() != 2 * slots * width)
            throw std::invalid_argument("CPU resident MTP hidden root has inconsistent geometry");
        publications_.resize(2 * slots);
    }

    /** @brief Copy one produced row into its authoritative bank and publish a fresh receipt. */
    Lease publish(int slot, Bank bank, std::span<const float> source)
    {
        const auto index = checkedIndex(slot, bank);
        if (source.size() != width_)
            throw std::invalid_argument("CPU resident MTP hidden publication has wrong residual width");
        auto &publication = publications_[index];
        requireAdvance(publication);
        // All producers execute on the same runner thread. Publication follows
        // the complete copy, so an invalid source cannot expose a partial row.
        std::copy(source.begin(), source.end(), storage_.begin() + index * width_);
        ++publication.generation;
        publication.current = true;
        return {this, index, publication.generation};
    }

    /** @brief Read exactly a current publication; stale or foreign receipts fail closed. */
    std::span<const float> read(const Lease &lease) const
    {
        if (lease.owner != this || lease.index >= publications_.size())
            throw std::logic_error("CPU resident MTP hidden lease has a foreign owner");
        const auto &publication = publications_[lease.index];
        if (!publication.current || publication.generation != lease.generation)
            throw std::logic_error("CPU resident MTP hidden lease is stale or retired");
        return storage_.subspan(lease.index * width_, width_);
    }

    /** @brief Invalidate both banks before reuse; retained bytes do not imply a live publication. */
    void retire(int slot)
    {
        const auto main = checkedIndex(slot, Bank::Main);
        const auto draft = checkedIndex(slot, Bank::Draft);
        requireAdvance(publications_[main]);
        requireAdvance(publications_[draft]);
        for (const auto index : {main, draft})
        {
            ++publications_[index].generation;
            publications_[index].current = false;
        }
    }

private:
    /** @brief One publication version, not a duplicate model-state value. */
    struct Publication { std::uint64_t generation = 0; bool current = false; };
    /** @brief Reject wrap rather than authenticating an ancient receipt. */
    static void requireAdvance(const Publication &publication)
    {
        if (publication.generation == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("CPU resident MTP hidden publication generation exhausted");
    }
    /** @brief Resolve a bank only within admitted slot geometry. */
    std::size_t checkedIndex(int slot, Bank bank) const
    {
        if (slot < 0 || static_cast<std::size_t>(slot) >= slots_ ||
            (bank != Bank::Main && bank != Bank::Draft))
            throw std::invalid_argument("CPU resident MTP hidden slot or bank is invalid");
        return static_cast<std::size_t>(slot) + (bank == Bank::Draft ? slots_ : 0);
    }
    std::span<float> storage_;
    std::size_t slots_;
    std::size_t width_;
    std::vector<Publication> publications_;
};
}
