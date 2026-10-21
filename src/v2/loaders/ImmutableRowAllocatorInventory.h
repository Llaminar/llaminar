/**
 * @file ImmutableRowAllocatorInventory.h
 * @brief Non-owning physical allocator projection over one immutable placement.
 *
 * A mixed placement's GPU consumer and CPU source-service identities already
 * live in its sole checked layout. This two-span view joins those original
 * records without retaining a duplicate owner map or allocating setup storage.
 */
#pragma once
#include "planning/PhysicalMemoryAuthority.h"
#include <iterator>
#include <span>
namespace llaminar2
{
/** @brief Read-only concatenation of existing stable allocator records. */
class ImmutableRowAllocatorInventory final
{
public:
    /** @brief Standard forward iterator retaining only immutable source spans and a position. */
    class iterator final
    {
    public:
        using iterator_category=std::forward_iterator_tag;
        using value_type=PhysicalMemoryAllocatorIdentity;
        using difference_type=std::ptrdiff_t;
        using reference=const value_type &;
        using pointer=const value_type *;
        iterator()=default;
        /** @return The original allocator record, never a copied owner authority. */
        reference operator*() const { return index_<first_.size()?first_[index_]:second_[index_-first_.size()]; }
        /** @return Pointer to the original immutable record. */
        pointer operator->() const { return &**this; }
        /** @return This iterator advanced by one physical owner. */
        iterator &operator++() { ++index_;return *this; }
        /** @return Previous position while advancing by one owner. */
        iterator operator++(int) { auto prior=*this;++*this;return prior; }
        /** @return Exact position and original inventory identity equality. */
        bool operator==(const iterator &other) const noexcept {
            return index_==other.index_ && first_.data()==other.first_.data() && first_.size()==other.first_.size() &&
                second_.data()==other.second_.data() && second_.size()==other.second_.size();
        }
    private:
        friend class ImmutableRowAllocatorInventory;
        /** @brief Only an inventory creates an iterator over its two borrowed spans. */
        iterator(std::span<const value_type> first,std::span<const value_type> second,std::size_t index)
            :first_(first),second_(second),index_(index) {}
        std::span<const value_type> first_,second_;
        std::size_t index_=0;
    };
    /** @brief Borrow one or two original immutable inventories without making another stored map. */
    ImmutableRowAllocatorInventory(std::span<const PhysicalMemoryAllocatorIdentity> first,
        std::span<const PhysicalMemoryAllocatorIdentity> second={}) :first_(first),second_(second) {}
    /** @return Original allocator count. */
    std::size_t size() const noexcept { return first_.size()+second_.size(); }
    /** @return Whether there is no physical owner. */
    bool empty() const noexcept { return size()==0; }
    /** @return Original first allocator; checked placements guarantee a nonempty inventory. */
    const PhysicalMemoryAllocatorIdentity &front() const { return *begin(); }
    /** @return First original allocator position. */
    iterator begin() const noexcept { return {first_,second_,0}; }
    /** @return One-past-last original allocator position. */
    iterator end() const noexcept { return {first_,second_,size()}; }
private:
    std::span<const PhysicalMemoryAllocatorIdentity> first_,second_;
};
} // namespace llaminar2
