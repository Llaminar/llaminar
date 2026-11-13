/**
 * @file PooledImmutableNativeReplayBytes.h
 * @brief Aligned exact-byte storage for recorded native scalar replay inputs.
 *
 * Native owner/frame/progress records require 64-byte alignment even on the CPU.
 * A byte vector does not supply that contract. This fixture owner follows the
 * aligned-block storage already used by native scalar fixtures, while exposing
 * only the original logical extent. Padding is never uploaded or compared as
 * part of an original GPU root. These are diagnostic recorded inputs, not a
 * live execution-state mirror or an admission/reservation authority.
 */
#pragma once
#include <cstddef>
#include <utility>
#include <vector>
namespace llaminar2::test {
/** @brief Recorded native bytes whose copies retain the original alignment contract. */
class PooledImmutableNativeReplayBytes final {
public:
    static constexpr std::size_t Alignment=64;
    /** @brief Represent an empty recorded extent without allocating bytes. */
    PooledImmutableNativeReplayBytes()=default;
    /** @brief Allocate aligned diagnostic storage for an exact original extent.
     * @param bytes Original logical root bytes, excluding allocation padding.
     * @throws std::bad_alloc If diagnostic storage cannot be allocated.
     * @throws std::length_error If the recorded extent exceeds vector limits.
     */
    explicit PooledImmutableNativeReplayBytes(std::size_t bytes)
        :bytes_(bytes),blocks_(bytes/Alignment+(bytes%Alignment!=0)){}
    /** @brief Copy all recorded bytes into separately aligned diagnostic storage.
     * @param other Original recorded input.
     */
    PooledImmutableNativeReplayBytes(const PooledImmutableNativeReplayBytes& other)=default;
    /** @brief Preserve the destination on failed diagnostic copy allocation.
     * @param other Original recorded input.
     * @return This aligned owner after copying.
     */
    PooledImmutableNativeReplayBytes& operator=(const PooledImmutableNativeReplayBytes& other){
        if(this!=&other){PooledImmutableNativeReplayBytes copy(other);swap(copy);}return *this;
    }
    /** @brief Transfer the recorded extent and leave an empty valid source.
     * @param other Original recorded input owner.
     */
    PooledImmutableNativeReplayBytes(PooledImmutableNativeReplayBytes&& other)noexcept
        :bytes_(std::exchange(other.bytes_,0)),blocks_(std::move(other.blocks_)){}
    /** @brief Transfer recorded storage without changing alignment or logical extent.
     * @param other Original recorded input owner.
     * @return This aligned owner after transfer.
     */
    PooledImmutableNativeReplayBytes& operator=(PooledImmutableNativeReplayBytes&& other)noexcept{
        if(this!=&other){bytes_=std::exchange(other.bytes_,0);blocks_=std::move(other.blocks_);}return *this;
    }
    /** @return Original logical bytes; the rounded CPU allocation is never a GPU copy extent. */
    std::size_t size()const noexcept{return bytes_;}
    /** @return 64-byte-aligned diagnostic bytes, or null for the empty extent. */
    std::byte* data()noexcept{return bytes_?reinterpret_cast<std::byte*>(blocks_.data()):nullptr;}
    /** @return Same aligned immutable recorded input, or null for the empty extent. */
    const std::byte* data()const noexcept{return bytes_?reinterpret_cast<const std::byte*>(blocks_.data()):nullptr;}
private:
    /** @brief Match original native record alignment without extending the logical root. */
    struct alignas(Alignment) Block {std::byte bytes[Alignment];};
    /** @brief Exchange complete diagnostic ownership, never individual root fields.
     * @param other Another recorded-byte owner.
     */
    void swap(PooledImmutableNativeReplayBytes& other)noexcept{std::swap(bytes_,other.bytes_);blocks_.swap(other.blocks_);}
    std::size_t bytes_=0;
    std::vector<Block> blocks_;
};
} // namespace llaminar2::test
