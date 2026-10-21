/**
 * @file AdaptiveImmutableRowCacheTestFixture.h
 * @brief Tiny setup-owned arrays exercising the actual shared adaptive metadata operations.
 *
 * This device-free fixture does not emulate a directory, replacement policy,
 * source authentication or GPU submission. It owns precisely the declared
 * record types, invokes production Operations, and inspects external invariants.
 * Simulated source success advances only the integer algorithm's validation
 * frontier; native/source/admission publication remains a separate test gate.
 */
#pragma once
#include "kernels/qwen4exp/AdaptiveImmutableRowCacheOps.h"
#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
namespace llaminar2::test::adaptive {
namespace ops=qwen4exp::adaptive_lookup;
/** @brief Setup-only physical record owners, with no production GPU-state shadow. */
template<class Slot> struct ActualMetadata final {
    std::vector<Slot> slots;
    std::vector<ops::RadixNode> nodes;
    std::vector<std::uint32_t> epochs;
    std::vector<std::uint16_t> counts;
    ops::OwnerControl owner{};
    ops::FrameControl frame{};
    ops::Metadata<Slot> view;
    /** @brief Bind the exact production structs; small test capacity bounds setup allocations. */
    ActualMetadata(std::uint32_t capacity,std::uint64_t sourceRows=1024,
        std::uint32_t protectedLimit=1,std::uint64_t aging=1000000)
        :slots(capacity),nodes(capacity?capacity-1:0),epochs(capacity?64:0),counts(capacity?64:0),
         view{slots.data(),nodes.data(),capacity?&owner:nullptr,&frame,epochs.data(),counts.data(),
             capacity,std::min(protectedLimit,capacity?capacity-1:0),capacity?16u:0u,sourceRows,aging} {
        auto actual=operations();
        for(std::uint32_t i=0;i<capacity;++i)actual.initializeSlot(i);
        for(std::uint32_t i=0;i+1<capacity;++i)actual.initializeNode(i);
        actual.initializeOwner();actual.initializeFrame();
    }
    ActualMetadata(const ActualMetadata &)=delete;
    ActualMetadata &operator=(const ActualMetadata &)=delete;
    /** @return The same integer operations called by native adapters, without copying metadata. */
    ops::Operations<Slot> operations(){return ops::Operations<Slot>(view);}
    /** @brief Fail an unexpectedly invalid fixture operation without continuing into unsafe indices. */
    static void require(bool valid){if(!valid)throw std::logic_error("Actual adaptive operation rejected fixture trace");}
    /**
     * @brief Complete one integer-policy batch, preserving first occurrence and all initial pins.
     * @param ids Original coordinate order, including duplicates.
     * @return Per-unique-row decisions in first-occurrence order, including declined reservations.
     *
     * The unique list is only bounded input-frame metadata. Actual find() owns
     * lookup and actual observe() chooses every victim/admission. No reference
     * cache, frequency estimator or alternate victim implementation is present.
     */
    std::vector<ops::Decision> complete(std::span<const std::uint64_t> ids){
        auto actual=operations();require(actual.begin(ids.size()));
        std::vector<std::uint64_t> unique;
        std::vector<ops::Decision> decisions;std::uint64_t sourceRows=0;
        for(const auto row:ids){
            if(std::find(unique.begin(),unique.end(),row)!=unique.end())continue;
            unique.push_back(row);const auto slot=actual.find(row);
            ops::Decision d;
            if(slot!=ops::None){d={slot,slots[slot].generation};require(actual.pinInitial(d.slot,d.generation));}
            else {++sourceRows;}
            decisions.push_back(d);
        }
        require(actual.beginPolicy(unique.size()));
        for(const auto row:ids){
            const auto at=static_cast<std::size_t>(std::find(unique.begin(),unique.end(),row)-unique.begin());
            decisions[at]=actual.observe(row,decisions[at]);require(frame.status==ops::Status::Success);
        }
        require(actual.prepare());
        validateResponse(sourceRows);
        for(const auto d:decisions)if(d.slot!=ops::None &&
            ops::Operations<Slot>::state(slots[d.slot])==ops::SlotState::Reserved)require(actual.makeReady(d));
        for(const auto d:decisions)require(actual.release(d));
        require(actual.finish());return decisions;
    }
    /**
     * @brief Simulate only a successful protocol frontier for the pure integer algorithm.
     * @param sourceRows Exact unique initial misses, including unreserved all-pinned responses.
     *
     * This writes test-owned frame metadata and follows actual phase operations;
     * it does not authenticate source bytes, ACKs, device admission or events.
     * Source-free frames alone may skip publication and response-copy phases.
     */
    void validateResponse(std::uint64_t sourceRows){
        auto actual=operations();frame.source_rows=sourceRows;
        if(sourceRows){
            require(actual.transition(ops::Phase::Prepared,ops::Phase::Published));
            require(actual.transition(ops::Phase::Published,ops::Phase::ResponseCopied));
            require(actual.transition(ops::Phase::ResponseCopied,ops::Phase::Validated));
        }else {require(actual.transition(ops::Phase::Prepared,ops::Phase::Validated));}
    }
    /** @brief Inspect list uniqueness, reciprocal links, stable ordering and exact directory membership. */
    void expectInvariants(){
        if(slots.empty()){EXPECT_EQ(frame.held_pins,0u);return;}
        auto actual=operations();std::vector<unsigned> seen(slots.size(),0);
        std::size_t linked=0;
        for(unsigned list=0;list<3;++list){
            std::uint32_t previous=ops::None,current=owner.head[list];
            std::size_t visited=0;
            while(current!=ops::None){
                ASSERT_LT(current,slots.size());ASSERT_LT(visited++,slots.size());
                const auto &slot=slots[current];EXPECT_EQ(++seen[current],1u);++linked;
                EXPECT_EQ(slot.previous,previous);
                EXPECT_EQ(static_cast<unsigned>(ops::Operations<Slot>::stream(slot)),list);
                EXPECT_NE(ops::Operations<Slot>::state(slot),ops::SlotState::Empty);
                if(previous!=ops::None){EXPECT_LT(slots[previous].touched,slot.touched);}
                previous=current;current=slot.next;
            }
            EXPECT_EQ(owner.tail[list],previous);
        }
        std::size_t occupied=0,protectedCount=0;
        for(std::uint32_t i=0;i<slots.size();++i){
            const auto state=ops::Operations<Slot>::state(slots[i]);
            if(state==ops::SlotState::Empty){EXPECT_EQ(seen[i],0u);continue;}
            ++occupied;EXPECT_EQ(seen[i],1u);
            if(ops::Operations<Slot>::stream(slots[i])==ops::Stream::Protected)++protectedCount;
            if(state==ops::SlotState::Ready){EXPECT_EQ(actual.find(slots[i].row),i);}
            else {EXPECT_EQ(actual.find(slots[i].row),ops::None);}
        }
        EXPECT_EQ(occupied,linked);EXPECT_EQ(owner.protected_count,protectedCount);
        EXPECT_LE(protectedCount,view.protected_limit);EXPECT_EQ(frame.status,ops::Status::Success);
    }
};
} // namespace llaminar2::test::adaptive
