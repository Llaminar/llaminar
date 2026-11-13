/**
 * @file PooledImmutableDirectedMessageGeometry.h
 * @brief Pure admitted role extents shared by directed bindings and remote graph dispatch declarations.
 *
 * A source bank's capacity may exceed one owner's addressed message maximum.
 * These extents describe admitted message geometry, never the device-owned live
 * transfer count. Original native root offsets remain the geometry authority.
 */
#pragma once
#include "kernels/qwen4exp/PooledImmutableNativePlan.h"
#include <limits>
#include <stdexcept>
namespace llaminar2::qwen4exp {
/** @brief Identity-free one-owner wire geometry, with no fabricated table or source generation. */
class PooledImmutableDirectedMessageGeometry final {
public:
    /** @brief Derive exact original capacity from the same native root-layout implementation. */
    static PooledImmutableDirectedMessageGeometry fromGeometry(const PooledImmutablePhysicalOwnerGeometry& physical,
        const PooledImmutableCoordinatorGeometry& coordinator,std::uint32_t coordinate){
        const auto native=PooledImmutableOwnerNativePlan::rootOffsets(physical,coordinator,coordinate);
        return {native.reply_capacity,physical.rowBytes(),coordinator.parameters().participants()};
    }
    /** @brief Project one already checked actual native plan, preserving its addressed owner capacity. */
    static PooledImmutableDirectedMessageGeometry fromPlan(const PooledImmutableOwnerNativePlan& plan){
        const auto& p=plan.parameters();return {p.native.reply_capacity,p.row_bytes,p.owner.participants()};
    }
    /** @return Immutable admitted owner reply capacity. */
    std::uint32_t replyCapacity()const noexcept{return replies_;}
    /** @return Addressed command/grant/fill/row/release bank capacity; zero means no payload endpoint. */
    std::size_t commands()const{return product(replies_,sizeof(adaptive_lookup::pooled::OwnerCommand));}
    /** @return Addressed free/victim allocation decisions. */
    std::size_t decisions()const{return product(replies_,sizeof(adaptive_lookup::pooled::AllocationDecision));}
    /** @return Addressed grant bank maximum, never whole coordinator capacity. */
    std::size_t grants()const{return product(replies_,sizeof(adaptive_lookup::pooled::Grant));}
    /** @return Addressed source fill bank maximum. */
    std::size_t fills()const{return product(replies_,sizeof(pooled_native::Fill));}
    /** @return Addressed native row payload maximum. */
    std::size_t rows()const{return product(replies_,row_bytes_);}
    /** @return Addressed reader-release payload maximum. */
    std::size_t releases()const{return product(replies_,sizeof(adaptive_lookup::pooled::ReleaseReceipt));}
    /** @return Complete initial-pin quorum metadata, including C0 owners. */
    std::size_t pinHeaders()const{return product(participants_,sizeof(adaptive_lookup::pooled::PinHeader));}
    /** @return Complete protocol quorum metadata, independent of optional payload capacity. */
    std::size_t freeReceipts()const{return product(participants_,sizeof(adaptive_lookup::pooled::FreeAllocationReceipt));}
    /** @return Complete publication quorum, including metadata-only owners. */
    std::size_t publicationHeaders()const{return product(participants_,sizeof(adaptive_lookup::pooled::PublishHeader));}
private:
    /** @brief Checked multiplication before any message or resource declaration can be admitted. */
    static std::size_t product(std::size_t a,std::size_t b){
        if(a && b>std::numeric_limits<std::size_t>::max()/a)throw std::overflow_error("Directed owner message extent overflow");
        return a*b;
    }
    PooledImmutableDirectedMessageGeometry(std::uint32_t replies,std::size_t rowBytes,std::uint32_t participants)
        :replies_(replies),row_bytes_(rowBytes),participants_(participants){
        if(!rowBytes || !participants)throw std::invalid_argument("Directed owner message geometry is incomplete");
    }
    std::uint32_t replies_;std::size_t row_bytes_;std::uint32_t participants_;
};
}
