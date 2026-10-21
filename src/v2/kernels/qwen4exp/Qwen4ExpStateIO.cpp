/**
 * @file Qwen4ExpStateIO.cpp
 * @brief CPU and static validation for exact auxiliary-state transfer extents.
 *
 * Native words preserve token integers and arbitrary finite float bits. Dynamic
 * checkpoint payload capacity is retained, while inactive tail rows stay unread
 * and unwritten, matching the device implementation's traffic contract.
 */
#include "Qwen4ExpStateIO.h"
#include <cstring>
#include <limits>
#include <stdexcept>
namespace llaminar2::qwen4exp
{
void validateQwenStateTransfer(const QwenStateTransfer &p)
{
    if(!p.source || !p.destination || !p.rowBytes || p.rowBytes%4 || p.capacityRows<=0 ||
        p.rowBytes>std::numeric_limits<std::size_t>::max()/std::size_t(p.capacityRows))
        throw std::invalid_argument("QFN auxiliary native copy has invalid geometry");
    const auto source=reinterpret_cast<std::uintptr_t>(p.source),destination=reinterpret_cast<std::uintptr_t>(p.destination);
    const auto extent=p.rowBytes*std::size_t(p.capacityRows);
    if(source%4 || destination%4 || (p.count && reinterpret_cast<std::uintptr_t>(p.count)%4) ||
        (p.publishedRows && reinterpret_cast<std::uintptr_t>(p.publishedRows)%4) ||
        extent>std::numeric_limits<std::uintptr_t>::max()-source || extent>std::numeric_limits<std::uintptr_t>::max()-destination ||
        (source!=destination && source<destination+extent && destination<source+extent))
        throw std::invalid_argument("QFN auxiliary transfer needs aligned disjoint or identical native intervals");
    if(p.copiedBytes) {
        const auto counter=reinterpret_cast<std::uintptr_t>(p.copiedBytes);
        const auto count=reinterpret_cast<std::uintptr_t>(p.count);
        const auto published=reinterpret_cast<std::uintptr_t>(p.publishedRows);
        const auto metadataOverlaps=[&](std::uintptr_t word) {
            return word && (word>std::numeric_limits<std::uintptr_t>::max()-4 ||
                (counter<word+4 && word<counter+8));
        };
        if(counter%alignof(std::uint64_t) || counter>std::numeric_limits<std::uintptr_t>::max()-8 ||
            metadataOverlaps(count) || metadataOverlaps(published) ||
            (counter<source+extent && source<counter+8) ||
            (counter<destination+extent && destination<counter+8))
            throw std::invalid_argument("QFN live-copy byte observation must be aligned and disjoint from payloads/count metadata");
    }
    if(p.aggregateCopiedBytes.bound()) {
        const auto counter=reinterpret_cast<std::uintptr_t>(p.aggregateCopiedBytes.words);
        const auto overlaps=[&](const void*address,std::size_t bytes) {
            const auto begin=reinterpret_cast<std::uintptr_t>(address);
            return address && (begin>std::numeric_limits<std::uintptr_t>::max()-bytes ||
                (counter<begin+bytes && begin<counter+8));
        };
        if(!p.aggregateCopiedBytes.valid() || counter>std::numeric_limits<std::uintptr_t>::max()-8 ||
            overlaps(p.source,extent) || overlaps(p.destination,extent) || overlaps(p.count,4) ||
            overlaps(p.publishedRows,4) || overlaps(p.copiedBytes,8))
            throw std::invalid_argument("QFN terminal byte counter must be aligned and disjoint from payload/metadata");
    }
    switch(p.rowSource) {
    case QwenStateRowSource::Fixed:
        if(p.fixedRows<0 || p.fixedRows>p.capacityRows)throw std::invalid_argument("QFN auxiliary fixed row count exceeds admitted capacity");
        break;
    case QwenStateRowSource::CanonicalModuloFour:case QwenStateRowSource::CheckpointHeader:case QwenStateRowSource::ExecutionCount:
        if(!p.count)throw std::invalid_argument("QFN auxiliary copy is missing its execution-owned live count");
        break;
    default:throw std::invalid_argument("QFN auxiliary copy has an invalid row-count authority");
    }
}
void validateQwenStateFill(const QwenStateFill &p)
{
    if(!p.destination || reinterpret_cast<std::uintptr_t>(p.destination)%4 || !p.words || p.words>std::numeric_limits<std::size_t>::max()/4)
        throw std::invalid_argument("QFN auxiliary fill has invalid native geometry");
}
void cpuQwenStateTransfer(const QwenStateTransfer &p)
{
    validateQwenStateTransfer(p);
    const int count=p.rowSource==QwenStateRowSource::Fixed ? p.fixedRows :
        p.rowSource==QwenStateRowSource::CanonicalModuloFour ? *p.count%4 : *p.count;
    if(count<0 || count>p.capacityRows)throw std::invalid_argument("QFN auxiliary checkpoint contains invalid live rows");
    if(p.publishedRows)*p.publishedRows=count;
    if(count)std::memmove(p.destination,p.source,std::size_t(count)*p.rowBytes);
    if(p.copiedBytes)*p.copiedBytes=std::uint64_t(count)*p.rowBytes;
    if(p.aggregateCopiedBytes.bound() && !addNativeCounter64(p.aggregateCopiedBytes,std::uint64_t(count)*p.rowBytes))
        throw std::overflow_error("QFN terminal copy byte counter overflow");
}
void cpuQwenStateFill(const QwenStateFill &p)
{
    validateQwenStateFill(p);
    auto *output=static_cast<std::uint32_t *>(p.destination);
    for(std::size_t word=0;word<p.words;++word)output[word]=p.bits;
}
}
