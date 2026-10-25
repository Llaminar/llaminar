/**
 * @file GDNPreparedQKBinding.h
 * @brief Explicit raw/prepared scalar GDN stage contracts over one arena root.
 *
 * A graph builder passes the same immutable binding to producer and consumer.
 * Buffer contracts and the original capture-storage leases own coherence and
 * retirement; this record neither allocates nor retains an accounting mirror.
 */
#pragma once
#include "kernels/gdn/GDNPreparedQK.h"
#include "memory/BufferId.h"
#include <cstdint>
#include <variant>
namespace llaminar2
{
class ITensor;
/** @brief Original raw post-convolution normalization program. */
struct GDNRawQKInput final {};
/** @brief One scalar Q-then-K activation, with its exact producer arithmetic. */
struct GDNPreparedQKBinding final
{
    ITensor *storage;
    static constexpr BufferId buffer_id=BufferId::GDN_PREPARED_QK;
    int key_heads,value_heads,key_width,value_width;
    bool normalize_qk;
    /**
     * @brief Resolve this original head geometry through the canonical capacity helper.
     * @return Exact scalar activation capacity.
     * @throws std::invalid_argument For invalid or overflowing original geometry.
     */
    [[nodiscard]] gdn::ScalarPreparedQKLayout layout() const
    {return gdn::ScalarPreparedQKLayout::fromHeads(key_heads,value_heads,key_width,value_width);}
};
/** @brief Cold program identity; absent prepared data can never select raw recovery. */
using GDNQKInputContract=std::variant<GDNRawQKInput,GDNPreparedQKBinding>;
/**
 * @brief Hash every added immutable producer/consumer binding field for capture identity.
 * @param contract Explicit raw program or complete compact activation binding.
 * @return Zero for the original raw program; otherwise the added capture identity.
 */
[[nodiscard]] inline std::uint64_t gdnQKCaptureIdentity(const GDNQKInputContract& contract)
{
    const auto *p=std::get_if<GDNPreparedQKBinding>(&contract);
    if(!p)return 0;
    std::uint64_t value=0x514b505245504152ULL;
    const auto mix=[&](std::uint64_t word){value^=word+0x9e3779b97f4a7c15ULL+(value<<6)+(value>>2);};
    mix(reinterpret_cast<std::uintptr_t>(p->storage));mix(static_cast<std::uint64_t>(p->buffer_id));
    mix(p->key_heads);mix(p->value_heads);mix(p->key_width);mix(p->value_width);mix(p->normalize_qk);
    return value;
}
}
