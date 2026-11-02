/**
 * @file PooledNgramReplayTrace.h
 * @brief Shared original IQ4_NL rows and fixed diagnostic lookup input trajectories.
 *
 * These immutable inputs are test declarations, never a mirror of live cache
 * placement. Each head owns a disjoint source range. Distinct traces retain16
 * different IDs per token; duplicate-heavy replay is a separately named control.
 * Performance rows carry a full UInt64 source witness and deterministic native
 * code bytes. No floating-source conversion or source/model payload is required.
 */
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace llaminar2::test::ngram_replay {
inline constexpr std::size_t Width=160, RowBytes=90, Heads=16;
inline constexpr const char* NativePattern="provider_iq4nl_row_u64_identity_v2";
/** @brief Closed original-ID trajectories; banks are disjoint immutable input sets. */
enum class Pattern {DistinctA,DistinctB,DistinctC,MixedB,MixedC,DuplicateHeavy,Empty};
/** @brief Explicit source locality; scattered is primary, contiguous is a named control. */
enum class Locality {Scattered,Contiguous};
/** @brief Original source and retained/live token geometry, independent of cache placement. */
struct Shape {
    std::uint64_t table_rows=1048576;
    std::uint32_t capacity_tokens=1;
    std::uint32_t tokens=1;
    Locality locality=Locality::Scattered;
};
/**
 * @brief Author five original IQ4_NL blocks with an injective UInt64 row witness.
 * @param row Original UInt64 source coordinate, including meaningful high bits.
 * @param output Exactly90 original native bytes, never padded slot storage.
 * @throws std::invalid_argument For a different native row extent.
 */
inline void fillRow(std::uint64_t row,std::span<std::byte> output) {
    if(output.size()!=RowBytes)throw std::invalid_argument("IQ4_NL replay requires its original90-byte row");
    // Splitmix64 operates on native integer bits. Every resulting nibble is a
    // legal IQ4_NL code; scales stay exactly half1, never floating requantization.
    std::uint64_t state=row^0x726f775f69713432ULL;
    const auto next=[&]() {
        state+=0x9e3779b97f4a7c15ULL;auto z=state;
        z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;
        return z^(z>>31);
    };
    for(unsigned block=0;block<5;++block){output[block*18]=std::byte{0};output[block*18+1]=std::byte{0x3c};
        for(unsigned word=0;word<2;++word){const auto bits=next();for(unsigned byte=0;byte<8;++byte)
            output[block*18+2+word*8+byte]=static_cast<std::byte>((bits>>(byte*8))&0xff);}}
    // Codebook entries are distinct. These16decoded columns therefore carry all
    // original64source bits, detecting neighboring/head/bank/high-bit row swaps.
    for(unsigned byte=0;byte<8;++byte)output[2+byte]=static_cast<std::byte>((row>>(byte*8))&0xff);

}
/**
 * @brief Produce one fixed original-order sixteen-head UInt64 request.
 * @param shape Declared source and captured/live token bounds.
 * @param pattern Exact trace bank; mixed keeps12heads inA and alternates4cold heads.
 * @return Original IDs, token-major/head-minor, or empty for the explicit control.
 * @throws std::invalid_argument For incomplete source/head/capacity geometry.
 */
inline std::vector<std::uint64_t> ids(Shape shape,Pattern pattern) {
    if(!shape.capacity_tokens || shape.tokens>shape.capacity_tokens || shape.capacity_tokens>512 ||
       shape.table_rows%Heads || shape.table_rows/Heads<=3ull*(shape.capacity_tokens+1))
        throw std::invalid_argument("Ngram replay requires bounded token geometry and disjoint original head ranges");
    const auto stride=shape.table_rows/Heads;
    constexpr std::uint64_t multiplier=40503;
    if((shape.locality!=Locality::Scattered && shape.locality!=Locality::Contiguous) ||
       (shape.locality==Locality::Scattered && std::gcd(multiplier,stride)!=1))
        throw std::invalid_argument("Scattered replay requires its declared bijective per-head permutation");
    switch(pattern){case Pattern::DistinctA:case Pattern::DistinctB:case Pattern::DistinctC:case Pattern::MixedB:case Pattern::MixedC:case Pattern::DuplicateHeavy:case Pattern::Empty:break;
        default:throw std::invalid_argument("Unknown ngram replay pattern");}
    if(pattern==Pattern::Empty)return {};
    std::vector<std::uint64_t> result(std::size_t{shape.tokens}*Heads);
    for(std::uint32_t token=0;token<shape.tokens;++token)for(std::size_t head=0;head<Heads;++head){
        unsigned bank=0;
        switch(pattern){case Pattern::DistinctA:case Pattern::DuplicateHeavy:break;
        case Pattern::DistinctB:bank=1;break;case Pattern::DistinctC:bank=2;break;
        case Pattern::MixedB:bank=head<12?0:1;break;case Pattern::MixedC:bank=head<12?0:2;break;
        case Pattern::Empty:throw std::logic_error("Empty replay reached nonempty input author");
        default:throw std::invalid_argument("Unknown ngram replay pattern");}
        const auto position=pattern==Pattern::DuplicateHeavy?token%2:token;
        const auto original=std::uint64_t{bank}*(shape.capacity_tokens+1)+position+1;
        const auto offset=shape.locality==Locality::Contiguous?original:(multiplier*original+17*head+13)%stride;
        result[std::size_t{token}*Heads+head]=head*stride+offset;
    }
    return result;
}
/** @brief Count immutable distinct input IDs without inspecting any runtime value.
 * @param original Fixed input trace. @return Exact original unique request cardinality. */
inline std::size_t uniqueCount(std::span<const std::uint64_t> original) {
    std::vector<std::uint64_t> sorted(original.begin(),original.end());std::sort(sorted.begin(),sorted.end());
    return static_cast<std::size_t>(std::unique(sorted.begin(),sorted.end())-sorted.begin());
}
} // namespace llaminar2::test::ngram_replay
