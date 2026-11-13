/**
 * @file QwenSparseSelectionCases.h
 * @brief Small immutable score distributions and an independent sort oracle.
 *
 * This diagnostic oracle is never a production implementation. It compares
 * numeric FP32 scores directly, preserving lower original IDs for ties, then
 * sorts the chosen IDs. It does not call bounded key/radix helper code.
 */
#pragma once
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace llaminar2::qwen4exp::selection_test
{
inline constexpr std::array policies{QwenSparsePoolSelection::Bounded256_512,
    QwenSparsePoolSelection::Bounded512_512, QwenSparsePoolSelection::Bounded1024_1024};
enum class Distribution { Hash, Equal, Ascending, Descending, Alternating, ByteBoundary, FiniteExtremes,
    PersistentFourDigitFrontier, HotHistogramFrontier };
static_assert(int(Distribution::PersistentFourDigitFrontier)==7);
static_assert(int(Distribution::HotHistogramFrontier)==8);
inline constexpr std::array distributions{Distribution::Hash, Distribution::Equal, Distribution::Ascending,
    Distribution::Descending, Distribution::Alternating, Distribution::ByteBoundary, Distribution::FiniteExtremes,
    Distribution::PersistentFourDigitFrontier, Distribution::HotHistogramFrontier};
/** @brief Construct finite nonnegative immutable FP32 data without floating reductions. */
inline float score(int id, int row, Distribution distribution)
{
    switch (distribution)
    {
    case Distribution::Equal: return 0.0f; // Actual score producer publishes canonical +0.
    case Distribution::Ascending: return float(id / 3);
    case Distribution::Descending: return 16777216.0f - float(id / 3);
    case Distribution::Alternating: return id % 2 ? float(id % 521) : 1024.0f;
    case Distribution::ByteBoundary:
    {
        const std::uint32_t bits = 0x3f7fff00u + std::uint32_t(id % 1025);
        float result; std::memcpy(&result, &bits, sizeof(result)); return result;
    }
    case Distribution::HotHistogramFrontier:
    {
        // A single permanent high key leaves 511 low survivors. The larger
        // common-prefix populations stress histogram updates without changing
        // aligned compaction/digit work. This is not a global latency worst case.
        const std::uint32_t bits=id==0 ? 0x7f7fffffu : 0x01010101u+std::uint32_t(id-1);
        float result;std::memcpy(&result,&bits,sizeof(result));return result;
    }
    case Distribution::PersistentFourDigitFrontier:
    {
        // The first 511 keys stay strictly above a growing low-score frontier.
        // Base +1 leaves two keys in each aligned terminal high-byte bucket,
        // so aligned compactions need four score digits. Carry tails may need three.
        const std::uint32_t bits=id<511 ? 0x7f7fffffu : 0x01010101u+std::uint32_t(id-511);
        float result;std::memcpy(&result,&bits,sizeof(result));return result;
    }
    case Distribution::FiniteExtremes:
        return id % 3 == 0 ? std::numeric_limits<float>::max() :
            id % 3 == 1 ? std::numeric_limits<float>::denorm_min() : 0.0f;
    case Distribution::Hash:
    {
        std::uint32_t value = std::uint32_t(id) + 0x9e3779b9u * (std::uint32_t(row) + 1u);
        value ^= value >> 16; value *= 0x7feb352du; value ^= value >> 15;
        return float(value % 65537u) / 256.0f;
    }
    }
    return 0.0f;
}
/** @brief Fill each admitted score row; causal consumption is tested separately. */
inline std::vector<float> scores(int rows, int blocks, Distribution distribution)
{
    std::vector<float> result(std::size_t(rows) * blocks);
    for (int row = 0; row < rows; ++row)
        for (int block = 0; block < blocks; ++block)
            result[std::size_t(row) * blocks + block] = score(block, row, distribution);
    return result;
}
/**
 * @brief Derive exact diagnostic totals for the authored persistent-frontier fixture.
 * @param pools Live complete pools, at most the timing fixture's 65536-pool limit.
 * @param policy One of the three explicit bounded geometries.
 * @return Main scan, compactions, bank visits and executed score-digit totals.
 * @throws std::invalid_argument For an unknown policy or outside the authored range.
 * @details This is a closed-form workload identity, not a mirrored selector or a
 * correctness oracle. Every arriving low score exceeds the retained frontier,
 * so no main-scan key is rejected. After the first aligned compaction, another
 * occurs per ceil(Slack/Block) complete chunks, plus a possible terminal bank.
 * Conservation gives W=N-512+512C. Aligned final frontier bytes end01 and need
 * four digits; an unaligned N%256==255 tail ends00 and needs three for its last
 * compaction. Independent numeric sorting remains the membership authority.
 */
inline QwenSparseSelectionObservation persistentFrontierWork(int pools,QwenSparsePoolSelection policy)
{
    int block=0,slack=0;
    switch(policy){
    case QwenSparsePoolSelection::Bounded256_512:block=256;slack=512;break;
    case QwenSparsePoolSelection::Bounded512_512:block=512;slack=512;break;
    case QwenSparsePoolSelection::Bounded1024_1024:block=1024;slack=1024;break;
    default:throw std::invalid_argument("Persistent frontier work requires an explicit bounded policy");}
    if(pools<0 || pools>65536)throw std::invalid_argument("Persistent frontier outside its authored pool range");
    if(pools<=512)return {};
    const int first=((512+slack+block-1)/block)*block;
    const int stride=((slack+block-1)/block)*block;
    const int regular=pools<first ? 0 : 1+(pools-first)/stride;
    const int last=regular ? first+(regular-1)*stride : 0;
    const auto calls=std::uint64_t(regular+(pools>last));
    return {std::uint64_t(pools),calls,std::uint64_t(pools-512)+512*calls,
        4*calls-(pools%256==255 ? 1u : 0u)};
}

/** @brief Correctness-only inputs; existing performance distribution IDs remain unchanged. */
enum class Adversarial { SignedZero, CutoffOneULP };
/**
 * @brief Author zeros of both signs or a511-high/one-ULP-separated cutoff tie.
 * @details No production key/radix helper constructs these values. For the
 * cutoff,511 scores are exactly one float step above1, later candidates tie at1
 * except lower neighbors. The oracle must retain the lowest original IDs.
 */
inline std::vector<float> adversarialScores(int rows,int blocks,Adversarial distribution)
{
    std::vector<float> result(std::size_t(rows)*blocks);
    for(int row=0;row<rows;++row)for(int id=0;id<blocks;++id){
        const std::uint32_t bits=distribution==Adversarial::SignedZero
            ? ((id+row)%2?0x80000000u:0u)
            : (id<511?0x3f800001u:(id%17?0x3f800000u:0x3f7fffffu));
        std::memcpy(&result[std::size_t(row)*blocks+id],&bits,sizeof(bits));
    }
    return result;
}
/** @brief Independent numeric score/ID sort, exact pool expansion and raw tail. */
inline std::vector<std::int32_t> oracle(const float *scores, int visible)
{
    const int blocks = visible / QwenSparsePool;
    std::vector<int> ids(blocks);
    std::iota(ids.begin(), ids.end(), 0);
    if (blocks > QwenSparseBlocks)
    {
        std::sort(ids.begin(), ids.end(), [&](int a, int b)
        { return scores[a] > scores[b] || (scores[a] == scores[b] && a < b); });
        ids.resize(QwenSparseBlocks);
        std::sort(ids.begin(), ids.end());
    }
    std::vector<std::int32_t> result;
    for (int id : ids)
        for (int member = 0; member < QwenSparsePool; ++member) result.push_back(id * QwenSparsePool + member);
    for (int tail = 0; tail < visible % QwenSparsePool; ++tail) result.push_back(blocks * QwenSparsePool + tail);
    return result;
}
} // namespace llaminar2::qwen4exp::selection_test
