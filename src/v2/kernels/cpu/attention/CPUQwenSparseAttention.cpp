/**
 * @file CPUQwenSparseAttention.cpp
 * @brief Allocation-free CPU QSA with persistent causal pools and native sparse reads.
 *
 * Pool, query, score, and attention work is distributed over independent rows,
 * heads, or completed blocks. Each dot and each value accumulation keeps one
 * arithmetic order for every positive worker count and grouped query geometry.
 * The heap stores only 512 block indices; attention reads at most 2051 native
 * cache rows rather than scanning a dense bank and testing a sparse mask.
 * CPUQwenSparseExecution seals original bindings and executes native fragments;
 * no selected cache head or weighted-value row is decoded into a FP32 mirror.
 */
#include "kernels/qwen4exp/QwenSparseQueryRowWork.h"
#include "../../qwen4exp/QwenSparseReplay.h"
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "kernels/qwen4exp/QwenSparseMath.h"
#include "kernels/qwen4exp/QwenSparseBoundedSelection.h"
#include "utils/OpenMPUtils.h"
#include "CPUQwenSparseExecution.h"
#include "CPUQwenSparseIndexerProgram.h"
#include "utils/CPUFeatures.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace llaminar2::qwen4exp
{
namespace
{
using cpu_sparse::validateReplay;
using cpu_sparse::validateReplayRoot;

/** @brief Fail canonical counter corruption without throwing through an OpenMP single/barrier.
 * @param stage Original publication whose checked terminal counter overflowed.
 */
[[noreturn]] void counterOverflow(const char*stage)noexcept
{
    std::fputs("ERROR: CPU QSA canonical counter overflow in ",stderr);std::fputs(stage,stderr);std::fputc('\n',stderr);std::abort();
}

/** @brief Reject consumed numeric corruption without masking it through ReLU/selection or a collective exception.
 * @param stage Original producer whose declared finite arithmetic domain failed.
 */
[[noreturn]] void invalidNumeric(const char*stage)noexcept
{
    std::fputs("ERROR: CPU QSA invalid consumed numeric data in ",stderr);std::fputs(stage,stderr);std::fputc('\n',stderr);std::abort();
}

/** @brief Resolve the production ISA once per entrypoint; scalar execution is not advertised.
 * @return Original available compiled/runtime native program.
 * @throws std::invalid_argument For Scalar, foreign or unavailable execution.
 */
ISALevel nativeISA()
{
    const auto isa = activeISALevel();
    ::llaminar2::cpu::qsa_native::requireNativeProgram(isa);
    return isa;
}

/** @brief Prepare one rounded indexer head with fixed-tree RMSNorm and split-half RoPE.
 * @param input Original128-coordinate projection or produced pool.
 * @param weight Original converted128-coordinate normalization weights.
 * @param position Original absolute RoPE position; @param p Admitted representation/factor policy.
 * @param output Exclusive prepared head; full same-head input reuse is permitted.
 * @details Cast boundaries and the128-element FP32 butterfly remain the declared
 * indexer program. Its produced representation stages are not decoded KV rows.
 */
void prepare(const float *input, const float *weight, int position,
             const QwenSparseIndexer &p, float *output)
{
    // The representation is dispatched once, never once per coordinate. Every
    // admitted mode owns the same original 128-square tree and cast boundaries.
    switch(p.representation)
    {
    case QwenSparseScalar::FP32:return ::llaminar2::cpu::qsa_native::indexer::prepare<QwenSparseScalar::FP32>(input,weight,position,p,output);
    case QwenSparseScalar::FP16:return ::llaminar2::cpu::qsa_native::indexer::prepare<QwenSparseScalar::FP16>(input,weight,position,p,output);
    case QwenSparseScalar::BF16:return ::llaminar2::cpu::qsa_native::indexer::prepare<QwenSparseScalar::BF16>(input,weight,position,p,output);
    }
    ::llaminar2::cpu::qsa_native::rejectProgram();
}

/** @brief Dispatch one original four-row pool to the exact native representation program.
 * @param raw Original raw128-coordinate roots; no projected operand is copied.
 * @param weight Original converted key normalization weights.
 * @param position Original block-start RoPE position; @param p Original admitted policy.
 * @param output Exclusive prepared128-coordinate pool.
 */
void preparePool(const std::array<const float*,4>&raw,const float*weight,int position,const QwenSparseIndexer&p,float*output)noexcept
{
    switch(p.representation)
    {
    case QwenSparseScalar::FP32:return ::llaminar2::cpu::qsa_native::indexer::preparePool<QwenSparseScalar::FP32>(raw,weight,position,p,output);
    case QwenSparseScalar::FP16:return ::llaminar2::cpu::qsa_native::indexer::preparePool<QwenSparseScalar::FP16>(raw,weight,position,p,output);
    case QwenSparseScalar::BF16:return ::llaminar2::cpu::qsa_native::indexer::preparePool<QwenSparseScalar::BF16>(raw,weight,position,p,output);
    }
    ::llaminar2::cpu::qsa_native::rejectProgram();
}
/** @brief Tie original pool loads to the completed preceding head without changing the root.
 * @param original Original admitted128-coordinate pool root.
 * @param completed Completed preceding head sum, or positive zero before head0.
 * @return The identical original pointer bits, with a compiler-only dependency.
 * @details Streaming each head's fragments bounds vector liveness instead of
 * retaining the entire pool across heads. This identity copies no operand and
 * establishes no allocation, lease, publication or runtime synchronization.
 */
[[gnu::always_inline]] inline const float* followScoreHead(const float*original,float completed)noexcept
{
#if defined(__aarch64__)
    asm volatile("" : "+r"(original) : "w"(completed));
#elif defined(__x86_64__)
    asm volatile("" : "+r"(original) : "x"(completed));
#else
#error "CPU QSA score-head ordering requires a native AArch64 or x86-64 GNU-compatible register ABI"
#endif
    return original;
}
/** @brief Score the complete original pool frontier for one already-admitted ranking row.
 * @param query Original produced four-head query row, exactly4*128 coordinates.
 * @param pools Original persistent128-coordinate pool rows; never a decoded mirror.
 * @param scores Exclusive original output row for the declared complete pools.
 * @param blocks Original visible complete-pool count, strictly greater than512.
 * @param isa Original execution-admitted native arithmetic program.
 * @details One call covers every block/head in this score phase, not a complete
 * M1 indexer operation. Keeping its original roots/geometry in value arguments
 * bounds GPR liveness independently of the caller's selection heap and policy.
 * No per-dot call, operand copy, allocation or new collective edge is introduced.
 * Every head retains the declared ordered native FMA chains/fold, finite check,
 * ReLU, increasing-head addition and final separate FP32 scale.
 */
[[gnu::noinline]] void scorePools(const float*query,const float*pools,float*scores,int blocks,ISALevel isa)noexcept
{
    for(int block=0;block<blocks;++block)
    {
        float score=0.0f;const float*headPool=pools+std::size_t(block)*128;
        for(int head=0;head<4;++head)
        {
            headPool=followScoreHead(headPool,score);
            const auto value=::llaminar2::cpu::qsa_native::dot<128>(query+std::size_t(head)*128,headPool,isa);
            if(!std::isfinite(value))invalidNumeric("indexer native score");
            score+=std::max(0.0f,value);
        }
        if(!std::isfinite(score))invalidNumeric("indexer head-score sum");
        scores[block]=score*0.08838834764831844055f;
    }
}
/** @brief Resolve a transaction-relative raw key without touching future rows.
 * @param p Original tail/current-projection roots and replay.
 * @param token Original logical token; @return Exact original raw128-coordinate row.
 */
const float *rawKey(const QwenSparseIndexer &p, int token)
{
    const int prefix = p.replay->prefix_tokens;
    return token < prefix
        ? p.partial_keys + std::size_t(token - (prefix / 4) * 4) * 128
        : p.raw_keys + std::size_t(token - prefix) * 128;
}

/**
 * @brief Select one actual row, preserving the installed heap and exact tail writes.
 * @param p Original persistent score/replay/output bindings.
 * @param row Original physical row, including inactive rows.
 * @return False only for a malformed explicit bounded-candidate score frontier.
 */
bool selectRow(const QwenSparseIndexer &p, int row)
{
    auto *observation = p.selection_observations ? p.selection_observations + row : nullptr;
    if (observation) *observation = {0, 0, 0, 0};
    if (row >= p.replay->live_rows) { p.selected_count[row] = 0; return true; }
    const int visible = p.replay->prefix_tokens + row + 1;
    const int blocks = visible / QwenSparsePool;
    const int keep = std::min(blocks, QwenSparseBlocks);
    auto *selected = p.selected + std::size_t(row) * QwenSparseSelected;
    // No-ranking replay intentionally owns no score bank. Even forming a
    // nonzero offset from its null root is UB, despite never reading a score.
    const auto *scores = blocks>QwenSparseBlocks ? p.scores+std::size_t(row)*p.block_capacity : nullptr;
    std::array<std::int32_t, QwenSparseBlocks> heap;
    if (blocks <= QwenSparseBlocks)
        for (int block = 0; block < blocks; ++block) heap[block] = block;
    else if (p.pool_selection == QwenSparsePoolSelection::Installed)
    {
        const auto better = [&](int a, int b)
        {
            return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
        };
        int size = 0;
        for (int block = 0; block < blocks; ++block)
        {
            if (size < keep)
            {
                heap[size++] = block;
                std::push_heap(heap.begin(), heap.begin() + size, better);
            }
            else if (better(block, heap[0]))
            {
                std::pop_heap(heap.begin(), heap.begin() + size, better);
                heap[size - 1] = block;
                std::push_heap(heap.begin(), heap.begin() + size, better);
            }
        }
        std::sort(heap.begin(), heap.begin() + keep);
    }
    else
    {
        bool valid = false;
        switch (p.pool_selection)
        {
        case QwenSparsePoolSelection::Bounded256_512:
            valid = bounded_selection::selectScalar<256, 512>(scores, blocks, heap.data(), observation); break;
        case QwenSparsePoolSelection::Bounded512_512:
            valid = bounded_selection::selectScalar<512, 512>(scores, blocks, heap.data(), observation); break;
        case QwenSparsePoolSelection::Bounded1024_1024:
            valid = bounded_selection::selectScalar<1024, 1024>(scores, blocks, heap.data(), observation); break;
        default: return false;
        }
        if (!valid) return false;
    }
    for (int i = 0; i < keep; ++i)
        for (int member = 0; member < QwenSparsePool; ++member)
            selected[i * QwenSparsePool + member] = heap[i] * QwenSparsePool + member;
    for (int tail = 0; tail < visible % QwenSparsePool; ++tail)
        selected[keep * QwenSparsePool + tail] = blocks * QwenSparsePool + tail;
    p.selected_count[row] = keep * QwenSparsePool + visible % QwenSparsePool;
    return true;
}
/** @brief Fail a stale no-ranking program before any CPU-owned output publication.
 * @param p Original admitted query program and replay.
 * @throws std::invalid_argument If the original no-ranking horizon is exceeded.
 */
void validateQueryProgramReplay(const QwenSparseIndexer &p)
{
    if (p.query_demand == QwenSparseQueryDemand::ProvenNoRanking &&
        static_cast<std::int64_t>(p.replay->prefix_tokens) + p.replay->live_rows > QwenSparseFirstRankingLogicalRow)
        throw std::invalid_argument("QSA ProvenNoRanking replay exceeds admitted query omission boundary");
}
} // namespace

/** @brief Prepare original causal pools/queries and select sorted native cache rows.
 * @param p Existing host-owned replay and admitted projection/pool/selection bindings.
 * @throws std::invalid_argument Before worksharing for unsupported static policy,
 * ISA, replay geometry or a stale no-ranking query program.
 * @details Consumed numeric corruption is fatal inside its owner; it cannot be
 * masked by ReLU or reported through a worker-local flag after release.
 */
void cpuQwenSparseIndexer(const QwenSparseIndexer &p)
{
    validateReplayRoot(p.replay);
    validateQwenSparseIndexer(p);
    const auto isa = nativeISA();
    validateReplay(*p.replay, p.row_capacity, p.token_capacity);
    validateQueryProgramReplay(p);
    if(p.transaction_prior && p.transaction_prior->row_capacity!=p.row_capacity)
        throw std::invalid_argument("QSA indexer transaction prior row geometry changed");
    const auto replay = *p.replay;
    if (p.rotary_source==QwenSparseRotarySource::TablePlanes &&
        static_cast<std::int64_t>(replay.position_origin) + replay.prefix_tokens + replay.live_rows > p.position_capacity)
        throw std::invalid_argument("CPU Qwen sparse indexer: RoPE position capacity exceeded");
    // CPU owns these words. Resolve once before the OpenMP region; no query
    // operand outside this original row/head span is an authored value.
    const bool noRanking = p.query_demand == QwenSparseQueryDemand::ProvenNoRanking;
    const auto queryHeads = noRanking ? ProjectionRowSpan{0,0} :
        p.query_demand==QwenSparseQueryDemand::RankingOnly
            ? qwenSparseIndexerQueryWork(p).repeatEachRow(4).resolveHost()
            : ProjectionRowSpan{0,replay.live_rows*4};
    const int first_block = replay.prefix_tokens / 4;
    const int complete_blocks = (replay.prefix_tokens + replay.live_rows) / 4;
    auto preparation = [&]()
    {
#pragma omp for schedule(static)
        for (int block = first_block; block < complete_blocks; ++block)
        {
            const std::array<const float*,4>raw{rawKey(p,block*4),rawKey(p,block*4+1),rawKey(p,block*4+2),rawKey(p,block*4+3)};
            preparePool(raw,p.key_norm,replay.position_origin+block*4,p,p.pooled_keys+std::size_t(block)*128);
        }
#pragma omp for schedule(static)
        for (int row_head = queryHeads.first; row_head < queryHeads.end; ++row_head)
            prepare(p.query + std::size_t(row_head) * 128, p.query_norm,
                    replay.position_origin + replay.prefix_tokens + row_head / 4, p,
                    p.prepared_query + std::size_t(row_head) * 128);
    };
    OMP_WORKSHARE_REGION(preparation);

    auto selection = [&]()
    {
#pragma omp for schedule(static)
        for (int row = 0; row < p.row_capacity; ++row)
        {
            if (row < replay.live_rows)
            {
                const int blocks = (replay.prefix_tokens + row + 1) / QwenSparsePool;
                if (!noRanking && blocks > QwenSparseBlocks)
                    scorePools(p.prepared_query+std::size_t(row)*4*128,p.pooled_keys,
                        p.scores+std::size_t(row)*p.block_capacity,blocks,isa);
            }
            if(!selectRow(p,row))invalidNumeric("indexer bounded selection");
        }
    };
    OMP_WORKSHARE_REGION(selection);
}

/** @brief Select original CPU score frontiers without changing their arithmetic.
 * @param p Existing score/replay roots and exclusive selected/count/diagnostic outputs.
 * @throws std::invalid_argument Before stores for malformed geometry or consumed scores.
 * @details Only live ranking rows are inspected; inactive/nonranking score poison
 * is not an authored input and must not be scanned after collective completion.
 */
void cpuQwenSparseSelectPools(const QwenSparseIndexer &p)
{
    validateReplayRoot(p.replay);
    validateQwenSparseSelection(p);
    validateReplay(*p.replay, p.row_capacity, p.token_capacity);
    validateQueryProgramReplay(p);
    // No exception may escape OpenMP. Validate only scores actually consumed by
    // a long active row before changing ANY selection/count/diagnostic output.
    for (int row = 0; row < p.replay->live_rows; ++row)
    {
        const int blocks = (p.replay->prefix_tokens + row + 1) / QwenSparsePool;
        if (blocks <= QwenSparseBlocks) continue;
        for (int block = 0; block < blocks; ++block)
            if (!bounded_selection::validScoreBits(detail::floatBits(p.scores[std::size_t(row) * p.block_capacity + block])))
                throw std::invalid_argument("CPU QSA selection requires finite nonnegative live scores");
    }
    auto selection = [&]()
    {
#pragma omp for schedule(static)
        for(int row=0;row<p.row_capacity;++row)
            if(!selectRow(p,row))invalidNumeric("selected frontier changed after admission");
    };
    OMP_WORKSHARE_REGION(selection);
}

/** @brief Execute the admitted original-cache CPU QSA program with one owner per row/head.
 * @param p Existing arena/cache bindings and original host-owned replay.
 * @throws std::invalid_argument Before worksharing for invalid policy, ISA,
 * replay, selected mappings, byte geometry or publication exclusions.
 */
void cpuQwenSparseAttention(const QwenSparseAttention &p)
{
    const auto invocation=cpu_sparse::Invocation::bind(p);
    const auto attention=[&]()
    {
#pragma omp for schedule(static)
        for(int rowHead=0;rowHead<invocation.rowHeads();++rowHead)invocation.execute(rowHead);
    };
    // The empty loop still participates in the retained team's completion edge.
    OMP_WORKSHARE_REGION(attention);
}

/** @brief Publish the exact original replay/raw-tail snapshot under one retained-team owner.
 * @param p Original CPU replay/tail and distinct admitted snapshot/traffic/counter outputs.
 * @throws std::invalid_argument Before worksharing for invalid original bindings/replay.
 * @details Counter corruption is always fatal, never an exception through a collective.
 */
void cpuQwenSparseSnapshotPrior(const QwenSparsePriorSnapshot& p)
{
    validateReplayRoot(p.replay);
    validateQwenSparsePriorSnapshot(p);
    validateReplay(*p.replay,p.row_capacity,p.token_capacity);
    const int tail_rows=p.replay->live_rows ? p.replay->prefix_tokens%4 : 0;
    const auto snapshot=[&]() {
        p.output->replay=*p.replay;p.output->row_capacity=p.row_capacity;
        std::copy_n(p.accepted_partial,std::size_t(tail_rows)*128,p.output->partial);
        *p.traffic={16u,static_cast<std::uint32_t>(tail_rows*128*sizeof(float))};
        if(p.work_counters.bound() && !addQwenSparseHostCounters(p.work_counters.control,p.work_counters.words,
            {16u,std::uint64_t(tail_rows)*128*sizeof(float),0,0}))
            counterOverflow("original-tail snapshot");
    };
    OMP_SINGLE(snapshot);
}

/** @brief Commit the accepted original raw-tail suffix without changing its canonical frontier.
 * @param p Original replay/tail/candidates, canonical accepted-prefix authority and output.
 * @throws std::invalid_argument Before worksharing for invalid bindings or acceptance.
 * @details All potential overlapping raw sources are retained before the one
 * publication; no-write calls still complete the retained team's lexical edge.
 */
void cpuQwenSparseCommitPartial(const QwenSparsePartialCommit &p)
{
    validateReplayRoot(p.replay);
    validateQwenSparsePartialCommit(p);
    validateReplay(*p.replay,p.acceptance.candidates().rowsPerRequest(),p.transaction_prior ? p.token_capacity : std::numeric_limits<int>::max());
    if(p.transaction_prior && p.transaction_prior->row_capacity!=p.acceptance.candidates().rowsPerRequest())
        throw std::invalid_argument("QSA commit transaction prior row geometry changed");
    const int accepted=p.acceptance.resolveNative(0);
    if (accepted < 0 || accepted > p.replay->live_rows)
        throw std::invalid_argument("CPU Qwen sparse partial commit: acceptance exceeds live rows");
    const bool publish=p.replay->live_rows!=0 && (accepted!=0 || p.transaction_prior);
    auto commit = [&]()
    {
        // A no-write request still retires through the lexical single/barrier.
        if(!publish)return;
        const int total = p.replay->prefix_tokens + accepted;
        const int begin = (total / 4) * 4;
        float pending[4 * 128];
        for (int member = 0; member < total % 4; ++member)
        {
            const int token = begin + member;
            const float *source = token < p.replay->prefix_tokens
                ? p.prior_partial + std::size_t(token - (p.replay->prefix_tokens / 4) * 4) * 128
                : p.raw_keys + std::size_t(token - p.replay->prefix_tokens) * 128;
            std::copy_n(source, 128, pending + member * 128);
        }
        std::copy_n(pending, (total % 4) * 128, p.partial_keys);
        if(p.work_counters.bound() && !addQwenSparseHostCounters(p.work_counters.control,p.work_counters.words,
            {0,0,0,std::uint64_t(total%4)*128*sizeof(float)}))
            counterOverflow("accepted-tail commit");
    };
    OMP_SINGLE(commit);
}
namespace
{
/** @brief Typed CPU implementation of the compute-stage sparse primitive boundary. */
class CPUQwenSparseAttentionKernels final : public IQwenSparseAttentionKernels
{
public:
    /** @copydoc IQwenSparseAttentionKernels::indexer */
    void indexer(const QwenSparseIndexer &p, void *) const override { cpuQwenSparseIndexer(p); }
    /** @copydoc IQwenSparseAttentionKernels::attention */
    void attention(const QwenSparseAttention &p, void *) const override { cpuQwenSparseAttention(p); }
    /** @copydoc IQwenSparseAttentionKernels::prepareReplay */
    void prepareReplay(const QwenSparsePrepareReplay &p,void *) const override { cpuQwenSparsePrepareReplay(p); }
    /** @copydoc IQwenSparseAttentionKernels::snapshotPrior */
    void snapshotPrior(const QwenSparsePriorSnapshot& p,void*) const override { cpuQwenSparseSnapshotPrior(p); }
    /** @copydoc IQwenSparseAttentionKernels::commitPartial */
    void commitPartial(const QwenSparsePartialCommit &p, void *) const override { cpuQwenSparseCommitPartial(p); }
    /** @copydoc IQwenSparseAttentionKernels::retainCandidates */
    void retainCandidates(const QwenSparseCandidateRetention &p,void *) const override
    {
        cpuQwenStateTransfer(qwenSparseCandidateTransfer(p));
    }
};
}
/** @brief Construct the existing allocation-free-invocation CPU primitive bundle during setup.
 * @return One RAII-owned native bundle; no cache/workspace allocation occurs in its calls.
 */
std::unique_ptr<IQwenSparseAttentionKernels> createCPUQwenSparseAttentionKernels()
{
    return std::make_unique<CPUQwenSparseAttentionKernels>();
}
} // namespace llaminar2::qwen4exp
