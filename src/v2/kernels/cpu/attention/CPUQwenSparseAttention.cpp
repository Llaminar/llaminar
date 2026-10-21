/**
 * @file CPUQwenSparseAttention.cpp
 * @brief Allocation-free CPU QSA with persistent causal pools and native sparse reads.
 *
 * Pool, query, score, and attention work is distributed over independent rows,
 * heads, or completed blocks. Each dot and each value accumulation keeps one
 * arithmetic order for every positive worker count and grouped query geometry.
 * The heap stores only 512 block indices; attention reads at most 2051 native
 * cache rows rather than scanning a dense bank and testing a sparse mask.
 */
#include "../../qwen4exp/QwenSparseReplay.h"
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "kernels/qwen4exp/QwenSparseMath.h"
#include "kernels/qwen4exp/QwenSparseBoundedSelection.h"
#include "utils/OpenMPUtils.h"
#include <atomic>
#include "utils/CPUFeatures.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <immintrin.h>
#include <limits>
#include <stdexcept>

namespace llaminar2::qwen4exp
{
namespace
{
/** @brief Validate CPU-owned dynamic replay metadata before entering worksharing. */
void validateReplay(const QwenSparseReplay &r, int rows, int tokens)
{
    if (r.prefix_tokens < 0 || r.live_rows < 0 || r.live_rows > rows ||
        static_cast<std::int64_t>(r.prefix_tokens) + r.live_rows > tokens || r.position_origin < 0 ||
        static_cast<std::int64_t>(r.position_origin) + r.prefix_tokens + r.live_rows > std::numeric_limits<int>::max())
        throw std::invalid_argument("CPU Qwen sparse replay: invalid causal prefix or live row count");
}

/** @brief Fixed AVX lane reduction, independent of the number of query rows. */
float dot(const float *a, const float *b, int dim, ISALevel isa)
{
#if defined(__AVX512F__)
    if (isa == ISALevel::AVX512)
    {
    __m512 sum = _mm512_setzero_ps();
    for (int d = 0; d < dim; d += 16)
        sum = _mm512_fmadd_ps(_mm512_loadu_ps(a + d), _mm512_loadu_ps(b + d), sum);
    alignas(64) float lanes[16];
    _mm512_store_ps(lanes, sum);
    for (int width = 8; width; width /= 2)
        for (int lane = 0; lane < width; ++lane) lanes[lane] += lanes[lane + width];
    return lanes[0];
    }
#endif
#if defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    for (int d = 0; d < dim; d += 8)
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), sum);
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    for (int width = 4; width; width /= 2)
        for (int lane = 0; lane < width; ++lane) lanes[lane] += lanes[lane + width];
    return lanes[0];
#else
#error "CPU Qwen sparse attention requires the project's AVX2 or AVX512 production ISA"
#endif
}

/** @brief Resolve the production ISA once per entrypoint; scalar execution is not advertised. */
ISALevel nativeISA()
{
    const auto isa = activeISALevel();
    if (isa != ISALevel::AVX2 && isa != ISALevel::AVX512)
        throw std::invalid_argument("CPU Qwen sparse attention requires an available AVX2 or AVX512 ISA");
    return isa;
}

/** @brief Prepare one rounded indexer head with fixed-tree RMSNorm and split-half RoPE. */
void prepare(const float *input, const float *weight, int position,
             const QwenSparseIndexer &p, float *output)
{
    float rounded[128], squares[128], normalized[128];
    for (int d = 0; d < 128; ++d)
    {
        rounded[d] = detail::roundRepresentation(input[d], p.representation);
        squares[d] = rounded[d] * rounded[d];
    }
    for (int width = 64; width; width /= 2)
        for (int d = 0; d < width; ++d) squares[d] += squares[d + width];
    const float inverse = 1.0f / std::sqrt(squares[0] / 128.0f + p.norm_epsilon);
    for (int d = 0; d < 128; ++d)
        normalized[d] = detail::roundRepresentation((rounded[d] * inverse) * weight[d], p.representation);
    const int half = p.rotary_dim / 2;
    for (int d = 0; d < 128; ++d)
    {
        if (d >= p.rotary_dim) { output[d] = normalized[d]; continue; }
        const int pair = d % half;
        float cosine,sine;
        detail::rotaryFactors(p.rotary_source,p.cosine,p.sine,position,pair,p.rotary_dim,p.rope_theta,cosine,sine,p.inverse_frequencies.data());
        const float c = detail::roundRepresentation(cosine, p.representation);
        const float s = detail::roundRepresentation(sine, p.representation);
        const float partner = d < half ? -normalized[d + half] : normalized[d - half];
        const float first = detail::roundRepresentation(normalized[d] * c, p.representation);
        const float second = detail::roundRepresentation(partner * s, p.representation);
        output[d] = detail::roundRepresentation(first + second, p.representation);
    }
}

/** @brief Resolve a transaction-relative raw key without touching future rows. */
const float *rawKey(const QwenSparseIndexer &p, int token)
{
    const int prefix = p.replay->prefix_tokens;
    return token < prefix
        ? p.partial_keys + std::size_t(token - (prefix / 4) * 4) * 128
        : p.raw_keys + std::size_t(token - prefix) * 128;
}

/** @brief Decode one selected head with eight-wide native SIMD and canonical packed codes. */
void readHead(const QwenSparseNativeBank &bank, int physical, int head, detail::BankRole role, float *output)
{
    const auto *bytes = static_cast<const unsigned char *>(bank.data) +
        std::size_t(physical) * bank.token_stride_bytes + std::size_t(head) * bank.head_stride_bytes;
    switch (bank.codec)
    {
    case TensorType::FP32:
        std::copy_n(reinterpret_cast<const float *>(bytes), 256, output); return;
    case TensorType::FP16:
        for (int d = 0; d < 256; d += 8)
            _mm256_storeu_ps(output + d, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(bytes + d * 2))));
        return;
    case TensorType::BF16:
        for (int d = 0; d < 256; d += 8)
            _mm256_storeu_ps(output + d, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(bytes + d * 2))), 16)));
        return;
    case TensorType::Q8_1:
        for (int block_index = 0; block_index < 8; ++block_index)
        {
            const auto &block = reinterpret_cast<const Q8_1Block *>(bytes)[block_index];
            const auto scale = _mm256_set1_ps(detail::decodeHalf(block.d));
            for (int d = 0; d < 32; d += 8)
                _mm256_storeu_ps(output + block_index * 32 + d, _mm256_mul_ps(scale, _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(block.qs + d))))));
        }
        return;
    case TensorType::Q16_1:
        for (int base = 0; base < 256; base += bank.q16_block_size)
        {
            const auto *block = bytes + std::size_t(base / bank.q16_block_size) * (8 + 2 * bank.q16_block_size);
            const auto scale = _mm256_set1_ps(*reinterpret_cast<const float *>(block));
            for (int d = 0; d < bank.q16_block_size; d += 8)
                _mm256_storeu_ps(output + base + d, _mm256_mul_ps(scale, _mm256_cvtepi32_ps(
                    _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(block + 8 + d * 2))))));
        }
        return;
    case TensorType::AQ8:
    {
        const auto &block = *reinterpret_cast<const AttentionKeyQ8Block<256> *>(bytes);
        const auto scale = _mm256_set1_ps(block.quadratic_scale);
        const auto sign = _mm256_set1_ps(-0.0f);
        for (int d = 0; d < 256; d += 8)
        {
            const auto codes = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(block.codes + d))));
            const auto residual = _mm256_mul_ps(_mm256_mul_ps(scale, codes), _mm256_andnot_ps(sign, codes));
            _mm256_storeu_ps(output + d, _mm256_add_ps(residual, _mm256_loadu_ps(bank.anchor + head * 256 + d)));
        }
        return;
    }
    case TensorType::TQ8: case TensorType::TQ4:
    {
        const float source_norm = *reinterpret_cast<const float *>(bytes);
        if (source_norm < 1e-30f) { std::fill_n(output, 256, 0.0f); return; }
        const float radius = role == detail::BankRole::Key ? source_norm : reinterpret_cast<const float *>(bytes)[1];
        const auto scale = _mm256_set1_ps(radius * 0.0625f);
        for (int d = 0; d < 256; d += 8)
        {
            __m256i indices;
            if (bank.codec == TensorType::TQ8)
                indices = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(bytes + 8 + d)));
            else
            {
                std::uint8_t low[8];
                tq3_unpack_8(bytes + 8 + (d / 8) * 3, low);
                const unsigned high = bytes[8 + 96 + d / 8];
                alignas(32) std::int32_t expanded[8];
                for (int lane = 0; lane < 8; ++lane) expanded[lane] = low[lane] | (((high >> lane) & 1u) << 3);
                indices = _mm256_load_si256(reinterpret_cast<const __m256i *>(expanded));
            }
            _mm256_storeu_ps(output + d, _mm256_mul_ps(_mm256_i32gather_ps(bank.centroids, indices, sizeof(float)), scale));
        }
        return;
    }
    default: __builtin_unreachable(); // Static admission has already rejected non-KV codecs.
    }
}

/** @brief Accumulate one selected value head with a fixed SIMD FMA order. */
void accumulate(float *output, const float *value, float probability, ISALevel isa)
{
#if defined(__AVX512F__)
    if (isa == ISALevel::AVX512)
    {
    const auto scale = _mm512_set1_ps(probability);
    for (int d = 0; d < 256; d += 16)
        _mm512_storeu_ps(output + d, _mm512_fmadd_ps(scale, _mm512_loadu_ps(value + d), _mm512_loadu_ps(output + d)));
        return;
    }
#endif
    const auto scale = _mm256_set1_ps(probability);
    for (int d = 0; d < 256; d += 8)
        _mm256_storeu_ps(output + d, _mm256_fmadd_ps(scale, _mm256_loadu_ps(value + d), _mm256_loadu_ps(output + d)));
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
    const auto *scores = p.scores + std::size_t(row) * p.block_capacity;
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
} // namespace

void cpuQwenSparseIndexer(const QwenSparseIndexer &p)
{
    validateQwenSparseIndexer(p);
    const auto isa = nativeISA();
    validateReplay(*p.replay, p.row_capacity, p.token_capacity);
    if(p.transaction_prior && p.transaction_prior->row_capacity!=p.row_capacity)
        throw std::invalid_argument("QSA indexer transaction prior row geometry changed");
    const auto replay = *p.replay;
    if (p.rotary_source==QwenSparseRotarySource::TablePlanes &&
        static_cast<std::int64_t>(replay.position_origin) + replay.prefix_tokens + replay.live_rows > p.position_capacity)
        throw std::invalid_argument("CPU Qwen sparse indexer: RoPE position capacity exceeded");
    const int first_block = replay.prefix_tokens / 4;
    const int complete_blocks = (replay.prefix_tokens + replay.live_rows) / 4;
    auto preparation = [&]()
    {
#pragma omp for schedule(static)
        for (int block = first_block; block < complete_blocks; ++block)
        {
            float pooled[128];
            for (int d = 0; d < 128; ++d)
            {
                float sum = detail::roundRepresentation(rawKey(p, block * 4)[d], p.representation);
                for (int member = 1; member < 4; ++member)
                    sum += detail::roundRepresentation(rawKey(p, block * 4 + member)[d], p.representation);
                pooled[d] = detail::roundRepresentation(sum * 0.25f, p.representation);
            }
            prepare(pooled, p.key_norm, replay.position_origin + block * 4, p,
                    p.pooled_keys + std::size_t(block) * 128);
        }
#pragma omp for schedule(static)
        for (int row_head = 0; row_head < replay.live_rows * 4; ++row_head)
            prepare(p.query + std::size_t(row_head) * 128, p.query_norm,
                    replay.position_origin + replay.prefix_tokens + row_head / 4, p,
                    p.prepared_query + std::size_t(row_head) * 128);
    };
    OMP_WORKSHARE_REGION(preparation);

    std::atomic<bool> valid{true};
    auto selection = [&]()
    {
#pragma omp for schedule(static)
        for (int row = 0; row < p.row_capacity; ++row)
        {
            if (row < replay.live_rows)
            {
                const int blocks = (replay.prefix_tokens + row + 1) / QwenSparsePool;
                if (blocks > QwenSparseBlocks)
                    for (int block = 0; block < blocks; ++block)
                    {
                        float score = 0.0f;
                        for (int head = 0; head < 4; ++head)
                            score += std::max(0.0f, dot(p.prepared_query + (std::size_t(row) * 4 + head) * 128,
                                                       p.pooled_keys + std::size_t(block) * 128, 128, isa));
                        p.scores[std::size_t(row) * p.block_capacity + block] = score * 0.08838834764831844055f;
                    }
            }
            if (!selectRow(p, row)) valid.store(false, std::memory_order_relaxed);
        }
    };
    OMP_WORKSHARE_REGION(selection);
    if (!valid.load(std::memory_order_relaxed))
        throw std::invalid_argument("CPU QSA bounded selection consumed an invalid finite/nonnegative score frontier");
}

void cpuQwenSparseSelectPools(const QwenSparseIndexer &p)
{
    validateQwenSparseSelection(p);
    validateReplay(*p.replay, p.row_capacity, p.token_capacity);
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
    std::atomic<bool> valid{true};
    auto selection = [&]()
    {
#pragma omp for schedule(static)
        for (int row = 0; row < p.row_capacity; ++row)
            if (!selectRow(p, row)) valid.store(false, std::memory_order_relaxed);
    };
    OMP_WORKSHARE_REGION(selection);
    if (!valid.load(std::memory_order_relaxed))
        throw std::logic_error("CPU QSA bounded selection frontier failed after validation");
}

void cpuQwenSparseAttention(const QwenSparseAttention &p)
{
    validateQwenSparseAttention(p);
    const auto isa = nativeISA();
    validateReplay(*p.replay, p.row_capacity, p.token_capacity);
    if (p.kv.key_policy.encoding != attention::AttentionKeyCacheEncoding::PostRotary)
        throw std::invalid_argument("CPU Qwen sparse attention requires canonical post-RoPE cache publication");
    // Validate all consumed addresses before parallel work to keep failures fatal
    // and diagnostic without exceptions escaping an OpenMP worksharing region.
    for (int row = 0; row < p.replay->live_rows; ++row)
    {
        const int count = p.selected_count[row];
        int origin = 0, capacity = 0;
        if (!detail::resolveMapping(p.kv, *p.replay, row, p.token_capacity, origin, capacity))
            throw std::invalid_argument("CPU Qwen sparse attention: stale or invalid canonical ring mapping");
        if (count <= 0 || count > QwenSparseSelected)
            throw std::invalid_argument("CPU Qwen sparse attention: invalid selected live count");
        int previous = -1;
        for (int i = 0; i < count; ++i)
        {
            const int token = p.selected[std::size_t(row) * QwenSparseSelected + i];
            if (token <= previous || token >= p.replay->prefix_tokens + row + 1)
                throw std::invalid_argument("CPU Qwen sparse attention: noncausal, repeated, or unsorted selected index");
            previous = token;
        }
    }
    auto attention = [&]()
    {
#pragma omp for schedule(static)
        // Capacity owns storage; the canonical replay owns this invocation's
        // writes. Inactive output and traffic records may belong to another
        // request or an earlier retained transaction and must remain untouched.
        for (int row_head = 0; row_head < p.replay->live_rows * p.query_heads; ++row_head)
        {
            float *output = p.output + std::size_t(row_head) * 256;
            const int row = row_head / p.query_heads;
            const int head = row_head % p.query_heads;
            const int kv_head = (p.query_head_start + head) / 12 - p.kv_head_start;
            const int count = p.selected_count[row];
            const float *query = p.query + std::size_t(row_head) * 256;
            float rotated_query[256];
            if (detail::isRotated(p.kv.keys))
            {
                for (int d = 0; d < 256; ++d) rotated_query[d] = dot(p.kv.keys.rotations[kv_head] + d * 256, query, 256, isa);
                query = rotated_query;
            }
            const auto *selected = p.selected + std::size_t(row) * QwenSparseSelected;
            int origin = 0, capacity = 0;
            (void)detail::resolveMapping(p.kv, *p.replay, row, p.token_capacity, origin, capacity);
            float logits[QwenSparseSelected];
            float native[256];
            float maximum = -INFINITY;
            std::uint64_t key_reads = 0, value_reads = 0;
            for (int index = 0; index < count; ++index)
            {
                const int physical = detail::physicalRow(selected[index], origin, capacity);
                readHead(p.kv.keys, physical, kv_head, detail::BankRole::Key, native);
                key_reads += 256;
                logits[index] = dot(query, native, 256, isa) * 0.0625f;
                maximum = std::max(maximum, logits[index]);
            }
            float denominator = 0.0f;
            for (int index = 0; index < count; ++index)
            {
                logits[index] = std::exp(logits[index] - maximum);
                denominator += logits[index];
            }
            std::fill_n(output, 256, 0.0f);
            for (int index = 0; index < count; ++index)
            {
                const float probability = logits[index] / denominator;
                const int physical = detail::physicalRow(selected[index], origin, capacity);
                readHead(p.kv.values, physical, kv_head, detail::BankRole::Value, native);
                accumulate(output, native, probability, isa);
                value_reads += 256;
            }
            if (detail::isRotated(p.kv.values))
            {
                std::copy_n(output, 256, native);
                for (int d = 0; d < 256; ++d)
                {
                    float result = 0.0f;
                    for (int coordinate = 0; coordinate < 256; ++coordinate)
                        result = std::fma(p.kv.values.rotations[kv_head][coordinate * 256 + d], native[coordinate], result);
                    output[d] = result;
                }
            }
            if (p.gate)
                for (int d = 0; d < 256; ++d)
                    output[d] *= 1.0f / (1.0f + std::exp(-p.gate[std::size_t(row_head) * 256 + d]));
            if (p.observations) p.observations[row_head] = {key_reads, value_reads, static_cast<std::uint32_t>(count)};
        }
    };
    OMP_WORKSHARE_REGION(attention);
}

void cpuQwenSparseSnapshotPrior(const QwenSparsePriorSnapshot& p)
{
    validateQwenSparsePriorSnapshot(p);
    validateReplay(*p.replay,p.row_capacity,p.token_capacity);
    const int tail_rows=p.replay->live_rows ? p.replay->prefix_tokens%4 : 0;
    const auto snapshot=[&]() {
        p.output->replay=*p.replay;p.output->row_capacity=p.row_capacity;
        std::copy_n(p.accepted_partial,std::size_t(tail_rows)*128,p.output->partial);
        *p.traffic={16u,static_cast<std::uint32_t>(tail_rows*128*sizeof(float))};
        if(p.work_counters.bound() && !addQwenSparseHostCounters(p.work_counters.control,p.work_counters.words,
            {16u,std::uint64_t(tail_rows)*128*sizeof(float),0,0}))
            throw std::overflow_error("QSA original-tail terminal counter overflow");
    };
    OMP_SINGLE(snapshot);
}

void cpuQwenSparseCommitPartial(const QwenSparsePartialCommit &p)
{
    validateQwenSparsePartialCommit(p);
    validateReplay(*p.replay,p.acceptance.candidates().rowsPerRequest(),p.transaction_prior ? p.token_capacity : std::numeric_limits<int>::max());
    if(p.transaction_prior && p.transaction_prior->row_capacity!=p.acceptance.candidates().rowsPerRequest())
        throw std::invalid_argument("QSA commit transaction prior row geometry changed");
    const int accepted=p.acceptance.resolveNative(0);
    if (accepted < 0 || accepted > p.replay->live_rows)
        throw std::invalid_argument("CPU Qwen sparse partial commit: acceptance exceeds live rows");
    if(p.replay->live_rows==0 || (accepted==0 && !p.transaction_prior))return;
    auto commit = [&]()
    {
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
            throw std::overflow_error("QSA accepted-tail terminal counter overflow");
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
    void prepareReplay(const QwenSparsePrepareReplay &p,void *stream) const override { cpuQwenSparsePrepareReplay(p); }
    /** @copydoc IQwenSparseAttentionKernels::snapshotPrior */
    void snapshotPrior(const QwenSparsePriorSnapshot& p,void*) const override { cpuQwenSparseSnapshotPrior(p); }
    /** @copydoc IQwenSparseAttentionKernels::commitPartial */
    void commitPartial(const QwenSparsePartialCommit &p, void *) const override { cpuQwenSparseCommitPartial(p); }
    /** @copydoc IQwenSparseAttentionKernels::retainCandidates */
    void retainCandidates(const QwenSparseCandidateRetention &p,void *stream) const override
    {
        cpuQwenStateTransfer(qwenSparseCandidateTransfer(p));
    }
};
}
std::unique_ptr<IQwenSparseAttentionKernels> createCPUQwenSparseAttentionKernels()
{
    return std::make_unique<CPUQwenSparseAttentionKernels>();
}
} // namespace llaminar2::qwen4exp
