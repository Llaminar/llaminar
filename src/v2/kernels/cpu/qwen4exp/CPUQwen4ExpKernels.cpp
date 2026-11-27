/**
 * @file CPUQwen4ExpKernels.cpp
 * @brief CPU GR/PLE primitives with fixed arithmetic and immutable history.
 *
 * Worksharing uses the caller's existing team when present. Pointwise kernels
 * distribute contiguous 256-element channel blocks; similarity reductions use
 * 256 canonical lanes irrespective of CPU ISA, worker count or grouped rows.
 * No operation allocates, creates host mirrors, or mutates candidate history.
 * HC two-term input publication retains the original activated FP32 value,
 * backend nonlinear arithmetic and prepared projection/workspace authority.
 */
#include "../../qwen4exp/Qwen4ExpKernels.h"
#include "../../qwen4exp/Qwen4ExpValidation.h"
#include "CPUQwen4ExpSIMD.h"
#include "../../../utils/OpenMPUtils.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace llaminar2::qwen4exp
{
namespace
{
/** @brief Read and validate the CPU execution owner's current live row count. */
int liveRows(RowExtent extent)
{
    const int count = extent.liveRows ? *extent.liveRows : extent.rows;
    detail::require(count >= 0 && count <= extent.rows, "Qwen4Exp live rows exceed admitted capacity");
    return count;
}
/** @brief Resolve CPU acceptance without altering the original candidate extent. */
int acceptedRows(const AcceptedPrefix &prefix)
{
    const int count = prefix.acceptedRows ? *prefix.acceptedRows : prefix.hostAcceptedRows;
    detail::require(count >= 0 && count <= liveRows(prefix.candidates),
                    "Qwen4Exp accepted prefix exceeds live candidate rows");
    return count;
}
/** @brief Stable FP32 sigmoid shared by serial and grouped execution. */
float sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }
/** @brief Dispatch contiguous pointwise blocks through nested-safe CPU worksharing. */
template <class Function>
void pointwise(std::size_t count, Function function)
{
    constexpr std::size_t block = 256;
    auto work = [&]() {
#pragma omp for schedule(static)
        for (std::size_t base = 0; base < count; base += block)
        {
            const auto end = std::min(base + block, count);
            for (auto index = base; index < end; ++index) function(index);
        }
    };
    OMP_WORKSHARE_REGION(work);
}
/** @brief Distribute canonical row-local blocks, preserving serial SIMD tails. */
template <class Function>
void channelBlocks(int rows, int width, Function function)
{
    const auto blocks = (static_cast<std::size_t>(width) + 255) / 256;
    const auto count = static_cast<std::size_t>(rows) * blocks;
    auto work = [&]() {
#pragma omp for schedule(static)
        for (std::size_t task = 0; task < count; ++task)
        {
            const int begin = static_cast<int>((task % blocks) * 256);
            function(task / blocks, begin, std::min(begin + 256, width));
        }
    };
    OMP_WORKSHARE_REGION(work);
}
/** @brief Stateless CPU bundle; graph-owned persistent buffers carry all state. */
class CPUQwen4ExpKernels final : public IQwen4ExpKernels
{
public:
    /** @copydoc IQwen4ExpKernels::branchRMSNorm */
    void branchRMSNorm(const BranchRMSNormArgs &a, void *) const override
    {
        detail::validate(a);
        const int rows = liveRows(a.extent);
        const int vectorWidth = cpu_detail::vectorWidth();
        auto work = [&]() {
#pragma omp for schedule(static)
            for (std::size_t task = 0; task < static_cast<std::size_t>(rows) * 4; ++task)
            {
                const auto base = task * a.hidden;
                cpu_detail::normalizeBranch(a.hidden, a.epsilon, a.input + base,
                                            a.gain + (task % 4) * a.hidden,
                                            a.output + base, vectorWidth);
            }
        };
        OMP_WORKSHARE_REGION(work);
    }
    /** @copydoc IQwen4ExpKernels::residualExpand */
    void residualExpand(const ResidualExpandArgs &a, void *) const override
    {
        detail::validate(a);
        const auto hidden = static_cast<std::size_t>(a.hidden);
        pointwise(static_cast<std::size_t>(liveRows(a.extent)) * 4 * hidden, [&](std::size_t i) {
            a.output[i] = a.input[(i / (4 * hidden)) * hidden + i % hidden];
        });
    }
    /** @copydoc IQwen4ExpKernels::residualDown */
    void residualDown(const ResidualDownArgs &a, void *) const override
    {
        detail::validate(a);
        const int vectorWidth = cpu_detail::vectorWidth();
        channelBlocks(liveRows(a.extent), a.lowRank, [&](std::size_t row, int begin, int end) {
            const auto base = row * a.lowRank;
            cpu_detail::residualDownChannels(a.input + base,a.output + base,begin,end,vectorWidth);
        });
    }
    /** @copydoc IQwen4ExpKernels::residualRead */
    void residualRead(const ResidualReadArgs &a, void *) const override
    {
        detail::validate(a);
        const auto hidden = static_cast<std::size_t>(a.hidden);
        const int rows = liveRows(a.extent);
        const int vectorWidth = cpu_detail::vectorWidth();
        // Use 64-channel blocks on M=1 decode (40 tasks for hidden=2560 across 28 cores
        // instead of 10) with temporal stores so mixed[hidden] stays hot in L2/L3 for
        // the immediately following staggered-L1d activation readers.
        const std::size_t block = (rows == 1) ? 64u : 256u;
        const std::size_t blocks = (hidden + block - 1u) / block;
        const std::size_t count = static_cast<std::size_t>(rows) * blocks;
        auto work = [&]() {
#pragma omp for schedule(static)
            for (std::size_t task = 0; task < count; ++task)
            {
                const std::size_t row = task / blocks;
                const int begin = static_cast<int>((task % blocks) * block);
                const int end = std::min(begin + static_cast<int>(block), a.hidden);
                int channel = begin;
#if defined(__aarch64__) && defined(__ARM_NEON)
                // Four-lane copy of the AVX2 program: polynomial sigmoid, one FMA
                // per branch in branch order, then the separately rounded quarter.
                if (vectorWidth == 4)
                    for (; channel + 4 <= end; channel += 4)
                    {
                        auto sum = vdupq_n_f32(0.0f);
                        for (int branch = 0; branch < 4; ++branch)
                        {
                            const auto offset = (row * 4 + branch) * hidden + channel;
                            sum = vfmaq_f32(sum, neon::fast_sigmoid(vld1q_f32(a.gateLogits + offset)),
                                            vld1q_f32(a.normalized + offset));
                        }
                        vst1q_f32(a.output + row * hidden + channel, vmulq_n_f32(sum, 0.25f));
                    }
#endif
#if defined(__AVX512F__)
                if (vectorWidth == 16)
                {
                    for (; channel + 16 <= end; channel += 16)
                    {
                        auto sum = _mm512_setzero_ps();
                        for (int branch = 0; branch < 4; ++branch)
                        {
                            const auto offset = (row * 4 + branch) * hidden + channel;
                            sum = _mm512_fmadd_ps(cpu_detail::sigmoid(_mm512_loadu_ps(a.gateLogits + offset)),
                                                 _mm512_loadu_ps(a.normalized + offset), sum);
                        }
                        const __m512 out_v = _mm512_mul_ps(sum, _mm512_set1_ps(0.25f));
                        _mm512_storeu_ps(a.output + row * hidden + channel, out_v);
                    }
                }
#endif
#if defined(__AVX2__)
                if (vectorWidth >= 8)
                {
                    for (; channel + 8 <= end; channel += 8)
                    {
                        auto sum = _mm256_setzero_ps();
                        for (int branch = 0; branch < 4; ++branch)
                        {
                            const auto offset = (row * 4 + branch) * hidden + channel;
                            sum = _mm256_fmadd_ps(avx2::fast_sigmoid(_mm256_loadu_ps(a.gateLogits + offset)),
                                                 _mm256_loadu_ps(a.normalized + offset), sum);
                        }
                        const __m256 out_v = _mm256_mul_ps(sum, _mm256_set1_ps(0.25f));
                        _mm256_storeu_ps(a.output + row * hidden + channel, out_v);
                    }
                }
#endif
                for (; channel < end; ++channel)
                {
                    const auto base = row * 4 * hidden + channel;
                    float sum = 0.0f;
                    for (int branch = 0; branch < 4; ++branch)
                    {
                        const auto offset = base + branch * hidden;
                        sum = std::fma(sigmoid(a.gateLogits[offset]), a.normalized[offset], sum);
                    }
                    a.output[row * hidden + channel] = sum * 0.25f;
                }
            }
        };
        OMP_WORKSHARE_REGION(work);
    }
    /** @copydoc IQwen4ExpKernels::residualWrite */
    void residualWrite(const ResidualWriteArgs &a, void *) const override
    {
        detail::validate(a);
        const auto hidden = static_cast<std::size_t>(a.hidden);
        const auto blocks = (hidden + 255) / 256;
        const auto tasks = static_cast<std::size_t>(liveRows(a.extent)) * 4 * blocks;
        const int vectorWidth = cpu_detail::vectorWidth();
        auto work = [&]() {
#pragma omp for schedule(static)
            for (std::size_t task = 0; task < tasks; ++task)
            {
                const auto rowBranch = task / blocks;
                const auto row = rowBranch / 4;
                const auto begin = (task % blocks) * 256;
                const auto end = std::min(begin + 256, hidden);
                const float gate = 2.0f * sigmoid(a.gateLogits[rowBranch] * 0.25f);
                const float *res = a.residual + rowBranch * hidden;
                const float *blk = a.block + row * hidden;
                float *out = a.output + rowBranch * hidden;
                std::size_t channel = begin;
#if defined(__AVX512F__)
                if (vectorWidth == 16)
                {
                    const __m512 vgate = _mm512_set1_ps(gate);
                    const __m512 vzero = _mm512_setzero_ps();
                    for (; channel + 16 <= end; channel += 16)
                    {
                        const __m512 upd = _mm512_fmadd_ps(vgate, _mm512_loadu_ps(blk + channel), vzero);
                        _mm512_storeu_ps(out + channel, _mm512_add_ps(_mm512_loadu_ps(res + channel), upd));
                    }
                }
#endif
#if defined(__AVX2__)
                if (vectorWidth >= 8)
                {
                    const __m256 vgate = _mm256_set1_ps(gate);
                    const __m256 vzero = _mm256_setzero_ps();
                    for (; channel + 8 <= end; channel += 8)
                    {
                        const __m256 upd = _mm256_fmadd_ps(vgate, _mm256_loadu_ps(blk + channel), vzero);
                        _mm256_storeu_ps(out + channel, _mm256_add_ps(_mm256_loadu_ps(res + channel), upd));
                    }
                }
#endif
                for (; channel < end; ++channel)
                {
                    // Explicit zero fma materializes the product before addition.
                    const float update = std::fma(gate, blk[channel], 0.0f);
                    out[channel] = res[channel] + update;
                }
            }
        };
        OMP_WORKSHARE_REGION(work);
    }
    /** @copydoc IQwen4ExpKernels::residualWriteBranchRMSNorm */
    void residualWriteBranchRMSNorm(const ResidualWriteBranchRMSNormArgs &a, void *) const override
    {
        detail::validate(a);
        const int rows = liveRows(a.extent);
        const int vectorWidth = cpu_detail::vectorWidth();
        auto work = [&]() {
#pragma omp for schedule(static)
            for (std::size_t task = 0; task < static_cast<std::size_t>(rows) * 4; ++task)
            {
                const auto base = task * a.hidden;
                const auto blockBase = (task / 4) * a.hidden;
                const float gate = 2.0f * sigmoid(a.gateLogits[task] * 0.25f);
                cpu_detail::residualWriteAndNormalizeBranch(
                    a.hidden, a.epsilon, gate,
                    a.residual + base, a.block + blockBase,
                    a.gain + (task % 4) * a.hidden,
                    a.output + base, a.normalized + base, vectorWidth);
            }
        };
        OMP_WORKSHARE_REGION(work);
    }
    /** @copydoc IQwen4ExpKernels::ngramHash */
    void ngramHash(const NgramHashArgs &a, void *) const override
    {
        detail::validate(a);
        pointwise(static_cast<std::size_t>(liveRows(a.extent)) * kNgramHeads, [&](std::size_t i) {
            const auto row = i / kNgramHeads;
            const auto head = i % kNgramHeads;
            const auto previous = row > 0 ? a.tokens[row - 1] : a.priorTokens[1];
            auto secondPrevious = row > 1 ? a.tokens[row - 2] : a.priorTokens[row];
            if (previous == a.policy.eosToken) secondPrevious = a.policy.eosToken;
            const auto current = static_cast<std::uint64_t>(a.tokens[row]);
            auto hash = (current * a.policy.multipliers[0]) ^
                        (static_cast<std::uint64_t>(previous) * a.policy.multipliers[1]);
            if (head >= 8) hash ^= static_cast<std::uint64_t>(secondPrevious) * a.policy.multipliers[2];
            a.output[i] = static_cast<std::int64_t>(hash % a.policy.moduli[head] + a.policy.offsets[head]);
        });
    }
    /** @copydoc IQwen4ExpKernels::ngramHashNextWindow */
    void ngramHashNextWindow(const NgramNextWindowHashArgs& a,void*)const override
    {
        detail::validate(a);const auto window=a.nextWindow.resolve();
        detail::require(window.valid,"QFN next-window hash has malformed original live inputs");
        if(!window.rows)return;
        ngramHash({{window.rows},a.tokens+window.first,a.priorTokens,a.output,a.policy},nullptr);
    }
    /** @copydoc IQwen4ExpKernels::pleGate */
    void pleGate(const PLEGateArgs &a, void *) const override
    {
        detail::validate(a);
        const int rows = liveRows(a.extent);
        auto work = [&]() {
#pragma omp for schedule(static)
            for (std::size_t task = 0; task < static_cast<std::size_t>(rows) * 4; ++task)
            {
                const auto base = static_cast<std::size_t>(task) * a.hidden;
                alignas(64) float partial[256]{};
                // Segments update the same 256 arithmetic lanes in ascending
                // channel order; SIMD merely evaluates independent lanes together.
                for (int segment = 0; segment < a.hidden; segment += 256)
                {
                    const int count = std::min(256, a.hidden - segment);
                    int lane = 0;
#if defined(__AVX512F__)
                    if (cpu_detail::vectorWidth() == 16)
                        for (; lane + 16 <= count; lane += 16)
                            _mm512_store_ps(partial + lane, _mm512_fmadd_ps(
                                _mm512_loadu_ps(a.normalizedKey + base + segment + lane),
                                _mm512_loadu_ps(a.normalizedQuery + base + segment + lane),
                                _mm512_load_ps(partial + lane)));
#endif
#if defined(__AVX2__)
                    if (cpu_detail::vectorWidth() >= 8)
                        for (; lane + 8 <= count; lane += 8)
                            _mm256_store_ps(partial + lane, _mm256_fmadd_ps(
                                _mm256_loadu_ps(a.normalizedKey + base + segment + lane),
                                _mm256_loadu_ps(a.normalizedQuery + base + segment + lane),
                                _mm256_load_ps(partial + lane)));
#endif
                    for (; lane < count; ++lane)
                        partial[lane] = std::fma(a.normalizedKey[base + segment + lane],
                                                 a.normalizedQuery[base + segment + lane], partial[lane]);
                }
                for (int stride = 128; stride > 0; stride /= 2)
                    for (int lane = 0; lane < stride; ++lane) partial[lane] += partial[lane + stride];
                const float similarity = partial[0] / std::sqrt(static_cast<float>(a.hidden));
                const float sign = similarity > 0 ? 1.0f : (similarity < 0 ? -1.0f : 0.0f);
                const float gate = sigmoid(sign * std::sqrt(std::max(std::abs(similarity), 1e-6f)));
                const auto valueBase = static_cast<std::size_t>(task / 4) * a.hidden;
                for (int channel = 0; channel < a.hidden; ++channel)
                    a.gated[base + channel] = gate * a.value[valueBase + channel];
            }
        };
        OMP_WORKSHARE_REGION(work);
    }
    /** @copydoc IQwen4ExpKernels::pleConv */
    void pleConv(const PLEConvArgs &a, void *) const override
    {
        detail::validate(a);
        const auto channels = static_cast<std::size_t>(a.channels);
        const int vectorWidth = cpu_detail::vectorWidth();
        channelBlocks(liveRows(a.extent), a.channels, [&](std::size_t row, int begin, int end) {
            int channel = begin;
#if defined(__AVX512F__)
            if (vectorWidth == 16)
                for (; channel + 16 <= end; channel += 16)
                {
                    auto sum = _mm512_setzero_ps();
                    const auto indices = _mm512_setr_epi32(0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60);
                    for (int tap = 0; tap < 4; ++tap)
                    {
                        const auto source = static_cast<std::ptrdiff_t>(row) - 9 + 3 * tap;
                        const auto *values = source < 0 ? a.priorHistory + (source + 9) * channels
                                                       : a.normalized + source * channels;
                        sum = _mm512_fmadd_ps(_mm512_loadu_ps(values + channel),
                            _mm512_i32gather_ps(indices, a.weights + channel * 4 + tap, sizeof(float)), sum);
                    }
                    const auto index = row * channels + channel;
                    const auto activation = _mm512_mul_ps(sum, cpu_detail::sigmoid(sum));
                    const auto ple = _mm512_add_ps(_mm512_loadu_ps(a.gated + index), activation);
                    _mm512_storeu_ps(a.output + index, _mm512_add_ps(_mm512_loadu_ps(a.residual + index), ple));
                }
#endif
#if defined(__AVX2__)
            if (vectorWidth >= 8)
                for (; channel + 8 <= end; channel += 8)
                {
                    auto sum = _mm256_setzero_ps();
                    const auto indices = _mm256_setr_epi32(0,4,8,12,16,20,24,28);
                    for (int tap = 0; tap < 4; ++tap)
                    {
                        const auto source = static_cast<std::ptrdiff_t>(row) - 9 + 3 * tap;
                        const auto *values = source < 0 ? a.priorHistory + (source + 9) * channels
                                                       : a.normalized + source * channels;
                        sum = _mm256_fmadd_ps(_mm256_loadu_ps(values + channel),
                            _mm256_i32gather_ps(a.weights + channel * 4 + tap, indices, sizeof(float)), sum);
                    }
                    const auto index = row * channels + channel;
                    const auto ple = _mm256_add_ps(_mm256_loadu_ps(a.gated + index), avx2::fast_silu(sum));
                    _mm256_storeu_ps(a.output + index, _mm256_add_ps(_mm256_loadu_ps(a.residual + index), ple));
                }
#endif
            for (; channel < end; ++channel)
            {
                float sum = 0.0f;
                for (int tap = 0; tap < 4; ++tap)
                {
                    const auto source = static_cast<std::ptrdiff_t>(row) - 9 + 3 * tap;
                    const float value = source < 0 ? a.priorHistory[(source + 9) * channels + channel]
                                                   : a.normalized[source * channels + channel];
                    sum = std::fma(value, a.weights[channel * 4 + tap], sum);
                }
                const auto index = row * channels + channel;
                const float ple = a.gated[index] + sum * sigmoid(sum);
                a.output[index] = a.residual[index] + ple;
            }
        });
    }
    /** @copydoc IQwen4ExpKernels::commitConvHistory */
    void commitConvHistory(const ConvHistoryCommitArgs &a, void *) const override
    {
        detail::validate(a);
        const auto accepted = static_cast<std::ptrdiff_t>(acceptedRows(a.prefix));
        const auto channels = static_cast<std::size_t>(a.channels);
        pointwise(9 * channels, [&](std::size_t i) {
            const auto source = accepted - 9 + static_cast<std::ptrdiff_t>(i / channels);
            const auto channel = i % channels;
            a.outputHistory[i] = source < 0 ? a.priorHistory[(source + 9) * channels + channel]
                                           : a.candidates[source * channels + channel];
        });
    }
    /** @copydoc IQwen4ExpKernels::commitTokenHistory */
    void commitTokenHistory(const TokenHistoryCommitArgs &a, void *) const override
    {
        detail::validate(a);
        const int accepted = acceptedRows(a.prefix);
        // Two scalar stores are cheaper than a worksharing team. Each caller in
        // an existing team must still execute the publication exactly once.
        auto work = [&]() {
            for (int index = 0; index < 2; ++index)
            {
                const int source = accepted - 2 + index;
                a.outputTokens[index] = source < 0 ? a.priorTokens[source + 2] : a.candidates[source];
            }
        };
        OMP_SINGLE(work);
    }
};
} // namespace
std::unique_ptr<IQwen4ExpKernels> createCPUQwen4ExpKernels()
{
    return std::make_unique<CPUQwen4ExpKernels>();
}
} // namespace llaminar2::qwen4exp
