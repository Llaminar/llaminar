/**
 * @file GPUDeepSeekV4Kernels.cuh
 * @brief Shared CUDA (and future HIP) DeepSeek-V4 primitive kernels.
 *
 * Capture-safe: every launch uses the caller's exact non-default stream, no
 * allocation, synchronization, transfer or host read of device state. Live row
 * counts and positions are read on device only. Per-row arithmetic uses fixed
 * 256-thread trees so results do not depend on row count.
 *
 * Kernel shapes:
 *  - hcRead: one CTA per row (sum of squares over hc*H, Sinkhorn in lane 0).
 *  - headNormRope / compressorPool / rawStore: one CTA per (row, head) / row,
 *    staging the head vector in shared memory for RMS, RoPE, the QAT
 *    rounding (Hadamard butterflies, per-block scale exponents) and packing.
 *  - QAT, Hadamard, record packing/decoding and indexer scoring use the
 *    explicit non-FTZ qat:: operations, so they keep IEEE subnormals and match
 *    the CPU bundle bit-for-bit despite the TU's fast-math policy.
 *  - indexer: bounded passes of scoreRows rows; each pass scores (row, block
 *    tile) into the admitted scratch, then runs a per-row 4-pass radix
 *    selection of the top-k float keys (ties by lower id).
 *  - sparseAttention: one CTA per (row, head); each warp runs an online
 *    softmax over a strided subset of keys, warps merge in shared memory.
 *  - groupedProjection: one warp per output, decoding Q8_0/BF16/FP32 rows.
 */
#pragma once
#include "DeepSeekV4Kernels.h"
#include "DeepSeekV4Math.h"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <stdexcept>
#include <string>

#if defined(DSV4_CUDA)
#include <cuda_runtime.h>
#define DSV4_NATIVE_STREAM cudaStream_t
#define DSV4_NATIVE_ERROR cudaError_t
#define DSV4_GET_ERROR cudaGetLastError
#define DSV4_ERROR_STRING cudaGetErrorString
#define DSV4_SUCCESS cudaSuccess
#define DSV4_DEFAULT_STREAM cudaStreamLegacy
#define DSV4_THREAD_STREAM cudaStreamPerThread
#define DSV4_FACTORY createCUDADeepSeekV4Kernels
#define DSV4_IMPLEMENTATION CUDADeepSeekV4Kernels
#define DSV4_BACKEND_NAMESPACE cuda_detail
#define DSV4_SHFL_DOWN(v, s) __shfl_down_sync(0xffffffffu, v, s)
#define DSV4_SHFL_XOR(v, s) __shfl_xor_sync(0xffffffffu, v, s)
#else
#error "Compile DeepSeek-V4 GPU primitives through a native backend translation unit"
#endif

namespace llaminar2::deepseek4
{
namespace DSV4_BACKEND_NAMESPACE
{
inline constexpr unsigned kThreads = 256;
inline constexpr unsigned kWarps = kThreads / 32;

__device__ inline void invalidDeviceState() { __trap(); }

__device__ inline int liveRows(RowExtent extent)
{
    const int count = extent.liveRows ? *extent.liveRows : extent.rows;
    if (count < 0 || count > extent.rows) invalidDeviceState();
    return count;
}

/** @brief Fixed-tree CTA sum; every thread receives the total. */
__device__ inline float blockSum(float value, float *scratch)
{
    const unsigned lane = threadIdx.x;
    scratch[lane] = value;
    __syncthreads();
    for (unsigned stride = kThreads / 2; stride > 0; stride /= 2)
    {
        if (lane < stride) scratch[lane] = __fadd_rn(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    const float total = scratch[0];
    __syncthreads();
    return total;
}

__device__ inline float warpSum(float value)
{
    for (int stride = 16; stride > 0; stride /= 2) value += DSV4_SHFL_XOR(value, stride);
    return value;
}

// --------------------------------------------------------------------------- hyper-connections
__global__ __launch_bounds__(kThreads) void hcReadKernel(HCReadArgs a)
{
    const int r = static_cast<int>(blockIdx.x);
    if (r >= liveRows(a.extent)) return;
    __shared__ float scratch[kThreads];
    __shared__ float pre[4];
    const int hc = a.hc;
    const int width = hc * a.hidden;
    const int mixWidth = a.head ? hc : (2 + hc) * hc;
    const float *x = a.trunk + static_cast<std::size_t>(r) * width;
    float partial = 0.0f;
    for (int i = threadIdx.x; i < width; i += kThreads) partial = fmaf(x[i], x[i], partial);
    const float ss = blockSum(partial, scratch);
    if (threadIdx.x == 0)
    {
        const float rs = 1.0f / sqrtf(ss / static_cast<float>(width) + a.normEps);
        const float *m = a.mixes + static_cast<std::size_t>(r) * mixWidth;
        for (int j = 0; j < hc; ++j) pre[j] = math::sigmoid(m[j] * rs * a.scale[0] + a.base[j]) + a.hcEps;
        if (!a.head)
        {
            float *post = a.post + static_cast<std::size_t>(r) * hc;
            for (int j = 0; j < hc; ++j) post[j] = 2.0f * math::sigmoid(m[hc + j] * rs * a.scale[1] + a.base[hc + j]);
            float comb[16];
            for (int i = 0; i < hc * hc; ++i) comb[i] = m[2 * hc + i] * rs * a.scale[2] + a.base[2 * hc + i];
            math::sinkhorn(comb, hc, a.sinkhornIters, a.hcEps);
            float *dst = a.comb + static_cast<std::size_t>(r) * hc * hc;
            for (int i = 0; i < hc * hc; ++i) dst[i] = comb[i];
        }
    }
    __syncthreads();
    float *out = a.output + static_cast<std::size_t>(r) * a.hidden;
    float sq = 0.0f;
    for (int d = threadIdx.x; d < a.hidden; d += kThreads)
    {
        float y = 0.0f;
        for (int j = 0; j < hc; ++j) y = fmaf(pre[j], x[static_cast<std::size_t>(j) * a.hidden + d], y);
        out[d] = y;
        if (a.mixed) a.mixed[static_cast<std::size_t>(r) * a.hidden + d] = y;
        sq = fmaf(y, y, sq);
    }
    if (!a.gain) return;
    const float total = blockSum(sq, scratch);
    const float inv = 1.0f / sqrtf(total / static_cast<float>(a.hidden) + a.outputEps);
    for (int d = threadIdx.x; d < a.hidden; d += kThreads) out[d] = out[d] * inv * a.gain[d];
}

__global__ __launch_bounds__(kThreads) void hcWriteKernel(HCWriteArgs a)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    const std::size_t r = index / a.hidden;
    const int d = static_cast<int>(index % a.hidden);
    if (r >= static_cast<std::size_t>(liveRows(a.extent))) return;
    const int hc = a.hc;
    const std::size_t base = r * hc * a.hidden;
    float in[4];
    for (int j = 0; j < hc; ++j) in[j] = a.trunk[base + static_cast<std::size_t>(j) * a.hidden + d];
    const float block = a.block[r * a.hidden + d];
    const float *post = a.post + r * hc;
    const float *comb = a.comb + r * hc * hc;
    for (int k = 0; k < hc; ++k)
    {
        float y = post[k] * block;
        for (int j = 0; j < hc; ++j) y = fmaf(comb[j * hc + k], in[j], y);
        a.output[base + static_cast<std::size_t>(k) * a.hidden + d] = y;
    }
}

__global__ __launch_bounds__(kThreads) void hcExpandKernel(HCExpandArgs a)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    const std::size_t r = index / a.hidden;
    const int d = static_cast<int>(index % a.hidden);
    if (r >= static_cast<std::size_t>(liveRows(a.extent))) return;
    const float v = a.input[index];
    for (int j = 0; j < a.hc; ++j) a.output[(r * a.hc + j) * a.hidden + d] = v;
}

// --------------------------------------------------------------------------- QAT / packing helpers
/** @brief First chunk position; traps unless live row r carries position start + r. */
__device__ inline int contiguousStart(const std::int32_t *positions, int r)
{
    const int start = positions[0];
    if (start < 0 || positions[r] != start + r) invalidDeviceState();
    return start;
}

/**
 * @brief In-place Hadamard (Sylvester order, scaled by n^-0.5) of a shared vector.
 * @details Every butterfly stage updates disjoint pairs, so the parallel
 * stage order reproduces qat::hadamard bit-for-bit. All threads must call.
 */
__device__ inline void hadamardShared(float *vec, int n)
{
    for (int h = 1; h < n; h *= 2)
    {
        for (int t = threadIdx.x; t < n / 2; t += kThreads)
        {
            const int i = (t / h) * 2 * h + t % h;
            const float x = vec[i], y = vec[i + h];
            vec[i] = qat::addRn(x, y);
            vec[i + h] = qat::subRn(x, y);
        }
        __syncthreads();
    }
    const float scale = qat::hadamardScale(n);
    for (int c = threadIdx.x; c < n; c += kThreads) vec[c] = qat::mulRn(vec[c], scale);
    __syncthreads();
}

/** @brief Per-block scale exponents of a shared vector; all threads must call. */
__device__ inline void blockExponents(const float *vec, int blocks, int blockSize, bool fp4, int *exponents)
{
    for (int b = threadIdx.x; b < blocks; b += kThreads)
    {
        const float amax = qat::absMax(vec + b * blockSize, blockSize);
        exponents[b] = fp4 ? qat::fp4ScaleExponent(amax) : qat::fp8ScaleExponent(amax);
    }
    __syncthreads();
}

/** @brief Apply an ActivationQuant to a shared head vector (qat::apply); all threads must call. */
__device__ inline void quantizeShared(ActivationQuant quant, float *vec, int dim, int ropeDim, int *exponents)
{
    if (quant == ActivationQuant::FP8NopeBF16Rope)
    {
        const int nope = dim - ropeDim;
        blockExponents(vec, nope / kFP8BlockSize, kFP8BlockSize, false, exponents);
        for (int c = threadIdx.x; c < dim; c += kThreads)
        {
            if (c < nope)
            {
                const int k = exponents[c / kFP8BlockSize];
                vec[c] = qat::mulRn(qat::decodeE4M3(qat::encodeE4M3(qat::mulRn(vec[c], qat::exp2i(-k)))), qat::exp2i(k));
            }
            else
                vec[c] = qat::decodeBF16(qat::encodeBF16(vec[c]));
        }
        __syncthreads();
    }
    else if (quant == ActivationQuant::HadamardFP4)
    {
        hadamardShared(vec, dim);
        blockExponents(vec, dim / kFP4BlockSize, kFP4BlockSize, true, exponents);
        for (int c = threadIdx.x; c < dim; c += kThreads)
        {
            const int k = exponents[c / kFP4BlockSize];
            vec[c] = qat::mulRn(qat::decodeE2M1(qat::encodeE2M1(qat::mulRn(vec[c], qat::exp2i(-k)))), qat::exp2i(k));
        }
        __syncthreads();
    }
}

/** @brief qat::packKVEntry of a shared vector; all threads must call. */
__device__ inline void packKVShared(const PackedKVEntryFormat &format, const float *vec, int *exponents,
                                    std::uint8_t *entry)
{
    const int nope = format.nopeDim();
    blockExponents(vec, format.scaleCount(), kFP8BlockSize, false, exponents);
    for (int c = threadIdx.x; c < nope; c += kThreads)
        entry[c] = qat::encodeE4M3(qat::mulRn(vec[c], qat::exp2i(-exponents[c / kFP8BlockSize])));
    for (int c = threadIdx.x; c < format.ropeDim; c += kThreads)
    {
        const std::uint16_t payload = qat::encodeBF16(vec[nope + c]);
        entry[format.ropeOffset() + 2 * c] = static_cast<std::uint8_t>(payload & 0xFFu);
        entry[format.ropeOffset() + 2 * c + 1] = static_cast<std::uint8_t>(payload >> 8);
    }
    for (int b = threadIdx.x; b < format.entryBytes() - format.scaleOffset(); b += kThreads)
        entry[format.scaleOffset() + b] =
            b < format.scaleCount() ? static_cast<std::uint8_t>(exponents[b] + 127) : std::uint8_t{0};
    __syncthreads();
}

/** @brief qat::packIndexerKey of a shared (already rotated) vector; all threads must call. */
__device__ inline void packIndexerShared(const PackedIndexerKeyFormat &format, const float *vec, int *exponents,
                                         std::uint8_t *entry)
{
    blockExponents(vec, format.scaleCount(), kFP4BlockSize, true, exponents);
    for (int t = threadIdx.x; t < format.headDim / 2; t += kThreads)
    {
        const float inverse = qat::exp2i(-exponents[(2 * t) / kFP4BlockSize]);
        entry[t] = static_cast<std::uint8_t>(qat::encodeE2M1(qat::mulRn(vec[2 * t], inverse)) |
                                             (qat::encodeE2M1(qat::mulRn(vec[2 * t + 1], inverse)) << 4));
    }
    for (int b = threadIdx.x; b < format.entryBytes() - format.scaleOffset(); b += kThreads)
        entry[format.scaleOffset() + b] =
            b < format.scaleCount() ? static_cast<std::uint8_t>(exponents[b] + 127) : std::uint8_t{0};
    __syncthreads();
}

// --------------------------------------------------------------------------- norm + rope
__global__ __launch_bounds__(kThreads) void headNormRopeKernel(HeadNormRopeArgs a)
{
    const int task = static_cast<int>(blockIdx.x);
    const int r = task / a.heads;
    if (r >= liveRows(a.extent)) return;
    __shared__ float scratch[kThreads];
    __shared__ float vec[kMaxHeadDim];
    __shared__ int exponents[kMaxHeadDim / kFP4BlockSize];
    float *v = a.data + static_cast<std::size_t>(task) * a.headDim;
    for (int d = threadIdx.x; d < a.headDim; d += kThreads) vec[d] = v[d];
    __syncthreads();
    if (a.normalize)
    {
        float partial = 0.0f;
        for (int d = threadIdx.x; d < a.headDim; d += kThreads) partial = fmaf(vec[d], vec[d], partial);
        const float ss = blockSum(partial, scratch);
        const float inv = 1.0f / sqrtf(ss / static_cast<float>(a.headDim) + a.eps);
        for (int d = threadIdx.x; d < a.headDim; d += kThreads) vec[d] = vec[d] * inv * (a.gain ? a.gain[d] : 1.0f);
        __syncthreads();
    }
    if (a.rope.ropeDim > 0)
    {
        const int position = a.positions[r];
        float *tail = vec + (a.headDim - a.rope.ropeDim);
        for (int pair = threadIdx.x; pair < a.rope.ropeDim / 2; pair += kThreads)
        {
            const float angle = static_cast<float>(position) * math::ropeFrequency(a.rope, pair);
            const float c = cosf(angle);
            const float s = a.inverse ? -sinf(angle) : sinf(angle);
            const float x0 = tail[2 * pair], x1 = tail[2 * pair + 1];
            tail[2 * pair] = x0 * c - x1 * s;
            tail[2 * pair + 1] = x0 * s + x1 * c;
        }
        __syncthreads();
    }
    quantizeShared(a.quant, vec, a.headDim, a.rope.ropeDim, exponents);
    for (int d = threadIdx.x; d < a.headDim; d += kThreads) v[d] = vec[d];
}

// --------------------------------------------------------------------------- compressor
__global__ __launch_bounds__(kThreads) void compressorStoreKernel(CompressorStoreArgs a)
{
    const int width = a.geometry.projectionWidth();
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    const std::size_t r = index / width;
    const int c = static_cast<int>(index % width);
    const int live = liveRows(a.extent);
    if (r >= static_cast<std::size_t>(live)) return;
    const int start = contiguousStart(a.positions, static_cast<int>(r));
    const int pos = start + static_cast<int>(r);
    const int states = a.geometry.stateRows();
    // Only the final stateRows() positions survive; each slot has one writer.
    if (pos <= start + live - 1 - states) return;
    const std::size_t slot = static_cast<std::size_t>(pos % states) * width + c;
    a.stateKV[slot] = a.kv[index];
    a.stateScore[slot] = a.score[index] + a.ape[static_cast<std::size_t>(pos % a.geometry.ratio) * width + c];
}

__global__ __launch_bounds__(kThreads) void compressorPoolKernel(CompressorPoolArgs a)
{
    const int r = static_cast<int>(blockIdx.x);
    if (r >= liveRows(a.extent)) return;
    const auto &g = a.geometry;
    const int start = contiguousStart(a.positions, r);
    const int pos = start + r;
    if ((pos + 1) % g.ratio != 0) return;
    const int block = pos / g.ratio;
    if (block >= g.poolRows) invalidDeviceState();
    __shared__ float scratch[kThreads];
    __shared__ float vec[kMaxHeadDim];
    __shared__ int exponents[kMaxHeadDim / kFP4BlockSize];
    const int width = g.projectionWidth();
    const int states = g.stateRows();
    const int d = g.headDim;
    const int first = block * g.ratio;
    const int entries = g.overlap() ? 2 * g.ratio : g.ratio;
    // Projection of position p: the chunk scratch, else the carried state.
    const auto kvAt = [&](int p, int ch) {
        return p >= start ? a.kv[static_cast<std::size_t>(p - start) * width + ch]
                          : a.stateKV[static_cast<std::size_t>(p % states) * width + ch];
    };
    const auto scoreAt = [&](int p, int ch) {
        return p >= start ? a.score[static_cast<std::size_t>(p - start) * width + ch] +
                                a.ape[static_cast<std::size_t>(p % g.ratio) * width + ch]
                          : a.stateScore[static_cast<std::size_t>(p % states) * width + ch];
    };
    float sq = 0.0f;
    for (int c = threadIdx.x; c < d; c += kThreads)
    {
        float m = -FLT_MAX;
        for (int e = 0; e < entries; ++e)
        {
            const bool prev = g.overlap() && e < g.ratio;
            const int p = prev ? first - g.ratio + e : first + (g.overlap() ? e - g.ratio : e);
            if (p < 0) continue;
            m = fmaxf(m, scoreAt(p, prev ? c : (g.overlap() ? d + c : c)));
        }
        float sum = 0.0f, acc = 0.0f;
        for (int e = 0; e < entries; ++e)
        {
            const bool prev = g.overlap() && e < g.ratio;
            const int p = prev ? first - g.ratio + e : first + (g.overlap() ? e - g.ratio : e);
            if (p < 0) continue;
            const int ch = prev ? c : (g.overlap() ? d + c : c);
            const float w = expf(scoreAt(p, ch) - m);
            sum += w;
            acc = fmaf(w, kvAt(p, ch), acc);
        }
        const float value = acc / sum;
        vec[c] = value;
        sq = fmaf(value, value, sq);
    }
    const float ss = blockSum(sq, scratch);
    const float inv = 1.0f / sqrtf(ss / static_cast<float>(d) + a.eps);
    for (int c = threadIdx.x; c < d; c += kThreads) vec[c] = vec[c] * inv * a.gain[c];
    __syncthreads();
    float *tail = vec + (d - a.rope.ropeDim);
    for (int pair = threadIdx.x; pair < a.rope.ropeDim / 2; pair += kThreads)
    {
        const float angle = static_cast<float>(block * g.ratio) * math::ropeFrequency(a.rope, pair);
        const float c = cosf(angle), s = sinf(angle);
        const float x0 = tail[2 * pair], x1 = tail[2 * pair + 1];
        tail[2 * pair] = x0 * c - x1 * s;
        tail[2 * pair + 1] = x0 * s + x1 * c;
    }
    __syncthreads();
    std::uint8_t *entry = a.pool + static_cast<std::size_t>(block) * g.entryBytes();
    if (g.stream == CompressedStream::Attention)
        packKVShared({d, g.ropeDim}, vec, exponents, entry);
    else
    {
        hadamardShared(vec, d);
        packIndexerShared({d}, vec, exponents, entry);
    }
}

/** @brief One CTA per live row: pack the last @c window rows into the ring. */
__global__ __launch_bounds__(kThreads) void rawStoreKernel(RawStoreArgs a)
{
    const int r = static_cast<int>(blockIdx.x);
    const int live = liveRows(a.extent);
    if (r >= live) return;
    const int start = contiguousStart(a.positions, r);
    const int pos = start + r;
    if (pos <= start + live - 1 - a.window) return;
    __shared__ float vec[kMaxHeadDim];
    __shared__ int exponents[kMaxHeadDim / kFP4BlockSize];
    const float *src = a.kv + static_cast<std::size_t>(r) * a.format.headDim;
    for (int c = threadIdx.x; c < a.format.headDim; c += kThreads) vec[c] = src[c];
    __syncthreads();
    packKVShared(a.format, vec, exponents,
                 a.ring + static_cast<std::size_t>(pos % a.window) * a.format.entryBytes());
}

// --------------------------------------------------------------------------- indexer
/**
 * @brief Score kernel for one bounded pass: grid (block tile, pass row).
 * @details Each thread scores one pool block against all heads with the
 * shared qat::indexerDot order and a fused-multiply-add head accumulation,
 * the same arithmetic as the CPU bundle (bit-identical scores).
 */
__global__ __launch_bounds__(kThreads) void indexerScoreKernel(IndexerTopKArgs a, int rowBase)
{
    const int r = rowBase + static_cast<int>(blockIdx.y);
    if (r >= liveRows(a.extent)) return;
    const int dim = a.format.headDim;
    const int visible = min((a.positions[r] + 1) / a.ratio, a.poolRows);
    const int b = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
    extern __shared__ float query[]; // [heads * headDim]
    const float *q = a.query + static_cast<std::size_t>(r) * a.heads * dim;
    for (int i = threadIdx.x; i < a.heads * dim; i += kThreads) query[i] = q[i];
    __syncthreads();
    if (b >= visible) return;
    const std::uint8_t *entry = a.pool + static_cast<std::size_t>(b) * a.format.entryBytes();
    const float *w = a.weights + static_cast<std::size_t>(r) * a.heads;
    float score = 0.0f;
    for (int h = 0; h < a.heads; ++h)
    {
        score = qat::indexerAccumulate(score, qat::indexerDot(a.format, query + h * dim, entry), w[h], a.weightScale);
    }
    a.scores[static_cast<std::size_t>(blockIdx.y) * a.poolRows + b] = score;
}

/** @brief Order-preserving map of a float to an unsigned key (larger float -> larger key). */
__device__ inline unsigned orderedKey(float value)
{
    // Bit-level zero test (an FTZ compare would also map subnormals to zero);
    // -0 ties +0, as in the CPU comparator.
    const unsigned raw = __float_as_uint(value);
    const unsigned bits = (raw & 0x7FFFFFFFu) == 0 ? 0u : raw;
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

/** @brief Per-row radix selection of the top-k keys of one pass; output set in ascending block order. */
__global__ __launch_bounds__(kThreads) void indexerSelectKernel(IndexerTopKArgs a, int rowBase)
{
    const int r = rowBase + static_cast<int>(blockIdx.x);
    if (r >= liveRows(a.extent)) return;
    const int visible = min((a.positions[r] + 1) / a.ratio, a.poolRows);
    const int keep = min(a.topK, visible);
    std::int32_t *out = a.selected + static_cast<std::size_t>(r) * a.topK;
    const float *scores = a.scores + static_cast<std::size_t>(blockIdx.x) * a.poolRows;
    __shared__ unsigned histogram[256];
    __shared__ unsigned prefix, prefixMask, remaining;
    if (threadIdx.x == 0) { prefix = 0; prefixMask = 0; remaining = static_cast<unsigned>(keep); }
    __syncthreads();
    // Four 8-bit passes from the most significant digit locate the keep-th
    // largest key T and how many ties at T must be admitted (`remaining`).
    for (int shift = 24; shift >= 0 && keep > 0; shift -= 8)
    {
        histogram[threadIdx.x] = 0;
        __syncthreads();
        for (int b = threadIdx.x; b < visible; b += kThreads)
        {
            const unsigned key = orderedKey(scores[b]);
            if ((key & prefixMask) == prefix) atomicAdd(&histogram[(key >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            unsigned need = remaining;
            int digit = 255;
            for (; digit > 0; --digit)
            {
                if (histogram[digit] >= need) break;
                need -= histogram[digit];
            }
            prefix |= static_cast<unsigned>(digit) << shift;
            prefixMask |= 0xFFu << shift;
            remaining = need;
        }
        __syncthreads();
    }
    // Emit keys > T, then the lowest-id `remaining` keys equal to T, in id order.
    __shared__ int written;
    __shared__ int tiesTaken;
    if (threadIdx.x == 0) { written = 0; tiesTaken = 0; }
    __syncthreads();
    if (keep > 0 && threadIdx.x == 0)
    {
        // Sequential in-order emission keeps the output deterministic and
        // sorted; the scan touches each visible block once.
        for (int b = 0; b < visible; ++b)
        {
            const unsigned key = orderedKey(scores[b]);
            if (key > prefix || (key == prefix && tiesTaken < static_cast<int>(remaining)))
            {
                if (key == prefix) ++tiesTaken;
                out[written++] = b;
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < a.topK; i += kThreads)
        if (i >= keep) out[i] = -1;
}

// --------------------------------------------------------------------------- sparse attention
/** @brief One CTA per (row, head): warps run online softmax over strided keys and merge. */
__global__ __launch_bounds__(kThreads) void sparseAttentionKernel(SparseAttentionArgs a)
{
    const int task = static_cast<int>(blockIdx.x);
    const int r = task / a.heads;
    const int h = task % a.heads;
    if (r >= liveRows(a.extent)) return;
    const int dim = a.format.headDim; // <= 32 * kMaxPerLane
    constexpr int kMaxPerLane = kMaxHeadDim / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int start = contiguousStart(a.positions, r);
    const int pos = start + r;
    const std::size_t entryBytes = static_cast<std::size_t>(a.format.entryBytes());
    const float *q = a.query + static_cast<std::size_t>(task) * dim;
    float qv[kMaxPerLane];
    float acc[kMaxPerLane];
    const int perLane = (dim + 31) / 32;
#pragma unroll
    for (int i = 0; i < kMaxPerLane; ++i)
    {
        const int c = lane + 32 * i;
        qv[i] = i < perLane && c < dim ? q[c] : 0.0f;
        acc[i] = 0.0f;
    }
    // Image spans widen the raw window to the whole span (both directions).
    const auto span = imageSpanExtent(a.tokens, liveRows(a.extent), r, a.imageIds);
    const int rawFirst = max(0, pos - max(a.window - 1, span.left));
    const int rawCount = pos + span.right - rawFirst + 1;
    int compCount = 0;
    if (a.access == CompressedAccess::All) compCount = (pos + 1) / a.ratio;
    else if (a.access == CompressedAccess::Selected) compCount = a.topK;
    const int total = rawCount + compCount;
    float m = -FLT_MAX, l = 0.0f;
    for (int t = warp; t < total; t += kWarps)
    {
        // Key source: this chunk's QAT kv (FP32), else a packed record.
        const float *chunkKey = nullptr;
        const std::uint8_t *entry = nullptr;
        if (t < rawCount)
        {
            const int p = rawFirst + t;
            if (p >= start) chunkKey = a.chunkKV + static_cast<std::size_t>(p - start) * dim;
            else entry = a.rawRing + static_cast<std::size_t>(p % a.window) * entryBytes;
        }
        else
        {
            int block = t - rawCount;
            if (a.access == CompressedAccess::Selected)
            {
                block = a.selected[static_cast<std::size_t>(r) * a.topK + block];
                if (block < 0) continue;
            }
            entry = a.pool + static_cast<std::size_t>(block) * entryBytes;
        }
        float kv[kMaxPerLane];
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < kMaxPerLane; ++i)
        {
            const int c = lane + 32 * i;
            kv[i] = 0.0f;
            if (i < perLane && c < dim)
                kv[i] = chunkKey ? chunkKey[c] : qat::decodeKVChannel(a.format, entry, c);
            dot = fmaf(qv[i], kv[i], dot);
        }
        const float logit = warpSum(dot) * a.scale;
        const float mNew = fmaxf(m, logit);
        const float correction = expf(m - mNew);
        const float p = expf(logit - mNew);
        l = l * correction + p;
#pragma unroll
        for (int i = 0; i < kMaxPerLane; ++i) acc[i] = acc[i] * correction + p * kv[i];
        m = mNew;
    }
    __shared__ float warpM[kWarps], warpL[kWarps];
    __shared__ float merged[kWarps][32 * kMaxPerLane];
    if (lane == 0) { warpM[warp] = m; warpL[warp] = l; }
#pragma unroll
    for (int i = 0; i < kMaxPerLane; ++i) merged[warp][lane + 32 * i] = acc[i];
    __syncthreads();
    if (warp == 0)
    {
        float gm = a.sinks[h];
        for (int w = 0; w < static_cast<int>(kWarps); ++w) gm = fmaxf(gm, warpM[w]);
        float denom = expf(a.sinks[h] - gm);
        for (int w = 0; w < static_cast<int>(kWarps); ++w)
            if (warpL[w] > 0.0f) denom += warpL[w] * expf(warpM[w] - gm);
        for (int i = 0; i < perLane; ++i)
        {
            const int c = lane + 32 * i;
            float value = 0.0f;
            for (int w = 0; w < static_cast<int>(kWarps); ++w)
                if (warpL[w] > 0.0f) value += merged[w][c] * expf(warpM[w] - gm);
            merged[0][c] = value / denom;
        }
    }
    __syncthreads();
    float *out = a.output + static_cast<std::size_t>(task) * dim;
    const int nope = dim - a.rope.ropeDim;
    for (int c = threadIdx.x; c < nope; c += kThreads) out[c] = merged[0][c];
    for (int pair = threadIdx.x; pair < a.rope.ropeDim / 2; pair += kThreads)
    {
        const float angle = static_cast<float>(pos) * math::ropeFrequency(a.rope, pair);
        const float c = cosf(angle), s = -sinf(angle);
        const float x0 = merged[0][nope + 2 * pair], x1 = merged[0][nope + 2 * pair + 1];
        out[nope + 2 * pair] = x0 * c - x1 * s;
        out[nope + 2 * pair + 1] = x0 * s + x1 * c;
    }
}

// --------------------------------------------------------------------------- grouped projection
__global__ __launch_bounds__(kThreads) void groupedProjectionKernel(GroupedProjectionArgs a)
{
    const int outputs = a.groups * a.rank;
    const std::size_t warpIndex = static_cast<std::size_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const std::size_t r = warpIndex / outputs;
    const int o = static_cast<int>(warpIndex % outputs);
    if (r >= static_cast<std::size_t>(liveRows(a.extent))) return;
    const int g = o / a.rank;
    const float *x = a.input + (r * a.groups + g) * a.groupWidth;
    float acc = 0.0f;
    if (a.format == WeightFormat::Q8_0)
    {
        const int blocks = a.groupWidth / 32;
        const auto *row = static_cast<const std::uint8_t *>(a.weight) +
                          static_cast<std::size_t>(o) * blocks * math::kQ8BlockBytes;
        for (int blk = 0; blk < blocks; ++blk)
        {
            const std::uint8_t *b = row + static_cast<std::size_t>(blk) * math::kQ8BlockBytes;
            const std::uint16_t dh = static_cast<std::uint16_t>(b[0] | (b[1] << 8));
            const float value = static_cast<float>(static_cast<std::int8_t>(b[2 + lane])) * x[blk * 32 + lane];
            acc = fmaf(math::halfToFloat(dh), value, acc);
        }
    }
    else if (a.format == WeightFormat::BF16)
    {
        const auto *w = static_cast<const std::uint16_t *>(a.weight) + static_cast<std::size_t>(o) * a.groupWidth;
        for (int c = lane; c < a.groupWidth; c += 32) acc = fmaf(math::bf16ToFloat(w[c]), x[c], acc);
    }
    else
    {
        const float *w = static_cast<const float *>(a.weight) + static_cast<std::size_t>(o) * a.groupWidth;
        for (int c = lane; c < a.groupWidth; c += 32) acc = fmaf(w[c], x[c], acc);
    }
    acc = warpSum(acc);
    if (lane == 0) a.output[r * outputs + o] = acc;
}

// --------------------------------------------------------------------------- routing
__global__ __launch_bounds__(kThreads) void routeKernel(SqrtSoftplusRouteArgs a)
{
    const int r = static_cast<int>(blockIdx.x);
    if (r >= liveRows(a.extent)) return;
    if (threadIdx.x != 0) return;
    // Lane-0 per row: experts (<= 256) x topK (<= 16) comparisons are tiny.
    const float *logit = a.logits + static_cast<std::size_t>(r) * a.experts;
    std::int32_t *ids = a.ids + static_cast<std::size_t>(r) * a.topK;
    float *w = a.weights + static_cast<std::size_t>(r) * a.topK;
    if (a.hashTable)
    {
        const int token = a.tokens[r];
        if (token < 0 || token >= a.vocab) invalidDeviceState();
        for (int i = 0; i < a.topK; ++i)
        {
            ids[i] = a.hashTable[static_cast<std::size_t>(token) * a.topK + i];
            if (ids[i] < 0 || ids[i] >= a.experts) invalidDeviceState();
        }
    }
    else
    {
        for (int i = 0; i < a.topK; ++i)
        {
            int best = -1;
            float bestScore = -FLT_MAX;
            for (int e = 0; e < a.experts; ++e)
            {
                bool taken = false;
                for (int j = 0; j < i; ++j) taken |= ids[j] == e;
                if (taken) continue;
                const float s = math::sqrtSoftplus(logit[e]) + a.bias[e];
                if (best < 0 || s > bestScore) { best = e; bestScore = s; }
            }
            ids[i] = best;
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < a.topK; ++i) { w[i] = math::sqrtSoftplus(logit[ids[i]]); sum += w[i]; }
    for (int i = 0; i < a.topK; ++i) w[i] = w[i] / sum * a.routeScale;
}

/** @brief One CTA per row: copy the bank row named by the position map over sentinel embeddings. */
__global__ __launch_bounds__(kThreads) void imageMergeKernel(ImageMergeArgs a)
{
    const int r = static_cast<int>(blockIdx.x);
    if (r >= liveRows(a.extent) || !a.imageIds.contains(a.tokens[r])) return;
    const int pos = a.positions[r];
    if (pos < 0 || pos >= a.mapRows) invalidDeviceState(); // admission keeps every prompt position inside the map
    const int v = a.map[pos];
    if (v <= 0) return;
    if (v > a.bankRows) invalidDeviceState();
    const float *src = a.bank + static_cast<std::size_t>(v - 1) * a.width;
    float *dst = a.embedding + static_cast<std::size_t>(r) * a.width;
    for (int c = threadIdx.x; c < a.width; c += kThreads) dst[c] = src[c];
}

/** @brief Grid-stride zero fill of 32-bit words. */
__global__ __launch_bounds__(kThreads) void zeroWordsKernel(std::uint32_t *words, std::size_t count)
{
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x; i < count;
         i += static_cast<std::size_t>(gridDim.x) * kThreads)
        words[i] = 0u;
}

// --------------------------------------------------------------------------- host bundle
DSV4_NATIVE_STREAM explicitStream(void *stream)
{
    const auto native = static_cast<DSV4_NATIVE_STREAM>(stream);
    if (!native || native == DSV4_DEFAULT_STREAM || native == DSV4_THREAD_STREAM)
        throw std::invalid_argument("DeepSeekV4 GPU launch requires an exact non-default producer stream");
    return native;
}

void checkLaunch(const char *name)
{
    const DSV4_NATIVE_ERROR status = DSV4_GET_ERROR();
    if (status != DSV4_SUCCESS)
        throw std::runtime_error(std::string("DeepSeekV4 GPU kernel ") + name + " launch failed: " + DSV4_ERROR_STRING(status));
}

unsigned grid(std::size_t count, std::size_t perBlock = kThreads)
{
    const auto value = (count + perBlock - 1) / perBlock;
    if (value == 0 || value > static_cast<std::size_t>(INT_MAX))
        throw std::invalid_argument("DeepSeekV4 launch grid is empty or exceeds backend capacity");
    return static_cast<unsigned>(value);
}

void require(bool condition, const char *message)
{
    if (!condition) throw std::invalid_argument(std::string("DeepSeekV4 GPU: ") + message);
}

class DSV4_IMPLEMENTATION final : public IDeepSeekV4Kernels
{
public:
    void hcRead(const HCReadArgs &a, void *stream) const override
    {
        require(a.trunk && a.mixes && a.scale && a.base && a.output && a.hc > 0 && a.hc <= 4,
                "hcRead operands (GPU supports hc <= 4)");
        require(a.head || (a.post && a.comb), "hcRead block site requires post/comb");
        hcReadKernel<<<grid(a.extent.rows, 1), kThreads, 0, explicitStream(stream)>>>(a);
        checkLaunch("hcRead");
    }
    void hcWrite(const HCWriteArgs &a, void *stream) const override
    {
        require(a.block && a.post && a.comb && a.trunk && a.output && a.hc <= 4, "hcWrite operands (GPU supports hc <= 4)");
        hcWriteKernel<<<grid(static_cast<std::size_t>(a.extent.rows) * a.hidden), kThreads, 0, explicitStream(stream)>>>(a);
        checkLaunch("hcWrite");
    }
    void hcExpand(const HCExpandArgs &a, void *stream) const override
    {
        require(a.input && a.output && a.input != a.output, "hcExpand operands");
        hcExpandKernel<<<grid(static_cast<std::size_t>(a.extent.rows) * a.hidden), kThreads, 0, explicitStream(stream)>>>(a);
        checkLaunch("hcExpand");
    }
    void headNormRope(const HeadNormRopeArgs &a, void *stream) const override
    {
        require(a.data && (a.positions || a.rope.ropeDim == 0) && a.headDim >= a.rope.ropeDim &&
                    a.headDim <= kMaxHeadDim, "headNormRope operands");
        require(a.quant == ActivationQuant::None ||
                    (a.quant == ActivationQuant::FP8NopeBF16Rope ? PackedKVEntryFormat{a.headDim, a.rope.ropeDim}.valid()
                                                                 : PackedIndexerKeyFormat{a.headDim}.valid()),
                "headNormRope QAT geometry is not representable");
        headNormRopeKernel<<<grid(static_cast<std::size_t>(a.extent.rows) * a.heads, 1), kThreads, 0,
                             explicitStream(stream)>>>(a);
        checkLaunch("headNormRope");
    }
    void compressorStore(const CompressorStoreArgs &a, void *stream) const override
    {
        require(a.positions && a.kv && a.score && a.ape && a.stateKV && a.stateScore, "compressorStore operands");
        compressorStoreKernel<<<grid(static_cast<std::size_t>(a.extent.rows) * a.geometry.projectionWidth()), kThreads, 0,
                                explicitStream(stream)>>>(a);
        checkLaunch("compressorStore");
    }
    void compressorPool(const CompressorPoolArgs &a, void *stream) const override
    {
        const auto &g = a.geometry;
        require(a.positions && a.kv && a.score && a.ape && a.stateKV && a.stateScore && a.gain && a.pool &&
                    g.headDim <= kMaxHeadDim,
                "compressorPool operands");
        require(g.stream == CompressedStream::Indexer ? PackedIndexerKeyFormat{g.headDim}.valid()
                    : PackedKVEntryFormat{g.headDim, g.ropeDim}.valid() && g.ropeDim == a.rope.ropeDim,
                "compressorPool record geometry is not representable");
        compressorPoolKernel<<<grid(a.extent.rows, 1), kThreads, 0, explicitStream(stream)>>>(a);
        checkLaunch("compressorPool");
    }
    void rawStore(const RawStoreArgs &a, void *stream) const override
    {
        require(a.positions && a.kv && a.ring && a.window > 0 && a.format.valid() && a.format.headDim <= kMaxHeadDim,
                "rawStore operands");
        rawStoreKernel<<<grid(a.extent.rows, 1), kThreads, 0, explicitStream(stream)>>>(a);
        checkLaunch("rawStore");
    }
    void indexerTopK(const IndexerTopKArgs &a, void *stream) const override
    {
        require(a.positions && a.query && a.weights && a.pool && a.selected && a.scores && a.topK > 0 &&
                    a.scoreRows > 0 && a.format.valid(),
                "indexerTopK operands (GPU requires bounded score scratch)");
        const std::size_t shared = static_cast<std::size_t>(a.heads) * a.format.headDim * sizeof(float);
        require(shared <= 48 * 1024, "indexer query exceeds the default shared-memory budget");
        const auto native = explicitStream(stream);
        // Fixed pass count from the admitted row bound: the scratch holds
        // scoreRows rows, reused in stream order by consecutive passes.
        for (int base = 0; base < a.extent.rows; base += a.scoreRows)
        {
            const int passRows = std::min(a.scoreRows, a.extent.rows - base);
            const dim3 scoreGrid(grid(a.poolRows), static_cast<unsigned>(passRows));
            indexerScoreKernel<<<scoreGrid, kThreads, shared, native>>>(a, base);
            checkLaunch("indexerScore");
            indexerSelectKernel<<<grid(passRows, 1), kThreads, 0, native>>>(a, base);
            checkLaunch("indexerSelect");
        }
    }
    void sparseAttention(const SparseAttentionArgs &a, void *stream) const override
    {
        require(a.positions && a.query && a.chunkKV && a.rawRing && a.sinks && a.output && a.window > 0 &&
                    a.format.valid() && a.format.headDim <= kMaxHeadDim && a.format.ropeDim == a.rope.ropeDim,
                "sparseAttention operands");
        require(a.access == CompressedAccess::None || (a.pool && a.ratio > 0), "compressed access requires a pool");
        sparseAttentionKernel<<<grid(static_cast<std::size_t>(a.extent.rows) * a.heads, 1), kThreads, 0,
                                explicitStream(stream)>>>(a);
        checkLaunch("sparseAttention");
    }
    void groupedProjection(const GroupedProjectionArgs &a, void *stream) const override
    {
        require(a.weight && a.input && a.output && (a.format != WeightFormat::Q8_0 || a.groupWidth % 32 == 0),
                "groupedProjection operands");
        groupedProjectionKernel<<<grid(static_cast<std::size_t>(a.extent.rows) * a.groups * a.rank, kWarps), kThreads,
                                  0, explicitStream(stream)>>>(a);
        checkLaunch("groupedProjection");
    }
    void route(const SqrtSoftplusRouteArgs &a, void *stream) const override
    {
        require(a.logits && a.ids && a.weights && a.topK > 0 && a.topK <= a.experts, "route operands");
        require(a.hashTable ? (a.tokens != nullptr) : (a.bias != nullptr), "route policy operands");
        routeKernel<<<grid(a.extent.rows, 1), 32, 0, explicitStream(stream)>>>(a);
        checkLaunch("route");
    }
    void imageMerge(const ImageMergeArgs &a, void *stream) const override
    {
        require(a.tokens && a.positions && a.bank && a.map && a.embedding && a.width > 0 && a.bankRows > 0 &&
                    a.mapRows > 0, "imageMerge operands");
        imageMergeKernel<<<grid(a.extent.rows, 1), kThreads, 0, explicitStream(stream)>>>(a);
        checkLaunch("imageMerge");
    }
    void zeroBank(void *destination, std::size_t bytes, void *stream) const override
    {
        require(destination && bytes % 4 == 0, "zeroBank requires 4-byte aligned device storage");
        const std::size_t words = bytes / 4;
        zeroWordsKernel<<<std::min<unsigned>(grid(words), 4096u), kThreads, 0, explicitStream(stream)>>>(
            static_cast<std::uint32_t *>(destination), words);
        checkLaunch("zeroBank");
    }
};
} // namespace DSV4_BACKEND_NAMESPACE

std::unique_ptr<IDeepSeekV4Kernels> DSV4_FACTORY()
{
    return std::make_unique<DSV4_BACKEND_NAMESPACE::DSV4_IMPLEMENTATION>();
}
} // namespace llaminar2::deepseek4
