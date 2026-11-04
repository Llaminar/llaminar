/**
 * @file CPUDeepSeekV4Kernels.cpp
 * @brief CPU DeepSeek-V4 primitives: hyper-connections, compressor, indexer,
 *        sparse sink attention, grouped projection and sqrt-softplus routing.
 *
 * Every kernel distributes independent rows (or row x head tasks) through
 * nested-safe OpenMP worksharing; per-row arithmetic is sequential and fixed,
 * so results do not depend on worker count. Persistent attention state is
 * read from and written to packed records (DeepSeekV4KVFormat.h); the
 * current chunk's projections are read from its scratch, so only O(window)
 * raw rows and O(ratio) compressor state persist besides the pools. The
 * only heap use is the indexer's per-row (visible blocks) score vector.
 */
#include "../../deepseek4/DeepSeekV4Kernels.h"
#include "../../deepseek4/DeepSeekV4Math.h"
#include "../../../utils/OpenMPUtils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::deepseek4
{
namespace
{
using namespace math;

/** @brief Throw a fatal configuration diagnostic. */
void require(bool condition, const char *message)
{
    if (!condition) throw std::invalid_argument(std::string("DeepSeekV4 CPU: ") + message);
}

/** @brief Validated host live row count. */
int liveRows(RowExtent extent)
{
    const int count = extent.liveRows ? *extent.liveRows : extent.rows;
    require(count >= 0 && count <= extent.rows, "live rows exceed admitted capacity");
    return count;
}

/** @brief Run @p function(task) for task in [0, count) with nested-safe worksharing. */
template <class Function>
void forTasks(int count, Function function)
{
    auto work = [&]() {
#pragma omp for schedule(static)
        for (int task = 0; task < count; ++task) function(task);
    };
    OMP_WORKSHARE_REGION(work);
}

/**
 * @brief First position of a chunk whose live rows carry contiguous positions.
 * @details Chunk-scratch readers address position p as row p - start; any
 * other row/position mapping is a fatal graph-construction defect.
 */
int contiguousStart(const std::int32_t *positions, int rows)
{
    const int start = positions[0];
    require(start >= 0, "positions must be non-negative");
    for (int r = 1; r < rows; ++r) require(positions[r] == start + r, "chunk positions must be contiguous");
    return start;
}

/** @brief Validate a quantization post-operation against the head geometry. */
void requireQuant(ActivationQuant quant, int headDim, int ropeDim)
{
    if (quant == ActivationQuant::FP8NopeBF16Rope)
        require(PackedKVEntryFormat{headDim, ropeDim}.valid(), "FP8 QAT requires nope width divisible by 64");
    else if (quant == ActivationQuant::HadamardFP4)
        require(PackedIndexerKeyFormat{headDim}.valid(), "Hadamard FP4 QAT requires a power-of-two width >= 32");
}

/** @brief Exact E4M3 decode table (256 codes). */
const float *e4m3Table()
{
    static const auto table = [] {
        std::array<float, 256> values{};
        for (int code = 0; code < 256; ++code) values[code] = qat::decodeE4M3(static_cast<std::uint8_t>(code));
        return values;
    }();
    return table.data();
}

/** @brief Decode one packed KV record into @p out (exactly qat::decodeKVChannel for every channel). */
void decodeKVEntry(const PackedKVEntryFormat &format, const std::uint8_t *entry, float *out)
{
    const float *table = e4m3Table();
    const int nope = format.nopeDim();
    for (int b = 0; b < format.scaleCount(); ++b)
    {
        const float scale = qat::exp2i(static_cast<int>(entry[format.scaleOffset() + b]) - 127);
        for (int i = b * kFP8BlockSize; i < (b + 1) * kFP8BlockSize; ++i) out[i] = table[entry[i]] * scale;
    }
    for (int c = 0; c < format.ropeDim; ++c)
    {
        const int at = format.ropeOffset() + 2 * c;
        out[nope + c] = qat::decodeBF16(static_cast<std::uint16_t>(entry[at] | (entry[at + 1] << 8)));
    }
}

class CPUDeepSeekV4Kernels final : public IDeepSeekV4Kernels
{
public:
    void hcRead(const HCReadArgs &a, void *) const override
    {
        require(a.trunk && a.mixes && a.scale && a.base && a.output && a.hidden > 0 && a.hc > 0,
                "hcRead requires trunk, mixes, scale, base and output");
        require(a.head || (a.post && a.comb), "hcRead block site requires post and comb outputs");
        const int rows = liveRows(a.extent);
        const int hc = a.hc;
        const int width = hc * a.hidden;
        const int mixWidth = a.head ? hc : (2 + hc) * hc;
        forTasks(rows, [&](int r) {
            const float *x = a.trunk + static_cast<std::size_t>(r) * width;
            double ss = 0.0;
            for (int i = 0; i < width; ++i) ss += static_cast<double>(x[i]) * x[i];
            const float rs = 1.0f / sqrtf(static_cast<float>(ss / width) + a.normEps);
            const float *m = a.mixes + static_cast<std::size_t>(r) * mixWidth;
            float pre[16];
            for (int j = 0; j < hc; ++j) pre[j] = sigmoid(m[j] * rs * a.scale[0] + a.base[j]) + a.hcEps;
            if (!a.head)
            {
                float *post = a.post + static_cast<std::size_t>(r) * hc;
                for (int j = 0; j < hc; ++j)
                    post[j] = 2.0f * sigmoid(m[hc + j] * rs * a.scale[1] + a.base[hc + j]);
                float *comb = a.comb + static_cast<std::size_t>(r) * hc * hc;
                for (int i = 0; i < hc * hc; ++i)
                    comb[i] = m[2 * hc + i] * rs * a.scale[2] + a.base[2 * hc + i];
                sinkhorn(comb, hc, a.sinkhornIters, a.hcEps);
            }
            float *out = a.output + static_cast<std::size_t>(r) * a.hidden;
            for (int d = 0; d < a.hidden; ++d)
            {
                float y = 0.0f;
                for (int j = 0; j < hc; ++j) y += pre[j] * x[static_cast<std::size_t>(j) * a.hidden + d];
                out[d] = y;
            }
            if (a.mixed)
                std::copy(out, out + a.hidden, a.mixed + static_cast<std::size_t>(r) * a.hidden);
            if (a.gain)
            {
                double s2 = 0.0;
                for (int d = 0; d < a.hidden; ++d) s2 += static_cast<double>(out[d]) * out[d];
                const float inv = 1.0f / sqrtf(static_cast<float>(s2 / a.hidden) + a.outputEps);
                for (int d = 0; d < a.hidden; ++d) out[d] = out[d] * inv * a.gain[d];
            }
        });
    }

    void hcWrite(const HCWriteArgs &a, void *) const override
    {
        require(a.block && a.post && a.comb && a.trunk && a.output && a.hc > 0 && a.hc <= 16,
                "hcWrite requires block, post, comb, trunk and output");
        const int rows = liveRows(a.extent);
        const int hc = a.hc;
        forTasks(rows, [&](int r) {
            const std::size_t base = static_cast<std::size_t>(r) * hc * a.hidden;
            const float *post = a.post + static_cast<std::size_t>(r) * hc;
            const float *comb = a.comb + static_cast<std::size_t>(r) * hc * hc;
            const float *block = a.block + static_cast<std::size_t>(r) * a.hidden;
            for (int d = 0; d < a.hidden; ++d)
            {
                float in[16];
                for (int j = 0; j < hc; ++j) in[j] = a.trunk[base + static_cast<std::size_t>(j) * a.hidden + d];
                for (int k = 0; k < hc; ++k)
                {
                    float y = post[k] * block[d];
                    for (int j = 0; j < hc; ++j) y += comb[j * hc + k] * in[j];
                    a.output[base + static_cast<std::size_t>(k) * a.hidden + d] = y;
                }
            }
        });
    }

    void hcExpand(const HCExpandArgs &a, void *) const override
    {
        require(a.input && a.output && a.input != a.output, "hcExpand requires distinct input and output");
        const int rows = liveRows(a.extent);
        forTasks(rows, [&](int r) {
            const float *in = a.input + static_cast<std::size_t>(r) * a.hidden;
            float *out = a.output + static_cast<std::size_t>(r) * a.hc * a.hidden;
            for (int j = 0; j < a.hc; ++j) std::copy(in, in + a.hidden, out + static_cast<std::size_t>(j) * a.hidden);
        });
    }

    void headNormRope(const HeadNormRopeArgs &a, void *) const override
    {
        require(a.data && (a.positions || a.rope.ropeDim == 0) && a.headDim >= a.rope.ropeDim &&
                    a.rope.ropeDim % 2 == 0,
                "headNormRope requires data, positions (when rotating) and headDim >= ropeDim");
        requireQuant(a.quant, a.headDim, a.rope.ropeDim);
        const int rows = liveRows(a.extent);
        forTasks(rows * a.heads, [&](int task) {
            const int r = task / a.heads;
            float *v = a.data + static_cast<std::size_t>(task) * a.headDim;
            if (a.normalize)
            {
                double ss = 0.0;
                for (int d = 0; d < a.headDim; ++d) ss += static_cast<double>(v[d]) * v[d];
                const float inv = 1.0f / sqrtf(static_cast<float>(ss / a.headDim) + a.eps);
                for (int d = 0; d < a.headDim; ++d) v[d] = v[d] * inv * (a.gain ? a.gain[d] : 1.0f);
            }
            if (a.rope.ropeDim > 0) rotateTail(v, a.headDim, a.rope, a.positions[r], a.inverse);
            qat::apply(a.quant, v, a.headDim, a.rope.ropeDim);
        });
    }

    void compressorStore(const CompressorStoreArgs &a, void *) const override
    {
        const auto &g = a.geometry;
        require(a.positions && a.kv && a.score && a.ape && a.stateKV && a.stateScore,
                "compressorStore requires positions, projections, ape and state");
        const int rows = liveRows(a.extent);
        if (rows == 0) return;
        const int start = contiguousStart(a.positions, rows);
        const int width = g.projectionWidth();
        const int states = g.stateRows();
        const int last = start + rows - 1;
        // Only the final stateRows() positions survive; each slot has one writer.
        forTasks(rows, [&](int r) {
            const int pos = start + r;
            if (pos <= last - states) return;
            const std::size_t slot = static_cast<std::size_t>(pos % states) * width;
            const float *kv = a.kv + static_cast<std::size_t>(r) * width;
            const float *sc = a.score + static_cast<std::size_t>(r) * width;
            const float *ape = a.ape + static_cast<std::size_t>(pos % g.ratio) * width;
            for (int c = 0; c < width; ++c)
            {
                a.stateKV[slot + c] = kv[c];
                a.stateScore[slot + c] = sc[c] + ape[c];
            }
        });
    }

    void compressorPool(const CompressorPoolArgs &a, void *) const override
    {
        const auto &g = a.geometry;
        require(a.positions && a.kv && a.score && a.ape && a.stateKV && a.stateScore && a.gain && a.pool,
                "compressorPool requires operands");
        require(g.headDim <= kMaxHeadDim && (g.stream == CompressedStream::Indexer
                    ? PackedIndexerKeyFormat{g.headDim}.valid()
                    : PackedKVEntryFormat{g.headDim, g.ropeDim}.valid() && g.ropeDim == a.rope.ropeDim),
                "compressorPool record geometry is not representable");
        const int rows = liveRows(a.extent);
        if (rows == 0) return;
        const int start = contiguousStart(a.positions, rows);
        const int width = g.projectionWidth();
        const int states = g.stateRows();
        const int d = g.headDim;
        const std::size_t entryBytes = static_cast<std::size_t>(g.entryBytes());
        // Projection of position p: the chunk scratch, else the carried state.
        const auto kvAt = [&](int p, int channel) {
            return p >= start ? a.kv[static_cast<std::size_t>(p - start) * width + channel]
                              : a.stateKV[static_cast<std::size_t>(p % states) * width + channel];
        };
        const auto scoreAt = [&](int p, int channel) {
            return p >= start ? a.score[static_cast<std::size_t>(p - start) * width + channel] +
                                    a.ape[static_cast<std::size_t>(p % g.ratio) * width + channel]
                              : a.stateScore[static_cast<std::size_t>(p % states) * width + channel];
        };
        forTasks(rows, [&](int r) {
            const int pos = start + r;
            if ((pos + 1) % g.ratio != 0) return;
            const int block = pos / g.ratio;
            require(block < g.poolRows, "compressed pool capacity exceeded");
            float out[kMaxHeadDim];
            const int first = block * g.ratio;
            const int entries = g.overlap() ? 2 * g.ratio : g.ratio;
            for (int c = 0; c < d; ++c)
            {
                // Entries: [previous block, first half] (overlap) then [this block, own half].
                float m = -std::numeric_limits<float>::infinity();
                for (int e = 0; e < entries; ++e)
                {
                    const bool prev = g.overlap() && e < g.ratio;
                    const int p = prev ? first - g.ratio + e : first + (g.overlap() ? e - g.ratio : e);
                    if (p < 0) continue;
                    m = std::max(m, scoreAt(p, prev ? c : (g.overlap() ? d + c : c)));
                }
                float sum = 0.0f, acc = 0.0f;
                for (int e = 0; e < entries; ++e)
                {
                    const bool prev = g.overlap() && e < g.ratio;
                    const int p = prev ? first - g.ratio + e : first + (g.overlap() ? e - g.ratio : e);
                    if (p < 0) continue;
                    const int channel = prev ? c : (g.overlap() ? d + c : c);
                    const float w = expf(scoreAt(p, channel) - m);
                    sum += w;
                    acc += w * kvAt(p, channel);
                }
                out[c] = acc / sum;
            }
            double ss = 0.0;
            for (int c = 0; c < d; ++c) ss += static_cast<double>(out[c]) * out[c];
            const float inv = 1.0f / sqrtf(static_cast<float>(ss / d) + a.eps);
            for (int c = 0; c < d; ++c) out[c] = out[c] * inv * a.gain[c];
            rotateTail(out, d, a.rope, block * g.ratio, false);
            std::uint8_t *entry = a.pool + static_cast<std::size_t>(block) * entryBytes;
            if (g.stream == CompressedStream::Attention)
                qat::packKVEntry({d, g.ropeDim}, out, entry);
            else
            {
                qat::hadamard(out, d);
                qat::packIndexerKey({d}, out, entry);
            }
        });
    }

    void rawStore(const RawStoreArgs &a, void *) const override
    {
        require(a.positions && a.kv && a.ring && a.window > 0 && a.format.valid(), "rawStore requires operands");
        const int rows = liveRows(a.extent);
        if (rows == 0) return;
        const int start = contiguousStart(a.positions, rows);
        const int last = start + rows - 1;
        const std::size_t entryBytes = static_cast<std::size_t>(a.format.entryBytes());
        forTasks(rows, [&](int r) {
            const int pos = start + r;
            if (pos <= last - a.window) return;
            qat::packKVEntry(a.format, a.kv + static_cast<std::size_t>(r) * a.format.headDim,
                             a.ring + static_cast<std::size_t>(pos % a.window) * entryBytes);
        });
    }

    void indexerTopK(const IndexerTopKArgs &a, void *) const override
    {
        require(a.positions && a.query && a.weights && a.pool && a.selected && a.topK > 0 && a.format.valid(),
                "indexerTopK requires operands");
        const int rows = liveRows(a.extent);
        // The CPU bundle scores every row at once; an optional score publication
        // (diagnostics / cross-backend proof) therefore needs one row per live row.
        require(!a.scores || a.scoreRows >= rows, "CPU indexer score publication needs scoreRows >= live rows");
        const int dim = a.format.headDim;
        const int blocks = a.format.scaleCount();
        const std::size_t entryBytes = static_cast<std::size_t>(a.format.entryBytes());
        forTasks(rows, [&](int r) {
            const int visible = std::min((a.positions[r] + 1) / a.ratio, a.poolRows);
            std::vector<std::pair<float, int>> scored(static_cast<std::size_t>(visible));
            const float *q = a.query + static_cast<std::size_t>(r) * a.heads * dim;
            const float *w = a.weights + static_cast<std::size_t>(r) * a.heads;
            float key[kMaxHeadDim];
            float scale[kMaxHeadDim / kFP4BlockSize];
            for (int b = 0; b < visible; ++b)
            {
                // Decode the key once (unscaled E2M1 values + block scales);
                // the per-head arithmetic below is qat::indexerDot's order.
                const std::uint8_t *entry = a.pool + static_cast<std::size_t>(b) * entryBytes;
                for (int c = 0; c < dim; c += 2)
                {
                    key[c] = qat::decodeE2M1(static_cast<std::uint8_t>(entry[c / 2] & 0xFu));
                    key[c + 1] = qat::decodeE2M1(static_cast<std::uint8_t>(entry[c / 2] >> 4));
                }
                for (int j = 0; j < blocks; ++j)
                    scale[j] = qat::exp2i(static_cast<int>(entry[a.format.scaleOffset() + j]) - 127);
                // Exactly qat::indexerDot / indexerAccumulate's operation sequence.
                float score = 0.0f;
                for (int h = 0; h < a.heads; ++h)
                {
                    const float *qh = q + static_cast<std::size_t>(h) * dim;
                    float dot = 0.0f;
                    for (int j = 0; j < blocks; ++j)
                    {
                        float partial = 0.0f;
                        for (int i = j * kFP4BlockSize; i < (j + 1) * kFP4BlockSize; ++i)
                            partial = qat::fmaRn(qh[i], key[i], partial);
                        dot = qat::fmaRn(partial, scale[j], dot);
                    }
                    score = qat::indexerAccumulate(score, dot, w[h], a.weightScale);
                }
                scored[b] = {score, b};
                if (a.scores) a.scores[static_cast<std::size_t>(r) * a.poolRows + b] = score;
            }
            const int keep = std::min(a.topK, visible);
            std::partial_sort(scored.begin(), scored.begin() + keep, scored.end(),
                              [](const auto &x, const auto &y) {
                                  return x.first > y.first || (x.first == y.first && x.second < y.second);
                              });
            // Publish the selected set in ascending block order: attention is
            // order-independent and the GPU radix selection yields the same set.
            std::int32_t *out = a.selected + static_cast<std::size_t>(r) * a.topK;
            for (int i = 0; i < a.topK; ++i) out[i] = i < keep ? scored[i].second : -1;
            std::sort(out, out + keep);
        });
    }

    void sparseAttention(const SparseAttentionArgs &a, void *) const override
    {
        require(a.positions && a.query && a.chunkKV && a.rawRing && a.sinks && a.output && a.window > 0 &&
                    a.format.valid() && a.format.headDim <= kMaxHeadDim && a.format.ropeDim == a.rope.ropeDim,
                "sparseAttention requires operands and a representable record format");
        require(a.access == CompressedAccess::None || (a.pool && a.ratio > 0),
                "compressed access requires a pool and ratio");
        require(a.access != CompressedAccess::Selected || (a.selected && a.topK > 0),
                "selected access requires indexer output");
        const int rows = liveRows(a.extent);
        if (rows == 0) return;
        const int start = contiguousStart(a.positions, rows);
        const int dim = a.format.headDim;
        const std::size_t entryBytes = static_cast<std::size_t>(a.format.entryBytes());
        forTasks(rows * a.heads, [&](int task) {
            const int r = task / a.heads;
            const int h = task % a.heads;
            const int pos = start + r;
            const float *q = a.query + static_cast<std::size_t>(task) * dim;
            float *out = a.output + static_cast<std::size_t>(task) * dim;
            float decoded[kMaxHeadDim];
            // Online softmax seeded with the sink: m = sink, l = exp(sink - m) = 1.
            float m = a.sinks[h], l = 1.0f;
            std::fill(out, out + dim, 0.0f);
            const auto attend = [&](const float *key) {
                float dot = 0.0f;
                for (int c = 0; c < dim; ++c) dot += q[c] * key[c];
                const float logit = dot * a.scale;
                const float mNew = std::max(m, logit);
                const float correction = expf(m - mNew);
                const float p = expf(logit - mNew);
                l = l * correction + p;
                for (int c = 0; c < dim; ++c) out[c] = out[c] * correction + p * key[c];
                m = mNew;
            };
            const auto attendPacked = [&](const std::uint8_t *entry) {
                decodeKVEntry(a.format, entry, decoded);
                attend(decoded);
            };
            for (int p = std::max(0, pos - a.window + 1); p <= pos; ++p)
            {
                if (p >= start) attend(a.chunkKV + static_cast<std::size_t>(p - start) * dim);
                else attendPacked(a.rawRing + static_cast<std::size_t>(p % a.window) * entryBytes);
            }
            if (a.access == CompressedAccess::All)
            {
                const int visible = (pos + 1) / a.ratio;
                for (int b = 0; b < visible; ++b) attendPacked(a.pool + static_cast<std::size_t>(b) * entryBytes);
            }
            else if (a.access == CompressedAccess::Selected)
            {
                const std::int32_t *sel = a.selected + static_cast<std::size_t>(r) * a.topK;
                for (int i = 0; i < a.topK; ++i)
                    if (sel[i] >= 0) attendPacked(a.pool + static_cast<std::size_t>(sel[i]) * entryBytes);
            }
            for (int c = 0; c < dim; ++c) out[c] /= l;
            rotateTail(out, dim, a.rope, pos, true);
        });
    }

    void groupedProjection(const GroupedProjectionArgs &a, void *) const override
    {
        require(a.weight && a.input && a.output && a.groups > 0 && a.rank > 0 && a.groupWidth > 0,
                "groupedProjection requires operands");
        require(a.format != WeightFormat::Q8_0 || a.groupWidth % 32 == 0, "Q8_0 group width must be a multiple of 32");
        const int rows = liveRows(a.extent);
        const int outputs = a.groups * a.rank;
        forTasks(rows * outputs, [&](int task) {
            const int r = task / outputs;
            const int o = task % outputs;
            const int g = o / a.rank;
            const float *x = a.input + (static_cast<std::size_t>(r) * a.groups + g) * a.groupWidth;
            float acc = 0.0f;
            switch (a.format)
            {
            case WeightFormat::FP32:
            {
                const float *w = static_cast<const float *>(a.weight) + static_cast<std::size_t>(o) * a.groupWidth;
                for (int c = 0; c < a.groupWidth; ++c) acc += w[c] * x[c];
                break;
            }
            case WeightFormat::BF16:
            {
                const auto *w = static_cast<const std::uint16_t *>(a.weight) + static_cast<std::size_t>(o) * a.groupWidth;
                for (int c = 0; c < a.groupWidth; ++c) acc += bf16ToFloat(w[c]) * x[c];
                break;
            }
            case WeightFormat::Q8_0:
            {
                const auto *row = static_cast<const std::uint8_t *>(a.weight) +
                                  static_cast<std::size_t>(o) * (a.groupWidth / 32) * kQ8BlockBytes;
                for (int blk = 0; blk < a.groupWidth / 32; ++blk)
                {
                    const std::uint8_t *b = row + static_cast<std::size_t>(blk) * kQ8BlockBytes;
                    std::uint16_t dh;
                    std::memcpy(&dh, b, 2);
                    const auto *qs = reinterpret_cast<const std::int8_t *>(b + 2);
                    float partial = 0.0f;
                    for (int i = 0; i < 32; ++i) partial += static_cast<float>(qs[i]) * x[blk * 32 + i];
                    acc += halfToFloat(dh) * partial;
                }
                break;
            }
            }
            a.output[static_cast<std::size_t>(r) * outputs + o] = acc;
        });
    }

    void route(const SqrtSoftplusRouteArgs &a, void *) const override
    {
        require(a.logits && a.ids && a.weights && a.topK > 0 && a.topK <= a.experts,
                "route requires logits, ids, weights and 0 < topK <= experts");
        require(a.hashTable ? (a.tokens && a.vocab > 0) : a.bias != nullptr,
                "score routing requires bias; hash routing requires tokens and table");
        const int rows = liveRows(a.extent);
        forTasks(rows, [&](int r) {
            const float *logit = a.logits + static_cast<std::size_t>(r) * a.experts;
            std::int32_t *ids = a.ids + static_cast<std::size_t>(r) * a.topK;
            float *w = a.weights + static_cast<std::size_t>(r) * a.topK;
            if (a.hashTable)
            {
                const int token = a.tokens[r];
                require(token >= 0 && token < a.vocab, "hash routing token out of range");
                for (int i = 0; i < a.topK; ++i)
                {
                    ids[i] = a.hashTable[static_cast<std::size_t>(token) * a.topK + i];
                    require(ids[i] >= 0 && ids[i] < a.experts, "hash routing expert id out of range");
                }
            }
            else
            {
                std::vector<std::pair<float, int>> biased(static_cast<std::size_t>(a.experts));
                for (int e = 0; e < a.experts; ++e) biased[e] = {sqrtSoftplus(logit[e]) + a.bias[e], e};
                std::partial_sort(biased.begin(), biased.begin() + a.topK, biased.end(),
                                  [](const auto &x, const auto &y) {
                                      return x.first > y.first || (x.first == y.first && x.second < y.second);
                                  });
                for (int i = 0; i < a.topK; ++i) ids[i] = biased[i].second;
            }
            float sum = 0.0f;
            for (int i = 0; i < a.topK; ++i)
            {
                w[i] = sqrtSoftplus(logit[ids[i]]);
                sum += w[i];
            }
            for (int i = 0; i < a.topK; ++i) w[i] = w[i] / sum * a.routeScale;
        });
    }

    void zeroBank(void *destination, std::size_t bytes, void *) const override
    {
        require(destination && bytes % 4 == 0, "zeroBank requires 4-byte aligned host storage");
        std::memset(destination, 0, bytes);
    }
};
} // namespace

std::unique_ptr<IDeepSeekV4Kernels> createCPUDeepSeekV4Kernels()
{
    return std::make_unique<CPUDeepSeekV4Kernels>();
}
} // namespace llaminar2::deepseek4
