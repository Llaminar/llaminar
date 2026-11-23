/**
 * @file DeepSeekV4VisionEncoder.cpp
 * @brief Memory-minimal FP32 BLAS implementation of the official vision.py tower.
 *
 * Matrices are row-major [out, in] (a GGUF tensor with ne = {in, out}) and stay
 * resident in their file precision. Every projection is ``Y = X . W^T (+ b)``
 * through oneDNN ``dnnl_sgemm`` (row-major); a 16-bit matrix is widened exactly
 * into a bounded scratch one output-row chunk at a time, so FP32 copies of the
 * weights never exist. Attention computes one head's scores for a tile of
 * query rows at a time ([tile, N], never [N, N]), applies an exact
 * max-subtracted softmax per row and multiplies by V; the MLP likewise runs
 * one tile of rows through gate/up/down. Rows are independent in both, so
 * tiling changes only BLAS blocking, not the arithmetic of a row.
 */
#include "DeepSeekV4VisionEncoder.h"

#include "../../loaders/ModelLoader.h"
#include "../../tensors/TensorFactory.h"
#include "../../tensors/Tensors.h"
#ifndef HAVE_ONEDNN
#error "The DeepSeek-V4 host vision tower requires oneDNN (dnnl_sgemm)"
#endif
#include <oneapi/dnnl/dnnl.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include "../../utils/OpenMPUtils.h"

namespace llaminar2::deepseek4
{
/** @brief Admitted decoded-pixel bound (matches DeepSeekV4PromptImages' decode cap). */
constexpr std::uint64_t kHostDecodedPixelBound = std::uint64_t{64} << 20;
constexpr int kVisionMaxTokens = 384;    ///< vision_max_n_token (official, fixed).
constexpr int kVisionDefaultMerge = 3;   ///< Llama.cpp default scale_factor.
/** @brief Query/MLP rows per tile: bounds the score tile to kVisionRowTile x N floats. */
constexpr std::size_t kVisionRowTile = 256;
/** @brief FP32 elements of the per-call weight-widening scratch (4 MiB). */
constexpr std::size_t kVisionWeightScratchFloats = std::size_t{1} << 20;
} // namespace llaminar2::deepseek4

namespace llaminar2::deepseek4
{
    namespace
    {
        constexpr float kNormEps = 1e-6f;
        constexpr double kRopeTheta = 10000.0;

        /** @brief Exact BF16 -> FP32 widening. */
        inline float widenBF16(std::uint16_t bits) noexcept
        {
            const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
            float f;
            std::memcpy(&f, &u, sizeof(f));
            return f;
        }

        /** @brief Exact IEEE binary16 -> FP32 widening (subnormals, inf and NaN included). */
        inline float widenF16(std::uint16_t h) noexcept
        {
            const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
            std::uint32_t exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu, u;
            if (exp == 0x1Fu)
                u = sign | 0x7F800000u | (mant << 13);
            else if (exp != 0)
                u = sign | ((exp + 112u) << 23) | (mant << 13);
            else if (mant == 0)
                u = sign;
            else
            {
                // Subnormal half: normalize into an FP32 normal.
                int e = -1;
                do { ++e; mant <<= 1; } while ((mant & 0x400u) == 0);
                u = sign | ((112u - static_cast<std::uint32_t>(e)) << 23) | ((mant & 0x3FFu) << 13);
            }
            float f;
            std::memcpy(&f, &u, sizeof(f));
            return f;
        }

        void requireShape(const VisionHostTensor &t, const std::string &name, std::initializer_list<std::size_t> shape)
        {
            if (!std::equal(shape.begin(), shape.end(), t.shape.begin(), t.shape.end()))
                throw std::invalid_argument("DeepSeek-V4 mmproj tensor " + name + " has an unexpected shape");
        }

        /** @brief Consume a vector tensor, widened once to FP32 (vectors are tiny). */
        std::vector<float> take(std::map<std::string, VisionHostTensor> &t, const std::string &name,
                                std::initializer_list<std::size_t> shape)
        {
            auto it = t.find(name);
            if (it == t.end())
                throw std::invalid_argument("DeepSeek-V4 mmproj is missing " + name);
            requireShape(it->second, name, shape);
            std::vector<float> out;
            if (it->second.precision == VisionHostPrecision::F32)
                out = std::move(it->second.data);
            else
            {
                out.resize(it->second.half_bits.size());
                const bool bf16 = it->second.precision == VisionHostPrecision::BF16;
                for (std::size_t i = 0; i < out.size(); ++i)
                    out[i] = bf16 ? widenBF16(it->second.half_bits[i]) : widenF16(it->second.half_bits[i]);
            }
            t.erase(it);
            return out;
        }

        /** @brief Consume a projection as [rows = out, cols = in] in its resident precision. */
        VisionMatrix takeMatrix(std::map<std::string, VisionHostTensor> &t, const std::string &name,
                                std::initializer_list<std::size_t> shape, std::size_t rows)
        {
            auto it = t.find(name);
            if (it == t.end())
                throw std::invalid_argument("DeepSeek-V4 mmproj is missing " + name);
            requireShape(it->second, name, shape);
            std::size_t numel = 1;
            for (const auto d : shape) numel *= d;
            if (rows == 0 || numel % rows != 0)
                throw std::invalid_argument("DeepSeek-V4 mmproj tensor " + name + " is not a row-major matrix");
            VisionMatrix m;
            m.rows = static_cast<int>(rows);
            m.cols = static_cast<int>(numel / rows);
            m.precision = it->second.precision;
            if (m.precision == VisionHostPrecision::F32)
                m.f32 = std::move(it->second.data);
            else
                m.half_bits = std::move(it->second.half_bits);
            if ((m.f32.size() + m.half_bits.size()) != numel)
                throw std::invalid_argument("DeepSeek-V4 mmproj tensor " + name + " storage disagrees with its shape");
            t.erase(it);
            return m;
        }

        /**
         * @brief Take the fused [3D, D] attention projection and its bias.
         *
         * llama.cpp-converted projectors carry one fused attn_qkv; Unsloth's
         * published deepseek4v mmproj carries separate attn_q/attn_k/attn_v
         * (each [D, D] with a bias). The forward pass reads output columns
         * [q | k | v], so stacking the separate matrices' rows in that order
         * reproduces the fused matrix exactly: a lossless copy of the stored
         * BF16/F16/F32 values with no arithmetic change.
         */
        std::pair<VisionMatrix, std::vector<float>> takeQKV(std::map<std::string, VisionHostTensor> &t,
                                                             const std::string &pre, std::size_t D)
        {
            if (t.count(pre + "attn_qkv.weight"))
            {
                auto w = takeMatrix(t, pre + "attn_qkv.weight", {D, 3 * D}, 3 * D);
                return {std::move(w), take(t, pre + "attn_qkv.bias", {3 * D})};
            }
            VisionMatrix fused;
            std::vector<float> bias;
            bias.reserve(3 * D);
            for (const char *part : {"attn_q", "attn_k", "attn_v"})
            {
                auto w = takeMatrix(t, pre + part + ".weight", {D, D}, D);
                if (fused.rows == 0)
                {
                    fused.cols = w.cols;
                    fused.precision = w.precision;
                }
                else if (w.precision != fused.precision)
                    throw std::invalid_argument("DeepSeek-V4 mmproj " + pre + "attn_q/k/v disagree on precision");
                fused.rows += w.rows;
                fused.f32.insert(fused.f32.end(), w.f32.begin(), w.f32.end());
                fused.half_bits.insert(fused.half_bits.end(), w.half_bits.begin(), w.half_bits.end());
                const auto b = take(t, pre + part + ".bias", {D});
                bias.insert(bias.end(), b.begin(), b.end());
            }
            return {std::move(fused), std::move(bias)};
        }

        /** @brief Row-major C[m, n] = alpha * op(A) . op(B); fatal on a oneDNN error. */
        void sgemm(char ta, char tb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb,
                   float *c, int ldc)
        {
            if (dnnl_sgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, 0.0f, c, ldc) != dnnl_success)
                throw std::runtime_error("DeepSeek-V4 vision tower: dnnl_sgemm failed");
        }

        /** @return Output rows of one widening chunk for a K-wide matrix. */
        inline std::size_t widenChunkRows(std::size_t k) noexcept
        {
            return std::max<std::size_t>(1, kVisionWeightScratchFloats / std::max<std::size_t>(k, 1));
        }

        /**
         * @brief Y[m, n] (row stride ldy) = X[m, k] . W[n, k]^T (+ bias[n]).
         * @param scratch Widening scratch of at least widenChunkRows(k) * k floats
         *        (unused for an FP32 matrix).
         */
        void linear(const float *x, int m, int k, const VisionMatrix &w, const float *bias, int n, float *y,
                    std::vector<float> &scratch, int ldy = -1)
        {
            if (w.rows != n || w.cols != k)
                throw std::invalid_argument("DeepSeek-V4 vision projection geometry mismatch");
            if (ldy < 0) ldy = n;
            if (w.precision == VisionHostPrecision::F32)
                sgemm('N', 'T', m, n, k, 1.0f, x, k, w.f32.data(), k, y, ldy);
            else
            {
                const std::size_t chunk = widenChunkRows(static_cast<std::size_t>(k));
                if (scratch.size() < chunk * static_cast<std::size_t>(k))
                    scratch.resize(chunk * static_cast<std::size_t>(k));
                const bool bf16 = w.precision == VisionHostPrecision::BF16;
                for (std::size_t r0 = 0; r0 < static_cast<std::size_t>(n); r0 += chunk)
                {
                    const std::size_t rows = std::min(chunk, static_cast<std::size_t>(n) - r0);
                    const std::uint16_t *src = w.half_bits.data() + r0 * static_cast<std::size_t>(k);
                    const std::size_t count = rows * static_cast<std::size_t>(k);
                    auto do_widen = [&]()
                    {
#pragma omp for schedule(static)
                        for (std::size_t i = 0; i < count; ++i)
                            scratch[i] = bf16 ? widenBF16(src[i]) : widenF16(src[i]);
                    };
                    OMP_WORKSHARE_REGION(do_widen);
                    sgemm('N', 'T', m, static_cast<int>(rows), k, 1.0f, x, k, scratch.data(), k, y + r0, ldy);
                }
            }
            if (bias)
            {
                auto do_bias = [&]()
                {
#pragma omp for schedule(static)
                    for (int r = 0; r < m; ++r)
                        for (int c = 0; c < n; ++c) y[static_cast<std::size_t>(r) * ldy + c] += bias[c];
                };
                OMP_WORKSHARE_REGION(do_bias);
            }
        }

        void rmsNorm(const float *x, int rows, int dim, const std::vector<float> &gain, float *y)
        {
            auto do_rms = [&]()
            {
#pragma omp for schedule(static)
                for (int r = 0; r < rows; ++r)
                {
                    const float *in = x + static_cast<std::size_t>(r) * dim;
                    float *out = y + static_cast<std::size_t>(r) * dim;
                    double ss = 0.0;
                    for (int c = 0; c < dim; ++c) ss += static_cast<double>(in[c]) * in[c];
                    const float inv = 1.0f / std::sqrt(static_cast<float>(ss / dim) + kNormEps);
                    for (int c = 0; c < dim; ++c) out[c] = gain[c] * (in[c] * inv);
                }
            };
            OMP_WORKSHARE_REGION(do_rms);
        }
    } // namespace

    DeepSeekV4VisionEncoder::DeepSeekV4VisionEncoder(std::map<std::string, VisionHostTensor> t,
                                                     VisionPreprocessConfig config, int layers, int heads)
        : config_(config), layers_(layers), heads_(heads)
    {
        const auto patch_it = t.find("v.patch_embd.weight");
        if (patch_it == t.end() || patch_it->second.shape.size() != 4)
            throw std::invalid_argument("DeepSeek-V4 mmproj is missing v.patch_embd.weight [p, p, 3, dim]");
        const std::size_t p = static_cast<std::size_t>(config_.patch);
        dim_ = static_cast<int>(patch_it->second.shape[3]);
        const std::size_t D = static_cast<std::size_t>(dim_);
        if (layers_ <= 0 || heads_ <= 0 || dim_ % heads_ || (dim_ / heads_) % 4)
            throw std::invalid_argument("DeepSeek-V4 vision geometry requires head_dim divisible by 4");
        patch_w_ = takeMatrix(t, "v.patch_embd.weight", {p, p, 3, D}, D);
        patch_b_ = take(t, "v.patch_embd.bias", {D});
        const auto gate_it = t.find("v.blk.0.ffn_gate.weight");
        if (gate_it == t.end() || gate_it->second.shape.size() != 2)
            throw std::invalid_argument("DeepSeek-V4 mmproj is missing v.blk.0.ffn_gate.weight");
        inter_ = static_cast<int>(gate_it->second.shape[1]);
        const std::size_t I = static_cast<std::size_t>(inter_);
        for (int i = 0; i < layers_; ++i)
        {
            const std::string pre = "v.blk." + std::to_string(i) + ".";
            Layer layer;
            layer.ln1 = take(t, pre + "ln1.weight", {D});
            std::tie(layer.qkv_w, layer.qkv_b) = takeQKV(t, pre, D);
            layer.out_w = takeMatrix(t, pre + "attn_out.weight", {D, D}, D);
            layer.out_b = take(t, pre + "attn_out.bias", {D});
            layer.ln2 = take(t, pre + "ln2.weight", {D});
            layer.gate_w = takeMatrix(t, pre + "ffn_gate.weight", {D, I}, I);
            layer.up_w = takeMatrix(t, pre + "ffn_up.weight", {D, I}, I);
            layer.down_w = takeMatrix(t, pre + "ffn_down.weight", {I, D}, D);
            blocks_.push_back(std::move(layer));
        }
        post_ln_ = take(t, "v.post_ln.weight", {D});
        const auto mm1 = t.find("mm.1.weight");
        const std::size_t merged = D * config_.merge * config_.merge;
        if (mm1 == t.end() || mm1->second.shape.size() != 2 || mm1->second.shape[0] != merged)
            throw std::invalid_argument("DeepSeek-V4 mmproj aligner mm.1.weight does not match the ViT width");
        text_dim_ = static_cast<int>(mm1->second.shape[1]);
        const std::size_t T = static_cast<std::size_t>(text_dim_);
        mm1_w_ = takeMatrix(t, "mm.1.weight", {merged, T}, T);
        mm1_b_ = take(t, "mm.1.bias", {T});
        mm2_w_ = takeMatrix(t, "mm.2.weight", {T, T}, T);
        mm2_b_ = take(t, "mm.2.bias", {T});
        sentinels_[static_cast<int>(ImageSentinel::Start)] = take(t, "v.token_embd.img_start", {T});
        sentinels_[static_cast<int>(ImageSentinel::Pad)] = take(t, "v.token_embd.img_pad", {T});
        sentinels_[static_cast<int>(ImageSentinel::NewLine)] = take(t, "v.image_newline", {T});
        sentinels_[static_cast<int>(ImageSentinel::End)] = take(t, "v.token_embd.img_end", {T});
        sentinels_[static_cast<int>(ImageSentinel::Image)].assign(T, 0.0f); // never read: IMAGE rows come from encode()
        if (!t.empty())
            throw std::invalid_argument("DeepSeek-V4 mmproj has unconsumed tensor " + t.begin()->first);

        // Resident bytes, by the same rule mmprojHostWeightBytes prices from
        // the tensor directory: matrices in file precision, vectors in FP32.
        const auto add = [&](const std::vector<float> &v) { weight_bytes_ += v.size() * sizeof(float); };
        const auto addm = [&](const VisionMatrix &m) { weight_bytes_ += m.bytes(); };
        for (const auto *v : {&patch_b_, &post_ln_, &mm1_b_, &mm2_b_}) add(*v);
        for (const auto *m : {&patch_w_, &mm1_w_, &mm2_w_}) addm(*m);
        for (const auto &b : blocks_)
        {
            for (const auto *v : {&b.ln1, &b.qkv_b, &b.out_b, &b.ln2}) add(*v);
            for (const auto *m : {&b.qkv_w, &b.out_w, &b.gate_w, &b.up_w, &b.down_w}) addm(*m);
        }
        for (const auto &s : sentinels_) add(s);
    }

    const std::vector<float> &DeepSeekV4VisionEncoder::sentinel(ImageSentinel role) const
    {
        if (role == ImageSentinel::Image)
            throw std::invalid_argument("IMAGE sentinels take encoded rows, not a learned vector");
        return sentinels_[static_cast<int>(role)];
    }

    std::vector<float> DeepSeekV4VisionEncoder::encode(const PreparedImage &img) const
    {
        const int N = img.vit_h * img.vit_w, D = dim_, H = heads_, hd = D / H, K = 3 * config_.patch * config_.patch;
        if (N <= 0 || img.patches.size() != static_cast<std::size_t>(N) * K)
            throw std::invalid_argument("DeepSeek-V4 vision input has inconsistent patch geometry");
        const std::size_t ND = static_cast<std::size_t>(N) * D;
        const std::size_t tile = std::min<std::size_t>(kVisionRowTile, static_cast<std::size_t>(N));
        // Full-width activations (attention needs every K/V row); the score and
        // MLP buffers cover one row tile only.
        std::vector<float> x(ND), a(ND), qkv(ND * 3), o(ND), scores(tile * static_cast<std::size_t>(N)),
            mlp_g(tile * static_cast<std::size_t>(inter_)), mlp_u(mlp_g.size()), widen;
        linear(img.patches.data(), N, K, patch_w_, patch_b_.data(), D, x.data(), widen);

        // 2D rotary tables: pair j of the rotated half (hd/2 pairs, rotate-half).
        const int half = hd / 2, quarter = hd / 4;
        std::vector<float> cos_t(static_cast<std::size_t>(N) * half), sin_t(cos_t.size());
        for (int n = 0; n < N; ++n)
            for (int j = 0; j < half; ++j)
            {
                const int fi = j % quarter;
                const float inv_freq = static_cast<float>(1.0 / std::pow(kRopeTheta, (2.0 * fi) / half));
                const float pos = static_cast<float>(j < quarter ? n / img.vit_w : n % img.vit_w);
                const float f = pos * inv_freq;
                cos_t[static_cast<std::size_t>(n) * half + j] = std::cos(f);
                sin_t[static_cast<std::size_t>(n) * half + j] = std::sin(f);
            }
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

        for (const auto &blk : blocks_)
        {
            rmsNorm(x.data(), N, D, blk.ln1, a.data());
            linear(a.data(), N, D, blk.qkv_w, blk.qkv_b.data(), 3 * D, qkv.data(), widen);
            // RoPE on q (columns [0, D)) and k ([D, 2D)).
            auto do_rope = [&]()
            {
#pragma omp for schedule(static)
                for (int n = 0; n < N; ++n)
                    for (int part = 0; part < 2; ++part)
                        for (int h = 0; h < H; ++h)
                        {
                            float *v = &qkv[static_cast<std::size_t>(n) * 3 * D + part * D + h * hd];
                            const float *c = &cos_t[static_cast<std::size_t>(n) * half];
                            const float *s = &sin_t[static_cast<std::size_t>(n) * half];
                            for (int j = 0; j < half; ++j)
                            {
                                const float x1 = v[j], x2 = v[j + half];
                                v[j] = x1 * c[j] - x2 * s[j];
                                v[j + half] = x2 * c[j] + x1 * s[j];
                            }
                        }
            };
            OMP_WORKSHARE_REGION(do_rope);
            for (int h = 0; h < H; ++h)
            {
                const float *k = &qkv[D + h * hd], *v = &qkv[2 * D + h * hd];
                // Query rows are independent: score one tile of rows against
                // every key, softmax each row exactly, then accumulate V.
                for (int r0 = 0; r0 < N; r0 += static_cast<int>(tile))
                {
                    const int rows = std::min(static_cast<int>(tile), N - r0);
                    const float *q = &qkv[static_cast<std::size_t>(r0) * 3 * D + h * hd];
                    sgemm('N', 'T', rows, N, hd, scale, q, 3 * D, k, 3 * D, scores.data(), N);
                    auto do_softmax = [&]()
                    {
#pragma omp for schedule(static)
                        for (int r = 0; r < rows; ++r)
                        {
                            float *row = &scores[static_cast<std::size_t>(r) * N];
                            const float m = *std::max_element(row, row + N);
                            double sum = 0.0;
                            for (int c = 0; c < N; ++c)
                            {
                                row[c] = std::exp(row[c] - m);
                                sum += row[c];
                            }
                            const float inv = static_cast<float>(1.0 / sum);
                            for (int c = 0; c < N; ++c) row[c] *= inv;
                        }
                    };
                    OMP_WORKSHARE_REGION(do_softmax);
                    sgemm('N', 'N', rows, hd, N, 1.0f, scores.data(), N, v, 3 * D,
                          &o[static_cast<std::size_t>(r0) * D + h * hd], D);
                }
            }
            linear(o.data(), N, D, blk.out_w, blk.out_b.data(), D, a.data(), widen);
            for (std::size_t i = 0; i < ND; ++i) x[i] += a[i];
            rmsNorm(x.data(), N, D, blk.ln2, a.data());
            // MLP rows are independent: gate/up/SwiGLU/down one row tile at a
            // time; o (free after the attention projection) receives down.
            for (int r0 = 0; r0 < N; r0 += static_cast<int>(tile))
            {
                const int rows = std::min(static_cast<int>(tile), N - r0);
                const float *ar = &a[static_cast<std::size_t>(r0) * D];
                linear(ar, rows, D, blk.gate_w, nullptr, inter_, mlp_g.data(), widen);
                linear(ar, rows, D, blk.up_w, nullptr, inter_, mlp_u.data(), widen);
                const std::size_t count = static_cast<std::size_t>(rows) * inter_;
                auto do_swiglu = [&]()
                {
#pragma omp for schedule(static)
                    for (std::size_t i = 0; i < count; ++i)
                    {
                        const float g = mlp_g[i];
                        mlp_g[i] = g / (1.0f + std::exp(-g)) * mlp_u[i];
                    }
                };
                OMP_WORKSHARE_REGION(do_swiglu);
                linear(mlp_g.data(), rows, inter_, blk.down_w, nullptr, D, &o[static_cast<std::size_t>(r0) * D], widen);
            }
            for (std::size_t i = 0; i < ND; ++i) x[i] += o[i];
        }
        rmsNorm(x.data(), N, D, post_ln_, a.data());

        // Aligner: 3x3 (merge x merge) unfold over the zero-padded grid, channel-major.
        const int r = config_.merge, LH = img.llm_h, LW = img.llm_w, U = D * r * r, L = LH * LW;
        if (LH != (img.vit_h + r - 1) / r || LW != (img.vit_w + r - 1) / r)
            throw std::invalid_argument("DeepSeek-V4 vision aligner grid disagrees with the patch grid");
        // The ViT activations are dead once `a` holds the final norm; release
        // them before the aligner buffers exist.
        std::vector<float>().swap(x);
        std::vector<float>().swap(qkv);
        std::vector<float>().swap(o);
        std::vector<float>().swap(scores);
        std::vector<float>().swap(mlp_g);
        std::vector<float>().swap(mlp_u);
        std::vector<float> unfolded(static_cast<std::size_t>(L) * U, 0.0f);
        auto do_unfold = [&]()
        {
#pragma omp for schedule(static)
            for (int cell = 0; cell < L; ++cell)
            {
                const int I = cell / LW, J = cell % LW;
                float *u = &unfolded[static_cast<std::size_t>(cell) * U];
                for (int ky = 0; ky < r; ++ky)
                    for (int kx = 0; kx < r; ++kx)
                    {
                        const int y = I * r + ky, xcol = J * r + kx;
                        if (y >= img.vit_h || xcol >= img.vit_w) continue;
                        const float *src = &a[(static_cast<std::size_t>(y) * img.vit_w + xcol) * D];
                        for (int c = 0; c < D; ++c) u[c * r * r + ky * r + kx] = src[c];
                    }
            }
        };
        OMP_WORKSHARE_REGION(do_unfold);
        std::vector<float>().swap(a);
        std::vector<float> h1(static_cast<std::size_t>(L) * text_dim_), h2(h1.size());
        linear(unfolded.data(), L, U, mm1_w_, mm1_b_.data(), text_dim_, h1.data(), widen);
        std::vector<float>().swap(unfolded);
        for (auto &v : h1) v = 0.5f * v * (1.0f + std::erf(v * static_cast<float>(M_SQRT1_2)));
        linear(h1.data(), L, text_dim_, mm2_w_, mm2_b_.data(), text_dim_, h2.data(), widen);
        std::vector<float>().swap(h1);

        if (img.perm.size() != static_cast<std::size_t>(L))
            throw std::invalid_argument("DeepSeek-V4 vision block order does not cover the aligner grid");
        std::vector<float> rows(h2.size());
        for (std::size_t slot = 0; slot < img.perm.size(); ++slot)
        {
            const int src = img.perm[slot];
            if (src < 0 || src >= L) throw std::invalid_argument("DeepSeek-V4 vision block order is out of range");
            std::copy_n(&h2[static_cast<std::size_t>(src) * text_dim_], text_dim_, &rows[slot * text_dim_]);
        }
        return rows;
    }

    std::unique_ptr<DeepSeekV4VisionEncoder> DeepSeekV4VisionEncoder::load(const std::string &path, const IMPIContext &mpi,
                                                                           int text_dim)
    {
        TensorFactory factory(mpi);
        ModelLoader loader(&factory);
        loader.setUseMmap(false);
        loader.loadModel(path);
        if (loader.architecture() != "clip" || loader.getString("clip.projector_type") != "deepseek4v")
            throw std::invalid_argument("--mmproj " + path + " is not a llama.cpp deepseek4v projector");
        VisionPreprocessConfig config;
        config.patch = loader.getInt("clip.vision.patch_size", 0);
        config.merge = loader.getInt("clip.vision.projector.scale_factor", 0);
        config.min_pixels = loader.getInt("clip.vision.image_min_pixels", 0);
        const int layers = loader.getInt("clip.vision.block_count", 0);
        const int heads = loader.getInt("clip.vision.attention.head_count", 0);
        if (config.patch <= 0 || config.merge <= 0 || config.min_pixels <= 0 || layers <= 0 || heads <= 0)
            throw std::invalid_argument("DeepSeek-V4 mmproj lacks its clip.vision geometry");
        std::map<std::string, VisionHostTensor> tensors;
        for (const auto &name : loader.tensorNames())
        {
            auto tensor = loader.loadTensor(name, DeviceId::cpu(), WeightPrecision::NATIVE);
            if (!tensor) throw std::invalid_argument("DeepSeek-V4 mmproj tensor " + name + " could not be read");
            VisionHostTensor host;
            // The encoder uses ggml order (ne0 first). The loader swaps only
            // 2D GGUF dimensions into [rows, cols]; other ranks keep ggml order.
            const auto &shape = tensor->shape();
            if (shape.size() == 2) host.shape.assign(shape.rbegin(), shape.rend());
            else host.shape.assign(shape.begin(), shape.end());
            // Residency rule (priced identically by mmprojHostWeightBytes):
            // rank >= 2 BF16/F16 matrices keep their exact 16-bit values;
            // every other tensor is widened to FP32 once.
            const auto *bf16 = shape.size() >= 2 ? dynamic_cast<const BF16Tensor *>(tensor.get()) : nullptr;
            const auto *f16 = shape.size() >= 2 ? dynamic_cast<const FP16Tensor *>(tensor.get()) : nullptr;
            if (bf16 || f16)
            {
                const std::uint16_t *bits = bf16 ? bf16->typed_data() : f16->typed_data();
                host.half_bits.assign(bits, bits + tensor->numel());
                host.precision = bf16 ? VisionHostPrecision::BF16 : VisionHostPrecision::F16;
            }
            else
            {
                host.data.resize(tensor->numel());
                tensor->to_fp32(host.data.data());
            }
            tensor.reset();
            tensors.emplace(name, std::move(host));
        }
        auto encoder = std::make_unique<DeepSeekV4VisionEncoder>(std::move(tensors), config, layers, heads);
        if (encoder->textDim() != text_dim)
            throw std::invalid_argument("DeepSeek-V4 mmproj projects to width " + std::to_string(encoder->textDim()) +
                                        " but the text model is " + std::to_string(text_dim));
        return encoder;
    }

    std::size_t DeepSeekV4VisionEncoder::mmprojHostWeightBytes(const std::string &path, const IMPIContext &mpi)
    {
        TensorFactory factory(mpi);
        ModelLoader loader(&factory);
        loader.setUseMmap(false);
        if (!loader.loadModel(path))
            throw std::runtime_error("Cannot read DeepSeek-V4 mmproj metadata from " + path);
        loader.sealTensorSources();
        // Residency rule of load(): rank >= 2 BF16/F16 tensors stay 16-bit,
        // every other tensor (vectors, F32 or quantized matrices) is FP32.
        std::size_t total = 0;
        for (const auto &info : loader.getModel().tensors)
        {
            std::size_t numel = 1;
            for (const auto d : info.dimensions)
            {
                if (d != 0 && numel > SIZE_MAX / d)
                    throw std::overflow_error("DeepSeek-V4 mmproj tensor element count overflow");
                numel *= static_cast<std::size_t>(d);
            }
            const bool half = info.dimensions.size() >= 2 &&
                (info.type == GGUFTensorType::BF16 || info.type == GGUFTensorType::F16);
            const std::size_t bytes = numel * (half ? sizeof(std::uint16_t) : sizeof(float));
            if (bytes / (half ? sizeof(std::uint16_t) : sizeof(float)) != numel || bytes > SIZE_MAX - total)
                throw std::overflow_error("DeepSeek-V4 mmproj host byte sum overflow");
            total += bytes;
        }
        // The tower also retains its synthetic IMAGE sentinel row (T floats),
        // which is not a file tensor (see the encoder constructor).
        const auto mm1_shape = loader.getTensorShape("mm.1.weight");
        if (mm1_shape && mm1_shape->size() == 2)
            total += (*mm1_shape)[0] * sizeof(float);
        return total;
    }

    /**
     * @brief Shared worst-case host request workspace (bytes); single authority.
     *
     * Mirrors the lifetimes of DeepSeekV4PromptImageEncoder::expand and encode():
     * images are processed one at a time, so the transient term is the larger of
     * one image's decode/prepare phase and its encode phase; only each image's
     * encoded rows and block/perm vectors are retained across the request.
     */
    std::size_t visionTowerWorkspaceFor(std::size_t D, std::size_t I, std::size_t T, std::size_t merge,
                                        std::size_t patch, std::size_t heads)
    {
        if (!D || !I || !T || !merge || !patch || !heads || D % heads)
            throw std::invalid_argument("DeepSeek-V4 vision workspace geometry is degenerate");
        const std::size_t L = static_cast<std::size_t>(kVisionMaxTokens); // aligner rows
        const std::size_t N = L * merge * merge;                           // ViT patch rows
        const std::size_t K = 3u * patch * patch;                          // patch width
        const std::size_t U = D * merge * merge;                           // unfolded width
        const std::size_t hd = D / heads;
        const std::size_t tile = std::min<std::size_t>(kVisionRowTile, N);
        const std::size_t P = static_cast<std::size_t>(kHostDecodedPixelBound);
        constexpr std::size_t f = sizeof(float);

        // Widening scratch: the largest chunk any projection uses.
        std::size_t widen = 0;
        for (const std::size_t k : {K, D, I, U, T})
            widen = std::max(widen, widenChunkRows(k) * k);

        // Decode/prepare phase (bytes): stb's buffer and its RgbImage copy
        // (2 x 3P); then the decoded image, one resize intermediate (<= 3P),
        // the resized/padded pixels and the padded copy (each <= N*K bytes),
        // and finally the FP32 patches beside the decoded image and pixels.
        const std::size_t prepare = std::max({2u * 3u * P, 3u * P + 3u * P + 2u * N * K,
                                              3u * P + N * K + f * N * K});
        // Encode phase (bytes): patches, widening scratch, RoPE tables and the
        // larger of the ViT working set and the aligner working set.
        const std::size_t vit = 6u * N * D + tile * N + 2u * tile * I;
        const std::size_t aligner = std::max(N * D + L * U, L * U + 2u * L * T);
        const std::size_t encode = f * (N * K + widen + N * hd + std::max(vit, aligner));
        // Retained per image: encoded rows plus block/perm order.
        const std::size_t retained = f * L * T + sizeof(std::int32_t) *
            (static_cast<std::size_t>(maxImageSlotsPerImage(kVisionMaxTokens)) + L);
        return retained * static_cast<std::size_t>(kMaxImagesPerRequest) + std::max(prepare, encode);
    }

    std::size_t DeepSeekV4VisionEncoder::encodeWorkspaceBytes() const noexcept
    {
        return visionTowerWorkspaceFor(static_cast<std::size_t>(dim_),
            static_cast<std::size_t>(inter_), static_cast<std::size_t>(text_dim_),
            static_cast<std::size_t>(config_.merge), static_cast<std::size_t>(config_.patch),
            static_cast<std::size_t>(heads_));
    }

    std::size_t visionTowerWorkspaceBytes(const std::string &path, const IMPIContext &mpi, int text_dim)
    {
        TensorFactory factory(mpi);
        ModelLoader loader(&factory);
        loader.setUseMmap(false);
        if (!loader.loadModel(path))
            throw std::runtime_error("Cannot read DeepSeek-V4 mmproj metadata from " + path);
        loader.sealTensorSources();
        // Block tensors are named per layer (v.<block>.attn_qkv.weight); the
        // first matching directory entry carries the shared width geometry.
        const auto find2D = [&](const std::string &needle) -> std::optional<std::vector<size_t>>
        {
            for (const auto &name : loader.tensorNames())
                if (name.find(needle) != std::string::npos)
                    return loader.getTensorShape(name);
            return std::nullopt;
        };
        // Mirror the encoder's exact derivations: dim_ from the patch embedding's
        // last axis, inter_ from ffn_gate's second axis, text_dim_ from mm.1's
        // second axis (see DeepSeekV4VisionEncoder constructor).
        const auto patch_shape = find2D("patch_embd.weight");
        const auto gate_shape = find2D("ffn_gate.weight");
        const auto mm1_shape = loader.getTensorShape("mm.1.weight");
        const auto mm2_shape = loader.getTensorShape("mm.2.weight");
        const std::size_t D = patch_shape && patch_shape->size() == 4 ? (*patch_shape)[3] : 0;
        // The loader reports 2-D shapes in file (ggml) order; the encoder swaps
        // them to host order, so the host second axis is the file first axis.
        const std::size_t I = gate_shape && gate_shape->size() == 2 ? (*gate_shape)[0] : 0;
        const std::size_t T = mm1_shape && mm1_shape->size() == 2 ? (*mm1_shape)[0] : 0;
        (void)mm2_shape;
        const std::int64_t merge_meta = loader.getInt("clip.vision.projector.scale_factor", kVisionDefaultMerge);
        const std::size_t merge = merge_meta > 0 ? static_cast<std::size_t>(merge_meta)
                                                 : static_cast<std::size_t>(kVisionDefaultMerge);
        const std::int64_t patch = loader.getInt("clip.vision.patch_size", 0);
        const std::int64_t heads = loader.getInt("clip.vision.attention.head_count", 0);
        if (D == 0 || I == 0 || T == 0 || merge == 0 || patch <= 0 || heads <= 0)
            throw std::invalid_argument("DeepSeek-V4 mmproj lacks the geometry for host workspace admission");
        if (text_dim > 0 && T != static_cast<std::size_t>(text_dim))
            throw std::invalid_argument("DeepSeek-V4 mmproj projects to width " + std::to_string(T) +
                                        " but the text model is " + std::to_string(text_dim));
        return visionTowerWorkspaceFor(D, I, T, merge, static_cast<std::size_t>(patch), static_cast<std::size_t>(heads));
    }
} // namespace llaminar2::deepseek4
