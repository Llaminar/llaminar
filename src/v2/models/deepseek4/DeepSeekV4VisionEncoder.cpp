/**
 * @file DeepSeekV4VisionEncoder.cpp
 * @brief FP32 BLAS implementation of the official vision.py tower.
 *
 * Matrices are row-major [out, in] (a GGUF tensor with ne = {in, out}).
 * Every projection is one oneDNN ``dnnl_sgemm`` (row-major) ``Y = X . W^T (+ b)``. Attention
 * materializes one head's [N, N] score matrix at a time (N <= the admitted
 * patch capacity), applies an exact max-subtracted softmax per row and
 * multiplies by V; heads are written to disjoint output columns.
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
#include <cmath>
#include <stdexcept>

namespace llaminar2::deepseek4
{
    namespace
    {
        constexpr float kNormEps = 1e-6f;
        constexpr double kRopeTheta = 10000.0;

        std::vector<float> take(std::map<std::string, VisionHostTensor> &t, const std::string &name,
                                std::initializer_list<std::size_t> shape)
        {
            auto it = t.find(name);
            if (it == t.end())
                throw std::invalid_argument("DeepSeek-V4 mmproj is missing " + name);
            if (!std::equal(shape.begin(), shape.end(), it->second.shape.begin(), it->second.shape.end()))
                throw std::invalid_argument("DeepSeek-V4 mmproj tensor " + name + " has an unexpected shape");
            auto out = std::move(it->second.data);
            t.erase(it);
            return out;
        }

        /** @brief Row-major C[m, n] = alpha * op(A) . op(B); fatal on a oneDNN error. */
        void sgemm(char ta, char tb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb,
                   float *c, int ldc)
        {
            if (dnnl_sgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, 0.0f, c, ldc) != dnnl_success)
                throw std::runtime_error("DeepSeek-V4 vision tower: dnnl_sgemm failed");
        }

        /** @brief Y[m, n] = X[m, k] . W[n, k]^T (+ bias[n]). */
        void linear(const float *x, int m, int k, const std::vector<float> &w, const float *bias, int n, float *y)
        {
            sgemm('N', 'T', m, n, k, 1.0f, x, k, w.data(), k, y, n);
            if (bias)
#pragma omp parallel for schedule(static)
                for (int r = 0; r < m; ++r)
                    for (int c = 0; c < n; ++c) y[static_cast<std::size_t>(r) * n + c] += bias[c];
        }

        void rmsNorm(const float *x, int rows, int dim, const std::vector<float> &gain, float *y)
        {
#pragma omp parallel for schedule(static)
            for (int r = 0; r < rows; ++r)
            {
                const float *in = x + static_cast<std::size_t>(r) * dim;
                float *out = y + static_cast<std::size_t>(r) * dim;
                double ss = 0.0;
                for (int c = 0; c < dim; ++c) ss += static_cast<double>(in[c]) * in[c];
                const float inv = 1.0f / std::sqrt(static_cast<float>(ss / dim) + kNormEps);
                for (int c = 0; c < dim; ++c) out[c] = gain[c] * (in[c] * inv);
            }
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
        patch_w_ = take(t, "v.patch_embd.weight", {p, p, 3, D});
        patch_b_ = take(t, "v.patch_embd.bias", {D});
        const auto gate_it = t.find("v.blk.0.ffn_gate.weight");
        if (gate_it == t.end() || gate_it->second.shape.size() != 2)
            throw std::invalid_argument("DeepSeek-V4 mmproj is missing v.blk.0.ffn_gate.weight");
        inter_ = static_cast<int>(gate_it->second.shape[1]);
        const std::size_t I = static_cast<std::size_t>(inter_);
        for (int i = 0; i < layers_; ++i)
        {
            const std::string pre = "v.blk." + std::to_string(i) + ".";
            blocks_.push_back({take(t, pre + "ln1.weight", {D}), take(t, pre + "attn_qkv.weight", {D, 3 * D}),
                               take(t, pre + "attn_qkv.bias", {3 * D}), take(t, pre + "attn_out.weight", {D, D}),
                               take(t, pre + "attn_out.bias", {D}), take(t, pre + "ln2.weight", {D}),
                               take(t, pre + "ffn_gate.weight", {D, I}), take(t, pre + "ffn_up.weight", {D, I}),
                               take(t, pre + "ffn_down.weight", {I, D})});
        }
        post_ln_ = take(t, "v.post_ln.weight", {D});
        const auto mm1 = t.find("mm.1.weight");
        const std::size_t merged = D * config_.merge * config_.merge;
        if (mm1 == t.end() || mm1->second.shape.size() != 2 || mm1->second.shape[0] != merged)
            throw std::invalid_argument("DeepSeek-V4 mmproj aligner mm.1.weight does not match the ViT width");
        text_dim_ = static_cast<int>(mm1->second.shape[1]);
        const std::size_t T = static_cast<std::size_t>(text_dim_);
        mm1_w_ = take(t, "mm.1.weight", {merged, T});
        mm1_b_ = take(t, "mm.1.bias", {T});
        mm2_w_ = take(t, "mm.2.weight", {T, T});
        mm2_b_ = take(t, "mm.2.bias", {T});
        sentinels_[static_cast<int>(ImageSentinel::Start)] = take(t, "v.token_embd.img_start", {T});
        sentinels_[static_cast<int>(ImageSentinel::Pad)] = take(t, "v.token_embd.img_pad", {T});
        sentinels_[static_cast<int>(ImageSentinel::NewLine)] = take(t, "v.image_newline", {T});
        sentinels_[static_cast<int>(ImageSentinel::End)] = take(t, "v.token_embd.img_end", {T});
        sentinels_[static_cast<int>(ImageSentinel::Image)].assign(T, 0.0f); // never read: IMAGE rows come from encode()
        if (!t.empty())
            throw std::invalid_argument("DeepSeek-V4 mmproj has unconsumed tensor " + t.begin()->first);

        const auto add = [&](const std::vector<float> &v) { weight_bytes_ += v.size() * sizeof(float); };
        for (const auto *v : {&patch_w_, &patch_b_, &post_ln_, &mm1_w_, &mm1_b_, &mm2_w_, &mm2_b_}) add(*v);
        for (const auto &b : blocks_)
            for (const auto *v : {&b.ln1, &b.qkv_w, &b.qkv_b, &b.out_w, &b.out_b, &b.ln2, &b.gate_w, &b.up_w, &b.down_w}) add(*v);
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
        std::vector<float> x(ND), a(ND), qkv(ND * 3), o(ND), mlp_g(static_cast<std::size_t>(N) * inter_),
            mlp_u(mlp_g.size()), scores(static_cast<std::size_t>(N) * N);
        linear(img.patches.data(), N, K, patch_w_, patch_b_.data(), D, x.data());

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
            linear(a.data(), N, D, blk.qkv_w, blk.qkv_b.data(), 3 * D, qkv.data());
            // RoPE on q (columns [0, D)) and k ([D, 2D)).
#pragma omp parallel for schedule(static)
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
            for (int h = 0; h < H; ++h)
            {
                const float *q = &qkv[h * hd], *k = &qkv[D + h * hd], *v = &qkv[2 * D + h * hd];
                sgemm('N', 'T', N, N, hd, scale, q, 3 * D, k, 3 * D, scores.data(), N);
#pragma omp parallel for schedule(static)
                for (int r = 0; r < N; ++r)
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
                sgemm('N', 'N', N, hd, N, 1.0f, scores.data(), N, v, 3 * D, &o[h * hd], D);
            }
            linear(o.data(), N, D, blk.out_w, blk.out_b.data(), D, a.data());
            for (std::size_t i = 0; i < ND; ++i) x[i] += a[i];
            rmsNorm(x.data(), N, D, blk.ln2, a.data());
            linear(a.data(), N, D, blk.gate_w, nullptr, inter_, mlp_g.data());
            linear(a.data(), N, D, blk.up_w, nullptr, inter_, mlp_u.data());
#pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < mlp_g.size(); ++i)
            {
                const float g = mlp_g[i];
                mlp_g[i] = g / (1.0f + std::exp(-g)) * mlp_u[i];
            }
            linear(mlp_g.data(), N, inter_, blk.down_w, nullptr, D, a.data());
            for (std::size_t i = 0; i < ND; ++i) x[i] += a[i];
        }
        rmsNorm(x.data(), N, D, post_ln_, a.data());

        // Aligner: 3x3 (merge x merge) unfold over the zero-padded grid, channel-major.
        const int r = config_.merge, LH = img.llm_h, LW = img.llm_w, U = D * r * r, L = LH * LW;
        if (LH != (img.vit_h + r - 1) / r || LW != (img.vit_w + r - 1) / r)
            throw std::invalid_argument("DeepSeek-V4 vision aligner grid disagrees with the patch grid");
        std::vector<float> unfolded(static_cast<std::size_t>(L) * U, 0.0f);
#pragma omp parallel for schedule(static)
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
        std::vector<float> h1(static_cast<std::size_t>(L) * text_dim_), h2(h1.size());
        linear(unfolded.data(), L, U, mm1_w_, mm1_b_.data(), text_dim_, h1.data());
        for (auto &v : h1) v = 0.5f * v * (1.0f + std::erf(v * static_cast<float>(M_SQRT1_2)));
        linear(h1.data(), L, text_dim_, mm2_w_, mm2_b_.data(), text_dim_, h2.data());

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
            host.data.resize(tensor->numel());
            tensor->to_fp32(host.data.data());
            tensors.emplace(name, std::move(host));
        }
        auto encoder = std::make_unique<DeepSeekV4VisionEncoder>(std::move(tensors), config, layers, heads);
        if (encoder->textDim() != text_dim)
            throw std::invalid_argument("DeepSeek-V4 mmproj projects to width " + std::to_string(encoder->textDim()) +
                                        " but the text model is " + std::to_string(text_dim));
        return encoder;
    }
} // namespace llaminar2::deepseek4
