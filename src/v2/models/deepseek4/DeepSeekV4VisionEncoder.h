/**
 * @file DeepSeekV4VisionEncoder.h
 * @brief Host DeepSeek-V4 vision tower (ViT + aligner) over a deepseek4v mmproj.
 *
 * Placement: the tower runs on the host CPU in FP32 with BLAS, the analogue of
 * production vLLM's UVA offload of ``vision``/``aligner``; its outputs are
 * request input to the text model (uploaded once into the image bank at
 * admission). Mathematics follows the official ``inference/vision.py``:
 *  - patch embedding: linear over flattened (3, p, p) patches with bias;
 *  - each block: x += Wo . MHA(RMSNorm(x)) + bo, then
 *    x += Wd . (silu(Wg . RMSNorm(x)) * Wu . RMSNorm(x)); RMSNorm eps 1e-6;
 *  - attention: bidirectional over all patches of the image, 2D RoPE in the
 *    rotate-half convention: rotary pair j < hd/4 uses the patch row,
 *    j >= hd/4 the column, inv_freq = theta^(-2j / (hd/2)), theta 10000;
 *  - final RMSNorm;
 *  - aligner: zero-pad the grid to a multiple of ``merge``, 3x3 unfold
 *    (channel-major, ``F.unfold``), W2 . gelu_erf(W1 . u + b1) + b2.
 * Rows are returned in block (``perm``) order. Weights are FP32 host copies
 * of the GGUF tensors (BF16/F16 mmproj values widen exactly).
 */
#pragma once

#include "DeepSeekV4ImagePreprocess.h"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{
    class IMPIContext;
}

namespace llaminar2::deepseek4
{
    /** @brief One named FP32 host tensor in ggml dimension order (ne0 first). */
    struct VisionHostTensor
    {
        std::vector<std::size_t> shape;
        std::vector<float> data;
    };

    /** @brief Host vision tower; immutable after construction, safe for concurrent encode(). */
    class DeepSeekV4VisionEncoder
    {
    public:
        /**
         * @brief Build from named host tensors (llama.cpp deepseek4v names).
         * @param tensors ``v.*`` / ``mm.*`` tensors, consumed.
         * @param config Preprocessing geometry (patch, merge, min pixels).
         * @param layers Number of ViT blocks; heads ViT attention heads.
         * @throws std::invalid_argument On missing tensors or inconsistent shapes.
         */
        DeepSeekV4VisionEncoder(std::map<std::string, VisionHostTensor> tensors, VisionPreprocessConfig config,
                                int layers, int heads);

        /**
         * @brief Load a llama.cpp ``deepseek4v`` mmproj GGUF.
         * @param path mmproj file.
         * @param mpi Rank context for the loader's tensor factory.
         * @param text_dim Hidden size of the text model; the projector must match it.
         * @throws std::invalid_argument When the file is not a deepseek4v projector for this model.
         */
        static std::unique_ptr<DeepSeekV4VisionEncoder> load(const std::string &path, const IMPIContext &mpi, int text_dim);

        /** @return Preprocessing geometry of this projector. */
        const VisionPreprocessConfig &preprocessConfig() const noexcept { return config_; }

        /** @return Output (text model) width. */
        int textDim() const noexcept { return text_dim_; }

        /**
         * @brief Encode one image.
         * @return [perm.size(), textDim()] aligner outputs in block order.
         */
        std::vector<float> encode(const PreparedImage &image) const;

        /** @return Learned vector of a non-IMAGE sentinel role (START, PAD, NEWLINE, END). */
        const std::vector<float> &sentinel(ImageSentinel role) const;

        /** @return FP32 bytes held by the tower's weights. */
        std::size_t weightBytes() const noexcept { return weight_bytes_; }

    private:
        struct Layer
        {
            std::vector<float> ln1, qkv_w, qkv_b, out_w, out_b, ln2, gate_w, up_w, down_w;
        };
        VisionPreprocessConfig config_;
        int layers_ = 0, heads_ = 0, dim_ = 0, inter_ = 0, text_dim_ = 0;
        std::vector<float> patch_w_, patch_b_, post_ln_, mm1_w_, mm1_b_, mm2_w_, mm2_b_;
        std::vector<Layer> blocks_;
        std::array<std::vector<float>, kImageSentinelCount> sentinels_;
        std::size_t weight_bytes_ = 0;
    };
} // namespace llaminar2::deepseek4
