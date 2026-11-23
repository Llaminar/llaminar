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
 * Rows are returned in block (``perm``) order.
 *
 * Memory: projection matrices stay resident in their file precision (BF16/F16
 * halves the FP32 footprint) and are widened exactly into a bounded scratch
 * one output-row chunk at a time; vectors (norm gains, biases, sentinels) are
 * widened once. Attention scores and the MLP are processed in row tiles, so
 * the per-image workspace never materializes an N x N or N x I matrix. Every
 * output row is computed by the same FP32 arithmetic on exactly widened
 * weights; only the BLAS blocking of the tiles differs.
 */
#pragma once

#include "DeepSeekV4ImagePreprocess.h"

#include <array>
#include <cstdint>
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
    /** @brief Resident precision of a host vision tensor. */
    enum class VisionHostPrecision : std::uint8_t
    {
        F32,  ///< `data` holds FP32 values.
        BF16, ///< `half_bits` holds bfloat16 bit patterns (widen exactly).
        F16,  ///< `half_bits` holds IEEE binary16 bit patterns (widen exactly).
    };

    /** @brief One named host tensor in ggml dimension order (ne0 first). */
    struct VisionHostTensor
    {
        std::vector<std::size_t> shape;
        std::vector<float> data;               ///< FP32 storage (precision F32).
        std::vector<std::uint16_t> half_bits;  ///< 16-bit storage (precision BF16/F16).
        VisionHostPrecision precision = VisionHostPrecision::F32;
    };

    /**
     * @brief Row-major [rows = out, cols = in] projection matrix in its resident precision.
     *
     * A 16-bit matrix is widened exactly (BF16/F16 -> FP32 is lossless) into
     * caller scratch one row chunk at a time; it is never held as FP32.
     */
    struct VisionMatrix
    {
        int rows = 0, cols = 0;
        VisionHostPrecision precision = VisionHostPrecision::F32;
        std::vector<float> f32;
        std::vector<std::uint16_t> half_bits;

        /** @return Resident bytes. */
        [[nodiscard]] std::size_t bytes() const noexcept
        {
            return f32.size() * sizeof(float) + half_bits.size() * sizeof(std::uint16_t);
        }

        /** @return Physically reserved bytes (vector capacity), never below bytes(). */
        [[nodiscard]] std::size_t capacityBytes() const noexcept
        {
            return f32.capacity() * sizeof(float) + half_bits.capacity() * sizeof(std::uint16_t);
        }
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

        /** @return Resident bytes held by the tower's weights. */
        std::size_t weightBytes() const noexcept { return weight_bytes_; }
        /** @return Physically reserved tower bytes (vector capacities); equals weightBytes() when no slack exists. */
        std::size_t physicalWeightBytes() const noexcept { return physical_weight_bytes_; }

        /**
         * @brief Exact resident host weight bytes of a deepseek4v mmproj, metadata only.
         *
         * Reads only the GGUF header/tensor directory (no payload) and applies
         * the residency rule of @ref load: rank >= 2 BF16/F16 tensors keep 2
         * bytes per value, every other tensor is widened to FP32, plus the
         * synthetic IMAGE sentinel row.
         */
        static std::size_t mmprojHostWeightBytes(const std::string &path, const IMPIContext &mpi);

        /**
         * @brief Worst-case host workspace of one request (decode + prepare + encode), bytes.
         *
         * Images are decoded, prepared and encoded one at a time, so the
         * transient terms are one image's worth; the retained terms (each
         * image's prepared geometry/pixels and encoded rows) scale with
         * kMaxImagesPerRequest.
         */
        std::size_t encodeWorkspaceBytes() const noexcept;

    private:
        struct Layer
        {
            std::vector<float> ln1, qkv_b, out_b, ln2;
            VisionMatrix qkv_w, out_w, gate_w, up_w, down_w;
        };
        VisionPreprocessConfig config_;
        int layers_ = 0, heads_ = 0, dim_ = 0, inter_ = 0, text_dim_ = 0;
        std::vector<float> patch_b_, post_ln_, mm1_b_, mm2_b_;
        VisionMatrix patch_w_, mm1_w_, mm2_w_;
        std::vector<Layer> blocks_;
        std::array<std::vector<float>, kImageSentinelCount> sentinels_;
        std::size_t weight_bytes_ = 0;
        std::size_t physical_weight_bytes_ = 0;
    };

    /**
     * @brief Metadata-derived worst-case host request workspace (bytes) for admission.
     *
     * Reads only the mmproj header/tensor directory; equals
     * `DeepSeekV4VisionEncoder::encodeWorkspaceBytes()` of the loaded tower.
     * @param path llama.cpp deepseek4v mmproj.
     * @param mpi Loader rank context.
     * @param text_dim Text-model width; the projector must match (0 skips the check).
     */
    std::size_t visionTowerWorkspaceBytes(const std::string &path, const IMPIContext &mpi, int text_dim);
} // namespace llaminar2::deepseek4
