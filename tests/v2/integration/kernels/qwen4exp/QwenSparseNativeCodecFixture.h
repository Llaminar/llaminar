/**
 * @file QwenSparseNativeCodecFixture.h
 * @brief Independent serialized KV fixtures for selected-index codec conformance.
 *
 * Known integer codes are packed directly into the canonical block structures.
 * Spatial reference values are retained separately, including a nontrivial
 * orthogonal Householder rotation for TQ. Tests therefore catch both bit-layout
 * mistakes and treating rotated centroids as original attention coordinates.
 * Canonical consumer-local slices copy whole original native head blocks with
 * their exact local rotations/anchors; no codec conversion or cache expansion
 * is used to prove logical degrees1..8 on a single CPU/GPU fixture.
 */
#pragma once
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "config/TensorParallelConfig.h"
#include "config/QueryHeadPartitionDeclaration.h"
#include <cstring>
#include "kernels/qwen4exp/QwenSparseMath.h"
#include "kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "tensors/BlockStructures.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace llaminar2::qwen4exp::test
{
/** @brief Independently serialized bank and its original-coordinate FP32 reference. */
struct NativeCodecFixture
{
    TensorType codec;
    int tokens, block_size;
    std::size_t head_bytes;
    std::vector<unsigned char> storage;
    std::vector<float> decoded, decoded_key, anchor, rotation, centroids;
    /** @brief Pack one complete supported native bank with independent original-basis values. */
    NativeCodecFixture(TensorType type, int token_count, int q16_size = 32, bool reference_values = true)
        : codec(type), tokens(token_count), block_size(q16_size), head_bytes(
              type == TensorType::FP32 ? 1024 : type == TensorType::FP16 || type == TensorType::BF16 ? 512 :
              type == TensorType::Q8_1 ? 288 : type == TensorType::Q16_1 ? (256 / q16_size) * (8 + 2 * q16_size) :
              type == TensorType::AQ8 ? 260 : type == TensorType::TQ4 ? 136 : 264),
          storage(std::size_t(tokens) * 2 * head_bytes, 0), decoded(reference_values ? std::size_t(tokens) * 512 : 0), decoded_key(decoded.size()), anchor(512), rotation(2 * 65536),
          centroids(type == TensorType::TQ4 ? std::vector<float>(TQ4_CENTROIDS.begin(), TQ4_CENTROIDS.end()) :
                    std::vector<float>(TQ8_CENTROIDS.begin(), TQ8_CENTROIDS.end()))
    {
        for (int i = 0; i < 512; ++i) anchor[i] = float((i % 17) - 8) / 32.0f;
        for (int head = 0; head < 2; ++head)
        for (int row = 0; row < 256; ++row)
            for (int column = 0; column < 256; ++column)
                rotation[head * 65536 + row * 256 + column] = (row == column ? 1.0f : 0.0f) -
                    ((row % (head == 0 ? 3 : 5) == 0) == (column % (head == 0 ? 3 : 5) == 0) ? 1.0f : -1.0f) / 128.0f;
        for (int token = 0; token < tokens; ++token)
        for (int head = 0; head < 2; ++head)
        {
            auto *bytes = storage.data() + (std::size_t(token) * 2 + head) * head_bytes;
            float basis[256];
            for (int d = 0; d < 256; ++d)
            {
                const int code = (token * 29 + head * 13 + d * 7) % 127 - 63;
                float result = float(code) / 64.0f;
                if (type == TensorType::FP32) reinterpret_cast<float *>(bytes)[d] = result;
                else if (type == TensorType::FP16)
                    reinterpret_cast<std::uint16_t *>(bytes)[d] = detail::encodeHalf(result);
                else if (type == TensorType::BF16)
                    reinterpret_cast<std::uint16_t *>(bytes)[d] = std::uint16_t(detail::floatBits(result) >> 16);
                else if (type == TensorType::Q8_1)
                {
                    auto &block = reinterpret_cast<Q8_1Block *>(bytes)[d / 32];
                    block.d = detail::encodeHalf(1.0f / 64.0f); block.qs[d % 32] = static_cast<std::int8_t>(code);
                }
                else if (type == TensorType::Q16_1)
                {
                    auto *block = bytes + std::size_t(d / block_size) * (8 + 2 * block_size);
                    *reinterpret_cast<float *>(block) = 1.0f / 4096.0f;
                    reinterpret_cast<std::int16_t *>(block + 8)[d % block_size] = static_cast<std::int16_t>(code * 64);
                }
                else if (type == TensorType::AQ8)
                {
                    auto &block = *reinterpret_cast<AttentionKeyQ8Block<256> *>(bytes);
                    block.quadratic_scale = 1.0f / 4096.0f; block.codes[d] = static_cast<std::int8_t>(code);
                    result = (float(code) / 4096.0f) * float(std::abs(code)) + anchor[head * 256 + d];
                }
                else
                {
                    const unsigned index = unsigned(token * 19 + head * 23 + d * 11) % (type == TensorType::TQ4 ? 16u : 256u);
                    if (type == TensorType::TQ8)
                    {
                        auto &block = *reinterpret_cast<TQ8Block<256> *>(bytes);
                        block.norm = 16.0f; block.reconstruction_norm = 8.0f; block.indices[d] = static_cast<std::uint8_t>(index);
                    }
                    else
                    {
                        auto &block = *reinterpret_cast<TQ4Block<256> *>(bytes);
                        block.norm = 16.0f; block.reconstruction_norm = 8.0f;
                        for (int bit = 0; bit < 3; ++bit)
                            if (index & (1u << bit)) block.mse_indices[(d * 3 + bit) / 8] |= std::uint8_t(1u << ((d * 3 + bit) % 8));
                        if (index & 8u) block.high_bits[d / 8] |= std::uint8_t(1u << (d % 8));
                    }
                    result = centroids[index] * 0.5f;
                }
                basis[d] = result;
            }
            for (int d = 0; reference_values && d < 256; ++d)
            {
                float result = basis[d];
                if (type == TensorType::TQ4 || type == TensorType::TQ8)
                {
                    result = 0.0f;
                    for (int i = 0; i < 256; ++i) result = std::fma(rotation[head * 65536 + i * 256 + d], basis[i], result);
                }
                decoded[(std::size_t(token) * 2 + head) * 256 + d] = result;
                decoded_key[(std::size_t(token) * 2 + head) * 256 + d] =
                    type == TensorType::TQ4 || type == TensorType::TQ8 ? result * 2.0f : result;
            }
        }
    }
    /** @brief Bind native bytes and canonical table pointers without preparing a shadow. */
    QwenSparseNativeBank bank() const
    {
        return {.data = storage.data(), .token_stride_bytes = 2 * head_bytes, .head_stride_bytes = head_bytes,
            .codec = codec, .q16_block_size = block_size, .centroids = centroids.data(), .rotations = {rotation.data(), rotation.data() + 65536}, .anchor = anchor.data()};
    }
    /** @brief Bind the independent spatial oracle using the same physical token order. */
    QwenSparseNativeBank reference(bool key = false) const
    {
        return {.data = key ? decoded_key.data() : decoded.data(), .token_stride_bytes = 512 * sizeof(float), .head_stride_bytes = 256 * sizeof(float), .codec = TensorType::FP32};
    }
};
/**
 * @brief Exact compact native bytes for one canonical original-query consumer.
 *
 * This fixture copies whole serialized head blocks from NativeCodecFixture;
 * it never decodes/requantizes a key or changes its original rotation/anchor.
 * The canonical assignment owns geometry. Tests execute logical slices on one
 * CPU/GPU and do not certify an eight-device model or planner/cache adoption.
 */
struct ConsumerLocalCodecFixture
{
    const NativeCodecFixture &source;
    const int first_head,head_count;
    std::vector<unsigned char> storage;
    std::vector<float> rotation,anchor;
    /**
     * @brief Retain the exact source interval assigned by the canonical typed config.
     * @param original Full original two-head serialized fixture, retained by caller.
     * @param config Authenticated QueryInterval assignment with original consumer grain.
     * @param rank Logical membership index, independent of physical device ordinal.
     * @throws std::invalid_argument If source/assignment identity is missing or foreign.
     */
    ConsumerLocalCodecFixture(const NativeCodecFixture &original,const TensorParallelConfig &config,int rank)
        :source(original),first_head(config.forRank(rank).kv_head_start),head_count(config.forRank(rank).kv_head_count)
    {
        const auto declared=config.kvHeadCoverageDeclaration();
        if(!config.validate() || config.kvHeadCoveragePolicy()!=KVHeadCoveragePolicy::QueryInterval ||
            !declared || declared->query_heads!=24 || declared->kv_heads!=2 ||
            !config.queryHeadPartitionDeclaration() || config.queryHeadGranularity()!=3 ||
            first_head<0 || head_count<=0 || first_head+head_count>2)
            throw std::invalid_argument("Compact sparse fixture requires canonical original24/2 consumer coverage");
        storage.resize(std::size_t(source.tokens)*head_count*source.head_bytes);
        for(int token=0;token<source.tokens;++token)
            std::memcpy(storage.data()+std::size_t(token)*head_count*source.head_bytes,
                source.storage.data()+(std::size_t(token)*2+first_head)*source.head_bytes,
                std::size_t(head_count)*source.head_bytes);
        if(source.codec==TensorType::TQ4 || source.codec==TensorType::TQ8)
            rotation.assign(source.rotation.begin()+std::size_t(first_head)*65536,
                source.rotation.begin()+std::size_t(first_head+head_count)*65536);
        if(source.codec==TensorType::AQ8)
            anchor.assign(source.anchor.begin()+first_head*256,source.anchor.begin()+(first_head+head_count)*256);
    }
    /**
     * @brief Bind only local compact bytes and matching source-owned codec metadata.
     * @return Original native codec with exact compact token/head stride and local metadata.
     */
    QwenSparseNativeBank bank() const
    {
        auto result=source.bank();result.data=storage.data();
        result.token_stride_bytes=std::size_t(head_count)*source.head_bytes;
        result.head_stride_bytes=source.head_bytes;
        result.rotations[0]=rotation.empty()?nullptr:rotation.data();
        result.rotations[1]=rotation.empty()||head_count<2?nullptr:rotation.data()+65536;
        result.anchor=anchor.empty()?nullptr:anchor.data();return result;
    }
};
} // namespace llaminar2::qwen4exp::test
