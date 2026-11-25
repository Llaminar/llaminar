/**
 * @file KVStorageLayout.cpp
 * @brief Resolve native KV codecs and separate token payload from request-wide basis storage.
 *
 * Block sizes come from the production codec structures. Hybrid GPU Q8 is
 * linear, while CPU and attention-only GPU Q8/TQ use anchored keys. Q16 uses
 * its selected head-block width. No page is initialized from a different basis
 * merely because its allocation or logical-token origin changes.
 */
#include "KVStorageLayout.h"
#include "tensors/BlockStructures.h"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
namespace
{
/** @brief Checked payload multiplication, with no capacity or live-state accounting. */
std::size_t multiply(std::size_t a, std::size_t b)
{
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("Native KV payload multiplication overflows");
    return a * b;
}
/** @brief Checked payload addition. */
std::size_t add(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("Native KV payload addition overflows");
    return a + b;
}
/** @brief Codec-block rounding without overflowing numerator addition. */
std::size_t blocks(std::size_t count, std::size_t width) { return count / width + (count % width != 0); }
}

KVStorageFormat parseKVStorageFormat(const std::string &precision)
{
    auto value = precision;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "fp32") return KVStorageFormat::FP32;
    if (value == "bf16") return KVStorageFormat::BF16;
    if (value == "fp16") return KVStorageFormat::FP16;
    if (value == "q8_1") return KVStorageFormat::Q8_1;
    if (value == "q16_1") return KVStorageFormat::Q16_1;
    if (value == "tq4") return KVStorageFormat::TQ4;
    if (value == "tq" || value == "tq8") return KVStorageFormat::TQ8;
    throw std::invalid_argument("Unsupported KV-cache precision token '" + precision + "'");
}

KVNativeStorageLayout KVNativeStorageLayout::resolve(KVCacheFamily family, int heads, int head_dim,
    const std::string &precision, DeviceId device)
{
    if ((family != KVCacheFamily::AttentionOnly && family != KVCacheFamily::Hybrid) || heads <= 0 || head_dim <= 0 ||
        (!device.is_cpu() && !device.is_cuda() && !device.is_rocm()))
        throw std::invalid_argument("Native KV layout requires a known family, positive geometry and actual backend");
    const auto codec = parseKVStorageFormat(precision);
    const auto h = static_cast<std::size_t>(heads), d = static_cast<std::size_t>(head_dim);
    const auto dimensions = multiply(h, d);
    const auto linear = [&](std::size_t bytes) { return KVNativeStorageLayout(codec, KVKeyBasisScope::None, bytes, bytes, 0); };
    switch (codec)
    {
    case KVStorageFormat::FP32: return linear(multiply(dimensions, sizeof(float)));
    case KVStorageFormat::FP16:
    case KVStorageFormat::BF16: return linear(multiply(dimensions, sizeof(std::uint16_t)));
    case KVStorageFormat::Q16_1:
    {
        if (!device.is_cpu()) throw std::invalid_argument("Q16_1 has no CUDA/ROCm KV-cache implementation");
        const auto selected = optimal_q16_block_size(head_dim);
        return linear(multiply(h, multiply(blocks(d, q16_block_size_elements(selected)), q16_block_size_bytes(selected))));
    }
    case KVStorageFormat::Q8_1:
        if (family == KVCacheFamily::Hybrid && device.is_gpu())
            return linear(multiply(blocks(dimensions, Q8_1Block::BLOCK_SIZE), sizeof(Q8_1Block)));
        break;
    case KVStorageFormat::TQ4:
    case KVStorageFormat::TQ8:
        if (family == KVCacheFamily::Hybrid)
            throw std::invalid_argument("Hybrid KV caches do not implement TurboQuant storage");
        break;
    }
    if (head_dim != 64 && head_dim != 128 && head_dim != 256)
        throw std::invalid_argument("Anchored KV-cache head_dim must be 64, 128, or 256");
    const auto key_block = head_dim == 64 ? sizeof(AttentionKeyQ8Block_64) :
        head_dim == 128 ? sizeof(AttentionKeyQ8Block_128) : sizeof(AttentionKeyQ8Block_256);
    std::size_t value_block = 0;
    if (codec == KVStorageFormat::Q8_1)
        value_block = multiply(blocks(d, Q8_1Block::BLOCK_SIZE), sizeof(Q8_1Block));
    else if (codec == KVStorageFormat::TQ4)
        value_block = head_dim == 64 ? sizeof(TQ4Block_64) : head_dim == 128 ? sizeof(TQ4Block_128) : sizeof(TQ4Block_256);
    else
        value_block = head_dim == 64 ? sizeof(TQ8Block_64) : head_dim == 128 ? sizeof(TQ8Block_128) : sizeof(TQ8Block_256);
    return KVNativeStorageLayout(codec, KVKeyBasisScope::RequestFirstSerialKey,
        multiply(h, key_block), multiply(h, value_block), multiply(dimensions, sizeof(float)));
}

LogicalKVPayload KVNativeStorageLayout::pagePayload(int tokens) const
{
    if (tokens <= 0) throw std::invalid_argument("Native KV payload requires positive live tokens");
    const auto k = multiply(static_cast<std::size_t>(tokens), key_position_bytes);
    const auto v = multiply(static_cast<std::size_t>(tokens), value_position_bytes);
    (void)add(k, v);
    return {k, v};
}
LogicalKVPayload KVNativeStorageLayout::archivePayload(int tokens) const
{
    auto payload = pagePayload(tokens);
    payload.k_bytes = add(payload.k_bytes, request_basis_bytes);
    (void)add(payload.k_bytes, payload.v_bytes);
    return payload;
}
}
