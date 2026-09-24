/**
 * @file AttentionKeyQ8Tensor.cpp
 * @brief Native anchored key storage and bounded explicit diagnostic decoding.
 *
 * The serialized basis travels with the compressed bytes so a copy cannot
 * reinterpret residuals under another request's anchor. Shape/layout checks
 * happen before any destination mutation. Diagnostic FP32 reads require caller
 * storage and never install an O(context) FP32 cache shadow.
 */
#include "AttentionKeyQ8Tensor.h"
#include "kernels/cpu/attention/CPUAttentionKeyQ8.h"

#include <array>
#include <limits>

namespace llaminar2
{
namespace
{
/** @brief Reject overflowing physical extents before allocating any storage. */
std::size_t multiply(std::size_t a, std::size_t b)
{
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("AQ8 key tensor extent overflows size_t");
    return a * b;
}

/** @brief Resolve only the supported native block types, without padding guesses. */
std::size_t blockBytes(int head_dim)
{
    switch (head_dim)
    {
    case 64: return sizeof(AttentionKeyQ8Block<64>);
    case 128: return sizeof(AttentionKeyQ8Block<128>);
    case 256: return sizeof(AttentionKeyQ8Block<256>);
    default: throw std::invalid_argument("AQ8 key head_dim must be 64, 128 or 256");
    }
}

/** @brief Decode one already indexed physical block with the production codec. */
template <int D>
void decodeHead(const uint8_t *block, const float *anchor, float *destination)
{
    cpu::attention_key_q8::dequantize<D>(
        *reinterpret_cast<const AttentionKeyQ8Block<D> *>(block),
        std::span<const float, D>(anchor, D), std::span<float, D>(destination, D));
}
} // namespace

AttentionKeyQ8Tensor::AttentionKeyQ8Tensor(
    std::size_t positions, std::size_t heads, int head_dim, TensorLayout layout, DeviceId device)
    : positions_(positions), heads_(heads), head_dim_(head_dim), block_bytes_(blockBytes(head_dim)),
      physical_layout_(layout), device_(device)
{
    if (!device.is_cpu() || !positions || !heads ||
        (layout != TensorLayout::KV_POS_HEAD_DIM && layout != TensorLayout::KV_HEAD_POS_DIM))
        throw std::invalid_argument("AQ8 key tensor requires positive CPU KV geometry and an explicit KV layout");
    const auto elements_per_token = multiply(heads, static_cast<std::size_t>(head_dim));
    const auto anchors = multiply(elements_per_token, sizeof(float));
    const auto blocks = multiply(multiply(positions, heads), block_bytes_);
    if (blocks > std::numeric_limits<std::size_t>::max() - anchors)
        throw std::overflow_error("AQ8 key tensor basis plus blocks overflows size_t");
    shape_ = layout == TensorLayout::KV_POS_HEAD_DIM
        ? std::vector<std::size_t>{positions, elements_per_token}
        : std::vector<std::size_t>{multiply(heads, positions), static_cast<std::size_t>(head_dim)};
    // The enclosing cache obtains the PMA lease before constructing its native
    // tensors. This allocation materializes that admitted BOM; it is not a
    // second admission decision or a second allocation ledger.
    storage_ = AlignedVector<uint8_t>(anchors + blocks);
}

AttentionKeyQ8Tensor::~AttentionKeyQ8Tensor()
{
    retireHostTransferLifetimeBeforeStorageDestruction();
}

void AttentionKeyQ8Tensor::setLayout(TensorLayout layout)
{
    if (layout != physical_layout_)
        throw std::invalid_argument("AQ8 key layout is immutable after allocation");
}

const float *AttentionKeyQ8Tensor::data() const
{
    throw std::logic_error("AQ8 keys have no FP32 shadow; use native attention or explicit diagnostic conversion");
}

float *AttentionKeyQ8Tensor::mutable_data()
{
    throw std::logic_error("AQ8 keys must be written by the native cache append owner");
}

bool AttentionKeyQ8Tensor::copyFrom(const TensorBase *source)
{
    const auto *other = dynamic_cast<const AttentionKeyQ8Tensor *>(source);
    if (!other || other->positions_ != positions_ || other->heads_ != heads_ ||
        other->head_dim_ != head_dim_ || other->physical_layout_ != physical_layout_)
        return false;
    std::memmove(storage_.data(), other->storage_.data(), storage_.size());
    return true;
}

std::span<const float> AttentionKeyQ8Tensor::anchor() const
{
    return {reinterpret_cast<const float *>(storage_.data()), heads_ * head_dim_};
}

std::span<float> AttentionKeyQ8Tensor::mutable_anchor()
{
    return {reinterpret_cast<float *>(storage_.data()), heads_ * head_dim_};
}

const uint8_t *AttentionKeyQ8Tensor::blocks() const { return storage_.data() + anchor_bytes(); }
uint8_t *AttentionKeyQ8Tensor::mutable_blocks() { return storage_.data() + anchor_bytes(); }

std::size_t AttentionKeyQ8Tensor::block_index(std::size_t position, std::size_t head) const
{
    if (position >= positions_ || head >= heads_)
        throw std::out_of_range("AQ8 key block exceeds physical geometry");
    return physical_layout_ == TensorLayout::KV_POS_HEAD_DIM ? position * heads_ + head : head * positions_ + position;
}

void AttentionKeyQ8Tensor::to_fp32(float *destination) const
{
    to_fp32_span(0, numel(), destination);
}

void AttentionKeyQ8Tensor::to_fp32_row(std::size_t row, float *destination) const
{
    if (row >= shape_[0]) throw std::out_of_range("AQ8 diagnostic row exceeds tensor shape");
    to_fp32_span(row * shape_[1], shape_[1], destination);
}

void AttentionKeyQ8Tensor::to_fp32_span(std::size_t offset, std::size_t count, float *destination) const
{
    if (offset > numel() || count > numel() - offset || (count && !destination))
        throw std::out_of_range("AQ8 diagnostic span exceeds tensor shape or has no output");
    alignas(64) std::array<float, 256> scratch;
    while (count)
    {
        const auto block = offset / head_dim_;
        const auto coordinate = offset % head_dim_;
        const auto head = physical_layout_ == TensorLayout::KV_POS_HEAD_DIM ? block % heads_ : block / positions_;
        const auto *basis = anchor().data() + head * head_dim_;
        const auto *encoded = blocks() + block * block_bytes_;
        switch (head_dim_)
        {
        case 64: decodeHead<64>(encoded, basis, scratch.data()); break;
        case 128: decodeHead<128>(encoded, basis, scratch.data()); break;
        case 256: decodeHead<256>(encoded, basis, scratch.data()); break;
        }
        const auto copied = std::min(count, static_cast<std::size_t>(head_dim_) - coordinate);
        std::memcpy(destination, scratch.data() + coordinate, copied * sizeof(float));
        offset += copied;
        count -= copied;
        destination += copied;
    }
}

void AttentionKeyQ8Tensor::to_bf16(uint16_t *) const
{
    throw std::logic_error("AQ8 keys do not expose BF16 activation conversion");
}
void AttentionKeyQ8Tensor::to_fp16(uint16_t *) const
{
    throw std::logic_error("AQ8 keys do not expose FP16 activation conversion");
}
void AttentionKeyQ8Tensor::to_int8_blocked(int8_t *, float *, std::size_t) const
{
    throw std::logic_error("AQ8 keys do not expose INT8 activation packing");
}
std::unique_ptr<ITensorGemm> AttentionKeyQ8Tensor::createGemm()
{
    throw std::logic_error("AQ8 is an attention key codec, not a GEMM weight format");
}
std::shared_ptr<TensorBase> AttentionKeyQ8Tensor::create_view(const std::vector<std::size_t> &, std::size_t)
{
    throw std::logic_error("AQ8 key views must preserve their request basis and complete KV geometry");
}
} // namespace llaminar2
