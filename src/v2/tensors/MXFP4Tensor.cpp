/**
 * @file MXFP4Tensor.cpp
 * @brief OCP MXFP4 (GGUF type 39) weight tensor.
 *
 * Storage keeps the exact 17-byte GGUF blocks (zero-copy for mmap). All
 * decoders share mxfp4::kValues and the exact block scale 2^(e - 128), so CPU
 * FP32 decode, Q8_0 re-encoding and NativeVNNI preparation describe the same
 * real numbers. NativeVNNI preparation stores the code bytes verbatim and the
 * scale as an exact FP16 word; it throws for an exponent whose scale is not an
 * exact FP16 value instead of introducing a rounding step.
 */

#include "TensorClasses.h"
#include "VnniPackContext.h"
#include "MXFP4Format.h"
#include "FP16Utils.h"
#include "../kernels/KernelFactory.h"
#include "TensorKernels.h"
#include "../utils/Logger.h"
#include "SIMDHelpers.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    namespace
    {
        /** @brief Expected native bytes for a rank-two or GGUF rank-three MXFP4 tensor. */
        size_t expectedMXFP4Bytes(const std::vector<size_t> &shape)
        {
            if (shape.size() != 2u && shape.size() != 3u)
                throw std::invalid_argument("MXFP4Tensor requires a 2D matrix or 3D GGUF expert tensor");
            // GGUF rank-three experts are [K, N, E]; rank two is [N, K].
            const size_t rows = shape.size() == 3u ? shape[1] * shape[2] : shape[0];
            const size_t cols = shape.size() == 3u ? shape[0] : shape[1];
            // Like the other block formats, every row owns its own (possibly
            // partially used) final block.
            return rows * ((cols + MXFP4Block::BLOCK_SIZE - 1) / MXFP4Block::BLOCK_SIZE) * sizeof(MXFP4Block);
        }
    } // namespace

    MXFP4Tensor::MXFP4Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : MXFP4Tensor(shape, AlignedVector<uint8_t>(raw_data))
    {
    }

    MXFP4Tensor::MXFP4Tensor(const std::vector<size_t> &shape, AlignedVector<uint8_t> raw_data, DeviceId device)
        : shape_(shape), is_view_(false), raw_data_(std::move(raw_data)), raw_data_ptr_(nullptr),
          view_byte_offset_(0), parent_(nullptr), device_(device)
    {
        if (!device.is_valid() || !device.is_cpu())
            throw std::invalid_argument("MXFP4Tensor owned native storage requires its exact CPU allocator");
        const size_t expected = expectedMXFP4Bytes(shape_);
        if (raw_data_.size() != expected)
            throw std::invalid_argument("MXFP4 raw data size mismatch: expected " + std::to_string(expected) +
                                        " bytes, got " + std::to_string(raw_data_.size()));
    }

    MXFP4Tensor::MXFP4Tensor(const std::vector<size_t> &shape, const uint8_t *raw_data_ptr,
                             size_t view_byte_offset, std::shared_ptr<TensorBase> parent)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(raw_data_ptr),
          view_byte_offset_(view_byte_offset), parent_(parent), device_(parent ? parent->home_device() : DeviceId::cpu())
    {
    }

    MXFP4Tensor::MXFP4Tensor(const std::vector<size_t> &shape, const uint8_t *mmap_data, size_t byte_size,
                             std::shared_ptr<void> mmap_lifetime_owner)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(mmap_data), view_byte_offset_(0),
          parent_(nullptr), mmap_owner_(std::move(mmap_lifetime_owner)), data_byte_size_(byte_size),
          device_(DeviceId::cpu())
    {
        const size_t expected = expectedMXFP4Bytes(shape_);
        if (byte_size < expected)
            throw std::invalid_argument("MXFP4Tensor: insufficient mmap data (" + std::to_string(byte_size) +
                                        " bytes, expected " + std::to_string(expected) + ")");
    }

    MXFP4Tensor::~MXFP4Tensor()
    {
        retireHostTransferLifetimeBeforeStorageDestruction();
        // Same explicit teardown order as the other block tensors (large 3D experts).
        { AlignedVector<uint8_t>().swap(raw_data_); }
        { std::vector<size_t>().swap(shape_); }
    }

    std::shared_ptr<TensorBase> MXFP4Tensor::create_view(const std::vector<size_t> &new_shape, size_t offset)
    {
        if (new_shape.size() != 2)
            throw std::invalid_argument("MXFP4Tensor::create_view: only 2D views supported");
        const size_t K = cols2d();
        const size_t total_rows = rows2d();
        if (new_shape[1] != K)
            throw std::invalid_argument("MXFP4Tensor::create_view: K dimension must match parent");
        if (offset % K != 0)
            throw std::invalid_argument("MXFP4Tensor::create_view: offset must be row-aligned");
        const size_t start_row = offset / K;
        if (start_row + new_shape[0] > total_rows)
            throw std::out_of_range("MXFP4Tensor::create_view: view exceeds parent bounds");

        const size_t byte_offset = start_row * blocksPerRow() * sizeof(MXFP4Block);
        if (is_view_)
            return std::shared_ptr<MXFP4Tensor>(new MXFP4Tensor(
                new_shape, raw_data_ptr_, view_byte_offset_ + byte_offset,
                parent_ ? parent_ : std::static_pointer_cast<TensorBase>(shared_from_this())));
        return std::shared_ptr<MXFP4Tensor>(new MXFP4Tensor(
            new_shape, raw_data_.data(), byte_offset, std::static_pointer_cast<TensorBase>(shared_from_this())));
    }

    std::unique_ptr<ITensorGemm> MXFP4Tensor::createGemm()
    {
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createGemm(this, dev_type);
    }

    void MXFP4Tensor::decodeBlock(const MXFP4Block &block, float *output)
    {
        const float d = mxfp4::scaleFP32(block.e);
        for (int j = 0; j < 16; ++j)
        {
            // Each product is a small integer times a power of two: exact in FP32.
            output[j] = d * static_cast<float>(mxfp4::kValues[block.qs[j] & 0x0F]);
            output[j + 16] = d * static_cast<float>(mxfp4::kValues[block.qs[j] >> 4]);
        }
    }

    const float *MXFP4Tensor::data() const
    {
        assertValid("MXFP4Tensor::data");
        if (is_raw_data_released())
        {
            LOG_DEBUG("MXFP4Tensor::data() called but raw data was released after GEMM packing");
            return nullptr;
        }
        if (dequant_cache_.empty())
        {
            const size_t rows = rows2d();
            const size_t cols = cols2d();
            dequant_cache_.resize(rows * cols);
#pragma omp parallel for schedule(static) if (rows > 4)
            for (size_t r = 0; r < rows; ++r)
                to_fp32_row(r, dequant_cache_.data() + r * cols);
        }
        return dequant_cache_.data();
    }

    float *MXFP4Tensor::mutable_data()
    {
        throw std::runtime_error("MXFP4Tensor::mutable_data: quantized tensors are immutable");
    }

    bool MXFP4Tensor::copyFrom(const TensorBase *src)
    {
        (void)src;
        LOG_ERROR("[MXFP4Tensor::copyFrom] Not implemented: weight tensors are read-only");
        return false;
    }

    void MXFP4Tensor::to_fp32_row(size_t row_idx, float *buffer) const
    {
        if (row_idx >= rows2d())
            throw std::out_of_range("MXFP4Tensor::to_fp32_row: row index out of bounds");
        const size_t bpr = blocksPerRow();
        const size_t cols = cols2d();
        for (size_t kb = 0; kb < bpr; ++kb)
        {
            float temp[MXFP4Block::BLOCK_SIZE];
            decodeBlock(blockAt(row_idx, kb), temp);
            const size_t begin = kb * MXFP4Block::BLOCK_SIZE;
            std::memcpy(buffer + begin, temp, std::min(MXFP4Block::BLOCK_SIZE, cols - begin) * sizeof(float));
        }
    }

    void MXFP4Tensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
    {
        const size_t cols = cols2d();
        if (offset + count > rows2d() * cols)
            throw std::out_of_range("MXFP4Tensor::to_fp32_span: range exceeds tensor bounds");
        size_t written = 0;
        while (written < count)
        {
            // Logical (row, column) addressing: rows own padded final blocks.
            const size_t element = offset + written;
            const size_t row = element / cols, col = element % cols;
            const size_t within = col % MXFP4Block::BLOCK_SIZE;
            float temp[MXFP4Block::BLOCK_SIZE];
            decodeBlock(blockAt(row, col / MXFP4Block::BLOCK_SIZE), temp);
            const size_t take = std::min({count - written, MXFP4Block::BLOCK_SIZE - within, cols - col});
            std::memcpy(buffer + written, temp + within, take * sizeof(float));
            written += take;
        }
    }

    void MXFP4Tensor::to_bf16(uint16_t *dst) const
    {
        const size_t rows = rows2d(), cols = cols2d();
#pragma omp parallel for schedule(static) if (rows > 4)
        for (size_t r = 0; r < rows; ++r)
        {
            std::vector<float> row(cols);
            to_fp32_row(r, row.data());
            for (size_t c = 0; c < cols; ++c)
                dst[r * cols + c] = simd::fp32_to_bf16(row[c]);
        }
    }

    void MXFP4Tensor::to_fp16(uint16_t *dst) const
    {
        const size_t rows = rows2d(), cols = cols2d();
#pragma omp parallel for schedule(static) if (rows > 4)
        for (size_t r = 0; r < rows; ++r)
        {
            std::vector<float> row(cols);
            to_fp32_row(r, row.data());
            for (size_t c = 0; c < cols; ++c)
                dst[r * cols + c] = fp32_to_fp16(row[c]);
        }
    }

    void MXFP4Tensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
    {
        const size_t total = rows2d() * cols2d();
        if (block_size == MXFP4Block::BLOCK_SIZE && cols2d() % MXFP4Block::BLOCK_SIZE == 0)
        {
            // Native granularity: codes are already integers, the scale is the block scale.
            const auto *data_ptr = static_cast<const uint8_t *>(raw_host_data_ptr());
            if (!data_ptr) throw std::runtime_error("Native tensor raw source was retired");
            const auto *all = reinterpret_cast<const MXFP4Block *>(data_ptr);
            const size_t blocks = total / MXFP4Block::BLOCK_SIZE;
#pragma omp parallel for schedule(static) if (blocks > 1024)
            for (size_t b = 0; b < blocks; ++b)
            {
                for (int j = 0; j < 16; ++j)
                {
                    dst_int8[b * 32 + j] = mxfp4::kValues[all[b].qs[j] & 0x0F];
                    dst_int8[b * 32 + j + 16] = mxfp4::kValues[all[b].qs[j] >> 4];
                }
                dst_scales[b] = mxfp4::scaleFP32(all[b].e);
            }
            return;
        }
        std::vector<float> fp32(total);
        to_fp32(fp32.data());
        const size_t blocks = (total + block_size - 1) / block_size;
        for (size_t b = 0; b < blocks; ++b)
        {
            const size_t begin = b * block_size, count = std::min(block_size, total - begin);
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i) max_abs = std::max(max_abs, std::abs(fp32[begin + i]));
            const float inv = max_abs > 0.0f ? 127.0f / max_abs : 0.0f;
            dst_scales[b] = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
            for (size_t i = 0; i < block_size; ++i)
                dst_int8[begin + i] = i < count
                    ? static_cast<int8_t>(std::lround(std::clamp(fp32[begin + i] * inv, -127.0f, 127.0f))) : 0;
        }
    }

    void MXFP4Tensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        const MXFP4Block &block = blockAt(row_idx, k_block_offset);
        if (!mxfp4::blockFitsFP16(block.e, block.qs))
            throw std::runtime_error("MXFP4Tensor::decode_to_q8_0: block exponent " + std::to_string(block.e) +
                                     " has no exact FP16 scale");
        for (int j = 0; j < 16; ++j)
        {
            output->qs[j] = mxfp4::kValues[block.qs[j] & 0x0F];
            output->qs[j + 16] = mxfp4::kValues[block.qs[j] >> 4];
        }
        output->d = mxfp4::preparedScaleFP16Bits(block.e);
    }

    void MXFP4Tensor::unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const
    {
        const MXFP4Block &block = blockAt(row_idx, k_block_offset);
        for (int j = 0; j < 16; ++j)
        {
            output[j] = mxfp4::kValues[block.qs[j] & 0x0F];
            output[j + 16] = mxfp4::kValues[block.qs[j] >> 4];
        }
    }

    float MXFP4Tensor::get_block_scale(size_t row_idx, size_t k_block_offset) const
    {
        return mxfp4::scaleFP32(blockAt(row_idx, k_block_offset).e);
    }

    void MXFP4Tensor::packVnniBlock(const VnniPackContext &ctx, int source_n, int destination_n, int b) const
    {
        const size_t linear = vnniLinearIdx(ctx, destination_n, b);
        const auto *all = typed_data();
        if (!all) throw std::runtime_error("Native tensor raw source was retired");
        const auto &blk = all[static_cast<size_t>(source_n) * ctx.blocks_per_row + b];
        // Code bytes keep the IQ4_NL nibble order; codebook 25 selects the MXFP4 lookup.
        switch (ctx.primary_layout)
        {
        case VnniPrimaryScaleLayout::FP16Words:
            if (!mxfp4::blockFitsFP16(blk.e, blk.qs))
                throw std::runtime_error("MXFP4Tensor::packVnniBlock: block exponent " + std::to_string(blk.e) +
                                         " (scale 2^" + std::to_string(static_cast<int>(blk.e) - 128) +
                                         ") has no exact FP16 scale; refusing a lossy NativeVNNI preparation");
            ctx.scales_array[linear] = mxfp4::preparedScaleFP16Bits(blk.e);
            break;
        case VnniPrimaryScaleLayout::NativeE8M0Bytes:
        case VnniPrimaryScaleLayout::PowerOfTwoBF16Words:
            if (!mxfp4::blockFitsNativeE8M0(blk.e, blk.qs))
                throw std::runtime_error("MXFP4Tensor::packVnniBlock: block exponent " + std::to_string(blk.e) +
                                         " is outside the native E8M0 range [2, 252]");
            if (ctx.primary_layout == VnniPrimaryScaleLayout::NativeE8M0Bytes)
                reinterpret_cast<uint8_t *>(ctx.scales_array)[linear] = mxfp4::preparedNativeE8M0Byte(blk.e);
            else
                ctx.scales_array[linear] = mxfp4::powerOfTwoBF16Word(mxfp4::preparedNativeE8M0Byte(blk.e));
            break;
        default:
            throw std::invalid_argument("MXFP4Tensor::packVnniBlock: unknown primary scale layout");
        }
        std::memcpy(vnniPayloadDst(ctx, linear), blk.qs, 16);
    }

} // namespace llaminar2
