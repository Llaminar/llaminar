#include "TensorFactory.h"
#include "TensorBase.h"
#include "SimpleTensor.h"
#include "CosmaTensor.h"
#include "BF16Tensor.h"
#include "ShardedTensorRegistry.h"
#include "../Logger.h"
#include "../utils/DebugEnv.h"
#include <mpi.h>
#include <atomic>
#include <array>
#include <cctype>
#include "../llama.cpp/ggml/src/ggml-common.h"
#include "../llama.cpp/ggml/src/ggml-quants.h"

namespace
{
    // COSMA currently recognizes only matrix labels 'A', 'B', or 'C'.
    // When callers provide arbitrary labels (e.g. "auto_512"), normalize
    // them to a valid single-character label, cycling through A/B/C to
    // minimize collisions when possible.
    std::string normalize_cosma_label(const std::string &label)
    {
        static const std::array<char, 3> kAllowedLabels = {'A', 'B', 'C'};
        static std::atomic<size_t> next_label{0};

        if (!label.empty())
        {
            char c = static_cast<char>(std::toupper(label.front()));
            for (char allowed : kAllowedLabels)
            {
                if (c == allowed)
                {
                    return std::string(1, allowed);
                }
            }
        }

        size_t idx = next_label.fetch_add(1, std::memory_order_relaxed) % kAllowedLabels.size();
        return std::string(1, kAllowedLabels[idx]);
    }
}

namespace llaminar
{

    // ===== QuantizedTensor static member definitions =====
    std::atomic<bool> QuantizedTensor::stats_enabled_{false};
    std::atomic<uint64_t> QuantizedTensor::stat_block_requests_{0};
    std::atomic<uint64_t> QuantizedTensor::stat_block_hits_{0};
    std::atomic<uint64_t> QuantizedTensor::stat_block_misses_{0};
    std::atomic<uint64_t> QuantizedTensor::stat_full_block_fastpath_{0};

    void QuantizedTensor::enable_cache_stats(bool v)
    {
        stats_enabled_.store(v, std::memory_order_relaxed);
    }
    void QuantizedTensor::reset_cache_stats()
    {
        stat_block_requests_.store(0, std::memory_order_relaxed);
        stat_block_hits_.store(0, std::memory_order_relaxed);
        stat_block_misses_.store(0, std::memory_order_relaxed);
        stat_full_block_fastpath_.store(0, std::memory_order_relaxed);
    }
    QuantizedTensor::CacheStats QuantizedTensor::cache_stats()
    {
        return CacheStats{stat_block_requests_.load(std::memory_order_relaxed),
                          stat_block_hits_.load(std::memory_order_relaxed),
                          stat_block_misses_.load(std::memory_order_relaxed),
                          stat_full_block_fastpath_.load(std::memory_order_relaxed)};
    }

    // Factory function implementations

    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_simple(const std::vector<int> &shape)
    {
        // Check for empty tensors (any dimension is 0)
        bool is_empty = false;
        for (int dim : shape)
        {
            if (dim == 0)
            {
                is_empty = true;
                break;
            }
        }

        // Always use SimpleTensor for empty tensors (avoid BF16Tensor complications)
        if (is_empty)
        {
            return std::make_shared<SimpleTensor>(shape);
        }

        // PHASE 5: Check environment flag for BF16 activation storage
        // This allows BF16 to be enabled globally without changing all call sites
        const auto &env = debugEnv();
        if (env.quant.output_bf16)
        {
            return std::make_shared<BF16Tensor>(shape);
        }
        return std::make_shared<SimpleTensor>(shape);
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_simple(const std::vector<int> &shape,
                                                                       const std::vector<float> &data)
    {
        // When data is provided, always use SimpleTensor (for initialization)
        // BF16 conversion happens via copy_from() if needed
        return std::make_shared<SimpleTensor>(shape, data);
    }

    // ================= QuantizedTensor Implementation (Phase 1 Skeleton) =================

    void QuantizedTensor::decodeBlock(size_t block_index, float *dst) const
    {
        if (!dst)
            return;
        const auto &layout = layout_;
        const auto &desc = layout.block_desc;
        if (desc.elements_per_block <= 0 || desc.bytes_per_block <= 0)
        {
            return; // nothing to decode
        }
        size_t offset = block_index * static_cast<size_t>(desc.bytes_per_block);
        if (offset + desc.bytes_per_block > raw_.size())
        {
            // Out of range: zero safety
            std::fill(dst, dst + desc.elements_per_block, 0.0f);
            return;
        }
        const uint8_t *block = raw_.data() + offset;
        auto fp16_to_fp32 = [](uint16_t h) -> float
        {
            // Minimal portable half->float (rounding not critical for scale) adapted from QuantDequant helpers
            uint32_t w = (uint32_t)h << 16;
            uint32_t sign = w & 0x80000000u;
            uint32_t two_w = w + w;
            uint32_t exp_offset = 0xE0u << 23;
            float exp_scale = 0x1.0p-112f;
            union { uint32_t u; float f; } cvt;
            cvt.u = (two_w >> 4) + exp_offset; float normalized = cvt.f * exp_scale;
            uint32_t magic_mask = 126u << 23; float magic_bias = 0.5f;
            cvt.u = (two_w >> 17) | magic_mask; float denorm = cvt.f - magic_bias;
            uint32_t denorm_cut = 1u << 27;
            float result = (two_w < denorm_cut) ? denorm : normalized;
            union { uint32_t u; float f; } out; out.u = sign | ((uint32_t&)result & 0x7FFFFFFFu);
            return result; };

        switch (layout.format)
        {
        case QuantFormat::Q4_0:
        {
            // Uses scale(2 bytes) + 16 packed nibbles -> 32 values
            uint16_t h;
            std::memcpy(&h, block, 2);
            float d = fp16_to_fp32(h);
            const uint8_t *qs = block + 2;
            for (int i = 0; i < 16; ++i)
            {
                uint8_t v = qs[i];
                int lo = (int)(v & 0x0F) - 8;
                int hi = (int)(v >> 4) - 8;
                dst[i] = lo * d;
                dst[i + 16] = hi * d;
            }
            break;
        }
        case QuantFormat::Q5_0:
        {
            // Layout: d(2) | qh(4) | qs(16)
            uint16_t h;
            std::memcpy(&h, block, 2);
            float d = fp16_to_fp32(h);
            uint32_t qh;
            std::memcpy(&qh, block + 2, 4);
            const uint8_t *qs = block + 6;
            // Mirror logic from dequant_block_q5_0 helper
            for (int j = 0; j < 16; ++j)
            {
                const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
                uint8_t q = qs[j];
                int x0 = ((q & 0x0F) | xh_0) - 16;
                int x1 = ((q >> 4) | xh_1) - 16;
                dst[j] = x0 * d;
                dst[j + 16] = x1 * d;
            }
            break;
        }
        case QuantFormat::Q8_0:
        {
            // Layout: d(2) + 32 signed bytes (scale is fp16)
            uint16_t h;
            std::memcpy(&h, block, 2);
            float d = fp16_to_fp32(h);
            const int8_t *vals = reinterpret_cast<const int8_t *>(block + 2);
            for (int i = 0; i < 32; ++i)
                dst[i] = (float)vals[i] * d;
            break;
        }
        case QuantFormat::Q4_K:
        {
            // Each block corresponds to 256 values (QK_K). Upstream struct block_q4_K.
            // We rely on ggml's dequantize_row_q4_K for correctness; decode single block.
            if (desc.elements_per_block != 256)
            {
                std::fill(dst, dst + desc.elements_per_block, 0.0f);
                break;
            }
            const block_q4_K *b = reinterpret_cast<const block_q4_K *>(block);
            float tmp[QK_K];
            dequantize_row_q4_K(b, tmp, QK_K);
            std::memcpy(dst, tmp, sizeof(float) * 256);
            break;
        }
        case QuantFormat::Q5_K:
        {
            if (desc.elements_per_block != 256)
            {
                std::fill(dst, dst + desc.elements_per_block, 0.0f);
                break;
            }
            const block_q5_K *b = reinterpret_cast<const block_q5_K *>(block);
            float tmp[QK_K];
            dequantize_row_q5_K(b, tmp, QK_K);
            std::memcpy(dst, tmp, sizeof(float) * 256);
            break;
        }
        case QuantFormat::Q6_K:
        {
            if (desc.elements_per_block != 256)
            {
                std::fill(dst, dst + desc.elements_per_block, 0.0f);
                break;
            }
            const block_q6_K *b = reinterpret_cast<const block_q6_K *>(block);
            float tmp[QK_K];
            dequantize_row_q6_K(b, tmp, QK_K);
            std::memcpy(dst, tmp, sizeof(float) * 256);
            break;
        }
        case QuantFormat::Q8_K:
        {
            // Q8_K: treat similarly—use upstream row dequant for one block (QK_K values)
            if (desc.elements_per_block != 256)
            {
                std::fill(dst, dst + desc.elements_per_block, 0.0f);
                break;
            }
            const block_q8_K *b = reinterpret_cast<const block_q8_K *>(block);
            float tmp[QK_K];
            dequantize_row_q8_K(b, tmp, QK_K);
            std::memcpy(dst, tmp, sizeof(float) * 256);
            break;
        }
        default:
            LOG_WARN("QuantizedTensor::decodeBlock unsupported quant format id=" << static_cast<int>(layout.format)
                                                                                 << " (block_index=" << block_index << ") — returning zeros");
            std::fill(dst, dst + desc.elements_per_block, 0.0f);
            break;
        }
    }

    // decodeTileFP16() removed - was dead code using wrong _Float16 type.
    // BF16 (bfloat16) decoding is handled via decodeTileBF16() in QuantizedSlabAllocator.

    // Factory helpers for quantized tensors
    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_quantized(const std::vector<int> &shape,
                                                                          QuantFormat format,
                                                                          const std::vector<uint8_t> &raw)
    {
        if (shape.size() != 2)
        {
            LOG_WARN("create_quantized: only 2D tensors supported initially; falling back to SimpleTensor");
            return create_simple(shape); // Fallback
        }
        // Derive block descriptor (simplified defaults per format)
        QuantBlockDescriptor desc;
        switch (format)
        {
        case QuantFormat::Q4_0:
            desc.elements_per_block = 32;
            desc.bytes_per_block = 18; // scale(2) + data(16)
            desc.bits_per_value = 4;
            break;
        case QuantFormat::Q5_0:
            desc.elements_per_block = 32;
            desc.bytes_per_block = 20; // scale(2) + data(18) approx placeholder
            desc.bits_per_value = 5;
            break;
        case QuantFormat::Q8_0:
            desc.elements_per_block = 32;
            desc.bytes_per_block = 34; // scale(2) + data(32)
            desc.bits_per_value = 8;
            break;
        case QuantFormat::Q4_K:
        {
            desc.elements_per_block = QK_K;            // 256
            desc.bytes_per_block = sizeof(block_q4_K); // expected 144 bytes (2 + 2 + 12 + 128)
            desc.bits_per_value = 4;
            desc.is_k_quant = true;
            break;
        }
        case QuantFormat::Q5_K:
        {
            desc.elements_per_block = QK_K;
            desc.bytes_per_block = sizeof(block_q5_K); // 2*half + scales + ql + qh pattern per ggml
            desc.bits_per_value = 5;
            desc.is_k_quant = true;
            break;
        }
        case QuantFormat::Q6_K:
        {
            desc.elements_per_block = QK_K;
            desc.bytes_per_block = sizeof(block_q6_K);
            desc.bits_per_value = 6;
            desc.is_k_quant = true;
            break;
        }
        case QuantFormat::Q8_K:
        {
            desc.elements_per_block = QK_K;
            desc.bytes_per_block = sizeof(block_q8_K);
            desc.bits_per_value = 8;
            desc.is_k_quant = true;
            break;
        }
        case QuantFormat::F16:
        case QuantFormat::F32:
        default:
            // Passthrough to SimpleTensor if unquantized
            LOG_DEBUG("create_quantized: format is unquantized passthrough; creating SimpleTensor");
            // Interpret raw as float bytes if size matches
            size_t expected = static_cast<size_t>(shape[0]) * static_cast<size_t>(shape[1]) * sizeof(float);
            if (raw.size() == expected)
            {
                // Reconstruct float vector
                std::vector<float> vals(shape[0] * shape[1]);
                memcpy(vals.data(), raw.data(), expected);
                return create_simple(shape, vals);
            }
            return create_simple(shape);
        }

        // Compute total blocks (round up columns to block granularity)
        int rows = shape[0];
        int cols = shape[1];
        int per = desc.elements_per_block;
        size_t blocks_per_row = (static_cast<size_t>(cols) + per - 1) / per;
        size_t total_blocks = static_cast<size_t>(rows) * blocks_per_row;

        QuantStorageLayout layout{format, shape, total_blocks, desc};
        return std::make_shared<QuantizedTensor>(layout, raw);
    }

    bool TensorFactory::is_quantized(const std::shared_ptr<llaminar::TensorBase> &t)
    {
        return (bool)std::dynamic_pointer_cast<QuantizedTensor>(t);
    }

    std::shared_ptr<QuantizedTensor> TensorFactory::to_quantized(std::shared_ptr<llaminar::TensorBase> t)
    {
        return std::dynamic_pointer_cast<QuantizedTensor>(t);
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_cosma(const std::vector<int> &shape,
                                                                      const std::string &label,
                                                                      int mpi_rank)
    {
        if (shape.size() != 2)
        {
            LOG_WARN("COSMATensor requires 2D shape, received " + std::to_string(shape.size()) + "D. Falling back to SimpleTensor.");
            return create_simple(shape);
        }

        std::string sanitized_label = normalize_cosma_label(label);

        try
        {
            auto tensor = std::make_shared<COSMATensor>(shape, sanitized_label, mpi_rank);
            if (!tensor->data() || tensor->size() == 0)
            {
                LOG_WARN("COSMATensor allocated with zero local storage for label '" << sanitized_label
                                                                                     << "' on rank " << mpi_rank << ". Falling back to SimpleTensor.");
                return create_simple(shape);
            }
            return tensor;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Failed to create COSMATensor: " + std::string(e.what()) +
                      ". Falling back to SimpleTensor.");
            return create_simple(shape);
        }
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_sharded(const std::vector<int> &shape,
                                                                        ShardSpec::Axis axis_kind,
                                                                        int axis_dim_index,
                                                                        int world,
                                                                        int rank)
    {
        if (axis_dim_index < 0 || axis_dim_index >= (int)shape.size())
            throw std::runtime_error("create_sharded: axis_dim_index out of range");
        if (axis_kind == ShardSpec::Axis::None)
            throw std::runtime_error("create_sharded: axis_kind cannot be None");
        int global_dim = shape[axis_dim_index];
        int base = global_dim / world;
        int rem = global_dim % world;
        int local = base + (rank < rem ? 1 : 0);
        int offset = base * rank + (rank < rem ? rank : rem);
        std::vector<int> local_shape = shape;
        local_shape[axis_dim_index] = local;
        ShardSpec spec;
        spec.type = ShardSpec::Type::Sharded;
        spec.axis = axis_kind;
        spec.world = world;
        spec.rank = rank;
        spec.global_dim = global_dim;
        spec.local_dim = local;
        spec.local_offset = offset;
        auto t = std::make_shared<ShardedSimpleTensor>(local_shape, spec);
        ShardedTensorRegistry::instance().register_tensor(std::static_pointer_cast<ShardedSimpleTensor>(t));
        return t;
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_heads_sharded(const std::vector<int> &shape,
                                                                              int axis_dim_index,
                                                                              int n_heads,
                                                                              int head_dim,
                                                                              int world,
                                                                              int rank)
    {
        if (axis_dim_index < 0 || axis_dim_index >= (int)shape.size())
            throw std::runtime_error("create_heads_sharded: axis_dim_index out of range");
        int global_dim = shape[axis_dim_index];
        if (global_dim != n_heads * head_dim)
            throw std::runtime_error("create_heads_sharded: global_dim mismatch n_heads*head_dim");
        int base_heads = n_heads / world;
        int rem_heads = n_heads % world;
        int local_heads = base_heads + (rank < rem_heads ? 1 : 0);
        int head_offset = base_heads * rank + (rank < rem_heads ? rank : rem_heads);
        int local_dim = local_heads * head_dim;
        int offset = head_offset * head_dim;
        std::vector<int> local_shape = shape;
        local_shape[axis_dim_index] = local_dim;
        ShardSpec spec;
        spec.type = ShardSpec::Type::Sharded;
        spec.axis = ShardSpec::Axis::Heads;
        spec.world = world;
        spec.rank = rank;
        spec.global_dim = global_dim;
        spec.local_dim = local_dim;
        spec.local_offset = offset;
        auto t = std::make_shared<ShardedSimpleTensor>(local_shape, spec);
        ShardedTensorRegistry::instance().register_tensor(std::static_pointer_cast<ShardedSimpleTensor>(t));
        return t;
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_auto(const std::vector<int> &shape,
                                                                     bool prefer_distributed)
    {
        // Heuristics for automatic tensor type selection

        // Check if MPI is available and we have multiple processes
        int mpi_size = 1;
        int mpi_rank = 0;
        int mpi_initialized;
        MPI_Initialized(&mpi_initialized);

        if (mpi_initialized)
        {
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        }

        // Decision criteria for using COSMA tensor:
        // 1. MPI environment with multiple processes
        // 2. 2D matrices (COSMA's strength)
        // 3. Large enough matrices to benefit from distribution
        // 4. User preference for distributed

        bool should_use_cosma = false;

        if (mpi_initialized && mpi_size > 1)
        {
            long total_elements = 1;
            for (int dim : shape)
            {
                total_elements *= dim;
            }

            // Use COSMA for large matrices or when explicitly preferred
            const long COSMA_THRESHOLD = 256 * 256; // 64K elements threshold

            // Additional constraints for COSMA - minimum dimensions for distribution
            bool has_sufficient_rows = (shape.size() >= 2 && shape[0] >= mpi_size);
            bool has_sufficient_cols = (shape.size() >= 2 && shape[1] >= mpi_size);
            bool dimensions_suitable = has_sufficient_rows || has_sufficient_cols;

            should_use_cosma = (total_elements >= COSMA_THRESHOLD) &&
                               dimensions_suitable &&
                               prefer_distributed;
        }

        if (should_use_cosma)
        {
            LOG_DEBUG("Auto-selecting COSMATensor for shape [" +
                      std::to_string(shape[0]) + ", " + std::to_string(shape[1]) +
                      "] with " + std::to_string(mpi_size) + " MPI processes");

            // Generate a label based on tensor characteristics
            std::string auto_label = "auto_" + std::to_string(shape[0]) + "x" + std::to_string(shape[1]);
            return create_cosma(shape, auto_label, mpi_rank);
        }
        else
        {
            LOG_DEBUG("Auto-selecting SimpleTensor for shape of size " + std::to_string(shape.size()));
            return create_simple(shape);
        }
    }

    std::shared_ptr<TensorBase> TensorFactory::convert_to_cosma(const std::shared_ptr<TensorBase> &tensor,
                                                                const std::string &label,
                                                                int mpi_rank)
    {
        if (!tensor)
        {
            return nullptr;
        }

        // If already a COSMA tensor, return as-is
        if (auto cosma_tensor = std::dynamic_pointer_cast<COSMATensor>(tensor))
        {
            return tensor;
        }

        // Create new COSMA tensor and copy data
        try
        {
            auto cosma_tensor = create_cosma(tensor->shape(), label, mpi_rank);
            cosma_tensor->copy_from(*tensor);

            LOG_DEBUG("Converted " + tensor->type_name() + " to COSMATensor");
            return cosma_tensor;
        }
        catch (const std::exception &e)
        {
            LOG_WARN("Failed to convert to COSMATensor: " + std::string(e.what()) +
                     ". Returning original tensor.");
            return tensor;
        }
    }

    void TensorFactory::for_each_sharded(const std::function<void(const ShardSpec &, const std::vector<int> &)> &fn)
    {
        ShardedTensorRegistry::instance().for_each([&](ShardedSimpleTensor &t)
                                                   { fn(t.shard_spec(), t.shape()); });
    }

    void TensorFactory::set_last_shard_role(const std::string &role)
    {
        if (auto last = ShardedTensorRegistry::instance().last())
        {
            last->shard_spec().role = role;
        }
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::convert_to_simple(const std::shared_ptr<llaminar::TensorBase> &tensor)
    {
        if (!tensor)
        {
            return nullptr;
        }

        // If already a simple tensor, return as-is
        if (auto simple_tensor = std::dynamic_pointer_cast<SimpleTensor>(tensor))
        {
            return tensor;
        }

        // Create new simple tensor and copy data
        auto simple_tensor = create_simple(tensor->shape());
        simple_tensor->copy_from(*tensor);

        LOG_DEBUG("Converted " + tensor->type_name() + " to SimpleTensor");
        return simple_tensor;
    }

    // Legacy compatibility methods

    std::shared_ptr<SimpleTensor> TensorFactory::to_simple_tensor(std::shared_ptr<llaminar::TensorBase> tensor)
    {
        if (!tensor)
        {
            return nullptr;
        }

        // If already a SimpleTensor, cast and return
        if (auto simple = std::dynamic_pointer_cast<SimpleTensor>(tensor))
        {
            return simple;
        }

        // Convert to SimpleTensor
        auto simple_tensor = std::make_shared<SimpleTensor>(tensor->shape());
        std::copy(tensor->data(), tensor->data() + tensor->size(), simple_tensor->data());
        return simple_tensor;
    }

    std::shared_ptr<COSMATensor> TensorFactory::to_cosma_tensor(std::shared_ptr<llaminar::TensorBase> tensor)
    {
        if (!tensor)
        {
            return nullptr;
        }

        // If already a COSMATensor, cast and return
        if (auto cosma = std::dynamic_pointer_cast<COSMATensor>(tensor))
        {
            return cosma;
        }

        // Convert to COSMATensor
        int mpi_rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

        try
        {
            auto cosma_tensor = std::make_shared<COSMATensor>(tensor->shape(), normalize_cosma_label("converted"), mpi_rank);
            std::copy(tensor->data(), tensor->data() + tensor->size(), cosma_tensor->data());
            return cosma_tensor;
        }
        catch (const std::exception &e)
        {
            LOG_WARN("Failed to convert to COSMATensor: " + std::string(e.what()));
            return nullptr;
        }
    }

    // ===== BF16 Tensor Creation (Phase 5) =====

    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_bf16(const std::vector<int> &shape)
    {
        return std::make_shared<BF16Tensor>(shape);
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::create_bf16(const std::vector<int> &shape,
                                                                     const std::vector<float> &fp32_data)
    {
        return std::make_shared<BF16Tensor>(shape, fp32_data);
    }

    std::shared_ptr<llaminar::TensorBase> TensorFactory::convert_to_bf16(const std::shared_ptr<llaminar::TensorBase> &tensor)
    {
        if (!tensor)
        {
            return nullptr;
        }

        // If already a BF16Tensor, return as-is
        if (auto bf16 = std::dynamic_pointer_cast<BF16Tensor>(tensor))
        {
            return tensor;
        }

        // Convert FP32 data to BF16
        std::vector<float> fp32_data(tensor->size());
        std::copy(tensor->data(), tensor->data() + tensor->size(), fp32_data.begin());

        return std::make_shared<BF16Tensor>(tensor->shape(), fp32_data);
    }

    std::shared_ptr<BF16Tensor> TensorFactory::to_bf16_tensor(std::shared_ptr<llaminar::TensorBase> tensor)
    {
        if (!tensor)
        {
            return nullptr;
        }

        // If already a BF16Tensor, cast and return
        if (auto bf16 = std::dynamic_pointer_cast<BF16Tensor>(tensor))
        {
            return bf16;
        }

        // Convert to BF16Tensor
        std::vector<float> fp32_data(tensor->size());
        std::copy(tensor->data(), tensor->data() + tensor->size(), fp32_data.begin());

        return std::make_shared<BF16Tensor>(tensor->shape(), fp32_data);
    }

} // namespace llaminar