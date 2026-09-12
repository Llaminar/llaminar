/**
 * @file KVCacheMemoryEstimator.cpp
 * @brief Exact persistent KV-cache allocation accounting by backend and codec.
 *
 * The estimator mirrors the concrete CPU, CUDA, and ROCm ring-cache owners.
 * Graph workspace is intentionally excluded because MemoryPlanner prices it
 * separately. Every persistent payload, pointer table, sequence-state row,
 * request anchor, and model-lifetime TurboQuant rotation is included here.
 */

#include "planning/KVCacheMemoryEstimator.h"

#include "tensors/BlockStructures.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    namespace
    {
        enum class KVStorageFormat
        {
            FP32,
            BF16,
            FP16,
            Q8_1,
            Q16_1,
            TQ4,
            TQ8,
        };

        [[nodiscard]] std::size_t checkedAdd(
            std::size_t left,
            std::size_t right,
            const char *what)
        {
            if (right > std::numeric_limits<std::size_t>::max() - left)
            {
                throw std::overflow_error(
                    std::string("KV-cache ") + what + " overflows size_t");
            }
            return left + right;
        }

        [[nodiscard]] std::size_t checkedMultiply(
            std::size_t left,
            std::size_t right,
            const char *what)
        {
            if (left != 0 &&
                right > std::numeric_limits<std::size_t>::max() / left)
            {
                throw std::overflow_error(
                    std::string("KV-cache ") + what + " overflows size_t");
            }
            return left * right;
        }

        [[nodiscard]] std::size_t ceilDivide(
            std::size_t numerator,
            std::size_t denominator)
        {
            if (denominator == 0)
                throw std::invalid_argument("KV-cache divisor must be positive");
            return numerator / denominator +
                   static_cast<std::size_t>(numerator % denominator != 0);
        }

        [[nodiscard]] std::string normalizedPrecision(std::string value)
        {
            std::transform(
                value.begin(), value.end(), value.begin(),
                [](unsigned char ch)
                { return static_cast<char>(std::tolower(ch)); });
            return value;
        }

        [[nodiscard]] KVStorageFormat parseFormat(
            const std::string &precision)
        {
            const std::string value = normalizedPrecision(precision);
            if (value == "fp32")
                return KVStorageFormat::FP32;
            if (value == "bf16")
                return KVStorageFormat::BF16;
            if (value == "fp16")
                return KVStorageFormat::FP16;
            if (value == "q8_1")
                return KVStorageFormat::Q8_1;
            if (value == "q16_1")
                return KVStorageFormat::Q16_1;
            if (value == "tq4")
                return KVStorageFormat::TQ4;
            if (value == "tq" || value == "tq8")
                return KVStorageFormat::TQ8;
            throw std::invalid_argument(
                "Unsupported KV-cache precision token '" + precision + "'");
        }

        void requireSupportedDevice(DeviceId device)
        {
            if (!device.is_cpu() && !device.is_cuda() && !device.is_rocm())
            {
                throw std::invalid_argument(
                    "KV-cache planning requires a CPU, CUDA, or ROCm device");
            }
        }

        void requireTurboQuantHeadDimension(int head_dim)
        {
            if (head_dim != 64 && head_dim != 128 && head_dim != 256)
            {
                throw std::invalid_argument(
                    "TurboQuant KV-cache head_dim must be 64, 128, or 256");
            }
        }

        [[nodiscard]] std::size_t tq4BlockBytes(int head_dim)
        {
            requireTurboQuantHeadDimension(head_dim);
            if (head_dim == 64)
                return sizeof(TQ4Block_64);
            if (head_dim == 128)
                return sizeof(TQ4Block_128);
            return sizeof(TQ4Block_256);
        }

        [[nodiscard]] std::size_t tq8BlockBytes(int head_dim)
        {
            requireTurboQuantHeadDimension(head_dim);
            if (head_dim == 64)
                return sizeof(TQ8Block_64);
            if (head_dim == 128)
                return sizeof(TQ8Block_128);
            return sizeof(TQ8Block_256);
        }

        [[nodiscard]] std::size_t aq8KeyBlockBytes(int head_dim)
        {
            requireTurboQuantHeadDimension(head_dim);
            if (head_dim == 64)
                return sizeof(AttentionKeyQ8Block_64);
            if (head_dim == 128)
                return sizeof(AttentionKeyQ8Block_128);
            return sizeof(AttentionKeyQ8Block_256);
        }

        [[nodiscard]] std::size_t gpuMetadataBytes(
            std::size_t entry_count,
            std::size_t pointer_tables)
        {
            /*
             * Every GPU cache owns device head/count rows. Ordinary caches
             * publish K/V pointer tables; compressed caches additionally
             * publish the request-anchor table.
             */
            const std::size_t sequence_state = checkedMultiply(
                checkedMultiply(entry_count, 2, "sequence-state row count"),
                sizeof(int),
                "sequence-state bytes");
            const std::size_t tables = checkedMultiply(
                checkedMultiply(
                    entry_count,
                    pointer_tables,
                    "pointer-table entry count"),
                sizeof(void *),
                "pointer-table bytes");
            return checkedAdd(
                sequence_state,
                tables,
                "GPU metadata bytes");
        }

        [[nodiscard]] std::size_t estimateLinearCache(
            std::size_t entry_count,
            std::size_t max_seq_len,
            std::size_t kv_dim,
            std::size_t element_bytes,
            DeviceId device)
        {
            const std::size_t one_horizon = checkedMultiply(
                checkedMultiply(max_seq_len, kv_dim, "linear horizon elements"),
                element_bytes,
                "linear horizon bytes");

            /*
             * CUDA's current concrete cache owns K/V plus permanent K/V
             * linearization horizons. ROCm binds linearization storage from
             * graph workspace, and CPU needs only the two ring payloads.
             */
            const std::size_t horizons = device.is_cuda() ? 4 : 2;
            const std::size_t payload = checkedMultiply(
                checkedMultiply(entry_count, horizons, "linear horizon count"),
                one_horizon,
                "linear cache payload");
            if (device.is_cpu())
                return payload;
            return checkedAdd(
                payload,
                gpuMetadataBytes(entry_count, 2),
                "linear GPU cache bytes");
        }

        [[nodiscard]] std::size_t estimateCPUQ16(
            std::size_t entry_count,
            std::size_t max_seq_len,
            std::size_t local_kv_heads,
            int head_dim)
        {
            /* Q16_1 production caches are head-major with one typed block row. */
            const Q16BlockSize block_size =
                optimal_q16_block_size(head_dim);
            const std::size_t blocks_per_head = ceilDivide(
                static_cast<std::size_t>(head_dim),
                q16_block_size_elements(block_size));
            const std::size_t one_head_row = checkedMultiply(
                blocks_per_head,
                q16_block_size_bytes(block_size),
                "Q16_1 head-row bytes");
            const std::size_t one_tensor = checkedMultiply(
                checkedMultiply(
                    max_seq_len,
                    local_kv_heads,
                    "Q16_1 head rows"),
                one_head_row,
                "Q16_1 tensor bytes");
            return checkedMultiply(
                checkedMultiply(entry_count, 2, "Q16_1 tensor count"),
                one_tensor,
                "Q16_1 cache bytes");
        }

        [[nodiscard]] std::size_t estimateCompressed(
            std::size_t layer_count,
            std::size_t batch_size,
            std::size_t max_seq_len,
            std::size_t local_kv_heads,
            int head_dim,
            KVStorageFormat format,
            DeviceId device)
        {
            requireTurboQuantHeadDimension(head_dim);
            const std::size_t entry_count = checkedMultiply(
                layer_count,
                batch_size,
                "compressed entry count");
            const std::size_t key_block = aq8KeyBlockBytes(head_dim);
            std::size_t value_block = 0;
            if (format == KVStorageFormat::Q8_1)
            {
                value_block = checkedMultiply(
                    ceilDivide(
                        static_cast<std::size_t>(head_dim),
                        Q8_1Block::BLOCK_SIZE),
                    sizeof(Q8_1Block),
                    "Q8_1 value-head bytes");
            }
            else if (format == KVStorageFormat::TQ4)
            {
                value_block = tq4BlockBytes(head_dim);
            }
            else
            {
                value_block = tq8BlockBytes(head_dim);
            }

            const std::size_t position_bytes = checkedMultiply(
                local_kv_heads,
                checkedAdd(key_block, value_block, "compressed K/V block bytes"),
                "compressed position bytes");
            std::size_t total = checkedMultiply(
                checkedMultiply(
                    entry_count,
                    max_seq_len,
                    "compressed position count"),
                position_bytes,
                "compressed ring bytes");

            /* One FP32 request anchor is permanent for every cache entry. */
            const std::size_t anchor_bytes = checkedMultiply(
                checkedMultiply(
                    entry_count,
                    local_kv_heads,
                    "compressed anchor head count"),
                checkedMultiply(
                    static_cast<std::size_t>(head_dim),
                    sizeof(float),
                    "compressed anchor head bytes"),
                "compressed anchor bytes");
            total = checkedAdd(total, anchor_bytes, "compressed payload plus anchors");
            // CPU and GPU share the exact physical K/V codec BOM. Only GPU
            // caches additionally own device metadata and rotation replicas.
            if (device.is_cpu()) return total;
            total = checkedAdd(
                total,
                gpuMetadataBytes(entry_count, 3),
                "compressed payload plus metadata");

            if (format != KVStorageFormat::Q8_1)
            {
                /* Forward and transposed FP32 rotations are model-lifetime. */
                const std::size_t matrix_elements = checkedMultiply(
                    checkedMultiply(
                        layer_count,
                        local_kv_heads,
                        "rotation matrix count"),
                    checkedMultiply(
                        static_cast<std::size_t>(head_dim),
                        static_cast<std::size_t>(head_dim),
                        "rotation matrix elements"),
                    "rotation elements");
                const std::size_t rotation_bytes = checkedMultiply(
                    checkedMultiply(
                        matrix_elements,
                        2,
                        "forward/transposed rotation count"),
                    sizeof(float),
                    "rotation bytes");
                total = checkedAdd(total, rotation_bytes, "compressed cache plus rotations");
            }
            return total;
        }
    } // namespace

    float KVCacheMemoryEstimator::getBytesPerElement(
        const std::string &kv_precision)
    {
        switch (parseFormat(kv_precision))
        {
        case KVStorageFormat::FP32:
            return 4.0f;
        case KVStorageFormat::BF16:
        case KVStorageFormat::FP16:
            return 2.0f;
        case KVStorageFormat::Q8_1:
            return static_cast<float>(sizeof(Q8_1Block)) /
                   static_cast<float>(Q8_1Block::BLOCK_SIZE);
        case KVStorageFormat::Q16_1:
        case KVStorageFormat::TQ4:
        case KVStorageFormat::TQ8:
            throw std::invalid_argument(
                "Q16_1 and TurboQuant bytes per element are head-dimension and backend dependent; use estimate()");
        }
        throw std::logic_error("Unreachable KV-cache storage format");
    }

    GPULogicalKVBlockEstimate
    KVCacheMemoryEstimator::estimateGPULogicalBlock(
        KVCacheFamily family,
        int token_count,
        int n_kv_heads,
        int head_dim,
        const std::string &kv_precision,
        DeviceId device)
    {
        if (!device.is_cuda() && !device.is_rocm())
        {
            throw std::invalid_argument(
                "Logical GPU KV-block planning requires CUDA or ROCm");
        }
        if (token_count <= 0 || n_kv_heads <= 0 || head_dim <= 0)
        {
            throw std::invalid_argument(
                "Logical GPU KV-block geometry must be positive");
        }

        const KVStorageFormat format = parseFormat(kv_precision);
        if (family == KVCacheFamily::Hybrid &&
            (format == KVStorageFormat::TQ4 || format == KVStorageFormat::TQ8))
            throw std::invalid_argument("Hybrid GPU KV caches do not implement TurboQuant storage");
        const std::size_t tokens = static_cast<std::size_t>(token_count);
        const std::size_t heads = static_cast<std::size_t>(n_kv_heads);
        const std::size_t dimensions = static_cast<std::size_t>(head_dim);
        const std::size_t kv_dim = checkedMultiply(
            heads, dimensions, "logical-block KV dimension");

        GPULogicalKVBlockEstimate result;
        switch (format)
        {
        case KVStorageFormat::FP32:
        case KVStorageFormat::BF16:
        case KVStorageFormat::FP16:
        {
            const std::size_t element_bytes =
                format == KVStorageFormat::FP32
                    ? sizeof(float)
                    : sizeof(std::uint16_t);
            const std::size_t row_bytes = checkedMultiply(
                kv_dim, element_bytes, "logical linear row bytes");
            result.k_bytes = checkedMultiply(
                tokens, row_bytes, "logical linear key bytes");
            result.v_bytes = checkedMultiply(
                tokens, row_bytes, "logical linear value bytes");
            break;
        }
        case KVStorageFormat::Q8_1:
        case KVStorageFormat::TQ4:
        case KVStorageFormat::TQ8:
        {
            if (family == KVCacheFamily::Hybrid)
            {
                // Hybrid GPU caches serialize their native linear Q8_1 rows;
                // unlike attention-only caches they have no AQ8 request anchor.
                const std::size_t row_bytes = checkedMultiply(
                    ceilDivide(kv_dim, Q8_1Block::BLOCK_SIZE),
                    sizeof(Q8_1Block), "hybrid Q8_1 row bytes");
                result.k_bytes = result.v_bytes = checkedMultiply(
                    tokens, row_bytes, "hybrid Q8_1 logical payload");
                break;
            }
            const std::size_t key_position_bytes = checkedMultiply(
                heads,
                aq8KeyBlockBytes(head_dim),
                "logical AQ8 key-position bytes");
            std::size_t value_head_bytes = 0;
            if (format == KVStorageFormat::Q8_1)
            {
                value_head_bytes = checkedMultiply(
                    ceilDivide(dimensions, Q8_1Block::BLOCK_SIZE),
                    sizeof(Q8_1Block),
                    "logical Q8_1 value-head bytes");
            }
            else if (format == KVStorageFormat::TQ4)
            {
                value_head_bytes = tq4BlockBytes(head_dim);
            }
            else
            {
                value_head_bytes = tq8BlockBytes(head_dim);
            }

            const std::size_t anchor_bytes = checkedMultiply(
                kv_dim, sizeof(float), "logical AQ8 anchor bytes");
            result.k_bytes = checkedAdd(
                anchor_bytes,
                checkedMultiply(
                    tokens,
                    key_position_bytes,
                    "logical AQ8 key payload bytes"),
                "logical AQ8 key payload plus anchor");
            result.v_bytes = checkedMultiply(
                tokens,
                checkedMultiply(
                    heads,
                    value_head_bytes,
                    "logical compressed value-position bytes"),
                "logical compressed value payload bytes");
            break;
        }
        case KVStorageFormat::Q16_1:
            throw std::invalid_argument(
                "Q16_1 has no CUDA/ROCm KV-cache implementation");
        }

        (void)checkedAdd(
            result.k_bytes,
            result.v_bytes,
            "logical GPU K/V block bytes");
        return result;
    }

    std::size_t KVCacheMemoryEstimator::estimate(
        KVCacheFamily family,
        int n_layers,
        int batch_size,
        int max_seq_len,
        int n_kv_heads,
        int head_dim,
        const std::string &kv_precision,
        DeviceId device)
    {
        if (n_layers <= 0 || batch_size <= 0 || max_seq_len <= 0 ||
            n_kv_heads <= 0 || head_dim <= 0)
        {
            return 0;
        }
        requireSupportedDevice(device);
        const KVStorageFormat format = parseFormat(kv_precision);
        if (family == KVCacheFamily::Hybrid &&
            (format == KVStorageFormat::TQ4 || format == KVStorageFormat::TQ8))
            throw std::invalid_argument("Hybrid KV caches do not implement TurboQuant storage");
        const std::size_t layers = static_cast<std::size_t>(n_layers);
        const std::size_t batches = static_cast<std::size_t>(batch_size);
        const std::size_t sequence = static_cast<std::size_t>(max_seq_len);
        const std::size_t heads = static_cast<std::size_t>(n_kv_heads);
        const std::size_t entry_count = checkedMultiply(
            layers,
            batches,
            "entry count");
        const std::size_t kv_dim = checkedMultiply(
            heads,
            static_cast<std::size_t>(head_dim),
            "KV dimension");

        switch (format)
        {
        case KVStorageFormat::FP32:
            return estimateLinearCache(
                entry_count, sequence, kv_dim, sizeof(float), device);
        case KVStorageFormat::BF16:
        case KVStorageFormat::FP16:
            return estimateLinearCache(
                entry_count, sequence, kv_dim, sizeof(std::uint16_t), device);
        case KVStorageFormat::Q8_1:
            if (family == KVCacheFamily::Hybrid && device.is_gpu())
            {
                // Charge the concrete linear-cache owner: CUDA retains its
                // linearization horizons; ROCm borrows them from workspace.
                return estimateLinearCache(entry_count, sequence,
                    ceilDivide(kv_dim, Q8_1Block::BLOCK_SIZE),
                    sizeof(Q8_1Block), device);
            }
            return estimateCompressed(layers, batches, sequence, heads, head_dim, format, device);
        case KVStorageFormat::Q16_1:
            if (!device.is_cpu())
            {
                throw std::invalid_argument(
                    "Q16_1 KV-cache storage is supported only on CPU");
            }
            return estimateCPUQ16(
                entry_count, sequence, heads, head_dim);
        case KVStorageFormat::TQ4:
        case KVStorageFormat::TQ8:
            return estimateCompressed(layers, batches, sequence, heads, head_dim, format, device);
        }
        throw std::logic_error("Unreachable KV-cache storage format");
    }
} // namespace llaminar2
