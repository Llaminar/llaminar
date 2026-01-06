/**
 * @file BlockStructures.h
 * @brief Quantization block structure definitions
 * @author David Sanftenberg
 *
 * Defines packed block structures for all supported quantization formats.
 * These are POD types matching GGML/llama.cpp binary layout for compatibility.
 *
 * This header is intentionally standalone (no dependencies) to avoid circular
 * includes. It's included early in the compilation order by both SIMDHelpers.h
 * and Tensors.h.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

namespace llaminar2
{
    // ========================================================================
    // Simple Quantization Formats (32-element blocks)
    // ========================================================================

    /**
     * @brief IQ4_NL block structure (exactly 18 bytes) representing 32 quantized elements.
     *
     * Layout mirrors GGML's block_iq4_nl. Two 4-bit indices per byte in @p qs select entries
     * in kvalues_iq4nl, scaled by FP16 value @p d.
     */
    struct IQ4_NLBlock
    {
        uint16_t d;     ///< FP16 scale factor
        uint8_t qs[16]; ///< Packed 4-bit indices (2 per byte)

        static constexpr size_t BLOCK_SIZE = 32; ///< Elements per block
    };
    static_assert(sizeof(IQ4_NLBlock) == 18, "IQ4_NLBlock must be 18 bytes");

    /** @brief Q8_0 block: 8-bit quantization (32 elements per block, 34 bytes) */
    struct Q8_0Block
    {
        uint16_t d;    ///< FP16 scale factor
        int8_t qs[32]; ///< 32 quantized int8 values
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q8_0Block) == 34, "Q8_0Block must be 34 bytes");

    /**
     * @brief Q8_1 block: 8-bit quantization with pre-computed sum (36 bytes)
     *
     * Like Q8_0 but stores the pre-computed sum for asymmetric quantization compensation.
     * This format is used as an intermediate format for activations (matching CUDA pattern):
     * - FP32/FP16/BF16 activations → Q8_1 (quantize once per panel)
     * - Q8_1 activations × quantized weights (many dot products)
     * - Pre-computed sum eliminates expensive horizontal reductions in K-loop
     *
     * Formula: output = d * sum(qs[i])
     *
     * GEMM OPTIMIZATION (Nov 2024):
     * The 'sum_qs' field now stores the raw integer sum of qs[i] values (INT16),
     * NOT d × sum(qs[i]). This eliminates FP16→FP32 conversion and division from
     * the GEMM K-loop, allowing pure VNNI compute followed by scaling in post-processing.
     *
     * Compensation formula in GEMM:
     *   C[i,j] = Σ_kb ((accum[kb] - 128*sum_qs[kb]) * d_a[kb] * d_b[kb])
     */
    struct Q8_1Block
    {
        uint16_t d;     ///< FP16 scale factor
        int16_t sum_qs; ///< INT16 pre-computed sum: Σ(qs[i]) - raw integer sum!
        int8_t qs[32];  ///< 32 quantized int8 values
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q8_1Block) == 36, "Q8_1Block must be 36 bytes");

    /**
     * @brief Q16_1 block: 16-bit quantization with pre-computed sum (72 bytes)
     *
     * Like Q8_1 but with 256× more precision (int16 vs int8).
     * Designed for residual stream where error accumulation is critical.
     *
     * Layout:
     * - float d: FP32 scale factor (4 bytes) - higher precision than Q8_1's FP16 scale!
     * - int32_t sum_qs: INT32 pre-computed sum of qs[i] values (wider range needed)
     * - int16_t qs[32]: 32 quantized int16 values
     *
     * Memory: 4 + 4 + 64 = 72 bytes per block (2× Q8_1)
     *
     * Range: [-32767, 32767] per element vs [-127, 127] for Q8_1
     * This provides 256× finer granularity at 2× memory cost.
     * FP32 scale eliminates scale quantization error that limited Q8_1.
     */
    struct Q16_1Block
    {
        float d;        ///< FP32 scale factor (higher precision than Q8_1's FP16!)
        int32_t sum_qs; ///< INT32 pre-computed sum: Σ(qs[i]) - wider range for int16 values!
        int16_t qs[32]; ///< 32 quantized int16 values
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q16_1Block) == 72, "Q16_1Block must be 72 bytes");

    // ========================================================================
    // Q16_1 Variable Block Size Formats (for Integer Attention v2)
    // ========================================================================
    //
    // These block sizes enable 1-block-per-head attention computation, eliminating
    // the need for per-block scale tracking during Q×K^T and P×V accumulation.
    //
    // Block size selection based on model head_dim:
    //   - 64:  Qwen2.5-0.5B, GPT-2 (1 block per head)
    //   - 128: Qwen3, Llama-3, Mistral (1 block per head)
    //   - 192: DeepSeek V3, Kimi K2 MLA Q/K (1 block per head)
    //
    // The original Q16_1Block (32 elements) is preserved for GEMM compatibility.
    // ========================================================================

    /**
     * @brief Q16BlockSize enum for compile-time block size selection
     *
     * Note: MLA architectures (DeepSeek V3, Kimi K2) should use separate
     * NOPE (128-dim) and ROPE (64-dim) tensors with their own scales,
     * rather than a single 192-block.
     */
    enum class Q16BlockSize : uint8_t
    {
        BLOCK_32 = 32,  ///< Legacy, GEMM compatibility (Q16_1Block)
        BLOCK_64 = 64,  ///< Universal for attention (head_dim=64)
        BLOCK_128 = 128 ///< Optimal for head_dim=128 (Llama-3, Qwen3)
    };

    /**
     * @brief Q16_1 block with 64 elements (optimal for head_dim=64)
     *
     * Memory: 4 + 4 + 128 = 136 bytes per block
     * Overhead: 8/64 = 12.5% (vs 25% for 32-element blocks)
     */
    struct alignas(4) Q16_1Block_64
    {
        float d;        ///< FP32 scale factor
        int32_t sum_qs; ///< INT32 pre-computed sum: Σ(qs[i])
        int16_t qs[64]; ///< 64 quantized int16 values
        static constexpr size_t BLOCK_SIZE = 64;
    };
    static_assert(sizeof(Q16_1Block_64) == 136, "Q16_1Block_64 must be 136 bytes");

    /**
     * @brief Q16_1 block with 128 elements (optimal for head_dim=128)
     *
     * Memory: 4 + 4 + 256 = 264 bytes per block
     * Overhead: 8/128 = 6.25% (vs 25% for 32-element blocks)
     */
    struct alignas(4) Q16_1Block_128
    {
        float d;         ///< FP32 scale factor
        int32_t sum_qs;  ///< INT32 pre-computed sum: Σ(qs[i])
        int16_t qs[128]; ///< 128 quantized int16 values
        static constexpr size_t BLOCK_SIZE = 128;
    };
    static_assert(sizeof(Q16_1Block_128) == 264, "Q16_1Block_128 must be 264 bytes");

    /**
     * @brief Type trait to get Q16_1 block type from Q16BlockSize enum
     */
    template <Q16BlockSize Size>
    struct Q16BlockType;

    template <>
    struct Q16BlockType<Q16BlockSize::BLOCK_32>
    {
        using type = Q16_1Block;
    };
    template <>
    struct Q16BlockType<Q16BlockSize::BLOCK_64>
    {
        using type = Q16_1Block_64;
    };
    template <>
    struct Q16BlockType<Q16BlockSize::BLOCK_128>
    {
        using type = Q16_1Block_128;
    };

    template <Q16BlockSize Size>
    using Q16BlockType_t = typename Q16BlockType<Size>::type;

    /**
     * @brief Select optimal Q16 block size for a given head dimension
     *
     * Returns the block size that achieves 1 block per head (optimal) or
     * minimizes blocks per head for non-standard dimensions.
     *
     * Note: For MLA architectures (DeepSeek V3, Kimi K2), use separate
     * NOPE and ROPE tensors with BLOCK_128 and BLOCK_64 respectively.
     */
    constexpr Q16BlockSize optimal_q16_block_size(int head_dim)
    {
        if (head_dim == 32)
            return Q16BlockSize::BLOCK_32;
        if (head_dim == 64)
            return Q16BlockSize::BLOCK_64;
        if (head_dim == 128)
            return Q16BlockSize::BLOCK_128;
        // Fallback: use largest block that divides evenly, or 64 as universal
        if (head_dim % 128 == 0)
            return Q16BlockSize::BLOCK_128;
        if (head_dim % 64 == 0)
            return Q16BlockSize::BLOCK_64;
        if (head_dim % 32 == 0)
            return Q16BlockSize::BLOCK_32;
        return Q16BlockSize::BLOCK_64; // Universal fallback (GCD of common head dims)
    }

    /**
     * @brief Get the byte size of a Q16 block given its element count
     *
     * Each Q16 block has: float d (4) + int32_t sum_qs (4) + int16_t qs[N] (2*N)
     * Total: 8 + 2*N bytes
     */
    constexpr size_t q16_block_size_bytes(Q16BlockSize size)
    {
        switch (size)
        {
        case Q16BlockSize::BLOCK_32:
            return sizeof(Q16_1Block); // 72 bytes
        case Q16BlockSize::BLOCK_64:
            return sizeof(Q16_1Block_64); // 136 bytes
        case Q16BlockSize::BLOCK_128:
            return sizeof(Q16_1Block_128); // 264 bytes
        default:
            return sizeof(Q16_1Block); // Fallback
        }
    }

    /**
     * @brief Get the element count for a Q16 block size
     */
    constexpr size_t q16_block_size_elements(Q16BlockSize size)
    {
        return static_cast<size_t>(size);
    }

    // ========================================================================
    // Type-Safe Q16 Block Pointer Wrappers
    // ========================================================================
    //
    // These wrappers eliminate the dangerous `const void*` pattern by storing
    // both the pointer and its actual block type together. Accessors return
    // correctly typed pointers or nullptr if the wrong type is requested.
    //
    // Usage:
    //   Q16BlockPtr ptr(tensor->as_block_64());  // Type is remembered
    //   auto* blocks = ptr.as_block_64();        // Returns typed pointer
    //   auto* wrong = ptr.as_block_128();        // Returns nullptr (type mismatch)
    // ========================================================================

    /**
     * @brief Type-safe wrapper for const Q16_1 block pointers with runtime block size.
     *
     * This allows implicit conversion from typed block pointers, automatically
     * setting the block_size field based on the source type.
     */
    struct Q16BlockPtr
    {
        const void *data = nullptr;
        Q16BlockSize block_size = Q16BlockSize::BLOCK_64;

        Q16BlockPtr() = default;

        // Type-safe constructors - block_size is set automatically
        // NON-EXPLICIT to allow implicit conversion (e.g., params.Q = block_ptr)
        Q16BlockPtr(const Q16_1Block *p)
            : data(p), block_size(Q16BlockSize::BLOCK_32)
        {
        }
        Q16BlockPtr(const Q16_1Block_64 *p)
            : data(p), block_size(Q16BlockSize::BLOCK_64)
        {
        }
        Q16BlockPtr(const Q16_1Block_128 *p)
            : data(p), block_size(Q16BlockSize::BLOCK_128)
        {
        }

        // Generic pointer constructor for tests - defaults to BLOCK_64
        // This allows assigning arbitrary pointers for null-check validation tests
        template <typename T,
                  typename = std::enable_if_t<!std::is_same_v<T, Q16_1Block> &&
                                              !std::is_same_v<T, Q16_1Block_64> &&
                                              !std::is_same_v<T, Q16_1Block_128>>>
        Q16BlockPtr(const T *p)
            : data(p), block_size(Q16BlockSize::BLOCK_64)
        {
        }

        // Type-safe accessors - return nullptr if wrong type requested
        const Q16_1Block *as_block_32() const
        {
            return block_size == Q16BlockSize::BLOCK_32
                       ? static_cast<const Q16_1Block *>(data)
                       : nullptr;
        }
        const Q16_1Block_64 *as_block_64() const
        {
            return block_size == Q16BlockSize::BLOCK_64
                       ? static_cast<const Q16_1Block_64 *>(data)
                       : nullptr;
        }
        const Q16_1Block_128 *as_block_128() const
        {
            return block_size == Q16BlockSize::BLOCK_128
                       ? static_cast<const Q16_1Block_128 *>(data)
                       : nullptr;
        }

        bool empty() const { return data == nullptr; }
        explicit operator bool() const { return data != nullptr; }

        /// Get block size in elements (32, 64, or 128)
        int block_elements() const { return static_cast<int>(block_size); }
    };

    /**
     * @brief Type-safe wrapper for mutable Q16_1 block pointers with runtime block size.
     */
    struct Q16BlockMutablePtr
    {
        void *data = nullptr;
        Q16BlockSize block_size = Q16BlockSize::BLOCK_64;

        Q16BlockMutablePtr() = default;

        // Type-safe constructors - NON-EXPLICIT to allow implicit conversion
        Q16BlockMutablePtr(Q16_1Block *p)
            : data(p), block_size(Q16BlockSize::BLOCK_32)
        {
        }
        Q16BlockMutablePtr(Q16_1Block_64 *p)
            : data(p), block_size(Q16BlockSize::BLOCK_64)
        {
        }
        Q16BlockMutablePtr(Q16_1Block_128 *p)
            : data(p), block_size(Q16BlockSize::BLOCK_128)
        {
        }

        // Mutable accessors
        Q16_1Block *as_block_32()
        {
            return block_size == Q16BlockSize::BLOCK_32
                       ? static_cast<Q16_1Block *>(data)
                       : nullptr;
        }
        Q16_1Block_64 *as_block_64()
        {
            return block_size == Q16BlockSize::BLOCK_64
                       ? static_cast<Q16_1Block_64 *>(data)
                       : nullptr;
        }
        Q16_1Block_128 *as_block_128()
        {
            return block_size == Q16BlockSize::BLOCK_128
                       ? static_cast<Q16_1Block_128 *>(data)
                       : nullptr;
        }

        // Const accessors for reading
        const Q16_1Block *as_block_32() const
        {
            return block_size == Q16BlockSize::BLOCK_32
                       ? static_cast<const Q16_1Block *>(data)
                       : nullptr;
        }
        const Q16_1Block_64 *as_block_64() const
        {
            return block_size == Q16BlockSize::BLOCK_64
                       ? static_cast<const Q16_1Block_64 *>(data)
                       : nullptr;
        }
        const Q16_1Block_128 *as_block_128() const
        {
            return block_size == Q16BlockSize::BLOCK_128
                       ? static_cast<const Q16_1Block_128 *>(data)
                       : nullptr;
        }

        bool empty() const { return data == nullptr; }
        explicit operator bool() const { return data != nullptr; }

        int block_elements() const { return static_cast<int>(block_size); }
    };

    /** @brief Q4_0 block: 4-bit quantization (32 elements per block, 18 bytes) */
    struct Q4_0Block
    {
        uint16_t d;     ///< FP16 scale factor
        uint8_t qs[16]; ///< 32 4-bit values packed (2 per byte)
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q4_0Block) == 18, "Q4_0Block must be 18 bytes");

    /** @brief Q4_1 block: 4-bit quantization with min (32 elements per block, 20 bytes) */
    struct Q4_1Block
    {
        uint16_t d;     ///< FP16 scale factor
        uint16_t m;     ///< FP16 minimum value
        uint8_t qs[16]; ///< 32 4-bit values packed (2 per byte)
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q4_1Block) == 20, "Q4_1Block must be 20 bytes");

    /** @brief Q5_0 block: 5-bit quantization (32 elements per block, 22 bytes) */
    struct Q5_0Block
    {
        uint16_t d;     ///< FP16 scale factor
        uint8_t qh[4];  ///< High bits (5th bit) for all 32 elements (32 bits = 4 bytes)
        uint8_t qs[16]; ///< Lower 4 bits of 32 5-bit values packed (2 per byte)
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q5_0Block) == 22, "Q5_0Block must be 22 bytes");

    /** @brief Q5_1 block: 5-bit quantization with min (32 elements per block, 24 bytes) */
    struct Q5_1Block
    {
        uint16_t d;     ///< FP16 scale factor
        uint16_t m;     ///< FP16 minimum value
        uint8_t qh[4];  ///< High bits (5th bit) for all 32 elements (32 bits = 4 bytes)
        uint8_t qs[16]; ///< Lower 4 bits of 32 5-bit values packed (2 per byte)
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q5_1Block) == 24, "Q5_1Block must be 24 bytes");

    // ========================================================================
    // K-Quant Formats (256-element super-blocks)
    // ========================================================================

    /** @brief Q2_K block: 2-bit K-quant (256 elements per super-block, 84 bytes) */
    struct Q2_KBlock
    {
        uint8_t scales[16]; ///< Scales and mins (packed)
        uint8_t qs[64];     ///< 2-bit values packed (4 per byte)
        uint16_t d;         ///< FP16 super-block scale for scales
        uint16_t dmin;      ///< FP16 super-block scale for mins
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q2_KBlock) == 84, "Q2_KBlock must be 84 bytes");

    /** @brief Q3_K block: 3-bit K-quant (256 elements per super-block, 110 bytes) */
    struct Q3_KBlock
    {
        uint8_t hmask[32];  ///< High bit masks (1 bit per element)
        uint8_t qs[64];     ///< Lower 2 bits of 3-bit values
        uint8_t scales[12]; ///< 16 scales packed (6 bits each)
        uint16_t d;         ///< FP16 super-block scale
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q3_KBlock) == 110, "Q3_KBlock must be 110 bytes");

    /** @brief Q4_K block: 4-bit K-quant (256 elements per super-block, 144 bytes) */
    struct Q4_KBlock
    {
        uint16_t d;         ///< FP16 super-block scale
        uint16_t dmin;      ///< FP16 super-block min scale
        uint8_t scales[12]; ///< 12 6-bit scales packed
        uint8_t qs[128];    ///< Lower 4 bits of 4-bit values
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q4_KBlock) == 144, "Q4_KBlock must be 144 bytes");

    /** @brief Q5_K block: 5-bit K-quant (256 elements per super-block, 176 bytes) */
    struct Q5_KBlock
    {
        uint16_t d;         ///< FP16 super-block scale
        uint16_t dmin;      ///< FP16 super-block min scale
        uint8_t scales[12]; ///< 8 6-bit scales packed
        uint8_t qh[32];     ///< High bits (5th bit of 5-bit values)
        uint8_t qs[128];    ///< Lower 4 bits of 5-bit values
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q5_KBlock) == 176, "Q5_KBlock must be 176 bytes");

    /** @brief Q6_K block: 6-bit K-quant (256 elements per super-block, 210 bytes) */
    struct Q6_KBlock
    {
        uint8_t ql[128];   ///< Lower 4 bits of 6-bit values
        uint8_t qh[64];    ///< Upper 2 bits of 6-bit values (packed)
        int8_t scales[16]; ///< Per-block scales
        uint16_t d;        ///< FP16 super-block scale
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q6_KBlock) == 210, "Q6_KBlock must be 210 bytes");

    /** @brief Q8_K block: 8-bit K-quant super-block (256 elements, 288 bytes) */
    struct Q8_KBlock
    {
        int8_t qs[256];    ///< 8-bit quantized values
        int16_t bsums[16]; ///< Block sums for fast dot products
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q8_KBlock) == 288, "Q8_KBlock must be 288 bytes");

    // ========================================================================
    // IQ Formats (Importance Quantization, 256-element super-blocks)
    // ========================================================================

    /** @brief IQ2_XXS block: 2-bit extra-extra-small IQ (256 elements per super-block, 66 bytes) */
    struct IQ2_XXSBlock
    {
        uint16_t d;      ///< FP16 scale factor
        uint16_t qs[32]; ///< Grid indices packed (QK_K/8 = 256/8 = 32 uint16_t)
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ2_XXSBlock) == 66, "IQ2_XXSBlock must be 66 bytes");

    /** @brief IQ2_XS block: 2-bit extra-small IQ (256 elements per super-block, 74 bytes) */
    struct IQ2_XSBlock
    {
        uint16_t d;        ///< FP16 scale factor
        uint16_t qs[32];   ///< Grid indices (QK_K/8 = 32 uint16_t)
        uint8_t scales[8]; ///< Per-block scales (QK_K/32 = 8)
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ2_XSBlock) == 74, "IQ2_XSBlock must be 74 bytes");

    /** @brief IQ2_S block: 2-bit small IQ (256 elements per super-block, 82 bytes) */
    struct IQ2_SBlock
    {
        uint16_t d;        ///< FP16 scale factor
        uint8_t qs[64];    ///< Quantized values (QK_K/4 = 64)
        uint8_t qh[8];     ///< High bits (QK_K/32 = 8)
        uint8_t scales[8]; ///< Scales (QK_K/32 = 8)
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ2_SBlock) == 82, "IQ2_SBlock must be 82 bytes");

    /** @brief IQ3_XXS block: 3-bit extra-extra-small IQ (256 elements per super-block, 98 bytes) */
    struct IQ3_XXSBlock
    {
        uint16_t d;     ///< FP16 scale factor
        uint8_t qs[96]; ///< Grid indices (3*QK_K/8 = 3*256/8 = 96)
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ3_XXSBlock) == 98, "IQ3_XXSBlock must be 98 bytes");

    /** @brief IQ3_S block: 3-bit small IQ (256 elements per super-block, 110 bytes) */
    struct IQ3_SBlock
    {
        uint16_t d;        ///< FP16 scale factor
        uint8_t qs[64];    ///< Quantized values (QK_K/4 = 64)
        uint8_t qh[8];     ///< High bits (QK_K/32 = 8)
        uint8_t signs[32]; ///< Sign patterns (QK_K/8 = 32)
        uint8_t scales[4]; ///< Scales (IQ3S_N_SCALE = QK_K/64 = 4)
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ3_SBlock) == 110, "IQ3_SBlock must be 110 bytes");

    /** @brief IQ4_NL block (already defined above as simple format) */
    // IQ4_NLBlock is defined in the simple quantization section

    /** @brief IQ4_XS block: 4-bit extra-small IQ (256 elements per super-block, 136 bytes) */
    struct IQ4_XSBlock
    {
        uint16_t d;          ///< FP16 scale factor
        uint16_t scales_h;   ///< High bits of scales
        uint8_t scales_l[4]; ///< Low bits of scales (QK_K/64 = 256/64 = 4)
        uint8_t qs[128];     ///< Grid indices (QK_K/2 = 256/2 = 128)
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ4_XSBlock) == 136, "IQ4_XSBlock must be 136 bytes");

    /** @brief IQ1_S block: 1-bit small IQ (256 elements per super-block, 50 bytes) */
    struct IQ1_SBlock
    {
        uint16_t d;     ///< FP16 scale factor
        uint8_t qs[32]; ///< Grid indices (QK_K/8 = 256/8 = 32 bytes)
        uint16_t qh[8]; ///< High bits and scales (QK_K/32 = 256/32 = 8 uint16_t values)
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ1_SBlock) == 50, "IQ1_SBlock must be 50 bytes");

    /** @brief IQ1_M block: 1-bit medium IQ (256 elements per super-block, 56 bytes) */
    struct IQ1_MBlock
    {
        uint8_t qs[32];    ///< Grid indices (QK_K/8 = 256/8 = 32 bytes)
        uint8_t qh[16];    ///< High bits and scale info (packed)
        uint8_t scales[8]; ///< Per-block scale adjustments
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ1_MBlock) == 56, "IQ1_MBlock must be 56 bytes");

} // namespace llaminar2
