#pragma once

#include "tensors/IQQuantTables.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iterator>

namespace llaminar2::cuda_native_vnni
{
    extern __device__ __constant__ int8_t d_iq4nl_values[16];

    extern __device__ uint32_t d_iq3s_grid[512];
    extern __device__ uint32_t d_iq3xxs_grid[256];
    extern __device__ uint64_t d_iq2s_grid[1024];
    extern __device__ uint64_t d_iq2xs_grid[512];
    extern __device__ uint64_t d_iq2xxs_grid[256];
    extern __device__ uint64_t d_iq1s_grid[2048];

    template <typename T, size_t N>
    inline bool copyHostArrayToDeviceSymbol(T (&symbol)[N], const T *src, size_t count)
    {
        void *dst = nullptr;
        size_t bytes = 0;
        cudaError_t err = cudaGetSymbolAddress(&dst, symbol);
        if (err != cudaSuccess)
            return false;
        err = cudaGetSymbolSize(&bytes, symbol);
        if (err != cudaSuccess || bytes < count * sizeof(T))
            return false;
        err = cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice);
        return err == cudaSuccess;
    }

    inline bool initIQGridTables()
    {
        if (!copyHostArrayToDeviceSymbol(d_iq3s_grid, iq3s_grid, std::size(iq3s_grid)))
            return false;
        if (!copyHostArrayToDeviceSymbol(d_iq3xxs_grid, iq3xxs_grid, std::size(iq3xxs_grid)))
            return false;
        if (!copyHostArrayToDeviceSymbol(d_iq2s_grid, iq2s_grid, std::size(iq2s_grid)))
            return false;
        if (!copyHostArrayToDeviceSymbol(d_iq2xs_grid, iq2xs_grid, std::size(iq2xs_grid)))
            return false;
        if (!copyHostArrayToDeviceSymbol(d_iq2xxs_grid, iq2xxs_grid, std::size(iq2xxs_grid)))
            return false;
        if (!copyHostArrayToDeviceSymbol(d_iq1s_grid, iq1s_grid, std::size(iq1s_grid)))
            return false;
        return true;
    }

    template <uint8_t CODEBOOK_ID>
    struct CodebookTraits
    {
        static constexpr bool is_iq3_grid = (CODEBOOK_ID == 11 || CODEBOOK_ID == 12);
        static constexpr bool is_iq2_grid = (CODEBOOK_ID == 13 || CODEBOOK_ID == 14 || CODEBOOK_ID == 15);
        static constexpr bool is_iq1_grid = (CODEBOOK_ID == 16 || CODEBOOK_ID == 17);
        static constexpr bool is_iq_grid = is_iq3_grid || is_iq2_grid || is_iq1_grid;
        static constexpr bool is_asymmetric = (CODEBOOK_ID == 5 || CODEBOOK_ID == 7 || CODEBOOK_ID == 16);
        static constexpr bool is_dual_scale = (CODEBOOK_ID == 8 || CODEBOOK_ID == 9 || CODEBOOK_ID == 10 ||
                                               CODEBOOK_ID == 13 || CODEBOOK_ID == 14 || CODEBOOK_ID == 17);
        static constexpr bool is_dual_scale_asym = (CODEBOOK_ID == 10);
        static constexpr bool is_iq1_m = (CODEBOOK_ID == 17);
        static constexpr int payload_bytes =
            (CODEBOOK_ID == 19)                        ? 32
            : (CODEBOOK_ID == 6 || CODEBOOK_ID == 7)   ? 20
            : (CODEBOOK_ID == 8)                       ? 24
            : (CODEBOOK_ID == 9)                       ? 12
            : (CODEBOOK_ID == 10)                      ? 8
            : (CODEBOOK_ID == 11)                      ? 13
            : (CODEBOOK_ID == 12)                      ? 12
            : (CODEBOOK_ID == 13 || CODEBOOK_ID == 14) ? 9
            : (CODEBOOK_ID == 15)                      ? 8
            : (CODEBOOK_ID == 16 || CODEBOOK_ID == 17) ? 6
                                                       : 16;
    };

    __device__ __forceinline__ float fp16_bits_to_float(uint16_t bits)
    {
        return __half2float(*reinterpret_cast<const __half *>(&bits));
    }

    __device__ __forceinline__ int sum_packed_i8(int32_t packed)
    {
        const int8_t *vals = reinterpret_cast<const int8_t *>(&packed);
        return static_cast<int>(vals[0]) + static_cast<int>(vals[1]) +
               static_cast<int>(vals[2]) + static_cast<int>(vals[3]);
    }

    __device__ __forceinline__ uint32_t load_payload_word(const uint8_t *payload, int group_idx)
    {
        return *reinterpret_cast<const uint32_t *>(payload + (group_idx & 3) * 4);
    }

    __device__ __forceinline__ int32_t pack_i8x4(int8_t v0, int8_t v1, int8_t v2, int8_t v3)
    {
        union
        {
            int8_t vals[4];
            int32_t packed;
        } out{{v0, v1, v2, v3}};
        return out.packed;
    }

    __device__ __forceinline__ int8_t iq4nl_lookup_reg(uint32_t idx)
    {
        constexpr uint32_t kPack0 = 0xbfad9881u;
        constexpr uint32_t kPack1 = 0xf6eaddcfu;
        constexpr uint32_t kPack2 = 0x26190d01u;
        constexpr uint32_t kPack3 = 0x71594535u;

        uint32_t packed = kPack0;
        switch ((idx >> 2) & 0x3u)
        {
        case 0:
            packed = kPack0;
            break;
        case 1:
            packed = kPack1;
            break;
        case 2:
            packed = kPack2;
            break;
        default:
            packed = kPack3;
            break;
        }

        const uint32_t shift = (idx & 0x3u) * 8u;
        return static_cast<int8_t>((packed >> shift) & 0xFFu);
    }

    // Per-word IQ4_NL decode: processes both low and high nibbles of a raw uint32
    // word in one call, producing 2 packed int8 outputs (8 decoded values total).
    // Computes PRMT selectors and blend masks directly from the raw word using
    // efficient bitwise packing, avoiding per-nibble extraction overhead.
    __device__ __forceinline__ void iq4nl_decode_word(
        uint32_t w, uint32_t &out_lo, uint32_t &out_hi)
    {
        constexpr uint32_t kLut0 = 0xbfad9881u; // vals[0..3]
        constexpr uint32_t kLut1 = 0xf6eaddcfu; // vals[4..7]
        constexpr uint32_t kLut2 = 0x26190d01u; // vals[8..11]
        constexpr uint32_t kLut3 = 0x71594535u; // vals[12..15]

        // --- Low nibbles: bits [3:0] of each byte ---
        // Pack 4 low nibbles from 8-bit stride to 4-bit PRMT selector:
        //   byte0[3:0] → sel[3:0], byte1[3:0] → sel[7:4],
        //   byte2[3:0] → sel[11:8], byte3[3:0] → sel[15:12]
        const uint32_t w_lo = w & 0x0F0F0F0Fu;
        const uint32_t lo_merged = (w_lo | (w_lo >> 4)) & 0x00FF00FFu;
        const uint32_t sel_lo = (lo_merged | (lo_merged >> 8)) & 0xFFFFu;

        // --- High nibbles: bits [7:4] of each byte ---
        const uint32_t w_hi = (w >> 4) & 0x0F0F0F0Fu;
        const uint32_t hi_merged = (w_hi | (w_hi >> 4)) & 0x00FF00FFu;
        const uint32_t sel_hi = (hi_merged | (hi_merged >> 8)) & 0xFFFFu;

        // PRMT lookups from both LUT halves
        const uint32_t lo_from_lo = __byte_perm(kLut0, kLut1, sel_lo);
        const uint32_t lo_from_hi = __byte_perm(kLut2, kLut3, sel_lo);
        const uint32_t hi_from_lo = __byte_perm(kLut0, kLut1, sel_hi);
        const uint32_t hi_from_hi = __byte_perm(kLut2, kLut3, sel_hi);

        // Blend masks from bit 3 of each nibble (bit3==1 means index >= 8)
        const uint32_t mask_lo = __vsub4(0u, (w >> 3) & 0x01010101u);
        const uint32_t mask_hi = __vsub4(0u, (w >> 7) & 0x01010101u);

        // Blend: (from_hi & mask) | (from_lo & ~mask) via LOP3 truth table 0xE4
        asm("lop3.b32 %0, %1, %2, %3, 0xE4;"
            : "=r"(out_lo) : "r"(lo_from_hi), "r"(lo_from_lo), "r"(mask_lo));
        asm("lop3.b32 %0, %1, %2, %3, 0xE4;"
            : "=r"(out_hi) : "r"(hi_from_hi), "r"(hi_from_lo), "r"(mask_hi));
    }

    __device__ __forceinline__ uint32_t centered_sub_16(uint32_t value)
    {
        value ^= 0x80808080u;
        value -= 0x10101010u;
        value ^= 0x80808080u;
        return value;
    }

    __device__ __forceinline__ uint32_t centered_sub_4(uint32_t value)
    {
        value ^= 0x80808080u;
        value -= 0x04040404u;
        value ^= 0x80808080u;
        return value;
    }

    __device__ __forceinline__ uint32_t q3k_highbit_pack(uint32_t hb4)
    {
        return ((hb4 & 0x1u) << 2) |
               ((hb4 & 0x2u) << 9) |
               ((hb4 & 0x4u) << 16) |
               ((hb4 & 0x8u) << 23);
    }

    __device__ __forceinline__ uint32_t iq_apply_signs_4(uint32_t grid4, uint8_t sign_lo4)
    {
        const int8_t *grid = reinterpret_cast<const int8_t *>(&grid4);
        return static_cast<uint32_t>(pack_i8x4(
            (sign_lo4 & 0x1) ? static_cast<int8_t>(-grid[0]) : grid[0],
            (sign_lo4 & 0x2) ? static_cast<int8_t>(-grid[1]) : grid[1],
            (sign_lo4 & 0x4) ? static_cast<int8_t>(-grid[2]) : grid[2],
            (sign_lo4 & 0x8) ? static_cast<int8_t>(-grid[3]) : grid[3]));
    }

    template <uint8_t CODEBOOK_ID>
    __device__ __forceinline__ uint32_t iq3_grid_lookup(int idx)
    {
        if constexpr (CODEBOOK_ID == 11)
        {
#if __CUDA_ARCH__ >= 350
            return __ldg(&d_iq3s_grid[idx]);
#else
            return d_iq3s_grid[idx];
#endif
        }
        else
            return d_iq3xxs_grid[idx];
    }

    template <uint8_t CODEBOOK_ID>
    __device__ __forceinline__ uint64_t iq2_grid_lookup(int idx)
    {
        if constexpr (CODEBOOK_ID == 13)
        {
#if __CUDA_ARCH__ >= 350
            return __ldg(&d_iq2s_grid[idx]);
#else
            return d_iq2s_grid[idx];
#endif
        }
        else if constexpr (CODEBOOK_ID == 14)
        {
#if __CUDA_ARCH__ >= 350
            return __ldg(&d_iq2xs_grid[idx]);
#else
            return d_iq2xs_grid[idx];
#endif
        }
        else
            return d_iq2xxs_grid[idx];
    }

    __device__ __forceinline__ uint64_t iq1_grid_lookup(int idx)
    {
#if __CUDA_ARCH__ >= 350
        return __ldg(&d_iq1s_grid[idx]);
#else
        return d_iq1s_grid[idx];
#endif
    }

    template <uint8_t CODEBOOK_ID>
    __device__ __forceinline__ int payload_bytes_for_codebook()
    {
        return CodebookTraits<CODEBOOK_ID>::payload_bytes;
    }

    template <uint8_t CODEBOOK_ID>
    __device__ __forceinline__ void decode_groups(const uint8_t *payload, int32_t (&packed_groups)[8])
    {
        if constexpr (CODEBOOK_ID == 0)
        {
            for (int g = 0; g < 4; ++g)
            {
                const uint32_t raw = *reinterpret_cast<const uint32_t *>(payload + g * 4);
                const uint32_t lo = raw & 0x0F0F0F0Fu;
                const uint32_t hi = (raw >> 4) & 0x0F0F0F0Fu;
                packed_groups[g] = static_cast<int32_t>(__vsub4(lo, 0x08080808u));
                packed_groups[g + 4] = static_cast<int32_t>(__vsub4(hi, 0x08080808u));
            }
        }
        else if constexpr (CODEBOOK_ID == 4)
        {
            for (int g = 0; g < 4; ++g)
            {
                const uint32_t raw = *reinterpret_cast<const uint32_t *>(payload + g * 4);
                packed_groups[g] = pack_i8x4(
                    iq4nl_lookup_reg(raw & 0x0F),
                    iq4nl_lookup_reg((raw >> 8) & 0x0F),
                    iq4nl_lookup_reg((raw >> 16) & 0x0F),
                    iq4nl_lookup_reg((raw >> 24) & 0x0F));
                packed_groups[g + 4] = pack_i8x4(
                    iq4nl_lookup_reg((raw >> 4) & 0x0F),
                    iq4nl_lookup_reg((raw >> 12) & 0x0F),
                    iq4nl_lookup_reg((raw >> 20) & 0x0F),
                    iq4nl_lookup_reg((raw >> 28) & 0x0F));
            }
        }
        else if constexpr (CODEBOOK_ID == 5)
        {
            for (int g = 0; g < 4; ++g)
            {
                const uint32_t raw = *reinterpret_cast<const uint32_t *>(payload + g * 4);
                packed_groups[g] = static_cast<int32_t>(raw & 0x0F0F0F0Fu);
                packed_groups[g + 4] = static_cast<int32_t>((raw >> 4) & 0x0F0F0F0Fu);
            }
        }
        else if constexpr (CODEBOOK_ID == 6 || CODEBOOK_ID == 7)
        {
            const uint32_t qh_bits = *reinterpret_cast<const uint32_t *>(payload + 16);
            for (int g = 0; g < 4; ++g)
            {
                const uint32_t raw = *reinterpret_cast<const uint32_t *>(payload + g * 4);
                const uint32_t hb4_lo = (qh_bits >> (g * 4)) & 0xFu;
                const uint32_t hb4_hi = (qh_bits >> (g * 4 + 16)) & 0xFu;
                const uint32_t hb_lo = ((hb4_lo & 1u) << 4) | ((hb4_lo & 2u) << 11) |
                                       ((hb4_lo & 4u) << 18) | ((hb4_lo & 8u) << 25);
                const uint32_t hb_hi = ((hb4_hi & 1u) << 4) | ((hb4_hi & 2u) << 11) |
                                       ((hb4_hi & 4u) << 18) | ((hb4_hi & 8u) << 25);
                uint32_t lo = (raw & 0x0F0F0F0Fu) | hb_lo;
                uint32_t hi = ((raw >> 4) & 0x0F0F0F0Fu) | hb_hi;
                if constexpr (CODEBOOK_ID == 6)
                {
                    lo = centered_sub_16(lo);
                    hi = centered_sub_16(hi);
                }
                packed_groups[g] = static_cast<int32_t>(lo);
                packed_groups[g + 4] = static_cast<int32_t>(hi);
            }
        }
        else if constexpr (CODEBOOK_ID == 8)
        {
            const uint8_t *qh = payload + 16;
            for (int g = 0; g < 4; ++g)
            {
                const uint32_t raw = *reinterpret_cast<const uint32_t *>(payload + g * 4);
                packed_groups[g] = pack_i8x4(
                    static_cast<int8_t>(((raw & 0x0F) | ((qh[g] & 0x03) << 4)) - 32),
                    static_cast<int8_t>((((raw >> 8) & 0x0F) | (((qh[g] >> 2) & 0x03) << 4)) - 32),
                    static_cast<int8_t>((((raw >> 16) & 0x0F) | (((qh[g] >> 4) & 0x03) << 4)) - 32),
                    static_cast<int8_t>((((raw >> 24) & 0x0F) | (((qh[g] >> 6) & 0x03) << 4)) - 32));
                packed_groups[g + 4] = pack_i8x4(
                    static_cast<int8_t>((((raw >> 4) & 0x0F) | ((qh[g + 4] & 0x03) << 4)) - 32),
                    static_cast<int8_t>((((raw >> 12) & 0x0F) | (((qh[g + 4] >> 2) & 0x03) << 4)) - 32),
                    static_cast<int8_t>((((raw >> 20) & 0x0F) | (((qh[g + 4] >> 4) & 0x03) << 4)) - 32),
                    static_cast<int8_t>((((raw >> 28) & 0x0F) | (((qh[g + 4] >> 6) & 0x03) << 4)) - 32));
            }
        }
        else if constexpr (CODEBOOK_ID == 9)
        {
            const uint32_t hbits = *reinterpret_cast<const uint32_t *>(payload + 8);
            for (int half = 0; half < 2; ++half)
            {
                const uint32_t raw = *reinterpret_cast<const uint32_t *>(payload + half * 4);
                for (int g = 0; g < 4; ++g)
                {
                    const uint32_t hb4 = (hbits >> ((half * 4 + g) * 4)) & 0xFu;
                    const uint32_t packed = ((raw >> (g * 2)) & 0x03030303u) | q3k_highbit_pack(hb4);
                    packed_groups[half * 4 + g] = static_cast<int32_t>(centered_sub_4(packed));
                }
            }
        }
        else if constexpr (CODEBOOK_ID == 10)
        {
            for (int half = 0; half < 2; ++half)
            {
                const uint32_t raw = *reinterpret_cast<const uint32_t *>(payload + half * 4);
                for (int g = 0; g < 4; ++g)
                {
                    packed_groups[half * 4 + g] = pack_i8x4(
                        static_cast<int8_t>((raw >> (g * 2 + 0)) & 0x03),
                        static_cast<int8_t>((raw >> (g * 2 + 8)) & 0x03),
                        static_cast<int8_t>((raw >> (g * 2 + 16)) & 0x03),
                        static_cast<int8_t>((raw >> (g * 2 + 24)) & 0x03));
                }
            }
        }
        else if constexpr (CODEBOOK_ID == 11 || CODEBOOK_ID == 12)
        {
            const uint8_t qh_byte = (CODEBOOK_ID == 11) ? payload[8] : 0;
            const int sign_base = (CODEBOOK_ID == 11) ? 9 : 8;
            for (int g = 0; g < 8; ++g)
            {
                int idx = payload[g];
                if constexpr (CODEBOOK_ID == 11)
                    idx |= static_cast<int>((qh_byte >> g) & 1u) << 8;
                const uint32_t grid4 = iq3_grid_lookup<CODEBOOK_ID>(idx);
                const uint8_t signs = payload[sign_base + g / 2];
                const uint8_t nibble = (g & 1) ? static_cast<uint8_t>(signs >> 4) : static_cast<uint8_t>(signs & 0x0F);
                packed_groups[g] = static_cast<int32_t>(iq_apply_signs_4(grid4, nibble));
            }
        }
        else if constexpr (CODEBOOK_ID == 13 || CODEBOOK_ID == 14 || CODEBOOK_ID == 15)
        {
            const uint8_t qh_byte = (CODEBOOK_ID == 15) ? 0 : payload[4];
            const int sign_base = (CODEBOOK_ID == 15) ? 4 : 5;
            for (int l = 0; l < 4; ++l)
            {
                int idx = payload[l];
                if constexpr (CODEBOOK_ID == 13)
                    idx |= static_cast<int>((qh_byte >> (2 * l)) & 0x3u) << 8;
                else if constexpr (CODEBOOK_ID == 14)
                    idx |= static_cast<int>((qh_byte >> l) & 0x1u) << 8;
                const uint64_t grid8 = iq2_grid_lookup<CODEBOOK_ID>(idx);
                const uint32_t lo = static_cast<uint32_t>(grid8);
                const uint32_t hi = static_cast<uint32_t>(grid8 >> 32);
                const uint8_t signs = payload[sign_base + l];
                packed_groups[l * 2] = static_cast<int32_t>(iq_apply_signs_4(lo, signs & 0x0F));
                packed_groups[l * 2 + 1] = static_cast<int32_t>(iq_apply_signs_4(hi, signs >> 4));
            }
        }
        else if constexpr (CODEBOOK_ID == 16)
        {
            const uint16_t qh_word = static_cast<uint16_t>(payload[4]) | (static_cast<uint16_t>(payload[5]) << 8);
            for (int l = 0; l < 4; ++l)
            {
                const int idx = payload[l] | (static_cast<int>((qh_word >> (3 * l)) & 0x7u) << 8);
                const uint64_t grid8 = iq1_grid_lookup(idx);
                packed_groups[l * 2] = static_cast<int32_t>(static_cast<uint32_t>(grid8));
                packed_groups[l * 2 + 1] = static_cast<int32_t>(static_cast<uint32_t>(grid8 >> 32));
            }
        }
        else if constexpr (CODEBOOK_ID == 17)
        {
            const uint8_t qh0 = payload[4];
            const uint8_t qh1 = payload[5];
            const int idx0 = payload[0] | (static_cast<int>(qh0 & 0x07u) << 8);
            const int idx1 = payload[1] | (static_cast<int>((qh0 >> 4) & 0x07u) << 8);
            const int idx2 = payload[2] | (static_cast<int>(qh1 & 0x07u) << 8);
            const int idx3 = payload[3] | (static_cast<int>((qh1 >> 4) & 0x07u) << 8);
            uint64_t grid8 = iq1_grid_lookup(idx0);
            packed_groups[0] = static_cast<int32_t>(static_cast<uint32_t>(grid8));
            packed_groups[1] = static_cast<int32_t>(static_cast<uint32_t>(grid8 >> 32));
            grid8 = iq1_grid_lookup(idx1);
            packed_groups[2] = static_cast<int32_t>(static_cast<uint32_t>(grid8));
            packed_groups[3] = static_cast<int32_t>(static_cast<uint32_t>(grid8 >> 32));
            grid8 = iq1_grid_lookup(idx2);
            packed_groups[4] = static_cast<int32_t>(static_cast<uint32_t>(grid8));
            packed_groups[5] = static_cast<int32_t>(static_cast<uint32_t>(grid8 >> 32));
            grid8 = iq1_grid_lookup(idx3);
            packed_groups[6] = static_cast<int32_t>(static_cast<uint32_t>(grid8));
            packed_groups[7] = static_cast<int32_t>(static_cast<uint32_t>(grid8 >> 32));
        }
        else if constexpr (CODEBOOK_ID == 19) // Q8_0: 32 raw int8 values → direct copy
        {
#pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                packed_groups[g] = *reinterpret_cast<const int32_t *>(payload + g * 4);
            }
        }
    }

    // =====================================================================
    // Vectorized variant: uses 128-bit (int4) loads for 16-byte codebooks.
    // Reduces instruction count at the cost of software pipelining.
    // Use for compute-bound kernels (kpar); prefer scalar decode_groups for
    // memory-bound kernels (wide) where load-compute overlap matters.
    // =====================================================================
    template <uint8_t CODEBOOK_ID>
    __device__ __forceinline__ void decode_groups_vec(const uint8_t *payload, int32_t (&packed_groups)[8])
    {
        if constexpr (CodebookTraits<CODEBOOK_ID>::payload_bytes == 16)
        {
            const int4 v = *reinterpret_cast<const int4 *>(payload);
            const uint32_t raws[4] = {
                static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y),
                static_cast<uint32_t>(v.z), static_cast<uint32_t>(v.w)};

            if constexpr (CODEBOOK_ID == 0)
            {
#pragma unroll
                for (int g = 0; g < 4; ++g)
                {
                    const uint32_t raw = raws[g];
                    const uint32_t lo = raw & 0x0F0F0F0Fu;
                    const uint32_t hi = (raw >> 4) & 0x0F0F0F0Fu;
                    packed_groups[g] = static_cast<int32_t>(__vsub4(lo, 0x08080808u));
                    packed_groups[g + 4] = static_cast<int32_t>(__vsub4(hi, 0x08080808u));
                }
            }
            else if constexpr (CODEBOOK_ID == 4)
            {
#pragma unroll
                for (int g = 0; g < 4; ++g)
                {
                    const uint32_t raw = raws[g];
                    packed_groups[g] = pack_i8x4(
                        iq4nl_lookup_reg(raw & 0x0F),
                        iq4nl_lookup_reg((raw >> 8) & 0x0F),
                        iq4nl_lookup_reg((raw >> 16) & 0x0F),
                        iq4nl_lookup_reg((raw >> 24) & 0x0F));
                    packed_groups[g + 4] = pack_i8x4(
                        iq4nl_lookup_reg((raw >> 4) & 0x0F),
                        iq4nl_lookup_reg((raw >> 12) & 0x0F),
                        iq4nl_lookup_reg((raw >> 20) & 0x0F),
                        iq4nl_lookup_reg((raw >> 28) & 0x0F));
                }
            }
            else if constexpr (CODEBOOK_ID == 5)
            {
#pragma unroll
                for (int g = 0; g < 4; ++g)
                {
                    packed_groups[g] = static_cast<int32_t>(raws[g] & 0x0F0F0F0Fu);
                    packed_groups[g + 4] = static_cast<int32_t>((raws[g] >> 4) & 0x0F0F0F0Fu);
                }
            }
            else
            {
                decode_groups<CODEBOOK_ID>(payload, packed_groups);
            }
        }
        else if constexpr (CODEBOOK_ID == 19) // Q8_0: 32 bytes — two 128-bit loads
        {
            const int4 v0 = *reinterpret_cast<const int4 *>(payload);
            const int4 v1 = *reinterpret_cast<const int4 *>(payload + 16);
            packed_groups[0] = v0.x;
            packed_groups[1] = v0.y;
            packed_groups[2] = v0.z;
            packed_groups[3] = v0.w;
            packed_groups[4] = v1.x;
            packed_groups[5] = v1.y;
            packed_groups[6] = v1.z;
            packed_groups[7] = v1.w;
        }
        else
        {
            // Non-16-byte codebooks: fall back to scalar decode
            decode_groups<CODEBOOK_ID>(payload, packed_groups);
        }
    }
}