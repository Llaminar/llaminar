#pragma once

#include "tensors/IQQuantTables.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iterator>

namespace llaminar2::cuda_native_payload
{
    extern __device__ __constant__ int8_t d_iq4nl_values[16];

    extern __device__ __constant__ uint32_t d_iq3s_grid[512];
    extern __device__ uint32_t d_iq3xxs_grid[256];
    extern __device__ __constant__ uint64_t d_iq2s_grid[1024];
    extern __device__ __constant__ uint64_t d_iq2xs_grid[512];
    extern __device__ uint64_t d_iq2xxs_grid[256];
    extern __device__ __constant__ uint64_t d_iq1s_grid[2048];

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
        cudaError_t err = cudaMemcpyToSymbol(d_iq3s_grid, iq3s_grid, sizeof(iq3s_grid));
        if (err != cudaSuccess)
            return false;
        if (!copyHostArrayToDeviceSymbol(d_iq3xxs_grid, iq3xxs_grid, std::size(iq3xxs_grid)))
            return false;
        err = cudaMemcpyToSymbol(d_iq2s_grid, iq2s_grid, sizeof(iq2s_grid));
        if (err != cudaSuccess)
            return false;
        err = cudaMemcpyToSymbol(d_iq2xs_grid, iq2xs_grid, sizeof(iq2xs_grid));
        if (err != cudaSuccess)
            return false;
        if (!copyHostArrayToDeviceSymbol(d_iq2xxs_grid, iq2xxs_grid, std::size(iq2xxs_grid)))
            return false;
        err = cudaMemcpyToSymbol(d_iq1s_grid, iq1s_grid, sizeof(iq1s_grid));
        return err == cudaSuccess;
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
            (CODEBOOK_ID == 6 || CODEBOOK_ID == 7) ? 20 :
            (CODEBOOK_ID == 8) ? 24 :
            (CODEBOOK_ID == 9) ? 12 :
            (CODEBOOK_ID == 10) ? 8 :
            (CODEBOOK_ID == 11) ? 13 :
            (CODEBOOK_ID == 12) ? 12 :
            (CODEBOOK_ID == 13 || CODEBOOK_ID == 14) ? 9 :
            (CODEBOOK_ID == 15) ? 8 :
            (CODEBOOK_ID == 16 || CODEBOOK_ID == 17) ? 6 :
            16;
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

    __device__ __forceinline__ uint32_t centered_sub_16(uint32_t value)
    {
        value ^= 0x80808080u;
        value -= 0x10101010u;
        value ^= 0x80808080u;
        return value;
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
            return d_iq3s_grid[idx];
        else
            return d_iq3xxs_grid[idx];
    }

    template <uint8_t CODEBOOK_ID>
    __device__ __forceinline__ uint64_t iq2_grid_lookup(int idx)
    {
        if constexpr (CODEBOOK_ID == 13)
            return d_iq2s_grid[idx];
        else if constexpr (CODEBOOK_ID == 14)
            return d_iq2xs_grid[idx];
        else
            return d_iq2xxs_grid[idx];
    }

    __device__ __forceinline__ uint64_t iq1_grid_lookup(int idx)
    {
        return d_iq1s_grid[idx];
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
                packed_groups[g] = pack_i8x4(
                    static_cast<int8_t>((raw & 0x0F) - 8),
                    static_cast<int8_t>(((raw >> 8) & 0x0F) - 8),
                    static_cast<int8_t>(((raw >> 16) & 0x0F) - 8),
                    static_cast<int8_t>(((raw >> 24) & 0x0F) - 8));
                packed_groups[g + 4] = pack_i8x4(
                    static_cast<int8_t>(((raw >> 4) & 0x0F) - 8),
                    static_cast<int8_t>(((raw >> 12) & 0x0F) - 8),
                    static_cast<int8_t>(((raw >> 20) & 0x0F) - 8),
                    static_cast<int8_t>(((raw >> 28) & 0x0F) - 8));
            }
        }
        else if constexpr (CODEBOOK_ID == 4)
        {
            for (int g = 0; g < 4; ++g)
            {
                const uint32_t raw = *reinterpret_cast<const uint32_t *>(payload + g * 4);
                packed_groups[g] = pack_i8x4(
                    d_iq4nl_values[raw & 0x0F],
                    d_iq4nl_values[(raw >> 8) & 0x0F],
                    d_iq4nl_values[(raw >> 16) & 0x0F],
                    d_iq4nl_values[(raw >> 24) & 0x0F]);
                packed_groups[g + 4] = pack_i8x4(
                    d_iq4nl_values[(raw >> 4) & 0x0F],
                    d_iq4nl_values[(raw >> 12) & 0x0F],
                    d_iq4nl_values[(raw >> 20) & 0x0F],
                    d_iq4nl_values[(raw >> 28) & 0x0F]);
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
                    packed_groups[half * 4 + g] = pack_i8x4(
                        static_cast<int8_t>((((raw >> (g * 2 + 0)) & 0x03) | ((hb4 & 1u) << 2)) - 4),
                        static_cast<int8_t>((((raw >> (g * 2 + 8)) & 0x03) | (((hb4 >> 1) & 1u) << 2)) - 4),
                        static_cast<int8_t>((((raw >> (g * 2 + 16)) & 0x03) | (((hb4 >> 2) & 1u) << 2)) - 4),
                        static_cast<int8_t>((((raw >> (g * 2 + 24)) & 0x03) | (((hb4 >> 3) & 1u) << 2)) - 4));
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
    }
}