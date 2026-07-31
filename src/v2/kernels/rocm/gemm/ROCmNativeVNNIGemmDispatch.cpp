/**
 * @file ROCmNativeVNNIGemmDispatch.cpp
 * @brief Public dispatch and constant-table publication for ROCm NativeVNNI.
 *
 * This translation unit intentionally contains no GPU template code. It maps
 * each runtime codebook to the one physical HIP shard that owns its kernel
 * instantiations. Exact ownership makes unsupported or duplicate mappings
 * structurally visible while allowing all eight expensive shards to compile in
 * parallel.
 */

#include "ROCmNativeVNNIGemmShard.h"

#include <array>
#include <cstdint>
#include <cstdio>

namespace
{
constexpr ROCmNativeVNNIGemmShardFn shardForCodebook(uint8_t codebook_id)
{
    switch (codebook_id)
    {
    case 0:
    case 4:
        return rocmGemm_native_vnni_fp32_shard_0;
    case 5:
    case 6:
        return rocmGemm_native_vnni_fp32_shard_1;
    case 7:
    case 8:
        return rocmGemm_native_vnni_fp32_shard_2;
    case 9:
    case 10:
        return rocmGemm_native_vnni_fp32_shard_3;
    case 11:
    case 12:
        return rocmGemm_native_vnni_fp32_shard_4;
    case 13:
    case 14:
        return rocmGemm_native_vnni_fp32_shard_5;
    case 15:
    case 16:
        return rocmGemm_native_vnni_fp32_shard_6;
    case 17:
    case 19:
        return rocmGemm_native_vnni_fp32_shard_7;
    default:
        return nullptr;
    }
}

constexpr std::array<ROCmNativeVNNIGridInitShardFn, 8> kGridInitializers = {
    rocmInitIQGridTables_gemm_shard_0,
    rocmInitIQGridTables_gemm_shard_1,
    rocmInitIQGridTables_gemm_shard_2,
    rocmInitIQGridTables_gemm_shard_3,
    rocmInitIQGridTables_gemm_shard_4,
    rocmInitIQGridTables_gemm_shard_5,
    rocmInitIQGridTables_gemm_shard_6,
    rocmInitIQGridTables_gemm_shard_7,
};
} // namespace

extern "C" bool rocmGemm_native_vnni_fp32(
    const int8_t *d_A_int8,
    const uint8_t *d_payload,
    const void *d_block_scales,
    const void *d_block_mins,
    const void *d_block_emins,
    float *d_output,
    const float *d_scales_A,
    const float *d_scales_A_blockwise,
    int M,
    int N,
    int K,
    uint8_t codebook_id,
    int device_id,
    void *stream)
{
    const ROCmNativeVNNIGemmShardFn shard = shardForCodebook(codebook_id);
    if (!shard)
    {
        std::fprintf(
            stderr,
            "[rocmGemm_native_vnni_fp32] unsupported codebook_id=%u\n",
            static_cast<unsigned>(codebook_id));
        return false;
    }

    return shard(
        d_A_int8,
        d_payload,
        d_block_scales,
        d_block_mins,
        d_block_emins,
        d_output,
        d_scales_A,
        d_scales_A_blockwise,
        M,
        N,
        K,
        codebook_id,
        device_id,
        stream);
}

extern "C" bool rocmInitIQGridTables_gemm(
    int device_id,
    const void *h_iq3s_grid,
    const void *h_iq3xxs_grid,
    const void *h_iq2s_grid,
    const void *h_iq2xs_grid,
    const void *h_iq2xxs_grid,
    const void *h_iq1s_grid)
{
    for (const ROCmNativeVNNIGridInitShardFn initialize : kGridInitializers)
    {
        if (!initialize(
                device_id,
                h_iq3s_grid,
                h_iq3xxs_grid,
                h_iq2s_grid,
                h_iq2xs_grid,
                h_iq2xxs_grid,
                h_iq1s_grid))
        {
            return false;
        }
    }
    return true;
}
