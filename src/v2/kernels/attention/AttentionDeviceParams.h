/**
 * @file AttentionDeviceParams.h
 * @brief Shared device-owned attention replay parameters and grouped-row policy.
 *
 * CUDA and ROCm attention graph replay consume this compact parameter block
 * directly from device memory.  The header also owns the maximum grouped MTP
 * verifier span so graph preparation, backend validation, device writers, and
 * workspace planning cannot silently drift to different row capacities.
 */

#pragma once

namespace llaminar2
{
    namespace attention
    {
        /**
         * @brief Maximum number of target-model rows in one grouped verifier pass.
         *
         * An MTP draft depth of one through three produces two through four
         * target-model rows: the current token plus each drafted continuation.
         * Production GPU attention must therefore reserve and validate four
         * row-local parameter records and four rows of split-decode partials.
         */
        inline constexpr int kMaxGroupedVerifierAttentionRows = 4;

        /**
         * @brief Row-local dynamic geometry consumed by graph-captured attention.
         *
         * Device kernels publish these fields immediately before attention
         * replay.  Keeping the block device-owned avoids a host mirror whose
         * lifetime or ordering could disagree with the captured GPU graph.
         */
        struct AttentionDeviceParams
        {
            int kv_len = 0;          ///< Visible KV positions for this query row.
            int kv_stride = 0;       ///< Physical request-major K/V row capacity.
            int position_offset = 0; ///< Absolute model position of this query row.
            int mask_stride = 0;     ///< Row stride of the optional attention mask.
        };
    } // namespace attention
} // namespace llaminar2
