/**
 * @file ROCmNativeQuantisedWmma.hip.inc
 * @brief Source-exact all-codebook INT8 WMMA prefill for gfx1100.
 *
 * Included by the dedicated wave64 translation unit after its compact decoder.
 * Four wave64 waves own disjoint output fragments. Operands are decoded once
 * per workgroup into four K-block LDS slots and replicated by the WMMA operand
 * mapping. All threads participate in decode; one barrier publishes all four blocks. Integer
 * half-block dots never overflow INT32; only the shared contribution helpers
 * perform FP32 arithmetic. The serial M1 partition boundaries and ordered fold
 * remain authoritative even though no inter-workgroup reduction is needed.
 *
 * This separately compiled wave64 entry leaves ordered wave64 kernels unchanged. This
 * symbol is absent from gfx906 code objects; host admission must require the
 * actual gfx1100 target. There is no emulated or no-op device implementation.
 */
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__)

using NativeWmmaI32x4 = int __attribute__((ext_vector_type(4)));

/** @brief Activation scale ownership, selected once before native submission. */
enum class NativeWmmaActivationScale { Blockwise, WholeRow };

/**
 * @brief Execute a complete exact projection tile with compact native weights.
 * @tparam FMT Physical decoder, including source-factor and expanded-min ABIs.
 * @tparam BM Workgroup row extent, divisible by 32.
 * @tparam BN Workgroup column extent, divisible by 32.
 * @tparam ScaleMode Prepared activation scale ownership and exact application point.
 * @param a Row-major signed activations [M,K].
 * @param payload Block-major compact weights [K/32,N,payload_bytes].
 * @param scales Primary source scale words [K/32,N].
 * @param aux Secondary source factors/minima when the decoder requires them.
 * @param emins Packed half minima for the dual-scale asymmetric decoder.
 * @param output Unique-writer FP32 output [M,N].
 * @param row_scales Whole-row scales, used only without block_scales.
 * @param block_scales Optional row-major activation scales [M,K/32].
 * @param M Live activation rows; only the final tile is padded.
 * @param N Live output columns; only the final tile is padded.
 * @param K Contraction width, a positive multiple of 32.
 * @param serial_partitions Exact public serial-M1 split count.
 */
template <NativeQuantisedGemmFormat FMT, int BM, int BN, NativeWmmaActivationScale ScaleMode>
__global__ __launch_bounds__(256, 1) void native_quantised_wmma_gemm(
    const int8_t *__restrict__ a, const uint8_t *__restrict__ payload,
    const uint16_t *__restrict__ scales, const uint16_t *__restrict__ aux,
    const uint32_t *__restrict__ emins, float *__restrict__ output,
    const float *__restrict__ row_scales, const float *__restrict__ block_scales,
    int M, int N, int K, int serial_partitions)
{
#if defined(__gfx1100__)
    static_assert(__builtin_amdgcn_is_invocable(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64),
        "WMMA requires its dedicated wave64 translation unit");
    using Traits = NATIVE_QUANTISEDGemmTraits<FMT>;
    constexpr int RM = BM / 32, RN = BN / 32;
    constexpr int KB = 4;
    constexpr int SUM_GROUPS = FMT == NATIVE_QUANTISED_GEMM_IQ1_M ? 4 :
        Traits::is_dual_scale_asym ? 2 : Traits::is_asymmetric ? 1 : 0;
    static_assert(BM % 32 == 0 && BN % 32 == 0);
    // Vector banks are permuted by row and stage. Consumers of eight adjacent
    // rows span all banks, while the eight copy lanes for one K128 row also
    // span all banks despite owning four different staging slots.
    __shared__ NativeWmmaI32x4 a_tile[KB][BM][2];
    __shared__ NativeWmmaI32x4 b_tile[KB][BN][2];
    __shared__ uint16_t scale_tile[KB][BN];
    __shared__ uint16_t aux_tile[KB][Traits::needs_aux_compute ? BN : 1];
    __shared__ uint32_t emin_tile[KB][Traits::is_dual_scale_asym ? BN : 1];
    __shared__ uint16_t qh_tile[KB][FMT == NATIVE_QUANTISED_GEMM_IQ1_M ? BN : 1];
    __shared__ int sums[KB][SUM_GROUPS ? SUM_GROUPS : 1][BM];
    __shared__ float activation_scales[KB][BM];
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
    // Adjacent workgroups reuse one compact weight slab across four row tiles.
    // The final group uses its live row-tile count, preserving a bijection even
    // when M is smaller than a group or ends in a partial output tile.
    constexpr unsigned M_GROUP = 4;
    const uint64_t linear_tile = static_cast<uint64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const unsigned group = linear_tile / (static_cast<uint64_t>(M_GROUP) * gridDim.x);
    const unsigned first_m_tile = group * M_GROUP;
    const unsigned live_m_tiles = min(M_GROUP, gridDim.y - first_m_tile);
    const uint64_t within_group = linear_tile - static_cast<uint64_t>(first_m_tile) * gridDim.x;
    const int m0 = (first_m_tile + within_group % live_m_tiles) * BM;
    const int n0 = (within_group / live_m_tiles) * BN;
    const int blocks = K / 32;
    const int partition_span = (blocks + serial_partitions - 1) / serial_partitions;
    int partition_end = partition_span;
    float partial[RM][RN][4] = {};
    float total[RM][RN][4] = {};

    for (int first = 0; first < blocks; first += KB)
    {
        // Each payload has one decode owner. The same representation and LUTs
        // are used by dot4 prefill, so WMMA cannot invent a second codec.
        #pragma unroll
        for (int item = tid; item < KB * BN; item += 256)
        {
            const int stage = item / BN, local_column = item % BN;
            const int kb = first + stage;
            const int column = n0 + local_column;
            uint32_t raw[Traits::payload_dwords] = {};
            uint16_t primary = 0, secondary = 0;
            uint32_t minimum_pair = 0;
            const size_t index = static_cast<size_t>(kb) * N + column;
            if (column < N && kb < blocks)
            {
                // Request all source metadata beside the payload. Keeping the
                // loads ahead of decode and LDS stores hides their latency
                // without changing source scale conversion or arithmetic.
                load_payload<Traits::payload_bytes>(payload + index * Traits::payload_bytes, raw);
                primary = scales[index];
                if constexpr (Traits::needs_aux_compute) secondary = aux[index];
                if constexpr (Traits::is_dual_scale_asym) minimum_pair = emins[index];
            }
            int decoded[8];
            decode_block<FMT>(raw, decoded);
            NativeWmmaI32x4 decoded_low, decoded_high;
            #pragma unroll
            for (int g = 0; g < 4; ++g)
            {
                decoded_low[g] = decoded[g];
                decoded_high[g] = decoded[g + 4];
            }
            b_tile[stage][local_column ^ stage][(local_column >> 2) & 1] = decoded_low;
            b_tile[stage][local_column ^ stage][((local_column >> 2) & 1) ^ 1] = decoded_high;
            scale_tile[stage][local_column] = primary;
            if constexpr (Traits::needs_aux_compute)
                aux_tile[stage][local_column] = secondary;
            if constexpr (Traits::is_dual_scale_asym)
                emin_tile[stage][local_column] = minimum_pair;
            if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M)
                qh_tile[stage][local_column] = static_cast<uint16_t>(raw[1]);
        }
        #pragma unroll
        for (int item = tid; item < KB * BM * 2; item += 256)
        {
            const int local_row = item / (KB * 2);
            const int stage = (item % (KB * 2)) / 2, half = item % 2;
            const int kb = first + stage;
            const int row = m0 + local_row;
            // Adjacent lanes copy consecutive K16 vectors, covering a complete
            // 128-byte row span per eight lanes instead of striding all lanes by
            // K. The LDS and WMMA consumers retain their original ownership.
            NativeWmmaI32x4 values = {};
            if (row < M && kb < blocks)
            {
                const auto *source = reinterpret_cast<const NativeWmmaI32x4 *>(
                    a + static_cast<size_t>(row) * K + kb * 32);
                values = source[half];
            }
            a_tile[stage][local_row ^ stage][half ^ ((local_row >> 2) & 1)] = values;
            if constexpr (SUM_GROUPS > 0)
            {
                if constexpr (SUM_GROUPS == 1)
                {
                    int sum = 0;
                    #pragma unroll
                    for (int g = 0; g < 4; ++g)
                        sum = llaminar2::rocm_integer_contract::signedDot4(
                            values[g], 0x01010101, sum);
                    // Both half owners are adjacent lanes of the same wave.
                    // Integer summation is exact; only the low owner publishes.
                    sum += __shfl_xor(sum, 1, 32);
                    if (half == 0) sums[stage][0][local_row] = sum;
                }
                else
                {
                    #pragma unroll
                    for (int group = 0; group < SUM_GROUPS / 2; ++group)
                    {
                        int sum = 0;
                        #pragma unroll
                        for (int g = 0; g < 8 / SUM_GROUPS; ++g)
                            sum = llaminar2::rocm_integer_contract::signedDot4(
                                values[group * (8 / SUM_GROUPS) + g], 0x01010101, sum);
                        sums[stage][half * (SUM_GROUPS / 2) + group][local_row] = sum;
                    }
                }
            }
            if constexpr (ScaleMode == NativeWmmaActivationScale::Blockwise)
                if (half == 0)
                    activation_scales[stage][local_row] = row < M && kb < blocks
                        ? block_scales[static_cast<size_t>(row) * blocks + kb] : 1.0f;
        }
        __syncthreads();
        // Consume blocks in source order; batching only amortizes operand publication.
        #pragma unroll 1
        for (int stage = 0; stage < KB && first + stage < blocks; ++stage)
        {
            const int kb = first + stage;
            #pragma unroll
            for (int mi = 0; mi < RM; ++mi)
            {
                const int a_row = (wave / 2) * (BM / 2) + mi * 16 + lane % 16;
                const NativeWmmaI32x4 al = a_tile[stage][a_row ^ stage][(a_row >> 2) & 1];
                const NativeWmmaI32x4 ah = a_tile[stage][a_row ^ stage][((a_row >> 2) & 1) ^ 1];
                // These four row scales are shared by every column fragment.
                // Publish them to registers before the exact contribution
                // helpers so their rounding barriers cannot force LDS reloads.
                float row_scale[4];
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    if constexpr (ScaleMode == NativeWmmaActivationScale::Blockwise)
                        row_scale[e] = activation_scales[stage][
                            (wave / 2) * (BM / 2) + mi * 16 + e * 4 + lane / 16];
                    else
                        row_scale[e] = 1.0f;
                int row_correction[SUM_GROUPS ? SUM_GROUPS : 1][4];
                if constexpr (SUM_GROUPS > 0)
                    #pragma unroll
                    for (int group = 0; group < SUM_GROUPS; ++group)
                        #pragma unroll
                        for (int e = 0; e < 4; ++e)
                            row_correction[group][e] = sums[stage][group][
                                (wave / 2) * (BM / 2) + mi * 16 + e * 4 + lane / 16];
                #pragma unroll
                for (int ni = 0; ni < RN; ++ni)
                {
                    const int column = (wave % 2) * (BN / 2) + ni * 16 + lane % 16;
                    const NativeWmmaI32x4 bl = b_tile[stage][column ^ stage][(column >> 2) & 1];
                    const NativeWmmaI32x4 bh = b_tile[stage][column ^ stage][((column >> 2) & 1) ^ 1];
                    const NativeWmmaI32x4 zero = {};
                    const auto lo = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, al, true, bl, zero, false);
                    const auto hi = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, ah, true, bh, Traits::is_dual_scale ? zero : lo, false);
                    const uint16_t primary = scale_tile[stage][column];
                    const uint16_t secondary = Traits::needs_aux_compute ? aux_tile[stage][column] : uint16_t{0};
                    const auto ws = [&] {
                        if constexpr (Traits::is_dual_scale)
                            return rocm_native_quantised_load_dual_weight_scales<FMT>(primary, secondary);
                        else
                            return ROCmNativeQuantisedDualWeightScales{
                                rocm_native_quantised_load_single_weight_scale<FMT>(primary, secondary), 0.0f};
                    }();
                    const float minimum = Traits::is_asymmetric
                        ? __half2float(*reinterpret_cast<const __half *>(&secondary)) : 0.0f;
                    const uint32_t minimum_pair = Traits::is_dual_scale_asym ? emin_tile[stage][column] : 0;
                    const uint16_t qh = FMT == NATIVE_QUANTISED_GEMM_IQ1_M ? qh_tile[stage][column] : 0;
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        // Wave64 interleaves four row groups in accumulator VGPRs.
                        const float sa = row_scale[e];
                        float &acc = partial[mi][ni][e];
                        if constexpr (Traits::is_dual_scale)
                        {
                            if constexpr (Traits::is_dual_scale_asym)
                            {
                                const uint16_t ml = static_cast<uint16_t>(minimum_pair);
                                const uint16_t mh = static_cast<uint16_t>(minimum_pair >> 16);
                                acc = rocm_native_quantised_commit_corrected_dual_scale_block(acc, lo[e], hi[e],
                                    ws.low, ws.high, row_correction[0][e], row_correction[1][e],
                                    __half2float(*reinterpret_cast<const __half *>(&ml)),
                                    __half2float(*reinterpret_cast<const __half *>(&mh)), sa);
                            }
                            else if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M)
                            {
                                acc = rocm_native_quantised_commit_iq1_m_block(acc, lo[e], hi[e],
                                    row_correction[0][e], row_correction[1][e], row_correction[2][e], row_correction[3][e],
                                    qh & 0x0008 ? -0.125f : 0.125f, qh & 0x0080 ? -0.125f : 0.125f,
                                    qh & 0x0800 ? -0.125f : 0.125f, qh & 0x8000 ? -0.125f : 0.125f,
                                    ws.low, ws.high, sa);
                            }
                            else
                                acc = rocm_native_quantised_commit_dual_scale_block(acc, lo[e], hi[e], ws.low, ws.high, sa);
                        }
                        else if constexpr (Traits::is_asymmetric)
                            acc = rocm_native_quantised_commit_corrected_single_scale_block(acc, hi[e],
                                ws.low,
                                row_correction[0][e],
                                minimum, sa);
                        else
                            acc = rocm_native_quantised_commit_single_scale_block(acc, hi[e],
                                ws.low, sa);
                    }
                }
            }
            if (kb + 1 == partition_end || kb + 1 == blocks)
            {
                #pragma unroll
                for (int mi = 0; mi < RM; ++mi)
                    #pragma unroll
                    for (int ni = 0; ni < RN; ++ni)
                        #pragma unroll
                        for (int e = 0; e < 4; ++e)
                        {
                            const int row = m0 + (wave / 2) * (BM / 2) + mi * 16 + e * 4 + lane / 16;
                            float value = partial[mi][ni][e];
                            if constexpr (ScaleMode == NativeWmmaActivationScale::WholeRow)
                                if (row < M) value = __fmul_rn(value, row_scales[row]);
                            // Serial KB1 publishes the scaled value directly.
                            // Adding positive zero would erase negative zero
                            // from a rounded whole-row scale underflow.
                            total[mi][ni][e] = serial_partitions == 1 ? value :
                                __fadd_rn(total[mi][ni][e], value);
                            partial[mi][ni][e] = 0.0f;
                        }
                partition_end += partition_span;
            }
        }
        // Retire the complete batch before any decode owner overwrites its slot.
        __syncthreads();
    }
    #pragma unroll
    for (int mi = 0; mi < RM; ++mi)
        #pragma unroll
        for (int ni = 0; ni < RN; ++ni)
            #pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const int row = m0 + (wave / 2) * (BM / 2) + mi * 16 + e * 4 + lane / 16;
                const int column = n0 + (wave % 2) * (BN / 2) + ni * 16 + lane % 16;
                if (row < M && column < N) output[static_cast<size_t>(row) * N + column] = total[mi][ni][e];
            }
#endif
}
#endif
