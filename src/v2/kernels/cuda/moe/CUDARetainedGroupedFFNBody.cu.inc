/**
 * @file CUDARetainedGroupedFFNBody.cu.inc
 * @brief CUDA grouped packet math with device-sized grids and no global Gate/Up partials.
 *
 * One CTA owns a complete 32-column packet and all of its serial K partitions.
 * It folds them in place and publishes Q8 directly to the corresponding Down
 * row. The original canonical consumer owns route reduction and any transport.
 */

#if CUDART_VERSION >= 12080
    constexpr int retained_grouped_down_threads = 128;
    static_assert(kQFNDecodeGateUpK % retained_grouped_down_threads == 0);

    /**
     * @brief Apply one device-owned live count to both arithmetic phases.
     * @param gate_up Node that publishes Q8 packets.
     * @param down Node that consumes the published packets.
     * @param count Original device plan's live row count.
     * @param capacity Admitted capacity used only to validate the plan.
     * @param packets Complete Gate/Up output packets per compact row.
     */
    __global__ void sizeRetainedQFNGroupedFFN(cudaGraphDeviceNode_t gate_up, cudaGraphDeviceNode_t down,
        const int32_t* count, int32_t capacity, int packets)
    {
        const int live = *count;
        if (live < 0 || live > capacity)
            __trap();
        if (live &&
            (cudaGraphKernelNodeSetGridDim(gate_up, dim3(packets, live)) != cudaSuccess ||
             cudaGraphKernelNodeSetGridDim(down,
                 dim3(kQFNDecodeGateUpK / retained_grouped_down_threads, live)) != cudaSuccess))
            __trap();
        if (cudaGraphKernelNodeSetEnabled(gate_up, live != 0) != cudaSuccess ||
            cudaGraphKernelNodeSetEnabled(down, live != 0) != cudaSuccess)
            __trap();
    }

    /**
     * @brief Publish one serial-order SwiGLU packet using only CTA-local FP32 partials.
     * @tparam Codebook Prepared physical Gate/Up decoder.
     * @tparam Width Exact local FFN width.
     * @param rows Prevalidated compact local work supplied by the device plan.
     */
    template<uint8_t Codebook, int Width>
    __global__ __launch_bounds__(kThreads) void retainedQFNGroupedGateUp(
        const llaminar2::cuda::moe::RetainedGroupedFFNRow* __restrict__ rows)
    {
        constexpr int partitions = llaminar2::CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
        __shared__ float partials[2 * partitions][32];
        const auto& row = rows[blockIdx.y];
        retainedGateUpPacket<Codebook, Width>(row.weights, blockIdx.x, row.down_codes, row.down_scales,
            row.input_codes, row.input_scales, partials);
    }

    /**
     * @brief Publish one ordered Down tile to its exact original contribution.
     * @tparam Codebook Prepared physical Down decoder.
     * @tparam Width Exact local source width.
     * @tparam Role Original routed or shared Down arithmetic.
     * @param rows Prevalidated compact work after its Q8 packet producer.
     */
    template<uint8_t Codebook, int Width, llaminar2::cuda::moe::RetainedGroupedFFNRole Role>
    __global__ __launch_bounds__(retained_grouped_down_threads) void retainedQFNGroupedDown(
        const llaminar2::cuda::moe::RetainedGroupedFFNRow* __restrict__ rows)
    {
        constexpr bool direct = Role == llaminar2::cuda::moe::RetainedGroupedFFNRole::Routed &&
            Width / 32 <= llaminar2::CUDAMoEBatchInvariantPolicy::down_k_partitions;
        const auto& row = rows[blockIdx.y];
        const int column = blockIdx.x * retained_grouped_down_threads + threadIdx.x;
        row.contribution[column] = retainedDown<Codebook, direct, Width>(
            row.weights.down, column, row.down_codes, row.down_scales);
    }
#endif
