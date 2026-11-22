/**
 * @file CUDADeviceSizedWorkWitness.cu
 * @brief Native witness for device-owned grid dimensions and producer ordering.
 *
 * The planner schedules two dependent packet phases. Each phase writes its
 * hardware-observed grid width so the host test can detect inactive CTAs,
 * stale positive grids, or Down execution before its original producer.
 */
#include <cuda_runtime.h>

#if CUDART_VERSION >= 12080
namespace
{
/** @brief Publish exactly the current number of rows to both native nodes. */
__global__ void publishLiveGrids(const int* count, cudaGraphDeviceNode_t producer,
    cudaGraphDeviceNode_t consumer, int* result)
{
    const int live = *count;
    const cudaGraphDeviceNode_t nodes[]{producer, consumer};
    cudaError_t code = cudaSuccess;
    for (const auto node : nodes)
    {
        if (live)
        {
            code = cudaGraphKernelNodeSetGridDim(node, dim3(live));
            if (code != cudaSuccess) break;
        }
        code = cudaGraphKernelNodeSetEnabled(node, live > 0);
        if (code != cudaSuccess) break;
    }
    *result = static_cast<int>(code);
}

/** @brief Each submitted CTA publishes its device-observed extent. */
__global__ void produceExtent(int* producer) { producer[blockIdx.x] = gridDim.x; }

/** @brief Consume the preceding live producer and record this phase's extent. */
__global__ void consumeExtent(const int* producer, int* output)
{
    output[blockIdx.x] = producer[blockIdx.x] * static_cast<int>(gridDim.x);
}
}

/** @return Native producer function submitted by the production capture helper. */
const void* cudaDeviceSizedProducerWitness() { return reinterpret_cast<const void*>(produceExtent); }

/** @return Native consumer function submitted by the production capture helper. */
const void* cudaDeviceSizedConsumerWitness() { return reinterpret_cast<const void*>(consumeExtent); }

/** @brief Record the live publisher on the original exact capture stream. */
bool cudaDeviceSizedPlannerWitness(const int* count, cudaGraphDeviceNode_t producer,
    cudaGraphDeviceNode_t consumer, int* result, cudaStream_t stream)
{
    publishLiveGrids<<<1, 1, 0, stream>>>(count, producer, consumer, result);
    return cudaPeekAtLastError() == cudaSuccess;
}
#endif
