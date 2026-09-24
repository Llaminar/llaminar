/**
 * @file Test__CUDAIQ4PackedLookup.cu
 * @brief Captured IQ4 lookup and packed-operand bytes against a scalar oracle.
 *
 * IQ4_NL and IQ4_XS use the same physical codebook after preparation. This
 * focused test exercises their actual shared word and vector helpers, not a
 * second GPU implementation that could drift alongside a projection kernel.
 * Every sixteen-bit selector occurs in both half-word positions, surrounded
 * by complementary or mixed codes. The host oracle indexes sixteen signed
 * values independently, so a shared serial/grouped lookup defect cannot hide.
 * No performance threshold belongs in this model-free integration gate.
 */
#include <gtest/gtest.h>
#include "kernels/cuda/gemm/CUDANativeVNNIDecodeCommon.cuh"

#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace
{
/** @brief Attribute a CUDA failure to this test instead of comparing stale bytes. */
void checked(cudaError_t status)
{
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

/** @brief Allocate before capture; the later execution owner joins before release. */
template<class T> class Storage final
{
public:
    /** @brief Reserve the complete fixture, including optional output guards. */
    explicit Storage(size_t count) { checked(cudaMalloc(&data, count * sizeof(T))); }
    /** @brief Release only after all references in retained graphs have retired. */
    ~Storage() { (void)cudaFree(data); }
    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;
    T *data = nullptr;
};

/** @brief Own an explicit stream and the graph that borrows the fixture storage. */
class Execution final
{
public:
    /** @brief The helper has no implicit/default stream or mutable device state. */
    Execution() { checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    /** @brief Terminal test cleanup joins before destroying captured pointers. */
    ~Execution()
    {
        (void)cudaStreamSynchronize(stream);
        if (executable) (void)cudaGraphExecDestroy(executable);
        if (graph) (void)cudaGraphDestroy(graph);
        (void)cudaStreamDestroy(stream);
    }
    Execution(const Execution &) = delete;
    Execution &operator=(const Execution &) = delete;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
};

/** @brief Both production interfaces must preserve the same packed dp4a operands. */
enum class LookupSurface { Word, Vector };

/**
 * @brief Invoke a production helper for one complete packed IQ4 block per thread.
 * @tparam Surface Shared per-word primitive or its vectorized block caller.
 * @param input Aligned immutable sixteen-byte packed blocks.
 * @param output Eight packed INT8 operand words per block, followed by guard bytes.
 * @param count Logical block count, deliberately not a multiple of blockDim.x.
 */
template<LookupSurface Surface>
__global__ void decode(const uint4 *input, uint32_t *output, unsigned count)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int32_t groups[8];
    if constexpr (Surface == LookupSurface::Vector)
        llaminar2::cuda_native_vnni::decode_groups_vec<4>(
            reinterpret_cast<const uint8_t *>(input + index), groups);
    else
    {
        const uint4 packed = input[index];
        const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (int word = 0; word < 4; ++word)
        {
            uint32_t low, high;
            llaminar2::cuda_native_vnni::iq4nl_decode_word(words[word], low, high);
            groups[word] = static_cast<int32_t>(low);
            groups[word + 4] = static_cast<int32_t>(high);
        }
    }
#pragma unroll
    for (int group = 0; group < 8; ++group)
        output[static_cast<size_t>(index) * 8 + group] = static_cast<uint32_t>(groups[group]);
}

TEST(CUDAIQ4PackedLookup, CapturedWordAndVectorMatchIndependentNibbleOracle)
{
    constexpr unsigned count = 65537; // All half-word selectors plus a partial CTA.
    constexpr size_t guard_bytes = 64;
    constexpr size_t output_bytes = static_cast<size_t>(count) * 32 + guard_bytes;
    constexpr std::array<int8_t, 16> table = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    std::vector<uint4> input(count);
    std::vector<uint8_t> expected(output_bytes, 0x7e), actual(output_bytes);
    for (unsigned index = 0; index < count; ++index)
    {
        const uint32_t selector = index & 0xffffu;
        const uint32_t opposite = selector ^ 0xffffu;
        const uint32_t mixed = index * 2654435761u;
        input[index] = make_uint4(selector | (opposite << 16), (selector << 16) | opposite,
                                 mixed, (mixed << 16) | (mixed >> 16));
        std::array<uint8_t, 16> packed{};
        std::memcpy(packed.data(), &input[index], packed.size());
        for (size_t byte = 0; byte < packed.size(); ++byte)
        {
            expected[static_cast<size_t>(index) * 32 + byte] =
                static_cast<uint8_t>(table[packed[byte] & 15]);
            expected[static_cast<size_t>(index) * 32 + 16 + byte] =
                static_cast<uint8_t>(table[packed[byte] >> 4]);
        }
    }
    Storage<uint4> device_input(count);
    Storage<uint8_t> word_output(output_bytes), vector_output(output_bytes);
    Execution execution; // Its join/destruction precedes all referenced storage.
    const auto stream = execution.stream;
    checked(cudaMemcpyAsync(device_input.data, input.data(), input.size() * sizeof(uint4),
                            cudaMemcpyHostToDevice, stream));
    checked(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    // Poison inside the retained graph: each replay must republish every byte,
    // while the tail guards prove the partial CTA never writes outside count.
    checked(cudaMemsetAsync(word_output.data, 0x7e, output_bytes, stream));
    checked(cudaMemsetAsync(vector_output.data, 0x7e, output_bytes, stream));
    decode<LookupSurface::Word><<<(count + 255) / 256, 256, 0, stream>>>(
        device_input.data, reinterpret_cast<uint32_t *>(word_output.data), count);
    decode<LookupSurface::Vector><<<(count + 255) / 256, 256, 0, stream>>>(
        device_input.data, reinterpret_cast<uint32_t *>(vector_output.data), count);
    checked(cudaGetLastError());
    checked(cudaStreamEndCapture(stream, &execution.graph));
    checked(cudaGraphInstantiate(&execution.executable, execution.graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 20; ++replay)
    {
        checked(cudaGraphLaunch(execution.executable, stream));
        for (const auto *output : {word_output.data, vector_output.data})
        {
            checked(cudaMemcpyAsync(actual.data(), output, output_bytes, cudaMemcpyDeviceToHost, stream));
            checked(cudaStreamSynchronize(stream)); // Independent test oracle, not a hot-path edge.
            ASSERT_EQ(std::memcmp(expected.data(), actual.data(), output_bytes), 0)
                << "replay=" << replay << " surface=" << (output == word_output.data ? "word" : "vector");
        }
    }
}
} // namespace
