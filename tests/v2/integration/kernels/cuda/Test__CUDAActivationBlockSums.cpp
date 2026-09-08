/**
 * @file Test__CUDAActivationBlockSums.cpp
 * @brief Captured CUDA quantizer publication across physical row pitches.
 *
 * Both production producers must retain serial-row activation/scale bytes and
 * publish exact INT32 block sums in block-major order. Every captured replay
 * poisons outputs and guards. Serial-row replay is an independent diagnostic
 * oracle only; production executes the grouped producer once. The existing
 * all-format prefill gate separately proves every consuming weight format.
 */
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "backends/cuda/CUDAGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

extern "C"
{
    bool cudaQuantGemm_quantizeActivationsBlockwiseWithSums(
        const float *, int8_t *, float *, int32_t *, int, int, int, void *);
    bool cudaOps_fused_swiglu_quantize_blockwise_with_sums(
        const float *, const float *, int8_t *, float *, int32_t *,
        int, int, int, void *);
}

namespace
{
/** @brief Surface the originating CUDA failure before inspecting output data. */
void checked(cudaError_t result)
{
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

/** @brief Test infrastructure storage; allocation never occurs during capture. */
template<class T> class Storage final
{
public:
    /** @brief Allocate exactly the requested extent, including explicit guards. */
    explicit Storage(size_t count) : count(count) { checked(cudaMalloc(&data, count * sizeof(T))); }
    /** @brief Free only after the execution scope has retired every graph. */
    ~Storage() { (void)cudaFree(data); }
    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;

    /** @brief Stage fixture inputs before recording their consumer graph. */
    void upload(const std::vector<T> &values, cudaStream_t stream)
    {
        if (!stream || values.size() != count) throw std::logic_error("invalid fixture upload");
        checked(cudaMemcpyAsync(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    T *data = nullptr;
    const size_t count;
};

/** @brief Last-declared execution owner, so native graphs retire before storage. */
class Execution final
{
public:
    /** @brief Create the exact non-default stream used for all fixture work. */
    Execution() { checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    /** @brief Join terminal work before destroying graph and stream resources. */
    ~Execution()
    {
        (void)cudaStreamSynchronize(stream);
        graphs = {};
        (void)cudaStreamDestroy(stream);
    }
    Execution(const Execution &) = delete;
    Execution &operator=(const Execution &) = delete;
    cudaStream_t stream = nullptr;
    std::array<std::unique_ptr<llaminar2::CUDAGraphCapture>, 2> graphs;
};

/** @brief The two production publishers of the shared CUDA sum layout. */
enum class Producer : uint8_t { Quantize, SwiGLUQuantize };

/**
 * @brief Submit one production producer without a test-side replacement kernel.
 * @param producer Ordinary or fused SwiGLU quantization.
 * @param input Row-major input, or gate projection for the fused producer.
 * @param up Row-major up projection, used only by the fused producer.
 * @param quantized Row-major INT8 output.
 * @param scales Row-major FP32 block scales.
 * @param sums Block-major INT32 block sums.
 * @param rows Physical row count and sum pitch for this launch.
 * @param k Input width divisible by 32.
 * @param stream Exact recording stream.
 */
void submit(Producer producer, const float *input, const float *up,
            int8_t *quantized, float *scales, int32_t *sums,
            int rows, int k, cudaStream_t stream)
{
    const bool ok = producer == Producer::Quantize
        ? cudaQuantGemm_quantizeActivationsBlockwiseWithSums(
              input, quantized, scales, sums, rows, k, 0, stream)
        : cudaOps_fused_swiglu_quantize_blockwise_with_sums(
              input, up, quantized, scales, sums, rows, k, 0, stream);
    if (!ok) throw std::runtime_error("production quantizer launch failed");
}

/**
 * @brief Compare all captured producer bytes through twenty guarded replays.
 * @param producer Production operation under test.
 * @param rows Physical row count, including odd and padded-bucket witnesses.
 * @param k Input width, including an odd number of quantization blocks.
 */
void verify(Producer producer, int rows, int k)
{
    const size_t elements = static_cast<size_t>(rows) * k;
    const size_t blocks = elements / 32;
    const int blocks_per_row = k / 32;
    std::vector<float> input(elements), up(elements);
    for (size_t i = 0; i < elements; ++i)
    {
        input[i] = (static_cast<int>((i * 173 + i / 32 * 97) % 4093) - 2046) / 257.0f;
        up[i] = (static_cast<int>((i * 53 + i / 32 * 71) % 4093) - 2046) / 1025.0f;
    }
    Storage<float> device_input(elements), device_up(elements);
    Storage<int8_t> actual_q(elements + 32), expected_q(elements + 32);
    Storage<float> actual_scales(blocks + 8), expected_scales(blocks + 8);
    Storage<int32_t> actual_sums(blocks + 8), expected_sums(blocks + 8);
    Execution execution;
    const auto stream = execution.stream;
    device_input.upload(input, stream);
    device_up.upload(up, stream);
    checked(cudaStreamSynchronize(stream));

    for (int path = 0; path < 2; ++path)
    {
        auto &graph = execution.graphs[path];
        graph = std::make_unique<llaminar2::CUDAGraphCapture>(stream, 0);
        llaminar2::ScopedBackendGraphCapture recording(*graph, "activation block sum publication");
        ASSERT_TRUE(recording.begin());
        auto *q = path == 0 ? expected_q.data : actual_q.data;
        auto *scales = path == 0 ? expected_scales.data : actual_scales.data;
        auto *sums = path == 0 ? expected_sums.data : actual_sums.data;
        checked(cudaMemsetAsync(q, 0x5a, elements + 32, stream));
        checked(cudaMemsetAsync(scales, 0x5a, (blocks + 8) * sizeof(float), stream));
        checked(cudaMemsetAsync(sums, 0x5a, (blocks + 8) * sizeof(int32_t), stream));
        if (path == 0)
        {
            // Independent M=1 launches have an unambiguous contiguous sum
            // row. These launches exist only in the captured test oracle.
            for (int row = 0; row < rows; ++row)
                submit(producer, device_input.data + static_cast<size_t>(row) * k,
                       device_up.data + static_cast<size_t>(row) * k,
                       q + static_cast<size_t>(row) * k,
                       scales + static_cast<size_t>(row) * blocks_per_row,
                       sums + static_cast<size_t>(row) * blocks_per_row,
                       1, k, stream);
        }
        else
            submit(producer, device_input.data, device_up.data, q, scales, sums, rows, k, stream);
        recording.finish();
        ASSERT_TRUE(graph->instantiate());
    }

    std::vector<int8_t> q(elements + 32), oracle_q(elements + 32);
    std::vector<float> scales(blocks + 8), oracle_scales(blocks + 8);
    std::vector<int32_t> sums(blocks + 8), oracle_sums(blocks + 8), reference_sums(blocks);
    for (int replay = 0; replay < 20; ++replay)
    {
        ASSERT_TRUE(execution.graphs[0]->launch());
        ASSERT_TRUE(execution.graphs[1]->launch());
        checked(cudaMemcpyAsync(q.data(), actual_q.data, q.size(), cudaMemcpyDeviceToHost, stream));
        checked(cudaMemcpyAsync(oracle_q.data(), expected_q.data, oracle_q.size(), cudaMemcpyDeviceToHost, stream));
        checked(cudaMemcpyAsync(scales.data(), actual_scales.data, scales.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        checked(cudaMemcpyAsync(oracle_scales.data(), expected_scales.data, oracle_scales.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        checked(cudaMemcpyAsync(sums.data(), actual_sums.data, sums.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        checked(cudaMemcpyAsync(oracle_sums.data(), expected_sums.data, oracle_sums.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        checked(cudaStreamSynchronize(stream));
        ASSERT_EQ(std::memcmp(q.data(), oracle_q.data(), q.size()), 0) << "replay=" << replay;
        ASSERT_EQ(std::memcmp(scales.data(), oracle_scales.data(), scales.size() * sizeof(float)), 0);
        if (replay == 0)
            for (size_t block = 0; block < blocks; ++block)
                for (int lane = 0; lane < 32; ++lane)
                    reference_sums[block] += oracle_q[block * 32 + lane];
        for (int row = 0; row < rows; ++row)
            for (int block = 0; block < blocks_per_row; ++block)
            {
                const size_t linear = static_cast<size_t>(row) * blocks_per_row + block;
                ASSERT_EQ(oracle_sums[linear], reference_sums[linear]);
                ASSERT_EQ(sums[static_cast<size_t>(block) * rows + row], reference_sums[linear])
                    << "row=" << row << " block=" << block << " replay=" << replay;
            }
        for (size_t i = elements; i < q.size(); ++i) ASSERT_EQ(q[i], 0x5a);
        for (size_t i = blocks; i < sums.size(); ++i)
        {
            ASSERT_EQ(sums[i], 0x5a5a5a5a);
            ASSERT_EQ(oracle_sums[i], 0x5a5a5a5a);
            ASSERT_EQ(std::bit_cast<uint32_t>(scales[i]), 0x5a5a5a5aU);
        }
    }
}
} // namespace

/** @brief Prove both publishers around serial, verifier, tile and bucket boundaries. */
TEST(CUDAActivationBlockSums, CapturedProducersPublishExactBlockMajorSums)
{
    int devices = 0;
    checked(cudaGetDeviceCount(&devices));
    ASSERT_GT(devices, 0) << "explicit CUDA integration requires a real device";
    checked(cudaSetDevice(0));
    for (const auto producer : {Producer::Quantize, Producer::SwiGLUQuantize})
        for (const int rows : {1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513})
            for (const int k : {32, 160, 5120})
            {
                SCOPED_TRACE(::testing::Message() << "producer=" << static_cast<int>(producer)
                                                << " rows=" << rows << " K=" << k);
                verify(producer, rows, k);
                if (::testing::Test::HasFatalFailure()) return;
            }
}
