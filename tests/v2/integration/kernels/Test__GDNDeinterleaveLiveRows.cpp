/**
 * @file Test__GDNDeinterleaveLiveRows.cpp
 * @brief Captured CUDA/HIP proof that GDN transforms touch only live rows.
 *
 * One immutable graph per geometry borrows the production length array. Raw
 * FP32 words, modular head repeats, unequal/empty requests and shrinking replay
 * prove exact copies, unchanged physical planes and untouched inactive bytes.
 * The fixture uses TransferEngine-owned storage and an exact GPU worker stream.
 */
#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "transfer/TransferEngine.h"
#ifdef HAVE_CUDA
#include "kernels/cuda/gdn/CUDAGatedDeltaNet.h"
#endif
#ifdef HAVE_ROCM
#include "kernels/rocm/gdn/ROCmGatedDeltaNet.h"
#endif
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Fail the worker future rather than continuing with invalid pointers. */
    void require(bool condition, const char *message)
    {
        if (!condition) throw std::runtime_error(message);
    }

    /** @brief Persistent test buffers outlive the one retained executable. */
    struct Storage final
    {
        DeviceId device;
        IBackend *backend;
        void *stream;
        std::vector<std::shared_ptr<DeviceTransferBuffer>> owners;
        std::unique_ptr<IGPUGraphCapture> graph;

        /** @brief Join the test terminal before releasing any captured pointer. */
        ~Storage()
        {
            (void)backend->synchronizeStream(stream, device.ordinal);
            graph.reset();
        }

        /** @return TransferEngine-owned setup storage; never called in capture/replay. */
        template <typename T> T *allocate(std::size_t elements)
        {
            auto owner = TransferEngine::instance().allocateDeviceTransferBuffer(elements * sizeof(T), device);
            require(owner && owner->isBound(), "GDN test allocation failed");
            auto *pointer = static_cast<T *>(owner->mutableDeviceData());
            owners.push_back(std::move(owner));
            return pointer;
        }

        /** @brief Publish test bytes outside the captured transform. */
        template <typename T> void upload(T *pointer, const std::vector<T> &values)
        {
            require(backend->hostToDeviceOnStream(pointer, values.data(), values.size() * sizeof(T),
                device.ordinal, stream), "GDN test upload failed");
            require(backend->synchronizeStream(stream, device.ordinal), "GDN test setup join failed");
        }

        /** @return Terminal bytes; synchronization is a test oracle boundary only. */
        std::vector<std::uint32_t> download(float *pointer, std::size_t elements)
        {
            std::vector<std::uint32_t> words(elements);
            require(backend->deviceToHostOnStream(words.data(), pointer, words.size() * sizeof(std::uint32_t),
                device.ordinal, stream), "GDN test download failed");
            require(backend->synchronizeStream(stream, device.ordinal), "GDN test terminal join failed");
            return words;
        }
    };

    /** @brief Freeze one native modular head mapping, independent of live counts. */
    struct Geometry final { int key_heads, value_heads, key_width, value_width, offset; };

    /**
     * @brief Compare production captured copies to an independent raw-word oracle.
     * @param device Exact native backend endpoint, not an emulated path.
     *
     * Counters never control the graph. Only the existing device lengths change
     * between replays; plane pointers and physical request coordinates stay fixed.
     */
    void prove(DeviceId device)
    {
        auto *backend = getBackendFor(device);
        if (!backend || backend->deviceCount() <= device.ordinal)
            GTEST_SKIP() << "Requested GPU backend is unavailable";
        auto &worker = GPUDeviceContextPool::instance().getContext(device);
        worker.submitAsync([&]
        {
            for (const Geometry geometry : std::array{
                Geometry{16, 40, 128, 128, 0}, Geometry{8, 20, 128, 64, 3},
                Geometry{3, 8, 5, 7, -5}, Geometry{16, 16, 128, 128, 0}})
                for (const int requests : {1, 2})
                {
                    constexpr int stride = 16;
                    constexpr std::uint32_t poison = 0xa5a5a5a5u;
                    const int matrix_rows = requests * stride;
                    const int source_query = geometry.key_heads * geometry.key_width;
                    const int query_width = geometry.value_heads * geometry.key_width;
                    const int value_width = geometry.value_heads * geometry.value_width;
                    const int source_width = 2 * source_query + value_width;
                    const std::size_t query_elements = std::size_t(matrix_rows) * query_width;
                    const std::size_t value_elements = std::size_t(matrix_rows) * value_width;
                    const std::size_t output_elements = 2 * query_elements + value_elements;
                    Storage storage{device, backend, ExplicitGPUStream(worker.defaultStream()).get(), {}, {}};
                    auto *input = storage.allocate<float>(std::size_t(matrix_rows) * source_width);
                    auto *output = storage.allocate<float>(output_elements);
                    auto *lengths = storage.allocate<std::int32_t>(requests);
                    std::vector<std::uint32_t> input_words(std::size_t(matrix_rows) * source_width);
                    for (std::size_t index = 0; index < input_words.size(); ++index)
                        input_words[index] = index % 31 == 0 ? 0x7fc12345u :
                            index % 29 == 0 ? 0x80000000u : static_cast<std::uint32_t>(index) ^ 0x3f000000u;
                    storage.upload(reinterpret_cast<std::uint32_t *>(input), input_words);
                    const auto rows = DeviceRequestRowRanges::deviceCounted(requests, stride, lengths);
                    std::unique_ptr<ITensorGatedDeltaNet> kernel;
#ifdef HAVE_CUDA
                    if (device.is_cuda()) kernel = std::make_unique<CUDAGatedDeltaNet>(device.ordinal);
#endif
#ifdef HAVE_ROCM
                    if (device.is_rocm()) kernel = std::make_unique<ROCmGatedDeltaNet>(device.ordinal);
#endif
                    require(bool(kernel), "GDN backend implementation missing");
                    float *q = nullptr, *k = nullptr, *v = nullptr;
                    kernel->bindDeinterleaveWorkspace(output, output_elements);
                    EXPECT_FALSE(kernel->deinterleave_qkv_device(input, q, k, v, rows,
                        geometry.key_heads, geometry.value_heads, geometry.key_width,
                        geometry.value_width, geometry.offset)); // No stream is never legal.
                    kernel->setGPUStream(storage.stream);
                    storage.graph = worker.createGraphCapture(storage.stream);
                    require(bool(storage.graph), "GDN graph owner missing");
                    {
                        ScopedBackendGraphCapture capture(worker, *storage.graph, "live GDN deinterleave");
                        require(capture.begin(), "GDN graph recording failed");
                        require(kernel->deinterleave_qkv_device(input, q, k, v, rows,
                            geometry.key_heads, geometry.value_heads, geometry.key_width,
                            geometry.value_width, geometry.offset), "GDN transform recording failed");
                        capture.finish();
                    }
                    EXPECT_EQ(q, output);
                    EXPECT_EQ(k, output + query_elements);
                    EXPECT_EQ(v, output + 2 * query_elements);
                    // This transform is exactly one kernel, not a hidden
                    // transfer, allocation or host-callback sequence. Inspect
                    // compiler/native metadata before executing any replay.
                    EXPECT_EQ(storage.graph->nodeCount(), 1u);
                    std::vector<GPUGraphKernelNodeInfo> nodes;
                    std::string inspection_error;
                    if (!storage.graph->inspectKernelNodes(nodes, &inspection_error))
                        throw std::runtime_error(inspection_error);
                    ASSERT_EQ(nodes.size(), 1u);
                    EXPECT_TRUE(nodes.front().valid());
                    EXPECT_EQ(nodes.front().grid_y, static_cast<std::uint32_t>(requests));
                    EXPECT_EQ(nodes.front().local_memory_bytes_per_thread, 0u);
                    std::cout << "GDN_DEINTERLEAVE_RESOURCES," << device.toString() << ','
                        << requests << ',' << geometry.value_heads << ',' << geometry.key_width << ','
                        << geometry.value_width << ',' << nodes.front().grid_x << ','
                        << nodes.front().registers_per_thread << ','
                        << nodes.front().max_active_blocks_per_sm << '\n';
                    require(storage.graph->instantiate(), "GDN graph instantiation failed");
                    for (const auto pair : std::array{
                        std::array{16, 16}, std::array{3, 4}, std::array{0, 16},
                        std::array{16, 0}, std::array{1, 2}, std::array{0, 0},
                        std::array{16, 16}, std::array{2, 1}})
                    {
                        std::vector<std::int32_t> live(pair.begin(), pair.begin() + requests);
                        storage.upload(lengths, live);
                        storage.upload(reinterpret_cast<std::uint32_t *>(output),
                            std::vector<std::uint32_t>(output_elements, poison));
                        require(storage.graph->launch(), "GDN retained replay failed");
                        const auto actual = storage.download(output, output_elements);
                        std::vector<std::uint32_t> expected(output_elements, poison);
                        for (int request = 0; request < requests; ++request)
                            for (int local_row = 0; local_row < live[request]; ++local_row)
                            {
                                const int row = request * stride + local_row;
                                const auto source = std::size_t(row) * source_width;
                                for (int column = 0; column < query_width; ++column)
                                {
                                    int head = (column / geometry.key_width + geometry.offset) % geometry.key_heads;
                                    if (head < 0) head += geometry.key_heads;
                                    const int source_column = head * geometry.key_width + column % geometry.key_width;
                                    const auto destination = std::size_t(row) * query_width + column;
                                    expected[destination] = input_words[source + source_column];
                                    expected[query_elements + destination] = input_words[source + source_query + source_column];
                                }
                                std::copy_n(input_words.begin() + source + 2 * source_query, value_width,
                                    expected.begin() + 2 * query_elements + std::size_t(row) * value_width);
                            }
                        EXPECT_EQ(actual, expected) << "backend=" << device.toString()
                            << " requests=" << requests << " live=" << live.front()
                            << " heads=" << geometry.value_heads << " offset=" << geometry.offset;
                    }
                    storage.graph.reset();
                    kernel->clearGPUStreamBinding();
                }
        }).get();
    }
}

#ifdef HAVE_CUDA
TEST(GDNDeinterleaveLiveRows, CUDA) { prove(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
TEST(GDNDeinterleaveLiveRows, ROCm) { prove(DeviceId::rocm(0)); }
#endif
