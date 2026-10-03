/**
 * @file Test__NativeReduceScatter.cpp
 * @brief Captured CUDA/ROCm proof of exact-stream, out-of-place native reduce-scatter.
 *
 * Reverse physical GPU order so participant indices cannot masquerade as
 * ordinals. Retain the graph across twenty input changes, check every native
 * element format and odd counts, and verify unused receive capacity stays
 * poisoned. This proves the public primitive, not a model performance claim.
 */
#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/LocalTPContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/FP16Utils.h"
#include "tensors/Tensors.h"
#include "utils/BFloat16.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Throw while fixture owners are still available for safe retirement. */
    void require(bool condition, const char *message)
    { if (!condition) throw std::runtime_error(message); }

    /** @brief Exact small integer, distinct by source coordinate, rank and replay. */
    int value(int rank, int request, std::size_t coordinate)
    { return static_cast<int>((coordinate * 13 + rank * 7 + request * 3) % 17) - 8; }

    /** @return Physical bytes per element, independent of the backing tensor's type. */
    std::size_t bytes(CollectiveDataType type)
    {
        if (type == CollectiveDataType::INT8) return 1;
        return type == CollectiveDataType::FLOAT32 || type == CollectiveDataType::INT32 ? 4 : 2;
    }

    /** @brief Encode exactly representable test integers without unaligned typed access. */
    void encode(std::uint8_t *target, CollectiveDataType type, int integer)
    {
        switch (type)
        {
        case CollectiveDataType::FLOAT32:
        { const auto v = static_cast<float>(integer); std::memcpy(target, &v, sizeof(v)); break; }
        case CollectiveDataType::FLOAT16:
        { const auto v = fp32_to_fp16(static_cast<float>(integer)); std::memcpy(target, &v, sizeof(v)); break; }
        case CollectiveDataType::BFLOAT16:
        { const auto v = bfloat16::from_float(static_cast<float>(integer)).data; std::memcpy(target, &v, sizeof(v)); break; }
        case CollectiveDataType::INT32:
        { const auto v = static_cast<std::int32_t>(integer); std::memcpy(target, &v, sizeof(v)); break; }
        case CollectiveDataType::INT8:
        { const auto v = static_cast<std::int8_t>(integer); std::memcpy(target, &v, sizeof(v)); break; }
        }
    }

    /** @brief Run the public native primitive through retained graphs on two real GPUs. */
    void verify(DeviceId first)
    {
        auto *backend = getBackendFor(first);
        ASSERT_NE(backend, nullptr);
        ASSERT_GE(backend->deviceCount(), 2);
        const std::array devices{first.is_cuda() ? DeviceId::cuda(1) : DeviceId::rocm(1), first};
        auto tp = createLocalTPContext({GlobalDeviceAddress::fromLocalDeviceId(devices[0]),
            GlobalDeviceAddress::fromLocalDeviceId(devices[1])}, {},
            first.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        ASSERT_NE(tp, nullptr);
        std::barrier rendezvous(2);
        std::array<std::exception_ptr, 2> errors{};
        std::array<std::thread, 2> workers;
        for (int participant = 0; participant < 2; ++participant)
        {
            workers[participant] = std::thread([&, participant] {
                try
                {
                    auto &gpu = GPUDeviceContextPool::instance().getContext(devices[participant]);
                    gpu.submitAndWait([&] {
                        const auto device = devices[participant];
                        const int ordinal = device.gpu_ordinal();
                        void *const stream = gpu.getOrCreateAuxiliaryStream("native_reduce_scatter_regression");
                        FP32Tensor input({2 * 4096 + 16}, device), output({4096 + 16}, device);
                        std::fill_n(input.mutable_data(), input.numel(), 0.0f);
                        std::fill_n(output.mutable_data(), output.numel(), 0.0f);
                        require(input.ensureOnDevice(device, stream) && output.ensureOnDevice(device, stream), "fixture upload");
                        require(gpu.synchronizeStreamChecked(stream), "fixture preparation");
                        auto capture = gpu.createGraphCapture(stream);
                        try
                        {
                            for (auto type : {CollectiveDataType::FLOAT32, CollectiveDataType::FLOAT16,
                                    CollectiveDataType::BFLOAT16, CollectiveDataType::INT32, CollectiveDataType::INT8})
                                for (std::size_t count : {1u, 7u, 16u, 257u, 4096u})
                                {
                                    rendezvous.arrive_and_wait();
                                    ScopedBackendGraphCapture recording(gpu, *capture, "native_reduce_scatter");
                                    require(recording.begin(), "begin native reduce-scatter graph");
                                    require(tp->reduceScatterRawOnStream(input.gpu_data_ptr(), output.gpu_data_ptr(),
                                        count, type, participant, stream, "reduce_scatter_regression"), "record native sum/scatter");
                                    recording.finish();
                                    require(capture->instantiate(), "instantiate native sum/scatter");
                                    std::vector<std::uint8_t> sent(input.size_bytes()), received(output.size_bytes());
                                    for (int request = 1; request <= 20; ++request)
                                    {
                                        std::fill(sent.begin(), sent.end(), 0x5a);
                                        std::fill(received.begin(), received.end(), 0xa5);
                                        for (std::size_t element = 0; element < 2 * count; ++element)
                                            encode(sent.data() + element * bytes(type), type, value(participant, request, element));
                                        require(backend->hostToDeviceOnStream(input.gpu_data_ptr(), sent.data(), sent.size(), ordinal, stream), "reset input");
                                        require(backend->hostToDeviceOnStream(output.gpu_data_ptr(), received.data(), received.size(), ordinal, stream), "poison output");
                                        rendezvous.arrive_and_wait();
                                        require(capture->launch(), "replay native sum/scatter");
                                        require(backend->deviceToHostOnStream(received.data(), output.gpu_data_ptr(), received.size(), ordinal, stream), "observe output");
                                        require(gpu.synchronizeStreamChecked(stream), "terminal observation");
                                        std::vector<std::uint8_t> expected(output.size_bytes(), 0xa5);
                                        for (std::size_t element = 0; element < count; ++element)
                                        {
                                            const auto coordinate = participant * count + element;
                                            encode(expected.data() + element * bytes(type), type,
                                                value(0, request, coordinate) + value(1, request, coordinate));
                                        }
                                        require(received == expected, "native scatter arithmetic, rank slice or guard bytes differ");
                                    }
                                    rendezvous.arrive_and_wait();
                                    capture->reset();
                                }
                        }
                        catch (...)
                        {
                            // Release peers from an in-flight native wait before
                            // this task unwinds its graph and persistent storage.
                            tp->requestAbort();
                            throw;
                        }
                    });
                }
                catch (...)
                {
                    errors[participant] = std::current_exception();
                    tp->requestAbort();
                    rendezvous.arrive_and_drop();
                }
            });
        }
        for (auto &worker : workers) worker.join();
        for (const auto &error : errors)
            if (error)
            {
                try { std::rethrow_exception(error); }
                catch (const std::exception &failure) { ADD_FAILURE() << failure.what(); }
            }
    }
}

#ifdef HAVE_CUDA
TEST(NativeReduceScatter, CUDA) { verify(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
TEST(NativeReduceScatter, ROCm) { verify(DeviceId::rocm(0)); }
#endif

namespace
{
    /**
     * @brief Prove native useful-byte receipts and untouched tails over retained replay.
     * @param first Backend's first physical GPU; communicator order is reversed.
     *
     * One count is uploaded before replay as fixture input, never downloaded to
     * schedule a collective. Reductions use exactly representable integers so
     * bit equality is independent of native SUM tree choices. The isolated
     * vendor proof separately checks cancellation-sensitive unchanged schedules.
     */
    void verifyLiveRows(DeviceId first)
    {
        auto *backend = getBackendFor(first);
        ASSERT_NE(backend, nullptr);
        ASSERT_GE(backend->deviceCount(), 2);
        const std::array devices{first.is_cuda() ? DeviceId::cuda(1) : DeviceId::rocm(1), first};
        auto tp = createLocalTPContext({GlobalDeviceAddress::fromLocalDeviceId(devices[0]),
            GlobalDeviceAddress::fromLocalDeviceId(devices[1])}, {},
            first.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        ASSERT_NE(tp, nullptr);
        std::barrier rendezvous(2);
        std::array<std::exception_ptr, 2> errors{};
        std::array<unsigned long long, 2> receipts{};
        std::array<std::thread, 2> workers;
        for (int participant = 0; participant < 2; ++participant)
            workers[participant] = std::thread([&, participant] {
                try
                {
                    auto &gpu = GPUDeviceContextPool::instance().getContext(devices[participant]);
                    gpu.submitAndWait([&] {
                        const auto device = devices[participant];
                        const int ordinal = device.gpu_ordinal();
                        void *stream = gpu.getOrCreateAuxiliaryStream("native_live_rows_regression");
                        constexpr int capacity = 17;
                        FP32Tensor input({2 * capacity * 1024 + 16}, device);
                        FP32Tensor output({2 * capacity * 1024 + 16}, device), metadata({4}, device);
                        for (auto *tensor : {&input, &output, &metadata})
                        {
                            std::fill_n(tensor->mutable_data(), tensor->numel(), 0.0f);
                            require(tensor->ensureOnDevice(device, stream), "live-row fixture upload");
                        }
                        require(gpu.synchronizeStreamChecked(stream), "live-row fixture preparation");
                        auto capture = gpu.createGraphCapture(stream);
                        const auto *count = static_cast<const std::int32_t *>(metadata.gpu_data_ptr());
                        auto *counter = reinterpret_cast<unsigned long long *>(
                            static_cast<std::uint8_t *>(metadata.gpu_data_ptr()) + 8);
                        for (auto operation : {NativeRowCollective::AllGather, NativeRowCollective::AllReduce,
                                NativeRowCollective::ReduceScatter})
                            for (auto type : {CollectiveDataType::FLOAT32, CollectiveDataType::FLOAT16,
                                    CollectiveDataType::BFLOAT16, CollectiveDataType::INT32, CollectiveDataType::INT8})
                                for (std::size_t width : {7u, 1024u})
                                    for (bool observed : {false, true})
                                    {
                                        const NativeCollectiveRows rows(DeviceRowRange::deviceCounted(capacity, count), width);
                                        const auto bank = capacity * width;
                                        const auto scalar_bytes = bytes(type);
                                        const auto op = operation == NativeRowCollective::AllGather
                                            ? CollectiveOp::ALLGATHER : CollectiveOp::ALLREDUCE_SUM;
                                        rendezvous.arrive_and_wait();
                                        ScopedBackendGraphCapture recording(gpu, *capture, "native_live_rows");
                                        require(recording.begin(), "begin live-row native capture");
                                        require(tp->nativeRowsOnStream(operation, input.gpu_data_ptr(), output.gpu_data_ptr(),
                                            rows, type, op, participant, stream, "live_rows_regression",
                                            observed ? counter : nullptr), "record live-row native collective");
                                        recording.finish();
                                        require(capture->instantiate(), "instantiate live-row collective");
                                        // Repeat large -> small -> empty -> growing without recapture.
                                        constexpr std::array<int, 10> lengths{17, 15, 0, 1, 7, 17, 0, 16, 3, 17};
                                        for (int request = 0; request < 20; ++request)
                                        {
                                            const int live = lengths[request % lengths.size()];
                                            const auto active = std::size_t(live) * width;
                                            std::vector<std::uint8_t> sent(input.size_bytes(), 0x5a);
                                            std::vector<std::uint8_t> received(output.size_bytes(), 0xa5);
                                            const int input_banks = operation == NativeRowCollective::ReduceScatter ? 2 : 1;
                                            for (int rank = 0; rank < input_banks; ++rank)
                                                for (std::size_t element = 0; element < active; ++element)
                                                    encode(sent.data() + (rank * bank + element) * scalar_bytes, type,
                                                        value(participant, request, rank * bank + element));
                                            std::array<std::uint32_t, 4> words{static_cast<std::uint32_t>(live), 0, 0, 0};
                                            require(backend->hostToDeviceOnStream(input.gpu_data_ptr(), sent.data(),
                                                sent.size(), ordinal, stream), "reset live-row input");
                                            require(backend->hostToDeviceOnStream(output.gpu_data_ptr(), received.data(),
                                                received.size(), ordinal, stream), "poison live-row output");
                                            require(backend->hostToDeviceOnStream(metadata.gpu_data_ptr(), words.data(),
                                                sizeof(words), ordinal, stream), "publish live-row test input");
                                            rendezvous.arrive_and_wait();
                                            require(capture->launch(), "replay live-row collective");
                                            require(backend->deviceToHostOnStream(received.data(), output.gpu_data_ptr(),
                                                received.size(), ordinal, stream), "observe live-row output");
                                            if (observed)
                                                require(backend->deviceToHostOnStream(&receipts[participant], counter,
                                                    sizeof(receipts[participant]), ordinal, stream), "observe useful-byte receipt");
                                            require(gpu.synchronizeStreamChecked(stream), "terminal live-row observation");
                                            std::vector<std::uint8_t> expected(output.size_bytes(), 0xa5);
                                            const int output_banks = operation == NativeRowCollective::AllGather ? 2 : 1;
                                            for (int rank = 0; rank < output_banks; ++rank)
                                                for (std::size_t element = 0; element < active; ++element)
                                                {
                                                    const auto source = operation == NativeRowCollective::ReduceScatter
                                                        ? participant * bank + element : element;
                                                    const int result = operation == NativeRowCollective::AllGather
                                                        ? value(rank, request, element)
                                                        : value(0, request, source) + value(1, request, source);
                                                    encode(expected.data() + (rank * bank + element) * scalar_bytes, type, result);
                                                }
                                            require(received == expected, "live-row output/rank-stride/tail mismatch");
                                            rendezvous.arrive_and_wait();
                                            // At degree two all three algorithms send two
                                            // copies of the bank's useful bytes in total.
                                            if (observed)
                                                require(receipts[0] + receipts[1] == 2 * active * scalar_bytes,
                                                    "native transport communicated inactive capacity");
                                            rendezvous.arrive_and_wait();
                                        }
                                        rendezvous.arrive_and_wait();
                                        capture->reset();
                                    }
                    });
                }
                catch (...)
                {
                    errors[participant] = std::current_exception();
                    tp->requestAbort();
                    rendezvous.arrive_and_drop();
                }
            });
        for (auto &worker : workers) worker.join();
        for (const auto &error : errors)
            if (error)
            {
                try { std::rethrow_exception(error); }
                catch (const std::exception &failure) { ADD_FAILURE() << failure.what(); }
            }
    }
}
#ifdef HAVE_CUDA
TEST(NativeLiveRows, CUDA) { verifyLiveRows(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
TEST(NativeLiveRows, ROCm) { verifyLiveRows(DeviceId::rocm(0)); }
#endif
