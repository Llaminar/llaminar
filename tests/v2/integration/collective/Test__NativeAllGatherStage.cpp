/**
 * @file Test__NativeAllGatherStage.cpp
 * @brief CUDA/NCCL and ROCm/RCCL proof of lossless captured byte allgathers.
 *
 * One retained graph transports several exact, sometimes unaligned byte
 * prefixes. Twenty input resets check communicator order, all bits (including
 * floating-point special values), untouched capacity tails and tensor event
 * publication. Physical GPU order is reversed deliberately. Setup/readback
 * waits belong only to the fixture; the production stage enqueues no waits on
 * the host and performs no allocation, conversion or extra copy.
 */
#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/LocalTPContext.h"
#include "execution/compute_stages/stages/NativeAllGatherStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Raise a fixture failure while native resources still have owners. */
    void require(bool value, const char *message)
    { if (!value) throw std::runtime_error(message); }

    /** @brief Include NaN payloads and signed zeros without doing FP arithmetic. */
    std::uint8_t payloadByte(int participant, int request, std::size_t offset)
    {
        constexpr std::array<std::uint8_t, 16> special{
            0, 0, 0, 128, 1, 0, 192, 127, 0, 0, 128, 255, 255, 255, 255, 255};
        if (offset < special.size())
            return special[(offset + request + participant) % special.size()];
        return static_cast<std::uint8_t>(offset * 37 + participant * 101 + request * 17);
    }

    /** @brief Exercise one existing native backend, without a model or policy override. */
    void verifyNativeAllGather(DeviceId first)
    {
        auto *backend = getBackendFor(first);
        ASSERT_NE(backend, nullptr);
        ASSERT_GE(backend->deviceCount(), 2) << "Native allgather proof requires two GPUs";
        const std::array<DeviceId, 2> devices{
            first.is_cuda() ? DeviceId::cuda(1) : DeviceId::rocm(1), first};
        auto tp = createLocalTPContext({GlobalDeviceAddress::fromLocalDeviceId(devices[0]),
            GlobalDeviceAddress::fromLocalDeviceId(devices[1])}, {},
            first.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        ASSERT_NE(tp, nullptr);
        constexpr std::array<std::size_t, 4> extents{1, 37, 4097, 32768};
        constexpr std::array input_ids{BufferId::HIDDEN_STATE, BufferId::MOE_GATE_SCRATCH,
            BufferId::MOE_UP_SCRATCH, BufferId::MOE_EXPERT_OUTPUT};
        constexpr std::array output_ids{BufferId::NORMALIZED, BufferId::MOE_COMBINED_OUTPUT,
            BufferId::MOE_SHARED_EXPERT_OUTPUT, BufferId::FFN_OUTPUT};
        std::barrier rendezvous(2);
        std::array<std::exception_ptr, 2> errors{};
        std::array<std::thread, 2> threads;
        for (int participant = 0; participant < 2; ++participant)
        {
            threads[participant] = std::thread([&, participant]
            {
                try
                {
                    auto &gpu = GPUDeviceContextPool::instance().getContext(devices[participant]);
                    gpu.submitAndWait([&]
                    {
                        const auto device = devices[participant];
                        const int ordinal = device.is_cuda() ? device.cuda_ordinal() : device.rocm_ordinal();
                        void *const stream = gpu.getOrCreateAuxiliaryStream("native_allgather_regression");
                        auto context = IDeviceContext::create(device, 1);
                        std::array<std::unique_ptr<FP32Tensor>, extents.size()> inputs, outputs;
                        std::unique_ptr<IGPUGraphCapture> capture;
                        try
                        {
                            ComputeGraph graph;
                            std::vector<GraphCaptureDependencyLedger::StagePlan> plans;
                            for (std::size_t index = 0; index < extents.size(); ++index)
                            {
                                // Extra capacity is deliberate: the native count
                                // must be the message prefix, not size_bytes().
                                inputs[index] = std::make_unique<FP32Tensor>(
                                    std::vector<std::size_t>{(extents[index] + 3) / 4 + 16}, device);
                                outputs[index] = std::make_unique<FP32Tensor>(
                                    std::vector<std::size_t>{(2 * extents[index] + 3) / 4 + 16}, device);
                                for (auto *tensor : {inputs[index].get(), outputs[index].get()})
                                {
                                    std::fill_n(tensor->mutable_data(), tensor->numel(), 0.0f);
                                    require(tensor->ensureOnDevice(device, stream), "fixture upload");
                                }
                                NativeAllGatherStage::Params params;
                                params.device_id = device;
                                params.tp_ctx = tp.get();
                                params.local_input = inputs[index].get();
                                params.rank_major_output = outputs[index].get();
                                params.bytes_per_participant = extents[index];
                                params.participant = participant;
                                params.stage_name = "native_byte_gather_" + std::to_string(index);
                                params.input_buffer_id = input_ids[index];
                                params.output_buffer_id = output_ids[index];
                                auto stage = std::make_unique<NativeAllGatherStage>(params);
                                require(stage->prepareGraphLaunch(context.get(), stream), "stage stream binding");
                                require(stage->isGraphCapturable(), "native collective capture contract");
                                GraphCaptureDependencyLedger::StagePlan plan;
                                plan.stage_identity = stage.get();
                                plan.stage_name = params.stage_name;
                                plan.external_inputs = {inputs[index].get()};
                                plan.outputs = {outputs[index].get()};
                                plans.push_back(std::move(plan));
                                graph.addNode(params.stage_name, std::move(stage), device);
                                if (index)
                                    graph.addDependency(params.stage_name, "native_byte_gather_" + std::to_string(index - 1));
                            }
                            require(gpu.synchronizeStreamChecked(stream), "fixture upload completion");
                            for (const auto &input : inputs)
                                TransferEngine::requireDeviceInput(input.get(), device, stream);
                            GraphCaptureDependencyLedger ledger(device, stream, std::move(plans), "native_allgather");
                            capture = gpu.createGraphCapture(stream);
                            rendezvous.arrive_and_wait();
                            ScopedBackendGraphCapture recording(gpu, *capture, "native_allgather", &ledger);
                            require(recording.begin(), "begin collective graph");
                            for (const auto &name : graph.getExecutionOrder())
                            {
                                auto *stage = graph.getNode(name)->stage.get();
                                ScopedGraphCaptureStage scope(stage);
                                require(stage->execute(context.get()), "record native allgather");
                                scope.complete();
                            }
                            recording.finish();
                            require(capture->instantiate(), "instantiate collective graph");

                            std::array<std::vector<std::uint8_t>, extents.size()> sent, received;
                            for (int request = 1; request <= 20; ++request)
                            {
                                for (std::size_t index = 0; index < extents.size(); ++index)
                                {
                                    sent[index].resize(inputs[index]->size_bytes());
                                    received[index].assign(outputs[index]->size_bytes(), 0xa5);
                                    for (std::size_t byte = 0; byte < sent[index].size(); ++byte)
                                        sent[index][byte] = payloadByte(participant, request, byte);
                                    require(backend->hostToDeviceOnStream(inputs[index]->gpu_data_ptr(), sent[index].data(),
                                        sent[index].size(), ordinal, stream), "reset input bytes");
                                    require(backend->hostToDeviceOnStream(outputs[index]->gpu_data_ptr(), received[index].data(),
                                        received[index].size(), ordinal, stream), "poison receive capacity");
                                }
                                rendezvous.arrive_and_wait();
                                require(capture->launch(), "replay unchanged collective graph");
                                for (std::size_t index = 0; index < extents.size(); ++index)
                                    require(backend->deviceToHostOnStream(received[index].data(), outputs[index]->gpu_data_ptr(),
                                        received[index].size(), ordinal, stream), "observe receive bytes");
                                require(gpu.synchronizeStreamChecked(stream), "terminal observation");
                                for (std::size_t index = 0; index < extents.size(); ++index)
                                {
                                    const auto bytes = extents[index];
                                    for (std::size_t byte = 0; byte < received[index].size(); ++byte)
                                    {
                                        const auto expected = byte < 2 * bytes
                                            ? payloadByte(static_cast<int>(byte / bytes), request, byte % bytes)
                                            : std::uint8_t{0xa5};
                                        require(received[index][byte] == expected,
                                            "native allgather changed payload, participant order or unused capacity");
                                    }
                                }
                            }
                            rendezvous.arrive_and_wait();
                            capture->reset();
                        }
                        catch (...)
                        {
                            // Abort peers before retained graphs/tensors retire:
                            // their streams may still be inside this collective.
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
        for (auto &thread : threads) thread.join();
        for (const auto &error : errors)
            if (error)
            {
                try { std::rethrow_exception(error); }
                catch (const std::exception &failure) { ADD_FAILURE() << failure.what(); }
            }
    }
} // namespace

#ifdef HAVE_CUDA
TEST(Test__NativeAllGatherStage, CUDA) { verifyNativeAllGather(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
TEST(Test__NativeAllGatherStage, ROCm) { verifyNativeAllGather(DeviceId::rocm(0)); }
#endif
