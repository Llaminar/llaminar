/**
 * @file Test__TPAllreduceRequestRows.cpp
 * @brief Captured CUDA/ROCm proof of economical ragged request-bank TP sums.
 *
 * Exercise the production stage, native grouping, persistent FP16 casts and
 * final publication with reversed physical device order. Independent request
 * prefixes shrink, empty and grow over twenty retained replays. Byte guards
 * prove padding is untouched; passive native receipts prove it is not sent.
 * Dense and MoE shifted-prefill builders share this exact collective contract.
 */
#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/AllreducePrecisionPolicy.h"
#include "collective/LocalTPContext.h"
#include "execution/compute_stages/stages/TPAllreduceStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "planning/CollectiveMemoryEstimator.h"
#include "tensors/FP16Utils.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "utils/DebugEnv.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Fail before retiring a live fixture or admitting another native operation. */
    void require(bool condition, const char *message)
    { if (!condition) throw std::runtime_error(message); }

    /** @return Fractional input that exposes both transport casts and FP16 sum rounding. */
    float contribution(int participant, int replay, std::size_t element)
    { return static_cast<float>((participant + 1) * (replay + 1) + element % 17) + 1.0f / 37.0f; }

    /**
     * @brief Prove request-bank extents and ordinary precision arithmetic on one backend.
     * @param first Backend's first physical GPU; communicator order is reversed.
     */
    void verifyRequestRows(DeviceId first)
    {
        auto *backend = getBackendFor(first);
        ASSERT_NE(backend, nullptr);
        ASSERT_GE(backend->deviceCount(), 2);
        const std::array devices{first.is_cuda() ? DeviceId::cuda(1) : DeviceId::rocm(1), first};
        constexpr int requests = 3, capacity = 17;
        constexpr std::array<std::array<std::int32_t, requests>, 6> lengths{{
            {17, 17, 17}, {0, 17, 1}, {16, 0, 0}, {0, 0, 0}, {1, 7, 16}, {17, 1, 17}}};
        for (const std::size_t width : {7u, 16384u})
        for (const std::string precision : {"fp32", "fp16"})
        {
            SCOPED_TRACE(::testing::Message() << "width=" << width << " precision=" << precision);
            const auto bank = std::size_t(capacity) * width;
            const auto count = requests * bank;
            const bool fp16 = fp32SumUsesFP16Transport(precision, bank, width,
                debugEnv().allreduce_fp16_min_elements);
            const auto wire_bytes = fp16 ? sizeof(std::uint16_t) : sizeof(float);
            auto tp = createLocalTPContext({GlobalDeviceAddress::fromLocalDeviceId(devices[0]),
                GlobalDeviceAddress::fromLocalDeviceId(devices[1])}, {},
                first.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
            ASSERT_NE(tp, nullptr);
            const auto bom = CollectiveMemoryEstimator::localTP(1, count, tp->backend());
            PhysicalMemoryPlanBuilder plan;
            for (const auto device : devices)
                plan.add({.world_rank = 0, .device = device, .total_bytes = bom.perDeviceBytes(),
                    .admission_available_bytes = bom.perDeviceBytes()},
                    PhysicalMemoryOwner::LocalCollective, bom.perDeviceBytes());
            auto authority = std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
            ASSERT_TRUE(tp->reserveCollectiveResources(bom.backend_payload_capacity_bytes,
                bom.fp16_scratch_elements, authority));
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
                            void *const stream = gpu.getOrCreateAuxiliaryStream("request_row_allreduce_regression");
                            FP32Tensor tensor({std::size_t(requests * capacity) + 1, width}, device);
                            FP32Tensor metadata({8}, device);
                            for (auto *value : {&tensor, &metadata})
                            {
                                std::fill_n(value->mutable_data(), value->numel(), 0.0f);
                                require(value->ensureOnDevice(device, stream), "request-bank fixture upload");
                            }
                            require(gpu.synchronizeStreamChecked(stream), "request-bank fixture preparation");
                            auto *counter = reinterpret_cast<unsigned long long *>(
                                static_cast<std::byte *>(metadata.gpu_data_ptr()) + 16);
                            TPAllreduceStage::Params params;
                            params.device_id = device;
                            params.tp_ctx = tp.get();
                            params.tensor = &tensor;
                            params.count = count;
                            params.stage_name = "shifted_mtp_embedding_allreduce";
                            params.precision = precision;
                            params.sideband_device_index = participant;
                            params.request_rows = NativeAllreduceRequestRows(requests, capacity, width,
                                static_cast<const std::int32_t *>(metadata.gpu_data_ptr())).withPayloadReceipt(counter);
                            TPAllreduceStage stage(params);
                            auto context = IDeviceContext::create(device, 1);
                            require(stage.prepareGraphLaunch(context.get(), stream), "request-bank stage preparation");
                            TransferEngine::requireDeviceInput(&tensor, device, stream);
                            TransferEngine::requireDeviceInput(&metadata, device, stream);
                            GraphCaptureDependencyLedger::StagePlan stage_plan;
                            stage_plan.stage_identity = &stage;
                            stage_plan.stage_name = params.stage_name;
                            stage_plan.external_inputs = {&tensor, &metadata};
                            stage_plan.outputs = {&tensor};
                            GraphCaptureDependencyLedger ledger(device, stream, {stage_plan}, params.stage_name);
                            auto capture = gpu.createGraphCapture(stream);
                            rendezvous.arrive_and_wait();
                            ScopedBackendGraphCapture recording(gpu, *capture, params.stage_name, &ledger);
                            require(recording.begin(), "begin request-bank capture");
                            {
                                ScopedGraphCaptureStage scope(&stage);
                                require(stage.execute(context.get()), "record production request-bank stage");
                                scope.complete();
                            }
                            recording.finish();
                            require(capture->instantiate(), "instantiate request-bank graph");
                            std::vector<std::uint8_t> sent(tensor.size_bytes()), actual(tensor.size_bytes());
                            for (int replay = 0; replay < 20; ++replay)
                            {
                                const auto &live = lengths[replay % lengths.size()];
                                std::fill(sent.begin(), sent.end(), 0xa5);
                                auto expected = sent;
                                std::size_t live_elements = 0;
                                for (int request = 0; request < requests; ++request)
                                {
                                    const auto active = std::size_t(live[request]) * width;
                                    live_elements += active;
                                    for (std::size_t element = 0; element < active; ++element)
                                    {
                                        const auto coordinate = request * bank + element;
                                        const float input = contribution(participant, replay, coordinate);
                                        float a = contribution(0, replay, coordinate), b = contribution(1, replay, coordinate);
                                        if (fp16)
                                        {
                                            a = fp16_to_fp32(fp32_to_fp16(a));
                                            b = fp16_to_fp32(fp32_to_fp16(b));
                                        }
                                        const float sum = fp16 ? fp16_to_fp32(fp32_to_fp16(a + b)) : a + b;
                                        std::memcpy(sent.data() + coordinate * sizeof(float), &input, sizeof(float));
                                        std::memcpy(expected.data() + coordinate * sizeof(float), &sum, sizeof(float));
                                    }
                                }
                                std::array<std::int32_t, 8> words{};
                                std::copy(live.begin(), live.end(), words.begin());
                                require(backend->hostToDeviceOnStream(tensor.gpu_data_ptr(), sent.data(),
                                    sent.size(), ordinal, stream), "reset request-bank inputs and guards");
                                require(backend->hostToDeviceOnStream(metadata.gpu_data_ptr(), words.data(),
                                    sizeof(words), ordinal, stream), "publish independent request counts");
                                rendezvous.arrive_and_wait();
                                require(capture->launch(), "replay request-bank graph");
                                require(backend->deviceToHostOnStream(actual.data(), tensor.gpu_data_ptr(),
                                    actual.size(), ordinal, stream), "observe request-bank output");
                                require(backend->deviceToHostOnStream(&receipts[participant], counter,
                                    sizeof(receipts[participant]), ordinal, stream), "observe native useful-byte receipt");
                                require(gpu.synchronizeStreamChecked(stream), "terminal request-bank observation");
                                require(actual == expected, "request-bank sum, inactive row or tensor guard differs bytewise");
                                rendezvous.arrive_and_wait();
                                require(receipts[0] + receipts[1] == 2 * live_elements * wire_bytes,
                                    "request-bank native transport communicated inactive capacity");
                                rendezvous.arrive_and_wait();
                            }
                            capture->reset();
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
}

#ifdef HAVE_CUDA
TEST(TPAllreduceRequestRows, CUDA) { verifyRequestRows(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
TEST(TPAllreduceRequestRows, ROCm) { verifyRequestRows(DeviceId::rocm(0)); }
#endif
