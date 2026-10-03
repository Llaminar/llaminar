/**
 * @file Test__MoERouterHiddenQ8.cpp
 * @brief Prove the live router's reusable Q8 activation publication bytewise.
 *
 * Qwen's expert stages normally borrow the router's quantized hidden rows. An
 * isolated expert-quantizer test therefore does not cover the producer used by
 * the model. This fixture captures the public routing interface, observes its
 * graph-owned publication, and compares every live byte and scale with the
 * independent host numerical contract. Retained replays vary inputs and the
 * device-owned logical length without recapture. CUDA verifier and ROCm
 * verifier/prefill routes share the same proof; no source weight format is
 * changed to manufacture a match.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "kernels/common/DeviceQ8ActivationNumericalContract.h"
#include "utils/TestTensorFactory.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using namespace llaminar2;
    using namespace llaminar2::test;

    /** @brief One explicitly selected real backend, never a silent device skip. */
    class MoERouterHiddenQ8 : public ::testing::TestWithParam<std::string> {};

    /** @brief Fail setup/replay while retaining the enclosing observation join. */
    void require(bool value, const char *message)
    {
        if (!value) throw std::runtime_error(message);
    }

    /** @brief Join test observation before graph, tensors or workspace retire. */
    struct ObservationJoin
    {
        IBackend *backend;
        DeviceId device;
        void *stream;
        /** @brief Also drain a partially submitted failing test transaction. */
        ~ObservationJoin() { (void)backend->synchronizeStream(stream, device.ordinal); }
    };

    /**
     * @brief Check original 32-value blocks without calling a GPU quantizer.
     * @param input Original host FP32 values for the currently replayed request.
     * @param bytes Published live Q8 bytes, in source row order.
     * @param scales Published live FP32 representations of half-rounded scales.
     */
    void proveBytes(const std::vector<float> &input, const std::vector<int8_t> &bytes,
                    const std::vector<float> &scales)
    {
        ASSERT_EQ(bytes.size(), scales.size() * 32);
        for (size_t block = 0; block < scales.size(); ++block)
        {
            float maximum = 0.f;
            for (size_t lane = 0; lane < 32; ++lane)
                maximum = std::max(maximum, std::fabs(input[block * 32 + lane]));
            const float scale = device_q8_activation_contract::scale(maximum);
            ASSERT_EQ(std::bit_cast<uint32_t>(scale), std::bit_cast<uint32_t>(scales[block]))
                << "scale block=" << block;
            const float inverse = device_fp32_contract::reciprocalPositive(scale);
            for (size_t lane = 0; lane < 32; ++lane)
            {
                const size_t index = block * 32 + lane;
                const float scaled = device_fp32_contract::multiply(input[index], inverse);
                const int expected = std::clamp(static_cast<int>(std::rint(scaled)), -127, 127);
                ASSERT_EQ(int(bytes[index]), expected) << "hidden element=" << index;
            }
        }
    }

    /**
     * @brief Capture every small M and representative prefill/workgroup tails.
     * @param device Endpoint owning all buffers and the exact capture stream.
     * @param width Hidden columns; 96 exercises a partially populated wave64.
     * @param kind Production prefill or grouped serial-equivalent verification.
     */
    void prove(DeviceId device, int width, MoERouteLaunchKind kind)
    {
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&]
        {
            constexpr int capacity = 512, experts = 256, top_k = 8;
            auto *backend = getBackendFor(device);
            void *stream = context.defaultStream();
            require(backend && stream, "router Q8 fixture needs an exact device stream");
            const auto requirements = device.is_cuda()
                ? MoEWorkspaceBuffers::cudaMoE(capacity, width, 256, experts, top_k)
                : MoEWorkspaceBuffers::rocmMoE(capacity, width, 256, experts, top_k);
            DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment());
            require(workspace.allocate(requirements), "router Q8 workspace admission");
            auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
            require(kernel != nullptr, "router Q8 kernel creation");
            kernel->setGPUStream(stream);
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
            require(consumer != nullptr, "router Q8 workspace binding");
            consumer->bindWorkspace(&workspace);
            auto publication = std::make_shared<MoERouterQ8HiddenPublication>();
            require(kernel->bindRouterQ8HiddenPublication(publication,
                MoERouterQ8PublicationAccess::ProducerAndConsumer), "router Q8 publication binding");

            auto hidden = TestTensorFactory::createFP32({capacity, size_t(width)});
            auto gate = TestTensorFactory::createFP32Random({experts, size_t(width)}, -.125f, .125f, 32718);
            auto indices = TestTensorFactory::createFP32({capacity, top_k});
            auto weights = TestTensorFactory::createFP32({capacity, top_k});
            auto length = TestTensorFactory::createINT32({1});
            std::fill_n(hidden->mutable_data(), hidden->numel(), .125f);
            length->mutable_int32_data()[0] = capacity;
            require(hidden->ensureOnDevice(device, stream) && gate->ensureOnDevice(device, stream) &&
                indices->ensureOnDevice(device, stream) && weights->ensureOnDevice(device, stream) &&
                length->ensureOnDevice(device, stream), "router Q8 fixture tensors");
            require(kernel->prepareRouteLaunch(gate.get(), {.kind = kind, .physical_rows = capacity,
                .d_model = width, .num_experts = experts, .top_k = top_k}), "router Q8 preparation");

            std::vector<int> row_counts{1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
                                       255, 256, 257, 448, 512};
            if (width == 256)
            {
                row_counts.clear();
                for (int rows = 1; rows <= 65; ++rows) row_counts.push_back(rows);
            }
            std::vector<float> input(hidden->numel(), .125f);
            const int *device_length = static_cast<const int *>(length->gpu_data_ptr());
            for (const int rows : row_counts)
            {
                // Ordinary ROCm M=1 routing does not publish reusable Q8 rows;
                // its serial-equivalent producer is covered by the verifier
                // case. Prefill publication begins with the multi-row path.
                if (kind == MoERouteLaunchKind::GroupedPrefill && rows == 1) continue;
                SCOPED_TRACE(::testing::Message() << "K=" << width << " M=" << rows
                    << " route=" << int(kind));
                MoERoutingResult unused_host_result;
                const auto enqueue = [&]
                {
                    if (kind == MoERouteLaunchKind::DecodeEquivalentVerifier)
                        return kernel->routeVerifierRowsDecodeEquivalent(hidden.get(), gate.get(),
                            rows, width, experts, top_k, true, indices.get(), weights.get(), device_length);
                    return kernel->routeWithTensorsEffectiveSeqLen(hidden.get(), gate.get(),
                        rows, width, experts, top_k, true, indices.get(), weights.get(),
                        unused_host_result, device_length);
                };
                auto graph = context.createGraphCapture(stream);
                require(graph != nullptr, "router Q8 graph owner");
                ObservationJoin join{backend, device, stream};
                require(enqueue() && backend->synchronizeStream(stream, device.ordinal), "router Q8 warmup");
                {
                    GraphCaptureGuard recording;
                    require(graph->beginCapture(), "router Q8 capture begin");
                    require(enqueue(), "router Q8 capture body");
                    require(graph->endCapture(), "router Q8 capture end");
                }
                require(graph->instantiate(), "router Q8 graph instantiation");
                ASSERT_NE(publication->quantized_rows, nullptr);
                ASSERT_NE(publication->row_scales, nullptr);
                ASSERT_EQ(publication->source_rows, hidden->gpu_data_ptr());
                ASSERT_EQ(publication->published_rows, rows);
                ASSERT_TRUE(publication->capture_recorded);

                int replay = 0;
                for (const int logical_rows : {rows, rows / 2, 0, rows})
                {
                    SCOPED_TRACE(::testing::Message() << "live=" << logical_rows);
                    ++replay;
                    for (size_t i = 0; i < input.size(); ++i)
                    {
                        const size_t group = i / 32;
                        // Adjacent blocks deliberately have unrelated maxima.
                        // Zero and signed-zero blocks probe the half-scale floor.
                        const float magnitude = std::ldexp(1.f, int((group + replay) % 5) * 4 - 12);
                        input[i] = group % 7 == 0 ? (i % 2 ? -0.f : 0.f) :
                            magnitude * float(int((i * 19 + replay * 13) % 101) - 50);
                    }
                    require(backend->hostToDevice(hidden->gpu_data_ptr(), input.data(), input.size() * sizeof(float),
                        device.ordinal, stream) && backend->hostToDevice(length->gpu_data_ptr(), &logical_rows,
                        sizeof(logical_rows), device.ordinal, stream), "router Q8 replay data");
                    require(graph->launch(), "router Q8 retained replay");
                    std::vector<int8_t> bytes(size_t(logical_rows) * width);
                    std::vector<float> scales(bytes.size() / 32), route_ids(size_t(rows) * top_k), probabilities(route_ids.size());
                    if (!bytes.empty())
                        require(backend->deviceToHost(bytes.data(), publication->quantized_rows, bytes.size(), device.ordinal, stream) &&
                            backend->deviceToHost(scales.data(), publication->row_scales, scales.size() * sizeof(float),
                                device.ordinal, stream), "router Q8 byte observation");
                    require(backend->deviceToHost(route_ids.data(), indices->gpu_data_ptr(), route_ids.size() * sizeof(float),
                        device.ordinal, stream) && backend->deviceToHost(probabilities.data(), weights->gpu_data_ptr(),
                        probabilities.size() * sizeof(float), device.ordinal, stream) &&
                        backend->synchronizeStream(stream, device.ordinal), "router Q8 route observation");
                    proveBytes(input, bytes, scales);
                    for (int row = 0; row < rows; ++row)
                        for (int route = 0; route < top_k; ++route)
                        {
                            const size_t slot = size_t(row) * top_k + route;
                            if (row >= logical_rows)
                            {
                                ASSERT_EQ(route_ids[slot], -1.f);
                                ASSERT_EQ(probabilities[slot], 0.f);
                            }
                            else
                            {
                                ASSERT_GE(route_ids[slot], 0.f);
                                ASSERT_LT(route_ids[slot], float(experts));
                                ASSERT_TRUE(std::isfinite(probabilities[slot]));
                            }
                        }
                }
            }
        });
    }

    /** @brief Both backends retain serial-equivalent grouped Q8 publications. */
    TEST_P(MoERouterHiddenQ8, CapturedVerifierBytesAndLiveRows)
    {
        const DeviceId device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
        for (int width : {96, 256, 2048, 3072})
            prove(device, width, MoERouteLaunchKind::DecodeEquivalentVerifier);
    }

#ifdef HAVE_ROCM
    /** @brief The actual ROCm prefill producer is independently exercised too. */
    TEST(MoERouterHiddenQ8Prefill, ROCmCapturedBytesAndLiveRows)
    {
        for (int width : {96, 256, 2048, 3072})
            prove(DeviceId::rocm(0), width, MoERouteLaunchKind::GroupedPrefill);
    }
#endif

    INSTANTIATE_TEST_SUITE_P(Backends, MoERouterHiddenQ8, ::testing::ValuesIn(std::vector<std::string>{
#ifdef HAVE_CUDA
        "CUDA",
#endif
#ifdef HAVE_ROCM
        "ROCm",
#endif
    }), [](const auto &info) { return info.param; });
} // namespace
