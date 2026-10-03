/**
 * @file Test__MoERouterOwnedRows.cpp
 * @brief Exact captured row ownership against each backend's installed router.
 *
 * Each participant retains one graph while live counts and hidden values change.
 * The unchanged complete-row router is the oracle, including its prepared gate
 * representation. Every owned selection bit must match; compact tails remain
 * poisoned, and ROCm's Q8 side product must still contain all live input rows.
 * This proves a local kernel boundary, not inter-device transport or model wiring.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/MoERouterOwnedRows.h"
#include "utils/TestTensorFactory.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <memory>
#include <initializer_list>
#include <stdexcept>
#include <vector>

extern "C"
{
    bool hipMoE_gate_logits_q8_weights_decode_equivalent_rows(const float *, int8_t *, float *,
        const int8_t *, const float *, float *, int, int, int, int, void *, const int *);
    bool hipMoE_gate_logits_fp32_decode_equivalent_rows(const float *, const float *, float *,
        int, int, int, int, void *, const int *);
    bool hipMoE_gate_logits_fp16_decode_equivalent_rows(const float *, const void *, float *,
        int, int, int, int, void *, const int *);
    bool hipMoE_gate_logits_bf16_decode_equivalent_rows(const float *, const void *, float *,
        int, int, int, int, void *, const int *);
    bool hipMoE_softmax_topk_decode_equivalent_rows(float *, float *, float *, int, int, int,
        bool, int, void *, const int *, void *);
    bool cudaMoE_route_logits(const float *, const float *, float *, int, int, int, int, void *);
    bool cudaMoE_route_logits_bf16(const float *, const void *, float *, int, int, int, int, void *);
    bool cudaMoE_softmax_topk(float *, float *, float *, int, int, int, bool, int, void *, const int *, void *);
}

namespace
{
    using namespace llaminar2;
    using namespace llaminar2::test;

    /** @brief Preserve the first backend failure across its worker-thread boundary. */
    void require(bool result, const char *edge)
    { if (!result) throw std::runtime_error(edge); }

    /** @brief Complete observation before retiring captured addresses on any exit. */
    struct ObservationJoin
    {
        IBackend *backend;
        DeviceId device;
        void *stream;
        /** @brief Test-only completion, never an inference-time synchronization. */
        ~ObservationJoin() { (void)backend->synchronizeStream(stream, device.ordinal); }
    };

    /** @brief Invoke only the explicitly requested native backend. */
    bool enqueueOwned(DeviceId device, const MoERouterOwnedRowsLaunch &launch, void *stream)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda()) return llaminar2::cuda::routeOwnedRows(launch, stream);
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) return llaminar2::rocm::routeOwnedRows(launch, stream);
#endif
        throw std::invalid_argument("Owned router fixture requires its compiled backend");
    }

    /** @brief Execute the unchanged full-row production bridges with separate outputs. */
    bool enqueueOracle(DeviceId device, const MoERouterOwnedRowsLaunch &p,
                       float *indices, float *weights, void *stream)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda())
        {
            const bool logits = p.format == MoERouterPreparedFormat::FP32
                ? cudaMoE_route_logits(p.hidden, static_cast<const float *>(p.gate), p.logits,
                    p.capacity, p.width, p.experts, device.ordinal, stream)
                : cudaMoE_route_logits_bf16(p.hidden, p.gate, p.logits,
                    p.capacity, p.width, p.experts, device.ordinal, stream);
            return logits && cudaMoE_softmax_topk(p.logits, indices, weights, p.capacity,
                p.experts, p.top_k, p.normalize, device.ordinal, stream, p.live_rows, nullptr);
        }
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm())
        {
            bool logits = false;
            switch (p.format)
            {
            case MoERouterPreparedFormat::FP32:
                logits = hipMoE_gate_logits_fp32_decode_equivalent_rows(p.hidden,
                    static_cast<const float *>(p.gate), p.logits, p.capacity, p.width,
                    p.experts, device.ordinal, stream, p.live_rows); break;
            case MoERouterPreparedFormat::FP16:
                logits = hipMoE_gate_logits_fp16_decode_equivalent_rows(p.hidden,
                    p.gate, p.logits, p.capacity, p.width, p.experts, device.ordinal, stream, p.live_rows); break;
            case MoERouterPreparedFormat::BF16:
                logits = hipMoE_gate_logits_bf16_decode_equivalent_rows(p.hidden,
                    p.gate, p.logits, p.capacity, p.width, p.experts, device.ordinal, stream, p.live_rows); break;
            case MoERouterPreparedFormat::BlockQ8:
                logits = hipMoE_gate_logits_q8_weights_decode_equivalent_rows(p.hidden,
                    p.hidden_q8, p.hidden_scales, static_cast<const int8_t *>(p.gate), p.gate_scales,
                    p.logits, p.capacity, p.width, p.experts, device.ordinal, stream, p.live_rows); break;
            }
            return logits && hipMoE_softmax_topk_decode_equivalent_rows(p.logits, indices,
                weights, p.capacity, p.experts, p.top_k, p.normalize, device.ordinal,
                stream, p.live_rows, nullptr);
        }
#endif
        throw std::invalid_argument("Owned router oracle requires its compiled backend");
    }

    /** @brief Capture each logical owner and exhaust retained live-prefix transitions.
     * @param device One physical endpoint; logical partitions are checked independently.
     * @param format Prepared router representation, not expert source codebook.
     * @param capacity Retained whole-matrix row capacity.
     * @param width Hidden columns, including ragged and Q8 serial-partition boundaries.
     * @param experts Router columns.
     * @param degrees Logical domain sizes, including more participants than live rows. */
    void prove(DeviceId device, MoERouterPreparedFormat format, int capacity, int width,
               int experts, const std::vector<int> &degrees)
    {
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&]
        {
            SCOPED_TRACE(::testing::Message() << device.toString() << " format=" << int(format)
                << " M=" << capacity << " K=" << width << " E=" << experts);
            auto *backend = getBackendFor(device);
            void *stream = context.defaultStream();
            require(backend && stream, "Owned router exact context stream");
            constexpr int top_k = 8;
            const size_t input_elements = size_t(capacity) * width;
            const size_t route_elements = size_t(capacity + 1) * top_k;
            auto hidden = TestTensorFactory::createFP32({input_elements});
            auto length = TestTensorFactory::createINT32({1});
            std::unique_ptr<TensorBase> gate;
            if (format == MoERouterPreparedFormat::FP16)
                gate = TestTensorFactory::createFP16Random({size_t(experts), size_t(width)}, -.125f, .125f, 1772);
            else if (format == MoERouterPreparedFormat::BF16)
                gate = TestTensorFactory::createBF16Random({size_t(experts), size_t(width)}, -.125f, .125f, 1772);
            else if (format == MoERouterPreparedFormat::BlockQ8)
            {
                auto bytes = TestTensorFactory::createINT32({size_t(experts) * width / 4});
                auto *values = reinterpret_cast<int8_t *>(bytes->mutable_int32_data());
                for (size_t i = 0; i < size_t(experts) * width; ++i)
                    values[i] = int8_t(int(i * 71 % 255) - 127);
                gate = std::move(bytes);
            }
            else gate = TestTensorFactory::createFP32Random({size_t(experts), size_t(width)}, -.125f, .125f, 1772);
            auto gate_scales = TestTensorFactory::createFP32({size_t(experts) * width / 32 + 1});
            std::fill_n(gate_scales->mutable_data(), gate_scales->numel(), .001f);
            auto logits = TestTensorFactory::createFP32({size_t(capacity + 1) * experts});
            auto reference_logits = TestTensorFactory::createFP32({size_t(capacity) * experts});
            auto selected = TestTensorFactory::createFP32({route_elements * 2});
            auto selected_bytes = TestTensorFactory::createINT32({2});
            auto reference_indices = TestTensorFactory::createFP32({route_elements});
            auto reference_weights = TestTensorFactory::createFP32({route_elements});
            auto q8 = TestTensorFactory::createINT32({input_elements / 4 + 1});
            auto reference_q8 = TestTensorFactory::createINT32({input_elements / 4 + 1});
            auto scales = TestTensorFactory::createFP32({input_elements / 32 + 1});
            auto reference_scales = TestTensorFactory::createFP32({input_elements / 32 + 1});
            std::fill_n(hidden->mutable_data(), input_elements, .125f);
            length->mutable_int32_data()[0] = capacity;
            for (TensorBase *tensor : std::initializer_list<TensorBase *>{hidden.get(), length.get(), gate.get(), gate_scales.get(),
                logits.get(), reference_logits.get(), selected.get(), selected_bytes.get(), reference_indices.get(),
                reference_weights.get(), q8.get(), reference_q8.get(), scales.get(), reference_scales.get()})
                require(tensor->ensureOnDevice(device, stream), "Owned router storage");
            MoERouterOwnedRowsLaunch launch{
                .partition = DeviceRowPartition::balanced(0, 2),
                .capacity = capacity, .width = width, .experts = experts, .top_k = top_k,
                .live_rows = static_cast<const int32_t *>(length->gpu_data_ptr()),
                .hidden = static_cast<const float *>(hidden->gpu_data_ptr()), .format = format,
                .gate = gate->gpu_data_ptr(), .gate_scales = static_cast<const float *>(gate_scales->gpu_data_ptr()),
                .hidden_q8 = static_cast<int8_t *>(q8->gpu_data_ptr()),
                .hidden_scales = static_cast<float *>(scales->gpu_data_ptr()),
                .logits = static_cast<float *>(logits->gpu_data_ptr()),
                .selected = static_cast<MoERouterSelectedRoute *>(selected->gpu_data_ptr()),
                .selected_bytes = static_cast<uint64_t *>(selected_bytes->gpu_data_ptr())};
            auto oracle = launch;
            oracle.logits = static_cast<float *>(reference_logits->gpu_data_ptr());
            auto *oracle_indices = static_cast<float *>(reference_indices->gpu_data_ptr());
            auto *oracle_weights = static_cast<float *>(reference_weights->gpu_data_ptr());
            oracle.hidden_q8 = static_cast<int8_t *>(reference_q8->gpu_data_ptr());
            oracle.hidden_scales = static_cast<float *>(reference_scales->gpu_data_ptr());
            ObservationJoin outer_join{backend, device, stream};
            EXPECT_FALSE(enqueueOwned(device, launch, nullptr));
            auto invalid = launch;
            invalid.live_rows = nullptr;
            EXPECT_FALSE(enqueueOwned(device, invalid, stream));
            invalid = launch; invalid.partition = {};
            EXPECT_FALSE(enqueueOwned(device, invalid, stream));
            invalid = launch; invalid.capacity = 1;
            EXPECT_FALSE(enqueueOwned(device, invalid, stream));
            invalid = launch; invalid.selected_bytes = nullptr;
            EXPECT_FALSE(enqueueOwned(device, invalid, stream));

            std::vector<int> counts{capacity, 0, capacity / 2, 1};
            if (capacity == 65 && width == 96)
                for (int count = 2; count <= capacity; ++count) counts.push_back(count);
            else
                for (int count : {2, 3, 7, 8, 15, 16, 17, 31, 32, 63, 64, 65, 447, 448, 449, capacity - 1})
                    if (count <= capacity) counts.push_back(count);
            counts.insert(counts.end(), {capacity, 1, 0, capacity});
            std::vector<float> input(input_elements), expected_ids(route_elements), expected_weights(route_elements);
            std::vector<MoERouterSelectedRoute> actual(route_elements);
            constexpr uint32_t poison = 0x7fc01234u;
            const std::vector<float> poisoned(route_elements * 2, std::bit_cast<float>(poison));
            for (int degree : degrees)
                for (int participant = 0; participant < degree; ++participant)
                {
                    SCOPED_TRACE(::testing::Message() << "degree=" << degree << " owner=" << participant);
                    launch.partition = DeviceRowPartition::balanced(participant, degree);
                    // Exercise both existing probability policies without
                    // changing either immutable policy during graph replay.
                    launch.normalize = degree != 3;
                    oracle.normalize = launch.normalize;
                    auto graph = context.createGraphCapture(stream);
                    require(bool(graph), "Owned router graph owner");
                    ObservationJoin join{backend, device, stream};
                    require(enqueueOwned(device, launch, stream) && backend->synchronizeStream(stream, device.ordinal),
                        "Owned router preparation");
                    {
                        ScopedBackendGraphCapture recording(context, *graph, "Owned router live rows");
                        require(recording.begin(), "Owned router capture begin");
                        require(enqueueOwned(device, launch, stream), "Owned router captured body");
                        recording.finish();
                    }
                    std::vector<GPUGraphKernelNodeInfo> nodes;
                    ASSERT_TRUE(graph->inspectKernelNodes(nodes));
                    const bool owns_capacity = launch.partition.capacityFor(capacity) > 0;
                    ASSERT_EQ(nodes.size(), 1u + owns_capacity + (format == MoERouterPreparedFormat::BlockQ8));
                    for (const auto &node : nodes) EXPECT_EQ(node.local_memory_bytes_per_thread, 0u) << node.name;
                    require(graph->instantiate(), "Owned router instantiate");
                    int generation = 0;
                    for (int count : counts)
                    {
                        SCOPED_TRACE(::testing::Message() << "live=" << count);
                        ++generation;
                        for (size_t i = 0; i < input.size(); ++i)
                            input[i] = generation % 5 == 0 ? (i % 2 ? -0.f : 0.f)
                                : std::sin(float(i + generation * 17) * .031f) * .25f;
                        require(backend->hostToDevice(hidden->gpu_data_ptr(), input.data(), input.size() * sizeof(float), device.ordinal, stream) &&
                            backend->hostToDevice(length->gpu_data_ptr(), &count, sizeof(count), device.ordinal, stream) &&
                            backend->hostToDevice(selected->gpu_data_ptr(), poisoned.data(), poisoned.size() * sizeof(float), device.ordinal, stream),
                            "Owned router replay inputs");
                        require(enqueueOracle(device, oracle, oracle_indices, oracle_weights, stream) && graph->launch(), "Owned router and independent oracle");
                        uint64_t bytes = UINT64_MAX;
                        require(backend->deviceToHost(expected_ids.data(), oracle_indices, route_elements * sizeof(float), device.ordinal, stream) &&
                            backend->deviceToHost(expected_weights.data(), oracle_weights, route_elements * sizeof(float), device.ordinal, stream) &&
                            backend->deviceToHost(actual.data(), launch.selected, actual.size() * sizeof(actual[0]), device.ordinal, stream) &&
                            backend->deviceToHost(&bytes, launch.selected_bytes, sizeof(bytes), device.ordinal, stream) &&
                            backend->synchronizeStream(stream, device.ordinal), "Owned router terminal observation");
                        const auto span = launch.partition.resolveFor(capacity, count);
                        ASSERT_EQ(bytes, uint64_t(span.count) * top_k * sizeof(MoERouterSelectedRoute));
                        for (size_t i = 0; i < route_elements; ++i)
                        {
                            const bool live = i < size_t(span.count) * top_k;
                            const size_t source = size_t(span.first) * top_k + i;
                            ASSERT_EQ(std::bit_cast<uint32_t>(actual[i].expert), live ? std::bit_cast<uint32_t>(expected_ids[source]) : poison) << "ID=" << i;
                            ASSERT_EQ(std::bit_cast<uint32_t>(actual[i].weight), live ? std::bit_cast<uint32_t>(expected_weights[source]) : poison) << "weight=" << i;
                        }
                        if (format == MoERouterPreparedFormat::BlockQ8 && count > 0)
                        {
                            const size_t bytes = size_t(count) * width;
                            std::vector<int8_t> actual(bytes), expected(bytes);
                            std::vector<uint32_t> actual_scales(bytes / 32), expected_scales(bytes / 32);
                            require(backend->deviceToHost(actual.data(), launch.hidden_q8, bytes, device.ordinal, stream) &&
                                backend->deviceToHost(expected.data(), oracle.hidden_q8, bytes, device.ordinal, stream) &&
                                backend->deviceToHost(actual_scales.data(), launch.hidden_scales, actual_scales.size() * 4, device.ordinal, stream) &&
                                backend->deviceToHost(expected_scales.data(), oracle.hidden_scales, expected_scales.size() * 4, device.ordinal, stream) &&
                                backend->synchronizeStream(stream, device.ordinal), "Owned router complete Q8 side product");
                            ASSERT_EQ(actual, expected);
                            ASSERT_EQ(actual_scales, expected_scales);
                        }
                    }
                }
        });
    }

    /** @brief Sweep supported prepared gate formats without changing numeric policy. */
    void proveBackend(DeviceId device)
    {
        std::vector<MoERouterPreparedFormat> formats{MoERouterPreparedFormat::FP32, MoERouterPreparedFormat::BF16};
        if (device.is_rocm())
            formats.insert(formats.end(), {MoERouterPreparedFormat::FP16, MoERouterPreparedFormat::BlockQ8});
        for (const auto format : formats)
        {
            prove(device, format, 2, 96, 17, {2, 8});
            prove(device, format, 65, 96, 17, {2, 3, 4, 8});
            prove(device, format, 512, 2048, 256, {2});
            prove(device, format, 33, 2080, 17, {2, 3});
            if (format != MoERouterPreparedFormat::BlockQ8)
                prove(device, format, 17, 31, 17, {2});
        }
    }
#ifdef HAVE_CUDA
    /** @test CUDA's native prefill FP32/BF16 rows retain exact selection bytes. */
    TEST(MoERouterOwnedRows, CUDA) { proveBackend(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
    /** @test Every ROCm prepared router format retains exact rows and input publication. */
    TEST(MoERouterOwnedRows, ROCm) { proveBackend(DeviceId::rocm(0)); }
#endif
}
