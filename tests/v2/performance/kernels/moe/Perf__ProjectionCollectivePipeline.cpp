/**
 * @file Perf__ProjectionCollectivePipeline.cpp
 * @brief Isolate row-pipelined output projection and native TP allreduce.
 *
 * Each participant retains one complete captured transaction. The control
 * computes the full projection before reducing it; the candidate publishes
 * disjoint row tiles to a collective stream while computing the next tile.
 * A tiled, same-stream control separates launch/geometry costs from overlap.
 * All variants reuse the same prepared Q6_K weights, FP32 input/output and
 * maximum workspace. No format, arithmetic, transport or memory trade is made.
 *
 * This is a fixed-live-extent scheduling experiment, not a dynamic-count
 * protocol, production lowering, model correctness or whole-model speed proof.
 * Native messages cover only the rows actually computed. Output guard rows
 * and byte comparison against the untiled captured transaction detect slicing
 * mistakes. Setup, result downloads and terminal event waits are not timed.
 */
#include <gtest/gtest.h>
#include "backends/GPUDeviceContextPool.h"
#include "collective/coordinators/NCCLCoordinator.h"
#include "collective/coordinators/RCCLCoordinator.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "transfer/TransferEngine.h"
#include "utils/GpuPreparedGemmHarness.h"
#include "utils/TestTensorFactory.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace llaminar2::test
{
namespace
{
constexpr int kParticipants = 2;
constexpr int kColumns = 2048;
constexpr int kReduction = 2048;
constexpr int kGuardRows = 3;
constexpr int kWarmups = 5;
constexpr int kSamples = 25;
constexpr int kOperations = 8;
constexpr float kGuard = -9876.5f;

/** @brief Keep failure unwinding active instead of returning from worker assertions. */
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

/** @brief The scheduling choice never changes tensor or wire precision. */
enum class Schedule { Serial, Overlap };

/** @brief One retained shape/schedule, shared identically by both participants. */
struct Variant
{
    int tiles;
    Schedule schedule;
};

/** @brief Own one exact-device event for setup or terminal timing. */
class Event final
{
public:
    /** @brief Create before capture; timing events remain outside the graph. */
    Event(IBackend &backend, DeviceId device, bool timing = false)
        : backend_(backend), device_(device), event_(timing
            ? backend.createTimingEvent(device.ordinal) : backend.createEvent(device.ordinal))
    {
        require(event_ != nullptr, "projection overlap event creation");
    }
    /** @brief Graphs and terminal work have retired before these handles. */
    ~Event() { backend_.destroyEvent(event_, device_.ordinal); }
    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;
    /** @brief Publish the preceding work on this exact producer stream. */
    void record(void *stream) { require(backend_.recordEvent(event_, device_.ordinal, stream), "projection event record"); }
    /** @brief Test-only terminal observation after the full graph has launched. */
    void join() { require(backend_.waitForEvent(event_, device_.ordinal), "projection terminal observation"); }
    /** @return Elapsed native event interval in microseconds. */
    double since(const Event &begin)
    {
        float ms = 0;
        require(backend_.eventElapsedTimeMs(begin.event_, event_, device_.ordinal, &ms), "projection event elapsed");
        return double(ms) * 1000.;
    }
private:
    IBackend &backend_;
    DeviceId device_;
    void *event_;
};

/** @brief An internal graph edge, deliberately distinct from external publication. */
class GraphEdge final
{
public:
    /** @brief Allocate one ordering event from its exact worker before capture. */
    explicit GraphEdge(IWorkerGPUContext &gpu) : gpu_(gpu), event_(gpu.createEvent())
    {
        require(event_ != nullptr, "projection graph event creation");
    }
    /** @brief Destroy after every graph using this edge has retired. */
    ~GraphEdge() { gpu_.destroyEvent(event_); }
    GraphEdge(const GraphEdge &) = delete;
    GraphEdge &operator=(const GraphEdge &) = delete;
    /** @brief Fork a recorded producer frontier onto its exact consumer stream. */
    void connect(void *producer, void *consumer)
    {
        require(gpu_.recordEventChecked(event_, producer) && gpu_.waitEventChecked(event_, consumer),
            "projection internal graph dependency");
    }
private:
    IWorkerGPUContext &gpu_;
    void *event_;
};

/**
 * @brief Test-only FP32 output slice of one immutable, retained GPU allocation.
 *
 * Ordinary FP32 create_view() describes host storage and can acquire a separate
 * GPU allocation; it is not a device alias. This diagnostic wrapper lends the
 * exact parent interval to multiply_tensor(), without a second allocation or
 * copy. It is output-only and never claims an independent input publication.
 * A production lowerer would need an admitted arena subregion contract.
 */
class OutputTile final : public FP32Tensor
{
public:
    /** @brief Bounds-check and retain the parent before lending its device rows. */
    OutputTile(std::shared_ptr<FP32Tensor> parent, DeviceId device, int first_row, int rows)
        : FP32Tensor({size_t(rows), kColumns}, device), parent_(std::move(parent))
    {
        require(parent_->current_device() == device && parent_->gpu_data_ptr() && first_row >= 0 && rows > 0 &&
            size_t(first_row + rows) * kColumns <= parent_->numel(), "projection output slice bounds/device");
        gpu_data_ptr_ = static_cast<float *>(parent_->gpu_data_ptr()) + size_t(first_row) * kColumns;
        gpu_device_ = device;
    }
    /** @brief Relinquish the borrowed address; only the retained parent frees it. */
    ~OutputTile() override
    {
        gpu_data_ptr_ = nullptr;
        gpu_device_.reset();
    }
private:
    std::shared_ptr<FP32Tensor> parent_;
};

/** @brief A prepared kernel borrows its workspace and stream only inside this scope. */
struct Binding
{
    ITensorGemm &kernel;
    IWorkspaceConsumer &consumer;
    /** @brief Release borrowed bindings before workspace retirement, including failures. */
    ~Binding() { kernel.clearGPUStreamBinding(); consumer.unbindWorkspace(); }
};

/** @brief Abort collectives before joining a failed participant's native work. */
template <typename Coordinator>
struct Join
{
    IWorkerGPUContext &gpu;
    Coordinator &coordinator;
    void *compute;
    void *collective;
    int exceptions = std::uncaught_exceptions();
    /** @brief Fixture-only lifetime join; never inside a recorded transaction. */
    ~Join()
    {
        if (std::uncaught_exceptions() > exceptions) coordinator.abortCommunicators();
        (void)gpu.synchronizeStreamChecked(compute);
        (void)gpu.synchronizeStreamChecked(collective);
    }
};

/**
 * @brief Measure one homogeneous pair using its production GEMM and native library.
 * @param first Visible device zero; environment visibility selects physical endpoints.
 * @param rows Exact live rows, not an allocation or bucket capacity.
 */
template <typename Coordinator>
void measure(DeviceId first, int rows)
{
    auto *backend = getBackendFor(first);
    ASSERT_NE(backend, nullptr);
    ASSERT_GE(backend->deviceCount(), kParticipants);
    std::vector<Variant> variants;
    for (int tiles : {1, 2, 4, 8})
    {
        if (rows % tiles != 0 || rows / tiles < 32) continue;
        variants.push_back({tiles, Schedule::Serial});
        if (tiles > 1) variants.push_back({tiles, Schedule::Overlap});
    }
    Coordinator coordinator;
    ASSERT_TRUE(coordinator.initialize({0, 1}));
    std::barrier rendezvous(kParticipants);
    std::array<std::exception_ptr, kParticipants> errors;
    std::array<std::thread, kParticipants> threads;
    std::array<std::vector<std::vector<double>>, kParticipants> timings;

    for (int participant = 0; participant < kParticipants; ++participant)
    {
        threads[participant] = std::thread([&, participant]
        {
            try
            {
                const auto device = first.is_cuda() ? DeviceId::cuda(participant) : DeviceId::rocm(participant);
                auto &gpu = GPUDeviceContextPool::instance().getContext(device);
                gpu.submitAndWait([&]
                {
                    auto *compute = gpu.getOrCreateAuxiliaryStream("projection_overlap_compute");
                    auto *collective = gpu.getOrCreateAuxiliaryStream("projection_overlap_collective");
                    auto weights = TestTensorFactory::createQ6_KRandom({kColumns, kReduction}, 182u + participant);
                    auto prepared = makeGpuPreparedGemm(weights.get(), device, "perf.projection_overlap.weight");
                    auto &kernel = *prepared.kernel;
                    auto *consumer = dynamic_cast<IWorkspaceConsumer *>(&kernel);
                    require(consumer != nullptr, "projection overlap workspace consumer");
                    WorkspaceRequirements requirements;
                    for (const auto variant : variants)
                        requirements.merge(consumer->getWorkspaceRequirements(rows / variant.tiles, kColumns, kReduction));
                    DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment());
                    require(workspace.allocate(requirements), "projection overlap workspace admission");
                    consumer->bindWorkspace(&workspace);
                    kernel.setGPUStream(compute);
                    Binding binding{kernel, *consumer};
                    std::shared_ptr<FP32Tensor> input = TestTensorFactory::createFP32Random(
                        {size_t(rows), kReduction}, -.25f, .25f, 57u + participant);
                    auto output = std::make_shared<FP32Tensor>(std::vector<size_t>{size_t(rows + kGuardRows), kColumns});
                    const size_t output_elements = size_t(rows + kGuardRows) * kColumns;
                    std::fill_n(output->mutable_data(), output_elements, kGuard);
                    require(input->ensureOnDevice(device, compute) && output->ensureOnDevice(device, compute),
                        "projection overlap input/output admission");
                    Event start(*backend, device, true), stop(*backend, device, true);
                    GraphEdge joined(gpu);
                    std::vector<std::unique_ptr<GraphEdge>> ready;
                    for (int tile = 0; tile < variants.back().tiles; ++tile)
                        ready.push_back(std::make_unique<GraphEdge>(gpu));
                    require(gpu.synchronizeStreamChecked(compute), "projection overlap setup completion");
                    /** @brief Views and graph handles share the already admitted output. */
                    struct Transaction
                    {
                        std::vector<std::shared_ptr<TensorBase>> outputs;
                        std::unique_ptr<IGPUGraphCapture> graph;
                    };
                    std::vector<Transaction> transactions(variants.size());
                    Join lifetime{gpu, coordinator, compute, collective};
                    for (size_t index = 0; index < variants.size(); ++index)
                    {
                        const auto variant = variants[index];
                        const int tile_rows = rows / variant.tiles;
                        auto &transaction = transactions[index];
                        for (int tile = 0; tile < variant.tiles; ++tile)
                        {
                            transaction.outputs.push_back(std::make_shared<OutputTile>(output, device, tile * tile_rows, tile_rows));
                        }
                        transaction.graph = gpu.createGraphCapture(compute);
                        require(bool(transaction.graph), "projection overlap graph owner");
                        // A host setup join does not authenticate coherence:
                        // acquire every external producer on the capture's
                        // exact stream before recording its first consumer.
                        TransferEngine::requireDeviceInput(input.get(), device, compute);
                        rendezvous.arrive_and_wait();
                        ScopedBackendGraphCapture recording(gpu, *transaction.graph, "projection native overlap");
                        require(recording.begin(), "projection overlap capture begin");
                        for (int operation = 0; operation < kOperations; ++operation)
                        {
                            for (int tile = 0; tile < variant.tiles; ++tile)
                            {
                                auto *destination = transaction.outputs[tile].get();
                                require(kernel.multiply_tensor(input.get(), destination, tile_rows, kColumns, kReduction,
                                    true, 1.f, 0.f, nullptr, nullptr, device.ordinal, &workspace, tile * tile_rows),
                                    "projection overlap GEMM");
                                // Distinct row intervals permit GEMM(tile+1)
                                // alongside native sum(tile). The scratch is
                                // compute-only and never borrowed by NCCL/RCCL.
                                void *sum_stream = compute;
                                if (variant.schedule == Schedule::Overlap)
                                {
                                    ready[tile]->connect(compute, collective);
                                    sum_stream = collective;
                                }
                                require(coordinator.allreduceSingleDeviceOnStream(destination->gpu_data_ptr(),
                                    size_t(tile_rows) * kColumns, CollectiveDataType::FLOAT32,
                                    CollectiveOp::ALLREDUCE_SUM, participant, sum_stream), "projection native allreduce");
                            }
                            if (variant.schedule == Schedule::Overlap)
                            {
                                // Join before either next-layer consumption or
                                // reuse. No participant can race its own output.
                                joined.connect(collective, compute);
                            }
                        }
                        rendezvous.arrive_and_wait();
                        recording.finish();
                        require(transaction.graph->instantiate(), "projection overlap graph instantiate");
                    }

                    std::vector<float> reference(output_elements), actual(reference.size());
                    const auto observe = [&](std::vector<float> &host)
                    {
                        require(backend->deviceToHostOnStream(host.data(), output->gpu_data_ptr(), output->size_bytes(),
                            device.ordinal, compute), "projection overlap result copy");
                        stop.record(compute); stop.join();
                        require(std::all_of(host.begin(), host.end(), [](float x) { return std::isfinite(x); }),
                            "projection overlap finite output");
                        require(std::all_of(host.begin() + size_t(rows) * kColumns, host.end(),
                            [](float x) { return x == kGuard; }), "projection overlap modified guard rows");
                    };
                    rendezvous.arrive_and_wait();
                    require(transactions.front().graph->launch(), "projection overlap reference replay");
                    observe(reference);
                    require(std::any_of(reference.begin(), reference.begin() + size_t(rows) * kColumns,
                        [](float x) { return x != 0.f; }), "projection overlap nonzero witness");
                    timings[participant].resize(variants.size());
                    for (int sample = -kWarmups; sample < kSamples; ++sample)
                    {
                        // Rotate/reverse candidate order to reduce thermal and
                        // transport startup bias without changing peer order.
                        for (size_t order = 0; order < variants.size(); ++order)
                        {
                            const size_t index = sample % 2 == 0 ? order : variants.size() - order - 1;
                            rendezvous.arrive_and_wait();
                            start.record(compute);
                            require(transactions[index].graph->launch(), "projection overlap timed replay");
                            stop.record(compute); stop.join();
                            if (sample >= 0) timings[participant][index].push_back(stop.since(start) / kOperations);
                            observe(actual);
                            for (size_t element = 0; element < actual.size(); ++element)
                                if (std::bit_cast<uint32_t>(actual[element]) != std::bit_cast<uint32_t>(reference[element]))
                                    throw std::runtime_error("projection overlap byte mismatch: tiles=" +
                                        std::to_string(variants[index].tiles) + " element=" + std::to_string(element));
                        }
                    }
                    rendezvous.arrive_and_wait();
                });
            }
            catch (...)
            {
                errors[participant] = std::current_exception();
                coordinator.abortCommunicators();
                rendezvous.arrive_and_drop();
            }
        });
    }
    for (auto &thread : threads) thread.join();
    for (const auto &error : errors) if (error)
        try { std::rethrow_exception(error); } catch (const std::exception &failure) { ADD_FAILURE() << failure.what(); }
    if (::testing::Test::HasFailure()) return;
    for (size_t index = 0; index < variants.size(); ++index)
    {
        std::vector<double> critical;
        for (int sample = 0; sample < kSamples; ++sample)
        {
            const double us = std::max(timings[0][index][sample], timings[1][index][sample]);
            critical.push_back(us);
            std::printf("projection_collective_sample,%s,%d,%d,%s,%d,%.3f\n", first.is_cuda() ? "CUDA" : "ROCm",
                rows, variants[index].tiles, variants[index].schedule == Schedule::Serial ? "serial" : "overlap", sample, us);
        }
        std::sort(critical.begin(), critical.end());
        std::printf("PROJECTION_COLLECTIVE,%s,rows=%d,N=%d,K=%d,format=Q6_K,tiles=%d,schedule=%s,"
            "median_us=%.3f,p10_us=%.3f,p90_us=%.3f,payload_bytes=%zu,byte_exact=1\n", first.is_cuda() ? "CUDA" : "ROCm",
            rows, kColumns, kReduction, variants[index].tiles,
            variants[index].schedule == Schedule::Serial ? "serial" : "overlap", critical[kSamples / 2],
            critical[kSamples / 10], critical[kSamples * 9 / 10], size_t(rows) * kColumns * sizeof(float));
    }
}

/** @brief Fixed-live production geometries; one backend/shape per profiler launch. */
class ProjectionCollectivePipelinePerf : public ::testing::TestWithParam<int> {};
#ifdef HAVE_CUDA
TEST_P(ProjectionCollectivePipelinePerf, CUDA) { measure<NCCLCoordinator>(DeviceId::cuda(0), GetParam()); }
#endif
#ifdef HAVE_ROCM
TEST_P(ProjectionCollectivePipelinePerf, ROCm) { measure<RCCLCoordinator>(DeviceId::rocm(0), GetParam()); }
#endif
INSTANTIATE_TEST_SUITE_P(Captured, ProjectionCollectivePipelinePerf, ::testing::Values(64, 448, 512),
    [](const ::testing::TestParamInfo<int> &info) { return "Rows" + std::to_string(info.param); });
} // namespace
} // namespace llaminar2::test
