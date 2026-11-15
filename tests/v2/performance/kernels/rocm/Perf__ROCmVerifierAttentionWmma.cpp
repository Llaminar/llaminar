/**
 * @file Perf__ROCmVerifierAttentionWmma.cpp
 * @brief Paired complete-operation timing for serial-equivalent grouped attention.
 *
 * A retained transaction contains device parameter publication, the production
 * grouped producer and its original split reducer. Internal timeline events
 * bracket eight complete operations, excluding host submission gaps. Timing
 * changes only device-published live KV/query counts. Poison and independent
 * serial M=1 observations certify each timed pair before another launch can
 * repair its output. Profiling is a separate single-operation recording.
 */
#include <gtest/gtest.h>
#ifdef HAVE_ROCM
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/attention/AttentionWorkspaceContract.h"
#include "kernels/rocm/attention/ROCmFlashAttentionKernelT.h"
#include "kernels/rocm/attention/ROCmVerifierAttentionWmma.h"
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include "transfer/TransferEngine.h"
#include "utils/MPIContext.h"
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace llaminar2;
using Engine = rocm::VerifierAttentionEngine;
/** @brief Captured query/head geometry; the retained KV capacity is 64K. */
struct AttentionCase { int rows, heads, kv_heads; };
/** @brief Separate lifecycle proof, paired timing and isolated kernel profiling. */
enum class Evidence { Replay, Timing, Profile };
/** @brief Fail immediately on an incomplete native edge. @param ok Edge result. @param edge Diagnostic. */
void requireAttention(bool ok, const char *edge)
{ if (!ok) throw std::runtime_error(edge); }
/** @brief Retire every borrowed byte before releasing the observation event. */
struct Completion
{
    IBackend *backend;
    DeviceId device;
    void *stream;
    void *event;
    /** @brief Failed retirement is fatal; borrowed owners cannot safely unwind. */
    ~Completion()
    {
        if (!backend->synchronizeStream(stream, device.ordinal)) std::terminate();
        if (event) backend->destroyEvent(event, device.ordinal);
    }
};
/** @brief Independently filterable production grouped-attention workload. */
class ROCmVerifierAttentionPerf : public ::testing::TestWithParam<AttentionCase> {};

/** @brief Verify and time the actual public grouped attention operation.
 * @param c Retained query and head geometry. @param evidence Requested evidence scope.
 * Timings never include observations, allocations, transfers or profiler overhead.
 */
void runAttention(const AttentionCase &c, Evidence evidence)
{
    initializeROCmHardwareMetadata();
    int ordinal = -1;
    for (int i = 0; i < int(rocmHardwareMetadata().size()); ++i)
    {
        const auto &isa = rocmHardwareMetadata().at(i).isa_target;
        if (isa == "gfx1100" || isa.starts_with("gfx1100:")) { ordinal = i; break; }
    }
    if (ordinal < 0) GTEST_SKIP() << "Native gfx1100 is not visible";
    const bool profile = evidence == Evidence::Profile;
    if (profile && ::testing::UnitTest::GetInstance()->test_to_run_count() != 1)
        throw std::invalid_argument("attention profiling requires exactly one selected geometry");
    const auto device = DeviceId::rocm(ordinal);
    const auto &hardware = rocmHardwareMetadata().at(ordinal);
    std::cout << "ROCM_VERIFIER_ATTN_DEVICE," << ordinal << ',' << hardware.isa_target << ',' << hardware.pci_address << ',';
    for (auto byte : hardware.uuid) std::cout << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    std::cout << std::dec << std::setfill(' ') << ',' << hardware.compute_units << '\n';
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]
    {
        constexpr int width = 256, capacity = 65536;
        const int repeats = profile ? 1 : 8;
        const int samples = evidence == Evidence::Timing ? 31 : 3;
        auto *backend = getBackendFor(device);
        void *stream = worker.defaultStream();
        const auto hip_stream = static_cast<hipStream_t>(stream);
        requireAttention(backend && stream, "attention owning worker stream");
        auto &transfer = TransferEngine::instance();
        MPIContext mpi(0, 1, MPI_COMM_WORLD);
        const size_t q_columns = size_t(c.heads) * width;
        const size_t kv_columns = size_t(c.kv_heads) * width;
        std::vector<float> queries(size_t(c.rows) * q_columns);
        std::vector<uint16_t> keys(size_t(capacity) * kv_columns), values(keys.size());
        for (size_t i = 0; i < queries.size(); ++i) queries[i] = std::sin(float(i % 32749) * .031f) * .23f;
        for (size_t i = 0; i < keys.size(); ++i)
        {
            keys[i] = fp32_to_fp16(std::sin(float(i % 32749) * .017f) * .19f);
            values[i] = fp32_to_fp16(std::cos(float(i % 32749) * .023f) * .17f);
        }
        FP32Tensor query({size_t(c.rows), q_columns});
        std::copy(queries.begin(), queries.end(), query.mutable_data());
        FP16Tensor key({size_t(capacity), kv_columns}, std::move(keys));
        FP16Tensor value({size_t(capacity), kv_columns}, std::move(values));
        FP32Tensor dot_output({size_t(c.rows), q_columns}), wmma_output({size_t(c.rows), q_columns});
        FP32Tensor serial_query({size_t{1}, q_columns}), serial_output({size_t{1}, q_columns});
        INT32Tensor cached({size_t{1}}, std::vector<int32_t>{capacity});
        INT32Tensor active({size_t{1}}, std::vector<int32_t>{c.rows});
        std::array<FP32Tensor *, 2> outputs{&dot_output, &wmma_output};
        std::array<rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32>, 3> kernels{
            rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32>(ordinal),
            rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32>(ordinal),
            rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32>(ordinal)};
        const auto requirements = attention_workspace::requirements({
            .compact_query_rows = c.rows, .request_count = 1,
            .local_query_heads = c.heads, .local_kv_heads = c.kv_heads,
            .head_dim = width, .context_rows = capacity, .include_fp32_kv_conversion = false});
        std::array<std::unique_ptr<DeviceWorkspaceManager>, 3> workspaces;
        std::array<std::unique_ptr<IGPUGraphCapture>, 2> fragments, graphs;
        std::array<std::vector<uint32_t>, 2> observed;
        for (auto &out : observed) out.resize(queries.size());
        std::vector<uint32_t> serial(queries.size());
        std::array<int, 2> published{capacity, c.rows};
        Completion completion{backend, device, stream, backend->createEvent(ordinal)};
        requireAttention(completion.event, "attention observation event");
        for (int path = 0; path < 3; ++path)
        {
            workspaces[path] = std::make_unique<DeviceWorkspaceManager>(device, requirements.total_bytes_with_alignment());
            requireAttention(workspaces[path]->allocate(requirements), "attention persistent workspace");
            kernels[path].setGPUStream(stream);
            kernels[path].bindWorkspace(workspaces[path].get());
        }
        for (TensorBase *tensor : std::array<TensorBase *, 9>{&query, &key, &value, &dot_output, &wmma_output,
                &serial_query, &serial_output, &cached, &active})
            requireAttention(transfer.uploadFull(tensor, device, stream).success, "attention persistent input/output");
        const auto produce = [&](int path)
        {
            return kernels[path].prepareDynamicAttnParamsFromDeviceSequenceState(
                static_cast<const int *>(cached.gpu_data_ptr()), c.rows, c.rows, stream,
                capacity, static_cast<const int *>(active.gpu_data_ptr())) &&
                kernels[path].compute_verifier_rows_decode_equivalent(&query, &key, &value, outputs[path],
                    c.rows, capacity, c.heads, c.kv_heads, width, true, -1, &mpi, ordinal);
        };
        for (int path = 0; path < 2; ++path)
        {
            rocm::ScopedVerifierAttentionEngine engine(path ? Engine::WmmaLaneExact : Engine::OrderedDot2);
            requireAttention(produce(path) && backend->synchronizeStream(stream, ordinal), "attention warm preparation");
            graphs[path] = worker.createGraphCapture(stream);
            fragments[path] = graphs[path]->createOrderedTimelineFragment();
            requireAttention(bool(fragments[path]), "attention retained fragment");
            ScopedBackendGraphCapture capture(worker, *fragments[path], "grouped attention complete operation");
            requireAttention(capture.begin(), "attention capture begin");
            for (int repeat = 0; repeat < repeats; ++repeat)
                requireAttention(produce(path), "attention captured production operation");
            capture.finish();
            const std::array<GPUOrderedTimelineStep, 1> steps{{{
                .name = path ? "attention_wmma" : "attention_dot2",
                .kind = GPUOrderedTimelineStepKind::CapturedFragment, .capture = fragments[path].get()}}};
            requireAttention(graphs[path]->buildOrderedTimelineTransaction(steps, profile ?
                GPUOrderedTimelineInstrumentation::Disabled : GPUOrderedTimelineInstrumentation::PerStepEvents),
                "attention internal timing transaction");
            std::vector<GPUGraphKernelNodeInfo> nodes;
            requireAttention(graphs[path]->inspectKernelNodes(nodes), "attention native inventory");
            ASSERT_EQ(nodes.size(), size_t(repeats * 3));
            const char *name = path ? "flash_decode_verifier_wmma_fp16_hd256_kernel" : "flash_decoding_mi50_kernel";
            int producers = 0;
            for (const auto &node : nodes)
            {
                ASSERT_EQ(node.local_memory_bytes_per_thread, 0u) << node.name;
                if (node.name.find(name) == std::string::npos) continue;
                if (!producers++) std::cout << "ROCM_VERIFIER_ATTN_RESOURCE," << c.rows << ',' << c.heads << ',' << c.kv_heads
                    << ',' << path << ',' << node.grid_x << ',' << node.grid_y << ',' << node.grid_z << ',' << node.block_x
                    << ',' << node.registers_per_thread << ',' << node.static_shared_memory_bytes << ','
                    << node.max_active_blocks_per_sm << ',' << node.local_memory_bytes_per_thread << '\n';
            }
            ASSERT_EQ(producers, repeats);
            requireAttention(graphs[path]->instantiate(), "attention retained executable");
        }
        const auto publish = [&](int length, int live)
        {
            published = {length, live};
            requireAttention(backend->hostToDevice(cached.gpu_data_ptr(), &published[0], sizeof(int), ordinal, stream) &&
                backend->hostToDevice(active.gpu_data_ptr(), &published[1], sizeof(int), ordinal, stream), "attention device metadata");
            // Independent scalar calls have their own params and output banks.
            for (int row = 0; row < c.rows; ++row)
            {
                const int effective_live = std::max(live, 1);
                const int visible = row < effective_live ? std::max(1, length - effective_live + row + 1) : 1;
                requireAttention(backend->hostToDevice(serial_query.gpu_data_ptr(), queries.data() + size_t(row) * q_columns,
                    q_columns * sizeof(float), ordinal, stream), "attention serial query");
                requireAttention(kernels[2].prepareDynamicAttnParams(visible, visible - 1, 1, stream, capacity) &&
                    kernels[2].compute_tensor(&serial_query, &key, &value, &serial_output, 1, 1, visible,
                        c.heads, c.kv_heads, width, true, -1, nullptr, nullptr, &mpi, ordinal), "attention serial witness");
                requireAttention(backend->deviceToHost(serial.data() + size_t(row) * q_columns, serial_output.gpu_data_ptr(),
                    q_columns * sizeof(uint32_t), ordinal, stream), "attention serial observation");
            }
            requireAttention(backend->synchronizeStream(stream, ordinal), "attention witness completion");
        };
        const auto poison = [&](int path)
        {
            requireAttention(hipMemsetAsync(outputs[path]->gpu_data_ptr(), 0xa5, queries.size() * sizeof(float), hip_stream) == hipSuccess,
                "attention output poison");
            for (const char *name : {rocm::AttentionWorkspaceBuffers::PARTIAL_OUTPUT, rocm::AttentionWorkspaceBuffers::PARTIAL_M,
                                    rocm::AttentionWorkspaceBuffers::PARTIAL_L})
                requireAttention(hipMemsetAsync(workspaces[path]->getBuffer(name), 0xa5, workspaces[path]->getBufferSize(name), hip_stream) == hipSuccess,
                    "attention partial poison");
        };
        const auto observe = [&]
        {
            for (int path = 0; path < 2; ++path)
                requireAttention(backend->deviceToHost(observed[path].data(), outputs[path]->gpu_data_ptr(),
                    queries.size() * sizeof(float), ordinal, stream), "attention timed pair observation");
            requireAttention(backend->synchronizeStream(stream, ordinal), "attention pair observation completion");
            ASSERT_EQ(observed[0], serial);
            ASSERT_EQ(observed[1], serial);
        };
        const auto verify = [&](int length, int live)
        {
            publish(length, live);
            for (int path = 0; path < 2; ++path) { poison(path); requireAttention(graphs[path]->launch(), "attention dirty replay"); }
            observe();
        };
        for (const auto [length, live] : std::array<std::pair<int, int>, 5>{{{capacity,c.rows},{17,1},{1,0},{513,c.rows-1},{capacity,c.rows}}})
        {
            verify(length, live);
            if (::testing::Test::HasFatalFailure()) return;
        }
        if (profile)
        {
            for (int warm = 0; warm < 10; ++warm) requireAttention(graphs[1]->launch(), "attention profile warmup");
            requireAttention(backend->synchronizeStream(stream, ordinal), "attention profile readiness");
            requireAttention(roctxRangePushA("ROCmVerifierAttentionWmma") >= 0, "attention profile range");
            if (roctxProfilerResume(0) != 0) { (void)roctxRangePop(); throw std::runtime_error("attention profiler resume"); }
            const bool launched = graphs[1]->launch();
            const bool joined = backend->synchronizeStream(stream, ordinal);
            const int paused = roctxProfilerPause(0), popped = roctxRangePop();
            requireAttention(launched && joined && paused == 0 && popped >= 0, "attention profiled operation");
            return;
        }
        const std::vector<int> lengths = evidence == Evidence::Timing
            ? std::vector<int>{64,256,512,1024,4096,8192,16384,32768,65536} : std::vector<int>{65536,17,65536};
        for (const int length : lengths)
        {
            verify(length, c.rows);
            if (::testing::Test::HasFatalFailure()) return;
            for (int warm = 0; warm < 10; ++warm)
                for (auto &graph : graphs) requireAttention(graph->launch(), "attention paired warmup");
            std::array<std::vector<double>, 2> times;
            for (auto &bank : times) bank.resize(samples);
            for (int sample = 0; sample < samples; ++sample)
            {
                for (int order = 0; order < 2; ++order)
                {
                    const int path = order ^ (sample & 1);
                    poison(path);
                    requireAttention(graphs[path]->launch() && backend->recordEvent(completion.event, ordinal, stream) &&
                        backend->waitForEvent(completion.event, ordinal), "attention timed completion");
                    const auto timing = graphs[path]->consumeOrderedTimelineTiming();
                    requireAttention(timing.state == GPUOrderedTimelineTimingState::Complete && timing.samples.size() == 1,
                        "attention fresh internal timing");
                    const auto &record = timing.samples.front();
                    requireAttention(record.name == (path ? "attention_wmma" : "attention_dot2") &&
                        record.kind == GPUOrderedTimelineStepKind::CapturedFragment && std::isfinite(record.elapsed_ms) && record.elapsed_ms > 0,
                        "attention timing identity and interval");
                    times[path][sample] = record.elapsed_ms * 1000.0 / repeats;
                }
                observe();
                if (::testing::Test::HasFatalFailure()) return;
                std::cout << std::setprecision(9) << "ROCM_VERIFIER_ATTN_PAIR," << c.rows << ',' << c.heads << ',' << c.kv_heads
                    << ',' << capacity << ',' << length << ',' << c.rows << ',' << sample << ',' << times[0][sample] << ',' << times[1][sample] << '\n';
            }
            for (auto &bank : times) std::sort(bank.begin(), bank.end());
            std::cout << "ROCM_VERIFIER_ATTN_MEDIAN," << c.rows << ',' << c.heads << ',' << c.kv_heads << ',' << length
                << ',' << times[0][samples/2] << ',' << times[1][samples/2] << ',' << times[0][samples/2]/times[1][samples/2] << '\n';
        }
        for (const auto [length, live] : std::array<std::pair<int,int>, 3>{{{1,0},{64,1},{capacity,c.rows}}})
        { verify(length, live); if (::testing::Test::HasFatalFailure()) return; }
        for (auto &kernel : kernels) kernel.unbindWorkspace();
    });
}
/** @brief Repeated timing must be fresh and match serial output after reset/regrowth. */
TEST(ROCmVerifierAttentionTiming, CapturedEventBoundaryReplay)
{ runAttention({9,8,1}, Evidence::Replay); }
/** @brief Explicit paired tournament; timing noise never becomes a correctness gate. */
TEST_P(ROCmVerifierAttentionPerf, DISABLED_PairedPrefixScaling)
{ runAttention(GetParam(), Evidence::Timing); }
/** @brief One native candidate and one exact context per profiler process. */
TEST_P(ROCmVerifierAttentionPerf, DISABLED_Profile)
{ runAttention(GetParam(), Evidence::Profile); }
INSTANTIATE_TEST_SUITE_P(NativeGfx1100, ROCmVerifierAttentionPerf,
    ::testing::Values(AttentionCase{2,8,1}, AttentionCase{4,8,1}, AttentionCase{8,8,1}, AttentionCase{9,8,1}, AttentionCase{16,8,1},
                      AttentionCase{2,16,2}, AttentionCase{4,16,2}, AttentionCase{8,16,2}, AttentionCase{9,16,2}, AttentionCase{16,16,2}),
    [](const auto &info) { return "M" + std::to_string(info.param.rows) + "H" + std::to_string(info.param.heads) + "KV" + std::to_string(info.param.kv_heads); });
} // namespace
#endif // HAVE_ROCM
