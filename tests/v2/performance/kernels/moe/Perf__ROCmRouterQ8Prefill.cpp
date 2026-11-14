/**
 * @file Perf__ROCmRouterQ8Prefill.cpp
 * @brief Captured production Q8 router tournament with raw paired native-event samples.
 *
 * Both candidates include hidden quantization, logits and original expert-bank
 * restoration. Poisoned retained replay authenticates outputs and side products
 * before and after timing. Setup, preparation and observations stay outside the
 * measured graph. Parameter names isolate one geometry and basis for profiling.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/RouterQ8Launch.h"
#include "../../../utils/TestTensorFactory.h"
#include "ROCmRouterModelInventoryGenerated.h"
#include "utils/PrefillGraphBucketDefaults.h"
#include "models/qwen4exp/Qwen4ExpPublishedGeometry.h"
#ifdef HAVE_ROCM
#include <rocprofiler-sdk-roctx/roctx.h>
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "kernels/rocm/moe/ROCmRouterQ8Prefill.h"
#endif
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <set>
#include <vector>

namespace
{
#ifdef HAVE_ROCM
using namespace llaminar2;
using namespace llaminar2::test;
/** @brief Exact captured geometry and paired gate/input basis. */
struct RouterCase { int rows, width, experts; RouterQ8Basis basis; };
/** @brief Fail at the first incomplete backend edge, across the worker boundary. */
void routerRequire(bool success, const char *edge)
{ if (!success) throw std::runtime_error(edge); }
/** @brief Own native timing events and drain only at the test observation boundary. */
struct RouterEvents
{
    IBackend *backend;
    DeviceId device;
    void *stream;
    void *start;
    void *stop;
    /** @brief Release events after every outstanding test observation completes. */
    ~RouterEvents()
    {
        (void)backend->synchronizeStream(stream, device.ordinal);
        if (start) backend->destroyEvent(start, device.ordinal);
        if (stop) backend->destroyEvent(stop, device.ordinal);
    }
};
/** @brief Production router component fixture, independently filterable by shape. */
class ROCmRouterQ8Prefill : public ::testing::TestWithParam<RouterCase> {};

/** @brief Verify one production component and either measure it or profile one native replay.
 * @param c Complete immutable geometry and source basis. @param profile Collect one selected region. */
void benchmarkRouter(const RouterCase &c, bool profile)
{
    if (profile && ::testing::UnitTest::GetInstance()->test_to_run_count() != 1)
        throw std::invalid_argument("isolated router profiling requires exactly one selected test case");
    initializeROCmHardwareMetadata();
    int ordinal = -1;
    for (int i = 0; i < int(rocmHardwareMetadata().size()); ++i)
    {
        const auto &isa = rocmHardwareMetadata().at(i).isa_target;
        if (isa == "gfx1100" || isa.starts_with("gfx1100:")) { ordinal = i; break; }
    }
    if (ordinal < 0) GTEST_SKIP() << "Native gfx1100 is not visible";
    const auto device = DeviceId::rocm(ordinal);
    const auto &hardware = rocmHardwareMetadata().at(ordinal);
    std::cout << "ROCM_ROUTER_DEVICE," << ordinal << ',' << hardware.isa_target << ',' << hardware.pci_address << ',';
    for (auto byte : hardware.uuid) std::cout << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    std::cout << std::dec << std::setfill(' ') << ',' << hardware.compute_units << ',' << hardware.native_wave_width << '\n';
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&]
    {
        auto *backend = getBackendFor(device);
        void *stream = context.defaultStream();
        routerRequire(backend && stream, "router benchmark worker stream");
        const size_t inputs = size_t(c.rows + 1) * c.width;
        const size_t outputs = size_t(c.rows + 1) * c.experts;
        const size_t weights = size_t(c.experts) * c.width;
        auto hidden = TestTensorFactory::createFP32({inputs});
        auto gate = TestTensorFactory::createFP32({weights});
        auto gate_codes = TestTensorFactory::createINT32({weights / 4});
        auto gate_scales = TestTensorFactory::createFP32({weights / 32});
        auto live = TestTensorFactory::createINT32({1});
        for (size_t i = 0; i < inputs; ++i)
            hidden->mutable_data()[i] = std::sin(float(i % 8191) * .017f) * .37f;
        for (size_t i = 0; i < weights; ++i)
            gate->mutable_data()[i] = std::cos(float(i % 8191) * .031f) * .19f;
        live->mutable_int32_data()[0] = c.rows;
        for (auto *t : std::array<TensorBase *,5>{hidden.get(),gate.get(),gate_codes.get(),gate_scales.get(),live.get()})
            routerRequire(t->ensureOnDevice(device,stream), "router benchmark source storage");
        auto fp = [](auto &t) { return static_cast<float *>(t->gpu_data_ptr()); };
        auto codes = [](auto &t) { return static_cast<int8_t *>(t->gpu_data_ptr()); };
        routerRequire(hipMoE_quantize_router_gate_q8(fp(gate),codes(gate_codes),fp(gate_scales),
            c.width,c.experts,device.ordinal,stream,c.basis), "router benchmark gate preparation");
        std::array<std::unique_ptr<TensorBase>,2> q, scales, out;
        auto baseline = context.createGraphCapture(stream), candidate = context.createGraphCapture(stream);
        std::array graphs{baseline.get(),candidate.get()};
        RouterEvents events{backend,device,stream,backend->createTimingEvent(ordinal),backend->createTimingEvent(ordinal)};
        routerRequire(events.start && events.stop, "router benchmark native events");
        const int replays = profile ? 1 : 16;
        constexpr int samples = 31, warmups = 10;
        for (int path = 0; path < 2; ++path)
        {
            q[path] = TestTensorFactory::createINT32({inputs/4});
            scales[path] = TestTensorFactory::createFP32({inputs/32});
            out[path] = TestTensorFactory::createFP32({outputs});
            for (auto *t : {q[path].get(),scales[path].get(),out[path].get()})
                routerRequire(t->ensureOnDevice(device,stream), "router benchmark destination storage");
            rocm::ScopedRouterQ8PrefillMode policy(path ? rocm::RouterQ8PrefillMode::NativeWmma : rocm::RouterQ8PrefillMode::OrderedDot4);
            const auto produce = [&]
            {
                return hipMoE_gate_logits_q8_weights_decode_equivalent_rows(fp(hidden),codes(q[path]),fp(scales[path]),
                    codes(gate_codes),fp(gate_scales),fp(out[path]),c.rows,c.width,c.experts,ordinal,stream,
                    static_cast<const int *>(live->gpu_data_ptr()),c.basis);
            };
            routerRequire(produce() && backend->synchronizeStream(stream,ordinal), "router benchmark preparation");
            ScopedBackendGraphCapture capture(context,*graphs[path],"router Q8 production component");
            routerRequire(capture.begin(), "router benchmark capture begin");
            for (int iteration = 0; iteration < replays; ++iteration)
                routerRequire(produce(), "router benchmark captured producer");
            capture.finish();
            std::vector<GPUGraphKernelNodeInfo> nodes;
            ASSERT_TRUE(graphs[path]->inspectKernelNodes(nodes));
            ASSERT_EQ(nodes.size(), size_t(replays * (c.basis == RouterQ8Basis::Original ? 2 : 3)));
            int matrix_nodes = 0;
            bool reported = false;
            for (const auto &node : nodes)
            {
                ASSERT_EQ(node.local_memory_bytes_per_thread,0u) << node.name;
                if (node.name.find("router_q8_prefill_detail") != std::string::npos) ++matrix_nodes;
                if (!reported && (node.name.find("router_q8_prefill_detail") != std::string::npos ||
                    node.name.find("gate_logits_q8_grouped_verifier_kernel") != std::string::npos))
                {
                    reported = true;
                    std::cout << "ROCM_ROUTER_RESOURCE," << c.rows << ',' << c.width << ',' << c.experts << ','
                        << int(c.basis) << ',' << path << ',' << node.registers_per_thread << ','
                        << node.static_shared_memory_bytes << ',' << node.local_memory_bytes_per_thread << ','
                        << node.max_active_blocks_per_sm << ',' << node.block_x << ',' << node.grid_x << ',' << node.grid_y
                        << ',' << node.name << '\n';
                }
            }
            ASSERT_EQ(matrix_nodes, path ? replays : 0);
            routerRequire(graphs[path]->instantiate(), "router benchmark instantiate");
        }
        std::vector<float> poison(outputs,-12345.f), a(outputs), b(outputs);
        std::vector<int8_t> poison_q(inputs,23), aq(inputs), bq(inputs);
        std::vector<float> poison_s(inputs/32,-17.f), as(inputs/32), bs(inputs/32);
        const auto verify = [&](int count)
        {
            routerRequire(backend->hostToDevice(live->gpu_data_ptr(),&count,sizeof(count),ordinal,stream), "router benchmark live publication");
            for (int path = 0; path < 2; ++path)
            {
                routerRequire(backend->hostToDevice(fp(out[path]),poison.data(),outputs*4,ordinal,stream) &&
                    backend->hostToDevice(codes(q[path]),poison_q.data(),inputs,ordinal,stream) &&
                    backend->hostToDevice(fp(scales[path]),poison_s.data(),poison_s.size()*4,ordinal,stream) &&
                    graphs[path]->launch(), "router benchmark poisoned replay");
            }
            routerRequire(backend->deviceToHost(a.data(),fp(out[0]),outputs*4,ordinal,stream) &&
                backend->deviceToHost(b.data(),fp(out[1]),outputs*4,ordinal,stream) &&
                backend->deviceToHost(aq.data(),codes(q[0]),inputs,ordinal,stream) &&
                backend->deviceToHost(bq.data(),codes(q[1]),inputs,ordinal,stream) &&
                backend->deviceToHost(as.data(),fp(scales[0]),as.size()*4,ordinal,stream) &&
                backend->deviceToHost(bs.data(),fp(scales[1]),bs.size()*4,ordinal,stream) &&
                backend->synchronizeStream(stream,ordinal), "router benchmark terminal observation");
            ASSERT_EQ(std::memcmp(a.data(),b.data(),outputs*4),0) << "live=" << count;
            ASSERT_EQ(aq,bq);
            ASSERT_EQ(std::memcmp(as.data(),bs.data(),as.size()*4),0);
            for (size_t i = size_t(count)*c.experts; i < outputs; ++i) ASSERT_EQ(b[i],poison[i]);
            for (size_t i = size_t(count)*c.width; i < inputs; ++i) ASSERT_EQ(bq[i],poison_q[i]);
            for (size_t i = size_t(count)*c.width/32; i < bs.size(); ++i) ASSERT_EQ(bs[i],poison_s[i]);
        };
        for (int count : {c.rows,0,1,c.rows/2,c.rows}) { verify(count); if (::testing::Test::HasFatalFailure()) return; }
        for (int iteration = 0; iteration < warmups; ++iteration)
            for (auto *graph : graphs) routerRequire(graph->launch(), "router benchmark warmup");
        if (profile)
        {
            routerRequire(backend->synchronizeStream(stream,ordinal), "router profile preconditioning");
            routerRequire(roctxRangePushA("ROCmRouterQ8NativeWmma") >= 0, "router profile range");
            if (roctxProfilerResume(0) != 0)
            {
                (void)roctxRangePop();
                throw std::runtime_error("router profile resume");
            }
            const bool launched = graphs[1]->launch();
            const bool completed = backend->synchronizeStream(stream,ordinal);
            const int paused = roctxProfilerPause(0), popped = roctxRangePop();
            routerRequire(launched && completed && paused == 0 && popped >= 0, "router profile replay");
            std::cout << "ROCM_ROUTER_PROFILE," << c.rows << ',' << c.width << ',' << c.experts << ',' << int(c.basis) << ",1\n";
            return;
        }
        std::array<std::array<double,samples>,2> timings{};
        for (int sample = 0; sample < samples; ++sample)
            for (int order = 0; order < 2; ++order)
            {
                const int path = order ^ (sample & 1);
                routerRequire(backend->recordEvent(events.start,ordinal,stream) && graphs[path]->launch() &&
                    backend->recordEvent(events.stop,ordinal,stream) && backend->waitForEvent(events.stop,ordinal), "router benchmark timing");
                float ms = 0;
                routerRequire(backend->eventElapsedTimeMs(events.start,events.stop,ordinal,&ms), "router benchmark elapsed events");
                routerRequire(std::isfinite(ms) && ms > 0, "router benchmark requires positive finite elapsed time");
                timings[path][sample] = ms * 1000.0 / replays;
            }
        // Observe timed outputs before another producer could repair an error.
        routerRequire(backend->deviceToHost(a.data(),fp(out[0]),outputs*4,ordinal,stream) &&
            backend->deviceToHost(b.data(),fp(out[1]),outputs*4,ordinal,stream) &&
            backend->deviceToHost(aq.data(),codes(q[0]),inputs,ordinal,stream) &&
            backend->deviceToHost(bq.data(),codes(q[1]),inputs,ordinal,stream) &&
            backend->deviceToHost(as.data(),fp(scales[0]),as.size()*4,ordinal,stream) &&
            backend->deviceToHost(bs.data(),fp(scales[1]),bs.size()*4,ordinal,stream) &&
            backend->synchronizeStream(stream,ordinal), "router timed-output observation");
        ASSERT_EQ(std::memcmp(a.data(),b.data(),outputs*4),0);
        ASSERT_EQ(aq,bq);
        ASSERT_EQ(std::memcmp(as.data(),bs.data(),as.size()*4),0);
        for (int count : {0,1,c.rows}) { verify(count); if (::testing::Test::HasFatalFailure()) return; }
        std::cout << std::setprecision(9);
        for (int sample = 0; sample < samples; ++sample)
            std::cout << "ROCM_ROUTER_SAMPLE," << c.rows << ',' << c.width << ',' << c.experts << ','
                << int(c.basis) << ',' << sample << ',' << timings[0][sample] << ',' << timings[1][sample] << '\n';
        for (auto &values : timings) std::sort(values.begin(),values.end());
        std::cout << "ROCM_ROUTER_MEDIAN," << c.rows << ',' << c.width << ',' << c.experts << ',' << int(c.basis)
            << ',' << timings[0][samples/2] << ',' << timings[1][samples/2] << ','
            << timings[0][samples/2]/timings[1][samples/2] << '\n';
    });
}

/** @test Publish paired timing only after captured bytes and resource identity agree. */
TEST_P(ROCmRouterQ8Prefill, DISABLED_CapturedProductionTournament) { benchmarkRouter(GetParam(),false); }
/** @test Select one native production replay while setup and numerical oracles stay paused. */
TEST_P(ROCmRouterQ8Prefill, DISABLED_IsolatedNativeProfile) { benchmarkRouter(GetParam(),true); }

/** @return Canonical row buckets, authenticated model routers and explicit boundary controls. */
std::vector<RouterCase> routerCases()
{
    std::set<std::pair<int,int>> shapes;
    for (const auto &shape : kRouterModelGeometries)
        shapes.insert(shape);
    // QFN's parser authenticates compute width (not the wider HC trunk) and
    // expert count. The installed Qwen3.8 first-shard metadata authenticates
    // K8192/E512. Neither is inferred from a release catalog lacking E.
    shapes.emplace(int(qwen4exp::published::ComputeWidth),512);
    shapes.emplace(8192,512);
    // Independent boundary/stress controls; these names do not claim a model.
    for (auto shape : {std::pair{96,17},std::pair{3072,128},std::pair{4096,256},std::pair{8192,64}})
        shapes.insert(shape);
    std::vector<int> rows{16,32};
    rows.insert(rows.end(),kSupportedPrefillGraphBucketSizes.begin(),kSupportedPrefillGraphBucketSizes.end());
    std::vector<RouterCase> result;
    for (auto basis : {RouterQ8Basis::Original,RouterQ8Basis::SignedHadamard32})
        for (int count : rows)
            for (auto [width,experts] : shapes)
                result.push_back({count,width,experts,basis});
    return result;
}
INSTANTIATE_TEST_SUITE_P(NativeGfx1100,ROCmRouterQ8Prefill,::testing::ValuesIn(routerCases()),
    [](const ::testing::TestParamInfo<RouterCase> &info)
    {
        const auto &c = info.param;
        return "M"+std::to_string(c.rows)+"K"+std::to_string(c.width)+"E"+std::to_string(c.experts)+
            (c.basis==RouterQ8Basis::Original ? "Original" : "Hadamard");
    });
#endif
}
