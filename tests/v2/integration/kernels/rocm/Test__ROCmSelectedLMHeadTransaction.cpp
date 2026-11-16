/**
 * @file Test__ROCmSelectedLMHeadTransaction.cpp
 * @brief Real device row selection followed by the quantized verifier LM head.
 *
 * Seeded Q6_K weights use original full-vocabulary model geometries. Two forced
 * controls, actual Automatic and an independent serial owner have separate prepared payloads,
 * activations, arenas and outputs. Fixture-owned device indices are inputs to
 * the real selector; this component does not certify the MTP metadata producer
 * or grant Automatic selection a new materialization authority. Every compact
 * row is written, including negative indices clamped to the first source row.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include "backends/GPUDeviceContextPool.h"
#include "backends/HardwareInventory.h"
#include "backends/rocm/HIPGraphCapture.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "execution/compute_stages/stages/HiddenStateRowsSelectStage.h"
#include "execution/compute_stages/stages/LMHeadStage.h"
#include "execution/local_execution/coherence/GpuCoherence.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/mtp/MTPSpecDecodeMetadata.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedVerifierLaunch.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedArchitecture.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "transfer/TransferEngine.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <set>
#include <stdexcept>
#include <vector>
#include <omp.h>

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;
namespace {
constexpr int vocabulary = 248320;
constexpr int source_rows = 19;
/** @brief Every currently supported single-request verifier width, without interpolation. */
constexpr auto widths = [] {
    std::array<int,15> values{};
    for (int m = 2; m <= 16; ++m) values[m-2] = m;
    return values;
}();
constexpr std::array candidates{NativeQuantisedVerifierCandidate::Dot4,
                                NativeQuantisedVerifierCandidate::Automatic,
                                NativeQuantisedVerifierCandidate::WmmaM16N64};
constexpr std::array names{"dot4", "automatic", "wmma"};
constexpr int paths = int(candidates.size());

/** @brief Collect replay, unprofiled events, or one separately profiled Automatic operation. */
enum class Observation { Replay, Paired, Profile };

/** @brief Pause outside the single retained graph so setup cannot pollute counters. */
class ProfileWindow final {
    bool active_=false;
public:
    /** @brief Resume only after all setup and poison work has drained. */
    ProfileWindow() {
        if (roctxRangePushA("ROCmSelectedLMHeadTransaction")<0) throw std::runtime_error("Head profiler range failed");
        if (roctxProfilerResume(0)!=0) {(void)roctxRangePop();throw std::runtime_error("Head profiler resume failed");}
        active_=true;
    }
    /** @brief Pause before output observation, including during failed replay cleanup. */
    void finish() {
        if (!active_) return;
        const int paused=roctxProfilerPause(0),popped=roctxRangePop();active_=false;
        if (paused!=0 || popped<0) throw std::runtime_error("Head profiler pause failed");
    }
    /** @brief Prevent failed replay from profiling unrelated teardown kernels. */
    ~ProfileWindow() {
        if (active_) {
            const int paused=roctxProfilerPause(0),popped=roctxRangePop();
            if (paused!=0 || popped<0) std::terminate();
        }
    }
    ProfileWindow(const ProfileWindow&)=delete;
    ProfileWindow &operator=(const ProfileWindow&)=delete;
};

/** @brief Stop the component before submitting more work after a broken invariant. */
void require(bool ok, const char *message)
{ if (!ok) throw std::runtime_error(message); }

/** @brief Retire exact-stream work before any borrowed host or device storage. */
struct Drain {
    hipStream_t stream;
    /** @brief Assertion unwinding must preserve every asynchronous owner. */
    ~Drain() { if (hipStreamSynchronize(stream) != hipSuccess) std::terminate(); }
};

/** @brief Bind real stages to one persistent arena; serial decode has no selector. */
class BoundStages final {
public:
    std::unique_ptr<DeviceWorkspaceManager> workspace;
    std::unique_ptr<HiddenStateRowsSelectStage> selector;
    std::unique_ptr<LMHeadStage> head;
    WorkspaceRequirements requirements;
    hipStream_t stream;

    /** @param p Original head policy. @param select Optional original row-copy policy.
     * @param context Native device authority. @param exact Nondefault execution stream. */
    BoundStages(LMHeadStage::Params p,
                std::optional<HiddenStateRowsSelectStage::Params> select,
                IDeviceContext &context, hipStream_t exact) : stream(exact)
    {
        const auto device = p.device_id;
        const int m = p.seq_len, n = p.vocab_size, k = p.d_model;
        head = std::make_unique<LMHeadStage>(std::move(p));
        requirements = head->getWorkspaceRequirements(m, n, k);
        if (select) {
            selector = std::make_unique<HiddenStateRowsSelectStage>(std::move(*select));
            requirements.merge(selector->getWorkspaceRequirements(source_rows, k, 0));
            requirements.buffers.emplace_back(
                MTPSpecDecodeWorkspaceBuffers::VERIFIER_LOGIT_ROWS,
                size_t(m) * sizeof(int32_t), alignof(int32_t), true,
                WorkspaceExecutionRegime::Any, WorkspaceContentLifetime::SerialGraphFamily);
        }
        try {
            workspace = std::make_unique<DeviceWorkspaceManager>(device, requirements.total_bytes_with_alignment());
            require(workspace->allocate(requirements), "Selected-head arena allocation failed");
            head->bindWorkspace(workspace.get()); head->setGPUStream(stream);
            if (selector) {
                selector->bindWorkspace(workspace.get()); selector->setGPUStream(stream);
                require(selector->prepareGraphLaunch(&context, stream), "Selector preparation failed");
            }
            require(head->prepareGraphLaunch(&context, stream), "Selected-head preparation failed");
        } catch (...) {
            if (hipStreamSynchronize(stream) != hipSuccess) std::terminate();
            if (selector) selector->unbindWorkspace();
            head->unbindWorkspace(); throw;
        }
    }
    /** @brief Captured owners retire first; then release both workspace borrows. */
    ~BoundStages() {
        if (hipStreamSynchronize(stream) != hipSuccess) std::terminate();
        if (selector) selector->unbindWorkspace();
        head->unbindWorkspace();
    }
    BoundStages(const BoundStages &) = delete;
    BoundStages &operator=(const BoundStages &) = delete;
};

/** @brief Keep a producer omission scoped to its original executable and stream. */
class Omission final {
    hipGraphExec_t graph_;
    hipStream_t stream_;
    const std::vector<hipGraphNode_t> &nodes_;
    bool active_ = false;
public:
    /** @param graph Original executable. @param stream Exact stream. @param nodes Captured family. */
    Omission(hipGraphExec_t graph, hipStream_t stream, const std::vector<hipGraphNode_t> &nodes)
        : graph_(graph), stream_(stream), nodes_(nodes) {}
    /** @brief Omit every repeated producer so another repetition cannot hide the defect. */
    void omit() {
        require(!active_ && !nodes_.empty(), "Invalid selected-head omission"); active_ = true;
        for (auto node : nodes_) require(hipGraphNodeSetEnabled(graph_, node, 0) == hipSuccess, "Head omission failed");
    }
    /** @brief Restore only after all consumers have retired. */
    void restore() noexcept {
        if (!active_) return;
        if (hipStreamSynchronize(stream_) != hipSuccess) std::terminate();
        for (auto node : nodes_) if (hipGraphNodeSetEnabled(graph_, node, 1) != hipSuccess) std::terminate();
        active_ = false;
    }
    /** @brief Failed assertions cannot leak a disabled producer. */
    ~Omission() { restore(); }
    Omission(const Omission &) = delete;
    Omission &operator=(const Omission &) = delete;
};

/** @brief Real model geometry with seeded weights, not original model-weight evidence. */
struct Geometry { const char *name; int hidden; };
constexpr std::array geometries{Geometry{"Dense", 5120}, Geometry{"MoE", 2048}};

/** @brief Generate the large original source once using independent fixed row chunks.
 * @param format Exact Q6_K source factory. @param n Original vocabulary. @param k Hidden width.
 * @return One immutable original tensor; chunk seeds do not depend on scheduling.
 * @note Chunking changes no production representation. It bounds temporary factory
 * storage per worker and avoids serial synthesis of a billion-element source. */
std::shared_ptr<TensorBase> makeSource(const QuantizedVerifierFormatCase &format, int n, int k)
{
    constexpr int chunk_rows = 1024;
    const int chunks = (n+chunk_rows-1)/chunk_rows;
    const auto *hardware = DeviceManager::instance().hardware();
    require(hardware && hardware->total_physical_cores()>0 && omp_get_max_threads()>0,
        "Head source synthesis requires physical-worker inventory");
    const int workers = std::min({chunks,hardware->total_physical_cores(),omp_get_max_threads()});
    const size_t row_bytes = size_t(k/256)*sizeof(Q6_KBlock);
    AlignedVector<uint8_t> bytes(size_t(n)*row_bytes);
    std::vector<std::exception_ptr> errors(chunks);
    const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic,1) num_threads(workers)
    for (int chunk = 0; chunk < chunks; ++chunk) {
        try {
            const int first = chunk*chunk_rows, count = std::min(chunk_rows,n-first);
            const auto source = format.create({size_t(count),size_t(k)},96911u+uint32_t(chunk)*997u);
            require(quantizedRawBytesForGpuPreparedTest(*source)==size_t(count)*row_bytes,"Head source chunk extent mismatch");
            std::memcpy(bytes.data()+size_t(first)*row_bytes,source->raw_data(),size_t(count)*row_bytes);
        } catch (...) { errors[chunk] = std::current_exception(); }
    }
    for (const auto &error : errors) if (error) std::rethrow_exception(error);
    std::printf("ROCM_SELECTED_HEAD_SETUP,%d,%d,chunks=%d,workers=%d,seconds=%.6f\n",n,k,chunks,workers,
        std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
    return std::make_shared<Q6_KTensor>(std::vector<size_t>{size_t(n),size_t(k)},std::move(bytes));
}

/** @brief Replay or time fully materialized selected rows against independent M1.
 * @param geometry Original full-vocabulary width.
 * @param observation Replay, balanced triples, or one separate profiler operation.
 * @param profile_m Exact profiled width, unused for replay/timing sweeps. */
void prove(const Geometry &geometry, Observation observation, int profile_m=0)
{
    const bool timed=observation==Observation::Paired;
    const bool profiled=observation==Observation::Profile;
    if (profiled) ASSERT_EQ(::testing::UnitTest::GetInstance()->test_to_run_count(),1)
        << "A profiler launch owns exactly one geometry and width";
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) GTEST_SKIP() << "No ROCm device";
    initializeROCmHardwareMetadata();
    int ordinal = -1;
    for (int d = 0; d < devices; ++d)
        if (nativeQuantisedHardwareForDevice(d).architecture == native_quantised::NativeQuantisedArchitecture::rocm(11,0,0)) { ordinal = d; break; }
    if (ordinal < 0) GTEST_SKIP() << "No native gfx1100 participant";
    const auto &hardware = rocmHardwareMetadata().at(ordinal);
    std::printf("ROCM_SELECTED_HEAD_DEVICE,%d,%s,%s,",ordinal,hardware.isa_target.c_str(),hardware.pci_address.c_str());
    for (auto byte : hardware.uuid) std::printf("%02x",unsigned(byte));
    std::printf("\n");
    const auto device = DeviceId::rocm(ordinal);
    // Raw HIP discovery does not publish the canonical physical CPU budget.
    if (!DeviceManager::instance().hardware()) DeviceManager::instance().initialize(-1,false);
    auto worker = GPUDeviceContextPool::instance().retainAMDContext(ordinal);
    worker->submitAndWait([&] {
        const auto stream = static_cast<hipStream_t>(worker->defaultStream());
        auto context = IDeviceContext::create(device); ASSERT_TRUE(context);
        const int k = geometry.hidden, n = vocabulary, repeats = timed ? 8 : 1;
        const auto &formats = quantizedMoEVerifierFormats();
        const auto format = std::find_if(formats.begin(), formats.end(), [](const auto &f) { return f.tensor_type == TensorType::Q6_K; });
        ASSERT_NE(format, formats.end());
        std::shared_ptr<TensorBase> weight;
        std::vector<GpuPreparedGemm> prepared;
        std::array<DeviceNativeQuantisedMatrixDesc,paths+1> descriptors;
        std::unique_ptr<FP32Tensor> serial_input, serial_output;
        auto enclosing_count = TestTensorFactory::createINT32({1});
        std::array<std::vector<float>,2> inputs;
        std::array<std::vector<uint32_t>,2> oracle;
        const Drain setup_drain{stream};
        weight = makeSource(*format,n,k);
        prepared = makeGpuPreparedGemmBatch(std::vector<TensorBase *>(paths+1,weight.get()), device,
            "test.selected_lm_head", ModelContextId{96911});
        enclosing_count->mutable_int32_data()[0] = 0;
        ASSERT_TRUE(enclosing_count->ensureOnDevice(device,stream));
        for (int owner = 0; owner <= paths; ++owner) {
            ASSERT_EQ(prepared[owner].store->binding(prepared[owner].ref)->tensor, weight.get());
            ASSERT_TRUE(prepared[owner].kernel->exportNativeQuantisedMatrixDesc(descriptors[owner]));
            const auto &desc = descriptors[owner];
            ASSERT_TRUE(desc.source_identity_present); ASSERT_EQ(desc.source_codebook_id, 8);
            ASSERT_TRUE(desc.source_is_superblock); ASSERT_EQ(desc.codebook_id, 8);
            ASSERT_EQ(desc.n, n); ASSERT_EQ(desc.k, k); ASSERT_EQ(desc.blocks_per_row, k/32u);
            for (int prior = 0; prior < owner; ++prior) {
                ASSERT_NE(desc.payload, descriptors[prior].payload);
                ASSERT_NE(prepared[owner].kernel, prepared[prior].kernel);
            }
            std::printf("ROCM_SELECTED_HEAD_SOURCE,%s,%d,%d,%d,8,1,8\n", geometry.name, owner, n, k);
        }
        const auto headParams = [&](int owner, int m, FP32Tensor *input, FP32Tensor *output) {
            LMHeadStage::Params p;
            p.device_id = device; p.hidden_states = input; p.lm_head_weight = weight.get(); p.logits = output;
            p.seq_len = m; p.d_model = k; p.vocab_size = n;
            p.compute_all_positions = true; p.use_prefill_replay_row_offset = false;
            p.force_decode_equivalent_verifier_prefill = m > 1;
            p.serial_equivalent_partition_width = 0;
            p.prepared_store = prepared[owner].store.get(); p.prepared_ref = prepared[owner].ref;
            return p;
        };
        serial_input = TestTensorFactory::createFP32({1,size_t(k)});
        serial_output = TestTensorFactory::createFP32({1,size_t(n)});
        ASSERT_TRUE(serial_input->allocateOnDevice(device, stream));
        ASSERT_TRUE(serial_output->allocateOnDevice(device, stream));
        {
            BoundStages serial(headParams(paths, 1, serial_input.get(), serial_output.get()), std::nullopt, *context, stream);
            const Drain serial_drain{stream};
            for (int revision = 0; revision < 2; ++revision) {
                inputs[revision].resize(size_t(source_rows)*k);
                oracle[revision].resize(size_t(source_rows)*n);
                for (size_t i = 0; i < inputs[revision].size(); ++i)
                    inputs[revision][i] = float(int((i*29+i/k*11+revision*43)%251)-125)/128.f;
                for (int row = 0; row < source_rows; ++row) {
                    ASSERT_EQ(hipMemcpyAsync(serial_input->gpu_data_ptr(), inputs[revision].data()+size_t(row)*k,
                        size_t(k)*sizeof(float), hipMemcpyHostToDevice, stream), hipSuccess);
                    TransferEngine::publishCurrentDeviceWrite(serial_input.get(), stream);
                    ASSERT_TRUE(with_gpu_coherence(device, {serial_input.get()}, {serial_output.get()}, stream,
                        [&] { return serial.head->execute(context.get()); }));
                    ASSERT_EQ(hipMemcpyAsync(oracle[revision].data()+size_t(row)*n, serial_output->gpu_data_ptr(),
                        size_t(n)*sizeof(uint32_t), hipMemcpyDeviceToHost, stream), hipSuccess);
                }
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
                ASSERT_TRUE(std::all_of(oracle[revision].begin(), oracle[revision].end(), [](auto word) { return std::isfinite(std::bit_cast<float>(word)); }));
                ASSERT_TRUE(std::any_of(oracle[revision].begin(), oracle[revision].end(), [](auto word) { return word != 0; }));
            }
        }
        // Each M owns a retained selector/head pair. Prepared payloads above are
        // reused after the previous captured operation and its arena retire.
        for (int m : widths) {
            if (profiled && m!=profile_m) continue;
            SCOPED_TRACE(::testing::Message() << geometry.name << " M=" << m);
            std::array<std::unique_ptr<FP32Tensor>,paths> original, compact, outputs;
            std::array<std::unique_ptr<BoundStages>,paths> stages;
            std::array<std::unique_ptr<IGPUGraphCapture>,paths> fragments, graphs;
            std::array<std::array<std::vector<hipGraphNode_t>,3>,paths> families;
            std::vector<int32_t> indices(m);
            std::vector<uint32_t> actual(size_t(m)*n), selected(size_t(m)*k);
            const auto rows = DeviceRowRange::fullyActive(m);
            const Drain drain{stream};
            for (int row = 0; row < m; ++row) indices[row] = (row*7+3)%source_rows;
            for (int path = 0; path < paths; ++path) {
                const bool expected_native = path == 2 || (path == 1 && m >= 5);
                original[path] = TestTensorFactory::createFP32({source_rows,size_t(k)});
                compact[path] = TestTensorFactory::createFP32({size_t(m),size_t(k)});
                outputs[path] = TestTensorFactory::createFP32({size_t(m),size_t(n)});
                for (auto *tensor : {original[path].get(),compact[path].get(),outputs[path].get()}) ASSERT_TRUE(tensor->allocateOnDevice(device, stream));
                HiddenStateRowsSelectStage::Params select;
                select.device_id = device; select.input = original[path].get(); select.output = compact[path].get();
                select.seq_len = source_rows; select.d_model = k; select.selected_row_count = m;
                select.device_row_index_source = HiddenStateRowsSelectStage::DeviceRowIndexSource::WorkspaceBoundDeviceIndices;
                select.workspace_buffer_name = MTPSpecDecodeWorkspaceBuffers::VERIFIER_LOGIT_ROWS;
                select.row_copy_policy = RowsSelectPolicy{.negative_rows=NegativeRowSelectionPolicy::ClampToFirst,
                    .source_stride=k,.destination_stride=k};
                stages[path] = std::make_unique<BoundStages>(headParams(path,m,compact[path].get(),outputs[path].get()), select, *context, stream);
                ASSERT_EQ(hipMemcpyAsync(original[path]->gpu_data_ptr(),inputs[0].data(),inputs[0].size()*sizeof(float),hipMemcpyHostToDevice,stream),hipSuccess);
                TransferEngine::publishCurrentDeviceWrite(original[path].get(),stream);
                auto &owner = *stages[path];
                auto *index_bank = owner.workspace->getBuffer(MTPSpecDecodeWorkspaceBuffers::VERIFIER_LOGIT_ROWS);
                ASSERT_EQ(hipMemcpyAsync(index_bank,indices.data(),size_t(m)*sizeof(int32_t),hipMemcpyHostToDevice,stream),hipSuccess);
                // Actual Automatic must discard the source request's empty,
                // differently sized prefix after selection materializes M rows.
                std::unique_ptr<ITensorGemm::VerifierKernelModeScope> enclosing;
                if (path == 1) enclosing = std::make_unique<ITensorGemm::VerifierKernelModeScope>(
                    DeviceRowRange::deviceCounted(source_rows,
                        static_cast<const int32_t *>(enclosing_count->gpu_data_ptr())));
                const auto assertEnclosingRestored = [&] {
                    if (path!=1) return;
                    const auto *restored=ITensorGemm::VerifierKernelModeScope::rowsFor(source_rows);
                    ASSERT_NE(restored,nullptr);
                    EXPECT_EQ(restored->captureIdentity(),DeviceRowRange::deviceCounted(source_rows,
                        static_cast<const int32_t *>(enclosing_count->gpu_data_ptr())).captureIdentity());
                };
                ScopedNativeQuantisedVerifierCandidate candidate(candidates[path],ordinal);
                ASSERT_TRUE(owner.selector->execute(context.get()));
                TransferEngine::publishCurrentDeviceWrite(compact[path].get(),stream);
                ASSERT_TRUE(with_gpu_coherence(device,{compact[path].get()},{outputs[path].get()},stream,
                    [&] { return owner.head->execute(context.get()); }));
                assertEnclosingRestored();
                ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
                TransferEngine::requireDeviceInput(original[path].get(),device,stream);
                std::vector<GraphCaptureDependencyLedger::StagePlan> plans;
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    const auto producer = plans.size();
                    plans.push_back({.stage_identity=owner.selector.get(),.stage_name="selected_head_rows",
                        .external_inputs={original[path].get()},.outputs={compact[path].get()}});
                    plans.push_back({.stage_identity=owner.head.get(),.stage_name="selected_lm_head",
                        .internal_inputs={{compact[path].get(),producer}},.outputs={outputs[path].get()}});
                }
                GraphCaptureDependencyLedger ledger(device,stream,std::move(plans),"Selected LM-head component");
                graphs[path] = worker->createGraphCapture(stream);
                fragments[path] = graphs[path]->createOrderedTimelineFragment(); ASSERT_TRUE(fragments[path]);
                ScopedBackendGraphCapture capture(*worker,*fragments[path],"selected LM head",&ledger);
                ASSERT_TRUE(capture.begin());
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    { ScopedGraphCaptureStage stage(owner.selector.get()); ASSERT_TRUE(owner.selector->execute(context.get())); stage.complete(); }
                    { ScopedGraphCaptureStage stage(owner.head.get()); ASSERT_TRUE(owner.head->execute(context.get())); stage.complete(); }
                    assertEnclosingRestored();
                }
                capture.finish();
                const std::array<GPUOrderedTimelineStep,1> steps{{{.name=names[path],.kind=GPUOrderedTimelineStepKind::CapturedFragment,.capture=fragments[path].get()}}};
                ASSERT_TRUE(graphs[path]->buildOrderedTimelineTransaction(steps,profiled?GPUOrderedTimelineInstrumentation::Disabled:GPUOrderedTimelineInstrumentation::PerStepEvents));
                ASSERT_TRUE(graphs[path]->instantiate());
                auto *parent = dynamic_cast<HIPGraphCapture *>(graphs[path].get()); ASSERT_TRUE(parent);
                size_t count = 0; ASSERT_EQ(hipGraphGetNodes(parent->graph(),nullptr,&count),hipSuccess);
                std::vector<hipGraphNode_t> nodes(count);
                ASSERT_EQ(hipGraphGetNodes(parent->graph(),nodes.data(),&count),hipSuccess);
                const auto quant = owner.workspace->getBuffer(GemmWorkspaceBuffers::QUANT_A);
                const auto scales = owner.workspace->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE);
                const auto sums = owner.workspace->getBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE);
                using Partial = std::pair<uintptr_t,int>;
                std::vector<Partial> produced, reduced;
                int native_producers = 0;
                for (auto node : nodes) {
                    hipGraphNodeType type; ASSERT_EQ(hipGraphNodeGetType(node,&type),hipSuccess);
                    if (type != hipGraphNodeTypeKernel) continue;
                    hipKernelNodeParams launch{}; ASSERT_EQ(hipGraphKernelNodeGetParams(node,&launch),hipSuccess);
                    const char *symbol = hipKernelNameRefByPtr(launch.func,stream); ASSERT_TRUE(symbol);
                    const std::string_view name(symbol);
                    if (name.find("rowsSelectFP32Kernel") != std::string_view::npos) {
                        families[path][0].push_back(node);
                        ASSERT_EQ(*static_cast<const float **>(launch.kernelParams[0]),original[path]->gpu_data_ptr());
                        ASSERT_EQ(*static_cast<float **>(launch.kernelParams[1]),compact[path]->gpu_data_ptr());
                        ASSERT_EQ(*static_cast<const int **>(launch.kernelParams[2]),index_bank);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[3]),source_rows);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[4]),k);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[5]),m);
                        const auto policy = *static_cast<const RowsSelectPolicy *>(launch.kernelParams[6]);
                        ASSERT_EQ(policy.negative_rows,NegativeRowSelectionPolicy::ClampToFirst);
                        ASSERT_EQ(policy.source_stride,k); ASSERT_EQ(policy.destination_stride,k);
                        ASSERT_EQ(launch.blockDim.x,256u);
                        ASSERT_EQ(launch.gridDim.x,unsigned(std::min(1024,(m*k+255)/256)));
                    }
                    if (name.find("quantizeActivationsQ8_blockwise_v4_with_sums_kernel") != std::string_view::npos) {
                        families[path][1].push_back(node);
                        ASSERT_NE(name.find("ILb0E"),std::string_view::npos);
                        ASSERT_EQ(*static_cast<const float **>(launch.kernelParams[0]),compact[path]->gpu_data_ptr());
                        ASSERT_EQ(*static_cast<int8_t **>(launch.kernelParams[1]),quant);
                        ASSERT_EQ(*static_cast<float **>(launch.kernelParams[2]),scales);
                        ASSERT_EQ(*static_cast<int32_t **>(launch.kernelParams[3]),sums);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[4]),m);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[5]),k);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[6]),k/32);
                        ASSERT_EQ(static_cast<const DeviceRowRange *>(launch.kernelParams[7])->captureIdentity(),rows.captureIdentity());
                        ASSERT_EQ(launch.gridDim.x,unsigned(m*(k/32))); ASSERT_EQ(launch.blockDim.x,32u);
                    }
                    if (name.find("gemv_native_quantised_reduce_small_m_kernel_t") != std::string_view::npos) {
                        ASSERT_EQ(*static_cast<float **>(launch.kernelParams[1]),outputs[path]->gpu_data_ptr());
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[2]),m);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[3]),n);
                        ASSERT_EQ(*static_cast<int *>(launch.kernelParams[4]),16);
                        ASSERT_EQ(static_cast<const DeviceRowRange *>(launch.kernelParams[5])->captureIdentity(),rows.captureIdentity());
                        ASSERT_EQ(static_cast<const ProjectionRowWork *>(launch.kernelParams[6])->captureIdentity(),ProjectionRowWork::fromPrefix(rows).captureIdentity());
                        reduced.emplace_back(reinterpret_cast<uintptr_t>(*static_cast<const float **>(launch.kernelParams[0])),16);
                    }
                    if (name.find("gemv_native_quantised_small_m_batched_shared_kernel_t") != std::string_view::npos ||
                        name.find("gemv_native_quantised_small_m_kernel_t") != std::string_view::npos) {
                        ASSERT_FALSE(expected_native);
                        families[path][2].push_back(node);
                    }
                    if (name.find("native_quantised_verifier_wmma_m16n64") == std::string_view::npos) continue;
                    ++native_producers;
                    families[path][2].push_back(node);
                    const auto &packet = *static_cast<const NativeQuantisedSmallMBatchedArgs *>(launch.kernelParams[3]);
                    const auto &desc = descriptors[path];
                    ASSERT_EQ(packet.num_projections,1); ASSERT_EQ(packet.N[0],n);
                    ASSERT_EQ(packet.payload[0],desc.payload); ASSERT_EQ(packet.scales[0],desc.scales);
                    ASSERT_EQ(packet.mins[0],desc.mins); ASSERT_EQ(packet.emins[0],desc.emins);
                    ASSERT_EQ(packet.codebook[0],desc.codebook_id); ASSERT_EQ(packet.bias[0],nullptr);
                    ASSERT_EQ(packet.output[0],outputs[path]->gpu_data_ptr());
                    ASSERT_EQ(packet.rows.captureIdentity(),rows.captureIdentity());
                    ASSERT_EQ(packet.work.captureIdentity(),ProjectionRowWork::fromPrefix(rows).captureIdentity());
                    ASSERT_EQ(*static_cast<const int8_t **>(launch.kernelParams[0]),quant);
                    ASSERT_EQ(*static_cast<const float **>(launch.kernelParams[1]),scales);
                    ASSERT_EQ(*static_cast<const int32_t **>(launch.kernelParams[2]),sums);
                    ASSERT_EQ(*static_cast<int *>(launch.kernelParams[4]),k);
                    ASSERT_EQ(*static_cast<int *>(launch.kernelParams[5]),k/512);
                    ASSERT_EQ(launch.gridDim.x,3880u); ASSERT_EQ(packet.total_grid_n,3880);
                    ASSERT_EQ(launch.gridDim.y,16u); ASSERT_EQ(launch.gridDim.z,1u);
                    ASSERT_EQ(launch.blockDim.x,128u);
                    ASSERT_NE(name.find("VerifierWmmaPublicationE2E"),std::string_view::npos);
                    const auto first = reinterpret_cast<uintptr_t>(packet.partial[0]);
                    const auto end = first+size_t(16)*m*n*sizeof(float); ASSERT_GT(end,first);
                    int admitted = 0;
                    for (const char *bank : {GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL,GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED}) {
                        const auto base = reinterpret_cast<uintptr_t>(owner.workspace->getBuffer(bank));
                        const auto bytes = owner.workspace->getBufferSize(bank);
                        admitted += base && first >= base && end <= base+bytes;
                    }
                    ASSERT_EQ(admitted,1); produced.emplace_back(first,16);
                }
                ASSERT_EQ(families[path][0].size(),size_t(repeats));
                ASSERT_EQ(families[path][1].size(),size_t(repeats));
                ASSERT_EQ(families[path][2].size(),size_t(repeats));
                ASSERT_EQ(native_producers,expected_native ? repeats : 0);
                if (expected_native) {
                    std::sort(produced.begin(),produced.end()); std::sort(reduced.begin(),reduced.end());
                    ASSERT_EQ(produced,reduced);
                }
                std::vector<GPUGraphKernelNodeInfo> inventory;
                ASSERT_TRUE(graphs[path]->inspectKernelNodes(inventory)); std::set<std::string> printed;
                for (const auto &node : inventory) {
                    ASSERT_TRUE(node.valid()); ASSERT_EQ(node.local_memory_bytes_per_thread,0u);
                    if (printed.insert(node.name).second)
                        std::printf("ROCM_SELECTED_HEAD_RESOURCE,%s,%d,%s,%u,%zu,%s\n",geometry.name,m,names[path],node.registers_per_thread,node.static_shared_memory_bytes,node.name.c_str());
                }
                std::printf("ROCM_SELECTED_HEAD_CAPTURE,%s,%d,%s,%zu,%zu,%zu,16,%d\n",geometry.name,m,names[path],
                    families[path][0].size(),families[path][1].size(),families[path][2].size(),k/512);
                std::printf("ROCM_SELECTED_HEAD_ENGINE,%s,%d,%s,%d\n",geometry.name,m,names[path],native_producers);
            }
            int revision = 0;
            const auto publish = [&] {
                for (int path = 0; path < paths; ++path) {
                    require(hipMemcpyAsync(original[path]->gpu_data_ptr(),inputs[revision].data(),inputs[revision].size()*sizeof(float),hipMemcpyHostToDevice,stream)==hipSuccess,"Head source publication failed");
                    TransferEngine::publishCurrentDeviceWrite(original[path].get(),stream);
                    require(hipMemcpyAsync(stages[path]->workspace->getBuffer(MTPSpecDecodeWorkspaceBuffers::VERIFIER_LOGIT_ROWS),indices.data(),size_t(m)*sizeof(int32_t),hipMemcpyHostToDevice,stream)==hipSuccess,"Head index publication failed");
                }
            };
            const auto poison = [&](int path) {
                require(hipMemsetAsync(compact[path]->gpu_data_ptr(),0xa5,size_t(m)*k*sizeof(float),stream)==hipSuccess,"Compact poison failed");
                require(hipMemsetAsync(outputs[path]->gpu_data_ptr(),0xa5,size_t(m)*n*sizeof(float),stream)==hipSuccess,"Head output poison failed");
                for (const auto &buffer : stages[path]->requirements.buffers)
                    if (buffer.content_lifetime == WorkspaceContentLifetime::ParticipantExecution) {
                        const auto bytes = stages[path]->workspace->getBufferSize(buffer.name);
                        if (bytes) require(hipMemsetAsync(stages[path]->workspace->getBuffer(buffer.name),0xff,bytes,stream)==hipSuccess,"Head scratch poison failed");
                    }
            };
            const auto observe = [&](int path) {
                require(hipMemcpyAsync(actual.data(),outputs[path]->gpu_data_ptr(),actual.size()*sizeof(uint32_t),hipMemcpyDeviceToHost,stream)==hipSuccess,"Head observation failed");
                require(hipMemcpyAsync(selected.data(),compact[path]->gpu_data_ptr(),selected.size()*sizeof(uint32_t),hipMemcpyDeviceToHost,stream)==hipSuccess,"Compact observation failed");
                require(hipStreamSynchronize(stream)==hipSuccess,"Head observation join failed");
            };
            const auto replay = [&](int path) {
                poison(path); require(graphs[path]->launch(),"Selected-head replay failed");observe(path);
            };
            const auto matches = [&] {
                for (int row = 0; row < m; ++row) {
                    const int source = std::clamp(indices[row],0,source_rows-1);
                    for (int col = 0; col < n; ++col)
                        if (actual[size_t(row)*n+col] != oracle[revision][size_t(source)*n+col]) return false;
                }
                return true;
            };
            const auto selectedMatches = [&] {
                for (int row = 0; row < m; ++row) {
                    const int source = std::clamp(indices[row],0,source_rows-1);
                    for (int col = 0; col < k; ++col)
                        if (selected[size_t(row)*k+col] != std::bit_cast<uint32_t>(inputs[revision][size_t(source)*k+col])) return false;
                }
                return true;
            };
            if (profiled) {
                for (int warm=0;warm<10;++warm) ASSERT_TRUE(graphs[1]->launch());
                ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
                poison(1);ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
                {
                    ProfileWindow window;
                    const bool launched=graphs[1]->launch();
                    const bool joined=hipStreamSynchronize(stream)==hipSuccess;
                    window.finish();ASSERT_TRUE(launched && joined);
                }
                observe(1);ASSERT_TRUE(selectedMatches());ASSERT_TRUE(matches());
                std::printf("ROCM_SELECTED_HEAD_PROFILE_OBSERVED,%s,%d,automatic\n",geometry.name,m);
                continue;
            }
            // Changed inputs, permutation, duplicate rows, both clamp boundaries,
            // reset, and repeated dirty replay all reuse the original graphs.
            for (int phase : {0,1,2,0,0}) {
                revision = phase == 1 ? 1 : 0;
                for (int row = 0; row < m; ++row)
                    indices[row] = phase == 0 ? (row*7+3)%source_rows :
                        phase == 1 ? (m-row)%3 : row%2 ? source_rows+4 : -1;
                publish();
                for (int path = 0; path < paths; ++path) {
                    replay(path); ASSERT_TRUE(selectedMatches()); ASSERT_TRUE(matches());
                    if (phase == 2) {
                        stages[path]->selector->resetSessionStatePreservingCapturedReplay();
                        stages[path]->head->resetSessionStatePreservingCapturedReplay();
                        ASSERT_TRUE(stages[path]->selector->prepareGraphLaunch(context.get(),stream));
                        ASSERT_TRUE(stages[path]->head->prepareGraphLaunch(context.get(),stream));
                    }
                }
            }
            for (int path = 0; path < paths; ++path) {
                auto *parent = dynamic_cast<HIPGraphCapture *>(graphs[path].get()); ASSERT_TRUE(parent);
                for (const auto &family : families[path]) if (!family.empty()) {
                    Omission omission(parent->executable(),stream,family); omission.omit();
                    replay(path); ASSERT_FALSE(matches());
                    replay((path+1)%paths); ASSERT_TRUE(selectedMatches()); ASSERT_TRUE(matches());
                    omission.restore(); replay(path); ASSERT_TRUE(selectedMatches()); ASSERT_TRUE(matches());
                }
            }
            if (timed) {
                for (int warm = 0; warm < 8; ++warm) for (auto &graph : graphs) ASSERT_TRUE(graph->launch());
                for (int sample = 0; sample < 31; ++sample) {
                    std::array<double,paths> elapsed{};
                    for (int order = 0; order < paths; ++order) {
                        const int rotation = (sample/2)%paths;
                        const int path = sample%2 ? (rotation+paths-1-order)%paths : (rotation+order)%paths;
                        replay(path); ASSERT_TRUE(selectedMatches()); ASSERT_TRUE(matches());
                        const auto timing = graphs[path]->consumeOrderedTimelineTiming();
                        ASSERT_EQ(timing.state,GPUOrderedTimelineTimingState::Complete);
                        ASSERT_EQ(timing.samples.size(),1u); ASSERT_EQ(timing.samples[0].name,names[path]);
                        ASSERT_EQ(timing.samples[0].kind,GPUOrderedTimelineStepKind::CapturedFragment);
                        elapsed[path] = timing.samples[0].elapsed_ms*1000./repeats;
                        ASSERT_TRUE(std::isfinite(elapsed[path]) && elapsed[path]>0);
                    }
                    std::printf("ROCM_SELECTED_HEAD_PAIR,%s,%d,%d,%d,%d,%.9f,%.9f,%.9f\n",geometry.name,n,k,m,sample,elapsed[0],elapsed[1],elapsed[2]);
                }
            }
        }
    });
}
} // namespace

/** @brief Separate original geometries bound the preparation and replay lifecycle. */
class ROCmSelectedLMHeadTransaction : public ::testing::TestWithParam<Geometry> {};
TEST_P(ROCmSelectedLMHeadTransaction, CapturedReplay) { prove(GetParam(),Observation::Replay); }
TEST_P(ROCmSelectedLMHeadTransaction, DISABLED_Paired) { prove(GetParam(),Observation::Paired); }
TEST_P(ROCmSelectedLMHeadTransaction, DISABLED_ProfileM4) { prove(GetParam(),Observation::Profile,4); }
TEST_P(ROCmSelectedLMHeadTransaction, DISABLED_ProfileM16) { prove(GetParam(),Observation::Profile,16); }
INSTANTIATE_TEST_SUITE_P(OriginalGeometry,ROCmSelectedLMHeadTransaction,::testing::ValuesIn(geometries),
    [](const ::testing::TestParamInfo<Geometry> &info) { return info.param.name; });
