/**
 * @file Test__Qwen4ExpDevicePrimitives.cpp
 * @brief Captured native CUDA/HIP proof of GR/PLE arithmetic and history publication.
 *
 * TransferEngine owns every persistent allocation. One retained ten-kernel graph
 * evaluates all ten primitives and commits an explicit device-owned accepted prefix.
 * Serial GPU execution is a test oracle only. Replays change live/accepted counts
 * without replacing pointers or recapturing, checking inactive-byte guards,
 * every candidate rejection boundary, EOS context and chronological nine-row
 * history. Resource inspection rejects hidden copies, callbacks or local spills.
 * Real RAM/mmap service-worker affinity is observed by exact-inode libc delegation.
 */
#include <gtest/gtest.h>
#include "utils/ImmutableLookupWorkerAffinityObservation.h"
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "memory/BufferArena.h"
#include "kernels/qwen4exp/Qwen4ExpKernels.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include "execution/compute_stages/stages/GatedRMSNormStage.h"
#include "execution/compute_stages/stages/qwen4exp/ImmutableLookupStage.h"
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "kernels/common/SamplingMath.h"
#include "execution/mtp/DeviceGenerationGraphProgram.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "loaders/ModelContext.h"
#include "loaders/ImmutableRowTableUploadGeometry.h"
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "planning/ImmutableRowTablePlacementPlanner.h"
#include "tensors/TensorFactory.h"
#include "../../../mocks/MockMPIContext.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::qwen4exp;
#ifdef HAVE_CUDA
/** @brief Production native gate kernel used only as the serial-row diagnostic oracle. */
extern "C" bool cudaGDN_gated_rmsnorm(const float *,const float *,const float *,float *,int,int,int,int,
    float,bool,GatedNormActivation,int,void *);
#endif
#ifdef HAVE_ROCM
/** @brief Production native gate kernel used only as the serial-row diagnostic oracle. */
extern "C" bool rocmGDN_gated_rmsnorm(const float *,const float *,const float *,float *,int,int,int,int,
    float,bool,GatedNormActivation,int,void *);
#endif
namespace
{
/** @brief Abort worker execution after an infrastructure failure. */
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}
/** @brief Persist native captured pointers until the last terminal observation. */
struct Storage
{
    DeviceId device; ///< Exact assigned native backend endpoint.
    IBackend *backend; ///< Backend providing explicit-stream test I/O.
    void *stream; ///< Borrowed exact worker stream.
    std::vector<std::shared_ptr<DeviceTransferBuffer>> owners; ///< PMA-accounted persistent storage.
    std::unique_ptr<IGPUGraphCapture> graph; ///< Retained executable, destroyed before its pointers.

    /** @brief Join the terminal test observation before destroying captured storage. */
    ~Storage()
    {
        (void)backend->synchronizeStream(stream, device.ordinal);
        graph.reset();
    }
    /** @brief Allocate admitted setup storage; never called during capture or replay. */
    template <typename T> T *allocate(std::size_t elements)
    {
        auto owner = TransferEngine::instance().allocateDeviceTransferBuffer(elements * sizeof(T), device);
        require(owner && owner->isBound(), "Qwen4Exp test allocation failed");
        auto *pointer = static_cast<T *>(owner->mutableDeviceData());
        owners.push_back(std::move(owner));
        return pointer;
    }
    /** @brief Upload setup/control bytes outside retained graph execution. */
    template <typename T> void upload(T *pointer, const std::vector<T> &values)
    {
        require(backend->hostToDeviceOnStream(pointer, values.data(), values.size() * sizeof(T),
                    device.ordinal, stream), "Qwen4Exp test upload failed");
        require(backend->synchronizeStream(stream, device.ordinal), "Qwen4Exp setup observation failed");
    }
    /** @brief Download terminal oracle bytes on the exact stream. */
    template <typename T> std::vector<T> download(const T *pointer, std::size_t elements)
    {
        std::vector<T> result(elements);
        require(backend->deviceToHostOnStream(result.data(), pointer, elements * sizeof(T),
                    device.ordinal, stream), "Qwen4Exp test terminal download failed");
        require(backend->synchronizeStream(stream, device.ordinal), "Qwen4Exp terminal observation failed");
        return result;
    }
    /** @brief Allocate and initialize immutable test inputs before capture. */
    template <typename T> T *input(const std::vector<T> &values)
    {
        auto *pointer = allocate<T>(values.size());
        upload(pointer, values);
        return pointer;
    }
};
/** @brief Generate bounded finite values without invoking a production arithmetic oracle. */
std::vector<float> values(std::size_t size, int phase)
{
    std::vector<float> result(size);
    for (std::size_t index = 0; index < size; ++index)
        result[index] = static_cast<float>((static_cast<int>(index % 71) * 13 + phase) % 67 - 33) / 41.0f;
    return result;
}
/** @brief Bind exact fixture constants, including multipliers above float64's integer range. */
NgramHashPolicy policy()
{
    NgramHashPolicy result;
    result.multipliers = {9007199254740993ULL, 9007199254740995ULL, 9007199254740997ULL};
    result.moduli = {1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097};
    std::uint64_t offset = 0;
    for (int head = 0; head < 16; ++head)
    {
        result.offsets[head] = offset;
        offset += result.moduli[head];
    }
    return result;
}
/** @brief Compare terminal raw bytes, retaining inactive poison and signed-zero evidence. */
template <typename T> void expectEqual(Storage &storage, const T *actual, const T *expected, std::size_t size)
{
    const auto a = storage.download(actual, size);
    const auto b = storage.download(expected, size);
    EXPECT_EQ(std::memcmp(a.data(), b.data(), size * sizeof(T)), 0);
}
/** @brief Prove one real native backend with production-shaped and adversarial row geometries. */
void prove(DeviceId device)
{
    auto *backend = getBackendFor(device);
    if (!backend || backend->deviceCount() <= device.ordinal) GTEST_SKIP() << "Requested GPU backend unavailable";
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        std::unique_ptr<IQwen4ExpKernels> kernel;
#ifdef HAVE_CUDA
        if (device.is_cuda()) kernel = createCUDAQwen4ExpKernels();
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) kernel = createROCmQwen4ExpKernels();
#endif
        require(bool(kernel), "Qwen4Exp native implementation missing");
        for (const auto geometry : std::array{
                 std::array{1, 1}, std::array{7, 2}, std::array{19, 15}, std::array{257, 31},
                 std::array{2560, 1}, std::array{2560, 15}, std::array{2560, 512},
                 std::array{2560, 1024}, std::array{2560, 1025}})
        {
            const int hidden = geometry[0], rows = geometry[1], channels = hidden * 4;
            constexpr int lowRank = 320;
            SCOPED_TRACE(::testing::Message() << device.toString() << " H=" << hidden << " M=" << rows);
            Storage storage{device, backend, ExplicitGPUStream(worker.defaultStream()).get(), {}, {}};
            const auto trunkSize = static_cast<std::size_t>(rows) * channels;
            const auto narrowSize = static_cast<std::size_t>(rows) * hidden;
            auto *normalized = storage.input(values(trunkSize, 3));
            auto *key = storage.input(values(trunkSize, 7));
            auto *logits = storage.input(values(trunkSize, 11));
            auto *residual = storage.input(values(trunkSize, 13));
            auto *block = storage.input(values(narrowSize, 17));
            auto *inject = storage.input(values(rows * 4, 19));
            auto *downInput = storage.input(values(rows * lowRank, 23));
            auto *weights = storage.input(values(channels * 4, 29));
            auto *gain = storage.input(values(channels, 37));
            auto *normOutput = storage.allocate<float>(trunkSize);
            auto *serialNormOutput = storage.allocate<float>(trunkSize);
            const auto hostHistory = values(9 * channels, 31);
            auto *priorHistory = storage.input(hostHistory);
            auto *serialHistoryA = storage.input(hostHistory);
            auto *serialHistoryB = storage.allocate<float>(9 * channels);
            const std::vector<std::int32_t> hostPrior{248044, 37};
            auto *priorTokens = storage.input(hostPrior);
            auto *serialTokensA = storage.input(hostPrior);
            auto *serialTokensB = storage.allocate<std::int32_t>(2);
            std::vector<std::int32_t> hostTokens(rows);
            for (int row = 0; row < rows; ++row) hostTokens[row] = row % 11 == 3 ? 248044 : (row * 4093 + 17) % 248000;
            auto *tokens = storage.input(hostTokens);
            auto *live = storage.input(std::vector<int>{rows});
            auto *accepted = storage.input(std::vector<int>{rows});
            std::array<float *, 7> grouped{}, serial{};
            const std::array<std::size_t, 7> sizes{
                trunkSize, static_cast<std::size_t>(rows) * lowRank, narrowSize,
                trunkSize, trunkSize, trunkSize, static_cast<std::size_t>(9) * channels};
            for (std::size_t index = 0; index < sizes.size(); ++index)
            {
                grouped[index] = storage.allocate<float>(sizes[index]);
                serial[index] = storage.allocate<float>(sizes[index]);
            }
            auto *hashes = storage.allocate<std::int64_t>(rows * 16);
            auto *serialHashes = storage.allocate<std::int64_t>(rows * 16);
            auto *nextTokens = storage.allocate<std::int32_t>(2);
            const RowExtent extent{rows, live};
            const auto hashPolicy = policy();
            const auto record = [&] {
                kernel->branchRMSNorm({extent, hidden, 1e-6f, normalized, gain, normOutput}, storage.stream);
                kernel->residualExpand({extent, hidden, block, grouped[0]}, storage.stream);
                kernel->residualDown({extent, lowRank, downInput, grouped[1]}, storage.stream);
                kernel->residualRead({extent, hidden, normalized, logits, grouped[2]}, storage.stream);
                kernel->residualWrite({extent, hidden, residual, block, inject, grouped[3]}, storage.stream);
                kernel->ngramHash({extent, tokens, priorTokens, hashes, hashPolicy}, storage.stream);
                kernel->pleGate({extent, hidden, key, normalized, block, grouped[4]}, storage.stream);
                kernel->pleConv({extent, channels, normalized, priorHistory, weights, grouped[4], residual, grouped[5]}, storage.stream);
                kernel->commitConvHistory({{extent, accepted}, channels, priorHistory, normalized, grouped[6]}, storage.stream);
                kernel->commitTokenHistory({{extent, accepted}, priorTokens, tokens, nextTokens}, storage.stream);
            };
            EXPECT_THROW(kernel->residualExpand({extent, hidden, block, grouped[0]}, nullptr), std::invalid_argument);
            EXPECT_THROW(kernel->residualExpand({extent, hidden, block, grouped[0]}, reinterpret_cast<void *>(1)), std::invalid_argument);
            EXPECT_THROW(kernel->residualExpand({extent, hidden, block, grouped[0]}, reinterpret_cast<void *>(2)), std::invalid_argument);
            storage.graph = worker.createGraphCapture(storage.stream);
            require(bool(storage.graph), "Qwen4Exp native graph owner missing");
            {
                ScopedBackendGraphCapture capture(worker, *storage.graph, "Qwen4Exp GR/PLE primitives");
                require(capture.begin(), "Qwen4Exp native recording failed");
                record();
                capture.finish();
            }
            EXPECT_EQ(storage.graph->nodeCount(), 10u);
            std::vector<GPUGraphKernelNodeInfo> nodes;
            std::string error;
            require(storage.graph->inspectKernelNodes(nodes, &error), error.c_str());
            EXPECT_EQ(nodes.size(), 10u);
            for (const auto &node : nodes)
            {
                EXPECT_TRUE(node.valid());
                EXPECT_EQ(node.local_memory_bytes_per_thread, 0u);
                std::cout << "QWEN4EXP_RESOURCES," << device.toString() << ',' << hidden << ',' << rows << ','
                          << node.name << ',' << node.registers_per_thread << ','
                          << node.max_active_blocks_per_sm << ',' << node.grid_x << '\n';
            }
            require(storage.graph->instantiate(), "Qwen4Exp native instantiation failed");
            for (const int liveCount : {rows, std::min(rows, 15), 1, 0, rows, std::min(rows, 2)})
            {
                const int acceptedCount = liveCount == rows ? liveCount : liveCount / 2;
                storage.upload(live, std::vector<int>{liveCount});
                storage.upload(accepted, std::vector<int>{acceptedCount});
                for (std::size_t index = 0; index < sizes.size(); ++index)
                {
                    storage.upload(grouped[index], std::vector<float>(sizes[index], -12345.0f));
                    storage.upload(serial[index], std::vector<float>(sizes[index], -12345.0f));
                }
                storage.upload(normOutput, std::vector<float>(trunkSize, -12345.0f));
                storage.upload(serialNormOutput, std::vector<float>(trunkSize, -12345.0f));
                storage.upload(hashes, std::vector<std::int64_t>(rows * 16, -1));
                storage.upload(serialHashes, std::vector<std::int64_t>(rows * 16, -1));
                require(storage.graph->launch(), "Qwen4Exp retained replay failed");
                storage.upload(serialHistoryA, hostHistory);
                storage.upload(serialTokensA, hostPrior);
                float *historyInput = serialHistoryA, *historyOutput = serialHistoryB;
                auto *tokenInput = serialTokensA, *tokenOutput = serialTokensB;
                for (int row = 0; row < liveCount; ++row)
                {
                    const auto trunkBase = static_cast<std::size_t>(row) * channels;
                    const auto narrowBase = static_cast<std::size_t>(row) * hidden;
                    kernel->branchRMSNorm({{1}, hidden, 1e-6f, normalized + trunkBase, gain, serialNormOutput + trunkBase}, storage.stream);
                    kernel->residualExpand({{1}, hidden, block + narrowBase, serial[0] + trunkBase}, storage.stream);
                    kernel->residualDown({{1}, lowRank, downInput + row * lowRank, serial[1] + row * lowRank}, storage.stream);
                    kernel->residualRead({{1}, hidden, normalized + trunkBase, logits + trunkBase, serial[2] + narrowBase}, storage.stream);
                    kernel->residualWrite({{1}, hidden, residual + trunkBase, block + narrowBase, inject + row * 4,
                                           serial[3] + trunkBase}, storage.stream);
                    kernel->ngramHash({{1}, tokens + row, tokenInput, serialHashes + row * 16, hashPolicy}, storage.stream);
                    kernel->pleGate({{1}, hidden, key + trunkBase, normalized + trunkBase, block + narrowBase,
                                    serial[4] + trunkBase}, storage.stream);
                    kernel->pleConv({{1}, channels, normalized + trunkBase, historyInput, weights, serial[4] + trunkBase,
                                    residual + trunkBase, serial[5] + trunkBase}, storage.stream);
                    kernel->commitConvHistory({{{1}, nullptr, 1}, channels, historyInput, normalized + trunkBase, historyOutput}, storage.stream);
                    kernel->commitTokenHistory({{{1}, nullptr, 1}, tokenInput, tokens + row, tokenOutput}, storage.stream);
                    std::swap(historyInput, historyOutput);
                    std::swap(tokenInput, tokenOutput);
                }
                for (std::size_t index = 0; index < sizes.size() - 1; ++index)
                    expectEqual(storage, grouped[index], serial[index], sizes[index]);
                expectEqual(storage, normOutput, serialNormOutput, trunkSize);
                expectEqual(storage, hashes, serialHashes, static_cast<std::size_t>(rows) * 16);
                // Independent tail assembly authenticates rejection at the selected prefix.
                const auto hostNormalized = storage.download(normalized, trunkSize);
                const auto actualHistory = storage.download(grouped[6], 9 * channels);
                for (int slot = 0; slot < 9; ++slot)
                {
                    const int source = acceptedCount - 9 + slot;
                    const auto *expected = source < 0 ? hostHistory.data() + (source + 9) * channels
                                                     : hostNormalized.data() + source * channels;
                    EXPECT_EQ(std::memcmp(expected, actualHistory.data() + slot * channels, channels * sizeof(float)), 0);
                }
                const auto actualTokens = storage.download(nextTokens, 2);
                for (int index = 0; index < 2; ++index)
                {
                    const int source = acceptedCount - 2 + index;
                    EXPECT_EQ(actualTokens[index], source < 0 ? hostPrior[source + 2] : hostTokens[source]);
                }
            }
        }
    }).get();
}
/** @brief Prove backend-neutral stages use prepared tensors and exact control-word subviews. */
void proveStages(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if (!backend || backend->deviceCount()<=device.ordinal) GTEST_SKIP() << "Requested GPU backend unavailable";
    auto context=IDeviceContext::create(device,1);
    require(bool(context),"Qwen4Exp stage device context creation failed");
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        using Stage=Qwen4ExpPrimitiveStage;
        using Float=Qwen4ExpFloatBinding;
        using Int=Qwen4ExpIntBinding;
        using Weight=Qwen4ExpImmutableWeightBinding;
        constexpr int rows=15,hidden=257,channels=4*hidden;
        Storage storage{device,backend,ExplicitGPUStream(worker.defaultStream()).get(),{}, {}};
        FP32Tensor input({rows,channels},DeviceId::cpu()), gain({4,hidden},DeviceId::cpu()),
            logits({rows,channels},DeviceId::cpu()), inject({rows,4},DeviceId::cpu()),
            normalized({rows,channels},DeviceId::cpu()), mixed({rows,hidden},DeviceId::cpu()),
            written({rows,channels},DeviceId::cpu());
        INT32Tensor controls({4}),tokens({rows}),prior({2}),nextTokens({2});
        const auto original=values(rows*channels,7);
        std::copy(original.begin(),original.end(),input.mutable_data());
        std::fill_n(logits.mutable_data(),logits.numel(),0.0f);
        std::fill_n(inject.mutable_data(),inject.numel(),0.0f);
        for(int branch=0;branch<4;++branch)
            std::fill_n(gain.mutable_data()+branch*hidden,hidden,static_cast<float>(branch+1));
        std::fill_n(normalized.mutable_data(),normalized.numel(),-12345.0f);
        std::fill_n(mixed.mutable_data(),mixed.numel(),-12345.0f);
        std::fill_n(written.mutable_data(),written.numel(),-12345.0f);
        auto *hostControls=static_cast<std::int32_t *>(controls.raw_mutable_data());
        hostControls[0]=-99;hostControls[1]=-99;hostControls[2]=rows;hostControls[3]=rows;
        auto *hostTokens=static_cast<std::int32_t *>(tokens.raw_mutable_data());
        for(int row=0;row<rows;++row)hostTokens[row]=row+17;
        auto *hostPrior=static_cast<std::int32_t *>(prior.raw_mutable_data());hostPrior[0]=7;hostPrior[1]=11;
        for(TensorBase *tensor:std::vector<TensorBase *>{&input,&gain,&logits,&inject,&normalized,&mixed,&written,
                                                       &controls,&tokens,&prior,&nextTokens})
        {
            TransferEngine::prepareDeviceInput(tensor,device,storage.stream);
            // Preparation materializes storage; the exact producer event must
            // also be joined before native capture starts, as the executor does.
            TransferEngine::requireDeviceInput(tensor,device,storage.stream);
        }
        const Qwen4ExpStageRows extent{rows,Int(controls,BufferId::PREFILL_CHUNK_GEOMETRY,8)};
        Stage norm({.device_id=device,.rows=extent,.operation=Stage::BranchRMSNorm{hidden,1e-6f,
            Float(input,BufferId::RESIDUAL),Weight(gain),Float(normalized,BufferId::NORMALIZED)}});
        Stage read({.device_id=device,.rows=extent,.operation=Stage::ResidualRead{hidden,
            Float(normalized,BufferId::NORMALIZED),Float(logits,BufferId::GATE_PROJ),Float(mixed,BufferId::ATTN_OUTPUT)}});
        Stage write({.device_id=device,.rows=extent,.operation=Stage::ResidualWrite{hidden,
            Float(input,BufferId::RESIDUAL),Float(mixed,BufferId::ATTN_OUTPUT),Float(inject,BufferId::Q_PROJ),
            Float(written,BufferId::HIDDEN_STATE)}});
        Stage commit({.device_id=device,.rows=extent,.operation=Stage::CommitTokenHistory{
            {Int(controls,BufferId::PREFILL_CHUNK_GEOMETRY,12),0},Int(prior,BufferId::PREFILL_CHUNK_TOKEN_IDS),
            Int(tokens,BufferId::REQUEST_TOKEN_IDS),Int(nextTokens,BufferId::MOE_EXPERT_INDICES)}});
        for(auto *stage:std::array<IComputeStage *,4>{&norm,&read,&write,&commit})stage->setGPUStream(storage.stream);
        EXPECT_EQ(norm.bufferContract().weight_tensors.size(),1u);
        EXPECT_EQ(norm.bufferContract().inputs.size(),2u);
        EXPECT_EQ(commit.bufferContract().inputs.size(),3u);
        storage.graph=worker.createGraphCapture(storage.stream);
        {
            ScopedBackendGraphCapture capture(worker,*storage.graph,"Qwen4Exp backend-neutral stages");
            require(capture.begin(),"Qwen4Exp stage recording failed");
            require(norm.execute(context.get()),"Qwen4Exp norm stage recording failed");
            require(read.execute(context.get()),"Qwen4Exp read stage recording failed");
            require(write.execute(context.get()),"Qwen4Exp write stage recording failed");
            require(commit.execute(context.get()),"Qwen4Exp token commit recording failed");
            capture.finish();
        }
        EXPECT_EQ(storage.graph->nodeCount(),4u);
        require(storage.graph->instantiate(),"Qwen4Exp stage instantiation failed");
        for(int live: {15,3,1,0,15,2})
        {
            const int accepted=live/2;
            auto *deviceControls=static_cast<std::int32_t *>(controls.gpu_data_ptr());
            storage.upload(deviceControls,std::vector<std::int32_t>{-99,-99,live,accepted});
            storage.upload(static_cast<float *>(normalized.gpu_data_ptr()),std::vector<float>(rows*channels,-12345));
            storage.upload(static_cast<float *>(mixed.gpu_data_ptr()),std::vector<float>(rows*hidden,-12345));
            storage.upload(static_cast<float *>(written.gpu_data_ptr()),std::vector<float>(rows*channels,-12345));
            require(storage.graph->launch(),"Qwen4Exp stage retained replay failed");
            const auto actual=storage.download(static_cast<const float *>(written.gpu_data_ptr()),rows*channels);
            for(int row=0;row<live;++row)
                for(int channel=0;channel<hidden;++channel)
                {
                    double mixedValue=0;
                    for(int branch=0;branch<4;++branch)
                    {
                        double squareSum=0;
                        for(int c=0;c<hidden;++c)
                        {
                            const double value=original[(row*4+branch)*hidden+c];squareSum+=value*value;
                        }
                        mixedValue+=0.125*original[(row*4+branch)*hidden+channel]*(branch+1)/
                                    std::sqrt(squareSum/hidden+1e-6);
                    }
                    for(int branch=0;branch<4;++branch)
                        EXPECT_NEAR(actual[(row*4+branch)*hidden+channel],
                                    original[(row*4+branch)*hidden+channel]+mixedValue,2e-6);
                }
            EXPECT_TRUE(std::all_of(actual.begin()+live*channels,actual.end(),[](float value){return value==-12345;}));
            const auto actualTokens=storage.download(static_cast<const std::int32_t *>(nextTokens.gpu_data_ptr()),2);
            for(int index=0;index<2;++index)
            {
                const int source=accepted-2+index;
                EXPECT_EQ(actualTokens[index],source<0?hostPrior[source+2]:hostTokens[source]);
            }
        }
        storage.graph.reset(); // Retire captured tensor pointers before tensor destruction.
    }).get();
}
/** @brief Capture Flash Next's actual forty-eight 128-channel sigmoid-normalized GDN heads. */
void proveSigmoidGDN(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested GPU backend unavailable";
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        auto context=IDeviceContext::create(device,1);
        constexpr int heads=48,headWidth=128,channels=heads*headWidth;
        for(const int rows:{1,15,512,1024})
        {
            SCOPED_TRACE(::testing::Message()<<device.toString()<<" sigmoid GDN M="<<rows);
            Storage storage{device,backend,ExplicitGPUStream(worker.defaultStream()).get(),{}, {}};
            const auto count=static_cast<std::size_t>(rows)*channels;
            FP32Tensor input({rows,channels},DeviceId::cpu()),gate({rows,channels},DeviceId::cpu()),
                gamma({headWidth},DeviceId::cpu()),output({rows,channels},DeviceId::cpu());
            const auto hostInput=values(count,7),hostGate=values(count,19),hostGamma=values(headWidth,29);
            std::copy(hostInput.begin(),hostInput.end(),input.mutable_data());
            std::copy(hostGate.begin(),hostGate.end(),gate.mutable_data());
            std::copy(hostGamma.begin(),hostGamma.end(),gamma.mutable_data());
            std::fill_n(output.mutable_data(),count,-12345.0f);
            for(TensorBase *tensor:std::array<TensorBase *,4>{&input,&gate,&gamma,&output})
            {
                TransferEngine::prepareDeviceInput(tensor,device,storage.stream);
                TransferEngine::requireDeviceInput(tensor,device,storage.stream);
            }
            GatedRMSNormStage stage({.device_id=device,.input=&input,.gate=&gate,.output=&output,.gamma=&gamma,
                .eps=1e-6f,.subtract_one=false,.seq_len=rows,.feature_dim=channels,.norm_dim=headWidth,
                .gate_activation=GatedNormActivation::Sigmoid,.input_buffer_id=BufferId::RESIDUAL,
                .gate_buffer_id=BufferId::GATE_PROJ,.output_buffer_id=BufferId::NORMALIZED});
            stage.setGPUStream(storage.stream);
            storage.graph=worker.createGraphCapture(storage.stream);
            {
                ScopedBackendGraphCapture capture(worker,*storage.graph,"Flash Next sigmoid GDN norm");
                require(capture.begin(),"Sigmoid GDN stage recording failed");
                require(stage.execute(context.get()),"Sigmoid GDN stage execute failed");
                capture.finish();
            }
            EXPECT_EQ(storage.graph->nodeCount(),1u);
            std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
            require(storage.graph->inspectKernelNodes(nodes,&error),error.c_str());
            for(const auto &node:nodes)EXPECT_EQ(node.local_memory_bytes_per_thread,0u);
            require(storage.graph->instantiate(),"Sigmoid GDN instantiation failed");
            auto *serial=storage.allocate<float>(count);
            for(int replay=0;replay<2;++replay)
            {
                require(storage.graph->launch(),"Sigmoid GDN retained replay failed");
                for(int row=0;row<rows;++row)
                {
                    const auto offset=static_cast<std::size_t>(row)*channels;bool ok=false;
#ifdef HAVE_CUDA
                    if(device.is_cuda())ok=cudaGDN_gated_rmsnorm(static_cast<const float *>(input.gpu_data_ptr())+offset,
                        static_cast<const float *>(gate.gpu_data_ptr())+offset,static_cast<const float *>(gamma.gpu_data_ptr()),
                        serial+offset,1,channels,headWidth,headWidth,1e-6f,false,GatedNormActivation::Sigmoid,device.ordinal,storage.stream);
#endif
#ifdef HAVE_ROCM
                    if(device.is_rocm())ok=rocmGDN_gated_rmsnorm(static_cast<const float *>(input.gpu_data_ptr())+offset,
                        static_cast<const float *>(gate.gpu_data_ptr())+offset,static_cast<const float *>(gamma.gpu_data_ptr()),
                        serial+offset,1,channels,headWidth,headWidth,1e-6f,false,GatedNormActivation::Sigmoid,device.ordinal,storage.stream);
#endif
                    require(ok,"Sigmoid GDN native serial oracle failed");
                }
                expectEqual(storage,static_cast<const float *>(output.gpu_data_ptr()),serial,count);
                const auto actual=storage.download(static_cast<const float *>(output.gpu_data_ptr()),count);
                for(int group=0;group<rows*heads;++group)
                {
                    double sum=0;
                    for(int c=0;c<headWidth;++c){const double value=hostInput[group*headWidth+c];sum+=value*value;}
                    const double scale=1/std::sqrt(sum/headWidth+1e-6);
                    for(int c=0;c<headWidth;++c)
                    {
                        const auto index=static_cast<std::size_t>(group)*headWidth+c;
                        const double expected=hostInput[index]*scale*hostGamma[c]/(1+std::exp(-static_cast<double>(hostGate[index])));
                        ASSERT_NEAR(actual[index],expected,3e-6);
                    }
                }
            }
            storage.graph.reset();
        }
    }).get();
}

/** @brief Private exact ninety-byte IQ4_NL rows matching Flash Next's native source geometry. */
struct LookupFile
{
    /** @brief Populate a tiny complete native table before immutable admission. */
    explicit LookupFile(std::size_t offset=0) : offset(offset)
    {
        char pattern[]="/tmp/llaminar-native-lookup-XXXXXX";fd=::mkstemp(pattern);
        if(fd<0)throw std::runtime_error("Native lookup fixture creation failed");path=pattern;
        for(int row=0;row<97;++row)for(int block=0;block<5;++block)
        {
            auto &value=blocks[row*5+block];value.d=0x3c00;
            for(int c=0;c<16;++c)value.qs[c]=static_cast<std::uint8_t>(((row+block+c)%16)|(((row*3+block+c)%16)<<4));
        }
        const std::vector<std::byte> prefix(offset);
        if(offset && ::write(fd,prefix.data(),prefix.size())!=static_cast<ssize_t>(prefix.size()))
            throw std::runtime_error("Native lookup fixture prefix write failed");
        if(::write(fd,blocks.data(),sizeof(blocks))!=sizeof(blocks))throw std::runtime_error("Native lookup fixture write failed");
    }
    /** @brief Retire only this fixture after its immutable table owners. */
    ~LookupFile(){::close(fd);::unlink(path.c_str());}
    /** @brief Exact original native directory entry. */
    ImmutableRowTableSource source() const
    {return {.path=path,.tensor_name="per_layer_token_embd.weight",.file_offset=offset,.rows=97,.row_elements=160,.format=GGUFTensorType::IQ4_NL};}
    int fd=-1;std::string path;std::size_t offset;std::array<IQ4_NLBlock,97*5> blocks{};
};

/**
 * @brief Exercise admitted resident loading and the real backend-neutral captured stage.
 * @param device Exact CUDA/HIP consumer, selected by its explicit integration lane.
 *
 * The original file is deliberately unaligned and exceeds the ordinary loader's
 * released upload slot. The production planner, loader, ModelContext, transfer
 * event acquire, canonical controller borrow and compute stage all participate.
 * Only terminal test observations copy bytes back; no host row service exists.
 */
void proveResidentOwnerStage(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested GPU backend unavailable";
    if(!hasCPUBackend())initCPUBackend(-1);
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&] {
        constexpr int capacity=1024,width=160;
        constexpr std::size_t slot=4096,rowBytes=90,maxIds=capacity*16;
        constexpr std::size_t controlBytes=sampling_math::kDeviceGenerationControlCount*sizeof(int);
        constexpr std::size_t arenaBytes=controlBytes+sizeof(int)+maxIds*sizeof(std::int64_t)+maxIds*width*sizeof(float);
        LookupFile file(37);const auto source=file.source();
        const auto resource=[](DeviceId id) {
            return PhysicalMemoryResource{.world_rank=0,.device=id,
                .total_bytes=1u<<28,.admission_available_bytes=1u<<28};
        };
        const auto gpu=resource(device),host=resource(DeviceId::cpu());
        PhysicalMemoryPlanBuilder higher;
        higher.add(gpu,PhysicalMemoryOwner::PrimaryModelWeights,slot);
        higher.add(gpu,PhysicalMemoryOwner::WeightLoadStaging,slot);
        higher.add(host,PhysicalMemoryOwner::WeightLoadStaging,slot);
        higher.add(gpu,PhysicalMemoryOwner::ActivationArena,arenaBytes);
        const auto geometry=resolveGPUWeightLoadMemoryGeometry(slot,
            {.staging_stream_count=1,.staging_budget_bytes=slot});
        ASSERT_TRUE(residentImmutableRowUploadFits(rowBytes,geometry));
        const auto input=ImmutableRowTablePlacementInput::forConsumers(
            ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()),host,std::array{gpu});
        const auto selected=ImmutableRowTablePlacementPlanner::select(
            {.storage=ImmutableRowStoragePreference::VRAM},input,higher.build(),
            {.vram=residentImmutableRowUploadIncrementalBOM()});
        auto memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(selected.physical_plan),0);

        // Issue the actual setup capability only after loading and draining an
        // ordinary weight. Ngram size never participates in staging resolution.
        LoadOrchestrator loader(backend,memory,PhysicalMemoryOwner::PrimaryModelWeights);
        loader.addDevice(device.ordinal);loader.planRawWeight(device.ordinal,"ordinary.fixture",1,slot/sizeof(float),slot);
        loader.allocate(slot,1);
        const std::vector<float> ordinary(slot/sizeof(float),0.25F);
        loader.addWeightJob(device.ordinal,{.name="ordinary.fixture",.host_raw_data=ordinary.data(),
            .raw_bytes=slot,.format=RepackFormat::RAW_FP,.N=1,.K=slot/sizeof(float),.is_asymmetric=false});
        loader.load();loader.finalize();
        const auto receipt=loader.releasedWeightLoadStaging(device.ordinal);
        auto model=ModelContext::createForTesting(file.path,nullptr,1,false);
        auto &directory=const_cast<GGUFModel &>(model->model());
        directory.split_paths={file.path};directory.data_offset=file.offset;
        directory.tensors.push_back({.name=source.tensor_name,.dimensions={source.rows,source.row_elements},
            .type=source.format,.offset=0,.size_bytes=source.tensorBytes(),.split_idx=0});
        void *upload=worker.getOrCreateAuxiliaryStream("resident_native_table_fixture_upload");
        ASSERT_NE(upload,nullptr);ASSERT_NE(upload,worker.defaultStream());
        model->prepareImmutableRowTableForConsumer(*selected.placement,memory,device,
            ImmutableRowTableDevicePreparation{device,upload,receipt},TransferEngine::instance());
        model->requireImmutableRowTablePlacementComplete(*selected.placement,memory);
        auto table=model->residentImmutableRowTable(device);ASSERT_TRUE(table);
        EXPECT_FALSE(model->hostImmutableRowProvider());EXPECT_EQ(table->nativeBytes(),source.tensorBytes());
        EXPECT_EQ(model->residentImmutableRowTableBytes(device),source.tensorBytes());
        EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableTableStorage,
            PhysicalMemoryMaterializationKind::NewAllocation),source.tensorBytes());
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
        model->prepareImmutableRowTableForConsumer(*selected.placement,memory,device,std::nullopt,TransferEngine::instance());
        EXPECT_EQ(model->residentImmutableRowTable(device),table);

        auto arenaClaim=memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,arenaBytes);
        test::MockMPIContext mpi;TensorFactory factory(mpi);BufferArena arena(ArenaConfig{.factory=&factory});
        ASSERT_TRUE(arena.registerBuffer(BufferId::MTP_GENERATION_CONTROL,1,
            sampling_math::kDeviceGenerationControlCount,"INT32",device));
        ASSERT_TRUE(arena.registerBuffer(BufferId::PREFILL_CHUNK_GEOMETRY,1,1,"INT32",device));
        ASSERT_TRUE(arena.registerBuffer(BufferId::QFN_NGRAM_ROW_IDS,capacity,32,"INT32",device));
        ASSERT_TRUE(arena.registerBuffer(BufferId::QFN_PLE_EMBEDDING,capacity,16*width,"FP32",device));
        ASSERT_TRUE(arena.allocate());
        for(auto id:{BufferId::MTP_GENERATION_CONTROL,BufferId::PREFILL_CHUNK_GEOMETRY,
            BufferId::QFN_NGRAM_ROW_IDS,BufferId::QFN_PLE_EMBEDDING})
        {
            auto *tensor=arena.getTensor(id);
            std::memset(tensor->raw_mutable_data(),0,tensor->size_bytes());
            TransferEngine::prepareDeviceInput(tensor,device,worker.defaultStream());
            TransferEngine::requireDeviceInput(tensor,device,worker.defaultStream());
        }
        ASSERT_EQ(arena.getTensor(BufferId::QFN_NGRAM_NATIVE_ROWS),nullptr);
        ASSERT_EQ(arena.getTensor(BufferId::QFN_NGRAM_LOOKUP_STATE),nullptr);
        const auto control=DeviceGenerationControlBinding::admit(arena,device,0);
        auto *idTensor=arena.getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS).get();
        auto *liveTensor=arena.getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY).get();
        auto *outputTensor=arena.getSharedTensor(BufferId::QFN_PLE_EMBEDDING).get();
        auto context=IDeviceContext::create(device,1);
        ImmutableLookupStage stage({.device_id=device,
            .rows={capacity,Qwen4ExpIntBinding(*liveTensor,BufferId::PREFILL_CHUNK_GEOMETRY)},
            .ids=Qwen4ExpIntBinding(*idTensor,BufferId::QFN_NGRAM_ROW_IDS),
            .output=Qwen4ExpFloatBinding(*outputTensor,BufferId::QFN_PLE_EMBEDDING),
            .format=TensorType::IQ4_NL,.rowElements=width,.rowBytes=rowBytes,
            .storage=ImmutableLookupStage::ResidentGPUStorage{table,control}});
        // Declared last: retire graph and join terminal I/O before its stage,
        // arena, resident owner and ordinary loader lose their backing storage.
        Storage storage{device,backend,ExplicitGPUStream(worker.defaultStream()).get(),{}, {}};
        ASSERT_TRUE(stage.prepareGraphLaunch(context.get(),storage.stream));
        const auto identity=stage.graphCaptureVariantSignature();
        storage.graph=worker.createGraphCapture(storage.stream);
        {
            ScopedBackendGraphCapture capture(worker,*storage.graph,"admitted resident native lookup stage");
            require(capture.begin(),"Resident stage capture failed");
            require(stage.execute(context.get()),"Resident stage recording failed");capture.finish();
        }
        EXPECT_EQ(storage.graph->nodeCount(),2u);
        std::vector<GPUGraphKernelNodeInfo> nodes;std::string detail;
        require(storage.graph->inspectKernelNodes(nodes,&detail),detail.c_str());ASSERT_EQ(nodes.size(),2u);
        for(const auto &node:nodes)EXPECT_EQ(node.local_memory_bytes_per_thread,0u);
        require(storage.graph->instantiate(),"Resident stage instantiate failed");
        auto *ids=static_cast<std::int64_t *>(idTensor->gpu_data_ptr());
        auto *live=static_cast<int *>(liveTensor->gpu_data_ptr());
        auto *output=static_cast<float *>(outputTensor->gpu_data_ptr());
        const auto nativePointer=table->view().nativeData();
        const std::vector<float> poison(maxIds*width,-12345.0F);
        std::vector<std::int64_t> hostIds(maxIds);
        for(const int rows:{capacity,15,1,0,512,2,capacity})
        {
            SCOPED_TRACE(rows);const auto count=static_cast<std::size_t>(rows)*16;
            for(std::size_t i=0;i<maxIds;++i)hostIds[i]=i<count?static_cast<std::int64_t>((i*31+i/16)%97):-1;
            storage.upload(ids,hostIds);storage.upload(live,std::vector{rows});storage.upload(output,poison);
            // Cold prefill owns no response-admission Ok flag yet. No host
            // health seed is required for this legitimate all-zero controller.
            storage.upload(control.row(),std::vector<int>(sampling_math::kDeviceGenerationControlCount));
            require(storage.graph->launch(),"Resident stage replay failed");
            const auto actual=storage.download(output,poison.size());
            auto expected=poison;std::vector<std::byte> native(count*rowBytes);
            for(std::size_t i=0;i<count;++i)
                std::memcpy(native.data()+i*rowBytes,file.blocks.data()+hostIds[i]*5,rowBytes);
            decodeCPUImmutableLookupRows(TensorType::IQ4_NL,width,rowBytes,native.data(),count,expected.data());
            ASSERT_EQ(std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(float)),0);
            EXPECT_EQ(storage.download(control.error(),1).front(),0);
            EXPECT_EQ(stage.graphCaptureVariantSignature(),identity);
            EXPECT_EQ(table->view().nativeData(),nativePointer);
            EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableTableStorage,
                PhysicalMemoryMaterializationKind::NewAllocation),source.tensorBytes());
        }
        hostIds[0]=-1;storage.upload(ids,hostIds);storage.upload(live,std::vector{1});storage.upload(output,poison);
        require(storage.graph->launch(),"Invalid resident stage replay failed");
        EXPECT_EQ(storage.download(control.error(),1).front(),static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
        const auto failed=storage.download(output,poison.size());
        EXPECT_EQ(std::memcmp(failed.data(),poison.data(),poison.size()*sizeof(float)),0);
        const auto native=storage.download(nativePointer,source.tensorBytes());
        EXPECT_EQ(std::memcmp(native.data(),file.blocks.data(),native.size()),0);
        storage.graph.reset();
    });
}
/** @brief Select the real capture-frontier admission proof without changing storage policy. */
enum class LookupFrontierProof { ExistingProtocol, AdmissionPublication };
/** @brief Retire the captured lookup before any borrowed stage/arena/source owner is destroyed. */
struct LookupCaptureRetirement
{
    Storage &storage;void *admissionStream;
    /** @brief Join both exact producers and release captured pointers before any owner on all exits. */
    ~LookupCaptureRetirement() noexcept
    {
        if(!storage.backend->synchronizeStream(admissionStream,storage.device.ordinal) ||
            !storage.backend->synchronizeStream(storage.stream,storage.device.ordinal))std::terminate();
        storage.graph.reset();
    }
};
/** @brief Own the exact distinct admission stream through capture-frontier and terminal cleanup. */
struct LookupAdmissionStream
{
    IBackend &backend;DeviceId device;void *stream;
    /** @brief Create one non-default setup stream; no inference-time allocation occurs.
     * @param native Exact native backend. @param target Owning participant.
     */
    LookupAdmissionStream(IBackend &native,DeviceId target):backend(native),device(target),stream(native.createStream(target.ordinal))
    {require(stream && reinterpret_cast<std::uintptr_t>(stream)>2,"Lookup admission stream creation failed");}
    /** @brief Retire admission work before releasing its exact stream. */
    ~LookupAdmissionStream() noexcept
    {if(!backend.synchronizeStream(stream,device.ordinal))std::terminate();backend.destroyStream(stream,device.ordinal);}
};
/** @brief Select optional actual-service placement observation without changing protocol work. */
enum class LookupWorkerPlacementProof { Disabled, ActualOwnedWorker };
/** @brief Prove a real captured CPU-storage boundary without exposing model tokens to its worker.
 * @param device Exact native participant. @param adaptive Existing bounded RAM/SSD provider arm.
 * @param frontier Whether the real external-arena RequireReadyBytes frontier is tested before/after admission.
 * @param placement Whether the actual retained service worker's placement is observed.
 */
void proveLookup(DeviceId device,bool adaptive=false,LookupFrontierProof frontier=LookupFrontierProof::ExistingProtocol,
    LookupWorkerPlacementProof placement=LookupWorkerPlacementProof::Disabled)
{
    auto *backend=getBackendFor(device);
    if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested GPU backend unavailable";
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        constexpr int capacity=1025,width=160;
        constexpr std::size_t rowBytes=90,maxIds=capacity*16;
        LookupFile file;
        // The same real captured service proof covers unchanged whole-tier
        // controls and explicit zero/nonzero adaptive RAM/SSD ownership.
        for(const auto mode:{ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped})
        {
            const std::size_t slots=mode==ImmutableRowStorage::RAM?0u:2u;
            SCOPED_TRACE(::testing::Message()<<device.toString()<<" adaptive="<<adaptive<<" mode="<<static_cast<int>(mode));
            const auto layout=ImmutableLookupLayout::create(maxIds,rowBytes);
            const auto policy=ImmutableRowTableStoragePolicy::forStorage(mode,4096,2);
            PhysicalMemoryPlanBuilder plan;
            const PhysicalMemoryResource resource{.world_rank=0,.device=DeviceId::cpu(),
                .total_bytes=1u<<28,.admission_available_bytes=1u<<28};
            std::optional<CpuAdaptiveImmutableRowCacheGeometry> cache;
            if(adaptive)
            {
                const PhysicalMemoryAllocatorIdentity host{0,DeviceId::cpu()};
                const auto src=file.source();const std::array tiers{ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
                const auto geometry=ImmutableRowTableLayoutGeometry::checked(
                    ImmutableRowTableExtent::fromNative(src.format,src.rows,src.rowBytes()),ImmutableRowTableTiers::of(tiers),
                    ImmutableRowTableResidency::Adaptive,{{{host,ImmutableRowStorage::RAM},0,slots,{}}},{{0,device}},{},
                    {{host,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2)}},{},std::nullopt,maxIds);
                cache=CpuAdaptiveImmutableRowCacheGeometry::fromLayout(geometry,host,maxIds);
                const auto bom=cache->bom(resource);for(const auto &line:bom.resources())plan.add(line);
            }
            else plan.add(resource,PhysicalMemoryOwner::ImmutableTableStorage,ImmutableRowTable::physicalBytes(file.source(),policy));
            plan.add(resource,PhysicalMemoryOwner::ImmutableLookupStaging,ImmutableLookupService::hostBytes(layout));
            auto memory=std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
            auto table=adaptive?HostImmutableRowProvider::fromAdaptiveCache(CpuAdaptiveImmutableRowCache::create(file.source(),*cache,memory)):
                HostImmutableRowProvider::fromWholeTable(std::make_unique<ImmutableRowTable>(
                    file.source(),policy, DeviceId::cpu(),memory,ImmutableRowTableHostPlacement::taskPolicy()));
            std::unique_ptr<immutable_lookup_affinity_test::Probe> affinity;
            if (placement == LookupWorkerPlacementProof::ActualOwnedWorker)
                affinity = std::make_unique<immutable_lookup_affinity_test::Probe>(file.fd);
            auto service=ImmutableLookupService::create(table,layout,std::array{device},memory,DeviceId::cpu());
            if (affinity) affinity->created();
            LookupAdmissionStream admissionStream(*backend,device);
            auto protocolOwner=std::make_shared<INT32Tensor>(std::vector<std::size_t>{16});
            auto &stateTensor=*protocolOwner;
            INT32Tensor liveTensor({1}),idTensor({capacity,32}),nativeTensor({maxIds*rowBytes/4});
            FP32Tensor outputTensor({capacity,16*width},DeviceId::cpu());
            Storage storage{device,backend,ExplicitGPUStream(worker.defaultStream()).get(),{}, {}};
            std::memset(stateTensor.raw_mutable_data(),0xa5,sizeof(ImmutableLookupDeviceState));
            *static_cast<int *>(liveTensor.raw_mutable_data())=capacity;
            std::vector<std::int64_t> hostIds(maxIds);
            for(std::size_t i=0;i<hostIds.size();++i)hostIds[i]=(i*31+i/16)%97;
            std::memcpy(idTensor.raw_mutable_data(),hostIds.data(),hostIds.size()*8);
            for(TensorBase *tensor:std::array<TensorBase *,5>{&stateTensor,&liveTensor,&idTensor,&nativeTensor,&outputTensor})
            {
                TransferEngine::prepareDeviceInput(tensor,device,storage.stream);
                TransferEngine::requireDeviceInput(tensor,device,storage.stream);
            }
            auto *state=static_cast<ImmutableLookupDeviceState *>(stateTensor.gpu_data_ptr());
            auto *generation=reinterpret_cast<std::uint64_t *>(reinterpret_cast<std::byte *>(state)+offsetof(ImmutableLookupDeviceState,admissionGeneration));
            std::unique_ptr<IImmutableLookupKernels> native;
#ifdef HAVE_CUDA
            if(device.is_cuda())native=createCUDAImmutableLookupKernels(device,TensorType::IQ4_NL,width,rowBytes);
#endif
#ifdef HAVE_ROCM
            if(device.is_rocm())native=createROCmImmutableLookupKernels(device,TensorType::IQ4_NL,width,rowBytes);
#endif
            auto boundary=std::make_shared<ImmutableLookupExecutionBoundary>(service,device,std::move(native),protocolOwner);
            EXPECT_THROW(boundary->terminalObservation(),std::logic_error);
            std::vector<int> initialController(sampling_math::kDeviceGenerationControlCount);
            initialController[sampling_math::kDeviceGenerationControlOk]=1;
            initialController[sampling_math::kDeviceGenerationControlDepthPolicyMode]=static_cast<int>(sampling_math::DeviceGenerationPolicyMode::Ordinary);
            auto *controller=storage.input(initialController);
            auto *healthy=controller+sampling_math::kDeviceGenerationControlOk;
            auto *error=controller+sampling_math::kDeviceGenerationControlErrorCode;
            boundary->bindGenerationControl({healthy,error});
            auto *live=static_cast<int *>(liveTensor.gpu_data_ptr());
            auto *ids=static_cast<std::int64_t *>(idTensor.gpu_data_ptr());
            auto *scratch=static_cast<std::byte *>(nativeTensor.gpu_data_ptr());
            auto *output=static_cast<float *>(outputTensor.gpu_data_ptr());
            auto context=IDeviceContext::create(device,1);
            auto stageOwner=std::make_unique<ImmutableLookupStage>(ImmutableLookupStage::Params{
                .device_id=device,.rows={capacity,Qwen4ExpIntBinding(liveTensor,BufferId::PREFILL_CHUNK_GEOMETRY)},
                .ids=Qwen4ExpIntBinding(idTensor,BufferId::QFN_NGRAM_ROW_IDS),
                .output=Qwen4ExpFloatBinding(outputTensor,BufferId::QFN_PLE_EMBEDDING),
                .format=TensorType::IQ4_NL,.rowElements=width,.rowBytes=rowBytes,
                .storage=ImmutableLookupStage::GPUStorage{boundary,Qwen4ExpIntBinding(stateTensor,BufferId::QFN_NGRAM_LOOKUP_STATE),Qwen4ExpIntBinding(nativeTensor,BufferId::QFN_NGRAM_NATIVE_ROWS)}});
            auto *stage=stageOwner.get();
            test::MockMPIContext mpi;TensorFactory factory(mpi);BufferArena frontierArena(ArenaConfig{.factory=&factory});
            for(const auto &[role,tensor]:std::array<std::pair<BufferId,ITensor *>,5>{{
                {BufferId::QFN_NGRAM_LOOKUP_STATE,&stateTensor},{BufferId::PREFILL_CHUNK_GEOMETRY,&liveTensor},
                {BufferId::QFN_NGRAM_ROW_IDS,&idTensor},{BufferId::QFN_NGRAM_NATIVE_ROWS,&nativeTensor},
                {BufferId::QFN_PLE_EMBEDDING,&outputTensor}}})
                require(frontierArena.registerExternalBuffer(role,tensor),"Lookup original external arena binding failed");
            ComputeGraph frontierGraph;frontierGraph.addNode("lookup_protocol_frontier",std::move(stageOwner),device);
            DeviceGraphExecutor executor;executor.setArena(&frontierArena);
            // Declared last so exceptional assertion/infrastructure exits retire
            // the native recording before its stage, arena and source owners.
            LookupCaptureRetirement retirement{storage,admissionStream.stream};
            require(stage->prepareGraphLaunch(context.get(),storage.stream),
                "Immutable lookup capture-only storage seal failed");
            require(executor.prepareGraphStorageForCapture(frontierGraph,context.get(),storage.stream,
                "lookup_initial_ready_frontier"),"Lookup real initial arena frontier failed");
            require(admissionStream.stream!=storage.stream,"Lookup admission/capture streams must be distinct");
            storage.graph=worker.createGraphCapture(storage.stream);
            {
                ScopedBackendGraphCapture capture(worker,*storage.graph,"retained exact native immutable lookup");
                require(capture.begin(),"Immutable lookup capture failed");
                require(stage->execute(context.get()),"Immutable lookup backend-neutral stage capture failed");
                boundary->recordTerminalClose(storage.stream);capture.finish();
            }
            EXPECT_EQ(storage.graph->nodeCount(),8u);
            std::vector<GPUGraphKernelNodeInfo> nodes;std::string message;
            require(storage.graph->inspectKernelNodes(nodes,&message),message.c_str());
            for(const auto &node:nodes)EXPECT_EQ(node.local_memory_bytes_per_thread,0u);
            require(storage.graph->instantiate(),"Immutable lookup retained instantiation failed");
            const auto captured_identity=stage->graphCaptureVariantSignature();
            const auto protocol_identity=boundary->captureIdentity();
            const auto retained_owner=table->retainedOwnerObservation();
            std::uint64_t epoch=0,idCount=0;
            for(const int rows:{capacity,15,1,0,512,1024,2,capacity})
            {
                storage.upload(live,std::vector<int>{rows});
                storage.upload(output,std::vector<float>(maxIds*width,-12345));
                storage.upload(scratch,std::vector<std::byte>(maxIds*rowBytes,std::byte{0xa5}));
                const bool proveFrontier=frontier==LookupFrontierProof::AdmissionPublication;
                if(proveFrontier)
                {
                    // Publish this complete diagnostic host overwrite explicitly:
                    // mutable access from DEVICE_AUTHORITATIVE alone has a SYNCED
                    // transition, so it cannot author this repeated stale-byte proof.
                    // Never upload the poison: the real native admit owns bytes.
                    std::memset(stateTensor.raw_mutable_data(),0xa5,sizeof(ImmutableLookupDeviceState));
                    TransferEngine::publishHostWrite(&stateTensor);
                    EXPECT_TRUE(stateTensor.isHostAuthoritative());
                    EXPECT_FALSE(executor.prepareGraphStorageForCapture(frontierGraph,context.get(),storage.stream,
                        "lookup_before_native_admission"));
                }
                boundary->beginAdmission(++epoch,proveFrontier?admissionStream.stream:storage.stream);
                boundary->requireArmedForLaunch();
                if(proveFrontier)
                {
                    // This checks and joins the actual native admit publication event
                    // on a different capture stream, before any cleanup-only join.
                    EXPECT_TRUE(executor.prepareGraphStorageForCapture(frontierGraph,context.get(),storage.stream,
                        "lookup_after_native_admission"));
                    EXPECT_FALSE(stateTensor.isHostAuthoritative());
                    EXPECT_EQ(boundary->state(),state);EXPECT_EQ(boundary->generation(),generation);
                    // A deliberately unpublished negative control must still
                    // close its armed worker: nonfatal EXPECT above, then setup-
                    // only ordering of the pre-existing safe native recording.
                    // This join cannot repair Tensor coherence or that EXPECT.
                    require(backend->synchronizeStream(admissionStream.stream,device.ordinal),
                        "Lookup diagnostic admission cleanup join failed");
                }
                EXPECT_THROW(boundary->terminalObservation(),std::logic_error);
                require(storage.graph->launch(),"Immutable lookup retained replay failed");
                const auto actual=storage.download(output,maxIds*width);
                boundary->awaitTerminal();
                const auto observation=boundary->terminalObservation();
                if (affinity) affinity->verify();
                EXPECT_EQ(stage->graphCaptureVariantSignature(),captured_identity);
                EXPECT_EQ(boundary->captureIdentity(),protocol_identity);
                EXPECT_EQ(protocolOwner->gpu_data_ptr(),state);
                EXPECT_EQ(observation.generation,epoch);EXPECT_EQ(observation.mode,adaptive?HostImmutableRowMode::AdaptiveRAMSSD:
                    mode==ImmutableRowStorage::RAM?HostImmutableRowMode::WholeRAM:HostImmutableRowMode::WholeMemoryMapped);
                if(adaptive)
                {
                    EXPECT_EQ(observation.counters.adaptiveBatches,1u);
                    EXPECT_EQ(observation.counters.ramRows+observation.counters.sourceRows+observation.counters.duplicateRows,
                        static_cast<std::uint64_t>(rows)*16);
                    if(!slots){EXPECT_EQ(observation.counters.ramRows,0u);EXPECT_EQ(observation.counters.cacheAdmissions,0u);}
                    EXPECT_EQ(table->retainedOwnerObservation().tableStorageBytes(),retained_owner.tableStorageBytes());
                    EXPECT_EQ(table->retainedOwnerObservation().lookupStagingBytes(),retained_owner.lookupStagingBytes());
                }
                EXPECT_EQ(observation.counters.requests,1u);EXPECT_EQ(observation.counters.ids,static_cast<std::uint64_t>(rows)*16);
                EXPECT_EQ(observation.counters.nativeBytes,static_cast<std::uint64_t>(rows)*16*rowBytes);
                EXPECT_EQ(observation.counters.terminalBytes,8u);
                const auto nativeBytes=storage.download(scratch,maxIds*rowBytes);
                const auto liveIds=static_cast<std::size_t>(rows)*16;
                std::vector<float> expected(liveIds*width);
                std::vector<std::byte> packed(liveIds*rowBytes);
                for(std::size_t i=0;i<liveIds;++i)
                    std::memcpy(packed.data()+i*rowBytes,file.blocks.data()+hostIds[i]*5,rowBytes);
                decodeCPUImmutableLookupRows(TensorType::IQ4_NL,width,rowBytes,packed.data(),liveIds,expected.data());
                if(!expected.empty())EXPECT_EQ(std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(float)),0);
                if(!packed.empty())EXPECT_EQ(std::memcmp(nativeBytes.data(),packed.data(),packed.size()),0);
                EXPECT_TRUE(std::all_of(actual.begin()+liveIds*width,actual.end(),[](float v){return v==-12345;}));
                EXPECT_TRUE(std::all_of(nativeBytes.begin()+liveIds*rowBytes,nativeBytes.end(),[](std::byte v){return v==std::byte{0xa5};}));
                EXPECT_EQ(storage.download(healthy,1)[0],1);EXPECT_EQ(storage.download(error,1)[0],0);
                idCount+=liveIds;
            }
            const auto stats=service->stats();EXPECT_EQ(stats.requests,epoch);EXPECT_EQ(stats.ids,idCount);
            EXPECT_EQ(stats.idBytes,idCount*8);EXPECT_EQ(stats.nativeBytes,idCount*rowBytes);
            EXPECT_EQ(stats.metadataBytes,epoch*(kLookupRequestLiveBytes+kLookupCompletionLiveBytes));
            EXPECT_EQ(stats.acknowledgementBytes,epoch*8);EXPECT_EQ(stats.terminalBytes,epoch*8);
            EXPECT_EQ(stats.failures,0u);
            const auto finalState=storage.download(state,1)[0];
            EXPECT_EQ(finalState.issuedSequence,epoch);EXPECT_EQ(finalState.consumedSequence,epoch);
            EXPECT_EQ(finalState.admissionGeneration,epoch);
            {
                // These are protocol-ordering fixtures: the device-copy producer
                // supplies a terminal predicate instead of certifying a sampler.
                // The lookup stage and all storage lifecycle machinery are real.
                auto *one=storage.input(std::vector<int>{1});
                auto child=worker.createGraphCapture(storage.stream);
                // Every new recording needs the current external producer event,
                // even after the previous executable replay has completed. Each
                // admission published a new protocol generation; the original
                // capture's frontier join authenticates only its earlier event.
                require(executor.prepareGraphStorageForCapture(frontierGraph,context.get(),storage.stream,
                    "lookup_complete_transaction_frontier"),"Lookup complete transaction frontier failed");
                {
                    ScopedBackendGraphCapture capture(worker,*child,"lookup complete captured transaction");
                    require(capture.begin(),"Lookup complete transaction recording failed");
                    require(stage->execute(context.get()),"Lookup complete transaction stage failed");
                    require(backend->deviceCopyAsync(controller+sampling_math::kDeviceGenerationControlRequestComplete,
                        one,sizeof(int),device.ordinal,storage.stream),"Lookup device terminal predicate publication failed");
                    capture.finish();
                }
                auto terminal=worker.createGraphCapture(storage.stream);
                auto parent=worker.createGraphCapture(storage.stream);
                const DeviceGenerationGraphControl control{sampling_math::DeviceGenerationPolicy::ordinary(),controller,
                    sampling_math::kDeviceGenerationControlCount,1};
                if(device.is_cuda())
                {
                    ScopedBackendGraphCapture capture(worker,*terminal,"lookup native generation epilogue");
                    require(capture.begin(),"Lookup terminal epilogue recording failed");
                    boundary->recordTerminalClose(storage.stream);capture.finish();
                    const DeviceControlledLoopFragment iteration[]={{"lookup transaction",child.get()}};
                    const DeviceControlledLoopEntryFragment epilogue[]={{"lookup service close",terminal.get()}};
                    std::string construction;
                    require(DeviceGenerationGraphProgram::native(*parent,control,
                        {.iteration=iteration,.terminal=epilogue},construction),construction.c_str());
                }
                else
                {
                    auto *tickets=storage.allocate<sampling_math::DeviceGenerationDispatchTicket>(1);
                    require(backend->enqueueInitializeDeviceGenerationDispatchTicket(1,1,controller,
                        sampling_math::kDeviceGenerationControlCount,1,tickets,device.ordinal,storage.stream),
                        "Lookup hosted ticket admission failed");
                    std::string construction;
                    require(DeviceGenerationGraphProgram::ticketPublisher(*terminal,worker,*backend,device,
                        control,nullptr,tickets,construction,boundary.get()),construction.c_str());
                    const GPUOrderedTimelineStep steps[]={{.name="lookup complete transaction",.capture=child.get()},
                        {.name="authenticated terminal close",.capture=terminal.get()}};
                    require(parent->buildOrderedTimelineTransaction(steps)&&parent->instantiate(),
                        "Lookup retained HIP transaction composition failed");
                }
                for(const auto admission:std::array{std::array{1,0},std::array{1,1},std::array{0,0}})
                {
                    storage.upload(controller,initialController);
                    storage.upload(healthy,std::vector<int>{admission[0]});
                    storage.upload(controller+sampling_math::kDeviceGenerationControlRequestComplete,std::vector<int>{admission[1]});
                    const auto before=service->stats();
                    boundary->beginAdmission(++epoch,storage.stream);
                    // HIP's final immutable ticket names its already-prepared
                    // terminal branch for a zero-work interval. No model bytes
                    // are read to choose between complete retained recordings.
                    const bool noWork=admission[0]==0||admission[1]!=0;
                    require((device.is_rocm()&&noWork?terminal->launch():parent->launch()),
                        "Lookup complete generation parent submission failed");
                    require(backend->synchronizeStream(storage.stream,device.ordinal),"Lookup generation terminal observation failed");
                    boundary->awaitTerminal();
                    const auto after=service->stats();
                    EXPECT_EQ(after.requests-before.requests,noWork?0u:1u);
                    EXPECT_EQ(after.nativeBytes-before.nativeBytes,noWork?0u:maxIds*rowBytes);
                    EXPECT_EQ(after.terminalBytes-before.terminalBytes,8u);
                    if(!noWork)idCount+=maxIds;
                }
                parent.reset();terminal.reset();child.reset();
            }
            // A bad immutable ID must release a typed failed completion, clear
            // canonical generation health, preserve all payload guards and make
            // this lane permanently inadmissible without another graph capture.
            hostIds[0]=-1;storage.upload(ids,hostIds);
            storage.upload(controller,initialController);
            storage.upload(output,std::vector<float>(maxIds*width,-12345));
            storage.upload(scratch,std::vector<std::byte>(maxIds*rowBytes,std::byte{0xa5}));
            boundary->beginAdmission(++epoch,storage.stream);
            require(storage.graph->launch(),"Immutable lookup failed-ID retained replay failed");
            const auto failedOutput=storage.download(output,maxIds*width);
            EXPECT_EQ(storage.download(healthy,1)[0],0);
            EXPECT_EQ(storage.download(error,1)[0],static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
            EXPECT_EQ(storage.download(state,1)[0].status,static_cast<std::uint64_t>(ImmutableLookupStatus::InvalidRowId));
            // Even a later controller initialization cannot disguise this
            // published storage error as another healthy generation interval.
            storage.upload(healthy,std::vector<int>{1});storage.upload(error,std::vector<int>{0});
            EXPECT_THROW(boundary->beginAdmission(epoch,storage.stream),std::runtime_error);
            EXPECT_THROW(boundary->requireArmedForLaunch(),std::runtime_error);
            EXPECT_THROW(boundary->awaitTerminal(),std::runtime_error);
            EXPECT_EQ(boundary->terminalObservation().counters.failures,1u);
            EXPECT_TRUE(std::all_of(failedOutput.begin(),failedOutput.end(),[](float value){return value==-12345;}));
            const auto failedNative=storage.download(scratch,maxIds*rowBytes);
            EXPECT_TRUE(std::all_of(failedNative.begin(),failedNative.end(),[](std::byte value){return value==std::byte{0xa5};}));
            EXPECT_EQ(service->stats().failures,1u);EXPECT_EQ(service->stats().nativeBytes,idCount*rowBytes);
            // Re-entry reports the preserved fatal source diagnostic before
            // considering a new admission epoch; it never rearms this lane.
            EXPECT_THROW(boundary->beginAdmission(epoch+1,storage.stream),std::runtime_error);
            storage.graph.reset();frontierGraph.clear();boundary.reset();service.reset();
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),cache?cache->executionBytes():0u);
            table.reset();
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),0u);
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableTableStorage,
                PhysicalMemoryMaterializationKind::NewAllocation),0u);
        }
    }).get();
}
} // namespace
#ifdef HAVE_CUDA
/** @test Real retained CUDA GR/PLE kernels preserve serial arithmetic and live extents. */
TEST(Qwen4ExpDevicePrimitives, CUDA) { prove(DeviceId::cuda(0)); }
/** @test Backend-neutral CUDA stages retain exact streams and immutable control subviews. */
TEST(Qwen4ExpDevicePrimitives, CUDAStages) { proveStages(DeviceId::cuda(0)); }
/** @test Flash Next CUDA sigmoid GDN norm is captured and grouped-byte invariant. */
TEST(Qwen4ExpDevicePrimitives, CUDASigmoidGDN) { proveSigmoidGDN(DeviceId::cuda(0)); }
/** @test CUDA retained storage service gathers exact IQ4_NL rows in both explicit table modes. */
TEST(Qwen4ExpDevicePrimitives, CUDAImmutableLookup) { proveLookup(DeviceId::cuda(0)); }
/** @test Original CUDA protocol owner becomes ready only after its exact native admission producer. */
TEST(Qwen4ExpDevicePrimitives, CUDAProtocolAdmissionPublication)
{proveLookup(DeviceId::cuda(0),false,LookupFrontierProof::AdmissionPublication);}
/** @test Captured CUDA live rows use an explicit bounded native RAM/SSD provider, including zero cache slots. */
TEST(Qwen4ExpDevicePrimitives, CUDAAdaptiveRAMSSDLookup) { proveLookup(DeviceId::cuda(0),true); }
/** @test CUDA loads one admitted original table and captures its complete backend-neutral lookup stage. */
TEST(Qwen4ExpDevicePrimitives, CUDAResidentOwnerStage) { proveResidentOwnerStage(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
/** @test Real retained HIP GR/PLE kernels preserve serial arithmetic and live extents. */
TEST(Qwen4ExpDevicePrimitives, ROCm) { prove(DeviceId::rocm(0)); }
/** @test Backend-neutral HIP stages retain exact streams and immutable control subviews. */
TEST(Qwen4ExpDevicePrimitives, ROCmStages) { proveStages(DeviceId::rocm(0)); }
/** @test Flash Next HIP sigmoid GDN norm is captured and grouped-byte invariant. */
TEST(Qwen4ExpDevicePrimitives, ROCmSigmoidGDN) { proveSigmoidGDN(DeviceId::rocm(0)); }
/** @test HIP retained storage service gathers exact IQ4_NL rows in both explicit table modes. */
TEST(Qwen4ExpDevicePrimitives, ROCmImmutableLookup) { proveLookup(DeviceId::rocm(0)); }
/** @test Original HIP protocol owner becomes ready only after its exact native admission producer. */
TEST(Qwen4ExpDevicePrimitives, ROCmProtocolAdmissionPublication)
{proveLookup(DeviceId::rocm(0),false,LookupFrontierProof::AdmissionPublication);}
/** @test Captured HIP live rows use an explicit bounded native RAM/SSD provider, including zero cache slots. */
TEST(Qwen4ExpDevicePrimitives, ROCmAdaptiveRAMSSDLookup) { proveLookup(DeviceId::rocm(0),true); }
/** @test HIP loads one admitted original table and captures its complete backend-neutral lookup stage. */
TEST(Qwen4ExpDevicePrimitives, ROCmResidentOwnerStage) { proveResidentOwnerStage(DeviceId::rocm(0)); }
#endif

#ifdef HAVE_CUDA
/** @test Actual CUDA RAM/mmap service retains admitted affinity across captured generations. */
TEST(Qwen4ExpDevicePrimitives, CUDAOwnedLookupWorkerAffinity)
{ proveLookup(DeviceId::cuda(0), false, LookupFrontierProof::ExistingProtocol, LookupWorkerPlacementProof::ActualOwnedWorker); }
#endif
#ifdef HAVE_ROCM
/** @test Actual ROCm RAM/mmap service retains admitted affinity across captured generations. */
TEST(Qwen4ExpDevicePrimitives, ROCmOwnedLookupWorkerAffinity)
{ proveLookup(DeviceId::rocm(0), false, LookupFrontierProof::ExistingProtocol, LookupWorkerPlacementProof::ActualOwnedWorker); }

#endif
