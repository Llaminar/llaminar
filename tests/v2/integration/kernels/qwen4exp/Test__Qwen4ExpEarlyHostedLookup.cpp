/**
 * @file Test__Qwen4ExpEarlyHostedLookup.cpp
 * @brief Native replay of the actual early Hash/Publish/trunk/Consume composition.
 *
 * The production graph-local frame, primitive stages, host table service and
 * delayed PLE consumer all execute in one retained CUDA/HIP recording. Only
 * explicit fixture ingress and terminal oracle reads touch host tensor values.
 * RAM and bounded mmap sources retain their original protocol and PMA owners.
 * This component proof does not certify a full model or claim measured overlap.
 */
#include "fixtures/execution/RetainedHostServiceFixture.h"
#include "backends/NativeOrderingEventGroup.h"
#include "../../../utils/GraphArenaTestHarness.h"
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPLEBuilder.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/KernelFactory.h"
#include "kernels/common/SamplingMath.h"
#include "transfer/TransferEngine.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unistd.h>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::qwen4exp;
namespace {
/** @brief Fail the fixture at the original infrastructure boundary. */
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

/** @brief Original quantized fixture source, removed only after its readers retire. */
class Source final {
public:
    static constexpr int rows = 16 * 7;
    static constexpr int width = 32;
    static constexpr std::size_t rowBytes = sizeof(IQ4_NLBlock);
    std::array<IQ4_NLBlock, rows> blocks{};
    std::string path;

    /** @brief Write one native IQ4_NL block per row with distinct finite values. */
    Source()
    {
        char pattern[] = "/tmp/qfn-early-hosted-XXXXXX";
        const int fd = ::mkstemp(pattern);
        require(fd >= 0, "Cannot create early lookup source");
        path = pattern;
        for (int row = 0; row < rows; ++row) {
            blocks[row].d = 0x3000;
            for (unsigned byte = 0; byte < sizeof(blocks[row].qs); ++byte)
                blocks[row].qs[byte] = static_cast<std::uint8_t>(row * 7 + byte * 13);
        }
        const bool written = ::write(fd, blocks.data(), sizeof(blocks)) == sizeof(blocks);
        const bool closed = ::close(fd) == 0;
        if (!written || !closed) {
            ::unlink(path.c_str());
            throw std::runtime_error("Cannot publish early lookup source");
        }
    }
    /** @brief Remove only this fixture's source after all graph/service owners. */
    ~Source() { ::unlink(path.c_str()); }
    /** @return Exact original row format and source extent, never a converted table. */
    ImmutableRowTableSource descriptor() const
    {
        return {.path=path, .tensor_name="per_layer_token_embd.weight",
            .rows=rows, .row_elements=width, .format=GGUFTensorType::IQ4_NL};
    }
};

/** @brief One retained native recording, retired before any captured owner is released. */
struct Recording final {
    IBackend &backend;
    DeviceId device;
    void *stream;
    std::unique_ptr<IGPUGraphCapture> capture;

    /** @brief Cleanup joins only the exact test terminal boundary; failure is fatal. */
    ~Recording() noexcept
    {
        if (!backend.synchronizeStream(stream, device.ordinal)) std::terminate();
        capture.reset();
    }
    /** @brief Observe terminal bytes without changing tensor coherence or addresses. */
    template<class T> std::vector<T> read(const void *source, std::size_t count) const
    {
        std::vector<T> output(count);
        require(backend.deviceToHostOnStream(output.data(), source, count * sizeof(T),
            device.ordinal, stream), "Early lookup terminal download failed");
        require(backend.synchronizeStream(stream, device.ordinal), "Early lookup terminal completion failed");
        return output;
    }
};

/**
 * @brief Replay real native early publication with original RAM/mmap storage.
 * @param device Exact backend participant selected by its integration registration.
 *
 * Changed token/history/count inputs prove the captured query is current. A
 * useful residual expansion lies between Publish and Consume and is checked
 * independently. Empty/partial requests preserve poisoned tails and transfer
 * only live rows. The same executable survives twenty request epochs per tier.
 */
void proveEarlyHostedLookup(DeviceId device)
{
    auto *backend = getBackendFor(device);
    if (!backend || backend->deviceCount() <= device.ordinal)
        GTEST_SKIP() << "Requested native participant unavailable";
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    auto cpu = llaminar::v2::kernels::KernelFactory::createQwen4Exp(DeviceId::cpu());
    worker.submitAndWait([&] {
        constexpr int capacity = 17, hidden = Source::width;
        constexpr std::size_t maxIds = capacity * 16;
        constexpr float poison = -12345.f;
        constexpr auto idPoison = std::bit_cast<std::int64_t>(std::uint64_t{0xa5a5a5a5a5a5a5a5});
        const std::array<int, 10> liveSequence{capacity, 1, 0, 15, 2, capacity, 0, 3, 16, 1};
        Source source;
        const PhysicalMemoryResource host{0,DeviceId::cpu(),1u<<26,1u<<26};
        const PhysicalMemoryResource gpu{0,device,backend->deviceMemoryTotal(device.ordinal),backend->deviceMemoryFree(device.ordinal)};
        const auto io=RetainedHostServiceProgram::original(RetainedHostServiceRole::ImmutableLookupIO);
        const auto caseBudget=[&](ImmutableRowStorage tier) {
            const auto policy=ImmutableRowTableStoragePolicy::forStorage(tier,4096,2);
            const auto layout = ImmutableLookupLayout::create(maxIds, Source::rowBytes);


            PhysicalMemoryPlanBuilder plan;
            plan.add(host, PhysicalMemoryOwner::ImmutableTableStorage,
                ImmutableRowTable::physicalBytes(source.descriptor(), policy));
            plan.add(host, PhysicalMemoryOwner::ImmutableLookupStaging, ImmutableLookupService::hostBytes(layout));
            return plan.build();
        };
        const std::array cases{caseBudget(ImmutableRowStorage::RAM),caseBudget(ImmutableRowStorage::MemoryMapped)};
        PhysicalMemoryPlanBuilder joint;
        joint.addMutuallyExclusive(cases);
        // Each boundary instance retains its own certified native backing;
        // serialized CPU table/channel storage alone shares the envelope.
        const auto ordering=NativeOrderingEventGroup::bom(NativeOrderingEventProgram::lookupSubmissionRetirement(),host,gpu);
        for(std::size_t i=0;i<cases.size();++i)
            for(const auto& bom:ordering.resources())joint.add(bom);
        const auto ioBOM=io.bom(host);
        for(const auto& row:ioBOM.resources())joint.add(row.resource(),PhysicalMemoryOwner::RetainedHostServiceStorage,row.bytes(PhysicalMemoryOwner::RetainedHostServiceStorage));
        for(std::size_t i=0;i<cases.size();++i)for(const auto& row:ioBOM.resources())joint.add(row.resource(),PhysicalMemoryOwner::RetainedHostServiceRuntime,row.bytes(PhysicalMemoryOwner::RetainedHostServiceRuntime));
        auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(joint.build()),0);
        std::uint64_t ioOrdinal=0;
        for (const auto tier : {ImmutableRowStorage::RAM, ImmutableRowStorage::MemoryMapped}) {
            SCOPED_TRACE(::testing::Message() << device.toString() << " tier=" << static_cast<int>(tier));
            const auto layout=ImmutableLookupLayout::create(maxIds,Source::rowBytes);
            const auto policy=ImmutableRowTableStoragePolicy::forStorage(tier,4096,2);
            {
            // Keep the original ledger outside the complete captured-owner scope
            // so retirement is observed after every copied source binding dies.
            auto table = HostImmutableRowProvider::fromWholeTable(std::make_unique<ImmutableRowTable>(
                source.descriptor(), policy, DeviceId::cpu(), memory, ImmutableRowTableHostPlacement::taskPolicy()));
            const RetainedHostServiceInstance ioKey{0,device,RetainedHostServiceRole::ImmutableLookupIO,ioOrdinal++};RetainedHostServiceDeclarations::Builder ioDeclaration(0);ioDeclaration.add(ioKey,io);
            const auto declared=std::move(ioDeclaration).freeze();auto ioBinding=declared->bind(memory,std::span{&ioKey,std::size_t{1}});
            auto service = ImmutableLookupService::create(table, layout, std::array{device}, memory, DeviceId::cpu(),ioBinding->take(ioKey));
            auto protocol = std::make_shared<INT32Tensor>(std::vector<std::size_t>{16});
            test::GraphArenaTestHarness arena;
            require(arena.arena().registerExternalBuffer(BufferId::QFN_NGRAM_LOOKUP_STATE, protocol.get()),
                "Early lookup protocol registration failed");
            const auto integer = [&](BufferId role, std::size_t count) {
                return arena.createPersistentTensor<INT32Tensor>(role, std::vector<std::size_t>{count});
            };
            const auto floating = [&](BufferId role, std::size_t width) {
                return arena.createPersistentTensor<FP32Tensor>(role,
                    std::vector<std::size_t>{capacity, width});
            };
            auto *tokens = integer(BufferId::REQUEST_TOKEN_IDS, capacity);
            auto *prior = integer(BufferId::QFN_PLE_PRIOR_TOKEN_HISTORY, 2);
            auto *count = integer(BufferId::PREFILL_CHUNK_GEOMETRY, 1);
            auto *ids = integer(BufferId::QFN_NGRAM_ROW_IDS, maxIds * 2);
            auto *native = integer(BufferId::QFN_NGRAM_NATIVE_ROWS, maxIds * Source::rowBytes / 4);
            auto *output = floating(BufferId::QFN_PLE_EMBEDDING, 16 * hidden);
            auto *input = floating(BufferId::QFN_EMBEDDING_ROWS, hidden);
            auto *trunk = floating(BufferId::HIDDEN_STATE, 4 * hidden);
            auto *control = integer(BufferId::MTP_GENERATION_CONTROL, sampling_math::kDeviceGenerationControlCount);
            const std::array<TensorBase *, 10> roots{protocol.get(), tokens, prior, count, ids,
                native, output, input, trunk, control};
            const auto stream = ExplicitGPUStream(worker.defaultStream()).get();
            for (auto *root : roots) {
                std::memset(root->raw_mutable_data(), 0, root->size_bytes());
                TransferEngine::prepareDeviceInput(root, device, stream);
            }
            auto context = IDeviceContext::create(device, 1);
            std::unique_ptr<IImmutableLookupKernels> kernels;
#ifdef HAVE_CUDA
            if (device.is_cuda()) kernels = createCUDAImmutableLookupKernels(device, TensorType::IQ4_NL,
                hidden, Source::rowBytes);
#endif
#ifdef HAVE_ROCM
            if (device.is_rocm()) kernels = createROCmImmutableLookupKernels(device, TensorType::IQ4_NL,
                hidden, Source::rowBytes);
#endif
            auto boundary = std::make_shared<ImmutableLookupExecutionBoundary>(service, device,
                std::move(kernels), protocol, memory);
            auto *controller = static_cast<int *>(control->gpu_data_ptr());
            boundary->bindGenerationControl({controller + sampling_math::kDeviceGenerationControlOk,
                controller + sampling_math::kDeviceGenerationControlErrorCode});
            using Int = Qwen4ExpIntBinding;
            using Float = Qwen4ExpFloatBinding;
            const Qwen4ExpStageRows rows{capacity, Int(*count, BufferId::PREFILL_CHUNK_GEOMETRY)};
            NgramHashPolicy hash;
            hash.multipliers = {9007199254740993ULL, 9007199254740995ULL, 9007199254740997ULL};
            for (unsigned head=0; head<16; ++head) { hash.moduli[head]=7; hash.offsets[head]=head*7; }
            const Qwen4ExpPLEContext declaration{.device=device, .prefix="early_", .rows=rows,
                .hidden=16*hidden, .epsilon=1e-6f, .hash=hash,
                .nativeFormat=TensorType::IQ4_NL, .nativeRowElements=hidden, .nativeRowBytes=Source::rowBytes};
            const ImmutableLookupStage::GPUStorage sourceBinding{boundary,
                Int(*protocol, BufferId::QFN_NGRAM_LOOKUP_STATE), Int(*native, BufferId::QFN_NGRAM_NATIVE_ROWS)};
            ComputeGraph graph;
            auto frame = Qwen4ExpEarlyLookupFrame::publish(graph, declaration,
                Int(*tokens, BufferId::REQUEST_TOKEN_IDS), Int(*prior, BufferId::QFN_PLE_PRIOR_TOKEN_HISTORY),
                Int(*ids, BufferId::QFN_NGRAM_ROW_IDS), Float(*output, BufferId::QFN_PLE_EMBEDDING), sourceBinding);
            const auto hashIdentity = graph.getNode("early_hash")->stage->graphCaptureVariantSignature();
            graph.addNode("trunk", std::make_unique<Qwen4ExpPrimitiveStage>(Qwen4ExpPrimitiveStage::Params{
                .device_id=device, .rows=rows, .operation=Qwen4ExpPrimitiveStage::ResidualExpand{
                    hidden, Float(*input, BufferId::QFN_EMBEDDING_ROWS), Float(*trunk, BufferId::HIDDEN_STATE)}}), device);
            graph.addDependency("trunk", frame->producer());
            auto consumed = frame->consume(graph, ImmutableLookupStage::Params{
                .device_id=device, .rows=rows, .ids=Int(*ids, BufferId::QFN_NGRAM_ROW_IDS),
                .output=Float(*output, BufferId::QFN_PLE_EMBEDDING), .format=TensorType::IQ4_NL,
                .rowElements=hidden, .rowBytes=Source::rowBytes, .storage=sourceBinding}, hashIdentity, {"trunk"});
            graph.addNode("ple_gather", std::move(consumed.stage), device);
            for (const auto &dependency : consumed.dependencies) graph.addDependency("ple_gather", dependency);
            frame->requireConsumed();
            EXPECT_EQ(graph.getExecutionOrder(), (std::vector<std::string>{"early_hash", "early_publish", "trunk", "ple_gather"}));
            DeviceGraphExecutor executor;
            arena.bindExecutor(executor);
            const auto stages = graph.getExecutionStages();
            for (auto *stage : stages) require(stage->prepareGraphLaunch(context.get(), stream),
                "Early lookup stage preparation failed");
            require(executor.prepareGraphStorageForCapture(graph, context.get(), stream, "early_hosted_native"),
                "Early lookup original arena preparation failed");
            const auto order = graph.getExecutionOrder();
            auto dependencies = executor.planGraphCaptureDependencies(
                graph, order, device, stream, "early_hosted_native_dependencies");
            require(bool(dependencies), "Early lookup capture dependency plan failed");
            Recording recording{*backend, device, stream, worker.createGraphCapture(stream)};
            {
                // The production recorder advances the typed dependency ledger:
                // Publish reads the hash produced inside this same executable,
                // never the fixture's earlier initialization event for IDs.
                ScopedBackendGraphCapture capture(worker, *recording.capture,
                    "early current hosted PLE lookup", dependencies.get());
                require(capture.begin(), "Early lookup capture failed");
                require(executor.recordSetupGraphMaterialization(graph, context.get()),
                    "Early lookup production graph recording failed");
                boundary->recordTerminalClose(stream);
                capture.finish();
            }
            std::vector<GPUGraphKernelNodeInfo> nodes;
            std::string diagnostic;
            require(recording.capture->inspectKernelNodes(nodes, &diagnostic), diagnostic.c_str());
            for (const auto &node : nodes) EXPECT_EQ(node.local_memory_bytes_per_thread, 0u);
            require(recording.capture->instantiate(), "Early lookup instantiate failed");
            const auto *executable = recording.capture.get();
            std::vector<std::uint64_t> identities;
            for (auto *stage : stages) identities.push_back(stage->graphCaptureVariantSignature());
            std::vector<const void *> addresses;
            for (auto *root : roots) addresses.push_back(root->gpu_data_ptr());
            std::uint64_t idCount=0;
            for (unsigned replay=0; replay<20; ++replay) {
                SCOPED_TRACE(replay);
                const int live = liveSequence[replay % liveSequence.size()];
                auto *tokenValues = static_cast<std::int32_t *>(tokens->raw_mutable_data());
                for (int row=0; row<capacity; ++row) tokenValues[row] = row==2 ? hash.eosToken : 17+row+31*replay;
                auto *priorValues = static_cast<std::int32_t *>(prior->raw_mutable_data());
                priorValues[0]=7+replay; priorValues[1]=11+replay;
                *static_cast<int *>(count->raw_mutable_data())=live;
                for (std::size_t i=0; i<input->numel(); ++i)
                    input->mutable_data()[i]=static_cast<float>(int((i*13+replay)%67)-33)/41.f;
                auto *controlValues=static_cast<int *>(control->raw_mutable_data());
                std::fill_n(controlValues, sampling_math::kDeviceGenerationControlCount, 0);
                controlValues[sampling_math::kDeviceGenerationControlOk]=1;
                for (auto *root : std::array<TensorBase *,5>{tokens, prior, count, input, control}) {
                    TransferEngine::publishHostWrite(root);
                    TransferEngine::prepareDeviceInput(root, device, stream);
                }
                for (auto *root : {output, trunk}) {
                    std::fill_n(root->mutable_data(), root->numel(), poison);
                    TransferEngine::publishHostWrite(root);
                    TransferEngine::prepareDeviceInput(root, device, stream);
                }
                require(backend->memset(ids->gpu_data_ptr(), 0xa5, ids->size_bytes(), device.ordinal, stream),
                    "Early lookup ID poison failed");
                require(backend->memset(native->gpu_data_ptr(), 0xa5, native->size_bytes(), device.ordinal, stream),
                    "Early lookup native poison failed");
                boundary->beginAdmission(replay+1, stream);
                boundary->requireArmedForLaunch();
                require(recording.capture->launch(), "Early lookup retained launch failed");
                const auto actual=recording.read<float>(output->gpu_data_ptr(), output->numel());
                boundary->awaitTerminal();
                const auto observation=boundary->terminalObservation();
                const std::size_t liveIds=live*16;
                std::vector<std::int64_t> expectedIds(maxIds, idPoison);
                cpu->ngramHash({{capacity, &live}, tokenValues, priorValues, expectedIds.data(), hash}, nullptr);
                EXPECT_EQ(recording.read<std::int64_t>(ids->gpu_data_ptr(), maxIds), expectedIds);
                std::vector<std::byte> packed(maxIds*Source::rowBytes, std::byte{0xa5});
                for (std::size_t i=0; i<liveIds; ++i)
                    std::memcpy(packed.data()+i*Source::rowBytes, &source.blocks[expectedIds[i]], Source::rowBytes);
                std::vector<float> expected(output->numel(), poison);
                decodeCPUImmutableLookupRows(TensorType::IQ4_NL, hidden, Source::rowBytes,
                    packed.data(), liveIds, expected.data());
                EXPECT_EQ(std::memcmp(actual.data(), expected.data(), expected.size()*sizeof(float)), 0);
                EXPECT_EQ(recording.read<std::byte>(native->gpu_data_ptr(), packed.size()), packed);
                std::vector<float> expectedTrunk(trunk->numel(), poison);
                for (int row=0; row<live; ++row) for (int branch=0; branch<4; ++branch)
                    std::copy_n(input->data()+row*hidden, hidden, expectedTrunk.data()+(row*4+branch)*hidden);
                EXPECT_EQ(recording.read<float>(trunk->gpu_data_ptr(), trunk->numel()), expectedTrunk);
                EXPECT_EQ(recording.read<int>(controller+sampling_math::kDeviceGenerationControlErrorCode, 1)[0], 0);
                EXPECT_EQ(observation.generation, replay+1);
                EXPECT_EQ(observation.counters.requests, 1u);
                EXPECT_EQ(observation.counters.idBytes, liveIds*sizeof(std::int64_t));
                EXPECT_EQ(observation.counters.nativeBytes, liveIds*Source::rowBytes);
                EXPECT_EQ(observation.counters.terminalBytes, 3u*sizeof(std::uint64_t));
                EXPECT_EQ(recording.capture.get(), executable);
                for (std::size_t i=0; i<stages.size(); ++i)
                    EXPECT_EQ(stages[i]->graphCaptureVariantSignature(), identities[i]);
                std::size_t index=0;
                for (auto *root : roots) EXPECT_EQ(root->gpu_data_ptr(), addresses[index++]);
                idCount+=liveIds;
            }
            const auto stats=service->stats();
            EXPECT_EQ(stats.requests, 20u);
            EXPECT_EQ(stats.ids, idCount);
            EXPECT_EQ(stats.idBytes, idCount*sizeof(std::int64_t));
            EXPECT_EQ(stats.nativeBytes, idCount*Source::rowBytes);
            EXPECT_EQ(stats.metadataBytes, 20u*(kLookupRequestLiveBytes+kLookupCompletionLiveBytes));
            EXPECT_EQ(stats.acknowledgementBytes, 20u*sizeof(std::uint64_t));
            EXPECT_EQ(stats.failures, 0u);
            require(backend->synchronizeStream(stream, device.ordinal), "Early lookup retirement failed");
            recording.capture.reset();
            graph.clear();
            frame.reset();
            // sourceBinding retains the exact same boundary until scope exit;
            // no assertion pretends that a copied descriptor has released it.
            EXPECT_FALSE(boundary->hasActiveInterval());
            }
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ImmutableLookupStaging,
                PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ImmutableTableStorage,
                PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        }
    });
}
} // namespace

#ifdef HAVE_CUDA
/** @test CUDA retains exact current lookups across RAM/mmap and changing request epochs. */
TEST(Qwen4ExpEarlyHostedLookup, CUDAReplay) { proveEarlyHostedLookup(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
/** @test HIP retains exact current lookups across RAM/mmap and changing request epochs. */
TEST(Qwen4ExpEarlyHostedLookup, ROCmReplay) { proveEarlyHostedLookup(DeviceId::rocm(0)); }
#endif
