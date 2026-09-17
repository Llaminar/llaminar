/**
 * @file Test__EmbeddingWeightResidency.cpp
 * @brief Native embedding preparation must satisfy its captured raw-read contract.
 *
 * Kernel-only tests upload their own source, masking a production loader that
 * incorrectly marks floating embeddings HOST_RESIDENT. These tests use real
 * WeightManager preload, immutable bindings and preparation before recording
 * EmbeddingStage. No test-side upload of model weights is permitted here.
 */
#include "backends/GPUDeviceContextPool.h"
#include "execution/compute_stages/stages/EmbeddingStage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/WeightManager.h"
#include "loaders/GPUVramPreflight.h"
#include "models/qwen/Qwen2Schema.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/WeightMemoryEstimator.h"
#include "mocks/MockModelLoader.h"
#include "transfer/TransferEngine.h"
#include "utils/EmbeddingVerifierFormats.h"
#include "utils/PlanningGGUFFixture.h"
#include "utils/ScopedGPUStream.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>

namespace llaminar2::test
{
namespace
{
/** @brief Own a tiny real GGUF so native TP slices use the production loader. */
class EmbeddingGGUF
{
public:
    /** @brief Serialize exact source bytes in GGUF's column-first directory order. */
    explicit EmbeddingGGUF(const TensorBase &source)
    {
        char pattern[] = "/tmp/llaminar-embedding-residency-XXXXXX.gguf";
        const int fd = ::mkstemps(pattern, 5);
        if (fd < 0) throw std::runtime_error("Cannot create embedding fixture");
        ::close(fd);
        path_ = pattern;
        std::ofstream out(path_, std::ios::binary | std::ios::trunc);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        try
        {
            out.write("GGUF", 4);
            scalar<uint32_t>(out, 3);
            scalar<uint64_t>(out, 1);
            scalar<uint64_t>(out, 0);
            const std::string name = "token_embd.weight";
            scalar<uint64_t>(out, name.size());
            out.write(name.data(), name.size());
            scalar<uint32_t>(out, 2);
            scalar<uint64_t>(out, source.cols());
            scalar<uint64_t>(out, source.rows());
            scalar<uint32_t>(out, static_cast<uint32_t>(PlanningGGUFFixture::sourceType(source.native_type())));
            scalar<uint64_t>(out, 0);
            while (static_cast<size_t>(out.tellp()) % 32u) out.put(0);
            out.write(static_cast<const char *>(source.raw_data()), source.size_bytes());
        }
        catch (...) { remove(); throw; }
    }
    /** @brief Remove only this fixture's independently named temporary file. */
    ~EmbeddingGGUF() { remove(); }
    EmbeddingGGUF(const EmbeddingGGUF &) = delete;
    EmbeddingGGUF &operator=(const EmbeddingGGUF &) = delete;
    /** @return Source path valid until the fixture is destroyed. */
    const std::string &path() const noexcept { return path_; }
private:
    /** @brief Write a GGUF integer independently of the host's byte order. */
    template<class T> static void scalar(std::ostream &out, T value)
    {
        for (size_t byte = 0; byte < sizeof(T); ++byte)
            out.put(static_cast<char>(value >> (8u * byte)));
    }
    /** @brief Nonthrowing cleanup, also used when construction fails. */
    void remove() noexcept
    {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }
    std::string path_;
};

/**
 * @brief Admit the fixture's actual native lookup/head views through the canonical BOM.
 * @param source Complete vocabulary source, shared by the two semantic uses.
 * @param devices Exact physical owners that the fixture will prepare.
 * @param sharded Whether lookup is split; the TP output head follows its schema.
 * @return One topology-wide authority retained by WeightManager during preparation.
 */
std::shared_ptr<PhysicalMemoryAuthority> vocabularyMemory(
    const TensorBase &source, const std::vector<DeviceId> &devices, bool sharded)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen2";
    profile.n_layers = 1;
    profile.d_model = source.cols();
    profile.vocab_size = source.rows();
    const std::string format = source.native_type() == TensorType::FP32 ? "F32" :
                               source.native_type() == TensorType::FP16 ? "F16" : "BF16";
    profile.tensors.push_back({"token_embd.weight", source.size_bytes(), format,
                              source.numel(), source.cols(), -1});
    const PhysicalMemoryResource host{.world_rank = 0, .device = DeviceId::cpu(),
        .total_bytes = 256u << 20, .admission_available_bytes = 256u << 20};
    PhysicalMemoryPlanBuilder builder;
    for (size_t index = 0; index < devices.size(); ++index)
    {
        const auto lookup = WeightMemoryEstimator::estimate(profile, devices[index],
            sharded ? static_cast<int>(index) : 0, sharded ? static_cast<int>(devices.size()) : 1,
            0, -1, {}, {}, WeightComponentScope::Embedding);
        const auto terminal = WeightMemoryEstimator::estimate(profile, devices[index],
            static_cast<int>(index), static_cast<int>(devices.size()),
            0, -1, {}, {}, WeightComponentScope::Terminal);
        const auto staging = resolveGPUWeightLoadMemoryGeometry(
            std::max(lookup.native_bytes, terminal.native_bytes), configuredGPUWeightLoadMemoryPolicy());
        const PhysicalMemoryResource gpu{.world_rank = 0, .device = devices[index],
            .total_bytes = 256u << 20, .admission_available_bytes = 256u << 20};
        builder.add(gpu, PhysicalMemoryOwner::PrimaryModelWeights, lookup.device_bytes)
            .add(gpu, PhysicalMemoryOwner::PrimaryModelWeights, terminal.device_bytes)
            .add(gpu, PhysicalMemoryOwner::WeightLoadStaging, staging.staging_bytes)
            .add(host, PhysicalMemoryOwner::WeightLoadStaging, staging.host_staging_bytes);
    }
    return std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
}

/**
 * @brief Prove replicated and vocabulary-sharded preparation on real devices.
 *
 * One device covers singleton execution; two, when available, also prove that
 * native floating operands retain independent physical device ownership.
 */
void proveNativeEmbeddingResidency(DeviceId first_device)
{
    constexpr int vocab = 64, hidden = 256, rows = 3;
    constexpr std::array<int32_t, rows> token_ids{0, 31, 63};
    auto *backend = getBackendFor(first_device);
    ASSERT_NE(backend, nullptr);
    ASSERT_GT(backend->deviceCount(), 0);
    for (int degree = 1; degree <= std::min(2, backend->deviceCount()); ++degree)
    {
        SCOPED_TRACE(degree);
        std::vector<DeviceId> devices;
        for (int ordinal = 0; ordinal < degree; ++ordinal)
            devices.emplace_back(first_device.type, ordinal);
        for (bool sharded : {false, true})
        {
            SCOPED_TRACE(sharded);
            for (const auto &format : embeddingVerifierFormats())
            {
                if (format.prepared_embed_q8)
                    continue; // Prepared EmbedQ8 has its own all-codebook captured gate.
                SCOPED_TRACE(format.label);
                auto source = format.create({vocab, hidden}, 192017);
                std::vector<float> reference(vocab * hidden);
                source->to_fp32(reference.data());
                EmbeddingGGUF fixture(*source);
                ModelLoader loader;
                ASSERT_TRUE(loader.loadModel(fixture.path()));
                WeightManager manager(loader);
                auto config = Qwen2SchemaFactory{}.getWeightShardingConfig();
                config.exact_matches["token_embd.weight"] = sharded
                    ? WeightShardingMode::ColumnParallel : WeightShardingMode::Replicate;
                manager.setWeightShardingConfig(config);
                manager.setTensorParallelConfig(std::make_shared<TensorParallelConfig>(
                    TensorParallelConfig::equalSplit(degree, 4, 2, 512, vocab, devices)));
                const ModelContextId model_id{192017};
                manager.setPreparedWeightStore(std::make_shared<PreparedWeightStore>(model_id));
                manager.installPhysicalMemoryAuthority(vocabularyMemory(*source, devices, sharded));
                ASSERT_TRUE(manager.preloadForDevices(devices));

                TensorBase *previous = nullptr;
                for (int index = 0; index < degree; ++index)
                {
                    const auto device = devices[index];
                    SCOPED_TRACE(device.toString());
                    ScopedGPUStream stream(device);
                    InferenceStrategy strategy;
                    strategy.model_id = model_id;
                    strategy.mode = WeightInferenceMode::LocalTP;
                    strategy.tp_degree = degree;
                    strategy.devices = devices;
                    WeightPlan plan(strategy);
                    WeightRequirement requirement;
                    requirement.canonical_name = "token_embd.weight";
                    requirement.target_device = device;
                    requirement.lookup_device = device;
                    requirement.tp_rank_or_device_index = index;
                    requirement.host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized;
                    plan.add(requirement);
                    // This GGUF has no output.weight. The canonical tied alias
                    // must still own a GEMM representation beside the lookup
                    // allocation; native float does not make those GPU uses
                    // physically identical.
                    auto head_requirement = requirement;
                    head_requirement.canonical_name = "output.weight";
                    // TP loading resolves this virtual canonical name to the
                    // source vocabulary using the output head's shard geometry.
                    head_requirement.source_name = "output.weight";
                    head_requirement.role = WeightRole::LMHead;
                    head_requirement.derivation = WeightDerivationKind::TiedAlias;
                    plan.add(head_requirement);
                    auto frozen = manager.materialize(plan);
                    ASSERT_TRUE(manager.prepareWeightsForDevice(frozen, device, false));
                    auto *weight = frozen.global("token_embd.weight").tensor;
                    ASSERT_NE(weight, nullptr);
                    ASSERT_NE(weight, previous);
                    previous = weight;
                    ASSERT_FALSE(weight->isHostResident());
                    ASSERT_TRUE(weight->is_on_device(device));
                    ASSERT_NE(weight->gpu_data_ptr(), nullptr);
                    const auto &head_binding = frozen.global("output.weight");
                    const auto head_ref = manager.preparedWeightStore()->preparedRefForBinding(
                        head_binding.binding_id, device);
                    ASSERT_TRUE(head_ref);
                    auto *head_kernel = manager.preparedWeightStore()->gemmKernel(*head_ref);
                    ASSERT_NE(head_kernel, nullptr);
                    ContiguousFloatingPointWeightDescriptor head_storage;
                    ASSERT_TRUE(head_kernel->exportContiguousFloatingPointWeights(head_storage));
                    ASSERT_TRUE(head_storage.valid());
                    ASSERT_EQ(head_storage.type, source->native_type());
                    EXPECT_NE(head_storage.data, weight->gpu_data_ptr());
                    EXPECT_EQ(head_storage.bytes,
                        static_cast<size_t>(head_storage.n) * head_storage.k *
                            (source->native_type() == TensorType::FP32 ? 4u : 2u));

                    // Only request tokens and outputs are allocated by this fixture.
                    // The model weight above must arrive through production setup.
                    auto tokens = TestTensorFactory::createINT32({rows});
                    std::copy(token_ids.begin(), token_ids.end(), tokens->mutable_int32_data());
                    ASSERT_TRUE(tokens->ensureOnDevice(device, stream.get()));
                    auto output = TestTensorFactory::createFP32({rows, hidden});
                    ASSERT_TRUE(output->allocateOnDevice(device, stream.get()));
                    EmbeddingStage::Params params;
                    params.device_id = device;
                    params.embed_table = weight;
                    params.token_ids_device = tokens->gpu_data_ptr();
                    params.output = output.get();
                    params.num_tokens = rows;
                    params.d_model = hidden;
                    params.vocab_size = vocab;
                    params.vocab_offset = sharded ? index * (vocab / degree) : 0;
                    params.local_vocab_size = sharded ? vocab / degree : vocab;
                    params.output_buffer_id = BufferId::HIDDEN_STATE;
                    EmbeddingStage stage(params);
                    stage.setGPUStream(stream.get());
                    const auto workspace_requirements = stage.getWorkspaceRequirements(rows, hidden);
                    DeviceWorkspaceManager workspace(device, workspace_requirements.total_bytes_with_alignment());
                    ASSERT_TRUE(workspace.allocate(workspace_requirements));
                    stage.bindWorkspace(&workspace);
                    ASSERT_TRUE(stage.validatePreparedWeights(nullptr));

                    const auto contract = stage.bufferContract();
                    ASSERT_EQ(contract.weight_tensors.size(), 1u);
                    // This is the same exact-device frontier check that rejected
                    // the real BF16 server before recording its prefill graph.
                    ASSERT_NO_THROW(TransferEngine::requireDeviceInput(
                        contract.weight_tensors.front(), device, stream.get()));
                    auto &context = GPUDeviceContextPool::instance().getContext(device);
                    auto graph = context.createGraphCapture(stream.get());
                    {
                        ScopedBackendGraphCapture capture(context, *graph, "native_embedding_residency");
                        ASSERT_TRUE(capture.begin());
                        const bool success = stage.execute(nullptr);
                        capture.finish();
                        ASSERT_TRUE(success);
                    }
                    ASSERT_TRUE(graph->instantiate());
                    std::vector<float> actual(rows * hidden), expected(rows * hidden, 0.f);
                    for (int row = 0; row < rows; ++row)
                        if (token_ids[row] >= params.vocab_offset &&
                            token_ids[row] < params.vocab_offset + params.local_vocab_size)
                            std::copy_n(reference.data() + token_ids[row] * hidden, hidden,
                                        expected.data() + row * hidden);
                    for (int replay = 0; replay < 2; ++replay)
                    {
                        ASSERT_TRUE(graph->launch());
                        ASSERT_TRUE(backend->deviceToHost(actual.data(), output->gpu_data_ptr(),
                            actual.size() * sizeof(float), device.ordinal, stream.get()));
                        ASSERT_EQ(std::memcmp(actual.data(), expected.data(),
                            actual.size() * sizeof(float)), 0);
                    }
                    stage.unbindWorkspace();
                    stage.resetSessionState();
                }
            }
        }
    }

    // A stale/misclassified host-only raw binding must fail at preparation,
    // never be reported ready and discovered only by a later capture frontier.
    for (const auto &format : embeddingVerifierFormats())
    {
        if (format.prepared_embed_q8)
            continue;
        SCOPED_TRACE(format.label);
        auto source = format.create({vocab, hidden}, 192018);
        source->setHostResident();
        MockModelLoader loader;
        loader.addTensor("token_embd.weight", std::move(source));
        WeightManager manager(loader);
        manager.setWeightShardingConfig(Qwen2SchemaFactory{}.getWeightShardingConfig());
        const ModelContextId model_id{192018};
        manager.setPreparedWeightStore(std::make_shared<PreparedWeightStore>(model_id));
        InferenceStrategy strategy;
        strategy.model_id = model_id;
        strategy.devices = {first_device};
        WeightPlan plan(strategy);
        WeightRequirement requirement;
        requirement.canonical_name = "token_embd.weight";
        requirement.target_device = first_device;
        plan.add(requirement);
        auto frozen = manager.materialize(plan);
        ASSERT_TRUE(frozen.global("token_embd.weight").tensor->isHostResident());
        EXPECT_THROW(manager.prepareWeightsForDevice(frozen, first_device, false), std::runtime_error);
    }
}

#ifdef HAVE_CUDA
/** @test CUDA native embedding ownership survives production preparation and capture. */
TEST(EmbeddingWeightResidency, CUDA) { proveNativeEmbeddingResidency(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
/** @test ROCm has the same raw-source contract, including BF16 vocabulary shards. */
TEST(EmbeddingWeightResidency, ROCm) { proveNativeEmbeddingResidency(DeviceId::rocm(0)); }
#endif
} // namespace
} // namespace llaminar2::test
