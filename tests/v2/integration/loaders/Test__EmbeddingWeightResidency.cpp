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
#include "utils/PerfStatsCollector.h"
#include "utils/DebugEnv.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>
#include <future>
#include <latch>

namespace llaminar2::test
{
namespace
{
/** @brief Own a tiny real GGUF so native TP slices use the production loader. */
class EmbeddingGGUF
{
public:
    /**
     * @brief Serialize exact source bytes in GGUF's column-first directory order.
     * @param source Original native vocabulary payload.
     * @param include_output Add a distinct output projection to exercise the GEMM pipeline.
     */
    explicit EmbeddingGGUF(const TensorBase &source, bool include_output = false)
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
            scalar<uint64_t>(out, include_output ? 2 : 1);
            scalar<uint64_t>(out, 0);
            const size_t output_offset = (source.size_bytes() + 31u) & ~size_t{31u};
            for (const auto &name : include_output
                     ? std::vector<std::string>{"token_embd.weight", "output.weight"}
                     : std::vector<std::string>{"token_embd.weight"})
            {
                scalar<uint64_t>(out, name.size());
                out.write(name.data(), name.size());
                scalar<uint32_t>(out, 2);
                scalar<uint64_t>(out, source.cols());
                scalar<uint64_t>(out, source.rows());
                scalar<uint32_t>(out, static_cast<uint32_t>(PlanningGGUFFixture::sourceType(source.native_type())));
                scalar<uint64_t>(out, name == "output.weight" ? output_offset : 0);
            }
            while (static_cast<size_t>(out.tellp()) % 32u) out.put(0);
            out.write(static_cast<const char *>(source.raw_data()), source.size_bytes());
            if (include_output)
            {
                for (size_t pad = source.size_bytes(); pad < output_offset; ++pad) out.put(0);
                out.write(static_cast<const char *>(source.raw_data()), source.size_bytes());
            }
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
                               source.native_type() == TensorType::FP16 ? "F16" :
                               source.native_type() == TensorType::BF16 ? "BF16" :
                               tensorTypeName(source.native_type());
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
                if (format.prepared_native_quantized)
                    continue; // Prepared native quantized has its own all-codebook captured gate.
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
        if (format.prepared_native_quantized)
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


/** @brief Isolate setup geometry and the exact structured allocation evidence. */
class ScopedReadOnlyUploadSettings final
{
public:
    /** @brief Force a small admitted slot and collect only this setup's counters. */
    explicit ScopedReadOnlyUploadSettings(int streams = 8)
        : old_budget_(debugEnv().rocm.repack_budget_mb),
          old_streams_(debugEnv().rocm.repack_streams)
    {
        for (const char *name : {"LLAMINAR_PERF_STATS_SUMMARY", "LLAMINAR_PERF_STATS_FILTER"})
        {
            const char *value = std::getenv(name);
            old_environment_.emplace_back(name, value ? std::optional<std::string>(value) : std::nullopt);
        }
        setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
        setenv("LLAMINAR_PERF_STATS_FILTER", "device_memory,weight_loading", 1);
        PerfStatsCollector::reloadConfigurationFromEnvironment();
        mutableDebugEnv().rocm.repack_budget_mb = 1;
        mutableDebugEnv().rocm.repack_streams = streams;
    }
    /** @brief Restore process policy after every assertion/exception path. */
    ~ScopedReadOnlyUploadSettings()
    {
        for (const auto &[name, value] : old_environment_)
            if (value) setenv(name.c_str(), value->c_str(), 1);
            else unsetenv(name.c_str());
        mutableDebugEnv().rocm.repack_budget_mb = old_budget_;
        mutableDebugEnv().rocm.repack_streams = old_streams_;
        PerfStatsCollector::reloadConfigurationFromEnvironment();
        PerfStatsCollector::reset();
    }
private:
    int old_budget_;
    int old_streams_;
    std::vector<std::pair<std::string, std::optional<std::string>>> old_environment_;
};

/**
 * @brief Upload a real read-only GGUF through one admitted, reusable pinned slot.
 * @param device Exact backend whose production setup/captured lookup is checked.
 * @details The byte cap deliberately cuts native Q6_K blocks and rows. The
 * destination is published only after every exact tail byte has completed.
 */
void proveReadOnlyNativeUpload(DeviceId device)
{
    constexpr int vocab = 2049, hidden = 256, rows = 4;
    constexpr std::array<int32_t, rows> ids{0, 624, 1248, vocab - 1};
    ScopedReadOnlyUploadSettings settings;
    auto *backend = getBackendFor(device);
    ASSERT_NE(backend, nullptr);
    ASSERT_GT(backend->deviceCount(), device.ordinal);
    ScopedGPUStream stream(device);
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    ASSERT_NE(context.defaultStream(), nullptr);
    auto original = TestTensorFactory::createQ6_KRandom({vocab, hidden}, 193702);
    EmbeddingGGUF fixture(*original);
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(fixture.path()));
    auto source = loader.loadTensor("token_embd.weight");
    ASSERT_NE(source, nullptr);
    ASSERT_TRUE(source->is_mmap_data());
    ASSERT_GT(source->size_bytes(), 64u * 1024u);
    const std::vector<uint8_t> original_bytes(
        static_cast<const uint8_t *>(original->raw_data()),
        static_cast<const uint8_t *>(original->raw_data()) + original->size_bytes());
    ASSERT_EQ(source->size_bytes(), original_bytes.size());
    ASSERT_EQ(std::memcmp(source->raw_data(), original_bytes.data(), original_bytes.size()), 0);
    std::vector<float> source_values(vocab * hidden);
    original->to_fp32(source_values.data());
    const auto geometry = resolveGPUWeightLoadMemoryGeometry(
        source->size_bytes(), configuredGPUWeightLoadMemoryPolicy());
    ASSERT_GT(source->size_bytes(), 3u * geometry.staging_slot_bytes);
    ASSERT_NE(source->size_bytes() % geometry.staging_slot_bytes, 0u);
    ASSERT_NE(geometry.staging_slot_bytes % sizeof(Q6_KBlock), 0u);
    auto authority = vocabularyMemory(*source, {device}, false);
    PreparedWeightStore store(ModelContextId{193702});
    store.installPhysicalMemoryAuthority(authority);
    WeightBinding binding;
    binding.binding_id = 1;
    binding.identity.model_id = ModelContextId{193702};
    binding.identity.logical_id = binding.identity.instance_id = 1;
    binding.identity.canonical_name = "token_embd.weight";
    binding.identity.role = WeightRole::Embedding;
    binding.identity.derivation = WeightDerivationKind::Source;
    binding.residency.home_device = device;
    binding.residency.resident_device = device;
    binding.tensor = source.get();
    PerfStatsCollector::reset();
    const auto ref = store.prepareEmbedding(binding, hidden);
    const auto records = PerfStatsCollector::snapshot();
    const auto counter = [&](const char *domain, const char *name)
    {
        double total = 0;
        for (const auto &record : records)
            if (record.domain == domain && record.name == name && record.device == device.toString())
                total += record.value;
        return total;
    };
    EXPECT_EQ(counter("device_memory", "pinned_host_allocations"), 1.0);
    EXPECT_EQ(counter("device_memory", "pinned_host_allocation_bytes"),
              static_cast<double>(geometry.staging_slot_bytes));
    EXPECT_EQ(counter("weight_loading", "native_embedding_pinned_upload_bytes"),
              static_cast<double>(source->size_bytes()));
    uint64_t chunks = 0;
    for (const auto &record : records)
        if (record.domain == "weight_loading" && record.name == "native_embedding_pinned_upload_bytes")
            chunks += record.count;
    EXPECT_EQ(chunks, (source->size_bytes() + geometry.staging_slot_bytes - 1) / geometry.staging_slot_bytes);
    EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::WeightLoadStaging,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    const auto *handle = store.embeddingHandle(ref);
    ASSERT_NE(handle, nullptr);
    ASSERT_NE(handle->weights, nullptr);
    std::vector<uint8_t> uploaded(original_bytes.size());
    ASSERT_TRUE(backend->deviceToHost(uploaded.data(), handle->weights->device_data,
        uploaded.size(), device.ordinal, stream.get()));
    EXPECT_EQ(uploaded, original_bytes);
    EXPECT_EQ(std::memcmp(source->raw_data(), original_bytes.data(), original_bytes.size()), 0);
    {
        auto tokens = TestTensorFactory::createINT32({rows});
        std::copy(ids.begin(), ids.end(), tokens->mutable_int32_data());
        ASSERT_TRUE(tokens->ensureOnDevice(device, stream.get()));
        auto output = TestTensorFactory::createFP32({rows, hidden});
        ASSERT_TRUE(output->allocateOnDevice(device, stream.get()));
        EmbeddingStage::Params params;
        params.device_id = device;
        params.embed_table = source.get();
        params.token_ids_device = tokens->gpu_data_ptr();
        params.output = output.get();
        params.num_tokens = rows;
        params.d_model = hidden;
        params.vocab_size = params.local_vocab_size = vocab;
        params.output_buffer_id = BufferId::HIDDEN_STATE;
        params.prepared_ref = ref;
        params.prepared_store = &store;
        EmbeddingStage stage(params);
        stage.setGPUStream(stream.get());
        const auto requirements = stage.getWorkspaceRequirements(rows, hidden);
        DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment());
        ASSERT_TRUE(workspace.allocate(requirements));
        stage.bindWorkspace(&workspace);
        ASSERT_TRUE(stage.validatePreparedWeights(nullptr));
        auto graph = context.createGraphCapture(stream.get());
        ASSERT_NE(graph, nullptr);
        {
            ScopedBackendGraphCapture capture(context, *graph, "readonly_native_embedding_upload");
            ASSERT_TRUE(capture.begin());
            const bool success = stage.execute(nullptr);
            capture.finish();
            ASSERT_TRUE(success);
        }
        ASSERT_TRUE(graph->instantiate());
        std::vector<float> actual(rows * hidden), expected(rows * hidden);
        for (int row = 0; row < rows; ++row)
            std::copy_n(source_values.data() + ids[row] * hidden, hidden, expected.data() + row * hidden);
        for (int replay = 0; replay < 2; ++replay)
        {
            ASSERT_TRUE(graph->launch());
            ASSERT_TRUE(backend->deviceToHost(actual.data(), output->gpu_data_ptr(),
                actual.size() * sizeof(float), device.ordinal, stream.get()));
            EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
        }
        stage.unbindWorkspace();
        stage.resetSessionState();
    }
    store.clear();
    EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::PrimaryModelWeights,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
}


/**
 * @brief Admit primary/decode views and only the existing per-device upload ring.
 * @param source Original quantized lookup and explicit output projection payload.
 * @param devices Concurrent TP participants sharing one host admission.
 * @return Exact canonical BOM; embedding staging has no additional allowance.
 */
std::shared_ptr<PhysicalMemoryAuthority> concurrentVocabularyMemory(
    const TensorBase &source, const std::vector<DeviceId> &devices)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen2";
    profile.n_layers = 1;
    profile.d_model = source.cols();
    profile.vocab_size = source.rows();
    for (const char *name : {"token_embd.weight", "output.weight"})
        profile.tensors.push_back({name, source.size_bytes(), tensorTypeName(source.native_type()),
                                  source.numel(), source.cols(), -1});
    const PhysicalMemoryResource host{.world_rank = 0, .device = DeviceId::cpu(),
        .total_bytes = 256u << 20, .admission_available_bytes = 256u << 20};
    PhysicalMemoryPlanBuilder builder;
    for (size_t index = 0; index < devices.size(); ++index)
    {
        const PhysicalMemoryResource gpu{.world_rank = 0, .device = devices[index],
            .total_bytes = 256u << 20, .admission_available_bytes = 256u << 20};
        for (bool decode : {false, true})
        {
            const int rank = decode ? 0 : static_cast<int>(index);
            const int degree = decode ? 1 : static_cast<int>(devices.size());
            const auto lookup = WeightMemoryEstimator::estimate(profile, devices[index], rank, degree,
                0, -1, {}, {}, WeightComponentScope::Embedding);
            const auto head = WeightMemoryEstimator::estimate(profile, devices[index], rank, degree,
                0, -1, {}, {}, WeightComponentScope::Terminal);
            const auto owner = decode ? PhysicalMemoryOwner::AdditionalModelWeights
                                      : PhysicalMemoryOwner::PrimaryModelWeights;
            builder.add(gpu, owner, lookup.device_bytes).add(gpu, owner, head.device_bytes);
        }
        // One admitted ring is reused by this participant's primary/decode
        // transactions. A separate embedding slot may not overlap that ring.
        const auto staging = resolveGPUWeightLoadMemoryGeometry(
            source.size_bytes(), configuredGPUWeightLoadMemoryPolicy());
        builder.add(gpu, PhysicalMemoryOwner::WeightLoadStaging, staging.staging_bytes)
            .add(host, PhysicalMemoryOwner::WeightLoadStaging, staging.host_staging_bytes);
    }
    return std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
}

/** @brief Compare completed prepared bytes and the captured lookup with original rows. */
void verifyConcurrentPreparedLookup(
    PreparedWeightStore &store, const FrozenModelWeightSet &frozen,
    DeviceId device, const std::vector<float> &reference, int vocab, int hidden)
{
    const auto &binding = frozen.global("token_embd.weight");
    const auto ref = store.preparedRefForBinding(binding.binding_id, device, PreparedWeightKind::PreparedEmbedding);
    ASSERT_TRUE(ref);
    const auto *handle = store.embeddingHandle(*ref);
    ASSERT_NE(handle, nullptr);
    auto *backend = getBackendFor(device);
    ScopedGPUStream stream(device);
    std::vector<uint8_t> uploaded(handle->weights->layout.byte_size);
    ASSERT_TRUE(backend->deviceToHost(uploaded.data(), handle->weights->device_data,
        uploaded.size(), device.ordinal, stream.get()));
    ASSERT_EQ(uploaded.size(), binding.tensor->size_bytes());
    EXPECT_EQ(std::memcmp(uploaded.data(), binding.tensor->raw_data(), uploaded.size()), 0);
    const auto &head = frozen.global("output.weight");
    const auto head_ref = store.preparedRefForBinding(head.binding_id, device);
    ASSERT_TRUE(head_ref);
    ASSERT_NE(store.gemmKernel(*head_ref), nullptr);

    constexpr int rows = 4;
    const std::array<int32_t, rows> ids{0, vocab / 2 - 1, vocab / 2, vocab - 1};
    auto tokens = TestTensorFactory::createINT32({rows});
    std::copy(ids.begin(), ids.end(), tokens->mutable_int32_data());
    ASSERT_TRUE(tokens->ensureOnDevice(device, stream.get()));
    auto output = TestTensorFactory::createFP32({rows, static_cast<size_t>(hidden)});
    ASSERT_TRUE(output->allocateOnDevice(device, stream.get()));
    EmbeddingStage::Params params;
    params.device_id = device;
    params.embed_table = binding.tensor;
    params.token_ids_device = tokens->gpu_data_ptr();
    params.output = output.get();
    params.num_tokens = rows;
    params.d_model = hidden;
    params.vocab_size = vocab;
    params.vocab_offset = static_cast<int>(handle->weights->vocab_offset);
    params.local_vocab_size = static_cast<int>(handle->weights->vocab_size);
    params.output_buffer_id = BufferId::HIDDEN_STATE;
    params.prepared_ref = *ref;
    params.prepared_store = &store;
    EmbeddingStage stage(params);
    stage.setGPUStream(stream.get());
    const auto requirements = stage.getWorkspaceRequirements(rows, hidden);
    DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment());
    ASSERT_TRUE(workspace.allocate(requirements));
    stage.bindWorkspace(&workspace);
    ASSERT_TRUE(stage.validatePreparedWeights(nullptr));
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    auto graph = context.createGraphCapture(stream.get());
    ASSERT_NE(graph, nullptr);
    {
        ScopedBackendGraphCapture capture(context, *graph, "concurrent_embedding_load_staging");
        ASSERT_TRUE(capture.begin());
        const bool success = stage.execute(nullptr);
        capture.finish();
        ASSERT_TRUE(success);
    }
    ASSERT_TRUE(graph->instantiate());
    std::vector<float> actual(rows * hidden), expected(rows * hidden, 0.f);
    for (int row = 0; row < rows; ++row)
        if (ids[row] >= params.vocab_offset && ids[row] < params.vocab_offset + params.local_vocab_size)
            std::copy_n(reference.data() + ids[row] * hidden, hidden, expected.data() + row * hidden);
    for (int replay = 0; replay < 2; ++replay)
    {
        ASSERT_TRUE(graph->launch());
        ASSERT_TRUE(backend->deviceToHost(actual.data(), output->gpu_data_ptr(),
            actual.size() * sizeof(float), device.ordinal, stream.get()));
        EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
    }
    stage.unbindWorkspace();
    stage.resetSessionState();
}

/**
 * @brief Exercise the real asynchronous GEMM/non-GEMM owner overlap under its exact BOM.
 * @param first_device Backend endpoint; TP2 is used whenever two participants exist.
 * @details The old ordering can schedule a full ring and native embedding slot
 * simultaneously. The production full-model failure is the retained causal
 * negative; this smaller real-path negative can depend on host scheduling.
 */
void proveEmbeddingLoadStagingOrder(DeviceId first_device)
{
    constexpr int vocab = 65536, hidden = 256;
    ScopedReadOnlyUploadSettings settings(3);
    auto *backend = getBackendFor(first_device);
    ASSERT_NE(backend, nullptr);
    ASSERT_GT(backend->deviceCount(), 0);
    const int degree = std::min(2, backend->deviceCount());
    std::vector<DeviceId> devices;
    for (int ordinal = 0; ordinal < degree; ++ordinal)
    {
        devices.emplace_back(first_device.type, ordinal);
        ASSERT_NE(GPUDeviceContextPool::instance().getContext(devices.back()).defaultStream(), nullptr);
    }
    auto original = TestTensorFactory::createQ6_KRandom({vocab, hidden}, 193703);
    std::vector<float> reference(vocab * hidden);
    original->to_fp32(reference.data());
    EmbeddingGGUF fixture(*original, true);
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(fixture.path()));
    WeightManager manager(loader);
    auto config = Qwen2SchemaFactory{}.getWeightShardingConfig();
    config.exact_matches["token_embd.weight"] = WeightShardingMode::ColumnParallel;
    manager.setWeightShardingConfig(config);
    manager.setTensorParallelConfig(std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(degree, 4, 2, 512, vocab, devices)));
    const ModelContextId model_id{193703};
    auto store = std::make_shared<PreparedWeightStore>(model_id);
    manager.setPreparedWeightStore(store);
    auto authority = concurrentVocabularyMemory(*original, devices);
    manager.installPhysicalMemoryAuthority(authority);
    ASSERT_TRUE(manager.preloadForDevices(devices));
    InferenceStrategy strategy;
    strategy.model_id = model_id;
    strategy.mode = WeightInferenceMode::LocalTP;
    strategy.tp_degree = degree;
    strategy.devices = devices;
    std::vector<FrozenModelWeightSet> primary, decode;
    primary.reserve(degree);
    decode.reserve(degree);
    for (int participant = 0; participant < degree; ++participant)
        for (bool is_decode : {false, true})
        {
            WeightPlan plan(strategy, is_decode ? PhysicalMemoryOwner::AdditionalModelWeights
                                               : PhysicalMemoryOwner::PrimaryModelWeights);
            for (const auto &[name, role] : std::array<std::pair<const char *, WeightRole>, 2>{
                     std::pair{"token_embd.weight", WeightRole::Embedding},
                     std::pair{"output.weight", WeightRole::LMHead}})
            {
                WeightRequirement requirement;
                requirement.canonical_name = name;
                requirement.target_device = devices[participant];
                requirement.lookup_device = devices[participant];
                requirement.tp_rank_or_device_index = participant;
                requirement.role = role;
                requirement.bypass_tensor_parallel = is_decode;
                requirement.host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized;
                plan.add(requirement);
            }
            (is_decode ? decode : primary).push_back(manager.materialize(plan));
        }
    PerfStatsCollector::reset();
    std::latch start(degree);
    std::vector<std::future<void>> workers;
    for (int participant = 0; participant < degree; ++participant)
        workers.push_back(std::async(std::launch::async, [&, participant]
        {
            start.arrive_and_wait();
            // Per-participant primary/decode order matches the runner; sibling
            // participants and each pipeline's internal streams remain concurrent.
            for (const auto *frozen : {&primary[participant], &decode[participant]})
                if (!manager.prepareWeightsForDevice(*frozen, devices[participant], false))
                    throw std::runtime_error("Concurrent frozen embedding/GEMM preparation returned false");
        }));
    std::string errors;
    for (auto &worker : workers)
        try { worker.get(); }
        catch (const std::exception &error) { errors += std::string(error.what()) + "\n"; }
    ASSERT_TRUE(errors.empty()) << errors;
    EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::WeightLoadStaging,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    const auto records = PerfStatsCollector::snapshot();
    for (int participant = 0; participant < degree; ++participant)
    {
        const auto device = devices[participant];
        EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::WeightLoadStaging,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        uint64_t chunks = 0;
        double bytes = 0;
        for (const auto &record : records)
            if (record.device == device.toString() && record.domain == "weight_loading" &&
                record.name == "native_embedding_pinned_upload_bytes")
            { chunks += record.count; bytes += record.value; }
        EXPECT_GT(chunks, 2u);
        EXPECT_EQ(bytes, static_cast<double>(primary[participant].global("token_embd.weight").tensor->size_bytes() +
                                            decode[participant].global("token_embd.weight").tensor->size_bytes()));
        verifyConcurrentPreparedLookup(*store, primary[participant], device, reference, vocab, hidden);
        verifyConcurrentPreparedLookup(*store, decode[participant], device, reference, vocab, hidden);
    }
}

#ifdef HAVE_CUDA
/** @test CUDA native embedding ownership survives production preparation and capture. */
TEST(EmbeddingWeightResidency, CUDA) { proveNativeEmbeddingResidency(DeviceId::cuda(0)); }
/** @test Read-only native source uses bounded admitted CUDA pinned setup. */
TEST(EmbeddingReadOnlyUpload, CUDA) { proveReadOnlyNativeUpload(DeviceId::cuda(0)); }
/** @test Concurrent frozen CUDA setup fits its existing staging admission. */
TEST(EmbeddingLoadStagingOrder, CUDA) { proveEmbeddingLoadStagingOrder(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
/** @test ROCm has the same raw-source contract, including BF16 vocabulary shards. */
TEST(EmbeddingWeightResidency, ROCm) { proveNativeEmbeddingResidency(DeviceId::rocm(0)); }
/** @test Read-only native source never reaches HIP pageable user-page pinning. */
TEST(EmbeddingReadOnlyUpload, ROCm) { proveReadOnlyNativeUpload(DeviceId::rocm(0)); }
/** @test Concurrent frozen ROCm setup fits its existing staging admission. */
TEST(EmbeddingLoadStagingOrder, ROCm) { proveEmbeddingLoadStagingOrder(DeviceId::rocm(0)); }
#endif
} // namespace
} // namespace llaminar2::test
