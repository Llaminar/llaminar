/**
 * @file Test__MoEIntermediateProjectionGeometry.cpp
 * @brief Device-free source geometry and prepared-identity proofs for F slices.
 *
 * Explicit unequal original intermediate intervals must survive N/K projection,
 * compact raw-source viewing and existing prepared adoption. These metadata and
 * tiny CPU-storage fixtures grant no TensorSharded execution/admission support.
 * Static projections require source-bound registry publication; legacy aliases
 * cannot admit a partition merely because a caller supplies its geometry.
 * No test constructs a backend, loads a model, or submits a kernel.
 */
#include "execution/moe/MoEExpertProjectionOwnership.h"
#include "execution/moe/ExpertPreparedMemoryGeometry.h"
#include "loaders/gpu_pipeline/WeightVRAMPool.h"
#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "loaders/MoEExpertSourceView.h"
#include "loaders/ExpertGemmRegistry.h"
#include "loaders/PreparedWeightStore.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/PlanningForwardWeightWork.h"
#include "planning/WeightShardGeometry.h"
#include "tensors/Tensors.h"
#include "tensors/TensorKernels.h"
#include "utils/QuantizedVerifierFormats.h"

#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace llaminar2::test
{
namespace
{
using Ownership = MoEExpertProjectionOwnership;

/** @return Observed cuts expressed as ordinary checked intervals, not a partition algorithm. */
std::vector<Ownership::IntermediateInterval> unequalIntervals()
{
    return {{0, 128}, {128, 128}, {256, 128}, {384, 256}};
}

/**
 * @brief Declare exact original GGUF provenance for a compact physical parent.
 * @param projection Existing checked GEMM N/K rectangle.
 * @return The existing source binding's original [K,N,E] axis convention.
 */
WeightSliceSpec compactSource(const Ownership::Projection &projection)
{
    return {.source_rows = static_cast<size_t>(projection.source_columns),
        .source_cols = static_cast<size_t>(projection.source_rows),
        .row_start = static_cast<size_t>(projection.first_column),
        .row_count = static_cast<size_t>(projection.columns),
        .col_start = static_cast<size_t>(projection.first_row),
        .col_count = static_cast<size_t>(projection.rows),
        .expert_count = 2, .expert_ids = {1, 7}, .inner_is_presliced = true};
}

/** @brief Nonexecuting engine used only by the existing registry identity API. */
class GeometryEngine final : public ITensorGemm
{
public:
    mutable int descriptor_queries = 0;
    /** @brief Record a pure descriptor query without any native allocation. */
    bool exportNativeVNNIMatrixDesc(DeviceNativeVNNIMatrixDesc &out) override
    {
        ++descriptor_queries;
        out = {};
        return false;
    }
    /** @brief Record a pure floating descriptor query without dereferencing storage. */
    bool exportContiguousFloatingPointWeights(ContiguousFloatingPointWeightDescriptor &out) const override
    {
        ++descriptor_queries;
        out = {};
        return false;
    }
    /** @return Availability metadata only; no backend is constructed. */
    bool supports_device(int) const override { return true; }
    /** @return False: this identity fixture never performs arithmetic. */
    bool multiply_tensor(const TensorBase *, TensorBase *, int, int, int,
        bool, float, float, const TensorBase *, const IMPIContext *, int,
        DeviceWorkspaceManager *, int) override { return false; }
};
}

/** @test Each participant retains all E, and gate/up N matches down K exactly. */
TEST(MoEIntermediateProjectionGeometry, UnequalIntervalsRetainOriginalAxesAndOldLayouts)
{
    const Ownership::Geometry geometry{512, 2560, 640};
    int frontier = 0;
    for (int participant = 0; participant < 4; ++participant)
    {
        const auto ownership = Ownership::withinExpertIntermediateSlices(geometry, participant, unequalIntervals());
        EXPECT_EQ(ownership.participants(), 4);
        const auto gate = ownership.projection(WeightRole::MoEExpertGate);
        const auto up = ownership.projection(WeightRole::MoEExpertUp);
        const auto down = ownership.projection(WeightRole::MoEExpertDown);
        EXPECT_EQ(gate, up);
        EXPECT_EQ(gate.source_rows, 640);
        EXPECT_EQ(gate.source_columns, 2560);
        EXPECT_EQ(gate.first_row, frontier);
        EXPECT_EQ(gate.first_column, 0);
        EXPECT_EQ(gate.columns, 2560);
        EXPECT_EQ(down.source_rows, 2560);
        EXPECT_EQ(down.source_columns, 640);
        EXPECT_EQ(down.first_row, 0);
        EXPECT_EQ(down.rows, 2560);
        EXPECT_EQ(down.first_column, gate.first_row);
        EXPECT_EQ(down.columns, gate.rows);
        frontier += gate.rows;
        for (const auto role : {WeightRole::MoEExpertGate, WeightRole::MoEExpertUp, WeightRole::MoEExpertDown})
        {
            for (const size_t owned : {0u, 128u, 512u})
                EXPECT_EQ(ownership.residentExperts(role, owned), 512u);
            for (int owner = 0; owner < 4; ++owner)
                EXPECT_TRUE(ownership.prepares(role, owner));
        }
        EXPECT_FALSE(ownership.hasMovableProjectionSet());
        EXPECT_THROW(ownership.movableProjections(), std::logic_error);
        DeviceMoEExpertDescriptor untouched{};
        std::array<unsigned char, sizeof(untouched)> before{};
        std::memcpy(before.data(), &untouched, sizeof(untouched));
        GeometryEngine untouched_engine;
        EXPECT_FALSE(exportDeviceMoEGateUpWeightDescriptors(&untouched_engine, &untouched_engine, ownership, untouched));
        EXPECT_EQ(untouched_engine.descriptor_queries, 0);
        EXPECT_EQ(std::memcmp(before.data(), &untouched, sizeof(untouched)), 0);
        const auto whole = Ownership::completeExperts(geometry, participant, 4);
        const auto old = Ownership::gateUpOwnedDownColumns(geometry, participant, 4);
        EXPECT_EQ(whole.projection(WeightRole::MoEExpertDown).columns, 640);
        EXPECT_EQ(old.projection(WeightRole::MoEExpertDown).columns, 640);
        EXPECT_EQ(old.projection(WeightRole::MoEExpertDown).first_column, 0);
        EXPECT_EQ(old.projection(WeightRole::MoEExpertDown).rows, 640);
        EXPECT_EQ(old.movableProjections(), DeviceMoEProjectionSet::GateUp);
        EXPECT_FALSE(exportDeviceMoEGateUpWeightDescriptors(&untouched_engine, &untouched_engine, old, untouched));
        EXPECT_GT(untouched_engine.descriptor_queries, 0); // Same pure engine is actually queried for the old representation.
    }
    EXPECT_EQ(frontier, 640);
    // Geometry is independent of QFN dimensions and does not silently balance.
    const auto small = Ownership::withinExpertIntermediateSlices({3, 7, 11}, 1, {{0, 2}, {2, 9}});
    EXPECT_EQ(small.projection(WeightRole::MoEExpertDown).columns, 9);
}

/** @test Gaps, overlap, empty intervals and foreign full-partition identities reject. */
TEST(MoEIntermediateProjectionGeometry, RejectsMalformedPartitionsAndRetainsWholePartitionIdentity)
{
    const Ownership::Geometry geometry{8, 16, 640};
    const std::vector<std::vector<Ownership::IntermediateInterval>> invalid{
        {}, {{1, 639}}, {{0, 0}, {0, 640}}, {{0, -1}}, {{0, 641}},
        {{0, 128}, {127, 513}}, {{0, 128}, {129, 511}}, {{0, 639}},
        {{0, 640}, {640, 1}}, {{0, std::numeric_limits<int>::max()}}};
    for (const auto &intervals : invalid)
        EXPECT_THROW(Ownership::withinExpertIntermediateSlices(geometry, 0, intervals), std::invalid_argument);
    EXPECT_THROW(Ownership::withinExpertIntermediateSlices(geometry, -1, unequalIntervals()), std::invalid_argument);
    EXPECT_THROW(Ownership::withinExpertIntermediateSlices(geometry, 4, unequalIntervals()), std::invalid_argument);
    const auto first = Ownership::withinExpertIntermediateSlices(geometry, 0, unequalIntervals());
    const auto different_peers = Ownership::withinExpertIntermediateSlices(geometry, 0,
        {{0, 128}, {128, 64}, {192, 192}, {384, 256}});
    EXPECT_EQ(first.projection(WeightRole::MoEExpertDown), different_peers.projection(WeightRole::MoEExpertDown));
    EXPECT_FALSE(first == different_peers); // Same local dimensions are not complete capture identity.
    EXPECT_THROW(first.projection(WeightRole::LMHead), std::invalid_argument);
}

/** @test N views remain zero-copy, while a narrow K requires genuine compact source provenance. */
TEST(MoEIntermediateProjectionGeometry, SourceViewRejectsFullParentKReinterpretationAndNeighbourOrigins)
{
    const auto ownership = Ownership::withinExpertIntermediateSlices({8, 16, 640}, 1, unequalIntervals());
    const MoEExpertSourceView gate(std::array<size_t, 3>{16, 640, 8}, {});
    const auto gate_matrix = gate.matrix(7, WeightRole::MoEExpertGate, &ownership);
    EXPECT_EQ(gate_matrix.element_offset, (7u * 640 + 128) * 16);
    EXPECT_EQ(gate_matrix.rows, 128u);
    EXPECT_EQ(gate_matrix.columns, 16u);
    const MoEExpertSourceView full_down(std::array<size_t, 3>{640, 16, 8}, {});
    EXPECT_THROW(full_down.matrix(7, WeightRole::MoEExpertDown, &ownership), std::invalid_argument);
    const auto projection = ownership.projection(WeightRole::MoEExpertDown);
    const auto identity = compactSource(projection);
    const MoEExpertSourceView compact(std::array<size_t, 3>{128, 16, 2}, identity);
    const auto matrix = compact.matrix(7, WeightRole::MoEExpertDown, &ownership);
    EXPECT_EQ(matrix.element_offset, 16u * 128);
    EXPECT_EQ(matrix.rows, 16u);
    EXPECT_EQ(matrix.columns, 128u);
    EXPECT_EQ(matrix.source_identity.source_rows, 16u); // Explicit GGUF-to-GEMM conversion.
    EXPECT_EQ(matrix.source_identity.source_cols, 640u);
    EXPECT_EQ(matrix.source_identity.col_start, 128u);
    EXPECT_EQ(matrix.source_identity.col_count, 128u);
    EXPECT_EQ(matrix.source_identity.expert_ids, (std::vector<int>{7}));
    for (int mutation = 0; mutation < 7; ++mutation)
    {
        auto foreign = identity;
        switch (mutation)
        {
        case 0: foreign.row_start = 256; break;
        case 1: foreign.source_rows = 0; break;
        case 2: foreign.row_count = 0; break;
        case 3: foreign.inner_is_presliced = false; break;
        case 4: foreign.source_cols = 17; break;
        case 5: foreign.col_start = 1; break;
        case 6: foreign.source_cols = 0; break;
        }
        const auto resolve = [&] {
            const MoEExpertSourceView wrong(std::array<size_t, 3>{128, 16, 2}, foreign);
            return wrong.matrix(7, WeightRole::MoEExpertDown, &ownership);
        };
        EXPECT_THROW((void)resolve(), std::invalid_argument) << mutation;
    }
}

/** @test Native compact views retain the real original block address and byte extent. */
TEST(MoEIntermediateProjectionGeometry, CompactViewsPreserveEveryNativeFormatWithoutRepacking)
{
    const auto prove = [](const char *label, int quantum, const auto &create) {
        SCOPED_TRACE(label);
        const int k = 4 * quantum;
        const auto ownership = Ownership::withinExpertIntermediateSlices({8, 4, 20 * quantum}, 1,
            {{0, k}, {k, k}, {2 * k, k}, {3 * k, 8 * quantum}});
        const auto projection = ownership.projection(WeightRole::MoEExpertDown);
        // This is an authored already compact source fixture, not loader/runtime proof.
        std::shared_ptr<TensorBase> source(create(std::vector<size_t>{8, static_cast<size_t>(k)}, 77));
        const MoEExpertSourceView compact(std::array<size_t, 3>{static_cast<size_t>(k), 4, 2},
            compactSource(projection));
        const auto matrix = compact.matrix(7, WeightRole::MoEExpertDown, &ownership);
        const auto view = source->create_view({matrix.rows, matrix.columns}, matrix.element_offset);
        ASSERT_NE(view, nullptr);
        EXPECT_EQ(view->native_type(), source->native_type());
        const auto *expected = static_cast<const uint8_t *>(source->raw_data()) + source->size_bytes() / 2;
        EXPECT_EQ(view->raw_data(), expected);
        EXPECT_EQ(view->size_bytes(), source->size_bytes() / 2);
        EXPECT_EQ(std::memcmp(view->raw_data(), expected, view->size_bytes()), 0);
    };
    for (const auto &format : quantizedVerifierFormats())
        prove(format.label, format.source_is_superblock ? 256 : 32, format.create);
    prove("FP32", 8, [](const auto &shape, uint32_t seed) {
        return TestTensorFactory::createFP32Random(shape, -1.f, 1.f, seed);
    });
    prove("FP16", 8, [](const auto &shape, uint32_t seed) {
        return TestTensorFactory::createFP16Random(shape, -1.f, 1.f, seed);
    });
    prove("BF16", 8, [](const auto &shape, uint32_t seed) {
        return TestTensorFactory::createBF16Random(shape, -1.f, 1.f, seed);
    });
}

/** @test Logical physical geometry and work retain local K and all E, without a second TP quotient. */
TEST(MoEIntermediateProjectionGeometry, SizingAndWorkUseExactLocalNKAndFullExpertInventory)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen35moe";
    profile.n_layers = 1;
    profile.expert_count = 512;
    profile.d_model = 2560;
    profile.expert_feed_forward_length = 640;
    const size_t elements = 512u * 2560 * 640;
    const std::array<TensorSizeInfo, 3> tensors{{
        {"blk.0.ffn_gate_exps.weight", 4 * elements, "F32", elements, 2560, 0},
        {"blk.0.ffn_up_exps.weight", 4 * elements, "F32", elements, 2560, 0},
        {"blk.0.ffn_down_exps.weight", 4 * elements, "F32", elements, 640, 0}}};
    const std::array roles{WeightRole::MoEExpertGate, WeightRole::MoEExpertUp, WeightRole::MoEExpertDown};
    const WeightShardGeometryResolver full(profile, DeviceId::cpu());
    for (int participant = 0; participant < 4; ++participant)
    {
        const auto ownership = Ownership::withinExpertIntermediateSlices({512, 2560, 640}, participant, unequalIntervals());
        const WeightShardGeometryResolver resolver(profile, DeviceId::cpu(), participant, 4);
        const auto operand = [&](size_t i) {
            return PlanningWeightOperand{tensors[i].name, roles[i], 0, tensors[i].quant_type,
                ModelPreparedWeightRepresentation::SourceNative, full.resolve(tensors[i], 1)};
        };
        const PlanningRoutedExpertWeightWork work{0, 512, 10,
            {operand(0), operand(1), operand(2)},
            {PlanningExpertExecutionShare(PlanningExpertExecution::OwnedExperts, 128)}, ownership};
        for (size_t i = 0; i < roles.size(); ++i)
        {
            const auto projection = ownership.projection(roles[i]);
            const WeightShardMatrix local{static_cast<size_t>(projection.rows),
                static_cast<size_t>(projection.columns), 512};
            for (const size_t owned : {0u, 128u, 512u})
            {
                const auto resolved = resolver.resolve(tensors[i], ownership, owned);
                ASSERT_TRUE(resolved.matrix());
                EXPECT_EQ(*resolved.matrix(), local);
                EXPECT_EQ(resolved.elements(), local.rows * local.columns * 512);
            }
            EXPECT_EQ(work.projectionMatrix(roles[i]), (WeightShardMatrix{local.rows, local.columns, 1}));
            EXPECT_DOUBLE_EQ(work.projectionExpectation(roles[i], 3).routed_rows, 30);
            EXPECT_DOUBLE_EQ(work.projectionExpectation(roles[i], 0).routed_rows, 0);
        }
    }
}

/** @test Existing prepared adoption rejects a foreign equal-shape K interval. */
TEST(MoEIntermediateProjectionGeometry, PreparedStoreRetainsOriginalLocalInterval)
{
    using Factory = llaminar::v2::kernels::KernelFactory;
    FP32Tensor original(std::vector<size_t>{4, 128});
    FP32Tensor replacement(std::vector<size_t>{4, 128});
    PreparedWeightStore store(ModelContextId{99});
    const auto ownership = Ownership::withinExpertIntermediateSlices({8, 4, 640}, 1, unequalIntervals());
    const MoEExpertSourceView source(std::array<size_t, 3>{128, 4, 2},
        compactSource(ownership.projection(WeightRole::MoEExpertDown)));
    WeightBinding binding;
    binding.binding_id = 10;
    binding.identity = makeSourceWeightIdentity("blk.0.ffn_down_exps.weight", ModelContextId{99}, 10);
    binding.tensor = &original;
    binding.immutable = true;
    binding.slice = source.matrix(7, WeightRole::MoEExpertDown, &ownership).source_identity;
    auto handle = std::make_shared<Factory::PreparedGemmHandle>();
    handle->tensor = &original;
    handle->device_id = DeviceId::cpu();
    handle->kind = Factory::GemmPreparationKind::FLOATING_POINT;
    handle->prepared_weights = std::make_shared<Factory::PreparedGemmWeights>();
    handle->prepared_weights->kind = handle->kind;
    store.registerPreparedGemmHandle(binding, PreparedWeightKind::CpuPackedGemm, DeviceId::cpu(), std::move(handle));
    for (int mutation = 0; mutation < 4; ++mutation)
    {
        auto requested = binding;
        requested.binding_id = static_cast<uint64_t>(20 + mutation);
        requested.tensor = &replacement;
        requested.prepared = PreparedWeightRef{ModelContextId{99}, requested.binding_id,
            PreparedWeightKind::CpuPackedGemm, DeviceId::cpu()};
        switch (mutation)
        {
        case 0: requested.slice.col_start = 256; break;
        case 1: requested.slice.source_cols = 768; break;
        case 2: requested.slice.expert_ids = {6}; break;
        case 3: break;
        }
        EXPECT_EQ(store.adoptPreparedGemmForBinding(requested, DeviceId::cpu()), mutation == 3);
    }
}

/** @test Static geometry cannot bypass source-bound preparation through legacy registry APIs. */
TEST(MoEIntermediateProjectionGeometry, RegistryRejectsStaticProjectionWithoutOriginalSource)
{
    const auto ownership = Ownership::withinExpertIntermediateSlices({8, 16, 640}, 0, unequalIntervals());
    const auto different_peers = Ownership::withinExpertIntermediateSlices({8, 16, 640}, 0,
        {{0, 128}, {128, 64}, {192, 192}, {384, 256}});
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        ExpertGemmRegistry registry;
        auto engine = std::make_shared<GeometryEngine>();
        for (const auto &partition : {ownership, different_peers})
        {
            EXPECT_THROW(registry.registerEngineForParticipant("original_domain", device, 3, 0, 0, 7,
                ExpertGemmRegistry::WeightRole::DOWN, engine.get(), engine, partition), std::invalid_argument);
            EXPECT_THROW(registry.registerEngineForDomain("original_domain", device, 0, 7,
                ExpertGemmRegistry::WeightRole::DOWN, engine.get(), engine, partition), std::invalid_argument);
            EXPECT_EQ(registry.size(), 0u);
            EXPECT_EQ(engine->descriptor_queries, 0);
        }
    }
}
} // namespace llaminar2::test

namespace llaminar2::test
{
/**
 * @brief Compare canonical admission with the real unallocated pool planner.
 *
 * Each case prices one source projection, including allocator rounding. No
 * backend or allocation is created. Source canonicalization (not the source's
 * unnormalized asymmetry flag) determines whether a physical minima plane exists.
 */
TEST(MoEIntermediateProjectionGeometry, PreparedPoolLayoutsMatchActualPoolPlans)
{
    using Layout = GPUPreparedWeightPoolLayout;
    for (const auto &entry : native_vnni_formats::kAllSourceFormats)
    {
        const auto &source = *entry.metadata;
        const auto format = ExpertWeightFormat::nativeVnni(
            {source.codebook_id, source.is_superblock, true});
        for (const int n : {1, 128, 256, 640})
        for (const int k : {256, 512, 2560})
        for (const auto layout : {Layout::SourceNative, Layout::MigrationReusable})
        {
            SCOPED_TRACE(std::string(entry.quant_type));
            SCOPED_TRACE(n);
            SCOPED_TRACE(k);
            SCOPED_TRACE(static_cast<int>(layout));
            const auto physical = layout == Layout::MigrationReusable
                ? reusableDeviceVnniAllocationFormat(source)
                : NativeVnniReusableDeviceAllocationFormat{
                      static_cast<uint8_t>(source.payload_bytes),
                      canonicalDeviceVnniFormat(source).is_asymmetric,
                      source.has_emins};
            WeightVRAMPool pool;
            pool.planWeight("original", n, k, physical.payload_bytes_per_block,
                physical.has_mins, physical.has_emins, 0);
            const auto geometry = resolveExpertPreparedProjectionMemoryGeometry(
                n, k, format, layout);
            EXPECT_EQ(geometry.gpu_live_bytes,
                alignGPUWeightLoadAllocation(pool.totalPlannedBytes()));
            EXPECT_EQ(geometry.cpu_bytes,
                resolveExpertPreparedProjectionMemoryGeometry(n, k, format).cpu_bytes);
            if (layout == Layout::MigrationReusable)
                EXPECT_EQ(geometry.gpu_live_bytes,
                    resolveExpertPreparedProjectionMemoryGeometry(n, k, format).gpu_live_bytes);
        }
    }
    for (const auto type : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
    for (const auto layout : {Layout::SourceNative, Layout::MigrationReusable})
    {
        const auto format = ExpertWeightFormat::floating(type);
        WeightVRAMPool pool;
        pool.planRawWeight("original", 128, 256,
            size_t{128} * 256 * format.floatingElementBytes());
        EXPECT_EQ(resolveExpertPreparedProjectionMemoryGeometry(128, 256, format, layout).gpu_live_bytes,
            alignGPUWeightLoadAllocation(pool.totalPlannedBytes()));
        EXPECT_THROW((void)resolveExpertPreparedProjectionMemoryGeometry(128, 256, format,
            static_cast<Layout>(99)), std::invalid_argument);
    }
    EXPECT_THROW((void)resolveExpertPreparedProjectionMemoryGeometry(128, 256,
        ExpertWeightFormat::nativeVnni({native_vnni_formats::Q5_0.codebook_id, false, true}),
        static_cast<Layout>(99)), std::invalid_argument);
}
} // namespace llaminar2::test
