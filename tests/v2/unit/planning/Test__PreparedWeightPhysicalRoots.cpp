/**
 * @file Test__PreparedWeightPhysicalRoots.cpp
 * @brief Replay actual store identity and physical weight BOM creator/alias lowering.
 *
 * Synthetic source metadata and one-scalar tensor facades exercise the production
 * contract without model payloads, native backend setup or GPU allocations. Exact
 * original source scopes, partial layers and full15 staging remain independent
 * obligations. A typed alias changes only duplicate persistent creator bytes.
 */
#include <gtest/gtest.h>
#include "loaders/PreparedWeightIdentityContract.h"
#include "loaders/PreparedWeightStore.h"
#include "planning/PreparedWeightPhysicalUses.h"
#include "planning/WeightShardGeometry.h"
#include "planning/WeightMemoryEstimator.h"
#include "planning/MemoryPlanner.h"
#include "tensors/TensorClasses.h"
#include "../../utils/CPUPreparedStoreTestAdmission.h"
#include <array>
#include <memory>

using namespace llaminar2;
namespace
{
    /** @brief One synthetic scalar backing; overridden metadata is never read as payload. */
    class IdentityTensor final : public FP32Tensor
    {
    public:
        /** @brief Retain tiny CPU storage while reporting the declared source geometry/type. */
        IdentityTensor(std::vector<size_t> shape, TensorType type)
            : FP32Tensor({1}), shape_(std::move(shape)), type_(type) {}
        /** @return Declared immutable metadata shape. */
        const std::vector<size_t> &shape() const override { return shape_; }
        /** @return Declared final prepared type. */
        TensorType native_type() const override { return type_; }
    private:
        std::vector<size_t> shape_;
        TensorType type_;
    };
    /** @brief Source binding in one exact model/device scope. */
    WeightBinding binding(DeviceId device, std::vector<size_t> shape = {8,32}, TensorType type = TensorType::Q8_0)
    {
        WeightBinding b;
        b.binding_id = 1;
        b.identity.model_id = ModelContextId{99};
        b.identity.canonical_name = "blk.0.attn_q.weight";
        b.identity.role = WeightRole::AttentionQ;
        b.tensor_owner = std::make_shared<IdentityTensor>(shape, type);
        b.tensor = b.tensor_owner.get();
        b.slice.source_rows = b.slice.row_count = shape[0];
        b.slice.source_cols = b.slice.col_count = shape[1];
        b.residency.home_device = device;
        b.residency.resident_device = device;
        return b;
    }
    /** @brief Backend kind already selected by the actual preparation dispatcher. */
    PreparedWeightKind kind(DeviceId d)
    {
        return d.is_cpu() ? PreparedWeightKind::CpuPackedGemm :
            d.is_cuda() ? PreparedWeightKind::CudaInt8PackedGemm : PreparedWeightKind::RocmInt8PackedGemm;
    }
    /** @brief Two original layers and one retained predictor with ordinary and replicated matrices. */
    ModelMemoryProfile profile()
    {
        ModelMemoryProfile p;
        p.architecture = "qwen2";
        p.n_layers = 3; p.mtp_layer_count = 1;
        p.d_model = 32; p.d_ff = 64; p.n_heads = 4; p.n_kv_heads = 2;
        p.head_dim = 8; p.vocab_size = 64; p.max_seq_len = 64;
        for (int layer = 0; layer != 3; ++layer)
        {
            const auto prefix = "blk." + std::to_string(layer);
            // The schema's default replicated legacy GEMM classification is
            // intentional: production must not special-case QFN family names.
            p.tensors.push_back({prefix + ".custom_projection.weight", 8*32*4, "F32", 8*32, 32, layer});
            p.tensors.push_back({prefix + ".attn_q.weight", 32*32*4, "F32", 32*32, 32, layer});
            p.tensors.push_back({prefix + ".attn_norm.weight", 32*4, "F32", 32, 32, layer});
        }
        return p;
    }
}

TEST(PreparedWeightPhysicalRoots, ActualStoreAliasesOneHandleAndKeepsOriginalProjectionOwner)
{
    for (auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        PreparedWeightStore store(ModelContextId{99});
        auto a = binding(device);
        auto h = std::make_shared<llaminar::v2::kernels::KernelFactory::PreparedGemmHandle>();
        h->tensor = a.tensor; h->device_id = device;
        h->prepared_weights = std::make_shared<llaminar::v2::kernels::KernelFactory::PreparedGemmWeights>();
        store.registerPreparedGemmHandle(a, kind(device), device, h, PhysicalMemoryOwner::PrimaryModelWeights);
        auto b = binding(device); b.binding_id = 2;
        b.prepared = PreparedWeightRef{ModelContextId{99},2,kind(device),device};
        ASSERT_TRUE(store.adoptPreparedGemmForBinding(b, device));
        EXPECT_EQ(h.use_count(), 3);
        auto changed = binding(device, {8,16}); changed.binding_id = 3;
        changed.slice.source_cols = 32; changed.slice.inner_is_presliced = true;
        changed.prepared = PreparedWeightRef{ModelContextId{99},3,kind(device),device};
        EXPECT_FALSE(store.adoptPreparedGemmForBinding(changed, device));
        b.binding_id = 4; b.prepared->binding_id = 4;
        b.floating_operand_precision = FloatingActivationOperandPrecision::FP32;
        EXPECT_FALSE(store.adoptPreparedGemmForBinding(b, device));
        b = binding(device, {8,32}, TensorType::FP16); b.binding_id = 5;
        b.prepared = PreparedWeightRef{ModelContextId{99},5,kind(device),device};
        EXPECT_FALSE(store.adoptPreparedGemmForBinding(b, device));
        b = binding(device); b.binding_id = 6;
        b.identity.canonical_name = "blk.1.attn_q.weight";
        b.prepared = PreparedWeightRef{ModelContextId{99},6,kind(device),device};
        EXPECT_FALSE(store.adoptPreparedGemmForBinding(b, device));
        b.identity.model_id = ModelContextId{100}; b.identity.canonical_name = a.identity.canonical_name;
        EXPECT_THROW(store.adoptPreparedGemmForBinding(b, device), std::runtime_error);
        b = binding(device); b.binding_id = 7;
        b.prepared = PreparedWeightRef{ModelContextId{99},7,PreparedWeightKind::PreparedEmbedding,device};
        EXPECT_FALSE(store.adoptPreparedForBinding(b, device));
        std::weak_ptr weak = h; h.reset();
        EXPECT_FALSE(weak.expired());
        store.releaseAllPreparedState();
        EXPECT_TRUE(weak.expired());
    }
    auto a = binding(DeviceId::cpu());
    a.projection_ownership = MoEExpertProjectionOwnership::completeExperts({2,32,8});
    a.projection_source_owner = std::make_shared<IdentityTensor>(std::vector<size_t>{2,8,32}, TensorType::Q8_0);
    auto b = a;
    auto original = PreparedWeightIdentityContract::fromBinding(a);
    ASSERT_TRUE(original);
    b.projection_source_owner = std::shared_ptr<TensorBase>(a.projection_source_owner.get(), [](TensorBase*){});
    EXPECT_FALSE(original->compatible(*PreparedWeightIdentityContract::fromBinding(b)));
    b = a;
    EXPECT_TRUE(original->compatible(*PreparedWeightIdentityContract::fromBinding(b)));
    b.projection_ownership.reset();
    EXPECT_FALSE(original->compatible(*PreparedWeightIdentityContract::fromBinding(b)));
}

TEST(PreparedWeightPhysicalRoots, EmptyFullAndOriginalSliceNormalizationPreservesRuntimeSemantics)
{
    auto a = binding(DeviceId::cpu()), b = a;
    b.slice = {};
    EXPECT_TRUE(PreparedWeightIdentityContract::fromBinding(a)->compatible(*PreparedWeightIdentityContract::fromBinding(b)));
    a.slice.source_cols = 64; a.slice.col_start = 32; a.slice.inner_is_presliced = true;
    EXPECT_FALSE(PreparedWeightIdentityContract::fromBinding(a)->compatible(*PreparedWeightIdentityContract::fromBinding(b)));
    b = a; b.slice.col_start = 0;
    EXPECT_FALSE(PreparedWeightIdentityContract::fromBinding(a)->compatible(*PreparedWeightIdentityContract::fromBinding(b)));
    b = a; b.tensor = nullptr;
    EXPECT_FALSE(PreparedWeightIdentityContract::fromBinding(b));
}

TEST(PreparedWeightPhysicalRoots, ExactOriginalInventoryAndDeviceBindEveryInspectedAliasEdge)
{
    auto p = profile();
    auto copy = p;
    for (auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        PreparedWeightPhysicalUses uses(p, device);
        WeightShardGeometryResolver full(p, device), tp(p, device, 0, 2);
        const auto &source = p.tensors[0];
        const auto creator = uses.record(p, device, source, tp.resolve(source), tp.sourceStorageView(source), PreparedWeightUseIntent::PublishCreator);
        ASSERT_TRUE(std::holds_alternative<PreparedWeightPhysicalCreator>(creator));
        const auto alias = uses.record(p, device, source, full.resolve(source), full.sourceStorageView(source), PreparedWeightUseIntent::ReuseExistingOrCreate);
        ASSERT_TRUE(std::holds_alternative<PreparedWeightPhysicalAlias>(alias));
        EXPECT_EQ(std::get<PreparedWeightPhysicalAlias>(alias).root, 0u);
        EXPECT_EQ(uses.originalSource(0), &source);
        EXPECT_EQ(uses.creatorCount(), 1u);
        EXPECT_THROW(uses.record(copy, device, copy.tensors[0], full.resolve(source), full.sourceStorageView(source),
            PreparedWeightUseIntent::ReuseExistingOrCreate), std::invalid_argument);
        EXPECT_THROW(uses.record(p, DeviceId::cuda(7), source, full.resolve(source), full.sourceStorageView(source),
            PreparedWeightUseIntent::ReuseExistingOrCreate), std::invalid_argument);
        const auto foreign_entry = uses.record(p, device, copy.tensors[0], full.resolve(source), full.sourceStorageView(source),
            PreparedWeightUseIntent::ReuseExistingOrCreate);
        ASSERT_TRUE(std::holds_alternative<PreparedWeightPhysicalCreator>(foreign_entry));
        EXPECT_EQ(uses.originalSource(std::get<PreparedWeightPhysicalCreator>(foreign_entry).root), nullptr);
        const auto sliced = uses.record(p, device, p.tensors[1], tp.resolve(p.tensors[1]), tp.sourceStorageView(p.tensors[1]), PreparedWeightUseIntent::PublishCreator);
        const auto replica = uses.record(p, device, p.tensors[1], full.resolve(p.tensors[1]), full.sourceStorageView(p.tensors[1]), PreparedWeightUseIntent::ReuseExistingOrCreate);
        EXPECT_TRUE(std::holds_alternative<PreparedWeightPhysicalCreator>(sliced));
        EXPECT_TRUE(std::holds_alternative<PreparedWeightPhysicalCreator>(replica));
        const auto next_layer = uses.record(p, device, p.tensors[3], full.resolve(p.tensors[3]), full.sourceStorageView(p.tensors[3]), PreparedWeightUseIntent::ReuseExistingOrCreate);
        EXPECT_TRUE(std::holds_alternative<PreparedWeightPhysicalCreator>(next_layer));
    }
}

TEST(PreparedWeightPhysicalRoots, ActualMemoryPlannerPricesCreatorsOnceWithFull15AndFullStaging)
{
    const auto p = profile();
    for (auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
        for (auto interval : {std::pair{0,0}, std::pair{1,1}})
    {
        DevicePlanConfig cfg;
        cfg.device = device; cfg.device_compute_units = device.is_cuda() ? 82 : 60;
        cfg.device_total_bytes = cfg.device_free_bytes = size_t{64} << 30;
        cfg.first_layer = interval.first; cfg.last_layer = interval.second;
        cfg.total_shards = 2; cfg.max_seq_len = 64; cfg.activation_seq_len = 64;
        cfg.mtp_enabled = true; cfg.mtp_target_query_rows = 16;
        cfg.additional_weight_sets = {AdditionalPersistentWeightSet::ReplicatedDenseDecode};
        cfg.weight_load_staging.device_bytes = cfg.weight_load_staging.host_bytes = 536870400;
        cfg.associated_host_memory = PhysicalMemoryResource{.world_rank=-1,.device=DeviceId::cpu(),
            .total_bytes=size_t{128}<<30,.admission_available_bytes=size_t{128}<<30};
        const auto plan = MemoryPlanner::plan(p, {cfg});
        const auto found = std::find_if(plan.devices.begin(), plan.devices.end(),
            [device](const auto &candidate) { return candidate.device() == device; });
        ASSERT_NE(found, plan.devices.end());
        const int last = interval.second == p.mainLayerCount()-1 ? p.totalLayerCount()-1 : interval.second;
        const auto scope = interval.first == 0 ? WeightComponentScope::Embedding : WeightComponentScope::EmbeddingAndTerminal;
        const auto logical = WeightMemoryEstimator::estimate(p, device, 0, 1, interval.first, last, {}, {}, scope);
        const size_t original_full_replicas = size_t(last - interval.first + 1) * 8 * 32 * sizeof(float);
        EXPECT_EQ(found->additional_weight_bytes(), logical.device_bytes - original_full_replicas);
        EXPECT_EQ(found->bom().charge(PhysicalMemoryOwner::WeightLoadStaging).planned_bytes, 536870400u);
        EXPECT_EQ(cfg.mtp_target_query_rows, 16);
    }
}

TEST(PreparedWeightPhysicalRoots, ActualCPUPackedCreatorHasOnePMAClaimThroughAdoptionAndRetirement)
{
    for (bool source_first : {false, true})
    {
        PreparedWeightStore store(ModelContextId{99});
        auto source = std::make_shared<Q8_0Tensor>(std::vector<size_t>{8,32}, std::vector<uint8_t>(8*34));
        test::admitCPUPreparedStoreForTest(store, {source.get()});
        const auto authority = store.physicalMemoryAuthority();
        const auto bytes = test::cpuPreparedStoreTestBytes(*source);
        ASSERT_GT(bytes, 0u);
        auto a = binding(DeviceId::cpu());
        a.tensor_owner = source; a.tensor = source.get(); a.immutable = true;
        const auto primary = store.prepareGemm(a);
        const auto live_bytes = [&] {
            size_t live = 0;
            for (const auto &entry : authority->rankAttestation())
                if (entry.owner == PhysicalMemoryOwner::PrimaryModelWeights) live += entry.materialized_new_bytes;
            return live;
        };
        EXPECT_EQ(live_bytes(), bytes);
        auto b = a; b.binding_id = 2;
        b.prepared = PreparedWeightRef{ModelContextId{99},2,PreparedWeightKind::CpuPackedGemm,DeviceId::cpu()};
        ASSERT_TRUE(store.adoptPreparedGemmForBinding(b, DeviceId::cpu()));
        EXPECT_EQ(store.gemmKernel(primary), store.gemmKernel(*b.prepared));
        EXPECT_EQ(live_bytes(), bytes);
        if (source_first) { source.reset(); a.tensor_owner.reset(); }
        else b.tensor_owner.reset();
        EXPECT_EQ(live_bytes(), bytes);
        store.releaseAllPreparedState();
        EXPECT_EQ(live_bytes(), 0u);
    }
}

TEST(PreparedWeightPhysicalRoots, TerminalInputsRequireTheActualSchemaComponentOwnerBeforeAliasPublication)
{
    auto p = profile(); p.architecture = "qwen4exp";
    p.tensors = {{"output_hc_down.weight", 8*32*4, "F32", 8*32, 32, -1}};
    const auto device = DeviceId::cuda(0);
    WeightShardGeometryResolver geometry(p, device);
    const auto &source = p.tensors.front();
    for (auto scope : {WeightComponentScope::Intermediate, WeightComponentScope::Terminal})
    {
        PreparedWeightPhysicalUses uses(p, device, scope);
        uses.record(p, device, source, geometry.resolve(source), geometry.sourceStorageView(source), PreparedWeightUseIntent::PublishCreator);
        const auto second = uses.record(p, device, source, geometry.resolve(source), geometry.sourceStorageView(source), PreparedWeightUseIntent::ReuseExistingOrCreate);
        EXPECT_EQ(std::holds_alternative<PreparedWeightPhysicalAlias>(second), weightScopeOwnsTerminal(scope));
        EXPECT_THROW(WeightMemoryEstimator::estimatePhysical(uses, PreparedWeightUseIntent::ReuseExistingOrCreate,
            p, device, 0, 1, 0, 0, {}, {}, WeightComponentScope::EmbeddingAndTerminal), std::invalid_argument);
    }
}

TEST(PreparedWeightPhysicalRoots, FullCoordinateRangeUnderTPSlicingPolicyIsNotAnOriginalSourceAlias)
{
    const auto p = profile();
    for (auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        DeviceShardingAssignment assignment;
        assignment.device = device; assignment.local_rank = 0;
        assignment.head_count = p.n_heads; assignment.kv_head_count = p.n_kv_heads;
        assignment.d_ff_count = p.d_ff; assignment.vocab_count = p.vocab_size;
        WeightShardGeometryResolver sliced(p, device, 0, 2, assignment), full(p, device);
        const auto &source = p.tensors[1];
        ASSERT_EQ(*sliced.resolve(source).matrix(), *full.resolve(source).matrix());
        ASSERT_EQ(sliced.sourceStorageView(source), WeightSourceStorageView::DeclaredSlice);
        PreparedWeightPhysicalUses uses(p, device);
        uses.record(p, device, source, sliced.resolve(source), sliced.sourceStorageView(source), PreparedWeightUseIntent::PublishCreator);
        EXPECT_TRUE(std::holds_alternative<PreparedWeightPhysicalCreator>(uses.record(p, device, source,
            full.resolve(source), full.sourceStorageView(source), PreparedWeightUseIntent::ReuseExistingOrCreate)));
        auto original = binding(device), replica = original;
        original.slice.inner_is_presliced = true;
        EXPECT_FALSE(PreparedWeightIdentityContract::fromBinding(original)->compatible(
            *PreparedWeightIdentityContract::fromBinding(replica)));
    }
}
