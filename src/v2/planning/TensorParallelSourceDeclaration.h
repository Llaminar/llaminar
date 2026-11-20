/**
 * @file TensorParallelSourceDeclaration.h
 * @brief One original-source/retained-family TP declaration for admission and construction.
 *
 * Model metadata owns original geometry; the schema owns main consumer intent.
 * This pure boundary neither constructs execution state nor distributes heads.
 * TensorParallelConfig remains the sole assignment resolver, and PMA remains
 * the sole byte authority. Explicit full-model families retain legacy coverage.
 */
#pragma once
#include "config/TensorParallelAssignmentDeclaration.h"
#include "execution/config/RuntimeConfig.h"
#include "execution/local_execution/graph/SchemaFactoryRegistry.h"
#include "models/qwen4exp/Qwen4ExpConfig.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"
#include "loaders/WeightIdentity.h"
#include "tensors/BlockStructures.h"
#include "planning/ModelMemoryProfile.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include <span>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
/** @brief One shared source width and its common native Down K block alignment. */
struct OriginalSharedExpertTPGeometry
{
    int columns;
    int group;

    /** @return Whether every member can own at least one complete original block. */
    bool supportsDegree(int degree) const noexcept
    { return degree > 0 && columns % group == 0 && columns / group >= degree; }
};

/** @return Shared source geometry if every retained main and predictor block has an original Down. */
inline std::optional<OriginalSharedExpertTPGeometry> originalSharedExpertTPGeometry(
    const ModelMemoryProfile &profile, const MTPRuntimeConfig &mtp)
{
    OriginalSharedExpertTPGeometry geometry{0, 1};
    const int layers = retainsMTPGraphCapacity(mtp) ? profile.totalLayerCount() : profile.mainLayerCount();
    std::vector<bool> found(layers);
    for (const auto &tensor : profile.tensors)
    {
        if (inferWeightRole(tensor.name) != WeightRole::SharedExpertDown || tensor.layer_index < 0 ||
            tensor.layer_index >= layers)
            continue;
        if (found[tensor.layer_index] || tensor.K == 0 || tensor.K > std::size_t(std::numeric_limits<int>::max()) ||
            (geometry.columns && geometry.columns != static_cast<int>(tensor.K)))
            throw std::invalid_argument("Original shared expert Down source has inconsistent or repeated intermediate geometry");
        found[tensor.layer_index] = true;
        geometry.columns = static_cast<int>(tensor.K);
        const auto *native = native_quantised_formats::forQuantType(tensor.quant_type);
        if (native)
            geometry.group = std::max(geometry.group, static_cast<int>(native->is_superblock
                ? Q4_KBlock::BLOCK_SIZE : Q4_0Block::BLOCK_SIZE));
        else if (tensor.quant_type != "F16" && tensor.quant_type != "FP16" &&
                 tensor.quant_type != "F32" && tensor.quant_type != "FP32" && tensor.quant_type != "BF16")
            throw std::invalid_argument("Original shared expert Down source uses an unsupported native codebook");
    }
    if (!geometry.columns || !std::all_of(found.begin(), found.end(), [](bool present) { return present; }))
        return std::nullopt;
    return geometry;
}

/** @return Whether the prepared continuation binds one original shared shard per GPU. */
inline bool usesNativeSharedExpertTP(const MoERoutedExpertPlacementPlan *plan)
{
    if (!plan) return false;
    const auto continuation = std::find_if(plan->domains.begin(), plan->domains.end(), [&](const auto &domain) {
        return domain.name == plan->continuation_domain;
    });
    return continuation != plan->domains.end() &&
        continuation->routed_compute_policy == RoutedExpertComputePolicy::TensorSharded &&
        continuation->participants.size() > 1 && continuation->participants.front().isGPU();
}

/** @return Original shared source partition applied to the exact selected TP membership. */
inline TensorParallelConfig bindOriginalSharedExpertTP(
    const ModelMemoryProfile &profile, const MTPRuntimeConfig &mtp, const TensorParallelConfig &assignment)
{
    const auto source = originalSharedExpertTPGeometry(profile, mtp);
    if (!source)
        throw std::invalid_argument("Original shared expert Down source must cover every main and predictor block");
    return assignment.withNativeSharedFFN(source->columns, source->group);
}

/** @brief Authenticated existing placement scope; it never derives locality from an ordinal/hostname. */
enum class TensorParallelMainPlacement { FullMainRankLocal, OtherDeclared };
/** @brief Original source arithmetic family selected before assignment; this is not a runtime recovery path. */
enum class TensorParallelKVProjectionSource { NativeQuantizedOriginalSlices, FloatingFullBank };
/**
 * @brief Classify every original main K/V projection through canonical source metadata.
 * @param original Complete planning source projection, including every original main matrix.
 * @return Native-only original-slice arithmetic or the existing explicitly floating full-bank family.
 * @throws std::invalid_argument For missing, duplicate, malformed or unsupported source descriptors.
 *
 * NativeQuantised's catalog owns format admission. Mixed native/floating K/V sources
 * belong to the full-bank family. No selected QueryInterval failure can choose
 * this family later; selection happens before any physical assignment.
 */
inline TensorParallelKVProjectionSource originalQfnKVProjectionSource(const ModelMemoryProfile &original)
{
    if(original.architecture!="qwen4exp" || !original.flash_next || original.n_kv_heads<=0 ||
        original.head_dim<=0 || original.d_model<=0)
        throw std::invalid_argument("QFN K/V source classification lacks original geometry");
    bool floating=false;
    for(int layer=0;layer<original.mainLayerCount();++layer)
        if(original.flashNextLayerKind(layer)==Qwen4ExpLayerKind::QSA)
            for(const char *role:{"attn_k.weight","attn_v.weight"})
            {
                const auto name="blk."+std::to_string(layer)+"."+role;
                const TensorSizeInfo *source=nullptr;
                for(const auto &tensor:original.tensors)if(tensor.name==name)
                {
                    if(source)throw std::invalid_argument("QFN K/V original descriptor is duplicated: "+name);
                    source=&tensor;
                }
                const auto columns=std::size_t(original.n_kv_heads)*std::size_t(original.head_dim);
                if(!source || source->K!=std::size_t(original.d_model) || source->elements/source->K!=columns ||
                    source->elements%source->K)
                    throw std::invalid_argument("QFN K/V original matrix descriptor is missing or malformed: "+name);
                if(native_quantised_formats::forQuantType(source->quant_type))continue;
                if(source->quant_type=="F32" || source->quant_type=="F16" || source->quant_type=="BF16")floating=true;
                else throw std::invalid_argument("QFN K/V original source arithmetic is unsupported: "+name);
            }
    return floating?TensorParallelKVProjectionSource::FloatingFullBank:
        TensorParallelKVProjectionSource::NativeQuantizedOriginalSlices;
}
/**
 * @brief Resolve original main sparse/GDN consumers before any physical split.
 * @param queries Complete original query partition width.
 * @param kv Original native KV head count.
 * @param gdns Every retained original main GDN consumer, in source order.
 * @param declared Main schema intent; other models preserve Legacy.
 * @param placement Existing admitted full-main/local scope or separately declared family.
 * @param mtp Actual retained graph-capacity policy, not only request execution enablement.
 * @param dense Actual explicit dense policy; replicated decode is a separate demand.
 * @param devices Actual ordered local endpoints, with physical identity validated by the caller's domain.
 * @param source_arithmetic Canonical original K/V format family, authenticated before physical splitting.
 * @return Immutable source/family identity; all interval construction stays canonical.
 */
inline TensorParallelAssignmentDeclaration resolveOriginalTPDeclaration(int queries,int kv,
    std::vector<OriginalGDNHeadGeometry> gdns,KVHeadCoveragePolicy declared,
    TensorParallelMainPlacement placement,const MTPRuntimeConfig &mtp,DenseParallelPolicy dense,
    std::span<const DeviceId> devices,TensorParallelKVProjectionSource source_arithmetic)
{
    if(declared!=KVHeadCoveragePolicy::Legacy && declared!=KVHeadCoveragePolicy::QueryInterval)
        throw std::invalid_argument("unknown source schema KV consumer policy");
    if(placement!=TensorParallelMainPlacement::FullMainRankLocal && placement!=TensorParallelMainPlacement::OtherDeclared)
        throw std::invalid_argument("unknown TP main consumer placement");
    if(devices.empty())throw std::invalid_argument("TP source declaration requires resolved participant endpoints");
    const bool homogeneous_gpu=std::all_of(devices.begin(),devices.end(),[&](DeviceId d)
        {return d.is_gpu() && d.type==devices.front().type;});
    if(source_arithmetic!=TensorParallelKVProjectionSource::NativeQuantizedOriginalSlices &&
        source_arithmetic!=TensorParallelKVProjectionSource::FloatingFullBank)
        throw std::invalid_argument("unknown original K/V arithmetic source family");
    const bool query=source_arithmetic==TensorParallelKVProjectionSource::NativeQuantizedOriginalSlices &&
        declared==KVHeadCoveragePolicy::QueryInterval &&
        placement==TensorParallelMainPlacement::FullMainRankLocal && homogeneous_gpu &&
        !retainsMTPGraphCapacity(mtp) && dense==DenseParallelPolicy::TensorParallel;
    // These are named, existing retained families, not an unsupported-query
    // recovery path. Only ordinary homogeneous main TP adopts source coverage.
    return TensorParallelAssignmentDeclaration::fromOriginal({queries,kv},
        query?KVHeadCoveragePolicy::QueryInterval:KVHeadCoveragePolicy::Legacy,
        query?std::optional{QueryHeadPartitionDeclaration::fromOriginalConsumers(queries,std::move(gdns))}:std::nullopt);
}
/**
 * @brief Enumerate every original retained main GDN consumer in canonical source order.
 * @param original Already-validated original source metadata, never a participant view.
 * @return Complete original geometry vector used by structural and physical declarations.
 */
inline std::vector<OriginalGDNHeadGeometry> originalQfnTPConsumers(const Qwen4ExpConfig &original)
{
    std::vector<OriginalGDNHeadGeometry> consumers;
    if(original.mixers().size()<original.mainLayers())throw std::invalid_argument("original QFN main consumers are incomplete");
    for(std::size_t layer=0;layer<original.mainLayers();++layer)
        if(original.mixers()[layer]==Qwen4ExpMixer::GatedDeltaNet)
            consumers.push_back({int(original.deltaNet().key_heads),int(original.deltaNet().value_heads)});
    return consumers;
}
/**
 * @brief Project the original planning profile using the same complete source-consumer contract.
 * @param original Canonical serialized source profile, never participant-local dimensions.
 * @param placement Actual rank-plan placement scope.
 * @param mtp Retained family policy.
 * @param dense Declared dense execution policy.
 * @param devices Actual ordered local endpoints.
 * @return Same source declaration as original-directory graph construction.
 */
inline TensorParallelAssignmentDeclaration resolveProfileTPDeclaration(const ModelMemoryProfile &original,
    TensorParallelMainPlacement placement,const MTPRuntimeConfig &mtp,DenseParallelPolicy dense,
    std::span<const DeviceId> devices)
{
    std::vector<OriginalGDNHeadGeometry> consumers;
    if(original.architecture=="qwen4exp" && (!original.flash_next || original.gdn_group_count<=0 || original.gdn_time_step_rank<=0))
        throw std::invalid_argument("QFN TP original main/GDN source geometry is absent");
    if(original.flash_next)
        for(int layer=0;layer<original.mainLayerCount();++layer)
            if(original.flashNextLayerKind(layer)==Qwen4ExpLayerKind::GDN)
                consumers.push_back({original.gdn_group_count,original.gdn_time_step_rank});
    const auto source_arithmetic=original.architecture=="qwen4exp"
        ?originalQfnKVProjectionSource(original):TensorParallelKVProjectionSource::FloatingFullBank;
    const auto schema=SchemaFactoryRegistry::getWeightShardingConfig(original.architecture);
    return resolveOriginalTPDeclaration(original.n_heads,original.n_kv_heads,std::move(consumers),
        schema.main_kv_head_coverage,placement,mtp,dense,devices,source_arithmetic);
}
/**
 * @brief Borrow the original directory and reuse the planning source projection/classifier.
 * @param original Original immutable GGUF directory, never participant-local tensors.
 * @param placement Existing admitted main placement or a separately declared family.
 * @param mtp Retained family policy.
 * @param dense Actual declared dense policy.
 * @param devices Actual ordered local endpoints.
 * @return Same source declaration as PMA and automatic planning; no payload is read/materialized.
 */
inline TensorParallelAssignmentDeclaration resolveQfnTPDeclaration(const GGUFModel &original,
    TensorParallelMainPlacement placement,const MTPRuntimeConfig &mtp,DenseParallelPolicy dense,
    std::span<const DeviceId> devices)
{
    if(original.architecture!="qwen4exp")throw std::invalid_argument("QFN TP directory has a foreign architecture");
    return resolveProfileTPDeclaration(ModelMemoryProfile::fromGGUF(original),placement,mtp,dense,devices);
}
} // namespace llaminar2
