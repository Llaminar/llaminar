/**
 * @file FullNativeHCFFNFamilyReplay.inc
 * @brief One source-authenticated staged/full replay and paired timing method for all decoder unions.
 *
 * The existing source factory, WeightManager, runtime bank, PMA workspace and
 * production graph owner supply both arms. Recipes describe genuine source
 * codebooks; the actual immutable declaration and prepared descriptors must
 * select the expected smallest decoder union. Unsupported external-engine
 * paths are never synthesized. Timing uses only independent-input final reads
 * before the original collective; middle/aliased replay remains correctness-only.
 */
#include "../../fixtures/qwen4exp/FullNativeHCFFNFamilyCases.h"

namespace llaminar2::test {
namespace {
/**
 * @brief Resolve a recipe through the original iterable format/source contracts.
 * @param recipe Exact semantic gate/down/shared sources.
 * @param shape Explicit original local geometry, validated for every source K.
 * @return Existing source factory with independent actual role/source pointers.
 */
SourceFormat fullNativeHCFFNFamilySource(const FullNativeHCFFNFamilyCase& recipe,OriginalFFNFixtureGeometry shape) {
    const auto& gate=quantizedVerifierFormat(recipe.gate);
    const auto& down=quantizedVerifierFormat(recipe.down);
    const auto& shared=quantizedVerifierFormat(recipe.shared);
    const auto& shared_down = recipe.shared_down ? quantizedVerifierFormat(recipe.shared_down) : shared;
    requireQuantizedVerifierSourceShape(shared_down, shape.hidden, shape.shared);
    requireQuantizedVerifierSourceShape(gate,shape.routed,shape.hidden);
    requireQuantizedVerifierSourceShape(down,shape.hidden,shape.routed);
    requireQuantizedVerifierSourceShape(shared,shape.shared,shape.hidden);
    NativeQuantisedExecutionFormatEnvelope envelope;
    const auto add=[&](const QuantizedVerifierFormatCase& source) {
        require(addNativeQuantisedExecutionFormat(envelope,source.device_execution_codebook_id,
            {source.source_codebook_id,source.source_is_superblock,true},false),
            "family recipe has an original supported source/migration envelope");
    };
    // This roster has only a GPU. A representation materialized specifically
    // on CPU cannot widen the reachable formats of its published expert bank.
    add(gate);
    add(down);
    add(shared);
    add(shared_down);
    require(rocm::ResidentMoEFFNDecoderProgram::fromExecutionMask(envelope.execution_codebook_mask).groups()==recipe.groups,
        "family recipe's actual source envelopes select its exact smallest decoder union");
    const auto sources=sourceFormats(OriginalSourceQualification::ResidentFFN);
    const auto original=std::find_if(sources.begin(),sources.end(),[&](const auto& source){return source.type==gate.tensor_type;});
    require(original!=sources.end(),"family recipe retains the original source factory");
    auto result=*original;result.name=recipe.name;result.down_quantized=&down;result.shared_quantized=&shared;
    if (recipe.shared_down) result.shared_down_quantized = &shared_down;
    result.expected_native_layout_groups=static_cast<std::uint8_t>(recipe.groups);return result;
}
/**
 * @brief Report logical once-per-selected-weight traffic and arithmetic for this input.
 * @param recipe Encodings of the six original FFN roles.
 * @param shape Positive source-aligned matrix widths and distinct selected expert count.
 * @details The storage authority supplies the prepared plane widths. These figures
 * exclude HC/router, activations, hardware caching and speculative loads; the
 * measured counter for memory traffic is independent of both logical byte counts.
 */
void reportFullNativeFamilyWork(const FullNativeHCFFNFamilyCase& recipe, OriginalFFNFixtureGeometry shape) {
    struct Role {
        const char* format;
        std::size_t rows, columns, instances;
    };
    const std::size_t h = shape.hidden, f = shape.routed, s = shape.shared, k = shape.top_k;
    const std::array roles{
        Role{recipe.gate, f, h, k}, Role{recipe.gate, f, h, k}, Role{recipe.down, h, f, k},
        Role{recipe.shared, s, h, 1}, Role{recipe.shared, s, h, 1},
        Role{recipe.shared_down ? recipe.shared_down : recipe.shared, h, s, 1}};
    std::size_t source_bytes = 0, prepared_bytes = 0, macs = 0;
    for (const auto& role : roles) {
        const auto& encoding = quantizedVerifierFormat(role.format);
        const auto* native = native_quantised_formats::forSourceIdentity(
            encoding.source_codebook_id, encoding.source_is_superblock);
        require(native != nullptr, "original FFN source has a physical packing declaration");
        const GGUFTensorInfo original{.type = PlanningGGUFFixture::sourceType(encoding.tensor_type)};
        const auto elements = role.rows * role.columns * role.instances;
        source_bytes += (elements / original.getBlockSize()) * original.getTypeSize();
        prepared_bytes += role.instances * nativeQuantisedPackedRegionSizes(
            role.rows, role.columns, *native).logicalBytes();
        macs += elements;
    }
    std::osyncstream(std::cout) << "FULL_NATIVE_FAMILY_WORK,groups=" << recipe.groups
        << ",ffn_macs=" << macs << ",original_ffn_bytes=" << source_bytes
        << ",prepared_ffn_bytes=" << prepared_bytes << ",selected_experts=" << k << '\n';
}

/**
 * @brief Replay and optionally time one actual source family on the selected original PCI owner.
 * @param recipe Genuine source-format declaration. @param shape Original local geometry.
 * @param timing Paired complete 32-operation observation after exact 20-history/reset/empty checks.
 * @param observation Explicit observed replay policy; timing always uses demanded-only production.
 */
void replayFullNativeHCFFNFamily(const FullNativeHCFFNFamilyCase& recipe,OriginalFFNFixtureGeometry shape,bool timing,
    FullProductionHCObservation observation=FullProductionHCObservation::ObservedAll) {
    const auto source=fullNativeHCFFNFamilySource(recipe,shape);
    const auto replay=[&](const auto& devices) {
        const auto mpi=MPIContextFactory::global();
        require(mpi&&mpi->clusterInventory()&&devices.size()==1&&devices.front().is_rocm(),
            "family replay has its original singleton ROCm inventory owner");
        const auto& rank=mpi->clusterInventory()->getRank(mpi->rank());
        require(rank.rank==mpi->rank(),"family replay inventory resolves the actual original rank identity");
        const auto observed=std::find_if(rank.gpus.begin(),rank.gpus.end(),[&](const auto& gpu) {
            return gpu.type==DeviceType::ROCm&&gpu.local_device_id==devices.front().ordinal;
        });
        require(observed!=rank.gpus.end()&&rankOwnsGPUDevice(rank,*observed)&&!observed->pci_bus_address.empty()&&
            std::string_view(observed->native_isa).substr(0,observed->native_isa.find(':'))=="gfx906",
            "family replay selected device belongs to its exact original rank and native ISA");
        std::osyncstream(std::cout)<<"FULL_NATIVE_FAMILY_DEVICE,groups="<<recipe.groups<<",rank="<<mpi->rank()
            <<",native_ordinal="<<observed->local_device_id<<",pci="<<observed->pci_bus_address
            <<",native_isa="<<observed->native_isa<<",H="<<shape.hidden<<",F="<<shape.routed<<",S="<<shape.shared<<'\n';
        if (timing) reportFullNativeFamilyWork(recipe, shape);
        proveOriginalFinalization(devices,false,OriginalSourceQualification::FullNativeHCFFNFinal,&source,
            LocalFFNProgram::Original,shape,CastagnaPhaseObservation::Disabled,
            FullProductionHCProbe::Disabled,FullProductionHCStorageReuse::Distinct,timing,
            timing?FullProductionHCObservation::DemandedOnly:observation);
    };
    const char* requested = std::getenv("LLAMINAR_TEST_RETAINED_HC_FFN_PCI");
    const std::array<std::string_view, 1> addresses{requested ? requested : ""};
    originalRankLocalReplay(DeviceType::ROCm, 1, replay, "gfx906",
        addresses.front().empty() ? std::span<const std::string_view>{} : std::span<const std::string_view>{addresses});
}
/** @brief Shared canonical geometry admits every genuine 256-element source block. */
OriginalFFNFixtureGeometry fullNativeHCFFNFamilyGeometry() {
    return {.hidden=2560,.routed=256,.shared=256,.experts=512,.top_k=10,.router_source=OriginalFFNRouterSource::Dense};
}
/** @brief Each source recipe uses one identical production replay and event protocol. */
class FullNativeHCFFNFamily : public ::testing::TestWithParam<FullNativeHCFFNFamilyCase> {};
/** @test Original staged public/state bytes certify the complete native kernel. */
TEST_P(FullNativeHCFFNFamily,Replay) {
    replayFullNativeHCFFNFamily(GetParam(),fullNativeHCFFNFamilyGeometry(),false);
}
/** @test Every physical source family executes the optimized demanded-only kernel and validates its consumer-visible writes. */
TEST_P(FullNativeHCFFNFamily,DemandedReplay) {
    replayFullNativeHCFFNFamily(GetParam(),fullNativeHCFFNFamilyGeometry(),false,FullProductionHCObservation::DemandedOnly);
}
/** @test Optional unprofiled event economy; never substitutes for the preceding exact replay. */
TEST_P(FullNativeHCFFNFamily,DISABLED_Events) {
    replayFullNativeHCFFNFamily(GetParam(),fullNativeHCFFNFamilyGeometry(),true);
}
INSTANTIATE_TEST_SUITE_P(OriginalSources,FullNativeHCFFNFamily,::testing::ValuesIn(fullNativeHCFFNFamilyCases()),
    [](const ::testing::TestParamInfo<FullNativeHCFFNFamilyCase>& info){return info.param.name;});
} // namespace
} // namespace llaminar2::test
