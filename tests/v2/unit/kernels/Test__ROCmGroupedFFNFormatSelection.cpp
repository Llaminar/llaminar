/**
 * @file Test__ROCmGroupedFFNFormatSelection.cpp
 * @brief Device-free admission tests for the counted NativeQuantised verifier.
 * Descriptor pointers identify inert storage: cold admission may inspect their
 * presence and source metadata but must never read, allocate or copy payloads.
 */
#include "kernels/rocm/gemm/ROCmGroupedFFNFormatSelection.h"
#include <gtest/gtest.h>
#include <type_traits>

namespace {
using namespace llaminar2;
namespace ffn=rocm::grouped_ffn;
namespace formats=native_quantised_formats;

constexpr int shared=256;
constexpr unsigned char plane=0;

/** @return Independently prepared gate, up or down metadata for the supplied representation. */
DeviceNativeQuantisedMatrixDesc matrix(const NativeQuantisedFormatInfo& source,std::uint8_t physical,bool down,int columns=shared)
{
    return {.payload=&plane,.scales=&plane,
        .mins=NativeQuantisedDescriptorValidation::secondary(physical)?&plane:nullptr,
        .emins=physical==10?&plane:nullptr,.n=down?ffn::hidden:columns,.k=down?columns:ffn::hidden,
        .blocks_per_row=unsigned(down?columns:ffn::hidden)/32,.codebook_id=physical,
        .source_codebook_id=source.codebook_id,.source_is_superblock=source.is_superblock,
        .source_identity_present=1};
}

/** @return Authoritative routed arrival envelope with optional CPU-migration representations. */
NativeQuantisedExecutionFormatEnvelope reachable(const NativeQuantisedFormatInfo& source,bool cpuMigration=true)
{
    NativeQuantisedExecutionFormatEnvelope result{};
    EXPECT_TRUE(addNativeQuantisedExecutionFormat(result,canonicalDeviceVnniCodebookId(source),
        {source.codebook_id,source.is_superblock,true},cpuMigration));
    return result;
}

/** @return The original expert's per-role identities, before compact or promoted device packing. */
auto sources(const NativeQuantisedFormatInfo& gate,const NativeQuantisedFormatInfo& up,
             const NativeQuantisedFormatInfo& down)
{
    const auto id=[](const NativeQuantisedFormatInfo& f){return NativeQuantisedSourceIdentity{f.codebook_id,f.is_superblock,true};};
    return std::array{id(gate),id(up),id(down)};
}

TEST(ROCmGroupedFFNFormatSelection, AllSourceAndMigrationFormatsRetainIndependentRolePolicies)
{
    const auto q8=reachable(formats::Q8_0);
    for(const auto& entry:formats::kAllSourceFormats) {
        SCOPED_TRACE(entry.quant_type);
        const auto& source=*entry.metadata;
        const auto routed=reachable(source);
        const auto policy=nativeQuantisedCodebookMaskBit(canonicalDeviceVnniCodebookId(source.codebook_id));
        const auto physical=canonicalDeviceVnniCodebookId(source);
        for(unsigned role=0;role<3;++role) {
            std::array weights{matrix(formats::Q8_0,19,false),matrix(formats::Q8_0,19,false),
                               matrix(formats::Q8_0,19,true)};
            weights[role]=matrix(source,physical,role==2);
            const auto program=ffn::declareSourceFormats(160,shared,routed,q8,sources(source,source,formats::Q8_0),weights);
            EXPECT_EQ(program.routedColumns(),160);
            EXPECT_EQ(program.sharedColumns(),shared);
            EXPECT_EQ(program.formats().routedGateUp.policy_codebook_mask,policy);
            EXPECT_EQ(program.formats().routedDown.execution_codebook_mask,q8.execution_codebook_mask);
            for(unsigned index=0;index<3;++index) {
                EXPECT_EQ(program.formats().shared[index].execution_codebook_mask,
                    index==role?nativeQuantisedCodebookMaskBit(physical):q8.execution_codebook_mask);
                EXPECT_EQ(program.formats().shared[index].policy_codebook_mask,index==role?policy:q8.policy_codebook_mask);
            }
            EXPECT_EQ(program.family(),rocm::ResidentMoEFFNDecoderProgram::fromExecutionMask(
                routed.execution_codebook_mask|q8.execution_codebook_mask|nativeQuantisedCodebookMaskBit(physical)).groups());
        }
    }
}

/** @test An indexed gate schedule depends on its gate/up readers, including every encoding reachable by movement. */
TEST(ROCmGroupedFFNFormatSelection, IndexedGateReadersAreIndependentOfDownAndRejectMixedEncodings)
{
    const auto& sharedGate=formats::IQ2_S;
    for(const auto* gate_format:{&formats::IQ3_XXS,&formats::Q5_K})
    for(const auto* down:{&formats::Q5_1,&formats::Q6_K}) {
        const auto& gate=*gate_format;
        SCOPED_TRACE(::testing::Message()<<int(gate.codebook_id)<<' '<<int(down->codebook_id));
        const std::array weights{matrix(sharedGate,sharedGate.codebook_id,false),
            matrix(sharedGate,sharedGate.codebook_id,false),matrix(*down,down->codebook_id,true)};
        const auto declare=[&](bool movable) {
            return ffn::declareSourceFormats(256,shared,reachable(gate,movable),reachable(*down),
                sources(gate,gate,*down),weights);
        };
        const auto compact=declare(false);
        EXPECT_TRUE(compact.hasOriginalGateReaders(gate,sharedGate,weights));
        EXPECT_TRUE(declare(true).hasOriginalGateReaders(gate,sharedGate,weights));
        EXPECT_FALSE(compact.hasOriginalGateReaders(formats::IQ3_S,sharedGate,weights));
        for(unsigned role=0;role<2;++role) {
            auto changed=weights;
            changed[role].source_identity_present=0;
            EXPECT_FALSE(compact.hasOriginalGateReaders(gate,sharedGate,changed));
            changed[role]=matrix(formats::Q4_0,formats::Q4_0.codebook_id,false);
            EXPECT_FALSE(compact.hasOriginalGateReaders(gate,sharedGate,changed));
        }
    }
}

TEST(ROCmGroupedFFNFormatSelection, RejectsMissingRolesSourcePlanesAndInvalidWidths)
{
    const auto q8=reachable(formats::Q8_0);
    std::array weights{matrix(formats::Q8_0,19,false),matrix(formats::Q8_0,19,false),matrix(formats::Q2_K,10,true)};
    EXPECT_THROW(ffn::declareSourceFormats(160,0,q8,q8,sources(formats::Q8_0,formats::Q8_0,formats::Q8_0),weights),std::invalid_argument);
    EXPECT_THROW(ffn::declareSourceFormats(33,shared,q8,q8,sources(formats::Q8_0,formats::Q8_0,formats::Q8_0),weights),std::invalid_argument);
    EXPECT_THROW(ffn::declareSourceFormats(160,shared,{},q8,sources(formats::Q8_0,formats::Q8_0,formats::Q8_0),weights),std::invalid_argument);
    EXPECT_THROW(ffn::declareSourceFormats(160,shared,q8,{},sources(formats::Q8_0,formats::Q8_0,formats::Q8_0),weights),std::invalid_argument);
    EXPECT_THROW(ffn::declareSourceFormats(160,shared,{1u<<31,1},q8,sources(formats::Q8_0,formats::Q8_0,formats::Q8_0),weights),std::invalid_argument);
    weights[2].emins=nullptr;
    EXPECT_THROW(ffn::declareSourceFormats(160,shared,q8,q8,sources(formats::Q8_0,formats::Q8_0,formats::Q8_0),weights),std::invalid_argument);
    weights[2]=matrix(formats::Q2_K,10,false);
    EXPECT_THROW(ffn::declareSourceFormats(160,shared,q8,q8,sources(formats::Q8_0,formats::Q8_0,formats::Q8_0),weights),std::invalid_argument);
    weights[2]=matrix(formats::Q2_K,10,true);
    weights[1].source_identity_present=0;
    EXPECT_THROW(ffn::declareSourceFormats(160,shared,q8,q8,sources(formats::Q8_0,formats::Q8_0,formats::Q8_0),weights),std::invalid_argument);
}

/** @test Equal physical IDs cannot replace the original role or manufacture another reachable decoder. */
TEST(ROCmGroupedFFNFormatSelection, SealsOriginalRoutedRolesAndRejectsForeignEnvelopes)
{
    const auto q8=reachable(formats::Q8_0);
    const auto q4=reachable(formats::Q4_K);
    const std::array weights{matrix(formats::Q8_0,19,false),matrix(formats::Q8_0,19,false),matrix(formats::Q8_0,19,true)};
    auto original=sources(formats::Q4_K,formats::Q4_1,formats::Q8_0);
    const auto prepared=ffn::declareSourceFormats(160,shared,q4,q8,original,weights);
    EXPECT_EQ(prepared.routedSources(),original);
    EXPECT_NE(prepared.routedSources()[0],prepared.routedSources()[1]);
    original[0].present=false;
    EXPECT_THROW(ffn::declareSourceFormats(160,shared,q4,q8,original,weights),std::invalid_argument);
    original=sources(formats::Q4_K,formats::Q4_1,formats::Q6_K);
    EXPECT_THROW(ffn::declareSourceFormats(160,shared,q4,q8,original,weights),std::invalid_argument);
    original=sources(formats::Q4_K,formats::Q4_1,formats::Q8_0);
    auto foreign=q4;
    foreign.execution_codebook_mask|=nativeQuantisedCodebookMaskBit(12);
    EXPECT_THROW(ffn::declareSourceFormats(160,shared,foreign,q8,original,weights),std::invalid_argument);
}

#ifdef HAVE_ROCM
/** @test The prepared six counts use the actual serial geometry, including independent routed and shared widths. */
TEST(ROCmGroupedFFNFormatSelection, ResolvesIndependentSerialTreesBeforeCapture)
{
    const auto gate=reachable(formats::Q4_K);
    const auto down=reachable(formats::Q5_1);
    const std::array weights{matrix(formats::Q8_0,19,false),matrix(formats::Q8_0,19,false),matrix(formats::Q8_0,19,true)};
    const auto original=sources(formats::Q4_K,formats::Q4_K,formats::Q5_1);
    const auto program=ffn::SourceFormatProgram::prepare(ffn::declareSourceFormats(160,shared,gate,down,original,weights));
    EXPECT_EQ(program.routedPartitions(),(std::array<int,3>{16,16,5}));
    EXPECT_EQ(program.sharedPartitions(),(std::array<int,3>{16,16,8}));
}

TEST(ROCmGroupedFFNFormatSelection, PairsGateAndUpOnlyForMeasuredPhysicalReader)
{
    const std::array weights{matrix(formats::Q8_0,19,false),matrix(formats::Q8_0,19,false),matrix(formats::Q8_0,19,true)};
    for (const auto* gate:{&formats::Q4_K,&formats::Q5_K}) {
        const auto layout=ffn::declareSourceFormats(160,shared,reachable(*gate,false),reachable(formats::Q8_0,false),
            sources(*gate,*gate,formats::Q8_0),weights);
        EXPECT_EQ(ffn::SourceFormatProgram::prepare(layout).pairsGateAndUp(),gate==&formats::Q4_K);
    }
    auto differentShared=weights;
    differentShared[1]=matrix(formats::IQ4_NL,formats::IQ4_NL.codebook_id,false);
    const auto layout=ffn::declareSourceFormats(160,shared,reachable(formats::Q4_K,false),reachable(formats::Q8_0,false),
        sources(formats::Q4_K,formats::Q4_K,formats::Q8_0),differentShared);
    EXPECT_FALSE(ffn::SourceFormatProgram::prepare(layout).pairsGateAndUp());
}

/** @test Packed pitch selection preserves original source and reachable physical-reader identity. */
TEST(ROCmGroupedFFNFormatSelection, FixedQ5PitchRequiresMeasuredWidthsAndExactGateSources)
{
    const auto q8=reachable(formats::Q8_0,false);
    const auto q5=reachable(formats::Q5_K,false);
    const auto q5k=sources(formats::Q5_K,formats::Q5_K,formats::Q8_0);
    const auto sharedWeights=[](int columns,const NativeQuantisedFormatInfo& up) {
        return std::array{matrix(formats::Q8_0,19,false,columns),matrix(up,19,false,columns),
            matrix(formats::Q8_0,19,true,columns)};
    };
    const auto classify=[&](int routed,int shared,const auto& format,const auto& routedSources,const auto& weights) {
        const auto declaration=ffn::declareSourceFormats(routed,shared,format,q8,routedSources,weights);
        return ffn::SourceFormatProgram::prepare(declaration).fixedQ5GatePitch(weights);
    };
    const auto exact=sharedWeights(160,formats::Q8_0);
    for(int width:{128,160,256})
        EXPECT_EQ(classify(width,160,q5,q5k,exact),static_cast<ffn::FixedQ5GatePitch>(width));
    for(int width:{96,192,320,384})
        EXPECT_FALSE(classify(width,160,q5,q5k,exact));
    for(int width:{128,320})
        EXPECT_FALSE(classify(160,width,q5,q5k,sharedWeights(width,formats::Q8_0)));
    EXPECT_FALSE(classify(160,160,q5,sources(formats::Q5_K,formats::Q5_1,formats::Q8_0),exact));
    EXPECT_FALSE(classify(160,160,q5,q5k,sharedWeights(160,formats::Q8_K)));
    const auto movable_q5 = reachable(formats::Q5_K);
    EXPECT_EQ(movable_q5.execution_codebook_mask,q5.execution_codebook_mask);
    EXPECT_EQ(movable_q5.policy_codebook_mask,q5.policy_codebook_mask);
    for(int width:{128,160,256})
        EXPECT_EQ(classify(width,160,movable_q5,q5k,exact),static_cast<ffn::FixedQ5GatePitch>(width));
}

/** @test A fixed down reader must remain correct for every descriptor reachable after a move. */
TEST(ROCmGroupedFFNFormatSelection, PairedDownRequiresOriginalSourcePitchAndSingleReader)
{
    const auto q4=reachable(formats::Q4_K,false);
    const auto q51=reachable(formats::Q5_1,false);
    const auto original=sources(formats::Q4_K,formats::Q4_K,formats::Q5_1);
    const auto select=[&](int width,int sharedWidth,const auto& routed,const auto& gate,
                          const auto& down,const NativeQuantisedFormatInfo& sharedDown) {
        const std::array weights{matrix(formats::Q8_0,19,false,sharedWidth),
                                 matrix(formats::Q8_0,19,false,sharedWidth),
                                 matrix(sharedDown,19,true,sharedWidth)};
        const auto declaration=ffn::declareSourceFormats(width,sharedWidth,gate,down,routed,weights);
        return ffn::SourceFormatProgram::prepare(declaration).fixedQ51DownPitch(weights[2]);
    };
    for(int width:{128,256})
        EXPECT_EQ(select(width,160,original,q4,q51,formats::Q8_0),static_cast<ffn::FixedQ51DownPitch>(width));
    for(int width:{64,96,160,320,384,640})
        EXPECT_FALSE(select(width,160,original,q4,q51,formats::Q8_0));
    EXPECT_FALSE(select(128,320,original,q4,q51,formats::Q8_0));
    EXPECT_FALSE(select(128,160,original,q4,q51,formats::Q8_K));
    EXPECT_FALSE(select(256,160,sources(formats::Q4_1,formats::Q4_K,formats::Q5_1),q4,q51,formats::Q8_0));
    EXPECT_FALSE(select(256,160,sources(formats::Q4_K,formats::Q4_K,formats::Q5_K),
                        q4,reachable(formats::Q5_K,false),formats::Q8_0));
    EXPECT_EQ(select(256,160,original,reachable(formats::Q4_K,true),q51,formats::Q8_0),
              ffn::FixedQ51DownPitch::F256);
    // CPU migration preserves the plain Q5_1 physical reader and K32 policy.
    const auto movable_q51 = reachable(formats::Q5_1,true);
    EXPECT_EQ(movable_q51.execution_codebook_mask,q51.execution_codebook_mask);
    EXPECT_EQ(movable_q51.policy_codebook_mask,q51.policy_codebook_mask);
    for(int width:{128,256})
        EXPECT_EQ(select(width,160,original,q4,movable_q51,formats::Q8_0),
                  static_cast<ffn::FixedQ51DownPitch>(width));
}

#endif

TEST(ROCmGroupedFFNFormatSelection, CapturesMeasuredDecoderOnlyForSupportedSourceAliases)
{
    const auto source=[](const NativeQuantisedFormatInfo& format) {
        return ExpertWeightFormat::nativeQuantised({format.codebook_id,format.is_superblock,true});
    };
    std::array routed{source(formats::Q5_K),source(formats::Q5_K),source(formats::Q8_0)};
    const auto gate=reachable(formats::Q5_K,false);
    const auto down=reachable(formats::Q8_0,false);
    const auto select=[&] {return ffn::measuredDecoder(160,gate,down,routed);};
    EXPECT_EQ(select(),ffn::RoutedDecoder::Q5K_Q80);
    for (int role:{0,1}) {
        routed[role]=source(formats::Q5_1);
        EXPECT_FALSE(select()) << "same physical decoder, different source on role " << role;
        routed[role]=source(formats::Q5_K);
    }
    routed[2]=source(formats::Q8_1);
    EXPECT_FALSE(select());
}

TEST(ROCmGroupedFFNFormatSelection, MeasuredSumAdmissionDistinguishesSourceAliases)
{
    const auto source=[](const NativeQuantisedFormatInfo& f) {
        return ExpertWeightFormat::nativeQuantised({f.codebook_id,f.is_superblock,true});
    };
    std::array routed{source(formats::Q4_K),source(formats::Q4_K),source(formats::Q5_1)};
    std::array weights{matrix(formats::Q8_0,19,false),matrix(formats::Q8_0,19,false),matrix(formats::Q8_0,19,true)};
    auto matches=[&] {return ffn::hasMeasuredSumLayout(ffn::RoutedGeometry::F160,
        ffn::RoutedDecoder::Q4K_Q51,routed,weights);};
    EXPECT_TRUE(matches());
    routed[1]=source(formats::Q4_1);
    EXPECT_FALSE(matches());
    routed[1]=routed[0];
    routed[2]=source(formats::Q5_K);
    EXPECT_FALSE(matches());
    routed[2]=source(formats::Q5_1);
    weights[0]=matrix(formats::Q8_K,19,false);
    EXPECT_FALSE(matches());
}
} // namespace


/** Raw runtime pointers cannot issue the receipt accepted by the numerical launcher. */
static_assert(!std::is_constructible_v<ffn::BoundSourceFormatProgram,const ffn::Program&,
    llaminar2::DeviceMoELayerRuntime*,int>);

/**
 * @test Every compiled family covers the envelope that selects it. The
 * linear | iq3/iq4 | iq1/iq2 family omits MXFP4 (gfx906 register budget), so
 * MXFP4 with exactly those groups selects the complete union; MXFP4 elsewhere
 * keeps its smallest group union.
 */
TEST(ROCmGroupedFFNFormatSelection, ResidentFamiliesCoverTheirEnvelopeIncludingMXFP4)
{
    using Program=llaminar2::rocm::ResidentMoEFFNDecoderProgram;
    const std::uint32_t mxfp4=1u<<llaminar2::kNativeQuantisedMXFP4Codebook;
    EXPECT_EQ(Program::familyMask<14>()&mxfp4,0u);
    EXPECT_NE(Program::familyMask<4>()&mxfp4,0u);
    EXPECT_NE(Program::familyMask<15>()&mxfp4,0u);
    const std::uint32_t linear=1u<<0,iq1=1u<<13,affine=1u<<5;
    EXPECT_EQ(Program::fromExecutionMask(mxfp4).groups(),4u);
    EXPECT_EQ(Program::fromExecutionMask(mxfp4|iq1).groups(),12u);
    EXPECT_EQ(Program::fromExecutionMask(mxfp4|linear|iq1).groups(),15u);
    EXPECT_EQ(Program::fromExecutionMask(mxfp4|linear|iq1|affine).groups(),15u);
    EXPECT_EQ(Program::fromExecutionMask((1u<<4)|linear|iq1).groups(),14u);
    // Exhaustive over every supported decoder combination of up to three.
    std::vector<std::uint32_t> bits;
    for(unsigned b=0;b<32;++b)if(Program::all_decoders&(1u<<b))bits.push_back(1u<<b);
    for(std::size_t i=0;i<bits.size();++i)for(std::size_t j=i;j<bits.size();++j)for(std::size_t k=j;k<bits.size();++k)
    {
        const auto required=bits[i]|bits[j]|bits[k];
        const auto program=Program::fromExecutionMask(required);
        EXPECT_EQ(required&~program.executionMask(),0u) << required;
        EXPECT_EQ(program.executionMask(),Program::familyMaskFor(program.groups()));
    }
}
