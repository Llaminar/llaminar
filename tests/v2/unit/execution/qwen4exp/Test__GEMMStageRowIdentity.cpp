/**
 * @file Test__GEMMStageRowIdentity.cpp
 * @brief Device-free composed HC arithmetic and prefix/request capture identity contracts.
 *
 * Only immutable recording metadata is inspected. No prepared engine, accelerator
 * or model executes; mutable native count words deliberately change between calls.
 */
#include "execution/compute_stages/stages/GEMMStage.h"
#include "tensors/Tensors.h"
#include <gtest/gtest.h>
using namespace llaminar2;
namespace {
/** @brief Full borrowed storage for constructor and capture-metadata validation. */
struct Fixture {
    FP32Tensor input{{9,32}},weight{{2,32}},output{{9,2}};
    Q8_0Tensor quantized{{2,32},std::vector<std::uint8_t>(2*sizeof(Q8_0Block))};
    std::int32_t counts[3]{0,1,2};
    /** @return Plain decode-equivalent floating projection, with no prepared/hot resources. */
    GEMMStage::Params params(int rows=9) {
        return {.device_id=DeviceId::cpu(),.A=&input,.B=&weight,.C=&output,
            .m=rows,.n=2,.k=32,.force_decode_equivalent_verifier_prefill=true};
    }
};
}
/** @brief Count pointer/capacity/slice/repeat are identity; count contents are execution state. */
TEST(GEMMStageRowIdentity, PrefixM1AndExpandedViewsKeepCompleteImmutableIdentity) {
    Fixture f;auto p=f.params(1);p.verifier_row_range=DeviceRowRange::deviceCounted(1,f.counts);
    GEMMStage m1(p);const auto identity=m1.graphCaptureVariantSignature();ASSERT_NE(identity,0u);
    for(int live:{0,1,0}){f.counts[0]=live;EXPECT_EQ(m1.graphCaptureVariantSignature(),identity);}
    p.verifier_row_range=DeviceRowRange::deviceCounted(1,f.counts+1);
    EXPECT_NE(GEMMStage(p).graphCaptureVariantSignature(),identity);
    p=f.params(6);p.verifier_row_range=DeviceRowRange::deviceCounted(3,f.counts).repeatEachRow(2);
    const auto repeated=GEMMStage(p).graphCaptureVariantSignature();
    p.verifier_row_range=DeviceRowRange::deviceCounted(6,f.counts);
    EXPECT_NE(GEMMStage(p).graphCaptureVariantSignature(),repeated);
    p.verifier_row_range=DeviceRowRange::deviceCounted(9,f.counts).slice(1,6);
    const auto sliced=GEMMStage(p).graphCaptureVariantSignature();
    p.verifier_row_range=DeviceRowRange::deviceCounted(9,f.counts).slice(2,6);
    EXPECT_NE(GEMMStage(p).graphCaptureVariantSignature(),sliced);
    p=f.params();EXPECT_EQ(GEMMStage(p).graphCaptureVariantSignature(),0u);
}
/** @brief Request holes are a distinct alternative; prefix overlap and unsupported arithmetic fail at setup. */
TEST(GEMMStageRowIdentity, RequestOwnersAndGeometryAreDistinctAndTwoTermHolesFailBeforeCapture) {
    Fixture f;auto p=f.params();p.request_rows=DeviceRequestRowRanges::deviceCounted(3,3,f.counts);
    GEMMStage stage(p);const auto identity=stage.graphCaptureVariantSignature();ASSERT_NE(identity,0u);
    f.counts[0]=3;f.counts[1]=0;f.counts[2]=1;EXPECT_EQ(stage.graphCaptureVariantSignature(),identity);
    p.request_rows=DeviceRequestRowRanges::deviceCounted(1,9,f.counts);
    EXPECT_NE(GEMMStage(p).graphCaptureVariantSignature(),identity);
    p.request_rows=DeviceRequestRowRanges::deviceCounted(3,3,f.counts+1);
    EXPECT_NE(GEMMStage(p).graphCaptureVariantSignature(),identity);
    p.verifier_row_range=DeviceRowRange::deviceCounted(9,f.counts);EXPECT_ANY_THROW(GEMMStage{p});
    p.verifier_row_range.reset();p.quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8;
    EXPECT_THROW(GEMMStage{p},std::invalid_argument);
    p.quantized_operand_precision=QuantizedActivationOperandPrecision::SingleQ8;
    p.force_decode_equivalent_verifier_prefill=false;EXPECT_ANY_THROW(GEMMStage{p});
}
/** @brief Composing row identity must retain the qualified HC arithmetic/quantizer scalar identity. */
TEST(GEMMStageRowIdentity, HCPrecisionAndQuantizerIdentityRemainExplicit) {
    Fixture f;auto p=f.params(1);p.B=&f.quantized;p.verifier_row_range=DeviceRowRange::deviceCounted(1,f.counts);
    const auto single=GEMMStage(p).graphCaptureVariantSignature();
    p.quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8;
    const auto paired=GEMMStage(p).graphCaptureVariantSignature();EXPECT_NE(paired,single);
    for(int live:{0,1,0}){f.counts[0]=live;EXPECT_EQ(GEMMStage(p).graphCaptureVariantSignature(),paired);}
    p.device_id=DeviceId::cuda(0);EXPECT_NE(GEMMStage(p).graphCaptureVariantSignature(),paired);
    const auto cuda=GEMMStage(p).graphCaptureVariantSignature();
    p.device_id=DeviceId::rocm(0);EXPECT_NE(GEMMStage(p).graphCaptureVariantSignature(),cuda);
}
