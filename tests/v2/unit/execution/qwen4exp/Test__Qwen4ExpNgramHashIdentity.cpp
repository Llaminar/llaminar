/**
 * @file Test__Qwen4ExpNgramHashIdentity.cpp
 * @brief Replay actual hash-stage capture declarations and stale prepared storage rejection.
 *
 * Native addresses are declarations only: no accelerator, sampled-token value,
 * host mirror or model is needed to prove immutable capture identity totality.
 */
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/Tensors.h"
#include <gtest/gtest.h>
#include <array>
using namespace llaminar2;
/** @brief Every token/history/count/workspace/hash declaration affects the actual stage identity. */
TEST(Qwen4ExpNgramHashIdentity, ExactDeclarationsAndRepeatedReplay){
    INT32Tensor tokens{{4}},history{{2}},otherHistory{{2}},ids{{4,32}},otherIds{{4,32}},count{{1}};
    std::array<std::int32_t,5> native{},other{};std::int32_t live=1,otherLive=1;
    qwen4exp::NgramHashPolicy policy;
    policy.multipliers={3,5,7};for(unsigned i=0;i<16;++i){policy.moduli[i]=17;policy.offsets[i]=i*32;}
    const auto make=[&](unsigned change){
        auto p=policy;if(change==1)++p.multipliers[0];if(change==2)++p.moduli[0];
        if(change==3)++p.offsets[0];if(change==4)++p.eosToken;
        const Qwen4ExpIntInput token=Qwen4ExpNativeIntInput{DeviceId::cpu(),change==5?other.data():native.data(),change==6?5u:4u};
        Qwen4ExpStageRows rows{change==7?3:4,Qwen4ExpIntInput(Qwen4ExpNativeIntInput{DeviceId::cpu(),change==8?&otherLive:&live,1})};
        return Qwen4ExpPrimitiveStage::Params{.rows=rows,.operation=Qwen4ExpPrimitiveStage::NgramHash{
            token,Qwen4ExpIntBinding(change==9?otherHistory:history,BufferId::PREFILL_CHUNK_TOKEN_IDS),
            Qwen4ExpIntBinding(change==10?otherIds:ids,BufferId::QFN_NGRAM_ROW_IDS),p}};
    };
    Qwen4ExpPrimitiveStage original(make(0));const auto identity=original.graphCaptureVariantSignature();ASSERT_NE(identity,0u);
    for(unsigned replay=0;replay<20;++replay){
        native[0]=static_cast<std::int32_t>(replay);live=replay%5;
        static_cast<std::int32_t*>(history.raw_mutable_data())[0]=static_cast<std::int32_t>(replay);
        Qwen4ExpPrimitiveStage same(make(0));EXPECT_EQ(same.graphCaptureVariantSignature(),identity);
        EXPECT_EQ(original.graphCaptureVariantSignature(),identity); // Reset data never changes topology.
    }
    for(unsigned change=1;change<=10;++change){Qwen4ExpPrimitiveStage changed(make(change));EXPECT_NE(changed.graphCaptureVariantSignature(),identity)<<change;}
    auto foreign=make(0);std::get<Qwen4ExpPrimitiveStage::NgramHash>(foreign.operation).tokens=
        Qwen4ExpNativeIntInput{DeviceId::cuda(0),native.data(),4};
    EXPECT_THROW(Qwen4ExpPrimitiveStage(std::move(foreign)),std::invalid_argument);
}
/** @brief Preparation seals workspace addresses; release cannot be hidden by a stable tensor object. */
TEST(Qwen4ExpNgramHashIdentity, PreparedWorkspaceReleaseIsFatal){
    INT32Tensor tokens{{1}},history{{2}},ids{{1,32}};
    qwen4exp::NgramHashPolicy policy;policy.multipliers={3,5,7};
    for(unsigned i=0;i<16;++i){policy.moduli[i]=17;policy.offsets[i]=17*i;}
    Qwen4ExpPrimitiveStage stage({.rows={1,{}},.operation=Qwen4ExpPrimitiveStage::NgramHash{
        Qwen4ExpIntBinding(tokens,BufferId::REQUEST_TOKEN_IDS),Qwen4ExpIntBinding(history,BufferId::PREFILL_CHUNK_TOKEN_IDS),
        Qwen4ExpIntBinding(ids,BufferId::QFN_NGRAM_ROW_IDS),policy}});
    CPUDeviceContext context(DeviceId::cpu(),1);
    const auto identity=stage.graphCaptureVariantSignature();ASSERT_TRUE(stage.prepareGraphLaunch(&context,nullptr));
    EXPECT_EQ(stage.graphCaptureVariantSignature(),identity);
    ids.release_raw_data();EXPECT_THROW(stage.graphCaptureVariantSignature(),std::logic_error);
    EXPECT_THROW(stage.execute(&context),std::logic_error);
}
