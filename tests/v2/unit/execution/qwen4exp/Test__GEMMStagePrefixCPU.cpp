/**
 * @file Test__GEMMStagePrefixCPU.cpp
 * @brief Device-free actual prepared M1 projection preserves original native counts.
 *
 * Compile/run only against the coherently composed prepared floating/HC ABI.
 * The real prepared store and GEMMStage execute0/1/0 with poisoned inactive
 * source/output bytes; no mock kernel, accelerator or model is involved.
 * Typed-bias regressions use exact basis-vector references and original tensor
 * slices to check ordinary and counted projection semantics independently.
 */
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/qwen4exp/QwenSparseMath.h"
#include "tensors/Tensors.h"
#include "tensors/TensorSlice.h"
#include "utils/OpenMPUtils.h"
#include "../../../utils/PreparedWeightTestHarness.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
using namespace llaminar2;
/** @brief Exact native countM1 is live execution state even though capacity is one. */
TEST(GEMMStagePrefixCPU, RealPreparedFloatingFamiliesPreserveZeroOneZeroAndPoisonBytes)
{
    llaminar::v2::ThreadCountGuard threads(1);CPUDeviceContext context(DeviceId::cpu(),1);
    constexpr int n=33,k=257,column=37;
    for(TensorType type:{TensorType::FP32,TensorType::FP16,TensorType::BF16}) {
        FP32Tensor input({1,k}),output({1,n}),floatWeight({n,k});FP16Tensor halfWeight({n,k});BF16Tensor bfloatWeight({n,k});
        TensorBase*weight=type==TensorType::FP32?static_cast<TensorBase*>(&floatWeight):
            type==TensorType::FP16?static_cast<TensorBase*>(&halfWeight):static_cast<TensorBase*>(&bfloatWeight);
        for(int row=0;row<n;++row)for(int col=0;col<k;++col) {
            const float value=float((row+col)%13-6)/16.0f;const int index=row*k+col;
            if(type==TensorType::FP32)static_cast<float*>(weight->raw_mutable_data())[index]=value;
            else static_cast<std::uint16_t*>(weight->raw_mutable_data())[index]=type==TensorType::FP16 ?
                qwen4exp::detail::encodeHalf(value):std::uint16_t(qwen4exp::detail::floatBits(value)>>16);
        }
        auto prepared=test::makePreparedGemmFixture(weight,DeviceId::cpu(),"qsa.indexer.m1.counted.cpu",ModelContextId{99291+static_cast<unsigned>(type)});
        std::int32_t count=0;
        GEMMStage stage({.device_id=DeviceId::cpu(),.A=&input,.B=weight,.C=&output,.m=1,.n=n,.k=k,
            .force_decode_equivalent_verifier_prefill=true,.verifier_row_range=DeviceRowRange::deviceCounted(1,&count),
            .prepared_ref=prepared.ref,.prepared_store=prepared.store.get()});
        std::string error;ASSERT_TRUE(stage.validatePreparedWeights(&error))<<error;
        const auto identity=stage.graphCaptureVariantSignature();ASSERT_NE(identity,0u);
        std::array<float,n> expected;expected.fill(std::bit_cast<float>(0x7fc02142u));
        std::memcpy(output.raw_mutable_data(),expected.data(),sizeof(expected));
        for(int live:{0,1,0}) {
            count=live;std::fill_n(input.mutable_data(),k,live?0.0f:std::numeric_limits<float>::quiet_NaN());
            if(live){input.mutable_data()[column]=1.0f;for(int row=0;row<n;++row)expected[row]=float((row+column)%13-6)/16.0f;}
            EXPECT_EQ(stage.graphCaptureVariantSignature(),identity);
            ASSERT_TRUE(stage.execute(&context));
            EXPECT_EQ(std::memcmp(output.raw_data(),expected.data(),sizeof(expected)),0)<<"type="<<static_cast<int>(type)<<" live="<<live;
        }
    }
}

/**
 * @brief Real prepared floating projection consumes the original typed bias.
 * Exact basis-vector dots provide an independent reference, while a second
 * actual unbiassed stage proves the selected bias changes every live output.
 * Counted zero/partial rows preserve poison, including a presliced bias view.
 */
TEST(GEMMStagePrefixCPU, TypedBiasOrdinaryAndVerifierUseOriginalTensorAndSlice)
{
    llaminar::v2::ThreadCountGuard threads(1);
    CPUDeviceContext context(DeviceId::cpu(),1);
    constexpr int n=3,k=5;
    FP32Tensor weight({n,k}),bias({n});
    for(int row=0;row<n;++row) {
        bias.mutable_data()[row]=float(row+1)*.5f;
        for(int col=0;col<k;++col)weight.mutable_data()[row*k+col]=float(row*8+col+1)*.125f;
    }
    auto inner=std::make_unique<FP32Tensor>(std::vector<size_t>{n});
    std::copy_n(bias.data(),n,inner->mutable_data());
    TensorSlice slice(std::unique_ptr<TensorBase>(std::move(inner)),
        SliceMetadata::forRowParallel(2*n,1,1,2,true));
    auto prepared=test::makePreparedGemmFixture(&weight,DeviceId::cpu(),"typed.bias.cpu.exact");
    constexpr uint32_t poison=0x7fc04217u;
    for(int m:{1,3})for(bool verifier:{false,true})for(const TensorBase*typedBias:{static_cast<const TensorBase*>(&bias),static_cast<const TensorBase*>(&slice)}) {
        SCOPED_TRACE(::testing::Message()<<"M="<<m<<" verifier="<<verifier<<" slice="<<(typedBias==&slice));
        FP32Tensor input({size_t(m),k}),output({size_t(m),n}),unbiassed({size_t(m),n});
        int32_t count=m;
        GEMMStage::Params params{.device_id=DeviceId::cpu(),.A=&input,.B=&weight,.C=&output,.m=m,.n=n,.k=k,
            .bias_tensor=typedBias,.force_decode_equivalent_verifier_prefill=verifier,
            .verifier_row_range=verifier?std::optional{DeviceRowRange::deviceCounted(m,&count)}:std::nullopt,
            .prepared_ref=prepared.ref,.prepared_store=prepared.store.get()};
        GEMMStage biased(params);
        params.C=&unbiassed;params.bias_tensor=nullptr;
        GEMMStage plain(params);
        for(int replay=0;replay<20;++replay) {
            count=verifier?std::array{m,0,1,0}[replay%4]:m;
            std::fill_n(input.mutable_data(),m*k,std::numeric_limits<float>::quiet_NaN());
            for(int row=0;row<count;++row){std::fill_n(input.mutable_data()+row*k,k,0.f);input.mutable_data()[row*k+2]=1.f;}
            std::fill_n(output.mutable_data(),m*n,std::bit_cast<float>(poison));
            std::fill_n(unbiassed.mutable_data(),m*n,std::bit_cast<float>(poison));
            ASSERT_TRUE(plain.execute(&context));ASSERT_TRUE(biased.execute(&context));
            for(int row=0;row<m;++row)for(int col=0;col<n;++col) {
                const auto index=row*n+col;
                if(row>=count) {
                    EXPECT_EQ(std::bit_cast<uint32_t>(output.data()[index]),poison);
                    EXPECT_EQ(std::bit_cast<uint32_t>(unbiassed.data()[index]),poison);
                } else {
                    const float dot=float(col*8+3)*.125f;
                    EXPECT_EQ(std::bit_cast<uint32_t>(unbiassed.data()[index]),std::bit_cast<uint32_t>(dot));
                    EXPECT_EQ(std::bit_cast<uint32_t>(output.data()[index]),std::bit_cast<uint32_t>(dot+float(col+1)*.5f));
                    EXPECT_NE(output.data()[index],unbiassed.data()[index]);
                }
            }
        }
    }
}

/** @brief Malformed typed bias metadata is rejected before prepared submission. */
TEST(GEMMStagePrefixCPU, TypedBiasRejectsNonFP32UndersizedAndMissingRequiredMetadata)
{
    FP32Tensor input({1,5}),weight({3,5}),output({1,3}),shortBias({2});FP16Tensor halfBias({3});
    auto prepared=test::makePreparedGemmFixture(&weight,DeviceId::cpu(),"typed.bias.cpu.admission");
    GEMMStage::Params params{.device_id=DeviceId::cpu(),.A=&input,.B=&weight,.C=&output,.m=1,.n=3,.k=5,
        .prepared_ref=prepared.ref,.prepared_store=prepared.store.get()};
    params.bias_tensor=&shortBias;EXPECT_THROW({GEMMStage stage(params);},std::invalid_argument);
    params.bias_tensor=&halfBias;EXPECT_THROW({GEMMStage stage(params);},std::invalid_argument);
    params.bias_tensor=nullptr;params.bias_required=true;EXPECT_THROW({GEMMStage stage(params);},std::invalid_argument);
}
