/**
 * @file Test__GEMMStagePrefixCPU.cpp
 * @brief Device-free actual prepared M1 projection preserves original native counts.
 *
 * Compile/run only against the coherently composed prepared floating/HC ABI.
 * The real prepared store and GEMMStage execute0/1/0 with poisoned inactive
 * source/output bytes; no mock kernel, accelerator or model is involved.
 */
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/qwen4exp/QwenSparseMath.h"
#include "tensors/Tensors.h"
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
