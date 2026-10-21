/**
 * @file Test__CPUNativeVNNIProjectionCapacity.cpp
 * @brief Prove routed projection bundles retain serial bytes beyond eight experts.
 *
 * Flash Next routes ten experts, requiring twenty simultaneous gate/up
 * descriptors. Every public fused entry must share the planner's admitted
 * descriptor bound. Mixed native formats, grouped rows and the maximum expert
 * geometry exercise that contract without model files or accelerators.
 */
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "utils/NativeVNNITestPartialStorage.h"
#include "utils/QuantizedVerifierFormats.h"
#include "utils/OpenMPUtils.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

namespace llaminar2::test
{
using namespace cpu::native_vnni;

/** @test Top-10 and maximum bundles preserve every mixed-codebook projection and row. */
TEST(CPUNativeVNNIProjectionCapacity, RouterBundlesCoverEveryAdmittedExpert)
{
    constexpr int n=64,k=256,maxRows=3;
    constexpr int capacity=NativeVNNIFusedRowsPlan::capacity;
    constexpr float poison=-12345.0f;
    static_assert(capacity==2*(moe_runtime_abi::kMaxExperts+1));
    static_assert(capacity==CPUNativeVNNIGemmKernel::kMaxBatchedPrequantizedProjections);
    std::vector<std::unique_ptr<TensorBase>> weights;
    std::vector<std::unique_ptr<CPUNativeVNNIGemmKernel>> kernels;
    for(const auto &format:quantizedVerifierFormats()) {
        weights.push_back(format.create({n,k},1409+weights.size()));
        kernels.push_back(std::make_unique<CPUNativeVNNIGemmKernel>(weights.back().get()));
        ASSERT_TRUE(kernels.back()->isValid())<<format.label;
    }
    auto input=TestTensorFactory::createFP32Random({maxRows,k},-.75f,.75f,4423);
    std::vector<Q8_1Block> q8(maxRows*k/32);
    quantize_activations_to_q8_1(input->data(),q8.data(),maxRows,k,k/32);
    std::vector<std::vector<float>> expected(kernels.size(),std::vector<float>(maxRows*n));
    std::vector<std::unique_ptr<FP32Tensor>> outputs;
    std::vector<ITensorGemm::TensorProjectionDesc> projections;
    std::vector<FusedVerifierRowsDesc> raw;
    for(int index=0;index<=capacity;++index) {
        auto *kernel=kernels[index%kernels.size()].get();
        outputs.push_back(std::make_unique<FP32Tensor>(std::vector<std::size_t>{maxRows,n}));
        projections.push_back({kernel,outputs.back().get(),n,nullptr,"expert_projection"});
        raw.push_back({.packed=&kernel->packedWeights(),.output=outputs.back()->mutable_data(),.N=n,.ldc=n});
    }
    for(const int workers:{1,3}) {
        llaminar::v2::ThreadCountGuard threads(workers);
        for(std::size_t index=0;index<kernels.size();++index) {
            NativeVNNITestPartialStorage partials(kernels[index]->packedWeights(),maxRows);
            for(int row=0;row<maxRows;++row)
                gemv_native_vnni_preq(kernels[index]->packedWeights(),q8.data()+row*k/32,
                    expected[index].data()+row*n,partials.span());
        }
        NativeVNNITestPartialStorage partials(raw.data(),capacity,maxRows);
        for(const int count:{1,16,17,20,32,512,capacity}) for(const int rows:{1,maxRows}) {
            SCOPED_TRACE("workers="+std::to_string(workers)+" count="+std::to_string(count)+" rows="+std::to_string(rows));
            auto bundle=std::vector<ITensorGemm::TensorProjectionDesc>(projections.begin(),projections.begin()+count);
            for(auto &output:outputs) std::fill_n(output->mutable_data(),maxRows*n,poison);
            ASSERT_TRUE(kernels.front()->multiply_fused_router_q8_hidden_grouped_decode_equivalent(
                q8.data(),bundle,rows,k,partials.span()));
            for(int index=0;index<count;++index) {
                ASSERT_EQ(std::memcmp(outputs[index]->data(),expected[index%kernels.size()].data(),rows*n*sizeof(float)),0)<<index;
                EXPECT_TRUE(std::all_of(outputs[index]->data()+rows*n,outputs[index]->data()+maxRows*n,
                    [](float value){return value==poison;}))<<index;
            }
            EXPECT_TRUE(std::all_of(outputs[count]->data(),outputs[count]->data()+maxRows*n,
                [](float value){return value==poison;}));
        }
        // Admission rejects oversized bundles before any projection mutates output.
        for(auto &output:outputs) std::fill_n(output->mutable_data(),maxRows*n,poison);
        EXPECT_FALSE(kernels.front()->multiply_fused_router_q8_hidden_grouped_decode_equivalent(
            q8.data(),projections,1,k,partials.span()));
        for(const auto &output:outputs) EXPECT_TRUE(std::all_of(output->data(),output->data()+maxRows*n,
            [](float value){return value==poison;}));

        // The down phase has distinct inputs and must also admit all experts,
        // including the shared expert beyond the old 512-descriptor boundary.
        std::vector<FusedGemvMultiInputDesc> down;
        for(int index=0;index<=static_cast<int>(moe_runtime_abi::kMaxExperts);++index)
            down.push_back({q8.data(),&kernels[index%kernels.size()]->packedWeights(),outputs[index]->mutable_data(),n});
        gemv_fused_multi_input_preq(down.data(),partials.span(),down.size());
        for(std::size_t index=0;index<down.size();++index)
            EXPECT_EQ(std::memcmp(outputs[index]->data(),expected[index%kernels.size()].data(),n*sizeof(float)),0)<<index;
    }
}
} // namespace llaminar2::test
