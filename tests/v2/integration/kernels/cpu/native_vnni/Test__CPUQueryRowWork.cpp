/**
 * @file Test__CPUQueryRowWork.cpp
 * @brief Focused production-primitive proof of query-row elision across native CPU formats.
 *
 * The canonical source registry supplies every native quantized family. Needed
 * rows compare bytewise against the original ordinary M1 prepared engine;
 * unused rows retain poisoned activation/Q8/output storage across dirty
 * large/short/empty/crossing replays. No model or accelerator is loaded.
 * This is a private unexecuted regression draft for root's integration gate.
 */
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "utils/QuantizedVerifierFormats.h"
#include "utils/CPUProjectionTestWorkspace.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

namespace llaminar2::test {
namespace {
using namespace cpu::native_vnni;
/** @brief Restore caller-owned OpenMP policy after focused native execution. */
struct QueryWorkerPolicy final {
    int workers=omp_get_max_threads(),dynamic=omp_get_dynamic();
    /** @brief Disable shrinking while the regression selects explicit physical worker counts. */
    QueryWorkerPolicy(){omp_set_dynamic(0);}
    /** @brief Restore worker count and shrinking independently. */
    ~QueryWorkerPolicy(){omp_set_num_threads(workers);omp_set_dynamic(dynamic);}
};
/** @brief Check exact untouched bytes without interpreting poisoned Q8 scales or sums. */
bool bytePattern(const void *source,std::size_t bytes,unsigned char pattern) {
    const auto *data=static_cast<const unsigned char *>(source);
    return std::all_of(data,data+bytes,[pattern](unsigned char value){return value==pattern;});
}
/**
 * @brief Exercise the unchanged prepared engine through ordinary and grouped query operations.
 * @param format Canonical source-format factory and source identity.
 * @param n Original output columns, including representative tails.
 * @param k Original logical K, including the real QFN indexer width.
 * @param rotated Whether the existing prepared rotation contract is enabled.
 */
void proveQueryRows(const QuantizedVerifierFormatCase &format,int n,int k,bool rotated) {
    QueryWorkerPolicy restore;
    std::optional<ActivationRotation> rotation;
    if(rotated)rotation.emplace(k,128);
    auto source=format.create({std::size_t(n),std::size_t(k)},17501);
    ASSERT_NE(source,nullptr);
    if(rotation)source->setActivationRotation(&*rotation);
    CPUNativeVNNIGemmKernel kernel(source.get());
    ASSERT_TRUE(kernel.isValid());
    constexpr int requests=3;
    for(int workers:{1,3,7})for(int rowsPerRequest:{1,3,8,16}) {
        omp_set_num_threads(workers);
        const int physical=requests*rowsPerRequest,blocks=kernel.packedWeights().blocks_per_row;
        SCOPED_TRACE(std::string(format.label)+" N="+std::to_string(n)+" K="+std::to_string(k)+
            " M="+std::to_string(physical)+" workers="+std::to_string(workers)+" rotation="+std::to_string(rotated));
        FP32Tensor input({std::size_t(physical),std::size_t(k)}),output({std::size_t(physical),std::size_t(n)}),
            single({1,std::size_t(k)}),serial({1,std::size_t(n)});
        CPUProjectionTestWorkspace workspace(physical,k,kernel.getWorkspaceRequirements(physical));
        auto *q8=static_cast<Q8_1Block *>(workspace.get()->getBuffer(kCPUProjectionQ8));
        ASSERT_NE(q8,nullptr);
        std::vector<float> original(std::size_t(physical)*k),expected(std::size_t(physical)*n);
        std::vector<Q8_1Block> expectedQ8(std::size_t(physical)*blocks);
        for(std::size_t i=0;i<original.size();++i)original[i]=float(int(i%31)-15)/37.f;
        for(int row=0;row<physical;++row) {
            std::copy_n(original.data()+std::size_t(row)*k,k,single.mutable_data());
            ASSERT_FALSE(ITensorGemm::VerifierKernelModeScope::active());
            ASSERT_TRUE(kernel.multiply_tensor(&single,&serial,1,n,k,true,1,0,nullptr,nullptr,-1,workspace.get()));
            std::copy_n(serial.data(),n,expected.data()+std::size_t(row)*n);
            std::copy_n(q8,blocks,expectedQ8.data()+std::size_t(row)*blocks);
        }
        // Separate original header owners force the immutable indirect layout.
        std::array<int,3> first{},second{},third{};
        const std::array<ProjectionRequestPublication,requests> publication{{
            {&first[0],&first[1]},{&second[0],&second[1]},{&third[0],&third[1]}}};
        const auto work=ProjectionRowWork::causalIndirect(requests,rowsPerRequest,publication.data(),
            16384,2051,175);
        const std::vector<ITensorGemm::TensorProjectionDesc> projection{{&kernel,&output,n,nullptr,"query"}};
        for(int replay=0;replay<5;++replay) {
            if(replay==0||replay==4)first=second=third={4096,rowsPerRequest,0};
            if(replay==1){first={4096,0,0};second={2048,rowsPerRequest,73};third={4096,std::min(2,rowsPerRequest),0};}
            if(replay==2)first=second=third={512,0,0};
            if(replay==3){first={2048,rowsPerRequest,0};second={2050,rowsPerRequest,0};third={2051,rowsPerRequest,0};}
            std::copy(original.begin(),original.end(),input.mutable_data());
            std::fill_n(output.mutable_data(),output.numel(),-12345.f);
            std::memset(q8,0xa5,std::size_t(physical)*blocks*sizeof(Q8_1Block));
            auto *rotatedBytes=workspace.get()->getBuffer(kCPUProjectionRotation);
            const auto rotatedCapacity=workspace.get()->getBufferSize(kCPUProjectionRotation);
            if(rotatedBytes)std::memset(rotatedBytes,0xa5,rotatedCapacity);
            for(int row=0;row<physical;++row)
                if(!work.resolveHost(work.requestForRow(row)).contains(row))
                    std::fill_n(input.mutable_data()+std::size_t(row)*k,k,std::numeric_limits<float>::quiet_NaN());
            {
                ITensorGemm::ProjectionWorkScope selection(work);
                auto arithmetic=kernel.beginVerifierDecodeEquivalentScope(std::nullopt);
                ASSERT_TRUE(kernel.multiply_fused_verifier_rows_decode_equivalent(&input,projection,physical,k,nullptr,workspace.get()));
            }
            for(int row=0;row<physical;++row) {
                const bool needed=work.resolveHost(work.requestForRow(row)).contains(row);
                const auto *actual=output.data()+std::size_t(row)*n;
                const auto *actualQ8=q8+std::size_t(row)*blocks;
                if(needed) {
                    EXPECT_EQ(std::memcmp(actual,expected.data()+std::size_t(row)*n,n*sizeof(float)),0)<<"row="<<row;
                    EXPECT_EQ(std::memcmp(actualQ8,expectedQ8.data()+std::size_t(row)*blocks,blocks*sizeof(Q8_1Block)),0)<<"row="<<row;
                } else {
                    EXPECT_TRUE(std::all_of(actual,actual+n,[](float value){return value==-12345.f;}))<<"row="<<row;
                    EXPECT_TRUE(bytePattern(actualQ8,blocks*sizeof(Q8_1Block),0xa5))<<"row="<<row;
                    if(rotatedBytes)EXPECT_TRUE(bytePattern(static_cast<const float *>(rotatedBytes)+std::size_t(row)*k,k*sizeof(float),0xa5));
                }
            }
        }
    }
}
class CPUQueryRowWork : public ::testing::TestWithParam<QuantizedVerifierFormatCase> {};
TEST_P(CPUQueryRowWork, EverySourceRetainsSerialBytesAndUnwrittenRowsAcrossDirtyRequests) {
    for(const auto shape:{std::pair{65,512},std::pair{512,2560},std::pair{128,8192}})
        proveQueryRows(GetParam(),shape.first,shape.second,false);
}
TEST(CPUQueryRowWorkRotation, OriginalPreparedRotationSkipsUnusedRows) {
    const auto &formats=quantizedVerifierFormats();
    const auto format=std::find_if(formats.begin(),formats.end(),[](const auto &value){return value.tensor_type==TensorType::Q4_0;});
    ASSERT_NE(format,formats.end());proveQueryRows(*format,65,512,true);
}
TEST_P(CPUQueryRowWork, OrdinaryM1WorkNeverEnablesVerifierMathAndPreservesUnusedScratch) {
    QueryWorkerPolicy restore;omp_set_num_threads(3);
    constexpr int n=65,k=512;
    auto source=GetParam().create({n,k},17531);
    CPUNativeVNNIGemmKernel kernel(source.get());ASSERT_TRUE(kernel.isValid());
    FP32Tensor input({1,k}),output({1,n}),expected({1,n});
    CPUProjectionTestWorkspace workspace(1,k,kernel.getWorkspaceRequirements(1));
    for(int i=0;i<k;++i)input.mutable_data()[i]=float(i%13-6)/19.f;
    ASSERT_TRUE(kernel.multiply_tensor(&input,&expected,1,n,k,true,1,0,nullptr,nullptr,-1,workspace.get()));
    std::array<int,3> replay{4096,1,0};
    const auto work=ProjectionRowWork::causalStrided(1,1,&replay[0],&replay[1],3,8192,2051);
    {
        ITensorGemm::ProjectionWorkScope selection(work);
        ASSERT_FALSE(ITensorGemm::VerifierKernelModeScope::active());
        ASSERT_TRUE(kernel.multiply_tensor(&input,&output,1,n,k,true,1,0,nullptr,nullptr,-1,workspace.get()));
    }
    EXPECT_EQ(std::memcmp(output.data(),expected.data(),n*sizeof(float)),0);
    std::fill_n(input.mutable_data(),k,std::numeric_limits<float>::quiet_NaN());
    auto *q8=workspace.get()->getBuffer(kCPUProjectionQ8);const auto q8Bytes=workspace.get()->getBufferSize(kCPUProjectionQ8);
    for(const auto frontierAndLive:{std::pair{512,1},std::pair{4096,0}}) {
        replay[0]=frontierAndLive.first;replay[1]=frontierAndLive.second;
        std::fill_n(output.mutable_data(),n,-12345.f);std::memset(q8,0xa5,q8Bytes);
        ITensorGemm::ProjectionWorkScope selection(work);
        ASSERT_FALSE(ITensorGemm::VerifierKernelModeScope::active());
        ASSERT_TRUE(kernel.multiply_tensor(&input,&output,1,n,k,true,1,0,nullptr,nullptr,-1,workspace.get()));
        EXPECT_TRUE(std::all_of(output.data(),output.data()+n,[](float value){return value==-12345.f;}));
        EXPECT_TRUE(bytePattern(q8,q8Bytes,0xa5));
    }
}
TEST(CPUQueryRowWorkAdmission, WorkBearingFusedBundleFailsBeforeOutputOrScratchWrites) {
    QueryWorkerPolicy restore;omp_set_num_threads(1);
    auto source=quantizedVerifierFormats().front().create({64,512},17541);
    CPUNativeVNNIGemmKernel kernel(source.get());
    FP32Tensor input({8,512}),firstOutput({8,64}),secondOutput({8,64});
    std::fill_n(firstOutput.mutable_data(),firstOutput.numel(),-12345.f);
    std::fill_n(secondOutput.mutable_data(),secondOutput.numel(),-12345.f);
    const std::vector<ITensorGemm::TensorProjectionDesc> projections{
        {&kernel,&firstOutput,64,nullptr,"query0"},{&kernel,&secondOutput,64,nullptr,"query1"}};
    CPUProjectionTestWorkspace workspace(8,512,kernel.getWorkspaceRequirements(8));
    const auto work=ProjectionRowWork::fromPrefix(DeviceRowRange::fullyActive(8));
    ITensorGemm::ProjectionWorkScope selection(work);auto arithmetic=kernel.beginVerifierDecodeEquivalentScope(std::nullopt);
    EXPECT_THROW((void)kernel.multiply_fused_verifier_rows_decode_equivalent(&input,projections,8,512,nullptr,workspace.get()),std::invalid_argument);
    EXPECT_TRUE(std::all_of(firstOutput.data(),firstOutput.data()+firstOutput.numel(),[](float value){return value==-12345.f;}));
    EXPECT_TRUE(std::all_of(secondOutput.data(),secondOutput.data()+secondOutput.numel(),[](float value){return value==-12345.f;}));
}
INSTANTIATE_TEST_SUITE_P(AllFormats,CPUQueryRowWork,::testing::ValuesIn(quantizedVerifierFormats()),
    [](const auto &info){return std::string(info.param.label);});
} // namespace
} // namespace llaminar2::test
