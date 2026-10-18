/**
 * @file Test__FixedHalfProjectionGroup.cpp
 * @brief Captured ragged half groups retain the production fixed arithmetic tree.
 *
 * Every short row count, projection cardinality, packed-load edge and model
 * width is compared bytewise with the retained width-grouped production path.
 * Large/small/empty live replays poison all outputs and retain input/weight
 * ownership. Native graph inspection proves one projection node, zero private
 * memory, and no pointer-publication work in the new short-row group.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/FloatingPointVerifierLaunch.h"
#include "utils/TestTensorFactory.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

extern "C" bool cudaFp32_stage_batched_projection_pointers(const float **, const float **,
    float **, const float *const *, const float *const *, float *const *, int, int, void *);
extern "C" bool rocmFp32_stage_batched_projection_pointers(const float **, const float **,
    float **, const float *const *, const float *const *, float *const *, int, int, void *);

namespace
{
using namespace llaminar2;
using namespace llaminar2::test;
/** @brief Reject backend failures at their first explicit observation edge. */
void requireHalf(bool ok, const char *edge) { if (!ok) throw std::runtime_error(edge); }
/** @brief Test-only completion before every borrowed DMA/graph owner retires. */
struct HalfObservationJoin
{
    IBackend *backend; DeviceId device; void *stream;
    ~HalfObservationJoin() { (void)backend->synchronizeStream(stream, device.ordinal); }
};
/** @brief Dispatch only the requested first-class native implementation. */
bool enqueueHalf(DeviceId device, const FixedHalfProjectionGroup &group,
                 int m, int k, void *stream, const DeviceRowRange *rows)
{
#ifdef HAVE_CUDA
    if (device.is_cuda()) return cudaFp16_short_projection_group(group, m, k, device.ordinal, stream, rows);
#endif
#ifdef HAVE_ROCM
    if (device.is_rocm()) return rocmFp16_short_projection_group(group, m, k, device.ordinal, stream, rows);
#endif
    throw std::invalid_argument("Half group fixture requires its compiled backend");
}
/** @brief Retain the independently compiled production width-grouped oracle. */
bool enqueueHalfOracle(DeviceId device, const FixedHalfProjectionGroup &group,
    const float **a, const float **b, float **c, const std::vector<float *> &outputs,
    int m, int k, void *stream, const DeviceRowRange *rows)
{
    for (int p = 0; p < group.size(); ++p)
    {
        const auto *input = reinterpret_cast<const float *>(group.input());
        const auto *weight = reinterpret_cast<const float *>(group.weight(p));
        float *output = outputs[p];
        bool ok = false;
#ifdef HAVE_CUDA
        if (device.is_cuda()) ok = cudaFp32_stage_batched_projection_pointers(a,b,c,
            &input,&weight,&output,1,device.ordinal,stream) &&
            cudaFp16_tiny_batched_projection(a,b,c,m,group.columns(p),k,1,device.ordinal,stream,rows);
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) ok = rocmFp32_stage_batched_projection_pointers(a,b,c,
            &input,&weight,&output,1,device.ordinal,stream) &&
            rocmFp16_batched_projection(a,b,c,m,group.columns(p),k,1,device.ordinal,stream,rows);
#endif
        if (!ok) return false;
    }
    return true;
}
/**
 * @brief Prove one compact geometry under adversarial live-row replays.
 * @param device Exact native endpoint. @param m Physical short-row capacity.
 * @param k Reduction width. @param widths Independent positive output extents.
 * @param special Include every half word with NaN/Inf arithmetic witnesses.
 */
void proveHalfGroup(DeviceId device, int m, int k, const std::vector<int> &widths, bool special=false)
{
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&]
    {
        SCOPED_TRACE(::testing::Message() << device.toString() << " M=" << m << " K=" << k << " count=" << widths.size());
        auto *backend = getBackendFor(device); void *stream = context.defaultStream();
        requireHalf(backend && stream, "Half group exact backend stream");
        std::vector<uint16_t> input_words(size_t(m)*k);
        const auto word=[](size_t i) { i%=63488; return uint16_t(i<31744?i:(i-31744)|0x8000u); };
        for(size_t i=0;i<input_words.size();++i) input_words[i]=word(i*73+1);
        if(special && k==1) {
            constexpr uint16_t values[]{0x3c00,0xbc00,0,0x8000,0x7c00,0xfc00,0x7c01,0x7e55};
            for(size_t i=0;i<input_words.size();++i) input_words[i]=values[i%8];
        } else if(special) for(size_t i=0;i<input_words.size();++i) input_words[i]=uint16_t(i*71+17);
        auto input=std::make_unique<FP16Tensor>(std::vector<size_t>{size_t(m),size_t(k)},input_words);
        auto count=TestTensorFactory::createINT32({1});
        auto pointers=TestTensorFactory::createINT32({3*sizeof(void*)/sizeof(int32_t)});
        std::vector<std::unique_ptr<FP16Tensor>> weights;
        std::vector<std::unique_ptr<FP32Tensor>> outputs,oracle_outputs;
        std::vector<std::vector<uint16_t>> weight_words;
        std::vector<std::vector<float>> observed,expected,poisoned;
        for(size_t p=0;p<widths.size();++p) {
            const int n=widths[p]; weight_words.emplace_back(size_t(n)*k);
            for(size_t i=0;i<weight_words.back().size();++i)
                weight_words.back()[i]=special?uint16_t(i*(k==1?1:193)+(k==1?0:17)):word(i*193+p*113+17);
            weights.push_back(std::make_unique<FP16Tensor>(std::vector<size_t>{size_t(n),size_t(k)},weight_words.back()));
            outputs.push_back(TestTensorFactory::createFP32Zeros({size_t(m+1),size_t(n)}));
            oracle_outputs.push_back(TestTensorFactory::createFP32Zeros({size_t(m+1),size_t(n)}));
            observed.emplace_back(size_t(m+1)*n); expected.emplace_back(size_t(m+1)*n);
            poisoned.emplace_back(size_t(m+1)*n,std::bit_cast<float>(UINT32_C(0x7fc01234)));
        }
        std::vector<float*> oracle_addresses;
        std::vector<uint16_t> returned_input(input_words.size()),returned_weight;
        int live=m;
        auto graph=context.createGraphCapture(stream),oracle=context.createGraphCapture(stream);
        HalfObservationJoin join{backend,device,stream};
        requireHalf(input->ensureOnDevice(device,stream) && count->ensureOnDevice(device,stream) &&
            pointers->allocateOnDevice(device,stream), "Half group admitted input storage");
        for(size_t p=0;p<widths.size();++p) {
            requireHalf(weights[p]->ensureOnDevice(device,stream) && outputs[p]->allocateOnDevice(device,stream) &&
                oracle_outputs[p]->allocateOnDevice(device,stream), "Half group admitted matrix storage");
            oracle_addresses.push_back(static_cast<float*>(oracle_outputs[p]->gpu_data_ptr()));
        }
        auto group=FixedHalfProjectionGroup::single(static_cast<const uint16_t*>(input->gpu_data_ptr()),
            static_cast<const uint16_t*>(weights[0]->gpu_data_ptr()),static_cast<float*>(outputs[0]->gpu_data_ptr()),widths[0]);
        for(size_t p=1;p<widths.size();++p) group.append(static_cast<const uint16_t*>(weights[p]->gpu_data_ptr()),
            static_cast<float*>(outputs[p]->gpu_data_ptr()),widths[p]);
        auto **a=static_cast<const float**>(pointers->gpu_data_ptr()); auto **b=a+1;
        auto **c=static_cast<float**>(pointers->gpu_data_ptr())+2;
        const auto rows=DeviceRowRange::deviceCounted(m,static_cast<const int32_t*>(count->gpu_data_ptr()));
        const auto wrong=DeviceRowRange::fullyActive(m+1);
        EXPECT_FALSE(enqueueHalf(device,group,m,k,nullptr,&rows));
        EXPECT_FALSE(enqueueHalf(device,group,0,k,stream,nullptr));
        EXPECT_FALSE(enqueueHalf(device,group,m,0,stream,&rows));
        EXPECT_FALSE(enqueueHalf(device,group,m,k,stream,&wrong));
        EXPECT_FALSE(enqueueHalf(device,group,kShortHalfProjectionMaxRows+1,k,stream,nullptr));
        requireHalf(backend->synchronizeStream(stream,device.ordinal),"Half group setup complete");
        {
            ScopedBackendGraphCapture capture(context,*graph,"ragged short half group");
            requireHalf(capture.begin(),"Half group capture begin");
            requireHalf(enqueueHalf(device,group,m,k,stream,&rows),"Half group capture body");
            capture.finish();
        }
        std::vector<GPUGraphKernelNodeInfo> nodes;
        ASSERT_TRUE(graph->inspectKernelNodes(nodes)); ASSERT_EQ(nodes.size(),1u);
        EXPECT_NE(nodes[0].name.find("short_half_projection_group_kernel"),std::string::npos);
        EXPECT_EQ(nodes[0].grid_x,unsigned(group.totalColumns())); EXPECT_EQ(nodes[0].grid_y,unsigned(m));
        EXPECT_EQ(nodes[0].local_memory_bytes_per_thread,0u);
        requireHalf(graph->instantiate(),"Half group instantiate");
        {
            ScopedBackendGraphCapture capture(context,*oracle,"retained production half oracle");
            requireHalf(capture.begin(),"Half oracle capture begin");
            requireHalf(enqueueHalfOracle(device,group,a,b,c,oracle_addresses,m,k,stream,&rows),"Half oracle capture body");
            capture.finish();
        }
        requireHalf(oracle->instantiate(),"Half oracle instantiate");
        for(int active:{m,0,1,m/2,m}) {
            live=active;
            requireHalf(backend->hostToDevice(count->gpu_data_ptr(),&live,sizeof(live),device.ordinal,stream),"Half group live publication");
            for(size_t p=0;p<widths.size();++p) requireHalf(
                backend->hostToDevice(outputs[p]->gpu_data_ptr(),poisoned[p].data(),poisoned[p].size()*4,device.ordinal,stream) &&
                backend->hostToDevice(oracle_outputs[p]->gpu_data_ptr(),poisoned[p].data(),poisoned[p].size()*4,device.ordinal,stream),"Half group poison publication");
            requireHalf(oracle->launch() && graph->launch(),"Half group retained replay");
            for(size_t p=0;p<widths.size();++p) requireHalf(
                backend->deviceToHost(observed[p].data(),outputs[p]->gpu_data_ptr(),observed[p].size()*4,device.ordinal,stream) &&
                backend->deviceToHost(expected[p].data(),oracle_outputs[p]->gpu_data_ptr(),expected[p].size()*4,device.ordinal,stream),"Half group terminal observation");
            requireHalf(backend->synchronizeStream(stream,device.ordinal),"Half group observation complete");
            for(size_t p=0;p<widths.size();++p) {
                ASSERT_EQ(std::memcmp(observed[p].data(),expected[p].data(),observed[p].size()*4),0) << "projection=" << p << " live=" << live;
                for(size_t i=size_t(live)*widths[p];i<observed[p].size();++i)
                    ASSERT_EQ(std::bit_cast<uint32_t>(observed[p][i]),UINT32_C(0x7fc01234));
            }
        }
        requireHalf(backend->deviceToHost(returned_input.data(),input->gpu_data_ptr(),returned_input.size()*2,device.ordinal,stream) &&
            backend->synchronizeStream(stream,device.ordinal),"Half input immutable observation");
        EXPECT_EQ(returned_input,input_words);
        for(size_t p=0;p<widths.size();++p) {
            returned_weight.resize(weight_words[p].size());
            requireHalf(backend->deviceToHost(returned_weight.data(),weights[p]->gpu_data_ptr(),returned_weight.size()*2,device.ordinal,stream) &&
                backend->synchronizeStream(stream,device.ordinal),"Half weights immutable observation");
            EXPECT_EQ(returned_weight,weight_words[p]);
        }
    });
}
/** @brief Sweep short-row totality, group cardinality and K scheduling boundaries. */
void proveHalfBackend(DeviceId device)
{
    const std::array<int,8> widths{1,3,7,17,32,33,73,128};
    for(int m=1;m<=kShortHalfProjectionMaxRows;++m) {
        proveHalfGroup(device,m,65,std::vector<int>(widths.begin(),widths.begin()+1+m%8));
        if(::testing::Test::HasFatalFailure()) return;
    }
    for(int k:{1,63,64,127,128,129,255,256,257,2047,2048,2049,4095,4096,4097})
        for(int m:{1,8,31}) {
            proveHalfGroup(device,m,k,{17,9,3,5});
            if(::testing::Test::HasFatalFailure()) return;
        }
    proveHalfGroup(device,8,1,{65536,32768},true);
    proveHalfGroup(device,8,257,{257,128,32,32},true);
}
#ifdef HAVE_CUDA
/** @test CUDA short projection groups retain exact bytes with one native node. */
TEST(FixedHalfProjectionGroup, CUDA) { proveHalfBackend(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
/** @test ROCm packed short projection groups retain exact bytes and live extents. */
TEST(FixedHalfProjectionGroup, ROCm) { proveHalfBackend(DeviceId::rocm(0)); }
#endif
} // namespace
