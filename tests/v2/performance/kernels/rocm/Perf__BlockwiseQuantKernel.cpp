/**
 * @file Perf__BlockwiseQuantKernel.cpp
 * @brief Captured production Q8 activation-producer economics and profiler entry.
 *
 * Explicit streams, persistent storage and retained graphs keep allocation and
 * host launch-loop overhead out of each sample. Every byte, scale word and
 * optional integer sum is checked against an independent host calculation.
 * The focused ProductionTestPreflight proof owns changing-input/tail coverage;
 * this Performance-only target owns timings, never a model certificate.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "backends/rocm/HIPGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/DeviceQ8ActivationNumericalContract.h"
#include "kernels/rocm/gemm/ROCmQuantisedActivationProducers.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>


namespace {
/** @brief Explicit diagnostic producer authority; ordinary remains the default. */
enum class RowPolicy { Ordinary, DeviceCounted };
/** @brief Parse the isolated diagnostic selector without changing production policy. */
RowPolicy rowPolicy() {
    const char *value=std::getenv("LLAMINAR_QUANT_PROFILE_COUNTED");
    if (!value || std::string(value)=="0") return RowPolicy::Ordinary;
    if (std::string(value)=="1") return RowPolicy::DeviceCounted;
    throw std::invalid_argument("LLAMINAR_QUANT_PROFILE_COUNTED requires 0 or 1");
}

/** @brief Surface a native failure immediately, never as a very fast sample. */
void checked(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

/** @brief Strict positive dimension selector for one isolated profiler launch. */
int dimension(const char *name, int default_value) {
    const char *raw = std::getenv(name);
    if (!raw) return default_value;
    size_t used = 0;
    const int value = std::stoi(raw,&used);
    if (value<=0 || raw[used]!='\0') throw std::invalid_argument(name);
    return value;
}

/** @brief Allocation lifetime is entirely outside the captured operation. */
template<class T> class Storage final {
public:
    /** @brief Allocate exactly the requested test extent. */
    explicit Storage(size_t count) { checked(hipMalloc(&data,count*sizeof(T))); }
    /** @brief The execution owner has joined every consumer before destruction. */
    ~Storage() { (void)hipFree(data); }
    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;
    T *data = nullptr;
};

/** @brief Last-declared owner joins work before graphs, events and storage retire. */
class Execution final {
public:
    /** @brief Allocate an explicit stream and external terminal timing events. */
    Execution() {
        checked(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        checked(hipEventCreate(&start)); checked(hipEventCreate(&stop));
    }
    /** @brief Synchronization is observation/retirement, never captured work. */
    ~Execution() {
        (void)hipStreamSynchronize(stream);
        graph.reset();
        (void)hipEventDestroy(start); (void)hipEventDestroy(stop);
        (void)hipStreamDestroy(stream);
    }
    Execution(const Execution &) = delete;
    Execution &operator=(const Execution &) = delete;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    std::unique_ptr<llaminar2::HIPGraphCapture> graph;
};

/**
 * @brief Time one exact public producer and verify every output after replay.
 * @param rows Captured physical row count, including the production bucket.
 * @param width Source row stride; partial blocks select the real scalar-tail route.
 * @param with_sums Include asymmetric-codebook INT32 sum publication.
 * @param samples Number of native-event observations, distinct from warmup.
 * @param policy Ordinary ABI or explicit full-live device-counted producer.
 */
void measure(int rows,int width,bool with_sums,int samples=25,RowPolicy policy=RowPolicy::Ordinary) {
    SCOPED_TRACE(::testing::Message() << "M=" << rows << " K=" << width
        << " sums=" << with_sums);
    constexpr int repetitions = 16;
    const int blocks_per_row = (width+31)/32;
    const size_t count = size_t(rows)*width, blocks = size_t(rows)*blocks_per_row;
    std::vector<float> input(count), scales(blocks);
    std::vector<int8_t> output(count);
    std::vector<int32_t> sums(blocks);
    for (size_t i=0; i<count; ++i) {
        const float magnitude = std::ldexp(1.f,int((i/32)%5)*3-6);
        input[i] = i/32%13==0 ? 0.f : magnitude *
            float(int((i*173+i/32*11)%4093)-2046)/257.f;
    }
    Storage<float> device_input(count), device_scales(blocks);
    Storage<int8_t> device_output(count);
    Storage<int32_t> device_sums(blocks), device_count(1);
    // Both host upload source and device scalar outlive terminal retirement.
    const int32_t live=rows;
    const auto row_range=llaminar2::DeviceRowRange::deviceCounted(rows,device_count.data);
    Execution execution;
    const auto stream = execution.stream;
    checked(hipMemcpyAsync(device_input.data,input.data(),count*sizeof(float),hipMemcpyHostToDevice,stream));
    if (policy==RowPolicy::DeviceCounted)
        checked(hipMemcpyAsync(device_count.data,&live,sizeof(live),hipMemcpyHostToDevice,stream));
    checked(hipStreamSynchronize(stream));
    execution.graph = std::make_unique<llaminar2::HIPGraphCapture>(stream,0);
    {
        llaminar2::ScopedBackendGraphCapture recording(*execution.graph,"activation quantizer economy");
        ASSERT_TRUE(recording.begin());
        for (int repetition=0; repetition<repetitions; ++repetition) {
            const bool submitted = policy==RowPolicy::DeviceCounted
                ? (with_sums
                    ? rocmQuantGemm_quantizeActivationsBlockwiseWithSums_rows(device_input.data,device_output.data,
                        device_scales.data,device_sums.data,rows,width,0,stream,32,&row_range)
                    : rocmQuantGemm_quantizeActivationsBlockwise_rows(device_input.data,device_output.data,
                        device_scales.data,rows,width,0,stream,32,&row_range))
                : with_sums
                ? rocmQuantGemm_quantizeActivationsBlockwiseWithSums(device_input.data,device_output.data,
                    device_scales.data,device_sums.data,rows,width,0,stream,32)
                : rocmQuantGemm_quantizeActivationsBlockwise(device_input.data,device_output.data,
                    device_scales.data,rows,width,0,stream,32);
            ASSERT_TRUE(submitted);
        }
        recording.finish();
        ASSERT_TRUE(execution.graph->instantiate());
    }
    std::vector<llaminar2::GPUGraphKernelNodeInfo> kernels;
    std::string inspection_error;
    ASSERT_TRUE(execution.graph->inspectKernelNodes(kernels,&inspection_error)) << inspection_error;
    ASSERT_EQ(execution.graph->nodeCount(),size_t(repetitions));
    ASSERT_EQ(kernels.size(),size_t(repetitions));
    const auto &native=kernels.front();
    ASSERT_TRUE(native.name_resolved);ASSERT_TRUE(native.valid());
    EXPECT_EQ(native.local_memory_bytes_per_thread,0u);
    for (const auto &node:kernels) {
        ASSERT_EQ(node.function_identity,native.function_identity);
        ASSERT_EQ(node.name,native.name);
        ASSERT_EQ(node.grid_x,native.grid_x);ASSERT_EQ(node.grid_y,native.grid_y);ASSERT_EQ(node.grid_z,native.grid_z);
        ASSERT_EQ(node.block_x,native.block_x);ASSERT_EQ(node.block_y,native.block_y);ASSERT_EQ(node.block_z,native.block_z);
        ASSERT_EQ(node.registers_per_thread,native.registers_per_thread);
        ASSERT_EQ(node.local_memory_bytes_per_thread,0u);
        ASSERT_EQ(node.static_shared_memory_bytes,native.static_shared_memory_bytes);
        ASSERT_EQ(node.dynamic_shared_memory_bytes,native.dynamic_shared_memory_bytes);
    }
    ASSERT_FALSE(::testing::Test::HasFailure());
    std::vector<double> times;
    for (int sample=-5; sample<samples; ++sample) {
        checked(hipEventRecord(execution.start,stream));
        ASSERT_TRUE(execution.graph->launch());
        checked(hipEventRecord(execution.stop,stream));
        checked(hipEventSynchronize(execution.stop));
        float ms=0.f;
        checked(hipEventElapsedTime(&ms,execution.start,execution.stop));
        if (sample>=0) times.push_back(double(ms)*1000/repetitions);
    }
    checked(hipMemcpyAsync(output.data(),device_output.data,count,hipMemcpyDeviceToHost,stream));
    checked(hipMemcpyAsync(scales.data(),device_scales.data,blocks*sizeof(float),hipMemcpyDeviceToHost,stream));
    if (with_sums)
        checked(hipMemcpyAsync(sums.data(),device_sums.data,blocks*sizeof(int32_t),hipMemcpyDeviceToHost,stream));
    checked(hipStreamSynchronize(stream));
    for (int row=0; row<rows; ++row)
        for (int block=0; block<blocks_per_row; ++block) {
            const size_t index = size_t(row)*blocks_per_row+block;
            const int begin=block*32,end=std::min(width,begin+32);
            float maximum=0.f;
            for (int k=begin;k<end;++k)
                maximum=std::max(maximum,std::abs(input[size_t(row)*width+k]));
            const float expected_scale=llaminar2::device_q8_activation_contract::scale(maximum);
            ASSERT_EQ(std::bit_cast<uint32_t>(scales[index]),std::bit_cast<uint32_t>(expected_scale));
            const float inverse=1.f/expected_scale;
            int expected_sum=0;
            for (int k=begin;k<end;++k) {
                const size_t element=size_t(row)*width+k;
                const float scaled=input[element]*inverse;
                const int expected=std::clamp(int(std::rint(scaled)),-127,127);
                ASSERT_EQ(int(output[element]),expected);
                expected_sum+=expected;
            }
            if (with_sums) ASSERT_EQ(sums[index],expected_sum);
        }
    // Native API resources expose registers and private/local bytes, not a
    // separate spill counter. Compiler spill-policy/ISA evidence remains the
    // authority for spills; this inventory must not invent that measurement.
    for (size_t sample=0;sample<times.size();++sample)
        std::printf("QUANTIZE_RAW,policy=%s,live=%d,M=%d,K=%d,sums=%d,sample=%zu,us=%.6f,symbol=%s,grid=%u/%u/%u,block=%u/%u/%u,registers=%u,local_bytes=%zu,static_lds=%zu,dynamic_lds=%zu,spill_counter=not_exposed\n",
            policy==RowPolicy::DeviceCounted?"counted":"ordinary",live,rows,width,int(with_sums),sample,times[sample],
            native.name.c_str(),native.grid_x,native.grid_y,native.grid_z,native.block_x,native.block_y,native.block_z,
            native.registers_per_thread,native.local_memory_bytes_per_thread,native.static_shared_memory_bytes,native.dynamic_shared_memory_bytes);
    std::sort(times.begin(),times.end());
    const double median=times[times.size()/2];
    const size_t logical_bytes=count*5+blocks*(with_sums?8:4);
    std::printf("QUANTIZE_CAPTURED,backend=ROCm,policy=%s,live=%d,M=%d,K=%d,sums=%d,median_us=%.4f,logical_GBps=%.3f,byte_exact=1\n",
        policy==RowPolicy::DeviceCounted?"counted":"ordinary",live,rows,width,int(with_sums),median,double(logical_bytes)/(median*1000));
}

/** @brief Public auto dispatch over decoder, verifier, main bucket and tail. */
TEST(BlockwiseQuantPerfTest,CapturedProductionShapeSweep) {
    checked(hipSetDevice(0));
    for (int rows : {1,4,16,17,64,448,512})
        for (int width : {512,1024,2048,4096,6144})
            for (bool sums : {false,true}) measure(rows,width,sums);
}

/** @brief One candidate/shape per profiler process, with no unrelated kernel. */
TEST(BlockwiseQuantPerfTest,ExactProductionShapeProfilerLaunch) {
    checked(hipSetDevice(0));
    const char *sum_mode=std::getenv("LLAMINAR_QUANT_PROFILE_SUMS");
    if (sum_mode && std::string(sum_mode)!="0" && std::string(sum_mode)!="1")
        throw std::invalid_argument("LLAMINAR_QUANT_PROFILE_SUMS requires 0 or 1");
    measure(dimension("LLAMINAR_QUANT_PROFILE_M",512),
            dimension("LLAMINAR_QUANT_PROFILE_K",2048),sum_mode && *sum_mode=='1',1,rowPolicy());
}
} // namespace
