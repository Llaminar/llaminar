/**
 * @file Test__ROCmActivationQuantization.cpp
 * @brief Byte-exact captured activation publication independent of weight format.
 *
 * Every NativeVNNI codebook consumes this same row-major INT8/FP32-scale ABI.
 * This focused producer proof complements, rather than replaces, the existing
 * all-codebook prepared GEMM tests. Replays change inputs without recapture,
 * check partial quantization blocks and guard words, and independently sum the
 * published bytes. No relaxed floating-point tolerance is used.
 */
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "backends/rocm/HIPGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/DeviceQ8ActivationNumericalContract.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

extern "C" {
    bool rocmQuantGemm_quantizeActivationsBlockwise(
        const float *, int8_t *, float *, int, int, int, void *, int);
    bool rocmQuantGemm_quantizeActivationsBlockwiseWithSums(
        const float *, int8_t *, float *, int32_t *, int, int, int, void *, int);
}

namespace {
/** @brief Report native failure at its origin before inspecting output bytes. */
void checked(hipError_t result) {
    if (result != hipSuccess) throw std::runtime_error(hipGetErrorString(result));
}

/** @brief Fixture-only admitted storage; no allocation belongs to capture. */
template<class T> class Storage final {
public:
    /** @brief Allocate one exact typed extent, including explicit test guards. */
    explicit Storage(size_t count) { checked(hipMalloc(&data, count * sizeof(T))); }
    /** @brief Retire after the last-declared execution owner joins its stream. */
    ~Storage() { (void)hipFree(data); }
    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;
    T *data = nullptr;
};

/** @brief One stream and two retained producer graphs, joined before storage. */
class Execution final {
public:
    /** @brief Never use the legacy default stream, including fixture copies. */
    Execution() { checked(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking)); }
    /** @brief Finish even partially submitted work before releasing graph owners. */
    ~Execution() {
        (void)hipStreamSynchronize(stream);
        graphs = {};
        (void)hipStreamDestroy(stream);
    }
    Execution(const Execution &) = delete;
    Execution &operator=(const Execution &) = delete;
    hipStream_t stream = nullptr;
    std::array<std::unique_ptr<llaminar2::HIPGraphCapture>,2> graphs;
};

/**
 * @brief Derive scale words and signed bytes from the independent host contract.
 * @param input Finite row-major values, including signed zeros.
 * @param rows Physical captured rows; every one is a live quantizer input.
 * @param width Physical row pitch, which need not be divisible by 32 or four.
 * @param bytes Expected payload and unchanged byte guards.
 * @param scales Expected per-row block scales and unchanged scale guards.
 * @param sums Expected exact integer sums and unchanged sum guards.
 */
void oracle(const std::vector<float> &input, int rows, int width,
            std::vector<int8_t> &bytes, std::vector<float> &scales,
            std::vector<int32_t> &sums) {
    std::memset(bytes.data(), 0x5a, bytes.size());
    std::memset(scales.data(), 0x5a, scales.size() * sizeof(float));
    std::memset(sums.data(), 0x5a, sums.size() * sizeof(int32_t));
    const int blocks = (width + 31) / 32;
    for (int row=0; row<rows; ++row)
        for (int block=0; block<blocks; ++block) {
            const int begin = block * 32, end = std::min(width, begin+32);
            float maximum = 0.f;
            for (int k=begin; k<end; ++k)
                maximum = std::max(maximum, std::abs(input[size_t(row)*width+k]));
            const size_t out = size_t(row)*blocks+block;
            const float scale = llaminar2::device_q8_activation_contract::scale(maximum);
            scales[out] = scale;
            sums[out] = 0;
            // The dense activation producer's ABI uses RN division followed
            // by a rounded multiply. This test does not change that contract
            // to the separately defined movable-expert reciprocal sequence.
            const float inverse = 1.f / scale;
            for (int k=begin; k<end; ++k) {
                const size_t i = size_t(row)*width+k;
                const float scaled = input[i] * inverse;
                const int q = std::clamp(int(std::rint(scaled)), -127, 127);
                bytes[i] = int8_t(q);
                sums[out] += q;
            }
        }
}

/**
 * @brief Change data over twenty replays, proving both public sum/no-sum APIs.
 * @param rows Capture-time row count, including verifier and prefill geometry.
 * @param width Input row pitch and independent quantization-block boundary.
 * @param unaligned Exercise legal views whose base cannot use vector alignment.
 */
void prove(int rows, int width, bool unaligned=false) {
    SCOPED_TRACE(::testing::Message() << "M=" << rows << " K=" << width
        << " unaligned=" << unaligned);
    const size_t offset = unaligned ? 1 : 0;
    const size_t count = size_t(rows)*width;
    const size_t blocks = size_t(rows)*((width+31)/32);
    Storage<float> input(count+offset), scales(blocks+8);
    Storage<int8_t> bytes(count+32+offset);
    Storage<int32_t> sums(blocks+8);
    Execution execution;
    const auto stream = execution.stream;
    checked(hipMemsetAsync(input.data,0,(count+offset)*sizeof(float),stream));
    checked(hipStreamSynchronize(stream));
    for (int with_sums=0; with_sums<2; ++with_sums) {
        auto &graph = execution.graphs[with_sums];
        graph = std::make_unique<llaminar2::HIPGraphCapture>(stream,0);
        llaminar2::ScopedBackendGraphCapture recording(*graph,"Q8 activation publication");
        ASSERT_TRUE(recording.begin());
        checked(hipMemsetAsync(bytes.data,0x5a,count+32+offset,stream));
        checked(hipMemsetAsync(scales.data,0x5a,(blocks+8)*sizeof(float),stream));
        checked(hipMemsetAsync(sums.data,0x5a,(blocks+8)*sizeof(int32_t),stream));
        const bool submitted = with_sums
            ? rocmQuantGemm_quantizeActivationsBlockwiseWithSums(input.data+offset,bytes.data+offset,
                  scales.data,sums.data,rows,width,0,stream,32)
            : rocmQuantGemm_quantizeActivationsBlockwise(input.data+offset,bytes.data+offset,
                  scales.data,rows,width,0,stream,32);
        ASSERT_TRUE(submitted);
        recording.finish();
        ASSERT_TRUE(graph->instantiate());
    }
    std::vector<float> host(count), expected_scales(blocks+8), actual_scales(blocks+8);
    std::vector<int8_t> expected_bytes(count+32), actual_bytes(count+32);
    std::vector<int32_t> expected_sums(blocks+8), actual_sums(blocks+8);
    for (int replay=0; replay<20; ++replay) {
        SCOPED_TRACE(::testing::Message() << "replay=" << replay);
        for (size_t i=0; i<count; ++i) {
            const size_t block = (i/width)*((width+31)/32)+(i%width)/32;
            const float magnitude = std::ldexp(1.f,int((block+replay)%5)*3-6);
            host[i] = block%13==0 ? (i%2 ? -0.f : 0.f) : magnitude *
                float(int((i*173+block*11+replay*13)%4093)-2046)/257.f;
        }
        oracle(host,rows,width,expected_bytes,expected_scales,expected_sums);
        checked(hipMemcpyAsync(input.data+offset,host.data(),count*sizeof(float),hipMemcpyHostToDevice,stream));
        for (int with_sums=0; with_sums<2; ++with_sums) {
            SCOPED_TRACE(::testing::Message() << "with_sums=" << with_sums);
            ASSERT_TRUE(execution.graphs[with_sums]->launch());
            int8_t leading_guard = 0x5a;
            if (unaligned)
                checked(hipMemcpyAsync(&leading_guard,bytes.data,1,hipMemcpyDeviceToHost,stream));
            checked(hipMemcpyAsync(actual_bytes.data(),bytes.data+offset,count+32,hipMemcpyDeviceToHost,stream));
            checked(hipMemcpyAsync(actual_scales.data(),scales.data,(blocks+8)*sizeof(float),hipMemcpyDeviceToHost,stream));
            checked(hipMemcpyAsync(actual_sums.data(),sums.data,(blocks+8)*sizeof(int32_t),hipMemcpyDeviceToHost,stream));
            checked(hipStreamSynchronize(stream));
            ASSERT_EQ(leading_guard,0x5a);
            ASSERT_EQ(actual_bytes,expected_bytes);
            ASSERT_EQ(std::memcmp(actual_scales.data(),expected_scales.data(),(blocks+8)*sizeof(float)),0);
            if (with_sums) ASSERT_EQ(actual_sums,expected_sums);
            else for (int32_t value : actual_sums) ASSERT_EQ(value,0x5a5a5a5a);
        }
    }
}

/** @brief Every grouped-verifier row count, including the first prefill rows. */
TEST(ROCmActivationQuantization, CapturedSmallMByteContract) {
    checked(hipSetDevice(0));
    for (int rows=1; rows<=65; ++rows) prove(rows,96);
}

/** @brief Real tensor shard widths, large buckets, and partial block/pitch tails. */
TEST(ROCmActivationQuantization, CapturedPrefillByteContract) {
    checked(hipSetDevice(0));
    for (int rows : {64,65,448,512})
        for (int width : {31,32,33,63,64,65,256,512,1024,2048,3072,4096,6144})
            prove(rows,width);
    for (int rows : {1024,2048,4096}) prove(rows,96);
    for (int rows : {64,512})
        for (int width : {33,2048}) prove(rows,width,true);
}

/** @brief Nonempty producers cannot silently enter the legacy default stream. */
TEST(ROCmActivationQuantization, RejectsDefaultStream) {
    checked(hipSetDevice(0));
    Storage<float> input(32), scales(1);
    Storage<int8_t> output(32);
    Storage<int32_t> sums(1);
    EXPECT_FALSE(rocmQuantGemm_quantizeActivationsBlockwise(
        input.data,output.data,scales.data,1,32,0,nullptr,32));
    EXPECT_FALSE(rocmQuantGemm_quantizeActivationsBlockwiseWithSums(
        input.data,output.data,scales.data,sums.data,1,32,0,nullptr,32));
}
} // namespace
