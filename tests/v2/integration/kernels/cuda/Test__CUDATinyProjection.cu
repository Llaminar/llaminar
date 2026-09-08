/**
 * @file Test__CUDATinyProjection.cu
 * @brief Captured floating projection equivalence and isolated economy probe.
 *
 * The independent oracle reproduces the original 256-thread reduction, so a
 * change shared by production serial and grouped launches cannot hide drift.
 * Functional registration covers all floating weights, K/N tails, MTP rows and
 * prefill. Shared-operand candidates additionally prove complete output bytes
 * after each poisoned replay, including exhaustive finite FP16/BF16 weights.
 * The separate performance test never enters production preflight.
 */
#include <gtest/gtest.h>
#include "kernels/common/FloatingPointVerifierLaunch.h"
#include "kernels/cuda/gemm/CUDATinyProjectionSharedKernel.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

extern "C" bool cudaFp32_stage_batched_projection_pointers(
    const float **, const float **, float **, const float *const *,
    const float *const *, float *const *, int, int, void *);

namespace
{
/** @brief Throw a precise test diagnostic for a failed CUDA operation. */
void checked(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

/**
 * @brief Preserve the old software FP16 conversion as an independent oracle.
 *
 * Production now uses the native conversion instruction. Keeping the previous
 * bit-expansion here makes the finite-encoding sweep sensitive to conversion
 * changes as well as to changes in the dot-product reduction.
 */
__device__ float legacyHalfToFloat(uint16_t bits)
{
    const uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000u) << 16;
    int exponent = (bits >> 10) & 31;
    uint32_t mantissa = bits & 1023;
    if (exponent == 31) return __uint_as_float(sign | 0x7f800000u | (mantissa << 13));
    if (exponent == 0)
    {
        if (!mantissa) return __uint_as_float(sign);
        exponent = 1;
        while (!(mantissa & 1024)) { mantissa <<= 1; --exponent; }
        mantissa &= 1023;
    }
    return __uint_as_float(sign | (static_cast<uint32_t>(exponent+112) << 23) | (mantissa << 13));
}

/** @brief Select ordinary signed inputs or an exhaustive finite 16-bit weight sweep. */
enum class WeightCoverage { SignedRandom, AllFiniteEncodings };

/** @brief Own test-only device storage; never allocate during graph replay. */
template<class T> class Storage
{
public:
    /** @brief Allocate the requested number of native elements before capture. */
    explicit Storage(size_t count) { checked(cudaMalloc(&data, count * sizeof(T))); }
    /** @brief Free storage after the fixture has retired all GPU work. */
    ~Storage() { (void)cudaFree(data); }
    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;
    T *data = nullptr;
};

/** @brief Independent pre-optimization arithmetic oracle, one block per output. */
__global__ void legacyProjection(const float *a, const void *b, float *c,
                                 int n, int k, int dtype)
{
    const size_t row = static_cast<size_t>(blockIdx.y) * k;
    const size_t col = (static_cast<size_t>(blockIdx.z) * n + blockIdx.x) * k;
    float sum = 0;
    for (int i = threadIdx.x; i < k; i += 256)
    {
        float w;
        if (dtype == 0) w = static_cast<const float *>(b)[col+i];
        else if (dtype == 1) w = legacyHalfToFloat(static_cast<const uint16_t *>(b)[col+i]);
        else w = __bfloat162float(static_cast<const __nv_bfloat16 *>(b)[col+i]);
        sum += a[row+i] * w;
    }
    __shared__ float parts[256];
    parts[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1)
    {
        if (threadIdx.x < stride) parts[threadIdx.x] += parts[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        c[(static_cast<size_t>(blockIdx.z)*gridDim.y+blockIdx.y)*n+blockIdx.x] = parts[0];
}

/** @brief Own one model-free projection bundle and all its replay resources. */
class ProjectionCase
{
public:
    /** @brief Prepare finite signed inputs and all three native weight formats. */
    ProjectionCase(int rows, int columns, int width, int dtype,
                   WeightCoverage coverage = WeightCoverage::SignedRandom)
        : m(rows), n(columns), k(width), type(dtype), a(size_t(m)*k),
          b(size_t(n)*k*2), output(size_t(m)*n*2), oracle(size_t(m)*n*2),
          aa(2), bb(2), cc(2)
    {
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::vector<float> input(size_t(m)*k), weights(size_t(n)*k*2);
        std::vector<uint16_t> native(weights.size());
        uint32_t rng = 1977;
        auto next = [&]() {
            rng = rng*1664525u+1013904223u;
            return static_cast<float>(static_cast<int>(rng>>12)-524288)/524288.0f;
        };
        for (float &x : input) x = next();
        for (float &x : weights) x = next();
        // BF16's largest finite weights need small activations to keep the
        // arithmetic finite. This still exercises every sign/exponent/mantissa.
        if (coverage == WeightCoverage::AllFiniteEncodings)
            for (float &x : input) x *= 0x1p-32f;
        checked(cudaMemcpyAsync(a.data,input.data(),input.size()*4,cudaMemcpyHostToDevice,stream));
        if (type == 0)
            checked(cudaMemcpyAsync(b.data,weights.data(),weights.size()*4,cudaMemcpyHostToDevice,stream));
        else
        {
            for(size_t i=0;i<weights.size();++i)
            {
                native[i] = type == 1 ? __half_as_ushort(__float2half_rn(weights[i]))
                                     : __bfloat16_as_ushort(__float2bfloat16_rn(weights[i]));
                if (coverage == WeightCoverage::AllFiniteEncodings)
                {
                    const uint16_t exponent_mask = type == 1 ? 0x7c00u : 0x7f80u;
                    const uint16_t bits = static_cast<uint16_t>(i);
                    native[i] = (bits & exponent_mask) == exponent_mask ? 0 : bits;
                }
            }
            checked(cudaMemcpyAsync(b.data,native.data(),native.size()*2,cudaMemcpyHostToDevice,stream));
        }
        const float *second_weights = type == 0 ? b.data+size_t(n)*k :
            reinterpret_cast<const float *>(reinterpret_cast<const uint16_t *>(b.data)+size_t(n)*k);
        const float *as[2] = {a.data,a.data}, *bs[2] = {b.data,second_weights};
        float *cs[2] = {output.data,output.data+size_t(m)*n};
        if(!cudaFp32_stage_batched_projection_pointers(aa.data,bb.data,cc.data,as,bs,cs,2,0,stream))
            throw std::runtime_error("projection pointer preparation failed");
        checked(cudaStreamSynchronize(stream));
        checked(cudaEventCreate(&start));
        checked(cudaEventCreate(&stop));
    }
    /** @brief Retire the captured transaction before releasing any storage. */
    ~ProjectionCase()
    {
        (void)cudaStreamSynchronize(stream);
        if(exec) (void)cudaGraphExecDestroy(exec);
        if(graph) (void)cudaGraphDestroy(graph);
        (void)cudaEventDestroy(start);
        (void)cudaEventDestroy(stop);
        (void)cudaStreamDestroy(stream);
    }
    /** @brief Submit the actual public production launch bridge. */
    void launch()
    {
        bool ok = type == 0 ? cudaFp32_tiny_batched_projection(aa.data,bb.data,cc.data,m,n,k,2,0,stream)
                            : cudaFp32x16_tiny_batched_projection(aa.data,bb.data,cc.data,m,n,k,2,type-1,0,stream);
        if(!ok) throw std::runtime_error("production projection launch failed");
    }
    /** @brief Record the production projection into a retained graph once. */
    void capture()
    {
        checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        launch();
        checked(cudaStreamEndCapture(stream,&graph));
        checked(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    }
    /** @brief Compare two distinct batched outputs after twenty retained graph replays. */
    void verify()
    {
        legacyProjection<<<dim3(n,m,2),256,0,stream>>>(a.data,b.data,oracle.data,n,k,type);
        checked(cudaGetLastError());
        capture();
        for(int replay=0;replay<20;++replay) checked(cudaGraphLaunch(exec,stream));
        std::vector<float> expected(size_t(m)*n*2), actual(expected.size());
        checked(cudaMemcpyAsync(expected.data(),oracle.data,expected.size()*4,cudaMemcpyDeviceToHost,stream));
        checked(cudaMemcpyAsync(actual.data(),output.data,actual.size()*4,cudaMemcpyDeviceToHost,stream));
        checked(cudaStreamSynchronize(stream));
        ASSERT_EQ(std::memcmp(expected.data(),actual.data(),expected.size()*4),0)
            << "M="<<m<<" N="<<n<<" K="<<k<<" dtype="<<type;
    }
    /**
     * @brief Prove the public bridge captured the economical row-sharing schedule.
     * @param rows_per_cta Expected native-format-specific physical row tile.
     *
     * Inspect the actual production graph symbol, not the fixture's separately
     * compiled candidate. This catches a tested kernel that never reaches live
     * dispatch and checks the final linked specialization for local spills.
     */
    void expectSharedRowTile(int rows_per_cta)
    {
        size_t count = 0;
        checked(cudaGraphGetNodes(graph, nullptr, &count));
        ASSERT_EQ(count, 1u);
        cudaGraphNode_t node = nullptr;
        checked(cudaGraphGetNodes(graph, &node, &count));
        cudaKernelNodeParams params{};
        checked(cudaGraphKernelNodeGetParams(node, &params));
        EXPECT_EQ(params.gridDim.x, static_cast<unsigned>((n+7)/8));
        EXPECT_EQ(params.gridDim.y, static_cast<unsigned>((m+rows_per_cta-1)/rows_per_cta));
        EXPECT_EQ(params.gridDim.z, 2u);
        EXPECT_EQ(params.blockDim.x, 256u);
        cudaFuncAttributes resources{};
        checked(cudaFuncGetAttributes(&resources, params.func));
        EXPECT_EQ(resources.localSizeBytes, 0u);
        EXPECT_EQ(resources.sharedSizeBytes,
                  static_cast<size_t>((8+rows_per_cta)*256)*sizeof(float));
    }
    /**
     * @brief Check a physical shared-operand tile without changing live dispatch.
     * @tparam RowsPerCTA Compiled row tile, independent of the arithmetic tree.
     *
     * Each replay starts from poisoned output storage and is compared in full.
     * A later successful replay therefore cannot hide an earlier publication
     * error. Candidate forcing is fixture-only; production policy is unchanged.
     */
    template<int RowsPerCTA> void verifyShared()
    {
        if (type == 0) verifySharedAs<float, RowsPerCTA>();
        else if (type == 1) verifySharedAs<__half, RowsPerCTA>();
        else verifySharedAs<__nv_bfloat16, RowsPerCTA>();
    }
    /** @brief Report unprofiled native-event latency; no economy assertion in preflight. */
    void measure()
    {
        capture();
        for(int i=0;i<20;++i) checked(cudaGraphLaunch(exec,stream));
        checked(cudaEventRecord(start,stream));
        for(int i=0;i<200;++i) checked(cudaGraphLaunch(exec,stream));
        checked(cudaEventRecord(stop,stream));
        checked(cudaEventSynchronize(stop));
        float ms=0; checked(cudaEventElapsedTime(&ms,start,stop));
        const float production_us=ms*1000/200;
        cudaGraph_t legacy_graph=nullptr;
        cudaGraphExec_t legacy_exec=nullptr;
        checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        legacyProjection<<<dim3(n,m,2),256,0,stream>>>(a.data,b.data,oracle.data,n,k,type);
        checked(cudaStreamEndCapture(stream,&legacy_graph));
        checked(cudaGraphInstantiate(&legacy_exec,legacy_graph,nullptr,nullptr,0));
        for(int i=0;i<20;++i) checked(cudaGraphLaunch(legacy_exec,stream));
        checked(cudaEventRecord(start,stream));
        for(int i=0;i<200;++i)
            checked(cudaGraphLaunch(legacy_exec,stream));
        checked(cudaEventRecord(stop,stream));
        checked(cudaEventSynchronize(stop));
        checked(cudaEventElapsedTime(&ms,start,stop));
        std::cout<<"projection_us,dtype="<<type<<",m="<<m<<",n="<<n<<",k="<<k
                 <<",production="<<production_us<<",legacy="<<ms*1000/200<<'\n';
        checked(cudaGraphExecDestroy(legacy_exec));
        checked(cudaGraphDestroy(legacy_graph));
    }
private:
    /**
     * @brief Admit and capture the exact native format/row specialization.
     * @tparam Weight Unchanged native weight type selected by the fixture.
     * @tparam RowsPerCTA Physical row tile selected by the candidate sweep.
     *
     * Compiler resource checks precede graph capture. The independent legacy
     * reduction is the oracle, not another launch of the candidate kernel.
     */
    template<class Weight, int RowsPerCTA> void verifySharedAs()
    {
        auto kernel = llaminar2::cuda::detail::sharedTinyProjectionKernel<Weight, RowsPerCTA>;
        cudaFuncAttributes attributes{};
        checked(cudaFuncGetAttributes(&attributes, kernel));
        ASSERT_EQ(attributes.localSizeBytes, 0u) << "dtype=" << type << " rows=" << RowsPerCTA;
        int resident_blocks = 0;
        checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident_blocks, kernel, 256, 0));
        ASSERT_GT(resident_blocks, 0);
        legacyProjection<<<dim3(n,m,2),256,0,stream>>>(a.data,b.data,oracle.data,n,k,type);
        checked(cudaGetLastError());
        checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        kernel<<<dim3((n+7)/8,(m+RowsPerCTA-1)/RowsPerCTA,2),256,0,stream>>>(
            aa.data,bb.data,cc.data,m,n,k, llaminar2::DeviceRowRange::fullyActive(m));
        const auto launch_status = cudaGetLastError();
        checked(cudaStreamEndCapture(stream,&graph));
        checked(launch_status);
        checked(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        std::vector<float> expected(size_t(m)*n*2), actual(expected.size());
        checked(cudaMemcpyAsync(expected.data(),oracle.data,expected.size()*sizeof(float),
                                cudaMemcpyDeviceToHost,stream));
        checked(cudaStreamSynchronize(stream));
        for (int replay=0; replay<20; ++replay)
        {
            checked(cudaMemsetAsync(output.data,0x7e,actual.size()*sizeof(float),stream));
            checked(cudaGraphLaunch(exec,stream));
            checked(cudaMemcpyAsync(actual.data(),output.data,actual.size()*sizeof(float),
                                    cudaMemcpyDeviceToHost,stream));
            checked(cudaStreamSynchronize(stream));
            ASSERT_EQ(std::memcmp(expected.data(),actual.data(),expected.size()*sizeof(float)),0)
                << "M=" << m << " N=" << n << " K=" << k << " dtype=" << type
                << " rows=" << RowsPerCTA << " replay=" << replay;
        }
    }
    int m,n,k,type;
    Storage<float> a,b,output,oracle;
    Storage<const float *> aa,bb;
    Storage<float *> cc;
    cudaStream_t stream=nullptr;
    cudaEvent_t start=nullptr,stop=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t exec=nullptr;
};
}

/** @brief Prove old-tree byte identity over floating types, tails and runtime rows. */
TEST(CUDATinyProjection, CapturedAllFloatingFormatsMatchLegacyTree)
{
    checked(cudaSetDevice(0));
    const std::array<std::array<int,2>,5> shapes{{{1,1},{7,31},{9,257},{48,5120},{64,6144}}};
    std::vector<int> rows;
    for (int m = 1; m <= 65; ++m) rows.push_back(m);
    for (int m : {127,128,129,511,512,513}) rows.push_back(m);
    for(int type=0;type<3;++type)
    {
        for(auto shape:shapes)
            for(int m:rows)
                ProjectionCase(m,shape[0],shape[1],type).verify();
        ProjectionCase(4096,9,257,type).verify();
        if (type != 0)
            for (int m : {1,4,32,33})
                ProjectionCase(m,64,512,type,WeightCoverage::AllFiniteEncodings).verify();
    }
}

/** @brief Certify production selection as well as bytes across floating prefill tiles. */
TEST(CUDATinyProjection, ProductionPrefillUsesSharedOperandsWithoutChangingTree)
{
    checked(cudaSetDevice(0));
    for (int type=0; type<3; ++type)
        for (const auto shape : {std::array{512,48,5120}, std::array{513,33,257},
                                 std::array{1024,64,6144}, std::array{4096,32,256}})
        {
            ProjectionCase projection(shape[0],shape[1],shape[2],type);
            projection.verify();
            projection.expectSharedRowTile(type == 1 ? 4 : 8);
        }
}

/** @brief Certify shared-operand row tiles across native formats and arithmetic tails. */
TEST(CUDATinyProjection, SharedOperandCandidatesMatchLegacyTree)
{
    checked(cudaSetDevice(0));
    const std::array<std::array<int,2>,3> shapes{{{1,1},{9,257},{48,5120}}};
    std::vector<int> rows;
    for (int m=1; m<=65; ++m) rows.push_back(m);
    for (int m : {127,128,129,511,512,513}) rows.push_back(m);
    for (int type=0; type<3; ++type)
    {
        for (auto shape : shapes)
            for (int m : rows)
            {
                ProjectionCase(m,shape[0],shape[1],type).verifyShared<2>();
                ProjectionCase(m,shape[0],shape[1],type).verifyShared<4>();
                ProjectionCase(m,shape[0],shape[1],type).verifyShared<8>();
            }
        ProjectionCase(4096,9,257,type).verifyShared<8>();
        if (type != 0)
            for (int m : {1,4,32,33})
            {
                ProjectionCase(m,64,512,type,WeightCoverage::AllFiniteEncodings).verifyShared<2>();
                ProjectionCase(m,64,512,type,WeightCoverage::AllFiniteEncodings).verifyShared<4>();
                ProjectionCase(m,64,512,type,WeightCoverage::AllFiniteEncodings).verifyShared<8>();
            }
    }
}

/** @brief Select one exact format/row candidate per profiler process. */
class CUDATinyProjectionEconomy : public testing::TestWithParam<std::tuple<int,int>> {};

/** @brief Separate optional perf entry point, excluded from the functional gate. */
TEST_P(CUDATinyProjectionEconomy, CapturedLatency)
{
    checked(cudaSetDevice(0));
    const auto [type, m] = GetParam();
    ProjectionCase(m,48,5120,type).measure();
}

/** @brief Measure shared-tile admission boundaries separately from correctness gates. */
TEST(CUDATinyProjectionGeometryEconomy, CapturedLatency)
{
    checked(cudaSetDevice(0));
    for (int type=0; type<3; ++type)
        for (const auto shape : {std::array{512,32,256}, std::array{513,33,257},
                                 std::array{512,64,1024}, std::array{1024,64,6144},
                                 std::array{4096,32,256}})
            ProjectionCase(shape[0],shape[1],shape[2],type).measure();
}

/** @brief Give timing/profile receipts stable native-format and row-count names. */
std::string economyCaseName(const testing::TestParamInfo<std::tuple<int,int>> &info)
{
    const auto [type,m] = info.param;
    return std::string(std::array{"FP32","FP16","BF16"}[type])+"_M"+std::to_string(m);
}

INSTANTIATE_TEST_SUITE_P(AllFloatingFormatsAndRows,CUDATinyProjectionEconomy,
    testing::Combine(testing::Values(0,1,2),testing::Values(1,4,16,31,32,64,512)),economyCaseName);
