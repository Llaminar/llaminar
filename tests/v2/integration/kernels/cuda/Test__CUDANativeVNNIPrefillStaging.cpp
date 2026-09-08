/**
 * @file Test__CUDANativeVNNIPrefillStaging.cpp
 * @brief Captured byte/resource proof for production packed-operand staging.
 *
 * All candidates capture the exact BK64 symbol in the production core. No
 * device body is recompiled by this fixture: RDC/compiler flags can otherwise
 * change resource use even when the shared source text is identical.
 * RegisterDecode is the arithmetic oracle; the existing broader GEMM gate
 * independently verifies it against public M1. Sweep all staged encodings, row
 * boundaries, partial tiles, odd K halves, canonical folds and epilogues. Poison
 * output/guards before twenty captured replays. Compiler spills cannot enter
 * capture, and no timing threshold belongs in this functional preflight.
 * Exact-overlay promotion also sweeps every structurally staged output tile;
 * byte proof for the historical tile alone cannot certify another launch grid.
 */
#include "kernels/cuda/gemm/CUDANativeVNNIPrefillDiagnostics.h"
#include "tensors/NativeVnniFormatInfo.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

/** @brief Surface the first backend error before accepting any evidence. */
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

/** @brief Stable typed storage owned only by the untimed diagnostic fixture. */
template<class T> struct Buffer {
    T* data = nullptr;
    std::size_t count;
    /** @brief Allocate once before graph construction. */
    explicit Buffer(std::size_t n) : count(n) { check(cudaMalloc(&data,n*sizeof(T))); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    /** @brief Retire after the fixture stream has drained. */
    ~Buffer() { if (data) cudaFree(data); }
    /** @brief Upload the exact host extent on the explicit producer stream. */
    void upload(const std::vector<T>& source,cudaStream_t stream) {
        if (!stream || source.size()!=count) throw std::runtime_error("bad upload");
        check(cudaMemcpyAsync(data,source.data(),count*sizeof(T),cudaMemcpyHostToDevice,stream));
    }
};

/** @brief Own graph lifetimes and the terminal join before buffers are released. */
struct Session {
    cudaStream_t stream = nullptr;
    std::array<cudaGraph_t,4> graphs{};
    std::array<cudaGraphExec_t,4> execs{};
    /** @brief Create the exact non-default stream before capture. */
    Session() {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    }
    /** @brief Drain the diagnostic queue and retire graphs before the stream. */
    ~Session() {
        if (stream) cudaStreamSynchronize(stream);
        for (auto e:execs) if(e) cudaGraphExecDestroy(e);
        for (auto g:graphs) if(g) cudaGraphDestroy(g);
        if(stream) cudaStreamDestroy(stream);
    }
};

/** @brief Cheap deterministic fixture bytes; no source-format conversion occurs. */
uint32_t mixed(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
    x *= 0x846ca68bU; return x ^ (x >> 16);
}

/** @brief Explicit final-output arithmetic covered independently of staging. */
enum class Epilogue { Identity, AffineBias };

/** @brief Compare the production query with the independently named symbol.
 * @tparam CB Actual packed codebook, including asymmetric metadata variants.
 * @tparam Canonical Whether this symbol publishes disjoint partition partials.
 * @tparam Mode Exact typed staging template encoded by the diagnostic API.
 * This detects a query that accidentally inspects RegisterDecode while a
 * candidate would launch another specialization, before accepting its timings.
 */
template<int CB, bool Canonical, int Mode>
void verifyResourceIdentity(std::set<const void *> &symbols)
{
    using llaminar2::cuda::prefill::PrefillStagingSchedule;
    constexpr int wn = CB == 7 ? 2 : 4;
    constexpr int threads = 4 * wn * 32;
    constexpr int tile = CB == 7 ? 2 : 5;
    constexpr auto staging = static_cast<PrefillStagingSchedule>(Mode);
    SCOPED_TRACE(::testing::Message() << "codebook=" << CB
        << " canonical=" << Canonical << " staging=" << Mode);
    CUDADensePrefillKernelResources primary{}, auxiliary{};
    ASSERT_TRUE(cudaNativeVNNIPrefill_queryCandidateResources(
        CB, tile, Canonical ? 1 : 0, 0, 0, &primary, &auxiliary, staging));
    ASSERT_NE(primary.kernel_symbol, nullptr);
    // A wrong visitor branch must not let two candidate identities inspect
    // and capture the same symbol under different schedule names.
    EXPECT_TRUE(symbols.insert(primary.kernel_symbol).second);
    const char *symbol_name = nullptr;
    check(cudaFuncGetName(&symbol_name, primary.kernel_symbol));
    ASSERT_NE(symbol_name, nullptr);
    EXPECT_NE(std::string(symbol_name).find("nativeVnniTC_BK64"), std::string::npos);
    cudaFuncAttributes attributes{};
    int resident = 0;
    check(cudaFuncGetAttributes(&attributes, primary.kernel_symbol));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, primary.kernel_symbol, threads, 0));
    EXPECT_EQ(primary.registers_per_thread, attributes.numRegs);
    EXPECT_EQ(primary.local_memory_bytes_per_thread, attributes.localSizeBytes);
    EXPECT_EQ(primary.static_shared_memory_bytes, attributes.sharedSizeBytes);
    EXPECT_EQ(primary.dynamic_shared_memory_bytes, 0u);
    EXPECT_EQ(primary.threads_per_block, threads);
    EXPECT_EQ(primary.max_threads_per_block, attributes.maxThreadsPerBlock);
    EXPECT_EQ(primary.max_active_blocks_per_sm, resident);
    if constexpr (Canonical)
    {
        ASSERT_NE(auxiliary.kernel_symbol, nullptr);
        check(cudaFuncGetName(&symbol_name, auxiliary.kernel_symbol));
        ASSERT_NE(symbol_name, nullptr);
        EXPECT_NE(std::string(symbol_name).find("canonical_kpart_reduce"), std::string::npos);
        EXPECT_EQ(auxiliary.threads_per_block, 256);
        EXPECT_EQ(auxiliary.local_memory_bytes_per_thread, 0u);
        EXPECT_GT(auxiliary.max_active_blocks_per_sm, 0);
    }
    else
    {
        EXPECT_EQ(auxiliary.kernel_symbol, nullptr);
        EXPECT_EQ(auxiliary.registers_per_thread, 0);
        EXPECT_EQ(auxiliary.threads_per_block, 0);
    }
}

/** @brief Query all four staging identities for both publication families. */
template<int CB>
void verifyCodebookResources(std::set<const void *> &symbols)
{
    [&]<int... Mode>(std::integer_sequence<int, Mode...>) {
        (verifyResourceIdentity<CB, false, Mode>(symbols), ...);
        (verifyResourceIdentity<CB, true, Mode>(symbols), ...);
    }(std::make_integer_sequence<int, 4>{});
}

/** @brief Compare one retained graph family without a performance threshold.
 * @tparam CB Physical codebook of the native nibble-payload family.
 * @tparam Canonical Write every grid-Z partition instead of folding in one CTA.
 * @param m Logical row count, including verifier and prefill boundaries.
 * @param n Logical column count, including partial physical fragments.
 * @param k Reduction width divisible by 32.
 * @param kb Exact partition count of the canonical ordered fold.
 * @param epilogue Identity or scaled/biased final output.
 * @param tile_id Exact production output-tile identity, independent of staging.
 */
template<int CB, bool Canonical = false>
void run(int m,int n,int k,int kb,Epilogue epilogue,int tile_id=CB==7?2:5) {
    if (tile_id < 0 || tile_id > 5 || tile_id == 1)
        throw std::runtime_error("unsupported staged tile identity");
    // Independent launch geometry makes a wrong resource-visitor selection
    // observable: the fixture does not invoke a second device implementation.
    const int tile_rows = tile_id >= 4 ? 128 : 64;
    const int tile_columns = tile_id == 0 ? 64 : 128;
    constexpr auto *format = llaminar2::native_vnni_formats::forSourceIdentity(CB, false);
    static_assert(format != nullptr, "fixture source identity is not in the canonical catalog");
    constexpr int pb = format->payload_bytes;
    if constexpr (Canonical)
        if (kb <= 1) throw std::runtime_error("partition producer requires multiple partitions");
    const std::size_t output_elements = std::size_t(m) * n * (Canonical ? kb : 1);
    const std::size_t blocks=std::size_t(n)*(k/32), elements=std::size_t(m)*k;
    std::vector<uint8_t> payload(blocks*pb);
    std::vector<uint16_t> scales(blocks),mins(blocks);
    std::vector<int8_t> activation(elements);
    std::vector<float> activation_scales(elements/32);
    std::vector<int32_t> sums(elements/32);
    for(std::size_t i=0;i<payload.size();++i) payload[i]=uint8_t(mixed(i+77));
    for(std::size_t i=0;i<blocks;++i) {
        scales[i]=uint16_t(0x2400u+(mixed(i+25)&0x7ffu));
        mins[i]=uint16_t(0xa000u+(mixed(i+51)&0x7ffu));
    }
    for(std::size_t i=0;i<elements;++i) activation[i]=int(mixed(i+133)%255)-127;
    for(int row=0;row<m;++row) for(int block=0;block<k/32;++block) {
        activation_scales[std::size_t(row)*(k/32)+block]=float(1+mixed(row+block*7)%31)/256;
        int sum=0;
        for(int e=0;e<32;++e) sum+=activation[std::size_t(row)*k+block*32+e];
        sums[std::size_t(block)*m+row]=sum;
    }
    Buffer<uint8_t> p(payload.size()); Buffer<uint16_t> s(blocks),b(blocks);
    Buffer<int8_t> a(elements); Buffer<float> sa(elements/32),out(output_elements+8);
    Buffer<int32_t> asum(elements/32);
    const bool affine = epilogue == Epilogue::AffineBias;
    float alpha = affine ? -0.375f : 1.0f;
    float beta = affine ? 0.625f : 0.0f;
    std::vector<float> previous(std::size_t(m)*n), bias(n);
    for(std::size_t i=0;i<previous.size();++i) previous[i]=float(int(mixed(i+501)%257)-128)/64.0f;
    for(int i=0;i<n;++i) bias[i]=float(int(mixed(i+901)%63)-31)/32.0f;
    Buffer<float> prior(previous.size()), column_bias(bias.size());
    Session session;
    prior.upload(previous,session.stream); column_bias.upload(bias,session.stream);
    p.upload(payload,session.stream);s.upload(scales,session.stream);b.upload(mins,session.stream);
    a.upload(activation,session.stream);sa.upload(activation_scales,session.stream);asum.upload(sums,session.stream);
    check(cudaStreamSynchronize(session.stream));
    bool admitted[4]{};
    auto prepare=[&]<int Mode>() {
        // Every mode uses the same production tile for this codebook. Only
        // payload staging and the selected shared metadata view differ.
        constexpr auto staging = static_cast<llaminar2::cuda::prefill::PrefillStagingSchedule>(Mode);
        CUDADensePrefillKernelResources primary{}, auxiliary{};
        if (!cudaNativeVNNIPrefill_queryCandidateResources(
                CB, tile_id, Canonical ? 1 : 0, 0, 0,
                &primary, &auxiliary, staging) || !primary.kernel_symbol)
            throw std::runtime_error("production candidate symbol is missing");
        admitted[Mode]=primary.local_memory_bytes_per_thread==0 && primary.max_active_blocks_per_sm>0;
        if(!admitted[Mode]) {
            // Some asymmetric specializations have compiler-spill exclusions.
            // Never capture them; the symmetric target family must stay total.
            if constexpr (Mode <= 1 || CB == 0 || CB == 4 || CB == 6)
                throw std::runtime_error("required staging symbol failed resource admission");
            return;
        }
        // Arguments use the same public partition ABI as the production body.
        // cudaLaunchKernel copies their values into the captured node; all
        // pointed-to storage outlives graph replay and the terminal join.
        uint32_t *emins = nullptr;
        float *prior_data = affine ? prior.data : nullptr;
        float *bias_data = affine ? column_bias.data : nullptr;
        llaminar2::cuda::prefill::CanonicalM1PartitionGeometry partitions{
            kb, (k / 32 + kb - 1) / kb};
        int ordered = 1;
        void *arguments[]{&a.data, &p.data, &s.data, &b.data, &emins,
            &out.data, &sa.data, &asum.data, &prior_data, &bias_data,
            &m, &n, &k, &alpha, &beta, &partitions, &ordered};
        check(cudaStreamBeginCapture(session.stream,cudaStreamCaptureModeThreadLocal));
        const auto launch_error = cudaLaunchKernel(primary.kernel_symbol,
            dim3((m+tile_rows-1)/tile_rows,(n+tile_columns-1)/tile_columns,Canonical?kb:1),
            dim3(primary.threads_per_block), arguments, 0, session.stream);
        const auto end_error=cudaStreamEndCapture(session.stream,&session.graphs[Mode]);
        check(launch_error);check(end_error);
        check(cudaGraphInstantiate(&session.execs[Mode],session.graphs[Mode],nullptr,nullptr,0));
    };
    prepare.template operator()<0>();prepare.template operator()<1>();
    prepare.template operator()<2>();prepare.template operator()<3>();
    if(!admitted[0]) throw std::runtime_error("baseline resources invalid");
    std::vector<float> oracle(out.count),actual(out.count);
    for(int mode=0;mode<4;++mode) if(admitted[mode]) {
        for(int replay=0;replay<20;++replay) {
            check(cudaMemsetAsync(out.data,0x7e,out.count*sizeof(float),session.stream));
            check(cudaGraphLaunch(session.execs[mode],session.stream));
            check(cudaMemcpyAsync(actual.data(),out.data,out.count*sizeof(float),cudaMemcpyDeviceToHost,session.stream));
            check(cudaStreamSynchronize(session.stream));
            for(std::size_t i=output_elements;i<out.count;++i) {
                uint32_t guard; std::memcpy(&guard,&actual[i],4);
                if(guard!=0x7e7e7e7eu) throw std::runtime_error("output guard corruption");
            }
            if(mode==0 && replay==0) oracle=actual;
            else if(std::memcmp(oracle.data(),actual.data(),out.count*sizeof(float)))
                throw std::runtime_error("captured output byte mismatch");
        }
    }
}

/** @brief Exercise all rows through M65 and larger physical-tile boundaries. */
template<int CB>
void verifyCodebook()
{
    for(int m=1;m<=65;++m) {
        SCOPED_TRACE(::testing::Message() << "codebook=" << CB << " m=" << m << " n=9 k=96 partitions=2");
        run<CB>(m,9,96,2,Epilogue::Identity);
    }
    for(int m : {1,33,129,513})
        for(int n : {1,127,128,129})
            for(int partitions : {1,3,20,40})
                for(auto epilogue : {Epilogue::Identity,Epilogue::AffineBias}) {
                    SCOPED_TRACE(::testing::Message() << "codebook=" << CB << " m=" << m << " n=" << n
                        << " k=1056 partitions=" << partitions << " epilogue=" << int(epilogue));
                    run<CB>(m,n,1056,partitions,epilogue);
                }
    // Private producers can begin on an odd 32-value half and can own empty
    // trailing partitions. Compare the entire partial buffer, not just final
    // output, so a later reducer cannot mask an unwritten or reordered slice.
    for (int m : {1, 31, 129})
        for (int partitions : {2, 3, 20, 40})
        {
            SCOPED_TRACE(::testing::Message() << "private codebook=" << CB
                << " m=" << m << " n=129 k=1056 partitions=" << partitions);
            run<CB, true>(m, 129, 1056, partitions, Epilogue::Identity);
        }
}
} // namespace

/** @brief Prove every admitted nibble-payload candidate in retained graphs. */
TEST(CUDANativeVNNIPrefillStaging, CapturedCandidatesMatchRegisterDecode)
{
    int count=0;
    ASSERT_EQ(cudaGetDeviceCount(&count),cudaSuccess);
    ASSERT_GT(count,0);
    ASSERT_EQ(cudaSetDevice(0),cudaSuccess);
    verifyCodebook<0>();
    verifyCodebook<4>();
    verifyCodebook<5>();
    verifyCodebook<6>();
    verifyCodebook<7>();
}

/** @brief Every staged tile preserves full output/partials over retained replay. */
TEST(CUDANativeVNNIPrefillStaging, AllStagedTilesMatchRegisterDecode)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto verify = []<int CB>() {
        for (int tile : {0, 2, 3, 4, 5})
            for (int m : {1, 129, 513})
            {
                SCOPED_TRACE(::testing::Message() << "codebook=" << CB
                    << " tile=" << tile << " rows=" << m);
                run<CB>(m, 129, 1056, 3, Epilogue::AffineBias, tile);
                run<CB, true>(m, 129, 1056, 3, Epilogue::Identity, tile);
            }
    };
    verify.template operator()<0>();
    verify.template operator()<4>();
    verify.template operator()<5>();
    verify.template operator()<6>();
    verify.template operator()<7>();
}

/** @brief Every diagnostic identity must inspect the kernel it would launch. */
TEST(CUDANativeVNNIPrefillStaging, ResourceQueriesNameExactStagingSymbols)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    std::set<const void *> symbols;
    verifyCodebookResources<0>(symbols);
    verifyCodebookResources<4>(symbols);
    verifyCodebookResources<5>(symbols);
    verifyCodebookResources<6>(symbols);
    verifyCodebookResources<7>(symbols);
    EXPECT_EQ(symbols.size(), 40u);
    using llaminar2::cuda::prefill::PrefillStagingSchedule;
    CUDADensePrefillKernelResources primary{}, auxiliary{};
    // BK256 has no staged-nibble specialization, and tile 1 lacks a copy
    // owner for each half-block. Neither may masquerade as RegisterDecode.
    EXPECT_FALSE(cudaNativeVNNIPrefill_queryCandidateResources(
        0, -2, 0, 0, 0, &primary, &auxiliary, PrefillStagingSchedule::AsyncPayload));
    EXPECT_FALSE(cudaNativeVNNIPrefill_queryCandidateResources(
        4, 1, 0, 0, 0, &primary, &auxiliary, PrefillStagingSchedule::AsyncPayload));
    EXPECT_FALSE(cudaNativeVNNIPrefill_queryCandidateResources(
        11, 2, 0, 0, 0, &primary, &auxiliary, PrefillStagingSchedule::AsyncPayload));
    EXPECT_FALSE(cudaNativeVNNIPrefill_queryCandidateResources(
        4, 2, 0, 0, 0, &primary, &auxiliary, static_cast<PrefillStagingSchedule>(255)));
}

/** @brief A diagnostic worker cannot change another thread's launch policy. */
TEST(CUDANativeVNNIPrefillStaging, StagingControlRejectsInvalidAndIsThreadLocal)
{
    using llaminar2::cuda::prefill::PrefillStagingSchedule;
    const auto original = cudaNativeVNNIPrefill_getStagingSchedule();
    std::array<bool, 4> outcomes{};
    std::array<std::jthread, 4> workers;
    for (int mode = 0; mode < 4; ++mode)
        workers[mode] = std::jthread([&, mode] {
            const auto chosen = static_cast<PrefillStagingSchedule>(mode);
            outcomes[mode] =
                cudaNativeVNNIPrefill_getStagingSchedule() == PrefillStagingSchedule::RegisterDecode &&
                cudaNativeVNNIPrefill_setStagingSchedule(chosen) &&
                !cudaNativeVNNIPrefill_setStagingSchedule(static_cast<PrefillStagingSchedule>(255)) &&
                cudaNativeVNNIPrefill_getStagingSchedule() == chosen;
        });
    for (auto &worker : workers) worker.join();
    for (bool outcome : outcomes) EXPECT_TRUE(outcome);
    EXPECT_EQ(cudaNativeVNNIPrefill_getStagingSchedule(), original);
}
