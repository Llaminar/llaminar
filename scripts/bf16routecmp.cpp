// Native-vs-oneDNN route comparison harness (BF16 / FP32 / AMX).
// Same process, interleaved rounds, min of 20, OpenMP threads.
//
// PURPOSE: reproduce the cpu_bf16 route comparisons recorded in
// docs/v2/projects/2026-11/CPU_BF16_NATIVE_GEMM_PLAN.md on another host.
// The harness runs the native AMX-BF16 engine, the native AVX512-BF16
// engine, the native scalar FP32 route, and oneDNN BF16/FP32 matmuls on
// the four canonical serving shapes, all inside one process so the
// native/oneDNN ratios stay valid under host load.
//
// BG build requirements
//   * This branch (feat/cpu-amx-bf16-gemm): the AMX engine, its
//     permission-gated admission probe (CPUFeatures.h) and the native
//     routes live in src/v2/kernels/cpu/gemm/ + src/v2/utils/.
//   * A oneDNN CPU x64 build (headers + libdnnl.so) - here
//     external/onednn/build-avx512bf16/include|lib.
//   * GCC >= 13 with -mamx-tile -mamx-bf16 support.
//   * libfort headers (FloatingPointGemmKernel.h includes KernelProfiler);
//     point -I at any libfort include dir (here build/_deps/libfort-src/lib).
//   * Adapt -I <generated-dir> to your tree (compile_commands.json of the
//     core target lists the exact includes).
//
// BUILD
//   g++ -O3 -std=gnu++23 -fopenmp -fPIC \
//     -mamx-tile -mamx-bf16 -mavx512f -mavx512bw -mavx512dq -mavx512vl \
//     -mavx512vnni -mavx512bf16 -msse4.1 -mavx -mavx2 -mfma -mf16c \
//     -mbmi -mbmi2 -mpopcnt \
//     -DLLAMINAR_COMPILED_WITH_AMX_BF16=1 -DLLAMINAR_COMPILED_WITH_AVX512_BF16=1 \
//     -DHAVE_ONEDNN -I src/v2 -I <generated-dir> -I <libfort>/lib \
//     bf16routecmp.cpp -o bf16routecmp \
//     -I external/onednn/build-avx512bf16/include \
//     -L external/onednn/build-avx512bf16/lib -ldnnl \
//     -Wl,-rpath,$PWD/external/onednn/build-avx512bf16/lib
//
// RUN
//   nice -n 15 taskset -c 0-7 ./bf16routecmp 8      # 8 threads
//   ./bf16routecmp 1                                 # single thread
//
// READING THE TABLE
//   * Column t/route = oneDNN_time/native_time; < 1.0 means the native
//     route is FASTER than oneDNN on that shape.
//   * The first line runs the exact AMX admission probe
//     (arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) followed by a
//     palette-1 tilezero). If it reports "no" or the process dies with
//     SIGILL, the AMX column prints n/a and the AMX tests below are the
//     hardware gate on a real AMX host.
//   * A/B are deliberately 64-byte aligned: the native AMX engine loads
//     tiles directly only from 64B-aligned bases (scratch-copy fallback
//     otherwise). Production weights/activations must match this posture.
//   * Exactness is separate from speed: on real AMX hardware run
//     'CPUBF16GemmAMX.*' (CMake LLAMINAR_CPU_ISA=AMXBF16 envelope) for the
//     5e-5 relative-L2-vs-serial gate. Virtualized AMX (e.g. the PCAI
//     appliance) is speed-representative but carries a small deterministic
//     per-cell deviation (~7e-3 relL2).
//
// RESULTS ON THE REFERENCE APPLIANCE (branch tip at time of writing)
//   8 threads: AMX-BF16 native beats oneDNN on all four shapes
//   (t/amx 0.51-0.88); AVX512-BF16 native 1.9-4.9x slower than oneDNN;
//   scalar FP32 native 21-41x slower (determinism-first by design).
// Route comparison harness: native (AMX-BF16 / AVX512-BF16 / FP32) vs oneDNN.
// Same process, interleaved rounds, min of 20, OpenMP threads.
//
// BUILD (run from a repo checkout that has external/onednn/build-avx512bf16):
//   export PATH=<toolchain>/bin:$PATH   (g++ >= 13 capable of -mamx-tile)
//   g++ -O3 -std=gnu++23 -fopenmp -fPIC \
//     -mamx-tile -mamx-bf16 -mavx512f -mavx512bw -mavx512dq -mavx512vl \
//     -mavx512vnni -mavx512bf16 -msse4.1 -mavx -mavx2 -mfma -mf16c \
//     -mbmi -mbmi2 -mpopcnt \
//     -DLLAMINAR_COMPILED_WITH_AMX_BF16=1 -DLLAMINAR_COMPILED_WITH_AVX512_BF16=1 \
//     -DHAVE_ONEDNN -I src/v2 -I <generated-dir> -I build/_deps/libfort-src/lib \
//     bf16routecmp.cpp -o bf16routecmp \
//     -I external/onednn/build-avx512bf16/include \
//     -L external/onednn/build-avx512bf16/lib -ldnnl \
//     -Wl,-rpath,$PWD/external/onednn/build-avx512bf16/lib
//
// RUN:
//   nice -n 15 taskset -c 0-7 ./bf16routecmp 8
//
// NOTES:
//  * A/B are 64-byte aligned on purpose: the native AMX engine loads tiles
//    directly only from 64B-aligned bases (this also matches the produced
//    weights-store posture).
//  * "l/amx" etc. = oneDNN_time/native_time; < 1.0 means native is faster.
//  * The host should be quiet for the numbers; the interleaved single-process
//    design keeps the native/oneDNN ratios meaningful under load.
//  * On a host where AMX is truly absent (arch_prctl refused or tilezero
//    SIGILLs) the AMX row prints "unusable" and native AMX is skipped.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <omp.h>
#include "kernels/cpu/gemm/CPUBF16Gemm.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <linux/prctl.h>
#include <oneapi/dnnl/dnnl.hpp>

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif

namespace cpu_bf16 = llaminar2::gemm::cpu_bf16;
namespace fp = llaminar2::gemm;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }
static float bf16f(uint16_t h){ uint32_t u=(uint32_t)h<<16; float f; memcpy(&f,&u,4); return f; }

static bool amxFunctional(){
    long r = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, 18); // XFEATURE_XTILEDATA
    if (r != 0) { printf("no (arch_prctl=%ld)\n", r); return false; }
    unsigned char cfg[64] = {}; cfg[0] = 1; // palette 1
    for (int i = 0; i < 8; ++i){ cfg[16+2*i] = 64; cfg[48+i] = 16; }
    _tile_loadconfig(cfg);
    _tile_zero(0);
    _tile_release();
    return true; // SIGILL would terminate before this point on a non-AMX host
}

int main(int argc, char** argv){
    const int T = argc > 1 ? atoi(argv[1]) : 8;
    omp_set_num_threads(T);
    printf("native vs oneDNN route comparison (same process, interleaved, min of 20, %d threads)\n", T);
    printf("AMX functional probe: ");
    const bool amx = amxFunctional();
    printf("YES (XTILEDATA granted, tilezero executed)\n\n");

    const int shapes[][3] = {{16,4096,4096},{32,4096,4096},{96,4096,4096},{32,8192,8192}};
    printf("%-15s | %12s %12s %7s | %12s %12s %7s | %12s %12s %7s\n",
           "shape","AMX-BF16","oneDNNBF","t/amx","AVX512BF","oneDNNBF","t/avx","FP32nat","oneDNN32","t/fp32");
    using namespace dnnl;
    for (auto& sh : shapes){
        const int M = sh[0], N = sh[1], K = sh[2];
        uint16_t* A16=(uint16_t*)aligned_alloc(64,(size_t)M*K*2);
        uint16_t* B16=(uint16_t*)aligned_alloc(64,(size_t)N*K*2);
        float* A32=(float*)aligned_alloc(64,(size_t)M*K*4);
        float* B32=(float*)aligned_alloc(64,(size_t)N*K*4);
        for(size_t i=0;i<(size_t)M*K;++i){ A16[i]=(uint16_t)(0x3F80|(i&0x7F)); A32[i]=0.5f+(float)((i*2654435761u)%97)/97.f; }
        for(size_t i=0;i<(size_t)N*K;++i){ B16[i]=(uint16_t)(0x3F80|((i*7)&0x7F)); B32[i]=0.25f+(float)((i*40503u)%89)/89.f; }
        std::vector<float> C((size_t)M*N,0.f);
        cpu_bf16::Engine sel;

        engine eng(engine::kind::cpu,0); stream str(eng);
        memory::desc sB({M,K},memory::data_type::bf16,memory::format_tag::ab);
        memory::desc wB({K,N},memory::data_type::bf16,memory::format_tag::ba);
        memory::desc dF({M,N},memory::data_type::f32,memory::format_tag::ab);
        matmul::primitive_desc pdB(eng,sB,wB,dF); matmul primB(pdB);
        memory msrc(sB,eng,A16), mwei(wB,eng,B16), mdst(dF,eng,C.data());
        memory::desc sF({M,K},memory::data_type::f32,memory::format_tag::ab);
        memory::desc wF({K,N},memory::data_type::f32,memory::format_tag::ba);
        matmul::primitive_desc pdF(eng,sF,wF,dF); matmul primF(pdF);
        memory fs(sF,eng,A32), fw(wF,eng,B32), fd(dF,eng,C.data());

        auto runL16=[&](){ primB.execute(str,{{DNNL_ARG_SRC,msrc},{DNNL_ARG_WEIGHTS,mwei},{DNNL_ARG_DST,mdst}}); str.wait(); };
        auto runL32=[&](){ primF.execute(str,{{DNNL_ARG_SRC,fs},{DNNL_ARG_WEIGHTS,fw},{DNNL_ARG_DST,fd}}); str.wait(); };

        double bestA=1e9,bestX=1e9,bestP=1e9,bestL16=1e9,bestL32=1e9;
        if(amx) cpu_bf16::run_amxbf16_matmul(A16,B16,C.data(),M,N,K,true,1.f,0.f);
        cpu_bf16::run_bf16_gemm(A16,B16,C.data(),M,N,K,true,1.f,0.f,cpu_bf16::Engine::AVX512BF16,&sel);
        fp::run_fp32_skinny_matmul(A32,B32,C.data(),M,N,K,true,1.f,0.f,nullptr,nullptr);
        for(int i=0;i<3;++i){ runL16(); runL32(); }

        for(int r=0;r<20;++r){
            if(amx){ auto t=Clock::now(); cpu_bf16::run_amxbf16_matmul(A16,B16,C.data(),M,N,K,true,1.f,0.f); double d=ms(t,Clock::now()); if(d<bestA)bestA=d; }
            { auto t=Clock::now(); cpu_bf16::run_bf16_gemm(A16,B16,C.data(),M,N,K,true,1.f,0.f,cpu_bf16::Engine::AVX512BF16,&sel); double d=ms(t,Clock::now()); if(d<bestX)bestX=d; }
            { auto t=Clock::now(); fp::run_fp32_skinny_matmul(A32,B32,C.data(),M,N,K,true,1.f,0.f,nullptr,nullptr); double d=ms(t,Clock::now()); if(d<bestP)bestP=d; }
            { auto t=Clock::now(); runL16(); double d=ms(t,Clock::now()); if(d<bestL16)bestL16=d; }
            { auto t=Clock::now(); runL32(); double d=ms(t,Clock::now()); if(d<bestL32)bestL32=d; }
        }
        auto gf=[&](double b){ return 2.0*M*N*K/1e9/b*1e3; };
        char sb[16]; snprintf(sb,sizeof sb,"%dx%dx%d",M,N,K);
        std::string amxG = amx ? std::to_string((int)gf(bestA)) : "n/a (no AMX)";
        std::string amxR = amx ? std::to_string(gf(bestL16)/gf(bestA)) : "n/a";
        printf("%-15s | %12s %12.0f %7s %12.0f %12.0f %6.2fx %12.0f %12.0f %6.2fx\n",
               sb, amxG.c_str(), gf(bestL16), amxR.c_str(),
               gf(bestX), gf(bestL16), gf(bestL16)/gf(bestX),
               gf(bestP), gf(bestL32), gf(bestL32)/gf(bestP));
        free(A16);free(B16);free(A32);free(B32);
    }
    printf("\nRatio semantics: column 't/route' = oneDNN_time/native_time; < 1.0 = native faster.\n");
    printf("oneDNN-BF16 is the reference for the AMX/AVX512 routes; oneDNN-F32 for the FP32 route.\n");
    return 0;
}
