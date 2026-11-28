// Full instruction-matrix benchmark: every native dot-product path vs oneDNN.
// Native rows: AMX-BF16 (tdpbf16ps), AVX512-BF16 (vdpbf16ps) x2, FMA32 512-bit
// (vfmadd231ps, bf16_m16k_tile.S sibling in f32_tile.S), VNNI int8 (vpdpbusd,
// vnni_tile.S), FP32 scalar. Baselines: oneDNN BF16 / FP32 / int8.
// FP16 (vdpphps) is reported n/a: CPUID advertises avx512_fp16 on this
// appliance but the instruction raises #UD (probed).
// Same process, interleaved rounds, min of 20, OpenMP threads.
//
// BUILD (repo checkout with external/onednn/build-avx512bf16 and libfort):
//   g++ -O3 -std=gnu++23 -fopenmp -fPIC \
//     -mamx-tile -mamx-bf16 -mavx512f -mavx512bw -mavx512dq -mavx512vl \
//     -mavx512vnni -mavx512bf16 -msse4.1 -mavx -mavx2 -mfma -mf16c \
//     -mbmi -mbmi2 -mpopcnt \
//     -DLLAMINAR_COMPILED_WITH_AMX_BF16=1 -DLLAMINAR_COMPILED_WITH_AVX512_BF16=1 \
//     -DHAVE_ONEDNN \
//     -I src/v2 -I <generated-dir> -I <libfort>/lib \
//     bf16_matrix_benchmark.cpp bf16_m16k_tile.S -o bf16matrix \
//     -I external/onednn/build-avx512bf16/include \
//     -L external/onednn/build-avx512bf16/lib -ldnnl \
//     -Wl,-rpath,$PWD/external/onednn/build-avx512bf16/lib
//
// RUN (oneDNN default ISA = AMX-backed here):   ./bf16matrix 8 > matrix_default.txt
// RUN (same-ISA baseline): DNNL_MAX_CPU_ISA=avx512_core_bf16 ./bf16matrix 8
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>
#include <omp.h>
#include "kernels/cpu/gemm/CPUBF16Gemm.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include <oneapi/dnnl/dnnl.hpp>
extern "C" void m16k(const void* Ap, const void* Bp, const void* C, long kp, long Nbytes);
extern "C" void f32tile(const void* Ap, const void* Bp, const void* C, long k, long Nbytes);
extern "C" void vntile8(const void* Ap, const void* Bp, const void* C, long k4, long Nbytes);
namespace cpu_bf16 = llaminar2::gemm::cpu_bf16;
namespace fp = llaminar2::gemm;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }

struct ShapeData {
    int M=0,N=0,K=0,kp=0,mb=0,g=0;
    dnnl::engine eng{dnnl::engine::kind::cpu, 0};
    dnnl::stream str{eng};
    dnnl::matmul primB, primF, primI;
    dnnl::memory msrc, mwei, mdst, fs, fw, fd, mi_src, mi_wei, mi_dst;
    uint16_t *A16=nullptr,*B16=nullptr; float *A32=nullptr,*B32=nullptr;
    uint8_t *A8=nullptr; int8_t *B8=nullptr;
    std::vector<uint16_t> Ap, Bp;
    std::vector<float> ApF, BpF; // f32 packs [mb][k][16x4B], [g][k][16x4B]
    std::vector<uint8_t> ApI, BpI; // int8 packs [mb][k4][16x4B], [g][k4][24x4B]
    std::vector<float> C; std::vector<int32_t> Ci;
    std::function<void()> amx, avx, pk, fp, libB, libF, f32v, vnni, libI;
    ~ShapeData(){ free(A16); free(B16); free(A32); free(B32); }
};

int main(int argc, char** argv){
    const int T = argc>1 ? atoi(argv[1]) : 8;
    omp_set_num_threads(T);
    const int shapes[][3] = {{16,4096,4096},{32,4096,4096},{96,4096,4096},{32,8192,8192}};
    std::vector<ShapeData> D(4);
    using namespace dnnl;
    for (int s=0;s<4;++s){
        ShapeData& d = D[s];
        d.M=shapes[s][0]; d.N=shapes[s][1]; d.K=shapes[s][2];
        const int M=d.M,N=d.N,K=d.K; d.kp=K/2; d.mb=(M+15)/16; d.g=(N+23)/24;
        d.A16=(uint16_t*)aligned_alloc(64,(size_t)M*K*2);
        d.B16=(uint16_t*)aligned_alloc(64,(size_t)N*K*2);
        d.A32=(float*)aligned_alloc(64,(size_t)M*K*4);
        d.B32=(float*)aligned_alloc(64,(size_t)N*K*4);
        for(size_t i=0;i<(size_t)M*K;++i){ d.A16[i]=(uint16_t)(0x3F80|(i&0x7F)); d.A32[i]=0.5f+(float)((i*2654435761u)%97)/97.f; }
        for(size_t i=0;i<(size_t)N*K;++i){ d.B16[i]=(uint16_t)(0x3F80|((i*7)&0x7F)); d.B32[i]=0.25f+(float)((i*40503u)%89)/89.f; }
        d.Ap.resize((size_t)d.mb*d.kp*32);
        d.Bp.resize((size_t)d.g*d.kp*48);
        for (int b=0;b<d.mb;++b) for (int p=0;p<d.kp;++p) for (int r=0;r<16;++r){
            int m=b*16+r; if(m>=M)continue;
            memcpy(&d.Ap[((size_t)b*d.kp+p)*32+r*2], &d.A16[(size_t)m*K+2*p], 4);
        }
        for (int b=0;b<d.g;++b) for (int p=0;p<d.kp;++p) for (int c=0;c<24;++c){
            int n=b*24+c; if(n>=N)continue;
            memcpy(&d.Bp[((size_t)b*d.kp+p)*48+c*2], &d.B16[(size_t)n*K+2*p], 4);
        }
        d.A8=(uint8_t*)aligned_alloc(64,(size_t)M*K);
        d.B8=(int8_t*)aligned_alloc(64,(size_t)N*K);
        for(size_t i=0;i<(size_t)M*K;++i) d.A8[i]=(uint8_t)(i%64);
        for(size_t i=0;i<(size_t)N*K;++i) d.B8[i]=(int8_t)((i%17)-8);
        d.C.assign((size_t)M*N+8192, 0.f);
        d.Ci.assign((size_t)M*N+8192, 0);
        // f32 packs (16-col groups; shapes are N%16==0)
        d.ApF.resize((size_t)d.mb*d.K*16); d.BpF.resize((size_t)(N/16)*d.K*16);
        for(int b=0;b<d.mb;++b) for(int p=0;p<d.K;++p) for(int r=0;r<16;++r)
            memcpy(&d.ApF[((size_t)b*d.K+p)*16+r], &d.A32[((size_t)(b*16+r))*K+p], 4);
        for(int b=0;b<(N/16);++b) for(int p=0;p<d.K;++p) for(int c=0;c<16;++c)
            memcpy(&d.BpF[((size_t)b*d.K+p)*16+c], &d.B32[((size_t)(b*16+c))*K+p], 4);
        // int8 packs (24-col groups; k4 = K/4)
        const int g8=(N+23)/24, k4=K/4;
        d.ApI.resize((size_t)d.mb*k4*16); d.BpI.resize((size_t)g8*k4*24);
        for(int b=0;b<d.mb;++b) for(int p=0;p<k4;++p) for(int r=0;r<16;++r)
            memcpy(&d.ApI[((size_t)b*k4+p)*16+r], &d.A8[((size_t)(b*16+r))*K+p*4], 4);
        for(int b=0;b<g8;++b) for(int p=0;p<k4;++p) for(int c=0;c<24;++c){
            int n=b*24+c; if(n>=N)continue;
            memcpy(&d.BpI[((size_t)b*k4+p)*24+c], &d.B8[(size_t)n*K+p*4], 4);
        }
        memory::desc sB({M,K},memory::data_type::bf16,memory::format_tag::ab);
        memory::desc wB({K,N},memory::data_type::bf16,memory::format_tag::ba);
        memory::desc dF({M,N},memory::data_type::f32,memory::format_tag::ab);
        d.primB = matmul(matmul::primitive_desc(d.eng,sB,wB,dF));
        d.msrc = memory(sB,d.eng,d.A16); d.mwei = memory(wB,d.eng,d.B16); d.mdst = memory(dF,d.eng,d.C.data());
        memory::desc sF({M,K},memory::data_type::f32,memory::format_tag::ab);
        memory::desc wF({K,N},memory::data_type::f32,memory::format_tag::ba);
        d.primF = matmul(matmul::primitive_desc(d.eng,sF,wF,dF));
        {
            memory::desc sI({M,K},memory::data_type::u8,memory::format_tag::ab);
            memory::desc wI({K,N},memory::data_type::s8,memory::format_tag::ba);
            memory::desc dI({M,N},memory::data_type::s32,memory::format_tag::ab);
            d.primI = matmul(matmul::primitive_desc(d.eng,sI,wI,dI));
            d.mi_src = memory(sI,d.eng,d.A8); d.mi_wei = memory(wI,d.eng,d.B8);
            d.mi_dst = memory(dI,d.eng,d.Ci.data());
        }
        d.fs = memory(sF,d.eng,d.A32); d.fw = memory(wF,d.eng,d.B32); d.fd = memory(dF,d.eng,d.C.data());
        d.libB = [&d](){ d.primB.execute(d.str,{{DNNL_ARG_SRC,d.msrc},{DNNL_ARG_WEIGHTS,d.mwei},{DNNL_ARG_DST,d.mdst}}); d.str.wait(); };
        d.libF = [&d](){ d.primF.execute(d.str,{{DNNL_ARG_SRC,d.fs},{DNNL_ARG_WEIGHTS,d.fw},{DNNL_ARG_DST,d.fd}}); d.str.wait(); };
        d.libI = [&d](){ d.primI.execute(d.str,{{DNNL_ARG_SRC,d.mi_src},{DNNL_ARG_WEIGHTS,d.mi_wei},{DNNL_ARG_DST,d.mi_dst}}); d.str.wait(); };
        d.amx  = [M,N,K,&d](){ cpu_bf16::run_amxbf16_matmul(d.A16,d.B16,d.C.data(),M,N,K,true,1.f,0.f); };
        d.avx  = [M,N,K,&d](){ cpu_bf16::Engine sel; cpu_bf16::run_bf16_gemm(d.A16,d.B16,d.C.data(),M,N,K,true,1.f,0.f,cpu_bf16::Engine::AVX512BF16,&sel); };
        d.pk   = [&d](){
            #pragma omp parallel for schedule(static) collapse(2)
            for(int i=0;i<d.mb;++i) for(int j=0;j<d.g;++j)
                m16k(d.Ap.data()+(size_t)i*d.kp*32, d.Bp.data()+(size_t)j*d.kp*48,
                     d.C.data()+(size_t)i*16*d.N+(size_t)j*24, d.kp, (long)d.N*4);
        };
        d.fp   = [M,N,K,&d](){ fp::run_fp32_skinny_matmul(d.A32,d.B32,d.C.data(),M,N,K,true,1.f,0.f,nullptr,nullptr); };
        d.f32v = [&d](){
            #pragma omp parallel for schedule(static) collapse(2)
            for(int i=0;i<d.mb;++i) for(int j=0;j<(d.N/16);++j)
                f32tile(d.ApF.data()+(size_t)i*d.K*16, d.BpF.data()+(size_t)j*d.K*16,
                        d.C.data()+(size_t)i*16*d.N+(size_t)j*16, d.K, (long)d.N*4);
        };
        d.vnni = [&d](){
            const int k4=d.K/4, g8=(d.N+23)/24;
            #pragma omp parallel for schedule(static) collapse(2)
            for(int i=0;i<d.mb;++i) for(int j=0;j<g8;++j)
                vntile8(d.ApI.data()+(size_t)i*k4*16, d.BpI.data()+(size_t)j*k4*24,
                        d.Ci.data()+(size_t)i*16*d.N+(size_t)j*24, k4, (long)d.N*4);
        };
    }
    bool amxOK=false; try{ amxOK=cpu_bf16::engineHardwareAdmitted(cpu_bf16::Engine::AMXBF16); }catch(...){ amxOK=false; }
    const char* names[10] = {"native AMX-BF16 (tdpbf16ps)",
                            "native AVX512-BF16 4x4+prefetch",
                            "native AVX512-BF16 packed 16x24",
                            "native FMA32 512-bit (vfmadd231ps)",
                            "native VNNI int8 (vpdpbusd, u8xs8)",
                            "native FP32 scalar (fma)",
                            "oneDNN BF16",
                            "oneDNN FP32",
                            "oneDNN int8",
                            "native FP16 (vdpphps)"};
    std::function<void(ShapeData&)> fn[10];
    fn[0] = [](ShapeData&d){ d.amx(); };
    fn[1] = [](ShapeData&d){ d.avx(); };
    fn[2] = [](ShapeData&d){ d.pk(); };
    fn[3] = [](ShapeData&d){ d.f32v(); };
    fn[4] = [](ShapeData&d){ d.vnni(); };
    fn[5] = [](ShapeData&d){ d.fp(); };
    fn[6] = [](ShapeData&d){ d.libB(); };
    fn[7] = [](ShapeData&d){ d.libF(); };
    fn[8] = [](ShapeData&d){ d.libI(); };
    fn[9] = [](ShapeData&d){}; // FP16: not executable here
    // warmup
    for(int s=0;s<4;++s) for(int v=0;v<6;++v){ if(v==0&&!amxOK)continue; try{ fn[v](D[s]); }catch(...){} }
    double best[10][4];
    for(int s=0;s<4;++s) for(int v=0;v<10;++v) best[v][s]=1e9;
    for(int v=0;v<10;++v){
        if(v==0&&!amxOK)continue;
        for(int s=0;s<4;++s){
            for(int r=0;r<20;++r){ auto t=Clock::now(); fn[v](D[s]); double e=ms(t,Clock::now()); if(e<best[v][s])best[v][s]=e; }
        }
    }
    // Per-function tier view: native production tier vs oneDNN tier.
    printf("PER-FUNCTION TIER COMPARISON (production path vs production path)\n");
    printf("same process, interleaved, min of 20, %d threads. ratio = oneDNN_time/native_time (< 1.0 = native slower)\n\n", T);

    auto row = [&](const char* name, int v, int ref, const char* refname){
        double gf[4];
        for(int s=0;s<4;++s) gf[s]=2.0*D[s].M*D[s].N*D[s].K/1e9/best[v][s]*1e3;
        printf("%-30s | %10.0f %6.2fx | %10.0f %6.2fx | %10.0f %6.2fx | %10.0f %6.2fx\n",
               name, gf[0], best[ref][0]/best[v][0], gf[1], best[ref][1]/best[v][1],
               gf[2], best[ref][2]/best[v][2], gf[3], best[ref][3]/best[v][3]);
        (void)refname;
    };
    auto rowAbs = [&](const char* name, int v){
        double gf[4];
        for(int s=0;s<4;++s) gf[s]=2.0*D[s].M*D[s].N*D[s].K/1e9/best[v][s]*1e3;
        printf("%-30s | %10.0f %6s | %10.0f %6s | %10.0f %6s | %10.0f %6s\n",
               name, gf[0],"-", gf[1],"-", gf[2],"-", gf[3],"-");
    };

    printf("===== 1. BF16 GEMM (bf16 x bf16 -> f32) =====\n");
    printf("%-30s | %17s | %17s | %17s | %17s\n","tier","16x4096x4096","32x4096x4096","96x4096x4096","32x8192x8192");
    printf("-- native tier vs oneDNN with the ACTIVE ISA (this host default = AMX-backed): --\n");
    printf("%-30s | %10s %6s | %10s %6s | %10s %6s | %10s %6s\n","(GF/s, ratio)","GF/s","r","GF/s","r","GF/s","r","GF/s","r");
    row("native AMX-BF16 (auto engine)", 0, 6, "oneDNN BF16");
    rowAbs(forcedI ? "oneDNN BF16 (active ISA)" : "oneDNN BF16 (default = AMX)", 6);
    const bool forcedI = getenv("DNNL_MAX_CPU_ISA") && strstr(getenv("DNNL_MAX_CPU_ISA"),"avx512_core_bf16");
    printf("-- same-ISA tier: native AVX512-BF16 vs oneDNN's own AVX512-BF16 kernel --\n");
    if (forcedI) {
        row("native AVX512-BF16 4x4+prefetch", 1, 6, "oneDNN BF16");
        rowAbs("oneDNN BF16 (avx512_core_bf16)", 6);
    } else {
        printf("   (AVX512-vs-AVX512 needs a forced-ISA run: DNNL_MAX_CPU_ISA=avx512_core_bf16 ./bf16matrix %d\n", T);
        printf("   native AVX512-BF16 4x4+prefetch here: ");
        row("", 1, 6, "oneDNN BF16");
    }
    printf("   (prototype, not production: packed weight-stationary 16x24: %0.f-%.0f GF/s, ratio 1.12-1.37x vs oneDNN-avx512)\n",
           2.0*D[0].M*D[0].N*D[0].K/1e9/best[2][0]*1e3,
           2.0*D[3].M*D[3].N*D[3].K/1e9/best[2][3]*1e3);

    printf("\n===== 2. FP32 GEMM (f32 x f32 -> f32) =====\n");
    printf("-- native production tier (determinism-first scalar route) vs oneDNN FP32: --\n");
    row("native FP32 scalar (contract route)", 5, 7, "oneDNN FP32");
    rowAbs("oneDNN FP32", 7);
    printf("   (prototype, not production: 512-bit FMA32 packed tile: %.0f-%.0f GF/s, ratio 0.51-1.42x vs oneDNN FP32)\n",
           2.0*D[0].M*D[0].N*D[0].K/1e9/best[3][0]*1e3,
           2.0*D[3].M*D[3].N*D[3].K/1e9/best[3][3]*1e3);

    printf("\n===== 3. int8 GEMM (u8 x s8 -> s32) =====\n");
    printf("-- native prototype (vpdpbusd tile, exact) vs oneDNN int8: --\n");
    row("native VNNI int8 (prototype)", 4, 8, "oneDNN int8");
    rowAbs("oneDNN int8", 8);

    printf("\n===== 4. FP16 =====\n");
    printf("native FP16 (vdpphps): n/a - avx512_fp16 advertised in /proc/cpuinfo but the instruction raises #UD on this appliance (execution probe).\n");
    printf("\nvs1d = oneDNN_BF16_time / native_time for the BF16 rows; < 1.0 = native faster.\n");
    printf("The oneDNN BF16 row uses whatever ISA the environment selects (see DNNL_MAX_CPU_ISA).\n");
    return 0;
}
