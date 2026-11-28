// Full instruction-matrix benchmark: every native BF16/FP32 path vs oneDNN.
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
namespace cpu_bf16 = llaminar2::gemm::cpu_bf16;
namespace fp = llaminar2::gemm;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }

struct ShapeData {
    int M=0,N=0,K=0,kp=0,mb=0,g=0;
    dnnl::engine eng{dnnl::engine::kind::cpu, 0};
    dnnl::stream str{eng};
    dnnl::matmul primB, primF;
    dnnl::memory msrc, mwei, mdst, fs, fw, fd;
    uint16_t *A16=nullptr,*B16=nullptr; float *A32=nullptr,*B32=nullptr;
    std::vector<uint16_t> Ap, Bp; std::vector<float> C;
    std::function<void()> amx, avx, pk, fp, libB, libF;
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
        d.C.assign((size_t)M*N+8192, 0.f);
        memory::desc sB({M,K},memory::data_type::bf16,memory::format_tag::ab);
        memory::desc wB({K,N},memory::data_type::bf16,memory::format_tag::ba);
        memory::desc dF({M,N},memory::data_type::f32,memory::format_tag::ab);
        d.primB = matmul(matmul::primitive_desc(d.eng,sB,wB,dF));
        d.msrc = memory(sB,d.eng,d.A16); d.mwei = memory(wB,d.eng,d.B16); d.mdst = memory(dF,d.eng,d.C.data());
        memory::desc sF({M,K},memory::data_type::f32,memory::format_tag::ab);
        memory::desc wF({K,N},memory::data_type::f32,memory::format_tag::ba);
        d.primF = matmul(matmul::primitive_desc(d.eng,sF,wF,dF));
        d.fs = memory(sF,d.eng,d.A32); d.fw = memory(wF,d.eng,d.B32); d.fd = memory(dF,d.eng,d.C.data());
        d.libB = [&d](){ d.primB.execute(d.str,{{DNNL_ARG_SRC,d.msrc},{DNNL_ARG_WEIGHTS,d.mwei},{DNNL_ARG_DST,d.mdst}}); d.str.wait(); };
        d.libF = [&d](){ d.primF.execute(d.str,{{DNNL_ARG_SRC,d.fs},{DNNL_ARG_WEIGHTS,d.fw},{DNNL_ARG_DST,d.fd}}); d.str.wait(); };
        d.amx  = [M,N,K,&d](){ cpu_bf16::run_amxbf16_matmul(d.A16,d.B16,d.C.data(),M,N,K,true,1.f,0.f); };
        d.avx  = [M,N,K,&d](){ cpu_bf16::Engine sel; cpu_bf16::run_bf16_gemm(d.A16,d.B16,d.C.data(),M,N,K,true,1.f,0.f,cpu_bf16::Engine::AVX512BF16,&sel); };
        d.pk   = [&d](){
            #pragma omp parallel for schedule(static) collapse(2)
            for(int i=0;i<d.mb;++i) for(int j=0;j<d.g;++j)
                m16k(d.Ap.data()+(size_t)i*d.kp*32, d.Bp.data()+(size_t)j*d.kp*48,
                     d.C.data()+(size_t)i*16*d.N+(size_t)j*24, d.kp, (long)d.N*4);
        };
        d.fp   = [M,N,K,&d](){ fp::run_fp32_skinny_matmul(d.A32,d.B32,d.C.data(),M,N,K,true,1.f,0.f,nullptr,nullptr); };
    }
    bool amxOK=false; try{ amxOK=cpu_bf16::engineHardwareAdmitted(cpu_bf16::Engine::AMXBF16); }catch(...){ amxOK=false; }
    const char* names[6] = {"native AMX-BF16 (tdpbf16ps)",
                            "native AVX512-BF16 4x4+prefetch",
                            "native AVX512-BF16 packed 16x24",
                            "native FP32 scalar (fma)",
                            "oneDNN BF16",
                            "oneDNN FP32"};
    std::function<void(ShapeData&)> fn[6];
    fn[0] = [](ShapeData&d){ d.amx(); };
    fn[1] = [](ShapeData&d){ d.avx(); };
    fn[2] = [](ShapeData&d){ d.pk(); };
    fn[3] = [](ShapeData&d){ d.fp(); };
    fn[4] = [](ShapeData&d){ d.libB(); };
    fn[5] = [](ShapeData&d){ d.libF(); };
    // warmup
    for(int s=0;s<4;++s) for(int v=0;v<6;++v){ if(v==0&&!amxOK)continue; try{ fn[v](D[s]); }catch(...){} }
    double best[6][4];
    for(int s=0;s<4;++s) for(int v=0;v<6;++v) best[v][s]=1e9;
    for(int v=0;v<6;++v){
        if(v==0&&!amxOK)continue;
        for(int s=0;s<4;++s){
            for(int r=0;r<20;++r){ auto t=Clock::now(); fn[v](D[s]); double e=ms(t,Clock::now()); if(e<best[v][s])best[v][s]=e; }
        }
    }
    printf("full BF16/FP32 instruction matrix (same process, interleaved, min of 20, %d threads)\n\n", T);
    printf("%-34s | %11s %8s | %11s %8s | %11s %8s | %11s %8s\n",
           "variant","16x4096","vs1d","32x4096","vs1d","96x4096","vs1d","32x8192","vs1d");
    for(int v=0;v<6;++v){
        if(v==0&&!amxOK){ printf("%-34s |  n/a (AMX not admitted on this host)\n", names[v]); continue; }
        double gf[4];
        for(int s=0;s<4;++s) gf[s]=2.0*D[s].M*D[s].N*D[s].K/1e9/best[v][s]*1e3;
        if(v<4){
            // ratios vs oneDNN BF16
            printf("%-34s | %11.0f %7.2fx %11.0f %7.2fx %11.0f %7.2fx %11.0f %7.2fx\n",
                   names[v], gf[0], best[4][0]/best[v][0], gf[1], best[4][1]/best[v][1],
                   gf[2], best[4][2]/best[v][2], gf[3], best[4][3]/best[v][3]);
        } else if(v==4){
            printf("%-34s | %11.0f %7s %11.0f %7s %11.0f %7s %11.0f %7s\n",
                   names[v], gf[0],"-", gf[1],"-", gf[2],"-", gf[3],"-");
        } else {
            printf("%-34s | %11.0f %7.2fx %11.0f %7.2fx %11.0f %7.2fx %11.0f %7.2fx\n",
                   names[v], gf[0], best[5][0]/best[v][0], gf[1], best[5][1]/best[v][1],
                   gf[2], best[5][2]/best[v][2], gf[3], best[5][3]/best[v][3]);
        }
    }
    printf("\nvs1d = oneDNN_BF16_time / native_time for the BF16 rows; < 1.0 = native faster.\n");
    printf("The oneDNN BF16 row uses whatever ISA the environment selects (see DNNL_MAX_CPU_ISA).\n");
    return 0;
}
