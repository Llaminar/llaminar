// Packed M-in-lanes AVX512-BF16 vs oneDNN (forced avx512_core_bf16) comparison.
// Closes the last AVX512-BF16 gap (32x8192x8192) with the weight-stationary
// packed 16x24 tile (see bf16_m16k_tile.S).
//
// BUILD (repo checkout with external/onednn/build-avx512bf16):
//   g++ -O3 -std=gnu++23 -fopenmp -mavx512f -mavx512bw -mavx512dq -mavx512vl \
//     -mavx512vnni -mavx512bf16 -msse4.1 -mavx -mavx2 -mfma -mf16c \
//     bf16_avx512_packed_benchmark.cpp bf16_m16k_tile.S -o bf16packbench \
//     -I external/onednn/build-avx512bf16/include \
//     -L external/onednn/build-avx512bf16/lib -ldnnl \
//     -Wl,-rpath,$PWD/external/onednn/build-avx512bf16/lib
//
// RUN (forces oneDNN onto its own AVX512-BF16 kernel for a fair same-ISA run):
//   DNNL_MAX_CPU_ISA=avx512_core_bf16 nice -n 15 taskset -c 0-7 ./bf16packbench 32 8192 8192 8
//
// NOTE: B is packed ONCE outside the timed region (weight-stationary
// deployment, PreparedWeightStore-owned); the report prints the compute-only
// ratio. ratio > 1.0 means the native packed route is faster than oneDNN.
extern "C" void m16k(const void* Ap,const void* Bp,const void* C,long kp,long Nbytes);
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>
#include <oneapi/dnnl/dnnl.hpp>
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a,Clock::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }
int main(int argc,char**argv){
    int M=atoi(argv[1]),N=atoi(argv[2]),K=atoi(argv[3]);
    int T=argc>4?atoi(argv[4]):8;
    omp_set_num_threads(T);
    const int kp=K/2, mb=(M+15)/16, g=(N+23)/24;
    std::vector<uint16_t>A((size_t)M*K),B((size_t)N*K);
    for(size_t i=0;i<A.size();++i)A[i]=(uint16_t)(0x3F80|(i&0x7F));
    for(size_t i=0;i<B.size();++i)B[i]=(uint16_t)(0x3F80|((i*7)&0x7F));
    std::vector<uint16_t>Ap((size_t)mb*kp*32),Bp((size_t)g*kp*48);
    for(int b=0;b<mb;++b)for(int p=0;p<kp;++p)for(int r=0;r<16;++r){
        int m=b*16+r; if(m>=M)continue;
        memcpy(&Ap[((size_t)b*kp+p)*32+r*2],&A[(size_t)m*K+2*p],4);
    }
    for(int b=0;b<g;++b)for(int p=0;p<kp;++p)for(int c=0;c<24;++c){
        int n=b*24+c; if(n>=N)continue;
        memcpy(&Bp[((size_t)b*kp+p)*48+c*2],&B[(size_t)n*K+2*p],4);
    }
    std::vector<float>C((size_t)M*N+8192,0.f);
    const auto runNative=[&](){
        #pragma omp parallel for schedule(static) collapse(2)
        for(int i=0;i<mb;++i)for(int j=0;j<g;++j)
            m16k(Ap.data()+(size_t)i*kp*32, Bp.data()+(size_t)j*kp*48, C.data()+(size_t)i*16*N+(size_t)j*24, kp, (long)N*4);
    };
    runNative();
    using namespace dnnl;
    engine eng(engine::kind::cpu,0); stream str(eng);
    memory::desc sB({M,K},memory::data_type::bf16,memory::format_tag::ab);
    memory::desc wB({K,N},memory::data_type::bf16,memory::format_tag::ba);
    memory::desc dF({M,N},memory::data_type::f32,memory::format_tag::ab);
    matmul::primitive_desc pd(eng,sB,wB,dF); matmul prim(pd);
    memory msrc(sB,eng,A.data()),mwei(wB,eng,B.data()),mdst(dF,eng,C.data());
    auto runLib=[&](){ prim.execute(str,{{DNNL_ARG_SRC,msrc},{DNNL_ARG_WEIGHTS,mwei},{DNNL_ARG_DST,mdst}}); str.wait(); };
    for(int i=0;i<3;++i){ runNative(); runLib(); }
    double bn=1e9,bl=1e9;
    for(int r=0;r<20;++r){
        {auto t=Clock::now();runNative();double d=ms(t,Clock::now());if(d<bn)bn=d;}
        {auto t=Clock::now();runLib();double d=ms(t,Clock::now());if(d<bl)bl=d;}
    }
    double gl=2.0*M*N*K/1e9;
    printf("m16k-packed: %.3f ms (%.0f GF/s)   oneDNN-avx512: %.3f ms (%.0f GF/s)  ratio=%.3f\n",
           bn, gl/bn*1e3, bl, gl/bl*1e3, bl/bn);
    return 0;
}
