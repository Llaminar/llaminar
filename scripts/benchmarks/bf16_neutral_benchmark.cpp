#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include "kernels/cpu/gemm/CPUBF16Gemm.h"
#include <oneapi/dnnl/dnnl.hpp>
namespace cb = llaminar2::gemm::cpu_bf16;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a,Clock::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }
int main(){
    omp_set_num_threads(8);
    using namespace dnnl;
    const int shapes[][3]={{16,4096,4096},{32,4096,4096},{96,4096,4096},{32,8192,8192}};
    // measure everything into arrays, then print one clean table
    double ba[4],bx[4],bl[4];
    bool forced = getenv("DNNL_MAX_CPU_ISA") && strstr(getenv("DNNL_MAX_CPU_ISA"),"avx512_core_bf16");
    cb::Engine autoSel[4];
    for(int s=0;s<4;++s){
        const int M=shapes[s][0],N=shapes[s][1],K=shapes[s][2];
        uint16_t* A=(uint16_t*)aligned_alloc(64,(size_t)M*K*2);
        uint16_t* B=(uint16_t*)aligned_alloc(64,(size_t)N*K*2);
        for(size_t i=0;i<(size_t)M*K;++i)A[i]=(uint16_t)(0x3F80|(i&0x7F));
        for(size_t i=0;i<(size_t)N*K;++i)B[i]=(uint16_t)(0x3F80|((i*7)&0x7F));
        std::vector<float>C((size_t)M*N+64,0.f);
        engine eng(engine::kind::cpu,0); stream str(eng);
        memory::desc sB({M,K},memory::data_type::bf16,memory::format_tag::ab);
        memory::desc wB({K,N},memory::data_type::bf16,memory::format_tag::ba);
        memory::desc dF({M,N},memory::data_type::f32,memory::format_tag::ab);
        matmul prim(matmul::primitive_desc(eng,sB,wB,dF));
        memory msrc(sB,eng,A),mwei(wB,eng,B),mdst(dF,eng,C.data());
        auto runLib=[&](){ prim.execute(str,{{DNNL_ARG_SRC,msrc},{DNNL_ARG_WEIGHTS,mwei},{DNNL_ARG_DST,mdst}}); str.wait(); };
        auto runLAMX=[&](){ cb::run_bf16_gemm(A,B,C.data(),M,N,K,true,1.f,0.f,cb::Engine::Automatic,&autoSel[s]); };
        auto runLAVX=[&](){ cb::run_bf16_gemm(A,B,C.data(),M,N,K,true,1.f,0.f,cb::Engine::AVX512BF16,&autoSel[s]); };
        runLAMX(); runLAVX(); runLib();
        ba[s]=bx[s]=bl[s]=1e9;
        for(int r=0;r<20;++r){
            {auto t=Clock::now();runLAMX();double d=ms(t,Clock::now());if(d<ba[s])ba[s]=d;}
            {auto t=Clock::now();runLAVX();double d=ms(t,Clock::now());if(d<bx[s])bx[s]=d;}
            {auto t=Clock::now();runLib();double d=ms(t,Clock::now());if(d<bl[s])bl[s]=d;}
        }
        free(A);free(B);
    }
    auto gf=[&](int s,double b){ return 2.0*shapes[s][0]*shapes[s][1]*shapes[s][2]/1e9/b*1e3; };
    printf("NEUTRAL: Llaminar engine vs oneDNN - same process, interleaved, min of 20, 8 threads\n");
    printf("ratio = oneDNN_time / Llaminar_time (>1.0 = Llaminar faster)\n\n");
    printf("%-40s | %11s %7s | %11s %7s | %11s %7s | %11s %7s\n",
           "tier","16x4096","ratio","32x4096","ratio","96x4096","ratio","32x8192","ratio");
    {
        char nm[64]; snprintf(nm,sizeof nm,"Llaminar engine (auto -> %s)", cb::engineName(autoSel[0]));
        printf("%-40s | %11.0f %6.2fx %11.0f %6.2fx %11.0f %6.2fx %11.0f %6.2fx\n",
               nm, gf(0,ba[0]), bl[0]/ba[0], gf(1,ba[1]), bl[1]/ba[1],
               gf(2,ba[2]), bl[2]/ba[2], gf(3,ba[3]), bl[3]/ba[3]);
    }
    printf("%-40s | %11.0f %6s | %11.0f %6s | %11.0f %6s | %11.0f %6s\n",
           "oneDNN BF16 (active ISA)", gf(0,bl[0]),"-", gf(1,bl[1]),"-", gf(2,bl[2]),"-", gf(3,bl[3]),"-");
    printf("%-40s | %11.0f %6.2fx %11.0f %6.2fx %11.0f %6.2fx %11.0f %6.2fx\n",
           "Llaminar engine (explicit AVX512BF16)", gf(0,bx[0]), bl[0]/bx[0], gf(1,bx[1]), bl[1]/bx[1],
           gf(2,bx[2]), bl[2]/bx[2], gf(3,bx[3]), bl[3]/bx[3]);
    printf("note: %s\n", forced
           ? "oneDNN ran under DNNL_MAX_CPU_ISA=avx512_core_bf16, so the AVX512 row is a same-ISA neutral comparison."
           : "oneDNN ran with its default ISA selection (AMX on this appliance); the AVX512 row here is NOT same-ISA - rerun with DNNL_MAX_CPU_ISA=avx512_core_bf16 for that view.");
    return 0;
}
