// 2D instruction-tier matrix: Llaminar engines (columns, low->high) x
// oneDNN tiers (rows, low->high). Cell = oneDNN_time / Llaminar_time
// (> 1.0 = Llaminar faster). Aligned same-instruction-type cells are the
// diagonal and should be >= 1.0.
//
// oneDNN's ISA is process-wide, so run twice:
//   ./tier2d 8            -> oneDNN default tiers (AMX-backed BF16/int8 here)
//   DNNL_MAX_CPU_ISA=avx512_core_bf16 ./tier2d 8  -> oneDNN avx512 tiers
//
// BUILD: same flags as bf16_matrix_benchmark.cpp plus the three .S tiles.
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>
#include <omp.h>
#include "kernels/cpu/gemm/CPUBF16Gemm.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include <oneapi/dnnl/dnnl.hpp>
extern "C" void m16k(const void*, const void*, const void*, long, long);
extern "C" void f32tile(const void*, const void*, const void*, long, long);
extern "C" void vntile8(const void*, const void*, const void*, long, long);
extern "C" void avx2f32(const void*, const void*, const void*, long, long);
namespace cb = llaminar2::gemm::cpu_bf16;
namespace fp = llaminar2::gemm;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }

struct ShapeData {
    int M=0,N=0,K=0,kp=0,mb=0,mb2=0,g=0,g8=0,k4=0;
    uint16_t *A16=nullptr,*B16=nullptr; float *A32=nullptr,*B32=nullptr;
    uint8_t *A8=nullptr; int8_t *B8=nullptr;
    std::vector<uint16_t> Ap, Bp;
    std::vector<float> ApF, BpF;
    std::vector<uint8_t> ApI; std::vector<int8_t> BpI;
    std::vector<float> Ap2, Bp2; // avx2 packs [mb][k][8x4B], [g][k][8x4B]
    std::vector<float> C; std::vector<int32_t> Ci;
    dnnl::engine eng{dnnl::engine::kind::cpu,0};
    dnnl::stream str{eng};
    dnnl::matmul primB, primF, primI;
    dnnl::memory msrc,mwei,mdst,fs,fw,fd,isrc,iwei,idst;
    ~ShapeData(){ free(A16); free(B16); free(A32); free(B32); free(A8); free(B8); }
};

static const char* kRowFile = "/tmp/tier2d_rows.txt";
static const char* Lnames[7]={"FP32 scalar","AVX2-FMA32 8x8","AVX512-BF16 4x4","AVX512-BF16 packed","FMA32 512","VNNI int8","AMX-BF16"};
static const char* Onames[7]={"oneDNN FP32 (avx512_core)",
                               "oneDNN int8 (avx512 vnni)",
                               "oneDNN BF16 (avx512_core_bf16)",
                               "oneDNN int8 (AMX int8)",
                               "oneDNN BF16 (AMX, default)",
                               "oneDNN BF16 (avx512_core_bf16, other)",
                               "(spare)"};
static int showMode(int){
    FILE* f=fopen(kRowFile,"r");
    if(!f){ printf("no saved rows - run once plain and once with DNNL_MAX_CPU_ISA=avx512_core_bf16 first\n"); return 1; }
    double mn[8][7]={}, mx[8][7]={}; bool have[8][7]={};
    char line[512];
    while(fgets(line,sizeof line,f)){
        // row|<canon>|v0min,v0max|v1min,v1max|...|v5min,v5max|
        int canon=-1;
        if(sscanf(line,"row|%d|",&canon)!=1 || canon<0 || canon>7) continue;
        char* p=strchr(line+5,'|');
        if(!p) continue;
        for(int l=0;l<7;++l){
            double a=0,b=0;
            if(sscanf(p+1,"%lf,%lf|",&a,&b)!=2) break;
            if(a<0){ p=strchr(p+1,'|'); if(!p) break; continue; }
            if(!have[canon][l] || a<mn[canon][l]) mn[canon][l]=a;
            if(!have[canon][l] || b>mx[canon][l]) mx[canon][l]=b;
            have[canon][l]=true;
            p=strchr(p+1,'|'); if(!p) break;
        }
    }
    fclose(f);
    printf("SINGLE 2D INSTRUCTION-TIER MATRIX: oneDNN tiers (rows, low->high) x Llaminar engines (cols, low->high)\n");
    printf("cell = oneDNN_time / Llaminar_time (range over the 4 shapes); >1.0 = Llaminar faster\n\n");
    const char* RN[8]={"oneDNN FP32 (avx2) - n/a: no avx2 matmul tier",
                       "oneDNN int8 (avx2) - n/a: no avx2 matmul tier",
                       "oneDNN BF16 (avx2) - n/a: no avx2 matmul tier",
                       "oneDNN FP32 (avx512_core)",
                       "oneDNN int8 (avx512 vnni)",
                       "oneDNN BF16 (avx512_core_bf16)",
                       "oneDNN int8 (AMX int8)",
                       "oneDNN BF16 (AMX, default)"};
    printf("%-24s | %-18s | %-18s | %-18s | %-18s | %-18s | %-18s | %-18s\n","","FP32 scalar","AVX2-FMA32 8x8","AVX512-BF16 4x4","AVX512-BF16 pk","FMA32 512","VNNI int8","AMX-BF16");
    for(int c=0;c<8;++c){
        printf("%-26s |", RN[c]);
        for(int l=0;l<7;++l){
            char cell[28];
            if(!have[c][l]) snprintf(cell,sizeof cell,"%s","-");
            else if(mx[c][l]-mn[c][l]<0.05) snprintf(cell,sizeof cell,"%.2fx",(mn[c][l]+mx[c][l])/2);
            else snprintf(cell,sizeof cell,"%.2f-%.2fx",mn[c][l],mx[c][l]);
            printf(" %-18s |", cell);
        }
        printf("\n");
    }
    printf("\naligned same-instruction-type cells (should be >= 1.0): AVX512-BF16 cols x row oneDNN BF16(avx512); AMX-BF16 col x row oneDNN BF16(AMX); VNNI int8 col x row oneDNN int8(avx512); AVX2-FMA32 col has no oneDNN matmul counterpart (no avx2 matmul tier); FP32 scalar col x oneDNN FP32 row is 0.04x by design.\n");
    printf("not executable on this appliance (verified): FP16 vdpphps #UD; AMX-int8 tdpbusd executes but returns zeros; VNNI-16 vpdpwssd has an unresolved deterministic deviation.\n");
    return 0;
}
int main(int argc, char** argv){
    if(argc>1 && !strcmp(argv[1],"--show")) return showMode(0);
    const int T = argc>1 ? atoi(argv[1]) : 8;
    omp_set_num_threads(T);
    const int shapes[][3]={{16,4096,4096},{32,4096,4096},{96,4096,4096},{32,8192,8192}};
    std::vector<ShapeData> D(4);
    using namespace dnnl;
    for(int s=0;s<4;++s){
        ShapeData& d=D[s];
        d.M=shapes[s][0]; d.N=shapes[s][1]; d.K=shapes[s][2];
        const int M=d.M,N=d.N,K=d.K; d.kp=K/2; d.mb=(M+15)/16; d.g=(N+23)/24; d.g8=(N+23)/24; d.k4=K/4;
        d.A16=(uint16_t*)aligned_alloc(64,(size_t)M*K*2);
        d.B16=(uint16_t*)aligned_alloc(64,(size_t)N*K*2);
        d.A32=(float*)aligned_alloc(64,(size_t)M*K*4);
        d.B32=(float*)aligned_alloc(64,(size_t)N*K*4);
        d.A8=(uint8_t*)aligned_alloc(64,(size_t)M*K);
        d.B8=(int8_t*)aligned_alloc(64,(size_t)N*K);
        for(size_t i=0;i<(size_t)M*K;++i){ d.A16[i]=(uint16_t)(0x3F80|(i&0x7F)); d.A32[i]=0.5f+(float)((i*2654435761u)%97)/97.f; d.A8[i]=(uint8_t)(i%64); }
        for(size_t i=0;i<(size_t)N*K;++i){ d.B16[i]=(uint16_t)(0x3F80|((i*7)&0x7F)); d.B32[i]=0.25f+(float)((i*40503u)%89)/89.f; d.B8[i]=(int8_t)((i%17)-8); }
        d.Ap.resize((size_t)d.mb*d.kp*32); d.Bp.resize((size_t)d.g*d.kp*48);
        for(int b=0;b<d.mb;++b) for(int p=0;p<d.kp;++p) for(int r=0;r<16;++r){ int m=b*16+r; if(m>=M)continue; memcpy(&d.Ap[((size_t)b*d.kp+p)*32+r*2],&d.A16[(size_t)m*K+2*p],4); }
        for(int b=0;b<d.g;++b) for(int p=0;p<d.kp;++p) for(int c=0;c<24;++c){ int n=b*24+c; if(n>=N)continue; memcpy(&d.Bp[((size_t)b*d.kp+p)*48+c*2],&d.B16[(size_t)n*K+2*p],4); }
        d.ApF.resize((size_t)d.mb*K*16); d.BpF.resize((size_t)(N/16)*K*16);
        for(int b=0;b<d.mb;++b) for(int p=0;p<K;++p) for(int r=0;r<16;++r) memcpy(&d.ApF[((size_t)b*K+p)*16+r],&d.A32[((size_t)(b*16+r))*K+p],4);
        for(int b=0;b<(N/16);++b) for(int p=0;p<K;++p) for(int c=0;c<16;++c) memcpy(&d.BpF[((size_t)b*K+p)*16+c],&d.B32[((size_t)(b*16+c))*K+p],4);
        d.ApI.resize((size_t)d.mb*d.k4*16); d.BpI.resize((size_t)d.g8*d.k4*24);
        for(int b=0;b<d.mb;++b) for(int p=0;p<d.k4;++p) for(int r=0;r<16;++r) memcpy(&d.ApI[((size_t)b*d.k4+p)*16+r],&d.A8[((size_t)(b*16+r))*K+p*4],4);
        for(int b=0;b<d.g8;++b) for(int p=0;p<d.k4;++p) for(int c=0;c<24;++c){ int n=b*24+c; if(n>=N)continue; memcpy(&d.BpI[((size_t)b*d.k4+p)*24+c],&d.B8[(size_t)n*K+p*4],4); }
        d.mb2=(M+7)/8;
        d.Ap2.resize((size_t)d.mb2*K*8); d.Bp2.resize((size_t)(N/8)*K*8);
        for(int b=0;b<d.mb2;++b) for(int p=0;p<K;++p) for(int r=0;r<8;++r)
            memcpy(&d.Ap2[((size_t)b*K+p)*8+r], &d.A32[((size_t)(b*8+r))*K+p], 4);
        for(int b=0;b<(N/8);++b) for(int p=0;p<K;++p) for(int c=0;c<8;++c)
            memcpy(&d.Bp2[((size_t)b*K+p)*8+c], &d.B32[((size_t)(b*8+c))*K+p], 4);
        d.C.assign((size_t)M*N+8192,0.f);
        d.Ci.assign((size_t)M*N+8192,0);
        memory::desc sB({M,K},memory::data_type::bf16,memory::format_tag::ab);
        memory::desc wB({K,N},memory::data_type::bf16,memory::format_tag::ba);
        memory::desc dF({M,N},memory::data_type::f32,memory::format_tag::ab);
        d.primB = matmul(matmul::primitive_desc(d.eng,sB,wB,dF));
        d.msrc = memory(sB,d.eng,d.A16); d.mwei=memory(wB,d.eng,d.B16); d.mdst=memory(dF,d.eng,d.C.data());
        memory::desc sF({M,K},memory::data_type::f32,memory::format_tag::ab);
        memory::desc wF({K,N},memory::data_type::f32,memory::format_tag::ba);
        d.primF = matmul(matmul::primitive_desc(d.eng,sF,wF,dF));
        d.fs=memory(sF,d.eng,d.A32); d.fw=memory(wF,d.eng,d.B32); d.fd=memory(dF,d.eng,d.C.data());
        memory::desc sI({M,K},memory::data_type::u8,memory::format_tag::ab);
        memory::desc wI({K,N},memory::data_type::s8,memory::format_tag::ba);
        memory::desc dI({M,N},memory::data_type::s32,memory::format_tag::ab);
        d.primI = matmul(matmul::primitive_desc(d.eng,sI,wI,dI));
        d.isrc=memory(sI,d.eng,d.A8); d.iwei=memory(wI,d.eng,d.B8); d.idst=memory(dI,d.eng,d.Ci.data());
    }
    // Llaminar engines (columns, low->high instructions)
    std::function<void(ShapeData&)> L[7];
    L[0]=[](ShapeData&d){ fp::run_fp32_skinny_matmul(d.A32,d.B32,d.C.data(),d.M,d.N,d.K,true,1.f,0.f,nullptr,nullptr); }; // FP32 scalar
    L[1]=[](ShapeData&d){
        #pragma omp parallel for schedule(static) collapse(2)
        for(int i=0;i<d.mb2;++i) for(int j=0;j<(d.N/8);++j){
            alignas(64) float st[8*8];
            avx2f32(d.Ap2.data()+(size_t)i*d.K*8, d.Bp2.data()+(size_t)j*d.K*8, st, d.K, 0);
            for(int c=0;c<8;++c) for(int r=0;r<8;++r)
                d.C[(size_t)(i*8+r)*d.N+j*8+c]=st[c*8+r];
        }
    }; // AVX2 FMA32 8x8 (VEX-only)
    L[2]=[](ShapeData&d){ cb::Engine sel; cb::run_bf16_gemm(d.A16,d.B16,d.C.data(),d.M,d.N,d.K,true,1.f,0.f,cb::Engine::AVX512BF16,&sel); }; // vdpbf16ps 4x4
    L[3]=[](ShapeData&d){
        #pragma omp parallel for schedule(static) collapse(2)
        for(int i=0;i<d.mb;++i) for(int j=0;j<d.g;++j)
            m16k(d.Ap.data()+(size_t)i*d.kp*32, d.Bp.data()+(size_t)j*d.kp*48, d.C.data()+(size_t)i*16*d.N+(size_t)j*24, d.kp, (long)d.N*4);
    }; // vdpbf16ps packed 16x24
    L[4]=[](ShapeData&d){
        #pragma omp parallel for schedule(static) collapse(2)
        for(int i=0;i<d.mb;++i) for(int j=0;j<(d.N/16);++j)
            f32tile(d.ApF.data()+(size_t)i*d.K*16, d.BpF.data()+(size_t)j*d.K*16, d.C.data()+(size_t)i*16*d.N+(size_t)j*16, d.K, (long)d.N*4);
    }; // vfmadd231ps 512
    L[5]=[](ShapeData&d){
        #pragma omp parallel for schedule(static) collapse(2)
        for(int i=0;i<d.mb;++i) for(int j=0;j<d.g8;++j)
            vntile8(d.ApI.data()+(size_t)i*d.k4*16, d.BpI.data()+(size_t)j*d.k4*24, d.Ci.data()+(size_t)i*16*d.N+(size_t)j*24, d.k4, (long)d.N*4);
    }; // vpdpbusd int8
    L[6]=[](ShapeData&d){ cb::Engine sel; cb::run_bf16_gemm(d.A16,d.B16,d.C.data(),d.M,d.N,d.K,true,1.f,0.f,cb::Engine::AMXBF16,&sel); }; // tdpbf16ps
    // oneDNN tiers (rows, low->high)
    std::function<void(ShapeData&)> O[3];
    O[0]=[](ShapeData&d){ d.primF.execute(d.str,{{DNNL_ARG_SRC,d.fs},{DNNL_ARG_WEIGHTS,d.fw},{DNNL_ARG_DST,d.fd}}); d.str.wait(); };
    O[1]=[](ShapeData&d){ d.primI.execute(d.str,{{DNNL_ARG_SRC,d.isrc},{DNNL_ARG_WEIGHTS,d.iwei},{DNNL_ARG_DST,d.idst}}); d.str.wait(); };
    O[2]=[](ShapeData&d){ d.primB.execute(d.str,{{DNNL_ARG_SRC,d.msrc},{DNNL_ARG_WEIGHTS,d.mwei},{DNNL_ARG_DST,d.mdst}}); d.str.wait(); };
    // warmup
    for(int s=0;s<4;++s) for(int l=0;l<7;++l){ if(l==6){ try{L[l](D[s]);}catch(...){} continue;} L[l](D[s]); }
    for(int s=0;s<4;++s) for(int o=0;o<3;++o) O[o](D[s]);
    double tl[7][4], to[3][4];
    for(int l=0;l<7;++l) for(int s=0;s<4;++s) tl[l][s]=1e9;
    for(int o=0;o<3;++o) for(int s=0;s<4;++s) to[o][s]=1e9;
    for(int l=0;l<7;++l){
        for(int s=0;s<4;++s){
            for(int r=0;r<20;++r){ auto t=Clock::now(); L[l](D[s]); double e=ms(t,Clock::now()); if(e<tl[l][s])tl[l][s]=e; }
        }
    }
    for(int o=0;o<3;++o){
        for(int s=0;s<4;++s){
            for(int r=0;r<20;++r){ auto t=Clock::now(); O[o](D[s]); double e=ms(t,Clock::now()); if(e<to[o][s])to[o][s]=e; }
        }
    }
    const bool forced = getenv("DNNL_MAX_CPU_ISA") && strstr(getenv("DNNL_MAX_CPU_ISA"),"avx512_core_bf16");
    const char* Oisa[3]={ "avx512_core", forced?"avx512_vnni":"AMX int8", forced?"avx512_core_bf16":"AMX (default)"};
    // canonical row ids: 0 fp32-avx2, 1 int8-avx2, 2 bf16-avx2, 3 fp32-avx512, 4 int8-avx512, 5 bf16-avx512, 6 int8-amx, 7 bf16-amx
    bool isAvx2 = forced==false && getenv("DNNL_MAX_CPU_ISA") && strstr(getenv("DNNL_MAX_CPU_ISA"),"avx2");
    int rowId[3];
    if (isAvx2) rowId[0]=0, rowId[1]=1, rowId[2]=2;
    else if (forced) rowId[0]=3, rowId[1]=4, rowId[2]=5;
    else rowId[0]=3, rowId[1]=6, rowId[2]=7;
    printf("this run\'s rows saved; run the second ISA mode too, then ./tier2d --show for the single merged matrix.\n");
    {
        FILE* f=fopen(kRowFile,"a");
        if(f){
            for(int o=0;o<3;++o){
                fprintf(f,"row|%d|", rowId[o]);
                for(int l=0;l<7;++l){
                    double vmin=99,vmax=0; bool any=false;
                    for(int s=0;s<4;++s){ double r=to[o][s]/tl[l][s]; if(r<vmin)vmin=r; if(r>vmax)vmax=r; any=true; }
                    if(any) fprintf(f,"%.4f,%.4f|", vmin, vmax);
                    else fprintf(f,"-1,-1|");
                }
                fprintf(f,"\n");
            }
            fclose(f);
        }
    }
    printf("single-run rows (raw):\n");
    printf("%-22s | %22s | %22s | %22s | %22s | %22s | %22s\n","","FP32 scalar","AVX512-BF16 4x4","AVX512-BF16 16x24 pk","FMA32 512","VNNI int8","AMX-BF16");
    for(int o=0;o<3;++o){
        char buf[64]; snprintf(buf,sizeof buf,"%s (%s)",Onames[o], forced?Oisa[o]:Oisa[o]);
        printf("%-22s |", buf);
        for(int l=0;l<7;++l){
            double vmin=99,vmax=0; bool any=false;
            for(int s=0;s<4;++s){ double r=to[o][s]/tl[l][s]; if(r<vmin)vmin=r; if(r>vmax)vmax=r; any=true; }
            if(!any){ printf(" %22s","n/a"); continue; }
            char cell[32];
            if (vmax-vmin<0.05) snprintf(cell,sizeof cell,"%.2fx",(vmin+vmax)/2);
            else snprintf(cell,sizeof cell,"%.2f-%.2fx",vmin,vmax);
            printf(" %22s |", cell);
        }
        printf("\n");
    }
    printf("\naligned cells (same instruction type): FP32 col x oneDNN FP32 row; AVX512-BF16 cols x oneDNN BF16(avx512) row; AMX-BF16 col x oneDNN BF16(AMX) row; VNNI int8 col x oneDNN int8 rows.\n");
    return 0;
}
