/** @file VectorIssue.hip
 * @brief Measure independent native wave instructions at the captured MI50 thread geometry.
 * Every lane publishes its checksum. Timings exclude initialization, readback and profiling.
 */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define HIP(call) do { const auto error = (call); if (error != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n",#call,hipGetErrorString(error)); std::exit(2); } } while(false)
enum class Operation { Add, Fma, Dot4 };
constexpr unsigned repetitions=128, accumulators=8, unroll=8;
template<Operation Op,unsigned Lanes>
__global__ __launch_bounds__(Lanes) void issue(unsigned* output) {
    const unsigned id=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned a=(id & 7)+1, b=((id>>3)&7)+1;
    using Number=std::conditional_t<Op==Operation::Fma,float,unsigned>;
    Number accum[accumulators];
#pragma unroll
    for(unsigned k=0;k<accumulators;++k) accum[k]=Number(k+a);
    for(unsigned rep=0;rep<repetitions;++rep) {
#pragma unroll
        for (unsigned u=0;u<unroll;++u) {
#pragma unroll
            for(unsigned k=0;k<accumulators;++k) {
                if constexpr (Op==Operation::Fma) {
                    asm volatile("v_fma_f32 %0, %1, %2, %0" : "+v"(accum[k]) : "v"(float(a)), "v"(float(b)));
                } else if constexpr (Op==Operation::Dot4) {
                    asm volatile("v_dot4_i32_i8 %0, %1, %2, %0" : "+v"(accum[k]) : "v"(a*0x01010101u), "v"(b*0x01010101u));
                } else {
                    asm volatile("v_add_u32 %0, %1, %0" : "+v"(accum[k]) : "v"(a));
                }
            }
        }
    }
    unsigned sum=0;
#pragma unroll
    for(unsigned k=0;k<accumulators;++k) sum+=unsigned(accum[k]);
    output[id]=sum;
}
template<Operation Op,unsigned Lanes>
void measure(unsigned round,unsigned blocks,unsigned wave,hipStream_t stream){
    const char* mode=Op==Operation::Add?"iadd":Op==Operation::Fma?"fma":"dot4";
    std::vector<unsigned> actual(size_t(blocks)*Lanes);
    unsigned* out; HIP(hipMalloc(&out, sizeof(unsigned)*actual.size()));
    hipEvent_t begin, end; HIP(hipEventCreate(&begin)); HIP(hipEventCreate(&end));
    auto launch=[&] { hipLaunchKernelGGL((issue<Op,Lanes>),dim3(blocks),dim3(Lanes),0,stream,out); HIP(hipGetLastError()); };
    for(unsigned i=0;i<4;++i) launch();
    const auto wave_ops=double(blocks)*Lanes/wave*repetitions*unroll*accumulators;
    for(unsigned sample=0;sample<13;++sample) {
        HIP(hipEventRecord(begin,stream));
        for(unsigned k=0;k<8;++k) launch();
        HIP(hipEventRecord(end,stream)); HIP(hipEventSynchronize(end));
        float ms; HIP(hipEventElapsedTime(&ms,begin,end)); const double us=ms*1000./8;
        std::printf("ISSUE,%u,%s,%u,%u,%u,%.6f,%.8f\n",round,mode,blocks,repetitions,sample,us,wave_ops/(1000.*us));
    }
    HIP(hipMemcpyAsync(actual.data(),out,actual.size()*sizeof(unsigned),hipMemcpyDeviceToHost,stream));
    HIP(hipEventRecord(end,stream)); HIP(hipEventSynchronize(end));
    for(unsigned id=0;id<actual.size();++id) {
        const unsigned a=(id&7)+1,b=((id>>3)&7)+1;
        const unsigned add=Op==Operation::Add?a:Op==Operation::Fma?a*b:4*a*b;
        const unsigned expected=accumulators*(a+add*unroll*repetitions)+accumulators*(accumulators-1)/2;
        if(expected!=actual[id]) {std::fprintf(stderr,"Mismatch %u %u %u\n", id,expected,actual[id]);std::exit(3);}
    }
    std::printf("CHECK,%u,%s,%u,%u\n",round,mode,blocks,repetitions);
    HIP(hipEventDestroy(end)); HIP(hipEventDestroy(begin)); HIP(hipFree(out));
}
/** @brief Run the three independent vector issue references at one launch geometry. */
template<unsigned Lanes>
void measureGeometry(unsigned round,unsigned blocks,unsigned wave,hipStream_t stream)
{
    measure<Operation::Add,Lanes>(round,blocks,wave,stream);
    measure<Operation::Fma,Lanes>(round,blocks,wave,stream);
    measure<Operation::Dot4,Lanes>(round,blocks,wave,stream);
}

int main(int argc,char** argv){
    if(argc!=3 && argc!=5)return 2;
    unsigned round=std::strtoul(argv[1],nullptr,10);int count,device=-1;HIP(hipGetDeviceCount(&count));
    if(count!=1)return 2;
    for(int i=0;i<count;++i){char pci[32]{};HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),i));if(!std::strcmp(pci,argv[2]))device=i;}
    if(device<0)return 2;
    HIP(hipSetDevice(device));hipDeviceProp_t props{}; HIP(hipGetDeviceProperties(&props,device));
    if(std::strncmp(props.gcnArchName,"gfx906",6)||props.warpSize!=64)return 2;
    std::printf("DEVICE,%s,%s,%d,%d\n",argv[2],props.gcnArchName,props.multiProcessorCount,props.warpSize);
    hipStream_t stream;HIP(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    if(argc==5) {
        char* end=nullptr;
        const auto blocks=std::strtoul(argv[3],&end,10);
        if(!blocks || *end || blocks>4u*static_cast<unsigned>(props.multiProcessorCount))return 2;
        if(std::strcmp(argv[4],"256")==0)
            measureGeometry<256>(round,static_cast<unsigned>(blocks),props.warpSize,stream);
        else if(std::strcmp(argv[4],"1024")==0)
            measureGeometry<1024>(round,static_cast<unsigned>(blocks),props.warpSize,stream);
        else return 2;
    } else {
        for(unsigned j=0;j<3;++j) {
            const auto blocks=props.multiProcessorCount*(1u<<((round+j)%3));
            measureGeometry<1024>(round,blocks,props.warpSize,stream);
        }
    }
    HIP(hipStreamDestroy(stream));
}
