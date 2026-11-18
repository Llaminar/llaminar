/** @file QfnRoutedPacketRead.hip
 * @brief Packet/stride calibration for native F160 M4 gate/up and down reads.
 * The real router receipt supplies forty expert indices per source. This benchmark
 * uses the same native payload/half-metadata spacings on generated checksum data.
 */
#include <hip/hip_runtime.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define HIP(call) do { auto err=(call); if(err!=hipSuccess){     std::fprintf(stderr,"%s: %s\n",#call,hipGetErrorString(err)); std::exit(1); } } while(false)
constexpr int Experts=512, Live=4, Top=10, H=2560, F=160, Banks=4;
using Word=unsigned int;
using Half=unsigned short;
struct Payload {
    const Word* words;
    const Half* scales;
    const Half* minimum;
};

/** @brief Read payload words and the exact independent coefficient planes for one K32 block. */
template<int Words,bool Affine>
__device__ __forceinline__ Word blockRead(Payload data,std::size_t ix)
{
    Word sum=data.scales[ix];
    if constexpr(Affine) sum+=data.minimum[ix];
#pragma unroll
    for(int w=0;w<Words;++w)sum+=data.words[ix*Words+w];
    return sum;
}

/** @brief Follow the native grouped geometry while omitting numerical dot/unpack work. */
template<int Words,bool Affine,bool Gate>
__global__ void packetRead(Payload data,const int* routes,int bank,Word* result)
{
    constexpr int Width=Gate?F:H;
    constexpr int Blocks=Gate?H/32:F/32;
    constexpr int Roles=Gate?2:1;
    const auto base=std::size_t(bank)*Experts*Roles*Width*Blocks;
    const int t=threadIdx.x;
    const int lane=t%64;
    Word folded=0;
    if constexpr(Gate) {
        const int tile=blockIdx.x*2+t/128;
        if(tile<Live*Top*(F/32)) {
            const int expert=routes[tile/(F/32)];
            const int column=tile%(F/32)*32+t%128/64*16+lane%16;
            for(int originalPart=lane/16;originalPart<16;originalPart+=4)
                for(int k=originalPart*5;k<(originalPart+1)*5;++k) {
                    const auto ix=base+(std::size_t(expert)*Roles*Blocks+k)*Width+column;
                    folded+=blockRead<Words,Affine>(data,ix);
                    folded+=blockRead<Words,Affine>(data,ix+Blocks*Width);
                }
        }
    } else {
        const int column=blockIdx.x*64+lane;
        for(int slot=t/64;slot<Top;slot+=4) {
            const int expert=routes[blockIdx.y*Top+slot];
            for(int k=0;k<Blocks;++k) {
                const auto ix=base+(std::size_t(expert)*Blocks+k)*Width+column;
                folded+=blockRead<Words,Affine>(data,ix);
            }
        }
    }
    result[(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+t]=folded;
}

using RouteRows=std::array<int,Live*Top>;
using CaseRoutes=std::array<RouteRows,3>;

/** @brief Load independent authenticated routed histories supplied with the benchmark. */
bool readCaseRoutes(const char* path,CaseRoutes& routes)
{
    std::ifstream input(path);
    for(auto& family:routes) {
        for(int row=0;row<Live;++row) {
            std::array<bool,Experts> seen{};
            for(int column=0;column<Top;++column) {
                auto& id=family[row*Top+column];
                if(!(input>>id) || id<0 || id>=Experts || seen[id])return false;
                seen[id]=true;
            }
        }
    }
    input >> std::ws;
    return input.eof() && !input.bad();
}

/**
 * @brief Measure fresh or revolving physical packet banks and verify every lane.
 * @param family Stable source-tuple position in the supplied original route file.
 * @param round Independent experiment round number.
 * @param stream The explicitly owned HIP stream.
 * @param routes The complete four-row, original-slot route history.
 */
template<int Words,bool Affine,bool Gate>
void run(int family,int round,hipStream_t stream,const RouteRows& routes)
{
    constexpr std::size_t BlocksPerExpert=H*F/32*(Gate?2:1);
    constexpr std::size_t blockCount=Banks*Experts*BlocksPerExpert;
    constexpr int BytesPerBlock=Words*4+2+(Affine?2:0);
    constexpr int gridX=Gate?120:H/64, gridY=Gate?1:Live;
    constexpr std::size_t outputCount=gridX*gridY*256;
    constexpr double requestBytes=Live*Top*BlocksPerExpert*BytesPerBlock;
    Word *words,*out;
    Half *scales,*mins=nullptr;
    int *ids;
    HIP(hipMalloc(&words,blockCount*Words*sizeof(Word)));
    HIP(hipMalloc(&scales,blockCount*sizeof(Half)));
    if constexpr(Affine) HIP(hipMalloc(&mins,blockCount*sizeof(Half)));
    HIP(hipMalloc(&out,outputCount*sizeof(Word)));
    HIP(hipMalloc(&ids,Live*Top*sizeof(int)));
    constexpr auto bankBlocks=blockCount/Banks;
    for(int bank=0;bank<Banks;++bank) {
        HIP(hipMemsetAsync(words+bank*bankBlocks*Words,bank+1,bankBlocks*Words*sizeof(Word),stream));
        HIP(hipMemsetAsync(scales+bank*bankBlocks,bank+1,bankBlocks*sizeof(Half),stream));
        if constexpr(Affine) HIP(hipMemsetAsync(mins+bank*bankBlocks,bank+1,bankBlocks*sizeof(Half),stream));
    }
    HIP(hipMemcpyAsync(ids,routes.data(),Live*Top*sizeof(int),hipMemcpyHostToDevice,stream));
    hipEvent_t begin,end;
    HIP(hipEventCreate(&begin)); HIP(hipEventCreate(&end));
    auto launch=[&](int bank) {
        hipLaunchKernelGGL((packetRead<Words,Affine,Gate>),dim3(gridX,gridY),dim3(256),0,stream,
                           Payload{words,scales,mins},ids,bank,out);
        HIP(hipGetLastError());
    };
    for(int bankMode: {1,Banks}) {
        for(int i=0;i<8;++i)launch(i%bankMode);
        for(int s=0;s<13;++s) {
            HIP(hipEventRecord(begin,stream));
            for(int i=0;i<16;++i)launch(i%bankMode);
            HIP(hipEventRecord(end,stream)); HIP(hipEventSynchronize(end));
            float ms=0;
            HIP(hipEventElapsedTime(&ms,begin,end));
            const double us=ms*1000/16;
            std::printf("PACKET,%d,%d,%s,%d,%d,%.6f,%.4f\n",round,family,Gate?"gate":"down",bankMode,s,us,requestBytes/(us*1000));
        }
        std::vector<Word> output(outputCount);
        HIP(hipMemcpyAsync(output.data(),out,outputCount*sizeof(Word),hipMemcpyDeviceToHost,stream));
        HIP(hipEventRecord(end,stream)); HIP(hipEventSynchronize(end));
        constexpr Word unit=Words*0x01010101u+(1+int(Affine))*0x0101u;
        for(std::size_t i=0;i<outputCount;++i) {
            const int t=i%256;
            const int cta=i/256;
            const unsigned n=Gate?(cta*2+t/128<Live*Top*5?40:0):((t/64<2?3:2)*5);
            if(output[i]!=n*unit*bankMode)std::exit(3);
        }
        std::printf("CHECK,%d,%d,%s,%d\n",round,family,Gate?"gate":"down",bankMode);
    }
    HIP(hipEventDestroy(end)); HIP(hipEventDestroy(begin));
    HIP(hipFree(ids)); HIP(hipFree(out)); HIP(hipFree(words)); HIP(hipFree(scales));
    if constexpr(Affine) HIP(hipFree(mins));
}

int main(int argc,char** argv)
{
    CaseRoutes routes;
    if(argc!=4 || !readCaseRoutes(argv[3],routes)) {
        std::fprintf(stderr,"Usage: %s ROUND PCI ROUTES (three groups of forty original slot IDs)\n",argv[0]);
        return 2;
    }
    int count=0,chosen=-1;
    HIP(hipGetDeviceCount(&count));
    if(count!=1)return 2;
    for(int i=0;i<count;++i) {
        char pci[32]{};
        HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),i));
        if(!std::strcmp(pci,argv[2]))chosen=i;
    }
    if(chosen<0)return 2;
    HIP(hipSetDevice(chosen));
    hipDeviceProp_t props{};
    HIP(hipGetDeviceProperties(&props,chosen));
    if(std::strncmp(props.gcnArchName,"gfx906",6)||props.warpSize!=64)return 2;
    std::printf("DEVICE,%s,%s,%d,%d\n",argv[2],props.gcnArchName,props.multiProcessorCount,props.l2CacheSize);
    hipStream_t stream;
    HIP(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    const int round=std::atoi(argv[1]);
    run<4,true,true>(0,round,stream,routes[0]); run<5,true,false>(0,round,stream,routes[0]);
    run<4,true,true>(1,round,stream,routes[1]); run<8,false,false>(1,round,stream,routes[1]);
    run<5,true,true>(2,round,stream,routes[2]); run<8,false,false>(2,round,stream,routes[2]);
    HIP(hipStreamDestroy(stream));
}
