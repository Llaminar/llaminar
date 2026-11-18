/** @file QfnRoutedPacketRead.hip
 * @brief Packet traffic reference for the QFN M4 parent direct native readers.
 *
 * Preserve original route/role order and the prepared [K32][column] layout.
 * Use generated measurement data, with a different value at each plane
 * address, and omit the arithmetic and non-weight reads of the FFN.
 */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string_view>
#include <vector>

#define HIP(call) do { auto err=(call); if(err!=hipSuccess){ \
    std::fprintf(stderr,"%s: %s\n",#call,hipGetErrorString(err)); std::exit(1); } } while(false)

constexpr int Experts=512, Live=4, Top=10, H=2560, Banks=4, Threads=256;
using Word=unsigned int;
using Half=unsigned short;
using Byte=unsigned char;
using RouteRows=std::array<int,Live*Top>;

/** @brief Immutable plane addresses in two role datasets, with independent coefficient streams. */
struct Payload {
    Byte* bytes{};
    Half* scales{};
    Half* secondary{};
};

struct Geometry {
    int routed=160;
    int shared=160;
    int routedParts=16;
    int sharedParts=16;
};

/** @brief Non-periodic, independent element values for the host/device checksum oracle. */
__host__ __device__ Word pattern(std::size_t index, Word plane, Word dataset)
{
    Word value=Word(index) ^ (0x9e3779b9u * (1+plane+dataset*7));
    value=(value ^ (value>>16))*0x85ebca6bu;
    value=(value ^ (value>>13))*0xc2b2ae35u;
    return value ^ (value>>16);
}

/** @brief Fill payload and coefficient allocations before timing, without staging complete banks on host. */
template<int Bytes,bool Secondary>
__global__ void fillPackets(Payload planes,std::size_t packets,Word dataset)
{
    for(std::size_t index=blockIdx.x*blockDim.x+threadIdx.x;index<packets;index+=gridDim.x*blockDim.x) {
        for(int j=0;j<Bytes;++j)planes.bytes[index*Bytes+j]=Byte(pattern(index*Bytes+j,0,dataset));
        planes.scales[index]=Half(pattern(index,1,dataset));
        if constexpr(Secondary) planes.secondary[index]=Half(pattern(index,2,dataset));
    }
}

/** @brief Read one native K32 block, with its scale and optional secondary coefficient. */
template<int Bytes,bool Secondary>
__device__ __forceinline__ Word readPacket(Payload data,std::size_t index)
{
    Word sum=data.scales[index];
    if constexpr(Secondary) sum+=data.secondary[index];
    const auto* p=data.bytes+index*Bytes;
#pragma unroll
    for(int j=0;j<Bytes/4;++j) {
        Word word;
        __builtin_memcpy(&word,p+j*4,4);
        sum+=word;
    }
    if constexpr(Bytes%4) {
        Word tail=0;
#pragma unroll
        for(int j=0;j<Bytes%4;++j)tail|=Word(p[Bytes/4*4+j])<<(8*j);
        sum+=tail;
    }
    return sum;
}

/** @brief Traverse the source partition stripes of the actual 128-thread gate tile. */
template<int Bytes,bool Secondary>
__device__ __forceinline__ Word gateTile(Payload data,std::size_t base,int width,int column,int parts,int worker)
{
    const auto roleExtent=std::size_t(H/32)*width;
    const int stride=(H/32+parts-1)/parts;
    Word sum=0;
    for(int role=0;role<2;++role)
        for(int part=worker;part<parts;part+=4)
            for(int k=part*stride;k<min((part+1)*stride,H/32);++k)
                sum+=readPacket<Bytes,Secondary>(data,base+role*roleExtent+k*width+column);
    return sum;
}

/** @brief Read exactly the gate/down weight packets owned by each active production lane. */
template<int Bytes,bool Secondary,bool Gate,int SharedBytes=32,bool SharedSecondary=false,bool SingleTile=false>
__global__ void packetRead(Payload routed,Payload shared,const int* routes,Geometry shape,int bank,Word* result)
{
    const int tid=threadIdx.x;
    const int lane=tid%64;
    Word sum=0;
    if constexpr(Gate) {
        const int routedBlocks=shape.routed/32;
        const int sharedBlocks=shape.shared/32;
        const int routedTiles=Live*Top*routedBlocks;
        const int tileCount=routedTiles+Live*sharedBlocks;
        for(int tile=blockIdx.x*2+tid/128;tile<tileCount;tile+=gridDim.x*2) {
            const int offset=tid%128/64*16+lane%16;
            if(tile<routedTiles) {
                const int expert=routes[tile/routedBlocks];
                const int column=tile%routedBlocks*32+offset;
                const auto base=std::size_t(bank*Experts+expert)*2*(H/32)*shape.routed;
                sum+=gateTile<Bytes,Secondary>(routed,base,shape.routed,column,shape.routedParts,lane/16);
            } else {
                const int column=(tile-routedTiles)%sharedBlocks*32+offset;
                sum+=gateTile<SharedBytes,SharedSecondary>(shared,0,shape.shared,column,shape.sharedParts,lane/16);
            }
            if constexpr(SingleTile)break;
        }
    } else {
        const int column=blockIdx.x*64+lane;
        for(int slot=tid/64;slot<=Top;slot+=4) {
            if(slot==Top) {
                for(int k=0;k<shape.shared/32;++k)
                    sum+=readPacket<SharedBytes,SharedSecondary>(shared,k*H+column);
            } else {
                const int expert=routes[blockIdx.y*Top+slot];
                const auto blocks=shape.routed/32;
                const auto base=std::size_t(bank*Experts+expert)*blocks*H;
                for(int k=0;k<blocks;++k)
                    sum+=readPacket<Bytes,Secondary>(routed,base+k*H+column);
            }
        }
    }
    result[(blockIdx.y*gridDim.x+blockIdx.x)*Threads+tid]=sum;
}

/** @brief Parse ordered histories and reject duplicate or invalid per-row IDs. */
bool readRoutes(const char* path,std::vector<RouteRows>& histories)
{
    std::ifstream input(path);
    for(auto& routes:histories)
        for(int row=0;row<Live;++row) {
            std::array<bool,Experts> seen{};
            for(int slot=0;slot<Top;++slot) {
                auto& id=routes[row*Top+slot];
                if(!(input>>id) || id<0 || id>=Experts || seen[id])return false;
                seen[id]=true;
            }
        }
    input>>std::ws;
    return input.eof() && !input.bad();
}

/** @brief Independent byte composition of a packet at an original logical matrix coordinate. */
Word expectedPacket(int bytes,bool secondary,std::size_t index,Word dataset)
{
    Word expected=Half(pattern(index,1,dataset));
    if(secondary)expected+=Half(pattern(index,2,dataset));
    for(int j=0;j<bytes;++j)
        expected+=Word(Byte(pattern(index*bytes+j,0,dataset)))<<(8*(j%4));
    return expected;
}

/** @brief Check every output lane using logical tile and matrix coordinates on the host. */
template<int Bytes,bool Secondary,bool Gate,int SharedBytes=32,bool SharedSecondary=false>
std::vector<Word> expectedLanes(Geometry shape,const RouteRows& routes,int gridX,int bank)
{
    std::vector<Word> expected(gridX*(Gate?1:Live)*Threads);
    if constexpr(Gate) {
        const int routeTiles=Live*Top*(shape.routed/32);
        const int totalTiles=routeTiles+Live*(shape.shared/32);
        for(int tile=0;tile<totalTiles;++tile) {
            const bool shared=tile>=routeTiles;
            const int n=shared?shape.shared:shape.routed;
            const int local=shared?tile-routeTiles:tile;
            const int first=local%(n/32)*32;
            const int stride=(80+(shared?shape.sharedParts:shape.routedParts)-1)/
                (shared?shape.sharedParts:shape.routedParts);
            const int expert=shared?0:bank*Experts+routes[local/(n/32)];
            const int owner=(tile/2%gridX)*Threads+(tile%2)*128;
            for(int k=0;k<80;++k)
                for(int colOffset=0;colOffset<32;++colOffset) {
                    const int col=first+colOffset;
                    const auto index=std::size_t(expert)*2*80*n+k*n+col;
                    const int lane=owner+colOffset/16*64+(k/stride%4)*16+colOffset%16;
                    for(int role=0;role<2;++role)
                        expected[lane]+=expectedPacket(shared?SharedBytes:Bytes,shared?SharedSecondary:Secondary,index+role*80*n,shared?1:0);
                }
        }
    } else {
        for(int row=0;row<Live;++row)
            for(int slot=0;slot<=Top;++slot) {
                const bool shared=slot==Top;
                const int blocks=(shared?shape.shared:shape.routed)/32;
                const int expert=shared?0:bank*Experts+routes[row*Top+slot];
                for(int column=0;column<H;++column) {
                    const int lane=(row*gridX+column/64)*Threads+(slot%4)*64+column%64;
                    for(int k=0;k<blocks;++k)
                        expected[lane]+=expectedPacket(shared?SharedBytes:Bytes,shared?SharedSecondary:Secondary,
                            (std::size_t(expert)*blocks+k)*H+column,shared?1:0);
                }
            }
    }
    return expected;
}

/** @brief Measure one exact original geometry and physically verify all lanes after each bank mode. */
template<int Bytes,bool Secondary,bool Gate,int SharedBytes=32,bool SharedSecondary=false,bool SingleTile=false>
void run(int family,int round,hipStream_t stream,const RouteRows& routes,Geometry shape)
{
    const std::size_t routedCount=std::size_t(H)*shape.routed/32*(Gate?2:1);
    const std::size_t sharedCount=std::size_t(H)*shape.shared/32*(Gate?2:1);
    const int gateTiles=Live*(Top*(shape.routed/32)+shape.shared/32);
    const int gridX=Gate?(SingleTile?(gateTiles+1)/2:std::clamp((gateTiles+1)/2,Live,120)):H/64;
    const int gridY=Gate?1:Live;
    const std::size_t outputCount=gridX*gridY*Threads;
    const double requestBytes=Live*(Top*routedCount*(Bytes+2+2*int(Secondary))+sharedCount*(SharedBytes+2+2*int(SharedSecondary)));
    Payload routed{},shared{};
    const auto packets=Banks*Experts*routedCount;
    HIP(hipMalloc(&routed.bytes,packets*Bytes));
    HIP(hipMalloc(&routed.scales,packets*sizeof(Half)));
    if constexpr(Secondary)HIP(hipMalloc(&routed.secondary,packets*sizeof(Half)));
    HIP(hipMalloc(&shared.bytes,sharedCount*SharedBytes));
    HIP(hipMalloc(&shared.scales,sharedCount*sizeof(Half)));
    if constexpr(SharedSecondary)HIP(hipMalloc(&shared.secondary,sharedCount*sizeof(Half)));
    hipLaunchKernelGGL((fillPackets<Bytes,Secondary>),dim3(240),dim3(Threads),0,stream,routed,packets,0);
    hipLaunchKernelGGL((fillPackets<SharedBytes,SharedSecondary>),dim3(240),dim3(Threads),0,stream,shared,sharedCount,1);
    HIP(hipGetLastError());
    Word* out;
    int* ids;
    HIP(hipMalloc(&out,outputCount*sizeof(Word)));
    HIP(hipMalloc(&ids,routes.size()*sizeof(int)));
    HIP(hipMemcpyAsync(ids,routes.data(),routes.size()*sizeof(int),hipMemcpyHostToDevice,stream));
    hipEvent_t begin,end;
    HIP(hipEventCreate(&begin));
    HIP(hipEventCreate(&end));
    auto launch=[&](int bank) {
        hipLaunchKernelGGL((packetRead<Bytes,Secondary,Gate,SharedBytes,SharedSecondary,SingleTile>),dim3(gridX,gridY),dim3(Threads),0,stream,
                           routed,shared,ids,shape,bank,out);
        HIP(hipGetLastError());
    };
    auto checkBank=[&](int bank) {
        std::vector<Word> output(outputCount);
        HIP(hipMemcpyAsync(output.data(),out,outputCount*sizeof(Word),hipMemcpyDeviceToHost,stream));
        HIP(hipEventRecord(end,stream));
        HIP(hipEventSynchronize(end));
        const auto expected=expectedLanes<Bytes,Secondary,Gate,SharedBytes,SharedSecondary>(shape,routes,gridX,bank);
        if(output!=expected) {
            const auto lane=std::mismatch(output.begin(),output.end(),expected.begin()).first-output.begin();
            std::fprintf(stderr,"Packet checksum mismatch: family=%d role=%s bank=%d lane=%td\n",
                family,Gate?"gate":"down",bank,lane);
            std::exit(3);
        }
    };
    std::printf("LAYOUT,%d,%d,%s,%d,%d,%d,%d,%d,%d,%.0f\n",round,family,Gate?"gate":"down",
        shape.routed,shape.shared,shape.routedParts,shape.sharedParts,gridX,gridY,requestBytes);
    for(int bankMode:{1,Banks}) {
        for(int i=0;i<8;++i)launch(i%bankMode);
        for(int sample=0;sample<13;++sample) {
            HIP(hipEventRecord(begin,stream));
            for(int i=0;i<16;++i)launch(i%bankMode);
            HIP(hipEventRecord(end,stream));
            HIP(hipEventSynchronize(end));
            float ms=0;
            HIP(hipEventElapsedTime(&ms,begin,end));
            const double us=ms*1000/16;
            std::printf("PACKET,%d,%d,%s,%d,%d,%.6f,%.4f\n",round,family,Gate?"gate":"down",
                bankMode,sample,us,requestBytes/(us*1000));
        }
        checkBank(bankMode-1);
        std::printf("CHECK,%d,%d,%s,%d\n",round,family,Gate?"gate":"down",bankMode);
    }
    for(int bank=1;bank<Banks-1;++bank) {
        launch(bank);
        checkBank(bank);
    }
    HIP(hipEventDestroy(end));
    HIP(hipEventDestroy(begin));
    HIP(hipFree(ids));
    HIP(hipFree(out));
    HIP(hipFree(routed.bytes));
    HIP(hipFree(routed.scales));
    if constexpr(Secondary) HIP(hipFree(routed.secondary));
    HIP(hipFree(shared.bytes));
    HIP(hipFree(shared.scales));
    if constexpr(SharedSecondary)HIP(hipFree(shared.secondary));
}

/** @brief Run the gate and down scopes for an independently prepared source tuple. */
void runTuple(int family,int round,hipStream_t stream,const RouteRows& routes,Geometry shape)
{
    if(family==3) {
        run<12,false,true,9,true>(family,round,stream,routes,shape);
        run<24,true,false>(family,round,stream,routes,shape);
        return;
    }
    if(family==2)run<20,true,true>(family,round,stream,routes,shape);
    else run<16,true,true>(family,round,stream,routes,shape);
    if(family==0)run<20,true,false>(family,round,stream,routes,shape);
    else run<32,false,false>(family,round,stream,routes,shape);
}

/** @brief Admit a complete decimal operand with a closed numerical range. */
bool parse(std::string_view value,int& output,int min,int max)
{
    const auto [end,error]=std::from_chars(value.data(),value.data()+value.size(),output);
    return error==std::errc{} && end==value.data()+value.size() && output>=min && output<=max;
}

int main(int argc,char** argv)
{
    Geometry shape;
    int family=-1,round=0;
    const bool singleTile=argc==10 && std::string_view(argv[9])=="single-tile";
    if((argc!=4 && argc!=9 && !singleTile) || !parse(argv[1],round,0,100) ||
       (argc>=9 && (!parse(argv[4],family,0,3) || !parse(argv[5],shape.routed,32,640) ||
           !parse(argv[6],shape.shared,32,640) || !parse(argv[7],shape.routedParts,1,80) ||
           !parse(argv[8],shape.sharedParts,1,80) || shape.routed%32 || shape.shared%32))) {
        std::fprintf(stderr,"Usage: %s ROUND PCI ROUTES [TUPLE F S ROUTED_GATE_PARTS SHARED_GATE_PARTS [single-tile]]\n",argv[0]);
        return 2;
    }
    if(singleTile && family<0)return 2;
    std::vector<RouteRows> histories(family<0?3:1);
    if(!readRoutes(argv[3],histories))return 2;
    int count=0;
    HIP(hipGetDeviceCount(&count));
    if(count!=1)return 2;
    char pci[32]{};
    HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    if(std::strcmp(pci,argv[2]))return 2;
    HIP(hipSetDevice(0));
    hipDeviceProp_t props{};
    HIP(hipGetDeviceProperties(&props,0));
    if(std::strncmp(props.gcnArchName,"gfx906",6)||props.warpSize!=64)return 2;
    std::printf("DEVICE,%s,%s,%d,%d\n",argv[2],props.gcnArchName,props.multiProcessorCount,props.l2CacheSize);
    hipStream_t stream;
    HIP(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    if(singleTile && family==3)
        run<12,false,true,9,true,true>(family,round,stream,histories.front(),shape);
    else if(singleTile && family==2)
        run<20,true,true,32,false,true>(family,round,stream,histories.front(),shape);
    else if(singleTile)
        run<16,true,true,32,false,true>(family,round,stream,histories.front(),shape);
    else if(family<0)
        for(int i=0;i<3;++i)runTuple(i,round,stream,histories[i],shape);
    else runTuple(family,round,stream,histories.front(),shape);
    HIP(hipStreamDestroy(stream));
}
