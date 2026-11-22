/**
 * @file CUDAPeerDecodeLatency.cu
 * @brief Compare receiver-visible CUDA peer transfers and FP32 decode reductions.
 *
 * Two original device streams join through native graph event edges. One
 * independent capture owns the elapsed interval, and the exact installed
 * NCCL provider supplies the control. The fixture verifies complete
 * publications after each reset, outside the timed graph.
 */

#include <cuda_runtime.h>
#include LLAMINAR_NCCL_ABI_HEADER
#include <array>
#include <cstdio>
#include <limits>
#include <string>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
constexpr int iterations=32;

void check(cudaError_t result,const char* operation) {
    if (result!=cudaSuccess)
        throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(result));
}
void check(ncclResult_t result,const char* operation) {
    if (result!=ncclSuccess)
        throw std::runtime_error(std::string(operation)+": "+ncclGetErrorString(result));
}

/** @brief Deterministic finite FP32 terms with exact sums in the exercised range. */
float value(size_t index,int rank,int seed) {
    return float((int(index % 251)*13+rank*53+seed*19)%509-254)*0.03125f;
}

/** @brief One local vector load and one directly addressed store into peer VRAM. */
__global__ void writePeer(const float4* source,float4* peer,size_t vectors) {
    const auto index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index<vectors)
        peer[index]=source[index];
}

/** @brief Publish the original two-rank FP32 add with the supplied rank order. */
__global__ void foldTwo(const float4* own,const float4* remote,float4* output,size_t vectors) {
    const auto index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index<vectors) {
        const auto a=own[index], b=remote[index];
        output[index]=make_float4(__fadd_rn(a.x,b.x),__fadd_rn(a.y,b.y),__fadd_rn(a.z,b.z),__fadd_rn(a.w,b.w));
    }
}

/** @brief Own one physical CUDA participant and disjoint fixture publications. */
struct Participant {
    int id;
    char pci[32]{};
    cudaStream_t stream{};
    cudaEvent_t sent{}, joined{};
    float* input=nullptr;
    float* inbox=nullptr;
    float* output=nullptr;

    explicit Participant(int ordinal,size_t bytes):id(ordinal) {
        use();
        check(cudaDeviceGetPCIBusId(pci,sizeof(pci),id),"physical PCI identity");
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"exact participant stream");
        check(cudaEventCreateWithFlags(&sent,cudaEventDisableTiming),"publication event");
        check(cudaEventCreateWithFlags(&joined,cudaEventDisableTiming),"join event");
        check(cudaMalloc(&input,bytes),"native input");
        check(cudaMalloc(&inbox,bytes),"receiver-local VRAM inbox");
        check(cudaMalloc(&output,bytes),"receiver-local publication");
    }
    ~Participant() {
        cudaSetDevice(id);
        cudaFree(input);
        cudaFree(inbox);
        cudaFree(output);
        cudaEventDestroy(sent);
        cudaEventDestroy(joined);
        cudaStreamDestroy(stream);
    }
    void use() const {check(cudaSetDevice(id),"select participant");}
    /** @brief Reset input and poison stale publications outside the captured interval. */
    void seed(size_t elements,int rank,int pattern) {
        use();
        std::vector<float> data(elements);
        for (size_t i=0;i<elements;++i) data[i]=value(i,rank,pattern);
        check(cudaMemcpyAsync(input,data.data(),elements*sizeof(float),cudaMemcpyHostToDevice,stream),"seed original");
        check(cudaMemsetAsync(inbox,0,elements*sizeof(float),stream),"dirty prior inbox cleared");
        check(cudaMemsetAsync(output,0,elements*sizeof(float),stream),"dirty prior output cleared");
        check(cudaStreamSynchronize(stream),"finish fixture setup");
    }
};

enum class Mode { Copy, Direct, DirectRoot, PullRoot, PublishRoot, NCCLRoot, DirectAll, PullAll, NCCLAll };

bool oneWay(Mode mode) {return mode==Mode::Copy || mode==Mode::Direct;}
bool rooted(Mode mode) {return mode==Mode::DirectRoot || mode==Mode::PullRoot || mode==Mode::PublishRoot || mode==Mode::NCCLRoot;}

/** @brief Retain one two-stream native capture for the duration of a measurement. */
struct Tape {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    ~Tape() {if (exec) cudaGraphExecDestroy(exec); if (graph) cudaGraphDestroy(graph);}
};

/** @brief Publish the completion of one directed DMA or CUDA kernel store. */
void send(Participant& source,Participant& destination, size_t vectors,unsigned blocks,unsigned threads,bool dma) {
    source.use();
    if (dma)
        check(cudaMemcpyAsync(destination.inbox,source.input,vectors*sizeof(float4),cudaMemcpyDeviceToDevice,source.stream),
              "peer DMA");
    else {
        writePeer<<<blocks,threads,0,source.stream>>>(reinterpret_cast<const float4*>(source.input),
            reinterpret_cast<float4*>(destination.inbox),vectors);
        check(cudaPeekAtLastError(),"direct peer store");
    }
    check(cudaEventRecord(source.sent,source.stream),"publish exact send completion");
}

/** @brief Join the incoming directed VRAM send before folding the two local views. */
void fold(Participant& receiver,Participant& source,size_t vectors,unsigned blocks,unsigned threads) {
    receiver.use();
    check(cudaStreamWaitEvent(receiver.stream,source.sent),"acquire peer completion");
    foldTwo<<<blocks,threads,0,receiver.stream>>>(reinterpret_cast<const float4*>(receiver.input),
        reinterpret_cast<const float4*>(receiver.inbox),reinterpret_cast<float4*>(receiver.output),vectors);
    check(cudaPeekAtLastError(),"ordered local publication");
    check(cudaEventRecord(receiver.joined,receiver.stream),"publish exact local fold");
}

/**
 * @brief Record the same complete operation interval for all modes.
 * @details Each iteration joins the two original streams before reusing
 * their checked buffers. In the pull and single-kernel publish modes both
 * source rows are immutable and ready before the first graph node.
 */
void record(Mode mode,Participant& source,Participant& destination,ncclComm_t source_comm,
            ncclComm_t destination_comm,size_t vectors,unsigned threads,Tape& tape) {
    const auto blocks=static_cast<unsigned>((vectors+threads-1)/threads);
    source.use();
    check(cudaStreamBeginCapture(source.stream,cudaStreamCaptureModeThreadLocal),"begin root capture");
    check(cudaEventRecord(source.joined,source.stream),"fork capture");
    destination.use();
    check(cudaStreamWaitEvent(destination.stream,source.joined),"join second stream to parent");

    for (int repeat=0;repeat<iterations;++repeat)
    {
        if (oneWay(mode)) {
            send(source,destination,vectors,blocks,threads,mode==Mode::Copy);
            destination.use();
            check(cudaStreamWaitEvent(destination.stream,source.sent),"acquire receiver-visible transfer");
            check(cudaEventRecord(destination.joined,destination.stream),"complete receiver timeline");
        } else if (mode==Mode::DirectRoot) {
            send(source,destination,vectors,blocks,threads,false);
            fold(destination,source,vectors,blocks,threads);
        } else if (mode==Mode::PullRoot || mode==Mode::PublishRoot || mode==Mode::PullAll) {
            auto issue=[&](Participant& executor,Participant& first,Participant& second,Participant& output) {
                executor.use();
                foldTwo<<<blocks,threads,0,executor.stream>>>(reinterpret_cast<const float4*>(first.input),
                    reinterpret_cast<const float4*>(second.input),reinterpret_cast<float4*>(output.output),vectors);
                check(cudaPeekAtLastError(),"ordered direct peer reduction with no stage inbox");
            };
            if (mode==Mode::PublishRoot) {
                issue(source,source,destination,destination);
                check(cudaEventRecord(source.sent,source.stream),"publish direct root VRAM success");
                destination.use();
                check(cudaStreamWaitEvent(destination.stream,source.sent),"acquire directly published root sum");
            } else {
                issue(destination,source,destination,destination);
                if (mode==Mode::PullAll)
                    issue(source,source,destination,source);
            }
            destination.use();
            check(cudaEventRecord(destination.joined,destination.stream),"publish peer final sum");
        } else if (mode==Mode::DirectAll) {
            send(source,destination,vectors,blocks,threads,false);
            send(destination,source,vectors,blocks,threads,false);
            fold(source,destination,vectors,blocks,threads);
            fold(destination,source,vectors,blocks,threads);
        } else {
            check(ncclGroupStart(),"NCCL group start");
            auto submit=[&](Participant& member,ncclComm_t comm) {
                member.use();
                if (mode==Mode::NCCLRoot)
                    check(ncclReduce(member.input,member.output,vectors*4,ncclFloat32,ncclSum,1,comm,member.stream),
                        "production-provider rooted reduce");
                else
                    check(ncclAllReduce(member.input,member.output,vectors*4,ncclFloat32,ncclSum,comm,member.stream),
                        "production-provider allreduce");
            };
            submit(source,source_comm);
            submit(destination,destination_comm);
            check(ncclGroupEnd(),"NCCL group commit");
            destination.use();
            check(cudaEventRecord(destination.joined,destination.stream),"publish NCCL receive");
        }
        source.use();
        check(cudaStreamWaitEvent(source.stream,destination.joined),"complete capture operation on source");
        // The next iteration's destination write must not overwrite an inbox
        // still owned by the previous iteration's receiver.
        check(cudaEventRecord(source.joined,source.stream),"publish operation retirement");
        destination.use();
        check(cudaStreamWaitEvent(destination.stream,source.joined),"reuse both original inboxes");
    }
    destination.use();
    check(cudaEventRecord(destination.joined,destination.stream),"close destination branch");
    source.use();
    check(cudaStreamWaitEvent(source.stream,destination.joined),"join capture");
    check(cudaStreamEndCapture(source.stream,&tape.graph),"capture shared parent");
    check(cudaGraphInstantiate(&tape.exec,tape.graph,0),"instantiate shared parent");
}

/** @brief Check every consumed byte at the real receiver outside native timing. */
void verify(Mode mode,Participant& source,Participant& target,size_t elements,int pattern) {
    auto check_output=[&](Participant& member,const float* pointer) {
        member.use();
        std::vector<float> observed(elements);
        check(cudaMemcpyAsync(observed.data(),pointer,elements*sizeof(float),cudaMemcpyDeviceToHost,member.stream),
              "read complete receiver publication");
        check(cudaStreamSynchronize(member.stream),"verify publication outside timer");
        for (size_t i=0;i<elements;++i) {
            const float expected=oneWay(mode)?value(i,0,pattern):value(i,0,pattern)+value(i,1,pattern);
            if (std::memcmp(&expected,&observed[i],sizeof(float)))
                throw std::runtime_error("peer publication differs at element "+std::to_string(i));
        }
    };
    check_output(target,oneWay(mode)?target.inbox:target.output);
    if (!oneWay(mode) && !rooted(mode)) check_output(source,source.output);
}

/** @brief Exercise reset and repeated capture before and after the event samples. */
void run(Mode mode,const char* name,Participant& source,Participant& target,
         ncclComm_t source_comm,ncclComm_t target_comm,size_t elements,unsigned threads) {
    Tape tape;
    source.seed(elements,0,19);
    target.seed(elements,1,19);
    record(mode,source,target,source_comm,target_comm,elements/4,threads,tape);
    for (int pattern:{19,42,97}) {
        source.seed(elements,0,pattern);
        target.seed(elements,1,pattern);
        source.use();
        check(cudaGraphLaunch(tape.exec,source.stream),"release single retained transaction");
        check(cudaStreamSynchronize(source.stream),"join complete graph for exact comparison");
        verify(mode,source,target,elements,pattern);
    }
    source.use();
    cudaEvent_t start{},end{};
    check(cudaEventCreate(&start),"native timer start");
    check(cudaEventCreate(&end),"native timer end");
    for (int sample=0;sample<51;++sample) {
        check(cudaEventRecord(start,source.stream),"native start");
        check(cudaGraphLaunch(tape.exec,source.stream),"retained graph");
        check(cudaEventRecord(end,source.stream),"native finish");
        check(cudaEventSynchronize(end),"retire sample");
        if (sample<8) continue;
        float milliseconds=0;
        check(cudaEventElapsedTime(&milliseconds,start,end),"retained event duration");
        std::printf("SAMPLE,%s,%s,%s,%zu,%d,%.6f\n",name,source.pci,target.pci,
            elements*sizeof(float),sample-8,double(milliseconds)*1000/iterations);
    }
    verify(mode,source,target,elements,97);
    source.use();
    check(cudaEventDestroy(start),"retire start event");
    check(cudaEventDestroy(end),"retire end event");
}
}

/** @brief Admit the measured device pair and run one exact native graph mode. */
int main(int argc,char** argv) {
    try {
        if (argc != 4 && argc != 5) throw std::runtime_error("usage: peer-test source-ordinal mode bytes [threads]");
        const int owner=std::stoi(argv[1]);
        if (owner!=0 && owner!=1) throw std::runtime_error("diagnostic requires one original endpoint of the two measured cards");
        const std::string_view name=argv[2];
        const auto requested_bytes = std::stoull(argv[3]);
        const auto requested_threads = argc == 5 ? std::stoul(argv[4]) : 128;
        if (!requested_bytes || requested_bytes % sizeof(float4) ||
            requested_bytes > std::numeric_limits<size_t>::max() || !requested_threads ||
            requested_threads % 32 || requested_threads > std::numeric_limits<unsigned>::max() ||
            requested_bytes / sizeof(float4) > std::numeric_limits<unsigned>::max() - requested_threads)
            throw std::runtime_error("invalid CUDA peer diagnostic geometry");
        const auto bytes = static_cast<size_t>(requested_bytes);
        const auto threads = static_cast<unsigned>(requested_threads);
        Mode mode;
        if (name=="copy") mode=Mode::Copy;
        else if (name=="direct") mode=Mode::Direct;
        else if (name=="direct-root") mode=Mode::DirectRoot;
        else if (name=="pull-root") mode=Mode::PullRoot;
        else if (name=="publish-root") mode=Mode::PublishRoot;
        else if (name=="nccl-root") mode=Mode::NCCLRoot;
        else if (name=="direct-all") mode=Mode::DirectAll;
        else if (name=="pull-all") mode=Mode::PullAll;
        else if (name=="nccl-all") mode=Mode::NCCLAll;
        else throw std::runtime_error("unknown mode");
        Participant source(owner,bytes),target(1-owner,bytes);
        for (const auto* original:{&source,&target}) {
            cudaDeviceProp physical{};
            check(cudaGetDeviceProperties(&physical, original->id), "peer CTA resource envelope");
            if (threads > static_cast<unsigned>(physical.maxThreadsPerBlock))
                throw std::runtime_error("CTA width exceeds an original participant");
            int available=0,native_atomic=0;
            const int other=1-original->id;
            original->use();
            check(cudaDeviceCanAccessPeer(&available,original->id,other),"direct CUDA peer capability");
            check(cudaDeviceGetP2PAttribute(&native_atomic,cudaDevP2PAttrNativeAtomicSupported,original->id,other),
                  "native intercard atomics");
            std::printf("PEER,%d,%d,%d,%d\n",original->id,other,available,native_atomic);
            if (!available) throw std::runtime_error("required directed peer access unavailable");
            const auto enabled=cudaDeviceEnablePeerAccess(other,0);
            if (enabled!=cudaErrorPeerAccessAlreadyEnabled) check(enabled,"enable direct peer access");
        }
        std::printf("GRID,%u,%zu\n", threads, (bytes / sizeof(float4) + threads - 1) / threads);
        std::array<ncclComm_t,2> comms{};
        if (mode==Mode::NCCLAll || mode==Mode::NCCLRoot) {
            const int ordinals[]{source.id,target.id};
            int version=0;
            check(ncclGetVersion(&version),"provider version");
            std::printf("NCCL,%d\n",version);
            check(ncclCommInitAll(comms.data(),comms.size(),ordinals),"matched two-member communicator");
        }
        run(mode,argv[2],source,target,comms[0],comms[1],bytes/sizeof(float),threads);
        for (auto comm:comms) if (comm) check(ncclCommDestroy(comm),"retire native communicator");
        std::puts("CHECK,pass");
        return 0;
    } catch(const std::exception& failure) {
        std::fprintf(stderr,"%s\n",failure.what());
        return 1;
    }
}
