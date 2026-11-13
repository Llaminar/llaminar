/**
 * @file PooledNativeProgramObservation.cpp
 * @brief Fixed-buffer host RSS/libc snapshots for explicit actual-program diagnostic targets.
 *
 * RSS includes diagnostic/runtime overhead. Allocator counters cover glibc only;
 * neither is exact native backing or a production General host certificate.
 */
#include "PooledNativeProgramObservation.h"
#include "PooledNativeGraphInventory.h"
#include <malloc.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <limits>
#include <cstring>
#include <sys/syscall.h>
namespace llaminar2::test::pooled_program_observation {
thread_local CallScope* CallScope::current_=nullptr;
CallScope::CallScope(const char* backend,const char* operation,std::uint64_t ordinal)noexcept
    :backend_(backend),operation_(operation),ordinal_(ordinal),thread_(0),depth_(current_?current_->depth_+1:1),parent_(current_){
    const auto thread=::syscall(SYS_gettid);if(thread<=0 || !backend || !operation || !ordinal || depth_>64)std::abort();
    thread_=static_cast<std::uint64_t>(thread);current_=this;
}
CallScope::~CallScope()noexcept{if(current_!=this)std::abort();current_=parent_;}
HostSample sample()noexcept{
    HostSample out{};timespec now{};if(clock_gettime(CLOCK_MONOTONIC,&now))std::abort();
    out.monotonic_ns=std::uint64_t(now.tv_sec)*1000000000ull+std::uint64_t(now.tv_nsec);
    rusage usage{};if(getrusage(RUSAGE_SELF,&usage))std::abort();out.peak=std::uint64_t(usage.ru_maxrss)*1024;
    const auto allocation=mallinfo2();out.allocator_live=allocation.uordblks;out.allocator_arena=allocation.arena;out.allocator_mapped=allocation.hblkhd;
    const int fd=open("/proc/self/statm",O_RDONLY|O_CLOEXEC);if(fd<0)std::abort();char text[256]{};
    const auto n=read(fd,text,sizeof(text)-1);const int closed=close(fd);if(n<=0 || closed)std::abort();
    unsigned long long pages=0,resident=0;if(std::sscanf(text,"%llu %llu",&pages,&resident)!=2)std::abort();
    const auto page=sysconf(_SC_PAGESIZE);if(page<=0 || resident>std::numeric_limits<std::uint64_t>::max()/std::uint64_t(page))std::abort();
    out.rss=resident*std::uint64_t(page);return out;
}
void record(const char* backend,const char* operation,const char* phase,std::uint64_t ordinal,
    const void* handle,const void* caller,int result,const HostSample& s,const void* stream,std::uint64_t flags)noexcept{
    const auto* call=CallScope::current_;
    if(!call || call->ordinal_!=ordinal || std::strcmp(call->backend_,backend) || std::strcmp(call->operation_,operation))std::abort();
    const auto* parent=call->parent_;
    char line[1024];const int n=std::snprintf(line,sizeof(line),
        "POOLED_NATIVE_API {\"schema\":2,\"pid\":%ld,\"thread\":%llu,\"parent_backend\":\"%s\",\"parent_operation\":\"%s\",\"parent_ordinal\":%llu,\"backend\":\"%s\",\"operation\":\"%s\",\"phase\":\"%s\",\"ordinal\":%llu,\"handle\":\"%p\",\"caller\":\"%p\",\"stream\":\"%p\",\"flags\":%llu,\"result\":%d,\"rss_bytes\":%llu,\"peak_rss_bytes\":%llu,\"glibc_live_bytes\":%llu,\"glibc_arena_bytes\":%llu,\"glibc_mapped_bytes\":%llu,\"monotonic_ns\":%llu,\"qualified\":false}\n",
        static_cast<long>(getpid()),static_cast<unsigned long long>(call->thread_),parent?parent->backend_:"",parent?parent->operation_:"",static_cast<unsigned long long>(parent?parent->ordinal_:0),backend,operation,phase,static_cast<unsigned long long>(ordinal),handle,caller,stream,static_cast<unsigned long long>(flags),result,
        static_cast<unsigned long long>(s.rss),static_cast<unsigned long long>(s.peak),static_cast<unsigned long long>(s.allocator_live),
        static_cast<unsigned long long>(s.allocator_arena),static_cast<unsigned long long>(s.allocator_mapped),static_cast<unsigned long long>(s.monotonic_ns));
    if(n<=0 || std::size_t(n)>=sizeof(line))std::abort();
    std::size_t sent=0;while(sent<std::size_t(n)){const auto done=write(STDERR_FILENO,line+sent,std::size_t(n)-sent);if(done<0 && errno==EINTR)continue;if(done<=0)std::abort();sent+=std::size_t(done);}
}
bool rejectInstantiation(std::uint64_t ordinal,const char* backend)noexcept{
    // Explicit test-only injection returns an SDK allocation error AFTER the
    // original graph capture prefix; it never invents a successful handle.
    const char* text=std::getenv("LLAMINAR_POOLED_DIAGNOSTIC_FAIL_INSTANTIATE");
    const char* selected=std::getenv("LLAMINAR_POOLED_DIAGNOSTIC_FAIL_BACKEND");
    if(selected && (!text || !backend || (std::strcmp(selected,"CUDA") && std::strcmp(selected,"ROCm")) ||
        (std::strcmp(backend,"CUDA") && std::strcmp(backend,"ROCm"))))std::abort();
    if(!text)return false;
    if(!*text)std::abort();std::uint64_t number=0;
    for(;*text;++text){if(*text<'0'||*text>'9'||number>(std::numeric_limits<std::uint64_t>::max()-9)/10)std::abort();number=number*10+std::uint64_t(*text-'0');}
    if(!number)std::abort();return (!selected || !std::strcmp(selected,backend)) && ordinal==number;
}
}
namespace llaminar2::test::pooled_program_observation {
void inventory(const char* backend,const char* operation,std::uint64_t ordinal,
    const void* graph,const char* observation,std::size_t nodes,std::size_t edges,
    std::size_t index,const void* node,const void* other,int kind,const char* error)noexcept{
    const auto thread=::syscall(SYS_gettid);if(thread<=0)std::abort();
    char line[1024];const int n=std::snprintf(line,sizeof(line),
        "POOLED_NATIVE_GRAPH_INVENTORY {\"schema\":1,\"pid\":%ld,\"thread\":%ld,\"backend\":\"%s\",\"operation\":\"%s\",\"ordinal\":%llu,\"graph\":\"%p\",\"observation\":\"%s\",\"nodes\":%zu,\"edges\":%zu,\"index\":%zu,\"node\":\"%p\",\"other\":\"%p\",\"native_kind\":%d,\"error\":\"%s\",\"qualified\":false}\n",
        static_cast<long>(getpid()),thread,backend,operation,static_cast<unsigned long long>(ordinal),graph,observation,nodes,edges,index,node,other,kind,error);
    if(n<=0 || std::size_t(n)>=sizeof(line))std::abort();
    std::size_t sent=0;while(sent<std::size_t(n)){const auto done=write(STDERR_FILENO,line+sent,std::size_t(n)-sent);if(done<0 && errno==EINTR)continue;if(done<=0)std::abort();sent+=std::size_t(done);}
}
void topology(const char* backend,const char* observation,const void* graph,const void* node,int kind,int result)noexcept{
    char line[512];const int n=std::snprintf(line,sizeof(line),
        "POOLED_NATIVE_TOPOLOGY {\"schema\":1,\"pid\":%ld,\"backend\":\"%s\",\"observation\":\"%s\",\"graph\":\"%p\",\"node\":\"%p\",\"native_kind\":%d,\"result\":%d,\"qualified\":false}\n",
        static_cast<long>(getpid()),backend,observation,graph,node,kind,result);
    if(n<=0 || std::size_t(n)>=sizeof(line))std::abort();
    std::size_t sent=0;while(sent<std::size_t(n)){const auto done=write(STDERR_FILENO,line+sent,std::size_t(n)-sent);if(done<0 && errno==EINTR)continue;if(done<=0)std::abort();sent+=std::size_t(done);}
}
}
