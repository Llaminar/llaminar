/** @file NativeHostServiceTLSProgram.h
 * @brief Closed diagnostic lazy-TLS task over the original loader module IDs and checked geometry.
 *
 * This exercises libc TLS storage on actual admitted workers. It is deliberately
 * not an ImmutableLookupService or DeviceGraphExecutor workload certificate.
 * Storage is forced; arbitrary C++ TLS constructors are not invoked.
 */
#pragma once
#include <link.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" void* __tls_get_addr(void*);
namespace llaminar2::test {
/** @brief Immutable TLS geometry copied under the loader's own traversal lock. */
class NativeHostServiceTLSProgram final {
public:
    struct Module {std::size_t id,fileBytes,memoryBytes,alignment;};
    /** @brief Capture existing modules only; this neither loads a provider nor creates a worker. */
    static NativeHostServiceTLSProgram original(){
        NativeHostServiceTLSProgram program;
        const auto page=::sysconf(_SC_PAGESIZE);if(page<=0)throw std::runtime_error("Diagnostic TLS page geometry unavailable");program.page_=static_cast<std::size_t>(page);
        struct Image{Module module;std::uint64_t offset;std::string path;};
        struct Snapshot{std::vector<Image> images;std::exception_ptr error;std::uint64_t adds=0,subs=0;bool counters=false;} snapshot;
        const auto copy=[](dl_phdr_info* info,std::size_t bytes,void* opaque)->int{
            auto& state=*static_cast<Snapshot*>(opaque);try{
                if(bytes<offsetof(dl_phdr_info,dlpi_tls_modid)+sizeof(info->dlpi_tls_modid))throw std::runtime_error("Diagnostic TLS loader ABI is incomplete");
                if(state.counters&&(state.adds!=info->dlpi_adds||state.subs!=info->dlpi_subs))throw std::runtime_error("Diagnostic TLS loader changed during traversal");
                state.counters=true;state.adds=info->dlpi_adds;state.subs=info->dlpi_subs;
                for(std::size_t i=0;i<info->dlpi_phnum;++i){const auto& ph=info->dlpi_phdr[i];if(ph.p_type!=PT_TLS)continue;
                    if(ph.p_filesz>ph.p_memsz||!info->dlpi_tls_modid||(ph.p_align&&(ph.p_align&(ph.p_align-1))))throw std::runtime_error("Diagnostic TLS module geometry is invalid");
                    if(!ph.p_memsz)continue;
                    state.images.push_back({{info->dlpi_tls_modid,ph.p_filesz,ph.p_memsz,ph.p_align},ph.p_offset,info->dlpi_name&&*info->dlpi_name?info->dlpi_name:"/proc/self/exe"});
                }
                return 0;
            }catch(...){state.error=std::current_exception();return 1;}
        };
        dl_iterate_phdr(copy,&snapshot);if(snapshot.error)std::rethrow_exception(snapshot.error);
        for(const auto& image:snapshot.images){
            struct stat file{};if(::stat(image.path.c_str(),&file)||file.st_size<0||!S_ISREG(file.st_mode))throw std::runtime_error("Diagnostic TLS original file extent unavailable");
            const auto size=static_cast<std::uint64_t>(file.st_size);
            if(image.offset>size||image.module.fileBytes>size-image.offset||image.module.memoryBytes>std::numeric_limits<std::size_t>::max()-program.bytes_)throw std::runtime_error("Diagnostic TLS initializer/total extent is invalid");
            for(const auto& prior:program.modules_)if(prior.id==image.module.id)throw std::runtime_error("Diagnostic TLS module ID is duplicated");
            program.bytes_+=image.module.memoryBytes;program.modules_.push_back(image.module);
        }
        program.adds_=snapshot.adds;program.subs_=snapshot.subs;return program;
    }
    /** @brief Force each existing dynamic TLS block and commit every page without changing stored values. */
    bool touchAll()const noexcept{
        struct TLSIndex{std::size_t module,offset;};
        static_assert(sizeof(TLSIndex)==2*sizeof(std::uintptr_t));
        for(const auto& module:modules_){
            TLSIndex index{module.id,0};auto* memory=static_cast<volatile unsigned char*>(::__tls_get_addr(&index));
            if(!memory||module.memoryBytes>std::numeric_limits<std::uintptr_t>::max()-reinterpret_cast<std::uintptr_t>(memory))return false;
            for(std::size_t at=0;at<module.memoryBytes;){const auto value=memory[at];memory[at]=value;if(module.memoryBytes-at<=page_)break;at+=page_;}
            const auto last=module.memoryBytes-1;const auto value=memory[last];memory[last]=value;
        }
        return true;
    }
    std::size_t modules()const noexcept{return modules_.size();}
    std::size_t bytes()const noexcept{return bytes_;}
    std::uint64_t loaderAdds()const noexcept{return adds_;}
    std::uint64_t loaderSubs()const noexcept{return subs_;}
private:
    std::vector<Module> modules_;
    std::size_t page_=0,bytes_=0;
    std::uint64_t adds_=0,subs_=0;
};
}
