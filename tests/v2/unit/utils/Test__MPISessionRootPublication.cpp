/**
 * @file Test__MPISessionRootPublication.cpp
 * @brief Actual CPU owner rejection retains the first failed root syscall and errno.
 *
 * Every negative child retires its inherited bootstrap loan descriptors and
 * supplies deliberately invalid native publication inputs. It never initializes
 * MPI, creates native readiness, changes parent environment/limits or removes an
 * original runtime root. Only this fixture's fresh nested directory is removed
 * after all children have exited. Recorded inputs certify diagnostics only.
 */
#include "utils/MPISessionOwner.h"
#include "utils/MPISessionEnvironment.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
namespace {
using namespace llaminar2;
using namespace llaminar2::mpi_session;
/** @brief Stop this original test on a missing assertion; no failure is retried. */
void require(bool accepted,const char* reason){if(!accepted)throw std::runtime_error(reason);}
/** @brief Supply one deliberately invalid native publication in a disposable child. */
int reject(const std::filesystem::path& base,bool shared,unsigned fault)
{
    for(int descriptor:{5,6,7,8})(void)::close(descriptor);
    const auto temporary=base/"temporary",memory=base/"shared";
    const auto bad=fault==0?base/"missing":fault==1?base/"regular":shared?memory:temporary;
    const auto temporaryPath=shared?temporary:bad,sharedPath=shared?bad:memory;
    for(const auto pair:{std::pair{EnvironmentKey::OmWorldSize,std::string("1")},
        std::pair{EnvironmentKey::RequiredProtocol,std::string(kProtocol)},
        std::pair{EnvironmentKey::ReadyPublication,std::string("invalid-negative-publication")},
        std::pair{EnvironmentKey::TemporaryDirectory,temporaryPath.string()},
        std::pair{EnvironmentKey::VaderBackingDirectory,sharedPath.string()}})
        require(::setenv(environmentName(pair.first),pair.second.c_str(),1)==0,"Cannot set negative publication");
    std::vector<int> occupied;
    if(fault==2){
        rlimit limit{};require(::getrlimit(RLIMIT_NOFILE,&limit)==0,"Cannot read child descriptor bound");
        limit.rlim_cur=32;require(::setrlimit(RLIMIT_NOFILE,&limit)==0,"Cannot bound child descriptors");
        for(;;){const int fd=::open("/dev/null",O_RDONLY|O_CLOEXEC);if(fd<0){require(errno==EMFILE,"Unexpected descriptor pressure errno");break;}occupied.push_back(fd);}
        if(shared){require(!occupied.empty(),"No descriptor retained");::close(occupied.back());occupied.pop_back();}
    }
    const int expected=fault==0?ENOENT:fault==1?ENOTDIR:EMFILE;
    bool rejected=false;
    try{MPISessionOwner::requireProcessOwner();}
    catch(const std::logic_error& error){
        const std::string message=error.what();
        rejected=message.starts_with("Owned MPI rank lost its original opened root publication;") &&
            message.find("operation=open;")!=std::string::npos &&
            message.find(std::string("root=")+(shared?"shared":"temporary")+";")!=std::string::npos &&
            message.find("path="+bad.string()+";")!=std::string::npos &&
            message.find("errno="+std::to_string(expected)+" (")!=std::string::npos;
        if(!rejected)std::fprintf(stderr,"Unexpected diagnostic: %s\n",error.what());
    }
    return rejected?0:1;
}
}
/** @brief Exercise ENOENT/ENOTDIR/EMFILE independently on both original root operations. */
int main()
{
    auto pattern=(std::filesystem::temp_directory_path()/"mpi-root-open-replay-XXXXXX").string();
    std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);
    const char* created=::mkdtemp(name.data());if(!created)return 1;
    const std::filesystem::path base=created;
    try{
        std::filesystem::create_directory(base/"temporary");std::filesystem::create_directory(base/"shared");
        std::ofstream(base/"regular")<<"fixture";
        for(unsigned repeat=0;repeat<20;++repeat)for(bool shared:{false,true})for(unsigned fault=0;fault<3;++fault){
            const auto child=::fork();require(child>=0,"Cannot fork original negative fixture");
            if(child==0){try{std::_Exit(reject(base,shared,fault));}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());std::_Exit(2);}}
            int status=0;pid_t joined;do{joined=::waitpid(child,&status,0);}while(joined<0 && errno==EINTR);
            // An unproven join cannot authorize even fixture directory cleanup.
            if(joined!=child)std::terminate();
            require(WIFEXITED(status) && WEXITSTATUS(status)==0,"Original root failure lost attribution");
            require(std::filesystem::is_directory(base/"temporary") && std::filesystem::is_directory(base/"shared"),"Owner diagnostic mutated root");
        }
        std::filesystem::remove_all(base);std::puts("120 original root rejection observations passed");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());std::filesystem::remove_all(base);return 1;}
}
