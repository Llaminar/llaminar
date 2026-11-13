/**
 * @file MoEOverlaySharedBackingStartupProbe.cpp
 * @brief Actual CPU MPI startup/file handshake with retained native fault packets.
 *
 * The production helper and MPIContext perform every send, receive and file
 * operation. A test-only context can corrupt a copy of one actual packet or
 * pause after actual receive completion. Copies stay live until the original
 * MPI_Request completes; no fake publisher, completion or process cleanup is
 * substituted. The ordinary native owner alone retires leftover backing.
 *
 * The custom entry owns one lexical native GPU lifetime without probing a
 * device. Native workers/caches retire before process exit and MPI finalization.
 */
#include "backends/GPUDeviceProcessLifetime.h"
#include "execution/moe/MoEOverlaySharedBackingStartup.h"
#include "utils/MPISessionOwner.h"
#include "utils/MPIContext.h"
#include "interfaces/IMPITopology.h"
#include <nlohmann/json.hpp>
#include <mpi.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
using namespace llaminar2;
namespace {
using Declaration=detail::MoEOverlaySharedBackingDeclarationV1;
using Reply=detail::MoEOverlaySharedBackingReplyV1;
/** @brief Publish only complete diagnostic observations outside runtime roots. */
void publish(const std::filesystem::path &path,const nlohmann::json &value)
{
    const auto pending=path.string()+".pending";std::ofstream output(pending);output<<value.dump()<<'\n';output.close();
    if(!output)throw std::runtime_error("Cannot publish native startup observation");std::filesystem::rename(pending,path);
}
/** @brief Real MPI context plus narrowly retained wire faults for this fixture. */
class FaultContext final : public MPIContext {
    struct Request final {bool receive;void *data;std::size_t capacity;std::shared_ptr<std::vector<std::byte>> copy;};
public:
    FaultContext(int rank,int size,std::string mode,std::filesystem::path reports,std::string basename)
        :MPIContext(rank,size,MPI_COMM_WORLD),mode_(std::move(mode)),reports_(std::move(reports)),basename_(std::move(basename))
    {directory_=MPISessionOwner::sharedDirectory();if(!directory_)throw std::runtime_error("Native startup probe needs actual owned directory");}
    /** @brief Prepare a real existing neighbor only for the actual creator-failure branch. */
    void prepare()
    {
        if(rank()==0 && mode_=="creator_failure") {
            neighbor_.emplace(directory_->createExclusive(basename_));
            if(::ftruncate(neighbor_->descriptor(),4096))throw std::runtime_error("Cannot size diagnostic neighbor");
            neighbor_identity_=neighbor_->identity();
        }
    }
    MPI_Request isend(const void *data,size_t count,MPI_Datatype type,int target,int tag) const override
    {
        if(tag!=kMoEOverlaySharedBackingStartupTag)return MPIContext::isend(data,count,type,target,tag);
        std::shared_ptr<std::vector<std::byte>> copy;
        size_t live=count;
        if(rank()==0 && !injected_ && count==sizeof(Declaration)) {
            Declaration actual;std::memcpy(&actual,data,sizeof(actual));
            const auto path=directory_->path()+"/"+basename_;
            publish(reports_/"creator.json",{{"pid",::getpid()},{"path",path},{"dev",actual.file.device},
                {"ino",actual.file.inode},{"uid",actual.file.uid},{"bytes",actual.bytes}});
            if(mode_=="wrong_size") {if(::truncate(path.c_str(),4095))throw std::runtime_error("Cannot inject exact size rejection");}
            if(mode_=="wrong_inode") {
                saved_=path+"-retained-original";original_identity_=actual.file;
                if(::rename(path.c_str(),saved_.c_str()))throw std::runtime_error("Cannot retain diagnostic original inode");
                replacement_.emplace(directory_->createExclusive(basename_));
                if(::ftruncate(replacement_->descriptor(),4096))throw std::runtime_error("Cannot size diagnostic replacement");
            }
            if(mode_=="malformed" || mode_=="partial") {
                copy=std::make_shared<std::vector<std::byte>>(count);std::memcpy(copy->data(),data,count);
                if(mode_=="malformed") {actual.version=99;std::memcpy(copy->data(),&actual,sizeof(actual));}
                else live-=sizeof(std::uint64_t);
            }
            injected_=true;
        } else if(rank()!=0 && !injected_ && mode_=="late_ack" && count==sizeof(Reply)) {
            copy=std::make_shared<std::vector<std::byte>>(count);Reply actual;std::memcpy(&actual,data,sizeof(actual));
            --actual.generation;std::memcpy(copy->data(),&actual,sizeof(actual));injected_=true;
        }
        const auto request=MPIContext::isend(copy?copy->data():data,live,type,target,tag);
        requests_.emplace(request,Request{false,nullptr,live,std::move(copy)});sent_bytes_+=live;return request;
    }
    MPI_Request irecv(void *data,size_t count,MPI_Datatype type,int source,int tag) const override
    {
        const auto request=MPIContext::irecv(data,count,type,source,tag);
        if(tag==kMoEOverlaySharedBackingStartupTag)requests_.emplace(request,Request{true,data,count,{}});
        return request;
    }
    bool test(MPI_Request *request,MPI_Status *status) const override
    {
        const auto original=*request;MPI_Status actual;
        const bool complete=MPIContext::test(request,status?status:&actual);
        if(!complete)return false;
        const auto found=requests_.find(original);
        if(found!=requests_.end()) {
            if(found->second.receive) {
                const int received=MPIContext::getCount(status?*status:actual,MPI_BYTE);received_bytes_+=static_cast<std::uint64_t>(received);
                if(mode_=="peer_exit" && rank()==1 && found->second.capacity==sizeof(Declaration) && received==sizeof(Declaration)) {
                    Declaration declared;std::memcpy(&declared,found->second.data,sizeof(declared));
                    publish(reports_/"peer-acquired.json",{{"pid",::getpid()},{"file_inode",declared.file.inode},
                        {"actual_received_bytes",received},{"mapped",false},{"request_completed",true}});
                    // Test-only pause after the actual native receive. Parent
                    // uses this exact pidfd to inject peer death before ACK.
                    for(;;)::pause();
                }
            }
            requests_.erase(found);
        }
        return true;
    }
    /** @brief Retire only exact diagnostic files after all real startup requests complete. */
    void finish() const
    {
        if(!requests_.empty())throw std::runtime_error("Native startup request storage still live");
        if(neighbor_) {if(neighbor_->identity()!=neighbor_identity_)throw std::runtime_error("Creator failure changed original neighbor");neighbor_->requireExtent(4096);}
        if(replacement_) {
            replacement_->requireExtent(4096);replacement_->unlink();replacement_.reset();
            struct stat original{};
            if(::lstat(saved_.c_str(),&original) || MPISessionOwner::SharedIdentity{original.st_dev,original.st_ino,original.st_uid}!=original_identity_ ||
               ::unlink(saved_.c_str()))throw std::runtime_error("Refuse cleanup of a changed diagnostic original inode");
        }
    }
    std::uint64_t sentBytes() const {return sent_bytes_;}
    std::uint64_t receivedBytes() const {return received_bytes_;}
private:
    std::string mode_;std::filesystem::path reports_;std::string basename_;
    std::optional<MPISessionOwner::SharedDirectory> directory_;
    mutable std::optional<MPISessionOwner::SharedFile> replacement_,neighbor_;
    mutable MPISessionOwner::SharedIdentity original_identity_{},neighbor_identity_{};
    mutable std::string saved_;mutable bool injected_=false;
    mutable std::map<MPI_Request,Request> requests_;
    mutable std::uint64_t sent_bytes_=0,received_bytes_=0;
};
}
/** @brief Run one exact production handshake, then truthful MPI/file retirement. */
int main(int argc,char **argv)
{
    if(auto terminal=MPISessionOwner::enterProcess(argc,argv))return *terminal;
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    if(argc!=3)return 125;const std::string mode=argv[1];const std::filesystem::path reports=argv[2];
    if(mode!="success" && mode!="subset" && mode!="creator_failure" && mode!="wrong_size" && mode!="wrong_inode" &&
       mode!="late_ack" && mode!="malformed" && mode!="partial" && mode!="peer_exit")return 125;
    int provided=0;if(MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided)!=MPI_SUCCESS || provided<MPI_THREAD_MULTIPLE)return 126;
    int rank=0,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    const std::string basename="moe-startup-probe";
    int result=0;
    try {
        FaultContext context(rank,size,mode,reports,basename);context.prepare();
        const auto *topology=context.topology();
        std::vector<int> participants;
        if(mode=="subset")participants={0,2};else for(int r=0;r<size;++r)participants.push_back(r);
        const bool active=std::binary_search(participants.begin(),participants.end(),rank);
        bool returned=false,mapped=false,failed=false;std::string error;MPISessionOwner::SharedFile file;
        if(active)try {
            file=establishMoEOverlaySharedBacking(context,{MoEOverlaySharedBackingKind::ActivationRows,
                topology->node_shared_memory_namespace(),37,41,43},0,participants,basename,4096);
            returned=true;void *mapping=::mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_SHARED,file.descriptor(),0);
            if(mapping==MAP_FAILED)throw std::runtime_error("Cannot mmap committed original file");mapped=true;::munmap(mapping,4096);
        } catch(const std::exception &failure) {failed=true;error=failure.what();}
        const bool expect_failure=mode=="creator_failure" || mode=="wrong_size" || mode=="wrong_inode";
        if(active && ((expect_failure && (returned || mapped || !failed)) || (!expect_failure && (!returned || !mapped || failed))))result=1;
        if(!active && (context.sentBytes() || context.receivedBytes()))result=1;
        publish(reports/("rank-"+std::to_string(rank)+".json"),{{"pid",::getpid()},{"rank",rank},{"returned",returned},{"mapped",mapped},
            {"failed",failed},{"error",error},{"sent_bytes",context.sentBytes()},{"received_bytes",context.receivedBytes()}});
        MPI_Barrier(MPI_COMM_WORLD);context.finish();
        int global=0;MPI_Allreduce(&result,&global,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);result=global;
    } catch(const std::exception &error) {std::fprintf(stderr,"Native startup probe failed: %s\n",error.what());MPI_Abort(MPI_COMM_WORLD,126);return 126;}
    llaminar2::GPUDeviceProcessLifetime::retireActive();
    if(MPI_Finalize()!=MPI_SUCCESS)return 126;return result;
}
