/**
 * @file MoEOverlaySharedBackingStartup.cpp
 * @brief Original creator/root/file authentication before shared-channel mmap.
 *
 * Each real participant receives only its fixed declaration, live basename and
 * required ACK/terminal metadata. Actual IMPIContext requests retain stack
 * packets until MPI_Test proves completion. Unknown request ownership is fatal;
 * ordinary physical rejection is drained through Abort before any mapping.
 */
#include "MoEOverlaySharedBackingStartup.h"
#include "MoEOverlayMPIFatal.h"
#include "interfaces/IMPIContext.h"
#include "interfaces/IMPITopology.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <cerrno>
#include <type_traits>
#include <unistd.h>
namespace llaminar2 {
namespace {
    constexpr int kStartupTag=kMoEOverlaySharedBackingStartupTag;
    constexpr std::uint64_t kMagic=detail::kMoEOverlaySharedBackingMagicV1;
    using Phase=detail::MoEOverlaySharedBackingPhaseV1;
    using Declaration=detail::MoEOverlaySharedBackingDeclarationV1;
    using Reply=detail::MoEOverlaySharedBackingReplyV1;
    using Clock=std::chrono::steady_clock;
    static_assert(std::is_trivially_copyable_v<Declaration> && sizeof(Declaration)==19*sizeof(std::uint64_t));
    static_assert(std::is_trivially_copyable_v<Reply> && sizeof(Reply)==11*sizeof(std::uint64_t));
    /** @brief Emit the existing unbuffered native-failure diagnostic without releasing unknown requests. */
    [[noreturn]] void fatal(const IMPIContext &context,const char *reason)
    { abortMoEOverlayMPI(context.communicator(),context.rank(),"owned_shared_backing",reason); }
    /**
     * @brief Retain one actual request through its standard-deadline completion.
     *
     * This is the same ownership pattern as PlanningMPITransferMeasurement's
     * setup Packet: never unwind a live request or infer completion from a timeout.
     */
    class Packet final {
    public:
        /**
         * @brief Take ownership of an already-submitted native request.
         * @param context Original MPI interface that submitted and must complete the request.
         * @param request Actual nonblocking request; its packet storage outlives this scope.
         * @param deadline Shared startup deadline, never extended by an individual packet.
         */
        Packet(const IMPIContext &context,MPI_Request request,Clock::time_point deadline)
            :context_(context),request_(request),deadline_(deadline){}
        /** @brief Fail fatally rather than unwind storage still referenced by MPI. */
        ~Packet() { if(request_!=MPI_REQUEST_NULL)fatal(context_,"abandoned actual startup request"); }
        Packet(const Packet &)=delete;
        /**
         * @brief Wait through the original interface until native completion is proven.
         * @param receive Whether the request owns received bytes requiring exact extent validation.
         * @return Actual receive byte extent, or zero for a completed send.
         * @note A deadline or native error retains unknown request ownership through the fatal edge.
         */
        int complete(bool receive)
        {
            MPI_Status status{};
            try {
                while(!context_.test(&request_,&status))
                    if(Clock::now()>=deadline_)fatal(context_,"startup request exceeded canonical30-second deadline");
                return receive?context_.getCount(status,MPI_BYTE):0;
            } catch(...) {fatal(context_,"actual startup request completion failed");}
        }
    private:
        const IMPIContext &context_;MPI_Request request_;Clock::time_point deadline_;
    };
    /** @brief Submit only the necessary fixed/live startup bytes and retain them to actual completion. */
    void send(const IMPIContext &context,const void *data,std::size_t bytes,int target,Clock::time_point deadline)
    {
        try {Packet packet(context,context.isend(data,bytes,MPI_BYTE,target,kStartupTag),deadline);packet.complete(false);}
        catch(...) {fatal(context,"actual startup send submission failed");}
    }
    /** @return Exact native receive extent; capacity is never a communication extent. */
    int receive(const IMPIContext &context,void *data,std::size_t capacity,int source,Clock::time_point deadline)
    {
        try {Packet packet(context,context.irecv(data,capacity,MPI_BYTE,source,kStartupTag),deadline);return packet.complete(true);}
        catch(...) {fatal(context,"actual startup receive submission failed");}
    }
    /** @return Epoch/file facts bound to the original received declaration. */
    Reply reply(const Declaration &declaration,Phase phase)
    {return {kMagic,static_cast<std::uint64_t>(phase),declaration.kind,declaration.node,declaration.generation,
        declaration.low,declaration.high,declaration.file,declaration.bytes};}
    /** @brief Reject malformed/stale replies instead of treating another transaction as a cleanup witness. */
    void requireReply(const IMPIContext &context,const Reply &actual,const Declaration &original)
    {
        if(actual.magic!=kMagic || actual.kind!=original.kind || actual.node!=original.node ||
           actual.generation!=original.generation || actual.low!=original.low || actual.high!=original.high ||
           actual.file!=original.file || actual.bytes!=original.bytes)
            fatal(context,"startup reply has stale/foreign epoch, original file or extent");
    }
}
MPISessionOwner::SharedFile establishMoEOverlaySharedBacking(
    const IMPIContext &context,const MoEOverlaySharedBackingEpoch &epoch,int creator,
    std::span<const int> participants,const std::string &basename,std::uint64_t bytes)
{
    const auto *topology=context.topology();
    if((epoch.kind!=MoEOverlaySharedBackingKind::ActivationRows && epoch.kind!=MoEOverlaySharedBackingKind::DeviceControllerFabric) ||
       !topology || !epoch.node_namespace || !epoch.generation || !bytes || basename.empty() || basename.size()>255 ||
       creator<0 || creator>=context.world_size() || participants.empty() ||
       !std::is_sorted(participants.begin(),participants.end()) || participants.front()<0 ||
       participants.back()>=context.world_size() || std::adjacent_find(participants.begin(),participants.end())!=participants.end() ||
       !std::binary_search(participants.begin(),participants.end(),creator) ||
       !std::binary_search(participants.begin(),participants.end(),context.rank()) ||
       topology->node_shared_memory_namespace()!=epoch.node_namespace ||
       std::any_of(participants.begin(),participants.end(),[&](int rank){return !topology->same_node(creator,rank);}))
        throw std::invalid_argument("Owned shared backing requires exact existing node-local startup topology/epoch");
    const auto deadline=Clock::now()+std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs);
    std::optional<MPISessionOwner::SharedDirectory> root;
    std::string failure;
    try {root=MPISessionOwner::sharedDirectory();if(!root)failure="Owned shared backing requires llaminar_mpi_session around external mpirun";}
    catch(const std::exception &error) {failure=error.what();}
    MPISessionOwner::SharedFile file;
    Declaration declaration;
    declaration.kind=static_cast<std::uint64_t>(epoch.kind);declaration.node=epoch.node_namespace;
    declaration.generation=epoch.generation;declaration.low=epoch.fingerprint_low;declaration.high=epoch.fingerprint_high;
    declaration.source=static_cast<std::uint64_t>(creator);declaration.participants=participants.size();
    declaration.bytes=bytes;declaration.name_bytes=basename.size();
    if(root)declaration.root=root->identity();
    if(context.rank()==creator) {
        if(failure.empty())try {
            file=root->createExclusive(basename);
            if(bytes>static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()))
                throw std::invalid_argument("Owned shared backing extent exceeds native off_t");
            if(::ftruncate(file.descriptor(),static_cast<off_t>(bytes))!=0)
                throw std::system_error(errno,std::generic_category(),"Cannot size exact original owned shared backing");
            file.requireExtent(bytes);declaration.file=file.identity();
        } catch(const std::exception &error) {failure=error.what();}
        declaration.phase=static_cast<std::uint64_t>(failure.empty()?Phase::Created:Phase::Failed);
        bool rejected=!failure.empty();
        for(int participant:participants)if(participant!=creator) {
            declaration.target=static_cast<std::uint64_t>(participant);
            send(context,&declaration,sizeof(declaration),participant,deadline);
            send(context,basename.data(),basename.size(),participant,deadline);
            Reply acknowledgement;
            if(receive(context,&acknowledgement,sizeof(acknowledgement),participant,deadline)!=sizeof(acknowledgement))
                fatal(context,"startup ACK has partial/wrong live extent");
            requireReply(context,acknowledgement,declaration);
            if(acknowledgement.phase!=static_cast<std::uint64_t>(Phase::Accepted) &&
               acknowledgement.phase!=static_cast<std::uint64_t>(Phase::Rejected))fatal(context,"startup ACK has invalid phase");
            rejected|=acknowledgement.phase==static_cast<std::uint64_t>(Phase::Rejected);
        }
        if(!rejected)try { file.requireExtent(bytes); }
            catch(const std::exception &error) {failure=error.what();rejected=true;}
        const Reply terminal=reply(declaration,rejected?Phase::Abort:Phase::Commit);
        for(int participant:participants)if(participant!=creator)send(context,&terminal,sizeof(terminal),participant,deadline);
        if(rejected)throw std::runtime_error(failure.empty()?"A peer rejected the creator-published original shared backing":failure);
    } else {
        Declaration received;
        if(receive(context,&received,sizeof(received),creator,deadline)!=sizeof(received))
            fatal(context,"startup declaration has partial/wrong live extent");
        std::array<char,255> name{};
        const int actual_name_bytes=receive(context,name.data(),name.size(),creator,deadline);
        if(received.magic!=kMagic || received.version!=1 || received.kind!=declaration.kind ||
           received.node!=declaration.node || received.generation!=declaration.generation ||
           received.low!=declaration.low || received.high!=declaration.high || received.source!=declaration.source ||
           received.target!=static_cast<std::uint64_t>(context.rank()) || received.participants!=declaration.participants ||
           received.bytes!=bytes || received.name_bytes!=basename.size() || actual_name_bytes!=static_cast<int>(basename.size()) ||
           std::memcmp(name.data(),basename.data(),basename.size()) ||
           (received.phase!=static_cast<std::uint64_t>(Phase::Created) && received.phase!=static_cast<std::uint64_t>(Phase::Failed)))
            fatal(context,"startup declaration has malformed/stale channel, epoch, role, name or extent");
        if(failure.empty())try {
            if(received.root!=root->identity())throw std::runtime_error("Owned shared backing participants have different original directories");
            if(received.phase!=static_cast<std::uint64_t>(Phase::Created))throw std::runtime_error("Original shared backing creator failed before publication");
            file=root->openExisting(basename,received.file,bytes);
        } catch(const std::exception &error) {failure=error.what();}
        const Reply acknowledgement=reply(received,failure.empty()?Phase::Accepted:Phase::Rejected);
        send(context,&acknowledgement,sizeof(acknowledgement),creator,deadline);
        Reply terminal;
        if(receive(context,&terminal,sizeof(terminal),creator,deadline)!=sizeof(terminal))
            fatal(context,"startup terminal has partial/wrong live extent");
        requireReply(context,terminal,received);
        if(terminal.phase!=static_cast<std::uint64_t>(Phase::Commit) && terminal.phase!=static_cast<std::uint64_t>(Phase::Abort))
            fatal(context,"startup terminal has invalid phase");
        if(terminal.phase==static_cast<std::uint64_t>(Phase::Abort))
            throw std::runtime_error(failure.empty()?"Original shared backing setup aborted before mapping":failure);
        if(!failure.empty() || received.phase!=static_cast<std::uint64_t>(Phase::Created))
            fatal(context,"startup committed a rejected or uncreated original file");
    }
    file.requireExtent(bytes);
    return file;
}
}
