/**
 * @file MPISessionSharedDirectory.cpp
 * @brief Collective startup validation of the process owner's actual shared root.
 *
 * MPI_COMM_TYPE_SHARED establishes physical local membership; original opened
 * directory dev/inode/uid establishes namespace identity. Neither PID layout,
 * rank numbers nor matching environment strings can stand in for those facts.
 */
#include "utils/MPISessionSharedDirectory.h"
#include <array>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    std::optional<MPISessionOwner::SharedDirectory> discoverMPISharedDirectory(MPI_Comm participants)
    {
        if(participants==MPI_COMM_NULL)throw std::invalid_argument("Owned shared backing requires its actual participant communicator");
        MPI_Comm shared=MPI_COMM_NULL;
        if(MPI_Comm_split_type(participants,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&shared)!=MPI_SUCCESS)
            throw std::runtime_error("Cannot prove node-local owned shared-backing participants");
        int size=0,local_size=0;
        const bool valid=MPI_Comm_size(participants,&size)==MPI_SUCCESS && MPI_Comm_size(shared,&local_size)==MPI_SUCCESS;
        const int free_status=MPI_Comm_free(&shared);
        if(!valid || free_status!=MPI_SUCCESS || size!=local_size)
            throw std::runtime_error("Owned shared backing participants do not share one physical node");
        std::optional<MPISessionOwner::SharedDirectory> directory;
        int failed=0;
        std::string failure;
        try { directory=MPISessionOwner::sharedDirectory(); }
        catch(const std::exception &error) { failed=1;failure=error.what(); }
        int any_failed=0;
        if(MPI_Allreduce(&failed,&any_failed,1,MPI_INT,MPI_MAX,participants)!=MPI_SUCCESS)
            throw std::runtime_error("Cannot agree owned shared-backing startup authentication");
        if(any_failed)throw std::runtime_error("Owned shared backing startup authentication failed"+(failure.empty()?std::string():": "+failure));
        int available=directory?1:0,minimum=0,maximum=0;
        if(MPI_Allreduce(&available,&minimum,1,MPI_INT,MPI_MIN,participants)!=MPI_SUCCESS ||
           MPI_Allreduce(&available,&maximum,1,MPI_INT,MPI_MAX,participants)!=MPI_SUCCESS)
            throw std::runtime_error("Cannot agree owned shared-backing availability");
        if(!maximum)return std::nullopt;
        if(!minimum)throw std::runtime_error("Owned shared backing has mixed owned/unowned participants; wrap external mpirun with llaminar_mpi_session");
        const auto actual=directory->identity();
        std::array<std::uint64_t,3> original{actual.device,actual.inode,actual.uid};
        if(MPI_Bcast(original.data(),original.size(),MPI_UINT64_T,0,participants)!=MPI_SUCCESS)
            throw std::runtime_error("Cannot publish original owned shared-directory identity");
        int mismatch=original!=std::array<std::uint64_t,3>{actual.device,actual.inode,actual.uid};
        int any_mismatch=0;
        if(MPI_Allreduce(&mismatch,&any_mismatch,1,MPI_INT,MPI_MAX,participants)!=MPI_SUCCESS || any_mismatch)
            throw std::runtime_error("Owned shared backing participants have different original directories");
        return directory;
    }
}
