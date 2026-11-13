/**
 * @file MPIOwnedSharedBorrowRepeat.cpp
 * @brief Exercise ordinary native-rank directory descriptors after startup exec.
 *
 * Native MPI has already consumed/retired the process-entry FD loans. A later
 * legitimate app-owned directory FD must not be mistaken for a broken loan.
 */
#include "utils/MPISessionOwner.h"
#include <cerrno>
#include <cstdio>
#include <exception>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char** argv)
{
    if (auto status=llaminar2::MPISessionOwner::enterProcess(argc,argv)) return *status;
    auto original=llaminar2::MPISessionOwner::sharedDirectory();
    if (!original) return 2;
    // The native-rank startup probe intentionally runs before MPI_Init, so this
    // is an available descriptor, not a descriptor borrowed from MPI itself.
    if(argc!=2)return 125;
    const int slot=std::string(argv[1])=="6"?6:std::string(argv[1])=="7"?7:0;
    if(!slot)return 125;
    if (fcntl(slot,F_GETFD)!=-1 || errno!=EBADF) return 3;
    const int opened=open(original->path().c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC);
    if (opened<0) return 4;
    const int ordinary=fcntl(opened,F_DUPFD_CLOEXEC,slot);
    close(opened);
    if (ordinary!=slot) { if(ordinary>=0)close(ordinary); return 5; }
    try {
        for(int iteration=0;iteration<8;++iteration) {
            auto second=llaminar2::MPISessionOwner::sharedDirectory();
            if (!second || !(second->identity()==original->identity())) return 6;
        }
        close(ordinary);
        std::printf("{\"repeat_directory_borrow\":true,\"ordinary_fd\":%d}\n",slot);
        return 0;
    } catch(const std::exception& error) {
        close(ordinary);
        std::fprintf(stderr,"Valid native directory repeat rejected: %s\n",error.what());
        return 42;
    }
}
