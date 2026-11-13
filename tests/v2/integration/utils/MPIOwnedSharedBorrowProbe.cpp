/**
 * @file MPIOwnedSharedBorrowProbe.cpp
 * @brief Actual owner capability replay across a deterministic pathname replacement.
 *
 * The production getter and its normal startup proof are linked unchanged.
 * A test-only preload moves this run's exact directory between authentication
 * and adoption, then restores it before process retirement. Native mode runs
 * before MPI_Init, exactly where rank publication must already be sufficient.
 */
#include "utils/MPISessionOwner.h"
#include <dlfcn.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
using llaminar2::MPISessionOwner;
/** @brief Return success only when the actual getter rejects the injected foreign inode. */
int main(int argc,char **argv)
{
    if(auto terminal=MPISessionOwner::enterProcess(argc,argv))return *terminal;
    if(argc!=4)return 125;
    const int mode=std::string(argv[1])=="managed"?1:std::string(argv[1])=="native"?2:0;
    if(!mode || std::string(argv[2])!="--observation")return 125;
    const auto arm=reinterpret_cast<void(*)(int)>(::dlsym(RTLD_DEFAULT,"llaminarTestSharedBorrowArm"));
    const auto swapped=reinterpret_cast<int(*)()>(::dlsym(RTLD_DEFAULT,"llaminarTestSharedBorrowSwapped"));
    const auto restore=reinterpret_cast<int(*)()>(::dlsym(RTLD_DEFAULT,"llaminarTestSharedBorrowRestore"));
    if(!arm || !swapped || !restore)return 125;
    auto before=MPISessionOwner::sharedDirectory();if(!before)return 125;
    const auto original=before->identity();bool rejected=false;std::uint64_t observed=0;std::string error;
    arm(mode);
    try {auto result=MPISessionOwner::sharedDirectory();if(result)observed=result->identity().inode;}
    catch(const std::exception &failure) {rejected=true;error=failure.what();}
    const bool injected=swapped()!=0;const bool restored=restore()==0;
    const std::filesystem::path output=argv[3],pending=output.string()+".pending";
    std::ofstream stream(pending);stream<<nlohmann::json{{"rejected",rejected},{"injected",injected},{"restored",restored},
        {"original_inode",original.inode},{"observed_inode",observed},{"error",error}}.dump()<<'\n';
    stream.close();if(!stream)return 126;std::filesystem::rename(pending,output);
    return rejected && injected && restored?0:42;
}
