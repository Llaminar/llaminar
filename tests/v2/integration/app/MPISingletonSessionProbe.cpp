/**
 * @file MPISingletonSessionProbe.cpp
 * @brief Actual production MPIProcessSession file lifetime without a model or GPU.
 *
 * The canonical target links the production core and uses its actual shutdown
 * boundary. The shared Python lifecycle regression owns this probe's process,
 * observations and explicit success/error/signal cases. An external MPI parent
 * retains ownership. A singleton runs below the same ancestor owner, whose
 * roots survive finalization until every actual frontend/daemon retires.
 *
 * The custom entry owns one lexical native GPU lifetime without probing a
 * device. Native workers/caches retire before process exit and MPI finalization.
 */
#include "backends/GPUDeviceProcessLifetime.h"
#include "utils/MPISessionOwner.h"
#include "app/MPIProcessSession.h"
#include <mpi.h>
#include <nlohmann/json.hpp>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

/** @brief Emit bounded original directory observations outside the owned roots. */
static void observe(const std::filesystem::path &path, const char *mode,
                    const std::string &temporary, const std::string &shared, const std::string &program)
{
    const auto pending=path.string()+".pending";
    std::ofstream output(pending);
    output << nlohmann::json{{"pid",::getpid()},{"mode",mode},{"temporary",temporary},
        {"shared",shared},{"argv0",program}}.dump() << '\n';
    output.close();
    if (!output) throw std::runtime_error("Cannot publish singleton observation");
    std::filesystem::rename(pending,path);
}

/** @brief Exercise actual production scope order and return the exact requested outcome. */
int main(int argc, char **argv)
{
    // Ownership starts before this frontend creates MPI or runtime state.
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv)) return *terminal;
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    if (argc != 4 || std::string(argv[1]) != "--observation") return 125;
    const std::filesystem::path observation = argv[2];
    const std::string mode = argv[3];
    if (mode != "success" && mode != "throw" && mode != "wait" && mode != "external" && mode != "inspect") return 125;
    try {
        std::string temporary, shared;
        {
            auto session = llaminar2::MPIProcessSession::initialize(argc,argv);
            const char *tmp = std::getenv("TMPDIR");
            const char *shm = std::getenv("OMPI_MCA_btl_vader_backing_directory");
            if (!tmp || !shm) throw std::runtime_error("Missing singleton owned runtime roots");
            temporary = tmp; shared = shm;
            std::ofstream(temporary + "/probe-leftover") << "owned temporary bytes";
            std::ofstream(shared + "/probe-leftover") << "owned shared bytes";
            int rank=0,world=0;
            if(MPI_Comm_rank(MPI_COMM_WORLD,&rank)!=MPI_SUCCESS || MPI_Comm_size(MPI_COMM_WORLD,&world)!=MPI_SUCCESS)
                throw std::runtime_error("Cannot observe actual MPI scope membership");
            const auto rank_observation=world>1 ? std::filesystem::path(observation.string()+".rank-"+std::to_string(rank)) : observation;
            observe(rank_observation,mode.c_str(),temporary,shared,argv[0]);
            if (mode == "throw") throw std::runtime_error("requested scoped MPI failure");
            if (mode == "wait") for (;;) ::pause();
            if (mode == "inspect") {
                for (int poll=0;poll<5000 && !std::filesystem::exists(rank_observation.string()+".release");++poll) ::usleep(1000);
                if (!std::filesystem::exists(rank_observation.string()+".release")) throw std::runtime_error("Singleton ancestry inspection timed out");
            }
        }
        int finalized = 0;
        if (MPI_Finalized(&finalized) != MPI_SUCCESS || !finalized) return 126;
        const bool temporary_live = std::filesystem::exists(temporary), shared_live = std::filesystem::exists(shared);
        // Finalization does not retire the process. Its ancestor keeps both
        // roots until this actual frontend and any singleton daemon exit.
        if (!temporary_live || !shared_live) return 126;
        return 0;
    } catch (const std::exception &) {
        int finalized = 0;
        return MPI_Finalized(&finalized) == MPI_SUCCESS && finalized && mode == "throw" ? 7 : 125;
    }
}
