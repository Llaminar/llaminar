/**
 * @file MPISessionProbe.cpp
 * @brief Real MPI initialization, communication and file observations for owner replay.
 *
 * The probe links only MPI, the CPU-only owner, and the C++ runtime. It creates known owned artifacts,
 * reports actual mapped MPI backing, and leaves retirement to the production
 * process owner. Its small collective proves that relocating runtime files did
 * not merely make an inert launcher return success.
 */
#include "utils/MPISessionOwner.h"
#include "utils/MPISessionEnvironment.h"
#include <chrono>
#include <thread>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

/** @brief Run one exact two-rank CPU transaction, then finalize, fail or await interruption. */
int main(int argc, char **argv)
{
    if (auto terminal=llaminar2::MPISessionOwner::enterProcess(argc,argv))return *terminal;
    if (argc != 3) return 125;
    const std::filesystem::path observations = argv[1];
    const std::string mode = argv[2];
    if (mode != "success" && mode != "failure" && mode != "wait" && mode != "inspect") return 125;
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS ||
        provided < MPI_THREAD_MULTIPLE) return 126;
    int rank = -1, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int value = rank + 1;
    int sum = 0;
    if (MPI_Allreduce(&value, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS ||
        sum != size * (size + 1) / 2) MPI_Abort(MPI_COMM_WORLD, 126);
    const char *temporary = std::getenv("TMPDIR");
    const char *shared = std::getenv("OMPI_MCA_btl_vader_backing_directory");
    if (!temporary || !shared) MPI_Abort(MPI_COMM_WORLD, 125);
    std::ofstream(std::filesystem::path(temporary) / ("rank-" + std::to_string(rank))) << "owned tmp";
    std::ofstream(std::filesystem::path(shared) / ("rank-" + std::to_string(rank))) << "owned shm";
    nlohmann::json mappings = nlohmann::json::array();
    std::ifstream maps("/proc/self/maps");
    for (std::string line; std::getline(maps, line);)
        if (line.find("pmix-gds-shmem.") != std::string::npos ||
            line.find("vader_segment.") != std::string::npos)
            mappings.push_back(line.substr(line.find('/')));
    nlohmann::json owned_bindings=nlohmann::json::object();
    for(const auto &binding:llaminar2::mpi_session::kLocationBindings)
        owned_bindings[binding.name]=std::getenv(binding.name)?std::getenv(binding.name):"";
    const auto report = observations / ("rank-" + std::to_string(rank) + ".json");
    const auto pending = report.string() + ".pending";
    std::ofstream output(pending);
    output << nlohmann::json{{"pid", ::getpid()}, {"rank", rank}, {"size", size},
                            {"sum", sum}, {"temporary", temporary}, {"shared", shared},
                            {"mapped_backing", mappings}, {"owned_bindings",owned_bindings}}.dump() << '\n';
    output.close();
    if (!output) MPI_Abort(MPI_COMM_WORLD, 125);
    std::filesystem::rename(pending, report);
    if(mode=="inspect") {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        while(!std::filesystem::exists(observations/"release")) {
            if(std::chrono::steady_clock::now()>=deadline)MPI_Abort(MPI_COMM_WORLD,125);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    if (mode == "wait") for (;;) ::pause();
    if (MPI_Finalize() != MPI_SUCCESS) return 126;
    return mode == "failure" ? 7 : 0;
}
