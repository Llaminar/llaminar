#include "../src/SystemTopology.h"
#include "../src/ArgumentParser.h"
#include <mpi.h>
#include <sched.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

// Get current NUMA node
int getCurrentNumaNode()
{
    int current_cpu = sched_getcpu();
    if (current_cpu < 0)
        return -1;

    // Read NUMA node from /sys/devices/system/cpu/cpuX/node
    std::string node_path = "/sys/devices/system/cpu/cpu" + std::to_string(current_cpu) + "/node";
    std::ifstream node_file(node_path);

    if (node_file.is_open())
    {
        int node;
        if (node_file >> node)
        {
            return node;
        }
    }
    return -1;
}

// NUMA-aware MPI test
void runNumaTest(int rank, int size)
{
    int numa_node = getCurrentNumaNode();
    int current_cpu = sched_getcpu();

    std::cout << "Process " << rank << "/" << size
              << " running on CPU " << current_cpu
              << " (NUMA node " << numa_node << ")" << std::endl;

    // Collect NUMA distribution information
    std::vector<int> numa_nodes(size);
    MPI_Allgather(&numa_node, 1, MPI_INT, numa_nodes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Count processes per NUMA node
        std::vector<int> numa_counts(8, 0); // Support up to 8 NUMA nodes
        int unique_numa_nodes = 0;

        for (int i = 0; i < size; ++i)
        {
            if (numa_nodes[i] >= 0 && numa_nodes[i] < 8)
            {
                if (numa_counts[numa_nodes[i]] == 0)
                {
                    unique_numa_nodes++;
                }
                numa_counts[numa_nodes[i]]++;
            }
        }

        std::cout << "\n=== NUMA Distribution Summary ===" << std::endl;
        std::cout << "Total MPI processes: " << size << std::endl;
        std::cout << "NUMA nodes utilized: " << unique_numa_nodes << std::endl;

        for (int i = 0; i < 8; ++i)
        {
            if (numa_counts[i] > 0)
            {
                std::cout << "NUMA node " << i << ": " << numa_counts[i] << " processes" << std::endl;
            }
        }

        // Check if we have good NUMA distribution
        bool good_distribution = (unique_numa_nodes >= 2) && (size >= unique_numa_nodes);
        if (good_distribution)
        {
            std::cout << "\n✓ NUMA MPI TEST SUCCESS: Good distribution across NUMA nodes" << std::endl;
        }
        else
        {
            std::cout << "\n⚠ NUMA MPI TEST WARNING: Suboptimal NUMA distribution" << std::endl;
        }
        std::cout << "===================================" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse arguments
    ArgumentParser parser(argc, argv);
    LlaminarParams params;

    if (!parser.parse(params))
    {
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        std::cout << "\n=== NUMA Topology Test ===" << std::endl;
        std::cout << "Testing NUMA awareness and process distribution" << std::endl;
    }

    runNumaTest(rank, size);

    MPI_Finalize();
    return 0;
}