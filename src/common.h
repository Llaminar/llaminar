#pragma once

#include <string>
#include <vector>
#include <map>

// CPU topology structure for system hardware detection
struct CPUTopology
{
    int total_cpus;
    int physical_cores;
    int sockets;
    int cores_per_socket;
    int threads_per_core;
    bool hyperthreading_detected;
    bool use_hyperthreading;
    std::vector<int> physical_core_ids;
    std::map<int, int> cpu_to_socket;
    std::map<int, int> cpu_to_physical_core;
    std::map<int, std::vector<int>> socket_to_cpus;
    std::map<int, std::vector<int>> socket_to_physical_cores;
    std::map<int, std::vector<int>> socket_to_primary_cpus; // socket_id -> primary CPU per core (no HT siblings)
};

// Test case structure for benchmarks
struct TestCase
{
    std::string name;
    int m, n, k;

    TestCase(const std::string &name, int m, int n, int k)
        : name(name), m(m), n(n), k(k) {}
};

// Benchmark result structure
struct BenchmarkResult
{
    std::string test_name;
    double time_ms;
    double gflops;
    double memory_gb;
    double bandwidth_gb_s;
    int m, n, k;   // Matrix dimensions
    int num_procs; // Number of MPI processes
    bool success;  // Whether the benchmark succeeded

    BenchmarkResult() : time_ms(0), gflops(0), memory_gb(0), bandwidth_gb_s(0),
                        m(0), n(0), k(0), num_procs(0), success(false) {}
};

// Forward declarations
class KernelManager;
class ModelLoader;
class TensorRepacker;