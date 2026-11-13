/**
 * @file rccl_p2p_selection_probe.cpp
 * @brief Run the actual native P2P selector against authored topology and environment inputs.
 *
 * Real paths.cc and topo.cc provide the selector, level parser, rank lookup and
 * CPU query. Each invocation is a fresh process because native user policy is
 * intentionally cached. Probe-local logging/environment seams avoid config-file
 * contamination and runtime initialization; neither substitutes policy logic.
 */
#include "core.h"
#include "graph.h"
#include "topo.h"
#include "debug.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <utility>

/** @brief Native warning-suppression state for the probe-local logging boundary. */
thread_local int ncclDebugNoWarn = 0;

/** @brief Read only the explicit probe environment. @param name Native key. @return Borrowed value or null. */
const char* ncclGetEnv(const char* name)
{
    return std::getenv(name);
}

/**
 * @brief Keep native diagnostics host-only; no diagnostic is used as the policy oracle.
 * @param level Native severity. @param flags Native subsystem mask.
 * @param filefunc Native source label. @param line Native source line.
 * @param fmt Native format, followed by its native variadic arguments.
 * @return No value; selection/status fields below are the observed contract.
 */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags,
                  const char* filefunc, int line, const char* fmt, ...)
{
    (void)level;
    (void)flags;
    (void)filefunc;
    (void)line;
    (void)fmt;
}

/**
 * @brief Check all CPU defaults, user path boundaries and unrepresented peers.
 * @param argc Argument count, exactly two. @param argv Expected level or -1 for CPU default.
 * @return Zero on complete success, one on policy mismatch, two on malformed probe input.
 */
int main(int argc, char** argv)
{
    if (argc != 2)
    {
        return 2;
    }
    char* end = nullptr;
    const long selected = std::strtol(argv[1], &end, 10);
    if (*end != '\0' || selected < -1 || selected > PATH_SYS)
    {
        return 2;
    }
    constexpr std::array<std::pair<int, int>, 5> cpus{{
        {NCCL_TOPO_CPU_ARCH_X86, NCCL_TOPO_CPU_VENDOR_INTEL},
        {NCCL_TOPO_CPU_ARCH_X86, NCCL_TOPO_CPU_VENDOR_AMD},
        {NCCL_TOPO_CPU_ARCH_X86, NCCL_TOPO_CPU_VENDOR_ZHAOXIN},
        {NCCL_TOPO_CPU_ARCH_ARM, NCCL_TOPO_CPU_VENDOR_MIXED},
        {NCCL_TOPO_CPU_ARCH_POWER, NCCL_TOPO_CPU_VENDOR_MIXED}
    }};
    auto system = std::make_unique<ncclTopoSystem>();
    system->nodes[GPU].count = 2;
    system->nodes[CPU].count = 1;
    system->nodes[GPU].nodes[0].gpu.rank = 0;
    system->nodes[GPU].nodes[1].gpu.rank = 1;
    system->nodes[GPU].nodes[0].gpu.cudaCompCap = 80;
    system->nodes[GPU].nodes[1].gpu.cudaCompCap = 80;
    std::array<ncclTopoLinkList, 2> paths{};
    system->nodes[GPU].nodes[0].paths[GPU] = paths.data();
    int checks = 0;
    int failures = 0;
    auto check = [&](bool ok) {
        ++checks;
        if (!ok)
        {
            ++failures;
        }
    };
    for (const auto& [arch, vendor] : cpus)
    {
        system->nodes[CPU].nodes[0].cpu.arch = arch;
        system->nodes[CPU].nodes[0].cpu.vendor = vendor;
        const bool restricted = arch == NCCL_TOPO_CPU_ARCH_ARM ||
            (arch == NCCL_TOPO_CPU_ARCH_X86 &&
             (vendor == NCCL_TOPO_CPU_VENDOR_INTEL || vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN));
        const int expectedLevel = selected >= 0 ? static_cast<int>(selected) :
            (restricted ? PATH_PXB : PATH_SYS);
        for (int path = PATH_LOC; path <= PATH_NET; ++path)
        {
            paths[1].type = path;
            int p2p = -1;
            int read = -1;
            int intermediate = -2;
            check(ncclTopoCheckP2p(nullptr, system.get(), 0, 1, &p2p, &read, &intermediate) == ncclSuccess);
            check(p2p == (path <= expectedLevel));
            check(read == (path == PATH_NVL));
            check(intermediate == -1);
        }
        int p2p = -1;
        int read = -1;
        int intermediate = -2;
        check(ncclTopoCheckP2p(nullptr, system.get(), 0, 2, &p2p, &read, &intermediate) == ncclSuccess);
        check(p2p == 0);
        check(read == 0);
        check(intermediate == -1);
    }
    std::printf("{\"checks\":%d,\"failures\":%d}\n", checks, failures);
    return failures ? 1 : 0;
}
