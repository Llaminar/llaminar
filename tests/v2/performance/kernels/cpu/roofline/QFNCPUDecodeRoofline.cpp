/**
 * @file QFNCPUDecodeRoofline.cpp
 * @brief Measure pinned x86 read and VNNI ceilings for one NUMA/cache domain.
 * @details The affinity mask supplies one hardware thread per physical core.
 *          The same pinned workers first-touch and measure every read segment.
 *          Timed bodies use full cache-line loads or independent register dots.
 *          Geometry, page-placement verification and output are outside timing.
 */

#include <immintrin.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <latch>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace
{
    using Clock = std::chrono::steady_clock;
    namespace fs = std::filesystem;

    constexpr size_t kLineBytes = 64;
    constexpr size_t kDotChains = 12;
    constexpr size_t kSamples = 9;
    constexpr auto kSampleTarget = std::chrono::milliseconds(30);

    enum class Mode { L2, SharedCache, Dram, Vnni };

    struct Placement
    {
        int socket = -1;
        int node = -1;
        size_t l2_bytes = 0;
        size_t llc_bytes = 0;
        std::vector<int> cpus;
    };

    struct Samples
    {
        std::array<double, kSamples> seconds;
        size_t repetitions;
    };

    /** @param path Required Linux topology attribute. @return Its first token. */
    std::string attribute(const fs::path &path)
    {
        std::string result;
        if (!(std::ifstream(path) >> result))
            throw std::runtime_error("Cannot read " + path.string());
        return result;
    }

    /** @param directory Data-cache sysfs directory. @return Capacity in bytes. */
    size_t cacheSize(const fs::path &directory)
    {
        const auto size = attribute(directory / "size");
        if (size.ends_with('K')) return std::stoull(size) * 1024;
        if (size.ends_with('M')) return std::stoull(size) * 1024 * 1024;
        throw std::runtime_error("Unrecognized cache size " + size);
    }

    /** @return Distinct physical cores and their common socket, NUMA node and shared cache. */
    Placement observePlacement()
    {
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
            throw std::runtime_error("Cannot read the process CPU affinity mask");

        Placement result;
        std::set<std::string> cores;
        std::string shared_domain;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        {
            if (!CPU_ISSET(cpu, &allowed)) continue;
            const fs::path directory = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
            const int socket = std::stoi(attribute(directory / "topology/physical_package_id"));
            int node = -1;
            for (const auto &entry : fs::directory_iterator(directory))
            {
                const auto name = entry.path().filename().string();
                if (name.starts_with("node") && name.size() > 4) node = std::stoi(name.substr(4));
            }
            const auto siblings = attribute(directory / "topology/thread_siblings_list");
            if (!cores.insert(siblings).second)
                throw std::runtime_error("Select one hardware thread per physical core in the CPU affinity mask");

            size_t l2 = 0, llc = 0;
            int last_level = 0;
            std::string domain;
            for (const auto &entry : fs::directory_iterator(directory / "cache"))
            {
                if (!entry.path().filename().string().starts_with("index")) continue;
                const int level = std::stoi(attribute(entry.path() / "level"));
                if (attribute(entry.path() / "type") == "Instruction") continue;
                if (level == 2) l2 = cacheSize(entry.path());
                if (level <= last_level) continue;
                last_level = level;
                llc = cacheSize(entry.path());
                domain = attribute(entry.path() / "shared_cpu_list");
            }
            if (node < 0 || l2 == 0 || last_level <= 2)
                throw std::runtime_error("The selected cores need observable NUMA, L2 and shared-cache geometry");
            if (result.cpus.empty())
            {
                result = {socket, node, l2, llc, {}};
                shared_domain = domain;
            }
            if (socket != result.socket || node != result.node || domain != shared_domain ||
                l2 != result.l2_bytes || llc != result.llc_bytes)
                throw std::runtime_error("Select cores from one NUMA node and one shared-cache domain");
            result.cpus.push_back(cpu);
        }
        if (result.cpus.empty()) throw std::runtime_error("No CPUs in the process affinity mask");
        return result;
    }

    /** @brief Own a fresh mapping with guard pages so numa_maps identifies its exact extent. */
    class ReadMapping
    {
    public:
        /** @brief Reserve untouched pages. @param bytes Live writable extent. @param page Page size in bytes. */
        ReadMapping(size_t bytes, size_t page) : page_(page), bytes_(bytes)
        {
            if (bytes_ == 0) return;
            base_ = mmap(nullptr, bytes_ + 2 * page_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (base_ == MAP_FAILED) throw std::runtime_error("Cannot map the read working set");
            if (mprotect(data(), bytes_, PROT_READ | PROT_WRITE) == 0) return;
            munmap(base_, bytes_ + 2 * page_);
            throw std::runtime_error("Cannot enable read working set access");
        }
        /** @brief Release only this probe's guarded mapping. */
        ~ReadMapping() { if (bytes_) munmap(base_, bytes_ + 2 * page_); }
        ReadMapping(const ReadMapping &) = delete;
        ReadMapping &operator=(const ReadMapping &) = delete;

        /** @return The live page-aligned read working set. */
        uint64_t *data() const
        {
            return reinterpret_cast<uint64_t *>(static_cast<char *>(base_) + page_);
        }

        /** @brief Require all live pages to reside on the observed worker node. @param node Local NUMA node. */
        void verifyNode(int node) const
        {
            if (bytes_ == 0) return;
            std::ifstream maps("/proc/self/numa_maps");
            std::ostringstream address;
            address << std::hex << reinterpret_cast<uintptr_t>(data());
            const std::string expected = "N" + std::to_string(node) + '=';
            for (std::string line; std::getline(maps, line);)
            {
                if (!line.starts_with(address.str() + ' ')) continue;
                size_t local_pages = 0;
                std::istringstream fields(line);
                for (std::string field; fields >> field;)
                {
                    if (field.starts_with(expected)) local_pages = std::stoull(field.substr(expected.size()));
                    else if (field.starts_with('N') && field.find('=') != std::string::npos)
                        throw std::runtime_error("Read working set has remote NUMA pages: " + line);
                }
                if (local_pages == bytes_ / page_) return;
                throw std::runtime_error("Read working set is not fully resident: " + line);
            }
            throw std::runtime_error("Cannot find the read working set in /proc/self/numa_maps");
        }

    private:
        void *base_ = nullptr;
        size_t page_;
        size_t bytes_;
    };

    /**
     * @brief Scan aligned cache lines with four independent read/accumulate chains.
     * @param data Pinned worker's live, cache-line-aligned segment.
     * @param bytes Complete page-aligned segment size.
     * @param repetitions Full scans of the segment.
     * @return Sum of all read values across all passes.
     */
    __attribute__((noinline, target("avx512f")))
    uint64_t readLines(const uint64_t *data, size_t bytes, size_t repetitions)
    {
        auto a = _mm512_setzero_si512(), b = a, c = a, d = a;
        for (size_t pass = 0; pass < repetitions; ++pass)
        {
            // Keep every pass observable to the optimizer without a machine instruction.
            asm volatile("" : : "r"(data) : "memory");
            for (size_t index = 0; index < bytes / sizeof(uint64_t); index += 32)
            {
                a = _mm512_add_epi64(a, _mm512_load_si512(data + index));
                b = _mm512_add_epi64(b, _mm512_load_si512(data + index + 8));
                c = _mm512_add_epi64(c, _mm512_load_si512(data + index + 16));
                d = _mm512_add_epi64(d, _mm512_load_si512(data + index + 24));
            }
        }
        return _mm512_reduce_add_epi64(_mm512_add_epi64(_mm512_add_epi64(a, b), _mm512_add_epi64(c, d)));
    }

    /**
     * @brief Issue independent packed unsigned-byte by signed-byte dots.
     * @param repetitions Iterations through all the independent chains.
     * @return Whether all lanes contain the expected modular sum.
     */
    __attribute__((noinline, target("avx512f,avx512vnni")))
    bool checkedDots(size_t repetitions)
    {
        auto a = _mm512_set1_epi32(0x02010201);
        auto b = _mm512_set1_epi32(0x01ff0101);
        __m512i acc[kDotChains];
        for (size_t chain = 0; chain < kDotChains; ++chain) acc[chain] = _mm512_set1_epi32(chain);
        asm volatile("" : "+v"(a), "+v"(b));
        for (size_t pass = 0; pass < repetitions; ++pass)
        {
#pragma GCC unroll 12
            for (size_t chain = 0; chain < kDotChains; ++chain)
                acc[chain] = _mm512_dpbusd_epi32(acc[chain], a, b);
        }
        auto sum = _mm512_setzero_si512();
        for (const auto &value : acc) sum = _mm512_add_epi32(sum, value);
        // Each lane adds 1*1 + 2*1 + 1*(-1) + 2*1 per instruction.
        const uint32_t expect = 66 + 4 * kDotChains * repetitions;
        return _mm512_cmpeq_epi32_mask(sum, _mm512_set1_epi32(expect)) == 0xffff;
    }

    /**
     * @param mode Selected memory or compute reference.
     * @param placement Observed worker and cache geometry.
     * @param page System page size in bytes.
     * @return Page-aligned read working set per worker, zero for the dot probe.
     */
    size_t bytesPerWorker(Mode mode, const Placement &placement, size_t page)
    {
        switch (mode)
        {
        case Mode::L2:
            return placement.l2_bytes / 2 / page * page;
        case Mode::SharedCache:
        {
            const size_t bytes = placement.llc_bytes * 7 / 8 / placement.cpus.size() / page * page;
            if (bytes <= placement.l2_bytes)
                throw std::runtime_error("A shared-cache footprint above private L2 needs fewer workers here");
            return bytes;
        }
        case Mode::Dram:
            return 4 * (placement.llc_bytes / placement.cpus.size() + placement.l2_bytes) / page * page;
        case Mode::Vnni:
            return 0;
        }
        std::abort();
    }

    /**
     * @brief Keep one pinned team for warmup and sampling.
     * @param mode Selected memory or compute reference.
     * @param placement CPUs and observed NUMA node.
     * @param bytes Read working-set size per worker.
     * @param page System page size in bytes.
     * @return Checksum-verified sustained team samples.
     */
    Samples measure(Mode mode, const Placement &placement, size_t bytes, size_t page)
    {
        ReadMapping mapping(bytes * placement.cpus.size(), page);
        Clock::time_point boundary;
        std::barrier synchronize(static_cast<ptrdiff_t>(placement.cpus.size()), [&]() noexcept {
            boundary = Clock::now();
        });
        Samples samples{};
        std::latch team_ready(1);
        std::vector<std::jthread> workers;
        workers.reserve(placement.cpus.size());
        const auto work = [&](std::stop_token stop, size_t worker) {
            team_ready.wait();
            if (stop.stop_requested()) return;
            cpu_set_t affinity;
            CPU_ZERO(&affinity);
            CPU_SET(placement.cpus[worker], &affinity);
            if (sched_setaffinity(0, sizeof(affinity), &affinity) != 0)
            {
                std::cerr << "Cannot pin worker to CPU " << placement.cpus[worker] << '\n';
                std::abort();
            }
            uint64_t *data = bytes ? mapping.data() + worker * bytes / sizeof(uint64_t) : nullptr;
            for (size_t index = 0; index < bytes / sizeof(uint64_t); ++index) data[index] = index % 8 + 1;
            bool calibrated = false;
            for (size_t repeats = 1, sample = 0; sample < kSamples;)
            {
                synchronize.arrive_and_wait();
                const auto start = boundary;
                const bool checked = mode == Mode::Vnni ? checkedDots(repeats) :
                    readLines(data, bytes, repeats) == repeats * (bytes / kLineBytes) * 36;
                synchronize.arrive_and_wait();
                const auto elapsed = boundary - start;
                if (!checked)
                {
                    std::cerr << "Arithmetic or read checksum mismatch on worker " << worker << '\n';
                    std::abort();
                }
                if (calibrated)
                {
                    if (worker == 0) samples.seconds[sample] = std::chrono::duration<double>(elapsed).count();
                    ++sample;
                }
                else if (elapsed >= kSampleTarget)
                {
                    if (worker == 0) samples.repetitions = repeats;
                    calibrated = true;
                }
                else
                {
                    repeats *= 2;
                }
            }
        };
        try
        {
            for (size_t worker = 0; worker < placement.cpus.size(); ++worker)
                workers.emplace_back(work, worker);
        }
        catch (...)
        {
            // An incomplete team must exit without entering its fixed-width barriers.
            for (auto &worker : workers) worker.request_stop();
            team_ready.count_down();
            throw;
        }
        team_ready.count_down();
        for (auto &worker : workers) worker.join();
        mapping.verifyNode(placement.node);
        return samples;
    }
}

/**
 * @brief Run one explicit hardware reference and emit its physical placement and raw samples.
 * @param argc Argument count including the program name.
 * @param argv Program name and one reference mode.
 * @return Zero for a complete verified measurement.
 */
int main(int argc, char **argv)
{
    try
    {
        constexpr std::array<std::string_view, 4> names{"l2", "shared", "dram", "vnni"};
        if (argc != 2) throw std::runtime_error("Usage: v2_perf_qfn_cpu_decode_roofline {l2|shared|dram|vnni}");
        size_t index = 0;
        while (index < names.size() && names[index] != argv[1]) ++index;
        if (index == names.size()) throw std::runtime_error("Unknown roofline mode");
        const auto mode = static_cast<Mode>(index);
        if (!__builtin_cpu_supports("avx512f") || (mode == Mode::Vnni && !__builtin_cpu_supports("avx512vnni")))
            throw std::runtime_error("This reference requires AVX-512F and the dot mode requires AVX-512 VNNI");
        const auto placement = observePlacement();
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0 || page % (4 * kLineBytes) != 0) throw std::runtime_error("Invalid page geometry");
        const size_t bytes = bytesPerWorker(mode, placement, page);
        if (mode != Mode::Vnni && bytes == 0) throw std::runtime_error("Empty read working set");
        std::cout << "# mode=" << names[index] << " socket=" << placement.socket << " node=" << placement.node
                  << " workers=" << placement.cpus.size() << " l2_bytes=" << placement.l2_bytes
                  << " llc_bytes=" << placement.llc_bytes << " bytes_per_worker=" << bytes << " cpus=";
        for (const int cpu : placement.cpus) std::cout << ' ' << cpu;
        std::cout << std::endl;
        const auto samples = measure(mode, placement, bytes, page);
        const size_t work = placement.cpus.size() * samples.repetitions * (mode == Mode::Vnni ? kDotChains : bytes);
        std::cout << "sample,repetitions,seconds," << (mode == Mode::Vnni ? "zmm_vpdpbusd" : "read_bytes") << ",giga_per_second\n";
        for (size_t sample = 0; sample < kSamples; ++sample)
        {
            const double seconds = samples.seconds[sample];
            std::cout << sample << ',' << samples.repetitions << ',' << std::setprecision(9) << seconds
                      << ',' << work << ',' << work / seconds / 1e9 << '\n';
        }
        std::cout << "# checksum=pass" << (mode == Mode::Vnni ? "" : " page_placement=local") << '\n';
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
