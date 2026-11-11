/**
 * @file NativeCPUPlacement.cpp
 * @brief Initial-entry native OpenMP place ownership shared with the process guardian.
 *
 * Only authenticated original main-thread entry can publish an initial exec.
 * Native place tables, not thread-id shadows, own recreated OpenMP workers.
 * Explicit non-core placement remains authored policy; automatic core receipts
 * are checked against independently resolved capacity order on every reentry.
 */
#include "utils/NativeCPUPlacement.h"
#include "utils/ScopedOpenMPPartitionAffinity.h"
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/syscall.h>
#include <unistd.h>

// The CPU-only guardian does not load MPI. A linked application's existing
// native symbol is queried without opening a runtime or initializing MPI.
extern "C" int MPI_Initialized(int *) __attribute__((weak));

namespace llaminar2
{
namespace
{
        using CorePlaces = std::vector<std::vector<int>>;
        // Process-launch protocol owned here, not a live diagnostic/tuning switch.
        constexpr const char *kNativeCorePlacesEnvironment="LLAMINAR_NATIVE_CORE_PLACES";

        /** @return Canonical explicit OpenMP syntax for an already ordered place list. */
        std::string encodeCorePlaces(const CorePlaces &places)
        {
            std::ostringstream output;
            for (std::size_t p=0;p<places.size();++p)
            {
                if(p)output<<',';
                output<<'{';
                for(std::size_t c=0;c<places[p].size();++c)
                {
                    if(c)output<<',';
                    output<<places[p][c];
                }
                output<<'}';
            }
            return output.str();
        }

        /** @return This caller's native partition order, not a guessed global worker roster. */
        CorePlaces nativeCorePlaces()
        {
            const int count=omp_get_partition_num_places();
            if(count<=0)return {};
            std::vector<int> indices(static_cast<std::size_t>(count));
            omp_get_partition_place_nums(indices.data());
            CorePlaces places;
            std::set<int> seen;
            for(const int index:indices)
            {
                const int width=omp_get_place_num_procs(index);
                if(width<=0)throw std::runtime_error("Native OpenMP partition contains an empty place");
                std::vector<int> cpus(static_cast<std::size_t>(width));
                omp_get_place_proc_ids(index,cpus.data());
                std::sort(cpus.begin(),cpus.end());
                for(const int cpu:cpus)
                    if(cpu<0 || cpu>=CPU_SETSIZE || !seen.insert(cpu).second)
                        throw std::runtime_error("Native OpenMP partition has invalid or overlapping CPUs");
                places.push_back(std::move(cpus));
            }
            return places;
        }

        /** @brief One initial-exec receipt; a different process must admit its own rank partition. */
        struct CorePlaceLaunchReceipt
        {
            long process;
            std::string places;
        };

        /**
         * @brief Parse one complete versioned native startup publication.
         * @param value Original publication bytes.
         * @return Parsed process identity and explicit place encoding.
         * @throws std::runtime_error for malformed or unsupported publication.
         */
        CorePlaceLaunchReceipt parseCorePlaceReceipt(std::string_view value)
        {
            const auto separator=value.find(';',3);
            if(!value.starts_with("v1;") || separator==std::string_view::npos || separator+1==value.size())
                throw std::runtime_error("Incompatible native OpenMP place launch receipt");
            long process=0;
            const auto parsed=std::from_chars(value.data()+3,value.data()+separator,process);
            if(parsed.ec!=std::errc{} || parsed.ptr!=value.data()+separator || process<=0)
                throw std::runtime_error("Invalid native OpenMP place launch process identity");
            return CorePlaceLaunchReceipt{process,std::string(value.substr(separator+1))};
        }

        /**
         * @brief Publish exact native startup data before exec or an already-compatible admission.
         * @param places Canonical physical-core place list.
         * @throws std::runtime_error when any environment publication fails.
         */
        void publishCorePlaces(const std::string &places)
        {
            const auto receipt="v1;"+std::to_string(::getpid())+';'+places;
            if(::setenv("OMP_PLACES",places.c_str(),1)!=0 || ::setenv("OMP_PROC_BIND","close",1)!=0 ||
               ::setenv(kNativeCorePlacesEnvironment,receipt.c_str(),1)!=0)
                throw std::runtime_error("Cannot publish native OpenMP place startup: "+std::string(std::strerror(errno)));
        }


    /** @brief Process-entry transitions; none can be reentered after application work. */
    enum class EntryPhase { Unclaimed, Claimed, Ready, OtherPlacement };
    EntryPhase phase=EntryPhase::Unclaimed;
    std::unique_ptr<NativeOpenMPCorePlaces> admitted;

    /** @return Whether every place is one distinct physical core. */
    bool physicalPlaces(const CorePlaces &places)
    {
        if(places.empty())return false;
        std::set<int> cores;
        for(const auto &place:places)
        {
            if(place.empty())return false;
            const int core=NativeCPUPlacement::physicalRepresentativeForCpu(place.front());
            if(!cores.insert(core).second || std::any_of(place.begin(),place.end(),[&](int cpu){
                return NativeCPUPlacement::physicalRepresentativeForCpu(cpu)!=core;
            }))return false;
        }
        return true;
    }

    /** @return The admitted CPUs, independent of their current native place order. */
    std::vector<int> placeCpus(const CorePlaces &places)
    {
        std::vector<int> cpus;
        for(const auto &place:places)cpus.insert(cpus.end(),place.begin(),place.end());
        std::sort(cpus.begin(),cpus.end());
        return cpus;
    }

    /** @brief Reject a process that already owns MPI or a live worker pool. */
    void requireInitialMainThread()
    {
        if(::syscall(SYS_gettid)!=::getpid() || omp_get_level()!=0)
            throw std::runtime_error("Native CPU placement requires original main-thread entry");
        int initialized=0;
        if(MPI_Initialized && (MPI_Initialized(&initialized)!=0 || initialized))
            throw std::runtime_error("Native CPU placement cannot enter an initialized MPI process");
        std::size_t threads=0;
        for(const auto &task:std::filesystem::directory_iterator("/proc/self/task"))
        {
            (void)task;
            ++threads;
        }
        if(threads!=1)
            throw std::runtime_error("Native CPU placement cannot enter a process with existing workers");
    }
}
    std::vector<int> NativeCPUPlacement::parseCpuList(const std::string &cpulist)
    {
        std::vector<int> cpus;
        std::stringstream ss(cpulist);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            if (token.empty())
            {
                continue;
            }
            auto dash = token.find('-');
            if (dash == std::string::npos)
            {
                cpus.push_back(std::stoi(token));
                continue;
            }
            int start = std::stoi(token.substr(0, dash));
            int end = std::stoi(token.substr(dash + 1));
            if (end < start)
            {
                std::swap(start, end);
            }
            for (int cpu = start; cpu <= end; ++cpu)
            {
                cpus.push_back(cpu);
            }
        }
        std::sort(cpus.begin(), cpus.end());
        cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
        return cpus;
    }

    int NativeCPUPlacement::physicalRepresentativeForCpu(int cpu)
    {
        std::ifstream siblings_file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
        if (!siblings_file.is_open())
        {
            return cpu;
        }

        std::string siblings;
        std::getline(siblings_file, siblings);
        auto sibling_cpus = parseCpuList(siblings);
        if (sibling_cpus.empty())
        {
            return cpu;
        }
        return *std::min_element(sibling_cpus.begin(), sibling_cpus.end());
    }

    long NativeCPUPlacement::cpuCapacity(int cpu, const std::string &sysfs_root)
    {
        const std::string root = sysfs_root + "/cpu" + std::to_string(cpu);
        for (const char *leaf : {"/cpu_capacity", "/cpufreq/cpuinfo_max_freq"})
        {
            std::ifstream file(root + leaf);
            long value = 0;
            if (file >> value && value > 0)
                return value;
        }
        return 0;
    }

    std::vector<int> NativeCPUPlacement::capacityOrderedCpus(
        std::vector<int> cpus, const std::string &sysfs_root)
    {
        std::vector<std::pair<long, int>> ranked;
        ranked.reserve(cpus.size());
        for (const int cpu : cpus)
            ranked.emplace_back(cpuCapacity(cpu, sysfs_root), cpu);
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
        for (std::size_t index = 0; index < ranked.size(); ++index)
            cpus[index] = ranked[index].second;
        return cpus;
    }

    std::vector<std::vector<int>> NativeCPUPlacement::capacityOrderedCorePlaces(
        const std::vector<int> &allowed_cpus, const std::string &sysfs_root)
    {
        // Group allowed CPUs by their physical core: the smallest CPU of the
        // core's sibling list identifies it, as in physicalRepresentativeForCpu.
        std::map<int, std::vector<int>> cores;
        for (const int cpu : allowed_cpus)
        {
            int representative = cpu;
            std::ifstream siblings_file(sysfs_root + "/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
            std::string siblings;
            if (siblings_file && std::getline(siblings_file, siblings))
            {
                const auto sibling_cpus = parseCpuList(siblings);
                if (!sibling_cpus.empty())
                    representative = *std::min_element(sibling_cpus.begin(), sibling_cpus.end());
            }
            cores[representative].push_back(cpu);
        }
        // Rank cores by their first allowed sibling, the same CPU the
        // cluster path has always ranked, so the two orders cannot diverge.
        std::vector<int> firsts;
        std::map<int, const std::vector<int> *> by_first;
        for (auto &[representative, cpus] : cores)
        {
            std::sort(cpus.begin(), cpus.end());
            firsts.push_back(cpus.front());
            by_first[cpus.front()] = &cpus;
        }
        std::vector<std::vector<int>> places;
        places.reserve(firsts.size());
        for (const int first : capacityOrderedCpus(std::move(firsts), sysfs_root))
            places.push_back(*by_first.at(first));
        return places;
    }


    void NativeCPUPlacement::validateCapacityReceipt(const std::string &publication,
        const std::vector<std::vector<int>> &native,long process,const std::string &root)
    {
        const auto receipt=parseCorePlaceReceipt(publication);
        const auto cpus=placeCpus(native);
        const std::set<int> unique(cpus.begin(),cpus.end());
        if(native.empty() || unique.size()!=cpus.size() || receipt.process!=process ||
           encodeCorePlaces(native)!=receipt.places || native!=capacityOrderedCorePlaces(cpus,root))
            throw std::runtime_error("Native OpenMP publication differs from canonical observed capacity order");
    }

    void NativeCPUPlacement::claimProcessEntry()
    {
        requireInitialMainThread();
        if(phase!=EntryPhase::Unclaimed)
            throw std::runtime_error("Native CPU process entry was already consumed");
        phase=EntryPhase::Claimed;
    }

    void NativeCPUPlacement::prepareProcessEntry(int argc,char *argv[])
    {
        requireInitialMainThread();
        if(phase!=EntryPhase::Claimed)
            throw std::runtime_error("Native CPU placement requires authenticated original process entry");
        auto native=nativeCorePlaces();
        const char *raw_receipt=::getenv(kNativeCorePlacesEnvironment);
        const auto receipt=raw_receipt ? std::optional{parseCorePlaceReceipt(raw_receipt)} : std::nullopt;
        const char *requested=::getenv("OMP_PLACES");
        const bool automatic=!requested || !*requested || std::string_view(requested)=="cores" ||
            (receipt && std::string_view(requested)==receipt->places);
        const bool current=receipt && receipt->process==static_cast<long>(::getpid());
        if(current)
        {
            validateCapacityReceipt(raw_receipt,native,::getpid());
            if(omp_get_proc_bind()!=omp_proc_bind_close || !physicalPlaces(native))
                throw std::runtime_error("Native OpenMP startup differs from its physical core binding policy");
        }
        if(!automatic && !current)
        {
            // An authored native core list retains its own order/binding.
            // Socket-wide/unbound modes do not acquire a core-admission token.
            if(!physicalPlaces(native) || omp_get_proc_bind()==omp_proc_bind_false)
            {
                phase=EntryPhase::OtherPlacement;
                return;
            }
            admitted.reset(new NativeOpenMPCorePlaces(native,encodeCorePlaces(native),
                omp_get_proc_bind(),::getpid(),::syscall(SYS_gettid)));
            phase=EntryPhase::Ready;
            return;
        }

        cpu_set_t partition;
        CPU_ZERO(&partition);
        {
            const ScopedOpenMPPartitionAffinity original_partition;
            if(::sched_getaffinity(0,sizeof(partition),&partition)!=0)
                throw std::runtime_error("Cannot observe original CPU partition for native startup");
        }
        std::vector<int> allowed;
        for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&partition))allowed.push_back(cpu);
        auto desired=capacityOrderedCorePlaces(allowed);
        if(desired.empty())throw std::runtime_error("Native CPU startup has no admitted physical core");
        const auto encoding=encodeCorePlaces(desired);
        const bool compatible=native==desired && omp_get_proc_bind()==omp_proc_bind_close;
        if(current && !compatible)
            throw std::runtime_error("Native CPU startup receipt is stale for the original partition");
        publishCorePlaces(encoding);
        if(compatible)
        {
            admitted.reset(new NativeOpenMPCorePlaces(std::move(desired),encoding,
                omp_proc_bind_close,::getpid(),::syscall(SYS_gettid)));
            phase=EntryPhase::Ready;
            return;
        }
        if(argc<=0 || !argv)throw std::runtime_error("Native CPU startup requires original application arguments");
        std::vector<char *> original;
        original.reserve(static_cast<std::size_t>(argc)+1);
        for(int i=0;i<argc;++i)
        {
            if(!argv[i])throw std::runtime_error("Native CPU startup received incomplete application arguments");
            original.push_back(argv[i]);
        }
        original.push_back(nullptr);
        // Only original entry reaches exec; no initialized MPI, worker pool or
        // application runtime is discarded. Reentry reauthenticates the owner.
        const ScopedOpenMPPartitionAffinity original_partition;
        ::execv("/proc/self/exe",original.data());
        throw std::runtime_error("Cannot enter native CPU places: "+std::string(std::strerror(errno)));
    }

    bool NativeCPUPlacement::hasCoreAdmission() noexcept { return phase==EntryPhase::Ready; }

    const NativeOpenMPCorePlaces& NativeCPUPlacement::admission()
    {
        if(phase!=EntryPhase::Ready || !admitted || admitted->process_!=::getpid() ||
           admitted->thread_!=::syscall(SYS_gettid) || omp_get_level()!=0 ||
           nativeCorePlaces()!=admitted->places_ || omp_get_proc_bind()!=admitted->binding_)
            throw std::runtime_error("Missing, foreign or stale original-entry native CPU core admission");
        return *admitted;
    }
}
