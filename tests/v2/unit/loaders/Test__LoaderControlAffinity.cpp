/**
 * @file Test__LoaderControlAffinity.cpp
 * @brief Native mask contracts for original GPU staging, progress and preparation boundaries.
 *
 * The generated include compiles the complete production staging pool and its
 * task/retirement implementation, actual progress start/stop, and the complete
 * GPU/CPU weight-preparation method. Only storage, backend and model operands
 * are doubled. Their callbacks record native masks and small authored payloads;
 * they implement no model-loading, sharding or quantization algorithm.
 * A one-shot executable-local pthread fault forwards every ordinary operation
 * unchanged and proves construction rollback without a production test seam.
 * Later-lane faults also count successful native creates/joins, proving every
 * partially constructed worker retires before its referenced lane storage.
 */
#include "utils/ScopedOpenMPPartitionAffinity.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <malloc.h>
#include <pthread.h>

#define LOG_DEBUG(message) do { if constexpr (false) { std::ostringstream unused_log; unused_log << message; } } while (false)
#define LOG_INFO(message) LOG_DEBUG(message)
#define LOG_WARN(message) LOG_DEBUG(message)
#define LOG_ERROR(message) LOG_DEBUG(message)

namespace loader_affinity {
/** @brief Stop immediately with the violated production-boundary contract. */
[[noreturn]] void fail(const char *message) { std::fprintf(stderr, "LOADER_AFFINITY_FAILURE: %s\n", message); std::exit(83); }
/** @brief Never turn an unobserved native boundary into a passing test. */
void require(bool value, const char *message) { if (!value) fail(message); }
thread_local bool fail_next_thread = false;
thread_local unsigned fail_thread_countdown = 0;
thread_local bool track_thread_retirement = false;
thread_local unsigned created_threads = 0, joined_threads = 0;
/** @return Exact native caller eligibility; failure is not substituted with topology. */
cpu_set_t mask() {
    cpu_set_t value{};
    require(sched_getaffinity(0, sizeof(value), &value) == 0, "cannot observe native loader mask");
    return value;
}
/** @return Initial-rank OpenMP places, or the exact native mask for unbound startup. */
cpu_set_t partition() {
    const char *policy = std::getenv("OMP_PROC_BIND");
    require(policy && (std::string(policy) == "close" || std::string(policy) == "false"), "missing pre-exec binding policy");
    require((omp_get_proc_bind() == omp_proc_bind_false) == (std::string(policy) == "false"), "OpenMP ignored selected startup policy");
    if (omp_get_proc_bind() == omp_proc_bind_false) return mask();
    require(omp_get_num_places() >= 2, "loader affinity contract requires two admitted CPU places");
    cpu_set_t value{};
    for (int place = 0; place < omp_get_num_places(); ++place) {
        std::vector<int> cpus(static_cast<std::size_t>(omp_get_place_num_procs(place)));
        omp_get_place_proc_ids(place, cpus.data());
        for (int cpu : cpus) CPU_SET(cpu, &value);
    }
    return value;
}
/** @brief Native entry/descendant observations, independent of preparation algorithms. */
struct Observation {
    cpu_set_t worker{}, descendant{};
    void record() { worker = mask(); std::thread child([&] { descendant = mask(); }); child.join(); }
    void expect(const cpu_set_t &expected) const {
        require(CPU_EQUAL(&worker, &expected), "actual loader child inherited only its creator place");
        require(CPU_EQUAL(&descendant, &expected), "loader descendant lost admitted/default affinity");
    }
};
/** @brief Restore caller-installed pthread defaults after testing their precedence. */
class DefaultAffinity {
    pthread_attr_t original_;
public:
    explicit DefaultAffinity(const cpu_set_t &value) {
        pthread_attr_t changed;
        require(pthread_getattr_default_np(&original_) == 0 && pthread_getattr_default_np(&changed) == 0, "cannot retain native pthread defaults");
        require(pthread_attr_setaffinity_np(&changed, sizeof(value), &value) == 0 && pthread_setattr_default_np(&changed) == 0, "cannot install authored default affinity");
        require(pthread_attr_destroy(&changed) == 0, "cannot retire temporary native attributes");
    }
    ~DefaultAffinity() {
        require(pthread_setattr_default_np(&original_) == 0 && pthread_attr_destroy(&original_) == 0, "cannot restore exact pthread defaults");
    }
};
}

/** @brief Inject one selected fixture construction; preserve libc's complete ordinary policy. */
extern "C" int pthread_create(pthread_t *thread, const pthread_attr_t *attributes,
                              void *(*entry)(void *), void *argument) noexcept {
    if (loader_affinity::fail_next_thread ||
        (loader_affinity::fail_thread_countdown != 0 && --loader_affinity::fail_thread_countdown == 0)) {
        loader_affinity::fail_next_thread = false; return EAGAIN;
    }
    using Create = int (*)(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
    static const auto native = reinterpret_cast<Create>(dlsym(RTLD_NEXT, "pthread_create"));
    if (!native) std::abort();
    const auto result = native(thread, attributes, entry, argument);
    if (result == 0 && loader_affinity::track_thread_retirement) ++loader_affinity::created_threads;
    return result;
}

/** @brief Count successful native joins only while the constructor-failure contract is armed. */
extern "C" int pthread_join(pthread_t thread, void **result) {
    using Join = int (*)(pthread_t, void **);
    static const auto native = reinterpret_cast<Join>(dlsym(RTLD_NEXT, "pthread_join"));
    if (!native) std::abort();
    const auto status = native(thread, result);
    if (status == 0 && loader_affinity::track_thread_retirement) ++loader_affinity::joined_threads;
    return status;
}

using namespace loader_affinity;
namespace llaminar2 {
/** @brief Authored bounded bytes plus the actual task's native entry observation. */
struct BoundedSource {
    Observation observed;
    bool throws = false;
    void read(std::size_t, std::span<std::byte> destination) {
        observed.record();
        if (throws) throw std::runtime_error("authored source I/O failure");
        std::fill(destination.begin(), destination.end(), std::byte{0x5a});
    }
};
/** @brief Staging collaborators own only small CPU buffers and event observations. */
struct IBackend { bool waitForEvent(void *, int) { return true; } };
struct PinnedRingBuffer {
    std::array<std::array<std::byte, 64>, 2> bytes{};
    void *getSlot(int lane) { return bytes.at(static_cast<std::size_t>(lane)).data(); }
    std::size_t slotSize() const { return 64; }
};
struct FileReadPlan { int fd = -1; };
struct MmapFileReader {
    void read(const FileReadPlan &, void *, std::size_t, const void *&, std::size_t &, std::string &) const {
        fail("unexpected file algorithm in control-affinity fixture");
    }
};
struct WeightJob {
    std::shared_ptr<BoundedSource> bounded_source;
    std::size_t source_offset_bytes = 0, raw_bytes = 32;
    const void *host_raw_data = nullptr;
};
/** @brief Renderer/model collaborators are unused operands, not alternate loading implementations. */
struct WeightLoadProgress {
    int registerDevice(const std::string &, std::size_t) { fail("unexpected model progress registration"); }
    std::string makeDeviceLabel(const std::string &) { fail("unexpected model progress labeling"); }
    void update(int, std::size_t) { fail("unexpected model progress update"); }
    void finish(int) { fail("unexpected model progress completion"); }
};
/** @brief Only MPI polling I/O is doubled; start, stop and their state publications are original. */
struct WeightLoadProgressAggregator {
    int rank_ = 0;
    std::atomic<bool> polling_{false};
    std::thread poll_thread_;
    std::shared_ptr<WeightLoadProgress> renderer_;
    std::promise<Observation> observed;
    void startPolling(std::shared_ptr<WeightLoadProgress>);
    void stopPolling();
    void pollLoop() {
        require(polling_.load(std::memory_order_acquire), "poll worker entered before active publication");
        Observation entry; entry.record(); observed.set_value(entry);
    }
    ~WeightLoadProgressAggregator() { stopPolling(); }
};
struct DeviceId {
    bool gpu = true;
    bool is_gpu() const { return gpu; }
    bool is_rocm() const { return gpu; }
    bool is_cuda() const { return false; }
    std::string to_string() const { return gpu ? "authored-gpu" : "authored-cpu"; }
};
struct FrozenModelWeightSet {};
enum class PhysicalMemoryOwner { PrimaryModelWeights };
/** @brief Observe the original preparation dispatch; no tensor or model algorithm is duplicated. */
struct WeightManager {
    using PreloadProgressCallback = std::function<void(std::size_t, std::size_t, const std::string &)>;
    struct UnusedTensor { std::size_t size_bytes() const { fail("unexpected tensor payload access"); } };
    std::shared_ptr<WeightLoadProgress> weight_load_progress_;
    std::mutex cache_mutex_;
    std::map<std::string, std::shared_ptr<UnusedTensor>> cache_;
    Observation gpu_entry, cpu_entry;
    cpu_set_t creator{};
    bool outcome = true, throws = false;
    int gpu_calls = 0, cpu_calls = 0, upload_calls = 0;
    bool prepareWeightsForDeviceImpl(DeviceId, std::function<bool(const std::string &)>, const FrozenModelWeightSet *, bool, PhysicalMemoryOwner);
    bool packGemmWeightsViaPipeline(DeviceId device, std::function<bool(const std::string &)>, const FrozenModelWeightSet *, bool,
                                   std::nullptr_t, std::optional<std::size_t>, PhysicalMemoryOwner) {
        require(device.is_gpu(), "CPU request entered GPU loading collaborator");
        ++gpu_calls; gpu_entry.record();
        if (throws) throw std::runtime_error("authored GPU preparation failure");
        return outcome;
    }
    bool uploadNonGemmWeights(DeviceId device, std::function<bool(const std::string &)>) {
        require(device.is_gpu(), "CPU request entered non-GEMM GPU upload");
        const auto actual = mask();
        require(CPU_EQUAL(&actual, &creator), "GPU launch scope moved subsequent creator upload");
        ++upload_calls; return true;
    }
    bool uploadFrozenNonGemmWeights(DeviceId device, const FrozenModelWeightSet &, std::function<bool(const std::string &)> filter,
                                   bool, PhysicalMemoryOwner) { return uploadNonGemmWeights(device, filter); }
    bool packGemmWeights(DeviceId device, PreloadProgressCallback, bool, std::function<bool(const std::string &)>) {
        require(!device.is_gpu(), "GPU request entered CPU packing collaborator");
        ++cpu_calls; cpu_entry.worker = mask(); cpu_entry.descendant = cpu_entry.worker; return outcome;
    }
    bool isGemmWeight(const std::string &) { fail("unexpected model weight classification"); }
};
}
#include "LoaderControlAffinityActual.inc"

namespace {
/** @brief Assert that a completed boundary retains the exact creator mask. */
void expectCreator(const cpu_set_t &before) { const auto after = mask(); require(CPU_EQUAL(&before, &after), "loader boundary failed to restore its creator"); }
/** @brief Exercise actual staging tasks across partial/reused lanes and failed source work. */
void checkStaging(const cpu_set_t &expected) {
    using namespace llaminar2;
    const auto before = mask();
    IBackend backend; PinnedRingBuffer pinned; MmapFileReader reader;
    std::vector<WeightJob> jobs(2);
    for (auto &job : jobs) job.bounded_source = std::make_shared<BoundedSource>();
    std::vector<FileReadPlan> plans(jobs.size()); std::vector<void *> events(jobs.size(), &backend);
    { StagingLanePool empty(backend, 0, pinned, reader, plans, jobs, events, 0); }
    expectCreator(before);
    {
        StagingLanePool pool(backend, 0, pinned, reader, plans, jobs, events, 2);
        expectCreator(before);
        for (int iteration = 0; iteration < 20; ++iteration) {
            // Partial buckets then reuse both slots; allocation capacity is not work.
            const int count = iteration % 2 + 1;
            for (int lane = 0; lane < count; ++lane) require(pool.submit(lane, lane, iteration != 0), "staging rejected a drained lane");
            for (int lane = 0; lane < count; ++lane) {
                StagedSource result;
                require(pool.wait(lane, lane, result) && result.error.empty() && result.payload, "actual staging task failed");
                jobs[lane].bounded_source->observed.expect(expected);
                require(result.bounded_source_bytes == jobs[lane].raw_bytes, "staging changed its authored live extent");
                const auto *bytes = static_cast<const std::byte *>(result.payload);
                require(std::all_of(bytes, bytes + jobs[lane].raw_bytes, [](std::byte value) { return value == std::byte{0x5a}; }), "staged task lost authored bytes");
            }
        }
        jobs[0].bounded_source->throws = true;
        require(pool.submit(0, 0, false), "staging rejected authored source failure");
        StagedSource failed;
        require(pool.wait(0, 0, failed) && !failed.error.empty(), "staging hid its source failure");
    }
    expectCreator(before);
    for (const unsigned failed_creation : {1u, 2u, 3u, 8u}) {
        fail_thread_countdown = failed_creation;
        created_threads = joined_threads = 0;
        track_thread_retirement = true;
        bool rejected = false;
        try { StagingLanePool failed(backend, 0, pinned, reader, plans, jobs, events, 8); }
        catch (const std::system_error &error) { rejected = error.code().value() == EAGAIN; }
        track_thread_retirement = false;
        require(rejected && fail_thread_countdown == 0, "staging lost its original native construction failure");
        require(created_threads == failed_creation - 1 && joined_threads == created_threads,
                "partial staging construction did not join every successfully created worker");
        expectCreator(before);
    }
}
/** @brief Exercise actual progress start/stop, early returns and failed construction rollback. */
void checkProgress(const cpu_set_t &expected) {
    using namespace llaminar2;
    const auto before = mask();
    WeightLoadProgressAggregator owner;
    auto renderer = std::make_shared<WeightLoadProgress>();
    owner.startPolling(nullptr); owner.rank_ = 1; owner.startPolling(renderer); owner.rank_ = 0;
    require(!owner.polling_.load() && !owner.poll_thread_.joinable(), "non-root/null progress request created a worker");
    for (int iteration = 0; iteration < 20; ++iteration) {
        owner.observed = std::promise<Observation>(); auto observed = owner.observed.get_future();
        owner.startPolling(renderer); expectCreator(before);
        bool rejected = false;
        try { owner.startPolling(renderer); } catch (const std::logic_error &) { rejected = true; }
        require(rejected, "duplicate polling admission was accepted");
        observed.get().expect(expected); owner.stopPolling(); owner.stopPolling();
        require(!owner.polling_.load() && !owner.poll_thread_.joinable(), "progress stop retained an active owner");
    }
    fail_next_thread = true; bool rejected = false;
    try { owner.startPolling(renderer); } catch (const std::system_error &) { rejected = true; }
    require(rejected && !fail_next_thread && !owner.polling_.load() && !owner.poll_thread_.joinable(), "failed progress construction published an active owner");
    expectCreator(before);
}
/** @brief Qualify only preparation launch roles/outcomes and creator-local joins/upload. */
void checkPreparation(const cpu_set_t &expected) {
    using namespace llaminar2;
    const auto before = mask();
    for (int iteration = 0; iteration < 20; ++iteration) {
        WeightManager owner; owner.creator = before;
        FrozenModelWeightSet frozen;
        require(owner.prepareWeightsForDeviceImpl({true}, {}, iteration % 2 ? &frozen : nullptr, true, PhysicalMemoryOwner::PrimaryModelWeights), "GPU preparation routing failed");
        owner.gpu_entry.expect(expected);
        require(owner.gpu_calls == 1 && owner.cpu_calls == 0 && owner.upload_calls == 1, "GPU preparation changed dispatch roles");
        require(owner.prepareWeightsForDeviceImpl({false}, {}, nullptr, true, PhysicalMemoryOwner::PrimaryModelWeights), "CPU preparation routing failed");
        owner.cpu_entry.expect(before);
        require(owner.cpu_calls == 1 && owner.gpu_calls == 1, "CPU packing was moved into GPU preparation");
        expectCreator(before);
    }
    WeightManager failed; failed.creator = before; failed.throws = true;
    bool rejected = false;
    try { (void)failed.prepareWeightsForDeviceImpl({true}, {}, nullptr, true, PhysicalMemoryOwner::PrimaryModelWeights); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected && failed.upload_calls == 0, "GPU preparation exception crossed into subsequent upload");
    expectCreator(before);
    fail_next_thread = true; rejected = false;
    try { (void)failed.prepareWeightsForDeviceImpl({true}, {}, nullptr, true, PhysicalMemoryOwner::PrimaryModelWeights); }
    catch (const std::system_error &) { rejected = true; }
    require(rejected && !fail_next_thread, "GPU async construction hid its native failure");
    expectCreator(before);
    WeightManager declined; declined.creator = before; declined.outcome = false;
    require(!declined.prepareWeightsForDeviceImpl({true}, {}, nullptr, true, PhysicalMemoryOwner::PrimaryModelWeights) &&
            declined.gpu_calls == 1 && declined.upload_calls == 0, "declined GPU preparation entered subsequent upload");
    require(!declined.prepareWeightsForDeviceImpl({false}, {}, nullptr, true, PhysicalMemoryOwner::PrimaryModelWeights) &&
            declined.cpu_calls == 1, "declined CPU preparation changed dispatch roles");
    declined.cpu_entry.expect(before);
    expectCreator(before);
}
/** @brief Reject compute-region creation without unwinding an exception across OpenMP. */
void checkNested(const std::string &boundary) {
    using namespace llaminar2;
    const auto before = mask(); bool rejected = false;
#pragma omp parallel num_threads(1)
    {
        try {
            if (boundary == "Staging") {
                IBackend backend; PinnedRingBuffer pinned; MmapFileReader reader;
                std::vector<WeightJob> jobs(1); std::vector<FileReadPlan> plans(1); std::vector<void *> events(1);
                StagingLanePool owner(backend, 0, pinned, reader, plans, jobs, events, 1);
            } else if (boundary == "Progress") {
                WeightLoadProgressAggregator owner;
                try { owner.startPolling(std::make_shared<WeightLoadProgress>()); }
                catch (...) { require(!owner.polling_.load() && !owner.poll_thread_.joinable(), "nested polling changed active state"); throw; }
            } else {
                WeightManager owner; owner.creator = mask();
                (void)owner.prepareWeightsForDeviceImpl({true}, {}, nullptr, true, PhysicalMemoryOwner::PrimaryModelWeights);
            }
        } catch (const std::exception &error) { rejected = std::string(error.what()).find("outer host thread") != std::string::npos; }
        catch (...) { rejected = false; }
    }
    require(rejected, "nested loader construction entered independent control"); expectCreator(before);
}
}

/** @brief Run one explicit production-boundary contract in its native pre-exec environment. */
int main(int argc, char **argv) {
    require(argc == 2, "expected Staging, Progress or GPUPreparation boundary");
    const std::string boundary = argv[1]; const auto expected = partition();
    const auto check = [&](const cpu_set_t &value) {
        if (boundary == "Staging") checkStaging(value);
        else if (boundary == "Progress") checkProgress(value);
        else if (boundary == "GPUPreparation") checkPreparation(value);
        else fail("unknown loader boundary");
    };
    check(expected);
    cpu_set_t explicit_default{};
    for (int cpu = CPU_SETSIZE - 1; cpu >= 0; --cpu)
        if (CPU_ISSET(cpu, &expected)) { CPU_SET(cpu, &explicit_default); break; }
    { const DefaultAffinity defaults(explicit_default); check(explicit_default); }
    checkNested(boundary);
    std::printf("PASS actual %s loader affinity: native masks/tasks/defaults/role separation/retirement/failure\n", argv[1]);
}
