/**
 * @file Test__CoordinatorReadiness.cpp
 * @brief Deterministic CPU scheduling replay of actual NCCL/RCCL owner methods.
 *
 * The generated include contains actual constructors, initialize, shutdown and
 * complete worker loops. Observable mutex/CV wrappers expose scheduling edges;
 * they implement no coordinator policy. Only hardware creation/retirement and
 * dependency discovery are doubled. No native SDK or driver is linked.
 */
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <queue>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define HAVE_NCCL 1
#define HAVE_RCCL 1
#define LOG_DEBUG(...) ((void)0)
#define LOG_TRACE(...) ((void)0)
#define LOG_INFO(...) ((void)0)
#define LOG_WARN(...) ((void)0)
#define LOG_ERROR(...) ((void)0)

namespace coordinator_replay {
/** @brief Report an exact causal checkpoint instead of hanging an old-source control. */
[[noreturn]] void fail(const char *message) {
    std::fprintf(stderr, "COORDINATOR_REPLAY_FAILURE: %s\n", message);
    std::fflush(stderr);
    std::_Exit(83);
}
/** @brief A violated replay invariant is a hard failure, never a scheduling retry. */
void require(bool value, const char *message) { if (!value) fail(message); }
enum class Mode { Stop, Gap, Lifecycle };
/** @brief Scheduling receipts belong only to this fixture, not production ownership. */
struct Schedule {
    Mode mode = Mode::Lifecycle;
    const std::thread::id caller = std::this_thread::get_id();
    std::binary_semaphore start_native{0}, progress{0}, stop_sent{0};
    std::atomic<bool> gap{false}, stopped{false}, notified{false};
    std::atomic<int> creates{0}, retires{0}, work{0};
    bool success = true;
    bool worker() const { return std::this_thread::get_id() != caller; }
};
/** @brief Real mutex with a receipt before the publisher blocks on constructor exclusion. */
struct ReplayMutex {
    std::mutex native;
    Schedule *schedule = nullptr;
    void lock() {
        if (schedule && schedule->worker() && schedule->gap.load()) schedule->progress.release();
        native.lock();
    }
    void unlock() { native.unlock(); }
};
/** @brief Real condition wait plus exact pre-registration and post-publication schedules. */
struct ReplayCondition {
    std::condition_variable_any native;
    Schedule *schedule = nullptr;
    template<class Lock, class Predicate> void wait(Lock &lock, Predicate predicate) {
        while (!predicate()) {
            if (schedule && !schedule->worker() && schedule->mode == Mode::Gap) {
                schedule->gap.store(true);
                schedule->start_native.release();
                // The corrected publisher reports its lock attempt while this
                // caller still owns the real predicate mutex. The old one
                // publishes into the gap and fails in notify_all instead.
                schedule->progress.acquire();
                schedule->gap.store(false);
            }
            if (schedule && schedule->worker() && schedule->mode == Mode::Stop && schedule->stopped.load())
                fail("running publication overwrote the owner's completed stop");
            native.wait(lock);
        }
    }
    void notify_all() {
        if (schedule && schedule->worker() && schedule->mode == Mode::Gap && schedule->gap.load())
            fail("readiness notification passed the predicate-to-wait gap without its mutex");
        native.notify_all();
        if (schedule && schedule->mode == Mode::Stop) {
            if (schedule->worker() && !schedule->notified.exchange(true)) schedule->stop_sent.acquire();
            else if (!schedule->worker()) { schedule->stopped.store(true); schedule->stop_sent.release(); }
        }
    }
};
/** @brief Loader/discovery doubles do not initialize any native runtime. */
namespace dependency {
bool isLoaded() { return true; }
bool load() { fail("unexpected native loader call"); }
bool isCommFinalizeAvailable() { return true; }
const char *getLastError() { return "device-free source replay"; }
}
}
using namespace coordinator_replay;
namespace nccl = coordinator_replay::dependency;
namespace rccl = coordinator_replay::dependency;

namespace llaminar2 {
/** @brief Existing physical facts are fixture inputs, not native queries or capacity allowances. */
void initializeCUDAHardwareMetadata() {}
const std::vector<int> &cudaHardwareMetadata() { static const std::vector<int> facts(16); return facts; }
bool installNCCLGraphOrderingPolicy(std::string &) { return true; }

/**
 * @brief Replay operands for the actual owner methods; no lifecycle algorithm is reproduced here.
 *
 * The original thread, atomic flags and work queue remain the sole authority.
 * Native setup/cleanup below are the only semantic doubles. Concrete classes
 * differ only by the actual method definitions generated from their sources.
 */
#include "CoordinatorReadinessFields.inc"
/** @brief Hardware-only initialization double, including rollback before failure publication. */
template<class Coordinator> void nativeInitialize(Coordinator &owner) {
    if (owner.schedule.mode == Mode::Gap) owner.schedule.start_native.acquire();
    ++owner.schedule.creates;
    if (!owner.schedule.success) owner.cleanupOnThread();
    owner.init_success_.store(owner.schedule.success);
}
void NCCLCoordinator::initializeOnThread() { nativeInitialize(*this); }
void RCCLCoordinator::initializeOnThread() { nativeInitialize(*this); }
/** @brief Actual worker control flow determines the hardware-only cleanup boundary. */
void NCCLCoordinator::cleanupOnThread() { ++schedule.retires; }
void RCCLCoordinator::cleanupOnThread() { ++schedule.retires; }
}
#include "CoordinatorReadinessActual.inc"

namespace {
/** @brief Exercise actual producer/wait/retire methods with no timeout or sleep. */
template<class Coordinator> void replay(Mode mode, bool success, int iterations) {
    Coordinator owner;
    owner.queue_mutex_.schedule = &owner.schedule;
    owner.queue_cv_.schedule = &owner.schedule;
    owner.schedule.mode = mode;
    owner.schedule.success = success;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        owner.schedule.notified.store(false); owner.schedule.stopped.store(false);
        require(owner.initialize({0, 1}) == success, "actual initialize returned a wrong outcome");
        require(owner.init_complete_.load(), "actual initialization never published completion");
        if (success && mode == Mode::Lifecycle) {
            std::lock_guard lock(owner.queue_mutex_);
            for (int item = 0; item < 3; ++item) owner.work_queue_.push([&] { ++owner.schedule.work; });
        }
        owner.shutdown(); owner.shutdown();
        require(!owner.running_.load() && !owner.initialized_.load() && !owner.coordinator_thread_.joinable(),
                "actual shutdown left a live owner/thread");
        require(owner.schedule.creates.load() == iteration + 1 && owner.schedule.retires.load() == iteration + 1,
                "actual native lifetime was dropped or retired twice");
    }
    require(owner.schedule.work.load() == ((mode == Mode::Lifecycle && success) ? 3 * iterations : 0),
            "actual stopped queue did not drain all original work");
}
}

/** @brief Run each backend's exact source separately, preserving causal old-source failures. */
int main(int argc, char **argv) {
    if (argc != 3) fail("expected backend and stop/gap-success/gap-failure/lifecycle case");
    const std::string backend = argv[1], name = argv[2];
    require(name == "stop" || name == "gap-success" || name == "gap-failure" || name == "lifecycle",
            "unknown replay schedule");
    const auto mode = name == "stop" ? Mode::Stop : name.starts_with("gap-") ? Mode::Gap : Mode::Lifecycle;
    const bool success = name != "gap-failure";
    if (backend == "NCCL") replay<llaminar2::NCCLCoordinator>(mode, success, 20);
    else if (backend == "RCCL") replay<llaminar2::RCCLCoordinator>(mode, success, 20);
    else fail("unknown backend");
    std::printf("PASS actual %s coordinator %s: 20 lifecycles; no SDK/runtime/device\n", argv[1], argv[2]);
}
