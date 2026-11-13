/**
 * @file Test__CoordinatorReadiness.cpp
 * @brief Deterministic CPU scheduling replay of actual NCCL/RCCL owner methods.
 *
 * The generated include contains actual constructors, initialize, shutdown and
 * complete worker loops. Observable mutex/CV wrappers expose scheduling edges;
 * they implement no coordinator policy. Only hardware creation/retirement and
 * dependency discovery are doubled. No native SDK or driver is linked.
 * Separate lifecycle contracts compile the complete bridge initializer and
 * device-controller start body. Real scheduler masks prove control/descendant
 * inheritance, native self-placement, exact caller restoration and explicit
 * pthread-default precedence under enabled and disabled OpenMP binding.
 */
#include "utils/ScopedOpenMPPartitionAffinity.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <pthread.h>

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
enum class Mode { Stop, Gap, Lifecycle, Affinity };
/** @brief Scheduling receipts belong only to this fixture, not production ownership. */
struct Schedule {
    Mode mode = Mode::Lifecycle;
    const std::thread::id caller = std::this_thread::get_id();
    std::binary_semaphore start_native{0}, progress{0}, stop_sent{0};
    std::atomic<bool> gap{false}, stopped{false}, notified{false};
    std::atomic<int> creates{0}, retires{0}, work{0};
    cpu_set_t creator_mask{}, control_mask{}, descendant_mask{}, native_mask{};
    bool success = true;
    bool worker() const { return std::this_thread::get_id() != caller; }
};
/** @brief One authored communicator receipt; real abort helper threads own its observations. */
struct NativeAbortObservation {
    Schedule schedule;
    std::atomic<int> calls{0};
};
/** @return The calling native thread's allowed CPUs; observation failure is fatal. */
cpu_set_t nativeMask() {
    cpu_set_t result{};
    require(sched_getaffinity(0, sizeof(result), &result) == 0, "cannot read native control affinity");
    return result;
}
/** @return An independent union of initial-rank OpenMP places, or exact unbound mask. */
cpu_set_t admittedControlMask() {
    if (omp_get_proc_bind() == omp_proc_bind_false) return nativeMask();
    cpu_set_t result{};
    require(omp_get_num_places() >= 2, "control affinity contract requires two admitted CPU places");
    for (int place = 0; place < omp_get_num_places(); ++place) {
        std::vector<int> cpus(static_cast<std::size_t>(omp_get_place_num_procs(place)));
        omp_get_place_proc_ids(place, cpus.data());
        for (int cpu : cpus) CPU_SET(cpu, &result);
    }
    return result;
}
/** @brief Observe real control/descendant inheritance, then permit native self-placement. */
void observeControl(Schedule &schedule) {
    schedule.control_mask = nativeMask();
    std::thread descendant([&] { schedule.descendant_mask = nativeMask(); });
    descendant.join();
    cpu_set_t local{};
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &schedule.control_mask)) { CPU_SET(cpu, &local); break; }
    require(sched_setaffinity(0, sizeof(local), &local) == 0, "native control self-placement failed");
    schedule.native_mask = nativeMask();
}
/** @brief Assert resource/capture preparation retains creator placement. */
void observePreparation(const Schedule &schedule) {
    const auto actual = nativeMask();
    require(CPU_EQUAL(&actual, &schedule.creator_mask), "control scope moved resource/capture preparation");
}
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
using ncclComm_t = NativeAbortObservation *;
enum ncclResult_t { ncclSuccess, ncclInvalidArgument, ncclUnhandledCudaError };
/** @brief Native abort collaborator observes actual helper/descendant masks without loading a runtime. */
ncclResult_t ncclCommAbort(ncclComm_t comm) { ++comm->calls; observeControl(comm->schedule); return ncclSuccess; }
bool isLoaded() { return true; }
bool load() { fail("unexpected native loader call"); }
bool isCommFinalizeAvailable() { return true; }
const char *getLastError() { return "device-free source replay"; }
}
}
using namespace coordinator_replay;
namespace nccl = coordinator_replay::dependency;
namespace rccl = coordinator_replay::dependency;
using cudaError_t = int;
using hipError_t = int;
constexpr int cudaSuccess = 0, cudaErrorInvalidDevice = 1, hipSuccess = 0, hipErrorInvalidDevice = 1;
/** @brief Authored device selection validates explicit fixture ordinals without hardware inventory. */
int cudaSetDevice(int ordinal) { return ordinal >= 0 ? cudaSuccess : cudaErrorInvalidDevice; }
/** @brief Symmetric authored HIP selector; no native dependency is linked. */
int selectHipDevice(int ordinal) { return ordinal >= 0 ? hipSuccess : hipErrorInvalidDevice; }

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
    if (owner.schedule.mode == Mode::Affinity) observeControl(owner.schedule);
    if (!owner.schedule.success) owner.cleanupOnThread();
    owner.init_success_.store(owner.schedule.success);
}
void NCCLCoordinator::initializeOnThread() { nativeInitialize(*this); }
void RCCLCoordinator::initializeOnThread() { nativeInitialize(*this); }
/** @brief Actual worker control flow determines the hardware-only cleanup boundary. */
void NCCLCoordinator::cleanupOnThread() { ++schedule.retires; }
void RCCLCoordinator::cleanupOnThread() { ++schedule.retires; }

/** @brief Explicit fixture participant identity; no inventory or GPU count is queried. */
struct DeviceId {
    int ordinal = 0;
    int gpu_ordinal() const { return ordinal; }
    bool is_gpu() const { return true; }
    std::string toString() const { return "authored-participant:" + std::to_string(ordinal); }
};
/** @brief Hardware allocation collaborators observe placement without allocating devices. */
struct ControlResourceBackend {
    Schedule *schedule;
    void *createStream(int) { observePreparation(*schedule); return this; }
    void *createEvent(int) { observePreparation(*schedule); return this; }
    void *allocatePinned(std::size_t, int) { observePreparation(*schedule); return this; }
};
/** @brief Original bridge initializer operands with hardware resource operations doubled. */
struct HeterogeneousBackend : HeterogeneousControlGeometry {
    Schedule schedule;
    ControlResourceBackend backend{&schedule};
    struct { std::vector<DeviceId> devices{{0}, {1}}; } device_group_;
    struct AsyncBridgeParticipant {
        DeviceId device;
        ControlResourceBackend *backend = nullptr;
        void *transfer_stream = nullptr;
        std::array<void *, ASYNC_BRIDGE_HOST_BUFFER_SLOTS> host_buffers{}, input_ready_events{}, output_consumed_events{};
    };
    struct Transaction { std::vector<void *> buffers, producer_streams, producer_events; };
    bool initialized_ = true, async_bridge_ready_ = false, async_stop_requested_ = false;
    std::size_t async_max_payload_bytes_ = 0, async_pending_head_ = 0, async_pending_tail_ = 0,
        async_pending_count_ = 0, async_submission_cursor_ = 0;
    std::uint64_t async_next_generation_ = 0, async_lifecycle_epoch_ = 0;
    std::vector<AsyncBridgeParticipant> async_participants_;
    std::vector<Transaction> async_transactions_;
    std::string last_error_;
    std::thread async_worker_;
    bool initializeAsyncBridgeResources(std::size_t);
    ControlResourceBackend *getBackendFor(DeviceId) { return &backend; }
    void asyncBridgeWorkerMain() { ++schedule.creates; observeControl(schedule); }
    void releaseAsyncBridgeResources() noexcept {
        if (async_worker_.joinable()) async_worker_.join();
        async_participants_.clear(); async_transactions_.clear(); async_bridge_ready_ = false;
    }
    ~HeterogeneousBackend() { releaseAsyncBridgeResources(); }
};
/** @brief Authored activation operands; production start() owns all state transitions. */
enum class MoEOverlayDeviceControllerActivationState { Prepared, Starting, Running, Failed };
struct MoEOverlayDeviceControllerGraphService {
    enum class ExecutionMode { Static, Dynamic };
    Schedule schedule;
    struct Context { int size = 1; int world_size() const { return size; } };
    struct { ExecutionMode execution_mode = ExecutionMode::Dynamic; std::shared_ptr<Context> mpi_ctx = std::make_shared<Context>(); } config_;
    struct Boundary {
        Schedule *schedule;
        bool prepareMoEOverlayFinitePlacementGraphs() { observePreparation(*schedule); return true; }
    };
    struct Endpoint { struct { Boundary *inference_boundary; } runtime_binding; };
    Boundary boundary{&schedule};
    std::vector<std::shared_ptr<Endpoint>> endpoints_{std::make_shared<Endpoint>(Endpoint{{&boundary}})};
    struct DynamicWorker {
        Schedule *schedule;
        std::jthread thread;
        std::condition_variable wake_cv;
        void run(std::stop_token) { ++schedule->creates; observeControl(*schedule); }
    };
    std::unique_ptr<DynamicWorker> dynamic_worker_ = std::make_unique<DynamicWorker>();
    std::mutex lifecycle_mutex_;
    std::atomic<MoEOverlayDeviceControllerActivationState> activation_state_{MoEOverlayDeviceControllerActivationState::Prepared};
    bool static_certified_ = false;
    MoEOverlayDeviceControllerGraphService() { dynamic_worker_->schedule = &schedule; }
    void start();
    std::string failureMessage() const { return "authored activation failure"; }
};
}
#include "CoordinatorReadinessActual.inc"

namespace {
/** @brief Preserve explicit caller-installed pthread defaults through native control creation. */
class ExplicitDefaultAffinity {
    pthread_attr_t original_;
public:
    /** @brief Install one authored native affinity override without changing other default attributes. */
    explicit ExplicitDefaultAffinity(const cpu_set_t &mask) {
        pthread_attr_t changed;
        require(pthread_getattr_default_np(&original_) == 0, "cannot retain pthread defaults");
        require(pthread_getattr_default_np(&changed) == 0, "cannot clone pthread defaults");
        require(pthread_attr_setaffinity_np(&changed, sizeof(mask), &mask) == 0, "cannot set authored default affinity");
        require(pthread_setattr_default_np(&changed) == 0, "cannot publish authored default affinity");
        require(pthread_attr_destroy(&changed) == 0, "cannot retire temporary pthread attributes");
    }
    /** @brief Restore the exact native default attribute set before another test owner starts. */
    ~ExplicitDefaultAffinity() {
        require(pthread_setattr_default_np(&original_) == 0, "cannot restore pthread defaults");
        require(pthread_attr_destroy(&original_) == 0, "cannot retire original pthread attributes");
    }
};
/** @brief Check control inheritance independently of the native worker's later self-placement. */
void requireControlObservation(const Schedule &schedule, const cpu_set_t &expected) {
    require(CPU_EQUAL(&schedule.control_mask, &expected), "actual control constructor inherited only the creator place");
    require(CPU_EQUAL(&schedule.descendant_mask, &expected), "native descendant lost admitted/default affinity");
    require(CPU_COUNT(&schedule.native_mask) == 1, "native worker self-placement was overwritten");
    const auto restored = nativeMask();
    require(CPU_EQUAL(&schedule.creator_mask, &restored), "control construction failed to restore its creator");
}
/** @brief Exercise complete actual initialize/shutdown bodies and their retained native child masks. */
template<class Coordinator> void checkCoordinatorControl(const cpu_set_t &expected, int iterations) {
    Coordinator owner;
    owner.schedule.mode = Mode::Affinity;
    owner.schedule.creator_mask = nativeMask();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        require(owner.initialize({0, 1}), "actual coordinator control initialization failed");
        requireControlObservation(owner.schedule, expected);
        owner.shutdown(); owner.shutdown();
        require(!owner.coordinator_thread_.joinable(), "coordinator control failed to join");
    }
    require(!owner.initialize({}), "empty coordinator admission unexpectedly created control work");
    require(owner.schedule.creates.load() == iterations, "empty/repeated coordinator admission created unexpected workers");
    owner.schedule.success = false;
    require(!owner.initialize({0, 1}), "failed native initialization unexpectedly succeeded");
    requireControlObservation(owner.schedule, expected);
    require(!owner.coordinator_thread_.joinable(), "failed native initialization left an unjoined control worker");
}
/** @brief Reject unsupported nested callers without allowing an exception to cross an OpenMP region. */
template<class Coordinator> void checkNestedCoordinator() {
    Coordinator owner;
    const auto original = nativeMask();
    bool rejected = false;
#pragma omp parallel num_threads(1)
    {
        try { (void)owner.initialize({0, 1}); }
        catch (const std::exception &error) {
            rejected = std::string(error.what()).find("outer host thread") != std::string::npos;
        }
        catch (...) { rejected = false; }
    }
    const auto actual = nativeMask();
    require(rejected && owner.schedule.creates.load() == 0 && !owner.coordinator_thread_.joinable(),
            "nested coordinator admission entered native control");
    require(CPU_EQUAL(&actual, &original), "nested coordinator rejection changed caller placement");
}
/** @brief Compile and execute the full production bridge initializer with resource-operation doubles. */
void checkBridgeControl(const cpu_set_t &expected, int iterations) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
        llaminar2::HeterogeneousBackend owner;
        owner.schedule.creator_mask = nativeMask();
        require(!owner.initializeAsyncBridgeResources(0), "zero bridge payload created a control worker");
        require(owner.schedule.creates.load() == 0, "zero bridge admission invoked native work");
        require(owner.initializeAsyncBridgeResources(1), "actual bridge control initialization failed");
        owner.async_worker_.join();
        requireControlObservation(owner.schedule, expected);
        require(owner.initializeAsyncBridgeResources(1), "ready bridge admission lost its original owner");
        require(owner.schedule.creates.load() == 1, "ready bridge admission created duplicate control work");
    }
}
/** @brief Execute actual controller activation after creator-local cold capture and before worker self-placement. */
void checkDeviceControllerControl(const cpu_set_t &expected, int iterations) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
        llaminar2::MoEOverlayDeviceControllerGraphService owner;
        owner.schedule.creator_mask = nativeMask();
        owner.start();
        owner.dynamic_worker_->thread.join();
        requireControlObservation(owner.schedule, expected);
        require(owner.activation_state_.load() == llaminar2::MoEOverlayDeviceControllerActivationState::Running,
                "actual device controller lost activation state");
    }
}
/** @brief Exercise native construction from the actual four production boundaries, including default-attribute precedence. */
void checkControlAffinity(const std::string &backend) {
    const char *policy = std::getenv("OMP_PROC_BIND");
    require(policy && (std::string(policy) == "false" || std::string(policy) == "close"),
            "control affinity contract requires an explicit pre-exec binding policy");
    require((omp_get_proc_bind() == omp_proc_bind_false) == (std::string(policy) == "false"),
            "native OpenMP runtime ignored the selected pre-exec binding policy");
    const auto expected = admittedControlMask();
    const auto check = [&](const cpu_set_t &mask, int iterations) {
        if (backend == "NCCL") checkCoordinatorControl<llaminar2::NCCLCoordinator>(mask, iterations);
        else if (backend == "RCCL") checkCoordinatorControl<llaminar2::RCCLCoordinator>(mask, iterations);
        else if (backend == "Heterogeneous") checkBridgeControl(mask, iterations);
        else if (backend == "MoEController") checkDeviceControllerControl(mask, iterations);
        else fail("unknown control-affinity backend");
    };
    check(expected, 20);
    cpu_set_t explicit_default{};
    for (int cpu = CPU_SETSIZE - 1; cpu >= 0; --cpu)
        if (CPU_ISSET(cpu, &expected)) { CPU_SET(cpu, &explicit_default); break; }
    {
        const ExplicitDefaultAffinity defaults(explicit_default);
        check(explicit_default, 1);
    }
    if (backend == "NCCL") checkNestedCoordinator<llaminar2::NCCLCoordinator>();
    else if (backend == "RCCL") checkNestedCoordinator<llaminar2::RCCLCoordinator>();
    else {
        // Bridge startup catches construction errors; controller activation
        // publishes Failed and rethrows. Neither supports a compute-region caller.
        bool rejected = false;
        const auto original = nativeMask();
        if (backend == "Heterogeneous") {
            llaminar2::HeterogeneousBackend owner;
            owner.schedule.creator_mask = original;
#pragma omp parallel num_threads(1)
            { rejected = !owner.initializeAsyncBridgeResources(1); }
            require(!owner.async_worker_.joinable() && owner.schedule.creates.load() == 0,
                    "nested bridge admission created control work");
        } else {
            for (const auto mode : {llaminar2::MoEOverlayDeviceControllerGraphService::ExecutionMode::Static,
                                    llaminar2::MoEOverlayDeviceControllerGraphService::ExecutionMode::Dynamic}) {
                llaminar2::MoEOverlayDeviceControllerGraphService empty;
                empty.config_.execution_mode = mode;
                empty.static_certified_ = true;
                empty.dynamic_worker_.reset();
                empty.start();
                require(empty.schedule.creates.load() == 0 &&
                        empty.activation_state_.load() == llaminar2::MoEOverlayDeviceControllerActivationState::Running,
                        "static/participant-empty activation created control work");
            }
            llaminar2::MoEOverlayDeviceControllerGraphService owner;
            owner.config_.mpi_ctx->size = 2; // Multi-rank activation does not enter local cold capture.
#pragma omp parallel num_threads(1)
            {
                try { owner.start(); }
                catch (const std::exception &error) {
                    rejected = std::string(error.what()).find("outer host thread") != std::string::npos;
                }
                catch (...) { rejected = false; }
            }
            require(!owner.dynamic_worker_->thread.joinable() && owner.schedule.creates.load() == 0,
                    "nested device controller activation created control work");
            require(owner.activation_state_.load() == llaminar2::MoEOverlayDeviceControllerActivationState::Failed,
                    "rejected device controller activation did not fail closed");
        }
        const auto actual = nativeMask();
        require(rejected && CPU_EQUAL(&actual, &original), "nested progress construction changed caller placement");
    }
}
/** @brief Exercise actual cold abort methods for sparse handles, both caller contexts and unused RCCL owners. */
template<class Coordinator> void checkAbortControl(const cpu_set_t &outer_expected, bool nested) {
    Coordinator owner;
    std::array<NativeAbortObservation, 2> observations;
    const auto original = nativeMask();
    for (auto &value : observations) value.schedule.creator_mask = original;
    owner.device_ordinals_ = {0, 1, 2};
    owner.comms_ = {&observations[0], nullptr, &observations[1]};
    owner.initialized_ = true;
    if constexpr (requires { owner.collective_performed_; }) owner.collective_performed_ = true;
    cpu_set_t expected = outer_expected;
    if (nested) {
#pragma omp parallel num_threads(1)
        {
            expected = nativeMask();
            for (auto &value : observations) value.schedule.creator_mask = expected;
            try { owner.abortCommunicators(); }
            catch (...) { fail("any-thread abort rejected an active OpenMP caller"); }
            const auto restored = nativeMask();
            require(CPU_EQUAL(&restored, &expected), "nested abort changed its assigned caller mask");
        }
    } else owner.abortCommunicators();
    for (const auto &value : observations) {
        require(value.calls.load() == 1, "sparse abort handles were skipped or submitted twice");
        requireControlObservation(value.schedule, expected);
    }
    require(std::all_of(owner.comms_.begin(), owner.comms_.end(), [](void *value) { return !value; }),
            "abort cleanup retained a live handle");
    require(!owner.running_.load() && !owner.initialized_.load(), "abort left its owner active");
    owner.abortCommunicators(); // The absorbing empty state must not create another helper.
    for (const auto &value : observations) require(value.calls.load() == 1, "empty abort recreated native work");
    if constexpr (requires { owner.collective_performed_; }) {
        owner.comms_ = {&observations[0], &observations[1]};
        owner.collective_performed_ = false;
        owner.abortCommunicators();
        require(owner.comms_[0] && owner.comms_[1], "unused RCCL abort invalidated ordinary-finalization handles");
        for (const auto &value : observations) require(value.calls.load() == 1, "unused RCCL owner entered native abort");
    }
    const auto restored = nativeMask();
    require(CPU_EQUAL(&restored, &original), "cold abort failed to restore outer caller placement");
}
/** @brief Qualify the any-thread cold abort contract under outer and active OpenMP callers. */
template<class Coordinator> void checkAbortAffinity() {
    const auto expected = admittedControlMask();
    for (int iteration = 0; iteration < 20; ++iteration) {
        checkAbortControl<Coordinator>(expected, false);
        checkAbortControl<Coordinator>(expected, true);
    }
}
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
    if (name == "abort-affinity") {
        if (backend == "NCCL") checkAbortAffinity<llaminar2::NCCLCoordinator>();
        else if (backend == "RCCL") checkAbortAffinity<llaminar2::RCCLCoordinator>();
        else fail("unknown abort-affinity backend");
        std::printf("PASS actual %s abort affinity: sparse/empty/unused lifetimes; outer and active OpenMP callers\n", argv[1]);
        return 0;
    }
    if (name == "affinity") {
        checkControlAffinity(backend);
        std::printf("PASS actual %s control affinity: native masks/defaults/descendants/creator restoration/nested rejection\n", argv[1]);
        return 0;
    }
    require(name == "stop" || name == "gap-success" || name == "gap-failure" || name == "lifecycle",
            "unknown replay schedule");
    const auto mode = name == "stop" ? Mode::Stop : name.starts_with("gap-") ? Mode::Gap : Mode::Lifecycle;
    const bool success = name != "gap-failure";
    if (backend == "NCCL") replay<llaminar2::NCCLCoordinator>(mode, success, 20);
    else if (backend == "RCCL") replay<llaminar2::RCCLCoordinator>(mode, success, 20);
    else fail("unknown backend");
    std::printf("PASS actual %s coordinator %s: 20 lifecycles; no SDK/runtime/device\n", argv[1], argv[2]);
}
