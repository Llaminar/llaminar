/**
 * @file thread_replay.cpp
 * @brief Real pthread launch/join and native group failure-boundary replay.
 *
 * Complete actual group.cc is compiled below. Only pthread scheduling/failure injection
 * and inactive GPU cleanup references are test seams. Actual ordinary native job
 * allocations remain queued until real joins; no admitted-storage interface is used.
 */
#include "group.h"
#include "debug.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>

using Queue = ncclIntruQueue<ncclAsyncJob, &ncclAsyncJob::next>;
extern "C" ncclResult_t qfn_actual_launch(Queue*, volatile bool*);
extern "C" void qfn_actual_cleanup(Queue*, ncclResult_t);
extern void* ncclAsyncJobMain(void*);
int ncclDebugLevel = 0;
unsigned long ncclDebugMask = 0;
thread_local int ncclDebugNoWarn = 0;
__thread ncclThreadSignal ncclThreadSignalLocalInstance = ncclThreadSignalStaticInitializer();
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {
}
#ifndef QFN_ORIGINAL_SOURCE
#include "actual_ensure_ready.inc"
extern "C" ncclResult_t ncclCommGetAsyncError(ncclComm_t, ncclResult_t*) { std::abort(); }
#endif
/** @brief This replay has no nonblocking comm heads; entering that branch is unsupported. */
ncclResult_t ncclCommSetAsyncError(ncclComm*, ncclResult_t) {
  std::abort();
}

namespace {
enum class Mode {
  DelayedEntry,
  WaitingForAbort,
  ReturnedBeforeCreate,
  AllReturned,
  SingleDirect,
  JoinFailure,
  TwoCreatedThenFailure
};
Mode mode;
std::mutex scheduleMutex;
std::condition_variable schedule;
bool mayEnter = false;
std::atomic<size_t> creates = 0, joins = 0, entered = 0, returned = 0, undone = 0, destroyed = 0;
std::atomic<size_t> abortObserved = 0;
std::atomic<size_t> assertions = 0;

bool injectJoinFailure = false;
int expectedFatal = 0;
size_t expectedJobs = 0;
ncclGroupJob* expectedGroup = nullptr;
bool returnWorkerError = false;

/** @brief Stop at the exact source ownership invariant without inventing progress. */
void check(bool condition, const char* message) {
  ++assertions;
  if (!condition) {
    std::cerr << "REJECT " << message << std::endl;
    _exit(87);
  }
}

/** @brief Actual queued metadata root; source intrusive queue retains its lifetime. */
struct Job : ncclAsyncJob {
  uint32_t abort = 0;
  uint32_t abortDevice = 0;
  size_t index = 0;
};

/** @brief Worker body can retire only after the real native abort field is published. */
ncclResult_t body(ncclAsyncJob* base) {
    auto* job = base;
  ++entered;
  if (mode == Mode::DelayedEntry || mode == Mode::WaitingForAbort || mode == Mode::JoinFailure ||
      mode == Mode::TwoCreatedThenFailure) {
    while (!__atomic_load_n(job->abortFlag, __ATOMIC_ACQUIRE))
      std::this_thread::yield();
    check(__atomic_load_n(job->abortFlag, __ATOMIC_ACQUIRE) == 1,
          "real native abort signal observed before worker retirement");
    ++abortObserved;
  }
  return returnWorkerError ? ncclSystemError : ncclSuccess;
}

/** @brief Undo cannot read any job/comm metadata while a created worker remains unjoined. */
void undo(ncclAsyncJob*) {
  check(joins.load() == creates.load(), "undo before all created workers joined");
  ++undone;
}

/** @brief Retire actual metadata backing only after all native joins, exactly once. */
void destroy(void* base) {
  check(joins.load() == creates.load(), "cleanup before all created workers joined");
  auto* job = static_cast<Job*>(base);
  ++destroyed;
  std::free(job);
}

struct Start {
  void* (*function)(void*);
  void* argument;
};
std::array<Start, 3> starts;

/** @brief Hold only the native worker entry, without changing actual worker code. */
void* trampoline(void* argument) {
  auto* start = static_cast<Start*>(argument);
  if (mode == Mode::DelayedEntry) {
    std::unique_lock lock(scheduleMutex);
    schedule.wait(lock, [] { return mayEnter; });
  }
  auto* result = start->function(start->argument);
  ++returned;
  return result;
}
}  // namespace

/** @brief Real pthread creation plus deterministic later-create failure/scheduling. */
extern "C" int qfn_pthread_create(pthread_t* thread, const pthread_attr_t* attributes,
                                  void* (*function)(void*), void* argument) noexcept {
  const size_t index = creates.load();
  const size_t successfulCreates =
      mode == Mode::TwoCreatedThenFailure || mode == Mode::JoinFailure ? 2 : 1;
  if (index == successfulCreates && mode != Mode::AllReturned)
    return EAGAIN;
  check(index < starts.size(), "actual declared thread count");
  starts[index] = {function, argument};
  const int result = pthread_create(thread, attributes, trampoline, &starts[index]);
  if (result)
    return result;
  ++creates;
  if (mode == Mode::WaitingForAbort || mode == Mode::JoinFailure ||
      mode == Mode::TwoCreatedThenFailure)
    while (entered.load() != creates.load())
      std::this_thread::yield();
  if (mode == Mode::ReturnedBeforeCreate || mode == Mode::AllReturned)
    while (returned.load() != creates.load())
      std::this_thread::yield();
  return 0;
}

/** @brief Real join; delayed entry is released only after native abort publication. */
extern "C" int qfn_pthread_join(pthread_t thread, void** result) noexcept {
  check(destroyed == 0 && undone == 0, "metadata cleanup before real join boundary");
#ifndef QFN_ORIGINAL_SOURCE
  ncclAsyncJob* actual = nullptr;
  for (size_t i = 0; i < creates.load(); ++i) {
    auto* candidate = static_cast<ncclAsyncJob*>(starts[i].argument);
    if (pthread_equal(candidate->thread, thread))
      actual = candidate;
  }
  check(actual && ncclLlaminarLoadThreadOwnership(actual) == ncclNativeThreadOwnership::Created,
        "real created-thread ownership survives independently of worker entry/completion");
  if (mode == Mode::ReturnedBeforeCreate || mode == Mode::AllReturned)
    check(__atomic_load_n(&actual->state, __ATOMIC_ACQUIRE) == ncclGroupJobDone,
          "worker returned before parent publishes successful create receipt");
#endif
  if (mode != Mode::AllReturned) {
    for (size_t i = 0; i < creates.load(); ++i) {
      auto* job = static_cast<ncclAsyncJob*>(starts[i].argument);
      check(__atomic_load_n(job->abortFlag, __ATOMIC_ACQUIRE) == 1 &&
                __atomic_load_n(job->abortFlagDev, __ATOMIC_ACQUIRE) == 1,
            "join before native abort publication");
    }
  }
  {
    std::lock_guard lock(scheduleMutex);
    mayEnter = true;
  }
  schedule.notify_all();
  if (injectJoinFailure) {
    injectJoinFailure = false;
    return EINVAL;
  }
  const int status = pthread_join(thread, result);
  if (status == 0)
    ++joins;
  return status;
}

#ifndef QFN_ORIGINAL_SOURCE
extern "C" [[noreturn]] void __real_abort() noexcept;
/** @brief Observe exact ownership at the actual source fatal boundary, then really abort. */
extern "C" [[noreturn]] void __wrap_abort() noexcept {
  check(expectedFatal != 0 && destroyed == 0 && undone == 0 && joins == 0,
        "actual join fatal precedes every cleanup and ownership release");
  if (expectedFatal == 1) {
    auto* first = static_cast<ncclAsyncJob*>(starts[0].argument);
    check(ncclLlaminarLoadThreadOwnership(first) == ncclNativeThreadOwnership::JoinFailed,
          "actual failing queue job retains its permanent failed-join receipt");
    size_t count = 0;
    for (auto* job = first; job; job = job->next) ++count;
    check(count == expectedJobs, "all actual queue metadata owners remain reachable at fatal exit");
  } else {
    check(expectedGroup && expectedGroup->groupRefCount == 2 &&
              ncclLlaminarLoadThreadOwnership(&expectedGroup->base) == ncclNativeThreadOwnership::JoinFailed,
          "actual shared group metadata/refcounts survive complete and ignored-result abort callers");
  }
  std::cout << "PREFATAL verified actual join ownership kind " << expectedFatal << std::endl;
  __real_abort();
}
#endif

/** @brief Run one actual source launch sequence; failed joins retain all owners at fatal exit. */
void scenario(Mode selected) {
  mode = selected;
  returnWorkerError = false;
  expectedFatal = mode == Mode::JoinFailure ? 1 : 0;
  expectedGroup = nullptr;
  injectJoinFailure = mode == Mode::JoinFailure;
  creates = joins = entered = returned = undone = destroyed = abortObserved = 0;
  mayEnter = false;
  const size_t count = mode == Mode::SingleDirect                                           ? 1
                       : (mode == Mode::TwoCreatedThenFailure || mode == Mode::JoinFailure) ? 3
                                                                                            : 2;
  expectedJobs = count;
  Queue queue;
  ncclIntruQueueConstruct(&queue);
  for (size_t i = 0; i < count; ++i) {
    Job* job = nullptr;
    check(ncclCalloc(&job, 1) == ncclSuccess && job, "actual native job allocation");
    job->index = i;
    job->func = body;
#ifdef QFN_ORIGINAL_SOURCE
    job->undo = nullptr;  // Actual init jobs have no undo; expose the original destructor edge.
#else
    job->undo = undo;
#endif
    job->destructor = destroy;
    job->abortFlag = &job->abort;
    job->abortFlagDev = &job->abortDevice;
    job->state = ncclGroupJobRunning;
    ncclIntruQueueEnqueue(&queue, static_cast<ncclAsyncJob*>(job));
  }
  volatile bool abort = false;
  const auto result = qfn_actual_launch(&queue, &abort);
  if (mode == Mode::JoinFailure) {
    check(false, "real join failure returned past the native fatal boundary");
  }
  const bool complete = mode == Mode::AllReturned || mode == Mode::SingleDirect;
  check((result == ncclSuccess) == complete, "actual later-create failure result");
#ifndef QFN_ORIGINAL_SOURCE
  check(joins == creates && destroyed == 0 && undone == 0, "actual joins precede all cleanup");
  check(entered == creates || mode == Mode::SingleDirect, "exact actual worker entry count");
  if (mode == Mode::DelayedEntry || mode == Mode::WaitingForAbort ||
      mode == Mode::TwoCreatedThenFailure)
    check(abortObserved == creates, "actual body retired only after missing-member abort");
#endif
  qfn_actual_cleanup(&queue, result);
  check(destroyed == count && undone == count, "actual native undo/destructor count");
  check(ncclIntruQueueEmpty(&queue), "all actual native roots retired");
}

#ifndef QFN_ORIGINAL_SOURCE
/** @brief Exercise complete or the actual ignored-result abort caller on a real group worker. */
void failedGroupJoin(bool ignoredCaller) {
  mode = Mode::AllReturned;
  returnWorkerError = false;
  creates = joins = entered = returned = undone = destroyed = abortObserved = 0;
  mayEnter = false;
  ncclGroupJob* group = nullptr;
  check(ncclCalloc(&group, 1) == ncclSuccess && group, "actual native group metadata allocation");
  group->groupRefCount = 2;
  group->nonBlockingInit = true;
  group->base.func = body;
  check(qfn_pthread_create(&group->base.thread, nullptr, ncclAsyncJobMain, &group->base) == 0,
        "actual nonblocking group worker creation");
  ncclLlaminarStoreThreadOwnership(&group->base, ncclNativeThreadOwnership::Created);
  injectJoinFailure = true;
  expectedFatal = ignoredCaller ? 3 : 2;
  expectedGroup = group;
  if (ignoredCaller) {
    uint32_t abort = 1;
    ncclComm comm{};
    comm.abortFlag = &abort;
    comm.groupJob = group;
    (void)ncclCommEnsureReady(&comm);
  } else {
    (void)ncclGroupJobComplete(group);
  }
  check(false, "join failure returned to a native caller that can release ownership");
}

/** @brief Ordinary worker-result error remains returnable after its real successful join. */
void ordinaryGroupError() {
  mode = Mode::AllReturned;
  expectedFatal = 0;
  returnWorkerError = true;
  creates = joins = entered = returned = undone = destroyed = abortObserved = 0;
  ncclGroupJob* group = nullptr;
  check(ncclCalloc(&group, 1) == ncclSuccess && group, "native error-result group allocation");
  group->groupRefCount = 2;
  group->nonBlockingInit = true;
  group->base.func = body;
  check(qfn_pthread_create(&group->base.thread, nullptr, ncclAsyncJobMain, &group->base) == 0,
        "native error-result worker created");
  ncclLlaminarStoreThreadOwnership(&group->base, ncclNativeThreadOwnership::Created);
  check(ncclGroupJobComplete(group) == ncclSystemError && joins == 1 && group->groupRefCount == 1 &&
            ncclLlaminarLoadThreadOwnership(&group->base) == ncclNativeThreadOwnership::Joined,
        "ordinary init result stays returnable after successful join");
  check(ncclGroupJobAbort(group) == ncclSuccess, "ordinary init error permits final joined group retirement");
  returnWorkerError = false;
}
#endif

/** @brief Replay source launch/cleanup and isolated fatal join failures without a device runtime.
 */
int main() {
  for (int replay = 0; replay < 20; ++replay) {
    for (Mode selected : {Mode::DelayedEntry, Mode::WaitingForAbort, Mode::ReturnedBeforeCreate,
                          Mode::AllReturned, Mode::SingleDirect, Mode::TwoCreatedThenFailure})
      scenario(selected);
    const auto child = fork();
    check(child >= 0, "fatal native join-control process");
    if (child == 0) {
      scenario(Mode::JoinFailure);
      _exit(88);
    }
    int status = 0;
    check(waitpid(child, &status, 0) == child && WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
          "failed join forbids all native cleanup and retirement retries");
#ifndef QFN_ORIGINAL_SOURCE
    for (bool ignoredCaller : {false, true}) {
      const auto groupChild = fork();
      check(groupChild >= 0, "native group join-failure process");
      if (groupChild == 0) failedGroupJoin(ignoredCaller);
      int groupStatus = 0;
      check(waitpid(groupChild, &groupStatus, 0) == groupChild && WIFSIGNALED(groupStatus) &&
                WTERMSIG(groupStatus) == SIGABRT,
            "actual native complete/ignored-abort caller cannot retire a failed join");
    }
    ordinaryGroupError();
#endif
  }
  std::cout << "PASS actual native thread ownership: 140 successful-retirement + 60 fatal-join "
               "lifecycles, "
            << assertions
            << " assertions; delayed entry, missing-member abort, returned-before-create receipt, "
               "all joined before cleanup, direct no-thread path, native job roots retired\n";
}

#ifndef NATIVE_GROUP_SOURCE
#error The exact native group implementation must be bound by the dependency builder
#endif
#define pthread_create qfn_pthread_create
#define pthread_join qfn_pthread_join
#include NATIVE_GROUP_SOURCE
#undef pthread_create
#undef pthread_join
/** @brief Exercise the actual private native launch body without replacing its algorithm. */
extern "C" ncclResult_t qfn_actual_launch(Queue* queue, volatile bool* abort) {
  return asyncJobLaunch(queue, abort);
}
/** @brief Exercise the actual private cleanup body with no unrelated active comm heads. */
extern "C" void qfn_actual_cleanup(Queue* queue, ncclResult_t error) {
  ncclComm* heads[ncclGroupTaskTypeNum]{};
  groupCleanup(heads, queue, error);
}
