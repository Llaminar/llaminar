/**
 * @file Test__NativeCollectiveOwnedErrors.cpp
 * @brief Replay actual native enqueue/error methods with CPU entrypoint doubles.
 * No GPU, communicator, backend readiness or transfer is created. Generated
 * definitions contain production decisions; recorded control objects provide
 * only their inspected fields. Original writes are ordered to avoid test UB.
 */
#include <gtest/gtest.h>
#include "collective/NativeCollectiveBytes.h"
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdlib>
#include <future>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#ifndef HAVE_NCCL
#define HAVE_NCCL 1
#endif
#ifndef HAVE_RCCL
#define HAVE_RCCL 1
#endif

namespace diagnostic_runtime {
using llaminar2::NativeCollectiveBytes;
using llaminar2::NativeCollectiveEnqueueResult;
enum class Fault { None, Selection, Producer, Enqueue, AfterEnqueue, EnqueueAndAfter };
enum class CollectiveBackendType { NCCL, RCCL, HOST };
inline thread_local int participant = 0, error_reads = 0;
inline Fault fault = Fault::None;
inline std::barrier<>* entry_gate = nullptr;
inline std::atomic<int> active{0}, maximum_active{0}, native_calls{0};
inline std::array<std::string, 2> logs;
/** @brief Reset CPU diagnostic operands between independent replay trajectories. */
void reset(Fault next)
{
    fault = next;
    entry_gate = nullptr;
    active = 0;
    maximum_active = 0;
    native_calls = 0;
    logs = {};
}
/** @return Stable runtime error text distinct for each original participant. */
const char* errorText(int code)
{
    switch (code / 100) {
    case 1: return code % 2 == 0 ? "selection-original-worker-0" : "selection-original-worker-1";
    case 2: return code % 2 == 0 ? "producer-original-worker-0" : "producer-original-worker-1";
    case 3: return code % 2 == 0 ? "native-original-worker-0" : "native-original-worker-1";
    default: return code % 2 == 0 ? "after-original-worker-0" : "after-original-worker-1";
    }
}
/** @return Recorded selection disposition without native device discovery. */
int selectDevice(int)
{
    return fault == Fault::Selection ? 100 + participant : 0;
}
/** @return Original producer error before enqueue, independently of the later error. */
int readError()
{
    ++error_reads;
    if (fault == Fault::Producer && error_reads == 1)
        return 200 + participant;
    if ((fault == Fault::AfterEnqueue || fault == Fault::EnqueueAndAfter) && error_reads == 2)
        return 400 + participant;
    return 0;
}
/** @brief Prove both actual entries overlap; no native arrival or receipt is fabricated. */
int broadcast(const void*, void*, const NativeCollectiveBytes&, int, void*, void*)
{
    const int now = active.fetch_add(1) + 1;
    int previous = maximum_active.load();
    while (previous < now && !maximum_active.compare_exchange_weak(previous, now)) {}
    ++native_calls;
    if (entry_gate)
        entry_gate->arrive_and_wait();
    --active;
    return (fault == Fault::Enqueue || fault == Fault::EnqueueAndAfter) ? 300 + participant : 0;
}
} // namespace diagnostic_runtime
#define LOG_ERROR(expression) do { std::ostringstream text; text << expression; \
    diagnostic_runtime::logs[diagnostic_runtime::participant] = text.str(); } while (false)

namespace current_nccl {
using namespace diagnostic_runtime;
inline constexpr int cudaSuccess = 0, hipSuccess = 0;
inline int cudaSetDevice(int device) { return selectDevice(device); }
inline int selectHipDevice(int device) { return selectDevice(device); }
inline int cudaGetLastError() { return readError(); }
inline int hipGetLastError() { return readError(); }
inline const char* cudaGetErrorString(int code) { return errorText(code); }
inline const char* hipGetErrorString(int code) { return errorText(code); }
namespace nccl {
using ncclComm_t = void*;
inline constexpr int ncclSuccess = 0;
inline int nativeBroadcastBytes(const void* send, void* receive,
    const NativeCollectiveBytes& bytes, int root, void* comm, void* stream)
{ return broadcast(send, receive, bytes, root, comm, stream); }
inline const char* ncclGetErrorString(int code) { return errorText(code); }
}
/** @brief Recorded host fields only; never an actual native coordinator/readiness claim. */
class NCCLCoordinator {
public:
    std::atomic<bool> initialized_{true}, collective_performed_{false};
    int num_devices_ = 2;
    std::vector<int> device_ordinals_{0, 1};
    std::vector<void*> comms_{reinterpret_cast<void*>(0x1000), reinterpret_cast<void*>(0x2000)};
    std::string last_error_ = "legacy-coordinator-sentinel";
    const std::string& lastError() const { return last_error_; }
    NativeCollectiveEnqueueResult nativeBroadcastBytesOnStream(const void*, void*, const NativeCollectiveBytes&, int, int, void*);
};
/** @brief Recorded backend fields for the actual production host wrapper. */
class NCCLBackend {
public:
    bool initialized_ = true, is_multi_gpu_single_process_ = true;
    std::shared_ptr<NCCLCoordinator> coordinator_ = std::make_shared<NCCLCoordinator>();
    std::string last_error_ = "legacy-backend-sentinel";
    bool isMultiGpuSingleProcess() const { return is_multi_gpu_single_process_; }
    NativeCollectiveEnqueueResult nativeBroadcastBytesOnStream(
        const void*, void*, const NativeCollectiveBytes&, int, int, void*);
};
/** @brief Host diagnostic scope; actual abort publication is covered by its original-owner suite. */
class LocalTPContext {
public:
    std::atomic<bool> abort_requested_{false};
    bool backend_initialized_ = true;
    std::shared_ptr<NCCLBackend> backend_impl_ = std::make_shared<NCCLBackend>();
    CollectiveBackendType backend_ = CollectiveBackendType::NCCL;
    int degree() const { return 2; }
    void requestAbort() { abort_requested_ = true; }
    bool nativeBroadcastBytesOnStream(const void*, void*, const NativeCollectiveBytes&,
        int, int, void*, const std::string&);
};
}

namespace current_rccl {
using namespace diagnostic_runtime;
inline constexpr int cudaSuccess = 0, hipSuccess = 0;
inline int cudaSetDevice(int device) { return selectDevice(device); }
inline int selectHipDevice(int device) { return selectDevice(device); }
inline int cudaGetLastError() { return readError(); }
inline int hipGetLastError() { return readError(); }
inline const char* cudaGetErrorString(int code) { return errorText(code); }
inline const char* hipGetErrorString(int code) { return errorText(code); }
namespace rccl {
using ncclComm_t = void*;
inline constexpr int ncclSuccess = 0;
inline int nativeBroadcastBytes(const void* send, void* receive,
    const NativeCollectiveBytes& bytes, int root, void* comm, void* stream)
{ return broadcast(send, receive, bytes, root, comm, stream); }
inline const char* ncclGetErrorString(int code) { return errorText(code); }
}
/** @brief Recorded host fields only; never an actual native coordinator/readiness claim. */
class RCCLCoordinator {
public:
    std::atomic<bool> initialized_{true}, collective_performed_{false};
    int num_devices_ = 2;
    std::vector<int> device_ordinals_{0, 1};
    std::vector<void*> comms_{reinterpret_cast<void*>(0x1000), reinterpret_cast<void*>(0x2000)};
    std::string last_error_ = "legacy-coordinator-sentinel";
    const std::string& lastError() const { return last_error_; }
    NativeCollectiveEnqueueResult nativeBroadcastBytesOnStream(const void*, void*, const NativeCollectiveBytes&, int, int, void*);
};
/** @brief Recorded backend fields for the actual production host wrapper. */
class RCCLBackend {
public:
    bool initialized_ = true, is_multi_gpu_single_process_ = true;
    std::shared_ptr<RCCLCoordinator> coordinator_ = std::make_shared<RCCLCoordinator>();
    std::string last_error_ = "legacy-backend-sentinel";
    bool isMultiGpuSingleProcess() const { return is_multi_gpu_single_process_; }
    NativeCollectiveEnqueueResult nativeBroadcastBytesOnStream(
        const void*, void*, const NativeCollectiveBytes&, int, int, void*);
};
/** @brief Host diagnostic scope; actual abort publication is covered by its original-owner suite. */
class LocalTPContext {
public:
    std::atomic<bool> abort_requested_{false};
    bool backend_initialized_ = true;
    std::shared_ptr<RCCLBackend> backend_impl_ = std::make_shared<RCCLBackend>();
    CollectiveBackendType backend_ = CollectiveBackendType::RCCL;
    int degree() const { return 2; }
    void requestAbort() { abort_requested_ = true; }
    bool nativeBroadcastBytesOnStream(const void*, void*, const NativeCollectiveBytes&,
        int, int, void*, const std::string&);
};
}

namespace original_nccl {
using namespace diagnostic_runtime;
inline constexpr int cudaSuccess = 0, hipSuccess = 0;
inline int cudaSetDevice(int device) { return selectDevice(device); }
inline int selectHipDevice(int device) { return selectDevice(device); }
inline int cudaGetLastError() { return readError(); }
inline int hipGetLastError() { return readError(); }
inline const char* cudaGetErrorString(int code) { return errorText(code); }
inline const char* hipGetErrorString(int code) { return errorText(code); }
namespace nccl {
using ncclComm_t = void*;
inline constexpr int ncclSuccess = 0;
inline int nativeBroadcastBytes(const void* send, void* receive,
    const NativeCollectiveBytes& bytes, int root, void* comm, void* stream)
{ return broadcast(send, receive, bytes, root, comm, stream); }
inline const char* ncclGetErrorString(int code) { return errorText(code); }
}
/** @brief Recorded host fields only; never an actual native coordinator/readiness claim. */
class NCCLCoordinator {
public:
    std::atomic<bool> initialized_{true}, collective_performed_{false};
    int num_devices_ = 2;
    std::vector<int> device_ordinals_{0, 1};
    std::vector<void*> comms_{reinterpret_cast<void*>(0x1000), reinterpret_cast<void*>(0x2000)};
    std::string last_error_ = "legacy-coordinator-sentinel";
    const std::string& lastError() const { return last_error_; }
    bool nativeBroadcastBytesOnStream(const void*, void*, const NativeCollectiveBytes&, int, int, void*);
};
}

namespace original_rccl {
using namespace diagnostic_runtime;
inline constexpr int cudaSuccess = 0, hipSuccess = 0;
inline int cudaSetDevice(int device) { return selectDevice(device); }
inline int selectHipDevice(int device) { return selectDevice(device); }
inline int cudaGetLastError() { return readError(); }
inline int hipGetLastError() { return readError(); }
inline const char* cudaGetErrorString(int code) { return errorText(code); }
inline const char* hipGetErrorString(int code) { return errorText(code); }
namespace rccl {
using ncclComm_t = void*;
inline constexpr int ncclSuccess = 0;
inline int nativeBroadcastBytes(const void* send, void* receive,
    const NativeCollectiveBytes& bytes, int root, void* comm, void* stream)
{ return broadcast(send, receive, bytes, root, comm, stream); }
inline const char* ncclGetErrorString(int code) { return errorText(code); }
}
/** @brief Recorded host fields only; never an actual native coordinator/readiness claim. */
class RCCLCoordinator {
public:
    std::atomic<bool> initialized_{true}, collective_performed_{false};
    int num_devices_ = 2;
    std::vector<int> device_ordinals_{0, 1};
    std::vector<void*> comms_{reinterpret_cast<void*>(0x1000), reinterpret_cast<void*>(0x2000)};
    std::string last_error_ = "legacy-coordinator-sentinel";
    const std::string& lastError() const { return last_error_; }
    bool nativeBroadcastBytesOnStream(const void*, void*, const NativeCollectiveBytes&, int, int, void*);
};
}

#include "NativeCollectiveDiagnosticBodies.inc"
#include "../../fixtures/collective/OriginalNativeCollectiveErrors.inc"
#undef LOG_ERROR
namespace {
using namespace diagnostic_runtime;
/** @brief Exercise actual backend+coordinator failures; read every outcome after both calls. */
template<class Backend> void concurrentFailures()
{
    Backend backend;
    const auto bytes = NativeCollectiveBytes::fixed(8);
    std::array<std::byte, 8> buffer{};
    for (const auto next : {Fault::Selection, Fault::Producer, Fault::Enqueue, Fault::AfterEnqueue, Fault::EnqueueAndAfter, Fault::None}) {
        reset(next);
        std::barrier gate(2);
        if (next == Fault::Enqueue || next == Fault::AfterEnqueue || next == Fault::EnqueueAndAfter || next == Fault::None)
            entry_gate = &gate;
        std::array<std::optional<NativeCollectiveEnqueueResult>, 2> results;
        std::array<std::jthread, 2> workers;
        for (int index = 0; index < 2; ++index) {
            workers[index] = std::jthread([&, index] {
                participant = index;
                error_reads = 0;
                results[index] = backend.nativeBroadcastBytesOnStream(buffer.data(), buffer.data(),
                    bytes, 1, index, reinterpret_cast<void*>(0x100));
            });
        }
        for (auto& worker : workers)
            worker.join();
        entry_gate = nullptr;
        for (int index = 0; index < 2; ++index) {
            ASSERT_TRUE(results[index]);
            if (next == Fault::None) {
                EXPECT_TRUE(static_cast<bool>(*results[index]));
                EXPECT_TRUE(results[index]->message().empty());
            } else {
                EXPECT_FALSE(static_cast<bool>(*results[index]));
                EXPECT_NE(results[index]->message().find("original-worker-" + std::to_string(index)), std::string::npos);
                if (next == Fault::EnqueueAndAfter) {
                    EXPECT_NE(results[index]->message().find("native-original-worker-"), std::string::npos);
                    EXPECT_EQ(results[index]->message().find("after-original-worker-"), std::string::npos);
                }
                if (next == Fault::Producer)
                    EXPECT_NE(results[index]->message().find("original producer"), std::string::npos);
            }
        }
        if (next == Fault::Selection || next == Fault::Producer)
            EXPECT_EQ(native_calls.load(), 0);
        else
            EXPECT_EQ(maximum_active.load(), 2);
        const auto retained = *results[0];
        const auto original = retained.message();
        reset(Fault::Enqueue);
        participant = 1;
        error_reads = 0;
        EXPECT_FALSE(static_cast<bool>(backend.nativeBroadcastBytesOnStream(buffer.data(), buffer.data(),
            bytes, 1, 1, reinterpret_cast<void*>(0x100))));
        EXPECT_EQ(retained.message(), original);
        EXPECT_EQ(backend.last_error_, "legacy-backend-sentinel");
        EXPECT_EQ(backend.coordinator_->last_error_, "legacy-coordinator-sentinel");
    }
}
/** @brief Exercise the actual LocalTP method's readback after simultaneous native failures. */
template<class Context> void contextReadback()
{
    reset(Fault::Enqueue);
    Context context;
    std::barrier gate(2);
    entry_gate = &gate;
    std::array<bool, 2> succeeded{};
    std::array<std::byte, 8> buffer{};
    const auto bytes = NativeCollectiveBytes::fixed(8);
    std::array<std::jthread, 2> workers;
    for (int index = 0; index < 2; ++index) {
        workers[index] = std::jthread([&, index] {
            participant = index;
            error_reads = 0;
            succeeded[index] = context.nativeBroadcastBytesOnStream(buffer.data(), buffer.data(), bytes,
                1, index, reinterpret_cast<void*>(0x100), "actual_native_diagnostic_replay");
        });
    }
    for (auto& worker : workers)
        worker.join();
    entry_gate = nullptr;
    EXPECT_EQ(maximum_active.load(), 2);
    for (int index = 0; index < 2; ++index) {
        EXPECT_FALSE(succeeded[index]);
        EXPECT_NE(logs[index].find("original-worker-" + std::to_string(index)), std::string::npos);
    }
    EXPECT_TRUE(context.abort_requested_.load());
}
/** @return Original crossed readback with ordered writes, never undefined concurrent string mutation. */
template<class Coordinator> std::array<std::string, 2> originalReadback()
{
    reset(Fault::Enqueue);
    Coordinator coordinator;
    std::promise<void> first, second;
    auto first_done = first.get_future();
    auto second_done = second.get_future();
    std::array<std::string, 2> observed;
    std::array<std::byte, 8> buffer{};
    const auto bytes = NativeCollectiveBytes::fixed(8);
    std::jthread earlier([&] {
        participant = 0;
        error_reads = 0;
        if (coordinator.nativeBroadcastBytesOnStream(buffer.data(), buffer.data(), bytes, 1, 0,
                reinterpret_cast<void*>(0x100)))
            return;
        first.set_value();
        if (second_done.wait_for(std::chrono::seconds(1)) == std::future_status::ready)
            observed[0] = coordinator.lastError();
    });
    std::jthread later([&] {
        if (first_done.wait_for(std::chrono::seconds(1)) != std::future_status::ready)
            return;
        participant = 1;
        error_reads = 0;
        if (coordinator.nativeBroadcastBytesOnStream(buffer.data(), buffer.data(), bytes, 1, 1,
                reinterpret_cast<void*>(0x100)))
            return;
        observed[1] = coordinator.lastError();
        second.set_value();
    });
    earlier.join();
    later.join();
    return observed;
}
}
TEST(NativeCollectiveOwnedErrors, ConcurrentFailuresKeepOriginalOwnedReadback)
{
    concurrentFailures<current_nccl::NCCLBackend>();
    concurrentFailures<current_rccl::RCCLBackend>();
}
TEST(NativeCollectiveOwnedErrors, ActualLocalTPReadsEachOriginalFailure)
{
    contextReadback<current_nccl::LocalTPContext>();
    contextReadback<current_rccl::LocalTPContext>();
}
TEST(NativeCollectiveOwnedErrors, OriginalBodyLosesDeferredCallerError)
{
    for (const auto& observed : {originalReadback<original_nccl::NCCLCoordinator>(),
                               originalReadback<original_rccl::RCCLCoordinator>()}) {
        ASSERT_FALSE(observed[0].empty());
        ASSERT_FALSE(observed[1].empty());
        if (std::getenv("LLAMINAR_NATIVE_OWNED_ERROR_ORIGINAL_CONTROL"))
            EXPECT_NE(observed[0].find("original-worker-0"), std::string::npos);
        else
            EXPECT_EQ(observed[0].find("original-worker-0"), std::string::npos);
        EXPECT_EQ(observed[0], observed[1]);
    }
}
/** @brief The actual wrapper result owns its cause after the recorded host owner dies. */
template<class Backend> void ownerLifetime()
{
    std::optional<NativeCollectiveEnqueueResult> retained;
    reset(Fault::Enqueue);
    participant = 0;
    error_reads = 0;
    {
        Backend owner;
        std::array<std::byte, 8> buffer{};
        retained = owner.nativeBroadcastBytesOnStream(buffer.data(), buffer.data(),
            NativeCollectiveBytes::fixed(8), 1, 0, reinterpret_cast<void*>(0x100));
    }
    ASSERT_TRUE(retained);
    EXPECT_FALSE(static_cast<bool>(*retained));
    EXPECT_NE(retained->message().find("native-original-worker-0"), std::string::npos);
}
TEST(NativeCollectiveOwnedErrors, ResultOutlivesOriginalHostDiagnosticOwner)
{
    ownerLifetime<current_nccl::NCCLBackend>();
    ownerLifetime<current_rccl::RCCLBackend>();
}
TEST(NativeCollectiveOwnedErrors, FailureValuesRequireOriginalDiagnostic)
{
    EXPECT_THROW((void)NativeCollectiveEnqueueResult::failure(""), std::invalid_argument);
    const auto value = NativeCollectiveEnqueueResult::failure("first original failure");
    const auto copied = value;
    EXPECT_EQ(copied.message(), "first original failure");
    auto source = NativeCollectiveEnqueueResult::failure("owned moved failure");
    const auto moved = std::move(source);
    EXPECT_FALSE(static_cast<bool>(source));
    EXPECT_FALSE(static_cast<bool>(moved));
    EXPECT_EQ(moved.message(), "owned moved failure");
    EXPECT_TRUE(static_cast<bool>(NativeCollectiveEnqueueResult::success()));
}
