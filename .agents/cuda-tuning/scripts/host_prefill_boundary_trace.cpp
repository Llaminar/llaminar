/**
 * @file host_prefill_boundary_trace.cpp
 * @brief Diagnostic ELF interposition for production host prefill boundaries.
 *
 * This helper never invokes a GPU API or changes the execution policy. Fixed
 * storage records wall intervals only inside the existing benchmark prefill.
 * Calls may include queued GPU progress; durations are not CPU service time.
 * Symbol spellings must be checked against the measured core before use.
 */
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {
struct Record {
    const char *name{};
    void *owner{};
    uint64_t start{}, end{}, bytes{}, request{};
    long thread{};
};
std::array<Record, 8192> records;
std::atomic<unsigned> record_count{};
std::atomic<uint64_t> request_count{}, active_request{};

/** Resolve the original symbol; missing ABI is a diagnostic failure. */
template <class Function> Function resolve(const char *symbol) {
    void *address = dlsym(RTLD_NEXT, symbol);
    if (!address) {
        std::fprintf(stderr, "host observer cannot resolve %s: %s\n", symbol, dlerror());
        std::abort();
    }
    return reinterpret_cast<Function>(address);
}

/** Monotonic host clock, independent of every GPU event clock. */
uint64_t now() {
    timespec stamp{};
    if (clock_gettime(CLOCK_MONOTONIC, &stamp) != 0) std::abort();
    return static_cast<uint64_t>(stamp.tv_sec) * 1000000000ULL + stamp.tv_nsec;
}

/** Reserve one fixed record per participating host call, without hot malloc. */
class Span {
public:
    /**
     * @brief Start a span only when a benchmark prefill owns this process.
     * @param name Stable diagnostic boundary, not an engine lifecycle value.
     * @param owner Borrowed receiver address used only to distinguish instances.
     * @param bytes Requested allocation size, or zero for a non-allocation call.
     */
    Span(const char *name, void *owner, uint64_t bytes = 0) {
        const auto request = active_request.load(std::memory_order_relaxed);
        if (!request) return;
        const unsigned index = record_count.fetch_add(1, std::memory_order_relaxed);
        if (index >= records.size()) std::abort();
        record_ = &records[index];
        *record_ = {.name=name, .owner=owner, .start=now(), .bytes=bytes,
                    .request=request, .thread=syscall(SYS_gettid)};
    }
    /** @brief Close the interval without file I/O or GPU synchronization. */
    ~Span() { if (record_) record_->end = now(); }
private:
    Record *record_{};
};

/** Export only after normal engine retirement; inference performs no log I/O. */
__attribute__((destructor)) void export_records() {
    const char *path = std::getenv("LLAMINAR_HOST_BOUNDARY_OUTPUT");
    if (!path || record_count.load() == 0) return;
    FILE *file = std::fopen(path, "w");
    if (!file) std::abort();
    for (unsigned i = 0; i < record_count.load(); ++i) {
        const auto &r = records[i];
        if (!r.end) std::abort();
        std::fprintf(file,
            "{\"name\":\"%s\",\"request\":%llu,\"owner\":\"%p\","
            "\"thread\":%ld,\"start_ns\":%llu,\"end_ns\":%llu,"
            "\"duration_ms\":%.9f,\"bytes\":%llu}\n",
            r.name, static_cast<unsigned long long>(r.request), r.owner, r.thread,
            static_cast<unsigned long long>(r.start), static_cast<unsigned long long>(r.end),
            static_cast<double>(r.end-r.start)/1e6, static_cast<unsigned long long>(r.bytes));
    }
    if (std::fclose(file) != 0) std::abort();
}
}

/** @brief Bracket the original benchmark call, including its terminal wait. */
extern "C" std::pair<bool, double> observe_benchmark(void *, const std::vector<int> &)
    asm("_ZN9llaminar215BenchmarkRunner10runPrefillERKSt6vectorIiSaIiEE");
extern "C" std::pair<bool, double> observe_benchmark(void *self, const std::vector<int> &tokens) {
    static auto call = resolve<decltype(&observe_benchmark)>(
        "_ZN9llaminar215BenchmarkRunner10runPrefillERKSt6vectorIiSaIiEE");
    const auto request = request_count.fetch_add(1) + 1;
    if (active_request.exchange(request) != 0) std::abort();
    std::pair<bool, double> result;
    { Span span("benchmark_prefill", self); result = call(self, tokens); }
    active_request.store(0);
    return result;
}

// The measured core exports these exact C++ member ABI symbols. Receivers and
// reference arguments are borrowed unchanged; no engine objects are fabricated.
/** @brief Forward a token-vector member call with its exact unchanged ABI. */
#define HOST_BOOL_VECTOR(wrapper, symbol, label) \
    extern "C" bool wrapper(void *, const std::vector<int> &) asm(symbol); \
    extern "C" bool wrapper(void *self, const std::vector<int> &tokens) { \
        static auto call = resolve<decltype(&wrapper)>(symbol); \
        Span span(label, self); return call(self, tokens); \
    }
HOST_BOOL_VECTOR(observe_prefill,
    "_ZN9llaminar219OrchestrationRunner7prefillERKSt6vectorIiSaIiEE", "runner_prefill")

/** @brief Time an existing completion wait; never introduce another wait. */
#define HOST_BOOL_EMPTY(wrapper, symbol, label) \
    extern "C" bool wrapper(void *) asm(symbol); \
    extern "C" bool wrapper(void *self) { \
        static auto call = resolve<decltype(&wrapper)>(symbol); \
        Span span(label, self); return call(self); \
    }
HOST_BOOL_EMPTY(observe_terminal_wait,
    "_ZN9llaminar219OrchestrationRunner42waitForLastInferenceCompletionForBenchmarkEv",
    "benchmark_terminal_wait")
HOST_BOOL_EMPTY(observe_device_wait,
    "_ZN9llaminar223DeviceGraphOrchestrator42waitForLastInferenceCompletionForBenchmarkEv",
    "device_terminal_wait")

/** @brief Measure the backend allocation call, including any native waiting. */
#define HOST_PIN(wrapper, symbol, label) \
    extern "C" void *wrapper(void *, size_t, int) asm(symbol); \
    extern "C" void *wrapper(void *self, size_t bytes, int ordinal) { \
        static auto call = resolve<decltype(&wrapper)>(symbol); \
        Span span(label, self, bytes); return call(self, bytes, ordinal); \
    }
HOST_PIN(observe_cuda_pin, "_ZN9llaminar211CUDABackend14allocatePinnedEmi", "cuda_pinned_allocate")
HOST_PIN(observe_rocm_pin, "_ZN9llaminar211ROCmBackend14allocatePinnedEmi", "rocm_pinned_allocate")

/** @brief Borrow the original lookup/tokens unchanged through prefix harvest. */
#define HOST_HARVEST(wrapper, symbol, label) \
    extern "C" bool wrapper(void *, const void *, const std::vector<int> &, int) asm(symbol); \
    extern "C" bool wrapper(void *self, const void *hit, const std::vector<int> &tokens, int count) { \
        static auto call = resolve<decltype(&wrapper)>(symbol); \
        Span span(label, self); return call(self, hit, tokens, count); \
    }
HOST_HARVEST(observe_rank_harvest,
    "_ZN9llaminar216RankOrchestrator13harvestPrefixERKNS_18PrefixLookupResultERKSt6vectorIiSaIiEEi",
    "rank_harvest")
HOST_HARVEST(observe_device_harvest,
    "_ZN9llaminar223DeviceGraphOrchestrator13harvestPrefixERKNS_18PrefixLookupResultERKSt6vectorIiSaIiEEi",
    "device_harvest")

/** @brief Distinguish Qwen MoE runtime archival from nested storage allocation. */
extern "C" bool observe_runtime(void *, std::vector<uint8_t> &, void *)
    asm("_ZN9llaminar214Qwen35MoEGraph30capturePrefixCacheRuntimeStateERSt6vectorIhSaIhEEPv");
extern "C" bool observe_runtime(void *self, std::vector<uint8_t> &state, void *stream) {
    static auto call = resolve<decltype(&observe_runtime)>(
        "_ZN9llaminar214Qwen35MoEGraph30capturePrefixCacheRuntimeStateERSt6vectorIhSaIhEEPv");
    Span span("model_runtime_archive", self);
    return call(self, state, stream);
}
