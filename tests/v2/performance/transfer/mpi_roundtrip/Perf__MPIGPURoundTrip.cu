/**
 * @file Perf__MPIGPURoundTrip.cu
 * @brief Explicit allocation candidates through one retained GPU/MPI round trip.
 *
 * The experiment captures one cooperative GPU program on each actual host.
 * The initiating GPU produces every message, the peer GPU checks and adds its
 * own contribution, and both consume the complete result. The explicit MPI or
 * direct-verbs follower transports the sealed payload and publishes completion only. There
 * are no per-transaction launches, CUDA copies, synchronization or host numeric
 * reductions. Identical GPU checks and computation apply to every candidate.
 *
 * Raw native allocation is confined to this standalone diagnostic fixture; it
 * does not bypass or implement production TransferEngine/PMA admission. Results
 * establish component cost, not a production collective/capture certificate.
 * The fixed two-host fixture must never be promoted as an engine degree limit.
 */
#include "RoundTripContract.h"
#if defined(ROUNDTRIP_VERBS)
#include "VerbsRoundTrip.h"
#endif
#include "utils/MPISessionOwner.h"
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace rt = llaminar2::mpi_roundtrip;
namespace cg = cooperative_groups;
namespace {
constexpr float guardValue = -12345.0f;
constexpr unsigned threads = 256;
/** @brief Throw an exact CUDA diagnostic; failures never change the candidate. */
void cudaCheck(cudaError_t value, const char* operation) {
    if (value != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(value));
}
/** @brief Preserve a native MPI error rather than continuing with partial bytes. */
void mpiCheck(int value, const char* operation) {
    if (value != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING]; int length = 0;
        MPI_Error_string(value, message, &length);
        throw std::runtime_error(std::string(operation) + ": " + std::string(message, length));
    }
}
/** @brief Separate ownership/cache lines for producer and transport publications. */
struct Control {
    alignas(64) unsigned long long txReady = 0;
    alignas(64) unsigned long long txRetired = 0;
    alignas(64) unsigned long long rxReady = 0;
    alignas(64) unsigned long long abort = 0;
};
/** @brief Device-only oracle and cooperative stop state, collected after retirement. */
struct DeviceResult {
    unsigned long long errors = 0;
    unsigned stop = 0;
};
/** @brief Acquire a system-visible epoch from its one physical writer. */
__device__ unsigned long long acquire(unsigned long long& word) {
    return cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>(word).load(cuda::memory_order_acquire);
}
/** @brief Publish GPU payload readiness after every producer issued its system fence. */
__device__ void release(unsigned long long& word, unsigned long long value) {
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>(word).store(value, cuda::memory_order_release);
}
/** @brief Read the native device global timer; only same-device durations are compared. */
__device__ unsigned long long timestamp() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}
/** @brief One leader waits on protocol metadata; the grid join publishes uniform stop state. */
__device__ bool waitEpoch(cg::grid_group grid, Control* control, DeviceResult* result,
                          unsigned long long& flag, unsigned long long epoch) {
    if (grid.thread_rank() == 0) {
        while (acquire(flag) < epoch && !acquire(control->abort)) __nanosleep(64);
        result->stop = acquire(control->abort) != 0;
    }
    grid.sync();
    return result->stop == 0;
}
/**
 * @brief Retained complete synthetic reduction/return with live-sized worker geometry.
 * @details All producer threads fence before the leader publishes its epoch.
 * MPI owns buffers between ready and retired; next-epoch writes wait for its
 * exact send retirement. The host sees no model data or GPU timing array.
 */
__global__ void roundTrip(float* tx, float* rx, float* consumed, std::size_t words,
                          Control* control, DeviceResult* result, unsigned long long* samples,
                          unsigned iterations, bool initiator) {
    auto grid = cg::this_grid();
    for (unsigned epoch = 1; epoch <= iterations; ++epoch) {
        unsigned long long begin = 0;
        if (grid.thread_rank() == 0) begin = timestamp();
        if (!waitEpoch(grid, control, result, control->txRetired, epoch - 1)) return;
        if (!initiator && !waitEpoch(grid, control, result, control->rxReady, epoch)) return;
        for (std::size_t i = grid.thread_rank(); i < words; i += grid.size()) {
            if (initiator) tx[i] = rt::sourceValue(i, epoch);
            else {
                if (rx[i] != rt::sourceValue(i, epoch)) atomicAdd(&result->errors, 1ULL);
                tx[i] = rx[i] + rt::peerValue(i, epoch);
                consumed[i] = tx[i] * 0.5f;
            }
        }
        __threadfence_system();
        grid.sync();
        if (grid.thread_rank() == 0) release(control->txReady, epoch);
        if (initiator) {
            if (!waitEpoch(grid, control, result, control->rxReady, epoch)) return;
            for (std::size_t i = grid.thread_rank(); i < words; i += grid.size()) {
                if (rx[i] != rt::reducedValue(i, epoch)) atomicAdd(&result->errors, 1ULL);
                consumed[i] = rx[i] * 0.5f;
            }
        }
        grid.sync();
        if (grid.thread_rank() == 0) samples[epoch - 1] = timestamp() - begin;
    }
}
/** @brief Fixture-owned native allocation with guards and an explicit MPI address. */
class Buffer {
public:
    /** @brief Allocate before capture; registered system pages retain their original owner. */
    Buffer(rt::Allocation kind, rt::Geometry geometry, std::size_t alignment = 1)
        : kind_(kind), bytes_(geometry.alignedAllocationBytes(alignment)), words_(geometry.words) {
        if (kind == rt::Allocation::Device) {
            cudaCheck(cudaMalloc(&gpu_, bytes_), "device allocation");
            network_ = gpu_;
        } else if (kind == rt::Allocation::Registered) {
            const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
            const auto rounded = ((bytes_ + page - 1) / page) * page;
            if (posix_memalign(&host_, page, rounded)) throw std::runtime_error("aligned host allocation failed");
            const auto status = cudaHostRegister(host_, rounded, cudaHostRegisterMapped);
            if (status != cudaSuccess) { std::free(host_); host_ = nullptr; cudaCheck(status, "register host pages"); }
            cudaCheck(cudaHostGetDevicePointer(&gpu_, host_, 0), "registered device alias");
            network_ = host_;
        } else {
            const unsigned flags = cudaHostAllocMapped |
                (kind == rt::Allocation::WriteCombined ? cudaHostAllocWriteCombined : 0);
            cudaCheck(cudaHostAlloc(&host_, bytes_, flags), "mapped allocation");
            cudaCheck(cudaHostGetDevicePointer(&gpu_, host_, 0), "mapped device alias");
            network_ = host_;
        }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    /** @brief Retire only after MPI and the retained graph have completely drained. */
    ~Buffer() {
        // An exception can leave an MPI request or cooperative graph owning
        // these bytes. Fail before native release; process teardown, not an
        // optimistic destructor, retires that failed distributed experiment.
        if (std::uncaught_exceptions()) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            std::_Exit(1);
        }
        if (kind_ == rt::Allocation::Device) cudaFree(gpu_);
        else if (kind_ == rt::Allocation::Registered) { cudaHostUnregister(host_); std::free(host_); }
        else cudaFreeHost(host_);
    }
    /** @return Participant-local device view excluding both guards. */
    float* gpu() { return static_cast<float*>(gpu_) + rt::Geometry::guardWords; }
    /** @return Explicit MPI-address-space view excluding both guards. */
    float* network() { return static_cast<float*>(network_) + rt::Geometry::guardWords; }
#if defined(ROUNDTRIP_VERBS)
    /** @return Borrowed views whose original backing outlives the entire native transport. */
    rt::VerbsBufferView verbsView() { return {network_, gpu_, network(), bytes_}; }
#endif
    /** @brief Poison complete storage outside the measured graph. */
    void poison(cudaStream_t stream) {
        std::vector<float> initial(bytes_ / sizeof(float), guardValue);
        cudaCheck(cudaMemcpyAsync(gpu_, initial.data(), bytes_, cudaMemcpyDefault, stream), "fixture poison");
        cudaCheck(cudaStreamSynchronize(stream), "fixture poison retirement");
    }
    /** @brief Check exact unused prefix/tail after all native ownership has retired. */
    void checkGuards(cudaStream_t stream) {
        std::vector<float> values(bytes_ / sizeof(float));
        cudaCheck(cudaMemcpyAsync(values.data(), gpu_, bytes_, cudaMemcpyDefault, stream), "guard collection");
        cudaCheck(cudaStreamSynchronize(stream), "guard collection retirement");
        for (std::size_t i = 0; i < values.size(); ++i)
            if ((i < rt::Geometry::guardWords || i >= rt::Geometry::guardWords + words_) && values[i] != guardValue)
                throw std::runtime_error("network/GPU touched outside the live extent");
    }
private:
    rt::Allocation kind_; std::size_t bytes_, words_; void* gpu_ = nullptr; void* host_ = nullptr; void* network_ = nullptr;
};
/** @brief CPU publication of transport completion, never a model-state decision. */
void publish(unsigned long long& field, unsigned long long epoch) {
    std::atomic_ref<unsigned long long>(field).store(epoch, std::memory_order_release);
}
/** @brief CPU acquisition of the GPU's immutable payload-readiness receipt. */
unsigned long long observe(unsigned long long& field) {
    return std::atomic_ref<unsigned long long>(field).load(std::memory_order_acquire);
}
/** @brief Run fixed-count nonblocking network operations without scheduling GPU work. */
void progress(Control& control, Buffer& tx, Buffer& rx, rt::Geometry geometry, int peer, unsigned iterations) {
    for (unsigned epoch = 1; epoch <= iterations; ++epoch) {
        MPI_Request send = MPI_REQUEST_NULL, receive = MPI_REQUEST_NULL;
        mpiCheck(MPI_Irecv(rx.network(), static_cast<int>(geometry.words), MPI_FLOAT, peer, 17,
                           MPI_COMM_WORLD, &receive), "post receive");
        bool posted = false, sent = false, received = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!sent || !received) {
            if (!posted && observe(control.txReady) >= epoch) {
                mpiCheck(MPI_Isend(tx.network(), static_cast<int>(geometry.words), MPI_FLOAT, peer, 17,
                                   MPI_COMM_WORLD, &send), "post send");
                posted = true;
            }
            int ready = 0;
            if (posted && !sent) {
                mpiCheck(MPI_Test(&send, &ready, MPI_STATUS_IGNORE), "send progress");
                if (ready) { publish(control.txRetired, epoch); sent = true; }
            }
            if (!received) {
                MPI_Status status;
                mpiCheck(MPI_Test(&receive, &ready, &status), "receive progress");
                if (ready) {
                    int count = 0; mpiCheck(MPI_Get_count(&status, MPI_FLOAT, &count), "receive count");
                    if (count != static_cast<int>(geometry.words)) throw std::runtime_error("truncated payload");
                    publish(control.rxReady, epoch); received = true;
                }
            }
            if (std::chrono::steady_clock::now() > deadline) {
                publish(control.abort, 1); throw std::runtime_error("30-second transport progress deadline");
            }
        }
    }
}
/** @return A deterministic nearest-rank percentile of sorted complete-round-trip samples. */
double quantile(const std::vector<unsigned long long>& sorted, double p) {
    const auto index = static_cast<std::size_t>(std::ceil(p * sorted.size())) - 1;
    return sorted.at(std::min(index, sorted.size() - 1)) / 1000.0;
}
/** @brief Reject an MPI image that cannot accept the explicitly requested CUDA pointers. */
bool cudaAwareMPI() {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}
} // namespace

/** @brief Enter canonical process ownership; external MPI retains its original rank tree. */
int main(int argc, char** argv) {
    int initialized = 0;
    try {
        if (const auto status = llaminar2::MPISessionOwner::enterProcess(argc, argv)) return *status;
        if (argc < 8) throw std::invalid_argument("usage: binary CANDIDATE BYTES WARMUP SAMPLES ROOT WORKERS OUTPUT_PREFIX [HCA:PORT:GID ...]");
        const auto policy = rt::parseRunPolicy(argv[1]);
        const auto kind = policy.allocation;
        if ((policy.plane == rt::DataPlane::MPI && argc != 8) || (policy.plane != rt::DataPlane::MPI && argc < 9))
            throw std::invalid_argument("explicit verbs candidates require rails; MPI candidates do not take verbs rails");
#if !defined(ROUNDTRIP_VERBS)
        if (policy.plane != rt::DataPlane::MPI) throw std::runtime_error("selected binary has no explicit verbs experiment");
#endif
        const rt::Geometry geometry(rt::positive(argv[2]));
        const auto warmup = rt::positive(argv[3]), measured = rt::positive(argv[4]);
        if (warmup > 1000000 || measured > 1000000 - warmup)
            throw std::invalid_argument("sample count exceeds bounded fixture storage");
        const unsigned total = static_cast<unsigned>(warmup + measured);
        const int root = std::string_view(argv[5]) == "0" ? 0 : std::string_view(argv[5]) == "1" ? 1 : -1;
        if (root < 0) throw std::invalid_argument("ROOT must be 0 or 1");
        const auto outputParent = std::filesystem::path(argv[7]).parent_path();
        if (!outputParent.empty()) std::filesystem::create_directories(outputParent);
        int provided = 0;
        mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "MPI init"); initialized = 1;
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI thread contract unavailable");
        mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN), "MPI errors");
        int rank = -1, size = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size != 2) throw std::runtime_error("this experiment requires exactly two original ranks");
        MPI_Comm local = MPI_COMM_NULL;
        mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local), "locality");
        int localSize = 0; MPI_Comm_size(local, &localSize); MPI_Comm_free(&local);
        if (localSize != 1) throw std::runtime_error("requires two physical hosts, not two aliases of one GPU");
        if (kind == rt::Allocation::Device && !cudaAwareMPI())
            throw std::runtime_error("UNQUALIFIED: selected MPI reports no CUDA-aware device-buffer support");
        cudaCheck(cudaSetDevice(0), "select original rank-local GPU");
        cudaDeviceProp properties{}; cudaCheck(cudaGetDeviceProperties(&properties, 0), "GPU properties");
        std::cerr << "GPU RDMA capability rank=" << rank << " supported=" << properties.gpuDirectRDMASupported
                  << " write_ordering=" << properties.gpuDirectRDMAWritesOrdering << '\n';
        if (kind == rt::Allocation::Device && !properties.gpuDirectRDMASupported)
            throw std::runtime_error("UNQUALIFIED: device reports no GPUDirect RDMA; staged CUDA-aware transport is not this candidate");
        int nativeAtomics = 0;
        cudaCheck(cudaDeviceGetAttribute(&nativeAtomics, cudaDevAttrHostNativeAtomicSupported, 0), "host atomics");
        if (!nativeAtomics || !properties.cooperativeLaunch || !properties.canMapHostMemory)
            throw std::runtime_error("UNQUALIFIED: required mapped system atomics/cooperative launch unavailable");
        int occupancy = 0;
        cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, roundTrip, threads, 0), "occupancy");
        if (occupancy < 1) throw std::runtime_error("no resident cooperative worker");
        const auto workers = std::string_view(argv[6]) == "all"
            ? static_cast<std::size_t>(properties.multiProcessorCount) : rt::positive(argv[6]);
        if (workers > static_cast<std::size_t>(properties.multiProcessorCount) * occupancy)
            throw std::invalid_argument("worker request exceeds cooperative residency");
        const auto blocks = geometry.blocks(threads, static_cast<unsigned>(workers));
        cudaStream_t stream = nullptr; cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "exact stream");
        {
            const auto nativePage = sysconf(_SC_PAGESIZE);
            if (nativePage <= 0) throw std::runtime_error("native page geometry unavailable");
            const std::size_t alignment = policy.plane == rt::DataPlane::MPI ? 1 : static_cast<std::size_t>(nativePage);
            Buffer tx(kind, geometry, alignment), rx(kind, geometry, alignment);
#if defined(ROUNDTRIP_VERBS)
            std::unique_ptr<rt::VerbsRoundTrip> verbs;
#endif
            // Diagnose before unwinding owners whose backing may still be used
            // by the GPU. Distributed retirement must precede their frees.
            try {
            tx.poison(stream); rx.poison(stream);
#if defined(ROUNDTRIP_VERBS)
            if (policy.plane != rt::DataPlane::MPI) {
                std::vector<std::string> rails(argv + 8, argv + argc);
                verbs = std::make_unique<rt::VerbsRoundTrip>(
                    policy.plane == rt::DataPlane::VerbsDmaBuf ? rt::VerbsRegistration::DmaBuf : rt::VerbsRegistration::Ordinary,
                    rails, tx.verbsView(), rx.verbsView(), geometry, 1 - rank);
                verbs->prepare(total);
            }
#endif
            Control* hostControl = nullptr; Control* gpuControl = nullptr;
            cudaCheck(cudaHostAlloc(&hostControl, sizeof(Control), cudaHostAllocMapped), "control allocation");
            new (hostControl) Control{};
            cudaCheck(cudaHostGetDevicePointer(&gpuControl, hostControl, 0), "control alias");
            DeviceResult* result = nullptr; float* consumed = nullptr; unsigned long long* durations = nullptr;
            cudaCheck(cudaMalloc(&result, sizeof(DeviceResult)), "result allocation");
            cudaCheck(cudaMalloc(&consumed, geometry.words * sizeof(float)), "consumer allocation");
            cudaCheck(cudaMalloc(&durations, total * sizeof(unsigned long long)), "duration allocation");
            cudaCheck(cudaMemsetAsync(result, 0, sizeof(DeviceResult), stream), "initialize oracle");
            cudaCheck(cudaStreamSynchronize(stream), "setup retirement");
            float* txPointer = tx.gpu(); float* rxPointer = rx.gpu(); auto words = geometry.words;
            unsigned iterations = total; bool initiator = rank == root;
            void* args[] = {&txPointer, &rxPointer, &consumed, &words, &gpuControl, &result, &durations, &iterations, &initiator};
            cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
            cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
            cudaCheck(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(roundTrip), dim3(blocks), dim3(threads), args, 0, stream), "capture complete roundtrip");
            cudaCheck(cudaStreamEndCapture(stream, &graph), "end capture");
            cudaCheck(cudaGraphInstantiate(&executable, graph, 0), "instantiate retained roundtrip");
            char hostname[256]{}; gethostname(hostname, sizeof(hostname) - 1);
            std::cerr << "endpoint rank=" << rank << " host=" << hostname << " gpu=" << properties.name
                      << " blocks=" << blocks << " threads=" << threads << " cuda_aware=" << cudaAwareMPI()
                      << " SMs=" << properties.multiProcessorCount << " resident_blocks_per_SM=" << occupancy << '\n';
            mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "cold startup join");
            const auto begin = std::chrono::steady_clock::now();
            cudaCheck(cudaGraphLaunch(executable, stream), "launch retained roundtrip");
#if defined(ROUNDTRIP_VERBS)
            if (verbs) verbs->progress(*hostControl, total);
            else
#endif
            progress(*hostControl, tx, rx, geometry, 1 - rank, total);
            cudaCheck(cudaStreamSynchronize(stream), "terminal diagnostic retirement");
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            DeviceResult observed{};
            std::vector<unsigned long long> samples(total);
            std::vector<float> terminal(geometry.words);
            cudaCheck(cudaMemcpyAsync(&observed, result, sizeof(observed), cudaMemcpyDeviceToHost, stream), "terminal oracle");
            cudaCheck(cudaMemcpyAsync(samples.data(), durations, samples.size() * sizeof(samples[0]), cudaMemcpyDeviceToHost, stream), "terminal timing");
            cudaCheck(cudaMemcpyAsync(terminal.data(), consumed, terminal.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "terminal consumer");
            cudaCheck(cudaStreamSynchronize(stream), "terminal observations");
            if (observed.errors || observed.stop) throw std::runtime_error("payload/epoch/reduction validation failed: " + std::to_string(observed.errors));
            for (std::size_t i = 0; i < terminal.size(); ++i)
                if (terminal[i] != rt::reducedValue(i, total) * 0.5f) throw std::runtime_error("terminal GPU consumer mismatch");
            tx.checkGuards(stream); rx.checkGuards(stream);
            mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "both endpoint validation");
            if (rank == root) {
                const std::string filename = std::string(argv[7]) + ".csv";
                std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(filename.c_str(), "w"), &std::fclose);
                if (!file) throw std::runtime_error("cannot create sample file");
                std::fprintf(file.get(), "sample,roundtrip_ns\n");
                std::vector<unsigned long long> sorted(samples.begin() + warmup, samples.end());
                for (std::size_t i = 0; i < sorted.size(); ++i) std::fprintf(file.get(), "%zu,%llu\n", i, sorted[i]);
                std::sort(sorted.begin(), sorted.end());
                double sum = 0; for (auto sample : sorted) sum += sample;
                const double meanUs = sum / sorted.size() / 1000.0;
                std::cout << "{\"candidate\":\"" << argv[1] << "\",\"bytes\":" << geometry.words * 4
                          << ",\"root\":" << root << ",\"samples\":" << sorted.size()
                          << ",\"blocks\":" << blocks << ",\"threads\":" << threads
                          << ",\"mean_us\":" << meanUs << ",\"p50_us\":" << quantile(sorted, .50)
                          << ",\"p95_us\":" << quantile(sorted, .95) << ",\"p99_us\":" << quantile(sorted, .99)
                          << ",\"payload_roundtrip_GBps\":" << (2.0 * geometry.words * 4 / (meanUs * 1000))
                          << ",\"wall_seconds_with_warmup\":" << wall << ",\"correct\":true}" << std::endl;
            }
            cudaCheck(cudaGraphExecDestroy(executable), "retire executable");
            cudaCheck(cudaGraphDestroy(graph), "retire graph");
            cudaCheck(cudaFree(durations), "retire samples"); cudaCheck(cudaFree(consumed), "retire consumer");
            cudaCheck(cudaFree(result), "retire oracle"); cudaCheck(cudaFreeHost(hostControl), "retire control");
            } catch (const std::exception& error) {
                std::cerr << "roundtrip fatal rank=" << rank << ": " << error.what() << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1); std::_Exit(1);
            }
        }
        cudaCheck(cudaStreamDestroy(stream), "retire stream");
        mpiCheck(MPI_Finalize(), "MPI retirement"); initialized = 0;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FATAL: " << error.what() << std::endl;
        if (initialized) MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
