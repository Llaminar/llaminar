/**
 * @file Probe__CUDADmaBuf.cpp
 * @brief Cold, allocation-specific CUDA DMA-BUF export and native NIC-import experiment.
 *
 * Capability queries are observations, not a shortcut around testing a different
 * allocation family. Every selected allocation/export/HCA combination produces a
 * receipt. A failed DMA-BUF import never substitutes ordinary ibv_reg_mr. This
 * isolated fixture owns small native allocations; it is not production PMA or
 * proof of GPU/NIC payload coherence. A successful import needs a later transfer
 * test. Canonical process ownership remains active before driver initialization.
 */
#include "utils/MPISessionOwner.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <infiniband/verbs.h>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
/** @brief Native rejection with its exact failing stage, not a guessed unsupported feature. */
struct Rejection : std::runtime_error {
    std::string stage; int code;
    /** @brief Retain the original native result and operation for the matrix receipt. */
    Rejection(std::string operation, int result, std::string explanation)
        : std::runtime_error(std::move(explanation)), stage(std::move(operation)), code(result) {}
};
/** @brief Scope-owned native retirement; release failure invalidates the whole probe. */
template<class F> class Retire {
    F fn_;
public:
    /** @brief Borrow only resources whose authority outlives this scope. */
    explicit Retire(F fn) : fn_(std::move(fn)) {}
    Retire(const Retire&) = delete;
    /** @brief Never continue a capability sweep after failed native retirement. */
    ~Retire() noexcept { try { fn_(); } catch (const std::exception& e) {
        std::cerr << "FATAL retirement: " << e.what() << '\n'; std::abort();
    } }
};
/** @brief Check a CUDA runtime call without substituting another allocation policy. */
void runtime(cudaError_t code, const char* operation) {
    if (code == cudaErrorMemoryAllocation) throw std::runtime_error("probe exhausted memory");
    if (code != cudaSuccess) throw Rejection(operation, code, cudaGetErrorName(code));
}
/** @brief Exact driver binding for a standalone diagnostic; never toolkit stub linkage. */
class Driver {
    void* library_;
    /** @brief Resolve a mandatory API from the actual host driver. */
    template<class T> T symbol(const char* name) {
        auto result = reinterpret_cast<T>(dlsym(library_, name));
        if (!result) throw std::runtime_error(std::string("missing driver API: ") + name);
        return result;
    }
public:
    decltype(&cuDeviceGetAttribute) attribute;
    decltype(&cuMemGetHandleForAddressRange) exportRange;
    decltype(&cuMemGetAllocationGranularity) granularity;
    decltype(&cuMemCreate) create;
    decltype(&cuMemRelease) release;
    decltype(&cuMemAddressReserve) reserve;
    decltype(&cuMemAddressFree) freeAddress;
    decltype(&cuMemMap) map;
    decltype(&cuMemUnmap) unmap;
    decltype(&cuMemSetAccess) access;
    decltype(&cuGetErrorName) errorName;
    /** @brief Bind after canonical process entry; reject incomplete driver interfaces. */
    Driver() : library_(dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL)) {
        if (!library_) throw std::runtime_error(dlerror());
        try {
            attribute = symbol<decltype(attribute)>("cuDeviceGetAttribute");
            exportRange = symbol<decltype(exportRange)>("cuMemGetHandleForAddressRange");
            granularity = symbol<decltype(granularity)>("cuMemGetAllocationGranularity");
            create = symbol<decltype(create)>("cuMemCreate");
            release = symbol<decltype(release)>("cuMemRelease");
            reserve = symbol<decltype(reserve)>("cuMemAddressReserve");
            freeAddress = symbol<decltype(freeAddress)>("cuMemAddressFree");
            map = symbol<decltype(map)>("cuMemMap");
            unmap = symbol<decltype(unmap)>("cuMemUnmap");
            access = symbol<decltype(access)>("cuMemSetAccess");
            errorName = symbol<decltype(errorName)>("cuGetErrorName");
        } catch (...) { dlclose(library_); throw; }
    }
    Driver(const Driver&) = delete;
    /** @brief Release the exact binding only after every driver-owned object is retired. */
    ~Driver() { dlclose(library_); }
    /** @brief Preserve a CUDA driver failure, distinguishing exhausted resources. */
    void check(CUresult code, const char* operation) const {
        if (code == CUDA_ERROR_OUT_OF_MEMORY) throw std::runtime_error("probe exhausted driver memory");
        if (code != CUDA_SUCCESS) {
            const char* name = "unknown CUDA error"; errorName(code, &name);
            throw Rejection(operation, code, name ? name : "unknown CUDA error");
        }
    }
};
/** @brief One observed HCA context and protection domain, not an invented GPU/rank alias. */
class Nic {
    ibv_context* context_ = nullptr;
public:
    std::string name;
    ibv_pd* pd = nullptr;
    /** @brief Open the exact named HCA and retain its own protection domain. */
    Nic(ibv_device* device, std::string selected) : name(std::move(selected)) {
        context_ = ibv_open_device(device);
        if (!context_) throw std::runtime_error("ibv_open_device: " + std::string(std::strerror(errno)));
        pd = ibv_alloc_pd(context_);
        if (!pd) { ibv_close_device(context_); throw std::runtime_error("ibv_alloc_pd failed"); }
    }
    Nic(const Nic&) = delete;
    /** @brief No live MR may remain when the PD and original HCA are retired. */
    ~Nic() {
        if (ibv_dealloc_pd(pd) || ibv_close_device(context_)) std::abort();
    }
};
constexpr std::array names = {"host-mapped", "host-write-combined", "host-registered", "device",
    "managed-cpu-touched", "managed-gpu-touched", "device-async", "vmm-device", "vmm-device-gdr",
    "vmm-host", "vmm-host-numa"};
constexpr std::array<unsigned long long, 2> exportFlags = {0, CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE};
/** @brief Derive the local NUMA ID from the actual executing CPU, never a fixed socket number. */
int localNuma() {
    const int cpu = sched_getcpu();
    if (cpu < 0) throw std::runtime_error("cannot observe current CPU");
    const auto directory = std::filesystem::path("/sys/devices/system/cpu") / ("cpu" + std::to_string(cpu));
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const auto name = entry.path().filename().string();
        if (name.starts_with("node") && name.size() > 4) return std::stoi(name.substr(4));
    }
    throw Rejection("numa-observation", -1, "no observed CPU NUMA membership");
}
/** @brief Produce one allocation and invoke the export test while its backing remains owned. */
template<class F>
void withAllocation(const std::string& kind, Driver& d, cudaStream_t stream, std::size_t bytes, F use) {
    if (kind.starts_with("vmm-")) {
        CUmemAllocationProp prop{}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = kind == "vmm-host" ? CU_MEM_LOCATION_TYPE_HOST :
            kind == "vmm-host-numa" ? CU_MEM_LOCATION_TYPE_HOST_NUMA : CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = kind == "vmm-host-numa" ? localNuma() : 0;
        prop.allocFlags.gpuDirectRDMACapable = kind == "vmm-device-gdr";
        std::size_t granularity = 0;
        d.check(d.granularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "cuMemGetAllocationGranularity");
        if (!granularity || granularity > 64 * 1024 * 1024) throw Rejection("geometry", -1, "granularity exceeds bounded probe");
        bytes = (bytes + granularity - 1) / granularity * granularity;
        CUmemGenericAllocationHandle handle{};
        d.check(d.create(&handle, bytes, &prop, 0), "cuMemCreate");
        Retire allocation([&] { d.check(d.release(handle), "cuMemRelease"); });
        CUdeviceptr address{};
        d.check(d.reserve(&address, bytes, 0, 0, 0), "cuMemAddressReserve");
        Retire reservation([&] { d.check(d.freeAddress(address, bytes), "cuMemAddressFree"); });
        d.check(d.map(address, bytes, 0, handle, 0), "cuMemMap");
        Retire mapping([&] { d.check(d.unmap(address, bytes), "cuMemUnmap"); });
        CUmemAccessDesc access{}; access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = 0; access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        d.check(d.access(address, bytes, &access, 1), "cuMemSetAccess-device");
        runtime(cudaMemsetAsync(reinterpret_cast<void*>(address), 0x5a, bytes, stream), "initialize-vmm");
        runtime(cudaStreamSynchronize(stream), "retire-vmm-initialization");
        use(address, bytes);
        return;
    }
    void* pointer = nullptr;
    if (kind == "host-registered") {
        if (posix_memalign(&pointer, static_cast<std::size_t>(sysconf(_SC_PAGESIZE)), bytes))
            throw std::runtime_error("aligned host allocation failed");
        Retire backing([&] { std::free(pointer); });
        std::memset(pointer, 0x5a, bytes);
        runtime(cudaHostRegister(pointer, bytes, cudaHostRegisterMapped), "cudaHostRegister");
        Retire registration([&] { runtime(cudaHostUnregister(pointer), "cudaHostUnregister"); });
        void* alias = nullptr;
        runtime(cudaHostGetDevicePointer(&alias, pointer, 0), "cudaHostGetDevicePointer");
        use(reinterpret_cast<CUdeviceptr>(alias), bytes);
    } else if (kind.starts_with("host-")) {
        const unsigned flags = cudaHostAllocMapped | (kind == "host-write-combined" ? cudaHostAllocWriteCombined : 0);
        runtime(cudaHostAlloc(&pointer, bytes, flags), "cudaHostAlloc");
        Retire allocation([&] { runtime(cudaFreeHost(pointer), "cudaFreeHost"); });
        std::memset(pointer, 0x5a, bytes);
        void* alias = nullptr;
        runtime(cudaHostGetDevicePointer(&alias, pointer, 0), "cudaHostGetDevicePointer");
        use(reinterpret_cast<CUdeviceptr>(alias), bytes);
    } else {
        if (kind == "device-async") runtime(cudaMallocAsync(&pointer, bytes, stream), "cudaMallocAsync");
        else if (kind.starts_with("managed-")) runtime(cudaMallocManaged(&pointer, bytes), "cudaMallocManaged");
        else runtime(cudaMalloc(&pointer, bytes), "cudaMalloc");
        Retire allocation([&] {
            if (kind == "device-async") {
                runtime(cudaFreeAsync(pointer, stream), "cudaFreeAsync");
                runtime(cudaStreamSynchronize(stream), "retire-async-free");
            } else runtime(cudaFree(pointer), "cudaFree");
        });
        if (kind == "managed-cpu-touched") std::memset(pointer, 0x5a, bytes);
        else {
            runtime(cudaMemsetAsync(pointer, 0x5a, bytes, stream), "initialize-device");
            runtime(cudaStreamSynchronize(stream), "retire-device-initialization");
        }
        use(reinterpret_cast<CUdeviceptr>(pointer), bytes);
    }
}
/** @brief Emit exact native-stage evidence; success means registration, not transfer qualification. */
void receipt(const std::string& allocation, unsigned long long flags, const Nic& nic,
             std::size_t bytes, const char* stage, int code, const std::string& detail) {
    std::cout << "{\"event\":\"case\",\"allocation\":" << std::quoted(allocation)
              << ",\"flags\":" << flags << ",\"nic\":" << std::quoted(nic.name)
              << ",\"bytes\":" << bytes << ",\"stage\":" << std::quoted(stage)
              << ",\"code\":" << code << ",\"detail\":" << std::quoted(detail) << "}\n";
}
/** @brief Test exports and actual DMA-BUF imports, retiring each MR before its FD and backing. */
unsigned probe(Driver& driver, const std::vector<std::unique_ptr<Nic>>& nics, cudaStream_t stream, std::size_t bytes) {
    unsigned successes = 0;
    for (const char* name : names) {
        try {
            withAllocation(name, driver, stream, bytes, [&](CUdeviceptr address, std::size_t extent) {
                for (const auto flags : exportFlags) {
                    int fd = -1;
                    const CUresult status = driver.exportRange(&fd, address, extent, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags);
                    if (status != CUDA_SUCCESS) {
                        const char* error = "unknown"; driver.errorName(status, &error);
                        for (const auto& nic : nics) receipt(name, flags, *nic, extent, "export", status, error ? error : "unknown");
                        continue;
                    }
                    if (fd < 0) throw std::runtime_error("successful export returned no owned FD");
                    Retire descriptor([&] { if (close(fd)) throw std::runtime_error("DMA-BUF FD retirement failed"); });
                    struct stat identity{};
                    if (fstat(fd, &identity)) throw std::runtime_error("exported FD is invalid");
                    std::cout << "{\"event\":\"export\",\"allocation\":" << std::quoted(std::string(name))
                              << ",\"flags\":" << flags << ",\"bytes\":" << extent
                              << ",\"fd_inode\":" << identity.st_ino << "}\n";
                    for (const auto& nic : nics) {
                        errno = 0;
                        ibv_mr* mr = ibv_reg_dmabuf_mr(nic->pd, 0, extent, address, fd,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
                        const int error = errno;
                        if (!mr) { receipt(name, flags, *nic, extent, "registration", error, std::strerror(error)); continue; }
                        Retire region([&] { if (ibv_dereg_mr(mr)) throw std::runtime_error("MR retirement failed"); });
                        if (mr->length != extent) throw std::runtime_error("NIC MR has wrong extent");
                        receipt(name, flags, *nic, extent, "registered", 0, "payload transfer not yet qualified");
                        ++successes;
                    }
                }
            });
        } catch (const Rejection& rejection) {
            for (const auto flags : exportFlags) for (const auto& nic : nics)
                receipt(name, flags, *nic, bytes, rejection.stage.c_str(), rejection.code, rejection.what());
        }
    }
    return successes;
}
} // namespace

/** @brief Enter native ownership, observe every capability independently and run the requested NIC matrix. */
int main(int argc, char** argv) {
    try {
        if (const auto status = llaminar2::MPISessionOwner::enterProcess(argc, argv)) return *status;
        if (argc < 2) throw std::invalid_argument("usage: cuda_dmabuf_probe NIC [NIC ...]");
        unsigned successes = 0;
        {
            Driver driver;
            runtime(cudaSetDevice(0), "select-rank-local-GPU");
            int driverVersion = 0; runtime(cudaDriverGetVersion(&driverVersion), "driver-version");
            char host[256]{}; if (gethostname(host, sizeof(host))) throw std::runtime_error("hostname unavailable");
            const auto page = sysconf(_SC_PAGESIZE);
            if (page <= 0) throw std::runtime_error("host page size unavailable");
            std::cout << "{\"event\":\"identity\",\"host\":" << std::quoted(std::string(host))
                      << ",\"page_bytes\":" << page << ",\"driver_version\":" << driverVersion << ",\"runtime_version\":" << CUDART_VERSION << "}\n";
            for (auto [name, attribute] : std::array{
                std::pair{"device_gdr", CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED},
                std::pair{"device_dmabuf", CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED},
                std::pair{"host_dmabuf", CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED},
                std::pair{"vmm_gdr", CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED},
                std::pair{"vmm", CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED}}) {
                int value = -1; const auto status = driver.attribute(&value, attribute, 0);
                std::cout << "{\"event\":\"capability\",\"name\":" << std::quoted(std::string(name))
                          << ",\"code\":" << status << ",\"value\":" << value << "}\n";
            }
            int count = 0; ibv_device** devices = ibv_get_device_list(&count);
            if (!devices) throw std::runtime_error("cannot enumerate native RDMA devices");
            Retire inventory([&] { ibv_free_device_list(devices); });
            std::vector<std::unique_ptr<Nic>> nics;
            for (int argument = 1; argument < argc; ++argument) {
                ibv_device* selected = nullptr;
                for (int i = 0; i < count; ++i) if (std::string(argv[argument]) == ibv_get_device_name(devices[i])) selected = devices[i];
                if (!selected) throw std::invalid_argument("requested HCA absent: " + std::string(argv[argument]));
                for (const auto& nic : nics) if (nic->name == argv[argument]) throw std::invalid_argument("duplicate HCA");
                nics.push_back(std::make_unique<Nic>(selected, argv[argument]));
            }
            const std::size_t bytes = (1024 * 1024 + page - 1) / page * page;
            for (const auto& nic : nics) {
                errno = 0;
                ibv_mr* unexpected = ibv_reg_dmabuf_mr(nic->pd, 0, bytes, bytes, -1, IBV_ACCESS_LOCAL_WRITE);
                const int error = errno;
                if (unexpected) { ibv_dereg_mr(unexpected); throw std::runtime_error("invalid DMA-BUF FD accepted"); }
                std::cout << "{\"event\":\"negative-control\",\"nic\":" << std::quoted(nic->name)
                          << ",\"code\":" << error << "}\n";
            }
            cudaStream_t stream = nullptr;
            runtime(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create-explicit-stream");
            Retire streamOwner([&] { runtime(cudaStreamDestroy(stream), "retire-stream"); });
            successes = probe(driver, nics, stream, bytes);
        }
        std::cout << "{\"event\":\"complete\",\"registration_successes\":" << successes
                  << ",\"transfer_qualified\":false}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FATAL: " << error.what() << '\n'; return 1; }
}
