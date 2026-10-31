/**
 * @file Test__CUDAHardwareMetadataNative.cu
 * @brief Native identity and captured replay for every admitted visible CUDA device.
 *
 * Independent runtime observations authenticate the immutable inventory. Test-only
 * ELF interposition counts actual discovery calls across shared and static core
 * linkage; repeated physical reads must neither rediscover nor change contexts.
 * A tiny captured kernel reports its native warp width and compilation ISA while
 * retaining an explicit per-device binding token. Runtime reset remains owned by
 * the existing exclusive-retirement gates, not this physical-facts fixture.
 */
#include "backends/cuda/CUDAHardwareMetadata.h"
#include "backends/BackendManager.h"
#include "CUDAHardwareQueryObserver.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
    using llaminar2::test::cuda_metadata_native::queryCounts;

    /** @brief Fail the native proof on a query, launch, ordering or cleanup error. */
    void checked(cudaError_t status)
    {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("CUDA metadata native replay: ") + cudaGetErrorString(status));
    }

    /** @brief Device-produced facts and caller binding identity, with no inference arithmetic. */
    struct KernelObservation { int warp_width, compiled_arch, binding; };

    /** @brief Capture a native device fact and the exact admitted ordinal's binding token. */
    __global__ void observeHardware(KernelObservation *output, int binding)
    {
        output->warp_width = warpSize;
#ifdef __CUDA_ARCH__
        output->compiled_arch = __CUDA_ARCH__;
#endif
        output->binding = binding;
    }

    /** @brief Restore only the fixture's selected context; never retire a runtime generation. */
    class CurrentDeviceScope final
    {
    public:
        /** @brief Retain the actual current device before alternating native selections. */
        CurrentDeviceScope() { checked(cudaGetDevice(&previous_)); }
        CurrentDeviceScope(const CurrentDeviceScope &) = delete;
        CurrentDeviceScope &operator=(const CurrentDeviceScope &) = delete;
        /** @brief Restore selection on success or test failure. */
        ~CurrentDeviceScope() { if (cudaSetDevice(previous_) != cudaSuccess) std::terminate(); }
    private:
        int previous_ = -1;
    };

    /** @brief Per-device retained graph/storage with exact stream and ordered terminal reads. */
    class CapturedObservation final
    {
    public:
        /** @brief Allocate once and capture one lightweight device observation. @param ordinal Exact selected ordinal. */
        explicit CapturedObservation(int ordinal) : ordinal_(ordinal)
        {
            try
            {
                checked(cudaSetDevice(ordinal_));
                checked(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
                checked(cudaMalloc(reinterpret_cast<void **>(&output_), sizeof(KernelObservation)));
                checked(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
                capturing_ = true;
                observeHardware<<<1, 1, 0, stream_>>>(output_, ordinal);
                checked(cudaGetLastError());
                const auto ended = cudaStreamEndCapture(stream_, &graph_);
                capturing_ = false;
                checked(ended);
                checked(cudaGraphInstantiate(&execution_, graph_, nullptr, nullptr, 0));
            }
            catch (...) { release(); throw; }
        }
        CapturedObservation(const CapturedObservation &) = delete;
        CapturedObservation &operator=(const CapturedObservation &) = delete;
        /** @brief Release graph and storage after the replay's explicit terminal boundary. */
        ~CapturedObservation()
        {
            release();
        }
        /** @return Device-produced observation after poisoning and replaying the same graph. */
        KernelObservation replay()
        {
            checked(cudaSetDevice(ordinal_));
            checked(cudaMemsetAsync(output_, 0xa5, sizeof(KernelObservation), stream_));
            checked(cudaGraphLaunch(execution_, stream_));
            KernelObservation result{};
            checked(cudaMemcpyAsync(&result, output_, sizeof(result), cudaMemcpyDeviceToHost, stream_));
            checked(cudaStreamSynchronize(stream_)); // Explicit native proof's terminal read, outside capture.
            return result;
        }
    private:
        /** @brief Retire every successfully constructed owner, including failed capture setup. */
        void release() noexcept
        {
            if (!stream_ && !output_ && !graph_ && !execution_ && !capturing_) return;
            if (cudaSetDevice(ordinal_) != cudaSuccess) std::terminate();
            if (capturing_)
            {
                // An invalidated capture can report an error while still
                // ending ownership; retain any graph returned for destruction.
                (void)cudaStreamEndCapture(stream_, &graph_);
                capturing_ = false;
            }
            if (execution_ && cudaGraphExecDestroy(execution_) != cudaSuccess) std::terminate();
            execution_ = nullptr;
            if (graph_ && cudaGraphDestroy(graph_) != cudaSuccess) std::terminate();
            graph_ = nullptr;
            if (output_ && cudaFree(output_) != cudaSuccess) std::terminate();
            output_ = nullptr;
            if (stream_ && cudaStreamDestroy(stream_) != cudaSuccess) std::terminate();
            stream_ = nullptr;
        }
        const int ordinal_; ///< Exact owner even when another fixture changes ambient selection.
        bool capturing_ = false; ///< Constructor-unwind ownership before a graph exists.
        cudaStream_t stream_ = nullptr;
        KernelObservation *output_ = nullptr;
        cudaGraph_t graph_ = nullptr;
        cudaGraphExec_t execution_ = nullptr;
    };
}

namespace llaminar2::test
{
    /** @test Every visible physical record matches native queries and survives reordered captured reads. */
    TEST(CUDAHardwareMetadataNative, AllVisibleDevicesRetainPhysicalIdentityAcrossCapturedReplay)
    {
        initializeCUDAHardwareMetadata();
        const auto &inventory = cudaHardwareMetadata();
        if (inventory.size() == 0)
            GTEST_SKIP() << "UNQUALIFIED: CUDA native metadata replay has no visible hardware";
        auto *backend = getCUDABackend();
        ASSERT_NE(backend, nullptr);
        CurrentDeviceScope current_device;
        int native_count = -1;
        checked(cudaGetDeviceCount(&native_count));
        ASSERT_EQ(inventory.size(), static_cast<std::size_t>(native_count));
        for (int ordinal = 0; ordinal < native_count; ++ordinal)
        {
            SCOPED_TRACE(ordinal);
            const auto &hardware = inventory.at(ordinal);
            cudaDeviceProp properties{};
            checked(cudaGetDeviceProperties(&properties, ordinal));
            char pci[32]{};
            checked(cudaDeviceGetPCIBusId(pci, sizeof(pci), ordinal));
            EXPECT_EQ(hardware.name, properties.name);
            EXPECT_EQ(hardware.pci_address, DevicePCIAddress::parse(pci).toString());
            EXPECT_EQ(hardware.pci_domain, properties.pciDomainID);
            EXPECT_EQ(hardware.pci_bus, properties.pciBusID);
            EXPECT_EQ(hardware.pci_device, properties.pciDeviceID);
            EXPECT_EQ(std::memcmp(hardware.uuid.data(), properties.uuid.bytes, hardware.uuid.size()), 0);
            EXPECT_EQ(hardware.sm_major, properties.major);
            EXPECT_EQ(hardware.sm_minor, properties.minor);
            EXPECT_EQ(hardware.sm_count, properties.multiProcessorCount);
            EXPECT_EQ(hardware.native_warp_width, properties.warpSize);
            EXPECT_EQ(hardware.maximum_threads_per_block, properties.maxThreadsPerBlock);
            EXPECT_EQ(hardware.maximum_threads_per_sm, properties.maxThreadsPerMultiProcessor);
            EXPECT_EQ(hardware.maximum_blocks_per_sm, properties.maxBlocksPerMultiProcessor);
            EXPECT_EQ(hardware.registers_per_block, properties.regsPerBlock);
            EXPECT_EQ(hardware.registers_per_sm, properties.regsPerMultiprocessor);
            EXPECT_EQ(hardware.shared_memory_per_block, properties.sharedMemPerBlock);
            EXPECT_EQ(hardware.shared_memory_optin_per_block, properties.sharedMemPerBlockOptin);
            EXPECT_EQ(hardware.shared_memory_per_sm, properties.sharedMemPerMultiprocessor);
            EXPECT_EQ(hardware.total_memory, properties.totalGlobalMem);
            EXPECT_EQ(hardware.last_level_cache, static_cast<std::size_t>(std::max(properties.l2CacheSize, 0)));
            std::array<char, 33> uuid{};
            for (std::size_t byte = 0; byte < hardware.uuid.size(); ++byte)
                std::snprintf(uuid.data() + byte * 2, 3, "%02x", static_cast<unsigned>(hardware.uuid[byte]));
            std::cout << "[CUDAHardwareMetadataNative] ordinal=" << ordinal << " pci=" << hardware.pci_address
                      << " uuid=" << uuid.data()
                      << " name=" << hardware.name << " sm=" << hardware.sm_major << '.' << hardware.sm_minor
                      << " sm_count=" << hardware.sm_count << '\n';
        }
        const auto admitted_queries = queryCounts();
        EXPECT_GE(admitted_queries[0], 1u);
        EXPECT_GE(admitted_queries[1], static_cast<unsigned>(native_count));
        EXPECT_GE(admitted_queries[2], static_cast<unsigned>(native_count));
        // Every native selected device is visited in both orders; no fixed two-card assumption.
        for (int pass = 0; pass < 4; ++pass)
            for (int index = 0; index < native_count; ++index)
            {
                const int ordinal = pass % 2 == 0 ? index : native_count - 1 - index;
                SCOPED_TRACE(ordinal);
                checked(cudaSetDevice(ordinal));
                const auto &hardware = inventory.at(ordinal);
                const auto *identity = &hardware;
                for (int read = 0; read < 128; ++read)
                {
                    initializeCUDAHardwareMetadata();
                    EXPECT_EQ(&cudaHardwareMetadata().at(ordinal), identity);
                    EXPECT_EQ(backend->deviceCount(), native_count);
                    EXPECT_EQ(backend->deviceName(ordinal), hardware.name);
                    EXPECT_EQ(backend->deviceMemoryTotal(ordinal), hardware.total_memory);
                    const int capability = hardware.sm_major * 10 + hardware.sm_minor;
                    EXPECT_EQ(backend->supportsFP16(ordinal), capability >= 53);
                    EXPECT_EQ(backend->supportsBF16(ordinal), capability >= 80);
                    EXPECT_EQ(backend->supportsINT8(ordinal), capability >= 61);
                }
                int selected = -1;
                checked(cudaGetDevice(&selected));
                EXPECT_EQ(selected, ordinal);
                EXPECT_EQ(queryCounts(), admitted_queries);
                CapturedObservation capture(ordinal);
                for (int replay = 0; replay < 3; ++replay)
                {
                    const auto actual = capture.replay();
                    EXPECT_EQ(actual.warp_width, hardware.native_warp_width);
                    // Compiled ISA may be older PTX JIT-compatible code, never a claimed native family.
                    EXPECT_GT(actual.compiled_arch, 0);
                    EXPECT_LE(actual.compiled_arch, hardware.sm_major * 100 + hardware.sm_minor * 10);
                    EXPECT_EQ(actual.binding, ordinal);
                    EXPECT_EQ(&cudaHardwareMetadata().at(ordinal), identity);
                }
                checked(cudaGetDevice(&selected));
                EXPECT_EQ(selected, ordinal);
                EXPECT_EQ(queryCounts(), admitted_queries);
            }
        std::cout << "[CUDAHardwareMetadataNative] exercised_visible_devices=" << native_count
                  << " count_queries=" << admitted_queries[0]
                  << " property_queries=" << admitted_queries[1]
                  << " pci_queries=" << admitted_queries[2]
                  << " repeated_reads_and_public_getters_added_queries="
                  << (queryCounts()[0] - admitted_queries[0]) +
                     (queryCounts()[1] - admitted_queries[1]) +
                     (queryCounts()[2] - admitted_queries[2]) << '\n';
    }
}
