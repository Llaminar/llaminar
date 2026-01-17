/**
 * @file Test__DirectP2P.cpp
 * @brief Test direct cross-vendor P2P via PCIe BAR mapping
 *
 * Tests whether we can share memory directly between CUDA and ROCm
 * using PCIe BAR mapping. This is true peer-to-peer with no host memory bounce.
 */

#include <gtest/gtest.h>

#include "backends/p2p/DirectP2P.h"
#include "utils/Logger.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

namespace llaminar2
{
    namespace test
    {

        class Test__DirectP2P : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                // Check capabilities
                caps_ = DirectP2PEngine::probeCapabilities();
            }

            DirectP2PCapability caps_;
        };

        //----------------------------------------------------------------------
        // Capability Probe Tests
        //----------------------------------------------------------------------

        TEST_F(Test__DirectP2P, ProbeCapabilities)
        {
            LOG_INFO("\n"
                     << caps_.describe());

            // At minimum, we should get kernel version
            EXPECT_FALSE(caps_.kernel_version.empty());

            // Log what we found
            LOG_INFO("PCIe BAR accessible: " << (caps_.pcie_bar_accessible ? "YES" : "NO"));
            LOG_INFO("CUDA IOMEMORY support: " << (caps_.pcie_bar_iomemory_supported ? "YES" : "NO"));
            LOG_INFO("AMD BARs discovered: " << caps_.discovered_bars.size());
            LOG_INFO("Direct P2P possible: " << (caps_.canDoDirectP2P() ? "YES" : "NO"));
        }

        TEST_F(Test__DirectP2P, PCIeBar_ProbeCapabilities)
        {
            LOG_INFO("\n=== PCIe BAR Capability Probe ===");
            LOG_INFO("AMD BARs found: " << caps_.discovered_bars.size());
            for (const auto &bar : caps_.discovered_bars)
            {
                LOG_INFO("  - " << bar.pci_address << ": "
                                << (bar.bar_size / (1024 * 1024 * 1024)) << " GB");
            }
            LOG_INFO("BAR access: " << (caps_.pcie_bar_accessible ? "YES" : "NO (need root?)"));
            LOG_INFO("CUDA IOMEMORY: " << (caps_.pcie_bar_iomemory_supported ? "YES" : "NO"));
            LOG_INFO("PCIe BAR P2P possible: " << (caps_.canDoDirectP2P() ? "YES" : "NO"));

            // We should at least detect AMD GPUs
            if (caps_.discovered_bars.empty())
            {
                LOG_WARN("No AMD GPU BARs found - test environment may not have AMD GPU");
            }
        }

        //----------------------------------------------------------------------
        // PCIe BAR Single-GPU Tests
        //----------------------------------------------------------------------

        TEST_F(Test__DirectP2P, PCIeBar_Initialize)
        {
            if (!caps_.canDoDirectP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available (need root + AMD GPU)";
            }

            DirectP2PEngine engine;
            DeviceId cuda_dev = DeviceId::cuda(0);
            DeviceId rocm_dev = DeviceId::rocm(0);

            LOG_INFO("\n=== PCIe BAR Initialization ===");
            bool init = engine.initializePCIeBar(cuda_dev, rocm_dev);

            EXPECT_TRUE(init) << "PCIe BAR initialization should succeed with root access";
            EXPECT_TRUE(engine.isPCIeBarActive());

            if (init)
            {
                LOG_INFO("CUDA BAR pointer: " << engine.getCudaBarPointer());
                LOG_INFO("Mapped size: " << (engine.getBarMappedSize() / (1024 * 1024)) << " MB");
            }
        }

        TEST_F(Test__DirectP2P, PCIeBar_TransferToNVIDIA)
        {
            if (!caps_.canDoDirectP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available";
            }

            DirectP2PEngine engine;
            DeviceId cuda_dev = DeviceId::cuda(0);
            DeviceId rocm_dev = DeviceId::rocm(0);

            ASSERT_TRUE(engine.initializePCIeBar(cuda_dev, rocm_dev));

            size_t size = 16 * 1024 * 1024; // 16 MB

            // Allocate CUDA buffer
            void *cuda_buf = nullptr;
            cudaMalloc(&cuda_buf, size);
            ASSERT_NE(cuda_buf, nullptr);

            LOG_INFO("\n=== PCIe BAR Transfer: AMD → NVIDIA ===");
            auto result = engine.transferViaPCIeBar(cuda_buf, 0, size,
                                                    DirectP2PEngine::Direction::ToNVIDIA);

            LOG_INFO("Result:");
            LOG_INFO("  Success: " << result.success);
            LOG_INFO("  Throughput: " << result.throughput_gbps << " GB/s");
            LOG_INFO("  Path: " << result.transfer_path);

            EXPECT_TRUE(result.success);
            EXPECT_GT(result.throughput_gbps, 0.5); // At least 0.5 GB/s

            cudaFree(cuda_buf);
        }

        TEST_F(Test__DirectP2P, PCIeBar_TransferToAMD)
        {
            if (!caps_.canDoDirectP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available";
            }

            DirectP2PEngine engine;
            DeviceId cuda_dev = DeviceId::cuda(0);
            DeviceId rocm_dev = DeviceId::rocm(0);

            ASSERT_TRUE(engine.initializePCIeBar(cuda_dev, rocm_dev));

            size_t size = 16 * 1024 * 1024; // 16 MB

            // Allocate CUDA buffer
            void *cuda_buf = nullptr;
            cudaMalloc(&cuda_buf, size);
            ASSERT_NE(cuda_buf, nullptr);
            cudaMemset(cuda_buf, 0xAB, size); // Initialize

            LOG_INFO("\n=== PCIe BAR Transfer: NVIDIA → AMD ===");
            auto result = engine.transferViaPCIeBar(cuda_buf, 0, size,
                                                    DirectP2PEngine::Direction::ToAMD);

            LOG_INFO("Result:");
            LOG_INFO("  Success: " << result.success);
            LOG_INFO("  Throughput: " << result.throughput_gbps << " GB/s");
            LOG_INFO("  Path: " << result.transfer_path);

            EXPECT_TRUE(result.success);
            EXPECT_GT(result.throughput_gbps, 0.5); // At least 0.5 GB/s

            cudaFree(cuda_buf);
        }

        TEST_F(Test__DirectP2P, PCIeBar_Benchmark)
        {
            if (!caps_.canDoDirectP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available";
            }

            DirectP2PEngine engine;
            DeviceId cuda_dev = DeviceId::cuda(0);
            DeviceId rocm_dev = DeviceId::rocm(0);

            ASSERT_TRUE(engine.initializePCIeBar(cuda_dev, rocm_dev));

            LOG_INFO("\n=== PCIe BAR Direct P2P Benchmark ===");
            size_t size = 64 * 1024 * 1024; // 64 MB
            auto result = engine.benchmarkPCIeBar(size, 5);

            LOG_INFO("\nFinal Results:");
            LOG_INFO("  AMD → NVIDIA (read): " << result.read_gbps << " GB/s");
            LOG_INFO("  NVIDIA → AMD (write): " << result.write_gbps << " GB/s");
            LOG_INFO("  Average: " << result.throughput_gbps << " GB/s");

            EXPECT_TRUE(result.success);
            EXPECT_GT(result.read_gbps, 0.5);  // Should get at least 0.5 GB/s read
            EXPECT_GT(result.write_gbps, 0.5); // Should get at least 0.5 GB/s write

            // Write is typically faster than read for PCIe BAR (without rBAR)
            LOG_INFO("  Write/Read ratio: " << (result.write_gbps / result.read_gbps));
        }

        TEST_F(Test__DirectP2P, PCIeBar_Benchmark_60s)
        {
            if (!caps_.canDoDirectP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available";
            }

            DirectP2PEngine engine;
            DeviceId cuda_dev = DeviceId::cuda(0);
            DeviceId rocm_dev = DeviceId::rocm(0);

            ASSERT_TRUE(engine.initializePCIeBar(cuda_dev, rocm_dev));

            size_t size = 64 * 1024 * 1024; // 64 MB

            LOG_INFO("\n");
            LOG_INFO("╔══════════════════════════════════════════════════════════════╗");
            LOG_INFO("║     60-SECOND P2P STRESS TEST - MONITOR WITH pcm-memory      ║");
            LOG_INFO("╠══════════════════════════════════════════════════════════════╣");
            LOG_INFO("║ If this were using host memory bounce:                       ║");
            LOG_INFO("║   - pcm-memory would show ~5+ GB/s DRAM bandwidth            ║");
            LOG_INFO("║   - CPU utilization would spike                              ║");
            LOG_INFO("║ With true P2P:                                               ║");
            LOG_INFO("║   - pcm-memory shows near-zero DRAM traffic                  ║");
            LOG_INFO("║   - nvidia-smi shows ~2.6 GB/s PCIe TX/RX                    ║");
            LOG_INFO("╚══════════════════════════════════════════════════════════════╝");
            LOG_INFO("");

            auto start_time = std::chrono::steady_clock::now();
            auto end_time = start_time + std::chrono::seconds(60);

            size_t total_bytes = 0;
            int iterations = 0;

            while (std::chrono::steady_clock::now() < end_time)
            {
                auto result = engine.benchmarkPCIeBar(size, 3);
                total_bytes += size * 2 * 3; // read + write, 3 iterations each
                iterations++;

                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                   std::chrono::steady_clock::now() - start_time)
                                   .count();

                LOG_INFO("[" << elapsed << "s] Iteration " << iterations
                             << ": R=" << result.read_gbps << " GB/s, W=" << result.write_gbps << " GB/s");
            }

            double total_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
            LOG_INFO("\n=== 60-Second Stress Test Complete ===");
            LOG_INFO("  Total data transferred: " << total_gb << " GB");
            LOG_INFO("  Iterations completed: " << iterations);

            EXPECT_GT(iterations, 0);
        }

        //----------------------------------------------------------------------
        // PCIe BAR Multi-GPU Tests
        //----------------------------------------------------------------------

        TEST_F(Test__DirectP2P, PCIeBar_MultiGPU_Initialize)
        {
            if (!caps_.canDoDirectP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available";
            }

            DirectP2PEngine engine;
            DeviceId cuda_dev = DeviceId::cuda(0);
            std::vector<DeviceId> rocm_devs = {DeviceId::rocm(0)};

            // Try to add more ROCm devices if available
            if (caps_.discovered_bars.size() > 1)
            {
                rocm_devs.push_back(DeviceId::rocm(1));
            }

            LOG_INFO("\n=== PCIe BAR Multi-GPU Initialization ===");
            bool init = engine.initializePCIeBarMultiGPU(cuda_dev, rocm_devs);

            if (init)
            {
                EXPECT_TRUE(engine.isPCIeBarActive());
                LOG_INFO("Mapped " << engine.getNumMappedBars() << " AMD GPU BARs");
            }
            else
            {
                LOG_WARN("Multi-GPU initialization failed - may need more root permissions");
            }
        }

        TEST_F(Test__DirectP2P, PCIeBar_BroadcastToAMD)
        {
            if (!caps_.canDoDirectP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available";
            }

            DirectP2PEngine engine;
            DeviceId cuda_dev = DeviceId::cuda(0);
            std::vector<DeviceId> rocm_devs = {DeviceId::rocm(0)};

            if (!engine.initializePCIeBarMultiGPU(cuda_dev, rocm_devs))
            {
                // Try device 1 if device 0's BAR isn't accessible
                rocm_devs = {DeviceId::rocm(1)};
                if (!engine.initializePCIeBarMultiGPU(cuda_dev, rocm_devs))
                {
                    GTEST_SKIP() << "Cannot map any AMD GPU BAR (need root permissions)";
                }
            }

            size_t size = 16 * 1024 * 1024; // 16 MB

            // Allocate CUDA buffer
            void *cuda_buf = nullptr;
            cudaMalloc(&cuda_buf, size);
            ASSERT_NE(cuda_buf, nullptr);
            cudaMemset(cuda_buf, 0xAB, size);

            LOG_INFO("\n=== PCIe BAR Broadcast: NVIDIA → AMD ===");
            auto result = engine.broadcastToAMD(cuda_buf, size);

            LOG_INFO("Result:");
            LOG_INFO("  Success: " << result.success);
            LOG_INFO("  Throughput: " << result.throughput_gbps << " GB/s");
            LOG_INFO("  Path: " << result.transfer_path);

            EXPECT_TRUE(result.success);
            EXPECT_GT(result.throughput_gbps, 0.5);

            cudaFree(cuda_buf);
        }

        TEST_F(Test__DirectP2P, PCIeBar_GatherFromAMD)
        {
            if (!caps_.canDoDirectP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available";
            }

            DirectP2PEngine engine;
            DeviceId cuda_dev = DeviceId::cuda(0);
            std::vector<DeviceId> rocm_devs = {DeviceId::rocm(0)};

            if (!engine.initializePCIeBarMultiGPU(cuda_dev, rocm_devs))
            {
                // Try device 1 if device 0's BAR isn't accessible
                rocm_devs = {DeviceId::rocm(1)};
                if (!engine.initializePCIeBarMultiGPU(cuda_dev, rocm_devs))
                {
                    GTEST_SKIP() << "Cannot map any AMD GPU BAR (need root permissions)";
                }
            }

            size_t size = 16 * 1024 * 1024; // 16 MB per GPU

            // Allocate CUDA buffer (big enough for all GPUs)
            void *cuda_buf = nullptr;
            cudaMalloc(&cuda_buf, size * engine.getNumMappedBars());
            ASSERT_NE(cuda_buf, nullptr);

            LOG_INFO("\n=== PCIe BAR Gather: AMD → NVIDIA ===");
            auto result = engine.gatherFromAMDOverlapped(cuda_buf, size);

            LOG_INFO("Result:");
            LOG_INFO("  Success: " << result.success);
            LOG_INFO("  Throughput: " << result.throughput_gbps << " GB/s");
            LOG_INFO("  Path: " << result.transfer_path);

            EXPECT_TRUE(result.success);
            EXPECT_GT(result.throughput_gbps, 0.3); // Reads are slower

            cudaFree(cuda_buf);
        }

        TEST_F(Test__DirectP2P, PCIeBar_BenchmarkAllModes)
        {
            if (!caps_.canDoDirectP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available";
            }

            DirectP2PEngine engine;
            DeviceId cuda_dev = DeviceId::cuda(0);
            std::vector<DeviceId> rocm_devs;

            // Add all discovered AMD GPUs
            for (size_t i = 0; i < caps_.discovered_bars.size() && i < 2; ++i)
            {
                rocm_devs.push_back(DeviceId::rocm(static_cast<int>(i)));
            }

            if (rocm_devs.empty())
            {
                GTEST_SKIP() << "No AMD GPUs found";
            }

            ASSERT_TRUE(engine.initializePCIeBarMultiGPU(cuda_dev, rocm_devs));

            LOG_INFO("\n=== PCIe BAR All Modes Benchmark ===");
            size_t size = 32 * 1024 * 1024; // 32 MB
            auto result = engine.benchmarkAllModes(size, 3);

            EXPECT_TRUE(result.success);
            EXPECT_GT(result.write_gbps, 0.5);
            EXPECT_GT(result.read_gbps, 0.3);
        }

    } // namespace test
} // namespace llaminar2

#else // !HAVE_CUDA || !HAVE_ROCM

// Stub test when backends not available
namespace llaminar2::test
{
    TEST(Test__DirectP2P, RequiresBothBackends)
    {
        GTEST_SKIP() << "DirectP2P requires both CUDA and ROCm backends";
    }
} // namespace llaminar2::test

#endif // HAVE_CUDA && HAVE_ROCM
