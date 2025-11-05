/**
 * @file CudaGemmJITPhase5.cu
 * @brief Implementation of Phase 5 JIT compiler
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#include "CudaGemmJITPhase5.h"
#include "CudaGemmKernelTemplatePhase5.h"
#include "jitify.hpp"
#include <cuda.h>
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include <cstdlib>

namespace llaminar2
{
    namespace cuda
    {

        namespace fs = std::filesystem;

// CUDA error checking macro
#define CU_CHECK(call)                                                         \
    do                                                                         \
    {                                                                          \
        CUresult result = call;                                                \
        if (result != CUDA_SUCCESS)                                            \
        {                                                                      \
            const char *error_str;                                             \
            cuGetErrorString(result, &error_str);                              \
            throw std::runtime_error(std::string("CUDA error: ") + error_str); \
        }                                                                      \
    } while (0)

        CudaGemmJITPhase5 &CudaGemmJITPhase5::instance()
        {
            static CudaGemmJITPhase5 inst;
            return inst;
        }

        CudaGemmJITPhase5::CudaGemmJITPhase5()
        {
            CU_CHECK(cuInit(0));
            gpu_arch_ = getGPUArchitecture();
        }

        CUfunction CudaGemmJITPhase5::getKernel(const CudaGemmConfigPhase5 &config)
        {
            if (!config.is_valid())
            {
                throw std::runtime_error("Invalid Phase 5 configuration: " + config.config_id());
            }

            std::string cache_key = getCacheKey(config);

            // Memory cache lookup
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                auto it = kernel_cache_.find(cache_key);
                if (it != kernel_cache_.end())
                {
                    stats_.memory_hits++;
                    return it->second->function;
                }
            }

            // Compile from source
            auto compiled = compileKernel(config);
            if (!compiled.isValid())
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                stats_.compilation_failures++;
                throw std::runtime_error("Kernel compilation failed for config: " + cache_key);
            }

            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                stats_.compiles++;
                kernel_cache_[cache_key] = std::make_shared<CompiledKernelPhase5>(std::move(compiled));
                return kernel_cache_[cache_key]->function;
            }
        }

        std::string CudaGemmJITPhase5::generateKernelSource(const CudaGemmConfigPhase5 &config)
        {
            std::string source = PHASE5_GEMM_KERNEL_TEMPLATE;

            // Helper lambda for string replacement
            auto replace = [&source](const std::string &placeholder, const std::string &value)
            {
                size_t pos = 0;
                while ((pos = source.find(placeholder, pos)) != std::string::npos)
                {
                    source.replace(pos, placeholder.length(), value);
                    pos += value.length();
                }
            };

            // Substitute CuTe headers first
            replace("${CUTE_HEADERS}", PHASE5_CUTE_HEADERS);

            // Substitute configuration parameters
            replace("${TILE_M}", std::to_string(config.tile_m));
            replace("${TILE_N}", std::to_string(config.tile_n));
            replace("${TILE_K}", std::to_string(config.tile_k));
            replace("${SUB_K}", std::to_string(config.sub_k));
            replace("${MMA_M}", std::to_string(config.mma_m));
            replace("${MMA_N}", std::to_string(config.mma_n));
            replace("${BUFFER_STAGES}", std::to_string(config.buffer_stages));
            replace("${THREADS_PER_BLOCK}", std::to_string(config.threads_per_block));
            replace("${SWIZZLE_B}", std::to_string(config.swizzle_b));
            replace("${SWIZZLE_M}", std::to_string(config.swizzle_m));
            replace("${SWIZZLE_S}", std::to_string(config.swizzle_s));
            replace("${VECTORIZE_A}", std::to_string(config.vectorize_a));

            return source;
        }

        CompiledKernelPhase5 CudaGemmJITPhase5::compileKernel(const CudaGemmConfigPhase5 &config)
        {
            try
            {
                std::string source = generateKernelSource(config);

                const char *kernel_name = "iq4nl_gemm_phase5_kernel";

                // Compilation options for Jitify
                std::string arch_option = "--gpu-architecture=" + gpu_arch_;
                std::vector<std::string> opts = {
                    arch_option,
                    "-std=c++17",
                    "--use_fast_math",
                    "--extra-device-vectorization",
                    "-default-device",           // Treat unannotated functions as __device__
                    " -I/opt/cutlass/include",   // CuTe and CUTLASS headers (note leading space!)
                    " -I/usr/local/cuda/include" // CUDA headers (note leading space!)
                };

                // Add transpose_b define if enabled (NCU-guided coalescing optimization)
                if (config.transpose_b)
                {
                    opts.push_back("-DTRANSPOSE_B_LAYOUT=1");
                }

                // Use Jitify to compile the kernel
                // NOTE: Using local static here is intentional - Jitify's JitCache has its own
                // internal persistence. This is fine for Phase 5 since we have our own
                // 2-level cache (memory + disk) wrapping Jitify.
                static jitify::JitCache kernel_cache;

                jitify::Program program;
                try
                {
                    program = kernel_cache.program(source, {}, opts);
                }
                catch (const std::runtime_error &e)
                {
                    // Check if this is a resource constraint failure
                    std::string error_msg = e.what();
                    if (error_msg.find("uses too much") != std::string::npos ||
                        error_msg.find("too many registers") != std::string::npos)
                    {
                        // Return empty CompiledKernelPhase5 to signal graceful skip
                        return CompiledKernelPhase5(nullptr, nullptr);
                    }

                    // Actual compilation error
                    std::cerr << "[CudaGemmJITPhase5] Jitify compilation failed:\n"
                              << error_msg << std::endl;
                    throw std::runtime_error("Phase 5 kernel compilation failed:\n" + error_msg);
                }

                // Instantiate the kernel (no template parameters needed)
                auto kernel_inst = program.kernel(kernel_name).instantiate();

                // NOTE: Jitify only provides PTX (not CUBIN)
                // PTX requires JIT compilation on load (~100-200ms first time, then cached by driver)
                const std::string &ptx = kernel_inst.ptx();

                // Load module from PTX (driver will JIT compile and cache)
                CUmodule module;
                CU_CHECK(cuModuleLoadDataEx(&module, ptx.data(), 0, nullptr, nullptr));

                // Get kernel function using mangled name from Jitify
                CUfunction function;
                const std::string &mangled_name = kernel_inst.mangled_name();
                CU_CHECK(cuModuleGetFunction(&function, module, mangled_name.c_str()));

                return CompiledKernelPhase5(module, function);
            }
            catch (const std::exception &e)
            {
                std::cerr << "[CudaGemmJITPhase5] Exception in compileKernel: " << e.what() << std::endl;
                throw;
            }
        }

        std::string CudaGemmJITPhase5::getCacheKey(const CudaGemmConfigPhase5 &config) const
        {
            return config.config_id();
        }

        std::string CudaGemmJITPhase5::getGPUArchitecture() const
        {
            CUdevice device;
            CU_CHECK(cuDeviceGet(&device, 0));

            int major, minor;
            CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
            CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

            return "sm_" + std::to_string(major) + std::to_string(minor);
        }

        void CudaGemmJITPhase5::clearMemoryCache()
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            kernel_cache_.clear();
        }

        void CudaGemmJITPhase5::precompile(const std::vector<CudaGemmConfigPhase5> &configs)
        {
            for (const auto &config : configs)
            {
                try
                {
                    getKernel(config); // Will cache if successful
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Precompile failed for " << config.config_id()
                              << ": " << e.what() << std::endl;
                }
            }
        }

        CudaGemmJITPhase5::CacheStats CudaGemmJITPhase5::getStats() const
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            return stats_;
        }

    } // namespace cuda
} // namespace llaminar2
