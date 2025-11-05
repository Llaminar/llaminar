/**
 * @file CudaGemmJIT.cu
 * @brief Implementation of JIT compilation for CUDA GEMM kernels
 */

#include "CudaGemmJIT.h"
#include "CudaGemmKernelTemplate.h"
#include <nvrtc.h>
#include <cuda.h>
#include <vector> // Required for NVCC
#include <string> // Required for NVCC
#include <map>    // Required for NVCC
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <sys/stat.h>

namespace llaminar2
{
    namespace cuda
    {

        namespace fs = std::filesystem;

// Helper macros for error checking
#define NVRTC_CHECK(call)                                                                       \
    do                                                                                          \
    {                                                                                           \
        nvrtcResult _res = (call);                                                              \
        if (_res != NVRTC_SUCCESS)                                                              \
        {                                                                                       \
            throw std::runtime_error(std::string("NVRTC error: ") + nvrtcGetErrorString(_res)); \
        }                                                                                       \
    } while (0)

#define CU_CHECK(call)                                                       \
    do                                                                       \
    {                                                                        \
        CUresult _res = (call);                                              \
        if (_res != CUDA_SUCCESS)                                            \
        {                                                                    \
            const char *err_str;                                             \
            cuGetErrorString(_res, &err_str);                                \
            throw std::runtime_error(std::string("CUDA error: ") + err_str); \
        }                                                                    \
    } while (0)

        CudaGemmJIT &CudaGemmJIT::instance()
        {
            static CudaGemmJIT inst;
            return inst;
        }

        CudaGemmJIT::CudaGemmJIT()
        {
            // Initialize CUDA driver API
            CU_CHECK(cuInit(0));

            // Get GPU architecture
            gpu_arch_ = getGPUArchitecture();

            // Set up cache directory
            const char *home = std::getenv("HOME");
            if (home)
            {
                cache_dir_ = std::string(home) + "/.cache/llaminar/cuda_kernels/" + gpu_arch_;
            }
            else
            {
                cache_dir_ = "/tmp/llaminar_cuda_cache/" + gpu_arch_;
            }

            ensureCacheDirectory();
        }

        CUfunction CudaGemmJIT::getKernel(const CudaGemmConfig &config)
        {
            std::string key = getCacheKey(config);

            // Check in-memory cache first
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                auto it = kernel_cache_.find(key);
                if (it != kernel_cache_.end())
                {
                    stats_.memory_hits++;
                    return it->second->function;
                }
            }

            // Try disk cache
            if (auto cached = loadFromDiskCache(config))
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                auto shared_kernel = std::make_shared<CompiledKernel>(std::move(*cached));
                CUfunction func = shared_kernel->function;
                kernel_cache_[key] = shared_kernel;
                stats_.disk_hits++;
                return func;
            }

            // Compile from source
            auto start = std::chrono::high_resolution_clock::now();

            CompiledKernel compiled = compileKernel(config);

            // Check if compilation succeeded (may fail due to resource constraints)
            if (!compiled.isValid())
            {
                stats_.compilation_failures++;
                return nullptr; // Return null to signal failure
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            // Cache in memory
            std::lock_guard<std::mutex> lock(cache_mutex_);
            auto shared_kernel = std::make_shared<CompiledKernel>(std::move(compiled));
            CUfunction func = shared_kernel->function;
            kernel_cache_[key] = shared_kernel;
            stats_.compiles++;

            return func;
        }

        CompiledKernel CudaGemmJIT::compileKernel(const CudaGemmConfig &config)
        {
            try
            {
                // Generate specialized source code
                std::string source = generateKernelSource(config);
                std::cout << "[CudaGemmJIT] Generated source: " << source.length() << " bytes" << std::endl;

                // Debug: Save generated source to disk for inspection
                static bool debug_dump = std::getenv("CUDA_JIT_DEBUG_DUMP") != nullptr;
                if (debug_dump)
                {
                    std::string debug_path = cache_dir_ + "/debug_source_" + getCacheKey(config) + ".cu";
                    std::ofstream debug_file(debug_path);
                    debug_file << source;
                    debug_file.close();
                    std::cout << "[CudaGemmJIT] DEBUG: Saved generated source to " << debug_path << std::endl;
                }

                // Create NVRTC program
                nvrtcProgram prog;
                nvrtcResult create_res = nvrtcCreateProgram(&prog, source.c_str(),
                                                            "gemm_kernel.cu", 0, nullptr, nullptr);

                if (create_res != NVRTC_SUCCESS)
                {
                    throw std::runtime_error(std::string("nvrtcCreateProgram failed: ") + nvrtcGetErrorString(create_res));
                }

                // Compile options (match your build configuration)
                // IMPORTANT: Create arch string separately to avoid dangling pointer
                std::string arch_option = "--gpu-architecture=" + gpu_arch_;
                std::vector<const char *> opts = {
                    arch_option.c_str(),
                    "-std=c++17",
                    "--use_fast_math",
                    "--extra-device-vectorization"};

                // Compile
                nvrtcResult compile_res = nvrtcCompileProgram(prog, opts.size(), opts.data());

                // Check for compilation errors
                if (compile_res != NVRTC_SUCCESS)
                {
                    size_t log_size;
                    nvrtcGetProgramLogSize(prog, &log_size);
                    std::string log(log_size, '\0');
                    nvrtcGetProgramLog(prog, &log[0]);
                    nvrtcDestroyProgram(&prog);

                    // Check if this is a resource constraint failure (not an error)
                    // "uses too much shared data", "too many registers", etc.
                    if (log.find("uses too much") != std::string::npos ||
                        log.find("too many registers") != std::string::npos)
                    {
                        // Return empty CompiledKernel (null module/function) to signal graceful skip
                        // Note: This is expected for ~30% of configurations due to GPU hardware limits
                        return CompiledKernel(nullptr, nullptr);
                    }

                    // Actual compilation error - throw exception
                    throw std::runtime_error("Kernel compilation failed:\n" + log);
                }

                // Get compiled CUBIN
                size_t cubin_size;
                nvrtcResult cubin_res = nvrtcGetCUBINSize(prog, &cubin_size);

                if (cubin_res != NVRTC_SUCCESS)
                {
                    // Fall back to PTX
                    size_t ptx_size;
                    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
                    std::vector<char> ptx(ptx_size);
                    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));

                    // For PTX, we need to load as PTX, not CUBIN
                    // Save to disk cache
                    saveToDiskCache(config, ptx.data(), ptx_size);

                    // Load module from PTX
                    CUmodule module;
                    CU_CHECK(cuModuleLoadDataEx(&module, ptx.data(), 0, nullptr, nullptr));

                    // Get kernel function
                    CUfunction kernel;
                    CU_CHECK(cuModuleGetFunction(&kernel, module, "quantized_gemm_kernel_iq4nl"));

                    nvrtcDestroyProgram(&prog);
                    return CompiledKernel(module, kernel);
                }

                std::vector<char> cubin(cubin_size);
                nvrtcResult get_cubin_res = nvrtcGetCUBIN(prog, cubin.data());
                NVRTC_CHECK(get_cubin_res);

                // Save to disk cache
                saveToDiskCache(config, cubin.data(), cubin_size);

                // Load module
                CUmodule module;
                CU_CHECK(cuModuleLoadData(&module, cubin.data()));

                // Get kernel function
                CUfunction kernel;
                CU_CHECK(cuModuleGetFunction(&kernel, module, "quantized_gemm_kernel_iq4nl"));

                nvrtcDestroyProgram(&prog);

                return CompiledKernel(module, kernel);
            }
            catch (const std::exception &e)
            {
                std::cerr << "[CudaGemmJIT] Exception in compileKernel: " << e.what() << std::endl;
                throw;
            }
        }

        std::string CudaGemmJIT::generateKernelSource(const CudaGemmConfig &config)
        {
            std::string source = GEMM_KERNEL_TEMPLATE;

            // Replace placeholders
            auto replace = [&source](const std::string &placeholder, const std::string &value)
            {
                size_t pos = 0;
                while ((pos = source.find(placeholder, pos)) != std::string::npos)
                {
                    source.replace(pos, placeholder.length(), value);
                    pos += value.length();
                }
            };

            // Substitute decoder source first
            replace("${DECODER_SOURCE}", IQ4_NL_DECODER_SOURCE);

            // Substitute configuration parameters
            replace("${TILE_M}", std::to_string(config.tile_m));
            replace("${TILE_N}", std::to_string(config.tile_n));
            replace("${TILE_K}", std::to_string(config.tile_k));
            replace("${THREADS_M}", std::to_string(config.threads_m));
            replace("${THREADS_N}", std::to_string(config.threads_n));
            replace("${WORK_M}", std::to_string(config.work_per_thread_m));
            replace("${WORK_N}", std::to_string(config.work_per_thread_n));
            replace("${PREFETCH_STAGES}", std::to_string(config.prefetch_stages));
            replace("${TRANSPOSE_SMEM}", config.transpose_smem ? "true" : "false");
            replace("${VECTORIZE_LOAD}", std::to_string(config.vectorize_load));

            return source;
        }

        std::optional<CompiledKernel> CudaGemmJIT::loadFromDiskCache(const CudaGemmConfig &config)
        {
            std::string path = getCachePath(config);

            if (!fs::exists(path))
            {
                return std::nullopt;
            }

            // Read CUBIN from file
            std::ifstream file(path, std::ios::binary);
            if (!file)
            {
                return std::nullopt;
            }

            file.seekg(0, std::ios::end);
            size_t size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<char> cubin(size);
            file.read(cubin.data(), size);

            if (!file)
            {
                return std::nullopt;
            }

            // Load module
            CUmodule module;
            CUresult res = cuModuleLoadData(&module, cubin.data());
            if (res != CUDA_SUCCESS)
            {
                // Cache corruption - delete and return nullopt
                fs::remove(path);
                return std::nullopt;
            }

            // Get function
            CUfunction kernel;
            res = cuModuleGetFunction(&kernel, module, "quantized_gemm_kernel_iq4nl");
            if (res != CUDA_SUCCESS)
            {
                cuModuleUnload(module);
                fs::remove(path);
                return std::nullopt;
            }

            return CompiledKernel(module, kernel);
        }

        void CudaGemmJIT::saveToDiskCache(const CudaGemmConfig &config, const void *cubin, size_t size)
        {
            std::string path = getCachePath(config);

            // Ensure parent directory exists
            fs::create_directories(fs::path(path).parent_path());

            // Write CUBIN to file
            std::ofstream file(path, std::ios::binary);
            if (file)
            {
                file.write(static_cast<const char *>(cubin), size);
                stats_.cache_size_bytes += size;
            }
        }

        std::string CudaGemmJIT::getCacheKey(const CudaGemmConfig &config) const
        {
            std::ostringstream oss;
            oss << config.tile_m << "_" << config.tile_n << "_" << config.tile_k << "_"
                << config.threads_m << "_" << config.threads_n << "_"
                << config.work_per_thread_m << "_" << config.work_per_thread_n << "_"
                << config.prefetch_stages << "_" << (config.transpose_smem ? 1 : 0) << "_"
                << config.vectorize_load;
            return oss.str();
        }

        std::string CudaGemmJIT::getCachePath(const CudaGemmConfig &config) const
        {
            return cache_dir_ + "/gemm_" + getCacheKey(config) + ".cubin";
        }

        std::string CudaGemmJIT::getGPUArchitecture() const
        {
            CUdevice device;
            CU_CHECK(cuDeviceGet(&device, 0));

            int major, minor;
            CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
            CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

            // Use sm_XX (real architecture) instead of compute_XX (virtual architecture)
            // to get CUBIN output from NVRTC instead of PTX
            return "sm_" + std::to_string(major) + std::to_string(minor);
        }

        void CudaGemmJIT::ensureCacheDirectory()
        {
            fs::create_directories(cache_dir_);
        }

        void CudaGemmJIT::precompile(const std::vector<CudaGemmConfig> &configs)
        {
            std::cout << "[CudaGemmJIT] Precompiling " << configs.size() << " kernels..." << std::endl;

            for (const auto &config : configs)
            {
                getKernel(config); // Will compile if not cached
            }

            std::cout << "[CudaGemmJIT] Precompilation complete" << std::endl;
        }

        void CudaGemmJIT::clearMemoryCache()
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            kernel_cache_.clear();
        }

        void CudaGemmJIT::clearDiskCache()
        {
            if (fs::exists(cache_dir_))
            {
                fs::remove_all(cache_dir_);
                ensureCacheDirectory();
            }
            stats_.cache_size_bytes = 0;
        }

        CudaGemmJIT::CacheStats CudaGemmJIT::getStats() const
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            return stats_;
        }

    } // namespace cuda
} // namespace llaminar2
