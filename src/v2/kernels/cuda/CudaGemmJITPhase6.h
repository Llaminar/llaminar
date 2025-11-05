/**
 * @file CudaGemmJITPhase6.h
 * @brief JIT compiler for Phase 6 Int8 DP4A kernel using raw NVRTC
 *
 * @author David Sanftenberg
 * @date November 5, 2025
 */

#ifndef LLAMINAR2_CUDA_GEMM_JIT_PHASE6_H
#define LLAMINAR2_CUDA_GEMM_JIT_PHASE6_H

#include "CudaGemmConfigPhase6.h"
#include "CudaGemmKernelPhase6_Int8.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>
#include <string>
#include <stdexcept>
#include <sstream>

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief JIT compiler for Phase 6 Int8 DP4A kernel
         *
         * Uses raw NVRTC for minimal overhead
         */
        class CudaGemmJITPhase6
        {
        public:
            struct CompiledKernel
            {
                CUfunction function;
                CUmodule module;

                // Launch helper
                void launch(dim3 grid, dim3 block, const float *A, const void *B, float *C,
                            int M, int N, int K, cudaStream_t stream = 0)
                {
                    void *args[] = {(void *)&A, (void *)&B, (void *)&C, (void *)&M, (void *)&N, (void *)&K};

                    cuLaunchKernel(function,
                                   grid.x, grid.y, grid.z,
                                   block.x, block.y, block.z,
                                   0, stream,
                                   args, nullptr);
                }
            };

            /**
             * @brief Compile Phase 6 kernel with given configuration
             *
             * @param config Configuration (tile sizes, thread count)
             * @return Compiled kernel
             */
            static CompiledKernel compile(const CudaGemmConfigPhase6 &config)
            {
                // 1. Build complete kernel source
                std::string kernel_source = R"(
#include <cuda_fp16.h>

// Type definitions (NVRTC doesn't support <cstdint>)
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef int int32_t;

)";
                // Add headers
                kernel_source += PHASE6_INT8_HEADERS;
                kernel_source += "\n\n";

                // Add kernel template with substitutions
                std::string kernel_code = PHASE6_INT8_KERNEL_TEMPLATE;
                kernel_code = replace_all(kernel_code, "${TILE_M}", std::to_string(config.tile_m));
                kernel_code = replace_all(kernel_code, "${TILE_N}", std::to_string(config.tile_n));
                kernel_code = replace_all(kernel_code, "${TILE_K}", std::to_string(config.tile_k));
                kernel_code = replace_all(kernel_code, "${THREADS_PER_BLOCK}",
                                          std::to_string(config.threads_per_block));

                kernel_source += kernel_code;

                // 2. Create NVRTC program
                nvrtcProgram prog;

                // Add CUDA include paths
                const char *headers[] = {
                    "cuda_fp16.h",
                    "cuda_runtime.h"};
                const char *include_names[] = {
                    "cuda_fp16.h",
                    "cuda_runtime.h"};

                nvrtcResult create_result = nvrtcCreateProgram(&prog, kernel_source.c_str(),
                                                               "phase6_int8_kernel.cu",
                                                               0, nullptr, nullptr);
                if (create_result != NVRTC_SUCCESS)
                {
                    throw std::runtime_error("Failed to create NVRTC program");
                }

                // 3. Compile with optimizations and include paths
                const char *opts[] = {
                    "-std=c++17",
                    "--use_fast_math",
                    "-DNDEBUG",
                    "-arch=compute_86", // RTX 3090
                    "-I/usr/local/cuda/include"};

                nvrtcResult compile_result = nvrtcCompileProgram(prog, 5, opts);

                // Check for errors
                if (compile_result != NVRTC_SUCCESS)
                {
                    size_t log_size;
                    nvrtcGetProgramLogSize(prog, &log_size);
                    std::string log(log_size, '\0');
                    nvrtcGetProgramLog(prog, &log[0]);
                    nvrtcDestroyProgram(&prog);

                    std::stringstream ss;
                    ss << "Phase 6 kernel compilation failed:\n"
                       << log;
                    throw std::runtime_error(ss.str());
                }

                // 4. Get PTX
                size_t ptx_size;
                nvrtcGetPTXSize(prog, &ptx_size);
                std::string ptx(ptx_size, '\0');
                nvrtcGetPTX(prog, &ptx[0]);
                nvrtcDestroyProgram(&prog);

                // 5. Load into CUDA module
                CompiledKernel kernel;
                cuModuleLoadDataEx(&kernel.module, ptx.c_str(), 0, 0, 0);
                cuModuleGetFunction(&kernel.function, kernel.module, "iq4nl_gemm_phase6_int8_kernel");

                return kernel;
            }

        private:
            static std::string replace_all(std::string str, const std::string &from, const std::string &to)
            {
                size_t start_pos = 0;
                while ((start_pos = str.find(from, start_pos)) != std::string::npos)
                {
                    str.replace(start_pos, from.length(), to);
                    start_pos += to.length();
                }
                return str;
            }
        };

    } // namespace cuda
} // namespace llaminar2

#endif // LLAMINAR2_CUDA_GEMM_JIT_PHASE6_H
