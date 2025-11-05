/**
 * Quick test for software pipelining in Phase 5 kernel
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <nvrtc.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>

#define CU_CHECK(call)                                                                                   \
    do                                                                                                   \
    {                                                                                                    \
        CUresult err = call;                                                                             \
        if (err != CUDA_SUCCESS)                                                                         \
        {                                                                                                \
            const char *errStr;                                                                          \
            cuGetErrorString(err, &errStr);                                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << errStr << std::endl; \
            exit(1);                                                                                     \
        }                                                                                                \
    } while (0)

#define NVRTC_CHECK(call)                                                                                                   \
    do                                                                                                                      \
    {                                                                                                                       \
        nvrtcResult err = call;                                                                                             \
        if (err != NVRTC_SUCCESS)                                                                                           \
        {                                                                                                                   \
            std::cerr << "NVRTC error at " << __FILE__ << ":" << __LINE__ << ": " << nvrtcGetErrorString(err) << std::endl; \
            exit(1);                                                                                                        \
        }                                                                                                                   \
    } while (0)

int main()
{
    // Initialize CUDA Driver API
    CU_CHECK(cuInit(0));

    CUdevice device;
    CUcontext context;
    CU_CHECK(cuDeviceGet(&device, 0));
    CU_CHECK(cuCtxCreate(&context, 0, device));

    // Test parameters
    const int M = 1024;
    const int N = 896;
    const int K = 896;

    std::cout << "Testing Software Pipelining in Phase 5 Kernel\n";
    std::cout << "Matrix: " << M << "×" << N << "×" << K << "\n\n";

    // Read the Phase 5 template header
    std::ifstream template_file("src/v2/kernels/cuda/CudaGemmKernelTemplatePhase5.h");
    if (!template_file.is_open())
    {
        std::cerr << "Could not open CudaGemmKernelTemplatePhase5.h\n";
        return 1;
    }

    std::stringstream buffer;
    buffer << template_file.rdbuf();
    std::string template_content = buffer.str();

    // Find and extract the kernel source between PHASE5_GEMM_KERNEL_TEMPLATE markers
    size_t start_pos = template_content.find("const char *PHASE5_GEMM_KERNEL_TEMPLATE = R\"(");
    if (start_pos == std::string::npos)
    {
        std::cerr << "Could not find PHASE5_GEMM_KERNEL_TEMPLATE\n";
        return 1;
    }
    start_pos += 47; // Skip the marker

    size_t end_pos = template_content.find(")\";\n", start_pos);
    if (end_pos == std::string::npos)
    {
        std::cerr << "Could not find end of PHASE5_GEMM_KERNEL_TEMPLATE\n";
        return 1;
    }

    std::string kernel_source = template_content.substr(start_pos, end_pos - start_pos);

    std::cout << "Extracted kernel source: " << kernel_source.length() << " bytes\n";

    // Extract headers too
    start_pos = template_content.find("const char *PHASE5_CUTE_HEADERS = R\"(");
    if (start_pos == std::string::npos)
    {
        std::cerr << "Could not find PHASE5_CUTE_HEADERS\n";
        return 1;
    }
    start_pos += 38;

    end_pos = template_content.find(")\";\n", start_pos);
    std::string headers = template_content.substr(start_pos, end_pos - start_pos);

    // Combine with configuration
    std::string full_source = headers + "\n\n" +
                              "#define TILE_M 64\n"
                              "#define TILE_N 64\n"
                              "#define TILE_K 64\n"
                              "#define SUB_K 16\n"
                              "#define MMA_M 2\n"
                              "#define MMA_N 2\n"
                              "#define BUFFER_STAGES 2\n"
                              "#define THREADS_PER_BLOCK 128\n"
                              "#define SWIZZLE_B 3\n"
                              "#define SWIZZLE_M 3\n"
                              "#define SWIZZLE_S 3\n"
                              "#define VECTORIZE_A 4\n"
                              "#define USE_TRANSPOSED_A 0\n\n" +
                              kernel_source;

    std::cout << "Full source: " << full_source.length() << " bytes\n";
    std::cout << "Compiling with NVRTC...\n";

    // Compile with NVRTC
    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, full_source.c_str(), "phase5_kernel.cu", 0, nullptr, nullptr));

    const char *opts[] = {
        "--gpu-architecture=compute_80",
        "--std=c++17",
        "-I/usr/local/cuda/include",
        "--use_fast_math",
        "-DNDEBUG"};

    nvrtcResult compile_result = nvrtcCompileProgram(prog, 5, opts);

    // Get log
    size_t log_size;
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
    if (log_size > 1)
    {
        std::vector<char> log(log_size);
        NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data()));
        std::cout << "NVRTC Log:\n"
                  << log.data() << "\n";
    }

    if (compile_result != NVRTC_SUCCESS)
    {
        std::cerr << "Compilation failed\n";
        return 1;
    }

    // Get PTX
    size_t ptx_size;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
    std::vector<char> ptx(ptx_size);
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
    nvrtcDestroyProgram(&prog);

    std::cout << "Compilation successful! PTX size: " << ptx_size << " bytes\n\n";

    // Load module
    CUmodule module;
    CU_CHECK(cuModuleLoadDataEx(&module, ptx.data(), 0, nullptr, nullptr));

    CUfunction function;
    CU_CHECK(cuModuleGetFunction(&function, module, "iq4nl_gemm_phase5_kernel"));

    std::cout << "Kernel loaded successfully\n";
    std::cout << "TODO: Allocate memory and benchmark...\n";

    // Cleanup
    CU_CHECK(cuModuleUnload(module));
    CU_CHECK(cuCtxDestroy(context));

    return 0;
}
