// Minimal reproduction case for NVRTC + CuTe compilation issue
// Goal: Understand why decltype(*declval<T&>()) fails with cutlass::half_t

#include <cuda.h>
#include <nvrtc.h>
#include <iostream>
#include <string>
#include <vector>

#define NVRTC_CHECK(call)                                                    \
    do                                                                       \
    {                                                                        \
        nvrtcResult result = call;                                           \
        if (result != NVRTC_SUCCESS)                                         \
        {                                                                    \
            std::cerr << "NVRTC error: " << nvrtcGetErrorString(result)      \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            if (result == NVRTC_ERROR_COMPILATION)                           \
            {                                                                \
                size_t log_size;                                             \
                nvrtcGetProgramLogSize(prog, &log_size);                     \
                std::vector<char> log(log_size);                             \
                nvrtcGetProgramLog(prog, log.data());                        \
                std::cerr << "Compilation log:\n"                            \
                          << log.data() << std::endl;                        \
            }                                                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// Test 1: Absolute minimal CuTe usage
const char *test1_source = R"(
#include <cute/tensor.hpp>

__global__ void test_kernel() {
    // Do nothing - just test if CuTe headers parse
}
)";

// Test 2: Create smem_ptr (this is where it fails)
const char *test2_source = R"(
#include <cuda_fp16.h>
#include <cute/pointer.hpp>

using namespace cute;

__global__ void test_kernel() {
    extern __shared__ half smem_raw[];
    auto smem = make_smem_ptr(smem_raw);  // FAILS: decltype(*declval<half*>())
}
)";

// Test 3: Even simpler - just include cute/pointer_base.hpp
const char *test3_source = R"(
#include <cuda_fp16.h>
#include <cute/pointer_base.hpp>

template<typename T>
struct test_iter_ref {
    using type = decltype(*cute::declval<T&>());  // Use cute::declval
};

// Instantiate with half
template struct test_iter_ref<half>;

__global__ void test_kernel() {}
)";

// Test 5: Full make_tensor with swizzle layout (the actual failure case)
const char *test5_source = R"(
#include <cuda_fp16.h>
#include <cute/tensor.hpp>

using namespace cute;

__global__ void test_kernel() {
    extern __shared__ half smem_raw[];
    
    // This is what fails in our actual kernel
    auto smem = make_smem_ptr(smem_raw);
    auto layout = composition(Swizzle<3,3,3>{}, Layout<Shape<_128,_64>, Stride<_64,_1>>{});
    auto tCsB = make_tensor(smem, layout);  // Works!
}
)";

// Test 6: The actual bug - passing reference instead of pointer
const char *test6_source = R"(
#include <cuda_fp16.h>
#include <cute/tensor.hpp>

using namespace cute;

__global__ void test_kernel() {
    __shared__ half s_B[64][64];
    
    // WRONG: s_B[0][ki] is a half&, not half*
    int ki = 0;
    auto layout = composition(Swizzle<3,3,3>{}, Layout<Shape<_64,_64>, Stride<_64,_1>>{});
    auto sB_tensor = make_tensor(make_smem_ptr(s_B[0][ki]), layout);  // FAILS
}
)";

// Test 7: The fix - pass pointer, not reference
const char *test7_source = R"(
#include <cuda_fp16.h>
#include <cute/tensor.hpp>

using namespace cute;

__global__ void test_kernel() {
    __shared__ half s_B[64][64];
    
    // CORRECT: &s_B[0][ki] is a half*, or better: s_B[0] + ki
    int ki = 0;
    auto layout = composition(Swizzle<3,3,3>{}, Layout<Shape<_64,_64>, Stride<_64,_1>>{});
    auto sB_tensor = make_tensor(make_smem_ptr(&s_B[0][ki]), layout);  // Works!
}
)";

// Test 4: Workaround attempt - avoid decltype in template context
const char *test4_source = R"(
#include <cuda_fp16.h>

// Manual iter_ref implementation avoiding decltype in constexpr
template<typename T>
struct simple_iter_ref {
    // Don't use decltype - just assume T* dereferences to T&
    using type = T&;
};

template<typename P>
struct simple_smem_ptr {
    using reference = typename simple_iter_ref<P>::type;
    
    __device__ simple_smem_ptr(P* ptr) : ptr_(ptr) {}
    __device__ reference operator*() const { return *ptr_; }
    
    P* ptr_;
};

__global__ void test_kernel() {
    extern __shared__ half smem_raw[];
    simple_smem_ptr<half> smem(smem_raw);
    half val = *smem;  // Should work
}
)";

void compile_test(const char *name, const char *source)
{
    std::cout << "\n========================================\n";
    std::cout << "Testing: " << name << "\n";
    std::cout << "========================================\n";

    nvrtcProgram prog;
    nvrtcResult result;

    result = nvrtcCreateProgram(&prog, source, "test.cu", 0, nullptr, nullptr);
    if (result != NVRTC_SUCCESS)
    {
        std::cerr << "Failed to create program: " << nvrtcGetErrorString(result) << std::endl;
        return;
    }

    std::vector<const char *> opts = {
        "--gpu-architecture=compute_86",
        "-std=c++17",
        "--include-path=/opt/cutlass/include",
        "--include-path=/usr/local/cuda/include",
        "-default-device"};

    result = nvrtcCompileProgram(prog, opts.size(), opts.data());

    if (result == NVRTC_SUCCESS)
    {
        std::cout << "✓ SUCCESS: Kernel compiled!\n";
    }
    else
    {
        std::cout << "✗ FAILED: Compilation error\n";
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::vector<char> log(log_size);
        nvrtcGetProgramLog(prog, log.data());
        std::cout << "Error log:\n"
                  << log.data() << std::endl;
    }

    nvrtcDestroyProgram(&prog);
}

int main()
{
    std::cout << "NVRTC + CuTe Debugging Session\n";
    std::cout << "==============================\n";

    // Start simple, increase complexity
    compile_test("Test 1: Include CuTe headers only", test1_source);
    compile_test("Test 2: Create smem_ptr", test2_source);
    compile_test("Test 3: Isolated decltype test", test3_source);
    compile_test("Test 4: Workaround without decltype", test4_source);
    compile_test("Test 5: Full make_tensor with swizzle", test5_source);
    compile_test("Test 6: Bug - pass reference instead of pointer", test6_source);
    compile_test("Test 7: Fix - pass pointer with &", test7_source);

    return 0;
}
