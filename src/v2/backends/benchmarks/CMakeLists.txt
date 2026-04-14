# Device Benchmark Library
#
# Provides lightweight benchmarks for measuring:
# - CPU memory bandwidth and compute throughput
# - GPU memory bandwidth (CUDA/ROCm)
# - Host-device transfer rates
# - Peer-to-peer transfer rates (NVLink, xGMI)
#
# This helps make informed placement decisions based on actual
# measured capabilities rather than estimates.

# Core source files (always compiled)
set(BENCHMARK_SOURCES
    DeviceBenchmark.cpp
    CPUBenchmark.cpp
)

# Header files
set(BENCHMARK_HEADERS
    DeviceBenchmark.h
    CPUBenchmark.h
)

# Conditionally add CUDA sources
if(HAVE_CUDA)
    list(APPEND BENCHMARK_SOURCES CUDABenchmark.cu)
    list(APPEND BENCHMARK_HEADERS CUDABenchmark.h)
endif()

# Conditionally add ROCm sources
if(HAVE_ROCM)
    list(APPEND BENCHMARK_SOURCES ROCmBenchmark.cpp)
    list(APPEND BENCHMARK_HEADERS ROCmBenchmark.h)
endif()

# Cross-vendor P2P requires both CUDA and ROCm
if(HAVE_CUDA AND HAVE_ROCM)
    list(APPEND BENCHMARK_SOURCES
        CrossVendorP2P.cpp
        CrossVendorP2P_CUDA.cu
        CrossVendorP2P_ROCm.cpp
    )
    list(APPEND BENCHMARK_HEADERS
        CrossVendorP2P.h
        CrossVendorP2P_CUDA.h
        CrossVendorP2P_ROCm.h
    )
endif()

# Create object library for inclusion in main library
add_library(device_benchmarks OBJECT
    ${BENCHMARK_SOURCES}
    ${BENCHMARK_HEADERS}
)

target_include_directories(device_benchmarks
    PUBLIC
        ${CMAKE_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(device_benchmarks
    PRIVATE
        OpenMP::OpenMP_CXX
)

if(HAVE_CUDA)
    # Need both cudart (runtime) and cuda (driver) for DMA-BUF operations
    target_link_libraries(device_benchmarks PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(device_benchmarks PRIVATE HAVE_CUDA=1)
endif()

if(HAVE_ROCM)
    find_package(hip REQUIRED)
    target_link_libraries(device_benchmarks PRIVATE hip::host)
    target_compile_definitions(device_benchmarks PRIVATE HAVE_ROCM=1)
endif()

# Optimization flags for benchmarks - we want accurate measurements
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(device_benchmarks PRIVATE
        -O3
        -march=native
        -ffast-math
    )
endif()
