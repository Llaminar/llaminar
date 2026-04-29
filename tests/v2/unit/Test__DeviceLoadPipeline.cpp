/**
 * @file Test__DeviceLoadPipeline.cpp
 * @brief Integration tests for DeviceLoadPipeline: pipelined H2D + GPU repack
 *
 * Tests the complete pipeline: host blocks → pinned staging → H2D async →
 * GPU repack → final VNNI layout in WeightVRAMPool slots.
 * Verifies byte-for-byte payload parity and FP16 scale/min parity with
 * CPU reference packing.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>

#include "loaders/gpu_pipeline/DeviceLoadPipeline.h"
#include "loaders/gpu_pipeline/WeightVRAMPool.h"
#include "loaders/gpu_pipeline/PinnedRingBuffer.h"
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "loaders/gpu_pipeline/RepackFormat.h"
#include "backends/BackendManager.h"
#include "tensors/BlockStructures.h"

#ifdef HAVE_ROCM
#include "kernels/rocm/repack/VnniRepackKernels.h"
#include <hip/hip_runtime.h>
#endif

#ifdef HAVE_CUDA
#include "kernels/cuda/repack/CUDAVnniRepackKernels.h"
#endif

namespace llaminar2
{

// ============================================================================
// CPU reference helpers (copied from Phase 2 tests for self-containment)
// ============================================================================

namespace
{

inline float cpu_fp16_to_fp32(uint16_t h)
{
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0)
    {
        if (mant == 0)
        {
            uint32_t bits = sign;
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
        }
        while (!(mant & 0x400))
        {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FF;
    }
    else if (exp == 31)
    {
        uint32_t bits = sign | 0x7F800000u | (mant << 13);
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    uint32_t bits = sign | ((exp + 112) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t cpu_fp32_to_fp16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (exp <= 0)
    {
        if (exp < -10) return sign;
        mant = (mant | 0x800000) >> (1 - exp);
        if ((mant & 0x1FFF) > 0x1000 || ((mant & 0x1FFF) == 0x1000 && (mant & 0x2000)))
            mant += 0x2000;
        return sign | static_cast<uint16_t>(mant >> 13);
    }
    if (exp >= 31) return sign | 0x7C00;

    if ((mant & 0x1FFF) > 0x1000 || ((mant & 0x1FFF) == 0x1000 && (mant & 0x2000)))
    {
        mant += 0x2000;
        if (mant & 0x800000)
        {
            mant = 0;
            exp++;
        }
    }
    if (exp >= 31) return sign | 0x7C00;

    return sign | static_cast<uint16_t>(exp << 10) | static_cast<uint16_t>(mant >> 13);
}

inline void cpu_get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m)
{
    if (j < 4)
    {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    }
    else
    {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void cpu_pack_q4_0(const Q4_0Block* blocks, int N, int K,
                   std::vector<uint8_t>& payload, std::vector<uint16_t>& scales)
{
    const int blocks_per_row = K / 32;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 16);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            const auto& blk = blocks[n * blocks_per_row + b];
            const size_t linear = static_cast<size_t>(b) * N + n;
            std::memcpy(payload.data() + linear * 16, blk.qs, 16);
            scales[linear] = blk.d;
        }
    }
}

void cpu_pack_q4k(const Q4_KBlock* blocks, int N, int K,
                  std::vector<uint8_t>& payload, std::vector<uint16_t>& scales,
                  std::vector<uint16_t>& mins)
{
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 16);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);
    mins.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            const size_t linear = static_cast<size_t>(b) * N + n;
            const int sb_idx = b / 8;
            const int sub_idx = b % 8;
            const auto& blk = blocks[n * sb_per_row + sb_idx];

            const int group_idx = sub_idx / 2;
            const int is_high = sub_idx & 1;
            const uint8_t* src32 = blk.qs + group_idx * 32;

            uint8_t repacked[16];
            if (is_high)
            {
                for (int i = 0; i < 16; ++i)
                    repacked[i] = (src32[i] >> 4) | (src32[i + 16] & 0xF0);
            }
            else
            {
                for (int i = 0; i < 16; ++i)
                    repacked[i] = (src32[i] & 0xF) | ((src32[i + 16] & 0xF) << 4);
            }

            std::memcpy(payload.data() + linear * 16, repacked, 16);

            uint8_t sc, m_val;
            cpu_get_scale_min_k4(sub_idx, blk.scales, &sc, &m_val);
            const float d = cpu_fp16_to_fp32(blk.d);
            const float dmin = cpu_fp16_to_fp32(blk.dmin);
            scales[linear] = cpu_fp32_to_fp16(d * static_cast<float>(sc));
            mins[linear] = cpu_fp32_to_fp16(-dmin * static_cast<float>(m_val));
        }
    }
}

void fill_q4_0_blocks(Q4_0Block* blocks, int count)
{
    for (int i = 0; i < count; ++i)
    {
        blocks[i].d = 0x3C00; // 1.0 in FP16
        for (int j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) & 0xFF);
        }
    }
}

void fill_q4k_blocks(Q4_KBlock* blocks, int count)
{
    for (int i = 0; i < count; ++i)
    {
        blocks[i].d = 0x3800;    // 0.5 in FP16
        blocks[i].dmin = 0x3400; // 0.25 in FP16
        for (int j = 0; j < 12; ++j)
        {
            blocks[i].scales[j] = static_cast<uint8_t>(((i + j) * 7 + 3) & 0x3F);
        }
        for (int j = 0; j < 128; ++j)
        {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 128 + j) & 0xFF);
        }
    }
}

inline bool fp16_approx_equal(uint16_t a, uint16_t b)
{
    if ((a & 0x7FFF) == 0 && (b & 0x7FFF) == 0) return true;
    int diff = std::abs(static_cast<int>(a) - static_cast<int>(b));
    return diff <= 1;
}

} // anonymous namespace

// ============================================================================
// Test fixture
// ============================================================================

class Test__DeviceLoadPipeline : public ::testing::Test
{
  protected:
    IBackend* backend_ = nullptr;
    RepackKernels kernels_{};

    void SetUp() override
    {
#ifdef HAVE_ROCM
        backend_ = getROCmBackend();
        if (backend_)
        {
            kernels_.vnniRepack = launchVnniRepack;
        }
#endif
#ifdef HAVE_CUDA
        if (!backend_)
        {
            backend_ = getCUDABackend();
            if (backend_)
            {
                kernels_.vnniRepack = launchVnniRepackCUDA;
            }
        }
#endif
        if (!backend_)
        {
            GTEST_SKIP() << "No GPU backend available (need HAVE_ROCM or HAVE_CUDA)";
        }
        backend_->setDevice(0);
    }
};

// ============================================================================
// Helper: GPU download RAII wrapper
// ============================================================================

#ifdef HAVE_ROCM

template <typename T>
struct GpuBuffer
{
    T* ptr = nullptr;
    size_t count = 0;

    GpuBuffer() = default;

    explicit GpuBuffer(size_t n) : count(n)
    {
        if (n > 0)
        {
            hipError_t err = hipMalloc(&ptr, n * sizeof(T));
            EXPECT_EQ(err, hipSuccess) << "hipMalloc failed: " << hipGetErrorString(err);
        }
    }

    ~GpuBuffer()
    {
        if (ptr) (void)hipFree(ptr);
    }

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    void upload(const T* host_data)
    {
        (void)hipMemcpy(ptr, host_data, count * sizeof(T), hipMemcpyHostToDevice);
    }

    void download(T* host_data) const
    {
        (void)hipMemcpy(host_data, ptr, count * sizeof(T), hipMemcpyDeviceToHost);
    }

    void upload(const std::vector<T>& v) { upload(v.data()); }

    void download(std::vector<T>& v) const
    {
        v.resize(count);
        download(v.data());
    }
};

/// Download raw bytes from a device pointer into a host vector
inline void downloadBytes(const void* d_ptr, std::vector<uint8_t>& dst, size_t bytes)
{
    dst.resize(bytes);
    (void)hipMemcpy(dst.data(), d_ptr, bytes, hipMemcpyDeviceToHost);
}

inline void downloadU16(const void* d_ptr, std::vector<uint16_t>& dst, size_t count)
{
    dst.resize(count);
    (void)hipMemcpy(dst.data(), d_ptr, count * sizeof(uint16_t), hipMemcpyDeviceToHost);
}

#endif // HAVE_ROCM

// ============================================================================
// Helper: verify Q4_0 weight in pool slot vs CPU reference
// ============================================================================

#ifdef HAVE_ROCM

static void verifyQ4_0Slot(const WeightVRAMPool::WeightSlot& slot,
                           const Q4_0Block* host_blocks, int N, int K)
{
    const int blocks_per_row = K / 32;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    // CPU reference
    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    cpu_pack_q4_0(host_blocks, N, K, cpu_payload, cpu_scales);

    // Download GPU results
    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    downloadBytes(slot.d_native_vnni_payload, gpu_payload, total_output * 16);
    downloadU16(slot.d_native_vnni_scales, gpu_scales, total_output);

    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "Q4_0 payload mismatch";

    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    EXPECT_EQ(gpu_scales, cpu_scales) << "Q4_0 scales mismatch";
}

static void verifyQ4_KSlot(const WeightVRAMPool::WeightSlot& slot,
                           const Q4_KBlock* host_blocks, int N, int K)
{
    const int blocks_per_row = K / 32;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    // CPU reference
    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    std::vector<uint16_t> cpu_mins;
    cpu_pack_q4k(host_blocks, N, K, cpu_payload, cpu_scales, cpu_mins);

    // Download GPU results
    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    std::vector<uint16_t> gpu_mins;
    downloadBytes(slot.d_native_vnni_payload, gpu_payload, total_output * 16);
    downloadU16(slot.d_native_vnni_scales, gpu_scales, total_output);
    ASSERT_NE(slot.d_native_vnni_mins, nullptr);
    downloadU16(slot.d_native_vnni_mins, gpu_mins, total_output);

    // Payload: byte-for-byte
    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "Q4_K payload mismatch";

    // Scales/mins: 1-ULP FP16 tolerance
    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    ASSERT_EQ(gpu_mins.size(), cpu_mins.size());

    int scale_mismatches = 0;
    for (size_t i = 0; i < cpu_scales.size(); ++i)
    {
        if (!fp16_approx_equal(gpu_scales[i], cpu_scales[i]))
        {
            scale_mismatches++;
            if (scale_mismatches <= 3)
            {
                ADD_FAILURE() << "Q4_K scale mismatch at index " << i << ": GPU=0x" << std::hex
                              << gpu_scales[i] << " CPU=0x" << cpu_scales[i] << std::dec;
            }
        }
    }
    EXPECT_EQ(scale_mismatches, 0);

    int min_mismatches = 0;
    for (size_t i = 0; i < cpu_mins.size(); ++i)
    {
        if (!fp16_approx_equal(gpu_mins[i], cpu_mins[i]))
        {
            min_mismatches++;
            if (min_mismatches <= 3)
            {
                ADD_FAILURE() << "Q4_K min mismatch at index " << i << ": GPU=0x" << std::hex
                              << gpu_mins[i] << " CPU=0x" << cpu_mins[i] << std::dec;
            }
        }
    }
    EXPECT_EQ(min_mismatches, 0);
}

#endif // HAVE_ROCM

// ============================================================================
// Test 1: Q4_0 single weight through pipeline
// ============================================================================

TEST_F(Test__DeviceLoadPipeline, Q4_0_SingleWeight)
{
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 128;
    const int blocks_per_row = K / 32;
    const int total_blocks = N * blocks_per_row;
    const size_t raw_bytes = total_blocks * sizeof(Q4_0Block);
    const int num_streams = 3;

    // Create synthetic blocks
    std::vector<Q4_0Block> host_blocks(total_blocks);
    fill_q4_0_blocks(host_blocks.data(), total_blocks);

    // Setup pool
    WeightVRAMPool pool;
    pool.planWeight("test_q4_0", N, K, 16, false, false, raw_bytes);
    ASSERT_TRUE(pool.allocate(backend_, 0, num_streams));

    // Setup pinned ring
    PinnedRingBuffer pinned(raw_bytes, num_streams);
    ASSERT_TRUE(pinned.allocate(backend_, 0));

    // Create pipeline
    DeviceLoadPipeline pipeline(*backend_, 0, pool, pinned, kernels_, num_streams);
    ASSERT_TRUE(pipeline.initialize());

    // Create job
    WeightJob job;
    job.name = "test_q4_0";
    job.host_raw_data = host_blocks.data();
    job.raw_bytes = raw_bytes;
    job.format = RepackFormat::Q4_0;
    job.N = N;
    job.K = K;
    job.is_asymmetric = false;

    ASSERT_TRUE(pipeline.processJobs({job}));
    EXPECT_EQ(pipeline.numProcessed(), 1u);

    // Verify
    auto slot = pool.getSlot("test_q4_0");
    ASSERT_TRUE(slot.has_value());
    verifyQ4_0Slot(*slot, host_blocks.data(), N, K);

    pinned.release();
    pool.release();
#endif
}

// ============================================================================
// Test 2: Q4_K single weight through pipeline
// ============================================================================

TEST_F(Test__DeviceLoadPipeline, Q4_K_SingleWeight)
{
#ifdef HAVE_ROCM
    const int N = 32;
    const int K = 256;
    const int sb_per_row = (K + 255) / 256;
    const int total_superblocks = N * sb_per_row;
    const size_t raw_bytes = total_superblocks * sizeof(Q4_KBlock);
    const int num_streams = 3;

    std::vector<Q4_KBlock> host_blocks(total_superblocks);
    fill_q4k_blocks(host_blocks.data(), total_superblocks);

    WeightVRAMPool pool;
    pool.planWeight("test_q4k", N, K, 16, true, false, raw_bytes);
    ASSERT_TRUE(pool.allocate(backend_, 0, num_streams));

    PinnedRingBuffer pinned(raw_bytes, num_streams);
    ASSERT_TRUE(pinned.allocate(backend_, 0));

    DeviceLoadPipeline pipeline(*backend_, 0, pool, pinned, kernels_, num_streams);
    ASSERT_TRUE(pipeline.initialize());

    WeightJob job;
    job.name = "test_q4k";
    job.host_raw_data = host_blocks.data();
    job.raw_bytes = raw_bytes;
    job.format = RepackFormat::Q4_K;
    job.N = N;
    job.K = K;
    job.is_asymmetric = true;

    ASSERT_TRUE(pipeline.processJobs({job}));
    EXPECT_EQ(pipeline.numProcessed(), 1u);

    auto slot = pool.getSlot("test_q4k");
    ASSERT_TRUE(slot.has_value());
    verifyQ4_KSlot(*slot, host_blocks.data(), N, K);

    pinned.release();
    pool.release();
#endif
}

// ============================================================================
// Test 3: Multiple weights (mixed Q4_0 + Q4_K)
// ============================================================================

TEST_F(Test__DeviceLoadPipeline, MultipleWeights)
{
#ifdef HAVE_ROCM
    const int num_streams = 3;

    // Define 6 weights: 3 x Q4_0 + 3 x Q4_K with different dimensions
    struct WeightSpec
    {
        std::string name;
        int N, K;
        RepackFormat format;
        bool is_asymmetric;
    };

    std::vector<WeightSpec> specs = {
        {"q4_0_a", 32, 64, RepackFormat::Q4_0, false},
        {"q4_0_b", 64, 128, RepackFormat::Q4_0, false},
        {"q4_0_c", 128, 256, RepackFormat::Q4_0, false},
        {"q4k_a", 32, 256, RepackFormat::Q4_K, true},
        {"q4k_b", 64, 512, RepackFormat::Q4_K, true},
        {"q4k_c", 128, 256, RepackFormat::Q4_K, true},
    };

    // Create host blocks and compute raw sizes
    std::vector<std::vector<Q4_0Block>> q4_0_blocks_vec(3);
    std::vector<std::vector<Q4_KBlock>> q4k_blocks_vec(3);
    size_t max_raw = 0;

    WeightVRAMPool pool;

    for (size_t i = 0; i < specs.size(); ++i)
    {
        const auto& s = specs[i];
        size_t raw_bytes;

        if (s.format == RepackFormat::Q4_0)
        {
            int total = s.N * (s.K / 32);
            q4_0_blocks_vec[i].resize(total);
            fill_q4_0_blocks(q4_0_blocks_vec[i].data(), total);
            raw_bytes = total * sizeof(Q4_0Block);
        }
        else
        {
            int sb_per_row = (s.K + 255) / 256;
            int total = s.N * sb_per_row;
            q4k_blocks_vec[i - 3].resize(total);
            fill_q4k_blocks(q4k_blocks_vec[i - 3].data(), total);
            raw_bytes = total * sizeof(Q4_KBlock);
        }

        max_raw = std::max(max_raw, raw_bytes);
        pool.planWeight(s.name, s.N, s.K, 16, s.is_asymmetric, false, raw_bytes);
    }

    ASSERT_TRUE(pool.allocate(backend_, 0, num_streams));

    PinnedRingBuffer pinned(max_raw, num_streams);
    ASSERT_TRUE(pinned.allocate(backend_, 0));

    DeviceLoadPipeline pipeline(*backend_, 0, pool, pinned, kernels_, num_streams);
    ASSERT_TRUE(pipeline.initialize());

    // Build jobs
    std::vector<WeightJob> jobs;
    for (size_t i = 0; i < specs.size(); ++i)
    {
        const auto& s = specs[i];
        WeightJob job;
        job.name = s.name;
        job.format = s.format;
        job.N = s.N;
        job.K = s.K;
        job.is_asymmetric = s.is_asymmetric;

        if (s.format == RepackFormat::Q4_0)
        {
            job.host_raw_data = q4_0_blocks_vec[i].data();
            job.raw_bytes = q4_0_blocks_vec[i].size() * sizeof(Q4_0Block);
        }
        else
        {
            job.host_raw_data = q4k_blocks_vec[i - 3].data();
            job.raw_bytes = q4k_blocks_vec[i - 3].size() * sizeof(Q4_KBlock);
        }

        jobs.push_back(job);
    }

    ASSERT_TRUE(pipeline.processJobs(jobs));
    EXPECT_EQ(pipeline.numProcessed(), 6u);

    // Verify all 6
    for (size_t i = 0; i < specs.size(); ++i)
    {
        const auto& s = specs[i];
        auto slot = pool.getSlot(s.name);
        ASSERT_TRUE(slot.has_value()) << "Missing slot for " << s.name;

        if (s.format == RepackFormat::Q4_0)
        {
            verifyQ4_0Slot(*slot, q4_0_blocks_vec[i].data(), s.N, s.K);
        }
        else
        {
            verifyQ4_KSlot(*slot, q4k_blocks_vec[i - 3].data(), s.N, s.K);
        }
    }

    pinned.release();
    pool.release();
#endif
}

// ============================================================================
// Test 4: More weights than streams (exercises staging slot reuse)
// ============================================================================

TEST_F(Test__DeviceLoadPipeline, MultipleWeightsExercisesAllStreams)
{
#ifdef HAVE_ROCM
    const int num_streams = 3;
    const int num_weights = 9; // 3x more than streams
    const int N = 64;
    const int K = 128;
    const int blocks_per_row = K / 32;
    const int total_blocks = N * blocks_per_row;
    const size_t raw_bytes = total_blocks * sizeof(Q4_0Block);

    // Create unique block data for each weight
    std::vector<std::vector<Q4_0Block>> all_blocks(num_weights);
    WeightVRAMPool pool;

    for (int w = 0; w < num_weights; ++w)
    {
        all_blocks[w].resize(total_blocks);
        // Use different seed per weight for distinct data
        for (int i = 0; i < total_blocks; ++i)
        {
            all_blocks[w][i].d = static_cast<uint16_t>(0x3C00 + w);
            for (int j = 0; j < 16; ++j)
            {
                all_blocks[w][i].qs[j] =
                    static_cast<uint8_t>((w * 1000 + i * 16 + j) & 0xFF);
            }
        }

        std::string name = "w_" + std::to_string(w);
        pool.planWeight(name, N, K, 16, false, false, raw_bytes);
    }

    ASSERT_TRUE(pool.allocate(backend_, 0, num_streams));

    PinnedRingBuffer pinned(raw_bytes, num_streams);
    ASSERT_TRUE(pinned.allocate(backend_, 0));

    DeviceLoadPipeline pipeline(*backend_, 0, pool, pinned, kernels_, num_streams);
    ASSERT_TRUE(pipeline.initialize());

    std::vector<WeightJob> jobs;
    for (int w = 0; w < num_weights; ++w)
    {
        WeightJob job;
        job.name = "w_" + std::to_string(w);
        job.host_raw_data = all_blocks[w].data();
        job.raw_bytes = raw_bytes;
        job.format = RepackFormat::Q4_0;
        job.N = N;
        job.K = K;
        job.is_asymmetric = false;
        jobs.push_back(job);
    }

    ASSERT_TRUE(pipeline.processJobs(jobs));
    EXPECT_EQ(pipeline.numProcessed(), static_cast<size_t>(num_weights));

    // Verify all 9 weights
    for (int w = 0; w < num_weights; ++w)
    {
        std::string name = "w_" + std::to_string(w);
        auto slot = pool.getSlot(name);
        ASSERT_TRUE(slot.has_value()) << "Missing slot for " << name;
        verifyQ4_0Slot(*slot, all_blocks[w].data(), N, K);
    }

    pinned.release();
    pool.release();
#endif
}

// ============================================================================
// Test 5: LoadOrchestrator end-to-end
// ============================================================================

TEST_F(Test__DeviceLoadPipeline, LoadOrchestratorEndToEnd)
{
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 128;
    const int blocks_per_row = K / 32;
    const int total_blocks = N * blocks_per_row;
    const size_t raw_bytes = total_blocks * sizeof(Q4_0Block);

    const int N2 = 32;
    const int K2 = 256;
    const int sb_per_row = (K2 + 255) / 256;
    const int total_sb = N2 * sb_per_row;
    const size_t raw_bytes2 = total_sb * sizeof(Q4_KBlock);

    const size_t max_raw = std::max(raw_bytes, raw_bytes2);

    // Create host data
    std::vector<Q4_0Block> q4_0_data(total_blocks);
    fill_q4_0_blocks(q4_0_data.data(), total_blocks);

    std::vector<Q4_KBlock> q4k_data(total_sb);
    fill_q4k_blocks(q4k_data.data(), total_sb);

    // Setup orchestrator
    LoadOrchestrator orch(backend_);
    orch.addDevice(0);

    orch.planWeight(0, "w_q4_0", N, K, 16, false, false, raw_bytes);
    orch.planWeight(0, "w_q4k", N2, K2, 16, true, false, raw_bytes2);

    ASSERT_NO_THROW(orch.allocate(max_raw, 3));

    // Add weight jobs
    WeightJob job1;
    job1.name = "w_q4_0";
    job1.host_raw_data = q4_0_data.data();
    job1.raw_bytes = raw_bytes;
    job1.format = RepackFormat::Q4_0;
    job1.N = N;
    job1.K = K;
    job1.is_asymmetric = false;
    orch.addWeightJob(0, job1);

    WeightJob job2;
    job2.name = "w_q4k";
    job2.host_raw_data = q4k_data.data();
    job2.raw_bytes = raw_bytes2;
    job2.format = RepackFormat::Q4_K;
    job2.N = N2;
    job2.K = K2;
    job2.is_asymmetric = true;
    orch.addWeightJob(0, job2);

    // Load
    ASSERT_NO_THROW(orch.load());
    orch.finalize();

    // Verify via pool
    auto* pool = orch.getPool(0);
    ASSERT_NE(pool, nullptr);

    auto slot1 = pool->getSlot("w_q4_0");
    ASSERT_TRUE(slot1.has_value());
    verifyQ4_0Slot(*slot1, q4_0_data.data(), N, K);

    auto slot2 = pool->getSlot("w_q4k");
    ASSERT_TRUE(slot2.has_value());
    verifyQ4_KSlot(*slot2, q4k_data.data(), N2, K2);

    orch.release();
#endif
}

// ============================================================================
// Test: Empty jobs list succeeds
// ============================================================================

TEST_F(Test__DeviceLoadPipeline, EmptyJobsSucceeds)
{
#ifdef HAVE_ROCM
    WeightVRAMPool pool;
    pool.planWeight("dummy", 64, 128, 16, false, false, 1024);
    ASSERT_TRUE(pool.allocate(backend_, 0, 3));

    PinnedRingBuffer pinned(1024, 3);
    ASSERT_TRUE(pinned.allocate(backend_, 0));

    DeviceLoadPipeline pipeline(*backend_, 0, pool, pinned, kernels_, 3);
    ASSERT_TRUE(pipeline.initialize());

    EXPECT_TRUE(pipeline.processJobs({}));
    EXPECT_EQ(pipeline.numProcessed(), 0u);

    pinned.release();
    pool.release();
#endif
}

} // namespace llaminar2
