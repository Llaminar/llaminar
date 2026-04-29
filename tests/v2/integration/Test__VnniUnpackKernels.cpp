/**
 * @file Test__VnniUnpackKernels.cpp
 * @brief Integration tests for GPU VNNI reverse repack and WeightTranslator
 *
 * Tests two things:
 *
 * 1. **Round-trip repack**: For each reversible format (Q4_0, IQ4_NL, Q4_1,
 *    Q5_0, Q5_1, Q8_0), creates synthetic GGUF blocks, runs forward repack
 *    on GPU (raw → separated), then reverse repack (separated → raw), and
 *    compares with the original blocks byte-for-byte.
 *
 * 2. **WeightTranslator pack/upload**: Exercises the GPU→host→GPU transfer
 *    path via GpuPackedWeightsFormat + WeightTranslator.
 *
 * Parameterized over CUDA and ROCm backends.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cstdint>
#include <vector>

#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "loaders/gpu_pipeline/GpuPackedWeightsFormat.h"
#include "loaders/gpu_pipeline/RepackFormat.h"
#include "loaders/gpu_pipeline/WeightTranslator.h"
#include "tensors/BlockStructures.h"

#ifdef HAVE_CUDA
#include "kernels/cuda/repack/CUDAVnniRepackKernels.h"
#include "kernels/cuda/repack/CUDAVnniUnpackKernels.h"
#endif
#ifdef HAVE_ROCM
#include "kernels/rocm/repack/VnniRepackKernels.h"
#include "kernels/rocm/repack/VnniUnpackKernels.h"
#endif

using namespace llaminar2;

namespace {

// ============================================================================
// Deterministic block fillers (same patterns as Test__VnniRepackKernels.cpp)
// ============================================================================

void fill_q4_0_blocks(Q4_0Block* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3C00; // 1.0 in FP16
        for (int j = 0; j < 16; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) & 0xFF);
    }
}

void fill_q4_1_blocks(Q4_1Block* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3C00; // 1.0 in FP16
        blocks[i].m = 0x3800; // 0.5 in FP16
        for (int j = 0; j < 16; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) & 0xFF);
    }
}

void fill_q5_0_blocks(Q5_0Block* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3C00; // 1.0 in FP16
        for (int j = 0; j < 16; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) & 0xFF);
        for (int j = 0; j < 4; ++j)
            blocks[i].qh[j] = static_cast<uint8_t>((i + j * 37) & 0xFF);
    }
}

void fill_q5_1_blocks(Q5_1Block* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3C00; // 1.0 in FP16
        blocks[i].m = 0x3400; // 0.25 in FP16
        for (int j = 0; j < 16; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) & 0xFF);
        for (int j = 0; j < 4; ++j)
            blocks[i].qh[j] = static_cast<uint8_t>((i + j * 37) & 0xFF);
    }
}

void fill_q8_0_blocks(Q8_0Block* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3C00; // 1.0 in FP16
        for (int j = 0; j < 32; ++j)
            blocks[i].qs[j] = static_cast<int8_t>(((i * 32 + j) % 256) - 128);
    }
}

// ============================================================================
// Test fixture — parameterized over backend name
// ============================================================================

class VnniUnpackTest : public ::testing::TestWithParam<std::string> {
protected:
    IBackend* backend_ = nullptr;
    int device_id_ = 0;
    DeviceType device_type_ = DeviceType::CPU;

    void SetUp() override {
        const auto& backend_name = GetParam();

        if (backend_name == "CUDA") {
#ifdef HAVE_CUDA
            backend_ = getCUDABackend();
            if (!backend_) GTEST_SKIP() << "CUDA backend not available";
            device_type_ = DeviceType::CUDA;
#else
            GTEST_SKIP() << "HAVE_CUDA not defined";
#endif
        } else if (backend_name == "ROCm") {
#ifdef HAVE_ROCM
            backend_ = getROCmBackend();
            if (!backend_) GTEST_SKIP() << "ROCm backend not available";
            device_type_ = DeviceType::ROCm;
#else
            GTEST_SKIP() << "HAVE_ROCM not defined";
#endif
        } else {
            FAIL() << "Unknown backend: " << backend_name;
        }
    }

    // Helper: allocate GPU buffer, auto-free on scope exit
    struct GpuMem {
        IBackend* be;
        int dev;
        void* ptr;
        size_t bytes;
        GpuMem(IBackend* b, int d, size_t n)
            : be(b), dev(d), ptr(nullptr), bytes(n) {
            if (n > 0) ptr = be->allocate(n, dev);
        }
        ~GpuMem() { if (ptr) be->free(ptr, dev); }
        GpuMem(const GpuMem&) = delete;
        GpuMem& operator=(const GpuMem&) = delete;
        uint8_t*  u8()  { return static_cast<uint8_t*>(ptr); }
        uint16_t* u16() { return static_cast<uint16_t*>(ptr); }
    };

    // ========================================================================
    // Backend-agnostic forward repack dispatch
    // ========================================================================
    bool forwardRepack(RepackFormat format, const void* d_raw,
                       uint8_t* d_payload, uint16_t* d_scales, uint16_t* d_mins,
                       int N, int K) {
        return WeightTranslator::forwardRepackOnDevice(
            format, d_raw, d_payload, d_scales, d_mins, nullptr,
            N, K, device_type_, nullptr);
    }

    // ========================================================================
    // Backend-agnostic reverse repack dispatch
    // ========================================================================
    bool reverseRepack(RepackFormat format, const uint8_t* d_payload,
                       const uint16_t* d_scales, const uint16_t* d_mins,
                       void* d_raw, int N, int K) {
        return WeightTranslator::reverseRepackOnDevice(
            format, d_payload, d_scales, d_mins, d_raw, N, K,
            device_type_, nullptr);
    }

    // ========================================================================
    // Round-trip test template
    // ========================================================================
    template<typename BlockT>
    void roundTripTest(RepackFormat format,
                       void (*filler)(BlockT*, int),
                       int payload_bytes,
                       bool is_asymmetric,
                       int N, int K) {
        const int blocks_per_row = K / 32;
        const int total_blocks = N * blocks_per_row;
        const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

        // 1. Create and fill host blocks
        std::vector<BlockT> host_blocks(total_blocks);
        filler(host_blocks.data(), total_blocks);

        // 2. Allocate GPU buffers
        GpuMem d_raw_in(backend_, device_id_, total_blocks * sizeof(BlockT));
        GpuMem d_payload(backend_, device_id_, total_output * payload_bytes);
        GpuMem d_scales(backend_, device_id_, total_output * sizeof(uint16_t));
        GpuMem d_mins(backend_, device_id_,
                      is_asymmetric ? total_output * sizeof(uint16_t) : 0);
        GpuMem d_raw_out(backend_, device_id_, total_blocks * sizeof(BlockT));

        ASSERT_NE(d_raw_in.ptr, nullptr);
        ASSERT_NE(d_payload.ptr, nullptr);
        ASSERT_NE(d_scales.ptr, nullptr);
        ASSERT_NE(d_raw_out.ptr, nullptr);

        // 3. Upload original blocks
        ASSERT_TRUE(backend_->hostToDevice(d_raw_in.ptr, host_blocks.data(),
                                           total_blocks * sizeof(BlockT),
                                           device_id_));

        // 4. Forward repack: raw → separated
        ASSERT_TRUE(forwardRepack(format, d_raw_in.ptr,
                                  d_payload.u8(), d_scales.u16(),
                                  is_asymmetric ? d_mins.u16() : nullptr,
                                  N, K));
        backend_->streamSynchronize(device_id_);

        // 5. Reverse repack: separated → raw
        ASSERT_TRUE(reverseRepack(format, d_payload.u8(), d_scales.u16(),
                                  is_asymmetric ? d_mins.u16() : nullptr,
                                  d_raw_out.ptr, N, K));
        backend_->streamSynchronize(device_id_);

        // 6. Download recovered blocks
        std::vector<BlockT> recovered(total_blocks);
        ASSERT_TRUE(backend_->deviceToHost(recovered.data(), d_raw_out.ptr,
                                           total_blocks * sizeof(BlockT),
                                           device_id_));

        // 7. Byte-for-byte comparison
        for (int i = 0; i < total_blocks; ++i) {
            ASSERT_EQ(std::memcmp(&host_blocks[i], &recovered[i], sizeof(BlockT)), 0)
                << "Block " << i << " mismatch after round-trip for format "
                << static_cast<int>(format);
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    GPU,
    VnniUnpackTest,
    ::testing::Values("CUDA", "ROCm"),
    [](const ::testing::TestParamInfo<std::string>& info) {
        return info.param;
    });

// ============================================================================
// Round-trip tests: forward repack → reverse repack → compare
// ============================================================================

TEST_P(VnniUnpackTest, RoundTrip_Q4_0) {
    roundTripTest<Q4_0Block>(RepackFormat::Q4_0, fill_q4_0_blocks,
                             /*payload_bytes=*/16, /*asymmetric=*/false,
                             /*N=*/64, /*K=*/128);
}

TEST_P(VnniUnpackTest, RoundTrip_IQ4_NL) {
    // IQ4_NL shares the same block layout as Q4_0 (18 bytes, 16B payload)
    // but uses a different codebook for dequantization.  The repack/unpack
    // kernels operate on raw bytes, so we re-use Q4_0Block.
    roundTripTest<Q4_0Block>(RepackFormat::IQ4_NL, fill_q4_0_blocks,
                             /*payload_bytes=*/16, /*asymmetric=*/false,
                             /*N=*/64, /*K=*/128);
}

TEST_P(VnniUnpackTest, RoundTrip_Q4_1) {
    roundTripTest<Q4_1Block>(RepackFormat::Q4_1, fill_q4_1_blocks,
                             /*payload_bytes=*/16, /*asymmetric=*/true,
                             /*N=*/64, /*K=*/128);
}

TEST_P(VnniUnpackTest, RoundTrip_Q5_0) {
    roundTripTest<Q5_0Block>(RepackFormat::Q5_0, fill_q5_0_blocks,
                             /*payload_bytes=*/20, /*asymmetric=*/false,
                             /*N=*/64, /*K=*/128);
}

TEST_P(VnniUnpackTest, RoundTrip_Q5_1) {
    roundTripTest<Q5_1Block>(RepackFormat::Q5_1, fill_q5_1_blocks,
                             /*payload_bytes=*/20, /*asymmetric=*/true,
                             /*N=*/64, /*K=*/128);
}

TEST_P(VnniUnpackTest, RoundTrip_Q8_0) {
    roundTripTest<Q8_0Block>(RepackFormat::Q8_0, fill_q8_0_blocks,
                             /*payload_bytes=*/32, /*asymmetric=*/false,
                             /*N=*/64, /*K=*/128);
}

// ============================================================================
// Larger matrix sizes (stress test alignment and boundary conditions)
// ============================================================================

TEST_P(VnniUnpackTest, RoundTrip_Q4_0_LargeMatrix) {
    roundTripTest<Q4_0Block>(RepackFormat::Q4_0, fill_q4_0_blocks,
                             /*payload_bytes=*/16, /*asymmetric=*/false,
                             /*N=*/512, /*K=*/4096);
}

TEST_P(VnniUnpackTest, RoundTrip_Q8_0_LargeMatrix) {
    roundTripTest<Q8_0Block>(RepackFormat::Q8_0, fill_q8_0_blocks,
                             /*payload_bytes=*/32, /*asymmetric=*/false,
                             /*N=*/256, /*K=*/2048);
}

TEST_P(VnniUnpackTest, RoundTrip_Q5_1_LargeMatrix) {
    roundTripTest<Q5_1Block>(RepackFormat::Q5_1, fill_q5_1_blocks,
                             /*payload_bytes=*/20, /*asymmetric=*/true,
                             /*N=*/256, /*K=*/2048);
}

// ============================================================================
// Non-reversible format rejection
// ============================================================================

TEST_P(VnniUnpackTest, RejectsNonReversibleFormat) {
    EXPECT_FALSE(isReversibleFormat(RepackFormat::Q4_K));
    EXPECT_FALSE(isReversibleFormat(RepackFormat::Q5_K));
    EXPECT_FALSE(isReversibleFormat(RepackFormat::Q6_K));
    EXPECT_FALSE(isReversibleFormat(RepackFormat::Q3_K));
    EXPECT_FALSE(isReversibleFormat(RepackFormat::Q2_K));
    EXPECT_FALSE(isReversibleFormat(RepackFormat::IQ4_XS));
    EXPECT_FALSE(isReversibleFormat(RepackFormat::IQ3_S));
    EXPECT_FALSE(isReversibleFormat(RepackFormat::IQ2_XXS));

    EXPECT_TRUE(isReversibleFormat(RepackFormat::Q4_0));
    EXPECT_TRUE(isReversibleFormat(RepackFormat::IQ4_NL));
    EXPECT_TRUE(isReversibleFormat(RepackFormat::Q4_1));
    EXPECT_TRUE(isReversibleFormat(RepackFormat::Q5_0));
    EXPECT_TRUE(isReversibleFormat(RepackFormat::Q5_1));
    EXPECT_TRUE(isReversibleFormat(RepackFormat::Q8_0));
}

TEST_P(VnniUnpackTest, ReverseRepackReturnsFailOnSuperblock) {
    // reverseRepackOnDevice should return false for non-reversible formats
    EXPECT_FALSE(WeightTranslator::reverseRepackOnDevice(
        RepackFormat::Q4_K, nullptr, nullptr, nullptr, nullptr,
        64, 128, device_type_, nullptr));
}

// ============================================================================
// rawBlockSizeBytes / rawBlockBufferSize helpers
// ============================================================================

TEST_P(VnniUnpackTest, RawBlockSizeBytes) {
    EXPECT_EQ(rawBlockSizeBytes(RepackFormat::Q4_0),   18u);
    EXPECT_EQ(rawBlockSizeBytes(RepackFormat::IQ4_NL), 18u);
    EXPECT_EQ(rawBlockSizeBytes(RepackFormat::Q4_1),   20u);
    EXPECT_EQ(rawBlockSizeBytes(RepackFormat::Q5_0),   22u);
    EXPECT_EQ(rawBlockSizeBytes(RepackFormat::Q5_1),   24u);
    EXPECT_EQ(rawBlockSizeBytes(RepackFormat::Q8_0),   34u);
    EXPECT_EQ(rawBlockSizeBytes(RepackFormat::Q4_K),   0u);
    EXPECT_EQ(rawBlockSizeBytes(RepackFormat::Q6_K),   0u);
}

TEST_P(VnniUnpackTest, RawBlockBufferSize) {
    // Q4_0: N=64, K=128 → 64 * (128/32) * 18 = 64 * 4 * 18 = 4608
    EXPECT_EQ(rawBlockBufferSize(RepackFormat::Q4_0, 64, 128), 4608u);
    // Q8_0: N=64, K=128 → 64 * 4 * 34 = 8704
    EXPECT_EQ(rawBlockBufferSize(RepackFormat::Q8_0, 64, 128), 8704u);
    // Non-reversible → 0
    EXPECT_EQ(rawBlockBufferSize(RepackFormat::Q4_K, 64, 128), 0u);
}

// ============================================================================
// WeightTranslator: pack → parse → upload round-trip
// ============================================================================

TEST_P(VnniUnpackTest, PackUploadRoundTrip_Q4_0) {
    const int N = 64;
    const int K = 128;
    const int blocks_per_row = K / 32;
    const int total_blocks = N * blocks_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;
    const int payload_bytes = 16;

    // 1. Create synthetic blocks and forward-repack on GPU
    std::vector<Q4_0Block> host_blocks(total_blocks);
    fill_q4_0_blocks(host_blocks.data(), total_blocks);

    GpuMem d_raw(backend_, device_id_, total_blocks * sizeof(Q4_0Block));
    GpuMem d_payload(backend_, device_id_, total_output * payload_bytes);
    GpuMem d_scales(backend_, device_id_, total_output * sizeof(uint16_t));

    ASSERT_TRUE(backend_->hostToDevice(d_raw.ptr, host_blocks.data(),
                                       total_blocks * sizeof(Q4_0Block),
                                       device_id_));
    ASSERT_TRUE(forwardRepack(RepackFormat::Q4_0, d_raw.ptr,
                              d_payload.u8(), d_scales.u16(), nullptr,
                              N, K));
    backend_->streamSynchronize(device_id_);

    // 2. Pack GPU → host buffer via WeightTranslator
    auto header = buildGpuPackedHeader(
        RepackFormat::Q4_0, /*codebook_id=*/0, payload_bytes,
        /*is_asymmetric=*/false, /*is_superblock=*/false, /*has_emins=*/false,
        N, K);

    ASSERT_TRUE(validateGpuPackedHeader(header, gpuPackedTotalSize(header)));

    auto packed_buf = WeightTranslator::packGpuWeightsForTransfer(
        *backend_, device_id_,
        d_payload.u8(), d_scales.u16(), nullptr, nullptr,
        header);

    ASSERT_EQ(packed_buf.size(), gpuPackedTotalSize(header));

    // 3. Parse the buffer
    GpuPackedWeightsView view;
    ASSERT_TRUE(parseGpuPackedBuffer(packed_buf.data(), packed_buf.size(), view))
        << "parseGpuPackedBuffer failed";
    EXPECT_EQ(view.header.N, static_cast<uint32_t>(N));
    EXPECT_EQ(view.header.K, static_cast<uint32_t>(K));
    EXPECT_EQ(view.header.format, static_cast<uint8_t>(RepackFormat::Q4_0));
    EXPECT_NE(view.payload, nullptr);
    EXPECT_NE(view.scales, nullptr);
    EXPECT_EQ(view.mins, nullptr);

    // 4. Upload back to GPU into fresh buffers
    GpuMem d_payload2(backend_, device_id_, total_output * payload_bytes);
    GpuMem d_scales2(backend_, device_id_, total_output * sizeof(uint16_t));

    ASSERT_TRUE(WeightTranslator::uploadGpuPackedWeights(
        *backend_, device_id_, view,
        d_payload2.u8(), d_scales2.u16(), nullptr, nullptr));
    backend_->streamSynchronize(device_id_);

    // 5. Download both sets and compare byte-for-byte
    std::vector<uint8_t> orig_payload(total_output * payload_bytes);
    std::vector<uint8_t> recv_payload(total_output * payload_bytes);
    std::vector<uint16_t> orig_scales(total_output);
    std::vector<uint16_t> recv_scales(total_output);

    ASSERT_TRUE(backend_->deviceToHost(orig_payload.data(), d_payload.ptr,
                                       orig_payload.size(), device_id_));
    ASSERT_TRUE(backend_->deviceToHost(recv_payload.data(), d_payload2.ptr,
                                       recv_payload.size(), device_id_));
    ASSERT_TRUE(backend_->deviceToHost(orig_scales.data(), d_scales.ptr,
                                       orig_scales.size() * sizeof(uint16_t),
                                       device_id_));
    ASSERT_TRUE(backend_->deviceToHost(recv_scales.data(), d_scales2.ptr,
                                       recv_scales.size() * sizeof(uint16_t),
                                       device_id_));

    EXPECT_EQ(orig_payload, recv_payload) << "Payload mismatch after pack/upload";
    EXPECT_EQ(orig_scales, recv_scales) << "Scales mismatch after pack/upload";
}

TEST_P(VnniUnpackTest, PackUploadRoundTrip_Q4_1_Asymmetric) {
    const int N = 64;
    const int K = 128;
    const int blocks_per_row = K / 32;
    const int total_blocks = N * blocks_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;
    const int payload_bytes = 16;

    // 1. Create and forward-repack
    std::vector<Q4_1Block> host_blocks(total_blocks);
    fill_q4_1_blocks(host_blocks.data(), total_blocks);

    GpuMem d_raw(backend_, device_id_, total_blocks * sizeof(Q4_1Block));
    GpuMem d_payload(backend_, device_id_, total_output * payload_bytes);
    GpuMem d_scales(backend_, device_id_, total_output * sizeof(uint16_t));
    GpuMem d_mins(backend_, device_id_, total_output * sizeof(uint16_t));

    ASSERT_TRUE(backend_->hostToDevice(d_raw.ptr, host_blocks.data(),
                                       total_blocks * sizeof(Q4_1Block),
                                       device_id_));
    ASSERT_TRUE(forwardRepack(RepackFormat::Q4_1, d_raw.ptr,
                              d_payload.u8(), d_scales.u16(), d_mins.u16(),
                              N, K));
    backend_->streamSynchronize(device_id_);

    // 2. Pack → parse → upload
    auto header = buildGpuPackedHeader(
        RepackFormat::Q4_1, /*codebook_id=*/5, payload_bytes,
        /*is_asymmetric=*/true, /*is_superblock=*/false, /*has_emins=*/false,
        N, K);
    ASSERT_TRUE(validateGpuPackedHeader(header, gpuPackedTotalSize(header)));

    auto packed_buf = WeightTranslator::packGpuWeightsForTransfer(
        *backend_, device_id_,
        d_payload.u8(), d_scales.u16(), d_mins.u16(), nullptr,
        header);
    ASSERT_EQ(packed_buf.size(), gpuPackedTotalSize(header));

    GpuPackedWeightsView view;
    ASSERT_TRUE(parseGpuPackedBuffer(packed_buf.data(), packed_buf.size(), view));
    EXPECT_NE(view.mins, nullptr) << "Asymmetric format should have mins";

    GpuMem d_payload2(backend_, device_id_, total_output * payload_bytes);
    GpuMem d_scales2(backend_, device_id_, total_output * sizeof(uint16_t));
    GpuMem d_mins2(backend_, device_id_, total_output * sizeof(uint16_t));

    ASSERT_TRUE(WeightTranslator::uploadGpuPackedWeights(
        *backend_, device_id_, view,
        d_payload2.u8(), d_scales2.u16(), d_mins2.u16(), nullptr));
    backend_->streamSynchronize(device_id_);

    // 3. Compare
    std::vector<uint16_t> orig_mins(total_output), recv_mins(total_output);
    ASSERT_TRUE(backend_->deviceToHost(orig_mins.data(), d_mins.ptr,
                                       orig_mins.size() * sizeof(uint16_t),
                                       device_id_));
    ASSERT_TRUE(backend_->deviceToHost(recv_mins.data(), d_mins2.ptr,
                                       recv_mins.size() * sizeof(uint16_t),
                                       device_id_));
    EXPECT_EQ(orig_mins, recv_mins) << "Mins mismatch after pack/upload";
}

// ============================================================================
// GpuPackedWeightsHeader validation
// ============================================================================

TEST_P(VnniUnpackTest, HeaderValidation) {
    auto header = buildGpuPackedHeader(
        RepackFormat::Q8_0, /*codebook_id=*/19, /*payload_bytes=*/32,
        /*is_asymmetric=*/false, /*is_superblock=*/false, /*has_emins=*/false,
        /*N=*/256, /*K=*/4096);

    EXPECT_TRUE(validateGpuPackedHeader(header, gpuPackedTotalSize(header)));
    EXPECT_EQ(header.magic, 0x54575047u); // "GPWT"
    EXPECT_EQ(header.version, 1u);
    EXPECT_EQ(header.N, 256u);
    EXPECT_EQ(header.K, 4096u);
    EXPECT_EQ(header.blocks_per_row, 4096u / 32);
    EXPECT_EQ(header.payload_size, static_cast<uint64_t>(128) * 256 * 32);
    EXPECT_EQ(header.scales_size, static_cast<uint64_t>(128) * 256 * 2);
    EXPECT_EQ(header.mins_size, 0u);
    EXPECT_EQ(header.emins_size, 0u);

    // Corrupted magic should fail
    auto bad = header;
    bad.magic = 0xDEADBEEF;
    EXPECT_FALSE(validateGpuPackedHeader(bad, gpuPackedTotalSize(bad)));

    // Wrong version should fail
    bad = header;
    bad.version = 99;
    EXPECT_FALSE(validateGpuPackedHeader(bad, gpuPackedTotalSize(bad)));
}

TEST_P(VnniUnpackTest, GpuPackedTotalSize) {
    auto header = buildGpuPackedHeader(
        RepackFormat::Q4_1, /*codebook_id=*/5, /*payload_bytes=*/16,
        /*is_asymmetric=*/true, /*is_superblock=*/false, /*has_emins=*/false,
        /*N=*/64, /*K=*/128);

    // 64 bytes header + payload + scales + mins
    size_t expected = sizeof(GpuPackedWeightsHeader)
        + header.payload_size + header.scales_size + header.mins_size;
    EXPECT_EQ(gpuPackedTotalSize(header), expected);
}

// ============================================================================
// Full end-to-end: blocks → forward → pack → upload → reverse → compare
// ============================================================================

TEST_P(VnniUnpackTest, FullPipeline_Q4_0) {
    // Complete path: host blocks → GPU forward repack → pack for transfer →
    // upload to new GPU buffers → reverse repack → download → compare with original
    const int N = 128;
    const int K = 256;
    const int blocks_per_row = K / 32; // 8
    const int total_blocks = N * blocks_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;
    const int payload_bytes = 16;

    // 1. Create original blocks
    std::vector<Q4_0Block> original(total_blocks);
    fill_q4_0_blocks(original.data(), total_blocks);

    // 2. GPU: upload → forward repack
    GpuMem d_raw1(backend_, device_id_, total_blocks * sizeof(Q4_0Block));
    GpuMem d_payload1(backend_, device_id_, total_output * payload_bytes);
    GpuMem d_scales1(backend_, device_id_, total_output * sizeof(uint16_t));

    ASSERT_TRUE(backend_->hostToDevice(d_raw1.ptr, original.data(),
                                       total_blocks * sizeof(Q4_0Block),
                                       device_id_));
    ASSERT_TRUE(forwardRepack(RepackFormat::Q4_0, d_raw1.ptr,
                              d_payload1.u8(), d_scales1.u16(), nullptr,
                              N, K));
    backend_->streamSynchronize(device_id_);

    // 3. Pack → host buffer
    auto header = buildGpuPackedHeader(
        RepackFormat::Q4_0, 0, payload_bytes,
        false, false, false, N, K);
    auto packed = WeightTranslator::packGpuWeightsForTransfer(
        *backend_, device_id_,
        d_payload1.u8(), d_scales1.u16(), nullptr, nullptr, header);

    // 4. Parse and upload to fresh GPU buffers
    GpuPackedWeightsView view;
    ASSERT_TRUE(parseGpuPackedBuffer(packed.data(), packed.size(), view));

    GpuMem d_payload2(backend_, device_id_, total_output * payload_bytes);
    GpuMem d_scales2(backend_, device_id_, total_output * sizeof(uint16_t));
    ASSERT_TRUE(WeightTranslator::uploadGpuPackedWeights(
        *backend_, device_id_, view,
        d_payload2.u8(), d_scales2.u16(), nullptr, nullptr));
    backend_->streamSynchronize(device_id_);

    // 5. Reverse repack back to raw blocks
    GpuMem d_raw2(backend_, device_id_, total_blocks * sizeof(Q4_0Block));
    ASSERT_TRUE(reverseRepack(RepackFormat::Q4_0, d_payload2.u8(),
                              d_scales2.u16(), nullptr, d_raw2.ptr, N, K));
    backend_->streamSynchronize(device_id_);

    // 6. Download and compare with original
    std::vector<Q4_0Block> recovered(total_blocks);
    ASSERT_TRUE(backend_->deviceToHost(recovered.data(), d_raw2.ptr,
                                       total_blocks * sizeof(Q4_0Block),
                                       device_id_));

    for (int i = 0; i < total_blocks; ++i) {
        ASSERT_EQ(std::memcmp(&original[i], &recovered[i], sizeof(Q4_0Block)), 0)
            << "Block " << i << " mismatch in full pipeline round-trip";
    }
}

} // anonymous namespace
