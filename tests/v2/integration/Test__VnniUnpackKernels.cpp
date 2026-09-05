/**
 * @file Test__VnniUnpackKernels.cpp
 * @brief Integration tests for GPU VNNI reverse repack and WeightTranslator
 *
 * Tests two things:
 *
 * 1. **Round-trip repack**: For each reversible format (Q4_0, IQ4_NL, Q4_1,
 *    Q5_0, Q5_1, Q8_0, Q8_1, Q8_K), creates synthetic GGUF blocks, runs forward repack
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
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "loaders/MmapRegion.h"
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "loaders/gpu_pipeline/GpuPackedWeightsFormat.h"
#include "loaders/gpu_pipeline/RepackFormat.h"
#include "loaders/gpu_pipeline/WeightTranslator.h"
#include "tensors/BlockStructures.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "../utils/TestTensorFactory.h"

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

void fill_q8_1_blocks(Q8_1Block* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = static_cast<uint16_t>(0x3000 + (i & 0x3F));
        int sum = 0;
        for (int j = 0; j < 32; ++j) {
            blocks[i].qs[j] = static_cast<int8_t>(((i * 37 + j * 11) % 255) - 127);
            sum += blocks[i].qs[j];
        }
        blocks[i].sum_qs = static_cast<int16_t>(sum);
    }
}

void fill_q8_k_blocks(Q8_KBlock* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        for (int partial = 0; partial < 16; ++partial) {
            int sum = 0;
            for (int lane = 0; lane < 16; ++lane) {
                const int index = partial * 16 + lane;
                blocks[i].qs[index] = static_cast<int8_t>(
                    ((i * 53 + partial * 17 + lane * 7) % 255) - 127);
                sum += blocks[i].qs[index];
            }
            blocks[i].bsums[partial] = static_cast<int16_t>(sum);
        }
    }
}

/**
 * @brief Owns a temporary file and its demand-paged mapping for direct-I/O tests.
 *
 * Destruction releases the mapping before unlinking the file so MmapRegion's
 * live-source registry cannot retain a path that no longer exists.
 */
class TemporaryMappedSourceFile {
public:
    TemporaryMappedSourceFile(
        const std::string& backend_name,
        const std::vector<uint8_t>& bytes)
        : path_(
              std::filesystem::temp_directory_path() /
              ("llaminar_direct_loader_" + backend_name + "_" +
               std::to_string(::getpid()) + ".bin")) {
        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        if (!output)
            return;
        output.write(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
        output.close();
        region_ = MmapRegion::create(
            path_.string(),
            /*numa_node=*/-1,
            /*skip_cache_eviction=*/false,
            MmapRegion::PrefaultPolicy::DemandPaged);
    }

    ~TemporaryMappedSourceFile() {
        region_.reset();
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    MmapRegion* region() const { return region_.get(); }

private:
    std::filesystem::path path_;
    std::unique_ptr<MmapRegion> region_;
};

using VnniTensorFactory =
    std::function<std::unique_ptr<TensorBase>(size_t, size_t)>;

/**
 * @brief One launchable source format and its deterministic tensor factory.
 */
struct VnniFormatCase
{
    const char* name;
    VnniTensorFactory create;
};

/**
 * @brief Return the single all-format catalog shared by packing regressions.
 *
 * Keeping row-chunk and grouped-layout tests on one catalog prevents a newly
 * supported codebook from being certified by one storage mode but silently
 * omitted from the other.
 */
const std::vector<VnniFormatCase>& allLaunchableVnniFormats()
{
    static const std::vector<VnniFormatCase> formats = {
        {"Q4_0", [](size_t n, size_t k) { return test::TestTensorFactory::createQ4_0Random({n, k}); }},
        {"IQ4_NL", [](size_t n, size_t k) { return test::TestTensorFactory::createIQ4_NLRandom({n, k}); }},
        {"Q4_1", [](size_t n, size_t k) { return test::TestTensorFactory::createQ4_1Random({n, k}); }},
        {"Q5_0", [](size_t n, size_t k) { return test::TestTensorFactory::createQ5_0Random({n, k}); }},
        {"Q5_1", [](size_t n, size_t k) { return test::TestTensorFactory::createQ5_1Random({n, k}); }},
        {"Q8_0", [](size_t n, size_t k) { return test::TestTensorFactory::createQ8_0Random({n, k}); }},
        {"Q8_1", [](size_t n, size_t k) { return test::TestTensorFactory::createQ8_1Random({n, k}); }},
        {"Q8_K", [](size_t n, size_t k) { return test::TestTensorFactory::createQ8_KRandom({n, k}); }},
        {"IQ4_XS", [](size_t n, size_t k) { return test::TestTensorFactory::createIQ4_XSRandom({n, k}); }},
        {"Q4_K", [](size_t n, size_t k) { return test::TestTensorFactory::createQ4_KRandom({n, k}); }},
        {"Q5_K", [](size_t n, size_t k) { return test::TestTensorFactory::createQ5_KRandom({n, k}); }},
        {"Q6_K", [](size_t n, size_t k) { return test::TestTensorFactory::createQ6_KRandom({n, k}); }},
        {"Q3_K", [](size_t n, size_t k) { return test::TestTensorFactory::createQ3_KRandom({n, k}); }},
        {"Q2_K", [](size_t n, size_t k) { return test::TestTensorFactory::createQ2_KRandom({n, k}); }},
        {"IQ3_S", [](size_t n, size_t k) { return test::TestTensorFactory::createIQ3_SRandom({n, k}); }},
        {"IQ3_XXS", [](size_t n, size_t k) { return test::TestTensorFactory::createIQ3_XXSRandom({n, k}); }},
        {"IQ2_S", [](size_t n, size_t k) { return test::TestTensorFactory::createIQ2_SRandom({n, k}); }},
        {"IQ2_XS", [](size_t n, size_t k) { return test::TestTensorFactory::createIQ2_XSRandom({n, k}); }},
        {"IQ2_XXS", [](size_t n, size_t k) { return test::TestTensorFactory::createIQ2_XXSRandom({n, k}); }},
        {"IQ1_S", [](size_t n, size_t k) { return test::TestTensorFactory::createIQ1_SRandom({n, k}); }},
        {"IQ1_M", [](size_t n, size_t k) { return test::TestTensorFactory::createIQ1_MRandom({n, k}); }},
    };
    return formats;
}

// ============================================================================
// Test fixture — parameterized over backend name
// ============================================================================

class VnniUnpackTest : public ::testing::TestWithParam<std::string> {
protected:
    IBackend* backend_ = nullptr;
    int device_id_ = 0;
    DeviceType device_type_ = DeviceType::CPU;
    void* stream_ = nullptr;

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

        ASSERT_NE(backend_, nullptr);
        stream_ = backend_->createStream(device_id_);
        ASSERT_NE(stream_, nullptr) << backend_name << " explicit non-blocking stream";
    }

    void TearDown() override {
        if (backend_ && stream_) {
            EXPECT_TRUE(backend_->synchronizeStream(stream_, device_id_));
            backend_->destroyStream(stream_, device_id_);
            stream_ = nullptr;
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
        uint32_t* u32() { return static_cast<uint32_t*>(ptr); }
    };

    // ========================================================================
    // Backend-agnostic forward repack dispatch
    // ========================================================================
    bool forwardRepack(RepackFormat format, const void* d_raw,
                       uint8_t* d_payload, uint16_t* d_scales, uint16_t* d_mins,
                       int N, int K) {
        return WeightTranslator::forwardRepackOnDevice(
            format, d_raw, d_payload, d_scales, d_mins, nullptr,
            N, K, device_type_, stream_);
    }

    /**
     * @brief Launch the production row-chunk-aware forward repack entry point.
     *
     * The source contains only N contiguous rows, while the destination has the
     * complete output_N stride. This is the exact contract used by the bounded
     * model loader for matrices larger than a pinned staging slot.
     */
    bool forwardRepackChunk(RepackFormat format, const void* d_raw,
                            uint8_t* d_payload, uint16_t* d_scales,
                            uint16_t* d_mins, uint32_t* d_emins,
                            int N, int K, int output_N,
                            int output_row_offset,
                            int packed_group_rows = 0,
                            int allocation_payload_bytes_per_block = 0) {
        const int payload_capacity =
            allocation_payload_bytes_per_block > 0
                ? allocation_payload_bytes_per_block
                : repackPayloadBytesPerBlock(format);
        if (device_type_ == DeviceType::CUDA) {
#ifdef HAVE_CUDA
            return launchVnniRepackCUDA(
                format, d_raw, d_payload, d_scales, d_mins, d_emins,
                N, K, output_N, output_row_offset,
                packed_group_rows, payload_capacity, stream_);
#else
            return false;
#endif
        }
        if (device_type_ == DeviceType::ROCm) {
#ifdef HAVE_ROCM
            return launchVnniRepack(
                format, d_raw, d_payload, d_scales, d_mins, d_emins,
                N, K, output_N, output_row_offset,
                packed_group_rows, payload_capacity, stream_);
#else
            return false;
#endif
        }
        return false;
    }

    // ========================================================================
    // Backend-agnostic reverse repack dispatch
    // ========================================================================
    bool reverseRepack(RepackFormat format, const uint8_t* d_payload,
                       const uint16_t* d_scales, const uint16_t* d_mins,
                       void* d_raw, int N, int K) {
        return WeightTranslator::reverseRepackOnDevice(
            format, d_payload, d_scales, d_mins, d_raw, N, K,
            device_type_, stream_);
    }

    // ========================================================================
    // Round-trip test template
    // ========================================================================
    template<typename BlockT>
    void roundTripTest(RepackFormat format,
                       void (*filler)(BlockT*, int),
                       int payload_bytes,
                       bool is_asymmetric,
                       int N, int K,
                       int source_block_elements = 32) {
        const int blocks_per_row = K / 32;
        const int source_blocks_per_row =
            (K + source_block_elements - 1) / source_block_elements;
        const int total_blocks = N * source_blocks_per_row;
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
                                           device_id_, stream_));

        // 4. Forward repack: raw → separated
        ASSERT_TRUE(forwardRepack(format, d_raw_in.ptr,
                                  d_payload.u8(), d_scales.u16(),
                                  is_asymmetric ? d_mins.u16() : nullptr,
                                  N, K));
        ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

        // 5. Reverse repack: separated → raw
        ASSERT_TRUE(reverseRepack(format, d_payload.u8(), d_scales.u16(),
                                  is_asymmetric ? d_mins.u16() : nullptr,
                                  d_raw_out.ptr, N, K));
        ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

        // 6. Download recovered blocks
        std::vector<BlockT> recovered(total_blocks);
        ASSERT_TRUE(backend_->deviceToHost(recovered.data(), d_raw_out.ptr,
                                           total_blocks * sizeof(BlockT),
                                           device_id_, stream_));
        ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

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

/**
 * @brief Prove a previous runner's sticky launch status cannot poison repack.
 *
 * CUDA and HIP retain launch errors per host thread until get-last-error
 * consumes them.  Model teardown and construction happen on that same thread,
 * so the production repack entry point must clear any predecessor status
 * immediately before dispatching its own kernel.
 */
TEST_P(VnniUnpackTest, ForwardRepackOwnsItsLaunchStatus) {
    constexpr int N = 8;
    constexpr int K = 32;
    constexpr int blocks_per_row = K / 32;
    constexpr size_t output_blocks =
        static_cast<size_t>(N) * blocks_per_row;

    std::vector<Q4_0Block> host_blocks(output_blocks);
    fill_q4_0_blocks(host_blocks.data(), static_cast<int>(host_blocks.size()));

    GpuMem d_raw(backend_, device_id_, host_blocks.size() * sizeof(Q4_0Block));
    GpuMem d_payload(backend_, device_id_, output_blocks * 16);
    GpuMem d_scales(backend_, device_id_, output_blocks * sizeof(uint16_t));
    ASSERT_NE(d_raw.ptr, nullptr);
    ASSERT_NE(d_payload.ptr, nullptr);
    ASSERT_NE(d_scales.ptr, nullptr);
    ASSERT_TRUE(backend_->hostToDevice(
        d_raw.ptr,
        host_blocks.data(),
        host_blocks.size() * sizeof(Q4_0Block),
        device_id_,
        stream_));

    /*
     * This reaches cudaStreamWaitEvent/hipStreamWaitEvent with an invalid null
     * event and therefore seeds runtime error state without changing the valid
     * stream, current device, or allocations used by the repack below.
     */
    EXPECT_FALSE(backend_->streamWaitEvent(stream_, nullptr, device_id_));

    ASSERT_TRUE(forwardRepack(
        RepackFormat::Q4_0,
        d_raw.ptr,
        d_payload.u8(),
        d_scales.u16(),
        nullptr,
        N,
        K));
    ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));
}

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

TEST_P(VnniUnpackTest, RoundTrip_Q8_1) {
    roundTripTest<Q8_1Block>(RepackFormat::Q8_1, fill_q8_1_blocks,
                             /*payload_bytes=*/32, /*asymmetric=*/false,
                             /*N=*/64, /*K=*/128);
}

TEST_P(VnniUnpackTest, RoundTrip_Q8_K) {
    roundTripTest<Q8_KBlock>(RepackFormat::Q8_K, fill_q8_k_blocks,
                             /*payload_bytes=*/32, /*asymmetric=*/false,
                             /*N=*/64, /*K=*/512,
                             /*source_block_elements=*/256);
}

/**
 * @brief Prove bounded row-chunk publication is byte-identical for every format.
 *
 * This regression compares one whole-matrix repack with three contiguous source
 * row chunks that publish into the same full-N destination geometry. It covers
 * every launchable GGUF quantization format symmetrically on CUDA and ROCm, so a
 * format-specific kernel cannot accidentally retain the old chunk-local output
 * stride.
 */
TEST_P(VnniUnpackTest, RowChunkedRepackMatchesWholeMatrixForEveryFormat) {
    const auto& formats = allLaunchableVnniFormats();

    constexpr int N = 11;
    constexpr int K = 256;
    constexpr int kMaximumChunkRows = 5;
    constexpr int kChunkRows[] = {3, 5, 3};
    const size_t output_blocks = static_cast<size_t>(N) * (K / 32);

    for (const auto& format_case : formats) {
        SCOPED_TRACE(::testing::Message()
                     << GetParam() << "/" << format_case.name);

        auto tensor = format_case.create(N, K);
        ASSERT_NE(tensor, nullptr);
        const auto* unpackable = dynamic_cast<const IINT8Unpackable*>(tensor.get());
        ASSERT_NE(unpackable, nullptr);
        const auto* info = unpackable->vnniFormatInfo();
        ASSERT_NE(info, nullptr);
        const auto format =
            codebookIdToRepackFormat(info->codebook_id, info->is_superblock);
        ASSERT_TRUE(format.has_value());
        ASSERT_EQ(tensor->size_bytes() % static_cast<size_t>(N), 0u);

        const size_t source_row_bytes =
            tensor->size_bytes() / static_cast<size_t>(N);
        const size_t payload_bytes =
            output_blocks * static_cast<size_t>(info->payload_bytes);
        const size_t scales_bytes = output_blocks * sizeof(uint16_t);
        const size_t mins_bytes =
            info->is_asymmetric ? scales_bytes : 0;
        const size_t emins_bytes =
            info->has_emins ? output_blocks * sizeof(uint32_t) : 0;

        GpuMem d_raw_whole(backend_, device_id_, tensor->size_bytes());
        GpuMem d_raw_chunk(
            backend_, device_id_,
            source_row_bytes * static_cast<size_t>(kMaximumChunkRows));
        GpuMem whole_payload(backend_, device_id_, payload_bytes);
        GpuMem whole_scales(backend_, device_id_, scales_bytes);
        GpuMem whole_mins(backend_, device_id_, mins_bytes);
        GpuMem whole_emins(backend_, device_id_, emins_bytes);
        GpuMem chunked_payload(backend_, device_id_, payload_bytes);
        GpuMem chunked_scales(backend_, device_id_, scales_bytes);
        GpuMem chunked_mins(backend_, device_id_, mins_bytes);
        GpuMem chunked_emins(backend_, device_id_, emins_bytes);

        ASSERT_NE(d_raw_whole.ptr, nullptr);
        ASSERT_NE(d_raw_chunk.ptr, nullptr);
        ASSERT_NE(whole_payload.ptr, nullptr);
        ASSERT_NE(whole_scales.ptr, nullptr);
        ASSERT_NE(chunked_payload.ptr, nullptr);
        ASSERT_NE(chunked_scales.ptr, nullptr);

        ASSERT_TRUE(backend_->hostToDevice(
            d_raw_whole.ptr, tensor->raw_data(), tensor->size_bytes(),
            device_id_, stream_));
        ASSERT_TRUE(forwardRepackChunk(
            *format, d_raw_whole.ptr,
            whole_payload.u8(), whole_scales.u16(),
            info->is_asymmetric ? whole_mins.u16() : nullptr,
            info->has_emins ? whole_emins.u32() : nullptr,
            N, K, N, 0));

        int row_offset = 0;
        const auto* source = static_cast<const uint8_t*>(tensor->raw_data());
        for (const int chunk_rows : kChunkRows) {
            const size_t chunk_bytes =
                source_row_bytes * static_cast<size_t>(chunk_rows);
            ASSERT_TRUE(backend_->hostToDevice(
                d_raw_chunk.ptr,
                source + static_cast<size_t>(row_offset) * source_row_bytes,
                chunk_bytes, device_id_, stream_));
            ASSERT_TRUE(forwardRepackChunk(
                *format, d_raw_chunk.ptr,
                chunked_payload.u8(), chunked_scales.u16(),
                info->is_asymmetric ? chunked_mins.u16() : nullptr,
                info->has_emins ? chunked_emins.u32() : nullptr,
                chunk_rows, K, N, row_offset));
            row_offset += chunk_rows;
        }
        ASSERT_EQ(row_offset, N);
        ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

        auto expect_device_bytes_equal =
            [&](const GpuMem& whole, const GpuMem& chunked,
                size_t bytes, const char* field) {
                if (bytes == 0)
                    return;
                std::vector<uint8_t> whole_host(bytes);
                std::vector<uint8_t> chunked_host(bytes);
                ASSERT_TRUE(backend_->deviceToHost(
                    whole_host.data(), whole.ptr, bytes, device_id_, stream_));
                ASSERT_TRUE(backend_->deviceToHost(
                    chunked_host.data(), chunked.ptr, bytes, device_id_, stream_));
                ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));
                EXPECT_EQ(chunked_host, whole_host) << field;
            };

        expect_device_bytes_equal(
            whole_payload, chunked_payload, payload_bytes, "payload");
        expect_device_bytes_equal(
            whole_scales, chunked_scales, scales_bytes, "scales");
        expect_device_bytes_equal(
            whole_mins, chunked_mins, mins_bytes, "mins");
        expect_device_bytes_equal(
            whole_emins, chunked_emins, emins_bytes, "emins");
    }
}

/**
 * @brief Prove coalesced expert packing equals independent expert launches.
 *
 * Production reads one contiguous GGUF parent and packs it into a slab whose
 * expert subregions must remain directly consumable by the existing GEMM and
 * transfer kernels. Every expert is allocated for the union of its compact
 * source representation and its CPU-promotion representation, while the live
 * payload stays compact. The bounded chunks below intentionally begin and end
 * inside different seven-row experts. This catches full-N interleaving,
 * capacity-stride overwrites, and chunk-boundary mistakes for all source
 * codebooks on both GPU backends.
 */
TEST_P(VnniUnpackTest, GroupedExpertReusableCapacityMatchesStandaloneForEveryFormat) {
    constexpr int kExpertCount = 3;
    constexpr int kRowsPerExpert = 7;
    constexpr int kTotalRows = kExpertCount * kRowsPerExpert;
    constexpr int kColumns = 256;
    constexpr int kMaximumChunkRows = 8;
    constexpr int kChunkRows[] = {5, 8, 8};

    for (const auto& format_case : allLaunchableVnniFormats()) {
        SCOPED_TRACE(::testing::Message()
                     << GetParam() << "/" << format_case.name);

        auto tensor = format_case.create(kTotalRows, kColumns);
        ASSERT_NE(tensor, nullptr);
        const auto* unpackable =
            dynamic_cast<const IINT8Unpackable*>(tensor.get());
        ASSERT_NE(unpackable, nullptr);
        const auto* info = unpackable->vnniFormatInfo();
        ASSERT_NE(info, nullptr);
        const auto format = codebookIdToRepackFormat(
            info->codebook_id, info->is_superblock);
        ASSERT_TRUE(format.has_value());
        ASSERT_EQ(
            tensor->size_bytes() % static_cast<size_t>(kTotalRows), 0u);

        const size_t source_row_bytes =
            tensor->size_bytes() / static_cast<size_t>(kTotalRows);
        const auto expert_regions = nativeVnniPackedRegionSizes(
            kRowsPerExpert, kColumns, *info);
        const auto compact_slab_regions = nativeVnniPackedRegionSizes(
            kTotalRows, kColumns, *info);
        const auto allocation = reusableDeviceVnniAllocationFormat(*info);
        const NativeVnniFormatInfo allocation_info{
            .codebook_id = info->codebook_id,
            .payload_bytes = allocation.payload_bytes_per_block,
            .is_asymmetric = allocation.has_mins,
            .is_superblock = info->is_superblock,
            .has_emins = allocation.has_emins,
            .max_abs_factor = info->max_abs_factor,
        };
        const auto expert_allocation_regions = nativeVnniPackedRegionSizes(
            kRowsPerExpert, kColumns, allocation_info);
        const auto slab_allocation_regions = nativeVnniPackedRegionSizes(
            kTotalRows, kColumns, allocation_info);

        ASSERT_EQ(
            compact_slab_regions.payload_bytes,
            kExpertCount * expert_regions.payload_bytes);
        ASSERT_EQ(
            compact_slab_regions.scales_bytes,
            kExpertCount * expert_regions.scales_bytes);
        ASSERT_EQ(
            compact_slab_regions.mins_bytes,
            kExpertCount * expert_regions.mins_bytes);
        ASSERT_EQ(
            compact_slab_regions.emins_bytes,
            kExpertCount * expert_regions.emins_bytes);
        ASSERT_EQ(
            slab_allocation_regions.payload_bytes,
            kExpertCount * expert_allocation_regions.payload_bytes);

        GpuMem standalone_raw(
            backend_, device_id_,
            source_row_bytes * static_cast<size_t>(kRowsPerExpert));
        GpuMem grouped_raw(
            backend_, device_id_,
            source_row_bytes * static_cast<size_t>(kMaximumChunkRows));
        GpuMem standalone_payload(
            backend_, device_id_, slab_allocation_regions.payload_bytes);
        GpuMem standalone_scales(
            backend_, device_id_, slab_allocation_regions.scales_bytes);
        GpuMem standalone_mins(
            backend_, device_id_, slab_allocation_regions.mins_bytes);
        GpuMem standalone_emins(
            backend_, device_id_, slab_allocation_regions.emins_bytes);
        GpuMem grouped_payload(
            backend_, device_id_, slab_allocation_regions.payload_bytes);
        GpuMem grouped_scales(
            backend_, device_id_, slab_allocation_regions.scales_bytes);
        GpuMem grouped_mins(
            backend_, device_id_, slab_allocation_regions.mins_bytes);
        GpuMem grouped_emins(
            backend_, device_id_, slab_allocation_regions.emins_bytes);

        ASSERT_NE(standalone_raw.ptr, nullptr);
        ASSERT_NE(grouped_raw.ptr, nullptr);
        ASSERT_NE(standalone_payload.ptr, nullptr);
        ASSERT_NE(standalone_scales.ptr, nullptr);
        ASSERT_NE(grouped_payload.ptr, nullptr);
        ASSERT_NE(grouped_scales.ptr, nullptr);

        const auto initialize_region = [&](GpuMem& memory) {
            if (memory.bytes == 0)
                return;
            ASSERT_TRUE(backend_->memset(
                memory.ptr, 0xA5, memory.bytes, device_id_, stream_));
        };
        initialize_region(standalone_payload);
        initialize_region(standalone_scales);
        initialize_region(standalone_mins);
        initialize_region(standalone_emins);
        initialize_region(grouped_payload);
        initialize_region(grouped_scales);
        initialize_region(grouped_mins);
        initialize_region(grouped_emins);

        const auto* source =
            static_cast<const uint8_t*>(tensor->raw_data());
        for (int expert = 0; expert < kExpertCount; ++expert) {
            const size_t source_offset =
                static_cast<size_t>(expert * kRowsPerExpert) *
                source_row_bytes;
            ASSERT_TRUE(backend_->hostToDevice(
                standalone_raw.ptr,
                source + source_offset,
                source_row_bytes * static_cast<size_t>(kRowsPerExpert),
                device_id_, stream_));

            auto* payload = standalone_payload.u8() +
                            static_cast<size_t>(expert) *
                                expert_allocation_regions.payload_bytes;
            auto* scales = reinterpret_cast<uint16_t*>(
                standalone_scales.u8() +
                static_cast<size_t>(expert) *
                    expert_allocation_regions.scales_bytes);
            auto* mins = info->is_asymmetric
                             ? reinterpret_cast<uint16_t*>(
                                   standalone_mins.u8() +
                                   static_cast<size_t>(expert) *
                                       expert_allocation_regions.mins_bytes)
                             : nullptr;
            auto* emins = info->has_emins
                              ? reinterpret_cast<uint32_t*>(
                                    standalone_emins.u8() +
                                    static_cast<size_t>(expert) *
                                        expert_allocation_regions.emins_bytes)
                              : nullptr;
            ASSERT_TRUE(forwardRepackChunk(
                *format, standalone_raw.ptr,
                payload, scales, mins, emins,
                kRowsPerExpert, kColumns, kRowsPerExpert, 0));
        }

        int row_offset = 0;
        for (const int chunk_rows : kChunkRows) {
            const size_t chunk_bytes =
                static_cast<size_t>(chunk_rows) * source_row_bytes;
            ASSERT_TRUE(backend_->hostToDevice(
                grouped_raw.ptr,
                source + static_cast<size_t>(row_offset) * source_row_bytes,
                chunk_bytes, device_id_, stream_));
            ASSERT_TRUE(forwardRepackChunk(
                *format, grouped_raw.ptr,
                grouped_payload.u8(), grouped_scales.u16(),
                info->is_asymmetric ? grouped_mins.u16() : nullptr,
                info->has_emins ? grouped_emins.u32() : nullptr,
                chunk_rows, kColumns, kTotalRows, row_offset,
                kRowsPerExpert,
                allocation.payload_bytes_per_block));
            row_offset += chunk_rows;
        }
        ASSERT_EQ(row_offset, kTotalRows);
        ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

        auto expect_device_bytes_equal =
            [&](const GpuMem& standalone,
                const GpuMem& grouped,
                size_t bytes,
                const char* field) {
                if (bytes == 0)
                    return;
                std::vector<uint8_t> expected(bytes);
                std::vector<uint8_t> actual(bytes);
                ASSERT_TRUE(backend_->deviceToHost(
                    expected.data(), standalone.ptr, bytes,
                    device_id_, stream_));
                ASSERT_TRUE(backend_->deviceToHost(
                    actual.data(), grouped.ptr, bytes,
                    device_id_, stream_));
                ASSERT_TRUE(
                    backend_->synchronizeStream(stream_, device_id_));
                EXPECT_EQ(actual, expected) << field;
            };

        expect_device_bytes_equal(
            standalone_payload, grouped_payload,
            slab_allocation_regions.payload_bytes, "payload");
        expect_device_bytes_equal(
            standalone_scales, grouped_scales,
            slab_allocation_regions.scales_bytes, "scales");
        expect_device_bytes_equal(
            standalone_mins, grouped_mins,
            slab_allocation_regions.mins_bytes, "mins");
        expect_device_bytes_equal(
            standalone_emins, grouped_emins,
            slab_allocation_regions.emins_bytes, "emins");
    }
}

/**
 * @brief Proves bounded buffered-file staging across producer-lane reuse.
 *
 * Six unaligned matrix ranges force every one of the three pinned lanes to be
 * refilled after its first H2D. No tensor metadata advertises that those ranges
 * are mapped: production must discover that fact from the concrete addresses.
 * The output is compared byte-for-byte with a CPU layout oracle, catching
 * provenance, exact source offsets, short reads, early pinned overwrite, and
 * device-staging reuse bugs.
 */
TEST_P(VnniUnpackTest, BoundedBufferedMmapPipelineMatchesAcrossLaneReuse) {
    constexpr int N = 11;
    constexpr int K = 256;
    constexpr int kJobCount = 6;
    constexpr size_t kPageBytes = 4096;
    constexpr size_t kSourcePrefix = 37;
    constexpr size_t kPinnedSlotBytes = 16 * 1024;
    constexpr int kLaneCount = 3;
    constexpr size_t kBlocksPerMatrix =
        static_cast<size_t>(N) * (K / 32);
    constexpr size_t kRawBytes =
        kBlocksPerMatrix * sizeof(Q4_0Block);

    std::vector<std::vector<Q4_0Block>> matrices(kJobCount);
    std::vector<uint8_t> file_bytes(
        static_cast<size_t>(kJobCount + 1) * kPageBytes, 0xA5);
    for (int job_index = 0; job_index < kJobCount; ++job_index) {
        auto& blocks = matrices[static_cast<size_t>(job_index)];
        blocks.resize(kBlocksPerMatrix);
        fill_q4_0_blocks(blocks.data(), static_cast<int>(blocks.size()));
        for (size_t block = 0; block < blocks.size(); ++block) {
            blocks[block].d = static_cast<uint16_t>(
                0x3000 + job_index * 0x20 + static_cast<int>(block % 0x1F));
            for (int byte = 0; byte < 16; ++byte) {
                blocks[block].qs[byte] ^= static_cast<uint8_t>(
                    job_index * 29 + static_cast<int>(block));
            }
        }

        const size_t source_offset =
            static_cast<size_t>(job_index) * kPageBytes + kSourcePrefix;
        std::memcpy(
            file_bytes.data() + source_offset,
            blocks.data(),
            kRawBytes);
    }

    TemporaryMappedSourceFile mapped(GetParam(), file_bytes);
    ASSERT_NE(mapped.region(), nullptr);

    LoadOrchestrator orchestrator(
        backend_, kTestOnlyUnadmittedGPUAllocation);
    orchestrator.addDevice(device_id_);
    for (int job_index = 0; job_index < kJobCount; ++job_index) {
        orchestrator.planWeight(
            device_id_,
            "direct_q4_" + std::to_string(job_index),
            N, K,
            /*payload_bytes_per_block=*/16,
            /*is_asymmetric=*/false,
            /*has_emins=*/false,
            kRawBytes);
    }
    orchestrator.allocate(kPinnedSlotBytes, kLaneCount);

    for (int job_index = 0; job_index < kJobCount; ++job_index) {
        const size_t source_offset =
            static_cast<size_t>(job_index) * kPageBytes + kSourcePrefix;
        WeightJob job{
            .name = "direct_q4_" + std::to_string(job_index),
            .host_raw_data = mapped.region()->data() + source_offset,
            .raw_bytes = kRawBytes,
            .format = RepackFormat::Q4_0,
            .N = N,
            .K = K,
            .is_asymmetric = false,
        };
        orchestrator.addWeightJob(device_id_, job);
    }

    ASSERT_NO_THROW(orchestrator.load());
    auto* pool = orchestrator.getPool(device_id_);
    ASSERT_NE(pool, nullptr);

    for (int job_index = 0; job_index < kJobCount; ++job_index) {
        const auto slot =
            pool->getSlot("direct_q4_" + std::to_string(job_index));
        ASSERT_TRUE(slot.has_value());

        std::vector<uint8_t> expected_payload(kBlocksPerMatrix * 16);
        std::vector<uint16_t> expected_scales(kBlocksPerMatrix);
        const auto& blocks = matrices[static_cast<size_t>(job_index)];
        for (int row = 0; row < N; ++row) {
            for (int block = 0; block < K / 32; ++block) {
                const size_t source_index =
                    static_cast<size_t>(row) * (K / 32) +
                    static_cast<size_t>(block);
                const size_t destination_index =
                    static_cast<size_t>(block) * N +
                    static_cast<size_t>(row);
                std::memcpy(
                    expected_payload.data() + destination_index * 16,
                    blocks[source_index].qs,
                    16);
                expected_scales[destination_index] = blocks[source_index].d;
            }
        }

        std::vector<uint8_t> actual_payload(expected_payload.size());
        std::vector<uint16_t> actual_scales(expected_scales.size());
        ASSERT_TRUE(backend_->deviceToHost(
            actual_payload.data(),
            slot->d_native_vnni_payload,
            actual_payload.size(),
            device_id_, stream_));
        ASSERT_TRUE(backend_->deviceToHost(
            actual_scales.data(),
            slot->d_native_vnni_scales,
            actual_scales.size() * sizeof(uint16_t),
            device_id_, stream_));
        ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));
        EXPECT_EQ(actual_payload, expected_payload)
            << "payload job=" << job_index;
        EXPECT_EQ(actual_scales, expected_scales)
            << "scales job=" << job_index;
    }

    orchestrator.finalize();
}

/**
 * @brief Reproduces the multi-GPU bounded-budget staging alignment failure.
 *
 * Production divides one bounded startup budget first across GPU devices and
 * then across three upload lanes. Those integer divisions can produce an odd
 * logical slot capacity. Historically WeightVRAMPool also used that capacity
 * as its physical device stride, so lane 1 and lane 2 became under-aligned and
 * Q6_K repack faulted on its naturally aligned packed-source loads.
 *
 * This test deliberately requests an odd capacity smaller than one Q6_K
 * matrix. LoadOrchestrator consequently splits the matrix into three row
 * chunks, exercising every production staging lane. It verifies both the
 * device-address invariant and byte equality against a whole-matrix launch.
 */
TEST_P(
    VnniUnpackTest,
    OddBoundedDeviceStagingCapacityKeepsEveryLaneAlignedAndQ6KExact) {
    constexpr int N = 2048;
    constexpr int K = 256;
    constexpr int kLaneCount = 3;
    constexpr size_t kOddSlotCapacity = 160001;
    constexpr size_t kRequiredDeviceAlignment = 256;

    auto tensor = test::TestTensorFactory::createQ6_KRandom({N, K});
    ASSERT_NE(tensor, nullptr);
    const auto* unpackable =
        dynamic_cast<const IINT8Unpackable*>(tensor.get());
    ASSERT_NE(unpackable, nullptr);
    const auto* info = unpackable->vnniFormatInfo();
    ASSERT_NE(info, nullptr);
    ASSERT_EQ(info->codebook_id, 8);
    ASSERT_TRUE(info->is_superblock);
    ASSERT_EQ(tensor->size_bytes(), static_cast<size_t>(N) * sizeof(Q6_KBlock));
    ASSERT_GT(tensor->size_bytes(), kOddSlotCapacity);

    LoadOrchestrator orchestrator(
        backend_, kTestOnlyUnadmittedGPUAllocation);
    orchestrator.addDevice(device_id_);
    orchestrator.planWeight(
        device_id_, "odd_stride_q6_k", N, K,
        info->payload_bytes, info->is_asymmetric, info->has_emins,
        tensor->size_bytes());
    orchestrator.allocate(kOddSlotCapacity, kLaneCount);

    auto* pool = orchestrator.getPool(device_id_);
    ASSERT_NE(pool, nullptr);
    ASSERT_EQ(pool->maxStagingSlotBytes(), kOddSlotCapacity);
    ASSERT_GE(pool->stagingSlotStrideBytes(), kOddSlotCapacity);
    ASSERT_EQ(
        pool->stagingSlotStrideBytes() % kRequiredDeviceAlignment, 0u);
    for (int lane = 0; lane < kLaneCount; ++lane) {
        const auto* staging = pool->getStagingSlot(lane);
        ASSERT_NE(staging, nullptr);
        ASSERT_EQ(
            reinterpret_cast<uintptr_t>(staging) %
                kRequiredDeviceAlignment,
            0u)
            << "under-aligned device staging lane " << lane;
    }

    const size_t output_blocks =
        static_cast<size_t>(N) * static_cast<size_t>(K / 32);
    const size_t payload_bytes =
        output_blocks * static_cast<size_t>(info->payload_bytes);
    const size_t scales_bytes = output_blocks * sizeof(uint16_t);
    const size_t mins_bytes =
        info->is_asymmetric ? scales_bytes : 0;
    const size_t emins_bytes =
        info->has_emins ? output_blocks * sizeof(uint32_t) : 0;

    GpuMem whole_raw(
        backend_, device_id_, tensor->size_bytes());
    GpuMem whole_payload(backend_, device_id_, payload_bytes);
    GpuMem whole_scales(backend_, device_id_, scales_bytes);
    GpuMem whole_mins(backend_, device_id_, mins_bytes);
    GpuMem whole_emins(backend_, device_id_, emins_bytes);
    ASSERT_NE(whole_raw.ptr, nullptr);
    ASSERT_NE(whole_payload.ptr, nullptr);
    ASSERT_NE(whole_scales.ptr, nullptr);
    ASSERT_TRUE(backend_->hostToDevice(
        whole_raw.ptr, tensor->raw_data(), tensor->size_bytes(),
        device_id_, stream_));
    ASSERT_TRUE(forwardRepackChunk(
        RepackFormat::Q6_K,
        whole_raw.ptr,
        whole_payload.u8(),
        whole_scales.u16(),
        info->is_asymmetric ? whole_mins.u16() : nullptr,
        info->has_emins ? whole_emins.u32() : nullptr,
        N, K, N, 0));
    ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

    WeightJob job{
        .name = "odd_stride_q6_k",
        .host_raw_data = tensor->raw_data(),
        .raw_bytes = tensor->size_bytes(),
        .format = RepackFormat::Q6_K,
        .N = N,
        .K = K,
        .is_asymmetric = info->is_asymmetric,
    };
    orchestrator.addWeightJob(device_id_, job);
    ASSERT_EQ(orchestrator.pendingJobCount(device_id_), 3u);
    ASSERT_NO_THROW(orchestrator.load());

    const auto chunked = pool->getSlot("odd_stride_q6_k");
    ASSERT_TRUE(chunked.has_value());
    auto expect_pool_field_equal =
        [&](const GpuMem& whole, const void* chunked_device,
            size_t bytes, const char* field) {
            if (bytes == 0)
                return;
            ASSERT_NE(chunked_device, nullptr);
            std::vector<uint8_t> expected(bytes);
            std::vector<uint8_t> actual(bytes);
            ASSERT_TRUE(backend_->deviceToHost(
                expected.data(), whole.ptr, bytes, device_id_, stream_));
            ASSERT_TRUE(backend_->deviceToHost(
                actual.data(), chunked_device, bytes, device_id_, stream_));
            ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));
            EXPECT_EQ(actual, expected) << field;
        };

    expect_pool_field_equal(
        whole_payload, chunked->d_native_vnni_payload,
        payload_bytes, "payload");
    expect_pool_field_equal(
        whole_scales, chunked->d_native_vnni_scales,
        scales_bytes, "scales");
    expect_pool_field_equal(
        whole_mins, chunked->d_native_vnni_mins,
        mins_bytes, "mins");
    expect_pool_field_equal(
        whole_emins, chunked->d_native_vnni_emins,
        emins_bytes, "emins");

    orchestrator.finalize();
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
    EXPECT_TRUE(isReversibleFormat(RepackFormat::Q8_1));
    EXPECT_TRUE(isReversibleFormat(RepackFormat::Q8_K));
}

TEST_P(VnniUnpackTest, ReverseRepackReturnsFailOnSuperblock) {
    // reverseRepackOnDevice should return false for non-reversible formats
    EXPECT_FALSE(WeightTranslator::reverseRepackOnDevice(
        RepackFormat::Q4_K, nullptr, nullptr, nullptr, nullptr,
        64, 128, device_type_, stream_));
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
    EXPECT_EQ(rawBlockSizeBytes(RepackFormat::Q8_1),   36u);
    EXPECT_EQ(rawBlockSizeBytes(RepackFormat::Q8_K),  288u);
    EXPECT_EQ(rawBlockSizeBytes(RepackFormat::Q4_K),   0u);
    EXPECT_EQ(rawBlockSizeBytes(RepackFormat::Q6_K),   0u);
}

TEST_P(VnniUnpackTest, RawBlockBufferSize) {
    // Q4_0: N=64, K=128 → 64 * (128/32) * 18 = 64 * 4 * 18 = 4608
    EXPECT_EQ(rawBlockBufferSize(RepackFormat::Q4_0, 64, 128), 4608u);
    // Q8_0: N=64, K=128 → 64 * 4 * 34 = 8704
    EXPECT_EQ(rawBlockBufferSize(RepackFormat::Q8_0, 64, 128), 8704u);
    // Q8_1: N=64, K=128 -> 64 * 4 * 36 = 9216
    EXPECT_EQ(rawBlockBufferSize(RepackFormat::Q8_1, 64, 128), 9216u);
    // Q8_K: N=64, K=512 -> 64 * 2 * 288 = 36864
    EXPECT_EQ(rawBlockBufferSize(RepackFormat::Q8_K, 64, 512), 36864u);
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
                                       device_id_, stream_));
    ASSERT_TRUE(forwardRepack(RepackFormat::Q4_0, d_raw.ptr,
                              d_payload.u8(), d_scales.u16(), nullptr,
                              N, K));
    ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

    // 2. Pack GPU → host buffer via WeightTranslator
    auto header = buildGpuPackedHeader(
        RepackFormat::Q4_0, /*codebook_id=*/0, payload_bytes,
        /*is_asymmetric=*/false, /*is_superblock=*/false, /*has_emins=*/false,
        N, K);

    ASSERT_TRUE(validateGpuPackedHeader(header, gpuPackedTotalSize(header)));

    auto packed_buf = WeightTranslator::packGpuWeightsForTransfer(
        *backend_, device_id_,
        d_payload.u8(), d_scales.u16(), nullptr, nullptr,
        header, stream_);
    ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

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
        d_payload2.u8(), d_scales2.u16(), nullptr, nullptr, stream_));
    ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

    // 5. Download both sets and compare byte-for-byte
    std::vector<uint8_t> orig_payload(total_output * payload_bytes);
    std::vector<uint8_t> recv_payload(total_output * payload_bytes);
    std::vector<uint16_t> orig_scales(total_output);
    std::vector<uint16_t> recv_scales(total_output);

    ASSERT_TRUE(backend_->deviceToHost(orig_payload.data(), d_payload.ptr,
                                       orig_payload.size(), device_id_, stream_));
    ASSERT_TRUE(backend_->deviceToHost(recv_payload.data(), d_payload2.ptr,
                                       recv_payload.size(), device_id_, stream_));
    ASSERT_TRUE(backend_->deviceToHost(orig_scales.data(), d_scales.ptr,
                                       orig_scales.size() * sizeof(uint16_t),
                                       device_id_, stream_));
    ASSERT_TRUE(backend_->deviceToHost(recv_scales.data(), d_scales2.ptr,
                                       recv_scales.size() * sizeof(uint16_t),
                                       device_id_, stream_));
    ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

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
                                       device_id_, stream_));
    ASSERT_TRUE(forwardRepack(RepackFormat::Q4_1, d_raw.ptr,
                              d_payload.u8(), d_scales.u16(), d_mins.u16(),
                              N, K));
    ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

    // 2. Pack → parse → upload
    auto header = buildGpuPackedHeader(
        RepackFormat::Q4_1, /*codebook_id=*/5, payload_bytes,
        /*is_asymmetric=*/true, /*is_superblock=*/false, /*has_emins=*/false,
        N, K);
    ASSERT_TRUE(validateGpuPackedHeader(header, gpuPackedTotalSize(header)));

    auto packed_buf = WeightTranslator::packGpuWeightsForTransfer(
        *backend_, device_id_,
        d_payload.u8(), d_scales.u16(), d_mins.u16(), nullptr,
        header, stream_);
    ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));
    ASSERT_EQ(packed_buf.size(), gpuPackedTotalSize(header));

    GpuPackedWeightsView view;
    ASSERT_TRUE(parseGpuPackedBuffer(packed_buf.data(), packed_buf.size(), view));
    EXPECT_NE(view.mins, nullptr) << "Asymmetric format should have mins";

    GpuMem d_payload2(backend_, device_id_, total_output * payload_bytes);
    GpuMem d_scales2(backend_, device_id_, total_output * sizeof(uint16_t));
    GpuMem d_mins2(backend_, device_id_, total_output * sizeof(uint16_t));

    ASSERT_TRUE(WeightTranslator::uploadGpuPackedWeights(
        *backend_, device_id_, view,
        d_payload2.u8(), d_scales2.u16(), d_mins2.u16(), nullptr, stream_));
    ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

    // 3. Compare
    std::vector<uint16_t> orig_mins(total_output), recv_mins(total_output);
    ASSERT_TRUE(backend_->deviceToHost(orig_mins.data(), d_mins.ptr,
                                       orig_mins.size() * sizeof(uint16_t),
                                       device_id_, stream_));
    ASSERT_TRUE(backend_->deviceToHost(recv_mins.data(), d_mins2.ptr,
                                       recv_mins.size() * sizeof(uint16_t),
                                       device_id_, stream_));
    ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));
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
                                       device_id_, stream_));
    ASSERT_TRUE(forwardRepack(RepackFormat::Q4_0, d_raw1.ptr,
                              d_payload1.u8(), d_scales1.u16(), nullptr,
                              N, K));
    ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

    // 3. Pack → host buffer
    auto header = buildGpuPackedHeader(
        RepackFormat::Q4_0, 0, payload_bytes,
        false, false, false, N, K);
    auto packed = WeightTranslator::packGpuWeightsForTransfer(
        *backend_, device_id_,
        d_payload1.u8(), d_scales1.u16(), nullptr, nullptr, header, stream_);
    ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

    // 4. Parse and upload to fresh GPU buffers
    GpuPackedWeightsView view;
    ASSERT_TRUE(parseGpuPackedBuffer(packed.data(), packed.size(), view));

    GpuMem d_payload2(backend_, device_id_, total_output * payload_bytes);
    GpuMem d_scales2(backend_, device_id_, total_output * sizeof(uint16_t));
    ASSERT_TRUE(WeightTranslator::uploadGpuPackedWeights(
        *backend_, device_id_, view,
        d_payload2.u8(), d_scales2.u16(), nullptr, nullptr, stream_));
    ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

    // 5. Reverse repack back to raw blocks
    GpuMem d_raw2(backend_, device_id_, total_blocks * sizeof(Q4_0Block));
    ASSERT_TRUE(reverseRepack(RepackFormat::Q4_0, d_payload2.u8(),
                              d_scales2.u16(), nullptr, d_raw2.ptr, N, K));
    ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

    // 6. Download and compare with original
    std::vector<Q4_0Block> recovered(total_blocks);
    ASSERT_TRUE(backend_->deviceToHost(recovered.data(), d_raw2.ptr,
                                       total_blocks * sizeof(Q4_0Block),
                                       device_id_, stream_));
    ASSERT_TRUE(backend_->synchronizeStream(stream_, device_id_));

    for (int i = 0; i < total_blocks; ++i) {
        ASSERT_EQ(std::memcmp(&original[i], &recovered[i], sizeof(Q4_0Block)), 0)
            << "Block " << i << " mismatch in full pipeline round-trip";
    }
}

} // anonymous namespace
