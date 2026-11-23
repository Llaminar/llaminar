/**
 * @file Test__ROCmSuperblockCodedMetadata.cpp
 * @brief Superblock-coded GPU metadata reproduces the per-block planes on ROCm.
 *
 * The same raw GGUF superblocks are repacked twice by the production ROCm
 * repack launcher: once into the per-block planes (metadata layout zero) and
 * once into the superblock-coded planes (per-block codes plus per-superblock
 * d/dmin). For every codable source format, on whole K (phase 0) and on a
 * column slice whose first block sits mid-superblock (phase 3), the test proves:
 *  - both repacks emit identical payload bytes, except the sign-mask parity
 *    bits of IQ3_XXS/IQ2_XXS/IQ2_XS (and IQ2_XS's spare qh nibble), which the
 *    coded payload uses for its scale code; restoring parity recovers the
 *    per-block payload exactly;
 *  - the shared codec reconstructs every legacy per-block word (primary,
 *    consumed secondary and Q2_K emin) from the coded planes at the
 *    production block-major addresses;
 *  - the coded plane sizes equal sb::planeBytes (file density), and a coded
 *    layout with the wrong phase is rejected without being written;
 *  - the production ROCm serial GEMV (M=1), grouped small-M verifier GEMV and
 *    dense prefill GEMM, captured in a graph and replayed onto poisoned
 *    outputs, produce bit-identical FP32 results from the two representations.
 * Together these establish the representation-change contract on ROCm:
 * file-density metadata with unchanged arithmetic.
 */
#include <gtest/gtest.h>

#include "backends/GPUDeviceProcessLifetime.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "backends/rocm/ROCmRuntimeStartup.h"
#include "kernels/rocm/gemm/ROCmGroupedVerifierLaunch.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedBlockMetadata.h"
#include "kernels/rocm/repack/VnniRepackKernels.h"
#include "tensors/IQQuantTables.h"
#include "utils/MPISessionOwner.h"
#include "loaders/gpu_pipeline/RepackFormat.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "tensors/NativeQuantisedScalePlane.h"
#include "tensors/NativeQuantisedSuperblockMetadata.h"
#include "utils/TestTensorFactory.h"

#include <hip/hip_runtime.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C"
{
    bool rocmInitIQGridTables(int, const void *, const void *, const void *, const void *, const void *,
                              const void *);
    bool rocmInitIQGridTables_gemm(int, const void *, const void *, const void *, const void *, const void *,
                                   const void *);
    void rocmGemv_native_quantised_set_decode_equivalent_m1_config(int enabled);
    bool rocmGemv_native_quantised_fp32_with_policy(const int8_t *, const uint8_t *,
                                                    ::llaminar2::rocm_native_metadata::MetadataPlanes, float *,
                                                    const float *, float *, int, int, uint8_t, uint8_t, int, void *,
                                                    const float *);
    bool rocmGemm_native_quantised_fp32_with_policy(const int8_t *, const uint8_t *,
                                                    ::llaminar2::rocm_native_metadata::MetadataPlanes, float *,
                                                    const float *, const float *, int, int, int, uint8_t, uint8_t,
                                                    int, void *);
}

namespace llaminar2::test
{
    namespace
    {
        namespace sb = native_quantised_superblock;
        using rocm_native_metadata::MetadataPlanes;

        /** @brief Throw on a HIP runtime failure so the test reports it precisely. */
        void checked(hipError_t status)
        {
            if (status != hipSuccess)
                throw std::runtime_error(hipGetErrorString(status));
        }

        /** @brief Zero-initialized device bytes released at scope exit. */
        struct DeviceBytes
        {
            explicit DeviceBytes(std::size_t bytes) : size(bytes ? bytes : 1)
            {
                checked(hipMalloc(&data, size));
                checked(hipMemset(data, 0, size));
            }
            ~DeviceBytes() { (void)hipFree(data); }
            DeviceBytes(const DeviceBytes &) = delete;
            DeviceBytes &operator=(const DeviceBytes &) = delete;
            template <class T> T *as() const { return static_cast<T *>(data); }
            template <class T> std::vector<T> download() const
            {
                std::vector<T> out(size / sizeof(T));
                checked(hipMemcpy(out.data(), data, out.size() * sizeof(T), hipMemcpyDeviceToHost));
                return out;
            }
            void *data = nullptr;
            std::size_t size;
        };

        /** @brief One codable superblock source format and its random-tensor factory. */
        struct Case
        {
            const char *name;
            std::function<std::unique_ptr<TensorBase>(const std::vector<size_t> &)> make;
        };

        /** @return Every superblock format whose physical codebook is codable. */
        const std::vector<Case> &cases()
        {
            static const std::vector<Case> all{
                {"Q4_K", [](const std::vector<size_t> &s) { return TestTensorFactory::createQ4_KRandom(s); }},
                {"Q5_K", [](const std::vector<size_t> &s) { return TestTensorFactory::createQ5_KRandom(s); }},
                {"IQ4_XS", [](const std::vector<size_t> &s) { return TestTensorFactory::createIQ4_XSRandom(s); }},
                {"Q6_K", [](const std::vector<size_t> &s) { return TestTensorFactory::createQ6_KRandom(s); }},
                {"Q3_K", [](const std::vector<size_t> &s) { return TestTensorFactory::createQ3_KRandom(s); }},
                {"Q2_K", [](const std::vector<size_t> &s) { return TestTensorFactory::createQ2_KRandom(s); }},
                {"IQ3_S", [](const std::vector<size_t> &s) { return TestTensorFactory::createIQ3_SRandom(s); }},
                {"IQ3_XXS", [](const std::vector<size_t> &s) { return TestTensorFactory::createIQ3_XXSRandom(s); }},
                {"IQ2_S", [](const std::vector<size_t> &s) { return TestTensorFactory::createIQ2_SRandom(s); }},
                {"IQ2_XS", [](const std::vector<size_t> &s) { return TestTensorFactory::createIQ2_XSRandom(s); }},
                {"IQ2_XXS", [](const std::vector<size_t> &s) { return TestTensorFactory::createIQ2_XXSRandom(s); }},
                {"IQ1_S", [](const std::vector<size_t> &s) { return TestTensorFactory::createIQ1_SRandom(s); }},
                {"IQ1_M", [](const std::vector<size_t> &s) { return TestTensorFactory::createIQ1_MRandom(s); }},
            };
            return all;
        }
    } // namespace

    /** @brief One explicit non-blocking stream on ROCm device 0 per case. */
    class ROCmSuperblockCodedMetadata : public ::testing::TestWithParam<int>
    {
    protected:
        void SetUp() override
        {
            int devices = 0;
            if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
                GTEST_SKIP() << "No ROCm device: unqualified";
            checked(hipSetDevice(0));
            // Every IQ decoder reads its translation unit's constant grid tables.
            for (auto initialize : {rocmInitIQGridTables, rocmInitIQGridTables_gemm})
                ASSERT_TRUE(initialize(0, iq3s_grid, iq3xxs_grid, iq2s_grid, iq2xs_grid, iq2xxs_grid, iq1s_grid));
            checked(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
        }
        void TearDown() override
        {
            if (stream_)
            {
                (void)hipStreamSynchronize(stream_);
                (void)hipStreamDestroy(stream_);
            }
        }
        hipStream_t stream_ = nullptr;
    };

    TEST_P(ROCmSuperblockCodedMetadata, CodedRepackReproducesEveryLegacyWordAtFileDensity)
    {
        const auto &c = cases().at(static_cast<std::size_t>(GetParam()));
        SCOPED_TRACE(c.name);
        constexpr int N = 96, source_K = 1024;
        auto tensor = c.make({N, source_K});
        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr);
        const auto *source = unpackable->vnniFormatInfo();
        ASSERT_NE(source, nullptr);
        const auto format = codebookIdToRepackFormat(source->codebook_id, source->is_superblock);
        ASSERT_TRUE(format);
        const std::uint8_t codebook = canonicalDeviceVnniCodebookId(*source);
        ASSERT_TRUE(nativeQuantisedSuperblockCodable(codebook));

        // Whole K (phase 0) and a slice starting at block 11 (phase 3).
        for (const auto &[first_column, columns] : {std::pair{0, source_K}, std::pair{352, 512}})
        {
            SCOPED_TRACE("first_column=" + std::to_string(first_column));
            const auto selection = NativeQuantisedRepackColumns::slice(source_K, first_column, columns);
            const int K = columns, blocks = K / 32;
            const std::uint32_t phase = static_cast<std::uint32_t>(selection.firstBlock() % 8);
            const std::size_t block_count = static_cast<std::size_t>(N) * blocks;
            const int payload_bytes = repackPayloadBytesPerBlock(*format);

            DeviceBytes raw(tensor->size_bytes());
            checked(hipMemcpy(raw.data, tensor->raw_data(), tensor->size_bytes(), hipMemcpyHostToDevice));
            DeviceBytes legacy_payload(block_count * payload_bytes), coded_payload(block_count * payload_bytes);
            DeviceBytes legacy_scales(block_count * 2), legacy_mins(block_count * 2), legacy_emins(block_count * 4);
            const auto planes = sb::planeBytes(codebook, N, static_cast<std::uint32_t>(blocks), phase);
            DeviceBytes codes(planes.code), superblocks(planes.superblock);

            const auto layout = nativeQuantisedMetadataLayout(NativeQuantisedMetadataRepresentation::SuperblockCoded,
                                                              phase);
            ASSERT_TRUE(launchVnniRepack(*format, raw.data, legacy_payload.as<uint8_t>(),
                                         legacy_scales.as<uint16_t>(), legacy_mins.as<uint16_t>(),
                                         legacy_emins.as<uint32_t>(), N, K, N, 0, 0, payload_bytes, selection,
                                         std::uint8_t{0}, stream_));
            const bool in_payload = sb::superblockCodeInPayload(codebook);
            // File density: the code plane holds exactly the source code bits
            // plus the fixed read padding; the superblock plane holds d(/dmin)
            // once per selected source superblock and row.
            ASSERT_EQ(planes.code, in_payload ? 0u
                                              : (block_count * sb::superblockCodeBits(codebook) + 7u) / 8u +
                                                    sb::kSuperblockCodePlanePadBytes);
            ASSERT_EQ(planes.superblock, static_cast<std::size_t>(N) *
                                             sb::superblocksPerRow(static_cast<std::uint32_t>(blocks), phase) *
                                             sb::superblockFieldBytes(codebook));
            ASSERT_TRUE(launchVnniRepack(*format, raw.data, coded_payload.as<uint8_t>(),
                                         in_payload ? nullptr : codes.as<uint16_t>(), superblocks.as<uint16_t>(),
                                         nullptr, N, K, N, 0, 0, payload_bytes, selection, layout, stream_));
            // A coded layout with the wrong phase is rejected, never written.
            EXPECT_FALSE(launchVnniRepack(
                *format, raw.data, coded_payload.as<uint8_t>(), codes.as<uint16_t>(), superblocks.as<uint16_t>(),
                nullptr, N, K, N, 0, 0, payload_bytes, selection,
                nativeQuantisedMetadataLayout(NativeQuantisedMetadataRepresentation::SuperblockCoded, (phase + 1) % 8),
                stream_));
            checked(hipStreamSynchronize(stream_));

            const auto legacy_bytes = legacy_payload.download<uint8_t>();
            const auto coded_bytes = coded_payload.download<uint8_t>();
            if (!in_payload)
                ASSERT_EQ(legacy_bytes, coded_bytes);
            else
            {
                // Restoring sign parity (and clearing IQ2_XS's spare qh nibble)
                // recovers every per-block payload byte.
                const int signs = codebook == 12 ? 8 : codebook == 15 ? 4 : codebook == 14 ? 5 : -1;
                for (std::size_t i = 0; i < coded_bytes.size(); ++i)
                {
                    const int byte = static_cast<int>(i % payload_bytes);
                    std::uint8_t value = coded_bytes[i];
                    if (signs >= 0 && byte >= signs && byte < signs + 4)
                        value = sb::restoreSignParity(value);
                    if (codebook == 14 && byte == 4)
                        value &= 0x0Fu;
                    ASSERT_EQ(value, legacy_bytes[i]) << "payload byte " << i;
                }
            }

            // Shared codec at the production block-major addresses equals every legacy word.
            const auto scales = legacy_scales.download<uint16_t>();
            const auto mins = legacy_mins.download<uint16_t>();
            const auto emins = legacy_emins.download<uint32_t>();
            const auto code_bytes = codes.download<uint8_t>();
            const auto sb_bytes = superblocks.download<uint8_t>();
            const bool consumes_secondary = source->is_asymmetric || codebook == 8 || codebook == 9 || codebook == 24 ||
                                            codebook == 11 || codebook == 13 || codebook == 14 || codebook == 17;
            for (int block = 0; block < blocks; ++block)
                for (int row = 0; row < N; ++row)
                {
                    const auto address = sb::blockMajorAddress(block, row, N, phase);
                    std::uint32_t code = 0, word = 0;
                    if (in_payload)
                    {
                        const auto *block_payload = coded_bytes.data() + address.linear * payload_bytes;
                        code = codebook == 12   ? sb::codeFromPayload<12>(block_payload)
                               : codebook == 14 ? sb::codeFromPayload<14>(block_payload)
                               : codebook == 15 ? sb::codeFromPayload<15>(block_payload)
                                                : sb::codeFromPayload<16>(block_payload);
                    }
                    else
                        code = sb::loadCode(codebook, code_bytes.data(), address.linear);
                    std::memcpy(&word, sb_bytes.data() + address.superblock * sb::superblockFieldBytes(codebook),
                                sb::superblockFieldBytes(codebook));
                    const auto words = sb::decodeSuperblockMetadata(codebook, code, word);
                    ASSERT_EQ(words.primary, scales[address.linear]) << "block " << block << " row " << row;
                    if (consumes_secondary)
                        ASSERT_EQ(words.secondary, mins[address.linear]) << "block " << block << " row " << row;
                    if (codebook == 10)
                        ASSERT_EQ(words.emin, emins[address.linear]) << "block " << block << " row " << row;
                }

            // Production kernels: bit-identical FP32 outputs from both representations.
            const MetadataPlanes legacy = rocm_native_metadata::perBlockMetadataPlanes(
                legacy_scales.data, consumes_secondary ? legacy_mins.data : nullptr,
                codebook == 10 ? legacy_emins.data : nullptr);
            const MetadataPlanes coded = rocm_native_metadata::planesFromLayout(
                in_payload ? coded_payload.data : codes.data, superblocks.data, nullptr, layout);
            ASSERT_TRUE(rocm_native_metadata::metadataPlanesAdmitted(codebook, coded));
            constexpr int max_rows = 33;
            std::mt19937 rng(1234u + static_cast<unsigned>(GetParam()));
            std::vector<int8_t> activation(static_cast<std::size_t>(max_rows) * K);
            std::vector<float> block_scales(static_cast<std::size_t>(max_rows) * blocks);
            std::vector<int32_t> block_sums(static_cast<std::size_t>(max_rows) * blocks, 0);
            std::vector<float> row_scales(max_rows, 1.0f);
            for (auto &v : activation) v = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
            for (auto &v : block_scales) v = 0.001f + static_cast<float>(rng() % 1000) * 1e-5f;
            for (int r = 0; r < max_rows; ++r)
                for (int b = 0; b < blocks; ++b)
                    for (int i = 0; i < 32; ++i)
                        block_sums[static_cast<std::size_t>(r) * blocks + b] +=
                            activation[static_cast<std::size_t>(r) * K + b * 32 + i];
            DeviceBytes a(activation.size()), sa(block_scales.size() * 4), sums(block_sums.size() * 4),
                sr(row_scales.size() * 4);
            checked(hipMemcpy(a.data, activation.data(), activation.size(), hipMemcpyHostToDevice));
            checked(hipMemcpy(sa.data, block_scales.data(), sa.size, hipMemcpyHostToDevice));
            checked(hipMemcpy(sums.data, block_sums.data(), sums.size, hipMemcpyHostToDevice));
            checked(hipMemcpy(sr.data, row_scales.data(), sr.size, hipMemcpyHostToDevice));
            // IQ4_XS's physical 24 executes IQ4_NL's certified arithmetic policy.
            const std::uint8_t policy = codebook == 24 ? std::uint8_t{4} : codebook;
            DeviceBytes partials(static_cast<std::size_t>(128) * max_rows * N * sizeof(float));
            const auto run = [&](const MetadataPlanes &meta, int rows) {
                const auto *weights = meta.coded() ? coded_payload.as<uint8_t>() : legacy_payload.as<uint8_t>();
                DeviceBytes out(static_cast<std::size_t>(rows) * N * sizeof(float));
                // Production executes NativeQuantised projections inside captured
                // graphs; record and replay exactly that path.
                checked(hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal));
                bool ok = false;
                if (rows == 1)
                    ok = rocmGemv_native_quantised_fp32_with_policy(a.as<int8_t>(), weights, meta, out.as<float>(),
                                                                    sr.as<float>(), partials.as<float>(), N, K,
                                                                    codebook, policy, 0, stream_, sa.as<float>());
                else if (rows <= 8)
                {
                    // Runtime-M decode runs only as the grouped verifier inside the
                    // decode-equivalent M1 dispatch scope.
                    rocmGemv_native_quantised_set_decode_equivalent_m1_config(1);
                    ok = rocmGemv_native_quantised_small_m_fp32_with_sums_policy(
                        a.as<int8_t>(), weights, meta, out.as<float>(), sa.as<float>(), sums.as<int32_t>(),
                        partials.as<float>(), rows, N, K, codebook, policy, 0, stream_, nullptr);
                    rocmGemv_native_quantised_set_decode_equivalent_m1_config(0);
                }
                else
                    ok = rocmGemm_native_quantised_fp32_with_policy(a.as<int8_t>(), weights, meta, out.as<float>(),
                                                                    sr.as<float>(), sa.as<float>(), rows, N, K,
                                                                    codebook, policy, 0, stream_);
                hipGraph_t graph = nullptr;
                checked(hipStreamEndCapture(stream_, &graph));
                EXPECT_TRUE(ok) << "rows " << rows;
                hipGraphExec_t exec = nullptr;
                checked(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                checked(hipGraphDestroy(graph));
                // Retained replay onto a poisoned output: a replay that skips
                // publication cannot pass on stale bytes.
                std::vector<uint32_t> first;
                for (int replay = 0; replay < 3; ++replay)
                {
                    checked(hipMemsetAsync(out.data, 0x5a, out.size, stream_));
                    checked(hipGraphLaunch(exec, stream_));
                    checked(hipStreamSynchronize(stream_));
                    auto bits = out.download<uint32_t>();
                    if (replay == 0)
                        first = std::move(bits);
                    else
                        EXPECT_EQ(bits, first) << "replay " << replay << " differs";
                }
                checked(hipGraphExecDestroy(exec));
                return first;
            };
            for (const int rows : {1, 4, max_rows})
            {
                SCOPED_TRACE("rows=" + std::to_string(rows));
                const auto expected = run(legacy, rows);
                const auto actual = run(coded, rows);
                ASSERT_EQ(expected.size(), actual.size());
                EXPECT_EQ(expected, actual) << "FP32 output bits differ between representations";
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(CodableFormats, ROCmSuperblockCodedMetadata, ::testing::Range(0, 13),
                             [](const ::testing::TestParamInfo<int> &info) {
                                 return std::string(cases().at(static_cast<std::size_t>(info.param)).name);
                             });
} // namespace llaminar2::test

/** @brief Canonical ROCm integration process: MPI session, runtime startup, hardware metadata. */
int main(int argc, char **argv)
{
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv))
        return *terminal;
    ::testing::InitGoogleTest(&argc, argv);
    if (::testing::GTEST_FLAG(list_tests))
        return RUN_ALL_TESTS();
    llaminar2::GPUDeviceProcessLifetime native_lifetime;
    llaminar2::requireROCmRuntimeStartup();
    llaminar2::initializeROCmHardwareMetadata();
    return RUN_ALL_TESTS();
}
