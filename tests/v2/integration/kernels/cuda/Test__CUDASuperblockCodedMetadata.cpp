/**
 * @file Test__CUDASuperblockCodedMetadata.cpp
 * @brief Superblock-coded GPU metadata is byte-identical to per-block planes on CUDA.
 *
 * The same raw GGUF superblocks are repacked twice by the production CUDA
 * repack launcher: once into the per-block planes (metadata layout zero) and
 * once into the superblock-coded planes (per-block codes plus per-superblock
 * d/dmin). The test proves, for every codable source format and for a column
 * slice whose first block sits mid-superblock (nonzero phase):
 *  - both repacks emit identical payload bytes, except the parity bits of
 *    IQ3_XXS/IQ2_XXS/IQ2_XS sign masks (and IQ2_XS's spare qh nibble), which
 *    the coded payload uses for its scale code; restoring parity recovers the
 *    per-block payload exactly;
 *  - the host codec reconstructs every legacy per-block word from the coded
 *    planes at the production addresses (block-major, grouped by phase);
 *  - production GEMV (M=1), grouped small-M GEMV and dense prefill produce
 *    byte-identical FP32 outputs from the two representations.
 * Together these establish the representation-change contract: file-density
 * metadata with unchanged arithmetic.
 */
#include <gtest/gtest.h>

#include "backends/cuda/CUDAHardwareMetadata.h"
#include "backends/cuda/CUDAGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/NativeQuantisedScaleLoad.h"
#include "kernels/cuda/gemm/CUDADeviceWorkspace.h"
#include "kernels/common/DeviceRowRange.h"
#include "kernels/cuda/repack/CUDAVnniRepackKernels.h"
#include "loaders/gpu_pipeline/RepackFormat.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "tensors/NativeQuantisedScalePlane.h"
#include "tensors/NativeQuantisedSuperblockMetadata.h"
#include "utils/TestTensorFactory.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C"
{
    bool cudaNativeQuantisedInitIQGridTables_tuned();
    void cudaNativeQuantisedGemvTuned_setDecodeEquivalentM1Config(int);
    int cudaNativeQuantisedGemvTuned_getDecodeEquivalentM1Config();
    bool cudaNativeQuantisedGemvTuned_fp32_withPolicy(
        const int8_t *, const uint8_t *, ::llaminar2::native_quantised_scale::NativeQuantisedMetadataPlanes,
        float *, const float *, int, int, float, float, const float *, const float *, uint8_t, uint8_t, int, void *,
        CUDAGemvContext *, CUDARowMajorWeights **);
    bool cudaNativeQuantisedGemvTuned_small_m_fp32_withPolicy(
        const int8_t *, const uint8_t *, ::llaminar2::native_quantised_scale::NativeQuantisedMetadataPlanes,
        float *, const float *, int, int, int, float, float, const float *, const float *, uint8_t, uint8_t, int,
        void *, CUDAGemvContext *, CUDARowMajorWeights **, const llaminar2::DeviceRowRange *);
    bool cudaNativeQuantisedPrefill_fp32_withPolicy(
        const std::int8_t *, const std::uint8_t *, ::llaminar2::native_quantised_scale::NativeQuantisedMetadataPlanes,
        float *, const float *, const std::int32_t *, int, int, int, float, float, const float *, const float *,
        std::uint8_t, std::uint8_t, int, void *, CUDAPrefillContext *);
}

namespace llaminar2::test
{
    namespace
    {
        namespace sb = native_quantised_superblock;
        using native_quantised_scale::NativeQuantisedMetadataPlanes;

        void checked(cudaError_t status)
        {
            if (status != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(status));
        }

        /** @brief Zero-initialized device bytes released at scope exit. */
        struct DeviceBytes
        {
            explicit DeviceBytes(std::size_t bytes) : size(bytes ? bytes : 1)
            {
                checked(cudaMalloc(&data, size));
                checked(cudaMemset(data, 0, size));
            }
            ~DeviceBytes() { (void)cudaFree(data); }
            DeviceBytes(const DeviceBytes &) = delete;
            DeviceBytes &operator=(const DeviceBytes &) = delete;
            template <class T> T *as() const { return static_cast<T *>(data); }
            template <class T> std::vector<T> download() const
            {
                std::vector<T> out(size / sizeof(T));
                checked(cudaMemcpy(out.data(), data, out.size() * sizeof(T), cudaMemcpyDeviceToHost));
                return out;
            }
            void *data = nullptr;
            std::size_t size;
        };

        struct Case
        {
            const char *name;
            std::function<std::unique_ptr<TensorBase>(const std::vector<size_t> &)> make;
        };

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

    class CUDASuperblockCodedMetadata : public ::testing::TestWithParam<int>
    {
    protected:
        void SetUp() override
        {
            int devices = 0;
            if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
                GTEST_SKIP() << "No CUDA device: unqualified";
            checked(cudaSetDevice(0));
            // Dispatch reads admitted physical facts (SM count, limits).
            llaminar2::initializeCUDAHardwareMetadata();
            ASSERT_TRUE(cudaNativeQuantisedInitIQGridTables_tuned());
            checked(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        }
        void TearDown() override
        {
            if (stream_)
            {
                (void)cudaStreamSynchronize(stream_);
                (void)cudaStreamDestroy(stream_);
            }
        }
        cudaStream_t stream_ = nullptr;
    };

    TEST_P(CUDASuperblockCodedMetadata, CodedPlanesReproduceLegacyWordsAndOutputsByteExactly)
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
            checked(cudaMemcpy(raw.data, tensor->raw_data(), tensor->size_bytes(), cudaMemcpyHostToDevice));
            DeviceBytes legacy_payload(block_count * payload_bytes), coded_payload(block_count * payload_bytes);
            DeviceBytes legacy_scales(block_count * 2), legacy_mins(block_count * 2), legacy_emins(block_count * 4);
            const auto planes = sb::planeBytes(codebook, N, static_cast<std::uint32_t>(blocks), phase);
            DeviceBytes codes(planes.code), superblocks(planes.superblock);

            const auto layout = nativeQuantisedMetadataLayout(NativeQuantisedMetadataRepresentation::SuperblockCoded,
                                                              phase);
            ASSERT_TRUE(launchVnniRepackCUDA(*format, raw.data, legacy_payload.as<uint8_t>(),
                                             legacy_scales.as<uint16_t>(), legacy_mins.as<uint16_t>(),
                                             legacy_emins.as<uint32_t>(), N, K, N, 0, 0, payload_bytes, selection,
                                             std::uint8_t{0}, stream_));
            const bool in_payload = sb::superblockCodeInPayload(codebook);
            ASSERT_EQ(planes.code, in_payload ? 0u
                                              : (block_count * sb::superblockCodeBits(codebook) + 7u) / 8u +
                                                    sb::kSuperblockCodePlanePadBytes);
            ASSERT_TRUE(launchVnniRepackCUDA(*format, raw.data, coded_payload.as<uint8_t>(),
                                             in_payload ? nullptr : codes.as<uint16_t>(),
                                             superblocks.as<uint16_t>(), nullptr, N, K, N, 0, 0, payload_bytes,
                                             selection, layout, stream_));
            // A coded layout with the wrong phase is rejected, never written.
            EXPECT_FALSE(launchVnniRepackCUDA(
                *format, raw.data, coded_payload.as<uint8_t>(), codes.as<uint16_t>(), superblocks.as<uint16_t>(),
                nullptr, N, K, N, 0, 0, payload_bytes, selection,
                nativeQuantisedMetadataLayout(NativeQuantisedMetadataRepresentation::SuperblockCoded, (phase + 1) % 8),
                stream_));
            checked(cudaStreamSynchronize(stream_));

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

            // Host codec at the production block-major addresses equals every legacy word.
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

            // Production kernels: byte-identical outputs from both representations.
            const NativeQuantisedMetadataPlanes legacy{
                legacy_scales.as<uint16_t>(), consumes_secondary ? legacy_mins.data : nullptr,
                codebook == 10 ? legacy_emins.as<uint32_t>() : nullptr, 0,
                NativeQuantisedMetadataRepresentation::PerBlockPlanes};
            const NativeQuantisedMetadataPlanes coded{in_payload ? coded_payload.as<uint16_t>() : codes.as<uint16_t>(),
                                                      superblocks.data, nullptr, phase,
                                                      NativeQuantisedMetadataRepresentation::SuperblockCoded};
            constexpr int max_rows = 33;
            std::mt19937 rng(1234u + GetParam());
            std::vector<int8_t> activation(static_cast<std::size_t>(max_rows) * K);
            std::vector<float> activation_scales(static_cast<std::size_t>(max_rows) * blocks);
            std::vector<int32_t> activation_sums(static_cast<std::size_t>(max_rows) * blocks, 0);
            for (auto &v : activation) v = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
            for (auto &v : activation_scales) v = 0.001f + static_cast<float>(rng() % 1000) * 1e-5f;
            for (int r = 0; r < max_rows; ++r)
                for (int b = 0; b < blocks; ++b)
                    for (int i = 0; i < 32; ++i)
                        activation_sums[static_cast<std::size_t>(b) * max_rows + r] +=
                            activation[static_cast<std::size_t>(r) * K + b * 32 + i];
            DeviceBytes a(activation.size()), sa(activation_scales.size() * 4), sums(activation_sums.size() * 4);
            checked(cudaMemcpy(a.data, activation.data(), activation.size(), cudaMemcpyHostToDevice));
            checked(cudaMemcpy(sa.data, activation_scales.data(), sa.size, cudaMemcpyHostToDevice));
            checked(cudaMemcpy(sums.data, activation_sums.data(), sums.size, cudaMemcpyHostToDevice));

            auto *gemv = cudaGemvContext_create(0);
            ASSERT_NE(gemv, nullptr);
            DeviceBytes partials(static_cast<std::size_t>(64) * max_rows * N * sizeof(float));
            cudaGemvContext_bindWorkspace(gemv, partials.as<float>(), partials.size);
            auto *prefill = cudaPrefillContext_create(0);
            ASSERT_NE(prefill, nullptr);
            std::size_t prefill_bytes = 0;
            int prefill_partitions = 1;
            // IQ4_XS's physical 24 executes IQ4_NL's certified arithmetic policy.
            const std::uint8_t policy = codebook == 24 ? std::uint8_t{4} : codebook;
            ASSERT_TRUE(cudaNativeQuantisedPrefill_getWorkspacePlanWithPolicy(codebook, policy, max_rows, N, K, 0,
                                                                              &prefill_bytes, &prefill_partitions));
            DeviceBytes prefill_partials(prefill_bytes);
            cudaPrefillContext_bindWorkspace(prefill, prefill_partials.as<float>(), prefill_bytes);

            const auto run = [&](const NativeQuantisedMetadataPlanes &meta, int rows) {
                const auto *weights = meta.coded() ? coded_payload.as<uint8_t>() : legacy_payload.as<uint8_t>();
                DeviceBytes out(static_cast<std::size_t>(rows) * N * sizeof(float));
                bool ok = false;
                // Production executes every NativeQuantised projection inside a
                // captured graph, whose dispatch policy differs from eager
                // launches; record and replay exactly that path.
                llaminar2::CUDAGraphCapture graph(stream_, 0);
                llaminar2::ScopedBackendGraphCapture recording(graph, "superblock coded metadata parity");
                EXPECT_TRUE(recording.begin());
                if (rows == 1)
                    ok = cudaNativeQuantisedGemvTuned_fp32_withPolicy(a.as<int8_t>(), weights, meta,
                                                           out.as<float>(), sa.as<float>(), N, K, 1.0f, 0.0f, nullptr,
                                                           nullptr, codebook, policy, 0, stream_, gemv, nullptr);
                else if (rows <= 8)
                {
                    // Production launches runtime-M decode only as the grouped
                    // verifier, inside the decode-equivalent M1 dispatch scope
                    // (ScopedNativeQuantisedDecodeEquivalentDispatch); outside
                    // it no generated small-M policy exists.
                    const int previous = cudaNativeQuantisedGemvTuned_getDecodeEquivalentM1Config();
                    cudaNativeQuantisedGemvTuned_setDecodeEquivalentM1Config(1);
                    ok = cudaNativeQuantisedGemvTuned_small_m_fp32_withPolicy(
                        a.as<int8_t>(), weights, meta, out.as<float>(), sa.as<float>(), rows, N,
                        K, 1.0f, 0.0f, nullptr, nullptr, codebook, policy, 0, stream_, gemv, nullptr, nullptr);
                    cudaNativeQuantisedGemvTuned_setDecodeEquivalentM1Config(previous);
                }
                else
                    ok = cudaNativeQuantisedPrefill_fp32_withPolicy(a.as<int8_t>(), weights, meta,
                                                         out.as<float>(), sa.as<float>(), sums.as<int32_t>(), rows,
                                                         N, K, 1.0f, 0.0f, nullptr, nullptr, codebook, policy, 0,
                                                         stream_, prefill);
                recording.finish();
                EXPECT_TRUE(ok) << "rows " << rows;
                EXPECT_TRUE(graph.instantiate());
                EXPECT_TRUE(graph.launch());
                checked(cudaStreamSynchronize(stream_));
                return out.download<uint32_t>();
            };
            for (const int rows : {1, 4, max_rows})
            {
                SCOPED_TRACE("rows=" + std::to_string(rows));
                const auto expected = run(legacy, rows);
                const auto actual = run(coded, rows);
                ASSERT_EQ(expected.size(), actual.size());
                EXPECT_EQ(expected, actual) << "FP32 output bits differ between representations";
            }
            cudaPrefillContext_destroy(prefill);
            cudaGemvContext_destroy(gemv);
        }
    }

    INSTANTIATE_TEST_SUITE_P(CodableFormats, CUDASuperblockCodedMetadata,
                             ::testing::Range(0, 13),
                             [](const ::testing::TestParamInfo<int> &info) {
                                 return std::string(cases().at(static_cast<std::size_t>(info.param)).name);
                             });
} // namespace llaminar2::test
