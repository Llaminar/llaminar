/**
 * @file Test__CPUNativeQuantisedPublicationAdmission.cpp
 * @brief Model-free publication admission without MPI process startup.
 *
 * Threadsafe death tests re-execute this binary for each rejected publication.
 * Keeping this owner separate from model-backed GEMV tests avoids repeating
 * their global MPI initialization in every child. The executable's MPI entry
 * tripwires reject accidental initialization instead of hiding that dependency.
 * Native entrypoints, positive controls and ISA selection remain unchanged.
 */
#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <functional>
#include <string_view>
#include <span>
#include <utility>
#include <vector>

#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "utils/TestTensorFactory.h"

namespace
{
    /** @brief Exit distinctly if model-free admission ever initializes MPI. */
    [[noreturn]] void rejectMPIInitialization()
    {
        std::fputs("unexpected MPI initialization in publication admission\n", stderr);
        std::fflush(stderr);
        std::_Exit(86);
    }
}

/** @brief Reject ordinary MPI startup in this test executable and its children. */
extern "C" int MPI_Init(int *, char ***)
{
    rejectMPIInitialization();
}

/** @brief Reject threaded MPI startup without modifying caller-owned outputs. */
extern "C" int MPI_Init_thread(int *, char ***, int, int *)
{
    rejectMPIInitialization();
}

/** @brief Reject direct access to the profiling interface's ordinary initializer. */
extern "C" int PMPI_Init(int *, char ***)
{
    rejectMPIInitialization();
}

/** @brief Reject direct access to the profiling interface's threaded initializer. */
extern "C" int PMPI_Init_thread(int *, char ***, int, int *)
{
    rejectMPIInitialization();
}

using namespace llaminar2;
using namespace llaminar2::cpu::native_quantised;
using namespace llaminar2::test;

namespace
{
    /** @test Both MPI startup APIs and profiling aliases fail before initialization. */
    TEST(CPUNativeQuantisedPublicationAdmission, RejectsAccidentalMPIStartup)
    {
        GTEST_FLAG_SET(death_test_style, "threadsafe");
        // Resolve the linked process symbols instead of directly calling a
        // local body the optimizer could inline. All four bindings must reach
        // the executable's tripwires, not the MPI DSO's real initializers.
        for (const auto *name : {"MPI_Init", "PMPI_Init"})
        {
            SCOPED_TRACE(name);
            const auto init = reinterpret_cast<int (*)(int *, char ***)>(dlsym(RTLD_DEFAULT, name));
            ASSERT_NE(init, nullptr);
            EXPECT_EXIT((void)init(nullptr, nullptr),
                ::testing::ExitedWithCode(86), "unexpected MPI initialization");
        }
        for (const auto *name : {"MPI_Init_thread", "PMPI_Init_thread"})
        {
            SCOPED_TRACE(name);
            const auto init = reinterpret_cast<int (*)(int *, char ***, int, int *)>(dlsym(RTLD_DEFAULT, name));
            ASSERT_NE(init, nullptr);
            EXPECT_EXIT((void)init(nullptr, nullptr, 0, nullptr),
                ::testing::ExitedWithCode(86), "unexpected MPI initialization");
        }
    }

    /**
     * @test Every CPU NativeQuantised execution entry admits its publication
     * before reading it: each entry executes sealed MXFP4 weights, and dies on
     * stale or never-sealed bytes on the selected runtime ISA (scalar, AVX2,
     * AVX512 or NEON) instead of reading them.
     */
    TEST(CPUNativeQuantisedPublicationAdmission, EveryEntryRejectsUnsealedAndStaleBytes)
    {
        // Threadsafe death children re-execute the binary and this body up to
        // their own check, so each one costs a process start. The checks run
        // before any sealed execution, and each entry is probed once: with
        // never-sealed bytes, which proves the entry consults the admission.
        // Stale-generation rejection is a property of that one admission and
        // is checked once here (and exhaustively by CPUNativeQuantisedMetadataSeal).
        GTEST_FLAG_SET(death_test_style, "threadsafe");
        constexpr int N = 128, K = 256, M = 4;
        const auto weights = TestTensorFactory::createMXFP4Random({(size_t)N, (size_t)K});
        CPUNativeQuantisedGemmKernel kernel(weights.get());
        ASSERT_TRUE(kernel.isValid());
        const auto &sealed = *kernel.exportCPUNativeQuantisedPackedWeights();
        ASSERT_TRUE(sealed.publicationSealed());

        CPUNativeQuantisedPackedWeights stale = sealed; // A current seal carries over to the copy.
        ASSERT_TRUE(stale.publicationSealed());
        (void)stale.native_interleaved.beginRewrite();

        CPUNativeQuantisedPackedWeights never;
        never.N = sealed.N; never.K = sealed.K; never.N_padded = sealed.N_padded;
        never.blocks_per_row = sealed.blocks_per_row; never.payload_bytes = sealed.payload_bytes;
        never.codebook_id = sealed.codebook_id; never.is_superblock = sealed.is_superblock;
        never.encoding = sealed.encoding; never.data_stride = sealed.data_stride;
        never.interleaved_block_stride = sealed.interleaved_block_stride;
        never.native_interleaved.resize(sealed.native_interleaved.size());

        std::vector<float> input(static_cast<size_t>(M) * K, 0.25f), out(static_cast<size_t>(M) * N),
            product(static_cast<size_t>(M) * N);
        std::vector<Q8_1Block> activation(static_cast<size_t>(M) * (K / 32)), q8(activation.size());
        quantize_activations_to_q8_1(input.data(), q8.data(), M, K, K / 32);
        std::vector<float> partial(4 * nativeQuantisedProjectionPartialEnvelopeFloats(sealed, M) + 4096);
        const std::span<float> partials{partial};

        using Entry = std::function<void(const CPUNativeQuantisedPackedWeights &)>;
        const std::vector<std::pair<const char *, Entry>> entries{
            {"gemv_preq", [&](const auto &w) { gemv_native_quantised_preq(w, q8.data(), out.data(), partials); }},
            {"gemv", [&](const auto &w) { gemv_native_quantised(w, input.data(), out.data(), partials, activation); }},
            {"gemm_preq", [&](const auto &w) { gemm_native_quantised_preq(w, q8.data(), out.data(), partials, M, N); }},
            {"gemm_preq_rows", [&](const auto &w) {
                gemm_native_quantised_preq_decode_equivalent_rows(w, q8.data(), out.data(), partials, M, N); }},
            {"gemm", [&](const auto &w) { gemm_native_quantised(w, input.data(), out.data(), partials, M, N, activation); }},
            {"multiply", [&](const auto &w) {
                multiply_native_quantised_with_epilogue(w, input.data(), out.data(), M, N, 1.0f, 0.0f,
                    activation, product, partials); }},
            {"fused_gemv", [&](const auto &w) {
                FusedGemvDesc d{};
                d.packed = &w; d.output = out.data(); d.N = N;
                gemv_native_quantised_fused_preq(q8.data(), &d, partials, 1); }},
            {"fused_multi_input", [&](const auto &w) {
                FusedGemvMultiInputDesc d{};
                d.A_q8 = q8.data(); d.packed = &w; d.output = out.data(); d.N = N;
                gemv_fused_multi_input_preq(&d, partials, 1); }},
            {"fused_verifier_rows", [&](const auto &w) {
                FusedVerifierRowsDesc d{};
                d.packed = &w; d.output = out.data(); d.N = N; d.ldc = N; d.rows = M; d.input = q8.data();
                ASSERT_TRUE(gemm_native_quantised_fused_verifier_rows_preq(q8.data(), &d, partials, 1, M, K / 32)); }},
        };
        EXPECT_DEATH(entries.front().second(stale), "stale generation");
        for (const auto &[name, entry] : entries)
        {
            SCOPED_TRACE(name);
            EXPECT_DEATH(entry(never), "never sealed");
        }
        for (const auto &[name, entry] : entries)
        {
            SCOPED_TRACE(name);
            EXPECT_NO_THROW(entry(sealed));
        }
    }
} // anonymous namespace

/**
 * @brief Run the isolated fixture, with explicit startup negative controls.
 * @param argc Argument count, including GTest's threadsafe re-execution flags.
 * @param argv Process arguments forwarded unchanged to GTest.
 * @return GTest status; forced MPI startup exits distinctly before tests run.
 *
 * The test-only environment value is used by the registered negative-control
 * driver. In ordinary execution neither parent nor death child starts MPI.
 */
int main(int argc, char **argv)
{
    bool death_child = false;
    for (int i = 1; i < argc; ++i)
        death_child |= std::string_view(argv[i]).starts_with("--gtest_internal_run_death_test=");
    if (const char *raw = std::getenv("LLAMINAR_TEST_PUBLICATION_FORCE_MPI"))
    {
        const std::string_view mode(raw);
        if (mode != "parent" && mode != "death-child")
        {
            std::fputs("invalid publication startup negative-control mode\n", stderr);
            return 2;
        }
        if ((mode == "parent" && !death_child) || (mode == "death-child" && death_child))
        {
            const auto init = reinterpret_cast<int (*)(int *, char ***)>(dlsym(RTLD_DEFAULT, "MPI_Init"));
            if (!init)
                return 87;
            (void)init(nullptr, nullptr);
            return 87; // The authenticated tripwire must not return.
        }
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
