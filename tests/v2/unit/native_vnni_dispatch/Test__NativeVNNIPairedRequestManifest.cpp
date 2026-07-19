/**
 * @file Test__NativeVNNIPairedRequestManifest.cpp
 * @brief CPU-only unit tests for the CUDA paired-request JSON boundary.
 *
 * These tests parse temporary metadata only; they never initialize CUDA or
 * execute GPU work. They lock down the fail-closed contract used before the
 * performance trainer starts an expensive paired tournament transaction.
 */

#include <gtest/gtest.h>

#include "../../performance/kernels/native_vnni_dispatch/NativeVNNIPairedRequestManifest.h"

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

namespace
{
    using llaminar2::test::native_vnni_dispatch::
        loadNativeVnniPairedRequestManifest;

    /** RAII owner for one unique temporary JSON path. */
    class TemporaryManifest
    {
    public:
        explicit TemporaryManifest(const nlohmann::json &payload)
        {
            static size_t sequence = 0;
            path_ = std::filesystem::temp_directory_path() /
                    ("llaminar-native-vnni-paired-request-" +
                     std::to_string(++sequence) + ".json");
            std::ofstream output(path_);
            if (!output)
                throw std::runtime_error("unable to create temporary manifest");
            output << payload.dump(2) << '\n';
        }

        ~TemporaryManifest()
        {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
        }

        TemporaryManifest(const TemporaryManifest &) = delete;
        TemporaryManifest &operator=(const TemporaryManifest &) = delete;

        [[nodiscard]] std::string path() const
        {
            return path_.string();
        }

    private:
        std::filesystem::path path_;
    };

    /** Build one complete planner-emitted request transaction. */
    nlohmann::json validManifest()
    {
        return {
            {"schema_version", "native-vnni-cuda-paired-request-v1"},
            {"backend", "cuda"},
            {"development_corpus_digest", "sha256:" + std::string(64, '1')},
            {"paired_evidence_digest", "sha256:" + std::string(64, '2')},
            {"max_regret", 0.03},
            {"request_count", 1},
            {"requests", nlohmann::json::array({
                {
                    {"request_id", "cuda-pair-unit-0001"},
                    {"source_format", "Q4_0"},
                    {"source_codebook", 0},
                    {"execution_codebook", 0},
                    {"shape", "UnitShape"},
                    {"shape_group_id", "cuda-decode:UnitShape:n1152:k3584"},
                    {"execution_mode", "graph_captured"},
                    {"m", 1},
                    {"n", 1152},
                    {"k", 3584},
                    {"selected_candidate_id",
                     "cuda.nvnni.decode.fast_m1.kpar.tn64.cpt2.kb7"},
                    {"exact_candidate_id",
                     "cuda.nvnni.decode.fast_m1.direct.tn64.cpt1"},
                    // Equality is actionable because installation requires
                    // p95 regret to be strictly lower than the budget.
                    {"observed_cv_regret", 0.03},
                    {"reason", "missing_direct_tournament_edge"},
                },
            })},
        };
    }
} // namespace

TEST(Test__NativeVNNIPairedRequestManifest, ParsesConcreteForceableRequest)
{
    TemporaryManifest file(validManifest());
    const auto manifest = loadNativeVnniPairedRequestManifest(file.path());
    ASSERT_EQ(manifest.requests.size(), 1u);
    EXPECT_EQ(manifest.requests.front().execution_mode, "graph_captured");
    EXPECT_EQ(
        manifest.requests.front().selected_candidate_id,
        "cuda.nvnni.decode.fast_m1.kpar.tn64.cpt2.kb7");
}

TEST(Test__NativeVNNIPairedRequestManifest, ParsesArchitectureQualifiedCPURequest)
{
    nlohmann::json payload = validManifest();
    payload["schema_version"] = "native-vnni-paired-request-v2";
    payload["backend"] = "cpu";
    payload["requests"][0]["architecture_class"] =
        "x86_64|build=AVX512|runtime=AVX2|threads=28";
    payload["requests"][0]["execution_mode"] = "eager";
    payload["requests"][0]["selected_candidate_id"] =
        "cpu.nvnni.decode.n_chunk_grid.nbc2";
    payload["requests"][0]["exact_candidate_id"] =
        "cpu.nvnni.decode.n_chunk_grid.nbc1";

    TemporaryManifest file(payload);
    const auto manifest = loadNativeVnniPairedRequestManifest(file.path());
    ASSERT_EQ(manifest.backend, "cpu");
    ASSERT_EQ(manifest.requests.size(), 1u);
    EXPECT_EQ(
        manifest.requests.front().architecture_class,
        "x86_64|build=AVX512|runtime=AVX2|threads=28");
}

TEST(Test__NativeVNNIPairedRequestManifest, RejectsUnresolvedFormulaCandidate)
{
    nlohmann::json payload = validManifest();
    payload["requests"][0]["selected_candidate_id"] =
        "cuda.nvnni.decode.fast_m1.kpar_formula.tn64.cpt2.bpp4";
    TemporaryManifest file(payload);
    EXPECT_THROW(
        (void)loadNativeVnniPairedRequestManifest(file.path()),
        std::runtime_error);
}

TEST(Test__NativeVNNIPairedRequestManifest, RejectsUnknownSchemaField)
{
    nlohmann::json payload = validManifest();
    payload["requests"][0]["silently_ignored_discriminator"] = "unsafe";
    TemporaryManifest file(payload);
    EXPECT_THROW(
        (void)loadNativeVnniPairedRequestManifest(file.path()),
        std::runtime_error);
}

TEST(Test__NativeVNNIPairedRequestManifest, RejectsCountMismatch)
{
    nlohmann::json payload = validManifest();
    payload["request_count"] = 2;
    TemporaryManifest file(payload);
    EXPECT_THROW(
        (void)loadNativeVnniPairedRequestManifest(file.path()),
        std::runtime_error);
}
