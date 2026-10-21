/**
 * @file Q8ResidualFixtureExport.h
 * @brief Explicit host-only original-format fixture export for the existing
 * projection benchmark.
 *
 * Canonical synthetic source creators and FP32 input helpers are shared with
 * correctness tests. No prepared matrix, runtime kernel, GPU, model file or
 * quantization conversion is involved. A fresh directory owns bounded
 * artifacts; the complete manifest is published last. These are component
 * geometry inputs, never original QFN model weights, reference logits or
 * qualification evidence.
 */
#pragma once
#include "kernels/common/Q8ResidualGroupedGeometry.h"
#include "utils/Q8ResidualAllFormatOracle.h"
#include "utils/Sha256.h"
#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>

namespace llaminar2::test::q8_all_formats {
/** @brief Recognize only an explicit option, never an arbitrary option value.
 */
inline bool q8ResidualFixtureExportRequested(int argc, char **argv) {
  for (int i = 1; i < argc; i += 2)
    if (std::string_view(argv[i]) == "--export-fixtures")
      return true;
  return false;
}
/** @brief One bounded original-source export request, with no backend
 * selection. */
struct Q8ResidualFixtureExportOptions {
  std::filesystem::path output;
  int n = 0, k = 0;
};
/** @brief Reject ambiguous geometry/options before constructing any tensor or
 * directory. */
inline Q8ResidualFixtureExportOptions
q8ResidualFixtureExportOptions(int argc, char **argv) {
  Q8ResidualFixtureExportOptions o;
  std::set<std::string> seen;
  if (argc != 7)
    throw std::invalid_argument(
        "Export requires exactly --export-fixtures DIR --n {65|10240} --k 512");
  const auto integer = [](const std::string &text) {
    std::size_t used = 0;
    const int value = std::stoi(text, &used);
    if (used != text.size())
      throw std::invalid_argument(
          "Fixture export integer contains trailing characters");
    return value;
  };
  for (int i = 1; i < argc; i += 2) {
    const std::string key = argv[i], value = argv[i + 1];
    if (!seen.insert(key).second)
      throw std::invalid_argument("Duplicate fixture export option");
    if (key == "--export-fixtures")
      o.output = value;
    else if (key == "--n")
      o.n = integer(value);
    else if (key == "--k")
      o.k = integer(value);
    else
      throw std::invalid_argument("Unknown fixture export option");
  }
  if (o.output.empty() || (o.n != 65 && o.n != 10240) || o.k != 512)
    throw std::invalid_argument("Fixture export supports only N65 or "
                                "production-width N10240 at legal K512");
  return o;
}
/**
 * @brief Write one newly owned bounded file and observe its exact small-file
 * identity.
 * @param root Fresh directory owned by this invocation.
 * @param name Canonical safe filename, derived only from the source catalog.
 * @param bytes Original native bytes or IEEE FP32 bytes; never re-encoded.
 * @return Passive path/byte/SHA256 metadata after successful close.
 */
inline nlohmann::json
writeQ8ResidualFixtureArtifact(const std::filesystem::path &root,
                               const std::string &name,
                               std::string_view bytes) {
  constexpr std::size_t maximum = 8u * 1024 * 1024;
  if (bytes.empty() || bytes.size() > maximum || name.empty() ||
      std::filesystem::path(name).filename() != name)
    throw std::invalid_argument(
        "Fixture artifact name/extent exceeds its bounded export contract");
  const auto path = root / name;
  if (std::filesystem::exists(path))
    throw std::invalid_argument("Fixture artifact already exists");
  {
    std::ofstream stream(path, std::ios::binary);
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    stream.close();
  }
  std::string error;
  const auto digest = sha256FileHex(path, &error);
  if (!digest || std::filesystem::file_size(path) != bytes.size())
    throw std::runtime_error("Fixture artifact identity failed: " + error);
  return {{"file", name},
          {"path", path.string()},
          {"bytes", bytes.size()},
          {"sha256", *digest}};
}
/**
 * @brief Export the sole canonical inventory without model/backend
 * initialization.
 * @return Zero after publishing the final complete manifest.
 *
 * Each source is destroyed before the next is created. Only the new bounded
 * artifacts are hashed. No model/cache payload is opened. The explicit
 * synthetic HC-role name matches the existing stage fixture; it is not a
 * model-source claim.
 */
inline int exportQ8ResidualFixtures(int argc, char **argv) {
  const auto o = q8ResidualFixtureExportOptions(argc, argv);
  if (sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559 ||
      std::endian::native != std::endian::little)
    throw std::invalid_argument(
        "Fixture export requires IEEE FP32 little-endian host");
  const auto &formats = quantizedVerifierFormats();
  if (formats.empty())
    throw std::invalid_argument(
        "Fixture export requires a nonempty canonical format inventory");
  std::set<std::string> labels;
  for (const auto &format : formats) {
    const std::string label = format.label;
    if (label.empty() ||
        label.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") !=
            std::string::npos ||
        !labels.insert(label).second)
      throw std::invalid_argument(
          "Canonical fixture inventory has unsafe or duplicate format names");
  }
  const auto requested = std::filesystem::absolute(o.output);
  const auto parent = std::filesystem::canonical(requested.parent_path());
  const auto root = parent / requested.filename();
  if (!std::filesystem::is_directory(parent) ||
      !std::filesystem::create_directory(root))
    throw std::invalid_argument(
        "Fixture export requires a fresh directory under an existing parent");
  nlohmann::json result = {
      {"schema", "llaminar.q8-residual-synthetic-fixtures.v1"},
      {"scope",
       "synthetic_raw_original_native_sources_component_geometry_not_model"},
      {"N", o.n},
      {"K", o.k},
      {"source_creator", "llaminar2::test::q8_all_formats::source"},
      {"input_creator", "llaminar2::test::q8_all_formats::input"},
      {"expected_catalog_entries", formats.size()},
      {"formats", nlohmann::json::array()},
      {"benchmark_commands", nlohmann::json::array()},
      {"complete", false}};
  const auto one = input(1, o.k), seven = input(7, o.k);
  const auto asBytes = [](const auto &v) {
    return std::string_view(reinterpret_cast<const char *>(v.data()),
                            v.size() * sizeof(v[0]));
  };
  result["input_one"] =
      writeQ8ResidualFixtureArtifact(root, "input-one.f32", asBytes(one));
  result["input_seven"] =
      writeQ8ResidualFixtureArtifact(root, "input-seven.f32", asBytes(seven));
  std::size_t rawBytes = 0;
  for (const auto &format : formats) {
    auto tensor = source(format, o.n, o.k);
    auto *unpackable =
        tensor ? dynamic_cast<IINT8Unpackable *>(tensor.get()) : nullptr;
    const auto *native = unpackable ? unpackable->vnniFormatInfo() : nullptr;
    if (!tensor || tensor->native_type() != format.tensor_type ||
        !tensor->raw_data() || tensor->rows() != std::size_t(o.n) ||
        tensor->cols() != std::size_t(o.k) || !native ||
        native->codebook_id != format.source_codebook_id ||
        native->is_superblock != format.source_is_superblock)
      throw std::runtime_error("Canonical fixture did not retain its original "
                               "native identity, geometry and bytes");
    auto artifact = writeQ8ResidualFixtureArtifact(
        root, std::string(format.label) + ".native",
        std::string_view(static_cast<const char *>(tensor->raw_data()),
                         tensor->size_bytes()));
    rawBytes += tensor->size_bytes();
    if (rawBytes > 192u * 1024 * 1024)
      throw std::runtime_error(
          "Complete fixture export exceeds its bounded raw byte inventory");
    artifact.update(
        {{"format", format.label},
         {"tensor_type", unsigned(format.tensor_type)},
         {"source_seed", 801u + unsigned(format.tensor_type)},
         {"N", o.n},
         {"K", o.k},
         {"source_codebook_id", format.source_codebook_id},
         {"source_is_superblock", format.source_is_superblock},
         {"expected_execution_codebook_id",
          format.device_execution_codebook_id},
         {"native_bytes_policy", "unchanged_raw_output_of_the_same_"
                                 "strengthened_canonical_test_creator"}});
    result["formats"].push_back(artifact);
    for (const auto backend :
         {Q8ResidualGroupedBackend::CUDA, Q8ResidualGroupedBackend::ROCm})
      for (const int rows : {1, 9, 15, 64, 65, 512, 2048}) {
        const auto geometry = q8ResidualGroupedGeometry(
            backend, rows, o.n, format.device_execution_codebook_id);
        const bool grouped =
            geometry.strategy == Q8ResidualGroupedStrategy::CooperativeMatrix;
        const std::string backendName =
            backend == Q8ResidualGroupedBackend::CUDA ? "cuda" : "rocm";
        const auto &inputArtifact =
            result[rows == 1 ? "input_one" : "input_seven"];
        std::vector<std::string> args = {
            "--backend",
            backendName,
            "--m",
            std::to_string(rows),
            "--n",
            std::to_string(o.n),
            "--k",
            std::to_string(o.k),
            "--live",
            std::to_string(rows),
            "--phase",
            rows == 1 ? "serial" : "verifier",
            "--implementation",
            grouped ? "grouped" : "vector",
            "--input-mode",
            rows == 1 ? "exact" : "tile",
            "--weights",
            artifact["path"].get<std::string>(),
            "--input",
            inputArtifact["path"].get<std::string>(),
            "--source-name",
            "blk.2.hc_ffn_up.weight",
            "--source-format",
            format.label,
            "--iterations",
            "32",
            "--batches",
            "6",
            "--warmup",
            "3"};
        result["benchmark_commands"].push_back(
            {{"backend", backendName},
             {"format", format.label},
             {"M", rows},
             {"scope", rows == 1
                           ? "synthetic_original_source_M1_component"
                           : "synthetic_seven_row_tiled_geometry_component"},
             {"expected_projection_threads", geometry.threads},
             {"expected_projection_blocks", geometry.blocks},
             {"args", args}});
      }
  }
  result["raw_source_bytes"] = rawBytes;
  result["complete"] = true;
  const auto text = result.dump(2) + "\n";
  writeQ8ResidualFixtureArtifact(root, "manifest.pending.json", text);
  std::filesystem::rename(root / "manifest.pending.json",
                          root / "manifest.json");
  std::cout << "SYNTHETIC_FIXTURE_EXPORT " << (root / "manifest.json").string()
            << '\n';
  return 0;
}
} // namespace llaminar2::test::q8_all_formats
