/**
 * @file AccuracyPanelHistory.h
 * @brief Immutable frozen-history inputs for the canonical full-stage parity fixture.
 *
 * The existing fixture retains model, placement, precision, graph and numerical
 * policy ownership. This handoff changes only request inputs and the matching
 * CPU/FP32 reference directory. It authenticates small metadata, never scans
 * model weights, and cannot select a different runner or disable checkpoints.
 */
#pragma once

#include "utils/Sha256.h"
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::test::parity
{
/** @brief Keep independent native state oracles outside frozen teacher observations. */
enum class AccuracyObservationPurpose
{
    ReferenceHistory, ///< One original prefill/decode distribution in the authored panel.
    NativeStateOracle, ///< Prefix/reset diagnostic with a separate native oracle.
};

/** @brief One driver-owned request and its independently generated reference. */
class AccuracyPanelHistory final
{
    nlohmann::json value_; ///< Original metadata retained until fixture retirement.
    std::string digest_; ///< Digest of the exact immutable descriptor bytes.

    /**
     * @brief Read bounded metadata without touching model or tensor payloads.
     * @param path Original regular descriptor or reference metadata file.
     * @return Complete original metadata bytes.
     * @throws std::invalid_argument For missing/oversized input; std::runtime_error on read failure.
     */
    static std::string read(const std::filesystem::path &path)
    {
        if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) > 1024 * 1024)
            throw std::invalid_argument("Accuracy panel metadata is missing or oversized");
        std::ifstream file(path, std::ios::binary);
        std::string bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        if (file.bad()) throw std::runtime_error("Cannot read accuracy panel metadata");
        return bytes;
    }

public:
    /**
     * @brief Bind a complete driver handoff to this exact original model/cell.
     * @param cell Actual GoogleTest cell name, never a user-selected replacement.
     * @param model Canonical GGUF path before the fixture's normal tmpfs mapping.
     * @return No history outside explicit panel execution.
     * @throws std::invalid_argument For any partial or mismatched handoff.
     */
    static std::optional<AccuracyPanelHistory> fromDriver(const std::string &cell, const std::string &model)
    {
        const char *path = std::getenv("LLAMINAR_ACCURACY_PANEL_HISTORY");
        const char *digest = std::getenv("LLAMINAR_ACCURACY_PANEL_SHA256");
        if (!path && !digest) return std::nullopt;
        if (!path || !digest) throw std::invalid_argument("Incomplete accuracy panel handoff");
        AccuracyPanelHistory history;
        const auto bytes = read(path);
        history.digest_ = digest;
        if (sha256BytesHex(bytes) != std::optional<std::string>(history.digest_))
            throw std::invalid_argument("Accuracy panel descriptor changed");
        history.value_ = nlohmann::json::parse(bytes);
        const auto &value = history.value_;
        if (value.at("schema") != "llaminar.accuracy-panel-history.v1" ||
            value.at("canonical_cell") != cell ||
            std::filesystem::path(value.at("model").get<std::string>()).filename() != std::filesystem::path(model).filename() ||
            value.at("reference_scope") != "full_stages" ||
            value.at("reference_arithmetic") != "cpu_fp32" ||
            value.at("reference_execution") != "incremental_cached")
            throw std::invalid_argument("Accuracy panel changed original cell, model or reference policy");
        const auto &v = value.at("vocabulary");
        const auto &d = value.at("decode_rows");
        if (!v.is_number_integer() || v < 5 || v > std::numeric_limits<int>::max() ||
            !d.is_number_integer() || d < 1 || d > 256)
            throw std::invalid_argument("Accuracy panel has invalid vocabulary or decode geometry");
        const auto vocabulary = v.get<int>();
        const auto depth = d.get<int>();
        for (const char *field : {"prompt_ids", "continuation_ids"})
            for (const auto &token : value.at(field))
                if (!token.is_number_integer() || token < 0 || token >= vocabulary)
                    throw std::invalid_argument("Accuracy panel token is outside the original vocabulary");
        const auto prompt = value.at("prompt_ids").get<std::vector<int>>();
        const auto teachers = value.at("continuation_ids").get<std::vector<int>>();
        if (vocabulary < 5 || depth <= 0 || depth > 256 || prompt.empty() ||
            teachers.size() != static_cast<std::size_t>(depth + 1))
            throw std::invalid_argument("Accuracy panel requires full vocabulary and D+1 frozen teachers");
        for (const auto *tokens : {&prompt, &teachers})
            for (const int token : *tokens)
                if (token < 0 || token >= vocabulary)
                    throw std::invalid_argument("Accuracy panel token is outside the original vocabulary");
        const auto metadata = std::filesystem::path(value.at("reference_directory").get<std::string>()) / "metadata.txt";
        if (sha256BytesHex(read(metadata)) != std::optional<std::string>(value.at("metadata_sha256").get<std::string>()))
            throw std::invalid_argument("Accuracy panel reference metadata changed");
        return history;
    }

    /**
     * @brief Apply request data only, retaining all topology and checkpoint policies.
     * @param config Existing canonical parity configuration.
     * @throws std::invalid_argument For a mixed replay mode.
     */
    template<class Config> void apply(Config &config) const
    {
        if (std::getenv("LLAMINAR_PARITY_HISTORY_REPLAY_FILE"))
            throw std::invalid_argument("Accuracy panels cannot mix independent history handoffs");
        config.prompt = value_.at("prompt").get<std::string>();
        config.token_ids = value_.at("prompt_ids").get<std::vector<int>>();
        config.decode_steps = value_.at("decode_rows").get<int>();
        config.snapshot_dir = value_.at("reference_directory").get<std::string>();
        if (config.final_logit_mean_kl) config.final_logit_mean_kl->decode_steps = config.decode_steps;
    }

    /**
     * @brief Reject a reference pack whose token timeline differs from the handoff.
     * @param prompt Exact original metadata prompt IDs.
     * @param teachers Exact original D+1 continuation IDs.
     * @throws std::invalid_argument When either frozen sequence differs.
     */
    void requireTokens(const std::vector<int> &prompt, const std::vector<int> &teachers) const
    {
        if (prompt != value_.at("prompt_ids").get<std::vector<int>>() ||
            teachers != value_.at("continuation_ids").get<std::vector<int>>())
            throw std::invalid_argument("Accuracy panel reference changed its frozen token history");
    }

    /**
     * @brief Admit only declared teacher rows, preserving auxiliary native checks separately.
     * @param offset Minus one for final prefill, otherwise the consumed decode offset.
     * @param purpose Typed producer intent; a native state oracle is never a scored prompt.
     * @return Whether this observation belongs to the frozen numerical history.
     * @throws std::invalid_argument For an unknown purpose or undeclared teacher position.
     */
    [[nodiscard]] bool observes(int offset, AccuracyObservationPurpose purpose) const
    {
        switch (purpose)
        {
        case AccuracyObservationPurpose::NativeStateOracle: return false;
        case AccuracyObservationPurpose::ReferenceHistory:
            if (offset < -1 || offset >= value_.at("decode_rows").get<int>())
                throw std::invalid_argument("Accuracy observation is outside the frozen teacher history");
            return true;
        }
        throw std::invalid_argument("Unknown accuracy observation purpose");
    }

    /**
     * @brief Publish the binding only after the fixture creates its fresh artifact directory.
     * @param directory Existing artifact directory owned by this exact fixture.
     * @param config Canonical request configuration after the history override.
     * @throws std::runtime_error When the complete binding cannot be written and closed.
     */
    template<class Config> void publish(const std::filesystem::path &directory, const Config &config) const
    {
        auto binding = value_;
        binding["descriptor_sha256"] = digest_;
        binding["native_limits"] = {
            {"prefill_maximum", config.final_logit_mean_kl ? config.final_logit_mean_kl->prefill_maximum : config.prefillKLThreshold()},
            {"decode_maximum", config.final_logit_mean_kl ? config.final_logit_mean_kl->decode_maximum : config.kl_threshold}};
        std::ofstream file(directory / "accuracy_panel_binding.json");
        file << binding.dump(2) << '\n';
        file.close();
        if (!file) throw std::runtime_error("Cannot publish accuracy panel binding");
    }
};
} // namespace llaminar2::test::parity
