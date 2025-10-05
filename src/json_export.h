/**
 * @file json_export.h
 * @brief JSON export utilities for logits and token data
 * @author David Sanftenberg
 */

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

namespace llaminar
{
    /**
     * @brief Structure to hold generation data for JSON export
     */
    struct GenerationData
    {
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> generated_tokens;
        std::vector<std::vector<float>> logits; // One vector per generation step
    };

    /**
     * @brief Export generation data to JSON file
     *
     * @param filepath Path to output JSON file
     * @param data Generation data to export
     * @return true if successful, false otherwise
     *
     * JSON format:
     * {
     *   "prompt_tokens": [1, 2, 3, ...],
     *   "generated_tokens": [42, 57, ...],
     *   "logits": [[...], [...], ...]
     * }
     */
    bool exportToJson(const std::string &filepath, const GenerationData &data);

} // namespace llaminar
