/**
 * @file SharedPredictorGGUFSource.h
 * @brief Explicit user-selected native GGUF predictor artifact, with no discovery policy.
 *
 * The source is bound during metadata admission. It never changes the target
 * directory, tokenizer rendering authority, or identities of borrowed globals.
 */
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
namespace llaminar2
{
/** @brief Consumed preparation policy; an inactive admitted source still participates in identity. */
enum class SharedPredictorMaterialization : uint8_t { Active, Inactive };
/** @brief One explicit predictor path; absence is represented by std::optional. */
class SharedPredictorGGUFSource final
{
public:
    /**
     * @brief Select an exact artifact rather than an autodiscovery request.
     * @param path User-selected original predictor GGUF path.
     * @throws std::invalid_argument If the selection is empty.
     */
    explicit SharedPredictorGGUFSource(std::string path) : path_(std::move(path))
    {
        if (path_.empty()) throw std::invalid_argument("Shared GGUF predictor requires an explicit nonempty path");
    }
    /** @return Exact selected path without rewriting or resolving another candidate. */
    const std::string &path() const noexcept { return path_; }
    /** @brief Source selection participates in immutable startup identity. */
    bool operator==(const SharedPredictorGGUFSource &) const = default;
private:
    std::string path_;
};
} // namespace llaminar2
