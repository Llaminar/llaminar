/**
 * @file ParityArtifactDirectory.h
 * @brief Rank-owned parity directory reset followed by original metadata publication.
 *
 * This narrow setup operation owns no model, reference, numerical result or
 * MPI protocol. The fixture broadcasts its returned status before failing so
 * a publisher error cannot strand peers at the existing readiness barrier.
 */
#pragma once
#include <cstdint>
#include <exception>
#include <filesystem>
#include <string>
#include <string_view>
#include <stdexcept>
#include <utility>

namespace llaminar2::test::parity
{
/**
 * @brief Encode the shared source/numerical artifact identity without truncation or hashes.
 * @param test_name Complete logical identity; reserved filesystem characters retain the existing underscore sanitization.
 * @return The unchanged short component or bounded240-byte chunks under the reserved long-identities namespace.
 * @throws std::invalid_argument For empty, dot, parent traversal or embedded-null identities.
 * @details The reserved namespace is itself encoded, preventing collision with a short identity.
 * Chunk components use a c prefix and the final artifacts leaf has a distinct name;
 * one complete identity cannot become another identity's ancestor. Retiring its
 * terminal directory therefore preserves prefix-neighbor evidence.
 */
inline std::filesystem::path artifactIdentityPath(std::string test_name)
{
    if (test_name.empty() || test_name == "." || test_name == ".." || test_name.find('\0') != std::string::npos)
        throw std::invalid_argument("Parity artifact identity cannot be empty");
    for (char &character : test_name)
    {
        if (character == '/' || character == '\\' || character == ':' ||
            character == '*' || character == '?' || character == '\"' ||
            character == '<' || character == '>' || character == '|')
        {
            character = '_';
        }
    }
    constexpr std::string_view long_namespace = "long-identities";
    if (test_name.size() <= 255 && test_name != long_namespace)
        return test_name;
    std::filesystem::path result(long_namespace);
    constexpr std::size_t chunk_bytes = 240;
    for (std::size_t offset = 0; offset < test_name.size(); offset += chunk_bytes)
        result /= "c" + test_name.substr(offset, chunk_bytes);
    return result / "artifacts";
}

/** @brief Local setup result consumed by the fixture's existing readiness consensus. */
struct ParityArtifactDirectoryPreparation
{
    std::int32_t ready = 1;
    std::string error;
};
/**
 * @brief Reset this cell's generated directory, then publish original setup metadata once.
 * @param directory Original per-cell output directory; never a reference/model directory.
 * @param artifact_owner Whether this rank is the fixture's original artifact authority.
 * @param publish Original validated metadata publisher; called only by that authority.
 * @return Ready status or the actual filesystem/publication diagnostic for consensus.
 *
 * Peers perform no filesystem access or publication. A failed reset never calls
 * the publisher; a failed publisher never reports ready. The caller must share
 * this result before asserting failure or entering any model admission path.
 */
template<class Publish>
ParityArtifactDirectoryPreparation prepareParityArtifactDirectory(
    const std::filesystem::path &directory, bool artifact_owner, Publish &&publish)
{
    if (!artifact_owner) return {};
    try
    {
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        std::forward<Publish>(publish)();
        return {};
    }
    catch (const std::exception &error)
    {
        return {.ready = 0, .error = error.what()};
    }
}
} // namespace llaminar2::test::parity
