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
#include <utility>

namespace llaminar2::test::parity
{
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
