/**
 * @file ProductionParityModelPath.h
 * @brief Identity-bound tmpfs path resolution for production parity children.
 *
 * The aggregate production-parity driver is the sole authority that stages,
 * identity-binds, locks, and publishes real GGUF files. Registered CTest
 * production campaigns are internal children of that driver: allowing one to
 * silently use its source GGUF would make direct CTest execution both slower
 * and materially different from the canonical campaign. These device-free
 * helpers reject that invalid lifecycle before ModelContext can mmap a model.
 */

#pragma once

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace llaminar2::test::parity
{
    /** Environment contract identifying one process-resident campaign child. */
    inline constexpr std::string_view
        kProductionParityProcessCampaignEnvironment =
            "LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN";

    /** Environment contract naming the driver's identity-bound GGUF directory. */
    inline constexpr std::string_view
        kProductionParityModelRamdiskEnvironment =
            "LLAMINAR_PRODUCTION_PARITY_MODEL_RAMDISK";

    /** Environment contract containing the exact campaign-owned GGUF manifest. */
    inline constexpr std::string_view
        kProductionParityDeclaredModelsEnvironment =
            "LLAMINAR_PRODUCTION_PARITY_DECLARED_MODELS";

    /**
     * @return Whether this process is an aggregate production-campaign child.
     *
     * Only the exact generated value `1` enters the process-campaign
     * lifecycle. A focused diagnostic without that contract may still use its
     * explicitly configured model path.
     */
    inline bool productionParityIsProcessCampaign()
    {
        const char *raw = std::getenv(
            kProductionParityProcessCampaignEnvironment.data());
        return raw != nullptr && std::string_view(raw) == "1";
    }

    /**
     * @brief Resolve a declared production GGUF to its authenticated RAM copy.
     *
     * The aggregate campaign stages every selected model before any inference
     * process starts. A test may use either the repository `models` symlink or
     * the canonical source path, so membership is checked by canonical source
     * identity while the collision-checked basename selects the staged file.
     * A focused invocation outside the process-campaign lifecycle has no
     * staging contract and retains its configured path unchanged.
     *
     * A process campaign is different: absence of the staged directory is a
     * fatal lifecycle error. This check deliberately runs before ModelContext
     * construction, preventing even a transient source-GGUF mmap or page-cache
     * read when somebody invokes an internal CTest campaign directly.
     *
     * @param configured_path Model path selected by the concrete parity case.
     * @return The authenticated tmpfs path for a campaign child, otherwise the
     *         focused invocation's configured path.
     * @throws std::runtime_error if a campaign bypasses staging, or if its
     *         manifest/staged publication is missing, stale, or unauthorized.
     */
    inline std::string productionParityResolvedModelPath(
        const std::string &configured_path)
    {
        const char *raw_ramdisk = std::getenv(
            kProductionParityModelRamdiskEnvironment.data());
        if (!raw_ramdisk || !*raw_ramdisk)
        {
            if (productionParityIsProcessCampaign())
            {
                throw std::runtime_error(
                    "production parity process campaign bypassed authenticated "
                    "tmpfs model staging; ProductionCampaign CTest entries are "
                    "internal children and must be launched through "
                    "scripts/ci/run_production_parity_campaigns.py");
            }
            return configured_path;
        }

        const char *raw_manifest = std::getenv(
            kProductionParityDeclaredModelsEnvironment.data());
        if (!raw_manifest || !*raw_manifest)
        {
            throw std::runtime_error(
                "RAM-staged production parity has no declared GGUF manifest");
        }

        std::error_code error;
        const auto ramdisk = std::filesystem::canonical(raw_ramdisk, error);
        if (error || !std::filesystem::is_directory(ramdisk))
        {
            throw std::runtime_error(
                "production parity RAM model directory is unavailable: " +
                std::string(raw_ramdisk));
        }
        const auto configured = std::filesystem::canonical(
            configured_path, error);
        if (error)
        {
            throw std::runtime_error(
                "cannot canonicalize declared production GGUF '" +
                configured_path + "': " + error.message());
        }

        bool declared = false;
        std::istringstream manifest(raw_manifest);
        std::string entry;
        while (std::getline(manifest, entry, '|'))
        {
            if (entry.empty())
                continue;
            error.clear();
            const auto candidate = std::filesystem::canonical(entry, error);
            if (!error &&
                (candidate == configured ||
                 (configured.parent_path() == ramdisk &&
                  candidate.filename() == configured.filename())))
            {
                declared = true;
                break;
            }
        }
        if (!declared)
        {
            throw std::runtime_error(
                "production parity configured an undeclared GGUF: " +
                configured_path);
        }

        const auto staged = ramdisk / configured.filename();
        error.clear();
        if (!std::filesystem::is_regular_file(staged, error) || error)
        {
            throw std::runtime_error(
                "declared production GGUF was not staged in RAM: " +
                staged.string());
        }
        return staged.string();
    }
}
