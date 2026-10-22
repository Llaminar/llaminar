/**
 * @file NativeVNNIShapeManifestProjection.cpp
 * @brief Device-free projection of the actual shared C++ trainer shape loader.
 *
 * The companion Python regression compares every resolved shape field with its
 * canonical loader. This executable links no inference core or accelerator
 * runtime and neither invents matrix lists nor initializes model/backend state.
 * Both implementations read the same CMake-owned JSON and release catalog.
 */
#include "../../performance/kernels/native_vnni_dispatch/NativeVNNIShapeManifest.h"

#include <exception>
#include <iostream>
#include <optional>
#include <string>

namespace
{
    /**
     * @brief Retain an absent split as JSON null rather than inventing ownership.
     * @param value The original loader's optional split assignment.
     * @return The unchanged split string, or JSON null.
     */
    nlohmann::json partition(const std::optional<std::string> &value)
    {
        return value ? nlohmann::json(*value) : nlohmann::json(nullptr);
    }
}

/**
 * @brief Emit the immutable original loader result or fail with its diagnostic.
 * @return Zero only after all original shape records have been projected.
 */
int main()
{
    try
    {
        nlohmann::json result = nlohmann::json::array();
        for (const auto &shape :
             llaminar2::test::native_vnni_dispatch::nativeVnniShapeManifest())
        {
            result.push_back({
                {"name", shape.name},
                {"n", shape.N},
                {"k", shape.K},
                {"model_family", shape.model_family},
                {"role", shape.role},
                {"exact_overlay", shape.exact_overlay},
                {"aspect_bucket", shape.aspect_bucket},
                {"fast_partition", partition(shape.fast_partition)},
                {"verifier_partition", partition(shape.verifier_partition)},
                {"prefill_partition", partition(shape.prefill_partition)}});
        }
        std::cout << result.dump() << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
