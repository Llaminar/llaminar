/**
 * @file parity_hooks.cpp
 * @brief Default no-op implementation of parity hooks (production build)
 * @author David Sanftenberg
 *
 * This provides a default implementation of the parity hooks that does nothing.
 * When tests link with parity_test_framework.cpp, this implementation is replaced
 * with the real snapshot capture logic.
 *
 * This approach ensures:
 * - Production code can call parity hooks without conditional compilation
 * - Zero overhead when not testing (functions inline to empty)
 * - Tests get full functionality by linking parity framework
 */

#include "parity_hooks.h"
#include <cstdlib>

namespace llaminar
{
    namespace parity
    {
        // Static member initialization
        bool LlaminarSnapshotHook::enabled_ = false;

        void LlaminarSnapshotHook::capture(
            PipelineStage stage,
            int layer_index,
            const float *data,
            int seq_len,
            int feature_dim)
        {
            // Default no-op implementation
            // Tests will provide real implementation via parity_test_framework.cpp
            (void)stage;
            (void)layer_index;
            (void)data;
            (void)seq_len;
            (void)feature_dim;
        }

        void LlaminarSnapshotHook::capture(
            const std::string &stage_name,
            int layer_index,
            const float *data,
            int seq_len,
            int feature_dim)
        {
            // Default no-op implementation
            (void)stage_name;
            (void)layer_index;
            (void)data;
            (void)seq_len;
            (void)feature_dim;
        }

        void LlaminarSnapshotHook::set_enabled(bool enabled)
        {
            enabled_ = enabled;
        }

        bool LlaminarSnapshotHook::is_enabled()
        {
            // Check environment variable on first call
            static bool env_checked = false;
            if (!env_checked)
            {
                env_checked = true;
                const char *env = std::getenv("LLAMINAR_PARITY_CAPTURE");
                if (env != nullptr && env[0] != '\0' && env[0] != '0')
                {
                    enabled_ = true;
                }
            }
            return enabled_;
        }

    } // namespace parity
} // namespace llaminar
