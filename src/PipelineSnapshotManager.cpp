/**
 * @file PipelineSnapshotManager.cpp
 * @brief Implementation of debug-only snapshot capture
 * @author David Sanftenberg
 */

#include "PipelineSnapshotManager.h"
#include "Logger.h"
#include "utils/DebugEnv.h"

#ifndef NDEBUG
// Debug build: full implementation
#include "../tests/ParityTestFramework.h"
#include <cstring>
#endif

namespace llaminar
{

#ifdef NDEBUG
    // ==================== RELEASE BUILD (NO-OP) ====================

    PipelineSnapshotManager &PipelineSnapshotManager::instance()
    {
        static PipelineSnapshotManager inst;
        return inst;
    }

    PipelineSnapshotManager::PipelineSnapshotManager() {}

    void PipelineSnapshotManager::capture(
        PipelineStage /*stage*/,
        int /*layer_index*/,
        const float * /*data*/,
        int /*seq_len*/,
        int /*feature_dim*/,
        const std::string & /*source*/)
    {
        // No-op in release builds
    }

    bool PipelineSnapshotManager::isEnabled() const
    {
        return false;
    }

    void PipelineSnapshotManager::setEnabled(bool /*enabled*/)
    {
        // No-op in release builds
    }

    size_t PipelineSnapshotManager::exportToNPZ(const std::string & /*filepath*/) const
    {
        return 0;
    }

    void PipelineSnapshotManager::clear()
    {
        // No-op in release builds
    }

    size_t PipelineSnapshotManager::count() const
    {
        return 0;
    }

    SnapshotScope::SnapshotScope(bool /*enable*/) {}
    SnapshotScope::~SnapshotScope() {}

#else
    // ==================== DEBUG BUILD (FULL IMPLEMENTATION) ====================

    struct PipelineSnapshotManager::Impl
    {
        bool enabled = false;
        bool env_checked = false;

        void checkEnvironment()
        {
            if (!env_checked)
            {
                const auto &env = debugEnv();
                // Check if capture is enabled via environment
                const char *capture_env = std::getenv("LLAMINAR_PARITY_CAPTURE");
                if (capture_env && std::atoi(capture_env) > 0)
                {
                    enabled = true;
                    LOG_INFO("Snapshot capture enabled via LLAMINAR_PARITY_CAPTURE");
                }
                env_checked = true;
            }
        }
    };

    PipelineSnapshotManager &PipelineSnapshotManager::instance()
    {
        static PipelineSnapshotManager inst;
        return inst;
    }

    PipelineSnapshotManager::PipelineSnapshotManager()
        : impl_(std::make_unique<Impl>())
    {
        impl_->checkEnvironment();
    }

    void PipelineSnapshotManager::capture(
        PipelineStage stage,
        int layer_index,
        const float *data,
        int seq_len,
        int feature_dim,
        const std::string &source)
    {
        if (!impl_->enabled)
        {
            LOG_DEBUG("[SNAPSHOT_DEBUG] PipelineSnapshotManager::capture() called but impl_->enabled=false, skipping");
            return;
        }

        // Use the parity framework's snapshot hook (static methods)
        // CRITICAL: Pass the source parameter to differentiate batch vs sequential
        if (parity::LlaminarSnapshotHook::is_enabled())
        {
            LOG_DEBUG("[SNAPSHOT_DEBUG] Calling LlaminarSnapshotHook::capture for stage=" << static_cast<int>(stage) << " layer=" << layer_index);
            parity::LlaminarSnapshotHook::capture(stage, layer_index, data, seq_len, feature_dim, source);
        }
        else
        {
            LOG_DEBUG("[SNAPSHOT_DEBUG] LlaminarSnapshotHook::is_enabled() returned false, skipping");
        }
    }

    bool PipelineSnapshotManager::isEnabled() const
    {
        return impl_->enabled;
    }

    void PipelineSnapshotManager::setEnabled(bool enabled)
    {
        impl_->enabled = enabled;

        // Also control the underlying parity framework hook
        parity::LlaminarSnapshotHook::set_enabled(enabled);

        if (enabled)
        {
            LOG_INFO("Snapshot capture manually enabled");
        }
        else
        {
            LOG_INFO("Snapshot capture manually disabled");
        }
    }

    size_t PipelineSnapshotManager::exportToNPZ(const std::string &filepath) const
    {
        if (!impl_->enabled)
        {
            return 0;
        }

        // TODO: Implement NPZ export
        // The SnapshotRegistry is only available in test binaries,
        // so we can't access it from the core library.
        // For now, snapshots are automatically captured via LlaminarSnapshotHook
        // and can be accessed by test code or comparison tools.
        LOG_INFO("Snapshot export requested to " << filepath);
        LOG_INFO("Direct NPZ export from core library not yet implemented");
        LOG_INFO("Use test framework's LayerByLayerComparator or Python tools instead");

        return 0; // Placeholder - actual count not available from core library
    }

    void PipelineSnapshotManager::clear()
    {
        if (impl_->enabled)
        {
            // Snapshot clearing is handled by test framework
            // LlaminarSnapshotHook doesn't provide a clear() method
            LOG_DEBUG("Clear requested - snapshots managed by test framework");
        }
    }

    size_t PipelineSnapshotManager::count() const
    {
        if (!impl_->enabled)
        {
            return 0;
        }

        // Count not available from core library
        // SnapshotRegistry is only accessible in test binaries
        return 0; // Placeholder
    }

    // ==================== RAII Scope Guard ====================

    SnapshotScope::SnapshotScope(bool enable)
        : previous_state_(PipelineSnapshotManager::instance().isEnabled())
    {
        PipelineSnapshotManager::instance().setEnabled(enable);
    }

    SnapshotScope::~SnapshotScope()
    {
        PipelineSnapshotManager::instance().setEnabled(previous_state_);
    }

#endif // NDEBUG

} // namespace llaminar
