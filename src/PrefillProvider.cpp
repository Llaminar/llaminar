/**
 * @file PrefillProvider.cpp
 * @brief Implementation of PrefillProvider base class
 * @author David Sanftenberg
 */

#include "PrefillProvider.h"
#include "OpenblasPrefillProvider.h"
#include "CosmaPrefillProvider.h"
#include "PipelineSnapshotManager.h"
#include "Logger.h"
#include "utils/DebugEnv.h"
#include <cstdlib>

namespace llaminar
{
    void PrefillProvider::captureSnapshot(
        PipelineStage stage,
        int layer_index,
        const float *data,
        int seq_len,
        int feature_dim)
    {
#ifdef NDEBUG
        // Release build: compiled out completely
        (void)stage;
        (void)layer_index;
        (void)data;
        (void)seq_len;
        (void)feature_dim;
#else
        // Debug build: delegate to PipelineSnapshotManager
        // Only rank 0 typically captures to avoid duplication in MPI runs
        // Check if snapshot capture is enabled (can be disabled for benchmarks)
        if (mpi_ctx_.rank == 0 && debugEnv().attention.capture_enabled)
        {
            PipelineSnapshotManager::instance().capture(
                stage,
                layer_index,
                data,
                seq_len,
                feature_dim,
                "llaminar"); // Use "llaminar" as source for parity tests
        }
#endif
    }

    bool PrefillProvider::isSnapshotEnabled() const
    {
#ifdef NDEBUG
        // Release build: always disabled
        return false;
#else
        // Debug build: check PipelineSnapshotManager
        return PipelineSnapshotManager::instance().isEnabled();
#endif
    }

    void PrefillProvider::incrementSnapshotCounter(PrefillMetrics &metrics)
    {
#ifndef NDEBUG
        if (isSnapshotEnabled())
        {
            metrics.snapshots_captured++;
        }
#else
        (void)metrics; // Suppress unused warning
#endif
    }

    // === PrefillProviderFactory Implementation ===

    std::unique_ptr<PrefillProvider> PrefillProviderFactory::create(
        const ModelConfig &config,
        const MPIContext &mpi_ctx,
        int seq_len)
    {
        const auto &env = debugEnv();

        // Check if COSMA is disabled globally
        if (env.adaptive.disable_cosma)
        {
            if (mpi_ctx.rank == 0)
            {
                LOG_INFO("PrefillProviderFactory: COSMA disabled by ADAPTIVE_DISABLE_COSMA, using OpenBLAS");
            }
            return std::make_unique<OpenBLASPrefillProvider>(config, mpi_ctx);
        }

        // Check for forced COSMA execution
        if (env.cosma.force_direct || env.cosma.force_replicated ||
            env.cosma.force_replicated_diag || env.cosma.force_distributed_act)
        {
            if (mpi_ctx.rank == 0)
            {
                LOG_INFO("PrefillProviderFactory: COSMA forced by environment, using COSMA provider");
            }
            return std::make_unique<COSMAPrefillProvider>(config, mpi_ctx);
        }

        // Sequence length-based decision
        const int cosma_threshold = env.cosma.prefill_threshold;

        if (seq_len >= cosma_threshold)
        {
            // Large sequences: Use COSMA for distributed computation
            if (mpi_ctx.rank == 0)
            {
                LOG_INFO("PrefillProviderFactory: seq_len=" << seq_len
                                                            << " >= threshold=" << cosma_threshold
                                                            << ", using COSMA provider");
            }
            return std::make_unique<COSMAPrefillProvider>(config, mpi_ctx);
        }
        else
        {
            // Small sequences: Use OpenBLAS for lower overhead
            if (mpi_ctx.rank == 0)
            {
                LOG_DEBUG("PrefillProviderFactory: seq_len=" << seq_len
                                                             << " < threshold=" << cosma_threshold
                                                             << ", using OpenBLAS provider");
            }
            return std::make_unique<OpenBLASPrefillProvider>(config, mpi_ctx);
        }
    }

    std::unique_ptr<PrefillProvider> PrefillProviderFactory::createByName(
        const std::string &provider_name,
        const ModelConfig &config,
        const MPIContext &mpi_ctx)
    {
        if (provider_name == "openblas")
        {
            return std::make_unique<OpenBLASPrefillProvider>(config, mpi_ctx);
        }
        else if (provider_name == "cosma")
        {
            return std::make_unique<COSMAPrefillProvider>(config, mpi_ctx);
        }
        else if (provider_name == "gpu")
        {
            LOG_ERROR("PrefillProviderFactory: GPU provider not yet implemented");
            return nullptr;
        }
        else
        {
            LOG_ERROR("PrefillProviderFactory: Unknown provider name '" << provider_name << "'");
            return nullptr;
        }
    }

} // namespace llaminar
