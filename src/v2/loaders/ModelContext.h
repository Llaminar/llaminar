/**
 * @file ModelContext.h
 * @brief Model context for pipeline initialization
 *
 * Wraps GGUF model metadata and provides convenient access for pipelines.
 * Eliminates need for pipelines to hardcode architecture parameters.
 * Immutable native tables retain one rank-owned host service or one original
 * source-format bank per actual GPU consumer. Participant setup authenticates
 * one source/PMA generation; graph families borrow Ready owners without reload.
 *
 * @author David Sanftenberg
 *
 * Original-source metadata inspection borrows the already-owned loader directory;
 * it neither initializes weights nor changes prepared ownership or coherence.
 */

#pragma once

#include "ModelLoader.h"
#include "WeightManager.h"
#include "WeightPlacementMap.h"
#include "ModelContextConfig.h"
#include "ImmutableRowTable.h"
#include "HostImmutableRowProvider.h"
#include "ResidentImmutableRowTable.h"
#include "ImmutableLookupNativeOwnership.h"
#include <map>
#include <span>
#include <condition_variable>
#include <exception>
#include <thread>
#include "../interfaces/IModelContext.h"
#include "../backends/DeviceId.h"
#include "../utils/MPIContext.h"
#include "../tensors/Tensors.h"
#include <memory>
#include <mutex>
#include <string>

namespace llaminar2
{

    /**
     * @brief Model context for pipeline initialization
     *
     * Implements IModelContext interface for production use with GGUF files.
     *
     * Contains:
     * - Model file path
     * - Parsed GGUF metadata (architecture, hyperparameters)
     * - ModelLoader for on-demand tensor loading
     * - WeightManager for distributed weight allocation
     *
     * Usage:
     *   auto ctx = ModelContext::create("model.gguf", mpi_ctx);
     *   auto pipeline = PipelineFactory::create(ctx->architecture(), ctx, mpi_ctx, device_idx);
     *   // Pipeline reads hyperparameters from ctx->model()
     *   // Pipeline loads weights via ctx->getWeightForDevice(name, device_idx)
     */
    class ModelContext : public IModelContext
    {
    public:
        /**
         * @brief Create model context from GGUF file
        /**
         * @param model_path Path to GGUF model file
         * @param mpi_ctx MPI context for distributed weight management (nullptr = single rank)
         * @param placement_map Fine-grained weight placement decisions (nullptr = default all to device 0)
         * @param factory Optional TensorFactory for NUMA-aware allocation
         * @param strategy Weight distribution strategy (default: REPLICATED)
         * @param weight_precision Weight loading precision (affects dequantization, default: NATIVE)
         * @return Shared pointer to context, or nullptr on error
         */
        static std::shared_ptr<ModelContext> create(
            const std::string &model_path,
            std::shared_ptr<IMPIContext> mpi_ctx = nullptr,
            std::shared_ptr<WeightPlacementMap> placement_map = nullptr,
            TensorFactory *factory = nullptr,
            WeightDistributionStrategy strategy = WeightDistributionStrategy::REPLICATED,
            WeightPrecision weight_precision = WeightPrecision::NATIVE);

        /**
         * @brief Create model context with unified configuration
         *
         * This is the preferred factory method. Supports all scenarios:
         * - Single device (default config)
         * - TP (config.total_shards > 1)
         * - PP (config.first_layer != 0 or config.last_layer != -1)
         * - Combined TP + PP
         *
         * @param model_path Path to GGUF model file
         * @param config Configuration for weight loading
         * @return Shared pointer to context, or nullptr on error
         */
        static std::shared_ptr<ModelContext> create(
            const std::string &model_path,
            const ModelContextConfig &config);

        /**
         * @brief Create model context for a Pipeline Parallelism stage
         *
         * Creates a ModelContext with LAYER_PARTITIONED strategy that only loads
         * weights for the specified layer range. This reduces memory usage by
         * ~50% for 2-stage PP.
         *
         * Layer range is [first_layer, last_layer) - first inclusive, last exclusive.
         *
         * Special weights:
         * - has_embedding: if true, token embedding is loaded (typically stage 0)
         * - has_lm_head: if true, output norm and LM head are loaded (typically last stage)
         *
         * @param model_path Path to GGUF model file
         * @param first_layer First layer index (inclusive)
         * @param last_layer Last layer index (exclusive)
         * @param has_embedding True if this stage should load embedding
         * @param has_lm_head True if this stage should load output norm and LM head
         * @param mpi_ctx MPI context (optional)
         * @param weight_precision Weight loading precision (default: NATIVE)
         * @return Shared pointer to context, or nullptr on error
         */
        static std::shared_ptr<ModelContext> createForPPStage(
            const std::string &model_path,
            int first_layer,
            int last_layer,
            bool has_embedding,
            bool has_lm_head,
            std::shared_ptr<IMPIContext> mpi_ctx = nullptr,
            WeightPrecision weight_precision = WeightPrecision::NATIVE);

        /**
         * @brief Create test-only model context (doesn't load actual model)
         *
         * For unit tests that need a ModelContext but don't need real model data.
         * Initializes a minimal valid GGUFModel structure to prevent undefined behavior.
         * Most tests should leave with_weight_manager=false and use MockModelContext when
         * they need interface-level weight behavior. Tests that exercise production-only
         * WeightManager facilities, such as ExpertGemmRegistry graph lowering, can opt in
         * to a concrete WeightManager without loading a GGUF file.
         *
         * @param model_path Dummy path (can be anything)
         * @param mpi_ctx Optional MPI context for multi-rank tests (use nullptr for single-rank)
         * @return Shared pointer to context (always succeeds)
         */
        static std::shared_ptr<ModelContext> createForTesting(
            const std::string &model_path = "test.gguf",
            std::shared_ptr<IMPIContext> mpi_ctx = nullptr,
            uint32_t block_count = 1,
            bool with_weight_manager = false);

        // =========================================================================
        // IModelContext Implementation
        // =========================================================================

        /**
         * @brief Get model file path
         */
        const std::string &path() const override { return model_path_; }

        /** @brief Return the complete ordered GGUF shard set loaded by this context. */
        std::vector<std::string> artifactPaths() const override
        {
            const auto paths = loader_.artifactPaths();
            return paths.empty() ? std::vector<std::string>{model_path_} : paths;
        }

        /**
         * @brief Get GGUF model metadata
         */
        const GGUFModel &model() const { return loader_.getModel(); }

        /**
         * @brief Get architecture string
         */
        const std::string &architecture() const override { return loader_.getModel().architecture; }

        /**
         * @brief Get ModelLoader interface for tensor loading
         */
        std::shared_ptr<IModelLoader> loader() override { return loader_interface_; }
        /** @return Original complete loader-owned metadata; never loads or converts a tensor. */
        const GGUFModel *ggufDirectory() const noexcept override { return loader_.ggufDirectory(); }


        /** @brief Return the single admitted rank-owned native row service, if present. */
        std::shared_ptr<HostImmutableRowProvider> hostImmutableRowProvider() const override;
        /** @return Exact immutable declaration shared by preparing/Ready participants; completeness is explicit. */
        std::shared_ptr<const ResolvedImmutableRowTablePlacement> resolvedImmutableRowTablePlacement() const override;


        /** @brief Borrow one original-format whole/range GPU owner; every graph family shares its exact storage. */
        std::shared_ptr<const ResidentImmutableRowTable> residentImmutableRowTable(DeviceId device) const override;
        /** @brief Borrow the source-owned adaptive boundary; original roots and live claims survive the last graph borrow. */
        std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> adaptiveImmutableLookupBoundary(DeviceId device) const override;
        /** @brief Exact live immutable-bank bytes for the existing model-retention owner row. */
        std::size_t residentImmutableRowTableBytes(DeviceId device) const override;

        /**
         * @brief Materialize the sealed rank placement after ordinary weight-load staging drains.
         * @param placement Concrete policy/source extent and exact admitted allocator set.
         * @param authority Existing canonical rank PMA; replacement is forbidden.
         * @param uploads Exact per-GPU setup streams and original released-slot receipts; empty for host storage.
         * @param transfers Sole public allocation/copy/event authority.
         * All upload identities validate before allocation, then this convenience
         * entry point delegates to the same participant preparation primitive.
         */
        void prepareImmutableRowTable(const ResolvedImmutableRowTablePlacement &placement,
            std::shared_ptr<PhysicalMemoryAuthority> authority,
            std::span<const ImmutableRowTableDevicePreparation> uploads,
            const TransferEngine &transfers);

        /**
         * @brief Materialize one actual PLE participant after its ordinary loader drains.
         * @param placement One shared sealed rank placement/source generation.
         * @param authority Same canonical rank PMA retained by every participant.
         * @param consumer Exact actual PLE graph resource from placement.consumers().
         * @param upload Required for a fresh whole/range native writer; absent for host or already-Ready owners.
         * @param transfers Sole native allocation/copy/event authority.
         * @param adaptive_setup Fresh adaptive metadata setup on the exact worker stream; absent for Ready reuse.
         * Distinct GPU writes proceed outside the shared mutex. Duplicate setup
         * callers join the original owner; recursive setup and every failure
         * fail closed. A sibling failure prevents all late Ready publication.
         */
        void prepareImmutableRowTableForConsumer(const ResolvedImmutableRowTablePlacement &placement,
            std::shared_ptr<PhysicalMemoryAuthority> authority, DeviceId consumer,
            const std::optional<ImmutableRowTableDevicePreparation> &upload,
            const TransferEngine &transfers,
            const std::optional<AdaptiveImmutableLookupDevicePreparation> &adaptive_setup=std::nullopt);

        /** @brief Require every exact planned allocator Ready before orchestration publishes inference-ready.
         * @param placement Original sealed placement, compared by complete immutable value.
         * @param authority Same canonical rank PMA, never a recreated certificate.
         * No allocation, source read, synchronization or partial-owner inference occurs.
         */
        void requireImmutableRowTablePlacementComplete(const ResolvedImmutableRowTablePlacement &placement,
            const std::shared_ptr<PhysicalMemoryAuthority> &authority) const;


        /**
         * @brief Get concrete ModelLoader for internal use
         */
        ModelLoader &concreteLoader() { return loader_; }
        const ModelLoader &concreteLoader() const { return loader_; }

        /**
         * @brief Report whether model tensors use file-backed mapped storage.
         *
         * This exposes the effective loader policy to memory planning without
         * allowing callers to mutate it after model metadata has been loaded.
         * GPU upload preflight uses this distinction because mmap payloads are
         * consumed through a bounded pinned staging ring rather than copied
         * wholesale into anonymous host memory.
         *
         * @return true when the underlying loader was configured for mmap.
         */
        [[nodiscard]] bool usesMmap() const noexcept
        {
            return loader_.usesMmap();
        }

        /**
         * @brief Get weight tensor for a specific device (device-isolated instance)
         *
         * Loads from GGUF with appropriate distribution strategy.
         * For multi-device scenarios, returns device-isolated clones.
         *
         * @param name GGUF tensor name (e.g., "token_embd.weight", "blk.0.attn_q.weight")
         * @param device Target device for this tensor instance (default: CPU)
         * @return Device-specific tensor instance, or nullptr on error
         */
        std::shared_ptr<TensorBase> getWeightForDevice(const std::string &name, DeviceId device = DeviceId::cpu()) override
        {
            return weight_manager_->getWeightForDevice(name, device);
        }

        /**
         * @brief Check if a tensor exists in the model
         * @param name GGUF tensor name to check
         * @return true if tensor exists, false otherwise
         */
        bool hasTensor(const std::string &name) const override
        {
            return loader_.hasTensor(name);
        }

        /**
         * @brief Get weight manager interface (for IModelContext)
         */
        std::shared_ptr<IWeightManager> weightManager() override { return weight_manager_; }

        /**
         * @brief Get concrete weight manager for production code
         *
         * Use this when you need WeightManager-specific methods not on the interface.
         */
        std::shared_ptr<WeightManager> concreteWeightManager() { return weight_manager_; }

        // =========================================================================
        // Convenience Hyperparameter Accessors (IModelContext)
        // =========================================================================

        int blockCount() const override
        {
            // For PP stages, return the layer count for this stage (not total model layers)
            if (pp_block_count_override_ >= 0)
            {
                return pp_block_count_override_;
            }
            return static_cast<int>(loader_.blockCount());
        }
        int totalBlockCount() const override
        {
            // Always return the full model layer count (ignoring PP override)
            return static_cast<int>(loader_.blockCount());
        }
        int embeddingLength() const override { return static_cast<int>(loader_.embeddingLength()); }
        int headCount() const override { return static_cast<int>(loader_.headCount()); }
        int headCountKV() const override { return static_cast<int>(loader_.headCountKV()); }
        int vocabSize() const override { return static_cast<int>(loader_.vocabSize()); }
        int contextLength() const override { return static_cast<int>(loader_.contextLength()); }
        int feedForwardLength() const override { return static_cast<int>(loader_.feedForwardLength()); }
        int keyLength() const override { return static_cast<int>(loader_.keyLength()); }

        /**
         * @brief Get weight placement map (Phase 6: Multi-GPU support)
         *
         * Returns the placement map from the weight manager.
         * @return Shared pointer to placement map (may be nullptr if single device)
         */
        std::shared_ptr<WeightPlacementMap> placementMap() const
        {
            return weight_manager_ ? weight_manager_->placementMap() : nullptr;
        }

    private:
        // Private constructor - use create() factory method
    private:
        /**
         * @brief Private constructor (use create() instead)
         */
        ModelContext(const std::string &model_path,
                     std::shared_ptr<IMPIContext> mpi_ctx,
                     std::shared_ptr<WeightPlacementMap> placement_map = nullptr,
                     TensorFactory *factory = nullptr,
                     WeightDistributionStrategy strategy = WeightDistributionStrategy::REPLICATED);

        std::string model_path_;
        std::unique_ptr<TensorFactory> owned_test_factory_; // For createForTesting() only - must be declared BEFORE loader_!
        ModelLoader loader_;
        std::shared_ptr<WeightManager> weight_manager_;
        /** @brief Extract actual immutable source coordinates from the model-owned directory. */
        ImmutableRowTableSource immutableRowTableSource() const;
        /**
         * @brief Materialize one checked GPU/subordinate source through the existing setup state machine.
         * @param placement Sole resolved layout and original request ceilings.
         * @param authority Existing canonical PMA; duplicate authorities reject before publication.
         * @param source Original loader-authenticated file/tensor extent.
         * @param consumer Exact one GPU PLE endpoint.
         * @param setup Ordinary explicit stream, required only for a fresh domain.
         * @throws std::exception On partial preparation; no fallback or second source owner is published.
         */
        void prepareGPUAdaptiveImmutableLookup(const ResolvedImmutableRowTablePlacement &placement,
            std::shared_ptr<PhysicalMemoryAuthority> authority,const ImmutableRowTableSource &source,DeviceId consumer,
            const std::optional<AdaptiveImmutableLookupDevicePreparation> &setup);
        /** @brief Legal setup-only owner states; no inference progress is mirrored here. */
        enum class ImmutableRowPreparationState { Preparing, Ready, Failed };
        /** @brief One bounded setup owner, named by the actual physical allocator. */
        struct ImmutableRowPreparation
        {
            ImmutableRowPreparationState state;
            std::thread::id preparing_thread;
        };
        mutable std::mutex immutable_row_table_mutex_;
        std::condition_variable immutable_row_table_ready_;
        std::map<DeviceId, ImmutableRowPreparation> immutable_row_table_preparations_;
        std::optional<ImmutableRowTableSource> immutable_row_table_source_;
        std::optional<ImmutableRowFileIdentity> immutable_row_table_file_identity_;
        std::exception_ptr immutable_row_table_preparation_failure_;

        std::shared_ptr<HostImmutableRowProvider> host_row_provider_;
        std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> adaptive_lookup_boundary_; ///< One source domain; graph variants share the same original roots.
        std::shared_ptr<const ResolvedImmutableRowTablePlacement> immutable_row_table_placement_;
        std::map<DeviceId, std::shared_ptr<const ResidentImmutableRowTable>> resident_row_tables_;
        std::shared_ptr<PhysicalMemoryAuthority> immutable_row_table_authority_;

        // Interface wrapper for loader (since loader_ is stored by value, not as shared_ptr)
        std::shared_ptr<IModelLoader> loader_interface_;

        // PP stage layer count override (-1 = use loader's blockCount)
        // Set by createForPPStage() to return correct layer count for nested MDO graph building
        int pp_block_count_override_ = -1;
    };

} // namespace llaminar2
