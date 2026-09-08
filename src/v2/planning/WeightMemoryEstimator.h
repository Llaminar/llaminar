#pragma once
#include "backends/DeviceId.h"
#include "config/TensorParallelConfig.h"
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{

    struct ModelMemoryProfile;

    /**
     * @brief Typed device-residency contract used by preflight weight sizing.
     *
     * A conventional graph owns every tensor selected by its PP/TP interval.
     * An ExpertOverlay graph instead owns either the dense/shared model plus a
     * precise subset of routed experts (the continuation participant), or only
     * a precise routed-expert subset (an auxiliary participant).  Encoding that
     * distinction here keeps memory admission aligned with the immutable
     * expert owner map instead of applying an after-the-fact byte discount.
     */
    class DeviceWeightResidency
    {
    public:
        /** @brief Kind of model-weight authority held by one device. */
        enum class Kind
        {
            FullModel,
            ContinuationWithSelectedRoutedExperts,
            SelectedRoutedExpertsOnly,
        };

        /** @brief Construct the ordinary full-model residency contract. */
        DeviceWeightResidency() = default;

        /**
         * @brief Construct continuation residency with exact routed counts.
         * @param model_expert_count Number of routed experts in every layer.
         * @param selected_by_layer Number resident on this device per layer.
         * @throws std::invalid_argument when counts cannot describe a subset.
         */
        static DeviceWeightResidency continuationWithSelectedRoutedExperts(
            int model_expert_count,
            std::vector<int> selected_by_layer);

        /**
         * @brief Construct expert-only residency with exact routed counts.
         * @param model_expert_count Number of routed experts in every layer.
         * @param selected_by_layer Number resident on this device per layer.
         * @throws std::invalid_argument when counts cannot describe a subset.
         */
        static DeviceWeightResidency selectedRoutedExpertsOnly(
            int model_expert_count,
            std::vector<int> selected_by_layer);

        /** @return Residency kind. */
        [[nodiscard]] Kind kind() const noexcept { return kind_; }
        /** @return Whether non-routed dense/shared/global weights are resident. */
        [[nodiscard]] bool includesNonRoutedWeights() const noexcept;
        /** @return Whether routed tensors are filtered through an owner map. */
        [[nodiscard]] bool selectsRoutedExperts() const noexcept;
        /** @return Model-wide routed expert count used as the slice denominator. */
        [[nodiscard]] int modelExpertCount() const noexcept { return model_expert_count_; }
        /**
         * @brief Return the selected routed-expert count for one model layer.
         * @throws std::out_of_range when the owner map omits the layer.
         */
        [[nodiscard]] int selectedRoutedExpertsForLayer(int layer) const;
        /** @return Number of layer entries carried by the immutable owner map. */
        [[nodiscard]] size_t layerCount() const noexcept { return selected_by_layer_.size(); }

    private:
        DeviceWeightResidency(
            Kind kind,
            int model_expert_count,
            std::vector<int> selected_by_layer);

        Kind kind_ = Kind::FullModel;
        int model_expert_count_ = 0;
        std::vector<int> selected_by_layer_;
    };

    struct WeightEstimate
    {
        size_t native_bytes = 0; // As stored in GGUF
        size_t device_bytes = 0; // After device-specific packing/repacking
        /** Prepared embedding bytes owned by this exact TP view. */
        size_t prepared_embedding_bytes = 0;
        /** LM-head bytes owned by this exact TP view, tied or explicit. */
        size_t lm_head_bytes = 0;
        /** Subset of @ref lm_head_bytes synthesized from a tied embedding. */
        size_t tied_lm_head_bytes = 0;
    };

    class WeightMemoryEstimator
    {
    public:
        /**
         * @brief Estimate prepared weight memory for one physical participant.
         *
         * Tensor-parallel policy is resolved from the model's registered
         * `WeightShardingConfig`; this estimator deliberately owns no tensor-
         * name classifier. Rank-local TP callers pass the exact assignment
         * used by weight loading so GQA replication and uneven ranges are
         * priced from the same typed authority. Uniform cross-rank TP may omit
         * it after topology validation has proved an equal split.
         *
         * @param profile Parsed model geometry and complete tensor inventory.
         * @param device Physical participant receiving the prepared weights.
         * @param shard_index Zero-based participant index in the TP group.
         * @param total_shards Number of participants in the TP group.
         * @param first_layer First included pipeline layer.
         * @param last_layer Last included pipeline layer, or -1 for all.
         * @param residency Exact full/continuation/expert-only residency view.
         * @param tensor_parallel_assignment Exact rank-local TP assignment when
         *        the production topology has one.
         * @return Native-source and prepared-device byte inventory.
         */
        static WeightEstimate estimate(
            const ModelMemoryProfile &profile,
            DeviceId device,
            int shard_index = 0,
            int total_shards = 1,
            int first_layer = 0,
            int last_layer = -1, // -1 = all layers
            const DeviceWeightResidency &residency = {},
            const std::optional<DeviceShardingAssignment>
                &tensor_parallel_assignment = std::nullopt
        );

        /// Bytes per weight element for native (GGUF on-disk) format.
        static float getNativeBytesPerWeight(const std::string &quant_type);

        /// Bytes per weight element after CUDA/ROCm repacking (Q8_0 → int8 packed).
        static float getCUDAPackedBytesPerWeight(size_t K);

        /// Bytes per weight element after GPU packing for the tensor's native quantization format.
        static float getGPUPackedBytesPerWeight(const std::string &quant_type, size_t K);

        /// Bytes per weight element after CPU VNNI packing.
        static float getCPUPackedBytesPerWeight(const std::string &quant_type);

    };

} // namespace llaminar2
