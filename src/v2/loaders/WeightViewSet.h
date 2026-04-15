/**
 * @file WeightViewSet.h
 * @brief Lightweight read-only accessor for a subset of weights from a SharedWeightPool
 *
 * WeightViewSet provides a non-owning, read-only view into a SharedWeightPool's
 * weight tensors for a specific PP stage. Each PP stage receives its own
 * WeightViewSet filtered to its assigned layer range.
 *
 * Key properties:
 * - Does NOT own tensor data (SharedWeightPool retains ownership)
 * - Immutable after construction
 * - Thread-safe (read-only access)
 * - Validates layer range on construction
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#pragma once

#include "WeightManagerConfig.h"
#include "../tensors/TensorClasses.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    /**
     * @brief A single weight view: non-owning reference to a tensor in the pool
     */
    struct WeightView
    {
        std::string name;                   ///< Weight name (e.g., "blk.3.attn_q.weight")
        std::shared_ptr<TensorBase> tensor; ///< Shared pointer to tensor (pool retains primary ownership)
        int layer_index = -1;               ///< Layer index (-1 for non-layer weights like embedding)
        bool is_gemm_weight = false;        ///< Whether this weight needs GEMM repacking
    };

    /**
     * @brief Read-only collection of weight views for a PP stage
     *
     * Created by SharedWeightPool::createViewSet() with a specific layer range.
     * Provides name-based lookup and iteration over assigned weights.
     *
     * Usage:
     * @code
     * // Created by SharedWeightPool for a PP stage
     * auto views = pool.createViewSet(0, 12, true, false);
     *
     * // Look up a specific weight
     * auto* view = views.find("blk.5.attn_q.weight");
     * if (view) {
     *     const TensorBase* tensor = view->tensor.get();
     * }
     *
     * // Iterate all views
     * for (const auto& view : views) {
     *     prepareWeight(view);
     * }
     * @endcode
     */
    class WeightViewSet
    {
    public:
        /**
         * @brief Construct an empty view set with layer range metadata
         */
        WeightViewSet(int first_layer, int last_layer, bool has_embedding, bool has_lm_head)
            : layer_range_{first_layer, last_layer, has_embedding, has_lm_head}
        {
        }

        /**
         * @brief Add a weight view to this set
         *
         * @param view The weight view to add (name must be unique)
         * @throws std::runtime_error if a view with the same name already exists
         */
        void addView(WeightView view)
        {
            const std::string name = view.name;
            auto [it, inserted] = index_.emplace(name, views_.size());
            if (!inserted)
            {
                throw std::runtime_error("WeightViewSet: duplicate weight name: " + name);
            }
            views_.push_back(std::move(view));
        }

        /**
         * @brief Find a weight view by name
         * @return Pointer to the view, or nullptr if not found
         */
        const WeightView *find(const std::string &name) const
        {
            auto it = index_.find(name);
            if (it == index_.end())
            {
                return nullptr;
            }
            return &views_[it->second];
        }

        /**
         * @brief Get a weight tensor by name (convenience)
         * @return Shared pointer to the tensor, or nullptr if not found
         */
        std::shared_ptr<TensorBase> getTensor(const std::string &name) const
        {
            const auto *view = find(name);
            return view ? view->tensor : nullptr;
        }

        /** @brief Number of weight views in this set */
        size_t size() const { return views_.size(); }

        /** @brief Whether this set is empty */
        bool empty() const { return views_.empty(); }

        /** @brief Get the layer range for this view set */
        const LayerRange &layerRange() const { return layer_range_; }

        /** @brief First layer index (inclusive) */
        int firstLayer() const { return layer_range_.first; }

        /** @brief Last layer index (exclusive) */
        int lastLayer() const { return layer_range_.last; }

        /** @brief Whether this view set includes the embedding table */
        bool hasEmbedding() const { return layer_range_.has_embedding; }

        /** @brief Whether this view set includes the LM head */
        bool hasLMHead() const { return layer_range_.has_lm_head; }

        // =========================================================================
        // Iteration (range-based for)
        // =========================================================================

        using const_iterator = std::vector<WeightView>::const_iterator;

        const_iterator begin() const { return views_.begin(); }
        const_iterator end() const { return views_.end(); }

        /** @brief Get all weight names in this set */
        std::vector<std::string> weightNames() const
        {
            std::vector<std::string> names;
            names.reserve(views_.size());
            for (const auto &view : views_)
            {
                names.push_back(view.name);
            }
            return names;
        }

    private:
        std::vector<WeightView> views_;
        std::unordered_map<std::string, size_t> index_; ///< name → index in views_
        LayerRange layer_range_;
    };

} // namespace llaminar2
