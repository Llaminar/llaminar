/**
 * @file PreparedGraphTemplate.h
 * @brief Cold immutable recipes shared by independent graph specializations.
 *
 * A builder owns one recipe by value, so publication adds no detached cache,
 * payload owner or pointer-keyed lookup. Source mutation is closed once the
 * recipe has been prepared. Specializations borrow that original recipe while
 * creating their own stages, buffers and complete native capture identities.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace llaminar2
{
    /** @brief Host construction observations; never execution or memory authority. */
    struct PreparedGraphTemplateStats
    {
        std::uint64_t preparations = 0;
        std::uint64_t specialization_attempts = 0;
        /// Actual constructor attempts, including validation failures.
        std::uint64_t canonical_owner_map_builds = 0;
        std::uint64_t original_owner_map_reuses = 0;
        /// Actual resolver attempts, including validation failures.
        std::uint64_t continuation_resolutions = 0;
        /// Actual routed layer producer entries, independent of whole-forward attempts.
        std::uint64_t routed_layer_specialization_attempts = 0;
    };

    /**
     * @brief One original builder's immutable recipe, without a global cache key.
     * @tparam Recipe Static source views or already-owned immutable descriptors.
     * @details The nonmovable owner prevents a recipe's views from silently
     * rebinding to a different builder. It deliberately owns no graph stages.
     * Existing model/PMA owners retain all payloads. Cold setup is serialized
     * by the original builder owner, just like its existing configuration.
     */
    template <typename Recipe>
    class PreparedGraphTemplate final
    {
    public:
        /** @brief Create an unprepared cold owner, without allocating storage. */
        PreparedGraphTemplate() = default;
        PreparedGraphTemplate(const PreparedGraphTemplate &) = delete;
        PreparedGraphTemplate &operator=(const PreparedGraphTemplate &) = delete;
        PreparedGraphTemplate(PreparedGraphTemplate &&) = delete;
        PreparedGraphTemplate &operator=(PreparedGraphTemplate &&) = delete;

        /**
         * @brief Publish one complete immutable recipe, atomically on success.
         * @param factory Original builder's cold source validator/preparer.
         * @return The same recipe on every later preparation call.
         * @details A throwing factory leaves the owner unprepared; no partial
         * source or success observation is published.
         */
        template <typename Factory>
        const Recipe &prepare(Factory &&factory)
        {
            if (!recipe_)
            {
                recipe_.emplace(std::invoke(std::forward<Factory>(factory)));
                ++preparations_;
            }
            return *recipe_;
        }

        /** @return Whether the original cold source has been published. */
        [[nodiscard]] bool prepared() const noexcept { return recipe_.has_value(); }

        /** @brief Reject source replacement after immutable recipe publication. */
        void requireUnprepared(const char *operation) const
        {
            if (recipe_)
                throw std::logic_error(std::string(operation) +
                    " cannot replace a prepared graph-template source");
        }

        /** @brief Borrow original source views without constructing mutable stages. */
        [[nodiscard]] const Recipe &recipe() const
        {
            if (!recipe_)
                throw std::logic_error("Graph template source has not been prepared");
            return *recipe_;
        }

        /**
         * @brief Begin one independent production graph construction attempt.
         * @return The original immutable recipe; invocation state remains external.
         */
        [[nodiscard]] const Recipe &beginSpecialization() const
        {
            const auto &result = recipe();
            ++specialization_attempts_;
            return result;
        }

        /** @return Exact successful recipe publications by this owner. */
        [[nodiscard]] std::uint64_t preparations() const noexcept { return preparations_; }
        /** @return Actual graph specialization attempts, including thrown builds. */
        [[nodiscard]] std::uint64_t specializationAttempts() const noexcept
        { return specialization_attempts_; }

    private:
        std::optional<Recipe> recipe_;
        std::uint64_t preparations_ = 0;
        mutable std::uint64_t specialization_attempts_ = 0;
    };
}
