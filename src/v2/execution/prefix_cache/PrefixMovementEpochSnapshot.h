/**
 * @file PrefixMovementEpochSnapshot.h
 * @brief Preserve independent placement epochs in optional state diagnostics.
 *
 * Epochs are ordered only within one movement authority. A pipeline snapshot
 * retains every child in declared stage order; taking their maximum would hide
 * movement in any stage behind a larger sibling. This passive value contains
 * metadata only and never observes, owns, hashes, or validates cache payloads.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace llaminar2
{
    /** @brief A single authority or the complete ordered children of a pipeline. */
    class PrefixMovementEpochSnapshot final
    {
    public:
        /** @brief An uninitialized or non-MoE participant has epoch zero. */
        PrefixMovementEpochSnapshot() = default;

        /** @param epoch Completed publication in one authority's namespace.
         * @return A leaf observation, with no subordinate stage epochs. */
        static PrefixMovementEpochSnapshot leaf(uint64_t epoch)
        {
            PrefixMovementEpochSnapshot result;
            result.value_ = epoch;
            return result;
        }

        /** @param children Every pipeline child in its declared stage order.
         * @return An observation which cannot be reduced to a scalar epoch.
         * @throws std::invalid_argument for an empty pipeline observation. */
        static PrefixMovementEpochSnapshot pipeline(std::vector<PrefixMovementEpochSnapshot> children)
        {
            if (children.empty())
                throw std::invalid_argument("Pipeline movement diagnostics require every stage");
            PrefixMovementEpochSnapshot result;
            result.value_ = std::move(children);
            return result;
        }

        /** @return Whether this value belongs to a single movement authority. */
        bool isLeaf() const noexcept { return std::holds_alternative<uint64_t>(value_); }

        /** @return The single authority's epoch.
         * @throws std::logic_error when a caller tries to flatten a pipeline. */
        uint64_t epoch() const
        {
            if (!isLeaf())
                throw std::logic_error("Independent pipeline movement epochs have no scalar reduction");
            return std::get<uint64_t>(value_);
        }

        /** @return Ordered child observations, borrowing this snapshot.
         * @throws std::logic_error for a leaf without subordinate stages. */
        std::span<const PrefixMovementEpochSnapshot> stages() const &
        {
            if (isLeaf())
                throw std::logic_error("A movement authority has no pipeline children");
            return std::get<std::vector<PrefixMovementEpochSnapshot>>(value_);
        }
        /** @brief A temporary snapshot cannot lend a child view. */
        std::span<const PrefixMovementEpochSnapshot> stages() const && = delete;

        /** @param other Another boundary from the same declared execution topology.
         * @return Whether their complete stage structure agrees, ignoring epochs. */
        bool sameScope(const PrefixMovementEpochSnapshot &other) const
        {
            if (isLeaf() != other.isLeaf()) return false;
            if (isLeaf()) return true;
            const auto lhs = stages(), rhs = other.stages();
            if (lhs.size() != rhs.size()) return false;
            for (size_t i = 0; i < lhs.size(); ++i)
                if (!lhs[i].sameScope(rhs[i])) return false;
            return true;
        }

        /** @brief Compare every epoch without collapsing independent namespaces. */
        bool operator==(const PrefixMovementEpochSnapshot &) const = default;

    private:
        std::variant<uint64_t, std::vector<PrefixMovementEpochSnapshot>> value_{uint64_t{0}};
    };

    /** @param output Diagnostic text destination, including CSV cells.
     * @param snapshot Complete movement observation.
     * @return The destination after writing a scalar or semicolon-delimited tree. */
    inline std::ostream &operator<<(std::ostream &output, const PrefixMovementEpochSnapshot &snapshot)
    {
        if (snapshot.isLeaf()) return output << snapshot.epoch();
        output << '[';
        const auto children = snapshot.stages();
        for (size_t i = 0; i < children.size(); ++i)
        {
            if (i) output << ';'; // One field even in diagnostic CSV output.
            output << children[i];
        }
        return output << ']';
    }

    /** @tparam Json JSON value supporting the standard nlohmann ADL interface.
     * @param output JSON destination.
     * @param snapshot Scalar leaf or complete declared pipeline stage tree.
     * Leaf values retain their numeric representation; pipeline values publish
     * an explicit ordered `stages` object instead of a fabricated maximum. */
    template<class Json>
    void to_json(Json &output, const PrefixMovementEpochSnapshot &snapshot)
    {
        if (snapshot.isLeaf())
        {
            output = snapshot.epoch();
            return;
        }
        auto children = Json::array();
        for (const auto &child : snapshot.stages()) children.push_back(child);
        output = {{"stages", std::move(children)}};
    }
}
