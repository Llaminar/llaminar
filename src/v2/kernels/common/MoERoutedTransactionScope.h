/**
 * @file MoERoutedTransactionScope.h
 * @brief Original static routed transaction identity before grouping chooses its producer.
 *
 * This lexical capture-worker binding contains immutable call arguments only.
 * It neither reads device counts nor caches a backend dispatch decision. Grouping
 * and complete projection authenticate the same original operation; the existing
 * grouping publication remains the sole authority for its device directory.
 */
#pragma once
#include <climits>
#include <exception>
#include <optional>
#include <stdexcept>

namespace llaminar2
{
class IMoEKernel;
class ITensor;
class DeviceWorkspaceManager;

/** @brief Original unsliced call identity, borrowed only during synchronous recording. */
struct MoERoutedTransactionBinding
{
    const IMoEKernel *owner = nullptr; ///< Exact prepared kernel instance.
    ITensor *indices = nullptr, *weights = nullptr; ///< Original routing tensor authorities.
    ITensor *hidden = nullptr, *output = nullptr, *contributions = nullptr; ///< Original activation/publication targets.
    int gateup_table = -1, down_table = -1; ///< Immutable instance-owned descriptor tables.
    int rows = 0, hidden_width = 0, intermediate = 0, experts = 0, top_k = 0; ///< Original physical geometry.
    void *stream = nullptr; ///< Exact caller-owned recording stream.
    DeviceWorkspaceManager *workspace = nullptr; ///< Retained arena view, never its live contents.

    /** @brief Reject incomplete original identities before any grouping launch. */
    void validate() const
    {
        if (!owner || !indices || !weights || !hidden || (!output && !contributions) ||
            gateup_table < 0 || down_table < 0 || rows <= 0 || hidden_width <= 0 ||
            intermediate <= 0 || experts <= 0 || top_k <= 0 || rows > INT_MAX / top_k ||
            !stream || !workspace)
            throw std::invalid_argument("incomplete original routed transaction binding");
    }
    /** @return Whether every immutable call argument is identical. */
    bool operator==(const MoERoutedTransactionBinding &) const = default;
};

/** @brief LIFO scope; empty nested boundaries hide the enclosing operation. */
class ScopedMoERoutedTransaction final
{
public:
    /** @param binding Complete original identity, or an explicitly unmeasured nested call. */
    explicit ScopedMoERoutedTransaction(std::optional<MoERoutedTransactionBinding> binding)
        : binding_(binding), previous_(active_)
    {
        if (binding_) binding_->validate();
        active_ = this;
    }
    /** @brief Restore the enclosing operation; non-LIFO retirement is fatal. */
    ~ScopedMoERoutedTransaction()
    {
        if (active_ != this) std::terminate();
        active_ = previous_;
    }
    ScopedMoERoutedTransaction(const ScopedMoERoutedTransaction &) = delete;
    ScopedMoERoutedTransaction &operator=(const ScopedMoERoutedTransaction &) = delete;
    /** @return This worker's original call, never a mutable backend plan. */
    [[nodiscard]] static const MoERoutedTransactionBinding *current() noexcept
    { return active_ && active_->binding_ ? &*active_->binding_ : nullptr; }
    /** @param expected Original immutable call arguments at a consumer boundary.
     * @throws std::logic_error If a consumer borrows an unrelated enclosing call. */
    static void requireMatching(const MoERoutedTransactionBinding &expected)
    {
        if (!current() || *current() != expected)
            throw std::logic_error("routed transaction changed its original call binding");
    }
private:
    const std::optional<MoERoutedTransactionBinding> binding_;
    const ScopedMoERoutedTransaction *previous_;
    inline static thread_local const ScopedMoERoutedTransaction *active_ = nullptr;
};
}
