/**
 * @file ImmutableHardwarePublication.h
 * @brief Publish one complete immutable hardware inventory without hot-read locks.
 *
 * Inventory construction validates all records before publication. Discovery or
 * validation failure is retained permanently; another caller cannot retry with a
 * different physical identity. Native resource generations have a separate owner.
 */
#pragma once
#include <atomic>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Backend-independent once-publication mechanism, not a hardware ledger. */
    template <typename Inventory>
    class ImmutableHardwarePublication final
    {
    public:
        /**
         * @brief Construct and publish one fully validated immutable inventory.
         * @tparam Source Callable returning the inventory constructor's records.
         * @param source Explicit setup-only reader, or a device-free recorded reader.
         * @throws The original discovery/admission exception on every failed initialization.
         */
        template <typename Source>
        void initialize(Source &&source)
        {
            std::call_once(initialization_, [&] {
                try
                {
                    inventory_ = std::make_unique<const Inventory>(std::invoke(source));
                    published_.store(inventory_.get(), std::memory_order_release);
                }
                catch (...) { failure_ = std::current_exception(); }
            });
            if (failure_) std::rethrow_exception(failure_);
        }

        /**
         * @brief Read a previously admitted inventory without native calls or locks.
         * @param absent_diagnostic Backend-specific precise pre-admission diagnostic.
         * @return Process-lifetime immutable inventory.
         * @throws std::logic_error Before successful explicit admission.
         */
        [[nodiscard]] const Inventory &snapshot(const char *absent_diagnostic) const
        {
            const auto *inventory = published_.load(std::memory_order_acquire);
            if (!inventory) throw std::logic_error(absent_diagnostic);
            return *inventory;
        }
    private:
        std::once_flag initialization_;
        std::unique_ptr<const Inventory> inventory_;
        std::atomic<const Inventory *> published_{nullptr};
        std::exception_ptr failure_;
    };
}
