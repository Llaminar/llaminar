/**
 * @file FloatingOutputPartitionScope.h
 * @brief Recording-time contract for column-invariant floating projections.
 *
 * Mirroring an LM head changes physical N, not its dot-product arithmetic.
 * Floating kernels satisfy the contract with their fixed per-column reduction,
 * independent of N, rather than NativeVNNI's serial-partition dispatch policy.
 * This scope binds that requirement to one kernel on one recording thread; it
 * is not execution state and introduces no GPU transfers or mutable graph input.
 */
#pragma once

#include "tensors/TensorKernels.h"
#include <exception>

namespace llaminar2
{
    /** @brief Nestable, thread-local requirement for a prepared floating kernel. */
    class FloatingOutputPartitionScope final : public ITensorGemm::OutputPartitionEquivalenceScope
    {
    public:
        /**
         * @brief Validate physical geometry and require column-invariant arithmetic.
         * @param kernel Prepared projection whose physical N must match the request.
         * @param actual_columns Full physical output width, including a tail shard.
         * @param serial_columns Regular serial shard width (at most physical N).
         * @return Token retained through workspace admission or kernel recording.
         * @throws std::invalid_argument For inconsistent or nonpositive geometry.
         */
        static std::unique_ptr<ITensorGemm::OutputPartitionEquivalenceScope> begin(
            const ITensorGemm &kernel, int actual_columns, int serial_columns)
        {
            if (actual_columns <= 0 || serial_columns <= 0 ||
                serial_columns > actual_columns || actual_columns != kernel.get_n())
                throw std::invalid_argument("Floating output partition disagrees with prepared geometry");
            return std::unique_ptr<ITensorGemm::OutputPartitionEquivalenceScope>(
                new FloatingOutputPartitionScope(kernel, actual_columns));
        }

        /** @brief Restore the enclosing scope; out-of-order destruction is fatal. */
        ~FloatingOutputPartitionScope() override
        {
            if (current_ != this)
                std::terminate();
            current_ = previous_;
        }

        /**
         * @brief Query this kernel's recorded arithmetic contract, validating N.
         * @param kernel Exact prepared kernel being launched, not a sibling view.
         * @param columns Physical output width of the launch.
         * @return True only when an enclosing scope names this kernel.
         * @throws std::logic_error When launch geometry differs from its scope.
         */
        static bool requiresFixedColumns(const ITensorGemm &kernel, int columns)
        {
            for (auto *scope = current_; scope; scope = scope->previous_)
            {
                if (scope->kernel_ != &kernel)
                    continue;
                if (scope->columns_ != columns)
                    throw std::logic_error("Floating output launch disagrees with partition scope");
                return true;
            }
            return false;
        }

        FloatingOutputPartitionScope(const FloatingOutputPartitionScope &) = delete;
        FloatingOutputPartitionScope &operator=(const FloatingOutputPartitionScope &) = delete;

    private:
        /** @brief Push validated recording metadata without modifying the prepared engine. */
        FloatingOutputPartitionScope(const ITensorGemm &kernel, int columns)
            : kernel_(&kernel), columns_(columns), previous_(current_)
        {
            current_ = this;
        }

        const ITensorGemm *kernel_; ///< Identity only; the caller retains the prepared engine.
        int columns_; ///< Physical geometry, never a surrogate shard allocation size.
        FloatingOutputPartitionScope *previous_; ///< Enclosing recording contract.
        inline static thread_local FloatingOutputPartitionScope *current_ = nullptr;
    };
}
