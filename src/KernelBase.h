#pragma once

#include <vector>
#include <memory>
#include <string>
#include "tensors/TensorBase.h"

namespace llaminar
{
    /**
     * @brief Base interface for all computational operators
     *
     * This interface provides a common API for all operators to enable
     * polymorphic execution within the compute graph framework.
     */
    class OperatorBase
    {
    public:
        virtual ~OperatorBase() = default;

        /**
         * @brief Execute the operator computation
         * @param inputs Vector of input tensors
         * @param outputs Vector of output tensors (should be pre-allocated)
         * @return true if execution succeeded, false otherwise
         */
        virtual bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                             std::vector<std::shared_ptr<TensorBase>> &outputs) = 0;

        /**
         * @brief Validate input and output tensor shapes and types
         * @param inputs Input tensors to validate
         * @param outputs Output tensors to validate
         * @return true if tensors are valid, false otherwise
         */
        virtual bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                              const std::vector<std::shared_ptr<TensorBase>> &outputs) const = 0;

        /**
         * @brief Get the operator type name for debugging/logging
         * @return String identifying the operator type
         */
        virtual std::string getOperatorType() const = 0;

        /**
         * @brief Get expected number of input tensors
         * @return Number of input tensors this operator expects
         */
        virtual size_t getExpectedInputCount() const = 0;

        /**
         * @brief Get expected number of output tensors
         * @return Number of output tensors this operator produces
         */
        virtual size_t getExpectedOutputCount() const = 0;
    };

} // namespace llaminar