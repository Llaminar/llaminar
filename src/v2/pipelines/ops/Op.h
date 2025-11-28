/**
 * @file Op.h
 * @brief Base operation interface and common utilities for pipeline operations
 *
 * Operations encapsulate the validate-execute-capture pattern:
 * 1. Validate inputs (null checks, dimension checks)
 * 2. Create kernel (from appropriate tensor interface)
 * 3. Execute kernel (with error handling)
 * 4. Capture snapshot (if enabled)
 *
 * Benefits:
 * - Self-validating: Operations handle all validation internally
 * - Reusable: Same operations work across Qwen2/Qwen3/MoE pipelines
 * - Testable: Operations can be unit tested in isolation
 * - Clean pipelines: Reduces attention_block() from 150 lines to ~50 lines
 *
 * Usage:
 * @code
 * RMSNormOp rmsnorm;
 * if (!rmsnorm(input, weight, output, rows, cols, eps, "ATTN_NORM", mpi, device))
 *     return false;
 *
 * // Or with TRY_OP macro:
 * TRY_OP(rmsnorm(input, weight, output, rows, cols, eps, "ATTN_NORM", mpi, device));
 * @endcode
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../utils/Logger.h"
#include "../../utils/MPIContext.h"
#include "../../tensors/Tensors.h"

namespace llaminar2
{

/**
 * @brief Early-return macro for operation chains
 *
 * Use in pipeline methods to propagate operation failures:
 * @code
 * TRY_OP(rmsnorm(input, weight, output, ...));
 * TRY_OP(gemm(A, W, C, ...));
 * @endcode
 */
#define TRY_OP(expr)      \
    do                    \
    {                     \
        if (!(expr))      \
            return false; \
    } while (0)

    /**
     * @brief Base class for pipeline operations
     *
     * Provides common infrastructure for operation validation and logging.
     * Derived classes implement operator() with operation-specific logic.
     */
    class OpBase
    {
    public:
        virtual ~OpBase() = default;

        /**
         * @brief Get operation name for logging
         */
        virtual const char *name() const = 0;

    protected:
        /**
         * @brief Log error with operation name prefix
         */
        void logError(const char *message) const
        {
            LOG_ERROR(name() << ": " << message);
        }

        /**
         * @brief Validate pointer is non-null
         */
        bool validatePointer(const void *ptr, const char *desc) const
        {
            if (!ptr)
            {
                LOG_ERROR(name() << ": null " << desc);
                return false;
            }
            return true;
        }

        /**
         * @brief Validate tensor pointer and basic properties
         */
        bool validateTensor(const TensorBase *tensor, const char *desc) const
        {
            if (!tensor)
            {
                LOG_ERROR(name() << ": null " << desc << " tensor");
                return false;
            }
            if (!tensor->data())
            {
                LOG_ERROR(name() << ": " << desc << " tensor has null data");
                return false;
            }
            return true;
        }

        /**
         * @brief Validate dimensions are positive
         */
        bool validateDimensions(int rows, int cols, const char *context) const
        {
            if (rows <= 0 || cols <= 0)
            {
                LOG_ERROR(name() << ": invalid dimensions for " << context
                                 << " (rows=" << rows << ", cols=" << cols << ")");
                return false;
            }
            return true;
        }
    };

} // namespace llaminar2
