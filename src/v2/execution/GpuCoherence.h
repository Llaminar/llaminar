/**
 * @file GpuCoherence.h
 * @brief RAII utilities for GPU tensor coherence management
 *
 * This header provides intuitive, self-documenting patterns for managing
 * GPU tensor coherence when calling kernels directly (outside GraphExecutor).
 *
 * ## Why This Exists
 *
 * When using GraphExecutor, coherence is handled automatically at stage boundaries.
 * However, when calling kernels directly (in tests, utilities, or custom pipelines),
 * you must manually:
 * 1. Call `ensureOnDevice()` on inputs before the kernel runs
 * 2. Call `mark_device_dirty()` on outputs after the kernel completes
 *
 * These utilities make that pattern RAII-based and self-documenting.
 *
 * ## Usage Patterns
 *
 * ### Simple Case: Single Output
 * ```cpp
 * {
 *     auto output = gpu_output(output_tensor.get(), gpu_device);
 *     kernel->multiply_tensor(input.get(), output, M, N, K, ...);
 * } // ← output automatically marked dirty when scope exits
 * ```
 *
 * ### Complex Case: Multiple Inputs/Outputs
 * ```cpp
 * bool ok = with_gpu_coherence(
 *     gpu_device,
 *     {input.get()},                              // inputs to cohere
 *     {out_q.get(), out_k.get(), out_v.get()},    // outputs to cohere + mark dirty
 *     [&] {
 *         return kernel->multiply_fused_tensor(input.get(), projections, M, K, nullptr);
 *     }
 * );
 * ```
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "backends/DeviceId.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"

#include <initializer_list>
#include <vector>
#include <type_traits>
#include <concepts>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;

    // =========================================================================
    // Concepts
    // =========================================================================

    /**
     * @brief Concept for tensors that support GPU coherence operations
     */
    template <typename T>
    concept CoherableTensor = requires(T *t, DeviceId d) {
        { t->ensureOnDevice(d) } -> std::same_as<bool>;
        { t->mark_device_dirty() } -> std::same_as<void>;
    };

    // =========================================================================
    // GpuOutput - RAII Proxy Wrapper
    // =========================================================================

    /**
     * @brief RAII wrapper that ensures a tensor is on GPU and marks it dirty on destruction
     *
     * Use this for simple single-output cases. The wrapper:
     * 1. Calls ensureOnDevice() on construction
     * 2. Calls mark_device_dirty() on destruction
     *
     * The implicit conversion to T* allows passing directly to kernel functions.
     *
     * @tparam T Tensor type (must satisfy CoherableTensor concept)
     *
     * @example
     * ```cpp
     * {
     *     auto output = gpu_output(output_cuda.get(), gpu_device_);
     *     kernel->multiply_tensor(input.get(), output, M, N, K, ...);
     * } // output automatically marked dirty
     * ```
     */
    template <CoherableTensor T>
    class GpuOutput
    {
        T *tensor_;
        bool valid_ = false;

    public:
        /**
         * @brief Construct and ensure tensor is on specified device
         * @param tensor The tensor to wrap
         * @param device Target GPU device
         */
        explicit GpuOutput(T *tensor, DeviceId device)
            : tensor_(tensor)
        {
            if (tensor_)
            {
                valid_ = tensor_->ensureOnDevice(device);
                if (!valid_)
                {
                    LOG_ERROR("[GpuOutput] Failed to ensure tensor on device " << device.toString());
                }
            }
        }

        /**
         * @brief Destructor marks output as device-dirty if valid
         */
        ~GpuOutput()
        {
            if (tensor_ && valid_)
            {
                tensor_->mark_device_dirty();
            }
        }

        // Non-copyable, moveable
        GpuOutput(const GpuOutput &) = delete;
        GpuOutput &operator=(const GpuOutput &) = delete;

        GpuOutput(GpuOutput &&other) noexcept
            : tensor_(other.tensor_), valid_(other.valid_)
        {
            other.tensor_ = nullptr;
            other.valid_ = false;
        }

        GpuOutput &operator=(GpuOutput &&other) noexcept
        {
            if (this != &other)
            {
                // Mark current dirty before taking over
                if (tensor_ && valid_)
                {
                    tensor_->mark_device_dirty();
                }
                tensor_ = other.tensor_;
                valid_ = other.valid_;
                other.tensor_ = nullptr;
                other.valid_ = false;
            }
            return *this;
        }

        // Accessors
        T *get() const { return tensor_; }
        T *operator->() const { return tensor_; }
        T &operator*() const { return *tensor_; }

        /**
         * @brief Implicit conversion to T* for seamless kernel calls
         */
        operator T *() const { return tensor_; }

        /**
         * @brief Check if the tensor was successfully placed on device
         */
        bool is_valid() const { return valid_; }

        /**
         * @brief Explicit bool conversion for validity check
         */
        explicit operator bool() const { return valid_; }
    };

    /**
     * @brief Factory function to create GpuOutput with type deduction
     *
     * @param tensor The tensor to wrap
     * @param device Target GPU device
     * @return GpuOutput wrapper
     *
     * @example
     * ```cpp
     * auto output = gpu_output(my_tensor.get(), gpu_device);
     * kernel->compute(input, output);  // implicit conversion to TensorBase*
     * ```
     */
    template <CoherableTensor T>
    [[nodiscard]] auto gpu_output(T *tensor, DeviceId device)
    {
        return GpuOutput<T>(tensor, device);
    }

    // =========================================================================
    // GpuInput - RAII Input Coherence (read-only)
    // =========================================================================

    /**
     * @brief RAII wrapper that ensures a tensor is on GPU (read-only, no dirty marking)
     *
     * Use this for inputs that need to be on GPU but won't be modified.
     *
     * @tparam T Tensor type (must satisfy CoherableTensor concept)
     */
    template <CoherableTensor T>
    class GpuInput
    {
        T *tensor_;
        bool valid_ = false;

    public:
        explicit GpuInput(T *tensor, DeviceId device)
            : tensor_(tensor)
        {
            if (tensor_)
            {
                valid_ = tensor_->ensureOnDevice(device);
                if (!valid_)
                {
                    LOG_ERROR("[GpuInput] Failed to ensure tensor on device " << device.toString());
                }
            }
        }

        // Destructor does NOT mark dirty (inputs are read-only)
        ~GpuInput() = default;

        // Non-copyable, moveable
        GpuInput(const GpuInput &) = delete;
        GpuInput &operator=(const GpuInput &) = delete;
        GpuInput(GpuInput &&) noexcept = default;
        GpuInput &operator=(GpuInput &&) noexcept = default;

        T *get() const { return tensor_; }
        T *operator->() const { return tensor_; }
        T &operator*() const { return *tensor_; }
        operator T *() const { return tensor_; }
        bool is_valid() const { return valid_; }
        explicit operator bool() const { return valid_; }
    };

    /**
     * @brief Factory function to create GpuInput with type deduction
     */
    template <CoherableTensor T>
    [[nodiscard]] auto gpu_input(T *tensor, DeviceId device)
    {
        return GpuInput<T>(tensor, device);
    }

    // =========================================================================
    // with_gpu_coherence - Lambda Wrapper for Complex Cases
    // =========================================================================

    /**
     * @brief Execute a kernel with automatic input/output coherence management
     *
     * This function:
     * 1. Calls ensureOnDevice() on all inputs
     * 2. Calls ensureOnDevice() on all outputs (to allocate GPU memory)
     * 3. Executes the kernel function
     * 4. If successful, calls mark_device_dirty() on all outputs
     *
     * @param device Target GPU device
     * @param inputs List of input tensors to cohere (read-only)
     * @param outputs List of output tensors to cohere and mark dirty
     * @param kernel_fn Lambda/function that executes the kernel, returns bool
     * @return true if all coherence operations and kernel execution succeeded
     *
     * @example
     * ```cpp
     * bool ok = with_gpu_coherence(
     *     gpu_device_,
     *     {input.get()},                              // inputs
     *     {out_q.get(), out_k.get(), out_v.get()},    // outputs
     *     [&] {
     *         return kernel->multiply_fused_tensor(input.get(), projections, M, K, nullptr);
     *     }
     * );
     * ```
     */
    template <typename F>
        requires std::is_invocable_r_v<bool, F>
    bool with_gpu_coherence(
        DeviceId device,
        std::initializer_list<TensorBase *> inputs,
        std::initializer_list<TensorBase *> outputs,
        F &&kernel_fn)
    {
        // Cohere inputs (ensure on device)
        for (auto *tensor : inputs)
        {
            if (tensor && !tensor->ensureOnDevice(device))
            {
                LOG_ERROR("[with_gpu_coherence] Failed to cohere input tensor to device "
                          << device.toString());
                return false;
            }
        }

        // Cohere outputs (ensure on device - allocates GPU memory)
        for (auto *tensor : outputs)
        {
            if (tensor && !tensor->ensureOnDevice(device))
            {
                LOG_ERROR("[with_gpu_coherence] Failed to cohere output tensor to device "
                          << device.toString());
                return false;
            }
        }

        // Execute kernel
        bool success = std::forward<F>(kernel_fn)();

        // Mark outputs as device-dirty (GPU now has authoritative data)
        if (success)
        {
            for (auto *tensor : outputs)
            {
                if (tensor)
                {
                    tensor->mark_device_dirty();
                }
            }
        }

        return success;
    }

    /**
     * @brief Overload for kernels that don't return bool (void or other)
     *
     * Always marks outputs dirty after execution completes.
     */
    template <typename F>
        requires(!std::is_invocable_r_v<bool, F>)
    bool with_gpu_coherence(
        DeviceId device,
        std::initializer_list<TensorBase *> inputs,
        std::initializer_list<TensorBase *> outputs,
        F &&kernel_fn)
    {
        // Cohere inputs
        for (auto *tensor : inputs)
        {
            if (tensor && !tensor->ensureOnDevice(device))
            {
                LOG_ERROR("[with_gpu_coherence] Failed to cohere input tensor to device "
                          << device.toString());
                return false;
            }
        }

        // Cohere outputs
        for (auto *tensor : outputs)
        {
            if (tensor && !tensor->ensureOnDevice(device))
            {
                LOG_ERROR("[with_gpu_coherence] Failed to cohere output tensor to device "
                          << device.toString());
                return false;
            }
        }

        // Execute kernel (no return value)
        std::forward<F>(kernel_fn)();

        // Mark outputs dirty
        for (auto *tensor : outputs)
        {
            if (tensor)
            {
                tensor->mark_device_dirty();
            }
        }

        return true;
    }

    // =========================================================================
    // GpuCoherenceScope - Multi-tensor RAII Scope
    // =========================================================================

    /**
     * @brief RAII scope that manages coherence for multiple inputs/outputs
     *
     * Alternative to with_gpu_coherence when you need more control over
     * the kernel execution flow.
     *
     * @example
     * ```cpp
     * {
     *     GpuCoherenceScope scope(gpu_device_);
     *     scope.add_input(input.get());
     *     scope.add_output(out_q.get());
     *     scope.add_output(out_k.get());
     *
     *     if (!scope.cohere()) {
     *         return false;  // coherence failed
     *     }
     *
     *     kernel->multiply_fused_tensor(...);
     *     scope.mark_success();  // enables dirty marking on destruction
     * }
     * ```
     */
    class GpuCoherenceScope
    {
        DeviceId device_;
        std::vector<TensorBase *> inputs_;
        std::vector<TensorBase *> outputs_;
        bool cohered_ = false;
        bool success_ = false;

    public:
        explicit GpuCoherenceScope(DeviceId device) : device_(device) {}

        ~GpuCoherenceScope()
        {
            if (cohered_ && success_)
            {
                for (auto *tensor : outputs_)
                {
                    if (tensor)
                    {
                        tensor->mark_device_dirty();
                    }
                }
            }
        }

        // Non-copyable
        GpuCoherenceScope(const GpuCoherenceScope &) = delete;
        GpuCoherenceScope &operator=(const GpuCoherenceScope &) = delete;

        /**
         * @brief Add an input tensor (will be cohered, not marked dirty)
         */
        GpuCoherenceScope &add_input(TensorBase *tensor)
        {
            if (tensor)
                inputs_.push_back(tensor);
            return *this;
        }

        /**
         * @brief Add an output tensor (will be cohered AND marked dirty on success)
         */
        GpuCoherenceScope &add_output(TensorBase *tensor)
        {
            if (tensor)
                outputs_.push_back(tensor);
            return *this;
        }

        /**
         * @brief Perform coherence on all added tensors
         * @return true if all tensors successfully cohered to device
         */
        bool cohere()
        {
            for (auto *tensor : inputs_)
            {
                if (!tensor->ensureOnDevice(device_))
                {
                    LOG_ERROR("[GpuCoherenceScope] Failed to cohere input to device "
                              << device_.toString());
                    return false;
                }
            }
            for (auto *tensor : outputs_)
            {
                if (!tensor->ensureOnDevice(device_))
                {
                    LOG_ERROR("[GpuCoherenceScope] Failed to cohere output to device "
                              << device_.toString());
                    return false;
                }
            }
            cohered_ = true;
            return true;
        }

        /**
         * @brief Mark kernel execution as successful (enables dirty marking)
         */
        void mark_success() { success_ = true; }

        /**
         * @brief Get coherence status
         */
        bool is_cohered() const { return cohered_; }
    };

} // namespace llaminar2
