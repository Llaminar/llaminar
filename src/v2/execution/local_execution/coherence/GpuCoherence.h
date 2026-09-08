/**
 * @file GpuCoherence.h
 * @brief Explicit-stream coherence helper for direct GPU kernel tests
 *
 * Production GPU execution must express tensor movement and write publication
 * through graph stages and TransferEngine. This header exists only for focused
 * integration tests that invoke a kernel directly, outside DeviceGraphExecutor,
 * and therefore need one small utility that performs the same explicit-stream
 * setup and publication protocol.
 *
 * ## Why This Exists
 *
 * When using DeviceGraphExecutor, coherence is handled automatically at stage
 * boundaries. A direct-kernel integration test must:
 * 1. make every input and output allocation available on the kernel stream;
 * 2. launch the kernel on that exact stream; and
 * 3. publish successful writes with that same producer stream.
 *
 * `with_gpu_coherence()` keeps those three operations adjacent and rejects a
 * null stream. It deliberately is not an RAII object: publication must be an
 * explicit consequence of successful kernel execution, never a destructor side
 * effect whose ordering is hidden from the caller.
 *
 * ## Usage Patterns
 *
 * ```cpp
 * bool ok = with_gpu_coherence(
 *     gpu_device,
 *     {input.get()},                              // inputs to cohere
 *     {out_q.get(), out_k.get(), out_v.get()},    // outputs to cohere + mark dirty
 *     producer_stream,
 *     [&] {
 *         return kernel->multiply_fused_tensor(
 *             input.get(), projections, M, K, producer_stream);
 *     }
 * );
 * ```
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../../../backends/DeviceId.h"
#include "../../../tensors/Tensors.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/Logger.h"

#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace llaminar2
{

    // Forward declaration
    class TensorBase;

    /**
     * @brief Execute one direct test kernel with explicit-stream coherence
     *
     * This function:
     * 1. Prepares all inputs through TransferEngine on the consumer stream
     * 2. Prepares output-only storage without uploading stale host bytes
     * 3. Executes the kernel function
     * 4. If successful, publishes all output writes through TransferEngine
     *
     * @param device Target GPU device
     * @param inputs List of input tensors to cohere (read-only)
     * @param outputs List of output tensors to cohere and mark dirty
     * @param producer_stream Exact non-null stream used by @p kernel_fn
     * @param kernel_fn Lambda/function that executes the kernel, returns bool
     * @return true if all coherence operations and kernel execution succeeded
     *
     * @example
     * ```cpp
     * bool ok = with_gpu_coherence(
     *     gpu_device_,
     *     {input.get()},                              // inputs
     *     {out_q.get(), out_k.get(), out_v.get()},    // outputs
     *     producer_stream,
     *     [&] {
     *         return kernel->multiply_fused_tensor(
     *             input.get(), projections, M, K, producer_stream);
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
        void *producer_stream,
        F &&kernel_fn)
    {
        if (!producer_stream)
        {
            throw std::invalid_argument(
                "with_gpu_coherence requires the exact non-null producer stream");
        }

        // Join every input producer to the exact consumer stream.
        for (auto *tensor : inputs)
        {
            if (tensor)
                TransferEngine::prepareDeviceInput(
                    tensor, device, producer_stream);
        }

        // Allocate output-only storage without importing stale host contents.
        for (auto *tensor : outputs)
        {
            if (tensor)
                TransferEngine::prepareDeviceOutput(
                    tensor, device, producer_stream);
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
                    TransferEngine::publishDeviceWrite(
                        tensor, device, producer_stream);
                }
            }
        }

        return success;
    }

    /**
     * @brief Execute a void direct test kernel with explicit-stream coherence
     *
     * Always marks outputs dirty after execution completes.
     */
    template <typename F>
        requires std::is_invocable_v<F> &&
                 std::is_void_v<std::invoke_result_t<F>>
    bool with_gpu_coherence(
        DeviceId device,
        std::initializer_list<TensorBase *> inputs,
        std::initializer_list<TensorBase *> outputs,
        void *producer_stream,
        F &&kernel_fn)
    {
        if (!producer_stream)
        {
            throw std::invalid_argument(
                "with_gpu_coherence requires the exact non-null producer stream");
        }

        // Join every input producer to the exact consumer stream.
        for (auto *tensor : inputs)
        {
            if (tensor)
                TransferEngine::prepareDeviceInput(
                    tensor, device, producer_stream);
        }

        // Allocate output-only storage without importing stale host contents.
        for (auto *tensor : outputs)
        {
            if (tensor)
                TransferEngine::prepareDeviceOutput(
                    tensor, device, producer_stream);
        }

        // Execute kernel (no return value)
        std::forward<F>(kernel_fn)();

        // Mark outputs dirty
        for (auto *tensor : outputs)
        {
            if (tensor)
            {
                TransferEngine::publishDeviceWrite(
                    tensor, device, producer_stream);
            }
        }

        return true;
    }
} // namespace llaminar2
