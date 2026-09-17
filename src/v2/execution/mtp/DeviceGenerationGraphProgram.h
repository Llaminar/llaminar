/**
 * @file DeviceGenerationGraphProgram.h
 * @brief Algorithm-independent compilation of resident generation parents.
 *
 * Request/controller storage and graph lifetime remain with their existing
 * owners. This builder only lowers a complete declared program: a native
 * device-controlled loop, or an isolated scheduler-ticket publisher. Neither
 * form chooses a different execution policy when construction fails.
 */
#pragma once

#include "backends/DeviceId.h"
#include "backends/IGPUGraphCapture.h"
#include "kernels/common/SamplingMath.h"

#include <span>
#include <string>

namespace llaminar2
{
class IBackend;
class IWorkerGPUContext;

/** @brief Borrowed complete controller geometry and immutable admitted algorithm. */
struct DeviceGenerationGraphControl
{
    sampling_math::DeviceGenerationPolicy policy;
    int *rows = nullptr; ///< Persistent device ABI rows, never host mirrors.
    int stride = 0; ///< INT32 elements between independent requests.
    int requests = 0; ///< Active controller rows.

    /** @return Whether the declared algorithm and physical geometry are usable. */
    [[nodiscard]] bool valid() const noexcept;
};

/**
 * @brief Stateless compiler shared by ordinary and speculative generation.
 *
 * All addresses are borrowed from already-admitted storage. A successful call
 * leaves one instantiated, unlaunched executable. Reuse/reset remains the
 * existing graph owner's responsibility; this API rejects replacing a live
 * executable instead of silently recapturing it. It does not allocate request
 * buffers, inspect device data, launch inference or synchronize any stream.
 */
class DeviceGenerationGraphProgram final
{
public:
    /**
     * @brief Compile one complete transaction into a device-owned native loop.
     * @param destination Empty parent owner with its exact non-null stream.
     * @param control Borrowed controller ABI and admitted algorithm.
     * @param program Captured arrival, admitted initialization and repeated transaction.
     * @param error Precise first construction error, cleared on success.
     * @return True only after the complete native executable is ready.
     *
     * Ordinary/forward-only and fixed MTP use the same continuation predicate.
     * Dynamic MTP adds selector validation without changing controller ownership.
     * Children must be recorded but need not be independently instantiated:
     * composition consumes their native graph, while the parent owns execution.
     */
    [[nodiscard]] static bool native(
        IGPUGraphCapture &destination,
        const DeviceGenerationGraphControl &control,
        const DeviceControlledLoopProgram &program,
        std::string &error);

    /**
     * @brief Compile only the immutable ticket publisher for a declared hosted policy.
     * @param destination Empty publisher owner on the scheduler's exact stream.
     * @param context Worker owning the backend's capture-active lifecycle.
     * @param backend Backend of the owning device.
     * @param device Exact GPU ordinal on which all borrowed storage lives.
     * @param control Borrowed controller ABI and admitted algorithm.
     * @param maintenance_due Optional completed-transaction maintenance predicate.
     * @param tickets Persistent fixed-size ticket rows; no response/cache pointers.
     * @param error Precise first construction error, cleared on success.
     * @return True only after the isolated publisher is instantiated.
     *
     * The existing scheduler retains and authenticates complete transaction
     * branches. This publisher does not perform or decide model work. RAII
     * closes native capture on enqueue failure or exception, so a local error
     * cannot strand the stream in capture mode and poison unrelated requests.
     */
    [[nodiscard]] static bool ticketPublisher(
        IGPUGraphCapture &destination,
        IWorkerGPUContext &context,
        IBackend &backend,
        DeviceId device,
        const DeviceGenerationGraphControl &control,
        const uint32_t *maintenance_due,
        sampling_math::DeviceGenerationDispatchTicket *tickets,
        std::string &error);
};
} // namespace llaminar2
