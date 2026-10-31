/**
 * @file GPUDeviceProcessLifetime.h
 * @brief SDK/core-independent observation boundary for actual executable-main replay.
 *
 * This target-private substitute witnesses public lifetime calls only. It never
 * simulates device discovery, worker teardown or the production authority's
 * internals. The harness records lexical construction, explicit/idempotent
 * retirement and destruction while compiling the complete selected real main.
 */
#pragma once
namespace llaminar2
{
    /** @brief Observe exactly one process-frame owner per replayed child invocation. */
    class GPUDeviceProcessLifetime final
    {
    public:
        GPUDeviceProcessLifetime();
        GPUDeviceProcessLifetime(const GPUDeviceProcessLifetime &) = delete;
        GPUDeviceProcessLifetime &operator=(const GPUDeviceProcessLifetime &) = delete;
        GPUDeviceProcessLifetime(GPUDeviceProcessLifetime &&) = delete;
        GPUDeviceProcessLifetime &operator=(GPUDeviceProcessLifetime &&) = delete;
        ~GPUDeviceProcessLifetime() noexcept;
        void retire() noexcept;
        static void requireActive();
        static void retireActive() noexcept;
    };
}
