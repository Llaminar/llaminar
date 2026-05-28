/**
 * @file CUDAKernelProfiler.cu
 * @brief CUDA event-based kernel timing implementation
 * @author David Sanftenberg
 *
 * Implements CUDA event-based timing for accurate GPU kernel profiling.
 */

#include "CUDAKernelProfiler.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace llaminar2
{

    // ========================================================================
    // ScopedCUDAKernelTimer Implementation
    // ========================================================================

    ScopedCUDAKernelTimer::ScopedCUDAKernelTimer(CUDAKernelType type, cudaStream_t stream)
        : type_(type), enabled_(CUDAKernelProfiler::isEnabled()),
          start_event_(nullptr), stop_event_(nullptr), stream_(stream)
    {
        if (enabled_)
        {
            // Skip event timing if the stream is currently in graph capture mode.
            // cudaEventRecord/cudaEventSynchronize are not permitted on capturing
            // streams and would poison the capture context.
            cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
            if (stream_ && cudaStreamIsCapturing(stream_, &capture_status) == cudaSuccess &&
                capture_status != cudaStreamCaptureStatusNone)
            {
                enabled_ = false;
                return;
            }

            cudaError_t err;
            err = cudaEventCreate(&start_event_);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "[CUDAKernelProfiler] Failed to create start event: %s\n",
                        cudaGetErrorString(err));
                enabled_ = false;
                return;
            }

            err = cudaEventCreate(&stop_event_);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "[CUDAKernelProfiler] Failed to create stop event: %s\n",
                        cudaGetErrorString(err));
                cudaEventDestroy(start_event_);
                enabled_ = false;
                return;
            }

            // Record start event on the stream
            err = cudaEventRecord(start_event_, stream_);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "[CUDAKernelProfiler] Failed to record start event: %s\n",
                        cudaGetErrorString(err));
                cudaEventDestroy(start_event_);
                cudaEventDestroy(stop_event_);
                enabled_ = false;
            }
        }
    }

    ScopedCUDAKernelTimer::~ScopedCUDAKernelTimer()
    {
        if (enabled_)
        {
            // Record stop event
            cudaError_t err = cudaEventRecord(stop_event_, stream_);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "[CUDAKernelProfiler] Failed to record stop event: %s\n",
                        cudaGetErrorString(err));
                cudaEventDestroy(start_event_);
                cudaEventDestroy(stop_event_);
                return;
            }

            // Synchronize on stop event to get accurate timing
            err = cudaEventSynchronize(stop_event_);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "[CUDAKernelProfiler] Failed to synchronize stop event: %s\n",
                        cudaGetErrorString(err));
                cudaEventDestroy(start_event_);
                cudaEventDestroy(stop_event_);
                return;
            }

            // Calculate elapsed time in milliseconds
            float elapsed_ms = 0.0f;
            err = cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
            if (err == cudaSuccess)
            {
                // Convert to microseconds and record
                CUDAKernelProfiler::record(type_, static_cast<double>(elapsed_ms) * 1000.0);
            }
            else
            {
                fprintf(stderr, "[CUDAKernelProfiler] Failed to get elapsed time: %s\n",
                        cudaGetErrorString(err));
            }

            // Clean up events
            cudaEventDestroy(start_event_);
            cudaEventDestroy(stop_event_);
        }
    }

    // ========================================================================
    // ManualCUDAKernelTimer Implementation
    // ========================================================================

    ManualCUDAKernelTimer::ManualCUDAKernelTimer()
        : enabled_(CUDAKernelProfiler::isEnabled()), started_(false),
          start_event_(nullptr), stop_event_(nullptr), stream_(nullptr)
    {
        if (enabled_)
        {
            cudaError_t err;
            err = cudaEventCreate(&start_event_);
            if (err != cudaSuccess)
            {
                enabled_ = false;
                return;
            }

            err = cudaEventCreate(&stop_event_);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(start_event_);
                enabled_ = false;
            }
        }
    }

    ManualCUDAKernelTimer::~ManualCUDAKernelTimer()
    {
        if (enabled_)
        {
            if (start_event_)
                cudaEventDestroy(start_event_);
            if (stop_event_)
                cudaEventDestroy(stop_event_);
        }
    }

    void ManualCUDAKernelTimer::begin(cudaStream_t stream)
    {
        if (enabled_ && !started_)
        {
            // Skip if the stream is in graph capture mode — event timing is
            // incompatible with stream capture.
            cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
            if (stream && cudaStreamIsCapturing(stream, &capture_status) == cudaSuccess &&
                capture_status != cudaStreamCaptureStatusNone)
            {
                return;
            }

            stream_ = stream;
            cudaError_t err = cudaEventRecord(start_event_, stream_);
            if (err == cudaSuccess)
            {
                started_ = true;
            }
        }
    }

    void ManualCUDAKernelTimer::end(CUDAKernelType type)
    {
        if (enabled_ && started_)
        {
            started_ = false;

            // Record stop event
            cudaError_t err = cudaEventRecord(stop_event_, stream_);
            if (err != cudaSuccess)
                return;

            // Synchronize
            err = cudaEventSynchronize(stop_event_);
            if (err != cudaSuccess)
                return;

            // Get elapsed time
            float elapsed_ms = 0.0f;
            err = cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
            if (err == cudaSuccess)
            {
                CUDAKernelProfiler::record(type, static_cast<double>(elapsed_ms) * 1000.0);
            }
        }
    }

} // namespace llaminar2
