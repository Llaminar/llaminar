/**
 * @file CUDAMoEArrivalCapture.h
 * @brief Give isolated CUDA unpack diagnostics an actual retained native owner.
 *
 * A device-updatable work node is not legal as an eager CUDA launch. Fixtures
 * without an enclosing transaction use an isolated native capture and retire it
 * after its one diagnostic replay. Real captured fixtures keep their own owner.
 */
#pragma once

#include "backends/DeviceId.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDAGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#endif

namespace llaminar2::test
{
    /** @brief Execute the real CUDA recording path, using the enclosing graph if present. */
    template <class Record>
    bool recordNativeMoEArrival(DeviceId device, void *stream, Record record)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda())
        {
            auto native = static_cast<cudaStream_t>(stream);
            cudaStreamCaptureStatus status{};
            if (cudaStreamIsCapturing(native, &status) != cudaSuccess)
                return false;
            if (status == cudaStreamCaptureStatusActive)
                return record();
            if (status != cudaStreamCaptureStatusNone)
                return false;
            CUDAGraphCapture owner(native, device.ordinal);
            ScopedBackendGraphCapture transaction(owner, "isolated CUDA rebalance arrival");
            if (!transaction.begin())
                return false;
            const bool recorded = record();
            transaction.finish();
            return recorded && owner.instantiate() && owner.launch() && cudaStreamSynchronize(native) == cudaSuccess;
        }
#endif
        return record();
    }
}
