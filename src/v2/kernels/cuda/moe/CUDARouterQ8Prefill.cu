/**
 * @file CUDARouterQ8Prefill.cu
 * @brief Geometry-driven tensor-core router launch with fixed block32 arithmetic.
 *
 * Setup authenticates the exact stream and borrows already admitted hardware
 * limits. Captured launches consume that binding and persistent operands. Power-of-two hidden widths
 * permit constant propagation independently of model, expert count and M.
 * Other widths retain the same first-class tensor-core implementation and tree.
 */
#include "CUDARouterQ8Prefill.h"
#include "CUDARouterQ8PrefillDevice.cuh"
#include <algorithm>

namespace llaminar2::cuda
{
    bool prepareRouterQ8PrefillHardware(int device, void *stream, RouterQ8PrefillHardware &result)
    {
        if (device < 0 || !stream) return false;
        const auto native = static_cast<cudaStream_t>(stream);
        cudaStreamCaptureStatus status{};
        int actual_device = -1;
        if (cudaStreamIsCapturing(native, &status) != cudaSuccess || status != cudaStreamCaptureStatusNone ||
            cudaStreamGetDevice(native, &actual_device) != cudaSuccess || actual_device != device) return false;
        return bindRouterQ8PrefillHardware(device, actual_device,
            status != cudaStreamCaptureStatusNone, cudaHardwareMetadata().at(device), result);
    }

    namespace
    {
        /** @brief Submit one exact output/K tile over persistent borrowed banks.
         * @tparam R Row warp tiles. @tparam C Expert warp tiles.
         * @tparam K Log2 of cooperating K warps. @tparam L Log2 of nonempty K leaves.
         * @tparam W Constant hidden width, or zero. @tparam Owned Compact row ownership.
         * @param v Frozen operands. @param rows Local capacity. @param stream Exact stream. */
        template<int R, int C, int K, int L, int W, bool Owned>
        void submit(const RouterQ8PrefillView &v, int rows, cudaStream_t stream)
        {
            router_q8_prefill_detail::kernel<R, C, L, Owned, W, K>
                <<<dim3((v.experts + 8 * C - 1) / (8 * C),
                         (std::max(1, rows) + 16 * R - 1) / (16 * R)), R * C * (1 << K) * 32, 0, stream>>>(
                    v.hidden_codes, v.hidden_scales, v.gate_codes, v.gate_scales,
                    v.logits, v.capacity, v.width, v.experts, v.live_rows, v.partition);
        }

        /** @brief Specialize widths and leaf extents without model-specific expert counts.
         * @tparam R Row warp tiles. @tparam C Expert warp tiles. @tparam K Log2 K warps.
         * @tparam Owned Compact row ownership.
         * @param v Frozen operands. @param rows Local capacity. @param stream Exact stream. */
        template<int R, int C, int K, bool Owned>
        void submitWidth(const RouterQ8PrefillView &v, int rows, cudaStream_t stream)
        {
#define ROUTER_CONSTANT_WIDTH(W, L) if constexpr (K <= L) { if (v.width == W) { submit<R,C,K,L,W,Owned>(v,rows,stream); return; } }
            ROUTER_CONSTANT_WIDTH(32, 0)
            ROUTER_CONSTANT_WIDTH(64, 1)
            ROUTER_CONSTANT_WIDTH(128, 2)
            ROUTER_CONSTANT_WIDTH(256, 3)
            ROUTER_CONSTANT_WIDTH(512, 4)
            ROUTER_CONSTANT_WIDTH(1024, 5)
            ROUTER_CONSTANT_WIDTH(2048, 6)
            ROUTER_CONSTANT_WIDTH(4096, 7)
            ROUTER_CONSTANT_WIDTH(8192, 7)
#undef ROUTER_CONSTANT_WIDTH
#define ROUTER_PARTIAL_WIDTH(W, L) if constexpr (K <= L) { if (v.width <= W) { submit<R,C,K,L,0,Owned>(v,rows,stream); return; } }
            ROUTER_PARTIAL_WIDTH(32, 0)
            ROUTER_PARTIAL_WIDTH(64, 1)
            ROUTER_PARTIAL_WIDTH(128, 2)
            ROUTER_PARTIAL_WIDTH(256, 3)
            ROUTER_PARTIAL_WIDTH(512, 4)
            ROUTER_PARTIAL_WIDTH(1024, 5)
            ROUTER_PARTIAL_WIDTH(2048, 6)
#undef ROUTER_PARTIAL_WIDTH
            submit<R,C,K,7,0,Owned>(v,rows,stream);
        }

        /** @brief Materialize the pure selector's complete typed geometry.
         * @tparam Owned Compact row ownership. @param v Frozen operands.
         * @param rows Local capacity. @param stream Exact stream. */
        template<bool Owned>
        void submitGeometry(const RouterQ8PrefillView &v, int rows, cudaStream_t stream)
        {
            const auto tile = routerQ8PrefillGeometry(rows, v.width, v.experts, v.hardware);
            if (tile.k_warps == 16) submitWidth<1,1,4,Owned>(v,rows,stream);
            else if (tile.k_warps == 8) submitWidth<1,1,3,Owned>(v,rows,stream);
            else if (tile.k_warps == 4) submitWidth<1,1,2,Owned>(v,rows,stream);
            else if (tile.k_warps == 2) submitWidth<1,1,1,Owned>(v,rows,stream);
            else if (tile.rows == 64) submitWidth<4,1,0,Owned>(v,rows,stream);
            else if (tile.rows == 32 && tile.experts == 16) submitWidth<2,2,0,Owned>(v,rows,stream);
            else if (tile.rows == 32) submitWidth<2,1,0,Owned>(v,rows,stream);
            else if (tile.experts == 32) submitWidth<1,4,0,Owned>(v,rows,stream);
            else if (tile.experts == 16) submitWidth<1,2,0,Owned>(v,rows,stream);
            else submitWidth<1,1,0,Owned>(v,rows,stream);
        }
    }

    bool routerQ8Prefill(const RouterQ8PrefillView &view, void *stream)
    {
        if (!stream || !view.valid()) return false;
        const int rows = view.partition.capacityFor(view.capacity);
        const auto native = static_cast<cudaStream_t>(stream);
        if (view.partition.participants() > 1) submitGeometry<true>(view, rows, native);
        else submitGeometry<false>(view, rows, native);
        return cudaGetLastError() == cudaSuccess;
    }
}
