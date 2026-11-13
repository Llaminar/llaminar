/**
 * @file DeviceRequestQueryBinding.h
 * @brief Immutable original-request query rows from canonical arena ingress.
 *
 * Non-chunked prefill consumes the request admission bank directly. Its query
 * count belongs to REQUEST_BATCH_GEOMETRY, not KV progress or an arbitrary
 * sequence-length pointer. This checked borrow retains no live count mirror,
 * allocates nothing, and validates original roots/device/runtime on reuse.
 */
#pragma once
#include "backends/IBackend.h"
#include "execution/mtp/MTPRequestTerminalPublicationGraph.h"
#include "memory/BufferArena.h"
#include "tensors/Tensors.h"
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
/** @brief Storage proof for one admitted request bank, excluding its mutable contents. */
class DeviceRequestQueryBinding final
{
public:
    /**
     * @brief Borrow only the canonical already-materialized request roles.
     * @param arena Actual execution arena, which outlives its graphs and this borrow.
     * @param device Exact GPU participant that owns all three original ingress roots.
     * @return Checked source geometry; request publication remains the DGO event authority.
     * @throws std::invalid_argument For external/views, malformed geometry or missing/foreign storage.
     */
    static DeviceRequestQueryBinding fromArena(BufferArena &arena, DeviceId device)
    {
        std::array<TensorBase *,3> roots{};
        std::array<const std::int32_t *,3> bases{};
        constexpr std::array roles{BufferId::REQUEST_TOKEN_IDS,BufferId::REQUEST_POSITION_IDS,BufferId::REQUEST_BATCH_GEOMETRY};
        IBackend *backend=nullptr;
        for(std::size_t i=0;i<roles.size();++i)
        {
            auto *root=arena.getSharedTensor(roles[i]).get();
            if(!device.is_gpu() || !root || arena.getTensor(roles[i])!=root || root->is_view() ||
                root->transferStorageOwner()!=root || root->native_type()!=TensorType::INT32 ||
                root->shape().size()!=2 || root->shape()[0]!=1 || !root->shape()[1] ||
                root->shape()[1]>static_cast<std::size_t>(std::numeric_limits<int>::max()) || root->current_device()!=device)
                throw std::invalid_argument("Request query requires canonical original INT32 arena ingress roots");
            roots[i]=root;bases[i]=static_cast<const std::int32_t *>(arena.getDevicePtr(roles[i],device));
            auto *candidate=root->resolveBackend(device);
            if(!bases[i] || !candidate || (backend && backend!=candidate))
                throw std::invalid_argument("Request query requires prepared same-backend original ingress storage");
            backend=candidate;
        }
        if(roots[0]==roots[1] || roots[0]==roots[2] || roots[1]==roots[2] ||
            bases[0]==bases[1] || bases[0]==bases[2] || bases[1]==bases[2] ||
            roots[0]->shape()!=roots[1]->shape() || roots[2]->shape()[1]<2)
            throw std::invalid_argument("Request query ingress and geometry owners must be distinct and complete");
        const DeviceRequestBatchGeometryLayout layout(static_cast<int>(roots[2]->shape()[1]-1));
        const auto generation=backend->deviceRuntimeGeneration(device.ordinal);
        if(!layout.valid() || !generation)
            throw std::invalid_argument("Request query requires original request geometry and native runtime generation");
        return DeviceRequestQueryBinding(arena,device,roots,bases,layout,backend,generation);
    }
    /**
     * @brief Reject changed original root/bank/runtime geometry without reading any device value.
     * @param device Exact intended projection participant.
     * @throws std::invalid_argument For stale/foreign original root, shape, device or runtime.
     */
    void requireFor(DeviceId device) const
    {
        const auto current=fromArena(*arena_,device);
        if(device!=device_ || current.roots_!=roots_ || current.bases_!=bases_ || current.rows_!=rows_ ||
            current.layout_.requestCapacity()!=layout_.requestCapacity() || current.backend_!=backend_ || current.generation_!=generation_)
            throw std::invalid_argument("Original request query source changed after binding");
    }
    /** @return Original token address, never a copied or reindexed query bank. */
    const std::int32_t *tokens() const noexcept {return bases_[0];}
    /** @return Original absolute position rows paired with token admission. */
    const std::int32_t *positions() const noexcept {return bases_[1];}
    /** @return Original request0 query count; this first slice supports a single request. */
    const std::int32_t *lengths() const noexcept {return layout_.sequenceLengths(bases_[2]);}
    /** @return Full original ingress row capacity, not current live query rows. */
    int rowCapacity() const noexcept {return rows_;}
    /** @return Complete immutable original-source identity; mutable count values are excluded. */
    std::uint64_t captureIdentity() const noexcept
    {
        std::uint64_t hash=1469598103934665603ULL;
        const auto mix=[&](std::uint64_t value){hash^=value;hash*=1099511628211ULL;};
        mix(1);mix(reinterpret_cast<std::uintptr_t>(arena_));mix(static_cast<unsigned>(device_.type));mix(device_.ordinal);
        for(std::size_t i=0;i<roots_.size();++i){mix(reinterpret_cast<std::uintptr_t>(roots_[i]));mix(reinterpret_cast<std::uintptr_t>(bases_[i]));}
        mix(rows_);mix(layout_.requestCapacity());mix(reinterpret_cast<std::uintptr_t>(backend_));mix(generation_);
        return hash?hash:1;
    }
private:
    /**
     * @brief Retain checked immutable metadata while the enclosing DGO retains its actual arena.
     * @param arena Original ingress storage authority. @param device Exact native participant.
     * @param roots Canonical token/position/geometry roots. @param bases Original native addresses.
     * @param layout Source-owned request geometry. @param backend Exact shared native backend.
     * @param generation Authenticated original runtime generation.
     */
    DeviceRequestQueryBinding(BufferArena &arena,DeviceId device,std::array<TensorBase *,3> roots,
        std::array<const std::int32_t *,3> bases,DeviceRequestBatchGeometryLayout layout,IBackend *backend,std::uint64_t generation)
        :arena_(&arena),device_(device),roots_(roots),bases_(bases),rows_(static_cast<int>(roots[0]->shape()[1])),
         layout_(layout),backend_(backend),generation_(generation){}
    BufferArena *arena_;
    DeviceId device_;
    std::array<TensorBase *,3> roots_;
    std::array<const std::int32_t *,3> bases_;
    int rows_;
    DeviceRequestBatchGeometryLayout layout_;
    IBackend *backend_;
    std::uint64_t generation_;
};
} // namespace llaminar2
