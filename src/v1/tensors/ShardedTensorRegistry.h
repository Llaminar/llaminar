// Sharded Tensor Registry
// Tracks live sharded tensor instances for diagnostics / topology printing.
#pragma once
#include <vector>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <algorithm>
#include <unordered_map>
#include <sstream>
#include "TensorFactory.h"

namespace llaminar
{

    class ShardedSimpleTensor; // fwd

    struct ShardedTensorRecord
    {
        std::weak_ptr<ShardedSimpleTensor> tensor;
    };

    class ShardedTensorRegistry
    {
    public:
        static ShardedTensorRegistry &instance()
        {
            static ShardedTensorRegistry r;
            return r;
        }

        void register_tensor(const std::shared_ptr<ShardedSimpleTensor> &t)
        {
            // Exclusive writer lock
            std::unique_lock<std::shared_mutex> lock(m_);
            records_.push_back({t});
            last_ = t;
            prune_locked(); // prune under exclusive lock
        }
        void prune()
        {
            std::unique_lock<std::shared_mutex> lock(m_);
            prune_locked();
        }

        template <typename Fn>
        void for_each(Fn fn)
        {
            // Shared reader lock: skip pruning to avoid write contention, expired entries ignored.
            std::shared_lock<std::shared_mutex> lock(m_);
            for (auto &rec : records_)
            {
                if (auto sp = rec.tensor.lock())
                    fn(*sp);
            }
        }

        // ---------------- Snapshot / Metrics API ----------------
        struct ShardDetail
        {
            ShardSpec spec;               // copy of the shard spec at snapshot time
            std::vector<int> local_shape; // local shape of the tensor on this rank
            size_t local_elems = 0;       // product(local_shape)
            size_t local_bytes = 0;       // assuming fp32 (SimpleTensor backing)
        };
        struct AxisMetrics
        {
            ShardSpec::Axis axis = ShardSpec::Axis::None;
            size_t count = 0;       // number of shard tensors using this axis
            size_t local_elems = 0; // aggregate local elems across tensors
            size_t global_dim = 0;  // last observed global_dim (should be consistent for axis)
        };
        struct Snapshot
        {
            std::vector<ShardDetail> details;
            size_t total_local_elems = 0;
            size_t total_local_bytes = 0;
            std::vector<AxisMetrics> per_axis; // one entry per axis present

            std::string summary_line() const
            {
                std::ostringstream oss;
                double mb = total_local_bytes / (1024.0 * 1024.0);
                oss << "Aggregate shard footprint: local_bytes=" << total_local_bytes
                    << " (" << mb << " MB) local_elems=" << total_local_elems;
                return oss.str();
            }
        };

        Snapshot snapshot()
        {
            // Shared reader lock: non-intrusive; does not mutate registry (no pruning) for concurrency.
            std::shared_lock<std::shared_mutex> lock(m_);
            Snapshot snap;
            std::unordered_map<int, AxisMetrics> axis_map; // key = static_cast<int>(axis)
            for (auto &rec : records_)
            {
                if (auto sp = rec.tensor.lock())
                {
                    ShardDetail d;
                    d.spec = sp->shard_spec();
                    d.local_shape = sp->shape();
                    size_t elems = 1;
                    for (int v : d.local_shape)
                        elems *= (size_t)v;
                    d.local_elems = elems;
                    d.local_bytes = elems * sizeof(float);
                    snap.total_local_elems += d.local_elems;
                    snap.total_local_bytes += d.local_bytes;
                    snap.details.push_back(std::move(d));
                    if (sp->shard_spec().is_sharded())
                    {
                        int key = static_cast<int>(sp->shard_spec().axis);
                        auto &m = axis_map[key];
                        m.axis = sp->shard_spec().axis;
                        m.count += 1;
                        m.local_elems += elems;
                        m.global_dim = sp->shard_spec().global_dim; // last wins (should be consistent)
                    }
                }
            }
            // move axis metrics into vector with deterministic axis ordering (Hidden, Heads)
            auto push_if = [&](ShardSpec::Axis ax)
            {
                int key = static_cast<int>(ax);
                auto it = axis_map.find(key);
                if (it != axis_map.end())
                    snap.per_axis.push_back(it->second);
            };
            push_if(ShardSpec::Axis::Hidden);
            push_if(ShardSpec::Axis::Heads);
            return snap;
        }

    private:
        void prune_locked()
        {
            records_.erase(std::remove_if(records_.begin(), records_.end(), [](auto &r)
                                          { return r.tensor.expired(); }),
                           records_.end());
        }
        mutable std::shared_mutex m_;
        std::vector<ShardedTensorRecord> records_;
        std::weak_ptr<ShardedSimpleTensor> last_;

    public:
        std::shared_ptr<ShardedSimpleTensor> last()
        {
            std::shared_lock<std::shared_mutex> lock(m_);
            return last_.lock();
        }
    };

} // namespace llaminar
