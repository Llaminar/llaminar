/**
 * @file MoERebalanceController.cpp
 * @brief Orchestrates MoE decode histogram tracking and socket-aware rebalancing
 */

#include "MoERebalanceController.h"
#include "../../utils/Logger.h"
#include "fort.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llaminar2
{
    const char *toString(MoERebalanceDecisionReason reason)
    {
        switch (reason)
        {
        case MoERebalanceDecisionReason::ModeOff:
            return "mode_off";
        case MoERebalanceDecisionReason::DynamicDisabledForDomain:
            return "dynamic_disabled_for_domain";
        case MoERebalanceDecisionReason::SingleParticipantObserveOnly:
            return "single_participant_observe_only";
        case MoERebalanceDecisionReason::WindowNotFull:
            return "window_not_full";
        case MoERebalanceDecisionReason::Ready:
            return "ready";
        }
        return "unknown";
    }

    // =========================================================================
    // ExpertReplicaSet — deterministic per-token dispatch
    // =========================================================================

    void ExpertReplicaSet::assignForToken(
        const int *expert_indices,
        const float * /*expert_weights*/,
        int top_k,
        int my_socket_id,
        const std::vector<bool> &expert_mask,
        bool *compute_here) const
    {
        if (num_replicated == 0)
        {
            // No replicas — simple mask check
            for (int k = 0; k < top_k; ++k)
                compute_here[k] = expert_mask[expert_indices[k]];
            return;
        }

        // Phase 1: Count fixed assignments (non-replicated experts go to owner)
        int load[8] = {};  // Per-socket load counter (max 8 sockets)
        bool is_fixed[16]; // Stack-allocated, max top_k

        for (int k = 0; k < top_k; ++k)
        {
            int e = expert_indices[k];
            if (!is_replicated[e])
            {
                is_fixed[k] = true;
                load[owner_socket[e]]++;
            }
            else
            {
                is_fixed[k] = false;
            }
        }

        // Phase 2: Greedily assign replicated experts to least-loaded socket.
        // Process in index order for determinism (both ranks see same routing).
        for (int k = 0; k < top_k; ++k)
        {
            if (is_fixed[k])
                continue;

            int e = expert_indices[k];
            int owner = owner_socket[e];

            // Find the socket with the lowest current load.
            // Break ties: prefer the owner socket (avoids unnecessary replica use).
            int best = owner;
            for (int s = 0; s < num_sockets; ++s)
            {
                if (load[s] < load[best] || (load[s] == load[best] && s == owner))
                    best = s;
            }

            // Assign to chosen socket
            load[best]++;
            is_fixed[k] = true; // Mark as assigned

            // Store assignment result directly
            compute_here[k] = (best == my_socket_id);
        }

        // Phase 3: Fill in non-replicated assignment results
        for (int k = 0; k < top_k; ++k)
        {
            int e = expert_indices[k];
            if (!is_replicated[e])
                compute_here[k] = (owner_socket[e] == my_socket_id);
        }
    }

    void ExpertReplicaSet::buildPrefillMask(int my_socket_id, const std::vector<bool> &expert_mask)
    {
        prefill_mask.resize(expert_mask.size());
        for (size_t e = 0; e < expert_mask.size(); ++e)
        {
            // During prefill, only the owner socket processes replicated experts.
            // Non-replicated experts use the standard expert_mask.
            prefill_mask[e] = expert_mask[e] &&
                              (!is_replicated[e] || owner_socket[e] == my_socket_id);
        }
    }

    bool ExpertReplicaSet::sameReplicaPlacement(const ExpertReplicaSet &other) const
    {
        return domain_id == other.domain_id &&
               num_replicated == other.num_replicated &&
               num_sockets == other.num_sockets &&
               is_replicated == other.is_replicated &&
               owner_socket == other.owner_socket;
    }

    ExpertReplicaSet ExpertReplicaSet::arrivalsSince(const ExpertReplicaSet &previous) const
    {
        ExpertReplicaSet arrivals;
        arrivals.domain_id = domain_id;
        arrivals.is_replicated.assign(is_replicated.size(), false);
        arrivals.owner_socket = owner_socket;
        arrivals.num_sockets = num_sockets;

        for (size_t expert = 0; expert < is_replicated.size(); ++expert)
        {
            if (!is_replicated[expert])
                continue;

            const bool previously_resident =
                previous.num_sockets == num_sockets &&
                expert < previous.is_replicated.size() &&
                previous.is_replicated[expert] &&
                expert < previous.owner_socket.size() &&
                expert < owner_socket.size() &&
                previous.owner_socket[expert] == owner_socket[expert];

            if (!previously_resident)
            {
                arrivals.is_replicated[expert] = true;
                ++arrivals.num_replicated;
            }
        }

        return arrivals;
    }

    MoERebalanceController::MoERebalanceController(Config config)
        : requested_mode_(config.mode),
          config_(std::move(config)),
          current_placement_(config_.initial_expert_to_socket),
          current_window_size_(config_.window_size)
    {
        current_replicas_.domain_id = config_.domain_id;

        if (config_.mode == MoERebalanceMode::DYNAMIC && config_.sockets.size() < 2)
        {
            config_.mode = MoERebalanceMode::OBSERVE;
            LOG_INFO("[MoERebalanceController] Dynamic rebalance downgraded to OBSERVE for single-participant domain");
        }

        if (config_.mode == MoERebalanceMode::OFF)
            return;

        // Create histogram for OBSERVE and DYNAMIC modes
        DecodeExpertHistogramConfig hcfg;
        hcfg.num_layers = config_.num_layers;
        hcfg.num_experts = config_.num_experts;
        hcfg.top_k = config_.top_k;
        hcfg.window_size = config_.window_size;
        hcfg.sockets = config_.sockets;
        hcfg.expert_to_socket = config_.initial_expert_to_socket;
        histogram_ = std::make_unique<DecodeExpertHistogram>(std::move(hcfg));

        // Create rebalancer for DYNAMIC mode
        if (config_.mode == MoERebalanceMode::DYNAMIC)
        {
            rebalancer_ = std::make_unique<SocketAwareRebalancer>(config_.rebalance_config);
        }

        LOG_DEBUG("[MoERebalanceController] Initialized: mode="
                  << (config_.mode == MoERebalanceMode::OBSERVE ? "OBSERVE" : "DYNAMIC")
                  << " layers=" << config_.num_layers
                  << " experts=" << config_.num_experts
                  << " top_k=" << config_.top_k
                  << " window=" << config_.window_size
                  << " sockets=" << config_.sockets.size());
    }

    bool MoERebalanceController::shouldRebalance() const
    {
        return rebalanceDecision().ready;
    }

    MoERebalanceDecision MoERebalanceController::rebalanceDecision() const
    {
        if (requested_mode_ == MoERebalanceMode::OFF || config_.mode == MoERebalanceMode::OFF)
        {
            return {false, MoERebalanceDecisionReason::ModeOff};
        }

        if (requested_mode_ == MoERebalanceMode::DYNAMIC && config_.sockets.size() < 2)
        {
            return {false, MoERebalanceDecisionReason::SingleParticipantObserveOnly};
        }

        if (config_.mode != MoERebalanceMode::DYNAMIC)
        {
            return {false, MoERebalanceDecisionReason::DynamicDisabledForDomain};
        }

        if (!histogram_ || !histogram_->windowFull())
        {
            return {false, MoERebalanceDecisionReason::WindowNotFull};
        }

        return {true, MoERebalanceDecisionReason::Ready};
    }

    std::vector<int> MoERebalanceController::rebalance()
    {
        if (!rebalancer_ || !histogram_)
            return {};

        auto proposal = rebalancer_->propose(*histogram_);

        if (proposal.empty())
        {
            LOG_DEBUG("[MoERebalanceController] No beneficial swaps found, resetting window");
            histogram_->resetWindow();
            growWindowIfAdaptive();
            return {};
        }

        // Apply swaps to get new placement
        auto new_placement = rebalancer_->apply(current_placement_, proposal);
        current_placement_ = new_placement;
        ++placement_epoch_;

        // Update histogram's socket mapping for next window
        histogram_->updatePlacement(current_placement_);
        histogram_->resetWindow();
        growWindowIfAdaptive();

        total_rebalances_++;
        total_swaps_ += proposal.numSwaps();

        LOG_DEBUG("[MoERebalanceController] Rebalance #" << total_rebalances_
                                                         << ": " << proposal.summary());

        return new_placement;
    }

    void MoERebalanceController::resetRebalanceWindow()
    {
        if (!histogram_)
            return;
        histogram_->resetWindow();
        growWindowIfAdaptive();
    }

    void MoERebalanceController::logHistogramSummary() const
    {
        if (!histogram_)
        {
            LOG_DEBUG("[MoERebalanceController] No histogram (mode=OFF)");
            return;
        }

        LOG_DEBUG("[MoERebalanceController] Histogram summary (window tokens="
                  << histogram_->windowTokenCount()
                  << " gen=" << histogram_->windowGeneration()
                  << " avg_imbalance=" << histogram_->averageSocketImbalance() << "):");

        for (int l = 0; l < config_.num_layers; ++l)
        {
            LOG_DEBUG("  " << histogram_->layerSummary(l));
        }
    }

    std::vector<std::vector<bool>> MoERebalanceController::computeExpertMasks(int socket_id) const
    {
        const int num_layers = config_.num_layers;
        const int num_experts = config_.num_experts;

        std::vector<std::vector<bool>> masks(num_layers);

        if (use_per_layer_placement_ && !per_layer_placement_.empty())
        {
            // Per-layer placement (after rebalanceLPT)
            for (int l = 0; l < num_layers; ++l)
            {
                masks[l].resize(num_experts, false);
                for (int e = 0; e < num_experts; ++e)
                {
                    if (per_layer_placement_[l][e] == socket_id)
                        masks[l][e] = true;
                }
            }
        }
        else
        {
            // Global placement (initial or after swap-based rebalance)
            std::vector<bool> global_mask(num_experts, false);
            for (int e = 0; e < num_experts; ++e)
            {
                if (e < static_cast<int>(current_placement_.size()) && current_placement_[e] == socket_id)
                    global_mask[e] = true;
            }
            masks.assign(num_layers, global_mask);
        }

        // If replicas are active, expand masks to include replicated experts
        // that this socket should also have GEMM engines for.
        if (current_replicas_.num_replicated > 0)
        {
            for (int l = 0; l < num_layers; ++l)
            {
                for (int e = 0; e < num_experts; ++e)
                {
                    if (current_replicas_.is_replicated[e])
                        masks[l][e] = true; // Both sockets get this expert
                }
            }
        }

        return masks;
    }

    std::vector<std::vector<std::vector<bool>>> MoERebalanceController::computeGpuCacheExpertMasks(
        int gpu_cache_experts_per_layer) const
    {
        const int num_layers = config_.num_layers;
        const int num_experts = config_.num_experts;
        const int num_sockets = static_cast<int>(config_.sockets.size());

        std::vector<std::vector<std::vector<bool>>> masks_by_socket(num_sockets);
        for (int s = 0; s < num_sockets; ++s)
            masks_by_socket[s].assign(num_layers, std::vector<bool>(num_experts, false));

        auto fallback = [&]()
        {
            for (int s = 0; s < num_sockets; ++s)
                masks_by_socket[s] = computeExpertMasks(s);
            return masks_by_socket;
        };

        if (gpu_cache_experts_per_layer <= 0 || !histogram_ || num_layers <= 0 || num_experts <= 0)
            return fallback();

        std::vector<int> gpu_sockets;
        std::vector<int> cpu_sockets;
        for (int s = 0; s < num_sockets; ++s)
        {
            if (config_.sockets[s].is_gpu())
                gpu_sockets.push_back(s);
            else
                cpu_sockets.push_back(s);
        }

        if (gpu_sockets.empty() || cpu_sockets.empty())
            return fallback();

        const int cache_count = std::min(gpu_cache_experts_per_layer, num_experts);
        int total_gpu_assignments = 0;

        for (int l = 0; l < num_layers; ++l)
        {
            auto counts = histogram_->layerHistogram(l);
            std::vector<int> experts(num_experts);
            std::iota(experts.begin(), experts.end(), 0);
            std::sort(experts.begin(), experts.end(),
                      [&](int a, int b)
                      {
                          if (counts[a] != counts[b])
                              return counts[a] > counts[b];
                          return a < b;
                      });

            std::vector<bool> gpu_cached(num_experts, false);
            for (int i = 0; i < cache_count; ++i)
            {
                const int expert = experts[i];
                gpu_cached[expert] = true;
            }

            auto assign_lpt = [&](const std::vector<int> &socket_ids,
                                  const std::vector<int> &expert_ids)
            {
                std::vector<uint64_t> loads(socket_ids.size(), 0);
                std::vector<int> expert_counts(socket_ids.size(), 0);
                for (int expert : expert_ids)
                {
                    size_t best = 0;
                    for (size_t i = 1; i < socket_ids.size(); ++i)
                    {
                        if (loads[i] < loads[best] ||
                            (loads[i] == loads[best] && expert_counts[i] < expert_counts[best]))
                        {
                            best = i;
                        }
                    }
                    const int socket = socket_ids[best];
                    masks_by_socket[socket][l][expert] = true;
                    loads[best] += counts[expert];
                    expert_counts[best]++;
                }
            };

            std::vector<int> hot_experts;
            hot_experts.reserve(cache_count);
            for (int expert : experts)
            {
                if (gpu_cached[expert])
                    hot_experts.push_back(expert);
            }

            std::vector<int> cold_experts;
            cold_experts.reserve(num_experts - cache_count);
            for (int expert : experts)
            {
                if (!gpu_cached[expert])
                    cold_experts.push_back(expert);
            }

            assign_lpt(gpu_sockets, hot_experts);
            assign_lpt(cpu_sockets, cold_experts);
            total_gpu_assignments += static_cast<int>(hot_experts.size());
        }

        LOG_DEBUG("[MoERebalanceController] GPU expert cache masks: "
                  << cache_count << "/" << num_experts << " routed experts per layer on GPU domain"
                  << " (gpu_sockets=" << gpu_sockets.size()
                  << ", cpu_sockets=" << cpu_sockets.size()
                  << ", assignments=" << total_gpu_assignments << ")");

        return masks_by_socket;
    }

    void MoERebalanceController::rebalanceLPT()
    {
        if (!histogram_)
            return;

        auto t_start = std::chrono::high_resolution_clock::now();

        const int num_layers = config_.num_layers;
        const int num_experts = config_.num_experts;
        const int num_sockets = static_cast<int>(config_.sockets.size());

        if (num_sockets < 2)
            return;

        // Aggregate per-expert activation counts across ALL layers.
        // This produces a single global partition optimized for the average case,
        // ensuring each rank still has ~N/2 experts and GEMM engines (no 2x memory).
        std::vector<uint64_t> total_counts(num_experts, 0);
        for (int l = 0; l < num_layers; ++l)
        {
            auto layer_counts = histogram_->layerHistogram(l);
            for (int e = 0; e < num_experts; ++e)
                total_counts[e] += layer_counts[e];
        }

        // Compute imbalance with initial contiguous partition
        std::vector<uint64_t> old_loads(num_sockets, 0);
        for (int e = 0; e < num_experts; ++e)
            old_loads[current_placement_[e]] += total_counts[e];
        auto [old_min_it, old_max_it] = std::minmax_element(old_loads.begin(), old_loads.end());
        float imbalance_before = (*old_min_it > 0)
                                     ? static_cast<float>(*old_max_it) / static_cast<float>(*old_min_it)
                                     : 1.0f;

        // LPT: sort experts by total activation count descending
        std::vector<int> sorted(num_experts);
        std::iota(sorted.begin(), sorted.end(), 0);
        std::sort(sorted.begin(), sorted.end(),
                  [&](int a, int b)
                  { return total_counts[a] > total_counts[b]; });

        // Greedy assign to least-loaded socket
        std::vector<uint64_t> loads(num_sockets, 0);
        std::vector<int> new_placement(num_experts);

        for (int idx = 0; idx < num_experts; ++idx)
        {
            int e = sorted[idx];
            int target = 0;
            for (int s = 1; s < num_sockets; ++s)
            {
                if (loads[s] < loads[target])
                    target = s;
            }
            new_placement[e] = target;
            loads[target] += total_counts[e];
        }

        // Compute imbalance after
        auto [new_min_it, new_max_it] = std::minmax_element(loads.begin(), loads.end());
        float imbalance_after = (*new_min_it > 0)
                                    ? static_cast<float>(*new_max_it) / static_cast<float>(*new_min_it)
                                    : 1.0f;

        // Count how many experts changed socket
        int experts_moved = 0;
        for (int e = 0; e < num_experts; ++e)
        {
            if (new_placement[e] != current_placement_[e])
                experts_moved++;
        }

        // Also compute per-layer imbalance improvement
        float per_layer_before = 0.0f, per_layer_after = 0.0f;
        float worst_before = 0.0f;
        int worst_layer_before = 0;
        for (int l = 0; l < num_layers; ++l)
        {
            auto lc = histogram_->layerHistogram(l);
            std::vector<uint64_t> lb(num_sockets, 0), la(num_sockets, 0);
            for (int e = 0; e < num_experts; ++e)
            {
                lb[current_placement_[e]] += lc[e];
                la[new_placement[e]] += lc[e];
            }
            auto [lb_min, lb_max] = std::minmax_element(lb.begin(), lb.end());
            auto [la_min, la_max] = std::minmax_element(la.begin(), la.end());
            float imb_b = (*lb_min > 0) ? float(*lb_max) / float(*lb_min) : 1.0f;
            float imb_a = (*la_min > 0) ? float(*la_max) / float(*la_min) : 1.0f;
            per_layer_before += imb_b;
            per_layer_after += imb_a;
            if (imb_b > worst_before)
            {
                worst_before = imb_b;
                worst_layer_before = l;
            }
        }
        per_layer_before /= num_layers;
        per_layer_after /= num_layers;

        current_placement_ = new_placement;
        if (experts_moved > 0)
            ++placement_epoch_;
        use_per_layer_placement_ = false; // global partition, not per-layer
        total_rebalances_++;
        last_experts_moved_ = experts_moved;
        last_avg_imbalance_before_ = per_layer_before;
        last_avg_imbalance_after_ = per_layer_after;
        last_worst_imbalance_before_ = worst_before;
        last_worst_layer_before_ = worst_layer_before;

        auto t_end = std::chrono::high_resolution_clock::now();
        last_rebalance_duration_ms_ = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        LOG_DEBUG("[MoERebalanceController] LPT global rebalance: "
                  << experts_moved << "/" << num_experts << " experts moved, "
                  << "per-layer avg imbalance " << std::fixed << std::setprecision(3)
                  << per_layer_before << "x -> " << per_layer_after << "x"
                  << " (worst: " << worst_before << "x layer " << worst_layer_before << ")"
                  << " in " << std::setprecision(2) << last_rebalance_duration_ms_ << " ms");

        histogram_->resetWindow();
        growWindowIfAdaptive();
    }

    void MoERebalanceController::growWindowIfAdaptive()
    {
        if (config_.max_window_size <= 0 || config_.window_growth_factor <= 1.0f)
            return;

        int new_size = static_cast<int>(current_window_size_ * config_.window_growth_factor);
        new_size = std::min(new_size, config_.max_window_size);
        if (new_size > current_window_size_)
        {
            LOG_DEBUG("[MoERebalanceController] Adaptive window: "
                      << current_window_size_ << " -> " << new_size);
            current_window_size_ = new_size;
            histogram_->setWindowSize(new_size);
        }
    }

    ExpertReplicaSet MoERebalanceController::proposeReplicas(int max_replicas_per_socket)
    {
        ExpertReplicaSet result;
        result.domain_id = config_.domain_id;
        result.is_replicated.resize(config_.num_experts, false);
        result.owner_socket = current_placement_;
        result.num_replicated = 0;
        result.num_sockets = static_cast<int>(config_.sockets.size());

        if (!histogram_ || max_replicas_per_socket <= 0 || config_.sockets.size() < 2)
            return result;

        const int num_experts = config_.num_experts;
        const int num_sockets = result.num_sockets;

        // Aggregate per-expert activation counts across all layers
        std::vector<uint64_t> total_counts(num_experts, 0);
        for (int l = 0; l < config_.num_layers; ++l)
        {
            auto layer_counts = histogram_->layerHistogram(l);
            for (int e = 0; e < num_experts; ++e)
                total_counts[e] += layer_counts[e];
        }

        // For each socket, find the hottest experts on OTHER sockets to replicate locally.
        for (int target_socket = 0; target_socket < num_sockets; ++target_socket)
        {
            // Collect experts NOT owned by this socket, sorted by count descending
            std::vector<int> candidates;
            for (int e = 0; e < num_experts; ++e)
            {
                if (current_placement_[e] != target_socket && total_counts[e] > 0)
                    candidates.push_back(e);
            }

            std::sort(candidates.begin(), candidates.end(),
                      [&](int a, int b)
                      { return total_counts[a] > total_counts[b]; });

            // Mark the top-K as replicated
            int replicated = 0;
            for (int e : candidates)
            {
                if (replicated >= max_replicas_per_socket)
                    break;
                if (result.is_replicated[e])
                    continue; // Already marked by another socket
                result.is_replicated[e] = true;
                result.num_replicated++;
                replicated++;
            }
        }

        const bool replica_placement_changed = !current_replicas_.sameReplicaPlacement(result);
        current_replicas_ = result;
        if (replica_placement_changed && result.num_replicated > 0)
            ++placement_epoch_;

        LOG_DEBUG("[MoERebalanceController] Proposed " << result.num_replicated
                                                       << " expert replicas (max " << max_replicas_per_socket << " per socket)");

        // Log the top replicas per socket
        for (int s = 0; s < num_sockets; ++s)
        {
            std::ostringstream oss;
            oss << "  Socket " << s << " gets replicas: ";
            int count = 0;
            for (int e = 0; e < num_experts; ++e)
            {
                // Expert replicated on socket s = expert NOT owned by s but is_replicated
                if (result.is_replicated[e] && result.owner_socket[e] != s)
                {
                    if (count > 0)
                        oss << ", ";
                    oss << "e" << e << "(" << total_counts[e] << ")";
                    count++;
                }
            }
            LOG_DEBUG("[MoERebalanceController] " << oss.str());
        }

        return result;
    }

    std::string MoERebalanceController::getProfilingSummary() const
    {
        std::ostringstream oss;

        // ── Title ──────────────────────────────────────────────────
        {
            fort::utf8_table title;
            title.set_border_style(FT_DOUBLE2_STYLE);
            title << "MOE EXPERT REBALANCE PROFILING" << fort::endr;
            title[0][0].set_cell_text_align(fort::text_align::center);
            title.row(0).set_cell_row_type(fort::row_type::header);
            oss << "\n"
                << title.to_string();
        }

        // ── Config table ───────────────────────────────────────────
        {
            fort::utf8_table t;
            t.set_border_style(FT_DOUBLE2_STYLE);
            t << fort::header << "Parameter" << "Value" << fort::endr;
            t.column(0).set_cell_text_align(fort::text_align::left);
            t.column(1).set_cell_text_align(fort::text_align::right);

            t << "Mode" << (config_.mode == MoERebalanceMode::OBSERVE ? "OBSERVE" : config_.mode == MoERebalanceMode::DYNAMIC ? "DYNAMIC"
                                                                                                                              : "OFF")
              << fort::endr;
            if (requested_mode_ != config_.mode)
            {
                t << "Requested mode" << (requested_mode_ == MoERebalanceMode::OBSERVE ? "OBSERVE" : requested_mode_ == MoERebalanceMode::DYNAMIC ? "DYNAMIC"
                                                                                                                                                   : "OFF")
                  << fort::endr;
            }
            t << "Rebalance state" << toString(rebalanceDecision().reason) << fort::endr;
            t << "Experts" << config_.num_experts << fort::endr;
            t << "Participants" << config_.sockets.size() << fort::endr;
            t << "Top-K" << config_.top_k << fort::endr;
            t << "Window size" << config_.window_size << fort::endr;
            if (config_.max_window_size > 0 && config_.window_growth_factor > 1.0f)
            {
                std::ostringstream ws;
                ws << config_.window_size << " -> " << current_window_size_
                   << " (x" << std::fixed << std::setprecision(1) << config_.window_growth_factor
                   << ", max " << config_.max_window_size << ")";
                t << "Adaptive window" << ws.str() << fort::endr;
            }
            oss << t.to_string();
        }

        // ── Rebalance timing (if any rebalancing occurred) ─────────
        if (total_rebalances_ > 0)
        {
            fort::utf8_table t;
            t.set_border_style(FT_DOUBLE2_STYLE);
            t << fort::header << "Rebalance Metric" << "Value" << fort::endr;
            t.column(0).set_cell_text_align(fort::text_align::left);
            t.column(1).set_cell_text_align(fort::text_align::right);

            t << "Total rebalances" << total_rebalances_ << fort::endr;
            t << "Total swaps" << total_swaps_ << fort::endr;

            if (last_experts_moved_ > 0)
            {
                std::ostringstream val;
                val << last_experts_moved_ << " / " << config_.num_experts;
                t << "Experts moved (last)" << val.str() << fort::endr;
            }

            if (last_rebalance_duration_ms_ > 0)
            {
                std::ostringstream val;
                val << std::fixed << std::setprecision(2) << last_rebalance_duration_ms_ << " ms";
                t << "Solve time (last)" << val.str() << fort::endr;
            }

            if (last_prep_duration_ms_ > 0)
            {
                std::ostringstream val;
                val << std::fixed << std::setprecision(2) << last_prep_duration_ms_ << " ms";
                t << "VNNI prep time" << val.str() << fort::endr;
            }

            if (last_avg_imbalance_before_ > 0)
            {
                std::ostringstream val;
                val << std::fixed << std::setprecision(3)
                    << last_avg_imbalance_before_ << "x -> " << last_avg_imbalance_after_ << "x";
                t << "Avg imbalance" << val.str() << fort::endr;
            }

            if (last_worst_imbalance_before_ > 0)
            {
                std::ostringstream val;
                val << std::fixed << std::setprecision(3)
                    << last_worst_imbalance_before_ << "x (layer " << last_worst_layer_before_ << ")";
                t << "Worst (before)" << val.str() << fort::endr;
            }

            oss << t.to_string();
        }

        // ── Histogram summary ──────────────────────────────────────
        if (histogram_ && histogram_->windowTokenCount() > 0)
        {
            int num_layers = config_.num_layers;
            int num_sockets = static_cast<int>(config_.sockets.size());

            // Collect per-layer imbalance data
            struct LayerImbalance
            {
                int layer;
                float imbalance;
                std::vector<uint64_t> socket_loads;
            };
            std::vector<LayerImbalance> layer_data;
            layer_data.reserve(num_layers);
            float worst_imbalance = 0.0f;
            int worst_layer = 0;
            float avg_imbalance = 0.0f;

            for (int l = 0; l < num_layers; ++l)
            {
                float imb = histogram_->socketImbalanceRatio(l);
                auto loads = histogram_->socketLoads(l);
                avg_imbalance += imb;
                if (imb > worst_imbalance)
                {
                    worst_imbalance = imb;
                    worst_layer = l;
                }
                layer_data.push_back({l, imb, std::move(loads)});
            }
            avg_imbalance /= std::max(1, num_layers);

            // Top-5 hottest experts globally
            std::vector<uint64_t> global_counts(config_.num_experts, 0);
            for (int l = 0; l < num_layers; ++l)
            {
                auto lc = histogram_->layerHistogram(l);
                for (int e = 0; e < config_.num_experts; ++e)
                    global_counts[e] += lc[e];
            }
            std::vector<int> sorted(config_.num_experts);
            std::iota(sorted.begin(), sorted.end(), 0);
            std::sort(sorted.begin(), sorted.end(),
                      [&](int a, int b)
                      { return global_counts[a] > global_counts[b]; });

            // Per-socket stats
            std::vector<int> socket_expert_counts(num_sockets, 0);
            for (int e = 0; e < config_.num_experts; ++e)
            {
                if (e < static_cast<int>(current_placement_.size()))
                    socket_expert_counts[current_placement_[e]]++;
            }
            std::vector<uint64_t> socket_total_load(num_sockets, 0);
            for (int l = 0; l < num_layers; ++l)
            {
                for (int s = 0; s < num_sockets; ++s)
                {
                    if (s < static_cast<int>(layer_data[l].socket_loads.size()))
                        socket_total_load[s] += layer_data[l].socket_loads[s];
                }
            }
            uint64_t total_load = 0;
            for (int s = 0; s < num_sockets; ++s)
                total_load += socket_total_load[s];

            // Histogram overview table
            {
                fort::utf8_table t;
                t.set_border_style(FT_DOUBLE2_STYLE);
                t << fort::header << "Histogram Metric" << "Value" << fort::endr;
                t.column(0).set_cell_text_align(fort::text_align::left);
                t.column(1).set_cell_text_align(fort::text_align::right);

                t << "Window tokens" << histogram_->windowTokenCount() << fort::endr;

                {
                    std::ostringstream val;
                    val << std::fixed << std::setprecision(3) << avg_imbalance << "x";
                    t << "Avg imbalance" << val.str() << fort::endr;
                }
                {
                    std::ostringstream val;
                    val << std::fixed << std::setprecision(3) << worst_imbalance << "x (layer " << worst_layer << ")";
                    t << "Worst imbalance" << val.str() << fort::endr;
                }
                {
                    std::ostringstream val;
                    int show = std::min(5, config_.num_experts);
                    for (int i = 0; i < show; ++i)
                    {
                        if (i > 0)
                            val << ", ";
                        val << "e" << sorted[i] << "(" << global_counts[sorted[i]] << ")";
                    }
                    t << "Top-5 hottest" << val.str() << fort::endr;
                }
                {
                    std::ostringstream val;
                    for (int s = 0; s < num_sockets; ++s)
                    {
                        if (s > 0)
                            val << ", ";
                        val << "s" << s << "=" << socket_expert_counts[s];
                    }
                    t << "Experts/socket" << val.str() << fort::endr;
                }
                {
                    std::ostringstream val;
                    for (int s = 0; s < num_sockets; ++s)
                    {
                        if (s > 0)
                            val << ", ";
                        double pct = total_load > 0 ? 100.0 * socket_total_load[s] / total_load : 0;
                        val << "s" << s << "=" << std::fixed << std::setprecision(1) << pct << "%";
                    }
                    t << "Load/socket" << val.str() << fort::endr;
                }
                oss << t.to_string();
            }

            // Top-5 most imbalanced layers table
            if (num_layers > 1 && num_sockets >= 2)
            {
                auto sorted_layers = layer_data;
                std::sort(sorted_layers.begin(), sorted_layers.end(),
                          [](const auto &a, const auto &b)
                          { return a.imbalance > b.imbalance; });

                int show_layers = std::min(5, num_layers);

                fort::utf8_table t;
                t.set_border_style(FT_DOUBLE2_STYLE);
                t << fort::header << "Layer" << "Imbalance";
                for (int s = 0; s < num_sockets; ++s)
                {
                    std::ostringstream hdr;
                    hdr << "Socket " << s;
                    t << hdr.str();
                }
                t << fort::endr;

                t.column(0).set_cell_text_align(fort::text_align::right);
                t.column(1).set_cell_text_align(fort::text_align::right);
                for (int s = 0; s < num_sockets; ++s)
                    t.column(2 + s).set_cell_text_align(fort::text_align::right);

                for (int i = 0; i < show_layers; ++i)
                {
                    const auto &ld = sorted_layers[i];
                    std::ostringstream imb_ss;
                    imb_ss << std::fixed << std::setprecision(2) << ld.imbalance << "x";
                    t << ld.layer << imb_ss.str();
                    for (int s = 0; s < num_sockets; ++s)
                    {
                        uint64_t load = (s < static_cast<int>(ld.socket_loads.size()))
                                            ? ld.socket_loads[s]
                                            : 0;
                        t << load;
                    }
                    t << fort::endr;
                }

                oss << t.to_string();
            }
        }

        return oss.str();
    }

} // namespace llaminar2
