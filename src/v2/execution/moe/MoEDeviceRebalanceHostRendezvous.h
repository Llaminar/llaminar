/**
 * @file MoEDeviceRebalanceHostRendezvous.h
 * @brief Host-side rendezvous for graph-side MoE rebalance maintenance decisions.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace llaminar2
{
    class MoEDeviceRebalanceHostRendezvous
    {
    public:
        struct ProbeOutcome
        {
            bool valid = false;
            bool useful_work = false;
            uint32_t status_code = 0;
            uint32_t payload_bucket_slots = 0;
            uint64_t payload_edge_mask = 0;
        };

        struct PayloadDecision
        {
            bool valid = false;
            bool useful_work = false;
            uint32_t status_code = 0;
            uint32_t payload_bucket_slots = 0;
            uint64_t payload_edge_mask = 0;

            bool launchPayloadGraph() const
            {
                return valid &&
                       status_code == 0u &&
                       payload_bucket_slots != 0u &&
                       payload_edge_mask != 0ULL;
            }
        };

        struct ReadinessDecision
        {
            bool valid = false;
            bool all_ready = false;
            size_t ready_count = 0;
            size_t participant_count = 0;
        };

        static bool rendezvousCompletionReadiness(
            const void *domain_key,
            uint64_t generation,
            uint64_t operation_key,
            size_t expected_participants,
            const std::string &participant_name,
            bool local_ready,
            int timeout_ms,
            ReadinessDecision *decision,
            std::string *error)
        {
            if (decision)
                *decision = ReadinessDecision{};
            if (!domain_key)
            {
                if (error)
                    *error = "missing domain key";
                return false;
            }
            if (expected_participants == 0)
            {
                if (error)
                    *error = "expected participants must be non-zero";
                return false;
            }

            struct Key
            {
                const void *domain = nullptr;
                uint64_t generation = 0;
                uint64_t operation = 0;

                bool operator==(const Key &other) const
                {
                    return domain == other.domain &&
                           generation == other.generation &&
                           operation == other.operation;
                }
            };

            struct KeyHash
            {
                size_t operator()(const Key &key) const
                {
                    size_t hash = std::hash<const void *>{}(key.domain);
                    auto mix = [&](uint64_t value)
                    {
                        hash ^= std::hash<uint64_t>{}(value) +
                                0x9e3779b97f4a7c15ULL +
                                (hash << 6u) + (hash >> 2u);
                    };
                    mix(key.generation);
                    mix(key.operation);
                    return hash;
                }
            };

            struct State
            {
                size_t expected = 0;
                size_t arrivals = 0;
                size_t departures = 0;
                size_t release_count = 0;
                size_t ready_count = 0;
                bool completed = false;
                bool failed = false;
                std::string error;
            };

            static std::mutex mutex;
            static std::condition_variable cv;
            static std::unordered_map<Key, State, KeyHash> states;

            const Key key{domain_key, generation, operation_key};
            const auto timeout =
                std::chrono::milliseconds(std::max(1, timeout_ms));
            const auto deadline = std::chrono::steady_clock::now() + timeout;

            std::unique_lock<std::mutex> lock(mutex);
            auto it = states.end();
            bool inserted = false;
            while (true)
            {
                auto [candidate, was_inserted] = states.try_emplace(key);
                if (was_inserted || !candidate->second.completed)
                {
                    it = candidate;
                    inserted = was_inserted;
                    break;
                }
                const bool reusable = cv.wait_until(
                    lock,
                    deadline,
                    [&]()
                    {
                        const auto existing = states.find(key);
                        return existing == states.end() || !existing->second.completed;
                    });
                if (!reusable)
                {
                    if (error)
                    {
                        std::ostringstream message;
                        message << "timed out waiting for prior readiness rendezvous to release at generation "
                                << generation << " operation " << operation_key
                                << " participant=" << participant_name;
                        *error = message.str();
                    }
                    return false;
                }
            }
            State &state = it->second;
            if (inserted)
            {
                state.expected = expected_participants;
            }
            else if (state.expected != expected_participants)
            {
                state.failed = true;
                state.completed = true;
                state.release_count = std::max<size_t>(1, state.arrivals);
                state.error = "participant count mismatch";
                cv.notify_all();
            }

            if (!state.completed)
            {
                ++state.arrivals;
                if (local_ready)
                    ++state.ready_count;
                if (state.arrivals > state.expected)
                {
                    state.failed = true;
                    state.completed = true;
                    state.release_count = state.arrivals;
                    state.error = "too many participants arrived";
                    cv.notify_all();
                }
                else if (state.arrivals == state.expected)
                {
                    state.completed = true;
                    state.release_count = state.expected;
                    cv.notify_all();
                }
            }

            if (!state.completed)
            {
                const bool ready = cv.wait_for(lock, timeout, [&]()
                                               { return state.completed; });
                if (!ready)
                {
                    state.failed = true;
                    state.completed = true;
                    state.release_count = state.arrivals;
                    std::ostringstream message;
                    message << "timed out waiting for " << state.expected
                            << " readiness participants at generation " << generation
                            << " operation " << operation_key
                            << " after " << state.arrivals << " arrived";
                    state.error = message.str();
                    cv.notify_all();
                }
            }

            const bool ok = state.completed && !state.failed;
            if (decision)
            {
                decision->valid = ok;
                decision->ready_count = state.ready_count;
                decision->participant_count = state.expected;
                decision->all_ready = ok && state.ready_count == state.expected;
            }
            if (!ok && error)
            {
                std::ostringstream message;
                message << (state.error.empty() ? "completion readiness rendezvous failed" : state.error)
                        << " participant=" << participant_name;
                *error = message.str();
            }

            ++state.departures;
            const bool erase_state =
                state.release_count == 0 ||
                state.departures >= state.release_count;
            if (erase_state)
            {
                states.erase(it);
                cv.notify_all();
            }

            return ok;
        }

        static bool rendezvousProbeOutcome(
            const void *domain_key,
            uint64_t generation,
            size_t expected_participants,
            const std::string &participant_name,
            const ProbeOutcome &local,
            int timeout_ms,
            PayloadDecision *decision,
            std::string *error)
        {
            if (decision)
                *decision = PayloadDecision{};
            if (!domain_key)
            {
                if (error)
                    *error = "missing domain key";
                return false;
            }
            if (expected_participants == 0)
            {
                if (error)
                    *error = "expected participants must be non-zero";
                return false;
            }

            struct Key
            {
                const void *domain = nullptr;
                uint64_t generation = 0;

                bool operator==(const Key &other) const
                {
                    return domain == other.domain &&
                           generation == other.generation;
                }
            };

            struct KeyHash
            {
                size_t operator()(const Key &key) const
                {
                    const auto ptr_hash = std::hash<const void *>{}(key.domain);
                    const auto gen_hash = std::hash<uint64_t>{}(key.generation);
                    return ptr_hash ^ (gen_hash + 0x9e3779b97f4a7c15ULL +
                                       (ptr_hash << 6u) + (ptr_hash >> 2u));
                }
            };

            struct State
            {
                size_t expected = 0;
                size_t arrivals = 0;
                size_t departures = 0;
                size_t release_count = 0;
                bool completed = false;
                bool failed = false;
                std::string error;
                PayloadDecision aggregate;
            };

            static std::mutex mutex;
            static std::condition_variable cv;
            static std::unordered_map<Key, State, KeyHash> states;

            const Key key{domain_key, generation};
            const auto timeout =
                std::chrono::milliseconds(std::max(1, timeout_ms));
            const auto deadline = std::chrono::steady_clock::now() + timeout;

            std::unique_lock<std::mutex> lock(mutex);
            auto it = states.end();
            bool inserted = false;
            while (true)
            {
                auto [candidate, was_inserted] = states.try_emplace(key);
                if (was_inserted || !candidate->second.completed)
                {
                    it = candidate;
                    inserted = was_inserted;
                    break;
                }
                const bool reusable = cv.wait_until(
                    lock,
                    deadline,
                    [&]()
                    {
                        const auto existing = states.find(key);
                        return existing == states.end() || !existing->second.completed;
                    });
                if (!reusable)
                {
                    if (error)
                    {
                        std::ostringstream message;
                        message << "timed out waiting for prior probe outcome rendezvous to release at generation "
                                << generation << " participant=" << participant_name;
                        *error = message.str();
                    }
                    return false;
                }
            }
            State &state = it->second;
            if (inserted)
            {
                state.expected = expected_participants;
                state.aggregate.valid = true;
            }
            else if (state.expected != expected_participants)
            {
                state.failed = true;
                state.completed = true;
                state.release_count = std::max<size_t>(1, state.arrivals);
                state.error = "participant count mismatch";
                cv.notify_all();
            }

            if (!state.completed)
            {
                ++state.arrivals;
                if (state.arrivals > state.expected)
                {
                    state.failed = true;
                    state.completed = true;
                    state.release_count = state.arrivals;
                    state.error = "too many participants arrived";
                    cv.notify_all();
                }
                else
                {
                    if (!local.valid)
                    {
                        state.failed = true;
                        state.aggregate.valid = false;
                        state.error = "invalid local probe outcome from " + participant_name;
                    }
                    if (local.status_code != 0u)
                    {
                        state.failed = true;
                        state.aggregate.status_code = local.status_code;
                        state.error = "non-ok local probe outcome from " + participant_name;
                    }

                    state.aggregate.useful_work =
                        state.aggregate.useful_work || local.useful_work;
                    state.aggregate.payload_bucket_slots =
                        std::max(state.aggregate.payload_bucket_slots,
                                 local.payload_bucket_slots);
                    state.aggregate.payload_edge_mask |= local.payload_edge_mask;

                    if (state.arrivals == state.expected)
                    {
                        state.completed = true;
                        state.release_count = state.expected;
                        if (state.failed)
                            state.aggregate.valid = false;
                        cv.notify_all();
                    }
                }
            }

            if (!state.completed)
            {
                const bool ready = cv.wait_for(lock, timeout, [&]()
                                               { return state.completed; });
                if (!ready)
                {
                    state.failed = true;
                    state.completed = true;
                    state.release_count = state.arrivals;
                    std::ostringstream message;
                    message << "timed out waiting for " << state.expected
                            << " participants at generation " << generation
                            << " after " << state.arrivals << " arrived";
                    state.error = message.str();
                    cv.notify_all();
                }
            }

            const bool ok = state.completed && !state.failed;
            if (decision)
                *decision = state.aggregate;
            if (!ok && error)
                *error = state.error.empty() ? "probe outcome rendezvous failed" : state.error;

            ++state.departures;
            const bool erase_state =
                state.release_count == 0 ||
                state.departures >= state.release_count;
            if (erase_state)
            {
                states.erase(it);
                cv.notify_all();
            }

            return ok;
        }
    };
} // namespace llaminar2
