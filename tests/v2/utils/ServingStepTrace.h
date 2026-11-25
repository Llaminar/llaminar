/**
 * @file ServingStepTrace.h
 * @brief Recorded serving-session trace for recorded-input replay.
 *
 * A trace captures, from a real model run of the production ServingSession:
 * every client command (submit/grant/force/cancel) with the step it preceded,
 * every admission (slot, resolved seed, restored prefix), every shared forward
 * the session issued (slots, token rows) with each row's TOP-K logits, and
 * every step's events. Replay feeds the recorded logits back through the same
 * production ServingSession and requires identical forwards and events.
 *
 * Exactness: logits are stored as hex floats (bit-exact). Only the top-K
 * entries are kept; a replayed row holds them and a floor strictly below all
 * of them elsewhere. Decisions are then identical for requests whose sampler
 * depends only on the top-k candidates: greedy, or top_k <= K without
 * repetition/presence/frequency penalties. The recorder rejects other laws.
 */
#pragma once

#include "execution/serving/ServingSession.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    struct ServingStepTrace
    {
        static constexpr int kTopK = 64;

        struct Command
        {
            enum class Kind { Submit, Grant, Force, Cancel } kind = Kind::Submit;
            int step = 0; ///< Issued before this step.
            ServingBatchRequest request; ///< Submit only.
            std::uint64_t id = 0;
            int value = 0; ///< Grant tokens or forced token.
        };
        struct Admission
        {
            std::uint64_t id = 0;
            int slot = -1;
            unsigned int seed = 0;
            int restored = 0;
        };
        struct Forward
        {
            std::vector<int> slots;
            std::vector<std::vector<int>> rows;
            std::vector<std::vector<std::pair<int, float>>> top; ///< Per row, descending.
        };
        struct Step
        {
            std::vector<ServingBatchEvent> events;
        };

        int context = 0;
        int vocab = 0;
        int capacity = 0;
        std::vector<Command> commands;
        std::vector<Admission> admissions;
        std::vector<Forward> forwards;
        std::vector<Step> steps;

        /** @brief Top-K (index, value) of a vocab row, descending, ties by index. */
        static std::vector<std::pair<int, float>> topOf(const float *logits, int vocab)
        {
            std::vector<int> order(static_cast<std::size_t>(vocab));
            for (int i = 0; i < vocab; ++i)
                order[static_cast<std::size_t>(i)] = i;
            const int k = std::min(kTopK, vocab);
            std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int a, int b) {
                return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
            });
            std::vector<std::pair<int, float>> top;
            for (int i = 0; i < k; ++i)
                top.emplace_back(order[static_cast<std::size_t>(i)], logits[order[static_cast<std::size_t>(i)]]);
            return top;
        }

        /** @brief Dense replay row: recorded entries, a floor below them elsewhere. */
        static void denseRow(const std::vector<std::pair<int, float>> &top, int vocab, std::vector<float> &out)
        {
            float floor_value = std::numeric_limits<float>::max();
            for (const auto &[index, value] : top)
                floor_value = std::min(floor_value, value);
            out.assign(static_cast<std::size_t>(vocab), floor_value - 1.0e4f);
            for (const auto &[index, value] : top)
                out[static_cast<std::size_t>(index)] = value;
        }

        /** @throws std::invalid_argument When a law depends on more than the top-K logits. */
        static void requireReplayableLaw(const SamplingParams &params)
        {
            if (params.has_penalties())
                throw std::invalid_argument("trace replay requires sampling without penalties");
            if (!params.is_greedy() && (params.top_k <= 0 || params.top_k > kTopK))
                throw std::invalid_argument("trace replay requires greedy or top_k <= 64");
        }

        static std::string hexFloat(float value)
        {
            char buffer[64];
            std::snprintf(buffer, sizeof(buffer), "%a", static_cast<double>(value));
            return buffer;
        }

        void write(const std::string &path) const
        {
            std::ofstream out(path);
            out << "llaminar-serving-trace v1\n";
            out << "geometry " << context << ' ' << vocab << ' ' << capacity << '\n';
            for (const auto &c : commands)
            {
                switch (c.kind)
                {
                case Command::Kind::Submit:
                {
                    const auto &r = c.request;
                    out << "submit " << c.step << ' ' << r.id << ' ' << r.max_new_tokens << ' ' << (r.paced ? 1 : 0)
                        << ' ' << hexFloat(r.sampling.temperature) << ' ' << r.sampling.top_k << ' '
                        << hexFloat(r.sampling.top_p) << ' ' << r.sampling.seed << ' ' << r.stop_tokens.size();
                    for (const auto t : r.stop_tokens)
                        out << ' ' << t;
                    out << ' ' << r.prompt.size();
                    for (const auto t : r.prompt)
                        out << ' ' << t;
                    out << '\n';
                    break;
                }
                case Command::Kind::Grant: out << "grant " << c.step << ' ' << c.id << ' ' << c.value << '\n'; break;
                case Command::Kind::Force: out << "force " << c.step << ' ' << c.id << ' ' << c.value << '\n'; break;
                case Command::Kind::Cancel: out << "cancel " << c.step << ' ' << c.id << '\n'; break;
                }
            }
            for (const auto &a : admissions)
                out << "admit " << a.id << ' ' << a.slot << ' ' << a.seed << ' ' << a.restored << '\n';
            for (const auto &f : forwards)
            {
                out << "forward " << f.slots.size();
                for (std::size_t r = 0; r < f.slots.size(); ++r)
                {
                    out << ' ' << f.slots[r] << ' ' << f.rows[r].size();
                    for (const auto t : f.rows[r])
                        out << ' ' << t;
                }
                out << '\n';
                for (const auto &row : f.top)
                {
                    out << "top " << row.size();
                    for (const auto &[index, value] : row)
                        out << ' ' << index << ' ' << hexFloat(value);
                    out << '\n';
                }
            }
            for (const auto &s : steps)
            {
                out << "step " << s.events.size() << '\n';
                for (const auto &e : s.events)
                {
                    out << "event " << e.id << ' ' << static_cast<int>(e.finish) << ' ' << e.completion_tokens << ' '
                        << e.tokens.size();
                    for (const auto t : e.tokens)
                        out << ' ' << t;
                    out << '\n';
                }
            }
            if (!out)
                throw std::runtime_error("cannot write serving trace " + path);
        }

        static ServingStepTrace read(const std::string &path)
        {
            std::ifstream in(path);
            if (!in)
                throw std::runtime_error("cannot read serving trace " + path);
            ServingStepTrace trace;
            std::string line;
            std::getline(in, line);
            if (line != "llaminar-serving-trace v1")
                throw std::runtime_error("unsupported serving trace header: " + line);
            auto hexf = [](std::istringstream &s) {
                std::string token;
                s >> token;
                return std::strtof(token.c_str(), nullptr);
            };
            while (std::getline(in, line))
            {
                std::istringstream s(line);
                std::string kind;
                s >> kind;
                if (kind == "geometry")
                    s >> trace.context >> trace.vocab >> trace.capacity;
                else if (kind == "submit")
                {
                    Command c;
                    c.kind = Command::Kind::Submit;
                    auto &r = c.request;
                    int paced = 0;
                    std::size_t n = 0;
                    s >> c.step >> r.id >> r.max_new_tokens >> paced;
                    r.paced = paced != 0;
                    r.sampling.temperature = hexf(s);
                    s >> r.sampling.top_k;
                    r.sampling.top_p = hexf(s);
                    s >> r.sampling.seed >> n;
                    r.stop_tokens.resize(n);
                    for (auto &t : r.stop_tokens)
                        s >> t;
                    s >> n;
                    r.prompt.resize(n);
                    for (auto &t : r.prompt)
                        s >> t;
                    c.id = r.id;
                    trace.commands.push_back(std::move(c));
                }
                else if (kind == "grant" || kind == "force" || kind == "cancel")
                {
                    Command c;
                    c.kind = kind == "grant" ? Command::Kind::Grant
                             : kind == "force" ? Command::Kind::Force
                                               : Command::Kind::Cancel;
                    s >> c.step >> c.id;
                    if (c.kind != Command::Kind::Cancel)
                        s >> c.value;
                    trace.commands.push_back(c);
                }
                else if (kind == "admit")
                {
                    Admission a;
                    s >> a.id >> a.slot >> a.seed >> a.restored;
                    trace.admissions.push_back(a);
                }
                else if (kind == "forward")
                {
                    Forward f;
                    std::size_t rows = 0;
                    s >> rows;
                    f.slots.resize(rows);
                    f.rows.resize(rows);
                    for (std::size_t r = 0; r < rows; ++r)
                    {
                        std::size_t n = 0;
                        s >> f.slots[r] >> n;
                        f.rows[r].resize(n);
                        for (auto &t : f.rows[r])
                            s >> t;
                    }
                    trace.forwards.push_back(std::move(f));
                }
                else if (kind == "top")
                {
                    std::size_t n = 0;
                    s >> n;
                    std::vector<std::pair<int, float>> row(n);
                    for (auto &[index, value] : row)
                    {
                        s >> index;
                        value = hexf(s);
                    }
                    if (trace.forwards.empty())
                        throw std::runtime_error("serving trace top row without a forward");
                    trace.forwards.back().top.push_back(std::move(row));
                }
                else if (kind == "step")
                    trace.steps.emplace_back();
                else if (kind == "event")
                {
                    ServingBatchEvent e;
                    int finish = 0;
                    std::size_t n = 0;
                    s >> e.id >> finish >> e.completion_tokens >> n;
                    e.finish = static_cast<ServingFinishReason>(finish);
                    e.tokens.resize(n);
                    for (auto &t : e.tokens)
                        s >> t;
                    if (trace.steps.empty())
                        throw std::runtime_error("serving trace event without a step");
                    trace.steps.back().events.push_back(std::move(e));
                }
                else if (!kind.empty())
                    throw std::runtime_error("unknown serving trace record: " + kind);
            }
            return trace;
        }
    };

    /** @brief Records a live session into a ServingStepTrace. */
    class ServingStepTraceRecorder final : public IServingTraceRecorder
    {
    public:
        explicit ServingStepTraceRecorder(ServingStepTrace &trace) : trace_(trace) {}

        void admitted(const ServingBatchRequest &request, int slot, int restored) override
        {
            ServingStepTrace::requireReplayableLaw(request.sampling);
            trace_.admissions.push_back({request.id, slot, request.sampling.seed, restored});
        }

        void forwarded(const std::vector<int> &slots, const std::vector<std::vector<int>> &rows,
                       const std::vector<const float *> &row_logits, int vocab) override
        {
            ServingStepTrace::Forward forward;
            forward.slots = slots;
            forward.rows = rows;
            for (const float *logits : row_logits)
                forward.top.push_back(ServingStepTrace::topOf(logits, vocab));
            trace_.forwards.push_back(std::move(forward));
        }

        void stepped(const std::vector<ServingBatchEvent> &events) override
        {
            trace_.steps.push_back({events});
        }

    private:
        ServingStepTrace &trace_;
    };

    /**
     * @brief Replays a trace through the production ServingSession.
     *
     * The executor requires the session to issue exactly the recorded
     * forwards (same slots and token rows, in order) and returns the recorded
     * logits. replay() re-issues the recorded commands at their steps and
     * compares every step's events with the recording.
     */
    class ServingStepTraceReplayer final : public IServingRowExecutor
    {
    public:
        explicit ServingStepTraceReplayer(const ServingStepTrace &trace) : trace_(trace) {}

        bool forwardServingRows(const std::vector<int> &slots, const std::vector<std::vector<int>> &rows) override
        {
            if (next_forward_ >= trace_.forwards.size())
                throw std::runtime_error("replay issued an unrecorded forward #" + std::to_string(next_forward_));
            const auto &recorded = trace_.forwards[next_forward_];
            if (recorded.slots != slots || recorded.rows != rows)
                throw std::runtime_error("replay forward #" + std::to_string(next_forward_) +
                                         " differs from the recording (slots/rows)");
            dense_.resize(recorded.top.size());
            for (std::size_t r = 0; r < recorded.top.size(); ++r)
                ServingStepTrace::denseRow(recorded.top[r], trace_.vocab, dense_[r]);
            ++next_forward_;
            return true;
        }

        const float *servingRowLogits(int row) const override
        {
            return row >= 0 && static_cast<std::size_t>(row) < dense_.size()
                       ? dense_[static_cast<std::size_t>(row)].data()
                       : nullptr;
        }

        bool resetServingRow(int) override { return true; }

        int restoreServingRowPrefix(int slot, const std::vector<int32_t> &, PrefixLookupResult &admission) override
        {
            admission = {};
            if (next_admission_ >= trace_.admissions.size())
                throw std::runtime_error("replay admitted an unrecorded request");
            const auto &recorded = trace_.admissions[next_admission_++];
            if (recorded.slot != slot)
                throw std::runtime_error("replay admitted request " + std::to_string(recorded.id) + " into slot " +
                                         std::to_string(slot) + ", recorded " + std::to_string(recorded.slot));
            return recorded.restored;
        }

        bool harvestServingRowPrefix(int, const PrefixLookupResult &, const std::vector<int32_t> &) override
        {
            return true;
        }

        std::size_t forwardsConsumed() const { return next_forward_; }
        std::size_t admissionsConsumed() const { return next_admission_; }

        /**
         * @brief Run the recorded commands through a fresh production session.
         * @return Empty on an identical replay, else the first difference.
         */
        static std::string replay(const ServingStepTrace &trace)
        {
            ServingStepTraceReplayer executor(trace);
            ServingSession session(executor, trace.context, trace.vocab);
            std::string error;
            if (!session.begin(trace.capacity, trace.capacity, error))
                return "begin failed: " + error;
            std::map<std::uint64_t, unsigned int> seeds;
            for (const auto &a : trace.admissions)
                seeds[a.id] = a.seed;
            try
            {
                for (std::size_t step = 0; step < trace.steps.size(); ++step)
                {
                    for (const auto &c : trace.commands)
                    {
                        if (c.step != static_cast<int>(step))
                            continue;
                        switch (c.kind)
                        {
                        case ServingStepTrace::Command::Kind::Submit:
                        {
                            auto request = c.request;
                            // The recorded resolved seed (entropy for seed 0).
                            if (auto it = seeds.find(request.id); it != seeds.end())
                                request.sampling.seed = it->second;
                            if (!session.submit(std::move(request), error))
                                return "submit failed: " + error;
                            break;
                        }
                        case ServingStepTrace::Command::Kind::Grant: session.grant(c.id, c.value); break;
                        case ServingStepTrace::Command::Kind::Force: session.force(c.id, c.value); break;
                        case ServingStepTrace::Command::Kind::Cancel: session.cancel(c.id); break;
                        }
                    }
                    const auto events = session.step();
                    const auto &expected = trace.steps[step].events;
                    if (events.size() != expected.size())
                        return "step " + std::to_string(step) + ": " + std::to_string(events.size()) +
                               " events, recorded " + std::to_string(expected.size());
                    for (std::size_t e = 0; e < events.size(); ++e)
                    {
                        const auto &got = events[e];
                        const auto &want = expected[e];
                        if (got.id != want.id || got.finish != want.finish || got.tokens != want.tokens ||
                            got.completion_tokens != want.completion_tokens)
                            return "step " + std::to_string(step) + ": event for request " + std::to_string(got.id) +
                                   " differs from the recording";
                    }
                }
            }
            catch (const std::exception &failure)
            {
                return failure.what();
            }
            if (executor.forwardsConsumed() != trace.forwards.size())
                return "replay issued " + std::to_string(executor.forwardsConsumed()) + " of " +
                       std::to_string(trace.forwards.size()) + " recorded forwards";
            if (executor.admissionsConsumed() != trace.admissions.size())
                return "replay admitted " + std::to_string(executor.admissionsConsumed()) + " of " +
                       std::to_string(trace.admissions.size()) + " recorded requests";
            return {};
        }

    private:
        const ServingStepTrace &trace_;
        std::size_t next_forward_ = 0;
        std::size_t next_admission_ = 0;
        std::vector<std::vector<float>> dense_;
    };
} // namespace llaminar2::test
