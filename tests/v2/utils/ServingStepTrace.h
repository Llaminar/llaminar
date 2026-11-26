/**
 * @file ServingStepTrace.h
 * @brief Recorded serving-session trace for recorded-input replay.
 *
 * A trace captures, from a real model run of the production ServingSession:
 * every client command (submit/grant/force/cancel) with the step it preceded,
 * every admission (slot, resolved seed, restored prefix), every shared forward
 * the session issued (slots, token rows) with explicitly selected logit coverage, and
 * every step's events. Replay feeds the recorded logits back through the same
 * production ServingSession and requires identical forwards and events.
 *
 * Exactness: logits are stored as hex floats (bit-exact). v2-full retains every
 * vocabulary entry and all penalty parameters. Legacy v1 retains only top-K
 * entries; a replayed row holds them and a floor strictly below all
 * of them elsewhere. Decisions are then identical for requests whose sampler
 * depends only on the top-k candidates: greedy, or top_k <= K without
 * repetition/presence/frequency penalties. That restriction applies only to
 * v1. Full-vocabulary recording retains the observations needed for penalties
 * and unrestricted nucleus sampling; native DRY execution needs separate proof.
 */
#pragma once

#include "execution/serving/ServingSession.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <set>
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
        /** @brief Full vocabulary preserves every candidate under history-dependent sampling. */
        enum class LogitCoverage { TopK, FullVocabulary };
        LogitCoverage coverage = LogitCoverage::TopK;

        struct Command
        {
            enum class Kind { Submit, Grant, Force, Cancel, Suspend, Restore } kind = Kind::Submit;
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
            std::vector<std::vector<std::pair<int, float>>> top; ///< Token/value pairs: v1 ranked, v2-full vocabulary order.
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

        /** @brief Reconstruct a validated row; full coverage never synthesizes missing observations.
         * @param top Recorded token/value pairs.
         * @param vocab Admitted vocabulary size.
         * @param out Destination row.
         * @param coverage Required recorded coverage.
         * @throws std::invalid_argument For missing, duplicate, nonfinite or out-of-range observations.
         */
        static void denseRow(const std::vector<std::pair<int, float>> &top, int vocab, std::vector<float> &out,
                             LogitCoverage coverage = LogitCoverage::TopK)
        {
            if (vocab <= 0 || top.empty() ||
                (coverage == LogitCoverage::FullVocabulary && top.size() != static_cast<std::size_t>(vocab)))
                throw std::invalid_argument("incomplete serving trace logits");
            std::vector<bool> seen(static_cast<std::size_t>(vocab), false);
            for (const auto &[index, value] : top)
            {
                if (index < 0 || index >= vocab || seen[static_cast<std::size_t>(index)] || !std::isfinite(value))
                    throw std::invalid_argument("invalid serving trace logit observation");
                seen[static_cast<std::size_t>(index)] = true;
            }
            float floor_value = std::numeric_limits<float>::max();
            for (const auto &[index, value] : top)
                floor_value = std::min(floor_value, value);
            out.assign(static_cast<std::size_t>(vocab), floor_value - 1.0e4f);
            for (const auto &[index, value] : top)
                out[static_cast<std::size_t>(index)] = value;
        }

        /** @throws std::invalid_argument When a law depends on more than the top-K logits. */
        static void requireReplayableLaw(const SamplingParams &params,
                                         LogitCoverage coverage = LogitCoverage::TopK)
        {
            if (coverage == LogitCoverage::FullVocabulary)
                return;
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
            out << (coverage == LogitCoverage::FullVocabulary ? "llaminar-serving-trace v2-full\n"
                                                             : "llaminar-serving-trace v1\n");
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
                    requireReplayableLaw(r.sampling, coverage);
                    if (coverage == LogitCoverage::FullVocabulary)
                    {
                        const auto &p = r.sampling;
                        // JSON escapes embedded newlines in DRY breakers; floats use the
                        // same exact hexadecimal spelling as recorded logits.
                        nlohmann::json law = {
                            {"presence", hexFloat(p.presence_penalty)}, {"frequency", hexFloat(p.frequency_penalty)},
                            {"repetition", hexFloat(p.repetition_penalty)}, {"dry_multiplier", hexFloat(p.dry_multiplier)},
                            {"dry_base", hexFloat(p.dry_base)}, {"dry_allowed_length", p.dry_allowed_length},
                            {"dry_penalty_last_n", p.dry_penalty_last_n}, {"dry_sequence_breakers", p.dry_sequence_breakers}};
                        out << "law " << law.dump() << '\n';
                    }
                    break;
                }
                case Command::Kind::Grant: out << "grant " << c.step << ' ' << c.id << ' ' << c.value << '\n'; break;
                case Command::Kind::Force: out << "force " << c.step << ' ' << c.id << ' ' << c.value << '\n'; break;
                case Command::Kind::Cancel: out << "cancel " << c.step << ' ' << c.id << '\n'; break;
                case Command::Kind::Suspend: out << "suspend " << c.step << ' ' << c.id << '\n'; break;
                case Command::Kind::Restore: out << "restore " << c.step << ' ' << c.id << '\n'; break;
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
            if (coverage == LogitCoverage::FullVocabulary)
                out << "end\n";
            if (!out)
                throw std::runtime_error("cannot write serving trace " + path);
        }

        /** @brief Reject commands that cannot be issued by this complete-session recording.
         * @throws std::runtime_error When a command is outside the recorded step domain.
         */
        void validateSchedule() const
        {
            for (const auto &command : commands)
            {
                if (command.step < 0 || static_cast<std::size_t>(command.step) >= steps.size())
                    throw std::runtime_error("recorded command lies outside recorded steps");
                switch (command.kind)
                {
                case Command::Kind::Submit:
                case Command::Kind::Grant:
                case Command::Kind::Force:
                case Command::Kind::Cancel:
                case Command::Kind::Suspend:
                case Command::Kind::Restore:
                    break;
                default: throw std::runtime_error("unknown recorded command kind");
                }
            }
            // Balanced suspension schedule: every suspend has a matching restore
            // (one manifest per request identity at a time).
            std::set<std::uint64_t> suspended;
            for (const auto &command : commands)
            {
                if (command.kind == Command::Kind::Suspend)
                {
                    if (!suspended.insert(command.id).second)
                        throw std::runtime_error("recorded double suspension of one request");
                }
                else if (command.kind == Command::Kind::Restore)
                {
                    if (suspended.erase(command.id) != 1)
                        throw std::runtime_error("recorded restore without a matching suspend");
                }
            }
            if (!suspended.empty())
                throw std::runtime_error("recorded suspension without a matching restore");
            std::set<std::uint64_t> admitted_ids;
            for (const auto &admission : admissions)
            {
                if (!admitted_ids.insert(admission.id).second || admission.seed == 0)
                    throw std::runtime_error("duplicate admission or unresolved seed");
                const auto submitted = std::find_if(commands.begin(), commands.end(), [&](const Command &c) {
                    return c.kind == Command::Kind::Submit && c.request.id == admission.id;
                });
                if (submitted == commands.end() ||
                    (submitted->request.sampling.seed != 0 && submitted->request.sampling.seed != admission.seed))
                    throw std::runtime_error("admission identity or authored seed disagrees with submit");
            }
        }

        static ServingStepTrace read(const std::string &path)
        {
            std::ifstream in(path);
            if (!in)
                throw std::runtime_error("cannot read serving trace " + path);
            ServingStepTrace trace;
            std::string line;
            std::getline(in, line);
            if (line == "llaminar-serving-trace v2-full")
                trace.coverage = LogitCoverage::FullVocabulary;
            else if (line != "llaminar-serving-trace v1")
                throw std::runtime_error("unsupported serving trace header: " + line);
            std::size_t laws = 0;
            bool needs_law = false;
            bool geometry_seen = false;
            bool ended = false;
            std::size_t pending_events = 0;
            auto hexf = [](std::istringstream &s) {
                std::string token;
                s >> token;
                char *end = nullptr;
                const float value = std::strtof(token.c_str(), &end);
                if (!s || end == token.c_str() || end != token.c_str() + token.size() || !std::isfinite(value))
                    throw std::runtime_error("invalid recorded floating-point value");
                return value;
            };
            while (std::getline(in, line))
            {
                std::istringstream s(line);
                std::string kind;
                s >> kind;
                if (ended)
                    throw std::runtime_error("record after serving trace end");
                if (!geometry_seen && kind != "geometry")
                    throw std::runtime_error("serving trace requires geometry first");
                if (pending_events != 0 && kind != "event")
                    throw std::runtime_error("missing recorded step events");
                if (needs_law && kind != "law")
                    throw std::runtime_error("missing recorded sampling law after submit");
                if (kind == "geometry")
                {
                    if (geometry_seen)
                        throw std::runtime_error("duplicate serving trace geometry");
                    s >> trace.context >> trace.vocab >> trace.capacity;
                    if (trace.context <= 0 || trace.vocab <= 0 || trace.capacity <= 0)
                        throw std::runtime_error("invalid serving trace geometry");
                    geometry_seen = true;
                }
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
                    needs_law = trace.coverage == LogitCoverage::FullVocabulary;
                }
                else if (kind == "law")
                {
                    if (!needs_law || trace.coverage != LogitCoverage::FullVocabulary || trace.commands.empty() ||
                        trace.commands.back().kind != Command::Kind::Submit)
                        throw std::runtime_error("sampling law without full-trace submit");
                    std::string encoded_law;
                    std::getline(s, encoded_law);
                    std::set<std::string> keys;
                    const auto law = nlohmann::json::parse(encoded_law,
                        [&](int, nlohmann::json::parse_event_t event, nlohmann::json &value) {
                            if (event == nlohmann::json::parse_event_t::key &&
                                !keys.insert(value.get<std::string>()).second)
                                throw std::runtime_error("duplicate recorded sampling law key");
                            return true;
                        });
                    const std::set<std::string> expected_keys = {"presence", "frequency", "repetition",
                        "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n", "dry_sequence_breakers"};
                    if (!law.is_object() || keys != expected_keys || law.size() != expected_keys.size())
                        throw std::runtime_error("recorded sampling law has missing or unknown fields");
                    auto &p = trace.commands.back().request.sampling;
                    auto number = [&](const char *key) {
                        const auto value = law.at(key).get<std::string>();
                        char *end = nullptr;
                        const float parsed = std::strtof(value.c_str(), &end);
                        if (end != value.c_str() + value.size() || end == value.c_str() || !std::isfinite(parsed))
                            throw std::runtime_error("invalid recorded sampling scalar");
                        return parsed;
                    };
                    p.presence_penalty = number("presence");
                    p.frequency_penalty = number("frequency");
                    p.repetition_penalty = number("repetition");
                    p.dry_multiplier = number("dry_multiplier");
                    p.dry_base = number("dry_base");
                    auto integer = [&](const char *key) {
                        const auto &value = law.at(key);
                        if (!value.is_number_integer())
                            throw std::runtime_error("recorded sampling integer has non-integral type");
                        if (value.is_number_unsigned())
                        {
                            const auto wide = value.get<std::uint64_t>();
                            if (wide > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
                                throw std::runtime_error("recorded sampling integer overflows int");
                            return static_cast<int>(wide);
                        }
                        const auto wide = value.get<std::int64_t>();
                        if (wide < std::numeric_limits<int>::min() || wide > std::numeric_limits<int>::max())
                            throw std::runtime_error("recorded sampling integer overflows int");
                        return static_cast<int>(wide);
                    };
                    p.dry_allowed_length = integer("dry_allowed_length");
                    p.dry_penalty_last_n = integer("dry_penalty_last_n");
                    p.dry_sequence_breakers = law.at("dry_sequence_breakers").get<std::vector<std::string>>();
                    ++laws;
                    needs_law = false;
                }
                else if (kind == "grant" || kind == "force" || kind == "cancel" ||
                         kind == "suspend" || kind == "restore")
                {
                    Command c;
                    c.kind = kind == "grant" ? Command::Kind::Grant
                             : kind == "force" ? Command::Kind::Force
                             : kind == "cancel" ? Command::Kind::Cancel
                             : kind == "suspend" ? Command::Kind::Suspend
                                                  : Command::Kind::Restore;
                    s >> c.step >> c.id;
                    if (c.kind == Command::Kind::Grant || c.kind == Command::Kind::Force)
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
                    if (rows == 0 || rows > static_cast<std::size_t>(trace.capacity))
                        throw std::runtime_error("invalid serving trace forward width");
                    f.slots.resize(rows);
                    f.rows.resize(rows);
                    for (std::size_t r = 0; r < rows; ++r)
                    {
                        std::size_t n = 0;
                        s >> f.slots[r] >> n;
                        if (f.slots[r] < 0 || f.slots[r] >= trace.capacity || n == 0 ||
                            n > static_cast<std::size_t>(trace.context))
                            throw std::runtime_error("invalid serving trace forward geometry");
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
                    if (n == 0 || n > static_cast<std::size_t>(trace.vocab))
                        throw std::runtime_error("invalid serving trace logit count");
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
                {
                    s >> pending_events;
                    trace.steps.emplace_back();
                }
                else if (kind == "end")
                {
                    if (trace.coverage != LogitCoverage::FullVocabulary)
                        throw std::runtime_error("end marker requires full trace version");
                    ended = true;
                }
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
                    if (trace.steps.empty() || pending_events == 0)
                        throw std::runtime_error("serving trace event without a declared step event");
                    --pending_events;
                    trace.steps.back().events.push_back(std::move(e));
                }
                else
                    throw std::runtime_error("unknown serving trace record: " + kind);
                if (s.fail())
                    throw std::runtime_error("incomplete serving trace record: " + kind);
                std::string extra;
                if (s >> extra)
                    throw std::runtime_error("trailing serving trace fields: " + kind);
            }
            if (!geometry_seen || pending_events != 0 ||
                (trace.coverage == LogitCoverage::FullVocabulary && !ended))
                throw std::runtime_error("incomplete serving trace");
            const auto submissions = std::count_if(trace.commands.begin(), trace.commands.end(),
                [](const Command &c) { return c.kind == Command::Kind::Submit; });
            if (needs_law || (trace.coverage == LogitCoverage::FullVocabulary && laws != static_cast<std::size_t>(submissions)))
                throw std::runtime_error("incomplete recorded sampling laws");
            for (const auto &f : trace.forwards)
            {
                if (f.slots.size() != f.top.size())
                    throw std::runtime_error("incomplete recorded forward logits");
                std::vector<float> checked;
                for (const auto &row : f.top)
                    denseRow(row, trace.vocab, checked, trace.coverage);
            }
            trace.validateSchedule();
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
            ServingStepTrace::requireReplayableLaw(request.sampling, trace_.coverage);
            trace_.admissions.push_back({request.id, slot, request.sampling.seed, restored});
        }

        void forwarded(const std::vector<int> &slots, const std::vector<std::vector<int>> &rows,
                       const std::vector<const float *> &row_logits, int vocab) override
        {
            ServingStepTrace::Forward forward;
            forward.slots = slots;
            forward.rows = rows;
            for (const float *logits : row_logits)
            {
                if (trace_.coverage == ServingStepTrace::LogitCoverage::FullVocabulary)
                {
                    auto &row = forward.top.emplace_back();
                    row.reserve(static_cast<std::size_t>(vocab));
                    for (int i = 0; i < vocab; ++i)
                        row.emplace_back(i, logits[i]);
                }
                else
                    forward.top.push_back(ServingStepTrace::topOf(logits, vocab));
            }
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
    class ServingStepTraceReplayer final : public IServingRowExecutor, public IServingTraceRecorder
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
                ServingStepTrace::denseRow(recorded.top[r], trace_.vocab, dense_[r], trace_.coverage);
            ++next_forward_;
            return true;
        }

        const float *servingRowLogits(int row) const override
        {
            return row >= 0 && static_cast<std::size_t>(row) < dense_.size()
                       ? dense_[static_cast<std::size_t>(row)].data()
                       : nullptr;
        }

        /** @brief Bind the actual production admission to its recorded identity and resolved policy. */
        void admitted(const ServingBatchRequest &request, int slot, int restored) override
        {
            if (observed_admissions_ >= trace_.admissions.size())
                throw std::runtime_error("unrecorded production admission");
            const auto &expected = trace_.admissions[observed_admissions_++];
            if (request.id != expected.id || slot != expected.slot ||
                request.sampling.seed != expected.seed || restored != expected.restored)
                throw std::runtime_error("production admission differs from recorded identity");
        }
        /** @brief Forwards are checked by the executor before publishing logits. */
        void forwarded(const std::vector<int> &, const std::vector<std::vector<int>> &,
                       const std::vector<const float *> &, int) override {}
        /** @brief Step outcomes are checked by replay after the production call returns. */
        void stepped(const std::vector<ServingBatchEvent> &) override {}

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
            try { trace.validateSchedule(); }
            catch (const std::exception &failure) { return failure.what(); }
            auto executor = std::make_shared<ServingStepTraceReplayer>(trace);
            ServingSession session(*executor, trace.context, trace.vocab);
            session.setTraceRecorder(executor);
            std::map<std::uint64_t, ServingContinuationManifest> suspended;
            std::string error;
            // The replay preserves the recorded admission order; the queue
            // bound is the recorded capacity, so any accepted request in the
            // recording remains acceptable here (order and outcomes typed).
            if (!session.begin(trace.capacity, trace.capacity, error, nullptr, trace.capacity))
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
                            ServingStepTrace::requireReplayableLaw(request.sampling, trace.coverage);
                            // The recorded resolved seed (entropy for seed 0).
                            if (auto it = seeds.find(request.id); it != seeds.end() && request.sampling.seed == 0)
                                request.sampling.seed = it->second;
                            if (session.submit(std::move(request), error) == ServingSubmissionOutcome::Refused)
                                return "submit failed: " + error;
                            break;
                        }
                        case ServingStepTrace::Command::Kind::Grant: session.grant(c.id, c.value); break;
                        case ServingStepTrace::Command::Kind::Force: session.force(c.id, c.value); break;
                        case ServingStepTrace::Command::Kind::Cancel: session.cancel(c.id); break;
                        case ServingStepTrace::Command::Kind::Suspend:
                        {
                            auto manifest = session.suspend(c.id, error);
                            if (!manifest)
                                return "suspend failed: " + error;
                            suspended.emplace(c.id, std::move(*manifest));
                            break;
                        }
                        case ServingStepTrace::Command::Kind::Restore:
                        {
                            auto it = suspended.find(c.id);
                            if (it == suspended.end())
                                return "restore failed: no suspended manifest for " +
                                    std::to_string(c.id);
                            auto manifest = std::move(it->second);
                            suspended.erase(it);
                            if (!session.restore(std::move(manifest), error))
                                return "restore failed: " + error;
                            break;
                        }
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
            if (!session.idle())
                return "complete-session trace ended with live requests";
            if (executor->forwardsConsumed() != trace.forwards.size())
                return "replay issued " + std::to_string(executor->forwardsConsumed()) + " of " +
                       std::to_string(trace.forwards.size()) + " recorded forwards";
            if (executor->observed_admissions_ != trace.admissions.size())
                return "replay did not observe every recorded admission";
            if (executor->admissionsConsumed() != trace.admissions.size())
                return "replay admitted " + std::to_string(executor->admissionsConsumed()) + " of " +
                       std::to_string(trace.admissions.size()) + " recorded requests";
            return {};
        }

    private:
        const ServingStepTrace &trace_;
        std::size_t next_forward_ = 0;
        std::size_t next_admission_ = 0;
        std::size_t observed_admissions_ = 0;
        std::vector<std::vector<float>> dense_;
    };
} // namespace llaminar2::test
