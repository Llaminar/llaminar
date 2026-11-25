/**
 * @file StreamingDeltaFragments.h
 * @brief Synchronous borrowed-input fragmentation for HTTP streaming deltas.
 *
 * The caller owns the source JSON throughout publication. Only the current
 * selected-field fragment is materialized; cancellation stops before building
 * another. Metadata and upstream parser storage are outside this local bound.
 */
#pragma once
#include "app/modes/SSEPublishChannel.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <string_view>
namespace llaminar2::streaming_detail
{
using json = nlohmann::json;
/**
 * @brief Visit borrowed UTF-8 pieces without materializing the entire split.
 * @param text Source retained by the caller throughout publication.
 * @param max_bytes Maximum piece size; must fit one UTF-8 code point.
 * @param publish Receives the piece and whether it is last; false stops immediately.
 * @return Whether all pieces were delivered.
 */
template <typename Publish>
bool publishUtf8Pieces(const std::string &text, std::size_t max_bytes, Publish &&publish)
{
    std::size_t start = 0;
    while (start < text.size())
    {
        std::size_t end = std::min(text.size(), start + max_bytes);
        while (end < text.size() && end > start &&
               (static_cast<unsigned char>(text[end]) & 0xC0u) == 0x80u)
            --end; // never split inside a multi-byte sequence
        if (end == start)
            end = std::min(text.size(), start + max_bytes);
        if (!publish(std::string_view(text).substr(start, end - start), end == text.size()))
            return false;
        start = end;
    }
    return true;
}


/**
 * @brief Publish protocol deltas without eagerly copying the selected large field.
 * @param delta Caller-owned immutable source.
 * @param finish_reason Optional terminal reason, emitted only with the last piece.
 * @param emit_one_chunk Synchronous callback; false cancels further construction.
 * @return Whether every piece was accepted.
 */
template <typename Emit>
bool publishDeltaFragments(const json &delta, const char *finish_reason, Emit &&emit_one_chunk)
{
    constexpr std::size_t kMax = SSEPublishChannelBudget::kMaxDeltaTextBytes;
    for (const char *field : {"content", "reasoning_content"})
        if (delta.contains(field) && delta[field].is_string() &&
            delta[field].get_ref<const std::string &>().size() > kMax)
        {
            json part = json::object();
            for (auto it = delta.begin(); it != delta.end(); ++it)
                if (it.key() != field)
                    part[it.key()] = it.value();
            return publishUtf8Pieces(delta[field].get_ref<const std::string &>(), kMax,
                                     [&](std::string_view piece, bool last) {
                part[field] = std::string(piece);
                return emit_one_chunk(part, last ? finish_reason : nullptr);
            });
        }
    if (delta.contains("tool_calls") && delta["tool_calls"].is_array() && delta["tool_calls"].size() == 1)
    {
        const auto &call = delta["tool_calls"][0];
        const auto arguments_path = json::json_pointer("/function/arguments");
        if (call.contains(arguments_path) && call[arguments_path].is_string() &&
            call[arguments_path].get_ref<const std::string &>().size() > kMax)
        {
            json part_call = json::object();
            for (auto it = call.begin(); it != call.end(); ++it)
                if (it.key() != "function")
                    part_call[it.key()] = it.value();
            part_call["function"] = json::object();
            for (auto it = call["function"].begin(); it != call["function"].end(); ++it)
                if (it.key() != "arguments")
                    part_call["function"][it.key()] = it.value();
            return publishUtf8Pieces(call[arguments_path].get_ref<const std::string &>(), kMax,
                                     [&](std::string_view piece, bool last) {
                part_call["function"]["arguments"] = std::string(piece);
                const bool delivered = emit_one_chunk({{"tool_calls", json::array({part_call})}},
                                                      last ? finish_reason : nullptr);
                if (!delivered || last)
                    return delivered;
                part_call = json{{"index", call["index"]}, {"function", json::object()}};
                return true;
            });
        }
    }
    return emit_one_chunk(delta, finish_reason);
}
} // namespace llaminar2::streaming_detail
