/**
 * @file Tokenizer.cpp
 * @brief Native BPE tokenizer implementation
 * @author David Sanftenberg
 * @date 2025
 *
 * Implements Byte Pair Encoding (BPE) tokenization by reading vocabulary
 * directly from GGUF metadata.
 */

#include "Tokenizer.h"
#include "Logger.h"
#include "../loaders/ModelContext.h"
#include "../loaders/ModelLoader.h"
#include <regex>
#include <algorithm>
#include <sstream>
#include <queue>

namespace llaminar2
{

    // ============================================================================
    // BPETokenizer Implementation
    // ============================================================================

    std::shared_ptr<BPETokenizer> BPETokenizer::create(std::shared_ptr<ModelContext> model_ctx)
    {
        if (!model_ctx)
        {
            LOG_ERROR("[BPETokenizer] ModelContext is null");
            return nullptr;
        }

        auto tokenizer = std::shared_ptr<BPETokenizer>(new BPETokenizer());

        if (!tokenizer->initializeFromMetadata(model_ctx))
        {
            LOG_ERROR("[BPETokenizer] Failed to initialize from metadata");
            return nullptr;
        }

        LOG_DEBUG("[BPETokenizer] Initialized (vocab_size=" << tokenizer->vocab_size() << ")");
        return tokenizer;
    }

    bool BPETokenizer::initializeFromMetadata(std::shared_ptr<ModelContext> model_ctx)
    {
        const auto &metadata = model_ctx->model().metadata;

        // Extract vocabulary tokens
        auto tokens_it = metadata.find("tokenizer.ggml.tokens");
        if (tokens_it == metadata.end() || tokens_it->second.type != GGUFValueType::ARRAY)
        {
            LOG_ERROR("[BPETokenizer] Missing tokenizer.ggml.tokens in metadata");
            return false;
        }

        // Load vocabulary from stored string array
        vocab_ = tokens_it->second.asStringArray();
        if (vocab_.empty())
        {
            LOG_ERROR("[BPETokenizer] Empty vocabulary");
            return false;
        }

        // Build reverse mapping
        for (size_t i = 0; i < vocab_.size(); ++i)
        {
            vocab_map_[vocab_[i]] = i;
        }

        // Extract BPE merges
        auto merges_it = metadata.find("tokenizer.ggml.merges");
        if (merges_it != metadata.end() && merges_it->second.type == GGUFValueType::ARRAY)
        {
            const auto &merge_strings = merges_it->second.asStringArray();
            merges_.reserve(merge_strings.size());
            merge_result_ids_.resize(merge_strings.size(), -1);

            // Parse merge strings "token1 token2" into pairs
            for (size_t i = 0; i < merge_strings.size(); ++i)
            {
                const auto &merge_str = merge_strings[i];
                size_t space_pos = merge_str.find(' ');
                if (space_pos != std::string::npos)
                {
                    std::string first = merge_str.substr(0, space_pos);
                    std::string second = merge_str.substr(space_pos + 1);
                    merges_.emplace_back(first, second);
                    merge_ranks_[{first, second}] = static_cast<int>(i);

                    // Integer optimization
                    auto it1 = vocab_map_.find(first);
                    auto it2 = vocab_map_.find(second);
                    auto it3 = vocab_map_.find(first + second);

                    if (it1 != vocab_map_.end() && it2 != vocab_map_.end())
                    {
                        merge_ranks_int_[{it1->second, it2->second}] = static_cast<int>(i);
                    }
                    if (it3 != vocab_map_.end())
                    {
                        merge_result_ids_[i] = it3->second;
                    }
                }
            }
        }

        // Extract special token IDs
        auto get_uint_metadata = [&metadata](const std::string &key, int default_val) -> int
        {
            auto it = metadata.find(key);
            if (it != metadata.end())
            {
                return it->second.asUInt32();
            }
            return default_val;
        };

        bos_token_ = get_uint_metadata("tokenizer.ggml.bos_token_id", 0);
        eos_token_ = get_uint_metadata("tokenizer.ggml.eos_token_id", 0);
        pad_token_ = get_uint_metadata("tokenizer.ggml.padding_token_id", 0);

        // Initialize byte-level encoder (GPT-2 style)
        initializeByteEncoder();

        // Initialize byte to token ID map
        byte_to_token_id_.resize(256, -1);
        for (int i = 0; i < 256; ++i)
        {
            std::string s = byte_encoder_[i];
            auto it = vocab_map_.find(s);
            if (it != vocab_map_.end())
            {
                byte_to_token_id_[i] = it->second;
            }
        }

        LOG_DEBUG("[BPETokenizer] Initialized with " << vocab_.size() << " tokens, "
                                                     << merges_.size() << " merges");
        LOG_DEBUG("[BPETokenizer] Special tokens: BOS=" << bos_token_
                                                        << ", EOS=" << eos_token_
                                                        << ", PAD=" << pad_token_);

        // Load chat template from metadata if available
        auto chat_template_it = metadata.find("tokenizer.chat_template");
        if (chat_template_it != metadata.end() && chat_template_it->second.type == GGUFValueType::STRING)
        {
            chat_template_string_ = chat_template_it->second.asString();
            if (!chat_template_string_.empty())
            {
                chat_template_ = ChatTemplate::create(chat_template_string_);
                if (chat_template_)
                {
                    LOG_INFO("[BPETokenizer] Loaded chat template: "
                             << chatTemplateTypeName(chat_template_->type()));
                }
                else
                {
                    LOG_WARN("[BPETokenizer] Failed to parse chat template");
                }
            }
        }
        else
        {
            LOG_DEBUG("[BPETokenizer] No chat template in model metadata");
        }

        return true;
    }

    void BPETokenizer::initializeByteEncoder()
    {
        byte_encoder_.resize(256);
        // GPT-2 byte-level encoding
        // Maps bytes to printable Unicode characters to avoid control chars
        //
        // The GPT-2 scheme treats bytes as Unicode codepoints:
        // - Printable ASCII (33-126): identity mapping (single byte UTF-8)
        // - Latin-1 Supplement (161-172, 174-255): identity mapping BUT stored as UTF-8
        // - Other bytes (0-32, 127-160, 173): map to U+0100 + n
        //
        // Important: For bytes >= 128, even "identity" mappings need UTF-8 encoding
        // because the vocab stores them as UTF-8 strings.
        int n = 0;
        for (int b = 0; b < 256; ++b)
        {
            if (b >= 33 && b <= 126)
            {
                // Printable ASCII: single-byte identity mapping
                byte_encoder_[b] = std::string(1, static_cast<char>(b));
                byte_decoder_[std::string(1, static_cast<char>(b))] = b;
            }
            else if ((b >= 161 && b <= 172) || (b >= 174 && b <= 255))
            {
                // Latin-1 Supplement: identity mapping (codepoint == byte value)
                // but needs UTF-8 encoding for bytes >= 128
                int codepoint = b; // Identity: codepoint equals byte value
                // UTF-8 encoding for codepoints 0x80-0xFF: 110000xx 10xxxxxx
                char utf8[3];
                utf8[0] = static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F));
                utf8[1] = static_cast<char>(0x80 | (codepoint & 0x3F));
                utf8[2] = '\0';
                std::string encoded(utf8, 2);
                byte_encoder_[b] = encoded;
                byte_decoder_[encoded] = b;
            }
            else
            {
                // Non-printable: map to Unicode U+0100 + n
                // Must encode as UTF-8 (2 bytes for U+0100 to U+07FF)
                int codepoint = 256 + n;
                // UTF-8 encoding for codepoints 0x0100-0x07FF: 110xxxxx 10xxxxxx
                char utf8[3];
                utf8[0] = static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F));
                utf8[1] = static_cast<char>(0x80 | (codepoint & 0x3F));
                utf8[2] = '\0';
                std::string encoded(utf8, 2);
                byte_encoder_[b] = encoded;
                byte_decoder_[encoded] = b;
                n++;
            }
        }
    }

    std::vector<int> BPETokenizer::encode(const std::string &text,
                                          bool add_bos,
                                          bool add_eos) const
    {
        std::vector<int> tokens;

        if (add_bos)
        {
            tokens.push_back(bos_token_);
        }

        if (text.empty())
        {
            if (add_eos)
            {
                tokens.push_back(eos_token_);
            }
            return tokens;
        }

        // Apply BPE tokenization
        auto bpe_tokens = applyBPE(text);

        // Append BPE tokens
        tokens.insert(tokens.end(), bpe_tokens.begin(), bpe_tokens.end());

        if (add_eos)
        {
            tokens.push_back(eos_token_);
        }

        return tokens;
    }

    std::string BPETokenizer::decode(const std::vector<int> &tokens,
                                     bool remove_special) const
    {
        std::string result;

        for (int token_id : tokens)
        {
            // Skip special tokens if requested
            if (remove_special &&
                (token_id == bos_token_ || token_id == eos_token_ || token_id == pad_token_))
            {
                continue;
            }

            if (token_id >= 0 && token_id < static_cast<int>(vocab_.size()))
            {
                result += vocab_[token_id];
            }
        }

        // Decode byte-level representation back to UTF-8
        result = unicodeToBytes(result);

        return result;
    }

    std::string BPETokenizer::decode_token(int token) const
    {
        if (token >= 0 && token < static_cast<int>(vocab_.size()))
        {
            return unicodeToBytes(vocab_[token]);
        }
        return "";
    }

    std::vector<int> BPETokenizer::applyBPE(const std::string &text) const
    {
        // Optimized BPE implementation using priority queue
        // Time complexity: O(n log n) instead of O(n²)
        //
        // Algorithm:
        // 1. Initialize tokens from bytes
        // 2. Build priority queue of all valid merges (sorted by rank)
        // 3. Pop best merge, apply it, update affected neighbors in queue
        // 4. Repeat until no more merges

        if (text.empty())
        {
            return {};
        }

        // Convert text to initial token sequence (one token per byte)
        std::vector<int> tokens;
        tokens.reserve(text.size());
        for (unsigned char c : text)
        {
            int id = byte_to_token_id_[c];
            if (id != -1)
            {
                tokens.push_back(id);
            }
            else
            {
                // Unknown byte - try to find it in vocab directly
                std::string byte_str = byte_encoder_[c];
                auto it = vocab_map_.find(byte_str);
                if (it != vocab_map_.end())
                {
                    tokens.push_back(it->second);
                }
                // else: skip unknown bytes (shouldn't happen with proper vocab)
            }
        }

        if (tokens.size() <= 1)
        {
            return tokens;
        }

        // Use a linked-list style structure for efficient merging
        // Each element stores: token_id, prev_idx, next_idx, is_valid
        struct Node
        {
            int token_id;
            int prev; // -1 if first
            int next; // -1 if last
            bool valid;
        };

        std::vector<Node> nodes(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i)
        {
            nodes[i].token_id = tokens[i];
            nodes[i].prev = (i > 0) ? static_cast<int>(i - 1) : -1;
            nodes[i].next = (i < tokens.size() - 1) ? static_cast<int>(i + 1) : -1;
            nodes[i].valid = true;
        }

        // Priority queue: (rank, position) - lower rank = higher priority
        // We use negative rank so std::priority_queue (max-heap) gives us min-rank first
        using PQEntry = std::pair<int, int>; // (rank, position)
        std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;

        // Helper to add a merge candidate to the queue
        auto add_merge_candidate = [&](int pos)
        {
            if (pos < 0 || !nodes[pos].valid)
                return;
            int next_pos = nodes[pos].next;
            if (next_pos < 0 || !nodes[next_pos].valid)
                return;

            auto it = merge_ranks_int_.find({nodes[pos].token_id, nodes[next_pos].token_id});
            if (it != merge_ranks_int_.end())
            {
                pq.push({it->second, pos});
            }
        };

        // Initialize queue with all valid merge pairs
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            add_merge_candidate(static_cast<int>(i));
        }

        // Process merges
        while (!pq.empty())
        {
            auto [rank, pos] = pq.top();
            pq.pop();

            // Skip if this node was already merged/invalidated
            if (!nodes[pos].valid)
                continue;
            int next_pos = nodes[pos].next;
            if (next_pos < 0 || !nodes[next_pos].valid)
                continue;

            // Verify the merge is still valid (tokens haven't changed)
            auto it = merge_ranks_int_.find({nodes[pos].token_id, nodes[next_pos].token_id});
            if (it == merge_ranks_int_.end() || it->second != rank)
                continue;

            // Get the merged token
            int new_token_id = merge_result_ids_[rank];
            if (new_token_id == -1)
                continue;

            // Apply the merge: update pos, invalidate next_pos
            nodes[pos].token_id = new_token_id;
            nodes[next_pos].valid = false;

            // Update linked list
            int new_next = nodes[next_pos].next;
            nodes[pos].next = new_next;
            if (new_next >= 0)
            {
                nodes[new_next].prev = pos;
            }

            // Add new merge candidates for affected pairs
            add_merge_candidate(nodes[pos].prev); // prev + pos (now merged)
            add_merge_candidate(pos);             // pos (merged) + new_next
        }

        // Collect valid tokens
        std::vector<int> result;
        result.reserve(tokens.size()); // Upper bound
        for (int i = 0; i >= 0 && static_cast<size_t>(i) < nodes.size();)
        {
            if (nodes[i].valid)
            {
                result.push_back(nodes[i].token_id);
            }
            i = nodes[i].next;
            if (i < 0)
                break;
        }

        // Handle edge case: if linked list traversal missed the first valid node
        if (result.empty())
        {
            for (const auto &node : nodes)
            {
                if (node.valid)
                {
                    result.push_back(node.token_id);
                }
            }
        }

        return result;
    }

    std::string BPETokenizer::bytesToUnicode(const std::string &text) const
    {
        std::string result;
        result.reserve(text.size());
        for (unsigned char c : text)
        {
            result += byte_encoder_[c];
        }
        return result;
    }

    std::string BPETokenizer::unicodeToBytes(const std::string &text) const
    {
        std::string result;
        result.reserve(text.size());

        for (size_t i = 0; i < text.size();)
        {
            unsigned char b1 = static_cast<unsigned char>(text[i]);

            // Check for 3-byte UTF-8 sequence FIRST (for higher codepoints)
            // UTF-8 pattern: 1110xxxx 10xxxxxx 10xxxxxx
            if (i + 2 < text.size() &&
                (b1 & 0xF0) == 0xE0 &&
                (static_cast<unsigned char>(text[i + 1]) & 0xC0) == 0x80 &&
                (static_cast<unsigned char>(text[i + 2]) & 0xC0) == 0x80)
            {
                std::string utf8_char(text.data() + i, 3);
                auto utf8_it = byte_decoder_.find(utf8_char);
                if (utf8_it != byte_decoder_.end())
                {
                    result += static_cast<char>(utf8_it->second);
                    i += 3;
                    continue;
                }
                // Not in byte_decoder - pass through as-is (real UTF-8 char)
                result += utf8_char;
                i += 3;
                continue;
            }

            // Check for 2-byte UTF-8 sequence (for GPT-2 encoded non-printable bytes)
            // UTF-8 pattern: 110xxxxx 10xxxxxx
            // This includes Ġ (U+0120) = \xc4\xa0 which maps to space
            if (i + 1 < text.size() && (b1 & 0xE0) == 0xC0)
            {
                unsigned char b2 = static_cast<unsigned char>(text[i + 1]);
                if ((b2 & 0xC0) == 0x80)
                {
                    std::string utf8_char(text.data() + i, 2);
                    auto utf8_it = byte_decoder_.find(utf8_char);
                    if (utf8_it != byte_decoder_.end())
                    {
                        result += static_cast<char>(utf8_it->second);
                        i += 2;
                        continue;
                    }
                    // Not in byte_decoder - pass through as-is (real UTF-8 char)
                    result += utf8_char;
                    i += 2;
                    continue;
                }
            }

            // Single-byte: try lookup in byte_decoder (for printable ASCII)
            std::string ch(1, text[i]);
            auto it = byte_decoder_.find(ch);
            if (it != byte_decoder_.end())
            {
                result += static_cast<char>(it->second);
                ++i;
                continue;
            }

            // No match found - pass through as-is
            result += text[i];
            ++i;
        }
        return result;
    }

    // ============================================================================
    // Chat Template Implementation
    // ============================================================================

    ChatTemplateType BPETokenizer::getChatTemplateType() const
    {
        if (chat_template_)
        {
            return chat_template_->type();
        }
        return ChatTemplateType::UNKNOWN;
    }

    void BPETokenizer::setChatTemplate(std::unique_ptr<ChatTemplate> tmpl)
    {
        chat_template_ = std::move(tmpl);
        if (chat_template_)
        {
            LOG_DEBUG("[BPETokenizer] Chat template set to: " << chatTemplateTypeName(chat_template_->type()));
        }
    }

    std::vector<int> BPETokenizer::encodeChat(
        const std::vector<ChatMessage> &messages,
        bool add_generation_prompt) const
    {
        // Apply template to get formatted text
        std::string formatted = applyTemplate(messages, add_generation_prompt);

        // Encode the formatted text
        // Note: We don't add BOS here because the template typically handles it
        return encode(formatted, false, false);
    }

    std::string BPETokenizer::applyTemplate(
        const std::vector<ChatMessage> &messages,
        bool add_generation_prompt) const
    {
        if (!chat_template_)
        {
            LOG_WARN("[BPETokenizer] No chat template available, using fallback format");
            // Simple fallback: concatenate role: content pairs
            std::string result;
            for (const auto &msg : messages)
            {
                result += msg.role + ": " + msg.content + "\n";
            }
            if (add_generation_prompt)
            {
                result += "assistant: ";
            }
            return result;
        }

        return chat_template_->apply(messages, add_generation_prompt);
    }

    // ============================================================================
    // Factory Function
    // ============================================================================

    std::shared_ptr<ITokenizer> createTokenizer(std::shared_ptr<ModelContext> model_ctx)
    {
        // For now, all architectures use BPETokenizer
        // In future, could add architecture detection and dispatch to
        // specialized implementations (e.g., SentencePiece for some models)
        return BPETokenizer::create(model_ctx);
    }

} // namespace llaminar2
