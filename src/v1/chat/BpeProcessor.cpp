#include "BpeProcessor.h"
#include "../Logger.h"
#include <regex>
#include <algorithm>
#include <sstream>
#include <cctype>

namespace llaminar
{
    namespace chat
    {
        void BPEProcessor::initialize(const std::unordered_map<std::string, int32_t> &token_to_id,
                                      const std::vector<std::string> &id_to_token,
                                      const std::vector<std::string> &merge_rules)
        {
            LOG_DEBUG("Initializing BPE processor with " << token_to_id.size() << " tokens");

            // Initialize byte-level mappings
            initializeByteMapping();

            // Parse and load actual merge rules from GGUF
            if (!merge_rules.empty())
            {
                parseMergeRules(merge_rules);
                LOG_INFO("BPE processor initialized with " << bpe_merges_.size() << " merge rules from GGUF");
            }
            else
            {
                // Fallback: try to extract merges from vocabulary (unreliable)
                LOG_WARN("No merge rules provided, attempting to extract from vocabulary (may be incorrect)");
                extractBPEMerges(token_to_id);
                LOG_INFO("BPE processor initialized with " << bpe_merges_.size() << " extracted merge rules");
            }
        }

        void BPEProcessor::parseMergeRules(const std::vector<std::string> &merge_rules)
        {
            bpe_merges_.clear();
            merge_ranks_.clear();

            int32_t rank = 0;
            for (const std::string &rule : merge_rules)
            {
                // Parse merge rule format: "token1 token2"
                size_t space_pos = rule.find(' ');
                if (space_pos == std::string::npos || space_pos == 0 || space_pos == rule.length() - 1)
                {
                    LOG_WARN("Invalid merge rule format: '" << rule << "', skipping");
                    continue;
                }

                std::string left = rule.substr(0, space_pos);
                std::string right = rule.substr(space_pos + 1);

                std::pair<std::string, std::string> merge_pair = {left, right};
                bpe_merges_.push_back(merge_pair);
                merge_ranks_[merge_pair] = rank++;
            }

            LOG_DEBUG("Parsed " << bpe_merges_.size() << " merge rules");
        }

        std::vector<int32_t> BPEProcessor::tokenize(const std::string &text,
                                                    const std::unordered_map<std::string, int32_t> &token_to_id,
                                                    int32_t unk_token_id)
        {
            if (text.empty())
            {
                return {};
            }

            std::vector<int32_t> tokens;

            // GPT-2 style: Convert spaces to Ġ (U+0120) and then process
            // The vocabulary has tokens like "Ġis", "Ġthe" with the special space marker
            std::string preprocessed;
            preprocessed.reserve(text.length() + 20);

            bool at_start = true;
            for (char c : text)
            {
                if (c == ' ')
                {
                    // Don't add space marker at the very start
                    // Next non-space character will get the marker
                    at_start = false;
                }
                else
                {
                    if (!at_start)
                    {
                        // This character follows a space, prepend Ġ marker
                        preprocessed += '\xC4'; // UTF-8 encoding of U+0120: 0xC4 0xA0
                        preprocessed += '\xA0';
                        at_start = true; // Only add marker once per word
                    }
                    preprocessed += c;
                }
            }

            // If text started with non-space, no marker needed for first char
            // This is handled by at_start flag

            // Apply the corrected preprocessing
            std::string processed_text;
            if (preprocessed.empty())
            {
                processed_text = text;
            }
            else
            {
                processed_text = preprocessed;
            }

            // Re-do with simpler logic matching GPT-2
            processed_text.clear();
            for (size_t i = 0; i < text.length(); ++i)
            {
                if (i > 0 && text[i - 1] == ' ' && text[i] != ' ')
                {
                    // Previous char was space, current is not - add Ġ marker
                    processed_text += '\xC4'; // Ġ in UTF-8: 0xC4 0xA0
                    processed_text += '\xA0';
                }
                if (text[i] != ' ')
                {
                    processed_text += text[i];
                }
            }

            // Split into individual characters first (byte-level BPE starts from chars)
            std::vector<std::string> char_tokens;
            for (size_t i = 0; i < processed_text.length();)
            {
                // Handle multi-byte UTF-8 sequences
                if ((processed_text[i] & 0x80) == 0)
                {
                    // Single byte ASCII
                    char_tokens.push_back(processed_text.substr(i, 1));
                    i += 1;
                }
                else if ((processed_text[i] & 0xE0) == 0xC0)
                {
                    // 2-byte UTF-8 (includes Ġ)
                    if (i + 1 < processed_text.length())
                    {
                        char_tokens.push_back(processed_text.substr(i, 2));
                        i += 2;
                    }
                    else
                    {
                        char_tokens.push_back(processed_text.substr(i, 1));
                        i += 1;
                    }
                }
                else if ((processed_text[i] & 0xF0) == 0xE0)
                {
                    // 3-byte UTF-8
                    if (i + 2 < processed_text.length())
                    {
                        char_tokens.push_back(processed_text.substr(i, 3));
                        i += 3;
                    }
                    else
                    {
                        char_tokens.push_back(processed_text.substr(i, 1));
                        i += 1;
                    }
                }
                else
                {
                    // 4-byte UTF-8 or error
                    if (i + 3 < processed_text.length() && (processed_text[i] & 0xF8) == 0xF0)
                    {
                        char_tokens.push_back(processed_text.substr(i, 4));
                        i += 4;
                    }
                    else
                    {
                        char_tokens.push_back(processed_text.substr(i, 1));
                        i += 1;
                    }
                }
            }

            // Apply BPE merges
            char_tokens = applyBPE(char_tokens);

            // Convert tokens to IDs
            for (const std::string &token : char_tokens)
            {
                auto it = token_to_id.find(token);
                if (it != token_to_id.end())
                {
                    tokens.push_back(it->second);
                }
                else if (unk_token_id != -1)
                {
                    LOG_WARN("Unknown token: '" << token << "' (using UNK)");
                    tokens.push_back(unk_token_id);
                }
                // Skip unknown tokens if no UNK token available
            }

            return tokens;
        }

        std::string BPEProcessor::detokenize(const std::vector<int32_t> &tokens,
                                             const std::vector<std::string> &id_to_token)
        {
            if (tokens.empty())
            {
                return "";
            }

            std::ostringstream result;

            for (int32_t token_id : tokens)
            {
                if (token_id >= 0 && token_id < static_cast<int32_t>(id_to_token.size()))
                {
                    result << id_to_token[token_id];
                }
                // Skip invalid token IDs
            }

            // Convert from byte-level representation back to text
            return bytesToText(result.str());
        }

        void BPEProcessor::initializeByteMapping()
        {
            // Standard GPT-2 style byte-level BPE mapping
            // Maps bytes 0-255 to Unicode characters that are safe for BPE

            // Printable ASCII characters map to themselves
            for (int i = 33; i <= 126; ++i)
            {
                if (i != 92)
                { // Skip backslash
                    byte_to_char_[i] = std::string(1, static_cast<char>(i));
                    char_to_byte_[std::string(1, static_cast<char>(i))] = static_cast<uint8_t>(i);
                }
            }

            // Space maps to itself
            byte_to_char_[32] = " ";
            char_to_byte_[" "] = 32;

            // Map remaining bytes to Unicode private use area
            int unicode_start = 256;
            for (int i = 0; i <= 255; ++i)
            {
                if (byte_to_char_.find(i) == byte_to_char_.end())
                {
                    // Use Unicode private use characters for non-printable bytes
                    std::string unicode_char;
                    int unicode_point = unicode_start++;

                    // Simple single-byte mapping for now
                    // In a full implementation, would use proper Unicode encoding
                    unicode_char = "Ġ" + std::to_string(i); // Simple placeholder

                    byte_to_char_[i] = unicode_char;
                    char_to_byte_[unicode_char] = static_cast<uint8_t>(i);
                }
            }
        }

        void BPEProcessor::extractBPEMerges(const std::unordered_map<std::string, int32_t> &token_to_id)
        {
            // Extract merge rules by analyzing the vocabulary
            // This is a simplified approach - in practice, merge rules are usually provided separately

            std::vector<std::string> all_tokens;
            all_tokens.reserve(token_to_id.size());

            for (const auto &pair : token_to_id)
            {
                all_tokens.push_back(pair.first);
            }

            // Sort tokens by ID to maintain merge order
            std::sort(all_tokens.begin(), all_tokens.end(),
                      [&token_to_id](const std::string &a, const std::string &b)
                      {
                          return token_to_id.at(a) < token_to_id.at(b);
                      });

            // Extract potential merges from multi-character tokens
            int32_t merge_rank = 0;
            for (const std::string &token : all_tokens)
            {
                if (token.length() > 1)
                {
                    // Try to split token into existing smaller tokens
                    for (size_t split = 1; split < token.length(); ++split)
                    {
                        std::string left = token.substr(0, split);
                        std::string right = token.substr(split);

                        // Check if both parts exist in vocabulary
                        if (token_to_id.find(left) != token_to_id.end() &&
                            token_to_id.find(right) != token_to_id.end())
                        {

                            std::pair<std::string, std::string> merge_pair = {left, right};

                            // Add merge rule if not already present
                            if (merge_ranks_.find(merge_pair) == merge_ranks_.end())
                            {
                                bpe_merges_.push_back(merge_pair);
                                merge_ranks_[merge_pair] = merge_rank++;
                            }
                            break; // Use first valid split
                        }
                    }
                }
            }

            LOG_DEBUG("Extracted " << bpe_merges_.size() << " BPE merge rules");
        }

        std::string BPEProcessor::textToBytes(const std::string &text)
        {
            std::string result;
            result.reserve(text.length() * 2); // Reserve space for potential expansion

            for (unsigned char byte : text)
            {
                auto it = byte_to_char_.find(byte);
                if (it != byte_to_char_.end())
                {
                    result += it->second;
                }
                else
                {
                    // Fallback for unmapped bytes
                    result += "Ġ" + std::to_string(byte);
                }
            }

            return result;
        }

        std::string BPEProcessor::bytesToText(const std::string &byte_text)
        {
            std::string result;
            result.reserve(byte_text.length());

            for (size_t i = 0; i < byte_text.length();)
            {
                // Handle GPT-2 style space marker: Ġ (U+0120) → space
                // UTF-8 encoding: 0xC4 0xA0
                if (i + 1 < byte_text.length() &&
                    static_cast<unsigned char>(byte_text[i]) == 0xC4 &&
                    static_cast<unsigned char>(byte_text[i + 1]) == 0xA0)
                {
                    result += ' ';
                    i += 2;
                    continue;
                }

                // Handle newline marker: Ċ (U+010A) → newline
                // UTF-8 encoding: 0xC4 0x8A
                if (i + 1 < byte_text.length() &&
                    static_cast<unsigned char>(byte_text[i]) == 0xC4 &&
                    static_cast<unsigned char>(byte_text[i + 1]) == 0x8A)
                {
                    result += '\n';
                    i += 2;
                    continue;
                }

                // Look for the longest match in char_to_byte mapping
                bool found = false;
                for (size_t len = std::min(byte_text.length() - i, size_t(10)); len > 0; --len)
                {
                    std::string substr = byte_text.substr(i, len);
                    auto it = char_to_byte_.find(substr);
                    if (it != char_to_byte_.end())
                    {
                        result += static_cast<char>(it->second);
                        i += len;
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    // Handle unmapped characters - try to decode placeholder format
                    if (i + 1 < byte_text.length() && byte_text.substr(i, 1) == "Ġ")
                    {
                        // Try to parse "Ġ123" format
                        size_t num_start = i + 1;
                        size_t num_end = num_start;
                        while (num_end < byte_text.length() && std::isdigit(byte_text[num_end]))
                        {
                            ++num_end;
                        }

                        if (num_end > num_start)
                        {
                            int byte_val = std::stoi(byte_text.substr(num_start, num_end - num_start));
                            if (byte_val >= 0 && byte_val <= 255)
                            {
                                result += static_cast<char>(byte_val);
                                i = num_end;
                                continue;
                            }
                        }
                    }

                    // Fallback: copy character as-is
                    result += byte_text[i];
                    ++i;
                }
            }

            return result;
        }

        std::vector<std::string> BPEProcessor::applyBPE(std::vector<std::string> word)
        {
            if (word.size() <= 1)
            {
                return word;
            }

            while (true)
            {
                auto pairs = getPairs(word);
                if (pairs.empty())
                {
                    break;
                }

                // Find the pair with the highest priority (lowest rank)
                std::pair<std::string, std::string> best_pair;
                int32_t best_rank = std::numeric_limits<int32_t>::max();
                bool found_pair = false;

                for (const auto &pair : pairs)
                {
                    auto it = merge_ranks_.find(pair);
                    if (it != merge_ranks_.end() && it->second < best_rank)
                    {
                        best_pair = pair;
                        best_rank = it->second;
                        found_pair = true;
                    }
                }

                if (!found_pair)
                {
                    break;
                }

                // Apply the merge
                std::vector<std::string> new_word;
                for (size_t i = 0; i < word.size();)
                {
                    if (i + 1 < word.size() &&
                        word[i] == best_pair.first &&
                        word[i + 1] == best_pair.second)
                    {
                        // Merge this pair
                        new_word.push_back(word[i] + word[i + 1]);
                        i += 2;
                    }
                    else
                    {
                        new_word.push_back(word[i]);
                        ++i;
                    }
                }

                word = std::move(new_word);
            }

            return word;
        }

        std::unordered_set<std::pair<std::string, std::string>, BPEProcessor::PairHash>
        BPEProcessor::getPairs(const std::vector<std::string> &word)
        {
            std::unordered_set<std::pair<std::string, std::string>, PairHash> pairs;

            for (size_t i = 0; i + 1 < word.size(); ++i)
            {
                pairs.insert({word[i], word[i + 1]});
            }

            return pairs;
        }

        std::vector<std::string> BPEProcessor::splitIntoWords(const std::string &text)
        {
            std::vector<std::string> words;

            // Use regex to split on whitespace while preserving the whitespace
            // This is a simplified version - GPT-2 uses a more complex regex
            std::regex word_regex(R"([a-zA-Z]+|[0-9]+|[^\w\s]+|\s+)");
            std::sregex_iterator iter(text.begin(), text.end(), word_regex);
            std::sregex_iterator end;

            for (; iter != end; ++iter)
            {
                std::string word = iter->str();
                if (!word.empty())
                {
                    words.push_back(word);
                }
            }

            // Fallback: if regex failed, split on spaces
            if (words.empty() && !text.empty())
            {
                std::istringstream iss(text);
                std::string word;
                while (iss >> word)
                {
                    words.push_back(word);
                }
            }

            return words;
        }

        bool BPEProcessor::isWhitespace(char c)
        {
            return std::isspace(static_cast<unsigned char>(c));
        }

    } // namespace chat
} // namespace llaminar