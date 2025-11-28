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

        return true;
    }

    void BPETokenizer::initializeByteEncoder()
    {
        byte_encoder_.resize(256);
        // GPT-2 byte-level encoding
        int n = 0;
        for (int b = 0; b < 256; ++b)
        {
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
            {
                byte_encoder_[b] = std::string(1, static_cast<char>(b));
                byte_decoder_[std::string(1, static_cast<char>(b))] = b;
            }
            else
            {
                byte_encoder_[b] = std::string(1, static_cast<char>(256 + n));
                byte_decoder_[std::string(1, static_cast<char>(256 + n))] = b;
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
        // Simplified BPE implementation
        // For production, this should match the original tokenizer's regex splitting

        // Split text into words (simple whitespace splitting for now)
        std::vector<std::string> words;
        std::string current_word;

        for (char c : text)
        {
            if (std::isspace(c))
            {
                if (!current_word.empty())
                {
                    words.push_back(current_word);
                    current_word.clear();
                }
            }
            else
            {
                current_word += c;
            }
        }
        if (!current_word.empty())
        {
            words.push_back(current_word);
        }

        // Convert each word to byte-level representation and tokenize
        std::vector<std::vector<int>> word_results(words.size());

// Only parallelize for sufficient workload to overcome thread overhead
#pragma omp parallel for schedule(static) if (words.size() > 32)
        for (size_t i = 0; i < words.size(); ++i)
        {
            const auto &word = words[i];

            std::vector<int> word_tokens;
            word_tokens.reserve(word.size());
            for (unsigned char c : word)
            {
                int id = byte_to_token_id_[c];
                if (id != -1)
                {
                    word_tokens.push_back(id);
                }
            }

            // Apply BPE merges
            while (word_tokens.size() > 1)
            {
                // Find the highest priority merge
                int best_merge_idx = -1;
                size_t best_merge_pos = 0;

                for (size_t j = 0; j < word_tokens.size() - 1; ++j)
                {
                    auto it = merge_ranks_int_.find({word_tokens[j], word_tokens[j + 1]});
                    if (it != merge_ranks_int_.end())
                    {
                        int rank = it->second;
                        if (best_merge_idx == -1 || rank < best_merge_idx)
                        {
                            best_merge_idx = rank;
                            best_merge_pos = j;
                        }
                    }
                }

                // Apply the best merge if found
                if (best_merge_idx != -1)
                {
                    int new_token_id = merge_result_ids_[best_merge_idx];
                    if (new_token_id != -1)
                    {
                        word_tokens[best_merge_pos] = new_token_id;
                        word_tokens.erase(word_tokens.begin() + best_merge_pos + 1);
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    break;
                }
            }

            word_results[i] = std::move(word_tokens);
        }

        std::vector<int> result;
        for (const auto &tokens : word_results)
        {
            result.insert(result.end(), tokens.begin(), tokens.end());
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
        for (size_t i = 0; i < text.size(); ++i)
        {
            std::string ch(1, text[i]);
            auto it = byte_decoder_.find(ch);
            if (it != byte_decoder_.end())
            {
                result += static_cast<char>(it->second);
            }
            else
            {
                result += text[i];
            }
        }
        return result;
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
