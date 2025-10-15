#include "gguf_tokenizer.h"
#include "../logger.h"
#include <sstream>
#include <algorithm>
#include <regex>

namespace llaminar
{
    namespace chat
    {

        GGUFTokenizer::GGUFTokenizer()
            : tokenizer_type_(TokenizerType::NONE), vocab_loaded_(false), bpe_processor_(std::make_unique<BPEProcessor>()), bos_token_id_(-1), eos_token_id_(-1), pad_token_id_(-1), unk_token_id_(-1), nl_token_id_(-1)
        {
        }

        bool GGUFTokenizer::loadVocabulary(const ModelLoader &model)
        {
            try
            {
                LOG_INFO("Loading vocabulary from GGUF model...");

                // Load tokenizer metadata first
                loadTokenizerMetadata(model);

                // Load vocabulary tokens
                loadVocabularyTokens(model);

                // Initialize special tokens
                initializeSpecialTokens(model);

                // Initialize BPE processor if using BPE tokenization
                if (tokenizer_type_ == TokenizerType::BPE)
                {
                    initializeBPEProcessor(model);
                }

                vocab_loaded_ = true;
                LOG_INFO("Vocabulary loaded successfully: " << id_to_token_.size() << " tokens");
                return true;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to load vocabulary: " << e.what());
                return false;
            }
        }

        void GGUFTokenizer::loadTokenizerMetadata(const ModelLoader &model)
        {
            const auto &gguf_model = model.getModel();

            // Get tokenizer model type
            if (gguf_model.hasMetadata("tokenizer.ggml.model"))
            {
                auto it = gguf_model.metadata.find("tokenizer.ggml.model");
                if (it != gguf_model.metadata.end())
                {
                    tokenizer_model_ = it->second.asString();
                    LOG_INFO("Tokenizer model: " << tokenizer_model_);
                    detectTokenizerType(tokenizer_model_);
                }
            }
            else
            {
                LOG_WARN("No tokenizer model metadata found, assuming SPM");
                tokenizer_type_ = TokenizerType::SPM;
                tokenizer_model_ = "llama";
            }
        }

        void GGUFTokenizer::detectTokenizerType(const std::string &model_name)
        {
            if (model_name == "llama")
            {
                tokenizer_type_ = TokenizerType::SPM;
                // Default Llama special tokens
                bos_token_id_ = 1;
                eos_token_id_ = 2;
                unk_token_id_ = 0;
            }
            else if (model_name == "gpt2")
            {
                tokenizer_type_ = TokenizerType::BPE;
                bos_token_id_ = 11;
                eos_token_id_ = 11;
                unk_token_id_ = -1;
            }
            else if (model_name == "bert")
            {
                tokenizer_type_ = TokenizerType::WPM;
                bos_token_id_ = 101;
                eos_token_id_ = -1;
                unk_token_id_ = 100;
                pad_token_id_ = 0;
            }
            else
            {
                LOG_WARN("Unknown tokenizer model: " << model_name << ", defaulting to SPM");
                tokenizer_type_ = TokenizerType::SPM;
                bos_token_id_ = 1;
                eos_token_id_ = 2;
                unk_token_id_ = 0;
            }
        }

        void GGUFTokenizer::loadVocabularyTokens(const ModelLoader &model)
        {
            const auto &gguf_model = model.getModel();

            // Get token list from model metadata
            if (!gguf_model.hasMetadata("tokenizer.ggml.tokens"))
            {
                throw std::runtime_error("No tokenizer tokens found in model metadata");
            }

            const auto &token_list = gguf_model.token_list;
            if (token_list.empty())
            {
                throw std::runtime_error("Token list is empty");
            }

            LOG_INFO("Loading " << token_list.size() << " tokens from vocabulary");

            // Reserve space for tokens
            id_to_token_.reserve(token_list.size());
            token_to_id_.reserve(token_list.size());

            // Load tokens with their IDs
            for (size_t i = 0; i < token_list.size(); ++i)
            {
                const std::string &token_text = token_list[i];

                // Create token data
                TokenData token_data;
                token_data.text = token_text;
                token_data.score = 0.0f; // Default score
                token_data.type = 1;     // Default type (normal)

                // Add to mappings
                id_to_token_.push_back(token_data);
                token_to_id_[token_text] = static_cast<int32_t>(i);
            }

            LOG_INFO("Token mappings created: " << id_to_token_.size() << " tokens");
        }

        void GGUFTokenizer::initializeSpecialTokens(const ModelLoader &model)
        {
            const auto &gguf_model = model.getModel();

            // Try to find BOS token
            if (gguf_model.hasMetadata("tokenizer.ggml.bos_token_id"))
            {
                auto it = gguf_model.metadata.find("tokenizer.ggml.bos_token_id");
                if (it != gguf_model.metadata.end())
                {
                    bos_token_id_ = it->second.as<int32_t>();
                }
            }

            // Try to find EOS token
            if (gguf_model.hasMetadata("tokenizer.ggml.eos_token_id"))
            {
                auto it = gguf_model.metadata.find("tokenizer.ggml.eos_token_id");
                if (it != gguf_model.metadata.end())
                {
                    eos_token_id_ = it->second.as<int32_t>();
                }
            }

            // Try to find UNK token
            if (gguf_model.hasMetadata("tokenizer.ggml.unknown_token_id"))
            {
                auto it = gguf_model.metadata.find("tokenizer.ggml.unknown_token_id");
                if (it != gguf_model.metadata.end())
                {
                    unk_token_id_ = it->second.as<int32_t>();
                }
            }

            // Try to find PAD token
            if (gguf_model.hasMetadata("tokenizer.ggml.padding_token_id"))
            {
                auto it = gguf_model.metadata.find("tokenizer.ggml.padding_token_id");
                if (it != gguf_model.metadata.end())
                {
                    pad_token_id_ = it->second.as<int32_t>();
                }
            }

            // Find newline token
            auto nl_it = token_to_id_.find("\n");
            if (nl_it != token_to_id_.end())
            {
                nl_token_id_ = nl_it->second;
            }
            else
            {
                // Try to find encoded newline like "<0x0A>"
                auto encoded_nl_it = token_to_id_.find("<0x0A>");
                if (encoded_nl_it != token_to_id_.end())
                {
                    nl_token_id_ = encoded_nl_it->second;
                }
            }

            LOG_INFO("Special tokens initialized - BOS: " << bos_token_id_ << ", EOS: " << eos_token_id_ << ", UNK: " << unk_token_id_ << ", PAD: " << pad_token_id_ << ", NL: " << nl_token_id_);
        }

        std::vector<int32_t> GGUFTokenizer::tokenize(const std::string &text)
        {
            if (!vocab_loaded_)
            {
                LOG_ERROR("Tokenizer vocabulary not loaded");
                return {};
            }

            // Use appropriate tokenization method based on tokenizer type
            if (tokenizer_type_ == TokenizerType::BPE)
            {
                return tokenizeBPE(text);
            }
            else
            {
                // Fallback to simple tokenization for other types
                return tokenizeSimple(text);
            }
        }

        std::vector<int32_t> GGUFTokenizer::tokenizeSimple(const std::string &text)
        {
            std::vector<int32_t> tokens;

            // Simple space-based tokenization for now
            // TODO: Implement proper BPE/SentencePiece tokenization
            std::istringstream iss(text);
            std::string word;

            while (iss >> word)
            {
                // Try to find exact token match
                auto it = token_to_id_.find(word);
                if (it != token_to_id_.end())
                {
                    tokens.push_back(it->second);
                }
                else
                {
                    // Try to find token by character splitting
                    for (char c : word)
                    {
                        std::string char_str(1, c);
                        auto char_it = token_to_id_.find(char_str);
                        if (char_it != token_to_id_.end())
                        {
                            tokens.push_back(char_it->second);
                        }
                        else
                        {
                            // Use UNK token if available
                            if (unk_token_id_ != -1)
                            {
                                tokens.push_back(unk_token_id_);
                            }
                        }
                    }
                }
            }

            return tokens;
        }

        std::string GGUFTokenizer::detokenize(const std::vector<int32_t> &tokens)
        {
            if (!vocab_loaded_)
            {
                LOG_ERROR("Tokenizer vocabulary not loaded");
                return "";
            }

            // Use appropriate detokenization method based on tokenizer type
            if (tokenizer_type_ == TokenizerType::BPE)
            {
                return detokenizeBPE(tokens);
            }
            else
            {
                // Fallback to simple detokenization for other types
                return detokenizeSimple(tokens);
            }
        }

        std::string GGUFTokenizer::detokenizeSimple(const std::vector<int32_t> &tokens)
        {
            std::ostringstream result;

            for (size_t i = 0; i < tokens.size(); ++i)
            {
                int32_t token_id = tokens[i];

                if (token_id >= 0 && token_id < static_cast<int32_t>(id_to_token_.size()))
                {
                    const std::string &token_text = id_to_token_[token_id].text;

                    // Simple space separation (improve this based on tokenizer type)
                    if (i > 0 && !token_text.empty() && token_text[0] != ' ')
                    {
                        result << " ";
                    }

                    result << token_text;
                }
                else
                {
                    LOG_WARN("Invalid token ID in detokenization: " << token_id);
                    result << "[UNK]";
                }
            }

            return result.str();
        }

        std::string GGUFTokenizer::applyTemplate(const std::vector<ChatMessage> &messages, bool add_generation_prompt)
        {
            std::ostringstream formatted;

            // Apply chat template based on tokenizer type
            // For now, use a generic ChatML-style template
            for (const auto &message : messages)
            {
                formatted << "<|im_start|>" << message.role << "\n"
                          << message.content << "\n"
                          << "<|im_end|>\n";
            }

            if (add_generation_prompt)
            {
                formatted << "<|im_start|>assistant\n";
            }

            return formatted.str();
        }

        int32_t GGUFTokenizer::getSpecialToken(const std::string &token_name)
        {
            if (token_name == "bos")
                return bos_token_id_;
            if (token_name == "eos")
                return eos_token_id_;
            if (token_name == "pad")
                return pad_token_id_;
            if (token_name == "unk")
                return unk_token_id_;
            if (token_name == "nl" || token_name == "newline")
                return nl_token_id_;
            return -1;
        }

        size_t GGUFTokenizer::getVocabSize() const
        {
            return id_to_token_.size();
        }

        bool GGUFTokenizer::isReady() const
        {
            return vocab_loaded_ && !id_to_token_.empty();
        }

        std::string GGUFTokenizer::getTokenString(int32_t token_id)
        {
            if (token_id >= 0 && token_id < static_cast<int32_t>(id_to_token_.size()))
            {
                return id_to_token_[token_id].text;
            }
            return "[INVALID:" + std::to_string(token_id) + "]";
        }

        void GGUFTokenizer::initializeBPEProcessor(const ModelLoader &model)
        {
            if (!bpe_processor_)
            {
                LOG_ERROR("BPE processor not initialized");
                return;
            }

            // Create token string vector for BPE processor
            std::vector<std::string> token_strings;
            token_strings.reserve(id_to_token_.size());

            for (const auto &token_data : id_to_token_)
            {
                token_strings.push_back(token_data.text);
            }

            // Load actual BPE merge rules from GGUF metadata
            const auto &gguf_model = model.getModel();

            std::vector<std::string> merge_rules;
            if (gguf_model.hasMetadata("tokenizer.ggml.merges"))
            {
                auto it = gguf_model.metadata.find("tokenizer.ggml.merges");
                if (it != gguf_model.metadata.end())
                {
                    try
                    {
                        merge_rules = it->second.asStringArray();
                        LOG_INFO("Loaded " << merge_rules.size() << " BPE merge rules from GGUF metadata");
                    }
                    catch (const std::exception &e)
                    {
                        LOG_ERROR("Failed to parse BPE merge rules: " << e.what());
                    }
                }
            }
            else
            {
                LOG_WARN("No BPE merge rules found in GGUF metadata (tokenizer.ggml.merges)");
            }

            // Initialize BPE processor with vocabulary and merge rules
            bpe_processor_->initialize(token_to_id_, token_strings, merge_rules);

            LOG_INFO("BPE processor initialized for " << tokenizer_model_ << " tokenizer");
        }

        std::vector<int32_t> GGUFTokenizer::tokenizeBPE(const std::string &text)
        {
            if (!bpe_processor_)
            {
                LOG_WARN("BPE processor not available, falling back to simple tokenization");
                return tokenizeSimple(text);
            }

            return bpe_processor_->tokenize(text, token_to_id_, unk_token_id_);
        }

        std::string GGUFTokenizer::detokenizeBPE(const std::vector<int32_t> &tokens)
        {
            if (!bpe_processor_)
            {
                LOG_WARN("BPE processor not available, falling back to simple detokenization");
                return detokenizeSimple(tokens);
            }

            // Create token string vector for detokenization
            std::vector<std::string> token_strings;
            token_strings.reserve(id_to_token_.size());

            for (const auto &token_data : id_to_token_)
            {
                token_strings.push_back(token_data.text);
            }

            return bpe_processor_->detokenize(tokens, token_strings);
        }

        std::unique_ptr<TokenizerInterface> createTokenizer(const ModelLoader &model)
        {
            auto tokenizer = std::make_unique<GGUFTokenizer>();
            if (tokenizer->loadVocabulary(model))
            {
                return std::move(tokenizer);
            }
            return nullptr;
        }

    } // namespace chat
} // namespace llaminar
