#pragma once

#include "tokenizer_interface.h"
#include "bpe_processor.h"
#include "../model_loader.h"
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>

namespace llaminar
{
    namespace chat
    {

        enum class TokenizerType
        {
            SPM, // SentencePiece (Llama)
            BPE, // Byte Pair Encoding (GPT-2)
            WPM, // WordPiece (BERT)
            NONE // No tokenizer
        };

        struct TokenData
        {
            std::string text;
            float score;
            int32_t type;
        };

        class GGUFTokenizer : public TokenizerInterface
        {
        public:
            GGUFTokenizer();
            ~GGUFTokenizer() override = default;

            std::vector<int32_t> tokenize(const std::string &text) override;
            std::string detokenize(const std::vector<int32_t> &tokens) override;
            std::string applyTemplate(const std::vector<ChatMessage> &messages,
                                      bool add_generation_prompt = true) override;
            bool loadVocabulary(const ModelLoader &model) override;
            int32_t getSpecialToken(const std::string &token_name) override;
            size_t getVocabSize() const override;
            bool isReady() const override;
            std::string getTokenString(int32_t token_id) override;

        private:
            // Vocabulary data
            std::vector<TokenData> id_to_token_;
            std::unordered_map<std::string, int32_t> token_to_id_;

            // Tokenizer configuration
            TokenizerType tokenizer_type_;
            std::string tokenizer_model_;
            bool vocab_loaded_;

            // BPE processor for proper tokenization
            std::unique_ptr<BPEProcessor> bpe_processor_;

            // Special token IDs
            int32_t bos_token_id_;
            int32_t eos_token_id_;
            int32_t pad_token_id_;
            int32_t unk_token_id_;
            int32_t nl_token_id_;

            // Helper methods
            void loadTokenizerMetadata(const ModelLoader &model);
            void loadVocabularyTokens(const ModelLoader &model);
            void initializeSpecialTokens(const ModelLoader &model);
            void detectTokenizerType(const std::string &model_name);
            void initializeBPEProcessor(const ModelLoader &model);
            std::vector<int32_t> tokenizeSimple(const std::string &text);
            std::string detokenizeSimple(const std::vector<int32_t> &tokens);
            std::vector<int32_t> tokenizeBPE(const std::string &text);
            std::string detokenizeBPE(const std::vector<int32_t> &tokens);
        };

        std::unique_ptr<TokenizerInterface> createTokenizer(const ModelLoader &model);

    } // namespace chat
} // namespace llaminar
