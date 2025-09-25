#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace llaminar
{
    namespace chat
    {
        /**
         * BPE (Byte Pair Encoding) Processor
         * Implements standard BPE tokenization algorithm
         */
        class BPEProcessor
        {
        public:
            /**
             * Constructor
             */
            BPEProcessor() = default;

            /**
             * Destructor
             */
            ~BPEProcessor() = default;

            /**
             * Initialize BPE from vocabulary
             * @param token_to_id Map from token strings to IDs
             * @param id_to_token Vector of token strings indexed by ID
             */
            void initialize(const std::unordered_map<std::string, int32_t> &token_to_id,
                            const std::vector<std::string> &id_to_token);

            /**
             * Tokenize text using BPE algorithm
             * @param text Input text to tokenize
             * @return Vector of token IDs
             */
            std::vector<int32_t> tokenize(const std::string &text,
                                          const std::unordered_map<std::string, int32_t> &token_to_id,
                                          int32_t unk_token_id = -1);

            /**
             * Detokenize tokens back to text
             * @param tokens Vector of token IDs
             * @param id_to_token Vector of token strings
             * @return Reconstructed text
             */
            std::string detokenize(const std::vector<int32_t> &tokens,
                                   const std::vector<std::string> &id_to_token);

        private:
            /**
             * Custom hash function for string pairs
             */
            struct PairHash
            {
                std::size_t operator()(const std::pair<std::string, std::string> &p) const
                {
                    auto h1 = std::hash<std::string>{}(p.first);
                    auto h2 = std::hash<std::string>{}(p.second);
                    return h1 ^ (h2 << 1);
                }
            };

            // BPE merge pairs extracted from vocabulary
            std::vector<std::pair<std::string, std::string>> bpe_merges_;

            // Rank of each merge (priority)
            std::unordered_map<std::pair<std::string, std::string>, int32_t, PairHash> merge_ranks_;

            // Special character mappings for byte-level BPE
            std::unordered_map<uint8_t, std::string> byte_to_char_;
            std::unordered_map<std::string, uint8_t> char_to_byte_;

            /**
             * Initialize byte-level encoding mappings
             */
            void initializeByteMapping();

            /**
             * Extract BPE merges from vocabulary
             * @param token_to_id Token vocabulary
             */
            void extractBPEMerges(const std::unordered_map<std::string, int32_t> &token_to_id);

            /**
             * Convert text to byte-level representation
             * @param text Input text
             * @return Byte-level string
             */
            std::string textToBytes(const std::string &text);

            /**
             * Convert byte-level representation back to text
             * @param byte_text Byte-level string
             * @return Original text
             */
            std::string bytesToText(const std::string &byte_text);

            /**
             * Apply BPE merges to a word
             * @param word Word split into characters/subwords
             * @return Word after applying BPE merges
             */
            std::vector<std::string> applyBPE(std::vector<std::string> word);

            /**
             * Get all possible pairs in a word
             * @param word Word split into subwords
             * @return Set of adjacent pairs
             */
            std::unordered_set<std::pair<std::string, std::string>, PairHash>
            getPairs(const std::vector<std::string> &word);

            /**
             * Split text into words for processing
             * @param text Input text
             * @return Vector of words
             */
            std::vector<std::string> splitIntoWords(const std::string &text);

            /**
             * Check if a character is whitespace
             * @param c Character to check
             * @return True if whitespace
             */
            bool isWhitespace(char c);
        };

    } // namespace chat
} // namespace llaminar