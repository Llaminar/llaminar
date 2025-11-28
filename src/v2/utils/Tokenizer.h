/**
 * @file Tokenizer.h
 * @brief Token encoding/decoding for LLM text generation
 * @author David Sanftenberg
 * @date 2025
 *
 * Native tokenizer implementation that reads vocabulary directly from
 * GGUF metadata, avoiding duplicate model loading.
 */

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <unordered_map>
#include <map>

namespace llaminar2
{
    // Forward declaration
    class ModelContext;

    /**
     * @brief Abstract tokenizer interface
     *
     * Provides encode/decode operations for any model architecture.
     * Implementations handle architecture-specific details.
     */
    class ITokenizer
    {
    public:
        virtual ~ITokenizer() = default;

        /**
         * @brief Encode text into tokens
         *
         * @param text Input text
         * @param add_bos Whether to add BOS token at start
         * @param add_eos Whether to add EOS token at end
         * @return Vector of token IDs
         */
        virtual std::vector<int> encode(const std::string &text,
                                        bool add_bos = true,
                                        bool add_eos = false) const = 0;

        /**
         * @brief Decode tokens into text
         *
         * @param tokens Vector of token IDs
         * @param remove_special Whether to skip special tokens (BOS, EOS)
         * @return Decoded text string
         */
        virtual std::string decode(const std::vector<int> &tokens,
                                   bool remove_special = true) const = 0;

        /**
         * @brief Decode a single token into text
         *
         * @param token Token ID
         * @return Decoded text for this token
         */
        virtual std::string decode_token(int token) const = 0;

        /**
         * @brief Get BOS (beginning-of-sequence) token ID
         */
        virtual int bos_token() const = 0;

        /**
         * @brief Get EOS (end-of-sequence) token ID
         */
        virtual int eos_token() const = 0;

        /**
         * @brief Get vocabulary size
         */
        virtual int vocab_size() const = 0;
    };

    /**
     * @brief Native BPE tokenizer implementation
     *
     * Reads vocabulary and merge rules directly from GGUF metadata.
     * Implements Byte Pair Encoding (BPE) algorithm for text tokenization.
     *
     * Supports GPT-2 style BPE tokenization used by Qwen, LLaMA, and similar models.
     */
    class BPETokenizer : public ITokenizer
    {
    public:
        /**
         * @brief Create tokenizer from ModelContext
         *
         * Extracts vocabulary from GGUF metadata without re-loading the file.
         *
         * @param model_ctx Model context with loaded GGUF metadata
         * @return Shared pointer to tokenizer, or nullptr on error
         */
        static std::shared_ptr<BPETokenizer> create(std::shared_ptr<ModelContext> model_ctx);

        /**
         * @brief Destructor
         */
        ~BPETokenizer() override = default;

        // ITokenizer implementation
        std::vector<int> encode(const std::string &text,
                                bool add_bos = true,
                                bool add_eos = false) const override;

        std::string decode(const std::vector<int> &tokens,
                           bool remove_special = true) const override;

        std::string decode_token(int token) const override;

        int bos_token() const override { return bos_token_; }
        int eos_token() const override { return eos_token_; }
        int vocab_size() const override { return vocab_.size(); }

    private:
        /**
         * @brief Private constructor - use create() factory method
         */
        BPETokenizer() = default;

        /**
         * @brief Initialize tokenizer from GGUF metadata
         */
        bool initializeFromMetadata(std::shared_ptr<ModelContext> model_ctx);

        /**
         * @brief Initialize byte encoder/decoder
         */
        void initializeByteEncoder();

        /**
         * @brief Apply BPE merges to text
         */
        std::vector<int> applyBPE(const std::string &text) const;

        /**
         * @brief Byte-level encoding (for handling any UTF-8)
         */
        std::string bytesToUnicode(const std::string &text) const;
        std::string unicodeToBytes(const std::string &text) const;

        // Vocabulary data
        std::vector<std::string> vocab_;
        std::unordered_map<std::string, int> vocab_map_;
        std::vector<std::pair<std::string, std::string>> merges_;
        std::map<std::pair<std::string, std::string>, int> merge_ranks_;
        std::map<std::pair<int, int>, int> merge_ranks_int_;
        std::vector<int> merge_result_ids_;
        std::vector<int> byte_to_token_id_;

        std::vector<std::string> byte_encoder_;
        std::unordered_map<std::string, int> byte_decoder_;

        int bos_token_ = 0;
        int eos_token_ = 0;
        int pad_token_ = 0;
    };

    /**
     * @brief Factory function to create appropriate tokenizer for model
     *
     * @param model_ctx Model context with loaded GGUF metadata
     * @return Shared pointer to tokenizer, or nullptr on error
     *
     * Currently creates BPETokenizer for all architectures.
     * In future, could dispatch to architecture-specific implementations
     * if needed (e.g., SentencePiece for some models).
     */
    std::shared_ptr<ITokenizer> createTokenizer(std::shared_ptr<ModelContext> model_ctx);

} // namespace llaminar2
