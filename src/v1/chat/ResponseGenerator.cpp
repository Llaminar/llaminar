#include "ResponseGenerator.h"
#include "../Logger.h"
#include "../tensors/TensorFactory.h"
#include "../JsonExport.h"
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <chrono>

namespace llaminar
{
    namespace chat
    {

        ResponseGenerator::ResponseGenerator(std::shared_ptr<TokenizerInterface> tokenizer,
                                             std::shared_ptr<AbstractPipeline> pipeline,
                                             const LlaminarParams &params)
            : tokenizer_(std::move(tokenizer)), pipeline_(std::move(pipeline)), params_(params), temperature_(params.temperature >= 0.0f ? params.temperature : 0.7f), top_k_(params.top_k > 0 ? params.top_k : 40), top_p_(params.top_p > 0.0f ? params.top_p : 0.9f), max_tokens_(params.n_predict > 0 ? params.n_predict : 128)
        {
            if (!pipeline_)
            {
                throw std::invalid_argument("ResponseGenerator: pipeline cannot be null");
            }
            if (params_.model_file.empty())
            {
                throw std::invalid_argument("ResponseGenerator: model_file must be set when not providing pre-loaded weights");
            }
            LOG_INFO("ResponseGenerator will lazily load weights on first generation from: " << params_.model_file);
            // Stop tokens / sequences
            if (tokenizer_)
            {
                int32_t eos = tokenizer_->getSpecialToken("eos");
                if (eos != -1)
                    stop_tokens_.push_back(eos);
            }
            stop_sequences_ = {"<|im_end|>", "</s>", "<|endoftext|>"};
        }

        ResponseGenerator::ResponseGenerator(std::shared_ptr<TokenizerInterface> tokenizer,
                                             std::shared_ptr<AbstractPipeline> pipeline,
                                             const LlaminarParams &params,
                                             const QwenModelWeights &weights)
            : tokenizer_(std::move(tokenizer)), pipeline_(std::move(pipeline)), params_(params), weights_(weights), have_weights_(true), temperature_(params.temperature >= 0.0f ? params.temperature : 0.7f), top_k_(params.top_k > 0 ? params.top_k : 40), top_p_(params.top_p > 0.0f ? params.top_p : 0.9f), max_tokens_(params.n_predict > 0 ? params.n_predict : 128)
        {
            if (!pipeline_)
                throw std::invalid_argument("ResponseGenerator: pipeline cannot be null");
            if (tokenizer_)
            {
                int32_t eos = tokenizer_->getSpecialToken("eos");
                if (eos != -1)
                    stop_tokens_.push_back(eos);
            }
            stop_sequences_ = {"<|im_end|>", "</s>", "<|endoftext|>"};
            LOG_INFO("ResponseGenerator initialized with pre-loaded weights");
        }

        std::string ResponseGenerator::generateResponse(const std::vector<int32_t> &prompt_tokens,
                                                        ResponseCallback callback)
        {
            if (prompt_tokens.empty())
            {
                LOG_ERROR("Cannot generate response from empty prompt");
                return "";
            }

            LOG_INFO("Starting response generation for " << prompt_tokens.size() << " prompt tokens");

            try
            {
                std::vector<int32_t> generated_tokens; // tokens generated (excluding prompt)
                std::string response_text;
                std::vector<int32_t> current_sequence = prompt_tokens;

                // For JSON export if requested
                GenerationData export_data;
                bool should_export = !params_.output_json_file.empty();
                if (should_export)
                {
                    export_data.prompt_tokens = prompt_tokens;
                }

                if (!prefilled_)
                {
                    if (!ensureWeights())
                        return ""; // error logged in ensureWeights
                    if (!pipeline_->prefill(prompt_tokens, weights_, stage_ctx_))
                    {
                        LOG_ERROR("Prefill failed");
                        return "";
                    }
                    prefilled_ = true;
                }

                // Buffer for streaming - prevents partial chat markers from being displayed
                std::string stream_buffer;
                const size_t max_marker_length = 15; // Length of longest marker like "<|endoftext|>"

                for (int32_t step = 0; step < max_tokens_; ++step)
                {
                    // Fetch logits produced by prefill (step 0) or previous decode (step>0)
                    std::shared_ptr<TensorBase> latest_logits;
                    if (!pipeline_->logits(latest_logits) || !latest_logits)
                    {
                        LOG_ERROR("Failed to fetch logits prior to sampling at step " << step);
                        break;
                    }
                    std::vector<float> logits = fetchLastLogitsRow(latest_logits);

                    // Store logits for JSON export if requested
                    if (should_export)
                    {
                        export_data.logits.push_back(logits);
                    }

                    // Sample next token
                    int32_t next_token = sampleToken(logits);

                    // Debug: Print first few logits and the sampled token
                    std::ostringstream oss;
                    oss << "Logits[0-9]: ";
                    for (size_t i = 0; i < std::min(logits.size(), size_t(10)); ++i)
                    {
                        oss << logits[i] << " ";
                    }
                    oss << "... Sampled token: " << next_token;
                    LOG_DEBUG(oss.str());

                    if (next_token < 0)
                    {
                        LOG_WARN("Invalid token sampled, stopping generation");
                        break;
                    }

                    generated_tokens.push_back(next_token);
                    current_sequence.push_back(next_token);

                    // Store generated token for export
                    if (should_export)
                    {
                        export_data.generated_tokens.push_back(next_token);
                    }

                    // Prepare logits for next iteration (unless stopping this iteration)
                    if (!shouldStop(current_sequence, response_text))
                    {
                        if (!pipeline_->decode(next_token, weights_, stage_ctx_))
                        {
                            LOG_ERROR("Pipeline incremental decode failed after sampling token at step " << step);
                            break;
                        }
                    }

                    // Convert token to text if we have a tokenizer
                    std::string token_text;

                    if (tokenizer_)
                    {
                        // Check if this is a structural special token (EOS/BOS)
                        int32_t eos_id = tokenizer_->getSpecialToken("eos");
                        int32_t bos_id = tokenizer_->getSpecialToken("bos");

                        if (next_token == eos_id || next_token == bos_id)
                        {
                            // These are actual EOS/BOS - stop generation
                            LOG_DEBUG("Hit EOS/BOS token, stopping generation");
                            break;
                        }

                        // Detokenize all generated tokens together for proper BPE handling
                        // This ensures multi-byte UTF-8 sequences are correctly decoded
                        response_text = tokenizer_->detokenize(generated_tokens);

                        // Calculate new token text by comparing with previous response
                        static std::string previous_response;
                        if (step == 0)
                        {
                            previous_response.clear();
                        }
                        token_text = response_text.substr(previous_response.length());
                        previous_response = response_text;
                        stream_buffer += token_text;

                        // Check if response contains chat template end markers
                        // These are generated as individual tokens: < | im _end | >
                        size_t marker_pos = response_text.find("<|im_end|>");
                        if (marker_pos == std::string::npos)
                        {
                            marker_pos = response_text.find("<|endoftext|>");
                        }
                        if (marker_pos == std::string::npos)
                        {
                            marker_pos = response_text.find("<|im_start|>");
                        }

                        if (marker_pos != std::string::npos)
                        {
                            // Found a chat template marker - truncate response and stop
                            response_text = response_text.substr(0, marker_pos);
                            // Truncate stream buffer too
                            if (marker_pos < stream_buffer.length())
                            {
                                stream_buffer = stream_buffer.substr(0, marker_pos);
                            }
                            else
                            {
                                // Marker was in previously streamed content - clear buffer
                                stream_buffer.clear();
                            }
                            LOG_DEBUG("Found chat template marker at position " << marker_pos << ", stopping generation");
                            // Stream remaining buffer before stopping
                            if (callback && !stream_buffer.empty())
                            {
                                callback(stream_buffer, false);
                                stream_buffer.clear(); // Clear after streaming to prevent double-flush
                            }
                            break;
                        }

                        // Stream buffer when it's large enough that we can safely output some
                        // Keep a tail to detect partial markers
                        if (stream_buffer.length() > max_marker_length)
                        {
                            size_t safe_length = stream_buffer.length() - max_marker_length;
                            std::string safe_part = stream_buffer.substr(0, safe_length);

                            if (callback && !safe_part.empty())
                            {
                                callback(safe_part, false);
                            }

                            // Keep the tail in buffer
                            stream_buffer = stream_buffer.substr(safe_length);
                        }
                    }
                    else
                    {
                        token_text = "[" + std::to_string(next_token) + "]";
                        response_text += token_text;
                        // No tokenizer means no special tokens to filter
                        if (callback)
                        {
                            callback(token_text, false);
                        }
                    }

                    // Check stop conditions
                    if (shouldStop(generated_tokens, response_text))
                    {
                        LOG_DEBUG("Stop condition met at step " << step);
                        break;
                    }

                    LOG_TRACE("Generated token " << step << ": " << next_token << " -> \"" << token_text << "\"");
                }

                // Flush any remaining buffered output
                if (callback && !stream_buffer.empty())
                {
                    callback(stream_buffer, false);
                }

                // Call final callback
                if (callback)
                {
                    callback("", true);
                }

                LOG_INFO("Response generation completed. Generated " << generated_tokens.size() << " tokens");

                // Export to JSON if requested
                if (should_export)
                {
                    if (exportToJson(params_.output_json_file, export_data))
                    {
                        LOG_INFO("Generation data exported to " << params_.output_json_file);
                    }
                    else
                    {
                        LOG_ERROR("Failed to export generation data to " << params_.output_json_file);
                    }
                }

                return response_text;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Response generation failed: " << e.what());
                return "";
            }
        }

        std::string ResponseGenerator::generateStreamingResponse(const std::vector<int32_t> &prompt_tokens,
                                                                 ResponseCallback callback)
        {
            // This is essentially the same as generateResponse but ensures callback is called
            if (!callback)
            {
                LOG_WARN("No callback provided for streaming response, using regular generation");
                return generateResponse(prompt_tokens);
            }

            return generateResponse(prompt_tokens, callback);
        }

        void ResponseGenerator::setGenerationParams(float temperature, int32_t top_k,
                                                    float top_p, int32_t max_tokens)
        {
            temperature_ = std::max(0.0f, temperature);
            top_k_ = std::max(1, top_k);
            top_p_ = std::max(0.0f, std::min(1.0f, top_p));
            max_tokens_ = std::max(1, max_tokens);

            LOG_DEBUG("Updated generation parameters - temp: " << temperature_
                                                               << ", top_k: " << top_k_ << ", top_p: " << top_p_
                                                               << ", max_tokens: " << max_tokens_);
        }

        void ResponseGenerator::setStopConditions(const std::vector<int32_t> &stop_tokens,
                                                  const std::vector<std::string> &stop_sequences)
        {
            stop_tokens_ = stop_tokens;
            stop_sequences_ = stop_sequences;

            LOG_DEBUG("Updated stop conditions - " << stop_tokens_.size() << " tokens, "
                                                   << stop_sequences_.size() << " sequences");
        }

        bool ResponseGenerator::shouldStop(const std::vector<int32_t> &tokens,
                                           const std::string &text) const
        {
            // Check stop tokens
            if (!tokens.empty())
            {
                int32_t last_token = tokens.back();
                if (std::find(stop_tokens_.begin(), stop_tokens_.end(), last_token) != stop_tokens_.end())
                {
                    return true;
                }
            }

            // Check stop sequences
            for (const auto &stop_seq : stop_sequences_)
            {
                if (text.find(stop_seq) != std::string::npos)
                {
                    return true;
                }
            }

            return false;
        }

        int32_t ResponseGenerator::sampleToken(const std::vector<float> &logits)
        {
            if (logits.empty())
            {
                LOG_ERROR("Cannot sample from empty logits");
                return -1;
            }

            std::vector<float> processed_logits = logits;

            // Debug: Show initial logit range
            auto min_max = std::minmax_element(processed_logits.begin(), processed_logits.end());
            LOG_DEBUG("Initial logits range: [" << *min_max.first << ", " << *min_max.second << "]");

            // Apply temperature scaling
            if (temperature_ != 1.0f)
            {
                applyTemperature(processed_logits, temperature_);
                LOG_DEBUG("Applied temperature " << temperature_);
            }

            // Apply top-k filtering
            if (top_k_ > 0 && top_k_ < static_cast<int32_t>(processed_logits.size()))
            {
                applyTopK(processed_logits, top_k_);
                LOG_DEBUG("Applied top-k " << top_k_);
            }

            // Apply top-p filtering
            if (top_p_ < 1.0f)
            {
                applyTopP(processed_logits, top_p_);
                LOG_DEBUG("Applied top-p " << top_p_);
            }

            // Convert to probabilities and sample
            std::vector<float> probs = softmax(processed_logits);

            // Debug: Show top probabilities
            std::vector<std::pair<float, int32_t>> top_probs;
            for (size_t i = 0; i < std::min(probs.size(), size_t(5)); ++i)
            {
                top_probs.emplace_back(probs[i], static_cast<int32_t>(i));
            }
            std::sort(top_probs.begin(), top_probs.end(), std::greater<>());

            std::ostringstream oss;
            oss << "Top probabilities: ";
            for (const auto &p : top_probs)
            {
                oss << "tok" << p.second << "=" << p.first << " ";
            }
            LOG_DEBUG(oss.str());

            return sampleFromProbs(probs);
        }

        void ResponseGenerator::applyTemperature(std::vector<float> &logits, float temperature)
        {
            if (temperature <= 0.0f)
            {
                // Set to deterministic (argmax)
                auto max_it = std::max_element(logits.begin(), logits.end());
                std::fill(logits.begin(), logits.end(), -std::numeric_limits<float>::infinity());
                *max_it = 0.0f;
            }
            else
            {
                for (auto &logit : logits)
                {
                    logit /= temperature;
                }
            }
        }

        void ResponseGenerator::applyTopK(std::vector<float> &logits, int32_t k)
        {
            if (k <= 0 || k >= static_cast<int32_t>(logits.size()))
            {
                return;
            }

            // Find the k-th largest element
            std::vector<std::pair<float, size_t>> indexed_logits;
            for (size_t i = 0; i < logits.size(); ++i)
            {
                indexed_logits.emplace_back(logits[i], i);
            }

            std::nth_element(indexed_logits.begin(),
                             indexed_logits.begin() + k,
                             indexed_logits.end(),
                             std::greater<std::pair<float, size_t>>());

            float threshold = indexed_logits[k].first;

            // Zero out logits below threshold
            for (size_t i = 0; i < logits.size(); ++i)
            {
                if (logits[i] < threshold)
                {
                    logits[i] = -std::numeric_limits<float>::infinity();
                }
            }
        }

        void ResponseGenerator::applyTopP(std::vector<float> &logits, float p)
        {
            if (p >= 1.0f)
            {
                return;
            }

            // Convert to probabilities for top-p calculation
            std::vector<float> probs = softmax(logits);

            // Sort indices by probability
            std::vector<size_t> indices(probs.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                      [&probs](size_t a, size_t b)
                      { return probs[a] > probs[b]; });

            // Find cumulative probability cutoff
            float cumulative_prob = 0.0f;
            size_t cutoff_idx = indices.size();

            for (size_t i = 0; i < indices.size(); ++i)
            {
                cumulative_prob += probs[indices[i]];
                if (cumulative_prob >= p)
                {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            // Zero out logits beyond cutoff
            for (size_t i = cutoff_idx; i < indices.size(); ++i)
            {
                logits[indices[i]] = -std::numeric_limits<float>::infinity();
            }
        }

        bool ResponseGenerator::ensureWeights()
        {
            if (have_weights_)
                return true;
            try
            {
                // Use pipeline's loadWeights method instead of deprecated free function
                auto loaded_weights = pipeline_->loadWeights(params_.model_file);
                auto *qwen_weights = dynamic_cast<QwenModelWeights *>(loaded_weights.get());
                if (!qwen_weights)
                {
                    LOG_ERROR("ensureWeights: loaded weights are not QwenModelWeights");
                    return false;
                }
                weights_.inner = std::move(qwen_weights->inner);
                have_weights_ = true;
                LOG_INFO("Weights loaded lazily in ensureWeights()");
                return true;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("ensureWeights failed: " << e.what());
                return false;
            }
        }

        std::vector<float> ResponseGenerator::fetchLastLogitsRow(const std::shared_ptr<TensorBase> &logits_tensor) const
        {
            std::vector<float> row;
            if (!logits_tensor)
                return row;
            const auto &sh = logits_tensor->shape();
            if (sh.size() != 2)
                return row;
            int rows = sh[0];
            int cols = sh[1];
            row.resize(cols);
            const float *base = logits_tensor->data();
            std::memcpy(row.data(), base + (rows - 1) * cols, sizeof(float) * cols);
            return row;
        }

        std::vector<float> ResponseGenerator::softmax(const std::vector<float> &logits)
        {
            if (logits.empty())
            {
                return {};
            }

            // Find max for numerical stability
            float max_logit = *std::max_element(logits.begin(), logits.end());

            // Compute exponentials
            std::vector<float> probs(logits.size());
            float sum = 0.0f;

            for (size_t i = 0; i < logits.size(); ++i)
            {
                probs[i] = std::exp(logits[i] - max_logit);
                sum += probs[i];
            }

            // Normalize
            if (sum > 0.0f)
            {
                for (auto &prob : probs)
                {
                    prob /= sum;
                }
            }

            return probs;
        }

        int32_t ResponseGenerator::sampleFromProbs(const std::vector<float> &probs)
        {
            if (probs.empty())
            {
                return -1;
            }

            // Use random generator with current time seed
            static thread_local std::mt19937 gen(
                std::chrono::steady_clock::now().time_since_epoch().count());
            std::uniform_real_distribution<float> dis(0.0f, 1.0f);

            float random_value = dis(gen);
            float cumulative = 0.0f;

            LOG_DEBUG("Random value: " << random_value << ", sampling...");

            for (size_t i = 0; i < probs.size(); ++i)
            {
                cumulative += probs[i];
                if (random_value <= cumulative)
                {
                    LOG_DEBUG("Selected token " << i << " at cumulative " << cumulative);
                    return static_cast<int32_t>(i);
                }
            }

            // Fallback to last token
            std::cout << "Fallback to last token " << (probs.size() - 1) << std::endl;
            return static_cast<int32_t>(probs.size() - 1);
        }

    } // namespace chat
} // namespace llaminar