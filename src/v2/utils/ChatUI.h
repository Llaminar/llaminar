/**
 * @file ChatUI.h
 * @brief FTXUI-based interactive chat interface for Llaminar V2
 * @author David Sanftenberg
 * @date 2025
 *
 * Provides a terminal-based chat UI using FTXUI library for interactive
 * multi-turn conversations with LLM models.
 */

#pragma once

#include "ChatTemplate.h"
#include "Tokenizer.h"
#include "Sampler.h"
#include "../execution/IInferenceRunner.h"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>

namespace llaminar2
{

    /**
     * @brief Configuration for ChatUI
     */
    struct ChatUIConfig
    {
        std::string system_prompt = ""; ///< Optional system message
        int max_tokens = -1;            ///< Max tokens per response (-1 = unlimited until EOS or context full)
        float temperature = 0.7f;       ///< Sampling temperature
        int top_k = 40;                 ///< Top-k sampling
        float top_p = 0.9f;             ///< Top-p (nucleus) sampling
        bool show_system = true;        ///< Show system prompt in UI
        bool show_stats = true;         ///< Show generation statistics
    };

    /**
     * @brief Interactive chat UI using FTXUI
     *
     * Provides a rich terminal interface for multi-turn conversations with
     * streaming token display, input history, and conversation scrolling.
     *
     * Features:
     * - Streaming token generation with real-time display
     * - Multi-turn conversation history
     * - System prompt support
     * - Input history navigation (up/down arrows)
     * - Scrollable conversation view
     * - Generation statistics (tokens/sec)
     * - Keyboard shortcuts (Ctrl+C to exit, Enter to send)
     *
     * Usage:
     * @code
     *   auto tokenizer = createTokenizer(model_ctx);
     *   auto pipeline = createPipeline(model_ctx);
     *
     *   ChatUIConfig config;
     *   config.system_prompt = "You are a helpful assistant.";
     *
     *   ChatUI chat(tokenizer, pipeline, config);
     *   chat.run();  // Blocks until user exits
     * @endcode
     */
    class ChatUI
    {
    public:
        /**
         * @brief Construct ChatUI
         *
         * @param tokenizer Tokenizer for encoding/decoding (must have chat template)
         * @param runner Inference runner for executing forward passes
         * @param config UI and generation configuration
         */
        ChatUI(std::shared_ptr<ITokenizer> tokenizer,
               std::shared_ptr<IInferenceRunner> runner,
               const ChatUIConfig &config = {});

        /**
         * @brief Destructor - ensures clean shutdown of generation thread
         */
        ~ChatUI();

        // Disable copy
        ChatUI(const ChatUI &) = delete;
        ChatUI &operator=(const ChatUI &) = delete;

        /**
         * @brief Run the interactive chat UI
         *
         * Blocks until user exits (Ctrl+C or /exit command).
         * Returns exit code (0 = normal exit, non-zero = error).
         */
        int run();

        /**
         * @brief Check if the tokenizer has a valid chat template
         * @return true if chat template is available
         */
        bool hasValidTemplate() const;

    private:
        // UI Components
        ftxui::Component createInputComponent();
        ftxui::Component createConversationView();
        ftxui::Component createLayout();

        // UI Rendering
        ftxui::Element renderSystemPrompt();
        ftxui::Element renderConversation();
        ftxui::Element renderCurrentResponse();
        ftxui::Element renderInput();
        ftxui::Element renderStatusBar();

        // Event handlers
        bool handleInput(ftxui::Event event);
        void onEnterPressed();
        void onUpArrow();
        void onDownArrow();

        // Generation
        void startGeneration();
        void stopGeneration();
        void generationThread();

        // Token callback (called from generation thread)
        void onTokenGenerated(const std::string &token);
        void onGenerationComplete(int tokens, double elapsed_ms);

        // Helpers
        std::string formatMessage(const ChatMessage &msg) const;
        void addToHistory(const std::string &role, const std::string &content);
        void scrollToBottom();

        // Dependencies
        std::shared_ptr<ITokenizer> tokenizer_;
        std::shared_ptr<IInferenceRunner> runner_;
        ChatUIConfig config_;

        // FTXUI
        ftxui::ScreenInteractive screen_;
        ftxui::Component main_component_;

        // Conversation state
        std::vector<ChatMessage> conversation_;
        std::string current_input_;
        std::string current_response_; // Accumulator for streaming
        std::vector<std::string> input_history_;
        int history_index_ = -1; // -1 = current input, 0+ = history

        // Generation state
        std::atomic<bool> is_generating_{false};
        std::atomic<bool> stop_requested_{false};
        std::thread generation_thread_;
        std::mutex response_mutex_;

        // Statistics
        int last_token_count_ = 0;
        double last_elapsed_ms_ = 0.0;

        // Scroll state
        int scroll_offset_ = 0;
    };

    /**
     * @brief Run a single-shot chat inference
     *
     * Applies chat template to a single user prompt and generates a response.
     * Used for --chat-single mode (non-interactive).
     *
     * @param tokenizer Tokenizer with chat template
     * @param runner Inference runner for executing forward passes
     * @param prompt User prompt
     * @param system_prompt Optional system prompt
     * @param config Generation configuration
     * @return Generated response text
     */
    std::string runSingleShotChat(
        std::shared_ptr<ITokenizer> tokenizer,
        std::shared_ptr<IInferenceRunner> runner,
        const std::string &prompt,
        const std::string &system_prompt = "",
        const ChatUIConfig &config = {});

} // namespace llaminar2
