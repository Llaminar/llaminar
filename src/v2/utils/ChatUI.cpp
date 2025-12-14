/**
 * @file ChatUI.cpp
 * @brief FTXUI-based interactive chat interface implementation
 * @author David Sanftenberg
 * @date 2025
 */

#include "ChatUI.h"
#include "Logger.h"

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

#include <chrono>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <climits>

namespace llaminar2
{

    using namespace ftxui;

    // ========================================================================
    // Construction / Destruction
    // ========================================================================

    ChatUI::ChatUI(std::shared_ptr<ITokenizer> tokenizer,
                   std::shared_ptr<PipelineBase> pipeline,
                   const ChatUIConfig &config)
        : tokenizer_(std::move(tokenizer)), pipeline_(std::move(pipeline)), config_(config), screen_(ScreenInteractive::Fullscreen())
    {
        // Add system prompt to conversation if provided
        if (!config_.system_prompt.empty())
        {
            conversation_.push_back(ChatMessage("system", config_.system_prompt));
        }
    }

    ChatUI::~ChatUI()
    {
        stopGeneration();
    }

    // ========================================================================
    // Public Interface
    // ========================================================================

    int ChatUI::run()
    {
        if (!hasValidTemplate())
        {
            LOG_ERROR("ChatUI: Model does not have a valid chat template. "
                      "Use --chat-template to specify one, or use a chat-finetuned model.");
            return 1;
        }

        main_component_ = createLayout();
        screen_.Loop(main_component_);
        return 0;
    }

    bool ChatUI::hasValidTemplate() const
    {
        return tokenizer_ && tokenizer_->hasChatTemplate();
    }

    // ========================================================================
    // UI Component Creation
    // ========================================================================

    Component ChatUI::createInputComponent()
    {
        auto input_option = InputOption::Default();
        input_option.multiline = false;
        input_option.on_enter = [this]
        { onEnterPressed(); };

        return Input(&current_input_, "Type your message...", input_option);
    }

    Component ChatUI::createConversationView()
    {
        // Renderer that displays the conversation
        return Renderer([this]
                        { return renderConversation(); });
    }

    Component ChatUI::createLayout()
    {
        auto input_component = createInputComponent();

        // Main layout with conversation and input
        auto layout = Container::Vertical({
            Renderer([this]
                     { return renderSystemPrompt(); }),
            Renderer([this]
                     { return renderConversation(); }),
            input_component,
            Renderer([this]
                     { return renderStatusBar(); }),
        });

        // Wrap with event handler for special keys
        return CatchEvent(layout, [this, input_component](Event event)
                          {
            // Handle global events
            if (event == Event::Escape || 
                (event.is_character() && event.character() == "q" && !is_generating_))
            {
                screen_.ExitLoopClosure()();
                return true;
            }

            // Ctrl+C to stop generation or exit
            if (event == Event::Special("\x03"))  // Ctrl+C
            {
                if (is_generating_)
                {
                    stopGeneration();
                }
                else
                {
                    screen_.ExitLoopClosure()();
                }
                return true;
            }

            // Arrow keys for history
            if (event == Event::ArrowUp)
            {
                onUpArrow();
                return true;
            }
            if (event == Event::ArrowDown)
            {
                onDownArrow();
                return true;
            }

            return false; });
    }

    // ========================================================================
    // UI Rendering
    // ========================================================================

    Element ChatUI::renderSystemPrompt()
    {
        if (!config_.show_system || config_.system_prompt.empty())
        {
            return emptyElement();
        }

        return vbox({
            text("System") | bold | color(Color::Blue),
            text(config_.system_prompt) | dim,
            separator(),
        });
    }

    Element ChatUI::renderConversation()
    {
        Elements messages;

        // Render each message in conversation (skip system, shown separately)
        for (const auto &msg : conversation_)
        {
            if (msg.role == "system")
                continue;

            Element role_elem;
            Element content_elem;

            if (msg.role == "user")
            {
                role_elem = text("You") | bold | color(Color::Green);
                content_elem = paragraph(msg.content);
            }
            else if (msg.role == "assistant")
            {
                role_elem = text("Assistant") | bold | color(Color::Cyan);
                content_elem = paragraph(msg.content);
            }
            else
            {
                role_elem = text(msg.role) | bold;
                content_elem = paragraph(msg.content);
            }

            messages.push_back(vbox({
                role_elem,
                content_elem,
                text(""), // Spacing
            }));
        }

        // Render current streaming response
        if (is_generating_ || !current_response_.empty())
        {
            messages.push_back(renderCurrentResponse());
        }

        // Empty state
        if (messages.empty())
        {
            messages.push_back(
                text("Start a conversation by typing a message below.") | dim | center);
        }

        return vbox(messages) | flex | border | vscroll_indicator | yframe;
    }

    Element ChatUI::renderCurrentResponse()
    {
        std::string response;
        {
            std::lock_guard<std::mutex> lock(response_mutex_);
            response = current_response_;
        }

        Element status;
        if (is_generating_)
        {
            status = text(" |") | blink; // Cursor indicator
        }
        else
        {
            status = emptyElement();
        }

        return vbox({
            text("Assistant") | bold | color(Color::Cyan),
            hbox({
                paragraph(response),
                status,
            }),
        });
    }

    Element ChatUI::renderInput()
    {
        // Input is handled by the Input component
        return emptyElement();
    }

    Element ChatUI::renderStatusBar()
    {
        std::string status_text;

        if (is_generating_)
        {
            status_text = "Generating... (Ctrl+C to stop)";
        }
        else if (last_token_count_ > 0 && config_.show_stats)
        {
            double tokens_per_sec = (last_elapsed_ms_ > 0)
                                        ? (last_token_count_ * 1000.0 / last_elapsed_ms_)
                                        : 0.0;
            std::ostringstream oss;
            oss << last_token_count_ << " tokens, "
                << std::fixed << std::setprecision(1) << tokens_per_sec << " tok/s";
            status_text = oss.str();
        }
        else
        {
            status_text = "Enter to send | Ctrl+C to exit | Up/Down for history";
        }

        return hbox({
                   text(status_text) | dim,
                   filler(),
                   text("Llaminar Chat") | bold | color(Color::Blue),
               }) |
               border;
    }

    // ========================================================================
    // Event Handlers
    // ========================================================================

    bool ChatUI::handleInput(Event event)
    {
        // Handled by createLayout's CatchEvent
        return false;
    }

    void ChatUI::onEnterPressed()
    {
        if (is_generating_)
            return;
        if (current_input_.empty())
            return;

        // Handle special commands
        if (current_input_ == "/exit" || current_input_ == "/quit")
        {
            screen_.ExitLoopClosure()();
            return;
        }

        if (current_input_ == "/clear")
        {
            conversation_.clear();
            if (!config_.system_prompt.empty())
            {
                conversation_.push_back(ChatMessage("system", config_.system_prompt));
            }
            current_input_.clear();
            last_token_count_ = 0;
            last_elapsed_ms_ = 0.0;
            return;
        }

        // Save to history
        input_history_.insert(input_history_.begin(), current_input_);
        if (input_history_.size() > 100) // Limit history
        {
            input_history_.pop_back();
        }
        history_index_ = -1;

        // Add user message to conversation
        addToHistory("user", current_input_);

        // Clear input and start generation
        current_input_.clear();
        startGeneration();
    }

    void ChatUI::onUpArrow()
    {
        if (input_history_.empty())
            return;

        if (history_index_ < static_cast<int>(input_history_.size()) - 1)
        {
            history_index_++;
            current_input_ = input_history_[history_index_];
        }
    }

    void ChatUI::onDownArrow()
    {
        if (history_index_ > 0)
        {
            history_index_--;
            current_input_ = input_history_[history_index_];
        }
        else if (history_index_ == 0)
        {
            history_index_ = -1;
            current_input_.clear();
        }
    }

    // ========================================================================
    // Generation
    // ========================================================================

    void ChatUI::startGeneration()
    {
        if (is_generating_)
            return;

        // Clear current response
        {
            std::lock_guard<std::mutex> lock(response_mutex_);
            current_response_.clear();
        }

        is_generating_ = true;
        stop_requested_ = false;

        // Start generation in background thread
        generation_thread_ = std::thread(&ChatUI::generationThread, this);
    }

    void ChatUI::stopGeneration()
    {
        if (!is_generating_)
            return;

        stop_requested_ = true;

        if (generation_thread_.joinable())
        {
            generation_thread_.join();
        }

        is_generating_ = false;
    }

    void ChatUI::generationThread()
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        int token_count = 0;

        try
        {
            // Encode conversation with chat template
            auto token_ids = tokenizer_->encodeChat(conversation_, true);

            if (token_ids.empty())
            {
                LOG_ERROR("ChatUI: Failed to encode conversation");
                is_generating_ = false;
                return;
            }

            // Run prefill forward pass
            if (!pipeline_->forward(token_ids.data(), static_cast<int>(token_ids.size())))
            {
                LOG_ERROR("ChatUI: Prefill failed");
                is_generating_ = false;
                return;
            }

            // Get EOS token for stopping
            int eos_token_id = tokenizer_->eos_token();

            // Create sampler with configured parameters
            Sampler sampler(0); // Random seed
            SamplingParams sampling_params;
            sampling_params.temperature = config_.temperature;
            sampling_params.top_k = config_.top_k;
            sampling_params.top_p = config_.top_p;

            // Decode loop
            // max_tokens = -1 means unlimited (generate until EOS)
            int max_decode = (config_.max_tokens == -1) ? INT_MAX : config_.max_tokens;
            for (int i = 0; i < max_decode && !stop_requested_; ++i)
            {
                // Get logits from last forward pass
                const float *logits_ptr = pipeline_->logits();
                if (!logits_ptr)
                {
                    LOG_ERROR("ChatUI: Failed to get logits");
                    break;
                }

                // Convert to vector for sampling
                size_t vocab_size = tokenizer_->vocab_size();
                std::vector<float> logits_vec(logits_ptr, logits_ptr + vocab_size);

                // Sample next token
                int next_token;
                if (config_.temperature < 0.01f)
                {
                    next_token = sampler.sample_greedy(logits_vec);
                }
                else
                {
                    next_token = sampler.sample(logits_vec, sampling_params);
                }
                token_count++;

                // Check for EOS
                if (next_token == eos_token_id)
                {
                    break;
                }

                // Decode token to text
                std::string token_text = tokenizer_->decode_token(next_token);
                onTokenGenerated(token_text);

                // Forward next token for decode step
                if (!pipeline_->forward(&next_token, 1))
                {
                    LOG_ERROR("ChatUI: Decode forward failed at token " << i);
                    break;
                }

                // Trigger screen refresh
                screen_.PostEvent(Event::Custom);
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("ChatUI: Generation error: " << e.what());
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        onGenerationComplete(token_count, elapsed_ms);
    }

    void ChatUI::onTokenGenerated(const std::string &token)
    {
        std::lock_guard<std::mutex> lock(response_mutex_);
        current_response_ += token;
    }

    void ChatUI::onGenerationComplete(int tokens, double elapsed_ms)
    {
        // Save response to conversation
        std::string final_response;
        {
            std::lock_guard<std::mutex> lock(response_mutex_);
            final_response = current_response_;
            current_response_.clear();
        }

        if (!final_response.empty())
        {
            addToHistory("assistant", final_response);
        }

        last_token_count_ = tokens;
        last_elapsed_ms_ = elapsed_ms;
        is_generating_ = false;

        // Trigger final screen refresh
        screen_.PostEvent(Event::Custom);
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    std::string ChatUI::formatMessage(const ChatMessage &msg) const
    {
        if (msg.role == "user")
        {
            return "You: " + msg.content;
        }
        else if (msg.role == "assistant")
        {
            return "Assistant: " + msg.content;
        }
        else
        {
            return msg.role + ": " + msg.content;
        }
    }

    void ChatUI::addToHistory(const std::string &role, const std::string &content)
    {
        conversation_.push_back(ChatMessage(role, content));
    }

    void ChatUI::scrollToBottom()
    {
        // FTXUI handles this automatically with yframe
    }

    // ========================================================================
    // Single-Shot Chat
    // ========================================================================

    std::string runSingleShotChat(
        std::shared_ptr<ITokenizer> tokenizer,
        std::shared_ptr<PipelineBase> pipeline,
        const std::string &prompt,
        const std::string &system_prompt,
        const ChatUIConfig &config)
    {
        if (!tokenizer || !tokenizer->hasChatTemplate())
        {
            LOG_ERROR("runSingleShotChat: Tokenizer has no chat template");
            return "";
        }

        // Build conversation
        std::vector<ChatMessage> conversation;
        if (!system_prompt.empty())
        {
            conversation.push_back(ChatMessage("system", system_prompt));
        }
        conversation.push_back(ChatMessage("user", prompt));

        // Encode with chat template
        auto token_ids = tokenizer->encodeChat(conversation, true);
        if (token_ids.empty())
        {
            LOG_ERROR("runSingleShotChat: Failed to encode conversation");
            return "";
        }

        LOG_DEBUG("runSingleShotChat: Encoded " << token_ids.size() << " tokens");

        // Run prefill
        if (!pipeline->forward(token_ids.data(), static_cast<int>(token_ids.size())))
        {
            LOG_ERROR("runSingleShotChat: Prefill failed");
            return "";
        }

        // Get EOS token
        int eos_token_id = tokenizer->eos_token();

        // Create sampler
        Sampler sampler(0);
        SamplingParams sampling_params;
        sampling_params.temperature = config.temperature;
        sampling_params.top_k = config.top_k;
        sampling_params.top_p = config.top_p;

        // Decode
        std::string response;

        // max_tokens = -1 means unlimited (generate until EOS)
        int max_decode = (config.max_tokens == -1) ? INT_MAX : config.max_tokens;
        for (int i = 0; i < max_decode; ++i)
        {
            const float *logits_ptr = pipeline->logits();
            if (!logits_ptr)
            {
                break;
            }

            size_t vocab_size = tokenizer->vocab_size();
            std::vector<float> logits_vec(logits_ptr, logits_ptr + vocab_size);

            int next_token;
            if (config.temperature < 0.01f)
            {
                next_token = sampler.sample_greedy(logits_vec);
            }
            else
            {
                next_token = sampler.sample(logits_vec, sampling_params);
            }

            if (next_token == eos_token_id)
            {
                break;
            }

            std::string token_text = tokenizer->decode_token(next_token);
            response += token_text;

            // Print streaming output
            std::cout << token_text << std::flush;

            // Forward next token
            if (!pipeline->forward(&next_token, 1))
            {
                break;
            }
        }

        std::cout << std::endl;
        return response;
    }

} // namespace llaminar2
