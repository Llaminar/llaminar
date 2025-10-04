#include "chat_interface.h"
#include "../logger.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <signal.h>
#include <thread>
#include <chrono>

namespace llaminar
{
    namespace chat
    {
        // Initialize static members
        ChatInterface *ChatInterface::g_chat_interface = nullptr;
        volatile int ChatInterface::signal_count = 0;

        // Signal handler for graceful shutdown
        void ChatInterface::signalHandler(int signal)
        {
            signal_count++;

            if (signal_count == 1)
            {
                std::cout << "\nReceived signal " << signal << ". Press Ctrl+C again to force exit..." << std::endl;
                if (g_chat_interface)
                {
                    g_chat_interface->should_stop_ = true;
                }
            }
            else if (signal_count >= 2)
            {
                std::cout << "\nForce exit requested. Goodbye!" << std::endl;
                std::exit(0);
            }
        }

        ChatInterface::ChatInterface(std::unique_ptr<ChatSession> session,
                                     std::shared_ptr<AbstractPipeline> pipeline,
                                     std::unique_ptr<ResponseGenerator> generator,
                                     const LlaminarParams &params)
            : session_(std::move(session)), pipeline_(pipeline), generator_(std::move(generator)), params_(params), should_stop_(false), is_mpi_rank_zero_(true) // Assume rank 0 for now, can be set properly
        {
            // Set global reference for signal handler
            g_chat_interface = this;

            // Set up signal handlers
            setupSignalHandlers();

            LOG_INFO("ChatInterface initialized");
        }

        ChatInterface::~ChatInterface()
        {
            g_chat_interface = nullptr;
            LOG_DEBUG("ChatInterface destroyed");
        }

        void ChatInterface::run()
        {
            if (!is_mpi_rank_zero_)
            {
                LOG_DEBUG("Non-root MPI rank, waiting for inference requests...");
                // Non-root ranks wait for work
                while (!should_stop_)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                return;
            }

            LOG_INFO("Starting interactive chat interface...");
            showWelcome();

            std::string input;
            while (!should_stop_)
            {
                // Reset signal count when starting new input
                signal_count = 0;

                // Display prompt
                std::cout << "\n> ";
                std::cout.flush();

                // Read user input
                if (!std::getline(std::cin, input))
                {
                    // EOF or input error (could be from signal interruption)
                    if (should_stop_)
                    {
                        LOG_DEBUG("Chat stopped by signal, exiting");
                        break;
                    }

                    if (std::cin.eof())
                    {
                        LOG_DEBUG("Input stream closed, exiting chat");
                        break;
                    }

                    // Clear error state and continue
                    std::cin.clear();
                    continue;
                }

                // Process input
                if (!processInput(input))
                {
                    break; // User requested exit
                }
            }

            std::cout << "\nGoodbye!\n"
                      << std::endl;
            LOG_INFO("Chat interface session ended");
        }

        bool ChatInterface::processInput(const std::string &input)
        {
            // Trim whitespace
            std::string trimmed = input;
            trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));
            trimmed.erase(trimmed.find_last_not_of(" \t\n\r") + 1);

            if (trimmed.empty())
            {
                return true; // Continue on empty input
            }

            // Check for commands
            if (trimmed[0] == '/')
            {
                return processCommand(trimmed);
            }

            // Regular chat message
            handleUserMessage(trimmed);
            return true; // Continue chat
        }

        void ChatInterface::stop()
        {
            should_stop_ = true;
            LOG_DEBUG("Chat interface stop requested");
        }

        void ChatInterface::showWelcome()
        {
            std::cout << "\n"
                      << std::string(50, '=') << std::endl;
            std::cout << "    Llaminar Interactive Chat Interface" << std::endl;
            std::cout << std::string(50, '=') << std::endl;

            if (session_)
            {
                std::cout << "Session: " << session_->getSessionStats() << std::endl;
            }

            std::cout << "Type '/help' for commands or start chatting..." << std::endl;
            std::cout << "Press Ctrl+C or type '/quit' to exit." << std::endl;
        }

        void ChatInterface::showHelp()
        {
            std::cout << "\nAvailable Commands:" << std::endl;
            std::cout << "  /help         - Show this help message" << std::endl;
            std::cout << "  /quit         - Exit the chat interface" << std::endl;
            std::cout << "  /reset        - Clear conversation history" << std::endl;
            std::cout << "  /stats        - Show session statistics" << std::endl;
            std::cout << "  /history      - Display conversation history" << std::endl;
            std::cout << "  /system <msg> - Set system prompt" << std::endl;
            std::cout << "  /temp <value> - Set temperature (0.0-2.0)" << std::endl;
            std::cout << "  /max <tokens> - Set max response tokens" << std::endl;
            std::cout << "\nJust type your message to start chatting!" << std::endl;
        }

        bool ChatInterface::processCommand(const std::string &command)
        {
            std::istringstream iss(command.substr(1)); // Remove '/'
            std::string cmd;
            iss >> cmd;

            if (cmd == "help" || cmd == "h")
            {
                showHelp();
            }
            else if (cmd == "quit" || cmd == "exit" || cmd == "q")
            {
                return false; // Exit chat
            }
            else if (cmd == "reset" || cmd == "clear")
            {
                session_->clearHistory();
                std::cout << "Conversation history cleared." << std::endl;
            }
            else if (cmd == "stats")
            {
                std::cout << "Session Statistics: " << session_->getSessionStats() << std::endl;

                // Show context warning if needed
                if (session_->isContextNearLimit())
                {
                    std::cout << "⚠️  Context is near limit - consider using /reset" << std::endl;
                }
            }
            else if (cmd == "history")
            {
                showHistory();
            }
            else if (cmd == "system")
            {
                std::string system_prompt;
                std::getline(iss, system_prompt);
                if (!system_prompt.empty())
                {
                    // Remove leading space
                    system_prompt = system_prompt.substr(1);
                    if (!system_prompt.empty())
                    {
                        session_->setSystemPrompt(system_prompt);
                        std::cout << "System prompt set to: " << system_prompt << std::endl;
                    }
                }
                else
                {
                    std::cout << "Usage: /system <prompt>" << std::endl;
                }
            }
            else if (cmd == "temp")
            {
                float temperature;
                if (iss >> temperature && temperature >= 0.0f && temperature <= 2.0f)
                {
                    generator_->setGenerationParams(temperature, -1, -1.0f, -1);
                    std::cout << "Temperature set to: " << temperature << std::endl;
                }
                else
                {
                    std::cout << "Usage: /temp <value> (0.0-2.0)" << std::endl;
                }
            }
            else if (cmd == "max")
            {
                int max_tokens;
                if (iss >> max_tokens && max_tokens > 0)
                {
                    generator_->setGenerationParams(-1.0f, -1, -1.0f, max_tokens);
                    std::cout << "Max response tokens set to: " << max_tokens << std::endl;
                }
                else
                {
                    std::cout << "Usage: /max <tokens> (positive integer)" << std::endl;
                }
            }
            else
            {
                std::cout << "Unknown command: " << cmd << std::endl;
                std::cout << "Type '/help' for available commands." << std::endl;
            }

            return true; // Continue chat
        }

        void ChatInterface::handleUserMessage(const std::string &message)
        {
            try
            {
                // Add user message to session
                session_->addMessage("user", message);

                // Prepare prompt for inference
                std::vector<int32_t> prompt_tokens = session_->preparePrompt();

                if (prompt_tokens.empty())
                {
                    std::cout << "⚠️  Unable to prepare prompt. Please try again." << std::endl;
                    return;
                }

                std::cout << "\n🤖 Assistant: ";
                std::cout.flush();

                // Generate streaming response
                std::string response = generator_->generateStreamingResponse(
                    prompt_tokens,
                    [](const std::string &token, bool is_complete)
                    {
                        std::cout << token;
                        std::cout.flush();
                    });

                std::cout << std::endl;

                if (!response.empty())
                {
                    // Add response to session
                    session_->addMessage("assistant", response);
                }
                else
                {
                    std::cout << "⚠️  No response generated. Please try again." << std::endl;
                }

                // Show context warning if needed
                if (session_->isContextNearLimit())
                {
                    std::cout << "\n⚠️  Context is getting full. Consider using '/reset' to clear history." << std::endl;
                }
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Error handling user message: " << e.what());
                std::cout << "⚠️  An error occurred while processing your message. Please try again." << std::endl;
            }
        }

        void ChatInterface::showHistory()
        {
            const auto &history = session_->getHistory();

            if (history.empty())
            {
                std::cout << "No conversation history." << std::endl;
                return;
            }

            std::cout << "\n"
                      << std::string(40, '-') << std::endl;
            std::cout << "Conversation History:" << std::endl;
            std::cout << std::string(40, '-') << std::endl;

            for (const auto &msg : history)
            {
                if (msg.role == "user")
                {
                    std::cout << "👤 You: " << msg.content << std::endl;
                }
                else if (msg.role == "assistant")
                {
                    std::cout << "🤖 Assistant: " << msg.content << std::endl;
                }
                else if (msg.role == "system")
                {
                    std::cout << "⚙️  System: " << msg.content << std::endl;
                }
            }

            std::cout << std::string(40, '-') << std::endl;
        }

        void ChatInterface::setupSignalHandlers()
        {
            signal(SIGINT, ChatInterface::signalHandler);
            signal(SIGTERM, ChatInterface::signalHandler);
        }

    } // namespace chat
} // namespace llaminar