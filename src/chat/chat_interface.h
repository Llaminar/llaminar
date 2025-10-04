#pragma once

#include "chat_session.h"
#include "response_generator.h"
#include "../argument_parser.h"
#include "../abstract_pipeline.h"
#include <memory>
#include <string>
#include <atomic>

namespace llaminar
{
    namespace chat
    {

        /**
         * Main interactive CLI chat interface
         * Handles user input, command processing, and response display
         */
        class ChatInterface
        {
        public:
            /**
             * Constructor
             * @param session Chat session manager
             * @param pipeline MPI transformer pipeline for inference
             * @param generator Response generator for output formatting
             * @param params Configuration parameters
             */
            ChatInterface(std::unique_ptr<ChatSession> session,
                          std::shared_ptr<AbstractPipeline> pipeline,
                          std::unique_ptr<ResponseGenerator> generator,
                          const LlaminarParams &params);

            /**
             * Destructor
             */
            ~ChatInterface();

            /**
             * Start interactive chat loop
             * Main entry point for chat functionality
             */
            void run();

            /**
             * Process a single user input or command
             * @param input User input string
             * @return True to continue chat, false to exit
             */
            bool processInput(const std::string &input);

            /**
             * Stop the chat interface gracefully
             */
            void stop();

        private:
            std::unique_ptr<ChatSession> session_;
            std::shared_ptr<AbstractPipeline> pipeline_;
            std::unique_ptr<ResponseGenerator> generator_;
            const LlaminarParams &params_;
            std::atomic<bool> should_stop_;
            bool is_mpi_rank_zero_;

            /**
             * Display welcome message and instructions
             */
            void showWelcome();

            /**
             * Display help information
             */
            void showHelp();

            /**
             * Process chat commands (starting with '/')
             * @param command Command string including the '/'
             * @return True if command was processed successfully
             */
            bool processCommand(const std::string &command);

            /**
             * Handle regular user message (not a command)
             * @param message User message content
             */
            void handleUserMessage(const std::string &message);

            /**
             * Generate and display assistant response
             * @param prompt_tokens Tokenized prompt for inference
             */
            void generateResponse(const std::vector<int32_t> &prompt_tokens);

            /**
             * Display streaming response text
             * @param text Text to display
             * @param is_complete True if this is the final text chunk
             */
            void displayResponse(const std::string &text, bool is_complete = false);

            /**
             * Get user input with prompt
             * @param prompt Prompt string to display
             * @return User input string
             */
            std::string getUserInput(const std::string &prompt = "> ");

            /**
             * Get multi-line user input (continue until empty line)
             * @return Combined multi-line input
             */
            std::string getMultiLineInput();

            /**
             * Setup signal handlers for graceful shutdown
             */
            void setupSignalHandlers();

            /**
             * Signal handler for SIGINT (Ctrl+C)
             */
            static void signalHandler(int signal);

            // Static pointer for signal handler access
            // Static pointer for signal handler access
            static ChatInterface *g_chat_interface;

            // Signal count for force exit
            static volatile int signal_count;
            /**
             * Display conversation history
             */
            void showHistory();
        };

    } // namespace chat
} // namespace llaminar