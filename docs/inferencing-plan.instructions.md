# Llaminar Interactive CLI Chat Interface - Implementation Plan

*Created: September 20, 2025*

## Overview

This document outlines the comprehensive plan for implementing prompt processing with an interactive CLI chat interface for Llaminar. The implementation will provide turn-based conversational inference while maintaining Llaminar's modular architecture and MPI distribution capabilities.

## Current State Analysis

### Existing Infrastructure ✅
- Command-line argument parsing with prompt parameter support (`ArgumentParser`)
- GGUF model loading with vocabulary extraction (`ModelLoader`)
- Abstract inference pipeline adapter (`AbstractPipeline` + concrete Qwen adapter)  
    - NOTE: Legacy `MPITransformerPipeline` is deprecated and only retained for temporary parity tests.
- Logging system and error handling
- System topology detection and COSMA integration

### Missing Components ❌
- Text-to-token conversion (tokenization)
- Chat session management and conversation state
- Interactive command loop and user interface
- Response generation and text output formatting
- Chat templates and prompt formatting

## Architecture Design

### Component Overview

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   ChatInterface     │    │    ChatSession       │    │  TokenizerInterface │
│  (User Interaction) │───▶│  (State Management)  │───▶│  (Text Processing)  │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
           │                          │                           │
           │                          ▼                           │
           │              ┌──────────────────────┐                │
           │              │  AbstractPipeline(Qwen) │              │
           │              │  (Distributed Inference)│              │
           │              └──────────────────────┘                │
           │                          │                           │
           ▼                          ▼                           ▼
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│  ResponseGenerator  │    │     ModelLoader      │    │   ArgumentParser    │
│  (Output Formatting)│    │   (GGUF + Vocab)    │    │   (Configuration)   │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### Core Interfaces

#### TokenizerInterface
```cpp
class TokenizerInterface {
public:
    virtual ~TokenizerInterface() = default;
    virtual std::vector<int32_t> tokenize(const std::string& text) = 0;
    virtual std::string detokenize(const std::vector<int32_t>& tokens) = 0;
    virtual std::string applyTemplate(const std::vector<ChatMessage>& messages) = 0;
    virtual bool loadVocabulary(const ModelLoader& model) = 0;
    virtual int32_t getSpecialToken(const std::string& token_name) = 0;
};
```

#### ChatMessage Structure
```cpp
struct ChatMessage {
    std::string role;      // "user", "assistant", "system"
    std::string content;   // Message content
    int64_t timestamp;     // Unix timestamp
    std::vector<int32_t> tokens; // Cached tokenization
};
```

#### ChatSession Class
```cpp
class ChatSession {
private:
    std::vector<ChatMessage> history_;
    std::unique_ptr<TokenizerInterface> tokenizer_;
    LlaminarParams config_;
    size_t max_context_tokens_;
    
public:
    void addMessage(const std::string& role, const std::string& content);
    std::vector<int32_t> preparePrompt();
    void updateContext(const std::string& response);
    void clearHistory();
    std::vector<ChatMessage> getHistory() const;
    bool isContextFull() const;
};
```

#### ChatInterface Class
```cpp
class ChatInterface {
private:
    std::unique_ptr<ChatSession> session_;
    std::unique_ptr<MPITransformerPipeline> pipeline_;
    std::unique_ptr<ResponseGenerator> generator_;
    LlaminarParams& params_;
    
public:
    void run();
    void processCommand(const std::string& command);
    void handleUserInput(const std::string& input);
    void displayResponse(const std::string& response);
    void showHelp();
};
```

## Implementation Plan

### Phase 1: Core Architecture (Todo 1) 🔄
**Files to Create:**
- `src/chat/chat_message.h` - Message structure definitions
- `src/chat/tokenizer_interface.h` - Tokenization interface
- `src/chat/chat_session.h` - Session management interface
- `src/chat/chat_interface.h` - Main chat interface
- `src/chat/response_generator.h` - Response formatting interface

**Integration Points:**
- Extend `LlaminarParams` with chat-specific parameters
- Add chat mode flag to argument parser
- Create factory methods for component initialization

### Phase 2: Tokenizer Implementation (Todo 2)
**Files to Create:**
- `src/chat/gguf_tokenizer.h/cpp` - GGUF vocabulary tokenizer
- `src/chat/chat_template.h/cpp` - Chat template processor

**Key Features:**
- Text preprocessing and normalization
- Special token handling (BOS, EOS, system tokens)
- Integration with existing `ModelLoader` vocabulary
- Support for multiple chat template formats
- Efficient token caching

### Phase 3: Session Management (Todo 3)
**Files to Create:**
- `src/chat/chat_session.cpp` - Session implementation

**Key Features:**
- Conversation history management
- Context length tracking and truncation
- Message role validation
- Token count monitoring
- Memory-efficient storage

### Phase 4: Interactive Interface (Todo 4)
**Files to Create:**
- `src/chat/chat_interface.cpp` - Main interface implementation
- `src/chat/command_processor.h/cpp` - Command handling

**Commands to Support:**
- `/help` - Display available commands
- `/reset` - Clear conversation history
- `/quit` - Exit chat mode
- `/history` - Show conversation history
- `/model <path>` - Load different model
- `/temp <value>` - Set temperature
- `/tokens` - Show token count information

**Features:**
- Multi-line input support (end with empty line)
- Real-time response streaming
- Graceful error handling
- Signal handling (Ctrl+C)

### Phase 5: MPI Integration (Todo 5)
**Files to Modify:**
- `src/mpi_transformer_pipeline.h/cpp` - Add chat support
- `src/main.cpp` - Add interactive mode

**Key Features:**
- Coordinate tokenization across MPI ranks
- Distribute inference workload efficiently
- Synchronize response generation
- Handle MPI communication for chat state
- Rank 0 handles user interaction, others compute

### Phase 6: Response Generation (Todo 6)
**Files to Create:**
- `src/chat/response_generator.cpp` - Response implementation
- `src/chat/sampling.h/cpp` - Sampling strategies

**Key Features:**
- Token-to-text conversion using GGUF vocabulary
- Streaming output with configurable flush intervals
- Temperature, top-k, top-p sampling
- Stop token detection
- Response formatting and cleanup

### Phase 7: Configuration & Persistence (Todo 7)
**Files to Create:**
- `src/chat/chat_config.h/cpp` - Configuration management
- `src/chat/session_persistence.h/cpp` - Save/load functionality

**Key Features:**
- Chat parameter management
- Conversation persistence (JSON format)
- User preference storage
- Configuration file support
- Backward compatibility with existing args

### Phase 8: Testing & Error Handling (Todo 8)
**Files to Create:**
- `tests/chat/test_tokenizer.cpp` - Tokenization tests
- `tests/chat/test_chat_session.cpp` - Session tests
- `tests/chat/test_chat_interface.cpp` - Interface tests

**Key Features:**
- Unit tests for all components
- Integration tests with sample conversations
- Error handling for edge cases
- MPI communication error recovery
- Memory leak detection

## Usage Examples

### Basic Interactive Chat
```bash
# Start interactive chat mode
mpirun -np 2 ./build/llaminar --model models/qwen2.5-0.5b.gguf --interactive

Llaminar Interactive Chat
Model: qwen2.5-0.5b.gguf
Context Size: 2048 tokens
Type '/help' for commands or start chatting...

> Hello! How are you today?
Hello! I'm doing well, thank you for asking. I'm an AI assistant created by 
Llaminar, and I'm here to help you with any questions or tasks you might have. 
How can I assist you today?

> /temp 0.8
Temperature set to 0.8

> Can you explain quantum computing?
[Assistant provides detailed explanation...]

> /quit
Goodbye!
```

### Command Line Usage
```bash
# Single prompt mode
./build/llaminar --model model.gguf --prompt "Explain machine learning"

# Load with custom parameters
./build/llaminar --model model.gguf --interactive --ctx-size 4096 --temperature 0.7
```

## File Structure

```
src/chat/
├── chat_message.h              # Message structures
├── tokenizer_interface.h       # Tokenization interface
├── gguf_tokenizer.h           # GGUF tokenizer implementation
├── gguf_tokenizer.cpp
├── chat_template.h            # Chat template processor
├── chat_template.cpp
├── chat_session.h             # Session management
├── chat_session.cpp
├── chat_interface.h           # Main interface
├── chat_interface.cpp
├── command_processor.h        # Command handling
├── command_processor.cpp
├── response_generator.h       # Response formatting
├── response_generator.cpp
├── sampling.h                 # Sampling strategies
├── sampling.cpp
├── chat_config.h              # Configuration
├── chat_config.cpp
├── session_persistence.h      # Persistence
└── session_persistence.cpp

tests/chat/
├── test_tokenizer.cpp
├── test_chat_session.cpp
├── test_chat_interface.cpp
└── test_integration.cpp
```

## Integration with Existing Systems

### Argument Parser Extensions
```cpp
// Add to LlaminarParams
bool interactive_chat = false;          // --interactive, -i
std::string system_prompt = "";         // --system
bool save_conversation = false;         // --save-chat
std::string chat_template = "";         // --chat-template
int32_t max_response_tokens = 512;      // --max-response
```

### Main.cpp Integration
- Add chat mode detection
- Initialize chat components when `--interactive` is used
- Handle graceful shutdown for MPI processes
- Maintain existing benchmark and inference modes

### Logging Integration
- Use existing `Logger` for debug information
- Add chat-specific log categories
- Support verbose tokenization logging
- MPI-aware logging coordination

## Performance Considerations

### Memory Management
- Efficient token storage and caching
- Conversation history pruning strategies
- NUMA-aware memory allocation
- Minimize copy operations for large tensors

### MPI Coordination
- Minimize inter-process communication
- Efficient token distribution strategies
- Overlap computation with communication
- Graceful handling of process failures

### Responsiveness
- Streaming response generation
- Asynchronous input handling
- Efficient text processing
- Real-time token display options

## Future Extensions

### Planned Features
- Web interface integration
- Multi-modal support (images, audio)
- Plugin system for custom commands
- Advanced chat templates
- Conversation branching and editing

### API Integration
- RESTful API endpoints
- WebSocket streaming support
- OpenAI-compatible API
- Batch processing support

This plan provides a solid foundation for implementing comprehensive chat functionality while maintaining Llaminar's performance and architectural advantages.