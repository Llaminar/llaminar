# Chat Mode Implementation Plan

**Status**: In Progress  
**Created**: 2025-11-28  
**Branch**: `feature/chat-mode`

## Overview

Implementing full chat template support and an interactive FTXUI-based terminal UI for Llaminar V2.

## Current State

- **No chat template support**: Prompts are passed directly to `tokenizer->encode()` without any formatting
- **Raw prompt encoding**: `Main.cpp:436` just encodes `args.prompt` directly with BOS token
- **GGUF metadata available**: Model files contain `tokenizer.chat_template` metadata
- **llama.cpp reference**: Has robust chat template support in `external/llama.cpp/src/llama-chat.{h,cpp}` with 50+ template formats

## Architecture Decision

**Native implementation** with inspiration from llama.cpp's approach.

**Rationale**:
- V2 architecture is designed to be self-contained (operator-free, tensor-centric)
- llama.cpp's chat template code has dependencies on their specific structs
- Native implementation allows better integration with our `ITokenizer` interface
- We can selectively support the templates needed for our target models (Qwen, LLaMA, Mistral)

**UI Framework**: FTXUI

**Rationale**:
- Header-only or single-library C++ - fits V2's modern C++ style
- No external dependencies (terminal escape codes only)
- Built-in components: text input, scrollable areas, decorators
- Functional reactive pattern - clean code
- MIT license, actively maintained
- Works great for chat UIs specifically
- Cross-platform: Linux, macOS, Windows terminal

---

## Implementation Phases

### Phase 1: Core Chat Template Infrastructure

#### 1.1 Create `ChatTemplate.h/cpp` (`src/v2/utils/`)

```cpp
struct ChatMessage {
    std::string role;      // "system", "user", "assistant"
    std::string content;
};

enum class ChatTemplateType {
    CHATML,        // Qwen, many others (<|im_start|>...<|im_end|>)
    LLAMA3,        // LLaMA 3 (<|start_header_id|>...<|eot_id|>)
    LLAMA2,        // LLaMA 2 ([INST]...[/INST])
    MISTRAL_V1,    // Mistral v1
    MISTRAL_V3,    // Mistral v3
    PHI3,          // Phi-3
    GEMMA,         // Gemma
    DEEPSEEK,      // DeepSeek
    UNKNOWN        // Fallback - use raw prompt
};

class ChatTemplate {
public:
    // Factory: auto-detect from GGUF metadata
    static std::unique_ptr<ChatTemplate> create(const std::string& template_str);
    
    // Apply template to messages
    std::string apply(const std::vector<ChatMessage>& messages, 
                      bool add_generation_prompt = true) const;
    
    // Get template type
    ChatTemplateType type() const;
    
private:
    ChatTemplateType type_;
    std::string raw_template_; // Original jinja-like template from GGUF
};
```

#### 1.2 Template Detection Logic
- Parse `tokenizer.chat_template` from GGUF metadata
- Use heuristic detection (like llama.cpp does) based on template contents
- Map to `ChatTemplateType` enum

#### 1.3 Template Application
- Implement format-specific string builders for each template type
- Handle `add_generation_prompt` (adds assistant header for model to complete)

---

### Phase 2: Tokenizer Integration

#### 2.1 Extend `ITokenizer` Interface (`src/v2/utils/Tokenizer.h`)

```cpp
class ITokenizer {
public:
    // Existing methods...
    
    // NEW: Chat template support
    virtual std::string getChatTemplate() const = 0;
    virtual std::vector<int> encodeChat(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt = true
    ) const = 0;
};
```

#### 2.2 Implement in `BPETokenizer`
- Read `tokenizer.chat_template` from GGUF metadata during initialization
- Store `ChatTemplate` instance
- Implement `encodeChat()` that applies template then encodes

---

### Phase 3: CLI Integration

#### 3.1 Extend `ArgContext` (`src/v2/utils/ArgParser.h`)

```cpp
struct ArgContext {
    // Existing...
    
    // NEW: Chat mode
    bool chat_mode = false;              // Enable interactive chat
    bool single_shot_chat = false;       // Single prompt with chat template
    std::string system_prompt = "";      // Optional system message
    std::string chat_template = "";      // Override template (optional)
};
```

#### 3.2 Add CLI Flags (`src/v2/utils/ArgParser.cpp`)
- `--chat` / `-c`: Enable interactive chat mode (FTXUI)
- `--chat-single`: Single prompt with chat template formatting
- `--system` / `-s`: Set system prompt
- `--chat-template`: Override template (e.g., "chatml", "llama3")

---

### Phase 4: FTXUI Integration

#### 4.1 Add FTXUI Dependency (`src/v2/CMakeLists.txt`)

```cmake
# Fetch FTXUI
include(FetchContent)
FetchContent_Declare(ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/ftxui
  GIT_TAG v5.0.0
)
FetchContent_MakeAvailable(ftxui)

target_link_libraries(llaminar2 PRIVATE ftxui::screen ftxui::dom ftxui::component)
```

#### 4.2 Create `ChatUI.h/cpp` (`src/v2/utils/`)

```cpp
class ChatUI {
public:
    ChatUI(std::shared_ptr<ITokenizer> tokenizer, 
           std::shared_ptr<IPipeline> pipeline,
           std::shared_ptr<ChatTemplate> tmpl,
           const std::string& system_prompt = "");
    
    // Main loop - blocks until user exits
    void run();
    
private:
    // Token streaming callback
    void onTokenGenerated(const std::string& token);
    void onGenerationComplete();
    
    // Generate response for current input
    void generateResponse();
    
    // FTXUI components
    ftxui::Component input_box_;
    ftxui::Component conversation_view_;
    ftxui::ScreenInteractive screen_;
    
    // State
    std::vector<ChatMessage> history_;
    std::string current_input_;
    std::string current_response_;  // Accumulator for streaming
    bool is_generating_ = false;
    
    // Dependencies
    std::shared_ptr<ITokenizer> tokenizer_;
    std::shared_ptr<IPipeline> pipeline_;
    std::shared_ptr<ChatTemplate> template_;
};
```

#### 4.3 UI Layout

```
╭─ System ─────────────────────────────────────────╮
│ You are a helpful assistant.                      │
╰──────────────────────────────────────────────────╯
╭─ Conversation ───────────────────────────────────╮
│ 👤 User: What is the capital of France?          │
│ 🤖 Assistant: The capital of France is Paris.    │
│ 👤 User: Tell me more about it.                  │
│ 🤖 Assistant: Paris is the largest city in...█   │
│                                                   │
╰──────────────────────────────────────────────────╯
╭─ Input ──────────────────────────────────────────╮
│ > Type your message here...                       │
╰──────────────────────────────────────────────────╯
[Ctrl+C: Exit] [Enter: Send] [↑↓: History]
```

#### 4.4 Features
- Streaming token display (incremental updates)
- Typing indicator while generating
- Input history (arrow keys)
- Scrollable conversation view
- System prompt display
- Status bar with keybindings

---

### Phase 5: Template Implementations

| Priority | Template | Models | Format |
|----------|----------|--------|--------|
| P0 | ChatML | Qwen, many instruction-tuned | `<\|im_start\|>{role}\n{content}<\|im_end\|>\n` |
| P1 | LLaMA 3 | LLaMA 3.x | `<\|start_header_id\|>{role}<\|end_header_id\|>\n\n{content}<\|eot_id\|>` |
| P2 | Mistral | Mistral v1-v7 | `[INST] {content} [/INST]` (variations) |
| P3 | Gemma | Gemma | `<start_of_turn>{role}\n{content}<end_of_turn>\n` |
| P4 | DeepSeek | DeepSeek v2/v3 | Format varies |

---

### Phase 6: Testing

#### 6.1 Unit Tests (`tests/v2/unit/Test__ChatTemplate.cpp`)
- Template detection from strings
- Template application with various message combinations
- Edge cases (empty messages, special characters, unicode)

#### 6.2 Integration Tests (`tests/v2/integration/Test__TokenizerChatTemplate.cpp`)
- Read template from real GGUF files
- Verify encoded tokens match expected format
- Test with Qwen models (primary target)

---

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/v2/utils/ChatTemplate.h` | **NEW** | Chat template types and interface |
| `src/v2/utils/ChatTemplate.cpp` | **NEW** | Template detection and application logic |
| `src/v2/utils/ChatUI.h` | **NEW** | FTXUI chat interface |
| `src/v2/utils/ChatUI.cpp` | **NEW** | Chat UI implementation |
| `src/v2/utils/Tokenizer.h` | **MODIFY** | Add `encodeChat()` to `ITokenizer` |
| `src/v2/utils/Tokenizer.cpp` | **MODIFY** | Implement chat template in `BPETokenizer` |
| `src/v2/utils/ArgParser.h` | **MODIFY** | Add chat mode CLI options |
| `src/v2/utils/ArgParser.cpp` | **MODIFY** | Parse chat mode flags |
| `src/v2/Main.cpp` | **MODIFY** | Add chat mode entry point |
| `src/v2/CMakeLists.txt` | **MODIFY** | Add FTXUI dependency, new source files |
| `tests/v2/unit/Test__ChatTemplate.cpp` | **NEW** | Unit tests |

---

## Estimated Effort

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: Core Infrastructure | 2-3 hours | None |
| Phase 2: Tokenizer Integration | 1-2 hours | Phase 1 |
| Phase 3: CLI Integration | 1 hour | Phase 2 |
| Phase 4: FTXUI Integration | 3-4 hours | Phase 1-3 |
| Phase 5: Template Implementations | 2-3 hours | Phase 1 |
| Phase 6: Testing | 2-3 hours | All above |

**Total: ~12-16 hours**

---

## Example Usage (After Implementation)

```bash
# Raw prompt (current behavior, still works)
./run_llaminar.sh -m model.gguf -p "Hello world"

# Interactive chat mode with FTXUI
./run_llaminar.sh -m qwen2.gguf --chat

# Interactive chat with system prompt
./run_llaminar.sh -m qwen2.gguf --chat --system "You are a helpful assistant."

# Single-shot with chat template formatting
./run_llaminar.sh -m qwen2.gguf --chat-single -p "What is the capital of France?"

# Override template (for models with wrong/missing template)
./run_llaminar.sh -m model.gguf --chat --chat-template chatml
```

---

## MPI Considerations

The chat UI will only run on rank 0. Other ranks will:
1. Wait for tokenized input broadcast from rank 0
2. Participate in distributed inference
3. Send results back to rank 0 for display

```cpp
if (mpi_ctx->rank() == 0) {
    ChatUI ui(tokenizer, pipeline, template, system_prompt);
    ui.run();  // Main loop on rank 0 only
} else {
    // Worker loop: wait for commands, execute, return results
    while (true) {
        auto command = mpi_ctx->receiveCommand();
        if (command.type == CommandType::EXIT) break;
        if (command.type == CommandType::FORWARD) {
            pipeline->forward(command.tokens.data(), command.tokens.size());
            // Results gathered by rank 0 via existing MPI reduction
        }
    }
}
```

---

## Open Questions

1. **Context window management**: How to handle conversation history that exceeds max_seq_len?
   - Option A: Truncate oldest messages
   - Option B: Summarize old messages
   - Option C: Sliding window with overlap

2. **Multi-turn KV cache**: Should we preserve KV cache between turns for efficiency?
   - Currently pipeline resets between calls
   - Could save significant compute for long conversations

3. **Save/load conversations**: Should we support saving chat history to file?
