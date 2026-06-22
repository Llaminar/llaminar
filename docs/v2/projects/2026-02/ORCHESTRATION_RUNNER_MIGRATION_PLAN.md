# OrchestrationRunner Migration Plan

> **Goal**: Replace `ArgParser` + `InferenceRunnerFactory` with `OrchestrationConfigParser` + `OrchestrationRunnerFactory`  
> **Scope**: Full migration - extend `OrchestrationConfig` to include all inference parameters  
> **Branch**: `tensor-parallel`

## Executive Summary

The current codebase has two parallel CLI parsing and runner creation paths:

| Component | Legacy (Active) | New (Built, Not Wired) |
|-----------|-----------------|------------------------|
| CLI Parser | `ArgParser` → `ArgContext` | `OrchestrationConfigParser` → `OrchestrationConfig` |
| Runner Factory | `InferenceRunnerFactory::createInferenceRunner()` | `OrchestrationRunnerFactory::createFromArgs()` |
| Runner | `IInferenceRunner` (DeviceGraphOrchestrator) | `IOrchestrationRunner` (OrchestrationRunner) |

This migration will:
1. Extend `OrchestrationConfig` to include all inference parameters (model, sampling, chat, benchmark)
2. Extend `OrchestrationConfigParser` to parse all CLI arguments
3. Wire `Main.cpp` to use `OrchestrationRunnerFactory`
4. Deprecate and remove `ArgParser` + `ArgContext`

## Architecture After Migration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Main.cpp                                                                   │
│                                                                             │
│  1. OrchestrationConfigParser::parseArgs(argc, argv)                        │
│     → OrchestrationConfig (contains ALL config: model, inference, orch)     │
│                                                                             │
│  2. OrchestrationRunnerFactory::createFromOrchestrationConfig(config)       │
│     → IOrchestrationRunner                                                  │
│                                                                             │
│  3. runner->initialize()                                                    │
│     - Builds execution plan                                                 │
│     - Sets up LOCAL TP context                                              │
│     - Loads model weights                                                   │
│     - Builds compute graph                                                  │
│                                                                             │
│  4. Main loop using runner->prefill() / runner->decodeStep()                │
│     OR runner->generate() for simple cases                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Extend OrchestrationConfig

### 1.1 Add Model Configuration

**File**: `src/v2/config/OrchestrationConfig.h`

Add to `OrchestrationConfig` struct:

```cpp
// =========================================================================
// Model Configuration
// =========================================================================

std::string model_path;              ///< Path to GGUF model file (-m, --model)
int max_seq_len = 2048;              ///< Maximum sequence length (--context-length)
bool use_mmap = true;                ///< Use memory-mapped file loading (--mmap/--no-mmap)

// Weight precision
WeightPrecision weight_precision = WeightPrecision::NATIVE;

// Activation precision  
ActivationPrecision activation_precision = ActivationPrecision::FP32;
```

### 1.2 Add Inference Configuration

```cpp
// =========================================================================
// Inference Configuration
// =========================================================================

std::string prompt;                  ///< Input prompt (-p, --prompt)
int n_predict = -1;                  ///< Tokens to generate (-n); -1 = until EOS
int batch_size = 1;                  ///< Batch size (--batch-size)
int n_threads = -1;                  ///< Thread count; -1 = auto (--threads)
int seed = -1;                       ///< RNG seed; -1 = random (--seed)
```

### 1.3 Add Sampling Configuration

```cpp
// =========================================================================
// Sampling Configuration
// =========================================================================

float temperature = 0.8f;            ///< Sampling temperature (--temperature)
int top_k = 40;                      ///< Top-K sampling (--top-k)
float top_p = 0.9f;                  ///< Top-P (nucleus) sampling (--top-p)
bool deterministic = false;          ///< Force deterministic mode (--deterministic)
```

### 1.4 Add Chat Configuration

```cpp
// =========================================================================
// Chat Configuration
// =========================================================================

bool chat_mode = false;              ///< Interactive chat mode (--chat)
bool single_shot_chat = false;       ///< Single prompt with chat template (--chat-single)
std::string system_prompt;           ///< System message (--system)
std::string chat_template_override;  ///< Template override (--chat-template)
```

### 1.5 Add Benchmark Configuration

```cpp
// =========================================================================
// Benchmark Configuration
// =========================================================================

bool benchmark_mode = false;         ///< Run benchmark (--benchmark)
```

### 1.6 Add Fused Attention Configuration

```cpp
// =========================================================================
// Fused Attention Configuration
// =========================================================================

bool use_fused_attention = false;    ///< Use fused attention+Wo kernel (--fused-attention)
FusedAttentionBackend fused_attention_backend = FusedAttentionBackend::JIT;
```

### 1.7 Add MPI Bootstrap Configuration

```cpp
// =========================================================================
// MPI Bootstrap Configuration
// =========================================================================

int mpi_procs = 0;                   ///< MPI process count; 0 = auto (--mpi-procs)
std::string hostfile;                ///< MPI hostfile path (--hostfile)
bool mpi_dry_run = false;            ///< Print MPI config and exit (--mpi-dry-run)
bool mpi_verbose = false;            ///< Verbose MPI output (--mpi-verbose)
bool mpi_no_bootstrap = false;       ///< Disable auto-bootstrap (--no-mpi-bootstrap)
bool mpi_oversubscribe = false;      ///< Allow oversubscription (--mpi-oversubscribe)
```

### 1.8 Add Verbosity/Debug Configuration

```cpp
// =========================================================================
// Verbosity and Debug
// =========================================================================

int verbose_level = 0;               ///< 0=INFO, 1=DEBUG (-v), 2=TRACE (-vv)
bool list_devices = false;           ///< List devices and exit (--list-devices)
bool show_help = false;              ///< Show help and exit (--help)
```

---

## Phase 2: Extend OrchestrationConfigParser

### 2.1 Parse All New Fields

**File**: `src/v2/config/OrchestrationConfigParser.cpp`

Extend `parseArgs()` to handle all arguments currently in `ArgParser`:

| Argument | Field | Type |
|----------|-------|------|
| `-m`, `--model` | `model_path` | string |
| `-p`, `--prompt` | `prompt` | string |
| `-n`, `--n-predict` | `n_predict` | int |
| `-c`, `--context-length` | `max_seq_len` | int |
| `-t`, `--temperature` | `temperature` | float |
| `--top-k` | `top_k` | int |
| `--top-p` | `top_p` | float |
| `--seed` | `seed` | int |
| `--batch-size` | `batch_size` | int |
| `--threads` | `n_threads` | int |
| `--mmap` / `--no-mmap` | `use_mmap` | bool |
| `--weight-precision` | `weight_precision` | enum |
| `--activation-precision` | `activation_precision` | enum |
| `--chat` | `chat_mode` | bool |
| `--chat-single` | `single_shot_chat` | bool |
| `--system` | `system_prompt` | string |
| `--chat-template` | `chat_template_override` | string |
| `--benchmark` | `benchmark_mode` | bool |
| `--fused-attention` | `use_fused_attention` | bool |
| `--fused-attention-backend` | `fused_attention_backend` | enum |
| `--mpi-procs` | `mpi_procs` | int |
| `--hostfile` | `hostfile` | string |
| `--mpi-dry-run` | `mpi_dry_run` | bool |
| `--mpi-verbose` | `mpi_verbose` | bool |
| `--no-mpi-bootstrap` | `mpi_no_bootstrap` | bool |
| `--mpi-oversubscribe` | `mpi_oversubscribe` | bool |
| `-v` | `verbose_level++` | int |
| `--list-devices` | `list_devices` | bool |
| `-h`, `--help` | `show_help` | bool |
| `--deterministic` | `deterministic` | bool |

### 2.2 Update Help Text

Update `OrchestrationConfigParser::getHelpText()` to include all options.

### 2.3 Validation

Add validation for:
- `model_path` must be non-empty (unless `--help`, `--list-devices`, `--dry-run`)
- `temperature >= 0`
- `top_k >= 0`
- `top_p` in range (0, 1]
- `n_predict >= -1`

---

## Phase 3: Update OrchestrationRunner

### 3.1 Use Config Fields for Weight Loading

**File**: `src/v2/execution/runner/OrchestrationRunner.cpp`

Update `loadWeights()` to use `config_.model_path`:

```cpp
bool OrchestrationRunner::loadWeights()
{
    if (config_.model_path.empty()) {
        return setError("No model path specified");
    }

    model_ctx_ = ModelContext::create(
        config_.model_path,
        mpi_ctx_,
        nullptr,
        nullptr,
        WeightDistributionStrategy::REPLICATED,  // TODO: derive from config
        config_.weight_precision);
    
    // ...
}
```

### 3.2 Pass Sampling Params Through

Ensure `OrchestrationRunner::generate()` and `decodeStep()` use config sampling params.

### 3.3 Expose Tokenizer Access

Add method to get tokenizer (needed for main loop):

```cpp
std::shared_ptr<ITokenizer> OrchestrationRunner::tokenizer() const;
```

---

## Phase 4: Rewrite Main.cpp

### 4.1 Replace ArgParser with OrchestrationConfigParser

```cpp
// OLD:
#include "utils/ArgParser.h"
ArgContext args = ArgParser::parse(argc, argv);

// NEW:
#include "config/OrchestrationConfigParser.h"
OrchestrationConfigParser parser;
OrchestrationConfig config = parser.parseArgs(argc, argv);
```

### 4.2 Replace InferenceRunnerFactory with OrchestrationRunnerFactory

```cpp
// OLD:
#include "execution/factory/InferenceRunnerFactory.h"
auto runner = createInferenceRunner(model_ctx, mpi_ctx, device_id, runner_config);

// NEW:
#include "execution/runner/IOrchestrationRunnerFactory.h"
auto factory = createOrchestrationRunnerFactory();
auto runner = factory->createFromOrchestrationConfig(config);
if (!runner->initialize()) {
    LOG_ERROR("Initialization failed: " << runner->lastError());
    return 1;
}
```

### 4.3 Handle Early Exit Cases

```cpp
if (config.show_help) {
    OrchestrationConfigParser::printHelp();
    return 0;
}

if (config.list_devices) {
    list_devices();
    return 0;
}

if (config.dry_run) {
    // Print configuration summary
    std::cout << config.toString() << std::endl;
    return 0;
}
```

### 4.4 Update Chat/Benchmark Modes

Adapt the chat and benchmark code paths to use `IOrchestrationRunner` interface.

### 4.5 Update Decode Loop

```cpp
// Main decode loop
for (int i = 0; i < config.n_predict; ++i) {
    GenerationResult step = runner->decodeStep();
    
    if (!step.success()) {
        LOG_ERROR("Decode failed: " << step.error);
        break;
    }
    
    // Output token
    for (int32_t token : step.tokens) {
        std::cout << tokenizer->decode_token(token) << std::flush;
        if (tokenizer->is_stop_token(token)) {
            goto done;
        }
    }
    
    if (step.is_complete) break;
}
done:
```

---

## Phase 5: Cleanup

### 5.1 Remove Legacy Files

After migration is complete and tests pass:

- [ ] Remove `src/v2/utils/ArgParser.h`
- [ ] Remove `src/v2/utils/ArgParser.cpp`
- [ ] Remove `ArgContext` struct
- [ ] Remove `LegacyOrchestrationConfig` from `DeviceOrchestrator.h`
- [ ] Update CMakeLists.txt to remove ArgParser sources

### 5.2 Update Tests

- [ ] Update any tests using `ArgParser` to use `OrchestrationConfigParser`
- [ ] Add unit tests for new `OrchestrationConfig` fields
- [ ] Add integration tests for `OrchestrationRunner` with real inference

---

## Implementation Order

| Step | Task | Files | Effort |
|------|------|-------|--------|
| 1 | Add new fields to `OrchestrationConfig` | `OrchestrationConfig.h` | Small |
| 2 | Extend `OrchestrationConfigParser::parseArgs()` | `OrchestrationConfigParser.cpp` | Medium |
| 3 | Update `OrchestrationRunner::loadWeights()` | `OrchestrationRunner.cpp` | Small |
| 4 | Add tokenizer accessor to `OrchestrationRunner` | `OrchestrationRunner.h/cpp`, `IOrchestrationRunner.h` | Small |
| 5 | Rewrite `Main.cpp` | `Main.cpp` | Large |
| 6 | Test and fix | - | Medium |
| 7 | Remove legacy code | Multiple | Small |

---

## Testing Checklist

- [ ] `./llaminar2 --help` shows all options
- [ ] `./llaminar2 --list-devices` works
- [ ] `./llaminar2 -m model.gguf -p "Hello"` basic inference
- [ ] `./llaminar2 -m model.gguf -p "Hello" -n 50` with token limit
- [ ] `./llaminar2 -m model.gguf --chat-single -p "Hi"` chat mode
- [ ] `./llaminar2 -m model.gguf --benchmark` benchmark mode
- [ ] `mpirun -np 2 ./llaminar2 -m model.gguf --tp 2` tensor parallel
- [ ] `./llaminar2 -m model.gguf --tp 2 --tp-scope local` local TP
- [ ] `./llaminar2 -m model.gguf --dry-run` dry run
- [ ] `./llaminar2 -m model.gguf --deterministic` deterministic mode

---

## Risk Mitigation

1. **Gradual Transition**: Keep `ArgParser` temporarily and add feature flag `--use-legacy-runner`
2. **Test Parity**: Run same prompts through old and new paths, compare outputs
3. **Rollback Plan**: Git branch allows easy revert if issues found

---

## Success Criteria

1. All existing CLI functionality works with new parser
2. `ArgParser.h/cpp` removed from codebase
3. All tests pass
4. No regression in inference quality or performance
