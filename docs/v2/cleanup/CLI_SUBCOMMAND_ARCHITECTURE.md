# CLI Subcommand Architecture

**Status**: In Progress  
**Date**: 2026-04-21  

## Overview

Introduce first-class CLI subcommands to replace the current flat-flag dispatch model.

```
llaminar <subcommand> [flags]

Subcommands:
  plan       Analyze cluster + produce execution plan file
  serve      Start OpenAI-compatible HTTP server
  oneshot    Single inference (completion, chat, benchmark) + exit
  describe   Print cluster/device inventory
```

## Current Architecture

```
Main.cpp → AppLifecycle::run()
  → OrchestrationConfigParser::parseArgs()   // ALL flags → monolithic OrchestrationConfig
  → early-exit checks (--help, --show-topology, --list-devices, --validate-only)
  → MPIBootstrapPhase::execute()
  → RuntimeInitPhase::execute()              // MPI init, model load, runner creation
  → IExecutionMode chain (first matches() wins):
      InteractiveChatMode → SingleShotChatMode → BenchmarkMode → ServerMode → CompletionMode
```

Problems:
- Every invocation pays the full MPI bootstrap + model load cost, even `--show-topology`
- Mode selection is implicit (boolean flag combos), not explicit
- `--help` output is a wall of 70+ flags with no grouping by use-case
- No concept of "plan" or "describe" as standalone operations

## Target Architecture

```
Main.cpp → AppLifecycle::run()
  Phase 0: Peel argv[1] as subcommand token
  Phase 1: Route to ICommand implementation
    ├── "plan"     → PlanCommand
    ├── "serve"    → ServeCommand
    ├── "oneshot"  → OneshotCommand
    ├── "describe" → DescribeCommand
    ├── "--help"   → print top-level help, exit 0
    └── fallback   → LegacyCommand (existing AppLifecycle::run body)

  ICommand interface:
    name()                → human label
    buildSpec()           → CliSpec<OrchestrationConfig> with command-specific flags
    execute(config, ...) → run the command
```

### Subcommand → Init Path

| Subcommand | Needs MPI? | Needs Model? | Init Path |
|------------|-----------|-------------|-----------|
| `describe` | No (local) / Optional (multi-node) | No | Topology detection only |
| `plan`     | Optional | Header only (arch, layers) | Lightweight model probe |
| `oneshot`  | Yes (if TP) | Full | MPIBootstrapPhase → RuntimeInitPhase |
| `serve`    | Yes (if TP) | Full | MPIBootstrapPhase → RuntimeInitPhase |
| (legacy)   | Yes | Full | Existing path unchanged |

### Flag Mapping

| Subcommand | Flags |
|------------|-------|
| `plan`     | `-m` (required), `-s/--strategy`, `-o/--output`, `--format`, `--benchmark-devices`, shared parallelism flags |
| `serve`    | `-m` (required unless `--plan`), `--plan`, `--host`, `--port`, shared parallelism flags |
| `oneshot`  | `-m`, `-p`, `-n`, `-t`, `--temperature`, `--chat`, `--benchmark`, shared parallelism/device flags |
| `describe` | `--hostfile`, `-o/--output`, `--format` |
| (shared)   | `-d`, `-tp`, `-pp`, `--device-map`, `--config`, `--log-level`, `--verbose` |

### Backward Compatibility

If `argv[1]` starts with `-` or no subcommand is given, `LegacyCommand` runs the
existing `AppLifecycle::run()` body identically. Zero breaking change.

## Implementation Plan

### Phase 1: Routing Skeleton (this PR)

New files:
- `src/v2/app/ICommand.h` — interface
- `src/v2/app/SubcommandRouter.h` — argv[1] dispatch + top-level help
- `src/v2/app/commands/LegacyCommand.h/.cpp` — wraps existing AppLifecycle body
- `src/v2/app/commands/DescribeCommand.h/.cpp` — `--show-topology` etc. as subcommand
- `tests/v2/unit/app/Test__SubcommandRouter.cpp` — routing unit tests

Modified files:
- `src/v2/app/AppLifecycle.cpp` — delegate to SubcommandRouter
- `src/v2/CMakeLists.txt` — add new sources
- `tests/v2/CMakeLists.txt` — add test target

### Phase 2: `oneshot` command (future PR)
### Phase 3: `serve` command (future PR)
### Phase 4: `plan` command (future PR)

## ICommand Interface

```cpp
class ICommand {
public:
    virtual ~ICommand() = default;
    virtual const char* name() const = 0;
    virtual const char* description() const = 0;
    virtual int execute(int argc, char* argv[]) = 0;
};
```

Commands own their full lifecycle: parse their own flags, perform their own init,
and run. This decouples heavy commands (serve, oneshot) from light ones (describe).

## SubcommandRouter

```cpp
class SubcommandRouter {
public:
    void add(std::unique_ptr<ICommand> cmd);
    int dispatch(int argc, char* argv[]);
    std::string getTopLevelHelp() const;
};
```

`dispatch()` logic:
1. If `argc < 2` or `argv[1]` starts with `-`: delegate to LegacyCommand
2. If `argv[1] == "--help"` or `-h`: print top-level help, return 0
3. Look up `argv[1]` in registered commands
4. If found: call `cmd->execute(argc - 1, argv + 1)` (shift past subcommand token)
5. If not found: delegate to LegacyCommand (backward compat for bare model paths etc.)
