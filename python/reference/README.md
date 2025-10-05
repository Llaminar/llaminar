# PyTorch Reference Implementation for Llaminar Parity Testing

This module provides PyTorch reference implementations of transformer models for comprehensive parity testing with Llaminar's C++ implementation. Unlike llama.cpp (which only exposes final outputs), these references capture **all intermediate pipeline stages**.

## Architecture

### Design Principles
- **Extensible**: Easy to add new model families (Qwen, LLaMA, DeepSeek, GPT, etc.)
- **Clean abstractions**: Base class defines interface, model-specific implementations inherit
- **Factory pattern**: Unified API via `create_reference_model()`
- **Zero boilerplate**: Minimal code needed to add new models

### Components

```
python/reference/
├── __init__.py              # Package exports
├── pipeline_stages.py       # PipelineStage enum (synced with C++)
├── base.py                  # AbstractReferenceModel base class
├── registry.py              # ModelRegistry factory pattern
├── qwen.py                  # Qwen/Qwen2 implementation
├── llama.py                 # LLaMA stub (TODO: complete)
├── utils.py                 # Snapshot comparison utilities
├── run_reference.py         # CLI tool for inference
└── tests/
    └── test_reference.py    # Unit tests
```

## Quick Start

### Installation

Dependencies are listed in `/workspaces/llaminar/requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch transformers safetensors accelerate bitsandbytes
```

### Basic Usage

```python
from python.reference import create_reference_model, PipelineStage

# Create and load model
model = create_reference_model("qwen", "Qwen/Qwen2-0.5B-Instruct")

# Run inference with stage capture
result = model.forward(
    token_ids=[1, 2, 3, 4],
    capture_stages=[
        PipelineStage.EMBEDDING,
        PipelineStage.ATTENTION_OUTPUT,
        PipelineStage.FFN_DOWN,
        PipelineStage.LM_HEAD
    ]
)

# Access outputs
logits = result["logits"]              # Final logits
snapshots = result["snapshots"]        # Captured stages

# Export for C++ test integration
model.export_snapshots("snapshots.npz", format="npz")
```

### CLI Tool

```bash
# Run inference and export snapshots
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4 \
    --output snapshots.npz \
    --verbose

# Capture specific stages only
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B \
    --tokens 1,2,3 \
    --stages EMBEDDING,ATTENTION_OUTPUT,LM_HEAD \
    --output test_snapshots.npz

# Use quantization
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B \
    --tokens 1,2,3 \
    --quantization 4bit \
    --output snapshots_q4.npz
```

## Captured Pipeline Stages

The reference implementation captures all 22 pipeline stages defined in `src/pipeline_stages.h`:

| Stage | Description | Layer Index |
|-------|-------------|-------------|
| `EMBEDDING` | Token embedding lookup | -1 (global) |
| `ATTENTION_NORM` | Pre-attention normalization | 0..N |
| `QKV_PROJECTION` | Q, K, V projections | 0..N |
| `ROPE_APPLICATION` | Rotary position embeddings | 0..N |
| `ATTENTION_OUTPUT` | Attention output projection | 0..N |
| `ATTENTION_RESIDUAL` | Post-attention residual | 0..N |
| `FFN_NORM` | Pre-FFN normalization | 0..N |
| `FFN_GATE` | Gate projection (SwiGLU) | 0..N |
| `FFN_UP` | Up projection | 0..N |
| `FFN_SWIGLU` | SwiGLU activation | 0..N |
| `FFN_DOWN` | Down projection | 0..N |
| `FFN_RESIDUAL` | Post-FFN residual | 0..N |
| `FINAL_NORM` | Final normalization | -1 (global) |
| `LM_HEAD` | Language model head logits | -1 (global) |

## Adding New Models

Adding support for a new model family is straightforward:

```python
# 1. Create new file (e.g., deepseek.py)
from .base import AbstractReferenceModel
from .registry import ModelRegistry
from transformers import AutoModelForCausalLM

class DeepSeekReferenceModel(AbstractReferenceModel):
    def load_model(self, **kwargs):
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, **kwargs
        )
        self._register_hooks()  # Similar to Qwen
    
    def forward(self, token_ids, **kwargs):
        # Similar to Qwen implementation
        pass

# 2. Register the model
ModelRegistry.register("deepseek", DeepSeekReferenceModel)

# 3. Import in __init__.py
from . import deepseek

# 4. Done! Now you can use it:
model = create_reference_model("deepseek", "deepseek-ai/deepseek-coder-6.7b")
```

## C++ Integration

### Export Snapshots

```python
model.export_snapshots("snapshots.npz", format="npz")
```

### Load in C++ Tests

```cpp
// TODO: Add snapshot loader to test_parity_framework.cpp
// Load .npz file and compare with Llaminar captures

#include <cnpy.h>  // or similar library

auto npz_data = cnpy::npz_load("snapshots.npz");
auto embedding_snapshot = npz_data["EMBEDDING_-1"];
// Compare with Llaminar snapshot...
```

## Snapshot Comparison

```python
from python.reference.utils import compare_snapshots, print_snapshot_summary

# Compare two runs
snapshots1 = model1.get_snapshots()
snapshots2 = model2.get_snapshots()

comparison = compare_snapshots(snapshots1, snapshots2)

print(f"Overall match: {comparison['overall_match']}")
print(f"Matching stages: {len(comparison['matching_stages'])}")

for key, metrics in comparison["metrics"].items():
    stage, layer = key
    print(f"{metrics['stage']} (layer {layer}):")
    print(f"  Max abs diff: {metrics['max_abs_diff']:.6e}")
    print(f"  Rel L2: {metrics['rel_l2']:.6e}")
    print(f"  Allclose: {metrics['allclose']}")
```

## Testing

```bash
# Run unit tests
cd /workspaces/llaminar
pytest python/reference/tests/ -v

# Specific test
pytest python/reference/tests/test_reference.py::TestModelRegistry -v
```

## Current Status

### Completed ✅
- Base architecture (`AbstractReferenceModel`, `ModelRegistry`)
- PipelineStage enum synchronized with C++
- Qwen/Qwen2 reference implementation
- Snapshot capture and export (.npz, .json)
- CLI tool for standalone inference
- Utility functions for comparison
- Unit tests

### TODO 📋
- Complete LLaMA implementation
- Add DeepSeek support
- Add GPT-style models support
- C++ snapshot loader (.npz → parity framework)
- Update `test_parity_framework.cpp` to use PyTorch snapshots
- CI integration
- Quantization validation (Q4 vs GGUF precision)

## Design Rationale

### Why not just use llama.cpp?

llama.cpp only exposes 2 API functions for state extraction:
- `llama_get_logits_ith()` - final logits only
- `llama_get_embeddings_ith()` - final hidden state only

This is insufficient for comprehensive parity testing. We need access to:
- Intermediate layer outputs
- Attention mechanism internals
- FFN intermediate activations
- Residual connection states
- Normalization outputs

### Why PyTorch + Transformers?

1. **HuggingFace transformers** provides reference implementations for all major architectures
2. **Forward hooks** allow capturing any intermediate state without modifying model code
3. **Quantization support** via bitsandbytes matches GGUF formats
4. **Easy extensibility** - adding new models is trivial
5. **Well-tested** - transformers library is industry standard

### Architecture Decisions

- **Abstract base class**: Enforces consistent interface across models
- **Factory pattern**: Clean API, extensible without modifying core code
- **Stage enum synchronization**: Explicit mapping to C++ ensures compatibility
- **NumPy export**: Language-neutral format for C++ integration
- **Lazy loading**: Models only loaded when needed (memory efficiency)

## Performance Notes

- Reference implementation is for **validation only**, not production inference
- PyTorch is slower than Llaminar's optimized C++/COSMA implementation
- Quantization (4bit/8bit) reduces memory but adds computation overhead
- Use `device="cuda"` for GPU acceleration if available

## Contributing

When adding a new model:
1. Create `<model_name>.py` in `python/reference/`
2. Inherit from `AbstractReferenceModel`
3. Implement `load_model()` and `forward()`
4. Register via `ModelRegistry.register()`
5. Add tests in `python/reference/tests/`
6. Update this README

See `qwen.py` as a reference implementation template.

## License

Same as Llaminar project.

## Author

David Sanftenberg
