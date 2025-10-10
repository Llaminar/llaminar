# MPIAttentionKernel Stage Contracts Design

## Problem Statement

The MPIAttentionKernel has suffered from repeated bugs related to:
- **Dimension confusion**: Is it `[seq_len, heads*head_dim]` or `[heads*head_dim, seq_len]`?
- **Transpose ambiguity**: Are tensors row-major or column-major? Transposed or not?
- **Layout assumptions**: Head-interleaved vs head-sequential memory layout
- **GQA mapping errors**: Which K/V head corresponds to which Q head?
- **Silent failures**: Dimension mismatches caught too late or not at all

## Solution: Explicit Stage Contracts

Define mini-contracts between each transformation stage with:
1. **Explicit shape contracts** - required dimensions with assertions
2. **Layout specifications** - memory layout (row-major, transposed, head-interleaved)
3. **Semantic guarantees** - what the tensor represents
4. **Runtime validation** - debug-mode expensive checks

---

## Data Flow Stages

```
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 0: Input & Weight Preparation                                 │
│   IN:  global_input [seq_len, d_model]                             │
│        global_wq/wk/wv [out_features, in_features] (PyTorch conv)  │
│   OUT: local_wq [d_model, local_head_dim]                          │
│        local_wk [d_model, local_kv_head_dim]                       │
│        local_wv [d_model, local_kv_head_dim]                       │
│        local_wo [local_head_dim, d_model]                          │
│   CONTRACT: Weights are column-sliced across ranks (head sharding) │
│   LAYOUT: Row-major, PyTorch nn.Linear format [out, in]            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1: Q/K/V Projections                                          │
│   IN:  global_input [seq_len, d_model]                             │
│        local_wq [d_model, local_head_dim]                          │
│        local_bq [local_head_dim] (optional bias)                   │
│   OUT: local_q [seq_len, local_head_dim]                           │
│   CONTRACT: local_q = input @ wq.T + bq (broadcast)                │
│   LAYOUT: Row-major, [seq_len, local_head_dim]                     │
│           Each row is one token, columns are head dims             │
│   SEMANTICS: local_head_dim = local_heads * head_dim               │
│              Heads are INTERLEAVED: [h0_d0..h0_d63, h1_d0..h1_d63] │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 2: RoPE Application                                           │
│   IN:  local_q [seq_len, local_head_dim]                           │
│        local_k [seq_len, local_kv_head_dim]                        │
│   OUT: local_q [seq_len, local_head_dim] (in-place rotation)      │
│        local_k [seq_len, local_kv_head_dim] (in-place rotation)   │
│   CONTRACT: Apply RoPE to each (head, position) pair              │
│   LAYOUT: UNCHANGED - still row-major, head-interleaved           │
│   SEMANTICS: Position n_past_ to n_past_+seq_len-1 embedded       │
│   INVARIANT: Head structure preserved (no reordering)             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 3: K/V Replication for GQA (if n_head != n_head_kv)          │
│   IN:  local_k [seq_len, local_kv_head_dim]                        │
│        local_v [seq_len, local_kv_head_dim]                        │
│   OUT: local_k_replicated [seq_len, local_head_dim]               │
│        local_v_replicated [seq_len, local_head_dim]               │
│   CONTRACT: Map each Q head to corresponding K/V head              │
│             global_q_head % n_head_kv → kv_head                    │
│   LAYOUT: Row-major, head-interleaved (same as Q)                  │
│   SEMANTICS: K/V heads duplicated to match Q head count            │
│   VALIDATION: After replication, local_k and local_q have same     │
│               head dimension (local_head_dim)                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 4: Attention Computation (QK^T, Softmax, Scores@V)           │
│   IN:  local_q [seq_len, local_head_dim]                           │
│        local_k [seq_len, local_head_dim] (or _replicated)          │
│        local_v [seq_len, local_head_dim] (or _replicated)          │
│   OUT: attn_output [seq_len, local_head_dim]                       │
│   CONTRACT: Per-head attention over sequence dimension             │
│             For each head h in 0..local_heads-1:                   │
│               Q_h: [seq_len, head_dim]                             │
│               K_h: [seq_len, head_dim]                             │
│               Scores = (Q_h @ K_h.T) / sqrt(head_dim)              │
│               Probs = softmax(Scores, causal_mask)                 │
│               Out_h = Probs @ V_h                                  │
│   LAYOUT: Row-major, head-interleaved                              │
│   SEMANTICS: Each token attends to all tokens (causal for decode)  │
│   VALIDATION: Output shape matches Q shape                         │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 5: Output Projection                                          │
│   IN:  attn_output [seq_len, local_head_dim]                       │
│        local_wo [local_head_dim, d_model]                          │
│   OUT: projected_output [seq_len, d_model]                         │
│   CONTRACT: projected = attn_output @ wo                           │
│   LAYOUT: Row-major                                                │
│   SEMANTICS: Project attention results back to model dimension     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 6: Cross-Rank Gathering (if multi-rank)                      │
│   IN:  projected_output [seq_len, d_model] (partial contribution)  │
│   OUT: final_output [seq_len, d_model] (sum across ranks)          │
│   CONTRACT: MPI_Allreduce(SUM) to combine partial results          │
│   SEMANTICS: Each rank computed subset of heads, now sum them      │
│   MODE: GatherHeadsPostProjection uses Allreduce                   │
│         LocalHeads returns partial (single-rank only)              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Memory Layout Specifications

### Head-Interleaved Layout
```
[seq_len, local_heads * head_dim]
Token 0: [h0_d0, h0_d1, ..., h0_d63, h1_d0, h1_d1, ..., h1_d63, ...]
Token 1: [h0_d0, h0_d1, ..., h0_d63, h1_d0, h1_d1, ..., h1_d63, ...]
...

Access pattern for head h, dimension d, token t:
  data[t * (local_heads * head_dim) + h * head_dim + d]
```

### Weight Layout (PyTorch nn.Linear)
```
[out_features, in_features]
For wq: [n_head * head_dim, d_model]
For wo: [d_model, n_head * head_dim]

Column-major interpretation for matmul:
  output = input @ weight.T
```

---

## Implementation Strategy

### 1. Define Layout Enums
```cpp
enum class TensorLayout {
    RowMajor,           // Standard C/C++ layout
    HeadInterleaved,    // [seq, heads*head_dim] with heads interleaved
    HeadSequential,     // [seq, heads*head_dim] with heads sequential
    Transposed          // Explicitly transposed
};

enum class TensorSemantic {
    Activation,         // Activation tensor (seq_len, features)
    Weight,             // Weight matrix (PyTorch format)
    Bias,               // Bias vector
    AttentionScores,    // QK^T scores [seq_len_q, seq_len_k]
    AttentionProbs      // Softmax probabilities
};
```

### 2. Create Contract Structs
```cpp
struct TensorContract {
    std::vector<int> expected_shape;     // Expected dimensions
    TensorLayout layout;                  // Memory layout
    TensorSemantic semantic;              // What tensor represents
    std::string name;                     // Debug name
    bool allow_broadcast;                 // For bias terms
    
    // Runtime validation
    void validate(const TensorBase* tensor) const;
    void assert_compatible(const TensorContract& other) const;
};

struct StageContract {
    std::string stage_name;
    std::vector<TensorContract> inputs;
    std::vector<TensorContract> outputs;
    std::function<bool(void)> custom_validator;  // Optional extra checks
    
    void validate_inputs(const std::vector<std::shared_ptr<TensorBase>>& tensors) const;
    void validate_outputs(const std::vector<std::shared_ptr<TensorBase>>& tensors) const;
};
```

### 3. Define Stage Contracts
```cpp
// In MPIAttentionKernel.h
class MPIAttentionKernel : public MPIKernelBase {
private:
    // Stage contract definitions
    StageContract contract_projection_;
    StageContract contract_rope_;
    StageContract contract_gqa_replication_;
    StageContract contract_attention_;
    StageContract contract_output_projection_;
    
    void initializeContracts();  // Called in constructor
};
```

### 4. Runtime Validation
```cpp
void MPIAttentionKernel::computeLocalProjections(...) {
    // PRE-CONDITIONS
    contract_projection_.validate_inputs({global_input, local_wq, local_bq});
    
    // ... actual computation ...
    
    // POST-CONDITIONS
    contract_projection_.validate_outputs({local_q, local_k, local_v});
}
```

### 5. Debug-Only Expensive Checks
```cpp
#ifdef LLAMINAR_VALIDATE_ATTENTION_STAGES
    // Expensive checks (disabled in release)
    void validateProjectionCorrectness(const Tensor& Q, const Tensor& input, 
                                        const Tensor& wq, const Tensor& bq);
    void validateRoPEInvariant(const Tensor& Q_before, const Tensor& Q_after);
    void validateGQAMapping(const Tensor& K_orig, const Tensor& K_replicated);
#endif
```

---

## Assertions to Add

### Stage 0: Weight Distribution
```cpp
// ASSERT: PyTorch weight format [out_features, in_features]
assert(global_wq->shape()[0] == n_head_ * head_dim_);
assert(global_wq->shape()[1] == d_model_);

// ASSERT: Bias matches output features
if (global_bq) {
    assert(global_bq->shape()[0] == n_head_ * head_dim_);
}

// ASSERT: Local slice dimensions
assert(local_wq->shape()[1] == local_heads * head_dim_);
assert(local_wo->shape()[0] == local_heads * head_dim_);
```

### Stage 1: Projection
```cpp
// PRE: Input shape
assert(global_input->shape()[0] == seq_len);
assert(global_input->shape()[1] == d_model_);

// PRE: Weight compatibility
assert(local_wq->shape()[0] == d_model_);
assert(local_wq->shape()[1] == local_heads * head_dim_);

// POST: Output shape
assert(local_q->shape()[0] == seq_len);
assert(local_q->shape()[1] == local_heads * head_dim_);
```

### Stage 2: RoPE
```cpp
// PRE/POST: Shape invariant
assert(local_q_after->shape() == local_q_before->shape());

// SEMANTIC: Position range valid
assert(n_past_ >= 0);
assert(n_past_ + seq_len <= context_length_);
```

### Stage 3: GQA Replication
```cpp
// PRE: K/V head count
assert(local_kv_heads == local_kv_head_dim / head_dim_);

// POST: Replicated matches Q
assert(local_k_replicated->shape()[1] == local_q->shape()[1]);
assert(local_v_replicated->shape()[1] == local_q->shape()[1]);

// SEMANTIC: Head mapping valid
for each q_head:
    kv_head = (head_offset + q_head) % n_head_kv_;
    assert(kv_head >= 0 && kv_head < n_head_kv_);
```

### Stage 4: Attention
```cpp
// PRE: Matching head dimensions
assert(local_q->shape()[1] == local_k->shape()[1]);
assert(local_q->shape()[1] == local_v->shape()[1]);

// PRE: Sequence length consistency
assert(local_k->shape()[0] >= local_q->shape()[0]);  // K can be longer (KV cache)

// POST: Output matches Q shape
assert(attn_output->shape() == local_q->shape());
```

### Stage 5: Output Projection
```cpp
// PRE: Wo compatibility
assert(local_wo->shape()[0] == local_heads * head_dim_);
assert(local_wo->shape()[1] == d_model_);

// POST: Back to model dimension
assert(projected_output->shape()[0] == seq_len);
assert(projected_output->shape()[1] == d_model_);
```

---

## Example: Contract-Based Function Signature

### Before (implicit contract)
```cpp
void computeLocalProjections(
    const std::shared_ptr<TensorBase>& input,
    const std::shared_ptr<TensorBase>& wq,
    std::shared_ptr<TensorBase>& q,
    ...
);
```

### After (explicit contract)
```cpp
void computeLocalProjections(
    const std::shared_ptr<TensorBase>& input,    // [seq_len, d_model]
    const std::shared_ptr<TensorBase>& wq,       // [d_model, local_head_dim]
    const std::shared_ptr<TensorBase>& bq,       // [local_head_dim] or nullptr
    std::shared_ptr<TensorBase>& q,              // OUT: [seq_len, local_head_dim]
    ...
) {
    // CONTRACT VALIDATION
    const TensorContract input_contract{
        .expected_shape = {(int)seq_len, d_model_},
        .layout = TensorLayout::RowMajor,
        .semantic = TensorSemantic::Activation,
        .name = "global_input"
    };
    input_contract.validate(input.get());
    
    // ... rest of implementation ...
}
```

---

## Benefits

1. **Self-Documenting**: Function signatures explicitly state tensor contracts
2. **Early Error Detection**: Assertions catch dimension mismatches immediately
3. **Debugging Aid**: Clear error messages pinpoint exactly which contract failed
4. **Refactoring Safety**: Contracts enforce correctness during code changes
5. **Onboarding**: New developers understand data flow from contracts
6. **Testing**: Contracts enable property-based testing of stages

---

## Rollout Plan

### Phase 1: Add Basic Assertions (Immediate)
- Add shape assertions at stage boundaries
- Validate weight dimensions match PyTorch format
- Check GQA head mapping logic

### Phase 2: Contract Infrastructure (Next PR)
- Implement `TensorContract` and `StageContract` classes
- Define contracts for all 6 stages
- Add runtime validation wrappers

### Phase 3: Enhanced Validation (Optional)
- Add debug-mode expensive checks
- Implement numerical validation (e.g., verify projection correctness)
- Add contract violation diagnostics with detailed error messages

---

## Open Questions

1. Should contracts be debug-only or always enabled?
   - **Recommendation**: Always enable basic shape checks, debug-only for expensive validation

2. Should we add contracts to other kernels (RMSNorm, RoPE, etc.)?
   - **Recommendation**: Yes, apply same pattern project-wide

3. How to handle dynamic shapes (variable seq_len)?
   - **Recommendation**: Use placeholder `-1` in expected_shape for dynamic dims

4. Should contracts include memory layout transformations?
   - **Recommendation**: Yes, explicitly track transpose operations
