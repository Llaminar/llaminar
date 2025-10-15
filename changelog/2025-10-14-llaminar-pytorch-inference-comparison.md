# Llaminar vs PyTorch Inference Comparison - October 14, 2025

## Test Objective
Compare real-world inference between Llaminar and PyTorch on an English question.

## Test Question
**Input**: "What is the capital of France?"

## PyTorch Reference Results ✅

### Tokenization
- **Tokens**: `[3838, 374, 279, 6722, 315, 9625, 30]`
- **Decoded**: "What is the capital of France?"

### Generated Output (10 tokens, greedy decoding)
```
Token 1: 576  -> ' The'
Token 2: 6722 -> ' capital'
Token 3: 315  -> ' of'
Token 4: 9625 -> ' France'
Token 5: 374  -> ' is'
Token 6: 12095-> ' Paris'
Token 7: 13   -> '.'
Token 8: 1084 -> ' It'
Token 9: 374  -> ' is'
Token 10: 279 -> ' the'
```

**Full Response**: "What is the capital of France? The capital of France is Paris. It is the"

**Generated portion**: " The capital of France is Paris. It is the"

## Llaminar Results ❌

### Critical Issue Discovered: Tokenizer Mismatch

**Llaminar Tokenization**:
- **Tokens**: `[1639, 266, 285, 339, 68, 11346, 275, 278, 1055, 75331, 346, 30]`
- **Decoded**: "Wh", "at", "is", "th", "e", "cap", "it", "al", "of", "Fran", "ce", "?"

**Problem**: Llaminar's tokenizer is producing **completely different tokens** than PyTorch!
- PyTorch: 7 tokens
- Llaminar: 12 tokens  
- **Zero overlap** in token IDs

### Root Cause Analysis

1. **Parity Tests Pass**: The parity framework tests (OpenBLAS, COSMA, TrueIncremental) all pass 100% because they use **PyTorch-generated tokens** from snapshot files.

2. **Real Inference Hangs**: When using Llaminar's built-in tokenizer, it:
   - Produces different tokens
   - Hangs during inference (appears to be stuck in a long operation)
   - Never produces output

3. **Tokenizer Source**:
   - PyTorch uses: `AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")`
   - Llaminar uses: Custom GGUF tokenizer implementation reading from the GGUF file

### Behavior Observed

```
[14:57:12.770] [INFO] Processing prompt: "What is the capital of France?"
[14:57:12.770] [INFO] Tokenized to 12 tokens
Prompt: "What is the capital of France?"
Tokens: [1639, 266, 285, 339, 68, 11346, 275, 278, 1055, 75331, 346, 30]
Token count: 12
Token strings: ["Wh", "at", "is", "th", "e", "cap", "it", "al", "of", "Fran", "ce", "?"]
[HANGS INDEFINITELY - timeout after 15+ seconds]
```

## Conclusions

### ✅ What Works
1. **Core inference engine**: All 3 parity tests pass 100% when using identical tokens
   - OpenBLASPrefillVsPyTorch: 387/387 stages ✅
   - COSMAPrefillVsPyTorch: 387/387 stages ✅  
   - TrueIncrementalDecodeVsPyTorch: 1170/1170 stages ✅

2. **Production code is healthy**: When fed the correct tokens, Llaminar produces bit-exact matching output with PyTorch

### ❌ What's Broken
1. **Tokenizer implementation**: The GGUF tokenizer in Llaminar produces completely different tokens than the HuggingFace tokenizer
   - Different vocabulary mapping
   - Different BPE merge rules or application
   - Results in semantically invalid token sequences

2. **Inference hangs** with wrong tokens (likely due to invalid token IDs or unexpected sequence patterns)

## Recommendations

### Immediate Fix Required
**Fix the GGUF tokenizer implementation** to match HuggingFace output:
1. Verify vocabulary loading from GGUF matches HF tokenizer
2. Check BPE merge rules are correctly extracted and applied
3. Ensure special token handling matches
4. Add tokenizer parity tests comparing GGUF vs HF tokenization

### Testing Strategy
1. **Add tokenizer unit tests**: Compare GGUF tokenizer output against HF for common phrases
2. **Tokenizer parity test**: Extend parity framework to validate tokenization step
3. **End-to-end test**: Once tokenizer is fixed, rerun this real inference test

### Alternative Workaround
For now, users can:
1. Use PyTorch to tokenize input
2. Pass raw token IDs to Llaminar
3. Use PyTorch to decode output tokens

This bypasses the broken tokenizer while leveraging the validated inference engine.

## Files Referenced
- PyTorch test script: `/workspaces/llaminar/test_real_inference.py`
- Verbose output log: `/tmp/llaminar_verbose.log`
- Input tokens file: `/tmp/test_tokens.txt`

## Next Steps
1. ⚠️ **PRIORITY**: Debug and fix GGUF tokenizer implementation
2. Add tokenizer validation tests
3. Re-run this end-to-end test after tokenizer fix
4. Document correct tokenizer usage in user guide
