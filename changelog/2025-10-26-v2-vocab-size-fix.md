# V2 Vocab Size Fix

**Date**: October 26, 2025  
**Component**: V2 ModelLoader  
**Issue**: `vocab_size_` = 0 causing token validation to fail  
**Status**: ✅ **FIXED**

## Problem

E2E tests were failing with:
```
Invalid token at position 0: value 151644 out of range [0, 0)
```

Root cause: `model_.vocab_size` was 0 because the GGUF file doesn't have `tokenizer.ggml.token_count` or `tokenizer.ggml.tokens.length` metadata keys. Instead, it has `tokenizer.ggml.tokens` which is an **array** of strings.

## Solution

Modified `GGUFValue` struct and `ModelLoader` to properly extract array length:

### Changes Made

1. **Added `array_length` field to `GGUFValue`** (`src/v2/loaders/ModelLoader.h`):
   ```cpp
   struct GGUFValue {
       GGUFValueType type;
       std::vector<uint8_t> data;
       uint64_t array_length = 0;  // For ARRAY type, stores number of elements
       
       uint64_t asArrayLength() const { return array_length; }
   };
   ```

2. **Store array length during parsing** (`src/v2/loaders/ModelLoader.cpp:673`):
   ```cpp
   value.type = GGUFValueType::ARRAY;
   value.array_length = array_len;  // Store array length
   ```

3. **Extract vocab_size from tokens array** (`src/v2/loaders/ModelLoader.cpp:994-999`):
   ```cpp
   // Fallback: Read array length from tokenizer.ggml.tokens (most common case)
   if (model_.vocab_size == 0) {
       auto it = model_.metadata.find("tokenizer.ggml.tokens");
       if (it != model_.metadata.end() && it->second.type == GGUFValueType::ARRAY) {
           model_.vocab_size = it->second.asArrayLength();
           LOG_INFO("[ModelLoader] Extracted vocab_size from tokenizer.ggml.tokens array: " << model_.vocab_size);
       }
   }
   ```

## Test Results

✅ **Model loads correctly**: `vocab_size` now populates to 151936  
✅ **Token validation passes**: BOS token (151644) now within valid range  
❌ **E2E test still fails**: Unrelated tensor shape mismatch (pre-existing bug)

## Next Issue

The E2E test now progresses past the vocab_size bug but encounters a different issue:
```
[TENSOR VALIDATION ERROR] after_attn_norm: Shape mismatch
  Expected: hidden[1,896] [1, 896]
  Actual:   [2048, 896]
```

This is a **buffer allocation bug** where a tensor is allocated with `max_seq_len` instead of `actual_seq_len`. This was already present but hidden by the vocab_size issue.

## Files Modified

- `src/v2/loaders/ModelLoader.h` - Added `array_length` field to `GGUFValue`
- `src/v2/loaders/ModelLoader.cpp` - Store array length and extract vocab_size from tokens array

## Commit Message

```
fix(v2): Extract vocab_size from tokenizer.ggml.tokens array

GGUF files don't always have explicit token_count metadata. Extract
vocab_size from the length of tokenizer.ggml.tokens array instead.

- Add array_length field to GGUFValue struct
- Store array length when parsing ARRAY type metadata
- Fallback to tokens array length if token_count not found
- Fixes E2E test failure: "Invalid token at position 0: value 151644 out of range [0, 0)"

Tested with Qwen 2.5 0.5B Q4_0: vocab_size correctly extracted as 151936
```

## References

- V1 approach: `src/v1/ModelLoader.cpp:1793` - reads `tokenizer.ggml.tokens` into `token_list`
- GGUF spec: Arrays store [elem_type(4B)][count(8B)][data...] - count must be extracted during parsing
