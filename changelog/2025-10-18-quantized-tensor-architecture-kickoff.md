# 2025-10-18 Quantized Tensor Architecture Kickoff

## Summary
Initial scaffolding for migrating from full FP32 weights to on-the-fly quantized execution:
- Created feature branch `feature/quantized-tensors`.
- Authored `docs/quantized_tensor_architecture.md` detailing phased plan (7 phases), data structures, kernel strategy, MPI considerations, and testing roadmap.
- Added skeleton `QuantizedTensor` type and associated `QuantFormat`, `QuantBlockDescriptor`, `QuantStorageLayout` in `TensorFactory.h`.
- Implemented placeholder factory helpers and decode stubs in `TensorFactory.cpp` (`create_quantized`, `is_quantized`, `to_quantized`).
- No functional runtime changes yet; decode methods return zeroed buffers (will be replaced with real block unpack logic).

## Files Added / Modified
| File | Change | Purpose |
|------|--------|---------|
| `docs/quantized_tensor_architecture.md` | Added | Architectural blueprint & intern-friendly guide |
| `src/tensors/TensorFactory.h` | Modified | Insert quantization type definitions & skeleton class |
| `src/tensors/TensorFactory.cpp` | Modified | Add stub decode and creation helpers |
| `changelog/2025-10-18-quantized-tensor-architecture-kickoff.md` | Added | Session summary & next steps |

## Design Highlights
- Non-invasive: New quant types coexist with existing FP32 `SimpleTensor` / `COSMATensor`.
- Dequant happens inside future kernels (Phase 3) rather than at model load (current ModelLoader path unchanged yet).
- Provides clear phased rollout: infra → loader integration → linear kernel → attention → activation precision → distributed optimization → cleanup.
- Debug/verification strategy outlined with env flags (`LLAMINAR_QUANT_ENABLE`, etc.).

## Next Implementation Targets
1. Integrate ModelLoader flag (`LLAMINAR_LOAD_QUANTIZED`) to return `QuantizedTensor` for matmul weights instead of FP32.
2. Implement real block decode for Q4_0 & Q8_0 in `QuantizedTensor::decodeBlock` (reuse `QuantDequant.h`).
3. Prototype `QuantLinearKernel` (Q8_0 path first) + unit test `QuantLinearKernelParityTest`.
4. Parity test comparing quant path vs existing FP32 for a tiny synthetic layer (8×32 × 32×64).
5. Introduce environment guard defaults (quant disabled unless enabled explicitly).

## Risk & Mitigation
- Performance uncertainty: Begin with correctness; profile after kernel exists.
- Memory alignment: Will add 64-byte aligned scratch buffers in Phase 3.
- MPI path divergence: Quantization limited to local linear kernels initially.

## Notes
This is a scaffolding milestone—no production behavior change. Parity and performance unaffected until Phase 2/3.

## Author
David Sanftenberg
