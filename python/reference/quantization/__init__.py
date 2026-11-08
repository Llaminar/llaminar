"""Exact quantized-operand references for exported model arithmetic.

These modules preserve activation quantization and arithmetic policy rather
than replacing a quantized matrix operation with dequantized FP32 weights.
They are independent numerical oracles, not production kernel dispatch.
"""
