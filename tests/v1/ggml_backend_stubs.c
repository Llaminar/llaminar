// Minimal stub implementations to satisfy ggml backend symbol references
// required only for quantize/dequant golden test. These provide no-op
// functionality sufficient for isolated quantization routines.

#include <stdint.h>
#include <stddef.h>

// Forward declare ggml_tensor to avoid pulling full backend complexity
struct ggml_tensor;

// Critical section stubs
void ggml_critical_section_start(void) {}
void ggml_critical_section_end(void) {}

// Backend tensor memory ops (no-op for golden harness scope)
void ggml_backend_tensor_memset(struct ggml_tensor *t, uint8_t value)
{
    (void)t;
    (void)value;
}
void ggml_backend_tensor_set(struct ggml_tensor *t, const void *data, size_t offset, size_t size)
{
    (void)t;
    (void)data;
    (void)offset;
    (void)size;
}
