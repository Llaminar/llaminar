/** @file partition-probe.hip
 * @brief Compile-only distinct pressure witnesses spanning parallel full-LTO partitions.
 * These kernels are never launched. Every emitted kernel must retain its own
 * final stack allocation record; private and spilling slots remain distinct.
 */
#include <hip/hip_runtime.h>
#define PRESSURE_PROBE(N) \
extern "C" __global__ __launch_bounds__(1024) void partition_probe_##N(const float* in,float* out) { \
 float values[128]; \
 _Pragma("unroll") \
 for(int i=0;i<128;++i) values[i]=in[i*1024+threadIdx.x+(N)]; \
 asm volatile("" ::: "memory"); \
 _Pragma("unroll") \
 for(int i=0;i<128;++i) out[i*1024+threadIdx.x+(N)]=values[i]; \
}
PRESSURE_PROBE(0) PRESSURE_PROBE(1) PRESSURE_PROBE(2) PRESSURE_PROBE(3)
PRESSURE_PROBE(4) PRESSURE_PROBE(5) PRESSURE_PROBE(6) PRESSURE_PROBE(7)
PRESSURE_PROBE(8) PRESSURE_PROBE(9) PRESSURE_PROBE(10) PRESSURE_PROBE(11)
PRESSURE_PROBE(12) PRESSURE_PROBE(13) PRESSURE_PROBE(14) PRESSURE_PROBE(15)
