// Mechanical per-source-format prepared encoding of one tree (no rotation): preparedEncodingForSource(source), source.is_asymmetric.
#include "kernels/cpu/gemm/CPUNativeQuantisedEncoding.h"
#include <cstdio>
using namespace llaminar2;
#define F(name) {#name, native_quantised_formats::name}
int main() {
    struct { const char *name; NativeQuantisedFormatInfo info; } formats[] = {
        F(IQ4_NL), F(Q8_0), F(Q8_1), F(Q4_0), F(Q4_1), F(Q5_0), F(Q5_1), F(Q6_K), F(Q2_K), F(Q5_K), F(Q3_K),
        F(Q4_K), F(Q8_K), F(IQ4_XS), F(IQ2_XXS), F(IQ2_XS), F(IQ3_XXS), F(IQ2_S), F(IQ3_S), F(IQ1_S), F(IQ1_M),
        F(MXFP4)};
    for (const auto &f : formats) {
        const auto encoding = cpu::native_quantised::preparedEncodingForSource(f.info);
        std::printf("%s %d %d %d %d\n", f.name, int(f.info.codebook_id), int(f.info.is_superblock),
                    int(static_cast<std::uint8_t>(encoding)), int(f.info.is_asymmetric));
    }
}
