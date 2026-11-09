// Byte-identity harness (argv[2] = start position; negative = exact n_past; rows 1..64 contiguous): same calls against the base (persistent-state) and new (angle-planes) RoPE TU.
#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include "tensors/FP16Utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <omp.h>
#ifdef NEW_API
#include "utils/RoPEAngleTestStorage.h"
#endif
using namespace llaminar2;
static FILE *out;
template <class T> static void dump(const std::vector<T> &v) { fwrite(v.data(), sizeof(T), v.size(), out); }
template <class W> static void seed(std::vector<W> &b, int salt) {
    for (size_t i = 0; i < b.size(); ++i) {
        if constexpr (std::is_same_v<W, float>) b[i] = float((int(i % 251) * 73 + salt) % 251 - 125) * 0.03125f;
        else if constexpr (std::is_same_v<W, uint16_t>) b[i] = uint16_t(0x3c00 + (int(i % 251) * 73 + salt) % 512);
        else { b[i].d = fp32_to_fp16(float(i % 13 + 1) * 0.001234f); int s = 0;
               for (int j = 0; j < 32; ++j) { b[i].qs[j] = int8_t((int(i % 251) * 13 + j * 73 + salt) % 256 - 128); s += b[i].qs[j]; }
               b[i].sum_qs = int16_t(s); }
    }
}
int main(int argc, char **argv) {
    out = fopen(argv[1], "wb");
    omp_set_num_threads(4);
    const int qh = 3, kh = 1;
    // BASE_NO_STATE: base ordinary one-row calls use the no-state API instead of RoPEPersistentState.
    [[maybe_unused]] const bool base_no_state = std::getenv("BASE_NO_STATE") != nullptr;
    for (int dim : {64, 128, 256})
    for (int rows = 1; rows <= 64; ++rows) {
        const int start = argc > 2 ? std::atoi(argv[2]) + (std::atoi(argv[2]) > 0 ? rows : 0) : 17231 + rows;
        std::vector<int> pos(rows); for (int r = 0; r < rows; ++r) pos[r] = r % 5 == 3 ? -1 : start + 3 * r;
#ifdef NEW_API
        test::RoPEAngleTestStorage st(rows, dim / 2);
#define ANG(v) , st.view(rows, v)
#define ANG1(v) , st1.view(1, v)
        test::RoPEAngleTestStorage st1(1, dim / 2);
#endif
        { std::vector<float> q(rows*qh*dim), k(rows*kh*dim); seed(q,1); seed(k,2);
#ifdef NEW_API
          primitives::apply_rope_vectorized(q.data(), k.data(), rows, dim, qh, kh, start, 10000.f ANG(dim/2));
#else
          primitives::RoPEPersistentState s; primitives::apply_rope_vectorized(q.data(), k.data(), rows, dim, qh, kh, start, 10000.f, rows == 1 && !base_no_state ? &s : nullptr);
#endif
          dump(q); dump(k); }
        { std::vector<float> q(rows*qh*dim), k(rows*kh*dim); seed(q,3); seed(k,4);
#ifdef NEW_API
          primitives::apply_rope_partial(q.data(), k.data(), rows, dim, dim/4, qh, kh, start, 10000.f ANG(dim/8));
#else
          primitives::apply_rope_partial(q.data(), k.data(), rows, dim, dim/4, qh, kh, start, 10000.f);
#endif
          dump(q); dump(k); }
        for (int fmt = 0; fmt < 2; ++fmt) { std::vector<uint16_t> q(rows*qh*dim), k(rows*kh*dim); seed(q,5+fmt); seed(k,7+fmt);
#ifdef NEW_API
          if (fmt) primitives::apply_rope_bf16(q.data(), k.data(), rows, dim, qh, kh, start, 10000.f ANG(dim/2));
          else primitives::apply_rope_fp16(q.data(), k.data(), rows, dim, qh, kh, start, 10000.f ANG(dim/2));
#else
          primitives::RoPEPersistentState s;
          if (fmt) primitives::apply_rope_bf16(q.data(), k.data(), rows, dim, qh, kh, start, 10000.f, rows == 1 && !base_no_state ? &s : nullptr);
          else primitives::apply_rope_fp16(q.data(), k.data(), rows, dim, qh, kh, start, 10000.f, rows == 1 && !base_no_state ? &s : nullptr);
#endif
          dump(q); dump(k); }
        // grouped verifier rows, every format, plus partial FP32
        for (int rot : {0, dim / 4}) { std::vector<float> q(rows*qh*dim), k(rows*kh*dim); seed(q,9); seed(k,10);
#ifdef NEW_API
          primitives::apply_rope_decode_equivalent_rows(q.data(), k.data(), pos.data(), rows, dim, rot, qh, kh, 0, 10000.f ANG((rot ? rot : dim)/2));
#else
          primitives::RoPEPersistentState s; primitives::apply_rope_decode_equivalent_rows(q.data(), k.data(), pos.data(), rows, dim, rot, qh, kh, 0, 10000.f, &s);
#endif
          dump(q); dump(k); }
        for (int fmt = 0; fmt < 2; ++fmt) { std::vector<uint16_t> q(rows*qh*dim), k(rows*kh*dim); seed(q,11+fmt); seed(k,13+fmt);
#ifdef NEW_API
          if (fmt) primitives::apply_rope_bf16_decode_equivalent_rows(q.data(), k.data(), pos.data(), rows, dim, qh, kh, 0, 10000.f ANG(dim/2));
          else primitives::apply_rope_fp16_decode_equivalent_rows(q.data(), k.data(), pos.data(), rows, dim, qh, kh, 0, 10000.f ANG(dim/2));
#else
          primitives::RoPEPersistentState s;
          if (fmt) primitives::apply_rope_bf16_decode_equivalent_rows(q.data(), k.data(), pos.data(), rows, dim, qh, kh, 0, 10000.f, &s);
          else primitives::apply_rope_fp16_decode_equivalent_rows(q.data(), k.data(), pos.data(), rows, dim, qh, kh, 0, 10000.f, &s);
#endif
          dump(q); dump(k); }
        { std::vector<Q8_1Block> q(rows*qh*dim/32), k(rows*kh*dim/32); seed(q,15); seed(k,16);
#ifdef NEW_API
          primitives::apply_rope_q8_1_decode_equivalent_rows(q.data(), k.data(), pos.data(), rows, qh, kh, dim, 0, 10000.f ANG(dim/2));
#else
          primitives::RoPEPersistentState s; primitives::apply_rope_q8_1_decode_equivalent_rows(q.data(), k.data(), pos.data(), rows, qh, kh, dim, 0, 10000.f, &s);
#endif
          dump(q); dump(k); }
        // Q8 integer. SKIP_Q8_ONE_ROW_NEGATIVE omits only the one-row call at a negative start: the base
        // one-row program there is the fixed stale-TLS defect. Every other Q8 integer cell is compared.
        if (!(std::getenv("SKIP_Q8_ONE_ROW_NEGATIVE") && rows == 1 && start < 0)) {
          std::vector<int> pp(rows); for (int r = 0; r < rows; ++r) pp[r] = start + 2 * r;
          std::vector<Q8_1Block> q(rows*qh*dim/32), k(rows*kh*dim/32); seed(q,17); seed(k,18);
#ifdef NEW_API
          primitives::apply_rope_q8_1_integer(q.data(), k.data(), pp.data(), rows, qh, kh, dim, 10000.f ANG(dim/2));
#else
          primitives::RoPEPersistentState s; primitives::apply_rope_q8_1_integer(q.data(), k.data(), pp.data(), rows, qh, kh, dim, 10000.f, rows == 1 ? &s : nullptr);
#endif
          dump(q); dump(k); }
        // one-row decode at many positions (memo vs fresh row function)
        if (rows == 1) for (int p : {0, 1, 5, 4095, 17231, 131071}) {
          std::vector<float> q(qh*dim), k(kh*dim); seed(q,19); seed(k,20);
#ifdef NEW_API
          primitives::apply_rope_vectorized(q.data(), k.data(), 1, dim, qh, kh, p, 10000.f ANG1(dim/2));
#else
          static primitives::RoPEPersistentState s; primitives::apply_rope_vectorized(q.data(), k.data(), 1, dim, qh, kh, p, 10000.f, &s);
#endif
          dump(q); dump(k); }
    }
    fclose(out);
    return 0;
}
