#!/usr/bin/env python3
"""List GGUF tensor types of remote (HF) GGUF shards using HTTP range reads only.

usage: gguf_remote_types.py URL [URL...]
Prints metadata scalars (no big arrays) and a per-type / per-tensor-family summary.
"""
import collections, json, re, struct, sys, urllib.request

GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
              17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
              24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
              35: "TQ2_0", 39: "MXFP4", 40: "NVFP4"}


class Remote:
    def __init__(self, url, chunk=4 << 20):
        self.url, self.chunk, self.buf, self.base, self.pos = url, chunk, b"", 0, 0

    def _need(self, n):
        while self.pos + n > self.base + len(self.buf):
            start = self.base + len(self.buf)
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + self.chunk - 1}"})
            data = urllib.request.urlopen(req).read()
            if not data:
                raise EOFError
            self.buf += data

    def read(self, n):
        self._need(n)
        off = self.pos - self.base
        self.pos += n
        return self.buf[off:off + n]

    def u32(self): return struct.unpack("<I", self.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.read(8))[0]
    def s(self): return self.read(self.u64()).decode("utf-8", "replace")


SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4),
          7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


def val(r, t):
    if t in SCALAR:
        f, n = SCALAR[t]
        return struct.unpack(f, r.read(n))[0]
    if t == 8:
        return r.s()
    if t == 9:
        et, n = r.u32(), r.u64()
        if et in SCALAR and n > 256:
            r.read(SCALAR[et][1] * n)
            return f"<array {n}>"
        items = [val(r, et) for _ in range(n)]
        return items if n <= 64 else f"<array {n}>"
    raise ValueError(t)


def main():
    for url in sys.argv[1:]:
        r = Remote(url)
        assert r.read(4) == b"GGUF"
        ver, nt, nkv = r.u32(), r.u64(), r.u64()
        print(f"== {url.rsplit('/', 1)[-1]} v{ver} tensors={nt} kv={nkv}")
        for _ in range(nkv):
            k, t = r.s(), r.u32()
            v = val(r, t)
            if "tokenizer.ggml." in k and k.split(".")[-1] in ("tokens", "merges", "token_type", "scores"):
                continue
            print(f"   {k} = {v}")
        types, fam, total = collections.Counter(), collections.defaultdict(collections.Counter), 0
        for _ in range(nt):
            name = r.s()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            ty, _off = r.u32(), r.u64()
            tn = GGML_TYPES.get(ty, str(ty))
            types[tn] += 1
            fam[re.sub(r"blk\.\d+\.", "blk.N.", name)][tn] += 1
        print("   types:", dict(types))
        for f, c in sorted(fam.items()):
            print(f"   {f}: {dict(c)}")


if __name__ == "__main__":
    main()
