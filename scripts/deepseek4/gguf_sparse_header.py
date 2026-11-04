#!/usr/bin/env python3
"""Materialize a header-only *sparse* local copy of a remote GGUF shard.

usage: gguf_sparse_header.py URL OUTPUT

Downloads only the GGUF header (metadata + tensor infos, up to the aligned data
offset) through HTTP range reads, writes it, and extends the file to the remote
size as a hole. The result has the exact tensor directory and file size of the
real shard but zero tensor payload: suitable for metadata-only tooling such as
`llaminar2 serve --dry-run --explain-placement` or `plan`, never for inference.
"""
import struct, sys, urllib.request

SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Remote:
    def __init__(self, url, chunk=8 << 20):
        self.url, self.chunk, self.buf, self.pos = url, chunk, b"", 0

    def _need(self, n):
        while self.pos + n > len(self.buf):
            start = len(self.buf)
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + self.chunk - 1}"})
            data = urllib.request.urlopen(req).read()
            if not data:
                raise EOFError
            self.buf += data

    def read(self, n):
        self._need(n)
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self): return struct.unpack("<I", self.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.read(8))[0]
    def s(self): return self.read(self.u64()).decode("utf-8", "replace")


def skip_value(r, t):
    if t in SCALAR:
        return r.read(SCALAR[t])
    if t == 8:
        return r.s()
    if t == 9:
        et, n = r.u32(), r.u64()
        if et in SCALAR:
            r.read(SCALAR[et] * n)
        else:
            for _ in range(n):
                skip_value(r, et)
        return None
    raise ValueError(f"unknown GGUF value type {t}")


def main():
    url, out = sys.argv[1], sys.argv[2]
    head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
    size = int(head.headers["Content-Length"])
    r = Remote(url)
    assert r.read(4) == b"GGUF"
    r.u32()
    n_tensors, n_kv = r.u64(), r.u64()
    alignment = 32
    for _ in range(n_kv):
        key, t = r.s(), r.u32()
        if key == "general.alignment":
            alignment = struct.unpack("<I", r.read(4))[0]
        else:
            skip_value(r, t)
    for _ in range(n_tensors):
        r.s()
        nd = r.u32()
        r.read(8 * nd + 4 + 8)
    data_offset = (r.pos + alignment - 1) // alignment * alignment
    r._need(data_offset - r.pos) if data_offset > r.pos else None
    with open(out, "wb") as f:
        f.write(r.buf[:data_offset])
        f.truncate(size)
    print(f"{out}: header {data_offset} bytes, sparse size {size}")


if __name__ == "__main__":
    main()
