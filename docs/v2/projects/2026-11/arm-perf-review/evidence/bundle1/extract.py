"""Split ctest --output-on-failure logs into per-test blocks and report failure signatures."""
import re, sys, json
STATUS = re.compile(r"^\s*\d+/\d+ Test\s+#\d+: (\S+) \.+\s*(\*{0,3}\S.*?)\s+[\d.]+ sec")
def blocks(path):
    cur, out = None, {}
    for line in open(path, errors="replace"):
        m = STATUS.match(line)
        if m:
            cur = m.group(1); out[cur] = {"status": m.group(2), "lines": []}; continue
        if cur: out[cur]["lines"].append(line.rstrip("\n"))
    return out
SIG = [
 ("mmap-window-4K", r"page-aligned and nonzero"),
 ("NUMAAlloc-revoke", r"Could not revoke"),
 ("NUMAAlloc-firsttouch", r"First-touch certification|Invalid exact page range|first-touch authority"),
 ("CPUBackend-mbind", r"\[CPUBackend\] mbind"),
 ("MPI-bind-memory", r"failed to bind memory"),
 ("numa-unavailable", r"NUMA policy APIs are unavailable|requires observed NUMA|has no physical CPU set"),
 ("set_mempolicy", r"set_mempolicy|ScopedThreadNodePlacement"),
 ("cxx-exception", r"C\+\+ exception with description|what\(\)"),
 ("expected", r"Expected|Value of:|Which is:"),
 ("timeout", r"Timeout"),
]
def sigs(lines):
    t = "\n".join(lines); hit = [k for k, p in SIG if re.search(p, t)]
    first = [l.strip() for l in lines if re.search(r"exception with description|failed to bind memory|page-aligned|Could not revoke|\[CPUBackend\] mbind|Expected|Which is|Value of|FAIL:|ERROR:|Error:|AssertionError|Failure", l)]
    return hit, first[:6]
if __name__ == "__main__":
    path, names = sys.argv[1], set(sys.argv[2:])
    b = blocks(path)
    res = {}
    for n, v in b.items():
        if "Passed" in v["status"] and n not in names: continue
        if names and n not in names: continue
        h, f = sigs(v["lines"]); res[n] = {"status": v["status"], "sigs": h, "first": f}
    json.dump(res, sys.stdout, indent=1)
