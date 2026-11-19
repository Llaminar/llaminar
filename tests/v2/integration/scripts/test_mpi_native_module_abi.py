#!/usr/bin/env python3
"""Load the actual native C/C++ MPI component and validate its public ABI.

This focused model/device-free preflight needs the installed native runtime,
not a compiler SDK. ORTE table layout is checked by the genuine C translation
unit; this independent public-contract check never guesses internal structures
or initializes MPI. Native daemon lifecycle tests separately exercise its hook.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path


class NativeModuleContract(ctypes.Structure):
    """Independent reader for the versioned public module descriptor, not ORTE internals."""
    _fields_ = [("magic", ctypes.c_uint64), ("protocol", ctypes.c_uint32),
                ("major", ctypes.c_uint32), ("minor", ctypes.c_uint32),
                ("release", ctypes.c_uint32), ("module_bytes", ctypes.c_uint32)]


def main() -> None:
    """Validate the actual loader symbol/descriptor and report immutable module identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--mpi-library", type=Path, required=True)
    args = parser.parse_args()
    path = args.module.resolve(strict=True)
    # The component needs ORTE/OPAL, not necessarily libmpi under --as-needed.
    # Inspect MPI state through the exact CMake-selected public library, never
    # a second runtime search or an assumed transitive component dependency.
    mpi = ctypes.CDLL(str(args.mpi_library.resolve(strict=True)), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    library = ctypes.CDLL(str(path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    query = library.llaminar_mpi_native_contract_v1
    query.argtypes = []
    query.restype = ctypes.POINTER(NativeModuleContract)
    pointer = query()
    if not pointer:
        raise RuntimeError("native module rejects the actually loaded OpenMPI ABI")
    contract = pointer.contents
    observed = (contract.magic, contract.protocol, contract.major,
                contract.minor, contract.release, contract.module_bytes)
    expected = (0x4C4D50494E415431, 1, 4, 1, 6, 10 * ctypes.sizeof(ctypes.c_void_p))
    if observed != expected:
        raise RuntimeError(f"native C/C++ module contract mismatch: {observed} != {expected}")
    # Presence only: never reconstruct the SDK's private component structure.
    ctypes.c_ubyte.in_dll(library, "mca_schizo_llaminar_component")
    initialized = ctypes.c_int(-1)
    initialized_query = mpi.MPI_Initialized
    initialized_query.argtypes = [ctypes.POINTER(ctypes.c_int)]
    initialized_query.restype = ctypes.c_int
    if initialized_query(ctypes.byref(initialized)) != 0 or initialized.value != 0:
        raise RuntimeError("native module ABI inspection unexpectedly entered an MPI session")
    print(json.dumps({"module": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "protocol": contract.protocol, "openmpi": "4.1.6",
                      "module_bytes": contract.module_bytes, "mpi_initialized": False}))


if __name__ == "__main__":
    main()
