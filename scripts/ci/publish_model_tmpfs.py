#!/usr/bin/env python3
"""Publish one owned private model tmpfs in Docker's host mount namespace.

Called through a privileged, short-lived nsenter helper by docker_paths.py.
Linux open_tree creates a detached bind mount (not a copy of its pages), and
move_mount attaches it to the narrowly named export directory. Unlike the old
mount(MS_BIND) API this safely crosses a mount-namespace boundary. See
https://man7.org/linux/man-pages/man2/open_tree.2.html and move_mount(2).
Nothing is unmounted or deleted automatically; the export pins the cache until
explicit host cleanup or reboot. A mismatched existing export fails closed.
"""
from __future__ import annotations

import ctypes
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys


def publish(source: Path, destination: Path) -> None:
    """Attach exactly one authenticated tmpfs root without replacing live data."""
    if not re.fullmatch(r"/proc/[1-9][0-9]*/root/.+", str(source)):
        raise ValueError("source must be an explicit containing-container root path")
    if not re.fullmatch(r"/mnt/llaminar-docker-tmpfs/[0-9a-f]{64}/[0-9a-f]{16}", str(destination)):
        raise ValueError("destination is outside the owned Docker tmpfs export namespace")
    if subprocess.check_output(["stat", "-f", "-c", "%T", str(source)], text=True).strip() != "tmpfs":
        raise ValueError("refusing to publish non-tmpfs model storage")
    expected = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.parent.resolve() != destination.parent:
        raise ValueError("export parents must not be symlinks")
    with destination.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.ismount(destination):
            destination.mkdir(exist_ok=True, mode=0o700)
            if destination.is_symlink() or any(destination.iterdir()):
                raise ValueError("refusing to hide existing export-directory data")
            libc = ctypes.CDLL(None, use_errno=True)
            libc.open_tree.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
            libc.open_tree.restype = ctypes.c_int
            libc.move_mount.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                       ctypes.c_char_p, ctypes.c_uint)
            libc.move_mount.restype = ctypes.c_int
            # Linux UAPI: AT_FDCWD=-100, OPEN_TREE_CLONE=1 and
            # MOVE_MOUNT_F_EMPTY_PATH=4. libc supplies architecture dispatch.
            # Linux requires cloning in the source mount namespace. Retain
            # directory/namespace handles, clone there, and return to the host
            # before attaching. No process root or global mount is changed.
            pid = str(source).split("/")[2]
            host_namespace = os.open("/proc/self/ns/mnt", os.O_RDONLY)
            source_namespace = os.open(f"/proc/{pid}/ns/mnt", os.O_RDONLY)
            source_root = os.open(f"/proc/{pid}/root", os.O_PATH | os.O_DIRECTORY)
            try:
                os.setns(source_namespace, 0)
                tree = libc.open_tree(source_root, os.fsencode(str(source).split("/root/", 1)[1]),
                                      1 | os.O_CLOEXEC)
                clone_errno = ctypes.get_errno()
            finally:
                os.setns(host_namespace, 0)
                os.close(source_root)
                os.close(source_namespace)
                os.close(host_namespace)
            if tree < 0:
                raise OSError(clone_errno, "open_tree model tmpfs")
            try:
                if libc.move_mount(tree, b"", -100, os.fsencode(destination), 4) != 0:
                    raise OSError(ctypes.get_errno(), "move_mount model tmpfs")
            finally:
                os.close(tree)
        actual = destination.stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("existing Docker tmpfs export points to a different cache")


if __name__ == "__main__":
    publish(Path(sys.argv[1]), Path(sys.argv[2]))
