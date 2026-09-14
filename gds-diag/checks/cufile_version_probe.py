#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated cuFileGetVersion() probe used by :mod:`cufile_version`.

This helper intentionally has no project imports.  It is executed by its
absolute repository path so gds-diag works directly from a checkout without
installation or PYTHONPATH setup.
"""
from __future__ import annotations

import ctypes
import json
import sys


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(json.dumps({"stage": "arguments", "error": "expected one libcufile path"}))
        return 2

    try:
        lib = ctypes.CDLL(args[0])
    except Exception as exc:
        print(json.dumps({"stage": "load", "error": str(exc)}))
        return 2

    try:
        fn = lib.cuFileGetVersion
        fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
        fn.restype = ctypes.c_int
        value = ctypes.c_int()
        rc = fn(ctypes.byref(value))
        print(json.dumps({"stage": "call", "rc": rc, "version": value.value}))
        return 0
    except Exception as exc:
        print(json.dumps({"stage": "symbol", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
