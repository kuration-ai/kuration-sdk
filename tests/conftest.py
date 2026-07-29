"""Shared test helpers.

`run_cli` invokes a CLI entry point in the test process. The CLIs are ordinary functions, so
calling them directly is faster than spawning an interpreter and — unlike a subprocess — the
executed lines are visible to coverage. Tests that need the behavior only a real process shows
(installed console scripts, `--debug` tracebacks) still use `subprocess`.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from importlib import import_module
from typing import NamedTuple


class CliResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


def run_cli(module: str, *args, expect: int | None = 0) -> CliResult:
    """Run `module`'s `main()` (or `module:attr`) with `args`, returning its output.

    Interpreter exit handling is emulated: `SystemExit(None)` is success, an int is the exit
    code, and a string is printed to stderr with exit code 1.
    """
    module_name, _, attribute = module.partition(":")
    entry = getattr(import_module(module_name), attribute or "main")
    argv = [module_name.rsplit(".", 1)[-1], *(str(arg) for arg in args)]
    out, err = io.StringIO(), io.StringIO()
    code = 0
    real_argv, main_module = sys.argv, sys.modules.get("__main__")
    real_spec = getattr(main_module, "__spec__", None)
    try:
        sys.argv = argv
        with redirect_stdout(out), redirect_stderr(err):
            try:
                entry()
            except SystemExit as exit_signal:
                if exit_signal.code is None:
                    code = 0
                elif isinstance(exit_signal.code, int):
                    code = exit_signal.code
                else:
                    print(exit_signal.code, file=sys.stderr)
                    code = 1
    finally:
        sys.argv = real_argv
        # The dispatcher clears __main__.__spec__ so argparse reads argv[0]; put it back.
        if main_module is not None:
            main_module.__spec__ = real_spec
    result = CliResult(code, out.getvalue(), err.getvalue())
    if expect is not None:
        assert result.returncode == expect, (
            f"{module} {args} -> exit {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result
