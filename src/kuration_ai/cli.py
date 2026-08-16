"""Command-line interface for Kuration AI.

Two subcommands:

    kai list             list all available Operations (built-in + user-registered),
                         annotated with the extras each one needs at runtime.
    kai describe <name>  show one Operation's docstring and parameter contract.

The `kai` entry point is wired via `[project.scripts]` in pyproject.toml.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import textwrap
from typing import Any


def _extras_annotation(cls: type) -> str:
    """Return a suffix like '  needs kuration-ai[X]' if the class declares EXTRAS."""
    extras = getattr(cls, "EXTRAS", ())
    if not extras:
        return ""
    return f"  needs kuration-ai[{','.join(extras)}]"


def _one_line_doc(cls: type) -> str:
    """Return the first non-empty line of the class docstring."""
    doc = (cls.__doc__ or "").strip()
    if not doc:
        return ""
    return doc.splitlines()[0].strip()


def _cmd_list(_args: Any) -> int:
    from kuration_ai import operations as _catalog
    from kuration_ai.operation import _USER_REGISTRY

    builtin_names = sorted(
        n for n in dir(_catalog)
        if not n.startswith("_") and n[0].isupper()
    )
    user_names = sorted(_USER_REGISTRY.keys())

    total = len(builtin_names) + len(user_names)
    print(
        f"Kuration AI operations: {len(builtin_names)} built-in, "
        f"{len(user_names)} user-registered ({total} total)"
    )
    print()
    print("  Built-in:")
    for name in builtin_names:
        cls = getattr(_catalog, name)
        print(f"    {name:<20}  {_one_line_doc(cls)}{_extras_annotation(cls)}")
    if user_names:
        print()
        print("  User-registered (via kuration_ai.register):")
        for name in user_names:
            cls = _USER_REGISTRY[name]
            print(f"    {name:<20}  {_one_line_doc(cls)}{_extras_annotation(cls)}")
    return 0


def _cmd_describe(args: Any) -> int:
    from kuration_ai.operation import _lookup_class, _available_operations

    name = args.name
    cls = _lookup_class(name)
    if cls is None:
        available = _available_operations()
        print(
            f"kai describe: unknown operation '{name}'. "
            f"Available: {', '.join(available)}.",
            file=sys.stderr,
        )
        return 1

    print(name)
    doc = (cls.__doc__ or "").strip()
    if doc:
        print(textwrap.indent(textwrap.dedent(doc), "  "))
    extras = getattr(cls, "EXTRAS", ())
    if extras:
        print()
        print(f"  Needs kuration-ai[{','.join(extras)}]")
    if dataclasses.is_dataclass(cls):
        print()
        print("  Parameters:")
        for f in dataclasses.fields(cls):
            has_default = (
                f.default is not dataclasses.MISSING
                or f.default_factory is not dataclasses.MISSING
            )
            marker = "optional" if has_default else "required"
            print(f"    {f.name:<20}  {f.type}  ({marker})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kai",
        description="Kuration AI CLI: list and inspect available Operations.",
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="list all available Operations")
    p_desc = sub.add_parser(
        "describe",
        help="show one Operation's docstring and parameter contract",
    )
    p_desc.add_argument("name", help="Operation class name (e.g. 'BalancedSelector')")

    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 1
    if args.cmd == "list":
        return _cmd_list(args)
    if args.cmd == "describe":
        return _cmd_describe(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
