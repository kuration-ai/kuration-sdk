"""kur — one dispatcher over the kur-* command family.

    kur curate job.json          # same as: kur-curate job.json
    kur --list
    kur completion zsh

Every command remains its own console script; this only makes them discoverable from a
single name. Core commands are named here for ordering and one-line help, and any other
installed `kur-*` console script is discovered through its entry point — so a domain or
plugin package's CLI appears without the core knowing the domain exists.
"""
from __future__ import annotations

import sys

_CORE: dict[str, tuple[str, str]] = {
    # authoring: sample data, scaffold a job, look before running
    "infer": ("kuration.core.cli.infer_schema_cli",
              "Infer a schema and adapter hints from a data sample."),
    "init": ("kuration.core.cli.init_job", "Scaffold a validated job spec from a sample."),
    "preview": ("kuration.core.cli.preview", "Read a bounded sample of a job's sources."),
    "validate": ("kuration.core.cli.validate", "Preflight a job spec without reading data."),
    "explain": ("kuration.core.cli.explain", "Explain a job's resolved plan."),
    # execution
    "curate": ("kuration.core.cli.curate", "Curate a dataset from a job spec."),
    "experiment": ("kuration.core.cli.experiment", "Compare curation arms from a job spec."),
    "report": ("kuration.core.cli.report", "Distribution report for a job's candidates."),
    # artifacts
    "inspect": ("kuration.core.cli.inspect", "Inspect a written dataset artifact."),
    "stats": ("kuration.core.cli.stats", "Distribution statistics for an artifact."),
    "check": ("kuration.core.cli.check", "Evaluate quality gates against an artifact."),
    "diff": ("kuration.core.cli.diff", "Compare two dataset artifacts."),
    "export": ("kuration.core.cli.export", "Export shards as plain array files."),
    "import": ("kuration.core.cli.import_dataset", "Import a raw corpus into Kuration."),
    # environment and maintenance
    "list": ("kuration.core.cli.discovery:list_main", "List installed components."),
    "describe": ("kuration.core.cli.discovery:describe_main", "Describe one component."),
    "plugin-check": ("kuration.core.cli.plugin_check", "Validate installed plugins."),
    "checkpoint": ("kuration.core.cli.checkpoint", "Inspect or clear stage checkpoints."),
    "migrate": ("kuration.core.cli.migrate", "Migrate a job or config document."),
}

_SHELLS = ("bash", "zsh", "fish")


def _discovered() -> dict[str, str]:
    """Installed `kur-*` console scripts, keyed by their subcommand name."""
    from importlib.metadata import entry_points
    found: dict[str, str] = {}
    for entry in entry_points(group="console_scripts"):
        if entry.name.startswith("kur-") and entry.name != "kur":
            found[entry.name[len("kur-"):]] = entry.value
    return found


def commands() -> dict[str, str]:
    """Subcommand -> description, core first, then anything else installed."""
    table = {name: description for name, (_target, description) in _CORE.items()}
    for name in sorted(_discovered()):
        table.setdefault(name, "(provided by an installed plugin)")
    return table


def _resolve(name: str):
    """Return the `main()` callable for one subcommand."""
    from importlib import import_module
    target = _CORE[name][0] if name in _CORE else _discovered().get(name)
    if target is None:
        raise KeyError(name)
    module_name, _, attribute = target.partition(":")
    return getattr(import_module(module_name), attribute or "main")


def _completion(shell: str) -> str:
    names = " ".join(commands())
    if shell == "bash":
        return (f'# kur bash completion: eval "$(kur completion bash)"\n'
                f'_kur_completions() {{\n'
                f'  local words="{names} completion"\n'
                f'  if [ "$COMP_CWORD" -eq 1 ]; then\n'
                f'    COMPREPLY=($(compgen -W "$words" -- "${{COMP_WORDS[1]}}"))\n'
                f'  else\n'
                f'    COMPREPLY=($(compgen -f -- "${{COMP_WORDS[COMP_CWORD]}}"))\n'
                f'  fi\n'
                f'}}\n'
                f'complete -F _kur_completions kur\n')
    if shell == "zsh":
        return (f'# kur zsh completion: eval "$(kur completion zsh)"\n'
                f'_kur() {{\n'
                f'  local -a subcommands\n'
                f'  subcommands=({names} completion)\n'
                f'  if (( CURRENT == 2 )); then\n'
                f'    compadd -- $subcommands\n'
                f'  else\n'
                f'    _files\n'
                f'  fi\n'
                f'}}\n'
                f'compdef _kur kur\n')
    return ("# kur fish completion: kur completion fish | source\n"
            + "".join(f"complete -c kur -n __fish_use_subcommand -a {name} "
                      f"-d '{description}'\n"
                      for name, description in commands().items())
            + "complete -c kur -n __fish_use_subcommand -a completion "
              "-d 'Print a shell completion script.'\n")


def _usage() -> str:
    table = commands()
    width = max(len(name) for name in table)
    lines = ["usage: kur <command> [args...]", "",
             "Each command is also its own console script (kur curate == kur-curate).", "",
             "commands:"]
    lines += [f"  {name:<{width}}  {description}" for name, description in table.items()]
    lines += ["", f"  {'completion':<{width}}  Print a shell completion script "
                  f"({'|'.join(_SHELLS)}).", "",
              "Run 'kur <command> --help' for a command's own options."]
    return "\n".join(lines)


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "help" and len(argv) > 1:
        # `kur help <command>` reads as `kur <command> --help`, including the unknown-command
        # suggestion path below; a bare `kur help` still prints the dispatcher's own usage.
        argv = argv[1:] + ["--help"]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(_usage())
        return
    if argv[0] in {"-V", "--version"}:
        from .. import __version__
        print(__version__)
        return
    if argv[0] in {"--list", "list-commands"}:
        for name, description in commands().items():
            print(f"{name:<14} {description}")
        return
    if argv[0] == "completion":
        shell = argv[1] if len(argv) > 1 else ""
        if shell not in _SHELLS:
            print(f"error: kur completion needs a shell: {', '.join(_SHELLS)}",
                  file=sys.stderr)
            raise SystemExit(1)
        print(_completion(shell), end="")
        return

    name = argv[0]
    try:
        entry = _resolve(name)
    except KeyError:
        import difflib
        close = difflib.get_close_matches(name, commands(), n=1)
        hint = f" Did you mean 'kur {close[0]}'?" if close else ""
        print(f"error: unknown command {name!r}.{hint} Run 'kur --help' for the list.",
              file=sys.stderr)
        raise SystemExit(2)
    except ImportError as e:
        # A discovered `kur-*` console script can name a module that fails to import (most
        # often a missing optional dependency for that plugin). The dispatcher has no
        # subcommand parser of its own here, so it cannot read the delegate's --debug; look
        # for it directly in the remaining argv, same convention every kur-* CLI uses.
        if "--debug" in argv[1:]:
            raise
        print(f"error: command {name!r} failed to load: {type(e).__name__}: {e} | "
              f"hint: install the optional extra the plugin depends on, or pass --debug for "
              f"the full traceback.", file=sys.stderr)
        raise SystemExit(1)
    # The delegate parses argv itself; make its usage line read as the dispatched form.
    sys.argv = [f"kur {name}"] + argv[1:]
    # Under `python -m`, argparse prefers __main__.__spec__ over argv[0] when deriving prog,
    # which would print the dispatcher's module path instead of the command the user ran.
    # Clearing it makes `python -m kuration.core.cli.main <cmd>` and `kur <cmd>` identical.
    dispatcher = sys.modules.get("__main__")
    if dispatcher is not None and getattr(dispatcher, "__spec__", None) is not None:
        dispatcher.__spec__ = None
    entry()


if __name__ == "__main__":
    main()
