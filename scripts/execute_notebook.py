"""Execute one checked-in notebook in a clean kernel for CI validation."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    parser.add_argument("output")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    import nbformat
    from nbclient import NotebookClient

    source = Path(args.notebook).resolve()
    output = Path(args.output).resolve()
    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        allow_errors=False,
    )
    client.execute(cwd=str(Path.cwd()))
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
