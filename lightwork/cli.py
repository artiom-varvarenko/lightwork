"""``lightwork <cmd>``: discovers command modules in ``lightwork/commands/`` (C10).

A command module defines::

    COMMAND = "verify"          # subcommand name
    HELP = "one-line help"
    def configure(parser): ...  # add arguments (argparse)
    def run(args) -> int: ...   # exit code: 0 ok, 1 diff/check failure, 2 usage/environment error
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys

from . import __version__, commands

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


def discover():
    mods = []
    for info in pkgutil.iter_modules(commands.__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{commands.__name__}.{info.name}")
        if hasattr(mod, "COMMAND") and hasattr(mod, "run"):
            mods.append(mod)
    return sorted(mods, key=lambda m: m.COMMAND)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lightwork", description="Projected, model-verified procedures.")
    p.add_argument("--version", action="version", version=f"lightwork {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<command>")
    for mod in discover():
        sp = sub.add_parser(mod.COMMAND, help=getattr(mod, "HELP", ""), description=getattr(mod, "HELP", ""))
        if hasattr(mod, "configure"):
            mod.configure(sp)
        sp.set_defaults(_run=mod.run)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "_run", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return int(args._run(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
