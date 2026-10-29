"""Read the results ledger without importing the simulator or learner."""
import argparse
import json

from .eval.ledger import read_rows


def main():
    parser = argparse.ArgumentParser(description="XERT experiment evidence")
    commands = parser.add_subparsers(dest="command", required=True)
    ledger = commands.add_parser("ledger")
    ledger.add_subparsers(dest="operation", required=True).add_parser("show")
    from .runtime.cli import add_commands, execute
    from .data import quality
    add_commands(commands)
    quality.add_command(commands)
    args = parser.parse_args()
    if args.command == "quality":
        quality.execute(args)
        return
    if args.command != "ledger":
        execute(args)
        return
    for row in read_rows():
        print(json.dumps(row, allow_nan=False))


if __name__ == "__main__":
    main()
