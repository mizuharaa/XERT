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
    from .data import cameras, lerobot_io, map_from_calibration, quality
    from .eval import image_act, replay_real
    add_commands(commands)
    quality.add_command(commands)
    cameras.add_command(commands)
    lerobot_io.add_commands(commands)
    replay_real.add_command(commands)
    map_from_calibration.add_command(commands)
    image_act.add_command(commands)
    args = parser.parse_args()
    if args.command == "quality":
        quality.execute(args)
        return
    if args.command == "render-cameras":
        cameras.execute(args)
        return
    if args.command in ("export-lerobot", "import-lerobot", "to-store"):
        lerobot_io.execute(args)
        return
    if args.command == "map-from-calibration":
        map_from_calibration.execute(args)
        return
    if args.command == "image-act":
        image_act.execute(args)
        return
    if args.command == "replay-real":
        replay_real.execute(args)
        return
    if args.command != "ledger":
        execute(args)
        return
    for row in read_rows():
        print(json.dumps(row, allow_nan=False))


if __name__ == "__main__":
    main()
