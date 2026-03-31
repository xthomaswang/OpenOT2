"""CLI entry point: ``python -m webapp``.

Usage::

    # UI-only mode
    python -m webapp --config .config/deck.yaml

    # With robot
    python -m webapp --config .config/deck.yaml --robot 169.254.8.56

    # Reconnect to existing run
    python -m webapp --config .config/deck.yaml --robot 169.254.8.56 --reconnect
"""

from __future__ import annotations

import argparse
import logging

from webapp.app import WebApp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenOT2 Web Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to deck config YAML file",
    )
    parser.add_argument(
        "--robot", type=str, default=None,
        help="OT-2 IP address (omit for UI-only mode)",
    )
    parser.add_argument(
        "--reconnect", action="store_true",
        help="Reconnect to last OT-2 run instead of creating a new one",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Web server port (default: 8000)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./run_data",
        help="Directory for run/event JSON files",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
    )

    if args.config:
        app = WebApp.from_yaml(
            args.config,
            robot_ip=args.robot,
            reconnect=args.reconnect,
            data_dir=args.data_dir,
        )
    else:
        app = WebApp(
            robot_ip=args.robot,
            reconnect=args.reconnect,
            data_dir=args.data_dir,
        )

    app.run(port=args.port)


if __name__ == "__main__":
    main()
