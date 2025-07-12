#!/usr/bin/env python3
"""
Entry point for running deeplin modules as executables.
Usage:
  python -m deeplin.inference_engine.hexin_server --port 9999
  python -m deeplin --module hexin_server --port 9999
"""

import sys
import argparse
import importlib

def main():
    """Main entry point for deeplin modules"""
    if sys.argv[0].endswith('hexin_server.py') or 'hexin_server' in sys.argv[0]:
        # Running hexin_server specifically
        from deeplin.inference_engine.hexin_server import main as hexin_main
        hexin_main()
    else:
        # Generic module runner
        parser = argparse.ArgumentParser(
            description="DeepLin module runner",
            prog="python -m deeplin"
        )
        parser.add_argument("--module", type=str, help="Module to run (e.g., hexin_server)")

        # Parse known args to allow module-specific arguments
        args, remaining = parser.parse_known_args()

        if args.module == "hexin_server":
            # Update sys.argv for the hexin_server module
            sys.argv = [sys.argv[0]] + remaining
            from deeplin.inference_engine.hexin_server import main as hexin_main
            hexin_main()
        else:
            parser.print_help()
            sys.exit(1)

if __name__ == "__main__":
    main()
