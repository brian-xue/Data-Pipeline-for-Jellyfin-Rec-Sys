from __future__ import annotations

import argparse
import signal

from config import load_config
from runner import run_simulation


def _raise_keyboard_interrupt(signum: int, frame: object) -> None:
	raise KeyboardInterrupt(f"Signal {signum} received")


def _install_signal_handlers() -> None:
	signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
	signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


def main() -> None:
	parser = argparse.ArgumentParser(description="Recommendation system event simulator")
	parser.add_argument("--config", required=True, help="Path to simulator YAML config")
	args = parser.parse_args()

	_install_signal_handlers()
	cfg = load_config(args.config)
	run_simulation(cfg)


if __name__ == "__main__":
	main()

