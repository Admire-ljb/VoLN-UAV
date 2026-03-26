#!/usr/bin/env python3
from __future__ import annotations

import sys
from replay_fix_trajectory import parse_args, replay


def main() -> None:
    sys.argv.extend(["--preset", "normal"]) if "--preset" not in sys.argv else None
    replay(parse_args())


if __name__ == "__main__":
    main()
