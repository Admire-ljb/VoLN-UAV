from __future__ import annotations

import argparse
import json

from voln_uav.benchmark.builder import BenchmarkBuilder
from voln_uav.common.config import load_config
from voln_uav.common.seed import set_seed



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    builder = BenchmarkBuilder(cfg)
    summary = builder.build()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
