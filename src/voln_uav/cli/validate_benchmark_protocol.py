from __future__ import annotations

import argparse
import json

from voln_uav.evaluation.paper_protocol import (
    inspect_benchmark_protocol,
    load_paper_protocol,
    write_protocol_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit existing split manifests against the VoLN-UAV benchmark protocol without modifying the dataset."
    )
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--protocol", default="configs/paper_protocol.yaml")
    parser.add_argument("--out", help="Optional JSON report path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail unless the complete 7,210-episode benchmark and all split invariants are present.",
    )
    args = parser.parse_args()

    report = inspect_benchmark_protocol(
        args.benchmark_root,
        load_paper_protocol(args.protocol),
    )
    if args.out:
        write_protocol_report(report, args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict and report["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
