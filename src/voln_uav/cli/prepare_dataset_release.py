from __future__ import annotations

import argparse
import json
from pathlib import Path

from voln_uav.data.release_packager import prepare_dataset_release


DEFAULT_ENV_URL = "https://huggingface.co/datasets/Louj/VoLN-UAV-ENV"
DEFAULT_DATASET_URL = "https://huggingface.co/datasets/Louj/VoLN-UAV-Dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the VoLN-UAV dataset release package.")
    parser.add_argument("--easy-root", required=True, help="Path to VoLN-simple or the Easy raw-data root.")
    parser.add_argument("--normal-root", required=True, help="Path to VoLN-normal or the Normal raw-data root.")
    parser.add_argument("--hard-root", required=True, help="Path to VoLN-hard or the Hard raw-data root.")
    parser.add_argument("--out-root", required=True, help="Output release directory.")
    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL)
    parser.add_argument("--env-url", default=DEFAULT_ENV_URL)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--camera", default=None, help="Preferred camera directory name, e.g. FrontCamera.")
    parser.add_argument(
        "--asset-mode",
        choices=["index", "copy"],
        default="index",
        help="'index' writes metadata only; 'copy' copies selected RGB frames into the release tree.",
    )
    parser.add_argument("--zip-path", default=None, help="Optional output ZIP path.")
    parser.add_argument("--no-zip", action="store_true", help="Do not create a ZIP archive.")
    parser.add_argument(
        "--max-episodes-per-source",
        "--max-episodes-per-difficulty",
        type=int,
        default=None,
        dest="max_episodes_per_source",
        help="Optional per-raw-source cap for smoke-testing the packaging process.",
    )
    args = parser.parse_args()

    summary = prepare_dataset_release(
        easy_root=Path(args.easy_root),
        normal_root=Path(args.normal_root),
        hard_root=Path(args.hard_root),
        out_root=Path(args.out_root),
        dataset_url=args.dataset_url,
        env_url=args.env_url,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        camera=args.camera,
        asset_mode=args.asset_mode,
        zip_path=Path(args.zip_path) if args.zip_path else None,
        write_zip=not args.no_zip,
        max_episodes_per_source=args.max_episodes_per_source,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
