#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from memseal.contracts import PLANS, PLATFORMS, load_config, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze MS-Q0 before any output")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "ms_q0.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPOSITORY_ROOT / "docs" / "MS_Q0_PROTOCOL.md",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def repository_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config = load_config(args.config)
    if not args.protocol.is_file():
        raise FileNotFoundError(f"protocol is absent: {args.protocol}")
    sources = [
        REPOSITORY_ROOT / "src" / "memseal" / "contracts.py",
        REPOSITORY_ROOT / "scripts" / "freeze_ms_q0.py",
        REPOSITORY_ROOT / "scripts" / "run_ms_q0.py",
        REPOSITORY_ROOT / "scripts" / "adjudicate_ms_q0.py",
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("MS-Q0 source is absent: " + ", ".join(missing))
    payload = {
        "schema": "memseal.ms_q0.freeze.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "pre_output_repository_commit": repository_commit(),
        "config_path": relative(args.config),
        "config_sha256": sha256(args.config),
        "protocol_path": relative(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "source_sha256": {relative(path): sha256(path) for path in sources},
        "matrix": [
            {
                "platform": platform,
                "plan": plan,
                "tensor_parallel_size": config["platforms"][platform]["tp"],
                "physical_devices": config["platforms"][platform]["physical_devices"],
                "fresh_processes": config["execution"][
                    "fresh_processes_per_platform_plan"
                ],
            }
            for platform in PLATFORMS
            for plan in PLANS
        ],
        "decisions": config["decisions"],
        "evidence_scope": config["evidence_scope"],
        "ms_g0d_unblocked": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "commit": payload["pre_output_repository_commit"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
