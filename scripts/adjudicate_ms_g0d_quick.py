#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from memseal.contracts import (
    REQUIRED_MEMORY_FIELDS,
    dispatch_rows_identical,
    graph_dispatch_observed,
    memory_snapshot_valid,
    output_digest,
    sha256,
)
from memseal.discovery import (
    MIB,
    PLATFORMS,
    TRACE_ORDER,
    load_quick_config,
    recompute_path_signals,
    reset_rows_valid,
    run_filename,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adjudicate MS-G0D-Q")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "ms_g0d_quick.json",
    )
    return parser.parse_args()


def load_run(results_dir: Path, platform: str) -> dict[str, Any]:
    path = results_dir / run_filename(platform)
    if not path.is_file():
        return {
            "status": "missing",
            "platform": platform,
            "error": f"missing result: {path.name}",
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "invalid",
            "platform": platform,
            "error": f"invalid result {path.name}: {error}",
        }


def output_complete(path: dict[str, Any], path_config: dict[str, Any]) -> bool:
    rows = path.get("output_rows")
    if not isinstance(rows, list):
        return False
    expected: dict[tuple[str, int], int] = {}
    if path_config["kind"] == "standard":
        expected = {
            ("standard", position): path_config["output_tokens"]
            for position in range(path_config["num_sequences"])
        }
    else:
        expected = {
            ("background", position): path_config["background_output_tokens"]
            for position in range(path_config["background_sequences"])
        }
        expected[("foreground", 0)] = path_config["foreground_output_tokens"]
        if path.get("background_unfinished_at_foreground_admission") is not True:
            return False
    observed: dict[tuple[str, int], int] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("token_ids"), list):
            return False
        key = (row.get("role"), row.get("position"))
        if key in observed:
            return False
        observed[key] = len(row["token_ids"])
    return (
        observed == expected
        and path.get("request_count") == len(expected)
        and path.get("all_requests_finished") is True
        and path.get("output_digest") == output_digest(rows)
    )


def path_valid(path: dict[str, Any], name: str, config: dict[str, Any]) -> bool:
    return (
        path.get("path_name") == name
        and path.get("kind") == config["paths"][name]["kind"]
        and path.get("status") == "success"
        and path.get("memory_before_error") is None
        and memory_snapshot_valid(
            path.get("memory_before_by_rank"), REQUIRED_MEMORY_FIELDS
        )
        and path.get("peak_reset_error") is None
        and reset_rows_valid(path.get("peak_reset"))
        and path.get("memory_after_error") is None
        and memory_snapshot_valid(
            path.get("memory_after_by_rank"), REQUIRED_MEMORY_FIELDS
        )
        and dispatch_rows_identical(path.get("worker_dispatch"))
        and output_complete(path, config["paths"][name])
        and path.get("engine_idle_after_path") is True
    )


def run_valid(
    run: dict[str, Any], platform: str, config: dict[str, Any], config_hash: str
) -> bool:
    paths = run.get("paths")
    return (
        run.get("schema") == "memseal.ms_g0d_quick.run.v1"
        and run.get("status") == "success"
        and run.get("platform") == platform
        and run.get("plan") == "compiled"
        and run.get("restart_index") == 0
        and run.get("config_sha256") == config_hash
        and run.get("tensor_parallel_size") == 4
        and run.get("physical_devices") == config["platforms"][platform]["physical_devices"]
        and run.get("model_hashes", {}).get("config_sha256")
        == config["model"]["config_sha256"]
        and run.get("model_hashes", {}).get("index_sha256")
        == config["model"]["index_sha256"]
        and run.get("engine_initialized") is True
        and run.get("workload_completed") is True
        and run.get("shutdown_complete") is True
        and not run.get("runtime_mismatches")
        and run.get("post_init_runtime_mismatches") == []
        and run.get("trace_order") == list(TRACE_ORDER)
        and isinstance(paths, list)
        and [path.get("path_name") for path in paths if isinstance(path, dict)]
        == list(TRACE_ORDER)
        and all(
            isinstance(path, dict) and path_valid(path, name, config)
            for path, name in zip(paths, TRACE_ORDER, strict=True)
        )
        and any(graph_dispatch_observed(path.get("worker_dispatch")) for path in paths)
        and run.get("resolved_runtime", {}).get("cache", {}).get(
            "configured_kv_cache_memory_bytes"
        )
        == config["engine"]["kv_cache_memory_bytes"]
        and run.get("engine_idle_after_run") is True
    )


def adjudicate(results_dir: Path, config: dict[str, Any], config_hash: str) -> dict[str, Any]:
    runs = {platform: load_run(results_dir, platform) for platform in PLATFORMS}
    valid = {
        platform: run_valid(runs[platform], platform, config, config_hash)
        for platform in PLATFORMS
    }
    signals: dict[str, dict[str, dict[str, Any]]] = {
        platform: {} for platform in PLATFORMS
    }
    if all(valid.values()):
        for platform in PLATFORMS:
            by_name = {path["path_name"]: path for path in runs[platform]["paths"]}
            for name in TRACE_ORDER:
                signals[platform][name] = recompute_path_signals(by_name[name])

    floor_bytes = config["decision"]["cross_platform_floor_mib"] * MIB
    strong_bytes = config["decision"]["one_platform_strong_mib"] * MIB
    qualifying_paths: list[str] = []
    if all(valid.values()):
        for name in TRACE_ORDER:
            per_platform = [
                signals[platform][name]["platform_signal_bytes"]
                for platform in PLATFORMS
            ]
            if min(per_platform) >= floor_bytes and max(per_platform) >= strong_bytes:
                qualifying_paths.append(name)

    if not all(valid.values()):
        decision_key = "blocked"
    elif qualifying_paths:
        decision_key = "continue"
    else:
        decision_key = "stop"
    verdict = config["decision"][decision_key]

    path_rows: dict[str, Any] = {}
    for name in TRACE_ORDER:
        path_rows[name] = {}
        for platform in PLATFORMS:
            recomputed = signals[platform].get(name)
            path_rows[name][platform] = (
                {
                    **recomputed,
                    "platform_signal_mib": recomputed["platform_signal_bytes"] / MIB,
                }
                if recomputed is not None
                else None
            )
        path_rows[name]["qualifies"] = name in qualifying_paths

    return {
        "schema": "memseal.ms_g0d_quick.decision.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate": "MS-G0D-Q",
        "evidence_scope": config["evidence_scope"],
        "decision_key": decision_key,
        "verdict": verdict,
        "continue_to_formal_stack_staging": decision_key == "continue",
        "scientific_oracle_census_unblocked": False,
        "thresholds": {
            "cross_platform_floor_bytes": floor_bytes,
            "one_platform_strong_bytes": strong_bytes,
        },
        "qualifying_paths": qualifying_paths,
        "runs": {
            platform: {
                "status": runs[platform].get("status"),
                "valid": valid[platform],
                "failure_stage": runs[platform].get("failure_stage"),
                "error_type": runs[platform].get("error_type"),
                "error": runs[platform].get("error"),
            }
            for platform in PLATFORMS
        },
        "paths": path_rows,
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config = load_quick_config(args.config)
    decision = adjudicate(args.results_dir, config, sha256(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": decision["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
