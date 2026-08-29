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
    PLANS,
    PLATFORMS,
    dispatch_rows_identical,
    eager_only_dispatch,
    graph_dispatch_observed,
    identical_request_outputs,
    load_config,
    memory_snapshot_valid,
    output_digest,
    run_filename,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adjudicate the frozen MS-Q0 matrix")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "ms_q0.json",
    )
    return parser.parse_args()


def load_run(results_dir: Path, platform: str, plan: str) -> dict[str, Any]:
    path = results_dir / run_filename(platform, plan)
    if not path.is_file():
        return {
            "status": "missing",
            "platform": platform,
            "plan": plan,
            "error": f"missing result: {path.name}",
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "invalid",
            "platform": platform,
            "plan": plan,
            "error": f"invalid result {path.name}: {error}",
        }


def engine_and_asset_valid(
    run: dict[str, Any],
    platform: str,
    plan: str,
    config: dict[str, Any],
    config_sha256: str,
) -> bool:
    return (
        run.get("schema") == "memseal.ms_q0.run.v1"
        and run.get("status") == "success"
        and run.get("platform") == platform
        and run.get("plan") == plan
        and run.get("restart_index") == 0
        and run.get("config_sha256") == config_sha256
        and run.get("tensor_parallel_size") == 4
        and run.get("physical_devices")
        == config["platforms"][platform]["physical_devices"]
        and run.get("model_hashes", {}).get("config_sha256")
        == config["model"]["config_sha256"]
        and run.get("model_hashes", {}).get("index_sha256")
        == config["model"]["index_sha256"]
        and run.get("engine_initialized") is True
        and run.get("workload_completed") is True
        and run.get("shutdown_complete") is True
        and not run.get("runtime_mismatches")
        and run.get("post_init_runtime_mismatches") == []
    )


def memory_observable(run: dict[str, Any], config: dict[str, Any]) -> bool:
    required = tuple(config["qualification"]["required_memory_fields"])
    waves = run.get("waves")
    if not isinstance(waves, list) or len(waves) != config["workload"]["waves"]:
        return False
    snapshots = [run.get("memory_after_init")] + [
        wave.get("memory_by_rank") for wave in waves if isinstance(wave, dict)
    ]
    resets_ok = all(
        isinstance(wave, dict)
        and wave.get("peak_reset_error") is None
        and isinstance(wave.get("peak_reset"), list)
        and len(wave["peak_reset"]) == 4
        and all(item.get("reset") for item in wave["peak_reset"])
        for wave in waves
    )
    return (
        run.get("memory_after_init_error") is None
        and resets_ok
        and len(snapshots) == config["workload"]["waves"] + 1
        and all(memory_snapshot_valid(rows, required) for rows in snapshots)
        and run.get("checks", {}).get("common_memory_observability") is True
    )


def local_correctness_valid(run: dict[str, Any], plan: str) -> bool:
    checks = run.get("checks", {})
    waves = run.get("waves")
    if not isinstance(waves, list) or len(waves) != 3:
        return False
    actual_digests = [
        output_digest(wave.get("output_rows"))
        for wave in waves
        if isinstance(wave, dict)
    ]
    dispatch = run.get("worker_dispatch")
    actual_dispatch_valid = dispatch_rows_identical(dispatch) and (
        eager_only_dispatch(dispatch)
        if plan == "eager"
        else graph_dispatch_observed(dispatch)
    )
    required = [
        "all_24_requests_finished",
        "within_wave_identical_requests",
        "within_run_determinism",
        "rank_dispatch_identity",
        "post_init_device_identity",
        "resolved_kv_and_graph_config_retained",
        "engine_idle_after_run",
    ]
    required.append("eager_only_dispatch" if plan == "eager" else "compiled_graph_dispatch")
    return (
        all(checks.get(name) is True for name in required)
        and sum(wave.get("request_count", 0) for wave in waves) == 24
        and identical_request_outputs(waves)
        and len(actual_digests) == 3
        and len(set(actual_digests)) == 1
        and actual_dispatch_valid
    )


def canonical_digest(run: dict[str, Any]) -> str | None:
    waves = run.get("waves")
    if not isinstance(waves, list) or not waves or not isinstance(waves[0], dict):
        return None
    return output_digest(waves[0].get("output_rows"))


def adjudicate(results_dir: Path, config: dict[str, Any], config_hash: str) -> dict[str, Any]:
    runs = {
        (platform, plan): load_run(results_dir, platform, plan)
        for platform in PLATFORMS
        for plan in PLANS
    }
    engine_valid = {
        cell: engine_and_asset_valid(run, *cell, config, config_hash)
        for cell, run in runs.items()
    }
    local_correctness = {
        cell: local_correctness_valid(run, cell[1]) for cell, run in runs.items()
    }
    observability = {
        cell: memory_observable(run, config) for cell, run in runs.items()
    }
    cross_plan_identity = {
        platform: (
            canonical_digest(runs[(platform, "eager")]) is not None
            and canonical_digest(runs[(platform, "eager")])
            == canonical_digest(runs[(platform, "compiled")])
        )
        for platform in PLATFORMS
    }

    if not all(engine_valid.values()):
        decision_key = "engine_or_asset_failure"
    elif not all(local_correctness.values()) or not all(cross_plan_identity.values()):
        decision_key = "correctness_or_rank_dispatch_failure"
    elif not all(observability.values()):
        decision_key = "common_memory_observability_failure"
    else:
        decision_key = "all_checks_pass"
    verdict = config["decisions"][decision_key]
    return {
        "schema": "memseal.ms_q0.decision.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate": "MS-Q0",
        "evidence_scope": config["evidence_scope"],
        "decision_key": decision_key,
        "verdict": verdict,
        "continue_to_stack_staging": decision_key == "all_checks_pass",
        "scientific_oracle_census_unblocked": False,
        "cells": {
            f"{platform}_{plan}": {
                "status": runs[(platform, plan)].get("status"),
                "engine_and_asset_valid": engine_valid[(platform, plan)],
                "local_correctness_valid": local_correctness[(platform, plan)],
                "common_memory_observable": observability[(platform, plan)],
                "canonical_output_digest": canonical_digest(runs[(platform, plan)]),
                "failure_stage": runs[(platform, plan)].get("failure_stage"),
                "error_type": runs[(platform, plan)].get("error_type"),
                "error": runs[(platform, plan)].get("error"),
            }
            for platform in PLATFORMS
            for plan in PLANS
        },
        "cross_plan_output_identity": cross_plan_identity,
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config = load_config(args.config)
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
