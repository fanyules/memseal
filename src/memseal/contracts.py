from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PLATFORMS = ("a100", "910b")
PLANS = ("eager", "compiled")
REQUIRED_MEMORY_FIELDS = (
    "free_bytes",
    "total_bytes",
    "allocated_bytes",
    "reserved_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "memseal.ms_q0.v1":
        raise ValueError("unexpected MS-Q0 config schema")
    if set(config.get("platforms", {})) != set(PLATFORMS):
        raise ValueError("MS-Q0 requires exactly the A100 and 910B lanes")
    for platform in PLATFORMS:
        lane = config["platforms"][platform]
        if lane.get("tp") != 4 or lane.get("physical_devices") != [0, 1, 2, 3]:
            raise ValueError(f"{platform} must remain TP4 on physical devices 0-3")
    if set(config.get("plans", {})) != set(PLANS):
        raise ValueError("MS-Q0 requires exactly eager and compiled plans")
    if config["plans"]["eager"].get("enforce_eager") is not True:
        raise ValueError("the eager plan must enforce eager execution")
    if config["plans"]["compiled"].get("enforce_eager") is not False:
        raise ValueError("the compiled plan must use runtime-default compilation")
    if any(config["plans"][plan].get("compilation_config") is not None for plan in PLANS):
        raise ValueError("MS-Q0 must not override either plan's compilation config")

    workload = config["workload"]
    expected_workload = {
        "num_sequences": 8,
        "prompt_tokens": 128,
        "output_tokens": 8,
        "waves": 3,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
    }
    for name, expected in expected_workload.items():
        if workload.get(name) != expected:
            raise ValueError(f"frozen workload field changed: {name}")
    execution = config["execution"]
    if execution.get("fresh_processes_per_platform_plan") != 1:
        raise ValueError("MS-Q0 requires one fresh process per platform-plan cell")
    if execution.get("engine_core_multiprocessing") is not False:
        raise ValueError("MS-Q0 requires in-process engine-core instrumentation")
    if not isinstance(execution.get("timeout_seconds"), int) or execution["timeout_seconds"] <= 0:
        raise ValueError("MS-Q0 requires a positive integer timeout")

    required_fields = tuple(config["qualification"].get("required_memory_fields", ()))
    if required_fields != REQUIRED_MEMORY_FIELDS:
        raise ValueError("the common memory field contract changed")
    decisions = config.get("decisions", {})
    expected_decisions = {
        "all_checks_pass": "pass_unlock_stack_staging",
        "correctness_or_rank_dispatch_failure": "stop_cross_runtime_memseal",
        "common_memory_observability_failure": "stop_no_common_observability",
        "engine_or_asset_failure": "blocked_stack_or_asset",
    }
    if decisions != expected_decisions:
        raise ValueError("MS-Q0 decision labels changed")

    model = config["model"]
    for name in ("config_sha256", "index_sha256"):
        value = model.get(name, "")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"invalid frozen model hash: {name}")
    return config


def run_filename(platform: str, plan: str, restart_index: int = 0) -> str:
    if platform not in PLATFORMS or plan not in PLANS:
        raise ValueError("unknown MS-Q0 platform or plan")
    if restart_index != 0:
        raise ValueError("MS-Q0 has exactly one fresh process per cell")
    return f"{platform}_tp4_{plan}_r{restart_index}.json"


def memory_snapshot_valid(
    rows: Any,
    required_fields: tuple[str, ...] = REQUIRED_MEMORY_FIELDS,
) -> bool:
    if not isinstance(rows, list) or len(rows) != 4:
        return False
    ranks: set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or not all(field in row for field in required_fields):
            return False
        rank = row.get("rank")
        if not isinstance(rank, int):
            return False
        ranks.add(rank)
        values = [row[field] for field in required_fields]
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            return False
        if row["free_bytes"] > row["total_bytes"]:
            return False
        if row["allocated_bytes"] > row["reserved_bytes"]:
            return False
        if row["peak_allocated_bytes"] < row["allocated_bytes"]:
            return False
        if row["peak_reserved_bytes"] < row["reserved_bytes"]:
            return False
    return ranks == {0, 1, 2, 3}


def dispatch_rows_identical(worker_dispatch: Any) -> bool:
    if not isinstance(worker_dispatch, list) or len(worker_dispatch) != 4:
        return False
    by_rank: dict[int, Any] = {}
    for item in worker_dispatch:
        if not isinstance(item, dict) or not isinstance(item.get("rank"), int):
            return False
        by_rank[item["rank"]] = item.get("rows")
    if set(by_rank) != {0, 1, 2, 3}:
        return False
    reference = by_rank[0]
    return isinstance(reference, list) and bool(reference) and all(
        by_rank[rank] == reference for rank in (1, 2, 3)
    )


def eager_only_dispatch(worker_dispatch: Any) -> bool:
    if not dispatch_rows_identical(worker_dispatch):
        return False
    rows = next(item["rows"] for item in worker_dispatch if item["rank"] == 0)
    return all(str(row.get("runtime_mode", "")).upper() in {"NONE", "EAGER"} for row in rows)


def graph_dispatch_observed(worker_dispatch: Any) -> bool:
    if not dispatch_rows_identical(worker_dispatch):
        return False
    rows = next(item["rows"] for item in worker_dispatch if item["rank"] == 0)
    return any(
        str(row.get("runtime_mode", "")).upper() in {"FULL", "PIECEWISE"}
        for row in rows
    )


def identical_request_outputs(waves: Any) -> bool:
    if not isinstance(waves, list) or not waves:
        return False
    for wave in waves:
        rows = wave.get("output_rows") if isinstance(wave, dict) else None
        if not isinstance(rows, list) or not rows:
            return False
        token_rows = [row.get("token_ids") for row in rows if isinstance(row, dict)]
        if len(token_rows) != len(rows) or any(not isinstance(row, list) for row in token_rows):
            return False
        if any(row != token_rows[0] for row in token_rows[1:]):
            return False
    return True
