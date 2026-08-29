from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memseal.contracts import REQUIRED_MEMORY_FIELDS, memory_snapshot_valid

PLATFORMS = ("a100", "910b")
TRACE_ORDER = (
    "baseline_decode",
    "sampler_first",
    "sampler_repeat",
    "max_prefill",
    "wide_decode",
    "mixed",
)
MIB = 1024 * 1024


def load_quick_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "memseal.ms_g0d_quick.v1":
        raise ValueError("unexpected MS-G0D-Q config schema")
    if set(config.get("platforms", {})) != set(PLATFORMS):
        raise ValueError("MS-G0D-Q requires exactly the A100 and 910B lanes")
    for platform in PLATFORMS:
        lane = config["platforms"][platform]
        if lane.get("tp") != 4 or lane.get("physical_devices") != [0, 1, 2, 3]:
            raise ValueError(f"{platform} must remain TP4 on physical devices 0-3")

    expected_engine = {
        "max_model_len": 4096,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 4096,
        "kv_cache_memory_bytes": 8 * 1024**3,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "disable_custom_all_reduce": True,
        "enforce_eager": False,
    }
    if config.get("engine") != expected_engine:
        raise ValueError("MS-G0D-Q engine contract changed")
    if tuple(config.get("trace_order", ())) != TRACE_ORDER:
        raise ValueError("MS-G0D-Q trace order changed")

    expected_paths = {
        "baseline_decode": {
            "kind": "standard",
            "num_sequences": 8,
            "prompt_tokens": 128,
            "output_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "sampler_first": {
            "kind": "standard",
            "num_sequences": 32,
            "prompt_tokens": 128,
            "output_tokens": 1,
            "temperature": 1.0,
            "top_p": 0.9,
        },
        "sampler_repeat": {
            "kind": "standard",
            "num_sequences": 32,
            "prompt_tokens": 128,
            "output_tokens": 1,
            "temperature": 1.0,
            "top_p": 0.9,
        },
        "max_prefill": {
            "kind": "standard",
            "num_sequences": 1,
            "prompt_tokens": 4095,
            "output_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "wide_decode": {
            "kind": "standard",
            "num_sequences": 32,
            "prompt_tokens": 128,
            "output_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "mixed": {
            "kind": "mixed",
            "background_sequences": 16,
            "background_prompt_tokens": 128,
            "background_output_tokens": 4,
            "foreground_prompt_tokens": 2048,
            "foreground_output_tokens": 1,
        },
    }
    if config.get("paths") != expected_paths:
        raise ValueError("MS-G0D-Q path contract changed")

    expected_decision = {
        "cross_platform_floor_mib": 256,
        "one_platform_strong_mib": 512,
        "continue": "continue_to_formal_stack_staging",
        "stop": "stop_no_large_cross_runtime_path_signal",
        "blocked": "blocked_technical_invalid",
    }
    if config.get("decision") != expected_decision:
        raise ValueError("MS-G0D-Q decision contract changed")
    for name in ("config_sha256", "index_sha256"):
        value = config.get("model", {}).get(name, "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"invalid frozen model hash: {name}")
    if not isinstance(config.get("prompt_pattern"), str) or not config["prompt_pattern"]:
        raise ValueError("MS-G0D-Q prompt pattern must be non-empty")
    return config


def run_filename(platform: str, restart_index: int = 0) -> str:
    if platform not in PLATFORMS:
        raise ValueError("unknown MS-G0D-Q platform")
    if restart_index != 0:
        raise ValueError("MS-G0D-Q has exactly one fresh process per platform")
    return f"{platform}_tp4_compiled_r{restart_index}.json"


def rank_signal(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    before_driver_used = int(before["total_bytes"]) - int(before["free_bytes"])
    after_driver_used = int(after["total_bytes"]) - int(after["free_bytes"])
    driver_delta = max(0, after_driver_used - before_driver_used)
    allocator_current_delta = max(
        0, int(after["allocated_bytes"]) - int(before["allocated_bytes"])
    )
    allocator_peak_increment = max(
        0, int(after["peak_allocated_bytes"]) - int(before["allocated_bytes"])
    )
    return {
        "driver_used_before_bytes": before_driver_used,
        "driver_used_after_bytes": after_driver_used,
        "persistent_driver_delta_bytes": driver_delta,
        "allocator_current_delta_bytes": allocator_current_delta,
        "allocator_peak_increment_bytes": allocator_peak_increment,
        "signal_bytes": max(driver_delta, allocator_peak_increment),
    }


def recompute_path_signals(path: dict[str, Any]) -> dict[str, Any]:
    before_rows = path.get("memory_before_by_rank")
    after_rows = path.get("memory_after_by_rank")
    if not memory_snapshot_valid(before_rows, REQUIRED_MEMORY_FIELDS):
        raise ValueError("invalid before-path memory snapshot")
    if not memory_snapshot_valid(after_rows, REQUIRED_MEMORY_FIELDS):
        raise ValueError("invalid after-path memory snapshot")
    before = {row["rank"]: row for row in before_rows}
    after = {row["rank"]: row for row in after_rows}
    rows = []
    for rank in range(4):
        row = {"rank": rank, **rank_signal(before[rank], after[rank])}
        rows.append(row)
    return {
        "by_rank": rows,
        "platform_signal_bytes": max(row["signal_bytes"] for row in rows),
    }


def reset_rows_valid(rows: Any) -> bool:
    return (
        isinstance(rows, list)
        and len(rows) == 4
        and {row.get("rank") for row in rows if isinstance(row, dict)}
        == {0, 1, 2, 3}
        and all(row.get("reset") is True for row in rows)
    )
