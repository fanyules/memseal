from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from adjudicate_ms_g0d_quick import adjudicate

from memseal.contracts import output_digest, sha256
from memseal.discovery import (
    MIB,
    PLATFORMS,
    TRACE_ORDER,
    load_quick_config,
    rank_signal,
    recompute_path_signals,
    run_filename,
)


def memory_rows(driver_delta: int = 0, allocator_peak: int = 0) -> tuple[list[dict], list[dict]]:
    before = []
    after = []
    total = 80 * 1024**3
    free = 60 * 1024**3
    allocated = 10 * 1024**3
    reserved = 12 * 1024**3
    for rank in range(4):
        before.append(
            {
                "rank": rank,
                "free_bytes": free,
                "total_bytes": total,
                "allocated_bytes": allocated,
                "reserved_bytes": reserved,
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
            }
        )
        after.append(
            {
                "rank": rank,
                "free_bytes": free - driver_delta,
                "total_bytes": total,
                "allocated_bytes": allocated,
                "reserved_bytes": reserved,
                "peak_allocated_bytes": allocated + allocator_peak,
                "peak_reserved_bytes": reserved + allocator_peak,
            }
        )
    return before, after


def dispatch(mode: str) -> list[dict]:
    rows = [
        {
            "num_unpadded_tokens": 8,
            "num_padded_tokens": 8,
            "num_paddings": 0,
            "runtime_mode": mode,
        }
    ]
    return [{"rank": rank, "rows": copy.deepcopy(rows)} for rank in range(4)]


def output_rows(path: dict) -> list[dict]:
    if path["kind"] == "standard":
        return [
            {
                "role": "standard",
                "position": position,
                "token_ids": [1] * path["output_tokens"],
            }
            for position in range(path["num_sequences"])
        ]
    rows = [
        {
            "role": "background",
            "position": position,
            "token_ids": [1] * path["background_output_tokens"],
        }
        for position in range(path["background_sequences"])
    ]
    rows.append(
        {
            "role": "foreground",
            "position": 0,
            "token_ids": [2] * path["foreground_output_tokens"],
        }
    )
    return rows


def successful_run(
    platform: str,
    config: dict,
    config_hash: str,
    path_signals: dict[str, int] | None = None,
) -> dict:
    path_signals = path_signals or {}
    paths = []
    for index, name in enumerate(TRACE_ORDER):
        spec = config["paths"][name]
        before, after = memory_rows(driver_delta=path_signals.get(name, 0))
        rows = output_rows(spec)
        record = {
            "path_name": name,
            "kind": spec["kind"],
            "status": "success",
            "memory_before_by_rank": before,
            "memory_before_error": None,
            "peak_reset": [{"rank": rank, "reset": True} for rank in range(4)],
            "peak_reset_error": None,
            "memory_after_by_rank": after,
            "memory_after_error": None,
            "worker_dispatch": dispatch("FULL" if index == 0 else "NONE"),
            "request_count": len(rows),
            "output_rows": rows,
            "output_digest": output_digest(rows),
            "all_requests_finished": True,
            "engine_idle_after_path": True,
        }
        if spec["kind"] == "mixed":
            record["background_unfinished_at_foreground_admission"] = True
        paths.append(record)
    return {
        "schema": "memseal.ms_g0d_quick.run.v1",
        "status": "success",
        "failure_stage": None,
        "platform": platform,
        "plan": "compiled",
        "restart_index": 0,
        "config_sha256": config_hash,
        "tensor_parallel_size": 4,
        "physical_devices": [0, 1, 2, 3],
        "model_hashes": {
            "config_sha256": config["model"]["config_sha256"],
            "index_sha256": config["model"]["index_sha256"],
        },
        "runtime_mismatches": [],
        "post_init_runtime_mismatches": [],
        "trace_order": list(TRACE_ORDER),
        "engine_initialized": True,
        "workload_completed": True,
        "shutdown_complete": True,
        "engine_idle_after_run": True,
        "resolved_runtime": {
            "cache": {
                "configured_kv_cache_memory_bytes": config["engine"][
                    "kv_cache_memory_bytes"
                ]
            }
        },
        "paths": paths,
    }


class MSG0DQuickTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = REPOSITORY_ROOT / "configs" / "ms_g0d_quick.json"
        self.config = load_quick_config(self.config_path)
        self.config_hash = sha256(self.config_path)

    def decide(self, runs: dict[str, dict]) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for platform, payload in runs.items():
                (root / run_filename(platform)).write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            return adjudicate(root, self.config, self.config_hash)

    def test_contract_is_two_compiled_tp4_processes_with_explicit_kv(self) -> None:
        self.assertEqual(set(self.config["platforms"]), set(PLATFORMS))
        self.assertTrue(all(row["tp"] == 4 for row in self.config["platforms"].values()))
        self.assertFalse(self.config["engine"]["enforce_eager"])
        self.assertEqual(self.config["engine"]["kv_cache_memory_bytes"], 8 * 1024**3)
        self.assertEqual(tuple(self.config["trace_order"]), TRACE_ORDER)

    def test_signal_never_sums_nested_driver_and_allocator_counters(self) -> None:
        before, after = memory_rows(driver_delta=300 * MIB, allocator_peak=300 * MIB)
        signal = rank_signal(before[0], after[0])
        self.assertEqual(signal["signal_bytes"], 300 * MIB)
        self.assertNotEqual(signal["signal_bytes"], 600 * MIB)

    def test_platform_signal_is_worst_rank_not_rank_sum(self) -> None:
        before, after = memory_rows(driver_delta=100 * MIB)
        result = recompute_path_signals(
            {
                "memory_before_by_rank": before,
                "memory_after_by_rank": after,
            }
        )
        self.assertEqual(result["platform_signal_bytes"], 100 * MIB)

    def test_same_path_cross_platform_signal_continues(self) -> None:
        runs = {
            "a100": successful_run(
                "a100", self.config, self.config_hash, {"wide_decode": 300 * MIB}
            ),
            "910b": successful_run(
                "910b", self.config, self.config_hash, {"wide_decode": 600 * MIB}
            ),
        }
        decision = self.decide(runs)
        self.assertEqual(decision["verdict"], "continue_to_formal_stack_staging")
        self.assertEqual(decision["qualifying_paths"], ["wide_decode"])

    def test_large_signals_on_different_paths_stop(self) -> None:
        runs = {
            "a100": successful_run(
                "a100", self.config, self.config_hash, {"baseline_decode": 600 * MIB}
            ),
            "910b": successful_run(
                "910b", self.config, self.config_hash, {"wide_decode": 600 * MIB}
            ),
        }
        decision = self.decide(runs)
        self.assertEqual(decision["verdict"], "stop_no_large_cross_runtime_path_signal")
        self.assertEqual(decision["qualifying_paths"], [])

    def test_rank_dispatch_mismatch_is_technical_invalid(self) -> None:
        runs = {
            platform: successful_run(platform, self.config, self.config_hash)
            for platform in PLATFORMS
        }
        runs["910b"]["paths"][3]["worker_dispatch"][3]["rows"][0][
            "runtime_mode"
        ] = "FULL"
        decision = self.decide(runs)
        self.assertEqual(decision["verdict"], "blocked_technical_invalid")
        self.assertFalse(decision["continue_to_formal_stack_staging"])

    def test_eager_fallback_path_is_valid_when_another_path_uses_graph(self) -> None:
        runs = {
            platform: successful_run(platform, self.config, self.config_hash)
            for platform in PLATFORMS
        }
        decision = self.decide(runs)
        self.assertTrue(all(row["valid"] for row in decision["runs"].values()))
        self.assertEqual(decision["decision_key"], "stop")


if __name__ == "__main__":
    unittest.main()
