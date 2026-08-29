import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from adjudicate_ms_q0 import adjudicate
from run_ms_q0 import post_init_device_mismatches, runtime_audit

from memseal.contracts import (
    dispatch_rows_identical,
    graph_dispatch_observed,
    identical_request_outputs,
    load_config,
    memory_snapshot_valid,
    run_filename,
    sha256,
)


def memory_rows() -> list[dict]:
    return [
        {
            "rank": rank,
            "free_bytes": 60_000,
            "total_bytes": 80_000,
            "allocated_bytes": 10_000,
            "reserved_bytes": 12_000,
            "peak_allocated_bytes": 11_000,
            "peak_reserved_bytes": 13_000,
        }
        for rank in range(4)
    ]


def dispatch_rows(plan: str) -> list[dict]:
    mode = "NONE" if plan == "eager" else "FULL"
    rows = [
        {
            "num_unpadded_tokens": 8,
            "num_padded_tokens": 8,
            "num_paddings": 0,
            "runtime_mode": mode,
        }
    ]
    return [{"rank": rank, "rows": copy.deepcopy(rows)} for rank in range(4)]


def successful_run(platform: str, plan: str, config: dict, config_hash: str) -> dict:
    digest = f"{platform}-canonical"
    outputs = [{"position": index, "token_ids": [1, 2, 3]} for index in range(8)]
    waves = [
        {
            "wave_index": wave,
            "request_count": 8,
            "output_rows": copy.deepcopy(outputs),
            "output_digest": digest,
            "peak_reset": [{"rank": rank, "reset": True} for rank in range(4)],
            "peak_reset_error": None,
            "memory_by_rank": memory_rows(),
            "memory_error": None,
        }
        for wave in range(3)
    ]
    return {
        "schema": "memseal.ms_q0.run.v1",
        "status": "success",
        "failure_stage": None,
        "platform": platform,
        "plan": plan,
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
        "engine_initialized": True,
        "workload_completed": True,
        "shutdown_complete": True,
        "memory_after_init": memory_rows(),
        "memory_after_init_error": None,
        "waves": waves,
        "worker_dispatch": dispatch_rows(plan),
        "canonical_output_digest": digest,
        "checks": {
            "all_24_requests_finished": True,
            "within_wave_identical_requests": True,
            "within_run_determinism": True,
            "rank_dispatch_identity": True,
            "post_init_device_identity": True,
            "eager_only_dispatch": True,
            "compiled_graph_dispatch": True,
            "common_memory_observability": True,
            "resolved_kv_and_graph_config_retained": True,
            "engine_idle_after_run": True,
        },
    }


def write_matrix(directory: Path, runs: dict[tuple[str, str], dict]) -> None:
    for (platform, plan), payload in runs.items():
        (directory / run_filename(platform, plan)).write_text(
            json.dumps(payload), encoding="utf-8"
        )


class MSQ0Tests(unittest.TestCase):
    def setUp(self):
        self.config_path = REPOSITORY_ROOT / "configs" / "ms_q0.json"
        self.config = load_config(self.config_path)
        self.config_hash = sha256(self.config_path)
        self.runs = {
            (platform, plan): successful_run(
                platform, plan, self.config, self.config_hash
            )
            for platform in ("a100", "910b")
            for plan in ("eager", "compiled")
        }

    def decide(self, runs: dict[tuple[str, str], dict]) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_matrix(root, runs)
            return adjudicate(root, self.config, self.config_hash)

    def test_frozen_contract_is_tp4_two_platform_two_plan(self):
        self.assertEqual(set(self.config["platforms"]), {"a100", "910b"})
        self.assertTrue(all(lane["tp"] == 4 for lane in self.config["platforms"].values()))
        self.assertEqual(set(self.config["plans"]), {"eager", "compiled"})
        self.assertEqual(self.config["workload"]["waves"], 3)

    def test_contract_rejects_tp_change(self):
        altered = copy.deepcopy(self.config)
        altered["platforms"]["910b"]["tp"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "TP4"):
                load_config(path)

    def test_common_memory_and_dispatch_contracts(self):
        self.assertTrue(memory_snapshot_valid(memory_rows()))
        malformed = memory_rows()
        del malformed[2]["peak_reserved_bytes"]
        self.assertFalse(memory_snapshot_valid(malformed))
        rows = dispatch_rows("compiled")
        self.assertTrue(dispatch_rows_identical(rows))
        self.assertTrue(graph_dispatch_observed(rows))
        rows[3]["rows"][0]["runtime_mode"] = "NONE"
        self.assertFalse(dispatch_rows_identical(rows))

    def test_parent_runtime_audit_does_not_touch_cuda_devices(self):
        class ForbiddenAccelerator:
            def __getattr__(self, name):
                raise AssertionError(f"parent accessed CUDA before worker spawn: {name}")

        torch = SimpleNamespace(
            __version__="2.11.0+cu128",
            version=SimpleNamespace(cuda="12.8"),
            cuda=ForbiddenAccelerator(),
        )
        vllm = SimpleNamespace(__version__="0.23.0")
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}):
            audit = runtime_audit("a100", torch, vllm)
        self.assertEqual(audit["visible_device_count"], 4)
        self.assertNotIn("device_names", audit)

    def test_device_identity_is_deferred_to_worker_snapshots(self):
        rows = memory_rows()
        for row in rows:
            row["device_name"] = "NVIDIA A100-PCIE-40GB"
        self.assertEqual(post_init_device_mismatches("a100", rows, self.config), [])
        rows[2]["device_name"] = "unexpected"
        self.assertTrue(post_init_device_mismatches("a100", rows, self.config))

    def test_identical_request_check_catches_known_tp_failure_shape(self):
        waves = self.runs[("910b", "eager")]["waves"]
        self.assertTrue(identical_request_outputs(waves))
        waves[0]["output_rows"][7]["token_ids"] = [9, 9, 9]
        self.assertFalse(identical_request_outputs(waves))

    def test_all_checks_pass_unlocks_only_stack_staging(self):
        decision = self.decide(self.runs)
        self.assertEqual(decision["verdict"], "pass_unlock_stack_staging")
        self.assertTrue(decision["continue_to_stack_staging"])
        self.assertFalse(decision["scientific_oracle_census_unblocked"])

    def test_rank_or_output_failure_stops_cross_runtime_claim(self):
        self.runs[("910b", "compiled")]["checks"]["rank_dispatch_identity"] = False
        decision = self.decide(self.runs)
        self.assertEqual(decision["verdict"], "stop_cross_runtime_memseal")

    def test_within_wave_distinct_outputs_stop_cross_runtime_claim(self):
        self.runs[("910b", "eager")]["waves"][1]["output_rows"][7][
            "token_ids"
        ] = [7, 7, 7]
        decision = self.decide(self.runs)
        self.assertEqual(decision["verdict"], "stop_cross_runtime_memseal")

    def test_cross_plan_output_difference_stops_cross_runtime_claim(self):
        for wave in self.runs[("a100", "compiled")]["waves"]:
            for row in wave["output_rows"]:
                row["token_ids"] = [4, 5, 6]
        decision = self.decide(self.runs)
        self.assertEqual(decision["verdict"], "stop_cross_runtime_memseal")

    def test_common_observability_failure_has_its_own_verdict(self):
        del self.runs[("a100", "eager")]["waves"][0]["memory_by_rank"][0][
            "peak_allocated_bytes"
        ]
        self.runs[("a100", "eager")]["checks"]["common_memory_observability"] = False
        decision = self.decide(self.runs)
        self.assertEqual(decision["verdict"], "stop_no_common_observability")

    def test_engine_or_asset_failure_remains_blocked_not_negative(self):
        failed = self.runs[("910b", "compiled")]
        failed["status"] = "failed"
        failed["failure_stage"] = "engine_init"
        failed["error_type"] = "RuntimeError"
        decision = self.decide(self.runs)
        self.assertEqual(decision["verdict"], "blocked_stack_or_asset")
        self.assertFalse(decision["continue_to_stack_staging"])

    def test_missing_cell_is_blocked_stack_or_asset(self):
        del self.runs[("a100", "compiled")]
        decision = self.decide(self.runs)
        self.assertEqual(decision["verdict"], "blocked_stack_or_asset")


if __name__ == "__main__":
    unittest.main()
