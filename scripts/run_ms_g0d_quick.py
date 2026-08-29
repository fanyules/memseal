#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROCESS_STARTED_NS = time.perf_counter_ns()
REQUEST_COUNTER = 0
REQUEST_TIMEOUT_SECONDS = 1200
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from run_ms_q0 import (
    DispatchTrace,
    commit_is_ancestor,
    configure_environment,
    deterministic_prompts,
    jsonable,
    post_init_device_mismatches,
    repository_commit,
    resolved_runtime,
    runtime_audit,
    runtime_mismatches,
    safe_collective,
    token_pool,
    validate_model_asset,
    worker_reset_peak_memory,
    worker_resource_snapshot,
)

from memseal.contracts import (
    REQUIRED_MEMORY_FIELDS,
    dispatch_rows_identical,
    graph_dispatch_observed,
    memory_snapshot_valid,
    output_digest,
    sha256,
)
from memseal.discovery import PLATFORMS, TRACE_ORDER, load_quick_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one MS-G0D-Q TP4 process")
    parser.add_argument("--platform", choices=PLATFORMS, required=True)
    parser.add_argument("--restart-index", type=int, default=0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "ms_g0d_quick.json",
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results"
            / "ms_g0d_quick"
            / "MS_G0D_QUICK_FREEZE.json"
        ),
    )
    return parser.parse_args()


def validate_freeze(
    args: argparse.Namespace, config: dict[str, Any], freeze: dict[str, Any]
) -> None:
    if args.restart_index != 0:
        raise ValueError("MS-G0D-Q has exactly one fresh process per platform")
    if freeze.get("status") != "frozen":
        raise ValueError("MS-G0D-Q freeze is absent or not frozen")
    if freeze.get("config_sha256") != sha256(args.config):
        raise ValueError("active config differs from the public MS-G0D-Q freeze")
    protocol = REPOSITORY_ROOT / freeze["protocol_path"]
    if sha256(protocol) != freeze.get("protocol_sha256"):
        raise ValueError("active protocol differs from the public MS-G0D-Q freeze")
    if not freeze.get("source_sha256"):
        raise ValueError("MS-G0D-Q freeze does not contain source hashes")
    for relative_path, expected_hash in freeze["source_sha256"].items():
        if sha256(REPOSITORY_ROOT / relative_path) != expected_hash:
            raise ValueError(f"frozen source changed: {relative_path}")
    commit = freeze.get("pre_output_repository_commit")
    if not isinstance(commit, str) or not commit_is_ancestor(commit):
        raise ValueError("MS-G0D-Q pre-output commit is not an ancestor")
    matching = [
        row
        for row in freeze.get("matrix", [])
        if row.get("platform") == args.platform
        and row.get("plan") == "compiled"
        and row.get("fresh_processes") == 1
    ]
    if len(matching) != 1:
        raise ValueError("requested process is outside the frozen matrix")
    if tuple(freeze.get("trace_order", ())) != TRACE_ORDER:
        raise ValueError("frozen trace order differs from the config")
    if config["platforms"][args.platform]["tp"] != 4:
        raise ValueError("requested lane no longer has the TP4 contract")


def path_sampling_params(path: dict[str, Any], seed: int, output_tokens: int | None = None) -> Any:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    params = SamplingParams(
        temperature=float(path.get("temperature", 0.0)),
        top_p=float(path.get("top_p", 1.0)),
        ignore_eos=True,
        max_tokens=int(output_tokens if output_tokens is not None else path["output_tokens"]),
        detokenize=False,
        seed=seed,
    )
    params.output_kind = RequestOutputKind.CUMULATIVE
    return params


def add_requests(
    llm: Any,
    prompts: list[dict[str, list[int]]],
    params: Any,
    *,
    platform: str,
    path_name: str,
    role: str,
    expected_tokens: int,
) -> dict[str, dict[str, Any]]:
    global REQUEST_COUNTER

    requests: dict[str, dict[str, Any]] = {}
    for position, prompt in enumerate(prompts):
        request_id = f"memseal-{platform}-{path_name}-{role}-{REQUEST_COUNTER}"
        REQUEST_COUNTER += 1
        llm.llm_engine.add_request(request_id, prompt, params)
        requests[request_id] = {
            "role": role,
            "position": position,
            "expected_tokens": expected_tokens,
        }
    return requests


def retain_finished(
    step_outputs: Any,
    requests: dict[str, dict[str, Any]],
    finished: dict[str, list[int]],
) -> None:
    for output in step_outputs:
        request_id = str(output.request_id)
        if request_id not in requests:
            raise RuntimeError(f"unexpected request ID returned: {request_id}")
        if output.finished:
            if request_id in finished:
                raise RuntimeError(f"duplicate completion returned: {request_id}")
            if not output.outputs:
                raise RuntimeError("finished request had no sequence output")
            finished[request_id] = [int(token) for token in output.outputs[0].token_ids]


def finish_requests(
    llm: Any,
    requests: dict[str, dict[str, Any]],
    finished: dict[str, list[int]],
    deadline: float,
) -> list[dict[str, Any]]:
    while llm.llm_engine.has_unfinished_requests():
        if time.monotonic() > deadline:
            raise TimeoutError("path exceeded the fixed request timeout")
        retain_finished(llm.llm_engine.step(), requests, finished)
    if set(finished) != set(requests):
        raise RuntimeError("path did not finish every request")
    rows = []
    for request_id, request in sorted(
        requests.items(), key=lambda item: (item[1]["role"], item[1]["position"])
    ):
        tokens = finished[request_id]
        if len(tokens) != request["expected_tokens"]:
            raise RuntimeError("ignore-EOS request produced the wrong output length")
        rows.append(
            {
                "role": request["role"],
                "position": request["position"],
                "token_ids": tokens,
            }
        )
    return rows


def run_standard_path(
    llm: Any,
    path: dict[str, Any],
    pool: list[int],
    *,
    platform: str,
    path_name: str,
    seed: int,
) -> dict[str, Any]:
    prompts = deterministic_prompts(path["num_sequences"], path["prompt_tokens"], pool)
    requests = add_requests(
        llm,
        prompts,
        path_sampling_params(path, seed),
        platform=platform,
        path_name=path_name,
        role="standard",
        expected_tokens=path["output_tokens"],
    )
    started_ns = time.perf_counter_ns()
    rows = finish_requests(llm, requests, {}, time.monotonic() + REQUEST_TIMEOUT_SECONDS)
    return {
        "request_count": len(rows),
        "output_rows": rows,
        "output_digest": output_digest(rows),
        "wall_ms": (time.perf_counter_ns() - started_ns) / 1_000_000,
        "all_requests_finished": True,
    }


def run_mixed_path(
    llm: Any,
    path: dict[str, Any],
    pool: list[int],
    *,
    platform: str,
    path_name: str,
    seed: int,
) -> dict[str, Any]:
    background = deterministic_prompts(
        path["background_sequences"], path["background_prompt_tokens"], pool
    )
    background_params = path_sampling_params(
        path, seed, path["background_output_tokens"]
    )
    requests = add_requests(
        llm,
        background,
        background_params,
        platform=platform,
        path_name=path_name,
        role="background",
        expected_tokens=path["background_output_tokens"],
    )
    finished: dict[str, list[int]] = {}
    deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS
    started_ns = time.perf_counter_ns()
    retain_finished(llm.llm_engine.step(), requests, finished)
    background_unfinished = llm.llm_engine.has_unfinished_requests()
    if not background_unfinished:
        raise RuntimeError("background requests finished before mixed admission")

    foreground = deterministic_prompts(1, path["foreground_prompt_tokens"], pool)
    foreground_requests = add_requests(
        llm,
        foreground,
        path_sampling_params(path, seed, path["foreground_output_tokens"]),
        platform=platform,
        path_name=path_name,
        role="foreground",
        expected_tokens=path["foreground_output_tokens"],
    )
    requests.update(foreground_requests)
    rows = finish_requests(llm, requests, finished, deadline)
    return {
        "request_count": len(rows),
        "output_rows": rows,
        "output_digest": output_digest(rows),
        "wall_ms": (time.perf_counter_ns() - started_ns) / 1_000_000,
        "background_unfinished_at_foreground_admission": background_unfinished,
        "all_requests_finished": True,
    }


def run_path(
    llm: Any,
    path: dict[str, Any],
    pool: list[int],
    *,
    platform: str,
    path_name: str,
    seed: int,
) -> dict[str, Any]:
    if path["kind"] == "standard":
        return run_standard_path(
            llm, path, pool, platform=platform, path_name=path_name, seed=seed
        )
    if path["kind"] == "mixed":
        return run_mixed_path(
            llm, path, pool, platform=platform, path_name=path_name, seed=seed
        )
    raise ValueError(f"unknown path kind: {path['kind']}")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    config = load_quick_config(args.config)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    validate_freeze(args, config, freeze)
    model = args.model or Path(config["model"]["path"])
    configure_environment(args.platform, config["platforms"][args.platform]["physical_devices"])
    model_hashes = validate_model_asset(model, config)

    import torch
    import vllm
    from vllm import LLM

    if args.platform == "910b":
        import torch_npu  # noqa: F401

    audit = runtime_audit(args.platform, torch, vllm)
    mismatches = runtime_mismatches(args.platform, audit, config)
    payload: dict[str, Any] = {
        "schema": "memseal.ms_g0d_quick.run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "failure_stage": None,
        "repository_commit": repository_commit(),
        "config_sha256": sha256(args.config),
        "freeze_sha256": sha256(args.freeze),
        "platform": args.platform,
        "plan": "compiled",
        "restart_index": args.restart_index,
        "tensor_parallel_size": 4,
        "physical_devices": config["platforms"][args.platform]["physical_devices"],
        "model_asset_id": config["model"]["asset_id"],
        "model_path": str(model),
        "model_hashes": model_hashes,
        "runtime": {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            **audit,
        },
        "runtime_mismatches": mismatches,
        "trace_order": list(TRACE_ORDER),
        "engine_initialized": False,
        "workload_completed": False,
        "shutdown_complete": False,
        "paths": [],
    }
    llm = None
    active_trace: DispatchTrace | None = None
    try:
        if mismatches:
            raise RuntimeError("frozen runtime mismatch: " + "; ".join(mismatches))
        engine = config["engine"]
        kwargs: dict[str, Any] = {
            "model": str(model),
            "tensor_parallel_size": 4,
            "dtype": config["model"]["dtype"],
            "seed": config["seed"],
            "max_model_len": engine["max_model_len"],
            "max_num_seqs": engine["max_num_seqs"],
            "max_num_batched_tokens": engine["max_num_batched_tokens"],
            "kv_cache_memory_bytes": engine["kv_cache_memory_bytes"],
            "enable_prefix_caching": engine["enable_prefix_caching"],
            "async_scheduling": engine["async_scheduling"],
            "disable_custom_all_reduce": engine["disable_custom_all_reduce"],
            "disable_log_stats": True,
            "cudagraph_metrics": False,
            "enforce_eager": engine["enforce_eager"],
        }
        additional = config["platforms"][args.platform].get("additional_config")
        if additional is not None:
            kwargs["additional_config"] = additional

        payload["failure_stage"] = "engine_init"
        init_started = time.perf_counter_ns()
        llm = LLM(**kwargs)
        payload["model_init_ms"] = (time.perf_counter_ns() - init_started) / 1_000_000
        payload["process_to_ready_ms"] = (
            time.perf_counter_ns() - PROCESS_STARTED_NS
        ) / 1_000_000
        payload["engine_initialized"] = True
        payload["resolved_runtime"] = resolved_runtime(llm)

        initial_memory, initial_error = safe_collective(
            llm, worker_resource_snapshot, timeout=120
        )
        payload["memory_after_init"] = initial_memory
        payload["memory_after_init_error"] = initial_error
        payload["post_init_runtime_mismatches"] = post_init_device_mismatches(
            args.platform, initial_memory, config
        )
        pool = token_pool(llm, config["prompt_pattern"])
        payload["prompt_token_pool"] = pool

        payload["failure_stage"] = "workload"
        for path_name in TRACE_ORDER:
            path_config = config["paths"][path_name]
            record: dict[str, Any] = {
                "path_name": path_name,
                "kind": path_config["kind"],
                "status": "running",
            }
            payload["paths"].append(record)
            if llm.llm_engine.has_unfinished_requests():
                raise RuntimeError(f"{path_name} started while the engine was not idle")

            before, before_error = safe_collective(
                llm, worker_resource_snapshot, timeout=120
            )
            record["memory_before_by_rank"] = before
            record["memory_before_error"] = before_error
            reset_rows, reset_error = safe_collective(
                llm, worker_reset_peak_memory, timeout=120
            )
            record["peak_reset"] = reset_rows
            record["peak_reset_error"] = reset_error
            if before_error is not None or reset_error is not None:
                raise RuntimeError(f"{path_name} memory observer failed before execution")

            active_trace = DispatchTrace(llm)
            path_error: Exception | None = None
            try:
                record["dispatch_start"] = active_trace.start()
                record.update(
                    run_path(
                        llm,
                        path_config,
                        pool,
                        platform=args.platform,
                        path_name=path_name,
                        seed=config["seed"],
                    )
                )
            except Exception as error:  # noqa: BLE001 - retain failed path
                path_error = error
                record["error_type"] = type(error).__name__
                record["error"] = str(error)
                record["traceback"] = traceback.format_exc()
            finally:
                if active_trace.active:
                    try:
                        record["worker_dispatch"] = active_trace.finish()
                    except Exception as error:  # noqa: BLE001
                        record["dispatch_finish_error"] = f"{type(error).__name__}: {error}"
                        if path_error is None:
                            path_error = error
                active_trace = None
                after, after_error = safe_collective(
                    llm, worker_resource_snapshot, timeout=120
                )
                record["memory_after_by_rank"] = after
                record["memory_after_error"] = after_error

            record["engine_idle_after_path"] = not llm.llm_engine.has_unfinished_requests()
            record["compiled_graph_dispatch_observed"] = graph_dispatch_observed(
                record.get("worker_dispatch")
            )
            record["checks"] = {
                "memory_before_valid": before_error is None
                and memory_snapshot_valid(before, REQUIRED_MEMORY_FIELDS),
                "peak_reset_valid": reset_error is None
                and isinstance(reset_rows, list)
                and len(reset_rows) == 4
                and all(row.get("reset") for row in reset_rows),
                "memory_after_valid": record["memory_after_error"] is None
                and memory_snapshot_valid(record["memory_after_by_rank"], REQUIRED_MEMORY_FIELDS),
                "rank_dispatch_identity": dispatch_rows_identical(
                    record.get("worker_dispatch")
                ),
                "output_complete": record.get("all_requests_finished") is True,
                "engine_idle_after_path": record["engine_idle_after_path"],
            }
            if path_error is not None:
                record["status"] = "failed"
                raise RuntimeError(f"{path_name} failed: {path_error}") from path_error
            if not all(record["checks"].values()):
                record["status"] = "failed"
                raise RuntimeError(f"{path_name} failed its observer/correctness checks")
            record["status"] = "success"

        payload["workload_completed"] = True
        payload["engine_idle_after_run"] = not llm.llm_engine.has_unfinished_requests()
        payload["checks"] = {
            "post_init_device_identity": not payload["post_init_runtime_mismatches"],
            "initial_memory_valid": initial_error is None
            and memory_snapshot_valid(initial_memory, REQUIRED_MEMORY_FIELDS),
            "explicit_kv_bytes_retained": payload["resolved_runtime"]["cache"][
                "configured_kv_cache_memory_bytes"
            ]
            == engine["kv_cache_memory_bytes"],
            "fixed_trace_order_completed": [row["path_name"] for row in payload["paths"]]
            == list(TRACE_ORDER)
            and all(row["status"] == "success" for row in payload["paths"]),
            "compiled_graph_observed_somewhere": any(
                row.get("compiled_graph_dispatch_observed") is True
                for row in payload["paths"]
            ),
            "engine_idle_after_run": payload["engine_idle_after_run"],
        }
        if not all(payload["checks"].values()):
            raise RuntimeError("run-level qualification checks failed")
        payload["failure_stage"] = None
        payload["status"] = "success"
    except Exception as error:  # noqa: BLE001 - always retain failed-run JSON
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
        payload["traceback"] = traceback.format_exc()
    finally:
        if active_trace is not None:
            try:
                active_trace.stop()
            except Exception as error:  # noqa: BLE001
                payload["trace_cleanup_error"] = f"{type(error).__name__}: {error}"
                payload["status"] = "failed"
                payload["failure_stage"] = "trace_cleanup"
        if llm is not None:
            shutdown = getattr(llm.llm_engine, "shutdown", None)
            if callable(shutdown):
                shutdown_started = time.perf_counter_ns()
                try:
                    shutdown()
                    payload["shutdown_complete"] = True
                    payload["shutdown_mode"] = "engine_method"
                except Exception as error:  # noqa: BLE001
                    payload["shutdown_error_type"] = type(error).__name__
                    payload["shutdown_error"] = str(error)
                    payload["status"] = "failed"
                    payload["failure_stage"] = "shutdown"
                payload["shutdown_ms"] = (
                    time.perf_counter_ns() - shutdown_started
                ) / 1_000_000
            else:
                payload["shutdown_complete"] = True
                payload["shutdown_mode"] = "fresh_process_exit"
        payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        payload["total_process_ms"] = (
            time.perf_counter_ns() - PROCESS_STARTED_NS
        ) / 1_000_000
    return jsonable(payload)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    try:
        payload = execute(args)
    except Exception as error:  # noqa: BLE001 - preserve preflight failures
        payload = {
            "schema": "memseal.ms_g0d_quick.run.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "failure_stage": "preflight_validation",
            "repository_commit": repository_commit(required=False),
            "platform": args.platform,
            "plan": "compiled",
            "restart_index": args.restart_index,
            "engine_initialized": False,
            "workload_completed": False,
            "shutdown_complete": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
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
                "platform": args.platform,
                "output": str(args.output),
            }
        )
    )
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
