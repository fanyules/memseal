#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROCESS_STARTED_NS = time.perf_counter_ns()
REQUEST_COUNTER = 0
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
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one frozen MS-Q0 TP4 cell")
    parser.add_argument("--platform", choices=PLATFORMS, required=True)
    parser.add_argument("--plan", choices=PLANS, required=True)
    parser.add_argument("--restart-index", type=int, default=0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "ms_q0.json",
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "ms_q0" / "MS_Q0_FREEZE.json",
    )
    return parser.parse_args()


def repository_commit(*, required: bool = True) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    if required:
        raise RuntimeError("MS-Q0 must run from a committed Git checkout")
    return None


def commit_is_ancestor(commit: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
        check=False,
    ).returncode == 0


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "name") and hasattr(value, "value"):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def configure_environment(platform: str, physical_devices: list[int]) -> None:
    visible = ",".join(str(device) for device in physical_devices)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    if platform == "a100":
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    else:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible


def validate_freeze(
    args: argparse.Namespace,
    config: dict[str, Any],
    freeze: dict[str, Any],
) -> None:
    if args.restart_index != 0:
        raise ValueError("MS-Q0 has exactly one fresh process per cell")
    if freeze.get("status") != "frozen":
        raise ValueError("MS-Q0 freeze is absent or not frozen")
    if freeze.get("config_sha256") != sha256(args.config):
        raise ValueError("active config differs from the public MS-Q0 freeze")
    protocol = REPOSITORY_ROOT / freeze["protocol_path"]
    if sha256(protocol) != freeze.get("protocol_sha256"):
        raise ValueError("active protocol differs from the public MS-Q0 freeze")
    for relative_path, expected_hash in freeze.get("source_sha256", {}).items():
        if sha256(REPOSITORY_ROOT / relative_path) != expected_hash:
            raise ValueError(f"frozen source changed: {relative_path}")
    if not freeze.get("source_sha256"):
        raise ValueError("MS-Q0 freeze does not contain source hashes")
    commit = freeze.get("pre_output_repository_commit")
    if not isinstance(commit, str) or not commit_is_ancestor(commit):
        raise ValueError("MS-Q0 pre-output commit is not an ancestor of this checkout")
    matrix = {
        (row["platform"], row["plan"])
        for row in freeze.get("matrix", [])
        if row.get("fresh_processes") == 1
    }
    if (args.platform, args.plan) not in matrix:
        raise ValueError("requested cell is outside the frozen MS-Q0 matrix")
    if config["platforms"][args.platform]["tp"] != 4:
        raise ValueError("requested lane no longer has the frozen TP4 contract")


def validate_model_asset(model: Path, config: dict[str, Any]) -> dict[str, str]:
    config_path = model / "config.json"
    index_path = model / "model.safetensors.index.json"
    observed = {
        "config_sha256": sha256(config_path),
        "index_sha256": sha256(index_path),
    }
    expected = config["model"]
    for name, value in observed.items():
        if value != expected[name]:
            raise ValueError(f"model asset differs from frozen {name}")
    return observed


def version_info(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def find_cann_install_info() -> Path:
    matches = sorted(Path("/usr/local/Ascend").glob("**/ascend_toolkit_install.info"))
    if not matches:
        raise FileNotFoundError("ascend_toolkit_install.info is absent")
    return matches[0]


def runtime_audit(platform: str, torch: Any, vllm: Any) -> dict[str, Any]:
    visible_variable = (
        "CUDA_VISIBLE_DEVICES" if platform == "a100" else "ASCEND_RT_VISIBLE_DEVICES"
    )
    visible_devices = [
        item for item in os.environ.get(visible_variable, "").split(",") if item
    ]
    if platform == "a100":
        return {
            "torch": str(torch.__version__),
            "vllm": str(vllm.__version__).split("+", 1)[0],
            "cuda": str(torch.version.cuda),
            "visible_device_count": len(visible_devices),
            "visible_device_source": visible_variable,
        }
    driver = version_info(Path("/usr/local/Ascend/driver/version.info"))
    firmware = version_info(Path("/usr/local/Ascend/firmware/version.info"))
    cann_path = find_cann_install_info()
    cann = version_info(cann_path)
    return {
        "torch": str(torch.__version__),
        "torch_npu": package_version("torch-npu"),
        "vllm": str(vllm.__version__).split("+", 1)[0],
        "vllm_ascend": package_version("vllm-ascend"),
        "driver": driver.get("Version"),
        "driver_package": driver.get("package_version"),
        "firmware": firmware.get("Version"),
        "firmware_package": firmware.get("package_version"),
        "cann": cann.get("version"),
        "cann_install_info_path": str(cann_path),
        "visible_device_count": len(visible_devices),
        "visible_device_source": visible_variable,
    }


def runtime_mismatches(
    platform: str,
    audit: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    expected = config["platforms"][platform]["expected"]
    mismatches: list[str] = []
    if audit.get("visible_device_count") != 4:
        mismatches.append(
            f"visible_device_count: expected 4, observed {audit.get('visible_device_count')}"
        )
    for name, wanted in expected.items():
        if name == "device_name_contains":
            continue
        observed = audit.get(name)
        if name in {"vllm", "vllm_ascend"} and observed is not None:
            observed = str(observed).split("+", 1)[0]
        if observed != wanted:
            mismatches.append(f"{name}: expected {wanted}, observed {audit.get(name)}")
    return mismatches


def post_init_device_mismatches(
    platform: str,
    rows: Any,
    config: dict[str, Any],
) -> list[str]:
    if not isinstance(rows, list) or len(rows) != 4:
        return [f"expected four post-init worker snapshots, observed {type(rows).__name__}"]
    by_rank = {
        row.get("rank"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("rank"), int)
    }
    if set(by_rank) != {0, 1, 2, 3}:
        return [f"post-init worker ranks differ from 0-3: {sorted(by_rank)}"]
    needle = config["platforms"][platform]["expected"]["device_name_contains"].lower()
    mismatches = []
    for rank in range(4):
        observed = str(by_rank[rank].get("device_name", ""))
        if needle not in observed.lower():
            mismatches.append(
                f"rank {rank} device name expected to contain {needle!r}, observed {observed!r}"
            )
    return mismatches


# Adapted from the audited GraphLease/KneeTP local probes. The callable runs on
# each rank, synchronizes its accelerator, and keeps driver and allocator values
# separate rather than summing nested memory domains.
def worker_reset_peak_memory(worker: Any) -> dict[str, Any]:
    import torch

    device = worker.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "npu":
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
    else:
        raise RuntimeError(f"unsupported worker device: {device}")
    return {"rank": int(getattr(worker, "rank", -1)), "reset": True}


def worker_resource_snapshot(worker: Any) -> dict[str, Any]:
    import torch

    device = worker.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        device_name = torch.cuda.get_device_name(device)
    elif device.type == "npu":
        torch.npu.synchronize()
        try:
            free_bytes, total_bytes = torch.npu.mem_get_info()
        except AttributeError:
            total_bytes = torch.npu.get_device_properties(0).total_memory
            free_bytes = total_bytes - torch.npu.memory_reserved()
        allocated = torch.npu.memory_allocated()
        reserved = torch.npu.memory_reserved()
        peak_allocated = torch.npu.max_memory_allocated()
        peak_reserved = torch.npu.max_memory_reserved()
        device_name = torch.npu.get_device_name()
    else:
        raise RuntimeError(f"unsupported worker device: {device}")

    compilation_counter: dict[str, int] | None = None
    try:
        from vllm.compilation.counter import compilation_counter as counter

        compilation_counter = {
            name: int(getattr(counter, name))
            for name in dir(counter)
            if name.startswith("num_") and isinstance(getattr(counter, name), int)
        }
    except (ImportError, AttributeError):
        pass

    acl_wrapper_count = None
    acl_graph_entry_count = None
    if device.type == "npu":
        try:
            from vllm_ascend.compilation.acl_graph import _acl_graph_wrappers

            wrappers = list(_acl_graph_wrappers)
            acl_wrapper_count = len(wrappers)
            acl_graph_entry_count = sum(
                len(wrapper.concrete_aclgraph_entries) for wrapper in wrappers
            )
        except (ImportError, AttributeError):
            pass

    kv_cache_bytes = None
    kv_cache_bytes_error = None
    try:
        cache = worker.vllm_config.cache_config
        num_blocks = int(cache.num_gpu_blocks or 0)
        specs = worker.model_runner.get_kv_cache_spec()
        kv_cache_bytes = num_blocks * sum(
            int(spec.page_size_bytes) for spec in specs.values()
        )
    except (AttributeError, TypeError, ValueError) as error:
        kv_cache_bytes_error = f"{type(error).__name__}: {error}"

    return {
        "rank": int(getattr(worker, "rank", -1)),
        "device_type": device.type,
        "device_name": str(device_name),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(allocated),
        "reserved_bytes": int(reserved),
        "peak_allocated_bytes": int(peak_allocated),
        "peak_reserved_bytes": int(peak_reserved),
        "kv_cache_bytes": kv_cache_bytes,
        "kv_cache_bytes_error": kv_cache_bytes_error,
        "compilation_counter": compilation_counter,
        "acl_wrapper_count": acl_wrapper_count,
        "acl_graph_entry_count": acl_graph_entry_count,
    }


def worker_start_dispatch_trace(worker: Any) -> dict[str, Any]:
    model_runner = worker.model_runner
    marker = "_memseal_dispatch_trace_state"
    if hasattr(worker, marker):
        raise RuntimeError("worker dispatch trace is already active")
    rows: list[dict[str, Any]] = []
    method_name = "_determine_batch_execution_and_padding"
    if hasattr(model_runner, method_name):
        original = getattr(model_runner, method_name)

        def traced_v1_dispatch(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            graph_stat = result[4]
            if graph_stat is not None:
                unpadded = int(graph_stat.num_unpadded_tokens)
                padded = int(graph_stat.num_padded_tokens)
                runtime_mode = str(graph_stat.runtime_mode)
            else:
                value = kwargs.get("num_tokens", args[0] if args else None)
                if value is None:
                    raise RuntimeError("dispatch trace cannot resolve V1 tokens")
                unpadded = int(value)
                padded = int(result[1].num_tokens)
                runtime_mode = str(result[0])
            rows.append(
                {
                    "num_unpadded_tokens": unpadded,
                    "num_padded_tokens": padded,
                    "num_paddings": padded - unpadded,
                    "runtime_mode": runtime_mode,
                }
            )
            return result

        target = model_runner
        name = method_name
        source = "runner_method_v1"
        replacement = traced_v1_dispatch
    else:
        module = sys.modules[type(model_runner).__module__]
        name = "dispatch_cg_and_sync_dp"
        original = getattr(module, name)

        def traced_v2_dispatch(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            value = kwargs.get("num_toks", args[2] if len(args) > 2 else None)
            if value is None:
                raise RuntimeError("dispatch trace cannot resolve V2 tokens")
            unpadded = int(value)
            batch_descriptor = result[0]
            padded = int(batch_descriptor.num_tokens)
            rows.append(
                {
                    "num_unpadded_tokens": unpadded,
                    "num_padded_tokens": padded,
                    "num_paddings": padded - unpadded,
                    "runtime_mode": str(batch_descriptor.cg_mode),
                }
            )
            return result

        target = module
        source = "module_function_v2"
        replacement = traced_v2_dispatch
    setattr(target, name, replacement)
    setattr(
        worker,
        marker,
        {
            "target": target,
            "name": name,
            "original": original,
            "rows": rows,
            "source": source,
        },
    )
    return {"rank": int(getattr(worker, "rank", -1)), "active": True, "source": source}


def worker_finish_dispatch_trace(worker: Any) -> dict[str, Any]:
    marker = "_memseal_dispatch_trace_state"
    state = getattr(worker, marker, None)
    rows = list(state["rows"]) if state is not None else []
    if state is not None:
        setattr(state["target"], state["name"], state["original"])
        delattr(worker, marker)
    return {
        "rank": int(getattr(worker, "rank", -1)),
        "active": state is not None,
        "source": state["source"] if state is not None else None,
        "rows": rows,
    }


class DispatchTrace:
    def __init__(self, llm: Any):
        self.llm = llm
        self.active = False

    def start(self) -> list[dict[str, Any]]:
        if self.active:
            raise RuntimeError("dispatch trace is already active")
        rows = self.llm.collective_rpc(worker_start_dispatch_trace, timeout=60)
        self.active = True
        if len(rows) != 4 or not all(item.get("active") for item in rows):
            self.stop()
            raise RuntimeError("dispatch trace did not start on all four ranks")
        return rows

    def finish(self) -> list[dict[str, Any]]:
        if not self.active:
            raise RuntimeError("dispatch trace is not active")
        rows = self.llm.collective_rpc(worker_finish_dispatch_trace, timeout=60)
        self.active = False
        return rows

    def stop(self) -> None:
        if self.active:
            self.llm.collective_rpc(worker_finish_dispatch_trace, timeout=60)
            self.active = False


def token_pool(llm: Any, pattern: str) -> list[int]:
    values = list(llm.get_tokenizer().encode(pattern, add_special_tokens=False))
    if not values:
        raise RuntimeError("prompt pattern produced no tokens")
    return [int(value) for value in values]


def deterministic_prompts(count: int, length: int, pool: list[int]) -> list[dict[str, list[int]]]:
    row = [pool[index % len(pool)] for index in range(length)]
    return [{"prompt_token_ids": list(row)} for _ in range(count)]


def sampling_params(config: dict[str, Any], seed: int) -> Any:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    workload = config["workload"]
    params = SamplingParams(
        temperature=workload["temperature"],
        top_p=workload["top_p"],
        ignore_eos=workload["ignore_eos"],
        max_tokens=workload["output_tokens"],
        detokenize=False,
        seed=seed,
    )
    params.output_kind = RequestOutputKind.CUMULATIVE
    return params


def run_wave(
    llm: Any,
    config: dict[str, Any],
    pool: list[int],
    *,
    platform: str,
    plan: str,
    wave_index: int,
) -> dict[str, Any]:
    global REQUEST_COUNTER

    if llm.llm_engine.has_unfinished_requests():
        raise RuntimeError("wave started while the engine was not idle")
    workload = config["workload"]
    request_ids: list[str] = []
    for position, prompt in enumerate(
        deterministic_prompts(
            workload["num_sequences"], workload["prompt_tokens"], pool
        )
    ):
        request_id = f"memseal-{platform}-{plan}-{wave_index}-{REQUEST_COUNTER}"
        REQUEST_COUNTER += 1
        llm.llm_engine.add_request(
            request_id,
            prompt,
            sampling_params(config, config["seed"]),
        )
        request_ids.append(request_id)
    positions = {request_id: index for index, request_id in enumerate(request_ids)}
    finished: dict[str, list[int]] = {}
    deadline = time.monotonic() + config["execution"]["timeout_seconds"]
    started_ns = time.perf_counter_ns()
    while llm.llm_engine.has_unfinished_requests():
        if time.monotonic() > deadline:
            raise TimeoutError(f"wave {wave_index} exceeded the frozen timeout")
        for output in llm.llm_engine.step():
            request_id = str(output.request_id)
            if request_id not in positions:
                raise RuntimeError(f"unexpected request ID returned: {request_id}")
            if output.finished:
                if not output.outputs:
                    raise RuntimeError("finished request had no sequence output")
                finished[request_id] = [int(token) for token in output.outputs[0].token_ids]
    if set(finished) != set(request_ids):
        raise RuntimeError("wave did not finish every request")
    rows = []
    for request_id, position in sorted(positions.items(), key=lambda item: item[1]):
        tokens = finished[request_id]
        if len(tokens) != workload["output_tokens"]:
            raise RuntimeError("ignore-EOS request produced the wrong output length")
        rows.append({"position": position, "token_ids": tokens})
    return {
        "wave_index": wave_index,
        "request_count": len(rows),
        "wall_ms": (time.perf_counter_ns() - started_ns) / 1_000_000,
        "output_rows": rows,
        "output_digest": output_digest(rows),
    }


def resolved_runtime(llm: Any) -> dict[str, Any]:
    config = llm.llm_engine.vllm_config
    compilation = config.compilation_config
    cache = config.cache_config
    num_blocks = int(cache.num_gpu_blocks or 0)
    block_size = int(cache.block_size)
    return {
        "compilation": {
            "mode": str(compilation.mode),
            "backend": compilation.backend,
            "cudagraph_mode": str(compilation.cudagraph_mode),
            "cudagraph_capture_sizes": list(compilation.cudagraph_capture_sizes or ()),
            "max_cudagraph_capture_size": compilation.max_cudagraph_capture_size,
        },
        "parallel": {
            "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
            "disable_custom_all_reduce": config.parallel_config.disable_custom_all_reduce,
        },
        "scheduler": {
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
            "async_scheduling": config.scheduler_config.async_scheduling,
        },
        "cache": {
            "num_gpu_blocks": num_blocks,
            "block_size": block_size,
            "kv_token_capacity": num_blocks * block_size,
            "configured_kv_cache_memory_bytes": getattr(
                cache, "kv_cache_memory_bytes", None
            ),
            "gpu_memory_utilization": cache.gpu_memory_utilization,
        },
        "attention_backend": str(config.attention_config.backend),
    }


def safe_collective(
    llm: Any,
    function: Callable[[Any], Any],
    *,
    timeout: int = 60,
) -> tuple[list[Any] | None, str | None]:
    try:
        return llm.collective_rpc(function, timeout=timeout), None
    except Exception as error:  # noqa: BLE001 - retain observer failure in JSON
        return None, f"{type(error).__name__}: {error}"


def execute(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
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
        "schema": "memseal.ms_q0.run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "failure_stage": None,
        "repository_commit": repository_commit(),
        "config_sha256": sha256(args.config),
        "freeze_sha256": sha256(args.freeze),
        "platform": args.platform,
        "plan": args.plan,
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
        "engine_initialized": False,
        "workload_completed": False,
        "shutdown_complete": False,
    }
    llm = None
    trace: DispatchTrace | None = None
    try:
        if mismatches:
            raise RuntimeError("frozen runtime mismatch: " + "; ".join(mismatches))
        engine = config["engine"]
        plan = config["plans"][args.plan]
        kwargs: dict[str, Any] = {
            "model": str(model),
            "tensor_parallel_size": 4,
            "dtype": config["model"]["dtype"],
            "seed": config["seed"],
            "max_model_len": engine["max_model_len"],
            "max_num_seqs": engine["max_num_seqs"],
            "max_num_batched_tokens": engine["max_num_batched_tokens"],
            "gpu_memory_utilization": engine["gpu_memory_utilization"],
            "enable_prefix_caching": engine["enable_prefix_caching"],
            "async_scheduling": engine["async_scheduling"],
            "disable_custom_all_reduce": engine["disable_custom_all_reduce"],
            "disable_log_stats": True,
            "cudagraph_metrics": False,
            "enforce_eager": plan["enforce_eager"],
        }
        if plan["compilation_config"] is not None:
            kwargs["compilation_config"] = plan["compilation_config"]
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
        pool = token_pool(llm, config["workload"]["prompt_pattern"])
        payload["prompt_token_pool"] = pool

        trace = DispatchTrace(llm)
        dispatch_start_error = None
        try:
            payload["dispatch_start"] = trace.start()
        except Exception as error:  # noqa: BLE001 - retain external runtime failure
            dispatch_start_error = f"{type(error).__name__}: {error}"
        payload["dispatch_start_error"] = dispatch_start_error

        payload["failure_stage"] = "workload"
        waves = []
        for wave_index in range(config["workload"]["waves"]):
            reset_rows, reset_error = safe_collective(llm, worker_reset_peak_memory)
            wave = run_wave(
                llm,
                config,
                pool,
                platform=args.platform,
                plan=args.plan,
                wave_index=wave_index,
            )
            memory_rows, memory_error = safe_collective(llm, worker_resource_snapshot)
            wave["peak_reset"] = reset_rows
            wave["peak_reset_error"] = reset_error
            wave["memory_by_rank"] = memory_rows
            wave["memory_error"] = memory_error
            waves.append(wave)
        payload["waves"] = waves
        payload["workload_completed"] = True

        worker_dispatch = None
        dispatch_finish_error = None
        if trace.active:
            try:
                worker_dispatch = trace.finish()
            except Exception as error:  # noqa: BLE001 - retain external runtime failure
                dispatch_finish_error = f"{type(error).__name__}: {error}"
        payload["worker_dispatch"] = worker_dispatch
        payload["dispatch_finish_error"] = dispatch_finish_error
        payload["engine_idle_after_run"] = not llm.llm_engine.has_unfinished_requests()

        wave_digests = [wave["output_digest"] for wave in waves]
        required_fields = tuple(config["qualification"]["required_memory_fields"])
        memory_rows = [payload["memory_after_init"]] + [
            wave["memory_by_rank"] for wave in waves
        ]
        reset_ok = all(
            wave["peak_reset_error"] is None
            and isinstance(wave["peak_reset"], list)
            and len(wave["peak_reset"]) == 4
            and all(item.get("reset") for item in wave["peak_reset"])
            for wave in waves
        )
        checks = {
            "all_24_requests_finished": sum(wave["request_count"] for wave in waves) == 24,
            "within_wave_identical_requests": identical_request_outputs(waves),
            "within_run_determinism": len(set(wave_digests)) == 1,
            "rank_dispatch_identity": dispatch_rows_identical(worker_dispatch),
            "eager_only_dispatch": (
                eager_only_dispatch(worker_dispatch) if args.plan == "eager" else True
            ),
            "compiled_graph_dispatch": (
                graph_dispatch_observed(worker_dispatch)
                if args.plan == "compiled"
                else True
            ),
            "common_memory_observability": reset_ok
            and all(memory_snapshot_valid(rows, required_fields) for rows in memory_rows),
            "post_init_device_identity": not payload["post_init_runtime_mismatches"],
            "resolved_kv_and_graph_config_retained": (
                payload["resolved_runtime"]["cache"]["kv_token_capacity"] > 0
                and bool(payload["resolved_runtime"]["compilation"])
            ),
            "engine_idle_after_run": payload["engine_idle_after_run"],
        }
        payload["checks"] = checks
        payload["wave_output_digests"] = wave_digests
        payload["canonical_output_digest"] = wave_digests[0] if wave_digests else None
        payload["failure_stage"] = None
        payload["status"] = "success"
    except Exception as error:  # noqa: BLE001 - emit a failed-run artifact
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
        payload["traceback"] = traceback.format_exc()
    finally:
        if trace is not None:
            try:
                trace.stop()
            except Exception as error:  # noqa: BLE001 - retain cleanup failure
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
                except Exception as error:  # noqa: BLE001 - retain shutdown failure
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
    except Exception as error:  # noqa: BLE001 - always emit a failed-run artifact
        payload = {
            "schema": "memseal.ms_q0.run.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "failure_stage": "preflight_validation",
            "repository_commit": repository_commit(required=False),
            "platform": args.platform,
            "plan": args.plan,
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
                "plan": args.plan,
                "output": str(args.output),
            }
        )
    )
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
