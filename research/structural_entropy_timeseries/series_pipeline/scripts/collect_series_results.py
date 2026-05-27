#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

METRIC_RE = re.compile(r"SERIES_METRIC\s+(.*)$")
KV_RE = re.compile(r"([a-zA-Z0-9_]+)=([^\s]+)")


def ssh_read(worker: dict, remote_cmd: str) -> str:
    ssh = worker.get("ssh", {})
    key = str(Path(ssh.get("key", "~/.ssh/id_rsa")).expanduser())
    args = [
        "ssh",
        "-i",
        key,
        "-p",
        str(ssh.get("port", 22)),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=10",
        f"{ssh.get('user', 'root')}@{ssh['host']}",
        remote_cmd,
    ]
    p = subprocess.run(args, capture_output=True, text=True)
    return p.stdout.strip()


def parse_metric(line: str) -> dict:
    m = METRIC_RE.search(line)
    if not m:
        return {}
    payload = m.group(1)
    out = {}
    for k, v in KV_RE.findall(payload):
        out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect SERIES_METRIC outputs from run logs")
    ap.add_argument("--root", default="/home/ubuntu/research-os")
    ap.add_argument("--project-id", default="structural_entropy_timeseries")
    ap.add_argument("--batch-prefix", default="series_v1")
    args = ap.parse_args()

    root = Path(args.root)
    run_queue = json.loads((root / "queues" / "run_queue.json").read_text())
    workers = json.loads(json.dumps({}))
    import yaml  # local import to avoid hard dependency for unrelated scripts

    workers_yaml = yaml.safe_load((root / "workers" / "workers.yaml").read_text())
    workers = {w["worker_id"]: w for w in workers_yaml.get("workers", [])}

    selected = [
        r for r in run_queue
        if r.get("project_id") == args.project_id
        and str(r.get("probe_id", "")).startswith(args.batch_prefix)
        and r.get("status") in {"done", "failed"}
    ]

    rows = []
    for run in selected:
        wid = run.get("worker_id", "")
        w = workers.get(wid)
        log_path = run.get("log_path", "")
        metric = {}
        if w and log_path and w.get("kind") == "ssh":
            out = ssh_read(w, f"grep -E 'SERIES_METRIC' {log_path} 2>/dev/null | tail -1")
            metric = parse_metric(out)
        row = {
            "run_id": run.get("id"),
            "probe_id": run.get("probe_id"),
            "status": run.get("status"),
            "worker_id": wid,
            "exit_code": run.get("exit_code", ""),
            "metric": metric,
        }
        rows.append(row)

    leaderboard = []
    for r in rows:
        m = r.get("metric") or {}
        if "mse" in m:
            try:
                mse = float(m["mse"])
            except ValueError:
                continue
            leaderboard.append(
                {
                    "model": m.get("model", r["probe_id"]),
                    "dataset": m.get("dataset", ""),
                    "horizon": m.get("horizon", ""),
                    "mse": mse,
                    "mae": float(m.get("mae", "nan")),
                    "run_id": r["run_id"],
                }
            )

    leaderboard.sort(key=lambda x: x["mse"])
    payload = {
        "project_id": args.project_id,
        "batch_prefix": args.batch_prefix,
        "num_runs": len(rows),
        "runs": rows,
        "leaderboard": leaderboard,
        "best": leaderboard[0] if leaderboard else None,
    }

    out_dir = root / "research" / args.project_id / "series_pipeline" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.batch_prefix}_leaderboard.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    ev_dir = root / "research" / args.project_id / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    ev_path = ev_dir / f"{args.batch_prefix}_summary.json"
    ev_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote {out_path}")
    print(f"Wrote {ev_path}")
    if payload["best"]:
        b = payload["best"]
        print(f"Best: model={b['model']} mse={b['mse']:.6f} mae={b['mae']}")
    else:
        print("No parsed metrics found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
