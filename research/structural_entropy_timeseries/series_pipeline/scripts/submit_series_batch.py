#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return p.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Submit a configured series batch into run_queue")
    ap.add_argument("--root", default="/home/ubuntu/research-os")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text())

    ros = root / "scripts" / "ros.py"
    project_id = cfg["project_id"]
    batch_id = cfg["batch_id"]

    print(f"Submitting batch={batch_id} project={project_id}")
    for c in cfg["candidates"]:
        cmd = [
            "python3",
            str(ros),
            "add-run",
            "--project-id",
            project_id,
            "--command",
            c["command"],
            "--probe-id",
            c["probe_id"],
            "--worker-pool",
            c.get("worker_pool", "vastai"),
            "--priority",
            str(c.get("priority", 10)),
            "--target-metric",
            cfg.get("metric", "mse"),
            "--target-direction",
            cfg.get("direction", "lower"),
        ]
        rid = run(cmd)
        print(f"  added {rid} probe={c['probe_id']}")

    print("Done. Dispatch with: python3 scripts/ros.py dispatch-runs --worker-pool vastai")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
