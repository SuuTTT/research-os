#!/usr/bin/env python3
"""Research OS CLI.

This CLI manages file-backed queues and project scaffolds. It is deliberately
small so the workflow stays inspectable before adding autonomous agents.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import textwrap
import uuid
import urllib.error
import urllib.request
import yaml
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("RESEARCH_OS_ROOT", str(Path(__file__).resolve().parent.parent)))
QUEUES = ROOT / "queues"
RESEARCH = ROOT / "research"
TEMPLATES = ROOT / "templates"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sid(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:7]}"


def read_json(path: Path):
    if not path.exists():
        return []
    return json.loads(path.read_text())


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def append_queue(name: str, item: dict):
    path = QUEUES / name
    q = read_json(path)
    q.append(item)
    write_json(path, q)


def project_dir(project_id: str) -> Path:
    return RESEARCH / project_id


def safe_project_id(s: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out


def cmd_init_project(args):
    project_id = args.project_id or safe_project_id(args.title)
    pdir = project_dir(project_id)
    for sub in [
        "deep_research",
        "benchmark/external",
        "benchmark/local",
        "probes",
        "evidence",
        "analysis",
        "blog",
        "paper/figures",
        "paper/tables",
        "artifacts",
    ]:
        (pdir / sub).mkdir(parents=True, exist_ok=True)

    project_yaml = TEMPLATES / "project" / "project.yaml"
    dst = pdir / "project.yaml"
    if not dst.exists():
        text = project_yaml.read_text()
        text = text.replace('project_id: ""', f'project_id: "{project_id}"')
        text = text.replace('title: ""', f'title: "{args.title}"')
        text = text.replace('idea: ""', f'idea: "{args.idea}"')
        text = text.replace('  claim: ""', f'  claim: "{args.target}"')
        text = text.replace('  blog_dir: ""', f'  blog_dir: "research/{project_id}/blog"')
        text = text.replace('  paper_dir: ""', f'  paper_dir: "research/{project_id}/paper"')
        dst.write_text(text)

    deep_req = pdir / "deep_research" / "request.md"
    if not deep_req.exists():
        req = (TEMPLATES / "agents" / "deep_research_prompt.md").read_text()
        deep_req.write_text(req.replace("<IDEA>", args.idea))

    paper = pdir / "paper" / "main.tex"
    if not paper.exists():
        tex = (TEMPLATES / "paper" / "main.tex").read_text()
        paper.write_text(tex.replace("PROJECT_TITLE", args.title))

    for name, content in {
        "benchmark/README.md": "# Benchmark\n\nTrack official repos, baseline reproduction, and local wrappers here.\n",
        "probes/README.md": "# Probes\n\nOne subfolder or launcher per probe.\n",
        "evidence/README.md": "# Evidence\n\nStore metric summaries, decisions, and links to artifacts here.\n",
        "analysis/README.md": "# Analysis\n\nMetric parsers, plots, and tables.\n",
    }.items():
        path = pdir / name
        if not path.exists():
            path.write_text(content)

    idea = {
        "id": sid("i"),
        "project_id": project_id,
        "status": "new",
        "priority": args.priority,
        "title": args.title,
        "hypothesis": args.idea,
        "metric": args.metric or "",
        "created_at": now(),
        "updated_at": now(),
        "claimed_by": "",
        "probe_specs": [],
        "evidence": [],
        "events": [{"at": now(), "event": "project_initialized"}],
    }
    append_queue("idea_queue.json", idea)
    print(project_id)
    print(f"project: {pdir}")
    print(f"idea: {idea['id']}")


def cmd_add_idea(args):
    item = {
        "id": sid("i"),
        "project_id": args.project_id,
        "status": "new",
        "priority": args.priority,
        "title": args.title,
        "hypothesis": args.hypothesis,
        "metric": args.metric or "",
        "created_at": now(),
        "updated_at": now(),
        "claimed_by": "",
        "probe_specs": [],
        "evidence": [],
        "events": [{"at": now(), "event": "idea_created"}],
    }
    append_queue("idea_queue.json", item)
    print(item["id"])


def cmd_list_ideas(_args):
    q = read_json(QUEUES / "idea_queue.json")
    q.sort(key=lambda x: (x.get("priority", 10), x.get("created_at", "")))
    for item in q:
        print(
            f"{item['id']} p{item.get('priority')} {item.get('status'):<14} "
            f"{item.get('project_id')} :: {item.get('title')}"
        )


def cmd_add_benchmark_task(args):
    item = {
        "id": sid("b"),
        "project_id": args.project_id,
        "status": "pending",
        "priority": args.priority,
        "title": args.title,
        "repo_url": args.repo_url or "",
        "command": args.command or "",
        "metric": args.metric or "",
        "pass_rule": args.pass_rule or "",
        "output_path": args.output_path or f"research/{args.project_id}/benchmark/reproduction.md",
        "created_at": now(),
        "updated_at": now(),
    }
    append_queue("benchmark_queue.json", item)
    print(item["id"])


def cmd_add_run(args):
    item = {
        "id": sid("r"),
        "project_id": args.project_id,
        "idea_id": args.idea_id or "",
        "benchmark_id": args.benchmark_id or "",
        "probe_id": args.probe_id or "",
        "status": "pending",
        "priority": args.priority,
        "worker_pool": args.worker_pool,
        "command": args.command,
        "env": args.env or "",
        "metric_parser": args.metric_parser or "",
        "target_metric": args.target_metric or "",
        "target_direction": args.target_direction,
        "created_at": now(),
        "updated_at": now(),
    }
    append_queue("run_queue.json", item)
    print(item["id"])


def cmd_add_publication(args):
    item = {
        "id": sid("pub"),
        "project_id": args.project_id,
        "status": "pending",
        "priority": args.priority,
        "kind": args.kind,
        "trigger": args.trigger,
        "source_evidence": args.source_evidence or [],
        "output_path": args.output_path,
        "created_at": now(),
        "updated_at": now(),
    }
    append_queue("publication_queue.json", item)
    print(item["id"])


def cmd_status(_args):
    for name in ["idea_queue.json", "benchmark_queue.json", "run_queue.json", "publication_queue.json"]:
        q = read_json(QUEUES / name)
        counts = {}
        for item in q:
            counts[item.get("status", "unknown")] = counts.get(item.get("status", "unknown"), 0) + 1
        print(f"{name}: {len(q)} {counts}")


# ── Worker helpers ────────────────────────────────────────────────────────────

def load_workers() -> list[dict]:
    path = ROOT / "workers" / "workers.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text())
    return data.get("workers", [])


def _ssh_args(w: dict, timeout: int = 10) -> list[str]:
    ssh = w.get("ssh", {})
    key = str(Path(ssh.get("key", "~/.ssh/id_rsa")).expanduser())
    return [
        "ssh", "-i", key,
        "-p", str(ssh.get("port", 22)),
        "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={min(timeout, 10)}",
        f"{ssh.get('user', 'root')}@{ssh['host']}",
    ]


def worker_ssh(w: dict, remote_cmd: str, timeout: int = 15, stream: bool = False) -> tuple[str, int]:
    args = _ssh_args(w, timeout) + [remote_cmd]
    if stream:
        result = subprocess.run(args, timeout=timeout)
        return "", result.returncode
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return result.stdout.strip(), result.returncode


# ── worker-status ─────────────────────────────────────────────────────────────

def cmd_worker_status(_args):
    workers = load_workers()
    if not workers:
        print("No workers configured.")
        return
    for w in workers:
        wid = w["worker_id"]
        kind = w.get("kind", "local")
        if not w.get("enabled", True):
            print(f"{wid:<26} DISABLED")
            continue
        h = w.get("health", {})
        try:
            if kind == "local":
                r = subprocess.run(h.get("heartbeat_command", "echo ok"), shell=True,
                                   capture_output=True, text=True, timeout=5)
                status = "alive" if r.returncode == 0 else "dead"
                print(f"{wid:<26} {kind:<8} {status}")
            elif kind == "ssh":
                _, rc = worker_ssh(w, h.get("heartbeat_command", "echo ok"), timeout=8)
                status = "alive" if rc == 0 else "dead"
                gpu, _ = worker_ssh(
                    w,
                    "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total"
                    " --format=csv,noheader 2>/dev/null | head -1 || echo n/a",
                    timeout=8,
                )
                print(f"{wid:<26} {kind:<8} {status:<8} gpu: {gpu}")
        except Exception as e:
            print(f"{wid:<26} {kind:<8} error: {e}")


# ── setup-worker ──────────────────────────────────────────────────────────────

def cmd_setup_worker(args):
    workers = load_workers()
    w = next((x for x in workers if x["worker_id"] == args.worker_id), None)
    if not w:
        ids = [x["worker_id"] for x in workers]
        print(f"Worker {args.worker_id!r} not found. Available: {ids}")
        return
    ws = w.get("workspace", "/root/research-worker")
    # Sync .env.local from control plane to worker when present.
    env_file = Path.home() / ".env.local"
    env_payload = ""
    if env_file.exists():
        env_payload = base64.b64encode(env_file.read_bytes()).decode()
    setup = f"""#!/bin/bash
set -e
echo '=== apt ==='
apt-get update -qq 2>&1 | tail -1
apt-get install -y git curl python3 python3-pip python3-venv -qq 2>&1 | tail -1
echo '=== workspace ==='
mkdir -p {ws}/runs/logs {ws}/runs/metrics {ws}/runs/checkpoints {ws}/runs/running
echo '=== env vars ==='
if [[ -n \"{env_payload}\" ]]; then
  echo {env_payload} | base64 -d > /root/.env.local
  chmod 600 /root/.env.local
fi
echo '=== GPU check ==='
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
echo '=== done ==='
"""
    encoded = base64.b64encode(setup.encode()).decode()
    print(f"Setting up {args.worker_id} ...")
    worker_ssh(w, f"echo {encoded} | base64 -d | bash", timeout=180, stream=True)


# ── dispatch-runs ─────────────────────────────────────────────────────────────

def _find_idle_worker(workers: list[dict], pool: str, busy_ids: set[str]) -> dict | None:
    for w in workers:
        if not w.get("enabled", True):
            continue
        if w["worker_id"] in busy_ids:
            continue
        if pool not in ("any", w.get("worker_pool", "default")):
            continue
        if w.get("kind") == "local":
            return w
        if w.get("kind") == "ssh":
            try:
                _, rc = worker_ssh(w, w.get("health", {}).get("heartbeat_command", "echo ok"), timeout=15)
                if rc == 0:
                    return w
            except Exception:
                continue
    return None


def cmd_dispatch_runs(args):
    q = read_json(QUEUES / "run_queue.json")
    workers = load_workers()
    busy_ids = {item.get("worker_id") for item in q if item.get("status") == "running"}
    pending = sorted(
        [x for x in q if x.get("status") == "pending"],
        key=lambda x: (x.get("priority", 10), x.get("created_at", "")),
    )
    if not pending:
        print("No pending runs.")
        return
    dispatched = 0
    for run in pending:
        pool = args.worker_pool if args.worker_pool != "any" else run.get("worker_pool", "default")
        if args.worker_pool == "any":
            pool = run.get("worker_pool", "default")
        w = _find_idle_worker(workers, pool, busy_ids)
        if not w:
            print(f"{run['id']}: no idle worker for pool={pool!r} — skipping")
            continue
        ws = w.get("workspace", "/root/research-worker")
        pid_dir = f"{ws}/runs/logs/{run['project_id']}/{run['id']}"
        log_path = f"{pid_dir}/stdout.log"
        run_script = (
            f"#!/bin/bash\n"
            f"if [[ -f /root/.env.local ]]; then set -a; source /root/.env.local; set +a; fi\n"
            f"if [[ -f /home/ubuntu/.env.local ]]; then set -a; source /home/ubuntu/.env.local; set +a; fi\n"
            f"{run.get('env', '') or ''}\n"
            f"{run.get('command', '')}\n"
            f"RET=$?\necho $RET > {pid_dir}/exit_code\nexit $RET\n"
        )
        enc = base64.b64encode(run_script.encode()).decode()
        setup_cmd = (
            f"mkdir -p {pid_dir} && "
            f"echo {enc} | base64 -d > /tmp/ros_{run['id']}.sh && "
            f"chmod +x /tmp/ros_{run['id']}.sh"
        )
        launch_cmd = f"setsid /tmp/ros_{run['id']}.sh </dev/null > {log_path} 2>&1 & echo $!"
        try:
            if w.get("kind") == "local":
                r = subprocess.run(setup_cmd, shell=True, capture_output=True, text=True, timeout=30)
                if r.returncode != 0:
                    print(f"{run['id']}: setup failed rc={r.returncode}")
                    continue
                r2 = subprocess.run(launch_cmd, shell=True, capture_output=True, text=True, timeout=15)
                pid, rc = r2.stdout.strip(), r2.returncode
            else:
                _, setup_rc = worker_ssh(w, setup_cmd, timeout=30)
                if setup_rc != 0:
                    print(f"{run['id']}: setup failed rc={setup_rc}")
                    continue
                pid, rc = worker_ssh(w, launch_cmd, timeout=15)
            if rc != 0:
                print(f"{run['id']}: dispatch failed rc={rc}")
                continue
            for item in q:
                if item["id"] == run["id"]:
                    item.update({
                        "status": "running", "worker_id": w["worker_id"],
                        "started_at": now(), "pid": pid,
                        "log_path": log_path, "updated_at": now(),
                    })
            write_json(QUEUES / "run_queue.json", q)
            busy_ids.add(w["worker_id"])
            print(f"dispatched {run['id']} → {w['worker_id']}  pid={pid}")
            dispatched += 1
        except Exception as e:
            print(f"{run['id']}: dispatch error: {e}")
    print(f"\n{dispatched}/{len(pending)} dispatched")


# ── check-runs ────────────────────────────────────────────────────────────────

def cmd_check_runs(_args):
    q = read_json(QUEUES / "run_queue.json")
    wmap = {w["worker_id"]: w for w in load_workers()}
    running = [x for x in q if x.get("status") == "running"]
    if not running:
        print("No running tasks.")
        return
    changed = False
    for run in running:
        wid = run.get("worker_id", "")
        pid = run.get("pid", "")
        w = wmap.get(wid)
        log_path = run.get("log_path", "")
        pid_dir = "/".join(log_path.rsplit("/", 1)[:1]) if log_path else ""
        if not w:
            print(f"{run['id']}: worker {wid!r} missing — marking stale")
            for item in q:
                if item["id"] == run["id"]:
                    item.update({"status": "stale", "updated_at": now()})
                    changed = True
            continue
        try:
            chk = f"kill -0 {pid} 2>/dev/null && echo running || echo done"
            if w.get("kind") == "local":
                alive = subprocess.run(chk, shell=True, capture_output=True, text=True).stdout.strip() == "running"
            else:
                out, _ = worker_ssh(w, chk, timeout=8)
                alive = out == "running"
            if alive:
                tail_cmd = f"tail -2 {log_path} 2>/dev/null | tr '\\n' ' '"
                if w.get("kind") == "local":
                    tail = subprocess.run(tail_cmd, shell=True, capture_output=True, text=True).stdout.strip()
                else:
                    tail, _ = worker_ssh(w, tail_cmd, timeout=8)
                print(f"{run['id']}: running  pid={pid}  | {tail}")
            else:
                ec_cmd = f"cat {pid_dir}/exit_code 2>/dev/null || echo unknown"
                if w.get("kind") == "local":
                    ec = subprocess.run(ec_cmd, shell=True, capture_output=True, text=True).stdout.strip()
                else:
                    ec, _ = worker_ssh(w, ec_cmd, timeout=8)
                ec = ec.strip() or "unknown"
                status = "done" if ec == "0" else "failed"
                for item in q:
                    if item["id"] == run["id"]:
                        item.update({"status": status, "ended_at": now(), "exit_code": ec, "updated_at": now()})
                        changed = True
                log_cmd = f"tail -5 {log_path} 2>/dev/null"
                if w.get("kind") == "local":
                    tail = subprocess.run(log_cmd, shell=True, capture_output=True, text=True).stdout.strip()
                else:
                    tail, _ = worker_ssh(w, log_cmd, timeout=8)
                indent = tail.replace("\n", "\n    ")
                print(f"{run['id']}: {status}  exit={ec}\n    {indent}")
        except Exception as e:
            print(f"{run['id']}: check error: {e}")
    if changed:
        write_json(QUEUES / "run_queue.json", q)


def _load_env_file(path: Path) -> dict[str, str]:
    envs: dict[str, str] = {}
    if not path.exists():
        return envs
    for line in path.read_text().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        envs[k.strip()] = v.strip().strip('"').strip("'")
    return envs


def _http_json(url: str, headers: dict[str, str], method: str = "GET", body: bytes | None = None) -> tuple[bool, int, dict]:
    req = urllib.request.Request(url=url, method=method, data=body)
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            code = resp.getcode()
            text = resp.read().decode("utf-8", errors="replace")
            data = json.loads(text) if text else {}
            return code == 200, code, data
    except urllib.error.HTTPError as e:
        data = {}
        try:
            data = json.loads(e.read().decode("utf-8", errors="replace"))
        except Exception:
            data = {}
        return False, int(e.code), data
    except Exception:
        return False, 0, {}


# ── project-status ───────────────────────────────────────────────────────────

def cmd_project_status(_args):
    """Print a dashboard of all projects and their current state."""
    projects = sorted(RESEARCH.glob("*/project.yaml"))
    if not projects:
        print("No projects found in research/")
        return

    q_runs = read_json(QUEUES / "run_queue.json")
    q_ideas = read_json(QUEUES / "idea_queue.json")

    for yaml_path in projects:
        try:
            data = yaml.safe_load(yaml_path.read_text()) or {}
        except Exception as e:
            print(f"[{yaml_path.parent.name}] parse error: {e}")
            continue

        pid = data.get("project_id", yaml_path.parent.name)
        status = data.get("status", "unknown")
        title = data.get("title", "")
        venue = (data.get("publication") or {}).get("target_venue", "")

        runs_for = [r for r in q_runs if r.get("project_id") == pid]
        run_counts = {}
        for r in runs_for:
            s = r.get("status", "unknown")
            run_counts[s] = run_counts.get(s, 0) + 1

        ideas_for = [i for i in q_ideas if i.get("project_id") == pid]
        idea_counts = {}
        for i in ideas_for:
            s = i.get("status", "unknown")
            idea_counts[s] = idea_counts.get(s, 0) + 1

        current = data.get("current_state", {})
        summary = (current.get("summary") or "").replace("\n", " ").strip()[:120]
        next_action = (current.get("next_action") or "").replace("\n", " ").strip()[:100]
        blockers = current.get("blockers") or []

        print(f"{'─'*70}")
        print(f"  {pid}  [{status}]")
        print(f"  Title   : {title}")
        if venue:
            print(f"  Venue   : {venue}")
        print(f"  Runs    : {run_counts or 'none queued'}")
        print(f"  Ideas   : {idea_counts or 'none queued'}")
        if summary:
            print(f"  State   : {summary}")
        if next_action:
            print(f"  Next    : {next_action}")
        if blockers:
            for b in blockers if isinstance(blockers, list) else [blockers]:
                print(f"  BLOCKER : {str(b).strip()[:100]}")
    print(f"{'─'*70}")


# ── add-worker ────────────────────────────────────────────────────────────────

def cmd_add_worker(args):
    """Register a new SSH worker (e.g. Vast.ai instance) in workers.yaml."""
    workers_path = ROOT / "workers" / "workers.yaml"
    data = yaml.safe_load(workers_path.read_text()) if workers_path.exists() else {"workers": []}
    workers: list[dict] = data.get("workers", [])

    if any(w["worker_id"] == args.worker_id for w in workers):
        print(f"Worker {args.worker_id!r} already exists. Edit workers.yaml to update it.")
        return

    # If a Vast.ai instance ID is provided, look up SSH details automatically.
    ssh_host = args.host
    ssh_port = args.port
    if args.vastai_instance_id and (not ssh_host or not ssh_port):
        try:
            out = subprocess.check_output(
                ["vastai", "show", "instance", str(args.vastai_instance_id), "--raw"],
                text=True, timeout=20,
            )
            inst = json.loads(out)
            if isinstance(inst, list):
                inst = inst[0]
            ssh_host = ssh_host or inst.get("ssh_host") or inst.get("public_ipaddr")
            ssh_port = ssh_port or inst.get("ssh_port") or 22
            print(f"Looked up instance {args.vastai_instance_id}: {ssh_host}:{ssh_port}")
        except Exception as e:
            print(f"Warning: could not look up Vast.ai instance: {e}")

    if not ssh_host or not ssh_port:
        print("ERROR: provide --host and --port, or --vastai-instance-id")
        return

    worker: dict = {
        "worker_id": args.worker_id,
        "kind": "ssh",
        "enabled": True,
        "worker_pool": args.pool,
        "workspace": args.workspace,
        "ssh": {
            "host": ssh_host,
            "port": int(ssh_port),
            "user": args.user,
            "key": args.key,
        },
        "runner": {"type": "bash", "docker_image": ""},
        "resources": {
            "gpu": 1,
            "vram_gb": args.vram_gb,
            "disk_gb": args.disk_gb,
        },
        "health": {
            "heartbeat_command": "echo ok",
            "busy_command": "pgrep -af 'python|nohup' || true",
        },
        "artifact_paths": {
            "logs": f"{args.workspace}/runs/logs",
            "metrics": f"{args.workspace}/runs/metrics",
            "checkpoints": f"{args.workspace}/runs/checkpoints",
        },
        "notes": args.notes or f"Added {now()}",
    }
    if args.vastai_instance_id:
        worker["vastai_instance_id"] = str(args.vastai_instance_id)

    workers.append(worker)
    data["workers"] = workers
    workers_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    print(f"Added worker {args.worker_id!r} ({ssh_host}:{ssh_port}) to workers.yaml")
    print(f"Run: python3 scripts/ros.py setup-worker --worker-id {args.worker_id}")


# ── vast-hunt ─────────────────────────────────────────────────────────────────

def cmd_vast_hunt(args):
    """Wrapper around vast_hunter.py that can read project hardware requirements."""
    hunter = Path(__file__).parent / "vast_hunter.py"
    if not hunter.exists():
        print("ERROR: scripts/vast_hunter.py not found")
        return

    cmd = ["python3", str(hunter)]

    # Load project hardware requirements if --project-id given
    if args.project_id:
        pfile = project_dir(args.project_id) / "project.yaml"
        if pfile.exists():
            pdata = yaml.safe_load(pfile.read_text()) or {}
            hw = pdata.get("hardware", {})
            profile = hw.get("profile", "pytorch")
            ref_hours = hw.get("ref_hours")
            ref_dlp = hw.get("ref_dlp")
            data_gb = hw.get("data_gb")
            disk_gb = hw.get("disk_gb")
            if profile and not args.profile:
                cmd += ["--profile", profile]
            if ref_hours and not args.ref_hours:
                cmd += ["--ref-hours", str(ref_hours)]
            if ref_dlp and not args.ref_dlp:
                cmd += ["--ref-dlp", str(ref_dlp)]
            if data_gb and not args.data_gb:
                cmd += ["--data-gb", str(data_gb)]
            if disk_gb and not args.disk_gb:
                cmd += ["--disk-gb", str(disk_gb)]

    if args.profile:
        cmd += ["--profile", args.profile]
    if args.max_dph is not None:
        cmd += ["--max-dph", str(args.max_dph)]
    if args.ref_hours is not None:
        cmd += ["--ref-hours", str(args.ref_hours)]
    if args.ref_dlp is not None:
        cmd += ["--ref-dlp", str(args.ref_dlp)]
    if args.data_gb is not None:
        cmd += ["--data-gb", str(args.data_gb)]
    if args.disk_gb is not None:
        cmd += ["--disk-gb", str(args.disk_gb)]
    if args.min_cuda is not None:
        cmd += ["--min-cuda", str(args.min_cuda)]
    if args.rent:
        cmd += ["--rent"]
        if args.image:
            cmd += ["--image", args.image]
        if args.onstart:
            cmd += ["--onstart", args.onstart]
    if args.extra:
        cmd += args.extra

    subprocess.run(cmd)


def cmd_key_smoke(args):
    env_file = Path(args.env_file)
    loaded = _load_env_file(env_file)
    for k, v in loaded.items():
        if k not in os.environ:
            os.environ[k] = v

    required = ["VASTAI_API_KEY", "WANDB_API_KEY", "HF_TOKEN", "GITHUB_TOKEN"]
    print("== Key Presence ==")
    for k in required:
        print(f"{k}: {'present' if os.environ.get(k) else 'missing'}")

    print("\n== Provider Smoke Tests ==")
    all_ok = True

    # VastAI: check CLI auth by listing instances.
    vast_cmd = ["vastai", "show", "instances"]
    path_env = dict(os.environ)
    path_env["PATH"] = f"{Path.home() / '.local' / 'bin'}:{path_env.get('PATH', '')}"
    try:
        r = subprocess.run(vast_cmd, env=path_env, capture_output=True, text=True, timeout=20)
        vast_ok = r.returncode == 0
    except Exception:
        vast_ok = False
    all_ok = all_ok and vast_ok
    print(f"VASTAI_API_KEY: {'PASS' if vast_ok else 'FAIL'}")

    gh_token = os.environ.get("GITHUB_TOKEN", "")
    gh_ok, gh_code, gh_data = _http_json(
        "https://api.github.com/user",
        {
            "Authorization": f"Bearer {gh_token}",
            "Accept": "application/vnd.github+json",
        },
    )
    all_ok = all_ok and gh_ok
    gh_user = gh_data.get("login", "unknown") if gh_ok else "-"
    print(f"GITHUB_TOKEN: {'PASS' if gh_ok else 'FAIL'} (http={gh_code}, user={gh_user})")

    hf_token = os.environ.get("HF_TOKEN", "")
    hf_ok, hf_code, hf_data = _http_json(
        "https://huggingface.co/api/whoami-v2",
        {"Authorization": f"Bearer {hf_token}"},
    )
    all_ok = all_ok and hf_ok
    hf_user = hf_data.get("name", "unknown") if hf_ok else "-"
    print(f"HF_TOKEN: {'PASS' if hf_ok else 'FAIL'} (http={hf_code}, user={hf_user})")

    wb_key = os.environ.get("WANDB_API_KEY", "")
    basic = base64.b64encode(f"api:{wb_key}".encode()).decode()
    wb_ok, wb_code, wb_data = _http_json(
        "https://api.wandb.ai/graphql",
        {
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/json",
        },
        method="POST",
        body=json.dumps({"query": "query { viewer { username } }"}).encode(),
    )
    # GraphQL can return 200 + errors. Validate viewer node explicitly.
    wb_user = ((wb_data.get("data") or {}).get("viewer") or {}).get("username", "")
    wb_final = wb_ok and bool(wb_user)
    all_ok = all_ok and wb_final
    print(f"WANDB_API_KEY: {'PASS' if wb_final else 'FAIL'} (http={wb_code}, user={wb_user or '-'})")

    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init-project")
    s.add_argument("--project-id")
    s.add_argument("--title", required=True)
    s.add_argument("--idea", required=True)
    s.add_argument("--target", required=True)
    s.add_argument("--metric")
    s.add_argument("--priority", type=int, default=5)
    s.set_defaults(func=cmd_init_project)

    s = sub.add_parser("add-idea")
    s.add_argument("--project-id", required=True)
    s.add_argument("--title", required=True)
    s.add_argument("--hypothesis", required=True)
    s.add_argument("--metric")
    s.add_argument("--priority", type=int, default=10)
    s.set_defaults(func=cmd_add_idea)

    s = sub.add_parser("list-ideas")
    s.set_defaults(func=cmd_list_ideas)

    s = sub.add_parser("add-benchmark-task")
    s.add_argument("--project-id", required=True)
    s.add_argument("--title", required=True)
    s.add_argument("--repo-url")
    s.add_argument("--command")
    s.add_argument("--metric")
    s.add_argument("--pass-rule")
    s.add_argument("--output-path")
    s.add_argument("--priority", type=int, default=10)
    s.set_defaults(func=cmd_add_benchmark_task)

    s = sub.add_parser("add-run")
    s.add_argument("--project-id", required=True)
    s.add_argument("--command", required=True)
    s.add_argument("--env")
    s.add_argument("--idea-id")
    s.add_argument("--benchmark-id")
    s.add_argument("--probe-id")
    s.add_argument("--worker-pool", default="default")
    s.add_argument("--metric-parser")
    s.add_argument("--target-metric")
    s.add_argument("--target-direction", choices=["higher", "lower"], default="lower")
    s.add_argument("--priority", type=int, default=10)
    s.set_defaults(func=cmd_add_run)

    s = sub.add_parser("add-publication")
    s.add_argument("--project-id", required=True)
    s.add_argument("--kind", choices=["blog", "paper", "release"], required=True)
    s.add_argument("--trigger", required=True)
    s.add_argument("--output-path", required=True)
    s.add_argument("--source-evidence", action="append")
    s.add_argument("--priority", type=int, default=10)
    s.set_defaults(func=cmd_add_publication)

    s = sub.add_parser("status")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("worker-status")
    s.set_defaults(func=cmd_worker_status)

    s = sub.add_parser("setup-worker")
    s.add_argument("--worker-id", required=True)
    s.set_defaults(func=cmd_setup_worker)

    s = sub.add_parser("dispatch-runs")
    s.add_argument("--worker-pool", default="any",
                   help="only dispatch to workers in this pool (default: any)")
    s.set_defaults(func=cmd_dispatch_runs)

    s = sub.add_parser("check-runs")
    s.set_defaults(func=cmd_check_runs)

    s = sub.add_parser("key-smoke")
    s.add_argument("--env-file", default="/home/ubuntu/.env.local")
    s.set_defaults(func=cmd_key_smoke)

    s = sub.add_parser("project-status",
                       help="Dashboard of all projects, run counts, and current state.")
    s.set_defaults(func=cmd_project_status)

    s = sub.add_parser("add-worker",
                       help="Register a new SSH worker (Vast.ai or any SSH machine).")
    s.add_argument("--worker-id", required=True, help="Unique name for this worker")
    s.add_argument("--vastai-instance-id", type=int, default=None,
                   help="Vast.ai instance ID — auto-fills --host and --port")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--user", default="root")
    s.add_argument("--key", default="~/.ssh/vastai_id_ed25519")
    s.add_argument("--pool", default="vastai")
    s.add_argument("--workspace", default="/root/research-worker")
    s.add_argument("--vram-gb", type=float, default=0.0)
    s.add_argument("--disk-gb", type=float, default=0.0)
    s.add_argument("--notes", default="")
    s.set_defaults(func=cmd_add_worker)

    s = sub.add_parser("vast-hunt",
                       help="Find GPU offers and estimate total cost. Reads hardware "
                            "requirements from project.yaml when --project-id is given.")
    s.add_argument("--project-id", default=None,
                   help="Load hardware profile from project.yaml hardware: section")
    s.add_argument("--profile", default=None, choices=["jax", "pytorch"],
                   help="Override hardware profile (jax or pytorch)")
    s.add_argument("--max-dph", type=float, default=None)
    s.add_argument("--ref-hours", type=float, default=None,
                   help="Reference experiment wall-clock hours (enables cost table)")
    s.add_argument("--ref-dlp", type=float, default=None,
                   help="DLPerf of reference GPU (e.g. 12.4 for RTX 3060)")
    s.add_argument("--data-gb", type=float, default=None)
    s.add_argument("--disk-gb", type=float, default=None)
    s.add_argument("--min-cuda", type=float, default=None)
    s.add_argument("--rent", action="store_true",
                   help="Print `vastai create instance` command for best result")
    s.add_argument("--image", default="")
    s.add_argument("--onstart", default="")
    s.add_argument("extra", nargs=argparse.REMAINDER,
                   help="Extra flags forwarded verbatim to vast_hunter.py")
    s.set_defaults(func=cmd_vast_hunt)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
