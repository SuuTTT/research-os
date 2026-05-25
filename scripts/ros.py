#!/usr/bin/env python3
"""Research OS CLI.

This CLI manages file-backed queues and project scaffolds. It is deliberately
small so the workflow stays inspectable before adding autonomous agents.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("RESEARCH_OS_ROOT", "/root/research-os"))
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

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
