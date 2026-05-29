#!/usr/bin/env python3
"""Find low-cost Vast.ai GPU offers and estimate total experiment cost.

The script intentionally does not store or accept API keys directly. Configure
the Vast CLI once with `vastai set api-key ...`, then run this script.

Profiles
--------
  jax      – high DLPerf/$ floor, CUDA 13+, verified only (original defaults)
  pytorch  – any DLPerf/$, CUDA 11.8+, lower disk bar (for PyTorch research)

Cost-estimate mode
------------------
Pass --ref-hours and --ref-dlp to estimate total GPU+storage+bandwidth cost:

  python3 scripts/vast_hunter.py --profile pytorch --max-dph 0.10 \\
      --ref-hours 4.5 --ref-dlp 12.4 --data-gb 0.5 \\
      --min-dlperf-usd 0 --min-disk-space 15 --min-cuda 11.8

  # → adds "Est.h" and "$TOTAL" columns to the table
  # → prints the cheapest option's `vastai create instance` command

Rent command
------------
  Add --rent to print the vastai create instance command for the top result.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any

# ---------------------------------------------------------------------------
# Profiles – hardware presets matching project needs
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict] = {
    "jax": {
        "max_dph": 0.10,
        "min_dlperf_usd": 200.0,
        "min_gpu_ram": 8.0,
        "min_disk_space": 50.0,
        "min_reliability": 0.95,
        "min_ports": 2,
        "min_cuda": 13.0,
        "storage": 50.0,
        "description": "JAX/XLA workloads. High DLPerf/$ floor, CUDA 13+.",
    },
    "pytorch": {
        "max_dph": 0.10,
        "min_dlperf_usd": 0.0,
        "min_gpu_ram": 8.0,
        "min_disk_space": 15.0,
        "min_reliability": 0.95,
        "min_ports": 1,
        "min_cuda": 11.8,
        "storage": 20.0,
        "description": "PyTorch workloads. No DLPerf/$ floor, CUDA 11.8+, 15GB disk.",
    },
}

DEFAULT_EXCLUDED_OFFER_IDS = {
    # 2026-05-26: good paper specs but SSH route repeatedly timed out during
    # env setup / rsync, making it unsuitable for queue workers.
    "34624617",  # created as contract 37907664, ssh4.vast.ai:27665
}

STORAGE_RATE_PER_GB_MONTH = 0.200   # Vast.ai default $/GB/month
BW_RATE_PER_GB = 0.0013             # Vast.ai bandwidth $/GB (in and out)
HOURS_PER_MONTH = 730.0


def money(v: Any) -> str:
    return "-" if v is None else f"{float(v):.4f}"


def one(v: Any) -> str:
    return "-" if v is None else f"{float(v):.1f}"


def build_query(args: argparse.Namespace) -> str:
    parts = [
        "verified=true",
        "rentable=true",
        "rented=false",
        f"dph < {args.max_dph}",
        f"gpu_ram >= {args.min_gpu_ram}",
        f"disk_space >= {args.min_disk_space}",
        f"reliability > {args.min_reliability}",
        f"direct_port_count >= {args.min_ports}",
    ]
    if args.min_dlperf_usd and args.min_dlperf_usd > 0:
        parts.append(f"dlperf_usd > {args.min_dlperf_usd}")
    if args.min_cuda:
        parts.append(f"cuda_vers >= {args.min_cuda}")
    if args.num_gpus:
        parts.append(f"num_gpus == {args.num_gpus}")
    if args.gpu:
        gpu_names = ",".join(f'"{name.strip()}"' for name in args.gpu.split(","))
        parts.append(f"gpu_name in [{gpu_names}]")
    if args.exclude_geolocation:
        locs = ",".join(loc.strip() for loc in args.exclude_geolocation.split(",") if loc.strip())
        if locs:
            parts.append(f"geolocation notin [{locs}]")
    return " ".join(parts)


def estimate_cost(row: dict, ref_hours: float, ref_dlp: float,
                  data_gb: float, disk_gb: float) -> dict:
    """Return cost breakdown dict for one offer given experiment reference."""
    dph = float(row.get("dph_total", row.get("dph", 0)) or 0)
    dlp = float(row.get("dlperf", 1) or 1)
    stor_rate = float(row.get("storage_cost", STORAGE_RATE_PER_GB_MONTH) or STORAGE_RATE_PER_GB_MONTH)

    est_hours = ref_hours * (ref_dlp / dlp) if dlp > 0 else 9999.0
    cost_gpu = est_hours * dph
    cost_stor = (disk_gb * stor_rate / HOURS_PER_MONTH) * est_hours
    cost_bw = data_gb * BW_RATE_PER_GB * 2   # upload + download
    total = cost_gpu + cost_stor + cost_bw
    return {
        "est_hours": est_hours,
        "cost_gpu": cost_gpu,
        "cost_stor": cost_stor,
        "cost_bw": cost_bw,
        "total": total,
    }


def search(args: argparse.Namespace) -> list[dict[str, Any]]:
    cmd = [
        "vastai", "search", "offers",
        build_query(args),
        "--raw",
        "--storage", str(args.storage),
        "--limit", str(args.limit),
        "-o", args.order,
    ]
    out = subprocess.check_output(cmd, text=True)
    rows = json.loads(out)
    filtered = []
    for row in rows:
        if str(row.get("id")) in args.exclude_offer_ids:
            continue
        dph = row.get("dph_total", row.get("dph"))
        if dph is None:
            continue
        if float(row.get("disk_space", 0.0) or 0.0) < args.min_disk_space:
            continue
        if float(dph) >= args.max_dph:
            continue
        if args.min_dlperf_usd > 0:
            dlusd = row.get("dlperf_per_dphtotal", row.get("dlperf_usd"))
            if dlusd is None or float(dlusd) <= args.min_dlperf_usd:
                continue
        filtered.append(row)
    return filtered


def _apply_profile(args: argparse.Namespace) -> None:
    """Fill in profile defaults for any flag the user left at None."""
    p = PROFILES.get(args.profile, {})
    for attr, key in [
        ("max_dph", "max_dph"),
        ("min_dlperf_usd", "min_dlperf_usd"),
        ("min_gpu_ram", "min_gpu_ram"),
        ("min_disk_space", "min_disk_space"),
        ("min_reliability", "min_reliability"),
        ("min_ports", "min_ports"),
        ("min_cuda", "min_cuda"),
        ("storage", "storage"),
    ]:
        if getattr(args, attr, None) is None and key in p:
            setattr(args, attr, p[key])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Profile
    ap.add_argument("--profile", default="jax", choices=list(PROFILES),
                    help="hardware profile: 'jax' (default) or 'pytorch'")

    # Hardware filters — all optional; profile fills gaps
    ap.add_argument("--max-dph", type=float, default=None)
    ap.add_argument("--min-dlperf-usd", type=float, default=None,
                    help="set 0 to disable DLPerf/$ filter (pytorch profile does this)")
    ap.add_argument("--min-gpu-ram", type=float, default=None)
    ap.add_argument("--min-disk-space", type=float, default=None)
    ap.add_argument("--min-reliability", type=float, default=None)
    ap.add_argument("--min-ports", type=int, default=None)
    ap.add_argument("--min-cuda", type=float, default=None)
    ap.add_argument("--storage", type=float, default=None,
                    help="storage GB to include in Vast pricing calculation")
    ap.add_argument("--num-gpus", type=int, default=None)
    ap.add_argument("--gpu", default="")
    ap.add_argument("--exclude-offer-id", action="append", default=[])
    ap.add_argument("--exclude-geolocation", default="")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--order", default="dph+")

    # Cost-estimate mode
    ap.add_argument("--ref-hours", type=float, default=None,
                    help="reference experiment wall-clock hours on --ref-dlp GPU")
    ap.add_argument("--ref-dlp", type=float, default=None,
                    help="DLPerf of reference GPU (e.g. 12.4 for RTX 3060)")
    ap.add_argument("--data-gb", type=float, default=0.5,
                    help="data transferred per experiment in GB (for bandwidth cost)")
    ap.add_argument("--disk-gb", type=float, default=20.0,
                    help="disk allocation in GB (for storage cost)")

    # Rent helper
    ap.add_argument("--rent", action="store_true",
                    help="print `vastai create instance` command for the top result")
    ap.add_argument("--image", default="pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel",
                    help="Docker image to use in --rent command")
    ap.add_argument("--onstart", default="",
                    help="onstart-cmd string for --rent command")

    ap.add_argument("--json", action="store_true")

    args = ap.parse_args()
    _apply_profile(args)
    args.exclude_offer_ids = set(args.exclude_offer_id) | DEFAULT_EXCLUDED_OFFER_IDS

    rows = search(args)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    cost_mode = args.ref_hours is not None and args.ref_dlp is not None

    print(f"Profile: {args.profile}  ({PROFILES[args.profile]['description']})")
    print(f"Query: {build_query(args)}")
    print()

    if cost_mode:
        print(f"Cost estimate: ref {args.ref_hours:.1f}h @ DLP={args.ref_dlp:.1f}, "
              f"data={args.data_gb:.2f}GB, disk={args.disk_gb:.0f}GB")
        print()
        header = (f"{'#':<3} {'GPU':<16} {'ID':<12} {'VRAM':>5} {'$/hr':>7} "
                  f"{'DLP':>5} {'Est.h':>6} {'$gpu':>6} {'$stor':>6} {'$bw':>5} {'$TOTAL':>7} "
                  f"{'Rel':>5}  {'Net dn':>8}")
        print(header)
        print("-" * len(header))
        best: dict | None = None
        best_total = float("inf")
        for i, r in enumerate(rows, 1):
            c = estimate_cost(r, args.ref_hours, args.ref_dlp, args.data_gb, args.disk_gb)
            dph = float(r.get("dph_total", r.get("dph", 0)) or 0)
            gpu_name = (r.get("gpu_name") or "-")[:16]
            vram = float(r.get("gpu_ram", 0) or 0)
            dlp = float(r.get("dlperf", 0) or 0)
            rel = float(r.get("reliability", 0) or 0) * 100
            net_dn = float(r.get("inet_down", 0) or 0)
            print(f"{i:<3} {gpu_name:<16} {r.get('id'):<12} {vram:>4.0f}G "
                  f"{dph:>7.4f} {dlp:>5.1f} {c['est_hours']:>6.1f} "
                  f"{c['cost_gpu']:>6.3f} {c['cost_stor']:>6.4f} {c['cost_bw']:>5.4f} "
                  f"{c['total']:>7.3f}  {rel:>5.1f}%  {net_dn:>6.0f}Mbps")
            if c["total"] < best_total:
                best_total = c["total"]
                best = r
        print()
        if best:
            c = estimate_cost(best, args.ref_hours, args.ref_dlp, args.data_gb, args.disk_gb)
            print(f"Best total cost: ID={best['id']} {best.get('gpu_name')} "
                  f"${best_total:.3f}  (${c['cost_gpu']:.3f} gpu + "
                  f"${c['cost_stor']:.4f} storage + ${c['cost_bw']:.4f} bw), "
                  f"~{c['est_hours']:.1f}h")
            if args.rent:
                _print_rent_cmd(best, args)
    else:
        # Standard DLPerf/$ table (original format)
        print("| Offer | GPU | GPUs | $/h | DLPerf | DLPerf/$ | VRAM GB | Rel | Net down/up |")
        print("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            dph = r.get("dph_total", r.get("dph"))
            dlusd = r.get("dlperf_per_dphtotal", r.get("dlperf_usd"))
            net = f"{one(r.get('inet_down'))}/{one(r.get('inet_up'))}"
            print(
                f"| {r.get('id')} | {r.get('gpu_name')} | {r.get('num_gpus')} | "
                f"{money(dph)} | {one(r.get('dlperf'))} | {one(dlusd)} | "
                f"{one(r.get('gpu_ram'))} | {one(r.get('reliability'))} | {net} |"
            )
        if args.rent and rows:
            _print_rent_cmd(rows[0], args)

    return 0


def _print_rent_cmd(r: dict, args: argparse.Namespace) -> None:
    oid = r.get("id")
    onstart = args.onstart or ""
    cmd = f"vastai create instance {oid} \\\n  --image {args.image} \\\n  --disk {int(args.disk_gb)}"
    if onstart:
        cmd += f' \\\n  --onstart-cmd "{onstart}"'
    print()
    print("# Rent command:")
    print(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
