#!/usr/bin/env python3
"""
Auto pipeline for poly(exp_mul)+FG-only lagcorr scan v5 (2026-02-14).

Intended launch (so SSH/rsync are allowed in the Codex sandbox too):

  /usr/bin/zsh -lc 'python3 3dnet/auto_poly_exp_mul_lagfg_v5_pipeline.py'

Pipeline:
  1) Monitor Stage A workers (already running on remotes) until completion.
  2) Collect + merge Stage A artifacts; write a local report.
  3) Select top candidates and dispatch Stage B long reruns (100k iters).
  4) Monitor Stage B; collect + merge; write a local report.

Notes:
  - We do NOT rsync/sync remote code in this pipeline (avoid mixing versions mid-run).
  - Ranking/selection uses strict status==ok and converged==True (robustly parsed from CSV strings).
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATE_BUCKET = "20260214"
DATE_REPORT = "2026-02-14"

STAGEA_NAME = "poly_exp_mul_lagfg_optim_scan_20260214_v5"
STAGEA_REMOTE_ROOT = f"/data/zhenghao/fg_rmw/runs/{STAGEA_NAME}_remote"
STAGEA_MANIFEST = PROJECT_ROOT / "runs" / "remote" / DATE_BUCKET / f"{STAGEA_NAME}_dispatch_manifest_manual.json"
STAGEA_CANDIDATES_JSONL = (
    PROJECT_ROOT / "runs" / "remote" / DATE_BUCKET / f"{STAGEA_NAME}_candidates" / "candidates.jsonl"
)
STAGEA_SPLITS_DIR = PROJECT_ROOT / "runs" / "remote" / DATE_BUCKET / f"{STAGEA_NAME}_candidates" / "splits"
STAGEA_MERGED_ROOT = PROJECT_ROOT / "runs" / "remote" / DATE_BUCKET / f"{STAGEA_NAME}_merged"
STAGEA_REPORT_MD = (
    PROJECT_ROOT
    / "reports"
    / "analysis"
    / DATE_REPORT
    / f"{DATE_REPORT}_poly_exp_mul_lagfg_optim_scan_v5_results.md"
)

STAGEB_NAME = f"{STAGEA_NAME}B"
STAGEB_REMOTE_ROOT = f"/data/zhenghao/fg_rmw/runs/{STAGEB_NAME}_remote"
STAGEB_CANDIDATES_DIR = PROJECT_ROOT / "runs" / "remote" / DATE_BUCKET / f"{STAGEB_NAME}_candidates"
STAGEB_MERGED_ROOT = PROJECT_ROOT / "runs" / "remote" / DATE_BUCKET / f"{STAGEB_NAME}_merged"
STAGEB_REPORT_MD = (
    PROJECT_ROOT
    / "reports"
    / "analysis"
    / DATE_REPORT
    / f"{DATE_REPORT}_poly_exp_mul_lagfg_optim_scan_v5B_results.md"
)

STAGEB_NUM_ITERS = 100_000
STAGEB_PRINT_EVERY = 200

POLL_SEC = 300


@dataclass(frozen=True)
class Worker:
    host: str
    alias: str
    gpu: int
    pid: int
    python_bin: str
    candidate_chunk: str
    output_dir: str

    @property
    def stagea_chunk_local(self) -> Path:
        return STAGEA_SPLITS_DIR / str(self.candidate_chunk)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run_capture(cmd: Sequence[str]) -> str:
    return subprocess.check_output(list(cmd), text=True)


def ssh(host: str, remote_cmd: str) -> str:
    return _run_capture(["ssh", "-o", "BatchMode=yes", host, remote_cmd])


def rsync_remote_to_local(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["rsync", "-az", src, str(dst)])


def rsync_local_to_remote(src: Path, dst: str) -> None:
    subprocess.check_call(["rsync", "-az", str(src), dst])


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def parse_workers(manifest_path: Path) -> List[Worker]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    workers: List[Worker] = []
    for rec in payload.get("workers", []):
        workers.append(
            Worker(
                host=str(rec["host"]),
                alias=str(rec["alias"]),
                gpu=int(rec["gpu"]),
                pid=int(rec["pid"]),
                python_bin=str(rec["python"]),
                candidate_chunk=str(rec["candidate_chunk"]),
                output_dir=str(rec["output_dir"]).rstrip("/"),
            )
        )
    if not workers:
        raise ValueError(f"No workers found in manifest: {manifest_path}")
    return workers


def chunk_expected_count(chunk_jsonl: Path) -> int:
    return sum(1 for ln in chunk_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip())


def is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def pid_alive(worker: Worker) -> bool:
    out = ssh(worker.host, f"ps -p {int(worker.pid)} -o pid= >/dev/null 2>&1 && echo 1 || echo 0").strip()
    return out == "1"


def count_done_profiles(worker: Worker) -> int:
    cmd = f"find {worker.output_dir} -maxdepth 3 -path '*/cube2/eor_corr_profile.csv' 2>/dev/null | wc -l"
    out = ssh(worker.host, cmd).strip()
    try:
        return int(out)
    except ValueError:
        return 0


def wait_until_complete(
    workers: Sequence[Worker],
    expected: Dict[str, int],
    *,
    stage_tag: str,
    log_path: Path,
) -> None:
    print(f"[{_now()}] monitor {stage_tag} start; poll_sec={POLL_SEC}", flush=True)
    while True:
        all_done = True
        for w in workers:
            exp = int(expected[w.alias])
            done = int(count_done_profiles(w))
            alive = bool(pid_alive(w))
            line = f"[{_now()}] {stage_tag} {w.alias} done={done}/{exp} alive={int(alive)}"
            print(line, flush=True)
            append_line(log_path, line)
            if done < exp:
                all_done = False
                if not alive:
                    raise RuntimeError(f"{stage_tag} worker died early: {w.alias} done={done}/{exp} host={w.host}")
                continue
            # done>=exp: we still wait for the main scan process to exit, to avoid a race
            # where eor_corr_profile.csv exists but the final CSV/MD artifacts are not written yet.
            if alive:
                all_done = False
        if all_done:
            print(f"[{_now()}] monitor {stage_tag} complete", flush=True)
            return
        time.sleep(float(POLL_SEC))


def read_csv_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def merge_worker_results(worker_dirs: Sequence[Path]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    merged: List[Dict[str, object]] = []
    for d in worker_dirs:
        p = d / "poly_lagfg_optim_scan_results.csv"
        for row in read_csv_rows(p):
            row["_source"] = str(p)
            merged.append(row)

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in merged:
        if str(row.get("status")) != "ok":
            continue
        if not is_true(row.get("converged", True)):
            continue
        if str(row.get("dataset")) == "cube1":
            continue
        grouped.setdefault(str(row.get("candidate")), []).append(row)

    ranked: List[Dict[str, object]] = []
    for cand, items in grouped.items():
        scores = [float(x.get("eor_corr_score")) for x in items if x.get("eor_corr_score") not in (None, "")]
        ps_mad = [float(x.get("ps2d_win_log10_mad")) for x in items if x.get("ps2d_win_log10_mad") not in (None, "")]
        ranked.append(
            {
                "candidate": cand,
                "n_ok_converged": int(len(items)),
                "eor_corr_score_mean": (sum(scores) / len(scores)) if scores else float("nan"),
                "ps2d_win_log10_mad_mean": (sum(ps_mad) / len(ps_mad)) if ps_mad else float("nan"),
            }
        )
    ranked.sort(key=lambda r: (-(r["eor_corr_score_mean"]), r["ps2d_win_log10_mad_mean"]))
    for i, row in enumerate(ranked, start=1):
        row["rank"] = int(i)
    return merged, ranked


def collect_stage(
    workers: Sequence[Worker],
    local_root: Path,
) -> List[Path]:
    out_dirs: List[Path] = []
    for w in workers:
        local_worker_dir = local_root / "worker_results" / w.alias
        local_worker_dir.mkdir(parents=True, exist_ok=True)
        out_dirs.append(local_worker_dir)
        for fn in (
            "poly_lagfg_optim_scan_results.csv",
            "poly_lagfg_optim_scan_rank.csv",
            "poly_lagfg_optim_scan_summary.md",
            "manifest.json",
        ):
            rsync_remote_to_local(f"{w.host}:{w.output_dir}/{fn}", local_worker_dir / fn)
    return out_dirs


def load_candidates_map(path: Path) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        d = json.loads(ln)
        out[str(d["name"])] = d
    return out


def select_top_candidates(
    merged_rows: Sequence[Dict[str, object]],
    ranked_rows: Sequence[Dict[str, object]],
    *,
    top_n: int,
) -> List[str]:
    ok = set()
    for r in merged_rows:
        if str(r.get("dataset")) != "cube2":
            continue
        if str(r.get("status")) != "ok":
            continue
        if not is_true(r.get("converged", True)):
            continue
        ok.add(str(r.get("candidate")))

    selected: List[str] = []
    for r in ranked_rows:
        name = str(r.get("candidate"))
        if name in ok:
            selected.append(name)
        if len(selected) >= int(top_n):
            break
    return selected


def write_report_md(
    path: Path,
    *,
    title: str,
    merged_root: Path,
    merged_rows: Sequence[Dict[str, object]],
    ranked_rows: Sequence[Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def row_for(name: str) -> Optional[Dict[str, object]]:
        for r in merged_rows:
            if str(r.get("dataset")) == "cube2" and str(r.get("candidate")) == name:
                return r
        return None

    best_name = str(ranked_rows[0]["candidate"]) if ranked_rows else ""
    best_row = row_for(best_name) if best_name else None

    lines: List[str] = []
    lines.append(f"# {title}\n\n")
    lines.append(f"- merged_root: `{merged_root}`\n")
    if best_row is not None:
        keep = [
            "candidate",
            "status",
            "converged",
            "eor_corr_score",
            "eor_corr_mean",
            "eor_corr_p10",
            "eor_corr_min",
            "ps2d_win_log10_mad",
            "ps2d_win_log10_rmse",
            "ps2d_win_power_sum_ratio",
            "config_path",
            "log_path",
        ]
        payload = {k: best_row.get(k) for k in keep if k in best_row}
        lines.append("\n## Best (by score)\n\n")
        lines.append("```json\n")
        lines.append(json.dumps(payload, indent=2))
        lines.append("\n```\n")

    lines.append("\n## Top 10\n\n")
    lines.append("| rank | candidate | score_mean | ps_mad_mean | n_ok |\n")
    lines.append("|---:|---|---:|---:|---:|\n")
    for r in ranked_rows[:10]:
        lines.append(
            f"| {int(r['rank'])} | {r['candidate']} | {float(r['eor_corr_score_mean']):.6f} | {float(r['ps2d_win_log10_mad_mean']):.6f} | {int(r['n_ok_converged'])} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def dispatch_stage_b(
    stagea_workers: Sequence[Worker],
    top_candidates: Sequence[str],
    candidates_map: Dict[str, Dict[str, object]],
    *,
    log_path: Path,
) -> List[Worker]:
    if not top_candidates:
        raise ValueError("No top candidates for Stage B.")

    workers = list(stagea_workers)
    chunks: List[List[str]] = [[] for _ in range(len(workers))]
    for i, name in enumerate(top_candidates):
        chunks[i % len(workers)].append(name)

    splits_dir = STAGEB_CANDIDATES_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Build local chunk jsonl files.
    for i, names in enumerate(chunks):
        p = splits_dir / f"candidates_chunk{i}.jsonl"
        lines = [json.dumps(candidates_map[n], sort_keys=True) for n in names]
        p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # Remote prep.
    for w in workers:
        ssh(w.host, f"mkdir -p {STAGEB_REMOTE_ROOT}/_dispatch {STAGEB_REMOTE_ROOT}/{w.alias}")

    # Upload chunk files and launch.
    stageb_workers: List[Worker] = []
    for i, w in enumerate(workers):
        local_chunk = splits_dir / f"candidates_chunk{i}.jsonl"
        remote_chunk = f"{STAGEB_REMOTE_ROOT}/_dispatch/candidates_chunk{i}.jsonl"
        rsync_local_to_remote(local_chunk, f"{w.host}:{remote_chunk}")

        out_dir = f"{STAGEB_REMOTE_ROOT}/{w.alias}"
        gpu_map = f"cube2:{int(w.gpu)}"
        cmd = (
            f"mkdir -p {out_dir} && nohup {w.python_bin} /data/zhenghao/fg_rmw/code/3dnet/run_poly_lagfg_optim_scan.py "
            f"--work-root /data/zhenghao/fg_rmw --code-dir /data/zhenghao/fg_rmw/code/3dnet --data-dir /data/zhenghao/fg_rmw/data "
            f"--output-dir {out_dir} --datasets cube2 --exclude-from-ranking cube1 --gpu-map {gpu_map} --max-concurrent-jobs 1 "
            f"--num-iters {int(STAGEB_NUM_ITERS)} --print-every {int(STAGEB_PRINT_EVERY)} --cut-size-frac 0.30 --freq-delta-mhz 0.1 "
            f"--power-config configs/power_eor_window.json --candidates-jsonl {remote_chunk} "
            f"--lagfg-prior-source obs_smooth --lagfg-prior-robust --python-bin {w.python_bin} "
            f"> {out_dir}/worker.log 2>&1 & echo $!"
        )
        pid_str = ssh(w.host, cmd).strip()
        pid = int(pid_str) if pid_str.isdigit() else 0

        stageb_workers.append(
            Worker(
                host=w.host,
                alias=w.alias,
                gpu=w.gpu,
                pid=pid,
                python_bin=w.python_bin,
                candidate_chunk=f"candidates_chunk{i}.jsonl",
                output_dir=out_dir,
            )
        )
        line = f"[{_now()}] B launch {w.alias} gpu={w.gpu} pid={pid} n_cands={len(chunks[i])}"
        print(line, flush=True)
        append_line(log_path, line)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "purpose": "Auto Stage B reruns (100k iters) for top candidates from Stage A.",
        "stagea": {"name": STAGEA_NAME, "remote_root": STAGEA_REMOTE_ROOT, "manifest": str(STAGEA_MANIFEST)},
        "stageb": {"name": STAGEB_NAME, "remote_root": STAGEB_REMOTE_ROOT, "num_iters": int(STAGEB_NUM_ITERS)},
        "top_candidates": list(top_candidates),
        "workers": [w.__dict__ for w in stageb_workers],
    }
    STAGEB_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    (STAGEB_CANDIDATES_DIR / "stageb_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return stageb_workers


def best_mean_corr(merged_rows: Sequence[Dict[str, object]], ranked_rows: Sequence[Dict[str, object]]) -> Tuple[float, str]:
    if not ranked_rows:
        return float("nan"), ""
    name = str(ranked_rows[0]["candidate"])
    for r in merged_rows:
        if str(r.get("dataset")) == "cube2" and str(r.get("candidate")) == name:
            try:
                return float(r.get("eor_corr_mean")), name
            except Exception:
                return float("nan"), name
    return float("nan"), name


def main() -> int:
    if not STAGEA_MANIFEST.exists():
        raise FileNotFoundError(f"Missing Stage A manifest: {STAGEA_MANIFEST}")
    workers_a = parse_workers(STAGEA_MANIFEST)
    expected_a = {w.alias: chunk_expected_count(w.stagea_chunk_local) for w in workers_a}

    stagea_log = PROJECT_ROOT / "runs" / "remote" / DATE_BUCKET / f"{STAGEA_NAME}_monitor.log"
    wait_until_complete(workers_a, expected_a, stage_tag="A", log_path=stagea_log)

    # Stage A collect + merge.
    worker_dirs_a = collect_stage(workers_a, STAGEA_MERGED_ROOT)
    merged_a, ranked_a = merge_worker_results(worker_dirs_a)
    merged_a_csv = STAGEA_MERGED_ROOT / "merged" / "poly_lagfg_optim_scan_results_merged.csv"
    ranked_a_csv = STAGEA_MERGED_ROOT / "merged" / "poly_lagfg_optim_scan_rank_merged.csv"
    write_csv(merged_a_csv, merged_a)
    write_csv(ranked_a_csv, ranked_a)
    write_report_md(
        STAGEA_REPORT_MD,
        title="Poly(exp_mul) + FG-only LagCorr Scan v5 (Stage A) Results",
        merged_root=STAGEA_MERGED_ROOT,
        merged_rows=merged_a,
        ranked_rows=ranked_a,
    )
    mean_a, best_a = best_mean_corr(merged_a, ranked_a)
    print(f"[{_now()}] Stage A best: {best_a} mean_corr={mean_a}", flush=True)

    if mean_a >= 0.90:
        print(f"[{_now()}] Target met in Stage A (>=0.9). Stop.", flush=True)
        return 0

    # Stage B dispatch (top 12).
    cand_map = load_candidates_map(STAGEA_CANDIDATES_JSONL)
    top = select_top_candidates(merged_a, ranked_a, top_n=12)
    if not top:
        raise RuntimeError("No ok+converged candidates to rerun in Stage B.")

    stageb_log = PROJECT_ROOT / "runs" / "remote" / DATE_BUCKET / f"{STAGEB_NAME}_monitor.log"
    workers_b = dispatch_stage_b(workers_a, top, cand_map, log_path=stageb_log)
    expected_b = {
        w.alias: chunk_expected_count(STAGEB_CANDIDATES_DIR / "splits" / w.candidate_chunk) for w in workers_b
    }
    wait_until_complete(workers_b, expected_b, stage_tag="B", log_path=stageb_log)

    # Stage B collect + merge.
    worker_dirs_b = collect_stage(workers_b, STAGEB_MERGED_ROOT)
    merged_b, ranked_b = merge_worker_results(worker_dirs_b)
    merged_b_csv = STAGEB_MERGED_ROOT / "merged" / "poly_lagfg_optim_scan_results_merged.csv"
    ranked_b_csv = STAGEB_MERGED_ROOT / "merged" / "poly_lagfg_optim_scan_rank_merged.csv"
    write_csv(merged_b_csv, merged_b)
    write_csv(ranked_b_csv, ranked_b)
    write_report_md(
        STAGEB_REPORT_MD,
        title="Poly(exp_mul) + FG-only LagCorr Scan v5 (Stage B 100k iters) Results",
        merged_root=STAGEB_MERGED_ROOT,
        merged_rows=merged_b,
        ranked_rows=ranked_b,
    )
    mean_b, best_b = best_mean_corr(merged_b, ranked_b)
    print(f"[{_now()}] Stage B best: {best_b} mean_corr={mean_b}", flush=True)
    if mean_b >= 0.90:
        print(f"[{_now()}] Target met in Stage B (>=0.9).", flush=True)
    else:
        print(f"[{_now()}] Target NOT met after Stage B (<0.9). Next: implement Stage C (eor_as_residual, etc).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
