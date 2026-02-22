"""
Benchmark & tune YOLO kill feed detection using Medal.tv ground truth.

Searches Medal.tv for "cs2 1k" through "cs2 5k", downloads videos,
runs YOLO kill detection, and compares detected vs expected kill counts.

Usage:
    python benchmark.py                    # Download + benchmark with current params
    python benchmark.py --per-query 20     # 20 videos per query
    python benchmark.py --tune             # Grid search on already-downloaded videos
    python benchmark.py --skip-download    # Benchmark only (no new downloads)
"""

import json
import logging
import argparse
import itertools
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
RESULTS_FILE = BASE_DIR / "benchmark_results.json"
GROUND_TRUTH_FILE = BASE_DIR / "benchmark_ground_truth.json"
BEST_PARAMS_FILE = BASE_DIR / "models" / "yolo_params.json"

DEFAULT_QUERIES = ["cs2 1k", "cs2 2k", "cs2 3k", "cs2 4k", "cs2 5k"]

# Default detection parameters (current pipeline values)
DEFAULT_PARAMS = {
    "sample_fps": 4,
    "conf": 0.45,
    "sim_threshold": 0.6,
    "width_ratio_threshold": 0.15,
    "gap_threshold": 15,
    "cooldown": 2.0,
}


# -------------------------------------------------------------------------
# Medal.tv API (reuse from collect_training_data)
# -------------------------------------------------------------------------

def search_and_collect(queries, per_query):
    """Search Medal.tv and return video metadata list."""
    from collect_training_data import (
        get_api_key, search_medal, parse_kill_count, build_medal_url
    )
    import time

    api_key = get_api_key()
    if not api_key:
        logger.error("Could not get Medal.tv API key")
        return []

    logger.info(f"API key: {api_key[:20]}...")
    videos = []

    for query in queries:
        logger.info(f"Searching: '{query}' (limit={per_query})")
        clips = search_medal(query, api_key, limit=per_query)
        if not clips:
            logger.warning(f"  No results for '{query}'")
            continue

        logger.info(f"  Found {len(clips)} clips")
        for clip in clips:
            content_id = str(clip.get("contentId", ""))
            if not content_id:
                continue

            title = clip.get("contentTitle", "")
            duration = clip.get("videoLengthSeconds", 0)

            if duration < 5 or duration > 90:
                continue

            expected = parse_kill_count(title, query)
            if expected is None:
                continue

            clip_url = clip.get("directClipUrl", "")
            if not clip_url:
                clip_url = build_medal_url(content_id)

            videos.append({
                "url": clip_url,
                "content_id": content_id,
                "expected_kills": expected,
                "title": title,
                "duration": duration,
                "query": query,
            })

        time.sleep(0.5)

    logger.info(f"Total videos found: {len(videos)}")
    return videos


def download_videos(videos):
    """Download videos and return updated list with file paths."""
    from collect_training_data import download_video

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for i, entry in enumerate(videos, 1):
        url = entry["url"]
        content_id = entry["content_id"]
        title = entry.get("title", "")

        # Check if already downloaded
        existing = list(DOWNLOADS_DIR.glob(f"*{content_id}*"))
        if existing:
            entry["video_path"] = str(existing[0])
            downloaded.append(entry)
            logger.info(f"[{i}/{len(videos)}] Already exists: {existing[0].name}")
            continue

        logger.info(f"[{i}/{len(videos)}] Downloading: {title[:50]}")
        video_path = download_video(url)
        if video_path:
            entry["video_path"] = str(video_path)
            downloaded.append(entry)
            logger.info(f"  Saved: {video_path.name}")
        else:
            logger.warning(f"  Download failed, skipping")

    logger.info(f"Downloaded: {len(downloaded)}/{len(videos)}")
    return downloaded


# -------------------------------------------------------------------------
# YOLO kill detection (standalone, parameterizable)
# -------------------------------------------------------------------------

def _compare_signatures(a, b):
    """Normalized correlation between two grayscale signatures."""
    if a is None or b is None:
        return 0.0
    fa = a.astype(float).flatten()
    fb = b.astype(float).flatten()
    fa -= fa.mean()
    fb -= fb.mean()
    na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(fa, fb) / (na * nb))


def detect_kills_yolo(model, video_path, params=None):
    """
    Run YOLO kill feed detection with given parameters.
    Returns (kill_count, kill_times, transitions_count, width_info).
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, [], 0, {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or w <= 0 or h <= 0:
        cap.release()
        return 0, [], 0, {}

    min_x = int(w * 0.55)
    max_y = int(h * 0.22)
    min_y = int(h * 0.02)

    interval = max(1, int(fps / p["sample_fps"]))
    frame_idx = 0
    prev_sig = None
    transitions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            ts = frame_idx / fps
            results = model(frame, conf=p["conf"], verbose=False)
            best = None
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    if x1 < min_x or y1 > max_y or y1 < min_y:
                        continue
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw < 50 or bh < 10 or bh > 80:
                        continue
                    if bw / bh < 2.0 or bw / bh > 15:
                        continue
                    if best is None or y1 < best[1]:
                        best = (x1, y1, x2, y2, bw, bh)

            if best is not None:
                x1, y1, x2, y2, bw, bh = best
                crop_w = min(int(bw * 0.40), int(w * 0.18))
                crop = frame[y1:y2, x1:min(x1 + crop_w, w)]
                if crop.size > 0:
                    sig = cv2.cvtColor(
                        cv2.resize(crop, (120, 24)), cv2.COLOR_BGR2GRAY
                    )
                    sim = (_compare_signatures(sig, prev_sig)
                           if prev_sig is not None else 0.0)
                    if prev_sig is None or sim < p["sim_threshold"]:
                        transitions.append({"ts": round(ts, 2), "bw": bw})
                    prev_sig = sig

        frame_idx += 1
    cap.release()

    if len(transitions) < 3:
        # Too few transitions — return all as kills
        times = [t["ts"] for t in transitions]
        return len(times), times, len(transitions), {"method": "too_few"}

    # Width analysis
    all_widths = [t["bw"] for t in transitions]
    widths = sorted(set(all_widths))
    max_gap = 0
    gap_idx = 0
    for i in range(len(widths) - 1):
        gap = widths[i + 1] - widths[i]
        if gap > max_gap:
            max_gap = gap
            gap_idx = i

    cooldown = p["cooldown"]

    def _dedup(entries):
        if not entries:
            return []
        times = sorted(e["ts"] for e in entries)
        d = [times[0]]
        for t in times[1:]:
            if t - d[-1] >= cooldown:
                d.append(t)
        return d

    width_range = max(all_widths) - min(all_widths)
    width_mean = np.mean(all_widths)
    width_ratio = width_range / width_mean if width_mean > 0 else 0

    width_info = {
        "range": width_range,
        "mean": round(width_mean, 1),
        "ratio": round(width_ratio, 3),
        "max_gap": max_gap,
        "n_transitions": len(transitions),
    }

    if width_ratio < p["width_ratio_threshold"] and max_gap < p["gap_threshold"]:
        # Homogeneous widths — all same player
        kill_times = _dedup(transitions)
        width_info["method"] = "homogeneous"
    else:
        # Gap-based split
        if max_gap >= p["gap_threshold"]:
            gap_thresh = (widths[gap_idx] + widths[gap_idx + 1]) / 2
        else:
            gap_thresh = width_mean
        gap_times = _dedup([t for t in transitions if t["bw"] >= gap_thresh])

        # Mean-based split
        mean_times = _dedup([t for t in transitions if t["bw"] >= width_mean])

        # Take more selective
        if len(gap_times) <= len(mean_times):
            kill_times = gap_times
            width_info["method"] = "gap"
            width_info["threshold"] = round(gap_thresh, 1)
        else:
            kill_times = mean_times
            width_info["method"] = "mean"
            width_info["threshold"] = round(width_mean, 1)

    return len(kill_times), kill_times, len(transitions), width_info


# -------------------------------------------------------------------------
# Benchmark runner
# -------------------------------------------------------------------------

def load_ground_truth():
    """Load ground truth from file."""
    if GROUND_TRUTH_FILE.exists():
        with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_ground_truth(entries):
    """Save ground truth to file."""
    with open(GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def run_benchmark(model, ground_truth, params=None):
    """Run detection on all ground truth videos. Returns results dict."""
    p = params or DEFAULT_PARAMS
    correct = 0
    total = 0
    details = []
    by_query = {}

    for entry in ground_truth:
        video_path = entry.get("video_path", "")
        if not video_path or not Path(video_path).exists():
            continue

        expected = entry["expected_kills"]
        query = entry.get("query", f"cs2 {expected}k")

        detected, times, n_trans, w_info = detect_kills_yolo(
            model, video_path, params=p
        )

        is_correct = detected == expected
        if is_correct:
            correct += 1
        total += 1

        detail = {
            "video": Path(video_path).name,
            "expected": expected,
            "detected": detected,
            "correct": is_correct,
            "transitions": n_trans,
            "width_method": w_info.get("method", ""),
            "times": [round(t, 1) for t in times],
        }
        details.append(detail)

        # By query stats
        if query not in by_query:
            by_query[query] = {"total": 0, "correct": 0}
        by_query[query]["total"] += 1
        if is_correct:
            by_query[query]["correct"] += 1

        status = "OK" if is_correct else f"WRONG({detected})"
        logger.info(f"  {Path(video_path).name}: {detected}/{expected} {status} "
                     f"[{w_info.get('method', '')}]")

    accuracy = correct / total if total > 0 else 0
    return {
        "timestamp": datetime.now().isoformat(),
        "params": p,
        "total_videos": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "by_query": by_query,
        "details": details,
    }


def run_grid_search(model, ground_truth):
    """Grid search over key parameters."""
    param_grid = {
        "sim_threshold": [0.50, 0.55, 0.60, 0.65, 0.70],
        "width_ratio_threshold": [0.10, 0.15, 0.20, 0.25],
        "cooldown": [1.5, 2.0, 2.5, 3.0],
        "gap_threshold": [10, 15, 20, 25],
    }

    # Fixed params
    fixed = {"sample_fps": 4, "conf": 0.45}

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    logger.info(f"Grid search: {len(combos)} parameter combinations")
    logger.info(f"Videos: {len(ground_truth)}")

    best_accuracy = -1
    best_params = None
    best_result = None
    all_results = []

    for i, combo in enumerate(combos, 1):
        params = {**fixed}
        for k, v in zip(keys, combo):
            params[k] = v

        result = run_benchmark(model, ground_truth, params=params)
        acc = result["accuracy"]
        all_results.append({"params": params, "accuracy": acc, "correct": result["correct"]})

        if acc > best_accuracy:
            best_accuracy = acc
            best_params = params.copy()
            best_result = result

        if i % 20 == 0 or i == len(combos):
            logger.info(f"  [{i}/{len(combos)}] Current best: {best_accuracy:.1%} "
                        f"({best_result['correct']}/{best_result['total_videos']})")

    logger.info(f"\nBest accuracy: {best_accuracy:.1%}")
    logger.info(f"Best params: {best_params}")

    # Also show top 5
    all_results.sort(key=lambda x: -x["accuracy"])
    logger.info("\nTop 5 parameter sets:")
    for r in all_results[:5]:
        logger.info(f"  {r['accuracy']:.1%} ({r['correct']}) - {r['params']}")

    return best_params, best_result, all_results


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO kill detection")
    parser.add_argument("--per-query", type=int, default=20,
                        help="Videos per search query (default: 20)")
    parser.add_argument("--queries", nargs="+", default=None,
                        help="Custom queries (default: cs2 1k..5k)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing videos")
    parser.add_argument("--tune", action="store_true",
                        help="Run parameter grid search")
    args = parser.parse_args()

    queries = args.queries or DEFAULT_QUERIES

    logger.info("=" * 60)
    logger.info("CS2 Kill Detection Benchmark")
    logger.info("=" * 60)

    # Load YOLO model
    from ultralytics import YOLO
    yolo_path = BASE_DIR / "models" / "best.pt"
    if not yolo_path.exists():
        logger.error(f"YOLO model not found: {yolo_path}")
        return
    model = YOLO(str(yolo_path))
    logger.info("YOLO model loaded")

    # Load or build ground truth
    ground_truth = load_ground_truth()

    if not args.skip_download and not args.tune:
        # Search and download new videos
        videos = search_and_collect(queries, args.per_query)
        if not videos:
            logger.error("No videos found from Medal.tv search")
            if not ground_truth:
                return

        # Filter out already-in-ground-truth videos
        existing_ids = {e.get("content_id") for e in ground_truth}
        new_videos = [v for v in videos if v["content_id"] not in existing_ids]
        logger.info(f"New videos to download: {len(new_videos)}")

        if new_videos:
            downloaded = download_videos(new_videos)
            ground_truth.extend(downloaded)
            save_ground_truth(ground_truth)
            logger.info(f"Ground truth updated: {len(ground_truth)} total videos")

    # Filter to videos that exist on disk
    valid = [e for e in ground_truth
             if e.get("video_path") and Path(e["video_path"]).exists()]
    logger.info(f"Valid videos for benchmark: {len(valid)}")

    if not valid:
        logger.error("No valid videos found. Run without --skip-download first.")
        return

    if args.tune:
        # Grid search
        logger.info("\n" + "=" * 60)
        logger.info("PARAMETER GRID SEARCH")
        logger.info("=" * 60)

        best_params, best_result, all_results = run_grid_search(model, valid)

        # Save best params
        BEST_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BEST_PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "params": best_params,
                "accuracy": best_result["accuracy"],
                "total_videos": best_result["total_videos"],
                "correct": best_result["correct"],
                "tuned_at": datetime.now().isoformat(),
            }, f, indent=2)
        logger.info(f"Best params saved to {BEST_PARAMS_FILE}")

        # Save full results
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(best_result, f, indent=2, ensure_ascii=False)
        logger.info(f"Full results saved to {RESULTS_FILE}")

    else:
        # Single benchmark run with current/default params
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK RUN")
        logger.info("=" * 60)

        # Use tuned params if available, else defaults
        params = DEFAULT_PARAMS.copy()
        if BEST_PARAMS_FILE.exists():
            try:
                with open(BEST_PARAMS_FILE, "r") as f:
                    saved = json.load(f)
                params.update(saved.get("params", {}))
                logger.info(f"Using tuned params from {BEST_PARAMS_FILE}")
            except Exception:
                pass

        result = run_benchmark(model, valid, params=params)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"RESULTS: {result['correct']}/{result['total_videos']} "
                     f"({result['accuracy']:.1%} accuracy)")
        logger.info(f"Params: {result['params']}")

        for query, stats in sorted(result["by_query"].items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            logger.info(f"  {query}: {stats['correct']}/{stats['total']} ({acc:.0%})")

        # Show failures
        failures = [d for d in result["details"] if not d["correct"]]
        if failures:
            logger.info(f"\nFailed videos ({len(failures)}):")
            for f in failures:
                logger.info(f"  {f['video']}: detected={f['detected']} "
                             f"expected={f['expected']} [{f['width_method']}]")

        # Save results
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
