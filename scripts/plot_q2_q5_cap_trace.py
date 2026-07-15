#!/usr/bin/env python3
import argparse
import html
import math
import re
from pathlib import Path


LOG_PATH = Path("visualizations/sift_20m_q2_q5_cap_trace.out")
OUT_PATH = Path("visualizations/q2_q5_cap_height_and_centroid_distance.svg")
QUERIES = [2, 5]
PHASE = "before_iteration"
BATCH_SIZE = 50
MAX_RANK = 1000


def esc(value):
    return html.escape(str(value), quote=True)


def parse_log(path):
    candidates = {q: [] for q in QUERIES}
    batches = {q: {} for q in QUERIES}
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("DYNAMIC_BATCH_CANDIDATE"):
            d = dict(re.findall(r"(\w+)=([^ ]+)", line))
            q = int(d["query"])
            if d.get("phase") != PHASE or q not in candidates:
                continue
            rank = int(d["selected_rank"]) + 1
            if rank > MAX_RANK:
                continue
            candidates[q].append(
                {
                    "rank": rank,
                    "batch": int(d["batch"]),
                    "selected_l1": int(d["selected_l1"]),
                    "mini_id": int(d["mini_id"]),
                    "cap": float(d["l1_cap_height_score"]),
                    "dist": float(d["query_distance"]),
                    "radius": float(d["radius"]),
                    "q_r": float(d["q_radius_after"]),
                }
            )
        elif line.startswith("DYNAMIC_BATCH_QR"):
            d = dict(re.findall(r"(\w+)=([^ ]+)", line))
            q = int(d["query"])
            if d.get("phase") != PHASE or q not in batches:
                continue
            batches[q][int(d["selected_l1"])] = {
                "batch": int(d["batch"]),
                "recall": float(d["recall"]),
                "q_r": float(d["q_radius_after"]),
            }

    for q in QUERIES:
        candidates[q].sort(key=lambda row: row["rank"])
    return candidates, batches


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


class Svg:
    def __init__(self, width, height):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fafafa"/>',
        ]

    def add(self, text):
        self.parts.append(text)

    def text(self, x, y, text, size=13, fill="#111827", weight="normal", anchor="middle"):
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(text)}</text>'
        )

    def finish(self):
        self.parts.append("</svg>")
        return "\n".join(self.parts)


def make_transform(xmin, xmax, ymin, ymax, px, py, pw, ph):
    def tx(x, y):
        return (
            px + (x - xmin) / (xmax - xmin) * pw,
            py + ph - (y - ymin) / (ymax - ymin) * ph,
        )
    return tx


def polyline(rows, x_key, y_key, tx):
    return " ".join(f"{tx(row[x_key], row[y_key])[0]:.2f},{tx(row[x_key], row[y_key])[1]:.2f}" for row in rows)


def draw_axes(svg, px, py, pw, ph, xmin, xmax, ymin, ymax, y_ticks, title, y_label):
    svg.add(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="14" fill="white" stroke="#e5e7eb"/>')
    svg.text(px + pw / 2, py + 24, title, size=15, weight="bold")
    plot = (px + 58, py + 46, pw - 78, ph - 80)
    tx = make_transform(xmin, xmax, ymin, ymax, *plot)
    x0, y0 = tx(xmin, ymin)
    x1, y1 = tx(xmax, ymax)
    svg.add(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y0:.2f}" stroke="#374151"/>')
    svg.add(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y1:.2f}" stroke="#374151"/>')
    for x in range(0, MAX_RANK + 1, 100):
        gx, gy0 = tx(max(1, x), ymin)
        _, gy1 = tx(max(1, x), ymax)
        svg.add(f'<line x1="{gx:.2f}" y1="{gy0:.2f}" x2="{gx:.2f}" y2="{gy1:.2f}" stroke="#f3f4f6"/>')
        if x > 0:
            svg.text(gx, gy0 + 18, str(x), size=10)
    for y in y_ticks:
        gx0, gy = tx(xmin, y)
        gx1, _ = tx(xmax, y)
        svg.add(f'<line x1="{gx0:.2f}" y1="{gy:.2f}" x2="{gx1:.2f}" y2="{gy:.2f}" stroke="#e5e7eb"/>')
        svg.text(gx0 - 8, gy + 4, f"{y:.2f}" if ymax <= 1.5 else f"{y:.0f}", size=10, anchor="end")
    svg.text(px + pw / 2, py + ph - 14, "selected L1 centroid rank", size=11)
    svg.text(px + 18, py + ph / 2, y_label, size=11)
    return tx


def draw_batch_lines(svg, tx, ymin, ymax):
    for rank in range(BATCH_SIZE, MAX_RANK + 1, BATCH_SIZE):
        x, y0 = tx(rank, ymin)
        _, y1 = tx(rank, ymax)
        color = "#d1d5db" if rank % 100 else "#9ca3af"
        width = "0.7" if rank % 100 else "1.0"
        svg.add(f'<line x1="{x:.2f}" y1="{y0:.2f}" x2="{x:.2f}" y2="{y1:.2f}" stroke="{color}" stroke-width="{width}" opacity="0.7"/>')


def draw_query(svg, q, rows, batch_rows, px, py, pw, ph):
    caps = [r["cap"] for r in rows]
    dists = [r["dist"] for r in rows]
    qrs = [r["q_r"] for r in rows]
    recall_1000 = batch_rows.get(1000, {}).get("recall", float("nan"))
    min_dist = min(dists)
    threshold_19 = 1.9 * min_dist
    threshold_20 = 2.0 * min_dist
    count_19 = sum(1 for d in dists if d <= threshold_19)
    count_20 = sum(1 for d in dists if d <= threshold_20)
    max_cap = max(caps)
    cap_threshold_19 = max_cap / 1.9
    cap_threshold_20 = max_cap / 2.0
    cap_count_19 = sum(1 for c in caps if c >= cap_threshold_19)
    cap_count_20 = sum(1 for c in caps if c >= cap_threshold_20)

    svg.text(px + pw / 2, py - 10, f"Query {q}: first 1000 selected L1 centroids, cap-height ordering", size=19, weight="bold")
    stats = (
        f"cap p5/p50/p95={percentile(caps, 5):.3f}/{percentile(caps, 50):.3f}/{percentile(caps, 95):.3f}, "
        f"distance p5/p50/p95={percentile(dists, 5):.1f}/{percentile(dists, 50):.1f}/{percentile(dists, 95):.1f}, "
        f"recall@1000={recall_1000:.1f}, <=1.9x={count_19}, <=2.0x={count_20}"
    )
    svg.text(px + pw / 2, py + 12, stats, size=12, fill="#4b5563")

    cap_tx = draw_axes(
        svg, px, py + 28, pw, ph,
        1, MAX_RANK, 0.0, 1.03, [0.0, 0.25, 0.50, 0.75, 1.00],
        "cap_height_score after each batch q_r update",
        "cap score",
    )
    draw_batch_lines(svg, cap_tx, 0.0, 1.03)
    svg.add(f'<polyline points="{polyline(rows, "rank", "cap", cap_tx)}" fill="none" stroke="#7c3aed" stroke-width="2.0"/>')
    for factor, threshold, count, color in [
        (1.9, cap_threshold_19, cap_count_19, "#059669"),
        (2.0, cap_threshold_20, cap_count_20, "#ea580c"),
    ]:
        p1 = cap_tx(1, threshold)
        p2 = cap_tx(MAX_RANK, threshold)
        svg.add(
            f'<line x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" x2="{p2[0]:.2f}" y2="{p2[1]:.2f}" '
            f'stroke="{color}" stroke-width="1.8" stroke-dasharray="9 5"/>'
        )
        svg.text(
            p2[0] - 4,
            p2[1] - 5,
            f"max/{factor:.1f} cap={threshold:.3f}, n={count}",
            size=10,
            fill=color,
            anchor="end",
        )
    for row in rows[::20]:
        x, y = cap_tx(row["rank"], row["cap"])
        svg.add(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.0" fill="#7c3aed"/>')

    dist_min = min(min(dists), min(qrs)) * 0.985
    dist_max = max(max(dists), max(qrs), threshold_19, threshold_20) * 1.015
    step_rows = [{"rank": r["rank"], "q_r": r["q_r"]} for r in rows]
    dist_tx = draw_axes(
        svg, px, py + ph + 50, pw, ph,
        1, MAX_RANK, dist_min, dist_max,
        [dist_min, (dist_min + dist_max) / 2.0, dist_max],
        "centroid distance to query for the same selected L1s",
        "distance",
    )
    draw_batch_lines(svg, dist_tx, dist_min, dist_max)
    svg.add(f'<polyline points="{polyline(rows, "rank", "dist", dist_tx)}" fill="none" stroke="#2563eb" stroke-width="2.0"/>')
    svg.add(f'<polyline points="{polyline(step_rows, "rank", "q_r", dist_tx)}" fill="none" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="6 5"/>')
    for factor, threshold, color in [(1.9, threshold_19, "#059669"), (2.0, threshold_20, "#ea580c")]:
        p1 = dist_tx(1, threshold)
        p2 = dist_tx(MAX_RANK, threshold)
        svg.add(
            f'<line x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" x2="{p2[0]:.2f}" y2="{p2[1]:.2f}" '
            f'stroke="{color}" stroke-width="1.8" stroke-dasharray="9 5"/>'
        )
        svg.text(
            p2[0] - 4,
            p2[1] - 5,
            f"{factor:.1f}x min={threshold:.1f}",
            size=10,
            fill=color,
            anchor="end",
        )
    for row in rows[::20]:
        x, y = dist_tx(row["rank"], row["dist"])
        svg.add(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.0" fill="#2563eb"/>')
    svg.add(f'<rect x="{px + pw - 245}" y="{py + ph + 88}" width="220" height="92" rx="8" fill="white" stroke="#d4d4d8"/>')
    svg.add(f'<line x1="{px + pw - 205}" y1="{py + ph + 108}" x2="{px + pw - 165}" y2="{py + ph + 108}" stroke="#2563eb" stroke-width="2"/>')
    svg.text(px + pw - 155, py + ph + 112, "centroid distance", size=11, anchor="start")
    svg.add(f'<line x1="{px + pw - 205}" y1="{py + ph + 128}" x2="{px + pw - 165}" y2="{py + ph + 128}" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="6 5"/>')
    svg.text(px + pw - 155, py + ph + 132, "q_r after batch", size=11, anchor="start")
    svg.add(f'<line x1="{px + pw - 205}" y1="{py + ph + 148}" x2="{px + pw - 165}" y2="{py + ph + 148}" stroke="#059669" stroke-width="1.8" stroke-dasharray="9 5"/>')
    svg.text(px + pw - 155, py + ph + 152, f"1.9x min: {count_19}/1000", size=11, anchor="start")
    svg.add(f'<line x1="{px + pw - 205}" y1="{py + ph + 168}" x2="{px + pw - 165}" y2="{py + ph + 168}" stroke="#ea580c" stroke-width="1.8" stroke-dasharray="9 5"/>')
    svg.text(px + pw - 155, py + ph + 172, f"2.0x min: {count_20}/1000", size=11, anchor="start")


def main():
    global LOG_PATH, OUT_PATH, QUERIES

    parser = argparse.ArgumentParser(description="Plot cap-height traces for two queries.")
    parser.add_argument("--log", default=str(LOG_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--queries", default="2,5", help="Two comma-separated query ids, e.g. 4,5")
    args = parser.parse_args()

    QUERIES = [int(q) for q in args.queries.split(",") if q.strip()]
    if len(QUERIES) != 2:
        raise ValueError("--queries must contain exactly two query ids")
    LOG_PATH = Path(args.log)
    OUT_PATH = Path(args.out)

    rows, batches = parse_log(LOG_PATH)
    OUT_PATH.parent.mkdir(exist_ok=True)
    svg = Svg(1500, 1320)
    svg.text(750, 38, f"q{QUERIES[0]}/q{QUERIES[1]} cap-height score and L1 centroid distance trace", size=27, weight="bold")
    svg.text(
        750,
        66,
        "Each point is one selected L1 centroid. Vertical grid lines mark 50-L1 batches. Cap scores use the q_r after that batch.",
        size=14,
        fill="#4b5563",
    )

    draw_query(svg, QUERIES[0], rows[QUERIES[0]], batches[QUERIES[0]], 60, 105, 660, 245)
    draw_query(svg, QUERIES[1], rows[QUERIES[1]], batches[QUERIES[1]], 780, 105, 660, 245)

    OUT_PATH.write_text(svg.finish())
    print(OUT_PATH)


if __name__ == "__main__":
    main()
