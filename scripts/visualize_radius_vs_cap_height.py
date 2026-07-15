#!/usr/bin/env python3
import html
import math
from pathlib import Path


def cap_height_score(d, q_r, r_l1):
    if d >= q_r + r_l1:
        return 0.0
    if d + r_l1 <= q_r:
        return 1.0
    x = (d * d - q_r * q_r + r_l1 * r_l1) / (2.0 * d)
    return max(0.0, min(1.0, (r_l1 - x) / r_l1))


def esc(value):
    return html.escape(str(value), quote=True)


class Svg:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            "<defs>",
            '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
            '<path d="M0,0 L0,6 L9,3 z" fill="#111827"/>',
            "</marker>",
            '<marker id="dot" markerWidth="8" markerHeight="8" refX="4" refY="4" markerUnits="strokeWidth">',
            '<circle cx="4" cy="4" r="2.7" fill="#111827"/>',
            "</marker>",
            "</defs>",
            '<rect width="100%" height="100%" fill="#fafafa"/>',
        ]

    def add(self, text):
        self.parts.append(text)

    def text(self, x, y, text, size=14, fill="#111827", weight="normal", anchor="middle"):
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(text)}</text>'
        )

    def multiline(self, x, y, lines, size=13, fill="#111827", anchor="start", leading=1.25):
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
            f'font-size="{size}" fill="{fill}" text-anchor="{anchor}">'
        )
        for i, line in enumerate(lines):
            dy = 0 if i == 0 else size * leading
            self.add(f'<tspan x="{x:.1f}" dy="{dy:.1f}">{esc(line)}</tspan>')
        self.add("</text>")

    def finish(self):
        self.parts.append("</svg>")
        return "\n".join(self.parts)


def transform(xmin, xmax, ymin, ymax, px, py, pw, ph):
    def tx(x, y):
        return (
            px + (x - xmin) / (xmax - xmin) * pw,
            py + ph - (y - ymin) / (ymax - ymin) * ph,
        )
    return tx


def cap_points(cx, cy, r, plane_x, tx, n=140):
    local_x = plane_x - cx
    if local_x <= -r:
        return []
    if local_x >= r:
        return [
            tx(cx + r * math.cos(2.0 * math.pi * i / n), cy + r * math.sin(2.0 * math.pi * i / n))
            for i in range(n + 1)
        ]
    theta = math.acos(local_x / r)
    return [
        tx(
            cx + r * math.cos(theta + (2.0 * math.pi - 2.0 * theta) * i / n),
            cy + r * math.sin(theta + (2.0 * math.pi - 2.0 * theta) * i / n),
        )
        for i in range(n + 1)
    ]


def polygon_path(points):
    if not points:
        return ""
    path = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    path.extend(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
    path.append("Z")
    return " ".join(path)


def arrow(svg, p1, p2, color="#111827", width=2.0, dashed=False):
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    svg.add(
        f'<line x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" x2="{p2[0]:.2f}" y2="{p2[1]:.2f}" '
        f'stroke="{color}" stroke-width="{width}"{dash} marker-start="url(#dot)" marker-end="url(#arrow)"/>'
    )


def draw_pair(svg, px, py, pw, ph, q_r, r_l1, margin, title):
    d = q_r + r_l1 - margin * r_l1
    qx = -d
    lx = 0.0
    x = (d * d - q_r * q_r + r_l1 * r_l1) / (2.0 * d)
    plane_x = -x
    h = r_l1 - x
    score = cap_height_score(d, q_r, r_l1)
    overlap_width = q_r + r_l1 - d
    chord_y = math.sqrt(max(0.0, r_l1 * r_l1 - x * x)) if abs(x) < r_l1 else 0.0

    pad = 0.55
    xmin = qx - q_r - pad
    xmax = r_l1 + pad
    ymin = -max(q_r, r_l1) - 1.35
    ymax = max(q_r, r_l1) + 1.55
    tx = transform(xmin, xmax, ymin, ymax, px, py, pw, ph)

    svg.add(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="18" fill="white" stroke="#e5e7eb"/>')
    svg.text(px + pw / 2, py + 29, title, size=18, weight="bold")

    cap = cap_points(lx, 0.0, r_l1, plane_x, tx)
    path = polygon_path(cap)
    if path:
        svg.add(f'<path d="{path}" fill="#fb923c" fill-opacity="0.35" stroke="none"/>')

    for cx, r, color, label in [(qx, q_r, "#2563eb", "query ball Q"), (lx, r_l1, "#f97316", "L1 ball")]:
        sc, sy = tx(cx, 0.0)
        edge, _ = tx(cx + r, 0.0)
        svg.add(f'<circle cx="{sc:.2f}" cy="{sy:.2f}" r="{abs(edge - sc):.2f}" fill="none" stroke="{color}" stroke-width="3"/>')
        svg.add(f'<circle cx="{sc:.2f}" cy="{sy:.2f}" r="5" fill="{color}"/>')
        svg.text(sc, sy - 10, label, size=12, fill=color, weight="bold")

    center_line_a = tx(qx, 0.0)
    center_line_b = tx(lx, 0.0)
    svg.add(f'<line x1="{center_line_a[0]:.2f}" y1="{center_line_a[1]:.2f}" x2="{center_line_b[0]:.2f}" y2="{center_line_b[1]:.2f}" stroke="#111827" stroke-dasharray="5 5" opacity="0.45"/>')

    plane_top = tx(plane_x, ymax)
    plane_bottom = tx(plane_x, ymin)
    svg.add(f'<line x1="{plane_top[0]:.2f}" y1="{plane_top[1]:.2f}" x2="{plane_bottom[0]:.2f}" y2="{plane_bottom[1]:.2f}" stroke="#7c3aed" stroke-width="2" stroke-dasharray="6 5" opacity="0.75"/>')
    if chord_y > 0.0:
        c1 = tx(plane_x, -chord_y)
        c2 = tx(plane_x, chord_y)
        svg.add(f'<line x1="{c1[0]:.2f}" y1="{c1[1]:.2f}" x2="{c2[0]:.2f}" y2="{c2[1]:.2f}" stroke="#7c3aed" stroke-width="5"/>')
        svg.text(c2[0] + 10, c2[1] - 8, "intersection chord", size=12, fill="#7c3aed", anchor="start")

    # Center distance d.
    d_y = -0.34 * max(q_r, r_l1)
    p1 = tx(qx, d_y)
    p2 = tx(lx, d_y)
    arrow(svg, p1, p2, width=2.0)
    svg.text((p1[0] + p2[0]) / 2, p1[1] - 9, f"d = {d:.2f}", size=13)

    # Query and L1 radii.
    qr_y = 0.44 * max(q_r, r_l1)
    p1 = tx(qx, qr_y)
    p2 = tx(qx + q_r, qr_y)
    arrow(svg, p1, p2, color="#2563eb", width=2.0)
    svg.text((p1[0] + p2[0]) / 2, p1[1] - 9, f"q_r = {q_r:.1f}", size=12, fill="#2563eb")

    p1 = tx(lx, -qr_y)
    p2 = tx(-r_l1, -qr_y)
    arrow(svg, p1, p2, color="#f97316", width=2.0)
    svg.text((p1[0] + p2[0]) / 2, p1[1] + 20, f"r_l1 = {r_l1:.1f}", size=12, fill="#f97316")

    # Old radius-margin width.
    p1 = tx(-r_l1, ymax - 0.35)
    p2 = tx(qx + q_r, ymax - 0.35)
    arrow(svg, p1, p2, color="#dc2626", width=2.2)
    svg.text((p1[0] + p2[0]) / 2, p1[1] - 8, f"overlap width = q_r + r_l1 - d = {overlap_width:.2f}", size=12, fill="#dc2626")

    # x and cap height h.
    p1 = tx(lx, 0.20 * max(q_r, r_l1))
    p2 = tx(plane_x, 0.20 * max(q_r, r_l1))
    arrow(svg, p1, p2, color="#7c3aed", width=2.0)
    svg.text((p1[0] + p2[0]) / 2, p1[1] - 8, f"x = {x:.2f}", size=12, fill="#7c3aed")

    p1 = tx(-r_l1, ymin + 0.42)
    p2 = tx(plane_x, ymin + 0.42)
    arrow(svg, p1, p2, color="#7c3aed", width=2.2)
    svg.text((p1[0] + p2[0]) / 2, p1[1] + 22, f"h = r_l1 - x = {h:.2f}", size=12, fill="#7c3aed")

    box_x = px + 16
    box_y = py + 49
    svg.add(f'<rect x="{box_x - 8}" y="{box_y - 17}" width="360" height="132" rx="10" fill="white" stroke="#d4d4d8" opacity="0.96"/>')
    svg.multiline(box_x, box_y, [
        "radius_margin = (q_r + r_l1 - d) / r_l1",
        f"              = ({q_r:.1f} + {r_l1:.1f} - {d:.2f}) / {r_l1:.1f} = {margin:.2f}",
        "x = (d^2 - q_r^2 + r_l1^2) / (2d)",
        f"h = r_l1 - x = {h:.2f}",
        f"cap_height_score = h / r_l1 = {score:.2f}",
    ], size=12)


def draw_formula_panel(svg, px, py, pw, ph):
    svg.add(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="18" fill="#111827"/>')
    svg.text(px + pw / 2, py + 32, "What changed?", size=20, fill="white", weight="bold")
    svg.multiline(px + 28, py + 72, [
        "radius_margin looks only at the red boundary-overlap width.",
        "cap_height_score uses the purple intersection plane.",
        "",
        "Same red width can cut a shallow cap or a deep cap in the L1 ball.",
        "The deeper cap is more likely to contain useful vectors.",
    ], size=15, fill="#f9fafb")


def draw_curve(svg, px, py, pw, ph):
    svg.add(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="18" fill="white" stroke="#e5e7eb"/>')
    svg.text(px + pw / 2, py + 30, "Fixed radius_margin = 0.30, but cap_height_score changes", size=18, weight="bold")

    margin = 0.30
    q_r = 3.0
    xmin, xmax = 0.2, 3.0
    ymin, ymax = 0.0, 0.42
    tx = transform(xmin, xmax, ymin, ymax, px + 76, py + 62, pw - 112, ph - 112)

    x0, y0 = tx(xmin, ymin)
    x1, y1 = tx(xmax, ymax)
    svg.add(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y0:.2f}" stroke="#374151"/>')
    svg.add(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y1:.2f}" stroke="#374151"/>')

    for val in [0.0, 0.1, 0.2, 0.3, 0.4]:
        a = tx(xmin, val)
        b = tx(xmax, val)
        svg.add(f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke="#e5e7eb"/>')
        svg.text(a[0] - 12, a[1] + 4, f"{val:.1f}", size=12, anchor="end")
    for val in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        a = tx(val, ymin)
        b = tx(val, ymax)
        svg.add(f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke="#f3f4f6"/>')
        svg.text(a[0], a[1] + 22, f"{val:.1f}", size=12)

    a = tx(xmin, margin)
    b = tx(xmax, margin)
    svg.add(f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke="#dc2626" stroke-width="3" stroke-dasharray="8 6"/>')

    points = []
    for i in range(200):
        ratio = xmin + (xmax - xmin) * i / 199
        r_l1 = ratio * q_r
        d = q_r + r_l1 - margin * r_l1
        points.append(tx(ratio, cap_height_score(d, q_r, r_l1)))
    svg.add('<polyline points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in points) + '" fill="none" stroke="#7c3aed" stroke-width="4"/>')

    for ratio, label in [(1.0 / 3.0, "small L1 example"), (5.0 / 3.0, "large L1 example")]:
        r_l1 = ratio * q_r
        d = q_r + r_l1 - margin * r_l1
        x, y = tx(ratio, cap_height_score(d, q_r, r_l1))
        svg.add(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="#111827"/>')
        svg.text(x + 10, y - 10, label, size=12, anchor="start")

    svg.text(px + pw / 2, py + ph - 13, "r_l1 / q_r", size=14)
    svg.text(px + 20, py + ph / 2, "score", size=14)
    svg.add(f'<rect x="{px + pw - 330}" y="{py + 58}" width="300" height="62" rx="10" fill="white" stroke="#d4d4d8"/>')
    svg.add(f'<line x1="{px + pw - 310}" y1="{py + 79}" x2="{px + pw - 260}" y2="{py + 79}" stroke="#dc2626" stroke-width="3" stroke-dasharray="8 6"/>')
    svg.text(px + pw - 250, py + 84, "radius_margin fixed", size=12, anchor="start")
    svg.add(f'<line x1="{px + pw - 310}" y1="{py + 102}" x2="{px + pw - 260}" y2="{py + 102}" stroke="#7c3aed" stroke-width="4"/>')
    svg.text(px + pw - 250, py + 107, "cap_height_score varies", size=12, anchor="start")


def main():
    out_dir = Path("visualizations")
    out_dir.mkdir(exist_ok=True)

    svg = Svg(1380, 1080)
    svg.text(690, 38, "radius_margin vs l1_cap_height_score", size=30, weight="bold")
    svg.text(690, 68, "Same boundary overlap can imply different L1 covered mass.", size=16)

    draw_pair(svg, 45, 95, 620, 490, q_r=3.0, r_l1=1.0, margin=0.30,
              title="Small L1: same radius_margin")
    draw_pair(svg, 715, 95, 620, 490, q_r=3.0, r_l1=5.0, margin=0.30,
              title="Large L1: same radius_margin")
    draw_curve(svg, 45, 625, 850, 385)
    draw_formula_panel(svg, 930, 625, 405, 385)

    out = out_dir / "radius_margin_vs_cap_height.svg"
    out.write_text(svg.finish())
    print(out)


if __name__ == "__main__":
    main()
