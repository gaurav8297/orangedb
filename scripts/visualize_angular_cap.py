#!/usr/bin/env python3
import argparse
import html
import math
from pathlib import Path


def esc(value):
    return html.escape(str(value), quote=True)


class Svg:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f8fafc"/>',
            "<defs>",
            '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">',
            '<path d="M0,0 L0,6 L9,3 z" fill="#111827"/>',
            "</marker>",
            "</defs>",
        ]

    def add(self, text):
        self.parts.append(text)

    def text(self, x, y, text, size=14, fill="#111827", weight="normal", anchor="middle"):
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(text)}</text>'
        )

    def multiline(self, x, y, lines, size=14, fill="#111827", leading=1.35):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" fill="{fill}">')
        for i, line in enumerate(lines):
            dy = 0 if i == 0 else size * leading
            self.add(f'<tspan x="{x:.1f}" dy="{dy:.1f}">{esc(line)}</tspan>')
        self.add("</text>")

    def panel(self, x, y, w, h, title):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="#ffffff" stroke="#d1d5db" stroke-width="2"/>')
        self.text(x + w / 2, y + 34, title, size=20, weight="bold")

    def finish(self):
        self.parts.append("</svg>")
        return "\n".join(self.parts)


def polar(cx, cy, r, angle):
    return cx + r * math.cos(angle), cy - r * math.sin(angle)


def arc_path(cx, cy, r, a0, a1):
    x0, y0 = polar(cx, cy, r, a0)
    x1, y1 = polar(cx, cy, r, a1)
    large = 1 if abs(a1 - a0) > math.pi else 0
    sweep = 0 if a1 > a0 else 1
    return f"M {x0:.2f} {y0:.2f} A {r:.2f} {r:.2f} 0 {large} {sweep} {x1:.2f} {y1:.2f}"


def line(svg, p1, p2, color="#111827", width=2.0, dashed=False, arrow=False):
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    svg.add(
        f'<line x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" x2="{p2[0]:.2f}" y2="{p2[1]:.2f}" '
        f'stroke="{color}" stroke-width="{width}"{dash}{marker}/>'
    )


def point(svg, p, label, dx=0, dy=-10, color="#111827"):
    svg.add(f'<circle cx="{p[0]:.2f}" cy="{p[1]:.2f}" r="5" fill="{color}"/>')
    svg.text(p[0] + dx, p[1] + dy, label, size=15, weight="bold", fill=color)


def geometry():
    gamma = math.radians(54.0)
    alpha_l1 = math.radians(34.0)
    alpha_q = math.radians(43.0)
    c_angle = math.radians(142)
    q_angle = c_angle - gamma
    x_angle = c_angle - alpha_l1
    cos_phi = (math.cos(alpha_q) - math.cos(gamma) * math.cos(alpha_l1)) / (
        math.sin(gamma) * math.sin(alpha_l1)
    )
    phi = math.acos(max(-1.0, min(1.0, cos_phi)))
    return gamma, alpha_l1, alpha_q, c_angle, q_angle, x_angle, phi


def draw_sphere_panel(svg):
    svg.panel(40, 40, 560, 480, "1. Angular balls on the unit sphere")
    cx, cy, r = 320, 285, 190
    gamma, alpha_l1, alpha_q, c_angle, q_angle, x_angle, _ = geometry()
    O = (cx, cy)
    C = polar(cx, cy, r, c_angle)
    Q = polar(cx, cy, r, q_angle)
    X = polar(cx, cy, r, x_angle)

    svg.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#f8fafc" stroke="#111827" stroke-width="2"/>')
    svg.add(f'<path d="{arc_path(cx, cy, r, c_angle - alpha_l1, c_angle + alpha_l1)}" '
            'fill="none" stroke="#2563eb" stroke-width="14" stroke-linecap="round" opacity="0.35"/>')
    svg.add(f'<path d="{arc_path(cx, cy, r, q_angle - alpha_q, q_angle + alpha_q)}" '
            'fill="none" stroke="#dc2626" stroke-width="14" stroke-linecap="round" opacity="0.35"/>')
    svg.add(f'<path d="{arc_path(cx, cy, r, c_angle - alpha_l1, q_angle + alpha_q)}" '
            'fill="none" stroke="#7c3aed" stroke-width="18" stroke-linecap="round" opacity="0.60"/>')

    for p, color in [(C, "#2563eb"), (Q, "#dc2626"), (X, "#7c3aed")]:
        line(svg, O, p, color=color, width=1.7, dashed=True)

    point(svg, C, "C", dx=-18, dy=24, color="#2563eb")
    point(svg, Q, "Q", dx=20, dy=-14, color="#dc2626")
    point(svg, X, "X", dx=18, dy=-12, color="#7c3aed")
    svg.add(f'<circle cx="{O[0]:.2f}" cy="{O[1]:.2f}" r="4" fill="#111827"/>')

    svg.add(f'<path d="{arc_path(cx, cy, r + 24, q_angle, c_angle)}" fill="none" stroke="#111827" stroke-width="2"/>')
    svg.text(*polar(cx, cy, r + 46, (q_angle + c_angle) / 2), "gamma = C-Q", size=14, weight="bold")

    svg.add(f'<path d="{arc_path(cx, cy, r - 28, x_angle, c_angle)}" fill="none" stroke="#2563eb" stroke-width="2"/>')
    svg.text(*polar(cx, cy, r - 58, (x_angle + c_angle) / 2), "alpha_l1", size=14, fill="#2563eb", weight="bold")

    svg.add(f'<path d="{arc_path(cx, cy, r - 66, x_angle, q_angle)}" fill="none" stroke="#dc2626" stroke-width="2"/>')
    svg.text(*polar(cx, cy, r - 95, (x_angle + q_angle) / 2), "alpha_q", size=14, fill="#dc2626", weight="bold")

    svg.add('<rect x="72" y="444" width="495" height="44" rx="10" fill="#f3f4f6" stroke="#e5e7eb"/>')
    svg.multiline(92, 464, [
        "Blue arc = L1 angular ball; red arc = query angular ball;",
        "purple arc = overlap. X is one boundary point of the cap.",
    ], size=12)


def draw_triangle_panel(svg):
    svg.panel(640, 40, 540, 310, "2. Spherical triangle C-Q-X")
    gamma, alpha_l1, alpha_q, _, _, _, phi = geometry()

    C = (760, 250)
    Q = (1030, 160)
    X = (930, 285)
    line(svg, C, Q, "#111827", 2.2)
    line(svg, C, X, "#2563eb", 2.8)
    line(svg, Q, X, "#dc2626", 2.8)
    point(svg, C, "C", dx=-15, dy=25, color="#2563eb")
    point(svg, Q, "Q", dx=18, dy=-10, color="#dc2626")
    point(svg, X, "X", dx=18, dy=18, color="#7c3aed")

    svg.text(880, 190, "gamma", size=15)
    svg.text(850, 285, "alpha_l1", size=15, fill="#2563eb")
    svg.text(1010, 230, "alpha_q", size=15, fill="#dc2626")

    # Draw phi at C.
    svg.add('<path d="M 810 233 A 60 60 0 0 1 817 260" fill="none" stroke="#f59e0b" stroke-width="4"/>')
    svg.text(826, 240, "phi", size=16, fill="#b45309", weight="bold")
    svg.add('<rect x="675" y="285" width="465" height="42" rx="10" fill="#fef3c7" stroke="#f59e0b"/>')
    svg.text(908, 311, "phi is the cap radius measured from C's perspective", size=14, fill="#92400e")


def draw_tangent_panel(svg):
    svg.panel(40, 550, 560, 270, "3. Local tangent view near C")
    base_x, base_y = 135, 735
    line(svg, (base_x, base_y), (500, base_y), "#6b7280", 2)
    C = (base_x, base_y)
    Q = (385, base_y)
    X = (365, base_y - 95)
    line(svg, C, Q, "#111827", 2.5, arrow=True)
    line(svg, C, X, "#7c3aed", 2.5, arrow=True)
    svg.add('<path d="M 205 735 A 70 70 0 0 0 199 709" fill="none" stroke="#f59e0b" stroke-width="4"/>')
    svg.text(214, 704, "phi", size=16, fill="#b45309", weight="bold")
    point(svg, C, "C", dx=-8, dy=22, color="#2563eb")
    point(svg, Q, "toward Q", dx=42, dy=20, color="#111827")
    point(svg, X, "toward X", dx=48, dy=-4, color="#7c3aed")
    svg.multiline(80, 600, [
        "Zoom in at C. Directions from C form a small local angle.",
        "The overlap covers directions within phi of the C->Q direction.",
        "cap_score ~= phi / alpha_l1, clamped to [0, 1].",
    ], size=14)


def draw_formula_panel(svg):
    svg.panel(640, 380, 540, 440, "4. Why the formula works")
    svg.multiline(680, 430, [
        "Spherical law of cosines on triangle C-Q-X:",
        "",
        "cos(alpha_q) = cos(gamma) cos(alpha_l1)",
        "             + sin(gamma) sin(alpha_l1) cos(phi)",
        "",
        "Solve for phi:",
        "",
        "cos(phi) = [cos(alpha_q) - cos(gamma) cos(alpha_l1)]",
        "           / [sin(gamma) sin(alpha_l1)]",
        "",
        "Then:",
        "phi = acos(clamp(cos(phi), -1, 1))",
        "cap_score = clamp(phi / alpha_l1, 0, 1)",
        "",
        "Higher score means the query angular ball covers",
        "more of the L1 angular ball from the L1 centroid side.",
    ], size=15)


def draw_main(svg):
    svg.text(610, 28, "Angular cap overlap for cosine / inner-product search", size=22, weight="bold")
    draw_sphere_panel(svg)
    draw_triangle_panel(svg)
    draw_tangent_panel(svg)
    draw_formula_panel(svg)

    # Formula panel.
    svg.add('<rect x="660" y="70" width="520" height="660" rx="18" fill="#ffffff" stroke="#d1d5db" stroke-width="2"/>')
    svg.text(920, 110, "Spherical cap formula", size=22, weight="bold")
    svg.multiline(700, 155, [
        "Points on the unit sphere:",
        "C = L1 centroid direction",
        "Q = query direction",
        "X = cap boundary point",
        "",
        "Triangle side lengths:",
        "C-Q = gamma",
        "C-X = alpha_l1",
        "Q-X = alpha_q",
        "",
        "Spherical law of cosines:",
        "cos(alpha_q) = cos(gamma) cos(alpha_l1)",
        "             + sin(gamma) sin(alpha_l1) cos(phi)",
        "",
        "Solve for phi:",
        "cos(phi) = [cos(alpha_q) - cos(gamma) cos(alpha_l1)]",
        "           / [sin(gamma) sin(alpha_l1)]",
        "",
        "Interpretation:",
        "larger phi => query covers more of the L1 angular ball",
        "smaller phi => only a thin cap is covered",
    ], size=15)

    svg.add('<rect x="700" y="615" width="440" height="72" rx="12" fill="#fef3c7" stroke="#f59e0b"/>')
    svg.multiline(725, 643, [
        "Practical score: cap_score = clamp(phi / alpha_l1, 0, 1)",
        "Use acos-clamping because floating point can exceed [-1, 1].",
    ], size=13)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="visualizations/angular_cap_geometry.svg")
    args = parser.parse_args()

    svg = Svg(1220, 850)
    draw_main(svg)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg.finish())
    print(out)


if __name__ == "__main__":
    main()
