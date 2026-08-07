import cv2
import numpy as np
import random

from kolam_recreation import _draw_curved_edge


# ---------------------------------
# Generative Kolam Designer
# ---------------------------------
# Creates a brand new, original Kolam-style pattern from scratch,
# rather than analyzing an uploaded one. A dot grid is built, then a
# connection pattern is generated for one quadrant only and mirrored
# across the chosen symmetry axes -- exactly how traditional Kolam
# artists plan these designs around a symmetric dot grid.
def _build_dot_grid(grid_size, spacing=60, margin=70):
    """Return a grid_size x grid_size list of (x, y) dot positions."""

    points = {}

    for row in range(grid_size):
        for col in range(grid_size):
            x = margin + col * spacing
            y = margin + row * spacing
            points[(row, col)] = (x, y)

    return points


def _generate_quadrant_edges(grid_size, symmetry, density=0.55, seed=None):
    """Randomly (but reproducibly) choose grid-adjacency connections
    within the base region that, once mirrored, will fill the whole
    grid according to the chosen symmetry mode. Each dot may connect
    to its right neighbor and/or its neighbor below, loosely mimicking
    how a Sikku Kolam thread weaves between adjacent grid points."""

    rng = random.Random(seed)

    half = grid_size // 2 + 1

    # Only restrict the axes that will actually be mirrored -- an
    # axis that isn't mirrored needs edges generated across its full
    # range, otherwise that half of the design stays empty.
    row_range = half if symmetry in ("4-fold", "2-fold-horizontal") else grid_size
    col_range = half if symmetry in ("4-fold", "2-fold-vertical") else grid_size

    edges = []

    for row in range(row_range):
        for col in range(col_range):
            if col + 1 < grid_size and rng.random() < density:
                edges.append(((row, col), (row, col + 1)))
            if row + 1 < grid_size and rng.random() < density:
                edges.append(((row, col), (row + 1, col)))

    return edges


def _mirror_edge(edge, grid_size, axis):
    (r1, c1), (r2, c2) = edge

    if axis == "vertical":
        # mirror columns (left-right flip)
        c1 = grid_size - 1 - c1
        c2 = grid_size - 1 - c2
    elif axis == "horizontal":
        # mirror rows (top-bottom flip)
        r1 = grid_size - 1 - r1
        r2 = grid_size - 1 - r2
    elif axis == "both":
        r1 = grid_size - 1 - r1
        r2 = grid_size - 1 - r2
        c1 = grid_size - 1 - c1
        c2 = grid_size - 1 - c2

    return ((r1, c1), (r2, c2))


def generate_kolam_design(grid_size=7, symmetry="4-fold", density=0.55,
                           seed=None, canvas_size=600):
    """Procedurally generate a new, original Kolam design.

    grid_size : number of dots per row/column
    symmetry  : "2-fold-vertical", "2-fold-horizontal", or "4-fold"
    density   : how densely connected the base quadrant is (0-1)
    seed      : optional int for reproducible designs

    Returns: (canvas_bgr_image, dot_count, edge_count)
    """

    if seed is None:
        seed = random.randint(0, 999999)

    points = _build_dot_grid(grid_size)

    base_edges = _generate_quadrant_edges(grid_size, symmetry, density=density, seed=seed)

    all_edges = set()

    for e in base_edges:
        all_edges.add(e)

        if symmetry in ("2-fold-vertical", "4-fold"):
            all_edges.add(_mirror_edge(e, grid_size, "vertical"))

        if symmetry in ("2-fold-horizontal", "4-fold"):
            all_edges.add(_mirror_edge(e, grid_size, "horizontal"))

        if symmetry == "4-fold":
            all_edges.add(_mirror_edge(e, grid_size, "both"))

    margin = 70
    spacing = 60
    content_size = margin * 2 + (grid_size - 1) * spacing
    scale = (canvas_size - 40) / content_size

    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

    def to_canvas_xy(rc):
        x, y = points[rc]
        return (int(x * scale + 20), int(y * scale + 20))

    for idx, (a, b) in enumerate(all_edges):
        if a not in points or b not in points:
            continue

        p1 = to_canvas_xy(a)
        p2 = to_canvas_xy(b)

        bulge_sign = 1 if idx % 2 == 0 else -1
        _draw_curved_edge(canvas, p1, p2, bulge_sign, (60, 30, 90), 2)

    for rc in points:
        x, y = to_canvas_xy(rc)
        cv2.circle(canvas, (x, y), 4, (0, 0, 200), -1, lineType=cv2.LINE_AA)

    dot_count = grid_size * grid_size

    return canvas, dot_count, len(all_edges)
