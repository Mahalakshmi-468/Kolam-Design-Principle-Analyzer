import cv2
import numpy as np


# ---------------------------------
# Smart Kolam Recreation
# ---------------------------------
# Instead of connecting every pair of dots that happen to be within an
# arbitrary distance (which tends to produce a messy, unrealistic tangle
# on dense grids and nothing at all on sparse ones), this builds a
# k-Nearest-Neighbor graph: each dot connects only to its closest
# neighboring dots. This mimics how a real Kolam thread moves from one
# dot to its immediate neighbors, giving a cleaner, more grid-like,
# more "hand drawn" looking recreation.
#
# Connections are rendered as gentle Bezier curves rather than straight
# lines, echoing the flowing, looped thread strokes seen in real Kolam
# art instead of a rigid technical diagram look.
def _k_nearest_neighbor_edges(points, k):
    """Return a set of (i, j) index pairs connecting each point to its
    k nearest neighbors (deduplicated, undirected)."""

    n = len(points)
    edges = set()

    if n < 2:
        return edges

    k = min(k, n - 1)
    pts = np.array(points, dtype=np.float64)

    for i in range(n):
        dists = np.linalg.norm(pts - pts[i], axis=1)
        # Exclude the point itself (distance 0), take k closest
        nearest_idx = np.argsort(dists)[1:k + 1]

        for j in nearest_idx:
            edge = (min(i, int(j)), max(i, int(j)))
            edges.add(edge)

    return edges


def _quadratic_bezier_points(p0, p1, p2, n_points=20):
    """Sample points along a quadratic Bezier curve defined by
    start p0, control p1, and end p2."""

    t = np.linspace(0, 1, n_points)
    p0 = np.array(p0, dtype=np.float64)
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)

    curve = (
        (1 - t)[:, None] ** 2 * p0
        + 2 * (1 - t)[:, None] * t[:, None] * p1
        + t[:, None] ** 2 * p2
    )

    return curve.astype(np.int32)


def _draw_curved_edge(canvas, p1, p2, bulge_sign, color, thickness):
    """Draw a smooth curved connector between p1 and p2. The curve
    bulges perpendicular to the line, alternating direction so
    adjacent threads visually interleave like a real Kolam braid."""

    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)

    mid = (p1 + p2) / 2
    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length == 0:
        return

    # Perpendicular unit vector
    perp = np.array([-direction[1], direction[0]]) / length

    bulge_amount = max(4, length * 0.18) * bulge_sign
    control = mid + perp * bulge_amount

    curve_pts = _quadratic_bezier_points(p1, control, p2, n_points=16)

    cv2.polylines(
        canvas,
        [curve_pts],
        isClosed=False,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA
    )


def recreate_kolam(dots, k_neighbors=3, smooth=True):

    # Create white canvas
    canvas = np.ones((600, 600, 3), dtype=np.uint8) * 255

    if len(dots) == 0:
        return canvas

    # Scale dots to fit the canvas
    xs = [d[0] for d in dots]
    ys = [d[1] for d in dots]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    width = max(max_x - min_x, 1)
    height = max(max_y - min_y, 1)

    scale = min(500 / width, 500 / height)

    new_points = []

    for (x, y, r) in dots:
        nx = int((x - min_x) * scale + 50)
        ny = int((y - min_y) * scale + 50)
        new_points.append((nx, ny))

    # Build the k-nearest-neighbor connection graph
    edges = _k_nearest_neighbor_edges(new_points, k_neighbors)

    # Draw the thread lines first, so dots render cleanly on top
    for idx, (i, j) in enumerate(edges):

        p1 = new_points[i]
        p2 = new_points[j]

        if smooth:
            bulge_sign = 1 if idx % 2 == 0 else -1
            _draw_curved_edge(canvas, p1, p2, bulge_sign, (40, 40, 40), 2)
        else:
            cv2.line(
                canvas,
                p1,
                p2,
                (40, 40, 40),
                2,
                lineType=cv2.LINE_AA
            )

    # Draw dots on top of the lines
    for (nx, ny) in new_points:
        cv2.circle(
            canvas,
            (nx, ny),
            4,
            (0, 0, 255),
            -1,
            lineType=cv2.LINE_AA
        )

    return canvas
