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


def recreate_kolam(dots, k_neighbors=3):

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
    for (i, j) in edges:

        p1 = new_points[i]
        p2 = new_points[j]

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
