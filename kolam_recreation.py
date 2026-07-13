import cv2
import numpy as np


def recreate_kolam(dots):

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

        cv2.circle(
            canvas,
            (nx, ny),
            4,
            (0, 0, 255),
            -1
        )

    # Connect nearby dots
    for i in range(len(new_points)):

        for j in range(i + 1, len(new_points)):

            p1 = np.array(new_points[i])
            p2 = np.array(new_points[j])

            distance = np.linalg.norm(p1 - p2)

            if distance < 70:

                cv2.line(
                    canvas,
                    tuple(p1),
                    tuple(p2),
                    (0, 0, 0),
                    1
                )

    return canvas