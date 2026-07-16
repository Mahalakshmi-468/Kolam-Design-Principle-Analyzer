import cv2
import numpy as np

def recreate_kolam(image, dots):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5
    )

    # Remove small noise
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Slightly thicken the Kolam lines
    binary = cv2.dilate(binary, kernel, iterations=1)

    # White canvas
    canvas = np.ones_like(image) * 255

    # Draw extracted Kolam
    canvas[binary == 255] = (0, 0, 0)

    # Draw detected dots
    for (x, y, r) in dots:
        cv2.circle(canvas, (x, y), 2, (0, 0, 255), -1)

    return canvas