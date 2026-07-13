import cv2
import numpy as np


# ---------------------------------
# Read Image
# ---------------------------------
def read_image(image):
    return image


# ---------------------------------
# Convert to Grayscale
# ---------------------------------
def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------------------------------
# Remove Noise
# ---------------------------------
def remove_noise(gray_image):
    return cv2.GaussianBlur(gray_image, (5, 5), 0)


# ---------------------------------
# Edge Detection
# ---------------------------------
def detect_edges(image):
    return cv2.Canny(image, 50, 150)


# ---------------------------------
# Contour Detection
# ---------------------------------
def detect_contours(edge_image):

    contours, _ = cv2.findContours(
        edge_image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filtered = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 300:
            filtered.append(cnt)

    return filtered


# ---------------------------------
# Detect Kolam Region
# ---------------------------------
def detect_kolam_region(gray_image):

    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find the largest contour (Kolam region)
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return gray_image

    largest = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray_image)

    cv2.drawContours(
        mask,
        [largest],
        -1,
        255,
        -1
    )

    result = cv2.bitwise_and(gray_image, mask)

    return result


# ---------------------------------
# Detect Dots
# ---------------------------------
def detect_dots(gray_image):

    # Enhance contrast
    gray = cv2.equalizeHist(gray_image)

    # Binary inverse threshold (white dots become white blobs)
    _, thresh = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Remove noise
    kernel = np.ones((3,3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    dots = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        # Accept medium-sized blobs
        if area < 5 or area > 500:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)

        if radius < 2 or radius > 12:
            continue

        M = cv2.moments(cnt)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        duplicate = False

        for (dx, dy, dr) in dots:
            if np.sqrt((cx-dx)**2 + (cy-dy)**2) < 10:
                duplicate = True
                break

        if not duplicate:
            dots.append((cx, cy, int(radius)))

    return dots