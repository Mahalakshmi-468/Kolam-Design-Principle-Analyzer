import cv2
import numpy as np


# ---------------------------------
# Convert to Grayscale
# ---------------------------------
def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------------------------------
# Remove Noise
# ---------------------------------
# Remove Noise
# ---------------------------------
def remove_noise(gray_image):

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(gray_image)

    # Smooth image
    blur = cv2.GaussianBlur(
        enhanced,
        (5,5),
        0
    )

    return blur

# ---------------------------------
# Edge Detection
# ---------------------------------
# ---------------------------------
# Edge Detection
# ---------------------------------
def detect_edges(gray_image):

    edges = cv2.Canny(
        gray_image,
        40,
        120
    )

    return edges

# ---------------------------------
# Contour Detection
# ---------------------------------
# ---------------------------------
# Contour Detection
# ---------------------------------
def detect_contours(edge_image):

    contours, _ = cv2.findContours(
        edge_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filtered = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 150:
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

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return gray_image

    largest = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray_image)

    cv2.drawContours(mask, [largest], -1, 255, -1)

    result = cv2.bitwise_and(gray_image, mask)

    return result
# ---------------------------------
# Detect Dots
# ---------------------------------
def detect_dots(gray_image):

    # Reduce noise
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=22,
        param1=80,
        param2=22,
        minRadius=4,
        maxRadius=9
    )

    dots = []

    if circles is not None:

        circles = np.round(circles[0]).astype(int)

        for (x, y, r) in circles:

            duplicate = False

            for (dx, dy, dr) in dots:

                distance = np.sqrt((x - dx) ** 2 + (y - dy) ** 2)

                if distance < 12:
                    duplicate = True
                    break

            if not duplicate:
                dots.append((x, y, r))

    return dots