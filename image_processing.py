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
def _find_dots_with_mode(gray_image, mode):
    """Helper: detect blob-like dots using a given threshold polarity.
    Filter sizes scale with image resolution instead of using fixed
    pixel counts, so this works for both small and high-res photos.
    """

    h, w = gray_image.shape[:2]
    diag = float(np.sqrt(h ** 2 + w ** 2))

    # Scale-relative radius bounds (tune the multipliers if needed)
    min_radius = max(2, diag * 0.0015)
    max_radius = max(min_radius + 1, diag * 0.02)
    min_area = max(3, np.pi * (min_radius ** 2) * 0.3)

    gray = cv2.equalizeHist(gray_image)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        mode + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

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

        if area < min_area:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)

        if radius < min_radius or radius > max_radius:
            continue

        # Circularity check: dots should be roughly round,
        # this filters out short line/curve fragments.
        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        if circularity < 0.55:
            continue

        M = cv2.moments(cnt)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        duplicate = False

        for (dx, dy, dr) in dots:
            if np.sqrt((cx - dx) ** 2 + (cy - dy) ** 2) < min_radius * 2:
                duplicate = True
                break

        if not duplicate:
            dots.append((cx, cy, int(radius)))

    return dots


def detect_dots(gray_image):

    # Try both polarities (dark dots on light background,
    # and light dots on dark background) and keep whichever
    # finds more valid dot-shaped blobs.
    candidates = [
        _find_dots_with_mode(gray_image, cv2.THRESH_BINARY_INV),
        _find_dots_with_mode(gray_image, cv2.THRESH_BINARY),
    ]

    return max(candidates, key=len)


# ---------------------------------
# Original vs Recreated Difference Heatmap
# ---------------------------------
def compare_with_recreation(original_bgr, recreated_bgr):
    """Compare the original Kolam photo with the recreated version.

    Both images are resized to the same shape, converted to structural
    (edge) representations so that photo lighting/background differences
    don't dominate the comparison, then an absolute difference is
    computed and rendered as a color heatmap overlay.

    Returns:
        heatmap_overlay (BGR image), match_score (float 0-100)
    """

    h, w = 500, 500

    orig_resized = cv2.resize(original_bgr, (w, h))
    recreated_resized = cv2.resize(recreated_bgr, (w, h))

    orig_gray = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2GRAY)
    recreated_gray = cv2.cvtColor(recreated_resized, cv2.COLOR_BGR2GRAY)

    # Structural representation: edges only, so a photo's background
    # colour/lighting doesn't get compared against the plain white
    # recreation canvas.
    orig_edges = cv2.Canny(cv2.GaussianBlur(orig_gray, (5, 5), 0), 50, 150)
    recreated_edges = cv2.Canny(recreated_gray, 50, 150)

    # Thicken lines slightly so near-matching strokes still overlap
    kernel = np.ones((5, 5), np.uint8)
    orig_dilated = cv2.dilate(orig_edges, kernel, iterations=1)
    recreated_dilated = cv2.dilate(recreated_edges, kernel, iterations=1)

    diff = cv2.absdiff(orig_dilated, recreated_dilated)

    # Match score: how much of the structure overlaps vs differs
    total_structure = np.count_nonzero(orig_dilated) + np.count_nonzero(recreated_dilated)
    mismatched = np.count_nonzero(diff)

    if total_structure == 0:
        match_score = 0.0
    else:
        match_score = max(0.0, 100 - (mismatched / total_structure) * 100)

    # Build heatmap: overlay differences in color on top of the
    # original image so it's visually obvious where they diverge.
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    base = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(base, 0.6, heatmap, 0.4, 0)

    return overlay, round(match_score, 2)
