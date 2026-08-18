import cv2
import numpy as np

# -----------------------------
# Count Contours
# -----------------------------
def count_contours(contours):
    return len(contours)


# -----------------------------
# Estimate Symmetry
# -----------------------------
def estimate_symmetry(gray_image):

    # Flip image horizontally
    flipped = cv2.flip(gray_image, 1)

    # Calculate difference
    difference = cv2.absdiff(gray_image, flipped)

    # Calculate similarity percentage
    similarity = 100 - (
        np.sum(difference)
        / (gray_image.shape[0] * gray_image.shape[1] * 255)
    ) * 100

    # Decide symmetry level
    if similarity >= 85:
        symmetry = "Highly Symmetric"

    elif similarity >= 70:
        symmetry = "Moderately Symmetric"

    else:
        symmetry = "Low Symmetry"

    return symmetry, similarity


# -----------------------------
# Full Symmetry Analysis
# (Horizontal + Vertical + Rotational)
# -----------------------------
def _similarity_percent(gray_image, transformed):
    difference = cv2.absdiff(gray_image, transformed)
    return 100 - (
        np.sum(difference)
        / (gray_image.shape[0] * gray_image.shape[1] * 255)
    ) * 100


def estimate_full_symmetry(gray_image, threshold=85):
    """Analyze horizontal, vertical, and 180-degree rotational
    symmetry, and summarize the overall symmetry type.

    Returns a dict with each axis's similarity percentage, whether it
    passes the symmetry threshold, and an overall descriptive label
    (e.g. "4-Fold Symmetric", "Bi-Axial Symmetric",
    "Single-Axis Symmetric", "Asymmetric").
    """

    # Horizontal axis: mirror left-right (flip across the vertical
    # center line)
    horizontal_flip = cv2.flip(gray_image, 1)
    horizontal_similarity = _similarity_percent(gray_image, horizontal_flip)

    # Vertical axis: mirror top-bottom (flip across the horizontal
    # center line)
    vertical_flip = cv2.flip(gray_image, 0)
    vertical_similarity = _similarity_percent(gray_image, vertical_flip)

    # Rotational: 180-degree rotation about the center
    rotated_180 = cv2.rotate(gray_image, cv2.ROTATE_180)
    rotational_similarity = _similarity_percent(gray_image, rotated_180)

    horizontal_pass = horizontal_similarity >= threshold
    vertical_pass = vertical_similarity >= threshold
    rotational_pass = rotational_similarity >= threshold

    axes_passed = sum([horizontal_pass, vertical_pass, rotational_pass])

    if horizontal_pass and vertical_pass and rotational_pass:
        symmetry_type = "4-Fold Symmetric"
    elif axes_passed == 2:
        symmetry_type = "Bi-Axial Symmetric"
    elif axes_passed == 1:
        symmetry_type = "Single-Axis Symmetric"
    else:
        symmetry_type = "Asymmetric"

    return {
        "horizontal_similarity": round(horizontal_similarity, 2),
        "vertical_similarity": round(vertical_similarity, 2),
        "rotational_similarity": round(rotational_similarity, 2),
        "horizontal_symmetric": horizontal_pass,
        "vertical_symmetric": vertical_pass,
        "rotational_symmetric": rotational_pass,
        "symmetry_type": symmetry_type,
    }


# -----------------------------
# Pattern Classification
# -----------------------------
def classify_pattern(dot_count, contour_count, similarity, complexity):

    # Sikku Kolam
    if (
        dot_count >= 30
        and contour_count >= 20
        and similarity >= 80
        and complexity >= 70
    ):
        return "Sikku Kolam"

    # Pulli Kolam
    elif (
        dot_count >= 20
        and contour_count < 20
    ):
        return "Pulli Kolam"

    # Straight Line Kolam
    elif contour_count <= 10:
        return "Straight Line Kolam"

    # Curved Kolam
    else:
        return "Curved Kolam"


# -----------------------------
# Complexity Score
# -----------------------------
def calculate_complexity(dot_count, contour_count):

    score = (dot_count * 0.6) + (contour_count * 2)

    if score > 100:
        score = 100

    return round(score, 2)


# -----------------------------
# Grid Size Estimation
# -----------------------------
def estimate_grid_size(dot_count):

    size = round(np.sqrt(dot_count))

    return f"{size} x {size}"


# -----------------------------
# Design Style
# -----------------------------
def get_design_style(pattern):

    if pattern == "Sikku Kolam":
        return "Curved Loop"

    elif pattern == "Pulli Kolam":
        return "Dot Matrix"

    elif pattern == "Straight Line Kolam":
        return "Straight Line"

    else:
        return "Mixed Design"


# -----------------------------
# Difficulty Level
# -----------------------------
def get_difficulty(complexity):

    if complexity >= 80:
        return "Hard"

    elif complexity >= 50:
        return "Medium"

    else:
        return "Easy"


# -----------------------------
# Drawing Method
# -----------------------------
def get_drawing_method(pattern):

    if pattern == "Sikku Kolam":
        return "Continuous Line"

    elif pattern == "Pulli Kolam":
        return "Dot Connection"

    else:
        return "Free Hand"


# -----------------------------
# Estimated Drawing Time
# -----------------------------
def estimate_time(dot_count):

    minutes = max(5, dot_count // 3)

    return f"{minutes} Minutes"


# -----------------------------
# Recreation Status
# -----------------------------
def recreation_status():

    return "Completed"

def confidence_score(similarity, complexity):

    confidence = (similarity * 0.7) + (complexity * 0.3)

    if confidence > 100:
        confidence = 100

    return round(confidence, 2)