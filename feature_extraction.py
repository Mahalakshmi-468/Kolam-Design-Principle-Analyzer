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