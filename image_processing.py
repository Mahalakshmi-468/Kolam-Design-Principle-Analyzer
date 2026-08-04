import cv2
import numpy as np

# Convert image to grayscale
def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Remove noise from image
def remove_noise(gray_image):
    return cv2.GaussianBlur(gray_image, (5, 5), 0)

# Detect edges
def detect_edges(image):
    return cv2.Canny(image, 50, 150)

# Detect contours
def detect_contours(edge_image):
    contours, _ = cv2.findContours(
        edge_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours

# Detect Kolam region
def detect_kolam_region(gray_image):
    return gray_image

# Detect dots
def detect_dots(gray_image):
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=15,
        minRadius=3,
        maxRadius=10
    )

    dots = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            dots.append((x, y, r))

    return dots