import cv2
import numpy as np


def recreate_kolam(image, dots, symmetrize=True):
    """
    Builds a clean digital reconstruction of a Kolam from the
    original photograph.

    The old version used a single fixed brightness threshold (180),
    which only worked for one specific lighting/background
    combination — for most real photos it produced a blank or a
    solid white image. This version instead:

      1. Normalises contrast and uses ADAPTIVE thresholding, so it
         works whether the photo is bright, dim, or unevenly lit.
      2. Auto-detects whether the Kolam lines are the light pixels or
         the dark pixels in the image, instead of assuming lines are
         always white.
      3. Cleans the binary mask with morphological closing/opening to
         remove camera/floor noise while keeping the loops connected.
      4. (Optional) Enforces the left-right symmetry that almost all
         Kolams have, by merging the mask with its mirror image. This
         repairs small gaps caused by chalk fading on one side.
      5. Skeletonizes the result so every line is a clean, uniform
         single-pixel-wide stroke, instead of a blobby thick region.
      6. Draws the result on a black canvas with the detected dots
         marked, so the underlying dot grid used to build the Kolam
         is visible too.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold copes with uneven lighting far better than a
    # single fixed brightness cutoff. A light blur first keeps camera
    # grain from being amplified into noise by the adaptive step.
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=5,
    )

    # The Kolam lines should always be the minority of pixels. If
    # more than half the image came out white, the polarity is
    # flipped, so invert it.
    white_ratio = np.count_nonzero(binary == 255) / binary.size
    if white_ratio > 0.5:
        binary = cv2.bitwise_not(binary)

    # Morphological clean-up: close small gaps in the lines, then
    # strip out tiny isolated speckle noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    if symmetrize:
        mirrored = cv2.flip(opened, 1)
        combined = cv2.bitwise_or(opened, mirrored)
    else:
        combined = opened

    skeleton = _skeletonize(combined)

    # Render the final recreation.
    recreation = np.zeros_like(image)
    recreation[skeleton == 255] = [255, 255, 255]

    for (x, y, r) in dots:
        cv2.circle(recreation, (int(x), int(y)), 2, (0, 0, 255), -1)

    return recreation


def _skeletonize(binary_image):
    """
    Reduces a thick binary mask down to a 1-pixel-wide skeleton using
    iterative morphological erosion (no extra dependency like
    scikit-image is required).
    """
    img = binary_image.copy()
    skeleton = np.zeros(img.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Safety cap so a pathological image can never loop forever.
    for _ in range(500):
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded

        if cv2.countNonZero(img) == 0:
            break

    return skeleton
