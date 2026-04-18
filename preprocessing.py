"""
Clean up the raw image and produce a binary version that is ready for symbol detection.
0. load image and convert to grayscale
1. upscale small images to preserve thin strokes
2. apply constrast enhancement (CLAHE) to boost faint lines locally
3. gentle denoising (Gaussian blur) to smooth out noise while keeping strokes intact
4. dual thresholding:
   - Otsu as the reliable baseline for all high-contrast strokes (wires, resistor bodies)
   - Adaptive thresholding to rescue thin curved strokes (inductor arcs) that Otsu misses
5. combine thresholds with bitwise OR to get a comprehensive binary image
6. morphological closing to reconnect broken arc segments
7. remove small blobs to clean up noise from adaptive thresholding
"""

import cv2
import numpy as np

# Parameters for preprocessing tuned for typical schematic images
RESCALE_THRESHOLD = 800  # if either dimension is below this, we upscale
SCALE = 2.0 # upscale factor for small images to preserve thin strokes (e.g. inductor arcs) during processing

CLAHE_TILE_SIZE = (4, 4) # smaller tiles boost local contrast for thin strokes without over-amplifying noise
CLAHE_CLIP_LIMIT = 3.0 # cap how aggressively it amplifies (low values prevent noise from being over-boosted)

GAUSSIAN_KERNEL_SIZE = (3, 3) # small kernel for gentle denoising that preserves edges

ADAPTIVE_C_THRESHOLD = 3 # intentionally low to catch faint arcs; noise is handled downstream

MORPHOLOGICAL_CLOSING_KERNEL_SIZE = (3, 3) # reconnect broken arcs without merging nearby symbols (3x3 bridges gaps of 1–2px)

NOISE_CLEANUP_MIN_AREA = 10 # remove tiny blobs smaller than this area (tuned to remove speckle noise without affecting real strokes)


def preprocess(image_path: str, debug: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for a circuit image
    Returns:
        original_gray  : grayscale version of the original (used for cropping later)
        binary         : clean binarized image ready for detection
    """
    debug_steps = []

    # 0. Load image and convert to grayscale
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug: debug_steps.append(("0 - grayscale", gray.copy()))

    # 1. Upscale small images before any processing
    #    Inductor coil arcs are often only 2-3px thick on small schematics.
    #    Doubling the resolution gives thresholding and morphology more to work with,
    #    and makes the Hough transform far less likely to detect symbol edges as wires.
    h, w = gray.shape[:2]
    if max(h, w) < RESCALE_THRESHOLD:
        gray = cv2.resize(gray, (int(w * SCALE), int(h * SCALE)), interpolation=cv2.INTER_CUBIC)
    if debug: debug_steps.append(("1 - upscaled", gray.copy()))

    # 2. Constrast enhancement (CLAHE)
    #    smaller tile to boost thin curved strokes locally
    if False: # disable CLAHE for now since it adds more noise than it saves
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
        enhanced = clahe.apply(gray)
        if debug: debug_steps.append(("2 - enhanced", enhanced.copy()))

        # 3. Gaussian blur
        denoised = cv2.GaussianBlur(enhanced, ksize=GAUSSIAN_KERNEL_SIZE, sigmaX=0)
        if debug: debug_steps.append(("3 - denoised", denoised.copy()))

    # 4. dual thresholding
    #    both will use THRESH_BINARY_INV so dark strokes on white background 
    #    become white on black (easier to work with)
    # 4.1 Otsu 
    #     Finds one global threshold that best separates dark pixels from light ones. 
    #     Works great for bold, high-contrast lines. Misses faint or thin strokes.

    _, binary_otsu = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    if debug: debug_steps.append(("4.1 - Otsu", binary_otsu.copy()))

    if False: # disable adaptive thresholding for now since it adds more noise than it saves

        # 4.2 Adaptive (rescue thin curved strokes that Otsu drops)
        #    Computes a different threshold for every pixel based on its local neighborhood. 
        #    Catches faint arcs that are only slightly darker than their surroundings. 
        #    The tradeoff is it introduces more noise.
        block = _adaptive_block_size(gray)
        binary_adaptive = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block, # blockSize is scaled with the image so it stays proportional after upscaling.
            C= ADAPTIVE_C_THRESHOLD
        )
        if debug: debug_steps.append(("4.2 - Adaptive", binary_adaptive.copy()))

        # 5. Combine
        #    A pixel is white in the result if either method said it was a stroke
        binary = cv2.bitwise_or(binary_otsu, binary_adaptive)
        if debug: debug_steps.append(("5 - Combined", binary.copy()))

        # 6. Morphological closing 
        #    Reconnect broken arc segments without merging nearby symbols
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPHOLOGICAL_CLOSING_KERNEL_SIZE)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        if debug: debug_steps.append(("6 - Morphological closing", binary.copy()))

    # 7. Noise cleanup 
    #    Remove isolated specks introduced by adaptive threshold
    binary = _remove_small_blobs(binary_otsu, min_area=NOISE_CLEANUP_MIN_AREA)
    if debug: debug_steps.append(("7 - Noise cleanup", binary.copy()))

    if debug:
        for title, img in debug_steps:
            display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
            cv2.imshow(title, display)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    
    return gray, binary


def _adaptive_block_size(gray: np.ndarray) -> int:
    """
    Computes an adaptive blockSize for adaptiveThreshold proportional to image size.
    Larger images need a larger neighborhood to capture local contrast properly.
    Always returns an odd number >= 7 as required by OpenCV.
    """
    h, w = gray.shape[:2]
    # ~2% of the shorter dimension, clamped to [7, 31]
    block = int(min(h, w) * 0.02)
    block = max(7, min(31, block))
    if block % 2 == 0:
        block += 1
    return block


def _remove_small_blobs(binary: np.ndarray, min_area: int) -> np.ndarray:
    """
    Removes connected components smaller than min_area pixels.
    Cleans up speckle noise introduced by the adaptive threshold step
    without touching real symbol strokes.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


def estimate_stroke_width(binary: np.ndarray) -> int:
    """
    Estimates the average stroke width in the binary image.
    Used to set the wire-erasure thickness in detection.py.
    """
    run_lengths = []
    for row in binary:
        in_run = False
        run_len = 0
        for pixel in row:
            if pixel == 255:
                in_run = True
                run_len += 1
            else:
                if in_run and 1 <= run_len <= 20:
                    run_lengths.append(run_len)
                in_run = False
                run_len = 0

    if not run_lengths:
        return 3
    return max(1, int(np.median(run_lengths)))