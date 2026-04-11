"""
Finds candidate regions in the circuit image that may contain symbols.

Pipeline:
  0. Derive scale-aware defaults from image size
  1. Probabilistic Hough to detect straight line segments (wires) and erase them from the binary image
  2. Optional: merge symbol fragments
  3. Connected Component Analysis on the remainder to find blobs of connected white pixels
  4. Filter blobs by  size and shape
        - too small = noise
        - too elongated = a wire Hough missed
        - too large = a text label
  5. Each surviving blob becomes a Candidate (a cropped image region that might be a symbol)
"""

import cv2
import numpy as np
from dataclasses import dataclass

from preprocessing import estimate_stroke_width

@dataclass
class Candidate:
    """A region of the image that may contain a circuit symbol."""
    crop: np.ndarray        # pixel content cropped from original
    bbox: tuple             # (x, y, w, h) in original image coordinates
    blob_area: int          # area of the connected component in pixels


def _relative_defaults(
    binary: np.ndarray,
    *,
    min_line_length: int | None = None,
    min_blob_area: int | None = None,
    max_blob_area: int | None = None,
    bbox_padding: float | None = None,
) -> dict:
    """
    Compute scale-aware defaults from image size. Any non-None argument overrides
    the computed default.
    """
    h, w = binary.shape[:2]
    image_area = h * w

    if min_line_length is None:
        # Roughly 15% of image width; ensures only true wires/borders qualify.
        min_line_length = max(10, int(w * 0.15))

    if min_blob_area is None:
        # About 0.03% of the image area; keeps tiny specks out but preserves
        # small symbols on low-resolution schematics.
        min_blob_area = max(10, int(image_area * 0.0003))

    if max_blob_area is None:
        # About 5% of the image area; filters out large text blocks / frames.
        max_blob_area = int(image_area * 0.05)

    if bbox_padding is None:
        # More generous padding so small symbols aren't over-cropped.
        bbox_padding = 0.30

    return {
        "min_line_length": int(min_line_length),
        "min_blob_area": int(min_blob_area),
        "max_blob_area": int(max_blob_area),
        "bbox_padding": float(bbox_padding),
    }


def _remove_wires(
    binary: np.ndarray,
    min_line_length: int,
    max_line_gap: int,
    line_angle_tolerance_deg: float = 12.0,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Probabilistic Hough Transform to find straight line segments, then paints them black to erase them
    - It only erases near-horizontal or near-vertical segments (within 12° tolerance) to avoid cutting 
      into diagonal symbol strokes.
    - The erasure thickness is set by _estimate_erasure_thickness(), 
      which uses the estimated stroke width so it erases cleanly without damaging nearby symbols

    Returns the cleaned binary image, the list of detected lines, and the binary mask of erased wires.
    """
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    wire_mask = np.zeros_like(binary)
    erasure_thickness = _estimate_erasure_thickness(binary)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            dx = x2 - x1
            dy = y2 - y1
            angle = abs(np.degrees(np.arctan2(dy, dx))) % 180.0
            near_horizontal = angle <= line_angle_tolerance_deg or angle >= (180.0 - line_angle_tolerance_deg)
            near_vertical = abs(angle - 90.0) <= line_angle_tolerance_deg
            if length >= min_line_length and (near_horizontal or near_vertical):
                cv2.line(wire_mask, (x1, y1), (x2, y2), 255, erasure_thickness)

    binary_no_wires = cv2.subtract(binary, wire_mask)
    return binary_no_wires, lines, wire_mask

def _fragment_merge(binary_no_wires: np.ndarray, merge_gap_px: int | None) -> np.ndarray:
    """
    Closes small gaps between nearby components to reconnect fragmented symbols.
    Wire erasure sometimes cuts through a symbol at its connection point for example 
    a resistor zigzag gets severed into separate peaks. 
    Morphological closing reconnects pieces that are very close together. 
    The kernel size is based on stroke width so it doesn't accidentally merge two separate symbols.
    """
    sw = estimate_stroke_width(binary_no_wires)
    if merge_gap_px is None:
        merge_gap_px = max(1, int(sw))  # conservative default
    k = 2 * int(merge_gap_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary_for_cc = cv2.morphologyEx(binary_no_wires, cv2.MORPH_CLOSE, kernel)
    return binary_for_cc

def _near_wire_filter(binary: np.ndarray, wire_mask: np.ndarray, wire_proximity_px: int | None) -> np.ndarray:
    sw = estimate_stroke_width(binary)
    if wire_proximity_px is None:
        wire_proximity_px = max(2, int(sw * 2))
    kr = 2 * int(wire_proximity_px) + 1
    near_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kr, kr))
    wire_near = cv2.dilate(wire_mask, near_kernel)
    return wire_near

def _estimate_erasure_thickness(binary: np.ndarray) -> int:
    """
    Estimates a safe erasure thickness based on the image.
    Slightly thicker than a typical stroke so wire remnants are fully removed,
    but not so thick that nearby symbols get damaged.
    """
    sw = estimate_stroke_width(binary)
    # Use the estimated stroke width directly; thicker masks risk cutting into
    # symbol strokes at connection points on small images.
    return max(1, int(sw))

def detect_candidates(
    binary: np.ndarray,
    original_gray: np.ndarray,
    min_line_length: int | None = None,
    max_line_gap: int = 10,
    line_angle_tolerance_deg: float = 12.0,
    min_blob_area: int | None = None,
    max_blob_area: int | None = None,
    max_aspect_ratio: float = 12.0,
    bbox_padding: float | None = None,
    merge_fragments: bool = True,
    merge_gap_px: int | None = None,
    require_near_wires: bool = True,
    wire_proximity_px: int | None = None,
    debug: bool=False,
) -> list[Candidate]:
    """
    Full candidate detection pipeline.
    0. Derive scale-aware defaults from image size
    1. Hough + wire erasure
    2. Optional: merge symbol fragments
    3. Connected Component Analysis
    4. Filter blobs by size and shape
    5. Crop candidates with padding proportional to their size
    Args:
        binary          : binarized image (white strokes on black)
        original_gray   : grayscale original for cropping
        min_line_length : if None, set to 15% of image width (relative minimum length for a wire segment).
        max_line_gap    : max gap (px) allowed within a single wire segment.
        line_angle_tolerance_deg: erase only near-horizontal/vertical segments.
        min_blob_area   : if None, set to ~0.03% of image area.
        max_blob_area   : if None, set to ~5% of image area.
        max_aspect_ratio: blobs with w/h or h/w above this are likely missed wires
        bbox_padding    : if None, defaults to 0.30 (more context around small blobs)
        merge_fragments : if True, reconnect nearby fragments (e.g. resistor peaks)
        merge_gap_px    : if None, inferred from stroke width; larger merges more
        require_near_wires: drop components far from erased wires (often text/noise)
        wire_proximity_px : proximity radius for near-wire filtering
        
    Returns:
        List of Candidate objects, one per surviving blob.
    """

    debug_steps = []

    # 0. Derive scale-aware defaults from image size 
    rel = _relative_defaults(
        binary,
        min_line_length=min_line_length,
        min_blob_area=min_blob_area,
        max_blob_area=max_blob_area,
        bbox_padding=bbox_padding,
    )
    min_line_length = rel["min_line_length"]
    min_blob_area = rel["min_blob_area"]
    max_blob_area = rel["max_blob_area"]
    bbox_padding = rel["bbox_padding"]

    # 1. Hough + wire erasure
    binary_no_wires, _, wire_mask = _remove_wires(
        binary, min_line_length, max_line_gap, line_angle_tolerance_deg
    )
    if debug:
        debug_steps.append(("1 - wire mask", wire_mask.copy()))
        debug_steps.append(("1 - binary no wires", binary_no_wires.copy()))

    # 2. Optional: merge symbol fragments 
    # wire erasure sometimes cuts through a symbol at its connection point
    # (e.g a resistor zigzag gets severed into separate peaks)
    # with this we reconnect pieces that are very close together
    binary_for_cc = binary_no_wires 
    if merge_fragments:
        binary_for_cc = _fragment_merge(binary_no_wires, merge_gap_px)
        if debug:
            debug_steps.append(("2.1 - after fragment merge", binary_for_cc.copy()))

    # Filter used later to reject blobs that aren't near any erased wires; likely text or noise rather than symbols
    wire_near = None
    if require_near_wires:
        wire_near = _near_wire_filter(binary, wire_mask, wire_proximity_px)
        if debug:
            debug_steps.append(("4 - wire proximity mask for filter", wire_near.copy()))

    # 3. Connected Component Analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_for_cc, connectivity=8
    )

    candidates = []

    for i in range(1, num_labels):  # 0 is the background
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # 4. Filter blobs by size and shape

        # Too small => noise
        if area < min_blob_area:
            continue

        # Too large => text labels, borders, or large annotations
        if area > max_blob_area:
            continue

        # Extreme aspect ratio => wire that Hough missed
        if w == 0 or h == 0:
            continue
        ratio = max(w / h, h / w)
        if ratio > max_aspect_ratio:
            continue

        # Discard blobs that aren't near any erased wires; likely text or noise rather than symbols
        if require_near_wires and wire_near is not None:
            x1n = max(0, x - 1)
            y1n = max(0, y - 1)
            x2n = min(binary.shape[1], x + w + 1)
            y2n = min(binary.shape[0], y + h + 1)
            if np.count_nonzero(wire_near[y1n:y2n, x1n:x2n]) == 0:
                continue

        # 5. Crop candidates with padding proportional to their size
        pad_x = int(w * bbox_padding)
        pad_y = int(h * bbox_padding)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(original_gray.shape[1], x + w + pad_x)
        y2 = min(original_gray.shape[0], y + h + pad_y)

        crop = original_gray[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        candidates.append(Candidate(
            crop=crop,
            bbox=(x1, y1, x2 - x1, y2 - y1),
            blob_area=area,
        ))

    if debug:
        debug_steps.append(("final - candidates", _visualize_candidates(original_gray, candidates)))

        for title, img in debug_steps:
            display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
            cv2.imshow(title, display)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    return candidates, binary_no_wires


def _visualize_candidates(
    original_gray: np.ndarray,
    candidates: list[Candidate],
) -> np.ndarray:
    """
    Draws bounding boxes over all detected candidates on the original image.
    Useful for debugging the detection step before classification.
    """
    vis = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    for i, c in enumerate(candidates):
        x, y, w, h = c.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(vis, str(i), (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    return vis
