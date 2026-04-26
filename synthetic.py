"""
Generates synthetic training examples for each circuit symbol class
by drawing them programmatically with controlled variation.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import math

CLASSES = ["capacitor", "inductor", "resistor", "dc_source", "ac_source"]
IMG_SIZE = 60 


def make_canvas():
    """White canvas with some random padding variation."""
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
    return img


def draw_resistor(img, stroke=2):
    """
    Zigzag between two horizontal endpoints.
    American style (sharp zigzag).
    """
    h, w = img.shape
    cx, cy = w // 2, h // 2
    length = int(w * 0.6)
    n_peaks = np.random.randint(4, 7)
    peak_h = int(h * 0.2)

    x_start = cx - length // 2
    x_end = cx + length // 2
    
    # lead wires
    cv2.line(img, (0, cy), (x_start, cy), 0, stroke)
    cv2.line(img, (x_end, cy), (w, cy), 0, stroke)

    # zigzag
    points = []
    xs = np.linspace(x_start, x_end, n_peaks * 2 + 1, dtype=int)
    for i, x in enumerate(xs):
        if i % 2 == 0:
            points.append((x, cy))
        else:
            sign = 1 if (i // 2) % 2 == 0 else -1
            points.append((x, cy + sign * peak_h))

    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], 0, stroke)

    return img


def draw_capacitor(img, stroke=2):
    """Two parallel vertical lines with lead wires."""
    h, w = img.shape
    cx, cy = w // 2, h // 2
    plate_h = int(h * 0.4)
    gap = np.random.randint(6, 14)

    # lead wires
    cv2.line(img, (0, cy), (cx - gap // 2, cy), 0, stroke)
    cv2.line(img, (cx + gap // 2, cy), (w, cy), 0, stroke)

    # plates
    cv2.line(img, (cx - gap // 2, cy - plate_h // 2),
             (cx - gap // 2, cy + plate_h // 2), 0, stroke + 1)
    cv2.line(img, (cx + gap // 2, cy - plate_h // 2),
             (cx + gap // 2, cy + plate_h // 2), 0, stroke + 1)

    return img


def draw_inductor(img, stroke=2):
    """
    Vertical inductor with stacked ellipse coils and lead wires top and bottom.
    Matches the typical style in the training data.
    """
    h, w = img.shape
    cx, cy = w // 2, h // 2
    n_coils = np.random.randint(3, 6)
    coil_rx = int(w * 0.25)  # horizontal radius of ellipse
    coil_ry = int(h * 0.08)  # vertical radius of ellipse
    total_h = n_coils * coil_ry * 2
    y_start = cy - total_h // 2
    y_end = cy + total_h // 2

    # lead wires top and bottom
    cv2.line(img, (cx, 0), (cx, y_start), 0, stroke)
    cv2.line(img, (cx, y_end), (cx, h), 0, stroke)

    # stacked ellipses
    for i in range(n_coils):
        center_y = y_start + coil_ry + i * coil_ry * 2
        cv2.ellipse(img, (cx, center_y), (coil_rx, coil_ry), 0, 0, 360, 0, stroke)

    return img


def draw_dc_source(img, stroke=2):
    """
    Battery symbol: long line (positive) and short line (negative)
    alternating, with lead wires.
    """
    h, w = img.shape
    cx, cy = w // 2, h // 2
    n_pairs = np.random.randint(1, 3)
    long_h = int(h * 0.35)
    short_h = int(h * 0.2)
    gap = np.random.randint(5, 10)
    pair_w = gap * 2
    total_w = n_pairs * pair_w
    x_start = cx - total_w // 2
    x_end = cx + total_w // 2

    # lead wires
    cv2.line(img, (0, cy), (x_start, cy), 0, stroke)
    cv2.line(img, (x_end, cy), (w, cy), 0, stroke)

    for i in range(n_pairs):
        x1 = x_start + i * pair_w
        x2 = x1 + gap
        # long plate
        cv2.line(img, (x1, cy - long_h // 2), (x1, cy + long_h // 2), 0, stroke + 1)
        # short plate
        cv2.line(img, (x2, cy - short_h // 2), (x2, cy + short_h // 2), 0, stroke + 1)

    return img


def draw_ac_source(img, stroke=2):
    """Circle with sine wave inside."""
    h, w = img.shape
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * 0.3)

    # lead wires
    cv2.line(img, (0, cy), (cx - r, cy), 0, stroke)
    cv2.line(img, (cx + r, cy), (w, cy), 0, stroke)

    # circle
    cv2.circle(img, (cx, cy), r, 0, stroke)

    # sine wave inside
    sine_pts = []
    for x in range(cx - r + 4, cx + r - 4):
        t = (x - (cx - r + 4)) / (2 * r - 8) * 2 * math.pi
        y = int(cy - math.sin(t) * r * 0.4)
        sine_pts.append((x, y))
    for i in range(len(sine_pts) - 1):
        cv2.line(img, sine_pts[i], sine_pts[i+1], 0, stroke)

    return img

def add_background_noise(img, noise_type="random"):
    """
    Adds realistic background noise to simulate real crops from circuit images.
    Real crops often contain surrounding text, wire stubs, and annotation fragments.
    """
    img = img.copy()
    h, w = img.shape

    # 1. Random wire stubs entering from edges (simulates lead wire remnants)
    if np.random.random() < 0.7:
        for _ in range(np.random.randint(1, 4)):
            edge = np.random.choice(['top', 'bottom', 'left', 'right'])
            stroke = np.random.randint(1, 3)
            length = np.random.randint(5, 20)
            if edge == 'top':
                x = np.random.randint(w // 4, 3 * w // 4)
                cv2.line(img, (x, 0), (x, length), 0, stroke)
            elif edge == 'bottom':
                x = np.random.randint(w // 4, 3 * w // 4)
                cv2.line(img, (x, h), (x, h - length), 0, stroke)
            elif edge == 'left':
                y = np.random.randint(h // 4, 3 * h // 4)
                cv2.line(img, (0, y), (length, y), 0, stroke)
            elif edge == 'right':
                y = np.random.randint(h // 4, 3 * h // 4)
                cv2.line(img, (w, y), (w - length, y), 0, stroke)

    # 2. Random small text-like blobs in corners (simulates annotation fragments)
    if np.random.random() < 0.5:
        for _ in range(np.random.randint(1, 3)):
            # place in a corner or edge region, not center
            region_x = np.random.choice([
                np.random.randint(0, w // 4),
                np.random.randint(3 * w // 4, w - 5)
            ])
            region_y = np.random.choice([
                np.random.randint(0, h // 4),
                np.random.randint(3 * h // 4, h - 5)
            ])
            blob_w = np.random.randint(5, 15)
            blob_h = np.random.randint(3, 8)
            # draw as a few short horizontal strokes (mimics letter fragments)
            for line_y in range(region_y, min(region_y + blob_h, h - 1), 3):
                x1 = region_x
                x2 = min(region_x + blob_w, w - 1)
                cv2.line(img, (x1, line_y), (x2, line_y), 0, 1)

    # 3. Junction dots (small filled circles at connection points)
    if np.random.random() < 0.4:
        for _ in range(np.random.randint(1, 3)):
            jx = np.random.randint(5, w - 5)
            jy = np.random.randint(5, h - 5)
            cv2.circle(img, (jx, jy), np.random.randint(2, 4), 0, -1)

    # 4. Large annotation below/above (simulates value labels like "1 kΩ", "R1")
    if np.random.random() < 0.6:
        # pick top or bottom half
        region_y = np.random.choice([
            np.random.randint(0, h // 3),           # above symbol
            np.random.randint(2 * h // 3, h - 10)  # below symbol
        ])
        label_w = np.random.randint(15, 35)
        label_h = np.random.randint(8, 16)
        region_x = np.random.randint(w // 4, w // 2)
        # draw as several horizontal strokes of varying width
        for line_y in range(region_y, min(region_y + label_h, h - 1), 4):
            lw = np.random.randint(label_w // 2, label_w)
            cv2.line(img, (region_x, line_y), (region_x + lw, line_y), 0, np.random.randint(1, 3))

    return img


DRAW_FUNCS = {
    "resistor": draw_resistor,
    "capacitor": draw_capacitor,
    "inductor": draw_inductor,
    "dc_source": draw_dc_source,
    "ac_source": draw_ac_source,
}


def generate_synthetic(n_per_class=50, output_dir="labeled_data"):
    for cls in CLASSES:
        if cls not in DRAW_FUNCS:
            continue

        cls_path = Path(output_dir) / cls
        cls_path.mkdir(parents=True, exist_ok=True)

        existing = list(cls_path.glob("syn_*.png"))
        counter = len(existing)

        draw_fn = DRAW_FUNCS[cls]
        generated = 0

        while generated < n_per_class:
            img = make_canvas()
            stroke = np.random.randint(1, 4)
            img = draw_fn(img, stroke=stroke)
            img = add_background_noise(img)  # ← add this

            save_path = cls_path / f"syn_{counter:04d}.png"
            cv2.imwrite(str(save_path), img)
            counter += 1
            generated += 1

        print(f"  {cls}: generated {generated} synthetic examples")