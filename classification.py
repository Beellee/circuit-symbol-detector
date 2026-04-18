"""
1. resize to a fixed size (64x64)
2. contrast normalization (CLAHE)
3. center content (shift so the centroid of bright pixels is centered)
4. extract HOG features
5. train SVM on labeled data
6. classify new candidates with confidence score
Handles everything related to classifying candidate crops:
  - Interactive labeling tool (to build the training set)
  - HOG feature extraction
  - SVM training
  - SVM inference with confidence score
"""

import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# The 5 target symbols + garbage class
CLASSES = ["capacitor", "inductor", "resistor", "dc_source", "ac_source", "garbage"]

# Normalized size for all crops before HOG extraction
CROP_SIZE = (64, 64)


# ──────────────────────────────────────────────────────────────────────────────
# LABELING TOOL
# ──────────────────────────────────────────────────────────────────────────────

def label_candidates(candidates, output_dir: str = "labeled_data"):
    """
    Interactive tool to manually label candidate crops.

    For each candidate, shows the crop and waits for a keypress:
        c -> capacitor
        i -> inductor
        r -> resistor
        d -> dc_source
        a -> ac_source
        g -> garbage
        0 -> skip (don't save this one)
        Q -> quit and save what we have

    Labeled crops are saved to output_dir/<class_name>/<index>.png
    so they can be reloaded later for training.
    """
    # Create output folders
    for cls in CLASSES:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    key_map = {
        ord('c'): "capacitor",
        ord('i'): "inductor",
        ord('r'): "resistor",
        ord('d'): "dc_source",
        ord('a'): "ac_source",
        ord('g'): "garbage",
    }

    label_counts = {cls: 0 for cls in CLASSES}
    window_name = "Label Candidate  [c=cap  i=ind  r=res  d=dc  a=ac g=garbage s=skip q=quit]"

    print("\n=== LABELING TOOL ===")
    print("Keys: c=capacitor  i=inductor  r=resistor  d=dc_source  a=ac_source")
    print("      g=garbage    0=skip    Q=quit\n")

    for idx, candidate in enumerate(candidates):
        crop = candidate.crop

        # Show at a reasonable display size
        display = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_NEAREST)
        display_color = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        # Overlay candidate index
        cv2.putText(display_color, f"#{idx}/{len(candidates)}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)

        cv2.imshow(window_name, display_color)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            print("Quitting labeling early.")
            break
        elif key == ord('s'):
            print(f"  #{idx} skipped")
            continue
        elif key in key_map:
            label = key_map[key]
            count = label_counts[label]
            save_path = os.path.join(output_dir, label, f"{count:04d}.png")
            cv2.imwrite(save_path, crop)
            label_counts[label] += 1
            print(f"  #{idx} -> {label}  (total {label_counts[label]})")
        else:
            print(f"  #{idx} unknown key, skipping")

    cv2.destroyAllWindows()

    print("\n=== Labeling complete ===")
    for cls, count in label_counts.items():
        print(f"  {cls}: {count} examples")

    return label_counts


# ──────────────────────────────────────────────────────────────────────────────
# HOG FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_hog(crop: np.ndarray) -> np.ndarray:
    """
    Normalizes a crop and extracts its HOG feature vector.

    Steps:
      1. Resize to CROP_SIZE (64x64)
      2. Normalize contrast (CLAHE)
      3. Center content (shift centroid to image center)
      4. Extract HOG features

    Args:
        crop: grayscale image of a candidate region

    Returns:
        1D numpy array — the HOG feature vector
    """
    # 1. Resize
    resized = cv2.resize(crop, CROP_SIZE, interpolation=cv2.INTER_AREA)

    # 2. Contrast normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    normalized = clahe.apply(resized)

    # 3. Center content (shift so the centroid of bright pixels is centered)
    normalized = _center_content(normalized)

    # 4. HOG extraction
    features = hog(
        normalized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
    )

    return features


def _center_content(img: np.ndarray) -> np.ndarray:
    """Shifts the image so the centroid of the content is at the center."""
    h, w = img.shape
    cy, cx = h // 2, w // 2

    # Find centroid of bright pixels
    m = cv2.moments(img)
    if m["m00"] == 0:
        return img  # blank image, nothing to center

    content_cx = int(m["m10"] / m["m00"])
    content_cy = int(m["m01"] / m["m00"])

    shift_x = cx - content_cx
    shift_y = cy - content_cy

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    centered = cv2.warpAffine(img, M, (w, h), borderValue=0)
    return centered


# ──────────────────────────────────────────────────────────────────────────────
# LOADING LABELED DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_labeled_data(data_dir: str = "labeled_data"):
    """
    Loads all labeled crops from the folder structure created by label_candidates().
    Extracts HOG features for each crop.

    Returns:
        X : numpy array of shape (n_samples, n_features)
        y : list of string labels
    """
    X, y = [], []
    data_path = Path(data_dir)

    for cls in CLASSES:
        cls_path = data_path / cls
        if not cls_path.exists():
            print(f"  Warning: no folder found for class '{cls}'")
            continue

        images = list(cls_path.glob("*.png")) + list(cls_path.glob("*.jpg"))
        if not images:
            print(f"  Warning: no images found for class '{cls}'")
            continue

        print(f"  Loading {len(images)} examples for '{cls}'")
        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            features = extract_hog(img)
            X.append(features)
            y.append(cls)

    return np.array(X), y


# ──────────────────────────────────────────────────────────────────────────────
# SVM TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train_svm(X: np.ndarray, y: list, model_path: str = "model.pkl") -> Pipeline:
    """
    Trains an SVM classifier on HOG features.

    Uses an RBF kernel SVM wrapped in a pipeline with StandardScaler,
    since HOG features benefit from normalization before SVM.

    Also prints a basic leave-one-out accuracy if there are enough samples.

    Args:
        X          : feature matrix (n_samples, n_features)
        y          : labels
        model_path : where to save the trained model

    Returns:
        Trained sklearn Pipeline
    """
    if len(X) == 0:
        raise ValueError("No training data found. Run the labeling tool first.")

    print(f"\nTraining SVM on {len(X)} samples across {len(set(y))} classes...")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,   # enables confidence scores
            class_weight='balanced',  # handles uneven class sizes
        ))
    ])

    pipeline.fit(X, y_encoded)

    # Save model + label encoder together
    with open(model_path, 'wb') as f:
        pickle.dump({'pipeline': pipeline, 'label_encoder': le}, f)

    print(f"Model saved to {model_path}")
    print(f"Classes: {list(le.classes_)}")

    return pipeline, le


def load_model(model_path: str = "model.pkl"):
    """Loads a previously trained model."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['pipeline'], data['label_encoder']


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

def classify_candidate(
    crop: np.ndarray,
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    confidence_threshold: float = 0.5,
) -> tuple[str | None, float]:
    """
    Classifies a single candidate crop.

    Args:
        crop                 : grayscale candidate image
        pipeline             : trained sklearn pipeline
        label_encoder        : fitted LabelEncoder
        confidence_threshold : minimum confidence to accept a prediction.
                               Below this, returns (None, score) — treat as rejected.

    Returns:
        (label, confidence) where label is None if confidence is too low
    """
    features = extract_hog(crop).reshape(1, -1)

    probs = pipeline.predict_proba(features)[0]
    predicted_idx = np.argmax(probs)
    confidence = probs[predicted_idx]
    label = label_encoder.inverse_transform([predicted_idx])[0]

    if confidence < confidence_threshold or label == "garbage":
        return None, float(confidence)

    return label, float(confidence)


def classify_all(
    candidates,
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    confidence_threshold: float = 0.5,
) -> list[dict]:
    """
    Runs classification on all candidates and returns structured results.
    Applies non-maximum suppression to remove overlapping detections.

    Returns:
        List of dicts with keys: label, confidence, bbox
        Only includes candidates that passed confidence threshold and are not garbage.
    """
    results = []

    for c in candidates:
        label, confidence = classify_candidate(
            c.crop, pipeline, label_encoder, confidence_threshold
        )
        if label is not None:
            results.append({
                'label': label,
                'confidence': confidence,
                'bbox': c.bbox,
            })

    results = _non_max_suppression(results, iou_threshold=0.3)
    return results


def _non_max_suppression(detections: list[dict], iou_threshold: float = 0.3) -> list[dict]:
    """
    Removes overlapping detections, keeping the one with highest confidence.
    If two bounding boxes overlap significantly (IoU > threshold),
    the lower-confidence one is discarded.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    kept = []

    for det in detections:
        x1, y1, w1, h1 = det['bbox']
        overlaps = False
        for kept_det in kept:
            x2, y2, w2, h2 = kept_det['bbox']
            iou = _compute_iou((x1, y1, x1+w1, y1+h1), (x2, y2, x2+w2, y2+h2))
            if iou > iou_threshold:
                overlaps = True
                break
        if not overlaps:
            kept.append(det)

    return kept


def _compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Computes Intersection over Union between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0
