import argparse
import os
import cv2
from pathlib import Path

from preprocessing import preprocess
from detection import detect_candidates, _visualize_candidates
from classification import (
    label_candidates,
    load_labeled_data,
    train_svm,
    load_model,
    classify_all,
    CLASSES,
)
from augment import augment_dataset


# ──────────────────────────────────────────────────────────────────────────────
# MODE: DEBUG_PREPROCESS
# ──────────────────────────────────────────────────────────────────────────────

def run_debug_preprocess(image_path: str):
    """Runs preprocessing on a single image showing each intermediate step."""
    print(f"\nDebug preprocessing: {image_path}")
    preprocess(image_path, debug=True)


# ──────────────────────────────────────────────────────────────────────────────
# MODE: DEBUG_DETECT
# ──────────────────────────────────────────────────────────────────────────────

def run_debug_detect(image_path: str):
    """Runs preprocessing + detection on a single image showing each intermediate step."""
    print(f"\nDebug detection: {image_path}")
    gray, binary = preprocess(image_path)
    detect_candidates(binary, gray, debug=True)


# ──────────────────────────────────────────────────────────────────────────────
# MODE: LABEL
# ──────────────────────────────────────────────────────────────────────────────

def run_labeling(images_dir: str, labeled_data_dir: str = "labeled_data"):
    """
    Runs the detection pipeline on all images in images_dir,
    then opens the interactive labeling tool for each batch of candidates.
    """
    image_paths = sorted(
        list(Path(images_dir).glob("*.png")) +
        list(Path(images_dir).glob("*.jpg")) +
        list(Path(images_dir).glob("*.jpeg"))
    )

    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    print(f"Found {len(image_paths)} images.")
    all_candidates = []

    for path in image_paths:
        print(f"\nProcessing {path.name}...")
        try:
            gray, binary = preprocess(str(path))
            candidates, _ = detect_candidates(binary, gray)
            print(f"  → {len(candidates)} candidates found")
            all_candidates.extend(candidates)
        except Exception as e:
            print(f"  Error processing {path.name}: {e}")

    print(f"\nTotal candidates to label: {len(all_candidates)}")
    print("Tip: label real symbols carefully. For the garbage class, include")
    print("     junctions, text fragments, and missed wire segments.\n")

    label_candidates(all_candidates, output_dir=labeled_data_dir)


# ──────────────────────────────────────────────────────────────────────────────
# MODE: TRAIN
# ──────────────────────────────────────────────────────────────────────────────

def run_training(labeled_data_dir: str = "labeled_data", model_path: str = "model.pkl"):
    """Loads labeled crops, extracts HOG features, and trains the SVM."""
    print("Loading labeled data...")
    X, y = load_labeled_data(labeled_data_dir)

    if len(X) == 0:
        print("No labeled data found. Run --mode label first.")
        return

    print(f"\nDataset summary:")
    for cls in CLASSES:
        count = y.count(cls)
        print(f"  {cls}: {count} examples")

    if len(set(y)) < 2:
        print("\nNeed at least 2 classes to train. Label more examples.")
        return

    train_svm(X, y, model_path=model_path)
    print("\nTraining complete.")

# ──────────────────────────────────────────────────────────────────────────────
# MODE: AUGMENT
# ──────────────────────────────────────────────────────────────────────────────

def run_augmentation(labeled_data_dir: str = "labeled_data"):
    """Generates augmented variants of all labeled crops to expand the training set."""
    print("Augmenting dataset...")
    augment_dataset(labeled_data_dir, target_per_class=60)


# ──────────────────────────────────────────────────────────────────────────────
# MODE: RUN
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(image_path: str, model_path: str = "model.pkl", show_debug: bool = False):
    """Runs the full pipeline on a single image and displays results."""
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}. Run --mode train first.")
        return

    pipeline, label_encoder = load_model(model_path)

    print(f"\nProcessing {image_path}...")
    gray, binary = preprocess(image_path)

    candidates, binary_no_wires = detect_candidates(binary, gray)
    print(f"  {len(candidates)} candidates detected")

    if show_debug:
        cv2.imshow("Binary after wire removal", binary_no_wires)
        cv2.imshow("Candidates (before classification)", _visualize_candidates(gray, candidates))
        print("  Press any key to continue to classification...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    results = classify_all(candidates, pipeline, label_encoder)
    print(f"  {len(results)} symbols detected after classification\n")

    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colors = {
        "capacitor": (255, 100,   0),
        "inductor":  (  0, 200,   0),
        "resistor":  (  0,   0, 255),
        "dc_source": (200,   0, 200),
        "ac_source": (  0, 200, 200),
    }

    for det in results:
        x, y_coord, w, h = det['bbox']
        label = det['label']
        conf = det['confidence']
        color = colors.get(label, (200, 200, 200))
        cv2.rectangle(output, (x, y_coord), (x + w, y_coord + h), color, 2)
        cv2.putText(
            output,
            f"{label} {conf:.2f}",
            (x, y_coord - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
        print(f"  [{label}]  confidence={conf:.2f}  bbox={det['bbox']}")

    cv2.imshow("Detections", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Circuit symbol detector")
    parser.add_argument(
        "--mode",
        choices=["debug_preprocess", "debug_detect", "label", "train", "augment", "run"],
        required=True,
    )
    parser.add_argument("--image",      default=None,         help="Path to image")
    parser.add_argument("--images_dir", default="./circuits", help="Folder of images (--mode label)")
    parser.add_argument("--debug",      action="store_true",  help="Show intermediate steps (--mode run)")
    args = parser.parse_args()

    if args.mode == "debug_preprocess":
        if not args.image:
            print("Please provide --image <path>")
        else:
            run_debug_preprocess(args.image)

    elif args.mode == "debug_detect":
        if not args.image:
            print("Please provide --image <path>")
        else:
            run_debug_detect(args.image)

    elif args.mode == "label":
        run_labeling(args.images_dir)

    elif args.mode == "train":
        run_training()

    elif args.mode == "augment":
        run_augmentation()

    elif args.mode == "run":
        if not args.image:
            print("Please provide --image <path>")
        else:
            run_inference(args.image, show_debug=args.debug)