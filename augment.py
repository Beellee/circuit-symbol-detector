import cv2
import numpy as np
from pathlib import Path

CLASSES = ["capacitor", "inductor", "resistor", "dc_source", "ac_source", "garbage"]


def augment_image(img: np.ndarray) -> list[np.ndarray]:
    variants = []

    def rotate(im, angle):
        h, w = im.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(im, M, (w, h), borderValue=0)

    def add_noise(im, sigma):
        noise = np.random.normal(0, sigma, im.shape).astype(np.int16)
        return np.clip(im.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def dilate(im, k, iterations=2):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(im, kernel, iterations=iterations)

    def erode(im, k, iterations=2):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.erode(im, kernel, iterations=iterations)

    def shear(im, s):
        h, w = im.shape
        M = np.float32([[1, s, 0], [0, 1, 0]])
        return cv2.warpAffine(im, M, (w, h), borderValue=0)

    def elastic(im, alpha=20, sigma=3):
        h, w = im.shape
        rs = np.random.RandomState()
        dx = cv2.GaussianBlur((rs.rand(h, w)*2-1).astype(np.float32), (0,0), sigma) * alpha
        dy = cv2.GaussianBlur((rs.rand(h, w)*2-1).astype(np.float32), (0,0), sigma) * alpha
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = np.clip(x + dx, 0, w-1).astype(np.float32)
        map_y = np.clip(y + dy, 0, h-1).astype(np.float32)
        return cv2.remap(im, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)

    def brightness(im, alpha):
        return np.clip(im.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    def perspective(im):
        h, w = im.shape
        margin = 15
        src = np.float32([[0,0],[w,0],[w,h],[0,h]])
        dst = np.float32([
            [np.random.randint(0, margin), np.random.randint(0, margin)],
            [w - np.random.randint(0, margin), np.random.randint(0, margin)],
            [w - np.random.randint(0, margin), h - np.random.randint(0, margin)],
            [np.random.randint(0, margin), h - np.random.randint(0, margin)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(im, M, (w, h), borderValue=0)

    def blur(im, k):
        return cv2.GaussianBlur(im, (k, k), 0)

    def cutout(im):
        im = im.copy()
        h, w = im.shape
        cut_h = np.random.randint(h//6, h//3)
        cut_w = np.random.randint(w//6, w//3)
        y = np.random.randint(0, h - cut_h)
        x = np.random.randint(0, w - cut_w)
        im[y:y+cut_h, x:x+cut_w] = 0
        return im

    # --- single transforms ---
    """for angle in [-20, -15, -10, -5, 5, 10, 15, 20, 90, 180, 270]:
        variants.append(rotate(img, angle))
    variants.append(cv2.flip(img, 1))
    variants.append(cv2.flip(img, 0))
    for s in [-0.3, -0.2, 0.2, 0.3]:
        variants.append(shear(img, s))
    for k in [3, 5, 7]:
        variants.append(dilate(img, k))
        variants.append(erode(img, k))
    for sigma in [5, 10, 15, 20]:
        variants.append(add_noise(img, sigma))
    for alpha in [0.6, 0.75, 0.85, 1.15, 1.25, 1.4]:
        variants.append(brightness(img, alpha))
    for _ in range(5):
        variants.append(elastic(img))
        variants.append(elastic(img, alpha=30, sigma=2))
    for _ in range(5):
        variants.append(perspective(img))
    for k in [3, 5]:
        variants.append(blur(img, k))
    for _ in range(5):
        variants.append(cutout(img))
    """

    # --- double combinations ---
    for angle in [-20, -15, -10, -5, 5, 10, 15, 20]:
        variants.append(add_noise(rotate(img, angle), 15))
        variants.append(dilate(rotate(img, angle), 3))
        variants.append(erode(rotate(img, angle), 3))
        variants.append(elastic(rotate(img, angle)))

    for s in [-0.3, -0.2, 0.2, 0.3]:
        variants.append(erode(shear(img, s), 3))
        variants.append(dilate(shear(img, s), 3))
        variants.append(elastic(shear(img, s)))
        variants.append(add_noise(shear(img, s), 15))

    for _ in range(8):
        variants.append(add_noise(elastic(img), 15))
        variants.append(brightness(elastic(img), np.random.choice([0.6, 0.75, 1.25, 1.4])))
        variants.append(perspective(dilate(img, 3)))
        variants.append(add_noise(perspective(img), 15))
        variants.append(cutout(elastic(img)))
        variants.append(blur(elastic(img), 3))

    # --- triple combinations ---
    for _ in range(10):
        v = elastic(img)
        v = rotate(v, np.random.choice([-20,-15,-10,-5,5,10,15,20,90,180,270]))
        v = add_noise(v, np.random.randint(8, 25))
        variants.append(v)

    for _ in range(10):
        v = perspective(img)
        v = dilate(v, np.random.choice([3, 5, 7]))
        v = brightness(v, np.random.choice([0.6, 0.75, 1.25, 1.4]))
        variants.append(v)

    for _ in range(10):
        v = shear(img, np.random.choice([-0.3, -0.2, 0.2, 0.3]))
        v = elastic(v, alpha=25, sigma=3)
        v = erode(v, 3)
        variants.append(v)

    for _ in range(10):
        v = rotate(img, np.random.choice([-20,-15,-10,-5,5,10,15,20]))
        v = perspective(v)
        v = add_noise(v, np.random.randint(8, 20))
        variants.append(v)

    for _ in range(10):
        v = elastic(img, alpha=25, sigma=2)
        v = cutout(v)
        v = add_noise(v, np.random.randint(5, 15))
        variants.append(v)

    np.random.shuffle(variants)
    return variants


def augment_dataset(data_dir: str = "labeled_data", target_per_class: int = 150):
    data_path = Path(data_dir)

    for cls in CLASSES:
        cls_path = data_path / cls
        if not cls_path.exists():
            print(f"  Skipping '{cls}' — folder not found")
            continue

        images = list(cls_path.glob("*.png")) + list(cls_path.glob("*.jpg"))
        if not images:
            print(f"  Skipping '{cls}' — no images found")
            continue

        # skip if already enough examples 
        if len(images) >= target_per_class:
            print(f"  {cls}: {len(images)} examples already, skipping augmentation")
            continue

        needed = target_per_class - len(images)

        # Find the highest existing index to avoid overwriting
        existing = [int(p.stem) for p in images if p.stem.isdigit()]
        counter = max(existing) + 1 if existing else len(images)

        # Only augment originals, not previously augmented files
        originals = [p for p in images if "aug" not in p.stem]

        # distribute budget evenly across originals
        per_original = max(1, needed // len(originals))

        generated = 0
        for img_path in originals:
            if generated >= needed:
                break

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            count_this_original = 0
            for variant in augment_image(img):
                if generated >= needed:
                    break
                if count_this_original >= per_original:
                    break
                save_path = cls_path / f"aug_{counter:04d}.png"
                cv2.imwrite(str(save_path), variant)
                counter += 1
                generated += 1
                count_this_original += 1

        print(f"  {cls}: {len(images)} original → +{generated} augmented = {len(images) + generated} total")
