"""Synthetic-data mode: fake dataset folders with the real datasets' layouts.

SOFTWARE TESTING ONLY. Images are rendered fundus-like pictures with random
vessel trees; labels are random. Deliberate problems are planted so the
integrity, duplicate and split checks can be verified:
- an IDRiD segmentation test image that is a re-compressed copy of an IDRiD
  grading training image (cross-part duplicate -> must move to test);
- an APTOS duplicate pair with conflicting grades (-> excluded);
- a truncated (corrupt) APTOS PNG, a label row without an image, an image without a label;
- DRIVE and RITE sharing the same photographs (-> same CV fold).
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..vessel_graph.synthetic_masks import fundus_like


def fake_fundus_av(size: int = 256, seed: int = 0, return_disc: bool = False):
    """(RGB image, vessel mask, FOV mask, artery mask, vein mask[, disc centre (row, col) px]). Left/right eye by seed.

    As in real photographs, veins are drawn darker and wider than arteries, so a
    model can learn to tell them apart. Where the trees cross, both masks are set.
    """
    rng = np.random.default_rng(seed)
    disc_r, disc_c = rng.uniform(0.4, 0.6), rng.uniform(0.15, 0.3)
    v = fundus_like(shape=(size, size), seed=seed, generations=4, root_radius=size / 90,
                    segment_length=size / 7, speckle=0, disc_rc=(disc_r * size, disc_c * size),
                    trunk_angles_deg=(rng.uniform(-45, -10), rng.uniform(15, 50)))
    art, vein = v.tree_soft(0, supersample=2), v.tree_soft(1, supersample=2)
    if seed % 2:
        art, vein = art[:, ::-1], vein[:, ::-1]
        disc_c = 1 - disc_c
    yy, xx = np.mgrid[:size, :size] / size
    fov = (yy - 0.5) ** 2 + (xx - 0.5) ** 2 <= 0.47**2
    disc = np.exp(-(((yy - disc_r) ** 2 + (xx - disc_c) ** 2) / (2 * 0.04**2)))
    tilt = 0.75 + 0.5 * (rng.uniform(-0.3, 0.3) * (xx - 0.5) + rng.uniform(-0.3, 0.3) * (yy - 0.5) + 0.5)
    texture = ndimage.gaussian_filter(rng.normal(0, 1, (size, size)), 3)
    base = np.stack([170 + 25 * texture, 75 + 10 * texture, 35 + 5 * texture], axis=-1) * tilt[..., None]
    base += disc[..., None] * np.array([70, 110, 60])
    darkness = np.maximum(0.35 * art, 0.55 * vein)
    base *= (1 - darkness[..., None])
    img = np.where(fov[..., None], base, 0)
    vessels = np.maximum(art, vein) > 0.5
    out = (np.clip(img, 0, 255).astype(np.uint8), vessels & fov, fov, (art > 0.5) & fov, (vein > 0.5) & fov)
    return out + ((disc_r * size, disc_c * size),) if return_disc else out


def fake_fundus(size: int = 256, seed: int = 0) -> tuple[NDArray[np.uint8], NDArray[np.bool_], NDArray[np.bool_]]:
    """(RGB image, vessel mask, field-of-view mask)."""
    img, vessels, fov, _, _ = fake_fundus_av(size, seed)
    return img, vessels, fov


def _save(arr, path: Path, fmt: str | None = None, quality: int = 92) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(arr)
    if fmt == "JPEG" or path.suffix.lower() in (".jpg", ".jpeg"):
        im.convert("RGB").save(path, "JPEG", quality=quality)
    else:
        im.save(path)


def _variant(img: NDArray[np.uint8], scale: float = 0.7, gain: float = 1.08) -> NDArray[np.uint8]:
    """Resized, brightened, JPEG re-compressed copy of an image (a 'duplicate')."""
    from PIL import Image

    h, w = img.shape[:2]
    small = Image.fromarray(np.clip(img * gain, 0, 255).astype(np.uint8)).resize(
        (int(w * scale), int(h * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=75)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def _blob_mask(shape, n, rmax, rng):
    m = np.zeros(shape, bool)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    for _ in range(n):
        cy, cx, r = rng.uniform(0.25, 0.75) * shape[0], rng.uniform(0.25, 0.75) * shape[1], rng.uniform(1, rmax)
        m |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
    return m


def make_fake_datasets(root: str | Path, seed: int = 0, size: int = 256) -> dict:
    """Write fake DRIVE, RITE, IDRiD and APTOS2019 folders under `root`. Returns what was planted."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    s = 1000 * seed

    # --- IDRiD grading: 10 images (8 train, 2 test) + labels CSVs + OD centres.
    # As in the real set, test file names restart at IDRiD_001 (they reuse training names).
    g = root / "IDRiD" / "B. Disease Grading"
    grading = {}
    rows = {"train": [], "test": []}
    for k in range(1, 11):
        split = "train" if k <= 8 else "test"
        img, _, _ = fake_fundus(size, s + k)
        name = f"IDRiD_{k if k <= 8 else k - 8:03d}"
        grading[(split, name)] = img
        sub = "a. Training Set" if split == "train" else "b. Testing Set"
        _save(img, g / "1. Original Images" / sub / f"{name}.jpg")
        rows[split].append([name, int(rng.integers(0, 5)), int(rng.integers(0, 3))])
    for split, letter in (("train", "a"), ("test", "b")):
        p = g / "2. Groundtruths" / f"{letter}. IDRiD_Disease Grading_{'Training' if split == 'train' else 'Testing'} Labels.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Image name", "Retinopathy grade", "Risk of macular edema ", "", ""])
            w.writerows([r + ["", ""] for r in rows[split]])
    loc = root / "IDRiD" / "C. Localization" / "2. Groundtruths" / "1. Optic Disc Center Location"
    loc.mkdir(parents=True, exist_ok=True)
    with open(loc / "a. IDRiD_OD_Center_Training Set_Markups.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image No", "X- Coordinate", "Y - Coordinate"])
        for k in range(1, 9):
            w.writerow([f"IDRiD_{k:03d}", size * (0.8 if (s + k) % 2 else 0.2), size * 0.5])

    # --- IDRiD segmentation: 4 images (3 train, 1 test); IDRiD_04 duplicates grading IDRiD_002
    seg = root / "IDRiD" / "A. Segmentation"
    folders = {"MA": "1. Microaneurysms", "HE": "2. Haemorrhages", "EX": "3. Hard Exudates",
               "SE": "4. Soft Exudates", "OD": "5. Optic Disc"}
    for k in range(1, 5):
        split = "a. Training Set" if k <= 3 else "b. Testing Set"
        img = _variant(grading[("train", "IDRiD_002")]) if k == 4 else fake_fundus(size, s + 100 + k)[0]
        _save(img, seg / "1. Original Images" / split / f"IDRiD_{k:02d}.jpg")
        h, w = img.shape[:2]
        for lesion, folder in folders.items():
            if lesion == "SE" and k % 2 == 0:
                continue  # soft exudates absent for some images, as in IDRiD
            if lesion == "OD":
                yy, xx = np.mgrid[:h, :w]
                mask = (yy - h / 2) ** 2 + (xx - w * 0.2) ** 2 <= (0.08 * w) ** 2
            else:
                mask = _blob_mask((h, w), 3, 4, rng)
            _save((mask * 255).astype(np.uint8),
                  seg / "2. All Segmentation Groundtruths" / split / folder / f"IDRiD_{k:02d}_{lesion}.tif")

    # --- APTOS: 12 labelled images + planted problems
    a = root / "APTOS2019"
    ids, labels = [], []
    for k in range(12):
        iid = f"{(seed * 97 + k) * 7919:010x}"
        img, _, _ = fake_fundus(size, s + 200 + k)
        _save(img, a / "train_images" / f"{iid}.png")
        ids.append(iid)
        labels.append(int(rng.integers(0, 5)))
    dup_src, dup_id = ids[0], "dupe00000000"
    _save(_variant(fake_fundus(size, s + 200)[0], scale=0.8), a / "train_images" / f"{dup_id}.png")
    corrupt_id = "corrupt00000"
    good = io.BytesIO()
    from PIL import Image
    Image.fromarray(fake_fundus(size, s + 999)[0]).save(good, "PNG")
    (a / "train_images" / f"{corrupt_id}.png").write_bytes(good.getvalue()[: len(good.getvalue()) // 3])
    _save(fake_fundus(size, s + 998)[0], a / "train_images" / "nolabel00000.png")
    with open(a / "train.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id_code", "diagnosis"])
        w.writerows(zip(ids, labels))
        w.writerow([dup_id, (labels[0] + 2) % 5])  # conflicting grade for the duplicate
        w.writerow([corrupt_id, 1])
        w.writerow(["missing00000", 0])

    # --- DRIVE (3 annotated training images) and RITE (same photos + one test image)
    for k in (21, 22, 23):
        img, vessels, fov = fake_fundus(size, s + 300 + k)
        _save(img, root / "DRIVE" / "training" / "images" / f"{k}_training.tif")
        _save((vessels * 255).astype(np.uint8), root / "DRIVE" / "training" / "1st_manual" / f"{k}_manual1.gif")
        _save((fov * 255).astype(np.uint8), root / "DRIVE" / "training" / "mask" / f"{k}_training_mask.gif")
    for k, split in ((21, "training"), (22, "training"), (23, "training"), (1, "test")):
        img, vessels, _, artery, vein = fake_fundus_av(size, s + 300 + k)
        av = np.zeros((*vessels.shape, 3), np.uint8)
        av[artery & ~vein] = (255, 0, 0)
        av[vein & ~artery] = (0, 0, 255)
        av[artery & vein] = (0, 255, 0)
        tag = "training" if split == "training" else "test"
        _save(img, root / "RITE" / split / "images" / f"{k:02d}_{tag}.tif")
        _save(av, root / "RITE" / split / "av" / f"{k:02d}_{tag}.png")
        _save((vessels * 255).astype(np.uint8), root / "RITE" / split / "vessel" / f"{k:02d}_{tag}.png")

    return {"cross_part_duplicate": ("IDRiD_04", "IDRiD_002-train"), "aptos_conflict": (dup_src, dup_id),
            "aptos_corrupt": corrupt_id, "aptos_missing_image": "missing00000",
            "aptos_unlabeled": "nolabel00000", "synthetic": True}


def make_fake_drive_rite(root: str | Path, n_train: int = 20, n_test: int = 4, size: int = 160,
                         seed: int = 0) -> dict:
    """FAKE DRIVE (training photos 21..20+n_train) + RITE (A/V for those and test photos 1..n_test).

    Same folder layout and file naming as the real downloads. Software testing only.
    """
    root = Path(root)
    for k in range(21, 21 + n_train):
        img, vessels, fov, artery, vein = fake_fundus_av(size, 5000 + 97 * seed + k)
        _save(img, root / "DRIVE" / "training" / "images" / f"{k}_training.tif")
        _save((vessels * 255).astype(np.uint8), root / "DRIVE" / "training" / "1st_manual" / f"{k}_manual1.gif")
        _save((fov * 255).astype(np.uint8), root / "DRIVE" / "training" / "mask" / f"{k}_training_mask.gif")
        _write_rite(root, k, "training", img, vessels, artery, vein)
    for k in range(1, 1 + n_test):
        img, vessels, fov, artery, vein = fake_fundus_av(size, 5000 + 97 * seed + k)
        _write_rite(root, k, "test", img, vessels, artery, vein)
    return {"train": list(range(21, 21 + n_train)), "test": list(range(1, 1 + n_test)), "synthetic": True}


def _write_rite(root: Path, k: int, split: str, img, vessels, artery, vein) -> None:
    av = np.zeros((*vessels.shape, 3), np.uint8)
    av[artery & ~vein] = (255, 0, 0)
    av[vein & ~artery] = (0, 0, 255)
    av[artery & vein] = (0, 255, 0)
    av[vessels & ~artery & ~vein] = (255, 255, 255)  # uncertain
    _save(img, root / "RITE" / split / "images" / f"{k:02d}_{split}.tif")
    _save(av, root / "RITE" / split / "av" / f"{k:02d}_{split}.png")
    _save((vessels * 255).astype(np.uint8), root / "RITE" / split / "vessel" / f"{k:02d}_{split}.png")
