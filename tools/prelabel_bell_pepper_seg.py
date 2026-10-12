"""Prelabel bell pepper lesions on PlantDoc images with a small YOLO-seg model trained on the normalized segmentation labels.

Run from the repo root after normalization/normalized_bell_pepper.ipynb:  python tools/prelabel_bell_pepper_seg.py
Outputs go to prelabel/bell_pepper/ (never into normalized/: these labels are unverified until reviewed).
"""
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path("normalized/bell_pepper")
WORK = Path("prelabel/bell_pepper")
DATA = WORK / "data"
OUT = WORK / "plantdoc_prelabels"
LOW_CONF, TINY, HUGE = 0.5, 0.0005, 0.5  # review thresholds: confidence, polygon area as share of the image

manifest = pd.read_csv(ROOT / "manifest.csv")
names = (ROOT / "labels" / "segmentation" / "condition" / "classes.txt").read_text().split()
seg_labels = {p.stem: p for p in (ROOT / "labels" / "segmentation" / "condition").glob("*.txt") if p.name != "classes.txt"}

# ---------------------------------------------------------------- 1. YOLO dataset, split by leakage group (never by image)
shutil.rmtree(DATA, ignore_errors=True)
train_rows = manifest[manifest["image"].map(lambda n: Path(n).stem in seg_labels)]
split_of = lambda group: "val" if int(hashlib.md5(group.encode()).hexdigest(), 16) % 5 == 0 else "train"
for row in train_rows.itertuples():
    split = split_of(row.group)
    for sub in ("images", "labels"):
        (DATA / sub / split).mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "images" / row.image, DATA / "images" / split / row.image)
    shutil.copy(seg_labels[Path(row.image).stem], DATA / "labels" / split / f"{Path(row.image).stem}.txt")
(DATA / "data.yaml").write_text(f"path: {DATA.resolve()}\ntrain: images/train\nval: images/val\nnames: {dict(enumerate(names))}\n")
print("training images per split:", train_rows["group"].map(split_of).value_counts().to_dict(), "| classes:", names)

# ---------------------------------------------------------------- 2. fine-tune a small model
model = YOLO(WORK / "yolo11n-seg.pt")  # downloaded there on first run
model.train(data=str(DATA / "data.yaml"), epochs=150, patience=40, imgsz=640, batch=16, seed=0, deterministic=True,
            project=str(WORK.resolve() / "runs"), name="seg", exist_ok=True, plots=True, verbose=False)
best = YOLO(WORK / "runs" / "seg" / "weights" / "best.pt")
metrics = best.val(data=str(DATA / "data.yaml"), split="val", project=str(WORK.resolve() / "runs"), name="val", exist_ok=True, verbose=False)
print("val mask mAP50 per class:", dict(zip([names[i] for i in metrics.seg.ap_class_index], np.round(metrics.seg.ap50, 3))))
print(f"val mask mAP50 all: {metrics.seg.map50:.3f}, mAP50-95: {metrics.seg.map:.3f}")

# ---------------------------------------------------------------- 3. predict on PlantDoc bell pepper (its own leaf classes are not used)
shutil.rmtree(OUT, ignore_errors=True)
(OUT / "labels").mkdir(parents=True)
(OUT / "overlays").mkdir()
train_groups = set(train_rows["group"])
plantdoc = manifest[manifest["source"] == "plantdoc"]
leaked = plantdoc[plantdoc["group"].isin(train_groups)]
targets = plantdoc[~plantdoc["group"].isin(train_groups)]
print(f"PlantDoc bell pepper images: {len(plantdoc)}; skipped as near-duplicates of training images: {len(leaked)} {list(leaked['image'])}")


def polygon_area(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


rows = []
for name in targets["image"]:
    result = best.predict(ROOT / "images" / name, conf=0.25, imgsz=640, retina_masks=True, verbose=False)[0]
    stem, lines = Path(name).stem, []
    polys = result.masks.xyn if result.masks is not None else []
    classes = [names[int(c)] for c in result.boxes.cls] if len(polys) else []
    for xy, cls, conf in zip(polys, classes, result.boxes.conf.tolist() if len(polys) else []):
        if len(xy) < 3:
            continue
        xy = np.clip(xy, 0, 1)
        lines.append(f"{names.index(cls)} " + " ".join(f"{v:.6f}" for v in xy.ravel()))
        area = polygon_area(xy)
        rows.append({"image": name, "class": cls, "conf": round(conf, 3), "area_frac": round(area, 5),
                     "low_conf": conf < LOW_CONF, "tiny": area < TINY, "huge": area > HUGE})
    (OUT / "labels" / f"{stem}.txt").write_text("".join(f"{l}\n" for l in lines))
    result.save(filename=str(OUT / "overlays" / f"{stem}.jpg"))
    if not lines:
        rows.append({"image": name, "class": None, "conf": None, "area_frac": None, "low_conf": False, "tiny": False, "huge": False})

# ---------------------------------------------------------------- 4. review list: suspicious first
review = pd.DataFrame(rows)
per_image = review.groupby("image")["class"].nunique()
review["mixed_classes"] = review["image"].map(per_image) > 1  # one leaf scene rarely shows several of these diseases
review["no_prediction"] = review["class"].isna()
flags = ["no_prediction", "mixed_classes", "low_conf", "tiny", "huge"]
review["suspicious"] = review[flags].any(axis=1)
review = review.sort_values(["suspicious", "image", "conf"], ascending=[False, True, True])
review.to_csv(OUT / "review.csv", index=False)
(OUT / "classes.txt").write_text("\n".join(names) + "\n")

# self-check: every prelabel file parses as YOLO-seg with valid ids and coordinates
for path in (OUT / "labels").glob("*.txt"):
    for line in path.read_text().splitlines():
        values = line.split()
        assert 0 <= int(values[0]) < len(names) and len(values) >= 7 and len(values) % 2 == 1, path
        assert all(0 <= float(v) <= 1 for v in values[1:]), path

print(f"\nprelabeled {len(targets)} images, {review['class'].notna().sum()} polygons; per class:",
      review["class"].value_counts().to_dict())
print("flag counts:", {f: int(review[f].sum()) for f in flags}, "| images needing review:", review.loc[review["suspicious"], "image"].nunique())
print(f"wrote {OUT}/labels/*.txt (YOLO-seg), {OUT}/overlays/*.jpg, {OUT}/review.csv")
