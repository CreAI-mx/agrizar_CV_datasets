"""Builds normalization/normalized_<crop>.ipynb for the 6 target crops. Run from the repo root, then execute each notebook (see CLAUDE.md)."""
import os

import nbformat as nbf

# ---------------------------------------------------------------- shared text

INTRO = r"""
# Normalized dataset: {title}

Builds `normalized/{crop}/`, one dataset for **{title}** merged from the sources surveyed in `plague.ipynb` and `growth.ipynb`. All labels use one class vocabulary. The goal is **correct labels, not many labels**: when a label cannot be trusted or reconciled, it is dropped, and this notebook says why and counts it.

```
normalized/{crop}/
  images/                                 every image, flat, named <source>_<n>.<ext>
  labels/classification/condition.csv     image,class   plant condition (healthy / disease / pest / deficiency)
  labels/classification/maturity.csv      image,class   fruit maturity, only from sources that label it per image
  labels/classification/classes.json      class lists of both axes
  labels/detection/condition/<stem>.txt  YOLO "cls cx cy w h", normalized to 0-1, clipped to the image (+ classes.txt)
  labels/detection/maturity/<stem>.txt   same format, plant-stage objects only: flower, bud, fruit__<stage> (+ classes.txt)
  labels/segmentation/condition/<stem>.txt  YOLO-seg "cls x1 y1 x2 y2 ...", normalized polygons, one line per polygon (+ classes.txt)
  labels/segmentation/maturity/<stem>.txt   same format, plant-stage objects only (+ classes.txt)
  manifest.csv                            image, source, source_ref, license, width, height, group, dhash, exif_fix, label_space
```

**Two label axes, two models.** Like classification, detection and segmentation are split into `condition` (healthy / disease / pest / deficiency objects) and `maturity` (flower, bud, fruit stages). Each folder has its **own** `classes.txt` and class ids, so each trains its own model. An image can have a file in both.

**Why this format.** Image sizes go from about 60 px to 8,000 px, so boxes and polygons are stored **normalized** (0–1) as centre-x, centre-y, width, height. These values do not depend on image size, and YOLO tools read them as they are. Polygons are lighter than masks and can be rasterized at any resolution when training.

**Rules used for every source**
1. **Image-level classes taken from boxes.** A `condition` comes from boxes only when the source labels conditions on every image and all condition boxes in the image share one class. Images with mixed classes are left out of `condition.csv` (counted below).
2. **Healthy is never inferred from missing boxes.** It comes only from an explicit healthy label.
3. **Coarse source classes are not split.** When a source class is coarser than the vocabulary, it maps to a coarse class. Example: Strawberry_12 `unripe fruit` becomes `fruit__unripe`, which covers green, white and turning.
4. **Augmented copies and byte-identical duplicates are dropped.** Near-duplicates across sources share one `group`. A near-duplicate has the same 64-bit dHash, confirmed by a 256-bit dHash at most 10 bits apart. If their classification labels disagree, those labels are removed.
5. **Boxes** are converted from each source's convention (pixel xywh, YOLO, COCO) and clipped to the image. Boxes under 2 px after clipping are dropped. Boxes that reached more than 2 px outside are counted.
6. **Detection files.** An image has a detection file only if its source boxes objects in it. An **empty** file means a verified negative. Images with no boxes in a detection source get **no** file, because "no box" does not prove "nothing there".
7. **EXIF orientation.** If the annotation size matches the stored pixels, the EXIF tag is dropped and the pixels are kept. Otherwise the photo is rotated upright (the frame YOLO tools and AgML annotations use). Every other image is copied byte for byte.

**Columns for training**
- `group` is the **leakage key**: split train/val/test by `group`, never by image. It covers Roboflow stems, augmented copies, video sequences, pepper ids, capture sessions and near-duplicates.
- `label_space` lists the detection classes the image's source actually annotates. Sources label different things (a fruit-stage set never boxes diseases), so mask the loss of classes outside `label_space` instead of treating them as background.
"""

VOCAB_MD = r"""
## Class vocabulary

Detection and segmentation classes are named `<object>__<state>`:
- `leaf__healthy` = a whole leaf box (PlantDoc).
- `lesion__<disease>` = a lesion box or polygon (PlantSeg).
- `fruit__<stage>` = a fruit box.
- `affected__<condition>` = a box around the affected organ or region (Strawberry_12 draws these around a diseased fruit, a leaf with pests, a deficient leaf).

Plant parts with no state are plain (`flower`, `bud`). Keeping the object in the name stops a leaf-sized box and a lesion-sized box of the same disease from being merged into one class.
"""

HELPERS = r"""
import hashlib
import io
import json
import os
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download, snapshot_download
from PIL import Image, ImageDraw, ImageOps

if Path.cwd().name == "normalization":  # notebooks live in normalization/, paths below are relative to the repo root
    os.chdir("..")

CROP = "__CROP__"
OUT = Path("normalized") / CROP
DATA_DIR = Path("data")
EXAMPLE_DIR = Path("examples") / "normalized"
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)

shutil.rmtree(OUT, ignore_errors=True)  # the notebook always rebuilds its crop from scratch
for sub in ("images", "labels/classification", *(f"labels/{t}/{a}" for t in ("detection", "segmentation") for a in ("condition", "maturity"))):
    (OUT / sub).mkdir(parents=True)

IMAGES = {}                                   # image name -> manifest row
CLASSIFICATION = {"condition": {}, "maturity": {}}
CONFLICTS = {"condition": set(), "maturity": set()}
SHAPES = {(t, a): {} for t in ("detection", "segmentation") for a in ("condition", "maturity")}  # (task, axis) -> image -> lines
MATURITY_STATES = {"green", "white", "turning", "ripe", "overripe", "unripe", "half_ripe", "breaker", "reddish"}


def axis_of(cls):
    # plant-stage objects go to the maturity model; everything else (healthy, lesions, affected organs, diseased fruit) to condition
    return "maturity" if cls in ("flower", "bud") or (cls.startswith("fruit__") and cls.split("__")[1] in MATURITY_STATES) else "condition"


def axis_classes(task, axis):
    return [c for c in CLASSES[task] if axis_of(c) == axis]


def has_label(name):
    return any(name in d for d in (*CLASSIFICATION.values(), *SHAPES.values()))
LABEL_SPACE = {}                              # source -> detection classes the source annotates
DROPPED = Counter()                           # (source, reason) -> count
NOTES = Counter()                             # (source, note) -> count, things kept but worth knowing
SERIAL = Counter()
"""

HELPERS_2 = r"""
def dhash(img):
    # difference hashes: 64-bit (hex, for bucketing) and 256-bit (bits, to confirm a match). Resized or re-encoded copies agree.
    img.draft("L", (128, 128))  # fast JPEG decode at reduced size
    gray = img.convert("L")
    bits = {n: (lambda a: a[:, 1:] > a[:, :-1])(np.asarray(gray.resize((n + 1, n), Image.BILINEAR), dtype=np.int16)).ravel() for n in (8, 16)}
    # "0x" prefix: a bare hex string like "1e00..." makes pandas' CSV parser read a float with a huge exponent (segfault in pandas 2.3)
    return f"0x{int(''.join('1' if b else '0' for b in bits[8]), 2):016x}", bits[16]


def add_image(data, source, ref, license, group, frame=None):
    # data: encoded bytes (copied unchanged when possible) or a PIL image (saved as PNG).
    # frame: (W, H) the source annotated on, if it records one. Returns (name, W, H), or None if the image is dropped.
    img = Image.open(io.BytesIO(data)) if isinstance(data, bytes) else data
    orientation = img.getexif().get(274, 1) if isinstance(data, bytes) else 1
    try:
        digest = dhash(Image.open(io.BytesIO(data)) if isinstance(data, bytes) else img)  # also proves the file decodes
    except OSError:
        DROPPED[(source, "image file cannot be decoded")] += 1
        return None
    SERIAL[source] += 1
    stem = f"{source}_{SERIAL[source]:06d}"
    exif_fix = ""
    if isinstance(data, bytes) and orientation == 1 and img.format in ("JPEG", "MPO", "PNG"):
        name = stem + (".png" if img.format == "PNG" else ".jpg")
        (OUT / "images" / name).write_bytes(data)
    else:
        jpeg = img.format in ("JPEG", "MPO")
        if orientation != 1 and frame is not None and tuple(frame) == img.size:
            exif_fix = "tag dropped (annotations on stored pixels)"
        elif orientation != 1:
            img, exif_fix = ImageOps.exif_transpose(img), "rotated upright"
            digest = dhash(img)
        name = stem + (".jpg" if jpeg else ".png")
        if jpeg:
            img.convert("RGB").save(OUT / "images" / name, quality=95)
        else:
            img.save(OUT / "images" / name, compress_level=3)
    W, H = img.size
    if frame is not None and tuple(frame) != (W, H):
        (OUT / "images" / name).unlink()
        DROPPED[(source, "image size differs from the annotation size")] += 1
        return None
    IMAGES[name] = {"image": name, "source": source, "source_ref": ref, "license": license, "width": W, "height": H,
                    "group": f"{source}:{group}", "dhash": digest[0], "_bits": digest[1], "exif_fix": exif_fix}
    return name, W, H


def add_class(axis, name, cls):
    assert cls in CLASSES[axis], (axis, cls)
    old = CLASSIFICATION[axis].setdefault(name, cls)
    if old != cls:
        CONFLICTS[axis].add(name)


def add_boxes(name, boxes, negative=False):
    # boxes: [(x, y, w, h, class)] in pixels of the written image. negative=True writes an empty file for a verified negative.
    if not boxes and not negative:
        return
    W, H, source = IMAGES[name]["width"], IMAGES[name]["height"], IMAGES[name]["source"]
    lines = defaultdict(list)
    for x, y, w, h, cls in boxes:
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x < -2 or y < -2 or x + w > W + 2 or y + h > H + 2:
            NOTES[(source, "box clipped (reached > 2 px outside the image)")] += 1
        if x1 - x0 < 2 or y1 - y0 < 2:
            DROPPED[(source, "box < 2 px after clipping")] += 1
            continue
        axis = axis_of(cls)
        lines[axis].append(f"{axis_classes('detection', axis).index(cls)} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    if boxes and not lines:
        DROPPED[(source, "image lost all its boxes (not written as a negative)")] += 1
        return
    if negative:  # only condition negatives exist (healthy scenes): no source verifies "no flower / fruit here"
        SHAPES[("detection", "condition")].setdefault(name, [])
    for axis, axis_lines in lines.items():
        SHAPES[("detection", axis)].setdefault(name, []).extend(axis_lines)


def add_polygon(name, flat, cls):
    # flat: [x0, y0, x1, y1, ...] in pixels of the written image
    W, H, source = IMAGES[name]["width"], IMAGES[name]["height"], IMAGES[name]["source"]
    if len(flat) < 6:
        DROPPED[(source, "polygon with < 3 points")] += 1
        return
    xy = [f"{min(max(v / (W if i % 2 == 0 else H), 0), 1):.6f}" for i, v in enumerate(flat)]
    axis = axis_of(cls)
    SHAPES[("segmentation", axis)].setdefault(name, []).append(f"{axis_classes('segmentation', axis).index(cls)} " + " ".join(xy))


def condition_from_boxes(name, conditions):
    # rule 1: an image-level condition only if every condition box in the image agrees
    conditions = set(conditions)
    if len(conditions) == 1:
        add_class("condition", name, conditions.pop())
    elif len(conditions) > 1:
        DROPPED[(IMAGES[name]["source"], "condition: boxes of different conditions in one image")] += 1


def show_mapping(mapping):
    for k, v in mapping.items():
        print(f"  {str(k):<34} -> {v}")


def card_class_names(repo, column="objects"):
    info = HfApi().dataset_info(repo).card_data["dataset_info"]
    info = info[0] if isinstance(info, list) else info
    for feature in info["features"]:
        if feature["name"] == column:
            if "struct" in feature:
                names = next(s for s in feature["struct"] if s["name"] == "categories")["list"]["class_label"]["names"]
            else:
                names = feature["dtype"]["class_label"]["names"]
            return [names[str(i)] for i in range(len(names))]


def parquet_files(repo, folder="data"):
    root = snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{folder}/*.parquet", "README.md"])
    return sorted(Path(root, folder).glob("*.parquet"))


def parquet_rows(files, columns):
    # row by row, one row group in memory at a time
    for path in files:
        pf = pq.ParquetFile(path)
        for group in range(pf.num_row_groups):
            yield from pf.read_row_group(group, columns=columns).to_pylist()


def load_coco(folder, splits, json_name):
    # -> list of (split, image dict, [annotation dicts])
    out = []
    for split in splits:
        coco = json.loads(Path(folder, split, json_name).read_text())
        names = {c["id"]: c["name"] for c in coco["categories"]}
        by_image = defaultdict(list)
        for a in coco["annotations"]:
            by_image[a["image_id"]].append({**a, "class": names[a["category_id"]]})
        out += [(split, img, by_image[img["id"]]) for img in coco["images"]]
    return out


def rf_stem(file_name):
    # Roboflow names every copy <source>_<ext>.rf.<hash>.<ext>: copies of one photo share the stem
    return re.sub(r"\.rf\.[0-9a-zA-Z]+", "", file_name)
"""

FINAL = r"""
def finalize():
    # near-duplicates (same dHash) share one group; their classification labels must agree
    # near-duplicates: same 64-bit dHash, confirmed by <= 10 of 256 bits apart (fixed-camera scenes collide at 64 bits)
    by_hash = defaultdict(list)
    for name, r in IMAGES.items():
        by_hash[r["dhash"]].append(name)
    dup_groups = []
    for names in by_hash.values():
        while len(names) > 1:
            first, rest = names[0], names[1:]
            same = [first] + [n for n in rest if (IMAGES[n]["_bits"] != IMAGES[first]["_bits"]).sum() <= 10]
            if len(same) > 1:
                dup_groups.append(same)
            names = [n for n in rest if n not in same]
    parent = {}

    def find(g):
        while parent.get(g, g) != g:
            g = parent[g]
        return g

    for names in dup_groups:
        root = find(IMAGES[names[0]]["group"])
        for n in names[1:]:
            g = find(IMAGES[n]["group"])
            if g != root:
                parent[g] = root
    for r in IMAGES.values():
        r["group"] = find(r["group"])
    cross = sum(len({IMAGES[n]["source"] for n in v}) > 1 for v in dup_groups)
    print(f"near-duplicate groups (same dHash): {len(dup_groups):,} covering {sum(map(len, dup_groups)):,} images; "
          f"{cross:,} groups span two sources")

    for axis, labels in CLASSIFICATION.items():
        for n in CONFLICTS[axis]:
            labels.pop(n, None)
            DROPPED[(IMAGES[n]["source"], f"{axis}: two different labels for one image")] += 1
        for names in dup_groups:
            if len({labels[n] for n in names if n in labels}) > 1:
                for n in names:
                    if labels.pop(n, None):
                        DROPPED[(IMAGES[n]["source"], f"{axis}: near-duplicate of an image with another label")] += 1

    for name in [n for n in IMAGES if not has_label(n)]:
        (OUT / "images" / name).unlink()
        DROPPED[(IMAGES.pop(name)["source"], "image has no label left (not written)")] += 1

    cls_dir = OUT / "labels" / "classification"
    for axis, labels in CLASSIFICATION.items():
        pd.DataFrame(sorted(labels.items()), columns=["image", "class"]).to_csv(cls_dir / f"{axis}.csv", index=False)
    (cls_dir / "classes.json").write_text(json.dumps({k: CLASSES[k] for k in ("condition", "maturity")}, indent=1))
    for (task, axis), labels in SHAPES.items():
        folder = OUT / "labels" / task / axis
        (folder / "classes.txt").write_text("".join(f"{c}\n" for c in axis_classes(task, axis)))
        for name, lines in labels.items():
            (folder / f"{Path(name).stem}.txt").write_text("".join(f"{l}\n" for l in lines))
    manifest = pd.DataFrame(IMAGES.values()).drop(columns="_bits")
    manifest["label_space"] = manifest["source"].map(lambda s: ";".join(LABEL_SPACE.get(s, [])))
    manifest.to_csv(OUT / "manifest.csv", index=False)
    return manifest


def check():
    # read everything back from disk: every label points at an image, coordinates in [0, 1], class ids valid
    images = {p.name for p in (OUT / "images").iterdir()}
    stems = {Path(n).stem for n in images}
    manifest = pd.read_csv(OUT / "manifest.csv")
    assert set(manifest["image"]) == images and len(stems) == len(images)
    for axis in ("condition", "maturity"):
        df = pd.read_csv(OUT / "labels" / "classification" / f"{axis}.csv")
        assert df["image"].is_unique and set(df["image"]) <= images and set(df["class"]) <= set(CLASSES[axis]), axis
    for (task, axis), fields in ((k, 5 if k[0] == "detection" else None) for k in SHAPES):
        n_classes = len((OUT / "labels" / task / axis / "classes.txt").read_text().split())
        for path in (OUT / "labels" / task / axis).glob("*.txt"):
            if path.name == "classes.txt":
                continue
            assert path.stem in stems, path
            for line in path.read_text().splitlines():
                values = line.split()
                assert 0 <= int(values[0]) < n_classes, path
                assert all(0 <= float(v) <= 1 for v in values[1:]), path
                assert len(values) == fields if fields else (len(values) >= 7 and len(values) % 2 == 1), path
    print("self-check passed:", len(images), "images")


def summary(manifest):
    rows = []
    for source, df in manifest.groupby("source", sort=False):
        names = set(df["image"])
        rows.append({"source": source, "license": df["license"].iloc[0], "images": len(df),
                     "condition": sum(n in names for n in CLASSIFICATION["condition"]),
                     "maturity": sum(n in names for n in CLASSIFICATION["maturity"]),
                     **{f"{t[:3]}/{a[:4]} {'boxes' if t == 'detection' else 'polys'}": sum(len(v) for n, v in SHAPES[(t, a)].items() if n in names)
                        for t, a in SHAPES},
                     "groups": df["group"].nunique()})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\ntotal images: {len(manifest):,}; images with no label at all: "
          f"{sum(1 for n in manifest['image'] if not has_label(n)):,}")
    if DROPPED:
        print("\ndropped:")
        for (source, reason), n in sorted(DROPPED.items()):
            print(f"  {source:<26}{reason:<62}{n:>7,}")
    if NOTES:
        print("\nnotes (kept):")
        for (source, note), n in sorted(NOTES.items()):
            print(f"  {source:<26}{note:<62}{n:>7,}")
    print("\nexif fixes:", manifest["exif_fix"].fillna("").replace("", "none").value_counts().to_dict())


def class_counts():
    for axis, labels in CLASSIFICATION.items():
        print(f"{axis} (images):", dict(Counter(labels.values()).most_common()) or "none")
    for (task, axis), labels in SHAPES.items():
        counts = Counter(axis_classes(task, axis)[int(l.split()[0])] for lines in labels.values() for l in lines)
        print(f"{task}/{axis} ({'boxes' if task == 'detection' else 'polygons'}, {len(labels):,} images):", dict(counts.most_common()) or "none")
"""

SPLIT_PRINT = r"""
for task, axis in SHAPES:
    print(f"{task}/{axis}: {axis_classes(task, axis)}")
"""

ROUNDTRIP = r"""
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#7f4fc9", "#0d366b", "#898781"]


def read_labels(name):
    # labels as written to disk (not the in-memory copy), for a true round trip
    out = {}
    for task, axis in SHAPES:
        path = OUT / "labels" / task / axis / f"{Path(name).stem}.txt"
        out[(task, axis)] = [l.split() for l in path.read_text().splitlines()] if path.exists() else []
    return out


def show_roundtrip(source, n=6, thumb=420):
    names = [r["image"] for r in IMAGES.values() if r["source"] == source]
    with_shapes = [x for x in names if any(d.get(x) for d in SHAPES.values())]
    pool = with_shapes or names
    pick = [pool[i] for i in RNG.choice(len(pool), min(n, len(pool)), replace=False)]
    cols = 3
    fig, axes = plt.subplots(int(np.ceil(len(pick) / cols)), cols, figsize=(12, 3.8 * int(np.ceil(len(pick) / cols))), layout="constrained")
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, name in zip(np.ravel(axes), pick):
        img = Image.open(OUT / "images" / name).convert("RGB")
        scale = thumb / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        W, H = img.size
        draw = ImageDraw.Draw(img)
        labels, used = read_labels(name), set()
        for (task, axis), lines in labels.items():
            names_ = axis_classes(task, axis)
            for c, *values in lines:
                color = PALETTE[(int(c) + (4 if axis == "maturity" else 0)) % len(PALETTE)]
                if task == "detection":
                    cx, cy, w, h = float(values[0]) * W, float(values[1]) * H, float(values[2]) * W, float(values[3]) * H
                    draw.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=color, width=3)
                else:
                    draw.polygon([float(v) * (W if i % 2 == 0 else H) for i, v in enumerate(values)], outline=color, width=2)
                used.add(f"{axis[:4]}:{names_[int(c)]}")
        tags = [CLASSIFICATION[a][name] for a in CLASSIFICATION if name in CLASSIFICATION[a]]
        ax.imshow(img)
        ax.set_title(f"{name}\n{' | '.join(tags) or '-'}\n{', '.join(sorted(used))[:70]}", fontsize=7, color="#52514e")
    fig.suptitle(f"{CROP}: {source} (labels read back from disk)", x=0.01, ha="left", fontweight="bold")
    fig.savefig(EXAMPLE_DIR / f"{CROP}_{source}.png", dpi=110)
    plt.show()
"""

# ---------------------------------------------------------------- source loaders

LOADERS = {}

LOADERS["plantvillage"] = (r"""
## PlantVillage: `BrandonFors/Plant-Diseases-PlantVillage-Dataset`

Lab photos of single leaves (CC0 in the HF metadata). Each image has one `Crop___condition` label, which goes straight to `condition`. Only classes of this crop are used. ⚠️ The images are lab photos on plain backgrounds, so do not validate a field model on them.
""", r"""
from datasets import Image as HFImage
from datasets import load_dataset


def load_plantvillage(mapping):
    pv = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset")
    names = pv["train"].features["label"].names
    assert set(mapping) <= set(names), set(mapping) - set(names)
    wanted = {names.index(k): v for k, v in mapping.items()}
    for split in pv:
        ds = pv[split].cast_column("image", HFImage(decode=False))
        idx = [i for i, label in enumerate(ds["label"]) if label in wanted]
        for i, row in zip(idx, ds.select(idx)):
            item = row["image"]
            added = add_image(item["bytes"] or Path(item["path"]).read_bytes(), "plantvillage", f"{split}/{i}",
                              "CC0 1.0 (HF metadata)", f"{split}/{i}")
            if added:
                add_class("condition", added[0], wanted[row["label"]])
""")

LOADERS["plantseg"] = (r"""
## PlantSeg v3: `XuVV/plantsegv3`

In-the-wild disease photos with **lesion polygons and boxes**. One disease per image, which goes to `condition`. Boxes become `lesion__<disease>` detections and polygons become `lesion__<disease>` segmentation, one line per polygon part. ⚠️ **CC BY-NC-ND 4.0** (non-commercial), and the images are web-scraped. There is no healthy class.
""", r"""
def load_plantseg(prefix, diseases):
    files = sorted((Path(snapshot_download("XuVV/plantsegv3", repo_type="dataset")) / "data").glob("*.parquet"))
    found = Counter()
    for path in files:
        pf = pq.ParquetFile(path)
        for g in range(pf.num_row_groups):
            if not any(a.startswith(prefix) for a in pf.read_row_group(g, columns=["answer"]).column("answer").to_pylist()):
                continue
            for row in pf.read_row_group(g, columns=["image", "file_name", "width", "height", "answer", "bboxes", "segmentations"]).to_pylist():
                if not row["answer"].startswith(prefix):
                    continue
                found[row["answer"]] += 1
                cond = diseases[row["answer"][len(prefix):]]
                added = add_image(row["image"]["bytes"], "plantseg", f"{path.name.split('-')[0]}/{row['file_name']}",
                                  "CC BY-NC-ND 4.0 (original); web-scraped images", row["file_name"], frame=(row["width"], row["height"]))
                if not added:
                    continue
                name = added[0]
                add_class("condition", name, cond)
                add_boxes(name, [(*b, f"lesion__{cond}") for b in row["bboxes"]])
                for instance in row["segmentations"]:
                    for poly in instance:
                        add_polygon(name, poly, f"lesion__{cond}")
    print("PlantSeg images per class:", dict(found))
    assert set(found) == {prefix + d for d in diseases}, "mapping does not match the classes present"
""")

LOADERS["plantdoc"] = (r"""
## PlantDoc: `susnato/plant_disease_detection_processed`

In-the-wild leaf detection (CC BY 4.0, many web stock photos). Each box covers a **whole leaf** and gives its condition, so boxes become `leaf__<condition>`. The class names are not in the data; they come from the mapping recovered in `plague.ipynb`. Only the light columns and the `image` column are read, never `pixel_values`.
- Images that also contain boxes of another crop are dropped, because only some of their boxes could be labeled.
- The image-level `condition` is used only when every box in the image agrees (rule 1).
""", r"""
PD_REPO = "susnato/plant_disease_detection_processed"
PD_NAMES = [
    "Cherry leaf", "Peach leaf", "Corn leaf blight", "Apple rust leaf", "Potato leaf late blight",
    "Strawberry leaf", "Tomato leaf late blight", "Tomato mold leaf", "Potato leaf early blight", "Apple leaf",
    "Tomato leaf yellow virus", "Blueberry leaf", "Tomato leaf mosaic virus", "Raspberry leaf", "Tomato leaf bacterial spot",
    "Squash Powdery mildew leaf", "grape leaf", "Tomato Early blight leaf", "Apple Scab Leaf", "Tomato Septoria leaf spot",
    "Tomato leaf", "Soyabean leaf", "Bell_pepper leaf spot", "Bell_pepper leaf",
]


def load_plantdoc(mapping):
    # mapping: PlantDoc class -> condition; the box class is leaf__<condition>
    fs = HfFileSystem()
    for path in sorted(fs.glob(f"datasets/{PD_REPO}/data/*.parquet")):
        split = path.rsplit("/", 1)[-1].split("-")[0]
        with fs.open(path, "rb", block_size=2**20) as fh:
            pf = pq.ParquetFile(fh)
            meta = pf.read(columns=["image_id", "width", "height", "objects"]).to_pylist()
            keep = [i for i, r in enumerate(meta) if any(PD_NAMES[c] in mapping for c in r["objects"]["category"])]
            if not keep:
                continue
            images = pf.read(columns=["image"]).column("image").to_pylist()
        for i in keep:
            r = meta[i]
            names = [PD_NAMES[c] for c in r["objects"]["category"]]
            if not all(n in mapping for n in names):
                DROPPED[("plantdoc", "image also has boxes of another crop")] += 1
                continue
            added = add_image(images[i]["bytes"], "plantdoc", f"{split}/{r['image_id']}", "CC BY 4.0; web images",
                              f"{split}/{r['image_id']}", frame=(r["width"], r["height"]))
            if added:
                add_boxes(added[0], [(*b, f"leaf__{mapping[n]}") for b, n in zip(r["objects"]["bbox"], names)])
                condition_from_boxes(added[0], [mapping[n] for n in names])
""")

LOADERS["strawberry12"] = (r"""
## Strawberry_12: `Ethan0300/Strawberry_12-dataset` (MIT)

YOLO boxes for fruit, flower, 5 diseases, pests and 3 deficiencies (`data/strawberry12/`). Class names are not shipped; they come from file-name prefixes, as recovered in `plague.ipynb`.
- **Only original photos are used.** The `_blur/_bright/_dark/_noise` copies (57% of images) are dropped.
- Disease, pest and deficiency boxes enclose the **affected organ** (a whole moldy fruit, a leaf with mites), so they become `affected__<condition>`.
- `unripe fruit` is coarse (green + white + turning) and becomes `fruit__unripe`.
- The image `condition` comes from rule 1. Fruit and flower boxes are not conditions.
- Some "originals" are rotated augmented copies with black corners. Images with at least 2 black corners are dropped.
- ⚠️ Not fixable automatically: some "originals" are 2×2 mosaics, web images with text, or have black cut-out rectangles (see the round-trip figure).
""", r"""
S12_DIR = DATA_DIR / "strawberry12" / "Strawberry_12 dataset"
S12_NAMES = ["ripe fruit", "unripe fruit", "flower", "leaf spot", "powdery mildew", "gray mold", "anthracnose", "blight",
             "insect pests", "calcium deficiency", "iron deficiency", "potassium (K/P) deficiency"]
S12_MAP = {"ripe fruit": ("fruit__ripe", None), "unripe fruit": ("fruit__unripe", None), "flower": ("flower", None),
           "leaf spot": ("affected__leaf_spot", "leaf_spot"), "powdery mildew": ("affected__powdery_mildew", "powdery_mildew"),
           "gray mold": ("affected__gray_mold", "gray_mold"), "anthracnose": ("affected__anthracnose", "anthracnose"),
           "blight": ("affected__blight", "blight"), "insect pests": ("affected__insect_pest", "insect_pest"),
           "calcium deficiency": ("affected__calcium_deficiency", "calcium_deficiency"),
           "iron deficiency": ("affected__iron_deficiency", "iron_deficiency"),
           "potassium (K/P) deficiency": ("affected__potassium_or_phosphorus_deficiency", "potassium_or_phosphorus_deficiency")}
AUG = re.compile(r"_(blur|bright|dark|noise)$")


def load_strawberry12():
    show_mapping({k: f"{d}  (condition: {c})" for k, (d, c) in S12_MAP.items()})
    LABEL_SPACE["strawberry12"] = [d for d, _ in S12_MAP.values()]
    for label_file in sorted((S12_DIR / "labels").glob("*/*.txt")):
        if AUG.search(label_file.stem):
            DROPPED[("strawberry12", "augmented copy (_blur/_bright/_dark/_noise)")] += 1
            continue
        image = next((p for ext in (".jpg", ".jpeg", ".png", ".JPG") if (p := S12_DIR / "images" / label_file.parent.name / f"{label_file.stem}{ext}").exists()), None)
        if image is None:
            DROPPED[("strawberry12", "label file without image")] += 1
            continue
        with Image.open(image) as img:
            a = np.asarray(img.convert("L"))
        if sum(q.max() < 12 for q in (a[:6, :6], a[:6, -6:], a[-6:, :6], a[-6:, -6:])) >= 2:
            DROPPED[("strawberry12", "rotated copy (>= 2 black corners)")] += 1
            continue
        added = add_image(image.read_bytes(), "strawberry12", f"{label_file.parent.name}/{image.name}", "MIT", label_file.stem)
        if not added:
            continue
        name, W, H = added
        boxes = []
        for line in label_file.read_text().splitlines():
            if line.strip():
                c, x, y, w, h = line.split()[:5]
                x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
                boxes.append((x - w / 2, y - h / 2, w, h, S12_MAP[S12_NAMES[int(c)]][0]))
        add_boxes(name, boxes)
        condition_from_boxes(name, [b[4].removeprefix("affected__") for b in boxes if b[4].startswith("affected__")])
""")

LOADERS["agml_detection"] = (None, r"""
def load_agml_detection(repo, source, license, mapping, names=None, group_of=lambda row: row["image"]["path"], extra=(), on_row=None):
    # mapping: source class name -> detection class. AgML boxes are [x, y, w, h] in pixels of the upright (EXIF-applied) photo.
    names = names or card_class_names(repo)
    show_mapping(mapping)
    LABEL_SPACE[source] = sorted(set(mapping.values()))
    for row in parquet_rows(parquet_files(repo), ["image", "objects", *extra]):
        added = add_image(row["image"]["bytes"], source, row["image"]["path"], license, group_of(row))
        if not added:
            continue
        cats = [names[c] for c in row["objects"]["categories"]]
        add_boxes(added[0], [(*b, mapping[c]) for b, c in zip(row["objects"]["bbox"], cats)])
        if on_row:
            on_row(added[0], row, cats)
""")

LOADERS["roboflow"] = (None, r"""
def load_roboflow(folder, source, mapping, drop_image_if=lambda anns: False):
    # COCO export (raw version, no augmentation). Keeps one copy per source photo (stem) to avoid duplicates.
    show_mapping(mapping)
    LABEL_SPACE[source] = sorted(set(mapping.values()))
    seen = set()
    for split, img, anns in load_coco(folder, ("train", "valid", "test"), "_annotations.coco.json"):
        stem = rf_stem(img["file_name"])
        if stem in seen:
            DROPPED[(source, "second copy of the same source photo (.rf.<hash>)")] += 1
            continue
        seen.add(stem)
        if drop_image_if(anns):
            DROPPED[(source, "image has a broken (point) label")] += 1
            continue
        added = add_image(Path(folder, split, img["file_name"]).read_bytes(), source, f"{split}/{img['file_name']}", "CC BY 4.0",
                          stem, frame=(img["width"], img["height"]))
        if added:
            add_boxes(added[0], [(*a["bbox"], mapping[a["class"]]) for a in anns if a["class"] in mapping])
            NOTES[(source, "annotation of a class outside the mapping (ignored)")] += sum(a["class"] not in mapping for a in anns)
""")

# ---------------------------------------------------------------- crops

STRAWBERRY = dict(title="strawberry", classes=r"""
CLASSES = {
    "condition": ["healthy", "leaf_scorch", "anthracnose", "leaf_spot", "powdery_mildew", "gray_mold", "blight", "insect_pest",
                  "calcium_deficiency", "iron_deficiency", "potassium_or_phosphorus_deficiency"],
    "maturity": [],
    "detection": ["flower", "fruit__green", "fruit__white", "fruit__turning", "fruit__ripe", "fruit__overripe", "fruit__unripe",
                  "leaf__healthy", "lesion__anthracnose", "lesion__leaf_scorch",
                  "affected__leaf_spot", "affected__powdery_mildew", "affected__gray_mold", "affected__anthracnose", "affected__blight",
                  "affected__insect_pest", "affected__calcium_deficiency", "affected__iron_deficiency",
                  "affected__potassium_or_phosphorus_deficiency"],
    "segmentation": ["lesion__anthracnose", "lesion__leaf_scorch"],
}
""", note=r"""
`fruit__unripe` (Strawberry_12) is the **coarse parent** of `fruit__green`, `fruit__white` and `fruit__turning`. Merge them for a 2-stage model, or mask `fruit__unripe` when training the fine stages. No source labels strawberry maturity per image, so `maturity.csv` is empty.

**Excluded:** AgML `strawberry_avocado_ripeness_detection`. Its class names are inferred, not from the source, its boxes on portrait photos have swapped W/H with ambiguous fixes, and some images are composites.
""", sections=[
    ("plantvillage", r"""
load_plantvillage({"Strawberry___healthy": "healthy", "Strawberry___Leaf_scorch": "leaf_scorch"})
"""),
    ("plantseg", r"""
LABEL_SPACE["plantseg"] = ["lesion__anthracnose", "lesion__leaf_scorch"]
load_plantseg("strawberry ", {"anthracnose": "anthracnose", "leaf scorch": "leaf_scorch"})
"""),
    ("plantdoc", r"""
LABEL_SPACE["plantdoc"] = ["leaf__healthy"]
load_plantdoc({"Strawberry leaf": "healthy"})
"""),
    ("strawberry12", r"""
load_strawberry12()
"""),
    ("agml_detection", r"""
## Strawberry growth: `Project-AgML/strawberry_growth_detection` (CC BY 4.0)

Fruit-stage boxes. `small g` and `green` both become `fruit__green`. Both `tagged` (repeat photos of the same fruits over time) and `random` images are used. ⚠️ Tagged fruits appear on several dates, but the files do not carry a fruit id, so `group` cannot tie those photos together.
@@
load_agml_detection("Project-AgML/strawberry_growth_detection", "agml_strawberry_growth", "CC BY 4.0",
                    {"flower": "flower", "small g": "fruit__green", "green": "fruit__green", "white": "fruit__white",
                     "turning red": "fruit__turning", "red": "fruit__ripe", "overripe": "fruit__overripe"})
"""),
    ("roboflow", r"""
## Strawberry 5-stage (Roboflow): `strawberry-vnobp/strawberry-2-k4eap` (CC BY 4.0)

Raw export of the fork in `data/roboflow/strawberry5/`. Only one copy per source photo is kept. The file-name prefixes `plague`/`chlorosis`/`wilt` hint at plant condition, but they are not labels, so they are not used.
@@
load_roboflow(DATA_DIR / "roboflow" / "strawberry5", "rf_strawberry5",
              {"Flower": "flower", "Green": "fruit__green", "White": "fruit__white", "Turning Red": "fruit__turning", "Red": "fruit__ripe"})
"""),
])

BLUEBERRY = dict(title="blueberry", classes=r"""
CLASSES = {
    "condition": ["healthy", "anthracnose", "botrytis_blight", "mummy_berry", "rust", "scorch"],
    "maturity": [],
    "detection": ["fruit__unripe", "fruit__ripe", "leaf__healthy", "lesion__anthracnose", "lesion__botrytis_blight",
                  "lesion__mummy_berry", "lesion__rust", "lesion__scorch"],
    "segmentation": ["fruit__unripe", "fruit__ripe", "lesion__anthracnose", "lesion__botrytis_blight", "lesion__mummy_berry",
                     "lesion__rust", "lesion__scorch"],
}
""", note=r"""
Blueberry fruit is labeled in two stages only (`unripe`, `ripe`). No source labels maturity per image, so `maturity.csv` is empty. Disease data is **thin** (PlantSeg, 36–47 images per disease, web images).
""", sections=[
    ("plantvillage", r"""
load_plantvillage({"Blueberry___healthy": "healthy"})
"""),
    ("plantseg", r"""
LABEL_SPACE["plantseg"] = ["lesion__anthracnose", "lesion__botrytis_blight", "lesion__mummy_berry", "lesion__rust", "lesion__scorch"]
load_plantseg("blueberry ", {"anthracnose": "anthracnose", "botrytis blight": "botrytis_blight", "mummy berry": "mummy_berry",
                             "rust": "rust", "scorch": "scorch"})
"""),
    ("plantdoc", r"""
LABEL_SPACE["plantdoc"] = ["leaf__healthy"]
load_plantdoc({"Blueberry leaf": "healthy"})
"""),
    ("agml_detection", r"""
## Blueberry maturity: `Project-AgML/blueberry_maturity_detection` (CC BY 4.0)

Dense canopy images. ⚠️ The card's names (`unblue_visible`, `unblue_occluded`, ...) do not fit the data. Only ids 0 and 3 are used, and the mean box colour shows that **0 = unripe and 3 = ripe** (`growth.ipynb`). The ids are mapped directly.
@@
load_agml_detection("Project-AgML/blueberry_maturity_detection", "agml_blueberry_maturity", "CC BY 4.0",
                    {"id 0": "fruit__unripe", "id 3": "fruit__ripe"}, names={0: "id 0", 3: "id 3"})
"""),
    ("blueberry_seg", r"""
## Blueberry segmentation: `c-tan/Blueberry-fruit-segmentation`

Berry polygons, `Immature`/`Mature` (Ni et al. 2020; ⚠️ **CC BY-NC-SA 4.0**). They become `fruit__unripe`/`fruit__ripe` polygons, with boxes from the COCO bbox. `test2017` has no annotations and is dropped.
@@
BS_DIR = DATA_DIR / "blueberry_seg" / "berry_coco-formant-annotation"
if not BS_DIR.exists():
    zipfile.ZipFile(hf_hub_download("c-tan/Blueberry-fruit-segmentation", "berry_coco-formant-annotation.zip", repo_type="dataset")).extractall(BS_DIR.parent)
BS_MAP = {"Immature": "fruit__unripe", "Mature": "fruit__ripe"}
show_mapping(BS_MAP)
LABEL_SPACE["blueberry_seg"] = sorted(BS_MAP.values())
for split in ("train2017", "val2017"):
    coco = json.loads((BS_DIR / "annotations" / f"instances_{split}.json").read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    by_image = defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)
    for img in coco["images"]:
        if not by_image[img["id"]]:
            DROPPED[("blueberry_seg", "image without annotations")] += 1
            continue
        added = add_image((BS_DIR / split / img["file_name"]).read_bytes(), "blueberry_seg", f"{split}/{img['file_name']}",
                          "CC BY-NC-SA 4.0", img["file_name"], frame=(img["width"], img["height"]))
        if not added:
            continue
        anns = by_image[img["id"]]
        add_boxes(added[0], [(*a["bbox"], BS_MAP[names[a["category_id"]]]) for a in anns])
        for a in anns:
            for poly in a["segmentation"]:
                add_polygon(added[0], poly, BS_MAP[names[a["category_id"]]])
DROPPED[("blueberry_seg", "test2017 split (no annotations)")] += len(list((BS_DIR / "test2017").glob("*")))
"""),
])

RASPBERRY = dict(title="raspberry", classes=r"""
CLASSES = {
    "condition": ["healthy", "fire_blight", "gray_mold", "leaf_spot", "yellow_rust"],
    "maturity": [],
    "detection": ["bud", "flower", "fruit__unripe", "fruit__ripe", "leaf__healthy", "lesion__fire_blight", "lesion__gray_mold",
                  "lesion__leaf_spot", "lesion__yellow_rust"],
    "segmentation": ["lesion__fire_blight", "lesion__gray_mold", "lesion__leaf_spot", "lesion__yellow_rust"],
}
""", note=r"""
Disease data is **thin** (PlantSeg, 18–40 images per disease, web images). No source labels maturity per image, so `maturity.csv` is empty.
""", sections=[
    ("plantvillage", r"""
load_plantvillage({"Raspberry___healthy": "healthy"})
"""),
    ("plantseg", r"""
LABEL_SPACE["plantseg"] = ["lesion__fire_blight", "lesion__gray_mold", "lesion__leaf_spot", "lesion__yellow_rust"]
load_plantseg("raspberry ", {"fire blight": "fire_blight", "gray mold": "gray_mold", "leaf spot": "leaf_spot", "yellow rust": "yellow_rust"})
"""),
    ("plantdoc", r"""
LABEL_SPACE["plantdoc"] = ["leaf__healthy"]
load_plantdoc({"Raspberry leaf": "healthy"})
"""),
    ("roboflow", r"""
## Raspberry (Roboflow): `ssd-nv4zv/ssd-raspberry` (CC BY 4.0)

Buds → flowers → unripe → ripe berries. ⚠️ About 14% of train labels are **point labels turned into huge boxes** (polygon = centre point + junk point + `None`). Dropping only those boxes would leave the fruit unlabeled in the image, so **every image with a broken label is dropped whole**.
@@
def broken(anns):
    return any(v is None for a in anns for part in (a.get("segmentation") or []) for v in part)


load_roboflow(DATA_DIR / "roboflow" / "raspberry", "rf_raspberry",
              {"Buds": "bud", "Flowers": "flower", "Unripe_Berries": "fruit__unripe", "Ripe_Berries": "fruit__ripe"}, drop_image_if=broken)
"""),
])

TOMATO = dict(title="tomato", classes=r"""
CLASSES = {
    "condition": ["healthy", "bacterial_spot", "early_blight", "late_blight", "leaf_mold", "septoria_leaf_spot", "spider_mites",
                  "target_spot", "mosaic_virus", "yellow_leaf_curl_virus", "leaf_blight", "leaf_curl", "verticillium_wilt",
                  "gray_mold", "viral_disease", "bacterial_wilt"],
    "maturity": ["immature", "mature"],
    "detection": ["fruit__green", "fruit__breaker", "fruit__reddish", "fruit__half_ripe", "fruit__ripe",
                  "leaf__healthy", "leaf__bacterial_spot", "leaf__early_blight", "leaf__late_blight", "leaf__leaf_mold",
                  "leaf__septoria_leaf_spot", "leaf__mosaic_virus", "leaf__yellow_leaf_curl_virus",
                  "lesion__bacterial_spot", "lesion__early_blight", "lesion__late_blight", "lesion__leaf_mold", "lesion__mosaic_virus",
                  "lesion__septoria_leaf_spot", "lesion__yellow_leaf_curl_virus",
                  "fruit__gray_mold", "leaf__gray_mold", "leaf__viral_disease", "plant_top__viral_disease", "plant_middle__viral_disease",
                  "stem_base__bacterial_wilt", "leaf__bacterial_wilt", "plant_middle__bacterial_wilt", "stem__bacterial_wilt",
                  "plant_top__bacterial_wilt"],
    "segmentation": ["lesion__bacterial_spot", "lesion__early_blight", "lesion__late_blight", "lesion__leaf_mold", "lesion__mosaic_virus",
                     "lesion__septoria_leaf_spot", "lesion__yellow_leaf_curl_virus"],
}
""", note=r"""
**Name reconciliation.** PlantSeg `bacterial leaf spot` = PlantVillage `Bacterial_spot` (same disease, *Xanthomonas*). PlantDoc `leaf yellow virus` = `yellow_leaf_curl_virus` (PlantDoc's tomato classes mirror PlantVillage's). Some classes are **not merged** because the source does not say enough:
- CCMT `leaf blight` (could be early or late blight) → `leaf_blight`.
- CCMT `leaf curl` (cause not given) → `leaf_curl`.
- Liu `Viral` (virus not named) → `viral_disease`.

**Fruit stages.** `fruit__half_ripe` (Laboro) is the **coarse parent** of `fruit__breaker` and `fruit__reddish` (AgRob, USDA colour stages). Laboro's big/cherry (`b_`/`l_`) size split is dropped.

**Maturity.** Khatun `Mature` covers both turning and ripe fruit (see `growth.ipynb`).

**Excluded:**
- AgML `tomato_leaf_disease_detection` (Imtiaz): 6 of its 7 class names are unknown.
- AgML `tomato_leaf_disease`: a PlantVillage re-upload.
""", sections=[
    ("plantvillage", r"""
load_plantvillage({"Tomato___healthy": "healthy", "Tomato___Bacterial_spot": "bacterial_spot", "Tomato___Early_blight": "early_blight",
                   "Tomato___Late_blight": "late_blight", "Tomato___Leaf_Mold": "leaf_mold", "Tomato___Septoria_leaf_spot": "septoria_leaf_spot",
                   "Tomato___Spider_mites Two-spotted_spider_mite": "spider_mites", "Tomato___Target_Spot": "target_spot",
                   "Tomato___Tomato_mosaic_virus": "mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "yellow_leaf_curl_virus"})
"""),
    ("plantseg", r"""
TS_MAP = {"bacterial leaf spot": "bacterial_spot", "early blight": "early_blight", "late blight": "late_blight", "leaf mold": "leaf_mold",
          "mosaic virus": "mosaic_virus", "septoria leaf spot": "septoria_leaf_spot", "yellow leaf curl virus": "yellow_leaf_curl_virus"}
LABEL_SPACE["plantseg"] = [f"lesion__{v}" for v in TS_MAP.values()]
load_plantseg("tomato ", TS_MAP)
"""),
    ("plantdoc", r"""
TD_DOC = {"Tomato leaf": "healthy", "Tomato leaf bacterial spot": "bacterial_spot", "Tomato Early blight leaf": "early_blight",
          "Tomato leaf late blight": "late_blight", "Tomato mold leaf": "leaf_mold", "Tomato Septoria leaf spot": "septoria_leaf_spot",
          "Tomato leaf mosaic virus": "mosaic_virus", "Tomato leaf yellow virus": "yellow_leaf_curl_virus"}
LABEL_SPACE["plantdoc"] = [f"leaf__{v}" for v in TD_DOC.values()]
show_mapping(TD_DOC)
load_plantdoc(TD_DOC)
"""),
    ("ccmt", r"""
## CCMT tomato: `Project-AgML/crop_pest_disease_classification` (CC BY 4.0)

Field photos from Ghana, `raw` config, tomato rows only. ⚠️ The set has byte-identical duplicates. Identical images (md5) are reduced to one copy, and **every copy is dropped when the copies carry different labels**.
@@
CC_FILES = sorted((Path(snapshot_download("Project-AgML/crop_pest_disease_classification", repo_type="dataset",
                                          allow_patterns=["raw/*", "README.md"])) / "raw").glob("*.parquet"))
cc_names = json.loads(pq.ParquetFile(CC_FILES[0]).schema_arrow.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
CC_MAP = {"healthy": "healthy", "leaf blight": "leaf_blight", "leaf curl": "leaf_curl", "septoria leaf spot": "septoria_leaf_spot",
          "verticulium wilt": "verticillium_wilt"}
show_mapping(CC_MAP)
labels_of = defaultdict(set)
for row in parquet_rows(CC_FILES, ["image", "label", "crop"]):
    if row["crop"] == "Tomato":
        labels_of[hashlib.md5(row["image"]["bytes"]).hexdigest()].add(row["label"])
done = set()
for row in parquet_rows(CC_FILES, ["image", "label", "crop"]):
    if row["crop"] != "Tomato":
        continue
    md5 = hashlib.md5(row["image"]["bytes"]).hexdigest()
    if len(labels_of[md5]) > 1:
        DROPPED[("ccmt", "byte-identical copies with different labels")] += 1
        continue
    if md5 in done:
        DROPPED[("ccmt", "byte-identical duplicate")] += 1
        continue
    done.add(md5)
    added = add_image(row["image"]["bytes"], "ccmt", row["image"]["path"], "CC BY 4.0", md5)
    if added:
        add_class("condition", added[0], CC_MAP[cc_names[row["label"]]])
"""),
    ("agml_detection", r"""
## Tomato disease detection (Liu): `Project-AgML/tomato_disease_detection` (CC BY 4.0)

Boxes on diseased fruit, leaves, stem base and plant regions. Names come from the card text: 0 GrayMold_Fruit … 9 Wilt_Top. The image-level `disease` column (`GrayMold`/`Viral`/`Wilt`) gives `condition`, **but only when every box in the image is of that disease**. The 7 images where boxes and column disagree keep their boxes but get no class.
@@
TD_NAMES = ["GrayMold_Fruit", "GrayMold_Leaf", "Viral_Leaf", "Viral_Top", "Virus_Middle", "Wilt_Base", "Wilt_Leaf", "Wilt_Middle", "Wilt_Stem", "Wilt_Top"]
LIU_MAP = dict(zip(TD_NAMES, ["fruit__gray_mold", "leaf__gray_mold", "leaf__viral_disease", "plant_top__viral_disease",
                              "plant_middle__viral_disease", "stem_base__bacterial_wilt", "leaf__bacterial_wilt",
                              "plant_middle__bacterial_wilt", "stem__bacterial_wilt", "plant_top__bacterial_wilt"]))
LIU_DISEASE = {"GrayMold": "gray_mold", "Viral": "viral_disease", "Wilt": "bacterial_wilt"}


def liu_condition(name, row, cats):
    cond = LIU_DISEASE[row["disease"]]
    if not cats:
        NOTES[("liu_tomato_disease", "image without boxes (class kept, no detection file)")] += 1
    if all(LIU_MAP[c].endswith(cond) for c in cats):
        add_class("condition", name, cond)
    else:
        DROPPED[("liu_tomato_disease", "condition: boxes disagree with the disease column")] += 1


load_agml_detection("Project-AgML/tomato_disease_detection", "liu_tomato_disease", "CC BY 4.0", LIU_MAP, names=TD_NAMES,
                    extra=("disease",), on_row=liu_condition)
"""),
    ("agml_detection", r"""
## Laboro Tomato: `Project-AgML/tomato_ripeness_detection`

⚠️ **CC BY-NC 4.0** (original Laboro release). 305 photos carry an EXIF "rotate 90°" tag, and their boxes are drawn on the **rotated** photo, so those images are written upright (`exif_fix` = `rotated upright`). `half_ripened` is coarse and becomes `fruit__half_ripe`.
@@
load_agml_detection("Project-AgML/tomato_ripeness_detection", "laboro_tomato", "CC BY-NC 4.0 (original Laboro)",
                    {f"{s}_{stage}": cls for s in ("b", "l") for stage, cls in
                     (("green", "fruit__green"), ("half_ripened", "fruit__half_ripe"), ("fully_ripened", "fruit__ripe"))})
"""),
    ("agml_detection", r"""
## AgRob greenhouse tomato: `Project-AgML/Greenhouse_Tomato_AgRob_object_detection` (CC BY 4.0)

Robot video frames taken 3 s apart, so `group` = capture session. The 4 USDA colour stages keep their own names; `reddish` stays `fruit__reddish` and is not forced into another stage.
@@
load_agml_detection("Project-AgML/Greenhouse_Tomato_AgRob_object_detection", "agrob_tomato", "CC BY 4.0",
                    {"unripe": "fruit__green", "breaking": "fruit__breaker", "reddish": "fruit__reddish", "ripe": "fruit__ripe"},
                    group_of=lambda row: row["capture_session"], extra=("capture_session",))
"""),
    ("tomato_maturity", r"""
## Tomato maturity (Khatun): `Project-AgML/tomato_maturity_classification` (CC BY 4.0)

Image classification `Immature`/`Mature`, `raw` config only (the `augmented` config is generated copies). Most capture days hold only one class, so `group` = capture day (`IMG_YYYYMMDD`): the day's light and background would otherwise leak into the label.
@@
TM_REPO = "Project-AgML/tomato_maturity_classification"
tm_names = card_class_names(TM_REPO, column="label")
TM_MAP = {"Immature": "immature", "Mature": "mature"}
show_mapping(TM_MAP)
for row in parquet_rows(parquet_files(TM_REPO, "raw"), ["image", "label"]):
    m = re.search(r"IMG_(\d{8})", row["image"]["path"])
    added = add_image(row["image"]["bytes"], "tomato_maturity", row["image"]["path"], "CC BY 4.0", m.group(1) if m else row["image"]["path"])
    if added:
        add_class("maturity", added[0], TM_MAP[tm_names[row["label"]]])
"""),
])

BELL_PEPPER = dict(title="bell pepper", classes=r"""
CLASSES = {
    "condition": ["healthy", "bacterial_spot", "leaf_spot", "blossom_end_rot", "frogeye_leaf_spot", "powdery_mildew"],
    "maturity": ["stage_1_green", "stage_2", "stage_3", "stage_4_coloured"],
    "detection": ["leaf__healthy", "leaf__leaf_spot", "lesion__bacterial_spot", "lesion__blossom_end_rot", "lesion__frogeye_leaf_spot",
                  "lesion__powdery_mildew"],
    "segmentation": ["lesion__bacterial_spot", "lesion__blossom_end_rot", "lesion__frogeye_leaf_spot", "lesion__powdery_mildew"],
}
""", note=r"""
**Chili is excluded** (COLD chili leaves, Roboflow chili growth). It is the same species but a different crop type.

PlantDoc `Bell_pepper leaf spot` is kept as `leaf_spot` and **not** merged with `bacterial_spot`, because it was not checked to be the same disease.

Sweet-pepper maturity classes 1–4 (1 = fully green, 4 = fully coloured) keep their numbers.
""", sections=[
    ("plantvillage", r"""
load_plantvillage({"Pepper,_bell___healthy": "healthy", "Pepper,_bell___Bacterial_spot": "bacterial_spot"})
"""),
    ("plantseg", r"""
BP_MAP = {"bacterial spot": "bacterial_spot", "blossom end rot": "blossom_end_rot", "frogeye leaf spot": "frogeye_leaf_spot",
          "powdery mildew": "powdery_mildew"}
LABEL_SPACE["plantseg"] = [f"lesion__{v}" for v in BP_MAP.values()]
load_plantseg("bell pepper ", BP_MAP)
"""),
    ("plantdoc", r"""
LABEL_SPACE["plantdoc"] = ["leaf__healthy", "leaf__leaf_spot"]
load_plantdoc({"Bell_pepper leaf": "healthy", "Bell_pepper leaf spot": "leaf_spot"})
"""),
    ("sweet_pepper", r"""
## Sweet pepper maturity (Harel et al.): Mendeley `ttntwwxkfd` (CC BY 4.0)

Classification of maturity class 1–4 from several views of the same pepper, using **colour images only** (depth images are skipped). The data comes from a zip in `data/sweet_pepper_maturity/` and is read in place. BMP files are saved as lossless PNG.
- The label belongs to the **pepper**, so `group` = pepper id.
- The *Robotic* and *Orientation* subsets reuse the same physical peppers, so their group ignores the subset.
- File copies with a `(n)` suffix are dropped.
- ⚠️ Lab backgrounds only.
@@
SP_ZIP = zipfile.ZipFile(DATA_DIR / "sweet_pepper_maturity" / "ttntwwxkfd-1.zip")
PHOTOCELL = re.compile(r"Photocell Data/(Red|Yellow)/([RY])_L(\d)_P(\d+)_S(\d)")
COLOR = re.compile(r"(Robotic|Orientation) [Dd]ata/(Red|Yellow)/Color/color_pepper(\d+)\D.*_viewpoint_(\d)_class_(\d)")
SP_MAP = {1: "stage_1_green", 2: "stage_2", 3: "stage_3", 4: "stage_4_coloured"}
show_mapping(SP_MAP)
for name in SP_ZIP.namelist():
    if m := PHOTOCELL.search(name):
        pepper, maturity = f"photocell-{m[2]}L{m[3]}P{m[4]}", int(m[3])
    elif m := COLOR.search(name):
        pepper, maturity = f"robot-{m[2][0]}{m[3]}", int(m[5])
    else:
        continue
    if re.search(r"\(\d\)", name):
        DROPPED[("sweet_pepper_maturity", "file copy with '(n)' suffix")] += 1
        continue
    with Image.open(io.BytesIO(SP_ZIP.read(name))) as img:
        added = add_image(img.convert("RGB"), "sweet_pepper_maturity", name, "CC BY 4.0", pepper)
    if added:
        add_class("maturity", added[0], SP_MAP[maturity])
"""),
])

CUCUMBER = dict(title="cucumber", classes=r"""
CLASSES = {
    "condition": ["healthy_leaf", "healthy_fruit", "anthracnose", "bacterial_wilt", "belly_rot", "downy_mildew", "gummy_stem_blight",
                  "pythium_fruit_rot", "angular_leaf_spot", "powdery_mildew"],
    "maturity": [],
    "detection": ["leaf__abnormal_color", "leaf__damage", "lesion__angular_leaf_spot", "lesion__bacterial_wilt", "lesion__powdery_mildew"],
    "segmentation": ["lesion__angular_leaf_spot", "lesion__bacterial_wilt", "lesion__powdery_mildew"],
}
""", note=r"""
Healthy is split into `healthy_leaf` and `healthy_fruit`, because the cucumber disease set has fruit diseases (belly rot, Pythium fruit rot) and healthy fruit.

**Excluded:**
- AGC 2018 cucumber (`koidra/GH_Cucumber`) has no images.
- The 6,400 augmented copies in the AgML cucumber set.

No source labels cucumber maturity, so `maturity.csv` is empty.
""", sections=[
    ("agml_cucumber", r"""
## Cucumber diseases (Sultana): `Project-AgML/cucumber_disease_classification` (CC BY 4.0, Mendeley)

Field photos of leaf, fruit and stem diseases. ⚠️ AgML mixes 1,280 originals with 6,400 augmented copies and has no flag column. Only originals (`Name(n).jpg`) are used; copies (`Name_(n).jpg`) are dropped. The parquet label has no names, so names come from the card, in card order.
@@
CU_NAMES = ["Anthracnose", "Bacterial_Wilt", "Belly_Rot", "Downy_Mildew", "Fresh_Cucumber", "Fresh_Leaf", "Gummy_Stem_Blight", "Pythium_Fruit_Rot"]
CU_MAP = dict(zip(CU_NAMES, ["anthracnose", "bacterial_wilt", "belly_rot", "downy_mildew", "healthy_fruit", "healthy_leaf",
                             "gummy_stem_blight", "pythium_fruit_rot"]))
show_mapping(CU_MAP)
flat = lambda t: re.sub(r"[^a-z]", "", t.lower())
for row in parquet_rows(parquet_files("Project-AgML/cucumber_disease_classification"), ["image", "label"]):
    path = row["image"]["path"]
    if re.search(r"_\(\d+\)", path):
        DROPPED[("agml_cucumber", "augmented copy (Name_(n).jpg)")] += 1
        continue
    if not flat(path).startswith(flat(CU_NAMES[row["label"]])):
        DROPPED[("agml_cucumber", "label disagrees with the file-name class")] += 1
        continue
    added = add_image(row["image"]["bytes"], "agml_cucumber", path, "CC BY 4.0 (Mendeley)", path)
    if added:
        add_class("condition", added[0], CU_MAP[CU_NAMES[row["label"]]])
"""),
    ("plantseg", r"""
CS_MAP = {"angular leaf spot": "angular_leaf_spot", "bacterial wilt": "bacterial_wilt", "powdery mildew": "powdery_mildew"}
LABEL_SPACE["plantseg"] = [f"lesion__{v}" for v in CS_MAP.values()]
load_plantseg("cucumber ", CS_MAP)
"""),
    ("omarov", r"""
## Cucumber leaf detection (Omarov et al.): Mendeley `mb3vnhpk8z` (CC BY 4.0)

Greenhouse row scenes with YOLO boxes of two **symptom** classes: `color` becomes `leaf__abnormal_color` and `damage` becomes `leaf__damage`.
- The 250 **healthy** scenes are verified negatives: each gets an **empty** detection file and the condition `healthy_leaf`.
- Affected scenes stay out of `condition.csv`, because their labels are symptoms, not diseases.
- Many images are video frames, so `group` = video (`frame1`, `frame`, ...) or photo pair (`..._100435A/B`).
- ⚠️ Some `color` boxes are confounded with a yellow light cast (`plague.ipynb`).
@@
CL_ZIP = zipfile.ZipFile(DATA_DIR / "cucumber_leaf_detection" / "mb3vnhpk8z-1.zip")
cl_files = [n for n in CL_ZIP.namelist() if not n.endswith("/")]
cl_classes = [c.strip() for c in CL_ZIP.read(next(n for n in cl_files if n.endswith("classes.txt"))).decode().splitlines() if c.strip()]
CL_MAP = {"color": "leaf__abnormal_color", "damage": "leaf__damage"}
assert set(cl_classes) == set(CL_MAP), cl_classes
show_mapping(CL_MAP)
LABEL_SPACE["omarov_cucumber_leaf"] = sorted(CL_MAP.values())


def omarov_group(file_name):
    stem = re.sub(r"^[0-9a-f]{8}-", "", file_name.rsplit("/", 1)[-1].rsplit(".", 1)[0])  # drop the annotation tool's hash prefix
    return stem.split("_")[0] if stem.startswith("frame") else stem.rstrip("AB")


for name in cl_files:
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    healthy = "/Healthy leaves/" in name
    label = re.sub(r"/images/([^/]+)\.\w+$", r"/labels/\1.txt", name) if "/images/" in name else None
    if not healthy and label not in cl_files:
        DROPPED[("omarov_cucumber_leaf", "affected image without a label file")] += 1
        continue
    added = add_image(CL_ZIP.read(name), "omarov_cucumber_leaf", name.split("/Dataset/")[-1], "CC BY 4.0", omarov_group(name))
    if not added:
        continue
    image, W, H = added
    if healthy:
        add_boxes(image, [], negative=True)
        add_class("condition", image, "healthy_leaf")
        continue
    boxes = []
    for line in CL_ZIP.read(label).decode().splitlines():
        if line.strip():
            c, x, y, w, h = line.split()[:5]
            x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
            boxes.append((x - w / 2, y - h / 2, w, h, CL_MAP[cl_classes[int(c)]]))
    add_boxes(image, boxes, negative=True)  # the label file exists, so an empty one is a real "no symptom" annotation
"""),
])

CROPS = {"strawberry": STRAWBERRY, "blueberry": BLUEBERRY, "raspberry": RASPBERRY, "tomato": TOMATO,
         "bell_pepper": BELL_PEPPER, "cucumber": CUCUMBER}


def build(crop, spec):
    cells = [nbf.v4.new_markdown_cell(INTRO.format(title=spec["title"], crop=crop).strip()),
             nbf.v4.new_code_cell(HELPERS.replace("__CROP__", crop).strip()),
             nbf.v4.new_markdown_cell("### Helpers\n\n`add_image`, `add_class`, `add_boxes` and `add_polygon` are the only functions that write labels. They apply rules 5–7 above and record what they drop."),
             nbf.v4.new_code_cell(HELPERS_2.strip()),
             nbf.v4.new_markdown_cell(VOCAB_MD.strip() + "\n\n" + spec["note"].strip()),
             nbf.v4.new_code_cell(spec["classes"].strip() + "\n" + SPLIT_PRINT)]
    defined = set()
    for loader, body in spec["sections"]:
        text, loader_code = LOADERS.get(loader, (None, None))
        if "@@" in body:
            text, body = body.split("@@")
        if text:
            cells.append(nbf.v4.new_markdown_cell(text.strip()))
        source = body.strip()
        if loader_code and loader not in defined:
            source = loader_code.strip() + "\n\n\n" + source
            defined.add(loader)
        cells.append(nbf.v4.new_code_cell(source))
    cells += [nbf.v4.new_markdown_cell("## Write, de-duplicate, self-check\n\n`finalize` writes every file. `check` reads them back from disk and asserts the format: every label file has an image, every coordinate is in [0, 1], and every class id is valid."),
              nbf.v4.new_code_cell(FINAL.strip()),
              nbf.v4.new_code_cell("manifest = finalize()\ncheck()\nsummary(manifest)\nprint()\nclass_counts()"),
              nbf.v4.new_markdown_cell("## Round trip: labels read back from disk\n\nFor each source, sample images are drawn with the boxes and polygons **parsed from the written files**, so a bad conversion (swapped axes, wrong frame, wrong scale) is visible. Titles show the classification labels and the shape classes. The figures are saved to `examples/normalized/`."),
              nbf.v4.new_code_cell(ROUNDTRIP.strip()),
              nbf.v4.new_code_cell("for source in manifest[\"source\"].unique():\n    show_roundtrip(source)")]
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    nbf.write(nb, f"normalization/normalized_{crop}.ipynb")


os.makedirs("normalization", exist_ok=True)
for crop, spec in CROPS.items():
    build(crop, spec)
print("wrote", ", ".join(f"normalization/normalized_{c}.ipynb" for c in CROPS))
