"""Normalization engine shared by the master notebook (`normalization/normalize_all.ipynb`).

State lives at module level and is reset by `begin(crop, classes)`; every crop section of the notebook calls it first.
Run from the repo root with `tools/` on `sys.path`. Rules: see the notebook intro and docs/CONTEXT.md.
"""
import gc
import io
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageOps

from cvlib.common import card_class_names, parquet_files, rf_stem  # noqa: F401  (re-exported for the notebook)

DATA_DIR = Path("data")
EXAMPLE_DIR = Path("examples") / "normalized"
MATURITY_STATES = {"green", "white", "turning", "ripe", "overripe", "unripe", "half_ripe", "breaker", "reddish"}

# per-crop state: mutated in place by begin() so `from cvlib.normalize import IMAGES, ...` stays valid
IMAGES = {}                                   # image name -> manifest row
CLASSIFICATION = {"condition": {}, "maturity": {}}
CONFLICTS = {"condition": set(), "maturity": set()}
SHAPES = {(t, a): {} for t in ("detection", "segmentation") for a in ("condition", "maturity")}  # (task, axis) -> image -> lines
LABEL_SPACE = {}                              # source -> detection classes the source annotates
DROPPED = Counter()                           # (source, reason) -> count
NOTES = Counter()                             # (source, note) -> count, things kept but worth knowing
SERIAL = Counter()
CROP = OUT = CLASSES = RNG = None             # rebound by begin()


def begin(crop, classes):
    """Start a crop from scratch: wipe `normalized/<crop>/` and reset all state."""
    global CROP, OUT, CLASSES, RNG
    gc.collect()
    CROP, CLASSES, OUT = crop, classes, Path("normalized") / crop
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    RNG = np.random.default_rng(0)
    shutil.rmtree(OUT, ignore_errors=True)  # a run always rebuilds its crop from scratch
    for sub in ("images", "labels/classification", *(f"labels/{t}/{a}" for t in ("detection", "segmentation") for a in ("condition", "maturity"))):
        (OUT / sub).mkdir(parents=True)
    for state in (IMAGES, LABEL_SPACE, DROPPED, NOTES, SERIAL, *CLASSIFICATION.values(), *CONFLICTS.values(), *SHAPES.values()):
        state.clear()


def axis_of(cls):
    # plant-stage objects go to the maturity model; everything else (healthy, lesions, affected organs, diseased fruit) to condition
    return "maturity" if cls in ("flower", "bud") or (cls.startswith("fruit__") and cls.split("__")[1] in MATURITY_STATES) else "condition"


def axis_classes(task, axis):
    return [c for c in CLASSES[task] if axis_of(c) == axis]


def has_label(name):
    return any(name in d for d in (*CLASSIFICATION.values(), *SHAPES.values()))


def print_split():
    for task, axis in SHAPES:
        print(f"{task}/{axis}: {axis_classes(task, axis)}")

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
