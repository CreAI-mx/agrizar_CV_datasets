"""Source loaders shared by several crops. Each one reads a public dataset and writes through `cvlib.normalize`."""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, snapshot_download
from PIL import Image

from cvlib.normalize import (DATA_DIR, DROPPED, LABEL_SPACE, NOTES, add_boxes, add_class, add_image, add_polygon, card_class_names,
                             condition_from_boxes, load_coco, parquet_files, parquet_rows, rf_stem, show_mapping)

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
