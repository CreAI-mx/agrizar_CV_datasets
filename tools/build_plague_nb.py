"""Builds plague.ipynb (replaces lol.ipynb). Run from the repo root, then execute with nbconvert (see CLAUDE.md)."""
import nbformat as nbf

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------- intro
md(r"""
# Plant disease and pest ("plague") datasets

**Purpose:** survey public disease/pest/deficiency datasets (one section per dataset). **Inputs:** Hugging Face cache and `data/` (see CLAUDE.md). **Outputs:** `charts/plague_*.png`, `examples/plague/`, findings in `PLAGUE_DATASET_FINDINGS.md`. **Run:** `make plague`. **Owner:** CreAI (GRZR-16).

Datasets for detecting **plant diseases, pests and nutrient deficiencies**, both **general (all crops)** and with emphasis on **strawberry, blueberry and raspberry**. Findings are summarized in `PLAGUE_DATASET_FINDINGS.md`. The companion study on plant **growth** is `growth.ipynb` / `GROWTH_DATASET_FINDINGS.md`.

| # | Dataset | Role | Task | Images | License |
|---|---|---|---|---|---|
| 1 | PlantVillage `BrandonFors/Plant-Diseases-PlantVillage-Dataset` | general disease, **lab reference** | classification, 38 classes | 54,305 | CC0 (metadata) |
| 2 | LeafNet2.0 `enalis/LeafNet2.0` (gated) | general disease, **largest** | classification, 182 classes | 255,990 | non-commercial gate |
| 3 | PlantSeg v3 `XuVV/plantsegv3` | general disease **in the wild, with lesion masks**; berry diseases | segmentation / detection / classification, 114 diseases | 11,458 | CC BY-NC-ND 4.0 |
| 4 | PlantDoc `susnato/plant_disease_detection_processed` | general disease **detection** (commercial-friendly) | detection, 24 classes | 2,324 | CC BY 4.0 |
| 5 | IP102 `Project-AgML/CT_PestNet_crop_pest_classification` | general **insect pests** | classification, 102 pests | 75,222 | CC BY 4.0 (AgML) |
| 6 | CCMT `Project-AgML/crop_pest_disease_classification` | **field** pest damage + disease | classification, 18 classes, 4 crops | 25,170 (raw) | CC BY 4.0 |
| 7 | Strawberry_12 `Ethan0300/Strawberry_12-dataset` | **strawberry** disease, pests, deficiencies | detection (YOLO), 12 classes | 3,906 | MIT |
| 8 | Tomato leaf disease detection `Project-AgML/tomato_leaf_disease_detection` | **tomato** field leaf diseases | detection, 7 classes (names missing) | 689 | CC BY 4.0 |
| 9 | Tomato disease detection `Project-AgML/tomato_disease_detection` | **tomato** disease on fruit, leaf, stem | detection, 10 classes | 1,026 | CC BY 4.0 |
| 10 | Chili leaves (COLD) `Project-AgML/COLD_chili_leaf_disease_classification` | **chili** disease, pest, deficiency | classification, 5 classes | 532 (raw) | CC BY 4.0 |
| 11 | Cucumber diseases `Project-AgML/cucumber_disease_classification` | **cucumber** leaf, fruit, stem diseases | classification, 8 classes | 1,280 originals (+6,400 augmented) | CC BY 4.0 |
| 12 | Cucumber leaf detection, Mendeley `mb3vnhpk8z` | **cucumber** leaf symptoms | detection, 2 classes | 500 | CC BY 4.0 |

**Discarded** (kept in `archive/` and in the findings file): PlantWild (`EGOISTyrh/mvpdr-plant-disease`), because PlantSeg is the same group's newer, larger set with masks (overlap checked at the end); and the Roboflow strawberry-disease set (`luffywang/...`), which cannot be downloaded (workspace quota), while Strawberry_12 covers the same diseases. Also discarded (checked in the tomato/pepper part): `Project-AgML/tomato_leaf_disease`, a re-upload of PlantVillage's tomato classes, and `Project-AgML/pepper_disease_pest_classification`, whose images are **apple** leaves, not pepper.

**Colors in every chart** (validated categorical set): blue = healthy, orange = disease, aqua = pest, yellow = nutrient deficiency, gray = plant part (fruit/flower, not a condition). Every bar has a direct value label (aqua and yellow are below 3:1 contrast, so labels are required). Berry crops are listed first and in **bold**.
""")

code(r"""
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Image as HFImage
from datasets import load_dataset
from huggingface_hub import HfApi, HfFileSystem, snapshot_download
from matplotlib.patches import Patch
from PIL import Image, ImageDraw

CHART_DIR = Path("charts")
CHART_DIR.mkdir(exist_ok=True)
EXAMPLE_DIR = Path("examples") / "plague"
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")
RNG = np.random.default_rng(0)

KIND_COLORS = {"healthy": "#2a78d6", "disease": "#eb6834", "pest": "#1baf7a", "deficiency": "#eda100", "plant part": "#898781"}
KINDS = list(KIND_COLORS)
INK = "#0b0b0b"
MUTED_INK = "#52514e"
GRID_COLOR = "#e1e0d9"
SURFACE = "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED_INK, "xtick.color": "#898781",
                     "ytick.color": MUTED_INK, "font.size": 9})

BERRIES = ("strawberry", "blueberry", "raspberry")
SUMMARY = {}       # dataset -> summary row
BERRY_COUNTS = {}  # dataset -> Counter((berry, condition) -> examples)
KIND_OF = {}       # (berry, condition) -> kind, for coloring the berry charts
SIZES = {}         # dataset -> list of image widths
""")

md(r"""
### Helpers

Each dataset gets a `split_name(class) -> (crop, condition, kind)` function, where `kind` is one of healthy / disease / pest / deficiency / plant part.

- `plot_crops`: one bar per crop, stacked by kind. The number at the end is the crop total. Berries first and bold.
- `plot_berry_classes`: one bar per berry class.
- `show_samples`: example images with labels (and boxes/polygons for detection or segmentation sets). Saved to `examples/plague/`.
- `image_sizes` / `scan_parquet`: read only the image **header**, never the full image.
""")

code(r"""
def is_berry(crop):
    return crop.lower() in BERRIES


def legend(fig, kinds):
    fig.legend(handles=[Patch(color=KIND_COLORS[k], label=k) for k in KINDS if k in kinds],
               loc="outside upper right", ncol=len(kinds), frameon=False)


def plot_crops(counts, split_name, title, filename, unit="images"):
    crops = defaultdict(Counter)
    for name, n in counts.items():
        crop, _, kind = split_name(name)
        crops[crop][kind] += n
    names = sorted(crops, key=lambda c: (not is_berry(c), c.lower()))
    used = {k for c in crops.values() for k in c}
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(names) + 1.5), layout="constrained")
    for row, crop in enumerate(names):
        left = 0
        for kind in KINDS:
            n = crops[crop].get(kind, 0)
            if n:
                ax.barh(row, n, left=left, color=KIND_COLORS[kind], height=0.72, edgecolor=SURFACE, linewidth=1.5)
                left += n
        ax.text(left, row, f" {left:,}", va="center", fontsize=7, color=MUTED_INK)
    ax.set_yticks(range(len(names)), names)
    for label, crop in zip(ax.get_yticklabels(), names):
        if is_berry(crop):
            label.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(sum(c.values()) for c in crops.values()) * 1.15)
    ax.set_xlabel(unit)
    ax.set_title(title, loc="left", fontweight="bold", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    legend(fig, used)
    fig.savefig(CHART_DIR / filename, dpi=150)
    plt.show()


def berry_counts(counts, split_name):
    out = Counter()
    for name, n in counts.items():
        crop, condition, kind = split_name(name)
        if is_berry(crop) and kind != "plant part":
            key = (crop.lower(), condition.lower().replace("_", " "))
            out[key] += n
            KIND_OF[key] = kind
    return out


def plot_bars(items, title, unit="images", filename=None):
    # items: list of (label, value, kind)
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(items) + 1.2), layout="constrained")
    ax.barh(range(len(items)), [v for _, v, _ in items], height=0.65, edgecolor=SURFACE, linewidth=1.5,
            color=[KIND_COLORS[k] for _, _, k in items])
    for row, (_, v, _) in enumerate(items):
        ax.text(v, row, f" {v:,}", va="center", fontsize=8, color=MUTED_INK)
    ax.set_yticks(range(len(items)), [label for label, _, _ in items])
    ax.invert_yaxis()
    ax.set_xlim(0, max(v for _, v, _ in items) * 1.18)
    ax.set_xlabel(unit)
    ax.set_title(title, loc="left", fontweight="bold", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    legend(fig, {k for _, _, k in items})
    if filename:
        fig.savefig(CHART_DIR / filename, dpi=150)
    plt.show()


def plot_berry_classes(berry, title, unit="images"):
    items = sorted(berry.items(), key=lambda kv: (kv[0][0], KINDS.index(KIND_OF[kv[0]]), kv[0][1]))
    plot_bars([(f"{crop} · {cond}", n, KIND_OF[(crop, cond)]) for (crop, cond), n in items], title, unit)


def save_example(fig, title):
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60].strip("_")
    fig.savefig(EXAMPLE_DIR / f"{slug}.png", dpi=120)


def show_samples(samples, title, cols=3, thumb=420, color_of=lambda name: KIND_COLORS["disease"]):
    # samples: list of (PIL image, shapes, caption); a shape is (x, y, w, h, name) or ("poly", [x0, y0, x1, y1, ...], name)
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.7 * rows), layout="constrained")
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (img, shapes, caption) in zip(np.ravel(axes), samples):
        img = img.convert("RGB")
        scale = thumb / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        draw = ImageDraw.Draw(img)
        for shape in shapes:
            if shape[0] == "poly":
                pts = [v * scale for v in shape[1]]
                if len(pts) >= 6:
                    draw.polygon(pts, outline=color_of(shape[2]), width=3)
            else:
                x, y, w, h, name = shape
                draw.rectangle([x * scale, y * scale, (x + w) * scale, (y + h) * scale], outline=color_of(name), width=3)
        ax.imshow(img)
        ax.set_title(caption, fontsize=8, color=MUTED_INK)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", color=INK)
    save_example(fig, title)
    plt.show()


def image_sizes(dataset):
    out = []
    for row in dataset.select_columns(["image"]).cast_column("image", HFImage(decode=False)):  # row by row: low memory
        item = row["image"]
        with Image.open(io.BytesIO(item["bytes"]) if item["bytes"] else item["path"]) as img:
            out.append((img.width, img.height, img.mode))
    return out


def scan_parquet(files, columns):
    # one dict per row with the requested columns + width/height/mode read from the image header (bytes are not kept)
    rows = []
    for path in files:
        pf = pq.ParquetFile(path)
        for group in range(pf.num_row_groups):
            for row in pf.read_row_group(group, columns=["image", *columns]).to_pylist():
                image = row.pop("image")
                with Image.open(io.BytesIO(image["bytes"])) as img:
                    row.update(width=img.width, height=img.height, mode=img.mode, path=image["path"])
                rows.append(row)
    return rows


def parquet_images(files, wanted):
    wanted, found, index = set(wanted), {}, 0
    for path in files:
        pf = pq.ParquetFile(path)
        for group in range(pf.num_row_groups):
            for item in pf.read_row_group(group, columns=["image"]).column("image").to_pylist():
                if index in wanted:
                    found[index] = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
                index += 1
            if len(found) == len(wanted):
                return found
    return found


def describe_sizes(sizes):
    widths, heights = [w for w, _, _ in sizes], [h for _, h, _ in sizes]
    print(f"{len(sizes):,} images, {len(set((w, h) for w, h, _ in sizes)):,} distinct sizes; "
          f"most common {Counter((w, h) for w, h, _ in sizes).most_common(3)}")
    print(f"width min / median / max {min(widths)} / {int(np.median(widths))} / {max(widths)}; "
          f"height min / median / max {min(heights)} / {int(np.median(heights))} / {max(heights)}")
    if sizes[0][2] is not None:
        print("modes:", dict(Counter(m for _, _, m in sizes)))


def pick_samples(labels, names, split_name, n_berry=3, n_other=3):
    berry_rows, other_rows, seen = [], [], set()
    for i in RNG.permutation(len(labels)):
        name = names[labels[i]]
        crop = split_name(name)[0]
        if is_berry(crop) and name not in seen and len(berry_rows) < n_berry:
            berry_rows.append(int(i))
            seen.add(name)
        elif not is_berry(crop) and len(other_rows) < n_other:
            other_rows.append(int(i))
    return berry_rows + other_rows


def split_triple_underscore(name):
    crop, _, condition = name.partition("___")
    healthy = condition.lower() == "healthy"
    return crop.replace("_", " "), "healthy" if healthy else condition.replace("_", " "), "healthy" if healthy else "disease"


def ahash(img, n=16):
    a = np.asarray(img.convert("L").resize((n, n), Image.BILINEAR), dtype=float)
    return (a > a.mean()).ravel()


def matches(hashes_a, hashes_b, max_bits=10):
    # how many images of A have a near-identical image in B (perceptual hash, 256 bits)
    best = (np.array(hashes_a)[:, None, :] != np.array(hashes_b)[None, :, :]).sum(-1).min(1)
    return int((best <= max_bits).sum())
""")

# ---------------------------------------------------------------- PlantVillage
md(r"""
## 1. PlantVillage: `BrandonFors/Plant-Diseases-PlantVillage-Dataset`

Mirror of PlantVillage (Mohanty et al. 2016). **Lab photos of single leaves on plain backgrounds.** License `cc0-1.0` in metadata (the card body says "More Information Needed"). Public.

**Role:** clean reference images. Models trained only on it generalize poorly to field photos. Classification, 38 classes `Crop___condition` across 14 crops; train 43,456 / test 10,849.
""")

code(r"""
pv = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset")
pv_names = pv["train"].features["label"].names  # ClassLabel: use .names, the feature itself is not iterable
pv_labels = {split: pv[split]["label"] for split in pv}
print({split: len(v) for split, v in pv_labels.items()}, f"| {len(pv_names)} classes")
pv_counts = Counter(pv_names[i] for v in pv_labels.values() for i in v)
plot_crops(pv_counts, split_triple_underscore, f"PlantVillage: images per crop ({sum(pv_counts.values()):,})", "plague_plantvillage_crops.png")
BERRY_COUNTS["PlantVillage"] = berry_counts(pv_counts, split_triple_underscore)
plot_berry_classes(BERRY_COUNTS["PlantVillage"], "PlantVillage: berry classes")
""")

code(r"""
pv_sizes = image_sizes(pv["train"]) + image_sizes(pv["test"])
describe_sizes(pv_sizes)
SIZES["PlantVillage"] = [w for w, _, _ in pv_sizes]
rows = pick_samples(pv_labels["train"], pv_names, split_triple_underscore)
show_samples([(pv["train"][i]["image"], [], pv_names[pv_labels["train"][i]]) for i in rows], "PlantVillage: samples")
SUMMARY["PlantVillage"] = {"role": "lab reference", "kind": "disease", "task": "classification", "images": len(pv_sizes),
                           "classes": len(pv_names), "license": "CC0 (metadata)", "setting": "lab, plain background"}
""")

md(r"""
**Gotchas:** every image is 256×256 RGB. Orange and Squash have only a disease class; Blueberry, Raspberry and Soybean only a healthy class.
""")

# ---------------------------------------------------------------- LeafNet
md(r"""
## 2. LeafNet2.0: `enalis/LeafNet2.0`

⚠️ **Gated:** HF login + a form agreeing to **non-commercial use only** (auto-approved). No license declared. Actively updated.

**Role:** the largest general disease set: **182 classes**, 37 crops, 255,990 images (`images` config; `image_caption_unified` adds a description per image). Only a `train` split. ⚠️ It **contains PlantVillage** (check at the end), so de-duplicate before combining the two.
""")

code(r"""
ln = load_dataset("enalis/LeafNet2.0", "images")["train"]  # needs HF login + accepted gate
ln_names = ln.features["label"].names
ln_labels = ln["label"]
ln_counts = Counter(ln_names[i] for i in ln_labels)
print(f"{len(ln_labels):,} images, {len(ln_names)} classes")
small = sorted((n, c) for c, n in ln_counts.items() if n < 100)
print(f"classes with < 100 images: {len(small)}; smallest: {small[:4]}; largest: {ln_counts.most_common(2)}")
plot_crops(ln_counts, split_triple_underscore, f"LeafNet2.0: images per crop ({len(ln_labels):,})", "plague_leafnet_crops.png")
BERRY_COUNTS["LeafNet2.0"] = berry_counts(ln_counts, split_triple_underscore)
plot_berry_classes(BERRY_COUNTS["LeafNet2.0"], "LeafNet2.0: berry classes")
""")

code(r"""
ln_sizes = image_sizes(ln)
describe_sizes(ln_sizes)
SIZES["LeafNet2.0"] = [w for w, _, _ in ln_sizes]
rows = pick_samples(ln_labels, ln_names, split_triple_underscore)
show_samples([(ln[i]["image"], [], ln_names[ln_labels[i]]) for i in rows], "LeafNet2.0: samples")
SUMMARY["LeafNet2.0"] = {"role": "largest general disease set", "kind": "disease", "task": "classification (+captions)",
                         "images": len(ln_sizes), "classes": len(ln_names), "license": "none; gate = non-commercial",
                         "setting": "mostly close-up leaves"}
""")

md(r"""
**Gotchas:** apply `img.convert("RGB")` (RGBA PNGs, CMYK and grayscale among the 224×224 JPEGs). `captions_full` has no `label` column (use `folder_name`) and 179 vs 182 classes.
""")

# ---------------------------------------------------------------- PlantSeg
md(r"""
## 3. PlantSeg v3: `XuVV/plantsegv3`

Wei et al., *PlantSeg: A Large-Scale In-the-wild Dataset for Plant Disease Segmentation* (arXiv:2409.04038; University of Queensland / CSIRO; Zenodo v3 doi:10.5281/zenodo.14935094). ⚠️ **CC BY-NC-ND 4.0** on the original; this HF mirror declares no license. Same group as PlantWild, which it replaces here.

**Role:** in-the-wild disease images with **pixel masks of the lesions** (semantic mask + instance polygons + boxes), so it supports segmentation, detection and **severity** (lesion share of the image). 114 disease classes, **no healthy classes**. Splits train 7,916 / validation 1,247 / test 2,295. **Only source of raspberry diseases** (4) and the richest for blueberry (5).
""")

code(r"""
PS_FILES = sorted((Path(snapshot_download("XuVV/plantsegv3", repo_type="dataset")) / "data").glob("*.parquet"))
ps = []
for path in PS_FILES:
    for row in pq.read_table(path, columns=["file_name", "width", "height", "answer", "bboxes", "areas"]).to_pylist():
        row["split"] = path.name.split("-")[0]
        ps.append(row)
print(f"{len(ps):,} images, {len({r['answer'] for r in ps})} classes; per split {dict(Counter(r['split'] for r in ps))}")


def split_plantseg(name):
    if name.startswith("bell pepper "):
        crop, condition = "bell pepper", name[len("bell pepper "):]
    else:
        first, condition = name.split(" ", 1)
        crop = {"grapevine": "grape"}.get(first, first)
    return crop.capitalize(), condition, "disease"


ps_counts = Counter(r["answer"] for r in ps)
plot_crops(ps_counts, split_plantseg, f"PlantSeg v3: images per crop ({len(ps):,}, diseases only)", "plague_plantseg_crops.png")
BERRY_COUNTS["PlantSeg v3"] = berry_counts(ps_counts, split_plantseg)
plot_berry_classes(BERRY_COUNTS["PlantSeg v3"], "PlantSeg v3: berry diseases")
print(f"images per class: min {min(ps_counts.values())}, median {int(np.median(list(ps_counts.values())))}, max {max(ps_counts.values())}")
""")

code(r"""
lesion_share = np.array([min(1.0, sum(r["areas"]) / (r["width"] * r["height"])) for r in ps])
print(f"lesion area as share of the image: median {np.median(lesion_share):.1%}, p10 {np.percentile(lesion_share, 10):.1%}, "
      f"p90 {np.percentile(lesion_share, 90):.1%}; lesion instances per image median {int(np.median([len(r['bboxes']) for r in ps]))}")
web = Counter(next((s for s in ("google", "bing", "baidu") if s in r["file_name"].lower()), "other name") for r in ps)
print("file-name source tags:", dict(web), "(web-scraped images)")
describe_sizes([(r["width"], r["height"], None) for r in ps])
SIZES["PlantSeg v3"] = [r["width"] for r in ps]

# samples: one per berry crop + 3 others, polygons drawn from the instance segmentations
wanted = {}
for i in RNG.permutation(len(ps)):
    crop = split_plantseg(ps[i]["answer"])[0].lower()
    key = crop if crop in BERRIES else "other"
    if (key != "other" and key not in wanted.values()) or (key == "other" and list(wanted.values()).count("other") < 3):
        wanted[int(i)] = key
    if len(wanted) == 6:
        break
found, index = {}, 0
for path in PS_FILES:
    pf = pq.ParquetFile(path)
    for group in range(pf.num_row_groups):
        for item in pf.read_row_group(group, columns=["image", "segmentations"]).to_pylist():
            if index in wanted:
                found[index] = item
            index += 1
order = sorted(wanted, key=lambda i: wanted[i] == "other")
show_samples([(Image.open(io.BytesIO(found[i]["image"]["bytes"])),
               [("poly", poly, ps[i]["answer"]) for inst in found[i]["segmentations"] for poly in inst],
               ps[i]["answer"]) for i in order], "PlantSeg v3: samples (lesion polygons)")
SUMMARY["PlantSeg v3"] = {"role": "in-the-wild disease + lesion masks", "kind": "disease", "task": "segmentation / detection",
                          "images": len(ps), "classes": len(ps_counts), "license": "CC BY-NC-ND 4.0 (original)",
                          "setting": "in the wild, web images"}
""")

md(r"""
**Gotchas:** no healthy class, so pair it with healthy images from elsewhere for a "healthy vs diseased" model. Images are web-scraped (`_google_`, `_bing_` in file names), so image rights are not cleared by the license. 18–58 images per berry disease: enough to evaluate, thin for training.
""")

# ---------------------------------------------------------------- PlantDoc
md(r"""
## 4. PlantDoc detection: `susnato/plant_disease_detection_processed`

Processed copy of the PlantDoc object-detection dataset (Singh et al., CoDS-COMAD 2020). **CC BY 4.0.** Preprocessed for DETR (5.3 GB download, about 30 GB in memory), so only the light columns are read here.

**Role:** the only **commercial-friendly in-the-wild detection** set for general diseases (boxes around leaves). ⚠️ **Class names are not in the dataset**; they were recovered by matching every box against the original PlantDoc CSVs (8,541 of 8,544 boxes). 5 of the original 29 classes are missing.
""")

code(r"""
PD_REPO = "susnato/plant_disease_detection_processed"
hf_fs = HfFileSystem()


def read_columns(split, columns=("image_id", "width", "height", "objects")):
    tables = []
    for path in sorted(hf_fs.glob(f"datasets/{PD_REPO}/data/{split}-*.parquet")):
        with hf_fs.open(path, "rb", block_size=2**20) as fh:
            tables.append(pq.read_table(fh, columns=list(columns)))
    return pa.concat_tables(tables).to_pandas()


PD_NAMES = [
    "Cherry leaf", "Peach leaf", "Corn leaf blight", "Apple rust leaf", "Potato leaf late blight",
    "Strawberry leaf", "Tomato leaf late blight", "Tomato mold leaf", "Potato leaf early blight", "Apple leaf",
    "Tomato leaf yellow virus", "Blueberry leaf", "Tomato leaf mosaic virus", "Raspberry leaf", "Tomato leaf bacterial spot",
    "Squash Powdery mildew leaf", "grape leaf", "Tomato Early blight leaf", "Apple Scab Leaf", "Tomato Septoria leaf spot",
    "Tomato leaf", "Soyabean leaf", "Bell_pepper leaf spot", "Bell_pepper leaf",
]


def split_plantdoc(name):
    first, rest = name.split(" ", 1)
    crop = {"bell_pepper": "bell pepper", "soyabean": "soybean"}.get(first.lower(), first.lower()).capitalize()
    rest = rest.lower()
    if rest == "leaf":
        return crop, "healthy", "healthy"
    return crop, rest.removeprefix("leaf ").removesuffix(" leaf"), "disease"


pdd = {split: read_columns(split) for split in ("train", "test")}
for split, df in pdd.items():
    print(f"{split:<6}{len(df):>6,} images{sum(len(o['category']) for o in df['objects']):>7,} boxes")
pd_counts = Counter(PD_NAMES[int(c)] for df in pdd.values() for o in df["objects"] for c in o["category"])
plot_crops(pd_counts, split_plantdoc, f"PlantDoc: boxes per crop ({sum(pd_counts.values()):,} boxes)", "plague_plantdoc_crops.png", unit="boxes")
BERRY_COUNTS["PlantDoc"] = berry_counts(pd_counts, split_plantdoc)
""")

code(r"""
pd_all = pd.concat(pdd.values())
SIZES["PlantDoc"] = list(pd_all["width"])
describe_sizes(list(zip(pd_all["width"], pd_all["height"], [None] * len(pd_all))))
per_image = pd_all["objects"].map(lambda o: len(o["category"]))
print(f"boxes per image min / median / max: {per_image.min()} / {int(per_image.median())} / {per_image.max()}")

samples, berry_seen, other = [], set(), 0
for path in sorted(hf_fs.glob(f"datasets/{PD_REPO}/data/train-*.parquet"))[:6]:
    with hf_fs.open(path, "rb", block_size=2**20) as fh:
        table = pq.read_table(fh, columns=["image", "objects"]).to_pylist()
    for row in table:
        names = [PD_NAMES[int(c)] for c in row["objects"]["category"]]
        berry = next((n for n in names if is_berry(split_plantdoc(n)[0])), None)
        if (berry and berry not in berry_seen) or (not berry and other < 3):
            if berry:
                berry_seen.add(berry)
            else:
                other += 1
            samples.append((Image.open(io.BytesIO(row["image"]["bytes"])), [(*b, n) for b, n in zip(row["objects"]["bbox"], names)],
                            ", ".join(sorted(set(names)))[:50]))
    if len(berry_seen) == 3 and other >= 3:
        break
samples = sorted(samples, key=lambda s: not any(b in s[2].lower() for b in BERRIES))[:6]
show_samples(samples, "PlantDoc: samples", color_of=lambda n: KIND_COLORS[split_plantdoc(n)[2]])
SUMMARY["PlantDoc"] = {"role": "in-the-wild disease detection", "kind": "disease", "task": "object detection",
                       "images": len(pd_all), "classes": 24, "license": "CC BY 4.0", "setting": "in the wild, web images"}
""")

md(r"""
**Gotchas:** berry classes are **healthy leaves only**. Many images are web stock photos (white backgrounds, watermarks such as featurePics). Missing vs the original: Corn gray leaf spot, Corn rust, Grape black rot, Potato leaf, Tomato spider mites (use the original GitHub repo if needed).
""")

# ---------------------------------------------------------------- IP102
md(r"""
## 5. IP102 (insect pests): `Project-AgML/CT_PestNet_crop_pest_classification`

AgML calls it "CT PestNet"; its 75,222 images and 102 classes (in the same order) match **IP102** (Wu et al., *IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition*, CVPR 2019), the standard pest benchmark (*match by numbers and class list; the card does not cite IP102*). **CC BY 4.0** on AgML.

**Role:** general **insect pest** recognition (classification), 102 species across 8 crop groups: field crops (rice, corn, wheat, beet, alfalfa) and economic crops (grape, citrus, mango). Several classes are also **key strawberry/berry pests**: spider mites, thrips, aphids, whitefly, plant bugs, armyworms.
""")

code(r"""
IP_REPO = "Project-AgML/CT_PestNet_crop_pest_classification"
IP_FILES = sorted((Path(snapshot_download(IP_REPO, repo_type="dataset")) / "data").glob("*.parquet"))
ip_names = json.loads(pq.ParquetFile(IP_FILES[0]).schema_arrow.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
ip = scan_parquet(IP_FILES, ["label", "split"])
print(f"{len(ip):,} images, {len(ip_names)} classes; per split {dict(Counter(r['split'] for r in ip))}")

# IP102 class order is grouped by host crop (rice 14, corn 13, wheat 9, beet 8, alfalfa 13, grape 16, citrus 19, mango 10)
IP_GROUPS = [("rice", 14), ("corn", 13), ("wheat", 9), ("beet", 8), ("alfalfa", 13), ("grape (vitis)", 16), ("citrus", 19), ("mango", 10)]
ip_group = [g for g, n in IP_GROUPS for _ in range(n)]
ip_counts = Counter(ip_names[r["label"]] for r in ip)
plot_crops(ip_counts, lambda name: (ip_group[ip_names.index(name)], name, "pest"),
           f"IP102: images per host crop ({len(ip):,})", "plague_ip102_crops.png")
print(f"images per class: min {min(ip_counts.values())}, median {int(np.median(list(ip_counts.values())))}, max {max(ip_counts.values())}")
""")

code(r"""
BERRY_PESTS = {  # IP102 classes that are also major strawberry / berry pests
    "red spider": "two-spotted spider mite", "longlegged spider mite": "spider mite", "Polyphagotars onemus latus": "broad mite",
    "Thrips": "thrips", "Scirtothrips dorsalis Hood": "chilli thrips", "aphids": "aphids",
    "Trialeurodes vaporariorum": "greenhouse whitefly", "tarnished plant bug": "tarnished plant bug (Lygus)",
    "Apolygus lucorum": "plant bug", "army worm": "armyworm", "beet army worm": "beet armyworm", "Prodenia litura": "Spodoptera litura",
}
plot_bars([(f"{name}  ({common})", ip_counts[name], "pest") for name, common in BERRY_PESTS.items()],
          "IP102: classes that are also strawberry/berry pests", filename="plague_ip102_berry_pests.png")
BERRY_COUNTS["IP102 (pests)"] = Counter({("strawberry", f"pest: {common}"): ip_counts[name] for name, common in BERRY_PESTS.items()})
KIND_OF.update({k: "pest" for k in BERRY_COUNTS["IP102 (pests)"]})

ip_sizes = [(r["width"], r["height"], r["mode"]) for r in ip]
describe_sizes(ip_sizes)
SIZES["IP102 (pests)"] = [w for w, _, _ in ip_sizes]
pick = [next(i for i in RNG.permutation(len(ip)) if ip_names[ip[i]["label"]] == n) for n in ("red spider", "Thrips", "aphids", "Trialeurodes vaporariorum")]
pick += [int(i) for i in RNG.choice(len(ip), 2, replace=False)]
imgs = parquet_images(IP_FILES, pick)
show_samples([(imgs[i], [], ip_names[ip[i]["label"]]) for i in pick], "IP102: samples", color_of=lambda n: KIND_COLORS["pest"])
SUMMARY["IP102 (pests)"] = {"role": "general insect pests", "kind": "pest", "task": "classification", "images": len(ip),
                            "classes": len(ip_names), "license": "CC BY 4.0 (AgML)", "setting": "web + field photos, all life stages"}
""")

md(r"""
**Gotchas:** classification only (the original IP102 also has ~19k images with boxes; not in this mirror). Images include eggs, larvae, pupae and adults, plus illustrations and **watermarked web/stock photos** (gettyimages, 36TV.cn, ...), and the classes are long-tailed (71 to 5,740 images per class). Insects are photographed on any background, not necessarily on the crop. For detection of mites/thrips on strawberry leaves it is a **pretraining / recognition** source, not a field detector.
""")

# ---------------------------------------------------------------- CCMT
md(r"""
## 6. CCMT crop pest and disease: `Project-AgML/crop_pest_disease_classification`

Mensah et al., *CCMT: Dataset for crop pest and disease detection*, Data in Brief 49 (2023); Mendeley doi:10.17632/bwh3zbpkpv.1. Photos taken in **farms in Ghana**. **CC BY 4.0.**

**Role:** **field** images of **pest damage and diseases** on cashew, cassava, maize and tomato, 18 classes including healthy. The `raw` config (25,170 images) is used here; the `augmented` config (105,252) is 4× generated copies and is skipped.
""")

code(r"""
CC_REPO = "Project-AgML/crop_pest_disease_classification"
CC_FILES = sorted((Path(snapshot_download(CC_REPO, repo_type="dataset", allow_patterns=["raw/*", "README.md"])) / "raw").glob("*.parquet"))
cc_names = json.loads(pq.ParquetFile(CC_FILES[0]).schema_arrow.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
cc = scan_parquet(CC_FILES, ["label", "crop"])
CC_PESTS = {"fall armyworm", "grasshoper", "green mite", "leaf beetle", "leaf miner"}
print(f"{len(cc):,} images, {len(cc_names)} classes; crops: {dict(Counter(r['crop'] for r in cc))}")
cc_counts = Counter(f"{r['crop']}|{cc_names[r['label']]}" for r in cc)
cc_kind = lambda cond: "healthy" if cond == "healthy" else "pest" if cond in CC_PESTS else "disease"
plot_crops(cc_counts, lambda key: (key.split("|")[0].capitalize(), key.split("|")[1], cc_kind(key.split("|")[1])),
           f"CCMT: images per crop ({len(cc):,}, raw)", "plague_ccmt_crops.png")
print(pd.crosstab(pd.Series([r["crop"] for r in cc], name="crop"), pd.Series([cc_names[r["label"]] for r in cc], name="class")).T.to_string())
""")

code(r"""
cc_sizes = [(r["width"], r["height"], r["mode"]) for r in cc]
describe_sizes(cc_sizes)
SIZES["CCMT"] = [w for w, _, _ in cc_sizes]
digests = Counter()
for path in CC_FILES:
    for item in pq.read_table(path, columns=["image"]).column("image").to_pylist():
        digests[hashlib.md5(item["bytes"]).hexdigest()] += 1
print(f"byte-identical duplicates: {sum(v - 1 for v in digests.values() if v > 1):,} extra copies")
pick = [next(i for i in RNG.permutation(len(cc)) if cc_names[cc[i]["label"]] == n) for n in ("green mite", "fall armyworm", "leaf miner", "healthy", "mosaic", "leaf blight")]
imgs = parquet_images(CC_FILES, pick)
show_samples([(imgs[i], [], f"{cc[i]['crop']}: {cc_names[cc[i]['label']]}") for i in pick], "CCMT: samples")
SUMMARY["CCMT"] = {"role": "field pest damage + disease", "kind": "pest + disease", "task": "classification", "images": len(cc),
                   "classes": len(cc_names), "license": "CC BY 4.0", "setting": "field (Ghana farms)"}
""")

md(r"""
**Gotchas:** pest classes show the **damage** (holes, mines, stippling), not the insect itself, which is useful because that is what a field camera sees. Not berries. No split column in `raw` (the `augmented` config has one); split by image yourself. All images were resized to 400×400, and there are byte-identical duplicates (counted above): de-duplicate before splitting.
""")

# ---------------------------------------------------------------- Strawberry_12
md(r"""
## 7. Strawberry_12: `Ethan0300/Strawberry_12-dataset`

Hu, Jiang & Guan, *StrawDet-YOLO: A Real-Time and High Precision Object Detection for Strawberry*, IEEE SMC 2025. Greenhouse and open field, April 2022 – October 2023. **MIT license.**

**Role:** the main **strawberry** detector data: diseases, **pests** and **nutrient deficiencies** in one set, YOLO boxes. The repo has a zip (used here, extracted to `data/strawberry12/`) plus the same files unzipped.

⚠️ **Class names are not shipped** (labels are ids 0–11, no `data.yaml`). They were recovered from the file-name prefixes, which are unambiguous per class (e.g. every class-9 box is in a `fertile_Ca*` image), and checked on crops: 0 ripe fruit, 1 unripe fruit, 2 flower, 3 leaf spot, 4 powdery mildew, 5 gray mold, 6 anthracnose, 7 blight, 8 insect pests, 9 calcium deficiency, 10 iron deficiency, 11 potassium (K/P) deficiency.
""")

code(r"""
S12_DIR = DATA_DIR / "strawberry12" / "Strawberry_12 dataset"
if not S12_DIR.exists():
    import zipfile
    from huggingface_hub import hf_hub_download
    zipfile.ZipFile(hf_hub_download("Ethan0300/Strawberry_12-dataset", "Strawberry_12 dataset.zip", repo_type="dataset")).extractall(S12_DIR.parent)

S12_NAMES = ["ripe fruit", "unripe fruit", "flower", "leaf spot", "powdery mildew", "gray mold", "anthracnose", "blight",
             "insect pests", "calcium deficiency", "iron deficiency", "potassium (K/P) deficiency"]
S12_KIND = ["plant part"] * 3 + ["disease"] * 5 + ["pest"] + ["deficiency"] * 3
AUG = re.compile(r"_(blur|bright|dark|noise)$")

s12 = []
for label_file in sorted((S12_DIR / "labels").glob("*/*.txt")):
    image = next((p for ext in (".jpg", ".jpeg", ".png", ".JPG") if (p := S12_DIR / "images" / label_file.parent.name / f"{label_file.stem}{ext}").exists()), None)
    if image is None:
        continue
    with Image.open(image) as img:
        W, H = img.size
    boxes = []
    for line in label_file.read_text().splitlines():
        if line.strip():
            c, x, y, w, h = line.split()[:5]
            x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
            boxes.append((x - w / 2, y - h / 2, w, h, int(c)))
    m = AUG.search(label_file.stem)
    s12.append({"split": label_file.parent.name, "file": image, "width": W, "height": H, "boxes": boxes,
                "augmentation": m.group(1) if m else "original", "stem": AUG.sub("", label_file.stem)})
s12_df = pd.DataFrame(s12)
print(f"{len(s12):,} images, {sum(len(r['boxes']) for r in s12):,} boxes; per split {s12_df['split'].value_counts().to_dict()}")
print("augmentation copies:", s12_df["augmentation"].value_counts().to_dict())
""")

code(r"""
s12_counts = Counter(S12_NAMES[c] for r in s12 if r["augmentation"] == "original" for *_, c in r["boxes"])
plot_bars([(name, s12_counts[name], kind) for name, kind in zip(S12_NAMES, S12_KIND)],
          f"Strawberry_12: boxes per class (original images only, {sum(s12_counts.values()):,} boxes)", unit="boxes",
          filename="plague_strawberry12_classes.png")
BERRY_COUNTS["Strawberry_12"] = Counter()
for name, kind in zip(S12_NAMES, S12_KIND):
    if kind != "plant part":
        BERRY_COUNTS["Strawberry_12"][("strawberry", name)] = s12_counts[name]
        KIND_OF[("strawberry", name)] = kind

# leakage: does an original and its augmented copies sit in different splits?
splits_per_stem = s12_df.groupby("stem")["split"].agg(set)
print(f"{len(splits_per_stem):,} source photos; photos whose copies span more than one split: {int((splits_per_stem.map(len) > 1).sum())}")
areas = np.array([w * h / (r["width"] * r["height"]) for r in s12 for _, _, w, h, _ in r["boxes"]])
print(f"box area % of image: median {np.median(areas) * 100:.2f}, p10 {np.percentile(areas, 10) * 100:.2f}, p90 {np.percentile(areas, 90) * 100:.2f}")
describe_sizes([(r["width"], r["height"], None) for r in s12])
SIZES["Strawberry_12"] = [r["width"] for r in s12]
""")

code(r"""
originals = [r for r in s12 if r["augmentation"] == "original"]
pick = []
for cls in (8, 4, 5, 6, 9, 11):  # pests, powdery mildew, gray mold, anthracnose, Ca and K deficiency
    pick.append(next(r for r in (originals[i] for i in RNG.permutation(len(originals))) if any(b[4] == cls for b in r["boxes"])))
show_samples([(Image.open(r["file"]), [(x, y, w, h, S12_NAMES[c]) for x, y, w, h, c in r["boxes"]],
               ", ".join(sorted({S12_NAMES[c] for *_, c in r["boxes"]}))[:48]) for r in pick],
             "Strawberry_12: samples", color_of=lambda name: KIND_COLORS[S12_KIND[S12_NAMES.index(name)]])
SUMMARY["Strawberry_12"] = {"role": "strawberry disease + pest + deficiency", "kind": "disease + pest + deficiency",
                            "task": "object detection", "images": len(s12), "classes": 12, "license": "MIT",
                            "setting": "greenhouse + field (+ some web images)"}
""")

md(r"""
**Gotchas:** 57% of the images are blurred/brightened/darkened/noisy copies of the originals (counted above), so count and split by source photo. Some "original" images are themselves augmented: **black cutout rectangles**, **2×2 mosaics** of four photos (see the anthracnose sample), and heavily upscaled tiny images (pixelated). A few are web images with watermarks. The fruit/flower classes make it usable as a combined "plant part + problem" detector.
""")

# ---------------------------------------------------------------- tomato & pepper
md(r"""
# Tomato and pepper

Sections 8–10 add datasets for **tomato** and **pepper** (sweet/bell pepper and chili). PlantVillage, LeafNet2.0, PlantSeg, PlantDoc and CCMT already contain tomato and pepper classes (counted together at the end). The new sets were picked because they add what those lack: field photos with **boxes**, diseases on **fruit and stems**, and a pepper-family set with **pests and nutrient deficiencies**.

All three are AgML re-hosts on Hugging Face (parquet), read with `scan_parquet` like the sets above.
""")

code(r"""
def agml_files(repo, folder="data"):
    root = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{folder}/*", "README.md"]))
    return sorted((root / folder).glob("*.parquet"))


def roboflow_stem(path):
    # Roboflow exports name every copy <source>_<ext>.rf.<hash>.jpg: copies of one photo share the stem
    return re.sub(r"\.rf\.[0-9a-f]+", "", path)
""")

# ---------------------------------------------------------------- 8 tomato leaf disease detection
md(r"""
## 8. Tomato leaf disease detection: `Project-AgML/tomato_leaf_disease_detection`

Imtiaz et al., *Tomato leaf dataset: A dataset for multiclass disease detection and classification*, Data in Brief 60 (2025); Mendeley doi:10.17632/bpfd9cns5g. **License: CC BY 4.0.** Field photos (Canon EOS M50) from tomato gardens in three districts of Bangladesh.

**Role:** a small **field detection** set for tomato leaf diseases, with boxes on individual lesions.

⚠️ **Class names are not shipped**: the parquet and card name the classes `'0'`…`'6'`. The paper lists 7 classes (Early Blight, Black Spot, Late Blight, Leaf Mold, Bacterial Spot, Target Spot, Healthy) but not their id order, and the order does not match an alphabetical export. Looking at samples, **id 2 is healthy** (one box around the whole leaf). The other ids cannot be named with confidence, so they are kept as `disease id k`.
""")

code(r"""
TLD_FILES = agml_files("Project-AgML/tomato_leaf_disease_detection")
tld = scan_parquet(TLD_FILES, ["objects", "split"])
TLD_NAME = {k: "healthy" if k == 2 else f"disease id {k}" for k in range(7)}
tld_counts = Counter(TLD_NAME[c] for r in tld for c in r["objects"]["categories"])
print(f"{len(tld):,} images, {sum(tld_counts.values()):,} boxes; split column: {dict(Counter(r['split'] for r in tld))}")
plot_bars([(name, tld_counts[name], "healthy" if name == "healthy" else "disease") for name in TLD_NAME.values()],
          f"Tomato leaf disease detection: boxes per class ({sum(tld_counts.values()):,})", unit="boxes",
          filename="plague_tomato_leaf_detection_classes.png")
describe_sizes([(r["width"], r["height"], r["mode"]) for r in tld])
SIZES["Tomato leaf det. (Imtiaz)"] = [r["width"] for r in tld]
""")

code(r"""
tld_stems = defaultdict(set)
for r in tld:
    tld_stems[roboflow_stem(r["path"])].add(r["split"])
print(f"{len(tld):,} images come from {len(tld_stems):,} source photos (Roboflow .rf.<hash> copies); "
      f"copies per photo: {dict(Counter(Counter(roboflow_stem(r['path']) for r in tld).values()))}")
print(f"source photos whose copies sit in more than one split: {sum(len(s) > 1 for s in tld_stems.values())}")
areas = np.array([w * h / (r["width"] * r["height"]) for r in tld for _, _, w, h in r["objects"]["bbox"]])
print(f"box area % of image: median {np.median(areas) * 100:.2f}, p10 {np.percentile(areas, 10) * 100:.2f}, p90 {np.percentile(areas, 90) * 100:.2f}")
pick = [next(i for i in RNG.permutation(len(tld)) if c in tld[i]["objects"]["categories"]) for c in (2, 0, 1, 3, 5, 6)]
imgs = parquet_images(TLD_FILES, pick)
show_samples([(imgs[i], [(*b, TLD_NAME[c]) for b, c in zip(tld[i]["objects"]["bbox"], tld[i]["objects"]["categories"])],
               ", ".join(sorted({TLD_NAME[c] for c in tld[i]["objects"]["categories"]}))) for i in pick],
             "Tomato leaf disease detection: samples",
             color_of=lambda n: KIND_COLORS["healthy" if n == "healthy" else "disease"])
SUMMARY["Tomato leaf det. (Imtiaz)"] = {"role": "tomato field leaf-disease detection", "kind": "disease", "task": "object detection",
                                        "images": len(tld), "classes": 7, "license": "CC BY 4.0", "setting": "field (Bangladesh)"}
""")

md(r"""
**Gotchas:** it is a **Roboflow export**: images were resized to 640×640 and many are augmented copies (same stem, counted above). The original `split` column keeps copies of one photo in different splits, so re-split by stem. Names for 6 of the 7 ids are unknown; ask the authors or re-label before using it for per-disease training. Healthy-vs-diseased detection works as is.
""")

# ---------------------------------------------------------------- 9 tomato disease detection
md(r"""
## 9. Tomato disease detection: `Project-AgML/tomato_disease_detection`

Liu et al., *A labeled image dataset of common tomato diseases for classification and object detection*, Data in Brief 63 (2025); Mendeley doi:10.17632/c2x8rynybg.1. **License: CC BY 4.0.**

**Role:** whole-plant tomato disease detection, with boxes on **fruit, leaves, stem base and plant top**: gray mold, viral disease and bacterial wilt. It is the only set here with boxes on diseased **tomato fruit**.

The card has no class-label names, but its text gives the mapping: 0 GrayMold_Fruit, 1 GrayMold_Leaf, 2 Viral_Leaf, 3 Viral_Top, 4 Virus_Middle, 5 Wilt_Base, 6 Wilt_Leaf, 7 Wilt_Middle, 8 Wilt_Stem, 9 Wilt_Top.
""")

code(r"""
TD_FILES = agml_files("Project-AgML/tomato_disease_detection")
TD_NAMES = ["GrayMold_Fruit", "GrayMold_Leaf", "Viral_Leaf", "Viral_Top", "Virus_Middle",
            "Wilt_Base", "Wilt_Leaf", "Wilt_Middle", "Wilt_Stem", "Wilt_Top"]
td = scan_parquet(TD_FILES, ["objects", "disease"])
td_counts = Counter(TD_NAMES[c] for r in td for c in r["objects"]["categories"])
print(f"{len(td):,} images, {sum(td_counts.values()):,} boxes; images per disease column: {dict(Counter(r['disease'] for r in td))}")
print("class ids used:", sorted({c for r in td for c in r["objects"]["categories"]}))
plot_bars([(name, td_counts[name], "disease") for name in TD_NAMES],
          f"Tomato disease detection: boxes per class ({sum(td_counts.values()):,})", unit="boxes",
          filename="plague_tomato_disease_detection_classes.png")
describe_sizes([(r["width"], r["height"], r["mode"]) for r in td])
SIZES["Tomato disease det. (Liu)"] = [r["width"] for r in td]
print("images with no box:", sum(not r["objects"]["bbox"] for r in td))
mismatch = sum(not all(TD_NAMES[c].lower().startswith(r["disease"].lower()[:3]) for c in r["objects"]["categories"]) for r in td)
print("images whose boxes disagree with the image-level disease column:", mismatch)
""")

code(r"""
outside = sum(x < -2 or y < -2 or x + w > r["width"] + 2 or y + h > r["height"] + 2 for r in td for x, y, w, h in r["objects"]["bbox"])
print(f"boxes outside the image (2 px tolerance): {outside}")
pick = [next(i for i in RNG.permutation(len(td)) if c in td[i]["objects"]["categories"]) for c in (0, 1, 2, 3, 5, 8)]
imgs = parquet_images(TD_FILES, pick)
show_samples([(imgs[i], [(*b, TD_NAMES[c]) for b, c in zip(td[i]["objects"]["bbox"], td[i]["objects"]["categories"])],
               ", ".join(sorted({TD_NAMES[c] for c in td[i]["objects"]["categories"]}))) for i in pick],
             "Tomato disease detection: samples")
SUMMARY["Tomato disease det. (Liu)"] = {"role": "tomato whole-plant disease detection (fruit, leaf, stem)", "kind": "disease",
                                        "task": "object detection", "images": len(td), "classes": 10, "license": "CC BY 4.0",
                                        "setting": "greenhouse / field"}
""")

# ---------------------------------------------------------------- 10 chili
md(r"""
## 10. Chili leaf diseases, pests, deficiency: `Project-AgML/COLD_chili_leaf_disease_classification`

Aishwarya & Reddy, *Dataset of chilli and onion plant leaf images for classification and detection*, Data in Brief 54 (2024); Mendeley doi:10.17632/tf9dtfz9m6.2. **License: CC BY 4.0.**

**Role:** small, but the only pepper-family set with a **pest** class (mites and thrips), a **nutrient deficiency** class and a **healthy** class. The `raw` config (532 images) is used; `augmented` (10,974) is generated copies and is skipped. The parquet has no class names (plain integer label); names come from the card: 0 cercospora, 1 healthy, 2 mites_and_trips, 3 nutritional, 4 powdery mildew.
""")

code(r"""
CH_FILES = agml_files("Project-AgML/COLD_chili_leaf_disease_classification", "raw")
CH_NAMES = ["cercospora", "healthy", "mites_and_trips", "nutritional", "powdery mildew"]
CH_KIND = ["disease", "healthy", "pest", "deficiency", "disease"]
ch = scan_parquet(CH_FILES, ["label"])
ch_counts = Counter(CH_NAMES[r["label"]] for r in ch)
plot_bars([(name, ch_counts[name], kind) for name, kind in zip(CH_NAMES, CH_KIND)],
          f"Chili leaves (COLD): images per class ({len(ch):,}, raw)", filename="plague_chili_classes.png")
ch_sizes = [(r["width"], r["height"], r["mode"]) for r in ch]
describe_sizes(ch_sizes)
SIZES["Chili leaves (COLD)"] = [w for w, _, _ in ch_sizes]
digests = defaultdict(set)
for path in CH_FILES:
    for item, label in zip(*pq.read_table(path, columns=["image", "label"]).to_pydict().values()):
        digests[hashlib.md5(item["bytes"]).hexdigest()].add(label)
print(f"byte-identical duplicates: {len(ch) - len(digests):,} extra copies; "
      f"with conflicting labels: {sum(len(v) > 1 for v in digests.values())}")
pick = [next(i for i in RNG.permutation(len(ch)) if ch[i]["label"] == k) for k in range(5)]
imgs = parquet_images(CH_FILES, pick)
show_samples([(imgs[i], [], CH_NAMES[ch[i]["label"]]) for i in pick], "Chili leaves (COLD): samples")
SUMMARY["Chili leaves (COLD)"] = {"role": "chili disease + pest + deficiency", "kind": "disease + pest + deficiency",
                                  "task": "classification", "images": len(ch), "classes": 5, "license": "CC BY 4.0",
                                  "setting": "field (India)"}
""")

# ---------------------------------------------------------------- cucumber
md(r"""
# Cucumber

Sections 11–12 add **cucumber**. None of the general sets above is strong on cucumber (counted in the coverage chart at the end).
""")

# ---------------------------------------------------------------- 11 cucumber classification
md(r"""
## 11. Cucumber diseases (leaf, fruit, stem): `Project-AgML/cucumber_disease_classification`

Sultana et al., *Cucumber disease recognition dataset*, Data in Brief (2023); Mendeley doi:10.17632/y6d3z6f8z9.1 (Jahangirnagar University, Bangladesh). **License: CC BY 4.0** (Mendeley; the AgML card states none).

**Role:** field photos of cucumber diseases on **leaves, fruit and stems**: anthracnose, bacterial wilt, belly rot, downy mildew, gummy stem blight, Pythium fruit rot, plus healthy leaves and healthy fruit. The Mendeley release has 1,280 original photos and 6,400 augmented copies (flip, shear, zoom, rotation) in separate zips. ⚠️ **The AgML copy mixes both with no flag column**, and its parquet label has no class names (names taken from the card, in card order; checked against the file names below). The two groups can be told apart only by file-name style: `Anthracnose(12).jpg` (original) vs `Anthracnose_(12).jpg` (augmented).
""")

code(r"""
CU_FILES = agml_files("Project-AgML/cucumber_disease_classification")
# the parquet label is a plain integer; names in card order
cu_names = ["Anthracnose", "Bacterial_Wilt", "Belly_Rot", "Downy_Mildew", "Fresh_Cucumber", "Fresh_Leaf", "Gummy_Stem_Blight", "Pythium_Fruit_Rot"]
cu = scan_parquet(CU_FILES, ["label"])
for r in cu:
    r["copy"] = "augmented" if re.search(r"_\(\d+\)", r["path"]) else "original"
print(f"{len(cu):,} images; by file-name style: {dict(Counter(r['copy'] for r in cu))}")
flat = lambda t: re.sub(r"[^a-z]", "", t.lower())
print("labels that disagree with the file-name class:", sum(not flat(r["path"]).startswith(flat(cu_names[r["label"]])) for r in cu))
print(pd.crosstab(pd.Series([cu_names[r["label"]] for r in cu], name="class"), pd.Series([r["copy"] for r in cu], name="copy")).to_string())
CU_PART = {"Anthracnose": "leaf", "Bacterial_Wilt": "plant", "Belly_Rot": "fruit", "Downy_Mildew": "leaf", "Fresh_Cucumber": "fruit",
           "Fresh_Leaf": "leaf", "Gummy_Stem_Blight": "stem/leaf", "Pythium_Fruit_Rot": "fruit"}
cu_kind = lambda name: "healthy" if name.startswith("Fresh") else "disease"
cu_counts = Counter(cu_names[r["label"]] for r in cu if r["copy"] == "original")
plot_bars([(f"{name} ({CU_PART[name]})", cu_counts[name], cu_kind(name)) for name in cu_names],
          f"Cucumber diseases: original images per class ({sum(cu_counts.values()):,}; augmented copies excluded)",
          filename="plague_cucumber_classes.png")
""")

code(r"""
cu_sizes = [(r["width"], r["height"], r["mode"]) for r in cu]
describe_sizes(cu_sizes)
SIZES["Cucumber diseases (Sultana)"] = [w for w, _, _ in cu_sizes]
for copy in ("original", "augmented"):
    print(copy, "sizes:", Counter((r["width"], r["height"]) for r in cu if r["copy"] == copy).most_common(3))
digests = defaultdict(set)
for path in CU_FILES:
    pf = pq.ParquetFile(path)
    for group in range(pf.num_row_groups):
        t = pf.read_row_group(group, columns=["image", "label"]).to_pydict()
        for item, label in zip(t["image"], t["label"]):
            digests[hashlib.md5(item["bytes"]).hexdigest()].add(label)
print(f"byte-identical duplicates: {len(cu) - len(digests):,} extra copies; with conflicting labels: {sum(len(v) > 1 for v in digests.values())}")
originals = [i for i, r in enumerate(cu) if r["copy"] == "original"]
pick = [next(i for i in RNG.permutation(originals) if cu_names[cu[i]["label"]] == n)
        for n in ("Belly_Rot", "Pythium_Fruit_Rot", "Anthracnose", "Downy_Mildew", "Gummy_Stem_Blight")]
pick.append(next(i for i in RNG.permutation(len(cu)) if cu[i]["copy"] == "augmented"))
imgs = parquet_images(CU_FILES, pick)
show_samples([(imgs[i], [], f"{cu_names[cu[i]['label']]} ({cu[i]['copy']})") for i in pick], "Cucumber diseases: samples")
SUMMARY["Cucumber diseases (Sultana)"] = {"role": "cucumber leaf + fruit + stem diseases", "kind": "disease", "task": "classification",
                                          "images": len(originals), "classes": len(cu_names), "license": "CC BY 4.0",
                                          "setting": "field (Bangladesh)"}
""")

md(r"""
**Gotchas:** use only the originals (count above), or keep each original and its augmented copies in the same split. The augmented copies cannot be linked to their original by name (separate numbering), so a random split of the AgML table leaks. No pests and no deficiencies.
""")

# ---------------------------------------------------------------- 12 cucumber leaf detection
md(r"""
## 12. Cucumber leaf abnormality detection: Omarov et al., Mendeley `mb3vnhpk8z`

Omarov, Ibrayev, Imanbayeva & Belgitayeva, *An Annotated Image Dataset for Cucumber Leaf Disease Detection and Segmentation in Complex Orchard Environments*, Mendeley Data (2026), doi:10.17632/mb3vnhpk8z.1. **License: CC BY 4.0.** Not on Hugging Face; the zip (≈220 MB) is in `data/cucumber_leaf_detection/` and read in place.

**Role:** small **field detection** set: 250 healthy leaf images (no labels) + 250 affected leaf images with YOLO boxes of two **symptom** classes: `color` (abnormal colour such as yellowing/browning) and `damage` (visible disease damage). Labels describe symptoms, not pathogens.
""")

code(r"""
import zipfile

CL_ZIP = zipfile.ZipFile(DATA_DIR / "cucumber_leaf_detection" / "mb3vnhpk8z-1.zip")


cl_files = [n for n in CL_ZIP.namelist() if not n.endswith("/")]
cl_classes = [c.strip() for c in CL_ZIP.read(next(n for n in cl_files if n.endswith("classes.txt"))).decode().splitlines() if c.strip()]
print("classes.txt:", cl_classes)
cl = []
for name in cl_files:
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    with Image.open(io.BytesIO(CL_ZIP.read(name))) as img:
        W, H = img.size
    label = re.sub(r"/images/([^/]+)\.\w+$", r"/labels/\1.txt", name) if "/images/" in name else None
    boxes = []
    if label in cl_files:
        for line in CL_ZIP.read(label).decode().splitlines():
            if line.strip():
                c, x, y, w, h = line.split()[:5]
                x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
                boxes.append((x - w / 2, y - h / 2, w, h, cl_classes[int(c)]))
    cl.append({"file": name, "group": "healthy" if "/Healthy leaves/" in name else "affected", "width": W, "height": H,
               "boxes": boxes, "has_label": label in cl_files, "stem": re.sub(r"\d+", "#", name.rsplit("/", 1)[-1])})
cl_df = pd.DataFrame(cl)
print(cl_df.groupby("group").agg(images=("file", "size"), with_label_file=("has_label", "sum")).to_string())
cl_counts = Counter(b[4] for r in cl for b in r["boxes"])
print(f"boxes: {dict(cl_counts)}; affected images with no box: {sum(1 for r in cl if r['group'] == 'affected' and not r['boxes'])}")
plot_bars([(f"{c} (symptom)", cl_counts[c], "disease") for c in cl_classes], f"Cucumber leaf detection: boxes per class ({sum(cl_counts.values()):,})",
          unit="boxes", filename="plague_cucumber_leaf_detection_classes.png")
describe_sizes([(r["width"], r["height"], None) for r in cl])
SIZES["Cucumber leaf det. (Omarov)"] = [r["width"] for r in cl]
print("file-name patterns:", Counter(cl_df["stem"]).most_common(6))
""")

code(r"""
def ahash_bytes(name):
    return ahash(Image.open(io.BytesIO(CL_ZIP.read(name))))


hashes = {r["file"]: ahash_bytes(r["file"]) for r in cl}
names = list(hashes)
H = np.array([hashes[n] for n in names])
dist = (H[:, None, :] != H[None, :, :]).sum(-1)
np.fill_diagonal(dist, 999)
near = int((dist.min(1) <= 10).sum())
print(f"images with a near-identical twin (perceptual hash, <= 10 of 256 bits): {near} of {len(names)}")
outside = sum(x < -2 or y < -2 or x + w > r["width"] + 2 or y + h > r["height"] + 2 for r in cl for x, y, w, h, _ in r["boxes"])
print("boxes outside the image:", outside)
pick = [r for r in (cl[i] for i in RNG.permutation(len(cl))) if r["boxes"]][:5] + [next(r for r in cl if r["group"] == "healthy")]
show_samples([(Image.open(io.BytesIO(CL_ZIP.read(r["file"]))), [b for b in r["boxes"]], f"{r['group']}: {len(r['boxes'])} boxes")
              for r in pick], "Cucumber leaf detection: samples",
             color_of=lambda n: KIND_COLORS["deficiency"] if n == "color" else KIND_COLORS["disease"])
SUMMARY["Cucumber leaf det. (Omarov)"] = {"role": "cucumber leaf symptom detection", "kind": "disease (symptoms)", "task": "object detection",
                                          "images": len(cl), "classes": 2, "license": "CC BY 4.0", "setting": "greenhouse rows (video frames)"}
""")

md(r"""
**Gotchas:** despite "orchard" in the title, the samples show a **greenhouse** (high-wire cucumber rows, substrate slabs, heating pipes) filmed from the row. Many file names are video frames (`frame_####`), so neighbouring images are near-duplicates (count above): split by video/source prefix, not at random. Many `color` boxes cover whole leaves under a strong yellow light cast, so "abnormal colour" is partly confounded with lighting; check before training a colour class. Some boxes extend past the image border (count above; clip them). Healthy images have no boxes, so they work only as negatives. In the samples, yellow boxes = `color`, orange = `damage`.
""")

# ---------------------------------------------------------------- pepper set = apple
md(r"""
## Checked and discarded: `Project-AgML/pepper_disease_pest_classification` (it is apple, not pepper)

26,377 images, 5 classes, cited as Wang, Liu & Chen, *An advanced deep learning method for pepper diseases and pests detection*, Plant Methods 21 (2025); Mendeley "vegetable disease" doi:10.17632/tg3z7xxkdb.1. CC BY 4.0.

⚠️ **The images are apple leaves, not pepper.** The leaves are ovate with serrated margins, the "rust" class shows orange cedar-apple-rust spots, and the class list (Alternaria blotch, brown spot, grey spot, mosaic, rust, with the typo `Alternaria_Boltch`) is exactly the well-known Chinese apple-leaf-disease set. There is also no pest class and no healthy class, despite the repo name. It is kept out of the tomato/pepper comparison. As an apple set it would also need de-duplication (count below) and has a lab/field mix (white paper vs orchard backgrounds).
""")

code(r"""
PP_FILES = agml_files("Project-AgML/pepper_disease_pest_classification")
pp_names = json.loads(pq.ParquetFile(PP_FILES[0]).schema_arrow.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
pp_labels = pq.read_table(PP_FILES, columns=["label"]).column("label").to_pylist()
print(f"{len(pp_labels):,} images; per class: {dict(Counter(pp_names[l] for l in pp_labels))}")
digests = defaultdict(set)
index = 0
for path in PP_FILES:
    pf = pq.ParquetFile(path)
    for group in range(pf.num_row_groups):
        for item in pf.read_row_group(group, columns=["image"]).column("image").to_pylist():
            digests[hashlib.md5(item["bytes"]).hexdigest()].add(pp_labels[index])
            index += 1
print(f"byte-identical duplicates: {len(pp_labels) - len(digests):,} extra copies; with conflicting labels: {sum(len(v) > 1 for v in digests.values())}")
pick = sorted(RNG.choice(len(pp_labels), 6, replace=False))
imgs = parquet_images(PP_FILES, pick)
show_samples([(imgs[i], [], f"labelled {pp_names[pp_labels[i]]}") for i in pick], "Pepper diseases set: samples are apple leaves")
""")

# ---------------------------------------------------------------- tomato leaf disease = PlantVillage
md(r"""
## Checked and discarded: `Project-AgML/tomato_leaf_disease`

11,000 tomato leaf images, 10 classes × 1,100 (Kaggle `kaustubhb999/tomatoleaf`, CC0). The classes are exactly PlantVillage's tomato classes and the file names carry PlantVillage's `<uuid>___<source> <n>.JPG` pattern. The perceptual-hash check below compares a sample of 60 images per class with the same PlantVillage class.
""")

code(r"""
TL_FILES = agml_files("Project-AgML/tomato_leaf_disease")
tl_names = json.loads(pq.ParquetFile(TL_FILES[0]).schema_arrow.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
tl = pq.read_table(TL_FILES, columns=["image", "label"]).to_pydict()
print(f"{len(tl['label']):,} images; file names with PlantVillage's '___' pattern: {sum('___' in i['path'] for i in tl['image']):,}")
TL_TO_PV = {"Bacterial Spot": "Bacterial_spot", "Early Blight": "Early_blight", "Healthy": "healthy", "Late Blight": "Late_blight",
            "Leaf Mold": "Leaf_Mold", "Septoria Leaf Spot": "Septoria_leaf_spot",
            "Spider Mites Two-spotted Spider Mite": "Spider_mites Two-spotted_spider_mite", "Target Spot": "Target_Spot",
            "Tomato Mosaic Virus": "Tomato_mosaic_virus", "Tomato Yellow Leaf Curl Virus": "Tomato_Yellow_Leaf_Curl_Virus"}
found, total = 0, 0
for k, name in enumerate(tl_names):
    pv_class = f"Tomato___{TL_TO_PV[name]}"
    pid = pv_names.index(pv_class)
    b = [ahash(pv[s][i]["image"]) for s in pv for i, l in enumerate(pv_labels[s]) if l == pid]
    idx = [i for i, l in enumerate(tl["label"]) if l == k]
    a = [ahash(Image.open(io.BytesIO(tl["image"][i]["bytes"]))) for i in RNG.choice(idx, 60, replace=False)]
    found, total = found + matches(a, b), total + len(a)
print(f"sampled AgML tomato_leaf_disease images also in PlantVillage: {found} of {total}")
""")

# ---------------------------------------------------------------- cross dataset
md(r"""
## Cross-dataset comparison

### Berry coverage

Examples per berry and condition kind, per dataset. The IP102 row lists only the pests that also attack strawberries (insect images, not on strawberry plants).
""")

code(r"""
def coverage_rows(counts_by_dataset, crops):
    rows = []
    for dataset, counts in counts_by_dataset.items():
        for crop in crops:
            by_kind = Counter()
            for (b, c), n in counts.items():
                if b == crop:
                    by_kind[KIND_OF[(b, c)]] += n
            if by_kind:
                n_cond = sum(1 for (b, c) in counts if b == crop and KIND_OF[(b, c)] != "healthy")
                rows.append((f"{dataset} · {crop}", by_kind, n_cond))
    return rows


def plot_coverage(rows, title, xlabel, filename):
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.4), layout="constrained")
    for i, (label, by_kind, n_cond) in enumerate(rows):
        left = 0
        for kind in KINDS:
            if by_kind.get(kind):
                ax.barh(i, by_kind[kind], left=left, color=KIND_COLORS[kind], height=0.65, edgecolor=SURFACE, linewidth=1.5)
                left += by_kind[kind]
        ax.text(left, i, f" {left:,}  ({n_cond} condition{'' if n_cond == 1 else 's'})", va="center", fontsize=7, color=MUTED_INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(sum(r[1].values()) for r in rows) * 1.3)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    legend(fig, {k for r in rows for k in r[1]})
    fig.savefig(CHART_DIR / filename, dpi=150)
    plt.show()


plot_coverage(coverage_rows(BERRY_COUNTS, BERRIES), "Berry examples per dataset by condition kind",
              "examples (images; boxes for PlantDoc and Strawberry_12)", "plague_berry_coverage.png")

matrix = pd.DataFrame({d: {f"{b} · {c}": n for (b, c), n in counts.items()} for d, counts in BERRY_COUNTS.items()}).fillna(0).astype(int)
matrix.sort_index()
""")

md(r"""
### Tomato, pepper and cucumber coverage

Same view for tomato, pepper and cucumber, across the general sets (1–6) and the new ones (8–12). Bell pepper and chili are shown separately; LeafNet2.0's black pepper (*Piper nigrum*) is excluded. Cucumber #11 counts originals only. Pepper includes bell pepper and chili. Conditions named after mites, thrips, leaf miners, aphids or whiteflies count as pests, and nutritional ones as deficiencies, in every dataset (the general sets file spider mites under disease).
""")

code(r"""
def tp_crop(crop):
    # bell pepper and chili are both Capsicum but kept apart; black pepper (Piper nigrum, LeafNet2.0) is a different plant
    crop = crop.lower().replace("_", " ")
    if "black pepper" in crop:
        return None
    for name, keys in (("cucumber", ("cucumber",)), ("tomato", ("tomato",)), ("bell pepper", ("bell pepper", "pepper")), ("chili", ("chil",))):
        if any(k in crop for k in keys):
            return name
    return None


def tp_counts(counts, split_name):
    out = Counter()
    for name, n in counts.items():
        crop, condition, kind = split_name(name)
        if tp_crop(crop) and kind != "plant part":
            key = (tp_crop(crop), condition.lower().replace("_", " "))
            # same condition name = same kind in every dataset (general sets label mites as "disease")
            if re.search(r"mite|trips|thrips|miner|aphid|whitefly", key[1]):
                kind = "pest"
            elif re.search(r"nutrition|deficien", key[1]):
                kind = "deficiency"
            out[key] += n
            KIND_OF[key] = kind
    return out


cc_split = lambda key: (key.split("|")[0], key.split("|")[1], cc_kind(key.split("|")[1]))
TP_COUNTS = {
    "PlantVillage": tp_counts(pv_counts, split_triple_underscore),
    "LeafNet2.0": tp_counts(ln_counts, split_triple_underscore),
    "PlantSeg v3": tp_counts(ps_counts, split_plantseg),
    "PlantDoc": tp_counts(pd_counts, split_plantdoc),
    "CCMT": tp_counts(cc_counts, cc_split),
    "Tomato leaf det. (Imtiaz)": tp_counts(tld_counts, lambda n: ("tomato", n, "healthy" if n == "healthy" else "disease")),
    "Tomato disease det. (Liu)": tp_counts(td_counts, lambda n: ("tomato", n, "disease")),
    "Chili leaves (COLD)": tp_counts(ch_counts, lambda n: ("chili", n, CH_KIND[CH_NAMES.index(n)])),
    "Cucumber diseases (Sultana)": tp_counts(cu_counts, lambda n: ("cucumber", n, cu_kind(n))),
    "Cucumber leaf det. (Omarov)": tp_counts(cl_counts, lambda n: ("cucumber", f"{n} symptom", "disease")),
}
TP_COUNTS = {d: c for d, c in TP_COUNTS.items() if c}
plot_coverage(coverage_rows(TP_COUNTS, ("tomato", "bell pepper", "chili", "cucumber")), "Tomato, pepper and cucumber examples per dataset by condition kind",
              "examples (images; boxes for PlantDoc and the detection sets 8, 9, 12)", "plague_tomato_pepper_coverage.png")
tp_table = pd.DataFrame({d: {f"{b} · {c}": n for (b, c), n in counts.items()} for d, counts in TP_COUNTS.items()}).fillna(0).astype(int).sort_index()
tp_crop_of = tp_table.index.str.split(" · ").str[0]
print("examples per crop:\n", tp_table.groupby(tp_crop_of).sum().T.to_string(), sep="")
print("\nclasses (incl. healthy) per crop:\n", (tp_table > 0).groupby(tp_crop_of).sum().T.to_string(), sep="")
print("\nall classes:\n", tp_table.to_string(), sep="")
""")

md(r"""
### Overlaps between datasets

Perceptual-hash check (16×16, robust to resizing): how many images of one dataset also appear in another, for classes they share.
""")

code(r"""
for pv_class, ln_class in [("Raspberry___healthy", "Raspberry___Healthy"), ("Strawberry___Leaf_scorch", "Strawberry___Leaf_scorch"),
                           ("Strawberry___healthy", "Strawberry___Healthy")]:
    pid, lid = pv_names.index(pv_class), ln_names.index(ln_class)
    a = [ahash(pv[s][i]["image"]) for s in pv for i, l in enumerate(pv_labels[s]) if l == pid]
    b = [ahash(ln[i]["image"]) for i, l in enumerate(ln_labels) if l == lid]
    print(f"PlantVillage {pv_class:<26} {len(a):>5} images: {matches(a, b):>5} also in LeafNet2.0 ({len(b)})")

# PlantSeg vs PlantWild (the discarded dataset), on the shared berry classes
pw_root = Path(snapshot_download("EGOISTyrh/mvpdr-plant-disease", repo_type="dataset",
                                 allow_patterns=["plantwild/images/**", "plantwild/classes.txt"])) / "plantwild" / "images"
SHARED = ("strawberry leaf scorch", "strawberry anthracnose", "blueberry rust")
seg_hashes = defaultdict(list)
for path in PS_FILES:  # one pass over PlantSeg for all shared classes
    for item in pq.read_table(path, columns=["image", "answer"]).to_pylist():
        if item["answer"] in SHARED:
            seg_hashes[item["answer"]].append(ahash(Image.open(io.BytesIO(item["image"]["bytes"]))))
for cls in SHARED:
    seg_imgs = seg_hashes[cls]
    wild_imgs = [ahash(Image.open(p)) for p in (pw_root / cls).glob("*")]
    print(f"PlantSeg {cls:<24} {len(seg_imgs):>4} images: {matches(seg_imgs, wild_imgs):>4} also in PlantWild ({len(wild_imgs)})")
""")

md(r"""
### Image size

Median image width (dot) with the 10th–90th percentile range (line), log scale.
""")

code(r"""
names = list(SIZES)
fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1.3), layout="constrained")
for row, name in enumerate(names):
    w = np.array(SIZES[name])
    ax.plot([np.percentile(w, 10), np.percentile(w, 90)], [row, row], color="#86b6ef", linewidth=2, solid_capstyle="round")
    ax.plot(np.median(w), row, "o", color="#1c5cab", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.text(9000, row, f"median {int(np.median(w))} px", va="center", fontsize=8, color=MUTED_INK)
ax.set_xscale("log")
ax.set_xlim(100, 20000)
ax.set_xticks([128, 256, 512, 1024, 2048, 4096], ["128", "256", "512", "1024", "2048", "4096"])
ax.set_yticks(range(len(names)), names)
ax.invert_yaxis()
ax.set_xlabel("image width, px (log)")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
ax.set_title("Image width per dataset (median and 10-90% range)", loc="left", fontweight="bold", color=INK)
fig.savefig(CHART_DIR / "plague_image_width.png", dpi=150)
plt.show()
""")

md(r"""
### Summary
""")

code(r"""
pd.DataFrame(SUMMARY).T
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
nbf.write(nb, "plague.ipynb")
print("wrote", len(cells), "cells")
