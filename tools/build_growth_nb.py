"""Builds growth.ipynb. Run from the repo root, then execute the notebook with nbconvert (see CLAUDE.md)."""
import nbformat as nbf

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------- intro
md(r"""
# Plant growth datasets (berry focus)

**Purpose:** survey public growth/ripeness datasets (one section per dataset). **Inputs:** Hugging Face cache and `data/` (see CLAUDE.md). **Outputs:** `charts/growth_*.png`, `examples/growth/`, findings in `GROWTH_DATASET_FINDINGS.md`. **Run:** `make growth`. **Owner:** CreAI (GRZR-16).

Companion to `plague.ipynb` (plant **disease**). This notebook looks at datasets for plant **growth**: growth stages, flowering → fruit set → ripening, leaf count and time series. The focus is **strawberry, blueberry and raspberry**, with extra sections for **tomato, sweet pepper** and **hydroponic** growth. Findings are summarized in `GROWTH_DATASET_FINDINGS.md`.

| # | Dataset | Crop | Intended task | Annotation | Time series? |
|---|---|---|---|---|---|
| 1 | Strawberry growth `Project-AgML/strawberry_growth_detection` (HF) | strawberry | detection of 7 growth stages | boxes + fruit size + within-stage progress | yes: tagged fruits, 34 dates |
| 2 | Strawberry 5-stage `strawberry-vnobp/strawberry-2-k4eap` (Roboflow) | strawberry | detection of 5 stages | boxes | partly: 10 time-lapse plants |
| 3 | Strawberry & avocado ripeness `Project-AgML/strawberry_avocado_ripeness_detection` (HF) | strawberry (+ avocado) | ripeness detection | boxes, class names missing | no |
| 4 | Blueberry maturity `Project-AgML/blueberry_maturity_detection` (HF) | blueberry | detection + counting | boxes | no |
| 5 | Blueberry segmentation `c-tan/Blueberry-fruit-segmentation` (HF) | blueberry | instance segmentation | polygons | no |
| 6 | Raspberry `ssd-nv4zv/ssd-raspberry` (Roboflow) | raspberry | detection of buds → ripe berries | boxes (+ some polygons) | no |
| 7 | Chili plant growth `sdp-iiccq/chili-plant-growth-detection-r9kld` (Roboflow) | chili (general) | whole-plant growth-stage detection | boxes | partly: growth-chamber time-lapse |
| 8 | GrowliFlower `Voxel51/GrowliFlower` (HF) | cauliflower (general) | growth time series, segmentation | measured traits + masks + keypoints | **yes**: 739 plants × up to 15 dates |
| 9 | Mango growth `Project-AgML/mango_growth_detection` (HF) | mango (general, fruit) | detection of 4 fruit stages | boxes | no |
| 10 | Leaf counting `Project-AgML/leaf_counting_denmark` (HF) | field seedlings (general) | classification by leaf count | one label per image | no |
| 11 | Tomato ripeness (Laboro) `Project-AgML/tomato_ripeness_detection` (HF) | tomato | detection of 3 ripeness stages × 2 sizes | boxes (EXIF-rotated) | no |
| 12 | Greenhouse tomato (AgRob) `Project-AgML/Greenhouse_Tomato_AgRob_object_detection` (HF) | tomato | detection of 4 colour stages from a robot | boxes + occlusion flag | no (one day, video frames) |
| 13 | Tomato maturity `Project-AgML/tomato_maturity_classification` (HF) | tomato | classification immature / mature | one label per image | no |
| 14 | Sweet pepper maturity, Mendeley `ttntwwxkfd` | sweet pepper | classification of maturity class 1–4, multi-view | one label per pepper | no |
| 15 | Hydroponic lettuce (AGC 2021), 4TU `10.4121/15023088` | lettuce, **hydroponic** | regression of fresh weight, height, diameter, leaf area | measured traits + depth | no (destructive, week not given) |
| 16 | HydroGrowNet, Mendeley `g6cm3v3wdp` | lettuce, **hydroponic NFT** | growth monitoring + sensor logs | none (segmented crops) | daily, but no plant id |
| 17 | AGC 2018 cucumber `koidra/GH_Cucumber` (HF mirror of 4TU) | cucumber | crop growth / yield time series, no images | daily production, weekly crop traits | **yes** (per compartment) |

**How the data was obtained.** HF datasets come from the Hub cache (`snapshot_download`). Sections 14–16 come from Mendeley Data / 4TU zips in `data/` (see each section). The Roboflow sets were **forked into workspace `veggies-h2dsa`** (`ssd-raspberry-zflvo`, `strawberry-2-k4eap-nrnoa`, `chili-plant-growth-detection-r9kld-lcly3`), versioned **with no preprocessing and no augmentation**, exported as COCO and unzipped to `data/roboflow/<name>/`. Exporting the public Universe versions directly fails with `NoSuchKey`, but the fork route works.
""")

code(r"""
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, "tools")  # shared helpers
from cvlib.common import card_class_names, parquet_files, rf_stem

stem = rf_stem  # Roboflow stem, used in the sections below

CHART_DIR = Path("charts")
CHART_DIR.mkdir(exist_ok=True)
EXAMPLE_DIR = Path("examples") / "growth"
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")
RNG = np.random.default_rng(0)
""")

md(r"""
### Common stage scale

Each dataset names its stages differently (`small g`, `Turning Red`, `Premature`, `Immature`, `unblue_visible`...). To compare them, every class is mapped to one **common fruit-stage scale**:

`flower / bud` → `green (unripe)` → `white / pale` → `turning` → `ripe`, plus `overripe / rotten` outside the scale.

Colors: a single-hue ordinal ramp (light = early, dark = late), checked with the dataviz palette validator (monotone lightness, visible gaps between steps, light end ≥ 2:1 against the surface). Eight steps of one hue fail that check, so the scale is kept to five levels. Overripe/rotten is gray because it sits past the end of the scale. Whole-plant stages (seedling → fruiting, chili dataset) use their own 4-step ramp.
""")

code(r"""
STAGES = ["flower / bud", "green (unripe)", "white / pale", "turning", "ripe"]
STAGE_COLORS = dict(zip(STAGES, ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]))
PAST_RIPE = "overripe / rotten"
STAGE_COLORS[PAST_RIPE] = "#898781"
STAGE_ORDER = STAGES + [PAST_RIPE]
DARK_FILLS = {"#2a78d6", "#1c5cab", "#0d366b", "#898781"}  # white text on these

PLANT_STAGES = ["seedling", "vegetative", "flowering", "fruiting"]
PLANT_COLORS = dict(zip(PLANT_STAGES, ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]))

INK = "#0b0b0b"
MUTED_INK = "#52514e"
AXIS_INK = "#898781"
GRID_COLOR = "#e1e0d9"
SURFACE = "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED_INK, "xtick.color": AXIS_INK,
                     "ytick.color": MUTED_INK, "font.size": 9})

# Filled in by each section, used by the cross-dataset charts at the end.
STAGE_BOXES = {}   # dataset -> Counter(common stage -> boxes)
BOX_STATS = {}     # dataset -> dict from box_stats()
""")

md(r"""
### Helpers

- `card_class_names`: class names from the dataset card. For several AgML repos the names embedded in the parquet files differ from the card (see blueberry), so the card is the source of truth.
- `scan_parquet`: walks the parquet row groups straight from the HF cache and reads only the image **header** for width/height/mode. This avoids `load_dataset`, which would copy several GB into an Arrow cache.
- `box_stats`: object counts and sizes. *Tiny* = box area under 32×32 px (the COCO "small" threshold), the main difficulty for berries.
- `stem`: strips Roboflow's `.rf.<hash>` suffix, so augmented copies of the same source photo share a stem. That is how train/test leakage is measured.
""")

code(r"""
def scan_parquet(repo, columns, folder="data"):
    # One dict per image: path, width, height, mode + the requested non-image columns. Image bytes are not kept.
    rows = []
    for path in parquet_files(repo, folder):
        pf = pq.ParquetFile(path)
        for group in range(pf.num_row_groups):
            for row in pf.read_row_group(group, columns=["image", *columns]).to_pylist():
                image = row.pop("image")
                with Image.open(io.BytesIO(image["bytes"])) as img:
                    row.update(path=image["path"], width=img.width, height=img.height, mode=img.mode,
                               exif_orientation=img.getexif().get(274, 1))
                rows.append(row)
    return rows


def parquet_images(repo, wanted, folder="data", transpose=False):
    # Decoded images for the row indices in `wanted` (second pass over the files, stops when all are found).
    # transpose=True applies the EXIF orientation (needed when the boxes were drawn on the rotated photo).
    wanted, found, index = set(wanted), {}, 0
    for path in parquet_files(repo, folder):
        pf = pq.ParquetFile(path)
        for group in range(pf.num_row_groups):
            column = pf.read_row_group(group, columns=["image"]).column("image")
            for item in column.to_pylist():
                if index in wanted:
                    img = Image.open(io.BytesIO(item["bytes"]))
                    found[index] = (ImageOps.exif_transpose(img) if transpose else img).convert("RGB")
                index += 1
            if len(found) == len(wanted):
                return found
    return found


def box_stats(boxes, image_sizes, boxes_per_image):
    # boxes: list of (w, h) in pixels; image_sizes: matching list of (W, H)
    areas = np.array([w * h for w, h in boxes])
    rel = areas / np.array([W * H for W, H in image_sizes])
    per = np.array(boxes_per_image)
    return {
        "images": len(per), "boxes": len(boxes),
        "boxes/img median": float(np.median(per)), "boxes/img max": int(per.max()),
        "empty images": int((per == 0).sum()),
        "box area % median": float(np.median(rel) * 100), "box area % p10": float(np.percentile(rel, 10) * 100),
        "box area % p90": float(np.percentile(rel, 90) * 100), "tiny (<32² px) %": float((areas < 32 * 32).mean() * 100),
    }


def describe_sizes(widths, heights, modes=None):
    sizes = Counter(zip(widths, heights))
    print(f"{len(widths):,} images, {len(sizes):,} distinct sizes; most common: {sizes.most_common(4)}")
    print(f"width  min / median / max: {min(widths)} / {int(np.median(widths))} / {max(widths)}")
    print(f"height min / median / max: {min(heights)} / {int(np.median(heights))} / {max(heights)}")
    if modes is not None:
        print("modes:", dict(Counter(modes)))


def print_stats(stats):
    for key, value in stats.items():
        print(f"  {key:<20}{value:,.2f}" if isinstance(value, float) else f"  {key:<20}{value:,}")
""")

md(r"""
### Chart helpers

- `plot_stage_bars`: one horizontal bar per class, in growth order. Color shows the class's position on the common scale, and the number at the end is the count.
- `show_samples`: example images with their boxes drawn in the stage color, so you can check that boxes land on the fruit and that stage names match what is visible.
""")

code(r"""
def plot_stage_bars(counts, stage_of, title, filename=None, unit="boxes", palette=STAGE_COLORS, order=STAGE_ORDER):
    names = sorted(counts, key=lambda name: (order.index(stage_of(name)), name))
    values = [counts[name] for name in names]
    rows = range(len(names))
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(names) + 1.3), layout="constrained")
    ax.barh(rows, values, color=[palette[stage_of(name)] for name in names], height=0.7,
            edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(rows, [f"{name}  ({stage_of(name)})" for name in names])
    for row, value in zip(rows, values):
        ax.text(value, row, f" {value:,}", va="center", fontsize=8, color=MUTED_INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel(unit)
    ax.set_title(title, loc="left", fontweight="bold", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    if filename:
        fig.savefig(CHART_DIR / filename, dpi=150)
    plt.show()


def save_example(fig, title):
    # Every example figure is also saved as examples/growth/<title slug>.png
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower().split("(yellow")[0]).strip("_")[:60].strip("_")
    fig.savefig(EXAMPLE_DIR / f"{slug}.png", dpi=120)


def show_samples(samples, color_of, title, cols=3, thumb=420):
    # samples: list of (PIL image, [(x, y, w, h, class_name), ...], caption)
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), layout="constrained")
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (img, boxes, caption) in zip(np.ravel(axes), samples):
        img = img.copy()
        scale = thumb / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        draw = ImageDraw.Draw(img)
        line = max(2, thumb // 200)
        for x, y, w, h, name in boxes:
            draw.rectangle([x * scale, y * scale, (x + w) * scale, (y + h) * scale], outline=color_of(name), width=line)
        ax.imshow(img)
        ax.set_title(caption, fontsize=8, color=MUTED_INK)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", color=INK)
    save_example(fig, title)
    plt.show()


def legend_handles(palette, used):
    return [Patch(color=palette[name], label=name) for name in palette if name in used]
""")

# ---------------------------------------------------------------- 1 strawberry growth
md(r"""
## 1. Strawberry growth: `Project-AgML/strawberry_growth_detection`

Yang et al., *Predicting the growth trajectory and yield of greenhouse strawberries based on knowledge-guided computer vision*, Computers and Electronics in Agriculture 220 (2024), Zenodo doi:10.5281/zenodo.10957909. Reformatted by AgML. **License: CC BY 4.0.**

**Intended task:** object detection of **7 growth stages** (`flower`, `small g`, `green`, `white`, `turning red`, `red`, `overripe`). Each box also has optional **fruit diameter, fruit length** and a **`decimal_stage`**. The `source` column separates `tagged` fruits (photographed repeatedly over the season, with measurements) from `random` field images.

What to check: stage balance, how complete the measurements are, what `decimal_stage` means, the time structure of the `tagged` subset, and the bbox convention.
""")

code(r"""
SG_REPO = "Project-AgML/strawberry_growth_detection"
sg_names = card_class_names(SG_REPO)
print("classes from card:", sg_names)
sg = scan_parquet(SG_REPO, ["source", "objects"])  # reads 6.7 GB of parquet once (image headers only)
print(f"{len(sg):,} images, {sum(len(r['objects']['categories']) for r in sg):,} boxes")
print("source:", Counter(r["source"] for r in sg))
""")

code(r"""
SG_STAGE = {"flower": "flower / bud", "small g": "green (unripe)", "green": "green (unripe)", "white": "white / pale",
            "turning red": "turning", "red": "ripe", "overripe": PAST_RIPE}
sg_counts = Counter(sg_names[c] for r in sg for c in r["objects"]["categories"])
for source in ("tagged", "random"):
    print(source, dict(Counter(sg_names[c] for r in sg if r["source"] == source for c in r["objects"]["categories"])))
plot_stage_bars(sg_counts, SG_STAGE.get, f"Strawberry growth (AgML): boxes per stage ({sum(sg_counts.values()):,} boxes)",
                "growth_strawberry_agml_stages.png")
STAGE_BOXES["Strawberry growth (AgML)"] = Counter()
for k, v in sg_counts.items():
    STAGE_BOXES["Strawberry growth (AgML)"][SG_STAGE[k]] += v  # small g + green both map to "green"
""")

md(r"""
### Measurements and `decimal_stage`

`diameter`, `length` and `decimal_stage` are strings. Empty or `-1` means not measured. Measurements exist only on `tagged` fruits.
""")

code(r"""
def measured(values):
    return [float(v) for v in values if v not in (None, "", "-1")]


for field in ("diameter", "length", "decimal_stage"):
    values = [v for r in sg for v in r["objects"][field]]
    good = measured(values)
    print(f"{field:<14} measured on {len(good):,} of {len(values):,} boxes; min / median / max {min(good)} / {np.median(good)} / {max(good)}")

print("\ndecimal_stage by class (it restarts in every class, so it is progress *within* the stage, not a global scale):")
for name in sg_names:
    vals = measured(v for r in sg for c, v in zip(r["objects"]["categories"], r["objects"]["decimal_stage"]) if sg_names[c] == name)
    print(f"  {name:<12} n={len(vals):>4}  min {min(vals)}  median {np.median(vals)}  max {max(vals)}")
""")

md(r"""
### Time structure (`tagged` subset)

`tagged` file names are `<YYYYMMDD>_<n>.jpg`. Most `random` names are `img_<YYYYMMDD>_<n>.jpg`, and 61 are plain `img_<n>.jpg`.
""")

code(r"""
def capture_date(path):
    m = re.search(r"(20\d{6})", path)
    return m.group(1) if m else None


for source in ("tagged", "random"):
    dates = Counter(capture_date(r["path"]) for r in sg if r["source"] == source)
    undated = dates.pop(None, 0)
    print(f"{source}: {len(dates)} capture dates, {min(dates)} -> {max(dates)}, undated images {undated}")
    print("   images per month:", dict(sorted(Counter(d[:6] for d in dates.elements()).items())))
random_dates = Counter(capture_date(r["path"]) for r in sg if r["source"] == "random")
print(f"\n⚠️ {random_dates['20210101']} 'random' images are dated 20210101 (Jan 1): most likely an unset camera clock, not a real date.")
""")

code(r"""
describe_sizes([r["width"] for r in sg], [r["height"] for r in sg], [r["mode"] for r in sg])
print("EXIF orientation:", Counter(r["exif_orientation"] for r in sg))
out_of_bounds = sum(1 for r in sg for x, y, w, h in r["objects"]["bbox"] if x + w > r["width"] + 2 or y + h > r["height"] + 2)
print(f"bbox format [x, y, w, h] in pixels; boxes outside the image: {out_of_bounds}")
BOX_STATS["Strawberry growth (AgML)"] = box_stats(
    [(w, h) for r in sg for _, _, w, h in r["objects"]["bbox"]],
    [(r["width"], r["height"]) for r in sg for _ in r["objects"]["bbox"]],
    [len(r["objects"]["bbox"]) for r in sg])
print_stats(BOX_STATS["Strawberry growth (AgML)"])
""")

code(r"""
pick = sorted(RNG.choice([i for i, r in enumerate(sg) if len(r["objects"]["bbox"]) >= 3], 6, replace=False))
images = parquet_images(SG_REPO, pick)
show_samples([(images[i], [(*b, sg_names[c]) for b, c in zip(sg[i]["objects"]["bbox"], sg[i]["objects"]["categories"])],
               f"{sg[i]['path']} ({sg[i]['source']})") for i in pick],
             lambda name: STAGE_COLORS[SG_STAGE[name]], "Strawberry growth (AgML): samples")
""")

# ---------------------------------------------------------------- 2 strawberry 5-stage roboflow
md(r"""
## 2. Strawberry 5-stage (Roboflow): `strawberry-vnobp/strawberry-2-k4eap`

https://universe.roboflow.com/strawberry-vnobp/strawberry-2-k4eap. **License: CC BY 4.0.** Fork `veggies-h2dsa/strawberry-2-k4eap-nrnoa`, version 1 (raw, no preprocessing or augmentation).

**Intended task:** object detection, 5 stages `Flower`, `Green`, `White`, `Turning Red`, `Red`. It is a **merge of several sources**, which the file names reveal: greenhouse camera images named `<condition>_<plant>_<n>_<YYMMDDhhmmss>` (conditions `normal`, `chlorosis`, `wilt`, `plague`), a time-lapse named `YYYY-MM-DDTHH_P<plant>`, `tip*`, `DSC*` and numbered PNGs.

What to check: whether the Universe class counts match the export, empty images, sources, time-lapse coverage, and **duplicates across splits**.
""")

code(r"""
def load_coco(folder, splits=("train", "valid", "test"), json_name="_annotations.coco.json"):
    images, annotations = [], []
    for split in splits:
        coco = json.loads(Path(folder, split, json_name).read_text())
        names = {c["id"]: c["name"] for c in coco["categories"]}
        for img in coco["images"]:
            images.append({"split": split, "id": img["id"], "file": img["file_name"], "width": img["width"],
                           "height": img["height"], "path": Path(folder, split, img["file_name"])})
        for a in coco["annotations"]:
            x, y, w, h = a["bbox"]
            seg = a.get("segmentation") or []
            annotations.append({"split": split, "image_id": a["image_id"], "class": names[a["category_id"]],
                                "x": x, "y": y, "w": w, "h": h, "has_polygon": bool(seg),
                                "broken_polygon": any(v is None for part in seg for v in part),
                                "point": (seg[0][0], seg[0][1]) if seg and seg[0][0] is not None else None})
    images, annotations = pd.DataFrame(images), pd.DataFrame(annotations)
    images["stem"] = images["file"].map(stem)
    return images, annotations, names


def coco_overview(images, annotations, name):
    broken = int(annotations["broken_polygon"].sum())
    if broken:
        print(f"⚠️ {broken:,} annotations have a broken polygon (see below); they are left out of the box statistics\n")
        annotations = annotations[~annotations["broken_polygon"]]
    per_image = annotations.groupby(["split", "image_id"]).size()
    print("images per split:", images["split"].value_counts().to_dict())
    print(pd.crosstab(annotations["class"], annotations["split"], margins=True).to_string(), "\n")
    merged = annotations.merge(images, left_on=["split", "image_id"], right_on=["split", "id"])
    counts = images.apply(lambda r: per_image.get((r["split"], r["id"]), 0), axis=1)
    BOX_STATS[name] = box_stats(list(zip(merged["w"], merged["h"])), list(zip(merged["width"], merged["height"])), list(counts))
    print_stats(BOX_STATS[name])
    print("annotations that also carry a polygon:", int(annotations["has_polygon"].sum()))
    describe_sizes(list(images["width"]), list(images["height"]))


def leakage(images):
    copies = images.groupby("stem")["split"].agg(list)
    print(f"{len(images):,} images come from {len(copies):,} source photos; {sum(len(v) > 1 for v in copies):,} photos appear more than once")
    for a, b in (("train", "test"), ("train", "valid"), ("valid", "test")):
        shared = sum(1 for v in copies if a in v and b in v)
        print(f"  source photos in both {a} and {b}: {shared:,}")


S5_DIR = DATA_DIR / "roboflow" / "strawberry5"
s5_images, s5_ann, _ = load_coco(S5_DIR)
coco_overview(s5_images, s5_ann, "Strawberry 5-stage (Roboflow)")
""")

md(r"""
The Universe page lists Flower 6,147, Green 7,630, White 10,630, Turning Red 5,130 and Red 5,310 annotations. The raw export has about **half** of that (table above). The Universe numbers appear to count the stretched or augmented copies of the latest version, so **use the export counts**.
""")

code(r"""
S5_STAGE = {"Flower": "flower / bud", "Green": "green (unripe)", "White": "white / pale", "Turning Red": "turning", "Red": "ripe"}
s5_counts = Counter(s5_ann["class"])
plot_stage_bars(s5_counts, S5_STAGE.get, f"Strawberry 5-stage (Roboflow): boxes per stage, all splits ({len(s5_ann):,} boxes)",
                "growth_strawberry_roboflow_stages.png")
STAGE_BOXES["Strawberry 5-stage (Roboflow)"] = Counter({S5_STAGE[k]: v for k, v in s5_counts.items()})
""")

code(r"""
def s5_source(name):
    if re.match(r"\d{4}-\d{2}-\d{2}T\d{2}_P\d+", name):
        return "time-lapse YYYY-MM-DDTHH_P#"
    m = re.match(r"([a-z]+)_\d+_\d+_\d{12}", name)
    if m:
        return f"greenhouse cam: {m.group(1)}"
    if re.match(r"\d+_png", name):
        return "numbered PNG"
    m = re.match(r"([A-Za-z]+)", name)
    return m.group(1) if m else "other"


sources = s5_images.drop_duplicates("stem")["stem"].map(s5_source)
print("source photos by origin:", sources.value_counts().to_dict(), "\n")
leakage(s5_images)

lapse = s5_images.drop_duplicates("stem")["stem"].str.extract(r"(\d{4}-\d{2}-\d{2})T\d{2}_P(\d+)").dropna()
per_plant = lapse.groupby(1)[0].nunique()
print(f"\ntime-lapse: {per_plant.size} plants, dates per plant {sorted(per_plant)}, {lapse[0].min()} -> {lapse[0].max()}")
empty = set(s5_images["split"] + "/" + s5_images["id"].astype(str)) - set(s5_ann["split"] + "/" + s5_ann["image_id"].astype(str))
print(f"images with no boxes: {len(empty):,} of {len(s5_images):,}")
""")

code(r"""
def coco_samples(images, annotations, n=6, min_boxes=2):
    counts = annotations.groupby(["split", "image_id"]).size()
    candidates = [r for r in images.itertuples() if counts.get((r.split, r.id), 0) >= min_boxes]
    out = []
    for i in RNG.choice(len(candidates), min(n, len(candidates)), replace=False):
        r = candidates[i]
        boxes = annotations[(annotations["split"] == r.split) & (annotations["image_id"] == r.id)]
        out.append((Image.open(r.path).convert("RGB"), list(zip(boxes["x"], boxes["y"], boxes["w"], boxes["h"], boxes["class"])),
                    f"{r.split}: {r.stem[:40]}"))
    return out


show_samples(coco_samples(s5_images, s5_ann), lambda name: STAGE_COLORS[S5_STAGE[name]], "Strawberry 5-stage (Roboflow): samples")
""")

# ---------------------------------------------------------------- 3 strawberry avocado
md(r"""
## 3. Strawberry & avocado ripeness: `Project-AgML/strawberry_avocado_ripeness_detection`

Kamat et al., *Comprehensive dataset on ripening stages of strawberries and avocados: From unripe to rotten*, Data in Brief 60 (2025). Collected with a handheld 12 MP phone over 25 days in Maharashtra, India. **License: CC BY 4.0.**

**Intended task:** ripeness detection. ⚠️ **The 8 classes are named only `"0"`…`"7"`**, both in the card and in the parquet. The names below were recovered by looking at crops of every class (next cell): 0–3 are strawberry and 4–7 are avocado, each going unripe → ripening → ripe → overripe/rotten. *Inferred, not from the source.*
""")

code(r"""
SA_REPO = "Project-AgML/strawberry_avocado_ripeness_detection"
print("card class names:", card_class_names(SA_REPO))
sa = scan_parquet(SA_REPO, ["objects"])
SA_NAMES = ["strawberry unripe", "strawberry partly ripe", "strawberry ripe", "strawberry overripe/rotten",
            "avocado unripe", "avocado ripening", "avocado ripe", "avocado overripe/rotten"]
SA_STAGE = dict(zip(SA_NAMES, ["green (unripe)", "white / pale", "ripe", PAST_RIPE] * 2))
sa_counts = Counter(SA_NAMES[c] for r in sa for c in r["objects"]["categories"])
print(f"{len(sa):,} images, {sum(sa_counts.values()):,} boxes; boxes/img median {np.median([len(r['objects']['bbox']) for r in sa])}")
plot_stage_bars(sa_counts, SA_STAGE.get, "Strawberry & avocado (AgML): boxes per class (names inferred)", "growth_strawberry_avocado_classes.png")
STAGE_BOXES["Strawberry ripeness (AgML, strawberry part)"] = Counter(
    {SA_STAGE[k]: v for k, v in sa_counts.items() if k.startswith("strawberry")})
""")

code(r"""
# Class crop sheet used to recover the names: 8 crops per class, one row per class id.
sa_pick = defaultdict(list)
for i, r in enumerate(sa):
    for c in set(r["objects"]["categories"]):
        if len(sa_pick[c]) < 8:
            sa_pick[c].append(i)
sa_imgs = parquet_images(SA_REPO, {i for v in sa_pick.values() for i in v})
fig, axes = plt.subplots(8, 8, figsize=(9, 9), layout="constrained")
for c in range(8):
    for j, ax in enumerate(axes[c]):
        ax.axis("off")
        if j < len(sa_pick[c]):
            r = sa[sa_pick[c][j]]
            x, y, w, h = next(b for b, k in zip(r["objects"]["bbox"], r["objects"]["categories"]) if k == c)
            ax.imshow(sa_imgs[sa_pick[c][j]].crop((x, y, x + w, y + h)).resize((96, 96)))
    axes[c][0].set_title(f"class {c}: {SA_NAMES[c]}", fontsize=7, loc="left", color=MUTED_INK)
save_example(fig, "Strawberry & avocado: class crops")
plt.show()
""")

md(r"""
### ⚠️ Boxes are wrong on many portrait photos

Some crops above are partly black because the box runs outside the image. All out-of-bounds boxes are on **portrait** phone photos (e.g. 3000×4000), and their x extent reaches up to ~1.33× the width (= 4000/3000). The boxes were converted to pixels with **width and height swapped**. Rescaling (`x·W/H`, `y·H/W`) puts them back on the fruit. The cell counts, per portrait image, whether the stored box or the swapped fix fits inside the image:

- **stored box OUT, fix fits** → the stored box is wrong, use the fix.
- **stored fits, fix OUT** → the stored box is right.
- **both fit** → *ambiguous*. The figure below (bottom row) shows cases going both ways, so these images need a visual check or re-annotation.
""")

code(r"""
def swap_fix(box, W, H):
    x, y, w, h = box
    return (x * W / H, y * H / W, w * W / H, h * H / W)


def inside(box, W, H, tol=2):
    x, y, w, h = box
    return x >= -tol and y >= -tol and x + w <= W + tol and y + h <= H + tol


verdict = Counter()
for r in sa:
    W, H, boxes = r["width"], r["height"], r["objects"]["bbox"]
    orient = "portrait" if H > W else "landscape/square"
    raw_ok = all(inside(b, W, H) for b in boxes)
    fix_ok = all(inside(swap_fix(b, W, H), W, H) for b in boxes) if H > W else None
    verdict[(orient, "raw fits" if raw_ok else "raw OUT", {True: "fix fits", False: "fix OUT", None: "-"}[fix_ok])] += 1
for key, n in sorted(verdict.items()):
    print(f"  {key}: {n}")
print("EXIF orientation tags:", Counter(r["exif_orientation"] for r in sa), "(all 1, so this is not an EXIF-rotation issue)")
print(f"boxes outside the image as stored: {sum(not inside(b, r['width'], r['height']) for r in sa for b in r['objects']['bbox'])} of {sum(len(r['objects']['bbox']) for r in sa)}")
""")

code(r"""
both_fit = [i for i, r in enumerate(sa) if r["height"] > r["width"]
            and all(inside(b, r["width"], r["height"]) for b in r["objects"]["bbox"])
            and all(inside(swap_fix(b, r["width"], r["height"]), r["width"], r["height"]) for b in r["objects"]["bbox"])][:3]
raw_out = [i for i, r in enumerate(sa) if not all(inside(b, r["width"], r["height"]) for b in r["objects"]["bbox"])][:3]
imgs = parquet_images(SA_REPO, both_fit + raw_out)
samples = []
for i in raw_out + both_fit:
    r = sa[i]
    boxes = [(*b, "as stored") for b in r["objects"]["bbox"]] + [(*swap_fix(b, r["width"], r["height"]), "swap fix") for b in r["objects"]["bbox"]]
    samples.append((imgs[i], boxes, f"{r['path']} {r['width']}x{r['height']}"))
show_samples(samples, {"as stored": "#eda100", "swap fix": "#2a78d6"}.get,
             "Strawberry & avocado: stored boxes (yellow) vs swapped-W/H fix (blue). Top: stored OUT. Bottom: both fit (ambiguous)")
BOX_STATS["Strawberry ripeness (AgML)"] = box_stats(
    [(w, h) for r in sa for _, _, w, h in r["objects"]["bbox"]],
    [(r["width"], r["height"]) for r in sa for _ in r["objects"]["bbox"]],
    [len(r["objects"]["bbox"]) for r in sa])
describe_sizes([r["width"] for r in sa], [r["height"] for r in sa], [r["mode"] for r in sa])
""")

md(r"""
The crop sheet also shows **cut-out fruit pasted on solid-color backgrounds** (yellow, blue, red, purple) in several classes. The source mixes real photos with synthetic or composited images, which is worth knowing before training a field detector on it.
""")

# ---------------------------------------------------------------- 4 blueberry maturity
md(r"""
## 4. Blueberry maturity: `Project-AgML/blueberry_maturity_detection`

Deng, Lu & Li, *Detection, counting, and maturity assessment of blueberries in canopy images using YOLOv8 and YOLOv9*, Smart Agricultural Technology 9 (2024); original release BlueberryDCM, Zenodo doi:10.5281/zenodo.14002517. **License: CC BY 4.0.**

**Intended task:** detection + **counting** + maturity in dense canopy images. Few images, very many boxes.
""")

code(r"""
BM_REPO = "Project-AgML/blueberry_maturity_detection"
bm_card = card_class_names(BM_REPO)
bm_file = json.loads(pq.ParquetFile(parquet_files(BM_REPO)[0]).schema_arrow.metadata[b"huggingface"])
print("class names in card   :", bm_card)
print("class names in parquet:", bm_file["info"]["features"]["objects"]["categories"]["feature"]["names"])
bm = scan_parquet(BM_REPO, ["objects"])
print(f"{len(bm)} images; class ids actually used:", Counter(c for r in bm for c in r["objects"]["categories"]))
""")

md(r"""
⚠️ **Only ids 0 and 3 are used**, while the card lists 4 names (`unblue_visible`, `unblue_occluded`, `blue_visible`, `blue_occluded`) and the README says 2 categories. The mean color inside the boxes (below) and the sample image show that **id 3 is ripe blue fruit and id 0 is unripe (green/pink)**. Read the labels as `0 = unripe`, `3 = ripe` and ignore the "occluded/visible" part of the names.
""")

code(r"""
BM_NAMES = {0: "unripe (id 0, 'unblue_visible')", 3: "ripe (id 3, 'blue_occluded')"}
BM_STAGE = {BM_NAMES[0]: "green (unripe)", BM_NAMES[3]: "ripe"}
bm_imgs = parquet_images(BM_REPO, range(12))
centers = defaultdict(list)
for i in range(12):
    a = np.asarray(bm_imgs[i], dtype=float)
    for (x, y, w, h), c in zip(bm[i]["objects"]["bbox"], bm[i]["objects"]["categories"]):
        patch = a[int(y + h * .3):int(y + h * .7), int(x + w * .3):int(x + w * .7)]
        if patch.size:
            centers[c].append(patch.reshape(-1, 3).mean(0))
for c, v in centers.items():
    print(f"id {c}: mean RGB at box centers {np.mean(v, 0).round()} over {len(v)} boxes")

bm_counts = Counter(BM_NAMES[c] for r in bm for c in r["objects"]["categories"])
plot_stage_bars(bm_counts, BM_STAGE.get, f"Blueberry maturity (AgML): boxes per class ({sum(bm_counts.values()):,} boxes, {len(bm)} images)",
                "growth_blueberry_maturity_classes.png")
STAGE_BOXES["Blueberry maturity (AgML)"] = Counter({BM_STAGE[k]: v for k, v in bm_counts.items()})
""")

code(r"""
describe_sizes([r["width"] for r in bm], [r["height"] for r in bm], [r["mode"] for r in bm])
print("file names:", [r["path"] for r in bm[:3]], "(cropped canopy images, no dates)")
BOX_STATS["Blueberry maturity (AgML)"] = box_stats(
    [(w, h) for r in bm for _, _, w, h in r["objects"]["bbox"]],
    [(r["width"], r["height"]) for r in bm for _ in r["objects"]["bbox"]],
    [len(r["objects"]["bbox"]) for r in bm])
print_stats(BOX_STATS["Blueberry maturity (AgML)"])
show_samples([(bm_imgs[i], [(*b, BM_NAMES[c]) for b, c in zip(bm[i]["objects"]["bbox"], bm[i]["objects"]["categories"])],
               f"{bm[i]['path']} ({len(bm[i]['objects']['bbox'])} boxes)") for i in (0, 1, 2)],
             lambda name: STAGE_COLORS[BM_STAGE[name]], "Blueberry maturity (AgML): samples (light = unripe, dark = ripe)")
""")

# ---------------------------------------------------------------- 5 blueberry segmentation
md(r"""
## 5. Blueberry segmentation: `c-tan/Blueberry-fruit-segmentation`

Ni, Li, Jiang & Takeda, *Deep learning image segmentation and extraction of blueberry fruit traits associated with harvestability and yield*, Horticulture Research 7 (2020). **License on HF: CC BY-NC-SA 4.0 (non-commercial)**. The COCO file itself says "Unknown License".

**Intended task:** **instance segmentation** (polygons) of individual berries, with maturity (`Immature` / `Mature`), used to derive count, maturity ratio and cluster compactness. The HF viewer shows only `image` + a folder label. The annotations are inside `berry_coco-formant-annotation.zip` (COCO, exported from VGG Image Annotator), so the zip has to be extracted (`data/blueberry_seg/`).
""")

code(r"""
import zipfile

BS_DIR = DATA_DIR / "blueberry_seg" / "berry_coco-formant-annotation"
if not BS_DIR.exists():
    zip_path = hf_hub_download("c-tan/Blueberry-fruit-segmentation", "berry_coco-formant-annotation.zip", repo_type="dataset")
    zipfile.ZipFile(zip_path).extractall(BS_DIR.parent)

bs_images, bs_ann = [], []
for split in ("train2017", "val2017", "test2017"):
    coco = json.loads((BS_DIR / "annotations" / f"instances_{split}.json").read_text())
    if split == "train2017":
        print("categories:", coco["categories"], "| license field:", coco.get("licenses"))
        names = {c["id"]: c["name"] for c in coco["categories"]}
    bs_images += [{"split": split, "id": i["id"], "file": i["file_name"], "width": i["width"], "height": i["height"],
                   "path": BS_DIR / split / i["file_name"]} for i in coco["images"]]
    bs_ann += [{"split": split, "image_id": a["image_id"], "class": names[a["category_id"]], "x": a["bbox"][0], "y": a["bbox"][1],
                "w": a["bbox"][2], "h": a["bbox"][3], "polygon_points": sum(len(p) for p in a["segmentation"]) // 2}
               for a in coco["annotations"]]
bs_images, bs_ann = pd.DataFrame(bs_images), pd.DataFrame(bs_ann)
print("images per split:", bs_images["split"].value_counts().to_dict())
print(pd.crosstab(bs_ann["class"], bs_ann["split"], margins=True).to_string())
print(f"\n⚠️ test2017 has {int((bs_images['split'] == 'test2017').sum())} images and "
      f"{int((bs_ann['split'] == 'test2017').sum())} annotations: the test split is unlabeled.")
print("polygon points per berry: median", int(bs_ann["polygon_points"].median()), "min", bs_ann["polygon_points"].min())
""")

code(r"""
BS_STAGE = {"Immature": "green (unripe)", "Mature": "ripe"}
bs_counts = Counter(bs_ann["class"])
plot_stage_bars(bs_counts, BS_STAGE.get, f"Blueberry segmentation: berries per class ({len(bs_ann):,} polygons)", "growth_blueberry_seg_classes.png")
STAGE_BOXES["Blueberry segmentation (Ni 2020)"] = Counter({BS_STAGE[k]: v for k, v in bs_counts.items()})
labeled = bs_images[bs_images["split"] != "test2017"]
m = bs_ann.merge(labeled, left_on=["split", "image_id"], right_on=["split", "id"])
per = bs_ann.groupby(["split", "image_id"]).size()
BOX_STATS["Blueberry segmentation (Ni 2020)"] = box_stats(list(zip(m["w"], m["h"])), list(zip(m["width"], m["height"])),
                                                          [per.get((r.split, r.id), 0) for r in labeled.itertuples()])
print_stats(BOX_STATS["Blueberry segmentation (Ni 2020)"])
describe_sizes(list(bs_images["width"]), list(bs_images["height"]))
""")

code(r"""
def polygon_samples(n=3):
    out = []
    coco = json.loads((BS_DIR / "annotations" / "instances_train2017.json").read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    by_image = defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)
    both = [i for i in coco["images"] if len({a["category_id"] for a in by_image[i["id"]]}) == 2]  # show both classes
    for img in both[:n]:
        pil = Image.open(BS_DIR / "train2017" / img["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(pil)
        line = max(3, pil.width // 400)
        for a in by_image[img["id"]]:
            for poly in a["segmentation"]:
                draw.polygon(poly, outline={"Immature": "#eda100", "Mature": "#e34948"}[names[a["category_id"]]], width=line)
        counts = Counter(names[a["category_id"]] for a in by_image[img["id"]])
        out.append((pil, [], f"{img['file_name']}: {dict(counts)}"))
    return out


# outlines (not fills) so the berry's own color stays visible; yellow = Immature, red = Mature
show_samples(polygon_samples(), None, "Blueberry segmentation: polygon outlines (yellow = immature, red = mature)")
""")

# ---------------------------------------------------------------- 6 raspberry
md(r"""
## 6. Raspberry (Roboflow): `ssd-nv4zv/ssd-raspberry`

https://universe.roboflow.com/ssd-nv4zv/ssd-raspberry. **License: CC BY 4.0.** Fork `veggies-h2dsa/ssd-raspberry-zflvo`, version 1 (raw).

The **only raspberry growth-stage dataset found** on HF or Roboflow. Classes: `Buds` → `Flowers` → `Unripe_Berries` → `Ripe_Berries`, covering the full reproductive cycle. Every other raspberry set found had either one class (`raspberry`) or unnamed ids.
""")

code(r"""
RB_DIR = DATA_DIR / "roboflow" / "raspberry"
rb_images, rb_ann, _ = load_coco(RB_DIR)
coco_overview(rb_images, rb_ann, "Raspberry (Roboflow)")
print()
leakage(rb_images)
""")

code(r"""
RB_STAGE = {"Buds": "flower / bud", "Flowers": "flower / bud", "Unripe_Berries": "green (unripe)", "Ripe_Berries": "ripe"}
rb_counts = Counter(rb_ann["class"])
plot_stage_bars(rb_counts, RB_STAGE.get, f"Raspberry (Roboflow): boxes per stage ({len(rb_ann):,} boxes)", "growth_raspberry_stages.png")
STAGE_BOXES["Raspberry (Roboflow)"] = Counter()
for k, v in rb_counts.items():
    STAGE_BOXES["Raspberry (Roboflow)"][RB_STAGE[k]] += v
print(f"ripe berries are {rb_counts['Ripe_Berries'] / len(rb_ann):.1%} of all boxes; images containing at least one ripe berry: "
      f"{rb_ann[rb_ann['class'] == 'Ripe_Berries'].groupby(['split', 'image_id']).ngroups}")
""")

md(r"""
### ⚠️ 14% of the training labels are point labels disguised as boxes

Some annotations carry a "polygon". Each of these polygons is actually two points plus `None, None`: the **first point is the fruit/bud center** and the second is a junk point near the image's top-left corner (≈ 40, 45). Roboflow built the box from those two points, so these labels become **huge boxes stretching from the corner to the fruit**. The first figure below marks the real center points (red) and the junk points (yellow) on one image. The second shows what a detector would be trained on.

They are usable as **point labels (counting)**, not as boxes. Drop them for detection, or re-annotate those images.
""")

code(r"""
broken = rb_ann[rb_ann["broken_polygon"]]
print(f"broken annotations: {len(broken):,} ({len(broken) / len(rb_ann[rb_ann['split'] == 'train']):.0%} of train boxes) in "
      f"{broken.groupby(['split', 'image_id']).ngroups} images; per split {broken['split'].value_counts().to_dict()}")
print("per class:", broken["class"].value_counts().to_dict())
images_with_broken = set(zip(broken["split"], broken["image_id"]))
all_broken = sum(1 for key, group in rb_ann.groupby(["split", "image_id"]) if key in images_with_broken and group["broken_polygon"].all())
print(f"images where every label is broken: {all_broken}")

r = rb_images[(rb_images["split"] == broken.iloc[0]["split"]) & (rb_images["id"] == broken.iloc[0]["image_id"])].iloc[0]
same = rb_ann[(rb_ann["split"] == r["split"]) & (rb_ann["image_id"] == r["id"])]
img = Image.open(r["path"]).convert("RGB")
draw = ImageDraw.Draw(img)
for a in same.itertuples():
    if a.point:
        draw.ellipse([a.point[0] - 30, a.point[1] - 30, a.point[0] + 30, a.point[1] + 30], outline="#e34948", width=12)
        draw.ellipse([a.x - 22, a.y - 22, a.x + 22, a.y + 22], fill="#eda100")
boxes = list(zip(same["x"], same["y"], same["w"], same["h"], same["class"]))
show_samples([(img, [], f"{r['stem']}: red = center point, yellow = junk point"),
              (Image.open(r["path"]).convert("RGB"), boxes, "the boxes Roboflow derived from them")],
             lambda name: STAGE_COLORS[RB_STAGE[name]], "Raspberry: broken point labels", cols=2)
""")

code(r"""
clean = rb_ann[~rb_ann["broken_polygon"]]
show_samples(coco_samples(rb_images, clean, n=3, min_boxes=10), lambda name: STAGE_COLORS[RB_STAGE[name]],
             "Raspberry (Roboflow): samples with clean boxes only")
""")

# ---------------------------------------------------------------- 7 chili general
md(r"""
## 7. General purpose: Chili plant growth (Roboflow) `sdp-iiccq/chili-plant-growth-detection-r9kld`

https://universe.roboflow.com/sdp-iiccq/chili-plant-growth-detection-r9kld. **License: CC BY 4.0.** Fork `veggies-h2dsa/chili-plant-growth-detection-r9kld-lcly3`, version 1 (raw).

**Intended task:** detection of the **whole plant** with its growth stage: `seedling` → `vegetative` → `flowering` → `fruiting`. Unlike the fruit-ripeness sets above, this is a plant-level stage scale that applies to any fruiting crop (strawberry included). It is the candidate for a crop-agnostic "what stage is this plant in" detector, or for pretraining one.

The images come from two sources: **growth-chamber time-lapse cameras** (`Wachstum<N>[_Cam<M>]_<date>`, German for "growth") and phone photos (`WhatsApp-Image-...`).
""")

code(r"""
CH_DIR = DATA_DIR / "roboflow" / "chili"
ch_images, ch_ann, _ = load_coco(CH_DIR)
coco_overview(ch_images, ch_ann, "Chili plant growth (Roboflow)")
print()
leakage(ch_images)
""")

code(r"""
ch_counts = Counter(ch_ann["class"])
plot_stage_bars(ch_counts, lambda name: name, f"Chili plant growth (Roboflow): plants per stage ({len(ch_ann):,} boxes)",
                "growth_chili_plant_stages.png", palette=PLANT_COLORS, order=PLANT_STAGES)

src = ch_images.drop_duplicates("stem")["stem"].map(
    lambda s: "time-lapse (Wachstum)" if s.startswith("Wachstum") else "phone (WhatsApp)" if s.startswith("WhatsApp") else "camera (IMG/DSC)")
print("source photos by origin:", src.value_counts().to_dict())
series = ch_images.drop_duplicates("stem")["stem"].str.extract(r"(Wachstum\d+(?:_Cam\d+)?)_(\d{4}-\d{2}-\d{2})").dropna()
print(series.groupby(0)[1].agg(["nunique", "min", "max"]).rename(columns={"nunique": "dates"}).to_string())
""")

code(r"""
print("stage mix per source (boxes):")
m = ch_ann.merge(ch_images, left_on=["split", "image_id"], right_on=["split", "id"])
m["source"] = m["stem"].map(lambda s: "time-lapse" if s.startswith("Wachstum") else "phone" if s.startswith("WhatsApp") else "camera")
print(pd.crosstab(m["source"], m["class"]).to_string())
show_samples(coco_samples(ch_images, ch_ann, n=6, min_boxes=1), PLANT_COLORS.get, "Chili plant growth (Roboflow): samples")
""")

md(r"""
Two things stand out. **Stage and source are confounded:** `seedling` comes almost only from phone photos of a seedling tray, and `flowering`/`fruiting` only from the growth-chamber time-lapse. A model can learn "tray vs chamber" instead of the growth stage. Also, many uploaded images **already contain black rotation corners**, so the uploader pushed augmented copies as source images, even though our version has no augmentation. The next cell counts images with at least 2 black corners.
""")

code(r"""
rotated = Counter()
for r in ch_images.itertuples():
    a = np.asarray(Image.open(r.path).convert("L"))
    black_corners = sum(q.max() < 12 for q in (a[:6, :6], a[:6, -6:], a[-6:, :6], a[-6:, -6:]))
    rotated[black_corners >= 2] += 1
print(f"images with >= 2 black corners (rotated copies): {rotated[True]:,} of {sum(rotated.values()):,}")
""")

# ---------------------------------------------------------------- 8 growliflower
md(r"""
## 8. General purpose: GrowliFlower `Voxel51/GrowliFlower`

Kierdorf et al., *GrowliFlower: An image time series dataset for GROWth analysis of cauLIFLOWER*, arXiv:2204.00294. UAV images of two cauliflower fields near Cologne (2020 and 2021). Voxel51 re-packaged it as a FiftyOne dataset. ⚠️ **License: unknown**: the card says to confirm terms with the corresponding author before redistribution.

This is the only dataset here with **real growth time series**: the same physical plant (`plant_id`) is cropped from every UAV flight and joined to **field measurements** (height, diameter, **BBCH stage**, head diameter, harvested). FiftyOne is not needed. All labels are in `samples.json` (36 MB), and only a few of the 24,857 images are downloaded here.
""")

code(r"""
GF_REPO = "Voxel51/GrowliFlower"
gf = json.loads(Path(hf_hub_download(GF_REPO, "samples.json", repo_type="dataset")).read_text())["samples"]
print(f"{len(gf):,} samples")
print(pd.crosstab(pd.Series([s["task"] for s in gf], name="task"), pd.Series([(s["tags"] or ["-"])[0] for s in gf], name="split")).to_string())
print("\nsplit = spatially disjoint split proposed by the source")
""")

code(r"""
ref = pd.DataFrame([{k: s.get(k) for k in ("plant_id", "field", "acquisition_date", "day_after_planting", "in_situ_height",
                                           "in_situ_diameter", "in_situ_bbch_stage", "in_situ_head_diameter", "in_situ_harvested",
                                           "coverage_frac", "filepath")} for s in gf if s["task"] == "reference"])
print(f"reference (phenotyped) subset: {ref['plant_id'].nunique()} plants, {len(ref):,} plant-date crops")
print("dates per plant:", ref.groupby("plant_id").size().describe()[["min", "50%", "max"]].to_dict())
print("flights per field:", ref.groupby("field")["acquisition_date"].agg(["nunique", "min", "max"]).to_string())
print("\nhow often each measured trait is filled:")
for col in ("in_situ_height", "in_situ_diameter", "in_situ_bbch_stage", "in_situ_head_diameter", "in_situ_harvested"):
    print(f"  {col:<24}{ref[col].notna().mean():.0%}")
print("\nBBCH values (dirty strings):", ref["in_situ_bbch_stage"].value_counts().head(20).to_dict())
print("crops with coverage < 0.9 (plant partly outside the orthomosaic):", int((ref["coverage_frac"] < 0.9).sum()))
""")

md(r"""
### Growth curves

Measured plant diameter against days after planting for the 2021 field. Each thin line is one plant (a sample of 60), and the dark line is the median per flight.
""")

code(r"""
f2 = ref[ref["field"] == "Field2"].copy()
f2["diameter"] = pd.to_numeric(f2["in_situ_diameter"], errors="coerce")
fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
for pid in RNG.choice(f2["plant_id"].unique(), 60, replace=False):
    p = f2[f2["plant_id"] == pid].sort_values("day_after_planting")
    ax.plot(p["day_after_planting"], p["diameter"], color="#b7d3f6", linewidth=1)
med = f2.groupby("day_after_planting")["diameter"].median()
ax.plot(med.index, med.values, color="#0d366b", linewidth=2, marker="o", markersize=4)
ax.text(med.index[-1], med.values[-1], "  median", va="center", color=MUTED_INK)
ax.set_xlabel("days after planting")
ax.set_ylabel("measured plant diameter")
ax.set_title("GrowliFlower 2021: plant diameter over the season (60 plants + median)", loc="left", fontweight="bold", color=INK)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color=GRID_COLOR, linewidth=0.8)
fig.savefig(CHART_DIR / "growth_growliflower_diameter.png", dpi=150)
plt.show()
""")

code(r"""
best = ref.assign(n_bbch=ref["in_situ_bbch_stage"].notna()).groupby("plant_id").agg(n=("filepath", "size"), b=("n_bbch", "sum"))
pid = best.sort_values(["n", "b"], ascending=False).index[0]
series = ref[ref["plant_id"] == pid].sort_values("acquisition_date")
print(series[["acquisition_date", "day_after_planting", "in_situ_bbch_stage", "in_situ_diameter", "in_situ_head_diameter"]].to_string(index=False))
fig, axes = plt.subplots(1, len(series), figsize=(1.3 * len(series), 1.8), layout="constrained")
for ax, row in zip(axes, series.itertuples()):
    ax.imshow(Image.open(hf_hub_download(GF_REPO, row.filepath, repo_type="dataset")))
    ax.set_title(f"day {row.day_after_planting}", fontsize=7, color=MUTED_INK)
    ax.axis("off")
fig.suptitle(f"GrowliFlower: plant {pid} across {len(series)} flights (256×256 crops)", x=0.01, ha="left", fontweight="bold", color=INK, fontsize=9)
save_example(fig, "GrowliFlower: one plant over time")
plt.show()
""")

code(r"""
seg = [s for s in gf if s["task"] == "segmentation"]
print(f"segmentation patches: {len(seg):,}; with plants: {sum(s['has_plants'] for s in seg):,}")
print("plants per patch:", dict(sorted(Counter(len((s.get('plant_instances') or {}).get('detections', [])) for s in seg).items())))
print("leaf instances:", sum(len((s.get('leaf_instances') or {}).get('detections', [])) for s in seg),
      "| stem keypoints:", sum(len((s.get('stem_keypoints') or {}).get('keypoints', [])) for s in seg))
print("labels per patch: plant instance masks, leaf instance masks, stem keypoints, void (border-cut) mask")
print("defoliation subset:", len({s['plant_id'] for s in gf if s['task'] == 'defoliation'}), "plants, with a before/after image pair on the day the leaves were removed")
""")

# ---------------------------------------------------------------- 9 mango
md(r"""
## 9. General purpose (fruit): Mango growth `Project-AgML/mango_growth_detection`

Kabir et al., *Smartphone image dataset for machine learning-based monitoring and analysis of mango growth stages*, Data in Brief 61 (2025); Mendeley doi:10.17632/5snwpzdtzs.1. **License: CC BY 4.0.**

**Intended task:** detection of 4 fruit stages `Early-Fruit` → `Premature` → `Mature` → `Ripe`. It is a non-berry reference for fruit-stage detection. It turns out to be a **Roboflow export** (file names carry `.rf.<hash>`).
""")

code(r"""
MG_REPO = "Project-AgML/mango_growth_detection"
mg_names = card_class_names(MG_REPO)
mg = scan_parquet(MG_REPO, ["objects"])
MG_STAGE = {"Early-Fruit": "green (unripe)", "Premature": "white / pale", "Mature": "turning", "Ripe": "ripe"}
mg_counts = Counter(mg_names[c] for r in mg for c in r["objects"]["categories"])
plot_stage_bars(mg_counts, MG_STAGE.get, f"Mango growth (AgML): boxes per stage ({sum(mg_counts.values()):,} boxes)", "growth_mango_stages.png")
STAGE_BOXES["Mango growth (AgML)"] = Counter({MG_STAGE[k]: v for k, v in mg_counts.items()})
describe_sizes([r["width"] for r in mg], [r["height"] for r in mg], [r["mode"] for r in mg])
BOX_STATS["Mango growth (AgML)"] = box_stats(
    [(w, h) for r in mg for _, _, w, h in r["objects"]["bbox"]],
    [(r["width"], r["height"]) for r in mg for _ in r["objects"]["bbox"]],
    [len(r["objects"]["bbox"]) for r in mg])
print_stats(BOX_STATS["Mango growth (AgML)"])
""")

code(r"""
mg_stems = Counter(stem(r["path"]) for r in mg)
print(f"{len(mg):,} images come from {len(mg_stems):,} source photos; copies per photo: {dict(Counter(mg_stems.values()))}")
prefix = Counter((re.match(r"[A-Za-z\-]+", r["path"]).group(0), tuple(sorted({mg_names[c] for c in r["objects"]["categories"]}))) for r in mg)
print("file-name prefix -> classes in the image:", dict(prefix))
print("No split column: a random split would put copies of the same photo in train and test. Split by source photo (stem).")
""")

md(r"""
⚠️ **Stage is confounded with background:** all `Ripe` images (prefix `RS`) are harvested fruit photographed on white paper, while `Early-Fruit`, `Premature` and `Mature` are on the tree. A detector can learn "paper vs foliage" instead of ripeness. Some images also have baked-in rotation borders (augmented copies), and some fruits on the tree are not boxed.
""")

code(r"""
pick = sorted(RNG.choice(len(mg), 6, replace=False))
mg_imgs = parquet_images(MG_REPO, pick)
show_samples([(mg_imgs[i], [(*b, mg_names[c]) for b, c in zip(mg[i]["objects"]["bbox"], mg[i]["objects"]["categories"])],
               f"{stem(mg[i]['path'])[:34]}: {mg_names[mg[i]['objects']['categories'][0]]}") for i in pick],
             lambda name: STAGE_COLORS[MG_STAGE[name]], "Mango growth (AgML): samples")
""")

# ---------------------------------------------------------------- 10 leaf counting
md(r"""
## 10. General purpose: Leaf counting `Project-AgML/leaf_counting_denmark`

Field seedlings (mixed crop and weed species, grasses and broadleaf) cropped one plant per image and labeled with the **number of leaves** (`1`…`8`, `9+`). Leaf count is the standard proxy for early growth stage (BBCH 10–19). The card has **no description, citation or license**. It matches the Aarhus University leaf-counting data (Teimouri et al., 2018), but that is *unverified*.

**Intended task:** image classification (or regression) of leaf count.
""")

code(r"""
LC_REPO = "Project-AgML/leaf_counting_denmark"
lc_names = card_class_names(LC_REPO, column="label")
lc = scan_parquet(LC_REPO, ["label"])
lc_counts = Counter(lc_names[r["label"]] for r in lc)
print(f"{len(lc):,} images; per class: {dict(sorted(lc_counts.items()))}")
print("file names:", [r["path"] for r in lc[:5]], "(numbers only: no species, plot or date)")
describe_sizes([r["width"] for r in lc], [r["height"] for r in lc], [r["mode"] for r in lc])
print("file names reused across classes:", sum(v > 1 for v in Counter(r["path"] for r in lc).values()),
      "(each class folder was numbered from 0, so names repeat; these are different images)")
""")

code(r"""
import hashlib

digests = defaultdict(set)
for path in parquet_files(LC_REPO):
    table = pq.read_table(path, columns=["image", "label"])
    for image, label in zip(table.column("image").to_pylist(), table.column("label").to_pylist()):
        digests[hashlib.md5(image["bytes"]).hexdigest()].add((image["path"], label))
dupes = {h: v for h, v in digests.items() if len(v) > 1}
print(f"byte-identical images: {len(dupes)} groups, {sum(len(v) for v in dupes.values())} images")
print("groups with conflicting leaf-count labels:", [sorted(v) for v in dupes.values() if len({l for _, l in v}) > 1])
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
order = lc_names
ax.bar(range(len(order)), [lc_counts[n] for n in order], color="#2a78d6", width=0.7, edgecolor=SURFACE, linewidth=2)
for i, n in enumerate(order):
    ax.text(i, lc_counts[n], f"{lc_counts[n]:,}", ha="center", va="bottom", fontsize=8, color=MUTED_INK)
ax.set_xticks(range(len(order)), order)
ax.set_xlabel("leaves per plant (label)")
ax.set_ylabel("images")
ax.set_title(f"Leaf counting (AgML): images per leaf count ({len(lc):,} images)", loc="left", fontweight="bold", color=INK)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
ax.set_axisbelow(True)
fig.savefig(CHART_DIR / "growth_leaf_counting_classes.png", dpi=150)
plt.show()

pick = {}
for i, r in enumerate(lc):
    if len([k for k in pick if lc[k]["label"] == r["label"]]) < 2:
        pick[i] = r["label"]
lc_imgs = parquet_images(LC_REPO, pick)
fig, axes = plt.subplots(2, 9, figsize=(12, 3), layout="constrained")
for col, label in enumerate(range(9)):
    for row, i in enumerate([k for k, v in pick.items() if v == label]):
        axes[row][col].imshow(lc_imgs[i])
        axes[row][col].set_title(f"{lc_names[label]} leaves" if row == 0 else "", fontsize=8, color=MUTED_INK)
for ax in axes.ravel():
    ax.axis("off")
fig.suptitle("Leaf counting: two examples per label", x=0.01, ha="left", fontweight="bold", color=INK)
save_example(fig, "Leaf counting: examples per label")
plt.show()
""")

# ---------------------------------------------------------------- tomato & pepper
md(r"""
# Tomato and pepper

Sections 11–14 add fruit-stage datasets for **tomato** and **sweet pepper**. Their classes are mapped onto the same common fruit-stage scale as the berries. Tomato "breaker" (first color change) is placed on `white / pale`, the step where a fruit stops being fully green.
""")

# ---------------------------------------------------------------- 11 laboro tomato
md(r"""
## 11. Tomato ripeness (Laboro Tomato): `Project-AgML/tomato_ripeness_detection`

Laboro Tomato, Laboro.AI (2020), github.com/laboroai/LaboroTomato, re-hosted by AgML from the Kaggle copy `nexuswho/laboro-tomato`. ⚠️ **License: the AgML card states none; the original Laboro release is CC BY-NC 4.0 (non-commercial).**

**Intended task:** detection (+ instance segmentation in the original) of tomatoes in 3 ripeness stages × 2 fruit sizes: `b_` = big (normal) tomatoes, `l_` = little (cherry) tomatoes; `green` → `half_ripened` → `fully_ripened`. Greenhouse phone photos.
""")

code(r"""
LT_REPO = "Project-AgML/tomato_ripeness_detection"
lt_names = card_class_names(LT_REPO)
lt = scan_parquet(LT_REPO, ["objects"])
print("classes:", lt_names)
print("EXIF orientation tags:", dict(Counter(r["exif_orientation"] for r in lt)))
LT_STAGE = {n: "green (unripe)" if n.endswith("green") else "turning" if "half" in n else "ripe" for n in lt_names}
lt_counts = Counter(lt_names[c] for r in lt for c in r["objects"]["categories"])
plot_stage_bars(lt_counts, LT_STAGE.get, f"Tomato ripeness (Laboro): boxes per class ({sum(lt_counts.values()):,} boxes, {len(lt)} images)",
                "growth_tomato_laboro_classes.png")
STAGE_BOXES["Tomato ripeness (Laboro)"] = Counter({LT_STAGE[k]: v for k, v in lt_counts.items()})
""")

md(r"""
⚠️ **For the images with an EXIF "rotate 90°" tag (counted above), the pixels are stored landscape (4032×3024) but the boxes were drawn on the rotated (portrait) photo.** Drawn on the raw pixels, the boxes miss the fruit. Apply the EXIF orientation first (`ImageOps.exif_transpose`). Many training loaders ignore EXIF, so fix this when exporting the data. The next cell counts boxes that fall outside the image in each frame.
""")

code(r"""
def upright(r):
    # (W, H) of the photo after applying the EXIF orientation: tags 5-8 swap width and height
    return (r["height"], r["width"]) if r["exif_orientation"] in (5, 6, 7, 8) else (r["width"], r["height"])


def outside(boxes, W, H, tol=2):
    return sum(x < -tol or y < -tol or x + w > W + tol or y + h > H + tol for x, y, w, h in boxes)


print("boxes outside the raw (stored) frame:", sum(outside(r["objects"]["bbox"], r["width"], r["height"]) for r in lt))
print("boxes outside the EXIF-rotated frame:", sum(outside(r["objects"]["bbox"], *upright(r)) for r in lt))
describe_sizes([upright(r)[0] for r in lt], [upright(r)[1] for r in lt], [r["mode"] for r in lt])
BOX_STATS["Tomato ripeness (Laboro)"] = box_stats(
    [(w, h) for r in lt for _, _, w, h in r["objects"]["bbox"]],
    [upright(r) for r in lt for _ in r["objects"]["bbox"]],
    [len(r["objects"]["bbox"]) for r in lt])
print_stats(BOX_STATS["Tomato ripeness (Laboro)"])
lt_dates = Counter(m.group(1) if (m := re.match(r"IMG_(\d{8})_", r["path"])) else "no date" for r in lt)
print("capture date in file name:", dict(lt_dates), "| no split column")
""")

code(r"""
pick = [next(i for i in RNG.permutation(len(lt)) if c in lt[i]["objects"]["categories"]) for c in (2, 1, 0, 5, 4, 3)]
lt_imgs = parquet_images(LT_REPO, pick, transpose=True)
show_samples([(lt_imgs[i], [(*b, lt_names[c]) for b, c in zip(lt[i]["objects"]["bbox"], lt[i]["objects"]["categories"])],
               f"{lt[i]['path']}: {', '.join(sorted({lt_names[c] for c in lt[i]['objects']['categories']}))}") for i in pick],
             lambda name: STAGE_COLORS[LT_STAGE[name]], "Tomato ripeness (Laboro): samples, EXIF applied")
""")

# ---------------------------------------------------------------- 12 agrob
md(r"""
## 12. Greenhouse tomato (AgRobTomato): `Project-AgML/Greenhouse_Tomato_AgRob_object_detection`

Moreira et al., *Benchmark of Deep Learning and a Proposed HSV Colour Space Models for the Detection and Classification of Greenhouse Tomato*, Agronomy 12 (2022); dataset Zenodo doi:10.5281/zenodo.5596799. **License: CC BY 4.0.**

**Intended task:** detection of tomatoes in 4 USDA colour stages (`unripe`, `breaking`, `reddish`, `ripe`) from a **robot** (AgRob v16, ZED camera) driving along greenhouse rows in Portugal, August 2020. Frames sampled every 3 s, so neighbouring frames overlap. Each box has an `occluded` flag.
""")

code(r"""
AR_REPO = "Project-AgML/Greenhouse_Tomato_AgRob_object_detection"
ar_names = card_class_names(AR_REPO)
ar = scan_parquet(AR_REPO, ["objects", "capture_session"])
AR_STAGE = {"unripe": "green (unripe)", "breaking": "white / pale", "reddish": "turning", "ripe": "ripe"}
ar_counts = Counter(ar_names[c] for r in ar for c in r["objects"]["categories"])
plot_stage_bars(ar_counts, AR_STAGE.get, f"Greenhouse tomato (AgRob): boxes per class ({sum(ar_counts.values()):,} boxes, {len(ar)} images)",
                "growth_tomato_agrob_classes.png")
STAGE_BOXES["Greenhouse tomato (AgRob)"] = Counter({AR_STAGE[k]: v for k, v in ar_counts.items()})
print("images per capture session:", dict(Counter(r["capture_session"] for r in ar)))
occ = [o for r in ar for o in r["objects"]["occluded"]]
print(f"occluded boxes: {sum(occ):,} of {len(occ):,} ({np.mean(occ):.0%})")
describe_sizes([r["width"] for r in ar], [r["height"] for r in ar], [r["mode"] for r in ar])
print("boxes outside the image:", sum(outside(r["objects"]["bbox"], r["width"], r["height"]) for r in ar))
BOX_STATS["Greenhouse tomato (AgRob)"] = box_stats(
    [(w, h) for r in ar for _, _, w, h in r["objects"]["bbox"]],
    [(r["width"], r["height"]) for r in ar for _ in r["objects"]["bbox"]],
    [len(r["objects"]["bbox"]) for r in ar])
print_stats(BOX_STATS["Greenhouse tomato (AgRob)"])
""")

code(r"""
pick = [next(i for i in RNG.permutation(len(ar)) if c in ar[i]["objects"]["categories"]) for c in (3, 2, 1)]
pick += [i for i in RNG.permutation(len(ar)) if i not in pick][:3]
ar_imgs = parquet_images(AR_REPO, pick)
show_samples([(ar_imgs[i], [(*b, ar_names[c]) for b, c in zip(ar[i]["objects"]["bbox"], ar[i]["objects"]["categories"])],
               f"{ar[i]['path'][-26:]} ({len(ar[i]['objects']['bbox'])} boxes)") for i in pick],
             lambda name: STAGE_COLORS[AR_STAGE[name]], "Greenhouse tomato (AgRob): samples")
""")

md(r"""
**Gotchas:** 92% of boxes are `unripe`; `ripe` has very few boxes, so the late stages are thin. Consecutive frames (3 s apart) show the same fruits: split by capture session or by frame blocks, not at random. File names carry the frame number, which gives frame order but not growth over time (one day only).
""")

# ---------------------------------------------------------------- 13 tomato maturity
md(r"""
## 13. Tomato maturity (classification): `Project-AgML/tomato_maturity_classification`

Khatun et al., *An extensive real-world in field tomato image dataset involving maturity classification and recognition of fresh and defect tomatoes*, Data in Brief 51 (2023); Mendeley doi:10.17632/s42kpg8h37.1. **License: CC BY 4.0.**

**Intended task:** image classification `Immature` vs `Mature`. The `raw` config (1,000 images) is used; `augmented` (4,000) is generated copies and is skipped.
""")

code(r"""
TM_REPO = "Project-AgML/tomato_maturity_classification"
tm_names = card_class_names(TM_REPO, column="label")
tm = scan_parquet(TM_REPO, ["label"], folder="raw")
print(f"{len(tm):,} images; per class: {dict(Counter(tm_names[r['label']] for r in tm))}")
describe_sizes([r["width"] for r in tm], [r["height"] for r in tm], [r["mode"] for r in tm])
print("EXIF orientation tags:", dict(Counter(r["exif_orientation"] for r in tm)))
dates = Counter(re.search(r"IMG_(\d{8})", r["path"]).group(1) if re.search(r"IMG_(\d{8})", r["path"]) else "none" for r in tm)
print(f"capture dates in file names: {len(dates)} days, {min(dates)} to {max(dates)}")
by_class = defaultdict(set)
for r in tm:
    m = re.search(r"IMG_(\d{8})", r["path"])
    by_class[tm_names[r["label"]]].add(m.group(1) if m else "none")
print("days shared by both classes:", len(by_class["Immature"] & by_class["Mature"]))
""")

md(r"""
**Gotchas:** `Mature` is not only red fruit. The samples show yellow/orange and yellow-green (turning) tomatoes labeled `Mature`, so the two classes are roughly "green" vs "turning or ripe". Capture days are mostly split by class (most days hold only one class, counted above), so day-specific light and background can leak into the label: split by capture day. Varied image sizes; in-field phone photos (Bangladesh, per the paper).
""")

code(r"""
pick = [i for k in (0, 1) for i in [j for j in RNG.permutation(len(tm)) if tm[j]["label"] == k][:3]]
tm_imgs = parquet_images(TM_REPO, pick, folder="raw", transpose=True)
show_samples([(tm_imgs[i], [], f"{tm_names[tm[i]['label']]}: {tm[i]['path']}") for i in pick], None,
             "Tomato maturity (Khatun): samples")
""")

# ---------------------------------------------------------------- 14 sweet pepper
md(r"""
## 14. Sweet pepper maturity: Harel et al., Mendeley `ttntwwxkfd`

Harel, van Essen, Parmet & Edan, *Viewpoints analysis for sweet peppers maturity classification* (Biosystems Engineering, 2020 data paper); Mendeley Data doi:10.17632/ttntwwxkfd.1. **License: CC BY 4.0.** Not on Hugging Face; the full zip (1.5 GB, 6.4 GB unpacked, BMP files) is downloaded to `data/sweet_pepper_maturity/` and read directly from the zip, without extracting it.

**Intended task:** classification of **maturity class 1–4** (1 = fully green → 4 = fully coloured) for red and yellow sweet peppers, from several viewpoints of the **same pepper**. Three subsets: *Photocell* (harvested pepper in a lit box, 4 views incl. bottom), *Robotic* (pepper hung on a plant in the lab, RealSense RGB-D from a robot arm, classes 2–4 only), *Orientation* (same peppers rotated 3×). It is the only pepper-ripeness set found that can be downloaded; no detection boxes, no field images.
""")

code(r"""
import zipfile

SP_ZIP = zipfile.ZipFile(DATA_DIR / "sweet_pepper_maturity" / "ttntwwxkfd-1.zip")
PHOTOCELL = re.compile(r"Photocell Data/(Red|Yellow)/([RY])_L(\d)_P(\d+)_S(\d)")
COLOR = re.compile(r"(Robotic|Orientation) [Dd]ata/(Red|Yellow)/Color/color_pepper(\d+)\D.*_viewpoint_(\d)_class_(\d)")
sp = []
for name in SP_ZIP.namelist():
    if m := PHOTOCELL.search(name):
        sp.append({"subset": "Photocell", "color": m[1], "pepper": f"{m[2]}L{m[3]}P{m[4]}", "maturity": int(m[3]), "view": int(m[5]), "file": name})
    elif m := COLOR.search(name):
        sp.append({"subset": m[1], "color": m[2], "pepper": f"{m[2][0]}{m[3]}", "maturity": int(m[5]), "view": int(m[4]), "file": name})
sp = pd.DataFrame(sp)
print(f"{len(sp):,} colour images; depth images (Robotic/Orientation) not counted")
print(pd.crosstab([sp["subset"], sp["color"]], sp["maturity"]).to_string())
print("\npeppers per subset:", sp.groupby("subset")["pepper"].nunique().to_dict(), "| images per pepper (median):",
      sp.groupby(["subset", "pepper"]).size().groupby("subset").median().to_dict())
print("file-name copies with '(n)' suffix:", int(sp["file"].str.contains(r"\(\d\)").sum()))
""")

code(r"""
SP_STAGE = {1: "green (unripe)", 2: "turning", 3: "turning", 4: "ripe"}
sp_counts = Counter(f"class {m}" for m in sp["maturity"])
plot_stage_bars(sp_counts, lambda n: SP_STAGE[int(n[-1])], f"Sweet pepper maturity: images per class ({len(sp):,} colour images)",
                "growth_sweet_pepper_classes.png", unit="images")
sizes = Counter()
for f in sp.groupby("subset")["file"].first():
    with Image.open(io.BytesIO(SP_ZIP.read(f))) as img:
        sizes[(f.split("/")[1], img.size, img.mode)] += 1
print("image size per subset (first file):", list(sizes))
pick = [sp[(sp["subset"] == s) & (sp["maturity"] == m)].iloc[0] for s, m in (("Photocell", 1), ("Photocell", 2), ("Photocell", 4),
                                                                              ("Robotic", 2), ("Robotic", 4), ("Orientation", 3))]
show_samples([(Image.open(io.BytesIO(SP_ZIP.read(r["file"]))), [], f"{r['subset']} {r['color'].lower()} pepper, class {r['maturity']}")
              for r in pick], None, "Sweet pepper maturity: samples")
""")

md(r"""
**Gotchas:** maturity is labeled **per pepper**, not per image, so all views of one pepper share a label: split by pepper id, never by image. The *Robotic* and *Orientation* subsets reuse the same physical peppers. Classes 2 and 3 both map to `turning` on the common scale. Lab backgrounds only (black box, or a plant in the lab), so it does not transfer to field detection by itself.
""")

# ---------------------------------------------------------------- hydroponics
md(r"""
# Hydroponics

Datasets of plants grown **hydroponically**. All public ones found are **lettuce**; none for berries, tomato or pepper. Lettuce has no fruit, so these are not mapped to the fruit-stage scale: the growth signal is **biomass and size** (fresh weight, diameter, leaf area) or days since transplant.

Found but not loadable here: IEEE DataPort *Butterhead lettuce growth in hydroponic system* (doi:10.21227/d9g8-3w83) and *Hydroponic lettuce growth* (doi:10.21227/askr-ed22; aeroponics, NFT and deep-water culture), both need an IEEE login. Roboflow `hidroponic/hidroponic_final` (2,258 images; lettuce seedling / ready / sick + basil diseases) and `kdkd-psumx/lettuce-checker` (197 images, 4 lettuce stages) need a fork, and the Roboflow quota is used up.
""")

# ---------------------------------------------------------------- 15 AGC lettuce
md(r"""
## 15. Hydroponic lettuce (Autonomous Greenhouse Challenge): 4TU doi:10.4121/15023088

*3rd Autonomous Greenhouse Challenge: Online Challenge Lettuce Images*, Wageningen University & Research, Bleiswijk (NL), 2021. **License: CC BY 4.0.** Zip (886 MB) in `data/agc_lettuce/`, read directly from the zip.

**Intended task:** **regression of crop traits from RGB-D images**. The ReadMe states the lettuce was *"harvested from a hydroponic cultivation system"*. Plants of 4 varieties were sampled every 7 days, photographed from above (RealSense D415, 1080×1920, RGB + aligned depth) and then destructively measured: fresh weight, dry weight, height, diameter, leaf area.
""")

code(r"""
AGC_ZIP = zipfile.ZipFile(DATA_DIR / "agc_lettuce" / "OnlineChallenge.zip")
agc_gt = json.loads(AGC_ZIP.read("GroundTruth/GroundTruth_All_388_Images.json"))
agc = pd.DataFrame(agc_gt["Measurements"]).T
for col in ("FreshWeightShoot", "DryWeightShoot", "Height", "Diameter", "LeafArea"):
    agc[col] = pd.to_numeric(agc[col])
print(f"{len(agc)} plants, one RGB + one depth image each; keys per plant: {list(agc.columns)}")
print("plants per variety:", agc["Variety"].value_counts().to_dict())
print("'General' block:", agc_gt["General"] if len(str(agc_gt["General"])) < 600 else str(agc_gt["General"])[:600])
print("missing values per trait:", agc.isna().sum().to_dict())
agc[["FreshWeightShoot", "DryWeightShoot", "Height", "Diameter", "LeafArea"]].describe().round(1)
""")

code(r"""
VARIETY_COLORS = dict(zip(["Aphylion", "Salanova", "Satine", "Lugano"], ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
for variety, color in VARIETY_COLORS.items():
    d = agc[agc["Variety"] == variety]
    ax.scatter(d["Diameter"], d["FreshWeightShoot"], s=14, color=color, alpha=0.8, edgecolor=SURFACE, linewidth=0.5)
    top = d.loc[d["FreshWeightShoot"].idxmax()]
    ax.text(top["Diameter"], top["FreshWeightShoot"], f"  {variety}", va="center", fontsize=8, color=MUTED_INK)
ax.set_xlabel("measured diameter (cm)")
ax.set_ylabel("fresh weight of shoot (g)")
ax.set_title(f"Hydroponic lettuce (AGC 2021): fresh weight vs diameter, {len(agc)} plants", loc="left", fontweight="bold", color=INK)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color=GRID_COLOR, linewidth=0.8)
fig.savefig(CHART_DIR / "growth_agc_lettuce_weight.png", dpi=150)
plt.show()
""")

code(r"""
order = agc.sort_values("FreshWeightShoot")
pick = [order.iloc[int(q * (len(order) - 1))] for q in (0, 0.2, 0.4, 0.6, 0.8, 1)]
samples = []
for r in pick:
    with Image.open(io.BytesIO(AGC_ZIP.read(f"RGBImages/{r['RGB_Image']}"))) as img:
        samples.append((img.convert("RGB"), [], f"{r['Variety']}: {r['FreshWeightShoot']:.0f} g, {r['Diameter']:.0f} cm"))
        size, mode = img.size, img.mode
with Image.open(io.BytesIO(AGC_ZIP.read(f"DepthImages/{pick[0]['Depth_Information']}"))) as depth:
    print(f"RGB {size} {mode}; depth {depth.size} {depth.mode}")
show_samples(samples, None, "Hydroponic lettuce (AGC 2021): from lightest to heaviest")
""")

md(r"""
**Gotchas:** each plant appears **once** (destructive measurement), so there is no per-plant time series, and **the sampling week is not in the ground truth**: growth stage has to be inferred from size. Plants were photographed on a fixed imaging rig (white trays on a frame, people's legs at the edges), not in the growing gutters, so a model will not transfer directly to cameras in the hydroponic system. Lighting treatments were applied but are not linked to the images. One camera, top view only.
""")

# ---------------------------------------------------------------- 16 HydroGrowNet
md(r"""
## 16. Hydroponic lettuce (HydroGrowNet): Mendeley `g6cm3v3wdp` v5

Shalash et al., *HydroGrowNet of Batavia Dataset* (Eng. Appl. of AI, 2025, doi:10.1016/j.engappai.2025.111214). **License: CC BY 4.0.** Batavia lettuce in an **NFT** (nutrient film technique) system, three consecutive 30-day experiments in 2024. A rail camera rig photographed every plant daily, and sensors logged pH, EC, TDS, water temperature, air temperature, humidity and CO₂. Over 390,000 images in total, released **after YOLOv8 segmentation** (crops of single plants).

Only `Month2.zip` (934 MB, the smallest of the three) plus the sensor spreadsheets are downloaded to `data/hydrogrownet/`. Month 1 (2.7 GB) and Month 3 (0.9 GB) are skipped.
""")

code(r"""
HG_DIR = DATA_DIR / "hydrogrownet"
HG_ZIP = zipfile.ZipFile(HG_DIR / "Month2.zip")
HG_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})/(cam\d)=(\d{2}-\d{2}-\d{2})_segment_(\d+)\.png$")
hg = pd.DataFrame([dict(zip(("date", "camera", "time", "segment"), m.groups())) | {"file": n}
                   for n in HG_ZIP.namelist() if (m := HG_NAME.search(n))])
files = [n for n in HG_ZIP.namelist() if not n.endswith("/")]
print(f"{len(hg):,} segmented crops; files not matching the name pattern: {len(files) - len(hg)}; "
      f"date folders: {sum(n.endswith('/') for n in HG_ZIP.namelist())} (some empty)")
print(f"{hg['date'].nunique()} capture days from {hg['date'].min()} to {hg['date'].max()}; cameras: {hg['camera'].value_counts().to_dict()}")
print("segments per camera frame:", dict(Counter(hg.groupby(['date', 'camera', 'time']).size())))
per_day = hg.groupby("date").size()
print("crops per day min / median / max:", per_day.min(), int(per_day.median()), per_day.max())
sizes = Counter()
for f in hg["file"].iloc[::500]:
    with Image.open(io.BytesIO(HG_ZIP.read(f))) as img:
        sizes[(img.size, img.mode)] += 1
print("crop sizes (every 500th file):", sizes.most_common(5))
""")

code(r"""
import xml.etree.ElementTree as ET

# sheet names of the measurement workbooks (read with the standard library: openpyxl is not installed)
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
sensor_zip = zipfile.ZipFile(HG_DIR / "all_months_sensory_data.zip")
for name in sensor_zip.namelist():
    book = zipfile.ZipFile(io.BytesIO(sensor_zip.read(name)))
    sheets = [s.get("name") for s in ET.fromstring(book.read("xl/workbook.xml")).find("m:sheets", NS)]
    print(f"{name}:\n   {sheets}")
""")

code(r"""
days = sorted(hg["date"].unique())
pick_days = [days[int(q * (len(days) - 1))] for q in (0, 0.2, 0.4, 0.6, 0.8, 1)]
samples = []
for d in pick_days:
    r = hg[hg["date"] == d].sample(1, random_state=0).iloc[0]
    samples.append((Image.open(io.BytesIO(HG_ZIP.read(r["file"]))).convert("RGB"), [], f"{d} {r['camera']} {r['time']} seg {r['segment']}"))
show_samples(samples, None, "HydroGrowNet (Month 2): one random crop per date")
""")

md(r"""
**Gotchas:** each "segment" is a full 640×480 camera frame with **everything masked black except one plant** (the YOLOv8 mask), and plants are often cut by the frame border; some frames have a strong gray/blue colour cast. File names hold only date, camera, time and segment index: **there is no plant id**, so the same plant cannot be followed across days, and the crops carry **no labels** (no stage, no weight). Images can be linked to the sensor logs by timestamp only. Capture is irregular (gaps between dates, from a few hundred to a few thousand crops per day). The plant measurements are hand-made spreadsheets (seedling and harvest sheets, irregular headers, dimensions written as `24*27`), not a per-image table. Useful as unlabeled pre-training data or for population-level growth curves, not for supervised stage training as is.
""")

# ---------------------------------------------------------------- cucumber
md(r"""
# Cucumber

No public **image** dataset with cucumber growth or fruit-stage labels was found. Found and not loaded: HF `odrinari/Greenhouse-cucumber-and-tomato-plants` (1,072 raw greenhouse photos of cucumber and tomato plants with flowers and green fruit, **no labels, no license**); Roboflow `cnuvia/cucumber-9fjp3` (117 images, keypoints, `cucumber` / `immature`) and the cucumber harvest set of Riou et al. 2020 (arXiv 2011.02719, 555 images, harvest-ready fruit only; no download link found). A Norwegian cucumber canopy-growth dataset (arXiv 2609.10749) is announced as "public upon publication".

What exists is **crop growth and yield as time series** from a greenhouse experiment (below).
""")

# ---------------------------------------------------------------- 17 AGC 2018 cucumber
md(r"""
## 17. Cucumber crop time series: Autonomous Greenhouse Challenge 2018 (`koidra/GH_Cucumber`)

Hemming et al., *Remote Control of Greenhouse Vegetable Production with Artificial Intelligence—Greenhouse Climate, Irrigation, and Crop Production*, Sensors 19 (2019); dataset 4TU doi:10.4121/uuid:e4987a7b-04dd-4c89-9b18-883aad30ba9a, **license CC0**. Read from the HF mirror `koidra/GH_Cucumber` (50 MB, CSV; the mirror has no card or license of its own).

**What it is:** cucumber (cv. Hi Power) grown for about 4 months in 6 glasshouse compartments in Bleiswijk (NL), 2018: 5 AI teams controlled climate and irrigation remotely, plus a reference compartment run by growers. Per compartment: climate (5-min), irrigation, crop management, and **daily production** (cumulative class A/B harvest in kg/m² and fruit counts). Team Sonoma (Microsoft Research) also contributed a crop table with weekly **stem length growth, leaves per stem and fruit growth**. **No images.**
""")

code(r"""
GC_ROOT = Path(snapshot_download("koidra/GH_Cucumber", repo_type="dataset")) / "GH_Cucumber"
AGC18 = GC_ROOT / "AutonomousGreenhouseChallenge2018"
teams = sorted(d.name for d in AGC18.iterdir() if d.is_dir())
print("compartments:", teams)
print("files per compartment:", sorted(f.name for f in (AGC18 / teams[0]).iterdir()))
prod = {}
for team in teams:
    p = pd.read_csv(AGC18 / team / "Production.csv")
    p["date"] = pd.to_datetime(p["time"], unit="D", origin="1899-12-30")  # Excel serial dates
    prod[team] = p
p0 = prod[teams[0]]
print(f"production rows per compartment: {dict((t, len(p)) for t, p in prod.items())}; {p0['date'].min():%Y-%m-%d} to {p0['date'].max():%Y-%m-%d}")
print("columns:", list(p0.columns))
clim = pd.read_csv(AGC18 / teams[0] / "Greenhouse_climate.csv", nrows=3)
print(f"climate columns ({teams[0]}):", list(clim.columns)[:20], "...")
pd.DataFrame({t: {"final cumulative harvest (kg/m²)": round(p["Total_Prod_cum"].iloc[-1], 1),
                  "class A share": round(p["ProdA_cum"].iloc[-1] / p["Total_Prod_cum"].iloc[-1], 2)} for t, p in prod.items()}).T
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
for team, p in prod.items():
    grower = team.startswith("Reference")
    ax.plot(p["date"], p["Total_Prod_cum"], color="#0d366b" if grower else "#86b6ef", linewidth=2.2 if grower else 1.3)
    ax.text(p["date"].iloc[-1], p["Total_Prod_cum"].iloc[-1], f"  {team}", va="center", fontsize=7, color=MUTED_INK)
ax.set_ylabel("cumulative harvest (kg/m²)")
ax.set_title("AGC 2018 cucumber: cumulative harvest per compartment (dark = growers)", loc="left", fontweight="bold", color=INK)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color=GRID_COLOR, linewidth=0.8)
ax.set_xlim(right=p0["date"].max() + pd.Timedelta(days=22))
fig.savefig(CHART_DIR / "growth_agc2018_cucumber_harvest.png", dpi=150)
plt.show()

crop = pd.read_csv(GC_ROOT / "MicrosoftResearch" / "crop.csv")
print(f"Sonoma crop table: {len(crop)} rows, columns {list(crop.columns)}")
print("non-empty values per column:", crop.notna().sum().to_dict())
""")

md(r"""
**Gotchas:** this is a **yield/crop-model** dataset, not computer vision: no images, so it cannot train a detector. Growth is measured per compartment (one value per day), not per plant. Biological traits (stem length growth, leaves per stem, fruit growth) exist only in the Sonoma table and are sparse (counts above). Useful as ground truth for growth/yield forecasting from climate and irrigation, or to pair with our own camera data. The second challenge edition (2019, cherry tomato, also CC0 on 4TU) has the same structure.
""")

# ---------------------------------------------------------------- cross dataset
md(r"""
## Cross-dataset comparison

### Stage coverage on the common scale

Each bar is one dataset, split by the share of boxes in each stage of the common scale. Where a dataset has no class for a stage, that stage is simply missing from its bar, so *gaps in the bar mean the stage is missing from the dataset*.
""")

code(r"""
names = list(STAGE_BOXES)
fig, ax = plt.subplots(figsize=(10, 0.55 * len(names) + 1.6), layout="constrained")
for row, name in enumerate(names):
    counts = STAGE_BOXES[name]
    total = sum(counts.values())
    left = 0
    for stage in STAGE_ORDER:
        share = counts.get(stage, 0) / total
        if share == 0:
            continue
        ax.barh(row, share, left=left, color=STAGE_COLORS[stage], height=0.62, edgecolor=SURFACE, linewidth=2)
        if share >= 0.07:
            ax.text(left + share / 2, row, f"{share:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if STAGE_COLORS[stage] in DARK_FILLS else INK)
        left += share
    ax.text(1.01, row, f"{total:,}", va="center", fontsize=8, color=MUTED_INK)
ax.set_yticks(range(len(names)), names)
ax.invert_yaxis()
ax.set_xlim(0, 1.08)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}" if v <= 1 else "")
ax.set_xlabel("share of boxes (number at the right = total boxes)")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("Fruit-stage coverage per dataset (common scale)", loc="left", fontweight="bold", color=INK)
fig.legend(handles=legend_handles(STAGE_COLORS, STAGE_ORDER), loc="outside upper left", ncol=6, frameon=False, fontsize=8)
fig.savefig(CHART_DIR / "growth_stage_coverage.png", dpi=150)
plt.show()
""")

md(r"""
### Object size

Median box area as a percentage of the image (dot), with the 10th–90th percentile range (line), on a log scale. Left = small objects. The text at the right is the share of tiny boxes (< 32×32 px) and the median boxes per image.
""")

code(r"""
stats = pd.DataFrame(BOX_STATS).T.sort_values("box area % median")
fig, ax = plt.subplots(figsize=(9, 0.45 * len(stats) + 1.4), layout="constrained")
for row, (name, s) in enumerate(stats.iterrows()):
    ax.plot([s["box area % p10"], s["box area % p90"]], [row, row], color="#86b6ef", linewidth=2, solid_capstyle="round")
    ax.plot(s["box area % median"], row, "o", color="#1c5cab", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.text(150, row, f"tiny {s['tiny (<32² px) %']:.0f}%  ·  {s['boxes/img median']:.0f} boxes/img", va="center", fontsize=8, color=MUTED_INK)
ax.set_xscale("log")
ax.set_xlim(0.003, 1500)
ax.set_xticks([0.01, 0.1, 1, 10, 100], ["0.01%", "0.1%", "1%", "10%", "100%"])
ax.set_yticks(range(len(stats)), stats.index)
ax.set_xlabel("box area as % of image (log)")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
ax.set_title("How small are the objects? (median and 10-90% range)", loc="left", fontweight="bold", color=INK)
fig.savefig(CHART_DIR / "growth_object_size.png", dpi=150)
plt.show()
stats.round(2)
""")

md(r"""
### Berry coverage matrix

Which berry × stage combinations exist in at least one dataset (box counts).
""")

code(r"""
BERRY_OF = {"Strawberry growth (AgML)": "strawberry", "Strawberry 5-stage (Roboflow)": "strawberry",
            "Strawberry ripeness (AgML, strawberry part)": "strawberry", "Blueberry maturity (AgML)": "blueberry",
            "Blueberry segmentation (Ni 2020)": "blueberry", "Raspberry (Roboflow)": "raspberry"}
matrix = pd.DataFrame(0, index=["strawberry", "blueberry", "raspberry"], columns=STAGE_ORDER)
for name, berry in BERRY_OF.items():
    for stage, n in STAGE_BOXES[name].items():
        matrix.loc[berry, stage] += n
matrix
""")

md(r"""
Caveats for the matrix: blueberry "green (unripe)" also contains pink/purple turning berries, because both blueberry datasets have only two classes. Raspberry "flower / bud" combines `Buds` (11.8k) and `Flowers` (4.7k).

### Time series and licenses
""")

code(r"""
pd.DataFrame([
    ("Strawberry growth (AgML)", "strawberry", "CC BY 4.0", "yes: 'tagged' fruits, 34 dates over 2 seasons, + size", "detection (7 stages)"),
    ("Strawberry 5-stage (Roboflow)", "strawberry", "CC BY 4.0", "partly: 10 time-lapse plants × 10-20 dates; cam timestamps", "detection (5 stages)"),
    ("Strawberry & avocado (AgML)", "strawberry, avocado", "CC BY 4.0", "no (25-day collection, no dates)", "detection (8 unnamed classes)"),
    ("Blueberry maturity (AgML)", "blueberry", "CC BY 4.0", "no", "detection / counting (2 used ids)"),
    ("Blueberry segmentation (Ni 2020)", "blueberry", "CC BY-NC-SA 4.0", "no", "instance segmentation (2 classes)"),
    ("Raspberry (Roboflow)", "raspberry", "CC BY 4.0", "no", "detection (4 stages)"),
    ("Chili plant growth (Roboflow)", "chili", "CC BY 4.0", "partly: chamber time-lapse, up to 45 dates per camera", "detection (4 plant stages)"),
    ("GrowliFlower", "cauliflower", "unknown", "YES: 739 plants × up to 15 flights + measured traits", "regression / segmentation"),
    ("Mango growth (AgML)", "mango", "CC BY 4.0", "no", "detection (4 stages)"),
    ("Leaf counting (AgML)", "field seedlings", "none declared", "no", "classification (leaf count)"),
    ("Tomato ripeness (Laboro)", "tomato", "CC BY-NC 4.0 (original)", "no", "detection (3 stages × 2 sizes)"),
    ("Greenhouse tomato (AgRob)", "tomato", "CC BY 4.0", "no (one day; video frames 3 s apart)", "detection (4 stages)"),
    ("Tomato maturity (Khatun)", "tomato", "CC BY 4.0", "no (dates in file names)", "classification (2 classes)"),
    ("Sweet pepper maturity (Harel)", "sweet pepper", "CC BY 4.0", "no (multi-view per pepper)", "classification (4 classes)"),
    ("Hydroponic lettuce (AGC 2021)", "lettuce", "CC BY 4.0", "no (destructive; week not recorded)", "regression (5 traits) from RGB-D"),
    ("HydroGrowNet", "lettuce", "CC BY 4.0", "partly: daily captures, no plant id", "unlabeled crops + sensor logs"),
    ("AGC 2018 cucumber", "cucumber", "CC0", "yes: daily production per compartment, ~4 months", "tabular (no images)"),
], columns=["dataset", "crop", "license", "time series", "task"])
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
nbf.write(nb, "growth.ipynb")
print("wrote", len(cells), "cells")
