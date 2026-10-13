"""Builds normalization/normalize_all.ipynb: one master notebook with a section per crop plus hydroponics.

Run from the repo root:
    python tools/build_normalized_nbs.py                                  # all crops + hydroponics
    python tools/build_normalized_nbs.py --crops tomato cucumber          # only these (hydroponics: name it as `hydroponics`)
    python tools/build_normalized_nbs.py --crops tomato --out normalization/normalize_tomato.ipynb
Execute it with `make normalize` (see Makefile / CLAUDE.md). The engine lives in tools/cvlib/; this file holds the notebook text and
the per-crop class maps and source mappings (the notebook cells).
"""
import argparse
import os

import nbformat as nbf

# ---------------------------------------------------------------- shared text

INTRO = r"""
# Normalized datasets

**Purpose:** build `normalized/<crop>/` for each crop (and `normalized/hydroponics_lettuce/`) from the sources surveyed in `plague.ipynb` and `growth.ipynb`.
**Inputs:** public datasets in the Hugging Face cache and `data/` (see CLAUDE.md). **Outputs:** `normalized/`, `examples/normalized/`. **Owner:** CreAI (GRZR-16).
**How to run:** from the repo root, `make normalize` (all crops, one kernel, in sequence) or `make normalize CROPS="tomato cucumber"`. Each crop section starts from scratch.

Each section builds one dataset merged from the surveyed sources. All labels use one class vocabulary. The goal is **correct labels, not many labels**: when a label cannot be trusted or reconciled, it is dropped, and this notebook says why and counts it.

```
normalized/<crop>/
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

# ---------------------------------------------------------------- section text of the shared loaders (code: tools/cvlib/loaders.py)

LOADERS = {
    'plantvillage': r"""
## PlantVillage: `BrandonFors/Plant-Diseases-PlantVillage-Dataset`

Lab photos of single leaves (CC0 in the HF metadata). Each image has one `Crop___condition` label, which goes straight to `condition`. Only classes of this crop are used. ⚠️ The images are lab photos on plain backgrounds, so do not validate a field model on them.
""",
    'plantseg': r"""
## PlantSeg v3: `XuVV/plantsegv3`

In-the-wild disease photos with **lesion polygons and boxes**. One disease per image, which goes to `condition`. Boxes become `lesion__<disease>` detections and polygons become `lesion__<disease>` segmentation, one line per polygon part. ⚠️ **CC BY-NC-ND 4.0** (non-commercial), and the images are web-scraped. There is no healthy class.
""",
    'plantdoc': r"""
## PlantDoc: `susnato/plant_disease_detection_processed`

In-the-wild leaf detection (CC BY 4.0, many web stock photos). Each box covers a **whole leaf** and gives its condition, so boxes become `leaf__<condition>`. The class names are not in the data; they come from the mapping recovered in `plague.ipynb`. Only the light columns and the `image` column are read, never `pixel_values`.
- Images that also contain boxes of another crop are dropped, because only some of their boxes could be labeled.
- The image-level `condition` is used only when every box in the image agrees (rule 1).
""",
    'strawberry12': r"""
## Strawberry_12: `Ethan0300/Strawberry_12-dataset` (MIT)

YOLO boxes for fruit, flower, 5 diseases, pests and 3 deficiencies (`data/strawberry12/`). Class names are not shipped; they come from file-name prefixes, as recovered in `plague.ipynb`.
- **Only original photos are used.** The `_blur/_bright/_dark/_noise` copies (57% of images) are dropped.
- Disease, pest and deficiency boxes enclose the **affected organ** (a whole moldy fruit, a leaf with mites), so they become `affected__<condition>`.
- `unripe fruit` is coarse (green + white + turning) and becomes `fruit__unripe`.
- The image `condition` comes from rule 1. Fruit and flower boxes are not conditions.
- Some "originals" are rotated augmented copies with black corners. Images with at least 2 black corners are dropped.
- ⚠️ Not fixable automatically: some "originals" are 2×2 mosaics, web images with text, or have black cut-out rectangles (see the round-trip figure).
""",
}

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


SETUP = r"""
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download, snapshot_download
from PIL import Image, ImageDraw, ImageOps

if Path.cwd().name == "normalization":  # the notebook lives in normalization/; every path below is relative to the repo root
    os.chdir("..")
sys.path.insert(0, "tools")

import cvlib.hydroponics as hydro
from cvlib.loaders import load_agml_detection, load_plantdoc, load_plantseg, load_plantvillage, load_roboflow, load_strawberry12
from cvlib.normalize import (DATA_DIR, DROPPED, IMAGES, LABEL_SPACE, NOTES, SHAPES, add_boxes, add_class, add_image, add_polygon, axis_classes, begin,
                             card_class_names, check, class_counts, condition_from_boxes, finalize, load_coco, parquet_files, parquet_rows,
                             print_split, rf_stem, show_mapping, show_roundtrip, summary)
"""

HELPERS_MD = "### Engine\n\n`tools/cvlib/normalize.py` holds the helpers. `add_image`, `add_class`, `add_boxes` and `add_polygon` are the only functions that write labels: they apply rules 5–7 above and record what they drop. `begin(crop, CLASSES)` wipes `normalized/<crop>/` and resets the state, so each crop section starts from scratch."

FINAL_MD = "#### Write, de-duplicate, self-check\n\n`finalize` writes every file. `check` reads them back from disk and asserts the format: every label file has an image, every coordinate is in [0, 1], and every class id is valid."

ROUNDTRIP_MD = "#### Round trip: labels read back from disk\n\nFor each source, sample images are drawn with the boxes and polygons **parsed from the written files**, so a bad conversion (swapped axes, wrong frame, wrong scale) is visible. Titles show the classification labels and the shape classes. The figures are saved to `examples/normalized/`."

HYDRO_INTRO = r"""
# Hydroponics: `normalized/hydroponics_lettuce/`

The hydroponic sources have **no boxes or polygons**, so the YOLO layout does not apply. The labeled one (AGC 2021) is a **regression** set: one top-view RGB + depth image per plant with its measured traits. The other (HydroGrowNet) has no labels and no plant id, so it is stored apart as **pre-training only**.

```
normalized/hydroponics_lettuce/
  images/                       AGC RGB (labeled), copied byte for byte
  depth/                        AGC aligned depth, same file name
  images_unlabeled/             HydroGrowNet Month 2 crops (no labels)
  labels/regression/traits.csv  image, variety, fresh_weight_g, dry_weight_g, height_cm, diameter_cm, leaf_area_cm2
  labels/regression/schema.json units, ranges, notes
  manifest.csv                  image, source, source_ref, license, width, height, group, dhash, role, label_space
```

Rules: a plant whose images or traits are missing or not positive is dropped and counted. `group` = plant for AGC (each plant appears once) and **capture day** for HydroGrowNet (neighbouring frames show the same plants): split by `group`. The sampling week is not recorded in AGC, and HydroGrowNet has no plant id, so neither supports a per-plant time series. AGC 2018 cucumber is tabular without images and stays in `growth.ipynb`.
"""


def crop_cells(crop, spec):
    cells = [nbf.v4.new_markdown_cell(f"# {spec['title'].capitalize()}\n\n" + VOCAB_MD.strip() + "\n\n" + spec["note"].strip()),
             nbf.v4.new_code_cell(f'CROP = "{crop}"\n' + spec["classes"].strip() + "\nbegin(CROP, CLASSES)\nprint_split()")]
    for loader, body in spec["sections"]:
        text = LOADERS.get(loader)
        if "@@" in body:
            text, body = body.split("@@")
        if text:
            cells.append(nbf.v4.new_markdown_cell(text.strip()))
        cells.append(nbf.v4.new_code_cell(body.strip()))
    cells += [nbf.v4.new_markdown_cell(FINAL_MD),
              nbf.v4.new_code_cell("manifest = finalize()\ncheck()\nsummary(manifest)\nprint()\nclass_counts()"),
              nbf.v4.new_markdown_cell(ROUNDTRIP_MD),
              nbf.v4.new_code_cell('for source in manifest["source"].unique():\n    show_roundtrip(source)')]
    return cells


def hydroponics_cells():
    return [nbf.v4.new_markdown_cell(HYDRO_INTRO.strip()),
            nbf.v4.new_code_cell("hydro.begin()\nhydro.build_agc_lettuce()      # 4TU doi:10.4121/15023088, CC BY 4.0"),
            nbf.v4.new_markdown_cell("HydroGrowNet (Mendeley `g6cm3v3wdp` v5, CC BY 4.0): Month 2 only is downloaded (35,043 crops). Copied unchanged into `images_unlabeled/`; this takes a few minutes."),
            nbf.v4.new_code_cell("hydro.build_hydrogrownet()"),
            nbf.v4.new_markdown_cell(FINAL_MD),
            nbf.v4.new_code_cell("hydro_manifest = hydro.finalize()\nhydro.check()\nhydro.summary(hydro_manifest)"),
            nbf.v4.new_markdown_cell("#### Round trip: traits read back from disk\n\nSix plants from the lightest to the heaviest; titles show the traits parsed from `traits.csv`. Saved to `examples/normalized/`."),
            nbf.v4.new_code_cell("hydro.show_roundtrip()")]


CROPS = {"strawberry": STRAWBERRY, "blueberry": BLUEBERRY, "raspberry": RASPBERRY, "tomato": TOMATO,
         "bell_pepper": BELL_PEPPER, "cucumber": CUCUMBER}


def build(selected, out):
    cells = [nbf.v4.new_markdown_cell(INTRO.strip()), nbf.v4.new_code_cell(SETUP.strip()), nbf.v4.new_markdown_cell(HELPERS_MD)]
    for name in selected:
        cells += hydroponics_cells() if name == "hydroponics" else crop_cells(name, CROPS[name])
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    nbf.write(nb, out)
    print("wrote", out, "with sections:", ", ".join(selected))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops", nargs="+", default=[*CROPS, "hydroponics"], choices=[*CROPS, "hydroponics"])
    parser.add_argument("--out", default="normalization/normalize_all.ipynb")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build(args.crops, args.out)
