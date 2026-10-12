# Plant Disease and Pest ("Plague") Datasets: Findings

> **Purpose:** handoff context for the team and for future Claude sessions. Scope: **overall (all-crop) disease and pest detection, plus berries** (strawberry, blueberry, raspberry). Everything was checked hands-on (2026-10-08; tomato and pepper sections #8–#10 and cucumber #11–#12 added 2026-10-12) unless marked *unverified*.
> **Working code:** `plague.ipynb` (one section per dataset; built by `tools/build_plague_nb.py`). **Charts:** `charts/plague_*.png`. **Image examples:** `examples/plague/*.png`.
> **Previous versions:** `archive/PLAGUE_DATASET_FINDINGS_v1.md` (full sections on the discarded PlantWild and Roboflow strawberry-disease sets) and `archive/PLAGUE_DATASET_FINDINGS_v2.md` (before the tomato/pepper extension). Companion study: `GROWTH_DATASET_FINDINGS.md`.

## TL;DR

| # | Dataset | Role | Task | Size | License | Verdict |
|---|---|---|---|---|---|---|
| 1 | PlantVillage `BrandonFors/Plant-Diseases-PlantVillage-Dataset` | general disease, lab reference | classification, 38 classes / 14 crops | 54,305 imgs | CC0 (metadata) | ✅ clean reference; poor field transfer |
| 2 | LeafNet2.0 `enalis/LeafNet2.0` | general disease, **largest** | classification, 182 classes / 37 crops | 255,990 imgs | gated, **non-commercial** | ✅ research only; **contains PlantVillage** |
| 3 | PlantSeg v3 `XuVV/plantsegv3` | general disease **in the wild + lesion masks**; berry diseases | segmentation / detection, 114 diseases / 34 crops | 11,458 imgs | **CC BY-NC-ND 4.0** | ✅ research only; only raspberry-disease source |
| 4 | PlantDoc `susnato/plant_disease_detection_processed` | general disease **detection**, commercial-friendly | detection, 24 classes | 2,324 imgs / 8,544 boxes | CC BY 4.0 | ✅ class names must be added (mapping below) |
| 5 | IP102 `Project-AgML/CT_PestNet_crop_pest_classification` | general **insect pests** | classification, 102 pests / 8 host crops | 75,222 imgs | CC BY 4.0 (AgML) | ✅ best pest source; not a field detector |
| 6 | CCMT `Project-AgML/crop_pest_disease_classification` | **field** pest damage + disease | classification, 18 classes / 4 crops | 25,170 imgs (raw) | CC BY 4.0 | ✅ real field photos; de-duplicate |
| 7 | Strawberry_12 `Ethan0300/Strawberry_12-dataset` | **strawberry** disease + pests + deficiencies | detection (YOLO), 12 classes | 3,906 imgs / 16,107 boxes | **MIT** | ⭐ best strawberry set; names recovered, 57% augmented copies |
| 8 | Tomato leaf disease detection `Project-AgML/tomato_leaf_disease_detection` | **tomato** field leaf diseases | detection, 7 classes | 689 imgs / 2,278 boxes | CC BY 4.0 | ⚠️ **class names missing** (only healthy known); Roboflow copies |
| 9 | Tomato disease detection `Project-AgML/tomato_disease_detection` | **tomato** disease on fruit, leaf, stem | detection, 10 classes | 1,026 imgs / 3,167 boxes | CC BY 4.0 | ✅ only set with boxes on diseased tomato **fruit** and stems |
| 10 | Chili leaves (COLD) `Project-AgML/COLD_chili_leaf_disease_classification` | **chili** disease + pest + deficiency | classification, 5 classes | 532 imgs (raw) | CC BY 4.0 | ⚠️ small; probably already inside LeafNet2.0 |
| 11 | Cucumber diseases `Project-AgML/cucumber_disease_classification` | **cucumber** leaf, fruit, stem diseases | classification, 8 classes | 1,289 originals + 6,400 augmented | CC BY 4.0 (Mendeley) | ✅ best cucumber set; **use originals only** |
| 12 | Cucumber leaf detection (Omarov et al.), Mendeley `mb3vnhpk8z` | **cucumber** leaf symptoms | detection, 2 classes (`color`, `damage`) | 500 imgs / 2,705 boxes | CC BY 4.0 | ⚠️ symptom-level only; video frames; light-confounded colour class |

**Discarded:** PlantWild (superseded by PlantSeg; see "Discarded" below), the Roboflow strawberry-disease set (not downloadable; Strawberry_12 covers it), AgML `tomato_leaf_disease` (a PlantVillage re-upload) and AgML `pepper_disease_pest_classification` (**the images are apple leaves**).

**What to use for what:**
- **General disease classification:** LeafNet2.0 (research) or PlantVillage + CCMT (commercial).
- **General disease detection / segmentation in the field:** PlantSeg (masks, research) and PlantDoc (boxes, commercial).
- **Pests:** IP102 for recognizing the insect, CCMT for recognizing the **damage** in the field, Strawberry_12 for strawberry pests in place.
- **Strawberry:** Strawberry_12 (detector), plus PlantVillage leaf-scorch/healthy as clean extra data.
- **Blueberry / raspberry diseases:** only PlantSeg (5 blueberry and 4 raspberry diseases, 18–47 images each). Too thin to train on; this is the main data gap.
- **Tomato:** plenty of leaf classification data already in PlantVillage (18,160) and LeafNet2.0 (28,816). For **detection** in the field: Liu #9 (fruit/leaf/stem, 3 diseases) and PlantDoc; Imtiaz #8 only after its class names are recovered. Tomato pests: only spider mites (PlantVillage/LeafNet) and mites in LeafNet; no whitefly/Tuta absoluta set could be loaded.
- **Cucumber:** #11 for disease classification (leaf, fruit and stem), #12 for symptom detection in greenhouse rows. No cucumber pest or deficiency data could be loaded.
- **Pepper:** thin. Bell pepper = bacterial spot + healthy (PlantVillage, PlantDoc) and 4 web-image diseases in PlantSeg; chili = COLD #10 (532 images incl. mites/thrips and nutritional), which LeafNet2.0 seems to contain in augmented form. No pepper detection set with boxes on diseases.

---

## Environment and access notes (read first)

- **Hugging Face:** logged in as `bebedovis`, who has accepted the **LeafNet2.0 gate**. Teammates must log in and accept it themselves (non-commercial form). No other dataset here is gated.
- **Python:** pyenv `p312`; `datasets`, `huggingface_hub`, `pyarrow`, `pandas`, `matplotlib`, `pillow`, `nbformat`, `jupyter`. No `fiftyone`, `roboflow` or `ultralytics` needed.
- **Disk:** PlantSeg 1.7 GB, IP102 3.2 GB, CCMT raw 1.3 GB (the 7.1 GB `augmented` config is not downloaded), Strawberry_12 zip 2.66 GB, extracted to `data/strawberry12/`. PlantDoc is read remotely, column by column (never `load_dataset` it: 30 GB in memory).
- **Runtime:** `plague.ipynb` ≈ 15 min, mostly reading the headers of LeafNet's 256k images.
- **Tomato/pepper extension:** HF cache grew by about 4.1 GB (tomato disease detection #9 alone is 3.0 GB of 3–4k px photos; the apple "pepper" set 0.9 GB; the rest < 0.2 GB).
- **Licensing for commercial use:** clearly OK: PlantVillage (CC0), PlantDoc, IP102 (AgML), CCMT (CC BY 4.0, attribution), Strawberry_12 (MIT), tomato #8, #9, chili #10 and cucumber #11, #12 (CC BY 4.0). **Non-commercial:** LeafNet2.0 (gate), PlantSeg (NC-ND). **Image rights:** PlantSeg, PlantDoc, IP102 and part of Strawberry_12 contain **web-scraped / watermarked stock photos** (google/bing/baidu file names; gettyimages, alamy, featurePics, 36TV.cn watermarks). The dataset license does not clear those, so check with legal before product use.

---

## Condition kinds and colors

Every class is tagged as **healthy**, **disease**, **pest**, **deficiency** or **plant part** (fruit/flower, not a condition). Charts use a validated categorical set: blue / orange / aqua / yellow / gray, with direct value labels on every bar.

### Berry coverage (examples; boxes for PlantDoc and Strawberry_12)

| berry · condition | PlantVillage | LeafNet2.0 | PlantSeg v3 | PlantDoc | Strawberry_12 |
|---|---|---|---|---|---|
| strawberry · healthy | 456 | 641 (incl. all 456 PV) | – | 492 | – |
| strawberry · leaf scorch | 1,109 | 1,109 (same images) | 39 | – | – |
| strawberry · anthracnose | – | – | 58 | – | 700 |
| strawberry · leaf spot | – | – | – | – | 868 |
| strawberry · powdery mildew | – | – | – | – | 1,127 |
| strawberry · gray mold | – | – | – | – | 1,141 |
| strawberry · blight | – | – | – | – | 385 |
| strawberry · insect pests | – | – | – | – | 315 |
| strawberry · Ca / Fe / K(P) deficiency | – | – | – | – | 1,204 / 1,834 / 623 |
| blueberry · healthy | 1,502 | – | – | 849 | – |
| blueberry · anthracnose / botrytis blight / mummy berry / rust / scorch | – | – | 42 / 36 / 47 / 43 / 43 | – | – |
| raspberry · healthy | 371 | 371 (same images) | – | 556 | – |
| raspberry · fire blight / gray mold / leaf spot / yellow rust | – | – | 34 / 40 / 18 / 37 | – | – |

Plus **IP102 pests that also attack strawberries** (insect images, not on strawberry): aphids 4,094; beet armyworm 1,604; *Spodoptera litura* 1,304; armyworm 1,071; thrips 879; tarnished plant bug 820; chilli thrips 807; greenhouse whitefly 692; two-spotted spider mite ("red spider") 529; plant bug 381; spider mite 245; broad mite 84. Chart: `charts/plague_berry_coverage.png`.

---

## 1. PlantVillage: `BrandonFors/Plant-Diseases-PlantVillage-Dataset`

- **Task:** classification, 38 classes `Crop___condition` (14 crops). Splits train 43,456 / test 10,849. Lab photos of single leaves on plain backgrounds, all **256×256 RGB**.
- **License:** `cc0-1.0` in metadata; the card body says "More Information Needed".
- **Berries:** strawberry leaf scorch 1,109 + healthy 456; blueberry healthy 1,502; raspberry healthy 371.
- **Gotchas:** `features["label"]` is a `ClassLabel` (use `.names`). Orange and Squash are disease-only; Blueberry, Raspberry, Soybean healthy-only. Known to transfer poorly to field photos.

## 2. LeafNet2.0: `enalis/LeafNet2.0`

- **Access:** ⚠️ gated, auto-approved form, **non-commercial only**, no license declared. Actively updated (counts may shift).
- **Task:** classification, **182 classes / 37 crops**, 255,990 images (`images` config); `image_caption_unified` adds a description per image (captioning / VLM). Only a `train` split.
- **Images:** 224×224 except one 512×512; modes 252,980 RGB, 3,003 RGBA, 6 CMYK, 1 L → `img.convert("RGB")`.
- **Balance:** very skewed: Coffee healthy 18,983 and Coffee miner 16,978; 11 classes under 100 images, `Onion___Bulb_blight` has **1** image.
- ⚠️ **Contains PlantVillage:** every PlantVillage image of raspberry healthy (371/371), strawberry leaf scorch (1,109/1,109) and strawberry healthy (456/456) is also in LeafNet, resized (perceptual-hash distance 0). **De-duplicate before combining.**
- **Gotchas:** `captions_full` has no `label` column (use `folder_name`) and 179 classes vs 182.

## 3. PlantSeg v3: `XuVV/plantsegv3` ⭐ (masks)

- **Source:** Wei et al., *PlantSeg* (arXiv:2409.04038), Univ. of Queensland + CSIRO; Zenodo v3 doi:10.5281/zenodo.14935094. **CC BY-NC-ND 4.0** on the original; the HF mirror declares no license.
- **Task:** in-the-wild disease **segmentation** (semantic mask + instance polygons + boxes) → also detection, classification and **severity** (lesion share). 114 disease classes / 34 crops, **no healthy class**. Splits train 7,916 / val 1,247 / test 2,295.
- **Columns:** `image`, `mask` (PNG, class index per pixel), `segmentations` (polygons per instance), `bboxes` `[x, y, w, h]` px, `category_ids`, `areas`, `answer` (class name), `file_name`.
- **Stats:** 7–390 images per class (median 77). Lesions cover a median **13.1%** of the image (p10 2.3%, p90 52.5%), median 3 lesion instances per image. 5,709 distinct sizes, width 65–8,256 px (median 667).
- **Web-scraped:** file names say google 4,627, bing 3,270, baidu 700 (2,861 untagged).
- **Berries:** the only source of **raspberry diseases** (fire blight 34, gray mold 40, leaf spot 18, yellow rust 37) and of 5 **blueberry diseases** (anthracnose 42, botrytis blight 36, mummy berry 47, rust 43, scorch 43); strawberry anthracnose 58, leaf scorch 39.
- **Relation to PlantWild:** same group. For the shared berry classes, **every PlantSeg image is also in PlantWild** (39/39, 58/58, 43/43); PlantWild has more images per class (76, 98, 117) but no masks.

## 4. PlantDoc detection: `susnato/plant_disease_detection_processed`

- **Source:** processed PlantDoc (Singh et al., CoDS-COMAD 2020; GitHub `pratikkayal/PlantDoc-Object-Detection-Dataset`). **CC BY 4.0.**
- **Task:** detection (boxes around leaves), preprocessed for DETR (`pixel_values`, `labels`...). 5.3 GB download, **30.7 GB in memory**: read only `image_id, width, height, objects` from the parquet files. Raw boxes `objects.bbox` = `[x, y, w, h]` px.
- **Splits:** train 2,110 imgs / 8,114 boxes; test 214 / 430. Width 115–6,000 px (median 800); boxes per image median 2, max 42.
- ⚠️ **Class names are not in the dataset** (`objects.category` is 0–23). Mapping recovered by matching every box to the original CSVs (8,541 / 8,544):

  ```
  0 Cherry leaf               8 Potato leaf early blight     16 grape leaf
  1 Peach leaf                9 Apple leaf                   17 Tomato Early blight leaf
  2 Corn leaf blight         10 Tomato leaf yellow virus     18 Apple Scab Leaf
  3 Apple rust leaf          11 Blueberry leaf               19 Tomato Septoria leaf spot
  4 Potato leaf late blight  12 Tomato leaf mosaic virus     20 Tomato leaf
  5 Strawberry leaf          13 Raspberry leaf               21 Soyabean leaf
  6 Tomato leaf late blight  14 Tomato leaf bacterial spot   22 Bell_pepper leaf spot
  7 Tomato mold leaf         15 Squash Powdery mildew leaf   23 Bell_pepper leaf
  ```

- **Gotchas:** 5 original classes are missing (Corn gray leaf spot, Corn rust, Grape black rot, Potato leaf, Tomato spider mites). Berries are healthy-only. Many web stock photos on white backgrounds, some watermarked.

## 5. IP102 insect pests: `Project-AgML/CT_PestNet_crop_pest_classification` ⭐ (pests)

- **Source:** AgML calls it "CT PestNet". Its 75,222 images and 102 classes (same order) match **IP102** (Wu et al., CVPR 2019), the standard pest benchmark (*matched by numbers and class list; the card does not cite IP102*). **CC BY 4.0** (AgML).
- **Task:** classification of the **insect**, 102 species grouped by host crop (rice 14, corn 13, wheat 9, beet 8, alfalfa 13, grape 16, citrus 19, mango 10). Splits train 45,095 / val 7,508 / test 22,619 (`split` column).
- **Stats:** 71–5,740 images per class (long tail; largest *Cicadellidae*). 28,904 distinct sizes, width 63–8,113 px (median 439); 7 RGBA images.
- **Berry-relevant pests:** see the coverage section (aphids, armyworms, thrips, whitefly, spider mites, plant bugs).
- **Gotchas:** classification only (the original IP102 detection subset is not in this mirror). Images mix eggs/larvae/pupae/adults, illustrations and **watermarked stock photos**, and insects are not necessarily on the crop. Use it for **pest recognition / pretraining**, not as a field detector.

## 6. CCMT crop pest and disease: `Project-AgML/crop_pest_disease_classification`

- **Source:** Mensah et al., *CCMT: Dataset for crop pest and disease detection*, Data in Brief 49 (2023); Mendeley doi:10.17632/bwh3zbpkpv.1. Photographed on **farms in Ghana**. **CC BY 4.0.**
- **Task:** classification, 18 classes, 4 crops (cashew 6,549; cassava 7,508; maize 5,321; tomato 5,792). `raw` config = 25,170 images (used); `augmented` = 105,252 generated copies (skipped).
- **Pest damage classes:** fall armyworm 285 (maize), grasshopper 673 (maize), green mite 1,015 (cassava), leaf beetle 938 (maize), leaf miner 1,378 (cashew). Healthy 3,235. The rest are diseases (anthracnose, bacterial blight, brown spot, gumosis, leaf blight, leaf curl, leaf spot, mosaic, red rust, septoria, streak virus, verticillium wilt).
- **Why it matters:** the pest classes show the **damage on the plant** (holes, mines, stippling), which is what a field camera sees, unlike IP102's close-ups of insects.
- **Gotchas:** all images resized to 400×400 (25,128 of 25,170). **831 byte-identical duplicate copies**: de-duplicate before splitting. `raw` has no split column. Not berries.

## 7. Strawberry_12: `Ethan0300/Strawberry_12-dataset` ⭐ (strawberry)

- **Source:** Hu, Jiang & Guan, *StrawDet-YOLO*, IEEE SMC 2025 (doi:10.1109/SMC58881.2025.11343179). Greenhouse + open field, April 2022 – October 2023. **MIT license.**
- **Format:** YOLO (`labels/<split>/*.txt`, normalized `cls cx cy w h`). The repo holds a zip (used, extracted to `data/strawberry12/`) and the same files unzipped. Splits train 3,122 / val 385 / test 399 images. The inner README is UTF-16 encoded.
- ⚠️ **Class names are not shipped** (no `data.yaml`). Recovered from file-name prefixes, which are unambiguous per class, then checked on crops:

  | id | class | kind | boxes (originals / all) |
  |---|---|---|---|
  | 0 | ripe fruit | plant part | 2,028 / 4,732 |
  | 1 | unripe fruit | plant part | 717 / 1,673 |
  | 2 | flower | plant part | 645 / 1,505 |
  | 3 | leaf spot | disease | 372 / 868 |
  | 4 | powdery mildew | disease | 483 / 1,127 |
  | 5 | gray mold | disease | 489 / 1,141 |
  | 6 | anthracnose | disease | 300 / 700 |
  | 7 | blight | disease | 165 / 385 |
  | 8 | insect pests | pest | 135 / 315 |
  | 9 | calcium deficiency | deficiency | 516 / 1,204 |
  | 10 | iron deficiency | deficiency | 786 / 1,834 |
  | 11 | potassium (K/P) deficiency | deficiency | 267 / 623 |

- ⚠️ **57% augmented copies:** 1,674 original photos + 558 each of `_blur`, `_bright`, `_dark`, `_noise`. Copies stay in the same split as their original (no leakage), but counts are inflated about 2.3×.
- ⚠️ **Some "originals" are augmented too:** black cutout rectangles, **2×2 mosaics** of four photos, heavily upscaled tiny images, and a few watermarked web images.
- **Stats:** box area median 4.35% of the image (p10 0.92%, p90 18.1%), much bigger objects than berry-canopy sets. 353 distinct sizes, width 135–4,000 px (median 636).
- **Insect pests** is a single class (which pest is not specified); boxes go around damaged/infested leaves.

---

# Tomato and pepper (added 2026-10-12)

PlantVillage, LeafNet2.0, PlantSeg, PlantDoc and CCMT already contain tomato and pepper. The new sets were added for what those lack: field **boxes**, disease on **fruit and stems**, and a pepper-family set with pests and deficiencies.

### Tomato, bell pepper, chili and cucumber coverage (examples; boxes for PlantDoc, #8, #9, #12)

`charts/plague_tomato_pepper_coverage.png`. Bell pepper and chili are counted separately; LeafNet2.0's **black pepper** (*Piper nigrum*, a different plant) is excluded. Conditions named after mites/thrips/miners/aphids/whiteflies count as pests and nutritional ones as deficiencies in every dataset. Cucumber #11 counts originals only. Number in brackets = classes incl. healthy.

| Dataset | Tomato | Bell pepper | Chili | Cucumber |
|---|---|---|---|---|
| PlantVillage | 18,160 (10) | 2,475 (2) | 0 | 0 |
| LeafNet2.0 | 28,816 (16) | 2,603 (3) | 10,974 (5) | 800 (5) |
| PlantSeg v3 | 901 (7) | 247 (4) | 0 | 478 (3) |
| PlantDoc | 2,910 boxes (8) | 587 boxes (2) | 0 | 0 |
| CCMT | 5,792 (5) | 0 | 0 | 0 |
| #8 Tomato leaf det. (Imtiaz) | 2,278 boxes (7) | 0 | 0 | 0 |
| #9 Tomato disease det. (Liu) | 3,167 boxes (10) | 0 | 0 | 0 |
| #10 Chili leaves (COLD) | 0 | 0 | 532 (5) | 0 |
| #11 Cucumber diseases (Sultana) | 0 | 0 | 0 | 1,289 (8) |
| #12 Cucumber leaf det. (Omarov) | 0 | 0 | 0 | 2,705 boxes (2) |

⚠️ **LeafNet2.0 overlaps:** its chili classes total 10,974 with per-class counts (2,217 / 2,195 / 2,504 / 2,029 / 2,029) equal to COLD's `augmented` config, and its 5 cucumber classes have 160 images each, like #11's originals. LeafNet2.0 very likely contains both *(counts only; not hash-checked)*. Bell pepper beyond bacterial spot exists only as web images in PlantSeg (blossom end rot 112, frogeye leaf spot 31, powdery mildew 28) and LeafNet's leaf spot (67).

## 8. Tomato leaf disease detection: `Project-AgML/tomato_leaf_disease_detection`

- **Source:** Imtiaz et al., Data in Brief 60 (2025); Mendeley doi:10.17632/bpfd9cns5g. **CC BY 4.0.** Field photos (Canon EOS M50) from three districts in Bangladesh.
- **Task:** detection of lesions on leaves. 689 images, 2,278 boxes; split column train 597 / valid 61 / test 31.
- ⚠️ **Class names are not shipped:** parquet and card name the classes `'0'`…`'6'`. The paper and Mendeley page list Early Blight, Black Spot, Late Blight, Leaf Mold, Bacterial Spot, Target Spot, Healthy, but not the id order, and the Mendeley release has no `data.yaml`. From the samples, **id 2 = healthy** (one box around a whole leaf). The others are kept as `disease id k`. Id 3 has large dark lesions with yellow halos (early or late blight?) *(unverified)*.
- ⚠️ **Roboflow export:** all 640×640; 689 images come from 308 source photos (165 photos ×3, 51 ×2, 92 ×1). The original split keeps copies of a photo together (0 photos span splits), but counts are inflated.
- **Boxes are tiny:** median 0.11% of the image (p10 0.03%, p90 15.74%, the large ones being healthy whole-leaf boxes).
- **Verdict:** usable for healthy-vs-diseased leaf detection now; per-disease training needs the names from the authors or re-labeling.

## 9. Tomato disease detection: `Project-AgML/tomato_disease_detection` ⭐ (tomato fruit + stem)

- **Source:** Liu et al., Data in Brief 63 (2025); Mendeley doi:10.17632/c2x8rynybg.1. **CC BY 4.0.**
- **Task:** detection by disease × plant part. 1,026 images, 3,167 boxes; image-level `disease` column Wilt 527 / Viral 417 / GrayMold 82. The card has no class-label names, but its text gives them: 0 GrayMold_Fruit, 1 GrayMold_Leaf, 2 Viral_Leaf, 3 Viral_Top, 4 Virus_Middle, 5 Wilt_Base, 6 Wilt_Leaf, 7 Wilt_Middle, 8 Wilt_Stem, 9 Wilt_Top (mapping used in the builder). Box counts are in `charts/plague_tomato_disease_detection_classes.png`.
- **Images:** 3024×4032 (739) or 4032×3024 (287) phone photos. 0 boxes outside the image, 0 images without boxes. 7 images have boxes that disagree with their image-level disease column. Sample grid checked: boxes land on the diseased fruit, leaves and stems; the "Top"/"Middle"/"Base" boxes are large regions of the plant.
- **Gotchas:** gray mold is small (82 images); no healthy class; no split column.
- **Verdict:** the only set with **diseased tomato fruit** and stem boxes; good complement to the leaf-only sets.

## 10. Chili leaves (COLD): `Project-AgML/COLD_chili_leaf_disease_classification`

- **Source:** Aishwarya & Reddy, Data in Brief 54 (2024); Mendeley doi:10.17632/tf9dtfz9m6.2. **CC BY 4.0.** `raw` config used (532 images); `augmented` (10,974) skipped.
- **Task:** classification: cercospora 152, healthy 69, **mites_and_trips 107 (pest)**, **nutritional 102 (deficiency)**, powdery mildew 102. The parquet label is a plain integer; names come from the card.
- **Images:** all 256×256 close-ups of one leaf. 1 byte-identical duplicate, and it has conflicting labels.
- **Verdict:** the only pepper-family set with pest and deficiency classes, but small. Check overlap with LeafNet2.0 before combining (see the coverage note above).

## Checked and discarded (tomato / pepper)

| Dataset | Why discarded |
|---|---|
| `Project-AgML/tomato_leaf_disease` (11,000 imgs, 10 × 1,100; Kaggle `kaustubhb999/tomatoleaf`) | All 11,000 file names carry PlantVillage's `<uuid>___<source>` pattern, and **539 of 600** sampled images (60 per class) have a perceptual-hash match in the same PlantVillage class. It is a PlantVillage re-upload. |
| `Project-AgML/pepper_disease_pest_classification` (26,377 imgs; cited as Wang et al., Plant Methods 21, 2025; Mendeley "vegetable disease" doi:10.17632/tg3z7xxkdb.1) | ⚠️ **The images are apple leaves, not pepper:** serrated ovate leaves, cedar-apple-rust spots, and the class list (Alternaria_Boltch, Brown_Spot, Grey_spot, Mosaic, Rust) is the well-known Chinese apple-leaf-disease set. No pest and no healthy class. As an apple set it has 760 byte-identical extra copies, **364 with conflicting labels**, and a lab/orchard background mix. |

**Also screened, not added (tomato/pepper):** Roboflow `yolov8ntomatoleaf/v9-yqge9` (6,070 imgs, 10 tomato classes incl. leaf miner and spider mites), `9days/plant-disease-classifier-ehvf1` (19,574 imgs, tomato classes incl. N/K/Mg deficiency) and `animaldef/pepper-leaf-detection-4ug57` (870 imgs, healthy/damaged/fungus/sick) are not downloadable (Roboflow quota) and look PlantVillage-derived or loosely labeled *(unverified)*. Mendeley `ysk546n9np` tomato insect pests (6 pests incl. whitefly, thrips, spider mite) is augmented and partly built from Google images. The Tanzania *Tuta absoluta* study (arXiv 2004.04023) has no public data link found. Many more PlantVillage tomato re-uploads exist on HF.

---

# Cucumber (added 2026-10-12)

## 11. Cucumber diseases: `Project-AgML/cucumber_disease_classification`

- **Source:** Sultana et al., Data in Brief (2023); Mendeley doi:10.17632/y6d3z6f8z9.1 (Jahangirnagar University, Bangladesh). **CC BY 4.0** on Mendeley; the AgML card states none. HF copy 2.8 GB.
- **Task:** classification, 8 classes covering **leaf, fruit and stem**: Anthracnose, Bacterial_Wilt, Belly_Rot (fruit), Downy_Mildew, Gummy_Stem_Blight, Pythium_Fruit_Rot (fruit), Fresh_Leaf, Fresh_Cucumber (healthy fruit). The parquet label has no names; card names in card order match the file names (0 mismatches).
- ⚠️ **Originals and augmented copies are mixed with no flag:** 1,289 originals (`Class(n).jpg`, 160 per class, 169 Pythium) and 6,400 augmented copies (`Class_(n).jpg`, 800 per class). The copies are numbered separately, so they cannot be linked to their original by name: **use the originals only**, or a random split leaks. 19 byte-identical extra copies (no label conflicts).
- **Images:** 2296×1724 phone photos (29 augmented images are 1020×1020). Harvested fruit photographed on the ground, leaves in the field.
- **Verdict:** the best cucumber disease set and the only one with fruit rots; no pests, no deficiencies.

## 12. Cucumber leaf abnormality detection: Omarov et al., Mendeley `mb3vnhpk8z`

- **Source:** Omarov, Ibrayev, Imanbayeva & Belgitayeva, *An Annotated Image Dataset for Cucumber Leaf Disease Detection and Segmentation in Complex Orchard Environments*, Mendeley Data (2026), doi:10.17632/mb3vnhpk8z.1. **CC BY 4.0.** Zip ≈ 220 MB in `data/cucumber_leaf_detection/`, read in place.
- **Task:** detection of 2 **symptom** classes (labels are phenotypes, not pathogens): `color` 1,741 boxes (abnormal colour: yellowing/browning) and `damage` 964 boxes. 250 affected images (all with YOLO labels) + 250 healthy images (no labels, negatives only).
- **Images:** 3840×2160 (387), 4608×3456 (112), one 4000×2252. Many names are video frames (`frame_####`); **38 of 500 images have a near-identical twin** (perceptual hash).
- ⚠️ **Despite "orchard" in the title, the samples show a greenhouse** (high-wire cucumber rows, substrate slabs, heating pipes).
- ⚠️ Many `color` boxes cover whole leaves under a strong yellow light cast, so "abnormal colour" is partly confounded with lighting. **40 boxes extend past the image border** (clip them).
- **Verdict:** a small greenhouse detection set for "affected leaf region"; not for naming diseases.

**Cucumber sets screened, not added:** HF `YaswanthReddy23/Cucumber_leaf` (6.4k images, same 8 classes as #11, another copy of the augmented Mendeley data); Roboflow `capstone-qhqkt/cucumber-final` (9,422 imgs, classes coded `a3_cucumber`…`c4_cucumber` with no legend; MIT), `aimit-3uqkr/cucumber-pests` (574 imgs, pests) and `main-6joil/merge-apple-cucumber-tomato-grape2` (CucumberLeaf healthy/unhealthy) are not downloadable (Roboflow quota). Omer et al. 2022 (4,868 imgs, spider mites, leaf miners, downy and powdery mildew) is private, on request. Mendeley WMC-Leafset (`8m2ytxd4dg`) covers wax gourd and Mangalore cucumber, not cucumber.

---

## Discarded

| Dataset | Why discarded | If you need it |
|---|---|---|
| PlantWild `EGOISTyrh/mvpdr-plant-disease` | Same group and classes as PlantSeg, with **no masks**; PlantSeg's berry images are all inside it. Same NC-ND license and web-scraping issues. | It is the **larger** set for plain classification (18,542 images v1, more per class, includes healthy classes). Details in `archive/PLAGUE_DATASET_FINDINGS_v1.md`. |
| Strawberry disease `luffywang/strawberry-disease-detection-kszia` (Roboflow) | Public export broken (`NoSuchKey`); fork route blocked by the workspace image quota. Strawberry_12 covers the same diseases (MIT). | Roboflow quota, manual Universe download, or the Kaggle original (Afzaal et al. 2021). A larger variant exists: `hello-peter/strawberry_disease_7classes` (15,531 imgs, public domain, *unverified*). |

**Also screened, not added:** Plant Pathology 2021 (apple only); PDSNet segmentation (240 images, apple/soybean/wheat); Roboflow spider-mite/thrips sets (< 800 images, one class each); `EnmmmmOvO/insect-pest-dataset` (same size as IP102, no class names); `xiaopch/agricultural_pests_and_diseases_datasets` (5k images, includes non-pest classes like "pesticides"); many PlantVillage re-uploads.

## Cross-dataset notes

- **Overlaps:** LeafNet2.0 ⊃ PlantVillage (berry classes checked: 100%); PlantWild ⊃ PlantSeg (berry classes checked: 100%); AgML `tomato_leaf_disease` ⊂ PlantVillage (539/600 sampled); LeafNet2.0 probably ⊃ COLD augmented and ⊃ cucumber #11 originals (identical counts). Treat each pair as one source when splitting.
- **Domains:** lab (PlantVillage, much of LeafNet) → web stock (PlantSeg, PlantDoc, IP102) → real field (CCMT, most of Strawberry_12). A field model should be validated on field data (CCMT, Strawberry_12), not on lab/web sets.
- **Image size:** fixed 224/256/400 px for LeafNet/PlantVillage/CCMT; web sets from ~60 px to 8,000 px. See `charts/plague_image_width.png`.
- **Healthy class:** missing in PlantSeg and Strawberry_12 (Strawberry_12 has fruit/flower instead). Take healthy examples from PlantVillage / PlantDoc / CCMT.

## Open items / suggested Jira tickets

- [ ] Legal: LeafNet2.0 (non-commercial gate), PlantSeg (NC-ND), and web-image rights for PlantSeg / PlantDoc / IP102 / parts of Strawberry_12.
- [ ] Confirm the Strawberry_12 class names with the paper or the authors (recovered from file names here).
- [ ] De-duplicate: PlantVillage vs LeafNet2.0; CCMT internal duplicates (831); Strawberry_12 augmented copies (count and split by source photo).
- [ ] **Berry data gap:** blueberry and raspberry diseases exist only as 18–47 web images per class (PlantSeg). No dataset has berry pests photographed on berry plants. Consider collecting our own field images.
- [ ] If PlantDoc is used: add the class-name mapping above to the loader; decide whether the 5 missing classes matter.
- [ ] Each teammate: HF login + accept the LeafNet2.0 gate.
- [ ] **Tomato #8:** get the id → disease mapping from the authors (Imtiaz et al.) or re-label; until then use it only as healthy vs diseased.
- [ ] **Pepper:** no pepper disease **detection** data and no real pepper pest data. Collect our own, or get Roboflow quota for the screened sets. Report the mislabeled AgML `pepper_disease_pest_classification` (apple images) to AgML.
- [ ] **Tomato pests** (whitefly, *Tuta absoluta*, thrips): no loadable set found; search Kaggle/Mendeley further or collect.
- [ ] Hash-check LeafNet2.0 chili and cucumber classes against COLD and cucumber #11 before combining.
- [ ] **Cucumber:** loader for #11 must keep only `Class(n).jpg` originals; cucumber pests (spider mites, thrips, whitefly) need Roboflow quota (`aimit-3uqkr/cucumber-pests`) or the private Omer et al. set.
