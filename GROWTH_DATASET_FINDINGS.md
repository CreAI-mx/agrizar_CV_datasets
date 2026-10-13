# Plant Growth Datasets (Berry Focus): Findings

> **Purpose:** handoff context for the team and for future Claude sessions. This is the follow-up to `PLAGUE_DATASET_FINDINGS.md` (plant disease, notebook `plague.ipynb`). Everything was checked hands-on (2026-10-08; tomato, sweet pepper and hydroponics sections #11–#16 and cucumber #17 added 2026-10-12) unless marked *unverified*. Previous version: `archive/GROWTH_DATASET_FINDINGS_v1.md`.
> **Working code:** `growth.ipynb` in this folder (one section per dataset; it runs top to bottom in about 1.5 minutes once the data is cached). **Charts:** `charts/growth_*.png`. **Image examples:** `examples/growth/*.png` (one or more sample grids per dataset).

## TL;DR

| # | Dataset | Crop | Task it is built for | License | Size | Time series | Usable now? |
|---|---|---|---|---|---|---|---|
| 1 | Strawberry growth `Project-AgML/strawberry_growth_detection` (HF) | strawberry | Detection, **7 growth stages** + fruit size | CC BY 4.0 | 1,477 imgs / 3,997 boxes, 6.7 GB | ✅ tagged fruits, 34 dates | ✅ **Best strawberry set** |
| 2 | Strawberry 5-stage `strawberry-vnobp/strawberry-2-k4eap` (Roboflow) | strawberry | Detection, 5 stages | CC BY 4.0 | 6,465 imgs / 15,792 boxes | partly: 10 time-lapse plants | ⚠️ Yes, but **dedupe and re-split** |
| 3 | Strawberry & avocado `Project-AgML/strawberry_avocado_ripeness_detection` (HF) | strawberry + avocado | Ripeness detection | CC BY 4.0 | 758 imgs / 924 boxes | no | ❌ **Class names missing + wrong boxes on portrait photos** |
| 4 | Blueberry maturity `Project-AgML/blueberry_maturity_detection` (HF) | blueberry | Detection + **counting**, unripe/ripe | CC BY 4.0 | 140 imgs / **17,955 boxes** | no | ✅ Yes; **class names are wrong** (mapping below) |
| 5 | Blueberry segmentation `c-tan/Blueberry-fruit-segmentation` (HF) | blueberry | **Instance segmentation**, immature/mature | **CC BY-NC-SA 4.0** | 724 imgs / 9,277 polygons | no | ✅ Non-commercial only; test split unlabeled |
| 6 | Raspberry `ssd-nv4zv/ssd-raspberry` (Roboflow) | raspberry | Detection, buds → flowers → unripe → ripe | CC BY 4.0 | 2,038 imgs / 46,142 labels | no | ⚠️ **14% of train labels are broken**; ripe = 1% |
| 7 | Chili plant growth `sdp-iiccq/chili-plant-growth-detection-r9kld` (Roboflow) | chili (**general**) | Detection of **whole-plant stage**: seedling → vegetative → flowering → fruiting | CC BY 4.0 | 2,789 imgs / 5,836 boxes | partly: chamber time-lapse | ⚠️ Stage is **confounded with source**; heavy duplication |
| 8 | GrowliFlower `Voxel51/GrowliFlower` (HF) | cauliflower (**general**) | Growth time series (traits, BBCH), instance seg, keypoints | **unknown** | 24,866 samples, 1.2 GB | ✅ **739 plants × up to 15 flights** | ✅ For research; license must be cleared |
| 9 | Mango growth `Project-AgML/mango_growth_detection` (HF) | mango (general, fruit) | Detection, 4 fruit stages | CC BY 4.0 | 2,000 imgs / 3,174 boxes | no | ⚠️ 924 photos ×2–3 augmented copies; no split |
| 10 | Leaf counting `Project-AgML/leaf_counting_denmark` (HF) | field seedlings (general) | Classification by leaf count (1–8, 9+) | **none declared** | 9,372 imgs, 0.9 GB | no | ⚠️ No card, no license |
| 11 | Tomato ripeness (Laboro Tomato) `Project-AgML/tomato_ripeness_detection` (HF) | **tomato** | Detection, green / half / fully ripened × big / cherry | **CC BY-NC 4.0** (original; card states none) | 804 imgs / 9,777 boxes, 1.65 GB | no | ⚠️ Best tomato set, but **boxes need EXIF rotation** on 305 images; non-commercial |
| 12 | Greenhouse tomato (AgRobTomato) `Project-AgML/Greenhouse_Tomato_AgRob_object_detection` (HF) | **tomato** | Detection, 4 USDA colour stages, robot camera | CC BY 4.0 | 449 imgs / 6,084 boxes, 0.14 GB | no (one day of video frames) | ⚠️ 92% unripe; only 30 ripe boxes |
| 13 | Tomato maturity `Project-AgML/tomato_maturity_classification` (HF) | **tomato** | Classification immature / mature | CC BY 4.0 | 1,000 imgs (raw) | no | ⚠️ "Mature" includes turning fruit; class confounded with capture day |
| 14 | Sweet pepper maturity (Harel et al.), Mendeley `ttntwwxkfd` | **sweet pepper** | Classification of maturity class 1–4, multi-view per pepper | CC BY 4.0 | 1,515 colour imgs (+ depth), 1.5 GB zip | no | ✅ Only downloadable pepper-ripeness set; lab backgrounds only |
| 15 | Hydroponic lettuce, 3rd Autonomous Greenhouse Challenge, 4TU `10.4121/15023088` | lettuce, **hydroponic** | Regression of fresh/dry weight, height, diameter, leaf area from RGB-D | CC BY 4.0 | 388 plants (RGB + depth), 0.89 GB | no (destructive; week not recorded) | ✅ Best labeled hydroponic set |
| 16 | HydroGrowNet (Batavia lettuce, NFT), Mendeley `g6cm3v3wdp` | lettuce, **hydroponic** | Growth monitoring + water/air sensor logs | CC BY 4.0 | >390k crops; Month 2 = 35,043 (0.93 GB) | partly: daily, **no plant id** | ⚠️ No labels, no plant id; pre-training only |
| 17 | Autonomous Greenhouse Challenge 2018 cucumber, `koidra/GH_Cucumber` (HF mirror of 4TU) | **cucumber** | Crop growth / yield time series (climate, irrigation, daily harvest) | **CC0** | 6 compartments × 116 days, 50 MB CSV, **no images** | ✅ per compartment | ⚠️ Tabular only; no cucumber image set with stages exists |

**Bottom line for berries:** strawberry is well covered (#1 is the reference, #2 adds volume). Blueberry has only 2 maturity classes (#4 detection, #5 segmentation). **Raspberry has exactly one usable growth-stage dataset (#6), and it needs cleaning.** For a crop-agnostic growth model, GrowliFlower (#8) is the only true per-plant time series, and Chili (#7) is the only whole-plant stage detector.

**Bottom line for tomato, pepper and hydroponics:** tomato ripeness is covered by two detection sets: Laboro #11 (more boxes and stages, but non-commercial and with an EXIF trap) and AgRob #12 (CC BY, robot view, almost no ripe fruit). #13 is a weak image-level backup. **Sweet pepper has no downloadable detection set with stages;** #14 (lab, maturity 1–4) is the only one. **Hydroponic growth data exists only for lettuce:** AGC #15 (labeled traits, one image per plant) and HydroGrowNet #16 (huge, daily, but unlabeled and without plant ids). No hydroponic berry, tomato or pepper dataset was found.

**Bottom line for cucumber:** **no public image dataset with cucumber growth or fruit-stage labels was found.** The only growth data is the Autonomous Greenhouse Challenge 2018 (#17): daily harvest and climate per greenhouse compartment, plus sparse weekly crop traits, with no images.

---

## Environment and access notes (read first)

- **Hugging Face:** same account `bebedovis` as before. None of these HF datasets are gated.
- **Roboflow: exporting public Universe datasets directly is broken.** `versions_export` on the public project returns `ready: true`, but the link gives `NoSuchKey` (same failure as last session's strawberry-disease set). **The fork route works:** fork into our workspace → generate a version → export COCO → download.
  - Forks created in workspace **`veggies-h2dsa`**: `ssd-raspberry-zflvo`, `strawberry-2-k4eap-nrnoa`, `chili-plant-growth-detection-r9kld-lcly3`. Each has **version 1 with no preprocessing and no augmentation** (raw images, so sizes and duplicate checks are meaningful).
  - Exports are unzipped in `data/roboflow/{raspberry,strawberry5,chili}/{train,valid,test}/_annotations.coco.json`.
  - ⚠️ **Quota:** after these 3 forks (~11k images) the free workspace has **0 images left**, so forking the strawberry-disease dataset failed. Deleting a fork does not free quota (it stays in the Trash for 30 days).
- **Disk:** HF cache gained about 9.5 GB (strawberry growth is 6.7 GB of it). `data/` holds 4.2 GB (Roboflow exports + the extracted blueberry COCO zip). GrowliFlower images were **not** all downloaded (24,857 small files at about 10/s would take 40 min). The notebook reads `samples.json` and fetches only the 15 images it shows.
- **Python:** pyenv `p312`. No new packages were needed: no `fiftyone`, `roboflow`, `pycocotools` or `imagehash`.
- **Non-HF downloads (tomato/pepper/hydroponics extension):** Mendeley Data public API (`https://data.mendeley.com/public-api/zip/<id>/download/<version>` for the whole dataset, or per-file `download_url` from `/public-api/datasets/<id>/files?folder_id=root&version=<v>`) and 4TU (`https://data.4tu.nl/v2/articles/15023088/files`). Stored in `data/sweet_pepper_maturity/` (1.5 GB zip, 6.4 GB unpacked, read from the zip), `data/agc_lettuce/` (0.89 GB zip) and `data/hydrogrownet/` (Month2.zip 0.93 GB + sensor workbooks; Month 1 = 2.7 GB and Month 3 = 0.9 GB not downloaded). The HF tomato sets add about 1.9 GB to the cache. 4TU returned HTTP 502 for a while on 2026-10-12, then worked.
- **Licensing summary for commercial use:** CC BY 4.0 (attribution) for #1, #2, #3, #4, #6, #7, #9. **Non-commercial:** #5 (CC BY-NC-SA). Tomato/pepper/hydroponics: CC BY 4.0 for #12–#16; **#11 Laboro is CC BY-NC 4.0** in its original release (the AgML card states no license). **Unknown/none:** #8 GrowliFlower (the card says to contact the author, Jana Kierdorf, Uni Bonn) and #10 leaf counting. Roboflow user uploads (#2, #6, #7) are user-declared CC BY 4.0, and the image provenance of their merged sources is unknown.

---

## Common stage scale (used throughout)

Every dataset names its stages differently. To compare them, each class was mapped to one fruit-stage scale. The 5 levels are the most a single-hue ordinal color ramp can show legibly, which the palette validator confirmed:

| Common stage | Strawberry (#1) | Strawberry (#2) | Strawberry (#3, inferred) | Blueberry (#4) | Blueberry (#5) | Raspberry (#6) | Mango (#9) |
|---|---|---|---|---|---|---|---|
| flower / bud | flower | Flower | – | – | – | Buds, Flowers | – |
| green (unripe) | small g, green | Green | unripe | id 0 | Immature | Unripe_Berries | Early-Fruit |
| white / pale | white | White | partly ripe | – | – | – | Premature |
| turning | turning red | Turning Red | – | – | – | – | Mature |
| ripe | red | Red | ripe | id 3 | Mature | Ripe_Berries | Ripe |
| overripe / rotten | overripe | – | overripe/rotten | – | – | – | – |

The whole-plant scale (Chili #7) is separate: seedling → vegetative → flowering → fruiting. It sits **above** the fruit scale: "flowering/fruiting" is where the fruit scale starts.

**Berry coverage (labels per common stage, all splits):**

| | flower / bud | green | white / pale | turning | ripe | overripe |
|---|---|---|---|---|---|---|
| strawberry (#1+#2+#3) | 3,170 | 5,175 | 5,647 | 2,680 | 3,186 | 279 |
| blueberry (#4+#5) | **0** | 17,439 | **0** | **0** | 9,793 | 0 |
| raspberry (#6) | 16,531 | 29,149 | **0** | **0** | **462** | 0 |

Blueberry "green" also includes pink/purple berries (only 2 classes exist). There are **no blueberry flower datasets** and **no raspberry turning-stage labels**. Ripe raspberry is badly under-represented. Chart: `charts/growth_stage_coverage.png`.

---

## What was checked for each dataset, and why

| Check | Why it matters for growth data | Where saved |
|---|---|---|
| Stage taxonomy + mapping | Datasets use 2- to 7-stage scales, so merging needs one mapping | table above, `plot_stage_bars` per dataset |
| Annotation type and bbox convention | Decides classifier vs detector vs segmenter vs counter; wrong conventions silently ruin training | per-dataset sections below |
| Objects per image and size (tiny = <32² px) | Berries are small and clustered; small-object detection is the main difficulty | `charts/growth_object_size.png` |
| Stage balance | Ripe/flower/turning are often rare | per-dataset charts |
| Temporal info (dates, plant IDs) | True growth modelling needs repeated views of the same plant | "Time series" column in the TL;DR |
| Capture setting | Domain shift (greenhouse vs field vs lab vs synthetic backgrounds) | per-dataset sections |
| Duplicates / leakage (Roboflow `.rf.<hash>` stems, byte hashes) | Augmented copies inflate counts and leak across splits | per-dataset sections |
| License | Commercial use | TL;DR |

---

## 1. Strawberry growth: `Project-AgML/strawberry_growth_detection` ⭐

- **Source:** Yang et al., *Predicting the growth trajectory and yield of greenhouse strawberries based on knowledge-guided computer vision*, Comput. Electron. Agric. 220 (2024); Zenodo doi:10.5281/zenodo.10957909. Reformatted by AgML. **CC BY 4.0.**
- **Task:** object detection, **7 classes:** `flower` 539, `small g` 673, `green` 898, `white` 731, `turning red` 296, `red` 669, `overripe` 191 (3,997 boxes). Single `train` split, so you must split it yourself.
- **Columns:** `image`, `source` (`tagged` 637 / `random` 840), `objects` = {`bbox` [x, y, w, h] px, `categories`, `diameter`, `length`, `decimal_stage`}. The last three are **strings**, and empty or `-1` means missing.
- **Measurements (tagged fruits only):** diameter on 593 boxes (4.4–97 mm, median 13.4), length on 487, `decimal_stage` on 1,144.
- **`decimal_stage` is progress *within* the class** (0.1–0.95, median 0.5 for every class), not a global scale. A continuous stage = class index + decimal_stage.
- **Time:** `tagged` = `YYYYMMDD_n.jpg`, **34 capture dates** in two seasons (2022-07-22 → 2022-10-18 and 2023-07-26 → 2023-09-28), about 20 images per date. `random` = `img_YYYYMMDD_n.jpg` (21 dates) plus 61 undated. ⚠️ **316 `random` images are dated 20210101**, which is most likely an unset camera clock.
- **Images:** 3000–6000 px (6000×4000 most common, 631), all RGB, EXIF orientation 1, **no out-of-bounds boxes**. Median 2 boxes/img (max 27), median box 0.92% of the image, **0% tiny**.
- **Setting:** potted greenhouse plants and field rows (plastic mulch), with handheld photos; yellow tags visible on tagged plants.
- **Gotcha:** 6.7 GB. Read only the columns you need from the parquet files (the notebook's `scan_parquet` reads image headers only). `load_dataset` would also write a full Arrow copy.

## 2. Strawberry 5-stage (Roboflow): `strawberry-vnobp/strawberry-2-k4eap`

- **Task:** detection, `Flower` 2,631, `Green` 3,492, `White` 4,837, `Turning Red` 2,384, `Red` 2,448 (15,792 boxes). Splits train 5,377 / valid 545 / test 543 images. **CC BY 4.0.**
- ⚠️ **The Universe page counts are about 2× the real ones** (it lists Green 7,630, White 10,630 and so on, apparently counting augmented copies). Use the export.
- **It is a merge of several sources** (by file name, unique photos): numbered PNGs 1,391; Korean-style fixed greenhouse cameras `<condition>_<plant>_<n>_<YYMMDDhhmmss>` with conditions chlorosis 688, plague 353, wilt 335, normal 187; `tip*` 640; `chlorosis*` 436; `DSC*` 233; **time-lapse `YYYY-MM-DDTHH_P<plant>` 148**; KakaoTalk 1. Disease-condition images come from a disease dataset, so diseased plants appear in a growth set.
- **Time-lapse subset:** 10 plants, 10–20 dates each, 2023-01-02 → 2024-05-06.
- ⚠️ **Leakage:** 6,465 images come from 4,423 source photos; 1,288 photos appear 2–3×; **280 photos are in both train and test**, 298 in train and valid. Re-split by stem.
- ⚠️ **2,125 of 6,465 images (33%) have no boxes.**
- **Images:** 240 distinct sizes; 1920×1080 (3,999), 1008×756 (1,391), 3840×2160 (245). Median 2 boxes/img, median box 0.48%, 0.3% tiny.

## 3. Strawberry & avocado ripeness: `Project-AgML/strawberry_avocado_ripeness_detection` ❌

- **Source:** Kamat et al., *Comprehensive dataset on ripening stages of strawberries and avocados: From unripe to rotten*, Data in Brief 60 (2025). Phone photos over 25 days, Maharashtra, India. **CC BY 4.0.**
- ⚠️ **Classes are named `"0"`…`"7"`** in both the card and the parquet. Recovered visually from crops (*inferred*): **0** strawberry unripe (112 boxes), **1** strawberry partly ripe (79), **2** strawberry ripe (69), **3** strawberry overripe/rotten (88), **4** avocado unripe (156), **5** avocado ripening (112), **6** avocado ripe (48), **7** avocado overripe/rotten (260). Only 348 strawberry boxes in total.
- ⚠️ **Boxes are wrong on many portrait photos.** 171 of 924 boxes fall outside the image, all on portrait photos (3000×4000 etc.). The boxes were converted to pixels with **width and height swapped**; `x·W/H, y·H/W` fixes them. Per portrait image: stored box out of bounds + fix fits = **154** (use the fix); stored fits + fix out of bounds = 315 (stored is right); **both fit = 197 (ambiguous, verified to go both ways, so these need manual checking)**. Landscape images (92) are fine. EXIF orientation is 1 everywhere, so this is not an EXIF issue.
- **Synthetic backgrounds:** many crops are cut-out fruit on solid yellow/blue/red/cyan backgrounds.
- **Use:** only after fixing names and boxes. For strawberry stages, #1 and #2 are better anyway.

## 4. Blueberry maturity: `Project-AgML/blueberry_maturity_detection`

- **Source:** Deng, Lu & Li, *Detection, counting, and maturity assessment of blueberries in canopy images using YOLOv8 and YOLOv9*, Smart Agric. Technol. 9 (2024); BlueberryDCM, Zenodo doi:10.5281/zenodo.14002517. **CC BY 4.0.**
- **Task:** dense detection + counting. **140 images, 17,955 boxes, median 94 per image (max 800).**
- ⚠️ **Class names are wrong.** The card lists 4 names (`unblue_visible`, `unblue_occluded`, `blue_visible`, `blue_occluded`), the parquet's embedded metadata lists `"0"`–`"3"`, and the README says 2 categories. **Only ids 0 (10,981) and 3 (6,974) occur.** Box-center colors and visual checks show **id 0 = unripe (green/pink), id 3 = ripe (blue)**. Ignore the "visible/occluded" wording.
- ⚠️ **Tiny objects:** median box 0.04% of the image, **49.5% of boxes are under 32×32 px**. The hardest set by far, so it needs tiling / high input resolution.
- **Images:** 140 cropped canopy images, all different sizes (272–4270 px wide), RGB. File names `MSState_YL_Cropped_####.jpg` (Mississippi State), no dates. Single split.

## 5. Blueberry segmentation: `c-tan/Blueberry-fruit-segmentation`

- **Source:** Ni, Li, Jiang & Takeda, *Deep learning image segmentation and extraction of blueberry fruit traits associated with harvestability and yield*, Horticulture Research 7 (2020). **CC BY-NC-SA 4.0** on HF; the COCO file itself says "Unknown License".
- **Task:** **instance segmentation** (polygons, median 20 points per berry). Classes `Immature` 6,458 / `Mature` 2,819. Used in the paper for count, maturity ratio and **cluster compactness** (harvestability).
- **Format:** the HF viewer only shows `image` + folder label. **The annotations are in `berry_coco-formant-annotation.zip`** (COCO, VGG Image Annotator export) and must be extracted (`data/blueberry_seg/`).
- **Splits:** train2017 524 imgs / 7,480 polygons; val2017 145 / 1,797; ⚠️ **test2017 has 55 images and 0 annotations** (unlabeled).
- **Images:** mostly 4896×3264 (318) and 4896×2760 (109), plus downsized 1224×816 copies (53). Median 12 berries/img, 1.4% tiny. A **tape measure is visible in every sample viewed** (usable for absolute size; not counted over the whole set).

## 6. Raspberry (Roboflow): `ssd-nv4zv/ssd-raspberry` ⚠️ the only raspberry growth set

- **Task:** detection, `Buds` 11,786, `Flowers` 4,745, `Unripe_Berries` 29,149, `Ripe_Berries` **462 (1.0%)**. Splits train 1,629 / valid 204 / test 205. **CC BY 4.0.** Every other raspberry set found had a single class or unnamed ids (searched HF and Roboflow).
- ⚠️ **5,191 labels (14% of train boxes, 360 images) are broken point labels.** Their "polygon" is `[cx, cy, ~40, ~45, None, None]`. The first point is the **fruit center**, the second is junk near the top-left corner, and Roboflow turned each into a huge box from the corner to the fruit. **294 training images contain only broken labels.** They are usable as point labels (counting), not as boxes. Drop or re-annotate them.
- **After removing those:** 40,951 boxes, median 16 per image (max 298), median box 0.21%, 2.8% tiny.
- **Images:** all 1773×1773 (likely already cropped/resized before upload); `IMG_####_JPEG` names, no dates. **No duplicates and no leakage.**
- **Ripe is rare:** only 186 images contain a ripe berry. A ripeness model needs more ripe data (or merging with #2-style ripe strawberry is *not* a substitute).

## 7. General purpose: Chili plant growth (Roboflow) `sdp-iiccq/chili-plant-growth-detection-r9kld`

- **Task:** detection of the **whole plant + its growth stage:** `seedling` 3,719, `vegetative` 866, `flowering` 540, `fruiting` 711. Splits train 1,295 / valid 988 / test 506. **CC BY 4.0.** The only general-purpose (crop-agnostic stage names) growth detector found with real volume.
- **Sources:** 730 photos from **growth-chamber time-lapse cameras** (`Wachstum<N>[_Cam<M>]_<date>`, 6 series, up to 45 dates, 2023-09-08 → 2023-12-07); 546 phone photos (`WhatsApp-Image-2026-...`) of seedling trays; 29 camera images.
- ⚠️ **Stage is confounded with source:** seedling = phone tray photos (3,637 of 3,719); flowering/fruiting = chamber time-lapse only. A model can learn "tray vs chamber" instead of the stage. Needs more diverse data before trusting it.
- ⚠️ **Leakage:** 2,789 images come from 1,305 photos; **261 photos are in both train and test**, 441 in train and valid, 246 in valid and test.
- ⚠️ **931 of 2,789 images have black rotation corners:** the uploader uploaded augmented copies as source images (even with no augmentation on our version).
- **Images:** 640×640 (2,544) plus 810×1080. Chamber shots are one plant per image (box up to 77% of the frame); tray shots have tiny seedlings.
- **Verdict:** OK as a **pretraining / demo** source for a "plant stage" head; not enough on its own for a robust crop-agnostic stage detector.

## 8. General purpose: GrowliFlower `Voxel51/GrowliFlower` ⭐ (time series)

- **Source:** Kierdorf et al., *GrowliFlower: An image time series dataset for GROWth analysis of cauLIFLOWER*, arXiv:2204.00294. Uni Bonn / FZ Jülich. UAV (47 MP RGB + multispectral) over two fields near Cologne (2020, 2021). Repackaged as FiftyOne by Voxel51. ⚠️ **License unknown.**
- **No FiftyOne needed:** all labels are in `samples.json` (36 MB); images are `data/data_*/...jpg`.
- **Three subsets (field `task`), each with a spatially disjoint train/val/test split:**
  - **`reference`:** **739 plants × up to 15 flights** (median 15), 9,377 crops of 256×256. Joined field measurements per plant per date: diameter (filled 83%), **BBCH stage (30%)**, head diameter (36%), height (21%), harvested flag (16%). Also `day_after_planting` 1–97 and `plant_id`, which is stable across dates. Field1: 10 flights 2020-08-12 → 11-02; Field2: 15 flights 2021-06-16 → 09-08.
  - **`segmentation`:** 2,198 patches with **plant instance masks, 20,535 leaf instance masks, 2,516 stem keypoints**, void masks (228 patches are bare soil).
  - **`defoliation`:** 846 plants, time series plus a before/after pair on the day leaves were removed (exposes the head).
- **Gotchas:** measurements are **strings** with dirty values (BBCH `'41,41'`, `'3X'`, `'191'`, `'PD'`). BBCH is filled **only on some flights** (2021 field: only days 1–22 after planting; 2020 field: a subset of flights), and later flights carry head diameter instead. Only 230 crops have both. So BBCH alone cannot label the whole season. 238 crops have coverage < 0.9 (plant at the orthomosaic edge).
- **Verdict:** the best data for **growth-curve / stage-regression** methods and per-plant tracking. Not a berry, and the top-down UAV view differs from berry photos.

## 9. General purpose (fruit): Mango growth `Project-AgML/mango_growth_detection`

- **Source:** Kabir et al., Data in Brief 61 (2025); Mendeley doi:10.17632/5snwpzdtzs.1. **CC BY 4.0.**
- **Task:** detection, `Early-Fruit` 820, `Premature` 910, `Mature` 890, `Ripe` 554 (3,174 boxes). Card class names are correct (the parquet itself has plain ints).
- ⚠️ **It is a Roboflow export:** all 640×640, `.rf.<hash>` names. **2,000 images come from 924 photos** (764 photos ×2, 156 ×3). No split column, so split by stem.
- **Every image has one stage**, and the file-name prefix (`early-fruit`, `PMS`, `MS`, `RS`) equals the class. Effectively image-level stage + boxes. Median 1 box/img, big objects (17% of the image).
- ⚠️ **Stage is confounded with background:** `Ripe` images are harvested fruit on white paper, while the other stages are on the tree (seen in the sample grid). A detector can learn "paper vs foliage". Some images have baked-in rotation borders, and some fruits on the tree are not boxed.

## 10. General purpose: Leaf counting `Project-AgML/leaf_counting_denmark`

- **Task:** classification of **leaf count** (`1`…`8`, `9+`), the standard proxy for early growth (BBCH 1x). 9,372 images: 2 leaves 3,292; 4 leaves 2,082; 9+ 570; 7 and 8 leaves only 169/160.
- ⚠️ **Empty card:** no description, citation or license. It looks like the Aarhus University leaf-counting data (Teimouri et al., 2018), but that is *unverified*.
- **Images:** one field seedling per crop, mixed species (grasses and broadleaf weeds/crops), 708 sizes, 33–1853 px (median 155). File names are `<n>.png` **numbered per class**, so 2,082 names repeat but are different images. **14 byte-identical image pairs, 2 with conflicting labels** (`402.png`=4 vs `822.png`=3; `856.png`=1 vs `95.png`=2).

---

# Tomato, sweet pepper and hydroponics (added 2026-10-12)

Tomato "breaker" (first colour change) is mapped to `white / pale`, the step where a fruit stops being fully green. Sweet-pepper classes 2 and 3 both map to `turning`.

## 11. Tomato ripeness (Laboro Tomato): `Project-AgML/tomato_ripeness_detection`

- **Source:** Laboro.AI, LaboroTomato (2020), github.com/laboroai/LaboroTomato; AgML re-host of the Kaggle copy `nexuswho/laboro-tomato`. ⚠️ **License: CC BY-NC 4.0 in the original release; the AgML card states none.**
- **Task:** detection of 6 classes = 3 stages × 2 fruit sizes (`b_` big, `l_` little/cherry): `l_green` 4,596, `b_green` 1,854, `l_fully_ripened` 1,251, `l_half_ripened` 1,020, `b_half_ripened` 636, `b_fully_ripened` 420 (9,777 boxes, 804 images; median 9 boxes/img, max 126). Greenhouse phone photos.
- ⚠️ **EXIF rotation trap:** 305 images are stored landscape (4032×3024) with EXIF orientation 6, but their boxes were drawn on the rotated portrait photo. 693 boxes fall outside the raw frame, and 0 after `ImageOps.exif_transpose`. These 305 are exactly the undated `IMG_####.jpg` files; the other 499 are portrait `IMG_20191215_*.jpg` (one capture day). Sample grid checked after rotation: boxes land on the fruit.
- **Other:** no split column. Median box 0.79% of the image, almost no tiny boxes (0.02%) because the photos are 3–4k px.
- **Verdict:** the richest tomato stage set, and it has a cherry/normal split, but it is non-commercial and must be loaded with EXIF applied.

## 12. Greenhouse tomato (AgRobTomato): `Project-AgML/Greenhouse_Tomato_AgRob_object_detection`

- **Source:** Moreira et al., Agronomy 12 (2022); Zenodo doi:10.5281/zenodo.5596799. **CC BY 4.0.**
- **Task:** detection in 4 USDA colour stages from the AgRob v16 robot's ZED camera (commercial greenhouse, Portugal, 6 Aug 2020): `unripe` 5,594, `breaking` 276, `reddish` 184, `ripe` 30 (6,084 boxes, 449 images, all 1280×720). Each box has an `occluded` flag (2,227 boxes = 37%).
- **Capture:** two capture sessions (297 + 152 frames), frames 3 s apart, so neighbouring frames show the same fruit: split by session or frame block. 14 images have no box. Median box 0.27% of the image, 6.6% tiny.
- **Verdict:** commercially usable and realistic (robot view, occlusion), but late stages are almost absent. Good as a "green tomato" detector or a domain complement to #11.

## 13. Tomato maturity (classification): `Project-AgML/tomato_maturity_classification`

- **Source:** Khatun et al., Data in Brief 51 (2023); Mendeley doi:10.17632/s42kpg8h37.1. **CC BY 4.0.** `raw` config used (1,000 images); `augmented` (4,000) skipped.
- **Task:** `Immature` 500 / `Mature` 500. 16 image sizes (647–1280 px wide), no EXIF rotation.
- ⚠️ **`Mature` includes yellow/orange and yellow-green (turning) fruit**, not only red (seen in samples), so it is closer to "green vs turning-or-ripe".
- ⚠️ **Class confounded with capture day:** file names carry dates over 7 days (2023-01-08 to 2023-03-09), and only 2 days contain both classes. Split by day.
- **Verdict:** weak backup for image-level tomato maturity.

## 14. Sweet pepper maturity: Harel et al., Mendeley `ttntwwxkfd` (not on HF)

- **Source:** Harel, van Essen, Parmet & Edan, *Viewpoints analysis for sweet peppers maturity classification*; Mendeley doi:10.17632/ttntwwxkfd.1. **CC BY 4.0.** Red (cv. Banji) and yellow (cv. Liri) peppers, commercial greenhouse in Israel, 2019.
- **Task:** classification of maturity class 1 (green) to 4 (fully coloured), **one label per pepper**, several views per pepper. 1,515 colour images:
  - *Photocell* (pepper in a lit black box, 1600×1200 BMP): 195 peppers, 4 views each (incl. bottom), classes 1–4.
  - *Robotic* (pepper hung on a plant in the lab, RealSense RGB-D from a robot arm, 640×480): 141 pepper ids, 3 views, classes 2–4 only.
  - *Orientation* (same lab setup, pepper rotated 3×): 34 pepper ids, 9 images each, classes 2–4.
- **Gotchas:** split by pepper, never by image. 17 file-name copies with a `(n)` suffix (some photocell peppers have 8 files). Lab backgrounds only; no boxes; no field images.
- **Verdict:** the only pepper-ripeness data that could be downloaded. Good for a maturity classifier on crops, not for detection.

## Hydroponics

All public hydroponic growth datasets found are **lettuce**. Lettuce has no fruit, so they are not on the fruit-stage scale: the growth signal is biomass and size (or days after transplant).

## 15. Hydroponic lettuce, 3rd Autonomous Greenhouse Challenge: 4TU doi:10.4121/15023088

- **Source:** WUR Greenhouse Horticulture, Bleiswijk (NL), 2021. **CC BY 4.0.** The ReadMe states the lettuce was *"harvested from a hydroponic cultivation system"*.
- **Task:** regression of crop traits from top-view RGB-D (RealSense D415, 1920×1080, RGB 8-bit + aligned 16-bit depth). 388 plants, one image pair each, 4 varieties (Salanova 102, Satine 98, Lugano 96, Aphylion 92). Traits per plant, no missing values: fresh weight 1.4–459.7 g (median 78.2), dry weight 0.1–20.1 g, height 4.3–25 cm, diameter 8.2–42 cm, leaf area 57.6–6,875 cm².
- **Gotchas:** destructive sampling every 7 days, but **the sampling week is not in the ground truth**, and each plant appears once: no per-plant time series. Plants were photographed on a fixed imaging rig (white trays on a frame, people at the image edges), not in the growing gutters. Lighting treatments are not linked to images.
- ⚠️ **One of the 388 plants has no RGB image in the zip** (the ground truth lists it). The normalization (`normalized/hydroponics_lettuce/`) drops it and keeps **387** (Salanova 101); it is counted in the notebook's `dropped` table.
- **Verdict:** the best **labeled** hydroponic dataset; a reference for biomass-from-image models.

## 16. HydroGrowNet (Batavia lettuce, NFT): Mendeley `g6cm3v3wdp` v5

- **Source:** Shalash et al., Eng. Appl. of AI (2025), doi:10.1016/j.engappai.2025.111214. **CC BY 4.0.** NFT system, three 30-day experiments in 2024; a rail camera rig photographed every plant daily; pH, EC, TDS, water temperature, air temperature, humidity and CO₂ logged. The "Data Composition" document states 393,216 images at 640×480.
- **What was checked (Month 2 only):** 35,043 PNGs over 18 capture days (2024-04-16 to 2024-05-12; one more date folder is empty), 2 cameras, 300–4,773 crops per day (median 1,662). File names `date/camN=HH-MM-SS_segment_k.png`, 1–10 segments per camera frame.
- ⚠️ **Each "segment" is a full 640×480 frame with everything masked black except one plant**, often cut by the frame border, some with a gray/blue colour cast. **No plant id and no labels** (no stage, no weight), so plants cannot be followed across days. Images link to sensors by timestamp only.
- **Sensor workbooks** (`all_months_sensory_data.zip`): one `.xlsx` per experiment with sheets for seedling measurements, water quality (sensor + portable meter), nutrient/acid/water consumption, harvest measurements (EXP 1 also has a head-diameter sheet; EXP 2 a taste test). Hand-made layouts with irregular headers; head dimensions are stored as `W*H` text. Read in the notebook with the standard library only (`openpyxl` is not installed).
- **Verdict:** usable as **unlabeled pre-training** data or for population-level growth curves vs water quality. Not for supervised stage training as is.

---

# Cucumber (added 2026-10-12)

## 17. Cucumber crop time series: Autonomous Greenhouse Challenge 2018 (`koidra/GH_Cucumber`)

- **Source:** Hemming et al., Sensors 19 (2019); 4TU doi:10.4121/uuid:e4987a7b-04dd-4c89-9b18-883aad30ba9a, **CC0**. Read from the HF mirror `koidra/GH_Cucumber` (50 MB CSV; the mirror itself has no card or license).
- **What it is:** cucumber cv. Hi Power, 6 glasshouse compartments in Bleiswijk (NL), 2018-08-14 to 2018-12-07 (116 daily production rows per compartment). Five AI teams (AiCU, Croperators, Sonoma, deep_greens, iGrow) controlled climate and irrigation remotely, and one compartment was run by growers. Files per compartment: climate (5-min: air temperature, RH, CO₂, screens, vents, pipes, lighting), irrigation, crop management, production, resource calculations.
- **Growth signal:** cumulative harvest per compartment (kg/m², per the README) and fruit counts, class A/B. Final cumulative harvest: Sonoma 51.3, growers 49.4, Croperators 48.0, iGrow 46.8, AiCU 37.3, deep_greens 35.8 kg/m² (`charts/growth_agc2018_cucumber_harvest.png`). Team Sonoma's extra crop table (85 rows) has stem length growth, leaves per stem and fruit growth, but sparsely filled (12, 14 and 52 non-empty values).
- **Gotchas:** **no images**; growth is per compartment, not per plant; Excel serial dates.
- **Verdict:** ground truth for growth/yield forecasting from climate and irrigation, not for computer vision. The 2019 edition (cherry tomato, CC0 on 4TU) has the same structure *(not loaded)*.

**Cucumber image sets screened, not loaded:** HF `odrinari/Greenhouse-cucumber-and-tomato-plants` (1,072 raw greenhouse photos of cucumber and tomato plants with flowers and green fruit; no labels, no license); Roboflow `cnuvia/cucumber-9fjp3` (117 images, keypoints, `cucumber` / `immature`; Roboflow quota); Riou et al. 2020 cucumber harvest set (arXiv 2011.02719, 555 images of harvest-ready fruit, no download link found); NMBU cucumber canopy-growth set (arXiv 2609.10749), announced as public on publication.

---

## Cross-dataset notes

- **Object size** (`charts/growth_object_size.png`): blueberry maturity (median box 0.04% of the image, 50% tiny) ≪ raspberry (0.21%) < strawberry Roboflow (0.48%) < blueberry seg (0.75%) < strawberry AgML (0.92%) ≪ mango / strawberry-avocado (17–30%, one fruit per photo). The berry canopy sets need tiling or high resolution; the single-fruit sets are closer to classification.
- **Duplicates are the main data-quality issue in Roboflow-origin sets** (#2, #7, #9): always split by `.rf.` stem, never randomly.
- **Label-format bugs found:** wrong class names (#4), missing class names (#3), swapped W/H boxes (#3), point labels turned into boxes (#6), unlabeled test split (#5). None of these is mentioned on the dataset cards.
- **Time series availability:** only #8 (UAV, per plant, with traits), #1 (tagged fruits, 34 dates), and small time-lapse subsets in #2 (10 strawberry plants) and #7 (6 chamber cameras). No berry dataset tracks individual fruits with IDs across dates *(the #1 tagged filenames `_n` may be per-fruit indices; unverified)*.
- **Domain:** greenhouse pots (#1, #2), field canopy (#4, #5, #6), growth chamber (#7), UAV top-down (#8), tabletop/synthetic backgrounds (#3). Tomato: greenhouse phone photos (#11), robot camera (#12), field photos (#13). Pepper: lab only (#14). Hydroponics: imaging rig (#15), rail camera over NFT channels (#16).
- **New label-format bug:** EXIF-rotated photos with boxes in the rotated frame (#11). Same family as the swapped W/H in #3: always draw boxes on samples after deciding whether to apply EXIF.

## Also screened (not analysed in depth)

| Dataset | Why not |
|---|---|
| Roboflow `noc/strawberry-flower` (4,045 imgs, flower/unripe/ripe) | Covered by #1/#2 with finer stages |
| Roboflow `csw-sstmv/strawberry-eopx4` (1,723 imgs, Flower, ripe/unripe, Old_leaves, Picking_point) | Interesting for harvesting robots (picking point), not stages |
| Roboflow `raspberry-detect/raspberry-v3` (4,751 imgs, big/tiny × ripe/unripe, instance seg) | Ripe/unripe only; candidate if more ripe raspberry is needed |
| HF `sankarip/blue_berry_data_3classes_final` (153 imgs, semantic masks) | Tiny, no license |
| HF `Project-AgML/grapevine_growth_detection` (1,212 imgs) | Grape is out of berry scope; has 1 class only despite the name |
| HF `Project-AgML/wheat_growth_stage_classification` (450 imgs, 15 classes, CC0) | Lab wheat; possible general-purpose classification reference |
| HF `Fuqianshi/Plant-Time-Lapse` | Repo is empty |
| HF `Project-AgML/tomato_quality_classification` (fresh / rotten) | Quality, not growth stage |
| HF `Project-AgML/tomato_factory_detection`, `Project-AgML/tomato_detection_kaggle` | Single `tomato` class, no stage |
| Roboflow `jalals-lab/laboro-tomato-kwpth` (840 imgs) | Another copy of Laboro (#11) |
| Roboflow `tomato-ia1r7/tomato-iowni` (804 imgs, green / Turning / ripeness) | Not downloadable (Roboflow quota); probably Laboro again (same image count) *(unverified)* |
| Roboflow `yolo-7ffju/sweet-pepper-djzm8` (411 imgs, red / green / yellow pepper + peduncle) | Colour, not maturity; not downloadable (quota) |
| Uni Bonn BUP19 / BUP20 sweet pepper (agrobotics.uni-bonn.de) | Detection/tracking; no ripeness labels found |
| IEEE DataPort butterhead lettuce (doi:10.21227/d9g8-3w83) and hydroponic lettuce aeroponics/NFT/DWC (doi:10.21227/askr-ed22) | Hydroponic, 30–40 day top/front views, but download needs an IEEE account |
| Roboflow `hidroponic/hidroponic_final` (2,258 imgs: lettuce seedling / ready / sick, basil diseases) and `kdkd-psumx/lettuce-checker` (197 imgs, 4 lettuce stages) | Hydroponic stage labels, but not downloadable (quota) |
| HF `bromleyp/lettuce_growth` | JSON text only, no card |

## Open items / suggested Jira tickets

- [ ] **Raspberry #6:** drop or re-annotate the 5,191 broken point labels (294 images have no valid boxes); find more ripe raspberry (e.g. `raspberry-detect/raspberry-v3`).
- [ ] **Strawberry #2 / Chili #7 / Mango #9:** re-split by source-photo stem to remove train/test leakage.
- [ ] **Strawberry-avocado #3:** decide whether to fix (swap rule + manual check of 197 ambiguous images) or drop.
- [ ] **Blueberry #4:** record the id mapping (0 = unripe, 3 = ripe) in the loader.
- [ ] Legal: GrowliFlower (no license, contact the author), leaf counting (no license), blueberry seg (NC-SA).
- [ ] Strawberry-disease dataset (plague study): the fork route is blocked by the Roboflow free-plan image quota ("0 images left"). Needs more quota or a manual download. See `PLAGUE_DATASET_FINDINGS.md`.
- [ ] Decide the common stage scale with the team (5 levels proposed above) before merging #1, #2, #4, #5, #6.
- [ ] Clean-up: the three forks in `veggies-h2dsa` can be deleted once the exports are archived.
- [ ] **Tomato #11 (Laboro):** apply EXIF orientation in the loader; legal check for non-commercial use (or use the original GitHub release terms).
- [ ] **Tomato #12 (AgRob):** split by capture session; ripe fruit is too rare (30 boxes) to train that stage from this set alone.
- [ ] **Sweet pepper:** no field detection set with stages exists publicly. Options: label our own, or get Roboflow quota for `yolo-7ffju/sweet-pepper-djzm8` (colour only).
- [ ] **Cucumber growth:** no labeled image data. Options: label `odrinari/Greenhouse-cucumber-and-tomato-plants` (raw, ask the uploader for a license), or wait for the NMBU dataset. AGC 2018 (#17) can serve as yield ground truth.
- [ ] **Hydroponics:** decide whether lettuce is acceptable as a proxy crop. If yes, download HydroGrowNet Months 1 and 3 (3.6 GB) and ask the authors for plant ids; request the IEEE DataPort sets (need an account).
