"""Hydroponic lettuce normalization -> `normalized/hydroponics_lettuce/`.

Unlike the crop datasets there are no boxes or polygons here, so the YOLO layout does not apply:
  images/                        AGC RGB images (labeled), copied byte for byte
  depth/                         AGC aligned depth images, same stem as the RGB image
  images_unlabeled/              HydroGrowNet crops (no labels, no plant id): pre-training only
  labels/regression/traits.csv   image, variety, fresh_weight_g, dry_weight_g, height_cm, diameter_cm, leaf_area_cm2
  labels/regression/schema.json  units and ranges
  manifest.csv                   image, source, source_ref, license, width, height, group, dhash, role, label_space
Rules: correctness first (anything that cannot be trusted is dropped and counted), split by `group`, never by image.
"""
import io
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from cvlib.normalize import DATA_DIR, EXAMPLE_DIR, dhash

OUT = Path("normalized") / "hydroponics_lettuce"
TRAITS = {"FreshWeightShoot": "fresh_weight_g", "DryWeightShoot": "dry_weight_g", "Height": "height_cm",
          "Diameter": "diameter_cm", "LeafArea": "leaf_area_cm2"}
HG_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})/(cam\d)=(\d{2}-\d{2}-\d{2})_segment_(\d+)\.png$")

ROWS = []                # manifest rows
TRAIT_ROWS = []          # regression labels
DROPPED = Counter()      # (source, reason) -> count


def begin():
    """Wipe `normalized/hydroponics_lettuce/` and reset state."""
    shutil.rmtree(OUT, ignore_errors=True)
    for sub in ("images", "depth", "images_unlabeled", "labels/regression"):
        (OUT / sub).mkdir(parents=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    ROWS.clear(), TRAIT_ROWS.clear(), DROPPED.clear()


def build_agc_lettuce():
    """3rd Autonomous Greenhouse Challenge (4TU doi:10.4121/15023088, CC BY 4.0): one RGB + depth image per destroyed plant."""
    zf = zipfile.ZipFile(DATA_DIR / "agc_lettuce" / "OnlineChallenge.zip")
    names = set(zf.namelist())
    gt = json.loads(zf.read("GroundTruth/GroundTruth_All_388_Images.json"))["Measurements"]
    for key, m in sorted(gt.items(), key=lambda kv: int(re.sub(r"[^0-9]", "", kv[0]))):
        rgb, depth = f"RGBImages/{m['RGB_Image']}", f"DepthImages/{m['Depth_Information']}"
        if rgb not in names or depth not in names:
            DROPPED[("agc_lettuce", "RGB or depth image missing from the zip")] += 1
            continue
        try:
            values = {new: float(m[old]) for old, new in TRAITS.items()}
        except (TypeError, ValueError):
            DROPPED[("agc_lettuce", "trait value missing or not numeric")] += 1
            continue
        if any(v <= 0 for v in values.values()):
            DROPPED[("agc_lettuce", "trait value <= 0")] += 1
            continue
        data = zf.read(rgb)
        with Image.open(io.BytesIO(data)) as img:
            width, height, digest = img.width, img.height, dhash(Image.open(io.BytesIO(data)))[0]
        name = f"agc_lettuce_{int(re.sub(r'[^0-9]', '', key)):04d}.png"
        (OUT / "images" / name).write_bytes(data)
        (OUT / "depth" / name).write_bytes(zf.read(depth))
        ROWS.append({"image": name, "source": "agc_lettuce", "source_ref": f"{key}:{m['RGB_Image']}", "license": "CC BY 4.0",
                     "width": width, "height": height, "group": f"agc_lettuce:{key}", "dhash": digest, "role": "labeled", "label_space": "regression"})
        TRAIT_ROWS.append({"image": name, "variety": m["Variety"], **values})


def build_hydrogrownet():
    """HydroGrowNet Month 2 (Mendeley g6cm3v3wdp v5, CC BY 4.0): masked single-plant crops, no labels, no plant id.
    `group` = capture day (neighbouring frames show the same plants, so a day must not straddle train and test)."""
    zf = zipfile.ZipFile(DATA_DIR / "hydrogrownet" / "Month2.zip")
    for n in sorted(zf.namelist()):
        if n.endswith("/"):
            continue
        m = HG_NAME.search(n)
        if not m:
            DROPPED[("hydrogrownet", "file name does not match date/camN=HH-MM-SS_segment_k.png")] += 1
            continue
        data = zf.read(n)
        try:
            with Image.open(io.BytesIO(data)) as img:
                width, height = img.size
        except OSError:
            DROPPED[("hydrogrownet", "image file cannot be decoded")] += 1
            continue
        name = f"hydrogrownet_{m.group(1)}_{m.group(2)}_{m.group(3)}_{m.group(4)}.png"
        (OUT / "images_unlabeled" / name).write_bytes(data)
        ROWS.append({"image": name, "source": "hydrogrownet", "source_ref": n, "license": "CC BY 4.0", "width": width, "height": height,
                     "group": f"hydrogrownet:{m.group(1)}", "dhash": "", "role": "pretrain_only", "label_space": ""})


def finalize():
    traits = pd.DataFrame(TRAIT_ROWS)
    traits.to_csv(OUT / "labels" / "regression" / "traits.csv", index=False)
    (OUT / "labels" / "regression" / "schema.json").write_text(json.dumps({
        "task": "regression of lettuce shoot traits from a top-view RGB (optionally + depth) image",
        "columns": {"fresh_weight_g": "g", "dry_weight_g": "g", "height_cm": "cm", "diameter_cm": "cm", "leaf_area_cm2": "cm2"},
        "varieties": sorted(traits["variety"].unique()),
        "ranges": {c: [float(traits[c].min()), float(traits[c].max())] for c in TRAITS.values()},
        "notes": "one image per plant (destructive sampling): no per-plant time series; the sampling week is not recorded",
    }, indent=1))
    manifest = pd.DataFrame(ROWS)
    manifest.to_csv(OUT / "manifest.csv", index=False, columns=["image", "source", "source_ref", "license", "width", "height", "group",
                                                                  "dhash", "role", "label_space"])
    return manifest


def check():
    """Read everything back from disk and assert the format."""
    manifest = pd.read_csv(OUT / "manifest.csv", dtype={"dhash": str}).fillna("")
    traits = pd.read_csv(OUT / "labels" / "regression" / "traits.csv")
    labeled = {p.name for p in (OUT / "images").iterdir()}
    unlabeled = {p.name for p in (OUT / "images_unlabeled").iterdir()}
    assert not labeled & unlabeled
    assert set(manifest["image"]) == labeled | unlabeled and manifest["image"].is_unique
    assert set(traits["image"]) == labeled and traits["image"].is_unique
    assert {p.name for p in (OUT / "depth").iterdir()} == labeled
    assert (traits[list(TRAITS.values())] > 0).all().all() and traits.notna().all().all()
    for _, r in manifest[manifest["role"] == "labeled"].iterrows():  # sizes in the manifest match the files on disk
        with Image.open(OUT / "images" / r["image"]) as img:
            assert img.size == (r["width"], r["height"]), r["image"]
    assert (manifest["role"] == "labeled").sum() == len(labeled) and (manifest["role"] == "pretrain_only").sum() == len(unlabeled)
    print("self-check passed:", len(labeled), "labeled +", len(unlabeled), "unlabeled images")


def summary(manifest):
    print(manifest.groupby(["source", "role"]).agg(images=("image", "size"), groups=("group", "nunique"), license=("license", "first")).to_string())
    traits = pd.read_csv(OUT / "labels" / "regression" / "traits.csv")
    print("\nplants per variety:", traits["variety"].value_counts().to_dict())
    print(traits[list(TRAITS.values())].describe().round(1).to_string())
    if DROPPED:
        print("\ndropped:")
        for (source, reason), n in sorted(DROPPED.items()):
            print(f"  {source:<16}{reason:<70}{n:>7,}")


def show_roundtrip(n=6, thumb=420):
    """Six plants from lightest to heaviest, with the traits read back from disk."""
    traits = pd.read_csv(OUT / "labels" / "regression" / "traits.csv").sort_values("fresh_weight_g").reset_index(drop=True)
    pick = traits.iloc[[int(q * (len(traits) - 1)) for q in np.linspace(0, 1, n)]]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), layout="constrained")
    for ax, (_, r) in zip(np.ravel(axes), pick.iterrows()):
        img = Image.open(OUT / "images" / r["image"]).convert("RGB")
        img.thumbnail((thumb, thumb))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{r['image']}\n{r['variety']}: {r['fresh_weight_g']:.0f} g fresh, {r['dry_weight_g']:.1f} g dry, "
                     f"{r['diameter_cm']:.0f} cm, {r['leaf_area_cm2']:.0f} cm²", fontsize=7, color="#52514e")
    fig.suptitle("hydroponics_lettuce: agc_lettuce (traits read back from disk)", x=0.01, ha="left", fontweight="bold")
    fig.savefig(EXAMPLE_DIR / "hydroponics_lettuce_agc_lettuce.png", dpi=110)
    plt.show()
