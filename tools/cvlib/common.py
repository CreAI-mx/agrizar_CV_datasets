"""Helpers shared by growth.ipynb and the normalization notebook (Hugging Face / AgML access, Roboflow file names)."""
import re
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def card_class_names(repo, column="objects"):
    # class names from the dataset card: the names embedded in AgML parquet files can differ, so the card is the source of truth
    info = HfApi().dataset_info(repo).card_data["dataset_info"]
    info = info[0] if isinstance(info, list) else info  # multi-config cards list one entry per config
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


def rf_stem(file_name):
    # Roboflow names every copy <source>_<ext>.rf.<hash>.<ext>: copies of one photo share the stem
    return re.sub(r"\.rf\.[0-9a-zA-Z]+", "", file_name)
