# CONTEXT.md — agrizar_cv_datasets

> Public-dataset survey for Agrizar computer vision (plant disease/pest/deficiency = "plague", and growth/ripeness). Not an application, and it does not train the final model. Output: findings reports + per-crop normalized datasets used to pretrain or extend Agrizar's own data.
> **Ticket:** GRZR-16 · **Repo:** github.com/CreAI-mx/agrizar_CV_datasets · **Full docs:** Confluence "Investigación CV para Agrizar" (linked from the ticket).

## Status (GRZR-16, 2026-10-13)
- Plague datasets: 12 reviewed, 4 discarded → `PLAGUE_DATASET_FINDINGS.md`.
- Growth datasets: 17 reviewed → `GROWTH_DATASET_FINDINGS.md`.
- Normalized datasets: 6 crops (strawberry, blueberry, raspberry, tomato, bell pepper, cucumber), 47,837 images → `normalized/`.
- Pending: trained-model surveys (plague, growth), legal review, berry data gap (see Open items).

## Domain Vocabulary
- **Plague** — diseases + insect pests (*plaga*) + nutrient deficiencies. Not only diseases.
- **`condition` / `maturity`** — the two label axes. condition = plague; maturity = flower, bud, fruit stage. Separate `classes.txt`, trained as separate models.
- **`group`** — leakage key in `manifest.csv` (near-duplicates and augmented copies share one). **`label_space`** — detection classes the image's source annotates; mask the rest.
- **Verified negative** — empty detection file, only under `condition/`. No file = not annotated.
- **Common fruit-stage scale** — flower/bud → green → white/pale → turning → ripe (+ overripe).
- **dHash** — perceptual hash for near-duplicates (64-bit, confirmed with 256-bit ≤ 10 bits), written with `0x` prefix.
- **AgML** — Project-AgML on Hugging Face, re-hosts ag datasets with cards.

## Business Rules
- "Healthy" only from an explicit label, never from missing boxes.
- Split train/val/test by `group`, never by image.
- Labels that cannot be trusted or reconciled are dropped and counted in the notebook (correctness first).
- `prelabel/` is unverified: never merge into `normalized/` before review.
- Every number in a findings file comes from executed notebook output; copy the old findings to `archive/` before replacing; never edit `archive/`.
- Non-commercial subsets (PlantSeg, blueberry segmentation, Laboro tomato; LeafNet2.0 is excluded) must be filtered with the manifest `license` column before commercial use. Web/watermarked images need a legal check.

## External Integrations
- **Hugging Face** — account `bebedovis`; LeafNet2.0 is gated (non-commercial), each teammate accepts it.
- **Roboflow MCP** — workspace `veggies-h2dsa`; public exports broken, route = fork → raw version → COCO export; free quota exhausted. Ask before forking.
- **Mendeley / 4TU public APIs** — zip downloads into `data/`.
- **Confluence** — detailed documentation and the trained-model surveys.

## Key Concepts
- Notebooks are **generated** by `tools/build_*_nb.py`: edit the builder, rebuild, execute. Never edit a notebook by hand. Run the normalization notebooks one at a time (RAM).
- One study = notebook + findings md + `charts/` + `examples/`. Draw labels on samples and look at them (this is how most label bugs were found).
- `normalized/<crop>/` = `images/`, `labels/{classification,detection,segmentation}/`, `manifest.csv`. Excluded on purpose: chili, LeafNet2.0, AgML strawberry/avocado, Imtiaz tomato, broken raspberry point labels.
- Top traps (details in `CLAUDE.md`): AgML class names (use the card); Roboflow augmented copies; LeafNet2.0 ⊃ PlantVillage, PlantWild ⊃ PlantSeg; missing class names (Strawberry_12, PlantDoc); Laboro EXIF-rotated boxes; pandas 2.3 segfault on `1e00…` hashes.
- **Main data gap:** blueberry/raspberry diseases (18–47 web images per class), pests on berry plants, tomato whitefly/*Tuta absoluta*, cucumber growth images.

## Open items (GRZR-16)
- [ ] Legal review: LeafNet2.0, PlantSeg, Laboro, GrowliFlower, leaf counting, web-image rights.
- [ ] Close the berry data gap (consider own field images).
- [ ] Review bell-pepper segmentation prelabels with `tools/seg_labeler.py`; label cucumber growth.
- [ ] Agree the common growth-stage scale with the team.
- [ ] Decide on PlantDoc's 5 missing classes.
- [ ] Trained-model surveys: plague and growth.

## Read next
`CLAUDE.md` (conventions, traps) · `PLAGUE_DATASET_FINDINGS.md` · `GROWTH_DATASET_FINDINGS.md` · `README.md` (Spanish, team-facing) · `tools/`
