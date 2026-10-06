# Prithvi-EO for LUCAS land-cover classification

An MSc course project exploring image-level land-cover classification with the Prithvi-EO v2 300M temporal model and LUCAS labels. The located training code consumes six-band, 224 × 224 GeoTIFF chips from the NASA Harmonized Landsat Sentinel-2 (HLS) S30 product and predicts ten `STR18` classes.

The repository package is being reconstructed from the local course notebooks. The original Earth Engine asset, source chips, saved splits, checkpoints, and the later experiment logs are not included.

## Problem and approach

LUCAS supplies in-situ land-cover labels, while HLS provides harmonized 30 m surface-reflectance imagery. The notebook workflow samples Germany LUCAS points, creates chips from six HLS bands (`B4`, `B3`, `B2`, `B8`, `B11`, `B12`), then applies a linear classification head to the pretrained Prithvi-EO v2 300M TL backbone. The experiment compares a frozen backbone with partial fine-tuning of the final four Transformer blocks. The classifier uses the backbone CLS token, dropout, and a ten-class linear layer.

The located split-generation code shuffles chip IDs with seed 42 and assigns 70% / 15% / 15% to train, validation, and test. The files are point-ID based, but spatial or regional generalisation was not tested. The source training script selects checkpoints by validation accuracy and also reports macro-F1; these choices must be retained when comparing against historic runs.

The cleaned split utility rejects duplicate point IDs and class values outside 1–10. Its seeded split is a simple random partition, not class-stratified or spatially blocked; it does not reproduce a historical split without the original manifest and split files. Training uses mixed precision on CUDA and full precision on CPU.

## Results and limits

The available evidence does not support the portfolio's previously reported Macro-F1 change from 0.22 to 0.59. One executed local notebook run has 18 test chips and reports Macro-F1 0.1333 and accuracy 0.5000. A later notebook records a 921-chip split (644 / 138 / 139) but has no saved test metrics. The score pair 0.22 → 0.59 should remain unpublished until the exact runs, splits, and outputs are recovered and checked.

This is an exploratory course result, not a validated operational land-cover product. The current evaluation is small, split details differ across notebook revisions, and the LUCAS label represents a surveyed point rather than a pixel-perfect image-wide reference.

## Project files and attribution

The located project files contain the LUCAS chip workflow, the Prithvi classification wrapper, training and evaluation code, and experiment scripts. The notebooks identify this as an MSc GeoAI course project and use a course-managed Earth Engine feature collection. Public materials should preserve that course context and credit the upstream data, model, and toolkit sources.

## Data and model access

No imagery, labels, Earth Engine assets, or weights are distributed here. See [`data/README.md`](data/README.md) for input layout, access notes, and credits. The original chip-preparation notebook refers to a course-managed Earth Engine asset whose access and reuse terms have not been confirmed, so that asset ID is not copied into this repository.

The HLS data are openly shared by NASA; cite the HLS S30 v2 DOI listed in the data guide. LUCAS data reuse follows Eurostat's copyright and attribution provisions. The Prithvi-EO v2 model card and TerraTorch toolkit declare Apache-2.0. These upstream terms do not automatically license this project's code.

## Reproduce

The current package accepts a prepared chip directory and three split files. Full training has not been re-run from this cleaned package because the original chip files, exact larger-run split, and checkpoints were not found locally.

```text
data/chips/
  1/<Point_ID>.tif
  2/<Point_ID>.tif
  ...
  10/<Point_ID>.tif
data/splits/
  lucas_train.txt
  lucas_val.txt
  lucas_test.txt
```

Each split line uses the format `<STR18>/<Point_ID>`, for example `3/11021028`. Install PyTorch for the target hardware first, then install the remaining requirements and run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/create_splits.py \
  --manifest data/chips_LC_class.csv \
  --output-dir data/splits \
  --seed 42
python -m prithvi_lucas.train \
  --data-root data/chips \
  --train-split data/splits/lucas_train.txt \
  --val-split data/splits/lucas_val.txt \
  --test-split data/splits/lucas_test.txt \
  --unfreeze-last-blocks 4
```

Set `--unfreeze-last-blocks 0` for a frozen-backbone run. This command documents the cleaned workflow; it has not yet been validated end to end with the original input data and pretrained weights.

## Repository layout

```text
data/                  input contract and access notes (no data included)
scripts/               split generation
src/prithvi_lucas/     dataset and model code
```

## License

No license has been assigned to this project's code yet. Upstream data, weights, and toolkit terms are listed separately in the data guide. A project-code license must be selected before inviting reuse.
