# Data contract

This project does not include source observations or generated image chips.

## Expected inputs

- LUCAS 2018 labels from Eurostat, using the `STR18` land-cover class and a stable point identifier.
- HLS S30 v2 imagery, the Sentinel-2 A/B component of NASA's Harmonized Landsat Sentinel-2 product.
- Six bands in this order: `B4`, `B3`, `B2`, `B8`, `B11`, `B12`.
- GeoTIFF chips with shape `(6, 224, 224)`, stored as `data/chips/<STR18>/<Point_ID>.tif`.
- Three text split files. Each row is `<STR18>/<Point_ID>`.

The located chip-preparation notebook used Germany (`DE`), HLS imagery from 2018, a cloud-cover filter below 10%, and 224-pixel chips. It referenced a course-managed Earth Engine feature collection. The original asset's owner, access status, and terms for reuse are not confirmed; use an independently available LUCAS source or obtain permission before reproducing that exact path. Do not upload restricted imagery or course data to GitHub.

## Split notes

The source notebook shuffled chip IDs with Python seed 42 and assigned 70% / 15% / 15% to train, validation, and test. Different notebook revisions point to different data folders and produce different split sizes. The larger saved split contained 644 / 138 / 139 rows; another executed run used 81 / 17 / 18 rows. The split text files themselves were not found with the local source files. Do not compare scores across revisions without recovering their exact chip manifests and split files.

The cleaned utility validates `STR18` values in the 1–10 class range and rejects duplicate point IDs, but still creates an unstratified random split rather than a spatially blocked split. It is a new reproducible split for a supplied manifest, not a reconstruction of either missing historical split.

## Credits and access

- HLS S30 v2: NASA LP DAAC, DOI 10.5067/HLS/HLSS30.002(https://doi.org/10.5067/HLS/HLSS30.002). NASA promotes full and open data sharing; cite the product and acquisition date.
- LUCAS: European Commission / Eurostat. See the LUCAS data portal(https://ec.europa.eu/eurostat/web/lucas/overview) and Eurostat reuse notice(https://ec.europa.eu/eurostat/help/copyright-notice); provide attribution and follow any dataset-specific conditions.
- Prithvi-EO v2 300M TL: see the official model card(https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL), which declares Apache-2.0.
- TerraTorch: see the official documentation(https://github.com/torchgeo/terratorch), which declares Apache-2.0.
