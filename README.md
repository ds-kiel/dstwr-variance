# Precise Ranging: Bias and Variance of DS-TWR and TDoA Extraction

Simulation code, firmware, testbed data, and plotting scripts for the paper

> **Precise Ranging: Modeling Bias and Variance of Double-Sided Two-Way Ranging with TDoA Extraction under Multipath and NLOS Effects**
> Patrick Rathje, Christian Richter, Olaf Landsiedel

The paper derives the bias and variance of Double-Sided Two-Way Ranging (DS-TWR) and of the Time Difference of Arrival (TDoA) that passive devices extract from overheard DS-TWR exchanges, including multipath and non-line-of-sight (NLOS) effects. This repository contains everything needed to reproduce its figures.

## Repository layout

| Path | Content |
| --- | --- |
| `scripts/models.py` | Analytic bias and variance model for DS-TWR and TDoA extraction |
| `scripts/sim_tdoa.py` | Monte Carlo simulation of DS-TWR exchanges with a passive listener |
| `scripts/export_*.py` | One script per paper figure, writing PDFs to `export/` |
| `scripts/cache_*.py` | Pre-processing of raw testbed logs into cached data frames |
| `scripts/logs.py`, `scripts/testbed/` | Log parsing and testbed node positions |
| `data/trento_a/` | Raw logs of the testbed runs on the CLOVES testbed (7 DWM1001 nodes) |
| `cache/` | Cached intermediate results so that figures can be rebuilt without re-parsing the logs |
| `export/` | The exported figures used in the paper |
| `src/`, `prj.conf`, `CMakeLists.txt` | Zephyr firmware for the DWM1001 nodes |
| `override/` | Patched Zephyr DW1000 driver required by the firmware |

## Reproducing the figures

Dependencies are managed with [uv](https://docs.astral.sh/uv/), which installs a matching Python (3.10 or newer) and the locked packages from `uv.lock` on first use:

```bash
uv sync
```

Output and cache directories are configured in `.env` (`EXPORT_DIR=export`, `CACHE_DIR=cache`). All scripts are run from the repository root:

```bash
./export_all.sh                        # all figures
uv run scripts/export_nlos_sweep.py    # or a single figure
```

The scripts use the cached intermediate results in `cache/`. Delete the corresponding cache file to recompute a figure from the raw logs or to rerun a simulation. Cropped variants (`*_cropped.pdf`) are produced with `pdfcrop` from TeX Live when it is available on the `PATH`.

## Firmware

The firmware in `src/` runs DS-TWR rounds between all nodes and logs raw timestamps over serial. It targets Zephyr v2.7.2 and the `decawave_dwm1001_dev` board. The stock Zephyr DW1000 driver does not provide the required timestamping precision, so the files in `override/` have to be copied over the Zephyr tree first (or use the [patched Zephyr branch](https://github.com/prathje/zephyr/tree/feature/dwm_1001_ranging_api)).

```bash
cp -Rf override/* $ZEPHYR_BASE/
west build -b decawave_dwm1001_dev --pristine auto
west flash
```

The included `Dockerfile` and `docker-compose.yaml` provide a matching Zephyr build environment:

```bash
docker compose up -d --build
docker compose exec -it build /bin/bash
cp -Rf /app/override/* /zephyr/zephyr/
west build -b decawave_dwm1001_dev --pristine auto
```

## Citation

If you use this code or data, please cite the paper above.
