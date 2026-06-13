# Running the secondary datasets (Beijing, Delhi) on Google Colab (GPU)

The external-validity datasets are trained on Colab GPU and the artifacts
brought back for local asset regeneration. Each grid (11 models, 3 seeds for
learned models, two robustness suites) is ~19–20 h on a desktop CPU but a few
hours on a Colab GPU. Everything is file-existence resumable, so disconnects
are harmless: just re-run the cell.

- **Beijing** (UCI id 501) — auto-downloaded; see below.
- **Delhi** (CPCB; Mendeley `bzhzr9b64v`) — the intermediate-imputability
  third network; CSVs are placed manually (see the Delhi section).

## Beijing

## Steps

1. **Zip the repo locally** (checkpoints/predictions not needed — only code
   + configs; the Beijing data is downloaded on Colab):

   ```powershell
   # from the repo root's parent directory
   Compress-Archive -Path air-transformer -DestinationPath air-transformer.zip `
       -Force  # excludes nothing; ~30 MB without Dhaka raw/outputs is fine too
   ```

2. **In Colab** (Runtime → Change runtime type → **T4 GPU**), upload
   `air-transformer.zip` (Files panel or Drive), then run:

   ```python
   !unzip -q air-transformer.zip
   %cd air-transformer
   !pip install -q pyarrow pyyaml scikit-learn statsmodels seaborn openpyxl
   !python scripts/colab_run_beijing.py
   ```

   Colab ships its own (CUDA) torch/pandas/numpy/matplotlib — do not
   reinstall those. The script prints the GPU name, runs:

   - `01b_prepare_beijing.py` — downloads the UCI zip (~50 MB), cleans,
     writes the parquet;
   - `03_train_baselines.py` — persistence, seasonal_naive, gru, gru_d,
     dlinear, patchtst, two_stage_knn, two_stage_saits × seeds 42/43/44;
   - `04_train_proposed.py` ×3 — proposed, proposed_md (miss-dropout),
     variant_B × seeds 42/43/44;
   - `05_ablations.py --robustness` — MCAR + outage corruption at 10/30/50%;

   and finally zips `outputs/beijing/` + `data/processed/beijing/` into
   **`beijing_artifacts.zip`**.

3. **Download `beijing_artifacts.zip`**, unzip it into the local repo root
   (paths inside the zip line up with `outputs/beijing/...` and
   `data/processed/beijing/...`), then regenerate the paper assets locally:

   ```powershell
   python scripts/07_make_paper_assets.py --config config_beijing.yaml --skip-interpretability
   python scripts/07_make_paper_assets.py --secondary-config config_beijing.yaml --skip-interpretability
   ```

   The first writes the Beijing tables/figures (`main_results_pm25`,
   `robustness_rmse`, ... under `outputs/beijing/`); the second refreshes the
   Dhaka assets **and** the `cross_dataset_summary` table.

## Delhi

The 6 CPCB station files (CC BY 4.0) **auto-download** from their content-
addressed Mendeley URLs, so the run is one command. (Schema, units and bounds
in `config_delhi.yaml` were verified against the published files: integer
year/month/day/hour timestamp, `AT`/`Ozone`/`NOx` folded to `Temp`/`O3`/`NOX`,
numeric `WD` → `wd_sin`/`wd_cos`; the series is published complete, so Delhi is
the *complete-network* anchor of the imputability study.)

1. **In Colab** (GPU runtime), after unzipping the repo:

   ```python
   %cd air-transformer
   !pip install -q pyarrow pyyaml scikit-learn statsmodels seaborn openpyxl
   !python scripts/colab_run_delhi.py
   ```

   The runner mirrors Beijing: `01c_prepare_delhi.py` (download + clean +
   parquet), `03`/`04` (same model grid × seeds 42/43/44), `05 --robustness`,
   then zips `outputs/delhi/` + `data/processed/delhi/` into
   **`delhi_artifacts.zip`**. (If `archive.ics`-style access is ever blocked,
   set `data.archive_url` in `config_delhi.yaml` to an offline mirror zip, or
   drop the CSVs into `data/raw/delhi/` manually — the runner then skips the
   download.)

2. **Download `delhi_artifacts.zip`**, unzip into the local repo root, then
   regenerate the full **3-dataset** asset set locally:

   ```powershell
   python scripts/07_make_paper_assets.py --config config_delhi.yaml --skip-interpretability
   python scripts/07_make_paper_assets.py --config config.yaml `
       --secondary-config config_beijing.yaml --tertiary-config config_delhi.yaml `
       --skip-interpretability
   ```

   The first writes Delhi's own tables/figures (incl. its `imputability` table);
   the second refreshes Dhaka and builds the n-way `cross_dataset_summary`,
   `crossover_combined`, and the headline `imputability_crossover` figure +
   `decision_by_imputability` table across all three networks.

## Notes

- Checkpoints are saved with CPU-moved state dicts, so GPU-trained models
  load on CPU machines without `map_location`.
- Determinism: `seed_everything` sets `CUBLAS_WORKSPACE_CONFIG` and
  `torch.use_deterministic_algorithms(True)` on GPU too. CPU and GPU runs at
  the same seed are still **not bit-identical** — Beijing numbers are
  GPU-trained, which RESULTS.md states.
- If a deterministic-algorithms error ever surfaces on a CUDA op, set
  `device: cpu` in `config_beijing.yaml` for that step and report it; none of
  the models use ops without deterministic CUDA paths.
- `TODO(data)`: if Colab cannot reach archive.ics.uci.edu, download
  `beijing+multi+site+air+quality+data.zip` from
  https://archive.ics.uci.edu/dataset/501 manually and place the 12
  `PRSA_Data_*.csv` files in `data/raw/beijing/` before step 2; the runner
  then skips the download.
