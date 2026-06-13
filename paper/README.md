# Manuscript

`main.tex` is the submission draft (default class: `elsarticle`, targeting an
applied/environmental venue such as *Expert Systems with Applications* or
*Environmental Modelling & Software*; swap to `IEEEtran` for an IEEE venue).

## Build

```bash
cd paper
latexmk -pdf main.tex          # or: pdflatex main; bibtex main; pdflatex main; pdflatex main
```

## Dependencies on generated assets

The manuscript pulls figures from `../outputs/figures/` and
`../outputs/beijing/figures/` (via `\graphicspath`) and `\input`s the
`decision_summary.tex` table from `../outputs/tables/`. Regenerate them first:

```bash
# from the repo root
python scripts/05_ablations.py --robustness --config config.yaml          # fine-grained sweep
python scripts/05_ablations.py --robustness --config config_beijing.yaml
python scripts/07_make_paper_assets.py --config config.yaml \
    --secondary-config config_beijing.yaml --skip-interpretability
python scripts/07_make_paper_assets.py --config config_beijing.yaml --skip-interpretability
```

Key figures used: `robustness_curve.pdf`, `crossover_combined.pdf`,
`stratified_gap.pdf`. The result tables in the text are hand-set from the
generated CSVs (`outputs/tables/*.csv`) so the numbers are traceable to the
single source of truth; `decision_summary.tex` is `\input` directly.

Numbers in the prose are audited against the regenerated CSVs as part of the
release checklist (see `../UPGRADE_LOG.md`).
