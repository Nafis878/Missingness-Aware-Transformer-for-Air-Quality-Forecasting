# Ensemble leakage audit

Validation-calibrated ensembles are fit only from `*_val.npz` bundles and then applied to aligned `*_test.npz` bundles.
The audit checks row-key overlap `(station_id, anchor_time)` and chronological ordering for every available member/seed pair.

Checked 31 member/seed pairs; total overlap rows = 0.
All chronological gaps positive: True.
- `24_validation_calibrated_ensembles.py`: loads validation for fitting = True; loads test bundles for application = True.
- `25_global_validation_ensembles.py`: loads validation for fitting = True; loads test bundles for application = True.
