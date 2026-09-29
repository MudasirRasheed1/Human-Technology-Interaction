# Data (not committed)

K-EmoCon is distributed under a request-access licence and must not be redistributed, so nothing in this folder is committed.

1. Request access to K-EmoCon on Zenodo: https://doi.org/10.5281/zenodo.3931963
2. Extract these archives into `data/raw/` (the ~1.7 GB `debate_audios.tar.gz` is not needed):
   `metadata.tar.gz`, `data_quality_tables.tar.gz`, `emotion_annotations.tar.gz`, `e4_data.tar.gz`, `neurosky_polar_data.tar.gz`

   Expected layout:
   ```
   data/raw/metadata/subjects.csv
   data/raw/e4_data/<pid>/E4_*.csv
   data/raw/neurosky_polar_data/<pid>/*.csv
   data/raw/emotion_annotations/self_annotations/P<pid>.self.csv
   ```
3. Run `src/build_dataset.py`. It writes `data/processed/` (dataset.csv, train.csv, test.csv, baseline_stats.csv, ...).

To keep the data somewhere else, set the environment variable `KEMOCON_DATA_DIR` to a folder containing `raw/` (and `processed/` after the build).
