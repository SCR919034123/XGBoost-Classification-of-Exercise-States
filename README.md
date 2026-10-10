# SEAMS Multimodal Exercise-State Classification

**XGBoost · Leave-One-Subject-Out Cross-Validation (LOSO-CV) · SHAP Interpretability**

## Overview

This repository contains the Python analysis script for **four-state exercise classification** using multimodal recordings acquired with the **strain-insensitive electrode architecture for multimodal sensing (SEAMS)** wearable platform.

The analysis is associated with the research manuscript:

> Changrui Shi *et al.* **A general strain-decoupling architecture with robust interfacial coupling for stretchable bioelectronics.** (Manuscript)

The study uses multimodal physiological, sweat-dynamics, and biochemical information to investigate changes during exercise. The classification task distinguishes four stages:

1. **Rest**
2. **Warmup**
3. **Exercise** (incremental exercise)
4. **Recovery**

The provided script performs signal preprocessing, feature extraction, four-class XGBoost classification, subject-independent evaluation, and SHAP interpretation of the **classification model**. It is related primarily to the exercise-state analysis described around **Figure 6** of the manuscript.

**Scope:** Although the manuscript also reports continuous fatigue prediction from ratings of perceived exertion (RPE), **this particular script does not implement the separate XGBoost RPE regression workflow**. An `RPE=...` value may be read from stage annotations as metadata, but the predicted target here is the **four-class exercise state**.

## Analysis workflow

1. Read one multi-sheet Excel workbook per participant.
2. Align channel timestamps, including optional correction of whole-day date offsets.
3. Resample the sensor streams to **1 Hz** and optionally compensate for the delay of selected sweat biomarker signals.
4. Normalize sensor signals relative to an initial baseline and extract temporal/statistical features from windows or annotated substages.
5. Train and evaluate a four-class **XGBoost** model using **LOSO-CV**: each participant is held out for testing while the other participants form the training set.
6. Export per-subject classification metrics, predictions, confusion matrices, and figures.
7. Train a separate global four-class model for **SHAP interpretation** and generate class-specific visualizations.

**Important distinction:** The global SHAP model is trained on the pooled feature dataset for interpretation; it is **not** the held-out model of an individual LOSO fold.

## Software requirements

- **Python 3.10 or newer** (Python 3.11 is a reasonable choice).
- **Anaconda or Miniconda**; alternatively, a compatible Python installation.
- Python packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `xgboost`, `shap`, and `openpyxl`.

Example installation using **Anaconda Prompt**:

```bash
conda create -n seams-ml python=3.11
conda activate seams-ml
conda install -c conda-forge numpy pandas matplotlib scikit-learn xgboost shap openpyxl
```

Run the script using the same environment in which these packages are installed. If using Spyder, select an interpreter associated with that environment.

## Input data format

Place participant Excel workbooks (`.xlsx`) in the folder **`out_1Hz-1`**, immediately beside the Python script.

A suggested project layout is:

```text
SEAMS-Exercise-Classification/
├── RPE_4state_LOSO_SHAP_submission_single_v5_Figure3_exact_style_clean.py
├── README.md
├── out_1Hz-1/
│   ├── Subject_01.xlsx
│   ├── Subject_02.xlsx
│   └── Subject_03.xlsx
└── outputs_v15-1/                 # Created automatically during execution
```

Each Excel workbook represents **one participant**. The **workbook filename (without `.xlsx`)** is used as the participant ID.

Each worksheet holds a sensor stream; its worksheet name normally matches the signal column. The following channel names are used by the analysis and/or its SHAP visualization:

| Worksheet / signal | Description |
| --- | --- |
| `HR` | Heart rate |
| `Temp` | Skin temperature |
| `GRS` or `GSR` | Galvanic skin response; `GRS` is used in some code defaults |
| `SR` | Sweat rate |
| `TEC` | Sweat-related electrical channel; displayed as SIC in SHAP labels |
| `LA` | Lactate |
| `GLU` | Glucose |
| `AA` | Ascorbic acid |
| `UA` | Uric acid |
| `PH` | pH |

Each worksheet must include a timestamp column named exactly **`时间`**, plus a numeric sensor-value column (preferably named after the worksheet). At least one worksheet must also contain exercise-stage annotations. The script looks for annotation columns such as `状态` and `阶段`; it preferentially uses annotations from the `HR` worksheet when available.

For example, an `HR` worksheet could look like:

| 时间 | HR | 状态 |
| --- | ---: | --- |
| 2026-01-01 08:00:00 | 72 | 静息 RPE=6 |
| 2026-01-01 08:05:00 | 94 | 热身 RPE=9 |
| 2026-01-01 08:20:00 | 151 | 递增 RPE=15 |
| 2026-01-01 08:45:00 | 103 | 恢复 RPE=10 |

*The example values above are illustrative, not study data. Actual recordings should contain the available time-series samples, rather than just these four example rows.*

The current stage parser recognizes these **Chinese keywords**:

| Annotation contains | Classification label |
| --- | --- |
| `静息` | Rest |
| `热身` | Warmup |
| `递增` | Exercise |
| `恢复` | Recovery |

Other labels are classified as `Unknown` and excluded from feature construction. A stage annotation may optionally contain an `RPE=...` value; it is not required for four-state classification.

**Data sharing:** Check research consent, ethics approval, and institutional requirements before uploading participant recordings to a public repository. The code can be shared independently of private participant data.

## Quick start: run directly in Anaconda / Spyder

**This is the recommended method for the supplied version of the script. No command-line arguments are required.**

1. Download the Python `.py` file and place it in a project folder.
2. Create a subfolder named **`out_1Hz-1`** alongside that file.
3. Place the participant `.xlsx` workbooks in `out_1Hz-1`.
4. Open the script with **Spyder** (launched from Anaconda Navigator or from the correct conda environment).
5. Click **Run** (or press **F5**).
6. Wait for training, evaluation, SHAP analysis, and figure export to complete.
7. View the generated results in **`outputs_v15-1`**, located beside the script.

An equivalent way to start the **unchanged** script from Anaconda Prompt is:

```bash
conda activate seams-ml
python RPE_4state_LOSO_SHAP_submission_single_v5_Figure3_exact_style_clean.py
```

The script's final `run_all_in_one()` function performs **two consecutive operations**: (1) LOSO classification and global SHAP generation, followed by (2) redrawing SHAP figures in the configured style.

### Where the current script gets its paths

The following settings are **hard-coded near the end of the script** in `run_all_in_one()`:

- Input: `<script folder>/out_1Hz-1/*.xlsx`
- Output: `<script folder>/outputs_v15-1/`
- Restyled SHAP figures: `<script folder>/outputs_v15-1/shap_outputs/Figure3_style/`

The `Figure3_style` folder and `Figure3_SHAP_4state` filenames are **internal plot-style names**; they do **not** mean that these figures correspond to Figure 3 of the SEAMS manuscript.

**Important:** `run_all_in_one()` replaces `sys.argv` internally. Therefore, simply appending flags such as `--window_sec 60` to the `python filename.py` command **will not change the run**. To adjust settings without modifying the entry point, edit the argument lists inside `run_all_in_one()` as described next.

## Adjustable parameters

### How to change settings in Spyder

Near the **end of the `.py` file**, find `def run_all_in_one():`. In its **first** `_sys.argv = [...]` list (used for classification), add or change parameters as needed. For example:

```python
_sys.argv = [
    'classification',
    '--excel_dir', str(_base / 'out_1Hz-1'),
    '--excel_pattern', '*.xlsx',
    '--out', str(_out),
    '--prediction_profile', 'protocol_calibrated',
    '--window_sec', '120',
    '--stride_sec', '30',
    '--sweat_delay', '300',
    '--baseline_sec', '180',
    '--n_estimators', '250',
    '--shap_explain_n', '0',
    '--shap_decision_n', '0'
]
```

This **replaces only the first argument list**, not the rest of the script. The second `_sys.argv = [...]` list controls the later SHAP redrawing stage, including export formats, resolution, and plot-axis settings.

### Main analysis parameters

| Parameter | Current behavior / default | What it controls |
| --- | --- | --- |
| `--prediction_profile` | `protocol_calibrated` | Evaluation/post-processing profile. See below. |
| `--sample_level` | `window` | Extract features from time windows; `substage` selects annotated stage/substage rows. |
| `--window_sec` | **120 s in the supplied all-in-one run** | Window length. The parser specifies 60 s, but `protocol_calibrated` changes it to 120 s unless explicitly set. |
| `--stride_sec` | 30 s | Distance between consecutive window starts; smaller strides give more overlapping windows. |
| `--sweat_delay` | 300 s | Temporal compensation for channels listed in `--sweat_cols`. Set to `0` to disable. |
| `--sweat_cols` | `LA,PH,AA,UA,GLU` | Sweat biomarker channels to shift in time. |
| `--baseline_sec` | 180 s | Initial interval used to establish each session's normalization baseline. |
| `--n_segments` | 3 | Number of subdivisions used for within-window/substage feature statistics. |
| `--min_window_coverage` | 0.8 | Minimum acceptable sample coverage of a candidate window. |
| `--min_label_fraction` | 0.8 | Required fraction of the majority exercise label in a window. |
| `--feature_mode` | `compact` | `compact` uses a selected feature subset; `all` includes more numerical sensor features. |
| `--trajectory_mode` | `none` | Optional across-window features: `none`, `causal`, or `offline`. |
| `--n_estimators` | 250 | Number of XGBoost trees in the LOSO classification model. |
| `--max_depth` | 2 | Maximum tree depth; larger values can increase model complexity. |
| `--n_model_seeds` | 1 | Number of models trained with different seeds for probability averaging. |
| `--seed` | 42 | Random seed for reproducibility. |
| `--xgb_n_jobs` | 1 | Number of XGBoost worker threads. |
| `--clip_z` | 20 | Clipping bound applied after baseline normalization. |
| `--shap_tec_clip_z` | 30 | TEC clipping setting for the separate SHAP feature dataset; not a LOSO-metric setting. |

**Sweat-delay direction:** The implementation uses a negative shift (`shift(-delay)`) of selected sweat channels. A value of `300` aligns measurements from a later time point with an earlier reference time, reflecting an assumed sweat-response delay. This also removes trailing rows without aligned sweat values. Choose the delay based on the experimental protocol, not simply to increase accuracy.

### Prediction profiles

| Setting | Behavior |
| --- | --- |
| `paper_faithful` | Independent-window model predictions, without the automatic smoothing and sequence decoding enabled by the other profiles. This is the **code's profile name**, not a guarantee of reproducing every manuscript setting. |
| `stable` | Adds probability smoothing and correction of isolated short-lived label flips. |
| `high_accuracy` | Adds protocol-order sequence decoding to the stability operations. |
| `protocol_calibrated` | Adds a soft phase-duration prior derived from the LOSO **training participants** and uses 120-second windows unless overridden. This is the **current script default**. |

Temporal post-processing and protocol-based priors can change classification performance. In particular, the `protocol_calibrated` setting assumes a known sequence of exercise stages and uses sequence-level information. Results from this offline setting should **not be presented as equivalent to unconstrained real-time independent-window predictions**.

For a comparison focused on independent windows, choose `--prediction_profile paper_faithful` and explicitly set the desired `--window_sec` and `--stride_sec`. Keep settings identical when comparing experimental conditions.

### SHAP and figure parameters

| Parameter | Default in the all-in-one run | Meaning |
| --- | --- | --- |
| `--shap_explain_n` | **0** | Number of rows to explain with global SHAP; `0` means **all available rows** in this implementation. |
| `--shap_decision_n` | **0** | Rows used for class-specific decision plots; `0` means **all rows**. |
| `--shap_bg_n` | 500 | Maximum stratified background samples for the global SHAP calculation. |
| `--shap_model_estimators` | 400 | Number of trees in the separately trained global SHAP model. |
| `--shap_model_feature_mode` | `all` | Feature selection for the global interpretation model. |

For a faster exploratory run, consider setting `--shap_explain_n` and `--shap_decision_n` to positive values (for example, `300` and `100`). The SHAP redraw stage also has its own options in the **second** argument list, such as `--dpi`, `--formats`, axis limits, and tick spacing. Changing these settings affects plotting, not the underlying LOSO predictions.

## Main output files

After a successful run, `outputs_v15-1/` contains results similar to:

```text
outputs_v15-1/
├── window_feature_dataset.csv
├── subject_macro_f1_summary.csv
├── all_loso_predictions.csv
├── fig4e_like_subject_macro_f1.png
├── fig4e_subject_independent_metrics_boxplot.png
├── intermediate/
│   └── Subject_01__merged_1hz.csv
├── confusion_matrix_outputs/
│   ├── four_class_confusion_matrix.png
│   ├── four_class_confusion_matrix_counts.csv
│   └── four_class_confusion_matrix_percent.csv
└── shap_outputs/
    ├── global_4class_model_for_shap.json
    ├── global_4class_shap_raw.npz
    ├── sensor_level_shap_feature_importance.csv
    ├── rest/
    ├── warmup/
    ├── exercise/
    ├── recovery/
    └── Figure3_style/
        ├── Figure3_SHAP_4state.png
        ├── Figure3_SHAP_4state.svg
        └── Figure3_SHAP_4state.pdf
```

Selected outputs are available in multiple formats, including PDF and SVG. The exact file set depends on enabled options and whether each stage completes successfully.

- **`subject_macro_f1_summary.csv`**: per-held-out-participant classification metrics.
- **`all_loso_predictions.csv`**: true/predicted exercise-state labels and model probabilities for LOSO samples.
- **`four_class_confusion_matrix.*`**: aggregated four-class classification results.
- **`global_4class_shap_raw.npz`**: saved SHAP data used for the final SHAP plots.
- **`Figure3_SHAP_4state.*`**: composite classification-SHAP visualization.

## Troubleshooting

- **`No Excel files found`** — Confirm that `out_1Hz-1` is beside the `.py` file and contains `.xlsx` workbooks.
- **`missing 时间 column`** — Verify the exact Chinese timestamp-column name in every worksheet.
- **`No valid substage feature rows were built`** — Check the Chinese stage keywords, timestamps, recording duration, and window settings.
- **Too few final samples** — Check `--sweat_delay`, time alignment, and window coverage. A large delay can shorten the usable recording.
- **Missing SHAP figures** — Confirm that `shap_outputs/global_4class_shap_raw.npz` was generated. The restyling step requires saved SHAP data.
- **Slow execution or large figures** — Reduce the number of explained/decision-plot samples, reduce plot DPI for preview runs, or use fewer model trees during testing.

## Reproducibility and interpretation

The manuscript reports performance from its specified experimental dataset and analysis procedures; those values should **not be assumed for new participants, altered sensor inputs, or different processing settings**. Keep the input data, software versions, sampling settings, and post-processing profile consistent when comparing results.

Because the source code includes optional temporal smoothing, sequence decoding, and protocol-based priors, state clearly which configuration produced each reported metric. The delivered program is an **offline research-analysis script**, not a ready-to-deploy real-time inference application.

## Related manuscript

Shi, C., Gao, Y., Cao, J., Jia, H., Xue, R., Zhou, J., Tao, Y., Ruan, Q., and Ren, Y. **A general strain-decoupling architecture with robust interfacial coupling for stretchable bioelectronics.** Manuscript; bibliographic details/DOI can be added when available.
