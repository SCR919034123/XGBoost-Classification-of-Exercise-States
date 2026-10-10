# SEAMS: Exercise-State Classification and RPE Prediction

Python code for the machine-learning analysis of multimodal physiological and sweat-sensing data acquired using the **SEAMS** wearable platform, associated with the manuscript *A general strain-decoupling architecture with robust interfacial coupling for stretchable bioelectronics*.

The repository contains two complementary **XGBoost** workflows evaluated using **leave-one-subject-out cross-validation (LOSO-CV)**:

| Module                            | Prediction task                                                             | Interpretation                             |
| --------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------ |
| **Exercise-State Classification** | Four exercise states: Rest, Warmup, Exercise, and Recovery                  | Class-specific SHAP analysis               |
| **RPE Prediction**                | Continuous Borg RPE score (6–20), using Warmup and Exercise data by default | Regression SHAP summary and decision plots |

Both modules are **offline research-analysis scripts**; they are not standalone real-time inference applications.

## 1. Project structure

Keep the original Python filenames and place both scripts in the **same folder**:

```
SEAMS-ML/
├── RPE_4state_LOSO_SHAP_submission_single_v5_Figure3_exact_style_clean.py
├── RPE Prediction.py
├── README.md
└── out_1Hz-1/
    ├── Subject_01.xlsx
    ├── Subject_02.xlsx
    └── ...
```

The two scripts write results to **separate output folders**, which are created automatically.

## 2. Installation (Anaconda)

Open **Anaconda Prompt**, then create and activate a Python environment:

```bash
conda create -n seams-ml python=3.11 -y
conda activate seams-ml
conda install -c conda-forge numpy pandas matplotlib scikit-learn xgboost shap openpyxl -y
```

## 3. Input data

* Place participant workbooks (`.xlsx`) in **`out_1Hz-1/`**. Each workbook represents **one participant**; its filename identifies that participant.
* Each worksheet represents a sensor channel (for example, `HR`, `Temp`, `SR`, `LA`, `GLU`, `PH`, `AA`, or `UA`). The worksheet should contain the timestamp column **`时间`** and a numeric measurement column, preferably named after the worksheet.
* Exercise-stage annotations should be supplied in a column such as **`状态`** or **`阶段`** (preferably in the `HR` worksheet). The scripts recognize **`静息`** (Rest), **`热身`** (Warmup), **`递增`** (Exercise), and **`恢复`** (Recovery).
* **RPE prediction additionally requires** numeric RPE annotations embedded in the stage text, for example **`热身 RPE=10`** or **`递增 RPE=15`**. The RPE model uses these values as regression targets.
* Keep the sensor worksheet names consistent with the provided scripts and source data. Both models support multimodal channels, while sensor naming also affects feature aggregation and SHAP labels.

Data files are not supplied in this repository. Use appropriately authorized and de-identified recordings.

## 4. Run the two modules

Open **Anaconda Prompt** and change to the repository folder (`cd` to the folder containing both scripts).

### A. Exercise-State Classification

```bash
python "RPE_4state_LOSO_SHAP_submission_single_v5_Figure3_exact_style_clean.py"
```

You can also open the file in **Spyder** and press **F5**. The supplied script automatically runs the classification and SHAP-figure stages through its `run_all_in_one()` function.

**Output folder:** `outputs_v15-1/` (beside the classification script).

### B. RPE Prediction

```bash
python "RPE Prediction.py" --do_global_shap
```

This runs **LOSO XGBoost regression**, evaluates held-out RPE predictions, and additionally fits an all-participant model for **global SHAP interpretation**. Omitting `--do_global_shap` runs the regression evaluation without that global interpretation step.

**Output folder:** `outputs_rpe_loso_warmup_exercise_test_density_ALL/` (relative to the working directory).

## 5. Main adjustable parameters

### Exercise-State Classification

The classification script overrides command-line arguments inside **`run_all_in_one()`**. To change settings, edit its **first `_sys.argv = [...]` list** near the end of the Python file.

| Parameter              | Supplied setting      | Meaning                                                                                       |
| ---------------------- | --------------------- | --------------------------------------------------------------------------------------------- |
| `--prediction_profile` | `protocol_calibrated` | Uses smoothing, protocol-order decoding, and a training-subject-derived stage-duration prior. |
| `--window_sec`         | **120 s effective**   | Analysis-window length under the default profile.                                             |
| `--stride_sec`         | 30 s                  | Step between adjacent windows.                                                                |
| `--sweat_delay`        | 300 s                 | Time shift applied to selected sweat biomarker channels.                                      |
| `--baseline_sec`       | 180 s                 | Initial baseline interval for normalization.                                                  |
| `--n_estimators`       | 250                   | XGBoost tree count for LOSO classification.                                                   |

For independent-window predictions without the default protocol post-processing, the script also offers `--prediction_profile paper_faithful`. The default `protocol_calibrated` mode **uses offline protocol information**; do not interpret its scores as unconstrained real-time performance.

### RPE Prediction

RPE settings can be changed directly in the **Anaconda command**:

| Parameter          | Default             | Meaning                                                                      |
| ------------------ | ------------------- | ---------------------------------------------------------------------------- |
| `--target_stages`  | `Warmup,Exercise`   | Stages included in RPE regression.                                           |
| `--history_sec`    | 900 s               | Maximum sensor-history interval used for each sample.                        |
| `--short_windows`  | `30,60,120,180,300` | Feature-extraction window lengths (seconds).                                 |
| `--sample_step`    | 5 s                 | Sampling interval for regression examples.                                   |
| `--sweat_delay`    | 120 s               | Sweat biomarker time shift.                                                  |
| `--n_estimators`   | 700                 | XGBoost regression tree count.                                               |
| `--shap_explain_n` | `0`                 | Number of samples explained by global SHAP; `0` means all available samples. |

Example with adjusted RPE parameters:

```bash
python "RPE Prediction.py" --do_global_shap --history_sec 600 --sample_step 10 --no_cache
```

**Important:** Add `--no_cache` when changing preprocessing settings or input data in the RPE workflow, so previously cached features are rebuilt. The two scripts have **different default sweat delays and normalization methods**; keep module-specific settings documented when comparing results.

## 6. Key results

| Classification: `outputs_v15-1/`                       | RPE regression: `outputs_rpe_loso_warmup_exercise_test_density_ALL/`                   |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| `subject_macro_f1_summary.csv` — subject-level metrics | `summary_rpe_reg_loso.csv` — LOSO metrics                                              |
| `all_loso_predictions.csv` — held-out predictions      | `heldout_predictions_rpe_substage_all.csv` — held-out, substage-aggregated predictions |
| `confusion_matrix_outputs/` — confusion matrix         | `heldout_metrics_rpe_substage_all.json` — pooled MAE, RMSE, R², and ±1 RPE accuracy    |
| `shap_outputs/` — class-wise SHAP data and figures     | `global_interpretation/shap_rpe_reg/` — RPE SHAP data and plots                        |

The all-participant SHAP models are trained for **interpretability** and should be distinguished from the **held-out LOSO evaluation**. Numerical results depend on the dataset and configuration used.

## Reference

Shi, C. *et al.* *A general strain-decoupling architecture with robust interfacial coupling for stretchable bioelectronics.* Manuscript. (Add the journal reference and DOI when available.)
