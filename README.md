SEAMS: Exercise-State Classification and RPE Prediction
XGBoost · Leave-One-Subject-Out Cross-Validation (LOSO-CV) · SHAP Interpretability · CPU-based analysis
Overview
This repository contains two Python scripts for analyzing multimodal recordings obtained with the strain-insensitive electrode architecture for multimodal sensing (SEAMS) wearable platform.
The analysis is associated with the research manuscript:
> Changrui Shi *et al.* **A general strain-decoupling architecture with robust interfacial coupling for stretchable bioelectronics.** (Manuscript)
The scripts address two related, but distinct, tasks:
Script	Prediction target	Main outputs
`Exercise-State Classification.py`	Four exercise states: Rest, Warmup, Exercise (incremental exercise), and Recovery	Subject-independent classification metrics, predictions, confusion matrices, and SHAP interpretations
`RPE Prediction.py`	Continuous rating of perceived exertion (RPE) during the Warmup and Exercise stages	LOSO regression metrics, predicted versus observed RPE, derived fatigue-level summaries, and optional SHAP interpretations
Both programs perform offline research analysis using XGBoost. They share an Excel-based input-data format but use different feature-construction, training, evaluation, and output procedures. Results from the two scripts should not be treated as interchangeable.
In the classification script, an `RPE=...` annotation can be parsed as metadata, but the target is the four-class exercise state. In the RPE regression script, RPE annotations provide the continuous regression target.
Analysis workflows
Exercise-state classification
Read one multi-sheet Excel workbook per participant.
Align channel timestamps, including optional correction of whole-day date offsets.
Resample sensor streams to 1 Hz and optionally compensate for delays of selected sweat-biomarker signals.
Normalize sensor signals relative to an initial baseline and extract temporal/statistical features from windows or annotated substages.
Train and evaluate a four-class XGBoost classifier using LOSO-CV: each participant is held out for testing while the others form the training set.
Export per-participant metrics, predictions, confusion matrices, and figures.
Train a separate global four-class XGBoost model on pooled features to obtain SHAP interpretations.
Interpretation note: The global SHAP model is not an individual LOSO fold's held-out model. The classification script also includes optional offline smoothing, sequence decoding, and protocol-based priors; these affect how classification results must be interpreted.
RPE regression
Read and align participant workbooks and prepare synchronized physiological/biochemical signals.
Extract RPE-labeled samples and time-window features, primarily from Warmup and Exercise (the default target stages).
Train an XGBoost regressor and evaluate it using LOSO-CV by default.
Export predicted-versus-observed RPE values, per-fold and combined regression metrics, and derived fatigue-level evaluations.
Optionally train an all-subject model for global SHAP interpretation using `--do_global_shap`.
The RPE workflow is a regression analysis on the Borg RPE scale; fatigue-level plots are derived from predicted RPE values and do not replace the underlying continuous-regression evaluation.
System requirements
Operating system and Python
The following environment was identified from the working Conda installation and used to run both scripts successfully:
Operating system tested: Microsoft Windows 11 Pro, version 25H2 (OS Build 26200.9974)
Python: 3.9.25
Environment manager: Conda (Anaconda)
Tested Conda environment name: `XGBoost3.9`
These values describe the tested environment, not an assertion that every other operating system or package version has been tested.
Software dependencies
The following package versions were recorded using `conda list` in the `XGBoost3.9` environment:
Package	Tested version
NumPy	`2.0.2`
pandas	`2.3.3`
Matplotlib	`3.9.4`
scikit-learn	`1.6.1`
XGBoost	`2.1.4`
SHAP	`0.49.1`
openpyxl	`3.1.5`
SciPy	`1.13.1`
Numba	`0.60.0`
joblib	`1.5.3`
The additional dependencies SciPy, Numba, and joblib were also installed in the working environment. Other transitive dependencies are managed by the package installer.
Installation
If you already have the working `XGBoost3.9` environment, activate it directly:
```bash
conda activate XGBoost3.9
```
To create a new environment with the corresponding Python version, use Anaconda Prompt:
```bash
conda create -n XGBoost3.9 python=3.9.25 -y
conda activate XGBoost3.9
```
Install the recorded top-level package versions (single command):
```bash
python -m pip install numpy==2.0.2 pandas==2.3.3 matplotlib==3.9.4 scikit-learn==1.6.1 xgboost==2.1.4 shap==0.49.1 openpyxl==3.1.5 scipy==1.13.1 numba==0.60.0 joblib==1.5.3
```
The commands above recreate the named package versions, but do not guarantee identical builds or numerical results on every computer. A complete environment export can be prepared from the original tested environment if needed:
```bash
conda activate XGBoost3.9
conda env export --no-builds > environment.yml
```
Inspect the exported file and remove any machine-specific `prefix:` entry before committing it to the repository. Do not claim an `environment.yml` file is available in the repository unless it has actually been uploaded.
Typical installation time
Creating the Conda environment and installing the required Python packages took
approximately 25 minutes on the authors' tested Windows 11 computer.
This is an observed, approximate setup time; it may vary with computer
performance, internet connection, and package availability.
Tested environment and execution verification
Both `Exercise-State Classification.py` and `RPE Prediction.py` were successfully executed using the tested Windows/Conda environment above.
The source code uses XGBoost's histogram-based tree method (`tree_method='hist'`) without enabling `device='cuda'`. The analyses use CPU-based computation.
Successful execution confirms that the scripts run in the stated environment. Exact reproduction of manuscript performance metrics also depends on the input data, preprocessing, model parameters, random seeds, and evaluation settings.
Hardware requirements
No non-standard hardware is required to execute the offline analysis scripts. The XGBoost classification and regression workflows use the CPU; a dedicated GPU or CUDA installation is not required. Sufficient system memory and free disk space are still needed for the chosen dataset and any full-sample SHAP calculations.
The approximate measured setup and execution times are documented in the Typical installation time and Expected demo runtime sections below.
Input data format
Place participant Excel workbooks (`.xlsx`) in the folder `out_1Hz-1`, beside the Python files. The classification script reads this folder relative to its own location; the RPE script reads it relative to the current working directory unless `--excel_dir` is specified.
A suggested project layout is:
```text
XGBoost-Classification-of-Exercise-States/
├── Exercise-State Classification.py
├── RPE Prediction.py
├── README.md
├── LICENSE
├── out_1Hz-1/
│   ├── Subject_01.xlsx
│   ├── Subject_02.xlsx
│   └── Subject_03.xlsx
├── Exercise_State_Results/       # Generated by classification
└── RPE_Prediction_Results/       # Generated by RPE prediction
```
`LICENSE` should contain the finalized MIT license text with the correct copyright holder(s). The data and output folders are shown for illustration; they need not be committed to GitHub. The example filenames are not actual study data.
Each Excel workbook represents one participant. The workbook filename (without `.xlsx`) is used as the participant ID.
Each worksheet holds a sensor stream; its worksheet name normally matches the signal column. The following channel names are used by the analysis and/or its SHAP visualization:
Worksheet / signal	Description
`HR`	Heart rate
`Temp`	Skin temperature
`GRS` or `GSR`	Galvanic skin response; `GRS` is used in some code defaults
`SR`	Sweat rate
`TEC`	Sweat-related electrical channel; displayed as SIC in SHAP labels
`LA`	Lactate
`GLU`	Glucose
`AA`	Ascorbic acid
`UA`	Uric acid
`PH`	pH
Each worksheet must include a timestamp column named exactly `时间`, plus a numeric sensor-value column (preferably named after the worksheet). At least one worksheet must also contain exercise-stage annotations. The script looks for annotation columns such as `状态` and `阶段`; it preferentially uses annotations from the `HR` worksheet when available.
For example, an `HR` worksheet could look like:
时间	HR	状态
2026-01-01 08:00:00	72	静息 RPE=6
2026-01-01 08:05:00	94	热身 RPE=9
2026-01-01 08:20:00	151	递增 RPE=15
2026-01-01 08:45:00	103	恢复 RPE=10
The example values above are illustrative, not study data. Actual recordings should contain the available time-series samples, rather than just these four example rows.
The current stage parser recognizes these Chinese keywords:
Annotation contains	Classification label
`静息`	Rest
`热身`	Warmup
`递增`	Exercise
`恢复`	Recovery
Other labels are classified as `Unknown` and excluded from feature construction. An `RPE=...` annotation is optional for four-state classification but is required for RPE-labeled samples in the regression workflow. The RPE analysis additionally requires at least two valid participants for LOSO-CV.
Data sharing: Check research consent, ethics approval, and institutional requirements before uploading participant recordings to a public repository. The code can be shared independently of private participant data.
Quick start: Anaconda Prompt or Spyder
Place the two `.py` files and the `out_1Hz-1` folder in the same project directory. In Anaconda Prompt, activate the tested environment, then change to the repository directory:
```bat
conda activate XGBoost3.9
cd /d "C:\path\to\XGBoost-Classification-of-Exercise-States"
```
Replace the example path with the actual folder on your computer.
Run exercise-state classification
```bat
python "Exercise-State Classification.py"
```
The script's `run_all_in_one()` function runs LOSO classification and global SHAP generation, then redraws the saved SHAP figures. Its default paths are:
Input: `<script directory>/out_1Hz-1/*.xlsx`
Output: `<script directory>/Exercise_State_Results/`
Redrawn SHAP figures: `<script directory>/Exercise_State_Results/shap_outputs/Figure3_style/`
Important: The classification entry point sets `sys.argv` internally. Appending flags to the command above will not change its normal all-in-one run; modify the argument lists inside `run_all_in_one()` to change its settings.
Run RPE prediction
From the repository directory, run:
```bat
python "RPE Prediction.py"
```
For regression plus optional global SHAP interpretation:
```bat
python "RPE Prediction.py" --do_global_shap
```
The RPE script uses its normal command-line arguments:
Input (default): `./out_1Hz-1/*.xlsx`
Output (default): `./RPE_Prediction_Results/`
Cross-validation (default): `--cv loso`
Target stages (default): `--target_stages Warmup,Exercise`
Global SHAP: off by default; add `--do_global_shap` to enable it.
For an initial, limited technical check rather than a complete LOSO evaluation, the RPE script supports:
```bat
python "RPE Prediction.py" --max_folds 1
```
A one-fold test must not be reported as a full LOSO result.
Using Spyder
Open each script in Spyder with the Python interpreter set to the `XGBoost3.9` environment. For `Exercise-State Classification.py`, pressing F5 runs the built-in all-in-one workflow. For `RPE Prediction.py`, set Spyder's working directory to the repository directory so that the default `./out_1Hz-1` path is resolved correctly; configure command-line options separately if global SHAP is required.
Expected demo runtime
Running the analysis took approximately 15 minutes in the authors' local
test using the Windows 11 / Python 3.9.25 environment documented above.
This is an approximate reference time for the tested run, not a separately
measured runtime for each of the two scripts.
Actual execution time depends on the input dataset, number of participants,
computer performance, analysis settings, and whether optional global SHAP
computations are enabled.
About the figure names
The `Figure3_style` directory and `Figure3_SHAP_4state` filenames in the classification script are internal plotting names. They do not imply that these outputs correspond to Figure 3 of the SEAMS manuscript.
Adjustable parameters: exercise-state classification
How to change settings in Spyder
Near the end of the `.py` file, find `def run_all_in_one():`. In its first `_sys.argv = [...]` list (used for classification), add or change parameters as needed. For example:
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
This replaces only the first argument list, not the rest of the script. The second `_sys.argv = [...]` list controls the later SHAP redrawing stage, including export formats, resolution, and plot-axis settings.
Main analysis parameters
Parameter	Current behavior / default	What it controls
`--prediction_profile`	`protocol_calibrated`	Evaluation/post-processing profile. See below.
`--sample_level`	`window`	Extract features from time windows; `substage` selects annotated stage/substage rows.
`--window_sec`	120 s in the supplied all-in-one run	Window length. The parser specifies 60 s, but `protocol_calibrated` changes it to 120 s unless explicitly set.
`--stride_sec`	30 s	Distance between consecutive window starts; smaller strides give more overlapping windows.
`--sweat_delay`	300 s	Temporal compensation for channels listed in `--sweat_cols`. Set to `0` to disable.
`--sweat_cols`	`LA,PH,AA,UA,GLU`	Sweat biomarker channels to shift in time.
`--baseline_sec`	180 s	Initial interval used to establish each session's normalization baseline.
`--n_segments`	3	Number of subdivisions used for within-window/substage feature statistics.
`--min_window_coverage`	0.8	Minimum acceptable sample coverage of a candidate window.
`--min_label_fraction`	0.8	Required fraction of the majority exercise label in a window.
`--feature_mode`	`compact`	`compact` uses a selected feature subset; `all` includes more numerical sensor features.
`--trajectory_mode`	`none`	Optional across-window features: `none`, `causal`, or `offline`.
`--n_estimators`	250	Number of XGBoost trees in the LOSO classification model.
`--max_depth`	2	Maximum tree depth; larger values can increase model complexity.
`--n_model_seeds`	1	Number of models trained with different seeds for probability averaging.
`--seed`	42	Random seed for reproducibility.
`--xgb_n_jobs`	1	Number of XGBoost worker threads.
`--clip_z`	20	Clipping bound applied after baseline normalization.
`--shap_tec_clip_z`	30	TEC clipping setting for the separate SHAP feature dataset; not a LOSO-metric setting.
Sweat-delay direction: The implementation uses a negative shift (`shift(-delay)`) of selected sweat channels. A value of `300` aligns measurements from a later time point with an earlier reference time, reflecting an assumed sweat-response delay. This also removes trailing rows without aligned sweat values. Choose the delay based on the experimental protocol, not simply to increase accuracy.
Prediction profiles
Setting	Behavior
`paper_faithful`	Independent-window model predictions, without the automatic smoothing and sequence decoding enabled by the other profiles. This is the code's profile name, not a guarantee of reproducing every manuscript setting.
`stable`	Adds probability smoothing and correction of isolated short-lived label flips.
`high_accuracy`	Adds protocol-order sequence decoding to the stability operations.
`protocol_calibrated`	Adds a soft phase-duration prior derived from the LOSO training participants and uses 120-second windows unless overridden. This is the current script default.
Temporal post-processing and protocol-based priors can change classification performance. In particular, the `protocol_calibrated` setting assumes a known sequence of exercise stages and uses sequence-level information. Results from this offline setting should not be presented as equivalent to unconstrained real-time independent-window predictions.
For a comparison focused on independent windows, choose `--prediction_profile paper_faithful` and explicitly set the desired `--window_sec` and `--stride_sec`. Keep settings identical when comparing experimental conditions.
SHAP and figure parameters
Parameter	Default in the all-in-one run	Meaning
`--shap_explain_n`	0	Number of rows to explain with global SHAP; `0` means all available rows in this implementation.
`--shap_decision_n`	0	Rows used for class-specific decision plots; `0` means all rows.
`--shap_bg_n`	500	Maximum stratified background samples for the global SHAP calculation.
`--shap_model_estimators`	400	Number of trees in the separately trained global SHAP model.
`--shap_model_feature_mode`	`all`	Feature selection for the global interpretation model.
For a faster exploratory run, consider setting `--shap_explain_n` and `--shap_decision_n` to positive values (for example, `300` and `100`). The SHAP redraw stage also has its own options in the second argument list, such as `--dpi`, `--formats`, axis limits, and tick spacing. Changing these settings affects plotting, not the underlying LOSO predictions.
Adjustable parameters: RPE prediction
Unlike the classification script's all-in-one entry point, `RPE Prediction.py` accepts regular command-line options. The following defaults are defined in its current argument parser:
Parameter	Default	Purpose
`--cv`	`loso`	Leave-one-subject-out evaluation; `holdout` is also supported.
`--target_stages`	`Warmup,Exercise`	Exercise stages used for continuous RPE modeling.
`--sweat_delay`	`120` seconds	Optional alignment shift for selected sweat channels.
`--short_windows`	`30,60,120,180,300`	History-window lengths in seconds for feature extraction.
`--sample_step`	`5` seconds	Step between sampled feature positions.
`--history_sec`	`900` seconds	Maximum recent history used in feature construction.

`--n_estimators`	`700`	Number of regression-model trees.
`--learning_rate`	`0.025`	Learning rate of the regressor.
`--max_depth`	`4`	Maximum tree depth.
`--seed`	`42`	Random seed.
`--rpe_lo`, `--rpe_hi`	`6`, `20`	Lower and upper bounds for predicted/observed RPE.
`--plot_band`	`1.0`	Absolute-error tolerance shown in evaluation plots.
`--do_global_shap`	disabled	Train an additional all-subject model for global interpretation.
For example, to use an explicit input directory and an output folder of your choice:
```bat
python "RPE Prediction.py" --excel_dir "./out_1Hz-1" --out "./RPE_Prediction_Results" --cv loso --do_global_shap
```
The code generates regression predictions and can derive fatigue-level categories from RPE thresholds for secondary reporting. The global interpretation model, when requested, is trained separately from the held-out LOSO models.
Main output files
After a successful classification run, `Exercise_State_Results/` contains results similar to:
```text
Exercise_State_Results/
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
`subject_macro_f1_summary.csv`: per-held-out-participant classification metrics.
`all_loso_predictions.csv`: true/predicted exercise-state labels and model probabilities for LOSO samples.
`four_class_confusion_matrix.*`: aggregated four-class classification results.
`global_4class_shap_raw.npz`: saved SHAP data used for the final SHAP plots.
`Figure3_SHAP_4state.*`: composite classification-SHAP visualization.

RPE-regression outputs
The RPE script saves outputs under `RPE_Prediction_Results/` by default. Example structure:
```text
RPE_Prediction_Results/
├── summary_rpe_reg_loso.csv
├── heldout_predictions_rpe_substage_all.csv
├── heldout_metrics_rpe_substage_all.json
├── heldout_pred_vs_true_rpe_substage_all.png
├── cache_subjects/
├── loso_01_val_<subject>/
│   ├── model_rpe_reg.json
│   ├── substage_predictions_rpe_reg_val_ALL.csv
│   └── pred_vs_true_rpe_val_ALL.png
└── global_interpretation/          # Only when global SHAP is enabled
    ├── model_rpe_reg_all_subjects.json
    └── shap_rpe_reg/
```
The exact per-fold directory names depend on participant identifiers. `summary_rpe_reg_loso.csv` summarizes held-out RPE performance; combined held-out predictions and metrics are saved in the correspondingly named CSV and JSON files. Global SHAP output is conditional on `--do_global_shap`.
Troubleshooting
`No Excel files found` — Confirm that `out_1Hz-1` is beside the `.py` file and contains `.xlsx` workbooks.
`missing 时间 column` — Verify the exact Chinese timestamp-column name in every worksheet.
`No valid substage feature rows were built` — Check the Chinese stage keywords, timestamps, recording duration, and window settings.
Too few final samples — Check `--sweat_delay`, time alignment, and window coverage. A large delay can shorten the usable recording.
Missing SHAP figures — Confirm that `shap_outputs/global_4class_shap_raw.npz` was generated. The restyling step requires saved SHAP data.
Slow execution or large figures — Reduce the number of explained/decision-plot samples, reduce plot DPI for preview runs, or use fewer model trees during testing.
RPE regression: `Need at least 2 valid subjects` — Confirm the input folder contains data from at least two participants, that stage annotations include RPE values, and that the selected target stages have valid samples.
RPE regression: expected folder not found — Run from the repository directory or pass `--excel_dir` and `--out` explicitly. Unlike the classification all-in-one script, RPE paths are relative to the current working directory by default.
No RPE global SHAP figures — The `--do_global_shap` option must be enabled; global SHAP is not part of the default regression run.
Reproducibility and interpretation
How to reproduce the machine-learning analyses
The following steps describe how to rerun the manuscript's
exercise-state classification and RPE prediction analyses. These scripts
do not reproduce the paper's independent materials-characterization or
sensor-fabrication experiments.
Prepare the study's participant `.xlsx` files, using the specified
worksheet structure and stage/RPE annotations. Place the files in
`out_1Hz-1/` (see Input data format).
Install and activate the tested Conda environment described above.
Run the exercise-state classification analysis:
```bat
   python "Exercise-State Classification.py"
   ```
Run the RPE prediction analysis:
```bat
   python "RPE Prediction.py"
   ```
Add `--do_global_shap` when the optional global SHAP analysis is required:
```bat
   python "RPE Prediction.py" --do_global_shap
   ```
Examine the classification predictions, confusion matrices and metrics
under `Exercise_State_Results/`, and the RPE predictions and regression
metrics under `RPE_Prediction_Results/`. Compare them with the corresponding
machine-learning findings in the manuscript and Supplementary Information.
For a quantitative comparison, use the same participant dataset,
model parameters, random seeds, preprocessing, cross-validation scheme,
and any optional post-processing settings as the reported analysis.
Data requirement: The source-code repository alone does not supply the
study's participant input data. To enable independent reproduction, make the
corresponding data accessible to authorized reviewers through a suitable
file or download link. Only share participant data in accordance with
consent, ethics approval, and applicable data-sharing requirements.
The manuscript reports performance from its specified experimental dataset and analysis procedures; those values should not be assumed for new participants, altered sensor inputs, or different processing settings. Keep the input data, software versions, sampling settings, and post-processing profile consistent when comparing results.
Because the source code includes optional temporal smoothing, sequence decoding, and protocol-based priors, state clearly which configuration produced each reported metric. Both delivered programs are offline research-analysis scripts, not ready-to-deploy real-time inference applications. Successful local execution does not by itself establish reproduction of every numerical result in the manuscript.
Code license
The Python source code is made available under the MIT License, provided the repository includes a finalized `LICENSE` file with the correct copyright holder(s). This software license does not automatically license any participant data, manuscript figures, or other separately owned research materials. If datasets are shared, their availability and license should be stated separately.
Related manuscript
Shi, C., Gao, Y., Cao, J., Jia, H., Xue, R., Zhou, J., Tao, Y., Ruan, Q., Song, M., and Ren, Y. A general strain-decoupling architecture with robust interfacial coupling for stretchable bioelectronics. Manuscript; bibliographic details/DOI can be added when available.
