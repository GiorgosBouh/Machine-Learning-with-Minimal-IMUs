# Project 3.1 Reproducibility Code

This repository contains the code package that supports the paper Interpretable Machine Learning Distinguishes Correct from Incorrect Rehabilitation Movement with Fewer Wearable Sensors, including subject-level movement- quality benchmarks, sensor-reduction analyses, interpretable feedback summaries, and the Project 3.2 window-level proof-of-concept extension.

The package is intended for public release alongside the article so that readers can inspect the analysis logic and reproduce the reported outputs on a machine that also has access to the GAITEX data.

## Scope

This repository covers the code used for:

- loading GAITEX IMU recordings and timestamp files,
- segmenting recordings into repetitions,
- computing interpretable quaternion-based rotational features,
- building repetition-level and window-level feature tables,
- subject-level correct-versus-incorrect benchmarking,
- full-versus-minimal baseline comparisons,
- retained multiclass LOSO benchmarking,
- sensor configuration frontier analysis,
- shared compact-subset and model-family sensitivity checks,
- readable explainability summaries derived from explicit segment-feature deviations,
- segment-ablation and feature-relevance support analyses,
- Project 3.2 window-level early-detection and feedback simulations,
- reviewer-driven feedback-rule sweeps used in the revised closed-loop analysis.

## What Is Not Included

This repository does not include:

- the GAITEX raw dataset,
- the full private working directory,
- journal submission templates,
- unrelated exploratory scripts,
- pre-generated figures or CSV outputs that can be regenerated from the code.

To fully reproduce the paper outputs, the user must have access to the GAITEX data in the expected folder structure.

## Repository Layout

```text
project3repo/
  README.md
  requirements.txt
  src/gaitex/
    __init__.py
    io.py
    segment.py
    features.py
    qc_plots.py
    build_features.py
    build_window_features.py
    run_project31_pipeline.py
    run_project32_pipeline.py
    analysis_task_complexity_test.py
    analysis_all_vs_minimal_table.py
    analysis_classifier_report.py
    analysis_segment_ablation.py
    analysis_feature_relevance.py
    analysis_sensor_configuration_frontier.py
    analysis_reviewer2_model_sensitivity.py
    analysis_semantic_feedback.py
    analysis_segment_heatmap.py
    analysis_delta_signflip_permutation.py
    analysis_project32_closed_loop.py
    analysis_reviewer2_feedback_rule_sweep.py
```

## Main Scripts

Core preprocessing:

- `src/gaitex/io.py`: data discovery and CSV loading
- `src/gaitex/segment.py`: repetition segmentation from timestamp ranges
- `src/gaitex/features.py`: quaternion handling and interpretable rotational feature extraction
- `src/gaitex/build_features.py`: repetition-level feature extraction
- `src/gaitex/build_window_features.py`: window-level feature extraction

Project 3.1 subject-level analyses:

- `src/gaitex/analysis_task_complexity_test.py`: full-vs-minimal baseline comparison
- `src/gaitex/analysis_all_vs_minimal_table.py`: summary CSV and LaTeX table for the baseline comparison
- `src/gaitex/analysis_classifier_report.py`: multiclass LOSO benchmark and per-class reports
- `src/gaitex/analysis_segment_ablation.py`: single-segment ablation benchmark
- `src/gaitex/analysis_feature_relevance.py`: feature-relevance maps
- `src/gaitex/analysis_sensor_configuration_frontier.py`: unconstrained and lower-body sensor frontier search
- `src/gaitex/analysis_reviewer2_model_sensitivity.py`: model-family comparison on full and shared compact subsets
- `src/gaitex/analysis_semantic_feedback.py`: readable explainability summaries from paired feature deviations
- `src/gaitex/run_project31_pipeline.py`: convenience entry point for the main revised Project 3.1 pipeline

Project 3.2 window-level analyses:

- `src/gaitex/analysis_project32_closed_loop.py`: window-level classification, early detection, explanation tracking, and feedback simulation
- `src/gaitex/analysis_reviewer2_feedback_rule_sweep.py`: grid search over feedback trigger rules
- `src/gaitex/run_project32_pipeline.py`: convenience entry point for the window-level revised pipeline

## Expected Data Layout

The scripts expect a GAITEX-like directory structure under `data/`, for example:

```text
data/
  subject_1/
    rd/
      xsens_imu_data_segment_registered_subject_1_rd.csv
      timestamps_subject_1_rd.csv
    rgs/
      xsens_imu_data_segment_registered_subject_1_rgs.csv
      timestamps_subject_1_rgs.csv
```

The loader is somewhat flexible about exact filenames, but expects:

- one folder per subject,
- one folder per task such as `rd` or `rgs`,
- one segment-registered IMU CSV per task,
- one timestamps CSV per task.

## Environment Setup

Run from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducibility Run Order

### 1. Build repetition-level features

```bash
PYTHONPATH=src python -m gaitex.build_features --data_root data --out_dir output/features
```

Main outputs:

- `output/features/features.csv`
- QC figures under `output/figures/`

### 2. Run the revised Project 3.1 subject-level pipeline

```bash
PYTHONPATH=src python -m gaitex.run_project31_pipeline
```

This pipeline runs:

- full-vs-minimal binary benchmarking,
- baseline summary generation,
- multiclass LOSO benchmarking,
- segment ablation,
- feature relevance,
- sensor frontier analysis,
- model-family sensitivity checks,
- semantic feedback and explainability outputs.

### 3. Run the revised Project 3.2 window-level pipeline

```bash
PYTHONPATH=src python -m gaitex.run_project32_pipeline
```

This pipeline runs:

- window-level feature extraction,
- window-level classification,
- early-detection summaries,
- window-level explanation summaries,
- feedback simulations,
- feedback-rule grid search for the revised closed-loop analysis.

## Important Output Files

Representative outputs include:

- `output/features/features.csv`
- `output/features/task_complexity_results.csv`
- `output/features/task_complexity_folds_all_vs_minimal.csv`
- `output/features/all_vs_minimal_summary.csv`
- `output/features/classifier_loso_folds.csv`
- `output/features/semantic_fingerprint_effects.csv`
- `output/features/semantic_feedback_variants.csv`
- `output/features/sensor_frontier_all_subsets.csv`
- `output/features/sensor_frontier_best_by_count.csv`
- `output/features/reviewer2_model_subset_checks.csv`
- `output/features/project32_window_classification_summary.csv`
- `output/features/project32_early_detection_summary.csv`
- `output/features/project32_feedback_summary_tuned.csv`
- `output/features/reviewer2_feedback_grid_selected_summary.csv`

Figures are written under `output/figures/`.

## Dependencies

Dependencies are listed in `requirements.txt` and include:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`
- `pyyaml`


```
