# Project 3 Paper Code Package

This folder is a clean copy of the code that supports the methodology of the paper `project31_user_current_ai_clarified.tex`.

The goal of this package is simple:

- keep together the scripts that turn the GAITEX raw IMU recordings into interpretable features,
- run the main Project 3.1 analyses used in the paper,
- keep the Project 3.2 window-level extension in the same place,
- make it easy to upload the paper code to GitHub later.

This package does **not** include the full dataset and it does **not** include every exploratory script from the larger working directory. It only includes the code that is directly relevant to the paper methodology and the main reported analyses.

## Folder Structure

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
    analysis_sensor_configuration_frontier.py
    analysis_semantic_feedback.py
    analysis_project32_closed_loop.py
    analysis_segment_ablation.py
    analysis_feature_relevance.py
    analysis_segment_heatmap.py
    analysis_delta_signflip_permutation.py
    analysis_all_vs_minimal_table.py
```

## What Each File Does

### Core data loading and feature extraction

- `src/gaitex/io.py`
  Reads the GAITEX file structure, finds the IMU CSV files and timestamp CSV files, and loads them into Python.

- `src/gaitex/segment.py`
  Cuts each recording into individual repetitions using the time ranges from the timestamps files.

- `src/gaitex/features.py`
  Contains the quaternion handling and the interpretable rotational feature calculations used in the paper:
  mean angular speed, RMS angular speed, peak angular speed, RMS angular acceleration, and rotational range.

- `src/gaitex/build_features.py`
  Runs the full repetition-level feature extraction pipeline and writes `features.csv`.

- `src/gaitex/qc_plots.py`
  Produces simple quality-control plots from the extracted features and repetition durations.

### Project 3.1 paper analyses

- `src/gaitex/analysis_sensor_configuration_frontier.py`
  Tests many sensor combinations and finds the best-performing subsets for correct-versus-incorrect classification.

- `src/gaitex/analysis_semantic_feedback.py`
  Builds the explainability layer by comparing each error variant against the correct movement and turning the largest changes into readable summaries.

- `src/gaitex/run_project31_pipeline.py`
  Convenience entry point that runs the main Project 3.1 analysis sequence.

### Baseline and supporting paper figures

- `src/gaitex/analysis_segment_ablation.py`
  Creates the segment-ablation benchmark used to show that some body regions are more informative than others.

- `src/gaitex/analysis_feature_relevance.py`
  Estimates which segment-feature combinations matter most for the classifier and produces the feature-relevance plots.

- `src/gaitex/analysis_segment_heatmap.py`
  Creates the segment-level effect-size heatmap used in the explainability results.

- `src/gaitex/analysis_delta_signflip_permutation.py`
  Compares the full representation against the minimal representation with paired subject-level testing.

- `src/gaitex/analysis_all_vs_minimal_table.py`
  Turns the full-versus-minimal comparison into a compact summary table and a LaTeX table.

### Project 3.2 closed-loop extension

- `src/gaitex/build_window_features.py`
  Creates short-window features from each repetition for the proof-of-concept real-time analysis.

- `src/gaitex/analysis_project32_closed_loop.py`
  Runs the window-level classification, early detection analysis, explanation tracking, and simulated feedback trigger logic.

- `src/gaitex/run_project32_pipeline.py`
  Convenience entry point that runs the main Project 3.2 sequence.

## What This Code Covers in the Paper

This package covers the code behind the following methodological parts of the manuscript:

- loading GAITEX IMU and timestamp files,
- cutting recordings into repetitions,
- computing interpretable quaternion-based rotational features,
- subject-level correct-versus-incorrect classification,
- sensor reduction and sensor frontier analysis,
- readable explainability summaries,
- supporting baseline analyses and figures,
- the window-level Project 3.2 proof-of-concept extension.

## What Is Not Included

This package does not include:

- the raw dataset itself,
- the full working directory,
- old exploratory scripts that were not needed for the paper methodology,
- submission files from journal templates,
- generated outputs such as `output/features/*.csv` or `output/figures/*.png`.

Those can be regenerated after the code is run on a machine that also has the GAITEX data.

## Dependencies

The package uses the dependencies listed in `requirements.txt`:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`
- `pyyaml`

## Expected Data Layout

The scripts expect a GAITEX-style data folder such as:

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

The loader is somewhat flexible about exact filenames, but it expects the same general structure:

- one subject folder per participant,
- one task folder such as `rd` or `rgs`,
- one segment-registered IMU CSV,
- one timestamps CSV with repetition start and end times.

## Simple Run Order

Run the commands from the root of the repository.

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build repetition-level features

```bash
PYTHONPATH=src python -m gaitex.build_features --data_root data --out_dir output/features
```

This creates:

- `output/features/features.csv`
- quality-control figures in `output/figures/`

### 3. Run the main Project 3.1 paper pipeline

```bash
PYTHONPATH=src python -m gaitex.run_project31_pipeline
```

This runs:

- the sensor configuration frontier analysis,
- the semantic feedback and explainability analysis.

### 4. Run the supporting baseline analyses and figures

```bash
PYTHONPATH=src python -m gaitex.analysis_segment_ablation
PYTHONPATH=src python -m gaitex.analysis_feature_relevance
PYTHONPATH=src python -m gaitex.analysis_segment_heatmap
PYTHONPATH=src python -m gaitex.analysis_delta_signflip_permutation
PYTHONPATH=src python -m gaitex.analysis_all_vs_minimal_table
```

### 5. Run the Project 3.2 window-level extension

```bash
PYTHONPATH=src python -m gaitex.run_project32_pipeline
```

This runs:

- window-level feature extraction,
- window-level classification,
- early-detection summaries,
- explanation tracking,
- simulated feedback trigger analysis.

## Main Output Files

After running the code, the main output files are expected to appear in:

- `output/features/`
- `output/figures/`

Important examples include:

- `features.csv`
- `semantic_fingerprint_effects.csv`
- `semantic_feedback_variants.csv`
- `sensor_frontier_*`
- `project32_window_classification_summary.csv`
- `project32_early_detection_summary.csv`
- `project32_feedback_summary_tuned.csv`

## Suggested Paper Sentence for the Methods Section

When you decide the GitHub repository name, you can adapt this sentence directly in the paper:

> The code used to generate the features, run the movement-quality analyses, evaluate reduced sensor configurations, and produce the proof-of-concept window-level feedback extension is available in the GitHub repository `<REPO_NAME>`.

If you want a slightly longer version:

> All code used for the data loading, repetition segmentation, interpretable feature extraction, subject-level classification, sensor-subset analysis, explainability layer, and proof-of-concept window-level feedback analyses is available in the GitHub repository `<REPO_NAME>`.

## Practical Note for Copying This Folder to a Mac

From your Mac, you can copy this folder with a command like:

```bash
scp -r ilab@100.98.109.103:/home/ilab/project3/Project_3.0/project3repo /local/path/
```

Then you can upload that copied folder to your GitHub repository.
