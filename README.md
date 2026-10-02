# Network Intrusion Detection Research Project

This project studies whether machine learning methods can detect an attack family that was not represented during training. The planned experiments use the improved CIC-IDS2017 dataset and compare supervised, anomaly-detection, and semi-supervised approaches using a leave-one-attack-family-out evaluation.

## Folder structure

```text
.
├── artifacts/
│   └── models/       Saved model checkpoints, if needed
├── logs/             Run and debugging logs
├── notes/            Literature notes, one file per reviewed paper
├── progress/         Weekly progress records and updates
├── reports/          Reports prepared for sharing
├── results/
│   ├── figures/      Evaluation plots
│   ├── raw/           Per-run result files
│   ├── stats/         Statistical analysis outputs
│   ├── tables/        Summary tables
│   └── tuning/        Validation-based tuning records
├── src/
│   ├── data/          Dataset download, cleaning, and split code
│   ├── eval/          Metrics, thresholds, resource measures, and statistics
│   └── models/        Detector implementations
├── tests/             Synthetic and unit tests
└── thesis/
    └── chapters/      Thesis chapter source files
```

Each visible folder currently contains an empty `.gitkeep` marker. This initial GitHub snapshot records the planned layout only; it does not include the local Python source, requirements file, research notes, progress documents, or thesis drafts.

## Data and experiment handling

Dataset files stay on the local machine and are not part of this repository. Keep raw data and generated results out of Git unless a later project decision explicitly calls for sharing a specific derived artifact.

The planned experiment holds one attack family out of training, uses validation data for tuning and threshold selection, and evaluates on an untouched test set. Preprocessing must be fitted on training data only.

## What belongs in each area

- `src/` is for dataset preparation, model implementations, experiment runners, and evaluation code.
- `tests/` is for synthetic-data and unit tests.
- `notes/` is for notes on papers that have actually been reviewed.
- `progress/` is for factual weekly progress updates.
- `results/` is for run outputs, tuning records, tables, figures, and statistical summaries.
- `thesis/` is for the thesis source and chapter drafts.
- `artifacts/models/` is reserved for saved model checkpoints.
