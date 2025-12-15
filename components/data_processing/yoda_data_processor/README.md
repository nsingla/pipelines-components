# Yoda Data Processor ✨

## Overview 🧾

Prepare the training and evaluation datasets by downloading and preprocessing.

Downloads the yoda_sentences dataset from HuggingFace, renames columns to match
the expected format for training (prompt/completion), splits into train/eval sets,
and saves them as output artifacts.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `yoda_input_dataset` | `str` | `None` | Dataset to download from HuggingFace |
| `yoda_train_dataset` | `dsl.Output[dsl.Dataset]` | `None` | Output dataset for training. |
| `yoda_eval_dataset` | `dsl.Output[dsl.Dataset]` | `None` | Output dataset for evaluation. |
| `operation_map` | `dict[str, Any]` | `{'rename_column': {'sentence': 'prompt'}}` | Specify list of operations you want to perform on the data set before splitting it e.g. {"rename_column": {"sentence":"prompt"}, "remove_columns": "translation"} |
| `train_split_ratio` | `float` | `0.8` | Ratio of data to use for training (0.0-1.0).
Defaults to 0.8 (80% train, 20% eval). |



## Metadata 🗂️

- **Name**: yoda_data_processor
- **Stability**: alpha
- **Dependencies**: 
  - Kubeflow:
    - Name: Pipelines, Version: >=2.5
  - External Services:
    - Name: HuggingFace Datasets, Version: latest
- **Tags**: 
  - data_processing
  - dataset_preparation
  - text_processing
  - yoda_speak
  - translation
- **Last Verified**: 2025-12-15 00:00:00+00:00

## Additional Resources 📚

- **Issue Tracker**: [https://github.com/kubeflow/pipelines-components/issues](https://github.com/kubeflow/pipelines-components/issues)
