# Yoda Eval ✨

## Overview 🧾

A Kubeflow Pipelines component for yoda eval.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | `None` |  |
| `output_metrics` | `dsl.Output[dsl.Metrics]` | `None` |  |
| `output_results` | `dsl.Output[dsl.Artifact]` | `None` |  |
| `output_prompts` | `dsl.Output[dsl.Artifact]` | `None` |  |
| `lora_adapter` | `dsl.Input[dsl.Model]` | `None` |  |
| `batch_size` | `int` | `1` |  |
| `limit` | `int` | `None` |  |
| `max_model_len` | `int` | `4096` |  |
| `gpu_memory_utilization` | `float` | `0.8` |  |
| `dtype` | `str` | `auto` |  |
| `add_bos_token` | `bool` | `True` |  |
| `include_classification_tasks` | `bool` | `True` |  |
| `include_summarization_tasks` | `bool` | `True` |  |
| `custom_translation_dataset` | `dsl.Input[dsl.Dataset]` | `None` |  |
| `log_prompts` | `bool` | `True` |  |
| `verbosity` | `str` | `INFO` |  |
| `max_batch_size` | `int` | `None` |  |



## Metadata 🗂️

- **Name**: yoda_eval
- **Stability**: alpha
- **Dependencies**: 
  - Kubeflow:
    - Name: Pipelines, Version: >=2.5
  - External Services:
    - Name: CUDA GPU, Version: compatible
    - Name: HuggingFace Transformers, Version: latest
    - Name: PyTorch, Version: latest
    - Name: LM Evaluation Harness, Version: latest
    - Name: VLLM, Version: latest
    - Name: Unitxt, Version: latest
- **Tags**: 
  - evaluation
  - llm_evaluation
  - model_evaluation
  - yoda_speak
  - classification
  - summarization
  - translation
  - benchmarking
  - gpu
- **Last Verified**: 2025-12-15 00:00:00+00:00

## Additional Resources 📚

- **Issue Tracker**: [https://github.com/kubeflow/pipelines-components/issues](https://github.com/kubeflow/pipelines-components/issues)
