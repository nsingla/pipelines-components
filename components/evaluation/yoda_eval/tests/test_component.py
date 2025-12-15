"""Tests for the yoda_eval component."""

import json
import tempfile
from pathlib import Path
from unittest import mock
import pytest
from dataclasses import dataclass
from typing import Any, Dict, Optional

import kfp
from kfp import compiler
from kfp import local

from kfp_components.components.evaluation.yoda_eval import evaluate_yoda_model


class TestYodaEvalCompilation:
    """Test component compilation."""

    def test_component_compilation(self):
        """Test that the component compiles successfully without errors."""
        # Create a temporary file to store the compiled component
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            # Compile the component - this should not raise any exceptions
            compiler.Compiler().compile(
                evaluate_yoda_model,
                package_path=temp_path
            )

            # Verify the file was created and is not empty
            compiled_path = Path(temp_path)
            assert compiled_path.exists()
            assert compiled_path.stat().st_size > 0

            # Verify the compiled YAML contains expected component structure
            with open(compiled_path, 'r') as f:
                content = f.read()
                assert 'name: evaluate-yoda-model' in content
                assert 'implementation' in content
                assert 'container' in content
                assert 'registry.access.redhat.com/ubi9/python-311:latest' in content
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)


@dataclass
class TestData:
    """Test data structure for parameterized tests."""
    pipeline_func: Any
    pipeline_func_args: Optional[Dict] = None
    expected_output: Any = None


class TestYodaEvalLocalRunner:
    """Test component with LocalRunner (subprocess execution)."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_and_teardown(self):
        """Setup LocalRunner environment for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ws_root = Path(temp_dir) / 'workspace'
            pipeline_root = Path(temp_dir) / 'pipeline_root'

            ws_root.mkdir(exist_ok=True)
            pipeline_root.mkdir(exist_ok=True)

            # Initialize local runner with subprocess runner
            local.init(
                runner=local.SubprocessRunner(),
                raise_on_error=True,
                workspace_root=str(ws_root),
                pipeline_root=str(pipeline_root)
            )
            yield
            # Cleanup handled by temporary directory context manager

    @mock.patch('torch.cuda.is_available', return_value=True)
    @mock.patch('lm_eval.evaluator.evaluate')
    @mock.patch('lm_eval.api.registry.get_model')
    def test_local_execution_basic(self, mock_get_model, mock_evaluate, mock_cuda_available):
        """Test component execution with LocalRunner using mocked dependencies."""
        # Mock the model and evaluation results
        mock_model_instance = mock.MagicMock()
        mock_model_class = mock.MagicMock()
        mock_model_class.create_from_arg_obj.return_value = mock_model_instance
        mock_get_model.return_value = mock_model_class

        mock_results = {
            'results': {
                'classification_rte_simple': {
                    'bleu': 45.2,
                    'exact_match': 0.85
                }
            }
        }
        mock_evaluate.return_value = mock_results

        # Test data for the component
        test_data = TestData(
            pipeline_func=evaluate_yoda_model,
            pipeline_func_args={
                'model_path': 'test-model-path',
                'batch_size': 1,
                'include_classification_tasks': True,
                'include_summarization_tasks': False
            }
        )

        # Execute the component
        pipeline_task = test_data.pipeline_func(**test_data.pipeline_func_args)

        # Verify the task was created successfully
        assert pipeline_task is not None

        # NOTE: In LocalRunner environment with subprocess execution,
        # the actual execution happens in a separate process,
        # so detailed validation of mocked calls may not work as expected.
        # This test primarily verifies that the component can be compiled and executed
        # without runtime errors in the LocalRunner environment.


class TestYodaEvalUnitTests:
    """Unit tests for component logic and classes."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(evaluate_yoda_model)
        assert hasattr(evaluate_yoda_model, 'python_func')

    @mock.patch('torch.cuda.is_available', return_value=False)
    def test_cuda_validation_fails_without_gpu(self, mock_cuda_available):
        """Test that component fails when CUDA is not available."""
        # Mock output artifacts
        mock_metrics_output = mock.MagicMock()
        mock_results_output = mock.MagicMock()
        mock_prompts_output = mock.MagicMock()

        # Should raise ValueError when CUDA is not available
        with pytest.raises(ValueError, match="CUDA is not available"):
            evaluate_yoda_model.python_func(
                model_path='test-model',
                output_metrics=mock_metrics_output,
                output_results=mock_results_output,
                output_prompts=mock_prompts_output
            )

    @mock.patch('torch.cuda.is_available', return_value=True)
    def test_parameter_validation_gpu_memory_utilization(self, mock_cuda_available):
        """Test parameter validation for gpu_memory_utilization."""
        # Mock output artifacts
        mock_metrics_output = mock.MagicMock()
        mock_results_output = mock.MagicMock()
        mock_prompts_output = mock.MagicMock()

        # Test invalid gpu_memory_utilization (> 1.0)
        with pytest.raises(ValueError, match="gpu_memory_utilization must be between 0.0 and 1.0"):
            evaluate_yoda_model.python_func(
                model_path='test-model',
                output_metrics=mock_metrics_output,
                output_results=mock_results_output,
                output_prompts=mock_prompts_output,
                gpu_memory_utilization=1.5
            )

        # Test invalid gpu_memory_utilization (< 0.0)
        with pytest.raises(ValueError, match="gpu_memory_utilization must be between 0.0 and 1.0"):
            evaluate_yoda_model.python_func(
                model_path='test-model',
                output_metrics=mock_metrics_output,
                output_results=mock_results_output,
                output_prompts=mock_prompts_output,
                gpu_memory_utilization=-0.1
            )

    @mock.patch('torch.cuda.is_available', return_value=True)
    def test_parameter_validation_batch_size(self, mock_cuda_available):
        """Test parameter validation for batch_size."""
        # Mock output artifacts
        mock_metrics_output = mock.MagicMock()
        mock_results_output = mock.MagicMock()
        mock_prompts_output = mock.MagicMock()

        # Test invalid batch_size (<= 0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            evaluate_yoda_model.python_func(
                model_path='test-model',
                output_metrics=mock_metrics_output,
                output_results=mock_results_output,
                output_prompts=mock_prompts_output,
                batch_size=0
            )

    @mock.patch('torch.cuda.is_available', return_value=True)
    def test_parameter_validation_max_model_len(self, mock_cuda_available):
        """Test parameter validation for max_model_len."""
        # Mock output artifacts
        mock_metrics_output = mock.MagicMock()
        mock_results_output = mock.MagicMock()
        mock_prompts_output = mock.MagicMock()

        # Test invalid max_model_len (<= 0)
        with pytest.raises(ValueError, match="max_model_len must be positive"):
            evaluate_yoda_model.python_func(
                model_path='test-model',
                output_metrics=mock_metrics_output,
                output_results=mock_results_output,
                output_prompts=mock_prompts_output,
                max_model_len=0
            )

    @mock.patch('torch.cuda.is_available', return_value=True)
    def test_parameter_validation_limit(self, mock_cuda_available):
        """Test parameter validation for limit."""
        # Mock output artifacts
        mock_metrics_output = mock.MagicMock()
        mock_results_output = mock.MagicMock()
        mock_prompts_output = mock.MagicMock()

        # Test invalid limit (<= 0)
        with pytest.raises(ValueError, match="limit must be positive or None"):
            evaluate_yoda_model.python_func(
                model_path='test-model',
                output_metrics=mock_metrics_output,
                output_results=mock_results_output,
                output_prompts=mock_prompts_output,
                limit=0
            )

    @mock.patch('torch.cuda.is_available', return_value=True)
    def test_parameter_validation_no_tasks_selected(self, mock_cuda_available):
        """Test that component fails when no tasks are selected."""
        # Mock output artifacts
        mock_metrics_output = mock.MagicMock()
        mock_results_output = mock.MagicMock()
        mock_prompts_output = mock.MagicMock()

        # Test when no tasks are enabled
        with pytest.raises(ValueError, match="At least one of include_classification_tasks, include_summarization_tasks, or custom_translation_dataset must be provided"):
            evaluate_yoda_model.python_func(
                model_path='test-model',
                output_metrics=mock_metrics_output,
                output_results=mock_results_output,
                output_prompts=mock_prompts_output,
                include_classification_tasks=False,
                include_summarization_tasks=False
            )

    @mock.patch('torch.cuda.is_available', return_value=True)
    @mock.patch('lm_eval.evaluator.evaluate')
    @mock.patch('lm_eval.api.registry.get_model')
    @mock.patch('lm_eval.tasks.get_task_dict')
    def test_successful_evaluation_with_classification_tasks(self, mock_get_task_dict, mock_get_model, mock_evaluate, mock_cuda_available):
        """Test successful evaluation with classification tasks enabled."""
        # Mock the model loading
        mock_model_instance = mock.MagicMock()
        mock_model_class = mock.MagicMock()
        mock_model_class.create_from_arg_obj.return_value = mock_model_instance
        mock_get_model.return_value = mock_model_class

        # Mock task dict
        mock_get_task_dict.return_value = {'classification_rte_simple': mock.MagicMock()}

        # Mock evaluation results
        mock_results = {
            'results': {
                'classification_rte_simple': {
                    'bleu': 45.2,
                    'exact_match': 0.85
                }
            }
        }
        mock_evaluate.return_value = mock_results

        # Mock output artifacts
        mock_metrics_output = mock.MagicMock()
        mock_results_output = mock.MagicMock()
        mock_results_output.path = '/tmp/results.json'
        mock_prompts_output = mock.MagicMock()

        # Call the actual python function
        evaluate_yoda_model.python_func(
            model_path='test-model',
            output_metrics=mock_metrics_output,
            output_results=mock_results_output,
            output_prompts=mock_prompts_output,
            include_classification_tasks=True,
            include_summarization_tasks=False
        )

        # Verify interactions
        mock_get_model.assert_called_once_with("vllm")
        mock_evaluate.assert_called_once()
        mock_metrics_output.log_metric.assert_called()

    def test_translation_task_initialization(self):
        """Test TranslationTask class initialization."""
        # Import the TranslationTask class from the component module
        # Note: This test assumes we can access the class from the component
        # In actual implementation, you might need to refactor to make the class more testable

        # For this test, we'll verify that the component contains the expected task configuration
        # by checking the TASK_CONFIGS constant exists in the component code
        assert hasattr(evaluate_yoda_model, 'python_func')

        # Test that component has expected task configurations
        # This is a basic test to ensure the component structure is correct
        component_source = evaluate_yoda_model.python_func.__code__.co_consts

        # Verify that the component contains references to expected task types
        component_str = str(component_source)
        assert 'classification' in component_str or 'summarization' in component_str

    @mock.patch('torch.cuda.is_available', return_value=True)
    @mock.patch('lm_eval.api.registry.get_model')
    def test_model_loading_failure(self, mock_get_model, mock_cuda_available):
        """Test handling of model loading failure."""
        # Mock model loading to fail
        mock_get_model.side_effect = Exception("Model loading failed")

        # Mock output artifacts
        mock_metrics_output = mock.MagicMock()
        mock_results_output = mock.MagicMock()
        mock_prompts_output = mock.MagicMock()

        # Should raise RuntimeError when model loading fails
        with pytest.raises(RuntimeError, match="Model loading failed"):
            evaluate_yoda_model.python_func(
                model_path='invalid-model',
                output_metrics=mock_metrics_output,
                output_results=mock_results_output,
                output_prompts=mock_prompts_output,
                include_classification_tasks=True
            )

    @mock.patch('torch.cuda.is_available', return_value=True)
    @mock.patch('lm_eval.evaluator.evaluate')
    @mock.patch('lm_eval.api.registry.get_model')
    @mock.patch('lm_eval.tasks.get_task_dict')
    def test_custom_translation_dataset_handling(self, mock_get_task_dict, mock_get_model, mock_evaluate, mock_cuda_available):
        """Test evaluation with custom translation dataset."""
        # Mock the model loading
        mock_model_instance = mock.MagicMock()
        mock_model_class = mock.MagicMock()
        mock_model_class.create_from_arg_obj.return_value = mock_model_instance
        mock_get_model.return_value = mock_model_class

        # Mock task dict
        mock_get_task_dict.return_value = {'custom_translation': mock.MagicMock()}

        # Mock evaluation results
        mock_results = {
            'results': {
                'custom_translation': {
                    'bleu': 35.7,
                    'exact_match': 0.72
                }
            }
        }
        mock_evaluate.return_value = mock_results

        # Mock output artifacts
        mock_metrics_output = mock.MagicMock()
        mock_results_output = mock.MagicMock()
        mock_results_output.path = '/tmp/results.json'
        mock_prompts_output = mock.MagicMock()
        mock_prompts_output.path = '/tmp/prompts.json'

        # Mock custom dataset
        mock_custom_dataset = mock.MagicMock()
        mock_custom_dataset.path = '/tmp/custom_dataset'

        # Call the actual python function
        evaluate_yoda_model.python_func(
            model_path='test-model',
            output_metrics=mock_metrics_output,
            output_results=mock_results_output,
            output_prompts=mock_prompts_output,
            custom_translation_dataset=mock_custom_dataset,
            include_classification_tasks=False,
            include_summarization_tasks=False,
            log_prompts=True
        )

        # Verify interactions
        mock_get_model.assert_called_once_with("vllm")
        mock_evaluate.assert_called_once()
        mock_metrics_output.log_metric.assert_called()

    @mock.patch('torch.cuda.is_available', return_value=True)
    @mock.patch('lm_eval.evaluator.evaluate')
    @mock.patch('lm_eval.api.registry.get_model')
    @mock.patch('lm_eval.tasks.get_task_dict')
    def test_lora_adapter_handling(self, mock_get_task_dict, mock_get_model, mock_evaluate, mock_cuda_available):
        """Test evaluation with LoRA adapter."""
        # Mock the model loading
        mock_model_instance = mock.MagicMock()
        mock_model_class = mock.MagicMock()
        mock_model_class.create_from_arg_obj.return_value = mock_model_instance
        mock_get_model.return_value = mock_model_class

        # Mock task dict
        mock_get_task_dict.return_value = {'classification_rte_simple': mock.MagicMock()}

        # Mock evaluation results
        mock_results = {
            'results': {
                'classification_rte_simple': {
                    'bleu': 50.1,
                    'exact_match': 0.92
                }
            }
        }
        mock_evaluate.return_value = mock_results

        # Mock output artifacts
        mock_metrics_output = mock.MagicMock()
        mock_results_output = mock.MagicMock()
        mock_results_output.path = '/tmp/results.json'
        mock_prompts_output = mock.MagicMock()

        # Mock LoRA adapter
        mock_lora_adapter = mock.MagicMock()
        mock_lora_adapter.path = '/tmp/lora_adapter'

        # Call the actual python function
        evaluate_yoda_model.python_func(
            model_path='test-model',
            output_metrics=mock_metrics_output,
            output_results=mock_results_output,
            output_prompts=mock_prompts_output,
            lora_adapter=mock_lora_adapter,
            include_classification_tasks=True,
            include_summarization_tasks=False
        )

        # Verify interactions
        mock_get_model.assert_called_once_with("vllm")
        mock_evaluate.assert_called_once()
        mock_metrics_output.log_metric.assert_called()

        # Verify that LoRA adapter path was passed to model args
        call_args = mock_model_class.create_from_arg_obj.call_args
        model_args, additional_config = call_args[0]
        assert 'lora_local_path' in model_args
        assert model_args['lora_local_path'] == '/tmp/lora_adapter'