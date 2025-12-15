"""Tests for the yoda_data_processor component."""

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

from kfp_components.components.data_processing.yoda_data_processor import prepare_yoda_dataset


class TestYodaDataProcessorCompilation:
    """Test component compilation."""

    def test_component_compilation(self):
        """Test that the component compiles successfully without errors."""
        # Create a temporary file to store the compiled component
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            # Compile the component - this should not raise any exceptions
            compiler.Compiler().compile(
                prepare_yoda_dataset,
                package_path=temp_path
            )

            # Verify the file was created and is not empty
            compiled_path = Path(temp_path)
            assert compiled_path.exists()
            assert compiled_path.stat().st_size > 0

            # Verify the compiled YAML contains expected component structure
            with open(compiled_path, 'r') as f:
                content = f.read()
                assert 'name: prepare-yoda-dataset' in content
                assert 'implementation' in content
                assert 'container' in content
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)


@dataclass
class TestData:
    """Test data structure for parameterized tests."""
    pipeline_func: Any
    pipeline_func_args: Optional[Dict] = None
    expected_output: Any = None


class TestYodaDataProcessorLocalRunner:
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

    @mock.patch('datasets.load_dataset')
    def test_local_execution(self, mock_load_dataset):
        """Test component execution with LocalRunner."""
        # Create mock dataset
        mock_dataset_item = {
            'sentence': 'This is a test sentence.'
        }

        mock_dataset = mock.MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.rename_column.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.train_test_split.return_value = {
            'train': mock.MagicMock(),
            'test': mock.MagicMock()
        }

        # Configure the train and test splits
        mock_train = mock_dataset.train_test_split.return_value['train']
        mock_train.__len__.return_value = 80
        mock_test = mock_dataset.train_test_split.return_value['test']
        mock_test.__len__.return_value = 20

        mock_load_dataset.return_value = mock_dataset

        # Execute the component
        pipeline_task = prepare_yoda_dataset({
            'yoda_input_dataset': 'test-dataset',
            'train_split_ratio': 0.8
        })

        # Verify the task was created successfully
        assert pipeline_task is not None

        # Verify dataset loading was called
        mock_load_dataset.assert_called_once_with('test-dataset', split='train')

        # Verify dataset operations were called
        mock_dataset.rename_column.assert_called()
        mock_dataset.map.assert_called()
        mock_dataset.train_test_split.assert_called_with(test_size=0.2, seed=42)


class TestYodaDataProcessorUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(prepare_yoda_dataset)
        assert hasattr(prepare_yoda_dataset, 'python_func')

    @mock.patch('datasets.load_dataset')
    def test_component_with_default_parameters(self, mock_load_dataset):
        """Test component with default operation_map and train_split_ratio."""
        # Setup mock dataset
        mock_dataset = mock.MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.rename_column.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.train_test_split.return_value = {
            'train': mock.MagicMock(),
            'test': mock.MagicMock()
        }

        mock_train = mock_dataset.train_test_split.return_value['train']
        mock_train.__len__.return_value = 80
        mock_test = mock_dataset.train_test_split.return_value['test']
        mock_test.__len__.return_value = 20

        mock_load_dataset.return_value = mock_dataset

        # Mock output datasets
        mock_train_output = mock.MagicMock()
        mock_train_output.path = '/tmp/train'
        mock_eval_output = mock.MagicMock()
        mock_eval_output.path = '/tmp/eval'

        # Call the actual python function
        prepare_yoda_dataset.python_func(
            yoda_input_dataset='test-dataset',
            yoda_train_dataset=mock_train_output,
            yoda_eval_dataset=mock_eval_output
        )

        # Verify interactions
        mock_load_dataset.assert_called_once_with('test-dataset', split='train')
        mock_dataset.rename_column.assert_called_with('sentence', 'prompt')
        mock_dataset.train_test_split.assert_called_with(test_size=0.2, seed=42)

    @mock.patch('datasets.load_dataset')
    def test_component_with_custom_operation_map(self, mock_load_dataset):
        """Test component with custom operation_map."""
        # Setup mock dataset
        mock_dataset = mock.MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.rename_column.return_value = mock_dataset
        mock_dataset.remove_columns.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.train_test_split.return_value = {
            'train': mock.MagicMock(),
            'test': mock.MagicMock()
        }

        mock_train = mock_dataset.train_test_split.return_value['train']
        mock_train.__len__.return_value = 70
        mock_test = mock_dataset.train_test_split.return_value['test']
        mock_test.__len__.return_value = 30

        mock_load_dataset.return_value = mock_dataset

        # Mock output datasets
        mock_train_output = mock.MagicMock()
        mock_train_output.path = '/tmp/train'
        mock_eval_output = mock.MagicMock()
        mock_eval_output.path = '/tmp/eval'

        # Custom operation map with rename and remove operations
        operation_map = {
            "rename_column": {"sentence": "prompt"},
            "remove_columns": "translation"
        }

        # Call the actual python function
        prepare_yoda_dataset.python_func(
            yoda_input_dataset='test-dataset',
            yoda_train_dataset=mock_train_output,
            yoda_eval_dataset=mock_eval_output,
            operation_map=operation_map,
            train_split_ratio=0.7
        )

        # Verify interactions
        mock_load_dataset.assert_called_once_with('test-dataset', split='train')
        mock_dataset.rename_column.assert_called_with('sentence', 'prompt')
        mock_dataset.remove_columns.assert_called_with(['translation'])
        mock_dataset.train_test_split.assert_called_with(test_size=0.3, seed=42)

    @mock.patch('datasets.load_dataset')
    def test_component_invalid_operation_raises_error(self, mock_load_dataset):
        """Test that invalid operation_map raises appropriate error."""
        # Setup mock dataset
        mock_dataset = mock.MagicMock()
        mock_dataset.__len__.return_value = 100

        mock_load_dataset.return_value = mock_dataset

        # Mock output datasets
        mock_train_output = mock.MagicMock()
        mock_train_output.path = '/tmp/train'
        mock_eval_output = mock.MagicMock()
        mock_eval_output.path = '/tmp/eval'

        # Invalid operation map
        operation_map = {
            "invalid_operation": {"sentence": "prompt"}
        }

        # Should raise InvalidValue error
        with pytest.raises(Exception):  # The actual error type from the component
            prepare_yoda_dataset.python_func(
                yoda_input_dataset='test-dataset',
                yoda_train_dataset=mock_train_output,
                yoda_eval_dataset=mock_eval_output,
                operation_map=operation_map
            )

    @mock.patch('datasets.load_dataset')
    def test_component_invalid_rename_operation_value(self, mock_load_dataset):
        """Test that invalid rename operation value raises RuntimeError."""
        # Setup mock dataset
        mock_dataset = mock.MagicMock()
        mock_dataset.__len__.return_value = 100

        mock_load_dataset.return_value = mock_dataset

        # Mock output datasets
        mock_train_output = mock.MagicMock()
        mock_train_output.path = '/tmp/train'
        mock_eval_output = mock.MagicMock()
        mock_eval_output.path = '/tmp/eval'

        # Invalid operation map - rename_column should be dict, not string
        operation_map = {
            "rename_column": "invalid_value"  # Should be dict
        }

        # Should raise RuntimeError
        with pytest.raises(RuntimeError):
            prepare_yoda_dataset.python_func(
                yoda_input_dataset='test-dataset',
                yoda_train_dataset=mock_train_output,
                yoda_eval_dataset=mock_eval_output,
                operation_map=operation_map
            )