import numpy as np  # type: ignore
import pytest
import omegaconf
from src.onnx_utils import ONNXConverter
from src.onnx_utils import DataCreator


@pytest.fixture
def default_config():
    return omegaconf.OmegaConf.load("configs/test_config.yaml")


def test_onnx_converter(default_config):
    converter = ONNXConverter(**default_config.converter)
    converter.convert_and_optimize()


def test_data_creator(default_config):
    creator = DataCreator(**default_config.data_creator)
    creator.create_data(**default_config.test_data_params)


def test_compare_outputs(default_config):
    creator_for_original = DataCreator(
        model_path=default_config.converter.onnx_encoder_path,
        dictionary_size=default_config.data_creator.dictionary_size,
    )
    creator_for_optimized = DataCreator(**default_config.data_creator)

    for iter in range(10):
        creator_for_original.create_data(
            batch_size=1,
            sequence_length=10,
            input_path="onnx_utils/data/original_inputs",
            output_path="onnx_utils/data/original_outputs",
            n_iters=1,
            test_cases=1,
            random_seed=42 + iter,
        )

        creator_for_optimized.create_data(
            batch_size=1,
            sequence_length=10,
            input_path="onnx_utils/data/optimized_inputs",
            output_path="onnx_utils/data/optimized_outputs",
            n_iters=1,
            test_cases=1,
            random_seed=42 + iter,
        )

        original_inputs = DataCreator.read_data("onnx_utils/data/original_inputs")
        optimized_inputs = DataCreator.read_data("onnx_utils/data/optimized_inputs")

        for key in original_inputs:
            assert np.array_equal(original_inputs[key], optimized_inputs[key])

        original_outputs = DataCreator.read_data("onnx_utils/data/original_outputs")
        optimized_outputs = DataCreator.read_data("onnx_utils/data/optimized_outputs")

        for key in original_outputs:
            assert np.allclose(original_outputs[key], optimized_outputs[key], atol=1e-2, rtol=1e-3)
