import os
import numpy as np  # type: ignore
from time import time
from typing import Dict, Tuple, List
import onnx  # type: ignore
from onnx import ModelProto, TensorProto, helper  # type: ignore
from onnx.numpy_helper import to_array  # type: ignore
from onnxruntime import InferenceSession  # type: ignore
from onnxruntime.transformers import bert_test_data  # type: ignore
from onnxruntime.transformers.onnx_model import OnnxModel  # type: ignore


class DataCreator:
    def __init__(self, model_path: str, dictionary_size: int):
        """
        Class for generating test input data for given ONNX RoBERTa model and saving outputs.

        :param model_path: onnx model path
        :param dictionary_size: number of tokens in model's vocab (usually vocab_size field in huggingface's configs)
        """
        self.model_path = model_path
        self.dictionary_size = dictionary_size

    def create_data(
        self,
        batch_size: int,
        sequence_length: int,
        input_path: str,
        output_path: str,
        test_cases: int = 1,
        n_iters: int = 1,
        random_mask_length: bool = False,
        random_seed: int = None,
    ):
        """
        Performs all necessary steps:
            1) Generates fake input data
            2) Runs onnx model on inputs
            3) Saves resulting outputs

        :param batch_size: batch size for generating test inputs
        :param sequence_length: sequence length for generating test inputs
        :param random_mask_length: mask random number of words at the end when generating test data
        :param test_cases: number of test cases when generating test data
        :param random_seed: seed for generating test inputs
        :param n_iters: how many times model is ran on inputs (for calculating mean time?)
        :param input_path: path to folder for saving inputs
        :param output_path: path to folder for saving outputs
        """
        os.makedirs(input_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        # creating fake inputs for testing
        inputs = DataCreator.create_fake_test_inputs(
            model_path=self.model_path,
            batch_size=batch_size,
            sequence_length=sequence_length,
            dictionary_size=self.dictionary_size,
            test_cases=test_cases,
            random_mask_length=random_mask_length,
            random_seed=random_seed,
        )
        # saving created inputs
        DataCreator.save_input(data=inputs, output_dir=input_path)
        outputs = []
        # running model inference on inputs
        for input in inputs:
            outputs.append(DataCreator.run_inference(input, model_path=self.model_path, iters=n_iters))
        # saving created outputs
        DataCreator.save_output(data=outputs, output_dir=output_path)

    @staticmethod
    def get_roberta_inputs(onnx_file: str, input_ids_name: str = None, input_mask_name: str = None):
        """
        Copied from onnxruntime.transformers.bert_test_data.get_bert_inputs with slight changes:
           * remove all logic for segment_ids as RoBERTa doesn't use them;
           * provide path to onnx model_utils instead of OnnxModel as argument.

        #-------------------------------------------------------------------------
        # Copyright (c) Microsoft Corporation.  All rights reserved.
        # Licensed under the MIT License.
        #--------------------------------------------------------------------------
        Find graph inputs for RoBERTa model_utils. First, we will deduce inputs from EmbedLayerNormalization node.
        If not found, we will guess the meaning of graph inputs based on naming.

        Args:
            onnx_file (str): onnx model path
            input_ids_name (str, optional): Name of graph input for input IDs. Defaults to None.
            input_mask_name (str, optional): Name of graph input for attention mask. Defaults to None.

        Raises:
            ValueError: Graph does not have input named of input_ids_name or input_mask_name
            ValueError: Expected graph input number does not match with specified input_ids_name and input_mask_name

        Returns:
            Tuple[Union[None, np.ndarray], Union[None, np.ndarray]]: input tensors of input_ids and input_mask
        """
        model = ModelProto()
        with open(onnx_file, "rb") as f:
            model.ParseFromString(f.read())
        onnx_model = OnnxModel(model)

        graph_inputs = onnx_model.get_graph_inputs_excluding_initializers()

        if input_ids_name is not None:
            input_ids = onnx_model.find_graph_input(input_ids_name)
            if input_ids is None:
                raise ValueError(f"Graph does not have input named {input_ids_name}")

            input_mask = None
            if input_mask_name:
                input_mask = onnx_model.find_graph_input(input_mask_name)
                if input_mask is None:
                    raise ValueError(f"Graph does not have input named {input_mask_name}")

            expected_inputs = 1 + (1 if input_mask else 0)
            if len(graph_inputs) != expected_inputs:
                raise ValueError(f"Expect the graph to have {expected_inputs} inputs. Got {len(graph_inputs)}")

            return input_ids, input_mask

        if len(graph_inputs) != 2:
            raise ValueError("Expect the graph to have 2 inputs. Got {}".format(len(graph_inputs)))

        embed_nodes = onnx_model.get_nodes_by_op_type("EmbedLayerNormalization")
        if len(embed_nodes) == 1:
            embed_node = embed_nodes[0]
            input_ids = bert_test_data.get_graph_input_from_embed_node(onnx_model, embed_node, 0)
            input_mask = bert_test_data.get_graph_input_from_embed_node(onnx_model, embed_node, 7)
            return input_ids, input_mask

        # Try guess the inputs based on naming.
        input_ids = None
        input_mask = None
        for input in graph_inputs:
            input_name_lower = input.name.lower()
            if "mask" in input_name_lower:
                input_mask = input
            elif "token" in input_name_lower or "segment" in input_name_lower:
                pass
            else:
                input_ids = input

            if input_ids and input_mask:
                return input_ids, input_mask

    @staticmethod
    def create_fake_test_inputs(
        model_path: str,
        batch_size: int,
        sequence_length: int,
        dictionary_size: int,
        random_mask_length: bool,
        test_cases: int = 1,
        verbose: bool = False,
        random_seed: int = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Find graph inputs for given RoBERTa model_utils and create given number of fake input data for testing.

        Docstring is copied from onnxruntime.transformers.bert_test_data.fake_test_data with slight changes.
        #-------------------------------------------------------------------------
        # Copyright (c) Microsoft Corporation.  All rights reserved.
        # Licensed under the MIT License.
        #--------------------------------------------------------------------------
        Args:
            model_path (str): onnx model path
            batch_size (int): batch size
            sequence_length (int): sequence length
            test_cases (int): number of test cases
            dictionary_size (int): vocabulary size of dictionary for input_ids
            verbose (bool): print more information or not
            random_seed (int): random seed
            random_mask_length (bool): whether mask random number of words at the end

        Returns:
            List[Dict[str,numpy.ndarray]]: list of test cases, where each test case is a dictonary with input name
            as key and a tensor as value
        """
        input_ids, input_mask = DataCreator.get_roberta_inputs(model_path)
        inputs = bert_test_data.fake_test_data(
            batch_size=batch_size,
            sequence_length=sequence_length,
            test_cases=test_cases,
            dictionary_size=dictionary_size,
            verbose=verbose,
            random_seed=random_seed,
            input_ids=input_ids,
            segment_ids=None,
            input_mask=input_mask,
            random_mask_length=random_mask_length,
        )
        return inputs

    @staticmethod
    def save_input(data: List[Dict[str, np.ndarray]], output_dir: str):
        """
        Save given input data to .pb files.

        :param data: inputs to save
        :param output_dir: path to output directory
        """
        for i, example in enumerate(data):
            for key in example:
                with open(os.path.join(output_dir, f"{key}_{i}.pb"), "wb") as file:
                    file.write(
                        helper.make_tensor(
                            key, TensorProto.INT64, example[key].shape, example[key].flatten()
                        ).SerializeToString()
                    )

    @staticmethod
    def read_data(input_dir: str) -> Dict[str, np.ndarray]:
        """
        Read all .pb files from given directory into single dictionary

        :param input_dir: path to folder to read data from
        :return: inputs: dictionary with filenames as keys and np.arrays as values
        """
        # TODO: might not be the most convenient option with several test cases
        inputs = {}
        for input_name in os.listdir(input_dir):
            if ".pb" in input_name:
                inputs[input_name.split(".")[0]] = to_array(onnx.load_tensor(os.path.join(input_dir, input_name)))
        return inputs

    @staticmethod
    def save_output(data: List[Tuple[np.ndarray, np.ndarray]], output_dir: str):
        """
        Save given data to .pb files.

        :param data: outputs to save
        :param output_dir: path to output directory
        """
        for i, example in enumerate(data):
            last_hidden_state = example[0]
            with open(os.path.join(output_dir, f"last_hidden_state_{i}.pb"), "wb") as file:
                file.write(
                    helper.make_tensor(
                        "last_hidden_state", TensorProto.FLOAT, last_hidden_state.shape, last_hidden_state.flatten()
                    ).SerializeToString()
                )
            pooler_output = example[1]
            with open(os.path.join(output_dir, f"pooler_output_{i}.pb"), "wb") as file:
                file.write(
                    helper.make_tensor(
                        "pooler_output", TensorProto.FLOAT, pooler_output.shape, pooler_output.flatten()
                    ).SerializeToString()
                )

    @staticmethod
    def run_inference(inputs: Dict[str, np.ndarray], model_path: str, iters: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run model on given inputs given number of times.

        :param inputs: inputs for model
        :param model_path: onnx model path
        :param iters: number of iterations
        :return: output: model output
        """
        session = InferenceSession(model_path)
        start = time()
        output = session.run(None, inputs)
        for _ in range(iters - 1):
            output = session.run(None, inputs)
        print("Time:", (time() - start) / iters)
        return output
