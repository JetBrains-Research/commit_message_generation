import os
from time import time
from pathlib import Path
from typing import Tuple, Dict

import torch
import numpy as np

from transformers.convert_graph_to_onnx import convert
import onnxruntime as rt
from onnxruntime.transformers import optimizer
from onnxruntime.transformers import bert_test_data

import onnx
from onnx_model import OnnxModel
from onnx import ModelProto, TensorProto, helper
from onnx.numpy_helper import to_array


def merge_configs(base: Dict[str, int], patch: Dict[str, int]):
    new_config = base.copy()
    for name, value in patch.items():
        new_config[name] = value
    return new_config


def get_roberta_inputs(onnx_file: str):
    """Find graph inputs for BERT model.
       We will deduce from EmbedLayerNormalization node.

       Copied from onnxruntime.transformers.bert_test_data.get_bert_inputs with slight changes, mainly trying not to
       use segment_ids.

       Args:
           onnx_file (str): file with onnx model
       """
    model = ModelProto()
    with open(onnx_file, "rb") as f:
        model.ParseFromString(f.read())

    onnx_model = OnnxModel(model)
    graph_inputs = onnx_model.get_graph_inputs_excluding_initializers()

    embed_nodes = onnx_model.get_nodes_by_op_type('EmbedLayerNormalization')
    if len(embed_nodes) == 1:
        embed_node = embed_nodes[0]
        input_ids = bert_test_data.get_graph_input_from_embed_node(onnx_model, embed_node, 0)
        segment_ids = bert_test_data.get_graph_input_from_embed_node(onnx_model, embed_node, 1)
        input_mask = bert_test_data.get_graph_input_from_embed_node(onnx_model, embed_node, 7)
        return input_ids, input_mask

    # Try guess the inputs based on naming.
    input_ids = None
    input_mask = None
    for input in graph_inputs:
        input_name_lower = input.name.lower()
        if "mask" in input_name_lower:  # matches input with name like "attention_mask" or "input_mask"
            input_mask = input
        elif "token" in input_name_lower or "segment" in input_name_lower:  # matches input with name like "segment_ids" or "token_type_ids"
            pass
        else:
            input_ids = input

        if input_ids and input_mask:
            return input_ids, input_mask


def make_inputs(model_path: str, config: Dict[str, int]) -> Tuple[torch.Tensor]:
    # create fake test data
    input_ids, input_mask = get_roberta_inputs(model_path)
    inputs = bert_test_data.fake_test_data(config['batch_size'], config['sequence_length'], 1,
                                           config['vocab_size'], True, 42, input_ids, segment_ids=None,
                                           input_mask=input_mask,
                                           random_mask_length=True)
    return inputs


def write_inputs(inputs: Dict[str, np.array], config: Dict[str, int], dir_path: str):
    input_dim = [config["batch_size"], config["sequence_length"]]
    attention_mask_dim = [config["batch_size"], config["sequence_length"]]

    with open(f'{dir_path}/input_0.pb', 'wb') as the_file:
        the_file.write(helper.make_tensor('input_ids', TensorProto.INT64, input_dim,
                                          inputs[0]['input_ids'].flatten()).SerializeToString())
    with open(f'{dir_path}/input_1.pb', 'wb') as the_file:
        the_file.write(helper.make_tensor('attention_mask', TensorProto.INT64, attention_mask_dim,
                                          inputs[0]['attention_mask'].flatten()).SerializeToString())


def read_inputs(dir_path: str):
    inputs = {'input_ids': to_array(onnx.load_tensor(f"{dir_path}/input_0.pb")),
              'attention_mask': to_array(onnx.load_tensor(f"{dir_path}/input_1.pb"))}
    return inputs


def evaluate_model(inputs: Dict, model_path: str, iters: int = 1) -> Tuple[torch.Tensor, ...]:
    session = rt.InferenceSession(model_path)
    start = time()
    output = session.run(None, inputs)
    for _ in range(iters - 1):
        output = session.run(None, inputs)
    print("Time:", (time() - start) / iters)
    return output


def write_outputs(output: Tuple[torch.Tensor, ...], config: Dict[str, int], dir_path: str):
    last_hidden_state_dim = [config["batch_size"], config["sequence_length"], config["hidden_size"]]
    pooler_output_dim = [config["batch_size"], config["hidden_size"]]
    with open(f'{dir_path}/output_0.pb', 'wb') as the_file:
        the_file.write(helper.make_tensor(f'last_hidden_state', TensorProto.FLOAT, last_hidden_state_dim,
                                          output[0].flatten()).SerializeToString())
    with open(f'{dir_path}/output_1.pb', 'wb') as the_file:
        the_file.write(helper.make_tensor(f'pooler_output', TensorProto.FLOAT, pooler_output_dim,
                                          output[1].flatten()).SerializeToString())


def create_data_for_config(out_dir: str, model_path: str, name: str, config: Dict[str, int], n_iters: int = 1):
    data_dir = f"{out_dir}/{name}"
    os.makedirs(data_dir, exist_ok=True)

    print("Creating fake data")
    inputs = make_inputs(model_path, config)
    print("Writing fake data")
    write_inputs(inputs, config, data_dir)
    print("Reading fake data")
    inputs = read_inputs(data_dir)
    print(inputs)
    print("Evaluating model on fake data")
    output = evaluate_model(inputs, model_path, n_iters)  # type: ignore
    print("Writing outputs")
    write_outputs(output, config, data_dir)


if __name__ == '__main__':
    base_config = {
        'num_heads': 12,
        'hidden_size': 768,
        "num_layer": 4,
        "vocab_size": 50266}

    config0 = {
        "batch_size": 1,
        "sequence_length": 40
    }

    config1 = {
        "batch_size": 8,
        "sequence_length": 40
    }

    model_base = 'diff_w_changes'

    # convert model to onnx (if not already)
    if not Path(f"{model_base}/onnx/roberta.onnx").is_file():
        print("Converting model to ONNX")
        model_path = f'./{model_base}'
        tokenizer_path = './tokenizer'
        convert(framework="pt", model=model_path, tokenizer=tokenizer_path,
                output=Path(f"{model_base}/onnx/roberta.onnx"), opset=11)

    # optimize converted model (if not already)
    if not Path(f"{model_base}/onnx/roberta_fp16.onnx").is_file():
        print("Optimizing converted model")
        optimized_model = optimizer.optimize_model(f"{model_base}/onnx/roberta.onnx", model_type='bert',
                                                   num_heads=base_config['num_heads'],
                                                   hidden_size=base_config['hidden_size'])
        optimized_model.convert_model_float32_to_float16()
        optimized_model.save_model_to_file(f"{model_base}/onnx/roberta_fp16.onnx")

    # create input&output data for batch_size 1
    create_data_for_config(f'{model_base}/onnx/',
                           f"{model_base}/onnx/roberta_fp16.onnx", "test_data_set_batch1_seq40",
                           merge_configs(base_config, config0), 1)

    # create input&output data for batch_size 8
    create_data_for_config(f'{model_base}/onnx/',
                           f"{model_base}/onnx/roberta_fp16.onnx", "test_data_set_batch8_seq40",
                           merge_configs(base_config, config1), 1)
