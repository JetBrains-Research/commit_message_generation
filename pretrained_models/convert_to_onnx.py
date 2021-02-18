from pathlib import Path
from transformers.convert_graph_to_onnx import convert
from onnxruntime.transformers import optimizer


# convert model to onnx (if not already)
if not Path("onnx/roberta.onnx").is_file():
        model_path = './model'
        tokenizer_path = './tokenizer'
        convert(framework="pt", model=model_path, tokenizer=tokenizer_path,
                output=Path("onnx/roberta.onnx"), opset=11)

# optimize converted model (if not already)
if not Path("onnx/roberta_fp16.onnx").is_file():
        optimized_model = optimizer.optimize_model("onnx/roberta.onnx", model_type='bert', num_heads=12, hidden_size=768)
        optimized_model.convert_model_float32_to_float16()
        optimized_model.save_model_to_file("onnx/roberta_fp16.onnx")
