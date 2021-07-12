import hydra
import os
from omegaconf import DictConfig
from onnx_utils import ONNXConverter
from onnx_utils import DataCreator


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    for sub_cfg in cfg:
        for key in cfg[sub_cfg]:
            if "path" in key:
                cfg[sub_cfg][key] = os.path.join(hydra.utils.get_original_cwd(), cfg[sub_cfg][key])

    converter = ONNXConverter(**cfg.converter)
    converter.convert_and_optimize()

    data_creator = DataCreator(**cfg.data_creator)
    data_creator.create_data(**cfg.test_data_params)


if __name__ == "__main__":
    main()
