import pandas as pd
import omegaconf
from tqdm import tqdm
from src.data_utils import DataProcessor
from src.model import EncoderDecoder
from src.generate import generate
from transformers import AutoTokenizer

df = pd.read_pickle("data/test.df")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
outfile = "data/output_without_eos.csv"

cfg = omegaconf.OmegaConf.load("configs/dataset_config.yaml")
model = EncoderDecoder(**cfg.model)
model.to(cfg.device)
data_processor = DataProcessor(**cfg.data_processor)


for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    res = generate(model=model,
                   cfg=cfg,
                   data_processor=data_processor,
                   diff=row["diff"],
                   msg=row["short_message"],
                   history=row["history"],
                   crop_prompt=True)
    res["sequences"] = tokenizer.batch_decode(res["sequences"], skip_special_tokens=True)[0]
    res["scores"] = res["scores"].tolist()[0]
    if i == 1252844:
        pd.DataFrame(res, index=[i]).to_csv(outfile, mode='w', header=True)
    else:
        pd.DataFrame(res, index=[i]).to_csv(outfile, mode='a', header=False)
