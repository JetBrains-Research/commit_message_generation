import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.utils import BatchRetrieval

from ..utils import CommitEmbeddingExample


class TransformerEmbedder:
    """This class utilizes Transformer encoder to produce embeddings.

    Currently, the following architectures are supported:
    * BERT/RoBERTa
    * T5 (in this case, its encoder is used)
    """

    def __init__(self, name_or_path: str, precision: int, device: str, normalize_embeddings: bool):
        assert device in ["cpu", "cuda"]
        if device == "cuda" and torch.cuda.device_count() > 1:
            raise ValueError("Please, specify GPU by setting CUDA_VISIBLE_DEVICES environment variable.")

        self._device = device
        self._normalize_embeddings = normalize_embeddings
        self._precision = precision

        self.model = AutoModel.from_pretrained(name_or_path)
        if self.model.config.model_type == "t5":
            logging.info("T5 model is passed, extracting encoder")
            self.model = self.model.encoder
        self.model.to(self._device)
        self.model.eval()

    def _transform(self, batch: BatchRetrieval) -> torch.Tensor:
        outputs = self.model(
            input_ids=batch.encoder_input_ids.to(self._device),
            attention_mask=batch.encoder_attention_mask.to(self._device),
        )
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        if self._normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    @torch.no_grad()
    def transform(self, batch: BatchRetrieval) -> List[CommitEmbeddingExample]:
        """Return embeddings for given list of strings.

        It includes the following steps:
        * run through model, obtain last_hidden_state of shape (batch_size, sequence_length, hidden_size)
        * compute mean by sequence_length dimension and obtain embeddings of shape (batch_size, hidden_size)
        * (optional) normalize embeddings so that each embedding's L2 norm is equal to 1
        """
        if self._precision == 16 and self._device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                embeddings = self._transform(batch)
        else:
            embeddings = self._transform(batch)

        np_embeddings = embeddings.cpu().numpy()
        return CommitEmbeddingBatch(diff_embeddings=np_embeddings, pos_in_file=np.asarray(batch.pos_in_file))

    @property
    def embeddings_dim(self):
        if self.model.config.model_type == "t5":
            return self.model.config.d_model

        if self.model.config.model_type in ["bert", "roberta"]:
            return self.model.config.hidden_size
