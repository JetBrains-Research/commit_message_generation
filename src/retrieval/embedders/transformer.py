import logging
from typing import List

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

    def __init__(self, name_or_path: str, device: str = "cpu", normalize_embeddings: bool = True):
        assert device in ["cpu", "cuda"]
        if device == "cuda" and torch.cuda.device_count() > 1:
            raise ValueError("Please, specify GPU by setting CUDA_VISIBLE_DEVICES environment variable.")

        self._device = device
        self._normalize_embeddings = normalize_embeddings

        self.model = AutoModel.from_pretrained(name_or_path)
        if self.model.config.model_type == "t5":
            logging.info("T5 model is passed, extracting encoder")
            self.model = self.model.encoder
        self.model.to(self._device)
        self.model.eval()

    def transform(self, batch: BatchRetrieval, *args, **kwargs) -> List[CommitEmbeddingExample]:
        """Return embeddings for given list of strings.

        It includes the following steps:
        * run through model, obtain last_hidden_state of shape (batch_size, sequence_length, hidden_size)
        * compute mean by sequence_length dimension and obtain embeddings of shape (batch_size, hidden_size)
        * (optional) normalize embeddings so that each embedding's L2 norm is equal to 1
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch.encoder_input_ids.to(self._device),
                attention_mask=batch.encoder_attention_mask.to(self._device),
            )
            embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            if self._normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            np_embeddings = embeddings.cpu().numpy()
            return [
                CommitEmbeddingExample(diff_embedding=np_embedding, pos_in_file=pos_in_file)
                for np_embedding, pos_in_file in zip(np_embeddings, batch.pos_in_file)
            ]

    @property
    def embeddings_dim(self):
        if self.model.config.model_type == "t5":
            return self.model.config.d_model

        if self.model.config.model_type in ["bert", "roberta"]:
            return self.model.config.hidden_size
