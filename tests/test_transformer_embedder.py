from typing import List

import numpy as np
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.retrieval import TransformerEmbedder
from src.retrieval.utils import CommitEmbeddingExample
from src.utils import BatchRetrieval


def create_batch(inputs: List[str], tokenizer: PreTrainedTokenizerFast) -> BatchRetrieval:
    encoded_inputs = tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")  # type: ignore[operator]
    return BatchRetrieval(
        encoder_input_ids=encoded_inputs.input_ids,
        encoder_attention_mask=encoded_inputs.attention_mask,
        pos_in_file=[i for i, _ in enumerate(inputs)],
    )


def test_bert_embedder():
    embedder = TransformerEmbedder(name_or_path="bert-base-uncased")
    assert embedder.embeddings_dim == embedder.model.config.hidden_size

    inputs = ["example input", "another example input"]
    batch = create_batch(inputs, AutoTokenizer.from_pretrained("bert-base-uncased"))

    embeddings: List[CommitEmbeddingExample] = embedder.transform(batch)
    for i, embedding in enumerate(embeddings):
        assert embedding["diff_embedding"].shape == (embedder.embeddings_dim,)
        assert np.linalg.norm(embedding["diff_embedding"]) == pytest.approx(1)
        assert embedding["pos_in_file"] == i


def test_t5_embedder():
    embedder = TransformerEmbedder(name_or_path="t5-small")
    assert embedder.embeddings_dim == embedder.model.config.d_model

    inputs = ["example input", "another example input"]
    batch = create_batch(inputs, AutoTokenizer.from_pretrained("t5-small"))

    embeddings: List[CommitEmbeddingExample] = embedder.transform(batch)
    for i, embedding in enumerate(embeddings):
        assert embedding["diff_embedding"].shape == (embedder.embeddings_dim,)
        assert np.linalg.norm(embedding["diff_embedding"]) == pytest.approx(1)
        assert embedding["pos_in_file"] == i
