import numpy as np
import pytest

from src.embedders import TransformerEmbedder


def test_bert_embedder():
    embedder = TransformerEmbedder(name_or_path="bert-base-uncased")
    embeddings = embedder._transform(["example input", "another example input"])
    assert embedder.embeddings_dim == embedder.model.config.hidden_size
    assert embeddings.shape == (2, embedder.embeddings_dim)
    assert np.linalg.norm(embeddings, axis=1).tolist() == pytest.approx([1, 1])


def test_t5_embedder():
    embedder = TransformerEmbedder(name_or_path="t5-small")
    embeddings = embedder._transform(["example input", "another example input"])
    assert embedder.embeddings_dim == embedder.model.config.d_model
    assert embeddings.shape == (2, embedder.embeddings_dim)
    assert np.linalg.norm(embeddings, axis=1).tolist() == pytest.approx([1, 1])
