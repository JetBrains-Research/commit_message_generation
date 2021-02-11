import faiss
import numpy as np


class FaissKNN:
    def __init__(self, k, d):
        self.k = k  # number of nearest neighbor
        self.index = faiss.IndexFlatIP(d)  # dimension of vectors
        self.train_labels = []

    def fit(self, train_embeddings, train_labels):
        for labels in train_labels:
            self.train_labels.append(labels)             # add labels
        # normalize vectors to use cosine distance
        faiss.normalize_L2(train_embeddings)
        self.index.add(train_embeddings)                 # add embeddings to the index

    def predict(self, test_embeddings):
        # normalize vectors to use cosine distance
        faiss.normalize_L2(test_embeddings)
        _, test_index = self.index.search(test_embeddings, self.k)
        try:
            test_preds = self.train_labels[test_index]
        except TypeError:
            self.train_labels = np.array(self.train_labels)
            test_preds = self.train_labels[test_index]
        return test_preds