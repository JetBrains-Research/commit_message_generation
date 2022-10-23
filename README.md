# Commit Message Generation: kNN

This branch contains a simple implementation of kNN approach to commit message generation task.

It follows the basic steps of NNGen outlined in *"Neural-Machine-Translation-Based Commit Message
Generation: How Far Are We?"* paper from ASE, 2018:

* use bag-of-words approach to obtain diff embeddings;
* find k nearest diffs in terms of cosine similarity between embeddings;
* return message corresponding to diff with the highest BLEU score.

This implementation uses ANN library [annoy](https://github.com/spotify/annoy).

---

This is a work in progress, the possible next steps are:

* use sparse representations of one-hot embeddings
* try more sophisticated ways to construct diff embeddings
