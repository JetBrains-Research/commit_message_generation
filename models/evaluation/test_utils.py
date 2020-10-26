import os
from typing import List
import Config
import matplotlib.pyplot as plt


def save_perplexity_plot(perplexities: List[List[float]], labels: List[str], filepath: str, config: Config) -> None:
    """Plot perplexities"""
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    for perplexity_values, label in zip(perplexities, labels):
        plt.plot(perplexity_values, label=label)
        plt.legend()
    plt.savefig(os.path.join(config['OUTPUT_PATH'], filepath))
    plt.clf()
