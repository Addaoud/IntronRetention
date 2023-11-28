from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from .seed import set_seed

sns.set()
set_seed()


def plot_distribution(
    weights: List[float], file_path: str, use_log_scale: Optional[bool] = False
):
    sns.displot(weights)
    plt.xlabel("Weight")
    if use_log_scale:
        plt.yscale("log")
    plt.savefig(file_path)
    plt.close()
