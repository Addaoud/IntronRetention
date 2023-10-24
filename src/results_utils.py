from typing import Dict, Any, Optional, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def plot_distribution(
    weights: List[float], file_path: str, use_log_scale: Optional[bool] = False
):
    sns.displot(weights)
    plt.xlabel("Weight")
    if use_log_scale:
        plt.yscale("log")
    plt.savefig(file_path)
    plt.close()
