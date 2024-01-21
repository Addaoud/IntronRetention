from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn import metrics
from fastprogress import progress_bar
from .seed import set_seed

sns.set()
set_seed()


def onehot_encode_labels(labels_list: List[int], num_labels: int) -> np.array:
    """
    creates one hot encoded labels, e.g. [0,1,2] => [ [1,0,0] , [0,1,0] , [0,0,1] ]
    :param labels_list:
        list containing the labels
    :param num_labels:
        number of unique labels
    :return:
        numpy array containing the one hot encoded labels list
    """
    return np.eye(num_labels)[labels_list]


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.dataloader,
    activation_function: torch.nn.modules.loss,
    device: str,
    return_details: Optional[bool] = False,
) -> tuple[float, float, float]:
    """
    evaluate the model on dataloader and return the accuracy, auroc, and auprc
    """
    target_list = list()
    preds_list = list()
    print("Evaluating model")
    with torch.no_grad():
        model.eval()
        for _, (data, target) in enumerate(progress_bar(dataloader)):
            data = {key: data[key].to(device) for key in data}
            output = activation_function(model(**data))
            preds = output.cpu().detach().numpy()
            target_list.extend(target)
            preds_list.extend(preds)
    if preds.shape[1] == 1:
        fpr, tpr, _ = metrics.roc_curve(target_list, preds_list)
        auroc = np.mean(metrics.auc(fpr, tpr))
        amax = np.round(preds_list) == target_list
        accuracy = sum(list(amax)).item() / len(target_list)
        precision, recall, _ = metrics.precision_recall_curve(target_list, preds_list)
        auprc = np.mean(metrics.auc(recall, precision))
    elif preds.shape[1] == 2:
        fpr, tpr, _ = metrics.roc_curve(target_list, np.array(preds_list)[:, 1])
        auroc = np.mean(metrics.auc(fpr, tpr))
        amax = np.round(np.array(preds_list)[:, 1]) == target_list
        accuracy = sum(list(amax)).item() / len(target_list)
        precision, recall, _ = metrics.precision_recall_curve(
            target_list, np.array(preds_list)[:, 1]
        )
        auprc = np.mean(metrics.auc(recall, precision))
    else:
        auroc_list = []
        accuracy_list = []
        onehot_encoded_labels = onehot_encode_labels(target_list, preds.shape[1])
        for i in range(preds.shape[1]):
            auroc_list.append(
                metrics.roc_auc_score(
                    onehot_encoded_labels[:, i], np.array(preds_list)[:, i]
                )
            )
            amax = np.array(preds_list)[:, i].argmax(1) == onehot_encoded_labels[:, i]
            accuracy_list.append(sum(list(amax)).item() / len(target_list))
        accuracy = np.mean(accuracy_list)
        auroc = np.mean(auroc_list)
        auprc = metrics.average_precision_score(
            onehot_encode_labels(target_list, preds.shape[1]),
            np.array(preds_list),
            average="macro",
        )

    print(f"accuracy is {accuracy}")
    print(f"auroc is {auroc}")
    print(f"auprc is {auprc}")
    if return_details:
        return (accuracy, auroc, auprc, fpr, tpr, precision, recall)
    return (accuracy, auroc, auprc)


def plot_distribution(
    weights: List[float], file_path: str, use_log_scale: Optional[bool] = False
):
    sns.displot(weights)
    plt.xlabel("Weight")
    if use_log_scale:
        plt.yscale("log")
    plt.savefig(file_path)
    plt.close()
