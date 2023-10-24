import torch
import numpy as np
import os
from sklearn import metrics
from datetime import datetime
from utils import save_model_log, save_data_to_csv, plot_loss
from fastprogress import progress_bar
from typing import List


def train_loop(
    device, model, dataloader, loss_fn, optimizer, n_accumulated_batches: int
) -> float:
    """
    Executes a single epoch of training.
    """
    model.train()
    loss_per_epoch = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device, dtype=torch.float), target.to(device)
        pred = model(data)
        loss = loss_fn(pred, target)
        # normalize loss to account for batch accumulation
        loss = loss / n_accumulated_batches
        loss.backward()
        loss_per_epoch += loss.item()
        # weights update
        if ((batch_idx + 1) % n_accumulated_batches == 0) or (
            batch_idx + 1 == len(dataloader)
        ):
            optimizer.step()
            optimizer.zero_grad()
    return loss_per_epoch / (batch_idx + 1)


def get_valid_loss(model, device, dataloader, loss_fn) -> float:
    """
    get the average loss function on dataloader
    """
    with torch.no_grad():
        model.eval()
        valid_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device, dtype=torch.float), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            valid_loss += loss.item()
    valid_loss /= batch_idx + 1
    return valid_loss


def train_model(
    model,
    optimizer,
    loss_fn,
    device,
    max_epochs: int,
    train_dataloader,
    valid_loader,
    counter_for_early_stop_threshold: int,
    epochs_to_check_loss: int,
    batch_accumulation: bool,
    results_path: str,
):
    """
    Train the model
    """
    if batch_accumulation:
        n_accumulated_batches = len(train_dataloader)
    else:
        n_accumulated_batches = 1

    best_valid_loss = np.inf
    counter_for_early_stop = 0
    start_time = datetime.now()
    model_path = os.path.join(
        results_path,
        "model_{0}.pkl".format(start_time.strftime("%y_%m_%d:%H:%M")),
    )
    loss_csv_path = os.path.join(
        results_path,
        "loss_{0}.csv".format(start_time.strftime("%y_%m_%d:%H:%M")),
    )
    loss_plot_path = os.path.join(
        results_path,
        "loss_plot_{0}.png".format(start_time.strftime("%y_%m_%d:%H:%M")),
    )
    data_dict = dict()
    data_dict["training for "] = "%i epochs" % max_epochs
    data_dict["number of batches is"] = "%i batches" % len(train_dataloader)
    data_dict["number of accumulated batches"] = "%i batches" % n_accumulated_batches
    data_dict["started training on"] = start_time
    save_model_log(log_dir=results_path, data_dictionary=data_dict)

    for epoch in progress_bar(range(1, max_epochs + 1)):
        train_loss_per_epoch = train_loop(
            device, model, train_dataloader, loss_fn, optimizer, n_accumulated_batches
        )
        valid_loss_per_epoch = float("nan")
        if (epoch % epochs_to_check_loss == 0) and (
            counter_for_early_stop_threshold > 0
        ):
            valid_loss_per_epoch = get_valid_loss(
                model=model, device=device, dataloader=valid_loader, loss_fn=loss_fn
            )
            counter_for_early_stop += 1
            if valid_loss_per_epoch < best_valid_loss:
                best_valid_loss = valid_loss_per_epoch
                counter_for_early_stop = 0
            elif counter_for_early_stop == counter_for_early_stop_threshold:
                save_data_to_csv(
                    data_dictionary={
                        "epoch": epoch,
                        "train_loss": train_loss_per_epoch,
                        "valid_loss": valid_loss_per_epoch,
                    },
                    csv_file_path=loss_csv_path,
                )
                end_time = datetime.now()
                save_model_log(
                    log_dir=results_path,
                    data_dictionary={"early stopping at epoch": epoch},
                )
                save_model_log(
                    log_dir=results_path,
                    data_dictionary={"finished training on": end_time},
                )
                torch.save(model, model_path)
                plot_loss(loss_csv_path=loss_csv_path, loss_path=loss_plot_path)
                return model
        save_data_to_csv(
            data_dictionary={
                "epoch": epoch,
                "train_loss": train_loss_per_epoch,
                "valid_loss": valid_loss_per_epoch,
            },
            csv_file_path=loss_csv_path,
        )
    end_time = datetime.now()
    save_model_log(
        log_dir=results_path,
        data_dictionary={"finished training on": end_time},
    )
    torch.save(model, model_path)
    plot_loss(loss_csv_path=loss_csv_path, loss_path=loss_plot_path)
    return model_path


def onehot_encode_labels(labels_list: List[int], num_labels: int):
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


def evaluate_model(model, dataloader, activation_function, device):
    """
    evaluate the model on dataloader and return the accuracy, auroc, and auprc
    """
    auroc_list = []
    auprc_list = []
    accuracy_list = []
    with torch.no_grad():
        model.eval()
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device, dtype=torch.float), target.numpy()
            output = activation_function(model(data))
            preds = output.cpu().detach().numpy()
            if preds.shape[1] == 1:
                auroc_list.append(metrics.roc_auc_score(target, preds))
                amax = np.round(preds) == target
                accuracy_list.append(sum(list(amax)) / len(target))
                precision, recall, thresholds = metrics.precision_recall_curve(
                    target, preds
                )
                auprc_list.append(np.mean(metrics.auc(recall, precision)))
            elif preds.shape[1] == 2:
                auroc_list.append(metrics.roc_auc_score(target, preds[:, 1]))
                amax = np.round(preds[:, 1]) == target
                accuracy_list.append(sum(list(amax)).item() / len(target))
                precision, recall, thresholds = metrics.precision_recall_curve(
                    target, preds[:, 1]
                )
                auprc_list.append(np.mean(metrics.auc(recall, precision)))
            else:
                onehot_encoded_labels = onehot_encode_labels(
                    target.tolist(), preds.shape[1]
                )
                for i in range(preds.shape[1]):
                    auroc_list.append(
                        metrics.roc_auc_score(onehot_encoded_labels[:, i], preds[:, i])
                    )
                    amax = preds[:, i].argmax(1) == onehot_encoded_labels[:, i]
                    accuracy_list.append(sum(list(amax)).item() / len(target))
                    auprc_list.append(
                        np.mean(
                            metrics.average_precision_score(
                                onehot_encode_labels(target.tolist(), preds.shape[1]),
                                preds,
                                average="macro",
                            )
                        )
                    )
    auroc = np.mean(auroc_list)
    auprc = np.mean(auprc_list)
    accuracy = np.mean(accuracy_list)
    print(f"accuracy is {accuracy}")
    print(f"auroc is {auroc}")
    print(f"auprc is {auprc}")
    return (accuracy, auroc, auprc)
