import torch
import numpy as np
import os
from datetime import datetime
from .utils import save_model_log, save_data_to_csv, plot_loss
from .seed import set_seed
from fastprogress import progress_bar
from typing import Optional

set_seed()


def train_loop(
    device: str,
    model: torch.nn.Module,
    dataloader: torch.utils.data.dataloader,
    loss_fn: torch.nn.modules.loss,
    optimizer: torch.optim,
    n_accumulated_batches: int,
) -> float:
    """
    Executes a single epoch of training.
    """
    model.train()
    loss_per_epoch = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = {key: data[key].to(device) for key in data}, target.to(device)
        pred = model(**data)
        loss = loss_fn(pred, target)
        # normalize loss to account for batch accumulation
        loss = loss / n_accumulated_batches
        loss.backward()
        loss_per_epoch += loss.item() * n_accumulated_batches
        # weights update
        if ((batch_idx + 1) % n_accumulated_batches == 0) or (
            batch_idx + 1 == len(dataloader)
        ):
            optimizer.step()
            optimizer.zero_grad()
    return loss_per_epoch / (batch_idx + 1)


def get_valid_loss(
    model: torch.nn.Module,
    device: str,
    dataloader: torch.utils.data.dataloader,
    loss_fn: torch.nn.modules.loss,
) -> float:
    """
    get the average loss function on dataloader
    """
    with torch.no_grad():
        model.eval()
        valid_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = {key: data[key].to(device) for key in data}, target.to(
                device
            )
            output = model(**data)
            loss = loss_fn(output, target)
            valid_loss += loss.item()
    valid_loss /= batch_idx + 1
    return valid_loss


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim,
    loss_fn: torch.nn.modules.loss,
    device: str,
    results_path: str,
    max_epochs: int,
    train_dataloader: torch.utils.data.dataloader,
    valid_loader: torch.utils.data.dataloader = None,
    counter_for_early_stop_threshold: Optional[int] = 0,
    epochs_to_check_loss: Optional[int] = 0,
    batch_accumulation: Optional[bool] = False,
    n_accumulated_batches: Optional[int] = 1,
) -> str:
    """
    Train the model
    """
    if n_accumulated_batches > len(train_dataloader) or batch_accumulation:
        n_accumulated_batches = len(train_dataloader)
    best_valid_loss = np.inf
    counter_for_early_stop = 0
    start_time = datetime.now()
    model_path = os.path.join(
        results_path,
        "model_{0}.pt".format(start_time.strftime("%y_%m_%d:%H:%M")),
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
                torch.save(model.state_dict(), model_path)
                plot_loss(loss_csv_path=loss_csv_path, loss_path=loss_plot_path)
                return model_path
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
    torch.save(model.state_dict(), model_path)
    plot_loss(loss_csv_path=loss_csv_path, loss_path=loss_plot_path)
    return model_path
