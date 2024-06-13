import torch
import numpy as np
import os
from datetime import datetime
from .utils import save_model_log, save_data_to_csv, plot_loss
from .seed import set_seed
from fastprogress import progress_bar
from typing import Optional
from copy import deepcopy


set_seed()


class trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn: torch.nn.modules.loss,
        device: str,
        model_folder_path: str,
        max_epochs: int,
        train_dataloader: torch.utils.data.dataloader,
        valid_loader: torch.utils.data.dataloader = None,
        counter_for_early_stop_threshold: Optional[int] = 0,
        epochs_to_check_loss: Optional[int] = 0,
        batch_accumulation: Optional[bool] = False,
        n_accumulated_batches: Optional[int] = 1,
        use_scheduler: Optional[bool] = False,
        **kwargs
    ):
        self.best_model = model
        self.model = model
        self.model_folder_path = model_folder_path
        self.max_epochs = max_epochs
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.valid_loader = valid_loader
        self.counter_for_early_stop_threshold = counter_for_early_stop_threshold
        self.epochs_to_check_loss = epochs_to_check_loss
        self.batch_accumulation = batch_accumulation
        self.n_accumulated_batches = n_accumulated_batches
        self.use_scheduler = use_scheduler
        self.best_valid_loss = np.inf
        self.counter_for_early_stop = 0
        if (
            self.n_accumulated_batches > len(self.train_dataloader)
            or self.batch_accumulation
        ):
            self.n_accumulated_batches = len(train_dataloader)

    def train(self):
        start_time = datetime.now()
        model_path = os.path.join(
            self.model_folder_path,
            "model_{0}.pt".format(start_time.strftime("%y_%m_%d:%H:%M")),
        )
        loss_csv_path = os.path.join(
            self.model_folder_path,
            "loss_{0}.csv".format(start_time.strftime("%y_%m_%d:%H:%M")),
        )
        loss_plot_path = os.path.join(
            self.model_folder_path,
            "loss_plot_{0}.png".format(start_time.strftime("%y_%m_%d:%H:%M")),
        )
        data_dict = dict()
        data_dict["training for "] = "%i epochs" % self.max_epochs
        data_dict["number of batches is"] = "%i batches" % len(self.train_dataloader)
        data_dict["number of accumulated batches"] = (
            "%i batches" % self.n_accumulated_batches
        )
        data_dict["started training on"] = start_time
        save_model_log(log_dir=self.model_folder_path, data_dictionary=data_dict)
        for epoch in progress_bar(range(1, self.max_epochs + 1)):
            train_loss_per_epoch = self.train_loop()
            if self.use_scheduler:
                self.optimizer.update_lr(epoch)
            valid_loss_per_epoch = float("nan")
            if (epoch % self.epochs_to_check_loss == 0) and (
                self.counter_for_early_stop_threshold > 0
            ):
                valid_loss_per_epoch = self.get_valid_loss()
                save_data_to_csv(
                    data_dictionary={
                        "epoch": epoch,
                        "train_loss": train_loss_per_epoch,
                        "valid_loss": valid_loss_per_epoch,
                    },
                    csv_file_path=loss_csv_path,
                )
                self.counter_for_early_stop += 1
                if valid_loss_per_epoch < self.best_valid_loss:
                    self.best_valid_loss = valid_loss_per_epoch
                    self.best_model = deepcopy(self.model)
                    self.counter_for_early_stop = 0
                elif (
                    self.counter_for_early_stop == self.counter_for_early_stop_threshold
                ):
                    save_model_log(
                        log_dir=self.model_folder_path,
                        data_dictionary={"early stopping at epoch": epoch},
                    )
                    break
            else:
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
            log_dir=self.model_folder_path,
            data_dictionary={"finished training on": end_time},
        )
        torch.save(self.best_model.state_dict(), model_path)
        plot_loss(loss_csv_path=loss_csv_path, loss_path=loss_plot_path)
        return self.best_model, model_path

    def train_loop(self):
        # Executes a single epoch of training.
        self.model.train()
        loss_per_epoch = 0
        self.optimizer.zero_grad()
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            data, target = {key: data[key].to(self.device) for key in data}, target.to(
                self.device
            )
            pred = self.model(**data)
            loss = self.loss_fn(pred, target)
            # normalize loss to account for batch accumulation
            loss = loss / self.n_accumulated_batches
            loss.backward()
            loss_per_epoch += loss.item() * self.n_accumulated_batches
            # weights update
            if ((batch_idx + 1) % self.n_accumulated_batches == 0) or (
                batch_idx + 1 == len(self.train_dataloader)
            ):

                self.optimizer.step()
                self.optimizer.zero_grad()
        return loss_per_epoch / (batch_idx + 1)

    def get_valid_loss(self) -> float:
        """
        get the average loss function on dataloader
        """
        with torch.no_grad():
            self.model.eval()
            valid_loss = 0
            for batch_idx, (data, target) in enumerate(self.valid_loader):
                data, target = {
                    key: data[key].to(self.device) for key in data
                }, target.to(self.device)
                output = self.model(**data)
                loss = self.loss_fn(output, target)
                valid_loss += loss.item()
        valid_loss /= batch_idx + 1
        return valid_loss
