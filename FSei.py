import pandas as pd
import argparse
import os
import seaborn as sns

sns.set()
import torch
from src.utils import (
    create_path,
    save_model_log,
    save_data_to_csv,
    generate_UDir,
    split_targets,
    read_json,
)
from src.train_utils import train_model
from src.results_utils import evaluate_model
from src.dataset_utils import load_datasets
from src.networks import generate_FSei
from src.seed import set_seed

set_seed()


def parse_arguments(parser):
    parser.add_argument("--json", type=str, help="path to the json file")
    parser.add_argument(
        "-n",
        "--new",
        action="store_true",
        help="Build a new model",
    )
    parser.add_argument("-m", "--model_path", type=str, help="Existing model path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate FSei model")
    args = parse_arguments(parser)
    assert (
        args.json != None
    ), "Please specify the path to the json file with --json json_path"
    assert os.path.exists(
        args.json
    ), f"The path to the json file {args.json} does not exist. Please verify"
    assert (args.new == True) ^ (
        (args.model_path) != None
    ), "Wrong arguments. Either include -n to build a new model or specify -m model_path"

    config = read_json(json_path=args.json)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    (
        Sei_targets_list,
        TFs_list,
        TFs_indices,
        Histone_marks_list,
        Histone_marks_indices,
        Chromatin_access_list,
        Chromatin_access_indices,
    ) = split_targets(targets_file_pth="target.names")

    batch_size = config.train_params.get("batch_size")
    lazy_loading = config.train_params.get("lazy_loading")
    train_loader, valid_loader, test_loader = load_datasets(
        batchSize=batch_size, test_split=0.1, output_dir="data", lazyLoad=lazy_loading
    )

    if args.new:
        model = generate_FSei(
            new_model=True, use_pretrain=True, freeze_weights=False
        ).to(device)
        Udir = generate_UDir(path=config.paths.get("results_path"))
        model_folder_path = os.path.join(config.paths.get("results_path"), Udir)
        create_path(model_folder_path)
    else:
        model_folder_path = os.path.dirname(args.model_path)
        model = generate_FSei(new_model=False, model_path=args.model_path).to(device)

    # prepare the optimizer
    learning_rate = config.optimizer_params.get("learning_rate", 0.001)
    weight_decay = config.optimizer_params.get("weight_decay", 0.01)
    momentum = config.optimizer_params.get("momentum", 0)
    optimizer = config.optimizer_params.get("optimizer", "ADAM")
    if optimizer.upper() == "ADAMW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer.upper() == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare the loss function
    loss_function = torch.nn.CrossEntropyLoss(reduction="mean")
    activation_function = torch.nn.Softmax(dim=1)

    # Train params
    max_epochs = config.train_params.get("max_epochs")
    counter_for_early_stop = config.train_params.get("counter_for_early_stop")
    epochs_to_check_loss = config.train_params.get("epochs_to_check_loss")
    batch_accumulation = config.train_params.get("batch_accumulation")
    # Save train params in log file
    save_model_log(
        log_dir=model_folder_path,
        data_dictionary={
            "device": device,
            "batch_accumulation": batch_accumulation,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "loss_function": loss_function,
            "activation_function": activation_function,
            "counter_for_early_stop": counter_for_early_stop,
            "epochs_to_check_loss": epochs_to_check_loss,
            "batch_size": batch_size,
        },
    )

    model_path = train_model(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_function,
        device=device,
        max_epochs=max_epochs,
        train_dataloader=train_loader,
        valid_loader=valid_loader,
        counter_for_early_stop_threshold=counter_for_early_stop,
        epochs_to_check_loss=epochs_to_check_loss,
        batch_accumulation=batch_accumulation,
        results_path=model_folder_path,
    )
    save_model_log(log_dir=model_folder_path, data_dictionary={})

    accuracy, auroc, auprc = evaluate_model(
        model=model,
        dataloader=test_loader,
        activation_function=activation_function,
        device=device,
    )
    data_dict = {
        "path": model_path,
        "accuracy": accuracy,
        "auroc": auroc,
        "auprc": auprc,
    }
    results_csv_path = os.path.join(config.paths.get("results_path"), "results.csv")
    save_data_to_csv(data_dictionary=data_dict, csv_file_path=results_csv_path)
