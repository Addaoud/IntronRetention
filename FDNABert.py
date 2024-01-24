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
    read_json,
)
from src.train_utils import train_model
from src.results_utils import evaluate_model
from src.dataset_utils import load_datasets
from src.networks import generate_FDNABert
from transformers import AutoTokenizer


def parse_arguments(parser):
    parser.add_argument("--json", type=str, help="path to the json file")
    parser.add_argument(
        "-n",
        "--new",
        action="store_true",
        help="Build a new model",
    )
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Train the model",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        action="store_true",
        help="Evaluate the model",
    )
    parser.add_argument("-m", "--model_path", type=str, help="Existing model path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate LR model")
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
    assert (
        args.train or args.model_path
    ), "You need to either set on the training mode with the argument '-t' or provide the model path."
    config = read_json(json_path=args.json)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = config.train_params.get("batch_size")
    lazy_loading = config.train_params.get("lazy_loading")
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        cache_dir=None,
        model_max_length=150,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    train_loader, valid_loader, test_loader = load_datasets(
        batchSize=batch_size, test_split=0.1, output_dir="data", tokenizer=tokenizer
    )

    if args.new:
        Udir = generate_UDir(path=config.paths.get("results_path"))
        model_folder_path = os.path.join(config.paths.get("results_path"), Udir)
        create_path(model_folder_path)
    else:
        model_folder_path = os.path.dirname(args.model_path)
    model = generate_FDNABert(freeze_weights=False, model_path=args.model_path).to(
        device
    )
    # prepare the optimizer
    learning_rate = config.optimizer_params.get("learning_rate", 0.001)
    weight_decay = config.optimizer_params.get("weight_decay", 0)
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
    if args.train:
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

    if args.evaluate:
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
