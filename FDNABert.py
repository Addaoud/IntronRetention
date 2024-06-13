import argparse
import os
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from src.utils import (
    create_path,
    save_model_log,
    save_data_to_csv,
    generate_UDir,
    read_json,
    get_device,
)
from src.train_utils import train_model
from src.results_utils import evaluate_model
from src.dataset_utils import load_datasets
from src.networks import build_FDNABert
from src.config import DNABertConfig
from transformers import AutoTokenizer
from src.optimizers import ScheduledOptim


def parse_arguments(parser):
    parser.add_argument("--json", type=str, help="path to the json file")
    parser.add_argument(
        "-n",
        "--new",
        action="store_true",
        help="Build a new model",
    )
    parser.add_argument("-m", "--model_path", type=str, help="Existing model path")
    parser.add_argument(
        "-f",
        "--freeze",
        action="store_true",
        help="Freeze DNABert pretrained weights",
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
    config = DNABertConfig(**read_json(json_path=args.json))
    config_dict = config.dict()
    device = get_device()
    config_dict["device"] = device

    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        cache_dir=None,
        model_max_length=150,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    train_loader, valid_loader, test_loader = load_datasets(
        batchSize=config.batch_size,
        test_split=0.1,
        output_dir=config.data_path,
        tokenizer=tokenizer,
        lazyLoad=config.lazy_loading,
        use_reverse_complement=config.use_reverse_complement,
        targets_for_reverse_complement=config.targets_for_reverse_complement,
    )

    if args.new:
        Udir = generate_UDir(path=config.results_path)
        model_folder_path = os.path.join(config.results_path, Udir)
        create_path(model_folder_path)
    else:
        model_folder_path = os.path.dirname(args.model_path)
    model = build_FDNABert(
        new_model=args.new, freeze_weights=args.freeze, model_path=args.model_path
    ).to(device)

    # prepare the optimizer
    optimizer = ScheduledOptim(config)
    optimizer(model.parameters())

    # Prepare the loss function
    loss_function = torch.nn.CrossEntropyLoss(reduction="mean")
    activation_function = torch.nn.Softmax(dim=1)
    config_dict["loss_function"] = loss_function
    config_dict["activation_function"] = activation_function

    if args.train:
        # Save train params in log file
        save_model_log(log_dir=model_folder_path, data_dictionary=config_dict)

        model_path = train_model(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_function,
            device=device,
            max_epochs=config.max_epochs,
            train_dataloader=train_loader,
            valid_loader=valid_loader,
            counter_for_early_stop_threshold=config.counter_for_early_stop,
            epochs_to_check_loss=config.epochs_to_check_loss,
            batch_accumulation=config.batch_accumulation,
            results_path=model_folder_path,
            n_accumulated_batches=config.n_accumulated_batches,
            use_scheduler=config.use_scheduler,
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
        results_csv_path = os.path.join(config.results_path, "results.csv")
        save_data_to_csv(data_dictionary=data_dict, csv_file_path=results_csv_path)
