import argparse
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
from src.train_utils import trainer
from src.results_utils import evaluate_model
from src.dataset_utils import load_datasets
from src.networks import build_DNABert
from src.seed import set_seed
from transformers import AutoTokenizer
from src.config import DNABertConfig
from src.optimizers import ScheduledOptim

set_seed()


def parse_arguments(parser):
    parser.add_argument("--json", type=str, help="path to the json file")
    parser.add_argument(
        "-n",
        "--new",
        action="store_true",
        help="Use this option to build a new model",
    )
    parser.add_argument(
        "-f",
        "--freeze",
        action="store_true",
        help="Use this option to freeze DNABert-2 pretrained weights.",
    )
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Use this option to train the model",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        action="store_true",
        help="Use this option to evaluate the model",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Use this option to load an existing model from model_path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the DNABert2 model"
    )
    args = parse_arguments(parser)
    assert (
        args.json != None
    ), "Please specify the path to the json file with --json json_path"
    assert os.path.exists(
        args.json
    ), f"The path to the json file {args.json} does not exist. Please verify"
    assert (args.new == True) ^ (
        (args.model) != None
    ), "Wrong arguments. Either include -n to build a new model or specify -m model_path"
    assert (
        args.train or args.model
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
        model_folder_path = os.path.dirname(args.model)
        model_path = args.model
    model = build_DNABert(
        new_model=args.new,
        freeze_weights=args.freeze,
        model_path=args.model,
        n_genomic_features=2,
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
        # Train model
        trainer_ = trainer(
            model=model,
            loss_fn=loss_function,
            device=device,
            train_dataloader=train_loader,
            valid_loader=valid_loader,
            model_folder_path=model_folder_path,
            optimizer=optimizer,
            **config.dict(),
        )
        best_model, model_path = trainer_.train()
        save_model_log(log_dir=model_folder_path, data_dictionary={})

    if args.evaluate:
        accuracy, auroc, auprc = evaluate_model(
            model=best_model,
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
        # Save model performance
        save_data_to_csv(data_dictionary=data_dict, csv_file_path=results_csv_path)
