import pandas as pd
import argparse
import os
import seaborn as sns

sns.set_theme()
import torch
from src.utils import (
    create_path,
    save_model_log,
    save_data_to_csv,
    generate_UDir,
    split_targets,
    read_json,
    get_device,
)
from src.dataset_utils import dataLR
from src.train_utils import trainer
from src.results_utils import plot_distribution, evaluate_model
from src.networks import LogisticRegression
from src.config import LRConfig
from src.optimizers import ScheduledOptim
from src.seed import set_seed

set_seed()


def parse_arguments(parser):
    parser.add_argument("--json", type=str, help="path to the json file")
    parser.add_argument(
        "-n",
        "--new",
        action="store_true",
        help="Build a new logistic regression model",
    )
    parser.add_argument("-m", "--model_path", type=str, help="Existing model path")
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

    config = LRConfig(**read_json(json_path=args.json))
    config_dict = config.dict()
    device = get_device()
    config_dict["device"] = device

    (
        Sei_targets_list,
        TFs_list,
        TFs_indices,
        Histone_marks_list,
        Histone_marks_indices,
        Chromatin_access_list,
        Chromatin_access_indices,
    ) = split_targets(targets_file_pth="target.names")

    train_loader, valid_loader, test_loader, input_dim, output_dim = dataLR(
        config=config
    )
    model = LogisticRegression(input_dim, output_dim).to(device)
    if args.new:
        Udir = generate_UDir(path=config.results_path)
        model_folder_path = os.path.join(config.results_path, Udir)
        create_path(model_folder_path)
    else:
        model_path = args.model_path
        model_folder_path = os.path.dirname(model_path)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # prepare the optimizer
    optimizer = ScheduledOptim(config)
    optimizer(model.parameters())

    # Prepare the loss function
    if output_dim == 1:
        loss_function = torch.nn.BCEWithLogitsLoss()
        activation_function = torch.nn.Sigmoid()
    else:
        loss_function = torch.nn.CrossEntropyLoss()
        activation_function = torch.nn.Softmax(dim=1)
    config_dict["loss_function"] = loss_function
    config_dict["activation_function"] = activation_function

    if args.train:

        # Save train params in log file
        save_model_log(log_dir=model_folder_path, data_dictionary=config_dict)

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
        save_data_to_csv(data_dictionary=data_dict, csv_file_path=results_csv_path)

        # Save weights distribution
        weights = list(best_model.parameters())[0].cpu().detach().numpy()[0]
        plot_distribution(
            weights=weights,
            file_path=os.path.join(model_folder_path, "Distribution.png"),
        )

        # Save feature weights in a csv file
        for target_list in [Sei_targets_list, TFs_list, Histone_marks_list]:
            if len(target_list) == input_dim:
                break
        importance_csv_file = os.path.join(model_folder_path, "Importance_df.csv")
        df = pd.DataFrame({"Target": target_list, "Weights": weights}).sort_values(
            by="Weights", ascending=False
        )
        df.to_csv(importance_csv_file)
