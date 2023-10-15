import pandas as pd
import argparse
import os
import seaborn as sns
sns.set()
import torch
from utils import create_path, save_model_log, save_data_to_csv, generate_UDir, split_targets, read_json
from dataset_utils import dataLR
from train_utils import train_model, evaluate_model
from results_utils import plot_distribution
from networks import LogisticRegression


def parse_arguments(parser):
    parser.add_argument("--json", type=str, help="path to the json file")
    parser.add_argument(
        "-n",
        "--new",
        action="store_true",
        help="Build a new logistic regression model",
    )
    parser.add_argument(
        "-m", "--model_path", type=str, help="Existing model path"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate LR model")
    args = parse_arguments(parser)
    assert args.json != None, "Please specify the path to the json file with --json json_path"
    assert os.path.exists(
        args.json
    ), f"The path to the json file {args.json} does not exist. Please verify"
    assert (args.new == True) ^ ((args.model_path) != None
    ), "Wrong arguments. Either include -n to build a new model or specify -m model_path"

    config = read_json(json_path=args.json)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    (Sei_targets_list,
    TFs_list,
    TFs_indices,
    Histone_marks_list,
    Histone_marks_indices,
    Chromatin_access_list,
    Chromatin_access_indices
    ) = split_targets(targets_file_pth = 'target.names')
    
    train_loader,valid_loader,test_loader,input_dim,output_dim = dataLR(config=config)

    if args.new:
        model = LogisticRegression(input_dim,output_dim).to(device)
        Udir = generate_UDir(path=args.results_paths.get('results_path'))
        model_folder_path = os.path.join(args.results_paths.get('results_path'),Udir)
        create_path(model_folder_path)
    else:
        model_folder_path = os.path.dirname(args.model_path)
        model = torch.load(args.model_path).to(device)
    
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
    
    #Prepare the loss function
    if output_dim == 1:
        loss_function = torch.nn.BCEWithLogitsLoss()
        activation_function = torch.nn.Sigmoid(dim=1) 
    else:
        loss_function = torch.nn.CrossEntropyLoss()
        activation_function = torch.nn.Softmax(dim=1)
        
    #Train params
    max_epochs = args.train_params.get('max_epochs')
    counter_for_early_stop = args.train_params.get('counter_for_early_stop')
    epochs_to_check_loss = args.train_params.get('epochs_to_check_loss')
    batch_accumulation = args.train_params.get('batch_accumulation')
    imbalanced_data = config.train_params.get('imbalanced_data')
    #Save train params in log file
    save_model_log(log_dir = model_folder_path, data_dictionary = {'batch_accumulation': batch_accumulation,
                                                                   'Imbalanced data': imbalanced_data,
                                                                   'learning_rate': learning_rate,
                                                                   'counter_for_early_stop': counter_for_early_stop,
                                                                   'epochs_to_check_loss': epochs_to_check_loss})

    model_path = train_model(
                    model = model,
                    optimizer = optimizer,
                    loss_fn = loss_function,
                    device = device,
                    max_epochs = max_epochs,
                    train_dataloader = train_loader,
                    valid_loader = valid_loader,
                    counter_for_early_stop_threshold = counter_for_early_stop,
                    epochs_to_check_loss = epochs_to_check_loss,
                    batch_accumulation = batch_accumulation,
                    results_path = model_folder_path
                    )
    save_model_log(log_dir = model_folder_path, data_dictionary = {})

    accuracy, auroc, auprc = evaluate_model(model = model, dataloader = test_loader, activation_function = activation_function, device = device)
    data_dict = {'path':model_path,'accuracy':accuracy,'auroc':auroc,'auprc':auprc}
    results_csv_path = os.path.join(args.results_paths.get('results_path'),'results.csv')
    save_data_to_csv(data_dictionary = data_dict, csv_file_path = results_csv_path)

    weights = list(model.parameters())[0].cpu().detach().numpy()[0]
    #Save weights distribution
    plot_distribution(weights=weights, file_path=os.path.join(model_folder_path,'Distribution.png'))

    #Save feature weights in a csv file
    relevant_targets = config.data_paths.get("relevant_targets")
    if relevant_targets.upper() == 'TFs':
        target_list = TFs_list
    elif relevant_targets.upper() == 'HMs':
        target_list = Histone_marks_list
    else:
        target_list = Sei_targets_list
    importance_csv_file = os.path.join(model_folder_path,'Importance_df.csv')
    df = pd.DataFrame({'Target':target_list,'Weights':weights}).sort_values(by='importance',ascending=False)
    df.to_csv(importance_csv_file)