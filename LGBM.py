import numpy as np
import argparse
import os
from bayes_opt import BayesianOptimization
import lightgbm as lgb
from sklearn import metrics
import pandas as pd
import seaborn as sns
from datetime import datetime

sns.set_theme()
from src.utils import (
    create_path,
    save_model_log,
    save_data_to_csv,
    generate_UDir,
    split_targets,
    read_json,
)
from src.config import LGBMConfig
from src.results_utils import plot_distribution
from src.seed import set_seed

set_seed()


def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, learning_rate=0.01):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y)

    def lgb_eval(
        num_leaves,
        feature_fraction,
        bagging_fraction,
        max_depth,
        min_split_gain,
        min_child_weight,
    ):
        params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "bagging_freq": 1,
            "min_child_samples": 20,
            "reg_alpha": 1,
            "reg_lambda": 1,
            "boosting": "gbdt",
            "learning_rate": learning_rate,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": -1,
            "metric": "auc",
        }
        params["num_leaves"] = int(round(num_leaves))
        params["feature_fraction"] = max(min(feature_fraction, 1), 0)
        params["bagging_fraction"] = max(min(bagging_fraction, 1), 0)
        params["max_depth"] = int(round(max_depth))
        params["min_split_gain"] = min_split_gain
        params["min_child_weight"] = min_child_weight
        cv_result = lgb.cv(
            params, train_data, nfold=3, seed=6, stratified=False, verbose_eval=200
        )
        return (np.array(cv_result["auc"])).max()

    # parameters
    lgbBO = BayesianOptimization(
        lgb_eval,
        {
            "num_leaves": (24, 45),
            "feature_fraction": (0.1, 0.9),
            "bagging_fraction": (0.8, 0.99),
            "max_depth": (5, 9),
            "min_split_gain": (0.001, 0.1),
            "min_child_weight": (5, 50),
        },
        random_state=0,
    )
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    return (max(lgbBO.res, key=lambda x: x["target"]))["params"]


def evaluate_LGBM(model, data, target):
    preds = model.predict(data)
    auroc = metrics.roc_auc_score(target, preds)
    accuracy = np.sum(np.round(preds) == target) / len(target)
    precision, recall, thresholds = metrics.precision_recall_curve(target, preds)
    auprc = np.mean(metrics.auc(recall, precision))
    return (accuracy, auroc, auprc)


def parse_arguments(parser):
    parser.add_argument("--json", type=str, help="path to the json file")
    parser.add_argument(
        "-o",
        "--optimize",
        action="store_true",
        help="Do an optimization approach before training the model",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate LGBM model")
    args = parse_arguments(parser)
    assert (
        args.json != None
    ), "Please specify the path to the json file with --json json_path"
    config = LGBMConfig(**read_json(json_path=args.json))
    config_dict = config.dict()

    (
        Sei_targets_list,
        TFs_list,
        TFs_indices,
        Histone_marks_list,
        Histone_marks_indices,
        Chromatin_access_list,
        Chromatin_access_indices,
    ) = split_targets(targets_file_pth="target.names")
    print("Loading data")
    train_data = np.load(config.train_data)
    train_target = np.load(config.train_target)
    valid_data = np.load(config.valid_data)
    valid_target = np.load(config.valid_target)
    test_data = np.load(config.test_data)
    test_target = np.load(config.test_target)
    print("Data loaded")
    if args.optimize:
        params = bayes_parameter_opt_lgb(
            train_data, train_target, init_round=10, opt_round=100, learning_rate=0.01
        )
    else:
        params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.01,
            "max_depth": -1,
            "num_leaves": 50,
            "bagging_freq": 1,
            "early_stopping_round": config.early_stopping_round,
            "is_unbalance": config.imbalanced_data,
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
        }

    n_estimators = config.n_estimators
    # laoding data
    print("preparing data")
    lgb_train = lgb.Dataset(train_data, train_target)
    lgb_valid = lgb.Dataset(valid_data, valid_target)
    print("data prepared")
    Udir = generate_UDir(path=config.results_path)
    model_folder_path = os.path.join(config.results_path, Udir)
    create_path(model_folder_path)
    save_model_log(log_dir=model_folder_path, data_dictionary=params)
    start_time = datetime.now()
    print("start train")
    model = lgb.train(params, lgb_train, config.n_estimators, lgb_valid)
    end_time = datetime.now()
    model.save_model(os.path.join(model_folder_path, "lgbm.txt"))
    data_dict = dict()
    data_dict["started training on"] = start_time
    data_dict["finished training on"] = end_time
    save_model_log(log_dir=model_folder_path, data_dictionary=data_dict)
    save_model_log(log_dir=model_folder_path, data_dictionary={})
    accuracy, auroc, auprc = evaluate_LGBM(
        model=model, data=test_data, target=test_target
    )
    data_dict = {"UID": Udir, "accuracy": accuracy, "auroc": auroc, "auprc": auprc}
    results_csv_path = os.path.join(model_folder_path, "results.csv")
    save_data_to_csv(data_dictionary=data_dict, csv_file_path=results_csv_path)

    importances = model.feature_importance()
    # Save weights distribution
    plot_distribution(
        weights=importances,
        file_path=os.path.join(model_folder_path, "Distribution.png"),
    )

    # Save feature weights in a csv file
    for target_list in [Sei_targets_list, TFs_list, Histone_marks_list]:
        if len(target_list) == train_data.shape[1]:
            break
    importance_csv_file = os.path.join(model_folder_path, "Importance_df.csv")
    df = pd.DataFrame({"Target": target_list, "Importance": importances}).sort_values(
        by="Importance", ascending=False
    )
    df.to_csv(importance_csv_file)
