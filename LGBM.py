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


def bayes_opt_lgb(train_data, init_round=15, opt_round=20):

    def lgb_eval(
        n_estimators,
        num_leaves,
        max_depth,
        learning_rate,
        reg_alpha,
        reg_lambda,
        feature_fraction,
        bagging_fraction,
    ):
        params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "is_unbalance": True,
            "bagging_freq": 1,
            "device": "gpu",
            "gpu_device_id": 0,
            "min_split_gain ": 0.001,
            "verbosity": -1,
        }
        params["num_leaves"] = int(round(num_leaves))
        params["learning_rate"] = float(learning_rate)
        params["max_depth"] = int(round(max_depth))
        params["reg_alpha"] = float(reg_alpha)
        params["reg_lambda"] = float(reg_lambda)
        params["feature_fraction"] = max(min(feature_fraction, 1), 0)
        params["bagging_fraction"] = max(min(bagging_fraction, 1), 0)
        cv_result = lgb.cv(
            params,
            train_data,
            num_boost_round=int(round(n_estimators)),
            nfold=3,
            seed=42,
            stratified=True,
            shuffle=True,
            metrics=["auc"],
        )
        return (np.array(cv_result["valid auc-mean"])).max()

    # parameters
    lgbBO = BayesianOptimization(
        lgb_eval,
        {
            "n_estimators": (100, 1000),
            "num_leaves": (20, 50),
            "max_depth": (5, 25),
            "learning_rate": (0.001, 0.1),
            "reg_alpha": (0.0, 1.0),
            "reg_lambda": (0.0, 1.0),
            "feature_fraction": (0.5, 1.0),
            "bagging_fraction": (0.5, 1.0),
        },
        random_state=42,
    )
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    return {
        key: (
            int(round(value))
            if key in ["max_depth", "n_estimators", "num_leaves"]
            else value
        )
        for key, value in lgbBO.max["params"].items()
    }


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
    print("Loading the data")
    train_data = np.load(config.train_data)
    train_target = np.load(config.train_target)
    valid_data = np.load(config.valid_data)
    valid_target = np.load(config.valid_target)
    test_data = np.load(config.test_data)
    test_target = np.load(config.test_target)
    lgb_train = lgb.Dataset(train_data, train_target)
    lgb_valid = lgb.Dataset(valid_data, valid_target)
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "is_unbalance": True,
        "bagging_freq": 1,
        "device": "gpu",
        "gpu_device_id": 0,
        "verbosity": -1,
        "min_split_gain ": 0.001,
        "early_stopping_round": config.early_stopping_round,
    }
    if args.optimize:
        print("Starting Bayesian optimization")
        params.update(bayes_opt_lgb(lgb_train, init_round=30, opt_round=20))
    else:
        params.update(
            {
                "n_estimators": config.n_estimators,
                "num_leaves": config.num_leaves,
                "max_depth": config.max_depth,
                "learning_rate": config.learning_rate,
                "reg_alpha": config.reg_alpha,
                "reg_lambda": config.reg_lambda,
                "feature_fraction": config.feature_fraction,
                "bagging_fraction": config.bagging_fraction,
            }
        )

    # laoding data
    Udir = generate_UDir(path=config.results_path)
    model_folder_path = os.path.join(config.results_path, Udir)
    create_path(model_folder_path)
    config_dict.update(params)
    save_model_log(log_dir=model_folder_path, data_dictionary=config_dict)
    print("Starting training")
    start_time = datetime.now()
    model = lgb.train(params, lgb_train, valid_sets=lgb_valid)
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
    results_csv_path = os.path.join(config.results_path, "results.csv")
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
