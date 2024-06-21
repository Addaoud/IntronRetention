import numpy as np
import pandas as pd
import torch
import os
from fastprogress import progress_bar
import seaborn as sns
from prettytable import PrettyTable
import argparse

# import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests

sns.set_theme(font_scale=1.8)
from src.attribution_utils import (
    get_IGdata,
    extract_seq,
    get_motif,
    mat_product,
    load_jaspar_database,
    load_humans_tf_database,
)

from src.utils import hot_encode_sequence, create_path, save_data_to_csv, get_device
from src.networks import build_FSei
from src.seed import set_seed

set_seed()


def parse_arguments(parser):
    parser.add_argument(
        "-i", "--integrate", action="store_true", help="Run integrated gradients"
    )
    parser.add_argument(
        "-b",
        "--bind",
        action="store_true",
        help="Compare hot regions with TF binding sites",
    )
    parser.add_argument("-w", "--window", type=int, default=32, help="IG window size")
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.7, help="IG threshold"
    )
    parser.add_argument(
        "-p", "--prediction", type=float, default=0.7, help="Prediction threshold"
    )
    parser.add_argument(
        "-d",
        "--database",
        type=str,
        default="data/jaspar.meme.txt",
        help="Jaspar database path",
    )
    parser.add_argument(
        "-f",
        "--pwm",
        type=str,
        default="data/PWMs",
        help="Humans_tfs PWMs directory",
    )
    parser.add_argument(
        "-r",
        "--result",
        type=str,
        default="IG",
        help="results_folder in existing model path",
    )
    parser.add_argument("-m", "--model_path", type=str, help="Existing model path")
    args = parser.parse_args()
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Run IG and compare hot regions with motifs in Jaspar database"
    )
    args = parse_arguments(parser)
    IG_window_size = args.window
    IG_threshhold = args.threshold
    model_path = args.model_path
    result_path = os.path.join(os.path.dirname(model_path), args.result)
    # Udir_path = os.path.dirname(model_path)
    create_path(os.path.join(result_path, "Non_IR"))
    create_path(os.path.join(result_path, "IR"))

    if "jaspar" in args.database:
        motifs = load_jaspar_database(jaspar_db_path=args.database)
    else:
        motifs = load_humans_tf_database(
            humans_tf_db_path=args.database, pwms_path=args.pwm
        )
    IG_loader_Non_IR, IG_loader_IR, back_freq, DNAalphabet = get_IGdata()

    max_motif_scores = dict()
    log_odd_motifs = dict()
    for tf in motifs.keys():
        pseudo_motif = np.where(motifs[tf] == 0, 10**-20, motifs[tf])
        log_odd_motifs[tf] = np.log(np.divide(pseudo_motif, back_freq))
    for tf in motifs.keys():
        max_motif_scores[tf] = np.sum(np.max(motifs[tf]), axis=0)

    if args.integrate:
        device = get_device()
        model = build_FSei(new_model=False, model_path=model_path).to(device)
        print("Selecting sequences associated with non-IR")
        (
            headers_Non_IR,
            positions_Non_IR,
            sequences_Non_IR,
            IGs_Non_IR,
            Scores_Non_IR,
        ) = extract_seq(
            model, IG_loader_Non_IR, IG_window_size, IG_threshhold, 0, device
        )
        print()
        print("Selecting sequences associated IR")
        headers_IR, positions_IR, sequences_IR, IGs_IR, Scores_IR = extract_seq(
            model, IG_loader_IR, IG_window_size, IG_threshhold, 1, device
        )
        print()
        np.save(os.path.join(result_path, "Non_IR", "IGs.npy"), np.array(IGs_Non_IR))
        np.save(os.path.join(result_path, "IR", "IGs.npy"), np.array(IGs_IR))
        pd.DataFrame(
            {
                "Sequence id": [
                    "sequence_" + str(i) for i in range(len(sequences_Non_IR))
                ],
                "Header": headers_Non_IR,
                "Position": positions_Non_IR,
                "Sequence": sequences_Non_IR,
                "Score": Scores_Non_IR,
            }
        ).to_csv(os.path.join(result_path, "df_non_IR.csv"), index=False)
        pd.DataFrame(
            {
                "Sequence id": ["sequence_" + str(i) for i in range(len(sequences_IR))],
                "Header": headers_IR,
                "Position": positions_IR,
                "Sequence": sequences_IR,
                "Score": Scores_IR,
            }
        ).to_csv(os.path.join(result_path, "df_IR.csv"), index=False)
    else:
        sequences_Non_IR = pd.read_csv(
            os.path.join(result_path, "df_non_IR.csv")
        ).Sequence.tolist()
        headers_Non_IR = pd.read_csv(
            os.path.join(result_path, "df_non_IR.csv")
        ).Header.tolist()
        sequences_IR = pd.read_csv(
            os.path.join(result_path, "df_IR.csv")
        ).Sequence.tolist()
        headers_IR = pd.read_csv(os.path.join(result_path, "df_IR.csv")).Header.tolist()
    if args.bind:

        binding_sites_Non_IR = dict()
        binding_sites_IR = dict()
        hot_encode_seqs_Non_IR = list()
        hot_encode_seqs_IR = list()
        messages = [
            "Counting motifs hits associated with non-IR",
            "Counting motifs hits associated with IR",
        ]
        for seqs, lis in zip(
            [sequences_Non_IR, sequences_IR],
            [hot_encode_seqs_Non_IR, hot_encode_seqs_IR],
        ):
            for seq in seqs:
                lis.append(hot_encode_sequence(seq))

        for hot_encoded_seqs, binding_sites, csv_file, message in zip(
            [hot_encode_seqs_Non_IR, hot_encode_seqs_IR],
            [binding_sites_Non_IR, binding_sites_IR],
            [
                os.path.join(result_path, "Non_IR", "hits.csv"),
                os.path.join(result_path, "IR", "hits.csv"),
            ],
            messages,
        ):
            print(message)
            # for idx, hot_encode in enumerate(progress_bar(hot_encoded_seqs)):
            # for tf in motifs.keys():
            for tf in progress_bar(motifs.keys()):
                for idx, hot_encode in enumerate(hot_encoded_seqs):
                    if IG_window_size >= motifs[tf].shape[1]:
                        if mat_product(
                            log_odd_motifs[tf],
                            hot_encode,
                            threshold=0.8 * max_motif_scores[tf],
                        ):
                            binding_sites[tf] = binding_sites.get(tf, 0) + 1
                            """save_data_to_csv(
                                data_dictionary={
                                    "Sequence": "sequence_" + str(idx),
                                    "Motif": tf,
                                },
                                csv_file_path=csv_file,
                            )"""
            binding_sites = sorted(
                binding_sites.items(), key=lambda x: x[1], reverse=True
            )
            print()

        converted_dict_Non_IR = dict(binding_sites_Non_IR)
        converted_dict_IR = dict(binding_sites_IR)
        prob_diff = dict()
        for tf in list(
            set(converted_dict_Non_IR.keys()) | set(converted_dict_IR.keys())
        ):
            prob_diff[tf] = int(converted_dict_IR.get(tf, 0)) / len(sequences_IR) - int(
                converted_dict_Non_IR.get(tf, 0)
            ) / len(sequences_Non_IR)
        prob_diff = dict(
            sorted(prob_diff.items(), key=lambda item: item[1], reverse=True)
        )
        pvalue_dict = dict()
        for idx, tf in enumerate(list(prob_diff.keys())):
            # Define the counts and sample sizes for two groups
            count_non_IR = int(
                converted_dict_Non_IR.get(tf, 0)
            )  # Number of successes in group 1
            count_IR = int(
                converted_dict_IR.get(tf, 0)
            )  # Number of successes in group 2
            z_score, p_value = proportions_ztest(
                [count_non_IR, count_IR], [len(sequences_Non_IR), len(sequences_IR)]
            )
            pvalue_dict[tf] = p_value
        adjusted_pvalues = multipletests(
            list(pvalue_dict.values()), method="bonferroni"
        )[1]
        table = PrettyTable(
            ["rank", "TF", "motif", "non-IR", "IR", "diff prob", "pvalue"]
        )
        for idx, tf in enumerate(list(prob_diff.keys())):
            table.add_row(
                [
                    idx + 1,
                    tf,
                    get_motif(motifs[tf]),
                    converted_dict_Non_IR.get(tf, 0),
                    converted_dict_IR.get(tf, 0),
                    round(prob_diff.get(tf), 3),
                    adjusted_pvalues[idx],
                ]
            )
        results_file = os.path.join(result_path, "nsites.txt")
        with open(results_file, "w", encoding="utf-8") as file:
            file.write(
                f"{len(IG_loader_Non_IR.dataset)} total sequences associated with non-IR \n"
            )
            file.write(f"{len(set(headers_Non_IR))} sequences are selected by IG\n")
            file.write(f"{len(sequences_Non_IR)} hot spots are selected\n")
            file.write(
                f"{len(IG_loader_IR.dataset)} total sequences associated with IR\n"
            )
            file.write(f"{len(set(headers_IR))} sequences are selected by IG\n")
            file.write(f"{len(sequences_IR)} hot spots are selected\n")
            file.write(f"Background probabilities are: {DNAalphabet}\n")
            file.write(table.get_string())


if __name__ == "__main__":
    main()
