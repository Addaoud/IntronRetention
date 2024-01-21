import numpy as np
import pandas as pd
import torch
import os
from fastprogress import progress_bar
from torch.utils.data import DataLoader
import seaborn as sns
from prettytable import PrettyTable
import argparse

# import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests

sns.set(font_scale=1.8)
from src.attribution_utils import IGDataset, extract_seq, get_motif, mat_product
from src.utils import hot_encode_sequence
from src.networks import generate_FSei
from src.seed import set_seed

set_seed()
from typing import Optional


def load_jaspar_database(jaspar_db_path: Optional[str] = "jaspar.meme.txt"):
    tfs_data = open(jaspar_db_path).readlines()
    motifs = dict()
    matrix_data = False
    key = None
    binding_site_matrix = []
    for line in tfs_data:
        if line.startswith("MOTIF"):
            key = " ".join(line.strip().split(" ")[1:])
            binding_site_matrix = []
        if line.startswith("URL"):
            matrix_data = False
            if len(binding_site_matrix) != 0:
                motifs[key] = np.stack(binding_site_matrix, axis=1)
                binding_site_matrix = []
        if matrix_data:
            vector = np.array(
                [n.strip() for n in line.strip().split("  ")], dtype=float
            )
            binding_site_matrix.append(vector)
            # motif += bp_dict[np.argmax(vector)]
        if line.startswith("letter"):
            matrix_data = True
    return motifs


def get_back_frequence():
    df_path = "data/Labelled_Data_IR_iDiffIR_corrected"
    fa_file = "data/data.fa"
    df_path = df_path.split(".")[0]  # just in case the user provide extension
    df_all = pd.read_csv(df_path + ".txt", delimiter="\t", header=None)
    df_seq = pd.read_csv(fa_file, header=None)
    strand = df_seq[0][0][-3:]  # can be (+) or (.)
    df_all["header"] = df_all.apply(
        lambda x: ">" + x[0] + ":" + str(x[1]) + "-" + str(x[2]) + strand, axis=1
    )

    df_seq_all = pd.concat(
        [df_seq[::2].reset_index(drop=True), df_seq[1::2].reset_index(drop=True)],
        axis=1,
        sort=False,
    )
    df_seq_all.columns = ["header", "sequence"]
    df_seq_all["sequence"] = df_seq_all["sequence"].apply(lambda x: x.upper())

    df_all.rename(columns={7: "label"}, inplace=True)

    df_final = pd.merge(
        df_seq_all[["header", "sequence"]],
        df_all[["header", "label"]],
        on="header",
        how="inner",
    )
    df_final.drop_duplicates(inplace=True)

    DNAalphabet = {"A": 0, "C": 0, "G": 0, "T": 0}
    for _, row in df_final.iterrows():
        for nt in DNAalphabet.keys():
            DNAalphabet[nt] = DNAalphabet[nt] + row.sequence.count(nt)
    sum_nt = sum(DNAalphabet.values())
    for nt in DNAalphabet.keys():
        DNAalphabet[nt] = DNAalphabet[nt] / sum_nt
    back_freq = np.array(list(DNAalphabet.values())).reshape(4, 1)
    return back_freq, DNAalphabet


def get_IGdata():
    test_sampler = np.loadtxt("data/test_indices.txt", dtype=int)
    valid_sampler = np.loadtxt("data/valid_indices.txt", dtype=int)
    sampler = np.concatenate((test_sampler, valid_sampler))
    df_path = "data/Labelled_Data_IR_iDiffIR_corrected"
    fa_file = "data/data.fa"
    IG_dataset_0 = IGDataset(
        df_path=df_path,
        fa_file=fa_file,
        sampler=sampler,
        relevant_target=0,
    )
    IG_loader_0 = DataLoader(IG_dataset_0, batch_size=32)
    IG_dataset_1 = IGDataset(
        df_path=df_path,
        fa_file=fa_file,
        sampler=sampler,
        relevant_target=1,
    )
    IG_loader_1 = DataLoader(IG_dataset_1, batch_size=32)
    return IG_loader_0, IG_loader_1


def parse_arguments(parser):
    parser.add_argument("-m", "--model_path", type=str, help="Existing model path")
    args = parser.parse_args()
    return args


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate FSei model")
    args = parse_arguments(parser)
    model_path = args.model_path
    Udir_path = os.path.dirname(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = generate_FSei(new_model=False, model_path=model_path).to(device)
    IG_window_size = 48
    IG_threshhold = 0.8
    motifs = load_jaspar_database()
    back_freq, DNAalphabet = get_back_frequence()
    IG_loader_0, IG_loader_1 = get_IGdata()
    print("Selecting sequence associated with non-IR")
    seqs_0, IGs_0, Scores_0, n_unique_sequences_0 = extract_seq(
        model, IG_loader_0, IG_window_size, IG_threshhold, 0, device
    )
    binding_sites_0 = {}
    hot_encode_seqs_0 = []
    for seq in seqs_0:
        hot_encode_seqs_0.append(hot_encode_sequence(seq))
    print("Counting motifs sites")
    for tf in progress_bar(motifs.keys()):
        if IG_window_size >= motifs[tf].shape[1]:
            motif = motifs[tf]
            pseudo_motif = np.where(motif == 0, 0.001, motif)
            log_odd_motif = np.log(np.divide(pseudo_motif, back_freq))
            max_motif_score = np.sum(np.max(log_odd_motif), axis=0)
            for hot_encode in hot_encode_seqs_0:
                if mat_product(
                    log_odd_motif, hot_encode, threshold=0.8 * max_motif_score
                ):
                    binding_sites_0[tf] = binding_sites_0.get(tf, 0) + 1
    binding_sites_sorted_0 = sorted(
        binding_sites_0.items(), key=lambda x: x[1], reverse=True
    )
    converted_dict_0 = dict(binding_sites_sorted_0)
    print("Selecting sequence associated IR")
    seqs_1, IGs_1, Scores_1, n_unique_sequences_1 = extract_seq(
        model, IG_loader_1, IG_window_size, IG_threshhold, 1, device
    )
    binding_sites_1 = {}
    hot_encode_seqs_1 = []
    for seq in seqs_1:
        hot_encode_seqs_1.append(hot_encode_sequence(seq))
    print("Counting motifs sites")
    for tf in progress_bar(motifs.keys()):
        if IG_window_size >= motifs[tf].shape[1]:
            motif = motifs[tf]
            pseudo_motif = np.where(motif == 0, 0.001, motif)
            log_odd_motif = np.log(np.divide(pseudo_motif, back_freq))
            max_motif_score = np.sum(np.max(log_odd_motif), axis=0)
            for hot_encode in hot_encode_seqs_1:
                if mat_product(
                    log_odd_motif, hot_encode, threshold=0.7 * max_motif_score
                ):
                    binding_sites_1[tf] = binding_sites_1.get(tf, 0) + 1

    binding_sites_sorted_1 = sorted(
        binding_sites_1.items(), key=lambda x: x[1], reverse=True
    )
    converted_dict_1 = dict(binding_sites_sorted_1)
    prob_diff = dict()
    for tf in list(set(converted_dict_0.keys()) | set(converted_dict_1.keys())):
        prob_diff[tf] = int(converted_dict_1.get(tf, 0)) / len(seqs_1) - int(
            converted_dict_0.get(tf, 0)
        ) / len(seqs_0)
    prob_diff = dict(sorted(prob_diff.items(), key=lambda item: item[1], reverse=True))
    pvalue_dict = dict()
    for idx, tf in enumerate(list(prob_diff.keys())):
        # Define the counts and sample sizes for two groups
        count_non_IR = int(
            converted_dict_0.get(tf, 0)
        )  # Number of successes in group 1
        count_IR = int(converted_dict_1.get(tf, 0))  # Number of successes in group 2
        z_score, p_value = proportions_ztest(
            [count_non_IR, count_IR], [len(seqs_0), len(seqs_1)]
        )
        pvalue_dict[tf] = p_value
    adjusted_pvalues = multipletests(list(pvalue_dict.values()), method="bonferroni")[1]
    table = PrettyTable(["rank", "TF", "motif", "non-IR", "IR", "diff prob", "pvalue"])
    for idx, tf in enumerate(list(prob_diff.keys())):
        table.add_row(
            [
                idx + 1,
                tf,
                get_motif(motifs[tf]),
                converted_dict_0.get(tf, 0),
                converted_dict_1.get(tf, 0),
                round(prob_diff.get(tf), 3),
                adjusted_pvalues[idx],
            ]
        )
    results_file = os.path.join(Udir_path, "nsites.txt")
    with open(results_file, "w", encoding="utf-8") as file:
        file.write(
            f"{len(IG_loader_0.dataset)} total sequences associated with non-IR \n"
        )
        file.write(f"{n_unique_sequences_0} sequences are selected by IG\n")
        file.write(f"{len(seqs_0)} hot spots are selected\n")
        file.write(f"{len(IG_loader_1.dataset)} total sequences associated with IR\n")
        file.write(f"{n_unique_sequences_1} sequences are selected by IG\n")
        file.write(f"{len(seqs_1)} hot spots are selected\n")
        file.write(f"Background probabilities are: {DNAalphabet}\n")
        file.write(table.get_string())


if __name__ == "__main__":
    main()
