from typing import Dict, Any, Optional, Tuple, List
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

def create_path(path: str) -> None:
    """
    Creates path if it does not exists
    :params path:
        the path containing all directories to be created
    """
    os.makedirs(name=path, exist_ok=True)
    
def generate_UID(path: str, UID_length: int, format: Optional[str] = "") -> str:
    """
    Generates a UID of length UID_length that shouldn't exist, given the provided format, in the provided path.
    :param path:
        path where to search for directories or files with the same generated UID
    :param UID_length:
        the length of the generated UID
    :param format:
        the format of the file
    :return:
        return the UID with the format appended to it
    """
    assert 0 < UID_length < 10, "UID length should be greater than 0 and less than 10"
    counter = 0
    UID = "".join([str(random.randint(0, 9)) for i in range(UID_length)])
    while os.path.exists(os.path.join(path, UID + format)):
        counter += 1
        UID = "".join([str(random.randint(0, 9)) for i in range(UID_length)])
        if counter == 10**UID_length:
            raise (Exception("Hit UID counter. Try raising the UID length."))
    return UID + format

def generate_UDir(path: str, UID_length: Optional[int] = 6) -> str:
    """
    Generates a UID of length UID_length that shouldn't exist in the provided path.
    :param path:
        path where to search for directories with the same generated UID
    :param UID_length:
        the length of the generated UID
    :return:
        return the UID
    """
    assert 0 < UID_length < 10, "UID length should be greater than 0 and less than 10"
    return generate_UID(path=path, UID_length=UID_length)

def save_data_to_csv(data_dictionary: Dict[str, Any], csv_file_path: str) -> None:
    """
    Save data_dictionary as a new line in a csv file @ csv_file_path
    :param data_dictionary:
        a dictionary having keys as the header of the csv file and values as the new line to be stored
    :param csv_file_path:
        the path to the csv file
    """
    header = data_dictionary.keys()
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, "w") as fd:
            writer = csv.writer(fd)
            writer.writerow(header)
            writer = csv.DictWriter(fd, fieldnames=header)
            writer.writerow(data_dictionary)
    else:
        with open(csv_file_path, "a", newline="") as fd:
            writer = csv.DictWriter(fd, fieldnames=header)
            writer.writerow(data_dictionary)
            
def read_excel_csv_file(file_path: str) -> pd.DataFrame:
    """
    Read and return the dataframe at file_path.
    :param file_path:
        path to the csv/excel file
    :return:
        the file as a pandas dataframe
    """
    try:
        dataframe = pd.read_csv(file_path)
    except:
        dataframe = pd.read_excel(file_path)
    return dataframe

def save_model_log(log_dir: str, data_dictionary: Dict[str, Any]) -> None:
    """
    save the model logs in the log file
    """
    log_file_path = os.path.join(log_dir, "log_file")
    with open(log_file_path, "a") as log_file:
        if len(data_dictionary) > 0:
            for key, value in data_dictionary.items():
                print("{0}: {1}".format(key, value), file=log_file)
        else:
            print("\n", file=log_file)
            print("".join(["#" for i in range(50)]), file=log_file)
            print("\n", file=log_file)
            
def plot_loss(loss_csv_path: str, loss_path: str) -> None:
    """
    save the model loss to loss_path
    :param loss_list:
        list containing the model losses
    :param loss_path:
        path where to save the loss plot
    """
    table = read_excel_csv_file(file_path=loss_csv_path)
    mask = np.isfinite(table.valid_loss.values).tolist()
    plt.xlabel("epoch")
    plt.ylabel("loss value per epoch")
    plt.plot(
        table.epoch,
        table.train_loss,
        "b",
        linestyle="-",
        marker=".",
        label="train_loss",
    )
    plt.plot(
        table.epoch.values[mask],
        table.valid_loss.values[mask],
        "r",
        linestyle="-",
        marker=".",
        label="valid_loss",
    )
    plt.legend()
    plt.savefig(loss_path)
    plt.close()
    
def count_ambig_bps_in_sequence(DNA_sequence: str) -> int:
    """
    Returns the count of ambiguous base pairs (N,R,Y,W...) in a DNA sequence
    :param DNA_sequence:
        the DNA sequence in which to count the number of ambiguoug base pairs.
    :return:
        number of ambiguous bps in the DNA sequence
    """
    count = 0
    unambig_bps = {"A", "C", "G", "T"}
    for nucl in DNA_sequence.upper():
        if nucl not in unambig_bps:
            count += 1
    return count

def split_targets(targets_file_pth: str
                 ) -> Tuple[List[str],List[str],List[int],List[str],List[int],List[str],List[int]]:
    """
    Returns the list targets of the SEI framework, along with the list of TFs targets, non-TFs targets and their corresponding indices.
    :param targets_file_pth:
        the path to the SEI targets file
    :return:
        lists of the different modalities
    """
    Sei_targets_list = open(targets_file_pth,'r').readlines()
    TFs_list = list()
    TFs_indices = list()
    Histone_marks_list = list()
    Histone_marks_indices = list()
    Chromatin_access_list = list()
    Chromatin_access_indices = list()
    
    for idx,target in enumerate(Sei_targets_list):
        target_summary = target.strip().split('|')[1].strip()
        if target_summary.upper().startswith('CENPA') or target_summary.upper().startswith('H2A') or \
        target_summary.upper().startswith('H2B') or target_summary.upper().startswith('H3') or  target_summary.upper().startswith('H4'):
            Histone_marks_list.append(target.strip())
            Histone_marks_indices.append(idx)
        elif target_summary=='DNase' or target_summary.upper().startswith('ATAC'):
            Chromatin_access_list.append(target.strip())
            Chromatin_access_indices.append(idx)
        else:
            TFs_list.append(target.strip())
            TFs_indices.append(idx)
    return (Sei_targets_list,TFs_list,TFs_indices,Histone_marks_list,Histone_marks_indices,Chromatin_access_list,Chromatin_access_indices)
 
def hot_encode_sequence(
    sequence: str, length_after_padding: Optional[int] = 0, ambig_bp_coding_value: Optional[float] = 1.0,
) -> None:
    """
    Takes in a sequence of chars and one-hot encodes them.
    :param sequence:
        the sequence of chars.
    :param length_after_padding:
        the second dimension of the matrix after padding
    :output:
        save the one hot encoded sequence in the provided file_path
    """
    nucleotide_dict = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "U": [0, 0, 0, 1],
        "Y": [0, 0, 1, 1],
        "R": [1, 1, 0, 0],
        "W": [1, 0, 0, 1],
        "S": [0, 1, 1, 0],
        "K": [0, 1, 0, 1],
        "M": [1, 0, 1, 0],
        "D": [1, 1, 0, 1],
        "V": [1, 1, 1, 0],
        "H": [1, 0, 1, 1],
        "B": [0, 1, 1, 1],
        "N": [1, 1, 1, 1],
    }
    unambig_bases = {"A", "C", "G", "T"}
    if length_after_padding == 0:
        hot_encoded_seq = np.zeros((4, len(sequence)), dtype=np.float32)
    else:
        hot_encoded_seq = np.zeros((4, length_after_padding), dtype=np.float32)
    for i in range(len(sequence)):
        hot_encoded_seq[:, i] = nucleotide_dict.get(sequence[i], [0, 0, 0, 0])
        if sequence[i] not in unambig_bases:
            hot_encoded_seq[:, i] *= ambig_bp_coding_value / max(
                sum(nucleotide_dict.get(sequence[i], [0, 0, 0, 0])), 1
            )
    return hot_encoded_seq

def get_indices(dataset_size: int, test_split: float, output_dir: str, shuffle_data=True, seed_val=100):
    if os.path.exists(output_dir+'/valid_indices.txt') \
    and os.path.exists(output_dir+'/train_indices.txt') \
    and os.path.exists(output_dir+'/train_indices.txt'):
        print('loading indices')
        train_indices = np.loadtxt(output_dir+'/train_indices.txt', dtype=int)
        valid_indices = np.loadtxt(output_dir+'/valid_indices.txt', dtype=int)
        test_indices = np.loadtxt(output_dir+'/test_indices.txt', dtype=int)
    else:
        indices = list(range(dataset_size))
        split_val = int(np.floor(test_split*dataset_size))
        if shuffle_data:
            np.random.seed(seed_val)
            np.random.shuffle(indices)
        train_indices, test_indices, valid_indices = indices[2*split_val:], indices[:split_val], indices[split_val:2*split_val], 
        np.savetxt(output_dir+'/valid_indices.txt', valid_indices, fmt='%s')
        np.savetxt(output_dir+'/test_indices.txt', test_indices, fmt='%s')
        np.savetxt(output_dir+'/train_indices.txt', train_indices, fmt='%s')
    return train_indices, test_indices, valid_indices
