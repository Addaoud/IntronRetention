from typing import Optional
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data.sampler import SubsetRandomSampler
from utils import hot_encode_sequence


def get_indices(
    dataset_size: int,
    test_split: float,
    output_dir: str,
    shuffle_data=True,
    seed_val=100,
):
    """
    Return the indices of each dataset (train, validation, and test)
    """
    train_indices_path = os.path.join(output_dir, "train_indices.txt")
    valid_indices_path = os.path.join(output_dir, "valid_indices.txt")
    test_indices_path = os.path.join(output_dir, "test_indices.txt")
    if (
        os.path.exists(train_indices_path)
        and os.path.exists(valid_indices_path)
        and os.path.exists(test_indices_path)
    ):
        print("loading indices")
        train_indices = np.loadtxt(train_indices_path, dtype=int)
        valid_indices = np.loadtxt(valid_indices_path, dtype=int)
        test_indices = np.loadtxt(test_indices_path, dtype=int)
    else:
        indices = list(range(dataset_size))
        split_val = int(np.floor(test_split * dataset_size))
        if shuffle_data:
            np.random.seed(seed_val)
            np.random.shuffle(indices)
        train_indices, valid_indices, test_indices = (
            indices[2 * split_val :],
            indices[:split_val],
            indices[split_val : 2 * split_val],
        )
        np.savetxt(train_indices_path, train_indices, fmt="%s")
        np.savetxt(valid_indices_path, valid_indices, fmt="%s")
        np.savetxt(test_indices_path, test_indices, fmt="%s")
    return train_indices, valid_indices, test_indices


class datasetLR(Dataset):
    def __init__(self, data, target):
        self.data = torch.from_numpy(data)
        self.target = torch.from_numpy(target)
        self.len = len(data)

    def __getitem__(self, index):
        return self.data[index], self.target[index].float().reshape(
            -1,
        )

    def get_labels(self):
        return self.target

    def __len__(self):
        return self.len


def dataLR(config):
    train_data = np.load(config.data_paths.get("train_data"))
    train_target = np.load(config.data_paths.get("train_target"))
    valid_data = np.load(config.data_paths.get("valid_data"))
    valid_target = np.load(config.data_paths.get("valid_target"))
    test_data = np.load(config.data_paths.get("test_data"))
    test_target = np.load(config.data_paths.get("test_target"))
    input_dim = train_data.shape[1]
    output_dim = train_target.shape[1]
    batch_size = config.train_params.get("batch_size")
    imbalanced_data = config.train_params.get("imbalanced_data")
    Train_dataset = datasetLR(train_data, train_target)
    Valid_dataset = datasetLR(valid_data, valid_target)
    Test_dataset = datasetLR(test_data, test_target)
    if imbalanced_data:
        train_loader = DataLoader(
            Train_dataset,
            batch_size=batch_size,
            sampler=ImbalancedDatasetSampler(Train_dataset),
        )
    else:
        train_loader = DataLoader(Train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(Valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(Test_dataset, batch_size=batch_size)
    return train_loader, valid_loader, test_loader, input_dim, output_dim


class DatasetLoad(Dataset):
    def __init__(
        self,
        df_path: str,
        fa_file: str,
        lazyLoad: Optional[bool] = False,
        length_after_padding: Optional[int] = 0,
    ):
        self.DNAalphabet = {"A": "0", "C": "1", "G": "2", "T": "3"}
        df_path = df_path.split(".")[0]  # just in case the user provide extension
        self.df_all = pd.read_csv(df_path + ".txt", delimiter="\t", header=None)
        self.df_seq = pd.read_csv(fa_file, header=None)
        strand = self.df_seq[0][0][-3:]  # can be (+) or (.)
        self.df_all["header"] = self.df_all.apply(
            lambda x: ">" + x[0] + ":" + str(x[1]) + "-" + str(x[2]) + strand, axis=1
        )
        self.df_seq_all = pd.concat(
            [
                self.df_seq[::2].reset_index(drop=True),
                self.df_seq[1::2].reset_index(drop=True),
            ],
            axis=1,
            sort=False,
        )
        self.df_seq_all.columns = ["header", "sequence"]
        self.df_seq_all = self.df_seq_all["sequence"].apply(lambda x: x.upper())
        self.df_all.rename(columns={7: "label"}, inplace=True)
        self.df_final = pd.merge(
            self.df_seq_all["header", "sequence"],
            self.df_test["header", "label"],
            on="header",
            how="inner",
        )
        self.df_final.drop_duplicates(inplace=True)
        self.df_final = self.df_final.reset_index()
        self.Label_Tensors = torch.tensor(self.df_final["label"].tolist())
        self.lazyLoad = lazyLoad
        self.length_after_padding = length_after_padding
        if not self.lazyLoad:
            self.One_hot_Encoded_Tensors = []
            for i in range(0, self.df_final.shape[0]):
                X = self.df_final["sequence"][i]
                self.One_hot_Encoded_Tensors.append(
                    torch.tensor(
                        hot_encode_sequence(
                            sequence=X, length_after_padding=length_after_padding
                        )
                    )
                )

    def __len__(self):
        return self.df.shape[0]

    def get_all_data(self):
        return self.df_final

    def __getitem__(self, idx):
        if not self.lazyLoad:
            return self.One_hot_Encoded_Tensors[idx], self.Label_Tensors[idx].long()
        else:
            return (
                torch.tensor(
                    hot_encode_sequence(
                        sequence=self.df_final["sequence"][idx],
                        length_after_padding=self.length_after_padding,
                    )
                ),
                self.Label_Tensors[idx].long(),
            )


def load_datasets(
    batchSize: int,
    test_split: float,
    output_dir: str,
    lazyLoad: Optional[bool] = False,
    length_after_padding: Optional[int] = 0,
):
    """
    Loads and processes the data.
    """
    input_prefix = "data/Labelled_Data_IR_iDiffIR_corrected"
    fa_file = "data/data.fa"
    final_dataset = DatasetLoad(input_prefix, fa_file, lazyLoad, length_after_padding)
    train_indices, valid_indices, test_indices = get_indices(
        len(final_dataset), test_split, output_dir
    )
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = DataLoader(
        final_dataset, batch_size=batchSize, sampler=train_sampler
    )
    valid_loader = DataLoader(
        final_dataset, batch_size=batchSize, sampler=valid_sampler
    )
    test_loader = DataLoader(final_dataset, batch_size=batchSize, sampler=test_sampler)
    return train_loader, valid_loader, test_loader
