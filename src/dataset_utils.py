from typing import Optional, List
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
from .utils import hot_encode_sequence
from transformers import PreTrainedTokenizer
from Bio.Seq import reverse_complement
from .seed import set_seed

set_seed()


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
        return (
            dict(input=self.data[index].float()),
            self.target[index]
            .float()
            .reshape(
                -1,
            ),
        )

    def get_labels(self):
        return self.target

    def __len__(self):
        return self.len


class DatasetLoad(Dataset):
    def __init__(
        self,
        df_final: pd.DataFrame,
        use_reverse_complement: Optional[bool] = False,
        targets_for_reverse_complement: Optional[list[int]] = [1],
        lazyLoad: Optional[bool] = False,
        length_after_padding: Optional[int] = 0,
    ):
        self.df_final = df_final.reset_index(drop=True)
        if use_reverse_complement:
            for (
                idx,
                row,
            ) in self.df_final.loc[
                self.df_final.label.isin(targets_for_reverse_complement)
            ].iterrows():
                self.df_final.loc[len(self.df_final.index)] = [
                    row.header + "_r",
                    reverse_complement(row.sequence),
                    row.label,
                ]
        self.seqs = self.df_final["sequence"].tolist()
        self.Label_Tensors = torch.tensor(self.df_final["label"].tolist())
        self.lazyLoad = lazyLoad
        self.length_after_padding = length_after_padding
        if not self.lazyLoad:
            self.data = []
            for i in range(0, self.df_final.shape[0]):
                self.data.append(
                    torch.tensor(
                        hot_encode_sequence(
                            sequence=self.seqs[i],
                            length_after_padding=length_after_padding,
                        )
                    )
                )

    def __len__(self):
        return self.df_final.shape[0]

    def get_all_data(self):
        return self.df_final

    def __getitem__(self, idx):
        if not self.lazyLoad:
            return (
                dict(input=self.data[idx].float()),
                self.Label_Tensors[idx].long(),
            )
        else:
            return (
                dict(
                    input=torch.tensor(
                        hot_encode_sequence(
                            sequence=self.seqs[idx],
                            length_after_padding=self.length_after_padding,
                        )
                    ).float()
                ),
                self.Label_Tensors[idx].long(),
            )


class DatasetBert(Dataset):
    def __init__(
        self,
        df_final: pd.DataFrame,
        use_reverse_complement: Optional[bool] = False,
        targets_for_reverse_complement: Optional[list[int]] = [1],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        lazyLoad: Optional[bool] = False,
    ):
        self.df_final = df_final.reset_index(drop=True)
        self.lazyLoad = lazyLoad
        if use_reverse_complement:
            for (
                idx,
                row,
            ) in self.df_final.loc[
                self.df_final.label.isin(targets_for_reverse_complement)
            ].iterrows():
                self.df_final.loc[len(self.df_final.index)] = [
                    row.header + "_r",
                    reverse_complement(row.sequence),
                    row.label,
                ]
        self.seqs = self.df_final["sequence"].tolist()
        self.Label_Tensors = torch.tensor(self.df_final["label"].tolist())
        self.tokenizer = tokenizer
        if not self.lazyLoad:
            self.tokenizer_output = self.tokenizer(
                self.seqs,
                return_tensors="pt",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
        self.data = self.tokenizer_output["input_ids"]
        self.attention_mask = self.tokenizer_output[
            "attention_mask"
        ]  # self.data[idx].ne(self.tokenizer.pad_token_id) with pad_token_id=3
        self.token_type_ids = self.tokenizer_output["token_type_ids"]

    def __len__(self):
        return self.df_final.shape[0]

    def get_all_data(self):
        return self.df_final

    def __getitem__(self, idx):
        if not self.lazyLoad:
            return (
                dict(
                    input_ids=self.data[idx],
                    token_type_ids=self.token_type_ids[idx],
                    attention_mask=self.attention_mask[idx],
                ),
                self.Label_Tensors[idx],
            )
        else:
            return (
                self.tokenizer(
                    self.seqs[idx],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                ),
                self.Label_Tensors[idx],
            )


def dataLR(config):
    train_data = np.load(config.train_data)
    train_target = np.load(config.train_target)
    valid_data = np.load(config.valid_data)
    valid_target = np.load(config.valid_target)
    test_data = np.load(config.test_data)
    test_target = np.load(config.test_target)
    input_dim = train_data.shape[1]
    try:
        output_dim = train_target.shape[1]
    except:
        output_dim = 1
    batch_size = config.batch_size
    imbalanced_data = config.imbalanced_data
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


def load_datasets(
    batchSize: int,
    test_split: float,
    output_dir: str,
    lazyLoad: Optional[bool] = False,
    length_after_padding: Optional[int] = 0,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    return_loader: Optional[bool] = True,
    use_reverse_complement: Optional[bool] = False,
    targets_for_reverse_complement: Optional[list[int]] = [1],
):
    """
    Loads and processes the data.
    """
    # input_prefix = "data/Labelled_Data_IR_iDiffIR_corrected"
    # fa_file = "data/data.fa"
    print("Loading indices and preparing the data")
    final_dataset = pd.read_csv("data/final_data.csv")
    train_indices, valid_indices, test_indices = get_indices(
        len(final_dataset), test_split, output_dir
    )
    if tokenizer == None:
        train_dataset = DatasetLoad(
            final_dataset.iloc[train_indices],
            use_reverse_complement,
            targets_for_reverse_complement,
            lazyLoad,
            length_after_padding,
        )
        valid_dataset = DatasetLoad(
            final_dataset.iloc[valid_indices],
            use_reverse_complement,
            targets_for_reverse_complement,
            lazyLoad,
            length_after_padding,
        )
        test_dataset = DatasetLoad(
            final_dataset.iloc[test_indices],
            use_reverse_complement,
            targets_for_reverse_complement,
            lazyLoad,
            length_after_padding,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batchSize,
            shuffle=True,
            pin_memory=True,
            num_workers=6,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batchSize,
            shuffle=True,
            pin_memory=True,
            num_workers=6,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batchSize,
            shuffle=True,
            pin_memory=True,
            num_workers=6,
        )
        return train_loader, valid_loader, test_loader
    else:
        train_dataset = DatasetBert(
            final_dataset.iloc[train_indices],
            use_reverse_complement,
            targets_for_reverse_complement,
            tokenizer,
            lazyLoad,
        )
        valid_dataset = DatasetBert(
            final_dataset.iloc[valid_indices],
            use_reverse_complement,
            targets_for_reverse_complement,
            tokenizer,
            lazyLoad,
        )
        test_dataset = DatasetBert(
            final_dataset.iloc[test_indices],
            use_reverse_complement,
            targets_for_reverse_complement,
            tokenizer,
            lazyLoad,
        )
        if return_loader:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batchSize,
                shuffle=True,
                pin_memory=True,
                num_workers=6,
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=batchSize,
                shuffle=True,
                pin_memory=True,
                num_workers=6,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batchSize,
                shuffle=True,
                pin_memory=True,
                num_workers=6,
            )
            return train_loader, valid_loader, test_loader
        else:
            return train_dataset, valid_dataset, test_dataset
