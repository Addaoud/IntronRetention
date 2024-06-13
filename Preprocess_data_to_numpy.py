import os
import argparse
import torch
import numpy as np
from src.utils import split_targets, get_device, create_path
from src.dataset_utils import load_datasets
from src.networks import build_SEI
from fastprogress import progress_bar


def parse_arguments(parser):
    parser.add_argument("--data", type=str, help="path to the data directory")
    parser.add_argument("--result", type=str, help="path to the results directory")
    args = parser.parse_args()
    return args


def map_seq_to_sei_output(
    model, data_loader, device, data_paths, target_paths, TF_indices, HM_indices
):
    data_TFs = []
    data_HMs = []
    data_All = []
    target_TFs = []
    target_HMs = []
    target_All = []
    with torch.no_grad():
        model.eval()
        for _, (data, target) in enumerate(progress_bar(data_loader)):
            data, target = {key: data[key].to(device) for key in data}, target
            outputs = model(**data)
            outputs_TFs = outputs[:, TF_indices]
            outputs_HMs = outputs[:, HM_indices]
            data_All.extend(outputs.detach().cpu().numpy())
            data_HMs.extend(outputs_HMs.detach().cpu().numpy())
            data_TFs.extend(outputs_TFs.detach().cpu().numpy())
            target_All.extend(target)
            target_HMs.extend(target)
            target_TFs.extend(target)
    data_TFs = np.stack(data_TFs)
    data_HMs = np.stack(data_HMs)
    data_All = np.stack(data_All)
    target_All = np.stack(target_All)
    target_HMs = np.stack(target_HMs)
    target_TFs = np.stack(target_TFs)

    for data, target, data_path, target_path in zip(
        [data_All, data_TFs, data_HMs],
        [target_All, target_TFs, target_HMs],
        data_paths,
        target_paths,
    ):
        np.save(file=data_path, arr=data)
        np.save(file=target_path, arr=target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the numpy data for the LR and LGBM models"
    )
    args = parse_arguments(parser)
    assert (args.data != None) and (
        args.result != None
    ), "Please specify the path to the data and result directory"

    device = get_device()
    data_folder = args.data
    result_folder = args.result
    create_path(path=result_folder)

    (
        Sei_targets_list,
        TFs_list,
        TFs_indices,
        Histone_marks_list,
        Histone_marks_indices,
        Chromatin_access_list,
        Chromatin_access_indices,
    ) = split_targets(targets_file_pth="target.names")

    SEI = build_SEI().to(device)
    batchSize = 216
    train_loader, valid_loader, test_loader = load_datasets(
        batchSize=batchSize,
        test_split=0.1,
        output_dir=data_folder,
        length_after_padding=4000,
        use_reverse_complement=True,
    )
    train_all_data_path = os.path.join(result_folder, "data_ALL_train.npy")
    train_all_target_path = os.path.join(result_folder, "target_ALL_train.npy")
    valid_all_data_path = os.path.join(result_folder, "data_ALL_valid.npy")
    valid_all_target_path = os.path.join(result_folder, "target_ALL_valid.npy")
    test_all_data_path = os.path.join(result_folder, "data_ALL_test.npy")
    test_all_target_path = os.path.join(result_folder, "target_ALL_test.npy")
    train_tfs_data_path = os.path.join(result_folder, "data_TF_train.npy")
    train_tfs_target_path = os.path.join(result_folder, "target_TF_train.npy")
    valid_tfs_data_path = os.path.join(result_folder, "data_TF_valid.npy")
    valid_tfs_target_path = os.path.join(result_folder, "target_TF_valid.npy")
    test_tfs_data_path = os.path.join(result_folder, "data_TF_test.npy")
    test_tfs_target_path = os.path.join(result_folder, "target_TF_test.npy")
    train_hms_data_path = os.path.join(result_folder, "data_HM_train.npy")
    train_hms_target_path = os.path.join(result_folder, "target_HM_train.npy")
    valid_hms_data_path = os.path.join(result_folder, "data_HM_valid.npy")
    valid_hms_target_path = os.path.join(result_folder, "target_HM_valid.npy")
    test_hms_data_path = os.path.join(result_folder, "data_HM_test.npy")
    test_hms_target_path = os.path.join(result_folder, "target_HM_test.npy")
    data_paths, target_paths = [
        train_all_data_path,
        train_tfs_data_path,
        train_hms_data_path,
    ], [train_all_target_path, train_tfs_target_path, train_hms_target_path]
    map_seq_to_sei_output(
        model=SEI,
        data_loader=train_loader,
        device=device,
        data_paths=data_paths,
        target_paths=target_paths,
        TF_indices=TFs_indices,
        HM_indices=Histone_marks_indices,
    )
    data_paths, target_paths = [
        valid_all_data_path,
        valid_tfs_data_path,
        valid_hms_data_path,
    ], [valid_all_target_path, valid_tfs_target_path, valid_hms_target_path]
    map_seq_to_sei_output(
        model=SEI,
        data_loader=valid_loader,
        device=device,
        data_paths=data_paths,
        target_paths=target_paths,
        TF_indices=TFs_indices,
        HM_indices=Histone_marks_indices,
    )
    data_paths, target_paths = [
        test_all_data_path,
        test_tfs_data_path,
        test_hms_data_path,
    ], [test_all_target_path, test_tfs_target_path, test_hms_target_path]
    map_seq_to_sei_output(
        model=SEI,
        data_loader=test_loader,
        device=device,
        data_paths=data_paths,
        target_paths=target_paths,
        TF_indices=TFs_indices,
        HM_indices=Histone_marks_indices,
    )
