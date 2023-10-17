import os
import argparse
import torch
import numpy as np
from src.utils import split_targets
from src.dataset_utils import load_datasets
from src.networks import generate_SEI


def parse_arguments(parser):
    parser.add_argument("--dir", type=str, help="path to the data directory")
    args = parser.parse_args()
    return args

def map_seq_to_sei_output(model,data_loader,data_path,target_path,Target_indices=None):
    data_x = []
    data_y = []
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, target) in enumerate(data_loader):
            if (batch_idx%100==0) or (batch_idx==len(data_loader)-1):
                print(f"processed {batch_idx+1} out of {len(data_loader)} batches")
            data = data.to(device,dtype=torch.float)
            if Target_indices!=None:
                outputs = model(data)[:,Target_indices]
            else:
                outputs = model(data)
            data_x.extend(outputs.detach().cpu().numpy())
            data_y.extend(target)
    data_x = np.stack(data_x)
    data_y = np.stack(data_y)
    np.save(file = data_path, arr = data_x)
    np.save(file = target_path, arr = data_y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the numpy data for the LR and LGBM models")
    args = parse_arguments(parser)
    assert args.dir != None, "Please specify the path to the data directory where you want to save the numpy files"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_folder = args.dir

    (Sei_targets_list,
    TFs_list,
    TFs_indices,
    Histone_marks_list,
    Histone_marks_indices,
    Chromatin_access_list,
    Chromatin_access_indices
    ) = split_targets(targets_file_pth = 'target.names')

    SEI=generate_SEI().to(device)
    batchSize = 128
    train_loader, valid_loader, test_loader = load_datasets(batchSize=batchSize)

    train_data_path = os.path.join(data_folder,'data_ALL_train.npy')
    train_target_path = os.path.join(data_folder,'target_ALL_train.npy')
    valid_data_path = os.path.join(data_folder,'data_ALL_valid.npy')
    valid_target_path = os.path.join(data_folder,'target_ALL_valid.npy')
    test_data_path = os.path.join(data_folder,'data_ALL_test.npy')
    test_target_path = os.path.join(data_folder,'target_ALL_test.npy')
    map_seq_to_sei_output(model=SEI,data_loader=train_loader,data_path=train_data_path,target_path=train_target_path)
    map_seq_to_sei_output(model=SEI,data_loader=valid_loader,data_path=valid_data_path,target_path=valid_target_path)
    map_seq_to_sei_output(model=SEI,data_loader=test_loader,data_path=test_data_path,target_path=test_target_path)

    train_data_path = os.path.join(data_folder,'data_TF_train.npy')
    train_target_path = os.path.join(data_folder,'target_TF_train.npy')
    valid_data_path = os.path.join(data_folder,'data_TF_valid.npy')
    valid_target_path = os.path.join(data_folder,'target_TF_valid.npy')
    test_data_path = os.path.join(data_folder,'data_TF_test.npy')
    test_target_path = os.path.join(data_folder,'target_TF_test.npy')
    map_seq_to_sei_output(model=SEI,data_loader=train_loader,data_path=train_data_path,target_path=train_target_path,Target_indices=TFs_indices)
    map_seq_to_sei_output(model=SEI,data_loader=valid_loader,data_path=valid_data_path,target_path=valid_target_path,Target_indices=TFs_indices)
    map_seq_to_sei_output(model=SEI,data_loader=test_loader,data_path=test_data_path,target_path=test_target_path,Target_indices=TFs_indices)

    train_data_path = os.path.join(data_folder,'data_HM_train.npy')
    train_target_path = os.path.join(data_folder,'target_HM_train.npy')
    valid_data_path = os.path.join(data_folder,'data_HM_valid.npy')
    valid_target_path = os.path.join(data_folder,'target_HM_valid.npy')
    test_data_path = os.path.join(data_folder,'data_HM_test.npy')
    test_target_path = os.path.join(data_folder,'target_HM_test.npy')
    map_seq_to_sei_output(model=SEI,data_loader=train_loader,data_path=train_data_path,target_path=train_target_path,Target_indices=Histone_marks_indices)
    map_seq_to_sei_output(model=SEI,data_loader=valid_loader,data_path=valid_data_path,target_path=valid_target_path,Target_indices=Histone_marks_indices)
    map_seq_to_sei_output(model=SEI,data_loader=test_loader,data_path=test_data_path,target_path=test_target_path,Target_indices=Histone_marks_indices)