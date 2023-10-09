from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from captum.attr import IntegratedGradients
sns.set()

class IGDataset(Dataset):
    def __init__(self, df_path,fa_file,relevant_targets):
        self.DNAalphabet = {'A':'0', 'C':'1', 'G':'2', 'T':'3'}
        df_path = df_path.split('.')[0] #just in case the user provide extension
        self.df_all = pd.read_csv(df_path+'.txt',delimiter='\t',header=None)
        self.df_seq = pd.read_csv(fa_file,header=None)
        strand = self.df_seq[0][0][-3:] #can be (+) or (.) 
        self.df_all['header'] = self.df_all.apply(lambda x: '>'+x[0]+':'+str(x[1])+'-'+str(x[2])+strand, axis=1)
        
        self.chroms = self.df_all[0].unique()
        self.df_seq_all = pd.concat([self.df_seq[::2].reset_index(drop=True), self.df_seq[1::2].reset_index(drop=True)], axis=1, sort=False)
        self.df_seq_all.columns = ["header","sequence"]
        self.df_seq_all['sequence'] = self.df_seq_all['sequence'].apply(lambda x: x.upper())
        
        self.df_all.rename(columns={7: "label"},inplace=True)
    
        self.df_final = pd.merge(self.df_seq_all[['header',"sequence"]],self.df_all[['header','label']],on='header',how='inner')
        self.df_final.drop_duplicates(inplace=True)
        
        self.df_final = self.df_final.loc[self.df_final.label.isin(relevant_targets)]
        self.df_final = self.df_final.reset_index()
        self.One_hot_Encoded_Tensors = []
        self.Label_Tensors = torch.tensor(self.df_final['label'].tolist())
        for i in range(0,self.df_final.shape[0]): #tqdm() before
            X = self.df_final['sequence'][i]
            X = X.replace('N',list(self.DNAalphabet.keys())[random.choice([0,1,2,3])])
            X = X.replace('S',list(self.DNAalphabet.keys())[random.choice([1,2])])
            X = X.replace('W',list(self.DNAalphabet.keys())[random.choice([0,3])])
            X = X.replace('K',list(self.DNAalphabet.keys())[random.choice([2,3])])
            X = X.replace('Y',list(self.DNAalphabet.keys())[random.choice([1,3])])
            X = X.replace('R',list(self.DNAalphabet.keys())[random.choice([0,2])])
            X = X.replace('M',list(self.DNAalphabet.keys())[random.choice([0,1])])
            self.One_hot_Encoded_Tensors.append(torch.tensor(self.one_hot_encode(X)))
            
    def __len__(self):
        return self.df_final.shape[0]
    
    def one_hot_encode(self,seq):
        mapping = dict(zip("ACGT", range(4)))    
        seq2 = [mapping[i] for i in seq]
        return np.eye(4)[seq2].T.astype(np.compat.long)
    
    def __getitem__(self, idx):
        return self.df_final.sequence[idx],self.One_hot_Encoded_Tensors[idx],self.Label_Tensors[idx].long()


def motif_indices(IG_matrix, IG_window_size: int, IG_threshhold: float):
    l = IG_matrix.shape[-1]
    thresh = IG_matrix.max() * IG_threshhold
    for i in range(l - int(IG_window_size / 2)):
        if IG_matrix[:, i + int(IG_window_size / 2)].max() > thresh:
            yield (i, IG_matrix[:, i + int(IG_window_size / 2)].max())
           
def plot_filter_heat(param_matrix, file_path: str):
    param_range = abs(param_matrix).max()
    sns.set(font_scale=2)
    plt.figure(figsize=(param_matrix.shape[1], 4))
    sns.heatmap(
        param_matrix, cmap="PRGn", linewidths=0.2, vmin=-param_range, vmax=param_range
    )
    ax = plt.gca()
    ax.set_xticklabels(range(1, param_matrix.shape[1] + 1))
    ax.set_yticklabels("ACGT", rotation="horizontal")
    plt.savefig(file_path)
    plt.close()

def extract_seq(model,data_loader,window_size,IG_threshhold,relevant_target,device):
    with torch.no_grad():
        model.eval()
        seqs = []
        IGs = []
        Scores = []
        integrated_gradients = IntegratedGradients(model)
        for batch_idx, (seq , data, target) in enumerate(data_loader):
            print(f"processed {batch_idx} out of {len(data_loader)} batches")
            data = data.to(device)
            attributions_ig = integrated_gradients.attribute(data, target=relevant_target, n_steps=20)
            attributions_ig = attributions_ig.cpu().detach().numpy()
            for n in range (attributions_ig.shape[0]):
                for bps in list(
                    motif_indices(
                        attributions_ig[n, :, :],
                        IG_window_size=window_size,
                        IG_threshhold=IG_threshhold,
                    )
                ):
                    start_pos = bps[0]
                    end_pos = start_pos + window_size
                    score = bps[1]
                    seqs.append(seq[n][start_pos:end_pos])
                    IGs.append(attributions_ig[n, :, start_pos:end_pos])
                    Scores.append(score)
            if batch_idx == 50:
                break
    return (seqs,IGs,Scores)

def mat_product(mat1,mat2,threshold=0.7):
    assert (mat1.shape[1]<mat2.shape[1]), 'arg1 should be the smaller matrix'
    n = mat1.shape[1]
    for i in range (mat2.shape[1]-n):
        pr = np.multiply(mat1,mat2[:,i:i+n])
        pr /= mat1.max(axis=0)
        c = all([v.sum()>0 for v in pr.transpose()])
        pr = pr.sum() / mat1.shape[1]
        if pr>threshold and c:
            return True
    return False

def get_motif(mat):
    bp_dict = {0:'A',1:"C",2:"G",3:"T"}
    motif = ''
    for i in range(mat.shape[1]):
        motif+= bp_dict[np.argmax(mat[:,i])]
    return (motif)