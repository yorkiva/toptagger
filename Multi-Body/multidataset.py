import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

class MultiDataset(Dataset):
    def __init__(self, file_path, N=8, use_jet_mass=True):
        total_data = np.load(file_path)
        
        self.data = []
        #create labels
        #probabilites are (background, signal)
        sig_col = total_data[:, 26][...,None]
        bkg_col = abs(1-total_data[:, 26])[..., None]
        self.labels = np.append(bkg_col, sig_col, 1)
        
        #deletes labels from total_data
        tmp_data = np.delete(total_data, 26, 1)
        end = 25
        if not use_jet_mass:
            tmp_data = np.delete(tmp_data, 25, 1)
            end = 24
        
        self.data = tmp_data[:,0:3*N]
        self.data = np.append(self.data, tmp_data[:,24:end+1], 1)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.labels[idx]
        item = self.data[idx]
        return item, label
    
if __name__ == "__main__":

    mydataset = MultiDataset('../../datasets/n-subjettiness_data/test_all.npy')
    print(len(mydataset))
    
    for x,y in tqdm(mydataset):
        print(x.shape)
        print(y.shape)
        break
        
        #python dataloader.py