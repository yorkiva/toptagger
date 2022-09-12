import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

class PFNDataset(Dataset):
    def __init__(self, file_path):
        
        df = pd.read_parquet(file_path, engine='fastparquet')
        df['mask'] = df['part_dphi'].apply(np.ones_like)
        self.data = df[['part_energy', 'part_px', 'part_py', 'part_pz', 'mask', 'jet_nparticles']].to_numpy()
        self.labels = np.expand_dims(df['label'].to_numpy(), axis=0)
        #Output is prob_isQCD, prob_isSignal
        self.labels = np.append(1-self.labels, self.labels, 1)
        #padding input if it has less than 200 constituents
        #df['jet_nparticles'] = 200 - df['jet_nparticles']
        #self.zeroes = np.expand_dims(df['jet_nparticles'].apply(np.zeros).to_numpy(), axis=0)
        #print(self.zeroes[0].shape)
        #print(self.data[0,0])
        #for i in range(5):
        #    self.data[:,i] = np.append(np.expand_dims(self.data[:,i],axis=0),self.zeroes, axis=0)
        for row in tqdm(self.data):
            if row[5] < 200:
                row[0] = np.append(row[0], np.zeros(200-int(row[5])))
                row[1] = np.append(row[1], np.zeros(200-int(row[5])))
                row[2] = np.append(row[2], np.zeros(200-int(row[5])))
                row[3] = np.append(row[3], np.zeros(200-int(row[5])))
                row[4] = np.append(row[4], np.zeros(200-int(row[5])))
            row = np.stack(row[0:5])
            
        #self.data = np.delete(self.data, 5, 1)
        
    def __len__(self):
        return len(self.data)
    
    def columns(self):
        return ['part_energy', 'part_px', 'part_py', 'part_pz', 'mask']

    def __getitem__(self, idx):
        label = self.labels[idx]
        item = np.stack(self.data[idx][0:4])
        mask = self.data[idx][4]
        return item, mask, label
    
if __name__ == "__main__":

    mydataset = PFNDataset('../../datasets/TopLandscape/val_file.parquet')
    print(len(mydataset))
    print(mydataset.columns())
    trainloader = DataLoader(mydataset, batch_size=1000, shuffle=True, num_workers=40, pin_memory=True, persistent_workers=True)
    for i,m,l in trainloader:
        #print(i)
        #print(m)
        #print(l)
        print(i.shape)
        print(m.shape)
        print(l.shape)
        break
        
        #python pfn_dataset.py