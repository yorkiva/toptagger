import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from multidataset import MultiDataset
from multibodymodel import Net as Model
import numpy as np
from utils import seed_everything
seed_everything(42)


#Parameters to change
use_jet_mass = True
N = 8

#Training and validation paths
train_path = '../../datasets/n-subjettiness_data/train_all.npy'
val_path = '../../datasets/n-subjettiness_data/val_all.npy'

#Loading training and validation datasets
train_set = MultiDataset(train_path, N, use_jet_mass)
val_set = MultiDataset(val_path, N, use_jet_mass)

trainloader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=50, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_set, batch_size=512, shuffle=True, num_workers=50, pin_memory=True, persistent_workers=True)

features = 3*N
if use_jet_mass:
    features+=2
else:
    features+=1
# model
model = Model(features).double().cuda()

# loss func and opt
crit = torch.nn.BCELoss()
opt = torch.optim.Adam(model.parameters(),  lr=0.001)

prev_val_loss = 1000
patience=10
no_change=0
for epoch in range(70):
    print('Epoch ' + str(epoch))
    val_loss_total = 0
    #train loop
    model.train()
    for x,y in tqdm(trainloader):
        opt.zero_grad()
        x = x.cuda()
        y = y.cuda()
        
        pred = model(x)
        loss = crit(pred, y)        
        loss.backward()
        opt.step()
    
    #Early stopping after at least 20 epochs
    model.eval()
    with torch.no_grad():
        for x,y in tqdm(val_loader):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            loss = crit(pred, y)
            val_loss_total += loss.item()/len(x)
    print('Previous Validation Loss: ' + str(prev_val_loss))
    print('Current Validation Loss: ' + str(val_loss_total))
    if prev_val_loss <= val_loss_total and epoch >= 20:
        no_change+=1
        print('Validation Loss Has stayed the same or increased, will stop in ' + str(patience-no_change) + 
              ' epochs if this continues')
        if no_change==patience:
            print('Stopping training')
            break
    elif prev_val_loss > val_loss_total:
        no_change=0
        print('Saving best model')
        torch.save(model.state_dict(), 'models/MultiBody' + str(N) + '-Subjettiness_mass' +str(use_jet_mass)+'_best')
    prev_val_loss = val_loss_total

print('Saving last model')
torch.save(model.state_dict(), 'models/MultiBody' + str(N) + '-Subjettiness_mass' +str(use_jet_mass)+'_last')
        