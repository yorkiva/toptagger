import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_pfn import PFNDataset
from pfn_model import ParticleFlowNetwork as Model
import numpy as np
from utils import seed_everything,accuracy
from sklearn.metrics import accuracy_score
from torchinfo import summary
import torch.nn as nn
import wandb
init = True
seed_everything(42)


#Parameters to change
dataset = ''
epochs= 50
preprocessed=True
#used to differentiate different models
extra_name = ''

#LR Scheduler
use_lr_schedule = True
milestones=[10, 20]
gamma=0.1

#optimizer parameters
l_rate = 3e-4
opt_weight_decay = 0

#Early stopping parameters
early_stopping = True
min_epoch_early_stopping = 20
patience=10 #num_epochs if val acc doesn't change by 'tolerance' for 'patience' epochs.
tolerance=1e-5


#Loading training and validation datasets
if dataset == '':
    train_path = "../../datasets/train.h5"
    val_path = "../../datasets/val.h5"
    train_set = PFNDataset(train_path, preprocessed)
    val_set = PFNDataset(val_path, preprocessed)
    features=3

trainloader = DataLoader(train_set, batch_size=250, shuffle=True, num_workers=40, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_set, batch_size=250, shuffle=True, num_workers=40, pin_memory=True, persistent_workers=True)

# model
model = Model(features).cuda()
summary(model, ((1, 200, features), (1, 1, 200)))

# loss func and opt
crit = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(),  lr=l_rate, weight_decay=opt_weight_decay)

if use_lr_schedule:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma=gamma)
    
    
best_val_acc = 0
no_change=0
pre_val_acc = 0
for epoch in range(epochs):
    print('Epoch ' + str(epoch))
    val_loss_total = 0
    train_loss_total = 0
    train_top1_total = 0
    val_top1_total = 0
    #train loop
    model.train()
    for x,m,y in tqdm(trainloader):
        
        opt.zero_grad()
        x = x.cuda()
        m = m.cuda()
        y = y.cuda()
        
        pred = model(x,m)
        loss = crit(pred, y)
        
        train_loss_total += loss.item()
        #accuracy is determined by rounding. Any number <= 0.5 get's rounded down to 0
        #The rest get rounded up to 1
        with torch.no_grad():
            top1 = accuracy_score(pred[:,1].round().cpu(), y[:,1].cpu(), normalize=False)
            train_top1_total += top1.item()
        loss.backward()
        opt.step()
    
    #Early stopping after at least 20 epochs
    model.eval()
    with torch.no_grad():
        for x,m,y in tqdm(val_loader):
            x = x.cuda()
            m = m.cuda()
            y = y.cuda()
            pred = model(x,m)
            loss = crit(pred, y)
            
            val_loss_total += loss.item()
            #accuracy is determined by rounding. Any number <= 0.5 get's rounded down to 0
            #The rest get rounded up to 1
            with torch.no_grad():
                top1 = accuracy_score(pred[:,1].round().cpu(), y[:,1].cpu(), normalize=False)
                val_top1_total += top1.item()
    train_loss_total /= len(train_set)
    val_loss_total /= len(val_set)
    val_top1_total /= len(val_set)
    train_top1_total /= len(train_set)
    
    print('Best Validation Accuracy: ' + str(best_val_acc))
    print('Current Validation Accuracy: ' + str(val_top1_total))
    print('Current Validation Loss: ' + str(val_loss_total))
    
    if early_stopping:
        if abs(pre_val_acc - val_top1_total) < tolerance and epoch >= min_epoch_early_stopping:
            no_change+=1
            print('Validation Accuracy has not changed much, will stop in ' + str(patience-no_change) + 
                  ' epochs if this continues')
            if no_change==patience:
                print('Stopping training')
                break
        if abs(pre_val_acc - val_top1_total) >= tolerance and epoch >= min_epoch_early_stopping:
            no_change = 0
            
    if val_top1_total > best_val_acc:
        no_change=0
        print('Saving best model based on accuracy')
        torch.save(model.state_dict(), 'models/PFN'+'_best'+dataset+extra_name)
        best_val_acc = val_top1_total
        
    #pre_val_loss = val_loss_total
    pre_val_acc = val_top1_total
    
    if init:
        wandb.init('PFN')
        init = False
    wandb.log({
        "train_loss": train_loss_total,
        "val_loss": val_loss_total,
        "train_acc": train_top1_total,
        "val_acc": val_top1_total
    })
    
    if use_lr_schedule:
        scheduler.step()
    

print('Saving last model')
torch.save(model.state_dict(), 'models/PFN'+'_last'+dataset+extra_name)

