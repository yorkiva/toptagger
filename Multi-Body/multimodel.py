import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class Net(nn.Module):
    def __init__(self, N=8, use_jet_pt=True, use_jet_mass=True):
        super().__init__()
        #num of features
        features = 3*(N-1)-1
        if use_jet_mass:
            features+=1
        if use_jet_pt:
            features+=1

        
        self.dense1 = nn.Linear(features, 200)
        self.dense2 = nn.Linear(200, 200)
        self.dense3 = nn.Linear(200, 50)
        self.dense4 = nn.Linear(50, 50)
        self.dense5 = nn.Linear(50, 2)
        
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(.1)
        self.dropout2 = nn.Dropout(.2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.dense3(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.dense4(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.dense5(x)
        x = self.softmax(x)
        return x
    
    

if __name__ == "__main__":
    features=23
    model = Net().cuda()
    x = torch.rand(2, features).cuda()
    with torch.no_grad():
        y = model(x)
        print(y)
    summary(model, (1, 26))