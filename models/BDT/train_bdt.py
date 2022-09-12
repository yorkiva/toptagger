import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
from xgboost import XGBClassifier
import pickle
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(1, '../Multi-Body')
from multidataset import MultiDataset

#TopoDMM or Multi-Body
dataset = 'Multi-Body'
options = '_tau_x_1'
if dataset == 'TopoDNN':
    # Load inputs
    df_train = pd.read_pickle("../../datasets/topoprocessed/train.pkl")
    df_val = pd.read_pickle("../../datasets/topoprocessed/val.pkl")
    x_train = df_train.loc[:, df_train.columns != 'is_signal_new']
    y_train = df_train["is_signal_new"]
    x_val = df_val.loc[:, df_train.columns != 'is_signal_new']
    y_val = df_val["is_signal_new"]
    del df_train
    del df_val

    df_test = pd.read_pickle("../../datasets/topoprocessed/test.pkl")
    x_test = df_test.loc[:, df_test.columns != 'is_signal_new']
    y_test = df_test["is_signal_new"]
    del df_test
    
    #x_val = x_val.iloc[:, :30]
    #x_train = x_train.iloc[:, :30]
    #x_test = x_test.iloc[:, :30]
    
if dataset == 'Multi-Body':
    
    #Parameters to change
    use_jet_pt = True
    use_jet_mass = True
    tau_x_1 = True
    N = 8
    
    #Training and validation paths
    train_path = '../../datasets/n-subjettiness_data/train_all.npy'
    val_path = '../../datasets/n-subjettiness_data/val_all.npy'
    test_path = '../../datasets/n-subjettiness_data/test_all.npy'

    #Loading training and validation datasets
    train_set = MultiDataset(train_path, N, use_jet_pt, use_jet_mass, tau_x_1)
    val_set = MultiDataset(val_path, N, use_jet_pt, use_jet_mass, tau_x_1)
    test_set = MultiDataset(test_path, N, use_jet_pt, use_jet_mass, tau_x_1)
    labels = []
    if not tau_x_1:
        for i in range(N-1):
            if i != N-2:
                labels.append('tau_'+str(i+1)+'_'+str(0.5))
                labels.append('tau_'+str(i+1)+'_'+str(1))
                labels.append('tau_'+str(i+1)+'_'+str(2))
            else:
                labels.append('tau_'+str(i+1)+'_'+str(1))
                labels.append('tau_'+str(i+1)+'_'+str(2))
        if use_jet_pt:
            labels.append('jet_pt')
        if use_jet_mass:
            labels.append('jet_mass')
    else:
        for i in range(N-1):
            labels.append('tau_'+str(i+1)+'_'+str(1))
        if use_jet_pt:
            labels.append('jet_pt')
        if use_jet_mass:
            labels.append('jet_mass')

    x_train,y_train = train_set[:]
    x_val,y_val = val_set[:]
    x_test,y_test = test_set[:]    
    y_train = y_train[:,1]
    y_val = y_val[:,1]
    y_test = y_test[:,1]
    
    x_train = pd.DataFrame(x_train, columns=labels)
    y_train = pd.DataFrame(y_train, columns=['is_signal_new'])
    x_val = pd.DataFrame(x_val, columns=labels)
    y_val = pd.DataFrame(y_val, columns=['is_signal_new'])
    x_test = pd.DataFrame(x_test, columns=labels)
    y_test = pd.DataFrame(y_test, columns=['is_signal_new'])
    
    
model = XGBClassifier(verbosity=1)
print(model)
model.fit(x_train, y_train)
    
# save the model to disk
filename = 'models/bdt_model_'+dataset+options+'.sav'
pickle.dump(model, open(filename, 'wb'))


# make predictions for test data
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy: %.2f%%" % (accuracy * 100.0))


# make predictions for test data
y_pred = model.predict(x_val)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_val, predictions)
print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))
