import numpy as np
import torch
from model_nmnist_noise import *
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from NmnistLoader import nmnistDataset
from noise_base import NoiseBase
import zipfile
import time
import os

# -- zip file if needed ---
with zipfile.ZipFile('NMNISTsmall.zip') as zip_file:
    for member in zip_file.namelist():
        if not os.path.exists('./' + member):
            zip_file.extract(member, './')

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=40):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

# --- hyperparameters ---
N_CLASSES = 10

def generate(data):
    output = []
    for i, (x,z,y) in enumerate(data,0):
        output.append(np.random.randint(0,250,x.size(0)))
        
    return output


def run(LEARNING_RATE, N_EPOCHS, SEQ_LENGTH, BASE, PATH, LENS, LAMBDA, LOC):
    
    
    # --- n-mnist dataset ---
    testingSet = nmnistDataset(datasetPath  ='NMNISTsmall/',
                            sampleFile  ='NMNISTsmall/test100.txt',
                            samplingTime=1.0,
                            sampleLength=50)
    test_loader = DataLoader(dataset=testingSet, batch_size=1, shuffle=False, num_workers=0) 
    start_test = generate(test_loader)
    dict_data = {i:(x,t,y) for i, (x,t,y) in enumerate(test_loader, 0)}
    
    # --- generate noise base ---
    testing_size = len(testingSet)
    testing_base = NoiseBase(N=testing_size, T=SEQ_LENGTH)
    test_base_loader = torch.utils.data.DataLoader(dataset=testing_base,
                                           batch_size=1)
    dict_test = {i: j for i,j in enumerate(test_base_loader)}
    
    # --- initialize the model and the optimizer ---
    model = FirstToSpike(nhid=N_CLASSES, lens=LENS, nclasses=N_CLASSES)
    #optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    lam = LAMBDA

    # --- load base model ---
    if not BASE:
        base = torch.load(PATH)
        model.load_state_dict(base['model_state_dict'])
        model.eval()
    
    start_time = time.time()
    idx = np.random.randint(0,100,1)
    X = dict_data[int(idx)][0]
    y = dict_data[int(idx)][-1]
    noise = dict_test[int(idx)]
    noise = noise.permute(0,2,3,4,1)
    
    #start_idx = start_test[int(idx)]
    start_idx = np.array([LOC])
    rate,data = model(X, noise, start_idx)
    _, predictions = torch.max(rate, dim=1)
    print('pred: {}, truth tag: {}, truth loc: {}, state updates: {}, time cost: {}'.format(predictions, y, start_idx, model.bgt, time.time()-start_time))
    print(model.flops)
        
    if predictions == y:
        return model.t_mark, start_idx, data