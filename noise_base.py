import numpy as np
import torch
from torch.utils.data import Dataset


def CountData(input):
    count = 0
    for i, (x, y) in enumerate(input):
        count += x.size(0)
    return count

class NoiseBase(Dataset):
    def __init__(self, N, T):
        self.N = N
        self.T = T
        self.data  = self.generate()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ix):
        return self.data[ix]

    def generate(self):
        
        X = torch.zeros(self.N, self.T, 2, 34, 34)

        for i in range(self.N):
            for t in range(self.T):
                idx = np.random.randint(2,32,2)
                axis1 = np.random.randint(idx[0]-2,idx[0]+2,2)
                axis2 = np.random.randint(idx[1]-2,idx[1]+2,2)
                X[i, t, 0,axis1[0],axis2[0]] = 1
                X[i, t, 1,axis1[1],axis2[1]] = 1
                                                                                                                            
        data = torch.tensor(np.asarray(X).astype(np.float32), dtype=torch.float)
        #data = np.random.randint(2,126,(self.N, 2, self.T))

        return data
