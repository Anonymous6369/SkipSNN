from torch import nn
import torch


class SkipNet(nn.Module):
    
    def __init__(self, in_planes, out_planes):
        super(SkipNet, self).__init__()
        
        self.fc = nn.Linear(in_planes, out_planes)
        
    def forward(self, x):
        
        b = self.fc(x)
        
        return torch.sigmoid(b.view(x.size(0),1))


