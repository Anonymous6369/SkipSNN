import math
import random
import torch
from torch import nn
import torch.nn.functional as F
#from modules import SkipNet
import numpy as np
from functools import reduce


"""
initialize hyperparameters
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5
lens = 1.0
decay = 0.1
cfg_fc = [2*34*34, 800, 10]
"""
define activations of LIF
"""
class AcFun(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, thresh, lens):
        ctx.thresh = thresh
        ctx.lens = lens
        ctx.save_for_backward(input)
        return input.gt(ctx.thresh).float()
    
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #grad_cur = abs(input - ctx.thresh)<ctx.lens
        prob = ((input-ctx.thresh)/ctx.lens).sigmoid()
        grad_cur = prob*(1-prob)*1/ctx.lens
        return grad_input*grad_cur.float(), None, None
    
act_fun = AcFun.apply



def mem_update(opt, x, mem, spike, decay, thresh, lens):
    #mem = torch.clamp(mem*decay*(1.-spike) + opt(x), min=thresh-lens, max=thresh+lens)
    mem = mem*decay*(1.-spike) + opt(x) 
    act = act_fun(mem, thresh, lens)

    return mem, act


def _count_conv(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    num_zero_inputs = (input.abs() < 1e-9).sum()
    zero_weights_factor = 1 - torch.true_divide(num_zero_inputs, input.numel())
    overall_conv_flops = conv_per_position_flops * active_elements_count * zero_weights_factor.numpy()
    
    
    return overall_conv_flops

def _count_fc(module, input, output):
    input = input[0]
    output_last_dim = output.shape[-1]
    
    num_zero_inputs = (input.abs() < 1e-9).sum()
    zero_weights_factor = 1 - torch.true_divide(num_zero_inputs, input.numel())
    return int(np.prod(input.shape) * output_last_dim) * zero_weights_factor.numpy()


def lens_scheduler(lens, epoch, factor=0.9, lens_decay_epoch=10):
    """decay temperature by a factor every lens_decay_epoch"""
    if epoch > 1 and epoch % lens_decay_epoch == 0:
        lens *= factor
    
    return lens

def generate_time_tick(B, T, freq):
    freq_base = torch.zeros(B, T, 1, device=device)
    freq_base[:,np.arange(0,T,freq),0] += 1
    
    return freq_base


class FirstToSpike(nn.Module):
    """
    Parameters
    design gate a_t in the 1st layer for skip frame
    : updating rule of mem : u_(t+1) = decay * u_t * (1 - spike_t) + a_(t+1) * w * spike_(t+1)
    : compute of a_(t+1) : a_(t+1) = thresh(ahat_(t+1))
    : compute of ahat_(t+1) : ahat_(t+1) = a_t * delta_t + (1 - a_t) * (ahat_t + min(delta_t, 1-ahat_t))
    : compute of delta_t : sigmoid(w * u_t + b)
    : variable defination : a -> a_t
    """
    def __init__(self, nhid, lens, nclasses=10, lam=0.0):
        super(FirstToSpike, self).__init__()
        
        self.nclasses = nclasses
        self.lens = lens
        
        self.fc1 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc2 = nn.Linear(cfg_fc[1], cfg_fc[2])
        
        self.ctrl = nn.Linear(cfg_fc[1]+4, 1)
        self.ctrl.weight.data.uniform_(0, 1)
        self.ctrl.bias.data.uniform_(0, 1)
    
    def forward(self, input, X, idx):
        """Compute halting points and predictions"""
        for i in range(input.size(0)):
            X[i,:,:,:,int(idx[i]):int(idx[i])+50] = input[i].view(2,34,34,50)
        
        B = X.size(0)
        T = X.size(-1)

        h1_mem = h1_spike = torch.zeros(B, cfg_fc[1], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(B, cfg_fc[2], device=device)
        
        c_mem = c_spike = torch.zeros(B, 1, device=device)
        bgt = torch.ones(B, 1, device=device)
        a_log = []
        t_mark = []

        # time tick
        freq_1 = generate_time_tick(B, T, 1)
        freq_2 = generate_time_tick(B, T, 2)
        freq_10 = generate_time_tick(B, T, 10)
        freq_100 = generate_time_tick(B, T, 100)
        flops = 0
        
        
        # --- for each timestep, accumulate membrane, select a set of actions ---
        for t in range(T):
            # run Base SNN on new data at step t
            x = X[:,:,:,:,t]
            if t > 0:
                x_a = x.view(B, -1) * c_spike
                if c_spike == 1:
                    t_mark.append(t)
            else:
                x_a = x.view(B, -1)
                t_mark.append(t)
            #x_a = x.view(B, -1)
            bgt = torch.where((c_spike==1), bgt+1, bgt) #record budget over batch B*1
            a_log.append(c_spike)

            # Base : fc

            h1_mem,h1_spike = mem_update(self.fc1, x_a.float(), h1_mem, h1_spike, decay, thresh, self.lens)
            h2_mem,h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike, decay, thresh, self.lens)
            
            c_in = torch.cat((h1_spike.detach(), freq_1[:,t], freq_2[:,t], freq_10[:,t], freq_100[:,t]), dim=1)
            c_mem, c_spike = mem_update(self.ctrl, c_in, c_mem, c_spike, 0.1, 0.5, self.lens)#decay:1, thresh:0
            
            
            out = self.fc1(x_a.float())
            flops += _count_fc(self.fc1, x_a.float(), out)
            out = self.fc2(h1_spike)
            flops += _count_fc(self.fc2, h1_spike, out)
            out = self.ctrl(c_in)
            flops += _count_fc(self.ctrl, c_in, out)

            # input for controller
            h2_sumspike += h2_spike

        rate = h2_sumspike / bgt
        self.a_log = torch.stack(a_log).squeeze(1).transpose(0, 1)
        self.bgt = bgt.mean()/T
        self.B = B
        self.T = T
        self.t_mark = t_mark
        self.flops = flops
        

        return rate,X
    
    def computeLoss(self, rate, y, lam):
        MSE = torch.nn.MSELoss()
        CE = torch.nn.CrossEntropyLoss()
        # --- compute reward ---
        
        y_ = torch.zeros(self.B, self.nclasses).scatter_(1, y.view(-1, 1), 1)
        self.loss = MSE(rate, y_)
        self.penalty = lam * self.a_log.sum()/(self.B*self.T)
        #print(self.loss, self.penalty)
        
        
        return self.loss + self.penalty
            
        
    
