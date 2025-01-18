import torch
import torch.nn as nn
from snntorch import surrogate
from math import log

from .residual_block import ResidualBlock, ResidualLIFBlock
from .lif_model import LIF

import numpy as np
import copy
import tqdm
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader
import torch.optim as optim

import snntorch as snn
import brevitas.nn as qnn

import snntorch.functional as SF
from snntorch.functional import quant
from snntorch import spikegen
import random

from torchvision.utils import make_grid
import matplotlib.pyplot as plt


class QuantSNN(nn.Module):
    """
    SNN with dynamically changing time constant (TC)
    """

    def __init__(self,conf:dict):
        super(QuantSNN,self).__init__()

        self.in_size=conf["in-size"]
        self.hiddens = conf["hiddens"]
        self.out_size = conf["out-size"]
        self.clip_norm=conf["clip-norm"] if "clip-norm" in conf.keys() else 1.0
        self.dropout=conf["dropout"]
        self.output_mem=conf["output-membrane"]
        
        self.weight_bit_width=conf["wbw"]
        self.beta2 = torch.rand(self.out_size, dtype=torch.float32)
        
        self.dt = conf["dt"]
        self.init_tau = conf["init-tau"]
        self.min_tau=conf["min-tau"]
        self.is_train_tau=conf["train-tau"]
        self.v_threshold = conf["v-threshold"]
        self.v_rest = conf["v-rest"]
        self.reset_mechanism = conf["reset-mechanism"]
        if "fast".casefold() in conf["spike-grad"] and "sigmoid".casefold() in conf["spike-grad"]: 
            self.spike_grad = surrogate.fast_sigmoid()
        else:
            self.spike_grad = surrogate.atan() # arctan surrogate gradient function 


        modules=[]
        is_bias=True

        #>> Input Layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules+=[
            qnn.QuantLinear(self.in_size, self.hiddens[0],bias=is_bias, weight_bit_width=self.weight_bit_width),
            LIF(
                in_size=(self.hiddens[0],), dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=False,is_train_tau=self.is_train_tau
            ),
            nn.Dropout(self.dropout),
        ]
        #<< Input Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        #>> HIDDEN Layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        prev_hidden=self.hiddens[0]
        for hidden in self.hiddens[1:]:
            modules+=[
                qnn.QuantLinear(prev_hidden, hidden,bias=is_bias, weight_bit_width=self.weight_bit_width),
                LIF(
                    in_size=(hidden,), dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                    reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=False,is_train_tau=self.is_train_tau
                ),
                nn.Dropout(self.dropout),
            ]
            prev_hidden=hidden
        #<< HIDDEN Layer  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        #>> Output layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules+=[
            qnn.QuantLinear(self.hiddens[-1], self.out_size,bias=is_bias, weight_bit_width=self.weight_bit_width),
            LIF(
                in_size=(self.out_size,), dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=True,is_train_tau=self.is_train_tau
            ),
        ]
        #<< Output layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.model=nn.Sequential(*modules)


    def __init_lif(self):
        for layer in self.model:
            if isinstance(layer,LIF) or isinstance(layer,ResidualLIFBlock):
                layer.init_voltage()


    def forward(self,s:torch.Tensor):
        """
        :param s: Spike train [T x batch x ...]
        :return out_s: [T x batch x ...]
        :return out_v: [T x batch x ...]
        """

        T=s.shape[0]
        self.__init_lif()

        out_s,out_v=[],[]
        for st in s:
            st_out,vt_out=self.model(st)
            out_s.append(st_out)
            out_v.append(vt_out)

        out_s=torch.stack(out_s,dim=0)
        out_v=torch.stack(out_v,dim=0)

        if self.output_mem:
            return out_s,[],out_v
        
        elif not self.output_mem:
            return out_s

    def clip_gradients(self):
        """
        Clips gradients to prevent exploding gradients.
        
        :param max_norm: Maximum norm of the gradients.
        """
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)

def get_conv_outsize(model,in_size,in_channel):
    input_tensor = torch.randn(1, in_channel, in_size, in_size)
    with torch.no_grad():
        output = model(input_tensor)
    return output.shape



def add_csnn_block(
        in_size,in_channel,out_channel,kernel,stride,padding,is_bias,is_bn,pool_type,pool_size,dropout,
        lif_dt,lif_init_tau,lif_min_tau,lif_threshold,lif_vrest,lif_reset_mechanism,lif_spike_grad,lif_output,is_train_tau
        ):
    """
    param: in_size: width and height (assumed to be square)
    param: in_channel: channel size
    param: out_channel: size of output channel
    param: kernel: kernel size
    param: stride: stride size
    param: padding: padding size
    param: is_bias: whether to use bias
    param: is_bn: whether to use batch normalization
    param: pool_type: pooling type ("avg" or "max")
    param: dropout: dropout rate
    param: lif_dt: time step for LIF model
    param: lif_init_tau: initial time constant for LIF
    param: lif_min_tau: minimum time constant for LIF
    param: lif_threshold: firing threshold for LIF
    param: lif_vrest: resting membrane potential for LIF
    param: lif_reset_mechanism: membrane potential reset mechanism for LIF
    param: lif_spike_grad: LIF spike gradient function
    param: lif_output: Whether to return LIF output
    param: is_train_tau: Whether to train LIF train_tau
    """
    
    block=[]
    block.append(
        nn.Conv2d(
            in_channels=in_channel,out_channels=out_channel,
            kernel_size=kernel,stride=stride,padding=padding,bias=is_bias
        )
    )

    if is_bn:
        block.append(
            nn.BatchNorm2d(out_channel)
        )

    if pool_type=="avg".casefold():
        block.append(nn.AvgPool2d(pool_size))
    elif pool_type=="max".casefold():
        block.append(nn.MaxPool2d(pool_size))

    #Calculate the output size of block
    block_outsize=get_conv_outsize(nn.Sequential(*block),in_size=in_size,in_channel=in_channel) #[1(batch) x channel x h x w]

    block.append(
        LIF(
            in_size=tuple(block_outsize[1:]), beta1 = self.beta1, dt=lif_dt,
            init_tau=lif_init_tau, min_tau=lif_min_tau,
            threshold=lif_threshold, vrest=lif_vrest,
            reset_mechanism=lif_reset_mechanism, spike_grad=lif_spike_grad,
            output=lif_output,is_train_tau=is_train_tau
        )
    )

    if dropout>0:
        block.append(nn.Dropout2d(dropout))
    
    return block, block_outsize



