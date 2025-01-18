"""
Spike Encoder - A Gaussian function time scale that works well with speed changes
"""
import torch

class DiffEncoder():
    """
    Spikes are output according to the time variable. 
    This should produce spikes that are close to ideal for time scaling.  
    """

    def __init__(self,threshold):
        self.threshold=threshold

        self.is_init_state=False
        self.prev_x=torch.zeros(1)
        self.state=torch.zeros(1)

    def step_forward(self,xt:torch.Tensor):
        """
        :param xt: [batch x xdim]. 1 step input
        :retrun out_spike: [batch x xdim] output spike
        """

        with torch.no_grad():
            if not self.is_init_state:
                self.prev_x=xt.clone()
                self.state=torch.zeros_like(xt)
                self.is_init_state=True
            
            dx=torch.abs(xt-self.prev_x) #Take the absolute value of 1 step. Ideally, you should apply something like a low pass filter.
            self.state+=dx
            out_spike=torch.where(self.state>self.threshold,1.0,0.0) #Fires when a variable exceeds a threshold
            self.state[self.state>self.threshold]=0.0 #When it fires, set the state back to 0
            self.prev_x=xt

        return out_spike
    

    def reset_state(self):
        """
        Reset when one sequence ends
        """
        self.is_init_state=False
        self.state =torch.zeros(1)
        self.prev_x=torch.zeros(1)


import snntorch as snn
from snntorch import LIF
from torch import nn
from snntorch import surrogate

class DirectCSNNEncoder(nn.Module):
    """
    An Encoder that receives a series of values ​​and returns a spike
    """

    def __init__(self,conf:dict):
        super(DirectCSNNEncoder,self).__init__()

        self.in_size = conf["in-size"]
        self.in_channel = conf["in-channel"]
        self.pool_type = conf["pool-type"]
        
        self.output_mem=conf["output-membrane"]
        self.dt = conf["dt"]
        self.init_tau = conf["init-tau"]
        self.min_tau=conf["min-tau"]
        self.is_train_tau=conf["train-tau"]
        self.v_threshold = conf["v-threshold"]
        self.v_rest = conf["v-rest"]
        self.reset_mechanism = conf["reset-mechanism"]
        if "fast".casefold() in conf["spike-grad"] and "sigmoid".casefold() in conf["spike-grad"]: 
            self.spike_grad = surrogate.fast_sigmoid()


        modules=[]

        modules+=[
            nn.Conv2d(
                in_channels=self.in_channel,out_channels=self.in_channel,
                kernel_size=3,stride=1,padding=1,bias=True
            ),
            LIF(
                in_size=(self.in_channel,self.in_size,self.in_size), dt=self.dt,
                init_tau=self.init_tau, min_tau=self.min_tau,
                threshold=self.v_threshold, vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism, spike_grad=self.spike_grad,
                output=False,is_train_tau=self.is_train_tau
            )
        ]

        self.model=nn.Sequential(*modules)


    def __init_lif(self):
        for layer in self.model:
            if isinstance(layer,LIF):
                layer.init_voltage()

    def reset_state(self):
        self.__init_lif()

    def step_forward(self,x:torch.Tensor):
        """
        :param x: [batch x xdim...]
        """
        return self.model(x)
