import torch
import torch.nn as nn
from snntorch import surrogate
from math import log


class LIF(nn.Module):
    def __init__(self,in_size:tuple,dt,init_tau=0.5,min_tau=0.1,threshold=1.0,vrest=0,reset_mechanism="zero",spike_grad=surrogate.fast_sigmoid(),output=False,is_train_tau=True):
        """
        :param in_size: Current input size

        :param dt: ⊿t when the LIF model is made into a difference equation. If the original input is a spike time series, ⊿t is the same as the original data.

        :param init_tau: Initial value of membrane potential time constant τ

        :param threshold: Firing threshold

        :param vrest: Resting membrane potential.

        :param reset_mechanism: Specifies the method for resetting the membrane potential after firing

        :param spike_grad: Approximation function of firing gradient
        """
        super(LIF, self).__init__()

        self.dt=dt
        self.init_tau=init_tau
        self.min_tau=min_tau
        self.threshold=threshold
        self.vrest=vrest
        self.reset_mechanism=reset_mechanism
        self.spike_grad=spike_grad
        self.output=output

        #>> Adjustments to make tau learnable >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Reference [https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/blob/main/codes/models.py]
        self.is_train_tau=is_train_tau
        init_w=-log(1/(self.init_tau-min_tau)-1)
        if is_train_tau:
            self.w=nn.Parameter(init_w * torch.ones(size=in_size))  # Default Initialization
        elif not is_train_tau:
            self.w=(init_w * torch.ones(size=in_size))  # Default Initialization
        #<< Adjustments to make tau learnable <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.v=0.0
        self.r=1.0 #


    def forward(self,current:torch.Tensor):
        """
        :param current: Synaptic current [batch x ...]
        """


        # if torch.max(self.tau)<self.dt: #dt/tauが1を超えないようにする
        #     dtau=(self.tau<self.dt)*self.dt
        #     self.tau=self.tau-dtau

        # print(f"tau:{self.tau.shape}, v:{self.v.shape}, current:{current.shape}")
        # print(self.tau)
        # print(self.v)
        # print("--------------")
        device=current.device

        if not self.is_train_tau:
            self.w=self.w.to(device)

        # Move v and w to the same device as current if they are not already
        if isinstance(self.v,torch.Tensor):
            if self.v.device != device:
                self.v = self.v.to(device)
        if self.w.device != device:
            self.w = self.w.to(device)


        tau=self.min_tau+self.w.sigmoid() # If tau becomes too small, dt/tau will exceed 1.
        dv=self.dt/(tau) * ( -(self.v-self.vrest) + (self.r)*current ) # Increment of membrane potential v
        self.v=self.v+dv
        spike=self.__fire()
        v_tmp=self.v # Membrane potential before reset
        self.__reset_voltage(spike)

        if not self.output:
            return spike
        else:
            return spike, v_tmp


    def __fire(self):
        v_shift=self.v-self.threshold
        spike=self.spike_grad(v_shift)
        return spike
    

    def __reset_voltage(self,spike):
        if self.reset_mechanism=="zero":
            self.v=self.v*(1-spike.float())
        elif self.reset_mechanism=="subtract":
            self.v=self.v-self.threshold


    def init_voltage(self):
        if not self.v is None:
            self.v=0.0



class DynamicLIF(nn.Module):
    """
    LIF with dynamically changing time constant (TC)
    """

    def __init__(self,in_size:tuple,dt,init_tau=0.5,min_tau=0.1,threshold=1.0,vrest=0,reset_mechanism="zero",spike_grad=surrogate.fast_sigmoid(),output=False):
        """
        :param in_size: Current input size

        :param dt: ⊿t when the LIF model is made into a difference equation. If the original input is a spike time series, ⊿t is the same as the original data.

        :param init_tau: Initial value of membrane potential time constant τ

        :param threshold: Firing threshold

        :param vrest: Resting membrane potential.

        :param reset_mechanism: Specifies the method for resetting the membrane potential after firing

        :param spike_grad: Approximation function of firing gradient
        """
        super(DynamicLIF, self).__init__()

        self.dt=dt
        self.init_tau=init_tau
        self.min_tau=min_tau
        self.threshold=threshold
        self.vrest=vrest
        self.reset_mechanism=reset_mechanism
        self.spike_grad=spike_grad
        self.output=output

        #>> Adjustments to make tau learnable>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # reference: [https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/blob/main/codes/models.py]
        init_w=-log(1/(self.init_tau-min_tau)-1)
        self.w=nn.Parameter(init_w * torch.ones(size=in_size))  # Default Initialization
        #<< Adjustments to make tau learnable <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.v=0.0
        self.r=1.0 #
        self.a=1.0 #Timescale


    def forward(self,current:torch.Tensor):
        """
        :param current: Synaptic current [batch x ...]
        """


        # if torch.max(self.tau)<self.dt: #dt/tauが1を超えないようにする
        #     dtau=(self.tau<self.dt)*self.dt
        #     self.tau=self.tau-dtau

        # #shape debugging
        # try:
        #     print(f"tau:{self.w.shape}, v:{self.v.shape if not self.v==0.0 else 0}, current:{current.shape}")
        #     # print(self.w)
        #     # print(self.v)
        #     print("--------------")
        # except:
        #     pass

        device = current.device  # Get the device of the input tensor

        # Move v and w to the same device as current if they are not already
        if isinstance(self.v,torch.Tensor):
            if self.v.device != device:
                self.v = self.v.to(device)
        if self.w.device != device:
            self.w = self.w.to(device)

        tau=self.min_tau+self.w.sigmoid() # If tau becomes too small, dt/tau will exceed 1.
        dv=self.dt/(tau*self.a) * ( -(self.v-self.vrest) + (self.a*self.r)*current ) # Increment of membrane potential v
        # dv=self.dt/(tau*self.a) * ( -(self.v-self.vrest) + (1*self.r)*current ) #Do not multiply the input by a
        self.v=self.v+dv
        spike=self.__fire()
        v_tmp=self.v # Membrane potential before reset
        self.__reset_voltage(spike)

        if not self.output:
            return spike
        else:
            return spike, (current,v_tmp) # In addition to spikes, returns (synaptic current, membrane potential)


    def __fire(self):
        v_shift=self.v-self.threshold
        spike=self.spike_grad(v_shift)
        return spike
    

    def __reset_voltage(self,spike):
        if self.reset_mechanism=="zero":
            self.v=self.v*(1-spike.float())
        elif self.reset_mechanism=="subtract":
            self.v=self.v-self.threshold


    def init_voltage(self):
        self.v=0.0
