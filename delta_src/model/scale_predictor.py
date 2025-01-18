import torch
import numpy as np
from math import log,exp


class ScalePredictor():
    """
    A class to predict the time scale from the input
    """
    def __init__(self,datatype="xor"):

        self.datatype=datatype
        self.data_trj=torch.Tensor(np.array([]))


    def predict_scale(self,data:torch.Tensor):
        """
        1 step's worth of data is used as input
        :param data: [batch x x_dim]
        """

        scale=1
        if self.datatype=="xor":
            scale=self.__predict_xor(data)
        if self.datatype=="gesture":
            scale=self.__predict_gesture(data)

        return scale


    def __predict_xor(self,data:torch.Tensor):
        """
        Data for one step of xor is entered
        :param data: [batch x xdim]
        """

        #>> Coefficients and window when performing linear regression with scale and firing rate >>>>>>>>
        window_size=120
        slope,intercept=-1.0437631068421338,-0.6790105922709921 
        #<< Coefficients and window when performing linear regression with scale and firing rate <<<<<<<<
        
        self.data_trj=torch.cat([self.data_trj.to(data.device),data.unsqueeze(1)],dim=1)
        if self.data_trj.shape[1]>window_size:
            self.data_trj=self.data_trj[:,(self.data_trj.shape[1]-window_size):] #If it gets too long, cut it
        
        scale_log=log(self.firing_rate+1e-10)*slope + intercept
        scale=exp(scale_log)

        return scale
    

    def __predict_gesture(self,data:torch.Tensor):
        """
        Tested. Behaves as expected.
        Contains data for one gesture step.
        :param data: [batch x channel x w x h]
        """

        #>> Linear regression parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        window_size=100
        # slope,intercept=-1.3015304644834496,-4.641658840504729  #Approximated timescale from 1 to 20
        slope,intercept=-1.9131595865891335,-7.088173872389619  #Timescale approximated as 0.5, 1, 2
        #<< Linear regression parameters <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.data_trj=torch.cat([self.data_trj.to(data.device),data.unsqueeze(1)],dim=1)
        if self.data_trj.shape[1]>window_size:
            self.data_trj=self.data_trj[:,(self.data_trj.shape[1]-window_size):] #If it gets too long, cut it
        
        scale_log=log(self.firing_rate+1e-10)*slope + intercept
        scale=exp(scale_log)

        return scale

    def reset_trj(self):
        self.data_trj=torch.Tensor(np.array([]))
    
    @property
    def firing_rate(self):
        fr=torch.mean(self.data_trj,dim=1) #Average over time
        fr=torch.mean(fr,dim=tuple([i+1 for i in range(fr.ndim-1)])) #Average in space
        fr=torch.mean(fr) #Average across batches
        return fr.item()