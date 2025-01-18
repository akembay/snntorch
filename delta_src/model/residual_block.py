from torch import nn
from snntorch import surrogate
import torch

from .lif_model import DynamicLIF, LIF



def get_conv_outsize(model,in_size,in_channel):
    input_tensor = torch.randn(1, in_channel, in_size, in_size)
    with torch.no_grad():
        output = model(input_tensor)
    return output.shape




class ResidualBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3,stride=1,padding=1,num_block=1,bias=True,dropout=0.3):
        """
        Residual block with only CNN
        The size of the input and output does not change with this residual (the channel does change, of course)
        :param num_block: How many CNNs to stack in addition to the input CNN
        """
        super(ResidualBlock,self).__init__()

        modules=[]
        modules+=[
                nn.Conv2d(
                    in_channels=in_channel,out_channels=out_channel,
                    kernel_size=kernel,stride=stride,padding=padding,bias=bias
                ),
                nn.ReLU(inplace=False)
            ]
        for _ in range(num_block):
            if dropout>0:
                modules+=[
                    nn.Dropout(dropout,inplace=False)
                ]
            modules+=[
                    nn.Conv2d(
                        in_channels=out_channel,out_channels=out_channel,
                        kernel_size=kernel,stride=stride,padding=padding,bias=bias
                    ),
                    nn.ReLU(inplace=False)
                ]
        self.model=nn.Sequential(*modules)

        self.shortcut=nn.Sequential()
        if not in_channel==out_channel:
            self.shortcut=nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channel,out_channels=out_channel,
                        kernel_size=1,stride=1,padding=0,bias=bias
                    )
            )

    def forward(self,x):
        """
        Add residual and output
        """
        out=self.model(x)
        residual=self.shortcut(x)
        return out+residual


class ResidualLIFBlock(ResidualBlock):
    """
    Residual block using only LIF neurons
    """
    
    def __init__(
            self, in_size: tuple, in_channel, out_channel, kernel=3, stride=1, padding=1, 
            num_block=1, bias=False, dropout=0.3, dt=0.01, init_tau=0.5, min_tau=0.1, 
            threshold=1.0, vrest=0, reset_mechanism="zero", 
            spike_grad=surrogate.fast_sigmoid(), output=False, is_train_tau=True
            ):
        """
        Using only LIF neurons for computation
        """
        super(ResidualLIFBlock, self).__init__(
            in_channel, out_channel, kernel, stride, padding, num_block, bias, dropout,
        )
        
        # Calculate sizes
        h_in, w_in = in_size
        h_out = (h_in + 2 * padding - kernel) // stride + 1
        w_out = (w_in + 2 * padding - kernel) // stride + 1
        
        modules = []
        
        # Replace Conv2d with LIF
        input_size = in_channel * kernel * kernel
        modules += [
            LIF(
                in_size=(input_size, h_out, w_out), 
                out_channels=out_channel,
                dt=dt, init_tau=init_tau, min_tau=min_tau,
                threshold=threshold, vrest=vrest,
                reset_mechanism=reset_mechanism, 
                spike_grad=spike_grad,
                output=False, is_train_tau=is_train_tau
            )
        ]
        
        for _ in range(num_block):
            if dropout > 0:
                modules += [
                    LIFDropout(dropout)
                ]
            
            modules += [
                LIF(
                    in_size=(out_channel, h_out, w_out), 
                    out_channels=out_channel,
                    dt=dt, init_tau=init_tau, min_tau=min_tau,
                    threshold=threshold, vrest=vrest,
                    reset_mechanism=reset_mechanism, 
                    spike_grad=spike_grad,
                    output=False, is_train_tau=is_train_tau
                )
            ]
                
        self.model = nn.Sequential(*modules)
        
        self.shortcut = nn.Sequential()
        if not in_channel == out_channel:
            self.shortcut = LIF(
                in_size=in_size, 
                out_channels=out_channel,
                dt=dt, init_tau=init_tau, min_tau=min_tau,
                threshold=threshold, vrest=vrest,
                reset_mechanism=reset_mechanism, 
                spike_grad=spike_grad,
                output=False, is_train_tau=is_train_tau
            )
    
    def init_voltage(self):
        for layer in self.model:
            if isinstance(layer, LIF):
                layer.init_voltage()
        if isinstance(self.shortcut, LIF):
            self.shortcut.init_voltage()
                
    def forward(self, x):
        # Unfold input for LIF processing
        x_unf = F.unfold(x, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        x_unf = x_unf.view(x.shape[0], -1, *x_unf.shape[2:])
        
        identity = self.shortcut(x)
        out = self.model(x_unf)
        
        return out + identity



class ResidualLIFBlock_wrong_conv(ResidualBlock):
    """
    Residual block with activation as LIF
    """

    def __init__(
            self,in_size:tuple,in_channel,out_channel,kernel=3,stride=1,padding=1,num_block=1,bias=False,dropout=0.3,
            dt=0.01,init_tau=0.5,min_tau=0.1,threshold=1.0,vrest=0,reset_mechanism="zero",spike_grad=surrogate.fast_sigmoid(),output=False,is_train_tau=True
            ):
        """
        Output is current (= not spikes)
        """
        super(ResidualLIFBlock,self).__init__(
            in_channel,out_channel,kernel,stride,padding,num_block,bias,dropout,
        )

        modules=[]
        modules+=[
                nn.Conv2d(
                    in_channels=in_channel,out_channels=out_channel,
                    kernel_size=kernel,stride=stride,padding=padding,bias=bias
                ),
            ]
        
        for _ in range(num_block):

            #Calculate the output size of block
            module_outsize=get_conv_outsize(nn.Sequential(*modules),in_size=in_size,in_channel=in_channel) #[1(batch) x channel x h x w]
            modules.append(
                LIF(
                    in_size=tuple(module_outsize[1:]), dt=dt,
                    init_tau=init_tau, min_tau=min_tau,
                    threshold=threshold, vrest=vrest,
                    reset_mechanism=reset_mechanism, spike_grad=spike_grad,
                    output=False,is_train_tau=is_train_tau
                )
            )
            
            if dropout>0:
                modules+=[
                    nn.Dropout(dropout,inplace=False)
                ]
            modules+=[
                    nn.Conv2d(
                        in_channels=out_channel,out_channels=out_channel,
                        kernel_size=kernel,stride=stride,padding=padding,bias=bias
                    )
                ]
            
            
        self.model=nn.Sequential(*modules)

        self.shortcut=nn.Sequential()
        if not in_channel==out_channel:
            self.shortcut=nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channel,out_channels=out_channel,
                        kernel_size=1,stride=1,padding=0,bias=bias
                    )
            )


    def init_voltage(self):
        for layer in self.model:
            if isinstance(layer,LIF):
                layer.init_voltage()





class ResidualDynaLIFBlock(ResidualBlock):

    def __init__(
            self,in_size:tuple,in_channel,out_channel,kernel=3,stride=1,padding=1,num_block=1,bias=False,dropout=0.3,
            dt=0.01,init_tau=0.5,min_tau=0.1,threshold=1.0,vrest=0,reset_mechanism="zero",spike_grad=surrogate.fast_sigmoid(),output=False
            ):
        """
        Output is current (= not spikes)
        """
        super(ResidualDynaLIFBlock,self).__init__(
            in_channel,out_channel,kernel,stride,padding,num_block,bias,dropout,
        )

        modules=[]
        modules+=[
                nn.Conv2d(
                    in_channels=in_channel,out_channels=out_channel,
                    kernel_size=kernel,stride=stride,padding=padding,bias=bias
                ),
            ]
        
        for _ in range(num_block):

            #Calculate the output size of the block
            module_outsize=get_conv_outsize(nn.Sequential(*modules),in_size=in_size,in_channel=in_channel) #[1(batch) x channel x h x w]
            modules.append(
                DynamicLIF(
                    in_size=tuple(module_outsize[1:]), dt=dt,
                    init_tau=init_tau, min_tau=min_tau,
                    threshold=threshold, vrest=vrest,
                    reset_mechanism=reset_mechanism, spike_grad=spike_grad,
                    output=False
                )
            )
            
            if dropout>0:
                modules+=[
                    nn.Dropout(dropout,inplace=False)
                ]
            modules+=[
                    nn.Conv2d(
                        in_channels=out_channel,out_channels=out_channel,
                        kernel_size=kernel,stride=stride,padding=padding,bias=bias
                    )
                ]
            
            
        self.model=nn.Sequential(*modules)

        self.shortcut=nn.Sequential()
        if not in_channel==out_channel:
            self.shortcut=nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channel,out_channels=out_channel,
                        kernel_size=1,stride=1,padding=0,bias=bias
                    )
            )


    def init_voltage(self):
        for layer in self.model:
            if isinstance(layer,DynamicLIF):
                layer.init_voltage()


    def set_dynamic_params(self,a):
        """
        LIF time constant & function that changes membrane resistance
        :param a: [scalar] time scale at that moment
        """
        for layer in self.model:
            if isinstance(layer,DynamicLIF): #According to the Laplace transform, multiplying it by the time scale should work.
                layer.a = a


    def reset_params(self):
        for layer in self.model:
            if isinstance(layer,DynamicLIF):
                layer.a=1.0


    def get_tau(self):
        """
        tau 
        """

        taus={}
        layer_num=0
        for layer in self.model:
            if isinstance(layer,DynamicLIF):
                with torch.no_grad():
                    tau=layer.min_tau + layer.w.sigmoid()
                    # print(f"layer: Residual{layer._get_name()}-layer{layer_num}, tau shape: {tau.shape}")
                    taus["ResidualDynamicLIF"]=tau
                    layer_num+=1

        return taus
