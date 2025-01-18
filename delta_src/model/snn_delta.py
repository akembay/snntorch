import torch
import torch.nn as nn
from snntorch import surrogate
from math import log

from .residual_block import ResidualBlock, ResidualLIFBlock
from .lif_model import LIF
import snntorch as snn


from snntorch import utils

class SNN(nn.Module):
    """
    SNN with dynamically changing time constant (TC)
    """

    def __init__(self,conf:dict):
        super(SNN,self).__init__()

        self.in_size=conf["in-size"]
        self.hiddens = conf["hiddens"]
        self.out_size = conf["out-size"]
        self.clip_norm=conf["clip-norm"] if "clip-norm" in conf.keys() else 1.0
        self.dropout=conf["dropout"]
        self.output_mem=conf["output-membrane"]

    
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
            nn.Linear(self.in_size, self.hiddens[0],bias=is_bias),
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
                nn.Linear(prev_hidden, hidden,bias=is_bias),
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
            nn.Linear(self.hiddens[-1], self.out_size,bias=is_bias),
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




class CSNN(SNN):
    def __init__(self,conf):
        super(CSNN,self).__init__(conf)

        self.in_size = conf["in-size"]
        self.in_channel = conf["in-channel"]
        self.out_size = conf["out-size"]
        self.hiddens = conf["hiddens"]
        self.pool_type = conf["pool-type"]
        self.pool_size=conf["pool-size"]
        self.is_bn = conf["is-bn"]
        self.linear_hidden = conf["linear-hidden"]
        self.dropout = conf["dropout"]
        
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
        else:
            self.spike_grad = surrogate.atan() # arctan surrogate gradient function 



        modules=[]

        #>> Convolutional Layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        in_c=self.in_channel
        in_size=self.in_size
        for i,hidden_c in enumerate(self.hiddens):

            block,block_outsize=add_csnn_block(
                in_size=in_size, in_channel=in_c, out_channel=hidden_c,
                kernel=3, stride=1, padding=1,  # Setting kernel size, stride and padding
                is_bias=True, is_bn=self.is_bn, pool_type=self.pool_type, pool_size=self.pool_size[i] , dropout=self.dropout,
                lif_dt=self.dt, lif_init_tau=self.init_tau, lif_min_tau=self.min_tau,
                lif_threshold=self.v_threshold, lif_vrest=self.v_rest,
                lif_reset_mechanism=self.reset_mechanism, lif_spike_grad=self.spike_grad,
                lif_output=False,  # Setting to return no output
                is_train_tau=self.is_train_tau
            )
            modules+=block
            in_c=hidden_c
            in_size=block_outsize[-1]
        #<< Convolutional Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



        #>> Linear Layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules+=[
            nn.Flatten(),
            nn.Linear(block_outsize[1]*block_outsize[2]*block_outsize[3],self.linear_hidden,bias=True),
            LIF(
                in_size=(self.linear_hidden,), beta1 = self.beta1, dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=False,is_train_tau=self.is_train_tau
            ),
            nn.Linear(self.linear_hidden,self.out_size,bias=True),
            LIF(
                in_size=(self.out_size,), beta1 = self.beta1, dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=True,is_train_tau=self.is_train_tau
            ),
        ]
        #<< Linear Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        self.model=nn.Sequential(*modules)


def add_residual_block(
        in_size,in_channel,out_channel,kernel,stride,padding,is_bias,residual_block_num,is_bn,pool_type,pool_size,dropout,
        lif_dt,lif_init_tau,lif_min_tau,lif_threshold,lif_vrest,lif_reset_mechanism,lif_spike_grad,lif_output, is_train_tau,
        res_actfn
        ):
    """
    param: in_size: width and height (assume square)
    param: in_channel: channel size
    param: out_channel: size of output channel
    param: kernel: kernel size
    param: stride: stride size
    param: padding: padding size
    param: is_bias: whether to use bias
    param: residual_block_num: number of CNNs in ResBlock (can be 0)
    param: is_bn: whether to use batch normalization
    param: pool_type: type of pooling ("avg" or "max")
    param: pool_size: pool size
    param: dropout: dropout rate
    param: lif_dt: time step for LIF model
    param: lif_init_tau: initial time constant of LIF
    param: lif_min_tau: minimum time constant of LIF
    param: lif_threshold: firing threshold of LIF
    param: lif_vrest: LIF resting membrane potential
    param: lif_reset_mechanism: LIF membrane potential reset mechanism
    param: lif_spike_grad: LIF spike gradient function
    param: lif_output: Whether to return LIF output
    param: is_train_tau: Whether to learn LIF tau
    param: res_actfn: Activation function for residual block {relu, lif}
    """
    
    block=[]
    if res_actfn=="relu".casefold():
        block.append(
            ResidualBlock(
                in_channel=in_channel,out_channel=out_channel,
                kernel=kernel,stride=stride,padding=padding,
                num_block=residual_block_num,bias=is_bias
            )
        )
    elif res_actfn=="lif".casefold():
        block.append(
            ResidualLIFBlock(
                in_channel=in_channel,out_channel=out_channel,
                kernel=kernel,stride=stride,padding=padding,
                num_block=residual_block_num,bias=is_bias,
                in_size=in_size, dt=lif_dt,
                init_tau=lif_init_tau, min_tau=lif_min_tau,
                threshold=lif_threshold, vrest=lif_vrest,
                #reset_mechanism=lif_reset_mechanism, spike_grad=surrogate.fast_sigmoid(),
                reset_mechanism=lif_reset_mechanism, spike_grad=surrogate.atan(),
                output=False,is_train_tau=is_train_tau
            )
        )

    if is_bn:
        block.append(
            nn.BatchNorm2d(out_channel)
        )

    if pool_size>0:
        if pool_type=="avg".casefold():
            block.append(nn.AvgPool2d(pool_size))
        elif pool_type=="max".casefold():
            block.append(nn.MaxPool2d(pool_size))

    #Calculate the output size of the block
    block_outsize=get_conv_outsize(nn.Sequential(*block),in_size=in_size,in_channel=in_channel) #[1(batch) x channel x h x w]

    block.append(
        LIF(
            in_size=tuple(block_outsize[1:]), beta1 = self.beta1, dt=lif_dt,
            init_tau=lif_init_tau, min_tau=lif_min_tau,
            threshold=lif_threshold, vrest=lif_vrest,
            reset_mechanism=lif_reset_mechanism, spike_grad=lif_spike_grad,
            output=lif_output, is_train_tau=is_train_tau
        )
    )

    if dropout>0:
        block.append(nn.Dropout2d(dropout, inplace=False))
    
    return block, block_outsize


class ResCSNN(SNN):
    """
    SNN with CNN as residual
    """
    def __init__(self,conf):
        super(ResCSNN,self).__init__(conf)

        self.in_size = conf["in-size"]
        self.in_channel = conf["in-channel"]
        self.out_size = conf["out-size"]
        self.hiddens = conf["hiddens"]
        self.residual_blocks=conf["residual-block"] #Number of CNNs per residual block
        self.pool_type = conf["pool-type"]
        self.pool_size=conf["pool-size"]
        self.is_bn = conf["is-bn"]
        self.linear_hidden = conf["linear-hidden"]
        self.dropout = conf["dropout"]
        self.res_actfn=conf["res-actfn"] if "res-actfn" in conf.keys() else "relu" #Residual block activation function

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

        is_bias=True

        modules=[]

        #>> Convolutional Layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        in_c=self.in_channel
        in_size=self.in_size
        for i,hidden_c in enumerate(self.hiddens):

            block,block_outsize=add_residual_block(
                res_actfn=self.res_actfn,
                in_size=in_size, in_channel=in_c, out_channel=hidden_c,
                kernel=3, stride=1, padding=1,  # Setting kernel size, stride and padding
                is_bias=is_bias, residual_block_num=self.residual_blocks[i],
                is_bn=self.is_bn, pool_type=self.pool_type,pool_size=self.pool_size[i],dropout=self.dropout,
                lif_dt=self.dt, lif_init_tau=self.init_tau, lif_min_tau=self.min_tau,
                lif_threshold=self.v_threshold, lif_vrest=self.v_rest,
                lif_reset_mechanism=self.reset_mechanism, lif_spike_grad=self.spike_grad,
                lif_output=False, is_train_tau=self.is_train_tau  # Setting to return no output
            )
            modules+=block
            in_c=hidden_c
            in_size=block_outsize[-1]
        #<< Convolutional Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



        #>> Linear Layer >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        modules+=[
            nn.Flatten(),
            nn.Linear(block_outsize[1]*block_outsize[2]*block_outsize[3],self.linear_hidden,bias=is_bias),
            LIF(
                in_size=(self.linear_hidden,),dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=False,is_train_tau=self.is_train_tau
            ),
            nn.Linear(self.linear_hidden,self.out_size,bias=is_bias),
            LIF(
                in_size=(self.out_size,),dt=self.dt,init_tau=self.init_tau, min_tau=self.min_tau,threshold=self.v_threshold,vrest=self.v_rest,
                reset_mechanism=self.reset_mechanism,spike_grad=self.spike_grad,output=True,is_train_tau=self.is_train_tau
            ),
        ]
        #<< Linear Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        self.model=nn.Sequential(*modules)


#===================================  DeltaNET
#MNIST models 

# normal net SNN
class LeakySNNMNIST(nn.Module):
    def __init__(self, model_conf):
        super().__init__()
        
        # Extract configurations
        self.num_inputs = model_conf["in_features"]
        self.num_hidden = model_conf["hidden_size"]
        self.num_outputs = model_conf["out_features"]
        self.beta1 = model_conf.get("beta1", 0.9)  # Default to 0.9 if not specified
        self.beta2 = model_conf.get("beta2", 0.9)  # Default to 0.9 if not specified
        self.num_steps = model_conf.get("num_steps", 10)  # Default to 10 if not specified
        
        # Initialize layers
        self.fc1 = nn.Linear(self.num_inputs, self.num_hidden, bias=False)
        self.lif1 = snn.Leaky(beta=self.beta1)  # Fixed decay rate
        self.fc2 = nn.Linear(self.num_hidden, self.num_outputs, bias=False)
        self.lif2 = snn.Leaky(beta=self.beta2, learn_beta=True)  # Learnable decay rate
    
    def forward(self, x):
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        
        # Temporal processing
        for step in range(self.num_steps):
            cur1 = self.fc1(x.flatten(1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
        
        # Return accumulated spike output
        return torch.stack(spk2_rec).mean(0)
    
#===================================
#DeltaSNN 
class DeltaSNN(nn.Module):
    """
    SNN implementation using Delta_Leaky neurons instead of LIF
    """
    def __init__(self, conf: dict):
        super(DeltaSNN, self).__init__()
        
        self.in_size = conf["in-size"]
        self.hiddens = conf["hiddens"]
        self.out_size = conf["out-size"]  # Using num_classes from config
        self.clip_norm = conf["clip-norm"] if "clip-norm" in conf.keys() else 1.0
        self.dropout = conf["dropout"]
        self.output_mem = conf["output-membrane"]
        
        # Delta-specific parameters
        self.beta = conf["beta"] if "beta" in conf.keys() else 0.9
        self.delta_threshold = conf["delta-threshold"] #if "delta-threshold" in conf.keys() else 0.02
        self.learn_threshold = conf["learnable-threshold"] #if "learnable-threshold" in conf.keys() else True # False
        self.learnable_delta_threshold=conf["learnable_delta_threshold"]
        
        if "fast" in conf["spike-grad"].casefold() and "sigmoid" in conf["spike-grad"].casefold():
            self.spike_grad = surrogate.fast_sigmoid()
        else:
            self.spike_grad = surrogate.atan()
            
        if self.learnable_delta_threshold:
            delta_threshold = torch.Tensor([self.delta_threshold])
            self.delta_threshold = nn.Parameter(delta_threshold)

            
        modules = []
        is_bias = True
        
        # Input Layer
        modules += [
            nn.Linear(self.in_size, self.hiddens[0], bias=is_bias),
            snn.Delta_Leaky(
                beta=self.beta,
                spike_grad=self.spike_grad,
                init_hidden=True,
                delta_threshold=self.delta_threshold,
                learnable_delta_threshold=self.learnable_delta_threshold
            ),
            nn.Dropout(self.dropout),
        ]
        
        # Hidden Layers
        prev_hidden = self.hiddens[0]
        for hidden in self.hiddens[1:]:
            modules += [
                nn.Linear(prev_hidden, hidden, bias=is_bias),
                snn.Delta_Leaky(
                    beta=self.beta,
                    spike_grad=self.spike_grad,
                    init_hidden=True,
                    delta_threshold=self.delta_threshold,
                    learnable_delta_threshold=self.learnable_delta_threshold
                ),
                nn.Dropout(self.dropout),
            ]
            prev_hidden = hidden
            
        # Output Layer
        modules += [
            nn.Linear(self.hiddens[-1], self.out_size, bias=is_bias),
            snn.Delta_Leaky(
                beta=self.beta,
                spike_grad=self.spike_grad,
                init_hidden=True,
                output=True,
                delta_threshold=self.delta_threshold,
                learnable_delta_threshold=self.learnable_delta_threshold
            ),
        ]
        
        self.model = nn.Sequential(*modules)
    
    def forward(self, s: torch.Tensor):
        """
        Forward pass through the network
        
        Args:
            s: Spike train [T x batch x ...]
            
        Returns:
            out_s: Output spikes [T x batch x ...]
            out_v: Output membrane potentials [T x batch x ...] (if output_mem=True)
        """
        T = s.shape[0]
        utils.reset(self.model)  # Reset all neuron states using utils.reset
        
        out_s, out_v = [], []
        for st in s:
            spk_out, mem_out = self.model(st)
            out_s.append(spk_out)
            out_v.append(mem_out)
            
        out_s = torch.stack(out_s, dim=0)
        out_v = torch.stack(out_v, dim=0)
        
        if self.output_mem:
            return out_s, [], out_v
        else:
            return out_s
            
    def clip_gradients(self):
        """Clips gradients to prevent exploding gradients"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)

      
        

    
 
class DeltaCSNN(DeltaSNN):
    """
    Convolutional version of DeltaSNN
    """
    def __init__(self, conf):
        super(DeltaCSNN, self).__init__(conf)
        
        self.in_size = conf["in-size"]
        self.in_channel = conf["in-channel"]
        self.out_size = conf["out_size"]
        self.hiddens = conf["hiddens"]
        self.pool_type = conf["pool-type"]
        self.pool_size = conf["pool-size"]
        self.is_bn = conf["is-bn"]
        self.linear_hidden = conf["linear-hidden"]
        
        modules = []
        
        # Convolutional Layers
        in_c = self.in_channel
        in_size = self.in_size
        for i, hidden_c in enumerate(self.hiddens):
            modules += [
                nn.Conv2d(in_c, hidden_c, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(self.pool_size[i]) if self.pool_type.lower() == "max" else nn.AvgPool2d(self.pool_size[i]),
                snn.Delta_Leaky(
                    beta=self.beta,
                    spike_grad=self.spike_grad,
                    init_hidden=True,
                    delta_threshold=self.delta_threshold,
                    learnable_delta_threshold=self.learnable_delta_threshold
                )
            ]
            if self.is_bn:
                modules.append(nn.BatchNorm2d(hidden_c))
            if self.dropout > 0:
                modules.append(nn.Dropout2d(self.dropout))
            in_c = hidden_c
            in_size = in_size // self.pool_size[i]
        
        # Calculate final conv output size
        final_size = in_size * in_size * self.hiddens[-1]
        
        # Linear Layers
        modules += [
            nn.Flatten(),
            nn.Linear(final_size, self.linear_hidden),
            snn.Delta_Leaky(
                beta=self.beta,
                spike_grad=self.spike_grad,
                init_hidden=True,
                delta_threshold=self.delta_threshold,
                learnable_delta_threshold=self.learnable_delta_threshold
            ),
            nn.Linear(self.linear_hidden, self.out_size),
            snn.Delta_Leaky(
                beta=self.beta,
                spike_grad=self.spike_grad,
                init_hidden=True,
                output=True,
                delta_threshold=self.delta_threshold,
                learnable_delta_threshold=self.learnable_delta_threshold
            )
        ]
        
        self.model = nn.Sequential(*modules)
        