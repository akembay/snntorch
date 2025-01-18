import h5py
from multiprocessing import Pool
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
import numpy as np
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def print_terminal(contents="",pre="\033[96m",suffi="\033[0m"):
    """
    :param pre: prefix
    :param contentns: contents that can be cut off
    :param suffi: suffix
    """

    import shutil
    termianl_width=shutil.get_terminal_size().columns
    s= contents[:termianl_width] if len(contents)>termianl_width else contents

    print(pre+s+suffi)
    

def load_yaml(file_path):
    """
    Load a YAML file and return its contents as a dictionary.
    
    :param file_path: Path to the YAML file.
    :return: Dictionary containing the YAML file contents.
    """
    import yaml
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)



def load_single_hdf5(path):
    with h5py.File(path, 'r') as f:
        data = f['events'][:]
        target = f['target'][()]
    return data, target

def load_hdf5(file_path_list: list, num_workers: int = 64):
    """
    path Read hdf5 files from a list of paths and return the data.
    :return datas: [batch x time_sequence x ...]
    :return targets: [batch]
    """
    with Pool(num_workers) as pool:
        results = pool.map(load_single_hdf5, file_path_list)

    datas, targets = zip(*results)
    return list(datas), list(targets)


def save_dict2json(data, saveto):
    """
    Save a dictionary to a specified path in JSON format.

    Parameters:
    data (dict): The dictionary to save.
    saveto (str or Path): The path where the JSON file will be saved.
    """
    import json
    with open(saveto, 'w') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


def load_json2dict(file_path):
    """A function to load a JSON file as a dict"""
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def resample_scale(a: list, target_length: int) -> list:
    """
    A function that resamples a 1-dimensional list to a specified length
    Uses linear interpolation
    :param data: 1-dimensional list to resample
    :param target_length: Length after resampling
    :return: Resampled list
    """
    original_length = len(a)
    if original_length == target_length:
        return a

    original_indices = np.linspace(0, original_length - 1, num=original_length)
    target_indices = np.linspace(0, original_length - 1, num=target_length)
    
    resampled_data = np.interp(target_indices, original_indices, a)
    
    return resampled_data.tolist()

def spike2timestamp(spike:np.ndarray,dt:float):
    """
    A function that converts a spike sequence of 0 or 1 into a spike timestamp
    :param spike: [time-sequence x ...]
    :return timestamp: The time at which the spike occurs
    :return idx_x: The spatial position at which the spike occurs
    """
    idx_sp=np.argwhere(spike==1)
    idx_t=idx_sp[:,0]
    idx_x=idx_sp[:,1:]
    timestamp=dt*idx_t
    return timestamp.reshape(-1,1),idx_x


def timestamp2spike(timestamp:np.ndarray,idx_x,dt,spike_shape:tuple):
    """
    :param spike_shape: Size of non-time dimensions
    """
    from math import ceil

    T=ceil(np.max(timestamp)/dt)+1
    idx_time=np.array((timestamp/dt).round(),dtype=np.int64)

    spike=np.zeros(shape=(T,*spike_shape))
    spike_idx=np.concatenate([idx_time,idx_x],axis=1)
    spike[tuple(spike_idx.T)]=1

    return spike


def scale_sequence(data:np.ndarray,a:list,dt:float):
    """
    :param data: [batch x time-sequence x ...]
    :param a: List of scalings. 1 or more [time-sequence]
    :param dt: dataの⊿t
    """
    from math import ceil

    elapsed = np.cumsum(np.concatenate(([0], a[:-1]))) * dt
    T_max=ceil(elapsed[-1]/dt)
    scaled_data=[]
    for data_i in tqdm(data): #Process each batch one by one
        timestamp, idx_x=spike2timestamp(data_i,dt)

        scaled_timestamp = np.zeros_like(timestamp)

        for t in range(data_i.shape[0]):
            mask = (timestamp >= t * dt) & (timestamp < (t + 1) * dt)
            scaled_timestamp[mask] = elapsed[t]

        scaled_spike=timestamp2spike(
            scaled_timestamp,idx_x,
            dt,data_i.shape[1:]
        )

        if scaled_spike.shape[0]<T_max:
            scaled_spike=np.concatenate([
                scaled_spike, np.zeros(shape=(T_max-scaled_spike.shape[0], *data_i.shape[1:]))
                ],axis=0)

        scaled_data.append(scaled_spike)

    return np.array(scaled_data)


def calculate_accuracy(output, target):
    """
    A function for checking acc of LSTM etc.
    """
    import torch
    predicted:torch.Tensor = torch.argmax(output, 1)
    correct = (predicted == target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy




from math import ceil
class Event2Frame():
    def __init__(self, sensor_size,time_window):
        """
        :param sensor_size: (channel x h x w) * Be sure to give in this order
        """
        self.sensor_size=sensor_size
        self.time_window=time_window

    def __call__(self, events:np.ndarray):
        """
        :param events: [event_num x (x,y,p,t)]
        """

        t_start=events[0]["t"]
        t_end=events[-1]["t"]
        time_length=ceil((t_end-t_start)/self.time_window)

        frame=np.zeros(shape=(time_length,)+self.sensor_size,dtype=np.int16)
        current_time_window=t_start+self.time_window
        t=0
        for e in events:
            if e["t"]>current_time_window:
                current_time_window+=self.time_window
                t+=1
            frame[t,int(e["p"]),e["y"],e["x"]]=1

        return frame
    

class Pool2DTransform(nn.Module):
    def __init__(self, pool_size,pool_type="max"):
        super(Pool2DTransform,self).__init__()
        
        if pool_type=="max".casefold():
            self.pool_layer=nn.MaxPool2d(pool_size)
        if pool_type=="avg".casefold():
            self.pool_layer=nn.AvgPool2d(pool_size)

    def __call__(self, events):
        # tensor should be of shape (T, C, H, W)
        with torch.no_grad():
            events = self.pool_layer(events.to(torch.float))
        return events  # Remove batch dimension


def save_heatmap(frame,output_path,file_name,scale=5):
    """
    :param frame: [h x w]
    """
    import cv2
    import os
    import matplotlib.pyplot as plt

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    h,w=frame.shape
    normalized_frame = ((frame + 1) / 2 * 255).astype(np.uint8)
    # heatmap = cv2.applyColorMap(normalized_frame, cv2.COLORMAP_JET)
    resized_heatmap = cv2.resize(normalized_frame, (w*scale,h*scale), interpolation=cv2.INTER_NEAREST)

    plt.imsave(str(output_path / file_name), resized_heatmap,cmap="viridis")

def save_heatmap_video(frames, output_path, file_name, fps=30, scale=5):
    import cv2
    import subprocess
    import os

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    height, width = frames[0].shape
    new_height, new_width = int(height * scale), int(width * scale)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    tmpout = str(output_path / "tmp.avi")
    video = cv2.VideoWriter(tmpout, fourcc, fps, (new_width, new_height), isColor=True)

    for i, frame in enumerate(frames):
        # Normalize frame to range [0, 255] with original range [-1, 1]
        normalized_frame = ((frame + 1) / 2 * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized_frame, cv2.COLORMAP_JET)
        resized_heatmap = cv2.resize(heatmap, (new_width, new_height))

        # Add frame number text
        cv2.putText(resized_heatmap, f"Frame: {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        video.write(resized_heatmap)

    video.release()

    # Re-encode the video using ffmpeg
    file_name = file_name + ".mp4" if not ".mp4" in file_name else file_name
    ffmpeg_command = [
        'ffmpeg', '-y', '-i', tmpout,
        '-pix_fmt', 'yuv420p', '-vcodec', 'libx264',
        '-crf', '23', '-preset', 'medium', str(output_path / file_name)
    ]
    subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Remove the temporary file
    os.remove(tmpout)
    

    
def test_sparsity(model, threshold=1e-5):
    '''Test Weight Sparsity'''
    total_params = 0
    zero_params = 0
    near_zero_params = 0

    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()
            near_zero_params += torch.sum(torch.abs(param) < threshold).item()

    sparsity = zero_params / total_params * 100
    near_sparsity = near_zero_params / total_params * 100

    print(f"Total Parameters: {total_params}")
    print(f"Zero Parameters: {zero_params} ({sparsity:.2f}%)")
    print(f"Near-Zero Parameters: {near_zero_params} ({near_sparsity:.2f}%)")

    return sparsity, near_sparsity


def test_loss_list(model, data_loader, loss_function, device):
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    iteration_loss_list = []  # List to store loss for each iteration

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device).permute((1,0,*[i+2 for i in range(x.ndim-2)])).to(torch.float), y.to(device)
            x =torch.squeeze(x) #Remove unnecessary channel dimensions
            outputs = model(x)  # Get the model predictions

            # Extract the relevant output component if it's a tuple
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Assuming the first element is the spike recordings

            loss = loss_function(outputs, y)  # Calculate the loss for the current batch
            iteration_loss = loss.item()  # Extract the loss value
            iteration_loss_list.append(iteration_loss*100)  # Record the loss for the current iteration

    # You can also calculate the overall average loss if needed
    average_loss = sum(iteration_loss_list) / len(iteration_loss_list) if iteration_loss_list else 0

    return average_loss, iteration_loss_list  # Return the average loss and the list of losses per iteration


import torch
import torch.nn as nn
from snntorch import surrogate

def measure_activation_sparsity(model, dataloader, num_batches=None):
    """
    Measures activation sparsity in an SNN by tracking spike activities across LIF layers.
    Specifically designed for SNNs with LIF neurons.
    
    Args:
        model: The SNN model with LIF layers
        dataloader: DataLoader containing the test/validation data
        num_batches: Number of batches to process (None = all batches)
    """
    model.eval()
    device = next(model.parameters()).device
    activation_counts = {}
    total_counts = {}
    
    def hook_fn(layer_name):
        def hook(module, input, output):
            # For LIF layers, output could be either (spikes, mem) or just spikes
            if isinstance(output, tuple):
                spikes = output[0]  # Get spike output
            else:
                spikes = output
                
            # Ensure spikes tensor is on the correct device
            if spikes.device != device:
                spikes = spikes.to(device)
            
            # Count non-spiking neurons (zeros indicate no spike)
            inactive = (spikes == 0).float().sum().item()
            total = spikes.numel()
            
            if layer_name not in activation_counts:
                activation_counts[layer_name] = 0
                total_counts[layer_name] = 0
            
            activation_counts[layer_name] += inactive
            total_counts[layer_name] += total
            
        return hook
    
    # Register hooks specifically for LIF layers
    hooks = []
    for name, module in model.named_modules():
        if 'LIF' in str(type(module)):  # Detect LIF layers
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    try:
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                if num_batches and batch_idx >= num_batches:
                    break
                
                # Move data to the same device as the model
                data = data.to(device)
                
                # Forward pass - handle both time steps if input is temporal
                if len(data.shape) > 3:  # Temporal data [T x B x ...]
                    for t in range(data.shape[0]):
                        model(data[t:t+1])
                else:
                    model(data)
    
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
        print(f"Model device: {device}")
        print(f"Input data device: {data.device}")
        raise
    
    finally:
        # Remove the hooks
        for hook in hooks:
            hook.remove()
    
    # Calculate sparsity metrics
    sparsity_metrics = {}
    overall_inactive = 0
    overall_total = 0
    
    for layer_name in activation_counts:
        inactive = activation_counts[layer_name]
        total = total_counts[layer_name]
        sparsity = (inactive / total * 100) if total > 0 else 0
        
        sparsity_metrics[layer_name] = {
            'sparsity_percentage': sparsity,
            'inactive_count': inactive,
            'total_activations': total
        }
        
        overall_inactive += inactive
        overall_total += total
    
    # Calculate overall sparsity
    overall_sparsity = (overall_inactive / overall_total * 100) if overall_total > 0 else 0
    sparsity_metrics['overall'] = {
        'sparsity_percentage': overall_sparsity,
        'inactive_count': overall_inactive,
        'total_activations': overall_total
    }
    
    return sparsity_metrics

def print_sparsity_report(model, dataloader, num_batches=None):
    """
    Prints a formatted report of spike sparsity in the SNN.
    """
    try:
        metrics = measure_activation_sparsity(model, dataloader, num_batches)
        
        print("\nSpike Activity Report")
        print("-" * 50)
        
        for layer_name, layer_metrics in metrics.items():
            if layer_name == 'overall':
                print("\nOverall Network Statistics:")
            else:
                print(f"\nLayer: {layer_name}")
                
            sparsity = layer_metrics['sparsity_percentage']
            inactive = layer_metrics['inactive_count']
            total = layer_metrics['total_activations']
            
            print(f"Spike Sparsity: {sparsity:.2f}%")
            print(f"Non-spiking Neurons: {inactive:,}")
            print(f"Total Potential Spikes: {total:,}")
            if layer_name != 'overall':
                print(f"Average Firing Rate: {100-sparsity:.2f}%")
    
    except Exception as e:
        print(f"Error in sparsity measurement: {str(e)}")
        print("Device information:")
        print(f"Model device: {next(model.parameters()).device}")
        if len(list(dataloader)) > 0:
            batch = next(iter(dataloader))
            print(f"Data device: {batch[0].device}")
            


def measure_model_sparsity(model, dataloader, epoch, resultpath):
    # Measure weight sparsity
    weight_sparsity, near_zero = test_sparsity(model)
    
    # Measure activation sparsity with the function that returns metrics
    model.eval()
    with torch.no_grad():
        activation_metrics = measure_activation_sparsity(model, dataloader, num_batches=5)  # This returns the metrics
    
    # Save metrics
    sparsity_metrics = {
        'epoch': epoch,
        'weight_sparsity': weight_sparsity,
        'near_zero_weights': near_zero,
        'activation_sparsity': {
            'overall': activation_metrics['overall']['sparsity_percentage'],
            'layer_wise': {
                name: metrics['sparsity_percentage'] 
                for name, metrics in activation_metrics.items() 
                if name != 'overall'
            }
        }
    }
    
    # Save to file
    sparsity_file = resultpath / f"sparsity_metrics_epoch_{epoch}.json"
    save_dict2json(sparsity_metrics, sparsity_file)
    
    # Print report for visualization
    print("\nSpike Activity Report")
    print("-" * 50)
    for layer_name, metrics in activation_metrics.items():
        if layer_name == 'overall':
            print("\nOverall Network Statistics:")
        else:
            print(f"\nLayer: {layer_name}")
        
        sparsity = metrics['sparsity_percentage']
        inactive = metrics['inactive_count']
        total = metrics['total_activations']
        
        print(f"Spike Sparsity: {sparsity:.2f}%")
        print(f"Non-spiking Neurons: {inactive:,}")
        print(f"Total Potential Spikes: {total:,}")
        if layer_name != 'overall':
            print(f"Average Firing Rate: {100-sparsity:.2f}%")
    
    return sparsity_metrics



