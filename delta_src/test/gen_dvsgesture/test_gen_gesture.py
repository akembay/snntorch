"""
A test program to check whether the generated data is correct.
"""

from pathlib import Path
import sys
SRCDIR=Path(__file__).parent.parent.parent
ROOTDIR=Path(__file__).parent.parent.parent.parent
sys.path.append(str(SRCDIR))
sys.path.append(str(ROOTDIR))
import torch
import torchvision
import tonic
import os
import numpy as np
from PIL import Image
from typing import Callable, Optional
import argparse
from tqdm import tqdm

from utils import load_hdf5,Pool2DTransform, save_heatmap_video


class CustomDVSDataset(tonic.Dataset):
    """
    Class that creates a dataset from list data

    Parameters:
        data_path (string): Location of the .npy files.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
    """
    def __init__(
        self,
        data_list: list,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            save_to=str(ROOTDIR/"original-data"),
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.data = []
        self.targets = []

        for item in (data_list):
            self.data.append(item['events'])
            self.targets.append(item['target'])

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target
            class.
        """
        events = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)
        return events, target

    def __len__(self) -> int:
        return len(self.data)
    

def plot_events(s_ideal,s_real ,filename):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    T=s_ideal.shape[0]

    plt.subplot(2,1,1)
    spike_times =torch.flatten(s_ideal,start_dim=2).cpu().numpy()[:,0]
    print(s_ideal.shape,spike_times.shape)
    plt.imshow(spike_times.T,interpolation="nearest", label=f'Ideal In Spikes',cmap="viridis",aspect="auto")
    plt.colorbar()
    plt.title('Ideal In Spikes')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.xlim(0,T)


    plt.subplot(2,1,2)
    spike_times =torch.flatten(s_real,start_dim=2).cpu().numpy()[:,0]
    plt.imshow(spike_times.T,interpolation="nearest", cmap="viridis",aspect="auto")
    plt.colorbar()
    plt.title('Actual In Spikes')
    plt.xlabel('Time')
    plt.ylabel('Spike')
    plt.xlim(0,T)


    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"{filename}.png")



def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--save_video", action='store_true')
    args=parser.parse_args()

    custom_gesture_path=ROOTDIR/"custum-data/DVSGesture/dvsgesture-100us"

    testfiles=[custom_gesture_path/f"test/{file}" for file in os.listdir(custom_gesture_path/"test")]
    testdata=load_hdf5(testfiles)

    events,targets=testdata
    datalist=[]
    for i in range(len(events)):
        print(f"evetns shape: {events[i].shape}, target : {targets[i]}")
        print(f"events exp\n {events[i][:4]}\n")
        datalist.append(
            {"events":events[i], "target":targets[i]}
        )

    pooltype="avg"
    poolsize=2
    time_window=10000

    transform=torchvision.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size,time_window=time_window),
        torch.from_numpy,
        # Pool2DTransform(pool_size=poolsize,pool_type=pooltype)
    ])

    # dataset=CustomDVSDataset(
    #     data_list=datalist,
    #     transform=transform
    # )


    dataset=tonic.datasets.DVSGesture(
        save_to=str(ROOTDIR/"original-data"),  
        train=False,transform=transform
    )


    i_max=4
    if args.save_video:
        for i in range(len(dataset)):
            testframe=dataset[i][0]
            testframe[testframe>0]=1
            testframe=1.5*testframe[:,0] + 0.5*testframe[:,1] - 1
            save_heatmap_video(
                testframe.detach().to("cpu").numpy(),
                output_path=Path(__file__).parent/f"video-window{time_window}",file_name=f"eventframe_target{dataset[i][1]}",
                fps=30,scale=5
            )

            if i==i_max:
                break


    a=5
    for i in range(len(dataset)):

        transform=torchvision.transforms.Compose([
            # Event2Frame(sensor_size,time_window),
            tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size,time_window=int(time_window/a)),
            torch.from_numpy,
            # torchvision.transforms.Resize(resize,interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            # Pool2DTransform(pool_size=poolsize,pool_type=pooltype)
            ])
        # scaled_dataset=CustomDVSDataset(
        #     data_list=datalist,
        #     transform=transform
        # )

        scaled_dataset=tonic.datasets.DVSGesture(
            save_to=str(ROOTDIR/"original-data"),  
            train=False,transform=transform
        )


        in_real=scaled_dataset[i][0] #batch
        in_real[in_real>0]=1.0
        in_real=in_real.unsqueeze(1)
        print(f"real in-spike shape: {in_real.shape}")


        in_shape=dataset[i][0].shape
        input_size = in_shape[-1]
        in_channel=in_shape[1]
        in_ideal = torch.zeros(size=(int(a * in_shape[0]), 1, in_channel,input_size,input_size))
        # print(in_ideal.shape,in_shape)
        # exit(1)
        data = dataset[i][0][:in_shape[0]]  # Get data in advance
        for t in (range(in_shape[0])):
            in_idealdex = int(a * t)
            if in_idealdex < in_ideal.shape[0]:
                in_ideal[in_idealdex] = data[t]  # Reduce data access in loops
        in_ideal[in_ideal>0]=1
        print(f"ideal in-spike shape: {in_ideal.shape}")

        bias=12
        plot_size=16
        plot_events(
            s_ideal=in_ideal[:,:,:,bias:bias+plot_size,bias:bias+plot_size],
            s_real=in_real[:,:,:,bias:bias+plot_size,bias:bias+plot_size],
            filename=f"video-window{time_window}/ideal_vs_actual_target{dataset[i][1]}"
            )
        
        if i==i_max:
            break

if __name__=="__main__":
    main()