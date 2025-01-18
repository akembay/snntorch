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

from utils import load_hdf5,Pool2DTransform, save_heatmap_video,print_terminal


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--save_video", action='store_true')
    args=parser.parse_args()

    #>> Data preparation start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    time_window=3000
    insize=32
    transform=torchvision.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size,target_size=(insize,insize)),
        tonic.transforms.ToFrame(sensor_size=(insize,insize,2),time_window=time_window),
        torch.from_numpy,
    ])


    cache_path=str(Path(__file__).parent/f"cache/gesture-window{time_window}")
    testset=tonic.datasets.DVSGesture(save_to=ROOTDIR/"original-data",train=False,transform=transform)
    testset=tonic.DiskCachedDataset(testset,cache_path=cache_path)
    #<< Data preparation ending <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    i_max=4
    if args.save_video:
        print_terminal("saving video...")
        for i in tqdm(range(len(testset)),total=i_max):
            testframe=testset[i][0]
            testframe[testframe>0]=1
            testframe=1.5*testframe[:,0] + 0.5*testframe[:,1] - 1
            save_heatmap_video(
                testframe,
                output_path=Path(__file__).parent/f"video-window{time_window}",file_name=f"eventframe_target{testset[i][1]}",
                fps=30,scale=10
            )

            if i==i_max:
                break
        print_terminal("done")


if __name__=="__main__":
    main()