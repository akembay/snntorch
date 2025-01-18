"""
SHD：
tonic：https://tonic.readthedocs.io/en/latest/generated/tonic.datasets.SHD.html
Original data before it becomes an event：https://www.tensorflow.org/datasets/catalog/speech_commands?hl=ja
"""

import argparse
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent
import sys
sys.path.append(str(ROOT))
import os
import torch
import json
import h5py
import numpy as np
from tqdm import tqdm
import tonic
import matplotlib.pyplot as plt


from delta_src.utils import print_terminal

def plot_shd(event_frame, save_to):

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    plt.figure(figsize=(8, 8))  # Increase vertical size for image and histogram

    # Event frame image
    plt.subplot(2, 1, 1)
    plt.imshow(event_frame, cmap="viridis", aspect="auto", interpolation="nearest")
    plt.title("Event Frame")

    # histogram
    plt.subplot(2, 1, 2)
    plt.hist(event_frame.ravel(), bins=50, color='blue', alpha=0.7)
    plt.title("Event Frame Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(f"{str(save_to)}/exp_shd_event.png")
    plt.close()


def main():

    trainset = tonic.datasets.SHD(save_to=str(ROOT / "original-data"), train=True)
    testset = tonic.datasets.SHD(save_to=str(ROOT / "original-data"), train=False)

    print_terminal("original events"+"-"*1000)
    print(f"sensor size: {tonic.datasets.SHD.sensor_size}")
    print(f"train size: {len(trainset)}, test size: {len(testset)}")
    print(f"event shape: {trainset[0][0].shape}, \nexp event: \n{trainset[0][0][:20]}")


    # Get the number of classes in the target
    from tqdm import tqdm
    test_targets = [target for _, target in tqdm(testset)]
    num_classes = len(set(test_targets))
    print(f"Number of classes: {num_classes}")


    time_window=1000
    transform=tonic.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=tonic.datasets.SHD.sensor_size,time_window=time_window),
        torch.from_numpy
    ])
    trainset = tonic.datasets.SHD(save_to=str(ROOT / "original-data"), train=True,transform=transform)
    testset = tonic.datasets.SHD(save_to=str(ROOT / "original-data"), train=False,transform=transform)

    print_terminal("transformed frame events"+"-"*1000)
    print(f"train size: {len(trainset)}, test size: {len(testset)}")
    print(f"event frame shape: {trainset[0][0].shape}")

    plot_shd(
        testset[0][0][:,0].to("cpu").numpy().T,
        save_to=Path(__file__).parent
    )

if __name__=="__main__":
    main()

