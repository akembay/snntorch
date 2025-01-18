"""
I don't like the official ToFrame so I implemented it myself

(If the time-window is shortened, the event disappears)

*Addition
The problem was not with ToFrame but with Resize

From now on, I will use Pooling to reduce the size instead of Resize
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

from utils import Event2Frame, Pool2DTransform

def save_heatmap_video(frames, output_path, file_name, fps=30, scale=5):
    import cv2
    import subprocess

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    height, width = frames[0].shape
    new_height, new_width = int(height * scale), int(width * scale)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    tmpout = str(output_path / "tmp.avi")
    video = cv2.VideoWriter(tmpout, fourcc, fps, (new_width, new_height), isColor=True)

    for frame in frames:
        # Normalize frame to range [0, 255] with original range [-1, 1]
        normalized_frame = ((frame + 1) / 2 * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized_frame, cv2.COLORMAP_JET)
        resized_heatmap = cv2.resize(heatmap, (new_width, new_height))
        video.write(resized_heatmap)

    video.release()

    # Re-encode the video using ffmpeg
    file_name = file_name + ".mp4" if not ".mp4" in file_name else file_name
    ffmpeg_command = [
        'ffmpeg', '-y', '-i', tmpout,
        '-pix_fmt', 'yuv420p', '-vcodec', 'libx264',
        '-crf', '23', '-preset', 'medium', str(output_path / file_name)
    ]
    subprocess.run(ffmpeg_command)
    # Remove the temporary file
    os.remove(tmpout)


def main():

    datapath=ROOTDIR/"original-data"

    sensor_size=(2,128,128)
    time_window=5000
    resize=(32,32)
    poolsize=int(sensor_size[-1]/resize[-1])
    transform=torchvision.transforms.Compose([
        # Event2Frame(sensor_size,time_window),
        tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size,time_window=time_window),
        torch.from_numpy,
        # torchvision.transforms.Resize(resize,interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        Pool2DTransform(pool_size=poolsize)
        ])

    testset =  tonic.datasets.DVSGesture(save_to=str(datapath),  train=False,transform=transform)

    print(testset[0][0])
    print(f"testset[0] shape : {testset[0][0].shape}")


    testframe=testset[0][0]
    testframe[testframe>0]=1
    testframe=1.5*testframe[:,0] + 0.5*testframe[:,1] - 1
    savepath=Path(__file__).parent/"videos"
    save_heatmap_video(
        testframe.detach().to("cpu").numpy(),savepath,f"test_size{resize[0]}_window{time_window}",scale=10
    )

if __name__=="__main__":
    main()