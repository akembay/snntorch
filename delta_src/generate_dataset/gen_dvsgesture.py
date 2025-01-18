import argparse
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
import os
import torch
import json
import h5py
import numpy as np
from tqdm import tqdm
import tonic
from math import ceil
import time
import concurrent.futures


def save_to_hdf5(file_path, data, target):
    """
    Compress and save in hdf5
    Otherwise the data size will be T order
    """
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('events', data=data, compression='gzip')
        f.create_dataset('target', data=target)


def change_resolution_v2(events: np.ndarray, resolution):
    """
    :param events: [event_num x (x,y,t,p)]
    """
    t_start = events[0]["t"]
    t_end = events[-1]["t"]
    event_length = ceil((t_end - t_start) / resolution)

    new_events = []

    for i in tqdm(range(event_length)):
        t_window_start = t_start + i * resolution
        t_window_end = t_window_start + resolution
        events_i = events[(t_window_start <= events["t"]) & (events["t"] < t_window_end)]

        if len(events_i) == 0:
            continue

        for polarity in [True, False]:
            events_polarity = events_i[events_i["p"] == polarity]
            if len(events_polarity) > 0:
                xy = list({(event["x"], event["y"]) for event in events_polarity})

                new_events += [
                    (e_xy[0], e_xy[1], polarity, int((t_window_start + t_window_end) / 2))
                    for e_xy in xy
                ]

    dtype = np.dtype([('x', '<i8'), ('y', '<i8'), ('p', '<i8'), ('t', '<i8')])
    new_events = np.array(new_events, dtype=dtype)

    # print("original events shape: ", events.shape)
    # print(events[:10])
    # print("new events shape: ", new_events.shape)
    # print(new_events[:10])

    return new_events




def process_window(events, t_window_start, t_window_end, resolution):
    events_i = events[(t_window_start <= events["t"]) & (events["t"] < t_window_end)]
    new_events_i = []

    if len(events_i) == 0:
        return new_events_i

    for polarity in [True, False]:
        events_polarity = events_i[events_i["p"] == polarity]
        if len(events_polarity) > 0:
            xy = list({(event["x"], event["y"]) for event in events_polarity})

            new_events_i += [
                (e_xy[0], e_xy[1], polarity, int((t_window_start + t_window_end) / 2))
                for e_xy in xy
            ]

    return new_events_i

def process_window_batch(events, t_window_starts, t_window_ends, resolution):
    new_events_batch = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_window, events, t_window_start, t_window_end, resolution)
            for t_window_start, t_window_end in zip(t_window_starts, t_window_ends)
        ]

        for future in concurrent.futures.as_completed(futures):
            new_events_batch.extend(future.result())

    return new_events_batch

def change_resolution(events: np.ndarray, resolution, num_workers=64):
    """
    Processing can be divided into multiple processes and processed using multiple threads, allowing for high-speed processing.
    More detailed research here : https://www.notion.so/DVSGesture-27bb047931454bc5a5fdb142b36f150f?pvs=4#a480ffdb10cc4ce6a0ea3baba8e30611
    :param events: [event_num x (x,y,t,p)]
    """
    t_start = events[0]["t"]
    t_end = events[-1]["t"]
    event_length = ceil((t_end - t_start) / resolution)

    new_events = []

    batch_size = int(event_length/np.max([num_workers,16]))  # Set the batch size so that the number of processes is equal to num_worker.
    t_window_starts = [t_start + i * resolution for i in range(event_length)]
    t_window_ends = [t_start + (i + 1) * resolution for i in range(event_length)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_window_batch, events, t_window_starts[i:i + batch_size], t_window_ends[i:i + batch_size], resolution)
            for i in range(0, event_length, batch_size)
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            new_events.extend(future.result())

    dtype = np.dtype([('x', '<i8'), ('y', '<i8'), ('p', '<i8'), ('t', '<i8')])
    new_events = np.array(new_events, dtype=dtype)

    # Sort by t
    new_events.sort(order='t')

    print("original events shape: ", events.shape)
    print(events[:5])
    print("new events shape: ", new_events.shape)
    print(new_events[:5])

    return new_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="dvsgesture-v1")
    parser.add_argument("--resolution", default=20, type=float, help="How many times should the resolution be reduced? If it is 20, reduce it to 20μs.")
    parser.add_argument("--num_worker",default=8,type=int, help="Number of cores to use")
    args = parser.parse_args()

    trainset = tonic.datasets.DVSGesture(save_to=str(ROOT / "original-data"), train=True)
    testset = tonic.datasets.DVSGesture(save_to=str(ROOT / "original-data"), train=False)

    start_time = time.time()


    event_savepath=ROOT/f"custum-data/DVSGesture/{args.filename}"
    if not os.path.exists(event_savepath):
        os.makedirs(event_savepath/"train")
        os.makedirs(event_savepath/"test")


    i_max=4
    print("generate train data...")
    for i in range(len(trainset)):
        print(f"[{i+1}/{len(trainset)}]")
        events = trainset[i][0]
        new_events = change_resolution(events, args.resolution,args.num_worker)
        # new_events = change_resolution_v2(events, args.resolution)

        save_to_hdf5(
            event_savepath/f"train/data{i}.h5",new_events,trainset[i][1]
        )

        if i_max>0 and i==i_max:
            break
    print("\033[92mdone\033[0m\n")

    print("generate test data...")
    for i in range(len(testset)):
        print(f"[{i+1}/{len(testset)}]")
        events = testset[i][0]
        new_events = change_resolution(events, args.resolution,args.num_worker)
        # new_events = change_resolution_v2(events, args.resolution)

        save_to_hdf5(
            event_savepath/f"test/data{i}.h5",new_events,testset[i][1]
        )

        if i_max>0 and i==i_max:
            break
    print("\033[92mdone\033[0m")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Time taken for processing the loop: {elapsed_time:.2f} seconds")


if __name__=="__main__":
    main()