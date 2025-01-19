import argparse
from pathlib import Path
ROOT = Path(__file__).parent.parent
import sys
sys.path.append(str(ROOT))

import os
import torch
import json
import numpy as np
from tqdm import tqdm
from snntorch import functional as SF
from torch.utils.data import DataLoader
import pandas as pd
import torchvision
import matplotlib.pyplot as plt
from snntorch import surrogate


def set_seed(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
    
from delta_src.utils import load_yaml, save_dict2json
from delta_src.model import NormalSNN, DeltaSNN

def split_mnist(train_x, train_y, test_x, test_y, n_splits=5):
    """ Given the training set, split the tensors by the class label. """
    n_classes = 10
    if n_classes % n_splits != 0:
        print("n_classes should be a multiple of the number of splits!")
        raise NotImplemented
    class_for_split = n_classes // n_splits
    mnist_train_test = [[],[]]  # train and test
    for id, data_set in enumerate([(train_x, train_y), (test_x, test_y)]):
        for i in range(n_splits):
            start = i * class_for_split
            end = (i + 1) * class_for_split
            split_idxs = np.where(np.logical_and(data_set[1] >= start, data_set[1] < end))[0]
            mnist_train_test[id].append((data_set[0][split_idxs], data_set[1][split_idxs]))
    return mnist_train_test


class SubsetMNIST(torch.utils.data.Dataset):
    """Dataset class for MNIST splits"""
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        # Map labels to 0-1 for each task
        unique_labels = torch.unique(self.targets)
        self.label_mapping = {label.item(): idx for idx, label in enumerate(sorted(unique_labels))}
        self.mapped_targets = torch.tensor([self.label_mapping[t.item()] for t in self.targets])

    def __getitem__(self, index):
        # For MNIST, need to add channel dimension and normalize
        img = self.data[index].float() / 255.0
        img = img.unsqueeze(0)  # Add channel dimension [1, 28, 28]
        return img, self.mapped_targets[index]
        
    def __len__(self):
        return len(self.data)


def train_task(model, train_loader, seen_test_loaders, criterion, optim, device, task_id, epoch, resultpath, model_conf):
    """Unified training function that works for both normal SNN and DeltaSNN models"""
    result = []
    is_delta = isinstance(model, DeltaSNN)
    
    for e in range(epoch):
        model.train()
        train_loss_list = []
        train_acc_list = []
        
        # Training loop
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device).view(-1, model_conf["in_features"])  # [batch, 784]
            inputs = inputs.unsqueeze(0).repeat(model_conf["num_steps"], 1, 1)  # [time=10, batch, 784]
            targets = targets.to(device)
            
            optim.zero_grad()
            spk_rec = []
            
            # Initialize or reset states based on model type
            if is_delta:
                model.lif1.reset_hidden()
                model.lif2.reset_hidden()
            else:
                mem1 = model.lif1.init_leaky()
                mem2 = model.lif2.init_leaky()
            
            # Forward pass over time steps
            for step in range(model.num_steps):
                cur1 = model.fc1(inputs[step])
                
                if is_delta:
                    spk1 = model.lif1(cur1)
                    cur2 = model.fc2(spk1)
                    spk2 = model.lif2(cur2)
                else:
                    spk1, mem1 = model.lif1(cur1, mem1)
                    cur2 = model.fc2(spk1)
                    spk2, mem2 = model.lif2(cur2, mem2)
                
                spk_rec.append(spk2)
            
            spk_rec = torch.stack(spk_rec)
            loss = criterion(spk_rec, targets)
            
            loss.backward()
            optim.step()
            
            # Apply gradient clipping for DeltaSNN
            if is_delta:
                model.clip_gradients()
            
            train_loss_list.append(loss.item())
            train_acc = SF.accuracy_rate(spk_rec, targets)
            train_acc_list.append(train_acc)
            
            if batch_idx % 10 == 0:
                print(f"Task {task_id} - Epoch [{e+1}/{epoch}], Step [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Acc: {train_acc:.4f}")
        
        # Evaluation
        model.eval()
        task_accuracies = []
        task_stds = []
        
        with torch.no_grad():
            for test_id, test_loader in enumerate(seen_test_loaders):
                test_acc_list = []
                for inputs, targets in test_loader:
                    inputs = inputs.to(device).view(-1, model_conf["in_features"])
                    inputs = inputs.unsqueeze(0).repeat(model_conf["num_steps"], 1, 1)
                    targets = targets.to(device)
                    
                    spk_rec = []
                    
                    # Initialize or reset states based on model type
                    if is_delta:
                        model.lif1.reset_hidden()
                        model.lif2.reset_hidden()
                    else:
                        mem1 = model.lif1.init_leaky()
                        mem2 = model.lif2.init_leaky()
                    
                    # Forward pass over time steps
                    for step in range(model.num_steps):
                        cur1 = model.fc1(inputs[step])
                        
                        if model_conf["type"]=="deltasnn".casefold():
                            spk1 = model.lif1(cur1)
                            cur2 = model.fc2(spk1)
                            spk2 = model.lif2(cur2)
                        else:
                            spk1, mem1 = model.lif1(cur1, mem1)
                            cur2 = model.fc2(spk1)
                            spk2, mem2 = model.lif2(cur2, mem2)
                        
                        spk_rec.append(spk2)
                    
                    spk_rec = torch.stack(spk_rec)
                    test_acc_list.append(SF.accuracy_rate(spk_rec, targets))
                
                task_accuracies.append(np.mean(test_acc_list))
                task_stds.append(np.std(test_acc_list))
        
        result.append({
            'epoch': e,
            'train_loss': np.mean(train_loss_list),
            'train_acc': np.mean(train_acc_list),
            'task_accuracies': task_accuracies,
            'task_stds': task_stds
        })
        
        print(f"\nTask {task_id}, Epoch {e+1}/{epoch}:")
        print(f"Train Acc: {np.mean(train_acc_list):.4f}")
        for t in range(len(seen_test_loaders)):
            print(f"Task {t+1} Acc: {task_accuracies[t]:.4f} ± {task_stds[t]:.4f}")
        
    return model, result

def plot_accuracy_per_task(all_results, resultpath):
    """Plot accuracy curves for all tasks"""
    plt.figure(figsize=(10, 6))
    
    colors = ['red', 'orange', 'yellow', 'green', 'teal']
    n_tasks = len(all_results)
    
    for task_id in range(n_tasks):
        task_results = all_results[task_id]
        n_epochs = len(task_results)
        
        # X-axis points for this task
        x_points = np.linspace(2*(task_id), 2*(task_id+1), n_epochs)
        
        # Plot accuracy for each previous task
        for prev_task in range(task_id + 1):
            accuracies = [r['task_accuracies'][prev_task] for r in task_results]
            stds = [r['task_stds'][prev_task] for r in task_results]
            
            color = colors[prev_task]
            style = '-' if prev_task == task_id else '--'
            alpha = 0.8 if prev_task == task_id else 0.5
            
            plt.plot(x_points, accuracies, style, color=color, alpha=alpha,
                    label=f'T{prev_task+1}' if task_id == prev_task else None)
            
            plt.fill_between(x_points,
                           np.array(accuracies) - np.array(stds),
                           np.array(accuracies) + np.array(stds),
                           color=color, alpha=0.2)
    
    plt.xlabel('class so far')
    plt.ylabel('Accuracy')
    plt.title('Accuracy curve per task')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.0)
    plt.xlim(0, 10)
    
    # Set integer ticks
    plt.xticks(range(0, 11, 2))
    
    plt.savefig(resultpath / 'accuracy_per_task.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, help="config")
    parser.add_argument("--device", default=0, help="GPU number")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}")
    resultpath = Path(args.target)/"result"
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    # Load MNIST dataset
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True)
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True)
    
    # Convert to tensors
    train_x = trainset.data.float() / 255.0
    train_y = trainset.targets
    test_x = testset.data.float() / 255.0
    test_y = testset.targets
    
    # Split into 5 tasks
    splitmnist = split_mnist(train_x, train_y, test_x, test_y, n_splits=5)
    
    # Data loader parameters
    batch_size = 128
    dataloader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Create data loaders for each task
    train_loaders = []
    test_loaders = []
    
    for task_id in range(5):
        train_loader = DataLoader(
            SubsetMNIST(splitmnist[0][task_id][0], splitmnist[0][task_id][1]),
            **dataloader_kwargs
        )
        test_loader = DataLoader(
            SubsetMNIST(splitmnist[1][task_id][0], splitmnist[1][task_id][1]),
            **dataloader_kwargs
        )
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    # Initialize model and criterion
    conf = load_yaml(Path(args.target).parent/"conf.yml")
    train_conf, model_conf = conf["train"], conf["model"]
    epoch=train_conf["epoch"]
    #iter_max=train_conf["iter"]
    #save_interval=train_conf["save_interval"]
    #minibatch=train_conf["batch"]
    
    #>> Preparing the model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if model_conf["type"]=="snn".casefold():
        model=NormalSNN(model_conf).to(device)
        criterion=SF.ce_rate_loss() 
    elif model_conf["type"]=="deltasnn".casefold():
        model = DeltaSNN(model_conf).to(device)
        criterion = SF.ce_rate_loss()
    else:
        raise ValueError(f"model type {model_conf['type']} is not supportated...")
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training on all tasks sequentially
    all_results = []
    seen_test_loaders = []
    
    
    for task_id in range(5):
        print(f"\nStarting Task {task_id+1} Training (digits {2*task_id}-{2*task_id+1})")
        seen_test_loaders.append(test_loaders[task_id])
        
        model, task_results = train_task(
            model=model,
            train_loader=train_loaders[task_id],
            seen_test_loaders=seen_test_loaders,
            criterion=criterion,
            optim=optimizer,
            device=device,
            task_id=task_id+1,
            epoch=epoch,
            resultpath=resultpath,
            model_conf=model_conf
        )
        all_results.append(task_results)
        
        # Plot current progress
        plot_accuracy_per_task(all_results, resultpath)
        
        if task_id == 4:  # On the final task
            final_epoch_results = task_results[-1]  # Get the last epoch's results
            formatted_results = {
                "model": model_conf["type"],
                "epoch": f"{epoch}",
                "accuracies": {
                    f"Task {i+1}": f"{acc:.4f} ± {std:.4f}" 
                    for i, (acc, std) in enumerate(zip(
                        final_epoch_results['task_accuracies'],
                        final_epoch_results['task_stds']
                    ))
                }
            }
            save_dict2json(formatted_results, resultpath / 'final_accuracies.json')
    print("\nTraining completed!")



if __name__ == "__main__":
    main()
