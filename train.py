#!/usr/bin/env python

# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
from __future__ import print_function, division

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision          import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data     import ( 
          DataLoader
        , Subset
        )
import time
import os
import copy
import tqdm

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
classes = 5
data_dir = './'
image_datasets = { x: ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                   for x in ['train', 'val']
                 }


def load_data (train_datapoints=None, val_datapoints=70):
    num_samples = { "train": train_datapoints
                  , "val":   val_datapoints
                  }

    # Get a balanced set of classes.
    dataset_indicies = { x: iter(torch.randperm(len(image_datasets[x])).numpy())
                         for x in ['train', 'val'] }

    indicies = { x: { 0: [], 1: [], 2: [], 3: [], 4: [] }
                 for x in ['train', 'val'] }

    # Run through every datapoint.
    # TODO: Fix
    for x in ['train', 'val']:
        for idx in dataset_indicies[x]:
            _, label = image_datasets[x][idx]

            if len(indicies[x][label]) < num_samples[x]:
                indicies[x][label].append(idx)

    
    datasets =  { x: Subset( image_datasets[x]
                           , np.concatenate( [ v for _, v in indicies[x].items() ] )
                           )
                  for x in ['train', 'val']
                }

    dataloaders = { x: DataLoader( datasets[x]
                                 , batch_size=4
                                 , shuffle=True
                                 , num_workers=4
                                 )
                    for x in ['train', 'val']
                  }

    dataset_sizes = { x: len(datasets[x]) for x in ['train', 'val'] }

    # print("sizes", dataset_sizes)

    # labels = np.concatenate([ label.numpy() for _, label in iter(dataloaders["val"]) ] )
    # ones   = np.sum(labels)
    # zeros  = len(labels) - ones
    # print(ones, zeros, labels)

    return dataloaders, dataset_sizes


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, 
        scheduler, num_epochs=25, method=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm.tqdm(range(num_epochs)):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss     = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs  = model(inputs)

                    if method == "argmax":
                        # Argmax:
                        _, preds = torch.max(outputs, 1)
                    
                    if method == "sampling":
                        # Sampling:
                        preds = torch.flatten(torch.multinomial(torch.softmax(outputs, 1), 1))
                    
                    if method == "threshold>80":
                      threshold = 0.8
                      tmp = torch.softmax(outputs, 1).clone().detach().cpu()
                      _, preds = torch.max(tmp.apply_(lambda x: x if x > threshold else 0), 1)
                      preds = preds.cuda()

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc       = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, epoch_acc


def finetune (train_datapoints=None, num_epochs=None, method=None):
    """ Finetune all the weights.
    """
    model_ft    = models.resnet18(pretrained=True)
    num_ftrs    = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, classes)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    dataloaders, dataset_sizes = load_data(train_datapoints=train_datapoints)

    model_ft, best_acc, epoch_acc = train_model( model_ft
                                    , dataloaders
                                    , dataset_sizes
                                    , criterion
                                    , optimizer_ft
                                    , exp_lr_scheduler
                                    , num_epochs=num_epochs
                                    , method=method
                                    )
    return best_acc, epoch_acc


def specialise (train_datapoints=None, num_epochs=None, method=None):
    """ Only train the last layer.
    """
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, classes)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


    dataloaders, dataset_sizes = load_data(train_datapoints=train_datapoints)

    model_ft, best_acc, epoch_acc = train_model( model_conv
                                    , dataloaders
                                    , dataset_sizes
                                    , criterion
                                    , optimizer_conv
                                    , exp_lr_scheduler
                                    , num_epochs=num_epochs
                                    , method=method
                                    )
    return best_acc, epoch_acc



def go():
    # epochs  = [1, 2, 5, 10, 20]
    epochs  = [10]
    # sizes   = [1, 2, 5, 10, 20, 50, 100, 200, 400, 500]
    sizes   = [1, 3, 5, 50, 300]
    # seeds   = [1, 2, 3, 4, 5]
    seeds   = [1, 2]
    methods = ["threshold>80", "argmax", "sampling"]

    results = []


    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        for e in epochs:
            for size in sizes:
                for method in methods:
                    print(f"size = {size}, epochs = {e}, seed = {seed}, method = {method}")
                    print("Fine-tuning...")
                    print("==============")
                    ft_acc, ft_epoch_acc = finetune(train_datapoints=size, num_epochs=e, method=method)

                    print("Specialising...")
                    print("===============")
                    sp_acc, sp_epoch_acc = specialise(train_datapoints=size, num_epochs=e, method=method)

                    data = { "epochs": e
                           , "size":   size
                           , "ft_acc": ft_acc.cpu().numpy()
                           , "ft_epoch_acc": ft_epoch_acc.cpu().numpy()
                           , "sp_epoch_acc": sp_epoch_acc.cpu().numpy()
                           , "sp_acc": sp_acc.cpu().numpy()
                           , "seed": seed
                           , "method": method
                           }

                    results.append(data)
                    print()

    df = pd.DataFrame(results)
    df.to_csv("big-run-all-results.csv", index=False)


