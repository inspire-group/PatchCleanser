#######################################################################################
# Adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#######################################################################################

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from tqdm import tqdm
import random
import argparse
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from utils.cutout import Cutout

import timm

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",default='checkpoints',type=str)
parser.add_argument("--data_dir",default='data',type=str)
parser.add_argument("--dataset",default='imagenette',type=str)
parser.add_argument("--model",default='vit_base_patch16_224',type=str)
parser.add_argument("--epoch",default=10,type=int)
parser.add_argument("--lr",default=0.001,type=float)
parser.add_argument("--cutout_size",default=128,type=int)
parser.add_argument("--resume",action='store_true')
parser.add_argument("--n_holes",default=2,type=int)
parser.add_argument("--cutout",action='store_true')
args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir,args.dataset)

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

n_holes = args.n_holes
cutout_size = args.cutout_size
if args.cutout:
    model_name = args.model + '_cutout{}_{}_{}.pth'.format(n_holes,cutout_size,args.dataset)
else:
    model_name = args.model + '_{}.pth'.format(args.dataset)

device = 'cuda' 

if 'vit_base_patch16_224' in model_name:
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
elif 'resnetv2_50x1_bit_distilled' in model_name:
    model = timm.create_model('resnetv2_50x1_bit_distilled', pretrained=True)
elif 'resmlp_24_distilled_224' in model_name:
    model = timm.create_model('resmlp_24_distilled_224', pretrained=True)


# get data loader
if args.dataset in ['imagenette','flower102']:
    config = resolve_data_config({}, model=model)
    ds_transforms = create_transform(**config)
    if args.cutout:
        ds_transforms.transforms.append(Cutout(n_holes=n_holes, length=cutout_size))
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR,'train'),ds_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR,'val'),ds_transforms)
    num_classes = 10 if args.dataset=='imagenette' else 102
elif args.dataset in ['cifar','cifar100','svhn']:
    config = resolve_data_config({'crop_pct':1}, model=model)###############################to decide
    ds_transforms = create_transform(**config)
    if args.cutout:
        ds_transforms.transforms.append(Cutout(n_holes=n_holes, length=cutout_size))
    if args.dataset == 'cifar':
        train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=ds_transforms)
        val_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=ds_transforms)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=ds_transforms)
        val_dataset = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=ds_transforms)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(root=DATA_DIR, split='train', download=True, transform=ds_transforms)
        val_dataset = datasets.SVHN(root=DATA_DIR, split='test', download=True, transform=ds_transforms)
        num_classes = 10
print(ds_transforms)


image_datasets = {'train':train_dataset,'val':val_dataset}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,shuffle=True,num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64,shuffle=False,num_workers=4)

dataloaders={'train':train_loader,'val':val_loader}


print('device:',device)

def train_model(model, criterion, optimizer, scheduler, num_epochs=20 ,mask=False):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
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
                    outputs = model(inputs)
                    if isinstance(outputs,tuple):
                        outputs = (outputs[0]+outputs[1])/2
                        #outputs = outputs[0]

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':# and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('saving...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict()
                    }, os.path.join(MODEL_DIR,model_name))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if args.dataset!='imagenet':
    model.reset_classifier(num_classes=num_classes)
model = torch.nn.DataParallel(model)
model = model.to(device)
criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

#https://pytorch.org/tutorials/beginner/saving_loading_models.html
if args.resume:
    print('restoring model from checkpoint...')
    checkpoint = torch.load(os.path.join(MODEL_DIR,model_name))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    #https://discuss.pytorch.org/t/code-that-loads-sgd-fails-to-load-adam-state-to-gpu/61783/3
    optimizer_conv.load_state_dict(checkpoint['optimizer_state_dict'])
    exp_lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


model = train_model(model, criterion, optimizer,
                         exp_lr_scheduler, num_epochs=args.epoch)

