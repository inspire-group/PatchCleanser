import torch
import torch.backends.cudnn as cudnn

import numpy as np 
import os 
import argparse
import time
from tqdm import tqdm

from utils.setup import get_model,get_data_loader
from utils.defense import gen_mask_set,double_masking#,challenger_masking

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",default='checkpoints',type=str,help="directory of checkpoints")
parser.add_argument('--data_dir', default='data', type=str,help="directory of data")
parser.add_argument('--dataset', default='imagenette',type=str,choices=('imagenette','imagenet','cifar','cifar100','svhn','flower102'),help="dataset")
parser.add_argument("--model",default='vit_base_patch16_224',type=str,help="model name")
parser.add_argument("--num_img",default=-1,type=int,help="number of randomly selected images for this experiment (-1: using the all images)")
parser.add_argument("--mask_stride",default=-1,type=int,help="mask stride s (square patch; conflict with num_mask)")
parser.add_argument("--num_mask",default=-1,type=int,help="number of mask in one dimension (square patch; conflict with mask_stride)")
parser.add_argument("--patch_size",default=32,type=int,help="size of the adversarial patch (square patch)")
parser.add_argument("--pa",default=-1,type=int,help="size of the adversarial patch (first axis; for rectangle patch)")
parser.add_argument("--pb",default=-1,type=int,help="size of the adversarial patch (second axis; for rectangle patch)")

args = parser.parse_args()
DATASET = args.dataset
MODEL_DIR = os.path.join('.',args.model_dir)
DATA_DIR = os.path.join(args.data_dir,DATASET)
MODEL_NAME = args.model
NUM_IMG = args.num_img

#get model and data loader
model = get_model(MODEL_NAME,DATASET,MODEL_DIR)
val_loader,NUM_IMG,ds_config = get_data_loader(DATASET,DATA_DIR,model,batch_size=1,num_img=NUM_IMG,train=False)

device = 'cuda' 
model = model.to(device)
model.eval()
cudnn.benchmark = True

# generate the mask set
mask_list,MASK_SIZE,MASK_STRIDE = gen_mask_set(args,ds_config)


clean_corr = 0
time_list = []

for data,labels in tqdm(val_loader):
    data=data.to(device)
    labels = labels.numpy()
    start = time.time()
    preds = double_masking(data,mask_list,model)
    #preds = challenger_masking(data,mask_list,model)
    end = time.time()
    time_list.append(end-start)
    clean_corr += np.sum(preds==labels)

    
print("Clean accuracy with defense:",clean_corr/NUM_IMG)
print('per-example infernece time',np.sum(time_list)/NUM_IMG)