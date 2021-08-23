import torch
import torch.backends.cudnn as cudnn

import numpy as np 
import os 
import argparse
import time
from tqdm import tqdm

from utils.setup import get_model,get_data_loader

parser = argparse.ArgumentParser()

parser.add_argument("--model_dir",default='checkpoints',type=str,help="directory of checkpoints")
parser.add_argument('--data_dir', default='data', type=str,help="directory of data")
parser.add_argument('--dataset', default='imagenette',type=str,choices=('imagenette','imagenet','cifar','cifar100','svhn','flower102'),help="dataset")
parser.add_argument("--model",default='vit_base_patch16_224',type=str,help="model name")
parser.add_argument("--num_img",default=-1,type=int,help="number of randomly selected images for this experiment (-1: using the all images)")

args = parser.parse_args()

DATASET = args.dataset
MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir,DATASET)
MODEL_NAME = args.model
NUM_IMG = args.num_img

#get model and data loader
model = get_model(MODEL_NAME,DATASET,MODEL_DIR)
val_loader,NUM_IMG,_ = get_data_loader(DATASET,DATA_DIR,model,batch_size=16,num_img=NUM_IMG,train=False)

device = 'cuda' 
model = model.to(device)
model.eval()
cudnn.benchmark = True

accuracy_list=[]
time_list=[]
for data,labels in tqdm(val_loader):
	data,labels=data.to(device),labels.to(device)
	start = time.time()
	output_clean = model(data)
	end=time.time()
	time_list.append(end-start)
	acc_clean=torch.sum(torch.argmax(output_clean, dim=1) == labels).item()#cpu().detach().numpy()
	accuracy_list.append(acc_clean)
	
print("Test accuracy:",np.sum(accuracy_list)/NUM_IMG)
print('Per-example inference time:',np.sum(time_list)/NUM_IMG)

