import torch
import torch.backends.cudnn as cudnn

import numpy as np 
import os 
import argparse
import time
from tqdm import tqdm
import joblib

from utils.setup import get_model,get_data_loader
from utils.defense import gen_mask_set,double_masking_precomputed,certify_precomputed

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
parser.add_argument("--dump_dir",default='dump',type=str,help='directory to dump two-mask predictions')
parser.add_argument("--override",action='store_true',help='override dumped file')
parser.add_argument("--mode",choices=('shape','twopatch'),type=str,help='either analyze multiple shapes or two patches')

args = parser.parse_args()
DATASET = args.dataset
MODEL_DIR = os.path.join('.',args.model_dir)
DATA_DIR = os.path.join(args.data_dir,DATASET)
DUMP_DIR = os.path.join('.',args.dump_dir)
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)

MODEL_NAME = args.model
NUM_IMG = args.num_img

#get model and data loader
model = get_model(MODEL_NAME,DATASET,MODEL_DIR)
val_loader,NUM_IMG,ds_config = get_data_loader(DATASET,DATA_DIR,model,batch_size=16,num_img=NUM_IMG,train=False)

device = 'cuda' 
model = model.to(device)
model.eval()
cudnn.benchmark = True

#####################################################################################################################################
# generate the mask set
mask_list = []
if args.mode == 'shape':
    for pa,pb in [(5,224),(12,83),(23,38),(39,20),(84,12),(224,5)]:
    #for pa,pb in [(9,224),(16,101),(23,60),(32,42),(42,32),(60,23),(101,16),(224,9)]:
        args.pa=pa
        args.pb=pb
        tmp,MASK_SIZE,MASK_STRIDE = gen_mask_set(args,ds_config)
        mask_list+=tmp 
    SUFFIX = '_two_mask_shape_set_{}_{}_p{}_s{}_{}.z'.format(DATASET,MODEL_NAME,args.patch_size,MASK_STRIDE,NUM_IMG)
elif args.mode == 'twopatch':
    tmp,MASK_SIZE,MASK_STRIDE = gen_mask_set(args,ds_config)
    mask_list = [torch.logical_and(mask1,mask2) for mask1 in tmp for mask2 in tmp]
    SUFFIX = '_two_2mask_{}_{}_p{}_s{}_{}.z'.format(DATASET,MODEL_NAME,args.patch_size,MASK_STRIDE,NUM_IMG)
else:
    raise NotImplementedError
#####################################################################################################################################

print(len(mask_list))
# the computation of two-mask predictions is expensive; will dump (or resue the dumped) two-mask predictions.
if not args.override and os.path.exists(os.path.join(DUMP_DIR,'prediction_map_list'+SUFFIX)):
    print('loading two-mask predictions')
    prediction_map_list = joblib.load(os.path.join(DUMP_DIR,'prediction_map_list'+SUFFIX))
    orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,'orig_prediction_list_{}_{}_{}.z'.format(DATASET,MODEL_NAME,NUM_IMG)))
    label_list = joblib.load(os.path.join(DUMP_DIR,'label_list_{}_{}_{}.z'.format(DATASET,MODEL_NAME,NUM_IMG)))
else:
    print('computing two-mask predictions')
    prediction_map_list = []
    confidence_map_list = []
    label_list = []
    orig_prediction_list = []
    for data,labels in tqdm(val_loader):
        data=data.to(device)
        labels = labels.numpy()
        num_img = data.shape[0]
        num_mask = len(mask_list)

        #two-mask predictions
        prediction_map = np.zeros([num_img,num_mask,num_mask],dtype=int)
        for i,mask in enumerate(mask_list):
            for j in range(i,num_mask):
                mask2 = mask_list[j]
                masked_output = model(torch.where(torch.logical_and(mask,mask2),data,torch.tensor(0.).cuda()))
                _, masked_pred = masked_output.max(1)
                masked_pred = masked_pred.detach().cpu().numpy()
                prediction_map[:,i,j] = masked_pred
                
        #vanilla predictions
        clean_output = model(data)
        clean_conf, clean_pred = clean_output.max(1)  
        clean_pred = clean_pred.detach().cpu().numpy()
        orig_prediction_list.append(clean_pred)
        prediction_map_list.append(prediction_map)
        label_list.append(labels)
    
    prediction_map_list = np.concatenate(prediction_map_list)
    orig_prediction_list = np.concatenate(orig_prediction_list)
    label_list = np.concatenate(label_list)

    joblib.dump(prediction_map_list,os.path.join(DUMP_DIR,'prediction_map_list'+SUFFIX))
    joblib.dump(orig_prediction_list,os.path.join(DUMP_DIR,'orig_prediction_list_{}_{}_{}.z'.format(DATASET,MODEL_NAME,NUM_IMG)))
    joblib.dump(label_list,os.path.join(DUMP_DIR,'label_list_{}_{}_{}.z'.format(DATASET,MODEL_NAME,NUM_IMG)))


clean_corr = 0
robust = 0
orig_corr = 0
for i,(prediction_map,label,orig_pred) in enumerate(zip(prediction_map_list,label_list,orig_prediction_list)):
    prediction_map = prediction_map + prediction_map.T - np.diag(np.diag(prediction_map)) #generate a symmetric matrix from a triangle matrix
    robust += certify_precomputed(prediction_map,label)
    clean_corr += double_masking_precomputed(prediction_map) == label
    orig_corr += orig_pred == label

print("------------------------------")
print("Provable robust accuracy:",robust/NUM_IMG)
print("Clean accuracy with defense:",clean_corr/NUM_IMG)
print("Clean accuracy without defense:",orig_corr/NUM_IMG)

