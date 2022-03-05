# you are encouraged to read `notes_on_mr.md` first :)
import torch
import torch.backends.cudnn as cudnn

import numpy as np 
import os 
import argparse
import time
from tqdm import tqdm
import joblib

from utils.setup import get_model,get_data_loader

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
parser.add_argument("--dump_dir",default='dump/dump_mr',type=str,help='directory to dump two-mask predictions')
parser.add_argument("--override",action='store_true',help='override dumped file')
parser.add_argument("--mr",default=2,type=int,help="we will use (mr+1)x(mr+1) prediction to generate one vote (default value is 2 since the originial MR uses 3x3 grids)")

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
num_classes = 1000 if args.dataset == 'imagenet' else 10

def gen_mask_set_mr(args,ds_config,mr=2):

    # generate mask set
    #assert args.mask_stride * args.num_mask < 0 #can only set either mask_stride or num_mask
    assert args.mask_stride > 0
    IMG_SIZE = (ds_config['input_size'][1],ds_config['input_size'][2])

    if args.pa>0 and args.pb>0: #rectangle patch
        PATCH_SIZE = (args.pa,args.pb)
    else: #square patch
        PATCH_SIZE = (args.patch_size,args.patch_size)

    if args.mask_stride>0: #use specified mask stride
        MASK_STRIDE = (args.mask_stride,args.mask_stride)

    # calculate mask size

    MASK_SIZE = (min(PATCH_SIZE[0]+MASK_STRIDE[0]*(mr+1)-1,IMG_SIZE[0]),min(PATCH_SIZE[1]+MASK_STRIDE[1]*(mr+1)-1,IMG_SIZE[1]))

    mask_list = []
    idx_list1 = list(range(0,IMG_SIZE[0] - MASK_SIZE[0] + 1,MASK_STRIDE[0]))
    if (IMG_SIZE[0] - MASK_SIZE[0])%MASK_STRIDE[0]!=0:
        idx_list1.append(IMG_SIZE[0] - MASK_SIZE[0])

    idx_list1 = [-j*MASK_STRIDE[0] for j in range(-mr,0)]+idx_list1+[IMG_SIZE[0] - MASK_SIZE[0]+j*MASK_STRIDE[0] for j in range(1,mr+1)]

    idx_list2 = list(range(0,IMG_SIZE[1] - MASK_SIZE[1] + 1,MASK_STRIDE[1]))
    if (IMG_SIZE[1] - MASK_SIZE[1])%MASK_STRIDE[1]!=0:
        idx_list2.append(IMG_SIZE[1] - MASK_SIZE[1])
    idx_list2 = [-j*MASK_STRIDE[1] for j in range(-mr,0)]+idx_list2+[IMG_SIZE[1] - MASK_SIZE[1]+j*MASK_STRIDE[1] for j in range(1,mr+1)]

    for x in idx_list1:
        for y in idx_list2:
            mask = torch.ones([1,1,IMG_SIZE[0],IMG_SIZE[1]],dtype=bool).cuda()
            mask[...,max(x,0):min(x+MASK_SIZE[0],IMG_SIZE[0]),max(y,0):min(y+MASK_SIZE[1],IMG_SIZE[1])] = False
            mask_list.append(mask)
    return mask_list,MASK_SIZE,MASK_STRIDE




# generate the mask set
mask_list,MASK_SIZE,MASK_STRIDE = gen_mask_set_mr(args,ds_config,mr=args.mr)
print(len(mask_list))
args.num_mask = int((len(mask_list))**0.5)



def process_minority_report(prediction_map,confidence_map,mr=2):
    num_img = prediction_map.shape[0]
    num_pred =  prediction_map.shape[1]
    voting_grid_pred = np.zeros([num_img,num_pred-mr,num_pred-mr],dtype=int)
    voting_grid_conf = np.zeros([num_img,num_pred-mr,num_pred-mr])
    for a in range(num_img):
        for i in range(num_pred-mr):
            for j in range(num_pred-mr):
                confidence_vec = np.sum(confidence_map[a,i:i+mr+1,j:j+mr+1],axis=(0,1))
                if mr >0:
                    confidence_vec -= np.min(confidence_map[a,i:i+mr+1,j:j+mr+1],axis=(0,1)) 
                    confidence_vec /= (mr+1)**2
                pred = np.argmax(confidence_vec)
                conf = confidence_vec[pred]
                voting_grid_pred[a,i,j]=pred
                voting_grid_conf[a,i,j]=conf
    return voting_grid_pred,voting_grid_conf

# the computation of two-mask predictions is expensive; will dump (or resue the dumped) two-mask predictions.
SUFFIX = '_mr_one_mask_{}_{}_m{}_s{}_mr{}_{}.z'.format(DATASET,MODEL_NAME,MASK_SIZE,MASK_STRIDE,args.mr,NUM_IMG)
#SUFFIX = '_mr_one_mask_{}_{}_m{}_s{}_{}.z'.format(DATASET,MODEL_NAME,MASK_SIZE,MASK_STRIDE,NUM_IMG)
if not args.override and os.path.exists(os.path.join(DUMP_DIR,'prediction_map_list'+SUFFIX)):
    print('loading two-mask predictions')
    confidence_map_list = joblib.load(os.path.join(DUMP_DIR,'confidence_map_list'+SUFFIX))
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


        #two-mask predictions
        prediction_map = np.zeros([num_img,args.num_mask,args.num_mask],dtype=int)
        confidence_map = np.zeros([num_img,args.num_mask,args.num_mask,num_classes])

        for i,mask in enumerate(mask_list):

            masked_output = model(torch.where(mask,data,torch.tensor(0.).cuda()))
            masked_output = torch.nn.functional.softmax(masked_output,dim=1)
            _, masked_pred = masked_output.max(1)
            masked_pred = masked_pred.detach().cpu().numpy()
            masked_conf = masked_output.detach().cpu().numpy()

            a,b = divmod(i,args.num_mask)
            prediction_map[:,a,b] = masked_pred
            confidence_map[:,a,b,:] = masked_conf  
    
        prediction_map,confidence_map = process_minority_report(prediction_map,confidence_map,mr=args.mr)
                
        #vanilla predictions
        clean_output = model(data)
        clean_conf, clean_pred = clean_output.max(1)  
        clean_pred = clean_pred.detach().cpu().numpy()
        orig_prediction_list.append(clean_pred)
        prediction_map_list.append(prediction_map)
        confidence_map_list.append(confidence_map)
        label_list.append(labels)
    
    prediction_map_list = np.concatenate(prediction_map_list)
    confidence_map_list = np.concatenate(confidence_map_list)
    orig_prediction_list = np.concatenate(orig_prediction_list)
    label_list = np.concatenate(label_list)

    joblib.dump(confidence_map_list,os.path.join(DUMP_DIR,'confidence_map_list'+SUFFIX))
    joblib.dump(prediction_map_list,os.path.join(DUMP_DIR,'prediction_map_list'+SUFFIX))
    joblib.dump(orig_prediction_list,os.path.join(DUMP_DIR,'orig_prediction_list_{}_{}_{}.z'.format(DATASET,MODEL_NAME,NUM_IMG)))
    joblib.dump(label_list,os.path.join(DUMP_DIR,'label_list_{}_{}_{}.z'.format(DATASET,MODEL_NAME,NUM_IMG)))



def provable_detection(prediction_map,confidence_map,label,orig_pred,tau):
    if orig_pred != label: # clean prediction is incorrect
        return 0,0 # 0 for incorrect clean prediction
    clean = 1
    provable = 1
    if np.any(np.logical_or(prediction_map!=label,confidence_map<tau)):
        provable = 0
    if np.any(np.logical_and(prediction_map!=label,confidence_map>tau)):
        clean = 0
    return provable,clean


clean_list = []
robust_list = []
for tau in np.arange(0.,1.,0.1):
    print(tau)
    clean_corr = 0
    robust_cnt = 0 
    vanilla_corr =0 
    for i,(prediction_map,confidence_map,label,orig_pred) in enumerate(zip(prediction_map_list,confidence_map_list,label_list,orig_prediction_list)):
        #print(confidence_map)
        provable,clean = provable_detection(prediction_map,confidence_map,label,orig_pred,tau)
        robust_cnt+=provable
        clean_corr+=clean
        vanilla_corr +=orig_pred==label
    robust_list.append(robust_cnt/NUM_IMG)
    clean_list.append(clean_corr/NUM_IMG)
    print("Provable robust accuracy ({}):".format(tau),robust_cnt/NUM_IMG)
    print("Clean accuracy with defense:",clean_corr/NUM_IMG)
    print("Clean accuracy without defense:",vanilla_corr/NUM_IMG)
    print()
print('clean_list=', [100*x for x in clean_list])
print('robust_list=',[100*x for x in robust_list])