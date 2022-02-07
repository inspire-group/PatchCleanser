import numpy as np 
import torch 

def gen_mask_set(args,ds_config,mr=False):
    # generate a R-covering mask set
    '''
    INPUT:
    args            argparse.Namespace, the set of argumements/hyperparamters for mask set generation
    ds_config       dict, data preprocessing dict   

    OUTPUT:
    mask_list       list of torch.tensor, the generation R-covering mask set, the binary masks are moved to CUDA
    MASK_SIZE       tuple (int,int), the mask size along two axes
    MASK_STRIDE     tuple (int,int), the mask stride along two axes
    '''
    # generate mask set
    assert args.mask_stride * args.num_mask < 0 #can only set either mask_stride or num_mask

    IMG_SIZE = (ds_config['input_size'][1],ds_config['input_size'][2])

    if args.pa>0 and args.pb>0: #rectangle patch
        PATCH_SIZE = (args.pa,args.pb)
    else: #square patch
        PATCH_SIZE = (args.patch_size,args.patch_size)

    if args.mask_stride>0: #use specified mask stride
        MASK_STRIDE = (args.mask_stride,args.mask_stride)
    else: #calculate mask stride based on the computation budget
        MASK_STRIDE = (int(np.ceil((IMG_SIZE[0] - PATCH_SIZE[0] + 1)/args.num_mask)),int(np.ceil((IMG_SIZE[1] - PATCH_SIZE[1] + 1)/args.num_mask)))

    # calculate mask size
    if mr:
        MASK_SIZE = (min(PATCH_SIZE[0]+MASK_STRIDE[0]*3-1,IMG_SIZE[0]),min(PATCH_SIZE[1]+MASK_STRIDE[1]*3-1,IMG_SIZE[1]))
    else:
        MASK_SIZE = (min(PATCH_SIZE[0]+MASK_STRIDE[0]-1,IMG_SIZE[0]),min(PATCH_SIZE[1]+MASK_STRIDE[1]-1,IMG_SIZE[1]))

    mask_list = []
    idx_list1 = list(range(0,IMG_SIZE[0] - MASK_SIZE[0] + 1,MASK_STRIDE[0]))
    if (IMG_SIZE[0] - MASK_SIZE[0])%MASK_STRIDE[0]!=0:
        idx_list1.append(IMG_SIZE[0] - MASK_SIZE[0])
    idx_list2 = list(range(0,IMG_SIZE[1] - MASK_SIZE[1] + 1,MASK_STRIDE[1]))
    if (IMG_SIZE[1] - MASK_SIZE[1])%MASK_STRIDE[1]!=0:
        idx_list2.append(IMG_SIZE[1] - MASK_SIZE[1])

    for x in idx_list1:
        for y in idx_list2:
            mask = torch.ones([1,1,IMG_SIZE[0],IMG_SIZE[1]],dtype=bool).cuda()
            mask[...,x:x+MASK_SIZE[0],y:y+MASK_SIZE[1]] = False
            mask_list.append(mask)
    return mask_list,MASK_SIZE,MASK_STRIDE


def double_masking(data,mask_list,model):
    # perform double masking inference with the input image, the mask set, and the undefended model
    '''
    INPUT:
    data            torch.Tensor [B,C,W,H], a batch of data
    mask_list       a list of torch.Tensor, R-covering mask set
    model           torch.nn.module, the vanilla undefended model

    OUTPUT:
    output_pred     numpy.ndarray, the prediction labels
    '''

    # first-round masking 
    num_img = len(data)
    num_mask = len(mask_list)
    pred_one_mask_batch = np.zeros([num_img,num_mask],dtype=int)
    # compute one-mask prediction in batch
    for i,mask in enumerate(mask_list):
        masked_output = model(torch.where(mask,data,torch.tensor(0.).cuda()))
        _, masked_pred = masked_output.max(1)
        masked_pred = masked_pred.detach().cpu().numpy()
        pred_one_mask_batch[:,i] = masked_pred

    # determine the prediction label for each image
    output_pred = np.zeros([num_img],dtype=int)
    for j in range(num_img):
        pred_one_mask = pred_one_mask_batch[j]
        pred,cnt = np.unique(pred_one_mask,return_counts=True)

        if len(pred)==1: # unanimous agreement in the first-round masking
            defense_pred = pred[0] # Case I: agreed prediction
        else:
            sorted_idx = np.argsort(cnt)
            # get majority prediction and disagreer prediction
            majority_pred = pred[sorted_idx][-1]
            disagreer_pred = pred[sorted_idx][:-1]

            # second-round masking
            # get index list of the disagreer mask            
            tmp = np.zeros_like(pred_one_mask,dtype=bool)
            for dis in disagreer_pred:
                tmp = np.logical_or(tmp,pred_one_mask==dis)
            disagreer_pred_mask_idx = np.where(tmp)[0]

            for i in disagreer_pred_mask_idx:
                dis = pred_one_mask[i]
                mask = mask_list[i]
                flg=True
                for mask2 in mask_list:
                    # evaluate two-mask predictions
                    masked_output = model(torch.where(torch.logical_and(mask,mask2),data[j],torch.tensor(0.).cuda()))
                    masked_conf, masked_pred = masked_output.max(1)
                    masked_pred = masked_pred.item()
                    if masked_pred!=dis: # disagreement in the second-round masking -> discard the disagreer
                        flg=False
                        break
                if flg:
                    defense_pred = dis # Case II: disagreer prediction
                    break
            if not flg:
                defense_pred = majority_pred # Case III: majority prediction
        output_pred[j] = defense_pred
    return output_pred


def double_masking_precomputed(prediction_map):
    # perform double masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label 
    '''
    # first-round masking
    pred_one_mask = np.diag(prediction_map)
    pred,cnt = np.unique(pred_one_mask,return_counts=True)

    if len(pred) == 1: # unanimous agreement in the first-round masking
        return pred[0] # Case I: agreed prediction

    # get majority prediction and disagreer prediction
    sorted_idx = np.argsort(cnt)
    majority_pred = pred[sorted_idx][-1]
    disagreer_pred = pred[sorted_idx][:-1]

    # second-round masking
    # get index list of the disagreer mask
    tmp = np.zeros_like(pred_one_mask,dtype=bool)
    for dis in disagreer_pred:
        tmp = np.logical_or(tmp,pred_one_mask==dis)
    disagreer_pred_mask_idx = np.where(tmp)[0]

    for i in disagreer_pred_mask_idx:
        dis = pred_one_mask[i]
        # check all two-mask predictions
        tmp = prediction_map[i]==dis 
        if np.all(tmp):
            return dis # Case II: disagreer prediction

    return majority_pred # Case III: majority prediction

def certify_precomputed(prediction_map,label):
    # certify the robustness with pre-computed two mask prediction
    # check for two-mask correctness
    return np.all(prediction_map==label)




def challenger_masking(data,mask_list,model):
    # perform challenger masking inference (discussed in the appendix) with the input image, the mask set, and the undefended model
    '''
    INPUT:
    data            torch.Tensor [B,C,W,H], a batch of data
    mask_list       a list of torch.Tensor, R-covering mask set
    model           torch.nn.module, the vanilla undefended model

    OUTPUT:
    output_pred     numpy.ndarray, the prediction labels
    '''

    # first-round masking 
    num_img = len(data)
    num_mask = len(mask_list)
    pred_one_mask_batch = np.zeros([num_img,num_mask],dtype=int)
    # compute one-mask prediction in batch
    for i,mask in enumerate(mask_list):
        masked_output = model(torch.where(mask,data,torch.tensor(0.).cuda()))
        _, masked_pred = masked_output.max(1)
        masked_pred = masked_pred.detach().cpu().numpy()
        pred_one_mask_batch[:,i] = masked_pred

    # determine the prediction label for each image
    output_pred = np.zeros([num_img],dtype=int)
    for j in range(num_img):
        pred_one_mask = pred_one_mask_batch[j]
        pred,cnt = np.unique(pred_one_mask,return_counts=True)

        if len(pred)==1: # unanimous agreement in the first-round masking
            candidate_label = pred[0] 
        else:
            # second-round masking (challenger game)
            candidate = 0 # take the index 0 as the winner candidate
            candidate_label = pred_one_mask[candidate]
            candidate_mask = mask_list[candidate]
            used_flg = np.zeros([num_mask],dtype=bool) 
            #used_flg[candidate_mask]=True
            while len(np.unique(pred_one_mask[~used_flg]))>1:
                # find a challenger
                for challenger in range(0,num_mask):
                    if used_flg[challenger]:
                        continue
                    challenger_label = pred_one_mask[challenger]
                    if challenger_label==candidate_label:
                        continue
                    break
                # challenger game
                challenger_mask = mask_list[challenger] 
                masked_output = model(torch.where(torch.logical_and(candidate_mask,challenger_mask),data[j],torch.tensor(0.).cuda()))
                _, masked_pred = masked_output.max(1)
                masked_pred = masked_pred.item()
                if masked_pred == challenger_label:
                    used_flg[candidate]=True
                    candidate = challenger
                    candidate_label = challenger_label
                    candidate_mask = challenger_mask
                else:
                    used_flg[challenger]=True
        output_pred[j] = candidate_label
    return output_pred


def challenger_masking_precomputed(prediction_map):
    # perform challenger masking inference with the pre-computed two-mask predictions
    '''
    INPUT:
    prediction_map  numpy.ndarray [num_mask,num_mask], the two-mask prediction map for a single data point

    OUTPUT:         int, the prediction label 
    '''
    # first-round masking
    pred_one_mask = np.diag(prediction_map)
    pred,cnt = np.unique(pred_one_mask,return_counts=True)

    if len(pred) == 1: # unanimous agreement in the first-round masking
        candidate_label = pred[0] 
    else:
        # second-round masking (challenger game)
        candidate = 0 # take the index 0 as the winner candidate
        candidate_label = pred_one_mask[candidate]
        num_mask = len(pred_one_mask)
        used_flg = np.zeros([num_mask],dtype=bool) 
        while len(np.unique(pred_one_mask[~used_flg]))>1:
            # find a challenger
            for challenger in range(0,num_mask):
                if used_flg[challenger]:
                    continue
                challenger_label = pred_one_mask[challenger]
                if challenger_label==candidate_label:
                    continue
                break
            # challenger game
            masked_pred = prediction_map[candidate,challenger]
            if masked_pred == challenger_label:
                used_flg[candidate]=True
                candidate = challenger
                candidate_label = challenger_label
            else:
                used_flg[challenger]=True
    return candidate_label
