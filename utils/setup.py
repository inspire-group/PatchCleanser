import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import datasets
import numpy as np 
import os 


NUM_CLASSES_DICT = {'imagenette':10,'imagenet':1000,'flower102':102,'cifar':10,'cifar100':100,'svhn':10}

def get_model(model_name,dataset_name,model_dir):

    #build model and load weights

    '''
    INPUT:
    model_name      str, model name. The name should contrain one of ('resnetv2_50x1_bit_distilled', 'vit_base_patch16_224','resmlp_24_distilled_224')
    dataset_name    str, dataset name.  One of ('imagenette','imagenet','cifar','cifar100','svhn','flower102')  
    model_dir       str, the directory of model checkpoints

    OUTPUT:
    model           torch.nn.Module, the PyToch model with weights loaded
    '''

    timm_pretrained = (dataset_name == 'imagenet') and ('cutout' not in model_name)
    if 'resnetv2_50x1_bit_distilled' in model_name:
        model = timm.create_model('resnetv2_50x1_bit_distilled', pretrained=timm_pretrained)
    elif 'vit_base_patch16_224' in model_name:
        model = timm.create_model('vit_base_patch16_224', pretrained=timm_pretrained)
    elif 'resmlp_24_distilled_224' in model_name:
        model = timm.create_model('resmlp_24_distilled_224', pretrained=timm_pretrained)
    elif 'mae_finetuned_vit_base' in model_name:
        model = timm.create_model('vit_base_patch16_224', pretrained=False,global_pool='avg')
        timm_pretrained = False
        del model.pretrained_cfg['mean']
        del model.pretrained_cfg['std']
        model.pretrained_cfg['crop_pct'] = 224/256


    # modify classification head and load model weight
    if not timm_pretrained: 
        model.reset_classifier(num_classes=NUM_CLASSES_DICT[dataset_name])
        checkpoint_name = model_name + '_{}.pth'.format(dataset_name) 
        checkpoint = torch.load(os.path.join(model_dir,checkpoint_name))
        if 'mae' not in model_name:
            model.load_state_dict(checkpoint['state_dict']) 
        else:
            msg = model.load_state_dict(checkpoint['model'],strict=False) 
            print(msg)
            

    return model


def get_data_loader(dataset_name,data_dir,model,batch_size=1,num_img=-1,train=False):

    # get the data loader (possibly only a subset of the dataset) 

    '''
    INPUT:
    dataset_name    str, dataset name.  One of ('imagenette','imagenet','cifar','cifar100','svhn','flower102')  
    data_dir        str, the directory of data 
    model_name      str, model name. The name should contrain one of ('resnetv2_50x1_bit_distilled', 'vit_base_patch16_224','resmlp_24_distilled_224')
    model           torch.nn.Module / timm.models, the built model returned by get_model(), which has an attribute of default_cfg for data preprocessing
    batch_size      int, batch size. default value is 1 for per-example inference time evaluation. In practice, a larger batch size is preferred 
    num_img         int, number of images to construct a random image subset. if num_img<0, we return a data loader for the entire dataset
    train           bool, whether to return the training data split. 

    OUTPUT:
    loader          the PyToch data loader
    len(dataset)    the size of dataset
    config          data preprocessing configuration dict
    '''

    ### !!!!!ATTN!!!!!! Do not pass a DataParallel instance as `model`. 
    ### The DataParallel wrap makes the `default_cfg` invisible to `resolve_data_config` and might returns a incompatiable data preprocessing pipeline
    ### you can add the DataParallel wrapper after calling `get_data_loader`
    if isinstance(model,(torch.nn.DataParallel)):
        model = model.module


    # get dataset
    if dataset_name in ['imagenette','imagenet','flower102']:
        #high resolution images; use the default image preprocessing (all three models use 224x224 inputs)
        config = resolve_data_config({}, model=model)
        print(config)
        ds_transforms = create_transform(**config)
        split = 'train' if train else 'val'
        dataset_ = datasets.ImageFolder(os.path.join(data_dir,split),ds_transforms) 
    elif dataset_name in ['cifar','cifar100','svhn']: 
        #low resolution images; resize them to 224x224 without cropping
        config = resolve_data_config({'crop_pct':1}, model=model)
        ds_transforms = create_transform(**config)
        if dataset_name == 'cifar':
            dataset_ = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=ds_transforms) 
        elif dataset_name == 'cifar100':
            dataset_ = datasets.CIFAR100(root=data_dir, train=train, download=True, transform=ds_transforms)  
        elif dataset_name == 'svhn':
            split = 'train' if train else 'test'
            dataset_ = datasets.SVHN(root=data_dir, split=split, download=True, transform=ds_transforms) 

    # select a random set of test images (when args.num_img>0)
    np.random.seed(233333333)#random seed for selecting test images
    idxs=np.arange(len(dataset_))
    np.random.shuffle(idxs)
    if num_img>0:
        idxs=idxs[:num_img]
    dataset = torch.utils.data.Subset(dataset_, idxs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=train,num_workers=2)

    return loader,len(dataset),config

