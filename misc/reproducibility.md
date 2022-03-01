## Overview

This document provides a detailed guide to reproduce all experimental results in the main body of our PatchCleanser paper. 

## Setup

#### File directory 
Below is an overview of relevant files for the artifact evaluation. Please organize the files within the specified structure.
```shell
├── requirement.txt                  #required package
├── pc_certification.py              #PatchCleanser: certify robustness via two-mask correctness 
├── pc_clean_acc.py                  #PatchCleanser: evaluate clean accuracy and per-example inference time
| 
├── vanilla_clean_acc.py             #undefended vanilla models: evaluate clean accuracy and per-example inference time
| 
├── utils
|   ├── setup.py                     #utils for constructing models and data loaders
|   ├── defense.py                   #utils for PatchCleanser defenses
|   └── cutout.py                    #utils for masked model training
|
├── misc
|   ├── pc_mr.py                     #script for minority report (Figure 9)
|   └── pc_multiple.py               #script for multiple patch shapes and multiple patches (Table 4)
| 
├── data   
|   ├── imagenet                     #data directory for imagenet
|   |   └── val                      #imagenet validation set
|   ├── imagenette                   #data directory for imagenette
|   |   └── val                      #imagenette validation set
|   └── cifar                        #data directory for cifar-10
|
└── checkpoints                      #directory for checkpoints
    ├── README.md                    #details of checkpoints
    └── ...                          #model checkpoints
```
#### Dependency
Install [PyTorch](https://pytorch.org/get-started/locally/) with GPU support.

Install other packages `pip install -r requirement.txt`.

#### Datasets
- [ImageNet](https://image-net.org/download.php) (ILSVRC2012) - requires manual download; also available on [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)
- [ImageNette](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz) - requires manual download
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) - will be downloaded automatically within our code
Move manually downloaded data to the directory `data/`.

#### Checkpoints
1. Download the following checkpoints from the Google Drive [link](https://drive.google.com/drive/folders/1Ewks-NgJHDlpeAaGInz_jZ6iczcYNDlN?usp=sharing).

```
resnetv2_50x1_bit_distilled_cutout2_128_imagenet.pth
resnetv2_50x1_bit_distilled_cutout2_128_imagenette.pth
resnetv2_50x1_bit_distilled_cutout2_128_cifar.pth
resmlp_24_distilled_224_cutout2_128_imagenet.pth
resmlp_24_distilled_224_cutout2_128_imagenette.pth
resmlp_24_distilled_224_cutout2_128_cifar.pth
vit_base_patch16_224_cutout2_128_imagenet.pth
vit_base_patch16_224_cutout2_128_imagenette.pth
vit_base_patch16_224_cutout2_128_cifar.pth
```

2. Move downloaded weights to the directory `checkpoints/`.

## Experiments

In this section, we list all the shell commands used for getting experimental results for every table and figure.

1. Our evaluation metrics are **clean accuracy** and **certified robust accuracy**, which will be outputted to the console. Below is the expected output after running `python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img -1 --num_mask 6 --patch_size 32`. In this example, the clean accuracy of PatchCleanser defense is 83.9%, and the certified robust accuracy of PatchCleanser defense is 62.1%. These numbers match the results reported in Table 2 (ImageNet, PC-ViT, 2%-pixel patch).

```
Certified robust accuracy: 0.6207
Clean accuracy with defense: 0.8394
```

2. We also specified the estimated runtime (with one GPU) for each experiment below. 
   - Running experiments for the entire dataset can take a long time. Please feel free to set ``--num_img`` to a small positive integer (e.g., 1000) to run experiments on a subset of the dataset. This will give an approximated evaluation result.
   - When ``--num_img`` is set to a negative integer, we will use the entire dataset for experiments; when it is set to a positive integer, we will use a random subset (with ``num_img`` images) for experiments.

#### Table 2 (and Figure 2): our main evaluation results

The following scripts allow us to obtain results for our main evaluation in Table 2 and Figure 2.

```shell
# please feel free to set --num_img to a small positive integer (e.g., 1000) to reduce runtime; the script will run experiments on a random subset of the dataset to get an approximated result.

#### imagenette 
# resnet (each takes 1-2hrs)
python pc_certification.py --model resnetv2_50x1_bit_distilled_cutout2_128  --dataset imagenette --num_img -1  --num_mask 6 --patch_size 32 # 2% pixel patch
python pc_certification.py --model resnetv2_50x1_bit_distilled_cutout2_128  --dataset imagenette --num_img -1  --num_mask 6 --patch_size 39 # 3% pixel patch
python pc_certification.py --model resnetv2_50x1_bit_distilled_cutout2_128  --dataset imagenette --num_img -1  --num_mask 6 --patch_size 23 # 1% pixel patch
# mlp (each takes 1-2hrs)
python pc_certification.py --model resmlp_24_distilled_224_cutout2_128  --dataset imagenette --num_img -1  --num_mask 6 --patch_size 32 
python pc_certification.py --model resmlp_24_distilled_224_cutout2_128  --dataset imagenette --num_img -1  --num_mask 6 --patch_size 39 
python pc_certification.py --model resmlp_24_distilled_224_cutout2_128  --dataset imagenette --num_img -1  --num_mask 6 --patch_size 23 
# vit (each takes 3-4hrs)
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenette --num_img -1 --num_mask 6 --patch_size 32 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenette --num_img -1 --num_mask 6 --patch_size 39 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenette --num_img -1 --num_mask 6 --patch_size 23 

#### imagenet 
# resnet (each takes 16-17hrs)
python pc_certification.py --model resnetv2_50x1_bit_distilled_cutout2_128  --dataset imagenet --num_img -1  --num_mask 6 --patch_size 32 
python pc_certification.py --model resnetv2_50x1_bit_distilled_cutout2_128  --dataset imagenet --num_img -1  --num_mask 6 --patch_size 39 
python pc_certification.py --model resnetv2_50x1_bit_distilled_cutout2_128  --dataset imagenet --num_img -1  --num_mask 6 --patch_size 23 
# mlp (each takes 16-17hrs)
python pc_certification.py --model resmlp_24_distilled_224_cutout2_128  --dataset imagenet --num_img -1  --num_mask 6 --patch_size 32 
python pc_certification.py --model resmlp_24_distilled_224_cutout2_128  --dataset imagenet --num_img -1  --num_mask 6 --patch_size 39 
python pc_certification.py --model resmlp_24_distilled_224_cutout2_128  --dataset imagenet --num_img -1  --num_mask 6 --patch_size 23 
# vit (each takes 38-40hrs)
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img -1 --num_mask 6 --patch_size 32 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img -1 --num_mask 6 --patch_size 39 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img -1 --num_mask 6 --patch_size 23 

#### cifar
# resnet (each takes 4-5hrs)
python pc_certification.py --model resnetv2_50x1_bit_distilled_cutout2_128  --dataset cifar --num_img -1  --num_mask 6 --patch_size 35  # 2.4% pixel patch
python pc_certification.py --model resnetv2_50x1_bit_distilled_cutout2_128  --dataset cifar --num_img -1  --num_mask 6 --patch_size 14  # 0.4% pixel patch
# mlp (each takes 4-5hrs)
python pc_certification.py --model resmlp_24_distilled_224_cutout2_128  --dataset cifar --num_img -1  --num_mask 6 --patch_size 35 
python pc_certification.py --model resmlp_24_distilled_224_cutout2_128  --dataset cifar --num_img -1  --num_mask 6 --patch_size 14 
# vit (each takes 11-12hrs)
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset cifar --num_img -1 --num_mask 6 --patch_size 35 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset cifar --num_img -1 --num_mask 6 --patch_size 14 

```



#### Table 3: vanilla undefended models

The following script is used for Table 3 (clean accuracy of vanilla undefended model).

```shell
# takes a few minutes...
# imagenette
python vanilla_clean_acc.py --model resnetv2_50x1_bit_distilled_cutout2_128  --dataset imagenette --num_img -1  
python vanilla_clean_acc.py --model resmlp_24_distilled_224_cutout2_128  --dataset imagenette --num_img -1  
python vanilla_clean_acc.py --model vit_base_patch16_224_cutout2_128  --dataset imagenette --num_img -1  

#imagenet
python vanilla_clean_acc.py --model resnetv2_50x1_bit_distilled_cutout2_128  --dataset imagenet --num_img -1 
python vanilla_clean_acc.py --model resmlp_24_distilled_224_cutout2_128  --dataset imagenet --num_img -1  
python vanilla_clean_acc.py --model vit_base_patch16_224_cutout2_128  --dataset imagenet --num_img -1  

#cifar
python vanilla_clean_acc.py --model resnetv2_50x1_bit_distilled_cutout2_128  --dataset cifar --num_img -1  
python vanilla_clean_acc.py --model resmlp_24_distilled_224_cutout2_128  --dataset cifar --num_img -1  
python vanilla_clean_acc.py --model vit_base_patch16_224_cutout2_128  --dataset cifar --num_img -1  

```



#### Figure 4: defense with different numbers of masks $k$

The following script is used for Figure 4 (the effect of number of masks $k$). Note that we choose 5000 random ImageNet images and evaluate for 32x32 patches (2% pixels)

```shell
# each takes ~0.5hr
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 2 --patch_size 32 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 3 --patch_size 32 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 4 --patch_size 32 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 5 --patch_size 32 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 6 --patch_size 32 
```



#### Figure 5: defense runtime

The following script evaluates per-example runtime (Figure 8)

```shell
# each takes a few minutes
python pc_clean_acc.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 2 --patch_size 32 
python pc_clean_acc.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 3 --patch_size 32 
python pc_clean_acc.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 4 --patch_size 32 
python pc_clean_acc.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 5 --patch_size 32 
python pc_clean_acc.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 6 --patch_size 32 

```



#### Figure 6, 7, 8: defense with different (estimated) patch sizes

The following script is used for evaluating defense performance with different patch sizes (or estimated patch sizes), results are plotted in Figure 5, 6, 7. 

```shell
# each takes ~0.5hr
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 6 --patch_size 48 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 6 --patch_size 64 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 6 --patch_size 80 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 6 --patch_size 96 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 6 --patch_size 112 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 6 --patch_size 128 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 6 --patch_size 144 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 6 --patch_size 160 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 6 --patch_size 176 

python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 6 --patch_size 40 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 6 --patch_size 56 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 6 --patch_size 72 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 6 --patch_size 88 
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000 --num_mask 6 --patch_size 104 
```



#### Table 4: multiple shapes and multiple patches

The following script evaluates defense performance for all 1% rectangular pixel patches and two 1% square patches (Table 4)

```shell
# move the script to the main directory
mv misc/pc_multiple.py pc_multiple.py 

# multiple shapes for 1%-pixel patch
# each takes 11-12 hrs
python pc_multiple.py --mode shape --dataset imagenet --model vit_base_patch16_224_cutout2_128 --num_img 500 --mask_stride 32 --patch_size 23 
python pc_multiple.py --mode shape --dataset imagenette --model vit_base_patch16_224_cutout2_128 --num_img 500 --mask_stride 32 --patch_size 23 
python pc_multiple.py --mode shape --dataset cifar --model vit_base_patch16_224_cutout2_128 --num_img 500 --mask_stride 32 --patch_size 23 

# two 1%-pixel patches
# each takes 100hrs; setting --num_mask to a smaller number (e.g., 4) can significantly reduce runtime (at the cost of performance drops)
python pc_multiple.py --mode twopatch --dataset imagenet --model vit_base_patch16_224_cutout2_128 --num_img 500 --num_mask 5 --patch_size 23 
python pc_multiple.py --mode twopatch --dataset imagenette --model vit_base_patch16_224_cutout2_128 --num_img 500 --num_mask 5 --patch_size 23 
python pc_multiple.py --mode twopatch --dataset cifar --model vit_base_patch16_224_cutout2_128 --num_img 500 --num_mask 5 --patch_size 23 
```



#### Figure 9: Minority Report

The following script evaluates defense performance for Minority Report using our mask set (Figure 9).

```shell
# move the script to the main directory
mv misc/pc_mr.py pc_mr.py 
# each takes 3-4 hrs
python pc_mr.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img -1 --mask_stride 25 --patch_size 32 # the number of masks is 6x6=36
```

