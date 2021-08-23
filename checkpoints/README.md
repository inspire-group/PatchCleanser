## Checkpoints
Model checkpoints used in the paper can be downloaded from [link](https://drive.google.com/drive/folders/1Ewks-NgJHDlpeAaGInz_jZ6iczcYNDlN?usp=sharing).

Model training should be very easy with the provided training scripts. 

#### checkpoint name format:

`{model_name}_{dataset_name}.pth`

`{model_name}` options:

1. `resnetv2_50x1_bit_distilled`
2. `vit_base_patch16_224`
3. `resmlp_24_distilled_224`

`{dataset_name}` options:

1. `imagenet`
2. `imagenette`
3. `cifar`
4. `cifar100`
5. `flower102`
6. `svhn`

**Note 1:** We do not have weights for ImageNet; the pretrained weights can be loaded using `timm`.

**Note 2:** Models that use *masked model training* has a slightly different name:

`{model_name}_masked_{dataset_name}.pth`

**Note 3:** `'_{dataset_name}.pth'` will be automatically appended to `args.model` in the scripts `pc_certification.py`, `pc_clean_acc.py`, and `vanilla_clean_acc.py`.


