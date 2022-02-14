## Checkpoints
Model checkpoints used in the paper can be downloaded from [link](https://drive.google.com/drive/folders/1Ewks-NgJHDlpeAaGInz_jZ6iczcYNDlN?usp=sharing).

Model training should be very easy with the provided training scripts; see `example_cmds.sh` for examples.

#### checkpoint name format:

`{model_name}{cutout_setting}_{dataset_name}.pth`

`{model_name}` options:

1. `resnetv2_50x1_bit_distilled`
2. `vit_base_patch16_224`
3. `resmlp_24_distilled_224`

`{cutout_setting}` options:

1. empty 
2. `_cutout2_128` (2 cutout squares of size 128px; the default setting used in the paper)

`{dataset_name}` options:

1. `imagenet`
2. `imagenette`
3. `cifar`
4. `cifar100`
5. `flower102`
6. `svhn`

**Note 1:** We do not have weights for ImageNet; the pretrained weights can be loaded using `timm`.

**Note 2:** `'_{dataset_name}.pth'` will be automatically appended to `args.model` in the scripts `pc_certification.py`, `pc_clean_acc.py`, and `vanilla_clean_acc.py`.

