# certified robust accuracy of patchcleanser models
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img -1  --num_mask 6 --patch_size 32 # a simple usage example

python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenette --num_img -1  --num_mask 6 --patch_size 32 # experiment with a different dataset 
python pc_certification.py --model resnetv2_50x1_bit_distilled --dataset imagenette --num_img -1  --num_mask 6 --patch_size 32 # experiment with a different architecture 
python pc_certification.py --model resnetv2_50x1_bit_distilled --dataset imagenette --num_img -1  --num_mask 6 --patch_size 64 # experiment with a larger patch 
python pc_certification.py --model resnetv2_50x1_bit_distilled --dataset imagenette --num_img 1000  --num_mask 6 --patch_size 32 # experiment with a random subset of images 

python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenette --num_img -1  --num_mask 3 --patch_size 32 # adjust computation budget (number of masks)
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenette --num_img -1  --mask_stride 32 --patch_size 32 # set mask_stride instead of num_mask
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenette --num_img -1  --mask_stride 32 --pa 16 --pb 64 # consider a rectangle patch


# clean accuracy of patchcleanser models (the same usage as pc_certification.py)
python pc_clean_acc.py --model vit_base_patch16_224_cutout2_128 --dataset imagenet --num_img 5000  --num_mask 6 --patch_size 32


# clean accuracy of vanilla undefended models (similar usage)
python vanilla_clean_acc.py --model vit_base_patch16_224 --dataset imagenet


# train models (similar usage for other datasets)
python train_model.py --model vit_base_patch16_224 --dataset imagenette --lr 0.001 --epoch 10 
python train_model.py --model vit_base_patch16_224 --dataset imagenette --lr 0.001 --epoch 10 --cutout --cutout_size 128 --n_holes 2 
python train_model.py --model resnetv2_50x1_bit_distilled --dataset imagenette --lr 0.01 --epoch 10 
python train_model.py --model resnetv2_50x1_bit_distilled --dataset imagenette --lr 0.01 --epoch 10 --cutout --cutout_size 128 --n_holes 2 
python train_model.py --model resmlp_24_distilled_224 --dataset imagenette --lr 0.001 --epoch 10 
python train_model.py --model resmlp_24_distilled_224 --dataset imagenette --lr 0.001 --epoch 10 --cutout --cutout_size 128 --n_holes 2 
