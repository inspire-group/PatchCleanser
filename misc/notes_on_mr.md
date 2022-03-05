## Notes on the Minority Reports defense

In `pc_mr.py`, I discarded the voting grid design discussed in the original Minority Reports paper. See reasons below.

### MR Algorithm

The original [Minority Reports paper](https://arxiv.org/abs/2004.13799) focuses on the certified robustness on low resolution images (e.g., CIFAR-10). 

To counter a 5x5 patch on 32x32 CIFAR images, MR 

1. places 7x7 masks on the 32x32 image to generate a 25x25 prediction grid (25=32-7+1)
2. considers 3x3 regions on the 25x25 prediction grid to generate 23x23 voting grid (23=25-3+1)
3. If all voting grids vote for the correct class label, MR has certifiable robustness for attack detection

### Issue

The certification depends on the argument that "wherever the sticker is placed, there will be a 3×3 grid in the prediction grid that is unaffected by the sticker."

However, this argument is actually unsound for the approach described in the original paper. If we consider a 5x5 patch placed at the corner of the image, for example, the upper left coordinate of the patch is (0,0)...., there is *only one* unaffected masked prediction. This unaffected masked prediction is given a the mask whose upper left coordinate is (0,0). Any other mask will leave  the adversarial *pixel* at (0,0) unmasked. We cannot certify the robustness for this corner case.

### Fix

One possible fix is to start the mask location at (-2,-2) instead of (0,0)!

In `pc_mr_expertimental.py`, I implemented this secure version of MR.  Additionally, I add a parameters `--mr` to tune the number of predictions that participate in the voting. We will use `(mr+1)x(mr+1) ` predictions for voting. 

`mr=2` replicates the original MR. `mr=0` replicates the first-round masking of PatchCleanser.

### Observation

My experiments on ImageNet find that, if we use the same mask stride, then

1. setting `mr` to a non-zero value only slightly affects the defense performance.
2. however, if we use the insecure masking strategy discussed in the original paper, a non-zero `mr` can significantly improves the defense performance.
4. I simplified `pc_mr_expertimental.py` to `pc_mr.py`, which always uses `mr=0`. 

By the way, there are other possible fixes that might improve the robustness with a non-zero `mr` , further discussions are out of the scope of this repo...

PS. I did not spend much time on the Minority Reports. If you have better implementation strategies or have different observations, I am happy to discuss!

