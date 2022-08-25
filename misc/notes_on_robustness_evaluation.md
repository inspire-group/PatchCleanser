## Are you looking for "attack code" in this repository?

Unfortunately, there is no attack code here.

Note that PatchCleanser is a certifiably robust defense. Its robustness evaluation is done using the certification procedure (Algorithm 2), instead of any concrete attack code. The proof (Theorem 1) in the paper demonstrates the soundness of our certification algorithm: the certified robust accuracy is a lower bound on model accuracy against any attacker within the threat model. 

Please read [this note](https://github.com/xiangchong1/adv-patch-paper-list#empirically-robust-defenses-vs-provablycertifiably-robust-defenses) for more discussions between certifiably robust defenses and empirical defenses.

## What if I really want to evaluate the empirical robust accuracy of PatchCleanser?

You will probably have to implement the attack yourself. 

#### Here are a few notes/suggestions.

Of course, you need to use *adaptive* attacks (the attack algorithm should be targeted at PatchCleanser) to evaluate the empirical robust accuracy. The question is: what is a good adaptive attack against PatchCleanser? Here is one possible strategy:
1. Find a mask location where two-mask correctness is not satisfied, and place a patch there. -> since there is an incorrect two-mask prediction, PatchCleanser will not return a correct disagreer prediction in the second-round masking (Case II).
2. Optimize the patch content such that the one-mask majority prediction is incorrect -> then PatchCleanser will not return a correct majority prediction in Case III, or an agreed prediction via Case I.

#### How hard is this attack?
shouldn't be too hard. 
1. The first step is just to find violated two-mask correctness. If there is no violated two-mask correctness, PatchCleanser is certifiably robust, and there is no need to empirically attack PatchCleanser
2. The second step shouldn't be too hard. When the first-round mask does not remove the patch, the malicious masked predictions usually do not change (so we have an incorrect majority prediction). Moreover, the patch content can be further optimized for the malicious majority prediction if we do observe inconsistent malicious masked predictions.

#### Additional note
be careful with the image pixel normalization. Some models in the repo scale pixel value with a mean `[0.5,0.5,0.5]` instead of `[0.485, 0.456, 0.406]`. The inference on clean images might only be affected slightly, but the robustness can be very different if you use a different normalization parameter.
