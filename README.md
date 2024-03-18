# CDMAD
Code for the paper entitled "CDMAD: Class-Distribution-Mismatch-Aware Debiasing for Class-Imbalanced Semi-Supervised Learning" (2024 CVPR accepted paper, arXiv version available in https://arxiv.org/abs/2403.10391). 

if you want to run fixmatchcdmad.py with 0th gpu , N1 (number of labeled data points belonging to first class = num_max)=1500, M1=3000 , Imbalanced ratio of labeled set as 100, unlabeled set as 100, manualseed as 0, dataset as CIFAR-10:

python fixmatchcdmad.py --num_max 1500 --num_max_u 3000 --imb_ratio 100 --imb_ratio_u 100 --dataset cifar10 --gpu 0 --manualSeed 0

if you want to run remixmatchcdmad.py with 0th gpu , N1=450 , unknown number of unlabeled samples, Imbalanced ratio of labeled set as 20, unlabeled set as unknown, manualseed as 0, dataset as STL-10:

python remixmatchcdmad.py --num_max 450 --num_max_u 1 --imb_ratio 20 --imb_ratio_u 1 --dataset stl10 --gpu 0 --manualSeed 0

if you want to run reixmatchcdmad.py with 0th gpu , N1=150, M1=300, Imbalanced ratio of labeled set as 20, unlabeled set as 20, manualseed as 0, dataset as CIFAR-100:

python remixmatchcdmad.py --num_max 150 --num_max_u 300 --imb_ratio 20 --imb_ratio_u 20 --dataset cifar100 --gpu 0 --manualSeed 0

-------------------------------------------------------------------------------------------------------------------------------------------------------

These codes validate peformance of algorithms on testset after each epoch of training (500 iteration = 1epoch following previous studies)

-------------------------------------------------------------------------------------------------------------------------------------------------------

Performance of baseline algorithms and  the proposed algorithm are summarized in Section 4 of the paper

