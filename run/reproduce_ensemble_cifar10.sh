
## SAM rho: 0.1
#python train_ensemble.py --dataset_path ../vit_selfOT/ViT-pytorch/data --dataset cifar10 \
# --epochs 200 --batch_size 128 --sam --mode 1 --model_name R10x5  \
# --lr_schedule cosine  --rho 0.1 --random_seed 333 | tee -a reproduce_ensemble_cifar10.txt
#
#python train_ensemble.py --dataset_path ../vit_selfOT/ViT-pytorch/data --dataset cifar10 \
# --epochs 200 --batch_size 128 --sam --mode 1 --model_name R18x3 \
# --lr_schedule cosine  --rho 0.1 --random_seed 333 | tee -a reproduce_ensemble_cifar10.txt
#
#
#python train_ensemble.py --dataset_path ../vit_selfOT/ViT-pytorch/data --dataset cifar10 \
# --epochs 200 --batch_size 128 --sam --mode 1 --model_name r18mooeff \
# --lr_schedule cosine  --rho 0.1 --random_seed 333 | tee -a reproduce_ensemble_cifar10.txt

# OT_MRD rho: 0.05_0.1
python train_ensemble.py --dataset_path ../vit_selfOT/ViT-pytorch/data --dataset cifar10 \
 --epochs 200 --batch_size 128 --merge_grad --mode 1.1 --model_name R10x5 --noise_var 0.0001 \
 --lr_schedule cosine  --rho_lst 0.05_0.1 --random_seed 333 | tee -a reproduce_ensemble_cifar10.txt

python train_ensemble.py --dataset_path ../vit_selfOT/ViT-pytorch/data --dataset cifar10 \
 --epochs 200 --batch_size 128 --merge_grad --mode 1.1 --model_name R18x3 --noise_var 0.0001 \
 --lr_schedule cosine  --rho_lst 0.05_0.1 --random_seed 333 | tee -a reproduce_ensemble_cifar10.txt


python train_ensemble.py --dataset_path ../vit_selfOT/ViT-pytorch/data --dataset cifar10 \
 --epochs 200 --batch_size 128 --merge_grad --mode 1.1 --model_name r18mooeff --noise_var 0.0001 \
 --lr_schedule cosine  --rho_lst 0.05_0.1 --random_seed 333 | tee -a reproduce_ensemble_cifar10.txt