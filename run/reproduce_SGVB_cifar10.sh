# baseline SGVB - cifar10 - r10
CUDA_VISIBLE_DEVICES=3 python train_otmrd_BNN.py --epoch 100 --batch_size 1024 --learning_rate 0.001 \
--dataset_path ../vit_selfOT/ViT-pytorch/data --dataset cifar10 --model_name r10\
 --n_model 1 --lr_schedule plateau --beta_type 0.000005 \
 --adam --random_seed 123 | tee -a reproduce_SGVB_cifar10.txt


# baseline SGVB - cifar10 - r18
CUDA_VISIBLE_DEVICES=3 python train_otmrd_BNN.py --epoch 100 --batch_size 1024 --learning_rate 0.001 \
--dataset_path ../vit_selfOT/ViT-pytorch/data --dataset cifar10 --model_name r18\
 --n_model 1 --lr_schedule plateau --beta_type 0.000005 \
 --adam --random_seed 123 | tee -a reproduce_SGVB_cifar10.txt


# F-SGVB - cifar10 - r10
CUDA_VISIBLE_DEVICES=3 python train_otmrd_BNN.py --epoch 100 --batch_size 1024 --learning_rate 0.001 \
--dataset_path ../vit_selfOT/ViT-pytorch/data --dataset cifar10 --model_name r10\
 --n_model 1 --lr_schedule plateau --beta_type 0.000005 \
 --sam --random_seed 123  --rho 0.005  --ignore_sigma| tee -a reproduce_SGVB_cifar10.txt



# F-SGVB - cifar10 - r18
CUDA_VISIBLE_DEVICES=3 python train_otmrd_BNN.py --epoch 100 --batch_size 1024 --learning_rate 0.001 \
--dataset_path ../vit_selfOT/ViT-pytorch/data --dataset cifar10 --model_name r18\
 --n_model 1 --lr_schedule plateau --beta_type 0.000005 \
 --sam --random_seed 123  --rho 0.005  --ignore_sigma | tee -a reproduce_SGVB_cifar10.txt


#../data/data/
#../vit_selfOT/ViT-pytorch/data

# F-SGVB-geometry - cifar10 - r10
CUDA_VISIBLE_DEVICES=3 python train_otmrd_BNN.py --epoch 100 --batch_size 1024 --learning_rate 0.001 \
--dataset_path ../data/data/ --dataset cifar10 --model_name r10\
 --n_model 1 --lr_schedule plateau --beta_type 0.000005 \
 --sam --random_seed 123  --rho 0.0005  --ignore_sigma --geometry | tee -a reproduce_SGVB_cifar10.txt

python train_otmrd_BNN.py --epoch 100 --batch_size 1024 --learning_rate 0.001 \
--dataset_path ../data/data/ --dataset cifar10 --model_name r10\
 --n_model 1 --lr_schedule plateau --beta_type 0.000005 \
 --sam --random_seed 123  --rho 0.0005  --ignore_sigma --geometry --p_power | tee -a reproduce_SGVB_cifar10.txt



# F-SGVB-geometry - cifar10 - r18
CUDA_VISIBLE_DEVICES=3 python train_otmrd_BNN.py --epoch 100 --batch_size 1024 --learning_rate 0.001 \
--dataset_path ../vit_selfOT/ViT-pytorch/data --dataset cifar10 --model_name r18\
 --n_model 1 --lr_schedule plateau --beta_type 0.000005 \
 --sam --random_seed 123  --rho 0.0005  --ignore_sigma --geometry | tee -a reproduce_SGVB_cifar10.txt

python train_otmrd_BNN.py --epoch 100 --batch_size 1024 --learning_rate 0.001 \
--dataset_path ../vit_selfOT/ViT-pytorch/data --dataset cifar10 --model_name r18\
 --n_model 1 --lr_schedule plateau --beta_type 0.000005 \
 --sam --random_seed 123  --rho 0.0005  --ignore_sigma --geometry --p_power | tee -a reproduce_SGVB_cifar10.txt