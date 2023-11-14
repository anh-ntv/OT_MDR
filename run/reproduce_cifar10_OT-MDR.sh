#python train_otmrd.py  --dataset_path ../data/data/ --dataset cifar10  --epochs 200\
# --batch_size 128 --sam_chain --merge_grad --mode 1 --noise_var 0.0001\
#  --lr_schedule cosine  --rho_lst 0.05_0.1 --random_seed 989

#python train_otmrd.py  --dataset_path ../data/data/ --dataset cifar10  --epochs 200\
# --batch_size 128 --sam_chain --merge_grad --mode 1 --noise_var 0.0001 --model_name pyramid101\
#  --lr_schedule cosine  --rho_lst 0.05_0.1 --random_seed 989
#
#python train_otmrd.py  --dataset_path ../data/data/ --dataset cifar10  --epochs 200\
# --batch_size 128 --sam_chain --merge_grad --mode 1 --noise_var 0.0001 --model_name densenet121\
#  --lr_schedule cosine  --rho_lst 0.05_0.1 --random_seed 989

python train_otmrd.py  --dataset_path ../data/data/ --dataset cifar10  --epochs 200\
 --batch_size 128 --sam --rho 0.05 --lr_schedule cosine  --random_seed 989

python train_otmrd.py  --dataset_path ../data/data/ --dataset cifar10  --epochs 200\
 --batch_size 128 --sam --rho 0.05 --lr_schedule cosine  --random_seed 989 --model_name pyramid101

python train_otmrd.py  --dataset_path ../data/data/ --dataset cifar10  --epochs 200\
 --batch_size 128 --sam --rho 0.05 --lr_schedule cosine  --random_seed 989 --model_name densenet121





