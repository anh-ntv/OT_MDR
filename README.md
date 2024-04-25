# OT-MDR: Optimal Transport Model Distributional Robustness
This is the official implementation of OT-MDR that got accepted at NeurIPs 2023

<p float="middle">
  <img src="/figures/Cifar100_r10.png" width="250" />
  <img src="/figures/cifar100_r18_label.png" width="250" /> 
  <img src="/figures/cifar100_RME.png" width="250" />
</p>

## Environment
USing Annaconda to install  
`conda env create -f ot_mdr.yml`

## Dataset
Create a folder `./dataset` that includes `cifar10` and `cifar100` folder for these two dataset or allow the script to download and save itself 

## Experiment results

_Table1. Classification accuracy on the CIFAR datasets of the single model setting with one particle. All experiments are trained three times with different random seeds._

| Dataset   | Method                                     | WideResnet28x10       | Pyramid101           | Densenet121          |
|-----------|--------------------------------------------|-----------------------|----------------------|----------------------|
| CIFAR-10  | SAM                                        | 96.72 ± 0.007         | 96.20 ±  0.134       | 91.16  ±  0.240      |
|           | **OT-MDR (Ours)**                          | **96.97 ± 0.009**     | **96.61  ± 0.063**   | **91.44  ± 0.113**   |
| CIFAR-100 | SAM                                        | 82.69 ±  0.035        | 81.26 ±  0.636       | 68.09 ±  0.403       |
|           | **OT-MDR (Ours)**                          | **84.14  ± 0.172**    | **82.28  ± 0.183**   | **69.84  ± 0.176**   |


## Training model
We provide some training log in folder `log_files`.
  
Here, we provide the script to reproduce our results in the paper for CIFAR100 dataset. For CIFAR10 dataset, please change `rho` and `rho_lst` following setting in the paper.


### Single models
Please check the file `train_otmrd.py` for detail training for single model.
#### WideResnet28x10
```bash
# SAM for cifar100
python train_otmrd.py --rho 0.1 --dataset_path ./dataset --dataset cifar100 \
--epochs 200 --batch_size 128 --mode 1 --sam

# OT-MDR for cifar100
python train_otmrd.py  --dataset_path ./dataset --dataset cifar100 \
 --epochs 200 --batch_size 128 --otmdr --merge_grad --mode 1 --noise_var 0.0001 --lr_schedule cosine \
 --rho_lst 0.1_0.2 --random_seed 42
 ```

#### Pyramid101
```bash
# SAM for cifar100
python train_otmrd.py --rho 0.1 --dataset_path ./dataset --dataset cifar100 \
--epochs 200 --batch_size 128 --mode 1 --sam --model_name pyramid101

# OT-MDR for cifar100
python train_otmrd.py  --dataset_path ./dataset --dataset cifar100 \
 --epochs 200 --batch_size 128 --otmdr --merge_grad --mode 1 --noise_var 0.0001 --lr_schedule cosine \
 --rho_lst 0.1_0.2 --model_name pyramid101
 ```

#### Densenet121
```bash
# SAM for cifar100
python train_otmrd.py --rho 0.1 --dataset_path ./dataset --dataset cifar100 \
--epochs 200 --batch_size 128 --mode 1 --sam --model_name densenet121

# OT-MDR for cifar100
python train_otmrd.py  --dataset_path ./dataset --dataset cifar100 \
 --epochs 200 --batch_size 128 --otmdr --merge_grad --mode 1 --noise_var 0.0001 --lr_schedule cosine \
 --rho_lst 0.1_0.2 --model_name densenet121
 ```


#### Resnet18
The baseline on Resnet18 is taken from bSAM paper
```bash
# OT-MDR for cifar100
python train_otmrd.py  --dataset_path ./dataset --dataset cifar100 \
 --epochs 200 --batch_size 128 --otmdr --merge_grad --mode 1 --noise_var 0.0001 --lr_schedule cosine \
 --rho_lst 0.1_0.2 --model_name resnet18
 ```

## Ensemble models
Please check the file `train_ensemble.py` for detail training.

```bash
# OT-MDR for ensemble five of Resnet10
python train_ensemble.py --dataset_path ./dataset --dataset cifar100 \
 --epochs 200 --batch_size 128 --otmdr --merge_grad --mode 1 --model_name R10x5 \
 --lr_schedule cosine --log_dir ../log_files/cifar100/resnet10/ot-mdr_rho0.1-0.2 --rho_lst 0.1_0.2
 

# OT-MDR for ensemble three of Resnet18
python train_ensemble.py --dataset_path ./dataset --dataset cifar100 \
 --epochs 200 --batch_size 128 --otmdr --merge_grad --mode 1 --model_name R18x3 \
 --lr_schedule cosine --log_dir ../log_files/cifar100/resnet18/ot-mdr_rho0.1-0.2 --rho_lst 0.1_0.2
 
 
# OT-MDR for ensemble of ResNet18, MobileNet and EfficientNet
python train_ensemble.py --dataset_path ./dataset --dataset cifar100 \
 --epochs 200 --batch_size 128 --otmdr --merge_grad --mode 1 --model_name r18mooeff \
 --lr_schedule cosine --log_dir ../log_files/cifar100/r18mooeff/ot-mdr_rho0.1-0.2 --rho_lst 0.1_0.2
 ```

## Bayesian Neural Networks
Please check the file `train_otmrd_BNN.py` for detail.

```bash
# Baseline using Adam for SGVB on Resnet10
python train_otmrd_BNN.py --dataset_path ./dataset \
 --dataset cifar100  --epochs 200 --batch_size 128 --adam --model_name r10 \
 --learning_rate 0.001 --n_model 1 --lr_schedule plateau --beta_type 5e-6 
 
# OT-MDR for SGVB on Resnet10
python train_otmrd_BNN.py --dataset_path ./dataset --dataset cifar100 \
--epochs 100 --batch_size 1024 --otmdr --mode 1 --model_name r10 \
--n_model 1 --beta_type 5e-6 --noise_var 0.01 --merge_grad --ignore_sigma --rho_lst 0.005_0.01
```

```bash
# Baseline using Adam for SGVB on Resnet18
python train_otmrd_BNN.py --dataset_path ./dataset \
 --dataset cifar100  --epochs 200 --batch_size 128 --adam --model_name r18 \
 --learning_rate 0.001 --n_model 1 --lr_schedule plateau --beta_type 5e-6 
 
# OT-MDR for SGVB on Resnet18
python train_otmrd_BNN.py --dataset_path ./dataset --dataset cifar100 \
--epochs 100 --batch_size 1024 --otmdr --mode 1 --model_name r18 \
--n_model 1 --beta_type 5e-6 --noise_var 0.01 --merge_grad --ignore_sigma --rho_lst 0.005_0.01
```

### Citation
```bibtex
@inproceedings{  
    nguyen2023optimal,  
    title={Optimal Transport Model Distributional Robustness},  
    author={Van-Anh Nguyen and Trung Le and Anh Tuan Bui and Thanh-Toan Do and Dinh Phung},  
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},  
    year={2023},  
    url={https://openreview.net/forum?id=D7LdL2SCCi}
    } 
```

### Acknowledgement
This repository is based on [SAM](https://github.com/davda54/sam)



