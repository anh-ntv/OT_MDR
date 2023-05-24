import os
import argparse
import torch
import torch.nn as nn
import numpy as np

from model.wide_res_net import WideResNet
# from model.vit import VisionTransformer, CONFIGS
from model.smooth_cross_entropy import smooth_crossentropy
from model.pyramidNet import PyramidNet

from model.utils_arch import get_ensemble, EnsembleWrap

from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats


from torch.optim.lr_scheduler import CosineAnnealingLR

# import sys; sys.path.append("..")
from OT_MDR_optimizer import SAM, SAM_batch_chain, SAMnChain

import torchvision.transforms as transforms

def save_checkpoint(dir, epoch, name="checkpoint", replace=False, **kwargs):
    state = {}
    state.update(kwargs)
    if replace:
        for root, dirs, files in os.walk(dir):
            for f_n in files:
                if name in f_n:
                    os.remove(os.path.join(root, f_n))
                    print("Remove", f_n)
    filepath = os.path.join(dir, "%s-%d.pt" % (name, epoch))
    torch.save(state, filepath)


def train_sambatch_chain(args, model, dataset):

    print("===\nTrain with train_sambatch_chain\n===")
    base_optimizer = torch.optim.SGD
    optimizer = SAM_batch_chain(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive,
                              lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
                                merge_grad=args.merge_grad, mode=args.mode,
                                rho_lst=args.rho_lst, model=model)
    output_islogit=True
    if isinstance(model, EnsembleWrap):
        output_islogit=False
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    if args.lr_schedule in ['cosine']:
        scheduler = CosineAnnealingLR(optimizer, args.epochs)
    else:
        scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))
        iter = 0
        for input_raw, targets_raw in dataset.train:
            inputs = input_raw.to(device)
            targets = targets_raw.to(device)
            bz_instance = inputs.size(0)
            len_b1 = int(args.data_split[0] * bz_instance)
            len_b2 = int(args.data_split[1] * bz_instance)
            input_b1 = inputs[:len_b1]
            target_b1 = targets[:len_b1]

            input_b2 = inputs[-len_b2:]
            target_b2 = targets[-len_b2:]

            enable_running_stats(model)
            predictions_b1 = model(input_b1)
            loss_c1 = smooth_crossentropy(predictions_b1, target_b1, smoothing=args.label_smoothing, islogits=output_islogit)
            loss_c1.mean().backward()
            optimizer.first_step(zero_grad=True)

            predictions_b2 = model(input_b2)
            loss_c2 = smooth_crossentropy(predictions_b2, target_b2, smoothing=args.label_smoothing, islogits=output_islogit)
            loss_c2.mean().backward()
            optimizer.first_step(zero_grad=True)

            disable_running_stats(model)
            predictions_a = model(inputs)      # B1_B2_B
            loss_max = smooth_crossentropy(predictions_a, targets, smoothing=args.label_smoothing, islogits=output_islogit)
            loss_max.mean().backward()  # not update "running_mean" and "running_var"
            optimizer.second_step(zero_grad=True, mode=args.mode)
            sharpness = loss_max - torch.cat([loss_c1, loss_c2])

            with torch.no_grad():
                predictions = torch.cat([predictions_b1, predictions_b2])
                targets_merge = torch.cat([target_b1, target_b2])
                correct = torch.argmax(predictions.data, 1) == targets_merge
                if args.lr_schedule in ['cosine']:
                    lr = scheduler.get_lr()[0]
                    if iter == 0:
                        scheduler.step()
                else:
                    lr = scheduler.lr()
                    scheduler(epoch)
                loss = torch.cat([loss_c1, loss_c2])
                # loss = torch.cat([loss_c, loss_max])
                log(model, loss.cpu(), correct.cpu(), lr, sharpness=sharpness)
            iter += 1

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            prediction_test = []
            target_test = []
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss_c = smooth_crossentropy(predictions, targets, islogits=output_islogit)
                correct = torch.argmax(predictions, 1) == targets
                prediction_test.append(nn.functional.softmax(predictions, dim=-1))
                target_test.append(targets)

                log(model, loss_c.cpu(), correct.cpu())

        if (epoch + 1) % 50 == 0 and args.log_dir:
            os.makedirs(args.log_dir, exist_ok=True)
            save_checkpoint(args.log_dir, epoch=epoch,
                            name="checkpoint_{}".format(args.random_seed),
                            replace=True,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(),)


def train_sambatch_nchain(args, model, dataset):

    print("===\nTrain with train_sambatch_Nchain\n===")
    base_optimizer = torch.optim.SGD
    optimizer = SAMnChain(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, rho_lst=args.rho_lst,
                              lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
                                merge_grad=args.merge_grad, mode=args.mode, n_branch=args.n_branch, true_grad=args.true_grad)
    output_islogit=True
    if isinstance(model, EnsembleWrap):
        output_islogit=False
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    if args.lr_schedule in ['cosine']:
        scheduler = CosineAnnealingLR(optimizer, args.epochs)
    else:
        scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        if isinstance(model, EnsembleWrap):
            for submodel in models:
                submodel.train()
        log.train(len_dataset=len(dataset.train))
        iter = 0
        for input_raw, targets_raw in dataset.train:
            inputs_full = input_raw.to(device)
            targets_full = targets_raw.to(device)

            bz_instance = targets_full.size(0)
            len_b1 = int(args.data_split[0] * bz_instance)
            len_b2 = int(args.data_split[1] * bz_instance)

            prediction_lst = []
            target_lst = []
            loss_lst = []

            sharpness_lst = []
            for idx_p in range(args.n_branch):

                idx_shuffle = torch.randperm(bz_instance)
                inputs = inputs_full[idx_shuffle]
                targets = targets_full[idx_shuffle]
                input_b1 = inputs[:len_b1]
                target_b1 = targets[:len_b1]

                input_b2 = inputs[-len_b2:]
                target_b2 = targets[-len_b2:]


                enable_running_stats(model)
                predictions_b1 = model(input_b1)
                loss_c1 = smooth_crossentropy(predictions_b1, target_b1, smoothing=args.label_smoothing, islogits=output_islogit)
                loss_c1.mean().backward()
                optimizer.first_step(idx_p, zero_grad=True, p_a=True)

                predictions_b2 = model(input_b2)
                loss_c2 = smooth_crossentropy(predictions_b2, target_b2, smoothing=args.label_smoothing, islogits=output_islogit)
                loss_c2.mean().backward()

                optimizer.first_step_grad(idx_p, zero_grad=True, p_a=True)

                prediction_lst.append(predictions_b1)
                target_lst.append(target_b1)
                loss_lst.append(loss_c1)

                disable_running_stats(model)

                predictions_a = model(inputs)  # B1_B2_B
                loss_max = smooth_crossentropy(predictions_a, targets, smoothing=args.label_smoothing,
                                               islogits=output_islogit)
                loss_max.mean().backward()
                optimizer.save_grad(idx_p, "p_grad_update", data_name="old_p", zero_grad=True)
                sharpness = loss_max[:len_b1] - loss_c1
                sharpness_lst.append(sharpness)

            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                predictions = torch.cat(prediction_lst)
                targets_merge = torch.cat(target_lst)
                correct = torch.argmax(predictions.data, 1) == targets_merge
                if args.lr_schedule in ['cosine']:
                    lr = scheduler.get_lr()[0]
                    if iter == 0:
                        scheduler.step()
                else:
                    lr = scheduler.lr()
                    scheduler(epoch)
                loss = torch.cat(loss_lst)
                # sharpness = torch.cat(sharpness_lst)
                sharpness = torch.cat(sharpness_lst)
                # loss = torch.cat([loss_c, loss_max])
                log(model, loss.cpu(), correct.cpu(), lr, sharpness=sharpness)
            iter += 1

        model.eval()
        if isinstance(model, EnsembleWrap):
            for submodel in models:
                submodel.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss_c = smooth_crossentropy(predictions, targets, islogits=output_islogit)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss_c.cpu(), correct.cpu())

        if (epoch + 1) % 50 == 0 and args.log_dir:
            os.makedirs(args.log_dir, exist_ok=True)
            save_checkpoint(args.log_dir, epoch=epoch,
                            name="checkpoint_{}".format(args.random_seed),
                            replace=True,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(),)

def train_sam(args, model, dataset):
    print("===\nTrain with train_sam\n===")
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, is_sgd=args.sgd,
                    lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.lr_schedule in ['cosine']:
        scheduler = CosineAnnealingLR(optimizer, args.epochs)
    else:
        scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    # if args.fine_tune:

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))
        iter = 0
        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)
            bz_instance = inputs.size(0)

            enable_running_stats(model)
            predictions = model(inputs)
            loss_c = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss_c.mean().backward()  # update "running_mean" and "running_var"
            optimizer.first_step(zero_grad=True)

            disable_running_stats(model)
            predictions_max = model(inputs)
            loss_max = smooth_crossentropy(predictions_max, targets, smoothing=args.label_smoothing)
            loss_max.mean().backward()  # not update "running_mean" and "running_var"
            optimizer.second_step(zero_grad=True)

            sharpness = loss_max - loss_c
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                if args.lr_schedule in ['cosine']:
                    lr = scheduler.get_lr()[0]
                    if iter == 0:
                        scheduler.step()
                else:
                    lr = scheduler.lr()
                    scheduler(epoch)
                log(model, loss_c.cpu(), correct.cpu(), lr, sharpness=sharpness)
            iter += 1

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss_c = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss_c.cpu(), correct.cpu())

        if (epoch + 1) % 50 == 0 and args.log_dir:
            os.makedirs(args.log_dir, exist_ok=True)
            save_checkpoint(args.log_dir, epoch=epoch,
                            name="checkpoint_{}".format(args.random_seed),
                            replace=True,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(),)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", action="store_true", help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")

    parser.add_argument("--dataset_path", default="/home/ubuntu/vit_selfOT/ViT-pytorch/data", type=str, help="link to dataset")
    parser.add_argument("--mode", type=float, default=1., help="change grad mode: 1: grad=max-min+current, 2: grad=max-min")

    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset to load: cifar10 and cifar100")
    parser.add_argument("--model_name", type=str, default="WRN", help="model_name to load")
    parser.add_argument("--lr_schedule", type=str, default="step", help="lr_schedule: step, cosine")
    parser.add_argument("--rho_lst", type=str, default="0.05_0.05", help="0.05_0.05")
    parser.add_argument("--data_split", type=str, default="0.5_0.5", help="0.5_0.5")

    parser.add_argument("--sam", action="store_true")
    parser.add_argument("--sgd", action="store_true", help="Update model by SGD")
    parser.add_argument("--n_branch", type=int, default=1, help="Update model by SAMnBranch")
    parser.add_argument("--sam_chain", action="store_true", help="Update model by sam2branch with chain")

    parser.add_argument("--merge_grad", action="store_true", help="merge gradient for sam_batch_chain")
    parser.add_argument("--noise_var", type=float, default=0, help="Noise variance")
    parser.add_argument("--img_size", type=int, default=32, help="")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_dir", type=str, default=None, help="folder to save model")
    parser.add_argument("--fine_tune", action="store_true", help="fine tunning model")
    parser.add_argument("--true_grad", action="store_true", help="fine tunning model")

    args = parser.parse_args()
    args.data_split = [float(it) for it in args.data_split.split("_")]
    args.rho_lst = [float(it) for it in args.rho_lst.split("_")]
    for arg_item in vars(args):
        print(arg_item, getattr(args, arg_item))

    initialize(args, seed=args.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tran_trans = None
    test_trans = None
    if "resnet18" in args.model_name:
        tran_trans = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomCrop(224, padding=14),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        test_trans = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    dataset = Cifar(args.batch_size, args.threads, img_size=args.img_size, root=args.dataset_path, dataset=args.dataset,
                    bz_test=args.batch_size, train_transform=tran_trans, test_transform=test_trans)

    log = Log(log_each=10)
    num_cls = len(dataset.classes)
    if args.model_name == "WRN":
        # model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=num_cls)
        model = WideResNet(28, 10, args.dropout, in_channels=3, labels=num_cls)
    elif args.model_name == "densenet121":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False, num_classes=num_cls)
    elif args.model_name == "resnet18":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=num_cls)
    elif args.model_name == "pyramid101":
        model = PyramidNet(dataset=args.dataset, depth=101, alpha=64, num_classes=num_cls, bottleneck=True)
    elif args.model_name == "pyramid110":
        model = PyramidNet(dataset=args.dataset, depth=110, alpha=64, num_classes=num_cls, bottleneck=True)
    elif args.model_name == "Resnet18_imgnet":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=num_cls)
    else:
        ens_mode = 'average_prob'
        models = get_ensemble(arch=args.model_name, dataset=args.dataset)
        model = EnsembleWrap(models, mode=ens_mode)  # set islogits=False

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    if args.sam or args.sgd:
        train_sam(args, model, dataset)
    if args.sam_1chain or args.n_branch == 1:
        train_sambatch_chain(args, model, dataset)
    elif args.sam_chain or args.n_branch > 1:
        train_sambatch_nchain(args, model, dataset)
    else:
        print("BOOOOOM\nCheck your running params!!!!")

    log.flush()
