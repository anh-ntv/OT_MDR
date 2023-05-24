import os
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# from model.wide_res_net import WideResNet
# from model.vit import VisionTransformer, CONFIGS
from model.smooth_cross_entropy import smooth_crossentropy, logmeanexp
from model.preresnet_VD import PreResNet164_model
from model.resnet_sgvb import ResNet10, ResNet18
from model.resnet_small_svgd import resnet20, resnet8, resnet14, resnet32, resnet44, resnet56, resnet110
# from efficientnet_pytorch import EfficientNet

from model.utils_arch import get_ensemble, EnsembleWrap

from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from OT_MDR_optimizer import SAM, SAM_batch_chain

import bnn_metric as metric

def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    ps_log = np.log(ps)
    nll_sum = -np.sum(ps_log)
    nll_mean = -np.mean(ps_log)
    return nll_sum, nll_mean


def ece_score_np(py, y_test, n_bins=20, return_stat=False):
    """

    Args:
        py: softmax prediction of model
        y_test: target (onehot or label)
        n_bins: number of bins

    Returns:

    """
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.asarray(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    if return_stat:
        return ece / sum(Bm), acc, conf, Bm
    return ece / sum(Bm)


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

# def load_mode(dir, name="checkpoint", epoch=None):
#     filepath = os.path.join(dir, "%s-%d.pt" % (name, epoch))
#     checkpoint =

def train_sambatch_chain(args, model, dataset):
    # import time
    # debug = True
    print("===\nTrain with train_sambatch_chain\n===")
    # base_optimizer = torch.optim.SGD
    # optimizer = SAM_batch_chain(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive,
    #                           lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
    #                             merge_grad=args.merge_grad, mode=args.mode,
    #                             rho_lst=args.rho_lst)

    base_optimizer = torch.optim.Adam
    optimizer = SAM_batch_chain(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive,
                              lr=args.learning_rate, weight_decay=args.weight_decay,
                                merge_grad=args.merge_grad, mode=args.mode,
                                rho_lst=args.rho_lst, model=model, ignore_sigma=args.ignore_sigma,
                                noise_var=args.noise_var)
    output_islogit=True
    if isinstance(model, EnsembleWrap):
        output_islogit=False

    if args.lr_schedule in ['cosine']:
        scheduler = CosineAnnealingLR(optimizer, args.epochs)
    elif args.lr_schedule in ['plateau']:
        # scheduler = ReduceLROnPlateau(optimizer, patience=10, verbose=False, factor=0.2)
        scheduler = ReduceLROnPlateau(optimizer, patience=6, verbose=False)
    elif args.lr_schedule in ['step']:
        scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    elbo_loss = metric.ELBO(len(dataset.train)).to(device)
    for epoch in range(args.epochs):
        model.train()
        if isinstance(model, EnsembleWrap):
            for submodel in model:
                submodel.train()
        log.train(len_dataset=len(dataset.train))
        iter = 0
        for input_raw, targets_raw in dataset.train:
            # st = time.time()
            inputs = input_raw.to(device)
            targets = targets_raw.to(device)
            bz_instance = inputs.size(0)
            len_b1 = int(args.data_split[0] * bz_instance)
            len_b2 = int(args.data_split[1] * bz_instance)
            input_b1 = inputs[:len_b1]
            target_b1 = targets[:len_b1]

            input_b2 = inputs[-len_b2:]
            target_b2 = targets[-len_b2:]

            # print("process data", time.time() - st)
            # st = time.time()

            enable_running_stats(model)
            # disable_running_stats(model)

            predictions_b1 = []
            # optimizer.ignore_sigma = False
            kl_b1 = 0.0
            for j in range(args.n_model):
                _predictions_b1, _kl_b1 = model(input_b1)
                kl_b1 = kl_b1 + _kl_b1
                predictions_b1.append(F.log_softmax(_predictions_b1, dim=1))
            kl_b1 = kl_b1 / args.n_model
            predictions_b1 = torch.stack(predictions_b1, dim=-1)
            log_outputs1 = logmeanexp(predictions_b1, dim=2)
            # beta = metric.get_beta(iter - 1, len(dataset.train), 1/bz_instance, epoch, args.epochs)
            beta = metric.get_beta(iter - 1, len(dataset.train), args.beta_type, epoch, args.epochs)

            loss_c1 = elbo_loss(log_outputs1, target_b1, kl_b1, beta)
            loss_c1.backward()  # update "running_mean" and "running_var"
            optimizer.first_step(zero_grad=True)


            disable_running_stats(model)
            predictions_b2 = []
            # optimizer.ignore_sigma = True
            kl_b2 = 0.0
            for j in range(args.n_model):
                # _predictions_b2, _kl_b2 = model(input_b2, reuse_eps=True)
                _predictions_b2, _kl_b2 = model(input_b2, reuse_eps=False)
                # _predictions_b2, _kl_b2 = model(input_b2)
                kl_b2 = kl_b2 + _kl_b2
                predictions_b2.append(F.log_softmax(_predictions_b2, dim=1))
            kl_b2 = kl_b2 / args.n_model
            predictions_b2 = torch.stack(predictions_b2, dim=-1)
            log_outputs2 = logmeanexp(predictions_b2, dim=2)
            # beta = metric.get_beta(iter - 1, len(dataset.train), 1/bz_instance, epoch, args.epochs)
            beta = metric.get_beta(iter - 1, len(dataset.train), args.beta_type, epoch, args.epochs)

            loss_c2 = elbo_loss(log_outputs2, target_b2, kl_b2, beta)
            loss_c2.backward()  # update "running_mean" and "running_var"
            optimizer.first_step(zero_grad=True)

            # optimizer.ignore_sigma = False
            disable_running_stats(model)
            # enable_running_stats(model)
            predictions_a = []
            kl_a = 0.0
            for j in range(args.n_model):
                _predictions_a, _kl_a = model(inputs, reuse_eps=True)
                # _predictions_a, _kl_a = model(inputs)
                kl_a = kl_a + _kl_a
                predictions_a.append(F.log_softmax(_predictions_a, dim=1))
            kl_a = kl_a / args.n_model
            predictions_a = torch.stack(predictions_a, dim=-1)
            log_outputsa = logmeanexp(predictions_a, dim=2)
            betaa = metric.get_beta(iter - 1, len(dataset.train), args.beta_type, epoch, args.epochs)
            loss_max = elbo_loss(log_outputsa, targets, kl_a, betaa)
            loss_max.backward()
            optimizer.second_step(zero_grad=True)
            sharpness = loss_max - (loss_c1 + loss_c2)/2.0

            with torch.no_grad():
                acc_b1 = (log_outputs1.data.argmax(axis=1) == target_b1).sum()/len(log_outputs1)
                acc_b2 = (log_outputs2.data.argmax(axis=1) == target_b2).sum()/len(log_outputs2)
                acc = (acc_b1 + acc_b2) / 2
                if args.lr_schedule in ['step']:
                    lr = scheduler.lr()
                elif args.lr_schedule in ['plateau']:
                    lr = optimizer.param_groups[0]['lr']
                else:
                    lr = scheduler.get_lr()[0]
                loss = (loss_c1 + loss_c2)/2.0
                log(model, loss.view(1, -1), acc.view(1, -1), lr, sharpness=sharpness.view(1, -1))

            iter += 1

        model.eval()
        if isinstance(model, EnsembleWrap):
            for submodel in model:
                submodel.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            n_sample = 1
            r_e = True
            if epoch == args.epochs - 1 or epoch % 10 == 0:
                n_sample = 30
                r_e = False
                # print("Test with {} samples", n_sample)
            loss_test = []
            prediction_test = []
            target_test = []
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)
                target_test.append(targets)

                predictions = []
                predictions_ens = []
                kl = 0.0
                for _ in range(n_sample):
                    _predictions, _kl = model(inputs, reuse_eps=r_e)
                    kl = kl + _kl
                    predictions.append(F.log_softmax(_predictions, dim=1))
                    predictions_ens.append(F.softmax(_predictions, dim=-1))
                kl = kl / n_sample
                predictions = torch.stack(predictions, dim=-1)
                predictions_ens = torch.stack(predictions_ens, dim=-1)
                prediction_test.append(predictions_ens.mean(-1))
                log_outputs = logmeanexp(predictions, dim=2)
                # beta = metric.get_beta(iter, len(dataset.test), 1/bz_instance, epoch, args.epochs)
                beta = metric.get_beta(iter, len(dataset.test), args.beta_type, epoch, args.epochs)
                loss = elbo_loss(log_outputs, targets, kl, beta)
                loss_test.append(loss.item())
                acc = (log_outputs.argmax(axis=1) == targets).sum() / len(log_outputs)
                log(model, loss.view(1, -1), acc.view(1, -1))

        if n_sample > 1:
            pre_ens = torch.cat(prediction_test).cpu().numpy()
            target_ens = torch.cat(target_test).cpu().numpy()
            acc_test = np.mean(np.argmax(pre_ens, axis=1) == target_ens) * 100
            _, eval_nll = nll(pre_ens, target_ens)
            eval_ece, acc_bin, conf_bin, Bm_bin = ece_score_np(pre_ens, target_ens, n_bins=20, return_stat=True)
            print("\nACC", acc_test)
            print("NLL", eval_nll)
            print("ECE", eval_ece)
            # print('--------- ECE stat ----------', epoch)
            # print("ACC", acc_bin)
            # print("Conf", conf_bin)
            # print("Bm", Bm_bin)

        if args.lr_schedule in ['cosine']:
            scheduler.step()
        elif args.lr_schedule in ['plateau']:
            scheduler.step(sum(loss_test))
        else:
            scheduler(epoch)

        if (epoch + 1) % 50 == 0 and args.log_dir:
            os.makedirs(args.log_dir, exist_ok=True)
            save_checkpoint(args.log_dir, epoch=epoch,
                            name="checkpoint_{}".format(args.random_seed),
                            replace=True,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(), )


def train_sam(args, model, dataset):
    print("===\nTrain with train_sam\n===")
    # base_optimizer = torch.optim.SGD
    # optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, is_sgd=args.sgd,
    #                 lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    base_optimizer = torch.optim.Adam
    if args.sam:
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, is_sgd=args.sgd,
                    lr=args.learning_rate, weight_decay=args.weight_decay, model=model, ignore_sigma=args.ignore_sigma,
                    geometry=args.geometry, p_power=args.p_power)
    else:
        optimizer = base_optimizer(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.lr_schedule in ['cosine']:
        scheduler = CosineAnnealingLR(optimizer, args.epochs)
    elif args.lr_schedule in ['plateau']:
        scheduler = ReduceLROnPlateau(optimizer, patience=6, verbose=False)
    else:
        scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    # if args.fine_tune:
    elbo_loss = metric.ELBO(len(dataset.train)).to(device)
    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))
        iter = 0
        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)
            bz_instance = inputs.size(0)

            enable_running_stats(model)

            if isinstance(model, EnsembleWrap):
                for submodel in model:
                    enable_running_stats(submodel)
            predictions_b1 = []
            # import pdb; pdb.set_trace()

            kl_b1 = 0.0
            for j in range(args.n_model):
                _predictions_b1, _kl_b1 = model(inputs)
                kl_b1 = kl_b1 + _kl_b1
                predictions_b1.append(F.log_softmax(_predictions_b1, dim=1))
            kl_b1 = kl_b1/args.n_model
            predictions_b1 = torch.stack(predictions_b1, dim=-1)
            log_outputs1 = logmeanexp(predictions_b1, dim=2)
            # beta = metric.get_beta(iter - 1, len(dataset.train), 1/bz_instance, epoch, args.epochs)
            beta = metric.get_beta(iter - 1, len(dataset.train), args.beta_type, epoch, args.epochs)

            loss_c1 = elbo_loss(log_outputs1, targets, kl_b1, beta)
            # import pdb; pdb.set_trace()
            loss_c1.backward()  # update "running_mean" and "running_var"
            if args.sam:
                optimizer.first_step(zero_grad=True)


                disable_running_stats(model)

                predictions_max = []
                # import pdb; pdb.set_trace()

                kl_max = 0.0
                for j in range(args.n_model):
                    _predictions_b1, _kl_b1 = model(inputs, reuse_eps=True)
                    kl_max = kl_max + _kl_b1
                    predictions_max.append(F.log_softmax(_predictions_b1, dim=1))
                kl_max = kl_max/args.n_model
                predictions_max = torch.stack(predictions_max, dim=-1)
                log_outputs2 = logmeanexp(predictions_max, dim=2)
                # beta = metric.get_beta(iter - 1, len(dataset.train), 1/bz_instance, epoch, args.epochs)
                beta = metric.get_beta(iter - 1, len(dataset.train), args.beta_type, epoch, args.epochs)
                # import pdb; pdb.set_trace()
                loss_c2 = elbo_loss(log_outputs2, targets, kl_max, beta)
                loss_c2.backward()  # update "running_mean" and "running_var"
                optimizer.second_step(zero_grad=True)

                sharpness = loss_c2 - loss_c1
            else:
                optimizer.step()
                optimizer.zero_grad()
                sharpness = loss_c1 - loss_c1
            with torch.no_grad():
                correct = (log_outputs1.data.argmax(axis=1) == targets).sum()/len(log_outputs1)
                if args.lr_schedule in ['step']:
                    lr = scheduler.lr()
                elif args.lr_schedule in ['plateau']:
                    lr = optimizer.param_groups[0]['lr']
                else:
                    lr = scheduler.get_lr()[0]
                log(model, loss_c1.view(1, -1), correct.view(1, -1), lr, sharpness=sharpness.view(1, -1))

            iter += 1

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            n_sample = 1
            if epoch == args.epochs - 1 or epoch % 10 == 0:
                n_sample = 30
                # print("Test with {} samples", n_sample)
            loss_test = []
            prediction_test = []
            target_test = []
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)
                target_test.append(targets)
                bz_instance = inputs.size(0)
                predictions = []
                predictions_ens = []
                kl = 0.0
                for j in range(n_sample):
                    _predictions, _kl = model(inputs)
                    kl = kl + _kl
                    predictions.append(F.log_softmax(_predictions, dim=1))
                    predictions_ens.append(F.softmax(_predictions, dim=-1))
                kl = kl / n_sample
                predictions = torch.stack(predictions, dim=-1)
                predictions_ens = torch.stack(predictions_ens, dim=-1)
                prediction_test.append(predictions_ens.mean(-1))
                log_outputs = logmeanexp(predictions, dim=2)
                # beta = metric.get_beta(iter, len(dataset.test), 1/bz_instance, epoch, args.epochs)
                beta = metric.get_beta(iter, len(dataset.test), args.beta_type, epoch, args.epochs)
                loss = elbo_loss(log_outputs, targets, kl, beta)
                loss_test.append(loss.item())
                acc = (log_outputs.argmax(axis=1) == targets).sum() / len(log_outputs)
                log(model, loss.view(1, -1), acc.view(1, -1))

        if n_sample > 1:
            pre_ens = torch.cat(prediction_test).cpu().numpy()
            target_ens = torch.cat(target_test).cpu().numpy()
            acc_test = np.mean(np.argmax(pre_ens, axis=1) == target_ens) * 100
            _, eval_nll = nll(pre_ens, target_ens)
            eval_ece, acc_bin, conf_bin, Bm_bin = ece_score_np(pre_ens, target_ens, n_bins=20, return_stat=True)
            print("\nACC", acc_test)
            print("NLL", eval_nll)
            print("ECE", eval_ece)
            print('--------- ECE stat ----------', epoch)
            print("ACC", acc_bin)
            print("Conf", conf_bin)
            print("Bm", Bm_bin)

        if args.lr_schedule in ['cosine']:
            scheduler.step()
        elif args.lr_schedule in ['plateau']:
            scheduler.step(sum(loss_test))
        else:
            scheduler(epoch)

        if (epoch + 1) % 10 == 0 and args.log_dir:
            os.makedirs(args.log_dir, exist_ok=True)
            save_checkpoint(args.log_dir, epoch=epoch,
                            name="checkpoint_{}".format(args.random_seed),
                            replace=True,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(),)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--adaptive", action="store_true", help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    # parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--depth", default=28, type=int, help="Number of layers.")
    # parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--width_factor", default=10, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")

    parser.add_argument("--dataset_path", default="/home/ubuntu/vit_selfOT/ViT-pytorch/data", type=str, help="link to dataset")
    parser.add_argument("--gsam_alpha", type=float, default=0, help="alpha when apply gsam")
    parser.add_argument("--mode", type=float, default=1., help="change grad mode: 1: grad=max-min+current, 2: grad=max-min")

    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset to load: cifar10 and cifar100")
    parser.add_argument("--model_name", type=str, default="WRN", help="model_name to load")
    parser.add_argument("--lr_schedule", type=str, default="plateau", help="lr_schedule: plateau, step, cosine")
    # parser.add_argument("--lr_schedule", type=str, default="cosine", help="lr_schedule: step, cosine")

    parser.add_argument("--dis_BN_min", action="store_true", help="disable update statistic of BatchNorm")
    parser.add_argument("--dis_BN_max", action="store_true", help="disable update statistic of BatchNorm")
    parser.add_argument("--is_pretrained", action="store_true", help="loading pretrained weight")
    # parser.add_argument("--rho_lst", action="append", default=[], help="rho0*perturb + rgo1*old_g")
    parser.add_argument("--rho_lst", type=str, default="0.05_0.05", help="0.05_0.05")
    parser.add_argument("--data_split", type=str, default="0.5_0.5", help="0.5_0.5")

    parser.add_argument("--sam", action="store_true")
    parser.add_argument("--adam", action="store_true", help="Update model by Adam")
    parser.add_argument("--sam2branch", action="store_true", help="Update model by sam2branch")
    parser.add_argument("--n_branch", type=int, default=1, help="Update model by SAMnBranch")
    parser.add_argument("--otmrd", action="store_true", help="Update model by sam2branch with chain")
    parser.add_argument("--otmrd_1step", action="store_true", help="Update model by sam2branch with chain")
    parser.add_argument("--cutmix", action="store_true", help="Using cut-mix to transform B1, B2")

    parser.add_argument("--fft", action="store_true", help="Using cut-mix to transform B1, B2")
    parser.add_argument("--fft_alpha", type=float, default=0.5, help="alpha is how much to change, Using to transform B1 = 1-alpha*B1 + alpha*B2")
    parser.add_argument("--fft_ratio", type=float, default=1, help="Using fft to transform, ratio area to apply")

    parser.add_argument("--sam_switch", action="store_true", help="using switch for SAM2branch")
    parser.add_argument("--sam_branchchain", action="store_true", help="using sam_branchchain")
    parser.add_argument("--sam_reuse", action="store_true", help="using sam_reuse")
    parser.add_argument("--merge_grad", action="store_true", help="merge gradient for sam_batch_chain")
    parser.add_argument("--noise_var", type=float, default=0, help="Noise variance")
    parser.add_argument("--split_mode", type=float, default=1, help="split data equally with mode 1")
    parser.add_argument("--img_size", type=int, default=32, help="")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_dir", type=str, default=None, help="folder to save model")
    parser.add_argument("--fine_tune", action="store_true", help="fine tunning model")

    parser.add_argument("--n_model", type=int, default=1, help="")
    parser.add_argument("--beta_type", type=float, default=0.0001, help="")

    parser.add_argument("--ignore_sigma", action="store_true", help="ignore_sigma when calculate theta_a")
    parser.add_argument("--geometry", action="store_true", help="geometry when calculate theta_a")
    parser.add_argument("--p_power", action="store_true", help="geometry when calculate theta_a")
    parser.add_argument("--sgvb", action="store_true", help="Using W from gaussian")

    args = parser.parse_args()
    args.data_split = [float(it) for it in args.data_split.split("_")]
    args.rho_lst = [float(it) for it in args.rho_lst.split("_")]
    for arg_item in vars(args):
        print(arg_item, getattr(args, arg_item))

    initialize(args, seed=args.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.threads, img_size=args.img_size, root=args.dataset_path, dataset=args.dataset,
                    pre_trained=args.is_pretrained, bz_test=args.batch_size)

    log = Log(log_each=10)
    num_cls = len(dataset.classes)

    if args.model_name == 'r10':
        model = ResNet10(num_cls)
    elif args.model_name == 'r18':
        model = ResNet18(num_cls)
    elif args.model_name == 'r8':
        model = resnet8(num_classes=num_cls)
    elif args.model_name == 'r14':
        model = resnet14(num_classes=num_cls)
    elif args.model_name == 'r20':
        model = resnet20(num_classes=num_cls)
    elif args.model_name == 'r32':
        model = resnet32(num_classes=num_cls)
    elif args.model_name == 'r56':
        model = resnet56(num_classes=num_cls)
    elif args.model_name == 'preresnet164':
        model = PreResNet164_model(num_classes=num_cls)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    if args.sam or args.adam:
        train_sam(args, model, dataset)
    elif args.otmrd:
        train_sambatch_chain(args, model, dataset)


    log.flush()
