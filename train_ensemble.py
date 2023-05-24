import os
import argparse
import torch
import torch.nn as nn
import numpy as np

from model.wide_res_net import WideResNet
from model.vit import VisionTransformer, CONFIGS
from model.smooth_cross_entropy import smooth_crossentropy
from model.pyramidNet import PyramidNet
from model.resnet import ResNet10, ResNet18
from efficientnet_pytorch import EfficientNet

from model.utils_arch import get_ensemble, EnsembleWrap

from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats


from torch.optim.lr_scheduler import CosineAnnealingLR

import sys; sys.path.append("..")
from OT_MDR_optimizer import SAM, SAM_batch_chain, SAMnChain


from utility.uncertainty_metrics_my import nll, brier_score, expected_calibration_error, static_calibration_error
from utility.uncertainty_metrics import metrics_kfold
from utility.diversity_measure import DiversityMeasure

import torch.nn.functional as F

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
                                rho_lst=args.rho_lst)
    output_islogit=True
    if isinstance(model, EnsembleWrap):
        output_islogit=False
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    if args.lr_schedule in ['cosine']:
        scheduler = CosineAnnealingLR(optimizer, args.epochs)
    else:
        scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    grad_cosine_epochs = []
    for epoch in range(args.epochs):
        model.train()
        if isinstance(model, EnsembleWrap):
            for submodel in models:
                submodel.train()
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

            if isinstance(model, EnsembleWrap):
                for submodel in models:
                    enable_running_stats(submodel)
            predictions_b1 = model(input_b1)
            loss_c1 = smooth_crossentropy(predictions_b1, target_b1, smoothing=args.label_smoothing, islogits=output_islogit)

            loss_c1.mean().backward()
            optimizer.first_step(zero_grad=True)

            if isinstance(model, EnsembleWrap):
                for submodel in models:
                    submodel.zero_grad()
            predictions_b2 = model(input_b2)
            loss_c2 = smooth_crossentropy(predictions_b2, target_b2, smoothing=args.label_smoothing, islogits=output_islogit)
            loss_c2.mean().backward()
            optimizer.first_step(zero_grad=True)

            if isinstance(model, EnsembleWrap):
                for submodel in models:
                    submodel.zero_grad()

            disable_running_stats(model)

            if isinstance(model, EnsembleWrap):
                for submodel in models:
                    disable_running_stats(submodel)

            # if args.mode in [1.0, 1.01, 1.02, 1.03, 1.04]:
            predictions_a = model(inputs)      # B1_B2_B
            loss_max = smooth_crossentropy(predictions_a, targets, smoothing=args.label_smoothing, islogits=output_islogit)
            loss_max.mean().backward()  # not update "running_mean" and "running_var"
            optimizer.second_step(zero_grad=True, mode=args.mode)
            sharpness = loss_max - torch.cat([loss_c1, loss_c2])
            if isinstance(model, EnsembleWrap):
                for submodel in models:
                    submodel.zero_grad()

            with torch.no_grad():
                predictions = torch.cat([predictions_b1, predictions_b2])
                targets_merge = torch.cat([target_b1, target_b2])
                # predictions = torch.cat([predictions_b1, predictions_b2])
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
        if epoch % 50 == 0 or epoch >= args.epochs - 2:
            log.eval(len_dataset=len(dataset.test))
            model.eval()
            if isinstance(model, EnsembleWrap):
                for submodel in models:
                    submodel.eval()
            # log.eval(len_dataset=len(dataset.test))
            diverse_score = {}
            DM = DiversityMeasure(device=device)
            def update_score(name, value):
                if not diverse_score.__contains__(name):
                    diverse_score[name] = []
                diverse_score[name].append(value)
            with torch.no_grad():
                for batch in dataset.test:
                    inputs, targets = (b.to(device) for b in batch)

                    predictions, all_preds = model(inputs, return_probs=True)
                    loss_c = smooth_crossentropy(predictions, targets, islogits=output_islogit)
                    correct = torch.argmax(predictions, 1) == targets
                    # print(sum(correct) / len(correct))
                    # entropy = estimate_entropy(predictions, islogits=output_islogit)

                    dm_score = DM.standard(all_preds, targets)
                    for m in dm_score.keys():
                        update_score('diversity/{}'.format(m), dm_score[m].item())

                    u_nll = nll(targets, predictions)
                    u_bs = brier_score(targets, predictions)
                    u_ece = expected_calibration_error(targets, predictions)
                    u_sce = static_calibration_error(targets, predictions)

                    update_score('accuracy', correct.float().mean().item())
                    update_score('loss', loss_c.mean().item())
                    update_score('u_nll', u_nll.item())
                    update_score('u_bs', u_bs.item())
                    update_score('u_ece', u_ece.item())
                    update_score('u_sce', u_sce.item())

                    # Compute Calibrated Uncertainty Metrics
                    # unlike in diversity measure package, we use the aggregated prediction to estimate
                    targets_np = targets.cpu().numpy().astype(int)
                    pred_np = predictions.cpu().numpy()
                    metrics_ts = metrics_kfold(np.log(pred_np), targets_np, temp_scale=True)
                    for key in metrics_ts.keys():
                        update_score('cal_' + key, metrics_ts[key])

                    for im, preds in enumerate(all_preds):
                        single_correct = torch.argmax(preds, 1) == targets
                        update_score('accuracy_model_{}'.format(im), single_correct.float().mean().item())

                    log(model, loss_c.cpu(), correct.cpu())
                for k in diverse_score:
                    print(k, sum(diverse_score[k]) / len(diverse_score[k]))

        if (epoch + 1) % 50 == 0 and args.log_dir:
            for im, submodel in enumerate(models):
                os.makedirs(args.log_dir, exist_ok=True)
                save_checkpoint(args.log_dir, epoch=epoch,
                            name="checkpoint_{}_{}".format(args.random_seed, im),
                            state_dict=submodel.state_dict())


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
    parser.add_argument("--lr_schedule", type=str, default="step", help="lr_schedule: step, cosine")
    # parser.add_argument("--lr_schedule", type=str, default="cosine", help="lr_schedule: step, cosine")

    parser.add_argument("--dis_BN_min", action="store_true", help="disable update statistic of BatchNorm")
    parser.add_argument("--dis_BN_max", action="store_true", help="disable update statistic of BatchNorm")
    parser.add_argument("--is_pretrained", action="store_true", help="loading pretrained weight")
    # parser.add_argument("--rho_lst", action="append", default=[], help="rho0*perturb + rgo1*old_g")
    parser.add_argument("--rho_lst", type=str, default="0.05_0.05", help="0.05_0.05")
    parser.add_argument("--data_split", type=str, default="0.5_0.5", help="0.5_0.5")

    parser.add_argument("--sam", action="store_true")
    parser.add_argument("--sgd", action="store_true", help="Update model by SGD")
    parser.add_argument("--sam2branch", action="store_true", help="Update model by sam2branch")
    parser.add_argument("--n_branch", type=int, default=1, help="Update model by SAMnBranch")
    parser.add_argument("--otmdr", action="store_true", help="Update model by sam2branch with chain")
    parser.add_argument("--otmdr_1step", action="store_true", help="Update model by sam2branch with chain")
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
    parser.add_argument("--otmdr_kl", action="store_true", help="fine tunning model")
    parser.add_argument("--using_kl", action="store_true", help="fine tunning model")
    parser.add_argument("--true_grad", action="store_true", help="fine tunning model")

    args = parser.parse_args()
    args.data_split = [float(it) for it in args.data_split.split("_")]
    args.rho_lst = [float(it) for it in args.rho_lst.split("_")]
    for arg_item in vars(args):
        print(arg_item, getattr(args, arg_item))

    initialize(args, seed=args.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.threads, img_size=args.img_size, root=args.dataset_path,
                    dataset=args.dataset, bz_test=1024)

    log = Log(log_each=10)
    num_cls = len(dataset.classes)
    ens_mode = 'average_prob'
    models = get_ensemble(arch=args.model_name, dataset=args.dataset)
    model = EnsembleWrap(models, mode=ens_mode)  # set islogits=False

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    train_sambatch_chain(args, model, dataset)

    log.flush()
