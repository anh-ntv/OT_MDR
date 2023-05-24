import os
import argparse
import torch
import torch.nn as nn
import numpy as np

# from model.wide_res_net import WideResNet
# from model.vit import VisionTransformer, CONFIGS
# from model.smooth_cross_entropy import smooth_crossentropy
# from model.pyramidNet import PyramidNet
# from model.resnet import ResNet10, ResNet18
# from efficientnet_pytorch import EfficientNet

from model.utils_arch import get_ensemble, EnsembleWrap

from data.cifar import Cifar
# from utility.log import Log
from utility.initialize import initialize
# from utility.step_lr import StepLR
# from utility.bypass_bn import enable_running_stats, disable_running_stats
# from torchvision import transforms
#
# from torch.optim.lr_scheduler import CosineAnnealingLR

import sys; sys.path.append("..")
# from sam_sam import SAM, SAM_Maxmin, SAM_batch, SAM2branch, SAMnBranch, SAM_batch_chain, SAM2branch_chain, SAM_gsam, SAMnChain, SAMnChain_KL
#
#
# from utility.uncertainty_metrics_my import nll, brier_score, expected_calibration_error, static_calibration_error
# from utility.uncertainty_metrics import metrics_kfold
# from utility.diversity_measure import DiversityMeasure

import torch.nn.functional as F
from utility.uncertainty_metrics import ece_kfold

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

def load_checkpoint(model, root, name="checkpoint_42", epoch=199):
    # checkpoint_42_4-199.pt
    for im, _model in enumerate(model.models):
        _model_dir = os.path.join(root, "{}_{}-{}.pt".format(name, im, epoch))
        # name = "checkpoint"
        # _model_dir = os.path.join(root, "{}_{}.pth".format(name, im))
        state = torch.load(_model_dir)
        _model.load_state_dict(state['state_dict'])
        print('Loading successful: ' + _model_dir)

def estimate_entropy(predictions, islogits=True):
    if islogits:
        predictions = F.softmax(predictions, dim=-1)
    ent = torch.sum(-predictions * torch.log(predictions), dim=-1)
    return ent


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

def train_sambatch_chain(args, enmodel, dataset):

    print("===\nTrain with train_sambatch_chain\n===")
    output_islogit=False
    load_checkpoint(enmodel, args.log_dir)
    # print('Loading successful: ' + model_path)
    # DM = DiversityMeasure()
    enmodel.eval()
    if isinstance(enmodel, EnsembleWrap):
        for submodel in enmodel.models:
            submodel.eval()
    # log.eval(len_dataset=len(dataset.test))

    # diverse_score = {}

    # def update_score(name, value):
    #     if not diverse_score.__contains__(name):
    #         diverse_score[name] = []
    #     diverse_score[name].append(value)

    with torch.no_grad():
        prediction_test = []
        target_test = []
        for batch in dataset.test:
            inputs, targets = (b.to(device) for b in batch)

            predictions, all_preds = enmodel(inputs, return_probs=True)
            # loss_c = smooth_crossentropy(predictions, targets, islogits=output_islogit)
            # correct = torch.argmax(predictions, 1) == targets
            # print(sum(correct)/len(correct))
            # # entropy = estimate_entropy(predictions, islogits=output_islogit)
            prediction_test.append(predictions)
            target_test.append(targets)
            # dm_score = DM.standard(all_preds, targets)
            # for m in dm_score.keys():
            #     update_score('diversity/{}'.format(m), dm_score[m].item())

            # u_nll = nll(targets, predictions)
            # u_bs = brier_score(targets, predictions)
            # u_ece = expected_calibration_error(targets, predictions)
            # u_sce = static_calibration_error(targets, predictions)

            # update_score('accuracy', correct.float().mean().item())
            # update_score('loss', loss_c.mean().item())
            # update_score('u_nll', u_nll.item())
            # update_score('u_bs', u_bs.item())
            # update_score('u_ece', u_ece.item())
            # update_score('u_sce', u_sce.item())

            # Compute Calibrated Uncertainty Metrics
            # unlike in diversity measure package, we use the aggregated prediction to estimate
        #     targets_np = targets.cpu().numpy().astype(int)
        #     pred_np = predictions.cpu().numpy()
        #     metrics_ts = metrics_kfold(np.log(pred_np), targets_np, temp_scale=True)
        #     for key in metrics_ts.keys():
        #         update_score('cal_' + key, metrics_ts[key])
        #
        #     for im, preds in enumerate(all_preds):
        #         single_correct = torch.argmax(preds, 1) == targets
        #         update_score('accuracy_model_{}'.format(im), single_correct.float().mean().item())
        #
        #     # log(model, loss_c.cpu(), correct.cpu())
        # for k in diverse_score:
        #     print(k, sum(diverse_score[k])/len(diverse_score[k]))
        prediction_test = torch.cat(prediction_test)
        target_test = torch.cat(target_test)
        acc = torch.argmax(prediction_test, 1) == target_test
        # eval_ece, acc_bin, conf_bin, Bm_bin = ece_score_np(prediction_test.cpu().numpy(), target_test.cpu().numpy(), n_bins=20, return_stat=True)
        # eval_ece, acc_bin = ece_score_np(np.log(prediction_test.cpu().numpy()), target_test.cpu().numpy(), temp_scale=True)
        # eval_ece, aver_conf = ece_kfold(np.log(prediction_test.cpu().numpy()), target_test.cpu().numpy(), temp_scale=True)
        eval_ece, aver_conf = ece_kfold(np.log(prediction_test.cpu().numpy()), target_test.cpu().numpy())
        print(args.note)
        print('--------- ECE stat ----------', len(acc))
        print("ACC", sum(acc)/len(acc))
        print("Cal-ECE", eval_ece)
        print("aver_conf", list(aver_conf))
        # print("ACC", acc_bin)
        # print("Conf", conf_bin)
        # print("Bm", Bm_bin)


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
    parser.add_argument("--sam_mode", type=float, default=1., help="change grad mode: 1: grad=max-min+current, 2: grad=max-min")

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
    parser.add_argument("--sam_chain", action="store_true", help="Update model by sam2branch with chain")
    parser.add_argument("--sam_chain_1step", action="store_true", help="Update model by sam2branch with chain")
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
    parser.add_argument("--sam_chain_kl", action="store_true", help="fine tunning model")
    parser.add_argument("--using_kl", action="store_true", help="fine tunning model")
    parser.add_argument("--true_grad", action="store_true", help="fine tunning model")
    parser.add_argument("--note", type=str, default="")

    args = parser.parse_args()
    args.data_split = [float(it) for it in args.data_split.split("_")]
    args.rho_lst = [float(it) for it in args.rho_lst.split("_")]
    for arg_item in vars(args):
        print(arg_item, getattr(args, arg_item))

    initialize(args, seed=args.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    custom_dataset = {}
    if args.fft:
        custom_dataset['type'] = "fft"
        custom_dataset['data'] = [args.fft_alpha, args.fft_ratio]
        print("Using FFT")
    # test_transform = transforms.ToTensor()
    test_transform = None
    dataset = Cifar(args.batch_size, args.threads, img_size=args.img_size, root=args.dataset_path, dataset=args.dataset,
                    pre_trained=args.is_pretrained, bz_test=args.batch_size, custom=custom_dataset,
                    test_transform=test_transform, train_transform=test_transform)

    # log = Log(log_each=10)
    num_cls = len(dataset.classes)
    ens_mode = 'average_prob'
    models = get_ensemble(arch=args.model_name, dataset=args.dataset)
    enmodel = EnsembleWrap(models, mode=ens_mode)  # set islogits=False

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        enmodel = nn.DataParallel(enmodel)
    enmodel.to(device)

    train_sambatch_chain(args, enmodel, dataset)

    # log.flush()
