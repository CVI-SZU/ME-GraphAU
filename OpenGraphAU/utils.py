import subprocess
from math import cos, pi

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision import transforms


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def statistics(pred, y, thresh):
    batch_size = pred.size(0)
    class_nb = pred.size(1)
    pred = pred >= thresh
    pred = pred.long()
    statistics_list = []
    for j in range(class_nb):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(batch_size):
            if pred[i][j] == 1:
                if y[i][j] == 1:
                    TP += 1
                elif y[i][j] == 0:
                    FP += 1
                else:
                    continue
            elif pred[i][j] == 0:
                if y[i][j] == 1:
                    FN += 1
                elif y[i][j] == 0:
                    TN += 1
                else:
                    continue
            else:
                assert False
        statistics_list.append({"TP": TP, "FP": FP, "TN": TN, "FN": FN})
    return statistics_list


def calc_f1_score(statistics_list):
    f1_score_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]["TP"]
        FP = statistics_list[i]["FP"]
        FN = statistics_list[i]["FN"]

        precise = TP / (TP + FP + 1e-20)
        recall = TP / (TP + FN + 1e-20)
        f1_score = 2 * precise * recall / (precise + recall + 1e-20)
        f1_score_list.append(f1_score)
    mean_f1_score = sum(f1_score_list) / len(f1_score_list)

    return mean_f1_score, f1_score_list


def draw_text(path, words, probs):
    import cv2

    AU_names = [
        "Inner brow raiser",
        "Outer brow raiser",
        "Brow lowerer",
        "Upper lid raiser",
        "Cheek raiser",
        "Lid tightener",
        "Nose wrinkler",
        "Upper lip raiser",
        "Nasolabial deepener",
        "Lip corner puller",
        "Sharp lip puller",
        "Dimpler",
        "Lip corner depressor",
        "Lower lip depressor",
        "Chin raiser",
        "Lip pucker",
        "Tongue show",
        "Lip stretcher",
        "Lip funneler",
        "Lip tightener",
        "Lip pressor",
        "Lips part",
        "Jaw drop",
        "Mouth stretch",
        "Lip bite",
        "Nostril dilator",
        "Nostril compressor",
        "Left Inner brow raiser",
        "Right Inner brow raiser",
        "Left Outer brow raiser",
        "Right Outer brow raiser",
        "Left Brow lowerer",
        "Right Brow lowerer",
        "Left Cheek raiser",
        "Right Cheek raiser",
        "Left Upper lip raiser",
        "Right Upper lip raiser",
        "Left Nasolabial deepener",
        "Right Nasolabial deepener",
        "Left Dimpler",
        "Right Dimpler",
    ]
    AU_ids = [
        "1",
        "2",
        "4",
        "5",
        "6",
        "7",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "32",
        "38",
        "39",
        "L1",
        "R1",
        "L2",
        "R2",
        "L4",
        "R4",
        "L6",
        "R6",
        "L10",
        "R10",
        "L12",
        "R12",
        "L14",
        "R14",
    ]
    # from PIL import Image, ImageDraw, ImageFont
    img = cv2.imread(path)
    pos_y = img.shape[0] // 40
    pos_x = img.shape[1] + img.shape[1] // 100
    pos_x_ = img.shape[1] * 3 // 2 - img.shape[1] // 100

    img = cv2.copyMakeBorder(img, 0, 0, 0, img.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # num_aus = len(words)
    # for i, item in enumerate(words):
    #     y = pos_y + (i * img.shape[0] // 17 )
    #     img = cv2.putText(img, str(item), (pos_x, y), cv2.FONT_HERSHEY_SIMPLEX, round(img.shape[1] / 2048, 3), (0,0,255), 2)
    # pos_y = pos_y + (num_aus * img.shape[0] // 17 )
    for i, item in enumerate(range(21)):
        y = pos_y + (i * img.shape[0] // 22)
        color = (0, 0, 0)
        if float(probs[item]) > 0.5:
            color = (0, 0, 255)
        img = cv2.putText(
            img,
            AU_names[i] + " -- AU" + AU_ids[i] + ": {:.2f}".format(probs[i]),
            (pos_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            round(img.shape[1] / 2800, 3),
            color,
            2,
        )

    for i, item in enumerate(range(21, 41)):
        y = pos_y + (i * img.shape[0] // 22)
        color = (0, 0, 0)
        if float(probs[item]) > 0.5:
            color = (0, 0, 255)
        img = cv2.putText(
            img,
            AU_names[item] + " -- AU" + AU_ids[item] + ": {:.2f}".format(probs[item]),
            (pos_x_, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            round(img.shape[1] / 2800, 3),
            color,
            2,
        )
    return img


def calc_acc(statistics_list):
    acc_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]["TP"]
        FP = statistics_list[i]["FP"]
        FN = statistics_list[i]["FN"]
        TN = statistics_list[i]["TN"]

        acc = (TP + TN) / (TP + TN + FP + FN + 1e-20)
        acc_list.append(acc)
    mean_acc_score = sum(acc_list) / len(acc_list)

    return mean_acc_score, acc_list


def update_statistics_list(old_list, new_list):
    if not old_list:
        return new_list

    assert len(old_list) == len(new_list)

    for i in range(len(old_list)):
        old_list[i]["TP"] += new_list[i]["TP"]
        old_list[i]["FP"] += new_list[i]["FP"]
        old_list[i]["TN"] += new_list[i]["TN"]
        old_list[i]["FN"] += new_list[i]["FN"]

    return old_list


def BP4D_infolist(list):
    infostr = {
        "AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU6: {:.2f} AU7: {:.2f} AU10: {:.2f} AU12: {:.2f} AU14: {:.2f} AU15: {:.2f} AU17: {:.2f} AU23: {:.2f} AU24: {:.2f} ".format(
            100.0 * list[0],
            100.0 * list[1],
            100.0 * list[2],
            100.0 * list[3],
            100.0 * list[4],
            100.0 * list[5],
            100.0 * list[6],
            100.0 * list[7],
            100.0 * list[8],
            100.0 * list[9],
            100.0 * list[10],
            100.0 * list[11],
        )
    }
    return infostr


def DISFA_infolist(list):
    # infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f}  AU6: {:.2f} AU9: {:.2f} AU12: {:.2f}  AU25: {:.2f} AU26: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7])}
    infostr = {
        "AU1: {:.2f} AU2: {:.2f} AU4: {:.2f}  AU6: {:.2f} AU9: {:.2f} AU12: {:.2f}  AU25: {:.2f} AU26: {:.2f} ".format(
            *[100.0 * x for x in list]
        )
    }

    return infostr


def hybrid_infolist(list):
    infostr = {
        "AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU5: {:.2f} AU6: {:.2f} AU7: {:.2f} AU9: {:.2f} AU10: {:.2f} AU11: {:.2f} \
     AU12: {:.2f} AU13: {:.2f} AU14: {:.2f} AU15: {:.2f} AU16: {:.2f} AU17: {:.2f} AU18: {:.2f} AU19: {:.2f} AU20: {:.2f} \
     AU22: {:.2f} AU23: {:.2f} AU24: {:.2f} AU25: {:.2f} AU26: {:.2f} AU27: {:.2f} AU32: {:.2f} AU38: {:.2f} AU39: {:.2f}\
      AUL1: {:.2f} AUR1: {:.2f} AUL2: {:.2f} AUR2: {:.2f} AUL4: {:.2f} AUR4: {:.2f} AUL6: {:.2f} AUR6: {:.2f} AUL10: {:.2f} \
      AUR10: {:.2f} AUL12: {:.2f} AUR12: {:.2f} AUL14: {:.2f} AUR14: {:.2f}".format(
            *[100.0 * x for x in list]
        )
    }
    return infostr


def hybrid_prediction_infolist(pred, thresh):
    infostr_pred_probs = {
        "AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU5: {:.2f} AU6: {:.2f} AU7: {:.2f} AU9: {:.2f} AU10: {:.2f} AU11: {:.2f} \
     AU12: {:.2f} AU13: {:.2f} AU14: {:.2f} AU15: {:.2f} AU16: {:.2f} AU17: {:.2f} AU18: {:.2f} AU19: {:.2f} AU20: {:.2f} \
     AU22: {:.2f} AU23: {:.2f} AU24: {:.2f} AU25: {:.2f} AU26: {:.2f} AU27: {:.2f} AU32: {:.2f} AU38: {:.2f} AU39: {:.2f}\
      AUL1: {:.2f} AUR1: {:.2f} AUL2: {:.2f} AUR2: {:.2f} AUL4: {:.2f} AUR4: {:.2f} AUL6: {:.2f} AUR6: {:.2f} AUL10: {:.2f} \
      AUR10: {:.2f} AUL12: {:.2f} AUR12: {:.2f} AUL14: {:.2f} AUR14: {:.2f}".format(
            *[100.0 * x for x in pred]
        )
    }

    AU_name_lists = [
        "Inner brow raiser",
        "Outer brow raiser",
        "Brow lowerer",
        "Upper lid raiser",
        "Cheek raiser",
        "Lid tightener",
        "Nose wrinkler",
        "Upper lip raiser",
        "Nasolabial deepener",
        "Lip corner puller",
        "Sharp lip puller",
        "Dimpler",
        "Lip corner depressor",
        "Lower lip depressor",
        "Chin raiser",
        "Lip pucker",
        "Tongue show",
        "Lip stretcher",
        "Lip funneler",
        "Lip tightener",
        "Lip pressor",
        "Lips part",
        "Jaw drop",
        "Mouth stretch",
        "Lip bite",
        "Nostril dilator",
        "Nostril compressor",
        "Left Inner brow raiser",
        "Right Inner brow raiser",
        "Left Outer brow raiser",
        "Right Outer brow raiser",
        "Left Brow lowerer",
        "Right Brow lowerer",
        "Left Cheek raiser",
        "Right Cheek raiser",
        "Left Upper lip raiser",
        "Right Upper lip raiser",
        "Left Nasolabial deepener",
        "Right Nasolabial deepener",
        "Left Dimpler",
        "Right Dimpler",
    ]
    AU_indexs = np.where(pred >= thresh)[0]
    AU_prediction = [(AU_name_lists[i], pred[i]) for i in AU_indexs]
    infostr_au_pred = {*AU_prediction}
    return infostr_pred_probs, infostr_au_pred


def adjust_learning_rate(optimizer, epoch, epochs, init_lr, iteration, num_iter):
    current_iter = iteration + epoch * num_iter
    max_iter = epochs * num_iter
    lr = init_lr * (1 + cos(pi * current_iter / max_iter)) / 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class SetFlip(object):
    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class image_train(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.RandomCrop(self.crop_size),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                normalize,
            ]
        )
        img = transform(img)
        return img


class image_eval(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
        img = transform(img)
        return img


def load_state_dict(model, path):
    checkpoints = torch.load(path, map_location=torch.device("cpu"))
    state_dict = checkpoints["state_dict"]
    from collections import OrderedDict

    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     if "module." in k:
    #         k = k[7:]  # remove `module.`
    #     new_state_dict[k] = v
    # load params
    model.load_state_dict(state_dict, strict=False)
    return model


def download_checkpoint(link_checkpoint, folder_checkpoint="checkpoints", verbose=False, *args, **kwargs):
    print(f"Downloading checkpoint {link_checkpoint} ...")
    cmd = f"wget {link_checkpoint} -P {folder_checkpoint}"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


class WeightedAsymmetricLoss(nn.Module):
    def __init__(
        self,
        eps=1e-8,
        disable_torch_grad=True,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
    ):
        super(WeightedAsymmetricLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        self.weight = weight
        self.reduction = reduction
        self.reduce = reduce
        self.size_average = size_average

    def forward(self, x, y):
        mask = y.detach() != -1

        xs_pos = x
        xs_neg = 1 - x

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.disable_torch_grad:
            torch.set_grad_enabled(False)
        neg_weight = 1 - xs_neg
        if self.disable_torch_grad:
            torch.set_grad_enabled(True)
        loss = los_pos + neg_weight * los_neg

        if self.weight is not None:
            loss = loss * self.weight.view(1, -1)

        loss = -loss[mask]

        if self.reduction == "mean":
            loss = loss.mean(dim=-1)
        else:
            loss = loss.sum(dim=-1)

        if self.size_average is not None or self.reduce is not None:
            if self.reduce is False:
                return loss
            else:
                if self.size_average is True or self.size_average is None:
                    return loss.mean()
                else:
                    return loss.sum()

        else:
            return loss.mean()


if __name__ == "__main__":
    datainfo = DISFA_infolist
    f1 = [0.5, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2]
    print(DISFA_infolist(f1))
