import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from conf import get_config, set_env, set_logger, set_outdir
from dataset import *
from model.ANFL import MEFARG
from utils import *


def get_dataloader(conf):
    print("==> Preparing data...")
    trainset = HybridDataset(
        conf.dataset_path,
        phase="train",
        transform=image_train(crop_size=conf.crop_size),
        stage=1,
    )
    train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
    valset = HybridDataset(
        conf.dataset_path,
        phase="val",
        transform=image_eval(crop_size=conf.crop_size),
        stage=1,
    )
    val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)
    return train_loader, val_loader, len(trainset), len(valset)


# Train
def train(conf, net, train_loader, optimizer, epoch, criterion):
    losses = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(
            optimizer,
            epoch,
            conf.epochs,
            conf.learning_rate,
            batch_idx,
            train_loader_len,
        )
        targets = targets.float()
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))
    return losses.avg


# Val
def val(net, val_loader, criterion):
    losses = AverageMeter()
    net.eval()
    statistics_list = None
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            targets = targets.float()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.data.item(), inputs.size(0))
            update_list = statistics(outputs, targets.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return losses.avg, mean_f1_score, f1_score_list, mean_acc, acc_list


def main(conf):
    dataset_info = hybrid_infolist

    start_epoch = 0
    best_val_mean_f1_score = 0
    # data
    train_loader, val_loader, train_data_num, val_data_num = get_dataloader(conf)
    train_weight = torch.from_numpy(
        np.loadtxt(os.path.join(conf.dataset_path, "list", conf.dataset + "_train_weight.txt"))
    )
    logging.info("[ val_data_num: {} ]".format(val_data_num))

    net = MEFARG(
        num_main_classes=conf.num_main_classes,
        num_sub_classes=conf.num_sub_classes,
        backbone=conf.arc,
        neighbor_num=conf.neighbor_num,
        metric=conf.metric,
    )
    # resume
    if conf.resume != "":
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        train_weight = train_weight.cuda()

    criterion = WeightedAsymmetricLoss(weight=train_weight)

    optimizer = optim.AdamW(
        net.parameters(),
        betas=(0.9, 0.999),
        lr=conf.learning_rate,
        weight_decay=conf.weight_decay,
    )
    print("the init learning rate is ", conf.learning_rate)

    # train and val
    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]["lr"]
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        train_loss = train(conf, net, train_loader, optimizer, epoch, criterion)
        val_loss, val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val(net, val_loader, criterion)

        # log
        infostr = {
            "Epoch:  {}   train_loss: {:.5f}  val_loss: {:.5f}  val_mean_f1_score {:.2f}, val_mean_acc {:.2f}".format(
                epoch + 1,
                train_loss,
                val_loss,
                100.0 * val_mean_f1_score,
                100.0 * val_mean_acc,
            )
        }

        logging.info(infostr)
        infostr = {"F1-score-list:"}
        logging.info(infostr)
        infostr = dataset_info(val_f1_score)
        logging.info(infostr)
        infostr = {"Acc-list:"}
        logging.info(infostr)
        infostr = dataset_info(val_acc)
        logging.info(infostr)

        # save checkpoints
        if best_val_mean_f1_score < val_mean_f1_score:
            best_val_mean_f1_score = val_mean_f1_score
            checkpoint = {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(conf["outdir"], "best_model.pth"))

        checkpoint = {
            "epoch": epoch,
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(conf["outdir"], "cur_model.pth"))


# ---------------------------------------------------------------------------------


if __name__ == "__main__":
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)
